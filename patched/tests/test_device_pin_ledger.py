"""GPU/torch-free unit tests for the L2-bypass device-pin ledger
(mem_cache/device_pin_ledger.py) and the slot/lock balance of every path that
can end a deferred device pin.

Regression cover for the soak-test wedge:

    Prefill batch, #new-seq: 1, #new-token: 64, #cached-token: 0,
    token usage: 0.99, #running-req: 0, #queue-req: 0

— zero requests running or queued, yet the KV pool 99% full and the instance
never recovering. Under L2-bypass a write-through node's GPU KV slot is the RDMA
source for its own device->L3 PUT, so it stays `lock_ref`'d until that PUT acks:
the one lock in the design released by an *external* event. Every way that ack
can fail to arrive (a backend exception killing the backup thread, a hung PUT, a
dropped op) turns into a monotonic, unrecoverable device-memory leak.

Properties asserted here:
  1. the ledger's reap order is deterministic and oldest-first, so every TP rank
     reclaims the SAME pins (a divergent dec_lock_ref would corrupt the tree);
  2. ack and reap race safely — whichever loses is a no-op, never a double
     release and never a missed one;
  3. a pin is only ever released when no PUT can be reading its slots
     (try_start / try_cancel are mutually exclusive);
  4. audit_pins attributes locked tokens to an owner and, crucially, reports the
     ones nothing can account for — the leak signal.

Pure python; run with `python3 test_device_pin_ledger.py`.
"""
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mem_cache.device_pin_ledger import (  # noqa: E402
    DevicePinLedger,
    audit_pins,
)

PAGE = 2


class FakeKey:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def __len__(self):
        return len(self.tokens)

    def child_key(self, page_size):
        return tuple(self.tokens[:page_size])


class FakeNode:
    """Duck-typed TreeNode with just what the ledger and the audit touch."""

    _counter = [0]

    def __init__(self, tokens=(), parent=None, lock_ref=0):
        self.key = FakeKey(tokens)
        self.parent = parent
        self.children = {}
        self.lock_ref = lock_ref
        FakeNode._counter[0] += 1
        self.id = FakeNode._counter[0]
        if parent is not None:
            parent.children[self.key.child_key(PAGE)] = self


def chain(root, *lengths, lock_ref=0):
    """Build a parent-first chain under `root`; returns the list of nodes."""
    nodes = []
    parent = root
    base = 0
    for n in lengths:
        node = FakeNode(range(base, base + n), parent=parent, lock_ref=lock_ref)
        nodes.append(node)
        parent = node
        base += n
    return nodes


# --------------------------------------------------------------------------
# 1. Ledger bookkeeping and the TP-symmetric reap order
# --------------------------------------------------------------------------


class TestLedgerBookkeeping(unittest.TestCase):
    def setUp(self):
        self.root = FakeNode()
        self.ledger = DevicePinLedger()

    def test_add_and_pop_round_trip(self):
        node = FakeNode((0, 1), parent=self.root)
        self.ledger.add(7, node, tokens=2, now=100.0)
        self.assertEqual(len(self.ledger), 1)
        self.assertIn(7, self.ledger)

        rec = self.ledger.pop(7)
        self.assertIsNotNone(rec)
        self.assertIs(rec.node, node)
        self.assertEqual(rec.tokens, 2)
        self.assertEqual(len(self.ledger), 0)

    def test_second_pop_is_a_no_op(self):
        """The ack path and the reaper race by design; the loser must not release
        a second time (that would dec_lock_ref a node nobody is holding)."""
        node = FakeNode((0, 1), parent=self.root)
        self.ledger.add(7, node, tokens=2, now=100.0)
        self.assertIsNotNone(self.ledger.pop(7))
        self.assertIsNone(self.ledger.pop(7))

    def test_census_counts_tokens_and_oldest_age(self):
        a, b = chain(self.root, 2, 4)
        self.ledger.add(1, a, tokens=2, now=100.0)
        self.ledger.add(2, b, tokens=4, now=140.0)

        census = self.ledger.census(now=200.0)
        self.assertEqual(census.ops, 2)
        self.assertEqual(census.tokens, 6)
        self.assertEqual(census.oldest_age, 100.0)

    def test_empty_census_is_all_zero(self):
        census = self.ledger.census(now=200.0)
        self.assertEqual((census.ops, census.tokens, census.oldest_age), (0, 0, 0.0))

    def test_reapable_is_oldest_op_first(self):
        """Load-bearing for TP: each rank reaps the first K of this list, so the
        order must be identical everywhere. Insert out of order to prove the sort
        is by op id, not by insertion."""
        nodes = chain(self.root, 2, 2, 2)
        self.ledger.add(30, nodes[2], tokens=2, now=10.0)
        self.ledger.add(10, nodes[0], tokens=2, now=10.0)
        self.ledger.add(20, nodes[1], tokens=2, now=10.0)

        stale = self.ledger.reapable(now=1000.0, timeout=60.0)
        self.assertEqual([p.op_id for p in stale], [10, 20, 30])

    def test_reapable_excludes_pins_inside_the_deadline(self):
        nodes = chain(self.root, 2, 2)
        self.ledger.add(1, nodes[0], tokens=2, now=0.0)
        self.ledger.add(2, nodes[1], tokens=2, now=90.0)

        stale = self.ledger.reapable(now=100.0, timeout=60.0)
        self.assertEqual([p.op_id for p in stale], [1])

    def test_zero_timeout_disables_reaping(self):
        node = FakeNode((0, 1), parent=self.root)
        self.ledger.add(1, node, tokens=2, now=0.0)
        self.assertEqual(self.ledger.reapable(now=1e9, timeout=0.0), [])

    def test_drain_returns_everything_oldest_first_and_empties(self):
        nodes = chain(self.root, 2, 2)
        self.ledger.add(5, nodes[1], tokens=2, now=0.0)
        self.ledger.add(4, nodes[0], tokens=2, now=0.0)

        drained = self.ledger.drain()
        self.assertEqual([p.op_id for p in drained], [4, 5])
        self.assertEqual(len(self.ledger), 0)


# --------------------------------------------------------------------------
# 2. The cancel protocol: a pin is never released under an in-flight RDMA
# --------------------------------------------------------------------------


class FakeDeviceOp:
    """Mirror of cache_controller.DevicePinCancelMixin — same arbitration, no torch."""

    def __init__(self):
        self._pin_state_lock = threading.Lock()
        self.started = False
        self.cancelled = False

    def try_start(self):
        with self._pin_state_lock:
            if self.cancelled:
                return False
            self.started = True
            return True

    def try_cancel(self):
        with self._pin_state_lock:
            if self.started:
                return False
            self.cancelled = True
            return True


class TestCancelProtocol(unittest.TestCase):
    def test_cancel_before_start_wins_and_blocks_the_put(self):
        """The reaper got there first: the slots go back to the allocator, so the
        PUT must not run — RDMA-ing from a reused slot would store another
        request's KV under this page's hash."""
        op = FakeDeviceOp()
        self.assertTrue(op.try_cancel())
        self.assertFalse(op.try_start())

    def test_start_before_cancel_wins_and_holds_the_pin(self):
        op = FakeDeviceOp()
        self.assertTrue(op.try_start())
        self.assertFalse(op.try_cancel())

    def test_arbitration_is_exclusive_under_contention(self):
        """Whatever the interleaving, exactly one side may proceed: never both a
        live PUT and a freed slot."""
        for _ in range(200):
            op = FakeDeviceOp()
            outcomes = []
            barrier = threading.Barrier(2)

            def starter():
                barrier.wait()
                outcomes.append(("start", op.try_start()))

            def canceller():
                barrier.wait()
                outcomes.append(("cancel", op.try_cancel()))

            threads = [threading.Thread(target=starter), threading.Thread(target=canceller)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            won = [name for name, ok in outcomes if ok]
            self.assertEqual(
                len(won), 1, f"both sides proceeded: {outcomes}"
            )


# --------------------------------------------------------------------------
# 3. Reap simulation: lock balance across every exit path
# --------------------------------------------------------------------------


class PinHarness:
    """Minimal model of the tree side of the pin lifecycle: inc/dec lock_ref over
    a chain to root, plus the ledger and the ack/reap paths. Lets the tests assert
    the property that matters — every pin ends with the chain's lock_ref back
    where it started — without a GPU."""

    def __init__(self):
        self.root = FakeNode()
        self.ledger = DevicePinLedger()
        self.ops = {}
        self.pinned_nodes = {}
        self.reclaimed = 0
        self.stuck = 0

    def inc(self, node):
        while node is not self.root:
            node.lock_ref += 1
            node = node.parent

    def dec(self, node):
        while node is not self.root:
            node.lock_ref -= 1
            node = node.parent

    def write_backup(self, node, op_id, now):
        """_write_backup_device + _write_backup_storage_device, collapsed."""
        self.inc(node)
        op = FakeDeviceOp()
        self.ops[op_id] = op
        self.pinned_nodes[op_id] = node
        self.ledger.add(op_id, node, tokens=len(node.key), now=now)
        return op

    def ack(self, op_id):
        """_drain_backup: release only if the reaper has not already."""
        node = self.pinned_nodes.pop(op_id, None)
        released = self.ledger.pop(op_id)
        self.ops.pop(op_id, None)
        if node is not None and released is not None:
            self.dec(node)

    def reap(self, now, timeout, limit=None):
        """_reap_stale_device_pins."""
        stale = self.ledger.reapable(now, timeout)
        if limit is not None:
            stale = stale[:limit]
        for rec in stale:
            op = self.ops.get(rec.op_id)
            if op is not None and not op.try_cancel():
                self.stuck += 1
                continue
            self.ledger.pop(rec.op_id)
            self.ops.pop(rec.op_id, None)
            node = self.pinned_nodes.pop(rec.op_id, None)
            if node is None:
                continue
            self.dec(node)
            self.reclaimed += 1

    def total_lock_ref(self):
        total = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n is not self.root:
                total += n.lock_ref
        return total


class TestPinBalance(unittest.TestCase):
    def setUp(self):
        self.h = PinHarness()
        self.nodes = chain(self.h.root, 2, 2, 2)

    def test_normal_ack_returns_every_lock(self):
        self.h.write_backup(self.nodes[2], op_id=1, now=0.0)
        self.assertEqual(self.h.total_lock_ref(), 3)  # chain of 3 pinned to root
        self.h.ack(1)
        self.assertEqual(self.h.total_lock_ref(), 0)
        self.assertEqual(len(self.h.ledger), 0)

    def test_lost_ack_is_reclaimed_by_the_deadline(self):
        """The wedge: the PUT never acks. Before the reaper this pin was held for
        the life of the process."""
        self.h.write_backup(self.nodes[2], op_id=1, now=0.0)
        self.h.reap(now=1000.0, timeout=120.0)
        self.assertEqual(self.h.total_lock_ref(), 0)
        self.assertEqual(self.h.reclaimed, 1)

    def test_late_ack_after_a_reap_does_not_double_release(self):
        self.h.write_backup(self.nodes[2], op_id=1, now=0.0)
        self.h.reap(now=1000.0, timeout=120.0)
        self.h.ack(1)  # backup thread acks the op it was told to skip
        self.assertEqual(self.h.total_lock_ref(), 0)

    def test_reap_after_an_ack_does_not_double_release(self):
        self.h.write_backup(self.nodes[2], op_id=1, now=0.0)
        self.h.ack(1)
        self.h.reap(now=1000.0, timeout=120.0)
        self.assertEqual(self.h.total_lock_ref(), 0)
        self.assertEqual(self.h.reclaimed, 0)

    def test_in_flight_put_keeps_its_pin(self):
        """try_cancel loses to a PUT already reading the slots: the pin must stay,
        or the NIC writes another request's KV under this hash."""
        op = self.h.write_backup(self.nodes[2], op_id=1, now=0.0)
        op.try_start()
        self.h.reap(now=1000.0, timeout=120.0)
        self.assertEqual(self.h.stuck, 1)
        self.assertEqual(self.h.reclaimed, 0)
        self.assertEqual(self.h.total_lock_ref(), 3)
        # ...and it is still released normally once the PUT finally acks.
        self.h.ack(1)
        self.assertEqual(self.h.total_lock_ref(), 0)

    def test_one_stuck_op_does_not_block_the_rest_of_the_backlog(self):
        ops = [
            self.h.write_backup(self.nodes[i], op_id=i + 1, now=0.0) for i in range(3)
        ]
        ops[0].try_start()  # oldest is wedged in the backend
        self.h.reap(now=1000.0, timeout=120.0)
        self.assertEqual(self.h.stuck, 1)
        self.assertEqual(self.h.reclaimed, 2)

    def test_reap_limit_takes_the_oldest_prefix(self):
        """The cross-rank MIN caps how many are reaped; the prefix must be the
        oldest ops so all ranks pick the same ones."""
        for i in range(3):
            self.h.write_backup(self.nodes[i], op_id=i + 1, now=0.0)
        self.h.reap(now=1000.0, timeout=120.0, limit=2)
        self.assertEqual(self.h.reclaimed, 2)
        self.assertEqual(sorted(self.h.ledger._pins), [3])

    def test_mixed_sequence_never_leaves_a_stranded_lock(self):
        """Ack some, reap some, ack late, reap again — the chain always ends
        unlocked and the ledger empty."""
        for i, node in enumerate(self.nodes):
            self.h.write_backup(node, op_id=i + 1, now=0.0)
        self.h.ack(2)
        self.h.reap(now=1000.0, timeout=120.0)
        self.h.ack(1)
        self.h.ack(3)
        self.h.reap(now=2000.0, timeout=120.0)
        self.assertEqual(self.h.total_lock_ref(), 0)
        self.assertEqual(len(self.h.ledger), 0)

    def test_thread_death_recovery_force_acks_the_whole_backlog(self):
        """Dead backup thread => nothing in flight => the whole backlog is safe to
        ack, which is what recover_dead_backup_thread does."""
        for i, node in enumerate(self.nodes):
            self.h.write_backup(node, op_id=i + 1, now=0.0)
        self.assertGreater(self.h.total_lock_ref(), 0)
        for rec in self.h.ledger.drain():
            node = self.h.pinned_nodes.pop(rec.op_id)
            self.h.dec(node)
        self.assertEqual(self.h.total_lock_ref(), 0)


# --------------------------------------------------------------------------
# 4. audit_pins: name the owner, or name the leak
# --------------------------------------------------------------------------


class TestAuditPins(unittest.TestCase):
    def setUp(self):
        self.root = FakeNode()

    def test_backup_pinned_chain_is_fully_attributed(self):
        nodes = chain(self.root, 2, 2, 2, lock_ref=1)
        audit = audit_pins(self.root, [nodes[2]], [])
        self.assertEqual(audit.locked_nodes, 3)
        self.assertEqual(audit.locked_tokens, 6)
        self.assertEqual(audit.orphan_nodes, 0)
        self.assertEqual(audit.backup_pinned_nodes, 1)
        # ancestors count as accounted, just not as the pinned node itself
        self.assertEqual(audit.backup_pinned_tokens, 2)

    def test_load_pinned_chain_is_attributed_to_the_load(self):
        nodes = chain(self.root, 2, 2, lock_ref=1)
        audit = audit_pins(self.root, [], [nodes[1]])
        self.assertEqual(audit.orphan_nodes, 0)
        self.assertEqual(audit.load_pinned_nodes, 1)

    def test_lock_with_no_owner_is_reported_as_a_leak(self):
        """The signal the soak needed: locked device tokens that no in-flight
        backup and no in-flight load can explain."""
        nodes = chain(self.root, 2, 2, lock_ref=1)
        audit = audit_pins(self.root, [], [])
        self.assertEqual(audit.locked_nodes, 2)
        self.assertEqual(audit.locked_tokens, 4)
        self.assertEqual(audit.orphan_nodes, 2)
        self.assertEqual(audit.orphan_tokens, 4)
        self.assertEqual(audit.orphan_sample, [nodes[0].id, nodes[1].id])

    def test_unlocked_nodes_are_not_counted(self):
        chain(self.root, 2, 2, lock_ref=0)
        audit = audit_pins(self.root, [], [])
        self.assertEqual(audit.locked_nodes, 0)
        self.assertEqual(audit.orphan_nodes, 0)

    def test_mixed_tree_separates_owned_from_orphaned(self):
        owned = chain(self.root, 2, 2, lock_ref=1)
        # a separate locked branch nothing owns
        orphan = FakeNode((100, 101), parent=self.root, lock_ref=1)
        audit = audit_pins(self.root, [owned[1]], [])
        self.assertEqual(audit.locked_nodes, 3)
        self.assertEqual(audit.orphan_nodes, 1)
        self.assertEqual(audit.orphan_tokens, 2)
        self.assertEqual(audit.orphan_sample, [orphan.id])

    def test_sample_is_capped(self):
        parent = self.root
        for i in range(20):
            parent = FakeNode(range(i * 2, i * 2 + 2), parent=parent, lock_ref=1)
        audit = audit_pins(self.root, [], [], sample_limit=3)
        self.assertEqual(audit.orphan_nodes, 20)
        self.assertEqual(len(audit.orphan_sample), 3)

    def test_root_itself_is_never_counted(self):
        self.root.lock_ref = 5
        audit = audit_pins(self.root, [], [])
        self.assertEqual(audit.locked_nodes, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
