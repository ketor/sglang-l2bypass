"""A3 — deterministic demonstration that the pre-fix concurrent promotion was a
REAL defect, not a speculative one.

Bug 2 was recorded in the 5th-phase report as "code reasoning, never observed in a
log" because the crash-era container was rebuilt and its logs were lost. That is an
honest but unsatisfying place to leave a bug. This suite closes the mechanism half
of the question: it runs the ACTUAL pre-fix publish loop (transcribed verbatim from
`git show e1beee3^:patched/mem_cache/hiradix_cache.py`, `_promote_l3_async_load`,
lines 2108-2149) against a modeled radix/ack ledger and shows it produces three
distinct failures, then shows the shipped code produces none of them.

WHAT THIS PROVES: the pre-fix code path is defective, and both fixes close it.
WHAT THIS DOES NOT PROVE: that it actually fired on the box during the crash
window. The logs for that window are gone; no test can recover them. The claim
stays "mechanism demonstrated, occurrence unconfirmed".

Three independent failures from ONE precondition (two in-flight loads whose chains
share a node — i.e. exactly what increment 8 now makes impossible):

  F1  orphaned GPU slots   — `n.value = ...` overwrites the slots the tree (and the
                             first request) is already using; nothing frees them
  F2  KeyError in loading_check — both requests register `ongoing_load_back[id]`
                             for the SAME node id, so the dict holds ONE entry
                             while the ack queue holds TWO; the second pop raises
  F3  stuck lock_ref       — `inc_lock_ref` ran twice, `dec_lock_ref` once, so the
                             node is pinned forever (unevictable) even if F2 were
                             swallowed

Run with `python3 test_bug2_concurrent_promotion.py`.
"""
import os
import sys
import unittest

PATCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PATCHED)

from mem_cache.l3_marker_state import plan_promotion  # noqa: E402


class _Node:
    def __init__(self, node_id, ntokens):
        self.id = node_id
        self.key = list(range(ntokens))  # len(n.key) == token count
        self.value = None                # device slots, None => evicted
        self.lock_ref = 0
        self.l3_present = True
        self.l3_backed = False


class _Tree:
    """Models just the bookkeeping the two promotion paths touch."""

    def __init__(self):
        self.ongoing_load_back = {}
        self.ack_queue = []          # list of ack_lists, as cache_controller holds
        self.freed_slots = []        # slots explicitly returned to the allocator
        self.live_slots = set()      # slot ids currently referenced by some node

    def inc_lock_ref(self, node):
        node.lock_ref += 1

    def dec_lock_ref(self, node):
        node.lock_ref -= 1

    def enqueue_device_load(self, slots, node_ids):
        self.ack_queue.append(list(node_ids))

    def free_device_indices(self, slots):
        self.freed_slots.extend(slots)

    # --- the stock ack drain (hiradix_cache.loading_check, unchanged by us) ---
    def loading_check(self):
        while self.ack_queue:
            ack_list = self.ack_queue.pop(0)
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)  # KeyError == F2
                self.dec_lock_ref(end_node)


def promote_prefix(tree, nodes_to_load, device_slots, verified_tokens):
    """VERBATIM pre-fix logic (e1beee3^, _promote_l3_async_load 2108-2149).

    The only edits are dropping `.clone()`/`_record_store_event` (torch/telemetry)
    and the unverified-suffix warning. The defining property is preserved exactly:
    `n.value` is assigned with NO check of whether it is already set."""
    offset = 0
    loaded_nodes = []
    for n in nodes_to_load:
        n_len = len(n.key)
        if offset + n_len <= verified_tokens:
            n.value = device_slots[offset: offset + n_len]   # <-- unconditional
            n.l3_present = False
            n.l3_backed = True
            loaded_nodes.append(n)
            offset += n_len
        else:
            break
    if offset < len(device_slots):
        tree.free_device_indices(device_slots[offset:])
    last_loaded = loaded_nodes[-1]
    tree.ongoing_load_back[last_loaded.id] = last_loaded
    tree.inc_lock_ref(last_loaded)
    tree.enqueue_device_load(device_slots[:offset], node_ids=[last_loaded.id])


def promote_fixed(tree, nodes_to_load, device_slots, verified_tokens):
    """Shipped logic: plan_promotion decides, and an empty plan registers nothing."""
    plan = plan_promotion(nodes_to_load, verified_tokens)
    offset = plan.tokens
    for n, lo, hi in plan.assign:
        n.value = device_slots[lo:hi]
        n.l3_present = False
        n.l3_backed = True
    if offset < len(device_slots):
        tree.free_device_indices(device_slots[offset:])
    if not plan.assign:
        return                                    # no fence, no ack, no lock
    last_loaded = plan.assign[-1][0]
    tree.ongoing_load_back[last_loaded.id] = last_loaded
    tree.inc_lock_ref(last_loaded)
    tree.enqueue_device_load(device_slots[:offset], node_ids=[last_loaded.id])


def _two_requests_on_one_chain():
    """The precondition: one shared chain, two loads in flight, each with its own
    GPU slots. Request 1's GET lands first, then request 2's."""
    node = _Node(node_id=42, ntokens=4)
    slots_req1 = ["A0", "A1", "A2", "A3"]
    slots_req2 = ["B0", "B1", "B2", "B3"]
    return node, slots_req1, slots_req2


class PreFixIsDefective(unittest.TestCase):
    def test_F1_second_promotion_orphans_the_first_requests_slots(self):
        tree = _Tree()
        node, s1, s2 = _two_requests_on_one_chain()
        promote_prefix(tree, [node], s1, verified_tokens=4)
        self.assertEqual(node.value, s1)
        promote_prefix(tree, [node], s2, verified_tokens=4)
        # The tree now points at request 2's slots...
        self.assertEqual(node.value, s2)
        # ...and request 1's are referenced by nobody and freed by nobody.
        self.assertNotIn("A0", tree.freed_slots)
        self.assertEqual(
            tree.freed_slots, [],
            "pre-fix path leaks the superseded request's whole slot span")

    def test_F2_double_ack_raises_KeyError_in_loading_check(self):
        tree = _Tree()
        node, s1, s2 = _two_requests_on_one_chain()
        promote_prefix(tree, [node], s1, verified_tokens=4)
        promote_prefix(tree, [node], s2, verified_tokens=4)
        # Same node id assigned twice => ONE dict entry, TWO queued acks.
        self.assertEqual(len(tree.ongoing_load_back), 1)
        self.assertEqual(len(tree.ack_queue), 2)
        with self.assertRaises(KeyError):
            tree.loading_check()

    def test_F3_lock_ref_is_stuck_even_if_the_KeyError_were_swallowed(self):
        tree = _Tree()
        node, s1, s2 = _two_requests_on_one_chain()
        promote_prefix(tree, [node], s1, verified_tokens=4)
        promote_prefix(tree, [node], s2, verified_tokens=4)
        self.assertEqual(node.lock_ref, 2, "inc'd once per promotion")
        try:
            tree.loading_check()
        except KeyError:
            pass  # F2; F3 is what remains after it
        self.assertEqual(
            node.lock_ref, 1,
            "only one dec happened: the node stays pinned and unevictable forever")


class ShippedCodeIsClean(unittest.TestCase):
    def test_superseded_promotion_publishes_nothing(self):
        tree = _Tree()
        node, s1, s2 = _two_requests_on_one_chain()
        promote_fixed(tree, [node], s1, verified_tokens=4)
        self.assertEqual(node.value, s1)
        promote_fixed(tree, [node], s2, verified_tokens=4)
        # F1 closed: the tree keeps request 1's slots, request 2's are returned.
        self.assertEqual(node.value, s1)
        self.assertEqual(tree.freed_slots, s2)

    def test_no_second_ack_and_no_stuck_lock(self):
        tree = _Tree()
        node, s1, s2 = _two_requests_on_one_chain()
        promote_fixed(tree, [node], s1, verified_tokens=4)
        promote_fixed(tree, [node], s2, verified_tokens=4)
        self.assertEqual(len(tree.ack_queue), 1, "F2 closed: one ack, one entry")
        tree.loading_check()                       # must not raise
        self.assertEqual(node.lock_ref, 0, "F3 closed: lock balance returns to 0")

    def test_partial_supersede_still_publishes_the_clean_prefix(self):
        """The fix must not be a blunt 'give up when anything is superseded': the
        head of the chain that nobody published is still ours to publish."""
        tree = _Tree()
        head, tail = _Node(1, 2), _Node(2, 2)
        tail.value = ["X0", "X1"]                  # someone else published the tail
        slots = ["C0", "C1", "C2", "C3"]
        promote_fixed(tree, [head, tail], slots, verified_tokens=4)
        self.assertEqual(head.value, ["C0", "C1"], "clean head published")
        self.assertEqual(tail.value, ["X0", "X1"], "superseded tail untouched")
        self.assertEqual(tree.freed_slots, ["C2", "C3"], "our duplicate tail freed")


class Increment8RemovesThePrecondition(unittest.TestCase):
    """Belt and braces: plan_promotion makes a shared-chain collision SAFE, while
    increment 8 makes it IMPOSSIBLE — the second request never gets its own load."""

    def test_source_guarantees_one_owner_per_node(self):
        with open(os.path.join(PATCHED, "mem_cache", "hiradix_cache.py")) as f:
            src = f.read()
        # A load is only submitted when no node of its chain is already claimed.
        i = src.index("owner = self._find_inflight_owner(nodes_to_load)")
        self.assertIn("if owner is not None:", src[i:i + 200])
        self.assertIn("return False", src[i:i + 900])
        # And the claim is taken for EVERY node of the chain that does load.
        j = src.index("for n in nodes_to_load:\n            self._bypass_inflight_owner[n.id] = req_id")
        self.assertLess(j, src.index("self._bypass_load_state[req_id] = _BypassLoadState"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
