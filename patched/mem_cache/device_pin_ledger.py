from __future__ import annotations

"""L2-bypass (SGLANG_HICACHE_L2_BYPASS=1) device-pin ledger.

Torch-free bookkeeping for the ONE lock in the bypass design whose release
depends on something outside this process (same rationale as l3_marker_state /
device_page_meta: keep the logic unit-testable without a GPU).

**Why this exists.** In stock HiCache a write-through node's GPU KV slot is
pinned (``inc_lock_ref``) only until the D2H copy completes — a bounded, local,
CUDA-event-driven ack. The host copy then becomes the source of truth and the
storage backup only holds a *host* protection (``protect_host``), which cannot
wedge device memory.

L2-bypass has no host tier: the GPU slot itself is the RDMA source for the
device->L3 PUT, so the pin is deferred all the way to the **storage backup ack**
(``_drain_backup``). That makes device memory hostage to an unbounded external
path — the backend PUT, on a background thread. Any single break in that path
(thread death, a hung backend, a dropped ack) converts into a *monotonic,
unrecoverable* GPU KV leak: the pinned nodes never become evictable,
``evictable_size_`` bleeds to zero, and the scheduler wedges with zero running
requests and a 99%-full pool.

This ledger makes that class of failure bounded and visible:

* every deferred pin is recorded with its op id, node, token count and birth time;
* :meth:`DevicePinLedger.census` answers "how much device memory is hostage, and
  for how long" without walking the radix tree;
* :meth:`DevicePinLedger.reapable` lists pins past a deadline, **oldest op id
  first** — a deterministic order, which is what lets every TP rank reap the same
  prefix of the same ledger and stay symmetric.

**Safety rule encoded here (do not weaken).** A pin may only be dropped when the
backend is *not* reading the slot. Releasing a pin whose PUT is in flight lets the
slot be evicted and reused while the NIC is still reading it, which publishes some
other request's KV under this page's hash — silent wrong-KV on every later read.
So the reaper must first *cancel* the operation (see
``cache_controller.DeviceStorageOperation.try_cancel``, which loses the race to a
PUT that already started); only ops it successfully cancelled are released here.
"""

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

__all__ = [
    "PinRecord",
    "PinCensus",
    "DevicePinLedger",
    "PinAudit",
    "audit_pins",
]


class PinRecord(NamedTuple):
    """One deferred device pin: the node whose GPU slot is held unevictable until
    storage operation ``op_id`` acks."""

    op_id: int
    node: Any
    tokens: int
    created: float


class PinCensus(NamedTuple):
    """Snapshot of the outstanding deferred pins."""

    ops: int  # number of un-acked device->L3 backups
    tokens: int  # GPU KV tokens they hold unevictable
    oldest_age: float  # seconds the longest-waiting pin has been held (0 if none)


class DevicePinLedger:
    """Ordered ledger of deferred device pins, keyed by storage operation id.

    Insertion order is operation-id order (ids come from a monotonic counter), and
    every TP rank issues the same write-throughs in the same order, so iterating
    this ledger is deterministic and rank-symmetric — the property the stale-pin
    reaper relies on to release the *same* pins on every rank.
    """

    __slots__ = ("_pins",)

    def __init__(self) -> None:
        self._pins: Dict[int, PinRecord] = {}

    def __len__(self) -> int:
        return len(self._pins)

    def __contains__(self, op_id: int) -> bool:
        return op_id in self._pins

    def add(self, op_id: int, node: Any, tokens: int, now: float) -> None:
        """Record a pin taken at write-through enqueue time."""
        self._pins[op_id] = PinRecord(op_id, node, tokens, now)

    def pop(self, op_id: int) -> Optional[PinRecord]:
        """Release a pin (the backup acked, or the reaper cancelled it). Returns
        None if it was already released — the ack path and the reaper race by
        design, and whichever loses must be a no-op."""
        return self._pins.pop(op_id, None)

    def node_ids(self) -> set:
        """Ids of the nodes currently pinned by a pending backup (for the census
        that attributes locked tree nodes to an owner)."""
        return {id(p.node) for p in self._pins.values()}

    def census(self, now: float) -> PinCensus:
        if not self._pins:
            return PinCensus(0, 0, 0.0)
        tokens = 0
        oldest = now
        for p in self._pins.values():
            tokens += p.tokens
            if p.created < oldest:
                oldest = p.created
        return PinCensus(len(self._pins), tokens, max(0.0, now - oldest))

    def reapable(self, now: float, timeout: float) -> List[PinRecord]:
        """Pins held longer than ``timeout`` seconds, oldest operation first.

        Returning them in op-id order is load-bearing: the reaper releases only the
        first K of this list (K being the cross-rank MIN), so every rank must agree
        on which K those are.
        """
        if timeout <= 0:
            return []
        stale = [p for p in self._pins.values() if now - p.created > timeout]
        stale.sort(key=lambda p: p.op_id)
        return stale

    def drain(self) -> List[PinRecord]:
        """Remove and return every pin, oldest operation first. Used when the
        backup thread is known dead — nothing can be in flight, so the whole
        backlog is safe to release."""
        drained = sorted(self._pins.values(), key=lambda p: p.op_id)
        self._pins.clear()
        return drained


class PinAudit(NamedTuple):
    """Attribution of every locked (unevictable) tree node to the thing holding
    it. ``orphan_*`` is the leak: locked device tokens that no in-flight
    backup, no in-flight load and no running request can account for."""

    locked_nodes: int
    locked_tokens: int
    backup_pinned_nodes: int
    backup_pinned_tokens: int
    load_pinned_nodes: int
    load_pinned_tokens: int
    orphan_nodes: int
    orphan_tokens: int
    orphan_sample: List[int]  # ids of the first few orphans, for the log line


def audit_pins(
    root: Any,
    backup_pinned: Iterable[Any],
    load_pinned: Iterable[Any],
    sample_limit: int = 8,
) -> PinAudit:
    """Walk the tree once and attribute every locked node to an owner.

    A locked node is accounted for if it is, or is an ancestor of, a node held by
    a pending device->L3 backup or an in-flight device load — those are the two
    bypass-owned pins, and both pin the whole chain up to root. Anything else
    locked with no running request behind it is the signal the caller wants: a
    ``lock_ref`` that nothing will ever release.

    Cold path only (called when eviction fell short), so an O(nodes) walk is fine.
    """
    accounted = set()
    for node in list(backup_pinned) + list(load_pinned):
        n = node
        while n is not None and id(n) not in accounted:
            accounted.add(id(n))
            n = getattr(n, "parent", None)

    backup_ids = {id(n) for n in backup_pinned}
    load_ids = {id(n) for n in load_pinned}

    locked_nodes = locked_tokens = 0
    b_nodes = b_tokens = l_nodes = l_tokens = 0
    o_nodes = o_tokens = 0
    sample: List[int] = []

    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children.values())
        if node is root or getattr(node, "lock_ref", 0) <= 0:
            continue
        tokens = len(node.key)
        locked_nodes += 1
        locked_tokens += tokens
        if id(node) in backup_ids:
            b_nodes += 1
            b_tokens += tokens
        elif id(node) in load_ids:
            l_nodes += 1
            l_tokens += tokens
        elif id(node) not in accounted:
            o_nodes += 1
            o_tokens += tokens
            if len(sample) < sample_limit:
                sample.append(getattr(node, "id", -1))

    return PinAudit(
        locked_nodes,
        locked_tokens,
        b_nodes,
        b_tokens,
        l_nodes,
        l_tokens,
        o_nodes,
        o_tokens,
        sample,
    )
