from __future__ import annotations

"""L2-bypass (SGLANG_HICACHE_L2_BYPASS=1) radix-node L3 state machine.

Torch-free tree logic extracted from hiradix_cache so it can be unit-tested
without a GPU (same rationale as device_page_meta). It owns the answer to one
question the bypass tree asks everywhere: **is this evicted node's KV still
retrievable, and from where?**

In stock HiCache an in-tree node with `value is None` (evicted from device) is
ALWAYS `backuped` (it holds `host_value`) — that is what makes the match_prefix
climb and load_back safe. L2-bypass has no host tier, so the same role is played
by two dynamic markers:

  * ``l3_backed``  — THIS instance wrote the page device->L3 (write-through).
  * ``l3_present`` — the page was DISCOVERED in L3 by an exist query.

Either one makes the node loadable device-direct, so the bypass invariant is:

    an evicted, in-tree node is either backuped or L3-resident.

A node that violates it (evicted, no host copy, no L3 claim) is a **gap**: its
tokens are not retrievable from anywhere, so nothing at or below it may be
served as a cache hit. Gaps used to be an assert (which took the whole scheduler
down); the helpers here treat them as a hard miss boundary instead — the match
truncates to the retrievable part above the gap and the tokens recompute.

Safety direction is one-way on purpose: a gap only ever SHRINKS a match. A page
is never served because a marker claims it exists — every device-direct load
re-verifies page by page (consecutive_ok_pages + the cross-rank MIN) before any
token is marked usable — so a stale claim costs at most one retry, never wrong
KV.
"""

from typing import Any, Iterable, List, NamedTuple, Sequence, Tuple

__all__ = [
    "node_l3_backed",
    "node_l3_present",
    "node_l3_resident",
    "ClimbResult",
    "climb_evicted_chain",
    "collect_loadable_chain",
    "prune_l3_markers",
]


def node_l3_backed(node: Any) -> bool:
    """L2-bypass analogue of node.backuped: the node's KV has been handed to the
    device->L3 write-through. Tracked on a dynamic attribute so the stock
    TreeNode (radix_cache.py) is untouched; defaults False for stock nodes."""
    return getattr(node, "l3_backed", False)


def node_l3_present(node: Any) -> bool:
    """Read-side marker: this node's KV was DISCOVERED in L3 via an exist query
    but is not device-resident here yet — it has hash_value and l3_present=True,
    but no value and no host_value. Clears on a successful device-direct load.
    Distinct from l3_backed (which means THIS instance wrote it)."""
    return getattr(node, "l3_present", False)


def node_l3_resident(node: Any) -> bool:
    """True if the node's KV is retrievable from L3 by hash — either this
    instance wrote it (l3_backed) or it was discovered via exist (l3_present).
    Both are loadable device-direct."""
    return node_l3_backed(node) or node_l3_present(node)


class ClimbResult(NamedTuple):
    last_device_node: Any
    last_host_node: Any
    host_hit_length: int
    gap_nodes: List[Any]


def climb_evicted_chain(last_node: Any, l2_bypass: bool) -> ClimbResult:
    """match_prefix's climb: walk up the contiguous evicted chain from the
    deepest matched node, counting the tokens that are retrievable (host copy in
    stock, L3 claim in bypass) and returning the deepest device-resident node
    plus the load-back start node.

    On a gap the accounting RESTARTS above it: everything below is unreachable
    (its prefix cannot be reconstructed), so those tokens are dropped from the
    hit length and the load-back start node is moved to the gap's parent. The
    gap nodes are returned so the caller can log/count them.

    With ``l2_bypass=False`` this is the stock walk (every evicted node is
    backuped, so the gap branch is unreachable)."""
    host_hit_length = 0
    last_host_node = last_node
    gap_nodes: List[Any] = []

    node = last_node
    while node.evicted:
        if node.backuped:
            host_hit_length += len(node.host_value)
        elif l2_bypass and node_l3_resident(node):
            # No host_value in bypass, so the node's own key length is its
            # retrievable token count.
            host_hit_length += len(node.key)
        else:
            gap_nodes.append(node)
            host_hit_length = 0
            last_host_node = node.parent
        node = node.parent
    last_device_node = node

    if l2_bypass:
        # Climb to the deepest L3-resident (or, if some page is host-backed on a
        # mixed instance, backuped) node — the load-back start node.
        while not (last_host_node.backuped or node_l3_resident(last_host_node)):
            last_host_node = last_host_node.parent
    else:
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

    return ClimbResult(last_device_node, last_host_node, host_hit_length, gap_nodes)


def collect_loadable_chain(
    node: Any, root_node: Any
) -> Tuple[List[Any], Any, List[Any]]:
    """Collect the parent-first chain of evicted L3 nodes to load device-direct,
    climbing from the deepest matched node up to the first device-resident
    ancestor.

    Returns ``(nodes_to_load, ancestor, gap_nodes)``. A node is loadable only if
    it is L3-resident AND carries page hashes; hitting a gap DISCARDS everything
    collected below it (that suffix is unreachable until the gap recomputes) and
    the climb continues, so the returned chain is always contiguous with the
    device-resident ancestor."""
    nodes_to_load: List[Any] = []
    gap_nodes: List[Any] = []
    while node is not root_node and node.evicted:
        if node_l3_resident(node) and node.hash_value:
            nodes_to_load.insert(0, node)
        else:
            gap_nodes.append(node)
            nodes_to_load.clear()
        node = node.parent
    return nodes_to_load, node, gap_nodes


def prune_l3_markers(
    nodes: Sequence[Any], page_size: int, in_flight_ids: Iterable[int] = ()
) -> Tuple[List[Any], List[Any]]:
    """Drop L3 marker nodes whose device-direct load failed or was truncated, so
    their tokens recompute instead of serving unverified KV.

    Returns ``(detached, kept)``: `detached` nodes were removed from their
    parent's children (the caller refreshes the parents' leaf status); `kept`
    nodes KEEP their L3 claim because clearing it would violate the bypass
    invariant or race another request:

      * the node still has children (another request's marker branch, or a
        sibling inserted below it): it cannot be removed without orphaning them,
        and clearing the claim would leave an evicted, unservable in-tree node —
        the gap that used to trip match_prefix's assert. Keeping the claim is
        safe: the next load re-verifies every page and drops it again (this time
        detaching, once the children are gone) if it really is not in L3.
      * another request's device-direct GET is in flight over this exact node
        (``in_flight_ids``): detaching it now would leave that load promoting a
        node that is no longer in the tree, leaking its GPU slots.

    Nodes that became device-resident in the meantime just lose the (now moot)
    l3_present flag. Deepest first, so a parent is only reached after its child
    markers are gone.
    """
    in_flight = set(in_flight_ids)
    detached: List[Any] = []
    kept: List[Any] = []

    for n in reversed(nodes):
        if n.value is not None:
            # Loaded/recomputed while this drop was pending: node is usable.
            n.l3_present = False
            continue

        parent = n.parent
        attached = (
            parent is not None
            and parent.children.get(n.key.child_key(page_size)) is n
        )
        if not attached:
            # Already out of the tree (another request dropped it): the flag is
            # unreachable, clearing it keeps the node from being resurrected.
            n.l3_present = False
            continue

        if n.id in in_flight or n.children:
            kept.append(n)
            continue

        n.l3_present = False
        parent.children.pop(n.key.child_key(page_size), None)
        detached.append(n)

    return detached, kept
