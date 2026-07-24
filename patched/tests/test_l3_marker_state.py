"""GPU/torch-free unit tests for the L2-bypass radix-node L3 state machine
(mem_cache/l3_marker_state.py).

Regression cover for the production crash

    AssertionError: evicted non-backuped node 2 outside L2-bypass
      hiradix_cache.match_prefix -> while last_node.evicted: assert ...

whose root cause was _drop_l3_markers clearing `l3_present` on a marker it could
NOT detach (the node had acquired a child — another request's marker branch), so
the node stayed in the tree evicted, host-less and claim-less: a GAP. The next
request whose match walked through it tripped the assert and took the scheduler
(all 8 ranks) down.

Two properties are asserted here:
  1. prune_l3_markers NEVER creates a gap (root-cause fix), and
  2. climb_evicted_chain / collect_loadable_chain treat a gap as a MISS boundary
     rather than crashing (bounded fallback), while never counting an
     unretrievable node as a hit — the safety direction is miss-only.

Pure python; run with `python3 test_l3_marker_state.py`.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mem_cache.l3_marker_state import (  # noqa: E402
    climb_evicted_chain,
    collect_loadable_chain,
    node_l3_resident,
    prune_l3_markers,
)

PAGE = 2


class FakeKey:
    """Stand-in for RadixKey: a token list whose child_key is its first page."""

    def __init__(self, tokens):
        self.tokens = list(tokens)

    def __len__(self):
        return len(self.tokens)

    def child_key(self, page_size):
        return tuple(self.tokens[:page_size])


class FakeNode:
    """Duck-typed TreeNode with just what l3_marker_state touches."""

    _counter = [0]

    def __init__(self, tokens=(), parent=None, value=None, host_value=None):
        self.key = FakeKey(tokens)
        self.parent = parent
        self.children = {}
        self.value = value
        self.host_value = host_value
        self.hash_value = ["h"] * (len(tokens) // PAGE) if tokens else []
        self.l3_present = False
        self.l3_backed = False
        FakeNode._counter[0] += 1
        self.id = FakeNode._counter[0]
        if parent is not None:
            parent.children[self.key.child_key(PAGE)] = self

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"<node {self.id} tokens={self.key.tokens} evicted={self.evicted}>"


def make_root():
    # Mirrors RadixCache.reset(): the root is never evicted and is "backuped",
    # so every climb terminates there.
    return FakeNode(tokens=(), value=[], host_value=[])


def marker(tokens, parent):
    """An L3 marker node: discovered in L3, no device slots, no host slots."""
    n = FakeNode(tokens, parent=parent)
    n.l3_present = True
    return n


def device_node(tokens, parent):
    n = FakeNode(tokens, parent=parent, value=list(tokens))
    n.l3_backed = True
    return n


def gap_node(tokens, parent):
    """The broken state: in the tree, evicted, no host copy, no L3 claim."""
    return FakeNode(tokens, parent=parent)


class TestPruneNeverCreatesGap(unittest.TestCase):
    """Root-cause fix: after any drop path, no in-tree evicted node is left
    without a host copy or an L3 claim."""

    def assert_no_gap(self, root):
        stack = [root]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n is root or not n.evicted:
                continue
            self.assertTrue(
                n.backuped or node_l3_resident(n),
                f"gap left in tree: {n}",
            )

    def test_childless_chain_is_detached(self):
        root = make_root()
        a = marker([1, 2], root)
        b = marker([3, 4], a)
        detached, kept = prune_l3_markers([a, b], PAGE)
        self.assertEqual([n.id for n in detached], [b.id, a.id])
        self.assertEqual(kept, [])
        self.assertEqual(root.children, {})
        self.assert_no_gap(root)

    def test_marker_with_sibling_branch_keeps_its_claim(self):
        # THE CRASH SCENARIO: while A->B was being loaded, another request
        # discovered a different suffix under A (sibling C). The load fails, the
        # chain [A, B] is dropped: B detaches, A cannot (C hangs off it).
        root = make_root()
        a = marker([1, 2], root)
        b = marker([3, 4], a)
        c = marker([5, 6], a)

        detached, kept = prune_l3_markers([a, b], PAGE)

        self.assertEqual([n.id for n in detached], [b.id])
        self.assertEqual([n.id for n in kept], [a.id])
        self.assertTrue(a.l3_present, "kept marker must keep its L3 claim")
        self.assertIs(root.children[a.key.child_key(PAGE)], a)
        self.assertIn(c.key.child_key(PAGE), a.children)
        self.assert_no_gap(root)

    def test_node_that_became_device_resident_only_loses_the_flag(self):
        root = make_root()
        a = marker([1, 2], root)
        a.value = [1, 2]  # promoted by another request's load in the meantime
        detached, kept = prune_l3_markers([a], PAGE)
        self.assertEqual(detached, [])
        self.assertEqual(kept, [])
        self.assertFalse(a.l3_present)
        self.assertIs(root.children[a.key.child_key(PAGE)], a)
        self.assert_no_gap(root)

    def test_node_in_flight_for_another_request_is_kept(self):
        # Another request's background GET still owns this node; detaching it
        # would leave that load promoting a node outside the tree (leaked slots).
        root = make_root()
        a = marker([1, 2], root)
        detached, kept = prune_l3_markers([a], PAGE, in_flight_ids={a.id})
        self.assertEqual(detached, [])
        self.assertEqual([n.id for n in kept], [a.id])
        self.assertTrue(a.l3_present)
        self.assertIs(root.children[a.key.child_key(PAGE)], a)
        self.assert_no_gap(root)

    def test_already_detached_node_is_not_re_popped(self):
        # Two requests drop the same chain; the second must not pop a node the
        # parent no longer owns (nor resurrect it).
        root = make_root()
        a = marker([1, 2], root)
        b = marker([3, 4], a)
        prune_l3_markers([a, b], PAGE)
        detached, kept = prune_l3_markers([a, b], PAGE)
        self.assertEqual(detached, [])
        self.assertEqual(kept, [])
        self.assertFalse(a.l3_present)
        self.assertFalse(b.l3_present)
        self.assert_no_gap(root)

    def test_l3_backed_node_keeps_claim_when_present_is_cleared(self):
        # A self-written page that was also rediscovered: dropping the read-side
        # claim must not strip the write-side one.
        root = make_root()
        a = marker([1, 2], root)
        a.l3_backed = True
        marker([3, 4], a)  # child keeps a from being detached
        detached, kept = prune_l3_markers([a], PAGE)
        self.assertEqual(detached, [])
        self.assertEqual([n.id for n in kept], [a.id])
        self.assertTrue(node_l3_resident(a))
        self.assert_no_gap(root)

    def test_partial_suffix_drop_keeps_loaded_prefix(self):
        # Partial verify: prefix promoted to device-resident, suffix dropped.
        root = make_root()
        a = device_node([1, 2], root)
        b = marker([3, 4], a)
        detached, kept = prune_l3_markers([b], PAGE)
        self.assertEqual([n.id for n in detached], [b.id])
        self.assertEqual(a.children, {})
        self.assertTrue(node_l3_resident(a))
        self.assert_no_gap(root)


class TestClimbEvictedChain(unittest.TestCase):
    def test_all_markers_counted_as_host_hit(self):
        root = make_root()
        d = device_node([1, 2], root)
        a = marker([3, 4], d)
        b = marker([5, 6], a)
        r = climb_evicted_chain(b, l2_bypass=True)
        self.assertIs(r.last_device_node, d)
        self.assertIs(r.last_host_node, b)
        self.assertEqual(r.host_hit_length, 4)
        self.assertEqual(r.gap_nodes, [])

    def test_gap_truncates_hit_and_moves_load_start_above_it(self):
        # d -> a(marker) -> g(GAP) -> b(marker): only `a` is servable.
        root = make_root()
        d = device_node([1, 2], root)
        a = marker([3, 4], d)
        g = gap_node([5, 6], a)
        b = marker([7, 8], g)

        r = climb_evicted_chain(b, l2_bypass=True)

        self.assertIs(r.last_device_node, d)
        self.assertIs(r.last_host_node, a, "load must start above the gap")
        self.assertEqual(r.host_hit_length, 2, "only `a` is retrievable")
        self.assertEqual([n.id for n in r.gap_nodes], [g.id])

    def test_gap_directly_below_device_prefix_is_a_full_miss(self):
        root = make_root()
        d = device_node([1, 2], root)
        g = gap_node([3, 4], d)
        b = marker([5, 6], g)

        r = climb_evicted_chain(b, l2_bypass=True)

        self.assertIs(r.last_device_node, d)
        self.assertIs(r.last_host_node, d)
        self.assertEqual(r.host_hit_length, 0)
        self.assertEqual([n.id for n in r.gap_nodes], [g.id])

    def test_gap_at_top_level_falls_back_to_root(self):
        # The reported crash shape: node 2 (top-level shared prefix) is the gap.
        root = make_root()
        g = gap_node([1, 2], root)
        b = marker([3, 4], g)

        r = climb_evicted_chain(b, l2_bypass=True)

        self.assertIs(r.last_device_node, root)
        self.assertIs(r.last_host_node, root)
        self.assertEqual(r.host_hit_length, 0)

    def test_multiple_gaps_keep_only_the_top_servable_run(self):
        root = make_root()
        d = device_node([1, 2], root)
        a = marker([3, 4], d)
        g1 = gap_node([5, 6], a)
        m = marker([7, 8], g1)
        g2 = gap_node([9, 10], m)
        leaf = marker([11, 12], g2)

        r = climb_evicted_chain(leaf, l2_bypass=True)

        self.assertIs(r.last_host_node, a)
        self.assertEqual(r.host_hit_length, 2)
        self.assertEqual({n.id for n in r.gap_nodes}, {g1.id, g2.id})

    def test_device_resident_start_node_is_untouched(self):
        root = make_root()
        d = device_node([1, 2], root)
        r = climb_evicted_chain(d, l2_bypass=True)
        self.assertIs(r.last_device_node, d)
        self.assertIs(r.last_host_node, d)
        self.assertEqual(r.host_hit_length, 0)
        self.assertEqual(r.gap_nodes, [])

    def test_stock_host_backed_chain_unchanged_with_flag_off(self):
        # flag off: evicted nodes are host-backed; behavior identical to stock.
        root = make_root()
        d = FakeNode([1, 2], parent=root, value=[1, 2], host_value=[1, 2])
        h1 = FakeNode([3, 4], parent=d, host_value=[3, 4])
        h2 = FakeNode([5, 6], parent=h1, host_value=[5, 6])
        r = climb_evicted_chain(h2, l2_bypass=False)
        self.assertIs(r.last_device_node, d)
        self.assertIs(r.last_host_node, h2)
        self.assertEqual(r.host_hit_length, 4)
        self.assertEqual(r.gap_nodes, [])

    def test_bypass_ignores_l3_claims_when_flag_off(self):
        # A tree built under bypass, read with the flag off: markers are NOT
        # counted as hits (their KV is not reachable through the stock path).
        root = make_root()
        d = device_node([1, 2], root)
        m = marker([3, 4], d)
        r = climb_evicted_chain(m, l2_bypass=False)
        self.assertEqual(r.host_hit_length, 0)
        self.assertIs(r.last_host_node, root)
        self.assertEqual([n.id for n in r.gap_nodes], [m.id])


class TestCollectLoadableChain(unittest.TestCase):
    def test_full_marker_chain_is_parent_first(self):
        root = make_root()
        d = device_node([1, 2], root)
        a = marker([3, 4], d)
        b = marker([5, 6], a)
        nodes, ancestor, gaps = collect_loadable_chain(b, root)
        self.assertEqual([n.id for n in nodes], [a.id, b.id])
        self.assertIs(ancestor, d)
        self.assertEqual(gaps, [])

    def test_gap_discards_everything_below_it(self):
        root = make_root()
        d = device_node([1, 2], root)
        a = marker([3, 4], d)
        g = gap_node([5, 6], a)
        b = marker([7, 8], g)
        nodes, ancestor, gaps = collect_loadable_chain(b, root)
        self.assertEqual([n.id for n in nodes], [a.id])
        self.assertIs(ancestor, d)
        self.assertEqual([n.id for n in gaps], [g.id])

    def test_gap_only_chain_yields_nothing_to_load(self):
        root = make_root()
        d = device_node([1, 2], root)
        g = gap_node([3, 4], d)
        b = marker([5, 6], g)
        nodes, ancestor, gaps = collect_loadable_chain(b, root)
        self.assertEqual(nodes, [])
        self.assertIs(ancestor, d)
        self.assertEqual([n.id for n in gaps], [g.id])

    def test_marker_without_hashes_is_treated_as_a_gap(self):
        # No page hashes => nothing to GET by; must never be handed to the loader.
        root = make_root()
        d = device_node([1, 2], root)
        a = marker([3, 4], d)
        a.hash_value = None
        b = marker([5, 6], a)
        nodes, ancestor, gaps = collect_loadable_chain(b, root)
        self.assertEqual(nodes, [])
        self.assertIs(ancestor, d)
        self.assertEqual([n.id for n in gaps], [a.id])

    def test_device_resident_node_yields_empty_chain(self):
        root = make_root()
        d = device_node([1, 2], root)
        nodes, ancestor, gaps = collect_loadable_chain(d, root)
        self.assertEqual(nodes, [])
        self.assertIs(ancestor, d)
        self.assertEqual(gaps, [])

    def test_chain_climbs_to_root_when_nothing_is_device_resident(self):
        root = make_root()
        a = marker([1, 2], root)
        b = marker([3, 4], a)
        nodes, ancestor, gaps = collect_loadable_chain(b, root)
        self.assertEqual([n.id for n in nodes], [a.id, b.id])
        self.assertIs(ancestor, root)
        self.assertEqual(gaps, [])


class TestDropThenMatchEndToEnd(unittest.TestCase):
    """The full production sequence, on the fake tree: discover -> sibling
    branch -> failed load drops the chain -> a later request matches through the
    kept node. Before the fix this left a gap and match_prefix asserted."""

    def test_sequence_never_asserts_and_never_over_reports(self):
        root = make_root()
        a = marker([1, 2], root)  # shared prefix discovered in L3
        b = marker([3, 4], a)  # request 1's suffix
        c = marker([5, 6], a)  # request 2's suffix (sibling)

        # Request 1's device-direct load verifies 0 pages -> drop [a, b].
        prune_l3_markers([a, b], PAGE)

        # Request 2 now matches ... a -> c.
        r = climb_evicted_chain(c, l2_bypass=True)
        self.assertEqual(r.gap_nodes, [], "no gap must remain")
        self.assertIs(r.last_host_node, c)
        self.assertEqual(r.host_hit_length, 4)

        # And the load chain it hands to the GET is exactly [a, c].
        nodes, ancestor, gaps = collect_loadable_chain(c, root)
        self.assertEqual([n.id for n in nodes], [a.id, c.id])
        self.assertIs(ancestor, root)
        self.assertEqual(gaps, [])

    def test_gap_from_any_source_degrades_to_miss(self):
        # Defence in depth: even if some future path manufactures a gap, the
        # match degrades instead of crashing, and reports FEWER hits, never more.
        root = make_root()
        d = device_node([1, 2], root)
        g = gap_node([3, 4], d)
        leaf = marker([5, 6], g)
        r = climb_evicted_chain(leaf, l2_bypass=True)
        self.assertEqual(r.host_hit_length, 0)
        self.assertIs(r.last_host_node, d)
        nodes, ancestor, _ = collect_loadable_chain(r.last_host_node, root)
        self.assertEqual(nodes, [], "nothing below a gap may be loaded")
        self.assertIs(ancestor, d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
