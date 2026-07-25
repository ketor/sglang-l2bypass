"""GPU/torch-free unit tests for the increment-8 IN-FLIGHT LOAD DEDUP on the
SGLang side: two concurrent requests that share a prefix must issue ONE device SG
GET, not two, and the waiter must resolve no matter how its owner exits.

`mem_cache/hiradix_cache.py` imports torch, so it cannot be imported here. As with
the other suites, these tests AST-extract the REAL function bodies from the shipped
source and execute them against a minimal stand-in `self`, so an edit to those
functions is actually covered.

The properties under test (each one a way the dedup could go wrong):
  1. no overlap            -> no park, the request loads normally
  2. overlap               -> park, and the claimed pages are counted as saved
  3. owner promotes        -> waiter resumes (its chain is resident, it re-plans)
  4. owner aborts          -> waiter resumes and becomes the owner itself
  5. wait deadline expires -> waiter resumes anyway (an owner cannot strand it)
  6. claims are owner-keyed-> releasing req A never steals req B's claim
  7. no waiter chains      -> a parked waiter never appears as an owner

Run with `python3 test_inflight_dedup.py`.
"""
import ast
import os
import sys
import types
import unittest

PATCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PATCHED)

HRC_SRC = os.path.join(PATCHED, "mem_cache", "hiradix_cache.py")


def _extract(path, *names):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    wanted = {}

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name in names:
                    wanted[child.name] = child
            if isinstance(child, ast.ClassDef):
                walk(child)

    walk(tree)
    missing = set(names) - set(wanted)
    assert not missing, f"not found in {path}: {sorted(missing)}"
    ns = {
        "time": time_stub,
        "logger": types.SimpleNamespace(
            debug=lambda *a, **k: None, info=lambda *a, **k: None,
            warning=lambda *a, **k: None, error=lambda *a, **k: None),
    }
    module = ast.Module(body=[wanted[n] for n in names], type_ignores=[])
    for fn in module.body:
        fn.decorator_list = []
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), ns)
    return ns


class _Clock:
    """Controllable monotonic clock so the wait-deadline test needs no sleeping."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


time_stub = _Clock()

NS = _extract(
    HRC_SRC,
    "_find_inflight_owner",
    "_release_inflight_claims",
    "_resume_deduped_waiter",
)


class _Node:
    """Minimal TreeNode stand-in: an id and the page hashes the chain would load."""

    def __init__(self, node_id, pages=1):
        self.id = node_id
        self.hash_value = [f"h{node_id}_{i}" for i in range(pages)]


class _Cache:
    """Minimal HiRadixCache stand-in carrying only the dedup state."""

    def __init__(self, timeout=30.0):
        self._bypass_inflight_owner = {}
        self._bypass_waiters = {}
        self._bypass_load_state = {}
        self._bypass_dedup_parks = 0
        self._bypass_dedup_pages_saved = 0
        self._bypass_dedup_wait_timeouts = 0
        self._device_load_timeout = timeout

    find_owner = NS["_find_inflight_owner"]
    release_claims = NS["_release_inflight_claims"]
    resume_waiter = NS["_resume_deduped_waiter"]

    # --- helpers mirroring what _start_l3_async_load does around the extracted fns
    def claim(self, req_id, nodes):
        for n in nodes:
            self._bypass_inflight_owner[n.id] = req_id
        self._bypass_load_state[req_id] = object()

    def park(self, req_id, owner, nodes):
        self._bypass_dedup_parks += 1
        self._bypass_dedup_pages_saved += sum(
            len(n.hash_value) for n in nodes
            if self._bypass_inflight_owner.get(n.id) == owner
        )
        self._bypass_waiters[req_id] = (owner, time_stub.monotonic())


class InflightDedupTest(unittest.TestCase):
    def setUp(self):
        time_stub.t = 1000.0

    # 1 ------------------------------------------------------------------
    def test_no_overlap_does_not_park(self):
        c = _Cache()
        c.claim("reqA", [_Node(1), _Node(2)])
        self.assertIsNone(c.find_owner([_Node(7), _Node(8)]))
        self.assertEqual(c._bypass_dedup_parks, 0)

    # 2 ------------------------------------------------------------------
    def test_overlap_parks_and_counts_saved_pages(self):
        c = _Cache()
        chainA = [_Node(1, pages=3), _Node(2, pages=2)]
        c.claim("reqA", chainA)
        # reqB shares node 1 and 2, then extends with its own node 3.
        chainB = [_Node(1, pages=3), _Node(2, pages=2), _Node(3, pages=4)]
        owner = c.find_owner(chainB)
        self.assertEqual(owner, "reqA")
        c.park("reqB", owner, chainB)
        # 5 pages of duplicate RDMA avoided (3 + 2); node 3 is not reqA's.
        self.assertEqual(c._bypass_dedup_pages_saved, 5)
        self.assertEqual(c._bypass_dedup_parks, 1)

    def test_owner_is_the_parentmost_claim(self):
        """The parent-most claimed node decides the owner: that owner publishes the
        longest prefix usable by the waiter."""
        c = _Cache()
        c.claim("reqA", [_Node(1)])
        c.claim("reqC", [_Node(2)])
        self.assertEqual(c.find_owner([_Node(1), _Node(2)]), "reqA")

    # 3 ------------------------------------------------------------------
    def test_waiter_resumes_after_owner_promotes(self):
        c = _Cache()
        chain = [_Node(1)]
        c.claim("reqA", chain)
        c.park("reqB", "reqA", chain)
        self.assertFalse(c.resume_waiter("reqB"))  # owner still loading -> park
        # promote(): release claims, leave _bypass_load_state
        c.release_claims("reqA", chain)
        del c._bypass_load_state["reqA"]
        self.assertTrue(c.resume_waiter("reqB"))
        self.assertNotIn("reqB", c._bypass_waiters)

    # 4 ------------------------------------------------------------------
    def test_waiter_resumes_after_owner_aborts_and_can_own(self):
        c = _Cache()
        chain = [_Node(1)]
        c.claim("reqA", chain)
        c.park("reqB", "reqA", chain)
        # abort(): same teardown as promote for dedup purposes
        c.release_claims("reqA", chain)
        del c._bypass_load_state["reqA"]
        self.assertTrue(c.resume_waiter("reqB"))
        # The pages are unclaimed again, so reqB plans its own load.
        self.assertIsNone(c.find_owner(chain))
        c.claim("reqB", chain)
        self.assertEqual(c._bypass_inflight_owner[1], "reqB")

    # 5 ------------------------------------------------------------------
    def test_wait_deadline_releases_waiter(self):
        c = _Cache(timeout=30.0)
        chain = [_Node(1)]
        c.claim("reqA", chain)
        c.park("reqB", "reqA", chain)
        time_stub.t += 29.0
        self.assertFalse(c.resume_waiter("reqB"))
        self.assertEqual(c._bypass_dedup_wait_timeouts, 0)
        time_stub.t += 2.0  # now past the 30s deadline
        self.assertTrue(c.resume_waiter("reqB"))
        self.assertEqual(c._bypass_dedup_wait_timeouts, 1)
        self.assertNotIn("reqB", c._bypass_waiters)

    def test_zero_timeout_means_wait_forever_on_a_live_owner(self):
        """_device_load_timeout <= 0 disables the deadline everywhere else in the
        state machine; the waiter must follow the same convention."""
        c = _Cache(timeout=0.0)
        chain = [_Node(1)]
        c.claim("reqA", chain)
        c.park("reqB", "reqA", chain)
        time_stub.t += 10_000.0
        self.assertFalse(c.resume_waiter("reqB"))

    # 6 ------------------------------------------------------------------
    def test_release_is_owner_keyed(self):
        """A node re-claimed by a later load must not be released by the earlier
        owner's teardown — otherwise the new owner's chain silently loses its
        claim and a third request would issue a duplicate GET."""
        c = _Cache()
        n = _Node(1)
        c.claim("reqA", [n])
        c._bypass_inflight_owner[n.id] = "reqB"  # reqB re-claimed it
        c.release_claims("reqA", [n])
        self.assertEqual(c._bypass_inflight_owner[n.id], "reqB")

    # 7 ------------------------------------------------------------------
    def test_waiters_never_become_owners(self):
        """A waiter holds no task, so it must never appear in the owner map — that
        is what makes waiter chains (A waits on B waits on C) impossible."""
        c = _Cache()
        chain = [_Node(1)]
        c.claim("reqA", chain)
        c.park("reqB", "reqA", chain)
        self.assertNotIn("reqB", set(c._bypass_inflight_owner.values()))


class SourceLevelInvariantsTest(unittest.TestCase):
    """Source-level asserts for the wiring the unit tests above cannot reach."""

    def setUp(self):
        with open(HRC_SRC) as f:
            self.src = f.read()

    def test_claims_released_on_both_exits(self):
        """Promote and abort are the ONLY exits from _bypass_load_state; if either
        forgets to release, claims leak and every later request parks forever."""
        for fn in ("_promote_l3_async_load", "_abort_async_load"):
            i = self.src.index(f"def {fn}(")
            body = self.src[i:i + 2500]
            self.assertIn(
                "_release_inflight_claims", body,
                f"{fn} must release the dedup claims")

    def test_park_keeps_discovery_pending(self):
        """The park branch must push the discovery back, or the dispatcher would
        fall through to _poll on a request that has no load state."""
        i = self.src.index("owner = self._find_inflight_owner(nodes_to_load)")
        body = self.src[i:i + 900]
        self.assertIn("self._pending_l3_discovery[req_id] = pending", body)
        self.assertIn("self._bypass_waiters[req_id] = (owner", body)
        self.assertIn("return False", body)

    def test_park_branch_issues_no_collective(self):
        """The whole point of deriving the park from TP-symmetric state: a park must
        not add a reduce, or the per-round collective sequence unbalances."""
        i = self.src.index("owner = self._find_inflight_owner(nodes_to_load)")
        body = self.src[i:i + 900]
        for bad in ("_all_reduce_attn_groups", "all_reduce", "ReduceOp"):
            self.assertNotIn(bad, body, "park branch must be collective-free")

    def test_aborted_request_drops_waiter_record(self):
        i = self.src.index("def release_aborted_request(")
        body = self.src[i:i + 1200]
        self.assertIn("self._bypass_waiters.pop(rid, None)", body)

    def test_reset_clears_dedup_state(self):
        self.assertIn("self._bypass_inflight_owner.clear()", self.src)
        self.assertIn("self._bypass_waiters.clear()", self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
