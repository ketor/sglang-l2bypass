"""GPU/torch-free unit tests for the increment-7 FUSED-DRAFT gate on the SGLang
side: the env parser, the `draft_rides_target_batch` decision table, and the
inertness of the standalone draft ops once the draft rides the target's batch.

`managers/cache_controller.py` imports torch, so it cannot be imported here. These
tests do not simulate the logic — they AST-extract the REAL function bodies from
the shipped source file and execute them, so a future edit to those functions is
actually covered. The four call sites that must pass `with_draft=` are asserted at
the source level (a dropped kwarg would silently double the draft's RDMA ops).

Run with `python3 test_fused_draft_gate.py`.
"""
import ast
import os
import sys
import types
import unittest

PATCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PATCHED)

CC_SRC = os.path.join(PATCHED, "managers", "cache_controller.py")
HYBRID_SRC = os.path.join(
    PATCHED, "mem_cache", "hybrid_cache", "hybrid_cache_controller.py")


def _extract(path, *names):
    """Exec the named top-level functions / class-body methods from `path` into a
    fresh namespace with the module-level imports they need stubbed out."""
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
        "os": os,
        # _draft_device_set / _maybe_device_draft_get log on the failure branch.
        "logger": types.SimpleNamespace(
            debug=lambda *a, **k: None, info=lambda *a, **k: None,
            warning=lambda *a, **k: None),
    }
    module = ast.Module(body=[wanted[n] for n in names], type_ignores=[])
    # Strip decorators (@property) — we want the plain function to call directly.
    for fn in module.body:
        fn.decorator_list = []
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), ns)
    return ns


class TestEnvParser(unittest.TestCase):
    """SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT defaults ON (fusion is the point of the
    increment); the documented falsy spellings turn it back off for an A/B."""

    def setUp(self):
        self.fn = _extract(CC_SRC, "env_l2_bypass_fuse_draft")[
            "env_l2_bypass_fuse_draft"]
        self._saved = os.environ.get("SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT")

    def tearDown(self):
        os.environ.pop("SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT", None)
        if self._saved is not None:
            os.environ["SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT"] = self._saved

    def _set(self, v):
        if v is None:
            os.environ.pop("SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT", None)
        else:
            os.environ["SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT"] = v

    def test_default_is_off(self):
        """Flipped 2026-07-25: in the one controlled A/B we have, fusion did not
        pay for itself (R2 +17.5% but R3 -43.5% tok/s, and an op mix showing
        re-derivation: reads 13 -> 8, writes 55 -> 134). No root cause established
        — see env_l2_bypass_fuse_draft's docstring for why the first attribution
        was withdrawn. Default off until a repeated-R3 A/B says otherwise."""
        self._set(None)
        self.assertFalse(self.fn())

    def test_falsy_spellings_turn_it_off(self):
        for v in ("0", "false", "FALSE", "no", "off", "", "  off  "):
            self._set(v)
            self.assertFalse(self.fn(), f"{v!r} should disable fusion")

    def test_truthy_spellings_keep_it_on(self):
        for v in ("1", "true", "yes", "on", "anything"):
            self._set(v)
            self.assertTrue(self.fn(), f"{v!r} should keep fusion on")


class _Backend:
    def __init__(self):
        self.calls = []

    def batch_set_v1_device_draft(self, hashes, indices):
        self.calls.append(("set", tuple(hashes)))
        return [True] * len(hashes)

    def batch_get_v1_device_draft(self, hashes, indices):
        self.calls.append(("get", tuple(hashes)))
        return [True] * len(hashes)


class _Task:
    def __init__(self):
        self.hash_values = ["h0", "h1"]
        self.device_indices = [0, 1]


class TestDraftGateDecisionTable(unittest.TestCase):
    """`draft_rides_target_batch` is the single switch the four device call sites
    read; the standalone `_draft_device_set` / `_maybe_device_draft_get` must be
    exactly its complement, or the draft is written twice (fused + standalone) or
    not at all."""

    def setUp(self):
        self.ns = _extract(
            CC_SRC, "draft_rides_target_batch", "_draft_device_set",
            "_maybe_device_draft_get")

    def _self(self, enabled, fused):
        return types.SimpleNamespace(
            draft_device_enabled=enabled, draft_device_fused=fused,
            storage_backend=_Backend())

    def test_rides_only_when_enabled_and_fused(self):
        rides = self.ns["draft_rides_target_batch"]
        self.assertTrue(rides(self._self(True, True)))
        self.assertFalse(rides(self._self(True, False)))
        self.assertFalse(rides(self._self(False, True)))
        self.assertFalse(rides(self._self(False, False)))

    def test_standalone_write_is_the_complement(self):
        for enabled, fused, expect in (
            (True, True, []),                    # fused: target batch carried it
            (True, False, [("set", ("h0",))]),   # unfused: its own RDMA op
            (False, True, []),                   # draft L3 off entirely
            (False, False, []),
        ):
            s = self._self(enabled, fused)
            self.ns["_draft_device_set"](s, ["h0"], [0])
            self.assertEqual(s.storage_backend.calls, expect,
                             f"enabled={enabled} fused={fused}")

    def test_standalone_read_is_the_complement(self):
        for enabled, fused, expect in (
            (True, True, []),
            (True, False, [("get", ("h0", "h1"))]),
            (False, True, []),
            (False, False, []),
        ):
            s = self._self(enabled, fused)
            self.ns["_maybe_device_draft_get"](s, _Task())
            self.assertEqual(s.storage_backend.calls, expect,
                             f"enabled={enabled} fused={fused}")

    def test_standalone_write_swallows_backend_failure(self):
        """Best-effort contract: a draft write failure must never propagate (the
        target verifies the draft, so it only costs acceptance)."""
        s = self._self(True, False)

        def boom(*a):
            raise RuntimeError("rdma down")

        s.storage_backend.batch_set_v1_device_draft = boom
        self.ns["_draft_device_set"](s, ["h0"], [0])  # must not raise

    def test_standalone_read_swallows_backend_failure(self):
        s = self._self(True, False)

        def boom(*a):
            raise RuntimeError("rdma down")

        s.storage_backend.batch_get_v1_device_draft = boom
        self.ns["_maybe_device_draft_get"](s, _Task())  # must not raise


class TestCallSitesPassWithDraft(unittest.TestCase):
    """Source-level guard: every device set/get call site must hand the backend
    `with_draft=self.draft_rides_target_batch`. Dropping it on one site would make
    that path pay for the draft twice (fused kwarg missing => standalone op runs)
    or lose the draft entirely, and neither shows up as a test failure elsewhere."""

    EXPECTED = {
        CC_SRC: ["batch_set_v1_device(", "batch_get_v1_device("],
        HYBRID_SRC: ["batch_set_v2_device(", "batch_get_v2_device("],
    }

    def test_every_device_io_call_site_passes_the_flag(self):
        for path, calls in self.EXPECTED.items():
            with open(path) as f:
                src = f.read()
            for call in calls:
                # Every `storage_backend.<call>` invocation must carry the kwarg
                # within the following few lines (the call spans a line break).
                start = 0
                found = 0
                needle = "storage_backend." + call
                while True:
                    i = src.find(needle, start)
                    if i < 0:
                        break
                    found += 1
                    window = src[i:i + 400]
                    self.assertIn(
                        "with_draft=", window,
                        f"{os.path.basename(path)}: {needle} at offset {i} does "
                        f"not pass with_draft=")
                    start = i + 1
                self.assertGreater(found, 0, f"{path}: no call to {needle}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
