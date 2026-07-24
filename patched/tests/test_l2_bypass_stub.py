"""GPU/torch-free unit tests for the L2-bypass STUB host-pool sizing logic
(mem_cache/pool_host/bypass.py).

Covers the invariants HostKVCache.__init__ relies on when it replaces the
--hicache-size pinned allocation with a minimal stub under
SGLANG_HICACHE_L2_BYPASS=1:
  * the env flag parser (what counts as "requested"),
  * the stub token count is a POSITIVE, page-aligned, non-zero value for any
    page size (so page_num / staging math never divides by zero or underflows),
  * the final size after the constructor's standard page-align is
    raw + page_size (one extra page from `// page_size + 1`),
  * the stub gate short-circuits on the env flag and otherwise defers to
    device_page_meta.supported (single source of truth for "device-expressible").

Pure python; run with `python3 test_l2_bypass_stub.py`.
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mem_cache.pool_host import bypass  # noqa: E402


class _EnvGuard:
    """Set/restore SGLANG_HICACHE_L2_BYPASS around a block."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self._prev = os.environ.get("SGLANG_HICACHE_L2_BYPASS")
        if self.value is None:
            os.environ.pop("SGLANG_HICACHE_L2_BYPASS", None)
        else:
            os.environ["SGLANG_HICACHE_L2_BYPASS"] = self.value
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("SGLANG_HICACHE_L2_BYPASS", None)
        else:
            os.environ["SGLANG_HICACHE_L2_BYPASS"] = self._prev


class TestEnvParser(unittest.TestCase):
    def test_unset_is_false(self):
        with _EnvGuard(None):
            self.assertFalse(bypass.env_l2_bypass_requested())

    def test_truthy_values(self):
        for v in ("1", "true", "TRUE", "Yes", "on", "  1 ", "On\n"):
            with _EnvGuard(v):
                self.assertTrue(bypass.env_l2_bypass_requested(), v)

    def test_falsy_values(self):
        for v in ("0", "false", "no", "off", "", "  ", "2", "enabled"):
            with _EnvGuard(v):
                self.assertFalse(bypass.env_l2_bypass_requested(), v)


class TestStubSizing(unittest.TestCase):
    PAGE_SIZES = (1, 16, 32, 64, 128, 256)

    def test_stub_pages_is_at_least_one(self):
        # < 1 would let page_num/staging degenerate; the whole point of a non-zero
        # stub is that every slot computation stays well-defined.
        self.assertGreaterEqual(bypass._L2_BYPASS_STUB_PAGES, 1)

    def test_raw_tokens_positive_page_aligned(self):
        for ps in self.PAGE_SIZES:
            raw = bypass.l2_bypass_stub_raw_tokens(ps)
            self.assertGreater(raw, 0, ps)
            self.assertEqual(raw % ps, 0, ps)
            self.assertEqual(raw, bypass._L2_BYPASS_STUB_PAGES * ps, ps)

    def test_final_tokens_match_constructor_align(self):
        # HostKVCache.__init__ does: page_num = size // page_size + 1;
        # size = page_num * page_size. Reproduce it and assert the helper agrees.
        for ps in self.PAGE_SIZES:
            raw = bypass.l2_bypass_stub_raw_tokens(ps)
            page_num = raw // ps + 1
            expected = page_num * ps
            self.assertEqual(bypass.l2_bypass_stub_tokens(ps), expected, ps)
            self.assertGreater(expected, 0, ps)
            self.assertEqual(expected % ps, 0, ps)
            # one extra page from the +1 align
            self.assertEqual(expected, raw + ps, ps)

    def test_footprint_is_small(self):
        # Sanity: at a realistic GLM-5.2-ish per-token size the stub is a few MB,
        # not GB (the whole deliverable). ~45 KB/token main MLA latent, page 64.
        tokens = bypass.l2_bypass_stub_tokens(64)
        bytes_ = tokens * 45_000
        self.assertLess(bytes_ / 1e6, 50.0)  # tens of MB ceiling


class _FakeDevicePageMeta:
    """Stand-in for sglang.srt.mem_cache.device_page_meta.supported."""

    def __init__(self, supported_result):
        self._result = supported_result
        self.calls = []

    def supported(self, pool):
        self.calls.append(pool)
        return self._result


def _install_fake_device_page_meta(fake):
    """Build the minimal fake package chain so bypass's lazy
    `from sglang.srt.mem_cache import device_page_meta` resolves to `fake`.
    Returns a restore() callable."""
    saved = {
        name: sys.modules.get(name)
        for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache")
    }
    pkg_sglang = sys.modules.get("sglang") or types.ModuleType("sglang")
    pkg_srt = sys.modules.get("sglang.srt") or types.ModuleType("sglang.srt")
    pkg_mc = sys.modules.get("sglang.srt.mem_cache") or types.ModuleType(
        "sglang.srt.mem_cache"
    )
    pkg_sglang.srt = pkg_srt
    pkg_srt.mem_cache = pkg_mc
    pkg_mc.device_page_meta = fake
    sys.modules["sglang"] = pkg_sglang
    sys.modules["sglang.srt"] = pkg_srt
    sys.modules["sglang.srt.mem_cache"] = pkg_mc

    def restore():
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    return restore


class TestStubGate(unittest.TestCase):
    def test_env_off_never_stubs(self):
        fake = _FakeDevicePageMeta(True)
        restore = _install_fake_device_page_meta(fake)
        try:
            with _EnvGuard(None):
                self.assertFalse(bypass.l2_bypass_stub_applies(object()))
            # short-circuit: supported() must not even be consulted when env is off
            self.assertEqual(fake.calls, [])
        finally:
            restore()

    def test_env_on_and_supported_stubs(self):
        fake = _FakeDevicePageMeta(True)
        restore = _install_fake_device_page_meta(fake)
        try:
            with _EnvGuard("1"):
                self.assertTrue(bypass.l2_bypass_stub_applies(object()))
            self.assertEqual(len(fake.calls), 1)
        finally:
            restore()

    def test_env_on_but_unsupported_pool_not_stubbed(self):
        fake = _FakeDevicePageMeta(False)
        restore = _install_fake_device_page_meta(fake)
        try:
            with _EnvGuard("1"):
                self.assertFalse(bypass.l2_bypass_stub_applies(object()))
        finally:
            restore()

    def test_env_on_but_module_missing_not_stubbed(self):
        # No fake installed and the real sglang isn't importable off-GPU -> the
        # lazy import raises -> gate returns False (keep the real host pool).
        saved = {
            name: sys.modules.pop(name, None)
            for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache")
        }
        try:
            with _EnvGuard("1"):
                self.assertFalse(bypass.l2_bypass_stub_applies(object()))
        finally:
            for name, mod in saved.items():
                if mod is not None:
                    sys.modules[name] = mod


if __name__ == "__main__":
    unittest.main(verbosity=2)
