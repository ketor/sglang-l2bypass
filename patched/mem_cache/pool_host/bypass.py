from __future__ import annotations

"""Pure (torch-free) helpers for the HiCache L2-bypass STUB host pool
(SGLANG_HICACHE_L2_BYPASS=1).

When L2-bypass is active the host KV pool is a NON-load-bearing placeholder: KV
moves GPU<->L3 by GPUDirect RDMA and never allocates a host slot (audited in
PATCH-MANIFEST "Host-pool residual audit"). So under bypass HostKVCache.__init__
does NOT pin the --hicache-size buffer; it builds a minimal stub whose pinned
footprint is a few MB instead of the GB an unstubbed --hicache-size would pin.
--hicache-size is IGNORED in this mode.

Gating (so the stock path stays byte-identical):
  stub applies  <=>  SGLANG_HICACHE_L2_BYPASS requested
                     AND device_page_meta.supported(device_pool)  (MLA incl. DSA
                     main latent, or MHA -- the pools bypass can own device-direct).
Pools bypass cannot express (Mamba/SWA/sparse) are never stubbed: they keep the
real host pool and the honest stock path.

Safety net for the residual case (flag on, pool expressible, but the controller
later DECLINES bypass for a backend reason -- e.g. a non-device backend, or an HCA
too narrow for the @sg chunking): the stock host path still runs correctly against
a stub, because every mem_pool_host.alloc caller treats alloc()==None (a full/tiny
pool) as a recompute-safe skip -- write-back and prefetch simply no-op, L2 goes
ineffective, correctness is preserved. It is LOUD (stub log at construction +
the controller's decline warning), never silent corruption.

Kept torch/sglang-free on purpose so the sizing invariants are unit-testable off
the GPU box (test/test_l2_bypass_stub.py), like device_page_meta.
"""

import os

# Page-units the stub host pool holds BEFORE the standard page-alignment in
# HostKVCache.__init__. Must be >= 1 so page_num (= size // page_size + 1) and the
# write-back staging capacity (min(page_num, chunk)) stay well-defined and never
# underflow to zero. Kept tiny so the pinned footprint is a few MB. After the
# standard align the pool ends up with (_L2_BYPASS_STUB_PAGES + 1) * page_size
# tokens (e.g. 2 pages).
_L2_BYPASS_STUB_PAGES = 1

_TRUTHY = ("1", "true", "yes", "on")


def env_l2_bypass_requested() -> bool:
    """Whether SGLANG_HICACHE_L2_BYPASS requests bypass. Mirrors the flag the cache
    controller reads (managers/cache_controller.env_l2_bypass); a request is
    necessary but NOT sufficient -- the controller's capability gate decides
    whether bypass is actually enabled at runtime."""
    return os.environ.get("SGLANG_HICACHE_L2_BYPASS", "").strip().lower() in _TRUTHY


def l2_bypass_stub_raw_tokens(page_size: int) -> int:
    """Token count the stub assigns to self.size BEFORE HostKVCache.__init__'s
    standard page-align. Positive multiple of page_size for any page_size >= 1."""
    return _L2_BYPASS_STUB_PAGES * page_size


def l2_bypass_stub_tokens(page_size: int) -> int:
    """Final stub host-pool token count AFTER the standard page-align that
    HostKVCache.__init__ applies (page_num = size // page_size + 1;
    size = page_num * page_size). Exposed for tests / footprint reporting; the
    constructor reaches the same value via the shared align path."""
    raw = l2_bypass_stub_raw_tokens(page_size)
    page_num = raw // page_size + 1
    return page_num * page_size


def l2_bypass_stub_applies(device_pool) -> bool:
    """True iff bypass is requested AND this GPU pool is device-direct-expressible.

    device_page_meta.supported is the single source of truth for "expressible"; it
    is imported LAZILY so the stock import path is untouched and only the flag-on
    path pulls in the bypass module. If the bypass module is somehow unavailable,
    do NOT stub -- keep the real (stock) host pool.
    """
    if not env_l2_bypass_requested():
        return False
    try:
        from sglang.srt.mem_cache import device_page_meta

        return bool(device_page_meta.supported(device_pool))
    except Exception:
        return False
