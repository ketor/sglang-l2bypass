from __future__ import annotations

"""Pure (torch-free) helpers for the HiCache L2-bypass STUB host pool
(SGLANG_HICACHE_L2_BYPASS=1).

When L2-bypass is active the host KV pool is a NON-load-bearing placeholder: KV
moves GPU<->L3 by GPUDirect RDMA and never allocates a host slot (audited in
PATCH-MANIFEST "Host-pool residual audit"). So under bypass HostKVCache.__init__
does NOT pin the --hicache-size buffer; it builds a minimal stub whose pinned
footprint is a few hundred MB at most instead of the tens/hundreds of GB an
unstubbed --hicache-size would pin. --hicache-size is IGNORED in this mode.

The stub is sized in PAGES, floored at the pool's layer_num (see
l2_bypass_stub_pages: page-major layouts index dim0 -- page_num -- by layer), and
shared across every stub pool in the process (see
l2_bypass_shared_stub_raw_tokens: host slot indices are 1-to-1 target<->draft).

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
# underflow to zero. This is the FLOOR only; the effective stub is
# max(_L2_BYPASS_STUB_PAGES, layer_num) pages -- see l2_bypass_stub_pages.
_L2_BYPASS_STUB_PAGES = 1

_TRUTHY = ("1", "true", "yes", "on")

# Per-process stub slot count, keyed by page_size. See
# l2_bypass_shared_stub_raw_tokens for why every stub pool must agree on it.
_SHARED_STUB_RAW_TOKENS: dict = {}


def env_l2_bypass_requested() -> bool:
    """Whether SGLANG_HICACHE_L2_BYPASS requests bypass. Mirrors the flag the cache
    controller reads (managers/cache_controller.env_l2_bypass); a request is
    necessary but NOT sufficient -- the controller's capability gate decides
    whether bypass is actually enabled at runtime."""
    return os.environ.get("SGLANG_HICACHE_L2_BYPASS", "").strip().lower() in _TRUTHY


def l2_bypass_stub_pages(layer_num: int = 1) -> int:
    """Page-units the stub holds before the standard align.

    Floored at ``layer_num`` because several stock host-pool layouts are PAGE-major
    -- their buffer's dim0 is page_num, not layer_num (MLA page_first_direct /
    page_first_kv_split: memory_pool_host.py:1338-1354; MHA page_first_direct /
    page_head: memory_pool_host.py:154-171) -- while the constructor still builds
    per-layer views out of that same dim0 with
    ``[kv_buffer[i] for i in range(self.layer_num)]``
    (memory_pool_host.py:1295 for MLA, :126 for MHA). Those views are only in range
    because a real host pool has page_num >> layer_num. A flat 1-page stub gave
    page_num=2 < layer_num=78 and the constructor died with
    ``IndexError: index 2 is out of bounds for dimension 0 with size 2``.

    page_num = pages + 1, so pages >= layer_num keeps every per-layer view in range
    with one page of margin. This changes CAPACITY only -- the buffer's dimension
    semantics stay exactly stock (layer_num is still whatever dim the layout says).
    Cost is layer_num pages of the stub instead of 1 (a few hundred MB at
    GLM-5.2 scale vs. the ~100 GB an unstubbed --hicache-size would pin).
    """
    return max(_L2_BYPASS_STUB_PAGES, max(1, int(layer_num)))


def l2_bypass_stub_raw_tokens(page_size: int, layer_num: int = 1) -> int:
    """Token count the stub assigns to self.size BEFORE HostKVCache.__init__'s
    standard page-align. Positive multiple of page_size for any page_size >= 1."""
    return l2_bypass_stub_pages(layer_num) * page_size


def l2_bypass_stub_tokens(page_size: int, layer_num: int = 1) -> int:
    """Final stub host-pool token count AFTER the standard page-align that
    HostKVCache.__init__ applies (page_num = size // page_size + 1;
    size = page_num * page_size). Exposed for tests / footprint reporting; the
    constructor reaches the same value via the shared align path."""
    raw = l2_bypass_stub_raw_tokens(page_size, layer_num)
    page_num = raw // page_size + 1
    return page_num * page_size


def l2_bypass_shared_stub_raw_tokens(page_size: int, layer_num: int = 1) -> int:
    """Stub token count for a pool, kept IDENTICAL across every stub host pool in
    the process (monotonic max, keyed by page_size).

    Host slot indices are 1-to-1 across the pools of one HiCache stack: the draft
    host pool is deliberately built with ``host_to_device_ratio = primary.size /
    draft_device.size`` so it has the same slot count as the target host pool
    (kv_cache_builder.py:101-110), and the target's host_indices are then used
    verbatim on the draft pool (cache_controller backup_from_device_all_layer /
    get_data_page). The stub ignores ratio/--hicache-size, so without this the
    per-pool layer_num floor would give the 78-layer target 79 pages and the
    1-layer EAGLE draft 1 page, and the residual (bypass-declined) host path could
    index the draft pool out of range.

    The target pool is constructed first (the draft registers against an existing
    tree cache), so the max is reached on the first call and later pools just
    reuse it. Growing later is still safe for the pool being built; it only means
    an earlier, smaller pool exists -- which cannot happen for the EAGLE draft
    (layer_num=1).
    """
    raw = l2_bypass_stub_raw_tokens(page_size, layer_num)
    shared = max(_SHARED_STUB_RAW_TOKENS.get(page_size, 0), raw)
    _SHARED_STUB_RAW_TOKENS[page_size] = shared
    return shared


def reset_l2_bypass_stub_sizing() -> None:
    """Drop the per-process shared stub slot count (tests only)."""
    _SHARED_STUB_RAW_TOKENS.clear()


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
