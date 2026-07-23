from __future__ import annotations

"""Device-side page buffer meta for the HiCache L2-bypass (device-direct write)
prototype (SGLANG_HICACHE_L2_BYPASS=1).

Mirrors memory_pool_host.MHATokenToKVPoolHost.get_page_buffer_meta (the host
zero-copy arithmetic) but over the GPU KV pool. The GPU pool is *layer-first*:
each layer is a separate allocation (MHATokenToKVPool.k_buffer/v_buffer are
lists of per-layer tensors; MLATokenToKVPool.kv_buffer likewise). A page's KV is
therefore NOT contiguous across layers, so — unlike the host page-first pool,
whose page is one contiguous blob — a device page must be expressed as a
scatter-gather list of per-layer segments. Each sub-object (k, and v for MHA) of
a page becomes ``layer_num`` device segments; the dfkv backend hands them to the
scatter-gather C ABI (dfkv_batch_put_sg), which stores the payload as the
concatenation of the segments.

CORRECTNESS NOTE (layer-major vs page-first ordering): the segments here
concatenate in LAYER-major order (layer0[tok0..tokP], layer1[tok0..tokP], ...).
The stock host read path (batch_get_v1) scatters the stored blob into a
page-first host buffer, whose byte order is TOKEN-major (tok0[layer0..layerN],
...). The two orderings are transposes of each other. A page written
device-direct is therefore byte-coherent only with a matching device-direct
(layer-major SG) reader — increment 2 — not with the unchanged page-first host
read. Increment 1 wires the write path and its offload; enabling it in isolation
is a benchmark/prototype mode. See PATCH-MANIFEST.md.
"""

from typing import List, Sequence, Tuple


def consecutive_ok_pages(
    kv_ok: Sequence[bool],
    sidecar_oks: Sequence[Sequence[bool]],
    npages: int,
) -> int:
    """Longest consecutive page prefix (from the start) where the main KV AND every
    sidecar hit — the verified prefix of an L2-bypass device-direct GET (increment
    2/3). A page is usable only if it and all before it are complete, so the first
    miss / short-read list truncates the prefix and the caller recomputes the tail
    rather than serving a hole. Pure logic (no torch/GPU) so it is unit-testable off
    the GPU box; used by HiCacheController._run_device_get (dense, sidecar_oks=[])
    and HybridCacheController._run_device_get (DSA main KV + indexer sidecars)."""
    ok = 0
    for p in range(npages):
        if p >= len(kv_ok) or not kv_ok[p]:
            break
        if any(p >= len(s) or not s[p] for s in sidecar_oks):
            break
        ok += 1
    return ok


def _mla_layer_bases(pool) -> Tuple[List[int], int]:
    """(per-layer base data_ptr, per-token byte stride) for an MLA GPU pool."""
    layers = pool.kv_buffer  # list[layer] of (size+page_size, 1, kv_cache_dim)
    itemsize = layers[0].element_size()
    token_stride = pool.kv_cache_dim * itemsize
    return [int(t.data_ptr()) for t in layers], token_stride


def _mha_layer_bases(pool) -> Tuple[List[int], int, List[int], int]:
    """(k bases, k token stride, v bases, v token stride) for an MHA GPU pool."""
    k_layers = pool.k_buffer  # list[layer] of (size+page_size, head_num, head_dim)
    v_layers = pool.v_buffer
    k_itemsize = k_layers[0].element_size()
    v_itemsize = v_layers[0].element_size()
    k_stride = pool.head_num * pool.head_dim * k_itemsize
    v_stride = pool.head_num * pool.v_head_dim * v_itemsize
    return (
        [int(t.data_ptr()) for t in k_layers],
        k_stride,
        [int(t.data_ptr()) for t in v_layers],
        v_stride,
    )


def _is_mla(pool) -> bool:
    return hasattr(pool, "kv_buffer") and hasattr(pool, "kv_cache_dim")


def _is_mha(pool) -> bool:
    return hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer")


def supported(pool) -> bool:
    """True if get_device_page_buffer_meta can express this GPU pool's MAIN KV.

    DSATokenToKVPool subclasses MLATokenToKVPool: its primary "kv" pool IS a real
    layer-first MLA latent (kv_buffer + kv_cache_dim), so the MLA arithmetic below
    expresses it device-direct exactly like a dense MLA pool. The DSA indexer
    sidecar (index_k_with_scale_buffer) is a SEPARATE, smaller buffer that does NOT
    go device-direct here — it rides the host v2 path, driven by the hybrid
    controller (increment 2.5), not this module. So a use_dsa pool is supported for
    its main latent; the sidecar's coexistence is the controller's concern.

    Increment 1 declined use_dsa pools outright (the sidecar had no home yet);
    increment 2.5 gives the sidecar the host v2 path, so the main latent is now
    expressible. The hybrid controller's own capability gate additionally requires
    the backend's v2-device ABI before enabling DSA bypass.
    """
    return _is_mla(pool) or _is_mha(pool)


def get_device_page_buffer_meta(pool, indices) -> Tuple[List[List[int]], List[List[int]]]:
    """Scatter-gather device page meta, parallel in shape to the host
    get_page_buffer_meta (one entry per page sub-object, k then v for MHA; k only
    for MLA), but each entry is a LIST of per-layer (ptr)/(size) segments.

    Returns (seg_ptrs, seg_sizes), each a list of length ``n_pages * sub`` whose
    element ``[p * sub + j]`` is the per-layer segment list of page ``p``'s
    sub-object ``j``.
    """
    page_size = pool.page_size
    idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    assert len(idx) % page_size == 0, (
        f"device page meta needs page-aligned indices, got {len(idx)} "
        f"(page_size={page_size})"
    )
    n_pages = len(idx) // page_size

    seg_ptrs: List[List[int]] = []
    seg_sizes: List[List[int]] = []

    if _is_mla(pool):
        bases, token_stride = _mla_layer_bases(pool)
        seg_len = page_size * token_stride
        for p in range(n_pages):
            slot = idx[p * page_size]
            off = slot * token_stride
            seg_ptrs.append([b + off for b in bases])
            seg_sizes.append([seg_len] * len(bases))
    elif _is_mha(pool):
        k_bases, k_stride, v_bases, v_stride = _mha_layer_bases(pool)
        k_len = page_size * k_stride
        v_len = page_size * v_stride
        for p in range(n_pages):
            slot = idx[p * page_size]
            seg_ptrs.append([b + slot * k_stride for b in k_bases])
            seg_sizes.append([k_len] * len(k_bases))
            seg_ptrs.append([b + slot * v_stride for b in v_bases])
            seg_sizes.append([v_len] * len(v_bases))
    else:
        raise ValueError(
            f"get_device_page_buffer_meta: unsupported GPU pool {type(pool).__name__}"
        )

    return seg_ptrs, seg_sizes


def device_pool_regions(pool) -> List[Tuple[int, int]]:
    """(base, nbytes) of every per-layer device buffer, for RDMA registration."""
    regions: List[Tuple[int, int]] = []
    if _is_mla(pool):
        tensors = list(pool.kv_buffer)
    elif _is_mha(pool):
        tensors = list(pool.k_buffer) + list(pool.v_buffer)
    else:
        return regions
    for t in tensors:
        regions.append((int(t.data_ptr()), int(t.numel()) * int(t.element_size())))
    return regions


# --- DSA indexer sidecar (device-direct), increment 2.5 -> task 4 -------------
#
# The DSA indexer sidecar (index_k_with_scale_buffer) is ALSO layer-first: a list
# of ``layer_num`` per-layer 2D tensors (page_num, page_bytes), uint8. Unlike the
# main MLA latent, which is TOKEN-indexed (row = slot), the indexer is PAGE-indexed
# (row = slot // page_size holds the whole page's index+scale for that layer). So a
# page's sidecar is ``layer_num`` per-layer segments, each one contiguous page-row —
# the SAME scatter-gather shape as the main latent (layer_num segments per page), so
# it needs the same @sg chunking on a narrow HCA. It is NOT single-segment. Task 4
# gives it a GPUDirect MR (sidecar_device_pool_regions) so the indexer RDMAs
# straight from/into its GPU buffer, eliminating the last host-staging residual of
# increment 2.5.


def _is_indexer(pool) -> bool:
    layers = getattr(pool, "index_k_with_scale_buffer", None)
    return (
        isinstance(layers, (list, tuple))
        and len(layers) > 0
        and hasattr(pool, "page_size")
    )


def sidecar_supported(pool) -> bool:
    """True if the DSA indexer sidecar's device pool can be expressed as layer-first
    page-indexed scatter-gather segments (index_k_with_scale_buffer present). A
    sidecar shape this module cannot express keeps the stock host v2 path (the
    hybrid controller declines DSA bypass rather than silently drop the sidecar)."""
    return _is_indexer(pool)


def sidecar_device_pool_regions(pool) -> List[Tuple[int, int]]:
    """(base, nbytes) of every per-layer DSA indexer device buffer, for RDMA
    registration (task 4: the sidecar gets its own GPUDirect MR)."""
    regions: List[Tuple[int, int]] = []
    for t in getattr(pool, "index_k_with_scale_buffer", None) or []:
        regions.append((int(t.data_ptr()), int(t.numel()) * int(t.element_size())))
    return regions


def get_device_sidecar_page_buffer_meta(
    pool, indices
) -> Tuple[List[List[int]], List[List[int]]]:
    """Scatter-gather device page meta for the DSA indexer sidecar, parallel in
    shape to get_device_page_buffer_meta but PAGE-indexed and single-object per page
    (sub=1). Given the SAME token indices as the main KV (the indexer rides the KV
    page slots), each page becomes one per-layer segment list: layer L contributes
    the contiguous page-row (slot // page_size) of its buffer.

    Returns (seg_ptrs, seg_sizes), each a list of length ``n_pages`` whose element
    ``[p]`` is page ``p``'s per-layer segment list. The concatenation is LAYER-major
    (layer0[page], layer1[page], ...), matching the main-latent write/read ordering,
    so a page written device-direct reads back byte-identical (proven by the dfkv
    sidecar-device roundtrip test)."""
    page_size = pool.page_size
    idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    assert len(idx) % page_size == 0, (
        f"sidecar device page meta needs page-aligned indices, got {len(idx)} "
        f"(page_size={page_size})"
    )
    n_pages = len(idx) // page_size

    layers = pool.index_k_with_scale_buffer
    itemsize = layers[0].element_size()
    # One page-row's payload bytes and its row-to-row stride (equal for a
    # contiguous (page_num, page_bytes) buffer; use stride to stay exact).
    row_bytes = int(layers[0].shape[1]) * itemsize
    row_stride = int(layers[0].stride(0)) * itemsize
    bases = [int(t.data_ptr()) for t in layers]

    seg_ptrs: List[List[int]] = []
    seg_sizes: List[List[int]] = []
    for p in range(n_pages):
        slot = idx[p * page_size]
        off = (slot // page_size) * row_stride
        seg_ptrs.append([b + off for b in bases])
        seg_sizes.append([row_bytes] * len(bases))
    return seg_ptrs, seg_sizes
