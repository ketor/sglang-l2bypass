"""GPU/torch-free unit tests for device_page_meta (HiCache L2-bypass write path).

Validates the layer-first scatter-gather pointer arithmetic against hand-computed
offsets for both MLA (single object/page) and MHA (k+v/page) GPU pools, the
page-alignment assert, DSA decline, and the region enumeration used for RDMA
registration. Pure python: run with `python3 test_device_page_meta.py`.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mem_cache import device_page_meta as dpm  # noqa: E402


class FakeTensor:
    """Minimal stand-in for a torch tensor exposing the 3 accessors dpm uses."""

    def __init__(self, base, numel, itemsize):
        self._base = base
        self._numel = numel
        self._itemsize = itemsize

    def data_ptr(self):
        return self._base

    def numel(self):
        return self._numel

    def element_size(self):
        return self._itemsize


class FakeMlaPool:
    """Layer-first MLA GPU pool: kv_buffer[L] is (size+page_size, 1, kv_cache_dim)."""

    def __init__(self, layer_num=4, size=256, page_size=64, kv_cache_dim=576, itemsize=2):
        self.page_size = page_size
        self.kv_cache_dim = kv_cache_dim
        self.use_dsa = False
        slots = size + page_size
        self.itemsize = itemsize
        # Distinct, non-overlapping fake bases, one per layer.
        self.kv_buffer = [
            FakeTensor(0x100000 * (L + 1), slots * kv_cache_dim, itemsize)
            for L in range(layer_num)
        ]


class FakeMhaPool:
    """Layer-first MHA GPU pool: separate k_buffer/v_buffer lists per layer."""

    def __init__(self, layer_num=3, size=128, page_size=64, head_num=8, head_dim=128,
                 v_head_dim=128, itemsize=2):
        self.page_size = page_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.itemsize = itemsize
        slots = size + page_size
        self.k_buffer = [
            FakeTensor(0x1000000 * (L + 1), slots * head_num * head_dim, itemsize)
            for L in range(layer_num)
        ]
        self.v_buffer = [
            FakeTensor(0x9000000 * (L + 1), slots * head_num * v_head_dim, itemsize)
            for L in range(layer_num)
        ]


class FakeIndexerTensor:
    """Stand-in for a layer-first, page-indexed indexer buffer (page_num, page_bytes),
    exposing shape/stride(0) alongside data_ptr/numel/element_size."""

    def __init__(self, base, page_num, page_bytes, itemsize=1):
        self._base = base
        self._page_num = page_num
        self._page_bytes = page_bytes
        self._itemsize = itemsize

    def data_ptr(self):
        return self._base

    def numel(self):
        return self._page_num * self._page_bytes

    def element_size(self):
        return self._itemsize

    @property
    def shape(self):
        return (self._page_num, self._page_bytes)

    def stride(self, dim=None):
        strides = (self._page_bytes, 1)  # contiguous (page_num, page_bytes)
        return strides if dim is None else strides[dim]


class FakeIndexerPool:
    """DSA indexer device pool: index_k_with_scale_buffer[L] is (page_num, page_bytes)."""

    def __init__(self, layer_num=4, page_num=8, page_size=64, page_bytes=132):
        self.page_size = page_size
        self.index_k_with_scale_buffer = [
            FakeIndexerTensor(0x200000 * (L + 1), page_num, page_bytes)
            for L in range(layer_num)
        ]


class TestDeviceSidecarPageMeta(unittest.TestCase):
    def test_sidecar_supported(self):
        self.assertTrue(dpm.sidecar_supported(FakeIndexerPool()))
        # A plain MLA/MHA main pool has no indexer buffer -> not a sidecar pool.
        self.assertFalse(dpm.sidecar_supported(FakeMlaPool()))

    def test_sidecar_page_indexed_layer_segments(self):
        pool = FakeIndexerPool(layer_num=4, page_num=8, page_size=64, page_bytes=132)
        # Two pages at slots 0 (row 0) and 128 (row 2 = 128 // 64).
        indices = list(range(0, 64)) + list(range(128, 192))
        seg_ptrs, seg_sizes = dpm.get_device_sidecar_page_buffer_meta(pool, indices)
        self.assertEqual(len(seg_ptrs), 2)  # sub=1, page-indexed
        for p, page_row in enumerate((0, 2)):
            self.assertEqual(len(seg_ptrs[p]), 4)      # one segment per layer
            self.assertEqual(seg_sizes[p], [132] * 4)  # one page-row payload/layer
            for L in range(4):
                base = 0x200000 * (L + 1)
                self.assertEqual(seg_ptrs[p][L], base + page_row * 132)

    def test_sidecar_page_alignment_assert(self):
        with self.assertRaises(AssertionError):
            dpm.get_device_sidecar_page_buffer_meta(
                FakeIndexerPool(page_size=64), list(range(0, 63)))

    def test_sidecar_device_pool_regions(self):
        pool = FakeIndexerPool(layer_num=3, page_num=8, page_bytes=132)
        regions = dpm.sidecar_device_pool_regions(pool)
        self.assertEqual(len(regions), 3)
        for L, (base, size) in enumerate(regions):
            self.assertEqual(base, 0x200000 * (L + 1))
            self.assertEqual(size, 8 * 132)


class TestDevicePageMeta(unittest.TestCase):
    def test_mla_single_object_per_page(self):
        pool = FakeMlaPool(layer_num=4, page_size=64, kv_cache_dim=576, itemsize=2)
        # Two pages: slots [0..64) and [128..192).
        indices = list(range(0, 64)) + list(range(128, 192))
        seg_ptrs, seg_sizes = dpm.get_device_page_buffer_meta(pool, indices)

        self.assertEqual(len(seg_ptrs), 2)  # sub=1 (MLA) * 2 pages
        token_stride = 576 * 2
        seg_len = 64 * token_stride
        for p, slot0 in enumerate((0, 128)):
            self.assertEqual(len(seg_ptrs[p]), 4)  # one segment per layer
            self.assertEqual(seg_sizes[p], [seg_len] * 4)
            for L in range(4):
                base = 0x100000 * (L + 1)
                self.assertEqual(seg_ptrs[p][L], base + slot0 * token_stride)

    def test_mha_k_then_v_per_page(self):
        pool = FakeMhaPool(layer_num=3, page_size=64, head_num=8, head_dim=128)
        indices = list(range(64, 128))  # one page at slot 64
        seg_ptrs, seg_sizes = dpm.get_device_page_buffer_meta(pool, indices)

        self.assertEqual(len(seg_ptrs), 2)  # sub=2 (k,v) * 1 page
        k_stride = 8 * 128 * 2
        v_stride = 8 * 128 * 2
        # entry 0 = k, entry 1 = v
        self.assertEqual(seg_sizes[0], [64 * k_stride] * 3)
        self.assertEqual(seg_sizes[1], [64 * v_stride] * 3)
        for L in range(3):
            self.assertEqual(seg_ptrs[0][L], 0x1000000 * (L + 1) + 64 * k_stride)
            self.assertEqual(seg_ptrs[1][L], 0x9000000 * (L + 1) + 64 * v_stride)

    def test_page_alignment_assert(self):
        pool = FakeMlaPool(page_size=64)
        with self.assertRaises(AssertionError):
            dpm.get_device_page_buffer_meta(pool, list(range(0, 63)))  # not a page

    def test_supported_including_dsa_main_latent(self):
        # Increment 2.5 lifted the increment-1 use_dsa veto: a DSA pool's MAIN latent
        # (kv_buffer + kv_cache_dim) is a real layer-first MLA buffer and IS
        # expressible device-direct here; the indexer sidecar is the hybrid
        # controller's concern, not this module's. So supported() is True for MLA,
        # MHA, AND a DSA (use_dsa) MLA-shaped pool.
        self.assertTrue(dpm.supported(FakeMlaPool()))
        self.assertTrue(dpm.supported(FakeMhaPool()))
        dsa = FakeMlaPool()
        dsa.use_dsa = True
        self.assertTrue(dpm.supported(dsa))

    def test_device_pool_regions(self):
        pool = FakeMlaPool(layer_num=4, size=256, page_size=64, kv_cache_dim=576,
                           itemsize=2)
        regions = dpm.device_pool_regions(pool)
        self.assertEqual(len(regions), 4)
        nbytes = (256 + 64) * 576 * 2
        for L, (base, size) in enumerate(regions):
            self.assertEqual(base, 0x100000 * (L + 1))
            self.assertEqual(size, nbytes)
        # MHA enumerates k then v (2 * layer_num regions).
        self.assertEqual(len(dpm.device_pool_regions(FakeMhaPool(layer_num=3))), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
