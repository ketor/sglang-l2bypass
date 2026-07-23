from __future__ import annotations

import json
import logging
import os
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation
from sglang.srt.managers.cache_controller import (
    DeviceLoadTask,
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.device_page_meta import consecutive_ok_pages
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()


class CacheOperation(BaseCacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.pool_transfers = pool_transfers

    @staticmethod
    def merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[list[PoolTransfer]]:
        grouped: dict[tuple[PoolName, Optional[PoolName]], list[PoolTransfer]] = {}
        for op in ops:
            for t in op.pool_transfers or []:
                grouped.setdefault((t.name, t.indices_from_pool), []).append(t)
        if not grouped:
            return None

        def cat_or_none(tensors):
            parts = [x for x in tensors if x is not None]
            return torch.cat(parts) if parts else None

        return [
            PoolTransfer(
                name=ts[0].name,
                host_indices=cat_or_none(t.host_indices for t in ts),
                device_indices=cat_or_none(t.device_indices for t in ts),
                keys=[k for t in ts if t.keys for k in t.keys] or None,
                hit_policy=ts[0].hit_policy,
                indices_from_pool=ts[0].indices_from_pool,
            )
            for ts in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation.merge_pool_transfers(ops),
        )
        merged.node_ids = node_ids
        return merged


class StorageOperation(BaseStorageOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()
        super().__init__(
            host_indices,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        self.pool_transfers_done = not bool(pool_transfers)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
    ):
        startup_storage_backend = storage_backend
        self.extra_host_mem_release_queues: dict[PoolName, Queue[torch.Tensor]] = {}
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=pp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def _start_storage_threads(self):
        super()._start_storage_threads()
        self._init_extra_host_mem_release_queues()

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        for entry in host_pools or []:
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    # --- L2-bypass (increment 2.5: DSA main-KV device-direct + sidecar host) ---
    def _sidecar_entries(self) -> list[PoolEntry]:
        """Non-anchor host pool entries (the DSA indexer sidecar). Ordered as the
        host_pool_group registered them so keys/indices stay aligned."""
        anchor = self.mem_pool_host.anchor_entry
        return [
            e
            for e in self.mem_pool_host.entries
            if e is not anchor and not e.is_primary_index_anchor
        ]

    def _bypass_sidecar_supported(self) -> bool:
        """DSA bypass is enabled only for the anchor+INDEXER shape: every sidecar
        reuses the KV page indices (indices_from_pool=KV, ALL_PAGES) so the main-KV
        device slots address the sidecar too. SWA/Mamba (trailing-page states with
        their own indices) are NOT device-direct here — they keep the stock host
        path. Enforced by requiring exactly the DSA sidecars (INDEXER-like)."""
        sidecars = self._sidecar_entries()
        if not sidecars:
            return False
        return all(e.name == PoolName.INDEXER for e in sidecars)

    def _maybe_enable_l2_bypass(self) -> None:
        """Hybrid gate: on top of the base requirements (flag, supports_device_
        transfer, zero-copy v1 write surface for the main KV, per-pool device meta,
        register_mem_pool_device), DSA bypass ALSO needs the backend's v2-device
        split-value ABI and the anchor+INDEXER pool shape. Any miss -> stock host
        path with a clear warning."""
        self.l2_bypass = False
        if not self.l2_bypass_requested:
            return
        if not (
            callable(getattr(self.storage_backend, "batch_set_v2_device", None))
            and callable(getattr(self.storage_backend, "batch_get_v2_device", None))
        ):
            logger.warning(
                "SGLANG_HICACHE_L2_BYPASS=1 but backend %r lacks the v2-device "
                "split-value ABI (batch_set_v2_device/batch_get_v2_device); DSA "
                "bypass off, using the stock host path.",
                self.storage_backend_type,
            )
            return
        if not self._bypass_sidecar_supported():
            logger.warning(
                "SGLANG_HICACHE_L2_BYPASS=1 but the hybrid host pools are not the "
                "DSA anchor+INDEXER shape (sidecars=%s); DSA bypass off, using the "
                "stock host path.",
                [str(e.name) for e in self._sidecar_entries()],
            )
            return
        # Base gate does the rest (device meta, register_mem_pool_device, flip
        # l2_bypass + page_set_func). The hybrid _page_backup overrides the write
        # dispatch, so page_set_func is unused here, but keep the base contract.
        super()._maybe_enable_l2_bypass()
        if self.l2_bypass:
            logger.info(
                "HiCache DSA L2-bypass ENABLED: main MLA latent RDMAs straight "
                "from/to GPU slots (device-direct SG); the DSA indexer sidecar "
                "rides the host v2 path. Backend=%r.",
                self.storage_backend_type,
            )

    @staticmethod
    def parse_storage_backend_extra_config(
        storage_backend_extra_config: Optional[str],
    ) -> tuple[dict, int, float, float, bool]:
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
                path = storage_backend_extra_config[1:]
                ext = os.path.splitext(path)[1].lower()
                with open(path, "rb" if ext == ".toml" else "r") as f:
                    if ext == ".json":
                        extra_config = json.load(f)
                    elif ext == ".toml":
                        import tomllib

                        extra_config = tomllib.load(f)
                    elif ext in (".yaml", ".yml"):
                        import yaml

                        extra_config = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file {path} (config format: {ext})"
                        )
            else:
                extra_config = json.loads(storage_backend_extra_config)

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                "prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        if not self.enable_storage:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False
        if not hasattr(self.storage_backend, "clear"):
            logger.warning(
                "Storage backend %s does not support clear operation.",
                type(self.storage_backend).__name__,
            )
            return False
        self.storage_backend.clear()
        return True

    def _init_extra_host_mem_release_queues(self) -> None:
        self.extra_host_mem_release_queues = {}
        entries = getattr(self.mem_pool_host, "entries", None) or []
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None)
        for entry in entries:
            if entry is anchor_entry or entry.is_primary_index_anchor:
                continue
            self.extra_host_mem_release_queues[entry.name] = Queue()

    def _append_host_mem_release_pages(
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int
    ) -> None:
        if host_indices.numel() == 0:
            return
        for page in host_indices.split(page_size):
            release_queue.put(page)

    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ):
        if host_indices is not None:
            self._append_host_mem_release_pages(
                self.host_mem_release_queue,
                host_indices,
                self.mem_pool_host.page_size,
            )
        for transfer in extra_pools or []:
            if transfer.host_indices is None or transfer.host_indices.numel() == 0:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            if (
                entry is None
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue
            release_queue = self.extra_host_mem_release_queues.get(transfer.name)
            if release_queue is None:
                continue
            self._append_host_mem_release_pages(
                release_queue, transfer.host_indices, entry.host_pool.page_size
            )

    def reset(self):
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            for release_queue in self.extra_host_mem_release_queues.values():
                release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=True,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None

        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        self.start_writing()
        return host_indices

    def write_device(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        sidecar_host_indices: Optional[torch.Tensor] = None,
    ) -> int:
        """DSA L2-bypass write-through: the main MLA latent stays in its GPU slots
        (pinned; RDMA'd device-direct at storage backup). The DSA indexer sidecar is
        D2H'd to the caller-allocated host slots (sidecar_host_indices) in
        start_writing, exactly as the stock hybrid path does — the small sidecar
        keeps the proven host v2 path. The sidecar transfer reuses the main-KV
        device slots (indices_from_pool=KV: the indexer lives at the same token
        slots). No host slot for the main KV. Returns node_id (always succeeds; the
        sidecar host alloc happened in the caller)."""
        sidecars = (
            [
                PoolTransfer(
                    name=e.name,
                    host_indices=sidecar_host_indices,
                    device_indices=device_indices,
                    hit_policy=PoolHitPolicy.ALL_PAGES,
                )
                for e in self._sidecar_entries()
            ]
            if sidecar_host_indices is not None
            else None
        )
        host_placeholder = device_indices.new_empty(0)
        self.write_queue.append(
            CacheOperation(
                host_placeholder,
                device_indices,
                node_id,
                priority,
                pool_transfers=sidecars or None,
            )
        )
        self.start_writing()
        return node_id

    def write_storage_device(
        self,
        device_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        """DSA L2-bypass storage backup: the operation's host_indices field carries
        the main-KV DEVICE slot indices (SG put from GPU); extra_pools carry the
        already-D2H'd sidecar host slots for the host v2 write. _page_backup combines
        both into one batch_set_v2_device call per batch."""
        operation = StorageOperation(
            device_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def start_writing(self) -> None:
        if not self.write_queue:
            return

        if self.l2_bypass:
            # DSA device-direct: NO main-KV D2H (it RDMAs from GPU slots at backup).
            # Only the small DSA indexer sidecar is D2H'd to host here, so the stock
            # v2 write can read it. Anchor indices are empty (skips the anchor
            # backup inside HostPoolGroup.backup_from_device_all_layer); the sidecar
            # transfers carry their own host+device indices.
            op = CacheOperation.merge_ops(self.write_queue)
            _, _, resolved_pool_transfers = self.move_hybrid_indices(op)
            self.write_queue.clear()
            empty = op.device_indices.new_empty(0)
            start_event = device_module.Event()
            finish_event = device_module.Event()
            start_event.record()
            with device_module.stream(self.write_stream):
                start_event.wait(self.write_stream)
                # Sidecar-only D2H (drive the sidecar entries directly; the anchor
                # main KV is NOT copied to host — it RDMAs device-direct at backup).
                for transfer in resolved_pool_transfers or []:
                    entry = self.mem_pool_host.entry_map.get(transfer.name)
                    if entry is None or transfer.host_indices is None:
                        continue
                    entry.host_pool.backup_from_device_all_layer(
                        entry.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        self.io_backend,
                    )
                finish_event.record()
                self._record_transfer_indices_on_stream(
                    self.write_stream,
                    empty,
                    op.device_indices,
                    resolved_pool_transfers,
                )
            self.ack_write_queue.append(
                HiCacheAck(start_event, finish_event, op.node_ids)
            )
            return

        op = CacheOperation.merge_ops(self.write_queue)
        # Page-first write-back JIT kernels can keep destination host indices on CPU.
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            host_indices = op.host_indices
            device_indices = op.device_indices
            resolved_pool_transfers = op.pool_transfers
        else:
            host_indices, device_indices, resolved_pool_transfers = (
                self.move_hybrid_indices(op)
            )
        self.write_queue.clear()
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                self.io_backend,
                pool_transfers=resolved_pool_transfers,
            )
            if self.has_draft and host_indices.numel() > 0:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            self._record_transfer_indices_on_stream(
                self.write_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        need_load_kv = host_indices.numel() > 0

        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if not need_load_kv:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None

        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=False,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                full_allocator.free(device_indices)
            return None

        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def load_device_direct(
        self, hash_values: List[str], node_id: int = -1
    ) -> tuple[Optional[torch.Tensor], int]:
        """DSA L2-bypass on-demand load: allocate GPU KV slots for the marker span,
        RDMA the main MLA latent straight into them AND read the DSA indexer sidecar
        into a transient host staging buffer (one batch_get_v2_device call), then
        H2D the sidecar into its device index buffer at the SAME GPU slots. On
        return the GPU slots hold both the latent (SG GET) and the indexer (H2D).

        Returns (device_indices, ok_pages) like the dense variant: ok_pages is the
        consecutive prefix where BOTH the main KV and every sidecar page hit (the
        first miss truncates). (None, 0) if the device allocation failed."""
        npages = len(hash_values)
        total_tokens = npages * self.page_size
        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        device_indices = full_allocator.alloc(total_tokens)
        if device_indices is None:
            return None, 0
        # Transient host staging for the sidecar (freed after H2D). The indexer
        # rides the KV page indices, so the sidecar device slots == the main-KV
        # device slots just allocated.
        side_host = self.mem_pool_host.alloc(total_tokens)
        if side_host is None:
            full_allocator.free(device_indices)
            return None, 0
        sidecars = [
            PoolTransfer(
                name=e.name,
                host_indices=side_host,
                device_indices=device_indices,
                keys=list(hash_values),
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
            for e in self._sidecar_entries()
        ]
        try:
            results = self.storage_backend.batch_get_v2_device(
                hash_values, device_indices, sidecars
            )
            kv_ok = results.get("kv") or []
            # A page is usable only if the main KV AND every sidecar hit it.
            ok_pages = 0
            for p in range(npages):
                if p >= len(kv_ok) or not kv_ok[p]:
                    break
                if not all(
                    (results.get(str(e.name)) or [False] * npages)[p]
                    for e in self._sidecar_entries()
                ):
                    break
                ok_pages += 1
            if ok_pages > 0:
                # H2D the verified sidecar prefix into the device index buffer at the
                # main-KV slots, then it is device-resident (staging freed below).
                self._sidecar_h2d(side_host, device_indices, sidecars, ok_pages)
        finally:
            # The staging host slots have served their purpose (sidecar now on
            # device for the hit prefix); the main KV never touched host.
            self.mem_pool_host.free(side_host)
        return device_indices, ok_pages

    def _sidecar_h2d(
        self,
        side_host: torch.Tensor,
        device_indices: torch.Tensor,
        sidecars: list[PoolTransfer],
        ok_pages: int,
    ) -> None:
        """H2D the sidecar's verified prefix from the host staging slots into its
        device index buffer (at the main-KV device slots). Runs on the load stream
        and synchronizes before returning, so the staging host slots can be freed
        and the index is present for the compute fence recorded in start_loading."""
        n_tokens = ok_pages * self.page_size
        prefix_op = CacheOperation(
            device_indices[:0],  # empty anchor host -> anchor H2D skipped
            device_indices[:0],  # empty anchor device
            -1,
            pool_transfers=[
                PoolTransfer(
                    name=tr.name,
                    host_indices=tr.host_indices[:n_tokens],
                    device_indices=tr.device_indices[:n_tokens],
                    hit_policy=tr.hit_policy,
                )
                for tr in sidecars
            ],
        )
        host_moved, device_moved, resolved = self.move_hybrid_indices(prefix_op)
        with device_module.stream(self.load_stream):
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_moved,
                    device_moved,
                    i,
                    self.io_backend,
                    pool_transfers=resolved,
                )
        self.load_stream.synchronize()

    def enqueue_device_load(
        self, device_indices: torch.Tensor, node_ids: List[int]
    ) -> None:
        """DSA L2-bypass: queue an already-loaded device span for start_loading()'s
        fence pass (the RDMA GET + sidecar H2D already filled the slots in
        load_device_direct). Uses the hybrid CacheOperation so merge_ops' pool-
        transfer handling stays valid (no extra pools on this fence op)."""
        host_placeholder = device_indices.new_empty(0)
        op = CacheOperation(host_placeholder, device_indices, -1)
        op.node_ids = list(node_ids)
        self.load_queue.append(op)

    # ---- Increment 3: async device-direct read (DSA overrides) -----------------
    #
    # Split the synchronous load_device_direct across the thread boundary exactly
    # like the dense base, but the DSA read is a split value: the background thread
    # RDMAs the main MLA latent into the GPU slots AND reads the indexer sidecar
    # into a transient HOST staging span (batch_get_v2_device — pure C/RDMA + host
    # memcpy, no CUDA). The sidecar H2D (a CUDA op) is deferred to the scheduler
    # thread in finalize_device_load, keeping all CUDA + collectives off the
    # background thread.

    def _full_allocator(self):
        return getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )

    def make_device_load_task(
        self, hash_values: List[str]
    ) -> Optional[DeviceLoadTask]:
        """Alloc GPU main-KV slots + transient sidecar host staging + build the
        sidecar PoolTransfers (scheduler thread). None if either alloc fails (the
        caller evicts + retries). No GET, no CUDA."""
        total_tokens = len(hash_values) * self.page_size
        full_allocator = self._full_allocator()
        device_indices = full_allocator.alloc(total_tokens)
        if device_indices is None:
            return None
        side_host = self.mem_pool_host.alloc(total_tokens)
        if side_host is None:
            full_allocator.free(device_indices)
            return None
        sidecars = [
            PoolTransfer(
                name=e.name,
                host_indices=side_host,
                device_indices=device_indices,
                keys=list(hash_values),
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
            for e in self._sidecar_entries()
        ]
        return DeviceLoadTask(
            hash_values, device_indices, sidecars=sidecars, side_host=side_host
        )

    def _run_device_get(self, task: DeviceLoadTask) -> int:
        """Background thread: batch_get_v2_device fills the main KV via RDMA (device
        slots) AND the indexer into task.side_host (host staging). Count the prefix
        where the main KV AND every sidecar page hit. NO H2D here (deferred to
        finalize on the scheduler thread)."""
        results = self.storage_backend.batch_get_v2_device(
            task.hash_values, task.device_indices, task.sidecars
        )
        kv_ok = results.get("kv") or []
        sidecar_oks = [
            results.get(str(e.name)) or [] for e in self._sidecar_entries()
        ]
        return consecutive_ok_pages(kv_ok, sidecar_oks, len(task.hash_values))

    def finalize_device_load(self, task: DeviceLoadTask, ok_pages: int) -> None:
        """Scheduler thread, at promotion: H2D the verified sidecar prefix from the
        host staging into its device index buffer (at the main-KV slots), then free
        the staging. After this the GPU slots hold both latent and indexer."""
        if ok_pages > 0 and task.side_host is not None:
            self._sidecar_h2d(
                task.side_host, task.device_indices, task.sidecars, ok_pages
            )
        if task.side_host is not None:
            self.mem_pool_host.free(task.side_host)
            task.side_host = None

    def free_device_load(self, task: DeviceLoadTask) -> None:
        """Abort path (scheduler thread): free the main-KV GPU slots + the sidecar
        host staging. Idempotent."""
        if task.device_indices is not None:
            self._full_allocator().free(task.device_indices)
            task.device_indices = None
        if task.side_host is not None:
            self.mem_pool_host.free(task.side_host)
            task.side_host = None

    def free_device_indices(self, device_indices: torch.Tensor) -> None:
        """Free a span of main-KV device slots via the full-attn allocator that owns
        them (the tree uses this to release an unverified load suffix)."""
        self._full_allocator().free(device_indices)

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1

        if self.l2_bypass:
            # DSA device-direct: the main KV (SG GET) and the sidecar (H2D) already
            # filled the GPU slots in load_device_direct. Record ONE fence: mark
            # every layer event complete on the load stream so the compute path's
            # per-layer wait_until(i) orders after the device writes. No per-layer
            # H2D to stream (a single RDMA op + one synced H2D filled all layers).
            producer_id = self.layer_done_counter.update_producer()
            op = CacheOperation.merge_ops(self.load_queue)
            self.load_queue.clear()
            producer_event = self.layer_done_counter.events[producer_id]
            producer_event.start_event.record()
            with device_module.stream(self.load_stream):
                producer_event.start_event.wait(self.load_stream)
                for i in range(self.layer_num):
                    producer_event.complete(i)
            self.ack_load_queue.append(
                HiCacheAck(
                    producer_event.start_event,
                    producer_event.finish_event,
                    op.node_ids,
                )
            )
            return producer_id

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                    pool_transfers=resolved_pool_transfers,
                )
                if (
                    self.has_draft
                    and host_indices.numel() > 0
                    and i < self.mem_pool_host_draft.layer_num
                ):
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            self._record_transfer_indices_on_stream(
                self.load_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_load_queue.append(
            HiCacheAck(
                producer_event.start_event,
                producer_event.finish_event,
                op.node_ids,
            )
        )
        return producer_id

    def _record_transfer_indices_on_stream(
        self,
        stream: torch.Stream,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> None:
        if host_indices.is_cuda:
            host_indices.record_stream(stream)
        if device_indices.is_cuda:
            device_indices.record_stream(stream)
        for transfer in pool_transfers or []:
            if transfer.host_indices is not None and transfer.host_indices.is_cuda:
                transfer.host_indices.record_stream(stream)
            if transfer.device_indices is not None and transfer.device_indices.is_cuda:
                transfer.device_indices.record_stream(stream)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        hash_value = self.get_hash_str(
            operation.token_ids, operation.last_hash, page_size=self.page_size
        )

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None
        )
        if operation.pool_transfers:
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _page_transfer(self, operation):
        # KV pools first — determines actual completed page count
        super()._page_transfer(operation)

        # Extra pools only after KV fully completes. If KV terminated early
        # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
        # data misalignment.
        kv_completed_pages = operation.completed_tokens // self.page_size
        if operation.pool_transfers and kv_completed_pages == len(operation.hash_value):
            self._sync_trailing_keys(
                operation.pool_transfers, operation.hash_value, kv_completed_pages
            )
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_get_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)
        operation.pool_transfers_done = True

    def _page_backup(self, operation):
        if self.l2_bypass:
            self._page_backup_device(operation)
            return
        # Backup extra pools
        if operation.pool_transfers:
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        # Backup kv pools
        super()._page_backup(operation)

    def _page_backup_device(self, operation):
        """DSA L2-bypass backup: one batch_set_v2_device call per batch writes the
        main MLA latent device-direct (operation.host_indices carries the GPU slot
        indices) AND the DSA indexer sidecar from its host slots (operation.pool_
        transfers, already D2H'd in start_writing). Split value: main KV under the
        @sg-chunked v1-style 'kv' keys, sidecar under its own '_indexer_k' keys —
        two namespaces, no collision, isolated by model_hash."""
        prefix_keys = operation.prefix_keys
        page_size = self.page_size
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            n = len(batch_hashes)
            batch_kv_device = operation.host_indices[
                i * page_size : (i + n) * page_size
            ]
            batch_sidecars = []
            for tr in operation.pool_transfers or []:
                bh = (
                    tr.host_indices[i * page_size : (i + n) * page_size]
                    if tr.host_indices is not None
                    else None
                )
                batch_sidecars.append(
                    PoolTransfer(
                        name=tr.name,
                        host_indices=bh,
                        keys=batch_hashes,
                        hit_policy=tr.hit_policy,
                    )
                )
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            results = self.storage_backend.batch_set_v2_device(
                batch_hashes, batch_kv_device, batch_sidecars, extra_info
            )
            kv_ok = results.get("kv") or []
            if not (len(kv_ok) == n and all(kv_ok)):
                logger.warning(
                    "DSA device-direct backup: %d main-KV pages failed.", n
                )
                break
            operation.pool_storage_result.update_extra_pool_hit_pages(
                {k: v for k, v in results.items() if k != "kv"}
            )
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += page_size * n

    def _resolve_sidecar_derived_pool_transfers(self, operation):
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                continue
            if transfer.indices_from_pool != PoolName.KV:
                source = next(
                    (
                        t
                        for t in operation.pool_transfers
                        if t.indices_from_pool is None
                        and t.name == transfer.indices_from_pool
                    ),
                    None,
                )
                if source is None:
                    raise AssertionError(
                        "Storage sidecar derived pool source missing: "
                        f"{transfer.name} from {transfer.indices_from_pool}."
                    )
                transfer.host_indices = source.host_indices
                if transfer.keys is None:
                    transfer.keys = source.keys
            else:
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """Re-align trailing-page sidecar keys after KV hit truncation.

        When the storage hit is shorter than the original target prefix, each
        pool transfer's keys must be updated to the last N hashes of the actual
        hit range instead of the last N hashes of the original target range.
        For mamba (N=1) this is just the last hit page hash; for SWA (N>1) it
        is a sliding window of the last N hit pages.
        """
        for transfer in pool_transfers:
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                continue
            trailing_n = len(transfer.keys) if transfer.keys else 1
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]

    def _resolve_pool_transfers_allocation(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Auto-alloc host or device indices for PoolTransfers where they are None."""
        if not extra_pools:
            return None
        # (pool, free_fn, indices) for atomic rollback on failure.
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                if alloc_host:
                    prev_pool.host_indices = None
                else:
                    prev_pool.device_indices = None

        for pool in extra_pools:
            if pool.indices_from_pool is not None:
                derived_transfers.append(pool)
                continue
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                continue
            if alloc_host:
                if pool.host_indices is not None or pool.device_indices is None:
                    continue
                alloc_fn = entry.host_pool.alloc
                free_fn = entry.host_pool.free
                evict_fn = entry.host_evict_fn
                size = len(pool.device_indices)
            else:
                if pool.device_indices is not None or pool.host_indices is None:
                    continue
                # device_alloc_fn / device_free_fn override entry.device_pool's
                # methods for pools whose device_pool is a raw KV pool (layout)
                # rather than an allocator (e.g. SWA).
                alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
                free_fn = entry.device_free_fn or entry.device_pool.free
                evict_fn = entry.device_evict_fn
                size = len(pool.host_indices)
            indices = alloc_fn(size)
            if indices is None and evict_fn:
                evict_fn(size)
                indices = alloc_fn(size)
            if indices is None:
                # Atomic rollback: free everything we successfully allocated.
                rollback_allocated()
                return None
            if alloc_host:
                pool.host_indices = indices
            else:
                pool.device_indices = indices
            newly_allocated.append((pool, free_fn, indices))

        # Assign indices to deferred pools from their source.
        for pool in derived_transfers:
            if pool.indices_from_pool == PoolName.KV:
                pool.host_indices = kv_host_indices
                pool.device_indices = kv_device_indices
                continue

            source = next(
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None
                    and transfer.name == pool.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback_allocated()
                return None
            pool.host_indices = source.host_indices
            pool.device_indices = source.device_indices
        return extra_pools
