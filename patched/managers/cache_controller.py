from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.pool_host import HostKVCache

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.device_page_meta import consecutive_ok_pages
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


def env_l2_bypass() -> bool:
    """Prototype flag: SGLANG_HICACHE_L2_BYPASS=1 requests the device-direct
    (L2-bypass) write path. Read once at controller/cache init; effective only if
    the storage backend also advertises supports_device_transfer()."""
    return os.environ.get("SGLANG_HICACHE_L2_BYPASS", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def env_l2_bypass_sync_read() -> bool:
    """Escape hatch: SGLANG_HICACHE_L2_BYPASS_SYNC_READ=1 reverts the L2-bypass
    READ path to the increment-2 synchronous form (the on-demand SG GET runs on the
    scheduler thread inside init_load_back). Default OFF => the increment-3 async
    read path (background device-load thread; check_prefetch_progress parks the
    request until every rank's local GET is done). Only meaningful when
    SGLANG_HICACHE_L2_BYPASS=1; ignored otherwise. Provided for A/B and safety."""
    return os.environ.get(
        "SGLANG_HICACHE_L2_BYPASS_SYNC_READ", "0"
    ).strip().lower() not in ("", "0", "false", "no", "off")


class DeviceLoadTask:
    """L2-bypass async read (increment 3): one on-demand device-direct load handed
    to the background device-load thread.

    Division of labor (the TP-safety contract): the SCHEDULER thread allocates the
    GPU slots (and, for DSA, the transient sidecar host staging) and builds this
    task; the BACKGROUND thread runs ONLY the blocking backend GET
    (batch_get_v1_device / batch_get_v2_device) plus the local per-rank page
    verification — NO CUDA stream ops, NO collectives. The scheduler thread then
    does the TP MIN all_reduces, the sidecar H2D (DSA), and the radix-tree
    promotion. `done` is a plain threading.Event; check_prefetch_progress polls it
    (via a per-round TP MIN reduce over every rank's 0/1 done flag), so ranks whose
    background GET finishes at different wall-clock times still run an identical,
    balanced collective sequence."""

    def __init__(self, hash_values, device_indices, sidecars=None, side_host=None):
        self.hash_values = hash_values
        self.device_indices = device_indices
        # DSA/hybrid only: the sidecar PoolTransfers (indexer -> side_host staging)
        # and the staging tensor to free after the H2D. None for dense bypass.
        self.sidecars = sidecars
        self.side_host = side_host
        self.ok_pages = 0
        self.error = None
        self.done = threading.Event()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: StorageOperation):
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)

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


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
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
        enable_storage_metrics: bool = False,
    ):
        self.tp_group = tp_group
        self.attn_cp_group = attn_cp_group
        self.attn_tp_group = attn_tp_group
        self.pp_group = pp_group
        self.prefetch_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage_metrics = enable_storage_metrics

        # L2-bypass (device-direct write) prototype. `requested` reflects the env
        # flag; `l2_bypass` is only turned on at attach once the backend advertises
        # supports_device_transfer(). Off => byte-identical to stock.
        self.l2_bypass_requested = env_l2_bypass()
        self.l2_bypass = False
        # Increment 3: async device-direct read. When l2_bypass is on and this is
        # False (default), the on-demand read runs on a background thread; True
        # keeps the increment-2 synchronous read. Inert unless l2_bypass.
        self.l2_bypass_sync_read = env_l2_bypass_sync_read()
        # Background device-load thread + its queue (created in _start_storage_
        # threads only when l2_bypass and async read are on). None otherwise.
        self.device_load_queue: Optional[Queue] = None
        self.device_load_thread: Optional[threading.Thread] = None

        # Draft KV pool support (best-effort piggyback on target L2/L3 ops).
        self.has_draft = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        # Task 6: EAGLE draft KV device-direct L3 under L2-bypass. Turned on at
        # attach only if the backend exposes the device-draft ABI AND the draft GPU
        # pool is a plain MLA/MHA (device-SG expressible, non-DSA). Off => draft L3
        # stays disabled under bypass (honest degrade, as increments 1-3).
        self.draft_device_enabled = False

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        self.storage_stop_event = threading.Event()

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group."""
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_prefetch_sync_groups(self) -> None:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.prefetch_sync_groups = []
        seen_rank_sets = set()

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            self.prefetch_sync_groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )

    def _destroy_prefetch_sync_groups(self) -> None:
        for group in self.prefetch_sync_groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_sync_groups = []

    def _all_reduce_prefetch_groups(self, tensor: torch.Tensor, op) -> None:
        for group in self.prefetch_sync_groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()

        self.prefetch_revoke_queue: Queue[str] = Queue()
        self.ack_backup_queue: Queue[StorageOperation] = Queue()
        self.host_mem_release_queue: Queue[torch.Tensor] = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

        # Increment 3: the background device-load thread consumes DeviceLoadTasks
        # (blocking SG GETs into GPU slots). Only spun up for async L2-bypass read;
        # stock and sync-read paths never create it (byte-identical to stock off).
        if self.l2_bypass and not self.l2_bypass_sync_read:
            self.device_load_queue = Queue()
            self.device_load_thread = threading.Thread(
                target=self.device_load_thread_func, daemon=True
            )
            self.device_load_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear storage_stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
            if self.device_load_queue is not None:
                self.device_load_queue.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)
        if self.device_load_thread is not None:
            threads.append(self.device_load_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

        # Device-load thread has stopped (joined above); clear its handles so a
        # subsequent attach re-creates a fresh thread + queue.
        self.device_load_thread = None
        self.device_load_queue = None

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            # Budget speculative prefetch at half the host pool, leaving the rest for the write-back staging path.
            self.prefetch_capacity_limit = int(0.5 * self.mem_pool_host.size)
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            self._create_prefetch_sync_groups()

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm", "mori"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            self._maybe_enable_l2_bypass()
            self._maybe_register_draft_with_storage()
            self._maybe_enable_device_draft()

            if self.l2_bypass:
                # Increment 3 (gate #1 re-anchor): the stock speculative-prefetch
                # budget is 0.5 * host-pool tokens (staging L2). L2-bypass keeps no
                # host staging for the main KV; the resource an in-flight read
                # actually occupies is GPU KV slots. Re-anchor the rate-limit budget
                # to a fraction of the DEVICE token capacity so
                # prefetch_rate_limited() throttles new discoveries by GPU pressure,
                # not by a host pool that bypass barely uses. 0.3x leaves headroom
                # for running batches + write-through pins. prefetch_tokens_occupied
                # is charged in device tokens by the async read path.
                device_capacity = int(self.mem_pool_device_allocator.size)
                self.prefetch_capacity_limit = int(0.3 * device_capacity)
                logger.info(
                    "HiCache L2-bypass: prefetch_capacity_limit re-anchored to "
                    "0.3 * device token capacity = %d tokens (was 0.5 * host = %d).",
                    self.prefetch_capacity_limit,
                    int(0.5 * self.mem_pool_host.size),
                )

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_prefetch_sync_groups()
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            self.draft_page_get_func = None
            self.draft_page_set_func = None
            self.l2_bypass = False
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_prefetch_sync_groups()

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        self.l2_bypass = False
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        self.pp_rank = get_pipeline_model_parallel_rank()
        self.pp_size = get_pipeline_model_parallel_world_size()

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        # DeepSeekV4TokenToKVPool has compressed MLA-style rank-replicated cache
        # data. storage only needs rank 0 to write it back.
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)
        is_compressed_mla_model = isinstance(
            self.mem_pool_device, DeepSeekV4TokenToKVPool
        )
        is_rank_replicated = is_mla_model or is_compressed_mla_model
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_rank_replicated
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            # TODO(hzh): Rename is_mla_model to is_rank_replicated.
            is_mla_model=is_rank_replicated,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()
            self.host_mem_release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0
            if self.device_load_thread is not None:
                # Wake + join the background device-load thread, then drop its queue
                # (any in-flight DeviceLoadTask is abandoned; the tree is being
                # reset, so its GPU slots go with the pool wipe).
                try:
                    self.device_load_queue.put_nowait(None)
                except Exception:
                    pass
                self.device_load_thread.join()
                self.device_load_queue = None
                self.device_load_thread = None

        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()
            if self.l2_bypass and not self.l2_bypass_sync_read:
                self.device_load_queue = Queue()
                self.device_load_thread = threading.Thread(
                    target=self.device_load_thread_func, daemon=True
                )
                self.device_load_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def write_device(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> int:
        """L2-bypass write-through: enqueue a device-only backup op. No host slot
        is allocated (the GPU slot itself is the RDMA source, kept pinned via the
        deferred unlock). Returns the node id as the ack handle (always succeeds;
        there is no host allocation that can fail)."""
        # Empty host placeholder keeps CacheOperation.merge_ops' torch.cat happy
        # while carrying no host slot; start_writing skips all host I/O in bypass.
        host_placeholder = device_indices.new_empty(0)
        self.write_queue.append(
            CacheOperation(host_placeholder, device_indices, node_id, priority)
        )
        self.start_writing()
        return node_id

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        if self.l2_bypass:
            # Device-direct: no D2H copy. Record empty start/finish events so the
            # existing writing_check -> _finish_write_through_ack machinery still
            # fires (which triggers the device->L3 storage backup and, on backup
            # ack, the deferred device-slot unlock). The GPU slots stay pinned by
            # the caller's inc_lock_ref until that backup ack.
            op = CacheOperation.merge_ops(self.write_queue)
            self.write_queue.clear()
            start_event = device_module.Event()
            finish_event = device_module.Event()
            start_event.record()
            finish_event.record()
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
            host_indices, device_indices = op.host_indices, op.device_indices
        else:
            host_indices, device_indices = self.move_indices(
                op.host_indices, op.device_indices
            )
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            if self.has_draft:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def load_device_direct(
        self, hash_values: List[str], node_id: int = -1
    ) -> tuple[Optional[torch.Tensor], int]:
        """L2-bypass on-demand load: allocate GPU KV slots for len(hash_values)
        pages and RDMA the pages straight into them via batch_set... no: via
        batch_get_v1_device (a blocking scatter-gather GET; the NIC writes device
        memory). No host staging, no separate H2D.

        Returns (device_indices, ok_pages): ok_pages is the CONSECUTIVE hit prefix
        from the start (KV must be a contiguous prefix — the first miss/short read
        truncates it), so a transient partial failure just shortens the loaded
        prefix and the caller recomputes the tail. (None, 0) if the device
        allocation failed (caller may evict + retry, mirroring load()).

        The fence: this call returns only after the RDMA completions are observed,
        so the device slots hold the KV on return. start_loading() then records ONE
        CUDA event so the compute stream orders after the device writes."""
        npages = len(hash_values)
        total_tokens = npages * self.page_size
        device_indices = self.mem_pool_device_allocator.alloc(total_tokens)
        if device_indices is None:
            return None, 0
        results = self.storage_backend.batch_get_v1_device(hash_values, device_indices)
        # Task 6: best-effort device-direct draft GET into the draft GPU slots (sync
        # read mode; the async path does the equivalent in _run_device_get).
        if self.has_draft and self.draft_device_enabled:
            try:
                self.storage_backend.batch_get_v1_device_draft(
                    hash_values, device_indices)
            except Exception:
                logger.debug("Device-direct draft L3 read failed (best-effort).",
                             exc_info=True)
        ok_pages = 0
        for ok in results:
            if not ok:
                break
            ok_pages += 1
        return device_indices, ok_pages

    def enqueue_device_load(
        self, device_indices: torch.Tensor, node_ids: List[int]
    ) -> None:
        """L2-bypass: queue an already-loaded device span for start_loading()'s
        fence pass. The RDMA GET already filled `device_indices` (in
        load_device_direct); this op carries only the slots + node ids so
        start_loading records the completion event on the load stream. The empty
        host placeholder keeps CacheOperation.merge_ops' torch.cat happy."""
        host_placeholder = device_indices.new_empty(0)
        op = CacheOperation(host_placeholder, device_indices, -1)
        op.node_ids = list(node_ids)
        self.load_queue.append(op)

    # ---- Increment 3: async device-direct read (background load thread) --------
    #
    # make_device_load_task / submit_device_load / _run_device_get /
    # finalize_device_load / free_device_load split the increment-2 synchronous
    # load_device_direct across the scheduler/background thread boundary:
    #   scheduler thread : make_device_load_task (alloc GPU slots) -> submit ->
    #                       [park] -> finalize (dense: no-op) + free suffix
    #   background thread : _run_device_get (blocking SG GET + local verify)
    # The dense (v1) versions live here; HybridCacheController overrides them for
    # the DSA v2-device split value + sidecar H2D.

    def make_device_load_task(
        self, hash_values: List[str]
    ) -> Optional["DeviceLoadTask"]:
        """Allocate GPU KV slots for len(hash_values) pages (scheduler thread). No
        GET yet. Returns None if the device allocation failed (the caller may evict
        and retry, mirroring load()). No collective — the TP alloc-consistency MIN
        is the caller's job."""
        total_tokens = len(hash_values) * self.page_size
        device_indices = self.mem_pool_device_allocator.alloc(total_tokens)
        if device_indices is None:
            return None
        return DeviceLoadTask(hash_values, device_indices)

    def submit_device_load(self, task: "DeviceLoadTask") -> None:
        """Hand a built task to the background device-load thread (scheduler
        thread). The thread will run the blocking GET and set task.done."""
        self.device_load_queue.put(task)

    def _run_device_get(self, task: "DeviceLoadTask") -> int:
        """Background thread: the blocking scatter-gather GET into the pre-allocated
        GPU slots + local hit-prefix count. Dense v1. No CUDA ops, no collective."""
        results = self.storage_backend.batch_get_v1_device(
            task.hash_values, task.device_indices
        )
        # Task 6: best-effort device-direct draft GET into the draft GPU slots (same
        # slots), also pure RDMA — background-safe. Does not gate the target verify.
        self._maybe_device_draft_get(task)
        return consecutive_ok_pages(results, [], len(task.hash_values))

    def finalize_device_load(self, task: "DeviceLoadTask", ok_pages: int) -> None:
        """Scheduler thread, at promotion: dense bypass has nothing to finalize (the
        SG GET already wrote the GPU slots). Hybrid overrides to H2D the sidecar."""
        return

    def free_device_load(self, task: "DeviceLoadTask") -> None:
        """Free a task's GPU slots (abort path, scheduler thread). Idempotent."""
        if task.device_indices is not None:
            self.mem_pool_device_allocator.free(task.device_indices)
            task.device_indices = None

    def free_device_indices(self, device_indices: torch.Tensor) -> None:
        """Free a span of device KV slots (used by the tree to release the unverified
        suffix after a partial load). Base = the plain allocator; hybrid overrides to
        the full-attn allocator that owns the main-KV slots."""
        self.mem_pool_device_allocator.free(device_indices)

    def device_load_thread_func(self):
        """Background consumer of DeviceLoadTasks: run each blocking GET, record the
        local verified page count, signal done. Never touches CUDA streams,
        collectives, or the radix tree — those stay on the scheduler thread."""
        while not self.storage_stop_event.is_set():
            try:
                task = self.device_load_queue.get(block=True, timeout=1)
            except Empty:
                continue
            if task is None:
                continue
            try:
                task.ok_pages = self._run_device_get(task)
            except Exception as e:
                logger.exception("device-direct background GET failed: %s", e)
                task.error = e
                task.ok_pages = 0
            finally:
                task.done.set()

    def move_indices(self, host_indices: torch.Tensor, device_indices: torch.Tensor):
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        if self.l2_bypass:
            # Device-direct: the KV was already RDMA'd into the GPU slots by
            # load_device_direct (a blocking SG GET during init_load_back). There
            # is no H2D copy to stream. Record ONE fence: mark every layer event
            # complete on the load stream so the compute path's per-layer
            # wait_until(i) orders after the NIC's device writes. The blocking GET
            # already returned (writes observed), so recording the events here is
            # the stream-side fence for the GPUDirect writes; there is no per-layer
            # overlap (a single RDMA op filled all layers at once).
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
                    start_event=producer_event.start_event,
                    finish_event=producer_event.finish_event,
                    node_ids=op.node_ids,
                )
            )
            return producer_id

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
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
                )
                if self.has_draft and i < self.mem_pool_host_draft.layer_num:
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def set_draft_kv_pool(self, draft_device_pool, draft_host_pool) -> None:
        """Register draft KV pools so L2/L3 ops piggyback draft transfers."""
        self.has_draft = True
        self.mem_pool_device_draft = draft_device_pool
        self.mem_pool_host_draft = draft_host_pool
        logger.info(
            "HiCache draft KV registered: %s (host %d slots)",
            type(draft_device_pool).__name__,
            draft_host_pool.size,
        )

        # If storage is already attached, wire up the draft I/O path now.
        # Otherwise this will be deferred until attach_storage_backend().
        self._maybe_register_draft_with_storage()
        # Task 6: if bypass is already on, wire the device-direct draft L3 path too
        # (attach may have run before the draft pool was registered).
        self._maybe_enable_device_draft()

    def _maybe_enable_l2_bypass(self) -> None:
        """Turn on device-direct write only if requested AND the backend can RDMA
        from GPU KV slots. Otherwise fall back to the stock host write path with a
        clear warning. Called from attach after the zero-copy page funcs are set."""
        self.l2_bypass = False
        if not self.l2_bypass_requested:
            return

        supports = getattr(self.storage_backend, "supports_device_transfer", None)
        if not (callable(supports) and supports()):
            logger.warning(
                "SGLANG_HICACHE_L2_BYPASS=1 but storage backend %r does not "
                "advertise supports_device_transfer(); falling back to the stock "
                "host (D2H) write path.",
                self.storage_backend_type,
            )
            return

        # Device-direct requires the same zero-copy v1 write surface the host path
        # uses; the generic copy path has no device variant. Compare underlying
        # functions: attribute access mints a fresh bound method each time, so an
        # `is` check against `self._page_set_zero_copy` would never match.
        if (
            getattr(self.page_set_func, "__func__", self.page_set_func)
            is not type(self)._page_set_zero_copy
        ):
            logger.warning(
                "SGLANG_HICACHE_L2_BYPASS=1 requires the zero-copy v1 write path "
                "(backend %r is on the generic copy path); falling back to the "
                "stock host write path.",
                self.storage_backend_type,
            )
            return

        # Per-pool device meta: the GPU KV pool must be expressible as layer-first
        # scatter-gather segments (MLA/MHA, incl. the DSA main latent). A pool the
        # device_page_meta module cannot express (e.g. a future exotic layout) keeps
        # the stock host path even if the backend advertises the ABI.
        from sglang.srt.mem_cache import device_page_meta

        if not device_page_meta.supported(self.mem_pool_device):
            logger.warning(
                "SGLANG_HICACHE_L2_BYPASS=1 but the GPU KV pool %r is not "
                "expressible as device page meta; falling back to the stock host "
                "write path.",
                type(self.mem_pool_device).__name__,
            )
            return

        try:
            self.storage_backend.register_mem_pool_device(self.mem_pool_device)
        except Exception:
            logger.exception(
                "Failed to register GPU KV pool for device-direct write; falling "
                "back to the stock host write path."
            )
            return

        self.l2_bypass = True
        self.page_set_func = self._page_set_zero_copy_device
        logger.info(
            "HiCache L2-bypass ENABLED: write-through RDMAs straight from GPU KV "
            "slots to L3 (no D2H). Backend=%r. Read path is unchanged.",
            self.storage_backend_type,
        )

    def _maybe_register_draft_with_storage(self) -> None:
        """Pick the draft L3 IO implementation."""
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        if not self.has_draft or not self.enable_storage:
            return

        backend = self.storage_backend_type

        # Multi-pool zero-copy backends.
        if backend == "mooncake":
            if self.storage_config.should_split_heads:
                logger.warning(
                    "HiCache draft L3 disabled: should_split_heads not yet "
                    "supported on the mooncake v2 path."
                )
                return
            self.storage_backend.register_mem_host_pool_v2(
                self.mem_pool_host_draft, PoolName.DRAFT
            )
            self.draft_page_get_func = self._draft_page_get_v2
            self.draft_page_set_func = self._draft_page_set_v2
            return

        # TODO: support "hf3fs", "eic", "nixl", "simm"
        if backend in {"hf3fs", "eic", "nixl", "simm"}:
            logger.warning(
                "HiCache draft L3 disabled: backend %s does not yet support "
                "draft pool registration.",
                backend,
            )
            return

        # Generic backends.
        self.draft_page_get_func = self._draft_page_get_generic
        self.draft_page_set_func = self._draft_page_set_generic

    def _maybe_enable_device_draft(self) -> None:
        """Task 6: turn on device-direct draft KV L3 under L2-bypass.

        Under bypass there is no host D2H for the target, so the stock draft L3 path
        (which reads/writes the draft HOST pool) has nothing to stage. Instead the
        draft KV is RDMA'd straight from/into the draft GPU pool's slots (the same
        slots the target rode). Enabled only if: bypass is on, a draft is registered,
        the backend exposes the device-draft ABI (register_mem_pool_device_draft +
        batch_set/get_v1_device_draft), AND the draft GPU pool's MAIN latent is a
        device-SG-expressible MLA/MHA.

        DSA DRAFT (GLM-5.2): a DSA draft is a DSATokenToKVPool — its main latent IS a
        real MLA latent (device-expressible) but it ALSO carries an indexer sidecar
        (index_k_with_scale_buffer). The original task-6 gate declined it outright
        (`use_dsa` veto) because loading the latent WITHOUT the matching indexer would
        give the draft's sparse attention a garbage index → worse acceptance than
        recompute. We now lift that veto: the draft's indexer is the SAME layer-first,
        page-indexed shape as the target's (increment 4), so we device-register it too
        and the draft KV is coherent (latent + indexer) on an L3 hit. The gate is:
        for a DSA draft, the backend must expose register_mem_pool_device_draft_sidecar
        AND device_page_meta.sidecar_supported(draft pool) must hold; if either is
        missing we DECLINE honestly (leave draft L3 off) rather than serve a half page.
        A dense (non-DSA) draft is unaffected — latent only, no sidecar.

        Best-effort: any miss logs once and leaves draft L3 disabled (recompute-safe)."""
        self.draft_device_enabled = False
        if not (self.l2_bypass and self.has_draft and self.enable_storage):
            return
        backend = self.storage_backend
        if not all(
            callable(getattr(backend, m, None))
            for m in ("register_mem_pool_device_draft",
                      "batch_set_v1_device_draft", "batch_get_v1_device_draft")
        ):
            logger.info(
                "HiCache draft L3 stays OFF under L2-bypass: backend %r lacks the "
                "device-draft ABI.", self.storage_backend_type)
            return
        from sglang.srt.mem_cache import device_page_meta

        draft_pool = self.mem_pool_device_draft
        if not device_page_meta.supported(draft_pool):
            logger.info(
                "HiCache draft L3 stays OFF under L2-bypass: draft GPU pool %r main "
                "latent is not device-SG-expressible.", type(draft_pool).__name__)
            return
        # DSA draft: the indexer sidecar must also be device-registrable, else decline.
        draft_is_dsa = bool(getattr(draft_pool, "use_dsa", False))
        if draft_is_dsa and not (
            callable(getattr(backend, "register_mem_pool_device_draft_sidecar", None))
            and device_page_meta.sidecar_supported(draft_pool)
        ):
            logger.info(
                "HiCache draft L3 stays OFF under L2-bypass: DSA draft pool %r has an "
                "indexer sidecar that this backend/geometry cannot device-register "
                "(a latent-only DSA draft would corrupt the draft indexer — declining "
                "is the honest choice).", type(draft_pool).__name__)
            return
        try:
            backend.register_mem_pool_device_draft(draft_pool)
            if draft_is_dsa:
                backend.register_mem_pool_device_draft_sidecar(draft_pool)
        except Exception:
            logger.exception(
                "Failed to register draft GPU pool for device-direct draft L3; "
                "leaving draft L3 off under bypass.")
            return
        self.draft_device_enabled = True
        logger.info(
            "HiCache draft L3 ENABLED under L2-bypass: draft KV RDMAs device-direct "
            "(best-effort) alongside the target%s. Backend=%r.",
            " (DSA: latent + indexer sidecar)" if draft_is_dsa else "",
            self.storage_backend_type)

    def _draft_device_set(self, hash_values, device_indices) -> None:
        """Best-effort device-direct draft L3 write (task 6). Mirrors _draft_page_set
        but RDMAs from the draft GPU pool's slots (device_indices)."""
        if not self.draft_device_enabled:
            return
        try:
            self.storage_backend.batch_set_v1_device_draft(hash_values, device_indices)
        except Exception:
            logger.debug(
                "Device-direct draft L3 write failed (best-effort), skipping.",
                exc_info=True)

    def _maybe_device_draft_get(self, task: "DeviceLoadTask") -> None:
        """Best-effort device-direct draft L3 read (task 6), run on the background
        device-load thread alongside the target GET (pure RDMA into the draft GPU
        slots — no CUDA, no collective, so it is background-safe). Failure just means
        the draft model recomputes those pages (EAGLE verifies against the target, so
        a missing/partial draft only lowers acceptance, never correctness)."""
        if not self.draft_device_enabled:
            return
        try:
            self.storage_backend.batch_get_v1_device_draft(
                task.hash_values, task.device_indices)
        except Exception:
            logger.debug(
                "Device-direct draft L3 read failed (best-effort), skipping.",
                exc_info=True)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

    def _page_transfer(self, operation):
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            # Best-effort draft L3 read before publishing target completion.
            # Otherwise wait_complete can race and load back target KV before
            # draft KV reaches host memory.
            if self.has_draft:
                self._draft_page_get(batch_hashes, batch_host_indices)

            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            # Check termination
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_transfer(operation)
                # operation terminated by controller, release pre-allocated memory
                self.append_host_mem_release(
                    operation.host_indices[operation.completed_tokens :]
                )
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []
        page_hashes = self.get_hash_str(
            tokens_to_fetch, last_hash, page_size=self.page_size
        )

        for start in range(0, len(page_hashes), STORAGE_BATCH_SIZE):
            batch_hashes = page_hashes[start : start + STORAGE_BATCH_SIZE]
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        self.prefetch_buffer = Queue()
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                hash_value, storage_hit_count = self._storage_hit_query(operation)
                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce_prefetch_groups(
                    storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
                )
                storage_hit_count = storage_hit_count_tensor.item()

                if storage_hit_count < self.prefetch_threshold:
                    # not to prefetch if not enough benefits
                    self.prefetch_revoke_queue.put(operation.request_id)
                    self.append_host_mem_release(operation.host_indices)
                    logger.debug(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                else:
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    # free the pre-allocated memory for pages that are not hit
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    logger.debug(
                        f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                    )
                    self.prefetch_buffer.put(operation)

            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    def write_storage_device(
        self,
        device_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """L2-bypass backup: same as write_storage, but the operation's
        `host_indices` field carries DEVICE slot indices. The backup thread's
        page_set_func (_page_set_zero_copy_device) reads device page meta and
        RDMAs straight from the GPU KV slots."""
        operation = StorageOperation(
            device_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    def _page_set_zero_copy_device(
        self, hash_values, device_indices, extra_info=None
    ) -> bool:
        # L2-bypass: `device_indices` are GPU slot indices (StorageOperation carried
        # them in its host_indices field via write_storage_device). The backend
        # reads device page meta and RDMAs from the GPU pool.
        return all(
            self.storage_backend.batch_set_v1_device(
                hash_values, device_indices, extra_info
            )
        )

    def _draft_page_set(self, hash_values, host_indices) -> None:
        """Best-effort write draft KV pages to L3 alongside the target backup."""
        if self.draft_page_set_func is None:
            return
        try:
            self.draft_page_set_func(hash_values, host_indices)
        except Exception:
            logger.debug(
                "Draft L3 write failed (best-effort), skipping.", exc_info=True
            )

    def _draft_page_get(self, hash_values, host_indices) -> None:
        """Best-effort read draft KV pages from L3 (mirrors `_draft_page_set`)."""
        if self.draft_page_get_func is None:
            return
        try:
            self.draft_page_get_func(hash_values, host_indices)
        except Exception:
            logger.debug("Draft L3 read failed (best-effort), skipping.", exc_info=True)

    def _draft_page_set_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_get_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_set_generic(self, hash_values, host_indices) -> None:
        # `{hash}.draft` mirrors HiCacheStorage._get_component_key's
        # `{key}.{pool_name}` convention so target/draft pages never collide.
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_data = [
            self.mem_pool_host_draft.get_data_page(host_indices[i * self.page_size])
            for i in range(len(draft_keys))
        ]
        self.storage_backend.batch_set(draft_keys, draft_data)

    def _draft_page_get_generic(self, hash_values, host_indices) -> None:
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_dummy = [
            self.mem_pool_host_draft.get_dummy_flat_data_page() for _ in draft_keys
        ]
        draft_pages = self.storage_backend.batch_get(draft_keys, draft_dummy)
        if draft_pages is None:
            return
        for i, p in enumerate(draft_pages):
            if p is not None:
                self.mem_pool_host_draft.set_from_flat_data_page(
                    host_indices[i * self.page_size], p
                )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            # Best-effort draft L3 write alongside target. Under L2-bypass the draft
            # host pool holds no data (no D2H staging), so the draft rides the
            # device-direct path (task 6): batch_host_indices carries the GPU slot
            # indices in bypass. Off bypass, the stock host draft path is used. If
            # device-draft could not be enabled (non-expressible/DSA draft pool),
            # draft L3 simply stays off (recompute-safe).
            if self.has_draft:
                if self.l2_bypass:
                    self._draft_device_set(batch_hashes, batch_host_indices)
                else:
                    self._draft_page_set(batch_hashes, batch_host_indices)

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                continue
