# PATCH-MANIFEST — HiCache L2-bypass, Increments 1+2 (device-direct write + read)

SGLang v0.5.15.post1. Every change is guarded by the prototype flag
`SGLANG_HICACHE_L2_BYPASS=1`; with the flag off (or a backend that does not
advertise `supports_device_transfer()`), behavior is byte-identical to stock.

- **Increment 1** = device-direct WRITE (write-through RDMAs straight from GPU KV
  slots to L3, no D2H). Documented below.
- **Increment 2** = device-direct READ (on-demand: discover L3 by exist query,
  then RDMA pages straight INTO GPU KV slots via SG GET, no host staging). New
  section "Increment 2" at the end. The scheduler and schedule_policy are
  UNPATCHED — the read reuses their existing hicache entry points
  (prefetch_from_storage / check_prefetch_progress / match_prefix /
  init_load_back / ready_to_load_host_cache / is_load_back_event_done), dispatched
  to device-direct behavior internally when `self.l2_bypass`.

DSA / hybrid-pool models (GLM-5.2 etc.) take a DIFFERENT controller
(`HybridCacheController`). Increments 1+2 left DSA on the stock host path;
**Increment 2.5** (new section at the very end) extends device-direct to DSA:
the main MLA latent goes device-direct while the small DSA indexer sidecar stays
on the host v2 path, driven by a patched `HybridCacheController` and the
backend's `batch_set_v2_device` / `batch_get_v2_device` split-value ABI.

## Bind-mount targets

Resolve the installed SGLang location once inside the container:

```bash
SGLANG=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
```

Then add these to the serve `docker run` (source = this `patched/` dir, mounted
read-only over the matching site-packages file):

```
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/mem_cache/hiradix_cache.py:$SGLANG/srt/mem_cache/hiradix_cache.py:ro
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/mem_cache/hicache_storage.py:$SGLANG/srt/mem_cache/hicache_storage.py:ro
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/mem_cache/device_page_meta.py:$SGLANG/srt/mem_cache/device_page_meta.py:ro
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/mem_cache/l3_marker_state.py:$SGLANG/srt/mem_cache/l3_marker_state.py:ro
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/mem_cache/device_pin_ledger.py:$SGLANG/srt/mem_cache/device_pin_ledger.py:ro
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/managers/cache_controller.py:$SGLANG/srt/managers/cache_controller.py:ro
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/mem_cache/hybrid_cache/hybrid_cache_controller.py:$SGLANG/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py:ro
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/mem_cache/pool_host/base.py:$SGLANG/srt/mem_cache/pool_host/base.py:ro
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/mem_cache/pool_host/bypass.py:$SGLANG/srt/mem_cache/pool_host/bypass.py:ro
```

The last two lines (`pool_host/base.py`, `pool_host/bypass.py`) are **Increment 6**
— the stub host pool. `base.py` overwrites an EXISTING container file (straight
replace); `bypass.py` is a NEW file — the mount creates it (same new-file caveat
as `device_page_meta.py`: if the runtime rejects a bind-mount over a non-existent
path, `touch` the target first or copy instead). The container base path is
`/sgl-workspace/sglang/python/sglang/srt/mem_cache/pool_host/`.

The last line (`hybrid_cache_controller.py`) is **Increment 2.5** — the DSA
device-direct controller. It overwrites an EXISTING container file (unlike
`device_page_meta.py`, which is new), so the mount is a straight replace. The
container base path is `/sgl-workspace/sglang/python/sglang/srt/mem_cache/hybrid_cache/`.

`device_page_meta.py` is a NEW file — the mount creates it (bind-mount over a
non-existent path works for files; if the runtime rejects it, `touch` the target
first in an init step, or copy instead of mount).

The dfkv backend plugin (`integration/hicache/dfkv_hicache.py`) is delivered on
the dfkv branch and mounted the same way the current deploy already mounts it
(the `/root/dfkv-*` plugin dir); no new mount line beyond the existing one.

Enable at runtime: set `SGLANG_HICACHE_L2_BYPASS=1` in the serve env.

## Changed files / functions / guards

### mem_cache/device_page_meta.py  (NEW)
Layer-first GPU-pool scatter-gather page meta (the device analogue of
`memory_pool_host.get_page_buffer_meta`). Pure python, no torch import.
- `get_device_page_buffer_meta(pool, indices)` — per page, per sub-object
  (k[,v]) a LIST of per-layer `(ptr,size)` segments; page-alignment assert.
- `device_pool_regions(pool)` — `(base,nbytes)` per layer buffer for RDMA reg.
- `supported(pool)` — MHA/MLA yes, DSA anchor pool no.
Guard: only imported/called on the bypass path.

### mem_cache/hicache_storage.py
Base-class capability hooks on `HiCacheStorage` (defaults keep every existing
backend on the stock path):
- `supports_device_transfer()` → `False`
- `register_mem_pool_device(mem_pool_device)` → store only
- `batch_set_v1_device(...)` → `NotImplementedError`
Guard: defaults are inert; only the dfkv backend overrides them.

### managers/cache_controller.py
- `env_l2_bypass()` (NEW module fn) — reads `SGLANG_HICACHE_L2_BYPASS`.
- `HiCacheController.__init__` — `self.l2_bypass_requested = env_l2_bypass()`,
  `self.l2_bypass = False`.
- `attach_storage_backend` — calls `_maybe_enable_l2_bypass()`; resets
  `l2_bypass=False` in the rollback path.
- `detach_storage_backend` — resets `l2_bypass=False`.
- `_maybe_enable_l2_bypass()` (NEW) — the capability gate: requires the flag,
  `supports_device_transfer()`, the zero-copy v1 write path, and a successful
  `register_mem_pool_device`; on success flips `self.page_set_func` to
  `_page_set_zero_copy_device` and `self.l2_bypass=True`, else warns + stays
  stock.
- `write_device(...)` (NEW) — enqueue a device-only write op (no host alloc).
- `start_writing()` — bypass branch: record empty start/finish events (no D2H),
  append the ack; stock branch unchanged. Guard: `if self.l2_bypass`.
- `write_storage_device(...)` (NEW) — enqueue a backup op whose `host_indices`
  field carries DEVICE slot indices.
- `_page_set_zero_copy_device(...)` (NEW) — `batch_set_v1_device` shim.
- `_page_backup` — draft-L3 write skipped under bypass (`and not self.l2_bypass`).

### mem_cache/hiradix_cache.py
- `l2_bypass` (NEW property) — reads `cache_controller.l2_bypass`.
- `_node_l3_backed(node)` (NEW) — bypass analogue of `node.backuped`, via a
  dynamic `node.l3_backed` attribute (TreeNode/radix_cache.py untouched).
- `write_backup` — dispatches to `_write_backup_device` under bypass.
- `_write_backup_device(...)` (NEW) — no host slot; `write_device`; mark
  `l3_backed`; `inc_lock_ref` to pin the GPU slot (RDMA source, deferred unlock).
- `_finish_write_through_ack` — under bypass: skip the CPU store event, and do
  NOT `dec_lock_ref` (deferred to the storage-backup ack). Guards on
  `self.l2_bypass`.
- `write_backup_storage` — dispatches to `_write_backup_storage_device` under
  bypass.
- `_write_backup_storage_device(...)` (NEW) — hand device slot indices (snapshot
  to CPU) to `write_storage_device`; track in `ongoing_backup` for the deferred
  unlock; no `protect_host()`.
- `_walk_split_chain` (NEW, refactor) — the key/hash chain walk, shared by
  `_concat_split_chain` (host) and the device backup.
- `_concat_split_chain` — now a thin wrapper over `_walk_split_chain`.
- `_inc_hit_count` — bypass re-write guard uses `l3_backed` (backuped stays
  False in bypass).
- `_drain_storage_control_queues_impl._drain_backup` — under bypass,
  `dec_lock_ref(node)` (deferred device-slot unlock) instead of `release_host`.
- `_force_release_pending_storage_ops` — same bypass unlock on detach/shutdown.
- `_split_node` — propagate `l3_backed` + pending-write-through tracking to both
  halves under bypass (stock only did this for `backuped` nodes).

## Deferred-unlock state machine (the correctness-critical piece)

Stock frees the GPU slot at the D2H ack (host now owns the copy). Bypass has no
D2H, so the GPU slot IS the RDMA source and must stay pinned until the L3 PUT
completes. States for one write-through node:

1. `_inc_hit_count` ≥ threshold → `write_backup` → `_write_backup_device`:
   `write_device` (records write ack event), `node.l3_backed=True`,
   `inc_lock_ref(node)` (PIN). Tracked in `ongoing_write_through[node.id]`.
2. `writing_check` sees the (immediate) write ack → `_finish_write_through_ack`:
   clears `write_through_pending_id`, calls `write_backup_storage`
   (→ `write_storage_device`, `ongoing_backup[op_id]=node`), and — unlike stock —
   does NOT `dec_lock_ref`. Slot stays PINNED.
3. Backup thread RDMAs device→L3 (`batch_set_v1_device`), enqueues the backup ack.
4. `drain_storage_control_queues` → `_drain_backup` pops the ack →
   `dec_lock_ref(node)` (UNPIN). Slot now evictable; L3 holds the page.

A split between (1) and (4) carries the pin along the chain (lock_ref is copied
to the new parent) and re-points the pending/backup tracking via
`_replace_pending_write_through_node`; the `dec_lock_ref` at (4) balances the
`inc_lock_ref` at (1) on the same (post-split) node identity. Detach/shutdown
force-release (`_force_release_pending_storage_ops`) unpins any op stuck at (3).

## Host-slot economy & read discoverability

Bypass never allocates a host slot and never sets `host_value` (`backuped`
stays False); L3 residence is tracked by the `l3_backed` marker only. The
unchanged read path discovers L3 content by hash via the prefetch exist query
(`_storage_hit_query`/`check_prefetch_progress`), NOT via the writer's local
host nodes — so cross-instance reads do not depend on the writer's `host_value`.
Same-instance re-load after GPU eviction, however, WOULD (in stock) rely on
`backuped`/`host_value` to demote-not-drop; in bypass the evicted node is
dropped and re-use goes through the L3 exist-query prefetch. That is increment-2
territory and is left as a documented limitation, not hacked.

## Load-bearing limitation (see report)

The device segments concatenate LAYER-major; the stock page-first host read
reconstructs TOKEN-major. They are transposes. A page written device-direct is
byte-coherent only with a matching device-direct (layer-major) reader
(increment 2), NOT with the unchanged host read. Increment 1 wires and offloads
the write; enabling it in isolation is a benchmark/prototype mode. Proven by
`test/python/test_dfkv_hicache_device_direct.py::TestDeviceDirectEndToEnd`.
Increment 2 supplies that matching layer-major reader (below), so with both on,
a device-written page reads back byte-identical (proven by
`::TestDeviceDirectEndToEnd::test_device_direct_write_then_read_roundtrip`).

---

# Increment 2 — device-direct READ (on-demand)

No new bind-mount files. All changes live in the already-mounted
`hiradix_cache.py` / `managers/cache_controller.py` / `hicache_storage.py`, plus
the dfkv backend plugin `integration/hicache/dfkv_hicache.py` (delivered on the
dfkv branch, mounted as before). `device_page_meta.py` is reused UNCHANGED — the
same per-layer `(ptr, size)` segment meta serves as SG-GET destinations/caps.

Enable identically: `SGLANG_HICACHE_L2_BYPASS=1`. With it off, or on a backend
without the SG-GET capability, the read path is byte-identical to stock.

## Read state machine (bypass)

Two scheduler-visible phases, both reusing UNPATCHED scheduler entry points:

1. **Discover** (exist → markers). `prefetch_from_storage` (bypass) does NOT
   prefetch into host and does NOT reserve HBM; it only RECORDS the request's
   discovery intent (`_pending_l3_discovery[req_id]`) and pins the anchor
   (`inc_lock_ref`) — collective-free, safe at queue-add. `check_prefetch_progress`
   (bypass → `_run_l3_discovery`) then, INSIDE the TP-synchronized scheduling
   loop, runs the exist query (`_storage_hit_query`), the cross-rank MIN
   all_reduce (gate #3), the 256-token threshold (gate #2), and inserts
   `l3_present` marker nodes (`_insert_helper_l3`) for the hit prefix. Returns
   True (discovery is synchronous; the request advances the same round).
2. **Load** (device-direct). `match_prefix` (bypass) climbs the `l3_present`
   markers (they have no `host_value`, so it counts `len(node.key)`) and reports
   the climb as `host_hit_length`, with `best_match_node` = deepest marker — so
   `req.needs_host_load_back()` fires unchanged. `init_load_back` (bypass →
   `load_from_storage_device`) allocates GPU slots for the marker chain and RDMAs
   the pages straight in via `cache_controller.load_device_direct`
   (→ `batch_get_v1_device`, a blocking scatter-gather GET into the per-layer
   device segments — layer-major, matching the writer). `start_loading` (bypass)
   records ONE CUDA fence event across all layers (see "the fence"). Completion
   flows through the stock `is_load_back_event_done` / `loading_check` →
   `dec_lock_ref` path, unchanged.

## Changed files / functions / guards (increment 2)

### managers/cache_controller.py
- `load_device_direct(hash_values, node_id)` (NEW) — alloc GPU slots + blocking
  `batch_get_v1_device` SG GET; returns `(device_indices, ok_pages)` where
  `ok_pages` is the consecutive hit prefix (first miss/short read truncates).
- `enqueue_device_load(device_indices, node_ids)` (NEW) — queue an
  already-loaded device span onto `load_queue` for the fence pass.
- `start_loading` — bypass branch: no H2D; record `start_event` + all layer
  events on `load_stream` as the fence, append the ack. Guard `if self.l2_bypass`.

### mem_cache/hicache_storage.py
- `HiCacheStorage.batch_get_v1_device(...)` (NEW base hook) → `NotImplementedError`;
  only the dfkv backend overrides. Inert for every other backend.

### mem_cache/hiradix_cache.py
- `_pending_l3_discovery` (NEW dict) — req_id → deferred discovery context.
- `_node_l3_present(node)` / `_node_l3_resident(node)` (NEW) — marker predicates
  (`l3_present` = discovered via exist; `_resident` = `l3_backed or l3_present`).
- `prefetch_from_storage` — bypass branch: record intent + pin anchor; no host
  prefetch, no HBM reservation, no collective.
- `_run_l3_discovery(req_id)` (NEW) + `check_prefetch_progress` bypass dispatch —
  exist query + cross-rank MIN + threshold + `_insert_helper_l3`; releases anchor.
- `_insert_helper_l3(node, key, hash_value)` (NEW) — marker insert (value=None,
  host_value=None, hash_value set, `l3_present=True`); device-direct analogue of
  `_insert_helper_host`.
- `_drop_l3_markers(nodes)` (NEW) — detach failed/partial markers deepest-first so
  their tokens recompute.
- `match_prefix` — bypass climb: count marker tokens by `len(key)` (no
  `host_value`); best/last-host node climb uses `_node_l3_resident`. Stock climb
  is asserted byte-identical (evicted stock nodes are always `backuped`).
- `init_load_back` — bypass branch → `load_from_storage_device`; returns the
  DEEPEST device-resident node after a partial load.
- `load_from_storage_device(node, mem_quota)` (NEW) — the on-demand load: walk
  marker chain, alloc + SG GET, TP-MIN gates (below), assign verified prefix,
  drop failed suffix, pin + track for `loading_check`, enqueue the fence.
- `reset` / `release_aborted_request` — drop `_pending_l3_discovery` and release
  the anchor pin.

## TP-MIN gate wiring (correctness-critical)

Every rank runs the scheduling loop over the same requests in the same order, so
all these NCCL all_reduces are balanced:
- **Discovery** (`_run_l3_discovery`): one MIN of `storage_hit_count` — a page is
  usable only if EVERY attn rank holds it (gate #3), before markers are inserted.
- **Load** (`load_from_storage_device`): (a) MIN of device-alloc success — if any
  rank could not allocate, all free + abort together (no per-rank prefix
  divergence); (b) MIN of per-rank verified pages — the usable prefix is
  truncated to the minimum, and the failed suffix's markers are dropped and
  recompute. No partial-rank silent success (the exact hole vLLM's connector
  guards at load: a rank that "loaded" fewer/short pages must not serve them).

## The fence (RDMA → compute ordering)

`batch_get_v1_device` is a BLOCKING scatter-gather GET; the NIC (GPUDirect)
writes the GPU slots and the call returns only after the completions are
observed. It runs on the scheduler thread inside `init_load_back`, before the
batch forward launches — so a CPU happens-before already orders the writes ahead
of any compute kernel. `start_loading` additionally records ONE CUDA event per
layer on `load_stream` (the LayerDoneCounter the attention backend waits on via
`wait_until`), so the compute stream also has an explicit stream-side dependency.
A single event covers all layers — the one RDMA op filled every layer at once, so
there is no per-layer overlap to stream (unlike stock's per-layer H2D).

## DSA decision (honest fallback, no faking)

Bypass is gated OFF for DSA / hybrid-pool models and they use the correct stock
host path for BOTH write and read — no device-direct sidecar was implemented and
none is faked. This is enforced THREE ways: (1) DSA models construct a
`HybridCacheController`, which has no `l2_bypass` attribute, so the
`l2_bypass` property is False; (2) `device_page_meta.supported()` returns False
for `use_dsa` pools; (3) `_maybe_enable_l2_bypass` requires the single-pool
zero-copy v1 write surface (`_page_set_zero_copy`), not the hybrid v2 path. So a
GLM-5.2 DSA instance reads its sidecar (index_k) correctly via the unchanged
host v2 machinery; it simply does not get device-direct. Revisiting DSA
device-direct (sidecar host-read + H2D, or a v2 device variant) is future work.

## Known limitations / deviations (increment 2)

- **Synchronous on-demand GET.** The load's SG GET runs synchronously on the
  scheduler thread during admission (`init_load_back`), not on a background load
  thread. Correct and race-free (the fence is trivial — data is present on
  return), but it does not overlap the GET with scheduling. The design's async
  "in-flight load delays the request one round" refinement needs cross-thread
  CUDA event ordering (a stream semaphore) that is not safely expressible at the
  Python layer; deferred. `is_load_back_event_done` therefore returns True
  immediately for bypass loads.
- **Marker accumulation.** `l3_present` markers that are discovered but never
  loaded (request never scheduled) are not device-evicted (they hold no memory)
  and persist in the radix tree; loaded-then-evicted markers ARE dropped
  (`_evict_regular`). A dedicated marker-pruning pass is future work.
- **Optimistic budget on partial load.** `host_hit_length` reflects the full
  discovered prefix; a rare transient partial GET loads less, so the request's
  input budget was computed optimistically and the tail recomputes (handled by
  chunked prefill). Not a correctness issue.
- **SGLang-side pure-python tests not extractable here.** The marker
  insert/climb logic is coupled to `TreeNode`/`RadixKey`/`_split_node` and needs
  the torch+sglang runtime (absent in the dev box), so it is verified by
  py_compile + review; the byte-exact behavior it depends on (layer-major SG
  write/read) is proven on the dfkv side by the real-cache-node roundtrip test.

---

# Increment 2.5 — DSA / hybrid device-direct (main KV device + indexer host)

Extends device-direct write AND read to DSA / hybrid-pool models (GLM-5.2,
`DSATokenToKVPool` → `attach_hybrid_dsa_pool_to_hiradix_cache` →
`HybridCacheController`). Enabled identically by `SGLANG_HICACHE_L2_BYPASS=1`;
with it off, or a non-DSA hybrid (SWA/Mamba), or a backend without the v2-device
ABI, the hybrid path is byte-identical to stock.

## DSA value-layout decision (the split value)

A DSA page's KV is a **composite**: the big MLA latent (`kv_buffer`, layer-first)
plus a small DSA indexer sidecar (`index_k_with_scale_buffer`, layer-first). The
split is honest, across two key namespaces, with NO C-server change:

- **Main MLA latent → device-direct.** Stored under the v1-style, `@sg`-chunked
  keys (`model/hash_k@sg{n}`) — the exact key scheme `batch_set_v1_device` /
  `batch_exists` (`@sg0` probe) already use. RDMA'd straight from/into the GPU
  `kv_buffer` slots via the layer-major SG put/get. `device_page_meta.supported()`
  now accepts the DSA main latent (it IS MLA-shaped; the increment-1 `use_dsa`
  veto is lifted because the sidecar finally has a home).
- **DSA indexer sidecar → host v2 path.** Stored under its own keys
  (`model/hash_indexer_k`) exactly as stock `batch_set_v2`. Written from / read
  into its host buffer (`DSAIndexerPoolHost`), then H2D'd into the device index
  buffer.

The two components share nothing but the page hash and never collide (proven by
`test_dsa_split_value_kv_and_sidecar_use_distinct_keys`).

## Sidecar coexistence mechanics

The indexer reuses the KV **page indices** (SidecarPoolSpec `indices_from_pool=KV`),
so the main-KV device slots address the indexer device buffer too. The host slots
that stage the indexer share the anchor MLA host pool's slot layout.

- **WRITE.** `hiradix._write_backup_device` (hybrid branch) allocates anchor host
  slots `side_h` for the indexer and calls `HybridCacheController.write_device
  (device_indices, sidecar_host_indices=side_h)`. `start_writing` (bypass) does a
  **sidecar-only D2H** (drives the indexer entry's `backup_from_device_all_layer`
  directly — the anchor main KV is NOT copied to host). At storage backup,
  `_page_backup_device` issues ONE `batch_set_v2_device(kv_keys, kv_device_indices,
  sidecar_transfers)` per batch: main KV device-direct + indexer from host.
  `side_h` is freed at the backup ack (`_drain_backup` →
  `_release_bypass_sidecar_host`); the main-KV GPU slot's deferred unlock is
  unchanged from increment 1.
- **READ.** `HybridCacheController.load_device_direct` allocates the main-KV GPU
  slots + a transient host staging span, issues ONE `batch_get_v2_device` (main KV
  → GPU, indexer → host staging), then **synchronously H2Ds** the indexer prefix
  into its device index buffer (`_sidecar_h2d`, on the load stream + sync) and
  frees the staging. On return the GPU slots hold both latent and indexer.
  `start_loading` (bypass) records the one-event fence. `hiradix.load_from_storage
  _device`, discovery, markers, and TP-MIN gates are the SHARED increment-2
  machinery — unchanged.

## Capability gating (four-way, DSA)

`HybridCacheController._maybe_enable_l2_bypass` requires, in order:
1. the base gate (flag, `supports_device_transfer()`, the zero-copy v1 write
   surface, **`device_page_meta.supported(main pool)`** — now added to the base
   gate for all bypass, DSA or dense — and a successful `register_mem_pool_device`);
2. the backend's v2-device split-value ABI (`batch_set_v2_device` AND
   `batch_get_v2_device`);
3. the DSA **anchor+INDEXER pool shape** (`_bypass_sidecar_supported`: every
   sidecar is `INDEXER`). SWA/Mamba hybrids (trailing-page states with their own
   indices) are declined — they keep the stock host path.

Any miss logs a clear warning and stays on the stock host path.

## Changed files / functions (increment 2.5)

### mem_cache/device_page_meta.py
- `supported(pool)` — lifted the `use_dsa` veto: a DSA pool's main latent is
  MLA-shaped and expressible; the sidecar is the controller's concern, not this
  module's.

### mem_cache/hicache_storage.py
- `HiCacheStorage.batch_set_v2_device(...)` / `batch_get_v2_device(...)` (NEW base
  hooks) → `NotImplementedError`; only the dfkv backend overrides.

### managers/cache_controller.py (base)
- `_maybe_enable_l2_bypass` — added the `device_page_meta.supported(mem_pool_device)`
  per-pool gate (inert for dense MLA/MHA; the honest capability check for DSA).

### mem_cache/hybrid_cache/hybrid_cache_controller.py (NEW patched file)
- `_sidecar_entries()`, `_bypass_sidecar_supported()`, `_maybe_enable_l2_bypass()`
  (override, the four-way DSA gate).
- `write_device(..., sidecar_host_indices)` (NEW) — bypass write enqueue (main KV
  device + indexer sidecar host).
- `write_storage_device(..., extra_pools)` (NEW) — bypass storage enqueue.
- `start_writing` — bypass branch (sidecar-only D2H, no main-KV D2H).
- `_page_backup` — bypass branch → `_page_backup_device` (`batch_set_v2_device`).
- `load_device_direct` (NEW) — bypass v2-device read + synchronous `_sidecar_h2d`.
- `_sidecar_h2d` (NEW), `enqueue_device_load` (override, hybrid CacheOperation),
  `start_loading` — bypass one-event fence branch.

### mem_cache/hiradix_cache.py
- `_write_backup_device` — hybrid branch allocates the indexer host staging slots
  and passes `sidecar_host_indices`; stores them on `node.bypass_sidecar_host`.
- `_write_backup_storage_device` — hybrid branch builds the concrete sidecar
  `PoolTransfer` (concatenated over a split chain) and passes it as `extra_pools`;
  stashes the concatenated slots on `node.bypass_backup_sidecar_host`.
- `_release_bypass_sidecar_host` (NEW) — frees those staging slots at the backup
  ack; wired into `_drain_backup` and `_force_release_pending_storage_ops`.
- `_split_node` — split `bypass_sidecar_host` alongside `value` (bypass).
- `_run_l3_discovery` — build a `HybridPrefetchOperation` with the sidecar
  `pool_transfers` for hybrid, so discovery gates via `batch_exists_v2` (a page is
  present only if BOTH the main KV `@sg0` AND the indexer are in L3).

### dfkv backend `integration/hicache/dfkv_hicache.py` (branch feat/hicache-device-direct-put)
- `batch_set_v2_device` / `batch_get_v2_device` (already committed at 8d6ec96),
  `_kv_device_set` / `_kv_device_get` helpers: main KV device SG IO + sidecar host
  v2 IO, preserving the stock DSA metric split (`on_set`/`on_get` for the anchor,
  `on_set_v2`/`on_get_v2` for the sidecar).

## Tests (increment 2.5)

dfkv side (real cache node, no GPU): `test/python/test_dfkv_hicache_device_direct.py`
- `test_dsa_split_value_write_then_read_roundtrip` — write a DSA page
  (main KV device layer-major + indexer host) then read BOTH back into fresh
  destination pools; assert every main-KV layer page AND the indexer page are
  byte-identical to source.
- `test_dsa_split_value_kv_and_sidecar_use_distinct_keys` — the two components
  live under distinct key namespaces (no collision).
- All 22 device-direct tests pass (20 prior + 2 new); the full hicache suite is
  91 passed. Committed on the branch (no push).

## Known limitations / deviations (increment 2.5)

- **SGLang-side verified by py_compile + review only** (no GPU / GLM-5.2 in the
  dev box), exactly as increments 1+2. The byte-exact split-value roundtrip — the
  load-bearing correctness claim — IS proven at the dfkv backend by the new
  real-cache-node test.
- **Host-pool allocation NOT eliminated for DSA.** The indexer reuses the anchor
  MLA host pool's slot layout, so bypass still allocates anchor host slots to index
  the indexer staging; only the expensive main-KV **D2H** is eliminated, not the
  host-pool sizing. An operator who shrank the host pool expecting full L2 removal
  will see `write_backup` return 0 (recompute-safe) under host pressure.
- **Synchronous sidecar H2D on the scheduler thread** (inside `load_device_direct`),
  like increment 2's synchronous GET — correct/race-free but no overlap.
- **Split-chain sidecar path** (a write-through node split between enqueue and
  backup) concatenates the chain's `bypass_sidecar_host`; review-verified, not
  GPU-exercised.

---

# Increment 3 — async device-direct READ + read-hit no-re-PUT + gate re-anchor

Three changes on top of increments 1/2/2.5. All still guarded by
`SGLANG_HICACHE_L2_BYPASS=1`; with it off, byte-identical to stock. **No new
bind-mount files** — every change lives in the already-mounted
`hiradix_cache.py` / `managers/cache_controller.py` /
`hybrid_cache/hybrid_cache_controller.py` / `device_page_meta.py`, plus the
existing dfkv backend mount. Dense (v1) and DSA hybrid (v2) are both covered.

## Task 1 — async read (background device-load thread)

The increment-2 read ran the on-demand SG GET **synchronously on the scheduler
thread** inside `init_load_back` (GLM 100k R3 TTFT median 29.2s — worse than the
cold round). Increment 3 moves the GET to a background thread and parks the
request in the waiting queue until it lands, reusing SGLang's existing
`check_prefetch_progress` → `continue` wait gate (stock scheduler.py:2886-2889) —
**no scheduler patch**.

New sub-flag `SGLANG_HICACHE_L2_BYPASS_SYNC_READ` (default **off** = async). Set
to 1 to keep the increment-2 synchronous read (A/B and safety escape hatch).

### The read state machine (async), all in `hiradix_cache.py`
`check_prefetch_progress` (bypass, async) dispatches per req_id to:
1. **`_start_l3_async_load`** (round 1, from `_pending_l3_discovery`): exist query +
   cross-rank MIN (gate #3) + threshold (gate #2) + `_insert_helper_l3` markers
   [as increment 2]; then collect the evicted l3-marker chain, allocate GPU slots
   (`make_device_load_task`, evict+retry once), **alloc-success MIN**, submit the
   GET to the background thread, pin the ancestor, and **return False** (park).
2. **`_poll_l3_async_load`** (rounds 2..N, from `_bypass_load_state`): **one balanced
   TP MIN over every rank's 0/1 "background GET done?" flag, every round**. Return
   False (park) until the MIN is 1 (slowest rank landed).
3. **`_promote_l3_async_load`** (final round): **verified-page MIN**, DSA sidecar
   H2D (`finalize_device_load`, scheduler thread), assign the verified marker
   prefix as device-resident, drop the failed suffix, `enqueue_device_load` (the
   fence), release the ancestor pin, return True. The nodes are now device-resident,
   so the unpatched `match_prefix`/`init_load_back` find them as a device hit and do
   NOT re-load.

### Background thread (`managers/cache_controller.py`)
- `DeviceLoadTask` — one in-flight load (hash_values, pre-allocated device_indices,
  DSA sidecars + host staging, `ok_pages`, `done` threading.Event).
- `device_load_thread_func` / `_run_device_get` — the thread runs **only** the
  blocking `batch_get_v1_device` / `batch_get_v2_device` + the local page count
  (`device_page_meta.consecutive_ok_pages`). **No CUDA stream ops, no collectives.**
- `make_device_load_task` / `submit_device_load` / `finalize_device_load` /
  `free_device_load` / `free_device_indices` — the scheduler-thread halves (alloc,
  enqueue, DSA sidecar H2D at promotion, frees). Hybrid overrides all of them for
  the v2-device split value.
- Thread lifecycle: started in `_start_storage_threads` **only when
  `l2_bypass and not l2_bypass_sync_read`**; joined in `_stop_storage_threads`;
  restarted in `reset`.

### 🔴 TP-consistency (the correctness-critical piece)
All `all_reduce`s stay on the scheduler thread; the background thread does the
per-rank GET + verify only. Ranks finish at different wall-clock rounds, so the
gate is **polled**: EVERY rank runs one done-MIN EVERY round for a parked request
(not "reduce only when I'm done", which would desync). When the MIN is 1, all ranks
do the page-count MIN + promotion **in the same round**. Result: for one request
every rank issues the identical collective sequence
`exist [, alloc [, done×k, pages]]`. Proven by
`tests/test_async_read_state_machine.py::TestCollectiveBalance` (ranks with
divergent completion rounds emit identical tag sequences). Cost of the extra
per-round done-MIN: one `int` all_reduce per parked request per round (a few µs
over NVLink); bounded by the gate-#1 rate limit below.

### GPU-slot lifecycle (pinning)
The device slots are allocated on the scheduler thread and held by
`DeviceLoadTask.device_indices` in `_bypass_load_state`; they are NOT attached to a
tree node during the GET, so eviction cannot touch them and the allocator will not
rehand them. The marker chain + ancestor are protected by the ancestor
`inc_lock_ref` held across the loading window (markers themselves carry no
value/host_value, so no eviction path touches them). On promotion the verified
slots become node `.value` (evictable via the node), the unverified suffix is freed
(`free_device_indices`), and the fence pins `last_loaded` until `loading_check`.
On abort/detach/reset (`_abort_async_load`) the task's `done` is awaited (bounded
RDMA) before its slots/staging are freed, the markers dropped, and the ancestor
unpinned. Lock-ref accounting is balanced on every path (success / below-threshold /
nothing-loadable / alloc-fail / 0-verified / abort).

### Deviation from the brief (honest)
The brief suggested "record a CUDA event on completion; gate on event ready". Cross-
thread CUDA event ordering is exactly what increment 2 flagged as *not safely
expressible at the Python layer*. Instead: the background completion is a **CPU
`threading.Event`** (polled via the per-round MIN), and the compute-ordering fence
is the **existing** `LayerDoneCounter` mechanism recorded on the scheduler thread in
`start_loading` (a blocking RDMA GET means the writes are CPU-observed done before
promotion, so recording the fence after promotion gives the same happens-before the
synchronous increment-2 path already relied on). Same guarantee, no new cross-thread
CUDA primitive.

## Task 2 — read-hit no re-PUT

A node made device-resident by a device-direct READ came FROM L3, so it is already
backed. `_promote_l3_async_load` (and the sync `load_from_storage_device`) now set
`node.l3_backed = True` alongside `l3_present = False`. `_inc_hit_count`'s
already-backed gate (`node.backuped or (l2_bypass and _node_l3_backed(node))`) then
skips the redundant write-through/backup PUT after the read — eliminating the R3
`batch_set` re-probe. `_split_node` already propagates `l3_backed`; a read-loaded
node has no `write_through_pending_id`, so `_replace_pending_write_through_node`
early-returns (no spurious tracking).

## Task 3 — gate #1 re-anchor + gate #4 cleanup

- **Gate #1** (`prefetch_capacity_limit`, `attach_storage_backend`): stock budgets
  speculative prefetch at `0.5 * host-pool tokens` (staging L2). Bypass keeps no
  host staging for the main KV, so under bypass it is re-anchored to
  `0.3 * device token capacity` (`mem_pool_device_allocator.size`) — the resource an
  in-flight device-direct read actually occupies. The async read charges
  `prefetch_tokens_occupied` in **device** tokens (at submit) and releases it (at
  promotion/abort), so `prefetch_rate_limited()` now throttles new discoveries by
  GPU pressure. TP-safe: the charge is identical across ranks (post-MIN page count).
- **Gate #4** (host-full prefetch skip): already replaced by increment 2's on-demand
  load. Confirmed the only residual host-size dependency for the prefetch budget was
  gate #1 (re-anchored above); the bypass `prefetch_from_storage` allocates no host
  slots.

## Changed files / functions (increment 3)

### mem_cache/device_page_meta.py
- `consecutive_ok_pages(kv_ok, sidecar_oks, npages)` (NEW, pure) — the verified-hit
  prefix count, shared by dense + hybrid `_run_device_get`; unit-tested off-GPU.

### managers/cache_controller.py
- `env_l2_bypass_sync_read()` (NEW), `self.l2_bypass_sync_read`.
- `DeviceLoadTask` (NEW).
- `attach_storage_backend` — gate-#1 re-anchor when `l2_bypass`.
- `_start_storage_threads` / `_stop_storage_threads` / `reset` — device-load thread
  lifecycle (created only for async bypass).
- `make_device_load_task` / `submit_device_load` / `_run_device_get` /
  `finalize_device_load` / `free_device_load` / `free_device_indices` /
  `device_load_thread_func` (NEW). `load_device_direct` (increment 2, sync) is kept
  for `SYNC_READ` mode.

### mem_cache/hybrid_cache/hybrid_cache_controller.py
- Overrides of the six device-load methods above: `make_device_load_task` (alloc
  main-KV slots + sidecar host staging + sidecar PoolTransfers), `_run_device_get`
  (`batch_get_v2_device`, no H2D), `finalize_device_load` (sidecar H2D + free
  staging), `free_device_load` / `free_device_indices` (full-attn allocator), plus
  `_full_allocator`.

### mem_cache/hiradix_cache.py
- `_BypassLoadState` (NEW), `self._bypass_load_state`, `l2_bypass_sync_read` property.
- `check_prefetch_progress` — async dispatch (`_advance_l3_async`) vs sync
  (`_run_l3_discovery`).
- `_l3_exist_query` / `_advance_l3_async` / `_start_l3_async_load` /
  `_poll_l3_async_load` / `_promote_l3_async_load` / `_abort_async_load` (NEW).
- `load_from_storage_device` (sync) — `l3_backed=True` on verified nodes;
  `free_device_indices` for the frees (hybrid-correct).
- `reset` / `release_aborted_request` / `_force_release_pending_storage_ops` —
  `_bypass_load_state` cleanup.

## Tests (increment 3)
`tests/test_async_read_state_machine.py` (pure python, no GPU):
- `TestConsecutiveOkPages` (8) — dense + hybrid verified-prefix counting.
- `TestCollectiveBalance` (5) — ranks with divergent GET-completion rounds emit
  identical collective-tag sequences (the TP-balance invariant).
Also updated the stale `test_device_page_meta.py::test_supported_...` to assert
DSA main-latent IS supported (increment 2.5 lifted the veto).

## Known limitations / deviations (increment 3)
- **SGLang-side verified by py_compile + review + pure-logic unit tests only** (no
  GPU on the dev box). The async state machine's tree/CUDA coupling (marker
  climb, slot assignment, fence) is review-verified; the extractable logic (hit
  counting, collective balance) is unit-tested. Requires a GPU A/B (async vs
  `SYNC_READ=1`) to confirm the R3 latency win.
- **Marker accumulation** (increment 2) unchanged: discovered-but-never-loaded
  markers persist.
- **DSA host staging not eliminated** (increment 2.5) unchanged: async still allocs
  transient sidecar host staging per load (freed at finalize); the gate-#1 device
  budget bounds concurrency.
- **Synchronous DSA sidecar H2D at promotion** (scheduler thread) — small indexer
  only, but still a `load_stream.synchronize()` per promoted DSA req.
- **Fence timing**: if a promoted req is not scheduled the round it promotes (batch
  full), its `enqueue_device_load` op is fenced by a later `start_loading` — the
  same property the increment-2 sync path already had; nodes stay pinned meanwhile.

---

# Increment 4 — DSA indexer sidecar device-direct (task 4) + EAGLE draft L3 (task 6)

Two changes on top of increments 1/2/2.5/3, both still guarded by
`SGLANG_HICACHE_L2_BYPASS=1` (with it off, byte-identical to stock). **No new
bind-mount files** — every change lives in the already-mounted `device_page_meta.py`
/ `hicache_storage.py` / `hiradix_cache.py` / `hybrid_cache/hybrid_cache_controller.py`
/ `managers/cache_controller.py`, plus the existing dfkv backend mount.

## Task 4 — DSA indexer sidecar device-direct (eliminates the host-pool residual)

Increment 2.5 left the DSA indexer sidecar (`index_k_with_scale_buffer`) on the host
v2 path: written via a D2H into anchor host staging, read into transient host staging
then H2D'd. That anchor host allocation was the last host-pool承重 in DSA bypass.
Increment 4 makes the sidecar device-direct too — it gets its own GPUDirect MR and
RDMAs straight from/into its GPU index buffer, so DSA bypass allocates **zero** host
slots on every operational path.

**Sidecar geometry (important):** the indexer buffer is ALSO layer-first — a list of
`layer_num` per-layer 2D `(page_num, page_bytes)` tensors — but PAGE-indexed (row
`slot // page_size`), not token-indexed like the main latent. So a page's sidecar is
`layer_num` per-layer segments (same SG shape as the main latent, NOT single-segment)
and needs the SAME `@sg` chunking on a narrow HCA.

### device_page_meta.py
- `sidecar_supported(pool)` / `sidecar_device_pool_regions(pool)` /
  `get_device_sidecar_page_buffer_meta(pool, indices)` (NEW) — the layer-first,
  page-indexed device page meta + RDMA regions for the indexer, parallel to the
  main-latent `get_device_page_buffer_meta` but page-row addressed, sub=1.

### hicache_storage.py (base hooks, inert defaults)
- `register_mem_pool_device_sidecar(name, device_pool)` → no-op; only the dfkv
  backend overrides (GPUDirect MR for the indexer buffers).

### managers/cache_controller.py (base)
- no sidecar change (dense bypass has no sidecar); the DSA overrides are in the
  hybrid controller.

### mem_cache/hybrid_cache/hybrid_cache_controller.py
- `_maybe_enable_l2_bypass` — added two gates: the backend must expose
  `register_mem_pool_device_sidecar`, and every sidecar's device pool must be
  `device_page_meta.sidecar_supported`; on success it registers each sidecar device
  pool. Any miss → stock host path (honest: no half-device sidecar).
- `write_device` — dropped `sidecar_host_indices`; the indexer rides the KV device
  slots (device-direct at backup), so no host slot / D2H.
- `start_writing` (bypass) — dropped the sidecar-only D2H; now records empty
  start/finish events like the dense base bypass branch (NO D2H at all).
- `_page_backup_device` — builds DEVICE sidecar PoolTransfers
  (`device_indices=batch_kv_device`, `host_indices=None`) from `_sidecar_entries()`;
  one `batch_set_v2_device` writes main KV + indexer both device-direct.
- `load_device_direct` (sync) / `make_device_load_task` + `_run_device_get` (async) —
  device sidecar transfers, no host staging; `_sidecar_h2d` **deleted**;
  `finalize_device_load` override **removed** (inherits the base no-op — nothing to
  H2D on the scheduler thread anymore); `free_device_load` no longer frees a
  `side_host` (there is none).
- `write_storage_device` — `extra_pools` is now unused under bypass (kept for
  signature compat); the sidecar is derived from `_sidecar_entries()` in the backup.

### mem_cache/hiradix_cache.py
- `_write_backup_device` — hybrid and dense branches collapse to one: no sidecar host
  alloc; just `write_device(device_indices=node.value)`.
- `_write_backup_storage_device` — dropped the sidecar host concat / `extra_pools`
  plumbing and `node.bypass_backup_sidecar_host`.
- `_release_bypass_sidecar_host` **deleted**; removed its two call sites in
  `_drain_backup` and `_force_release_pending_storage_ops` (no host staging to free).
- `_split_node` — removed the `bypass_sidecar_host` split block (the sidecar rides
  the KV slots, split implicitly with `node.value`).

### dfkv backend `integration/hicache/dfkv_hicache.py` (branch feat/hicache-device-direct-put)
- `_flatten_device` generalized with `keys_fn`/`sub` params (the sidecar + draft reuse
  the identical `@sg` chunking under distinct namespaces).
- `register_mem_pool_device_sidecar(name, device_pool)` — GPUDirect MR for the
  indexer buffers (deduped against already-registered regions).
- `_sidecar_device_set` / `_sidecar_device_get` — device SG put/get of the indexer
  under `_pool_keys(name, hash)@sg{n}`, from `get_device_sidecar_page_buffer_meta`.
- `batch_set_v2_device` / `batch_get_v2_device` — route a DEVICE sidecar transfer
  (`_is_device_transfer`: device_indices set, host_indices None) through the device
  path; a host sidecar transfer keeps the stock host v2 path. Metrics unchanged
  (main = on_set/on_get; sidecar = on_set_v2/on_get_v2).
- `batch_exists_v2` — probes `@sg0` for a device-registered sidecar (mirrors the main
  KV exist probe), so discovery gates on the chunked indexer key.

### extra_config geometry keys
None added. The sidecar device geometry is derived from the pool
(`index_k_with_scale_buffer` shape/stride); no new launch-config key.

## Task 6 — EAGLE draft KV device-direct L3 (best-effort)

Increments 1-3 disabled draft L3 under bypass (the draft host pool holds no data with
no D2H staging). Task 6 restores it via the SAME device-direct mechanism as the main
pool: draft KV RDMAs straight from/into the draft GPU pool's slots (the same slots the
target rode) under a distinct `.draft` key namespace. Best-effort (try/except): a
missing/partial draft only lowers EAGLE acceptance, never correctness (the target
verifies the draft), so it never gates the target load.

**Route decision:** device-direct, gated on `device_page_meta.supported(draft pool)
AND not use_dsa`. A DSA draft (indexer sidecar) is DECLINED (honest degrade, logged):
its sidecar is not handled for draft, and loading an incomplete draft KV is left off
rather than silently partial. If the backend lacks the device-draft ABI, draft L3
stays off.

### managers/cache_controller.py (base — dense + used by hybrid)
- `draft_device_enabled` (NEW state), `_maybe_enable_device_draft()` (NEW, called
  from `attach_storage_backend` and `set_draft_kv_pool`), `_draft_device_set()` /
  `_maybe_device_draft_get()` (NEW best-effort wrappers).
- `_page_backup` — bypass branch writes the draft device-direct (`_draft_device_set`)
  instead of the stock host draft path.
- `_run_device_get` (async) + `load_device_direct` (sync) — best-effort device-direct
  draft GET into the draft GPU slots (pure RDMA, background-safe).

### mem_cache/hybrid_cache/hybrid_cache_controller.py
- `_page_backup_device` — best-effort `_draft_device_set` after each main-KV batch.
- `_run_device_get` — `_maybe_device_draft_get` alongside the target GET.

### hicache_storage.py (base hooks, inert)
- `register_mem_pool_device_draft` (no-op) / `batch_set_v1_device_draft` /
  `batch_get_v1_device_draft` (NotImplementedError); only dfkv overrides.

### dfkv backend
- `register_mem_pool_device_draft` (GPUDirect MR for the draft pool),
  `_draft_keys(hash, sub)` (`.draft` namespace, TP-aware: MLA draft sub=1 →
  replicated, no tp suffix, rank-0-only write; MHA draft sub=2 → tp_size/tp_rank
  suffix), `batch_set_v1_device_draft` / `batch_get_v1_device_draft`.

## TP-consistency (unchanged invariant)
No collective sequence changed. The sidecar device write/read are local RDMA (the
write's MLA rank-skip matches the main latent + stock host v2 path; the read has no
rank skip). `finalize_device_load` became a no-op — it never held a collective. The
async read's balanced per-round done-MIN / alloc-MIN / pages-MIN (increment 3) are
untouched. Draft GET runs on the background thread (pure RDMA, no collective).

## Host-pool residual audit (task 4 goal: sidecar host承重 = 0 under bypass)
`grep mem_pool_host.alloc/.free` over the bypass paths: the remaining references are
all STOCK (non-bypass) code — `write()`, `load()`, the stock `start_writing` /
`start_loading` branches (after the `if self.l2_bypass: return`), and the stock
`prefetch_from_storage` branch (after the bypass `return`). `_sidecar_entries()` and
`_init_extra_host_mem_release_queues` only READ `mem_pool_host.entries` metadata (pool
names), not slots. So under bypass NO host slot is allocated for the main KV OR the
sidecar; the host pools exist structurally but can be shrunk to a stub. This resolves
the increment-2.5 known limitation ("Host-pool allocation NOT eliminated for DSA").

## Tests
- dfkv `test/python/test_dfkv_hicache_device_direct.py` (+4, `-k hicache` = 95 passed,
  was 91): `test_dsa_split_value_device_sidecar_roundtrip` (real cache node: main KV +
  indexer BOTH device-direct, byte-exact per layer), `test_device_sidecar_read_miss_
  returns_false`, `test_draft_device_direct_roundtrip`, `test_draft_keys_distinct_and_
  tp_aware`. (The full non-`-k hicache` suite has pre-existing cross-test subprocess-
  server interference — the stock baseline fails the same set; unrelated to this work.)
- SGLang `tests/test_device_page_meta.py` (+4): `TestDeviceSidecarPageMeta`
  (page-indexed layer segments, alignment assert, regions, supported).
- SGLang side otherwise py_compile + review only (no GPU on the dev box), as
  increments 1-3.

## Known limitations / deviations (increment 4)
- **SGLang-side verified by py_compile + review + the dfkv-side byte-exact roundtrip**
  (no GPU / GLM-5.2 in the dev box). The load-bearing correctness claim — the sidecar
  layer-major device write/read reassemble byte-exact under distinct keys — IS proven
  on the dfkv side against a real cache node.
- **Draft L3 is best-effort.** A DSA draft pool is declined (device-only main latent,
  sidecar unhandled for draft); requires a GPU EAGLE A/B to confirm the acceptance
  benefit. **[SUPERSEDED by increment 5 below — DSA draft is now device-direct
  latent + indexer; the increment-4 "honest decline" was over-conservative.]**
- **Marker accumulation / gate-#1 device budget** (increments 2/3) unchanged.

---

# Increment 5 — DSA draft indexer sidecar device-direct (lifts the task-6 DSA-draft decline)

One change on top of increments 1/2/2.5/3/4, still guarded by
`SGLANG_HICACHE_L2_BYPASS=1` (with it off, byte-identical to stock). **No new
bind-mount files** — the SGLang change lives in the already-mounted
`managers/cache_controller.py` and `mem_cache/hicache_storage.py`, plus the existing
dfkv backend mount.

## Why the increment-4 decline was over-conservative

Increment 4 (task 6) enabled EAGLE draft L3 device-direct for a **dense** draft only,
and declined a DSA draft (`not use_dsa` gate) reasoning "its sidecar isn't handled for
draft." Ground-truth from the SGLang source resolves the geometry:

- **The GLM-5.2 MTP draft IS a `DSATokenToKVPool`.** For a draft worker,
  `ModelConfig` rewrites `GlmMoeDsaForCausalLM` → `DeepseekV3ForCausalLMNextN`
  (`configs/model_config.py`), which `is_deepseek_dsa()` matches (and it inherits
  `index_topk`), so `model_runner_kv_cache_mixin` builds a `DSATokenToKVPool` for the
  draft — it DOES carry its own `index_k_with_scale_buffer` indexer sidecar.
- **A latent-only DSA draft is net-negative, not merely "lower acceptance."** Loading
  the draft's MLA latent WITHOUT its matching indexer feeds the draft's sparse
  attention a garbage index → near-zero acceptance on those pages → wasted draft
  compute. So declining (increment 4) was the right call *given* only the latent was
  handled — but the fix is to handle the indexer, not to stay declined.
- The draft indexer has the **identical** layer-first, page-indexed geometry as the
  target indexer (increment 4's `sidecar_supported` / `get_device_sidecar_page_buffer_meta`
  are pool-generic and already express it), just with the draft's small `layer_num`.
  So it device-registers with the exact same machinery.

(Note: stock's own `maybe_register_hicache_draft` builds a plain `MLATokenToKVPoolHost`
for a DSA draft — DSA subclasses MLA — so **stock never persists the draft indexer
either**. This increment gives the draft indexer a home that stock never had.)

## Route decision

Device-direct the DSA draft's indexer sidecar alongside its main latent, both under
distinct `.draft` namespaces (`.draft_k` for the latent, `draft_indexer` for the
indexer). Gate: for a DSA draft, the backend must expose
`register_mem_pool_device_draft_sidecar` AND `device_page_meta.sidecar_supported(draft
pool)` must hold; otherwise **still decline honestly** (a half-loaded DSA draft is
net-negative). A dense draft is unaffected (latent only, no sidecar registered).
Best-effort throughout (try/except): a missing/partial draft only lowers EAGLE
acceptance, never correctness (the target verifies the draft).

### mem_cache/hicache_storage.py (base hook, inert default)
- `register_mem_pool_device_draft_sidecar(mem_pool_device_draft)` (NEW) → no-op; only
  the dfkv backend overrides.

### managers/cache_controller.py
- `_maybe_enable_device_draft` — **lifted the `use_dsa` veto**. Now: gate the draft
  MAIN latent on `device_page_meta.supported(draft pool)` (unchanged); if the draft is
  DSA (`use_dsa`), additionally require the backend's draft-sidecar ABI +
  `sidecar_supported(draft pool)` and register the draft sidecar device pool — else
  decline honestly (clear log). Dense draft path unchanged. `_draft_device_set` /
  `_maybe_device_draft_get` are UNCHANGED: the sidecar rides the same
  `batch_set/get_v1_device_draft` calls (dfkv handles it internally once registered),
  so no scheduler/collective sequence changed.

### dfkv backend `integration/hicache/dfkv_hicache.py` (branch feat/hicache-device-direct-put)
- `register_mem_pool_device_draft_sidecar(mem_pool_device_draft)` (NEW) — registers the
  draft pool's `index_k_with_scale_buffer` regions (GPUDirect MR) under the reused
  sidecar name `draft_indexer`; sets `_draft_sidecar_name`.
- `_draft_sidecar_device_set` / `_draft_sidecar_device_get` (NEW) — best-effort wrappers
  over the shared `_sidecar_device_set/get` for the `draft_indexer` namespace; return
  None when no draft sidecar is registered (dense draft), `[False]*n` on hard error.
- `batch_set_v1_device_draft` / `batch_get_v1_device_draft` — after the main-latent SG
  put/get, also put/get the draft indexer sidecar (when registered) and **AND** the
  per-page result with the latent result: a page is stored / is a hit only when BOTH
  the latent and indexer land, so the read never serves a latent without its indexer.

## TP-consistency (unchanged invariant)
No collective sequence changed. The draft indexer write is TP-replicated (MLA rank-skip,
via the shared `_sidecar_device_set`'s `is_mla and tp_rank!=0` skip — matching the draft
latent), the read has no rank skip. The draft GET still runs on the background load
thread (pure RDMA, no collective).

## Tests
- dfkv `test/python/test_dfkv_hicache_device_direct.py` (+4, `-k hicache` 95 → 99, all
  real-cache-node except the two pure-key tests):
  `test_dsa_draft_device_direct_latent_and_indexer_roundtrip` (DSA draft latent +
  indexer BOTH device-direct, byte-exact per layer), `test_dsa_draft_indexer_read_miss_
  is_page_failure` (latent-hit + indexer-miss ⇒ page miss ⇒ recompute),
  `test_dense_draft_no_sidecar_is_latent_only` (non-DSA draft unchanged, sidecar hooks
  return None), `test_draft_indexer_keys_distinct_namespace`.
- SGLang side py_compile + review only (no GPU on the dev box), as increments 1-4.

## Known limitations / deviations (increment 5)
- **SGLang-side verified by py_compile + review + the dfkv-side byte-exact roundtrip**
  (no GPU / GLM-5.2 in the dev box). The load-bearing correctness claim — the draft
  latent + indexer reassemble byte-exact under distinct keys — IS proven on the dfkv
  side against a real cache node. The end-to-end EAGLE-acceptance win needs a GPU A/B
  (EAGLE R3 with DSA draft-L3 on vs off).
- **Still honest-declines** when the draft sidecar geometry is not device-expressible
  or the backend lacks the draft-sidecar ABI (a half-loaded DSA draft would corrupt the
  draft indexer). No faking.
- **Draft indexer host residual = 0.** Like the target sidecar (task 4), the draft
  indexer rides its GPU buffer device-direct; no draft host staging is allocated.

---

# Increment 6 — L2-bypass STUB host pool (drop the idle --hicache-size pinned buffer)

Increments 1-5 eliminated every host-slot USE under bypass (main KV, DSA indexer
sidecar, and draft all device-direct; "Host-pool residual audit" above = 0 host
slots allocated on any bypass path). But the host pool was still CONSTRUCTED at
its full `--hicache-size` — a multi-GB `cudaHostRegister`'d buffer sitting idle.
Increment 6 makes it a true stub: under bypass the host pool holds a few pages
(~a few MB) instead of GB, so host footprint drops from `--hicache-size` GB to
tens of MB while bypass behavior/performance is unchanged (the data path is
GPUDirect GPU<->L3, never these host slots). Still guarded by
`SGLANG_HICACHE_L2_BYPASS=1`; with it off, byte-identical to stock.

## New bind-mount files
`mem_cache/pool_host/base.py` (overwrites; the `HostKVCache` base constructor) and
`mem_cache/pool_host/bypass.py` (NEW; pure torch-free helpers). Both listed in the
bind-mount block at the top.

## The construction-order problem (why the gate lives in the base constructor)
The host pool is built in `HiRadixCache.__init__` / the DSA assembler **before**
the storage backend attaches, so the controller's full capability gate
(`_maybe_enable_l2_bypass`, which also needs `supports_device_transfer()` + the
v2-device ABI) has NOT run yet. At construction we know only: the env flag + the
GPU pool shape. So the stub gate is `SGLANG_HICACHE_L2_BYPASS requested AND
device_page_meta.supported(device_pool)` — the same "is this pool device-direct
expressible" predicate the controller uses, evaluated on the info available early.

**Safety net for the residual case** (flag on, pool expressible, but the
controller later DECLINES bypass for a backend reason — non-device backend, or an
HCA too narrow for the `@sg` chunking): the stock host path still runs correctly
against a stub, because **every** `mem_pool_host.alloc` caller treats
`alloc()==None` (a full/tiny pool) as a **recompute-safe skip** — `write()`
returns None, `prefetch_from_storage` releases and returns, both no-op. L2 just
goes ineffective; correctness is preserved. Verified by reading all stock alloc
sites (`cache_controller.write`, `hiradix.prefetch_from_storage`, the hybrid
controller's `write`). The degrade is LOUD: the stub log at construction + the
controller's decline warning. Never silent corruption.

## What gets stubbed (one surgical point covers all)
The change is entirely in `HostKVCache.__init__` (the base every host pool goes
through). One gate there transitively covers:
- **Dense MLA / MHA** host pools (direct `MLATokenToKVPoolHost` /
  `get_mha_host_pool_cls`).
- **DSA main latent** (`MLATokenToKVPoolHost` via the hybrid assembler's
  `_build_kv_host_pool`).
- **DSA indexer sidecar** (`DSAIndexerPoolHost`) — NOT patched directly: it sets
  `self.size = anchor_host.size` / `self.page_num = anchor_host.page_num`, so when
  the anchor MLA host pool is stubbed the indexer inherits the tiny size and
  allocates a tiny buffer automatically. Its own memory check/log then report the
  (tiny) figure honestly. No `memory_pool_host.py` change needed.
- **EAGLE draft** host pool (`MLATokenToKVPoolHost` for a DSA draft) — same base
  constructor, same stub.

`HostPoolGroup.size` also derives from `anchor.host_pool.size`, so the aggregate
DSA pool view inherits the stub too.

**Not stubbed (correctly):** Mamba/SWA/sparse pools — `device_page_meta.supported`
is False for them, so they keep the real host pool and the honest stock path
(bypass is declined for those models anyway). Note `MambaPoolHost` also has its own
inline sizing (not the base path), so it is doubly excluded.

## Downstream `host.size` dependencies (grep-verified none break)
- **Gate #1 prefetch budget** (`0.5 * mem_pool_host.size`): already re-anchored to
  `0.3 * device capacity` under bypass by increment 3; the stubbed host size is not
  read on the bypass path. On the stock fallback it becomes `0.5 * tiny` → prefetch
  effectively off, consistent with a tiny pool.
- **`DSAIndexerPoolHost.size` / `.page_num`**: inherit the stubbed anchor (above).
- **`HostPoolGroup.size`**: inherits the stubbed anchor.
- **`available_size()` reads** (hiradix prefetch shrink path): return the tiny free
  count → the None-handling shrink/skip path fires. Recompute-safe.
- **`draft_host_pool.size`** (cache_controller `set_draft_kv_pool`): log string only —
  but the draft pool's SLOT COUNT is load-bearing (target host_indices are used on
  it verbatim), hence the process-shared stub count below.
- **Slot-range asserts**: `alloc` only asserts `need_size % page_size == 0` and
  returns None when `need_size > available` — no hard assert on `size`. Staging
  buffers use `min(page_num, 64)` with `page_num >= 1` → no divide-by-zero.

## Sizing math
Stub `self.size = max(_L2_BYPASS_STUB_PAGES, layer_num) * page_size` (raw), then the
UNCHANGED stock page-align (`page_num = size // page_size + 1;
size = page_num * page_size`). Positive, page-aligned, non-zero for any page size
(the `< 4GB` divide-by-zero the operator hit was the GB→token conversion
underflowing / downstream framework asserts; the stub sets the token count directly
and never touches that path). At GLM-5.2 (78 layers, ~44.9 KB/token main latent,
page 64) the main buffer is ~227 MB, the DSA indexer ~53 MB, the EAGLE draft a few
MB — a few hundred MB total, against ~100 GB for an unstubbed `--hicache-size`.

### Why the floor is `layer_num` pages, not 1 (fixes the on-node IndexError)
The first cut used a flat 1 page and the scheduler died at construction:
`IndexError: index 2 is out of bounds for dimension 0 with size 2`
(`memory_pool_host.py:1295`). Root cause: **page-major layouts put `page_num` in
dim0**, but the stock subclass constructors still build per-layer views out of that
same dim0:

| layout | buffer dim0 | per-layer view |
|---|---|---|
| MLA `layer_first` (`memory_pool_host.py:1325`) | `layer_num` | `kv_buffer[i]`, in range |
| MLA `page_first` (`:1331`) | `size` | `transpose(0,1)[i]` (`:1293`), in range |
| **MLA `page_first_direct` (`:1338`) / `page_first_kv_split` (`:1348`)** | **`page_num`** | **`kv_buffer[i] for i in range(layer_num)` (`:1295`)** |
| **MHA `page_first_direct` (`:154`) / `page_head` (`:163`)** | **`page_num`** | **`k/v_buffer[i] ... range(layer_num)` (`:126-127`)** |

Those page-major per-layer views are only ever in range because a real host pool has
`page_num >> layer_num`; nothing in stock enforces it. The production node runs
`--hicache-mem-layout page_first_direct`, so a 2-page stub with `layer_num=78` blew
up immediately. The floor (`page_num = pages + 1 >= layer_num`, one page of margin)
restores that implicit invariant **by capacity alone** — buffer shapes, layouts and
dimension semantics stay exactly stock, no new branch in `memory_pool_host.py`.

### Why the stub slot count is process-shared
The draft host pool is deliberately built with
`host_to_device_ratio = primary.size / draft_device.size` so its slot count matches
the target host pool 1-to-1 (`kv_cache_builder.py:101-110`), and the controller then
indexes the draft pool with the *target's* `host_indices`
(`cache_controller.py:888` `backup_from_device_all_layer`, `:1716` `get_data_page`).
The stub ignores ratio/`--hicache-size`, so a per-pool `layer_num` floor alone would
give the 78-layer target 79 pages and the 1-layer EAGLE draft 1 page — a real
out-of-range risk on the residual (bypass-declined) host path.
`l2_bypass_shared_stub_raw_tokens` therefore memoizes one monotonic-max token count
per `page_size` for the whole process; the target is constructed first (the draft
registers against an existing tree cache), so every later stub pool reuses it.

## `--hicache-size` semantics under bypass+stub
`--hicache-size` becomes **irrelevant** — the stub ignores it and logs so at
construction: `HiCache L2-bypass stub host pool (<cls>): --hicache-size ignored,
host footprint ~XX MB (<n> tokens x <b> B/token); GPUDirect GPU<->L3 owns the data
path.` Operators can leave the production `--hicache-size 32` in the serve command
unchanged; it no longer sizes anything under bypass. (The increment-4 proof that
`hicache=4` still reads back byte-correct was the earlier "shrink the arg"
demonstration; increment 6 makes the shrink automatic and total.)

## Changed files / functions (increment 6)
### mem_cache/pool_host/bypass.py (NEW, pure — no torch/sglang at module load)
- `_L2_BYPASS_STUB_PAGES = 1` (floor); `env_l2_bypass_requested()`;
  `l2_bypass_stub_pages(layer_num)` — `max(_L2_BYPASS_STUB_PAGES, layer_num)`, the
  page-major dim0 invariant above; `l2_bypass_stub_raw_tokens(page_size, layer_num)`
  / `l2_bypass_stub_tokens(page_size, layer_num)` (pure pre-/post-align token counts,
  exposed for tests); `l2_bypass_shared_stub_raw_tokens(page_size, layer_num)` — the
  process-shared monotonic-max count the constructor uses (+
  `reset_l2_bypass_stub_sizing()` for tests);
  `l2_bypass_stub_applies(device_pool)` — the gate (env flag AND lazy
  `device_page_meta.supported`; any import failure → False = keep real pool).

### mem_cache/pool_host/base.py
- `HostKVCache.__init__` — compute `self.l2_bypass_stub = l2_bypass_stub_applies(
  device_pool)`; when set, size from
  `l2_bypass_shared_stub_raw_tokens(page_size, device_pool.layer_num)` instead of
  `--hicache-size`/ratio and skip the big-memory warning+check in favor of the stub
  log (which also reports `page_num` and `layer_num`). **Stock branches are byte-identical, merely re-indented under `else:`**; with
  the flag off `l2_bypass_stub` is False (env short-circuit, no device_page_meta
  import) and the original path runs verbatim.

## Tests (increment 6)
`tests/test_l2_bypass_stub.py` (pure python, no GPU/torch): env-flag parser
(truthy/falsy/whitespace), stub token count is positive+page-aligned+non-zero for
page sizes 1..256, final == raw + page_size (matches the constructor align),
footprint stays small, and the gate short-circuits on the env flag /
defers to a (fake-injected) `device_page_meta.supported` / declines on
missing-module. Plus the post-crash invariants: `page_num >= layer_num` for every
(page_size 1..256) x (layer_num 1..126) pair including the exact GLM-5.2 regression
case (64/78), the floor stays capacity-only (page-aligned, `final == raw +
page_size`), the default `layer_num` keeps the 1-page stub, footprint stays well
under a GB, and the shared slot count makes the 1-layer draft match the 78-layer
target (monotonic max, per-page_size, resettable). 21 tests pass. The base constructor's torch-coupled allocation is
py_compile + review only (no GPU on the dev box), as increments 1-5.

## Known limitations / deviations (increment 6)
- **SGLang-side allocation verified by py_compile + review + pure-logic unit tests**
  (no GPU on the dev box). Needs a GPU run to confirm the host RSS drop (expect the
  hicache buffer to vanish from host memory) and that GLM-5.2 DSA+EAGLE+bypass
  needle stays byte-correct with the stub — i.e. the increment-4 `hicache=4` proof,
  now with the buffer stubbed at the source rather than shrunk by the arg.
- **Flag-on + expressible pool + later backend decline** degrades to stock-with-no-
  effective-L2 (recompute-safe, loud), NOT byte-identical to a real-host stock run.
  This only affects a config that requests bypass on a backend that cannot honor it;
  the production node (dfkv device-direct) always enables bypass. Documented, not
  hidden.

---

# Increment 7 — FUSE the EAGLE draft into the target's SG batch (draft L3 costs no extra RDMA op)

One change on top of increments 1/2/2.5/3/4/5/6, still guarded by
`SGLANG_HICACHE_L2_BYPASS=1` (with it off, byte-identical to stock). **No new
bind-mount files** — the SGLang changes live in the already-mounted
`managers/cache_controller.py`, `mem_cache/hybrid_cache/hybrid_cache_controller.py`
and `mem_cache/hicache_storage.py`, plus the existing dfkv backend mount.

## What this is NOT (diagnosis first — two proposed fixes were provably no-ops)

The brief was "draft-L3 re-writes hot pages every decode; add a dedup gate like the
main KV's `l3_backed`". Reading the shipped paths, **that premise does not hold and
two of the three proposed routes would have changed nothing**:

1. **There is no draft-only write path.** `_draft_device_set` is called from exactly
   two places — `HiCacheController._page_backup` (dense) and
   `HybridCacheController._page_backup_device` (DSA) — both INSIDE the per-batch loop
   of a storage backup operation. A backup op only exists if
   `hiradix._inc_hit_count` → `write_backup` → `_write_backup_device` ran, and that is
   already gated by
   `already_backed = node.backuped or (l2_bypass and _node_l3_backed(node))`
   (`hiradix_cache.py:1108`). **Draft writes are a strict subset of main-KV
   write-through events**, so they cannot fire "every decode step" while the main KV
   stays quiet. Route (a) `draft_l3_backed` and route (b) "reuse the main
   `l3_backed`" are therefore both no-ops: (b) IS the status quo, and (a) would be a
   second marker on the same node with the same lifetime.
2. **Route (c) — the dfkv exist-dedup gate — is already wired on the draft path.**
   `batch_set_v1_device_draft` puts through `_put_sg_flat`
   (`dfkv_hicache.py:939`), whose `_backup_exist_gate` (default on,
   `DFKV_BACKUP_EXIST_GATE`) probes first and skips every already-present sub-key;
   the draft indexer goes through `_sidecar_device_set` → the same `_put_sg_flat`.
   So a genuinely redundant draft PUT already costs a probe and zero bytes.

**What draft-L3 actually costs in a hot round is RDMA OP COUNT, not bytes.** With the
draft on, a DSA backup batch issued 4 exist probes + up to 4 SG puts (target latent,
target indexer, draft latent, draft indexer) instead of 2 + 2, and — the one that
sits on the TTFT critical path — an async device LOAD issued 4 SG GETs instead of 2,
on the background load thread every rank waits for. The draft's bytes are ~1.3% of a
page (GLM-5.2: draft latent 576 B/token + indexer 128 B/token vs the target's
44,928 + 9,984), but it **doubled the ops**.

## The change: same keys, same bytes, half the ops

The draft is addressed by *exactly* the same page hashes and the same device slot
indices as the target (it rides the slots the target rode); only the key namespace
differs (`.draft_k` / `draft_indexer`). So its sub-keys can simply be **appended to
the target's scatter-gather batch** instead of getting their own round trip. Nothing
about what lands in L3 changes — which is why **R3 is preserved by construction**
(proven both directions by the roundtrip tests below: fused-written pages read back
through the standalone draft ABI and vice versa).

New sub-flag `SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT` (default **1** = fused). Set to 0
to revert to the increment-4/5 standalone draft ops for a single-variable A/B without
redeploying.

### mem_cache/hicache_storage.py (base hooks, inert defaults)
- `supports_fused_draft_device()` (NEW) → `False`; only dfkv overrides.
- `batch_set_v1_device` / `batch_get_v1_device` / `batch_set_v2_device` /
  `batch_get_v2_device` — added `with_draft: bool = False` to the signatures.

### managers/cache_controller.py
- `env_l2_bypass_fuse_draft()` (NEW module fn); `self.draft_fuse_requested`,
  `self.draft_device_fused` (`__init__`).
- `_maybe_enable_device_draft` — resets `draft_device_fused=False` at entry, and on
  success grants it when the env flag is on AND the backend advertises
  `supports_fused_draft_device()`. The enable log now states FUSED / unfused.
- `draft_rides_target_batch` (NEW property) = `draft_device_enabled and
  draft_device_fused` — the single switch every call site reads.
- `_draft_device_set` / `_maybe_device_draft_get` — return early when fused (the
  target's batch already carried the draft; a second op would be a pure duplicate).
- `_page_set_zero_copy_device` (dense backup), `load_device_direct` (sync read),
  `_run_device_get` (async read) — pass `with_draft=self.draft_rides_target_batch`.

### mem_cache/hybrid_cache/hybrid_cache_controller.py
- `_page_backup_device`, `load_device_direct`, `_run_device_get` — pass
  `with_draft=self.draft_rides_target_batch` to `batch_set_v2_device` /
  `batch_get_v2_device`.

### dfkv backend `integration/hicache/dfkv_hicache.py` (branch feat/hicache-device-direct-put)
- `supports_fused_draft_device()` (NEW) → True (capability probe).
- `_draft_device_flat(keys, device_indices, putting)` (NEW) — builds the draft's flat
  SG group (latent `@sg` sub-keys + , for a DSA draft, the `draft_indexer` sub-keys)
  from the SAME keys/indices, or returns None when it must not fuse (see the rank-skip
  guard below).
- `_fused_draft_or_fallback(...)` (NEW) — returns the group, or runs the standalone
  `batch_set/get_v1_device_draft` and returns None. Best-effort (swallows), because
  the SGLang side has already skipped its own call.
- `_kv_device_set` / `_kv_device_get` — new `extra=(sks, ptrs, sizes)` param appended
  to the one batch; the fold and the byte count are sliced back to the MAIN prefix so
  the target's per-page results and the `on_set`/`on_get` byte attribution are
  unchanged (the draft's bytes belong to no target metric).
- `batch_set_v1_device` / `batch_get_v1_device` / `batch_set_v2_device` /
  `batch_get_v2_device` — new `with_draft=False` param. `batch_set_v1_device` now
  delegates its body to `_kv_device_set` (identical logic, no duplication).
- `batch_set_v1_device_draft` / `batch_get_v1_device_draft` are UNCHANGED and remain
  the fallback + the existing tests' entry point.

## 🔴 The rank-skip guard (the one correctness trap)
Fusing merges two skip decisions into one, so it is only sound when they agree:
- target write: skipped on `is_mla and tp_rank != 0` (replicated MLA latent);
- draft latent write: skipped on `draft_sub == 1 and tp_rank != 0` — the draft's own
  MLA-ness, which is INDEPENDENT of the target's.

`_draft_device_flat(putting=True)` returns None when `(draft_sub == 1) != is_mla`
(e.g. MLA target + MHA draft), and the caller keeps the standalone draft op so its
semantics are untouched. GLM-5.2 (MLA target + MLA MTP draft) and dense+dense both
agree, so both fuse. The draft **indexer** skips on `is_mla` (it goes through the
shared `_sidecar_device_set`), i.e. identical to the target's, so once the latent
check passes the whole group is skip-compatible. **Reads have no rank skip anywhere**,
so a read always fuses.

## TP-consistency (unchanged invariant)
No collective sequence changed — this is entirely below the collective layer. The
fused write/read are local RDMA on the backup / background-load threads exactly as
before; the async read's balanced per-round done-MIN / alloc-MIN / pages-MIN
(increment 3) are untouched. The draft's per-page results are still discarded by the
caller (best-effort), so a draft miss cannot truncate the target's verified prefix —
asserted by `test_fused_read_of_absent_draft_still_serves_the_target`.

## Tests
dfkv `test/python/test_dfkv_hicache_device_direct.py` (+11, `-k hicache` 99 → **110**):
- real cache node: `test_fused_draft_write_is_readable_by_the_unfused_draft_path`
  and `test_unfused_draft_write_is_readable_by_the_fused_read` (**the R3-preservation
  proof, both directions** — a fused write is byte-exact through the standalone ABI
  and vice versa, which is the actual cross-restart shape),
  `test_fused_v2_device_roundtrip_all_four_components` (DSA: target latent + target
  indexer + draft latent + draft indexer all byte-exact in one put + one get),
  `test_fused_read_of_absent_draft_still_serves_the_target`.
- pure `TestFusedDraftGrouping` (7): which sub-keys get fused (DSA vs dense), decline
  with no draft pool, **the put-side rank-skip disagreement declines while the get
  fuses**, the fallback really issues the standalone op, and the op-count collapse —
  `{exist:3, put:3} → {exist:1, put:1}` and 2 SG GETs → 1.
- Mutation-checked: corrupting the fused key namespace fails 5 of the new tests.

SGLang `patched/tests/test_fused_draft_gate.py` (NEW, 9 tests, torch-free): the env
parser, the `draft_rides_target_batch` decision table, the standalone ops being its
exact complement (so the draft is never written twice nor lost), best-effort failure
swallowing, and a source-level guard that all four device call sites pass
`with_draft=`. These AST-extract and execute the REAL function bodies from the
shipped files rather than re-implementing them. `py_compile` clean on all patched
files; increments 3/4/6 unit tests still pass (13 / 9 / 11).

## Known limitations / deviations (increment 7)
- **The 13% R2 figure in the brief is not attributable to the draft WRITE.** It came
  from comparing two builds (increment 4's DSA-draft decline vs increment 5's
  device-direct DSA draft) in single runs, on a metric the campaign itself measured
  as noisy (R2 4G 38,098 vs 32G 45,339 = a 16% swing recorded as "single-run noise",
  台账 三期). The op-count analysis above is the mechanism this increment removes;
  the GPU A/B (`SGLANG_HICACHE_L2_BYPASS_FUSE_DRAFT=1` vs `0`, EAGLE, R1/R2/R3) is
  what will size it. It is a strict op reduction with identical L3 content, so the
  downside is bounded at "no measurable change".
- **Target latent and target indexer are still two ops.** Fusing those as well would
  halve the op count again even with the draft off, but it would change the draft-off
  arm too and muddy this A/B. Deliberately left as a follow-up.
- **SGLang side is py_compile + review + torch-free unit tests only** (no GPU on the
  dev box), as increments 1-6; the byte-exact claim is proven on the dfkv side
  against a real cache node.
- **A pre-existing asymmetry worth knowing** (not introduced or changed here): the
  draft latent's write rank-skip keys off the DRAFT pool's `sub`, while the draft
  indexer's keys off the TARGET's `is_mla`. They coincide for every shape currently
  deployed; a mixed MLA/MHA target-draft pair would write the draft indexer on ranks
  that skip the draft latent. The increment-7 guard declines fusion for exactly that
  pair, so it does not make the asymmetry worse.

---

# Increment 7.1 (hotfix) — the L3 marker state machine has no gaps

**Crash it fixes** (100k×16 C8, GLM-5.2 DSA + EAGLE + bypass, HICACHE=4, after
several restart→warmup→bench cycles; 8/8 ranks down, instance DOWN):

```
mem_cache/hiradix_cache.py, match_prefix
    assert self.l2_bypass and self._node_l3_resident(last_node), (
AssertionError: evicted non-backuped node 2 outside L2-bypass
```

## 🔴 New bind-mount file

`mem_cache/l3_marker_state.py` is NEW (torch-free, unit-tested; same treatment as
`device_page_meta.py`). **`hiradix_cache.py` imports it — mount it or the server
will not start.** The line is already in the bind-mount block at the top.

## Root cause (one function)

`_drop_l3_markers` cleared `l3_present` **unconditionally** but only detached the
node when it was childless:

```python
for n in reversed(nodes):
    n.l3_present = False                      # <-- always
    if n.value is None and len(n.children) == 0 and n.parent is not None:
        ...detach...                          # <-- only sometimes
```

A marker that had acquired a child — another request discovering a different
suffix under the same shared prefix inserts a SIBLING under it
(`_insert_helper_l3`) — therefore stayed in the tree with `value=None`,
`host_value=None`, `l3_present=False`, `l3_backed=False`. That is a **gap**: an
evicted node whose KV is nowhere. The bypass tree's invariant is "an evicted
in-tree node is backuped or L3-resident", and `match_prefix`'s climb asserted it.
Any later request matching *through* the gap (its descendants are live markers,
so this is the common case) crashed the scheduler.

Why only after several bench cycles: it needs (a) a warm L3 so discovery
materializes markers at all, (b) a branched tree — the 16 prompts must have split
the shared prefix into sibling branches, which only exists after the first pass,
and (c) one marker-drop event (0-verified / partial-verify / alloc-fail /
abort). Node id 2 in the traceback is the top-level shared-prefix marker, i.e.
the first node the first discovery created — which is exactly the node with the
most siblings.

## Fix

- **Root cause** — `l3_marker_state.prune_l3_markers` never leaves a gap. A
  marker that cannot be detached (has children) **keeps its L3 claim**; a marker
  another request's GET still owns (`in_flight_ids`, derived from
  `_bypass_load_state`) is left alone too — detaching it would have that load
  promote a node no longer in the tree and leak its GPU slots. Only nodes that
  are genuinely removable are cleared and popped.
- **Bounded fallback** (production must not die on cache metadata) — the assert
  is gone. `climb_evicted_chain` treats a gap as a hard **miss boundary**: the
  hit length restarts above it and the load-back start node moves above it. The
  same rule in `collect_loadable_chain` (both the sync `load_from_storage_device`
  and the async `_start_l3_async_load` chain walks, which had the same assert):
  hitting a gap discards everything collected below it. Gaps are counted
  (`_l3_gap_count`) and logged throttled (first, then every 100th).
- **Eviction** — `_evict_write_through` now skips a bypass device leaf that still
  has children. "Device leaf" means *all children evicted*, and in bypass those
  children are L3 markers still in the tree, so `_evict_regular` (which asserts
  leafness and deletes the node) would have been a second crash / an orphaned
  marker subtree. Unreachable with the flag off: a stock evicted child is
  backuped, which makes its parent backuped, so the parent takes
  `_evict_backuped`.

## The concurrency half: two requests loading one chain

The lead's reproduction narrowed it further — **concurrent** warmup (mixed
lengths, 4-way parallel) crashes every time, **serial** warmup never does. That
is the mechanism above, sharpened: with requests in flight simultaneously over a
shared prefix, one request's deeper markers are *children* of another's chain, so
a drop by any of them strands the shared ancestors. Serially, a chain is a clean
leaf chain and detaches without residue — which is exactly why v1 never
reproduced.

The same overlap has a second consequence — ⚠️ **inferred from code reading, NOT
observed**: the crash-era logs were lost when the container was rebuilt on the
fixed build, so this one is neither confirmed nor refuted by evidence (the
`match_prefix` assert above IS evidenced, 16 occurrences). It is fixed here
because the reasoning holds and the fix is cheap, not because it was seen.

`_start_l3_async_load` collects the chain by climbing from the deepest matched
node, so request 2's chain **contains request 1's still-in-flight markers**, and
a request whose match lands on in-flight markers takes the SYNC fallback
(`schedule_policy.py:1105` → `init_load_back` → `load_from_storage_device`) over
the very same nodes. When both loads land:

- the later one **overwrites `n.value`**, orphaning the slots the tree (and the
  other request) is already using — a device-memory bleed that can only appear
  under concurrency, and
- both register `ongoing_load_back[same node id]`, while `loading_check:1171`
  does `self.ongoing_load_back.pop(ack_id)` **once per ack** — so the second ack
  would pop a missing key (`KeyError`, scheduler down) or, at best, leave an
  unmatched `inc_lock_ref` pinning the node forever.

Fix: `l3_marker_state.plan_promotion` + `_apply_promotion` publish only nodes
that are **still evicted**. A node already carrying slots means a concurrent
request (or a recompute + `insert()`) published it first; the tree's slots win,
ours are freed, and publishing stops there so the freed remainder stays a
contiguous suffix of the buffer. If nothing is left to publish, no fence, no ack
and no lock are registered — which is precisely the case that would otherwise
double-register. Markers inside the verified prefix are never dropped by the
supersede break (the drop boundary follows verification, not publication).

## Safety: a gap can only shrink a match, never fake a hit

The verification of KV content is unchanged and stays where it was — at load
time (`consecutive_ok_pages` per page + the cross-rank MIN in
`_promote_l3_async_load` / `load_from_storage_device`). Markers were never
evidence of content, only a claim, so keeping a claim on an undetachable node
cannot make a page be served: the next load re-verifies it and, if it really is
gone, drops it again (detaching it that time, once the children are gone) and the
tokens recompute. In the other direction, every gap branch only ever REMOVES
nodes from the hit (`host_hit_length` restarts at 0 above the gap;
`nodes_to_load.clear()`), and `host_hit_length` is a scheduling budget hint
(`schedule_policy.py:1043`), so under-reporting is always safe. Nothing new is
marked device-resident anywhere in this change.

## TP-consistency (unchanged invariant)

No collective added, removed or reordered. The new early return in
`load_from_storage_device` (empty chain) and the gap branches are decided purely
from tree state, and the tree is TP-symmetric: every marker drop comes from a
POST-MIN decision (alloc MIN, verified-pages MIN) or from an abort, which is
broadcast to all ranks. So all ranks take the same branch, exactly as the
pre-existing `if not nodes_to_load:` early return already assumed.

## Changed files / functions

- `mem_cache/l3_marker_state.py` — **NEW**: `node_l3_backed/_present/_resident`,
  `climb_evicted_chain`, `collect_loadable_chain`, `plan_promotion`,
  `prune_l3_markers`.
- `mem_cache/hiradix_cache.py` — `match_prefix` (assert → gap-tolerant climb),
  `_drop_l3_markers` (delegates to `prune_l3_markers` + in-flight set),
  `load_from_storage_device` / `_start_l3_async_load` (chain walk → 
  `collect_loadable_chain`, asserts gone), `_promote_l3_async_load` /
  `load_from_storage_device` publish via `plan_promotion` + `_apply_promotion`
  (never clobber a concurrently published node), `_evict_write_through` (bypass
  children guard), `_log_l3_gap` + `_l3_gap_count` (new), `_node_l3_*` (thin
  delegates).

## Tests

`patched/tests/test_l3_marker_state.py` (NEW, 32 tests, torch-free): the drop
paths (childless / sibling branch / device-resident / in-flight / already
detached / l3_backed+present / partial-suffix) each asserted to leave **no gap in
the tree**; the climb with gaps at the top level, directly under the device
prefix, mid-chain and doubled; `collect_loadable_chain` discarding below a gap
and treating a hash-less marker as a gap; flag-off equivalence on a host-backed
chain; and the end-to-end production sequence (discover → sibling → failed drop →
later match); plus `plan_promotion` — full/partial verify, a node only partly
covered by the verified prefix, and the concurrency cases (supersede mid-chain,
fully superseded, superseded node keeps ITS slots, an evicted node behind a
supersede is neither published nor dropped). The pre-fix logic run against the
same fake tree reproduces `evicted non-backuped node 2 outside L2-bypass`
verbatim.

`py_compile` clean; increments 3/4/6/7 suites still pass (13 / 9 / 21 / 9).
Reproduction to re-run after the fix: `warmup.sh` v2 (concurrent, mixed lengths)
— the serial v1 never triggered any of this.

## Known limitations (increment 7.1)

- A kept claim on an undetachable marker costs one futile SG GET per request
  until a recompute rehydrates the node (`insert()` refills evicted nodes it
  walks through, which also re-marks it `l3_backed` on write-through). Bounded
  and self-healing, and it only happens when the page truly left L3.
- `_evict_write_through`'s skip can under-evict when many device leaves carry
  marker children. It returns fewer tokens; the caller retries/retracts as it
  already does. Never observed to matter at HICACHE=4 (markers resolve within a
  round or two), but it is the reason `_l3_gap_count` is worth watching.
- Not GPU-verified on this box (no torch) — same standing caveat as increments
  1-7.

---

# Increment 7.2 — bound the deferred device pin (GPU KV slot leak / pool wedge)

Fixes the long-soak wedge: after 8-10 rounds the instance stopped with

```
Prefill batch, #new-seq: 1, #new-token: 64, #cached-token: 0, token usage: 0.99,
#running-req: 0, #queue-req: 0
```

— **no requests running or queued, yet the KV pool 99% full**, health checks
failing, never recovering. No assert, no crash, no GPU-memory growth: the tokens
were not lost, they were *locked*.

## Root cause: the one lock released by an external event

Increment 1 made the GPU KV slot the RDMA source for its own device->L3 backup,
so it must stay unevictable until that PUT completes. The pin is therefore taken
at write-through enqueue (`hiradix_cache._write_backup_device` -> `inc_lock_ref`)
and released only at the **storage backup ack** (`_drain_backup` ->
`dec_lock_ref`), five hops later:

```
_write_backup_device        inc_lock_ref(node)          [scheduler]
  -> ongoing_write_through
  -> writing_check -> _finish_write_through_ack
  -> _write_backup_storage_device -> ongoing_backup[op.id]
  -> backup_thread_func -> _page_backup -> batch_set_v*_device   [backup thread]
  -> ack_backup_queue
  -> _drain_backup            dec_lock_ref(node)        [scheduler]
```

Stock never has this exposure. There the pin is released at the *D2H* ack — local,
bounded, CUDA-event-driven — and the storage backup only holds a `protect_host()`,
which cannot wedge device memory. Under bypass the same chain holds HBM, and it
has an unguarded break in the middle: `backup_thread_func`
(`managers/cache_controller.py:1838`) catches only `Empty`, so **any** exception
out of `_page_backup` / `_page_backup_device` (i.e. out of the dfkv
`batch_set_v1_device` / `batch_set_v2_device` call) kills the backup thread. From
that instant nothing ever acks again: every subsequent write-through pins its
slots permanently, `protected_size_` climbs monotonically, `evictable_size_` goes
to zero, and `PrefillAdder`'s `available_size() + evictable_size()` reaches zero
— which is precisely "zero requests, 99% usage, wedged". Abrupt onset after N
healthy rounds is the signature of a single fatal op, not of gradual pressure.

Two smaller holds on the same theme, both unbounded, were found alongside it:

* `_poll_l3_async_load` parked a request forever if its background GET never
  returned. A parked request is in the waiting queue, so it is never scheduled and
  never aborted — its GPU slots and ancestor pin had no release path at all.
* `_abort_async_load` waited 30s for the GET then **freed the slots regardless**.
  On a stall that hands slots the NIC is still writing back to the allocator:
  silent wrong-KV, not just a leak.

## Fix

**1. The ack becomes mandatory** (`cache_controller._run_device_backup`). Bypass
device backups run through a body that always acks, whatever the backend does.
Off bypass the method is not reached and a raising `_page_backup` still kills the
thread exactly as stock — flag-off behavior unchanged.

**2. Thread-death recovery** (`cache_controller.recover_dead_backup_thread`,
called from `drain_storage_control_queues`). If the thread dies anyway
(BaseException, kill), its backlog is force-acked and the thread respawned. Safe
*because* the thread is dead: no PUT can be in flight, so no NIC is reading the
slots being unpinned. FIFO drain keeps `ack_backup_queue` identical across ranks.

**3. Stale-pin reaper** (`hiradix_cache._reap_stale_device_pins`, deadline
`SGLANG_HICACHE_L2_BYPASS_PIN_TIMEOUT_S`, default 120s). Backstop for any lost ack
we have not diagnosed. Two invariants:

* *No RDMA under the reclaim.* A pin is released only if `try_cancel()` beats the
  backup thread's `try_start()` (`DevicePinCancelMixin`, one lock, mutually
  exclusive). An op whose PUT already began keeps its pin and is only counted —
  freeing it would let the NIC publish another request's KV under this page's
  hash. One stuck PUT blocks one op, not the backlog.
* *TP symmetry.* The reap count is the cross-rank MIN (4th slot of the existing
  `drain_storage_control_queues` all_reduce, so no extra collective) and
  `DevicePinLedger.reapable` yields stale pins oldest-op-id first — every rank
  releases the same pins in the same order, no `dec_lock_ref` divergence.

A reclaimed page never reached L3, so its `l3_backed` claim is cleared and it
recomputes: a hit-rate cost, not a correctness one.

**4. Async-load deadline + deferred slot free.** `_poll_l3_async_load` carries a
negated stall flag inside its existing MIN reduce (an OR: one stalled rank aborts
all, still one collective per round; `SGLANG_HICACHE_L2_BYPASS_LOAD_TIMEOUT_S`,
default 180s). `_abort_async_load` no longer blocks and no longer frees under an
in-flight GET — the task is parked in `_orphaned_device_tasks` and swept once
`done` fires, FIFO, under a cross-rank MIN (5th slot of the same all_reduce) so
allocator state stays identical across ranks. Deferred, never dropped.

## Observability

- `device_pin_census()` — pending backup ops, pinned tokens, oldest pin age,
  reclaimed/stuck counts, backup-thread alive + restart count, protected vs
  evictable size, in-flight loads.
- `_check_device_pin_health()` (every step, throttled 30s, no tree walk) warns
  once pins exceed 30s or 25% of the pool — the *leading edge* of a wedge.
- `_log_device_pin_audit()` fires when eviction falls short and answers the
  question directly: locked tokens split into *by pending L3 backup* / *by
  in-flight load* / **UNACCOUNTED** with sample node ids. Unaccounted > 0 is a
  `lock_ref` nothing will release — exactly the "lock_ref>0 with no in-flight
  task" signal, and where to look next time.

## Files

- `mem_cache/device_pin_ledger.py` (NEW, torch-free) — ledger + `audit_pins`.
  **Needs a new bind-mount** (added to the list above).
- `managers/cache_controller.py` — `DevicePinCancelMixin`,
  `DeviceStorageOperation`, `_run_device_backup`, `recover_dead_backup_thread`,
  health counters. `write_storage_device` now returns the operation (bypass-only
  method, single caller) so the tree can cancel it.
- `mem_cache/hybrid_cache/hybrid_cache_controller.py` — DSA
  `DeviceStorageOperation` (the indexer sidecar rides the same slots, so one pin
  covers both and one cancel protects both).
- `mem_cache/hiradix_cache.py` — ledger wiring, reaper, orphan sweep, async-load
  deadline, census/audit.

## Flag-off equivalence

Every hunk is behind `self.l2_bypass` or on a bypass-only method. The one
unconditional change is the `drain_storage_control_queues` all_reduce going from
3 to 5 elements: deliberately fixed-shape regardless of the flag, because
`l2_bypass` is resolved per-rank at attach and a flag-dependent tensor shape would
hang NCCL if ranks ever disagreed.

## Tests

`tests/test_device_pin_ledger.py` (NEW, 27 cases, pure python): ledger
bookkeeping; reap order is oldest-op-first (the TP-symmetry property); the
cancel/start arbitration is exclusive under 200 rounds of real thread
contention; and a lock-balance harness covering every exit path — normal ack,
lost ack reclaimed by deadline, late ack after a reap, reap after an ack,
in-flight PUT keeps its pin then releases normally, one stuck op not blocking the
backlog, MIN-capped reap taking the oldest prefix, mixed sequences, and
thread-death drain — each asserting the chain's `lock_ref` returns to zero and the
ledger empties. Plus `audit_pins` attribution incl. the orphan case.

`py_compile` clean. All 5 existing suites still pass (marker state / async read /
device page meta / fused draft gate / stub host pool).

Not GPU-verified on this box (no torch) — same standing caveat as increments 1-7.

---

# Increment 8 — one GET per chain (in-flight load dedup)

## Problem

Two concurrent requests that share a prefix each run their own exist-discovery,
each end up with the SAME evicted l3-marker chain, and each allocate GPU slots and
issue their own device SG GET for **identical pages**. One of those GETs is pure
waste, and it is waste on the TTFT critical path — the second request pays full L3
read latency for bytes already in flight.

This is not hypothetical: the async read (increment 3) parks a request for the
whole load window, so the window in which a sibling request can discover the same
chain is exactly as long as an L3 read of a 100k-token prompt.

## Fix

A request that finds any node of its chain already claimed by an in-flight load
becomes a **waiter**: no slots, no task, no GET. It keeps its discovery pending and
re-plans once the owner resolves — by then the chain is device-resident, so the
re-plan finds nothing to load and the request proceeds on the owner's bytes.

Two dicts carry it (`hiradix_cache.py`):

* `_bypass_inflight_owner: node id -> req_id`, written only **after** the alloc MIN
  (so every rank writes the same entries) and released on promote **and** abort.
* `_bypass_waiters: req_id -> (owner req_id, since)`.

Why an overlap is always a prefix: the chain is a contiguous parent-first path, so
a shared node implies a shared head. Loading only the suffix would leave a hole the
request could not be scheduled over anyway, which is why the waiter parks whole
rather than splitting the chain.

## 🔴 No new collective

`_bypass_inflight_owner` is a pure function of TP-symmetric state (claims are
written post-MIN; node ids are TP-symmetric), so **every rank takes the same park
branch** and a waiter round issues **zero** collectives. This is asserted at the
source level in the tests — a reduce accidentally added to the park branch would
unbalance the per-round sequence and hang the ring.

## Termination (every way a waiter can end)

| Owner outcome | Waiter |
|---|---|
| promotes | claims released at promote entry -> re-plans, chain resident, nothing to load |
| aborts / stalls | claims released in `_abort_async_load` -> re-plans, becomes the owner itself |
| never answers | `_device_load_timeout` (same deadline that bounds the owner) fires -> re-plans |
| waiter itself aborted | `release_aborted_request` drops the waiter record with the discovery |
| tree reset | both dicts cleared alongside `_bypass_load_state` |

No waiter chains are possible: a waiter holds no task, so it never appears in
`_bypass_inflight_owner`, so nobody can wait on a waiter.

The anchor pin taken at discovery is **held across the park** (not re-taken): the
park branch does not `dec_lock_ref`, and whichever exit is finally taken decs it
exactly once.

## Observability

`_bypass_stats` gains `dedup_parks` (duplicate GETs avoided), `dedup_pages_saved`
(the RDMA pages those GETs would have re-read), `dedup_waiters_now`, and
`dedup_wait_timeouts` — the last should stay **0**; nonzero means owners are
stalling, not that the dedup is misbehaving.

## Files

`mem_cache/hiradix_cache.py` only. **No new bind-mount file.**

## Tests

`tests/test_inflight_dedup.py` (NEW, 14 cases, pure python, AST-extracts the real
function bodies): no-overlap does not park; overlap parks and counts the saved
pages; the owner is the parent-most claim; the waiter resumes after the owner
promotes, after it aborts (and can then own the chain itself), and on deadline;
`timeout <= 0` means wait forever on a live owner (same convention as the rest of
the state machine); releases are owner-keyed so a re-claimed node is not stolen;
waiters never become owners. Plus source-level asserts: both exits release claims,
the park branch keeps the discovery pending, the park branch is collective-free,
aborted requests drop the waiter record, and reset clears both dicts.

`py_compile` clean. All 7 suites pass (marker state / async read / device page meta
/ fused draft gate / stub host pool / device pin ledger / inflight dedup).

Not GPU-verified yet — same standing caveat as increments 1-7.
