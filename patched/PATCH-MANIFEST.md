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
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/managers/cache_controller.py:$SGLANG/srt/managers/cache_controller.py:ro
-v /home/ketor/Code/git/ketor/sglang-l2bypass/patched/mem_cache/hybrid_cache/hybrid_cache_controller.py:$SGLANG/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py:ro
```

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
  benefit.
- **Marker accumulation / gate-#1 device budget** (increments 2/3) unchanged.
