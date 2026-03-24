# CUDA Implementation Plan: `qd_RTI_CUDA` (v4.1)

## 1. Overview

CUDA port of `qd_RTI_AVX2` — the Möller–Trumbore ray-triangle intersection algorithm
used in QuaDRiGa Ray Tracing (QRT). The AVX2 version processes one ray at a time across
all sub-meshes (outer loop = rays, inner loop = AABBs then faces). The CUDA version
inverts this: it processes all rays in parallel, using stream compaction between phases
to avoid scheduling empty work.

**Architecture choice:** Option B (global work queue with warp-level atomics) +
Approach 2 (64-bit `atomicMin` for closest-hit arbitration, u/v recomputed from winner).

**n_mesh alignment:** Unlike the AVX2 version (which requires `n_mesh` to be a multiple
of 8 due to 8-wide SIMD processing), the CUDA version has **no alignment requirement on
`n_mesh`**. Kernels 2/3 iterate faces scalar per thread; the broadcast pattern (all warp
threads read the same `Tx[i_face]`) does not depend on array length alignment.
`cudaMalloc` guarantees 256-byte base pointer alignment, which is sufficient.

**Key outputs per ray (same as AVX2):**

| Output | Type     | Description |
|--------|----------|-------------|
| `Wf`   | float    | Normalized distance of FBS hit (0 = origin, 1 = dest). `1.0` means no hit. |
| `Ws`   | float    | Normalized distance of SBS hit. Must be ≥ `Wf`. |
| `If`   | uint32   | Face index of FBS hit, **1-based**. `0` = no hit. |
| `Is`   | uint32   | Face index of SBS hit, **1-based**. `0` = no hit. |
| `hit_cnt` | uint32 | Total number of face hits in `[0, 1)` range (optional). |

---

## 2. Data Layout on GPU

### 2.1 Scene Data (resident for the entire run, uploaded once)

All arrays use SoA layout, matching the AVX2 version.

```
// Triangle mesh — 9 float arrays, each of length n_mesh
float *d_Tx, *d_Ty, *d_Tz;     // First vertex (V1) coordinates
float *d_E1x, *d_E1y, *d_E1z;  // Edge 1: V2 - V1
float *d_E2x, *d_E2y, *d_E2z;  // Edge 2: V3 - V1

// Sub-mesh index table — defines face ranges per AABB
// Length n_sub + 1. SMI[n_sub] = n_mesh (sentinel).
// Faces for AABB i = [SMI[i], SMI[i+1])
uint32_t *d_SMI;

// AABB bounds — 6 float arrays, each of length n_sub (no padding needed)
float *d_Xmin, *d_Xmax;
float *d_Ymin, *d_Ymax;
float *d_Zmin, *d_Zmax;
```

**Memory footprint:** `9 * n_mesh * 4 + (n_sub + 1) * 4 + 6 * n_sub * 4` bytes.
For a large scene (4M faces, 4000 AABBs): ~144 MB.

**SMI sentinel:** The host wrapper allocates a temporary array of length `n_sub + 1`,
copies the caller's `SMI[0..n_sub-1]`, and sets `SMI[n_sub] = n_mesh` before uploading.
This eliminates the `(aabb_idx == n_sub - 1) ? n_mesh : SMI[aabb_idx + 1]` branch
inside the face loop of Kernels 2/3, avoiding warp divergence when threads in a warp
process different AABBs.

### 2.2 Per-Ray Data (batched, double-buffered)

```
// Input — uploaded H→D per batch
float *d_Ox, *d_Oy, *d_Oz;     // Ray origin, length batch_size
float *d_Dx, *d_Dy, *d_Dz;     // Ray direction (origin → destination), length batch_size

// Output — downloaded D→H per batch
float    *d_Wf, *d_Ws;         // FBS/SBS normalized distance, length batch_size
uint32_t *d_If, *d_Is;         // FBS/SBS face index (1-based), length batch_size
uint32_t *d_hit_cnt;           // Hit count per ray (optional), length batch_size

// Intermediate — GPU-only, not transferred
uint64_t *d_fbs_packed;        // Packed (t_as_uint32 | face_idx), length batch_size
uint64_t *d_sbs_packed;        // Same format for SBS, length batch_size
uint32_t *d_hit_cnt_atomic;    // Accumulator for hit counting, length batch_size
```

### 2.3 Work Queue (GPU-only, per batch)

```
// Each entry is a (ray_idx, aabb_idx) pair.
// Two copies (A/B) are needed for CUB radix sort double-buffering.
uint32_t *d_wq_ray_idx_A;     // Length = queue_capacity
uint32_t *d_wq_ray_idx_B;     // Length = queue_capacity  (sort output buffer)
uint16_t *d_wq_aabb_idx_A;    // Length = queue_capacity  (uint16 — sufficient for n_sub ≤ 65535)
uint16_t *d_wq_aabb_idx_B;    // Length = queue_capacity  (sort output buffer)

uint32_t *d_queue_tail;        // Single atomic counter (device memory, own cache line)

// Hit flag per work item — set by Kernel 2, consumed by CUB compaction
uint8_t  *d_wq_had_hit;       // Length = queue_capacity. 1 = at least one face hit.

// Compacted work queue indices for Kernel 3 (output of CUB DeviceSelect)
uint32_t *d_compact_indices;   // Length = queue_capacity (worst case: all items had hits)
uint32_t *d_num_selected;      // Single value: number of compacted entries

// CUB temp buffer — shared between sort and compaction (allocated once, reused)
void     *d_cub_temp;          // Length = cub_temp_bytes (see Section 5.1)
```

**Queue tail placement:** Allocate `d_queue_tail` via its own `cudaMalloc` (or at a
128-byte aligned offset) to avoid false sharing with adjacent allocations.

**Queue capacity:** Pre-allocate `queue_capacity = batch_size * EST_AVG_HITS` where
`EST_AVG_HITS` is a conservative estimate (e.g. 8–12). Kernel 1 guards writes with
`pos < queue_capacity` (Section 4.1, 7.2) so overflow never causes out-of-bounds
writes. The unguarded `atomicAdd` on `queue_tail` records the true count. After
Phase 1, read `queue_tail` back to the host. If it exceeds `queue_capacity`,
synchronize both streams, `cudaFree` all queue-related buffers for both streams,
reallocate at `2 × h_queue_tail`, and re-run Kernels 0 and 1. In practice, with
typical scenes (1–5 AABB hits per ray), overflow should never happen.

**aabb_idx type:** `uint16_t` is sufficient for `n_sub ≤ 65535`. This halves the sort
key size, which directly reduces CUB radix sort cost. If `n_sub > 65535` is ever needed,
widen to `uint32_t` and adjust `end_bit` in the sort call (Section 6).

### 2.4 Memory Budget per Ray

| Item | Bytes |
|------|-------|
| Ray input (6 floats) | 24 |
| Ray output (Wf, Ws, If, Is, hit_cnt) | 20 |
| Intermediate (fbs_packed, sbs_packed, hit_cnt_atomic) | 20 |
| Work queue A (ray_idx:4 + aabb_idx:2) × ~8 entries | 48 |
| Work queue B / sort output (same) | 48 |
| had_hit + compact_indices (1 + 4) × ~8 entries | 40 |
| CUB temp share (~8 entries × 4 bytes, conservative) | 32 |
| **Total** | **~232 bytes/ray** |

The actual per-ray cost depends on the average number of AABB hits (here estimated at 8).
Section 5.1 uses a conservative formula that accounts for this variability.

---

## 3. Kernel Architecture

The pipeline has 6 kernels plus two CUB operations executed in sequence per batch:

```
┌─────────────────────────────────────────────────────┐
│ Kernel 0: init_per_ray_state                        │
│   Initialize fbs_packed, sbs_packed, hit_cnt to     │
│   sentinel values. Reset queue_tail to 0.           │
│   Launch: 1 thread per ray.                         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Kernel 1: aabb_test_and_enqueue  (Phase 1)          │
│   Each thread = one ray. Loops over all AABBs.      │
│   On hit: warp-ballot + atomicAdd to enqueue.       │
│   Output: work queue + queue_tail.                  │
└──────────────────────┬──────────────────────────────┘
                       │
                       │  D→H copy of queue_tail (4 bytes)
                       │  to determine sort + K2 launch size
                       │
┌──────────────────────▼──────────────────────────────┐
│ CUB sort: SortPairs on work queue by aabb_idx       │
│   uint16 keys, uint32 values, 0..16 bits only.      │
│   Groups threads accessing the same face data.      │
│   Critical for L2 locality on TU104 (4 MB L2).     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Kernel 2: moller_trumbore_fbs  (Phase 2a)           │
│   Each thread = one sorted work queue entry.        │
│   Loops over all faces in AABB.                     │
│   Finds local closest hit → 64-bit atomicMin on     │
│   fbs_packed[ray_idx].                              │
│   Also: atomicAdd on hit_cnt_atomic[ray_idx].       │
│   Sets d_wq_had_hit[i] = 1 if any face was hit.    │
│   Launch: queue_tail threads.                       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ CUB compact: DeviceSelect::Flagged                  │
│   Selects work queue indices where had_hit == 1.    │
│   Output: d_compact_indices, d_num_selected.        │
│   Eliminates warp divergence in Kernel 3.           │
│   D→H copy of num_selected (4 bytes) for K3 grid.  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Kernel 3: moller_trumbore_sbs  (Phase 2b)           │
│   Processes ONLY compacted work items (had hit).    │
│   Reads sorted ray_idx/aabb_idx via compact index.  │
│   Reads fbs_face_idx from fbs_packed[ray_idx].      │
│   Skips that face.                                  │
│   atomicMin on sbs_packed[ray_idx].                 │
│   Launch: num_selected threads (typically 20-50%    │
│   of queue_tail).                                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Kernel 4: finalize_outputs                          │
│   Unpacks fbs_packed/sbs_packed into Wf, Ws, If, Is │
│   Converts 0-based face index to 1-based.           │
│   Copies hit_cnt_atomic to hit_cnt (if requested).  │
│   Launch: 1 thread per ray.                         │
└─────────────────────────────────────────────────────┘
```

### Why Kernel 3 is a separate launch from Kernel 2

FBS must be fully resolved for ALL rays before SBS can be determined. If Kernels 2
and 3 were fused, a thread processing AABB-B for ray-7 might read an incomplete FBS
(from AABB-A for ray-7 which hasn't finished yet). The kernel boundary provides a
free global sync.

This does NOT have the float-equality race condition of the generic "two-pass"
approach discussed earlier: Kernel 3 compares face indices (integers), not distances.
The FBS face index is deterministic because `atomicMin` on the 64-bit packed value
`(t_as_uint32 << 32 | face_idx)` breaks distance ties by face index.

### Why Kernel 3 cannot be avoided

One might consider tracking both local_best and local_second_best within each Kernel 2
thread, writing both via atomicMin to `fbs_packed` and `sbs_packed`. However, this
does not work: a thread's local best might be the global FBS, and its local second-best
might be the global SBS — but equally, two different work items might contribute the
global FBS and global SBS respectively. The correct SBS can only be determined after
the global FBS is known, which requires a global sync (kernel boundary).

The Kernel 3 cost is mitigated by two optimizations:
1. **CUB stream compaction:** Between Kernels 2 and 3, `cub::DeviceSelect::Flagged`
   compacts the work queue to only those items where at least one face was hit. In
   typical scenes, 50–80% of AABB-ray pairs produce zero face hits, so Kernel 3
   processes only 20–50% of the original queue — with zero warp divergence, since all
   threads in a warp are guaranteed to have work. Without compaction, early-exiting
   threads (had_hit == 0) still occupy warp slots, reducing effective warp utilization
   to ~35% when the no-hit fraction is 65%.
2. **L2 warmth:** Face data and ray data are warm in L2 from Kernel 2. The compacted
   queue preserves the aabb_idx sort order from the original queue, so L2 locality
   is maintained.

### Why the work queue is sorted by default

See Section 6. On TU104 (RTX 2070, 4 MB L2), unsorted work queues cause severe L2
thrashing in Kernels 2/3 because threads in a warp access different face ranges. Sorting
by `aabb_idx` groups threads reading the same face data, giving up to 32× L2
amplification. This is the difference between memory-bound and compute-bound execution.
The CUB radix sort cost (~3–5 ms for 30M entries with 16-bit keys) is small relative to
the MT kernel savings (estimated 3–5× speedup). Sorting is therefore the default path;
an unsorted code path can be retained as a debug fallback.

### Scattered ray access pattern after sorting

After sorting by `aabb_idx`, threads in a warp process the same AABB but *different
rays*. Face data access is perfectly coalesced (broadcast from L2 — all 32 threads read
the same `Tx[i_face]`, `E1x[i_face]`, etc.). But ray data access (`Ox[ray_idx]`,
`Dy[ray_idx]`, etc.) is scattered — 32 threads load from 32 random ray indices. At
128-byte cache line granularity, that is 32 cache lines per ray component × 6
components = 192 cache lines per warp, versus 6 if rays were coalesced (~32×
amplification on ray loads).

This is a known cost of sorting by AABB and is almost certainly worth it: face data
volume per work item (9 floats × faces_per_AABB iterations) vastly exceeds ray data
volume (6 floats loaded once per work item). The scattered ray loads are a one-time
cost at the start of each work item, while face data is streamed for the entire face
loop. In Nsight Compute profiling, scattered ray loads will show up as L2 misses on
the global load metrics for Kernels 2/3 — this is expected and acceptable.

If profiling shows ray loads as a bottleneck (unlikely unless faces_per_AABB is very
small), consider: (a) caching ray data in shared memory at warp granularity, or
(b) sorting by `(aabb_idx, ray_idx)` which improves ray locality within each AABB
group at no extra sort cost — but this only helps if many rays hit the same AABB
(dense scenes).

**Shared memory ray caching (option a):** At the start of each work item in Kernels
2/3, all 32 threads in a warp cooperatively load their 6 ray floats into a shared
memory buffer (32 × 6 × 4 = 768 bytes per warp). The face loop then reads ray data
from shared memory instead of global memory, eliminating repeated L2 misses if the
compiler reloads ray values across loop iterations. With 8 warps per block (256
threads), this costs 6 KB of shared memory per block — well within TU104's 48 KB/SM
limit. Implement this only if Nsight Compute shows "Long Scoreboard" stalls on ray
loads in the face loop of Kernels 2/3.

---

## 4. Kernel Details

### 4.0 Kernel 0 — `init_per_ray_state`

```
Launch: <<<ceil(batch_size / 256), 256>>>

Per thread (one ray):
    if ray_idx >= current_batch_count: return  // safe: no warp-level ops
    fbs_packed[ray_idx] = 0x3F800000'FFFFFFFF   // t=1.0f, face=UINT32_MAX
    sbs_packed[ray_idx] = 0x3F800000'FFFFFFFF   // same sentinel
    hit_cnt_atomic[ray_idx] = 0

One thread (thread 0):
    *queue_tail = 0
```

The sentinel encodes `t = 1.0f` (as uint32: `0x3F800000`) in the upper 32 bits,
meaning "hit at destination = no hit". Face index `UINT32_MAX` ensures any real hit
wins the `atomicMin`.

### 4.1 Kernel 1 — `aabb_test_and_enqueue`

```
Launch: <<<ceil(batch_size / 256), 256>>>
Thread mapping: thread i = ray i
```

**Algorithm per thread:**

```
// All 32 lanes in a warp must participate in __ballot_sync.
// Invalid threads (past end of batch) participate but never enqueue.
bool valid = (ray_idx < current_batch_count)

if valid:
    load ray: ox, oy, oz, dx, dy, dz
    compute: dx_i = 1/dx, dy_i = 1/dy, dz_i = 1/dz

for i_sub = 0 .. n_sub-1:
    has_hit = false
    if valid:
        // Slab test (identical math to AVX2 version, scalar)
        t0_low  = (Xmin[i_sub] - ox) * dx_i
        t0_high = (Xmax[i_sub] - ox) * dx_i
        if t0_low > t0_high: swap(t0_low, t0_high)  // See NaN safety note below
        // ... same for Y, Z axes ...
        t_min = max(t0_low, t1_low, t2_low)          // See NaN safety note below
        t_max = min(t0_high, t1_high, t2_high)        // See NaN safety note below

        has_hit = (t_max > 0) && (t_max >= t_min) && (t_min <= 1.0)

    // Warp-level compaction (one atomicAdd per warp, not per thread)
    // All 32 lanes participate — invalid threads have has_hit = false
    hit_mask = __ballot_sync(0xFFFFFFFF, has_hit)
    if hit_mask == 0: continue

    lane = threadIdx.x % 32
    count = __popc(hit_mask)
    if lane == 0:
        warp_base = atomicAdd(queue_tail, count)
    warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0)
    local_offset = __popc(hit_mask & ((1u << lane) - 1))
    if has_hit:
        pos = warp_base + local_offset
        if pos < queue_capacity:       // guard against overflow
            wq_ray_idx_A[pos]  = ray_idx
            wq_aabb_idx_A[pos] = (uint16_t)i_sub
```

**Queue overflow safety:** The `atomicAdd` on `queue_tail` runs unconditionally (even
when `pos >= queue_capacity`) so that the host readback after Kernel 1 sees the *true*
count and can size the reallocation correctly. The `pos < queue_capacity` guard prevents
out-of-bounds writes. Dropped writes are harmless because the batch is re-run from
Kernel 0 after reallocation (Section 5.1). The guard is after `__ballot_sync`, so it
introduces no warp-divergence issues with the ballot.

**Partial warp safety:** The last warp in the last block may contain threads with
`ray_idx >= current_batch_count`. These threads must NOT return early — that would
cause `__ballot_sync(0xFFFFFFFF, ...)` to be UB (the mask promises all 32 lanes
participate). Instead, all lanes execute the full loop but invalid threads always set
`has_hit = false`, so they never enqueue anything. Note: this invariant requires the
block size to be a multiple of 32. Add a `static_assert(BLOCK_SIZE % 32 == 0)` where
the block size is defined.

**Memory access pattern for AABB bounds:** All threads in a warp read the same AABB
(same `i_sub` in the same loop iteration), so AABB data should be declared
`const float * __restrict__` to use the read-only data cache (L1 texture path). This
is a perfect broadcast pattern — 32 threads reading the same 6 floats. On compute
capability 7.5 (TU104), `const __restrict__` is sufficient — the compiler generates
`LDG` instructions automatically. No need for explicit `__ldg()` intrinsics.

**AABB loop bounds:** The loop runs over `n_sub` (exact count), not a padded value.
Unlike the AVX2 version which processes 8 AABBs in parallel and needs padding to a
multiple of 8, the CUDA kernel tests one AABB per loop iteration per thread. No
padding is needed; just use the exact count.

**NaN safety in the slab test:** The low/high swap and the t_min/t_max reduction MUST
be implemented using ordered comparisons and conditional assignment (ternary operator
or predicated move), NOT `fminf`/`fmaxf`. The reason: when a ray direction component
is zero, `dx_i = ±INF`. If `(Xmin - ox)` is also zero, the product `0 × INF = NaN`.
The AVX2 version uses ordered comparisons (`_CMP_GE_OQ`) with `_mm256_blendv_ps`:
NaN fails the comparison and propagates through to `t_min`/`t_max`, causing the final
conditions (`t_max > 0`, `t_max >= t_min`, `t_min <= 1.0`) to all evaluate to false
(ordered comparisons with NaN are false). The AABB is correctly rejected. With
`fminf(NaN, x) = x` (IEEE 754 minNum semantics), the NaN is silently discarded,
potentially causing a false-positive AABB hit. Such false positives are not incorrect
— the MT kernel would reject the faces — but they waste work queue slots and inflate
queue sizes unnecessarily. Concrete implementation:

```cuda
// CORRECT: NaN propagates, matching AVX2 behavior
float t_lo = (Xmin[i_sub] - ox) * dx_i;
float t_hi = (Xmax[i_sub] - ox) * dx_i;
if (t_lo > t_hi) { float tmp = t_lo; t_lo = t_hi; t_hi = tmp; }
// ... same for Y, Z ...
float t_min = (t0_lo > t1_lo) ? t0_lo : t1_lo;
t_min = (t_min > t2_lo) ? t_min : t2_lo;
float t_max = (t0_hi < t1_hi) ? t0_hi : t1_hi;
t_max = (t_max < t2_hi) ? t_max : t2_hi;

// WRONG: NaN silently discarded, may enqueue non-intersecting AABBs
float t_lo = fminf(...); float t_hi = fmaxf(...);    // DO NOT USE
float t_min = fmaxf(t0_lo, fmaxf(t1_lo, t2_lo));    // DO NOT USE
```

**Phase 1 / Phase 2 cost balance and `n_sub` tuning:** The cost structure of Phase 1
(AABB testing) and Phase 2 (Möller-Trumbore) is:

```
Phase 1 cost  ∝  n_ray × n_sub × C_aabb
Phase 2 cost  ∝  n_ray × avg_hits × faces_per_aabb × C_mt
```

where `C_aabb` is the per-test slab cost (~20 FLOPs), `C_mt` is the Möller-Trumbore
cost (~40 FLOPs + memory), `avg_hits` is the average number of AABB hits per ray, and
`faces_per_aabb ≈ n_mesh / n_sub`.

Ideally, Phase 1 and Phase 2 should be roughly balanced. If Phase 1 dominates, `n_sub`
is too large (too many AABB tests per ray). If Phase 2 dominates, `n_sub` is too small
(too many faces per AABB). The optimal balance depends on `avg_hits` and the relative
cost ratio `C_mt / C_aabb`, which differs between AVX2 and CUDA due to their different
memory hierarchies and execution models.

Setting ∂(total)/∂(n_sub) = 0 and noting that `avg_hits` is roughly proportional to
`n_sub` (more AABBs → more hits, though sub-linearly due to spatial locality) yields:

```
n_sub_optimal ≈ K × sqrt(n_mesh)
```

where `K` is a scene-dependent constant tuned empirically. For the AVX2 version,
`K ≈ 10` works well. For the CUDA version, `K` may differ because:

- Phase 1 on GPU is bandwidth-bound (AABB loads broadcast across warp) and benefits
  from high thread count. The per-AABB cost is lower relative to CPU.
- Phase 2 on GPU benefits enormously from the CUB sort (L2 locality). With sorting,
  Phase 2 is compute-bound, so `C_mt` is lower relative to the unsorted case.
- Both factors tend to shift the optimum toward *larger* `n_sub` (more but smaller
  AABBs), since Phase 1 is relatively cheaper on GPU and Phase 2 benefits from tighter
  spatial grouping.

**Recommendation:** Start with `K = 10` (same as AVX2). After Kernels 1 and 2 are
implemented, profile both phases with Nsight Compute on the Bertramshof scene and the
large outdoor scene. If Phase 1 takes >30% of total kernel time, reduce `K`. If
Phase 2 dominates at >80%, try increasing `K` to 15 or 20. The optimal `K` for CUDA
can be determined by sweeping `K` ∈ {5, 10, 15, 20, 25} on a representative workload
and measuring end-to-end time (including sort + compaction).

### 4.2 Kernel 2 — `moller_trumbore_fbs`

```
Launch: <<<ceil(queue_tail / 256), 256>>>
Thread mapping: thread i = sorted work queue entry i
```

After the CUB sort (Section 6), the sorted queue is in whichever buffer CUB's
DoubleBuffer designated as "current." Let `sorted_ray_idx` and `sorted_aabb_idx`
refer to the current buffers (A or B).

**Algorithm per thread:**

```
// Partial-warp safety: no warp-level intrinsics in this kernel,
// so a simple bounds check + early return is safe.
if i >= queue_tail: return

ray_idx  = sorted_ray_idx[i]
aabb_idx = sorted_aabb_idx[i]

// Load ray — scattered access, see "Scattered ray access pattern" in Section 3.
// These loads will miss L2 (32 different ray indices per warp) but are a one-time
// cost per work item, amortized over the face loop.
ox, oy, oz = Ox[ray_idx], Oy[ray_idx], Oz[ray_idx]
dx, dy, dz = Dx[ray_idx], Dy[ray_idx], Dz[ray_idx]

// Face range for this AABB (no branch — SMI has sentinel at n_sub)
face_start = SMI[aabb_idx]
face_end   = SMI[aabb_idx + 1]

local_best_t = 1.0f          // sentinel: no hit
local_best_face = UINT32_MAX
local_hit_count = 0

for i_face = face_start .. face_end - 1:
    // Load triangle data — broadcast from L2 (all threads in warp read same face)
    tx = ox - Tx[i_face];  ty = oy - Ty[i_face];  tz = oz - Tz[i_face]
    e1x = E1x[i_face]; e1y = E1y[i_face]; e1z = E1z[i_face]
    e2x = E2x[i_face]; e2y = E2y[i_face]; e2z = E2z[i_face]

    // Möller–Trumbore (same math as AVX2 version, scalar)
    // ... compute DT, U, V, W ...
    // DT = 1.0f / DT   (full precision, see Section 7.3)
    // U *= DT; V *= DT; W *= DT;

    // Hit condition (matches AVX2: >= 0, >= 0, <= 1, >= 0, < 1)
    if U >= 0 && V >= 0 && (U+V) <= 1 && W >= 0 && W < 1:
        // Force positive zero to prevent negative-zero atomicMin bug
        // (see Section 7.1 for explanation)
        W = fmaxf(W, 0.0f)

        local_hit_count++
        if W < local_best_t:
            local_best_t = W
            local_best_face = i_face

// Write results via atomics
if local_best_face != UINT32_MAX:
    uint64_t packed = ((uint64_t)__float_as_uint(local_best_t) << 32)
                    | (uint64_t)local_best_face
    atomicMin(&fbs_packed[ray_idx], packed)

if local_hit_count > 0:
    atomicAdd(&hit_cnt_atomic[ray_idx], local_hit_count)

// Set hit flag for CUB compaction before Kernel 3
wq_had_hit[i] = (local_hit_count > 0) ? 1 : 0
```

**On `__fdividef` vs full-precision division:** The AVX2 version uses full-precision
`_mm256_div_ps(r1, DT)`. The CUDA version should match. Use `1.0f / DT` (compiled to
full-precision `div.full.f32` with `--prec-div=true`, which is the default). Do NOT
use `__fdividef` — it flushes denormals and loses precision for near-grazing rays.
Alternatively, if a fast-approximate mode is desired later, this can be a compile-time
option.

**Tie-breaking rule:** The 64-bit packed encoding `(t_as_uint32 << 32 | face_idx)`
means `atomicMin` selects the smallest `t`, breaking exact distance ties by lowest
`face_idx`. This is deterministic across runs. Note that the AVX2 version breaks ties
differently: it uses an 8-lane parallel reduction where ties are broken by AVX lane
position (i.e., by memory layout within the current 8-face block). As a result, `If`
and `Is` may differ between AVX2 and CUDA when two or more faces have exactly the same
`W`. Both answers are valid — the tie-breaking choice is arbitrary. See Section 10.1
for the corresponding validation strategy.

**Face loop and memory coalescing:** After work queue sorting (Section 6), threads in
a warp process the same AABB, so they access the same face range. This creates a
broadcast pattern on face data — all 32 threads read the same `Tx[i_face]`,
`E1x[i_face]`, etc. — which is served from L2 (one cache line, broadcast to all
threads). Without sorting, threads access different face ranges and L2 is thrashed.

### 4.2a CUB Stream Compaction (between Kernels 2 and 3)

After Kernel 2 completes, `d_wq_had_hit[i]` contains 1 for work items that produced
at least one face hit. Before launching Kernel 3, compact the work queue to eliminate
items with zero hits. This removes warp divergence in Kernel 3 — every thread is
guaranteed to have meaningful work.

```
// Use CUB CountingInputIterator to avoid materializing a sequence array
cub::CountingInputIterator<uint32_t> iota(0);

cub::DeviceSelect::Flagged(
    d_cub_temp, cub_temp_bytes,       // reuse sort's temp buffer
    iota,                             // input: indices 0..queue_tail-1
    d_wq_had_hit,                     // flags: 1 = select
    d_compact_indices,                // output: selected indices
    d_num_selected,                   // output: count of selected items
    h_queue_tail,                     // num_items
    stream);

// Read compacted count to host (4 bytes — same pattern as queue_tail readback)
cudaMemcpyAsync(&h_num_compact, d_num_selected, 4, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

**Cost:** One pass over the work queue (~queue_tail bytes of flag reads + queue_tail
uint32 writes for selected indices). For 30M entries, this takes ~1–2 ms on TU104.
The CUB temp buffer is reused from the sort (it is more than large enough).

**Compacted queue ordering:** `DeviceSelect::Flagged` preserves input order. Since the
input was sorted by `aabb_idx`, the compacted queue retains L2 locality — adjacent
indices in `d_compact_indices` still point to the same AABB group.

**Alternative considered (hit-flag early exit):** Instead of compaction, Kernel 3 could
check `had_hit[i] == 0` and early-return. This avoids the compaction cost but causes
severe warp divergence: in a warp where 20 of 32 threads have no hit, all 32 threads
still occupy the warp for the entire face loop. With compaction, the same warp runs
12 threads at 100% utilization. The face loop is the most expensive computation in the
pipeline, so warp efficiency matters more than the ~1–2 ms compaction overhead.

### 4.3 Kernel 3 — `moller_trumbore_sbs`

```
Launch: <<<ceil(h_num_compact / 256), 256>>>
Thread mapping: thread j = compacted work item j
```

**Algorithm per thread:**

```
if j >= h_num_compact: return

// Look up original sorted queue position via compact index
uint32_t orig_idx = compact_indices[j]
ray_idx  = sorted_ray_idx[orig_idx]
aabb_idx = sorted_aabb_idx[orig_idx]

// Read the winning FBS face for this ray
uint64_t fbs = fbs_packed[ray_idx]
uint32_t fbs_face = (uint32_t)(fbs & 0xFFFFFFFF)

// Face range (no branch — SMI sentinel)
face_start = SMI[aabb_idx]
face_end   = SMI[aabb_idx + 1]

local_best_t = 1.0f
local_best_face = UINT32_MAX

for i_face = face_start .. face_end - 1:
    if i_face == fbs_face:
        continue

    // ... same Möller–Trumbore ...
    // ... same negative-zero clamp: W = fmaxf(W, 0.0f) ...

    if hit:
        if W < local_best_t:
            local_best_t = W
            local_best_face = i_face

// Write SBS
if local_best_face != UINT32_MAX:
    uint64_t packed = ((uint64_t)__float_as_uint(local_best_t) << 32)
                    | (uint64_t)local_best_face
    atomicMin(&sbs_packed[ray_idx], packed)
```

**Compaction guarantees:** Every thread in Kernel 3 has at least one face hit (from
Kernel 2). There is no warp divergence at the work-item level — all threads execute
the full face loop (minus the one `fbs_face` skip).

**One level of indirection:** The `compact_indices[j] → orig_idx → ray_idx / aabb_idx`
lookup adds one dependent global load per work item. This is negligible compared to the
face loop cost (tens to thousands of iterations). The alternative — compacting the
ray_idx and aabb_idx arrays directly — would require two separate `DeviceSelect`
calls or a custom fused kernel, for no measurable benefit.

**L2 locality preserved:** The compacted indices are a subsequence of 0..queue_tail-1,
preserving the aabb_idx sort order. Adjacent threads in a warp still process the same
AABB and read the same face data from L2.

**Why re-run the full face loop instead of storing partial results?** Storing per-work-item
second-best hits would require a large intermediate buffer (up to `queue_tail * 12` bytes).
Re-running Möller–Trumbore is cheaper than the memory traffic of writing and reading that
buffer. The face data and ray data will be warm in L2 from Kernel 2.

### 4.4 Kernel 4 — `finalize_outputs`

```
Launch: <<<ceil(batch_size / 256), 256>>>
Thread mapping: thread i = ray i
```

```
if ray_idx >= current_batch_count: return

uint64_t fbs = fbs_packed[ray_idx]
uint64_t sbs = sbs_packed[ray_idx]

float Wf_val = __uint_as_float((uint32_t)(fbs >> 32))
float Ws_val = __uint_as_float((uint32_t)(sbs >> 32))
uint32_t If_val = (uint32_t)(fbs & 0xFFFFFFFF)
uint32_t Is_val = (uint32_t)(sbs & 0xFFFFFFFF)

Wf[ray_idx] = Wf_val
Ws[ray_idx] = Ws_val
If[ray_idx] = (Wf_val < 1.0f) ? If_val + 1 : 0   // Convert to 1-based, 0 = no hit
Is[ray_idx] = (Ws_val < 1.0f) ? Is_val + 1 : 0
hit_cnt[ray_idx] = hit_cnt_atomic[ray_idx]          // if hit counting requested

// Debug invariant check (compile-time flag, e.g. #ifdef QD_RTI_DEBUG)
assert(Ws_val >= Wf_val)   // SBS must never be closer than FBS
```

**Debug invariant:** In debug builds, Kernel 4 should verify `Ws >= Wf` for every ray.
This catches logic errors in the K2/K3 pipeline (e.g. incorrect FBS face skipping in K3,
or a race on the packed atomics). Use a compile-time `#ifdef QD_RTI_DEBUG` guard so the
assertion is compiled out in release builds. Alternatively, use `assert()` which CUDA
supports in device code when compiled with `-G` (device debug).

---

## 5. Batching and Double-Buffered Transfer

### 5.1 Batch Size Calculation

At startup, after uploading scene data, calculate a conservative batch size that leaves
ample headroom for driver allocations, CUB temporaries, and fragmentation:

```
cudaMemGetInfo(&free_mem, &total_mem)

// --- Per-ray memory (both buffers counted separately for double-buffering) ---
const int E = EST_AVG_HITS;  // 10 (generous estimate for avg AABB hits per ray)

per_ray_bytes = 6 * 4         // input: Ox,Oy,Oz,Dx,Dy,Dz
              + 5 * 4         // output: Wf,Ws,If,Is,hit_cnt
              + 2 * 8 + 4     // intermediate: fbs_packed, sbs_packed, hit_cnt_atomic
              + E * (4 + 2)   // work queue A: ray_idx(4) + aabb_idx(2) per entry
              + E * (4 + 2)   // work queue B: sort output buffers (same size)
              + E * 1         // had_hit flags
              + E * 4         // compact_indices
              ;
// = 24 + 20 + 20 + 60 + 60 + 10 + 40 = 234 bytes/ray

// --- CUB temp buffer (shared between sort and compaction) ---
// Query the actual requirement at runtime instead of a fixed estimate.
// Use a dummy call with nullptr to get the required size for each operation,
// then take the maximum. This typically yields 2–8 MB (much less than a
// fixed 32 MB estimate), recovering ~50 MB for additional batch capacity.
size_t cub_sort_temp = 0, cub_compact_temp = 0;
cub::DeviceRadixSort::SortPairs(nullptr, cub_sort_temp,
    (uint16_t*)nullptr, (uint16_t*)nullptr,
    (uint32_t*)nullptr, (uint32_t*)nullptr,
    max_queue_estimate, 0, 16, (cudaStream_t)0);
cub::DeviceSelect::Flagged(nullptr, cub_compact_temp,
    cub::CountingInputIterator<uint32_t>(0),
    (uint8_t*)nullptr, (uint32_t*)nullptr, (uint32_t*)nullptr,
    max_queue_estimate, (cudaStream_t)0);
cub_temp_per_stream = max(cub_sort_temp, cub_compact_temp)
// Add 20% margin for CUB version variance
cub_temp_per_stream = (size_t)(cub_temp_per_stream * 1.2)

// --- Use only 75% of free memory ---
// Leaves 25% headroom for: CUDA runtime overhead, driver allocations,
// fragmentation from cudaMalloc, and any under-estimates above.
usable = free_mem * 0.75

// Double-buffering: 2 complete buffer sets + 2 CUB temp buffers
batch_size = (usable - 2 * cub_temp_per_stream) / (2 * per_ray_bytes)
batch_size = min(batch_size, n_ray)
batch_size = (batch_size / 256) * 256   // align to block size
```

**Worked example (RTX 2070, 8 GB, Bertramshof scene):**

```
free_mem ≈ 7800 MB  (after 144 MB scene upload + runtime overhead)
usable   = 7800 * 0.75 = 5850 MB
2 × cub_temp ≈ 10–16 MB  (queried at runtime, typically 5–8 MB per stream)
batch_size = (5850 - 16) MB / (2 × 234 bytes) ≈ 12.5M rays
```

For Bertramshof (10M rays): single batch. For the large outdoor scene (100M rays):
~8 batches. This is more batches than the v1 plan (which tried to maximize batch size),
but the conservative approach avoids OOM on memory-constrained GPUs.

**Queue capacity per stream:**

```
queue_capacity = batch_size * E    // 10 × batch_size
```

If `h_queue_tail` exceeds `queue_capacity` after Kernel 1, synchronize *both* streams,
reallocate *all* queue-related buffers (both streams) at `2 × h_queue_tail`, and re-run
the current batch from Kernel 0. Both streams must be synchronized because the other
stream's estimate is likely also wrong.

### 5.2 Double-Buffered Async Pipeline

Use two CUDA streams to overlap H→D transfer of batch N+1 with compute on batch N.
Each stream has its own set of per-ray device buffers, work queue buffers, and CUB
temp buffer.

```
stream[0], stream[1]          // Two CUDA streams
d_ray_input[2]                // Two sets of ray input buffers
d_ray_output[2]               // Two sets of output buffers
d_intermediate[2]             // Two sets of intermediate buffers
d_work_queue[2]               // Two work queues (A/B pairs + had_hit + compact)
d_cub_temp[2]                 // Two CUB temp buffers
h_ray_input_pinned[2]         // Pinned host memory for async transfer
h_ray_output_pinned[2]        // Pinned host memory for async transfer
```

**Pipeline (pseudocode):**

```
// Stage batch 0 into pinned memory BEFORE the loop
copy user ray arrays [0 .. batch_size) → h_ray_input_pinned[0]   // CPU memcpy

for batch_idx = 0 .. n_batches - 1:
    s = batch_idx % 2                // current stream index
    sp = (batch_idx - 1) % 2         // previous stream index

    // Wait for previous batch's D→H to finish (if any)
    if batch_idx > 0:
        cudaStreamSynchronize(stream[sp])
        copy h_ray_output_pinned[sp] → user output arrays (CPU memcpy)

    // H→D: upload ray data for this batch
    cudaMemcpyAsync(d_ray_input[s], h_ray_input_pinned[s],
                    batch_bytes_in, cudaMemcpyHostToDevice, stream[s])

    // Kernel 0: init state
    init_per_ray_state<<<grid0, 256, 0, stream[s]>>>(...)

    // Kernel 1: AABB test + enqueue
    aabb_test_and_enqueue<<<grid1, 256, 0, stream[s]>>>(...)

    // Read queue_tail back to host (small sync point, 4 bytes)
    cudaMemcpyAsync(&h_queue_tail[s], d_queue_tail[s], 4,
                    cudaMemcpyDeviceToHost, stream[s])
    cudaStreamSynchronize(stream[s])  // need queue_tail to size sort + Kernel 2 launch

    // Check for queue overflow (see Section 5.1 for realloc protocol)
    if h_queue_tail[s] > queue_capacity[s]:
        handle_overflow(...)  // sync both streams, realloc both, re-run

    // Issue sort + K2 + compact (all async — GPU will be busy for ~10-50 ms)
    cub_sort_work_queue(stream[s], h_queue_tail[s], ...)

    grid2 = ceil(h_queue_tail[s] / 256)
    moller_trumbore_fbs<<<grid2, 256, 0, stream[s]>>>(...)

    cub_compact_work_queue(stream[s], h_queue_tail[s], ...)
    cudaMemcpyAsync(&h_num_compact[s], d_num_selected[s], 4,
                    cudaMemcpyDeviceToHost, stream[s])

    // --- CPU staging window: GPU is busy with sort + K2 + compact ---
    // Stage NEXT batch into pinned memory (overlaps with GPU work above)
    if batch_idx + 1 < n_batches:
        next_s = (batch_idx + 1) % 2
        copy user ray arrays [next_batch_start .. next_batch_end) → h_ray_input_pinned[next_s]

    cudaStreamSynchronize(stream[s])  // need num_compact for Kernel 3 grid

    // Kernel 3: Möller–Trumbore SBS (only compacted work items)
    if h_num_compact[s] > 0:
        grid3 = ceil(h_num_compact[s] / 256)
        moller_trumbore_sbs<<<grid3, 256, 0, stream[s]>>>(...)

    // Kernel 4: finalize
    finalize_outputs<<<grid0, 256, 0, stream[s]>>>(...)

    // D→H: download results
    cudaMemcpyAsync(h_ray_output_pinned[s], d_ray_output[s],
                    batch_bytes_out, cudaMemcpyDeviceToHost, stream[s])

// Drain last batch
cudaStreamSynchronize(stream[(n_batches - 1) % 2])
copy last h_ray_output_pinned → user output arrays
```

**Sync points per batch:** Two small D→H copies + syncs:
1. `queue_tail` readback (4 bytes, ~4 µs) — needed to size the sort and Kernel 2 grid.
2. `num_selected` readback (4 bytes, ~4 µs) — needed to size the Kernel 3 grid.

Both are unavoidable without CUDA Dynamic Parallelism. Total sync overhead: ~8 µs per
batch, negligible relative to kernel execution.

**Optional: eliminate the `num_selected` sync.** Instead of reading `h_num_compact`
back to the host, launch Kernel 3 with the same grid size as Kernel 2
(`ceil(queue_tail / 256)`) and add a device-side bounds check at the top:

```
if j >= *d_num_selected: return   // read d_num_selected from device memory
```

This removes the second `cudaStreamSynchronize` at the cost of launching idle threads
(up to ~80% idle when the no-hit fraction is 65%). Whether this is net positive depends
on how much the sync stall costs relative to the wasted launch slots. Worth benchmarking
on the Bertramshof scene; likely beneficial for multi-batch workloads where pipeline
bubbles accumulate.

**Note on pinned memory staging:** The user's ray data lives in unpinned host arrays
(`Ox`, `Oy`, ...). The `memcpy` from user arrays into `h_ray_input_pinned` is a CPU
operation that must complete before `cudaMemcpyAsync` begins for that batch. Batch 0
is staged before the loop. Subsequent batches are staged in the window between the
`queue_tail` sync and the `num_selected` sync — during this window, the GPU is busy
with sort + Kernel 2 + compaction (~10–50 ms), which is ample time for the CPU to
memcpy the next batch (~30–50 ms for 12M rays × 24 bytes at DDR4 bandwidth). This
eliminates the GPU idle time that would occur if staging happened before the current
batch's H→D transfer.

---

## 6. Work Queue Sorting

### Problem

In Kernels 2 and 3, threads in a warp typically process different AABBs, meaning they
read different face data → no L2 locality across threads in a warp. On TU104 (RTX 2070,
4 MB L2 shared across all SMs), this is catastrophic: for an AABB with 1000 faces, each
thread streams 36 bytes/face × 1000 = 36 KB. With 32 threads per warp hitting different
AABBs, that is 1.15 MB of unique data per warp, thrashing the 4 MB L2.

### Solution

After Phase 1, sort the work queue by `aabb_idx` (primary key only — no secondary
sort on `ray_idx`). This groups all work items for the same AABB together. Adjacent
threads in a warp then read the same face data → broadcast from L2, up to 32×
amplification.

**Implementation:** Use `cub::DeviceRadixSort::SortPairs` with `uint16_t` keys
(`aabb_idx`) and `uint32_t` values (`ray_idx`). Use CUB's `DoubleBuffer` wrapper to
avoid explicit management of input/output swaps:

```cpp
cub::DoubleBuffer<uint16_t> d_keys(d_wq_aabb_idx_A, d_wq_aabb_idx_B);
cub::DoubleBuffer<uint32_t> d_values(d_wq_ray_idx_A, d_wq_ray_idx_B);

cub::DeviceRadixSort::SortPairs(
    d_cub_temp, cub_temp_bytes,
    d_keys, d_values,
    h_queue_tail,
    0,                     // begin_bit: sort from bit 0
    num_aabb_bits,         // end_bit: ceil(log2(n_sub)), max 16
    stream);

// After sort, current buffers are:
//   sorted_aabb_idx = d_keys.Current()
//   sorted_ray_idx  = d_values.Current()
// Kernel 2 reads from these pointers.
```

**Sort bit range:** Only `ceil(log2(n_sub))` bits need to be sorted. For `n_sub = 200`
(Bertramshof), that is 8 bits → 2 radix passes. For `n_sub = 4000` (large outdoor),
12 bits → 3 passes. CUB processes 4 bits per pass, so:

| n_sub | Bits to sort | Radix passes |
|-------|-------------|-------------|
| ≤ 16  | 4 | 1 |
| ≤ 256 | 8 | 2 |
| ≤ 4096 | 12 | 3 |
| ≤ 65536 | 16 | 4 |

Compared to the v1 plan's 64-bit composite key sort (16 passes), this is a 4–8×
reduction in sort cost.

**Why no secondary sort on `ray_idx`:** The v1 plan proposed sorting a 64-bit composite
key `(aabb_idx << 32 | ray_idx)`, which additionally sorts by `ray_idx` within each
AABB group. This secondary sort does not improve L2 locality: face data access depends
only on `aabb_idx`, and ray data access is scattered regardless (see Section 3,
"Scattered ray access pattern"). The secondary sort doubles the key size and quadruples
the number of radix passes for zero benefit.

**Cost estimates (TU104, revised):**

| Queue size | Sort passes (n_sub ≤ 4096) | Sort time (approx) |
|------------|---------------------------|-------------------|
| 3M | 3 | ~0.5 ms |
| 30M | 3 | ~3 ms |
| 90M | 3 | ~9 ms |

**CUB temp buffer with DoubleBuffer:** When using `DoubleBuffer`, the caller provides
both A and B arrays. CUB's temp buffer only needs space for per-block histograms
(proportional to `num_blocks × radix_size`), which is typically 2–8 MB. This is much
smaller than without `DoubleBuffer` (where CUB allocates the alternate arrays
internally in temp).

**This is the default code path**, not an optional optimization. An unsorted path can
be retained as a debugging/profiling fallback behind a compile-time flag.

---

## 7. CUDA-Specific Implementation Notes

### 7.1 Float-to-Int Packing for atomicMin

```cuda
// Pack: distance (float) + face index (uint32) → uint64 for atomicMin
__device__ __forceinline__
uint64_t pack_hit(float t, uint32_t face_idx) {
    // IMPORTANT: t must be non-negative. Caller must clamp with fmaxf(t, 0.0f)
    // to prevent negative zero (-0.0f) from breaking the comparison.
    //
    // Positive floats are order-preserving under uint32 comparison:
    //   a < b  ⟺  __float_as_uint(a) < __float_as_uint(b)
    // But __float_as_uint(-0.0f) = 0x80000000, which is LARGER than
    // __float_as_uint(1.0f) = 0x3F800000. A valid hit at t = -0.0f would
    // lose the atomicMin against the sentinel, silently dropping the hit.
    //
    // IEEE 754: fmaxf(-0.0, 0.0) returns +0.0 (per maxNum), so this is safe.
    return ((uint64_t)__float_as_uint(t) << 32) | (uint64_t)face_idx;
}

// Unpack
__device__ __forceinline__
float unpack_t(uint64_t packed) {
    return __uint_as_float((uint32_t)(packed >> 32));
}

__device__ __forceinline__
uint32_t unpack_face(uint64_t packed) {
    return (uint32_t)(packed & 0xFFFFFFFF);
}
```

**Correctness argument:** `t` is always in `[0.0, 1.0)` after the `fmaxf` clamp.
Positive IEEE 754 floats have the property that `a < b` ⟺ `__float_as_uint(a) <
__float_as_uint(b)`. Since the upper 32 bits of the packed value contain `t`,
`atomicMin` on the packed uint64 selects the smallest `t`, breaking ties by smallest
`face_idx`. This is deterministic.

**Why negative zero is a concern:** The Möller–Trumbore computation produces
`W = numerator / DT`. If the numerator evaluates to `-0.0f` (possible via FMA
cancellation) and `DT` is positive, `W = -0.0f`. The hit condition `W >= 0` is true
for `-0.0f` (IEEE 754 mandates `-0.0 == +0.0`), so the hit passes validation. But
`__float_as_uint(-0.0f) = 0x80000000 > 0x3F800000 = __float_as_uint(1.0f)`, so the
packed value is larger than the sentinel, and the `atomicMin` keeps the sentinel. The
hit is silently dropped. The `fmaxf(W, 0.0f)` clamp in the hit-condition block
(Section 4.2) prevents this.

### 7.2 Warp-Level Queue Compaction

```cuda
__device__ __forceinline__
void enqueue_warp(bool has_hit, uint32_t ray_idx, uint16_t aabb_idx,
                  uint32_t *wq_ray, uint16_t *wq_aabb, uint32_t *queue_tail,
                  uint32_t queue_capacity) {
    // All 32 lanes in the warp MUST execute this function.
    // Invalid threads (past batch end) must pass has_hit = false.
    uint32_t hit_mask = __ballot_sync(0xFFFFFFFF, has_hit);
    if (hit_mask == 0) return;

    uint32_t lane = threadIdx.x & 31;
    uint32_t count = __popc(hit_mask);
    uint32_t warp_base;

    if (lane == 0)
        warp_base = atomicAdd(queue_tail, count);
    warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

    uint32_t local_offset = __popc(hit_mask & ((1u << lane) - 1));
    if (has_hit) {
        uint32_t pos = warp_base + local_offset;
        if (pos < queue_capacity) {   // guard: prevent OOB writes on overflow
            wq_ray[pos]  = ray_idx;
            wq_aabb[pos] = aabb_idx;
        }
        // atomicAdd already recorded the true count for host-side overflow
        // detection. Dropped writes are re-done after reallocation (Section 5.1).
    }
}
```

**Partial warp safety:** `__ballot_sync(0xFFFFFFFF, ...)` requires all 32 lanes to
participate. The final warp in the last block may contain threads with `ray_idx >=
current_batch_count`. These threads must NOT early-return before reaching `__ballot_sync`.
Instead, they pass `has_hit = false` and participate in the ballot (contributing 0 to
the hit mask). See Section 4.1 for the full pattern.

### 7.3 Division Precision

The AVX2 version offers three modes (fast `rcp`, Newton-Raphson, full `div`). The
default is full precision. Match this in CUDA:

- **Full precision (default):** `1.0f / DT` → compiles to `div.full.f32`. Ensure
  `--prec-div=true` (nvcc default).
- **Fast mode (optional):** `__fdividef(1.0f, DT)` or `__frcp_rn(DT)`. Only enable
  via a compile-time flag if the user accepts reduced accuracy.

### 7.4 Block Size Selection

- **Kernels 0, 1, 4:** 256 threads/block. These are memory-bandwidth-bound (ray loads,
  AABB loads). 256 gives good occupancy without excessive register pressure.
- **Kernels 2, 3:** 256 threads/block. These are compute-bound (Möller–Trumbore FMA
  chains) after sorting. 256 is a good starting point. Profile with Nsight Compute and
  test 128 vs 256 vs 512 — the optimal choice depends on register usage and occupancy.
- **All kernels:** Block size must be a multiple of 32. Add `static_assert(BLOCK_SIZE %
  32 == 0)` at the definition site to prevent accidental misconfiguration.

**`__launch_bounds__` on Kernels 2/3:** Annotate both MT kernels with
`__launch_bounds__(256, MIN_BLOCKS_PER_SM)` to give the compiler an explicit register
budget. On TU104 (64K registers per SM), `MIN_BLOCKS_PER_SM = 2` yields a budget of
`65536 / (256 × 2) = 128` registers per thread — more than enough for the MT kernel
(~30 registers) but prevents the compiler from over-allocating and spilling. Without
this annotation, `nvcc` may assume worst-case occupancy targets and make suboptimal
spill decisions. Example:

```cuda
__global__ void __launch_bounds__(256, 2)
moller_trumbore_fbs(...) { ... }

__global__ void __launch_bounds__(256, 2)
moller_trumbore_sbs(...) { ... }
```

After profiling, adjust `MIN_BLOCKS_PER_SM` based on achieved occupancy. If Nsight
shows >80 registers per thread, increase to `3` or `4` to force the compiler to use
fewer registers and improve occupancy.

### 7.5 Read-Only Data Cache and Aliasing

Declare all scene data pointers as `const float * __restrict__` in kernel signatures.
On compute capability 7.5 (TU104 / RTX 2070), this is sufficient for the compiler to
route loads through the read-only data cache (L1 texture path) via `LDG` instructions.
No explicit `__ldg()` intrinsics are needed.

**Per-ray data pointers must also be `__restrict__`.** In Kernels 2 and 3, ray data
(`d_Ox`, `d_Oy`, ..., `d_Dz`) is read-only but shares the same address space as the
write targets (`d_fbs_packed`, `d_sbs_packed`, `d_hit_cnt_atomic`). Without
`__restrict__`, `nvcc` cannot prove that ray loads don't alias with the atomic stores
and may conservatively reload ray values from L2 after each `atomicMin`/`atomicAdd`
instead of keeping them in registers. Since ray data is loaded once per work item and
used across the entire face loop (tens to thousands of iterations), this can cause
measurable regressions. Mark all read-only pointer parameters `const ... * __restrict__`
in every kernel (including Kernels 0, 1, and 4) for consistency.

---

## 8. Function Signature

The CUDA wrapper should match the AVX2 signature as closely as possible, with
additional parameters for GPU selection:

```cpp
void qd_RTI_CUDA(
    // Same mesh/AABB/ray/output parameters as qd_RTI_AVX2
    const float *Tx, const float *Ty, const float *Tz,
    const float *E1x, const float *E1y, const float *E1z,
    const float *E2x, const float *E2y, const float *E2z,
    const size_t n_mesh,
    const unsigned *SMI,
    const float *Xmin, const float *Xmax,
    const float *Ymin, const float *Ymax,
    const float *Zmin, const float *Zmax,
    const size_t n_sub,
    const float *Ox, const float *Oy, const float *Oz,
    const float *Dx, const float *Dy, const float *Dz,
    const size_t n_ray,
    float *Wf, float *Ws,
    unsigned *If, unsigned *Is,
    unsigned *hit_cnt,          // optional, nullptr to skip
    // CUDA-specific
    int gpu_id = 0              // GPU device index
);
```

All input/output pointers are **host memory**. The function handles all GPU allocation,
H↔D transfers, batching, and cleanup internally. The caller sees the same interface as
the AVX2 version.

**GPU selection:** The wrapper calls `cudaSetDevice(gpu_id)` at the top, before any
CUDA API calls.

**n_mesh alignment:** Unlike `qd_RTI_AVX2`, the CUDA version does **not** require
`n_mesh` to be a multiple of 8. Any `n_mesh >= 1` is valid (see Section 1). The calling
code can pass unpadded mesh arrays directly.

**Early returns:** If `n_ray == 0`, return immediately (no GPU work). If `n_sub == 0`
or `n_mesh == 0`, zero-fill all outputs (`Wf = Ws = 1.0f`, `If = Is = 0`,
`hit_cnt = 0`) on the host and return — no GPU allocation needed.

**n_sub range check:** If `n_sub > 65535`, throw `std::invalid_argument` (or fall back
to a `uint32_t` AABB index code path if one exists). Without this check, the
`(uint16_t)i_sub` cast in Kernel 1's enqueue silently truncates AABB indices ≥ 65536,
causing incorrect face range lookups in Kernels 2/3. This is a hard requirement as
long as the work queue uses `uint16_t` for `aabb_idx` (Section 2.3).

**Resource cleanup (RAII):** The double-buffered pipeline allocates many device buffers
(scene data, 2× per-ray buffers, 2× work queues, 2× CUB temp, pinned host memory).
Use a single RAII wrapper struct (e.g. `CudaBuffers`) whose destructor calls `cudaFree`
/ `cudaFreeHost` on all allocations. This ensures cleanup on:
- Normal completion (destructor at scope exit)
- `std::invalid_argument` throws from range checks
- CUDA API failures (if the calling code wraps in try/catch)
- Queue overflow reallocation (old buffers freed before new ones allocated)

The struct should hold all device and pinned-host pointers as members, initialized to
`nullptr`. The destructor iterates all members and calls `cudaFree`/`cudaFreeHost` on
non-null pointers.

**SMI handling:** Internally, the wrapper allocates a host array of length `n_sub + 1`,
copies the caller's `SMI[0..n_sub-1]`, sets entry `[n_sub] = (uint32_t)n_mesh`, and
uploads that to `d_SMI`.

**Face range disjointness:** The caller is responsible for ensuring that the face ranges
`[SMI[i], SMI[i+1])` are disjoint (no face belongs to two AABBs). If ranges overlap,
`hit_cnt` will double-count faces, and FBS/SBS may select the same face from different
AABB work items. The AVX2 version has the same assumption.

---

## 9. Typical Workload Estimates

| Scene | n_mesh | n_sub | n_ray | Batches (8 GB) | Queue size | Phase 2 work items |
|-------|--------|-------|-------|----------------|------------|-------------------|
| Small indoor | 30K | 30 | 1M | 1 | ~3M | ~3M |
| Medium office (Bertramshof) | 200K | 200 | 10M | 1 | ~30M | ~30M |
| Large outdoor | 4M | 4000 | 100M | ~8 | ~90M/batch | ~90M/batch |

**n_sub heuristic:** `n_sub ≈ K × sqrt(n_mesh)`, where `K ≈ 10` for AVX2 (tuned
empirically). For CUDA, `K` may differ — see Section 4.1 for the tuning procedure.

**CUB sort cost estimates (TU104, 16-bit keys, 3 radix passes for n_sub ≤ 4096):**

| Queue size | Sort time (approx) |
|------------|-------------------|
| 3M | ~0.5 ms |
| 30M | ~3 ms |
| 90M | ~9 ms |

**CUB compaction cost estimates (TU104, single pass):**

| Queue size | Compact time (approx) |
|------------|----------------------|
| 3M | ~0.3 ms |
| 30M | ~1.5 ms |
| 90M | ~4 ms |

---

## 10. Testing and Validation Strategy

### 10.1 Numerical Equivalence

Run both `qd_RTI_AVX2` and `qd_RTI_CUDA` on the same inputs. Compare:

- `Wf`, `Ws`: must match to within 1 ULP (single precision). Both use full-precision
  division, same FMA sequence. Minor differences possible due to different FMA
  contraction rules between x86 and CUDA (`--fmad=true` by default enables FMA
  contraction beyond what the source specifies) — use a combined absolute + relative
  tolerance: `|a - b| < max(1e-6, 1e-5 * max(|a|, |b|))`. The relative term handles
  near-grazing rays (W close to 1.0) where absolute error from FMA contraction
  differences can exceed 1e-6.
- `If`, `Is`: must match exactly. If distance ties exist (coplanar faces), the CUDA
  version breaks ties by lowest face index (via the packed uint64 encoding), which may
  differ from AVX2 (which breaks ties by AVX lane position within the 8-face block).
  Both are valid. To make comparison deterministic, only assert `If`/`Is` match when
  `Wf`/`Ws` are not within the tolerance of a tie.
- `hit_cnt`: must match exactly **if all SMI values are multiples of 8**. If SMI values
  are not aligned to `VEC_SIZE` (8), the AVX2 version may double-count boundary faces:
  the AVX2 face loop steps in increments of 8 (`i_mesh += VEC_SIZE`), so the last block
  of sub-mesh N may include up to 7 faces belonging to sub-mesh N+1. Hits on those
  faces are counted in both sub-meshes. The CUDA version iterates face-by-face and
  counts each face exactly once. In this case, `hit_cnt_cuda <= hit_cnt_avx2` is the
  expected relationship. For validation, either ensure SMI values are 8-aligned in test
  inputs, or compare hit_cnt only on inputs with aligned SMI boundaries.

### 10.2 Edge Cases

- Rays parallel to an AABB face (dx_i = ±INF). The slab test handles this correctly
  if `t_low` and `t_high` become ±INF — the min/max logic still works. Test this.
- **NaN in slab test:** Ray with dx = 0 where the origin's x-coordinate equals an
  AABB x-boundary (`Xmin == ox` or `Xmax == ox`). This produces `0 × INF = NaN` in
  the slab computation. Verify that the AABB is correctly rejected (NaN propagation
  through ordered comparisons). This is the primary motivation for using conditional
  swaps instead of `fminf`/`fmaxf` in the slab test (Section 4.1).
- Rays exactly hitting a triangle edge (U = 0 or V = 0 or U+V = 1). Boundary behavior
  must match AVX2 (which uses `>=` and `<=` comparisons).
- Zero-length rays (D = 0). `dx_i` = INF for all axes. Should produce no hits.
- Rays with `W` exactly equal to 0.0 or 1.0. AVX2 uses `W >= 0` and `W < 1`.
- **Negative zero:** Construct a test case where the Möller–Trumbore numerator
  evaluates to `-0.0f`. Verify that the `fmaxf` clamp prevents the hit from being
  dropped. (Easiest: a ray whose origin lies exactly on a triangle vertex.)
- **Partial warps:** Test with `n_ray` values that are not multiples of 32 (e.g.
  `n_ray = 1`, `n_ray = 33`, `n_ray = 255`). Verify no UB or incorrect results.
- **Non-aligned n_mesh:** Test with `n_mesh` values that are NOT multiples of 8 (e.g.
  `n_mesh = 1`, `n_mesh = 7`, `n_mesh = 13`, `n_mesh = 100`). The CUDA version must
  handle these correctly since it has no SIMD-width alignment requirement. Compare
  against AVX2 results by padding the AVX2 input to a multiple of 8 with degenerate
  (zero-area) triangles.

### 10.3 Performance Benchmarks

Compare wall-clock time of `qd_RTI_CUDA` (including all H↔D transfers) vs
`qd_RTI_AVX2` (with OpenMP on all cores). Target: CUDA should be faster for
`n_ray >= 1M`. For `n_ray < 100K`, AVX2 may win due to transfer overhead — this is
acceptable (use AVX2 for small workloads, CUDA for large).

**Key metrics to profile with Nsight Compute on Kernels 2/3:**
- L2 hit rate (target: >80% with sorting enabled)
- Occupancy (target: >50%)
- Warp stall reasons (memory vs compute vs sync)
- Achieved vs theoretical FLOPs
- Scattered ray load overhead: check global load transactions vs requests — expect ~32×
  amplification on ray data loads, ~1× on face data loads.

**Phase 1 / Phase 2 balance check:** After profiling, verify that Kernel 1 time is
≤30% of total kernel time. If it exceeds this, `n_sub` is too large for the current
scene. See Section 4.1 for the `K`-factor tuning procedure.

---

## 11. File Structure

```
quadriga-lib/
├── src/
│   ├── ray_triangle_intersect_avx2.cpp    // Existing
│   ├── ray_triangle_intersect_cuda.cu     // New: kernels + wrapper
│   └── ray_triangle_intersect_cuda.cuh    // New: kernel signatures, device helpers
├── tests/
│   └── catch2_test_rti_cuda.cpp           // New: Catch2 tests
└── CMakeLists.txt                         // Add CUDA language, find CUB
```

CUB is header-only and ships with the CUDA Toolkit (>= 11.0). No external dependencies
beyond the CUDA Toolkit.

**CUB `uint16_t` key support:** `cub::DeviceRadixSort::SortPairs` with `uint16_t` keys
requires CUB from CUDA Toolkit >= 11.0. Earlier CUB versions may not instantiate the
`uint16_t` specialization. Since RTX 2070 (compute capability 7.5) is supported by
CUDA 10+, verify that the build system targets CUDA >= 11.0. If CUDA 10.x support is
ever needed, widen keys to `uint32_t` and adjust `end_bit` accordingly (doubling the
radix passes for 16-bit values).

---

## 12. Implementation Order

1. **Skeleton:** `qd_RTI_CUDA` wrapper with `cudaSetDevice(gpu_id)`, RAII cleanup struct
   (Section 8), GPU alloc, batching loop, H↔D transfers, pinned memory, stream creation.
   Kernels are stubs. Verify memory management with `cuda-memcheck`. Include the SMI
   sentinel setup (`SMI[n_sub] = n_mesh`). Include the conservative memory budget from
   Section 5.1 (with runtime CUB temp query). Add early returns for `n_ray == 0`,
   `n_sub == 0`, and `n_mesh == 0` (Section 8). Add `n_sub > 65535` range check
   (Section 8). Note: no `n_mesh % 8` check — the CUDA version accepts any `n_mesh >= 1`.

2. **Kernel 1 (AABB test):** Implement slab test + warp-level enqueue with partial-warp
   safety (no early return before `__ballot_sync`, `static_assert` on block size) and
   `pos < queue_capacity` overflow guard (Section 7.2). Use ordered comparisons (not
   `fminf`/`fmaxf`) in the slab test for NaN safety (Section 4.1). Validate by
   comparing the set of `(ray_idx, aabb_idx)` pairs against the AVX2 version's
   `sub_mesh_hit` results. Test with `n_ray` values that are not multiples of 32.
   Test queue overflow by temporarily setting `queue_capacity` to a very small value
   and verifying that the re-run produces correct results. Test rays with a zero
   direction component to verify NaN propagation.

3. **CUB sort integration:** Add work queue sorting by `aabb_idx` (uint16 key) after
   Phase 1. Use `DoubleBuffer` and `begin_bit=0, end_bit=ceil(log2(n_sub))`. Verify
   that the sorted queue contains the same entries as the unsorted queue (set equality).

4. **Kernel 2 (FBS):** Implement Möller–Trumbore + negative-zero clamp + atomicMin +
   hit flag. Add `__launch_bounds__(256, 2)` (Section 7.4). Validate `Wf`/`If` against
   AVX2. Include the negative-zero edge case test.

5. **CUB compaction:** Add `DeviceSelect::Flagged` after Kernel 2 using
   `CountingInputIterator` + `d_wq_had_hit`. Validate that compacted indices are a
   correct subset of the sorted queue. Add `d_num_selected` readback.

6. **Kernel 3 (SBS):** Implement using compacted index array. Validate `Ws`/`Is`.

7. **Kernel 4 + hit_cnt:** Finalize outputs, wire up hit counting. Add debug-mode
   `assert(Ws >= Wf)` invariant check (Section 4.4).

8. **Double buffering:** Add second stream + pinned memory. Include pre-loop staging of
   batch 0. Place next-batch CPU staging in the window between the `queue_tail` sync
   and the `num_selected` sync (overlapping with GPU sort + K2 + compact). Ensure
   queue overflow handler syncs both streams. Verify correctness is unchanged, measure
   overlap with `nvprof` timeline. Confirm that the CPU staging memcpy completes before
   the next iteration's H→D transfer.

9. **Profile and optimize:** Nsight Compute on Kernels 1, 2, 3.
   - Check Phase 1 / Phase 2 balance (Section 4.1). Sweep `K` factor if needed.
   - Check L2 hit rate on Kernels 2/3 (should be >80% with sorting).
   - Check scattered ray load amplification (global load transactions vs requests).
     If "Long Scoreboard" stalls on ray loads are significant, try shared memory ray
     caching (Section 3).
   - Check occupancy, warp stall reasons. Tune `__launch_bounds__` MIN_BLOCKS_PER_SM.
   - Test block sizes 128/256/512 for Kernels 2/3.
   - Benchmark eliminating the `num_selected` D→H sync (Section 5.2) for multi-batch
     workloads.