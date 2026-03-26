# CUDA Implementation Plan: `qd_RPI_CUDA` (v3.2)

## 1. Overview

CUDA port of `qd_RPI_AVX2` — the beam-point intersection algorithm used in QuaDRiGa
Ray Tracing (QRT). A modified Möller–Trumbore test determines which points in a 3-D
point cloud are intercepted by triangular ray tubes. The AVX2 version processes one ray
at a time (outer loop = rays via OpenMP, inner loop = AABBs then points with 8-wide
SIMD). The CUDA version inverts this: it processes all rays in parallel, using stream
compaction between phases to avoid scheduling empty work.

**Architecture choice:** Global work queue with warp-level atomics (matching `qd_RTI_CUDA`).
After sorting the work queue by sub-cloud index, a segment table groups consecutive
entries for the same sub-cloud. Kernel 2 assigns each thread-block a chunk of up to
`RAYS_PER_BLOCK` (default 32) rays within one segment. 256 threads cooperatively
process all points in the sub-cloud for each ray, reusing point data in L1 across
rays within the chunk.

**n_point alignment:** Unlike the AVX2 version (which requires `n_point` to be a
multiple of 8 due to 8-wide SIMD processing), the CUDA version has **no alignment
requirement on `n_point`**. Kernel 2 iterates points scalar per thread with a bounds
check. `cudaMalloc` guarantees 256-byte base pointer alignment, which is sufficient.

**Key output per ray:**

| Output         | Type                    | Description                               |
| -------------- | ----------------------- | ----------------------------------------- |
| `p_hit[i_ray]` | `std::vector<unsigned>` | List of point indices hit by ray `i_ray`. |

Unlike `qd_RTI_CUDA` (which produces fixed-size per-ray outputs: FBS/SBS distance and
face index), the RPI output is a **variable-length list per ray**. This means the GPU
writes an unordered bag of `(ray_id, point_id)` hit pairs into a flat output buffer;
the host scatters these into the per-ray vectors after each batch.

---

## 2. Data Layout on GPU

### 2.1 Scene Data (resident for the entire run, uploaded once)

All arrays use SoA layout, matching the AVX2 version.

```
// Point cloud — 3 float arrays, each of length n_point
float *d_Px, *d_Py, *d_Pz;

// Sub-cloud index table — defines point ranges per AABB
// Length n_sub + 1. SCI[n_sub] = n_point (sentinel).
// Points for AABB i = [SCI[i], SCI[i+1])
uint32_t *d_SCI;

// AABB bounds — 6 float arrays, each of length n_sub (no padding needed)
float *d_Xmin, *d_Xmax;
float *d_Ymin, *d_Ymax;
float *d_Zmin, *d_Zmax;
```

**Memory footprint:** `3 × n_point × 4 + (n_sub + 1) × 4 + 6 × n_sub × 4` bytes.
For a large scene (4M points, 4000 AABBs): ~48.1 MB.

**SCI sentinel:** The host wrapper allocates a temporary array of length `n_sub + 1`,
copies the caller's `SCI[0..n_sub-1]`, and sets `SCI[n_sub] = (uint32_t)n_point`
before uploading. This eliminates the
`(i_sub == n_sub - 1) ? n_point : SCI[i_sub + 1]` branch inside Kernel 2's point
loop, avoiding warp divergence when threads in a warp process different sub-clouds.

### 2.2 Per-Ray Data (batched, double-buffered)

Each ray has 24 float attributes (T1xyz, T2xyz, T3xyz, Nxyz, D1xyz, D2xyz, D3xyz,
rD1/rD2/rD3).

```
// Input — uploaded H→D per batch (24 SoA arrays, each of length batch_size)
float *d_T1x, *d_T1y, *d_T1z;     // Ray vertex 1
float *d_T2x, *d_T2y, *d_T2z;     // Ray vertex 2
float *d_T3x, *d_T3y, *d_T3z;     // Ray vertex 3
float *d_Nx,  *d_Ny,  *d_Nz;      // Ray tube normal vector
float *d_D1x, *d_D1y, *d_D1z;     // Ray direction 1
float *d_D2x, *d_D2y, *d_D2z;     // Ray direction 2
float *d_D3x, *d_D3y, *d_D3z;     // Ray direction 3
float *d_rD1, *d_rD2, *d_rD3;     // Inverse dot products
```

**Ray pointer table for Kernel 2:** Kernel 2 loads one ray at a time into shared
memory using threads 0–23, where each thread reads from a different SoA array. To
avoid passing 24 pointer arguments and to enable indexed access, each stream owns a
device-allocated pointer table:

```cuda
// Per-stream device array — 24 pointers, one per SoA ray attribute
// Order: T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
//        Nx,  Ny,  Nz,  D1x, D1y, D1z, D2x, D2y, D2z,
//        D3x, D3y, D3z, rD1, rD2, rD3
const float **d_ray_ptrs[2];   // d_ray_ptrs[s] = cudaMalloc(24 * sizeof(float*))
```

The host updates `d_ray_ptrs[s]` via `cudaMemcpyAsync` on the batch stream before
Kernel 2 launches on that stream's ray buffers. Cost: one 192-byte upload per batch —
negligible. Kernel 2 receives `d_ray_ptrs[s]` as a kernel argument and the ray load
becomes `s_ray[threadIdx.x] = d_ray_ptrs[threadIdx.x][sorted_ray_idx[r]]` for
threads 0–23.

**Double-buffering safety:** Each stream has its own `d_ray_ptrs` allocation (192
bytes each). Updates to stream `s`'s pointer table are ordered on `stream[s]` before
the Kernel 2 launch on that stream. No global state is shared between streams, so
there is no cross-stream hazard — even if the pipeline is restructured to allow
concurrent Kernel 2 launches in the future.

### 2.3 Work Queue (GPU-only, per batch)

```
// Each entry is a (ray_idx, sub_idx) pair.
// Two copies (A/B) needed for CUB radix sort double-buffering.
uint32_t *d_wq_ray_idx_A;         // Length = queue_capacity
uint32_t *d_wq_ray_idx_B;         // Length = queue_capacity (sort output buffer)
uint16_t *d_wq_sub_idx_A;         // Length = queue_capacity (uint16 — n_sub ≤ 65535)
uint16_t *d_wq_sub_idx_B;         // Length = queue_capacity (sort output buffer)

uint32_t *d_queue_tail;            // Single atomic counter (own cache line)

// CUB temp buffer (allocated once, reused for sort + segment ops)
void     *d_cub_temp;             // Length = cub_temp_bytes (see Section 5.1)

// Segment table — built after sort, consumed by Kernel 2
// CUB RunLengthEncode outputs:
uint16_t *d_seg_sub_idx;          // Length = n_sub (unique sub_idx per segment)
uint32_t *d_seg_run_len;          // Length = n_sub (number of rays in each segment)
uint32_t *d_seg_num_segments;     // Single value (number of segments)

// Derived from run lengths:
uint32_t *d_seg_offset;           // Length = n_sub + 1 (exclusive prefix sum of run_len)
uint32_t *d_seg_chunk_offset;     // Length = n_sub + 1 (exclusive prefix sum of
                                  //   ceil(run_len / RAYS_PER_BLOCK), maps blockIdx
                                  //   to segment via binary search)
uint32_t *d_total_chunks;         // Single value (total K2 grid size)
```

**Queue tail placement:** Allocate `d_queue_tail` via its own `cudaMalloc` (or at a
128-byte aligned offset) to avoid false sharing with adjacent allocations.

**Queue capacity:** Pre-allocate `queue_capacity = batch_size × EST_AVG_HITS` where
`EST_AVG_HITS` is a conservative estimate (e.g. 8–12). Kernel 1 guards writes with
`pos < queue_capacity` (Section 4.1, 7.2) so overflow never causes out-of-bounds
writes. The unguarded `atomicAdd` on `queue_tail` records the true count. After
Kernel 1, read `queue_tail` back to the host. If it exceeds `queue_capacity`,
synchronize both streams, `cudaFree` all queue-related buffers for both streams,
reallocate at `2 × h_queue_tail`, and re-run the batch from the counter reset. In practice, with
typical scenes (1–5 AABB hits per ray), overflow should never happen.

**sub_idx type:** `uint16_t` is sufficient for `n_sub ≤ 65535`. This halves the sort
key size, directly reducing CUB radix sort cost. If `n_sub > 65535` is ever needed,
widen to `uint32_t` and adjust `end_bit` in the sort call (Section 6).

### 2.4 Hit Output (GPU-only, per batch)

```
// Flat buffer of (ray_idx, point_idx) hit pairs.
uint32_t *d_hit_ray_idx;          // Length = hit_capacity
uint32_t *d_hit_point_idx;        // Length = hit_capacity
uint32_t *d_hit_count;            // Single atomic counter (own cache line)
uint32_t *d_overflow_flag;        // Single value: set to 1 by K2 if any block's
                                  //   tile hit count exceeds MAX_LOCAL_HITS.
                                  //   Reset to 0 alongside counters (cudaMemsetAsync).
```

**Hit capacity:** Pre-allocate for an expected hit rate. A generous budget:
`hit_capacity = batch_size × 2` (assumes an average of 2 point hits per ray across
all sub-clouds). If overflow occurs (detected after Kernel 2), re-run the batch with
a larger buffer. In practice, hits are extremely sparse — most rays hit 0–3 points
total.

### 2.5 Memory Budget per Ray

| Item                                                | Bytes              |
| --------------------------------------------------- | ------------------ |
| Ray input (24 floats)                               | 96                 |
| Work queue A (ray_idx:4 + sub_idx:2) × ~8 entries   | 48                 |
| Work queue B / sort output (same)                   | 48                 |
| CUB temp share (~8 entries × 4 bytes, conservative) | 32                 |
| Hit output (ray_idx:4 + point_idx:4) × ~2 entries   | 16                 |
| **Total**                                           | **~240 bytes/ray** |

The actual per-ray cost depends on the average number of AABB hits (here estimated
at 8) and point hits (here estimated at 2). Section 5.1 uses a conservative formula
that accounts for this variability.

**Per-stream fixed cost (not per-ray):** The segment table (Section 6.3) and CUB temp
buffers are sized by `n_sub` and `queue_capacity` respectively, not by `batch_size`.
These are small — see Section 5.1 for the exact accounting.

---

## 3. Kernel Architecture

The pipeline has 3 kernels plus CUB operations executed in sequence per batch:

```
┌─────────────────────────────────────────────────────┐
│ cudaMemsetAsync: reset queue_tail and hit_count to 0│
│   Two calls on the batch stream.                    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Kernel 1: aabb_test_and_enqueue                     │
│   Each thread = one ray. Loops over all AABBs.      │
│   AABB overlap test: advance wavefront to each of   │
│   8 AABB corners, build ray bounding box, test      │
│   overlap (same math as AVX2 lines 110–239).        │
│   On hit: warp-ballot + atomicAdd to enqueue.       │
│   Output: work queue + queue_tail.                  │
└──────────────────────┬──────────────────────────────┘
                       │  D→H copy of queue_tail (4 bytes)
                       │  to determine sort + K2 launch size
                       │
┌──────────────────────▼──────────────────────────────┐
│ CUB sort: SortPairs on work queue by sub_idx        │
│   uint16 keys, uint32 values, 0..end_bit bits only. │
│   Groups consecutive entries accessing the same     │
│   sub-cloud's point data.                           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Segment table construction (CUB + tiny kernel)      │
│   1. CUB RunLengthEncode → segments (unique         │
│      sub_idx + run lengths + num_segments)           │
│   2. Kernel 1b: compute chunks_per_segment =        │
│      ceil(run_len / RAYS_PER_BLOCK)                 │
│   3. CUB ExclusiveSum → chunk_offset[] + total      │
│   Maps blockIdx → (segment, chunk) for Kernel 2.    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ Kernel 2: point_intersect                           │
│   Each BLOCK = one chunk of a sorted segment.       │
│   RAYS_PER_BLOCK rays (default 32) against one      │
│   sub-cloud. 256 threads cooperatively process all  │
│   points for each ray. Modified Möller–Trumbore     │
│   test (same math as AVX2 lines 253–361).           │
│   On hit: block-local buffer + bulk flush to        │
│   global hit output via atomicAdd on hit_count.     │
│   Launch: total_chunks blocks × 256 threads.        │
└─────────────────────────────────────────────────────┘
```

### Why Kernel 2 processes multiple rays per block (not one)

In `qd_RTI_CUDA`, each work item processes a face loop (hundreds to thousands of faces)
per single thread. In `qd_RPI_CUDA`, the per-work-item computation is different:
the modified Möller–Trumbore test for beam-point intersection requires advancing all
three ray vertices to the wavefront at each point's distance, then computing
barycentric coordinates. This is ~50 FLOPs per point. With 500–1000 points per
sub-cloud, a single thread would perform 25K–50K FLOPs — feasible but not optimal.

A block of 256 threads cooperatively processes the sub-cloud: thread `t` handles points
at indices `start + t`, `start + t + 256`, etc. This gives 2–4 points per thread,
maximizing throughput. All threads in the block share the same ray data (loaded once
into shared memory), so there is no divergence in the ray-loading phase.

**Multi-ray batching:** After the CUB sort, consecutive work queue entries with the
same `sub_idx` form a *segment*. A segment table (Section 6) maps each block to a
contiguous chunk of up to `RAYS_PER_BLOCK` (default 32) rays within one segment. The
block processes these rays sequentially: for each ray, it loads the ray's 24 floats
into shared memory, then all 256 threads cooperatively test all points in the
sub-cloud against that ray. This yields two key benefits:

1. **Reduced block count.** Instead of launching one block per work queue entry
   (potentially 30M blocks for a medium scene), blocks are grouped by sub-cloud. A
   medium scene (200 sub-clouds, 30M work items) produces ~30M / 32 ≈ 940K blocks
   instead of 30M — a 32× reduction in block scheduling overhead.

2. **Point data L1 reuse.** All rays in a chunk access the same sub-cloud's point
   data. The first ray's point loop warms the L1 cache; subsequent rays hit L1
   instead of L2/DRAM. For a sub-cloud of 1000 points × 12 bytes = 12 KB, this
   fits comfortably in TU104's 64 KB L1 partition.

The per-ray cost is two `__syncthreads()` calls (one before loading the next ray, one
after) plus tile-boundary flush syncs (two per tile, see Section 4.2) plus a single
scattered global load of 24 floats (96 bytes) via the per-stream `d_ray_ptrs`
device pointer table. For a sub-cloud of 1000 points with `TILE_SIZE = 2`: 2 ray syncs + 2 tiles ×
2 syncs = 6 syncs per ray × 32 rays ≈ 192 syncs × ~20 cycles ≈ 3840 cycles overhead
— negligible compared to the point loop work.

### Why CUB sort is required (not optional)

The sort serves two purposes:

1. **Segment formation.** Multi-ray batching (above) requires that all work queue
   entries for the same sub-cloud are contiguous. The sort groups them, and the
   subsequent CUB RunLengthEncode (Section 6) identifies segment boundaries.

2. **Inter-block L2 reuse.** Even with multi-ray chunks, consecutive blocks
   (scheduled to the same or nearby SMs) process the same or adjacent sub-clouds, so
   point data loaded by block `b` is warm in L2 for block `b+1`. Without sorting,
   consecutive blocks access random sub-clouds, thrashing L2. For large sub-clouds
   (1000 points × 12 bytes = 12 KB each), this can be significant on TU104 (4 MB L2).

### Scattered ray access pattern after sorting

After sorting by `sub_idx`, rays within each segment have arbitrary `ray_idx` values.
Each ray load (24 floats from SoA global arrays via the per-stream `d_ray_ptrs`
device pointer table, loaded by threads 0–23 of the block into shared memory) is a
scattered access.
This is a one-time cost per ray within the chunk (24 × 4 = 96 bytes), amortized over
the entire point loop for that ray. With `RAYS_PER_BLOCK = 32` rays per chunk, each
chunk performs 32 scattered ray loads but reuses the same point data 32 times. The
point data access dominates bandwidth.

---

## 4. Kernel Details

### 4.0 Counter Reset — `cudaMemsetAsync`

```
cudaMemsetAsync(d_queue_tail,     0, sizeof(uint32_t), stream)
cudaMemsetAsync(d_hit_count,      0, sizeof(uint32_t), stream)
cudaMemsetAsync(d_overflow_flag,  0, sizeof(uint32_t), stream)
```

Three asynchronous memset calls on the batch stream, issued before Kernel 1. All are
guaranteed to complete before Kernel 1 begins (same-stream ordering). This replaces a
dedicated 1-thread kernel — `cudaMemsetAsync` avoids kernel launch overhead and is
semantically clearer for a simple zero-fill.

### 4.1 Kernel 1 — `aabb_test_and_enqueue`

```
Launch: <<<ceil(batch_size / 256), 256, 0, stream>>>
Thread mapping: thread i = ray i
```

**Algorithm per thread:**

```
// All 32 lanes in a warp must participate in __ballot_sync.
// Invalid threads (past end of batch) participate but never enqueue.
bool valid = (ray_idx < current_batch_count)

if valid:
    // Load ray's 24 floats from SoA arrays (coalesced)
    T1x, T1y, T1z = d_T1x[ray_idx], d_T1y[ray_idx], d_T1z[ray_idx]
    // ... (all 24 floats) ...

    // Precompute origin × normal products (same as AVX2 lines 116–124)
    ox0_x_nx = T1x * Nx;  oy0_x_ny = T1y * Ny;  oz0_x_nz = T1z * Nz
    ox1_x_nx = T2x * Nx;  oy1_x_ny = T2y * Ny;  oz1_x_nz = T2z * Nz
    ox2_x_nx = T3x * Nx;  oy2_x_ny = T3y * Ny;  oz2_x_nz = T3z * Nz

for i_sub = 0 .. n_sub - 1:
    has_hit = false
    if valid:
        // Load AABB bounds with slack (AVX2 lines 129–142)
        b0_low  = Xmin[i_sub] - 1e-5f;  b0_high = Xmax[i_sub] + 1e-5f
        b1_low  = Ymin[i_sub] - 1e-5f;  b1_high = Ymax[i_sub] + 1e-5f
        b2_low  = Zmin[i_sub] - 1e-5f;  b2_high = Zmax[i_sub] + 1e-5f

        // Enumerate 8 AABB corner points (AVX2 lines 145–147)
        rx[8] = {b0_low, b0_low, b0_low, b0_low, b0_high, b0_high, b0_high, b0_high}
        ry[8] = {b1_low, b1_low, b1_high, b1_high, b1_low, b1_low, b1_high, b1_high}
        rz[8] = {b2_low, b2_high, b2_low, b2_high, b2_low, b2_high, b2_low, b2_high}

        // Initialize advanced wavefront bounding box
        a0_low = a1_low = a2_low = +INF
        a0_high = a1_high = a2_high = -INF

        // For each corner: advance all 3 ray vertices, update min/max
        // (AVX2 lines 158–219, scalar, using fmaf())
        for c = 0..7:
            // Vertex 0: distance to wavefront at corner (stepwise FMA, matching K2)
            v = fmaf(rz[c], Nz, -oz0_x_nz)
            d = rD1 * v
            v = fmaf(ry[c], Ny, -oy0_x_ny)
            d = fmaf(rD1, v, d)
            v = fmaf(rx[c], Nx, -ox0_x_nx)
            d = fmaf(rD1, v, d)
            v = fmaf(d, D1x, T1x);  a0_low = fminf(v, a0_low);  a0_high = fmaxf(v, a0_high)
            v = fmaf(d, D1y, T1y);  a1_low = fminf(v, a1_low);  a1_high = fmaxf(v, a1_high)
            v = fmaf(d, D1z, T1z);  a2_low = fminf(v, a2_low);  a2_high = fmaxf(v, a2_high)
            // Vertex 1: same pattern with rD2, D2xyz, T2xyz
            // Vertex 2: same pattern with rD3, D3xyz, T3xyz

        // Test 6 overlap conditions (AVX2 lines 223–233)
        has_hit = (a0_high >= b0_low) && (a0_low <= b0_high)
               && (a1_high >= b1_low) && (a1_low <= b1_high)
               && (a2_high >= b2_low) && (a2_low <= b2_high)

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
            wq_sub_idx_A[pos]  = (uint16_t)i_sub
```

**Queue overflow safety:** The `atomicAdd` on `queue_tail` runs unconditionally (even
when `pos >= queue_capacity`) so that the host readback after Kernel 1 sees the *true*
count and can size the reallocation correctly. The `pos < queue_capacity` guard prevents
out-of-bounds writes. Dropped writes are harmless because the batch is re-run from
the counter reset after reallocation (Section 5.1). The guard is after `__ballot_sync`, so it
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
padding is needed.

**fminf/fmaxf in the wavefront bounding box:** Unlike the slab-test AABB check in
`qd_RTI_CUDA` (where `fminf`/`fmaxf` are dangerous due to NaN propagation from
`0 × INF`), the RPI AABB test uses `fminf`/`fmaxf` only to track the min/max of
advanced vertex positions. The inputs to these functions are products of finite ray
data and AABB coordinates — NaN does not arise in normal operation. Using
`fminf`/`fmaxf` here is safe and preferred (maps to hardware `FMNMX` instruction).

**Register pressure:** 24 floats for ray data + 9 precomputed origin×normal products
+ ~20 registers for the AABB loop temporaries ≈ 53 registers/thread. At 256
threads/block, this is well within budget for good occupancy.

### 4.2 Kernel 2 — `point_intersect`

```
Launch: <<<total_chunks, 256, 0, stream>>>
Thread mapping: block b = chunk b (maps to a segment + ray range via segment table)
                thread t within block = cooperative point index
```

After the CUB sort and segment table construction (Section 6), the kernel receives:
- `sorted_ray_idx`, `sorted_sub_idx` — sorted work queue (CUB DoubleBuffer current)
- `seg_sub_idx[]` — unique sub_idx per segment
- `seg_offset[]` — exclusive prefix sum of run lengths (start index into sorted queue)
- `chunk_offset[]` — exclusive prefix sum of ceil(run_len / RAYS_PER_BLOCK)
  (maps blockIdx to segment via binary search)
- `num_segments` — number of segments
- `total_chunks` — grid size (sum of all chunk counts)

**Algorithm per block:**

```
RAYS_PER_BLOCK = 32    // tunable: 16, 32, or 64
// MAX_LOCAL_HITS and TILE_SIZE are runtime kernel arguments (not compile-time constants).
// Defaults: TILE_SIZE = 2, MAX_LOCAL_HITS = TILE_SIZE × 256 = 512.
// The host can increase them on re-run if d_overflow_flag is set after Kernel 2.

b = blockIdx.x
if b >= total_chunks: return

// --- Binary search: map blockIdx → (segment, chunk_within_segment) ---
// Find largest seg_idx such that chunk_offset[seg_idx] <= b
seg_idx = upper_bound(chunk_offset, num_segments + 1, b) - 1
chunk_in_seg = b - chunk_offset[seg_idx]

sub_idx = seg_sub_idx[seg_idx]

// Ray range for this chunk within the sorted queue
ray_start = seg_offset[seg_idx] + chunk_in_seg * RAYS_PER_BLOCK
ray_end   = min(ray_start + RAYS_PER_BLOCK,
                seg_offset[seg_idx + 1])   // seg_offset has sentinel at num_segments

// Point range for this sub-cloud (no branch — SCI has sentinel at n_sub)
i_start = SCI[sub_idx]
i_end   = SCI[sub_idx + 1]

// --- Shared memory ---
__shared__ float s_ray[24];                         // current ray's 24 floats
__shared__ uint32_t s_ray_idx;                      // current ray's global index
__shared__ uint32_t s_hit_rays[MAX_LOCAL_HITS];     // hit buffer: ray indices
__shared__ uint32_t s_hit_points[MAX_LOCAL_HITS];   // hit buffer: point indices
__shared__ uint32_t s_hit_count;                    // local hit counter
__shared__ uint32_t s_flush_base;                   // global base for current flush

if threadIdx.x == 0:
    s_hit_count = 0
__syncthreads()

// --- Outer loop: iterate over rays in this chunk ---
for r = ray_start .. ray_end - 1:

    // Load ray into shared memory via per-stream device pointer table
    if threadIdx.x < 24:
        s_ray[threadIdx.x] = d_ray_ptrs[threadIdx.x][sorted_ray_idx[r]]
    if threadIdx.x == 24:
        s_ray_idx = sorted_ray_idx[r]
    __syncthreads()

    // Unpack from shared memory into registers
    T1x = s_ray[0]; T1y = s_ray[1]; T1z = s_ray[2];
    // ... all 24 values ...
    ray_idx = s_ray_idx

    // Precompute origin × normal products
    ox0_x_nx = T1x * Nx;  oy0_x_ny = T1y * Ny;  oz0_x_nz = T1z * Nz;
    ox1_x_nx = T2x * Nx;  oy1_x_ny = T2y * Ny;  oz1_x_nz = T2z * Nz;
    ox2_x_nx = T3x * Nx;  oy2_x_ny = T3y * Ny;  oz2_x_nz = T3z * Nz;

    // --- Tiled point loop: 256 threads cooperatively, flush every TILE_SIZE × 256 points ---
    for tile_base = i_start;  tile_base < i_end;  tile_base += 256 * TILE_SIZE:
        tile_end = min(tile_base + 256 * TILE_SIZE, i_end)

        for ip = tile_base + threadIdx.x;  ip < tile_end;  ip += 256:
            // Load point (coalesced across warp — consecutive threads read consecutive points)
            rx = Px[ip];  ry = Py[ip];  rz = Pz[ip]

            // Distance from vertex 0 origin to wavefront at point (AVX2 lines 261–266)
            v = fmaf(rz, Nz, -oz0_x_nz)
            d = rD1 * v
            v = fmaf(ry, Ny, -oy0_x_ny)
            d = fmaf(rD1, v, d)
            v = fmaf(rx, Nx, -ox0_x_nx)
            d = fmaf(rD1, v, d)

            // Advance vertex 0 → V (AVX2 lines 269–271)
            Vx = fmaf(d, D1x, T1x)
            Vy = fmaf(d, D1y, T1y)
            Vz = fmaf(d, D1z, T1z)

            // Compute edge e1 = advanced_V1 - V (AVX2 lines 274–285)
            v = fmaf(rz, Nz, -oz1_x_nz)
            d1 = rD2 * v
            v = fmaf(ry, Ny, -oy1_x_ny)
            d1 = fmaf(rD2, v, d1)
            v = fmaf(rx, Nx, -ox1_x_nx)
            d1 = fmaf(rD2, v, d1)

            e1x = fmaf(d1, D2x, T2x) - Vx
            e1y = fmaf(d1, D2y, T2y) - Vy
            e1z = fmaf(d1, D2z, T2z) - Vz

            // Compute edge e2 = advanced_V2 - V (AVX2 lines 289–299)
            v = fmaf(rz, Nz, -oz2_x_nz)
            d2 = rD3 * v
            v = fmaf(ry, Ny, -oy2_x_ny)
            d2 = fmaf(rD3, v, d2)
            v = fmaf(rx, Nx, -ox2_x_nx)
            d2 = fmaf(rD3, v, d2)

            e2x = fmaf(d2, D3x, T3x) - Vx
            e2y = fmaf(d2, D3y, T3y) - Vy
            e2z = fmaf(d2, D3z, T3z) - Vz

            // Vector from V to point (AVX2 lines 306–308)
            tx = rx - Vx;  ty = ry - Vy;  tz = rz - Vz

            // Cross products → barycentric coordinates (AVX2 lines 311–337)
            // 1st barycentric: PQ = N × e2, DT = dot(e1, PQ), U = dot(t, PQ)
            PQ = e2z*Ny - e2y*Nz;   DT  = e1x*PQ;  U  = tx*PQ
            PQ = e2x*Nz - e2z*Nx;   DT += e1y*PQ;  U += ty*PQ
            PQ = e2y*Nx - e2x*Ny;   DT += e1z*PQ;  U += tz*PQ

            // 2nd barycentric: PQ = e1 × t, V = dot(N, PQ)
            PQ = e1z*ty - e1y*tz;   V  = Nx*PQ
            PQ = e1x*tz - e1z*tx;   V += Ny*PQ
            PQ = e1y*tx - e1x*ty;   V += Nz*PQ

            // Normalize (AVX2 lines 340–341)
            DT = __fdividef(1.0f, DT)   // fast approximate division, see Section 7.1
            U *= DT
            V *= DT

            // Hit condition (AVX2 lines 345–355, all 3 vertex distances checked)
            hit = (U >= 0) && (V >= 0) && (U + V <= 1.0f) && (d >= 0) && (d1 >= 0) && (d2 >= 0)

            if hit:
                // Append to block-local hit buffer
                slot = atomicAdd(&s_hit_count, 1)
                if slot < MAX_LOCAL_HITS:
                    s_hit_rays[slot]   = ray_idx
                    s_hit_points[slot] = ip

        // --- Tile boundary flush ---
        // All 256 threads reach this point (uniform loop bounds: tile_base, tile_end).
        __syncthreads()

        if s_hit_count > 0:
            flush_count = min(s_hit_count, MAX_LOCAL_HITS)
            if threadIdx.x == 0:
                // Advance global counter by true hit count (not clamped flush_count)
                // so that the host sees the correct total even on tile overflow.
                // Hits beyond MAX_LOCAL_HITS are lost in this tile but the count
                // is preserved for overflow detection and reallocation sizing.
                s_flush_base = atomicAdd(hit_count, s_hit_count)
                if s_hit_count > MAX_LOCAL_HITS:
                    atomicOr(d_overflow_flag, 1u)
            __syncthreads()
            for i = threadIdx.x; i < flush_count; i += 256:
                if s_flush_base + i < hit_capacity:
                    hit_ray_idx[s_flush_base + i]   = s_hit_rays[i]
                    hit_point_idx[s_flush_base + i] = s_hit_points[i]
            if threadIdx.x == 0:
                s_hit_count = 0
            __syncthreads()

    __syncthreads()   // ensure all threads done with tiled point loop before next ray load
```

**Tiled flush design:** The point loop is tiled into chunks of `TILE_SIZE × 256`
points. At each tile boundary, the block flushes accumulated hits from shared memory
to the global hit output buffer, then resets the local counter. This eliminates the
risk of shared memory overflow when hit density is high — for example, a diverged beam
hitting a dense receiver plane at grazing incidence can produce tens of thousands of
hits per ray within a single sub-cloud.

The flush sync is safe: all 256 threads share identical tile loop bounds (`tile_base`,
`tile_end` are derived from block-uniform `i_start` / `i_end`), so all threads
converge at the `__syncthreads()` at the tile boundary. There is no divergence issue.

**`MAX_LOCAL_HITS` (runtime, default 512):** With `TILE_SIZE = 2`, each tile covers
512 points per thread-pass. The worst case is 100% hit density within a tile: 512
hits. The buffer is sized to match. The constraint `MAX_LOCAL_HITS >= TILE_SIZE × 256`
must hold; the host enforces this before launch. The buffer costs
`2 × MAX_LOCAL_HITS × 4` bytes of dynamic shared memory (4 KB at default).

**`TILE_SIZE` tuning (runtime, default 2):** `TILE_SIZE = 1` flushes after every 256 points (most
conservative; handles any hit density; 2 extra `__syncthreads()` per 256 points).
`TILE_SIZE = 2` flushes every 512 points (default). `TILE_SIZE = 4` flushes every
1024 points (less overhead for sparse scenes, but requires `MAX_LOCAL_HITS >= 1024`).
Even at `TILE_SIZE = 1`, the flush overhead is ~40 cycles per tile on a loop body that
does ~50 FLOPs × 256 = 12,800 FLOPs of work — negligible.

**Overflow diagnostic and true-count accounting:** If `s_hit_count > MAX_LOCAL_HITS`
at a tile boundary, hits beyond the buffer capacity are lost for that tile. To detect
this, the flush logic sets a global overflow flag via `atomicOr` and advances the
global `hit_count` by the *true* `s_hit_count` (not the clamped `flush_count`). This
ensures that the host readback of `hit_count` after Kernel 2 reflects the correct
total number of hits — including those lost to tile overflow. The host can therefore
use `h_hit_count` directly for reallocation sizing when re-running the batch with a
larger `TILE_SIZE` / `MAX_LOCAL_HITS`. The gap between the true count and the actually
written entries (at most `s_hit_count - MAX_LOCAL_HITS` per overflowed tile) is filled
with stale data in the global buffer; these are harmless because the overflow flag
triggers a full re-run before the hit data is consumed.

The host checks `d_overflow_flag` after Kernel 2. If set, the batch is re-run with
increased `TILE_SIZE` / `MAX_LOCAL_HITS` (runtime arguments — see below). In practice,
with `TILE_SIZE = 2` and `MAX_LOCAL_HITS = 512`, overflow requires >100% hit density
within a 512-point tile — essentially impossible for valid geometry.

**Shared memory layout:**

```
// Static shared memory:
__shared__ float s_ray[24];                //   96 bytes
__shared__ uint32_t s_ray_idx;             //    4 bytes
__shared__ uint32_t s_hit_count;           //    4 bytes
__shared__ uint32_t s_flush_base;          //    4 bytes
                                           //  108 bytes static

// Dynamic shared memory (sized by host at launch via 3rd <<< >>> argument):
// Layout: s_hit_rays[MAX_LOCAL_HITS] followed by s_hit_points[MAX_LOCAL_HITS]
// Total dynamic = 2 × MAX_LOCAL_HITS × 4 bytes
extern __shared__ uint32_t s_dyn[];
uint32_t *s_hit_rays   = s_dyn;                   // [0 .. MAX_LOCAL_HITS)
uint32_t *s_hit_points = s_dyn + MAX_LOCAL_HITS;   // [MAX_LOCAL_HITS .. 2*MAX_LOCAL_HITS)
```

The host computes `smem_bytes = 2 * max_local_hits * sizeof(uint32_t)` and passes it
as the third launch parameter:
`point_intersect<<<total_chunks, 256, smem_bytes, stream>>>(... , tile_size, max_local_hits, ...)`

Default: `tile_size = 2`, `max_local_hits = 512` → `smem_bytes = 4096`. If tile
overflow is detected (`d_overflow_flag != 0`), the host doubles both values and
re-runs the batch. The constraint `MAX_LOCAL_HITS >= TILE_SIZE × 256` guarantees
the buffer can hold the worst case (100% hit density within a tile).

**Binary search cost:** The `upper_bound` search over `chunk_offset[]` has at most
`ceil(log2(num_segments))` iterations. With `num_segments ≤ n_sub ≤ 65535`, this is
≤ 16 iterations per block — negligible compared to the point loop work. All threads
in the block execute the same search (uniform control flow), so there is no warp
divergence.

**Sync pattern:** The outer ray loop has one `__syncthreads()` at the end (after the
tiled point loop, before the next ray load). Combined with the sync after the ray load
at the top of the loop, this gives two syncs per ray plus two syncs per tile flush.
For a sub-cloud of 1000 points with `TILE_SIZE = 2`: ceil(1000 / 512) = 2 tiles ×
2 syncs = 4 tile syncs + 2 ray syncs = 6 syncs per ray × 32 rays = 192 syncs per
chunk at ~20 cycles each ≈ 3840 cycles overhead. Negligible.

**Warp-level coherence:** All threads in a block share the same ray, so the
Möller–Trumbore computation has excellent warp-level coherence — no divergence until
the final hit/miss branch, which is almost always "miss" (uniform no-op). The rare
hit path (shared memory atomic) causes negligible divergence.

**Point data L1 reuse:** All rays in a chunk access the same sub-cloud's point data.
After the first ray's point loop warms the L1 cache, subsequent rays benefit from L1
hits. For a sub-cloud of 1000 points × 12 bytes = 12 KB, this fits comfortably in
TU104's 64 KB L1 partition (configurable; default 32 KB is also sufficient).

**Point data coalescing:** Consecutive threads load consecutive points
(`Px[i_start + t]`, `Px[i_start + t + 1]`, ...) — perfectly coalesced within a warp.

**On `__fdividef` vs full-precision division:** The AVX2 version uses full-precision
`_mm256_div_ps(r1, DT)`. The CUDA version uses `__fdividef(1.0f, DT)` — the fast
approximate reciprocal. This trades ~2 ULP of precision for higher throughput
(`div.approx.f32` vs `div.full.f32`). Since the result is only used to normalize
barycentric coordinates for a binary hit/miss test (not for computing distances or
positions), the reduced precision is acceptable. Boundary-case differences relative to
AVX2 are confined to rays whose barycentric coordinates are extremely close to 0 or 1.

---

## 5. Batching and Double-Buffered Transfer

### 5.1 Batch Size Calculation

At startup, after uploading scene data, calculate a conservative batch size that leaves
ample headroom for driver allocations, CUB temporaries, and fragmentation:

```
cudaMemGetInfo(&free_mem, &total_mem)

// --- Per-ray memory (both buffers counted separately for double-buffering) ---
const int E = EST_AVG_HITS;  // 10 (generous estimate for avg AABB hits per ray)
const int H = EST_HIT_RATE;  //  2 (generous estimate for avg point hits per ray)

per_ray_bytes = 24 * 4        // input: T1xyz..rD3
              + E * (4 + 2)   // work queue A: ray_idx(4) + sub_idx(2) per entry
              + E * (4 + 2)   // work queue B: sort output buffers (same size)
              + H * (4 + 4)   // hit output: ray_idx(4) + point_idx(4) per hit
              ;
// = 96 + 60 + 60 + 16 = 232 bytes/ray

// --- CUB temp buffer (shared between batches, allocated once per stream) ---
// Query actual requirement at runtime for all CUB operations; take the max.
size_t cub_sort_temp = 0, cub_rle_temp = 0, cub_scan_temp = 0;
cub::DeviceRadixSort::SortPairs(nullptr, cub_sort_temp,
    (uint16_t*)nullptr, (uint16_t*)nullptr,
    (uint32_t*)nullptr, (uint32_t*)nullptr,
    max_queue_estimate, 0, 16, (cudaStream_t)0);
cub::DeviceRunLengthEncode::Encode(nullptr, cub_rle_temp,
    (uint16_t*)nullptr, (uint16_t*)nullptr, (uint32_t*)nullptr,
    (uint32_t*)nullptr, max_queue_estimate, (cudaStream_t)0);
cub::DeviceScan::ExclusiveSum(nullptr, cub_scan_temp,
    (uint32_t*)nullptr, (uint32_t*)nullptr,
    (int)(n_sub + 1), (cudaStream_t)0);
cub_temp_per_stream = (size_t)(max({cub_sort_temp, cub_rle_temp, cub_scan_temp}) * 1.2)

// --- Segment table (per stream, sized by n_sub — small fixed cost) ---
seg_table_per_stream = n_sub * (2 + 4 + 4 + 4)  // sub_idx(2) + run_len(4)
                     + (n_sub + 1) * (4 + 4)     //   + seg_offset(4) + chunk_offset(4)
                     + 4 + 4                      //   + num_segments(4) + total_chunks(4)
                     + n_sub * 4;                 //   + chunks_per_seg(4)
// For n_sub = 4000: ~112 KB per stream — negligible.

// --- Use only 75% of free memory ---
// Leaves 25% headroom for: CUDA runtime overhead, driver allocations,
// fragmentation from cudaMalloc, and any under-estimates above.
usable = free_mem * 0.75

// Double-buffering: 2 complete buffer sets + 2 CUB temp buffers + 2 seg tables
batch_size = (usable - 2 * cub_temp_per_stream - 2 * seg_table_per_stream)
           / (2 * per_ray_bytes)
batch_size = min(batch_size, n_ray)
batch_size = (batch_size / 256) * 256   // align to block size
```

**Worked example (RTX 2070, 8 GB, large outdoor scene):**

```
free_mem ≈ 7950 MB  (after 48 MB scene upload + runtime overhead)
usable   = 7950 * 0.75 = 5963 MB
2 × cub_temp ≈ 10–16 MB
2 × seg_table ≈ 0.2 MB  (for n_sub = 4000)
batch_size = (5963 - 16 - 0.2) MB / (2 × 232 bytes) ≈ 12.8M rays
```

For 100M total rays → ~8 batches.

**Queue capacity per stream:**

```
queue_capacity = batch_size * E    // 10 × batch_size
```

**Hit output capacity per stream:**

```
hit_capacity = batch_size * H      // 2 × batch_size
```

If `h_queue_tail` exceeds `queue_capacity` after Kernel 1, synchronize *both* streams,
reallocate *all* queue-related buffers (both streams) at `2 × h_queue_tail`, and re-run
the current batch from the counter reset. Both streams must be synchronized because the other
stream's estimate is likely also wrong. Same protocol for hit output overflow.

### 5.2 Double-Buffered Async Pipeline

Use two CUDA streams to overlap H→D transfer of batch N+1 with compute on batch N.
Each stream has its own set of per-ray device buffers, work queue buffers, hit output
buffers, and CUB temp buffer.

```
stream[0], stream[1]              // Two CUDA streams
d_ray_input[2]                    // Two sets of 24 ray SoA arrays
d_work_queue[2]                   // Two work queues (A/B pairs)
d_seg_table[2]                    // Two segment table buffer sets
d_hit_output[2]                   // Two hit output buffers
d_cub_temp[2]                     // Two CUB temp buffers
h_ray_input_pinned[2]             // Pinned host memory for async H→D
h_hit_output_pinned[2]            // Pinned host memory for async D→H
```

**Pipeline (pseudocode):**

```
// Stage batch 0 into pinned memory BEFORE the loop
copy user ray arrays [0 .. batch_size) → h_ray_input_pinned[0]   // CPU memcpy

for batch_idx = 0 .. n_batches - 1:
    s  = batch_idx % 2            // current stream index
    sp = 1 - s                    // previous stream index (always valid, no negative modulo)

    // Wait for previous batch's D→H to finish (if any)
    if batch_idx > 0:
        cudaStreamSynchronize(stream[sp])
        // Scatter hits into p_hit[] vectors on host (CPU work)
        for i = 0 .. h_hit_count[sp] - 1:
            actual_ray = prev_batch_offset + h_hit_ray_pinned[sp][i]
            p_hit[actual_ray].push_back(h_hit_point_pinned[sp][i])

    // H→D: upload ray data for this batch
    cudaMemcpyAsync(d_ray_input[s], h_ray_input_pinned[s],
                    batch_bytes_in, cudaMemcpyHostToDevice, stream[s])

    // Update per-stream ray pointer table for this stream's buffers
    cudaMemcpyAsync(d_ray_ptrs[s], h_ray_ptrs[s], 24 * sizeof(float*),
                    cudaMemcpyHostToDevice, stream[s])

    // Reset counters
    cudaMemsetAsync(d_queue_tail[s],    0, sizeof(uint32_t), stream[s])
    cudaMemsetAsync(d_hit_count[s],     0, sizeof(uint32_t), stream[s])
    cudaMemsetAsync(d_overflow_flag[s], 0, sizeof(uint32_t), stream[s])

    // Kernel 1: AABB test + enqueue
    aabb_test_and_enqueue<<<grid1, 256, 0, stream[s]>>>(...)

    // Read queue_tail back to host (small sync point, 4 bytes)
    cudaMemcpyAsync(&h_queue_tail[s], d_queue_tail[s], 4,
                    cudaMemcpyDeviceToHost, stream[s])
    cudaStreamSynchronize(stream[s])  // need queue_tail to size sort + segment ops

    // Check for queue overflow
    if h_queue_tail[s] > queue_capacity[s]:
        handle_overflow(...)  // sync both streams, realloc both, re-run

    if h_queue_tail[s] > 0:
        if n_sub == 1:
            // Fast path (Section 6.4): skip sort + segment table entirely.
            // Write minimal segment table via cudaMemcpyAsync.
            h_total_chunks[s] = (h_queue_tail[s] + RAYS_PER_BLOCK - 1) / RAYS_PER_BLOCK
            write_minimal_segment_table(stream[s], h_queue_tail[s], h_total_chunks[s])
        else:
            // CUB sort by sub_idx
            cub_sort_work_queue(stream[s], h_queue_tail[s], ...)

            // Segment table construction (Section 6)
            build_segment_table(stream[s], h_queue_tail[s], ...)
            // → produces total_chunks on host (small D→H + sync, 4 bytes)

        // Kernel 2: point intersection (dynamic smem sized by max_local_hits)
        smem_bytes = 2 * max_local_hits * sizeof(uint32_t)
        point_intersect<<<h_total_chunks[s], 256, smem_bytes, stream[s]>>>(
            ..., tile_size, max_local_hits, ...)

    // --- CPU staging window: GPU is busy with sort + segment ops + K2 ---
    // Stage NEXT batch into pinned memory (overlaps with GPU work)
    if batch_idx + 1 < n_batches:
        next_s = (batch_idx + 1) % 2
        copy user ray arrays [next_start .. next_end) → h_ray_input_pinned[next_s]

    // Read hit_count and overflow_flag, then download hit buffers
    cudaMemcpyAsync(&h_hit_count[s], d_hit_count[s], 4,
                    cudaMemcpyDeviceToHost, stream[s])
    cudaMemcpyAsync(&h_overflow_flag[s], d_overflow_flag[s], 4,
                    cudaMemcpyDeviceToHost, stream[s])
    cudaStreamSynchronize(stream[s])   // need hit_count to size D→H copy

    // Check for tile-level hit buffer overflow — auto-retry with doubled parameters.
    // h_hit_count reflects the true total (including lost hits) because the flush
    // advances the global counter by s_hit_count, not the clamped flush_count.
    if h_overflow_flag[s] != 0:
        tile_size *= 2
        max_local_hits = tile_size * 256   // maintain MAX_LOCAL_HITS >= TILE_SIZE × 256
        // Re-run Kernel 2 for this batch (sort + segment table are still valid).
        // Also reallocate hit output if h_hit_count exceeds hit_capacity.
        if h_hit_count[s] > hit_capacity[s]:
            handle_hit_overflow(...)       // realloc hit buffers for both streams
        cudaMemsetAsync(d_hit_count[s],     0, sizeof(uint32_t), stream[s])
        cudaMemsetAsync(d_overflow_flag[s], 0, sizeof(uint32_t), stream[s])
        smem_bytes = 2 * max_local_hits * sizeof(uint32_t)
        point_intersect<<<h_total_chunks[s], 256, smem_bytes, stream[s]>>>(
            ..., tile_size, max_local_hits, ...)
        // Re-read hit_count (overflow_flag should now be 0)
        cudaMemcpyAsync(&h_hit_count[s], d_hit_count[s], 4,
                        cudaMemcpyDeviceToHost, stream[s])
        cudaStreamSynchronize(stream[s])

    // Check for hit output overflow
    if h_hit_count[s] > hit_capacity[s]:
        handle_overflow(...)

    cudaMemcpyAsync(h_hit_ray_pinned[s], d_hit_ray_idx[s],
                    h_hit_count[s] * 4, cudaMemcpyDeviceToHost, stream[s])
    cudaMemcpyAsync(h_hit_point_pinned[s], d_hit_point_idx[s],
                    h_hit_count[s] * 4, cudaMemcpyDeviceToHost, stream[s])

// Drain last batch
cudaStreamSynchronize(stream[(n_batches - 1) % 2])
// scatter last batch's hits into p_hit[]
```

**Sync points per batch:** Three small D→H copies + syncs (four if `n_sub > 1`):
1. `queue_tail` readback (4 bytes) — needed to size the sort and segment ops.
2. `total_chunks` readback (4 bytes) — needed to size the Kernel 2 grid. This sync
   is folded into the segment table construction (Section 6) and overlaps with the
   CUB operations on the same stream. Skipped when `n_sub == 1` (computed on host).
3. `hit_count` + `overflow_flag` readback (8 bytes) — needed to size the D→H hit
   transfer and check for tile buffer overflow. If `overflow_flag` is set, Kernel 2
   is re-run with doubled `tile_size` / `max_local_hits` (one additional sync for
   the re-run's `hit_count` readback). This retry path is expected to be extremely
   rare in practice.

Total sync overhead: ~12 µs per batch, negligible relative to kernel execution.
When `h_queue_tail == 0`, the sort, segment ops, and Kernel 2 are all skipped — only
the counter reset and the queue_tail readback execute.

**Note on pinned memory staging:** The user's ray data lives in unpinned host arrays.
The `memcpy` from user arrays into `h_ray_input_pinned` is a CPU operation that must
complete before `cudaMemcpyAsync` begins for that batch. Batch 0 is staged before the
loop. Subsequent batches are staged in the window between the `queue_tail` sync and the
`hit_count` sync — during this window, the GPU is busy with sort + Kernel 2
(~10–50 ms), which is ample time for the CPU to memcpy the next batch.

---

## 6. Work Queue Sorting and Segment Table Construction

### 6.1 Problem

Without sorting, consecutive blocks in Kernel 2 process random sub-clouds. Each
sub-cloud's point data (500–1000 points × 12 bytes = 6–12 KB) is loaded from global
memory. On TU104 (4 MB L2), blocks from different SMs accessing different sub-clouds
thrash the L2 cache. Furthermore, multi-ray-per-block batching (Section 4.2) requires
that all work queue entries for the same sub-cloud are contiguous.

### 6.2 Sort

After Kernel 1, sort the work queue by `sub_idx` (primary key only — no secondary
sort on `ray_idx`). This groups all work items for the same sub-cloud together.

**Implementation:** Use `cub::DeviceRadixSort::SortPairs` with `uint16_t` keys
(`sub_idx`) and `uint32_t` values (`ray_idx`). Use CUB's `DoubleBuffer` wrapper:

```cpp
cub::DoubleBuffer<uint16_t> d_keys(d_wq_sub_idx_A, d_wq_sub_idx_B);
cub::DoubleBuffer<uint32_t> d_values(d_wq_ray_idx_A, d_wq_ray_idx_B);

// end_bit: ceil(log2(n_sub)), minimum 1 (handles n_sub == 1 where log2 = 0)
int num_sub_bits = max(1, (int)ceil(log2((double)n_sub)));

cub::DeviceRadixSort::SortPairs(
    d_cub_temp, cub_temp_bytes,
    d_keys, d_values,
    h_queue_tail,
    0,                     // begin_bit: sort from bit 0
    num_sub_bits,          // end_bit: clamped to >= 1
    stream);

// After sort, current buffers are:
//   sorted_sub_idx = d_keys.Current()
//   sorted_ray_idx = d_values.Current()
```

**Sort bit range:** Only `ceil(log2(n_sub))` bits need to be sorted, clamped to a
minimum of 1 (CUB requires `end_bit > begin_bit`). The `n_sub = 1` case is common
(single sub-cloud, all points in one AABB) — it still requires 1 radix pass even
though all keys are 0, because the sort also serves as the input to RunLengthEncode:

| n_sub   | Bits to sort | Radix passes |
| ------- | ------------ | ------------ |
| 1       | 1            | 1            |
| ≤ 16    | 4            | 1            |
| ≤ 256   | 8            | 2            |
| ≤ 4096  | 12           | 3            |
| ≤ 65536 | 16           | 4            |

**Why no secondary sort on `ray_idx`:** Sorting a 64-bit composite key
`(sub_idx << 32 | ray_idx)` would also sort by `ray_idx` within each sub-cloud group.
This does not improve locality: point data access depends only on `sub_idx`, and ray
data is loaded once per ray into shared memory (no intra-block scatter concern). The
secondary sort doubles the key size and quadruples radix passes for zero benefit.

### 6.3 Segment Table Construction

After sorting, build a segment table that maps `blockIdx.x` in Kernel 2 to a
(segment, chunk-within-segment) pair. This requires three CUB/kernel steps, all on
the same stream:

**Step 1 — RunLengthEncode:** Identify contiguous runs of equal `sub_idx`:

```cpp
cub::DeviceRunLengthEncode::Encode(
    d_cub_temp, cub_temp_bytes,
    sorted_sub_idx,            // input: sorted keys
    d_seg_sub_idx,             // output: unique sub_idx per segment
    d_seg_run_len,             // output: run length per segment
    d_seg_num_segments,        // output: number of segments (single uint32)
    h_queue_tail,              // num items
    stream);

// Read num_segments to host (4 bytes, small sync)
cudaMemcpyAsync(&h_num_segments, d_seg_num_segments, 4,
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

**Step 2 — Kernel 1b: compute chunk counts and write sentinels:**

A small kernel with `ceil((h_num_segments + 1) / 256)` blocks, 256 threads each:

```
// Thread i computes:
if i < num_segments:
    chunks_per_seg[i] = (run_len[i] + RAYS_PER_BLOCK - 1) / RAYS_PER_BLOCK
if i == num_segments:
    // Write sentinel values for the subsequent ExclusiveSum calls.
    // CUB processes num_segments + 1 elements; the sentinel element must be 0
    // so that the prefix sum produces the correct total at position [num_segments].
    run_len[num_segments]       = 0
    chunks_per_seg[num_segments] = 0
```

**Step 3 — Two exclusive prefix sums:**

```cpp
// seg_offset: cumulative ray offset into the sorted queue
// seg_offset[0] = 0, seg_offset[num_segments] = queue_tail (sentinel)
cub::DeviceScan::ExclusiveSum(
    d_cub_temp, cub_temp_bytes,
    d_seg_run_len, d_seg_offset,
    h_num_segments + 1,    // +1 to include sentinel (run_len appended with 0)
    stream);

// chunk_offset: cumulative block offset for Kernel 2 grid mapping
// chunk_offset[0] = 0, chunk_offset[num_segments] = total_chunks
cub::DeviceScan::ExclusiveSum(
    d_cub_temp, cub_temp_bytes,
    d_chunks_per_seg, d_seg_chunk_offset,
    h_num_segments + 1,
    stream);

// Read total_chunks to host (4 bytes, small sync — needed for K2 grid size)
cudaMemcpyAsync(&h_total_chunks, &d_seg_chunk_offset[h_num_segments], 4,
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

**Note on `seg_offset` sentinel:** Kernel 1b explicitly writes `run_len[num_segments] = 0`
and `chunks_per_seg[num_segments] = 0` before the prefix sums run. The `ExclusiveSum`
over `run_len[0..num_segments]` (i.e. `num_segments + 1` elements) then produces
`seg_offset[num_segments] = queue_tail` as the sum of all run lengths. Similarly,
`chunk_offset[num_segments] = total_chunks`. These sentinels let Kernel 2 compute
`ray_end` for the last segment without a special case.

**Cost:** All three steps operate on arrays of length ≤ `n_sub` (≤ 65535). The
RunLengthEncode processes `h_queue_tail` elements but is I/O bound and overlaps with
the sort's memory traffic pattern. Total wall time for the segment table: ~0.1–0.5 ms
for typical scenes — negligible relative to Kernel 2.

### 6.4 Fast Path for `n_sub == 1`

When `n_sub == 1`, every ray that passes the AABB test hits the same (only) sub-cloud.
The sort is a no-op (all keys are 0) but still costs one radix pass over the full
queue. The segment table construction is similarly trivial but still launches CUB
RunLengthEncode + two ExclusiveSums + Kernel 1b. All of this can be skipped entirely
with a host-side branch, because the segment table values are known analytically:

```cpp
if (n_sub == 1) {
    // Skip sort entirely — all entries are already in one segment.
    // The unsorted work queue is valid as-is (all sub_idx = 0).

    h_total_chunks = (h_queue_tail + RAYS_PER_BLOCK - 1) / RAYS_PER_BLOCK;

    // Write the minimal segment table directly via cudaMemcpyAsync:
    //   seg_sub_idx    = { 0 }
    //   seg_offset     = { 0, queue_tail }         (2 values)
    //   chunk_offset   = { 0, total_chunks }       (2 values)
    //   num_segments   = 1

    uint16_t h_seg_sub = 0;
    uint32_t h_seg_off[2]   = { 0, h_queue_tail };
    uint32_t h_chunk_off[2] = { 0, h_total_chunks };
    uint32_t h_num_seg = 1;

    cudaMemcpyAsync(d_seg_sub_idx,     &h_seg_sub,   sizeof(uint16_t),     cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_seg_offset,      h_seg_off,    2 * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_seg_chunk_offset,h_chunk_off,  2 * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_seg_num_segments,&h_num_seg,   sizeof(uint32_t),     cudaMemcpyHostToDevice, stream);

    // Kernel 2 reads sorted_ray_idx — point it to the unsorted buffer (wq_ray_idx_A).
    // No sort was performed, so DoubleBuffer was not used; the current buffer is A.
}
```

This saves ~0.5–1 ms per batch for the single-sub-cloud case. It also avoids the
`num_segments` D→H sync that would otherwise be needed for the segment table pipeline.
The `n_sub == 1` case is common: a single dense receiver plane with all points in one
AABB.

For `n_sub > 1`, the full sort + segment table pipeline runs as described above.

### 6.5 Cost Estimates

**Sort (TU104, 16-bit keys, 3 radix passes for n_sub ≤ 4096):**

| Queue size | Sort time (approx) |
| ---------- | ------------------ |
| 3M         | ~0.5 ms            |
| 30M        | ~3 ms              |
| 90M        | ~9 ms              |

**Segment table (all scenes):** ~0.1–0.5 ms (dominated by RunLengthEncode on the
sorted queue; the prefix sums on n_sub-length arrays are near-instant).

**This is the default code path**, not an optional optimization. An unsorted path can
be retained as a debugging/profiling fallback behind a compile-time flag.

---

## 7. CUDA-Specific Implementation Notes

### 7.1 Division Precision

The AVX2 version uses full-precision `_mm256_div_ps(r1, DT)`. The CUDA version uses
`__fdividef(1.0f, DT)` — the fast approximate reciprocal (≥2 ULP accuracy for normal
inputs). This is sufficient for the barycentric hit/miss decision, where the exact
value of U and V matters far less than their sign and sum. The `--use_fast_math` flag
(Section 11) enables FMA contraction and denormal flushing globally; `__fdividef` is
an explicit intrinsic that produces the same fast-path `div.approx.f32` instruction
regardless of compiler flags, making the behavior self-documenting.

Note: `__fdividef` flushes denormals and loses precision when `|DT|` is very large
(>2^126). In practice, DT is a mixed dot product of edge vectors and the ray normal —
it is never near the denormal or overflow range for valid geometry.

### 7.2 Warp-Level Queue Compaction

```cuda
__device__ __forceinline__
void enqueue_warp(bool has_hit, uint32_t ray_idx, uint16_t sub_idx,
                  uint32_t *wq_ray, uint16_t *wq_sub, uint32_t *queue_tail,
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
            wq_ray[pos] = ray_idx;
            wq_sub[pos] = sub_idx;
        }
        // atomicAdd already recorded the true count for host-side overflow
        // detection. Dropped writes are re-done after reallocation (Section 5.1).
    }
}
```

### 7.3 Block Size Selection

- **Kernel 1:** 256 threads/block. Memory-bandwidth-bound (ray loads, AABB broadcasts).
  256 gives good occupancy without excessive register pressure.
- **Kernel 1b:** 256 threads/block (trivial kernel over `num_segments` elements).
- **Kernel 2:** 256 threads/block. Each block cooperatively processes one chunk of up
  to `RAYS_PER_BLOCK` (default 32) rays against one sub-cloud. 256 threads are
  well-matched to sub-clouds of 500–1000 points (2–4 loop iterations per thread per
  ray). Profile with Nsight Compute and test 128 vs 256 vs 512 — the optimal choice
  depends on sub-cloud size distribution and register usage.
- **All kernels:** Block size must be a multiple of 32.
  `static_assert(BLOCK_SIZE % 32 == 0)`.

**`__launch_bounds__` on Kernel 2:** Annotate with
`__launch_bounds__(256, MIN_BLOCKS_PER_SM)`. On TU104 (64K registers per SM),
`MIN_BLOCKS_PER_SM = 4` yields `65536 / (256 × 4) = 64` registers per thread — more
than enough for the MT kernel (~40 registers) and ensures 4 blocks per SM for good
occupancy (1024 threads/SM = 50% of TU104's 2048-thread capacity). This prevents the
compiler from over-allocating registers (which would reduce occupancy to 2 blocks/SM)
while leaving ample headroom above the kernel's actual register demand.

```cuda
__global__ void __launch_bounds__(256, 4)
point_intersect(...) { ... }
```

### 7.4 Read-Only Data Cache and Aliasing

Declare all scene data pointers as `const float * __restrict__` in kernel signatures.
On compute capability 7.5 (TU104 / RTX 2070), this is sufficient for the compiler to
route loads through the read-only data cache (L1 texture path) via `LDG` instructions.

Per-ray data pointers must also be `__restrict__`. In Kernel 2, ray data is read-only
but shares the same address space as the write targets (`d_hit_ray_idx`,
`d_hit_point_idx`, `d_hit_count`). Without `__restrict__`, `nvcc` cannot prove that
ray loads don't alias with the atomic stores and may conservatively reload ray values
from L2 after each atomic instead of keeping them in registers.

---

## 8. Function Signature

The CUDA wrapper matches the AVX2 signature, with an additional parameter for GPU
selection:

```cpp
void qd_RPI_CUDA(
    const float *Px, const float *Py, const float *Pz,
    const size_t n_point,
    const unsigned *SCI,
    const float *Xmin, const float *Xmax,
    const float *Ymin, const float *Ymax,
    const float *Zmin, const float *Zmax,
    const size_t n_sub,
    const float *T1x, const float *T1y, const float *T1z,
    const float *T2x, const float *T2y, const float *T2z,
    const float *T3x, const float *T3y, const float *T3z,
    const float *Nx,  const float *Ny,  const float *Nz,
    const float *D1x, const float *D1y, const float *D1z,
    const float *D2x, const float *D2y, const float *D2z,
    const float *D3x, const float *D3y, const float *D3z,
    const float *rD1, const float *rD2, const float *rD3,
    const size_t n_ray,
    std::vector<unsigned> *p_hit,      // [n_ray] output hit lists (host)
    int gpu_id = 0                     // GPU device index
);
```

All input/output pointers are **host memory**. The function handles all GPU allocation,
H↔D transfers, batching, and cleanup internally. The caller sees the same interface as
the AVX2 version (plus `gpu_id`).

**GPU selection:** The wrapper calls `cudaSetDevice(gpu_id)` at the top.

**n_point alignment:** Unlike `qd_RPI_AVX2`, the CUDA version does **not** require
`n_point` to be a multiple of 8. Any `n_point >= 1` is valid.

**Early returns:** If `n_ray == 0`, return immediately. If `n_sub == 0` or
`n_point == 0`, clear all `p_hit` vectors on the host and return — no GPU allocation.

**n_sub range check:** If `n_sub > 65535`, throw `std::invalid_argument`. Without this
check, the `(uint16_t)i_sub` cast in Kernel 1's enqueue silently truncates indices
≥ 65536, causing incorrect point range lookups in Kernel 2.

**Resource cleanup (RAII):** Use a single RAII wrapper struct (e.g. `CudaBuffers`)
whose destructor calls `cudaFree` / `cudaFreeHost` on all allocations. This ensures
cleanup on normal completion, throws from range checks, CUDA API failures, and queue
overflow reallocation. The struct holds all device and pinned-host pointers as members,
initialized to `nullptr`. The destructor iterates all members and frees non-null
pointers.

**SCI handling:** Internally, the wrapper allocates a host array of length `n_sub + 1`,
copies the caller's `SCI[0..n_sub-1]`, sets entry `[n_sub] = (uint32_t)n_point`, and
uploads that to `d_SCI`.

---

## 9. Typical Workload Estimates

| Scene                | n_point | n_sub | n_ray | Batches (8 GB) | Queue size | K2 chunks (RPB=32) |
| -------------------- | ------- | ----- | ----- | -------------- | ---------- | ------------------ |
| Small indoor         | 30K     | 30    | 1M    | 1              | ~3M        | ~94K               |
| Medium (Bertramshof) | 200K    | 200   | 10M   | 1              | ~30M       | ~940K              |
| Large outdoor        | 4M      | 4000  | 100M  | ~8             | ~90M/batch | ~2.8M/batch        |

K2 chunks = sum over segments of ceil(run_length / RAYS_PER_BLOCK). Approximate as
queue_size / 32 when segments are large. The 32× block count reduction compared to the
one-block-per-work-item design significantly reduces scheduling overhead.

**CUB sort cost estimates (TU104, 16-bit keys, 3 radix passes for n_sub ≤ 4096):**

| Queue size | Sort time (approx) |
| ---------- | ------------------ |
| 3M         | ~0.5 ms            |
| 30M        | ~3 ms              |
| 90M        | ~9 ms              |

---

## 10. Testing and Validation Strategy

### 10.1 Numerical Equivalence

Run both `qd_RPI_AVX2` and `qd_RPI_CUDA` on the same inputs. Sort both per-ray hit
lists. Compare:

- **Hit lists:** must be identical sets per ray for the vast majority of cases. Both
  use float32 and the same mathematical sequence. Both versions check all three vertex
  distances (`d0 >= 0 && d1 >= 0 && d2 >= 0`). Minor differences are possible at
  boundary cases due to: (a) different FMA contraction rules between x86 and CUDA
  (`--fmad=true` by default enables FMA contraction beyond what the source specifies),
  and (b) the CUDA version's use of `__fdividef` for the barycentric normalization
  (≥2 ULP vs full precision on AVX2). These boundary differences should be rare and
  confined to rays whose barycentric coordinates are extremely close to 0 or 1.

- **n_point alignment:** The AVX2 version requires `n_point % 8 == 0` and processes
  8 points per SIMD step. The CUDA version processes points individually. If `n_point`
  is not a multiple of 8, the AVX2 version must be called with padded input for
  comparison. Ensure padding points are placed at `(0, 0, 0)` or a location that
  cannot be hit.

### 10.2 Edge Cases

- `n_ray = 0`, `n_sub = 0`, `n_point = 0` — early returns.
- `n_ray = 1` (single ray, all sub-clouds).
- `n_sub = 1` (single sub-cloud, all points — common case). Verify: fast path
  (Section 6.4) is taken, sort and segment table are skipped, Kernel 2 launches
  `ceil(queue_tail / RAYS_PER_BLOCK)` blocks. Results must match AVX2.
- Sub-cloud with 0 points (consecutive equal SCI entries).
- Sub-cloud with 1 point.
- `n_ray` values not multiples of 32 (e.g. 1, 33, 255) — partial warp safety.
- `n_point` values not multiples of 8 (e.g. 1, 7, 13, 100) — no SIMD alignment.
- Work queue overflow (artificially small `queue_capacity`). Verify re-run produces
  correct results.
- Hit output overflow (artificially small `hit_capacity`).
- Ray batch that produces zero AABB hits (sort + segment ops + Kernel 2 all skipped:
  `queue_tail == 0`).
- Ray that hits zero points in all sub-clouds.
- Ray that hits every point in a sub-cloud (stress test for tiled flush; verify all
  hits are captured across multiple tile flushes and `d_overflow_flag` is not set).
- Dense receiver plane at grazing incidence (thousands of hits per ray per sub-cloud;
  verify tiled flush produces correct results and `d_overflow_flag` is not set with
  `TILE_SIZE = 2`, `MAX_LOCAL_HITS = 512`).
- Segment with fewer than `RAYS_PER_BLOCK` rays (last chunk in segment is partial).

### 10.3 Performance Benchmarks

Compare wall-clock time of `qd_RPI_CUDA` (including all H↔D transfers) vs
`qd_RPI_AVX2` (with OpenMP on all cores). Target: CUDA should be faster for
`n_ray >= 1M`. For `n_ray < 100K`, AVX2 may win due to transfer overhead — this is
acceptable.

**Key metrics to profile with Nsight Compute on Kernel 2:**
- L1 hit rate on point data (target: high — multi-ray batching should give L1 reuse
  across rays within a chunk)
- L2 hit rate (target: >80% with sorting — inter-block reuse across chunks)
- Occupancy (target: >50%)
- Warp stall reasons (memory vs compute vs sync)
- Point data load coalescing (global load transactions vs requests — expect ~1×
  amplification on point data, ~24× on scattered ray loads in shared memory init)
- Effect of `RAYS_PER_BLOCK`: test 16, 32, 64 — measure L1 hit rate vs sync overhead

**Phase 1 / Phase 2 balance check:** After profiling, verify that Kernel 1 + sort +
segment table time is ≤30% of total kernel time. If it exceeds this, `n_sub` is too
large for the current scene.

---

## 11. File Structure

```
quadriga-lib/
├── src/
│   ├── ray_point_intersect_avx2.cpp     // Existing
│   ├── ray_point_intersect_cuda.cu      // New: kernels + wrapper
│   └── ray_point_intersect_cuda.cuh     // New: kernel signatures, device helpers
├── tests/
│   └── catch2_test_rpi_cuda.cpp         // New: Catch2 tests
└── CMakeLists.txt                       // Add CUDA language, find CUB
```

CUB is header-only and ships with the CUDA Toolkit (>= 11.0). No external dependencies
beyond the CUDA Toolkit.

**Compilation flags:** `-O3 --use_fast_math -arch=sm_75` (Turing; adjust for target).
`--use_fast_math` enables `--fmad=true` (FMA contraction, matching the AVX2 code's
reliance on FMA throughput), `--ftz=true` (flush denormals to zero), and
`--prec-div=false` / `--prec-sqrt=false` (fast approximate division and sqrt). The
explicit `__fdividef` intrinsic in Kernel 2 (Section 7.1) produces the same fast-path
instruction regardless of `--prec-div`, so the flag setting is consistent. Guard the
entire file with `#ifdef QUADRIGA_USE_CUDA` / `#endif`.

---

## 12. Implementation Order

1. **Skeleton:** `qd_RPI_CUDA` wrapper with `cudaSetDevice(gpu_id)`, RAII cleanup
   struct (Section 8), GPU alloc, batching loop, H↔D transfers, pinned memory, stream
   creation. Kernels are stubs. Verify memory management with `cuda-memcheck`. Include
   the SCI sentinel setup (`SCI[n_sub] = n_point`). Include the conservative memory
   budget from Section 5.1 (with runtime CUB temp query). Add early returns for
   `n_ray == 0`, `n_sub == 0`, and `n_point == 0`. Add `n_sub > 65535` range check.
   Counter reset uses `cudaMemsetAsync` (Section 4.0) for `queue_tail`, `hit_count`,
   and `overflow_flag`. Allocate per-stream `d_ray_ptrs[s]` device arrays
   (192 bytes each via `cudaMalloc`) and add `cudaMemcpyAsync` call to update them
   before Kernel 2 launches on each stream. Note: no
   `n_point % 8` check — the CUDA version accepts any `n_point >= 1`.

2. **Kernel 1 (AABB test):** Implement wavefront-advancement AABB overlap test +
   warp-level enqueue with partial-warp safety (no early return before `__ballot_sync`,
   `static_assert` on block size) and `pos < queue_capacity` overflow guard
   (Section 7.2). Validate by comparing the set of `(ray_idx, sub_idx)` pairs against
   the AVX2 version's `p_sub_hit` results. Test with `n_ray` values that are not
   multiples of 32. Test queue overflow by temporarily setting `queue_capacity` to a
   very small value and verifying that the re-run produces correct results.

3. **CUB sort integration + `n_sub == 1` fast path:** Add work queue sorting by
   `sub_idx` (uint16 key) after Kernel 1. Use `DoubleBuffer` and
   `begin_bit=0, end_bit=max(1, ceil(log2(n_sub)))`. Implement the `n_sub == 1` fast
   path (Section 6.4) that skips the sort and segment table entirely, writing the
   minimal segment table via `cudaMemcpyAsync`. Verify that the sorted queue contains
   the same entries as the unsorted queue (set equality). Test with `n_sub = 1`
   (fast path taken) and `n_sub = 2` (sort path taken, end_bit = 1).

4. **Segment table construction:** Implement RunLengthEncode + Kernel 1b +
   ExclusiveSum pipeline (Section 6.3). Kernel 1b always writes sentinel values
   (`run_len[num_segments] = 0`, `chunks_per_seg[num_segments] = 0`). Verify:
   `seg_offset[num_segments] == queue_tail`, `sum(chunks_per_segment) == total_chunks`,
   segment boundaries match sorted queue transitions. Test with every sub-cloud hit by
   exactly one ray (num_segments = queue_tail, 1 ray per chunk).

5. **Kernel 2 (point intersection):** Implement multi-ray-per-block Möller–Trumbore
   with binary search blockIdx → segment mapping, outer ray loop, cooperative tiled
   point processing with flush at tile boundaries (Section 4.2), shared memory ray
   load via per-stream `d_ray_ptrs` device pointer table, and `d_overflow_flag`
   diagnostic.
   Check all 3 vertex distances in the hit condition. Add `__launch_bounds__(256, 4)`
   (Section 7.3). Validate hit lists against AVX2. Stress-test with a dense receiver
   plane to verify tiled flush correctness under high hit density.

6. **Double buffering:** Add second stream + pinned memory. Include pre-loop staging
   of batch 0. Place next-batch CPU staging in the window between the `queue_tail` sync
   and the `hit_count` sync (overlapping with GPU sort + segment ops + K2). Ensure queue
   overflow handler syncs both streams. Each stream's `d_ray_ptrs` is updated via
   `cudaMemcpyAsync` independently — no cross-stream hazard. Verify correctness is
   unchanged.

7. **Profile and optimize:** Nsight Compute on Kernels 1 and 2.
   - Check Phase 1 / Phase 2 balance (K1 + sort + segment ops vs K2).
   - Check L1 hit rate on K2 point data (multi-ray batching benefit).
   - Check L2 hit rate on K2 (should be >80% with sorting).
   - Check point data coalescing.
   - Check occupancy, warp stall reasons.
   - Test block sizes 128/256/512 for Kernel 2.
   - Test `RAYS_PER_BLOCK` 16/32/64 — measure L1 reuse vs sync overhead.
   - Test `TILE_SIZE` 1/2/4 — measure flush overhead vs buffer safety.

---

## 13. Mapping AVX2 → CUDA: Line-by-Line Reference

### Kernel 1 ↔ AVX2 lines 78–239

| AVX2                                                            | CUDA Kernel 1                                                        |
| --------------------------------------------------------------- | -------------------------------------------------------------------- |
| Lines 79–108: broadcast ray floats into `__m256`                | Each thread loads its ray's 24 floats from SoA arrays (coalesced)    |
| Lines 116–124: precompute `ox0*nx`, etc.                        | Same precomputation, scalar per thread                               |
| Lines 126–239: loop over `n_sub_s` in steps of 8                | Loop over `n_sub` one-at-a-time per thread (no SIMD grouping needed) |
| Lines 129–134: load AABB bounds                                 | Load 6 floats from global (cached in L2, broadcast)                  |
| Lines 137–142: apply slack (±1e-5f)                             | Same: subtract/add 1e-5f                                             |
| Lines 145–147: enumerate 8 corners                              | Same: 8 corner combos from min/max                                   |
| Lines 158–219: advance 3 vertices to each corner, track min/max | Same math, scalar floats, use `fmaf()`                               |
| Lines 223–233: 6-way overlap test                               | Same comparisons, combine with `&&`                                  |
| Lines 238–239: store hit/miss to array                          | On hit → warp-level `atomicAdd` + write to work queue                |

### Kernel 2 ↔ AVX2 lines 244–361

| AVX2                                                      | CUDA Kernel 2                                                      |
| --------------------------------------------------------- | ------------------------------------------------------------------ |
| Outer ray loop (`omp for`)                                | Outer ray loop within chunk (RAYS_PER_BLOCK rays per block)        |
| Line 247: skip if `p_sub_hit == 0`                        | Not needed — work queue contains only hits                         |
| Lines 250–251: look up point range from SCI               | Same, from `sub_idx` via segment table. SCI sentinel elim. branch. |
| Line 253: point loop in steps of 8                        | 256 threads cooperatively, stride 256, tiled flush (Section 4.2)   |
| Lines 256–258: load point `Px, Py, Pz`                    | Each thread loads one point (coalesced, L1 reuse across rays)      |
| Lines 261–266: compute distance `d` for vertex 0          | Same math, scalar, using `fmaf()`                                  |
| Lines 269–271: advance vertex 0 → `V`                     | Same                                                               |
| Lines 274–303: compute edges `e1`, `e2` via vertices 1, 2 | Same                                                               |
| Lines 306–308: compute `t = point - V`                    | Same                                                               |
| Lines 311–337: cross products → `U`, `V`, `DT`            | Same math, matching (N × e2) sign convention                       |
| Lines 340–341: `DT = 1/DT`, scale `U`, `V`                | `__fdividef(1.0f, DT)` (fast approx), see Section 7.1              |
| Lines 345–355: check conditions `U≥0, V≥0, U+V≤1, d0≥0, d1≥0, d2≥0` | Same: all 3 vertex distances checked                               |
| Lines 358–360: append to `p_hit[i_ray]` vector            | Block-local shared mem buffer → tiled flush to global hit output   |