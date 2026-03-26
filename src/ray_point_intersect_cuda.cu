// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "quadriga_lib_cuda_functions.hpp"

// ============================================================================
// Configuration constants
// ============================================================================

#define K1_BLOCK_SIZE 256
#define K1B_BLOCK_SIZE 256
#define K2_BLOCK_SIZE 256
#define RAYS_PER_BLOCK 32
#define EST_AVG_HITS 10
#define EST_HIT_RATE 2
#define N_RAY_ATTRS 24

static_assert(K1_BLOCK_SIZE % 32 == 0, "Kernel 1 block size must be a multiple of 32");
static_assert(K2_BLOCK_SIZE % 32 == 0, "Kernel 2 block size must be a multiple of 32");

// ============================================================================
// CUDA error checking macro
// ============================================================================

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess)                                    \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                     cudaGetErrorString(err) +     \
                                     " at " + __FILE__ + ":" +     \
                                     std::to_string(__LINE__));    \
    } while (0)

// ============================================================================
// RAII cleanup struct (Section 8)
// Holds all device and pinned-host pointers; destructor frees non-null ones.
// ============================================================================

struct CudaBuffers
{
    // --- Scene data (uploaded once) ---
    float *d_Px = nullptr, *d_Py = nullptr, *d_Pz = nullptr;
    uint32_t *d_SCI = nullptr;
    float *d_Xmin = nullptr, *d_Xmax = nullptr;
    float *d_Ymin = nullptr, *d_Ymax = nullptr;
    float *d_Zmin = nullptr, *d_Zmax = nullptr;

    // --- Per-stream buffers (2 streams) ---

    // Ray input (24 SoA arrays per stream)
    float *d_ray[2][N_RAY_ATTRS] = {};

    // Ray pointer table (device array of 24 float*, per stream)
    const float **d_ray_ptrs[2] = {nullptr, nullptr};

    // Work queue A/B
    uint32_t *d_wq_ray_idx_A[2] = {}, *d_wq_ray_idx_B[2] = {};
    uint16_t *d_wq_sub_idx_A[2] = {}, *d_wq_sub_idx_B[2] = {};

    // Atomic counters (each in own cudaMalloc for cache-line isolation)
    uint32_t *d_queue_tail[2] = {};
    uint32_t *d_hit_count[2] = {};
    uint32_t *d_overflow_flag[2] = {};

    // CUB temp buffer
    void *d_cub_temp[2] = {nullptr, nullptr};
    size_t cub_temp_bytes[2] = {0, 0};

    // Segment table
    uint16_t *d_seg_sub_idx[2] = {};
    uint32_t *d_seg_run_len[2] = {};
    uint32_t *d_seg_num_segments[2] = {};
    uint32_t *d_seg_offset[2] = {};
    uint32_t *d_chunks_per_seg[2] = {};
    uint32_t *d_seg_chunk_offset[2] = {};
    uint32_t *d_total_chunks[2] = {};

    // Hit output
    uint32_t *d_hit_ray_idx[2] = {};
    uint32_t *d_hit_point_idx[2] = {};

    // Pinned host memory
    float *h_ray_pinned[2] = {nullptr, nullptr}; // N_RAY_ATTRS * batch_size floats
    uint32_t *h_hit_ray_pinned[2] = {nullptr, nullptr};
    uint32_t *h_hit_point_pinned[2] = {nullptr, nullptr};

    // Streams
    cudaStream_t stream[2] = {nullptr, nullptr};

    // Sizes for cleanup
    size_t batch_size = 0;
    uint32_t queue_capacity = 0;
    uint32_t hit_capacity = 0;

    ~CudaBuffers()
    {
        // Scene data
        if (d_Px)
            cudaFree(d_Px);
        if (d_Py)
            cudaFree(d_Py);
        if (d_Pz)
            cudaFree(d_Pz);
        if (d_SCI)
            cudaFree(d_SCI);
        if (d_Xmin)
            cudaFree(d_Xmin);
        if (d_Xmax)
            cudaFree(d_Xmax);
        if (d_Ymin)
            cudaFree(d_Ymin);
        if (d_Ymax)
            cudaFree(d_Ymax);
        if (d_Zmin)
            cudaFree(d_Zmin);
        if (d_Zmax)
            cudaFree(d_Zmax);

        for (int s = 0; s < 2; ++s)
        {
            for (int a = 0; a < N_RAY_ATTRS; ++a)
                if (d_ray[s][a])
                    cudaFree(d_ray[s][a]);

            if (d_ray_ptrs[s])
                cudaFree((void *)d_ray_ptrs[s]);

            if (d_wq_ray_idx_A[s])
                cudaFree(d_wq_ray_idx_A[s]);
            if (d_wq_ray_idx_B[s])
                cudaFree(d_wq_ray_idx_B[s]);
            if (d_wq_sub_idx_A[s])
                cudaFree(d_wq_sub_idx_A[s]);
            if (d_wq_sub_idx_B[s])
                cudaFree(d_wq_sub_idx_B[s]);

            if (d_queue_tail[s])
                cudaFree(d_queue_tail[s]);
            if (d_hit_count[s])
                cudaFree(d_hit_count[s]);
            if (d_overflow_flag[s])
                cudaFree(d_overflow_flag[s]);

            if (d_cub_temp[s])
                cudaFree(d_cub_temp[s]);

            if (d_seg_sub_idx[s])
                cudaFree(d_seg_sub_idx[s]);
            if (d_seg_run_len[s])
                cudaFree(d_seg_run_len[s]);
            if (d_seg_num_segments[s])
                cudaFree(d_seg_num_segments[s]);
            if (d_seg_offset[s])
                cudaFree(d_seg_offset[s]);
            if (d_chunks_per_seg[s])
                cudaFree(d_chunks_per_seg[s]);
            if (d_seg_chunk_offset[s])
                cudaFree(d_seg_chunk_offset[s]);
            if (d_total_chunks[s])
                cudaFree(d_total_chunks[s]);

            if (d_hit_ray_idx[s])
                cudaFree(d_hit_ray_idx[s]);
            if (d_hit_point_idx[s])
                cudaFree(d_hit_point_idx[s]);

            if (h_ray_pinned[s])
                cudaFreeHost(h_ray_pinned[s]);
            if (h_hit_ray_pinned[s])
                cudaFreeHost(h_hit_ray_pinned[s]);
            if (h_hit_point_pinned[s])
                cudaFreeHost(h_hit_point_pinned[s]);

            if (stream[s])
                cudaStreamDestroy(stream[s]);
        }
    }
};

// ============================================================================
// Device helper: warp-level enqueue (Section 7.2)
// All 32 lanes in the warp MUST execute this function.
// Invalid threads (past batch end) must pass has_hit = false.
// ============================================================================

__device__ __forceinline__ void enqueue_warp(bool has_hit, uint32_t ray_idx, uint16_t sub_idx,
                                             uint32_t *wq_ray, uint16_t *wq_sub, uint32_t *queue_tail,
                                             uint32_t queue_capacity)
{
    uint32_t hit_mask = __ballot_sync(0xFFFFFFFF, has_hit);
    if (hit_mask == 0u)
        return;

    uint32_t lane = threadIdx.x & 31;
    uint32_t count = __popc(hit_mask);
    uint32_t warp_base;

    if (lane == 0)
        warp_base = atomicAdd(queue_tail, count);
    warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

    uint32_t local_offset = __popc(hit_mask & ((1u << lane) - 1u));
    if (has_hit)
    {
        uint32_t pos = warp_base + local_offset;
        if (pos < queue_capacity)
        {
            wq_ray[pos] = ray_idx;
            wq_sub[pos] = sub_idx;
        }
    }
}

// ============================================================================
// Kernel 1: AABB test + enqueue (Section 4.1)
// Launch: <<<ceil(batch_count / K1_BLOCK_SIZE), K1_BLOCK_SIZE, 0, stream>>>
// Each thread = one ray. Loops over all AABBs.
// ============================================================================

__global__ void aabb_test_and_enqueue(
    // Ray data (SoA, length >= batch_count)
    const float *__restrict__ T1x, const float *__restrict__ T1y, const float *__restrict__ T1z,
    const float *__restrict__ T2x, const float *__restrict__ T2y, const float *__restrict__ T2z,
    const float *__restrict__ T3x, const float *__restrict__ T3y, const float *__restrict__ T3z,
    const float *__restrict__ Nx, const float *__restrict__ Ny, const float *__restrict__ Nz,
    const float *__restrict__ D1x, const float *__restrict__ D1y, const float *__restrict__ D1z,
    const float *__restrict__ D2x, const float *__restrict__ D2y, const float *__restrict__ D2z,
    const float *__restrict__ D3x, const float *__restrict__ D3y, const float *__restrict__ D3z,
    const float *__restrict__ rD1, const float *__restrict__ rD2, const float *__restrict__ rD3,
    // AABB data (SoA, length n_sub)
    const float *__restrict__ Xmin, const float *__restrict__ Xmax,
    const float *__restrict__ Ymin, const float *__restrict__ Ymax,
    const float *__restrict__ Zmin, const float *__restrict__ Zmax,
    // Dimensions
    uint32_t batch_count,
    uint32_t n_sub,
    // Work queue output
    uint32_t *wq_ray_idx,
    uint16_t *wq_sub_idx,
    uint32_t *queue_tail,
    uint32_t queue_capacity)
{
    uint32_t ray_idx = blockIdx.x * K1_BLOCK_SIZE + threadIdx.x;
    bool valid = (ray_idx < batch_count);

    // Load ray attributes (coalesced across warp)
    float t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
    float nx, ny, nz;
    float d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z;
    float rd1, rd2, rd3;

    // Precomputed origin × normal products
    float ox0_x_nx, oy0_x_ny, oz0_x_nz;
    float ox1_x_nx, oy1_x_ny, oz1_x_nz;
    float ox2_x_nx, oy2_x_ny, oz2_x_nz;

    if (valid)
    {
        t1x = T1x[ray_idx];
        t1y = T1y[ray_idx];
        t1z = T1z[ray_idx];
        t2x = T2x[ray_idx];
        t2y = T2y[ray_idx];
        t2z = T2z[ray_idx];
        t3x = T3x[ray_idx];
        t3y = T3y[ray_idx];
        t3z = T3z[ray_idx];
        nx = Nx[ray_idx];
        ny = Ny[ray_idx];
        nz = Nz[ray_idx];
        d1x = D1x[ray_idx];
        d1y = D1y[ray_idx];
        d1z = D1z[ray_idx];
        d2x = D2x[ray_idx];
        d2y = D2y[ray_idx];
        d2z = D2z[ray_idx];
        d3x = D3x[ray_idx];
        d3y = D3y[ray_idx];
        d3z = D3z[ray_idx];
        rd1 = rD1[ray_idx];
        rd2 = rD2[ray_idx];
        rd3 = rD3[ray_idx];

        ox0_x_nx = t1x * nx;
        oy0_x_ny = t1y * ny;
        oz0_x_nz = t1z * nz;
        ox1_x_nx = t2x * nx;
        oy1_x_ny = t2y * ny;
        oz1_x_nz = t2z * nz;
        ox2_x_nx = t3x * nx;
        oy2_x_ny = t3y * ny;
        oz2_x_nz = t3z * nz;
    }

    // Loop over all sub-clouds (AABBs)
    for (uint32_t i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        bool has_hit = false;

        if (valid)
        {
            // Load AABB bounds with slack (Section 4.1)
            float b0_low = Xmin[i_sub] - 1e-5f;
            float b0_high = Xmax[i_sub] + 1e-5f;
            float b1_low = Ymin[i_sub] - 1e-5f;
            float b1_high = Ymax[i_sub] + 1e-5f;
            float b2_low = Zmin[i_sub] - 1e-5f;
            float b2_high = Zmax[i_sub] + 1e-5f;

            // Enumerate 8 AABB corner points
            float rx[8] = {b0_low, b0_low, b0_low, b0_low, b0_high, b0_high, b0_high, b0_high};
            float ry[8] = {b1_low, b1_low, b1_high, b1_high, b1_low, b1_low, b1_high, b1_high};
            float rz[8] = {b2_low, b2_high, b2_low, b2_high, b2_low, b2_high, b2_low, b2_high};

            // Initialize advanced wavefront bounding box
            float a0_low = INFINITY, a1_low = INFINITY, a2_low = INFINITY;
            float a0_high = -INFINITY, a1_high = -INFINITY, a2_high = -INFINITY;

            // Advance all 3 ray vertices to each of 8 AABB corners, track min/max
            for (int c = 0; c < 8; ++c)
            {
                float v, d;

                // Vertex 0: distance to wavefront at corner
                v = fmaf(rz[c], nz, -oz0_x_nz);
                d = rd1 * v;
                v = fmaf(ry[c], ny, -oy0_x_ny);
                d = fmaf(rd1, v, d);
                v = fmaf(rx[c], nx, -ox0_x_nx);
                d = fmaf(rd1, v, d);

                v = fmaf(d, d1x, t1x);
                a0_low = fminf(v, a0_low);
                a0_high = fmaxf(v, a0_high);
                v = fmaf(d, d1y, t1y);
                a1_low = fminf(v, a1_low);
                a1_high = fmaxf(v, a1_high);
                v = fmaf(d, d1z, t1z);
                a2_low = fminf(v, a2_low);
                a2_high = fmaxf(v, a2_high);

                // Vertex 1
                v = fmaf(rz[c], nz, -oz1_x_nz);
                d = rd2 * v;
                v = fmaf(ry[c], ny, -oy1_x_ny);
                d = fmaf(rd2, v, d);
                v = fmaf(rx[c], nx, -ox1_x_nx);
                d = fmaf(rd2, v, d);

                v = fmaf(d, d2x, t2x);
                a0_low = fminf(v, a0_low);
                a0_high = fmaxf(v, a0_high);
                v = fmaf(d, d2y, t2y);
                a1_low = fminf(v, a1_low);
                a1_high = fmaxf(v, a1_high);
                v = fmaf(d, d2z, t2z);
                a2_low = fminf(v, a2_low);
                a2_high = fmaxf(v, a2_high);

                // Vertex 2
                v = fmaf(rz[c], nz, -oz2_x_nz);
                d = rd3 * v;
                v = fmaf(ry[c], ny, -oy2_x_ny);
                d = fmaf(rd3, v, d);
                v = fmaf(rx[c], nx, -ox2_x_nx);
                d = fmaf(rd3, v, d);

                v = fmaf(d, d3x, t3x);
                a0_low = fminf(v, a0_low);
                a0_high = fmaxf(v, a0_high);
                v = fmaf(d, d3y, t3y);
                a1_low = fminf(v, a1_low);
                a1_high = fmaxf(v, a1_high);
                v = fmaf(d, d3z, t3z);
                a2_low = fminf(v, a2_low);
                a2_high = fmaxf(v, a2_high);
            }

            // 6-way overlap test
            has_hit = (a0_high >= b0_low) && (a0_low <= b0_high) && (a1_high >= b1_low) && (a1_low <= b1_high) && (a2_high >= b2_low) && (a2_low <= b2_high);
        }

        // Warp-level compaction — all 32 lanes participate (invalid threads have has_hit = false)
        enqueue_warp(has_hit, ray_idx, (uint16_t)i_sub,
                     wq_ray_idx, wq_sub_idx, queue_tail, queue_capacity);
    }
}

// ============================================================================
// Kernel 1b: Compute chunks_per_segment + write sentinels (Section 6.3)
// Launch: <<<ceil((num_segments+1) / K1B_BLOCK_SIZE), K1B_BLOCK_SIZE, 0, stream>>>
// ============================================================================

__global__ void compute_chunks_per_segment(
    const uint32_t *__restrict__ run_len,
    uint32_t *chunks_per_seg,
    uint32_t num_segments,
    uint32_t rays_per_block)
{
    uint32_t i = blockIdx.x * K1B_BLOCK_SIZE + threadIdx.x;

    if (i < num_segments)
        chunks_per_seg[i] = (run_len[i] + rays_per_block - 1u) / rays_per_block;

    // Write sentinel values at position [num_segments] for the ExclusiveSum calls
    if (i == num_segments)
    {
        // Modifying run_len requires a non-const cast. The plan specifies this
        // sentinel write (Section 6.3, Step 2). The array is only consumed by
        // ExclusiveSum after this kernel completes (same-stream ordering).
        ((uint32_t *)run_len)[num_segments] = 0u;
        chunks_per_seg[num_segments] = 0u;
    }
}

// ============================================================================
// Kernel 2: Point intersection (Section 4.2)
// Launch: <<<total_chunks, K2_BLOCK_SIZE, smem_bytes, stream>>>
// Each block = one chunk of up to RAYS_PER_BLOCK rays within one segment.
// 256 threads cooperatively process all points in the sub-cloud for each ray.
// ============================================================================

__global__ void __launch_bounds__(K2_BLOCK_SIZE, 4)
    point_intersect(
        // Point cloud (SoA)
        const float *__restrict__ Px,
        const float *__restrict__ Py,
        const float *__restrict__ Pz,
        // Sub-cloud index table (length n_sub + 1, with sentinel)
        const uint32_t *__restrict__ SCI,
        // Sorted work queue (CUB DoubleBuffer current)
        const uint32_t *__restrict__ sorted_ray_idx,
        // Segment table
        const uint16_t *__restrict__ seg_sub_idx,
        const uint32_t *__restrict__ seg_offset,
        const uint32_t *__restrict__ chunk_offset,
        uint32_t num_segments,
        // Ray pointer table (24 float pointers, per-stream device array)
        const float *const *__restrict__ d_ray_ptrs,
        // Hit output
        uint32_t *hit_ray_idx,
        uint32_t *hit_point_idx,
        uint32_t *hit_count,
        uint32_t hit_capacity,
        uint32_t *overflow_flag,
        // Runtime tuning parameters
        uint32_t tile_size,
        uint32_t max_local_hits,
        uint32_t total_chunks)
{
    uint32_t b = blockIdx.x;
    if (b >= total_chunks)
        return;

    // --- Binary search: map blockIdx → (segment, chunk_within_segment) ---
    // Find largest seg_idx such that chunk_offset[seg_idx] <= b
    // Uses upper_bound - 1 pattern on chunk_offset[0..num_segments]
    uint32_t lo = 0, hi = num_segments;
    while (lo < hi)
    {
        uint32_t mid = (lo + hi + 1u) >> 1;
        if (chunk_offset[mid] <= b)
            lo = mid;
        else
            hi = mid - 1u;
    }
    uint32_t seg_idx = lo;
    uint32_t chunk_in_seg = b - chunk_offset[seg_idx];

    uint16_t sub_idx = seg_sub_idx[seg_idx];

    // Ray range for this chunk within the sorted queue
    uint32_t ray_start = seg_offset[seg_idx] + chunk_in_seg * RAYS_PER_BLOCK;
    uint32_t ray_end = seg_offset[seg_idx + 1]; // sentinel at num_segments
    ray_end = min(ray_start + RAYS_PER_BLOCK, ray_end);

    // Point range for this sub-cloud (no branch — SCI has sentinel at n_sub)
    uint32_t i_start = SCI[sub_idx];
    uint32_t i_end = SCI[sub_idx + 1u];

    // --- Dynamic shared memory layout ---
    extern __shared__ uint32_t s_dyn[];
    uint32_t *s_hit_rays = s_dyn;                    // [0 .. max_local_hits)
    uint32_t *s_hit_points = s_dyn + max_local_hits; // [max_local_hits .. 2*max_local_hits)

    // --- Static shared memory ---
    __shared__ float s_ray[N_RAY_ATTRS]; // current ray's 24 floats
    __shared__ uint32_t s_ray_idx;       // current ray's global index
    __shared__ uint32_t s_hit_count;     // local hit counter
    __shared__ uint32_t s_flush_base;    // global base for current flush

    if (threadIdx.x == 0)
        s_hit_count = 0;
    __syncthreads();

    // --- Outer loop: iterate over rays in this chunk ---
    for (uint32_t r = ray_start; r < ray_end; ++r)
    {
        // Load ray into shared memory via per-stream device pointer table
        if (threadIdx.x < N_RAY_ATTRS)
            s_ray[threadIdx.x] = d_ray_ptrs[threadIdx.x][sorted_ray_idx[r]];
        if (threadIdx.x == N_RAY_ATTRS)
            s_ray_idx = sorted_ray_idx[r];
        __syncthreads();

        // Unpack from shared memory into registers
        float t1x = s_ray[0], t1y = s_ray[1], t1z = s_ray[2];
        float t2x = s_ray[3], t2y = s_ray[4], t2z = s_ray[5];
        float t3x = s_ray[6], t3y = s_ray[7], t3z = s_ray[8];
        float nx = s_ray[9], ny = s_ray[10], nz = s_ray[11];
        float d1x = s_ray[12], d1y = s_ray[13], d1z = s_ray[14];
        float d2x = s_ray[15], d2y = s_ray[16], d2z = s_ray[17];
        float d3x = s_ray[18], d3y = s_ray[19], d3z = s_ray[20];
        float rd1 = s_ray[21], rd2 = s_ray[22], rd3 = s_ray[23];
        uint32_t ray_global = s_ray_idx;

        // Precompute origin × normal products
        float ox0_x_nx = t1x * nx, oy0_x_ny = t1y * ny, oz0_x_nz = t1z * nz;
        float ox1_x_nx = t2x * nx, oy1_x_ny = t2y * ny, oz1_x_nz = t2z * nz;
        float ox2_x_nx = t3x * nx, oy2_x_ny = t3y * ny, oz2_x_nz = t3z * nz;

        // --- Tiled point loop: K2_BLOCK_SIZE threads cooperatively ---
        uint32_t tile_points = (uint32_t)K2_BLOCK_SIZE * tile_size;
        for (uint32_t tile_base = i_start; tile_base < i_end; tile_base += tile_points)
        {
            uint32_t tile_end = min(tile_base + tile_points, i_end);

            for (uint32_t ip = tile_base + threadIdx.x; ip < tile_end; ip += K2_BLOCK_SIZE)
            {
                // Load point (coalesced across warp)
                float px = Px[ip], py = Py[ip], pz = Pz[ip];

                // Distance from vertex 0 origin to wavefront at point
                float v = fmaf(pz, nz, -oz0_x_nz);
                float d0 = rd1 * v;
                v = fmaf(py, ny, -oy0_x_ny);
                d0 = fmaf(rd1, v, d0);
                v = fmaf(px, nx, -ox0_x_nx);
                d0 = fmaf(rd1, v, d0);

                // Advance vertex 0 → V
                float Vx = fmaf(d0, d1x, t1x);
                float Vy = fmaf(d0, d1y, t1y);
                float Vz = fmaf(d0, d1z, t1z);

                // Compute edge e1 = advanced_V1 - V
                v = fmaf(pz, nz, -oz1_x_nz);
                float d1 = rd2 * v;
                v = fmaf(py, ny, -oy1_x_ny);
                d1 = fmaf(rd2, v, d1);
                v = fmaf(px, nx, -ox1_x_nx);
                d1 = fmaf(rd2, v, d1);

                float e1x = fmaf(d1, d2x, t2x) - Vx;
                float e1y = fmaf(d1, d2y, t2y) - Vy;
                float e1z = fmaf(d1, d2z, t2z) - Vz;

                // Compute edge e2 = advanced_V2 - V
                v = fmaf(pz, nz, -oz2_x_nz);
                float d2 = rd3 * v;
                v = fmaf(py, ny, -oy2_x_ny);
                d2 = fmaf(rd3, v, d2);
                v = fmaf(px, nx, -ox2_x_nx);
                d2 = fmaf(rd3, v, d2);

                float e2x = fmaf(d2, d3x, t3x) - Vx;
                float e2y = fmaf(d2, d3y, t3y) - Vy;
                float e2z = fmaf(d2, d3z, t3z) - Vz;

                // Vector from V to point
                float tx = px - Vx, ty = py - Vy, tz = pz - Vz;

                // Cross products → barycentric coordinates
                // 1st barycentric: PQ = N × e2, DT = dot(e1, PQ), U = dot(t, PQ)
                float PQ, DT, U;
                PQ = e2z * ny - e2y * nz;
                DT = e1x * PQ;
                U = tx * PQ;
                PQ = e2x * nz - e2z * nx;
                DT += e1y * PQ;
                U += ty * PQ;
                PQ = e2y * nx - e2x * ny;
                DT += e1z * PQ;
                U += tz * PQ;

                // 2nd barycentric: PQ = e1 × t, V_bary = dot(N, PQ)
                float V_bary;
                PQ = e1z * ty - e1y * tz;
                V_bary = nx * PQ;
                PQ = e1x * tz - e1z * tx;
                V_bary += ny * PQ;
                PQ = e1y * tx - e1x * ty;
                V_bary += nz * PQ;

                // Normalize (fast approximate division — Section 7.1)
                DT = __fdividef(1.0f, DT);
                U *= DT;
                V_bary *= DT;

                // Hit condition: all 3 vertex distances checked
                bool hit = (U >= 0.0f) && (V_bary >= 0.0f) && (U + V_bary <= 1.0f) && (d0 >= 0.0f) && (d1 >= 0.0f) && (d2 >= 0.0f);

                if (hit)
                {
                    uint32_t slot = atomicAdd(&s_hit_count, 1u);
                    if (slot < max_local_hits)
                    {
                        s_hit_rays[slot] = ray_global;
                        s_hit_points[slot] = ip;
                    }
                }
            }

            // --- Tile boundary flush ---
            __syncthreads();

            if (s_hit_count > 0u)
            {
                uint32_t local_count = s_hit_count;
                uint32_t flush_count = min(local_count, max_local_hits);

                if (threadIdx.x == 0)
                {
                    // Advance global counter by TRUE hit count (not clamped)
                    s_flush_base = atomicAdd(hit_count, local_count);
                    if (local_count > max_local_hits)
                        atomicOr(overflow_flag, 1u);
                }
                __syncthreads();

                for (uint32_t i = threadIdx.x; i < flush_count; i += K2_BLOCK_SIZE)
                {
                    uint32_t gpos = s_flush_base + i;
                    if (gpos < hit_capacity)
                    {
                        hit_ray_idx[gpos] = s_hit_rays[i];
                        hit_point_idx[gpos] = s_hit_points[i];
                    }
                }

                if (threadIdx.x == 0)
                    s_hit_count = 0;
                __syncthreads();
            }
        }

        // Ensure all threads done with tiled point loop before next ray load
        __syncthreads();
    }
}

// ============================================================================
// Host helper: stage one batch of ray data from user arrays into pinned memory
// The 24 SoA arrays are packed contiguously:
//   h_pinned[0 * batch_size .. 1 * batch_size) = T1x
//   h_pinned[1 * batch_size .. 2 * batch_size) = T1y
//   ...
// ============================================================================

static void stage_ray_batch(
    float *h_pinned,
    size_t batch_size,   // allocation size of pinned buffer (max batch)
    size_t batch_offset, // start ray index within user arrays
    size_t count,        // number of rays in this batch
    const float *T1x, const float *T1y, const float *T1z,
    const float *T2x, const float *T2y, const float *T2z,
    const float *T3x, const float *T3y, const float *T3z,
    const float *Nx, const float *Ny, const float *Nz,
    const float *D1x, const float *D1y, const float *D1z,
    const float *D2x, const float *D2y, const float *D2z,
    const float *D3x, const float *D3y, const float *D3z,
    const float *rD1, const float *rD2, const float *rD3)
{
    const float *src[N_RAY_ATTRS] = {
        T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
        Nx, Ny, Nz, D1x, D1y, D1z, D2x, D2y, D2z,
        D3x, D3y, D3z, rD1, rD2, rD3};
    for (int a = 0; a < N_RAY_ATTRS; ++a)
        memcpy(h_pinned + (size_t)a * batch_size, src[a] + batch_offset, count * sizeof(float));
}

// ============================================================================
// Host wrapper: qd_RPI_CUDA (Section 8 signature)
// ============================================================================

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
    const float *Nx, const float *Ny, const float *Nz,
    const float *D1x, const float *D1y, const float *D1z,
    const float *D2x, const float *D2y, const float *D2z,
    const float *D3x, const float *D3y, const float *D3z,
    const float *rD1, const float *rD2, const float *rD3,
    const size_t n_ray,
    std::vector<unsigned> *p_hit,
    int gpu_id)
{
    // --- Early returns (Section 8) ---
    if (n_ray == 0)
        return;

    // Clear all output vectors
    for (size_t i = 0; i < n_ray; ++i)
        p_hit[i].clear();

    if (n_sub == 0 || n_point == 0)
        return;

    // --- Range checks ---
    if (n_sub > 65535)
        throw std::invalid_argument("qd_RPI_CUDA: n_sub > 65535 is not supported (uint16_t sub_idx).");

    // --- GPU selection ---
    CUDA_CHECK(cudaSetDevice(gpu_id));

    // RAII cleanup struct — destructor frees everything on any exit path
    CudaBuffers buf;

    // ====================================================================
    // Step 1: Upload scene data (resident for the entire run)
    // ====================================================================

    // Point cloud
    CUDA_CHECK(cudaMalloc(&buf.d_Px, n_point * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_Py, n_point * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_Pz, n_point * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(buf.d_Px, Px, n_point * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_Py, Py, n_point * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_Pz, Pz, n_point * sizeof(float), cudaMemcpyHostToDevice));

    // SCI with sentinel: length n_sub + 1, SCI[n_sub] = n_point
    {
        std::vector<uint32_t> sci_host(n_sub + 1);
        for (size_t i = 0; i < n_sub; ++i)
            sci_host[i] = (uint32_t)SCI[i];
        sci_host[n_sub] = (uint32_t)n_point;

        CUDA_CHECK(cudaMalloc(&buf.d_SCI, (n_sub + 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(buf.d_SCI, sci_host.data(), (n_sub + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // AABB bounds
    CUDA_CHECK(cudaMalloc(&buf.d_Xmin, n_sub * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_Xmax, n_sub * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_Ymin, n_sub * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_Ymax, n_sub * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_Zmin, n_sub * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_Zmax, n_sub * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(buf.d_Xmin, Xmin, n_sub * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_Xmax, Xmax, n_sub * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_Ymin, Ymin, n_sub * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_Ymax, Ymax, n_sub * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_Zmin, Zmin, n_sub * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_Zmax, Zmax, n_sub * sizeof(float), cudaMemcpyHostToDevice));

    // ====================================================================
    // Step 2: Compute batch size (Section 5.1)
    // ====================================================================

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    // Segment table per stream (sized by n_sub — small fixed cost)
    size_t seg_table_per_stream = n_sub * (2 + 4 + 4 + 4) // sub_idx + run_len + chunks_per_seg + seg_offset
                                  + (n_sub + 1) * (4 + 4) // seg_offset + chunk_offset (with sentinel)
                                  + 4 + 4                 // num_segments + total_chunks
                                  + n_sub * 4;            // extra chunks_per_seg

    // Per-ray memory (excludes CUB temp, which depends on batch size)
    size_t per_ray_bytes = N_RAY_ATTRS * sizeof(float)       // 96: ray input
                           + (size_t)EST_AVG_HITS * (4 + 2)  // 60: work queue A
                           + (size_t)EST_AVG_HITS * (4 + 2)  // 60: work queue B
                           + (size_t)EST_HIT_RATE * (4 + 4); // 16: hit output

    // Use only 45% of free memory
    size_t usable = (size_t)(free_mem * 0.45);

    // CUB ExclusiveSum temp only depends on n_sub (fixed)
    size_t cub_scan_temp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_scan_temp,
                                  (uint32_t *)nullptr, (uint32_t *)nullptr,
                                  (int)(n_sub + 1), (cudaStream_t)0);

    // Iterative batch sizing: CUB sort/RLE temp depends on queue_capacity = batch_size * EST_AVG_HITS,
    // but batch_size depends on available memory after CUB temp. Iterate to converge.
    size_t batch_size = 0;
    size_t cub_temp_per_stream = 0;
    {
        // Initial estimate: ignore CUB sort/RLE temp
        size_t fixed_no_cub = 2 * seg_table_per_stream;
        if (usable <= fixed_no_cub)
            throw std::runtime_error("qd_RPI_CUDA: insufficient GPU memory.");

        batch_size = (usable - fixed_no_cub) / (2 * per_ray_bytes);
        if (batch_size > n_ray)
            batch_size = n_ray;

        // Iterate: query CUB temp for the actual batch queue capacity, then recompute batch_size
        for (int iter = 0; iter < 4; ++iter)
        {
            size_t queue_est = batch_size * (size_t)EST_AVG_HITS;
            if (queue_est < 1024)
                queue_est = 1024;

            size_t cub_sort_temp = 0, cub_rle_temp = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, cub_sort_temp,
                                            (uint16_t *)nullptr, (uint16_t *)nullptr,
                                            (uint32_t *)nullptr, (uint32_t *)nullptr,
                                            (int)queue_est, 0, 16, (cudaStream_t)0);
            cub::DeviceRunLengthEncode::Encode(nullptr, cub_rle_temp,
                                               (uint16_t *)nullptr, (uint16_t *)nullptr, (uint32_t *)nullptr,
                                               (uint32_t *)nullptr, (int)queue_est, (cudaStream_t)0);

            cub_temp_per_stream = (size_t)(std::max({cub_sort_temp, cub_rle_temp, cub_scan_temp}) * 1.2);
            if (cub_temp_per_stream < 1024)
                cub_temp_per_stream = 1024;

            size_t fixed_cost = 2 * cub_temp_per_stream + 2 * seg_table_per_stream;
            if (usable <= fixed_cost)
            {
                // Shrink batch_size drastically and retry
                batch_size = batch_size / 4;
                if (batch_size == 0)
                    throw std::runtime_error("qd_RPI_CUDA: insufficient GPU memory.");
                continue;
            }

            size_t new_batch_size = (usable - fixed_cost) / (2 * per_ray_bytes);
            if (new_batch_size > n_ray)
                new_batch_size = n_ray;

            // Converged if batch_size didn't change (or grew — CUB temp shrunk)
            if (new_batch_size >= batch_size)
                break;
            batch_size = new_batch_size;
        }
    }

    batch_size = (batch_size / K1_BLOCK_SIZE) * K1_BLOCK_SIZE; // align to block size
    if (batch_size == 0)
        batch_size = K1_BLOCK_SIZE; // minimum one block
    if (batch_size > n_ray)
        batch_size = n_ray;

    buf.batch_size = batch_size;
    uint32_t queue_capacity = (uint32_t)(batch_size * EST_AVG_HITS);
    uint32_t hit_capacity = (uint32_t)(batch_size * EST_HIT_RATE);
    buf.queue_capacity = queue_capacity;
    buf.hit_capacity = hit_capacity;

    // ====================================================================
    // Step 3: Allocate per-stream buffers + streams + pinned memory
    // ====================================================================

    CUDA_CHECK(cudaStreamCreate(&buf.stream[0]));
    CUDA_CHECK(cudaStreamCreate(&buf.stream[1]));

    for (int s = 0; s < 2; ++s)
    {
        // Ray input (24 SoA arrays)
        for (int a = 0; a < N_RAY_ATTRS; ++a)
            CUDA_CHECK(cudaMalloc(&buf.d_ray[s][a], batch_size * sizeof(float)));

        // Ray pointer table (24 float pointers on device)
        CUDA_CHECK(cudaMalloc(&buf.d_ray_ptrs[s], N_RAY_ATTRS * sizeof(float *)));

        // Work queue A/B
        CUDA_CHECK(cudaMalloc(&buf.d_wq_ray_idx_A[s], queue_capacity * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_wq_ray_idx_B[s], queue_capacity * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_wq_sub_idx_A[s], queue_capacity * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_wq_sub_idx_B[s], queue_capacity * sizeof(uint16_t)));

        // Atomic counters (own cudaMalloc for cache-line isolation)
        CUDA_CHECK(cudaMalloc(&buf.d_queue_tail[s], sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_hit_count[s], sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_overflow_flag[s], sizeof(uint32_t)));

        // CUB temp buffer
        buf.cub_temp_bytes[s] = cub_temp_per_stream;
        CUDA_CHECK(cudaMalloc(&buf.d_cub_temp[s], cub_temp_per_stream));

        // Segment table (sized by n_sub)
        // +2 extra elements for sentinel values in run_len and chunks_per_seg
        CUDA_CHECK(cudaMalloc(&buf.d_seg_sub_idx[s], n_sub * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_seg_run_len[s], (n_sub + 2) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_seg_num_segments[s], sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_seg_offset[s], (n_sub + 2) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_chunks_per_seg[s], (n_sub + 2) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_seg_chunk_offset[s], (n_sub + 2) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_total_chunks[s], sizeof(uint32_t)));

        // Hit output
        CUDA_CHECK(cudaMalloc(&buf.d_hit_ray_idx[s], hit_capacity * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&buf.d_hit_point_idx[s], hit_capacity * sizeof(uint32_t)));

        // Pinned host memory
        CUDA_CHECK(cudaMallocHost(&buf.h_ray_pinned[s], (size_t)N_RAY_ATTRS * batch_size * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&buf.h_hit_ray_pinned[s], hit_capacity * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&buf.h_hit_point_pinned[s], hit_capacity * sizeof(uint32_t)));
    }

    // ====================================================================
    // Step 4: Batch loop with double-buffered pipeline (Section 5.2)
    // ====================================================================

    size_t n_batches = (n_ray + batch_size - 1) / batch_size;

    // Host-side per-stream state
    uint32_t h_queue_tail[2] = {0, 0};
    uint32_t h_hit_count[2] = {0, 0};
    uint32_t h_overflow_flag[2] = {0, 0};
    uint32_t h_num_segments[2] = {0, 0};
    uint32_t h_total_chunks[2] = {0, 0};
    size_t batch_offsets[2] = {0, 0};

    // Host-side ray pointer tables (24 device pointers per stream)
    const float *h_ray_ptrs[2][N_RAY_ATTRS];

    // Runtime tuning parameters for Kernel 2 (may change on overflow retry)
    uint32_t tile_size = 2;
    uint32_t max_local_hits = tile_size * K2_BLOCK_SIZE; // 512

    // End_bit for CUB radix sort
    int num_sub_bits = std::max(1, (int)std::ceil(std::log2((double)n_sub)));

    // --- Stage batch 0 into pinned memory BEFORE the loop ---
    {
        size_t count0 = std::min(batch_size, n_ray);
        stage_ray_batch(buf.h_ray_pinned[0], batch_size, 0, count0,
                        T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
                        Nx, Ny, Nz, D1x, D1y, D1z, D2x, D2y, D2z,
                        D3x, D3y, D3z, rD1, rD2, rD3);
    }

    for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx)
    {
        int s = (int)(batch_idx % 2); // current stream index
        int sp = 1 - s;               // previous stream index

        size_t batch_offset = batch_idx * batch_size;
        size_t batch_count = std::min(batch_size, n_ray - batch_offset);
        batch_offsets[s] = batch_offset;

        // --- Wait for previous batch's D→H to finish (if any) ---
        if (batch_idx > 0)
        {
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[sp]));

            // Scatter hits from previous batch into p_hit[] vectors on host
            for (uint32_t i = 0; i < h_hit_count[sp]; ++i)
            {
                size_t actual_ray = batch_offsets[sp] + buf.h_hit_ray_pinned[sp][i];
                p_hit[actual_ray].push_back((unsigned)buf.h_hit_point_pinned[sp][i]);
            }
        }

        // --- H→D: upload ray data for this batch ---
        // Copy each SoA array from the packed pinned buffer to its device array
        for (int a = 0; a < N_RAY_ATTRS; ++a)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                buf.d_ray[s][a],
                buf.h_ray_pinned[s] + (size_t)a * batch_size,
                batch_count * sizeof(float),
                cudaMemcpyHostToDevice, buf.stream[s]));
        }

        // --- Update per-stream ray pointer table ---
        for (int a = 0; a < N_RAY_ATTRS; ++a)
            h_ray_ptrs[s][a] = buf.d_ray[s][a];
        CUDA_CHECK(cudaMemcpyAsync(
            (void *)buf.d_ray_ptrs[s], h_ray_ptrs[s],
            N_RAY_ATTRS * sizeof(float *),
            cudaMemcpyHostToDevice, buf.stream[s]));

        // --- Reset counters (Section 4.0) ---
        CUDA_CHECK(cudaMemsetAsync(buf.d_queue_tail[s], 0, sizeof(uint32_t), buf.stream[s]));
        CUDA_CHECK(cudaMemsetAsync(buf.d_hit_count[s], 0, sizeof(uint32_t), buf.stream[s]));
        CUDA_CHECK(cudaMemsetAsync(buf.d_overflow_flag[s], 0, sizeof(uint32_t), buf.stream[s]));

        // --- Kernel 1: AABB test + enqueue ---
        {
            uint32_t grid1 = ((uint32_t)batch_count + K1_BLOCK_SIZE - 1) / K1_BLOCK_SIZE;
            aabb_test_and_enqueue<<<grid1, K1_BLOCK_SIZE, 0, buf.stream[s]>>>(
                buf.d_ray[s][0], buf.d_ray[s][1], buf.d_ray[s][2],    // T1xyz
                buf.d_ray[s][3], buf.d_ray[s][4], buf.d_ray[s][5],    // T2xyz
                buf.d_ray[s][6], buf.d_ray[s][7], buf.d_ray[s][8],    // T3xyz
                buf.d_ray[s][9], buf.d_ray[s][10], buf.d_ray[s][11],  // Nxyz
                buf.d_ray[s][12], buf.d_ray[s][13], buf.d_ray[s][14], // D1xyz
                buf.d_ray[s][15], buf.d_ray[s][16], buf.d_ray[s][17], // D2xyz
                buf.d_ray[s][18], buf.d_ray[s][19], buf.d_ray[s][20], // D3xyz
                buf.d_ray[s][21], buf.d_ray[s][22], buf.d_ray[s][23], // rD1/rD2/rD3
                buf.d_Xmin, buf.d_Xmax,
                buf.d_Ymin, buf.d_Ymax,
                buf.d_Zmin, buf.d_Zmax,
                (uint32_t)batch_count,
                (uint32_t)n_sub,
                buf.d_wq_ray_idx_A[s],
                buf.d_wq_sub_idx_A[s],
                buf.d_queue_tail[s],
                queue_capacity);
        }

        // --- Read queue_tail back to host (small sync, 4 bytes) ---
        CUDA_CHECK(cudaMemcpyAsync(&h_queue_tail[s], buf.d_queue_tail[s],
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, buf.stream[s]));
        CUDA_CHECK(cudaStreamSynchronize(buf.stream[s]));

        // --- Check for queue overflow ---
        if (h_queue_tail[s] > queue_capacity)
        {
            // Sync both streams, reallocate both, re-run from counter reset
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[1]));

            uint32_t new_capacity = h_queue_tail[s] * 2;
            for (int ss = 0; ss < 2; ++ss)
            {
                cudaFree(buf.d_wq_ray_idx_A[ss]);
                buf.d_wq_ray_idx_A[ss] = nullptr;
                cudaFree(buf.d_wq_ray_idx_B[ss]);
                buf.d_wq_ray_idx_B[ss] = nullptr;
                cudaFree(buf.d_wq_sub_idx_A[ss]);
                buf.d_wq_sub_idx_A[ss] = nullptr;
                cudaFree(buf.d_wq_sub_idx_B[ss]);
                buf.d_wq_sub_idx_B[ss] = nullptr;
                CUDA_CHECK(cudaMalloc(&buf.d_wq_ray_idx_A[ss], new_capacity * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&buf.d_wq_ray_idx_B[ss], new_capacity * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&buf.d_wq_sub_idx_A[ss], new_capacity * sizeof(uint16_t)));
                CUDA_CHECK(cudaMalloc(&buf.d_wq_sub_idx_B[ss], new_capacity * sizeof(uint16_t)));
            }
            queue_capacity = new_capacity;
            buf.queue_capacity = new_capacity;

            // Also resize CUB temp if needed
            {
                size_t new_sort_temp = 0;
                cub::DeviceRadixSort::SortPairs(nullptr, new_sort_temp,
                                                (uint16_t *)nullptr, (uint16_t *)nullptr,
                                                (uint32_t *)nullptr, (uint32_t *)nullptr,
                                                (int)new_capacity, 0, 16, (cudaStream_t)0);
                size_t new_rle_temp = 0;
                cub::DeviceRunLengthEncode::Encode(nullptr, new_rle_temp,
                                                   (uint16_t *)nullptr, (uint16_t *)nullptr, (uint32_t *)nullptr,
                                                   (uint32_t *)nullptr, (int)new_capacity, (cudaStream_t)0);
                size_t needed = (size_t)(std::max(new_sort_temp, new_rle_temp) * 1.2);
                for (int ss = 0; ss < 2; ++ss)
                {
                    if (needed > buf.cub_temp_bytes[ss])
                    {
                        cudaFree(buf.d_cub_temp[ss]);
                        buf.d_cub_temp[ss] = nullptr;
                        buf.cub_temp_bytes[ss] = needed;
                        CUDA_CHECK(cudaMalloc(&buf.d_cub_temp[ss], needed));
                    }
                }
            }

            // Re-run: reset counters and re-launch Kernel 1
            CUDA_CHECK(cudaMemsetAsync(buf.d_queue_tail[s], 0, sizeof(uint32_t), buf.stream[s]));
            CUDA_CHECK(cudaMemsetAsync(buf.d_hit_count[s], 0, sizeof(uint32_t), buf.stream[s]));
            CUDA_CHECK(cudaMemsetAsync(buf.d_overflow_flag[s], 0, sizeof(uint32_t), buf.stream[s]));

            uint32_t grid1 = ((uint32_t)batch_count + K1_BLOCK_SIZE - 1) / K1_BLOCK_SIZE;
            aabb_test_and_enqueue<<<grid1, K1_BLOCK_SIZE, 0, buf.stream[s]>>>(
                buf.d_ray[s][0], buf.d_ray[s][1], buf.d_ray[s][2],
                buf.d_ray[s][3], buf.d_ray[s][4], buf.d_ray[s][5],
                buf.d_ray[s][6], buf.d_ray[s][7], buf.d_ray[s][8],
                buf.d_ray[s][9], buf.d_ray[s][10], buf.d_ray[s][11],
                buf.d_ray[s][12], buf.d_ray[s][13], buf.d_ray[s][14],
                buf.d_ray[s][15], buf.d_ray[s][16], buf.d_ray[s][17],
                buf.d_ray[s][18], buf.d_ray[s][19], buf.d_ray[s][20],
                buf.d_ray[s][21], buf.d_ray[s][22], buf.d_ray[s][23],
                buf.d_Xmin, buf.d_Xmax, buf.d_Ymin, buf.d_Ymax, buf.d_Zmin, buf.d_Zmax,
                (uint32_t)batch_count, (uint32_t)n_sub,
                buf.d_wq_ray_idx_A[s], buf.d_wq_sub_idx_A[s],
                buf.d_queue_tail[s], queue_capacity);

            CUDA_CHECK(cudaMemcpyAsync(&h_queue_tail[s], buf.d_queue_tail[s],
                                       sizeof(uint32_t), cudaMemcpyDeviceToHost, buf.stream[s]));
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[s]));
        }

        // --- Pointers for sorted work queue (updated by CUB DoubleBuffer or fast path) ---
        const uint32_t *sorted_ray_idx = buf.d_wq_ray_idx_A[s]; // default: unsorted buffer A
        // sorted_sub_idx only needed as CUB input, not by K2 directly

        // Staging locals for the n_sub == 1 fast path.  Declared here (not inside
        // the if-block) so they remain alive while cudaMemcpyAsync is in flight.
        uint16_t h_seg_sub = 0;
        uint32_t h_seg_off[2] = {0, 0};
        uint32_t h_chunk_off[2] = {0, 0};
        uint32_t h_num_seg = 1;

        if (h_queue_tail[s] > 0)
        {
            if (n_sub == 1)
            {
                // ============================================================
                // Fast path for n_sub == 1 (Section 6.4)
                // Skip sort + segment table entirely.
                // ============================================================

                h_total_chunks[s] = (h_queue_tail[s] + RAYS_PER_BLOCK - 1) / RAYS_PER_BLOCK;

                h_seg_off[1] = h_queue_tail[s];
                h_chunk_off[1] = h_total_chunks[s];

                CUDA_CHECK(cudaMemcpyAsync(buf.d_seg_sub_idx[s], &h_seg_sub, sizeof(uint16_t), cudaMemcpyHostToDevice, buf.stream[s]));
                CUDA_CHECK(cudaMemcpyAsync(buf.d_seg_offset[s], h_seg_off, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, buf.stream[s]));
                CUDA_CHECK(cudaMemcpyAsync(buf.d_seg_chunk_offset[s], h_chunk_off, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, buf.stream[s]));
                CUDA_CHECK(cudaMemcpyAsync(buf.d_seg_num_segments[s], &h_num_seg, sizeof(uint32_t), cudaMemcpyHostToDevice, buf.stream[s]));

                // sorted_ray_idx = unsorted buffer A (no sort performed)
                sorted_ray_idx = buf.d_wq_ray_idx_A[s];
                h_num_segments[s] = 1;
            }
            else
            {
                // ============================================================
                // CUB sort + segment table (Sections 6.2–6.3)
                // ============================================================

                // --- CUB radix sort by sub_idx ---
                cub::DoubleBuffer<uint16_t> d_keys(buf.d_wq_sub_idx_A[s], buf.d_wq_sub_idx_B[s]);
                cub::DoubleBuffer<uint32_t> d_values(buf.d_wq_ray_idx_A[s], buf.d_wq_ray_idx_B[s]);

                size_t sort_temp = buf.cub_temp_bytes[s];
                CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                    buf.d_cub_temp[s], sort_temp,
                    d_keys, d_values,
                    (int)h_queue_tail[s],
                    0,            // begin_bit
                    num_sub_bits, // end_bit
                    buf.stream[s]));

                // After sort, current buffers:
                sorted_ray_idx = d_values.Current();
                const uint16_t *sorted_sub_idx = d_keys.Current();

                // --- Segment table: Step 1 — RunLengthEncode ---
                size_t rle_temp = buf.cub_temp_bytes[s];
                CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
                    buf.d_cub_temp[s], rle_temp,
                    sorted_sub_idx,
                    buf.d_seg_sub_idx[s],
                    buf.d_seg_run_len[s],
                    buf.d_seg_num_segments[s],
                    (int)h_queue_tail[s],
                    buf.stream[s]));

                // Read num_segments to host (4 bytes, small sync)
                CUDA_CHECK(cudaMemcpyAsync(&h_num_segments[s], buf.d_seg_num_segments[s],
                                           sizeof(uint32_t), cudaMemcpyDeviceToHost, buf.stream[s]));
                CUDA_CHECK(cudaStreamSynchronize(buf.stream[s]));

                // --- Segment table: Step 2 — Kernel 1b ---
                {
                    uint32_t grid1b = (h_num_segments[s] + 1 + K1B_BLOCK_SIZE - 1) / K1B_BLOCK_SIZE;
                    compute_chunks_per_segment<<<grid1b, K1B_BLOCK_SIZE, 0, buf.stream[s]>>>(
                        buf.d_seg_run_len[s],
                        buf.d_chunks_per_seg[s],
                        h_num_segments[s],
                        RAYS_PER_BLOCK);
                }

                // --- Segment table: Step 3 — Two exclusive prefix sums ---
                // seg_offset: cumulative ray offset
                size_t scan_temp1 = buf.cub_temp_bytes[s];
                CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                    buf.d_cub_temp[s], scan_temp1,
                    buf.d_seg_run_len[s], buf.d_seg_offset[s],
                    (int)(h_num_segments[s] + 1),
                    buf.stream[s]));

                // chunk_offset: cumulative block offset for Kernel 2 grid mapping
                size_t scan_temp2 = buf.cub_temp_bytes[s];
                CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                    buf.d_cub_temp[s], scan_temp2,
                    buf.d_chunks_per_seg[s], buf.d_seg_chunk_offset[s],
                    (int)(h_num_segments[s] + 1),
                    buf.stream[s]));

                // Read total_chunks from chunk_offset[num_segments] (4 bytes, small sync)
                CUDA_CHECK(cudaMemcpyAsync(&h_total_chunks[s],
                                           buf.d_seg_chunk_offset[s] + h_num_segments[s],
                                           sizeof(uint32_t), cudaMemcpyDeviceToHost, buf.stream[s]));
                CUDA_CHECK(cudaStreamSynchronize(buf.stream[s]));
            }

            // --- Kernel 2: point intersection ---
            if (h_total_chunks[s] > 0)
            {
                uint32_t smem_bytes = 2u * max_local_hits * sizeof(uint32_t);

                point_intersect<<<h_total_chunks[s], K2_BLOCK_SIZE, smem_bytes, buf.stream[s]>>>(
                    buf.d_Px, buf.d_Py, buf.d_Pz,
                    buf.d_SCI,
                    sorted_ray_idx,
                    buf.d_seg_sub_idx[s],
                    buf.d_seg_offset[s],
                    buf.d_seg_chunk_offset[s],
                    h_num_segments[s],
                    buf.d_ray_ptrs[s],
                    buf.d_hit_ray_idx[s],
                    buf.d_hit_point_idx[s],
                    buf.d_hit_count[s],
                    hit_capacity,
                    buf.d_overflow_flag[s],
                    tile_size,
                    max_local_hits,
                    h_total_chunks[s]);
            }
        } // end if h_queue_tail > 0

        // --- CPU staging window: stage NEXT batch into pinned memory ---
        if (batch_idx + 1 < n_batches)
        {
            int next_s = (int)((batch_idx + 1) % 2);
            size_t next_offset = (batch_idx + 1) * batch_size;
            size_t next_count = std::min(batch_size, n_ray - next_offset);
            stage_ray_batch(buf.h_ray_pinned[next_s], batch_size, next_offset, next_count,
                            T1x, T1y, T1z, T2x, T2y, T2z, T3x, T3y, T3z,
                            Nx, Ny, Nz, D1x, D1y, D1z, D2x, D2y, D2z,
                            D3x, D3y, D3z, rD1, rD2, rD3);
        }

        // --- Read hit_count and overflow_flag, then download hit buffers ---
        CUDA_CHECK(cudaMemcpyAsync(&h_hit_count[s], buf.d_hit_count[s],
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, buf.stream[s]));
        CUDA_CHECK(cudaMemcpyAsync(&h_overflow_flag[s], buf.d_overflow_flag[s],
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, buf.stream[s]));
        CUDA_CHECK(cudaStreamSynchronize(buf.stream[s]));

        // --- Check for tile-level hit buffer overflow → auto-retry (Section 5.2) ---
        if (h_overflow_flag[s] != 0)
        {
            tile_size *= 2;
            max_local_hits = tile_size * K2_BLOCK_SIZE;

            // Reallocate hit output if h_hit_count exceeds hit_capacity
            if (h_hit_count[s] > hit_capacity)
            {
                CUDA_CHECK(cudaStreamSynchronize(buf.stream[0]));
                CUDA_CHECK(cudaStreamSynchronize(buf.stream[1]));

                uint32_t new_hit_cap = h_hit_count[s] * 2;
                for (int ss = 0; ss < 2; ++ss)
                {
                    cudaFree(buf.d_hit_ray_idx[ss]);
                    buf.d_hit_ray_idx[ss] = nullptr;
                    cudaFree(buf.d_hit_point_idx[ss]);
                    buf.d_hit_point_idx[ss] = nullptr;
                    CUDA_CHECK(cudaMalloc(&buf.d_hit_ray_idx[ss], new_hit_cap * sizeof(uint32_t)));
                    CUDA_CHECK(cudaMalloc(&buf.d_hit_point_idx[ss], new_hit_cap * sizeof(uint32_t)));
                    cudaFreeHost(buf.h_hit_ray_pinned[ss]);
                    buf.h_hit_ray_pinned[ss] = nullptr;
                    cudaFreeHost(buf.h_hit_point_pinned[ss]);
                    buf.h_hit_point_pinned[ss] = nullptr;
                    CUDA_CHECK(cudaMallocHost(&buf.h_hit_ray_pinned[ss], new_hit_cap * sizeof(uint32_t)));
                    CUDA_CHECK(cudaMallocHost(&buf.h_hit_point_pinned[ss], new_hit_cap * sizeof(uint32_t)));
                }
                hit_capacity = new_hit_cap;
                buf.hit_capacity = new_hit_cap;
            }

            // Re-run Kernel 2 (sort + segment table still valid)
            CUDA_CHECK(cudaMemsetAsync(buf.d_hit_count[s], 0, sizeof(uint32_t), buf.stream[s]));
            CUDA_CHECK(cudaMemsetAsync(buf.d_overflow_flag[s], 0, sizeof(uint32_t), buf.stream[s]));

            uint32_t smem_bytes = 2u * max_local_hits * sizeof(uint32_t);
            point_intersect<<<h_total_chunks[s], K2_BLOCK_SIZE, smem_bytes, buf.stream[s]>>>(
                buf.d_Px, buf.d_Py, buf.d_Pz,
                buf.d_SCI,
                sorted_ray_idx,
                buf.d_seg_sub_idx[s],
                buf.d_seg_offset[s],
                buf.d_seg_chunk_offset[s],
                h_num_segments[s],
                buf.d_ray_ptrs[s],
                buf.d_hit_ray_idx[s],
                buf.d_hit_point_idx[s],
                buf.d_hit_count[s],
                hit_capacity,
                buf.d_overflow_flag[s],
                tile_size,
                max_local_hits,
                h_total_chunks[s]);

            // Re-read hit_count
            CUDA_CHECK(cudaMemcpyAsync(&h_hit_count[s], buf.d_hit_count[s],
                                       sizeof(uint32_t), cudaMemcpyDeviceToHost, buf.stream[s]));
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[s]));
        }

        // --- Check for hit output overflow ---
        if (h_hit_count[s] > hit_capacity)
        {
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[1]));

            uint32_t new_hit_cap = h_hit_count[s] * 2;
            for (int ss = 0; ss < 2; ++ss)
            {
                cudaFree(buf.d_hit_ray_idx[ss]);
                buf.d_hit_ray_idx[ss] = nullptr;
                cudaFree(buf.d_hit_point_idx[ss]);
                buf.d_hit_point_idx[ss] = nullptr;
                CUDA_CHECK(cudaMalloc(&buf.d_hit_ray_idx[ss], new_hit_cap * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&buf.d_hit_point_idx[ss], new_hit_cap * sizeof(uint32_t)));
                cudaFreeHost(buf.h_hit_ray_pinned[ss]);
                buf.h_hit_ray_pinned[ss] = nullptr;
                cudaFreeHost(buf.h_hit_point_pinned[ss]);
                buf.h_hit_point_pinned[ss] = nullptr;
                CUDA_CHECK(cudaMallocHost(&buf.h_hit_ray_pinned[ss], new_hit_cap * sizeof(uint32_t)));
                CUDA_CHECK(cudaMallocHost(&buf.h_hit_point_pinned[ss], new_hit_cap * sizeof(uint32_t)));
            }
            hit_capacity = new_hit_cap;
            buf.hit_capacity = new_hit_cap;

            // Re-run Kernel 2 for this batch (sort + segment table still valid)
            CUDA_CHECK(cudaMemsetAsync(buf.d_hit_count[s], 0, sizeof(uint32_t), buf.stream[s]));
            CUDA_CHECK(cudaMemsetAsync(buf.d_overflow_flag[s], 0, sizeof(uint32_t), buf.stream[s]));

            uint32_t smem_bytes = 2u * max_local_hits * sizeof(uint32_t);
            point_intersect<<<h_total_chunks[s], K2_BLOCK_SIZE, smem_bytes, buf.stream[s]>>>(
                buf.d_Px, buf.d_Py, buf.d_Pz,
                buf.d_SCI,
                sorted_ray_idx,
                buf.d_seg_sub_idx[s],
                buf.d_seg_offset[s],
                buf.d_seg_chunk_offset[s],
                h_num_segments[s],
                buf.d_ray_ptrs[s],
                buf.d_hit_ray_idx[s],
                buf.d_hit_point_idx[s],
                buf.d_hit_count[s],
                hit_capacity,
                buf.d_overflow_flag[s],
                tile_size,
                max_local_hits,
                h_total_chunks[s]);

            CUDA_CHECK(cudaMemcpyAsync(&h_hit_count[s], buf.d_hit_count[s],
                                       sizeof(uint32_t), cudaMemcpyDeviceToHost, buf.stream[s]));
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[s]));
        }

        // --- Download hit data ---
        if (h_hit_count[s] > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(buf.h_hit_ray_pinned[s], buf.d_hit_ray_idx[s],
                                       h_hit_count[s] * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, buf.stream[s]));
            CUDA_CHECK(cudaMemcpyAsync(buf.h_hit_point_pinned[s], buf.d_hit_point_idx[s],
                                       h_hit_count[s] * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, buf.stream[s]));
        }
        // D→H will be consumed by the NEXT iteration's sync (or drain below)

    } // end batch loop

    // --- Drain last batch ---
    {
        int last_s = (int)((n_batches - 1) % 2);
        CUDA_CHECK(cudaStreamSynchronize(buf.stream[last_s]));

        for (uint32_t i = 0; i < h_hit_count[last_s]; ++i)
        {
            size_t actual_ray = batch_offsets[last_s] + buf.h_hit_ray_pinned[last_s][i];
            p_hit[actual_ray].push_back((unsigned)buf.h_hit_point_pinned[last_s][i]);
        }
    }

    // CudaBuffers destructor handles all cleanup (RAII)
}