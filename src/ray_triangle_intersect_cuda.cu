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

// CUDA variant of qd_RTI_AVX2 — Möller–Trumbore ray-triangle intersection

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>

#include <cub/cub.cuh>

#include "cuda_common.hpp" // replaces local CUDA_CHECK + cuda_runtime.h
#include "quadriga_lib_cuda_functions.hpp"

namespace
{ // ---------------------------------------------------------------

    // ============================================================================
    // Constants
    // ============================================================================

    static constexpr int BLOCK_SIZE = 256;
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32 (warp size)");

    // Conservative estimate for average AABB hits per ray (used for queue sizing)
    static constexpr int EST_AVG_HITS = 10;

    // Sentinel for packed FBS/SBS: t=1.0f (no hit), face=UINT32_MAX
    static constexpr uint64_t PACKED_SENTINEL = (uint64_t(0x3F800000u) << 32) | uint64_t(0xFFFFFFFFu);

    // ============================================================================
    // Device helper: pack/unpack float+face into uint64 for atomicMin
    // Section 7.1
    // ============================================================================

    __device__ __forceinline__
        uint64_t
        RTI_pack_hit(float t, uint32_t face_idx)
    {
        // IMPORTANT: t must already be clamped with fmaxf(t, 0.0f)
        // to prevent negative zero from breaking the comparison.
        return ((uint64_t)__float_as_uint(t) << 32) | (uint64_t)face_idx;
    }

    __device__ __forceinline__ float RTI_unpack_t(uint64_t packed)
    {
        return __uint_as_float((uint32_t)(packed >> 32));
    }

    __device__ __forceinline__
        uint32_t
        RTI_unpack_face(uint64_t packed)
    {
        return (uint32_t)(packed & 0xFFFFFFFFu);
    }

    // ============================================================================
    // Device helper: warp-level queue compaction (Section 7.2)
    // ============================================================================

    __device__ __forceinline__ void RTI_enqueue_warp(bool has_hit, uint32_t ray_idx, uint16_t aabb_idx,
                                                     uint32_t *__restrict__ wq_ray, uint16_t *__restrict__ wq_aabb,
                                                     uint32_t *__restrict__ queue_tail, uint32_t queue_capacity)
    {
        // All 32 lanes MUST execute this — invalid threads pass has_hit=false
        uint32_t hit_mask = __ballot_sync(0xFFFFFFFF, has_hit);
        if (hit_mask == 0u)
            return;

        uint32_t lane = threadIdx.x & 31u;
        uint32_t count = __popc(hit_mask);
        uint32_t warp_base;

        if (lane == 0u)
            warp_base = atomicAdd(queue_tail, count);
        warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

        uint32_t local_offset = __popc(hit_mask & ((1u << lane) - 1u));
        if (has_hit)
        {
            uint32_t pos = warp_base + local_offset;
            if (pos < queue_capacity) // guard against overflow — dropped writes are re-done
            {
                wq_ray[pos] = ray_idx;
                wq_aabb[pos] = aabb_idx;
            }
        }
    }

    // ============================================================================
    // Kernel 0: RTI_init_per_ray_state (Section 4.0)
    // ============================================================================

    __global__ void RTI_init_per_ray_state(
        uint64_t *__restrict__ fbs_packed,
        uint64_t *__restrict__ sbs_packed,
        uint32_t *__restrict__ hit_cnt_atomic,
        uint32_t *__restrict__ queue_tail,
        uint32_t n_ray_batch)
    {
        uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_ray_batch)
            return; // safe — no warp-level ops

        fbs_packed[ray_idx] = PACKED_SENTINEL;
        sbs_packed[ray_idx] = PACKED_SENTINEL;
        hit_cnt_atomic[ray_idx] = 0u;

        if (ray_idx == 0u)
            *queue_tail = 0u;
    }

    // ============================================================================
    // Kernel 1: RTI_aabb_test_and_enqueue (Section 4.1)
    // Slab test with NaN-safe ordered comparisons (no fminf/fmaxf).
    // ============================================================================

    __global__ void RTI_aabb_test_and_enqueue(
        // Ray data (batch)
        const float *__restrict__ d_Ox, const float *__restrict__ d_Oy, const float *__restrict__ d_Oz,
        const float *__restrict__ d_Dx, const float *__restrict__ d_Dy, const float *__restrict__ d_Dz,
        uint32_t n_ray_batch,
        // AABB bounds (scene)
        const float *__restrict__ Xmin, const float *__restrict__ Xmax,
        const float *__restrict__ Ymin, const float *__restrict__ Ymax,
        const float *__restrict__ Zmin, const float *__restrict__ Zmax,
        uint32_t n_sub,
        // Work queue output
        uint32_t *__restrict__ wq_ray_idx,
        uint16_t *__restrict__ wq_aabb_idx,
        uint32_t *__restrict__ queue_tail,
        uint32_t queue_capacity)
    {
        uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Partial warp safety: invalid threads must NOT early-return before __ballot_sync.
        // They participate with has_hit=false in every iteration.
        bool valid = (ray_idx < n_ray_batch);

        float ox, oy, oz, dx_i, dy_i, dz_i;
        if (valid)
        {
            ox = d_Ox[ray_idx];
            oy = d_Oy[ray_idx];
            oz = d_Oz[ray_idx];
            float dx = d_Dx[ray_idx];
            float dy = d_Dy[ray_idx];
            float dz = d_Dz[ray_idx];
            dx_i = 1.0f / dx;
            dy_i = 1.0f / dy;
            dz_i = 1.0f / dz;
        }

        for (uint32_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            bool has_hit = false;

            if (valid)
            {
                // Slab test — NaN-safe via conditional swaps (Section 4.1)
                float t0_lo = (Xmin[i_sub] - ox) * dx_i;
                float t0_hi = (Xmax[i_sub] - ox) * dx_i;
                if (t0_lo > t0_hi)
                {
                    float tmp = t0_lo;
                    t0_lo = t0_hi;
                    t0_hi = tmp;
                }

                float t1_lo = (Ymin[i_sub] - oy) * dy_i;
                float t1_hi = (Ymax[i_sub] - oy) * dy_i;
                if (t1_lo > t1_hi)
                {
                    float tmp = t1_lo;
                    t1_lo = t1_hi;
                    t1_hi = tmp;
                }

                float t2_lo = (Zmin[i_sub] - oz) * dz_i;
                float t2_hi = (Zmax[i_sub] - oz) * dz_i;
                if (t2_lo > t2_hi)
                {
                    float tmp = t2_lo;
                    t2_lo = t2_hi;
                    t2_hi = tmp;
                }

                // t_min = max of all lows, t_max = min of all highs (ordered comparisons)
                float t_min = (t0_lo > t1_lo) ? t0_lo : t1_lo;
                t_min = (t_min > t2_lo) ? t_min : t2_lo;
                float t_max = (t0_hi < t1_hi) ? t0_hi : t1_hi;
                t_max = (t_max < t2_hi) ? t_max : t2_hi;

                has_hit = (t_max > 0.0f) && (t_max >= t_min) && (t_min <= 1.0f);
            }

            // Warp-level compaction — all 32 lanes participate
            RTI_enqueue_warp(has_hit, ray_idx, (uint16_t)i_sub,
                             wq_ray_idx, wq_aabb_idx, queue_tail, queue_capacity);
        }
    }

    // ============================================================================
    // Kernel 2: RTI_moller_trumbore_fbs (Section 4.2)
    // First-bounce scatter — finds closest hit per ray via 64-bit atomicMin.
    // ============================================================================

    __global__ void __launch_bounds__(256, 2)
        RTI_moller_trumbore_fbs(
            // Sorted work queue
            const uint32_t *__restrict__ sorted_ray_idx,
            const uint16_t *__restrict__ sorted_aabb_idx,
            uint32_t queue_tail_val,
            // Ray data (batch)
            const float *__restrict__ d_Ox, const float *__restrict__ d_Oy, const float *__restrict__ d_Oz,
            const float *__restrict__ d_Dx, const float *__restrict__ d_Dy, const float *__restrict__ d_Dz,
            // Scene mesh
            const float *__restrict__ Tx, const float *__restrict__ Ty, const float *__restrict__ Tz,
            const float *__restrict__ E1x, const float *__restrict__ E1y, const float *__restrict__ E1z,
            const float *__restrict__ E2x, const float *__restrict__ E2y, const float *__restrict__ E2z,
            // Sub-mesh index (with sentinel)
            const uint32_t *__restrict__ SMI,
            // Outputs
            unsigned long long *__restrict__ fbs_packed, // uint64_t via atomicMin — use ULL for CUDA atomics
            uint32_t *__restrict__ hit_cnt_atomic,
            uint8_t *__restrict__ wq_had_hit)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= queue_tail_val)
            return; // safe: no warp-level intrinsics

        uint32_t ray_idx = sorted_ray_idx[i];
        uint16_t aabb_idx = sorted_aabb_idx[i];

        // Load ray — scattered access, one-time cost per work item
        float ox = d_Ox[ray_idx], oy = d_Oy[ray_idx], oz = d_Oz[ray_idx];
        float dx = d_Dx[ray_idx], dy = d_Dy[ray_idx], dz = d_Dz[ray_idx];

        // Face range — no branch, SMI has sentinel at n_sub
        uint32_t face_start = SMI[aabb_idx];
        uint32_t face_end = SMI[aabb_idx + 1u];

        float local_best_t = 1.0f;
        uint32_t local_best_face = 0xFFFFFFFFu;
        uint32_t local_hit_count = 0u;

        for (uint32_t i_face = face_start; i_face < face_end; ++i_face)
        {
            // Vector from V1 to origin
            float tx = ox - Tx[i_face];
            float ty = oy - Ty[i_face];
            float tz = oz - Tz[i_face];

            // Load edges
            float e1x = E1x[i_face], e1y = E1y[i_face], e1z = E1z[i_face];
            float e2x = E2x[i_face], e2y = E2y[i_face], e2z = E2z[i_face];

            // Möller–Trumbore — matching AVX2 FMA sequence exactly
            // PQ = cross(E2, D)
            float PQ_x = e2y * dz - e2z * dy;
            float PQ_y = e2z * dx - e2x * dz;
            float PQ_z = e2x * dy - e2y * dx;

            // DT = dot(E1, PQ)
            float DT = e1x * PQ_x + e1y * PQ_y + e1z * PQ_z;

            // U = dot(T, PQ)
            float U = tx * PQ_x + ty * PQ_y + tz * PQ_z;

            // QQ = cross(E1, T)  (reuse PQ variable names for clarity)
            float QQ_x = e1y * tz - e1z * ty;
            float QQ_y = e1z * tx - e1x * tz;
            float QQ_z = e1x * ty - e1y * tx;

            // V = dot(D, QQ)
            float V = dx * QQ_x + dy * QQ_y + dz * QQ_z;

            // W = dot(E2, QQ)
            float W = e2x * QQ_x + e2y * QQ_y + e2z * QQ_z;

            // Full-precision division (matches AVX2 default)
            float inv_DT = 1.0f / DT;
            U *= inv_DT;
            V *= inv_DT;
            W *= inv_DT;

            // Hit conditions (same as AVX2: U>=0, V>=0, U+V<=1, W>=0, W<1)
            if (U >= 0.0f && V >= 0.0f && (U + V) <= 1.0f && W >= 0.0f && W < 1.0f)
            {
                // Force positive zero to prevent negative-zero atomicMin bug (Section 7.1)
                W = fmaxf(W, 0.0f);

                local_hit_count++;
                if (W < local_best_t)
                {
                    local_best_t = W;
                    local_best_face = i_face;
                }
            }
        }

        // Write results via atomics
        if (local_best_face != 0xFFFFFFFFu)
        {
            uint64_t packed = RTI_pack_hit(local_best_t, local_best_face);
            atomicMin(fbs_packed + ray_idx, (unsigned long long)packed);
        }

        if (local_hit_count > 0u)
            atomicAdd(&hit_cnt_atomic[ray_idx], local_hit_count);

        // Set hit flag for CUB compaction
        wq_had_hit[i] = (local_hit_count > 0u) ? 1u : 0u;
    }

    // ============================================================================
    // Kernel 3: RTI_moller_trumbore_sbs (Section 4.3)
    // Second-bounce scatter — processes only compacted work items (had hit).
    // Skips the FBS face.
    // ============================================================================

    __global__ void __launch_bounds__(256, 2)
        RTI_moller_trumbore_sbs(
            // Compacted index array
            const uint32_t *__restrict__ compact_indices,
            uint32_t num_compact,
            // Sorted work queue (original, referenced via compact_indices)
            const uint32_t *__restrict__ sorted_ray_idx,
            const uint16_t *__restrict__ sorted_aabb_idx,
            // Ray data (batch)
            const float *__restrict__ d_Ox, const float *__restrict__ d_Oy, const float *__restrict__ d_Oz,
            const float *__restrict__ d_Dx, const float *__restrict__ d_Dy, const float *__restrict__ d_Dz,
            // Scene mesh
            const float *__restrict__ Tx, const float *__restrict__ Ty, const float *__restrict__ Tz,
            const float *__restrict__ E1x, const float *__restrict__ E1y, const float *__restrict__ E1z,
            const float *__restrict__ E2x, const float *__restrict__ E2y, const float *__restrict__ E2z,
            // Sub-mesh index (with sentinel)
            const uint32_t *__restrict__ SMI,
            // FBS packed (read-only here, to get fbs_face)
            const unsigned long long *__restrict__ fbs_packed,
            // SBS output
            unsigned long long *__restrict__ sbs_packed)
    {
        uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= num_compact)
            return;

        // Look up original sorted queue position via compact index
        uint32_t orig_idx = compact_indices[j];
        uint32_t ray_idx = sorted_ray_idx[orig_idx];
        uint16_t aabb_idx = sorted_aabb_idx[orig_idx];

        // Read the winning FBS face for this ray
        uint32_t fbs_face = RTI_unpack_face((uint64_t)fbs_packed[ray_idx]);

        // Load ray
        float ox = d_Ox[ray_idx], oy = d_Oy[ray_idx], oz = d_Oz[ray_idx];
        float dx = d_Dx[ray_idx], dy = d_Dy[ray_idx], dz = d_Dz[ray_idx];

        // Face range
        uint32_t face_start = SMI[aabb_idx];
        uint32_t face_end = SMI[aabb_idx + 1u];

        float local_best_t = 1.0f;
        uint32_t local_best_face = 0xFFFFFFFFu;

        for (uint32_t i_face = face_start; i_face < face_end; ++i_face)
        {
            if (i_face == fbs_face)
                continue;

            float tx = ox - Tx[i_face];
            float ty = oy - Ty[i_face];
            float tz = oz - Tz[i_face];

            float e1x = E1x[i_face], e1y = E1y[i_face], e1z = E1z[i_face];
            float e2x = E2x[i_face], e2y = E2y[i_face], e2z = E2z[i_face];

            float PQ_x = e2y * dz - e2z * dy;
            float PQ_y = e2z * dx - e2x * dz;
            float PQ_z = e2x * dy - e2y * dx;

            float DT = e1x * PQ_x + e1y * PQ_y + e1z * PQ_z;
            float U = tx * PQ_x + ty * PQ_y + tz * PQ_z;

            float QQ_x = e1y * tz - e1z * ty;
            float QQ_y = e1z * tx - e1x * tz;
            float QQ_z = e1x * ty - e1y * tx;

            float V = dx * QQ_x + dy * QQ_y + dz * QQ_z;
            float W = e2x * QQ_x + e2y * QQ_y + e2z * QQ_z;

            float inv_DT = 1.0f / DT;
            U *= inv_DT;
            V *= inv_DT;
            W *= inv_DT;

            if (U >= 0.0f && V >= 0.0f && (U + V) <= 1.0f && W >= 0.0f && W < 1.0f)
            {
                W = fmaxf(W, 0.0f); // negative-zero clamp
                if (W < local_best_t)
                {
                    local_best_t = W;
                    local_best_face = i_face;
                }
            }
        }

        // Write SBS via atomicMin
        if (local_best_face != 0xFFFFFFFFu)
        {
            uint64_t packed = RTI_pack_hit(local_best_t, local_best_face);
            atomicMin(sbs_packed + ray_idx, (unsigned long long)packed);
        }
    }

    // ============================================================================
    // Kernel 4: RTI_finalize_outputs (Section 4.4)
    // Unpacks packed FBS/SBS into user-facing Wf, Ws, If, Is, hit_cnt.
    // ============================================================================

    __global__ void RTI_finalize_outputs(
        const unsigned long long *__restrict__ fbs_packed,
        const unsigned long long *__restrict__ sbs_packed,
        const uint32_t *__restrict__ hit_cnt_atomic,
        float *__restrict__ d_Wf,
        float *__restrict__ d_Ws,
        uint32_t *__restrict__ d_If,
        uint32_t *__restrict__ d_Is,
        uint32_t *__restrict__ d_hit_cnt,
        uint32_t n_ray_batch,
        bool count_hits)
    {
        uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= n_ray_batch)
            return;

        uint64_t fbs = (uint64_t)fbs_packed[ray_idx];
        uint64_t sbs = (uint64_t)sbs_packed[ray_idx];

        float Wf_val = RTI_unpack_t(fbs);
        float Ws_val = RTI_unpack_t(sbs);
        uint32_t If_val = RTI_unpack_face(fbs);
        uint32_t Is_val = RTI_unpack_face(sbs);

        d_Wf[ray_idx] = Wf_val;
        d_Ws[ray_idx] = Ws_val;
        d_If[ray_idx] = (Wf_val < 1.0f) ? If_val + 1u : 0u; // 1-based, 0 = no hit
        d_Is[ray_idx] = (Ws_val < 1.0f) ? Is_val + 1u : 0u;

        if (count_hits)
            d_hit_cnt[ray_idx] = hit_cnt_atomic[ray_idx];
    }

    // ============================================================================
    // RAII cleanup struct (Section 8)
    // Holds all device and pinned-host pointers. Destructor frees everything.
    // ============================================================================

    struct CudaBuffers
    {
        // Scene data (device, uploaded once)
        float *d_Tx = nullptr, *d_Ty = nullptr, *d_Tz = nullptr;
        float *d_E1x = nullptr, *d_E1y = nullptr, *d_E1z = nullptr;
        float *d_E2x = nullptr, *d_E2y = nullptr, *d_E2z = nullptr;
        uint32_t *d_SMI = nullptr;
        float *d_Xmin = nullptr, *d_Xmax = nullptr;
        float *d_Ymin = nullptr, *d_Ymax = nullptr;
        float *d_Zmin = nullptr, *d_Zmax = nullptr;

        // Per-stream buffers (index 0 and 1 for double buffering)
        // Ray input (device)
        float *d_Ox[2] = {}, *d_Oy[2] = {}, *d_Oz[2] = {};
        float *d_Dx[2] = {}, *d_Dy[2] = {}, *d_Dz[2] = {};

        // Ray output (device)
        float *d_Wf[2] = {}, *d_Ws[2] = {};
        uint32_t *d_If[2] = {}, *d_Is[2] = {};
        uint32_t *d_hit_cnt[2] = {};

        // Intermediate (device)
        unsigned long long *d_fbs_packed[2] = {}; // uint64_t via ULL for CUDA atomicMin
        unsigned long long *d_sbs_packed[2] = {};
        uint32_t *d_hit_cnt_atomic[2] = {};

        // Work queue (device) — A/B pairs for CUB DoubleBuffer
        uint32_t *d_wq_ray_idx_A[2] = {}, *d_wq_ray_idx_B[2] = {};
        uint16_t *d_wq_aabb_idx_A[2] = {}, *d_wq_aabb_idx_B[2] = {};
        uint32_t *d_queue_tail[2] = {};
        uint8_t *d_wq_had_hit[2] = {};
        uint32_t *d_compact_indices[2] = {};
        uint32_t *d_num_selected[2] = {};

        // CUB temp buffer (device)
        void *d_cub_temp[2] = {};

        // Pinned host memory — staging buffers for async H↔D
        // Input: 6 float arrays interleaved into one flat buffer
        float *h_ray_input_pinned[2] = {};
        // Output: 2 floats + 3 uints interleaved into one flat buffer
        char *h_ray_output_pinned[2] = {};

        // CUDA streams
        cudaStream_t stream[2] = {};
        bool streams_created = false;

        // Sizes (for freeing)
        size_t pinned_input_bytes = 0;
        size_t pinned_output_bytes = 0;

        ~CudaBuffers()
        {
            // Scene data
            cudaFree(d_Tx);
            cudaFree(d_Ty);
            cudaFree(d_Tz);
            cudaFree(d_E1x);
            cudaFree(d_E1y);
            cudaFree(d_E1z);
            cudaFree(d_E2x);
            cudaFree(d_E2y);
            cudaFree(d_E2z);
            cudaFree(d_SMI);
            cudaFree(d_Xmin);
            cudaFree(d_Xmax);
            cudaFree(d_Ymin);
            cudaFree(d_Ymax);
            cudaFree(d_Zmin);
            cudaFree(d_Zmax);

            for (int s = 0; s < 2; ++s)
            {
                cudaFree(d_Ox[s]);
                cudaFree(d_Oy[s]);
                cudaFree(d_Oz[s]);
                cudaFree(d_Dx[s]);
                cudaFree(d_Dy[s]);
                cudaFree(d_Dz[s]);
                cudaFree(d_Wf[s]);
                cudaFree(d_Ws[s]);
                cudaFree(d_If[s]);
                cudaFree(d_Is[s]);
                cudaFree(d_hit_cnt[s]);
                cudaFree(d_fbs_packed[s]);
                cudaFree(d_sbs_packed[s]);
                cudaFree(d_hit_cnt_atomic[s]);
                cudaFree(d_wq_ray_idx_A[s]);
                cudaFree(d_wq_ray_idx_B[s]);
                cudaFree(d_wq_aabb_idx_A[s]);
                cudaFree(d_wq_aabb_idx_B[s]);
                cudaFree(d_queue_tail[s]);
                cudaFree(d_wq_had_hit[s]);
                cudaFree(d_compact_indices[s]);
                cudaFree(d_num_selected[s]);
                cudaFree(d_cub_temp[s]);

                if (h_ray_input_pinned[s])
                    cudaFreeHost(h_ray_input_pinned[s]);
                if (h_ray_output_pinned[s])
                    cudaFreeHost(h_ray_output_pinned[s]);
            }

            if (streams_created)
            {
                cudaStreamDestroy(stream[0]);
                cudaStreamDestroy(stream[1]);
            }
        }
    };

    // ============================================================================
    // Helper: ceil(log2(n)) — number of bits needed to represent values 0..n-1
    // ============================================================================

    static inline int RTI_ceil_log2(uint32_t n)
    {
        if (n <= 1u)
            return 1;
        int bits = 0;
        uint32_t v = n - 1u;
        while (v > 0u)
        {
            v >>= 1;
            ++bits;
        }
        return bits;
    }

    // ============================================================================
    // Helper: upload a float array host → device
    // ============================================================================

    static inline void RTI_upload_float(float *&d_ptr, const float *h_ptr, size_t count)
    {
        CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, count * sizeof(float), cudaMemcpyHostToDevice));
    }

    // ============================================================================
    // Helper: stage ray input batch from user arrays into pinned buffer
    // Layout in pinned buffer: [Ox ... | Oy ... | Oz ... | Dx ... | Dy ... | Dz ...]
    // ============================================================================

    static inline void RTI_stage_ray_input(float *pinned, size_t batch_size,
                                           const float *Ox, const float *Oy, const float *Oz,
                                           const float *Dx, const float *Dy, const float *Dz,
                                           size_t offset, size_t count)
    {
        memcpy(pinned + 0 * batch_size, Ox + offset, count * sizeof(float));
        memcpy(pinned + 1 * batch_size, Oy + offset, count * sizeof(float));
        memcpy(pinned + 2 * batch_size, Oz + offset, count * sizeof(float));
        memcpy(pinned + 3 * batch_size, Dx + offset, count * sizeof(float));
        memcpy(pinned + 4 * batch_size, Dy + offset, count * sizeof(float));
        memcpy(pinned + 5 * batch_size, Dz + offset, count * sizeof(float));
    }

    // ============================================================================
    // Helper: copy ray output from pinned buffer to user arrays
    // Layout: [Wf ... | Ws ... | If ... | Is ... | hit_cnt ...]
    // ============================================================================

    static inline void RTI_unstage_ray_output(const char *pinned, size_t batch_size,
                                              float *Wf, float *Ws,
                                              unsigned *If, unsigned *Is,
                                              unsigned *hit_cnt, bool count_hits,
                                              size_t offset, size_t count)
    {
        size_t f4 = batch_size * sizeof(float);
        size_t u4 = batch_size * sizeof(uint32_t);

        memcpy(Wf + offset, pinned, count * sizeof(float));
        memcpy(Ws + offset, pinned + f4, count * sizeof(float));
        memcpy(If + offset, pinned + 2 * f4, count * sizeof(uint32_t));
        memcpy(Is + offset, pinned + 2 * f4 + u4, count * sizeof(uint32_t));
        if (count_hits)
            memcpy(hit_cnt + offset, pinned + 2 * f4 + 2 * u4, count * sizeof(uint32_t));
    }

} // end anonymous namespace --------------------------------------------------

// Public entry point — external linkage
void qd_RTI_CUDA(
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
    unsigned *hit_cnt,
    int gpu_id)
{
    // --- Early returns (Section 8) ---
    if (n_ray == 0)
        return;

    if (n_sub == 0 || n_mesh == 0)
    {
        // Zero-fill all outputs on host — no GPU work needed
        for (size_t i = 0; i < n_ray; ++i)
        {
            Wf[i] = 1.0f;
            Ws[i] = 1.0f;
            If[i] = 0;
            Is[i] = 0;
        }
        if (hit_cnt)
            memset(hit_cnt, 0, n_ray * sizeof(unsigned));
        return;
    }

    // --- Range check: n_sub must fit in uint16 (Section 8) ---
    if (n_sub > 65535)
        throw std::invalid_argument("qd_RTI_CUDA: n_sub > 65535 not supported (uint16 AABB index).");

    bool count_hits = (hit_cnt != nullptr);

    // --- Set CUDA device ---
    CUDA_CHECK(cudaSetDevice(gpu_id));

    // --- RAII struct for all allocations ---
    CudaBuffers buf;

    // =====================================================================
    // Upload scene data (resident for the entire run)
    // =====================================================================

    RTI_upload_float(buf.d_Tx, Tx, n_mesh);
    RTI_upload_float(buf.d_Ty, Ty, n_mesh);
    RTI_upload_float(buf.d_Tz, Tz, n_mesh);
    RTI_upload_float(buf.d_E1x, E1x, n_mesh);
    RTI_upload_float(buf.d_E1y, E1y, n_mesh);
    RTI_upload_float(buf.d_E1z, E1z, n_mesh);
    RTI_upload_float(buf.d_E2x, E2x, n_mesh);
    RTI_upload_float(buf.d_E2y, E2y, n_mesh);
    RTI_upload_float(buf.d_E2z, E2z, n_mesh);

    // SMI sentinel: allocate n_sub+1 entries, set SMI[n_sub] = n_mesh
    {
        std::vector<uint32_t> smi_with_sentinel(n_sub + 1);
        for (size_t i = 0; i < n_sub; ++i)
            smi_with_sentinel[i] = (uint32_t)SMI[i];
        smi_with_sentinel[n_sub] = (uint32_t)n_mesh;

        CUDA_CHECK(cudaMalloc(&buf.d_SMI, (n_sub + 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(buf.d_SMI, smi_with_sentinel.data(),
                              (n_sub + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // AABB bounds
    RTI_upload_float(buf.d_Xmin, Xmin, n_sub);
    RTI_upload_float(buf.d_Xmax, Xmax, n_sub);
    RTI_upload_float(buf.d_Ymin, Ymin, n_sub);
    RTI_upload_float(buf.d_Ymax, Ymax, n_sub);
    RTI_upload_float(buf.d_Zmin, Zmin, n_sub);
    RTI_upload_float(buf.d_Zmax, Zmax, n_sub);

    // =====================================================================
    // Batch size calculation (Section 5.1)
    // =====================================================================

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    // Per-ray memory budget
    const size_t E = EST_AVG_HITS;
    size_t per_ray_bytes = 6 * 4         // input: Ox,Oy,Oz,Dx,Dy,Dz
                           + 5 * 4       // output: Wf,Ws,If,Is,hit_cnt
                           + 2 * 8 + 4   // intermediate: fbs_packed(8), sbs_packed(8), hit_cnt_atomic(4)
                           + E * (4 + 2) // work queue A: ray_idx(4) + aabb_idx(2)
                           + E * (4 + 2) // work queue B: sort output
                           + E * 1       // had_hit flags
                           + E * 4;      // compact_indices

    // Query CUB temp requirements at runtime
    size_t max_queue_estimate = (n_ray < 10000000) ? n_ray * E : 10000000ULL * E;

    size_t cub_sort_temp = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, cub_sort_temp,
                                    (uint16_t *)nullptr, (uint16_t *)nullptr,
                                    (uint32_t *)nullptr, (uint32_t *)nullptr,
                                    (int)std::min(max_queue_estimate, (size_t)INT32_MAX),
                                    0, 16, (cudaStream_t)0);

    size_t cub_compact_temp = 0;
    cub::DeviceSelect::Flagged(nullptr, cub_compact_temp,
                               cub::CountingInputIterator<uint32_t>(0),
                               (uint8_t *)nullptr, (uint32_t *)nullptr, (uint32_t *)nullptr,
                               (int)std::min(max_queue_estimate, (size_t)INT32_MAX),
                               (cudaStream_t)0);

    size_t cub_temp_per_stream = std::max(cub_sort_temp, cub_compact_temp);
    cub_temp_per_stream = (size_t)(cub_temp_per_stream * 1.2); // 20% margin
    if (cub_temp_per_stream < 1024)
        cub_temp_per_stream = 1024; // minimum sanity

    // Use 75% of free memory, double-buffered
    size_t usable = (size_t)(free_mem * 0.75);
    size_t batch_size = (usable - 2 * cub_temp_per_stream) / (2 * per_ray_bytes);
    batch_size = std::min(batch_size, n_ray);
    batch_size = (batch_size / BLOCK_SIZE) * BLOCK_SIZE; // align to block size
    if (batch_size == 0)
        batch_size = BLOCK_SIZE; // minimum 1 block

    size_t queue_capacity = batch_size * E;

    // =====================================================================
    // Create streams
    // =====================================================================

    CUDA_CHECK(cudaStreamCreate(&buf.stream[0]));
    CUDA_CHECK(cudaStreamCreate(&buf.stream[1]));
    buf.streams_created = true;

    // =====================================================================
    // Allocate per-stream device buffers + pinned host memory
    // =====================================================================

    size_t ray_f_bytes = batch_size * sizeof(float);
    size_t ray_u_bytes = batch_size * sizeof(uint32_t);
    size_t ray_8_bytes = batch_size * sizeof(uint64_t);
    size_t wq_ray_bytes = queue_capacity * sizeof(uint32_t);
    size_t wq_aabb_bytes = queue_capacity * sizeof(uint16_t);
    size_t wq_flag_bytes = queue_capacity * sizeof(uint8_t);
    size_t wq_comp_bytes = queue_capacity * sizeof(uint32_t);

    buf.pinned_input_bytes = 6 * ray_f_bytes;
    // Output: Wf(float) + Ws(float) + If(uint32) + Is(uint32) + hit_cnt(uint32)
    buf.pinned_output_bytes = 2 * ray_f_bytes + 3 * ray_u_bytes;

    for (int s = 0; s < 2; ++s)
    {
        // Ray input
        CUDA_CHECK(cudaMalloc(&buf.d_Ox[s], ray_f_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_Oy[s], ray_f_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_Oz[s], ray_f_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_Dx[s], ray_f_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_Dy[s], ray_f_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_Dz[s], ray_f_bytes));

        // Ray output
        CUDA_CHECK(cudaMalloc(&buf.d_Wf[s], ray_f_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_Ws[s], ray_f_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_If[s], ray_u_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_Is[s], ray_u_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_hit_cnt[s], ray_u_bytes));

        // Intermediate
        CUDA_CHECK(cudaMalloc(&buf.d_fbs_packed[s], ray_8_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_sbs_packed[s], ray_8_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_hit_cnt_atomic[s], ray_u_bytes));

        // Work queue
        CUDA_CHECK(cudaMalloc(&buf.d_wq_ray_idx_A[s], wq_ray_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_wq_ray_idx_B[s], wq_ray_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_wq_aabb_idx_A[s], wq_aabb_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_wq_aabb_idx_B[s], wq_aabb_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_queue_tail[s], sizeof(uint32_t))); // own alloc for cache line isolation
        CUDA_CHECK(cudaMalloc(&buf.d_wq_had_hit[s], wq_flag_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_compact_indices[s], wq_comp_bytes));
        CUDA_CHECK(cudaMalloc(&buf.d_num_selected[s], sizeof(uint32_t)));

        // CUB temp
        CUDA_CHECK(cudaMalloc(&buf.d_cub_temp[s], cub_temp_per_stream));

        // Pinned host memory
        CUDA_CHECK(cudaMallocHost(&buf.h_ray_input_pinned[s], buf.pinned_input_bytes));
        CUDA_CHECK(cudaMallocHost(&buf.h_ray_output_pinned[s], buf.pinned_output_bytes));
    }

    // =====================================================================
    // Compute number of bits for CUB radix sort
    // =====================================================================

    int num_aabb_bits = RTI_ceil_log2((uint32_t)n_sub);
    if (num_aabb_bits > 16)
        num_aabb_bits = 16;

    // =====================================================================
    // Batching loop — double-buffered async pipeline (Section 5.2)
    // =====================================================================

    size_t n_batches = (n_ray + batch_size - 1) / batch_size;

    // Host-side readback values (one per stream)
    uint32_t h_queue_tail[2] = {0, 0};
    uint32_t h_num_compact[2] = {0, 0};

    // Stage batch 0 into pinned memory BEFORE the loop
    {
        size_t count0 = std::min(batch_size, n_ray);
        RTI_stage_ray_input(buf.h_ray_input_pinned[0], batch_size,
                            Ox, Oy, Oz, Dx, Dy, Dz, 0, count0);
    }

    for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx)
    {
        int s = (int)(batch_idx % 2);        // current stream
        int sp = (int)((batch_idx - 1) % 2); // previous stream (wraps safely since unsigned)

        size_t ray_offset = batch_idx * batch_size;
        size_t n_ray_batch = std::min(batch_size, n_ray - ray_offset);
        uint32_t n_ray_batch_u = (uint32_t)n_ray_batch;

        // --- Wait for previous batch's D→H to finish, then copy to user arrays ---
        if (batch_idx > 0)
        {
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[sp]));

            size_t prev_offset = (batch_idx - 1) * batch_size;
            size_t prev_count = std::min(batch_size, n_ray - prev_offset);

            RTI_unstage_ray_output(buf.h_ray_output_pinned[sp], batch_size,
                                   Wf, Ws, If, Is, hit_cnt, count_hits,
                                   prev_offset, prev_count);
        }

        // --- H→D: upload ray data for this batch ---
        // Layout in pinned: [Ox | Oy | Oz | Dx | Dy | Dz], each batch_size floats
        cudaStream_t cs = buf.stream[s];

        CUDA_CHECK(cudaMemcpyAsync(buf.d_Ox[s], buf.h_ray_input_pinned[s] + 0 * batch_size,
                                   n_ray_batch * sizeof(float), cudaMemcpyHostToDevice, cs));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_Oy[s], buf.h_ray_input_pinned[s] + 1 * batch_size,
                                   n_ray_batch * sizeof(float), cudaMemcpyHostToDevice, cs));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_Oz[s], buf.h_ray_input_pinned[s] + 2 * batch_size,
                                   n_ray_batch * sizeof(float), cudaMemcpyHostToDevice, cs));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_Dx[s], buf.h_ray_input_pinned[s] + 3 * batch_size,
                                   n_ray_batch * sizeof(float), cudaMemcpyHostToDevice, cs));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_Dy[s], buf.h_ray_input_pinned[s] + 4 * batch_size,
                                   n_ray_batch * sizeof(float), cudaMemcpyHostToDevice, cs));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_Dz[s], buf.h_ray_input_pinned[s] + 5 * batch_size,
                                   n_ray_batch * sizeof(float), cudaMemcpyHostToDevice, cs));

        // --- Kernel 0: init per-ray state ---
        uint32_t grid0 = (n_ray_batch_u + BLOCK_SIZE - 1) / BLOCK_SIZE;
        RTI_init_per_ray_state<<<grid0, BLOCK_SIZE, 0, cs>>>(
            (uint64_t *)buf.d_fbs_packed[s],
            (uint64_t *)buf.d_sbs_packed[s],
            buf.d_hit_cnt_atomic[s],
            buf.d_queue_tail[s],
            n_ray_batch_u);

        // --- Kernel 1: AABB test + enqueue ---
        RTI_aabb_test_and_enqueue<<<grid0, BLOCK_SIZE, 0, cs>>>(
            buf.d_Ox[s], buf.d_Oy[s], buf.d_Oz[s],
            buf.d_Dx[s], buf.d_Dy[s], buf.d_Dz[s],
            n_ray_batch_u,
            buf.d_Xmin, buf.d_Xmax,
            buf.d_Ymin, buf.d_Ymax,
            buf.d_Zmin, buf.d_Zmax,
            (uint32_t)n_sub,
            buf.d_wq_ray_idx_A[s], buf.d_wq_aabb_idx_A[s],
            buf.d_queue_tail[s],
            (uint32_t)queue_capacity);

        // --- Read queue_tail back to host (4 bytes — sync point 1) ---
        CUDA_CHECK(cudaMemcpyAsync(&h_queue_tail[s], buf.d_queue_tail[s],
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, cs));
        CUDA_CHECK(cudaStreamSynchronize(cs));

        // --- Check for queue overflow ---
        if (h_queue_tail[s] > (uint32_t)queue_capacity)
        {
            // Sync both streams before reallocating
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[0]));
            CUDA_CHECK(cudaStreamSynchronize(buf.stream[1]));

            // Calculate new capacity: 2x the actual count
            size_t new_capacity = (size_t)h_queue_tail[s] * 2;

            size_t new_wq_ray_bytes = new_capacity * sizeof(uint32_t);
            size_t new_wq_aabb_bytes = new_capacity * sizeof(uint16_t);
            size_t new_wq_flag_bytes = new_capacity * sizeof(uint8_t);
            size_t new_wq_comp_bytes = new_capacity * sizeof(uint32_t);

            // Free and reallocate queue buffers for BOTH streams
            for (int ss = 0; ss < 2; ++ss)
            {
                cudaFree(buf.d_wq_ray_idx_A[ss]);
                buf.d_wq_ray_idx_A[ss] = nullptr;
                cudaFree(buf.d_wq_ray_idx_B[ss]);
                buf.d_wq_ray_idx_B[ss] = nullptr;
                cudaFree(buf.d_wq_aabb_idx_A[ss]);
                buf.d_wq_aabb_idx_A[ss] = nullptr;
                cudaFree(buf.d_wq_aabb_idx_B[ss]);
                buf.d_wq_aabb_idx_B[ss] = nullptr;
                cudaFree(buf.d_wq_had_hit[ss]);
                buf.d_wq_had_hit[ss] = nullptr;
                cudaFree(buf.d_compact_indices[ss]);
                buf.d_compact_indices[ss] = nullptr;

                CUDA_CHECK(cudaMalloc(&buf.d_wq_ray_idx_A[ss], new_wq_ray_bytes));
                CUDA_CHECK(cudaMalloc(&buf.d_wq_ray_idx_B[ss], new_wq_ray_bytes));
                CUDA_CHECK(cudaMalloc(&buf.d_wq_aabb_idx_A[ss], new_wq_aabb_bytes));
                CUDA_CHECK(cudaMalloc(&buf.d_wq_aabb_idx_B[ss], new_wq_aabb_bytes));
                CUDA_CHECK(cudaMalloc(&buf.d_wq_had_hit[ss], new_wq_flag_bytes));
                CUDA_CHECK(cudaMalloc(&buf.d_compact_indices[ss], new_wq_comp_bytes));
            }

            // Re-query CUB temp size for new capacity and reallocate if needed
            size_t new_sort_temp = 0, new_compact_temp = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, new_sort_temp,
                                            (uint16_t *)nullptr, (uint16_t *)nullptr,
                                            (uint32_t *)nullptr, (uint32_t *)nullptr,
                                            (int)std::min(new_capacity, (size_t)INT32_MAX), 0, 16, (cudaStream_t)0);
            cub::DeviceSelect::Flagged(nullptr, new_compact_temp,
                                       cub::CountingInputIterator<uint32_t>(0),
                                       (uint8_t *)nullptr, (uint32_t *)nullptr, (uint32_t *)nullptr,
                                       (int)std::min(new_capacity, (size_t)INT32_MAX), (cudaStream_t)0);
            size_t new_cub_temp = (size_t)(std::max(new_sort_temp, new_compact_temp) * 1.2);

            if (new_cub_temp > cub_temp_per_stream)
            {
                for (int ss = 0; ss < 2; ++ss)
                {
                    cudaFree(buf.d_cub_temp[ss]);
                    buf.d_cub_temp[ss] = nullptr;
                    CUDA_CHECK(cudaMalloc(&buf.d_cub_temp[ss], new_cub_temp));
                }
                cub_temp_per_stream = new_cub_temp;
            }

            queue_capacity = new_capacity;

            // Re-run current batch from Kernel 0
            RTI_init_per_ray_state<<<grid0, BLOCK_SIZE, 0, cs>>>(
                (uint64_t *)buf.d_fbs_packed[s],
                (uint64_t *)buf.d_sbs_packed[s],
                buf.d_hit_cnt_atomic[s],
                buf.d_queue_tail[s],
                n_ray_batch_u);

            RTI_aabb_test_and_enqueue<<<grid0, BLOCK_SIZE, 0, cs>>>(
                buf.d_Ox[s], buf.d_Oy[s], buf.d_Oz[s],
                buf.d_Dx[s], buf.d_Dy[s], buf.d_Dz[s],
                n_ray_batch_u,
                buf.d_Xmin, buf.d_Xmax,
                buf.d_Ymin, buf.d_Ymax,
                buf.d_Zmin, buf.d_Zmax,
                (uint32_t)n_sub,
                buf.d_wq_ray_idx_A[s], buf.d_wq_aabb_idx_A[s],
                buf.d_queue_tail[s],
                (uint32_t)queue_capacity);

            CUDA_CHECK(cudaMemcpyAsync(&h_queue_tail[s], buf.d_queue_tail[s],
                                       sizeof(uint32_t), cudaMemcpyDeviceToHost, cs));
            CUDA_CHECK(cudaStreamSynchronize(cs));
        }

        uint32_t qt = h_queue_tail[s];

        // --- CUB radix sort: work queue by aabb_idx (Section 6) ---
        // Use DoubleBuffer to avoid explicit input/output management
        cub::DoubleBuffer<uint16_t> d_sort_keys(buf.d_wq_aabb_idx_A[s], buf.d_wq_aabb_idx_B[s]);
        cub::DoubleBuffer<uint32_t> d_sort_vals(buf.d_wq_ray_idx_A[s], buf.d_wq_ray_idx_B[s]);

        if (qt > 0)
        {
            size_t sort_temp_bytes = cub_temp_per_stream;
            cub::DeviceRadixSort::SortPairs(
                buf.d_cub_temp[s], sort_temp_bytes,
                d_sort_keys, d_sort_vals,
                (int)qt,
                0, num_aabb_bits,
                cs);
        }

        // After sort, current buffers:
        uint16_t *sorted_aabb = d_sort_keys.Current();
        uint32_t *sorted_ray = d_sort_vals.Current();

        // --- Kernel 2: Möller–Trumbore FBS ---
        if (qt > 0)
        {
            uint32_t grid2 = (qt + BLOCK_SIZE - 1) / BLOCK_SIZE;
            RTI_moller_trumbore_fbs<<<grid2, BLOCK_SIZE, 0, cs>>>(
                sorted_ray, sorted_aabb, qt,
                buf.d_Ox[s], buf.d_Oy[s], buf.d_Oz[s],
                buf.d_Dx[s], buf.d_Dy[s], buf.d_Dz[s],
                buf.d_Tx, buf.d_Ty, buf.d_Tz,
                buf.d_E1x, buf.d_E1y, buf.d_E1z,
                buf.d_E2x, buf.d_E2y, buf.d_E2z,
                buf.d_SMI,
                buf.d_fbs_packed[s],
                buf.d_hit_cnt_atomic[s],
                buf.d_wq_had_hit[s]);
        }

        // --- CUB compaction: DeviceSelect::Flagged (Section 4.2a) ---
        if (qt > 0)
        {
            cub::CountingInputIterator<uint32_t> iota(0);
            size_t compact_temp_bytes = cub_temp_per_stream;

            cub::DeviceSelect::Flagged(
                buf.d_cub_temp[s], compact_temp_bytes,
                iota,
                buf.d_wq_had_hit[s],
                buf.d_compact_indices[s],
                buf.d_num_selected[s],
                (int)qt,
                cs);
        }
        else
        {
            // No queue entries → zero compacted
            CUDA_CHECK(cudaMemsetAsync(buf.d_num_selected[s], 0, sizeof(uint32_t), cs));
        }

        // Read num_selected to host (4 bytes — sync point 2)
        CUDA_CHECK(cudaMemcpyAsync(&h_num_compact[s], buf.d_num_selected[s],
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, cs));

        // --- CPU staging window: GPU is busy with sort + K2 + compact ---
        // Stage NEXT batch into pinned memory (overlaps with GPU work)
        if (batch_idx + 1 < n_batches)
        {
            int next_s = (int)((batch_idx + 1) % 2);
            size_t next_offset = (batch_idx + 1) * batch_size;
            size_t next_count = std::min(batch_size, n_ray - next_offset);
            RTI_stage_ray_input(buf.h_ray_input_pinned[next_s], batch_size,
                                Ox, Oy, Oz, Dx, Dy, Dz, next_offset, next_count);
        }

        CUDA_CHECK(cudaStreamSynchronize(cs)); // need num_compact for K3 grid

        uint32_t nc = h_num_compact[s];

        // --- Kernel 3: Möller–Trumbore SBS (only compacted work items) ---
        if (nc > 0)
        {
            uint32_t grid3 = (nc + BLOCK_SIZE - 1) / BLOCK_SIZE;
            RTI_moller_trumbore_sbs<<<grid3, BLOCK_SIZE, 0, cs>>>(
                buf.d_compact_indices[s], nc,
                sorted_ray, sorted_aabb,
                buf.d_Ox[s], buf.d_Oy[s], buf.d_Oz[s],
                buf.d_Dx[s], buf.d_Dy[s], buf.d_Dz[s],
                buf.d_Tx, buf.d_Ty, buf.d_Tz,
                buf.d_E1x, buf.d_E1y, buf.d_E1z,
                buf.d_E2x, buf.d_E2y, buf.d_E2z,
                buf.d_SMI,
                buf.d_fbs_packed[s],
                buf.d_sbs_packed[s]);
        }

        // --- Kernel 4: finalize outputs ---
        RTI_finalize_outputs<<<grid0, BLOCK_SIZE, 0, cs>>>(
            buf.d_fbs_packed[s],
            buf.d_sbs_packed[s],
            buf.d_hit_cnt_atomic[s],
            buf.d_Wf[s], buf.d_Ws[s],
            buf.d_If[s], buf.d_Is[s],
            buf.d_hit_cnt[s],
            n_ray_batch_u,
            count_hits);

        // --- D→H: download results ---
        // Layout in pinned output: [Wf | Ws | If | Is | hit_cnt]
        size_t out_off = 0;
        CUDA_CHECK(cudaMemcpyAsync(buf.h_ray_output_pinned[s] + out_off,
                                   buf.d_Wf[s], n_ray_batch * sizeof(float),
                                   cudaMemcpyDeviceToHost, cs));
        out_off += batch_size * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(buf.h_ray_output_pinned[s] + out_off,
                                   buf.d_Ws[s], n_ray_batch * sizeof(float),
                                   cudaMemcpyDeviceToHost, cs));
        out_off += batch_size * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(buf.h_ray_output_pinned[s] + out_off,
                                   buf.d_If[s], n_ray_batch * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, cs));
        out_off += batch_size * sizeof(uint32_t);

        CUDA_CHECK(cudaMemcpyAsync(buf.h_ray_output_pinned[s] + out_off,
                                   buf.d_Is[s], n_ray_batch * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, cs));
        out_off += batch_size * sizeof(uint32_t);

        if (count_hits)
        {
            CUDA_CHECK(cudaMemcpyAsync(buf.h_ray_output_pinned[s] + out_off,
                                       buf.d_hit_cnt[s], n_ray_batch * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, cs));
        }
    }

    // --- Drain last batch ---
    {
        int last_s = (int)((n_batches - 1) % 2);
        CUDA_CHECK(cudaStreamSynchronize(buf.stream[last_s]));

        size_t last_offset = (n_batches - 1) * batch_size;
        size_t last_count = n_ray - last_offset;

        RTI_unstage_ray_output(buf.h_ray_output_pinned[last_s], batch_size,
                               Wf, Ws, If, Is, hit_cnt, count_hits,
                               last_offset, last_count);
    }

    // RAII destructor handles all cleanup
}
