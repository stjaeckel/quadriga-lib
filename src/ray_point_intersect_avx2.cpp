// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <math.h>
#include <stdexcept>
#include <cstdint>
#include <immintrin.h>

#include "quadriga_lib_avx2_functions.hpp"
#include "quadriga_lib_generic_functions.hpp" // qd_RPI_assemble

#if defined(_MSC_VER) // Windows
#include <intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Vector size for AVX2
#define VEC_SIZE 8

// Number of packed hit records per staging chunk
static const size_t QD_RPI_CHUNK = 16384;

// Index of the lowest set bit, mask must not be zero
static inline int qd_first_set_bit(int mask)
{
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, (unsigned long)mask);
    return (int)index;
#else
    return __builtin_ctz((unsigned)mask);
#endif
}

// AVX2 accelerated implementation of RayPointIntersect
void qd_RPI_AVX2(const float *Px, const float *Py, const float *Pz,    // Point coordinates, length n_point (padded to multiple of 8)
                 const size_t n_point,                                 // Number of points, the arrays must be allocated up to the next multiple of 8
                 const unsigned *SCI,                                  // List of sub-cloud indices, length n_sub
                 const float *Xmin, const float *Xmax,                 // Minimum and maximum x-values of the AABB, length n_sub_s
                 const float *Ymin, const float *Ymax,                 // Minimum and maximum y-values of the AABB, length n_sub_s
                 const float *Zmin, const float *Zmax,                 // Minimum and maximum z-values of the AABB, length n_sub_s
                 const size_t n_sub,                                   // Number of sub-clouds (n_sub <= n_sub_s)
                 const float *T1x, const float *T1y, const float *T1z, // First ray vertex coordinate in GCS, length n_ray
                 const float *T2x, const float *T2y, const float *T2z, // Second ray vertex coordinate in GCS, length n_ray
                 const float *T3x, const float *T3y, const float *T3z, // Third ray vertex coordinate in GCS, length n_ray
                 const float *Nx, const float *Ny, const float *Nz,    // Ray tube normal vector, length n_ray
                 const float *D1x, const float *D1y, const float *D1z, // First ray direction in GCS, length n_ray
                 const float *D2x, const float *D2y, const float *D2z, // Second ray direction in GCS, length n_ray
                 const float *D3x, const float *D3y, const float *D3z, // Third ray direction in GCS, length n_ray
                 const float *rD1, const float *rD2, const float *rD3, // Inverse Dot product of ray direction and normal vector
                 const size_t n_ray,                                   // Number of rays
                 std::vector<unsigned> *hit_index,                     // Output: flat list of 0-based ray indices grouped by point, resized by the kernel
                 unsigned *hit_offset)                                 // Output: 0-based start of each point's block, length n_point + 1, allocated by the caller

{
    // Point and ray indices are packed into 32 bit fields of the hit records
    if ((unsigned long long)n_point > 0xFFFFFFFFULL)
        throw std::invalid_argument("Number of points exceeds maximum supported number.");
    if ((unsigned long long)n_ray > 0xFFFFFFFFULL)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    // Constant values needed for some operations
    const size_t n_sub_s = (n_sub % VEC_SIZE == 0) ? n_sub : VEC_SIZE * (n_sub / VEC_SIZE + 1);
    const long long n_point_l = (long long)n_point; // Number of points as signed 64 bit
    const long long n_ray_l = (long long)n_ray;     // Number of rays as signed 64 bit

    // The point arrays are read in blocks of 8, the last block may reach past
    // the real points. Those lanes are discarded when the hits are emitted.
    const long long n_point_pad = (long long)(VEC_SIZE * ((n_point + VEC_SIZE - 1) / VEC_SIZE));
    const __m256 r0 = _mm256_set1_ps(0.0f);         // Zero (float8)
    const __m256 r1 = _mm256_set1_ps(1.0f);         // One (float8)
    const __m256 r_slack = _mm256_set1_ps(1.0e-5f); // Small value for numeric stability

#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
#else
    const int n_threads = 1;
#endif

    // Per-thread staging, a list of fixed size chunks of packed hit records
    // Appending a chunk never moves the records already written, so the staged
    // data is written exactly once and the peak memory is the payload size
    std::vector<std::vector<std::vector<unsigned long long>>> stage((size_t)n_threads);

#pragma omp parallel
    {
#ifdef _OPENMP
        const size_t i_thread = (size_t)omp_get_thread_num();
#else
        const size_t i_thread = 0;
#endif
        std::vector<std::vector<unsigned long long>> &chunks = stage[i_thread];
        chunks.reserve(64);
        chunks.push_back(std::vector<unsigned long long>(QD_RPI_CHUNK));
        unsigned long long *p_write = chunks.back().data();
        unsigned long long *p_stop = p_write + QD_RPI_CHUNK;

        // Per-thread storage for sub-cloud hit indicators, reused across ray iterations
        std::vector<int> sub_hit_vec(n_sub_s);
        int *p_sub_hit = sub_hit_vec.data();

        // Static scheduling gives each thread one contiguous ray range, so the
        // staged hits are ray-ascending both within and across the buffers
#pragma omp for schedule(static)
        for (long long i_ray = 0; i_ray < n_ray_l; ++i_ray) // Ray loop
        {
        // Ray index in the low 32 bit of every hit record of this ray
        const unsigned long long ray_bits = (unsigned long long)i_ray;

        // Load origin into AVX2 registers
        __m256 ox0 = _mm256_set1_ps(T1x[i_ray]);
        __m256 oy0 = _mm256_set1_ps(T1y[i_ray]);
        __m256 oz0 = _mm256_set1_ps(T1z[i_ray]);
        __m256 ox1 = _mm256_set1_ps(T2x[i_ray]);
        __m256 oy1 = _mm256_set1_ps(T2y[i_ray]);
        __m256 oz1 = _mm256_set1_ps(T2z[i_ray]);
        __m256 ox2 = _mm256_set1_ps(T3x[i_ray]);
        __m256 oy2 = _mm256_set1_ps(T3y[i_ray]);
        __m256 oz2 = _mm256_set1_ps(T3z[i_ray]);

        // Load direction into AVX2 registers
        __m256 dx0 = _mm256_set1_ps(D1x[i_ray]);
        __m256 dy0 = _mm256_set1_ps(D1y[i_ray]);
        __m256 dz0 = _mm256_set1_ps(D1z[i_ray]);
        __m256 dx1 = _mm256_set1_ps(D2x[i_ray]);
        __m256 dy1 = _mm256_set1_ps(D2y[i_ray]);
        __m256 dz1 = _mm256_set1_ps(D2z[i_ray]);
        __m256 dx2 = _mm256_set1_ps(D3x[i_ray]);
        __m256 dy2 = _mm256_set1_ps(D3y[i_ray]);
        __m256 dz2 = _mm256_set1_ps(D3z[i_ray]);

        // Load normal vector into AVX2 registers
        __m256 nx = _mm256_set1_ps(Nx[i_ray]);
        __m256 ny = _mm256_set1_ps(Ny[i_ray]);
        __m256 nz = _mm256_set1_ps(Nz[i_ray]);

        // Load inverse dot product into AVX2 registers
        __m256 rdx = _mm256_set1_ps(rD1[i_ray]);
        __m256 rdy = _mm256_set1_ps(rD2[i_ray]);
        __m256 rdz = _mm256_set1_ps(rD3[i_ray]);

        // Step 1 - Check for possible hits
        // - Move the wavefront forward relative to the distance between vertex origin and AABB corner point
        // - Construct second AABB from advanced wavefronts
        // - If AABBs overlap, there is a potential hit and individual points must be checked in step 2

        // Multiply the origin and normal vector
        __m256 ox0_x_nx = _mm256_mul_ps(ox0, nx);
        __m256 oy0_x_ny = _mm256_mul_ps(oy0, ny);
        __m256 oz0_x_nz = _mm256_mul_ps(oz0, nz);
        __m256 ox1_x_nx = _mm256_mul_ps(ox1, nx);
        __m256 oy1_x_ny = _mm256_mul_ps(oy1, ny);
        __m256 oz1_x_nz = _mm256_mul_ps(oz1, nz);
        __m256 ox2_x_nx = _mm256_mul_ps(ox2, nx);
        __m256 oy2_x_ny = _mm256_mul_ps(oy2, ny);
        __m256 oz2_x_nz = _mm256_mul_ps(oz2, nz);

        for (size_t i_sub = 0; i_sub < n_sub_s; i_sub += VEC_SIZE)
        {
            // Load point bounding box
            __m256 b0_low = _mm256_loadu_ps(&Xmin[i_sub]);
            __m256 b0_high = _mm256_loadu_ps(&Xmax[i_sub]);
            __m256 b1_low = _mm256_loadu_ps(&Ymin[i_sub]);
            __m256 b1_high = _mm256_loadu_ps(&Ymax[i_sub]);
            __m256 b2_low = _mm256_loadu_ps(&Zmin[i_sub]);
            __m256 b2_high = _mm256_loadu_ps(&Zmax[i_sub]);

            // Add some slack for numeric stability
            b0_low = _mm256_sub_ps(b0_low, r_slack);
            b1_low = _mm256_sub_ps(b1_low, r_slack);
            b2_low = _mm256_sub_ps(b2_low, r_slack);
            b0_high = _mm256_add_ps(b0_high, r_slack);
            b1_high = _mm256_add_ps(b1_high, r_slack);
            b2_high = _mm256_add_ps(b2_high, r_slack);

            // AABB corner points
            __m256 rx[8] = {b0_low, b0_low, b0_low, b0_low, b0_high, b0_high, b0_high, b0_high};
            __m256 ry[8] = {b1_low, b1_low, b1_high, b1_high, b1_low, b1_low, b1_high, b1_high};
            __m256 rz[8] = {b2_low, b2_high, b2_low, b2_high, b2_low, b2_high, b2_low, b2_high};

            // Initialize coordinates for the vertex box
            __m256 a0_low = _mm256_set1_ps(INFINITY);
            __m256 a1_low = _mm256_set1_ps(INFINITY);
            __m256 a2_low = _mm256_set1_ps(INFINITY);
            __m256 a0_high = _mm256_set1_ps(-INFINITY);
            __m256 a1_high = _mm256_set1_ps(-INFINITY);
            __m256 a2_high = _mm256_set1_ps(-INFINITY);

            // Calculate the vertex box at the advanced wavefront
            for (int i = 0; i < 8; ++i)
            {
                // Distance between vertex origin and wavefront at corner point
                __m256 v = _mm256_fmsub_ps(rz[i], nz, oz0_x_nz);
                __m256 d = _mm256_mul_ps(rdx, v);
                v = _mm256_fmsub_ps(ry[i], ny, oy0_x_ny);
                d = _mm256_fmadd_ps(rdx, v, d);
                v = _mm256_fmsub_ps(rx[i], nx, ox0_x_nx);
                d = _mm256_fmadd_ps(rdx, v, d);

                // Update vertex box at advanced wavefront
                v = _mm256_fmadd_ps(d, dx0, ox0);
                a0_low = _mm256_min_ps(v, a0_low);
                a0_high = _mm256_max_ps(v, a0_high);

                v = _mm256_fmadd_ps(d, dy0, oy0);
                a1_low = _mm256_min_ps(v, a1_low);
                a1_high = _mm256_max_ps(v, a1_high);

                v = _mm256_fmadd_ps(d, dz0, oz0);
                a2_low = _mm256_min_ps(v, a2_low);
                a2_high = _mm256_max_ps(v, a2_high);

                // 2nd vertex
                v = _mm256_fmsub_ps(rz[i], nz, oz1_x_nz);
                d = _mm256_mul_ps(rdy, v);
                v = _mm256_fmsub_ps(ry[i], ny, oy1_x_ny);
                d = _mm256_fmadd_ps(rdy, v, d);
                v = _mm256_fmsub_ps(rx[i], nx, ox1_x_nx);
                d = _mm256_fmadd_ps(rdy, v, d);

                v = _mm256_fmadd_ps(d, dx1, ox1);
                a0_low = _mm256_min_ps(v, a0_low);
                a0_high = _mm256_max_ps(v, a0_high);

                v = _mm256_fmadd_ps(d, dy1, oy1);
                a1_low = _mm256_min_ps(v, a1_low);
                a1_high = _mm256_max_ps(v, a1_high);

                v = _mm256_fmadd_ps(d, dz1, oz1);
                a2_low = _mm256_min_ps(v, a2_low);
                a2_high = _mm256_max_ps(v, a2_high);

                // 3rd vertex
                v = _mm256_fmsub_ps(rz[i], nz, oz2_x_nz);
                d = _mm256_mul_ps(rdz, v);
                v = _mm256_fmsub_ps(ry[i], ny, oy2_x_ny);
                d = _mm256_fmadd_ps(rdz, v, d);
                v = _mm256_fmsub_ps(rx[i], nx, ox2_x_nx);
                d = _mm256_fmadd_ps(rdz, v, d);

                v = _mm256_fmadd_ps(d, dx2, ox2);
                a0_low = _mm256_min_ps(v, a0_low);
                a0_high = _mm256_max_ps(v, a0_high);

                v = _mm256_fmadd_ps(d, dy2, oy2);
                a1_low = _mm256_min_ps(v, a1_low);
                a1_high = _mm256_max_ps(v, a1_high);

                v = _mm256_fmadd_ps(d, dz2, oz2);
                a2_low = _mm256_min_ps(v, a2_low);
                a2_high = _mm256_max_ps(v, a2_high);
            }

            // Check for a potential overlap between the AABBs
            __m256 C = _mm256_cmp_ps(a0_high, b0_low, _CMP_GE_OQ); // a0_high >= b0_low
            __m256 D = _mm256_cmp_ps(a0_low, b0_high, _CMP_LE_OQ); // a0_low <= b0_high
            C = _mm256_and_ps(C, D);
            D = _mm256_cmp_ps(a1_high, b1_low, _CMP_GE_OQ); // a1_high >= b1_low
            C = _mm256_and_ps(C, D);
            D = _mm256_cmp_ps(a1_low, b1_high, _CMP_LE_OQ); // a1_low <= b1_high
            C = _mm256_and_ps(C, D);
            D = _mm256_cmp_ps(a2_high, b2_low, _CMP_GE_OQ); // a2_high >= b2_low
            C = _mm256_and_ps(C, D);
            D = _mm256_cmp_ps(a2_low, b2_high, _CMP_LE_OQ); // a2_low <= b2_high
            C = _mm256_and_ps(C, D);

            // Convert the result to an integer vector
            __m256i final_result_int = _mm256_castps_si256(C);

            // Store result (-1 = hit, 0 = miss)
            _mm256_storeu_si256((__m256i *)&p_sub_hit[i_sub], final_result_int);
        }

        // Step 2 - Check intersection with points within the sub-clouds

        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Skip if sub-cloud was not hit
            if (p_sub_hit[i_sub] == 0)
                continue;

            long long i_point_start = (long long)SCI[i_sub];
            long long i_point_end = (i_sub == n_sub - 1) ? n_point_pad : (long long)SCI[i_sub + 1];

            for (long long i_point = i_point_start; i_point < i_point_end; i_point += VEC_SIZE) // Points loop
            {
                // Load point coordinate
                __m256 rx = _mm256_loadu_ps(&Px[i_point]);
                __m256 ry = _mm256_loadu_ps(&Py[i_point]);
                __m256 rz = _mm256_loadu_ps(&Pz[i_point]);

                // Distance between vertex origin and wavefront at point
                __m256 v = _mm256_fmsub_ps(rz, nz, oz0_x_nz);
                __m256 d0 = _mm256_mul_ps(rdx, v);
                v = _mm256_fmsub_ps(ry, ny, oy0_x_ny);
                d0 = _mm256_fmadd_ps(rdx, v, d0);
                v = _mm256_fmsub_ps(rx, nx, ox0_x_nx);
                d0 = _mm256_fmadd_ps(rdx, v, d0);

                // Vertex position at advanced wavefront
                __m256 Vx = _mm256_fmadd_ps(d0, dx0, ox0);
                __m256 Vy = _mm256_fmadd_ps(d0, dy0, oy0);
                __m256 Vz = _mm256_fmadd_ps(d0, dz0, oz0);

                // Calculate edge from W1 to W2
                v = _mm256_fmsub_ps(rz, nz, oz1_x_nz);
                __m256 d1 = _mm256_mul_ps(rdy, v);
                v = _mm256_fmsub_ps(ry, ny, oy1_x_ny);
                d1 = _mm256_fmadd_ps(rdy, v, d1);
                v = _mm256_fmsub_ps(rx, nx, ox1_x_nx);
                d1 = _mm256_fmadd_ps(rdy, v, d1);

                __m256 e1x = _mm256_fmadd_ps(d1, dx1, ox1);
                __m256 e1y = _mm256_fmadd_ps(d1, dy1, oy1);
                __m256 e1z = _mm256_fmadd_ps(d1, dz1, oz1);

                e1x = _mm256_sub_ps(e1x, Vx);
                e1y = _mm256_sub_ps(e1y, Vy);
                e1z = _mm256_sub_ps(e1z, Vz);

                // Calculate edge from W1 to W3
                v = _mm256_fmsub_ps(rz, nz, oz2_x_nz);
                __m256 d2 = _mm256_mul_ps(rdz, v);
                v = _mm256_fmsub_ps(ry, ny, oy2_x_ny);
                d2 = _mm256_fmadd_ps(rdz, v, d2);
                v = _mm256_fmsub_ps(rx, nx, ox2_x_nx);
                d2 = _mm256_fmadd_ps(rdz, v, d2);

                __m256 e2x = _mm256_fmadd_ps(d2, dx2, ox2);
                __m256 e2y = _mm256_fmadd_ps(d2, dy2, oy2);
                __m256 e2z = _mm256_fmadd_ps(d2, dz2, oz2);

                e2x = _mm256_sub_ps(e2x, Vx);
                e2y = _mm256_sub_ps(e2y, Vy);
                e2z = _mm256_sub_ps(e2z, Vz);

                // Calculate vector from V to R
                __m256 tx = _mm256_sub_ps(rx, Vx);
                __m256 ty = _mm256_sub_ps(ry, Vy);
                __m256 tz = _mm256_sub_ps(rz, Vz);

                // Calculate 1st barycentric coordinate
                __m256 PQ = _mm256_mul_ps(e2y, nz); // PQ = e2y * nz
                PQ = _mm256_fmsub_ps(e2z, ny, PQ);  // PQ = e2z * ny - e2y * nz
                __m256 DT = _mm256_mul_ps(e1x, PQ); // DT = e1x * PQ
                __m256 U = _mm256_mul_ps(tx, PQ);   // U = tx * PQ

                PQ = _mm256_mul_ps(e2z, nx);       // PQ = e2z * nx
                PQ = _mm256_fmsub_ps(e2x, nz, PQ); // PQ = e2x * nz - e2z * nx
                DT = _mm256_fmadd_ps(e1y, PQ, DT); // DT = e1y * PQ + DT
                U = _mm256_fmadd_ps(ty, PQ, U);    // U = ty * PQ + U

                PQ = _mm256_mul_ps(e2x, ny);       // PQ = e2x * ny
                PQ = _mm256_fmsub_ps(e2y, nx, PQ); // PQ = e2y * nx - e2x * ny
                DT = _mm256_fmadd_ps(e1z, PQ, DT); // DT = e1z * PQ + DT
                U = _mm256_fmadd_ps(tz, PQ, U);    // U = tz * PQ + U

                // Calculate 2nd barycentric coordinate
                PQ = _mm256_mul_ps(e1y, tz);       // PQ = e1y * tz
                PQ = _mm256_fmsub_ps(e1z, ty, PQ); // PQ = e1z * ty - e1y * tz
                __m256 V = _mm256_mul_ps(nx, PQ);  // V = nx * PQ

                PQ = _mm256_mul_ps(e1z, tx);       // PQ = e1y * tx
                PQ = _mm256_fmsub_ps(e1x, tz, PQ); // PQ = e1x * tz - e1z * tx
                V = _mm256_fmadd_ps(ny, PQ, V);    // V = ny * PQ + V

                PQ = _mm256_mul_ps(e1x, ty);       // PQ = e1x * ty
                PQ = _mm256_fmsub_ps(e1y, tx, PQ); // PQ = e1y * tx - e1x * ty
                V = _mm256_fmadd_ps(nz, PQ, V);    // V = nz * PQ + V

                // Inverse of DT
                DT = _mm256_div_ps(r1, DT);
                U = _mm256_mul_ps(U, DT);
                V = _mm256_mul_ps(V, DT);

                // Check intersect conditions
                __m256 C = _mm256_cmp_ps(U, r0, _CMP_GE_OQ); // U >= 0
                __m256 D = _mm256_cmp_ps(V, r0, _CMP_GE_OQ); // V >= 0
                C = _mm256_and_ps(C, D);                     // U >= 0 & V >= 0
                U = _mm256_add_ps(U, V);                     // Compute U + V
                D = _mm256_cmp_ps(U, r1, _CMP_LE_OQ);        // (U + V) <= 1
                C = _mm256_and_ps(C, D);                     // U >= 0 & V >= 0 & (U + V) <= 1
                D = _mm256_cmp_ps(d0, r0, _CMP_GE_OQ);       // d0 >= 0
                C = _mm256_and_ps(C, D);
                D = _mm256_cmp_ps(d1, r0, _CMP_GE_OQ);       // d1 >= 0
                C = _mm256_and_ps(C, D);
                D = _mm256_cmp_ps(d2, r0, _CMP_GE_OQ);       // d2 >= 0
                C = _mm256_and_ps(C, D);                     // U >= 0 & V >= 0 & (U + V) <= 1 & d0 >= 0 & d1 >= 0 & d2 >= 0

                // Add hits to the staging buffer, most iterations have none
                int mask = _mm256_movemask_ps(C);
                while (mask != 0)
                {
                    const long long i = (long long)qd_first_set_bit(mask);
                    mask &= mask - 1;

                    if (i_point + i >= n_point_l) // Lane belongs to the padding
                        continue;

                    if (p_write == p_stop) // Current chunk is full
                    {
                        chunks.push_back(std::vector<unsigned long long>(QD_RPI_CHUNK));
                        p_write = chunks.back().data();
                        p_stop = p_write + QD_RPI_CHUNK;
                    }
                    *p_write++ = ray_bits | ((unsigned long long)(i_point + i) << 32);
                }
            }
        }

        } // end ray loop

        // Trim the last chunk to the number of records actually written
        chunks.back().resize((size_t)(p_write - chunks.back().data()));
    } // end omp parallel

    // Collect the chunks in thread order, which preserves the ray order
    std::vector<const unsigned long long *> p_buffer;
    std::vector<size_t> n_hit_buffer;

    for (size_t i_thread = 0; i_thread < (size_t)n_threads; ++i_thread)
        for (size_t i_chunk = 0; i_chunk < stage[i_thread].size(); ++i_chunk)
            if (!stage[i_thread][i_chunk].empty())
                p_buffer.push_back(stage[i_thread][i_chunk].data()),
                    n_hit_buffer.push_back(stage[i_thread][i_chunk].size());

    // Assemble the point-major CSR hit list
    qd_RPI_assemble(p_buffer.data(), n_hit_buffer.data(), p_buffer.size(), n_point, hit_index, hit_offset);
}