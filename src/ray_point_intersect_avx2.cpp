// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <math.h>
#include <stdexcept>
#include <cstdint>
#include <immintrin.h>

#include "quadriga_lib_avx2_functions.hpp"

#if defined(_MSC_VER) // Windows
#include <intrin.h>
#include <malloc.h> // Include for _aligned_malloc and _aligned_free
#endif

// Vector size for AVX2
#define VEC_SIZE 8

// AVX2 accelerated implementation of RayPointIntersect
void qd_RPI_AVX2(const float *Px, const float *Py, const float *Pz,    // Point coordinates, aligned to 32 byte, length n_point
                 const size_t n_point,                                 // Number of points
                 const unsigned *SCI,                                  // List of sub-cloud indices, length n_sub
                 const float *Xmin, const float *Xmax,                 // Minimum and maximum x-values of the AABB, aligned to 32 byte, length n_sub_s
                 const float *Ymin, const float *Ymax,                 // Minimum and maximum y-values of the AABB, aligned to 32 byte, length n_sub_s
                 const float *Zmin, const float *Zmax,                 // Minimum and maximum z-values of the AABB, aligned to 32 byte, length n_sub_s
                 const size_t n_sub,                                   // Number of sub-clouds (not aligned, i.e. n_sub <= n_sub_s)
                 const float *T1x, const float *T1y, const float *T1z, // First ray vertex coordinate in GCS, length n_ray
                 const float *T2x, const float *T2y, const float *T2z, // Second ray vertex coordinate in GCS, length n_ray
                 const float *T3x, const float *T3y, const float *T3z, // Third ray vertex coordinate in GCS, length n_ray
                 const float *Nx, const float *Ny, const float *Nz,    // Ray tube normal vector, length n_ray
                 const float *D1x, const float *D1y, const float *D1z, // First ray direction in GCS, length n_ray
                 const float *D2x, const float *D2y, const float *D2z, // Second ray direction in GCS, length n_ray
                 const float *D3x, const float *D3y, const float *D3z, // Third ray direction in GCS, length n_ray
                 const float *rD1, const float *rD2, const float *rD3, // Inverse Dot product of ray direction and normal vector
                 const size_t n_ray,                                   // Number of rays
                 std::vector<unsigned> *p_hit)                         // Output: Array of std::vector containing list of points that were hit by a ray, length n_ray

{
    if (n_point % VEC_SIZE != 0) // Check alignment
        throw std::invalid_argument("Number of points must be a multiple of 8.");
    if (n_point >= INT32_MAX)
        throw std::invalid_argument("Number of points exceeds maximum supported number.");
    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    // Constant values needed for some operations
    const size_t n_sub_s = (n_sub % VEC_SIZE == 0) ? n_sub : VEC_SIZE * (n_sub / VEC_SIZE + 1);
    const int n_point_i = (int)n_point;             // Number of points as int
    const int n_ray_i = (int)n_ray;                 // Number of rays as int
    const __m256 r0 = _mm256_set1_ps(0.0f);         // Zero (float8)
    const __m256 r1 = _mm256_set1_ps(1.0f);         // One (float8)
    const __m256 r_slack = _mm256_set1_ps(1.0e-5f); // Small value for numeric stability

#pragma omp parallel for
    for (int i_ray = 0; i_ray < n_ray_i; ++i_ray) // Ray loop
    {
        // Initialize indicator for sub-cloud hits
#if defined(_MSC_VER) // Windows
        int *p_sub_hit = (int *)_aligned_malloc(n_sub_s * sizeof(int), 32);
#else // Linux
        int *p_sub_hit = (int *)aligned_alloc(32, n_sub_s * sizeof(int));
#endif

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
            __m256 b0_low = _mm256_load_ps(&Xmin[i_sub]);
            __m256 b0_high = _mm256_load_ps(&Xmax[i_sub]);
            __m256 b1_low = _mm256_load_ps(&Ymin[i_sub]);
            __m256 b1_high = _mm256_load_ps(&Ymax[i_sub]);
            __m256 b2_low = _mm256_load_ps(&Zmin[i_sub]);
            __m256 b2_high = _mm256_load_ps(&Zmax[i_sub]);

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

            int i_point_start = (int)SCI[i_sub];
            int i_point_end = (i_sub == n_sub - 1) ? n_point_i : (int)SCI[i_sub + 1];

            for (int i_point = i_point_start; i_point < i_point_end; i_point += VEC_SIZE) // Points loop
            {
                // Load point coordinate
                __m256 rx = _mm256_load_ps(&Px[i_point]);
                __m256 ry = _mm256_load_ps(&Py[i_point]);
                __m256 rz = _mm256_load_ps(&Pz[i_point]);

                // Distance between vertex origin and wavefront at point
                __m256 v = _mm256_fmsub_ps(rz, nz, oz0_x_nz);
                __m256 d = _mm256_mul_ps(rdx, v);
                v = _mm256_fmsub_ps(ry, ny, oy0_x_ny);
                d = _mm256_fmadd_ps(rdx, v, d);
                v = _mm256_fmsub_ps(rx, nx, ox0_x_nx);
                d = _mm256_fmadd_ps(rdx, v, d);

                // Vertex position at advanced wavefront
                __m256 Vx = _mm256_fmadd_ps(d, dx0, ox0);
                __m256 Vy = _mm256_fmadd_ps(d, dy0, oy0);
                __m256 Vz = _mm256_fmadd_ps(d, dz0, oz0);

                // Calculate edge from W1 to W2
                v = _mm256_fmsub_ps(rz, nz, oz1_x_nz);
                d = _mm256_mul_ps(rdy, v);
                v = _mm256_fmsub_ps(ry, ny, oy1_x_ny);
                d = _mm256_fmadd_ps(rdy, v, d);
                v = _mm256_fmsub_ps(rx, nx, ox1_x_nx);
                d = _mm256_fmadd_ps(rdy, v, d);

                __m256 e1x = _mm256_fmadd_ps(d, dx1, ox1);
                __m256 e1y = _mm256_fmadd_ps(d, dy1, oy1);
                __m256 e1z = _mm256_fmadd_ps(d, dz1, oz1);

                e1x = _mm256_sub_ps(e1x, Vx);
                e1y = _mm256_sub_ps(e1y, Vy);
                e1z = _mm256_sub_ps(e1z, Vz);

                // Calculate edge from W1 to W3
                v = _mm256_fmsub_ps(rz, nz, oz2_x_nz);
                d = _mm256_mul_ps(rdz, v);
                v = _mm256_fmsub_ps(ry, ny, oy2_x_ny);
                d = _mm256_fmadd_ps(rdz, v, d);
                v = _mm256_fmsub_ps(rx, nx, ox2_x_nx);
                d = _mm256_fmadd_ps(rdz, v, d);

                __m256 e2x = _mm256_fmadd_ps(d, dx2, ox2);
                __m256 e2y = _mm256_fmadd_ps(d, dy2, oy2);
                __m256 e2z = _mm256_fmadd_ps(d, dz2, oz2);

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
                D = _mm256_cmp_ps(d, r0, _CMP_GE_OQ);        // d >= 0
                C = _mm256_and_ps(C, D);                     // U >= 0 & V >= 0 & (U + V) <= 1 & d > 0

                // Add point to points list
                int result[8];
                __m256i final_result_int = _mm256_castps_si256(C);
                _mm256_storeu_si256((__m256i *)result, final_result_int);
                for (int i = 0; i < 8; ++i)
                    if (result[i] != 0)
                        p_hit[i_ray].push_back(i_point + i);
            }
        }

// Free aligned memory
#if defined(_MSC_VER) // Windows
        _aligned_free(p_sub_hit);
#else // Linux
        free(p_sub_hit);
#endif
    }
}