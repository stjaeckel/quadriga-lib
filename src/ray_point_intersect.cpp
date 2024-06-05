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

#include <immintrin.h>
#include <cstring> // For std::memcopy
#include <cmath>   // For std::isnan
#include "quadriga_tools.hpp"

// Vector size for AVX2
#define VEC_SIZE 8

// Testing for AVX2 support at runtime
#if defined(_MSC_VER) // Windows
#include <intrin.h>
#include <malloc.h> // Include for _aligned_malloc and _aligned_free
#else               // Linux
#include <cpuid.h>
#endif

#if defined(__AVX2__) // Compiler support for AVX2
static bool isAVX2Supported()
{
    std::vector<int> cpuidInfo(4);

#if defined(_MSC_VER) // Windows
    __cpuidex(cpuidInfo.data(), 7, 0);
#else // Linux
    __cpuid_count(7, 0, cpuidInfo[0], cpuidInfo[1], cpuidInfo[2], cpuidInfo[3]);
#endif

    return (cpuidInfo[1] & (1 << 5)) != 0; // Check the AVX2 bit in EBX
}
#endif

// AVX2 accelerated implementation of RayPointIntersect
static inline void qd_RPI_AVX2(const float *Px, const float *Py, const float *Pz,    // Point coordinates, aligned to 32 byte, length n_point
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

// Generic C++ implementation of RayPointIntersect
static inline void qd_RPI_GENERIC(const float *Px, const float *Py, const float *Pz,    // Point coordinates, aligned to 32 byte, length n_point
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
    if (n_point >= INT32_MAX)
        throw std::invalid_argument("Number of points exceeds maximum supported number.");
    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    // Constant values needed for some operations
    const int n_point_i = (int)n_point; // Number of points as int
    const int n_ray_i = (int)n_ray;     // Number of rays as int

#pragma omp parallel for
    for (int i_ray = 0; i_ray < n_ray_i; ++i_ray) // Ray loop
    {
        // Initialize indicator for sub-cloud hits
        arma::s32_vec sub_hit(n_sub);
        int *p_sub_hit = sub_hit.memptr();

        // Load origin
        float ox[3], oy[3], oz[3];
        ox[0] = T1x[i_ray], oy[0] = T1y[i_ray], oz[0] = T1z[i_ray];
        ox[1] = T2x[i_ray], oy[1] = T2y[i_ray], oz[1] = T2z[i_ray];
        ox[2] = T3x[i_ray], oy[2] = T3y[i_ray], oz[2] = T3z[i_ray];

        // Load direction
        float dx[3], dy[3], dz[3];
        dx[0] = D1x[i_ray], dy[0] = D1y[i_ray], dz[0] = D1z[i_ray];
        dx[1] = D2x[i_ray], dy[1] = D2y[i_ray], dz[1] = D2z[i_ray];
        dx[2] = D3x[i_ray], dy[2] = D3y[i_ray], dz[2] = D3z[i_ray];

        // Load normal vector
        float nx = Nx[i_ray], ny = Ny[i_ray], nz = Nz[i_ray];

        // Load inverse dot product
        float rdx = rD1[i_ray], rdy = rD2[i_ray], rdz = rD3[i_ray];

        // Step 1 - Check for possible hits
        // - Move the wavefront forward relative to the distance between vertex origin and AABB corner point
        // - Construct second AABB from advanced wavefronts
        // - If AABBs overlap, there is a potential hit and individual points must be checked in step 2

        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Load point bounding box, add some slack for numeric stability
            float b0_low = Xmin[i_sub] - 1.0e-5f, b0_high = Xmax[i_sub] + 1.0e-5f,
                  b1_low = Ymin[i_sub] - 1.0e-5f, b1_high = Ymax[i_sub] + 1.0e-5f,
                  b2_low = Zmin[i_sub] - 1.0e-5f, b2_high = Zmax[i_sub] + 1.0e-5f;

            // AABB corner points
            float rx[8] = {b0_low, b0_low, b0_low, b0_low, b0_high, b0_high, b0_high, b0_high};
            float ry[8] = {b1_low, b1_low, b1_high, b1_high, b1_low, b1_low, b1_high, b1_high};
            float rz[8] = {b2_low, b2_high, b2_low, b2_high, b2_low, b2_high, b2_low, b2_high};

            // Initialize coordinates for the vertex box
            float a0_low = INFINITY, a0_high = -INFINITY,
                  a1_low = INFINITY, a1_high = -INFINITY,
                  a2_low = INFINITY, a2_high = -INFINITY;

            // Calculate the vertex box at the advanced wavefront
            for (int i = 0; i < 8; ++i)
            {
                // Distance between vertex origin and wavefront at corner point
                float d = rdx * ((rx[i] - ox[0]) * nx + (ry[i] - oy[0]) * ny + (rz[i] - oz[0]) * nz);

                // Update vertex box at advanced wavefront
                float V = d * dx[0] + ox[0];
                a0_low = (V < a0_low) ? V : a0_low;
                a0_high = (V > a0_high) ? V : a0_high;

                V = d * dy[0] + oy[0];
                a1_low = (V < a1_low) ? V : a1_low;
                a1_high = (V > a1_high) ? V : a1_high;

                V = d * dz[0] + oz[0];
                a2_low = (V < a2_low) ? V : a2_low;
                a2_high = (V > a2_high) ? V : a2_high;

                // 2nd vertex
                d = rdy * ((rx[i] - ox[1]) * nx + (ry[i] - oy[1]) * ny + (rz[i] - oz[1]) * nz);

                V = d * dx[1] + ox[1];
                a0_low = (V < a0_low) ? V : a0_low;
                a0_high = (V > a0_high) ? V : a0_high;

                V = d * dy[1] + oy[1];
                a1_low = (V < a1_low) ? V : a1_low;
                a1_high = (V > a1_high) ? V : a1_high;

                V = d * dz[1] + oz[1];
                a2_low = (V < a2_low) ? V : a2_low;
                a2_high = (V > a2_high) ? V : a2_high;

                // 3rd vertex
                d = rdz * ((rx[i] - ox[2]) * nx + (ry[i] - oy[2]) * ny + (rz[i] - oz[2]) * nz);

                V = d * dx[2] + ox[2];
                a0_low = (V < a0_low) ? V : a0_low;
                a0_high = (V > a0_high) ? V : a0_high;

                V = d * dy[2] + oy[2];
                a1_low = (V < a1_low) ? V : a1_low;
                a1_high = (V > a1_high) ? V : a1_high;

                V = d * dz[2] + oz[2];
                a2_low = (V < a2_low) ? V : a2_low;
                a2_high = (V > a2_high) ? V : a2_high;
            }

            // Check for a potential overlap between the AABBs
            if (a0_high >= b0_low && a0_low <= b0_high &&
                a1_high >= b1_low && a1_low <= b1_high &&
                a2_high >= b2_low && a2_low <= b2_high)
                p_sub_hit[i_sub] = 1;
        }

        // Step 2 - Check intersection with points within the sub-clouds

        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Skip if sub-cloud was not hit
            if (p_sub_hit[i_sub] == 0)
                continue;

            int i_point_start = (int)SCI[i_sub];
            int i_point_end = (i_sub == n_sub - 1) ? n_point_i : (int)SCI[i_sub + 1];

            for (int i_point = i_point_start; i_point < i_point_end; ++i_point) // Point loop
            {
                // Load point coordinate
                float rx = Px[i_point];
                float ry = Py[i_point];
                float rz = Pz[i_point];

                // Distance between vertex origin and wavefront at point
                float d = rdx * ((rx - ox[0]) * nx + (ry - oy[0]) * ny + (rz - oz[0]) * nz);

                // Vertex position at advanced wavefront
                float Vx = d * dx[0] + ox[0],
                      Vy = d * dy[0] + oy[0],
                      Vz = d * dz[0] + oz[0];

                // Calculate edge from W1 to W2
                d = rdy * ((rx - ox[1]) * nx + (ry - oy[1]) * ny + (rz - oz[1]) * nz);

                float e1x = d * dx[1] + ox[1] - Vx,
                      e1y = d * dy[1] + oy[1] - Vy,
                      e1z = d * dz[1] + oz[1] - Vz;

                // Calculate edge from W1 to W3
                d = rdz * ((rx - ox[2]) * nx + (ry - oy[2]) * ny + (rz - oz[2]) * nz);

                float e2x = d * dx[2] + ox[2] - Vx,
                      e2y = d * dy[2] + oy[2] - Vy,
                      e2z = d * dz[2] + oz[2] - Vz;

                // Calculate vector from V to R
                float tx = rx - Vx,
                      ty = ry - Vy,
                      tz = rz - Vz;

                // Calculate 1st barycentric coordinate
                float PQ = e2z * ny - e2y * nz;
                float DT = e1x * PQ;
                float U = tx * PQ;

                PQ = e2x * nz - e2z * nx;
                DT = e1y * PQ + DT;
                U = ty * PQ + U;

                PQ = e2y * nx - e2x * ny;
                DT = e1z * PQ + DT;
                U = tz * PQ + U;

                // Calculate 2nd barycentric coordinate
                PQ = e1z * ty - e1y * tz;
                float V = nx * PQ;

                PQ = e1x * tz - e1z * tx,
                V = ny * PQ + V;

                PQ = e1y * tx - e1x * ty,
                V = nz * PQ + V;

                // Inverse of DT
                DT = 1.0f / DT;

                U = U * DT;
                V = V * DT;

                // Check intersect conditions
                bool C1 = (U >= 0.0f) & (V >= 0.0f) & ((U + V) <= 1.0f) & (d >= 0.0f);

                // Add point to points list
                if (C1)
                    p_hit[i_ray].push_back(i_point);
            }
        }
    }
}

template <typename dtype>
std::vector<arma::Col<unsigned>> quadriga_lib::ray_point_intersect(const arma::Mat<dtype> *points,
                                                                   const arma::Mat<dtype> *orig,
                                                                   const arma::Mat<dtype> *trivec,
                                                                   const arma::Mat<dtype> *tridir,
                                                                   const arma::Col<unsigned> *sub_cloud_index,
                                                                   arma::Col<unsigned> *hit_count)
{
    // Input validation
    if (points == nullptr || points->n_elem == 0)
        throw std::invalid_argument("Input 'points' cannot be NULL.");
    if (orig == nullptr || orig->n_elem == 0)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (trivec == nullptr)
        throw std::invalid_argument("Input 'trivec' cannot be NULL.");
    if (tridir == nullptr)
        throw std::invalid_argument("Input 'tridir' cannot be NULL.");

    if (points->n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns containing x,y,z coordinates.");
    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    if (trivec->n_cols != 9)
        throw std::invalid_argument("Input 'trivec' must have 9 columns containing x,y,z coordinates of ray tube vertices.");
    if (tridir->n_cols != 9)
        throw std::invalid_argument("Input 'tridir' must have 9 columns containing ray directions in Cartesian format.");

    size_t n_ray_t = orig->n_rows;
    size_t n_point_t = (size_t)points->n_rows;
    int n_ray_i = (int)n_ray_t;

    // Bound chek
    if (n_point_t >= INT32_MAX)
        throw std::invalid_argument("Number of points exceeds maximum supported number.");
    if (n_ray_t >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    if (trivec->n_rows != n_ray_t)
        throw std::invalid_argument("Number of rows in 'orig' and 'trivec' dont match.");
    if (tridir->n_rows != n_ray_t)
        throw std::invalid_argument("Number of rows in 'orig' and 'tridir' dont match.");

    // Check if the sub-cloud indices are valid
    size_t n_sub_t = 1;                                             // Number of sub-clouds (at least 1)
    arma::Col<unsigned> sci(1);                                     // Sub-cloud-index (local copy)
    if (sub_cloud_index != nullptr && sub_cloud_index->n_elem != 0) // Input is available
    {
        n_sub_t = (size_t)sub_cloud_index->n_elem;
        const unsigned *p_sub = sub_cloud_index->memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-cloud must start at index 0.");

        for (size_t i = 1; i < n_sub_t; ++i)
        {
            if (p_sub[i] <= p_sub[i - 1])
                throw std::invalid_argument("Sub-cloud indices must be sorted in ascending order.");

            if (p_sub[i] % VEC_SIZE != 0)
                throw std::invalid_argument("Sub-clouds must be aligned with the SIMD vector size (8 for AVX2, 32 for CUDA).");
        }

        if (p_sub[n_sub_t - 1] >= (unsigned)n_point_t)
            throw std::invalid_argument("Sub-cloud indices cannot exceed number of points.");

        sci = *sub_cloud_index;
    }

    // Prepare ray data
    auto trivecA = arma::fmat(n_ray_t, 9, arma::fill::none); // Vertex origins in GCS
    auto normalA = arma::fmat(n_ray_t, 3, arma::fill::none); // Normal vector
    auto dirA = arma::fmat(n_ray_t, 9, arma::fill::none);    // Vertex directions (Cartesian)
    auto invDotA = arma::fmat(n_ray_t, 3, arma::fill::none); // Inverse dot product
    {
        const dtype *p_orig = orig->memptr();     // Origin pointer
        const dtype *p_trivec = trivec->memptr(); // Trivec pointer
        const dtype *p_tridir = tridir->memptr(); // Direction pointer

        float *p_trivecA = trivecA.memptr();
        float *p_normalA = normalA.memptr();
        float *p_dirA = dirA.memptr();
        float *p_invDotA = invDotA.memptr();

#pragma omp parallel for
        for (int i_ray = 0; i_ray < n_ray_i; ++i_ray)
        {
            // Load origin
            dtype Nx = p_orig[i_ray],
                  Ny = p_orig[i_ray + n_ray_i],
                  Nz = p_orig[i_ray + 2 * n_ray_i];

            // Load first vertex
            dtype Vx = p_trivec[i_ray],
                  Vy = p_trivec[i_ray + n_ray_i],
                  Vz = p_trivec[i_ray + 2 * n_ray_i];

            // Calculate first vertex location in GCS
            p_trivecA[i_ray] = float(Nx + Vx);
            p_trivecA[i_ray + n_ray_i] = float(Ny + Vy);
            p_trivecA[i_ray + 2 * n_ray_i] = float(Nz + Vz);

            // Load second vertex
            dtype Ux = p_trivec[i_ray + 3 * n_ray_i],
                  Uy = p_trivec[i_ray + 4 * n_ray_i],
                  Uz = p_trivec[i_ray + 5 * n_ray_i];

            // Calculate second vertex location in GCS
            p_trivecA[i_ray + 3 * n_ray_i] = float(Nx + Ux);
            p_trivecA[i_ray + 4 * n_ray_i] = float(Ny + Uy);
            p_trivecA[i_ray + 5 * n_ray_i] = float(Nz + Uz);

            // Calculate edge from first to second vertex
            Ux -= Vx, Uy -= Vy, Uz -= Vz;

            // Process third vertex
            dtype tmp = p_trivec[i_ray + 6 * n_ray_i];
            p_trivecA[i_ray + 6 * n_ray_i] = float(Nx + tmp);
            Vx = tmp - Vx;

            tmp = p_trivec[i_ray + 7 * n_ray_i];
            p_trivecA[i_ray + 7 * n_ray_i] = float(Ny + tmp);
            Vy = tmp - Vy;

            tmp = p_trivec[i_ray + 8 * n_ray_i];
            p_trivecA[i_ray + 8 * n_ray_i] = float(Nz + tmp);
            Vz = tmp - Vz;

            // Calculate Normal Vector
            Nx = Uy * Vz - Uz * Vy;
            Ny = Uz * Vx - Ux * Vz;
            Nz = Ux * Vy - Uy * Vx;

            // Convert to float
            p_normalA[i_ray] = float(Nx);
            p_normalA[i_ray + n_ray_i] = float(Ny);
            p_normalA[i_ray + 2 * n_ray_i] = float(Nz);

            // Load first vertex direction
            Vx = p_tridir[i_ray];
            Vy = p_tridir[i_ray + n_ray_i];
            Vz = p_tridir[i_ray + 2 * n_ray_i];

            // Normalize it, if needed
            tmp = Vx * Vx + Vy * Vy + Vz * Vz;
            if (std::abs(tmp - (dtype)1.0) > (dtype)2e-7)
                tmp = std::sqrt((dtype)1.0 / tmp), Vx *= tmp, Vy *= tmp, Vz *= tmp;

            // Store as float
            p_dirA[i_ray] = float(Vx);
            p_dirA[i_ray + n_ray_i] = float(Vy);
            p_dirA[i_ray + 2 * n_ray_i] = float(Vz);

            // Calculate inverse DotProduct from Normal Vector and Vertex direction
            tmp = Vx * Nx + Vy * Ny + Vz * Nz;
            p_invDotA[i_ray] = float((dtype)1.0 / tmp);

            // Load second vertex direction
            Vx = p_tridir[i_ray + 3 * n_ray_i];
            Vy = p_tridir[i_ray + 4 * n_ray_i];
            Vz = p_tridir[i_ray + 5 * n_ray_i];

            // Normalize it, if needed
            tmp = Vx * Vx + Vy * Vy + Vz * Vz;
            if (std::abs(tmp - (dtype)1.0) > (dtype)2e-7)
                tmp = std::sqrt((dtype)1.0 / tmp), Vx *= tmp, Vy *= tmp, Vz *= tmp;

            // Store as float
            p_dirA[i_ray + 3 * n_ray_i] = float(Vx);
            p_dirA[i_ray + 4 * n_ray_i] = float(Vy);
            p_dirA[i_ray + 5 * n_ray_i] = float(Vz);

            // Calculate inverse DotProduct from Normal Vector and Vertex direction
            tmp = Vx * Nx + Vy * Ny + Vz * Nz;
            p_invDotA[i_ray + n_ray_i] = float((dtype)1.0 / tmp);

            // Load third vertex direction
            Vx = p_tridir[i_ray + 6 * n_ray_i];
            Vy = p_tridir[i_ray + 7 * n_ray_i];
            Vz = p_tridir[i_ray + 8 * n_ray_i];

            // Normalize it, if needed
            tmp = Vx * Vx + Vy * Vy + Vz * Vz;
            if (std::abs(tmp - (dtype)1.0) > (dtype)2e-7)
                tmp = std::sqrt((dtype)1.0 / tmp), Vx *= tmp, Vy *= tmp, Vz *= tmp;

            // Store as float
            p_dirA[i_ray + 6 * n_ray_i] = float(Vx);
            p_dirA[i_ray + 7 * n_ray_i] = float(Vy);
            p_dirA[i_ray + 8 * n_ray_i] = float(Vz);

            // Calculate inverse DotProduct from Normal Vector and Vertex direction
            tmp = Vx * Nx + Vy * Ny + Vz * Nz;
            p_invDotA[i_ray + 2 * n_ray_i] = float((dtype)1.0 / tmp);
        }
    }

    // Alignment to 32 byte addresses is required when loading data into AVX2 registers
    // Not doing this may cause segmentation faults (e.g. in MATLAB)
    size_t n_point_s = (n_point_t % VEC_SIZE == 0) ? n_point_t : VEC_SIZE * (n_point_t / VEC_SIZE + 1);
    size_t n_sub_s = (n_sub_t % VEC_SIZE == 0) ? n_sub_t : VEC_SIZE * (n_sub_t / VEC_SIZE + 1);

#if defined(_MSC_VER) // Windows
    float *Px = (float *)_aligned_malloc(n_point_s * sizeof(float), 32);
    float *Py = (float *)_aligned_malloc(n_point_s * sizeof(float), 32);
    float *Pz = (float *)_aligned_malloc(n_point_s * sizeof(float), 32);
    float *Xmin = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Xmax = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Ymin = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Ymax = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Zmin = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Zmax = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
#else // Linux
    float *Px = (float *)aligned_alloc(32, n_point_s * sizeof(float));
    float *Py = (float *)aligned_alloc(32, n_point_s * sizeof(float));
    float *Pz = (float *)aligned_alloc(32, n_point_s * sizeof(float));
    float *Xmin = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Xmax = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Ymin = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Ymax = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Zmin = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Zmax = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
#endif

    // Convert points to float and write to aligned memory
    // Calculate bounding box for each sub-cloud
    const dtype *p_points = points->memptr();
    const unsigned *p_sub = sci.memptr();

    // Set parameters for the first AABB
    size_t i_sub = 0, i_next = (n_sub_t == 1) ? n_point_t - 1 : (size_t)p_sub[1] - 1;
    float x_min = INFINITY, x_max = -INFINITY,
          y_min = INFINITY, y_max = -INFINITY,
          z_min = INFINITY, z_max = -INFINITY;

    for (size_t i_point = 0; i_point < n_point_t; ++i_point)
    {
        // Load point
        dtype x = p_points[i_point],
              y = p_points[i_point + n_point_t],
              z = p_points[i_point + 2 * n_point_t];

        // Typecast to float and update AABB
        float xf = (float)x, yf = (float)y, zf = (float)z;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Write to aligned memory
        Px[i_point] = xf, Py[i_point] = yf, Pz[i_point] = zf;

        // Update sub-cloud data for the next AABB
        if (i_point == i_next)
        {
            // Write current AABB data to aligned memory
            Xmin[i_sub] = x_min, Xmax[i_sub] = x_max;
            Ymin[i_sub] = y_min, Ymax[i_sub] = y_max;
            Zmin[i_sub] = z_min, Zmax[i_sub] = z_max;

            // Reset registers
            x_min = INFINITY, x_max = -INFINITY,
            y_min = INFINITY, y_max = -INFINITY,
            z_min = INFINITY, z_max = -INFINITY;

            // Update counters
            ++i_sub;
            i_next = (i_sub == n_sub_t - 1) ? n_point_t - 1 : (size_t)p_sub[i_sub + 1] - 1;
        }
    }

    // Add padding to the aligned point data
    for (size_t i_point = n_point_t; i_point < n_point_s; ++i_point)
        Px[i_point] = 0.0f, Py[i_point] = 0.0f, Pz[i_point] = 0.0f;

    // Add padding to the aligned AABB data
    for (size_t i_sub = n_sub_t; i_sub < n_sub_s; ++i_sub)
    {
        Xmin[i_sub] = 0.0f, Xmax[i_sub] = 0.0f;
        Ymin[i_sub] = 0.0f, Ymax[i_sub] = 0.0f;
        Zmin[i_sub] = 0.0f, Zmax[i_sub] = 0.0f;
    }

    // Output container
    std::vector<unsigned> *p_hit = new std::vector<unsigned>[n_ray_t];

    // Call intersect function
#if defined(__AVX2__)      // Compiler support for AVX2
    if (isAVX2Supported()) // CPU support for AVX2
    {
        qd_RPI_AVX2(Px, Py, Pz, n_point_s,
                    sci.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub_t,
                    trivecA.colptr(0), trivecA.colptr(1), trivecA.colptr(2),
                    trivecA.colptr(3), trivecA.colptr(4), trivecA.colptr(5),
                    trivecA.colptr(6), trivecA.colptr(7), trivecA.colptr(8),
                    normalA.colptr(0), normalA.colptr(1), normalA.colptr(2),
                    dirA.colptr(0), dirA.colptr(1), dirA.colptr(2),
                    dirA.colptr(3), dirA.colptr(4), dirA.colptr(5),
                    dirA.colptr(6), dirA.colptr(7), dirA.colptr(8),
                    invDotA.colptr(0), invDotA.colptr(1), invDotA.colptr(2),
                    n_ray_t, p_hit);
    }
    else
    {
        qd_RPI_GENERIC(Px, Py, Pz, n_point_t,
                       sci.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub_t,
                       trivecA.colptr(0), trivecA.colptr(1), trivecA.colptr(2),
                       trivecA.colptr(3), trivecA.colptr(4), trivecA.colptr(5),
                       trivecA.colptr(6), trivecA.colptr(7), trivecA.colptr(8),
                       normalA.colptr(0), normalA.colptr(1), normalA.colptr(2),
                       dirA.colptr(0), dirA.colptr(1), dirA.colptr(2),
                       dirA.colptr(3), dirA.colptr(4), dirA.colptr(5),
                       dirA.colptr(6), dirA.colptr(7), dirA.colptr(8),
                       invDotA.colptr(0), invDotA.colptr(1), invDotA.colptr(2),
                       n_ray_t, p_hit);
    }
#else
    qd_RPI_GENERIC(Px, Py, Pz, n_point_t,
                   sci.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub_t,
                   trivecA.colptr(0), trivecA.colptr(1), trivecA.colptr(2),
                   trivecA.colptr(3), trivecA.colptr(4), trivecA.colptr(5),
                   trivecA.colptr(6), trivecA.colptr(7), trivecA.colptr(8),
                   normalA.colptr(0), normalA.colptr(1), normalA.colptr(2),
                   dirA.colptr(0), dirA.colptr(1), dirA.colptr(2),
                   dirA.colptr(3), dirA.colptr(4), dirA.colptr(5),
                   dirA.colptr(6), dirA.colptr(7), dirA.colptr(8),
                   invDotA.colptr(0), invDotA.colptr(1), invDotA.colptr(2),
                   n_ray_t, p_hit);
#endif

// Free aligned memory
#if defined(_MSC_VER) // Windows
    _aligned_free(Px), _aligned_free(Py), _aligned_free(Pz);
    _aligned_free(Xmin), _aligned_free(Xmax);
    _aligned_free(Ymin), _aligned_free(Ymax);
    _aligned_free(Zmin), _aligned_free(Zmax);
#else // Linux
    free(Px), free(Py), free(Pz);
    free(Xmin), free(Xmax);
    free(Ymin), free(Ymax);
    free(Zmin), free(Zmax);
#endif

    // Count hits per point
    unsigned *p_cnt = new unsigned[n_point_s]();

    for (size_t i_ray = 0; i_ray < n_ray_t; ++i_ray)
        for (size_t i_hit = 0; i_hit < p_hit[i_ray].size(); ++i_hit)
            ++p_cnt[p_hit[i_ray][i_hit]];

    if (hit_count != nullptr)
    {
        if (hit_count->n_elem != n_point_t)
            hit_count->set_size(n_point_t);

        std::memcpy(hit_count->memptr(), p_cnt, n_point_t * sizeof(unsigned));
    }

    // Generate output
    std::vector<arma::Col<unsigned>> output(n_point_t);

    for (size_t i_point = 0; i_point < n_point_t; ++i_point)
    {
        if (p_cnt[i_point] != 0)
            output[i_point].set_size(p_cnt[i_point]);
        p_cnt[i_point] = 0;
    }

    for (size_t i_ray = 0; i_ray < n_ray_t; ++i_ray)
        for (size_t i_hit = 0; i_hit < p_hit[i_ray].size(); ++i_hit)
        {
            unsigned i_point = p_hit[i_ray][i_hit];
            if (i_point < n_point_t)
                output[i_point].at(p_cnt[i_point]++) = (unsigned)i_ray;
        }

    delete[] p_cnt;
    delete[] p_hit;
    return output;
}

template std::vector<arma::Col<unsigned>> quadriga_lib::ray_point_intersect(const arma::Mat<float> *points,
                                                                            const arma::Mat<float> *orig,
                                                                            const arma::Mat<float> *trivec,
                                                                            const arma::Mat<float> *tridir,
                                                                            const arma::Col<unsigned> *sub_cloud_index,
                                                                            arma::Col<unsigned> *hit_count);

template std::vector<arma::Col<unsigned>> quadriga_lib::ray_point_intersect(const arma::Mat<double> *points,
                                                                            const arma::Mat<double> *orig,
                                                                            const arma::Mat<double> *trivec,
                                                                            const arma::Mat<double> *tridir,
                                                                            const arma::Col<unsigned> *sub_cloud_index,
                                                                            arma::Col<unsigned> *hit_count);