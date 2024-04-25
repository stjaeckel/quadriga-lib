// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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

// Use this to print the contents of a __m256 to stdout
// { // Display the elements of a __m256
//     float *f = (float *)&W;
//     printf("W = %f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
// }
// { // Display the elements of a __m256i
//     unsigned *f = (unsigned *)&I;
//     printf("I = %d %d %d %d %d %d %d %d\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
// }

// AVX2 accelerated implementation of RayTriangleIntersect
inline void qd_RTI_AVX2(const float *Tx, const float *Ty, const float *Tz,    // First vertex coordinate in GCS, aligned to 32 byte, length n_mesh
                        const float *E1x, const float *E1y, const float *E1z, // Edge 1 from first vertex to second vertex, aligned to 32 byte, length n_mesh
                        const float *E2x, const float *E2y, const float *E2z, // Edge 2 from first vertex to third vertex, aligned to 32 byte, length n_mesh
                        const size_t n_mesh,                                  // Number of triangles (multiple of VEC_SIZE)
                        const unsigned *SMI,                                  // List of sub-mesh indices, length n_sub
                        const float *Xmin, const float *Xmax,                 // Minimum and maximum x-values of the AABB, aligned to 32 byte, length n_sub_s
                        const float *Ymin, const float *Ymax,                 // Minimum and maximum y-values of the AABB, aligned to 32 byte, length n_sub_s
                        const float *Zmin, const float *Zmax,                 // Minimum and maximum z-values of the AABB, aligned to 32 byte, length n_sub_s
                        const size_t n_sub,                                   // Number of sub-meshes (not aligned, i.e. n_sub <= n_sub_s)
                        const float *Ox, const float *Oy, const float *Oz,    // Ray origin in GCS, length n_ray
                        const float *Dx, const float *Dy, const float *Dz,    // Vector from ray origin to ray destination, length n_ray
                        const size_t n_ray,                                   // Number of rays
                        float *Wf,                                            // Normalized distance (0-1) of FBS hit, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                        float *Ws,                                            // Normalized distance (0-1) of SBS hit, must be >= Wf, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                        unsigned *If,                                         // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                        unsigned *Is,                                         // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                        unsigned *hit_cnt = nullptr)                          // Number of hits between orig and dest, length n_ray, uninitialized, optional
{
    if (n_mesh % VEC_SIZE != 0) // Check alignment
        throw std::invalid_argument("Number of triangles must be a multiple of 8.");
    if (n_mesh >= INT32_MAX)
        throw std::invalid_argument("Number of triangles exceeds maximum supported number.");
    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    bool count_hits = hit_cnt != nullptr;

    // Constant values needed for some operations
    const size_t n_sub_s = (n_sub % VEC_SIZE == 0) ? n_sub : VEC_SIZE * (n_sub / VEC_SIZE + 1);
    const int n_ray_int = (int)n_ray;                              // Number of rays as int
    const int n_mesh_int = (int)n_mesh;                            // Number of mesh elements as int
    const __m256 r0 = _mm256_set1_ps(0.0f);                        // Zero (float8)
    const __m256 r1 = _mm256_set1_ps(1.0f);                        // One (float8)
    const __m256i i0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);  // Indices 0-7
    const __m256i i0p = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6); // Permuted indices
    const __m256i iX = _mm256_set_epi32(0, 1, 2, 3, 0, 1, 2, 3);   // Used for permuting indices

#pragma omp parallel for
    for (int i_ray = 0; i_ray < n_ray_int; ++i_ray) // Ray loop
    {
        // Load origin into AVX2 registers
        __m256 ox = _mm256_set1_ps(Ox[i_ray]);
        __m256 oy = _mm256_set1_ps(Oy[i_ray]);
        __m256 oz = _mm256_set1_ps(Oz[i_ray]);

        // Load vector from ray origin to ray destination into AVX2 registers
        __m256 dx = _mm256_set1_ps(Dx[i_ray]);
        __m256 dy = _mm256_set1_ps(Dy[i_ray]);
        __m256 dz = _mm256_set1_ps(Dz[i_ray]);

        // Initialize local variables
        __m256 W_fbs = _mm256_set1_ps(1.0f); // Set FBS location equal to dest
        __m256 W_sbs = W_fbs;                // Set SBS location equal to FBS location
        unsigned I_fbs = 0, I_sbs = 0;       // Set FBS index to 0
        unsigned hit_counter = 0;            // Hit counter

        // Step 1 - Check intersection with the AABBs of the sub-meshes (slab-method)
        // See: https://en.wikipedia.org/wiki/Slab_method

        // Inverse of the direction (may be infinite if ray is parallel to an axis)
        __m256 dx_i = _mm256_div_ps(r1, dx);
        __m256 dy_i = _mm256_div_ps(r1, dy);
        __m256 dz_i = _mm256_div_ps(r1, dz);

        arma::s32_vec sub_mesh_hit(n_sub_s, arma::fill::none);
        int *p_sub_mesh_hit = sub_mesh_hit.memptr();
        for (size_t i_sub = 0; i_sub < n_sub_s; i_sub += VEC_SIZE)
        {
            // Calculate the intersections of the ray with the two planes orthogonal to the i-th coordinate axis
            __m256 T = _mm256_load_ps(&Xmin[i_sub]);
            T = _mm256_sub_ps(T, ox);
            __m256 t0_low = _mm256_mul_ps(T, dx_i);

            T = _mm256_load_ps(&Xmax[i_sub]);
            T = _mm256_sub_ps(T, ox);
            __m256 t0_high = _mm256_mul_ps(T, dx_i);

            T = _mm256_load_ps(&Ymin[i_sub]);
            T = _mm256_sub_ps(T, oy);
            __m256 t1_low = _mm256_mul_ps(T, dy_i);

            T = _mm256_load_ps(&Ymax[i_sub]);
            T = _mm256_sub_ps(T, oy);
            __m256 t1_high = _mm256_mul_ps(T, dy_i);

            T = _mm256_load_ps(&Zmin[i_sub]);
            T = _mm256_sub_ps(T, oz);
            __m256 t2_low = _mm256_mul_ps(T, dz_i);

            T = _mm256_load_ps(&Zmax[i_sub]);
            T = _mm256_sub_ps(T, oz);
            __m256 t2_high = _mm256_mul_ps(T, dz_i);

            // Calculate the close and far extrema of the segment within the i-th slab
            __m256 M = _mm256_cmp_ps(t0_low, t0_high, _CMP_GE_OQ); // t_low >= t_high ?
            T = _mm256_blendv_ps(t0_low, t0_high, M);              // t_low = min( t_low, t_high )
            t0_high = _mm256_blendv_ps(t0_high, t0_low, M);        // t_high = max( t_low, t_high )
            t0_low = T;

            M = _mm256_cmp_ps(t1_low, t1_high, _CMP_GE_OQ);
            T = _mm256_blendv_ps(t1_low, t1_high, M);
            t1_high = _mm256_blendv_ps(t1_high, t1_low, M);
            t1_low = T;

            M = _mm256_cmp_ps(t2_low, t2_high, _CMP_GE_OQ);
            T = _mm256_blendv_ps(t2_low, t2_high, M);
            t2_high = _mm256_blendv_ps(t2_high, t2_low, M);
            t2_low = T;

            // Calculate the intersection of all segments
            M = _mm256_cmp_ps(t0_low, t1_low, _CMP_GE_OQ); // t0_low >= t1_low ?
            t0_low = _mm256_blendv_ps(t1_low, t0_low, M);  // t0_low = max( t0_low, t1_low )
            M = _mm256_cmp_ps(t0_low, t2_low, _CMP_GE_OQ); // t0_low >= t2_low
            t0_low = _mm256_blendv_ps(t2_low, t0_low, M);  // t0_low = max( t0_low, t2_low ) = t_min

            M = _mm256_cmp_ps(t0_high, t1_high, _CMP_LE_OQ); // t0_high <= t1_high ?
            t0_high = _mm256_blendv_ps(t1_high, t0_high, M); // t0_high = min( t0_high, t1_high )
            M = _mm256_cmp_ps(t0_high, t2_high, _CMP_LE_OQ); // t0_high <= t2_high
            t0_high = _mm256_blendv_ps(t2_high, t0_high, M); // t0_high = min( t0_high, t2_high ) = t_max

            // If t0_high < 0, the ray is intersecting AABB, but the whole AABB is behind us
            // If t0_low > t0_high, ray doesn't intersect AABB
            M = _mm256_cmp_ps(r0, t0_high, _CMP_LE_OQ);     // t0_high > 0 ?
            T = _mm256_cmp_ps(t0_low, t0_high, _CMP_LE_OQ); // t0_high > t0_low ?
            M = _mm256_and_ps(M, T);                        // AND

            // Read output
            // The resulting integer vector will contain -1 if the AABB was hit and 0 otherwise
            _mm256_storeu_ps((float *)&p_sub_mesh_hit[i_sub], M);
        }

        // Step 2 - Check intersection with triangles within the sub-meshes

        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Skip if sub-mesh was not hit
            if (p_sub_mesh_hit[i_sub] == 0)
                continue;

            int i_mesh_start = (int)SMI[i_sub];
            int i_mesh_end = (i_sub == n_sub - 1) ? n_mesh_int : (int)SMI[i_sub + 1];

            for (int i_mesh = i_mesh_start; i_mesh < i_mesh_end; i_mesh += VEC_SIZE) // Mesh loop
            {
                // Load first vertex coordinate into AVX2 registers
                __m256 tx = _mm256_load_ps(&Tx[i_mesh]);
                __m256 ty = _mm256_load_ps(&Ty[i_mesh]);
                __m256 tz = _mm256_load_ps(&Tz[i_mesh]);

                // Calculate vector from first vertex coordinate V1 to origin O
                tx = _mm256_sub_ps(ox, tx);
                ty = _mm256_sub_ps(oy, ty);
                tz = _mm256_sub_ps(oz, tz);

                // Load the two triangle edges E1 and E2 into AVX2 registers
                __m256 e1x = _mm256_load_ps(&E1x[i_mesh]), e2x = _mm256_load_ps(&E2x[i_mesh]);
                __m256 e1y = _mm256_load_ps(&E1y[i_mesh]), e2y = _mm256_load_ps(&E2y[i_mesh]);
                __m256 e1z = _mm256_load_ps(&E1z[i_mesh]), e2z = _mm256_load_ps(&E2z[i_mesh]);

                // Calculate 1st barycentric coordinate U
                __m256 PQ = _mm256_mul_ps(e2y, dz); // PQ = e2y * dz
                PQ = _mm256_fmsub_ps(e2z, dy, PQ);  // PQ = e2z * dy - e2y * dz
                __m256 DT = _mm256_mul_ps(e1x, PQ); // DT = e1x * PQ
                __m256 U = _mm256_mul_ps(tx, PQ);   // U = tx * PQ

                PQ = _mm256_mul_ps(e2z, dx);       // PQ = e2z * dx
                PQ = _mm256_fmsub_ps(e2x, dz, PQ); // PQ = e2x * dz - e2z * dx
                DT = _mm256_fmadd_ps(e1y, PQ, DT); // DT = e1y * PQ + DT
                U = _mm256_fmadd_ps(ty, PQ, U);    // U = ty * PQ + U

                PQ = _mm256_mul_ps(e2x, dy);       // PQ = e2x * dy
                PQ = _mm256_fmsub_ps(e2y, dx, PQ); // PQ = e2y * dx - e2x * dy
                DT = _mm256_fmadd_ps(e1z, PQ, DT); // DT = e1z * PQ + DT
                U = _mm256_fmadd_ps(tz, PQ, U);    // U = tz * PQ + U

                // Calculate and 2nd barycentric coordinate (V) and normalized intersect position (W)
                PQ = _mm256_mul_ps(e1y, tz);       // PQ = e1y * tz
                PQ = _mm256_fmsub_ps(e1z, ty, PQ); // PQ = e1z * ty - e1y * tz
                __m256 V = _mm256_mul_ps(dx, PQ);  // V = dx * PQ
                __m256 W = _mm256_mul_ps(e2x, PQ); // W = e2x * PQ

                PQ = _mm256_mul_ps(e1z, tx);       // PQ = e1y * tx
                PQ = _mm256_fmsub_ps(e1x, tz, PQ); // PQ = e1x * tz - e1z * tx
                V = _mm256_fmadd_ps(dy, PQ, V);    // V = dy * PQ + V
                W = _mm256_fmadd_ps(e2y, PQ, W);   // W = e2y * PQ + W

                PQ = _mm256_mul_ps(e1x, ty);       // PQ = e1x * ty
                PQ = _mm256_fmsub_ps(e1y, tx, PQ); // PQ = e1y * tx - e1x * ty
                V = _mm256_fmadd_ps(dz, PQ, V);    // V = dz * PQ + V
                W = _mm256_fmadd_ps(e2z, PQ, W);   // W = e2z * PQ + W

                // Fast approximate division:
                // DT = _mm256_rcp_ps(DT);

                // Newton-Raphson refinement (medium speed):
                // PQ = _mm256_rcp_ps(DT);                                                  // Fast division
                // DT = _mm256_fnmadd_ps(DT, _mm256_mul_ps(PQ, PQ), _mm256_add_ps(PQ, PQ)); // 2PQ - DT * PQ^2

                // Accurate, but slow division:
                DT = _mm256_div_ps(r1, DT);

                U = _mm256_mul_ps(U, DT);
                V = _mm256_mul_ps(V, DT);
                W = _mm256_mul_ps(W, DT);

                // Check intersect conditions, output is a bitwise mask
                // https://stackoverflow.com/questions/16988199/how-to-choose-avx-compare-predicate-variants
                __m256 C1 = _mm256_cmp_ps(U, r0, _CMP_GE_OQ); // U >= 0
                __m256 C2 = _mm256_cmp_ps(V, r0, _CMP_GE_OQ); // V >= 0
                C1 = _mm256_and_ps(C1, C2);                   // U >= 0 & V >= 0
                U = _mm256_add_ps(U, V);                      // Compute U + V
                C2 = _mm256_cmp_ps(U, r1, _CMP_LE_OQ);        // (U + V) <= 1
                C1 = _mm256_and_ps(C1, C2);                   // U >= 0 & V >= 0 & (U + V) <= 1
                C2 = _mm256_cmp_ps(W, r0, _CMP_GE_OQ);        // W >= 0
                C1 = _mm256_and_ps(C1, C2);                   // U >= 0 & V >= 0 & (U + V) <= 1 & W > 0
                C2 = _mm256_cmp_ps(W, r1, _CMP_LT_OQ);        // W < 1
                C1 = _mm256_and_ps(C1, C2);                   // U >= 0 & V >= 0 & (U + V) <= 1 & W > 0 & W < 1

                // Fast exit if no hit was detected
                // this should be the case in 99.9% of the time
                __m256i Ip = _mm256_castps_si256(C1); // Cast C1 to __m256i
                if (_mm256_testz_si256(Ip, Ip) != 0)  // any( hit ) == false
                    continue;                         // Continue to next mesh block

                // Set values of W to 1 for which the condition is false (set intersect point equal to dest)
                W = _mm256_blendv_ps(r1, W, C1);

                // Count hits
                if (count_hits)
                {
                    __m256i Ci = _mm256_castps_si256(C1);                // Cast bitmask to integer (-1 for hit, 0 for miss)
                    Ci = _mm256_add_epi32(Ci, _mm256_srli_si256(Ci, 4)); // Shuffle and add adjacent elements
                    Ci = _mm256_add_epi32(Ci, _mm256_srli_si256(Ci, 8)); // Shuffle and add adjacent elements
                    hit_counter -= _mm256_extract_epi32(Ci, 0) + _mm256_extract_epi32(Ci, 4);
                }

                // Test if any of the hits is closer to orig than the SBS
                C1 = _mm256_cmp_ps(W, W_sbs, _CMP_LT_OQ); // W < W_sbs
                Ip = _mm256_castps_si256(C1);             // Cast to __m256i
                if (_mm256_testz_si256(Ip, Ip) != 0)      // any( W < W_sbs ) == false
                    continue;                             // Continue to next mesh block

                // Perform parallel reduction to find the FBS location and index
                // FBS Step 1: Pairwise comparisons
                __m256 Wp = _mm256_permute_ps(W, 0b10110001);                      // Permuted values
                C1 = _mm256_cmp_ps(W, Wp, _CMP_LT_OQ);                             // Compare values
                __m256 Wm = _mm256_blendv_ps(Wp, W, C1);                           // Minimum values
                __m256i Im = _mm256_blendv_epi8(i0p, i0, _mm256_castps_si256(C1)); // Minimum indices

                // FBS Step 2: Logarithmic reduction
                Wp = _mm256_permute2f128_ps(Wm, Wm, 0x21);                // Swap 128 bit lanes
                Ip = _mm256_permute2f128_si256(Im, Im, 0x21);             // Swap 128 bit lanes
                C1 = _mm256_cmp_ps(Wm, Wp, _CMP_LT_OQ);                   // Compare values
                Wm = _mm256_blendv_ps(Wp, Wm, C1);                        // Minimum values
                Im = _mm256_blendv_epi8(Ip, Im, _mm256_castps_si256(C1)); // Minimum indices

                // FBS Step 3: Final reduction
                Wp = _mm256_permute_ps(Wm, 0b00001111);                   // Permuted values
                Ip = _mm256_permutevar8x32_epi32(Im, iX);                 // Permuted indices
                C1 = _mm256_cmp_ps(Wm, Wp, _CMP_LT_OQ);                   // Compare values
                Wm = _mm256_blendv_ps(Wp, Wm, C1);                        // Minimum values
                Im = _mm256_blendv_epi8(Ip, Im, _mm256_castps_si256(C1)); // Minimum indices
                int i_avx = _mm256_extract_epi32(Im, 0);                  // Extract index in the AVX vector

                // Update FBS and SBS position
                C1 = _mm256_cmp_ps(Wm, W_fbs, _CMP_LT_OQ); // Wm < W_fbs
                C2 = _mm256_cmp_ps(Wm, W_sbs, _CMP_LT_OQ); // Wm < W_sbs
                Ip = _mm256_castps_si256(C1);              // Cast C1 to __m256i
                __m256i Iq = _mm256_castps_si256(C2);      // Cast C2 to __m256i
                if (_mm256_testz_si256(Ip, Ip) == 0)       // any( Wm < W_fbs ) == true
                {                                          // Update FBS and SBS
                    W_sbs = W_fbs;                         // The previous FBS becomes the new SBS
                    I_sbs = I_fbs;                         // Update SBS index
                    W_fbs = Wm;                            // Store the new FBS position
                    I_fbs = i_avx + i_mesh;                // Set new FBS index (0-based)
                }
                else if (_mm256_testz_si256(Iq, Iq) == 0) // any( Wm < W_sbs ) == true
                {                                         // Update only SBS
                    W_sbs = Wm;                           // Store the new SBS position
                    I_sbs = i_avx + i_mesh;               // Set new SBS index (0-based)
                }

                // Remove FBS from W (set its location to dest)
                Im = _mm256_set1_epi32(i_avx);                        // Fix multiple hits at same distance
                Ip = _mm256_cmpeq_epi32(Im, i0);                      // Bit mask
                W = _mm256_blendv_ps(W, r1, _mm256_castsi256_ps(Ip)); // Update W

                // Test if any of the remaining hits is closer to orig than the SBS
                C1 = _mm256_cmp_ps(W, W_sbs, _CMP_LT_OQ); // W < W_sbs
                Ip = _mm256_castps_si256(C1);             // Cast to __m256i
                if (_mm256_testz_si256(Ip, Ip) != 0)      // any( W < W_sbs ) == false
                    continue;                             // Continue to next mesh block

                // Perform parallel reduction to find the SBS location and index
                // SBS Step 1: Pairwise comparisons
                Wp = _mm256_permute_ps(W, 0b10110001);                     // Permuted values
                C1 = _mm256_cmp_ps(W, Wp, _CMP_LT_OQ);                     // Compare values
                Wm = _mm256_blendv_ps(Wp, W, C1);                          // Minimum values
                Im = _mm256_blendv_epi8(i0p, i0, _mm256_castps_si256(C1)); // Minimum indices

                // SBS Step 2: Logarithmic reduction
                Wp = _mm256_permute2f128_ps(Wm, Wm, 0x21);                // Swap 128 bit lanes
                Ip = _mm256_permute2f128_si256(Im, Im, 0x21);             // Swap 128 bit lanes
                C1 = _mm256_cmp_ps(Wm, Wp, _CMP_LT_OQ);                   // Compare values
                Wm = _mm256_blendv_ps(Wp, Wm, C1);                        // Minimum values
                Im = _mm256_blendv_epi8(Ip, Im, _mm256_castps_si256(C1)); // Minimum indices

                // SBS Step 3: Final reduction
                Wp = _mm256_permute_ps(Wm, 0b00001111);                   // Permuted values
                Ip = _mm256_permutevar8x32_epi32(Im, iX);                 // Permuted indices
                C1 = _mm256_cmp_ps(Wm, Wp, _CMP_LT_OQ);                   // Compare values
                Wm = _mm256_blendv_ps(Wp, Wm, C1);                        // Minimum values
                Im = _mm256_blendv_epi8(Ip, Im, _mm256_castps_si256(C1)); // Minimum indices

                // Update SBS position
                C2 = _mm256_cmp_ps(Wm, W_sbs, _CMP_LT_OQ);        // Wm < W_sbs
                Iq = _mm256_castps_si256(C2);                     // Cast C2 to __m256i
                if (_mm256_testz_si256(Iq, Iq) == 0)              // any( Wm < W_sbs ) == true
                {                                                 // Update SBS
                    W_sbs = Wm;                                   // Store the new SBS position
                    I_sbs = _mm256_extract_epi32(Im, 0) + i_mesh; // Set new SBS index (0-based)
                }
            }
        }

        // Update output memory
        Wf[i_ray] = _mm256_cvtss_f32(W_fbs);            // Extract first value from W_fbs
        Ws[i_ray] = _mm256_cvtss_f32(W_sbs);            // First value from W_sbs
        If[i_ray] = (Wf[i_ray] < 1.0f) ? I_fbs + 1 : 0; // Set FBS index (1-based)
        Is[i_ray] = (Ws[i_ray] < 1.0f) ? I_sbs + 1 : 0; // Set SBS index (1-based)

        if (count_hits)
            hit_cnt[i_ray] = hit_counter;
    }
}

// Generic C++ implementation of RayTriangleIntersect
inline void qd_RTI_GENERIC(const float *Tx, const float *Ty, const float *Tz,    // First vertex coordinate in GCS, length n_mesh
                           const float *E1x, const float *E1y, const float *E1z, // Edge 1 from first vertex to second vertex, length n_mesh
                           const float *E2x, const float *E2y, const float *E2z, // Edge 2 from first vertex to third vertex, length n_mesh
                           const size_t n_mesh,                                  // Number of triangles (multiple of VEC_SIZE)
                           const unsigned *SMI,                                  // List of sub-mesh indices, length n_sub
                           const float *Xmin, const float *Xmax,                 // Minimum and maximum x-values of the AABB, aligned to 32 byte, length n_sub_s
                           const float *Ymin, const float *Ymax,                 // Minimum and maximum y-values of the AABB, aligned to 32 byte, length n_sub_s
                           const float *Zmin, const float *Zmax,                 // Minimum and maximum z-values of the AABB, aligned to 32 byte, length n_sub_s
                           const size_t n_sub,                                   // Number of sub-meshes (not aligned, i.e. n_sub <= n_sub_s)
                           const float *Ox, const float *Oy, const float *Oz,    // Ray origin in GCS, length n_ray
                           const float *Dx, const float *Dy, const float *Dz,    // Vector from ray origin to ray destination, length n_ray
                           const size_t n_ray,                                   // Number of rays
                           float *Wf,                                            // Normalized distance (0-1) of FBS hit, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                           float *Ws,                                            // Normalized distance (0-1) of SBS hit, must be >= Wf, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                           unsigned *If,                                         // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                           unsigned *Is,                                         // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                           unsigned *hit_cnt = nullptr)                          // Number of hits between orig and dest, length n_ray, uninitialized, optional
{
    if (n_mesh >= INT32_MAX)
        throw std::invalid_argument("Number of triangles exceeds maximum supported number.");
    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    bool count_hits = hit_cnt != nullptr;

    // Constant values needed for some operations
    const int n_mesh_int = (int)n_mesh; // Number of triangles as int
    const int n_ray_int = (int)n_ray;   // Number of rays as int

#pragma omp parallel for
    for (int i_ray = 0; i_ray < n_ray_int; ++i_ray) // Ray loop
    {
        // Load origin
        float ox = Ox[i_ray];
        float oy = Oy[i_ray];
        float oz = Oz[i_ray];

        // Load vector from ray origin to ray destination
        float dx = Dx[i_ray];
        float dy = Dy[i_ray];
        float dz = Dz[i_ray];

        // Initialize local variables
        float W_fbs = 1.0f;            // Set FBS location equal to dest
        float W_sbs = W_fbs;           // Set SBS location equal to FBS location
        unsigned I_fbs = 0, I_sbs = 0; // Set FBS index to 0
        unsigned hit_counter = 0;      // Hit counter

        // Step 1 - Check intersection with the AABBs of the sub-meshes (slab-method)
        // See: https://en.wikipedia.org/wiki/Slab_method

        // Inverse of the direction (may be infinite if ray is parallel to an axis)
        float dx_i = 1.0f / dx;
        float dy_i = 1.0f / dy;
        float dz_i = 1.0f / dz;

        arma::s32_vec sub_mesh_hit(n_sub, arma::fill::none);
        int *p_sub_mesh_hit = sub_mesh_hit.memptr();
        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Calculate the intersections of the ray with the two planes orthogonal to the i-th coordinate axis
            float t0_low = (Xmin[i_sub] - ox) * dx_i;
            float t0_high = (Xmax[i_sub] - ox) * dx_i;
            float t1_low = (Ymin[i_sub] - oy) * dy_i;
            float t1_high = (Ymax[i_sub] - oy) * dy_i;
            float t2_low = (Zmin[i_sub] - oz) * dz_i;
            float t2_high = (Zmax[i_sub] - oz) * dz_i;

            // Calculate the close and far extrema of the segment within the i-th slab
            bool M = t0_low >= t0_high;
            float T = M ? t0_high : t0_low;
            t0_high = M ? t0_low : t0_high;
            t0_low = T;

            M = t1_low >= t1_high;
            T = M ? t1_high : t1_low;
            t1_high = M ? t1_low : t1_high;
            t1_low = T;

            M = t2_low >= t2_high;
            T = M ? t2_high : t2_low;
            t2_high = M ? t2_low : t2_high;
            t2_low = T;

            // Calculate the intersection of all segments
            M = t0_low >= t1_low;
            t0_low = M ? t0_low : t1_low;
            M = t0_low >= t2_low;
            t0_low = M ? t0_low : t2_low;

            M = t0_high <= t1_high;
            t0_high = M ? t0_high : t1_high;
            M = t0_high <= t2_high;
            t0_high = M ? t0_high : t2_high;

            // If t0_high < 0, the ray is intersecting AABB, but the whole AABB is behind us
            // If t0_low > t0_high, ray doesn't intersect AABB
            p_sub_mesh_hit[i_sub] = t0_high > 0.0f && t0_high > t0_low;
        }

        // Step 2 - Check intersection with triangles within the sub-meshes

        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Skip if sub-mesh was not hit
            if (p_sub_mesh_hit[i_sub] == 0)
                continue;

            int i_mesh_start = (int)SMI[i_sub];
            int i_mesh_end = (i_sub == n_sub - 1) ? n_mesh_int : (int)SMI[i_sub + 1];

            for (int i_mesh = i_mesh_start; i_mesh < i_mesh_end; ++i_mesh) // Mesh loop
            {
                // Load first vertex coordinate
                float tx = Tx[i_mesh];
                float ty = Ty[i_mesh];
                float tz = Tz[i_mesh];

                // Calculate vector from first vertex coordinate V1 to origin O
                tx = ox - tx;
                ty = oy - ty;
                tz = oz - tz;

                // Load the two triangle edges E1 and E2
                float e1x = E1x[i_mesh], e2x = E2x[i_mesh];
                float e1y = E1y[i_mesh], e2y = E2y[i_mesh];
                float e1z = E1z[i_mesh], e2z = E2z[i_mesh];

                // Calculate 1st barycentric coordinate U
                float PQ = e2z * dy - e2y * dz;
                float DT = e1x * PQ;
                float U = tx * PQ;

                PQ = e2x * dz - e2z * dx;
                DT = e1y * PQ + DT;
                U = ty * PQ + U;

                PQ = e2y * dx - e2x * dy;
                DT = e1z * PQ + DT;
                U = tz * PQ + U;

                // Calculate and 2nd barycentric coordinate (V) and normalized intersect position (W)
                PQ = e1z * ty - e1y * tz;
                float V = dx * PQ;
                float W = e2x * PQ;

                PQ = e1x * tz - e1z * tx;
                V = dy * PQ + V;
                W = e2y * PQ + W;

                PQ = e1y * tx - e1x * ty;
                V = dz * PQ + V;
                W = e2z * PQ + W;

                // Inverse of DT
                DT = 1.0f / DT;

                U = U * DT;
                V = V * DT;
                W = W * DT;

                // Check intersect conditions
                bool C1 = (U >= 0.0f) & (V >= 0.0f) & ((U + V) <= 1.0f) & (W >= 0.0f) & (W < 1.0f);

                // Fast exit if no hit was detected
                if (!C1)
                    continue;

                // Count hits
                if (count_hits)
                    ++hit_counter;

                // Update FBS and SBS position
                if (W < W_fbs)
                {                   // Update FBS and SBS
                    W_sbs = W_fbs;  // The previous FBS becomes the new SBS
                    I_sbs = I_fbs;  // Update SBS index
                    W_fbs = W;      // Store the new FBS position
                    I_fbs = i_mesh; // Set new FBS index (0-based)
                }
                else if (W < W_sbs)
                {                   // Update only SBS
                    W_sbs = W;      // Store the new SBS position
                    I_sbs = i_mesh; // Set new SBS index (0-based)
                }
            }
        }

        // Update output memory
        Wf[i_ray] = W_fbs;                              // Extract first value from W_fbs
        Ws[i_ray] = W_sbs;                              // First value from W_sbs
        If[i_ray] = (Wf[i_ray] < 1.0f) ? I_fbs + 1 : 0; // Set FBS index (1-based)
        Is[i_ray] = (Ws[i_ray] < 1.0f) ? I_sbs + 1 : 0; // Set SBS index (1-based)

        if (count_hits)
            hit_cnt[i_ray] = hit_counter;
    }
}

template <typename dtype>
void quadriga_lib::ray_triangle_intersect(const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest, const arma::Mat<dtype> *mesh,
                                          arma::Mat<dtype> *fbs, arma::Mat<dtype> *sbs, arma::Col<unsigned> *no_interact,
                                          arma::Col<unsigned> *fbs_ind, arma::Col<unsigned> *sbs_ind,
                                          const arma::Col<unsigned> *sub_mesh_index)
{
    // Input validation
    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (dest == nullptr)
        throw std::invalid_argument("Input 'dest' cannot be NULL.");
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (orig->n_elem == 0 || dest->n_elem == 0 || mesh->n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    if (dest->n_cols != 3)
        throw std::invalid_argument("Input 'dest' must have 3 columns containing x,y,z coordinates.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");

    size_t n_ray_t = orig->n_rows;
    size_t n_mesh_t = (size_t)mesh->n_rows;

    if (dest->n_rows != n_ray_t)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");

    // Convert orig and dest to aligned floats and calculate (dest - orig)
    size_t n_ray_s = n_ray_t; // (n_ray_t % VEC_SIZE == 0) ? n_ray_t : VEC_SIZE * (n_ray_t / VEC_SIZE + 1);
    auto origA = arma::fmat(n_ray_s, 3, arma::fill::none);
    auto dest_minus_origA = arma::fmat(n_ray_s, 3, arma::fill::none);
    {
        const dtype *p_orig = orig->memptr(), *p_dest = dest->memptr();
        float *p_origA = origA.memptr(), *p_dest_minus_origA = dest_minus_origA.memptr();
        size_t n_elem = 3 * n_ray_s, i = 0;
        for (size_t j = 0; j < n_elem; ++j)
            if ((j % n_ray_s) / n_ray_t) // = 1 for rows where n_ray_s > n_ray_t
            {
                p_origA[j] = 0.0f;
                p_dest_minus_origA[j] = 0.0f;
            }
            else
            {
                dtype v_orig = p_orig[i];                          // Load orig
                p_origA[j] = (float)v_orig;                        // Cast to float
                p_dest_minus_origA[j] = float(p_dest[i] - v_orig); // Calculate dest - orig
                ++i;
            }
    }

    // Check if the sub-mesh indices are valid
    size_t n_sub_t = 1;                                           // Number of sub-meshes (at least 1)
    arma::Col<unsigned> smi(1);                                   // Sub-mesh-index (local copy)
    if (sub_mesh_index != nullptr && sub_mesh_index->n_elem != 0) // Input is available
    {
        n_sub_t = (size_t)sub_mesh_index->n_elem;
        const unsigned *p_sub = sub_mesh_index->memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-mesh must start at index 0.");

        for (size_t i = 1; i < n_sub_t; ++i)
        {
            if (p_sub[i] <= p_sub[i - 1])
                throw std::invalid_argument("Sub-mesh indices must be sorted in ascending order.");

            if (p_sub[i] % VEC_SIZE != 0)
                throw std::invalid_argument("Sub-meshes must be aligned with the SIMD vector size (8 for AVX2, 32 for CUDA).");
        }

        if (p_sub[n_sub_t - 1] >= (unsigned)n_mesh_t)
            throw std::invalid_argument("Sub-mesh indices cannot exceed number of mesh elements.");

        smi = *sub_mesh_index;
    }

    // Alignment to 32 byte addresses is required when loading data into AVX2 registers
    // Not doing this may cause segmentation faults (e.g. in MATLAB)
    size_t n_mesh_s = (n_mesh_t % VEC_SIZE == 0) ? n_mesh_t : VEC_SIZE * (n_mesh_t / VEC_SIZE + 1);
    size_t n_sub_s = (n_sub_t % VEC_SIZE == 0) ? n_sub_t : VEC_SIZE * (n_sub_t / VEC_SIZE + 1);

#if defined(_MSC_VER) // Windows
    float *Tx = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *Ty = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *Tz = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E1x = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E1y = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E1z = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E2x = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E2y = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E2z = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *Xmin = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Xmax = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Ymin = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Ymax = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Zmin = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Zmax = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
#else // Linux
    float *Tx = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *Ty = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *Tz = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E1x = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E1y = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E1z = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E2x = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E2y = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E2z = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *Xmin = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Xmax = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Ymin = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Ymax = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Zmin = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Zmax = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
#endif

    // Convert mesh to float and write to aligned memory
    // Calculate bounding box for each sub-meshes
    const dtype *p_mesh = mesh->memptr();
    const unsigned *p_sub = smi.memptr();

    // Set parameters for the first AABB
    size_t i_sub = 0, i_next = (n_sub_t == 1) ? n_mesh_t - 1 : (size_t)p_sub[1] - 1;
    float x_min = INFINITY, x_max = -INFINITY,
          y_min = INFINITY, y_max = -INFINITY,
          z_min = INFINITY, z_max = -INFINITY;

    for (size_t i_mesh = 0; i_mesh < n_mesh_t; ++i_mesh)
    {
        // Load first vertex
        dtype x1 = p_mesh[i_mesh],
              y1 = p_mesh[i_mesh + n_mesh_t],
              z1 = p_mesh[i_mesh + 2 * n_mesh_t];

        // Typecast to float and update AABB
        float xf = (float)x1, yf = (float)y1, zf = (float)z1;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Write to aligned memory
        Tx[i_mesh] = xf, Ty[i_mesh] = yf, Tz[i_mesh] = zf;

        // Load second vertex
        dtype x = p_mesh[i_mesh + 3 * n_mesh_t],
              y = p_mesh[i_mesh + 4 * n_mesh_t],
              z = p_mesh[i_mesh + 5 * n_mesh_t];

        // Typecast to float and update AABB
        xf = (float)x, yf = (float)y, zf = (float)z;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Calculate edge and write to aligned memory
        E1x[i_mesh] = float(x - x1);
        E1y[i_mesh] = float(y - y1);
        E1z[i_mesh] = float(z - z1);

        // Load third vertex
        x = p_mesh[i_mesh + 6 * n_mesh_t],
        y = p_mesh[i_mesh + 7 * n_mesh_t],
        z = p_mesh[i_mesh + 8 * n_mesh_t];

        // Typecast to float and update AABB
        xf = (float)x, yf = (float)y, zf = (float)z;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Calculate edge and write to aligned memory
        E2x[i_mesh] = float(x - x1);
        E2y[i_mesh] = float(y - y1);
        E2z[i_mesh] = float(z - z1);

        // Update sub-mesh data for the next AABB
        if (i_mesh == i_next)
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
            i_next = (i_sub == n_sub_t - 1) ? n_mesh_t - 1 : (size_t)p_sub[i_sub + 1] - 1;
        }
    }

    // Add padding to the aligned mesh data
    for (size_t i_mesh = n_mesh_t; i_mesh < n_mesh_s; ++i_mesh)
    {
        Tx[i_mesh] = 0.0f, Ty[i_mesh] = 0.0f, Tz[i_mesh] = 0.0f,
        E1x[i_mesh] = 0.0f, E1y[i_mesh] = 0.0f, E1z[i_mesh] = 0.0f,
        E2x[i_mesh] = 0.0f, E2y[i_mesh] = 0.0f, E2z[i_mesh] = 0.0f;
    }

    // Add padding to the aligned AABB data
    for (size_t i_sub = n_sub_t; i_sub < n_sub_s; ++i_sub)
    {
        Xmin[i_sub] = 0.0f, Xmax[i_sub] = 0.0f;
        Ymin[i_sub] = 0.0f, Ymax[i_sub] = 0.0f;
        Zmin[i_sub] = 0.0f, Zmax[i_sub] = 0.0f;
    }

    // Define and initialize temporary variables
    arma::fvec Wf(n_ray_s), Ws(n_ray_s);    // Normalized FBS and SBS hit distances, initialized to 0
    arma::u32_vec If(n_ray_s), Is(n_ray_s); // Index of mesh element hit at FBS/SBS, initialized to 0
    arma::u32_vec hit_cnt(n_ray_s);         // Hit counter

    // Pointer to hit counter
    unsigned *p_hit_cnt = (no_interact == nullptr) ? nullptr : hit_cnt.memptr();

#if defined(__AVX2__)      // Compiler support for AVX2
    if (isAVX2Supported()) // CPU support for AVX2
    {
        qd_RTI_AVX2(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_s,
                    smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub_t,
                    origA.colptr(0), origA.colptr(1), origA.colptr(2),
                    dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                    n_ray_s, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
    }
    else
    {
        qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_t,
                       smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub_t,
                       origA.colptr(0), origA.colptr(1), origA.colptr(2),
                       dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                       n_ray_s, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
    }
#else
    qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_t,
                   smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub_t,
                   origA.colptr(0), origA.colptr(1), origA.colptr(2),
                   dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                   n_ray_s, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
#endif

    // Free aligned memory
#if defined(_MSC_VER) // Windows
    _aligned_free(Tx), _aligned_free(Ty), _aligned_free(Tz);
    _aligned_free(E1x), _aligned_free(E1y), _aligned_free(E1z);
    _aligned_free(E2x), _aligned_free(E2y), _aligned_free(E2z);
    _aligned_free(Xmin), _aligned_free(Xmax);
    _aligned_free(Ymin), _aligned_free(Ymax);
    _aligned_free(Zmin), _aligned_free(Zmax);
#else // Linux
    free(Tx), free(Ty), free(Tz);
    free(E1x), free(E1y), free(E1z);
    free(E2x), free(E2y), free(E2z);
    free(Xmin), free(Xmax);
    free(Ymin), free(Ymax);
    free(Zmin), free(Zmax);
#endif

    // Pointer to origin coordinates
    const dtype *ox = orig->colptr(0), *oy = orig->colptr(1), *oz = orig->colptr(2);

    // Pointer to dest_minus_origA
    float *dx = dest_minus_origA.colptr(0), *dy = dest_minus_origA.colptr(1), *dz = dest_minus_origA.colptr(2);

    // Compute FBS location in GCS
    if (fbs != nullptr)
    {
        if (fbs->n_rows != n_ray_t || fbs->n_cols != 3)
            fbs->set_size(n_ray_t, 3);

        dtype *px = fbs->colptr(0), *py = fbs->colptr(1), *pz = fbs->colptr(2);
        float *w = Wf.memptr();

        for (size_t i = 0; i < n_ray_t; ++i)
        {
            px[i] = ox[i] + dtype(w[i] * dx[i]);
            py[i] = oy[i] + dtype(w[i] * dy[i]);
            pz[i] = oz[i] + dtype(w[i] * dz[i]);
        }
    }

    // Compute SBS location in GCS
    if (sbs != nullptr)
    {
        if (sbs->n_rows != n_ray_t || sbs->n_cols != 3)
            sbs->set_size(n_ray_t, 3);

        dtype *px = sbs->colptr(0), *py = sbs->colptr(1), *pz = sbs->colptr(2);
        float *w = Ws.memptr();

        for (size_t i = 0; i < n_ray_t; ++i)
        {
            px[i] = ox[i] + dtype(w[i] * dx[i]);
            py[i] = oy[i] + dtype(w[i] * dy[i]);
            pz[i] = oz[i] + dtype(w[i] * dz[i]);
        }
    }

    // Copy the rest
    size_t no_bytes = (size_t)n_ray_t * sizeof(unsigned);
    if (no_interact != nullptr)
    {
        if (no_interact->n_elem != n_ray_t)
            no_interact->set_size(n_ray_t);
        std::memcpy(no_interact->memptr(), p_hit_cnt, no_bytes);
    }
    if (fbs_ind != nullptr)
    {
        if (fbs_ind->n_elem != n_ray_t)
            fbs_ind->set_size(n_ray_t);
        std::memcpy(fbs_ind->memptr(), If.memptr(), no_bytes);
    }
    if (sbs_ind != nullptr)
    {
        if (sbs_ind->n_elem != n_ray_t)
            sbs_ind->set_size(n_ray_t);
        std::memcpy(sbs_ind->memptr(), Is.memptr(), no_bytes);
    }
}

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<float> *orig, const arma::Mat<float> *dest, const arma::Mat<float> *mesh,
                                                   arma::Mat<float> *fbs, arma::Mat<float> *sbs, arma::Col<unsigned> *no_interact,
                                                   arma::Col<unsigned> *fbs_ind, arma::Col<unsigned> *sbs_ind,
                                                   const arma::Col<unsigned> *sub_mesh_index);

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<double> *orig, const arma::Mat<double> *dest, const arma::Mat<double> *mesh,
                                                   arma::Mat<double> *fbs, arma::Mat<double> *sbs, arma::Col<unsigned> *no_interact,
                                                   arma::Col<unsigned> *fbs_ind, arma::Col<unsigned> *sbs_ind,
                                                   const arma::Col<unsigned> *sub_mesh_index);