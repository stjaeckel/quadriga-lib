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
inline void qd_RTI_AVX2(const float *Tx, const float *Ty, const float *Tz,    // First vertex coordinate in GCS, length n_mesh
                        const float *E1x, const float *E1y, const float *E1z, // Edge 1 from first vertex to second vertex, length n_mesh
                        const float *E2x, const float *E2y, const float *E2z, // Edge 2 from first vertex to third vertex, length n_mesh
                        size_t n_mesh,                                        // Number of triangles (multiple of VEC_SIZE)
                        const float *Ox, const float *Oy, const float *Oz,    // Ray origin in GCS, length n_ray
                        const float *Dx, const float *Dy, const float *Dz,    // Vector from ray origin to ray destination, length n_ray
                        size_t n_ray,                                         // Number of rays
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
    const int n_mesh_int = (int)n_mesh;                            // Number of triangles as int
    const int n_ray_int = (int)n_ray;                              // Number of rays as int
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

        for (int i_mesh = 0; i_mesh < n_mesh_int; i_mesh += VEC_SIZE) // Mesh loop
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

        // Update output memory
        Wf[i_ray] = _mm256_cvtss_f32(W_fbs);            // Extract first value from W_fbs
        Ws[i_ray] = _mm256_cvtss_f32(W_sbs);            // First value from W_sbs
        If[i_ray] = (Wf[i_ray] < 1.0f) ? I_fbs + 1 : 0; // Set FBS index (1-based)
        Is[i_ray] = (Ws[i_ray] < 1.0f) ? I_sbs + 1 : 0; // Set SBS index (1-based)

        if (count_hits)
            hit_cnt[i_ray] = hit_counter;

        // std::cout << hit_cnt[i_ray] << " " << Wf[i_ray] << " " << If[i_ray] << " " << Ws[i_ray] << " " << Is[i_ray] << std::endl;
    }
}

// Generic C++ implementation of RayTriangleIntersect
inline void qd_RTI_GENERIC(const float *Tx, const float *Ty, const float *Tz,    // First vertex coordinate in GCS, length n_mesh
                           const float *E1x, const float *E1y, const float *E1z, // Edge 1 from first vertex to second vertex, length n_mesh
                           const float *E2x, const float *E2y, const float *E2z, // Edge 2 from first vertex to third vertex, length n_mesh
                           size_t n_mesh,                                        // Number of triangles (multiple of VEC_SIZE)
                           const float *Ox, const float *Oy, const float *Oz,    // Ray origin in GCS, length n_ray
                           const float *Dx, const float *Dy, const float *Dz,    // Vector from ray origin to ray destination, length n_ray
                           size_t n_ray,                                         // Number of rays
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

        for (int i_mesh = 0; i_mesh < n_mesh_int; ++i_mesh) // Mesh loop
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

        // Update output memory
        Wf[i_ray] = W_fbs;                              // Extract first value from W_fbs
        Ws[i_ray] = W_sbs;                              // First value from W_sbs
        If[i_ray] = (Wf[i_ray] < 1.0f) ? I_fbs + 1 : 0; // Set FBS index (1-based)
        Is[i_ray] = (Ws[i_ray] < 1.0f) ? I_sbs + 1 : 0; // Set SBS index (1-based)

        if (count_hits)
            hit_cnt[i_ray] = hit_counter;

        // std::cout << hit_cnt[i_ray] << " " << Wf[i_ray] << " " << If[i_ray] << " " << Ws[i_ray] << " " << Is[i_ray] << std::endl;
    }
}

template <typename dtype>
void quadriga_lib::ray_triangle_intersect(const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest, const arma::Mat<dtype> *mesh,
                                          arma::Mat<dtype> *fbs, arma::Mat<dtype> *sbs, arma::Col<unsigned> *no_interact,
                                          arma::Col<unsigned> *fbs_ind, arma::Col<unsigned> *sbs_ind)
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

    arma::uword n_rays = orig->n_rows;
    arma::uword n_mesh = mesh->n_rows;

    if (dest->n_rows != n_rays)
        throw std::invalid_argument("Number of rows in 'orig' and 'dest' dont match.");

    // Convert orig and dest to aligned floats and calculate (dest - orig)
    arma::uword n_rayA = n_rays; // (n_rays % VEC_SIZE == 0) ? n_rays : VEC_SIZE * (n_rays / VEC_SIZE + 1);
    auto origA = arma::fmat(n_rayA, 3);
    auto dest_minus_origA = arma::fmat(n_rayA, 3);
    {
        const dtype *p_orig = orig->memptr(), *p_dest = dest->memptr();
        float *p_origA = origA.memptr(), *p_dest_minus_origA = dest_minus_origA.memptr();
        for (arma::uword i = 0; i < n_rays; ++i)
        {
            dtype x = p_orig[i], y = p_orig[i + n_rays], z = p_orig[i + 2 * n_rays];
            p_origA[i] = (float)x, p_origA[i + n_rayA] = (float)y, p_origA[i + 2 * n_rayA] = (float)z;
            p_dest_minus_origA[i] = float(p_dest[i] - x);
            p_dest_minus_origA[i + n_rayA] = float(p_dest[i + n_rays] - y);
            p_dest_minus_origA[i + 2 * n_rayA] = float(p_dest[i + 2 * n_rays] - z);
        }
    }

    // Convert mesh to aligned floats and calculate edges
    // Alignment to 32 byte addresses is required when loading data into AVX2 registers
    // Not doing this may cause segmentation faults (e.g. in MATLAB)
    size_t n_meshS = (n_mesh % VEC_SIZE == 0) ? (size_t)n_mesh : VEC_SIZE * size_t(n_mesh / VEC_SIZE + 1);

#if defined(_MSC_VER) // Windows
    float *Tx = (float *)_aligned_malloc(n_meshS * sizeof(float), 32);
    float *Ty = (float *)_aligned_malloc(n_meshS * sizeof(float), 32);
    float *Tz = (float *)_aligned_malloc(n_meshS * sizeof(float), 32);
    float *E1x = (float *)_aligned_malloc(n_meshS * sizeof(float), 32);
    float *E1y = (float *)_aligned_malloc(n_meshS * sizeof(float), 32);
    float *E1z = (float *)_aligned_malloc(n_meshS * sizeof(float), 32);
    float *E2x = (float *)_aligned_malloc(n_meshS * sizeof(float), 32);
    float *E2y = (float *)_aligned_malloc(n_meshS * sizeof(float), 32);
    float *E2z = (float *)_aligned_malloc(n_meshS * sizeof(float), 32);
#else // Linux
    float *Tx = (float *)aligned_alloc(32, n_meshS * sizeof(float));
    float *Ty = (float *)aligned_alloc(32, n_meshS * sizeof(float));
    float *Tz = (float *)aligned_alloc(32, n_meshS * sizeof(float));
    float *E1x = (float *)aligned_alloc(32, n_meshS * sizeof(float));
    float *E1y = (float *)aligned_alloc(32, n_meshS * sizeof(float));
    float *E1z = (float *)aligned_alloc(32, n_meshS * sizeof(float));
    float *E2x = (float *)aligned_alloc(32, n_meshS * sizeof(float));
    float *E2y = (float *)aligned_alloc(32, n_meshS * sizeof(float));
    float *E2z = (float *)aligned_alloc(32, n_meshS * sizeof(float));
#endif

    const dtype *p_mesh = mesh->memptr();
    for (arma::uword i = 0; i < n_mesh; ++i)
    {
        dtype x = p_mesh[i], y = p_mesh[i + n_mesh], z = p_mesh[i + 2 * n_mesh];
        Tx[i] = (float)x, Ty[i] = (float)y, Tz[i] = (float)z;
        E1x[i] = float(p_mesh[i + 3 * n_mesh] - x);
        E1y[i] = float(p_mesh[i + 4 * n_mesh] - y);
        E1z[i] = float(p_mesh[i + 5 * n_mesh] - z);
        E2x[i] = float(p_mesh[i + 6 * n_mesh] - x);
        E2y[i] = float(p_mesh[i + 7 * n_mesh] - y);
        E2z[i] = float(p_mesh[i + 8 * n_mesh] - z);
    }
    for (arma::uword i = n_mesh; i < (arma::uword)n_meshS; ++i)
    {
        Tx[i] = 0.0f, Ty[i] = 0.0f, Tz[i] = 0.0f,
        E1x[i] = 0.0f, E1y[i] = 0.0f, E1z[i] = 0.0f,
        E2x[i] = 0.0f, E2y[i] = 0.0f, E2z[i] = 0.0f;
    }

    // Define and initialize temporary variables
    arma::fvec Wf(n_rayA), Ws(n_rayA);    // Normalized FBS and SBS hit distances, initialized to 0
    arma::u32_vec If(n_rayA), Is(n_rayA); // Index of mesh element hit at FBS/SBS, initialized to 0
    arma::u32_vec hit_cnt(n_rayA);        // Hit counter

    // Pointer to hit counter
    unsigned *p_hit_cnt = (no_interact == nullptr) ? nullptr : hit_cnt.memptr();

#if defined(__AVX2__)      // Compiler support for AVX2
    if (isAVX2Supported()) // CPU support for AVX2
    {
        qd_RTI_AVX2(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_meshS,
                    origA.colptr(0), origA.colptr(1), origA.colptr(2),
                    dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                    (size_t)n_rayA,
                    Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
    }
    else
    {
        qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                       origA.colptr(0), origA.colptr(1), origA.colptr(2),
                       dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                       (size_t)n_rayA,
                       Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
    }
#else
    qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                   origA.colptr(0), origA.colptr(1), origA.colptr(2),
                   dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                   (size_t)n_rayA,
                   Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
#endif

    // Free aligned memory
#if defined(_MSC_VER) // Windows
    _aligned_free(Tx), _aligned_free(Ty), _aligned_free(Tz);
    _aligned_free(E1x), _aligned_free(E1y), _aligned_free(E1z);
    _aligned_free(E2x), _aligned_free(E2y), _aligned_free(E2z);
#else // Linux
    free(Tx), free(Ty), free(Tz);
    free(E1x), free(E1y), free(E1z);
    free(E2x), free(E2y), free(E2z);
#endif

    // Pointer to origin coordinates
    const dtype *ox = orig->colptr(0), *oy = orig->colptr(1), *oz = orig->colptr(2);

    // Pointer to dest_minus_origA
    float *dx = dest_minus_origA.colptr(0), *dy = dest_minus_origA.colptr(1), *dz = dest_minus_origA.colptr(2);

    // Compute FBS location in GCS
    if (fbs != nullptr)
    {
        if (fbs->n_rows != n_rays || fbs->n_cols != 3)
            fbs->set_size(n_rays, 3);

        dtype *px = fbs->colptr(0), *py = fbs->colptr(1), *pz = fbs->colptr(2);
        float *w = Wf.memptr();

        for (arma::uword i = 0; i < n_rays; ++i)
        {
            px[i] = ox[i] + dtype(w[i] * dx[i]);
            py[i] = oy[i] + dtype(w[i] * dy[i]);
            pz[i] = oz[i] + dtype(w[i] * dz[i]);
        }
    }

    // Compute SBS location in GCS
    if (sbs != nullptr)
    {
        if (sbs->n_rows != n_rays || sbs->n_cols != 3)
            sbs->set_size(n_rays, 3);

        dtype *px = sbs->colptr(0), *py = sbs->colptr(1), *pz = sbs->colptr(2);
        float *w = Ws.memptr();

        for (arma::uword i = 0; i < n_rays; ++i)
        {
            px[i] = ox[i] + dtype(w[i] * dx[i]);
            py[i] = oy[i] + dtype(w[i] * dy[i]);
            pz[i] = oz[i] + dtype(w[i] * dz[i]);
        }
    }

    // Copy the rest
    size_t no_bytes = (size_t)n_rays * sizeof(unsigned);
    if (no_interact != nullptr)
    {
        if (no_interact->n_elem != n_rays)
            no_interact->set_size(n_rays);
        std::memcpy(no_interact->memptr(), p_hit_cnt, no_bytes);
    }
    if (fbs_ind != nullptr)
    {
        if (fbs_ind->n_elem != n_rays)
            fbs_ind->set_size(n_rays);
        std::memcpy(fbs_ind->memptr(), If.memptr(), no_bytes);
    }
    if (sbs_ind != nullptr)
    {
        if (sbs_ind->n_elem != n_rays)
            sbs_ind->set_size(n_rays);
        std::memcpy(sbs_ind->memptr(), Is.memptr(), no_bytes);
    }
}

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<float> *orig, const arma::Mat<float> *dest, const arma::Mat<float> *mesh,
                                                   arma::Mat<float> *fbs, arma::Mat<float> *sbs, arma::Col<unsigned> *no_interact,
                                                   arma::Col<unsigned> *fbs_ind, arma::Col<unsigned> *sbs_ind);

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<double> *orig, const arma::Mat<double> *dest, const arma::Mat<double> *mesh,
                                                   arma::Mat<double> *fbs, arma::Mat<double> *sbs, arma::Col<unsigned> *no_interact,
                                                   arma::Col<unsigned> *fbs_ind, arma::Col<unsigned> *sbs_ind);