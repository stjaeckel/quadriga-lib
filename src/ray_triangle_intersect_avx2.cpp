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

#include <immintrin.h>
#include "quadriga_lib_avx2_functions.hpp"

// Vector size for AVX2
#define VEC_SIZE 8

// ---- Double-to-float load helpers ------------------------------------------

// Load 8 doubles from memory and convert to a single __m256 of floats
static inline __m256 _load8_cvt_pd_ps(const double *p)
{
    __m256d lo = _mm256_loadu_pd(p);
    __m256d hi = _mm256_loadu_pd(p + 4);
    return _mm256_set_m128(_mm256_cvtpd_ps(hi), _mm256_cvtpd_ps(lo));
}

// Masked load of up to 8 doubles, converted to a single __m256 of floats
// mask must be an int32 lane mask (sign bit set = active lane)
static inline __m256 _maskload8_cvt_pd_ps(const double *p, __m256i mask)
{
    __m128i mlo = _mm256_castsi256_si128(mask);
    __m128i mhi = _mm256_extracti128_si256(mask, 1);
    __m256i mask64_lo = _mm256_cvtepi32_epi64(mlo);
    __m256i mask64_hi = _mm256_cvtepi32_epi64(mhi);
    __m256d lo = _mm256_maskload_pd(p, mask64_lo);
    __m256d hi = _mm256_maskload_pd(p + 4, mask64_hi);
    return _mm256_set_m128(_mm256_cvtpd_ps(hi), _mm256_cvtpd_ps(lo));
}

// ---- Shared AABB slab test (float SIMD) ------------------------------------

// Returns a float bitmask: -1 (all bits set) for hit, 0 for miss per lane
static inline __m256 rti_aabb_slab_test(__m256 xmin_v, __m256 xmax_v,
                                        __m256 ymin_v, __m256 ymax_v,
                                        __m256 zmin_v, __m256 zmax_v,
                                        __m256 ox, __m256 oy, __m256 oz,
                                        __m256 dx_i, __m256 dy_i, __m256 dz_i,
                                        __m256 r0, __m256 r1, __m256i aabb_mask)
{
    // Calculate the intersections of the ray with the two planes orthogonal to each coordinate axis
    __m256 T = _mm256_sub_ps(xmin_v, ox);
    __m256 t0_low = _mm256_mul_ps(T, dx_i);

    T = _mm256_sub_ps(xmax_v, ox);
    __m256 t0_high = _mm256_mul_ps(T, dx_i);

    T = _mm256_sub_ps(ymin_v, oy);
    __m256 t1_low = _mm256_mul_ps(T, dy_i);

    T = _mm256_sub_ps(ymax_v, oy);
    __m256 t1_high = _mm256_mul_ps(T, dy_i);

    T = _mm256_sub_ps(zmin_v, oz);
    __m256 t2_low = _mm256_mul_ps(T, dz_i);

    T = _mm256_sub_ps(zmax_v, oz);
    __m256 t2_high = _mm256_mul_ps(T, dz_i);

    // Calculate the close and far extrema of the segment within each slab
    __m256 M = _mm256_cmp_ps(t0_low, t0_high, _CMP_GE_OQ);
    T = _mm256_blendv_ps(t0_low, t0_high, M);
    t0_high = _mm256_blendv_ps(t0_high, t0_low, M);
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
    M = _mm256_cmp_ps(t0_low, t1_low, _CMP_GE_OQ);
    t0_low = _mm256_blendv_ps(t1_low, t0_low, M);
    M = _mm256_cmp_ps(t0_low, t2_low, _CMP_GE_OQ);
    t0_low = _mm256_blendv_ps(t2_low, t0_low, M);

    M = _mm256_cmp_ps(t0_high, t1_high, _CMP_LE_OQ);
    t0_high = _mm256_blendv_ps(t1_high, t0_high, M);
    M = _mm256_cmp_ps(t0_high, t2_high, _CMP_LE_OQ);
    t0_high = _mm256_blendv_ps(t2_high, t0_high, M);

    // If t0_high < 0, the ray is intersecting AABB, but the whole AABB is behind us
    // If t0_low >= t0_high, ray doesn't intersect AABB
    M = _mm256_cmp_ps(t0_high, r0, _CMP_GT_OQ);           // t0_high > 0 ?
    T = _mm256_cmp_ps(t0_high, t0_low, _CMP_GE_OQ);       // t0_high >= t0_low ?
    M = _mm256_and_ps(M, T);                              // AND
    T = _mm256_cmp_ps(t0_low, r1, _CMP_LE_OQ);            // t0_low <= 1
    M = _mm256_and_ps(M, T);                              // AND
    M = _mm256_and_ps(M, _mm256_castsi256_ps(aabb_mask)); // Kill invalid lanes

    return M;
}

// ---- Shared Möller-Trumbore + FBS/SBS update (float SIMD) ------------------

// tx, ty, tz must already be (origin - vertex), i.e. ox-Tx, oy-Ty, oz-Tz
static inline void rti_moller_trumbore_update(__m256 tx, __m256 ty, __m256 tz,
                                              __m256 e1x, __m256 e1y, __m256 e1z,
                                              __m256 e2x, __m256 e2y, __m256 e2z,
                                              __m256 dx, __m256 dy, __m256 dz,
                                              __m256 r0, __m256 r1,
                                              __m256i tail_mask, __m256i i0, __m256i i0p, __m256i iX,
                                              int i_mesh, bool count_hits,
                                              __m256 &W_fbs, __m256 &W_sbs,
                                              unsigned &I_fbs, unsigned &I_sbs,
                                              unsigned &hit_counter)
{
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

    // Calculate 2nd barycentric coordinate (V) and normalized intersect position (W)
    PQ = _mm256_mul_ps(e1y, tz);       // PQ = e1y * tz
    PQ = _mm256_fmsub_ps(e1z, ty, PQ); // PQ = e1z * ty - e1y * tz
    __m256 V = _mm256_mul_ps(dx, PQ);  // V = dx * PQ
    __m256 W = _mm256_mul_ps(e2x, PQ); // W = e2x * PQ

    PQ = _mm256_mul_ps(e1z, tx);       // PQ = e1z * tx
    PQ = _mm256_fmsub_ps(e1x, tz, PQ); // PQ = e1x * tz - e1z * tx
    V = _mm256_fmadd_ps(dy, PQ, V);    // V = dy * PQ + V
    W = _mm256_fmadd_ps(e2y, PQ, W);   // W = e2y * PQ + W

    PQ = _mm256_mul_ps(e1x, ty);       // PQ = e1x * ty
    PQ = _mm256_fmsub_ps(e1y, tx, PQ); // PQ = e1y * tx - e1x * ty
    V = _mm256_fmadd_ps(dz, PQ, V);    // V = dz * PQ + V
    W = _mm256_fmadd_ps(e2z, PQ, W);   // W = e2z * PQ + W

    // Accurate division
    DT = _mm256_div_ps(r1, DT);

    U = _mm256_mul_ps(U, DT);
    V = _mm256_mul_ps(V, DT);
    W = _mm256_mul_ps(W, DT);

    // Check intersect conditions, output is a bitwise mask
    __m256 C1 = _mm256_cmp_ps(U, r0, _CMP_GE_OQ);           // U >= 0
    __m256 C2 = _mm256_cmp_ps(V, r0, _CMP_GE_OQ);           // V >= 0
    C1 = _mm256_and_ps(C1, C2);                             // U >= 0 & V >= 0
    U = _mm256_add_ps(U, V);                                // Compute U + V
    C2 = _mm256_cmp_ps(U, r1, _CMP_LE_OQ);                  // (U + V) <= 1
    C1 = _mm256_and_ps(C1, C2);                             // U >= 0 & V >= 0 & (U + V) <= 1
    C2 = _mm256_cmp_ps(W, r0, _CMP_GE_OQ);                  // W >= 0
    C1 = _mm256_and_ps(C1, C2);                             // U >= 0 & V >= 0 & (U + V) <= 1 & W > 0
    C2 = _mm256_cmp_ps(W, r1, _CMP_LT_OQ);                  // W < 1
    C1 = _mm256_and_ps(C1, C2);                             // U >= 0 & V >= 0 & (U + V) <= 1 & W > 0 & W < 1
    C1 = _mm256_and_ps(C1, _mm256_castsi256_ps(tail_mask)); // kill invalid lanes

    // Fast exit if no hit was detected
    // this should be the case in 99.9% of the time
    __m256i Ip = _mm256_castps_si256(C1); // Cast C1 to __m256i
    if (_mm256_testz_si256(Ip, Ip) != 0)  // any( hit ) == false
        return;

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
        return;

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
        return;

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

// ---- Float specialization --------------------------------------------------

template <>
void qd_RTI_AVX2<float>(const float *Tx, const float *Ty, const float *Tz, // First vertex coordinate in GCS, length n_mesh
                        const float *Ux, const float *Uy, const float *Uz, // Second vertex coordinate in GCS, length n_mesh
                        const float *Vx, const float *Vy, const float *Vz, // Third vertex coordinate in GCS, length n_mesh
                        const size_t n_mesh,                               // Number of triangles
                        const unsigned *SMI,                               // List of sub-mesh indices, length n_sub
                        const float *Xmin, const float *Xmax,              // Minimum and maximum x-values of the AABB, length n_sub
                        const float *Ymin, const float *Ymax,              // Minimum and maximum y-values of the AABB, length n_sub
                        const float *Zmin, const float *Zmax,              // Minimum and maximum z-values of the AABB, length n_sub
                        const size_t n_sub,                                // Number of sub-meshes
                        const float *Ox, const float *Oy, const float *Oz, // Ray origin in GCS, length n_ray
                        const float *Dx, const float *Dy, const float *Dz, // Ray destination in GCS, length n_ray
                        const size_t n_ray,                                // Number of rays
                        float *Wf,                                         // Normalized distance (0-1) of FBS hit, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                        float *Ws,                                         // Normalized distance (0-1) of SBS hit, must be >= Wf, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                        unsigned *If,                                      // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                        unsigned *Is,                                      // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                        unsigned *hit_cnt)                                 // Number of hits between orig and dest, length n_ray, uninitialized, optional
{
    if (n_mesh >= INT32_MAX)
        throw std::invalid_argument("Number of triangles exceeds maximum supported number.");
    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    bool count_hits = hit_cnt != nullptr;

    // Constant values needed for some operations
    const size_t n_sub_pad = (n_sub % VEC_SIZE == 0) ? n_sub : VEC_SIZE * (n_sub / VEC_SIZE + 1);
    const int n_ray_int = (int)n_ray;
    const int n_mesh_int = (int)n_mesh;
    const __m256 r0 = _mm256_set1_ps(0.0f);
    const __m256 r1 = _mm256_set1_ps(1.0f);
    const __m256i i0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i i0p = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
    const __m256i iX = _mm256_set_epi32(0, 1, 2, 3, 0, 1, 2, 3);

#pragma omp parallel if (n_ray_int >= 64)
    {
        // Per-thread storage for sub-mesh hit indicators, reused across ray iterations
        std::vector<int> sub_mesh_hit(n_sub_pad);
        int *p_sub_mesh_hit = sub_mesh_hit.data();

#pragma omp for schedule(static)
        for (int i_ray = 0; i_ray < n_ray_int; ++i_ray)
        {
            // Load origin and direction into AVX2 registers
            __m256 ox = _mm256_set1_ps(Ox[i_ray]);
            __m256 oy = _mm256_set1_ps(Oy[i_ray]);
            __m256 oz = _mm256_set1_ps(Oz[i_ray]);

            __m256 dx = _mm256_set1_ps(Dx[i_ray] - Ox[i_ray]);
            __m256 dy = _mm256_set1_ps(Dy[i_ray] - Oy[i_ray]);
            __m256 dz = _mm256_set1_ps(Dz[i_ray] - Oz[i_ray]);

            // Initialize local variables
            __m256 W_fbs = _mm256_set1_ps(1.0f);
            __m256 W_sbs = W_fbs;
            unsigned I_fbs = 0, I_sbs = 0;
            unsigned hit_counter = 0;

            // Step 1 - AABB intersection test (slab method)
            __m256 dx_i = _mm256_div_ps(r1, dx);
            __m256 dy_i = _mm256_div_ps(r1, dy);
            __m256 dz_i = _mm256_div_ps(r1, dz);

            for (size_t i_sub = 0; i_sub < n_sub; i_sub += VEC_SIZE)
            {
                int remaining = (int)(n_sub - i_sub);
                __m256i aabb_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(remaining), i0);

                __m256 xmin_v, xmax_v, ymin_v, ymax_v, zmin_v, zmax_v;
                if (remaining >= VEC_SIZE)
                {
                    xmin_v = _mm256_loadu_ps(&Xmin[i_sub]);
                    xmax_v = _mm256_loadu_ps(&Xmax[i_sub]);
                    ymin_v = _mm256_loadu_ps(&Ymin[i_sub]);
                    ymax_v = _mm256_loadu_ps(&Ymax[i_sub]);
                    zmin_v = _mm256_loadu_ps(&Zmin[i_sub]);
                    zmax_v = _mm256_loadu_ps(&Zmax[i_sub]);
                }
                else
                {
                    xmin_v = _mm256_maskload_ps(&Xmin[i_sub], aabb_mask);
                    xmax_v = _mm256_maskload_ps(&Xmax[i_sub], aabb_mask);
                    ymin_v = _mm256_maskload_ps(&Ymin[i_sub], aabb_mask);
                    ymax_v = _mm256_maskload_ps(&Ymax[i_sub], aabb_mask);
                    zmin_v = _mm256_maskload_ps(&Zmin[i_sub], aabb_mask);
                    zmax_v = _mm256_maskload_ps(&Zmax[i_sub], aabb_mask);
                }

                __m256 M = rti_aabb_slab_test(xmin_v, xmax_v, ymin_v, ymax_v, zmin_v, zmax_v,
                                              ox, oy, oz, dx_i, dy_i, dz_i, r0, r1, aabb_mask);

                if (remaining >= VEC_SIZE)
                    _mm256_storeu_ps((float *)&p_sub_mesh_hit[i_sub], M);
                else
                    _mm256_maskstore_ps((float *)&p_sub_mesh_hit[i_sub], aabb_mask, M);
            }

            // Step 2 - Check intersection with triangles within the sub-meshes
            for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
            {
                if (p_sub_mesh_hit[i_sub] == 0)
                    continue;

                int i_mesh_start = (int)SMI[i_sub];
                int i_mesh_end = (i_sub == n_sub - 1) ? n_mesh_int : (int)SMI[i_sub + 1];

                for (int i_mesh = i_mesh_start; i_mesh < i_mesh_end; i_mesh += VEC_SIZE)
                {
                    int remaining = i_mesh_end - i_mesh;
                    __m256i tail_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(remaining), i0);
                    __m256 tx, ty, tz, e1x, e1y, e1z, e2x, e2y, e2z;
                    if (remaining >= VEC_SIZE)
                    {
                        tx = _mm256_loadu_ps(&Tx[i_mesh]);
                        ty = _mm256_loadu_ps(&Ty[i_mesh]);
                        tz = _mm256_loadu_ps(&Tz[i_mesh]);
                        e1x = _mm256_loadu_ps(&Ux[i_mesh]);
                        e1y = _mm256_loadu_ps(&Uy[i_mesh]);
                        e1z = _mm256_loadu_ps(&Uz[i_mesh]);
                        e2x = _mm256_loadu_ps(&Vx[i_mesh]);
                        e2y = _mm256_loadu_ps(&Vy[i_mesh]);
                        e2z = _mm256_loadu_ps(&Vz[i_mesh]);
                    }
                    else
                    {
                        tx = _mm256_maskload_ps(&Tx[i_mesh], tail_mask);
                        ty = _mm256_maskload_ps(&Ty[i_mesh], tail_mask);
                        tz = _mm256_maskload_ps(&Tz[i_mesh], tail_mask);
                        e1x = _mm256_maskload_ps(&Ux[i_mesh], tail_mask);
                        e1y = _mm256_maskload_ps(&Uy[i_mesh], tail_mask);
                        e1z = _mm256_maskload_ps(&Uz[i_mesh], tail_mask);
                        e2x = _mm256_maskload_ps(&Vx[i_mesh], tail_mask);
                        e2y = _mm256_maskload_ps(&Vy[i_mesh], tail_mask);
                        e2z = _mm256_maskload_ps(&Vz[i_mesh], tail_mask);
                    }

                    // Compute edges inline
                    e1x = _mm256_sub_ps(e1x, tx);
                    e1y = _mm256_sub_ps(e1y, ty);
                    e1z = _mm256_sub_ps(e1z, tz);
                    e2x = _mm256_sub_ps(e2x, tx);
                    e2y = _mm256_sub_ps(e2y, ty);
                    e2z = _mm256_sub_ps(e2z, tz);

                    // Calculate vector from first vertex coordinate V1 to origin O
                    tx = _mm256_sub_ps(ox, tx);
                    ty = _mm256_sub_ps(oy, ty);
                    tz = _mm256_sub_ps(oz, tz);

                    rti_moller_trumbore_update(tx, ty, tz, e1x, e1y, e1z, e2x, e2y, e2z,
                                               dx, dy, dz, r0, r1, tail_mask, i0, i0p, iX,
                                               i_mesh, count_hits, W_fbs, W_sbs, I_fbs, I_sbs, hit_counter);
                }
            }

            // Update output memory
            Wf[i_ray] = _mm256_cvtss_f32(W_fbs);
            Ws[i_ray] = _mm256_cvtss_f32(W_sbs);
            If[i_ray] = (Wf[i_ray] < 1.0f) ? I_fbs + 1 : 0;
            Is[i_ray] = (Ws[i_ray] < 1.0f) ? I_sbs + 1 : 0;

            if (count_hits)
                hit_cnt[i_ray] = hit_counter;
        } // end parallel loop
    } // end omp parallel
}
// ---- Double specialization -------------------------------------------------

template <>
void qd_RTI_AVX2<double>(const double *Tx, const double *Ty, const double *Tz, // First vertex coordinate in GCS, length n_mesh
                         const double *Ux, const double *Uy, const double *Uz, // Second vertex coordinate in GCS, length n_mesh
                         const double *Vx, const double *Vy, const double *Vz, // Third vertex coordinate in GCS, length n_mesh
                         const size_t n_mesh,                                  // Number of triangles
                         const unsigned *SMI,                                  // List of sub-mesh indices, length n_sub
                         const double *Xmin, const double *Xmax,               // Minimum and maximum x-values of the AABB, length n_sub
                         const double *Ymin, const double *Ymax,               // Minimum and maximum y-values of the AABB, length n_sub
                         const double *Zmin, const double *Zmax,               // Minimum and maximum z-values of the AABB, length n_sub
                         const size_t n_sub,                                   // Number of sub-meshes
                         const double *Ox, const double *Oy, const double *Oz, // Ray origin in GCS, length n_ray
                         const double *Dx, const double *Dy, const double *Dz, // Ray destination in GCS, length n_ray
                         const size_t n_ray,                                   // Number of rays
                         double *Wf,                                           // Normalized distance (0-1) of FBS hit, length n_ray, uninitialized
                         double *Ws,                                           // Normalized distance (0-1) of SBS hit, length n_ray, uninitialized
                         unsigned *If,                                         // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                         unsigned *Is,                                         // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                         unsigned *hit_cnt)                                    // Number of hits between orig and dest, length n_ray, uninitialized, optional
{
    if (n_mesh >= INT32_MAX)
        throw std::invalid_argument("Number of triangles exceeds maximum supported number.");
    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    bool count_hits = hit_cnt != nullptr;

    const size_t n_sub_pad = (n_sub % VEC_SIZE == 0) ? n_sub : VEC_SIZE * (n_sub / VEC_SIZE + 1);
    const int n_ray_int = (int)n_ray;
    const int n_mesh_int = (int)n_mesh;
    const __m256 r0 = _mm256_set1_ps(0.0f);
    const __m256 r1 = _mm256_set1_ps(1.0f);
    const __m256i i0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i i0p = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
    const __m256i iX = _mm256_set_epi32(0, 1, 2, 3, 0, 1, 2, 3);

#pragma omp parallel if (n_ray_int >= 64)
    {
        // Per-thread storage for sub-mesh hit indicators, reused across ray iterations
        std::vector<int> sub_mesh_hit(n_sub_pad);
        int *p_sub_mesh_hit = sub_mesh_hit.data();

#pragma omp for schedule(static)
        for (int i_ray = 0; i_ray < n_ray_int; ++i_ray)
        {
            // Load origin and direction, convert double scalars to float broadcasts
            __m256 ox = _mm256_set1_ps((float)Ox[i_ray]);
            __m256 oy = _mm256_set1_ps((float)Oy[i_ray]);
            __m256 oz = _mm256_set1_ps((float)Oz[i_ray]);

            __m256 dx = _mm256_set1_ps(float(Dx[i_ray] - Ox[i_ray]));
            __m256 dy = _mm256_set1_ps(float(Dy[i_ray] - Oy[i_ray]));
            __m256 dz = _mm256_set1_ps(float(Dz[i_ray] - Oz[i_ray]));

            // Initialize local variables
            __m256 W_fbs = _mm256_set1_ps(1.0f);
            __m256 W_sbs = W_fbs;
            unsigned I_fbs = 0, I_sbs = 0;
            unsigned hit_counter = 0;

            // Step 1 - AABB intersection test (slab method)
            __m256 dx_i = _mm256_div_ps(r1, dx);
            __m256 dy_i = _mm256_div_ps(r1, dy);
            __m256 dz_i = _mm256_div_ps(r1, dz);

            for (size_t i_sub = 0; i_sub < n_sub; i_sub += VEC_SIZE)
            {
                int remaining = (int)(n_sub - i_sub);
                __m256i aabb_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(remaining), i0);

                __m256 xmin_v, xmax_v, ymin_v, ymax_v, zmin_v, zmax_v;
                if (remaining >= VEC_SIZE)
                {
                    xmin_v = _load8_cvt_pd_ps(&Xmin[i_sub]);
                    xmax_v = _load8_cvt_pd_ps(&Xmax[i_sub]);
                    ymin_v = _load8_cvt_pd_ps(&Ymin[i_sub]);
                    ymax_v = _load8_cvt_pd_ps(&Ymax[i_sub]);
                    zmin_v = _load8_cvt_pd_ps(&Zmin[i_sub]);
                    zmax_v = _load8_cvt_pd_ps(&Zmax[i_sub]);
                }
                else
                {
                    xmin_v = _maskload8_cvt_pd_ps(&Xmin[i_sub], aabb_mask);
                    xmax_v = _maskload8_cvt_pd_ps(&Xmax[i_sub], aabb_mask);
                    ymin_v = _maskload8_cvt_pd_ps(&Ymin[i_sub], aabb_mask);
                    ymax_v = _maskload8_cvt_pd_ps(&Ymax[i_sub], aabb_mask);
                    zmin_v = _maskload8_cvt_pd_ps(&Zmin[i_sub], aabb_mask);
                    zmax_v = _maskload8_cvt_pd_ps(&Zmax[i_sub], aabb_mask);
                }

                __m256 M = rti_aabb_slab_test(xmin_v, xmax_v, ymin_v, ymax_v, zmin_v, zmax_v,
                                              ox, oy, oz, dx_i, dy_i, dz_i, r0, r1, aabb_mask);

                if (remaining >= VEC_SIZE)
                    _mm256_storeu_ps((float *)&p_sub_mesh_hit[i_sub], M);
                else
                    _mm256_maskstore_ps((float *)&p_sub_mesh_hit[i_sub], aabb_mask, M);
            }

            // Step 2 - Check intersection with triangles within the sub-meshes
            for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
            {
                if (p_sub_mesh_hit[i_sub] == 0)
                    continue;

                int i_mesh_start = (int)SMI[i_sub];
                int i_mesh_end = (i_sub == n_sub - 1) ? n_mesh_int : (int)SMI[i_sub + 1];

                for (int i_mesh = i_mesh_start; i_mesh < i_mesh_end; i_mesh += VEC_SIZE)
                {
                    int remaining = i_mesh_end - i_mesh;
                    __m256i tail_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(remaining), i0);
                    __m256 tx, ty, tz, e1x, e1y, e1z, e2x, e2y, e2z;
                    if (remaining >= VEC_SIZE)
                    {
                        tx = _load8_cvt_pd_ps(&Tx[i_mesh]);
                        ty = _load8_cvt_pd_ps(&Ty[i_mesh]);
                        tz = _load8_cvt_pd_ps(&Tz[i_mesh]);
                        e1x = _load8_cvt_pd_ps(&Ux[i_mesh]);
                        e1y = _load8_cvt_pd_ps(&Uy[i_mesh]);
                        e1z = _load8_cvt_pd_ps(&Uz[i_mesh]);
                        e2x = _load8_cvt_pd_ps(&Vx[i_mesh]);
                        e2y = _load8_cvt_pd_ps(&Vy[i_mesh]);
                        e2z = _load8_cvt_pd_ps(&Vz[i_mesh]);
                    }
                    else
                    {
                        tx = _maskload8_cvt_pd_ps(&Tx[i_mesh], tail_mask);
                        ty = _maskload8_cvt_pd_ps(&Ty[i_mesh], tail_mask);
                        tz = _maskload8_cvt_pd_ps(&Tz[i_mesh], tail_mask);
                        e1x = _maskload8_cvt_pd_ps(&Ux[i_mesh], tail_mask);
                        e1y = _maskload8_cvt_pd_ps(&Uy[i_mesh], tail_mask);
                        e1z = _maskload8_cvt_pd_ps(&Uz[i_mesh], tail_mask);
                        e2x = _maskload8_cvt_pd_ps(&Vx[i_mesh], tail_mask);
                        e2y = _maskload8_cvt_pd_ps(&Vy[i_mesh], tail_mask);
                        e2z = _maskload8_cvt_pd_ps(&Vz[i_mesh], tail_mask);
                    }

                    // Compute edges inline
                    e1x = _mm256_sub_ps(e1x, tx);
                    e1y = _mm256_sub_ps(e1y, ty);
                    e1z = _mm256_sub_ps(e1z, tz);
                    e2x = _mm256_sub_ps(e2x, tx);
                    e2y = _mm256_sub_ps(e2y, ty);
                    e2z = _mm256_sub_ps(e2z, tz);

                    // Calculate vector from first vertex coordinate V1 to origin O
                    tx = _mm256_sub_ps(ox, tx);
                    ty = _mm256_sub_ps(oy, ty);
                    tz = _mm256_sub_ps(oz, tz);

                    rti_moller_trumbore_update(tx, ty, tz, e1x, e1y, e1z, e2x, e2y, e2z,
                                               dx, dy, dz, r0, r1, tail_mask, i0, i0p, iX,
                                               i_mesh, count_hits, W_fbs, W_sbs, I_fbs, I_sbs, hit_counter);
                }
            }

            // Update output memory - convert float result back to double
            Wf[i_ray] = (double)_mm256_cvtss_f32(W_fbs);
            Ws[i_ray] = (double)_mm256_cvtss_f32(W_sbs);
            If[i_ray] = (Wf[i_ray] < 1.0) ? I_fbs + 1 : 0;
            Is[i_ray] = (Ws[i_ray] < 1.0) ? I_sbs + 1 : 0;

            if (count_hits)
                hit_cnt[i_ray] = hit_counter;
        } // end parallel loop
    } // end omp parallel
}