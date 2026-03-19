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

#include "qd_arrayant_interpolate_avx2.hpp"
#include <immintrin.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <cmath>
#include "fastmath_avx2.h"

// Euler rotations (bank, tilt, head) to rotation matrix using ZYX intrinsic convention.
// Angles: a = heading (Z), b = tilt (Y), c = bank (X)
template <typename dtype>
static inline void qd_rotation_matrix(const dtype *euler_angles_3xN, // Orientation vectors (Euler rotations)
                                      dtype *rotation_matrix_9xN,    // Rotation matrix in column-major ordering
                                      size_t N = 1,                  // Number of elements
                                      bool invert_y_axis = false,    // Inverts the y-axis
                                      bool transposeR = false)       // Returns the transpose of R
{
    for (size_t i = 0; i < 3 * N; i += 3)
    {
        const dtype *O = &euler_angles_3xN[i];
        dtype *R = &rotation_matrix_9xN[3 * i];

        const double bank = (double)O[0], tilt = (double)O[1], head = (double)O[2];
        const double sa = std::sin(head), ca = std::cos(head);
        double sb = std::sin(tilt);
        const double cb = std::cos(tilt);
        const double sc = std::sin(bank), cc = std::cos(bank);
        sb = invert_y_axis ? -sb : sb;

        if (transposeR)
        {
            R[0] = dtype(ca * cb);
            R[1] = dtype(ca * sb * sc - sa * cc);
            R[2] = dtype(ca * sb * cc + sa * sc);
            R[3] = dtype(sa * cb);
            R[4] = dtype(sa * sb * sc + ca * cc);
            R[5] = dtype(sa * sb * cc - ca * sc);
            R[6] = dtype(-sb);
            R[7] = dtype(cb * sc);
            R[8] = dtype(cb * cc);
        }
        else
        {
            R[0] = dtype(ca * cb);
            R[1] = dtype(sa * cb);
            R[2] = dtype(-sb);
            R[3] = dtype(ca * sb * sc - sa * cc);
            R[4] = dtype(sa * sb * sc + ca * cc);
            R[5] = dtype(cb * sc);
            R[6] = dtype(ca * sb * cc + sa * sc);
            R[7] = dtype(sa * sb * cc - ca * sc);
            R[8] = dtype(cb * cc);
        }
    }
}

// =====================================================================================
// AVX2 HELPER TYPES AND FUNCTIONS
//
// These helpers factor out the repeated computation stages that are identical across
// the three code paths (element-vectorized, angle-vectorized, general).
// Each path differs only in how it loads inputs (broadcast vs. SoA gather) and how
// it stores outputs (contiguous vs. strided). The shared math pipeline is here.
// =====================================================================================

// -------------------------------------------------------------------------------------
// RMatrix8 — 3×3 rotation matrix, 8 SIMD lanes per entry
//
// Stores R^T in column-major order (equivalently, R in row-major order).
// r[0..2] = column 0 of R^T = row 0 of R
// r[3..5] = column 1 of R^T = row 1 of R
// r[6..8] = column 2 of R^T = row 2 of R
//
// Forward rotation:  d = R * v   uses r[0],r[3],r[6] / r[1],r[4],r[7] / r[2],r[5],r[8]
// Back-rotation:     d = R^T * v uses r[0],r[1],r[2] / r[3],r[4],r[5] / r[6],r[7],r[8]
// -------------------------------------------------------------------------------------
struct RMatrix8
{
    __m256 r[9];
};

// -------------------------------------------------------------------------------------
// AzGridParams — precomputed azimuth grid search parameters (cyclic wrapping)
// -------------------------------------------------------------------------------------
struct AzGridParams
{
    const float *grid_f;   // Float grid values, length [n + 1] with grid_f[n] = +inf sentinel
    const float *step_inv; // Reciprocal grid spacings [n], step_inv[i] = 1/(grid[i] - grid[i-1])
    size_t n;              // Number of azimuth grid points (n_azimuth)
    bool uniform;          // True if grid is uniformly spaced
    float fmin;            // Grid minimum: grid_f[0]
    float fmax;            // Grid maximum: grid_f[n-1]
    float step_rinv;       // 1 / uniform_step (only valid if uniform == true)
    int bsearch_iters;     // Binary search iterations: ceil(log2(n + 1))
    float eps_cubed;       // Tiny guard for weight computation (eps^3)
};

// -------------------------------------------------------------------------------------
// ElGridParams — precomputed elevation grid search parameters (clamped, no wrapping)
// -------------------------------------------------------------------------------------
struct ElGridParams
{
    const float *grid_f;   // Float grid values, length [n + 1] with grid_f[n] = +inf sentinel
    const float *step_inv; // Reciprocal grid spacings [n]
    size_t n;              // Number of elevation grid points (n_elevation)
    bool uniform;          // True if grid is uniformly spaced
    float fmin;            // Grid minimum: grid_f[0]
    float step_rinv;       // 1 / uniform_step (only valid if uniform == true)
    int bsearch_iters;     // Binary search iterations: ceil(log2(n + 1))
    float eps_cubed;       // Tiny guard for weight computation (eps^3)
};

// -------------------------------------------------------------------------------------
// avx2_grid_search_azimuth — Vectorized azimuth grid lookup with cyclic wrapping
//
// For 8 azimuth query values in `az8`, finds the bracketing grid indices and the
// interpolation weight for bilinear interpolation. Azimuth wraps cyclically: values
// below grid min or above grid max interpolate between the last and first grid points.
//
// Handles three sub-cases internally:
//   n == 1:         Single grid point, no interpolation needed
//   uniform grid:   O(1) index from (az - min) / step, with cyclic wrap handling
//   non-uniform:    Vectorized binary search + cyclic wrap handling
//
// Inputs:
//   az8     — 8 query azimuth values [rad]
//   p       — precomputed grid parameters (see AzGridParams)
//
// Outputs:
//   iUp8    — lower bracket indices (8× int32), range [0, n-1]
//   iUn8    — upper bracket indices (8× int32), range [0, n-1]
//   up8     — interpolation weight for the lower bracket, range [0, 1]
//             (weight for upper bracket is implicitly 1 - up8)
// -------------------------------------------------------------------------------------
static inline void avx2_grid_search_azimuth(
    __m256 az8,
    const AzGridParams &p,
    __m256i *iUp8, __m256i *iUn8, __m256 *up8)
{
    const __m256 ones_8 = _mm256_set1_ps(1.0f);
    const __m256 zeros_8 = _mm256_setzero_ps();
    const __m256 eps0_8 = _mm256_set1_ps(p.eps_cubed);

    // ---- Trivial case: single grid point ----
    if (p.n == 1)
    {
        *iUp8 = _mm256_setzero_si256();
        *iUn8 = _mm256_setzero_si256();
        *up8 = ones_8;
        return;
    }

    // ---- Compute normal-range indices and weight ----
    __m256i iUp_norm, iUn_norm;
    __m256 up_norm;
    __m256 mask_below, mask_above;

    if (p.uniform)
    {
        // O(1) index computation for uniform grids
        const __m256 az_min_v = _mm256_set1_ps(p.fmin);
        const __m256 az_max_v = _mm256_set1_ps(p.fmax);
        const __m256 az_rinv_v = _mm256_set1_ps(p.step_rinv);
        const __m256i az_max_idx = _mm256_set1_epi32((int)(p.n - 2));

        __m256 fidx = _mm256_mul_ps(_mm256_sub_ps(az8, az_min_v), az_rinv_v);

        // Branchless floor: truncate, then adjust if negative fractional part
        __m256i i_trunc = _mm256_cvttps_epi32(fidx);
        __m256 f_trunc = _mm256_cvtepi32_ps(i_trunc);
        __m256 neg_adj = _mm256_cmp_ps(fidx, f_trunc, _CMP_LT_OQ);
        __m256i i_floor = _mm256_add_epi32(i_trunc, _mm256_castps_si256(neg_adj));

        iUp_norm = _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(i_floor, az_max_idx));
        iUn_norm = _mm256_add_epi32(iUp_norm, _mm256_set1_epi32(1));

        __m256 grid_un = _mm256_i32gather_ps(p.grid_f, iUn_norm, 4);
        __m256 un_norm = _mm256_mul_ps(_mm256_sub_ps(grid_un, az8), az_rinv_v);
        un_norm = _mm256_max_ps(zeros_8, _mm256_min_ps(ones_8, un_norm));
        up_norm = _mm256_sub_ps(ones_8, un_norm);

        mask_below = _mm256_cmp_ps(az8, az_min_v, _CMP_LT_OQ);
        mask_above = _mm256_cmp_ps(az8, az_max_v, _CMP_GE_OQ);
    }
    else
    {
        // Vectorized binary search for non-uniform grids
        __m256i bs_lo = _mm256_setzero_si256();
        __m256i bs_hi = _mm256_set1_epi32((int)p.n);
        for (int iter = 0; iter < p.bsearch_iters; ++iter)
        {
            __m256i mid = _mm256_srli_epi32(_mm256_add_epi32(bs_lo, bs_hi), 1);
            __m256 val = _mm256_i32gather_ps(p.grid_f, mid, 4);
            __m256 cmp = _mm256_cmp_ps(val, az8, _CMP_LE_OQ);
            __m256i cmp_i = _mm256_castps_si256(cmp);
            bs_lo = _mm256_blendv_epi8(bs_lo, _mm256_add_epi32(mid, _mm256_set1_epi32(1)), cmp_i);
            bs_hi = _mm256_blendv_epi8(mid, bs_hi, cmp_i);
        }

        __m256i ub8_az = bs_lo;
        iUp_norm = _mm256_sub_epi32(ub8_az, _mm256_set1_epi32(1));
        iUn_norm = ub8_az;

        __m256i iUn_safe = _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(iUn_norm, _mm256_set1_epi32((int)(p.n - 1))));
        __m256 grid_un = _mm256_i32gather_ps(p.grid_f, iUn_safe, 4);
        __m256 diff_un = _mm256_i32gather_ps(p.step_inv, iUn_safe, 4);
        __m256 un_norm = _mm256_mul_ps(_mm256_sub_ps(grid_un, az8), diff_un);
        un_norm = _mm256_min_ps(ones_8, un_norm);
        up_norm = _mm256_sub_ps(ones_8, un_norm);

        mask_below = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ub8_az, _mm256_setzero_si256()));
        mask_above = _mm256_castsi256_ps(_mm256_cmpeq_epi32(ub8_az, _mm256_set1_epi32((int)p.n)));
    }

    // ---- Cyclic wrap handling (shared between uniform and non-uniform) ----
    __m256 wrap_mask = _mm256_or_ps(mask_below, mask_above);
    __m256i az_last_i = _mm256_set1_epi32((int)(p.n - 1));
    __m256 az_wrap_rinv_v = _mm256_set1_ps(p.step_inv[0]);
    __m256 az_min_v = _mm256_set1_ps(p.fmin);
    __m256 az_max_v = _mm256_set1_ps(p.fmax);

    __m256 wrap_un_below = _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(az_min_v, az8), eps0_8), az_wrap_rinv_v);
    __m256 wrap_up_below = _mm256_sub_ps(ones_8, _mm256_min_ps(ones_8, wrap_un_below));
    __m256 wrap_up_above = _mm256_min_ps(ones_8, _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(az8, az_max_v), eps0_8), az_wrap_rinv_v));
    __m256 wrap_up = _mm256_blendv_ps(wrap_up_above, wrap_up_below, mask_below);
    wrap_up = _mm256_max_ps(zeros_8, _mm256_min_ps(ones_8, wrap_up));

    // Select between normal-range and wrap results
    *iUp8 = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(iUp_norm), _mm256_castsi256_ps(az_last_i), wrap_mask));
    *iUn8 = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(iUn_norm), _mm256_setzero_ps(), wrap_mask));
    *up8 = _mm256_blendv_ps(up_norm, wrap_up, wrap_mask);
}

// -------------------------------------------------------------------------------------
// avx2_grid_search_elevation — Vectorized elevation grid lookup with clamping
//
// For 8 elevation query values in `el8`, finds the bracketing grid indices and the
// interpolation weight. Elevation is clamped (not cyclic): values below grid min
// snap to index 0 with weight 1; values above grid max snap to the last index.
//
// Handles three sub-cases internally:
//   n <= 1:         Single grid point, no interpolation needed
//   uniform grid:   O(1) index + bottom-clamp
//   non-uniform:    Vectorized binary search + full clamp (both ends)
//
// Inputs:
//   el8     — 8 query elevation values [rad]
//   p       — precomputed grid parameters (see ElGridParams)
//
// Outputs:
//   iVp8    — lower bracket indices (8× int32), range [0, n-1]
//   iVn8    — upper bracket indices (8× int32), range [0, n-1]
//   vp8     — interpolation weight for the lower bracket, range [0, 1]
// -------------------------------------------------------------------------------------
static inline void avx2_grid_search_elevation(
    __m256 el8,
    const ElGridParams &p,
    __m256i *iVp8, __m256i *iVn8, __m256 *vp8)
{
    const __m256 ones_8 = _mm256_set1_ps(1.0f);
    const __m256 zeros_8 = _mm256_setzero_ps();
    const __m256 eps0_8 = _mm256_set1_ps(p.eps_cubed);

    // ---- Trivial case: single grid point ----
    if (p.n <= 1)
    {
        *iVp8 = _mm256_setzero_si256();
        *iVn8 = _mm256_setzero_si256();
        *vp8 = ones_8;
        return;
    }

    if (p.uniform)
    {
        // O(1) index computation for uniform grids
        const __m256 el_min_v = _mm256_set1_ps(p.fmin);
        const __m256 el_rinv_v = _mm256_set1_ps(p.step_rinv);
        const int el_max_idx = (int)(p.n - 2);

        __m256 fidx = _mm256_mul_ps(_mm256_sub_ps(el8, el_min_v), el_rinv_v);

        __m256i i_trunc = _mm256_cvttps_epi32(fidx);
        __m256 f_trunc = _mm256_cvtepi32_ps(i_trunc);
        __m256 neg_adj = _mm256_cmp_ps(fidx, f_trunc, _CMP_LT_OQ);
        __m256i i_floor = _mm256_add_epi32(i_trunc, _mm256_castps_si256(neg_adj));

        __m256i iVp_c = _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(i_floor, _mm256_set1_epi32(el_max_idx)));
        __m256i iVn_c = _mm256_add_epi32(iVp_c, _mm256_set1_epi32(1));

        __m256 grid_vn = _mm256_i32gather_ps(p.grid_f, iVn_c, 4);
        __m256 vn_c = _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(grid_vn, el8), eps0_8), el_rinv_v);
        vn_c = _mm256_max_ps(zeros_8, _mm256_min_ps(ones_8, vn_c));
        __m256 vp_c = _mm256_sub_ps(ones_8, vn_c);

        // Clamp below: if el < grid_min, force weight = 1 (snap to first point)
        __m256 mask_below = _mm256_cmp_ps(el8, el_min_v, _CMP_LT_OQ);
        vp_c = _mm256_blendv_ps(vp_c, ones_8, mask_below);
        iVn_c = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(iVn_c), _mm256_setzero_ps(), mask_below));

        *iVp8 = iVp_c;
        *iVn8 = iVn_c;
        *vp8 = vp_c;
    }
    else
    {
        // Vectorized binary search for non-uniform grids (no wrapping for elevation)
        __m256i bs_lo = _mm256_setzero_si256();
        __m256i bs_hi = _mm256_set1_epi32((int)p.n);
        for (int iter = 0; iter < p.bsearch_iters; ++iter)
        {
            __m256i mid = _mm256_srli_epi32(_mm256_add_epi32(bs_lo, bs_hi), 1);
            __m256 val = _mm256_i32gather_ps(p.grid_f, mid, 4);
            __m256 cmp = _mm256_cmp_ps(val, el8, _CMP_LE_OQ);
            __m256i cmp_i = _mm256_castps_si256(cmp);
            bs_lo = _mm256_blendv_epi8(bs_lo, _mm256_add_epi32(mid, _mm256_set1_epi32(1)), cmp_i);
            bs_hi = _mm256_blendv_epi8(mid, bs_hi, cmp_i);
        }
        __m256i ub8_el = bs_lo;

        __m256 mask_past = _mm256_castsi256_ps(
            _mm256_cmpeq_epi32(ub8_el, _mm256_set1_epi32((int)p.n)));
        __m256 mask_before = _mm256_castsi256_ps(
            _mm256_cmpeq_epi32(ub8_el, _mm256_setzero_si256()));
        __m256 clamp_mask = _mm256_or_ps(mask_past, mask_before);

        __m256i iVp_norm = _mm256_sub_epi32(ub8_el, _mm256_set1_epi32(1));
        __m256i iVn_norm = ub8_el;

        __m256i iVn_safe = _mm256_min_epi32(iVn_norm, _mm256_set1_epi32((int)(p.n - 1)));
        __m256 grid_vn = _mm256_i32gather_ps(p.grid_f, iVn_safe, 4);
        __m256 diff_vn = _mm256_i32gather_ps(p.step_inv, iVn_safe, 4);
        __m256 vn_norm = _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(grid_vn, el8), eps0_8), diff_vn);
        vn_norm = _mm256_min_ps(ones_8, vn_norm);
        __m256 vp_norm = _mm256_sub_ps(ones_8, vn_norm);

        __m256i el_last_i = _mm256_set1_epi32((int)(p.n - 1));
        __m256i iVp_past = el_last_i;
        __m256i iVn_past = el_last_i;

        iVp_norm = _mm256_max_epi32(iVp_norm, _mm256_setzero_si256());
        *iVp8 = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(iVp_norm),
            _mm256_blendv_ps(_mm256_castsi256_ps(iVp_past), _mm256_setzero_ps(), mask_before),
            clamp_mask));
        *iVn8 = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(iVn_safe),
            _mm256_blendv_ps(_mm256_castsi256_ps(iVn_past), _mm256_setzero_ps(), mask_before),
            clamp_mask));
        *vp8 = _mm256_blendv_ps(vp_norm, ones_8, clamp_mask);
    }
}

// -------------------------------------------------------------------------------------
// avx2_rotate_and_polarize — Rotation, coordinate transform, and polarization gamma
//
// Combines three stages of the interpolation pipeline that share intermediate values:
//
//   Stage B: Rotate input direction vector by antenna orientation matrix R.
//            Convert rotated Cartesian direction to local az/el via cart2geo.
//            Derive sin/cos of output angles from the Cartesian components directly
//            (avoids extra sincos calls).
//
//   Stage C: Compute input/output basis vectors (theta-hat, phi-hat) and the
//            polarization rotation angle gamma from their dot products.
//            eTHi, ePHi are the input basis vectors (from global az/el).
//            eTHo is the output basis vector (from local az/el), back-rotated by R^T
//            to the global frame for the dot product.
//
// Inputs:
//   R           — Rotation matrix (8 lanes per entry, see RMatrix8)
//   sAZi8       — sin(input_azimuth),   8 lanes
//   cAZi8       — cos(input_azimuth),   8 lanes
//   sELi8       — sin(input_elevation), 8 lanes
//   cELi8       — cos(input_elevation), 8 lanes  (must already include pole guard: += eps)
//   eps_sq_8    — squared eps for rsqrt Newton-Raphson guard
//
// Outputs:
//   dx8, dy8, dz8   — rotated direction vector (needed for distance computation)
//   az8, el8        — local azimuth/elevation in antenna coordinates [rad]
//   cos_gamma8      — cos of polarization rotation angle
//   sin_gamma8      — sin of polarization rotation angle
// -------------------------------------------------------------------------------------
static inline void avx2_rotate_and_polarize(
    const RMatrix8 &R,
    __m256 sAZi8, __m256 cAZi8, __m256 sELi8, __m256 cELi8,
    __m256 eps_sq_8,
    __m256 *dx8, __m256 *dy8, __m256 *dz8,
    __m256 *az8, __m256 *el8,
    __m256 *cos_gamma8, __m256 *sin_gamma8)
{
    // Input direction in Cartesian: [cEL*cAZ, cEL*sAZ, sEL]
    __m256 Cx8 = _mm256_mul_ps(cELi8, cAZi8);
    __m256 Cy8 = _mm256_mul_ps(cELi8, sAZi8);

    // Forward rotation: d = R * [Cx, Cy, sEL]
    *dx8 = _mm256_fmadd_ps(R.r[0], Cx8, _mm256_fmadd_ps(R.r[3], Cy8, _mm256_mul_ps(R.r[6], sELi8)));
    *dy8 = _mm256_fmadd_ps(R.r[1], Cx8, _mm256_fmadd_ps(R.r[4], Cy8, _mm256_mul_ps(R.r[7], sELi8)));
    *dz8 = _mm256_fmadd_ps(R.r[2], Cx8, _mm256_fmadd_ps(R.r[5], Cy8, _mm256_mul_ps(R.r[8], sELi8)));

    // cart2geo: local az = atan2(dy, dx), local el = asin(clamp(dz))
    _fm256_cart2geo_ps(*dx8, *dy8, *dz8, az8, el8);

    // Derive sin/cos of output angles from dx/dy/dz directly (avoids extra sincos)
    __m256 hyp_sq8 = _mm256_fmadd_ps(*dx8, *dx8, _mm256_mul_ps(*dy8, *dy8));
    __m256 safe_sq8 = _mm256_max_ps(hyp_sq8, eps_sq_8);
    __m256 inv_hypot8 = _fm256_rsqrt_nr_ps(safe_sq8);
    __m256 cAZo8 = _mm256_mul_ps(*dx8, inv_hypot8);
    __m256 sAZo8 = _mm256_mul_ps(*dy8, inv_hypot8);
    __m256 cELo8 = _mm256_mul_ps(hyp_sq8, inv_hypot8);

    // Input basis vectors: eTHi = [sEL*cAZ, sEL*sAZ, -cEL], ePHi = [-sAZ, cAZ, 0]
    __m256 eTHi_x8 = _mm256_mul_ps(sELi8, cAZi8);
    __m256 eTHi_y8 = _mm256_mul_ps(sELi8, sAZi8);
    __m256 eTHi_z8 = _mm256_xor_ps(cELi8, _mm256_set1_ps(-0.0f)); // -cELi
    __m256 ePHi_x8 = _mm256_xor_ps(sAZi8, _mm256_set1_ps(-0.0f)); // -sAZi
    __m256 ePHi_y8 = cAZi8;

    // Output basis vector: eTHo = [dz*cAZo, dz*sAZo, -cELo]
    __m256 eTHo_x8 = _mm256_mul_ps(*dz8, cAZo8);
    __m256 eTHo_y8 = _mm256_mul_ps(*dz8, sAZo8);
    __m256 eTHo_z8 = _mm256_xor_ps(cELo8, _mm256_set1_ps(-0.0f)); // -cELo

    // Back-rotate eTHo into global frame: eTHor = R^T * eTHo
    __m256 eTHor_x8 = _mm256_fmadd_ps(R.r[0], eTHo_x8, _mm256_fmadd_ps(R.r[1], eTHo_y8, _mm256_mul_ps(R.r[2], eTHo_z8)));
    __m256 eTHor_y8 = _mm256_fmadd_ps(R.r[3], eTHo_x8, _mm256_fmadd_ps(R.r[4], eTHo_y8, _mm256_mul_ps(R.r[5], eTHo_z8)));
    __m256 eTHor_z8 = _mm256_fmadd_ps(R.r[6], eTHo_x8, _mm256_fmadd_ps(R.r[7], eTHo_y8, _mm256_mul_ps(R.r[8], eTHo_z8)));

    // Gamma from dot products: cos_gamma = dot(eTHi, eTHor), sin_gamma = dot(ePHi, eTHor)
    *cos_gamma8 = _mm256_fmadd_ps(eTHi_x8, eTHor_x8, _mm256_fmadd_ps(eTHi_y8, eTHor_y8, _mm256_mul_ps(eTHi_z8, eTHor_z8)));
    *sin_gamma8 = _mm256_fmadd_ps(ePHi_x8, eTHor_x8, _mm256_mul_ps(ePHi_y8, eTHor_y8));
}

// -------------------------------------------------------------------------------------
// avx2_compute_distance — Signed projected distance from element position onto direction
//
// Computes the effective distance as the signed projection of the element position
// vector onto the (unnormalized) direction vector [dx, dy, dz].
// Result is negated to match the convention used in the channel model.
//
// Inputs:
//   dx8, dy8, dz8  — rotated direction vector (from avx2_rotate_and_polarize)
//   px8, py8, pz8  — element position vector (8 lanes)
//
// Returns:
//   dist8 — signed projected distance (8 lanes, negated)
// -------------------------------------------------------------------------------------
static inline __m256 avx2_compute_distance(
    __m256 dx8, __m256 dy8, __m256 dz8,
    __m256 px8, __m256 py8, __m256 pz8)
{
    __m256 dot8 = _mm256_fmadd_ps(dx8, px8, _mm256_fmadd_ps(dy8, py8, _mm256_mul_ps(dz8, pz8)));
    __m256 dx2_8 = _mm256_mul_ps(dx8, dx8);
    __m256 dy2_8 = _mm256_mul_ps(dy8, dy8);
    __m256 dz2_8 = _mm256_mul_ps(dz8, dz8);
    __m256 sgn8 = _fm256_signum_ps(_mm256_fmadd_ps(dot8, dx2_8, _mm256_fmadd_ps(dot8, dy2_8, _mm256_mul_ps(dot8, dz2_8))));
    __m256 dot2_8 = _mm256_mul_ps(dot8, dot8);
    return _mm256_xor_ps(
        _mm256_mul_ps(sgn8, _mm256_sqrt_ps(_mm256_fmadd_ps(dot2_8, dx2_8, _mm256_fmadd_ps(dot2_8, dy2_8, _mm256_mul_ps(dot2_8, dz2_8))))),
        _mm256_set1_ps(-0.0f)); // negate
}

// -------------------------------------------------------------------------------------
// gather_d2f — Gather 8 doubles by index, convert to 8 floats in one __m256
//
// AVX2 can only gather 4 doubles at a time, so this splits the 8 indices into
// low (lanes 0-3) and high (lanes 4-7) halves, gathers each as __m256d, converts
// to __m128 (4 floats), and recombines into a single __m256.
//
// Inputs:
//   base   — pointer to double array to gather from
//   idx_lo — indices for lanes 0-3 (4× int32)
//   idx_hi — indices for lanes 4-7 (4× int32)
//
// Returns:
//   8 gathered-and-converted float values
// -------------------------------------------------------------------------------------
static inline __m256 gather_d2f(const double *base, __m128i idx_lo, __m128i idx_hi)
{
    return _mm256_set_m128(
        _mm256_cvtpd_ps(_mm256_i32gather_pd(base, idx_hi, 8)),
        _mm256_cvtpd_ps(_mm256_i32gather_pd(base, idx_lo, 8)));
}

// -------------------------------------------------------------------------------------
// avx2_gather_slerp_polrot — Gather pattern samples, SLERP interpolation, polarization rotation
//
// Combines three tightly-coupled stages:
//
//   Stage F: Build four gather indices from grid bracket indices (iUp/iUn × iVp/iVn),
//            element pattern offset, and n_elevation stride.
//
//   Gather:  Load 4 × 2 = 8 complex pattern samples (theta + phi polarizations) at the
//            four bilinear corner points (A=Up/Vp, B=Un/Vp, C=Up/Vn, D=Un/Vn).
//            dtype-dependent: float uses direct i32gather, double splits into 4+4 gathers
//            with double→float conversion.
//
//   SLERP:   6× complex spherical interpolation to produce the bilinear-interpolated
//            theta and phi field values.
//
//   Stage G: Apply polarization rotation (Jones matrix rotation by gamma) to transform
//            from local antenna coordinates to global coordinates.
//            Optionally compute gamma = atan2(sin_gamma, cos_gamma).
//
// Inputs:
//   iUp8, iUn8     — azimuth bracket indices from grid search
//   iVp8, iVn8     — elevation bracket indices from grid search
//   up8, vp8       — interpolation weights (lower bracket)
//   offset8        — per-element pattern offset: (i_element - 1) * n_pattern_samples
//   n_elevation    — number of elevation grid points (stride for column-major pattern)
//   p_theta_re/im  — theta (vertical) pattern data, real and imaginary parts
//   p_phi_re/im    — phi (horizontal) pattern data, real and imaginary parts
//   cos_gamma8     — cos of polarization rotation angle
//   sin_gamma8     — sin of polarization rotation angle
//   compute_gamma  — if true, compute gamma8 = atan2(sin_gamma, cos_gamma)
//
// Outputs:
//   v_re8, v_im8   — interpolated + rotated vertical field (real, imaginary)
//   h_re8, h_im8   — interpolated + rotated horizontal field (real, imaginary)
//   gamma8         — polarization rotation angle [rad] (only meaningful if compute_gamma)
// -------------------------------------------------------------------------------------
template <typename dtype>
static inline void avx2_gather_slerp_polrot(
    __m256i iUp8, __m256i iUn8, __m256i iVp8, __m256i iVn8,
    __m256 up8, __m256 vp8,
    __m256i offset8, size_t n_elevation,
    const dtype *p_theta_re, const dtype *p_theta_im,
    const dtype *p_phi_re, const dtype *p_phi_im,
    __m256 cos_gamma8, __m256 sin_gamma8,
    bool compute_gamma,
    __m256 *v_re8, __m256 *v_im8, __m256 *h_re8, __m256 *h_im8,
    __m256 *gamma8)
{
    // ---- Build gather indices for the 4 bilinear corners ----
    const __m256i n_el_i = _mm256_set1_epi32((int)n_elevation);

    __m256i row_up = _mm256_add_epi32(_mm256_mullo_epi32(iUp8, n_el_i), offset8);
    __m256i row_un = _mm256_add_epi32(_mm256_mullo_epi32(iUn8, n_el_i), offset8);
    __m256i iA8 = _mm256_add_epi32(row_up, iVp8); // corner A: (az_lower, el_lower)
    __m256i iB8 = _mm256_add_epi32(row_un, iVp8); // corner B: (az_upper, el_lower)
    __m256i iC8 = _mm256_add_epi32(row_up, iVn8); // corner C: (az_lower, el_upper)
    __m256i iD8 = _mm256_add_epi32(row_un, iVn8); // corner D: (az_upper, el_upper)

    // ---- Gather pattern data at all 4 corners, both polarizations ----
    __m256 VfAr8, VfAi8, VfBr8, VfBi8, VfCr8, VfCi8, VfDr8, VfDi8;
    __m256 HfAr8, HfAi8, HfBr8, HfBi8, HfCr8, HfCi8, HfDr8, HfDi8;

    if constexpr (sizeof(dtype) == sizeof(float))
    {
        const float *pTHr = (const float *)p_theta_re, *pTHi = (const float *)p_theta_im;
        const float *pPHr = (const float *)p_phi_re, *pPHi = (const float *)p_phi_im;

        VfAr8 = _mm256_i32gather_ps(pTHr, iA8, 4);
        VfAi8 = _mm256_i32gather_ps(pTHi, iA8, 4);
        VfBr8 = _mm256_i32gather_ps(pTHr, iB8, 4);
        VfBi8 = _mm256_i32gather_ps(pTHi, iB8, 4);
        VfCr8 = _mm256_i32gather_ps(pTHr, iC8, 4);
        VfCi8 = _mm256_i32gather_ps(pTHi, iC8, 4);
        VfDr8 = _mm256_i32gather_ps(pTHr, iD8, 4);
        VfDi8 = _mm256_i32gather_ps(pTHi, iD8, 4);

        HfAr8 = _mm256_i32gather_ps(pPHr, iA8, 4);
        HfAi8 = _mm256_i32gather_ps(pPHi, iA8, 4);
        HfBr8 = _mm256_i32gather_ps(pPHr, iB8, 4);
        HfBi8 = _mm256_i32gather_ps(pPHi, iB8, 4);
        HfCr8 = _mm256_i32gather_ps(pPHr, iC8, 4);
        HfCi8 = _mm256_i32gather_ps(pPHi, iC8, 4);
        HfDr8 = _mm256_i32gather_ps(pPHr, iD8, 4);
        HfDi8 = _mm256_i32gather_ps(pPHi, iD8, 4);
    }
    else
    {
        const double *dTHr = (const double *)p_theta_re, *dTHi = (const double *)p_theta_im;
        const double *dPHr = (const double *)p_phi_re, *dPHi = (const double *)p_phi_im;

        __m128i iA_lo = _mm256_castsi256_si128(iA8), iA_hi = _mm256_extracti128_si256(iA8, 1);
        __m128i iB_lo = _mm256_castsi256_si128(iB8), iB_hi = _mm256_extracti128_si256(iB8, 1);
        __m128i iC_lo = _mm256_castsi256_si128(iC8), iC_hi = _mm256_extracti128_si256(iC8, 1);
        __m128i iD_lo = _mm256_castsi256_si128(iD8), iD_hi = _mm256_extracti128_si256(iD8, 1);

        VfAr8 = gather_d2f(dTHr, iA_lo, iA_hi);
        VfAi8 = gather_d2f(dTHi, iA_lo, iA_hi);
        VfBr8 = gather_d2f(dTHr, iB_lo, iB_hi);
        VfBi8 = gather_d2f(dTHi, iB_lo, iB_hi);
        VfCr8 = gather_d2f(dTHr, iC_lo, iC_hi);
        VfCi8 = gather_d2f(dTHi, iC_lo, iC_hi);
        VfDr8 = gather_d2f(dTHr, iD_lo, iD_hi);
        VfDi8 = gather_d2f(dTHi, iD_lo, iD_hi);

        HfAr8 = gather_d2f(dPHr, iA_lo, iA_hi);
        HfAi8 = gather_d2f(dPHi, iA_lo, iA_hi);
        HfBr8 = gather_d2f(dPHr, iB_lo, iB_hi);
        HfBi8 = gather_d2f(dPHi, iB_lo, iB_hi);
        HfCr8 = gather_d2f(dPHr, iC_lo, iC_hi);
        HfCi8 = gather_d2f(dPHi, iC_lo, iC_hi);
        HfDr8 = gather_d2f(dPHr, iD_lo, iD_hi);
        HfDi8 = gather_d2f(dPHi, iD_lo, iD_hi);
    }

    // ---- 6× SLERP: bilinear spherical interpolation ----
    // Interpolate along azimuth (A↔B, C↔D), then along elevation (E↔F)
    __m256 VEr8, VEi8, VFr8, VFi8, Vr8, Vi8;
    _fm256_slerp_complex_ps(VfAr8, VfAi8, VfBr8, VfBi8, up8, &VEr8, &VEi8);
    _fm256_slerp_complex_ps(VfCr8, VfCi8, VfDr8, VfDi8, up8, &VFr8, &VFi8);
    _fm256_slerp_complex_ps(VEr8, VEi8, VFr8, VFi8, vp8, &Vr8, &Vi8);

    __m256 HEr8, HEi8, HFr8, HFi8, Hr8, Hi8;
    _fm256_slerp_complex_ps(HfAr8, HfAi8, HfBr8, HfBi8, up8, &HEr8, &HEi8);
    _fm256_slerp_complex_ps(HfCr8, HfCi8, HfDr8, HfDi8, up8, &HFr8, &HFi8);
    _fm256_slerp_complex_ps(HEr8, HEi8, HFr8, HFi8, vp8, &Hr8, &Hi8);

    // ---- Polarization rotation (Jones matrix rotation by gamma) ----
    *v_re8 = _mm256_fmsub_ps(cos_gamma8, Vr8, _mm256_mul_ps(sin_gamma8, Hr8));
    *v_im8 = _mm256_fmsub_ps(cos_gamma8, Vi8, _mm256_mul_ps(sin_gamma8, Hi8));
    *h_re8 = _mm256_fmadd_ps(sin_gamma8, Vr8, _mm256_mul_ps(cos_gamma8, Hr8));
    *h_im8 = _mm256_fmadd_ps(sin_gamma8, Vi8, _mm256_mul_ps(cos_gamma8, Hi8));

    // Gamma angle (only if requested by caller)
    *gamma8 = _mm256_setzero_ps();
    if (compute_gamma)
        *gamma8 = _fm256_atan2256_ps(sin_gamma8, cos_gamma8);
}

// =====================================================================================
// Main interpolation function
// =====================================================================================

template <typename dtype>
void qd_arrayant_interpolate_avx2(const arma::Cube<dtype> &e_theta_re,    // Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
                                  const arma::Cube<dtype> &e_theta_im,    // Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
                                  const arma::Cube<dtype> &e_phi_re,      // Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
                                  const arma::Cube<dtype> &e_phi_im,      // Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
                                  const arma::Col<dtype> &azimuth_grid,   // Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
                                  const arma::Col<dtype> &elevation_grid, // Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
                                  const arma::Mat<dtype> &azimuth,        // Azimuth angles for interpolation in [rad],                      Size [1, n_ang] or [n_out, n_ang]
                                  const arma::Mat<dtype> &elevation,      // Elevation angles for interpolation in [rad],                    Size [1, n_ang] or [n_out, n_ang]
                                  const arma::Col<unsigned> &i_element,   // Element indices, 1-based                                        Vector of length "n_out"
                                  const arma::Cube<dtype> &orientation,   // Orientation of the array antenna (bank, tilt, head) in [rad],   Size [3, 1, 1] or [3, n_out, 1] or [3, 1, n_ang] or [3, n_out, n_ang]
                                  const arma::Mat<dtype> &element_pos,    // Element positions                                               Size [3, n_out]
                                  arma::Mat<dtype> &V_re,                 // Interpolated vertical field, real part,                         Size [n_out, n_ang]
                                  arma::Mat<dtype> &V_im,                 // Interpolated vertical field, imaginary part,                    Size [n_out, n_ang]
                                  arma::Mat<dtype> &H_re,                 // Interpolated horizontal field, real part,                       Size [n_out, n_ang]
                                  arma::Mat<dtype> &H_im,                 // Interpolated horizontal field, imaginary part,                  Size [n_out, n_ang]
                                  arma::Mat<dtype> *dist,                 // Effective distances, optional                                   Size [n_out, n_ang] or []
                                  arma::Mat<dtype> *azimuth_loc,          // Azimuth angles [rad] in local antenna coordinates, optional,    Size [n_out, n_ang] or []
                                  arma::Mat<dtype> *elevation_loc,        // Elevation angles [rad] in local antenna coordinates, optional,  Size [n_out, n_ang] or []
                                  arma::Mat<dtype> *gamma)                // Polarization rotation angles in [rad], optional,                Size [n_out, n_ang] or []

{
    const size_t n_elevation = (size_t)e_theta_re.n_rows;
    const size_t n_azimuth = (size_t)e_theta_re.n_cols;
    const size_t n_pattern_samples = n_azimuth * n_elevation;
    const size_t n_out = (size_t)i_element.n_elem;
    const size_t n_ang = (size_t)azimuth.n_cols;

    // Early return for empty inputs (avoids wraparound UB on N_total-1)
    if (n_out == 0 || n_ang == 0)
        return;

    // Grid must have at least one sample
    if (n_azimuth == 0 || n_elevation == 0)
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: Grid dimensions must be >= 1.");

    // Pattern cubes must have matching dimensions
    const size_t n_elements = (size_t)e_theta_re.n_slices;
    if (e_theta_im.n_rows != n_elevation || e_theta_im.n_cols != n_azimuth || e_theta_im.n_slices != n_elements ||
        e_phi_re.n_rows != n_elevation || e_phi_re.n_cols != n_azimuth || e_phi_re.n_slices != n_elements ||
        e_phi_im.n_rows != n_elevation || e_phi_im.n_cols != n_azimuth || e_phi_im.n_slices != n_elements)
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: Pattern cubes must have matching [n_elevation, n_azimuth, n_elements] dimensions.");

    if (azimuth_grid.n_elem != n_azimuth)
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: azimuth_grid length must equal n_azimuth.");
    if (elevation_grid.n_elem != n_elevation)
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: elevation_grid length must equal n_elevation.");

    if (azimuth.n_cols != elevation.n_cols || azimuth.n_rows != elevation.n_rows)
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: azimuth and elevation must have the same size.");

    if (orientation.n_rows != 3)
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: orientation must have 3 rows.");

    if (element_pos.n_rows != 3 || element_pos.n_cols != n_out)
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: element_pos must be [3, n_out].");

    // Validate i_element bounds (1-based, must be in [1, n_elements])
    const unsigned *p_i_element = i_element.memptr();
    for (size_t o = 0; o < n_out; ++o)
        if (p_i_element[o] < 1 || p_i_element[o] > (unsigned)n_elements)
            throw std::out_of_range("qd_arrayant_interpolate_avx2: i_element contains out-of-range index.");

    // Validate mandatory output sizes (must be pre-allocated)
    if (V_re.n_rows != n_out || V_re.n_cols != n_ang ||
        V_im.n_rows != n_out || V_im.n_cols != n_ang ||
        H_re.n_rows != n_out || H_re.n_cols != n_ang ||
        H_im.n_rows != n_out || H_im.n_cols != n_ang)
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: Output matrices V_re/V_im/H_re/H_im must be pre-allocated to [n_out, n_ang].");

    // Validate optional output sizes if provided and non-empty
    if (dist != nullptr && !dist->is_empty() && (dist->n_rows != n_out || dist->n_cols != n_ang))
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: dist must be [n_out, n_ang] if provided.");
    if (azimuth_loc != nullptr && !azimuth_loc->is_empty() && (azimuth_loc->n_rows != n_out || azimuth_loc->n_cols != n_ang))
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: azimuth_loc must be [n_out, n_ang] if provided.");
    if (elevation_loc != nullptr && !elevation_loc->is_empty() && (elevation_loc->n_rows != n_out || elevation_loc->n_cols != n_ang))
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: elevation_loc must be [n_out, n_ang] if provided.");
    if (gamma != nullptr && !gamma->is_empty() && (gamma->n_rows != n_out || gamma->n_cols != n_ang))
        throw std::invalid_argument("qd_arrayant_interpolate_avx2: gamma must be [n_out, n_ang] if provided.");

    bool per_element_angles = azimuth.n_rows > 1;
    bool per_element_rotation = orientation.n_cols > 1;
    bool per_angle_rotation = orientation.n_slices > 1;

    // Output pointers
    dtype *p_dist = (dist == nullptr || dist->is_empty()) ? nullptr : dist->memptr();
    dtype *p_azimuth_loc = (azimuth_loc == nullptr || azimuth_loc->is_empty()) ? nullptr : azimuth_loc->memptr();
    dtype *p_elevation_loc = (elevation_loc == nullptr || elevation_loc->is_empty()) ? nullptr : elevation_loc->memptr();
    dtype *p_gamma = (gamma == nullptr || gamma->is_empty()) ? nullptr : gamma->memptr();

    // Rotation matrix [3,3,n_out] or [3,3,1]
    arma::Cube<dtype> R_typed(9, orientation.n_cols, orientation.n_slices, arma::fill::none);
    qd_rotation_matrix(orientation.memptr(), R_typed.memptr(), orientation.n_cols * orientation.n_slices, true, true);

    // Obtain pointers for direct memory access
    const dtype *p_theta_re = e_theta_re.memptr(), *p_theta_im = e_theta_im.memptr();
    const dtype *p_phi_re = e_phi_re.memptr(), *p_phi_im = e_phi_im.memptr();
    const dtype *p_az_global = azimuth.memptr(), *p_el_global = elevation.memptr();
    const dtype *p_azimuth_grid = azimuth_grid.memptr(), *p_elevation_grid = elevation_grid.memptr();
    const dtype *p_element_pos = element_pos.memptr();
    dtype *p_v_re = V_re.memptr(), *p_v_im = V_im.memptr(), *p_h_re = H_re.memptr(), *p_h_im = H_im.memptr();

    // ==================================================================================
    // All computation is done in float for AVX2 throughput.
    // When dtype=double, inputs are cast to float during precomputation,
    // results are written back to dtype* outputs (implicit float->double promotion).
    // ==================================================================================

    // All constants are float — avoids double-eps vanishing below float precision
    const float two_pi = (float)arma::Datum<dtype>::tau;
    const float eps_cubed = arma::Datum<float>::eps * arma::Datum<float>::eps * arma::Datum<float>::eps; // tiny guard for weight computation
    const float eps = arma::Datum<float>::eps;                                                           // pole guard
    constexpr float one = 1.0f;

    // Reciprocal grid spacings: 1/(grid[i] - grid[i-1]), used as interpolation weight multipliers
    auto az_step_inv_buf = std::make_unique<float[]>(n_azimuth);
    auto el_step_inv_buf = std::make_unique<float[]>(n_elevation);
    float *az_step_inv = az_step_inv_buf.get(), *el_step_inv = el_step_inv_buf.get();
    *az_step_inv = two_pi - (float)p_azimuth_grid[n_azimuth - 1] + (float)*p_azimuth_grid;
    *az_step_inv = one / *az_step_inv;
    *el_step_inv = one;
    for (size_t a = 1; a < n_azimuth; ++a)
        az_step_inv[a] = one / ((float)(p_azimuth_grid[a] - p_azimuth_grid[a - 1]));
    for (size_t a = 1; a < n_elevation; ++a)
        el_step_inv[a] = one / ((float)(p_elevation_grid[a] - p_elevation_grid[a - 1]));

    // ---- Precomputation: uniform-grid detection ----
    bool az_uniform = false, el_uniform = false;
    float az_fmin = 0.0f, az_fmax = 0.0f, az_step_rinv = 0.0f;
    float el_fmin = 0.0f, el_step_rinv = 0.0f;
    int az_bsearch_iters = 0, el_bsearch_iters = 0;

    if (n_azimuth > 1)
    {
        az_fmin = (float)p_azimuth_grid[0];
        az_fmax = (float)p_azimuth_grid[n_azimuth - 1];
        float step0 = (float)(p_azimuth_grid[1] - p_azimuth_grid[0]);
        az_uniform = (step0 > 0.0f);
        for (size_t a = 2; a < n_azimuth && az_uniform; ++a)
        {
            float step_a = (float)(p_azimuth_grid[a] - p_azimuth_grid[a - 1]);
            if (std::abs(step_a - step0) > 1e-4f * step0)
                az_uniform = false;
        }
        if (az_uniform)
            az_step_rinv = 1.0f / step0;

        int tmp = (int)n_azimuth + 1;
        while ((1 << az_bsearch_iters) < tmp)
            ++az_bsearch_iters;
    }

    if (n_elevation > 1)
    {
        el_fmin = (float)p_elevation_grid[0];
        float step0 = (float)(p_elevation_grid[1] - p_elevation_grid[0]);
        el_uniform = (step0 > 0.0f);
        for (size_t a = 2; a < n_elevation && el_uniform; ++a)
        {
            float step_a = (float)(p_elevation_grid[a] - p_elevation_grid[a - 1]);
            if (std::abs(step_a - step0) > 1e-4f * step0)
                el_uniform = false;
        }
        if (el_uniform)
            el_step_rinv = 1.0f / step0;

        int tmp = (int)n_elevation + 1;
        while ((1 << el_bsearch_iters) < tmp)
            ++el_bsearch_iters;
    }

    // ---- Pack grid search parameters into structs ----
    // grid_f fields are assigned below after the grid arrays are allocated
    AzGridParams az_gp = {nullptr, az_step_inv, n_azimuth, az_uniform,
                          az_fmin, az_fmax, az_step_rinv, az_bsearch_iters, eps_cubed};
    ElGridParams el_gp = {nullptr, el_step_inv, n_elevation, el_uniform,
                          el_fmin, el_step_rinv, el_bsearch_iters, eps_cubed};
    // grid_f arrays are allocated below and assigned to the structs

    // ---- Grid arrays (standalone, small allocations) ----
    auto az_grid_f_buf = std::make_unique<float[]>(n_azimuth + 1);
    auto el_grid_f_buf = std::make_unique<float[]>(n_elevation + 1);
    float *az_grid_f = az_grid_f_buf.get();
    float *el_grid_f = el_grid_f_buf.get();

    for (size_t a = 0; a < n_azimuth; ++a)
        az_grid_f[a] = (float)p_azimuth_grid[a];
    az_grid_f[n_azimuth] = std::numeric_limits<float>::infinity();
    for (size_t a = 0; a < n_elevation; ++a)
        el_grid_f[a] = (float)p_elevation_grid[a];
    el_grid_f[n_elevation] = std::numeric_limits<float>::infinity();

    az_gp.grid_f = az_grid_f;
    el_gp.grid_f = el_grid_f;

    // ---- Per-element precompute buffers (O(n_out), fits in L1) ----
    const size_t n_out_padded = ((n_out + 7) / 8) * 8;
    const size_t n_out_vec = n_out_padded / 8;

    // SoA layout: R_elem[9 * n_out_padded], pos_elem[3 * n_out_padded]
    const size_t elem_floats = 9 * n_out_padded + 3 * n_out_padded;
    auto elem_buf_ptr = std::make_unique<float[]>(elem_floats);
    float *elem_buf = elem_buf_ptr.get();
    float *R_elem = elem_buf;
    float *pos_elem = elem_buf + 9 * n_out_padded;

    // Precomputed (ie-1)*n_pattern_samples offsets (int32)
    auto offset_elem_ptr = std::make_unique<int[]>(n_out_padded);
    int *offset_elem = offset_elem_ptr.get();

    // Fill per-element buffers (for the non-per_angle_rotation case, R is constant across angles)
    for (size_t o = 0; o < n_out; ++o)
    {
        const dtype *Rp = R_typed.slice_colptr(0, per_element_rotation ? o : 0);
        for (size_t k = 0; k < 9; ++k)
            R_elem[k * n_out_padded + o] = (float)Rp[k];

        pos_elem[0 * n_out_padded + o] = (float)p_element_pos[3 * o + 0];
        pos_elem[1 * n_out_padded + o] = (float)p_element_pos[3 * o + 1];
        pos_elem[2 * n_out_padded + o] = (float)p_element_pos[3 * o + 2];

        offset_elem[o] = (int)(p_i_element[o] - 1) * (int)n_pattern_samples;
    }
    // Pad [n_out..n_out_padded) with last valid entry
    for (size_t o = n_out; o < n_out_padded; ++o)
    {
        for (size_t k = 0; k < 9; ++k)
            R_elem[k * n_out_padded + o] = R_elem[k * n_out_padded + n_out - 1];
        pos_elem[0 * n_out_padded + o] = pos_elem[0 * n_out_padded + n_out - 1];
        pos_elem[1 * n_out_padded + o] = pos_elem[1 * n_out_padded + n_out - 1];
        pos_elem[2 * n_out_padded + o] = pos_elem[2 * n_out_padded + n_out - 1];
        offset_elem[o] = offset_elem[n_out - 1];
    }

    // ---- AVX2 constants ----
    const __m256 eps_8 = _mm256_set1_ps(eps);
    const __m256 eps_sq_8 = _mm256_set1_ps(eps * eps); // squared eps for rsqrt guard

    // ---- Tail mask for masked stores (based on n_out % 8) ----
    const int tail = (int)(n_out % 8);
    __m256i tail_mask;
    if (tail == 0)
        tail_mask = _mm256_set1_epi32(-1); // all lanes active
    else
    {
        alignas(32) int32_t tm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int j = 0; j < tail; ++j)
            tm[j] = -1;
        tail_mask = _mm256_load_si256((__m256i *)tm);
    }
    const size_t n_valid_tail = (tail == 0) ? 8 : (size_t)tail;

    // ==================================================================================
    // ELEMENT-VECTORIZED PATH: n_out >= 8, !per_angle_rotation, !per_element_angles
    // Two-level loop: outer over angles (OMP), inner over elements (AVX2, 8 elements/vector)
    // ==================================================================================
    if (!per_angle_rotation && !per_element_angles && n_out >= 8)
    {
#pragma omp parallel for schedule(dynamic, 1) if (n_ang >= 4 && n_ang * n_out >= 512)
        for (long long a = 0; a < (long long)n_ang; ++a)
        {
            // Output base pointers for this angle's column (column-major: column a starts at offset a*n_out)
            dtype *out_vr = p_v_re + a * n_out;
            dtype *out_vi = p_v_im + a * n_out;
            dtype *out_hr = p_h_re + a * n_out;
            dtype *out_hi = p_h_im + a * n_out;
            dtype *out_dist = p_dist ? p_dist + a * n_out : nullptr;
            dtype *out_az = p_azimuth_loc ? p_azimuth_loc + a * n_out : nullptr;
            dtype *out_el = p_elevation_loc ? p_elevation_loc + a * n_out : nullptr;
            dtype *out_gam = p_gamma ? p_gamma + a * n_out : nullptr;

            // ---- Angle-dependent data (computed once per angle, broadcast) ----
            float az_a = (float)p_az_global[a];
            float el_a = (float)p_el_global[a];

            float sAZi_s = std::sin(az_a), cAZi_s = std::cos(az_a);
            float sELi_s = std::sin(el_a), cELi_s = std::cos(el_a);
            cELi_s += eps; // pole guard

            __m256 sAZi8 = _mm256_set1_ps(sAZi_s);
            __m256 cAZi8 = _mm256_set1_ps(cAZi_s);
            __m256 sELi8 = _mm256_set1_ps(sELi_s);
            __m256 cELi8 = _mm256_set1_ps(cELi_s);

            // ---- Inner element loop (AVX2) ----
            for (size_t ov = 0; ov < n_out_vec; ++ov)
            {
                const size_t o_base = ov * 8;
                const bool is_tail_iter = (ov == n_out_vec - 1) && (tail != 0);
                const size_t n_valid = is_tail_iter ? n_valid_tail : 8;

                // Load rotation matrix from per-element SoA buffer
                RMatrix8 R;
                for (int k = 0; k < 9; ++k)
                    R.r[k] = _mm256_loadu_ps(&R_elem[k * n_out_padded + o_base]);

                // ==== Stages B+C: Rotation, cart2geo, basis vectors, gamma ====
                __m256 dx8, dy8, dz8, az8, el8, cos_gamma8, sin_gamma8;
                avx2_rotate_and_polarize(R, sAZi8, cAZi8, sELi8, cELi8, eps_sq_8,
                                         &dx8, &dy8, &dz8, &az8, &el8, &cos_gamma8, &sin_gamma8);

                // ==== Stage D: Distance ====
                __m256 dist8 = _mm256_setzero_ps();
                if (p_dist != nullptr)
                {
                    __m256 px8 = _mm256_loadu_ps(&pos_elem[0 * n_out_padded + o_base]);
                    __m256 py8 = _mm256_loadu_ps(&pos_elem[1 * n_out_padded + o_base]);
                    __m256 pz8 = _mm256_loadu_ps(&pos_elem[2 * n_out_padded + o_base]);
                    dist8 = avx2_compute_distance(dx8, dy8, dz8, px8, py8, pz8);
                }

                // ==== Stage E: Grid search ====
                __m256i iUp8, iUn8, iVp8, iVn8;
                __m256 up8, vp8;
                avx2_grid_search_azimuth(az8, az_gp, &iUp8, &iUn8, &up8);
                avx2_grid_search_elevation(el8, el_gp, &iVp8, &iVn8, &vp8);

                // ==== Stages F+G: Gather, SLERP, polarization rotation ====
                __m256i offset8 = _mm256_loadu_si256((const __m256i *)&offset_elem[o_base]);
                __m256 v_re8, v_im8, h_re8, h_im8, gamma8;
                avx2_gather_slerp_polrot<dtype>(
                    iUp8, iUn8, iVp8, iVn8, up8, vp8, offset8, n_elevation,
                    p_theta_re, p_theta_im, p_phi_re, p_phi_im,
                    cos_gamma8, sin_gamma8, p_gamma != nullptr,
                    &v_re8, &v_im8, &h_re8, &h_im8, &gamma8);

                // ==== Stores (dtype-dependent) ====
                if constexpr (sizeof(dtype) == sizeof(float))
                {
                    float *fvr = (float *)out_vr, *fvi = (float *)out_vi;
                    float *fhr = (float *)out_hr, *fhi = (float *)out_hi;

                    if (is_tail_iter)
                    {
                        _mm256_maskstore_ps(&fvr[o_base], tail_mask, v_re8);
                        _mm256_maskstore_ps(&fvi[o_base], tail_mask, v_im8);
                        _mm256_maskstore_ps(&fhr[o_base], tail_mask, h_re8);
                        _mm256_maskstore_ps(&fhi[o_base], tail_mask, h_im8);
                        if (out_dist)
                            _mm256_maskstore_ps((float *)out_dist + o_base, tail_mask, dist8);
                        if (out_az)
                            _mm256_maskstore_ps((float *)out_az + o_base, tail_mask, az8);
                        if (out_el)
                            _mm256_maskstore_ps((float *)out_el + o_base, tail_mask, el8);
                        if (out_gam)
                            _mm256_maskstore_ps((float *)out_gam + o_base, tail_mask, gamma8);
                    }
                    else
                    {
                        _mm256_storeu_ps(&fvr[o_base], v_re8);
                        _mm256_storeu_ps(&fvi[o_base], v_im8);
                        _mm256_storeu_ps(&fhr[o_base], h_re8);
                        _mm256_storeu_ps(&fhi[o_base], h_im8);
                        if (out_dist)
                            _mm256_storeu_ps((float *)out_dist + o_base, dist8);
                        if (out_az)
                            _mm256_storeu_ps((float *)out_az + o_base, az8);
                        if (out_el)
                            _mm256_storeu_ps((float *)out_el + o_base, el8);
                        if (out_gam)
                            _mm256_storeu_ps((float *)out_gam + o_base, gamma8);
                    }
                }
                else
                {
                    // Double stores: convert float→double, store as __m256d pairs
#define STORE_F2D(dst, src, base)                                                                       \
    do                                                                                                  \
    {                                                                                                   \
        _mm256_storeu_pd((double *)(dst) + (base), _mm256_cvtps_pd(_mm256_castps256_ps128(src)));       \
        _mm256_storeu_pd((double *)(dst) + (base) + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(src, 1))); \
    } while (0)

                    if (is_tail_iter)
                    {
                        alignas(32) float tmp_vr[8], tmp_vi[8], tmp_hr[8], tmp_hi[8];
                        alignas(32) float tmp_dist[8], tmp_az[8], tmp_el[8], tmp_gam[8];
                        _mm256_store_ps(tmp_vr, v_re8);
                        _mm256_store_ps(tmp_vi, v_im8);
                        _mm256_store_ps(tmp_hr, h_re8);
                        _mm256_store_ps(tmp_hi, h_im8);
                        if (out_dist)
                            _mm256_store_ps(tmp_dist, dist8);
                        _mm256_store_ps(tmp_az, az8);
                        _mm256_store_ps(tmp_el, el8);
                        if (out_gam)
                            _mm256_store_ps(tmp_gam, gamma8);

                        for (size_t j = 0; j < n_valid; ++j)
                        {
                            out_vr[o_base + j] = (dtype)tmp_vr[j];
                            out_vi[o_base + j] = (dtype)tmp_vi[j];
                            out_hr[o_base + j] = (dtype)tmp_hr[j];
                            out_hi[o_base + j] = (dtype)tmp_hi[j];
                            if (out_dist)
                                out_dist[o_base + j] = (dtype)tmp_dist[j];
                            if (out_az)
                                out_az[o_base + j] = (dtype)tmp_az[j];
                            if (out_el)
                                out_el[o_base + j] = (dtype)tmp_el[j];
                            if (out_gam)
                                out_gam[o_base + j] = (dtype)tmp_gam[j];
                        }
                    }
                    else
                    {
                        STORE_F2D(out_vr, v_re8, o_base);
                        STORE_F2D(out_vi, v_im8, o_base);
                        STORE_F2D(out_hr, h_re8, o_base);
                        STORE_F2D(out_hi, h_im8, o_base);
                        if (out_dist)
                            STORE_F2D(out_dist, dist8, o_base);
                        if (out_az)
                            STORE_F2D(out_az, az8, o_base);
                        if (out_el)
                            STORE_F2D(out_el, el8, o_base);
                        if (out_gam)
                            STORE_F2D(out_gam, gamma8, o_base);
                    }

#undef STORE_F2D
                }
            } // end inner element loop
        } // end outer angle loop
    }
    // ==================================================================================
    // ANGLE-VECTORIZED PATH: n_out < 8, !per_angle_rotation, !per_element_angles
    // Outer loop over elements (1–7 iters), inner loop over angle chunks (AVX2, 8 angles/vector)
    // R, position, offset are broadcast (constant for one element across all angles).
    // sincos is vectorized across 8 angles in the inner loop.
    // ==================================================================================
    else if (!per_angle_rotation && !per_element_angles)
    {
        const size_t n_ang_padded = ((n_ang + 7) / 8) * 8;
        const long long n_ang_vec = (long long)(n_ang_padded / 8);
        const int ang_tail = (int)(n_ang % 8);
        __m256i ang_tail_mask;
        if (ang_tail == 0)
            ang_tail_mask = _mm256_set1_epi32(-1);
        else
        {
            alignas(32) int32_t atm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int j = 0; j < ang_tail; ++j)
                atm[j] = -1;
            ang_tail_mask = _mm256_load_si256((__m256i *)atm);
        }
        const size_t n_valid_ang_tail = (ang_tail == 0) ? 8 : (size_t)ang_tail;

        for (size_t o = 0; o < n_out; ++o)
        {
            // Broadcast per-element data across 8 angle lanes
            RMatrix8 R;
            for (int k = 0; k < 9; ++k)
                R.r[k] = _mm256_set1_ps(R_elem[k * n_out_padded + o]);

            __m256 px_bc = _mm256_set1_ps(pos_elem[0 * n_out_padded + o]);
            __m256 py_bc = _mm256_set1_ps(pos_elem[1 * n_out_padded + o]);
            __m256 pz_bc = _mm256_set1_ps(pos_elem[2 * n_out_padded + o]);

            __m256i offset_bc = _mm256_set1_epi32(offset_elem[o]);

            // Inner loop over angle chunks of 8 (AVX2)
#pragma omp parallel for schedule(static) if (n_ang_vec >= 16)
            for (long long av = 0; av < n_ang_vec; ++av)
            {
                const size_t a_base = (size_t)av * 8;
                const bool is_ang_tail_iter = (av == n_ang_vec - 1) && (ang_tail != 0);
                const size_t n_valid = is_ang_tail_iter ? n_valid_ang_tail : 8;

                // ==== Stage A: Load 8 input angles (contiguous in [1, n_ang] layout) ====
                __m256 az_in8, el_in8;
                if constexpr (sizeof(dtype) == sizeof(float))
                {
                    if (is_ang_tail_iter)
                    {
                        az_in8 = _mm256_maskload_ps((const float *)p_az_global + a_base, ang_tail_mask);
                        el_in8 = _mm256_maskload_ps((const float *)p_el_global + a_base, ang_tail_mask);
                    }
                    else
                    {
                        az_in8 = _mm256_loadu_ps((const float *)p_az_global + a_base);
                        el_in8 = _mm256_loadu_ps((const float *)p_el_global + a_base);
                    }
                }
                else
                {
                    if (is_ang_tail_iter)
                    {
                        alignas(32) float az_tmp[8] = {0}, el_tmp[8] = {0};
                        for (size_t j = 0; j < n_valid; ++j)
                        {
                            az_tmp[j] = (float)p_az_global[a_base + j];
                            el_tmp[j] = (float)p_el_global[a_base + j];
                        }
                        az_in8 = _mm256_load_ps(az_tmp);
                        el_in8 = _mm256_load_ps(el_tmp);
                    }
                    else
                    {
                        __m256d az_lo = _mm256_loadu_pd((const double *)p_az_global + a_base);
                        __m256d az_hi = _mm256_loadu_pd((const double *)p_az_global + a_base + 4);
                        __m256d el_lo = _mm256_loadu_pd((const double *)p_el_global + a_base);
                        __m256d el_hi = _mm256_loadu_pd((const double *)p_el_global + a_base + 4);
                        az_in8 = _mm256_set_m128(_mm256_cvtpd_ps(az_hi), _mm256_cvtpd_ps(az_lo));
                        el_in8 = _mm256_set_m128(_mm256_cvtpd_ps(el_hi), _mm256_cvtpd_ps(el_lo));
                    }
                }

                __m256 sAZi8, cAZi8, sELi8, cELi8;
                _fm256_sincos256_ps(az_in8, &sAZi8, &cAZi8);
                _fm256_sincos256_ps(el_in8, &sELi8, &cELi8);
                cELi8 = _mm256_add_ps(cELi8, eps_8); // pole guard

                // ==== Stages B+C: Rotation, cart2geo, basis vectors, gamma ====
                __m256 dx8, dy8, dz8, az8, el8, cos_gamma8, sin_gamma8;
                avx2_rotate_and_polarize(R, sAZi8, cAZi8, sELi8, cELi8, eps_sq_8,
                                         &dx8, &dy8, &dz8, &az8, &el8, &cos_gamma8, &sin_gamma8);

                // ==== Stage D: Distance ====
                __m256 dist8 = _mm256_setzero_ps();
                if (p_dist != nullptr)
                    dist8 = avx2_compute_distance(dx8, dy8, dz8, px_bc, py_bc, pz_bc);

                // ==== Stage E: Grid search ====
                __m256i iUp8, iUn8, iVp8, iVn8;
                __m256 up8, vp8;
                avx2_grid_search_azimuth(az8, az_gp, &iUp8, &iUn8, &up8);
                avx2_grid_search_elevation(el8, el_gp, &iVp8, &iVn8, &vp8);

                // ==== Stages F+G: Gather, SLERP, polarization rotation ====
                __m256 v_re8, v_im8, h_re8, h_im8, gamma8;
                avx2_gather_slerp_polrot<dtype>(
                    iUp8, iUn8, iVp8, iVn8, up8, vp8, offset_bc, n_elevation,
                    p_theta_re, p_theta_im, p_phi_re, p_phi_im,
                    cos_gamma8, sin_gamma8, p_gamma != nullptr,
                    &v_re8, &v_im8, &h_re8, &h_im8, &gamma8);

                // ==== Stores: angle-vectorized, strided by n_out ====
                // Output layout [n_out, n_ang] col-major: element o, angle a → index o + a * n_out
                // For n_out == 1: contiguous → vector store
                // For n_out > 1: extract to temp, scalar scatter

                alignas(32) float tmp_vr[8], tmp_vi[8], tmp_hr[8], tmp_hi[8];
                alignas(32) float tmp_d[8], tmp_a[8], tmp_e[8], tmp_g[8];

                _mm256_store_ps(tmp_vr, v_re8);
                _mm256_store_ps(tmp_vi, v_im8);
                _mm256_store_ps(tmp_hr, h_re8);
                _mm256_store_ps(tmp_hi, h_im8);
                if (p_dist)
                    _mm256_store_ps(tmp_d, dist8);
                _mm256_store_ps(tmp_a, az8);
                _mm256_store_ps(tmp_e, el8);
                if (p_gamma)
                    _mm256_store_ps(tmp_g, gamma8);

                if (n_out == 1)
                {
                    // Contiguous stores (stride == 1)
                    if constexpr (sizeof(dtype) == sizeof(float))
                    {
                        if (is_ang_tail_iter)
                        {
                            _mm256_maskstore_ps((float *)p_v_re + a_base, ang_tail_mask, v_re8);
                            _mm256_maskstore_ps((float *)p_v_im + a_base, ang_tail_mask, v_im8);
                            _mm256_maskstore_ps((float *)p_h_re + a_base, ang_tail_mask, h_re8);
                            _mm256_maskstore_ps((float *)p_h_im + a_base, ang_tail_mask, h_im8);
                            if (p_dist)
                                _mm256_maskstore_ps((float *)p_dist + a_base, ang_tail_mask, dist8);
                            if (p_azimuth_loc)
                                _mm256_maskstore_ps((float *)p_azimuth_loc + a_base, ang_tail_mask, az8);
                            if (p_elevation_loc)
                                _mm256_maskstore_ps((float *)p_elevation_loc + a_base, ang_tail_mask, el8);
                            if (p_gamma)
                                _mm256_maskstore_ps((float *)p_gamma + a_base, ang_tail_mask, gamma8);
                        }
                        else
                        {
                            _mm256_storeu_ps((float *)p_v_re + a_base, v_re8);
                            _mm256_storeu_ps((float *)p_v_im + a_base, v_im8);
                            _mm256_storeu_ps((float *)p_h_re + a_base, h_re8);
                            _mm256_storeu_ps((float *)p_h_im + a_base, h_im8);
                            if (p_dist)
                                _mm256_storeu_ps((float *)p_dist + a_base, dist8);
                            if (p_azimuth_loc)
                                _mm256_storeu_ps((float *)p_azimuth_loc + a_base, az8);
                            if (p_elevation_loc)
                                _mm256_storeu_ps((float *)p_elevation_loc + a_base, el8);
                            if (p_gamma)
                                _mm256_storeu_ps((float *)p_gamma + a_base, gamma8);
                        }
                    }
                    else
                    {
                        // double, n_out==1: contiguous, convert+store
#define STORE_F2D_AV(dst, src, base)                                                                    \
    do                                                                                                  \
    {                                                                                                   \
        _mm256_storeu_pd((double *)(dst) + (base), _mm256_cvtps_pd(_mm256_castps256_ps128(src)));       \
        _mm256_storeu_pd((double *)(dst) + (base) + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(src, 1))); \
    } while (0)

                        if (is_ang_tail_iter)
                        {
                            for (size_t j = 0; j < n_valid; ++j)
                            {
                                p_v_re[a_base + j] = (dtype)tmp_vr[j];
                                p_v_im[a_base + j] = (dtype)tmp_vi[j];
                                p_h_re[a_base + j] = (dtype)tmp_hr[j];
                                p_h_im[a_base + j] = (dtype)tmp_hi[j];
                                if (p_dist)
                                    p_dist[a_base + j] = (dtype)tmp_d[j];
                                if (p_azimuth_loc)
                                    p_azimuth_loc[a_base + j] = (dtype)tmp_a[j];
                                if (p_elevation_loc)
                                    p_elevation_loc[a_base + j] = (dtype)tmp_e[j];
                                if (p_gamma)
                                    p_gamma[a_base + j] = (dtype)tmp_g[j];
                            }
                        }
                        else
                        {
                            STORE_F2D_AV(p_v_re, v_re8, a_base);
                            STORE_F2D_AV(p_v_im, v_im8, a_base);
                            STORE_F2D_AV(p_h_re, h_re8, a_base);
                            STORE_F2D_AV(p_h_im, h_im8, a_base);
                            if (p_dist)
                                STORE_F2D_AV(p_dist, dist8, a_base);
                            if (p_azimuth_loc)
                                STORE_F2D_AV(p_azimuth_loc, az8, a_base);
                            if (p_elevation_loc)
                                STORE_F2D_AV(p_elevation_loc, el8, a_base);
                            if (p_gamma)
                                STORE_F2D_AV(p_gamma, gamma8, a_base);
                        }

#undef STORE_F2D_AV
                    }
                }
                else
                {
                    // Strided stores (n_out = 2..7): scalar extraction
                    const size_t n_store = is_ang_tail_iter ? n_valid : 8;
                    for (size_t j = 0; j < n_store; ++j)
                    {
                        const size_t idx = o + (a_base + j) * n_out;
                        p_v_re[idx] = (dtype)tmp_vr[j];
                        p_v_im[idx] = (dtype)tmp_vi[j];
                        p_h_re[idx] = (dtype)tmp_hr[j];
                        p_h_im[idx] = (dtype)tmp_hi[j];
                        if (p_dist)
                            p_dist[idx] = (dtype)tmp_d[j];
                        if (p_azimuth_loc)
                            p_azimuth_loc[idx] = (dtype)tmp_a[j];
                        if (p_elevation_loc)
                            p_elevation_loc[idx] = (dtype)tmp_e[j];
                        if (p_gamma)
                            p_gamma[idx] = (dtype)tmp_g[j];
                    }
                }
            } // end inner angle loop
        } // end outer element loop
    }
    // ==================================================================================
    // GENERAL PATH: per_angle_rotation || per_element_angles
    // Two-level loop without angle hoisting (sincos computed per vector iteration)
    // ==================================================================================
    else
    {
        // For per_angle_rotation with OMP, each thread needs its own R buffer
#pragma omp parallel for schedule(dynamic, 1) if (n_ang >= 4 && n_ang * n_out >= 512)
        for (long long a = 0; a < (long long)n_ang; ++a)
        {
            // Output base pointers for this angle's column
            dtype *out_vr = p_v_re + a * n_out;
            dtype *out_vi = p_v_im + a * n_out;
            dtype *out_hr = p_h_re + a * n_out;
            dtype *out_hi = p_h_im + a * n_out;
            dtype *out_dist = p_dist ? p_dist + a * n_out : nullptr;
            dtype *out_az = p_azimuth_loc ? p_azimuth_loc + a * n_out : nullptr;
            dtype *out_el = p_elevation_loc ? p_elevation_loc + a * n_out : nullptr;
            dtype *out_gam = p_gamma ? p_gamma + a * n_out : nullptr;

            // Determine which R buffer to use for this angle
            float *R_this = R_elem; // default: per-element R from precompute
            // Thread-local R buffer for per_angle_rotation
            // Stack allocation for typical sizes, heap for large n_out
            float R_local[9 * 64];
            std::unique_ptr<float[]> R_heap;
            float *R_work;
            if (n_out_padded <= 64)
                R_work = R_local;
            else if (per_angle_rotation)
            {
                R_heap = std::make_unique<float[]>(9 * n_out_padded);
                R_work = R_heap.get();
            }
            else
                R_work = nullptr; // unused

            if (per_angle_rotation)
            {
                // Rebuild R buffer for this angle
                for (size_t o = 0; o < n_out; ++o)
                {
                    size_t Rp_o = per_element_rotation ? o : 0;
                    const dtype *Rp = R_typed.slice_colptr((size_t)a, Rp_o);
                    for (size_t k = 0; k < 9; ++k)
                        R_work[k * n_out_padded + o] = (float)Rp[k];
                }
                for (size_t o = n_out; o < n_out_padded; ++o)
                    for (size_t k = 0; k < 9; ++k)
                        R_work[k * n_out_padded + o] = R_work[k * n_out_padded + n_out - 1];
                R_this = R_work;
            }

            // ---- Inner element loop (AVX2) ----
            for (size_t ov = 0; ov < n_out_vec; ++ov)
            {
                const size_t o_base = ov * 8;
                const bool is_tail_iter = (ov == n_out_vec - 1) && (tail != 0);
                const size_t n_valid = is_tail_iter ? n_valid_tail : 8;

                // ==== Stage A: Input Angles -> sincos ====
                __m256 az_in8, el_in8;
                if (per_element_angles)
                {
                    // Load per-element angles for this angle index
                    // azimuth layout: [n_out, n_ang], column a at offset a*n_out
                    const dtype *az_col = p_az_global + (size_t)a * n_out;
                    const dtype *el_col = p_el_global + (size_t)a * n_out;
                    if constexpr (sizeof(dtype) == sizeof(float))
                    {
                        az_in8 = _mm256_loadu_ps((const float *)az_col + o_base);
                        el_in8 = _mm256_loadu_ps((const float *)el_col + o_base);
                    }
                    else
                    {
                        // double→float conversion for 8 elements
                        __m256d az_lo = _mm256_loadu_pd((const double *)az_col + o_base);
                        __m256d az_hi = _mm256_loadu_pd((const double *)az_col + o_base + 4);
                        __m256d el_lo = _mm256_loadu_pd((const double *)el_col + o_base);
                        __m256d el_hi = _mm256_loadu_pd((const double *)el_col + o_base + 4);
                        az_in8 = _mm256_set_m128(_mm256_cvtpd_ps(az_hi), _mm256_cvtpd_ps(az_lo));
                        el_in8 = _mm256_set_m128(_mm256_cvtpd_ps(el_hi), _mm256_cvtpd_ps(el_lo));
                    }
                }
                else
                {
                    az_in8 = _mm256_set1_ps((float)p_az_global[a]);
                    el_in8 = _mm256_set1_ps((float)p_el_global[a]);
                }

                __m256 sAZi8, cAZi8, sELi8, cELi8;
                _fm256_sincos256_ps(az_in8, &sAZi8, &cAZi8);
                _fm256_sincos256_ps(el_in8, &sELi8, &cELi8);
                cELi8 = _mm256_add_ps(cELi8, eps_8); // pole guard

                // Load rotation matrix from SoA buffer
                RMatrix8 R;
                for (int k = 0; k < 9; ++k)
                    R.r[k] = _mm256_loadu_ps(&R_this[k * n_out_padded + o_base]);

                // ==== Stages B+C: Rotation, cart2geo, basis vectors, gamma ====
                __m256 dx8, dy8, dz8, az8, el8, cos_gamma8, sin_gamma8;
                avx2_rotate_and_polarize(R, sAZi8, cAZi8, sELi8, cELi8, eps_sq_8,
                                         &dx8, &dy8, &dz8, &az8, &el8, &cos_gamma8, &sin_gamma8);

                // ==== Stage D: Distance ====
                __m256 dist8 = _mm256_setzero_ps();
                if (p_dist != nullptr)
                {
                    __m256 px8 = _mm256_loadu_ps(&pos_elem[0 * n_out_padded + o_base]);
                    __m256 py8 = _mm256_loadu_ps(&pos_elem[1 * n_out_padded + o_base]);
                    __m256 pz8 = _mm256_loadu_ps(&pos_elem[2 * n_out_padded + o_base]);
                    dist8 = avx2_compute_distance(dx8, dy8, dz8, px8, py8, pz8);
                }

                // ==== Stage E: Grid search ====
                __m256i iUp8, iUn8, iVp8, iVn8;
                __m256 up8, vp8;
                avx2_grid_search_azimuth(az8, az_gp, &iUp8, &iUn8, &up8);
                avx2_grid_search_elevation(el8, el_gp, &iVp8, &iVn8, &vp8);

                // ==== Stages F+G: Gather, SLERP, polarization rotation ====
                __m256i offset8 = _mm256_loadu_si256((const __m256i *)&offset_elem[o_base]);
                __m256 v_re8, v_im8, h_re8, h_im8, gamma8;
                avx2_gather_slerp_polrot<dtype>(
                    iUp8, iUn8, iVp8, iVn8, up8, vp8, offset8, n_elevation,
                    p_theta_re, p_theta_im, p_phi_re, p_phi_im,
                    cos_gamma8, sin_gamma8, p_gamma != nullptr,
                    &v_re8, &v_im8, &h_re8, &h_im8, &gamma8);

                // ==== Stores (dtype-dependent) ====
                if constexpr (sizeof(dtype) == sizeof(float))
                {
                    float *fvr = (float *)out_vr, *fvi = (float *)out_vi;
                    float *fhr = (float *)out_hr, *fhi = (float *)out_hi;

                    if (is_tail_iter)
                    {
                        _mm256_maskstore_ps(&fvr[o_base], tail_mask, v_re8);
                        _mm256_maskstore_ps(&fvi[o_base], tail_mask, v_im8);
                        _mm256_maskstore_ps(&fhr[o_base], tail_mask, h_re8);
                        _mm256_maskstore_ps(&fhi[o_base], tail_mask, h_im8);
                        if (out_dist)
                            _mm256_maskstore_ps((float *)out_dist + o_base, tail_mask, dist8);
                        if (out_az)
                            _mm256_maskstore_ps((float *)out_az + o_base, tail_mask, az8);
                        if (out_el)
                            _mm256_maskstore_ps((float *)out_el + o_base, tail_mask, el8);
                        if (out_gam)
                            _mm256_maskstore_ps((float *)out_gam + o_base, tail_mask, gamma8);
                    }
                    else
                    {
                        _mm256_storeu_ps(&fvr[o_base], v_re8);
                        _mm256_storeu_ps(&fvi[o_base], v_im8);
                        _mm256_storeu_ps(&fhr[o_base], h_re8);
                        _mm256_storeu_ps(&fhi[o_base], h_im8);
                        if (out_dist)
                            _mm256_storeu_ps((float *)out_dist + o_base, dist8);
                        if (out_az)
                            _mm256_storeu_ps((float *)out_az + o_base, az8);
                        if (out_el)
                            _mm256_storeu_ps((float *)out_el + o_base, el8);
                        if (out_gam)
                            _mm256_storeu_ps((float *)out_gam + o_base, gamma8);
                    }
                }
                else
                {
#define STORE_F2D(dst, src, base)                                                                       \
    do                                                                                                  \
    {                                                                                                   \
        _mm256_storeu_pd((double *)(dst) + (base), _mm256_cvtps_pd(_mm256_castps256_ps128(src)));       \
        _mm256_storeu_pd((double *)(dst) + (base) + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(src, 1))); \
    } while (0)

                    if (is_tail_iter)
                    {
                        alignas(32) float tmp_vr[8], tmp_vi[8], tmp_hr[8], tmp_hi[8];
                        alignas(32) float tmp_dist[8], tmp_az[8], tmp_el[8], tmp_gam[8];
                        _mm256_store_ps(tmp_vr, v_re8);
                        _mm256_store_ps(tmp_vi, v_im8);
                        _mm256_store_ps(tmp_hr, h_re8);
                        _mm256_store_ps(tmp_hi, h_im8);
                        if (out_dist)
                            _mm256_store_ps(tmp_dist, dist8);
                        _mm256_store_ps(tmp_az, az8);
                        _mm256_store_ps(tmp_el, el8);
                        if (out_gam)
                            _mm256_store_ps(tmp_gam, gamma8);

                        for (size_t j = 0; j < n_valid; ++j)
                        {
                            out_vr[o_base + j] = (dtype)tmp_vr[j];
                            out_vi[o_base + j] = (dtype)tmp_vi[j];
                            out_hr[o_base + j] = (dtype)tmp_hr[j];
                            out_hi[o_base + j] = (dtype)tmp_hi[j];
                            if (out_dist)
                                out_dist[o_base + j] = (dtype)tmp_dist[j];
                            if (out_az)
                                out_az[o_base + j] = (dtype)tmp_az[j];
                            if (out_el)
                                out_el[o_base + j] = (dtype)tmp_el[j];
                            if (out_gam)
                                out_gam[o_base + j] = (dtype)tmp_gam[j];
                        }
                    }
                    else
                    {
                        STORE_F2D(out_vr, v_re8, o_base);
                        STORE_F2D(out_vi, v_im8, o_base);
                        STORE_F2D(out_hr, h_re8, o_base);
                        STORE_F2D(out_hi, h_im8, o_base);
                        if (out_dist)
                            STORE_F2D(out_dist, dist8, o_base);
                        if (out_az)
                            STORE_F2D(out_az, az8, o_base);
                        if (out_el)
                            STORE_F2D(out_el, el8, o_base);
                        if (out_gam)
                            STORE_F2D(out_gam, gamma8, o_base);
                    }

#undef STORE_F2D
                }
            } // end inner element loop
        } // end outer angle loop (general path)
    }
}

// Declare templates
template void qd_arrayant_interpolate_avx2(const arma::Cube<float> &e_theta_re, const arma::Cube<float> &e_theta_im,
                                           const arma::Cube<float> &e_phi_re, const arma::Cube<float> &e_phi_im,
                                           const arma::Col<float> &azimuth_grid, const arma::Col<float> &elevation_grid,
                                           const arma::Mat<float> &azimuth, const arma::Mat<float> &elevation,
                                           const arma::Col<unsigned> &i_element, const arma::Cube<float> &orientation,
                                           const arma::Mat<float> &element_pos,
                                           arma::Mat<float> &V_re, arma::Mat<float> &V_im,
                                           arma::Mat<float> &H_re, arma::Mat<float> &H_im,
                                           arma::Mat<float> *dist,
                                           arma::Mat<float> *azimuth_loc, arma::Mat<float> *elevation_loc, arma::Mat<float> *gamma);

template void qd_arrayant_interpolate_avx2(const arma::Cube<double> &e_theta_re, const arma::Cube<double> &e_theta_im,
                                           const arma::Cube<double> &e_phi_re, const arma::Cube<double> &e_phi_im,
                                           const arma::Col<double> &azimuth_grid, const arma::Col<double> &elevation_grid,
                                           const arma::Mat<double> &azimuth, const arma::Mat<double> &elevation,
                                           const arma::Col<unsigned> &i_element, const arma::Cube<double> &orientation,
                                           const arma::Mat<double> &element_pos,
                                           arma::Mat<double> &V_re, arma::Mat<double> &V_im,
                                           arma::Mat<double> &H_re, arma::Mat<double> &H_im,
                                           arma::Mat<double> *dist,
                                           arma::Mat<double> *azimuth_loc, arma::Mat<double> *elevation_loc, arma::Mat<double> *gamma);