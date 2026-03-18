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

        // Binary search iteration count: ceil(log2(n_azimuth + 1))
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

    // ---- Precomputation: collapsed flat loop dimensions ----
    const size_t N_total = n_ang * n_out;
    const size_t N_padded = ((N_total + 7) / 8) * 8;
    const size_t n_vec = N_padded / 8;

    // ---- Precomputation: allocate flat arrays (single allocation) ----
    // Layout (all float, SoA):
    //   Offset  0*N_padded:  az_flat    [N_padded]       input azimuth angles
    //   Offset  1*N_padded:  el_flat    [N_padded]       input elevation angles
    //   Offset  2*N_padded:  R_flat     [9*N_padded]     rotation matrices (9 rows, SoA)
    //   Offset 11*N_padded:  pos_flat   [3*N_padded]     element positions (x,y,z, SoA)
    //   Offset 14*N_padded:  az_grid_f  [n_azimuth+1]    float azimuth grid (+1 sentinel)
    //                        el_grid_f  [n_elevation+1]   float elevation grid (+1 sentinel)
    const size_t flat_floats = 2 * N_padded + 9 * N_padded + 3 * N_padded + (n_azimuth + 1) + (n_elevation + 1);
    auto flat_buf_ptr = std::make_unique<float[]>(flat_floats);
    float *flat_buf = flat_buf_ptr.get();
    float *az_flat = flat_buf;
    float *el_flat = flat_buf + N_padded;
    float *R_flat = flat_buf + 2 * N_padded;
    float *pos_flat = flat_buf + 11 * N_padded;
    float *az_grid_f = flat_buf + 14 * N_padded;
    float *el_grid_f = az_grid_f + (n_azimuth + 1);
    auto ie_flat_ptr = std::make_unique<unsigned[]>(N_padded);
    unsigned *ie_flat = ie_flat_ptr.get();

    // Initialize float grid arrays (+1 sentinel for safe gather)
    for (size_t a = 0; a < n_azimuth; ++a)
        az_grid_f[a] = (float)p_azimuth_grid[a];
    az_grid_f[n_azimuth] = std::numeric_limits<float>::infinity();
    for (size_t a = 0; a < n_elevation; ++a)
        el_grid_f[a] = (float)p_elevation_grid[a];
    el_grid_f[n_elevation] = std::numeric_limits<float>::infinity();

    // ---- Precomputation: build flat arrays ----
    for (size_t a = 0, i = 0; a < n_ang; ++a)
    {
        for (size_t o = 0; o < n_out; ++o, ++i)
        {
            // Angles
            az_flat[i] = (float)(per_element_angles ? p_az_global[a * n_out + o] : p_az_global[a]);
            el_flat[i] = (float)(per_element_angles ? p_el_global[a * n_out + o] : p_el_global[a]);

            // Rotation matrix
            size_t Rp_a = per_angle_rotation ? a : 0;
            size_t Rp_o = per_element_rotation ? o : 0;
            const dtype *Rp = R_typed.slice_colptr(Rp_a, Rp_o);
            for (size_t k = 0; k < 9; ++k)
                R_flat[k * N_padded + i] = (float)Rp[k];

            // Element positions
            pos_flat[0 * N_padded + i] = (float)p_element_pos[3 * o + 0];
            pos_flat[1 * N_padded + i] = (float)p_element_pos[3 * o + 1];
            pos_flat[2 * N_padded + i] = (float)p_element_pos[3 * o + 2];

            // Element index (1-based)
            ie_flat[i] = p_i_element[o];
        }
    }

    // Pad remaining slots [N_total..N_padded) with last valid item
    for (size_t i = N_total; i < N_padded; ++i)
    {
        az_flat[i] = az_flat[N_total - 1];
        el_flat[i] = el_flat[N_total - 1];
        for (size_t k = 0; k < 9; ++k)
            R_flat[k * N_padded + i] = R_flat[k * N_padded + N_total - 1];
        pos_flat[0 * N_padded + i] = pos_flat[0 * N_padded + N_total - 1];
        pos_flat[1 * N_padded + i] = pos_flat[1 * N_padded + N_total - 1];
        pos_flat[2 * N_padded + i] = pos_flat[2 * N_padded + N_total - 1];
        ie_flat[i] = ie_flat[N_total - 1];
    }

    // ---- AVX2 constants ----
    const __m256 eps_8 = _mm256_set1_ps(eps);
    const __m256 eps_sq_8 = _mm256_set1_ps(eps * eps); // squared eps for rsqrt guard

    // ---- Tail mask for masked stores ----
    const int tail = (int)(N_total % 8);
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

    // ---- Collapsed flat loop (AVX2 Stages A–G) ----
#pragma omp parallel for schedule(static) if (n_vec >= 4096)
    for (size_t iv = 0; iv < n_vec; ++iv)
    {
        const size_t i_base = iv * 8;
        const size_t n_valid = (iv == n_vec - 1 && N_total % 8 != 0) ? (N_total % 8) : 8;

        // ==== Stage A: Input Angles -> Cartesian (AVX2) ====
        __m256 az_in8 = _mm256_loadu_ps(&az_flat[i_base]);
        __m256 el_in8 = _mm256_loadu_ps(&el_flat[i_base]);

        __m256 sAZi8, cAZi8, sELi8, cELi8;
        _fm256_sincos256_ps(az_in8, &sAZi8, &cAZi8);
        _fm256_sincos256_ps(el_in8, &sELi8, &cELi8);
        cELi8 = _mm256_add_ps(cELi8, eps_8); // pole guard

        __m256 Cx8 = _mm256_mul_ps(cELi8, cAZi8);
        __m256 Cy8 = _mm256_mul_ps(cELi8, sAZi8);

        // ==== Stage B: Rotation + cart2geo (AVX2) ====
        // Load 9 rotation matrix rows (SoA contiguous loads)
        __m256 Rm0_8 = _mm256_loadu_ps(&R_flat[0 * N_padded + i_base]);
        __m256 Rm1_8 = _mm256_loadu_ps(&R_flat[1 * N_padded + i_base]);
        __m256 Rm2_8 = _mm256_loadu_ps(&R_flat[2 * N_padded + i_base]);
        __m256 Rm3_8 = _mm256_loadu_ps(&R_flat[3 * N_padded + i_base]);
        __m256 Rm4_8 = _mm256_loadu_ps(&R_flat[4 * N_padded + i_base]);
        __m256 Rm5_8 = _mm256_loadu_ps(&R_flat[5 * N_padded + i_base]);
        __m256 Rm6_8 = _mm256_loadu_ps(&R_flat[6 * N_padded + i_base]);
        __m256 Rm7_8 = _mm256_loadu_ps(&R_flat[7 * N_padded + i_base]);
        __m256 Rm8_8 = _mm256_loadu_ps(&R_flat[8 * N_padded + i_base]);

        // Matrix-vector multiply: d = R * [Cx, Cy, sEL]
        __m256 dx8 = _mm256_fmadd_ps(Rm0_8, Cx8, _mm256_fmadd_ps(Rm3_8, Cy8, _mm256_mul_ps(Rm6_8, sELi8)));
        __m256 dy8 = _mm256_fmadd_ps(Rm1_8, Cx8, _mm256_fmadd_ps(Rm4_8, Cy8, _mm256_mul_ps(Rm7_8, sELi8)));
        __m256 dz8 = _mm256_fmadd_ps(Rm2_8, Cx8, _mm256_fmadd_ps(Rm5_8, Cy8, _mm256_mul_ps(Rm8_8, sELi8)));

        // cart2geo: az = atan2(dy, dx), el = asin(clamp(dz))
        __m256 az8, el8;
        _fm256_cart2geo_ps(dx8, dy8, dz8, &az8, &el8);

        // Derive sin/cos of output angles from dx/dy/dz (avoids extra sincos calls)
        // rsqrt+NR gives 1/hypot in one step, replacing sqrt + 2×div
        __m256 hyp_sq8 = _mm256_fmadd_ps(dx8, dx8, _mm256_mul_ps(dy8, dy8));
        __m256 safe_sq8 = _mm256_max_ps(hyp_sq8, eps_sq_8); // guard rsqrt(0) at poles
        __m256 inv_hypot8 = _fm256_rsqrt_nr_ps(safe_sq8);
        __m256 cAZo8 = _mm256_mul_ps(dx8, inv_hypot8);
        __m256 sAZo8 = _mm256_mul_ps(dy8, inv_hypot8);
        __m256 cELo8 = _mm256_mul_ps(hyp_sq8, inv_hypot8); // hyp_sq / sqrt(hyp_sq) = sqrt(hyp_sq)

        // ==== Stage C: Basis Vectors + Gamma (AVX2) ====
        // Input basis vectors
        __m256 eTHi_x8 = _mm256_mul_ps(sELi8, cAZi8);
        __m256 eTHi_y8 = _mm256_mul_ps(sELi8, sAZi8);
        __m256 neg_cELi8 = _mm256_xor_ps(cELi8, _mm256_set1_ps(-0.0f)); // -cELi8
        __m256 eTHi_z8 = neg_cELi8;

        __m256 ePHi_x8 = _mm256_xor_ps(sAZi8, _mm256_set1_ps(-0.0f)); // -sAZi8
        __m256 ePHi_y8 = cAZi8;

        // Output basis vector theta-hat
        __m256 eTHo_x8 = _mm256_mul_ps(dz8, cAZo8);
        __m256 eTHo_y8 = _mm256_mul_ps(dz8, sAZo8);
        __m256 neg_cELo8 = _mm256_xor_ps(cELo8, _mm256_set1_ps(-0.0f));
        __m256 eTHo_z8 = neg_cELo8;

        // Rotate eTHo by R^T
        __m256 eTHor_x8 = _mm256_fmadd_ps(Rm0_8, eTHo_x8, _mm256_fmadd_ps(Rm1_8, eTHo_y8, _mm256_mul_ps(Rm2_8, eTHo_z8)));
        __m256 eTHor_y8 = _mm256_fmadd_ps(Rm3_8, eTHo_x8, _mm256_fmadd_ps(Rm4_8, eTHo_y8, _mm256_mul_ps(Rm5_8, eTHo_z8)));
        __m256 eTHor_z8 = _mm256_fmadd_ps(Rm6_8, eTHo_x8, _mm256_fmadd_ps(Rm7_8, eTHo_y8, _mm256_mul_ps(Rm8_8, eTHo_z8)));

        // Gamma: cos_gamma = dot(eTHi, eTHor), sin_gamma = dot(ePHi, eTHor)
        __m256 cos_gamma8 = _mm256_fmadd_ps(eTHi_x8, eTHor_x8, _mm256_fmadd_ps(eTHi_y8, eTHor_y8, _mm256_mul_ps(eTHi_z8, eTHor_z8)));
        __m256 sin_gamma8 = _mm256_fmadd_ps(ePHi_x8, eTHor_x8, _mm256_mul_ps(ePHi_y8, eTHor_y8));

        // ==== Stage D: Distance (AVX2, skip if not requested) ====
        // Signed projection of element position onto arrival direction, negated to give
        // effective path-length difference: dist = -dot(d, pos) * ||d||, with sign preservation.
        __m256 dist8 = _mm256_setzero_ps();
        if (p_dist != nullptr)
        {
            __m256 px8 = _mm256_loadu_ps(&pos_flat[0 * N_padded + i_base]);
            __m256 py8 = _mm256_loadu_ps(&pos_flat[1 * N_padded + i_base]);
            __m256 pz8 = _mm256_loadu_ps(&pos_flat[2 * N_padded + i_base]);

            __m256 dot8 = _mm256_fmadd_ps(dx8, px8, _mm256_fmadd_ps(dy8, py8, _mm256_mul_ps(dz8, pz8)));
            __m256 dx2_8 = _mm256_mul_ps(dx8, dx8);
            __m256 dy2_8 = _mm256_mul_ps(dy8, dy8);
            __m256 dz2_8 = _mm256_mul_ps(dz8, dz8);
            __m256 sgn8 = _fm256_signum_ps(_mm256_fmadd_ps(dot8, dx2_8, _mm256_fmadd_ps(dot8, dy2_8, _mm256_mul_ps(dot8, dz2_8))));
            __m256 dot2_8 = _mm256_mul_ps(dot8, dot8);
            dist8 = _mm256_xor_ps(
                _mm256_mul_ps(sgn8, _mm256_sqrt_ps(_mm256_fmadd_ps(dot2_8, dx2_8, _mm256_fmadd_ps(dot2_8, dy2_8, _mm256_mul_ps(dot2_8, dz2_8))))),
                _mm256_set1_ps(-0.0f)); // negate
        }

        // ==== Stage E: Vectorized Grid Search ====
        __m256i iUp8, iUn8, iVp8, iVn8;
        __m256 up8, vp8;

        const __m256 ones_8 = _mm256_set1_ps(1.0f);
        const __m256 zeros_8 = _mm256_setzero_ps();
        const __m256 eps0_8 = _mm256_set1_ps(eps_cubed);

        // ---- Azimuth grid search ----
        if (n_azimuth == 1)
        {
            iUp8 = _mm256_setzero_si256();
            iUn8 = _mm256_setzero_si256();
            up8 = ones_8;
        }
        else if (az_uniform)
        {
            // Uniform grid: O(1) arithmetic index lookup
            const __m256 az_min_v = _mm256_set1_ps(az_fmin);
            const __m256 az_max_v = _mm256_set1_ps(az_fmax);
            const __m256 az_rinv_v = _mm256_set1_ps(az_step_rinv);
            const __m256 az_wrap_rinv_v = _mm256_set1_ps(az_step_inv[0]);
            const __m256i az_last_i = _mm256_set1_epi32((int)(n_azimuth - 1));
            const __m256i az_max_idx = _mm256_set1_epi32((int)(n_azimuth - 2));

            __m256 fidx = _mm256_mul_ps(_mm256_sub_ps(az8, az_min_v), az_rinv_v);

            // Floor: truncate toward zero, then adjust for negative values
            __m256i i_trunc = _mm256_cvttps_epi32(fidx);
            __m256 f_trunc = _mm256_cvtepi32_ps(i_trunc);
            __m256 neg_adj = _mm256_cmp_ps(fidx, f_trunc, _CMP_LT_OQ);
            __m256i i_floor = _mm256_add_epi32(i_trunc, _mm256_castps_si256(neg_adj));

            // Normal case: i_up = clamp(floor, 0, n_az-2), i_un = i_up + 1
            __m256i iUp_norm = _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(i_floor, az_max_idx));
            __m256i iUn_norm = _mm256_add_epi32(iUp_norm, _mm256_set1_epi32(1));

            // Weight: gather grid[iUn], compute un = (grid[iUn] - az) * diff
            // This avoids catastrophic cancellation from fidx - floor(fidx) at large indices
            __m256 grid_un = _mm256_i32gather_ps(az_grid_f, iUn_norm, 4);
            __m256 un_norm = _mm256_mul_ps(_mm256_sub_ps(grid_un, az8), az_rinv_v);
            un_norm = _mm256_max_ps(zeros_8, _mm256_min_ps(ones_8, un_norm));
            __m256 up_norm = _mm256_sub_ps(ones_8, un_norm);

            // Wrap detection: below (az < grid[0]) or above (az >= grid[last])
            __m256 mask_below = _mm256_cmp_ps(az8, az_min_v, _CMP_LT_OQ);
            __m256 mask_above = _mm256_cmp_ps(az8, az_max_v, _CMP_GE_OQ);
            __m256 wrap_mask = _mm256_or_ps(mask_below, mask_above);

            // Wrap case: i_up = n_az-1, i_un = 0
            __m256 wrap_un_below = _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(az_min_v, az8), eps0_8), az_wrap_rinv_v);
            __m256 wrap_up_below = _mm256_sub_ps(ones_8, _mm256_min_ps(ones_8, wrap_un_below));
            __m256 wrap_up_above = _mm256_min_ps(ones_8, _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(az8, az_max_v), eps0_8), az_wrap_rinv_v));
            __m256 wrap_up = _mm256_blendv_ps(wrap_up_above, wrap_up_below, mask_below);
            wrap_up = _mm256_max_ps(zeros_8, _mm256_min_ps(ones_8, wrap_up));

            // Blend normal vs wrap
            iUp8 = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(iUp_norm), _mm256_castsi256_ps(az_last_i), wrap_mask));
            iUn8 = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(iUn_norm), _mm256_setzero_ps(), wrap_mask));
            up8 = _mm256_blendv_ps(up_norm, wrap_up, wrap_mask);
        }
        else
        {
            // Non-uniform grid: vectorized binary search
            // Find upper bound: first index where grid[ub] > az
            __m256i bs_lo = _mm256_setzero_si256();
            __m256i bs_hi = _mm256_set1_epi32((int)n_azimuth);
            for (int iter = 0; iter < az_bsearch_iters; ++iter)
            {
                __m256i mid = _mm256_srli_epi32(_mm256_add_epi32(bs_lo, bs_hi), 1);
                __m256 val = _mm256_i32gather_ps(az_grid_f, mid, 4);
                __m256 cmp = _mm256_cmp_ps(val, az8, _CMP_LE_OQ); // grid[mid] <= az
                __m256i cmp_i = _mm256_castps_si256(cmp);
                bs_lo = _mm256_blendv_epi8(bs_lo, _mm256_add_epi32(mid, _mm256_set1_epi32(1)), cmp_i);
                bs_hi = _mm256_blendv_epi8(mid, bs_hi, cmp_i);
            }
            // bs_lo = upper bound in [0, n_azimuth]

            // Normal case: i_un = ub, i_up = ub - 1
            __m256i ub8 = bs_lo;
            __m256i iUp_norm = _mm256_sub_epi32(ub8, _mm256_set1_epi32(1));
            __m256i iUn_norm = ub8;

            // Weight: un = (grid[i_un] - az + eps³) * az_step_inv[i_un]
            // Clamp iUn for safe gather (wrap lanes have ub=0 or ub=n_azimuth, blended away later)
            __m256i iUn_safe = _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(iUn_norm, _mm256_set1_epi32((int)(n_azimuth - 1))));
            __m256 grid_un = _mm256_i32gather_ps(az_grid_f, iUn_safe, 4);
            __m256 diff_un = _mm256_i32gather_ps(az_step_inv, iUn_safe, 4);
            __m256 un_norm = _mm256_mul_ps(_mm256_sub_ps(grid_un, az8), diff_un);
            un_norm = _mm256_min_ps(ones_8, un_norm);
            __m256 up_norm = _mm256_sub_ps(ones_8, un_norm);

            // Wrap below: ub == 0 (az < grid[0])
            __m256 mask_below = _mm256_castsi256_ps(
                _mm256_cmpeq_epi32(ub8, _mm256_setzero_si256()));
            // Wrap above: ub == n_azimuth (az >= grid[last])
            __m256 mask_above = _mm256_castsi256_ps(
                _mm256_cmpeq_epi32(ub8, _mm256_set1_epi32((int)n_azimuth)));
            __m256 wrap_mask = _mm256_or_ps(mask_below, mask_above);

            // Wrap: i_up = n_az-1, i_un = 0
            // NOTE: wrap weight logic mirrors the uniform path above; duplicated intentionally
            //       to keep each branch self-contained and branchless within its path.
            __m256i az_last_i = _mm256_set1_epi32((int)(n_azimuth - 1));
            __m256 az_wrap_rinv_v = _mm256_set1_ps(az_step_inv[0]);
            __m256 az_min_v = _mm256_set1_ps(az_fmin);
            __m256 az_max_v = _mm256_set1_ps(az_fmax);

            __m256 wrap_un_below = _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(az_min_v, az8), eps0_8), az_wrap_rinv_v);
            __m256 wrap_up_below = _mm256_sub_ps(ones_8, _mm256_min_ps(ones_8, wrap_un_below));
            __m256 wrap_up_above = _mm256_min_ps(ones_8, _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(az8, az_max_v), eps0_8), az_wrap_rinv_v));
            __m256 wrap_up = _mm256_blendv_ps(wrap_up_above, wrap_up_below, mask_below);
            wrap_up = _mm256_max_ps(zeros_8, _mm256_min_ps(ones_8, wrap_up));

            // Blend
            iUp8 = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(iUp_norm), _mm256_castsi256_ps(az_last_i), wrap_mask));
            iUn8 = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(iUn_norm), _mm256_setzero_ps(), wrap_mask));
            up8 = _mm256_blendv_ps(up_norm, wrap_up, wrap_mask);
        }

        // ---- Elevation grid search ----
        if (n_elevation <= 1)
        {
            iVp8 = _mm256_setzero_si256();
            iVn8 = _mm256_setzero_si256();
            vp8 = ones_8;
        }
        else if (el_uniform)
        {
            // Uniform grid: O(1) arithmetic
            const __m256 el_min_v = _mm256_set1_ps(el_fmin);
            const __m256 el_rinv_v = _mm256_set1_ps(el_step_rinv);
            const int el_max_idx = (int)(n_elevation - 2);

            __m256 fidx = _mm256_mul_ps(_mm256_sub_ps(el8, el_min_v), el_rinv_v);

            // Floor
            __m256i i_trunc = _mm256_cvttps_epi32(fidx);
            __m256 f_trunc = _mm256_cvtepi32_ps(i_trunc);
            __m256 neg_adj = _mm256_cmp_ps(fidx, f_trunc, _CMP_LT_OQ);
            __m256i i_floor = _mm256_add_epi32(i_trunc, _mm256_castps_si256(neg_adj));

            // Clamp to [0, n_el-2] — elevation does not wrap
            __m256i iVp_c = _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(i_floor, _mm256_set1_epi32(el_max_idx)));
            __m256i iVn_c = _mm256_add_epi32(iVp_c, _mm256_set1_epi32(1));

            // Weight: gather grid[iVn], compute vn = (grid[iVn] - el + R0) * rinv
            // Avoids catastrophic cancellation from fidx - floor(fidx) at large indices
            __m256 grid_vn = _mm256_i32gather_ps(el_grid_f, iVn_c, 4);
            __m256 vn_c = _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(grid_vn, el8), eps0_8), el_rinv_v);
            vn_c = _mm256_max_ps(zeros_8, _mm256_min_ps(ones_8, vn_c));
            __m256 vp_c = _mm256_sub_ps(ones_8, vn_c);

            // Edge clamp: below grid → iVp=iVn=0, vp=1
            __m256 mask_below = _mm256_cmp_ps(el8, el_min_v, _CMP_LT_OQ);
            vp_c = _mm256_blendv_ps(vp_c, ones_8, mask_below);
            iVn_c = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(iVn_c), _mm256_setzero_ps(), mask_below));

            iVp8 = iVp_c;
            iVn8 = iVn_c;
            vp8 = vp_c;
        }
        else
        {
            // Non-uniform grid: vectorized binary search (no wrapping for elevation)
            __m256i bs_lo = _mm256_setzero_si256();
            __m256i bs_hi = _mm256_set1_epi32((int)n_elevation);
            for (int iter = 0; iter < el_bsearch_iters; ++iter)
            {
                __m256i mid = _mm256_srli_epi32(_mm256_add_epi32(bs_lo, bs_hi), 1);
                __m256 val = _mm256_i32gather_ps(el_grid_f, mid, 4);
                __m256 cmp = _mm256_cmp_ps(val, el8, _CMP_LE_OQ); // grid[mid] <= el
                __m256i cmp_i = _mm256_castps_si256(cmp);
                bs_lo = _mm256_blendv_epi8(bs_lo, _mm256_add_epi32(mid, _mm256_set1_epi32(1)), cmp_i);
                bs_hi = _mm256_blendv_epi8(mid, bs_hi, cmp_i);
            }
            // bs_lo = upper bound in [0, n_elevation]
            __m256i ub8 = bs_lo;

            // Past end: ub == n_elevation → clamp: i_vp = i_vn = n_el-1
            __m256 mask_past = _mm256_castsi256_ps(
                _mm256_cmpeq_epi32(ub8, _mm256_set1_epi32((int)n_elevation)));
            // Before start: ub == 0 → clamp: i_vp = i_vn = 0
            __m256 mask_before = _mm256_castsi256_ps(
                _mm256_cmpeq_epi32(ub8, _mm256_setzero_si256()));
            __m256 clamp_mask = _mm256_or_ps(mask_past, mask_before);

            // Normal: i_vn = ub, i_vp = ub - 1
            __m256i iVp_norm = _mm256_sub_epi32(ub8, _mm256_set1_epi32(1));
            __m256i iVn_norm = ub8;

            // Weight: vn = (grid[i_vn] - el + eps³) * el_step_inv[i_vn]
            // Clamp i_vn to valid range for gather (avoid OOB at n_elevation)
            __m256i iVn_safe = _mm256_min_epi32(iVn_norm, _mm256_set1_epi32((int)(n_elevation - 1)));
            __m256 grid_vn = _mm256_i32gather_ps(el_grid_f, iVn_safe, 4);
            __m256 diff_vn = _mm256_i32gather_ps(el_step_inv, iVn_safe, 4);
            __m256 vn_norm = _mm256_mul_ps(_mm256_add_ps(_mm256_sub_ps(grid_vn, el8), eps0_8), diff_vn);
            vn_norm = _mm256_min_ps(ones_8, vn_norm);
            __m256 vp_norm = _mm256_sub_ps(ones_8, vn_norm);

            // Clamp past end: i_vp = i_vn = n_el-1, vp = 1
            __m256i el_last_i = _mm256_set1_epi32((int)(n_elevation - 1));
            __m256i iVp_past = el_last_i;
            __m256i iVn_past = el_last_i;

            // Clamp before start: i_vp = i_vn = 0, vp = 1
            // Both clamp cases: vp = 1 (weight doesn't matter since both indices are equal)

            // For clamped lanes: override indices and weight
            iVp_norm = _mm256_max_epi32(iVp_norm, _mm256_setzero_si256()); // guard negative from ub=0 case
            iVp8 = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(iVp_norm),
                _mm256_blendv_ps(_mm256_castsi256_ps(iVp_past), _mm256_setzero_ps(), mask_before),
                clamp_mask));
            iVn8 = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(iVn_safe),
                _mm256_blendv_ps(_mm256_castsi256_ps(iVn_past), _mm256_setzero_ps(), mask_before),
                clamp_mask));
            vp8 = _mm256_blendv_ps(vp_norm, ones_8, clamp_mask);
        }

        // ==== Stage F: Build gather indices (shared) ====

        __m256i ie8 = _mm256_loadu_si256((const __m256i *)&ie_flat[i_base]);
        const __m256i one_i = _mm256_set1_epi32(1);
        const __m256i n_pat_i = _mm256_set1_epi32((int)n_pattern_samples);
        const __m256i n_el_i = _mm256_set1_epi32((int)n_elevation);
        __m256i offset8 = _mm256_mullo_epi32(_mm256_sub_epi32(ie8, one_i), n_pat_i);

        __m256i row_up = _mm256_add_epi32(_mm256_mullo_epi32(iUp8, n_el_i), offset8);
        __m256i row_un = _mm256_add_epi32(_mm256_mullo_epi32(iUn8, n_el_i), offset8);
        __m256i iA8 = _mm256_add_epi32(row_up, iVp8);
        __m256i iB8 = _mm256_add_epi32(row_un, iVp8);
        __m256i iC8 = _mm256_add_epi32(row_up, iVn8);
        __m256i iD8 = _mm256_add_epi32(row_un, iVn8);

        // ==== Gather pattern data (dtype-dependent) ====
        __m256 VfAr8, VfAi8, VfBr8, VfBi8, VfCr8, VfCi8, VfDr8, VfDi8;
        __m256 HfAr8, HfAi8, HfBr8, HfBi8, HfCr8, HfCi8, HfDr8, HfDi8;

        if constexpr (sizeof(dtype) == sizeof(float))
        {
            // Float path: gather floats directly, scale=4
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
            // Double path: gather 4 doubles at a time, cvt to float, pack into __m256
            const double *dTHr = (const double *)p_theta_re, *dTHi = (const double *)p_theta_im;
            const double *dPHr = (const double *)p_phi_re, *dPHi = (const double *)p_phi_im;

            __m128i iA_lo = _mm256_castsi256_si128(iA8), iA_hi = _mm256_extracti128_si256(iA8, 1);
            __m128i iB_lo = _mm256_castsi256_si128(iB8), iB_hi = _mm256_extracti128_si256(iB8, 1);
            __m128i iC_lo = _mm256_castsi256_si128(iC8), iC_hi = _mm256_extracti128_si256(iC8, 1);
            __m128i iD_lo = _mm256_castsi256_si128(iD8), iD_hi = _mm256_extracti128_si256(iD8, 1);

#define GATHER_D2F(base, lo, hi)                           \
    _mm256_set_m128(                                       \
        _mm256_cvtpd_ps(_mm256_i32gather_pd(base, hi, 8)), \
        _mm256_cvtpd_ps(_mm256_i32gather_pd(base, lo, 8)))

            VfAr8 = GATHER_D2F(dTHr, iA_lo, iA_hi);
            VfAi8 = GATHER_D2F(dTHi, iA_lo, iA_hi);
            VfBr8 = GATHER_D2F(dTHr, iB_lo, iB_hi);
            VfBi8 = GATHER_D2F(dTHi, iB_lo, iB_hi);
            VfCr8 = GATHER_D2F(dTHr, iC_lo, iC_hi);
            VfCi8 = GATHER_D2F(dTHi, iC_lo, iC_hi);
            VfDr8 = GATHER_D2F(dTHr, iD_lo, iD_hi);
            VfDi8 = GATHER_D2F(dTHi, iD_lo, iD_hi);

            HfAr8 = GATHER_D2F(dPHr, iA_lo, iA_hi);
            HfAi8 = GATHER_D2F(dPHi, iA_lo, iA_hi);
            HfBr8 = GATHER_D2F(dPHr, iB_lo, iB_hi);
            HfBi8 = GATHER_D2F(dPHi, iB_lo, iB_hi);
            HfCr8 = GATHER_D2F(dPHr, iC_lo, iC_hi);
            HfCi8 = GATHER_D2F(dPHi, iC_lo, iC_hi);
            HfDr8 = GATHER_D2F(dPHr, iD_lo, iD_hi);
            HfDi8 = GATHER_D2F(dPHi, iD_lo, iD_hi);

#undef GATHER_D2F
        }

        // ==== 6× SLERP: bilinear spherical interpolation (shared) ====
        __m256 VEr8, VEi8, VFr8, VFi8, Vr8, Vi8;
        _fm256_slerp_complex_ps(VfAr8, VfAi8, VfBr8, VfBi8, up8, &VEr8, &VEi8);
        _fm256_slerp_complex_ps(VfCr8, VfCi8, VfDr8, VfDi8, up8, &VFr8, &VFi8);
        _fm256_slerp_complex_ps(VEr8, VEi8, VFr8, VFi8, vp8, &Vr8, &Vi8);

        __m256 HEr8, HEi8, HFr8, HFi8, Hr8, Hi8;
        _fm256_slerp_complex_ps(HfAr8, HfAi8, HfBr8, HfBi8, up8, &HEr8, &HEi8);
        _fm256_slerp_complex_ps(HfCr8, HfCi8, HfDr8, HfDi8, up8, &HFr8, &HFi8);
        _fm256_slerp_complex_ps(HEr8, HEi8, HFr8, HFi8, vp8, &Hr8, &Hi8);

        // ==== Stage G: Polarization Rotation (shared) ====
        __m256 v_re8 = _mm256_fmsub_ps(cos_gamma8, Vr8, _mm256_mul_ps(sin_gamma8, Hr8));
        __m256 v_im8 = _mm256_fmsub_ps(cos_gamma8, Vi8, _mm256_mul_ps(sin_gamma8, Hi8));
        __m256 h_re8 = _mm256_fmadd_ps(sin_gamma8, Vr8, _mm256_mul_ps(cos_gamma8, Hr8));
        __m256 h_im8 = _mm256_fmadd_ps(sin_gamma8, Vi8, _mm256_mul_ps(cos_gamma8, Hi8));

        // Conditional gamma computation
        __m256 gamma8 = _mm256_setzero_ps();
        if (__builtin_expect(p_gamma != nullptr, 0))
            gamma8 = _fm256_atan2256_ps(sin_gamma8, cos_gamma8);

        // ==== Stores (dtype-dependent) ====
        const bool is_tail = (iv == n_vec - 1) && (tail != 0);

        if constexpr (sizeof(dtype) == sizeof(float))
        {
            // Float stores with tail masking
            float *fvr = (float *)p_v_re, *fvi = (float *)p_v_im;
            float *fhr = (float *)p_h_re, *fhi = (float *)p_h_im;

            if (is_tail)
            {
                _mm256_maskstore_ps(&fvr[i_base], tail_mask, v_re8);
                _mm256_maskstore_ps(&fvi[i_base], tail_mask, v_im8);
                _mm256_maskstore_ps(&fhr[i_base], tail_mask, h_re8);
                _mm256_maskstore_ps(&fhi[i_base], tail_mask, h_im8);
                if (p_dist)
                    _mm256_maskstore_ps((float *)p_dist + i_base, tail_mask, dist8);
                if (p_azimuth_loc)
                    _mm256_maskstore_ps((float *)p_azimuth_loc + i_base, tail_mask, az8);
                if (p_elevation_loc)
                    _mm256_maskstore_ps((float *)p_elevation_loc + i_base, tail_mask, el8);
                if (p_gamma)
                    _mm256_maskstore_ps((float *)p_gamma + i_base, tail_mask, gamma8);
            }
            else
            {
                _mm256_storeu_ps(&fvr[i_base], v_re8);
                _mm256_storeu_ps(&fvi[i_base], v_im8);
                _mm256_storeu_ps(&fhr[i_base], h_re8);
                _mm256_storeu_ps(&fhi[i_base], h_im8);
                if (p_dist)
                    _mm256_storeu_ps((float *)p_dist + i_base, dist8);
                if (p_azimuth_loc)
                    _mm256_storeu_ps((float *)p_azimuth_loc + i_base, az8);
                if (p_elevation_loc)
                    _mm256_storeu_ps((float *)p_elevation_loc + i_base, el8);
                if (p_gamma)
                    _mm256_storeu_ps((float *)p_gamma + i_base, gamma8);
            }
        }
        else
        {
            // Double stores: convert float→double, store as __m256d pairs
            // Split 8×float register into two 4×double halves and store to double* destination
#define STORE_F2D(dst, src, base)                                                                       \
    do                                                                                                  \
    {                                                                                                   \
        _mm256_storeu_pd((double *)(dst) + (base), _mm256_cvtps_pd(_mm256_castps256_ps128(src)));       \
        _mm256_storeu_pd((double *)(dst) + (base) + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(src, 1))); \
    } while (0)

            if (is_tail)
            {
                // Tail: extract to float arrays, store valid lanes as double
                alignas(32) float tmp_vr[8], tmp_vi[8], tmp_hr[8], tmp_hi[8];
                alignas(32) float tmp_dist[8], tmp_az[8], tmp_el[8], tmp_gam[8];
                _mm256_store_ps(tmp_vr, v_re8);
                _mm256_store_ps(tmp_vi, v_im8);
                _mm256_store_ps(tmp_hr, h_re8);
                _mm256_store_ps(tmp_hi, h_im8);
                if (p_dist)
                    _mm256_store_ps(tmp_dist, dist8);
                _mm256_store_ps(tmp_az, az8);
                _mm256_store_ps(tmp_el, el8);
                if (p_gamma)
                    _mm256_store_ps(tmp_gam, gamma8);

                for (size_t j = 0; j < n_valid; ++j)
                {
                    p_v_re[i_base + j] = (dtype)tmp_vr[j];
                    p_v_im[i_base + j] = (dtype)tmp_vi[j];
                    p_h_re[i_base + j] = (dtype)tmp_hr[j];
                    p_h_im[i_base + j] = (dtype)tmp_hi[j];
                    if (p_dist)
                        p_dist[i_base + j] = (dtype)tmp_dist[j];
                    if (p_azimuth_loc)
                        p_azimuth_loc[i_base + j] = (dtype)tmp_az[j];
                    if (p_elevation_loc)
                        p_elevation_loc[i_base + j] = (dtype)tmp_el[j];
                    if (p_gamma)
                        p_gamma[i_base + j] = (dtype)tmp_gam[j];
                }
            }
            else
            {
                STORE_F2D(p_v_re, v_re8, i_base);
                STORE_F2D(p_v_im, v_im8, i_base);
                STORE_F2D(p_h_re, h_re8, i_base);
                STORE_F2D(p_h_im, h_im8, i_base);
                if (p_dist)
                    STORE_F2D(p_dist, dist8, i_base);
                if (p_azimuth_loc)
                    STORE_F2D(p_azimuth_loc, az8, i_base);
                if (p_elevation_loc)
                    STORE_F2D(p_elevation_loc, el8, i_base);
                if (p_gamma)
                    STORE_F2D(p_gamma, gamma8, i_base);
            }

#undef STORE_F2D
        }
    } // end iv loop over vectors
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