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

#include "qd_arrayant_interpolate.hpp"
#include <memory>
#include <stdexcept>

// Implements signum (-1, 0, or 1)
template <typename dtype>
static inline dtype signum(dtype val)
{
    constexpr dtype zero = dtype(0.0);
    return dtype((zero < val) - (val < zero));
}

// Euler rotations to rotation matrix
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

        double cc = (double)O[0], sc = std::sin(cc);
        double cb = (double)O[1], sb = std::sin(cb);
        double ca = (double)O[2], sa = std::sin(ca);
        ca = std::cos(ca), cb = std::cos(cb), cc = std::cos(cc);
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
void qd_arrayant_interpolate(const arma::Cube<dtype> &e_theta_re, const arma::Cube<dtype> &e_theta_im,
                             const arma::Cube<dtype> &e_phi_re, const arma::Cube<dtype> &e_phi_im,
                             const arma::Col<dtype> &azimuth_grid, const arma::Col<dtype> &elevation_grid,
                             const arma::Mat<dtype> &azimuth, const arma::Mat<dtype> &elevation,
                             const arma::Col<unsigned> &i_element, const arma::Cube<dtype> &orientation,
                             const arma::Mat<dtype> &element_pos,
                             arma::Mat<dtype> &V_re, arma::Mat<dtype> &V_im,
                             arma::Mat<dtype> &H_re, arma::Mat<dtype> &H_im,
                             arma::Mat<dtype> *dist,
                             arma::Mat<dtype> *azimuth_loc, arma::Mat<dtype> *elevation_loc, arma::Mat<dtype> *gamma)
{
    // Note: This function is the scalar reference implementation for antenna pattern interpolation.

    const size_t n_elevation = (size_t)e_theta_re.n_rows;    // Number of elevation angles in the pattern
    const size_t n_azimuth = (size_t)e_theta_re.n_cols;      // Number of azimuth angles in the pattern
    const size_t n_pattern_samples = n_azimuth * n_elevation; // Number of samples in the pattern
    const size_t n_out = (size_t)i_element.n_elem;            // Number of elements in the output
    const size_t n_ang = (size_t)azimuth.n_cols;              // Number of angles to be interpolated

    // Fix 1: Early return for empty inputs (avoids wraparound UB on n_azimuth-1 etc.)
    if (n_out == 0 || n_ang == 0)
        return;

    // Fix 7: Grid must have at least one sample
    if (n_azimuth == 0 || n_elevation == 0)
        throw std::invalid_argument("qd_arrayant_interpolate: Grid dimensions must be >= 1.");

    // Fix 5: Pattern cubes must have matching dimensions
    const size_t n_elements = (size_t)e_theta_re.n_slices;
    if (e_theta_im.n_rows != n_elevation || e_theta_im.n_cols != n_azimuth || e_theta_im.n_slices != n_elements ||
        e_phi_re.n_rows != n_elevation || e_phi_re.n_cols != n_azimuth || e_phi_re.n_slices != n_elements ||
        e_phi_im.n_rows != n_elevation || e_phi_im.n_cols != n_azimuth || e_phi_im.n_slices != n_elements)
        throw std::invalid_argument("qd_arrayant_interpolate: Pattern cubes must have matching [n_elevation, n_azimuth, n_elements] dimensions.");

    if (azimuth_grid.n_elem != n_azimuth)
        throw std::invalid_argument("qd_arrayant_interpolate: azimuth_grid length must equal n_azimuth.");
    if (elevation_grid.n_elem != n_elevation)
        throw std::invalid_argument("qd_arrayant_interpolate: elevation_grid length must equal n_elevation.");

    if (azimuth.n_cols != elevation.n_cols || azimuth.n_rows != elevation.n_rows)
        throw std::invalid_argument("qd_arrayant_interpolate: azimuth and elevation must have the same size.");

    if (orientation.n_rows != 3)
        throw std::invalid_argument("qd_arrayant_interpolate: orientation must have 3 rows.");

    if (element_pos.n_rows != 3 || element_pos.n_cols != n_out)
        throw std::invalid_argument("qd_arrayant_interpolate: element_pos must be [3, n_out].");

    // Fix 2: Validate i_element bounds (1-based, must be in [1, n_elements])
    const unsigned *p_i_element = i_element.memptr();
    for (size_t o = 0; o < n_out; ++o)
        if (p_i_element[o] < 1 || p_i_element[o] > (unsigned)n_elements)
            throw std::out_of_range("qd_arrayant_interpolate: i_element contains out-of-range index.");

    // Fix 4: Validate mandatory output sizes (must be pre-allocated)
    if (V_re.n_rows != n_out || V_re.n_cols != n_ang ||
        V_im.n_rows != n_out || V_im.n_cols != n_ang ||
        H_re.n_rows != n_out || H_re.n_cols != n_ang ||
        H_im.n_rows != n_out || H_im.n_cols != n_ang)
        throw std::invalid_argument("qd_arrayant_interpolate: Output matrices V_re/V_im/H_re/H_im must be pre-allocated to [n_out, n_ang].");

    // Validate optional output sizes if provided and non-empty
    if (dist != nullptr && !dist->is_empty() && (dist->n_rows != n_out || dist->n_cols != n_ang))
        throw std::invalid_argument("qd_arrayant_interpolate: dist must be [n_out, n_ang] if provided.");
    if (azimuth_loc != nullptr && !azimuth_loc->is_empty() && (azimuth_loc->n_rows != n_out || azimuth_loc->n_cols != n_ang))
        throw std::invalid_argument("qd_arrayant_interpolate: azimuth_loc must be [n_out, n_ang] if provided.");
    if (elevation_loc != nullptr && !elevation_loc->is_empty() && (elevation_loc->n_rows != n_out || elevation_loc->n_cols != n_ang))
        throw std::invalid_argument("qd_arrayant_interpolate: elevation_loc must be [n_out, n_ang] if provided.");
    if (gamma != nullptr && !gamma->is_empty() && (gamma->n_rows != n_out || gamma->n_cols != n_ang))
        throw std::invalid_argument("qd_arrayant_interpolate: gamma must be [n_out, n_ang] if provided.");

    bool per_element_angles = azimuth.n_rows > 1 ? true : false;
    bool per_element_rotation = orientation.n_cols > 1 ? true : false;
    bool per_angle_rotation = orientation.n_slices > 1 ? true : false;

    // Output pointers
    dtype *p_dist = (dist == nullptr || dist->is_empty()) ? nullptr : dist->memptr();
    dtype *p_azimuth_loc = (azimuth_loc == nullptr || azimuth_loc->is_empty()) ? nullptr : azimuth_loc->memptr();
    dtype *p_elevation_loc = (elevation_loc == nullptr || elevation_loc->is_empty()) ? nullptr : elevation_loc->memptr();
    dtype *p_gamma = (gamma == nullptr || gamma->is_empty()) ? nullptr : gamma->memptr();

    // Rotation matrix [3,3,n_out] or [3,3,1]
    arma::Cube<dtype> R_typed(9, orientation.n_cols, orientation.n_slices, arma::fill::none);
    qd_rotation_matrix(orientation.memptr(), R_typed.memptr(), orientation.n_cols * orientation.n_slices, true, true);

    // Obtain pointers for direct memory access
    const dtype *p_theta_re = e_theta_re.memptr(), *p_theta_im = e_theta_im.memptr(); // Vertical pattern
    const dtype *p_phi_re = e_phi_re.memptr(), *p_phi_im = e_phi_im.memptr();         // Horizontal pattern
    const dtype *p_az_global = azimuth.memptr(), *p_el_global = elevation.memptr();    // Angles
    const dtype *p_azimuth_grid = azimuth_grid.memptr(), *p_elevation_grid = elevation_grid.memptr();
    const dtype *p_element_pos = element_pos.memptr();
    dtype *p_v_re = V_re.memptr(), *p_v_im = V_im.memptr(), *p_h_re = H_re.memptr(), *p_h_im = H_im.memptr();

    // Declare constants at compile time to avoid unnecessary type conversions
    const dtype pi_double = arma::Datum<dtype>::tau;
    const dtype R0 = arma::Datum<dtype>::eps * arma::Datum<dtype>::eps * arma::Datum<dtype>::eps;
    const dtype R1 = arma::Datum<dtype>::eps;
    constexpr dtype one = dtype(1.0), neg_one = dtype(-1.0), zero = dtype(0.0),
                    tL = dtype(-0.999), tS = dtype(-0.99), dT = one / (tS - tL);

    // Calculate 1/dist in the pattern sampling
    auto az_diff_buf = std::make_unique<dtype[]>(n_azimuth);
    auto el_diff_buf = std::make_unique<dtype[]>(n_elevation);
    dtype *az_diff = az_diff_buf.get(), *el_diff = el_diff_buf.get();
    *az_diff = pi_double - p_azimuth_grid[n_azimuth - 1] + *p_azimuth_grid;
    *az_diff = one / *az_diff;
    *el_diff = one;
    for (size_t a = 1; a < n_azimuth; ++a)
        az_diff[a] = one / (p_azimuth_grid[a] - p_azimuth_grid[a - 1]);
    for (size_t a = 1; a < n_elevation; ++a)
        el_diff[a] = one / (p_elevation_grid[a] - p_elevation_grid[a - 1]);

    // Interpolate the pattern data using spheric interpolation
    // datatype "int" is required by MSVC to allow parallel for
#pragma omp parallel for
    for (int a_i32 = 0; a_i32 < (int)n_ang; ++a_i32)
    {
        // Convert a_i32 to 64 bit
        const size_t a = (size_t)a_i32;

        // Get the local pointer for the angles
        const dtype *p_az_local = per_element_angles ? &p_az_global[a * n_out] : &p_az_global[a];
        const dtype *p_el_local = per_element_angles ? &p_el_global[a * n_out] : &p_el_global[a];

        // Decare and initialize all local variables
        size_t i_up = 0, i_un = 0, i_vp = 0, i_vn = 0;  // Indices for reading the pattern
        dtype up = one, un = zero, vp = one, vn = zero; // Relative weights for interpolation
        dtype cAZi = one, sAZi = zero, cELi = one, sELi = zero, Cx = one, Cy = zero;
        dtype az = zero, el = zero, sin_gamma = zero, cos_gamma = one, dx = one, dy = zero, dz = zero;

        for (size_t o = 0; o < n_out; ++o)
        {
            // Check if we need to update the angles for the current output index "o"
            bool update_angles = per_element_angles || per_element_rotation || per_angle_rotation || o == 0;

            // Transform input angles to Cartesian coordinates
            if (per_element_angles || o == 0)
                sAZi = std::sin(*p_az_local), cAZi = std::cos(*p_az_local++),
                sELi = std::sin(*p_el_local), cELi = std::cos(*p_el_local++) + R1,
                Cx = cELi * cAZi, Cy = cELi * sAZi;

            // Apply rotation (Co = R * Ci) for antenna pattern interpolation
            // Transform from Cartesian coordinates to geographic coordinates
            size_t Rp_a = per_angle_rotation ? a : 0, Rp_o = per_element_rotation ? o : 0;
            const dtype *Rp = R_typed.slice_colptr(Rp_a, Rp_o);
            if (update_angles)
            {
                dtype cAZo, sAZo;
                dx = Rp[0] * Cx + Rp[3] * Cy + Rp[6] * sELi,
                dy = Rp[1] * Cx + Rp[4] * Cy + Rp[7] * sELi,
                dz = Rp[2] * Cx + Rp[5] * Cy + Rp[8] * sELi;
                dz = dz > one ? one : dz;
                dz = dz < neg_one ? neg_one : dz;
                az = std::atan2(dy, dx), el = std::asin(dz), sAZo = std::sin(az), cAZo = std::cos(az);

                // Calculate basis vectors
                dtype eTHi_x = sELi * cAZi, eTHi_y = sELi * sAZi, eTHi_z = -cELi;
                dtype ePHi_x = -sAZi, ePHi_y = cAZi;
                dtype eTHo_x = dz * cAZo, eTHo_y = dz * sAZo, eTHo_z = -std::cos(el);

                // Apply rotation to eTHo
                dtype eTHor_x = Rp[0] * eTHo_x + Rp[1] * eTHo_y + Rp[2] * eTHo_z;
                dtype eTHor_y = Rp[3] * eTHo_x + Rp[4] * eTHo_y + Rp[5] * eTHo_z;
                dtype eTHor_z = Rp[6] * eTHo_x + Rp[7] * eTHo_y + Rp[8] * eTHo_z;

                // Calculate polarization rotation angle
                cos_gamma = eTHi_x * eTHor_x + eTHi_y * eTHor_y + eTHi_z * eTHor_z;
                sin_gamma = ePHi_x * eTHor_x + ePHi_y * eTHor_y;
            }

            // Calculate the projected distance
            dtype dst = zero;
            if (p_dist != nullptr)
            {
                dst = dx * p_element_pos[3 * o] + dy * p_element_pos[3 * o + 1] + dz * p_element_pos[3 * o + 2];
                dtype dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
                dtype sgn = signum(dst * dx2 + dst * dy2 + dst * dz2);
                dst *= dst;
                dst = -sgn * std::sqrt(dst * dx2 + dst * dy2 + dst * dz2);
            }

            // Calc. indices for reading the pattern and relative weights for interpolation
            if (update_angles)
            {
                i_up = 0, i_un = 0, up = one, un = zero;
                i_vp = 0, i_vn = 0, vp = one, vn = zero;
                if (n_azimuth != 1)
                {
                    if (*p_azimuth_grid > az) // az is between -pi and first grid point
                    {
                        i_up = n_azimuth - 1;
                        un = (*p_azimuth_grid - az + R0) * *az_diff;
                        un = un > one ? one : un, up = one - un;
                    }
                    else
                    {
                        while (i_up < n_azimuth && p_azimuth_grid[i_up] <= az)
                            ++i_up;

                        if (i_up == n_azimuth) // az is between last grid point and pi
                        {
                            up = (az - p_azimuth_grid[--i_up] + R0) * *az_diff;
                            up = up > one ? one : up, un = one - up;
                        }
                        else
                        {
                            i_un = i_up--;
                            un = (p_azimuth_grid[i_un] - az) * az_diff[i_un];
                            un = un > one ? one : un, up = one - un;
                        }
                    }
                }

                while (i_vp < n_elevation && p_elevation_grid[i_vp] <= el)
                    ++i_vp;

                if (i_vp == n_elevation)
                    i_vn = --i_vp;
                else if (i_vp != 0)
                {
                    i_vn = i_vp--;
                    vn = (p_elevation_grid[i_vn] - el + R0) * el_diff[i_vn];
                    vn = vn > one ? one : vn, vp = one - vn;
                }
            }

            // Illustration of the 2D interpolation procedure:
            // Points A,B,C,D are given in the input, point X is to be calculated
            //
            //      C------F--------------------D
            //      |      |                    |
            //      |      | (1-v)=vwn          |
            //      |      |                    |
            //      |      |                    |
            //      | uwp  |     (1-u)=uwn      |
            //      G----- X ------------------ H
            //      |      |                    |
            //      |      | vwp                |
            //      |      |                    |
            //      A------E--------------------B

            // Calculate the indices to read points A,B,C,D from the input pattern data
            size_t offset = n_pattern_samples * (p_i_element[o] - 1);
            size_t iA = i_up * n_elevation + i_vp + offset;
            size_t iB = i_un * n_elevation + i_vp + offset;
            size_t iC = i_up * n_elevation + i_vn + offset;
            size_t iD = i_un * n_elevation + i_vn + offset;
            dtype Vr, Vi, Hr, Hi;

            for (size_t VH = 0; VH < 2; ++VH)
            {
                dtype fAr, fBr, fCr, fDr, fAi, fBi, fCi, fDi;
                if (VH == 0) // Read the pattern values
                    fAr = p_theta_re[iA], fAi = p_theta_im[iA],
                    fBr = p_theta_re[iB], fBi = p_theta_im[iB],
                    fCr = p_theta_re[iC], fCi = p_theta_im[iC],
                    fDr = p_theta_re[iD], fDi = p_theta_im[iD];
                else
                    fAr = p_phi_re[iA], fAi = p_phi_im[iA],
                    fBr = p_phi_re[iB], fBi = p_phi_im[iB],
                    fCr = p_phi_re[iC], fCi = p_phi_im[iC],
                    fDr = p_phi_re[iD], fDi = p_phi_im[iD];

                // Calculate amplitude
                dtype ampA = std::sqrt(fAr * fAr + fAi * fAi);
                dtype ampB = std::sqrt(fBr * fBr + fBi * fBi);
                dtype ampC = std::sqrt(fCr * fCr + fCi * fCi);
                dtype ampD = std::sqrt(fDr * fDr + fDi * fDi);

                // Normalize real and imaginary parts to obtain phase
                dtype gAr = one / ampA, gBr = one / ampB, gCr = one / ampC, gDr = one / ampD;
                dtype gAi = fAi * gAr, gBi = fBi * gBr, gCi = fCi * gCr, gDi = fDi * gDr;
                gAr = fAr * gAr, gBr = fBr * gBr, gCr = fCr * gCr, gDr = fDr * gDr;

                // Declare variables
                dtype cPhase = (ampA < R1 || ampB < R1) ? neg_one : gAr * gBr + gAi * gBi; // Cosine of phase
                bool linear_int = cPhase < tS;

                // Interpolation for point E
                dtype fEr = zero, fEi = zero, gEr = zero, gEi = zero, ampE = zero;
                if (linear_int) // Linear interpolation
                    fEr = un * fAr + up * fBr, fEi = un * fAi + up * fBi;
                if (cPhase > tL) // Spherical interpolation
                {
                    dtype Phase = (cPhase >= one) ? R0 : std::acos(cPhase) + R0, sPhase = one / std::sin(Phase),
                          wp = std::sin(up * Phase) * sPhase, wn = std::sin(un * Phase) * sPhase;
                    gEr = wn * gAr + wp * gBr, gEi = wn * gAi + wp * gBi, ampE = un * ampA + up * ampB;
                    if (linear_int) // Mixed mode
                        wp = (tS - cPhase) * dT, wn = one - wp,
                        fEr = wn * gEr * ampE + wp * fEr, fEi = wn * gEi * ampE + wp * fEi;
                    else
                        fEr = gEr * ampE, fEi = gEi * ampE;
                }
                if (linear_int)
                {
                    ampE = std::sqrt(fEr * fEr + fEi * fEi);
                    gEr = one / ampE;
                    gEi = fEi * gEr, gEr = fEr * gEr;
                }

                // Interpolation for point F
                dtype fFr = zero, fFi = zero, gFr = zero, gFi = zero, ampF = zero;
                cPhase = (ampC < R1 || ampD < R1) ? neg_one : gCr * gDr + gCi * gDi;
                linear_int = cPhase < tS;
                if (linear_int)
                    fFr = un * fCr + up * fDr, fFi = un * fCi + up * fDi;
                if (cPhase > tL)
                {
                    dtype Phase = (cPhase >= one) ? R0 : std::acos(cPhase) + R0, sPhase = one / std::sin(Phase),
                          wp = std::sin(up * Phase) * sPhase, wn = std::sin(un * Phase) * sPhase;
                    gFr = wn * gCr + wp * gDr, gFi = wn * gCi + wp * gDi, ampF = un * ampC + up * ampD;
                    if (linear_int) // Mixed mode
                        wp = (tS - cPhase) * dT, wn = one - wp,
                        fFr = wn * gFr * ampF + wp * fFr, fFi = wn * gFi * ampF + wp * fFi;
                    else
                        fFr = gFr * ampF, fFi = gFi * ampF;
                }
                if (linear_int)
                {
                    ampF = std::sqrt(fFr * fFr + fFi * fFi);
                    gFr = one / ampF;
                    gFi = fFi * gFr, gFr = fFr * gFr;
                }

                // Interpolation for point X
                dtype fXr = zero, fXi = zero;
                cPhase = (ampE < R1 || ampF < R1) ? neg_one : gEr * gFr + gEi * gFi;
                linear_int = cPhase < tS;
                if (linear_int)
                    fXr = vn * fEr + vp * fFr, fXi = vn * fEi + vp * fFi;
                if (cPhase > tL)
                {
                    dtype Phase = (cPhase >= one) ? R0 : std::acos(cPhase) + R0, sPhase = one / std::sin(Phase),
                          wp = std::sin(vp * Phase) * sPhase, wn = std::sin(vn * Phase) * sPhase;
                    dtype gXr = wn * gEr + wp * gFr, gXi = wn * gEi + wp * gFi, ampX = vn * ampE + vp * ampF;
                    if (linear_int) // Mixed mode
                        wp = (tS - cPhase) * dT, wn = one - wp,
                        fXr = wn * gXr * ampX + wp * fXr, fXi = wn * gXi * ampX + wp * fXi;
                    else
                        fXr = gXr * ampX, fXi = gXi * ampX;
                }

                if (VH == 0)
                    Vr = fXr, Vi = fXi;
                else
                    Hr = fXr, Hi = fXi;
            }

            // Compute and write output
            size_t ioa = a * n_out + o;
            p_v_re[ioa] = cos_gamma * Vr - sin_gamma * Hr;
            p_v_im[ioa] = cos_gamma * Vi - sin_gamma * Hi;
            p_h_re[ioa] = sin_gamma * Vr + cos_gamma * Hr;
            p_h_im[ioa] = sin_gamma * Vi + cos_gamma * Hi;

            // Write optional azimuth and elevation angles
            if (p_dist != nullptr)
                p_dist[ioa] = dst;
            if (p_azimuth_loc != nullptr)
                p_azimuth_loc[ioa] = az;
            if (p_elevation_loc != nullptr)
                p_elevation_loc[ioa] = el;
            if (p_gamma != nullptr)
                p_gamma[ioa] = std::atan2(sin_gamma, cos_gamma);
        }
    }
}

// Declare templates
template void qd_arrayant_interpolate(const arma::Cube<float> &e_theta_re, const arma::Cube<float> &e_theta_im,
                                      const arma::Cube<float> &e_phi_re, const arma::Cube<float> &e_phi_im,
                                      const arma::Col<float> &azimuth_grid, const arma::Col<float> &elevation_grid,
                                      const arma::Mat<float> &azimuth, const arma::Mat<float> &elevation,
                                      const arma::Col<unsigned> &i_element, const arma::Cube<float> &orientation,
                                      const arma::Mat<float> &element_pos,
                                      arma::Mat<float> &V_re, arma::Mat<float> &V_im,
                                      arma::Mat<float> &H_re, arma::Mat<float> &H_im,
                                      arma::Mat<float> *dist,
                                      arma::Mat<float> *azimuth_loc, arma::Mat<float> *elevation_loc, arma::Mat<float> *gamma);

template void qd_arrayant_interpolate(const arma::Cube<double> &e_theta_re, const arma::Cube<double> &e_theta_im,
                                      const arma::Cube<double> &e_phi_re, const arma::Cube<double> &e_phi_im,
                                      const arma::Col<double> &azimuth_grid, const arma::Col<double> &elevation_grid,
                                      const arma::Mat<double> &azimuth, const arma::Mat<double> &elevation,
                                      const arma::Col<unsigned> &i_element, const arma::Cube<double> &orientation,
                                      const arma::Mat<double> &element_pos,
                                      arma::Mat<double> &V_re, arma::Mat<double> &V_im,
                                      arma::Mat<double> &H_re, arma::Mat<double> &H_im,
                                      arma::Mat<double> *dist,
                                      arma::Mat<double> *azimuth_loc, arma::Mat<double> *elevation_loc, arma::Mat<double> *gamma);