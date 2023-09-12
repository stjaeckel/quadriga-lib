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

#include <iostream>
#include "qd_arrayant_interpolate.hpp"
#include "quadriga_tools.hpp"

// Implements signum (-1, 0, or 1)
template <typename dtype>
dtype signum(dtype val)
{
    constexpr dtype zero = dtype(0.0);
    return dtype((zero < val) - (val < zero));
}

template <typename dtype>
void qd_arrayant_interpolate(const arma::Cube<dtype> *e_theta_re, const arma::Cube<dtype> *e_theta_im,
                             const arma::Cube<dtype> *e_phi_re, const arma::Cube<dtype> *e_phi_im,
                             const arma::Col<dtype> *azimuth_grid, const arma::Col<dtype> *elevation_grid,
                             const arma::Mat<dtype> *azimuth, const arma::Mat<dtype> *elevation,
                             const arma::Col<unsigned> *i_element, const arma::Cube<dtype> *orientation,
                             const arma::Mat<dtype> *element_pos,
                             arma::Mat<dtype> *V_re, arma::Mat<dtype> *V_im,
                             arma::Mat<dtype> *H_re, arma::Mat<dtype> *H_im,
                             arma::Mat<dtype> *dist,
                             arma::Mat<dtype> *azimuth_loc, arma::Mat<dtype> *elevation_loc, arma::Mat<dtype> *gamma)
{
    // Inputs:
    // e_theta_re       Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
    // e_theta_im       Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
    // e_phi_re         Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
    // e_phi_im         Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
    // azimuth_grid     Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
    // elevation_grid   Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
    // azimuth          Azimuth angles for interpolation in [rad],                      Size [1, n_ang] or [n_out, n_ang]
    // elevation        Elevation angles for interpolation in [rad],                    Size [1, n_ang] or [n_out, n_ang]
    // i_element        Element indices, 1-based                                        Vector of length "n_out"
    // orientation      Orientation of the array antenna (bank, tilt, head) in [rad],   Size [3, 1, 1] or [3, n_out, 1] or [3, 1, n_ang] or [3, n_out, n_ang]
    // element_pos      Element positions                                               Size [3, n_out]

    // Outputs:
    // V_re             Interpolated vertical field, real part,                         Size [n_out, n_ang]
    // V_im             Interpolated vertical field, imaginary part,                    Size [n_out, n_ang]
    // H_re             Interpolated horizontal field, real part,                       Size [n_out, n_ang]
    // H_im             Interpolated horizontal field, imaginary part,                  Size [n_out, n_ang]
    // dist             Effective distances, optional                                   Size [n_out, n_ang] or []
    // azimuth_loc      Azimuth angles [rad] in local antenna coordinates, optional,    Size [n_out, n_ang] or []
    // elevation_loc    Elevation angles [rad] in local antenna coordinates, optional,  Size [n_out, n_ang] or []
    // gamma            Polarization rotation angles in [rad], optional,                Size [n_out, n_ang] or []

    // Note: This function is not intended to be publicly accessible. There is no input validation.
    // Incorrectly formatted arguments may lead to undefined behavior or segmentation faults.

    const uword n_elevation = e_theta_re->n_rows;            // Number of elevation angles in the pattern
    const uword n_azimuth = e_theta_re->n_cols;              // Number of azimuth angles in the pattern
    const uword n_pattern_samples = n_azimuth * n_elevation; // Number of samples in the pattern
    const uword n_out = i_element->n_elem;                   // Number of elements in the output
    const uword n_ang = azimuth->n_cols;                     // Number of angles to be interpolated

    bool per_element_angles = azimuth->n_rows > 1 ? true : false;
    bool per_element_rotation = orientation->n_cols > 1 ? true : false;
    bool per_angle_rotation = orientation->n_slices > 1 ? true : false;

    // Determine if we need to write the angles in local antenna coordinates
    bool write_az = !azimuth_loc->is_empty(), write_el = !elevation_loc->is_empty(),
         write_dist = !dist->is_empty(), write_gamma = !gamma->is_empty();

    // Rotation matrix [3,3,n_out] or [3,3,1], always double output
    arma::cube R = quadriga_tools::calc_rotation_matrix(*orientation, true, true);
    const arma::Cube<dtype> R_typed = arma::conv_to<arma::Cube<dtype>>::from(R);
    R.reset();

    // Obtain pointers for direct memory access (faster)
    const unsigned *p_i_element = i_element->memptr();                                  // Elements
    const dtype *p_theta_re = e_theta_re->memptr(), *p_theta_im = e_theta_im->memptr(); // Vertical pattern
    const dtype *p_phi_re = e_phi_re->memptr(), *p_phi_im = e_phi_im->memptr();         // Horizontal pattern
    const dtype *p_az_global = azimuth->memptr(), *p_el_global = elevation->memptr();   // Angles
    const dtype *p_azimuth_grid = azimuth_grid->memptr(), *p_elevation_grid = elevation_grid->memptr();
    const dtype *p_element_pos = element_pos->memptr();
    dtype *p_v_re = V_re->memptr(), *p_v_im = V_im->memptr(), *p_h_re = H_re->memptr(), *p_h_im = H_im->memptr();

    // Declare constants at compile time to avoid unnecessary type conversions
    const dtype pi_double = arma::Datum<dtype>::tau;
    const dtype R0 = arma::Datum<dtype>::eps * arma::Datum<dtype>::eps * arma::Datum<dtype>::eps;
    const dtype R1 = arma::Datum<dtype>::eps;
    constexpr dtype one = dtype(1.0), neg_one = dtype(-1.0), zero = dtype(0.0),
                    tL = dtype(-0.999), tS = dtype(-0.99), dT = one / (tS - tL);

    // Calculate 1/dist in the pattern sampling
    dtype *az_diff = new dtype[n_azimuth], *el_diff = new dtype[n_elevation];
    *az_diff = pi_double - p_azimuth_grid[n_azimuth - 1ULL] + *p_azimuth_grid;
    *az_diff = one / *az_diff;
    *el_diff = one;
    for (uword a = 1ULL; a < n_azimuth; a++)
        az_diff[a] = one / (p_azimuth_grid[a] - p_azimuth_grid[a - 1ULL]);
    for (uword a = 1ULL; a < n_elevation; a++)
        el_diff[a] = one / (p_elevation_grid[a] - p_elevation_grid[a - 1ULL]);

        // Interpolate the pattern data using spheric interpolation
        // datatype "int" is required by MSVC to allow parallel for
#pragma omp parallel for
    for (int a_i32 = 0; a_i32 < int(n_ang); a_i32++)
    {
        // Convert a_i32 to 64 bit
        const uword a = uword(a_i32);

        // Get the local pointer for the angles
        const dtype *p_az_local = per_element_angles ? &p_az_global[a * n_out] : &p_az_global[a];
        const dtype *p_el_local = per_element_angles ? &p_el_global[a * n_out] : &p_el_global[a];

        // Decare and initialize all local variables
        uword i_up = 0ULL, i_un = 0ULL, i_vp = 0ULL, i_vn = 0ULL; // Indices for reading the pattern
        dtype up = one, un = zero, vp = one, vn = zero;           // Relative weights for interpolation
        dtype cAZi = one, sAZi = zero, cELi = one, sELi = zero, Cx = one, Cy = zero;
        dtype az = zero, el = zero, sin_gamma = zero, cos_gamma = one, dx = one, dy = zero, dz = zero;

        for (uword o = 0ULL; o < n_out; o++)
        {
            // Check if we need to update the angles for the current output index "o"
            bool update_angles = per_element_angles || per_element_rotation || per_angle_rotation || o == 0ULL;

            // Transform input angles to Cartesian coordinates
            if (per_element_angles || o == 0ULL)
                sAZi = std::sin(*p_az_local), cAZi = std::cos(*p_az_local++),
                sELi = std::sin(*p_el_local), cELi = std::cos(*p_el_local++) + R1,
                Cx = cELi * cAZi, Cy = cELi * sAZi;

            // Apply rotation (Co = R * Ci) for antenna pattern interpolation
            // Transform from Cartesian coordinates to geographic coordinates
            uword Rp_a = per_angle_rotation ? a : 0ULL, Rp_o = per_element_rotation ? o : 0ULL;
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
            if (write_dist)
            {
                dst = dx * p_element_pos[3ULL * o] + dy * p_element_pos[3ULL * o + 1ULL] + dz * p_element_pos[3ULL * o + 2ULL];
                dtype dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
                dtype sgn = signum(dst * dx2 + dst * dy2 + dst * dz2);
                dst *= dst;
                dst = -sgn * std::sqrt(dst * dx2 + dst * dy2 + dst * dz2);
            }

            // Calc. indices for reading the pattern and relative weights for interpolation
            if (update_angles)
            {
                i_up = 0ULL, i_un = 0ULL, up = one, un = zero;
                i_vp = 0ULL, i_vn = 0ULL, vp = one, vn = zero;
                if (n_azimuth != 1ULL)
                {
                    if (*p_azimuth_grid > az) // az is between -pi and first grid point
                    {
                        i_up = n_azimuth - 1ULL;
                        un = (*p_azimuth_grid - az + R0) * *az_diff;
                        un = un > one ? one : un, up = one - un;
                    }
                    else
                    {
                        while (i_up < n_azimuth && p_azimuth_grid[i_up] <= az)
                            i_up++;

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
                    i_vp++;

                if (i_vp == n_elevation)
                    i_vn = --i_vp;
                else if (i_vp != 0ULL)
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
            uword offset = n_pattern_samples * (p_i_element[o] - 1ULL);
            uword iA = i_up * n_elevation + i_vp + offset;
            uword iB = i_un * n_elevation + i_vp + offset;
            uword iC = i_up * n_elevation + i_vn + offset;
            uword iD = i_un * n_elevation + i_vn + offset;
            dtype Vr, Vi, Hr, Hi;

            for (uword VH = 0ULL; VH < 2ULL; VH++)
            {
                dtype fAr, fBr, fCr, fDr, fAi, fBi, fCi, fDi;
                if (VH == 0) // Read the pattern values
                    fAr = p_theta_re[iA] + R0, fAi = p_theta_im[iA],
                    fBr = p_theta_re[iB] + R0, fBi = p_theta_im[iB],
                    fCr = p_theta_re[iC] + R0, fCi = p_theta_im[iC],
                    fDr = p_theta_re[iD] + R0, fDi = p_theta_im[iD];
                else
                    fAr = p_phi_re[iA] + R0, fAi = p_phi_im[iA],
                    fBr = p_phi_re[iB] + R0, fBi = p_phi_im[iB],
                    fCr = p_phi_re[iC] + R0, fCi = p_phi_im[iC],
                    fDr = p_phi_re[iD] + R0, fDi = p_phi_im[iD];

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
                dtype cPhase = gAr * gBr + gAi * gBi; // Cosine of phase
                bool linear_int = cPhase < tS;
                dtype gEr = zero, gEi = zero, ampE = zero, gFr = zero, gFi = zero, ampF = zero, fLr = zero, fLi = zero;

                // Interpolation for point E
                if (linear_int) // Linear interpolation
                    fLr = un * fAr + up * fBr, fLi = un * fAi + up * fBi;
                if (cPhase > tL) // Spherical interpolation
                {
                    dtype Phase = (cPhase >= one) ? R0 : std::acos(cPhase) + R0, sPhase = one / std::sin(Phase),
                          wp = std::sin(up * Phase) * sPhase, wn = std::sin(un * Phase) * sPhase;
                    gEr = wn * gAr + wp * gBr, gEi = wn * gAi + wp * gBi, ampE = un * ampA + up * ampB;
                    if (linear_int) // Mixed mode
                        wp = (tS - cPhase) * dT, wn = one - wp,
                        fLr = wn * gEr * ampE + wp * fLr, fLi = wn * gEi * ampE + wp * fLi;
                }
                if (linear_int)
                    ampE = std::sqrt(fLr * fLr + fLi * fLi), gEr = one / ampE, gEi = fLi * gEr, gEr = fLr * gEr;

                // Interpolation for point F
                cPhase = gCr * gDr + gCi * gDi, linear_int = cPhase < tS;
                if (linear_int)
                    fLr = un * fCr + up * fDr, fLi = un * fCi + up * fDi;
                if (cPhase > tL)
                {
                    dtype Phase = (cPhase >= one) ? R0 : std::acos(cPhase) + R0, sPhase = one / std::sin(Phase),
                          wp = std::sin(up * Phase) * sPhase, wn = std::sin(un * Phase) * sPhase;
                    gFr = wn * gCr + wp * gDr, gFi = wn * gCi + wp * gDi, ampF = un * ampC + up * ampD;
                    if (linear_int)
                        wp = (tS - cPhase) * dT, wn = one - wp,
                        fLr = wn * gFr * ampF + wp * fLr, fLi = wn * gFi * ampF + wp * fLi;
                }
                if (linear_int)
                    ampF = std::sqrt(fLr * fLr + fLi * fLi), gFr = one / ampF, gFi = fLi * gFr, gFr = fLr * gFr;

                // Interpolation for point X
                cPhase = gEr * gFr + gEi * gFi, linear_int = cPhase < tS;
                if (linear_int)
                    fLr = vn * gEr * ampE + vp * gFr * ampF, fLi = vn * gEi * ampE + vp * gFi * ampF;
                if (cPhase > tL)
                {
                    dtype Phase = (cPhase >= one) ? R0 : std::acos(cPhase) + R0, sPhase = one / std::sin(Phase),
                          wp = std::sin(vp * Phase) * sPhase, wn = std::sin(vn * Phase) * sPhase;
                    dtype gLr = wn * gEr + wp * gFr, gLi = wn * gEi + wp * gFi, ampL = vn * ampE + vp * ampF;
                    if (linear_int)
                        wp = (tS - cPhase) * dT, wn = one - wp,
                        fLr = wn * gLr * ampL + wp * fLr, fLi = wn * gLi * ampL + wp * fLi;
                    else
                        fLr = gLr * ampL, fLi = gLi * ampL;
                }

                if (VH == 0)
                    Vr = fLr, Vi = fLi;
                else
                    Hr = fLr, Hi = fLi;
            }

            // Compute and write output
            uword ioa = a * n_out + o;
            p_v_re[ioa] = cos_gamma * Vr - sin_gamma * Hr;
            p_v_im[ioa] = cos_gamma * Vi - sin_gamma * Hi;
            p_h_re[ioa] = sin_gamma * Vr + cos_gamma * Hr;
            p_h_im[ioa] = sin_gamma * Vi + cos_gamma * Hi;

            // Write optional azimuth and elevation angles
            if (write_dist)
                dist->at(ioa) = dst;
            if (write_az)
                azimuth_loc->at(ioa) = az;
            if (write_el)
                elevation_loc->at(ioa) = el;
            if (write_gamma)
                gamma->at(ioa) = std::atan2(sin_gamma, cos_gamma);
        }
    }

    delete[] az_diff;
    delete[] el_diff;
}

// Declare templates
template void qd_arrayant_interpolate(const arma::Cube<float> *e_theta_re, const arma::Cube<float> *e_theta_im,
                                      const arma::Cube<float> *e_phi_re, const arma::Cube<float> *e_phi_im,
                                      const arma::Col<float> *azimuth_grid, const arma::Col<float> *elevation_grid,
                                      const arma::Mat<float> *azimuth, const arma::Mat<float> *elevation,
                                      const arma::Col<unsigned> *i_element, const arma::Cube<float> *orientation,
                                      const arma::Mat<float> *element_pos,
                                      arma::Mat<float> *V_re, arma::Mat<float> *V_im,
                                      arma::Mat<float> *H_re, arma::Mat<float> *H_im,
                                      arma::Mat<float> *dist,
                                      arma::Mat<float> *azimuth_loc, arma::Mat<float> *elevation_loc, arma::Mat<float> *gamma);

template void qd_arrayant_interpolate(const arma::Cube<double> *e_theta_re, const arma::Cube<double> *e_theta_im,
                                      const arma::Cube<double> *e_phi_re, const arma::Cube<double> *e_phi_im,
                                      const arma::Col<double> *azimuth_grid, const arma::Col<double> *elevation_grid,
                                      const arma::Mat<double> *azimuth, const arma::Mat<double> *elevation,
                                      const arma::Col<unsigned> *i_element, const arma::Cube<double> *orientation,
                                      const arma::Mat<double> *element_pos,
                                      arma::Mat<double> *V_re, arma::Mat<double> *V_im,
                                      arma::Mat<double> *H_re, arma::Mat<double> *H_im,
                                      arma::Mat<double> *dist,
                                      arma::Mat<double> *azimuth_loc, arma::Mat<double> *elevation_loc, arma::Mat<double> *gamma);
