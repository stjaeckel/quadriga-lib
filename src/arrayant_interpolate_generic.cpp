// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib_generic_functions.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>

// Scalar kernel of arrayant_interpolate; preconditions in quadriga_lib_generic_functions.hpp
template <typename dtype>
void qd_arrayant_interpolate_GENERIC(const dtype *e_theta_re, const dtype *e_theta_im, const dtype *e_phi_re, const dtype *e_phi_im,
                                     const size_t n_elevation, const size_t n_azimuth,
                                     const dtype *azimuth_grid, const dtype *elevation_grid,
                                     const dtype *azimuth, const dtype *elevation, const size_t n_angle_rows, const size_t n_ang,
                                     const dtype *rotation, const size_t n_rotation_cols, const size_t n_rotation_slices,
                                     const dtype *element_pos, const unsigned long long *i_element, const size_t n_out,
                                     dtype *V_re, dtype *V_im, dtype *H_re, dtype *H_im,
                                     dtype *dist, dtype *azimuth_loc, dtype *elevation_loc, dtype *gamma)
{
    const size_t n_pattern_samples = n_azimuth * n_elevation; // Number of samples per element
    const bool per_element_angles = n_angle_rows > 1;
    const bool per_element_rotation = n_rotation_cols > 1;
    const bool per_angle_rotation = n_rotation_slices > 1;

    // Pointers for direct memory access
    const dtype *p_theta_re = e_theta_re, *p_theta_im = e_theta_im; // Vertical pattern
    const dtype *p_phi_re = e_phi_re, *p_phi_im = e_phi_im;         // Horizontal pattern
    const dtype *p_az_global = azimuth, *p_el_global = elevation;   // Angles
    const dtype *p_azimuth_grid = azimuth_grid, *p_elevation_grid = elevation_grid;
    const dtype *p_element_pos = element_pos;
    const unsigned long long *p_i_element = i_element;
    dtype *p_v_re = V_re, *p_v_im = V_im, *p_h_re = H_re, *p_h_im = H_im;
    dtype *p_dist = dist, *p_azimuth_loc = azimuth_loc, *p_elevation_loc = elevation_loc, *p_gamma = gamma;

    // Declare constants at compile time to avoid unnecessary type conversions
    const dtype pi_double = dtype(6.28318530717958647692528676655900576); // 2 pi
    const dtype R1 = std::numeric_limits<dtype>::epsilon();
    const dtype R0 = R1 * R1 * R1;
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
#pragma omp parallel for if (n_ang > 1 && n_ang * n_out >= 256)
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
            const dtype *Rp = &rotation[9 * (Rp_a * n_rotation_cols + Rp_o)];
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
                const dtype sgn_arg = dst * dx2 + dst * dy2 + dst * dz2;
                const dtype sgn = dtype((dtype(0.0) < sgn_arg) - (sgn_arg < dtype(0.0))); // Signum (-1, 0 or 1)
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
                    // az is between -pi and first grid point; also taken for NaN (from NaN or infinite input angles),
                    // so that the indices stay in bounds and the NaN weights give a NaN output
                    if (!(*p_azimuth_grid <= az))
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
            size_t offset = n_pattern_samples * (size_t)p_i_element[o];
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

template void qd_arrayant_interpolate_GENERIC(const float *e_theta_re, const float *e_theta_im, const float *e_phi_re, const float *e_phi_im,
                                              const size_t n_elevation, const size_t n_azimuth,
                                              const float *azimuth_grid, const float *elevation_grid,
                                              const float *azimuth, const float *elevation, const size_t n_angle_rows, const size_t n_ang,
                                              const float *rotation, const size_t n_rotation_cols, const size_t n_rotation_slices,
                                              const float *element_pos, const unsigned long long *i_element, const size_t n_out,
                                              float *V_re, float *V_im, float *H_re, float *H_im,
                                              float *dist, float *azimuth_loc, float *elevation_loc, float *gamma);

template void qd_arrayant_interpolate_GENERIC(const double *e_theta_re, const double *e_theta_im, const double *e_phi_re, const double *e_phi_im,
                                              const size_t n_elevation, const size_t n_azimuth,
                                              const double *azimuth_grid, const double *elevation_grid,
                                              const double *azimuth, const double *elevation, const size_t n_angle_rows, const size_t n_ang,
                                              const double *rotation, const size_t n_rotation_cols, const size_t n_rotation_slices,
                                              const double *element_pos, const unsigned long long *i_element, const size_t n_out,
                                              double *V_re, double *V_im, double *H_re, double *H_im,
                                              double *dist, double *azimuth_loc, double *elevation_loc, double *gamma);
