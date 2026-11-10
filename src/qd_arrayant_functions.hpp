// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef qd_arrayant_qdant_H
#define qd_arrayant_qdant_H

#include "quadriga_arrayant.hpp"

#include <armadillo>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

#if BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#endif

// Grid resolution in degrees for the pattern generators: values <= 0 select the default, values above 90 are set to 90;
// NaN, infinite values and values below 0.1 deg throw (a 0.1 deg grid has 3601 x 1801 samples per element)
template <typename dtype>
inline dtype qd_arrayant_grid_resolution(dtype res, dtype res_default, const char *name)
{
    if (!std::isfinite(res) || (res > (dtype)0.0 && res < (dtype)0.1))
        throw std::invalid_argument(std::string("Input '") + name + "' must be finite and at least 0.1 degrees, or <= 0 for the default.");
    return (res <= (dtype)0.0) ? res_default : (res > (dtype)90.0 ? (dtype)90.0 : res);
}

// Returns empty string if there was no error or an error message otherwise
template <typename dtype> // float or double
std::string qd_arrayant_qdant_read(const std::string fn, const int id,
                                   std::string *name,
                                   arma::Cube<dtype> *e_theta_re,
                                   arma::Cube<dtype> *e_theta_im,
                                   arma::Cube<dtype> *e_phi_re,
                                   arma::Cube<dtype> *e_phi_im,
                                   arma::Col<dtype> *azimuth_grid,
                                   arma::Col<dtype> *elevation_grid,
                                   arma::Mat<dtype> *element_pos,
                                   arma::Mat<dtype> *coupling_re,
                                   arma::Mat<dtype> *coupling_im,
                                   dtype *center_frequency,
                                   arma::Mat<unsigned> *layout);

// Returns empty string if there was no error or an error message otherwise
template <typename dtype> // float or double
std::string qd_arrayant_qdant_write(const std::string fn, const int id,
                                    const std::string *name,
                                    const arma::Cube<dtype> *e_theta_re,
                                    const arma::Cube<dtype> *e_theta_im,
                                    const arma::Cube<dtype> *e_phi_re,
                                    const arma::Cube<dtype> *e_phi_im,
                                    const arma::Col<dtype> *azimuth_grid,
                                    const arma::Col<dtype> *elevation_grid,
                                    const arma::Mat<dtype> *element_pos,
                                    const arma::Mat<dtype> *coupling_re,
                                    const arma::Mat<dtype> *coupling_im,
                                    const dtype *center_frequency,
                                    const arma::Mat<unsigned> *layout,
                                    unsigned *id_in_file);

// Block-diagonal coupling [C1 0; 0 C2] of two arrays with n_el1 and n_el2 elements (arrayant::append and
// arrayant_concat_multi). An empty coupling_re is an identity block (one port per element); an empty coupling_im is
// zero. Outputs are set to [n_el1 + n_el2, n_ports1 + n_ports2]; they must not alias the inputs.
template <typename dtype>
inline void qd_arrayant_coupling_block_diag(const arma::Mat<dtype> &coupling_re1, const arma::Mat<dtype> &coupling_im1,
                                            arma::uword n_el1,
                                            const arma::Mat<dtype> &coupling_re2, const arma::Mat<dtype> &coupling_im2,
                                            arma::uword n_el2,
                                            arma::Mat<dtype> &coupling_re, arma::Mat<dtype> &coupling_im)
{
    const arma::uword n_ports1 = coupling_re1.is_empty() ? n_el1 : coupling_re1.n_cols;
    const arma::uword n_ports2 = coupling_re2.is_empty() ? n_el2 : coupling_re2.n_cols;
    coupling_re.zeros(n_el1 + n_el2, n_ports1 + n_ports2);
    coupling_im.zeros(n_el1 + n_el2, n_ports1 + n_ports2);

    if (n_el1 != 0 && n_ports1 != 0)
    {
        if (coupling_re1.is_empty())
            coupling_re.submat(0, 0, n_el1 - 1, n_ports1 - 1).eye();
        else
            coupling_re.submat(0, 0, n_el1 - 1, n_ports1 - 1) = coupling_re1;
        if (!coupling_im1.is_empty())
            coupling_im.submat(0, 0, n_el1 - 1, n_ports1 - 1) = coupling_im1;
    }
    if (n_el2 != 0 && n_ports2 != 0)
    {
        const arma::uword r1 = n_el1 + n_el2 - 1, c1 = n_ports1 + n_ports2 - 1;
        if (coupling_re2.is_empty())
            coupling_re.submat(n_el1, n_ports1, r1, c1).eye();
        else
            coupling_re.submat(n_el1, n_ports1, r1, c1) = coupling_re2;
        if (!coupling_im2.is_empty())
            coupling_im.submat(n_el1, n_ports1, r1, c1) = coupling_im2;
    }
}

// Resolves the kernel selector 'use_kernel' of the arrayant interpolation functions to the kernel that runs: 1
// (GENERIC) for 0 (auto) and 1, 2 (AVX2) for 2. Throws std::invalid_argument for other values, and for 2 when the
// AVX2 kernel is not available (BUILD_WITH_AVX2 and the CPU check) or when a pattern cube has more than INT_MAX
// values (int32 gather offsets of the AVX2 kernel); n_pattern_values: number of values of the largest pattern cube
static inline int qd_arrayant_interpolate_kernel(int use_kernel, unsigned long long n_pattern_values)
{
    if (use_kernel < 0 || use_kernel > 2)
        throw std::invalid_argument("Input 'use_kernel' must be 0, 1 or 2.");
    if (use_kernel != 2)
        return 1; // Auto selects GENERIC
#if BUILD_WITH_AVX2
    const bool avx2_available = runtime_AVX2_Check();
#else
    const bool avx2_available = false;
#endif
    if (!avx2_available)
        throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");
    if (n_pattern_values > (unsigned long long)INT_MAX)
        throw std::invalid_argument("AVX2 kernel requested, but the antenna pattern has more than INT_MAX samples.");
    return 2;
}

// Interpolates the pattern of 'ant' (arrayant_interpolate without its argument handling), for library functions that
// have validated the arrayant themselves (e.g. with is_valid) and own the outputs
// - Preconditions, not checked: 'ant' is valid; no output aliases an input, a data member of 'ant' or another output
// - Checks, before a kernel runs: i_element not empty and its entries < n_elements; the sizes of the angles, the
//   orientation, the positions and the outputs; n_ang <= INT_MAX; kernel 1 or 2, for 2 as qd_arrayant_interpolate_kernel;
//   throws std::invalid_argument on a violation. With n_ang = 0 it returns after the checks.
// - All outputs are pre-sized by the caller and completely written; optional outputs may be nullptr
template <typename dtype>
void qd_arrayant_interpolate_dispatch(const quadriga_lib::arrayant<dtype> &ant, // Valid array antenna with n_elements elements
                                      const arma::Mat<dtype> &azimuth,          // Azimuth angles in GCS in rad; [1, n_ang] or [n_out, n_ang]
                                      const arma::Mat<dtype> &elevation,        // Elevation angles in GCS in rad; same size as azimuth
                                      const arma::uvec &i_element,              // Element indices, 0-based; [n_out], n_out >= 1
                                      const arma::Cube<dtype> &orientation,     // Euler angles in rad; [3, 1 or n_out, 1 or n_ang]
                                      const arma::Mat<dtype> &element_pos,      // Element positions in m; [3, n_out]
                                      arma::Mat<dtype> &V_re,                   // Output: e-theta field, real part; [n_out, n_ang]
                                      arma::Mat<dtype> &V_im,                   // Output: e-theta field, imaginary part; [n_out, n_ang]
                                      arma::Mat<dtype> &H_re,                   // Output: e-phi field, real part; [n_out, n_ang]
                                      arma::Mat<dtype> &H_im,                   // Output: e-phi field, imaginary part; [n_out, n_ang]
                                      arma::Mat<dtype> *dist,                   // Optional output: projected distance in m; [n_out, n_ang] or nullptr
                                      arma::Mat<dtype> *azimuth_loc,            // Optional output: local azimuth in rad; [n_out, n_ang] or nullptr
                                      arma::Mat<dtype> *elevation_loc,          // Optional output: local elevation in rad; [n_out, n_ang] or nullptr
                                      arma::Mat<dtype> *gamma,                  // Optional output: polarization rotation angle in rad; [n_out, n_ang] or nullptr
                                      int kernel);                              // Kernel: 1 = GENERIC, 2 = AVX2

#endif
