// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef qd_arrayant_qdant_H
#define qd_arrayant_qdant_H

#include "quadriga_arrayant.hpp"

#include <armadillo>
#include <atomic>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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

// Validity cache of an arrayant (member check_ptr): the data pointers of the members at the last passed check.
// The cache matches when all pointers are unchanged and the member sizes still fit together (Armadillo can keep a
// data pointer when a member is resized). A matching cache does not prove that the values are unchanged.
template <typename dtype>
inline bool qd_arrayant_cache_matches(const quadriga_lib::arrayant<dtype> &ant)
{
    // Relaxed atomic loads: is_valid may mark the object in another thread at the same time
    const dtype *p[9];
    for (int i = 0; i < 9; ++i)
        p[i] = ant.check_ptr[i].load(std::memory_order_relaxed);
    return arma::size(ant.e_theta_im) == arma::size(ant.e_theta_re) &&
           arma::size(ant.e_phi_re) == arma::size(ant.e_theta_re) &&
           arma::size(ant.e_phi_im) == arma::size(ant.e_theta_re) &&
           ant.azimuth_grid.n_elem == ant.e_theta_re.n_cols &&
           ant.elevation_grid.n_elem == ant.e_theta_re.n_rows &&
           ant.element_pos.n_rows == 3 && ant.element_pos.n_cols == ant.e_theta_re.n_slices &&
           ant.coupling_re.n_rows == ant.e_theta_re.n_slices &&
           arma::size(ant.coupling_im) == arma::size(ant.coupling_re) &&
           p[0] != nullptr && p[1] != nullptr && p[2] != nullptr && p[3] != nullptr && p[4] != nullptr && p[5] != nullptr &&
           p[0] == ant.e_theta_re.memptr() && p[1] == ant.e_theta_im.memptr() &&
           p[2] == ant.e_phi_re.memptr() && p[3] == ant.e_phi_im.memptr() &&
           p[4] == ant.azimuth_grid.memptr() && p[5] == ant.elevation_grid.memptr() &&
           p[6] == ant.element_pos.memptr() && p[7] == ant.coupling_re.memptr() && p[8] == ant.coupling_im.memptr();
}

// Marks an arrayant as validated: sets the cache to the current data pointers (relaxed atomic stores, so threads that
// read the object at the same time see either the old or the new mark)
template <typename dtype>
inline void qd_arrayant_mark_valid(const quadriga_lib::arrayant<dtype> &ant)
{
    const dtype *p[9] = {ant.e_theta_re.memptr(), ant.e_theta_im.memptr(), ant.e_phi_re.memptr(), ant.e_phi_im.memptr(),
                         ant.azimuth_grid.memptr(), ant.elevation_grid.memptr(), ant.element_pos.memptr(),
                         ant.coupling_re.memptr(), ant.coupling_im.memptr()};
    for (int i = 0; i < 9; ++i)
        ant.check_ptr[i].store(p[i], std::memory_order_relaxed);
}

// Removes the mark of an arrayant
template <typename dtype>
inline void qd_arrayant_clear_mark(const quadriga_lib::arrayant<dtype> &ant)
{
    for (auto &p : ant.check_ptr)
        p.store(nullptr, std::memory_order_relaxed);
}

// Full validity check of an arrayant (the checks of arrayant::is_valid), without writing to the object
// - Returns an empty string if valid or the error message of the first failed check
// - For library functions that need the full check: read only, so threads may check a shared object at the same time
template <typename dtype>
std::string qd_arrayant_check(const quadriga_lib::arrayant<dtype> &ant);

// Validity check for library functions: passes at once when the cache matches, else runs the full check
// - Read only, so threads may check a shared object at the same time
template <typename dtype>
inline std::string qd_arrayant_is_valid_quick(const quadriga_lib::arrayant<dtype> &ant)
{
    return qd_arrayant_cache_matches(ant) ? std::string() : qd_arrayant_check(ant);
}

// arrayant_is_valid_multi with the per-entry check of qd_arrayant_is_valid_quick (read only)
template <typename dtype>
std::string qd_arrayant_is_valid_multi_quick(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);

// arrayant_is_valid_multi with the per-entry full check qd_arrayant_check (read only)
template <typename dtype>
std::string qd_arrayant_check_multi(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);

#endif
