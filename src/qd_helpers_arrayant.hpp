// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the antenna array sources and the channel generators that use arrayants: value ranges, validity
// checks and their cache, the interpolation dispatch and the field scaling of the pattern metrics

#ifndef qd_helpers_arrayant_hpp
#define qd_helpers_arrayant_hpp

#include "quadriga_arrayant.hpp"
#include "qd_helpers_shared.hpp"

#include <armadillo>
#include <atomic>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// Check if elements of a vector are in a specific range; NaN is out of range
// With check_sorted, the values must also be strictly increasing
template <typename dtype>
static inline bool qd_in_range(const dtype *data, size_t n_elem, dtype min, dtype max, bool inclusive = true, bool check_sorted = false)
{
    if (inclusive)
    {
        for (size_t i = 0; i < n_elem; ++i)
        {
            if (!(data[i] >= min && data[i] <= max))
                return false;
            if (check_sorted && i != 0 && data[i] <= data[i - 1])
                return false;
        }
    }
    else
    {
        for (size_t i = 0; i < n_elem; ++i)
        {
            if (!(data[i] > min && data[i] < max))
                return false;
            if (check_sorted && i != 0 && data[i] <= data[i - 1])
                return false;
        }
    }
    return true;
}

// Grid resolution in degrees for the pattern generators: NaN and values <= 0 select the default; infinite values and
// values below 0.1 deg (a 0.1 deg grid has 3601 x 1801 samples per element) or above 90 deg throw
template <typename dtype>
static inline dtype qd_arrayant_grid_resolution(dtype res, dtype res_default, const char *name)
{
    if (std::isinf(res) || (res > (dtype)0.0 && res < (dtype)0.1) || res > (dtype)90.0)
        throw std::invalid_argument(std::string("Input '") + name + "' must be between 0.1 and 90 degrees, or NaN or <= 0 for the default.");
    return (std::isnan(res) || res <= (dtype)0.0) ? res_default : res;
}

// Inverse of the largest absolute field value of n samples (calc_directivity, export_pattern_obj, trim_pattern);
// scaling by it keeps the power of double patterns with amplitudes above about 1e154 inside the double range (only
// called when the unscaled power overflowed)
template <typename dtype>
static inline double qd_inverse_max_field(const dtype *a, const dtype *b, const dtype *c, const dtype *d, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double x = std::max(std::max(std::abs((double)a[i]), std::abs((double)b[i])), std::max(std::abs((double)c[i]), std::abs((double)d[i])));
        m = x > m ? x : m;
    }
    return 1.0 / m;
}

// Resolves the kernel selector 'use_kernel' of the arrayant interpolation functions to the kernel that runs: 1
// (GENERIC) for 0 (auto) and 1, 2 (AVX2) for 2. Throws std::invalid_argument for other values, and for 2 when the
// AVX2 kernel is not available (BUILD_WITH_AVX2 and the CPU check) or when a pattern cube has more than INT_MAX
// values (int32 gather offsets of the AVX2 kernel); n_pattern_values: number of values of the largest pattern cube
static inline int qd_interpolate_pattern_kernel(int use_kernel, unsigned long long n_pattern_values)
{
    const int kernel = qd_kernel_auto_avx2(use_kernel == 0 ? 1 : use_kernel); // Auto selects GENERIC
    if (kernel == 2 && n_pattern_values > (unsigned long long)INT_MAX)
        throw std::invalid_argument("AVX2 kernel requested, but the antenna pattern has more than INT_MAX samples.");
    return kernel;
}

// Interpolates the pattern of 'ant' (interpolate_pattern without its argument handling), for library functions that
// have validated the arrayant themselves (e.g. with is_valid) and own the outputs
// - Preconditions, not checked: 'ant' is valid; no output aliases an input, a data member of 'ant' or another output
// - Checks, before a kernel runs: i_element not empty and its entries < n_elements; the sizes of the angles, the
//   orientation, the positions and the outputs; n_ang <= INT_MAX; kernel 1 or 2, for 2 as qd_interpolate_pattern_kernel;
//   throws std::invalid_argument on a violation. With n_ang = 0 it returns after the checks.
// - All outputs are pre-sized by the caller and completely written; optional outputs may be nullptr
template <typename dtype>
void qd_interpolate_pattern_dispatch(const quadriga_lib::arrayant<dtype> &ant, // Valid antenna array with n_elements elements
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
                                      int kernel,                               // Kernel: 1 = GENERIC, 2 = AVX2
                                      arma::Mat<std::complex<dtype>> *V = nullptr,  // Optional complex output replacing V_re / V_im (not written); [n_out, n_ang]
                                      arma::Mat<std::complex<dtype>> *H = nullptr); // Optional complex output replacing H_re / H_im (not written); [n_out, n_ang]

// Access to the private validity cache of an arrayant (member check_ptr) for the library and its tests
namespace quadriga_lib
{
    struct qd_arrayant_access
    {
        template <typename dtype>
        static auto &check_ptr(const arrayant<dtype> &ant) { return ant.check_ptr; }
    };
}

// Validity cache of an arrayant (member check_ptr): the data pointers of the members at the last passed check.
// The cache matches when all pointers are unchanged and the member sizes still fit together (Armadillo can keep a
// data pointer when a member is resized); empty element_pos (zeros) and port weights (identity) fit.
// A matching cache does not prove that the values are unchanged.
template <typename dtype>
static inline bool qd_arrayant_cache_matches(const quadriga_lib::arrayant<dtype> &ant)
{
    // Relaxed atomic loads: is_valid may mark the object in another thread at the same time
    const dtype *p[9];
    for (int i = 0; i < 9; ++i)
        p[i] = quadriga_lib::qd_arrayant_access::check_ptr(ant)[i].load(std::memory_order_relaxed);
    return arma::size(ant.e_theta_im) == arma::size(ant.e_theta_re) &&
           arma::size(ant.e_phi_re) == arma::size(ant.e_theta_re) &&
           arma::size(ant.e_phi_im) == arma::size(ant.e_theta_re) &&
           ant.azimuth_grid.n_elem == ant.e_theta_re.n_cols &&
           ant.elevation_grid.n_elem == ant.e_theta_re.n_rows &&
           (ant.element_pos.is_empty() || (ant.element_pos.n_rows == 3 && ant.element_pos.n_cols == ant.e_theta_re.n_slices)) &&
           (ant.port_weights_re.is_empty() ? ant.port_weights_im.is_empty() : ant.port_weights_re.n_rows == ant.e_theta_re.n_slices) &&
           (ant.port_weights_im.is_empty() || arma::size(ant.port_weights_im) == arma::size(ant.port_weights_re)) &&
           p[0] != nullptr && p[1] != nullptr && p[2] != nullptr && p[3] != nullptr && p[4] != nullptr && p[5] != nullptr &&
           p[0] == ant.e_theta_re.memptr() && p[1] == ant.e_theta_im.memptr() &&
           p[2] == ant.e_phi_re.memptr() && p[3] == ant.e_phi_im.memptr() &&
           p[4] == ant.azimuth_grid.memptr() && p[5] == ant.elevation_grid.memptr() &&
           p[6] == ant.element_pos.memptr() && p[7] == ant.port_weights_re.memptr() && p[8] == ant.port_weights_im.memptr();
}

// Marks an arrayant as validated: sets the cache to the current data pointers (relaxed atomic stores, so threads that
// read the object at the same time see either the old or the new mark)
template <typename dtype>
static inline void qd_arrayant_mark_valid(const quadriga_lib::arrayant<dtype> &ant)
{
    const dtype *p[9] = {ant.e_theta_re.memptr(), ant.e_theta_im.memptr(), ant.e_phi_re.memptr(), ant.e_phi_im.memptr(),
                         ant.azimuth_grid.memptr(), ant.elevation_grid.memptr(), ant.element_pos.memptr(),
                         ant.port_weights_re.memptr(), ant.port_weights_im.memptr()};
    for (int i = 0; i < 9; ++i)
        quadriga_lib::qd_arrayant_access::check_ptr(ant)[i].store(p[i], std::memory_order_relaxed);
}

// Removes the mark of an arrayant
template <typename dtype>
static inline void qd_arrayant_clear_mark(const quadriga_lib::arrayant<dtype> &ant)
{
    for (auto &p : quadriga_lib::qd_arrayant_access::check_ptr(ant))
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
static inline std::string qd_arrayant_is_valid_quick(const quadriga_lib::arrayant<dtype> &ant)
{
    return qd_arrayant_cache_matches(ant) ? std::string() : qd_arrayant_check(ant);
}

// Validity check of the TX and RX arrays of the channel generators (qd_arrayant_is_valid_quick); throws
// std::invalid_argument with the message prefixed by "Input 'tx_array' is not valid: " or "Input 'rx_array' is not valid: "
template <typename dtype>
static inline void qd_arrayant_check_tx_rx(const quadriga_lib::arrayant<dtype> &tx_array, const quadriga_lib::arrayant<dtype> &rx_array)
{
    std::string error_message = qd_arrayant_is_valid_quick(tx_array);
    if (!error_message.empty())
        throw std::invalid_argument("Input 'tx_array' is not valid: " + error_message);
    error_message = qd_arrayant_is_valid_quick(rx_array);
    if (!error_message.empty())
        throw std::invalid_argument("Input 'rx_array' is not valid: " + error_message);
}

// Throws std::invalid_argument if one of the outputs out[0 .. N-1] is a data member of 'ant' or shares memory with
// one; nullptr outputs are skipped
template <typename T, std::size_t N, typename dtype>
static inline void qd_check_alias_arrayant(const T *const (&out)[N], const char *const (&name)[N],
                                           const quadriga_lib::arrayant<dtype> &ant)
{
    for (std::size_t i = 0; i < N; ++i)
        qd_check_alias(out[i], name[i], &ant.e_theta_re, &ant.e_theta_im, &ant.e_phi_re, &ant.e_phi_im, &ant.azimuth_grid,
                       &ant.elevation_grid, &ant.element_pos, &ant.port_weights_re, &ant.port_weights_im);
}

// is_valid of a vector with the per-entry check of qd_arrayant_is_valid_quick (read only)
template <typename dtype>
std::string qd_arrayant_vec_is_valid_quick(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);

// is_valid of a vector with the per-entry full check qd_arrayant_check (read only)
template <typename dtype>
std::string qd_arrayant_vec_check(const std::vector<quadriga_lib::arrayant<dtype>> &arrayant_vec);

#endif
