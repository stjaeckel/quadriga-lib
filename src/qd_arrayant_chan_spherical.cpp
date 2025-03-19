// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <stdexcept>
#include <cstring>   // std::memcopy
#include <iomanip>   // std::setprecision
#include <algorithm> // std::replace
#include <iostream>

#include "quadriga_arrayant.hpp"
#include "qd_arrayant_functions.hpp"

// Helper function "quick_geo2cart"
template <typename dtype>
static inline void quick_geo2cart(size_t n,                  // Number of values
                                  const dtype *az,           // Input azimuth angles
                                  dtype *x, dtype *y,        // 2D Output coordinates (x, y)
                                  const dtype *el = nullptr, // Input elevation angles (optional)
                                  dtype *z = nullptr,        // Output z-coordinate (optional)
                                  const dtype *r = nullptr)  // Input vector length (optional)
{
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);
    for (size_t in = 0; in < n; ++in)
    {
        dtype ca = az[in], sa = std::sin(ca);
        ca = std::cos(ca);

        dtype ce = (el == nullptr) ? one : std::cos(el[in]);
        dtype se = (el == nullptr) ? zero : std::sin(el[in]);
        dtype le = (r == nullptr) ? one : r[in];

        x[in] = le * ce * ca;
        y[in] = le * ce * sa;

        if (z != nullptr)
            z[in] = le * se;
    }
}

// Helper function "quick_cart2geo"
template <typename dtype>
static inline void quick_cart2geo(size_t n,                       // Number of values
                                  dtype *az,                      // Output azimuth angles
                                  const dtype *x, const dtype *y, // 2D Input coordinates (x, y)
                                  dtype *el = nullptr,            // Output elevation angles (optional)
                                  const dtype *z = nullptr,       // Input z-coordinate (optional)
                                  dtype *r = nullptr)             // Output vector length (optional)
{
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);
    for (size_t in = 0; in < n; ++in)
    {
        dtype xx = x[in], yy = y[in], zz = (z == nullptr) ? zero : z[in];
        dtype le = std::sqrt(xx * xx + yy * yy + zz * zz);

        if (r != nullptr)
            r[in] = le;

        le = one / le;
        xx *= le, yy *= le, zz *= le;
        xx = xx > one ? one : xx, yy = yy > one ? one : yy, zz = zz > one ? one : zz;

        if (az != nullptr)
            az[in] = std::atan2(yy, xx);

        if (el != nullptr)
            el[in] = std::asin(zz);
    }
}

// Helper function "quick_multiply_3_mat"
// Calculates the matrix product X = A^T * B * C
// A, C can be NULL
template <typename dtype>
static inline void quick_multiply_3_mat(const dtype *A, // n rows, m columns
                                        const dtype *B, // n rows, o columns
                                        const dtype *C, // o rows, p columns
                                        dtype *X,       // m rows, p columns
                                        size_t n, size_t m, size_t o, size_t p)
{
    // Avoid expensive typecasts
    constexpr dtype zero = dtype(0.0), one = dtype(1.0);

    // Calculate the output row by row
    for (size_t im = 0; im < m; ++im)
    {
        for (size_t ip = 0; ip < p; ip++) // Initialize output to zero
            X[ip * m + im] = zero;

        // Process temporary matrix T = A^H * B column-wise
        for (size_t io = 0; io < o; ++io)
        {
            // Calculate one value of the temporary matrix T
            dtype t = zero;
            for (size_t in = 0; in < n; ++in)
            {
                dtype a = (A == nullptr) ? (im == in ? one : zero) : A[im * n + in];
                t += a * B[io * n + in];
            }

            // Update all values of an entire row of the output matrix X = T * C
            for (size_t ip = 0; ip < p; ++ip)
            {
                dtype c = (C == nullptr) ? (io == ip ? one : zero) : C[ip * o + io];
                X[ip * m + im] += t * c;
            }
        }
    }
}

// Helper function "quick_multiply_3_complex_mat"
// Calculates the matrix product X = A^T * B * C
// Ar, Ai, Cr, Ci can be NULL
template <typename dtype>
static inline void quick_multiply_3_complex_mat(const dtype *Ar, const dtype *Ai, // n rows, m columns
                                                const dtype *Br, const dtype *Bi, // n rows, o columns
                                                const dtype *Cr, const dtype *Ci, // o rows, p columns
                                                dtype *Xr, dtype *Xi,             // m rows, p columns
                                                size_t n, size_t m, size_t o, size_t p)
{
    // Avoid expensive typecasts
    constexpr dtype zero = (dtype)0.0, one = (dtype)1.0;

    // Calculate the output row by row
    for (size_t im = 0; im < m; ++im)
    {
        // Initialize output to zero
        for (size_t ip = 0; ip < p; ++ip)
            Xr[ip * m + im] = zero, Xi[ip * m + im] = zero;

        // Process temporary matrix T = A^H * B column-wise
        for (size_t io = 0; io < o; ++io)
        {
            // Calculate one value of the temporary matrix T
            dtype tR = zero, tI = zero;
            for (size_t in = 0; in < n; ++in)
            {
                dtype a_real = (Ar == nullptr) ? (im == in ? one : zero) : Ar[im * n + in];
                dtype a_imag = (Ai == nullptr) ? zero : Ai[im * n + in];
                tR += a_real * Br[io * n + in] - a_imag * Bi[io * n + in];
                tI += a_real * Bi[io * n + in] + a_imag * Br[io * n + in];
            }

            // Update all values of an entire row of the output matrix X = T * C
            for (size_t ip = 0; ip < p; ++ip)
            {
                dtype c_real = (Cr == nullptr) ? (io == ip ? one : zero) : Cr[ip * o + io];
                dtype c_imag = (Ci == nullptr) ? zero : Ci[ip * o + io];
                Xr[ip * m + im] += tR * c_real - tI * c_imag;
                Xi[ip * m + im] += tR * c_imag + tI * c_real;
            }
        }
    }
}

// Helper-function "quick_rotate"
template <typename dtype>
static inline void quick_rotate_inplace(dtype bank, dtype tilt, dtype heading, dtype *data3xN, size_t N)
{
    dtype cc = std::cos(bank), sc = std::sin(bank);
    dtype cb = std::cos(tilt), sb = std::sin(tilt);
    dtype ca = std::cos(heading), sa = std::sin(heading);

    dtype R[9]; // Rotation Matrix
    R[0] = ca * cb;
    R[1] = sa * cb;
    R[2] = -sb;
    R[3] = ca * sb * sc - sa * cc;
    R[4] = sa * sb * sc + ca * cc;
    R[5] = cb * sc;
    R[6] = ca * sb * cc + sa * sc;
    R[7] = sa * sb * cc - ca * sc;
    R[8] = cb * cc;

    for (size_t i = 0; i < N; ++i)
    {
        size_t ix = 3 * i, iy = ix + 1, iz = ix + 2;
        dtype a = R[0] * data3xN[ix] + R[3] * data3xN[iy] + R[6] * data3xN[iz];
        dtype b = R[1] * data3xN[ix] + R[4] * data3xN[iy] + R[7] * data3xN[iz];
        dtype c = R[2] * data3xN[ix] + R[5] * data3xN[iy] + R[8] * data3xN[iz];
        data3xN[ix] = a, data3xN[iy] = b, data3xN[iz] = c;
    }
}

// Helper function "quick_power_mat"
// - Calculates X = abs( A ).^2 + abs ( B ).^2
// - Optional normalization of the columns by their sum-power
// - Returns identity matrix normalization is true and inputs A/B are NULL
template <typename dtype>
static inline void quick_power_mat(size_t n, size_t m,                                   // Matrix dimensions (n=rows, m=columns)
                                   dtype *X,                                             // Output X with n rows, m columns
                                   bool normalize_columns = false,                       // Optional normalization
                                   const dtype *Ar = nullptr, const dtype *Ai = nullptr, // Input A with n rows, m columns
                                   const dtype *Br = nullptr, const dtype *Bi = nullptr) // Input B with n rows, m columns
{
    constexpr dtype zero = (dtype)0.0, one = (dtype)1.0, limit = (dtype)1.0e-10;
    dtype avg = one / (dtype)n;

    for (size_t im = 0; im < n * m; im += n)
    {
        dtype sum = zero;
        for (size_t in = im; in < im + n; ++in)
        {
            X[in] = zero;
            X[in] += (Ar == nullptr) ? zero : Ar[in] * Ar[in];
            X[in] += (Ai == nullptr) ? zero : Ai[in] * Ai[in];
            X[in] += (Br == nullptr) ? zero : Br[in] * Br[in];
            X[in] += (Bi == nullptr) ? zero : Bi[in] * Bi[in];
            sum += X[in];
        }
        if (normalize_columns)
        {
            if (Ar == nullptr && Ai == nullptr && Br == nullptr && Bi == nullptr) // Return identity matrix
                X[im + im / n] = one;
            else if (sum > limit) // Scale values by sum
            {
                sum = one / sum;
                for (size_t in = im; in < im + n; ++in)
                    X[in] *= sum;
            }
            else
                for (size_t in = im; in < im + n; ++in)
                    X[in] = avg;
        }
    }
}

// Calculate channel coefficients for spherical waves
template <typename dtype>
void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<dtype> *tx_array, const quadriga_lib::arrayant<dtype> *rx_array,
                                          dtype Tx, dtype Ty, dtype Tz, dtype Tb, dtype Tt, dtype Th,
                                          dtype Rx, dtype Ry, dtype Rz, dtype Rb, dtype Rt, dtype Rh,
                                          const arma::Mat<dtype> *fbs_pos, const arma::Mat<dtype> *lbs_pos,
                                          const arma::Col<dtype> *path_gain, const arma::Col<dtype> *path_length, const arma::Mat<dtype> *M,
                                          arma::Cube<dtype> *coeff_re, arma::Cube<dtype> *coeff_im, arma::Cube<dtype> *delay,
                                          dtype center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                          arma::Cube<dtype> *aod, arma::Cube<dtype> *eod, arma::Cube<dtype> *aoa, arma::Cube<dtype> *eoa)
{
    // Constants
    constexpr dtype los_limit = (dtype)1.0e-4;
    constexpr dtype zero = (dtype)0.0;
    constexpr dtype rC = dtype(1.0 / 299792458.0); // 1 / (Speed of light)
    dtype wavelength = (center_frequency > zero) ? (dtype)299792458.0 / center_frequency : (dtype)1.0;
    dtype wave_number = (dtype)2.095845021951682e-08 * center_frequency; // 2 * pi / C

    // Catch NULL-Pointers
    std::string error_message;
    if (tx_array == nullptr || rx_array == nullptr ||
        fbs_pos == nullptr || lbs_pos == nullptr || path_gain == nullptr || path_length == nullptr || M == nullptr ||
        coeff_re == nullptr || coeff_im == nullptr || delay == nullptr)
    {
        error_message = "Mandatory inputs and outputs cannot be NULL";
        throw std::invalid_argument(error_message.c_str());
    }

    // Check if the antennas are valid
    error_message = tx_array->is_valid();
    if (error_message.length() != 0)
    {
        error_message = "Transmit antenna: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }
    error_message = rx_array->is_valid();
    if (error_message.length() != 0)
    {
        error_message = "Receive antenna: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }

    // Check if the number of paths is consistent in all inputs
    if (fbs_pos->n_elem == 0 || lbs_pos->n_elem == 0 || path_gain->n_elem == 0 || path_length->n_elem == 0 || M->n_elem == 0)
    {
        error_message = "Missing data for any of: fbs_pos, lbs_pos, path_gain, path_length, M";
        throw std::invalid_argument(error_message.c_str());
    }
    if (fbs_pos->n_rows != 3 || lbs_pos->n_rows != 3 || M->n_rows != 8)
    {
        error_message = "Inputs 'fbs_pos' and 'lbs_pos' must have 3 rows; 'M' must have 8 rows.";
        throw std::invalid_argument(error_message.c_str());
    }

    // 64-bit integers used by Armadillo
    const arma::uword n_path = fbs_pos->n_cols;                        // Number of paths
    const arma::uword n_out = add_fake_los_path ? n_path + 1 : n_path; // Number of output paths
    const arma::uword n_tx = tx_array->e_theta_re.n_slices;            // Number of TX antenna elements before coupling
    const arma::uword n_rx = rx_array->e_theta_re.n_slices;            // Number of RX antenna elements before coupling
    const arma::uword n_links = n_rx * n_tx;                           // Number of MIMO channel coefficients per path (n_rx * n_tx)
    const arma::uword n_tx_ports = tx_array->n_ports();                // Number of TX antenna elements after coupling
    const arma::uword n_rx_ports = rx_array->n_ports();                // Number of RX antenna elements after coupling
    const arma::uword n_ports = n_tx_ports * n_rx_ports;               // Total number of ports

    // Size types for loops and indexing
    // const size_t n_out_t = (size_t)n_out;
    const size_t n_tx_t = (size_t)n_tx;
    const size_t n_rx_t = (size_t)n_rx;
    const size_t n_links_t = (size_t)n_links;
    const size_t n_tx_ports_t = (size_t)n_tx_ports;
    const size_t n_rx_ports_t = (size_t)n_rx_ports;
    const size_t n_ports_t = (size_t)n_ports;

    // Integer types
    const int n_out_i = (int)n_out;

    if (lbs_pos->n_cols != n_path || path_gain->n_elem != n_path || path_length->n_elem != n_path || M->n_cols != n_path)
    {
        error_message = "Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length', and 'M' must have the same number of columns (n_paths).";
        throw std::invalid_argument(error_message.c_str());
    }

    // Set the output size
    if (coeff_re->n_rows != n_rx_ports || coeff_re->n_cols != n_tx_ports || coeff_re->n_slices != n_out)
        coeff_re->set_size(n_rx_ports, n_tx_ports, n_out);
    if (coeff_im->n_rows != n_rx_ports || coeff_im->n_cols != n_tx_ports || coeff_im->n_slices != n_out)
        coeff_im->set_size(n_rx_ports, n_tx_ports, n_out);
    if (delay->n_rows != n_rx_ports || delay->n_cols != n_tx_ports || delay->n_slices != n_out)
        delay->set_size(n_rx_ports, n_tx_ports, n_out);
    if (aod != nullptr && (aod->n_rows != n_rx_ports || aod->n_cols != n_tx_ports || aod->n_slices != n_out))
        aod->set_size(n_rx_ports, n_tx_ports, n_out);
    if (eod != nullptr && (eod->n_rows != n_rx_ports || eod->n_cols != n_tx_ports || eod->n_slices != n_out))
        eod->set_size(n_rx_ports, n_tx_ports, n_out);
    if (aoa != nullptr && (aoa->n_rows != n_rx_ports || aoa->n_cols != n_tx_ports || aoa->n_slices != n_out))
        aoa->set_size(n_rx_ports, n_tx_ports, n_out);
    if (eoa != nullptr && (eoa->n_rows != n_rx_ports || eoa->n_cols != n_tx_ports || eoa->n_slices != n_out))
        eoa->set_size(n_rx_ports, n_tx_ports, n_out);

    // Map output memory to internal representation
    arma::Cube<dtype> CR, CI, DL;        // Internal map for delays and coefficients
    arma::Mat<dtype> AOD, EOD, AOA, EOA; // Antenna interpolation requires matrix of size [n_out, n_ang]
    bool different_output_size = n_tx != n_tx_ports || n_rx != n_rx_ports;

    if (different_output_size) // Temporary internal data storage
    {
        CR.set_size(n_rx, n_tx, n_out);
        CI.set_size(n_rx, n_tx, n_out);
        DL.set_size(n_rx, n_tx, n_out);
        AOD.set_size(n_links, n_out);
        EOD.set_size(n_links, n_out);
        AOA.set_size(n_links, n_out);
        EOA.set_size(n_links, n_out);
    }
    else // Direct mapping of external memory
    {
        CR = arma::Cube<dtype>(coeff_re->memptr(), n_rx, n_tx, n_out, false, true);
        CI = arma::Cube<dtype>(coeff_im->memptr(), n_rx, n_tx, n_out, false, true);
        DL = arma::Cube<dtype>(delay->memptr(), n_rx, n_tx, n_out, false, true);

        if (aod == nullptr)
            AOD.set_size(n_links, n_out);
        else
            AOD = arma::Mat<dtype>(aod->memptr(), n_links, n_out, false, true);

        if (eod == nullptr)
            EOD.set_size(n_links, n_out);
        else
            EOD = arma::Mat<dtype>(eod->memptr(), n_links, n_out, false, true);

        if (aoa == nullptr)
            AOA.set_size(n_links, n_out);
        else
            AOA = arma::Mat<dtype>(aoa->memptr(), n_links, n_out, false, true);

        if (eoa == nullptr)
            EOA.set_size(n_links, n_out);
        else
            EOA = arma::Mat<dtype>(eoa->memptr(), n_links, n_out, false, true);
    }

    // Get pointers
    const dtype *p_fbs = fbs_pos->memptr();
    const dtype *p_lbs = lbs_pos->memptr();
    const dtype *p_gain = path_gain->memptr();
    const dtype *p_length = path_length->memptr();

    dtype *p_coeff_re = CR.memptr();
    dtype *p_coeff_im = CI.memptr();
    dtype *p_delays = DL.memptr();
    dtype *p_aod = AOD.memptr();
    dtype *p_eod = EOD.memptr();
    dtype *p_aoa = AOA.memptr();
    dtype *p_eoa = EOA.memptr();
    dtype *ptr;

    // Convert inputs to orientation vector
    arma::Cube<dtype> tx_orientation(3, 1, 1);
    arma::Cube<dtype> rx_orientation(3, 1, 1);
    bool tx_orientation_not_zero = (Tb != zero || Tt != zero || Th != zero);
    bool rx_orientation_not_zero = (Rb != zero || Rt != zero || Rh != zero);

    if (tx_orientation_not_zero)
        ptr = tx_orientation.memptr(), ptr[0] = Tb, ptr[1] = Tt, ptr[2] = Th;

    if (rx_orientation_not_zero)
        ptr = rx_orientation.memptr(), ptr[0] = Rb, ptr[1] = Rt, ptr[2] = Rh;

    // Calculate the antenna element positions in GCS
    arma::Mat<dtype> tx_element_pos(3, n_tx), rx_element_pos(3, n_rx);
    dtype *p_tx = tx_element_pos.memptr(), *p_rx = rx_element_pos.memptr();

    if (tx_array->element_pos.n_elem != 0)
        std::memcpy(p_tx, tx_array->element_pos.memptr(), 3 * n_tx * sizeof(dtype));
    if (tx_orientation_not_zero) // Apply TX antenna orientation
        quick_rotate_inplace(Tb, -Tt, Th, p_tx, n_tx);
    for (size_t t = 0; t < 3 * n_tx_t; t += 3) // Add TX position
        p_tx[t] += Tx, p_tx[t + 1] += Ty, p_tx[t + 2] += Tz;

    if (rx_array->element_pos.n_elem != 0)
        std::memcpy(p_rx, rx_array->element_pos.memptr(), 3 * n_rx * sizeof(dtype));
    if (rx_orientation_not_zero) // Apply RX antenna orientation
        quick_rotate_inplace(Rb, -Rt, Rh, p_rx, n_rx);
    for (size_t r = 0; r < 3 * n_rx_t; r += 3) // Add RX position
        p_rx[r] += Rx, p_rx[r + 1] += Ry, p_rx[r + 2] += Rz;

    // Calculate the Freespace distance
    dtype x = Rx - Tx, y = Ry - Ty, z = Rz - Tz;
    dtype dist_rx_tx = std::sqrt(x * x + y * y + z * z);

    // There may be multiple LOS paths. We need to find the real one
    // Detection is done by sing the shortest length difference to the TX-RX line
    arma::uword true_los_path = 0;
    dtype shortest_path = los_limit;

    // Calculate angles and delays
    // Cannot be parallelized due to "true_los_path" and "shortest_path"
    for (int i_out = 0; i_out < n_out_i; ++i_out) // Loop over paths
    {
        const size_t j = (size_t)i_out;
        const size_t i = add_fake_los_path ? j - 1 : j;
        const size_t ix = 3 * i, iy = ix + 1, iz = ix + 2;
        const size_t o = j * n_links_t; // Slice offset

        // Calculate the shortest possible path length (TX > FBS > LBS > RX)
        dtype d_shortest = dist_rx_tx, d_length = dist_rx_tx, d_fbs_lbs = zero;
        if (!add_fake_los_path || j != 0)
        {
            x = p_fbs[ix] - Tx, y = p_fbs[iy] - Ty, z = p_fbs[iz] - Tz;
            d_shortest = std::sqrt(x * x + y * y + z * z);
            x = p_lbs[ix] - p_fbs[ix], y = p_lbs[iy] - p_fbs[iy], z = p_lbs[iz] - p_fbs[iz];
            d_fbs_lbs = std::sqrt(x * x + y * y + z * z);
            d_shortest += d_fbs_lbs;
            x = Rx - p_lbs[ix], y = Ry - p_lbs[iy], z = Rz - p_lbs[iz];
            d_shortest += std::sqrt(x * x + y * y + z * z);
            d_length = (p_length[i] < d_shortest) ? d_shortest : p_length[i];
        }

        // Calculate path delays, departure angles and arrival angles
        // size_t o = j * n_links, o0 = o;
        if (std::abs(d_length - dist_rx_tx) < shortest_path) // LOS path
        {
            if (!add_fake_los_path || j != 0)
                true_los_path = j, shortest_path = std::abs(d_length - dist_rx_tx);

            for (size_t t = 0; t < n_tx_t; ++t)
                for (size_t r = 0; r < n_rx_t; ++r)
                {
                    x = p_rx[3 * r] - p_tx[3 * t];
                    y = p_rx[3 * r + 1] - p_tx[3 * t + 1];
                    z = p_rx[3 * r + 2] - p_tx[3 * t + 2];
                    dtype d = std::sqrt(x * x + y * y + z * z);

                    size_t io = o + t * n_rx_t + r;
                    p_aod[io] = std::atan2(y, x);
                    p_eod[io] = (d < los_limit) ? zero : std::asin(z / d);
                    p_aoa[io] = std::atan2(-y, -x);
                    p_eoa[io] = -p_eod[io];
                    p_delays[io] = d;
                }
        }
        else // NLOS path
        {
            dtype *dr = new dtype[n_rx];
            for (size_t r = 0; r < n_rx_t; ++r)
                x = p_lbs[ix] - p_rx[3 * r],
                y = p_lbs[iy] - p_rx[3 * r + 1],
                z = p_lbs[iz] - p_rx[3 * r + 2],
                dr[r] = std::sqrt(x * x + y * y + z * z),
                p_aoa[o + r] = std::atan2(y, x),
                p_eoa[o + r] = (dr[r] < los_limit) ? zero : std::asin(z / dr[r]);

            for (size_t t = 0; t < n_tx_t; ++t)
            {
                x = p_fbs[ix] - p_tx[3 * t],
                y = p_fbs[iy] - p_tx[3 * t + 1],
                z = p_fbs[iz] - p_tx[3 * t + 2];

                dtype dt = std::sqrt(x * x + y * y + z * z),
                      at = std::atan2(y, x),
                      et = (dt < los_limit) ? zero : std::asin(z / dt);

                for (size_t r = 0; r < n_rx_t; ++r)
                {
                    size_t io = o + t * n_rx_t + r;
                    p_aod[io] = at, p_eod[io] = et;
                    p_aoa[io] = p_aoa[o + r], p_eoa[io] = p_eoa[o + r];
                    p_delays[io] = dt + d_fbs_lbs + dr[r];
                }
            }
            delete[] dr;
        }
    }

    // Calculate TX element indices and positions
    arma::Col<unsigned> i_tx_element(n_links, arma::fill::none);
    unsigned *p_element = i_tx_element.memptr();
    arma::Mat<dtype> tx_element_pos_interp(3, n_links, arma::fill::none);
    ptr = tx_element_pos_interp.memptr();
    if (tx_array->element_pos.n_elem != 0)
        std::memcpy(p_tx, tx_array->element_pos.memptr(), 3 * n_tx * sizeof(dtype));
    else
        tx_element_pos.zeros();
    for (unsigned t = 0; t < (unsigned)n_tx; ++t)
        for (unsigned r = 0; r < (unsigned)n_rx; ++r)
            *p_element++ = t + 1, *ptr++ = p_tx[3 * t], *ptr++ = p_tx[3 * t + 1], *ptr++ = p_tx[3 * t + 2];

    // Calculate RX element indices and positions
    arma::Col<unsigned> i_rx_element(n_links, arma::fill::none);
    p_element = i_rx_element.memptr();
    arma::Mat<dtype> rx_element_pos_interp(3, n_links, arma::fill::none);
    ptr = rx_element_pos_interp.memptr();
    if (rx_array->element_pos.n_elem != 0)
        std::memcpy(p_rx, rx_array->element_pos.memptr(), 3 * n_rx * sizeof(dtype));
    else
        rx_element_pos.zeros();
    for (unsigned t = 0; t < unsigned(n_tx); ++t)
        for (unsigned r = 0; r < unsigned(n_rx); ++r)
            *p_element++ = r + 1, *ptr++ = p_rx[3 * r], *ptr++ = p_rx[3 * r + 1], *ptr++ = p_rx[3 * r + 2];

    // Determine if we need to apply antenna-element coupling
    bool apply_element_coupling = false;
    if (rx_array->coupling_re.n_elem != 0 || rx_array->coupling_im.n_elem != 0 ||
        tx_array->coupling_re.n_elem != 0 || tx_array->coupling_im.n_elem != 0)
    {
        if (n_rx_ports != n_rx || n_tx_ports != n_tx)
            apply_element_coupling = true;

        // Check if TX coupling real part is identity matrix
        if (!apply_element_coupling && tx_array->coupling_re.n_rows == n_tx && tx_array->coupling_re.n_cols == n_tx)
        {
            const dtype *p = tx_array->coupling_re.memptr();
            for (size_t r = 0; r < n_tx_t; ++r)
                for (size_t c = 0; c < n_tx_t; ++c)
                {
                    if (r == c && std::abs(p[c * n_tx + r] - (dtype)1.0) > (dtype)1.0e-6)
                        apply_element_coupling = true;
                    else if (r != c && std::abs(p[c * n_tx + r]) > (dtype)1.0e-6)
                        apply_element_coupling = true;
                }
        }

        // If TX coupling has imaginary part, check if there is any element not 0
        if (!apply_element_coupling && tx_array->coupling_im.n_elem != 0)
        {
            for (const dtype *p = tx_array->coupling_im.begin(); p < tx_array->coupling_im.end(); ++p)
                if (std::abs(*p) > (dtype)1.0e-6)
                    apply_element_coupling = true;
        }

        // Check if RX coupling real part is identity matrix
        if (!apply_element_coupling && rx_array->coupling_re.n_rows == n_rx && rx_array->coupling_re.n_cols == n_rx)
        {
            const dtype *p = rx_array->coupling_re.memptr();
            for (size_t r = 0; r < n_rx_t; ++r)
                for (size_t c = 0; c < n_rx_t; ++c)
                {
                    if (r == c && std::abs(p[c * n_rx + r] - (dtype)1.0) > (dtype)1.0e-6)
                        apply_element_coupling = true;
                    else if (r != c && std::abs(p[c * n_rx + r]) > (dtype)1.0e-6)
                        apply_element_coupling = true;
                }
        }

        // If RX coupling has imaginary part, check if there is any element not 0
        if (!apply_element_coupling && rx_array->coupling_im.n_elem != 0)
        {
            for (const dtype *p = rx_array->coupling_im.begin(); p < rx_array->coupling_im.end(); ++p)
                if (std::abs(*p) > (dtype)1.0e-6)
                    apply_element_coupling = true;
        }
    }

    // Calculate abs( cpl )^2 and normalize the row-sum to 1
    arma::Mat<dtype> tx_coupling_pwr(n_tx, n_tx_ports, arma::fill::none);
    arma::Mat<dtype> rx_coupling_pwr(n_rx, n_rx_ports, arma::fill::none);
    if (apply_element_coupling)
    {
        quick_power_mat(n_tx, n_tx_ports, tx_coupling_pwr.memptr(), true, tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr());
        quick_power_mat(n_rx, n_rx_ports, rx_coupling_pwr.memptr(), true, rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr());
    }

    // Calculate the MIMO channel coefficients for each path
#pragma omp parallel for
    for (int i_out = 0; i_out < n_out_i; ++i_out) // Loop over paths
    {
        const size_t j = (size_t)i_out;
        const size_t i = add_fake_los_path ? (j == 0 ? 0 : j - 1) : j;
        const size_t O = j * n_links_t; // Slice offset

        // Allocate memory for temporary data
        arma::Mat<dtype> Vt_re(n_links, 1, arma::fill::none), Vt_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> Ht_re(n_links, 1, arma::fill::none), Ht_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> Vr_re(n_links, 1, arma::fill::none), Vr_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> Hr_re(n_links, 1, arma::fill::none), Hr_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> EMPTY;

        // Calculate TX antenna response
        const arma::Mat<dtype> AOD_j(AOD.colptr(j), n_links, 1, false, true);
        const arma::Mat<dtype> EOD_j(EOD.colptr(j), n_links, 1, false, true);
        qd_arrayant_interpolate(&tx_array->e_theta_re, &tx_array->e_theta_im, &tx_array->e_phi_re, &tx_array->e_phi_im,
                                &tx_array->azimuth_grid, &tx_array->elevation_grid, &AOD_j, &EOD_j,
                                &i_tx_element, &tx_orientation, &tx_element_pos_interp,
                                &Vt_re, &Vt_im, &Ht_re, &Ht_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

        // Calculate RX antenna response
        const arma::Mat<dtype> AOA_j(AOA.colptr(j), n_links, 1, false, true);
        const arma::Mat<dtype> EOA_j(EOA.colptr(j), n_links, 1, false, true);
        qd_arrayant_interpolate(&rx_array->e_theta_re, &rx_array->e_theta_im, &rx_array->e_phi_re, &rx_array->e_phi_im,
                                &rx_array->azimuth_grid, &rx_array->elevation_grid, &AOA_j, &EOA_j,
                                &i_rx_element, &rx_orientation, &rx_element_pos_interp,
                                &Vr_re, &Vr_im, &Hr_re, &Hr_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

        // Calculate the MIMO channel coefficients
        const dtype *pM = M->colptr(i);
        const dtype *pVrr = Vr_re.memptr(), *pVri = Vr_im.memptr(),
                    *pHrr = Hr_re.memptr(), *pHri = Hr_im.memptr(),
                    *pVtr = Vt_re.memptr(), *pVti = Vt_im.memptr(),
                    *pHtr = Ht_re.memptr(), *pHti = Ht_im.memptr();
        const dtype path_amplitude = (add_fake_los_path && j == 0) ? zero : std::sqrt(p_gain[i]);

        for (size_t t = 0; t < n_tx_t; ++t)
            for (size_t r = 0; r < n_rx_t; ++r)
            {
                size_t R = t * n_rx_t + r;

                dtype re = zero, im = zero;
                re += pVrr[R] * pM[0] * pVtr[R] - pVri[R] * pM[1] * pVtr[R] - pVrr[R] * pM[1] * pVti[R] - pVri[R] * pM[0] * pVti[R];
                re += pHrr[R] * pM[2] * pVtr[R] - pHri[R] * pM[3] * pVtr[R] - pHrr[R] * pM[3] * pVti[R] - pHri[R] * pM[2] * pVti[R];
                re += pVrr[R] * pM[4] * pHtr[R] - pVri[R] * pM[5] * pHtr[R] - pVrr[R] * pM[5] * pHti[R] - pVri[R] * pM[4] * pHti[R];
                re += pHrr[R] * pM[6] * pHtr[R] - pHri[R] * pM[7] * pHtr[R] - pHrr[R] * pM[7] * pHti[R] - pHri[R] * pM[6] * pHti[R];

                im += pVrr[R] * pM[1] * pVtr[R] + pVri[R] * pM[0] * pVtr[R] + pVrr[R] * pM[0] * pVti[R] - pVri[R] * pM[1] * pVti[R];
                im += pHrr[R] * pM[3] * pVtr[R] + pHri[R] * pM[2] * pVtr[R] + pHrr[R] * pM[2] * pVti[R] - pHri[R] * pM[3] * pVti[R];
                im += pVrr[R] * pM[5] * pHtr[R] + pVri[R] * pM[4] * pHtr[R] + pVrr[R] * pM[4] * pHti[R] - pVri[R] * pM[5] * pHti[R];
                im += pHrr[R] * pM[7] * pHtr[R] + pHri[R] * pM[6] * pHtr[R] + pHrr[R] * pM[6] * pHti[R] - pHri[R] * pM[7] * pHti[R];

                dtype dl = p_delays[O + R]; // path length from previous calculation
                dtype phase = wave_number * std::fmod(dl, wavelength);
                dtype cp = std::cos(phase), sp = std::sin(phase);

                p_coeff_re[O + R] = (re * cp + im * sp) * path_amplitude;
                p_coeff_im[O + R] = (-re * sp + im * cp) * path_amplitude;

                dl = use_absolute_delays ? dl : dl - dist_rx_tx;
                p_delays[O + R] = dl * rC;
            }

        // Apply antenna element coupling
        if (apply_element_coupling)
        {
            // Allocate memory for temporary data
            size_t N = (n_ports_t > n_links_t) ? n_ports_t : n_links_t;
            dtype *tempX = new dtype[N];
            dtype *tempY = new dtype[N];
            dtype *tempZ = new dtype[N];
            dtype *tempT = new dtype[N];

            // Process coefficients and delays
            if (different_output_size) // Data is stored in internal memory, we can write directly to the output
            {
                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[O], &p_coeff_im[O],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             coeff_re->slice_memptr(j), coeff_im->slice_memptr(j),
                                             n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);

                // Apply coupling to delays
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), &p_delays[O], tx_coupling_pwr.memptr(), delay->slice_memptr(j),
                                     n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
            }
            else // Data has been written to external memory
            {
                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[O], &p_coeff_im[O],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             tempX, tempY,
                                             n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);

                std::memcpy(&p_coeff_re[O], tempX, n_ports_t * sizeof(dtype));
                std::memcpy(&p_coeff_im[O], tempY, n_ports_t * sizeof(dtype));

                // Apply coupling to delays
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), &p_delays[O], tx_coupling_pwr.memptr(), tempT, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                std::memcpy(&p_delays[O], tempT, n_ports_t * sizeof(dtype));
            }

            // Process departure angles
            if (aod != nullptr || eod != nullptr)
            {
                // Convert AOD and EOD to Cartesian coordinates
                quick_geo2cart(n_links_t, &p_aod[O], tempX, tempY, &p_eod[O], tempZ);

                // Apply coupling to the x, y, z, component independently
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempX, tx_coupling_pwr.memptr(), tempT, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempY, tx_coupling_pwr.memptr(), tempX, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempZ, tx_coupling_pwr.memptr(), tempY, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);

                // Convert back to geographic coordinates and save to external memory
                if (!different_output_size) // External memory is mapped to "p_aod" and "p_eod"
                    quick_cart2geo(n_ports_t, &p_aod[O], tempT, tempX, &p_eod[O], tempY);
                else if (aod == nullptr)
                    quick_cart2geo<dtype>(n_ports_t, NULL, tempT, tempX, eod->slice_memptr(j), tempY);
                else if (eod == nullptr)
                    quick_cart2geo<dtype>(n_ports_t, aod->slice_memptr(j), tempT, tempX, NULL, tempY);
                else
                    quick_cart2geo<dtype>(n_ports_t, aod->slice_memptr(j), tempT, tempX, eod->slice_memptr(j), tempY);
            }

            // Process arrival angles
            if (aoa != nullptr || eoa != nullptr)
            {
                // Convert AOD and EOD to Cartesian coordinates
                quick_geo2cart(n_links_t, &p_aoa[O], tempX, tempY, &p_eoa[O], tempZ);

                // Apply coupling to the x, y, z, component independently
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempX, tx_coupling_pwr.memptr(), tempT, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempY, tx_coupling_pwr.memptr(), tempX, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);
                quick_multiply_3_mat(rx_coupling_pwr.memptr(), tempZ, tx_coupling_pwr.memptr(), tempY, n_rx_t, n_rx_ports_t, n_tx_t, n_tx_ports_t);

                // Convert back to geographic coordinates and save to external memory
                if (!different_output_size) // External memory is mapped to "p_aoa" and "p_eoa"
                    quick_cart2geo(n_ports_t, &p_aoa[O], tempT, tempX, &p_eoa[O], tempY);
                else if (aoa == nullptr)
                    quick_cart2geo<dtype>(n_ports_t, NULL, tempT, tempX, eoa->slice_memptr(j), tempY);
                else if (eoa == nullptr)
                    quick_cart2geo<dtype>(n_ports_t, aoa->slice_memptr(j), tempT, tempX, NULL, tempY);
                else
                    quick_cart2geo<dtype>(n_ports_t, aoa->slice_memptr(j), tempT, tempX, eoa->slice_memptr(j), tempY);
            }

            // Free temporary memory
            delete[] tempX;
            delete[] tempY;
            delete[] tempZ;
            delete[] tempT;
        }
    }

    // Set the true LOS path as the first path
    if (add_fake_los_path && true_los_path != 0)
    {
        std::memcpy(p_coeff_re, CR.slice_memptr(true_los_path), n_links_t * sizeof(dtype));
        std::memcpy(p_coeff_im, CI.slice_memptr(true_los_path), n_links_t * sizeof(dtype));
        CR.slice(true_los_path).zeros();
        CI.slice(true_los_path).zeros();
    }
}

template void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<float> *tx_array, const quadriga_lib::arrayant<float> *rx_array,
                                                   float Tx, float Ty, float Tz, float Tb, float Tt, float Th,
                                                   float Rx, float Ry, float Rz, float Rb, float Rt, float Rh,
                                                   const arma::Mat<float> *fbs_pos, const arma::Mat<float> *lbs_pos,
                                                   const arma::Col<float> *path_gain, const arma::Col<float> *path_length, const arma::Mat<float> *M,
                                                   arma::Cube<float> *coeff_re, arma::Cube<float> *coeff_im, arma::Cube<float> *delay,
                                                   float center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                                   arma::Cube<float> *aod, arma::Cube<float> *eod, arma::Cube<float> *aoa, arma::Cube<float> *eoa);

template void quadriga_lib::get_channels_spherical(const quadriga_lib::arrayant<double> *tx_array, const quadriga_lib::arrayant<double> *rx_array,
                                                   double Tx, double Ty, double Tz, double Tb, double Tt, double Th,
                                                   double Rx, double Ry, double Rz, double Rb, double Rt, double Rh,
                                                   const arma::Mat<double> *fbs_pos, const arma::Mat<double> *lbs_pos,
                                                   const arma::Col<double> *path_gain, const arma::Col<double> *path_length, const arma::Mat<double> *M,
                                                   arma::Cube<double> *coeff_re, arma::Cube<double> *coeff_im, arma::Cube<double> *delay,
                                                   double center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                                   arma::Cube<double> *aod, arma::Cube<double> *eod, arma::Cube<double> *aoa, arma::Cube<double> *eoa);
