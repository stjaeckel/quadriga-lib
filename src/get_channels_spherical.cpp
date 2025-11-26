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
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# get_channels_spherical
Calculate channel coefficients for spherical waves

## Description:
- Calculates MIMO channel coefficients and delays for a set of spherical wave paths between two antenna arrays.
- Interpolates antenna patterns (including orientation and polarization) for both transmitter and receiver arrays.
- Accurately models path-based propagation using provided scatterer positions.
- Supports LOS path identification and handles complex polarization coupling.
- Element positions and antenna orientation are fully considered for delay and phase.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::get_channels_spherical(
                const arrayant<dtype> *tx_array,
                const arrayant<dtype> *rx_array,
                dtype Tx, dtype Ty, dtype Tz,
                dtype Tb, dtype Tt, dtype Th,
                dtype Rx, dtype Ry, dtype Rz,
                dtype Rb, dtype Rt, dtype Rh,
                const arma::Mat<dtype> *fbs_pos,
                const arma::Mat<dtype> *lbs_pos,
                const arma::Col<dtype> *path_gain,
                const arma::Col<dtype> *path_length,
                const arma::Mat<dtype> *M,
                arma::Cube<dtype> *coeff_re,
                arma::Cube<dtype> *coeff_im,
                arma::Cube<dtype> *delay,
                dtype center_frequency = dtype(0.0),
                bool use_absolute_delays = false,
                bool add_fake_los_path = false,
                arma::Cube<dtype> *aod = nullptr,
                arma::Cube<dtype> *eod = nullptr,
                arma::Cube<dtype> *aoa = nullptr,
                arma::Cube<dtype> *eoa = nullptr)
```

## Arguments:
- `const arrayant<dtype> ***tx_array**` (input)<br>
  Pointer to the transmit antenna array object (with `n_tx` elements).

- `const arrayant<dtype> ***rx_array**` (input)<br>
  Pointer to the receive antenna array object (with `n_rx` elements).

- `dtype **Tx**, **Ty**, **Tz**` (input)<br>
  Transmitter position in Cartesian coordinates [m].

- `dtype **Tb**, **Tt**, **Th**` (input)<br>
  Transmitter orientation (Euler) angles (bank, tilt, head) in [rad].

- `dtype **Rx**, **Ry**, **Rz**` (input)<br>
  Receiver position in Cartesian coordinates [m].

- `dtype **Rb**, **Rt**, **Rh**` (input)<br>
  Receiver orientation (Euler) angles (bank, tilt, head) in [rad].

- `const arma::Mat<dtype> ***fbs_pos**` (input)<br>
  First-bounce scatterer positions, Size: `[3, n_path]`.

- `const arma::Mat<dtype> ***lbs_pos**` (input)<br>
  Last-bounce scatterer positions, Size: `[3, n_path]`.

- `const arma::Col<dtype> ***path_gain**` (input)<br>
  Path gains in linear scale, Length `n_path`.

- `const arma::Col<dtype> ***path_length**` (input)<br>
  Path lengths from TX to RX phase center Length `n_path`.

- `const arma::Mat<dtype> ***M**` (input)<br>
  Polarization transfer matrix of size `[8, n_path]`, interleaved: (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH).

- `arma::Cube<dtype> ***coeff_re**` (output)<br>
  Real part of channel coefficients, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***coeff_im**` (output)<br>
  Imaginary part of channel coefficients, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***delay**` (output)<br>
  Propagation delays in seconds, Size `[n_rx, n_tx, n_path]`.

- `dtype **center_frequency** = 0.0` (optional input)<br>
  Center frequency in Hz; set to 0 to disable phase calculation. Default: `0.0`

- `bool **use_absolute_delays** = false` (optional input)<br>
  If true, includes LOS delay in all paths. Default: `false`

- `bool **add_fake_los_path** = false` (optional input)<br>
  Adds a zero-power LOS path if no LOS is present. Default: `false`

- `arma::Cube<dtype> ***aod** = nullptr` (optional output)<br>
  Azimuth of Departure angles in radians, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***eod** = nullptr` (optional output)<br>
  Elevation of Departure angles in radians, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***aoa** = nullptr` (optional output)<br>
  Azimuth of Arrival angles in radians, Size `[n_rx, n_tx, n_path]`.

- `arma::Cube<dtype> ***eoa = nullptr` (optional output)<br>
  Elevation of Arrival angles in radians, Size `[n_rx, n_tx, n_path]`.
MD!*/

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
    }
    if (different_output_size || aod == nullptr)
        AOD.set_size(n_links, n_out);
    if (different_output_size || eod == nullptr)
        EOD.set_size(n_links, n_out);
    if (different_output_size || aoa == nullptr)
        AOA.set_size(n_links, n_out);
    if (different_output_size || eoa == nullptr)
        EOA.set_size(n_links, n_out);

    dtype *p_coeff_re = different_output_size ? CR.memptr() : coeff_re->memptr();
    dtype *p_coeff_im = different_output_size ? CI.memptr() : coeff_im->memptr();
    dtype *p_delays = different_output_size ? DL.memptr() : delay->memptr();
    dtype *p_aod = (different_output_size || aod == nullptr) ? AOD.memptr() : aod->memptr();
    dtype *p_eod = (different_output_size || eod == nullptr) ? EOD.memptr() : eod->memptr();
    dtype *p_aoa = (different_output_size || aoa == nullptr) ? AOA.memptr() : aoa->memptr();
    dtype *p_eoa = (different_output_size || eoa == nullptr) ? EOA.memptr() : eoa->memptr();

    // Get pointers
    const dtype *p_fbs = fbs_pos->memptr();
    const dtype *p_lbs = lbs_pos->memptr();
    const dtype *p_gain = path_gain->memptr();
    const dtype *p_length = path_length->memptr();

    // Convert inputs to orientation vector
    dtype *ptr;
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
        qd_rotate_inplace(Tb, -Tt, Th, p_tx, n_tx);
    for (arma::uword t = 0ULL; t < 3ULL * n_tx; t += 3ULL) // Add TX position
        p_tx[t] += Tx, p_tx[t + 1ULL] += Ty, p_tx[t + 2ULL] += Tz;

    if (rx_array->element_pos.n_elem != 0)
        std::memcpy(p_rx, rx_array->element_pos.memptr(), 3 * n_rx * sizeof(dtype));
    if (rx_orientation_not_zero) // Apply RX antenna orientation
        qd_rotate_inplace(Rb, -Rt, Rh, p_rx, n_rx);
    for (arma::uword r = 0ULL; r < 3ULL * n_rx; r += 3ULL) // Add RX position
        p_rx[r] += Rx, p_rx[r + 1ULL] += Ry, p_rx[r + 2ULL] += Rz;

    // Calculate the Freespace distance
    dtype x = Rx - Tx, y = Ry - Ty, z = Rz - Tz;
    dtype dist_rx_tx = std::sqrt(x * x + y * y + z * z);

    // There may be multiple LOS paths. We need to find the real one
    // Detection is done by sing the shortest length difference to the TX-RX line
    arma::uword true_los_path = 0;
    dtype shortest_path = los_limit;

    // Calculate angles and delays
    // Cannot be parallelized due to "true_los_path" and "shortest_path"
    for (arma::uword i_out = 0ULL; i_out < n_out; ++i_out) // Loop over paths
    {
        const arma::uword i = add_fake_los_path ? i_out - 1ULL : i_out;
        const arma::uword ix = 3ULL * i, iy = ix + 1ULL, iz = ix + 2ULL;
        const arma::uword o_slice = i_out * n_links; // Slice offset

        // Calculate the shortest possible path length (TX > FBS > LBS > RX)
        dtype d_shortest = dist_rx_tx, d_length = dist_rx_tx, d_fbs_lbs = zero;
        if (!add_fake_los_path || i_out != 0ULL)
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
        if (std::abs(d_length - dist_rx_tx) < shortest_path) // LOS path
        {
            if (!add_fake_los_path || i_out != 0ULL)
                true_los_path = i_out, shortest_path = std::abs(d_length - dist_rx_tx);

            for (arma::uword t = 0ULL; t < n_tx; ++t)
            {
                dtype *pt = &p_tx[3ULL * t];
                for (arma::uword r = 0ULL; r < n_rx; ++r)
                {
                    dtype *pr = &p_rx[3ULL * r];
                    x = pr[0] - pt[0];
                    y = pr[1] - pt[1];
                    z = pr[2] - pt[2];
                    dtype d = std::sqrt(x * x + y * y + z * z);

                    arma::uword io = o_slice + t * n_rx + r;
                    p_aod[io] = std::atan2(y, x);
                    p_eod[io] = (d < los_limit) ? zero : std::asin(z / d);
                    p_aoa[io] = std::atan2(-y, -x);
                    p_eoa[io] = -p_eod[io];
                    p_delays[io] = d;
                }
            }
        }
        else // NLOS path
        {
            dtype *dr = new dtype[n_rx];
            for (arma::uword r = 0ULL; r < n_rx; ++r)
            {
                dtype *pr = &p_rx[3ULL * r];
                x = p_lbs[ix] - pr[0];
                y = p_lbs[iy] - pr[1];
                z = p_lbs[iz] - pr[2];
                dr[r] = std::sqrt(x * x + y * y + z * z);
                p_aoa[o_slice + r] = std::atan2(y, x);
                p_eoa[o_slice + r] = (dr[r] < los_limit) ? zero : std::asin(z / dr[r]);
            }

            for (arma::uword t = 0ULL; t < n_tx; ++t)
            {
                dtype *pt = &p_tx[3ULL * t];
                x = p_fbs[ix] - pt[0],
                y = p_fbs[iy] - pt[1],
                z = p_fbs[iz] - pt[2];

                dtype dt = std::sqrt(x * x + y * y + z * z),
                      at = std::atan2(y, x),
                      et = (dt < los_limit) ? zero : std::asin(z / dt);

                for (arma::uword r = 0ULL; r < n_rx; ++r)
                {
                    arma::uword io = o_slice + t * n_rx + r;
                    p_aod[io] = at;
                    p_eod[io] = et;
                    p_aoa[io] = p_aoa[o_slice + r];
                    p_eoa[io] = p_eoa[o_slice + r];
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
            for (arma::uword r = 0ULL; r < n_tx; ++r)
                for (arma::uword c = 0ULL; c < n_tx; ++c)
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
            for (arma::uword r = 0ULL; r < n_rx; ++r)
                for (arma::uword c = 0ULL; c < n_rx; ++c)
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
        qd_power_mat(n_tx, n_tx_ports, tx_coupling_pwr.memptr(), true, tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr());
        qd_power_mat(n_rx, n_rx_ports, rx_coupling_pwr.memptr(), true, rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr());
    }

    // Calculate the MIMO channel coefficients for each path
    const int n_out_i = (int)n_out;
#pragma omp parallel for
    for (int i_out = 0; i_out < n_out_i; ++i_out) // Loop over paths
    {
        const arma::uword j = (arma::uword)i_out;
        const arma::uword i = add_fake_los_path ? (j == 0 ? 0 : j - 1) : j;
        const arma::uword o_slice = j * n_links; // Slice offset

        // Allocate memory for temporary data
        arma::Mat<dtype> Vt_re(n_links, 1, arma::fill::none), Vt_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> Ht_re(n_links, 1, arma::fill::none), Ht_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> Vr_re(n_links, 1, arma::fill::none), Vr_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> Hr_re(n_links, 1, arma::fill::none), Hr_im(n_links, 1, arma::fill::none);
        arma::Mat<dtype> EMPTY;

        // Calculate TX antenna response
        const arma::Mat<dtype> AOD_j(&p_aod[o_slice], n_links, 1, false, true);
        const arma::Mat<dtype> EOD_j(&p_eod[o_slice], n_links, 1, false, true);
        qd_arrayant_interpolate(&tx_array->e_theta_re, &tx_array->e_theta_im, &tx_array->e_phi_re, &tx_array->e_phi_im,
                                &tx_array->azimuth_grid, &tx_array->elevation_grid, &AOD_j, &EOD_j,
                                &i_tx_element, &tx_orientation, &tx_element_pos_interp,
                                &Vt_re, &Vt_im, &Ht_re, &Ht_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

        // Calculate RX antenna response
        const arma::Mat<dtype> AOA_j(&p_aoa[o_slice], n_links, 1, false, true);
        const arma::Mat<dtype> EOA_j(&p_eoa[o_slice], n_links, 1, false, true);
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

        for (arma::uword t = 0ULL; t < n_tx; ++t)
            for (arma::uword r = 0ULL; r < n_rx; ++r)
            {
                arma::uword R = t * n_rx + r;

                dtype re = zero, im = zero;
                re += pVrr[R] * pM[0] * pVtr[R] - pVri[R] * pM[1] * pVtr[R] - pVrr[R] * pM[1] * pVti[R] - pVri[R] * pM[0] * pVti[R];
                re += pHrr[R] * pM[2] * pVtr[R] - pHri[R] * pM[3] * pVtr[R] - pHrr[R] * pM[3] * pVti[R] - pHri[R] * pM[2] * pVti[R];
                re += pVrr[R] * pM[4] * pHtr[R] - pVri[R] * pM[5] * pHtr[R] - pVrr[R] * pM[5] * pHti[R] - pVri[R] * pM[4] * pHti[R];
                re += pHrr[R] * pM[6] * pHtr[R] - pHri[R] * pM[7] * pHtr[R] - pHrr[R] * pM[7] * pHti[R] - pHri[R] * pM[6] * pHti[R];

                im += pVrr[R] * pM[1] * pVtr[R] + pVri[R] * pM[0] * pVtr[R] + pVrr[R] * pM[0] * pVti[R] - pVri[R] * pM[1] * pVti[R];
                im += pHrr[R] * pM[3] * pVtr[R] + pHri[R] * pM[2] * pVtr[R] + pHrr[R] * pM[2] * pVti[R] - pHri[R] * pM[3] * pVti[R];
                im += pVrr[R] * pM[5] * pHtr[R] + pVri[R] * pM[4] * pHtr[R] + pVrr[R] * pM[4] * pHti[R] - pVri[R] * pM[5] * pHti[R];
                im += pHrr[R] * pM[7] * pHtr[R] + pHri[R] * pM[6] * pHtr[R] + pHrr[R] * pM[6] * pHti[R] - pHri[R] * pM[7] * pHti[R];

                dtype dl = p_delays[o_slice + R]; // path length from previous calculation
                dtype phase = wave_number * std::fmod(dl, wavelength);
                dtype cp = std::cos(phase), sp = std::sin(phase);

                p_coeff_re[o_slice + R] = (re * cp + im * sp) * path_amplitude;
                p_coeff_im[o_slice + R] = (-re * sp + im * cp) * path_amplitude;

                dl = use_absolute_delays ? dl : dl - dist_rx_tx;
                p_delays[o_slice + R] = dl * rC;
            }

        // Apply antenna element coupling
        if (apply_element_coupling)
        {
            // Allocate memory for temporary data
            arma::uword N = (n_ports > n_links) ? n_ports : n_links;
            dtype *tempX = new dtype[N];
            dtype *tempY = new dtype[N];
            dtype *tempZ = new dtype[N];
            dtype *tempT = new dtype[N];

            // Process coefficients and delays
            if (different_output_size) // Data is stored in internal memory, we can write directly to the output
            {
                // Apply coupling to coefficients
                qd_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o_slice], &p_coeff_im[o_slice],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             coeff_re->slice_memptr(j), coeff_im->slice_memptr(j),
                                             n_rx, n_rx_ports, n_tx, n_tx_ports);

                // Apply coupling to delays
                qd_multiply_3_mat(rx_coupling_pwr.memptr(), &p_delays[o_slice], tx_coupling_pwr.memptr(), delay->slice_memptr(j),
                                     n_rx, n_rx_ports, n_tx, n_tx_ports);
            }
            else // Data has been written to external memory
            {
                // Apply coupling to coefficients
                qd_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o_slice], &p_coeff_im[o_slice],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             tempX, tempY,
                                             n_rx, n_rx_ports, n_tx, n_tx_ports);

                std::memcpy(&p_coeff_re[o_slice], tempX, n_ports * sizeof(dtype));
                std::memcpy(&p_coeff_im[o_slice], tempY, n_ports * sizeof(dtype));

                // Apply coupling to delays
                qd_multiply_3_mat(rx_coupling_pwr.memptr(), &p_delays[o_slice], tx_coupling_pwr.memptr(), tempT, n_rx, n_rx_ports, n_tx, n_tx_ports);
                std::memcpy(&p_delays[o_slice], tempT, n_ports * sizeof(dtype));
            }

            // Process departure angles
            if (aod != nullptr || eod != nullptr)
            {
                // Convert AOD and EOD to Cartesian coordinates
                qd_geo2cart(n_links, &p_aod[o_slice], tempX, tempY, &p_eod[o_slice], tempZ);

                // Apply coupling to the x, y, z, component independently
                qd_multiply_3_mat(rx_coupling_pwr.memptr(), tempX, tx_coupling_pwr.memptr(), tempT, n_rx, n_rx_ports, n_tx, n_tx_ports);
                qd_multiply_3_mat(rx_coupling_pwr.memptr(), tempY, tx_coupling_pwr.memptr(), tempX, n_rx, n_rx_ports, n_tx, n_tx_ports);
                qd_multiply_3_mat(rx_coupling_pwr.memptr(), tempZ, tx_coupling_pwr.memptr(), tempY, n_rx, n_rx_ports, n_tx, n_tx_ports);

                // Convert back to geographic coordinates and save to external memory
                if (!different_output_size) // External memory is mapped to "p_aod" and "p_eod"
                    qd_cart2geo(n_ports, &p_aod[o_slice], tempT, tempX, &p_eod[o_slice], tempY);
                else if (aod == nullptr)
                    qd_cart2geo<dtype>(n_ports, NULL, tempT, tempX, eod->slice_memptr(j), tempY);
                else if (eod == nullptr)
                    qd_cart2geo<dtype>(n_ports, aod->slice_memptr(j), tempT, tempX, NULL, tempY);
                else
                    qd_cart2geo<dtype>(n_ports, aod->slice_memptr(j), tempT, tempX, eod->slice_memptr(j), tempY);
            }

            // Process arrival angles
            if (aoa != nullptr || eoa != nullptr)
            {
                // Convert AOD and EOD to Cartesian coordinates
                qd_geo2cart(n_links, &p_aoa[o_slice], tempX, tempY, &p_eoa[o_slice], tempZ);

                // Apply coupling to the x, y, z, component independently
                qd_multiply_3_mat(rx_coupling_pwr.memptr(), tempX, tx_coupling_pwr.memptr(), tempT, n_rx, n_rx_ports, n_tx, n_tx_ports);
                qd_multiply_3_mat(rx_coupling_pwr.memptr(), tempY, tx_coupling_pwr.memptr(), tempX, n_rx, n_rx_ports, n_tx, n_tx_ports);
                qd_multiply_3_mat(rx_coupling_pwr.memptr(), tempZ, tx_coupling_pwr.memptr(), tempY, n_rx, n_rx_ports, n_tx, n_tx_ports);

                // Convert back to geographic coordinates and save to external memory
                if (!different_output_size) // External memory is mapped to "p_aoa" and "p_eoa"
                    qd_cart2geo(n_ports, &p_aoa[o_slice], tempT, tempX, &p_eoa[o_slice], tempY);
                else if (aoa == nullptr)
                    qd_cart2geo<dtype>(n_ports, NULL, tempT, tempX, eoa->slice_memptr(j), tempY);
                else if (eoa == nullptr)
                    qd_cart2geo<dtype>(n_ports, aoa->slice_memptr(j), tempT, tempX, NULL, tempY);
                else
                    qd_cart2geo<dtype>(n_ports, aoa->slice_memptr(j), tempT, tempX, eoa->slice_memptr(j), tempY);
            }

            // Free temporary memory
            delete[] tempX;
            delete[] tempY;
            delete[] tempZ;
            delete[] tempT;
        }
    }

    // Set the true LOS path as the first path
    if (add_fake_los_path && true_los_path != 0ULL)
    {
        dtype *ptrR = different_output_size ? CR.slice_memptr(true_los_path) : coeff_re->slice_memptr(true_los_path);
        dtype *ptrI = different_output_size ? CI.slice_memptr(true_los_path) : coeff_im->slice_memptr(true_los_path);

        std::memcpy(p_coeff_re, ptrR, n_links * sizeof(dtype));
        std::memcpy(p_coeff_im, ptrI, n_links * sizeof(dtype));

        for (arma::uword i = 0ULL; i < n_links; ++i)
            ptrR[i] = zero, ptrI[i] = zero;
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
