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
Array antenna functions
SECTION!*/

/*!MD
# get_channels_planar
Calculate channel coefficients for planar waves

## Description:
- Calculates MIMO channel coefficients and delays for a set of planar wave paths between two antenna arrays.
- Interpolates antenna patterns (including orientation and polarization) for both transmitter and receiver arrays.
- Supports LOS path identification based on distance (angles are ignored).
- Polarization transfer matrix models polarization coupling and must be normalized.
- Doppler weights can optionally be calculated from receiver motion relative to path direction.
- Element positions and antenna orientation are fully considered for delay and phase.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
void quadriga_lib::get_channels_planar(
                const arrayant<dtype> *tx_array,
                const arrayant<dtype> *rx_array,
                dtype Tx, dtype Ty, dtype Tz,
                dtype Tb, dtype Tt, dtype Th,
                dtype Rx, dtype Ry, dtype Rz,
                dtype Rb, dtype Rt, dtype Rh,
                const arma::Col<dtype> *aod,
                const arma::Col<dtype> *eod,
                const arma::Col<dtype> *aoa,
                const arma::Col<dtype> *eoa,
                const arma::Col<dtype> *path_gain,
                const arma::Col<dtype> *path_length,
                const arma::Mat<dtype> *M,
                arma::Cube<dtype> *coeff_re,
                arma::Cube<dtype> *coeff_im,
                arma::Cube<dtype> *delay,
                dtype center_frequency = dtype(0.0),
                bool use_absolute_delays = false,
                bool add_fake_los_path = false,
                arma::Col<dtype> *rx_Doppler = nullptr);
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

- `const arma::Col<dtype> ***aod**` (input)<br>
  Departure azimuth angles in radians, Length `n_path`.

- `const arma::Col<dtype> ***eod**` (input)<br>
  Departure elevation angles in radians, Length `n_path`.

- `const arma::Col<dtype> ***aoa**` (input)<br>
  Arrival azimuth angles in radians, Length `n_path`.

- `const arma::Col<dtype> ***eoa**` (input)<br>
  Arrival elevation angles in radians, Length `n_path`.

- `const arma::Col<dtype> ***path_gain**` (input)<br>
  Path gains in linear scale, Length `n_path`.

- `const arma::Col<dtype> ***path_length**` (input)<br>
  Path lengths from TX to RX phase center, Length `n_path`.

- `const arma::Mat<dtype> ***M**` (input)<br>
  Polarization transfer matrix of size `[8, n_path]`, interleaved: (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH).

- `arma::Cube<dtype> ***coeff_re**` (output)<br>
  Real part of channel coefficients, Size `[n_rx, n_tx, n_path(+1)]`.

- `arma::Cube<dtype> ***coeff_im**` (output)<br>
  Imaginary part of channel coefficients, Size `[n_rx, n_tx, n_path(+1)]`.

- `arma::Cube<dtype> ***delay**` (output)<br>
  Propagation delays in seconds, Size `[n_rx, n_tx, n_path(+1)]`.

- `dtype **center_frequency** = 0.0` (optional input)<br>
  Center frequency in Hz; set to 0 to disable phase calculation. Default: `0.0`

- `bool **use_absolute_delays** = false` (optional input)<br>
  If true, includes LOS delay in all paths. Default: `false`

- `bool **add_fake_los_path** = false` (optional input)<br>
  Adds a zero-power LOS path if no LOS is present. Default: `false`

- `arma::Col<dtype> ***rx_Doppler** = nullptr` (optional output)<br>
  Doppler weights for moving RX, Length `n_path(+1)`. Positive = towards path, Negative = away.
MD!*/

// Calculate channel coefficients for planar waves
template <typename dtype>
void quadriga_lib::get_channels_planar(const quadriga_lib::arrayant<dtype> *tx_array, const quadriga_lib::arrayant<dtype> *rx_array,
                                       dtype Tx, dtype Ty, dtype Tz, dtype Tb, dtype Tt, dtype Th,
                                       dtype Rx, dtype Ry, dtype Rz, dtype Rb, dtype Rt, dtype Rh,
                                       const arma::Col<dtype> *aod, const arma::Col<dtype> *eod, const arma::Col<dtype> *aoa, const arma::Col<dtype> *eoa,
                                       const arma::Col<dtype> *path_gain, const arma::Col<dtype> *path_length, const arma::Mat<dtype> *M,
                                       arma::Cube<dtype> *coeff_re, arma::Cube<dtype> *coeff_im, arma::Cube<dtype> *delay,
                                       dtype center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                       arma::Col<dtype> *rx_Doppler)
{
    // Constants
    constexpr dtype los_limit = dtype(1.0e-4);
    constexpr dtype zero = dtype(0.0);
    constexpr dtype one = dtype(1.0);
    constexpr dtype rC = dtype(1.0 / 299792458.0); // 1 / (Speed of light)
    dtype wavelength = center_frequency > zero ? dtype(299792458.0) / center_frequency : one;
    dtype wave_number = dtype(2.095845021951682e-08) * center_frequency; // 2 * pi / C

    // Catch NULL-Pointers
    std::string error_message;
    if (tx_array == nullptr || rx_array == nullptr ||
        aod == nullptr || eod == nullptr || aoa == nullptr || eoa == nullptr || path_gain == nullptr || path_length == nullptr || M == nullptr ||
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

    arma::uword n_path = aod->n_elem;                               // Number of paths
    arma::uword n_out = add_fake_los_path ? n_path + 1ULL : n_path; // Number of output paths
    arma::uword n_tx = tx_array->e_theta_re.n_slices;               // Number of TX antenna elements before coupling
    arma::uword n_rx = rx_array->e_theta_re.n_slices;               // Number of RX antenna elements before coupling
    arma::uword n_links = n_rx * n_tx;                              // Number of MIMO channel coefficients per path (n_rx * n_tx)
    arma::uword n_tx_ports = tx_array->n_ports();                   // Number of TX antenna elements after coupling
    arma::uword n_rx_ports = rx_array->n_ports();                   // Number of RX antenna elements after coupling
    arma::uword n_ports = n_tx_ports * n_rx_ports;                  // Total number of ports

    // Check if the number of paths is consistent in all inputs
    if (n_path == 0ULL || eod->n_elem != n_path || aoa->n_elem != n_path || eoa->n_elem != n_path || path_gain->n_elem != n_path || path_length->n_elem != n_path || M->n_cols != n_path)
    {
        error_message = "Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'M' must have the same number of columns (n_paths).";
        throw std::invalid_argument(error_message.c_str());
    }
    if (M->n_rows != 8ULL)
    {
        error_message = "Polarization transfer matrix 'M' must have 8 rows.";
        throw std::invalid_argument(error_message.c_str());
    }

    // Set the output size
    if (coeff_re->n_rows != n_rx_ports || coeff_re->n_cols != n_tx_ports || coeff_re->n_slices != n_out)
        coeff_re->set_size(n_rx_ports, n_tx_ports, n_out);
    if (coeff_im->n_rows != n_rx_ports || coeff_im->n_cols != n_tx_ports || coeff_im->n_slices != n_out)
        coeff_im->set_size(n_rx_ports, n_tx_ports, n_out);
    if (delay->n_rows != n_rx_ports || delay->n_cols != n_tx_ports || delay->n_slices != n_out)
        delay->set_size(n_rx_ports, n_tx_ports, n_out);

    // Map output memory to internal representation
    arma::Cube<dtype> CR, CI, DL;

    bool different_output_size = n_tx != n_tx_ports || n_rx != n_rx_ports;
    if (different_output_size) // Temporary internal data storage
    {
        CR.set_size(n_rx, n_tx, n_out);
        CI.set_size(n_rx, n_tx, n_out);
        DL.set_size(n_rx, n_tx, n_out);
    }

    dtype *p_coeff_re = different_output_size ? CR.memptr() : coeff_re->memptr();
    dtype *p_coeff_im = different_output_size ? CI.memptr() : coeff_im->memptr();
    dtype *p_delays = different_output_size ? DL.memptr() : delay->memptr();

    // Antenna interpolation requires matrix of size [1, n_ang], 1 angle per path
    // Angles are mapped to correct value ranges during antenna interpolation
    const arma::Mat<dtype> AOD = arma::Mat<dtype>(const_cast<dtype *>(aod->memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> EOD = arma::Mat<dtype>(const_cast<dtype *>(eod->memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> AOA = arma::Mat<dtype>(const_cast<dtype *>(aoa->memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> EOA = arma::Mat<dtype>(const_cast<dtype *>(eoa->memptr()), 1, n_path, false, true);

    // Get pointers
    const dtype *p_gain = path_gain->memptr();
    const dtype *p_length = path_length->memptr();

    // Convert inputs to orientation vector
    dtype *ptr;
    arma::Cube<dtype> tx_orientation(3, 1, 1);
    arma::Cube<dtype> rx_orientation(3, 1, 1);
    ptr = tx_orientation.memptr(), ptr[0] = Tb, ptr[1] = Tt, ptr[2] = Th;
    ptr = rx_orientation.memptr(), ptr[0] = Rb, ptr[1] = Rt, ptr[2] = Rh;

    // Calculate the Freespace distance
    dtype x = Rx - Tx, y = Ry - Ty, z = Rz - Tz;
    dtype dist_rx_tx = std::sqrt(x * x + y * y + z * z);

    // Interpolate the antenna patterns for all paths
    arma::Mat<dtype> Vt_re(n_tx, n_out, arma::fill::none), Vt_im(n_tx, n_out, arma::fill::none),
        Ht_re(n_tx, n_out, arma::fill::none), Ht_im(n_tx, n_out, arma::fill::none),
        Vr_re(n_rx, n_out, arma::fill::none), Vr_im(n_rx, n_out, arma::fill::none),
        Hr_re(n_rx, n_out, arma::fill::none), Hr_im(n_rx, n_out, arma::fill::none),
        Pt(n_tx, n_out, arma::fill::none), Pr(n_rx, n_out, arma::fill::none);
    arma::Mat<dtype> AOA_loc, EOA_loc, EMPTY;

    // To calculate the Doppler weights, we need the arrival in local antenna-coordinates
    if (rx_Doppler != nullptr)
    {
        AOA_loc.set_size(n_rx, n_path);
        EOA_loc.set_size(n_rx, n_path);
        if (rx_Doppler->n_elem != n_out)
            rx_Doppler->set_size(n_out);
    }

    arma::Col<unsigned> i_element(n_tx, arma::fill::none);
    unsigned *p_element = i_element.memptr();
    for (unsigned t = 0; t < unsigned(n_tx); ++t)
        *p_element++ = t + 1;

    arma::Mat<dtype> element_pos_interp(3ULL, n_tx);
    if (tx_array->element_pos.n_elem != 0ULL)
        std::memcpy(element_pos_interp.memptr(), tx_array->element_pos.memptr(), 3ULL * n_tx * sizeof(dtype));

    qd_arrayant_interpolate(&tx_array->e_theta_re, &tx_array->e_theta_im, &tx_array->e_phi_re, &tx_array->e_phi_im,
                            &tx_array->azimuth_grid, &tx_array->elevation_grid, &AOD, &EOD,
                            &i_element, &tx_orientation, &element_pos_interp,
                            &Vt_re, &Vt_im, &Ht_re, &Ht_im, &Pt, &EMPTY, &EMPTY, &EMPTY);

    i_element.set_size(n_rx);
    p_element = i_element.memptr();
    for (unsigned r = 0; r < unsigned(n_rx); ++r)
        *p_element++ = r + 1;

    element_pos_interp.zeros(3ULL, n_rx);
    if (rx_array->element_pos.n_elem != 0ULL)
        std::memcpy(element_pos_interp.memptr(), rx_array->element_pos.memptr(), 3ULL * n_rx * sizeof(dtype));

    qd_arrayant_interpolate(&rx_array->e_theta_re, &rx_array->e_theta_im, &rx_array->e_phi_re, &rx_array->e_phi_im,
                            &rx_array->azimuth_grid, &rx_array->elevation_grid, &AOA, &EOA,
                            &i_element, &rx_orientation, &element_pos_interp,
                            &Vr_re, &Vr_im, &Hr_re, &Hr_im, &Pr, &AOA_loc, &EOA_loc, &EMPTY);

    element_pos_interp.reset();

    // Calculate the Doppler weights
    if (rx_Doppler != nullptr)
    {
        dtype *pAz = AOA_loc.memptr(), *pEl = EOA_loc.memptr();
        dtype *pD = add_fake_los_path ? rx_Doppler->memptr() + 1 : rx_Doppler->memptr();
        for (auto i = 0ULL; i < n_path; ++i)
            pD[i] = std::cos(pAz[i * n_rx]) * std::cos(pEl[i * n_rx]);
    }

    // Calculate the MIMO channel coefficients for each path
    arma::uword true_los_path = 0ULL;
    dtype true_los_power = zero;
    for (arma::uword j = 0ULL; j < n_out; ++j) // Loop over paths
    {
        arma::uword i = add_fake_los_path ? (j == 0ULL ? 0ULL : j - 1ULL) : j;

        const dtype *pM = M->colptr(i);
        dtype *pVrr = Vr_re.colptr(i), *pVri = Vr_im.colptr(i),
              *pHrr = Hr_re.colptr(i), *pHri = Hr_im.colptr(i),
              *pVtr = Vt_re.colptr(i), *pVti = Vt_im.colptr(i),
              *pHtr = Ht_re.colptr(i), *pHti = Ht_im.colptr(i);

        dtype *pPt = Pt.colptr(i), *pPr = Pr.colptr(i);

        dtype path_amplitude = add_fake_los_path && j == 0ULL ? zero : std::sqrt(p_gain[i]);
        dtype path_length = add_fake_los_path && j == 0ULL ? dist_rx_tx : p_length[i];

        // LOS path detection
        if (std::abs(path_length - dist_rx_tx) < los_limit && add_fake_los_path && j != 0ULL && p_gain[i] > true_los_power)
            true_los_path = j, true_los_power = p_gain[i];

        arma::uword O = j * n_links; // Slice offset
        for (arma::uword t = 0ULL; t < n_tx; ++t)
            for (arma::uword r = 0ULL; r < n_rx; ++r)
            {
                arma::uword R = t * n_rx + r;

                dtype re = zero, im = zero;
                re += pVrr[r] * pM[0] * pVtr[t] - pVri[r] * pM[1] * pVtr[t] - pVrr[r] * pM[1] * pVti[t] - pVri[r] * pM[0] * pVti[t];
                re += pHrr[r] * pM[2] * pVtr[t] - pHri[r] * pM[3] * pVtr[t] - pHrr[r] * pM[3] * pVti[t] - pHri[r] * pM[2] * pVti[t];
                re += pVrr[r] * pM[4] * pHtr[t] - pVri[r] * pM[5] * pHtr[t] - pVrr[r] * pM[5] * pHti[t] - pVri[r] * pM[4] * pHti[t];
                re += pHrr[r] * pM[6] * pHtr[t] - pHri[r] * pM[7] * pHtr[t] - pHrr[r] * pM[7] * pHti[t] - pHri[r] * pM[6] * pHti[t];

                im += pVrr[r] * pM[1] * pVtr[t] + pVri[r] * pM[0] * pVtr[t] + pVrr[r] * pM[0] * pVti[t] - pVri[r] * pM[1] * pVti[t];
                im += pHrr[r] * pM[3] * pVtr[t] + pHri[r] * pM[2] * pVtr[t] + pHrr[r] * pM[2] * pVti[t] - pHri[r] * pM[3] * pVti[t];
                im += pVrr[r] * pM[5] * pHtr[t] + pVri[r] * pM[4] * pHtr[t] + pVrr[r] * pM[4] * pHti[t] - pVri[r] * pM[5] * pHti[t];
                im += pHrr[r] * pM[7] * pHtr[t] + pHri[r] * pM[6] * pHtr[t] + pHrr[r] * pM[6] * pHti[t] - pHri[r] * pM[7] * pHti[t];

                dtype dl = pPt[t] + path_length + pPr[r];
                dtype phase = wave_number * std::fmod(dl, wavelength);
                dtype cp = std::cos(phase), sp = std::sin(phase);

                p_coeff_re[O + R] = (re * cp + im * sp) * path_amplitude;
                p_coeff_im[O + R] = (-re * sp + im * cp) * path_amplitude;

                dl = use_absolute_delays ? dl : dl - dist_rx_tx;
                p_delays[O + R] = dl * rC;
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

    // Apply antenna element coupling
    if (rx_array->coupling_re.n_elem != 0ULL || rx_array->coupling_im.n_elem != 0ULL ||
        tx_array->coupling_re.n_elem != 0ULL || tx_array->coupling_im.n_elem != 0ULL)
    {
        // Calculate abs( cpl )^2 and normalize the row-sum to 1
        dtype *p_rx_cpl = new dtype[n_rx * n_rx_ports];
        dtype *p_tx_cpl = new dtype[n_tx * n_tx_ports];
        qd_power_mat(n_rx, n_rx_ports, p_rx_cpl, true, rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr());
        qd_power_mat(n_tx, n_tx_ports, p_tx_cpl, true, tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr());

        // Process coefficients and delays
        if (different_output_size) // Data is stored in internal memory, we can write directly to the output
        {
            for (arma::uword j = 0ULL; j < n_out; ++j) // Loop over paths
            {
                arma::uword o = j * n_links; // Slice offset

                // Apply coupling to coefficients
                qd_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o], &p_coeff_im[o],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             coeff_re->slice_memptr(j), coeff_im->slice_memptr(j),
                                             n_rx, n_rx_ports, n_tx, n_tx_ports);

                // Apply coupling to delays
                qd_multiply_3_mat(p_rx_cpl, &p_delays[o], p_tx_cpl, delay->slice_memptr(j), n_rx, n_rx_ports, n_tx, n_tx_ports);
            }
        }
        else // Data has been written to external memory
        {
            // Allocate memory for temporary data
            dtype *tempX = new dtype[n_ports];
            dtype *tempY = new dtype[n_ports];

            for (arma::uword o = 0ULL; o < n_links * n_out; o += n_links) // Loop over paths
            {
                // Apply coupling to coefficients
                qd_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o], &p_coeff_im[o],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             tempX, tempY,
                                             n_rx, n_rx_ports, n_tx, n_tx_ports);

                std::memcpy(&p_coeff_re[o], tempX, n_ports * sizeof(dtype));
                std::memcpy(&p_coeff_im[o], tempY, n_ports * sizeof(dtype));

                // Apply coupling to delays
                qd_multiply_3_mat(p_rx_cpl, &p_delays[o], p_tx_cpl, tempX, n_rx, n_rx_ports, n_tx, n_tx_ports);
                std::memcpy(&p_delays[o], tempX, n_ports * sizeof(dtype));
            }
            delete[] tempX;
            delete[] tempY;
        }

        // Free temporary memory
        delete[] p_rx_cpl;
        delete[] p_tx_cpl;
    }
}

template void quadriga_lib::get_channels_planar(const quadriga_lib::arrayant<float> *tx_array, const quadriga_lib::arrayant<float> *rx_array,
                                                float Tx, float Ty, float Tz, float Tb, float Tt, float Th,
                                                float Rx, float Ry, float Rz, float Rb, float Rt, float Rh,
                                                const arma::Col<float> *aod, const arma::Col<float> *eod, const arma::Col<float> *aoa, const arma::Col<float> *eoa,
                                                const arma::Col<float> *path_gain, const arma::Col<float> *path_length, const arma::Mat<float> *M,
                                                arma::Cube<float> *coeff_re, arma::Cube<float> *coeff_im, arma::Cube<float> *delay,
                                                float center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                                arma::Col<float> *rx_Doppler);

template void quadriga_lib::get_channels_planar(const quadriga_lib::arrayant<double> *tx_array, const quadriga_lib::arrayant<double> *rx_array,
                                                double Tx, double Ty, double Tz, double Tb, double Tt, double Th,
                                                double Rx, double Ry, double Rz, double Rb, double Rt, double Rh,
                                                const arma::Col<double> *aod, const arma::Col<double> *eod, const arma::Col<double> *aoa, const arma::Col<double> *eoa,
                                                const arma::Col<double> *path_gain, const arma::Col<double> *path_length, const arma::Mat<double> *M,
                                                arma::Cube<double> *coeff_re, arma::Cube<double> *coeff_im, arma::Cube<double> *delay,
                                                double center_frequency, bool use_absolute_delays, bool add_fake_los_path,
                                                arma::Col<double> *rx_Doppler);
