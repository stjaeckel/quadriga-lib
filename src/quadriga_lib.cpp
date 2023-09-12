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

#include <stdexcept>
#include <cstring> // For std::memcopy
#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"
#include "qd_arrayant_qdant.hpp"
#include "qd_arrayant_interpolate.hpp"

// Template for time measuring:
// #include <chrono>
//
// Init:
// std::chrono::high_resolution_clock::time_point ts = std::chrono::high_resolution_clock::now(), te;
// arma::uword dur = 0;
//
// Read:
// te = std::chrono::high_resolution_clock::now();
// dur = (arma::uword)std::chrono::duration_cast<std::chrono::nanoseconds>(te - ts).count();
// ts = te;
// std::cout << "A = " << 1.0e-9 * double(dur) << std::endl;

// Returns the arrayant_lib version number as a string
#define AUX(x) #x
#define STRINGIFY(x) AUX(x)
std::string quadriga_lib::quadriga_lib_version()
{
    std::string str = STRINGIFY(QUADRIGA_LIB_VERSION);
    std::size_t found = str.find_first_of("_");
    str.replace(found, 1, ".");
    found = str.find_first_of("_");
    str.replace(found, 1, ".");
    str = str.substr(1, str.length());
    return str;
}

#include "helper_functions.cpp"

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
    constexpr dtype los_limit = dtype(1.0e-4);
    constexpr dtype zero = dtype(0.0);
    constexpr dtype one = dtype(1.0);
    constexpr dtype rC = dtype(1.0 / 299792458.0); // 1 / (Speed of light)
    dtype wavelength = center_frequency > zero ? dtype(299792458.0) / center_frequency : one;
    dtype wave_number = dtype(2.095845021951682e-08) * center_frequency; // 2 * pi / C

    // Catch NULL-Pointers
    std::string error_message;
    if (tx_array == NULL || rx_array == NULL ||
        fbs_pos == NULL || lbs_pos == NULL || path_gain == NULL || path_length == NULL || M == NULL ||
        coeff_re == NULL || coeff_im == NULL || delay == NULL)
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
    arma::uword n_path = fbs_pos->n_cols;                        // Number of paths
    arma::uword n_out = add_fake_los_path ? n_path + 1 : n_path; // Number of output paths
    arma::uword n_tx = tx_array->e_theta_re.n_slices;            // Number of TX antenna elements before coupling
    arma::uword n_rx = rx_array->e_theta_re.n_slices;            // Number of RX antenna elements before coupling
    arma::uword n_links = n_rx * n_tx;                           // Number of MIMO channel coefficients per path (n_rx * n_tx)
    arma::uword n_tx_ports = tx_array->n_ports();                // Number of TX antenna elements after coupling
    arma::uword n_rx_ports = rx_array->n_ports();                // Number of RX antenna elements after coupling

    // 32-bit integers used in loops
    unsigned n32_out = unsigned(n_out);
    unsigned n32_tx = unsigned(n_tx);
    unsigned n32_rx = unsigned(n_rx);
    unsigned n32_links = unsigned(n_links);
    unsigned n32_tx_ports = unsigned(n_tx_ports);
    unsigned n32_rx_ports = unsigned(n_rx_ports);
    unsigned n32_ports = n32_tx_ports * n32_rx_ports;

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
    if (aod != NULL && (aod->n_rows != n_rx_ports || aod->n_cols != n_tx_ports || aod->n_slices != n_out))
        aod->set_size(n_rx_ports, n_tx_ports, n_out);
    if (eod != NULL && (eod->n_rows != n_rx_ports || eod->n_cols != n_tx_ports || eod->n_slices != n_out))
        eod->set_size(n_rx_ports, n_tx_ports, n_out);
    if (aoa != NULL && (aoa->n_rows != n_rx_ports || aoa->n_cols != n_tx_ports || aoa->n_slices != n_out))
        aoa->set_size(n_rx_ports, n_tx_ports, n_out);
    if (eoa != NULL && (eoa->n_rows != n_rx_ports || eoa->n_cols != n_tx_ports || eoa->n_slices != n_out))
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

        if (aod == NULL)
            AOD.set_size(n_links, n_out);
        else
            AOD = arma::Mat<dtype>(aod->memptr(), n_links, n_out, false, true);

        if (eod == NULL)
            EOD.set_size(n_links, n_out);
        else
            EOD = arma::Mat<dtype>(eod->memptr(), n_links, n_out, false, true);

        if (aoa == NULL)
            AOA.set_size(n_links, n_out);
        else
            AOA = arma::Mat<dtype>(aoa->memptr(), n_links, n_out, false, true);

        if (eoa == NULL)
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
    bool tx_orientation_not_zero = Tb != zero || Tt != zero || Th != zero;
    bool rx_orientation_not_zero = Rb != zero || Rt != zero || Rh != zero;

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
    for (unsigned t = 0; t < 3 * n32_tx; t += 3) // Add TX position
        p_tx[t] += Tx, p_tx[t + 1] += Ty, p_tx[t + 2] += Tz;

    if (rx_array->element_pos.n_elem != 0)
        std::memcpy(p_rx, rx_array->element_pos.memptr(), 3 * n_rx * sizeof(dtype));
    if (rx_orientation_not_zero) // Apply RX antenna orientation
        quick_rotate_inplace(Rb, -Rt, Rh, p_rx, n_rx);
    for (unsigned r = 0; r < 3 * n32_rx; r += 3) // Add RX position
        p_rx[r] += Rx, p_rx[r + 1] += Ry, p_rx[r + 2] += Rz;

    // Calculate the Freespace distance
    dtype x = Rx - Tx, y = Ry - Ty, z = Rz - Tz;
    dtype dist_rx_tx = std::sqrt(x * x + y * y + z * z);

    // There may be multiple LOS paths. We need to find the real one
    // Detection is done by sing the shortest length difference to the TX-RX line
    unsigned true_los_path = 0;
    dtype shortest_path = los_limit;

    // Calculate angles and delays
    for (unsigned j = 0; j < n32_out; j++) // Loop over paths
    {
        unsigned i = add_fake_los_path ? j - 1 : j;
        unsigned ix = 3 * i, iy = ix + 1, iz = ix + 2;

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
            d_length = p_length[i] < d_shortest ? d_shortest : p_length[i];
        }

        // Calculate path delays, departure angles and arrival angles
        unsigned o = j * n32_links, o0 = o;                  // Slice offset
        if (std::abs(d_length - dist_rx_tx) < shortest_path) // LOS path
        {
            if (!add_fake_los_path || j != 0)
                true_los_path = j, shortest_path = std::abs(d_length - dist_rx_tx);

            for (unsigned t = 0; t < n32_tx; t++)
                for (unsigned r = 0; r < n32_rx; r++)
                {
                    x = p_rx[3 * r] - p_tx[3 * t];
                    y = p_rx[3 * r + 1] - p_tx[3 * t + 1];
                    z = p_rx[3 * r + 2] - p_tx[3 * t + 2];
                    dtype d = std::sqrt(x * x + y * y + z * z);

                    p_aod[o] = std::atan2(y, x);
                    p_eod[o] = d < los_limit ? zero : std::asin(z / d);
                    p_aoa[o] = std::atan2(-y, -x);
                    p_eoa[o] = -p_eod[o];
                    p_delays[o++] = d;
                }
        }
        else // NLOS path
        {
            dtype *dr = new dtype[n32_rx];
            for (unsigned r = 0; r < n32_rx; r++)
                x = p_lbs[ix] - p_rx[3 * r],
                y = p_lbs[iy] - p_rx[3 * r + 1],
                z = p_lbs[iz] - p_rx[3 * r + 2],
                dr[r] = std::sqrt(x * x + y * y + z * z),
                p_aoa[o0 + r] = std::atan2(y, x),
                p_eoa[o0 + r] = dr[r] < los_limit ? zero : std::asin(z / dr[r]);

            for (unsigned t = 0; t < n32_tx; t++)
            {
                x = p_fbs[ix] - p_tx[3 * t],
                y = p_fbs[iy] - p_tx[3 * t + 1],
                z = p_fbs[iz] - p_tx[3 * t + 2];

                dtype dt = std::sqrt(x * x + y * y + z * z),
                      at = std::atan2(y, x),
                      et = dt < los_limit ? zero : std::asin(z / dt);

                for (unsigned r = 0; r < n32_rx; r++)
                    p_aod[o] = at, p_eod[o] = et,
                    p_aoa[o] = p_aoa[o0 + r], p_eoa[o] = p_eoa[o0 + r],
                    p_delays[o++] = dt + d_fbs_lbs + dr[r];
            }
            delete[] dr;
        }
    }

    // Interpolate the antenna patterns for all paths
    // - ToDo: Performance can be improved by omitting redundant computations for NLOS paths

    arma::Mat<dtype> Vt_re(n_links, n_out, arma::fill::none), Vt_im(n_links, n_out, arma::fill::none),
        Ht_re(n_links, n_out, arma::fill::none), Ht_im(n_links, n_out, arma::fill::none),
        Vr_re(n_links, n_out, arma::fill::none), Vr_im(n_links, n_out, arma::fill::none),
        Hr_re(n_links, n_out, arma::fill::none), Hr_im(n_links, n_out, arma::fill::none);
    arma::Mat<dtype> EMPTY;

    arma::Col<unsigned> i_element(n_links, arma::fill::none);
    unsigned *p_element = i_element.memptr();
    arma::Mat<dtype> element_pos_interp(3, n_links, arma::fill::none);
    ptr = element_pos_interp.memptr();
    if (tx_array->element_pos.n_elem != 0)
        std::memcpy(p_tx, tx_array->element_pos.memptr(), 3 * n_tx * sizeof(dtype));
    else
        tx_element_pos.zeros();
    for (unsigned t = 0; t < n32_tx; t++)
        for (unsigned r = 0; r < n32_rx; r++)
            *p_element++ = t + 1, *ptr++ = p_tx[3 * t], *ptr++ = p_tx[3 * t + 1], *ptr++ = p_tx[3 * t + 2];

    qd_arrayant_interpolate(&tx_array->e_theta_re, &tx_array->e_theta_im, &tx_array->e_phi_re, &tx_array->e_phi_im,
                            &tx_array->azimuth_grid, &tx_array->elevation_grid, &AOD, &EOD,
                            &i_element, &tx_orientation, &element_pos_interp,
                            &Vt_re, &Vt_im, &Ht_re, &Ht_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);

    p_element = i_element.memptr();
    ptr = element_pos_interp.memptr();
    if (rx_array->element_pos.n_elem != 0)
        std::memcpy(p_rx, rx_array->element_pos.memptr(), 3 * n_rx * sizeof(dtype));
    else
        rx_element_pos.zeros();
    for (unsigned t = 0; t < n32_tx; t++)
        for (unsigned r = 0; r < n32_rx; r++)
            *p_element++ = r + 1, *ptr++ = p_rx[3 * r], *ptr++ = p_rx[3 * r + 1], *ptr++ = p_rx[3 * r + 2];

    qd_arrayant_interpolate(&rx_array->e_theta_re, &rx_array->e_theta_im, &rx_array->e_phi_re, &rx_array->e_phi_im,
                            &rx_array->azimuth_grid, &rx_array->elevation_grid, &AOA, &EOA,
                            &i_element, &rx_orientation, &element_pos_interp,
                            &Vr_re, &Vr_im, &Hr_re, &Hr_im, &EMPTY, &EMPTY, &EMPTY, &EMPTY);
    element_pos_interp.reset();

    // Calculate the MIMO channel coefficients for each path
    for (unsigned j = 0; j < n32_out; j++) // Loop over paths
    {
        unsigned i = add_fake_los_path ? (j == 0 ? 0 : j - 1) : j;

        const dtype *pM = M->colptr(i);
        dtype *pVrr = Vr_re.colptr(j), *pVri = Vr_im.colptr(j),
              *pHrr = Hr_re.colptr(j), *pHri = Hr_im.colptr(j),
              *pVtr = Vt_re.colptr(j), *pVti = Vt_im.colptr(j),
              *pHtr = Ht_re.colptr(j), *pHti = Ht_im.colptr(j);

        dtype path_amplitude = add_fake_los_path && j == 0 ? zero : std::sqrt(p_gain[i]);

        unsigned O = j * n32_links; // Slice offset
        for (unsigned t = 0; t < n32_tx; t++)
            for (unsigned r = 0; r < n32_rx; r++)
            {
                unsigned R = t * n32_rx + r;

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
    }

    // Set the true LOS path as the first path
    if (add_fake_los_path && true_los_path != 0)
    {
        std::memcpy(p_coeff_re, CR.slice_memptr(true_los_path), n32_links * sizeof(dtype));
        std::memcpy(p_coeff_im, CI.slice_memptr(true_los_path), n32_links * sizeof(dtype));
        CR.slice(true_los_path).zeros();
        CI.slice(true_los_path).zeros();
    }

    // Apply antenna element coupling
    if (rx_array->coupling_re.n_elem != 0 || rx_array->coupling_im.n_elem != 0 ||
        tx_array->coupling_re.n_elem != 0 || tx_array->coupling_im.n_elem != 0)
    {
        // Calculate abs( cpl )^2 and normalize the row-sum to 1
        dtype *p_rx_cpl = new dtype[n32_rx * n32_rx_ports];
        dtype *p_tx_cpl = new dtype[n32_tx * n32_tx_ports];
        quick_power_mat(n32_rx, n32_rx_ports, p_rx_cpl, true, rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr());
        quick_power_mat(n32_tx, n32_tx_ports, p_tx_cpl, true, tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr());

        // Allocate memory for temporary data
        unsigned N = n32_ports > n32_links ? n32_ports : n32_links;
        dtype *tempX = new dtype[N];
        dtype *tempY = new dtype[N];
        dtype *tempZ = new dtype[N];
        dtype *tempT = new dtype[N];

        for (unsigned j = 0; j < n32_out; j++) // Loop over paths
        {
            unsigned o = j * n32_links; // Slice offset

            // Process coefficients and delays
            if (different_output_size) // Data is stored in internal memory, we can write directly to the output
            {
                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o], &p_coeff_im[o],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             coeff_re->slice_memptr(j), coeff_im->slice_memptr(j),
                                             n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);

                // Apply coupling to delays
                quick_multiply_3_mat(p_rx_cpl, &p_delays[o], p_tx_cpl, delay->slice_memptr(j), n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);
            }
            else // Data has been written to external memory
            {
                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o], &p_coeff_im[o],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             tempX, tempY,
                                             n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);

                std::memcpy(&p_coeff_re[o], tempX, n32_ports * sizeof(dtype));
                std::memcpy(&p_coeff_im[o], tempY, n32_ports * sizeof(dtype));

                // Apply coupling to delays
                quick_multiply_3_mat(p_rx_cpl, &p_delays[o], p_tx_cpl, tempT, n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);
                std::memcpy(&p_delays[o], tempT, n32_ports * sizeof(dtype));
            }

            // Process departure angles
            if (aod != NULL || eod != NULL)
            {
                // Convert AOD and EOD to Cartesian coordinates
                quick_geo2cart(n32_links, &p_aod[o], tempX, tempY, &p_eod[o], tempZ);

                // Apply coupling to the x, y, z, component independently
                quick_multiply_3_mat(p_rx_cpl, tempX, p_tx_cpl, tempT, n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);
                quick_multiply_3_mat(p_rx_cpl, tempY, p_tx_cpl, tempX, n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);
                quick_multiply_3_mat(p_rx_cpl, tempZ, p_tx_cpl, tempY, n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);

                // Convert back to geographic coordinates and save to external memory
                if (!different_output_size) // External memory is mapped to "p_aod" and "p_eod"
                    quick_cart2geo(n32_ports, &p_aod[o], tempT, tempX, &p_eod[o], tempY);
                else if (aod == NULL)
                    quick_cart2geo<dtype>(n32_ports, NULL, tempT, tempX, eod->slice_memptr(j), tempY);
                else if (eod == NULL)
                    quick_cart2geo<dtype>(n32_ports, aod->slice_memptr(j), tempT, tempX, NULL, tempY);
                else
                    quick_cart2geo<dtype>(n32_ports, aod->slice_memptr(j), tempT, tempX, eod->slice_memptr(j), tempY);
            }

            // Process arrival angles
            if (aoa != NULL || eoa != NULL)
            {
                // Convert AOD and EOD to Cartesian coordinates
                quick_geo2cart(n32_links, &p_aoa[o], tempX, tempY, &p_eoa[o], tempZ);

                // Apply coupling to the x, y, z, component independently
                quick_multiply_3_mat(p_rx_cpl, tempX, p_tx_cpl, tempT, n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);
                quick_multiply_3_mat(p_rx_cpl, tempY, p_tx_cpl, tempX, n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);
                quick_multiply_3_mat(p_rx_cpl, tempZ, p_tx_cpl, tempY, n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);

                // Convert back to geographic coordinates and save to external memory
                if (!different_output_size) // External memory is mapped to "p_aoa" and "p_eoa"
                    quick_cart2geo(n32_ports, &p_aoa[o], tempT, tempX, &p_eoa[o], tempY);
                else if (aoa == NULL)
                    quick_cart2geo<dtype>(n32_ports, NULL, tempT, tempX, eoa->slice_memptr(j), tempY);
                else if (eoa == NULL)
                    quick_cart2geo<dtype>(n32_ports, aoa->slice_memptr(j), tempT, tempX, NULL, tempY);
                else
                    quick_cart2geo<dtype>(n32_ports, aoa->slice_memptr(j), tempT, tempX, eoa->slice_memptr(j), tempY);
            }
        }

        // Free temporary memory
        delete[] tempX;
        delete[] tempY;
        delete[] tempZ;
        delete[] tempT;
        delete[] p_rx_cpl;
        delete[] p_tx_cpl;
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
    if (tx_array == NULL || rx_array == NULL ||
        aod == NULL || eod == NULL || aoa == NULL || eoa == NULL || path_gain == NULL || path_length == NULL || M == NULL ||
        coeff_re == NULL || coeff_im == NULL || delay == NULL)
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

    // 64-bit integers used by Armadillo
    arma::uword n_path = aod->n_elem;                            // Number of paths
    arma::uword n_out = add_fake_los_path ? n_path + 1 : n_path; // Number of output paths
    arma::uword n_tx = tx_array->e_theta_re.n_slices;            // Number of TX antenna elements before coupling
    arma::uword n_rx = rx_array->e_theta_re.n_slices;            // Number of RX antenna elements before coupling
    arma::uword n_links = n_rx * n_tx;                           // Number of MIMO channel coefficients per path (n_rx * n_tx)
    arma::uword n_tx_ports = tx_array->n_ports();                // Number of TX antenna elements after coupling
    arma::uword n_rx_ports = rx_array->n_ports();                // Number of RX antenna elements after coupling

    // 32-bit integers used in loops
    unsigned n32_out = unsigned(n_out);
    unsigned n32_tx = unsigned(n_tx);
    unsigned n32_rx = unsigned(n_rx);
    unsigned n32_links = unsigned(n_links);
    unsigned n32_tx_ports = unsigned(n_tx_ports);
    unsigned n32_rx_ports = unsigned(n_rx_ports);
    unsigned n32_ports = n32_tx_ports * n32_rx_ports;

    // Check if the number of paths is consistent in all inputs
    if (n_path == 0 || eod->n_elem != n_path || aoa->n_elem != n_path || eoa->n_elem != n_path || path_gain->n_elem != n_path || path_length->n_elem != n_path || M->n_cols != n_path)
    {
        error_message = "Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'M' must have the same number of columns (n_paths).";
        throw std::invalid_argument(error_message.c_str());
    }
    if (M->n_rows != 8)
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
    arma::Cube<dtype> CR, CI, DL; // Internal map for coefficients
    bool different_output_size = n_tx != n_tx_ports || n_rx != n_rx_ports;

    if (different_output_size) // Temporary internal data storage
    {
        CR.set_size(n_rx, n_tx, n_out);
        CI.set_size(n_rx, n_tx, n_out);
        DL.set_size(n_rx, n_tx, n_out);
    }
    else // Direct mapping of external memory
    {
        CR = arma::Cube<dtype>(coeff_re->memptr(), n_rx, n_tx, n_out, false, true);
        CI = arma::Cube<dtype>(coeff_im->memptr(), n_rx, n_tx, n_out, false, true);
        DL = arma::Cube<dtype>(delay->memptr(), n_rx, n_tx, n_out, false, true);
    }

    // Antenna interpolation requires matrix of size [1, n_ang], 1 angle per path
    // Angles are mapped to correct value ranges during antenna interpolation
    const arma::Mat<dtype> AOD = arma::Mat<dtype>(const_cast<dtype *>(aod->memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> EOD = arma::Mat<dtype>(const_cast<dtype *>(eod->memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> AOA = arma::Mat<dtype>(const_cast<dtype *>(aoa->memptr()), 1, n_path, false, true);
    const arma::Mat<dtype> EOA = arma::Mat<dtype>(const_cast<dtype *>(eoa->memptr()), 1, n_path, false, true);

    // Get pointers
    const dtype *p_gain = path_gain->memptr();
    const dtype *p_length = path_length->memptr();

    dtype *p_coeff_re = CR.memptr();
    dtype *p_coeff_im = CI.memptr();
    dtype *p_delays = DL.memptr();
    dtype *ptr;

    // Convert inputs to orientation vector
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
    if (rx_Doppler != NULL)
    {
        AOA_loc.set_size(n_rx, n_path);
        EOA_loc.set_size(n_rx, n_path);
        if (rx_Doppler->n_elem != n_out)
            rx_Doppler->set_size(n_out);
    }

    arma::Col<unsigned> i_element(n_tx, arma::fill::none);
    unsigned *p_element = i_element.memptr();
    for (unsigned t = 0; t < n32_tx; t++)
        *p_element++ = t + 1;

    arma::Mat<dtype> element_pos_interp(3, n_tx);
    if (tx_array->element_pos.n_elem != 0)
        std::memcpy(element_pos_interp.memptr(), tx_array->element_pos.memptr(), 3 * n_tx * sizeof(dtype));

    qd_arrayant_interpolate(&tx_array->e_theta_re, &tx_array->e_theta_im, &tx_array->e_phi_re, &tx_array->e_phi_im,
                            &tx_array->azimuth_grid, &tx_array->elevation_grid, &AOD, &EOD,
                            &i_element, &tx_orientation, &element_pos_interp,
                            &Vt_re, &Vt_im, &Ht_re, &Ht_im, &Pt, &EMPTY, &EMPTY, &EMPTY);

    i_element.set_size(n_rx);
    p_element = i_element.memptr();
    for (unsigned r = 0; r < n32_rx; r++)
        *p_element++ = r + 1;

    element_pos_interp.zeros(3, n_rx);
    if (rx_array->element_pos.n_elem != 0)
        std::memcpy(element_pos_interp.memptr(), rx_array->element_pos.memptr(), 3 * n_rx * sizeof(dtype));

    qd_arrayant_interpolate(&rx_array->e_theta_re, &rx_array->e_theta_im, &rx_array->e_phi_re, &rx_array->e_phi_im,
                            &rx_array->azimuth_grid, &rx_array->elevation_grid, &AOA, &EOA,
                            &i_element, &rx_orientation, &element_pos_interp,
                            &Vr_re, &Vr_im, &Hr_re, &Hr_im, &Pr, &AOA_loc, &EOA_loc, &EMPTY);

    element_pos_interp.reset();

    // Calculate the Doppler weights
    if (rx_Doppler != NULL)
    {
        dtype *pAz = AOA_loc.memptr(), *pEl = EOA_loc.memptr();
        dtype *pD = add_fake_los_path ? rx_Doppler->memptr() + 1 : rx_Doppler->memptr();
        for (arma::uword i = 0; i < n_path; i++)
            pD[i] = std::cos(pAz[i * n_rx]) * std::cos(pEl[i * n_rx]);
    }

    // Calculate the MIMO channel coefficients for each path
    unsigned true_los_path = 0;
    dtype true_los_power = zero;
    for (unsigned j = 0; j < n32_out; j++) // Loop over paths
    {
        unsigned i = add_fake_los_path ? (j == 0 ? 0 : j - 1) : j;

        const dtype *pM = M->colptr(i);
        dtype *pVrr = Vr_re.colptr(i), *pVri = Vr_im.colptr(i),
              *pHrr = Hr_re.colptr(i), *pHri = Hr_im.colptr(i),
              *pVtr = Vt_re.colptr(i), *pVti = Vt_im.colptr(i),
              *pHtr = Ht_re.colptr(i), *pHti = Ht_im.colptr(i);

        dtype *pPt = Pt.colptr(i), *pPr = Pr.colptr(i);

        dtype path_amplitude = add_fake_los_path && j == 0 ? zero : std::sqrt(p_gain[i]);
        dtype path_length = add_fake_los_path && j == 0 ? dist_rx_tx : p_length[i];

        // LOS path detection
        if (std::abs(path_length - dist_rx_tx) < los_limit && add_fake_los_path && j != 0 && p_gain[i] > true_los_power)
            true_los_path = j, true_los_power = p_gain[i];

        unsigned O = j * n32_links; // Slice offset
        for (unsigned t = 0; t < n32_tx; t++)
            for (unsigned r = 0; r < n32_rx; r++)
            {
                unsigned R = t * n32_rx + r;

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
    if (add_fake_los_path && true_los_path != 0)
    {
        std::memcpy(p_coeff_re, CR.slice_memptr(true_los_path), n32_links * sizeof(dtype));
        std::memcpy(p_coeff_im, CI.slice_memptr(true_los_path), n32_links * sizeof(dtype));
        CR.slice(true_los_path).zeros();
        CI.slice(true_los_path).zeros();
    }

    // Apply antenna element coupling
    if (rx_array->coupling_re.n_elem != 0 || rx_array->coupling_im.n_elem != 0 ||
        tx_array->coupling_re.n_elem != 0 || tx_array->coupling_im.n_elem != 0)
    {
        // Calculate abs( cpl )^2 and normalize the row-sum to 1
        dtype *p_rx_cpl = new dtype[n32_rx * n32_rx_ports];
        dtype *p_tx_cpl = new dtype[n32_tx * n32_tx_ports];
        quick_power_mat(n32_rx, n32_rx_ports, p_rx_cpl, true, rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr());
        quick_power_mat(n32_tx, n32_tx_ports, p_tx_cpl, true, tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr());

        // Process coefficients and delays
        if (different_output_size) // Data is stored in internal memory, we can write directly to the output
        {
            for (unsigned j = 0; j < n32_out; j++) // Loop over paths
            {
                unsigned o = j * n32_links; // Slice offset

                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o], &p_coeff_im[o],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             coeff_re->slice_memptr(j), coeff_im->slice_memptr(j),
                                             n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);

                // Apply coupling to delays
                quick_multiply_3_mat(p_rx_cpl, &p_delays[o], p_tx_cpl, delay->slice_memptr(j), n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);
            }
        }
        else // Data has been written to external memory
        {
            // Allocate memory for temporary data
            dtype *tempX = new dtype[n32_ports];
            dtype *tempY = new dtype[n32_ports];

            for (unsigned o = 0; o < n32_links * n32_out; o += n32_links) // Loop over paths
            {
                // Apply coupling to coefficients
                quick_multiply_3_complex_mat(rx_array->coupling_re.memptr(), rx_array->coupling_im.memptr(),
                                             &p_coeff_re[o], &p_coeff_im[o],
                                             tx_array->coupling_re.memptr(), tx_array->coupling_im.memptr(),
                                             tempX, tempY,
                                             n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);

                std::memcpy(&p_coeff_re[o], tempX, n32_ports * sizeof(dtype));
                std::memcpy(&p_coeff_im[o], tempY, n32_ports * sizeof(dtype));

                // Apply coupling to delays
                quick_multiply_3_mat(p_rx_cpl, &p_delays[o], p_tx_cpl, tempX, n32_rx, n32_rx_ports, n32_tx, n32_tx_ports);
                std::memcpy(&p_delays[o], tempX, n32_ports * sizeof(dtype));
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

// Prints the versions of all uses libraries to stdout
void quadriga_lib::print_lib_versions()
{
    arma::arma_version ver;
    std::cout << "quadriga_lib      " << quadriga_lib::quadriga_lib_version() << std::endl;
    std::cout << "Armadillo         " << ver.as_string() << std::endl;
    std::cout << "HDF5              " << quadriga_lib::get_HDF5_version() << std::endl;
    std::cout << "C++ standard      ";

    if (__cplusplus == 202101L) std::cout << "C++23";
    else if (__cplusplus == 202002L) std::cout << "C++20";
    else if (__cplusplus == 201703L) std::cout << "C++17";
    else if (__cplusplus == 201402L) std::cout << "C++14";
    else if (__cplusplus == 201103L) std::cout << "C++11";
    else if (__cplusplus == 199711L) std::cout << "C++98";
    else std::cout << "pre-standard C++." << __cplusplus;
    std::cout << std::endl;
}
