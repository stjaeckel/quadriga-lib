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

// Calculate channels for Intelligent Reflective Surfaces (IRS)
template <typename dtype>
std::vector<bool> quadriga_lib::get_channels_irs(const arrayant<dtype> *tx_array, const arrayant<dtype> *rx_array, const arrayant<dtype> *irs_array,
                                                 dtype Tx, dtype Ty, dtype Tz, dtype Tb, dtype Tt, dtype Th,
                                                 dtype Rx, dtype Ry, dtype Rz, dtype Rb, dtype Rt, dtype Rh,
                                                 dtype Ix, dtype Iy, dtype Iz, dtype Ib, dtype It, dtype Ih,
                                                 const arma::Mat<dtype> *fbs_pos_1, const arma::Mat<dtype> *lbs_pos_1,
                                                 const arma::Col<dtype> *path_gain_1, const arma::Col<dtype> *path_length_1, const arma::Mat<dtype> *M_1,
                                                 const arma::Mat<dtype> *fbs_pos_2, const arma::Mat<dtype> *lbs_pos_2,
                                                 const arma::Col<dtype> *path_gain_2, const arma::Col<dtype> *path_length_2, const arma::Mat<dtype> *M_2,
                                                 arma::Cube<dtype> *coeff_re, arma::Cube<dtype> *coeff_im, arma::Cube<dtype> *delay,
                                                 arma::uword i_irs, dtype threshold_dB, dtype center_frequency, bool use_absolute_delays,
                                                 arma::Cube<dtype> *aod, arma::Cube<dtype> *eod, arma::Cube<dtype> *aoa, arma::Cube<dtype> *eoa,
                                                 const arrayant<dtype> *irs_array_2)
{

    // Check if IRS array is valid
    if (irs_array == nullptr)
        throw std::invalid_argument("IRS array cannot be NULL");

    std::string error_message = irs_array->is_valid();
    if (error_message.length() != 0)
    {
        error_message = "IRS array: " + error_message;
        throw std::invalid_argument(error_message.c_str());
    }

    if (irs_array->coupling_re.n_cols == 0ULL)
        throw std::invalid_argument("IRS must have a Coupling matrix containing the IRS weights (phase shifts) for each element.");

    if (i_irs >= irs_array->coupling_re.n_cols)
        throw std::invalid_argument("IRS codebook index out of bound.");

    // Do we need to calculate the angles?
    bool calc_angles = aod != nullptr || eod != nullptr || aoa != nullptr || eoa != nullptr;

    // Construst IRS array with only the desired weight vector
    arma::uword n_azimuth_irs = irs_array->e_theta_re.n_cols;
    arma::uword n_elevation_irs = irs_array->e_theta_re.n_rows;
    arma::uword n_elements_irs = irs_array->e_theta_re.n_slices;

    quadriga_lib::arrayant<dtype> irs;
    irs.e_theta_re = arma::Cube<dtype>(const_cast<dtype *>(irs_array->e_theta_re.memptr()), n_elevation_irs, n_azimuth_irs, n_elements_irs, false, true);
    irs.e_theta_im = arma::Cube<dtype>(const_cast<dtype *>(irs_array->e_theta_im.memptr()), n_elevation_irs, n_azimuth_irs, n_elements_irs, false, true);
    irs.e_phi_re = arma::Cube<dtype>(const_cast<dtype *>(irs_array->e_phi_re.memptr()), n_elevation_irs, n_azimuth_irs, n_elements_irs, false, true);
    irs.e_phi_im = arma::Cube<dtype>(const_cast<dtype *>(irs_array->e_phi_im.memptr()), n_elevation_irs, n_azimuth_irs, n_elements_irs, false, true);
    irs.azimuth_grid = arma::Col<dtype>(const_cast<dtype *>(irs_array->azimuth_grid.memptr()), n_azimuth_irs, false, true);
    irs.elevation_grid = arma::Col<dtype>(const_cast<dtype *>(irs_array->elevation_grid.memptr()), n_elevation_irs, false, true);

    if (irs_array->element_pos.n_cols != n_elements_irs)
        irs.element_pos.zeros(3ULL, n_elements_irs);
    else
        irs.element_pos = arma::Mat<dtype>(const_cast<dtype *>(irs_array->element_pos.memptr()), 3ULL, n_elements_irs, false, true);

    irs.coupling_re = arma::Mat<dtype>(const_cast<dtype *>(irs_array->coupling_re.colptr(i_irs)), n_elements_irs, 1ULL, false, true);
    if (irs_array->coupling_im.n_rows != n_elements_irs)
        irs.coupling_im.zeros(n_elements_irs, 1ULL);
    else
        irs.coupling_im = arma::Mat<dtype>(const_cast<dtype *>(irs_array->coupling_im.colptr(i_irs)), n_elements_irs, 1ULL, false, true);

    if (center_frequency > dtype(1.0e-5))
        irs.center_frequency = center_frequency;
    irs.read_only = true;

    // Calculate the channel coefficients of the first segment from the TX to the IRS
    arma::Cube<dtype> coeff_re_1, coeff_im_1, delay_1, aod_1, eod_1;
    if (calc_angles)
    {
        quadriga_lib::get_channels_spherical<dtype>(tx_array, &irs, Tx, Ty, Tz, Tb, Tt, Th, Ix, Iy, Iz, Ib, It, Ih, fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1,
                                                    &coeff_re_1, &coeff_im_1, &delay_1, center_frequency, true, false, &aod_1, &eod_1);
    }
    else
    {
        quadriga_lib::get_channels_spherical<dtype>(tx_array, &irs, Tx, Ty, Tz, Tb, Tt, Th, Ix, Iy, Iz, Ib, It, Ih, fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1,
                                                    &coeff_re_1, &coeff_im_1, &delay_1, center_frequency, true, false);
    }

    // Update the IRS array for the second segment
    if (irs_array_2 != nullptr)
    {
        std::string error_message = irs_array_2->is_valid();
        if (error_message.length() != 0)
        {
            error_message = "IRS array 2: " + error_message;
            throw std::invalid_argument(error_message.c_str());
        }

        if (irs_array_2->e_theta_re.n_slices != n_elements_irs)
            throw std::invalid_argument("IRS 2 must have the same number of elements as IRS 1");

        if (irs_array_2->coupling_re.n_cols != irs_array->coupling_re.n_cols)
            throw std::invalid_argument("IRS 2 must have a Coupling matrix of the same size as of IRS 1.");

        n_azimuth_irs = irs_array_2->e_theta_re.n_cols;
        n_elevation_irs = irs_array_2->e_theta_re.n_rows;

        irs.e_theta_re = arma::Cube<dtype>(const_cast<dtype *>(irs_array_2->e_theta_re.memptr()), n_elevation_irs, n_azimuth_irs, n_elements_irs, false, true);
        irs.e_theta_im = arma::Cube<dtype>(const_cast<dtype *>(irs_array_2->e_theta_im.memptr()), n_elevation_irs, n_azimuth_irs, n_elements_irs, false, true);
        irs.e_phi_re = arma::Cube<dtype>(const_cast<dtype *>(irs_array_2->e_phi_re.memptr()), n_elevation_irs, n_azimuth_irs, n_elements_irs, false, true);
        irs.e_phi_im = arma::Cube<dtype>(const_cast<dtype *>(irs_array_2->e_phi_im.memptr()), n_elevation_irs, n_azimuth_irs, n_elements_irs, false, true);
        irs.azimuth_grid = arma::Col<dtype>(const_cast<dtype *>(irs_array_2->azimuth_grid.memptr()), n_azimuth_irs, false, true);
        irs.elevation_grid = arma::Col<dtype>(const_cast<dtype *>(irs_array_2->elevation_grid.memptr()), n_elevation_irs, false, true);

        if (irs_array_2->element_pos.n_cols != n_elements_irs)
            irs.element_pos.zeros(3ULL, n_elements_irs);
        else
            irs.element_pos = arma::Mat<dtype>(const_cast<dtype *>(irs_array_2->element_pos.memptr()), 3ULL, n_elements_irs, false, true);

        irs.coupling_re = arma::Mat<dtype>(const_cast<dtype *>(irs_array_2->coupling_re.colptr(i_irs)), n_elements_irs, 1ULL, false, true);
        if (irs_array_2->coupling_im.n_rows != n_elements_irs)
            irs.coupling_im.zeros(n_elements_irs, 1ULL);
        else
            irs.coupling_im = arma::Mat<dtype>(const_cast<dtype *>(irs_array_2->coupling_im.colptr(i_irs)), n_elements_irs, 1ULL, false, true);
    }

    // Calculate channel coefficients of the second segment from the IRS to the RX
    arma::Cube<dtype> coeff_re_2, coeff_im_2, delay_2, aoa_2, eoa_2;
    if (calc_angles)
    {
        quadriga_lib::get_channels_spherical<dtype>(&irs, rx_array, Ix, Iy, Iz, Ib, It, Ih, Rx, Ry, Rz, Rb, Rt, Rh, fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2,
                                                    &coeff_re_2, &coeff_im_2, &delay_2, center_frequency, true, false, nullptr, nullptr, &aoa_2, &eoa_2);
    }
    else
    {
        quadriga_lib::get_channels_spherical<dtype>(&irs, rx_array, Ix, Iy, Iz, Ib, It, Ih, Rx, Ry, Rz, Rb, Rt, Rh, fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2,
                                                    &coeff_re_2, &coeff_im_2, &delay_2, center_frequency, true, false);
    }

    // Calculate the LOS delay between TX and RX
    dtype los_delay = dtype(0.0);
    if (!use_absolute_delays)
    {
        dtype x = Rx - Tx, y = Ry - Ty, z = Rz - Tz;
        los_delay = std::sqrt(x * x + y * y + z * z);
        los_delay *= dtype(3.335640951981520e-09); // 1/C
    }

    // Combine the two channel segments
    arma::uword n_path_1 = coeff_re_1.n_slices;
    arma::uword n_path_2 = coeff_re_2.n_slices;
    arma::uword n_path = n_path_1 * n_path_2;
    arma::uword n_tx = coeff_re_1.n_cols;
    arma::uword n_rx = coeff_re_2.n_rows;

    arma::Cube<dtype> coeff_re_local(n_rx, n_tx, n_path, arma::fill::none);
    arma::Cube<dtype> coeff_im_local(n_rx, n_tx, n_path, arma::fill::none);
    arma::Cube<dtype> delay_local(n_rx, n_tx, n_path, arma::fill::none);

    arma::Cube<dtype> aod_local, eod_local, aoa_local, eoa_local;
    if (calc_angles)
    {
        aod_local.set_size(n_rx, n_tx, n_path);
        eod_local.set_size(n_rx, n_tx, n_path);
        aoa_local.set_size(n_rx, n_tx, n_path);
        eoa_local.set_size(n_rx, n_tx, n_path);
    }

    dtype threshold = (dtype)std::pow<double>(10.0, 0.1 * (double)threshold_dB);
    std::vector<bool> keep_path(n_path, false);

    arma::uword i_path = 0ULL, j_path = 0ULL;
    for (arma::uword i_path_1 = 0ULL; i_path_1 < n_path_1; ++i_path_1)
    {
        dtype *c1r = coeff_re_1.slice_memptr(i_path_1);
        dtype *c1i = coeff_im_1.slice_memptr(i_path_1);
        dtype *d1 = delay_1.slice_memptr(i_path_1);
        dtype *p_aod_1 = calc_angles ? aod_1.slice_memptr(i_path_1) : nullptr;
        dtype *p_eod_1 = calc_angles ? eod_1.slice_memptr(i_path_1) : nullptr;

        for (arma::uword i_path_2 = 0ULL; i_path_2 < n_path_2; ++i_path_2)
        {
            dtype *c2r = coeff_re_2.slice_memptr(i_path_2);
            dtype *c2i = coeff_im_2.slice_memptr(i_path_2);
            dtype *d2 = delay_2.slice_memptr(i_path_2);
            dtype *p_aoa_2 = calc_angles ? aoa_2.slice_memptr(i_path_2) : nullptr;
            dtype *p_eoa_2 = calc_angles ? eoa_2.slice_memptr(i_path_2) : nullptr;

            dtype *cr = coeff_re_local.slice_memptr(i_path);
            dtype *ci = coeff_im_local.slice_memptr(i_path);
            dtype *dl = delay_local.slice_memptr(i_path);
            dtype *p_aod = calc_angles ? aod_local.slice_memptr(i_path) : nullptr;
            dtype *p_eod = calc_angles ? eod_local.slice_memptr(i_path) : nullptr;
            dtype *p_aoa = calc_angles ? aoa_local.slice_memptr(i_path) : nullptr;
            dtype *p_eoa = calc_angles ? eoa_local.slice_memptr(i_path) : nullptr;

            dtype pow = dtype(0.0);
            for (arma::uword i_tx = 0ULL; i_tx < n_tx; ++i_tx)
            {
                dtype a = c1r[i_tx], b = c1i[i_tx], dd1 = d1[i_tx];
                for (arma::uword i_rx = 0ULL; i_rx < n_rx; ++i_rx)
                {
                    dtype c = c2r[i_rx], d = c2i[i_rx];
                    dtype re = a * c - b * d;
                    dtype im = a * d + b * c;

                    arma::uword i = i_tx * n_rx + i_rx;
                    cr[i] = re, ci[i] = im;
                    dl[i] = dd1 + d2[i_rx] - los_delay;

                    re = re * re + im * im;
                    pow = (re > pow) ? re : pow;

                    if (calc_angles)
                    {
                        p_aod[i] = p_aod_1[i_tx];
                        p_eod[i] = p_eod_1[i_tx];
                        p_aoa[i] = p_aoa_2[i_rx];
                        p_eoa[i] = p_eoa_2[i_rx];
                    }
                }
            }
            i_path = (pow > threshold) ? i_path + 1ULL : i_path;
            keep_path[j_path] = pow > threshold;
            ++j_path;
        }
    }

    // Write to output
    coeff_re->set_size(n_rx, n_tx, i_path);
    coeff_im->set_size(n_rx, n_tx, i_path);
    delay->set_size(n_rx, n_tx, i_path);

    arma::uword n_bytes = n_rx * n_tx * i_path * sizeof(dtype);
    std::memcpy(coeff_re->memptr(), coeff_re_local.memptr(), n_bytes);
    std::memcpy(coeff_im->memptr(), coeff_im_local.memptr(), n_bytes);
    std::memcpy(delay->memptr(), delay_local.memptr(), n_bytes);

    if (aod != nullptr)
    {
        aod->set_size(n_rx, n_tx, i_path);
        std::memcpy(aod->memptr(), aod_local.memptr(), n_bytes);
    }

    if (eod != nullptr)
    {
        eod->set_size(n_rx, n_tx, i_path);
        std::memcpy(eod->memptr(), eod_local.memptr(), n_bytes);
    }

    if (aoa != nullptr)
    {
        aoa->set_size(n_rx, n_tx, i_path);
        std::memcpy(aoa->memptr(), aoa_local.memptr(), n_bytes);
    }

    if (eoa != nullptr)
    {
        eoa->set_size(n_rx, n_tx, i_path);
        std::memcpy(eoa->memptr(), eoa_local.memptr(), n_bytes);
    }

    return keep_path;
}

template std::vector<bool> quadriga_lib::get_channels_irs(const arrayant<float> *tx_array, const arrayant<float> *rx_array, const arrayant<float> *irs_array,
                                                          float Tx, float Ty, float Tz, float Tb, float Tt, float Th,
                                                          float Rx, float Ry, float Rz, float Rb, float Rt, float Rh,
                                                          float Ix, float Iy, float Iz, float Ib, float It, float Ih,
                                                          const arma::Mat<float> *fbs_pos_1, const arma::Mat<float> *lbs_pos_1,
                                                          const arma::Col<float> *path_gain_1, const arma::Col<float> *path_length_1, const arma::Mat<float> *M_1,
                                                          const arma::Mat<float> *fbs_pos_2, const arma::Mat<float> *lbs_pos_2,
                                                          const arma::Col<float> *path_gain_2, const arma::Col<float> *path_length_2, const arma::Mat<float> *M_2,
                                                          arma::Cube<float> *coeff_re, arma::Cube<float> *coeff_im, arma::Cube<float> *delay,
                                                          arma::uword i_irs, float threshold_dB, float center_frequency, bool use_absolute_delays,
                                                          arma::Cube<float> *aod, arma::Cube<float> *eod, arma::Cube<float> *aoa, arma::Cube<float> *eoa,
                                                          const arrayant<float> *irs_array_2);

template std::vector<bool> quadriga_lib::get_channels_irs(const arrayant<double> *tx_array, const arrayant<double> *rx_array, const arrayant<double> *irs_array,
                                                          double Tx, double Ty, double Tz, double Tb, double Tt, double Th,
                                                          double Rx, double Ry, double Rz, double Rb, double Rt, double Rh,
                                                          double Ix, double Iy, double Iz, double Ib, double It, double Ih,
                                                          const arma::Mat<double> *fbs_pos_1, const arma::Mat<double> *lbs_pos_1,
                                                          const arma::Col<double> *path_gain_1, const arma::Col<double> *path_length_1, const arma::Mat<double> *M_1,
                                                          const arma::Mat<double> *fbs_pos_2, const arma::Mat<double> *lbs_pos_2,
                                                          const arma::Col<double> *path_gain_2, const arma::Col<double> *path_length_2, const arma::Mat<double> *M_2,
                                                          arma::Cube<double> *coeff_re, arma::Cube<double> *coeff_im, arma::Cube<double> *delay,
                                                          arma::uword i_irs, double threshold_dB, double center_frequency, bool use_absolute_delays,
                                                          arma::Cube<double> *aod, arma::Cube<double> *eod, arma::Cube<double> *aoa, arma::Cube<double> *eoa,
                                                          const arrayant<double> *irs_array_2);