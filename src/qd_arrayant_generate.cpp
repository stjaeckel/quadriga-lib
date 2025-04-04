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
#include "quadriga_tools.hpp"

// Generate : Isotropic radiator, vertical polarization, 1 deg resolution
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_omni()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    ant.name = "omni";
    ant.e_theta_re.ones(181, 361, 1);
    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, 181);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_omni();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_omni();

// Generate : Cross-polarized isotropic radiator, 1 deg resolution
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_xpol()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    ant.name = "xpol";
    ant.e_theta_re.ones(181, 361, 2);
    ant.e_theta_im.zeros(181, 361, 2);
    ant.e_phi_re.ones(181, 361, 2);
    ant.e_phi_im.zeros(181, 361, 2);

    dtype zero = (dtype)0.0;
    dtype *ptr0 = ant.e_phi_re.slice_memptr(0);
    dtype *ptr1 = ant.e_theta_re.slice_memptr(1);
    for (arma::uword i = 0ULL; i < 65341ULL; ++i)
        ptr0[i] = zero, ptr1[i] = zero;

    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, 181);
    ant.element_pos.zeros(3, 2);
    ant.coupling_re.eye(2, 2);
    ant.coupling_im.zeros(2, 2);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_xpol();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_xpol();

// Generate : Short dipole radiating with vertical polarization, 1 deg resolution
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_dipole()
{
    quadriga_lib::arrayant<dtype> ant;

    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);
    ant.name = "dipole";
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, 181);
    ant.e_theta_re.zeros(181, 361, 1);
    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);

    arma::Mat<dtype> tmp = arma::repmat(ant.elevation_grid, 1, 361);
    tmp = arma::cos(dtype(0.999999) * tmp) * dtype(std::sqrt(1.499961));
    std::memcpy(ant.e_theta_re.slice_memptr(0), tmp.memptr(), tmp.n_elem * sizeof(dtype));

    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_dipole();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_dipole();

// Generate : Half-wave dipole radiating with vertical polarization
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_half_wave_dipole()
{
    quadriga_lib::arrayant<dtype> ant;
    dtype pi = dtype(arma::datum::pi), pih = dtype(arma::datum::pi / 2.0);

    ant.name = "half-wave-dipole";
    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pih, pih, 181);
    ant.e_theta_re.zeros(181, 361, 1);
    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);

    arma::Mat<dtype> tmp = dtype(0.999999) * arma::repmat(ant.elevation_grid, 1, 361);
    tmp = arma::cos(pih * arma::sin(tmp)) / arma::cos(tmp);
    tmp = tmp * dtype(1.280968208215292);

    std::memcpy(ant.e_theta_re.slice_memptr(0), tmp.memptr(), tmp.n_elem * sizeof(dtype));

    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_half_wave_dipole();
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_half_wave_dipole();

// Generate : An antenna with a custom gain in elevation and azimuth
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_custom(dtype az_3dB, dtype el_3db, dtype rear_gain_lin)
{
    constexpr dtype zero = dtype(0.0), one = dtype(1.0), half = dtype(0.5),
                    limit = dtype(1e-7), step = dtype(-0.382), limit_inf = dtype(1e38);
    const dtype pi = dtype(arma::datum::pi), deg2rad = dtype(arma::datum::pi / 360.0);

    quadriga_lib::arrayant<dtype> ant;

    ant.azimuth_grid = arma::linspace<arma::Col<dtype>>(-pi, pi, 361);
    ant.elevation_grid = arma::linspace<arma::Col<dtype>>(-pi * half, pi * half, 181);
    arma::Col<dtype> phi_sq = ant.azimuth_grid % ant.azimuth_grid;
    arma::Col<dtype> cos_theta = arma::cos(ant.elevation_grid);
    cos_theta.at(0) = zero, cos_theta.at(180) = zero;
    arma::Col<dtype> az_3dB_rad(1), el_3db_rad(1);
    az_3dB_rad.at(0) = az_3dB * deg2rad;
    el_3db_rad.at(0) = el_3db * deg2rad;

    // Calculate azimuth pattern cut
    dtype a = one, d = half, x = limit_inf, delta = limit_inf;
    arma::Col<dtype> xn(1), C(361), D(181);
    for (unsigned lp = 0; lp < 5000; ++lp)
    {
        dtype an = lp == 0 ? a : a + d;
        delta = lp == 0 ? limit_inf : std::abs(a - an);
        C = rear_gain_lin + (one - rear_gain_lin) * arma::exp(-an * phi_sq);
        quadriga_lib::interp(&C, &ant.azimuth_grid, &az_3dB_rad, &xn);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    C = arma::exp(-a * phi_sq);

    // Calculate elevation pattern cut
    a = one, d = half, x = limit_inf, delta = limit_inf;
    for (unsigned lp = 0; lp < 5000; ++lp)
    {
        dtype an = lp == 0 ? a : a + d;
        delta = lp == 0 ? limit_inf : std::abs(a - an);
        D = arma::pow(cos_theta, an);
        quadriga_lib::interp(&D, &ant.elevation_grid, &el_3db_rad, &xn);
        dtype xm = std::abs(xn.at(0) - half);
        a = xm < x ? an : a;
        d = xm < x ? d : step * d;
        x = xm < x ? xm : x;
        if (delta < limit)
            break;
    }
    D = arma::pow(cos_theta, a);

    // Combined pattern
    ant.e_theta_re.zeros(181, 361, 1);
    dtype *ptr = ant.e_theta_re.memptr();
    for (dtype *col = C.begin(); col != C.end(); ++col)
        for (dtype *row = D.begin(); row != D.end(); ++row)
            *ptr++ = std::sqrt(rear_gain_lin + (one - rear_gain_lin) * *row * *col);

    ant.e_theta_im.zeros(181, 361, 1);
    ant.e_phi_re.zeros(181, 361, 1);
    ant.e_phi_im.zeros(181, 361, 1);
    ant.element_pos.zeros(3, 1);
    ant.coupling_re.ones(1, 1);
    ant.coupling_im.zeros(1, 1);
    ant.name = "custom";

    // Set the data pointers for the quick check.
    ant.check_ptr[0] = ant.e_theta_re.memptr();
    ant.check_ptr[1] = ant.e_theta_im.memptr();
    ant.check_ptr[2] = ant.e_phi_re.memptr();
    ant.check_ptr[3] = ant.e_phi_im.memptr();
    ant.check_ptr[4] = ant.azimuth_grid.memptr();
    ant.check_ptr[5] = ant.elevation_grid.memptr();
    ant.check_ptr[6] = ant.element_pos.memptr();
    ant.check_ptr[7] = ant.coupling_re.memptr();
    ant.check_ptr[8] = ant.coupling_im.memptr();

    // Normalize to Gain
    dtype directivity = ant.calc_directivity_dBi(0);
    directivity = dtype(std::pow(10.0, 0.1 * double(directivity)));
    dtype p_max = ant.e_theta_re.max();
    p_max *= p_max;
    ant.e_theta_re *= std::sqrt(directivity / p_max);

    return ant;
}
template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_custom(float az_3dB, float el_3db, float rear_gain_lin);
template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_custom(double az_3dB, double el_3db, double rear_gain_lin);

// Generate : Antenna model for the 3GPP-NR channel model
template <typename dtype>
quadriga_lib::arrayant<dtype> quadriga_lib::generate_arrayant_3GPP(unsigned long long M, unsigned long long N, dtype center_freq,
                                                                   unsigned pol, dtype tilt, dtype spacing,
                                                                   unsigned long long Mg, unsigned long long Ng, dtype dgv, dtype dgh,
                                                                   const arrayant<dtype> *pattern)
{
    double pi = arma::datum::pi, rad2deg = 180.0 / pi, deg2rad = pi / 180.0;
    double wavelength = 299792458.0 / double(center_freq);
    constexpr dtype zero = dtype(0.0);

    quadriga_lib::arrayant<dtype> ant = pattern == nullptr ? quadriga_lib::generate_arrayant_omni<dtype>() : pattern->copy();

    if (pattern != nullptr)
    {
        std::string error_message = ant.validate(); // Deep check
        if (error_message.length() != 0)
            throw std::invalid_argument(error_message.c_str());
    }

    ant.center_frequency = center_freq;
    unsigned long long n_az = ant.n_azimuth(), n_el = ant.n_elevation();

    if (pattern == nullptr) // Generate 3GPP default radiation pattern
    {
        // Single antenna element vertical radiation pattern cut in dB
        arma::Col<dtype> Y = ant.elevation_grid;
        for (dtype *py = Y.begin(); py < Y.end(); ++py)
        {
            double y = double(*py) * rad2deg / 65.0;
            y = 12.0 * y * y;
            *py = y > 30.0 ? dtype(30.0) : dtype(y);
        }

        // Full pattern (normalized to 8 dBi gain using factor 2.51..)
        dtype *ptr = ant.e_theta_re.memptr(), *py = Y.memptr(), *px = ant.azimuth_grid.memptr();
        for (auto ia = 0ULL; ia < n_az; ++ia)
        {
            double x = double(*px++) * rad2deg / 65.0;
            x = 12.0 * x * x;
            x = x > 30.0 ? 30.0 : x;

            for (auto ie = 0ULL; ie < n_el; ++ie)
            {
                double z = double(py[ie]) + x;
                z = z > 30.0 ? -30.0 : -z;
                *ptr++ = dtype(2.511886431509580 * std::sqrt(std::pow(10.0, 0.1 * z))); // 8dBi gain
            }
        }
        Y.reset();

        // Adjust polarization
        if (pol == 2 || pol == 5)
        {
            ant.copy_element(0, 1);
            ant.rotate_pattern(90.0, zero, zero, 2, 1);
        }
        else if (pol == 3 || pol == 6)
        {
            ant.copy_element(0, 1);
            ant.rotate_pattern(dtype(45.0), zero, zero, 2, 0);
            ant.rotate_pattern(dtype(-45.0), zero, zero, 2, 1);
        }
    }

    // Duplicate the existing elements in z-direction (vertical stacking)
    unsigned long long n_elements = ant.n_elements();
    if (M > 1ULL)
        for (unsigned long long source = n_elements; source > 0ULL; source--)
        {
            unsigned long long i_start = n_elements + source - 1ULL;
            unsigned long long i_end = M * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

    // Calculate the element z-position
    arma::Col<dtype> z_position(M);
    if (M > 1ULL)
    {
        z_position = arma::linspace<arma::Col<dtype>>(zero, dtype(M - 1ULL) * spacing * dtype(wavelength), M);
        z_position = z_position - arma::mean(z_position);

        for (auto m = 0ULL; m < M; ++m)
            for (auto n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(2ULL, m * n_elements + n) = z_position.at(m);
    }

    // Apply element coupling for polarization indicators 4, 5, and 6
    if (pol > 3 && M > 1ULL)
    {
        double tmp = 2.0 * pi * std::sin(double(tilt) * deg2rad) / wavelength;
        arma::Col<dtype> cpl_re = z_position * dtype(tmp);
        tmp = 1.0 / std::sqrt(double(M));
        arma::Col<dtype> cpl_im = arma::sin(cpl_re) * dtype(tmp);
        cpl_re = arma::cos(cpl_re) * dtype(tmp);

        ant.coupling_re.zeros(n_elements * M, n_elements);
        ant.coupling_im.zeros(n_elements * M, n_elements);

        for (auto m = 0ULL; m < M; ++m)
            for (auto n = 0ULL; n < n_elements; ++n)
            {
                ant.coupling_re.at(m * n_elements + n, n) = cpl_re.at(m);
                ant.coupling_im.at(m * n_elements + n, n) = cpl_im.at(m);
            }

        ant.combine_pattern();
        M = 1ULL;
    }

    // Duplicate the existing elements in y-direction (horizontal stacking)
    n_elements = ant.n_elements();
    if (N > 1ULL)
    {
        for (unsigned long long source = n_elements; source > 0ULL; source--)
        {
            unsigned long long i_start = n_elements + source - 1ULL;
            unsigned long long i_end = N * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> y_position = arma::linspace<arma::Col<dtype>>(zero, dtype(N - 1ULL) * spacing * dtype(wavelength), N);
        y_position = y_position - arma::mean(y_position);

        for (auto m = 0ULL; m < N; ++m)
            for (auto n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(1ULL, m * n_elements + n) = y_position.at(m);
    }

    // Duplicate panels in z-direction (vertical panel stacking)
    n_elements = ant.n_elements();
    if (Mg > 1ULL)
    {
        for (unsigned long long source = n_elements; source > 0ULL; source--)
        {
            unsigned long long i_start = n_elements + source - 1ULL;
            unsigned long long i_end = Mg * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> zg_position = arma::linspace<arma::Col<dtype>>(zero, dtype(Mg - 1ULL) * dgv * dtype(wavelength), Mg);
        zg_position = zg_position - arma::mean(zg_position);

        for (auto mg = 0ULL; mg < Mg; ++mg)
            for (auto n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(2ULL, mg * n_elements + n) += zg_position.at(mg);
    }

    // Duplicate panels in y-direction (horizontal panel stacking)
    n_elements = ant.n_elements();
    if (Ng > 1)
    {
        for (unsigned long long source = n_elements; source > 0; source--)
        {
            unsigned long long i_start = n_elements + source - 1ULL;
            unsigned long long i_end = Ng * n_elements - 1ULL;
            arma::uvec destination = arma::regspace<arma::uvec>(i_start, n_elements, i_end);
            ant.copy_element(source - 1ULL, destination);
        }

        arma::Col<dtype> yg_position = arma::linspace<arma::Col<dtype>>(zero, dtype(Ng - 1ULL) * dgh * dtype(wavelength), Ng);
        yg_position = yg_position - arma::mean(yg_position);

        for (auto mg = 0ULL; mg < Ng; ++mg)
            for (auto n = 0ULL; n < n_elements; ++n)
                ant.element_pos.at(1ULL, mg * n_elements + n) += yg_position.at(mg);
    }

    ant.name = "3gpp";
    return ant;
}

template quadriga_lib::arrayant<float> quadriga_lib::generate_arrayant_3GPP(unsigned long long M, unsigned long long N, float center_freq,
                                                                            unsigned pol, float tilt, float spacing,
                                                                            unsigned long long Mg, unsigned long long Ng, float dgv, float dgh,
                                                                            const quadriga_lib::arrayant<float> *pattern);

template quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_3GPP(unsigned long long M, unsigned long long N, double center_freq,
                                                                             unsigned pol, double tilt, double spacing,
                                                                             unsigned long long Mg, unsigned long long Ng, double dgv, double dgh,
                                                                             const quadriga_lib::arrayant<double> *pattern);
