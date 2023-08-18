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

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <iostream>
#include <string>

TEST_CASE("Get Channels Spherical - Minimal test")
{
    float pi = 3.141592653589793f;

    // Generate test antenna
    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    ant.copy_element(0, 1);
    ant.element_pos(1, 0) = 1.0;
    ant.element_pos(1, 1) = -1.0;
    ant.e_theta_re.slice(1) *= 2;

    arma::fmat fbs_pos(3, 2); //     FBS @ 11 m height
    fbs_pos(0, 0) = 10.0;     //      |
    fbs_pos(2, 0) = 1.0;      //     10 m
    fbs_pos(1, 1) = 10.0;     //      |           FBS LOS @ x = 10 m
    fbs_pos(2, 1) = 11.0;     //     TX --------------d = 20 m------------------ RX

    arma::fvec path_gain(2), path_length(2);
    path_gain(0) = 1;
    path_gain(1) = 0.25;

    arma::fmat M(8, 2);
    M(0, 0) = 1.0;
    M(0, 1) = 1.0;
    M(6, 0) = -1.0;
    M(6, 1) = -1.0;

    // Shortest path length is automatically obtained from FBS / LBS
    arma::fcube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<float>(&ant, &ant,
                                                0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
                                                &coeff_re, &coeff_im, &delay, 2997924580.0, true, false,
                                                &aod, &eod, &aoa, &eoa);

    aod *= 180.0f / pi;
    eod *= 180.0f / pi;
    aoa *= 180.0f / pi;
    eoa *= 180.0f / pi;

    // Check departure azimuth angles
    float alpha = std::atan(2.0 / 20.0) * 180.0f / pi, beta = 90.0f;
    CHECK(std::abs(aod.at(0, 0, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(aod.at(1, 0, 0) + alpha) < 1.0e-6f);
    CHECK(std::abs(aod.at(0, 1, 0) - alpha) < 1.0e-6f);
    CHECK(std::abs(aod.at(1, 1, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(aod.at(0, 0, 1) - beta) < 1.0e-6f);
    CHECK(std::abs(aod.at(1, 0, 1) - beta) < 1.0e-6f);
    CHECK(std::abs(aod.at(0, 1, 1) - beta) < 1.0e-6f);
    CHECK(std::abs(aod.at(1, 1, 1) - beta) < 1.0e-6f);

    // Check arrival azimuth angles
    alpha = 180.0 - alpha;
    CHECK(std::abs(std::abs(std::cos(aoa.at(0, 0, 0) * pi / 180.0)) - 1.0) < 1.0e-4f);
    CHECK(std::abs(aoa.at(1, 0, 0) - alpha) < 1.0e-4f);
    CHECK(std::abs(aoa.at(0, 1, 0) + alpha) < 1.0e-4f);
    CHECK(std::abs(std::abs(std::cos(aoa.at(1, 1, 0) * pi / 180.0)) - 1.0) < 1.0e-4f);
    alpha = 180.0 - std::atan(9.0 / 20.0) * 180.0f / pi, beta = 180.0 - std::atan(11.0 / 20.0) * 180.0f / pi;
    CHECK(std::abs(aoa.at(0, 0, 1) - alpha) < 1.0e-4f);
    CHECK(std::abs(aoa.at(1, 0, 1) - beta) < 1.0e-4f);
    CHECK(std::abs(aoa.at(0, 1, 1) - alpha) < 1.0e-4f);
    CHECK(std::abs(aoa.at(1, 1, 1) - beta) < 1.0e-4f);

    // Check departure elevation angles
    alpha = std::atan(10.0 / 11.0) * 180.0f / pi, beta = std::atan(10.0 / 9.0) * 180.0f / pi;
    CHECK(std::abs(eod.at(0, 0, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eod.at(1, 0, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eod.at(0, 1, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eod.at(1, 1, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eod.at(0, 0, 1) - beta) < 1.0e-6f);
    CHECK(std::abs(eod.at(1, 0, 1) - beta) < 1.0e-6f);
    CHECK(std::abs(eod.at(0, 1, 1) - alpha) < 1.0e-6f);
    CHECK(std::abs(eod.at(1, 1, 1) - alpha) < 1.0e-6f);

    // Check arrival elevation angles
    alpha = std::atan(10.0 / std::sqrt(9.0 * 9.0 + 20.0 * 20.0)) * 180.0f / pi;
    beta = std::atan(10.0 / std::sqrt(11.0 * 11.0 + 20.0 * 20.0)) * 180.0f / pi;
    CHECK(std::abs(eoa.at(0, 0, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eoa.at(1, 0, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eoa.at(0, 1, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eoa.at(1, 1, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eoa.at(0, 0, 1) - alpha) < 1.0e-4f);
    CHECK(std::abs(eoa.at(1, 0, 1) - beta) < 1.0e-4f);
    CHECK(std::abs(eoa.at(0, 1, 1) - alpha) < 1.0e-4f);
    CHECK(std::abs(eoa.at(1, 1, 1) - beta) < 1.0e-4f);

    // Check amplitude
    arma::fcube amp = arma::sqrt(coeff_re % coeff_re + coeff_im % coeff_im);
    CHECK(std::abs(amp.at(0, 0, 0) - 1.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(1, 0, 0) - 2.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(0, 1, 0) - 2.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(1, 1, 0) - 4.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(0, 0, 1) - 0.5f) < 1.0e-4f);
    CHECK(std::abs(amp.at(1, 0, 1) - 1.0f) < 1.0e-4f);
    CHECK(std::abs(amp.at(0, 1, 1) - 1.0f) < 1.0e-4f);
    CHECK(std::abs(amp.at(1, 1, 1) - 2.0f) < 1.0e-4f);

    // Check delays
    float C = 299792458.0;
    float d0 = 20.0, d1 = std::sqrt(20.0 * 20.0 + 2.0 * 2.0);
    float e0 = (std::sqrt(9.0 * 9.0 + 10.0 * 10.0) + std::sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
    float e1 = (std::sqrt(9.0 * 9.0 + 10.0 * 10.0) + std::sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));
    float e2 = (std::sqrt(11.0 * 11.0 + 10.0 * 10.0) + std::sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
    float e3 = (std::sqrt(11.0 * 11.0 + 10.0 * 10.0) + std::sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));
    CHECK(std::abs(delay.at(0, 0, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(1, 0, 0) - d1 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 1, 0) - d1 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(1, 1, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 0, 1) - e0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(1, 0, 1) - e1 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 1, 1) - e2 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(1, 1, 1) - e3 / C) < 1.0e-13f);

    // Check phases
    arma::fcube phase = arma::atan2(-coeff_im / amp, coeff_re / amp);
    for (float *val = phase.begin(); val < phase.end(); val++)
        *val = *val < 0.0f ? *val + 2.0f * pi : *val;

    CHECK(std::abs(phase.at(0, 0, 0) - 0.0f) < 1.0e-4f);
    CHECK(std::abs(phase.at(1, 0, 0) - 20.0f * pi * std::fmod(d1, 0.1f)) < 1.0e-3f);
    CHECK(std::abs(phase.at(0, 1, 0) - 20.0f * pi * std::fmod(d1, 0.1f)) < 1.0e-3f);
    CHECK(std::abs(phase.at(1, 1, 0) - 0.0f) < 1.0e-4f);
    CHECK(std::abs(phase.at(0, 0, 1) - 20.0f * pi * std::fmod(e0, 0.1f)) < 1.0e-3f);
    CHECK(std::abs(phase.at(1, 0, 1) - 20.0f * pi * std::fmod(e1, 0.1f)) < 1.0e-3f);
    CHECK(std::abs(phase.at(0, 1, 1) - 20.0f * pi * std::fmod(e2, 0.1f)) < 1.0e-3f);
    CHECK(std::abs(phase.at(1, 1, 1) - 20.0f * pi * std::fmod(e3, 0.1f)) < 1.0e-3f);
}

TEST_CASE("Get Channels Spherical - Tx rotation")
{
    double pi = 3.141592653589793f;

    // Generate test antenna
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    ant.copy_element(0, 1);
    ant.element_pos(1, 0) = 15;
    ant.element_pos(1, 1) = -15;
    ant.e_theta_re.slice(1) *= 2;

    auto probe = quadriga_lib::generate_arrayant_xpol<double>();

    arma::mat fbs_pos(3, 1);
    fbs_pos(0, 0) = 10.0;
    fbs_pos(2, 0) = 1.0;

    arma::vec path_gain(1), path_length(1);
    path_gain(0) = 1;

    arma::mat M(8, 1);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    // Shortest path length is automatically obtained from FBS / LBS
    arma::cube coeff_re, coeff_im, delay;

    quadriga_lib::get_channels_spherical<double>(&ant, &probe,
                                                 0.0, 0.0, 1.0, -pi / 2.0, 0.0, 0.0,
                                                 20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                 &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
                                                 &coeff_re, &coeff_im, &delay, 2997924580.0, false);

    double C = 299792458.0;
    double d0 = std::sqrt(20.0 * 20.0 + 15.0 * 15.0) - 20.0; // relative to LOS-Delay
    CHECK(std::abs(delay.at(0, 0, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(1, 0, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 1, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(1, 1, 0) - d0 / C) < 1.0e-13f);

    CHECK(std::abs(coeff_re.at(0, 0, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(1, 0, 0) - 1.0) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(0, 1, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(1, 1, 0) - 2.0) < 1.0e-6f);

    CHECK(std::abs(coeff_im.at(0, 0, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(1, 0, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(0, 1, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(1, 1, 0) - 0.0) < 1.0e-6f);

    quadriga_lib::get_channels_spherical<double>(&ant, &probe,
                                                 0.0, 0.0, 1.0, -pi / 2.0, pi / 2.0, 0.0,
                                                 20.0, 0.0, 1.0, pi / 2.0, 0.0, 0.0,
                                                 &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
                                                 &coeff_re, &coeff_im, &delay, 2997924580.0, false);

    CHECK(std::abs(delay.at(0, 0, 0) + 15.0 / C) < 1.0e-16f);
    CHECK(std::abs(delay.at(1, 0, 0) + 15.0 / C) < 1.0e-16f);
    CHECK(std::abs(delay.at(0, 1, 0) - 15.0 / C) < 1.0e-16f);
    CHECK(std::abs(delay.at(1, 1, 0) - 15.0 / C) < 1.0e-16f);

    CHECK(std::abs(coeff_re.at(0, 0, 0) + 1.0) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(1, 0, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(0, 1, 0) + 2.0) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(1, 1, 0) - 0.0) < 1.0e-6f);

    CHECK(std::abs(coeff_im.at(0, 0, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(1, 0, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(0, 1, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(1, 1, 0) - 0.0) < 1.0e-6f);

    quadriga_lib::get_channels_spherical<double>(&ant, &probe,
                                                 0.0, 0.0, 1.0, 0.0, 0.0, pi / 2.0,
                                                 20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                 &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
                                                 &coeff_re, &coeff_im, &delay, 2997924580.0, false);

    CHECK(std::abs(delay.at(0, 0, 0) - 15.0 / C) < 1.0e-16f);
    CHECK(std::abs(delay.at(1, 0, 0) - 15.0 / C) < 1.0e-16f);
    CHECK(std::abs(delay.at(0, 1, 0) + 15.0 / C) < 1.0e-16f);
    CHECK(std::abs(delay.at(1, 1, 0) + 15.0 / C) < 1.0e-16f);

    CHECK(std::abs(coeff_re.at(0, 0, 0) - 1.0) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(1, 0, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(0, 1, 0) - 2.0) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(1, 1, 0) - 0.0) < 1.0e-6f);

    CHECK(std::abs(coeff_im.at(0, 0, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(1, 0, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(0, 1, 0) - 0.0) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(1, 1, 0) - 0.0) < 1.0e-6f);
}

TEST_CASE("Get Channels Spherical - Fake LOS")
{

    float pi = 3.141592653589793f;
    auto ant = quadriga_lib::generate_arrayant_omni<float>();

    // LOS-Path and LBS / FBS are reversed

    arma::fmat fbs_pos(3, 2); //     FBS @ 11 m height
    fbs_pos(0, 1) = 10.0;     //      |
    fbs_pos(2, 1) = 1.0;      //     10 m
    fbs_pos(1, 0) = 10.0;     //      |           FBS LOS @ x = 10 m
    fbs_pos(2, 0) = 11.0;     //     TX --------------d = 20 m------------------ RX

    arma::fvec path_gain(2), path_length(2);
    path_gain(1) = 1;
    path_gain(0) = 0.25;

    arma::fmat M(8, 2);
    M(0, 0) = 1.0;
    M(0, 1) = 1.0;
    M(6, 0) = -1.0;
    M(6, 1) = -1.0;

    // Shortest path length is automatically obtained from FBS / LBS
    arma::fcube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<float>(&ant, &ant,
                                                0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
                                                &coeff_re, &coeff_im, &delay, 2997924580.0, true, true,
                                                &aod, &eod, &aoa, &eoa);

    aod *= 180.0f / pi;
    eod *= 180.0f / pi;
    aoa *= 180.0f / pi;
    eoa *= 180.0f / pi;

    REQUIRE(coeff_re.n_slices == 3);
    REQUIRE(coeff_im.n_slices == 3);
    REQUIRE(delay.n_slices == 3);
    REQUIRE(aod.n_slices == 3);
    REQUIRE(eod.n_slices == 3);
    REQUIRE(aoa.n_slices == 3);
    REQUIRE(eoa.n_slices == 3);

    // Check departure azimuth angles
    CHECK(std::abs(aod.at(0, 0, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(aod.at(0, 0, 1) - 90.0f) < 1.0e-6f);
    CHECK(std::abs(aod.at(0, 0, 2) - 0.0f) < 1.0e-6f);

    // Check arrival azimuth angles
    float alpha = 180.0 - std::atan(10.0 / 20.0) * 180.0f / pi;
    CHECK(std::abs(std::abs(aoa.at(0, 0, 0)) - 180.0f) < 1.0e-4f);
    CHECK(std::abs(aoa.at(0, 0, 1) - alpha) < 1.0e-4f);
    CHECK(std::abs(std::abs(aoa.at(0, 0, 2)) - 180.0f) < 1.0e-4f);

    // Check departure elevation angles
    CHECK(std::abs(eod.at(0, 0, 0) - 0.0f) < 1.0e-4f);
    CHECK(std::abs(eod.at(0, 0, 1) - 45.0f) < 1.0e-4f);
    CHECK(std::abs(eod.at(0, 0, 2) - 0.0f) < 1.0e-4f);

    // Check arrival elevation angles
    alpha = std::atan(10.0 / std::sqrt(10.0 * 10.0 + 20.0 * 20.0)) * 180.0f / pi;
    CHECK(std::abs(eoa.at(0, 0, 0) - 0.0f) < 1.0e-4f);
    CHECK(std::abs(eoa.at(0, 0, 1) - alpha) < 1.0e-4f);
    CHECK(std::abs(eoa.at(0, 0, 2) - 0.0f) < 1.0e-4f);

    // Check amplitude
    arma::fcube amp = arma::sqrt(coeff_re % coeff_re + coeff_im % coeff_im);
    CHECK(std::abs(amp.at(0, 0, 0) - 1.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(0, 0, 1) - 0.5f) < 1.0e-6f);
    CHECK(std::abs(amp.at(0, 0, 2) - 0.0f) < 1.0e-6f);

    // Check delays
    float C = 299792458.0;
    float d0 = 20.0;
    float e0 = (std::sqrt(10.0 * 10.0 + 10.0 * 10.0) + std::sqrt(10.0 * 10.0 + 20.0 * 20.0 + 10.0 * 10.0));
    CHECK(std::abs(delay.at(0, 0, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 0, 1) - e0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 0, 2) - d0 / C) < 1.0e-13f);
}

TEST_CASE("Get Channels Spherical - Coupling")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    ant.copy_element(0, 2);
    ant.element_pos(1, 0) = 1.0;
    ant.element_pos(1, 2) = -1.0;

    arma::mat fbs_pos(3, 1);
    fbs_pos(0, 0) = 10.0;
    fbs_pos(2, 0) = 1.0;

    arma::vec path_gain(1), path_length(1);
    path_gain(0) = 1;

    arma::mat M(8, 1);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    // Shortest path length is automatically obtained from FBS / LBS
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    // Case 1 : No coupling (identity matrix)
    quadriga_lib::get_channels_spherical<double>(&ant, &ant,
                                                 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                 20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                 &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
                                                 &coeff_re, &coeff_im, &delay, 0.0, false, true);

    REQUIRE(coeff_re.n_slices == 2);
    REQUIRE(coeff_im.n_slices == 2);
    REQUIRE(delay.n_slices == 2);

    // Possible delay offsets
    double da = (std::sqrt(20.0 * 20.0 + 1.0) - 20.0) / 299792458.0;
    double db = (std::sqrt(20.0 * 20.0 + 4.0) - 20.0) / 299792458.0;

    arma::mat T(3, 3); // Zeros
    CHECK(arma::approx_equal(coeff_im.slice(0), T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(coeff_im.slice(1), T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(coeff_re.slice(1), T, "absdiff", 1e-14));
    T.at(0, 0) = 1.0;
    T.at(0, 2) = 1.0;
    T.at(2, 0) = 1.0;
    T.at(2, 2) = 1.0;
    CHECK(arma::approx_equal(coeff_re.slice(0), T, "absdiff", 1e-14));

    T = {{0.0, da, db}, {da, 0.0, da}, {db, da, 0.0}};
    CHECK(arma::approx_equal(delay.slice(0), T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(delay.slice(1), T, "absdiff", 1e-14));

    // Case 2: n_ports == n_elements
    ant.coupling_re.zeros();
    ant.coupling_re.col(0).ones();
    ant.coupling_im.reset();

    quadriga_lib::get_channels_spherical<double>(&ant, &ant,
                                                 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                 20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                 &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
                                                 &coeff_re, &coeff_im, &delay, 0.0, true, false,
                                                 &aod, &eod, &aoa, &eoa);
    CHECK(coeff_re.n_slices == 1);
    CHECK(coeff_im.n_slices == 1);
    CHECK(delay.n_slices == 1);

    T.zeros();
    CHECK(arma::approx_equal(eod.slice(0), T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(aod.slice(0), T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(eoa.slice(0), T, "absdiff", 1e-14));
    CHECK(arma::approx_equal(coeff_im.slice(0), T, "absdiff", 1e-14));

    T.at(0, 0) = 4.0;
    CHECK(arma::approx_equal(coeff_re.slice(0), T, "absdiff", 1e-14));

    T.ones();
    CHECK(arma::approx_equal(arma::cos(aoa.slice(0)), -T, "absdiff", 1e-14));

    double dx = (4.0 * da + 2.0 * db) / 9.0 + 20.0 / 299792458.0;
    T = T * dx;
    CHECK(arma::approx_equal(delay.slice(0), T, "absdiff", 1e-14));

    // Case 3: n_ports != n_elements
    delay.zeros();
    ant.coupling_re.ones(3, 1);
    ant.coupling_re *= 2.0;

    quadriga_lib::get_channels_spherical<double>(&ant, &ant,
                                                 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                 20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                 &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
                                                 &coeff_re, &coeff_im, &delay, 0.0, true, false,
                                                 &aod, &eod, &aoa);

    CHECK(coeff_re.n_elem == 1);
    CHECK(coeff_im.n_elem == 1);
    CHECK(delay.n_elem == 1);
    CHECK(aod.n_elem == 1);
    CHECK(eod.n_elem == 1);
    CHECK(aoa.n_elem == 1);

    CHECK(std::abs(coeff_re.at(0, 0, 0) - 16.0) < 1.0e-14f);
    CHECK(std::abs(coeff_im.at(0, 0, 0) - 0.0) < 1.0e-14f);
    CHECK(std::abs(delay.at(0, 0, 0) - dx) < 1.0e-14f);
    CHECK(std::abs(aod.at(0, 0, 0) - 0.0) < 1.0e-14f);
    CHECK(std::abs(eod.at(0, 0, 0) - 0.0) < 1.0e-14f);
    CHECK(std::abs(std::cos(aoa.at(0, 0, 0)) + 1.0) < 1.0e-14f);
}
