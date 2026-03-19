// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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

    // Check amplitude (relaxed tolerance for AVX2 float interpolation)
    arma::fcube amp = arma::sqrt(coeff_re % coeff_re + coeff_im % coeff_im);
    CHECK(std::abs(amp.at(0, 0, 0) - 1.0f) < 1.0e-5f);
    CHECK(std::abs(amp.at(1, 0, 0) - 2.0f) < 1.0e-5f);
    CHECK(std::abs(amp.at(0, 1, 0) - 2.0f) < 1.0e-5f);
    CHECK(std::abs(amp.at(1, 1, 0) - 4.0f) < 1.0e-5f);
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
    CHECK(std::abs(delay.at(0, 0, 0) - d0 / C) < 1.0e-13);
    CHECK(std::abs(delay.at(1, 0, 0) - d0 / C) < 1.0e-13);
    CHECK(std::abs(delay.at(0, 1, 0) - d0 / C) < 1.0e-13);
    CHECK(std::abs(delay.at(1, 1, 0) - d0 / C) < 1.0e-13);

    CHECK(std::abs(coeff_re.at(0, 0, 0) - 0.0) < 1.0e-6);
    CHECK(std::abs(coeff_re.at(1, 0, 0) - 1.0) < 1.0e-6);
    CHECK(std::abs(coeff_re.at(0, 1, 0) - 0.0) < 1.0e-6);
    CHECK(std::abs(coeff_re.at(1, 1, 0) - 2.0) < 1.0e-6);

    CHECK(std::abs(coeff_im.at(0, 0, 0) - 0.0) < 1.0e-6);
    CHECK(std::abs(coeff_im.at(1, 0, 0) - 0.0) < 1.0e-6);
    CHECK(std::abs(coeff_im.at(0, 1, 0) - 0.0) < 1.0e-6);
    CHECK(std::abs(coeff_im.at(1, 1, 0) - 0.0) < 1.0e-6);

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
    CHECK(arma::approx_equal(coeff_im.slice(0), T, "absdiff", 1e-6));
    CHECK(arma::approx_equal(coeff_im.slice(1), T, "absdiff", 1e-6));
    CHECK(arma::approx_equal(coeff_re.slice(1), T, "absdiff", 1e-6));
    T.at(0, 0) = 1.0;
    T.at(0, 2) = 1.0;
    T.at(2, 0) = 1.0;
    T.at(2, 2) = 1.0;
    CHECK(arma::approx_equal(coeff_re.slice(0), T, "absdiff", 1e-6));

    T = arma::mat(3, 3, arma::fill::zeros);
    T.at(0, 0) = 0.0;  T.at(0, 1) = da;  T.at(0, 2) = db;
    T.at(1, 0) = da;   T.at(1, 1) = 0.0; T.at(1, 2) = da;
    T.at(2, 0) = db;   T.at(2, 1) = da;  T.at(2, 2) = 0.0;
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
    CHECK(arma::approx_equal(coeff_im.slice(0), T, "absdiff", 1e-6));

    T.at(0, 0) = 4.0;
    CHECK(arma::approx_equal(coeff_re.slice(0), T, "absdiff", 1e-6));

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

    CHECK(std::abs(coeff_re.at(0, 0, 0) - 16.0) < 1.0e-5);
    CHECK(std::abs(coeff_im.at(0, 0, 0) - 0.0) < 1.0e-5);
    CHECK(std::abs(delay.at(0, 0, 0) - dx) < 1.0e-14f);
    CHECK(std::abs(aod.at(0, 0, 0) - 0.0) < 1.0e-14f);
    CHECK(std::abs(eod.at(0, 0, 0) - 0.0) < 1.0e-14f);
    CHECK(std::abs(std::cos(aoa.at(0, 0, 0)) + 1.0) < 1.0e-14f);
}

// ================================================================================================
// New tests below
// ================================================================================================

TEST_CASE("Get Channels Spherical - Error handling: NULL pointers")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = 10.0;
    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1, arma::fill::zeros);
    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;
    arma::cube coeff_re, coeff_im, delay;

    // NULL tx_array
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            (quadriga_lib::arrayant<double> *)nullptr, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // NULL rx_array
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, (quadriga_lib::arrayant<double> *)nullptr,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // NULL fbs_pos
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            (arma::mat *)nullptr, &fbs_pos, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // NULL lbs_pos
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, (arma::mat *)nullptr, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // NULL path_gain
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, (arma::Col<double> *)nullptr, &path_length, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // NULL path_length
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, (arma::Col<double> *)nullptr, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // NULL M
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, (arma::mat *)nullptr,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // NULL coeff_re
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
            (arma::cube *)nullptr, &coeff_im, &delay),
        std::invalid_argument);

    // NULL coeff_im
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
            &coeff_re, (arma::cube *)nullptr, &delay),
        std::invalid_argument);

    // NULL delay
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, (arma::cube *)nullptr),
        std::invalid_argument);
}

TEST_CASE("Get Channels Spherical - Error handling: Dimension mismatches")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    arma::mat fbs_pos(3, 2, arma::fill::zeros);
    fbs_pos(0, 0) = 10.0;
    fbs_pos(0, 1) = 5.0;
    arma::mat lbs_pos(3, 1, arma::fill::zeros); // Wrong: 1 col vs 2
    lbs_pos(0, 0) = 10.0;
    arma::vec path_gain(2, arma::fill::ones);
    arma::vec path_length(2, arma::fill::zeros);
    arma::mat M(8, 2, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(0, 1) = 1.0;
    arma::cube coeff_re, coeff_im, delay;

    // n_path mismatch between fbs_pos (2 cols) and lbs_pos (1 col)
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &lbs_pos, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // Wrong number of rows in fbs_pos
    arma::mat fbs_bad(2, 2, arma::fill::zeros); // 2 rows instead of 3
    arma::mat lbs_ok(3, 2, arma::fill::zeros);
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_bad, &lbs_ok, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // Wrong number of rows in M (should be 8)
    arma::mat M_bad(4, 2, arma::fill::zeros);
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, &M_bad,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // Empty inputs
    arma::mat empty_fbs;
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &empty_fbs, &fbs_pos, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);

    // n_path mismatch for path_gain
    arma::vec pg_bad(3, arma::fill::ones); // 3 elements vs 2 paths
    CHECK_THROWS_AS(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &pg_bad, &path_length, &M,
            &coeff_re, &coeff_im, &delay),
        std::invalid_argument);
}

TEST_CASE("Get Channels Spherical - SISO single path LOS")
{
    // Simplest possible scenario: 1 TX element, 1 RX element, 1 LOS path along x-axis
    auto tx = quadriga_lib::generate_arrayant_omni<double>();
    auto rx = quadriga_lib::generate_arrayant_omni<double>();

    double dist = 100.0;

    // FBS = LBS = midpoint (LOS path, fbs=lbs means zero-length bounce)
    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = dist / 2.0; // Midpoint along x

    arma::vec path_gain(1), path_length(1);
    path_gain(0) = 1.0;
    path_length(0) = dist;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;  // VV = 1
    M(6, 0) = -1.0; // HH = -1

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    // No frequency → no phase rotation, absolute delays
    quadriga_lib::get_channels_spherical<double>(
        &tx, &rx,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        dist, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false,
        &aod, &eod, &aoa, &eoa);

    // Output shape: [1, 1, 1]
    REQUIRE(coeff_re.n_rows == 1);
    REQUIRE(coeff_re.n_cols == 1);
    REQUIRE(coeff_re.n_slices == 1);

    // With zero frequency, no phase rotation: coeff should be purely real
    CHECK(std::abs(coeff_im.at(0, 0, 0)) < 1.0e-14);

    // Amplitude should equal sqrt(path_gain) * antenna_response = 1.0
    double amp = std::sqrt(coeff_re.at(0, 0, 0) * coeff_re.at(0, 0, 0) +
                           coeff_im.at(0, 0, 0) * coeff_im.at(0, 0, 0));
    CHECK(std::abs(amp - 1.0) < 1.0e-6);

    // Absolute delay = distance / speed_of_light
    double C = 299792458.0;
    CHECK(std::abs(delay.at(0, 0, 0) - dist / C) < 1.0e-14);

    // Departure azimuth should be 0 (along +x)
    CHECK(std::abs(aod.at(0, 0, 0) - 0.0) < 1.0e-6);

    // Departure elevation should be 0 (in xy plane)
    CHECK(std::abs(eod.at(0, 0, 0) - 0.0) < 1.0e-6);

    // Arrival azimuth should be +/-pi (coming from -x direction)
    CHECK(std::abs(std::cos(aoa.at(0, 0, 0)) + 1.0) < 1.0e-6);

    // Arrival elevation should be 0
    CHECK(std::abs(eoa.at(0, 0, 0) - 0.0) < 1.0e-6);
}

TEST_CASE("Get Channels Spherical - Zero center frequency disables phase")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0;

    arma::vec path_gain(1), path_length(1);
    path_gain(0) = 1.0;
    path_length(0) = 100.0;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay;

    // center_frequency = 0 → phase calculation disabled
    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false);

    // With zero frequency, coeff_im should be near-zero (no phase rotation)
    CHECK(std::abs(coeff_im.at(0, 0, 0)) < 1.0e-6);

    // Coefficient should be purely real and positive
    CHECK(coeff_re.at(0, 0, 0) > 0.0);
    CHECK(std::abs(coeff_re.at(0, 0, 0) - 1.0) < 1.0e-6);
}

TEST_CASE("Get Channels Spherical - Absolute vs relative delays")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    double dist = 50.0;
    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = dist / 2.0;

    arma::vec path_gain(1), path_length(1);
    path_gain(0) = 1.0;
    path_length(0) = dist;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re_abs, coeff_im_abs, delay_abs;
    arma::cube coeff_re_rel, coeff_im_rel, delay_rel;

    double C = 299792458.0;

    // use_absolute_delays = true
    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        dist, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re_abs, &coeff_im_abs, &delay_abs, 0.0, true, false);

    // use_absolute_delays = false
    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        dist, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re_rel, &coeff_im_rel, &delay_rel, 0.0, false, false);

    // Absolute delay for LOS = dist / C
    CHECK(std::abs(delay_abs.at(0, 0, 0) - dist / C) < 1.0e-14);

    // Relative delay for LOS path should be 0 (delay = path_delay - LOS_delay)
    CHECK(std::abs(delay_rel.at(0, 0, 0) - 0.0) < 1.0e-14);

    // Coefficients should be identical regardless of delay mode
    CHECK(std::abs(coeff_re_abs.at(0, 0, 0) - coeff_re_rel.at(0, 0, 0)) < 1.0e-14);
    CHECK(std::abs(coeff_im_abs.at(0, 0, 0) - coeff_im_rel.at(0, 0, 0)) < 1.0e-14);
}

TEST_CASE("Get Channels Spherical - Optional angle outputs as nullptr")
{
    // Verify the function works when angle outputs are not requested
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0;

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1, arma::fill::zeros);
    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay;

    // No angle outputs at all (default nullptr)
    CHECK_NOTHROW(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay, 0.0, true, false));

    REQUIRE(coeff_re.n_rows == 1);
    REQUIRE(coeff_re.n_cols == 1);
    REQUIRE(coeff_re.n_slices == 1);

    // Only some angle outputs requested
    arma::cube aod, eoa;
    CHECK_NOTHROW(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay, 0.0, true, false,
            &aod, (arma::Cube<double> *)nullptr, (arma::Cube<double> *)nullptr, &eoa));

    REQUIRE(aod.n_rows == 1);
    REQUIRE(aod.n_cols == 1);
    REQUIRE(aod.n_slices == 1);
    REQUIRE(eoa.n_rows == 1);
    REQUIRE(eoa.n_cols == 1);
    REQUIRE(eoa.n_slices == 1);
}

TEST_CASE("Get Channels Spherical - Multiple NLOS paths, relative delays")
{
    // SISO scenario with 3 paths: 1 LOS + 2 NLOS
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    double dist = 100.0;
    double C = 299792458.0;

    // Path 0: LOS (fbs=lbs at midpoint)
    // Path 1: NLOS bounce above
    // Path 2: NLOS bounce to the side
    arma::mat fbs_pos(3, 3, arma::fill::zeros);
    fbs_pos(0, 0) = dist / 2.0; // LOS midpoint
    fbs_pos(0, 1) = dist / 2.0; // NLOS above
    fbs_pos(2, 1) = 30.0;
    fbs_pos(0, 2) = dist / 2.0; // NLOS to the side
    fbs_pos(1, 2) = 40.0;

    arma::vec path_gain(3), path_length(3);
    path_gain(0) = 1.0;
    path_gain(1) = 0.5;
    path_gain(2) = 0.25;
    path_length(0) = dist;
    path_length(1) = 0.0; // Will be calculated from geometry
    path_length(2) = 0.0;

    arma::mat M(8, 3, arma::fill::zeros);
    M(0, 0) = 1.0;  M(6, 0) = -1.0;
    M(0, 1) = 1.0;  M(6, 1) = -1.0;
    M(0, 2) = 1.0;  M(6, 2) = -1.0;

    arma::cube coeff_re, coeff_im, delay;

    // Relative delays (use_absolute_delays = false)
    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        dist, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, false, false);

    REQUIRE(coeff_re.n_slices == 3);

    // LOS delay should be 0 (relative)
    CHECK(std::abs(delay.at(0, 0, 0) - 0.0) < 1.0e-14);

    // NLOS delays should be > 0 (longer path than LOS)
    CHECK(delay.at(0, 0, 1) > 0.0);
    CHECK(delay.at(0, 0, 2) > 0.0);

    // NLOS delays should match geometry
    // Path 1: TX(0,0,0) -> FBS(50,0,30) -> LBS(50,0,30) -> RX(100,0,0)
    double d1_tx_fbs = std::sqrt(50.0 * 50.0 + 30.0 * 30.0);
    double d1_lbs_rx = d1_tx_fbs; // symmetric
    double d1_total = d1_tx_fbs + d1_lbs_rx;
    CHECK(std::abs(delay.at(0, 0, 1) - (d1_total - dist) / C) < 1.0e-14);

    // Path 2: TX(0,0,0) -> FBS(50,40,0) -> LBS(50,40,0) -> RX(100,0,0)
    double d2_tx_fbs = std::sqrt(50.0 * 50.0 + 40.0 * 40.0);
    double d2_lbs_rx = d2_tx_fbs; // symmetric
    double d2_total = d2_tx_fbs + d2_lbs_rx;
    CHECK(std::abs(delay.at(0, 0, 2) - (d2_total - dist) / C) < 1.0e-14);

    // Amplitude check: sqrt(path_gain)
    arma::cube amp = arma::sqrt(coeff_re % coeff_re + coeff_im % coeff_im);
    CHECK(std::abs(amp.at(0, 0, 0) - 1.0) < 1.0e-6);
    CHECK(std::abs(amp.at(0, 0, 1) - std::sqrt(0.5)) < 1.0e-6);
    CHECK(std::abs(amp.at(0, 0, 2) - 0.5) < 1.0e-6);
}

TEST_CASE("Get Channels Spherical - Path gain zero produces zero coefficients")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    arma::mat fbs_pos(3, 2, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0; // LOS midpoint
    fbs_pos(0, 1) = 50.0; // NLOS
    fbs_pos(1, 1) = 20.0;

    arma::vec path_gain(2), path_length(2);
    path_gain(0) = 1.0;
    path_gain(1) = 0.0; // Zero gain

    arma::mat M(8, 2, arma::fill::zeros);
    M(0, 0) = 1.0;  M(6, 0) = -1.0;
    M(0, 1) = 1.0;  M(6, 1) = -1.0;

    arma::cube coeff_re, coeff_im, delay;

    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false);

    REQUIRE(coeff_re.n_slices == 2);

    // Path 0 should have non-zero coefficients
    double amp0 = std::sqrt(coeff_re.at(0, 0, 0) * coeff_re.at(0, 0, 0) +
                            coeff_im.at(0, 0, 0) * coeff_im.at(0, 0, 0));
    CHECK(amp0 > 0.0);

    // Path 1 with zero gain should produce zero coefficients
    CHECK(std::abs(coeff_re.at(0, 0, 1)) < 1.0e-14);
    CHECK(std::abs(coeff_im.at(0, 0, 1)) < 1.0e-14);
}

TEST_CASE("Get Channels Spherical - Double precision SISO")
{
    // Repeat of the single-path LOS test in double precision to verify template instantiation
    auto tx = quadriga_lib::generate_arrayant_omni<double>();
    auto rx = quadriga_lib::generate_arrayant_omni<double>();

    double dist = 200.0;
    double C = 299792458.0;

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = dist / 2.0;

    arma::vec path_gain(1), path_length(1);
    path_gain(0) = 1.0;
    path_length(0) = dist;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<double>(
        &tx, &rx,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        dist, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false,
        &aod, &eod, &aoa, &eoa);

    // Double precision — coefficient tolerance relaxed for AVX2 float interpolation
    CHECK(std::abs(coeff_im.at(0, 0, 0)) < 1.0e-6);
    CHECK(std::abs(coeff_re.at(0, 0, 0) - 1.0) < 1.0e-6);
    CHECK(std::abs(delay.at(0, 0, 0) - dist / C) < 1.0e-14);
    CHECK(std::abs(aod.at(0, 0, 0) - 0.0) < 1.0e-14);
    CHECK(std::abs(eod.at(0, 0, 0) - 0.0) < 1.0e-14);
    CHECK(std::abs(eoa.at(0, 0, 0) - 0.0) < 1.0e-14);
}

TEST_CASE("Get Channels Spherical - Float precision SISO")
{
    // Same test in float to verify float template instantiation
    auto tx = quadriga_lib::generate_arrayant_omni<float>();
    auto rx = quadriga_lib::generate_arrayant_omni<float>();

    float dist = 200.0f;
    float C = 299792458.0f;

    arma::fmat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = dist / 2.0f;

    arma::fvec path_gain(1), path_length(1);
    path_gain(0) = 1.0f;
    path_length(0) = dist;

    arma::fmat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0f;
    M(6, 0) = -1.0f;

    arma::fcube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<float>(
        &tx, &rx,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        dist, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0f, true, false,
        &aod, &eod, &aoa, &eoa);

    // Float precision is looser
    CHECK(std::abs(coeff_im.at(0, 0, 0)) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(0, 0, 0) - 1.0f) < 1.0e-6f);
    CHECK(std::abs(delay.at(0, 0, 0) - dist / C) < 1.0e-13f);
    CHECK(std::abs(aod.at(0, 0, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eod.at(0, 0, 0) - 0.0f) < 1.0e-6f);
    CHECK(std::abs(eoa.at(0, 0, 0) - 0.0f) < 1.0e-6f);
}

TEST_CASE("Get Channels Spherical - Fake LOS with single path")
{
    // When add_fake_los_path = true, output should have n_path + 1 slices
    // The fake LOS path (slice 0) should have zero power
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0;
    fbs_pos(1, 0) = 30.0; // Off-axis NLOS path

    arma::vec path_gain(1), path_length(1);
    path_gain(0) = 1.0;
    path_length(0) = 0.0;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay;

    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, true);

    // Should have 2 slices: fake LOS + 1 real path
    REQUIRE(coeff_re.n_slices == 2);
    REQUIRE(delay.n_slices == 2);

    // Fake LOS path (slice 0) should have zero amplitude
    CHECK(std::abs(coeff_re.at(0, 0, 0)) < 1.0e-14);
    CHECK(std::abs(coeff_im.at(0, 0, 0)) < 1.0e-14);

    // Real path (slice 1) should have non-zero amplitude
    double amp1 = std::sqrt(coeff_re.at(0, 0, 1) * coeff_re.at(0, 0, 1) +
                            coeff_im.at(0, 0, 1) * coeff_im.at(0, 0, 1));
    CHECK(amp1 > 0.0);
}

TEST_CASE("Get Channels Spherical - Vertical path (elevation angles)")
{
    // TX at origin, RX directly above → departure elevation = 90 deg
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    double pi = 3.141592653589793;
    double height = 50.0;

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(2, 0) = height / 2.0; // Midpoint vertically

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1);
    path_length(0) = height;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, height, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false,
        &aod, &eod, &aoa, &eoa);

    // Departure elevation should be +90 deg (pi/2)
    CHECK(std::abs(eod.at(0, 0, 0) - pi / 2.0) < 1.0e-6);

    // Arrival elevation should be -90 deg (-pi/2)
    CHECK(std::abs(eoa.at(0, 0, 0) + pi / 2.0) < 1.0e-6);

    // Delay should match vertical distance
    double C = 299792458.0;
    CHECK(std::abs(delay.at(0, 0, 0) - height / C) < 1.0e-14);
}

TEST_CASE("Get Channels Spherical - Diagonal path (non-trivial azimuth + elevation)")
{
    // TX at origin, RX at (100, 100, 0) → AOD should be 45 deg
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    double pi = 3.141592653589793;
    double d = 100.0;

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = d / 2.0;
    fbs_pos(1, 0) = d / 2.0;

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1);
    path_length(0) = std::sqrt(2.0) * d;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        d, d, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false,
        &aod, &eod, &aoa, &eoa);

    // AOD should be 45 deg = pi/4
    CHECK(std::abs(aod.at(0, 0, 0) - pi / 4.0) < 1.0e-6);

    // EOD should be 0 (path in xy-plane)
    CHECK(std::abs(eod.at(0, 0, 0) - 0.0) < 1.0e-6);

    // AOA should be 180 + 45 = 225 deg = -135 deg = -3*pi/4
    // Or equivalently, atan2(-100, -100)
    double expected_aoa = std::atan2(-d, -d);
    CHECK(std::abs(aoa.at(0, 0, 0) - expected_aoa) < 1.0e-6);

    // Delay
    double C = 299792458.0;
    double dist = std::sqrt(d * d + d * d);
    CHECK(std::abs(delay.at(0, 0, 0) - dist / C) < 1.0e-14);
}

TEST_CASE("Get Channels Spherical - Complex polarization transfer matrix M")
{
    // Test with non-trivial imaginary components in M
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0;

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1);
    path_length(0) = 100.0;

    // M has complex VV component: VV = cos(45°) + j*sin(45°)
    arma::mat M(8, 1, arma::fill::zeros);
    double s2 = std::sqrt(2.0) / 2.0;
    M(0, 0) = s2;   // Re(VV)
    M(1, 0) = s2;   // Im(VV)
    M(6, 0) = -s2;  // Re(HH)
    M(7, 0) = -s2;  // Im(HH)

    arma::cube coeff_re, coeff_im, delay;

    // Zero frequency: no additional phase from propagation
    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false);

    // Amplitude should be 1.0 (|M_VV| = 1.0, omni antenna)
    double amp = std::sqrt(coeff_re.at(0, 0, 0) * coeff_re.at(0, 0, 0) +
                           coeff_im.at(0, 0, 0) * coeff_im.at(0, 0, 0));
    CHECK(std::abs(amp - 1.0) < 1.0e-6);

    // The imaginary component of M should introduce phase even without center_frequency
    // coeff_im should be non-zero
    CHECK(std::abs(coeff_im.at(0, 0, 0)) > 1.0e-3);
}

TEST_CASE("Get Channels Spherical - RX rotation")
{
    // Test that rotating the RX antenna affects the channel coefficients
    double pi = 3.141592653589793;

    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    auto probe = quadriga_lib::generate_arrayant_xpol<double>();

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0;

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1, arma::fill::zeros);
    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re_0, coeff_im_0, delay_0;
    arma::cube coeff_re_r, coeff_im_r, delay_r;

    // No RX rotation
    quadriga_lib::get_channels_spherical<double>(
        &ant, &probe,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re_0, &coeff_im_0, &delay_0, 0.0, false, false);

    // RX rotated by 90 degrees bank
    quadriga_lib::get_channels_spherical<double>(
        &ant, &probe,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, pi / 2.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re_r, &coeff_im_r, &delay_r, 0.0, false, false);

    // Delays should be the same (rotation doesn't change path length for point antenna)
    CHECK(std::abs(delay_0.at(0, 0, 0) - delay_r.at(0, 0, 0)) < 1.0e-14);
    CHECK(std::abs(delay_0.at(1, 0, 0) - delay_r.at(1, 0, 0)) < 1.0e-14);

    // Xpol probe has V and H components: rotating by 90° bank should swap them
    // The V-pol element response at 0° bank should correspond to H-pol at 90° bank
    double amp_v_0 = std::sqrt(coeff_re_0.at(0, 0, 0) * coeff_re_0.at(0, 0, 0) +
                               coeff_im_0.at(0, 0, 0) * coeff_im_0.at(0, 0, 0));
    double amp_h_0 = std::sqrt(coeff_re_0.at(1, 0, 0) * coeff_re_0.at(1, 0, 0) +
                               coeff_im_0.at(1, 0, 0) * coeff_im_0.at(1, 0, 0));
    double amp_v_r = std::sqrt(coeff_re_r.at(0, 0, 0) * coeff_re_r.at(0, 0, 0) +
                               coeff_im_r.at(0, 0, 0) * coeff_im_r.at(0, 0, 0));
    double amp_h_r = std::sqrt(coeff_re_r.at(1, 0, 0) * coeff_re_r.at(1, 0, 0) +
                               coeff_im_r.at(1, 0, 0) * coeff_im_r.at(1, 0, 0));

    // 90° bank rotation on xpol: V becomes H and vice versa (amplitudes swap)
    CHECK(std::abs(amp_v_0 - amp_h_r) < 1.0e-6);
    CHECK(std::abs(amp_h_0 - amp_v_r) < 1.0e-6);
}

TEST_CASE("Get Channels Spherical - TX along negative x-axis")
{
    // Verify angles when RX is in the -x direction from TX
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    double pi = 3.141592653589793;

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = -25.0; // Midpoint between TX(0) and RX(-50)

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1);
    path_length(0) = 50.0;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -50.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false,
        &aod, &eod, &aoa, &eoa);

    // AOD should be pi (or -pi), pointing toward -x
    CHECK(std::abs(std::cos(aod.at(0, 0, 0)) + 1.0) < 1.0e-6);

    // AOA should be 0, coming from +x direction
    CHECK(std::abs(aoa.at(0, 0, 0) - 0.0) < 1.0e-6);
}

TEST_CASE("Get Channels Spherical - FBS and LBS at different positions")
{
    // Verify delay computation when FBS != LBS (two-bounce scenario)
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    double C = 299792458.0;

    // TX at (0,0,0), RX at (100,0,0)
    // FBS at (20,30,0), LBS at (80,30,0)
    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = 20.0;
    fbs_pos(1, 0) = 30.0;

    arma::mat lbs_pos(3, 1, arma::fill::zeros);
    lbs_pos(0, 0) = 80.0;
    lbs_pos(1, 0) = 30.0;

    double d_tx_fbs = std::sqrt(20.0 * 20.0 + 30.0 * 30.0);
    double d_fbs_lbs = 60.0; // 80 - 20 = 60 along x
    double d_lbs_rx = std::sqrt(20.0 * 20.0 + 30.0 * 30.0);
    double d_shortest = d_tx_fbs + d_fbs_lbs + d_lbs_rx;

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1);
    path_length(0) = d_shortest + 10.0; // Longer than shortest

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &lbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false,
        &aod, &eod, &aoa, &eoa);

    // Per-element NLOS delay is computed from geometry: TX→FBS + FBS→LBS + LBS→RX
    // path_length only affects LOS/NLOS branch selection, not the stored delay
    CHECK(std::abs(delay.at(0, 0, 0) - d_shortest / C) < 1.0e-14);

    // AOD should point toward FBS: atan2(30, 20)
    double expected_aod = std::atan2(30.0, 20.0);
    CHECK(std::abs(aod.at(0, 0, 0) - expected_aod) < 1.0e-6);

    // AOA should point from LBS toward RX: atan2(30-0, 80-100) = atan2(30, -20)
    double expected_aoa = std::atan2(30.0, -20.0);
    CHECK(std::abs(aoa.at(0, 0, 0) - expected_aoa) < 1.0e-6);
}

TEST_CASE("Get Channels Spherical - Phase with center frequency")
{
    // Verify that center frequency correctly introduces phase shifts
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    double dist = 100.0;
    double C = 299792458.0;
    double fc = 1.0e9; // 1 GHz
    double wavelength = C / fc;

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = dist / 2.0;

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1);
    path_length(0) = dist;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re_f, coeff_im_f, delay_f;
    arma::cube coeff_re_0, coeff_im_0, delay_0;

    // With center frequency
    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        dist, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re_f, &coeff_im_f, &delay_f, fc, true, false);

    // Without center frequency
    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        dist, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re_0, &coeff_im_0, &delay_0, 0.0, true, false);

    // Amplitudes should match
    double amp_f = std::sqrt(coeff_re_f.at(0, 0, 0) * coeff_re_f.at(0, 0, 0) +
                             coeff_im_f.at(0, 0, 0) * coeff_im_f.at(0, 0, 0));
    double amp_0 = std::sqrt(coeff_re_0.at(0, 0, 0) * coeff_re_0.at(0, 0, 0) +
                             coeff_im_0.at(0, 0, 0) * coeff_im_0.at(0, 0, 0));
    CHECK(std::abs(amp_f - amp_0) < 1.0e-10);

    // Delays should match
    CHECK(std::abs(delay_f.at(0, 0, 0) - delay_0.at(0, 0, 0)) < 1.0e-14);

    // Phase from frequency: wave_number * fmod(distance, wavelength)
    double wave_number = 2.0 * 3.141592653589793 / wavelength;
    double expected_phase = wave_number * std::fmod(dist, wavelength);
    double actual_phase = std::atan2(-coeff_im_f.at(0, 0, 0), coeff_re_f.at(0, 0, 0));

    // Phases should match (mod 2pi)
    double phase_diff = actual_phase - expected_phase;
    double twopi = 2.0 * 3.141592653589793;
    phase_diff = std::fmod(phase_diff + twopi, twopi);
    if (phase_diff > 3.141592653589793)
        phase_diff -= twopi;
    CHECK(std::abs(phase_diff) < 1.0e-6);
}

TEST_CASE("Get Channels Spherical - Output size with MIMO and multiple paths")
{
    // Verify output cube dimensions for 2x3 MIMO with 4 paths
    auto tx = quadriga_lib::generate_arrayant_omni<double>();
    tx.copy_element(0, 1);
    tx.copy_element(0, 2);
    tx.element_pos(1, 0) = 0.5;
    tx.element_pos(1, 1) = 0.0;
    tx.element_pos(1, 2) = -0.5;

    auto rx = quadriga_lib::generate_arrayant_omni<double>();
    rx.copy_element(0, 1);
    rx.element_pos(1, 0) = 0.5;
    rx.element_pos(1, 1) = -0.5;

    arma::uword n_path = 4;
    arma::mat fbs_pos(3, n_path, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0;
    fbs_pos(0, 1) = 40.0; fbs_pos(1, 1) = 20.0;
    fbs_pos(0, 2) = 60.0; fbs_pos(2, 2) = 15.0;
    fbs_pos(0, 3) = 30.0; fbs_pos(1, 3) = -10.0;

    arma::vec path_gain(n_path, arma::fill::ones);
    arma::vec path_length(n_path, arma::fill::zeros);

    arma::mat M(8, n_path, arma::fill::zeros);
    for (arma::uword p = 0; p < n_path; ++p)
    {
        M(0, p) = 1.0;
        M(6, p) = -1.0;
    }

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    // Without fake LOS
    quadriga_lib::get_channels_spherical<double>(
        &tx, &rx,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false,
        &aod, &eod, &aoa, &eoa);

    // n_rx=2, n_tx=3, n_path=4
    REQUIRE(coeff_re.n_rows == 2);
    REQUIRE(coeff_re.n_cols == 3);
    REQUIRE(coeff_re.n_slices == n_path);
    REQUIRE(coeff_im.n_rows == 2);
    REQUIRE(coeff_im.n_cols == 3);
    REQUIRE(coeff_im.n_slices == n_path);
    REQUIRE(delay.n_rows == 2);
    REQUIRE(delay.n_cols == 3);
    REQUIRE(delay.n_slices == n_path);
    REQUIRE(aod.n_rows == 2);
    REQUIRE(aod.n_cols == 3);
    REQUIRE(aod.n_slices == n_path);
    REQUIRE(eod.n_rows == 2);
    REQUIRE(eod.n_cols == 3);
    REQUIRE(eod.n_slices == n_path);
    REQUIRE(aoa.n_rows == 2);
    REQUIRE(aoa.n_cols == 3);
    REQUIRE(aoa.n_slices == n_path);
    REQUIRE(eoa.n_rows == 2);
    REQUIRE(eoa.n_cols == 3);
    REQUIRE(eoa.n_slices == n_path);

    // With fake LOS
    quadriga_lib::get_channels_spherical<double>(
        &tx, &rx,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, true,
        &aod, &eod, &aoa, &eoa);

    // n_path + 1 slices with fake LOS
    REQUIRE(coeff_re.n_slices == n_path + 1);
    REQUIRE(delay.n_slices == n_path + 1);
    REQUIRE(aod.n_slices == n_path + 1);
}

TEST_CASE("Get Channels Spherical - Physical consistency: delays positive and monotonic ordering")
{
    // All absolute delays should be positive
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    arma::mat fbs_pos(3, 3, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0;
    fbs_pos(0, 1) = 50.0; fbs_pos(1, 1) = 30.0;
    fbs_pos(0, 2) = 50.0; fbs_pos(2, 2) = 40.0;

    arma::vec path_gain(3, arma::fill::ones);
    arma::vec path_length(3, arma::fill::zeros);

    arma::mat M(8, 3, arma::fill::zeros);
    for (arma::uword p = 0; p < 3; ++p)
    {
        M(0, p) = 1.0;
        M(6, p) = -1.0;
    }

    arma::cube coeff_re, coeff_im, delay;

    // Absolute delays
    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false);

    // All delays must be positive
    for (arma::uword s = 0; s < delay.n_slices; ++s)
        CHECK(delay.at(0, 0, s) > 0.0);

    // LOS path should have shortest delay
    CHECK(delay.at(0, 0, 0) <= delay.at(0, 0, 1));
    CHECK(delay.at(0, 0, 0) <= delay.at(0, 0, 2));
}

TEST_CASE("Get Channels Spherical - TX/RX at same height, path along y-axis")
{
    // TX and RX separated along y-axis only
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    double pi = 3.141592653589793;

    double dist = 80.0;

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(1, 0) = dist / 2.0; // midpoint along y

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1);
    path_length(0) = dist;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, dist, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false,
        &aod, &eod, &aoa, &eoa);

    // AOD should be 90 degrees (pi/2) — pointing along +y
    CHECK(std::abs(aod.at(0, 0, 0) - pi / 2.0) < 1.0e-6);

    // AOA should be -90 degrees (-pi/2) — coming from -y direction
    CHECK(std::abs(aoa.at(0, 0, 0) + pi / 2.0) < 1.0e-6);

    // Elevation should be 0
    CHECK(std::abs(eod.at(0, 0, 0)) < 1.0e-6);
    CHECK(std::abs(eoa.at(0, 0, 0)) < 1.0e-6);
}

TEST_CASE("Get Channels Spherical - Asymmetric TX/RX antennas")
{
    // TX has 2 elements, RX has 1 element → output should be [1, 2, n_path]
    auto tx = quadriga_lib::generate_arrayant_omni<double>();
    tx.copy_element(0, 1);
    tx.element_pos(1, 0) = 0.5;
    tx.element_pos(1, 1) = -0.5;

    auto rx = quadriga_lib::generate_arrayant_omni<double>();

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0;

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1, arma::fill::zeros);

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay;

    quadriga_lib::get_channels_spherical<double>(
        &tx, &rx,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false);

    REQUIRE(coeff_re.n_rows == 1);
    REQUIRE(coeff_re.n_cols == 2);
    REQUIRE(coeff_re.n_slices == 1);
    REQUIRE(delay.n_rows == 1);
    REQUIRE(delay.n_cols == 2);
    REQUIRE(delay.n_slices == 1);
}

TEST_CASE("Get Channels Spherical - Offset TX/RX positions")
{
    // TX and RX not at origin, verify position handling
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    double C = 299792458.0;

    double Tx = 10.0, Ty = 20.0, Tz = 5.0;
    double Rx = 110.0, Ry = 20.0, Rz = 5.0;
    double dist = 100.0; // Along x-axis

    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = (Tx + Rx) / 2.0; // midpoint
    fbs_pos(1, 0) = Ty;
    fbs_pos(2, 0) = Tz;

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1);
    path_length(0) = dist;

    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_spherical<double>(
        &ant, &ant,
        Tx, Ty, Tz, 0.0, 0.0, 0.0,
        Rx, Ry, Rz, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0, true, false,
        &aod, &eod, &aoa, &eoa);

    // Delay should be dist / C
    CHECK(std::abs(delay.at(0, 0, 0) - dist / C) < 1.0e-14);

    // AOD should be 0 (RX is along +x from TX)
    CHECK(std::abs(aod.at(0, 0, 0) - 0.0) < 1.0e-6);

    // EOD should be 0 (same height)
    CHECK(std::abs(eod.at(0, 0, 0) - 0.0) < 1.0e-6);
}

TEST_CASE("Get Channels Spherical - Cross-polarization transfer via M matrix")
{
    // Test that VH and HV entries in M produce cross-polarization coupling.
    // The xpol antenna has e_phi = 0 at azimuth 0 and pi (boresight/back).
    // We must use an off-axis scatterer so the H-pol (phi) response is non-zero.
    auto probe = quadriga_lib::generate_arrayant_xpol<double>();

    // Scatterer at (50, 50, 0): AOD = 45 deg, AOA = 135 deg → non-zero e_phi
    arma::mat fbs_pos(3, 1, arma::fill::zeros);
    fbs_pos(0, 0) = 50.0;
    fbs_pos(1, 0) = 50.0;

    arma::vec path_gain(1, arma::fill::ones);
    arma::vec path_length(1, arma::fill::zeros);

    // Case 1: Pure VH cross-polarization only
    arma::mat M_vh(8, 1, arma::fill::zeros);
    M_vh(2, 0) = 1.0; // Re(VH)

    arma::cube coeff_re_vh, coeff_im_vh, delay_vh;

    quadriga_lib::get_channels_spherical<double>(
        &probe, &probe,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M_vh,
        &coeff_re_vh, &coeff_im_vh, &delay_vh, 0.0, true, false);

    REQUIRE(coeff_re_vh.n_rows == 2);
    REQUIRE(coeff_re_vh.n_cols == 2);

    // Case 2: Pure VV (no cross-pol)
    arma::mat M_vv(8, 1, arma::fill::zeros);
    M_vv(0, 0) = 1.0; // Re(VV) only

    arma::cube coeff_re_vv, coeff_im_vv, delay_vv;

    quadriga_lib::get_channels_spherical<double>(
        &probe, &probe,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M_vv,
        &coeff_re_vv, &coeff_im_vv, &delay_vv, 0.0, true, false);

    // Compute total power across all MIMO links for each case
    double total_power_vh = arma::accu(coeff_re_vh % coeff_re_vh + coeff_im_vh % coeff_im_vh);
    double total_power_vv = arma::accu(coeff_re_vv % coeff_re_vv + coeff_im_vv % coeff_im_vv);

    // Both M configurations should produce non-zero total power
    CHECK(total_power_vh > 1.0e-6);
    CHECK(total_power_vv > 1.0e-6);

    // VV and VH should produce different channel matrices (different polarization coupling)
    double diff = arma::accu(arma::abs(coeff_re_vh - coeff_re_vv)) +
                  arma::accu(arma::abs(coeff_im_vh - coeff_im_vv));
    CHECK(diff > 1.0e-6);

    // Case 3: All-zero M should produce zero output
    arma::mat M_zero(8, 1, arma::fill::zeros);
    arma::cube coeff_re_z, coeff_im_z, delay_z;

    quadriga_lib::get_channels_spherical<double>(
        &probe, &probe,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M_zero,
        &coeff_re_z, &coeff_im_z, &delay_z, 0.0, true, false);

    double total_power_zero = arma::accu(coeff_re_z % coeff_re_z + coeff_im_z % coeff_im_z);
    CHECK(total_power_zero < 1.0e-14);
}

TEST_CASE("Get Channels Spherical - Large number of paths")
{
    // Stress test with many paths to verify no crashes or memory issues
    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    arma::uword n_path = 100;
    arma::mat fbs_pos(3, n_path, arma::fill::zeros);
    for (arma::uword p = 0; p < n_path; ++p)
    {
        double angle = 2.0 * 3.141592653589793 * (double)p / (double)n_path;
        fbs_pos(0, p) = 50.0 + 30.0 * std::cos(angle);
        fbs_pos(1, p) = 30.0 * std::sin(angle);
    }

    arma::vec path_gain(n_path);
    arma::vec path_length(n_path, arma::fill::zeros);
    for (arma::uword p = 0; p < n_path; ++p)
        path_gain(p) = 1.0 / (double)(p + 1);

    arma::mat M(8, n_path, arma::fill::zeros);
    for (arma::uword p = 0; p < n_path; ++p)
    {
        M(0, p) = 1.0;
        M(6, p) = -1.0;
    }

    arma::cube coeff_re, coeff_im, delay;

    CHECK_NOTHROW(
        quadriga_lib::get_channels_spherical<double>(
            &ant, &ant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
            &coeff_re, &coeff_im, &delay, 1.0e9, true, false));

    REQUIRE(coeff_re.n_slices == n_path);
    REQUIRE(delay.n_slices == n_path);

    // All delays should be positive
    for (arma::uword p = 0; p < n_path; ++p)
        CHECK(delay.at(0, 0, p) > 0.0);
}

TEST_CASE("Get Channels Spherical - Fake LOS swaps true LOS to first slice")
{
    // When the true LOS path is not path 0, add_fake_los_path should
    // ensure the real LOS coefficients end up in slice 0
    auto ant = quadriga_lib::generate_arrayant_omni<float>();

    // Path 0: NLOS (off-axis scatterer)
    // Path 1: true LOS (fbs = lbs at midpoint, path_length ≈ distance)
    arma::fmat fbs_pos(3, 2, arma::fill::zeros);
    fbs_pos(0, 0) = 30.0; fbs_pos(1, 0) = 25.0; // NLOS
    fbs_pos(0, 1) = 50.0;                         // LOS midpoint

    arma::fvec path_gain(2), path_length(2);
    path_gain(0) = 0.5f;
    path_gain(1) = 1.0f;
    path_length(0) = 0.0f;
    path_length(1) = 100.0f;

    arma::fmat M(8, 2, arma::fill::zeros);
    M(0, 0) = 1.0f; M(6, 0) = -1.0f;
    M(0, 1) = 1.0f; M(6, 1) = -1.0f;

    arma::fcube coeff_re, coeff_im, delay;

    quadriga_lib::get_channels_spherical<float>(
        &ant, &ant,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        100.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        &fbs_pos, &fbs_pos, &path_gain, &path_length, &M,
        &coeff_re, &coeff_im, &delay, 0.0f, true, true);

    // 3 slices: fake LOS (0) + 2 real paths
    REQUIRE(coeff_re.n_slices == 3);

    // Slice 0 should contain the true LOS path coefficients (swapped to front)
    // The LOS path had path_gain=1.0, so its amplitude should be 1.0
    float amp0 = std::sqrt(coeff_re.at(0, 0, 0) * coeff_re.at(0, 0, 0) +
                           coeff_im.at(0, 0, 0) * coeff_im.at(0, 0, 0));
    CHECK(std::abs(amp0 - 1.0f) < 1.0e-4f);

    // The original slot of the LOS path should now have zero coefficients
    // (true_los_path was at index 2 in the output, i.e. path 1 + fake offset)
    float amp2 = std::sqrt(coeff_re.at(0, 0, 2) * coeff_re.at(0, 0, 2) +
                           coeff_im.at(0, 0, 2) * coeff_im.at(0, 0, 2));
    CHECK(std::abs(amp2) < 1.0e-6f);
}