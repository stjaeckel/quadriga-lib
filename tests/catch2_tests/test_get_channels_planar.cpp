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

TEST_CASE("Get Channels Planar - Minimal test")
{

    float pi = 3.141592653589793f, pid180 = 0.017453292519943f;

    // Generate test antenna
    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    ant.copy_element(0, 1);
    ant.element_pos(1, 0) = 1.0;
    ant.element_pos(1, 1) = -1.0;
    ant.e_theta_re.slice(1) *= 2;

    arma::fvec aod(2), eod(2), aoa(2), eoa(2), path_gain(2), path_length(2);

    //     FBS @ 11 m height
    //      |
    //     10 m
    //      |           FBS LOS @ x = 10 m
    //     TX --------------d = 20 m------------------ RX

    aod(0) = 0.0;
    aod(1) = 90.0 * pid180;

    eod(0) = 0.0;
    eod(1) = 45.0 * pid180;

    aoa(0) = pi;
    aoa(1) = pi - std::atan(10.0 / 20.0);

    eoa(0) = 0.0;
    eoa(1) = std::atan(10.0 / std::sqrt(10.0 * 10.0 + 20.0 * 20.0));

    path_gain(0) = 1;
    path_gain(1) = 0.25;

    path_length(0) = 20.0;
    path_length(1) = std::sqrt(10.0 * 10.0 + 10.0 * 10.0) + std::sqrt(10.0 * 10.0 + 20.0 * 20.0 + 10.0 * 10.0);

    arma::fmat M(8, 2);
    M(0, 0) = 1.0;
    M(0, 1) = 1.0;
    M(6, 0) = -1.0;
    M(6, 1) = -1.0;

    // Shortest path length is automatically obtained from FBS / LBS
    arma::fcube coeff_re, coeff_im, delay;
    arma::fvec rx_Doppler;

    quadriga_lib::get_channels_planar<float>(&ant, &ant,
                                             0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                             20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                             &aod, &eod, &aoa, &eoa, &path_gain, &path_length, &M,
                                             &coeff_re, &coeff_im, &delay, 2997924580.0, true, false,
                                             &rx_Doppler);

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
    float d0 = 20.0;
    float e0 = (std::sqrt(9.0 * 9.0 + 10.0 * 10.0) + std::sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
    float e1 = (std::sqrt(9.0 * 9.0 + 10.0 * 10.0) + std::sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));
    float e2 = (std::sqrt(11.0 * 11.0 + 10.0 * 10.0) + std::sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
    float e3 = (std::sqrt(11.0 * 11.0 + 10.0 * 10.0) + std::sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));
    CHECK(std::abs(delay.at(0, 0, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(1, 0, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 1, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(1, 1, 0) - d0 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 0, 1) - e0 / C) < 1.2e-10f);
    CHECK(std::abs(delay.at(1, 0, 1) - e1 / C) < 1.2e-10f);
    CHECK(std::abs(delay.at(0, 1, 1) - e2 / C) < 1.2e-10f);
    CHECK(std::abs(delay.at(1, 1, 1) - e3 / C) < 1.2e-10f);

    // Check Doppler
    float Doppler = std::cos(aoa.at(1)) * std::cos(eoa.at(1));
    REQUIRE(rx_Doppler.n_elem == 2);
    CHECK(std::abs(rx_Doppler.at(0) + 1.0f) < 1.0e-6f);
    CHECK(std::abs(rx_Doppler.at(1) - Doppler) < 1.0e-6f);
}

TEST_CASE("Get Channels Planar - Fake LOS")
{

    float pi = 3.141592653589793f, pid180 = 0.017453292519943f;
    auto ant = quadriga_lib::generate_arrayant_omni<float>();

    // LOS-Path and LBS / FBS are reversed

    arma::fvec aod(2), eod(2), aoa(2), eoa(2), path_gain(2), path_length(2);

    aod(1) = 0.0;
    aod(0) = 90.0 * pid180;

    eod(1) = 0.0;
    eod(0) = 45.0 * pid180;

    aoa(1) = pi;
    aoa(0) = pi - std::atan(10.0 / 20.0);

    eoa(1) = 0.0;
    eoa(0) = std::atan(10.0 / std::sqrt(10.0 * 10.0 + 20.0 * 20.0));

    path_gain(1) = 1;
    path_gain(0) = 0.25;

    path_length(1) = 20.0;
    path_length(0) = std::sqrt(10.0 * 10.0 + 10.0 * 10.0) + std::sqrt(10.0 * 10.0 + 20.0 * 20.0 + 10.0 * 10.0);

    arma::fmat M(8, 2);
    M(0, 0) = 1.0;
    M(0, 1) = 1.0;
    M(6, 0) = -1.0;
    M(6, 1) = -1.0;

    // Shortest path length is automatically obtained from FBS / LBS
    arma::fcube coeff_re, coeff_im, delay;

    quadriga_lib::get_channels_planar<float>(&ant, &ant,
                                             0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                             20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                             &aod, &eod, &aoa, &eoa, &path_gain, &path_length, &M,
                                             &coeff_re, &coeff_im, &delay, 2997924580.0, true, true);

    REQUIRE(coeff_re.n_slices == 3);
    REQUIRE(coeff_im.n_slices == 3);
    REQUIRE(delay.n_slices == 3);

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
    CHECK(std::abs(delay.at(0, 0, 1) - e0 / C) < 1.2e-13f);
    CHECK(std::abs(delay.at(0, 0, 2) - d0 / C) < 1.0e-13f);
}

TEST_CASE("Get Channels Planar - Coupling")
{
    float pi = 3.141592653589793f;

    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    ant.copy_element(0, 2);
    ant.element_pos(1, 0) = 1.0;
    ant.element_pos(1, 2) = -1.0;

    arma::vec aod(1), eod(1), aoa(1), eoa(1), path_gain(1), path_length(1);
    aoa.at(0) = pi;
    path_length(0) = 20.0;
    path_gain(0) = 1;

    arma::mat M(8, 1);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    // Shortest path length is automatically obtained from FBS / LBS
    arma::cube coeff_re, coeff_im, delay;

    // Case 1 : No coupling (identity matrix)
    quadriga_lib::get_channels_planar<double>(&ant, &ant,
                                              0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                              20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                              &aod, &eod, &aoa, &eoa, &path_gain, &path_length, &M,
                                              &coeff_re, &coeff_im, &delay, 0.0, false, true);

    REQUIRE(coeff_re.n_slices == 2);
    REQUIRE(coeff_im.n_slices == 2);
    REQUIRE(delay.n_slices == 2);

    // Possible delay offsets
    double da = (std::sqrt(20.0 * 20.0 + 0.0) - 20.0) / 299792458.0;
    double db = (std::sqrt(20.0 * 20.0 + 0.0) - 20.0) / 299792458.0;

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

    arma::mat dd = delay.slice(0) - T;

    // Case 2: n_ports == n_elements
    ant.coupling_re.zeros();
    ant.coupling_re.col(0).ones();
    ant.coupling_im.reset();

    quadriga_lib::get_channels_planar<double>(&ant, &ant,
                                              0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                              20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                              &aod, &eod, &aoa, &eoa, &path_gain, &path_length, &M,
                                              &coeff_re, &coeff_im, &delay, 0.0, true, false);
    CHECK(coeff_re.n_slices == 1);
    CHECK(coeff_im.n_slices == 1);
    CHECK(delay.n_slices == 1);

    T.zeros();
    CHECK(arma::approx_equal(coeff_im.slice(0), T, "absdiff", 1e-14));

    T.at(0, 0) = 4.0;
    CHECK(arma::approx_equal(coeff_re.slice(0), T, "absdiff", 1e-14));

    T.ones();
    double dx = (4.0 * da + 2.0 * db) / 9.0 + 20.0 / 299792458.0;
    T = T * dx;
    CHECK(arma::approx_equal(delay.slice(0), T, "absdiff", 1e-14));

    // Case 3: n_ports != n_elements
    delay.zeros();
    ant.coupling_re.ones(3, 1);
    ant.coupling_re *= 2.0;

    quadriga_lib::get_channels_planar<double>(&ant, &ant,
                                              0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                              20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                              &aod, &eod, &aoa, &eoa, &path_gain, &path_length, &M,
                                              &coeff_re, &coeff_im, &delay, 0.0, true, false);

    CHECK(coeff_re.n_elem == 1);
    CHECK(coeff_im.n_elem == 1);
    CHECK(delay.n_elem == 1);
    CHECK(aod.n_elem == 1);
    CHECK(eod.n_elem == 1);
    CHECK(aoa.n_elem == 1);

    CHECK(std::abs(coeff_re.at(0, 0, 0) - 16.0) < 1.0e-14f);
    CHECK(std::abs(coeff_im.at(0, 0, 0) - 0.0) < 1.0e-14f);
    CHECK(std::abs(delay.at(0, 0, 0) - dx) < 1.0e-14f);
}
