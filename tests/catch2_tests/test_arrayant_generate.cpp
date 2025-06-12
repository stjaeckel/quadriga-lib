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

TEST_CASE("Generate Arrayant - Minimal test omni")
{
    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    CHECK(ant.name == "omni");
    CHECK(ant.n_elevation() == 181);
    CHECK(ant.n_azimuth() == 361);

    auto Z = arma::fmat(181, 361, arma::fill::zeros);
    auto O = arma::fmat(181, 361, arma::fill::ones);
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), O, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_re.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_theta_im.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_im.slice(0), Z, "absdiff", 1e-7));

    auto ant2 = quadriga_lib::generate_arrayant_omni<double>(10.0);
    CHECK(ant2.name == "omni");
    CHECK(ant2.n_elevation() == 19);
    CHECK(ant2.n_azimuth() == 37);
}

TEST_CASE("Generate Arrayant - Minimal test xpol")
{
    auto ant = quadriga_lib::generate_arrayant_xpol<float>();
    CHECK(ant.name == "xpol");
    CHECK(ant.n_elevation() == 181);
    CHECK(ant.n_azimuth() == 361);
    CHECK(ant.n_elements() == 2);

    auto Z = arma::fmat(181, 361, arma::fill::zeros);
    auto O = arma::fmat(181, 361, arma::fill::ones);
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), O, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_theta_re.slice(1), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_re.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_re.slice(1), O, "absdiff", 1e-7));

    CHECK(arma::approx_equal(ant.e_theta_im.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_theta_im.slice(1), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_im.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_im.slice(1), Z, "absdiff", 1e-7));

    auto ant2 = quadriga_lib::generate_arrayant_xpol<double>(10.0);
    CHECK(ant2.n_elevation() == 19);
    CHECK(ant2.n_azimuth() == 37);
}

TEST_CASE("Generate Arrayant - Minimal test dipole")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<float>();
    float directivity = ant.calc_directivity_dBi(0);
    CHECK(std::abs(directivity - 1.760964f) < 0.0001f);
    REQUIRE_THROWS_AS(ant.calc_directivity_dBi(1), std::invalid_argument);

    auto ant2 = quadriga_lib::generate_arrayant_dipole<double>(5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = ant2.calc_directivity_dBi(0);
    CHECK(std::abs(directivity2 - 1.760964) < 0.01f);
}

TEST_CASE("Generate Arrayant - Minimal test Half-wave dipole")
{
    auto ant = quadriga_lib::generate_arrayant_half_wave_dipole<float>();
    float directivity = ant.calc_directivity_dBi(0);
    CHECK(std::abs(directivity - 2.15f) < 0.001f);

    auto ant2 = quadriga_lib::generate_arrayant_half_wave_dipole<double>(5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = ant2.calc_directivity_dBi(0);
    CHECK(std::abs(directivity2 - 2.15) < 0.01f);
}

TEST_CASE("Generate Arrayant - Custom")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(10.0, 10.0);
    float directivity = ant.calc_directivity_dBi(0);
    CHECK(std::abs(directivity - 25.627f) < 0.001f);

    auto ant2 = quadriga_lib::generate_arrayant_custom<double>(10.0, 10.0, 0.0, 5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = ant2.calc_directivity_dBi(0);
    CHECK(std::abs(directivity2 - 25.62) < 0.01f);
}

TEST_CASE("Generate Arrayant - 3GPP")
{
    // Single element pattern
    auto ant = quadriga_lib::generate_arrayant_3GPP<float>();
    CHECK(ant.n_elements() == 1);
    REQUIRE(ant.n_azimuth() == 361);
    REQUIRE(ant.n_elevation() == 181);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 0) - 2.51188f) < 1.0e-5);
    CHECK(std::abs(ant.e_theta_re.at(0, 0, 0) - 0.0794328f) < 1.0e-5);

    // Low-Res
    ant = quadriga_lib::generate_arrayant_3GPP<float>(1, 1, 299792458.0, 1, 0.0, 0.5, 1, 1, 0.5, 0.5, nullptr, 10.0);
    CHECK(ant.n_elements() == 1);
    REQUIRE(ant.n_azimuth() == 37);
    REQUIRE(ant.n_elevation() == 19);
    CHECK(std::abs(ant.e_theta_re.at(9, 18, 0) - 2.51188f) < 1.0e-5);
    CHECK(std::abs(ant.e_theta_re.at(0, 0, 0) - 0.0794328f) < 1.0e-5);

    // H/V Polarization
    ant = quadriga_lib::generate_arrayant_3GPP<float>(1, 1, 3e8, 2);
    REQUIRE(ant.n_elements() == 2);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 0) - 2.51188f) < 1.0e-5);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 1)) < 1.0e-5);
    CHECK(std::abs(ant.e_phi_re.at(90, 180, 0)) < 1.0e-5);
    CHECK(std::abs(ant.e_phi_re.at(90, 180, 1) - 2.51188f) < 1.0e-5);

    // 45 deg polarizaion
    ant = quadriga_lib::generate_arrayant_3GPP<float>(1, 1, 3e8, 3);
    REQUIRE(ant.n_elements() == 2);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 0) - 1.776172f) < 1.0e-5);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 1) - 1.776172f) < 1.0e-5);
    CHECK(std::abs(ant.e_phi_re.at(90, 180, 0) - 1.776172f) < 1.0e-5);
    CHECK(std::abs(ant.e_phi_re.at(90, 180, 1) + 1.776172f) < 1.0e-5);

    // Custom pattern
    auto pattern = quadriga_lib::generate_arrayant_custom<float>(90.0, 90.0, 0.0);
    pattern.copy_element(0, 1);
    pattern.copy_element(0, 2);

    float gain = pattern.e_theta_re.at(90, 180, 0);
    gain = std::sqrt(2.0 * gain * gain);

    ant = quadriga_lib::generate_arrayant_3GPP<float>(2, 2, pattern.center_frequency,
                                                      4, 0.0, 0.5, 1, 1, 0.0, 0.0, &pattern);
    CHECK(ant.n_elements() == 6);

    CHECK(std::abs(ant.e_theta_re.at(90, 180, 0) - gain) < 1.0e-5);
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), ant.e_theta_re.slice(5), "absdiff", 1e-7));

    CHECK(std::abs(ant.element_pos.at(1, 0) + 0.25) < 1.0e-5);
    CHECK(std::abs(ant.element_pos.at(1, 1) + 0.25) < 1.0e-5);
    CHECK(std::abs(ant.element_pos.at(1, 2) + 0.25) < 1.0e-5);
    CHECK(std::abs(ant.element_pos.at(1, 3) - 0.25) < 1.0e-5);
    CHECK(std::abs(ant.element_pos.at(1, 4) - 0.25) < 1.0e-5);
    CHECK(std::abs(ant.element_pos.at(1, 5) - 0.25) < 1.0e-5);
}

TEST_CASE("Generate Arrayant - Multi-Beam")
{
    arma::vec az, el, weight;
    double freq = 3.75e9;

    // Generate a pattern with 2 weighted beams
    az = {20.0, 0.0};
    el = {-7.0, 30.0};
    weight = {2.0, 1.0};
    auto ant = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, weight, freq, 1, 0.4, 120.0, 120.0, 0.0, 15.0);

    CHECK(ant.n_elements() == 36);
    CHECK(ant.n_elevation() == 13);
    CHECK(ant.n_azimuth() == 25);
    CHECK(ant.n_ports() == 1);

    arma::vec azg = arma::regspace<arma::vec>(-180.0, 1.0, 180.0) * 0.017453292519943;
    arma::vec elg = arma::regspace<arma::vec>(-90.0, 1.0, 90.0) * 0.017453292519943;

    auto comb = ant.combine_pattern(&azg, &elg);

    CHECK(comb.n_elements() == 1);
    CHECK(comb.n_elevation() == 181);
    CHECK(comb.n_azimuth() == 361);

    double a = comb.e_theta_re(83, 199, 0), b = comb.e_theta_im(83, 199, 0);
    double gain = 10.0 * std::log10(a * a + b * b);

    CHECK(std::abs(gain - 35.6992) < 0.0001);
    CHECK(std::abs(a + 34.7096) < 0.01);
    CHECK(std::abs(b + 50.0993) < 0.01);

    // Generate the same pattern from 3 unweighted beams
    az = {20.0, 0.0, 20.0};
    el = {-7.0, 30.0, -7.0};
    auto ant2 = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, {}, freq, 1, 0.4, 120.0, 120.0, 0.0, 15.0);
    auto comb2 = ant2.combine_pattern(&azg, &elg);

    CHECK(arma::approx_equal(ant.coupling_re, ant2.coupling_re, "absdiff", 1e-6));
    CHECK(arma::approx_equal(ant.coupling_im, ant2.coupling_im, "absdiff", 1e-6));

    CHECK(arma::approx_equal(comb.e_theta_re, comb2.e_theta_re, "absdiff", 1e-6));
    CHECK(arma::approx_equal(comb.e_theta_im, comb2.e_theta_im, "absdiff", 1e-6));

    // Test if generating the patterns at 1 de resolution generates similar results as at 10 degree
    auto comb3 = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, {}, freq, 1, 0.4, 120.0, 120.0, 0.0, 1.0, false, true);

    CHECK(comb3.n_elements() == 1);
    CHECK(comb3.n_elevation() == 181);
    CHECK(comb3.n_azimuth() == 361);

    a = comb3.e_theta_re(83, 199, 0), b = comb3.e_theta_im(83, 199, 0);
    gain = 10.0 * std::log10(a * a + b * b);

    CHECK(std::abs(gain - 35.6992) < 0.1);
    CHECK(std::abs(a + 34.7096) < 0.5);
    CHECK(std::abs(b + 50.0993) < 0.5);

    // Test generation of separate beams
    auto ant4 = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, {}, freq, 1, 0.4, 120.0, 120.0, 0.0, 15.0, true);

    CHECK(ant4.n_elements() == 36);
    CHECK(ant4.n_elevation() == 13);
    CHECK(ant4.n_azimuth() == 25);
    CHECK(ant4.n_ports() == 3);

    // H/V Polarization
    auto ant5 = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, {}, freq, 2, 0.4, 120.0, 120.0, 0.0, 10.0);

    CHECK(ant5.n_elements() == 72);
    CHECK(ant5.n_elevation() == 19);
    CHECK(ant5.n_azimuth() == 37);
    CHECK(ant5.n_ports() == 2);

    arma::mat x = ant5.element_pos.submat(0, 0, 2, 35);
    arma::mat y = ant5.element_pos.submat(0, 36, 2, 71);
    CHECK(arma::approx_equal(x, y, "absdiff", 1e-14));

    x = ant5.coupling_re.submat(0, 0, 35, 0);
    y = ant5.coupling_re.submat(36, 1, 71, 1);
    CHECK(arma::approx_equal(x, y, "absdiff", 1e-14));

    x.zeros();
    y = ant5.coupling_re.submat(0, 1, 35, 1);
    CHECK(arma::approx_equal(x, y, "absdiff", 1e-14));

    y = ant5.coupling_re.submat(36, 0, 71, 0);
    CHECK(arma::approx_equal(x, y, "absdiff", 1e-14));
}
