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