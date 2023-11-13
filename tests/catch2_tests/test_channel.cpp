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
#include <iostream>

#include "quadriga_lib.hpp"

TEST_CASE("Channel - Effective Path Gain")
{
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat(3, 1);
    c.tx_pos(2, 0) = 1.0;

    c.rx_pos = arma::mat(3, 2);
    c.rx_pos(0, 0) = 20.0;
    c.rx_pos(1, 0) = 0.0;
    c.rx_pos(2, 0) = 1.0;
    c.rx_pos(0, 1) = 20.0;
    c.rx_pos(1, 1) = 1.0;
    c.rx_pos(2, 1) = 1.0;

    // Random polarization
    c.path_polarization.push_back(arma::mat(8, 1, arma::fill::randn));
    c.path_polarization.push_back(arma::mat(8, 1, arma::fill::randn));

    // Calc effective PG and check results
    auto p = c.calc_effective_path_gain();
    arma::vec q1 = c.path_polarization[0] % c.path_polarization[0];
    arma::vec q2 = c.path_polarization[1] % c.path_polarization[1];
    CHECK(std::abs(arma::sum(q1) - p.at(0)) < 1e-14);
    CHECK(std::abs(arma::sum(q2) - p.at(1)) < 1e-14);

    // Add a path length
    arma::Col<double> l = {1.0};
    c.path_length.push_back(l);
    l = {2.0};
    c.path_length.push_back(l);

    // Check additional gain
    p = c.calc_effective_path_gain();
    double a1 = std::pow(10.0, -3.245);
    double a2 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(2.0)));
    CHECK(std::abs(a1 * arma::sum(q1) - p.at(0)) < 1e-14);
    CHECK(std::abs(a2 * arma::sum(q2) - p.at(1)) < 1e-14);

    // Set same frequency for all snapshots
    c.center_frequency = 10.0e9;

    // Check additional gain
    p = c.calc_effective_path_gain();
    a1 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(10.0)));
    a2 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(10.0) - 20.0 * std::log10(2.0)));
    CHECK(std::abs(a1 * arma::sum(q1) - p.at(0)) < 1e-19);
    CHECK(std::abs(a2 * arma::sum(q2) - p.at(1)) < 1e-19);

    // Sett different frequencies for snapshots
    c.center_frequency = {12.0e9, 20.0e9};

    // Check gain
    p = c.calc_effective_path_gain();
    a1 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(12.0)));
    a2 = std::pow(10.0, 0.1 * (-32.45 - 20.0 * std::log10(20.0) - 20.0 * std::log10(2.0)));
    CHECK(std::abs(a1 * arma::sum(q1) - p.at(0)) < 1e-19);
    CHECK(std::abs(a2 * arma::sum(q2) - p.at(1)) < 1e-19);

    // Set a path gain, this disables FSPL calculation
    l = {0.1};
    c.path_gain.push_back(l);
    l = {0.2};
    c.path_gain.push_back(l);

    // Check gain
    p = c.calc_effective_path_gain();
    CHECK(std::abs(0.1 * arma::sum(q1) - p.at(0)) < 1e-14);
    CHECK(std::abs(0.2 * arma::sum(q2) - p.at(1)) < 1e-14);

    // Calculate coefficients
    arma::Cube<double> coeff_re, coeff_im, delay;
    auto ant = quadriga_lib::generate_arrayant_xpol<double>();

    l = {0.0}; // Angles
    quadriga_lib::get_channels_planar(&ant, &ant,
                                      0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                      c.rx_pos(0, 0), c.rx_pos(1, 0), c.rx_pos(2, 0),
                                      0.0, 0.0, 0.0, &l, &l, &l, &l,
                                      &c.path_gain[0], &c.path_length[0], &c.path_polarization[0],
                                      &coeff_re, &coeff_im, &delay);

    c.coeff_re.push_back(coeff_re);
    c.coeff_im.push_back(coeff_im);
    c.delay.push_back(delay);

    l = {0.1}; // Angles
    quadriga_lib::get_channels_planar(&ant, &ant,
                                      0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                      c.rx_pos(0, 1), c.rx_pos(1,1), c.rx_pos(2, 1),
                                      0.0, 0.0, 0.0, &l, &l, &l, &l,
                                      &c.path_gain[1], &c.path_length[1], &c.path_polarization[1],
                                      &coeff_re, &coeff_im, &delay);

    c.coeff_re.push_back(coeff_re);
    c.coeff_im.push_back(coeff_im);
    c.delay.push_back(delay);

    c.path_polarization.clear();
    c.path_length.clear();
    p.reset();

    p = c.calc_effective_path_gain();
    CHECK(std::abs(0.1 * arma::sum(q1) - p.at(0)) < 1e-14);
    CHECK(std::abs(0.2 * arma::sum(q2) - p.at(1)) < 1e-14);

    CHECK(c.is_valid().empty());
}