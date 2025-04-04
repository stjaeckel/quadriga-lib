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

#include <catch2/catch_test_macros.hpp>
#include "quadriga_lib.hpp"

TEST_CASE("Get Channels IRS - Path loss test")
{
    auto ant = quadriga_lib::generate_arrayant_omni<float>();

    arma::fmat fbs_pos_1(3, 2); //     FBS @ 11 m height
    fbs_pos_1(0, 0) = 10.0;     //      |
    fbs_pos_1(2, 0) = 1.0;      //     10 m
    fbs_pos_1(1, 1) = 10.0;     //      |           FBS LOS @ x = 10 m
    fbs_pos_1(2, 1) = 11.0;     //     TX --------------d = 20 m------------------ IRS

    arma::fmat fbs_pos_2(3, 2); //     IRS -------------d = 20 m------------------ RX
    fbs_pos_2(0, 0) = 30.0;     //                  FBS LOS @ x = 20 m             |
    fbs_pos_2(2, 0) = 1.0;      //                                                20 m
    fbs_pos_2(0, 1) = 40.0;     //                                                 |
    fbs_pos_2(1, 1) = -20.0;    //                                                FBS
    fbs_pos_2(2, 1) = 1.0;

    // Shortest path length is automatically obtained from FBS / LBS
    arma::fvec path_gain = {0.2, 0.1};
    arma::fvec path_length(2);

    arma::fmat M(8, 2);
    M(0, 0) = 1.0;
    M(0, 1) = 1.0;
    M(6, 0) = -1.0;
    M(6, 1) = -1.0;

    arma::fcube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    quadriga_lib::get_channels_irs<float>(&ant, &ant, &ant,
                                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          40.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          &fbs_pos_1, &fbs_pos_1, &path_gain, &path_length, &M,
                                          &fbs_pos_2, &fbs_pos_2, &path_gain, &path_length, &M,
                                          &coeff_re, &coeff_im, &delay);

    // Phase calculation is disabled for f = 0
    // IRS has no gain (omni)
    // Amplitude should be A = sqrt(G1) * sqrt(G2) = sqrt( G1 * G2 )

    CHECK(std::abs(coeff_re.at(0, 0, 0) - 0.2f) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(0, 0, 1) - 0.1414213562373095f) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(0, 0, 2) - 0.1414213562373095f) < 1.0e-6f);
    CHECK(std::abs(coeff_re.at(0, 0, 3) - 0.1f) < 1.0e-6f);

    CHECK(std::abs(coeff_im.at(0, 0, 0)) < 1.0e-6f); // No phase
    CHECK(std::abs(coeff_im.at(0, 0, 1)) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(0, 0, 2)) < 1.0e-6f);
    CHECK(std::abs(coeff_im.at(0, 0, 3)) < 1.0e-6f);

    // Check delays
    float C = 299792458.0;
    float d_TX_IRS_RX = 40.0;
    float d_TX_IRS = 20.0;
    float d_IRS_RX = 20.0;
    float d_TX_FBS1_IRS = std::sqrt(10.0 * 10.0 + 10.0 * 10.0) + std::sqrt(10.0 * 10.0 + 20.0 * 20.0 + 10.0 * 10.0);
    float d_IRS_FBS2_RX = std::sqrt(20.0 * 20.0 + 20.0 * 20.0) + 20.0;

    float d1 = d_TX_IRS + d_IRS_FBS2_RX - d_TX_IRS_RX;
    float d2 = d_TX_FBS1_IRS + d_IRS_RX - d_TX_IRS_RX;
    float d3 = d_TX_FBS1_IRS + d_IRS_FBS2_RX - d_TX_IRS_RX;

    CHECK(std::abs(delay.at(0, 0, 0)) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 0, 1) - d1 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 0, 2) - d2 / C) < 1.0e-13f);
    CHECK(std::abs(delay.at(0, 0, 3) - d3 / C) < 1.0e-13f);
}

TEST_CASE("Get Channels IRS - Minimal test")
{
    double lambda = 0.1;
    double C = 299792458.0;
    double f0 = C / lambda;

    auto ant = quadriga_lib::generate_arrayant_omni<double>();

    ant.copy_element(0, 1);
    ant.copy_element(0, 2);
    ant.element_pos(1, 0) = lambda * 0.5;
    ant.element_pos(1, 2) = -lambda * 0.5;
    ant.coupling_im.reset();

    // Gain in broadside direction: 4.77 dBi, factor 3
    auto irs = ant.copy();
    irs.coupling_re.ones(3, 2);
    irs.coupling_re.col(1).zeros();

    arma::mat fbs_pos_1(3, 2); //     FBS @ 11 m height
    fbs_pos_1(0, 0) = 10.0;    //      |
    fbs_pos_1(2, 0) = 1.0;     //     10 m
    fbs_pos_1(1, 1) = 10.0;    //      |           FBS LOS @ x = 10 m
    fbs_pos_1(2, 1) = 11.0;    //     TX --------------d = 20 m------------------ IRS

    arma::mat fbs_pos_2(3, 2); //     IRS -------------d = 20 m------------------ RX
    fbs_pos_2(0, 0) = 30.0;    //                  FBS LOS @ x = 20 m             |
    fbs_pos_2(2, 0) = 1.0;     //                                                20 m
    fbs_pos_2(0, 1) = 40.0;    //                                                 |
    fbs_pos_2(1, 1) = -20.0;   //                                                FBS
    fbs_pos_2(2, 1) = 1.0;

    // Shortest path length is automatically obtained from FBS / LBS
    arma::vec path_gain = {0.2, 0.1};
    arma::vec path_length(2);

    arma::mat M(8, 2);
    M(0, 0) = 1.0;
    M(0, 1) = 1.0;
    M(6, 0) = -1.0;
    M(6, 1) = -1.0;

    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;

    auto keep = quadriga_lib::get_channels_irs<double>(&ant, &ant, &irs,
                                                       0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                       40.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                       20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                       &fbs_pos_1, &fbs_pos_1, &path_gain, &path_length, &M,
                                                       &fbs_pos_2, &fbs_pos_2, &path_gain, &path_length, &M,
                                                       &coeff_re, &coeff_im, &delay, 0, -6.0, f0, 1, &aod, &eod, &aoa, &eoa);

    // First path should be close to 1.8, 0.2 path gain * 3 * 3 (IRS array gain)
    arma::mat T(3, 3, arma::fill::value(1.8));
    CHECK(arma::approx_equal(coeff_re.slice(0), T, "absdiff", 1e-3));

    T.zeros();
    CHECK(arma::approx_equal(coeff_im.slice(0), T, "absdiff", 0.05));

    T.fill(40.0 / C);
    CHECK(arma::approx_equal(delay.slice(0), T, "absdiff", 1e-6));

    CHECK(coeff_re.n_elem == 3 * 3 * 2);
    CHECK(coeff_im.n_elem == 3 * 3 * 2);
    CHECK(delay.n_elem == 3 * 3 * 2);
    CHECK(aod.n_elem == 3 * 3 * 2);
    CHECK(eod.n_elem == 3 * 3 * 2);
    CHECK(aoa.n_elem == 3 * 3 * 2);
    CHECK(eoa.n_elem == 3 * 3 * 2);

    CHECK(keep.size() == 4);
    CHECK(keep[0]);
    CHECK(!keep[1]);
    CHECK(keep[2]);
    CHECK(!keep[3]);

    // Second codebook contains only zeros - should create no paths
    quadriga_lib::get_channels_irs<double>(&ant, &ant, &irs,
                                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                           40.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                           20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                           &fbs_pos_1, &fbs_pos_1, &path_gain, &path_length, &M,
                                           &fbs_pos_2, &fbs_pos_2, &path_gain, &path_length, &M,
                                           &coeff_re, &coeff_im, &delay, 1, -6.0, f0, 1, &aod, &eod, &aoa, &eoa);

    CHECK(coeff_re.n_elem == 0);
    CHECK(coeff_im.n_elem == 0);
    CHECK(delay.n_elem == 0);
    CHECK(aod.n_elem == 0);
    CHECK(eod.n_elem == 0);
    CHECK(aoa.n_elem == 0);
    CHECK(eoa.n_elem == 0);
}

TEST_CASE("Tools - Combine IRS Coord")
{
    double Ix = 10.0, Iy = 10.0, Iz = 0.0;

    arma::u32_vec no_interact_1 = {2, 0};
    arma::mat interact_coord_1 = {{2.0, 8.0}, {2.0, 2.0}, {2.0, 2.0}};

    arma::u32_vec no_interact_2 = {0, 2};
    arma::mat interact_coord_2 = {{12.0, 12.0}, {2.0, 8.0}, {-2.0, -2.0}};

    arma::u32_vec no_interact;
    arma::mat interact_coord;

    std::vector active_path(3, true);
    REQUIRE_THROWS_AS(quadriga_lib::combine_irs_coord(Ix, Iy, Iz,
                                                      &no_interact_1, &interact_coord_1, &no_interact_2, &interact_coord_2,
                                                      &no_interact, &interact_coord, 1, 1, &active_path),
                      std::invalid_argument);

    active_path = std::vector(4, true);
    quadriga_lib::combine_irs_coord(Ix, Iy, Iz,
                                    &no_interact_1, &interact_coord_1, &no_interact_2, &interact_coord_2,
                                    &no_interact, &interact_coord);

    CHECK(no_interact.n_elem == 4);
    CHECK(no_interact(0) == 3);
    CHECK(no_interact(1) == 5);
    CHECK(no_interact(2) == 1);
    CHECK(no_interact(3) == 3);

    arma::mat T = {{2, 8, 10, 2, 8, 10, 12, 12, 10, 10, 12, 12},
                   {2, 2, 10, 2, 2, 10, 2, 8, 10, 10, 2, 8},
                   {2, 2, 0, 2, 2, 0, -2, -2, 0, 0, -2, -2}};
    CHECK(arma::approx_equal(interact_coord, T, "absdiff", 1e-14));

    // Reverse first segment
    quadriga_lib::combine_irs_coord(Ix, Iy, Iz,
                                    &no_interact_1, &interact_coord_1, &no_interact_2, &interact_coord_2,
                                    &no_interact, &interact_coord, 1, 0);

    T = {{8, 2, 10, 8, 2, 10, 12, 12, 10, 10, 12, 12},
         {2, 2, 10, 2, 2, 10, 2, 8, 10, 10, 2, 8},
         {2, 2, 0, 2, 2, 0, -2, -2, 0, 0, -2, -2}};
    CHECK(arma::approx_equal(interact_coord, T, "absdiff", 1e-14));

    // Reverse both segment2
    quadriga_lib::combine_irs_coord(Ix, Iy, Iz,
                                    &no_interact_1, &interact_coord_1, &no_interact_2, &interact_coord_2,
                                    &no_interact, &interact_coord, 1, 1);

    T = {{8, 2, 10, 8, 2, 10, 12, 12, 10, 10, 12, 12},
         {2, 2, 10, 2, 2, 10, 8, 2, 10, 10, 8, 2},
         {2, 2, 0, 2, 2, 0, -2, -2, 0, 0, -2, -2}};
    CHECK(arma::approx_equal(interact_coord, T, "absdiff", 1e-14));

    // Disable first path
    active_path[0] = false;
    quadriga_lib::combine_irs_coord(Ix, Iy, Iz,
                                    &no_interact_1, &interact_coord_1, &no_interact_2, &interact_coord_2,
                                    &no_interact, &interact_coord, 0, 1, &active_path);

    CHECK(no_interact.n_elem == 3);
    CHECK(no_interact(0) == 5);
    CHECK(no_interact(1) == 1);
    CHECK(no_interact(2) == 3);

    T = {{2, 8, 10, 12, 12, 10, 10, 12, 12},
         {2, 2, 10, 8, 2, 10, 10, 8, 2},
         {2, 2, 0, -2, -2, 0, 0, -2, -2}};
    CHECK(arma::approx_equal(interact_coord, T, "absdiff", 1e-14));

    // Disable first path
    active_path[1] = false;
    active_path[3] = false;
    quadriga_lib::combine_irs_coord(Ix, Iy, Iz,
                                    &no_interact_1, &interact_coord_1, &no_interact_2, &interact_coord_2,
                                    &no_interact, &interact_coord, 0, 0, &active_path);

    CHECK(no_interact.n_elem == 1);
    CHECK(no_interact(0) == 1);

    T = {10, 10, 0};
    CHECK(arma::approx_equal(interact_coord, T.t(), "absdiff", 1e-14));
}

TEST_CASE("Channel - Add Paths")
{
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat(3, 1);
    c.tx_pos(2, 0) = 1.0;

    c.rx_pos = arma::mat(3, 2); // 2 sanpshots
    c.rx_pos(0, 0) = 20.0;
    c.rx_pos(1, 0) = 0.0;
    c.rx_pos(2, 0) = 1.0;
    c.rx_pos(0, 1) = 20.0;
    c.rx_pos(1, 1) = 1.0;
    c.rx_pos(2, 1) = 1.0;

    // Adding nothing to nothing should work
    c.add_paths(0);

    auto coeff_new = arma::cube(2, 3, 5, arma::fill::randn);
    auto coeff_alt = arma::cube(1, 1, 5, arma::fill::randn);

    // Test out of bound
    REQUIRE_THROWS_AS(c.add_paths(2), std::invalid_argument);

    // Channel object requires exisiting MIMO coefficients
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_new, &coeff_new), std::invalid_argument);

    // Add coefficients to channel
    c.coeff_re.push_back(arma::cube(2, 3, 4, arma::fill::randn));
    c.coeff_re.push_back(arma::cube(2, 3, 4, arma::fill::randn));
    c.coeff_im.push_back(arma::cube(2, 3, 4, arma::fill::randn));
    c.coeff_im.push_back(arma::cube(2, 3, 4, arma::fill::randn));
    c.delay.push_back(arma::cube(2, 3, 4, arma::fill::randn));
    c.delay.push_back(arma::cube(2, 3, 4, arma::fill::randn));

    // The channel object has coefficients, but no coefficients are added
    REQUIRE_THROWS_AS(c.add_paths(0), std::invalid_argument);

    // Test incomplete coeff
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new), std::invalid_argument);
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_new), std::invalid_argument);
    REQUIRE_THROWS_AS(c.add_paths(0, nullptr, &coeff_new), std::invalid_argument);
    REQUIRE_THROWS_AS(c.add_paths(0, nullptr, nullptr, &coeff_new), std::invalid_argument);
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, nullptr, &coeff_new), std::invalid_argument);

    // Test wrong MIMO dimensions
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_alt, &coeff_new, &coeff_new), std::invalid_argument);
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_alt, &coeff_new), std::invalid_argument);
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_new, &coeff_alt), std::invalid_argument);

    // Add 5 additional paths to snapshot 1
    c.add_paths(0, &coeff_new, &coeff_new, &coeff_new);
    CHECK(c.is_valid().size() == 0);
    auto n_paths = c.n_path();
    CHECK(n_paths[0] == 9ULL);
    CHECK(n_paths[1] == 4ULL);

    // Test compact delays
    c.delay.clear();
    c.delay.push_back(arma::cube(1, 1, 9, arma::fill::randn));
    c.delay.push_back(arma::cube(1, 1, 4, arma::fill::randn));
    c.add_paths(1, &coeff_new, &coeff_new, &coeff_alt);
    CHECK(c.is_valid().size() == 0);
    n_paths = c.n_path();
    CHECK(n_paths[0] == 9ULL);
    CHECK(n_paths[1] == 9ULL);

    // Create IRS
    arma::u32_vec no_interact_irs = {1, 2, 3};
    arma::mat interact_coord_irs = arma::mat(3, 6, arma::fill::randn);
    auto coeff_irs = arma::cube(2, 3, 3, arma::fill::randn);
    auto delay_irs = arma::cube(1, 1, 3, arma::fill::randn);

    // Try to add IRS to channel without coords
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_irs, &coeff_irs, &delay_irs, &no_interact_irs, &interact_coord_irs), std::invalid_argument);

    // Add interaction coordinates to channel
    arma::u32_vec no_interact_add = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    c.no_interact.push_back(no_interact_add);
    c.no_interact.push_back(no_interact_add);
    auto sum_no_interact = (arma::uword)arma::sum(no_interact_add);
    c.interact_coord.push_back(arma::mat(3, sum_no_interact, arma::fill::randn));
    c.interact_coord.push_back(c.interact_coord[0]);

    // Missing Coord
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_new, &coeff_alt, &no_interact_irs), std::invalid_argument);
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_new, &coeff_alt, nullptr, &interact_coord_irs), std::invalid_argument);

    // Mismatching number of paths
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_new, &coeff_alt, &no_interact_irs, &interact_coord_irs), std::invalid_argument);

    // Correct
    c.add_paths(0, &coeff_irs, &coeff_irs, &delay_irs, &no_interact_irs, &interact_coord_irs);
    CHECK(c.is_valid().size() == 0);
    CHECK(arma::approx_equal(c.interact_coord[0].cols(0, sum_no_interact - 1), c.interact_coord[1], "absdiff", 1e-14));
    CHECK(arma::approx_equal(c.interact_coord[0].cols(sum_no_interact, sum_no_interact + 5), interact_coord_irs, "absdiff", 1e-14));
    n_paths = c.n_path();
    CHECK(n_paths[0] == 12ULL);
    CHECK(n_paths[1] == 9ULL);

    // Adding coords without coefficients should fail
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, &no_interact_irs, &interact_coord_irs), std::invalid_argument);

    // Add coords to a channel without coefficients
    c.coeff_re.clear();
    c.coeff_im.clear();
    c.delay.clear();
    REQUIRE_THROWS_AS(c.add_paths(1, &coeff_irs, &coeff_irs, &delay_irs, &no_interact_irs, &interact_coord_irs), std::invalid_argument);
    c.add_paths(1, nullptr, nullptr, nullptr, &no_interact_irs, &interact_coord_irs);
    CHECK(c.is_valid().size() == 0);
    CHECK(arma::approx_equal(c.interact_coord[1].cols(sum_no_interact, sum_no_interact + 5), interact_coord_irs, "absdiff", 1e-14));
    n_paths = c.n_path();
    CHECK(n_paths[0] == 12ULL);
    CHECK(n_paths[1] == 12ULL);

    // Add path gain
    auto path_gain = arma::vec(3, arma::fill::randn);
    c.path_gain.push_back(arma::vec(12, arma::fill::randn));
    c.path_gain.push_back(arma::vec(12, arma::fill::randn));
    c.add_paths(0, nullptr, nullptr, nullptr, &no_interact_irs, &interact_coord_irs, &path_gain);
    CHECK(c.is_valid().size() == 0);
    n_paths = c.n_path();
    CHECK(n_paths[0] == 15ULL);
    CHECK(n_paths[1] == 12ULL);

    // The channel object has 'path_gain', but no 'path_gain_add' are added!
    REQUIRE_THROWS_AS(c.add_paths(0, nullptr, nullptr, nullptr, &no_interact_irs, &interact_coord_irs), std::invalid_argument);

    // Add path_length
    c.no_interact.clear();
    c.interact_coord.clear();
    auto path_length = arma::vec(1, arma::fill::randn);

    // Channel has no path length
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, &path_length, &path_length), std::invalid_argument);
    c.path_length.push_back(arma::vec(15, arma::fill::randn));
    c.path_length.push_back(arma::vec(12, arma::fill::randn));

    // Path gain and path lenght mismatch
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, &path_gain, &path_length), std::invalid_argument);

    // No path length provided (but channel has it)
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, &path_length), std::invalid_argument);

    // OK
    c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, &path_length, &path_length);
    CHECK(c.is_valid().size() == 0);
    n_paths = c.n_path();
    CHECK(n_paths[0] == 15ULL);
    CHECK(n_paths[1] == 13ULL);

    // Add path_polarization
    c.path_gain.clear();
    auto path_polarization = arma::mat(8, 1, arma::fill::randn);

    // Channel has no path_polarization
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_length, &path_polarization), std::invalid_argument);
    c.path_polarization.push_back(arma::mat(8, 15, arma::fill::randn));
    c.path_polarization.push_back(arma::mat(8, 13, arma::fill::randn));

    // Mismatch
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_gain, &path_polarization), std::invalid_argument);

    // No path length provided (but channel has it)
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_length), std::invalid_argument);

    // OK
    c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_length, &path_polarization);
    CHECK(c.is_valid().size() == 0);
    n_paths = c.n_path();
    CHECK(n_paths[0] == 15ULL);
    CHECK(n_paths[1] == 14ULL);

    // Add path_angles
    c.path_length.clear();
    path_polarization = arma::mat(8, 2, arma::fill::randn);
    auto path_angles = arma::mat(2, 4, arma::fill::randn);

    // Channel has no path_polarization
    REQUIRE_THROWS_AS(c.add_paths(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_polarization, &path_angles), std::invalid_argument);
    c.path_angles.push_back(arma::mat(15, 4, arma::fill::randn));
    c.path_angles.push_back(arma::mat(14, 4, arma::fill::randn));

    // No path length provided (but channel has it)
    REQUIRE_THROWS_AS(c.add_paths(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_polarization), std::invalid_argument);

    // OK
    c.add_paths(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_polarization, &path_angles);
    CHECK(c.is_valid().size() == 0);
    n_paths = c.n_path();
    CHECK(n_paths[0] == 17ULL);
    CHECK(n_paths[1] == 14ULL);

    // Add path_fbs_pos
    c.path_polarization.clear();
    auto path_pos = arma::mat(3, 2, arma::fill::randn);

    // Channel has no path_polarization
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_angles, &path_pos), std::invalid_argument);
    c.path_fbs_pos.push_back(arma::mat(3, 17, arma::fill::randn));
    c.path_fbs_pos.push_back(arma::mat(3, 14, arma::fill::randn));

    // No path length provided (but channel has it)
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_angles), std::invalid_argument);

    // OK
    c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_angles, &path_pos);
    CHECK(c.is_valid().size() == 0);
    n_paths = c.n_path();
    CHECK(n_paths[0] == 17ULL);
    CHECK(n_paths[1] == 16ULL);

    // Channel has no path_polarization
    c.path_angles.clear();
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_pos, &path_pos), std::invalid_argument);
    c.path_lbs_pos.push_back(arma::mat(3, 17, arma::fill::randn));
    c.path_lbs_pos.push_back(arma::mat(3, 16, arma::fill::randn));

    // No path length provided (but channel has it)
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_pos), std::invalid_argument);

    // OK
    c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_pos, &path_pos);
    CHECK(c.is_valid().size() == 0);
    n_paths = c.n_path();
    CHECK(n_paths[0] == 17ULL);
    CHECK(n_paths[1] == 18ULL);
}