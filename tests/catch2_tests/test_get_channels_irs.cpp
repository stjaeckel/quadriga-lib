// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
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

    quadriga_lib::get_channels_irs<float>(ant, ant, ant,
                                          {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                          {40.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                          {20.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                          fbs_pos_1, fbs_pos_1, path_gain, path_length, M,
                                          fbs_pos_2, fbs_pos_2, path_gain, path_length, M,
                                          coeff_re, coeff_im, delay);

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

    quadriga_lib::arrayant_copy_element(ant, 0, {1});
    quadriga_lib::arrayant_copy_element(ant, 0, {2});
    ant.element_pos(1, 0) = lambda * 0.5;
    ant.element_pos(1, 2) = -lambda * 0.5;
    ant.coupling_im.reset();

    // Gain in broadside direction: 4.77 dBi, factor 3
    auto irs = ant;
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

    std::vector<bool> keep = {true}; // An old mask is replaced
    quadriga_lib::get_channels_irs<double>(ant, ant, irs,
                                           {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                           {40.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                           {20.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                           fbs_pos_1, fbs_pos_1, path_gain, path_length, M,
                                           fbs_pos_2, fbs_pos_2, path_gain, path_length, M,
                                           coeff_re, coeff_im, delay, 0, -6.0, f0, 1, &aod, &eod, &aoa, &eoa,
                                           {}, {}, &keep);

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
    quadriga_lib::get_channels_irs<double>(ant, ant, irs,
                                           {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                           {40.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                           {20.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                           fbs_pos_1, fbs_pos_1, path_gain, path_length, M,
                                           fbs_pos_2, fbs_pos_2, path_gain, path_length, M,
                                           coeff_re, coeff_im, delay, 1, -6.0, f0, 1, &aod, &eod, &aoa, &eoa);

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

    arma::u32_vec n_interact_1 = {2, 0};
    arma::mat interact_coord_1 = {{2.0, 8.0}, {2.0, 2.0}, {2.0, 2.0}};

    arma::u32_vec n_interact_2 = {0, 2};
    arma::mat interact_coord_2 = {{12.0, 12.0}, {2.0, 8.0}, {-2.0, -2.0}};

    arma::u32_vec n_interact;
    arma::mat interact_coord;

    std::vector active_path(3, true);
    REQUIRE_THROWS_AS(quadriga_lib::combine_irs_coord({Ix, Iy, Iz},
                                                      n_interact_1, interact_coord_1, n_interact_2, interact_coord_2,
                                                      n_interact, interact_coord, 1, 1, active_path),
                      std::invalid_argument);

    active_path = std::vector(4, true);
    quadriga_lib::combine_irs_coord({Ix, Iy, Iz},
                                    n_interact_1, interact_coord_1, n_interact_2, interact_coord_2,
                                    n_interact, interact_coord);

    CHECK(n_interact.n_elem == 4);
    CHECK(n_interact(0) == 3);
    CHECK(n_interact(1) == 5);
    CHECK(n_interact(2) == 1);
    CHECK(n_interact(3) == 3);

    arma::mat T = {{2, 8, 10, 2, 8, 10, 12, 12, 10, 10, 12, 12},
                   {2, 2, 10, 2, 2, 10, 2, 8, 10, 10, 2, 8},
                   {2, 2, 0, 2, 2, 0, -2, -2, 0, 0, -2, -2}};
    CHECK(arma::approx_equal(interact_coord, T, "absdiff", 1e-14));

    // Reverse first segment
    quadriga_lib::combine_irs_coord({Ix, Iy, Iz},
                                    n_interact_1, interact_coord_1, n_interact_2, interact_coord_2,
                                    n_interact, interact_coord, 1, 0);

    T = {{8, 2, 10, 8, 2, 10, 12, 12, 10, 10, 12, 12},
         {2, 2, 10, 2, 2, 10, 2, 8, 10, 10, 2, 8},
         {2, 2, 0, 2, 2, 0, -2, -2, 0, 0, -2, -2}};
    CHECK(arma::approx_equal(interact_coord, T, "absdiff", 1e-14));

    // Reverse both segment2
    quadriga_lib::combine_irs_coord({Ix, Iy, Iz},
                                    n_interact_1, interact_coord_1, n_interact_2, interact_coord_2,
                                    n_interact, interact_coord, 1, 1);

    T = {{8, 2, 10, 8, 2, 10, 12, 12, 10, 10, 12, 12},
         {2, 2, 10, 2, 2, 10, 8, 2, 10, 10, 8, 2},
         {2, 2, 0, 2, 2, 0, -2, -2, 0, 0, -2, -2}};
    CHECK(arma::approx_equal(interact_coord, T, "absdiff", 1e-14));

    // Disable first path
    active_path[0] = false;
    quadriga_lib::combine_irs_coord({Ix, Iy, Iz},
                                    n_interact_1, interact_coord_1, n_interact_2, interact_coord_2,
                                    n_interact, interact_coord, 0, 1, active_path);

    CHECK(n_interact.n_elem == 3);
    CHECK(n_interact(0) == 5);
    CHECK(n_interact(1) == 1);
    CHECK(n_interact(2) == 3);

    T = {{2, 8, 10, 12, 12, 10, 10, 12, 12},
         {2, 2, 10, 8, 2, 10, 10, 8, 2},
         {2, 2, 0, -2, -2, 0, 0, -2, -2}};
    CHECK(arma::approx_equal(interact_coord, T, "absdiff", 1e-14));

    // Disable first path
    active_path[1] = false;
    active_path[3] = false;
    quadriga_lib::combine_irs_coord({Ix, Iy, Iz},
                                    n_interact_1, interact_coord_1, n_interact_2, interact_coord_2,
                                    n_interact, interact_coord, 0, 0, active_path);

    CHECK(n_interact.n_elem == 1);
    CHECK(n_interact(0) == 1);

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
    arma::u32_vec n_interact_irs = {1, 2, 3};
    arma::mat interact_coord_irs = arma::mat(3, 6, arma::fill::randn);
    auto coeff_irs = arma::cube(2, 3, 3, arma::fill::randn);
    auto delay_irs = arma::cube(1, 1, 3, arma::fill::randn);

    // Try to add IRS to channel without coords
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_irs, &coeff_irs, &delay_irs, &n_interact_irs, &interact_coord_irs), std::invalid_argument);

    // Add interaction coordinates to channel
    arma::u32_vec n_interact_add = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    c.n_interact.push_back(n_interact_add);
    c.n_interact.push_back(n_interact_add);
    auto sum_no_interact = (arma::uword)arma::sum(n_interact_add);
    c.interact_coord.push_back(arma::mat(3, sum_no_interact, arma::fill::randn));
    c.interact_coord.push_back(c.interact_coord[0]);

    // Missing Coord
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_new, &coeff_alt, &n_interact_irs), std::invalid_argument);
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_new, &coeff_alt, nullptr, &interact_coord_irs), std::invalid_argument);

    // Mismatching number of paths
    REQUIRE_THROWS_AS(c.add_paths(0, &coeff_new, &coeff_new, &coeff_alt, &n_interact_irs, &interact_coord_irs), std::invalid_argument);

    // Correct
    c.add_paths(0, &coeff_irs, &coeff_irs, &delay_irs, &n_interact_irs, &interact_coord_irs);
    CHECK(c.is_valid().size() == 0);
    CHECK(arma::approx_equal(c.interact_coord[0].cols(0, sum_no_interact - 1), c.interact_coord[1], "absdiff", 1e-14));
    CHECK(arma::approx_equal(c.interact_coord[0].cols(sum_no_interact, sum_no_interact + 5), interact_coord_irs, "absdiff", 1e-14));
    n_paths = c.n_path();
    CHECK(n_paths[0] == 12ULL);
    CHECK(n_paths[1] == 9ULL);

    // Adding coords without coefficients should fail
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, &n_interact_irs, &interact_coord_irs), std::invalid_argument);

    // Add coords to a channel without coefficients
    c.coeff_re.clear();
    c.coeff_im.clear();
    c.delay.clear();
    REQUIRE_THROWS_AS(c.add_paths(1, &coeff_irs, &coeff_irs, &delay_irs, &n_interact_irs, &interact_coord_irs), std::invalid_argument);
    c.add_paths(1, nullptr, nullptr, nullptr, &n_interact_irs, &interact_coord_irs);
    CHECK(c.is_valid().size() == 0);
    CHECK(arma::approx_equal(c.interact_coord[1].cols(sum_no_interact, sum_no_interact + 5), interact_coord_irs, "absdiff", 1e-14));
    n_paths = c.n_path();
    CHECK(n_paths[0] == 12ULL);
    CHECK(n_paths[1] == 12ULL);

    // Add path gain
    auto path_gain = arma::vec(3, arma::fill::randn);
    c.path_gain.push_back(arma::vec(12, arma::fill::randn));
    c.path_gain.push_back(arma::vec(12, arma::fill::randn));
    c.add_paths(0, nullptr, nullptr, nullptr, &n_interact_irs, &interact_coord_irs, &path_gain);
    CHECK(c.is_valid().size() == 0);
    n_paths = c.n_path();
    CHECK(n_paths[0] == 15ULL);
    CHECK(n_paths[1] == 12ULL);

    // The channel object has 'path_gain', but no 'path_gain_add' are added!
    REQUIRE_THROWS_AS(c.add_paths(0, nullptr, nullptr, nullptr, &n_interact_irs, &interact_coord_irs), std::invalid_argument);

    // Add path_length
    c.n_interact.clear();
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
    c.polmat.push_back(arma::mat(8, 15, arma::fill::randn));
    c.polmat.push_back(arma::mat(8, 13, arma::fill::randn));

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

    // Add fbs_pos
    c.polmat.clear();
    auto path_pos = arma::mat(3, 2, arma::fill::randn);

    // Channel has no path_polarization
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_angles, &path_pos), std::invalid_argument);
    c.fbs_pos.push_back(arma::mat(3, 17, arma::fill::randn));
    c.fbs_pos.push_back(arma::mat(3, 14, arma::fill::randn));

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
    c.lbs_pos.push_back(arma::mat(3, 17, arma::fill::randn));
    c.lbs_pos.push_back(arma::mat(3, 16, arma::fill::randn));

    // No path length provided (but channel has it)
    REQUIRE_THROWS_AS(c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_pos), std::invalid_argument);

    // OK
    c.add_paths(1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &path_pos, &path_pos);
    CHECK(c.is_valid().size() == 0);
    n_paths = c.n_path();
    CHECK(n_paths[0] == 17ULL);
    CHECK(n_paths[1] == 18ULL);
}

TEST_CASE("Tools - Combine IRS Coord inactive")
{
    // n_path_1 = 1, n_path_2 = 3
    // active_path skips middle path (index 1)
    // pI2 = [1, 2, 1] — middle path has 2 coords that must still advance i_start_2
    // Without the bug fix, path 2 reads from the wrong offset in interact_coord_2

    float Ix = 1.0f, Iy = 2.0f, Iz = 3.0f;

    arma::u32_vec n_interact_1 = {1};
    arma::Mat<float> interact_coord_1(3, 1, arma::fill::zeros);
    interact_coord_1(0, 0) = 10.0f;
    interact_coord_1(1, 0) = 11.0f;
    interact_coord_1(2, 0) = 12.0f;

    arma::u32_vec n_interact_2 = {1, 2, 1};
    arma::Mat<float> interact_coord_2(3, 4, arma::fill::zeros);
    interact_coord_2(0, 0) = 20.0f;
    interact_coord_2(1, 0) = 21.0f;
    interact_coord_2(2, 0) = 22.0f;
    interact_coord_2(0, 1) = 30.0f;
    interact_coord_2(1, 1) = 31.0f;
    interact_coord_2(2, 1) = 32.0f;
    interact_coord_2(0, 2) = 33.0f;
    interact_coord_2(1, 2) = 34.0f;
    interact_coord_2(2, 2) = 35.0f;
    interact_coord_2(0, 3) = 40.0f;
    interact_coord_2(1, 3) = 41.0f;
    interact_coord_2(2, 3) = 42.0f;

    std::vector<bool> active_path = {true, false, true};

    arma::u32_vec n_interact;
    arma::Mat<float> interact_coord;

    quadriga_lib::combine_irs_coord({Ix, Iy, Iz},
                                    n_interact_1, interact_coord_1,
                                    n_interact_2, interact_coord_2,
                                    n_interact, interact_coord,
                                    false, false, active_path);

    // Two active paths, each with 3 interaction points: seg1 + IRS + seg2
    REQUIRE(n_interact.n_elem == 2);
    CHECK(n_interact(0) == 3);
    CHECK(n_interact(1) == 3);

    REQUIRE(interact_coord.n_cols == 6);

    // Path 0: seg1[0] → IRS → seg2[0]
    CHECK(interact_coord(0, 0) == 10.0f);
    CHECK(interact_coord(1, 0) == 11.0f);
    CHECK(interact_coord(2, 0) == 12.0f);
    CHECK(interact_coord(0, 1) == Ix);
    CHECK(interact_coord(1, 1) == Iy);
    CHECK(interact_coord(2, 1) == Iz);
    CHECK(interact_coord(0, 2) == 20.0f);
    CHECK(interact_coord(1, 2) == 21.0f);
    CHECK(interact_coord(2, 2) == 22.0f);

    // Path 1: seg1[0] → IRS → seg2[2]  (seg2 path index 2, col 3 in interact_coord_2)
    // Bug: without fix this reads col 1 ([30,31,32]) instead of col 3 ([40,41,42])
    CHECK(interact_coord(0, 3) == 10.0f);
    CHECK(interact_coord(1, 3) == 11.0f);
    CHECK(interact_coord(2, 3) == 12.0f);
    CHECK(interact_coord(0, 4) == Ix);
    CHECK(interact_coord(1, 4) == Iy);
    CHECK(interact_coord(2, 4) == Iz);
    CHECK(interact_coord(0, 5) == 40.0f);
    CHECK(interact_coord(1, 5) == 41.0f);
    CHECK(interact_coord(2, 5) == 42.0f);
}

TEST_CASE("Get Channels IRS - Delay is the sum of both segment path lengths")
{
    // TX (0,0,0) -> IRS (50,0,0) -> RX (100,0,0); one NLOS path per segment whose path_length exceeds the
    // TX-FBS-LBS-IRS (IRS-FBS-LBS-RX) polyline. The IRS has one element and a single unit codebook entry.
    const double C = 299792458.0, fc = 3.0e9, k = 2.0 * arma::datum::pi * fc / C;
    auto ant = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto irs = ant;
    irs.coupling_re.ones(1, 1);
    irs.coupling_im.reset();

    arma::mat f1 = arma::vec({20.0, 15.0, 0.0}), l1 = arma::vec({35.0, 15.0, 0.0});
    arma::mat f2 = arma::vec({60.0, -10.0, 0.0}), l2 = arma::vec({85.0, -10.0, 0.0});
    const double poly_1 = 25.0 + 15.0 + std::sqrt(15.0 * 15.0 + 15.0 * 15.0);
    const double poly_2 = std::sqrt(10.0 * 10.0 + 10.0 * 10.0) + 25.0 + std::sqrt(15.0 * 15.0 + 10.0 * 10.0);

    arma::vec g = {1.0};
    arma::mat M = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0});
    arma::cube cr, ci, dl;

    for (double extra : {0.0, 7.5})
    {
        arma::vec pl1 = {extra > 0.0 ? poly_1 + extra : 0.0}, pl2 = {extra > 0.0 ? poly_2 + 2.0 * extra : 0.0};
        const double L = poly_1 + poly_2 + 3.0 * extra;
        quadriga_lib::get_channels_irs<double>(ant, ant, irs,
                                               {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
                                               {100.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
                                               {50.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
                                               f1, l1, g, pl1, M, f2, l2, g, pl2, M,
                                               cr, ci, dl, 0, -200.0, fc, true);
        REQUIRE(dl.n_slices == 1);
        CHECK(std::abs(dl(0) * C - L) < 1e-6);
        const double a = std::hypot(cr(0), ci(0));
        CHECK(std::abs(cr(0) / a - std::cos(k * L)) < 1e-6);
        CHECK(std::abs(ci(0) / a + std::sin(k * L)) < 1e-6);
    }
}

TEST_CASE("Channel - Add Paths with 0 paths in the first field")
{
    // The first given field sets the number of added paths, also when it is 0; every other field must match
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.rx_pos = arma::mat(3, 1, arma::fill::ones);
    c.coeff_re.push_back(arma::cube(2, 2, 1, arma::fill::ones));
    c.coeff_im.push_back(arma::cube(2, 2, 1, arma::fill::zeros));
    c.delay.push_back(arma::cube(1, 1, 1, arma::fill::zeros));
    c.path_gain.push_back(arma::vec(1, arma::fill::ones));
    c.path_length.push_back(arma::vec(1, arma::fill::ones));

    arma::cube re0(2, 2, 0), im0(2, 2, 0), dl0(1, 1, 0);
    arma::vec v0, v5(5, arma::fill::ones);

    // 0 coefficient paths and 5 path gains
    CHECK_THROWS_AS(c.add_paths(0, &re0, &im0, &dl0, nullptr, nullptr, &v5, &v0), std::invalid_argument);
    // 0 path gains and 5 path lengths (no coefficients involved)
    quadriga_lib::channel<double> d = c;
    d.coeff_re.clear(), d.coeff_im.clear(), d.delay.clear();
    CHECK_THROWS_AS(d.add_paths(0, nullptr, nullptr, nullptr, nullptr, nullptr, &v0, &v5), std::invalid_argument);
    // The objects are unchanged and valid
    CHECK(c.is_valid().empty());
    CHECK(d.is_valid().empty());
    CHECK(c.path_gain[0].n_elem == 1);
    CHECK(d.path_length[0].n_elem == 1);

    // 0 paths in every field is accepted
    c.add_paths(0, &re0, &im0, &dl0, nullptr, nullptr, &v0, &v0);
    d.add_paths(0, nullptr, nullptr, nullptr, nullptr, nullptr, &v0, &v0);
    CHECK(c.is_valid().empty());
    CHECK(d.is_valid().empty());
    CHECK(c.n_path()(0) == 1);
}

TEST_CASE("Tools - Combine IRS Coord - shapes of interact_coord_1 / _2 are checked")
{
    // Throws std::invalid_argument whose message names the input
    auto throws_naming = [](const arma::u32_vec &n1, const arma::mat &c1, const arma::u32_vec &n2, const arma::mat &c2,
                            const std::string &name)
    {
        arma::u32_vec no;
        arma::mat out;
        bool thrown = false;
        try
        {
            quadriga_lib::combine_irs_coord<double>({0.0, 0.0, 0.0}, n1, c1, n2, c2, no, out);
        }
        catch (const std::invalid_argument &e)
        {
            thrown = std::string(e.what()).find(name) != std::string::npos;
            INFO(e.what());
            CHECK(thrown);
        }
        catch (...)
        {
        }
        return thrown;
    };

    arma::u32_vec n10 = {10}, n1 = {1}, n0 = {0}, n23 = {2, 3};
    arma::mat c3x10(3, 10, arma::fill::ones), c3x1(3, 1, arma::fill::ones), c3x5(3, 5, arma::fill::ones);

    CHECK(throws_naming(n10, arma::mat(2, 10, arma::fill::ones), n1, c3x1, "interact_coord_1")); // heap over-read before
    CHECK(throws_naming(n10, arma::mat(3, 9, arma::fill::ones), n1, c3x1, "interact_coord_1"));
    CHECK(throws_naming(n10, arma::mat(4, 10, arma::fill::ones), n1, c3x1, "interact_coord_1"));
    CHECK(throws_naming(n10, c3x10, n1, arma::mat(2, 1, arma::fill::ones), "interact_coord_2"));
    CHECK(throws_naming(n10, c3x10, n23, arma::mat(3, 4, arma::fill::ones), "interact_coord_2"));
    CHECK(throws_naming(n0, c3x1, n1, c3x1, "interact_coord_1"));

    // Valid shapes, including no interactions with an empty coordinate matrix
    arma::u32_vec no;
    arma::mat out;
    quadriga_lib::combine_irs_coord<double>({0.0, 0.0, 0.0}, n10, c3x10, n23, c3x5, no, out);
    CHECK(out.n_cols == 2 * (10 + 1) + 2 + 3);
    arma::mat empty;
    quadriga_lib::combine_irs_coord<double>({0.0, 0.0, 0.0}, n0, empty, n1, c3x1, no, out);
    CHECK(no.n_elem == 1);
    CHECK(out.n_cols == 2);
}

TEST_CASE("Get Channels IRS - A segment without paths gives empty outputs")
{
    // get_channels_spherical returns empty segment outputs, so the IRS channel is [n_rx_ports, n_tx_ports, 0]
    auto ant = quadriga_lib::generate_arrayant_omni<double>(10.0);
    quadriga_lib::arrayant_copy_element(ant, 0, {1});
    auto irs = ant;
    irs.coupling_re = arma::mat({{1.0, 0.0}, {0.0, 1.0}});
    irs.coupling_im = arma::mat(2, 2, arma::fill::zeros);
    arma::mat fb = {5.0, 0.0, 1.0}, f0(3, 0), M = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}, M0(8, 0);
    fb = fb.t(), M = M.t();
    arma::vec g = {1.0}, l = {12.0}, e;
    arma::cube cr, ci, dl, ad, ed, aa, ea;
    for (int seg = 1; seg <= 2; ++seg)
    {
        const bool s1 = seg == 1;
        std::vector<bool> keep;
        quadriga_lib::get_channels_irs<double>(ant, ant, irs, {0, 0, 0}, {0, 0, 0}, {10, 0, 0}, {0, 0, 0}, {5, 5, 0}, {0, 0, 0},
                                               *(s1 ? &f0 : &fb), *(s1 ? &f0 : &fb), *(s1 ? &e : &g), *(s1 ? &e : &l), *(s1 ? &M0 : &M),
                                               *(s1 ? &fb : &f0), *(s1 ? &fb : &f0), *(s1 ? &g : &e), *(s1 ? &l : &e), *(s1 ? &M : &M0),
                                               cr, ci, dl, 0, -140.0, 1e9, false, &ad, &ed, &aa, &ea, {}, {}, &keep);
        CHECK(keep.empty());
        for (const arma::cube *c : {&cr, &ci, &dl, &ad, &ed, &aa, &ea})
        {
            CHECK(c->n_rows == 2);
            CHECK(c->n_cols == 2);
            CHECK(c->n_slices == 0);
        }
    }
}

TEST_CASE("combine_irs_coord - outputs that alias inputs")
{
    arma::u32_vec n1 = {1, 0}, n2 = {1}, n_interact;
    arma::mat c1 = arma::vec({1.0, 2.0, 3.0}), c2 = arma::vec({4.0, 5.0, 6.0}), coord;
    CHECK_NOTHROW(quadriga_lib::combine_irs_coord({0.0, 0.0, 0.0}, n1, c1, n2, c2, n_interact, coord));

    arma::u32_vec m1 = n1, m2 = n2;
    arma::mat d1 = c1, d2 = c2;
    CHECK_THROWS_AS(quadriga_lib::combine_irs_coord({0.0, 0.0, 0.0}, m1, c1, n2, c2, m1, coord), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::combine_irs_coord({0.0, 0.0, 0.0}, n1, c1, m2, c2, m2, coord), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::combine_irs_coord({0.0, 0.0, 0.0}, n1, d1, n2, c2, n_interact, d1), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::combine_irs_coord({0.0, 0.0, 0.0}, n1, c1, n2, d2, n_interact, d2), std::invalid_argument);
    CHECK(arma::all(m1 == n1));
    CHECK(arma::approx_equal(d1, c1, "absdiff", 0.0));

    arma::mat view(d2.memptr(), 3, 1, false, true);
    CHECK_THROWS_AS(quadriga_lib::combine_irs_coord({0.0, 0.0, 0.0}, n1, c1, n2, d2, n_interact, view), std::invalid_argument);
}

TEST_CASE("Get Channels IRS - outputs that alias inputs")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto irs = ant;
    irs.coupling_re = arma::mat({1.0});
    irs.coupling_im = arma::mat({0.0});
    arma::mat fb = arma::mat({5.0, 0.0, 1.0}).t(), M = arma::mat({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}).t();
    arma::vec g = {1.0}, l = {12.0};
    arma::cube cr, ci, dl, ad, ed, aa, ea;
    auto call = [&](arma::cube *c_re, arma::cube *c_im, arma::cube *a1, arma::cube *a2, const arma::mat *M_1)
    {
        quadriga_lib::get_channels_irs<double>(ant, ant, irs, {0, 0, 0}, {0, 0, 0}, {10, 0, 0}, {0, 0, 0}, {5, 5, 0}, {0, 0, 0},
                                               fb, fb, g, l, *M_1, fb, fb, g, l, M,
                                               *c_re, *c_im, dl, 0, -140.0, 1e9, false, a1, &ed, &aa, a2);
    };
    CHECK_NOTHROW(call(&cr, &ci, &ad, &ea, &M));

    SECTION("Two outputs are the same object")
    {
        CHECK_THROWS_AS(call(&cr, &cr, &ad, &ea, &M), std::invalid_argument);
        CHECK_THROWS_AS(call(&cr, &ci, &ad, &ad, &M), std::invalid_argument);
    }
    SECTION("Output shares memory with an input")
    {
        arma::cube M_cube(8, 1, 1);
        arma::mat M_on_ea(M_cube.memptr(), 8, 1, false, true);
        M_on_ea = M;
        CHECK_THROWS_AS(call(&cr, &ci, &ad, &M_cube, &M_on_ea), std::invalid_argument);
    }
}

TEST_CASE("get_channels_irs - a positive tilt points down (3GPP TR 38.901)")
{
    // TX 10 m above the IRS and 10 m away on the -x side, the RX 10 m below it (or level with it); the IRS faces them
    // (heading pi). A beam along x reaches the other end only when turned toward it: TX tilt +pi/4 (down), RX tilt
    // -pi/4 (up, heading 0 toward the IRS), IRS tilt -pi/4 (up, toward the TX; the level RX is 45 degree off either way).
    const double q = arma::datum::pi / 4.0, pi = arma::datum::pi;
    const auto beam = quadriga_lib::generate_arrayant_custom<double>(40.0, 40.0, 0.0, 5.0);
    const auto omni = quadriga_lib::generate_arrayant_omni<double>(5.0);
    const arma::mat pos_1(arma::vec{5.0, 0.0, 5.0}); // A scatterer on the LOS of the TX-IRS segment
    const arma::vec gain = {1.0}, length = {0.0};
    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0, M(6, 0) = -1.0;
    auto power = [&](const quadriga_lib::arrayant<double> &tx, const quadriga_lib::arrayant<double> &rx,
                     const quadriga_lib::arrayant<double> &irs, double Tt, double Rt, double It, double Rz)
    {
        const arma::mat pos_2(arma::vec{5.0, 0.0, 0.5 * Rz}); // A scatterer on the LOS of the IRS-RX segment
        arma::cube cr, ci, dl;
        quadriga_lib::get_channels_irs<double>(tx, rx, irs, {0.0, 0.0, 10.0}, {0.0, Tt, 0.0}, {0.0, 0.0, Rz}, {0.0, Rt, 0.0},
                                               {10.0, 0.0, 0.0}, {0.0, It, pi}, pos_1, pos_1, gain, length, M,
                                               pos_2, pos_2, gain, length, M, cr, ci, dl);
        return arma::accu(cr % cr + ci % ci);
    };
    CHECK(power(beam, omni, omni, q, 0.0, 0.0, -10.0) > 100.0 * power(beam, omni, omni, -q, 0.0, 0.0, -10.0));
    CHECK(power(omni, beam, omni, 0.0, -q, 0.0, -10.0) > 100.0 * power(omni, beam, omni, 0.0, q, 0.0, -10.0));
    CHECK(power(omni, omni, beam, 0.0, 0.0, -q, 0.0) > 100.0 * power(omni, omni, beam, 0.0, 0.0, q, 0.0));
}

TEST_CASE("Get Channels IRS - Position and orientation shapes")
{
    // Positions and orientations: [3, 1] and [1, 3] give the same result; other sizes throw before any output is written
    auto ant = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto irs = ant;
    irs.coupling_re = arma::mat({1.0});
    irs.coupling_im = arma::mat({0.0});
    const arma::mat fb = arma::mat({5.0, 0.0, 1.0}).t(), M = arma::mat({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}).t();
    const arma::vec g = {1.0}, l = {0.0};
    const arma::mat col[6] = {arma::vec{0.0, 0.0, 1.0}, arma::vec{0.1, 0.2, 0.3}, arma::vec{10.0, 0.0, 1.0},
                              arma::vec{0.0, -0.1, 0.5}, arma::vec{5.0, 5.0, 1.0}, arma::vec{0.0, 0.0, -1.5}};
    const char *name[6] = {"tx_pos", "tx_orientation", "rx_pos", "rx_orientation", "irs_pos", "irs_orientation"};
    auto run = [&](const arma::mat *p, arma::cube &cr, arma::cube &ci, arma::cube &dl)
    {
        quadriga_lib::get_channels_irs(ant, ant, irs, p[0], p[1], p[2], p[3], p[4], p[5], fb, fb, g, l, M, fb, fb, g, l, M,
                                       cr, ci, dl, 0, -140.0, 1e9);
    };

    arma::cube cr, ci, dl, cr_row, ci_row, dl_row;
    const arma::mat row[6] = {col[0].t(), col[1].t(), col[2].t(), col[3].t(), col[4].t(), col[5].t()};
    run(col, cr, ci, dl), run(row, cr_row, ci_row, dl_row);
    CHECK(cr.n_slices == 1);
    CHECK(arma::approx_equal(cr, cr_row, "absdiff", 0.0));
    CHECK(arma::approx_equal(ci, ci_row, "absdiff", 0.0));
    CHECK(arma::approx_equal(dl, dl_row, "absdiff", 0.0));

    const arma::mat bad[] = {arma::mat(2, 1, arma::fill::zeros), arma::mat(4, 1, arma::fill::zeros), arma::mat(),
                             arma::mat(3, 3, arma::fill::zeros)};
    for (int k = 0; k < 6; ++k)
        for (const arma::mat &b : bad)
        {
            arma::mat p[6] = {col[0], col[1], col[2], col[3], col[4], col[5]};
            p[k] = b;
            arma::cube c_re = cr, c_im = ci, d = dl;
            CHECK_THROWS_WITH(run(p, c_re, c_im, d), Catch::Matchers::ContainsSubstring("Input '" + std::string(name[k]) + "' must have 3 elements"));
            CHECK(arma::approx_equal(c_re, cr, "absdiff", 0.0));
            CHECK(arma::approx_equal(d, dl, "absdiff", 0.0));
        }
}

TEST_CASE("Get Channels IRS - Empty element_pos and coupling_im of any shape")
{
    // An empty element_pos ([0, n] as well as [0, 0]) places all IRS elements at the phase centre, an empty coupling_im
    // ([n, 0] as well as [0, 0]) means real weights; both are valid and give the result of the explicit zeros
    auto ant = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto irs = quadriga_lib::generate_arrayant_ula<double>(4, 3.5e9, 0.5, {}, 10.0);
    irs.coupling_re = arma::mat(4, 3, arma::fill::ones), irs.coupling_re(1, 2) = -1.0;
    irs.element_pos.zeros(3, 4), irs.coupling_im.zeros(4, 3);
    const arma::mat fb1 = arma::mat({5.0, 5.0, 7.0}).t(), fb2 = arma::mat({15.0, 5.0, 3.0}).t();
    const arma::mat M = arma::mat({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}).t();
    const arma::vec g = {1.0}, l = {0.0};
    auto run = [&](const quadriga_lib::arrayant<double> &irs_1, const quadriga_lib::arrayant<double> &irs_2, arma::cube &cr, arma::cube &ci)
    {
        arma::cube dl;
        quadriga_lib::get_channels_irs<double>(ant, ant, irs_1, {0, 0, 10}, {0, 0, 0}, {20, 0, 1.5}, {0, 0, 0}, {10, 10, 5}, {0, 0, 0},
                                               fb1, fb1, g, l, M, fb2, fb2, g, l, M, cr, ci, dl, 2, -140.0, 3.5e9, false,
                                               nullptr, nullptr, nullptr, nullptr, irs_2);
    };
    arma::cube cr_ref, ci_ref, cr, ci;
    run(irs, {}, cr_ref, ci_ref);
    REQUIRE(cr_ref.n_slices == 1);

    for (int k = 0; k < 4; ++k)
    {
        auto e = irs;
        if (k == 0) e.element_pos.set_size(0, 4);
        if (k == 1) e.element_pos.reset();
        if (k == 2) e.coupling_im.set_size(4, 0);
        if (k == 3) e.coupling_im.reset();
        REQUIRE(e.is_valid().empty());
        run(e, {}, cr, ci);
        CHECK(arma::approx_equal(cr, cr_ref, "absdiff", 0.0));
        CHECK(arma::approx_equal(ci, ci_ref, "absdiff", 0.0));
        run(irs, e, cr, ci);
        CHECK(arma::approx_equal(cr, cr_ref, "absdiff", 0.0));
        CHECK(arma::approx_equal(ci, ci_ref, "absdiff", 0.0));
    }
}

TEST_CASE("Get Channels IRS - A second IRS array leaves the first one unchanged")
{
    // irs_array_2 is used for the RX-facing side only; both arrays are inputs and keep their data, so a repeated call
    // gives the same result, and the second array may use another angle grid
    auto ant = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto irs = quadriga_lib::generate_arrayant_ula<double>(4, 3.5e9, 0.5, {}, 10.0);
    irs.coupling_re = arma::mat(4, 3, arma::fill::ones), irs.coupling_im.zeros(4, 3);
    auto irs2 = irs;
    irs2.e_theta_re *= 0.5, irs2.coupling_re *= 3.0, irs2.element_pos.fill(0.1);
    const quadriga_lib::arrayant<double> irs_copy = irs, irs2_copy = irs2;
    const arma::mat fb1 = arma::mat({5.0, 5.0, 7.0}).t(), fb2 = arma::mat({15.0, 5.0, 3.0}).t();
    const arma::mat M = arma::mat({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}).t();
    const arma::vec g = {1.0}, l = {0.0};
    auto run = [&](const quadriga_lib::arrayant<double> &irs_2, arma::cube &cr, arma::cube &ci)
    {
        arma::cube dl;
        quadriga_lib::get_channels_irs<double>(ant, ant, irs, {0, 0, 10}, {0, 0, 0}, {20, 0, 1.5}, {0, 0, 0}, {10, 10, 5}, {0, 0, 0},
                                               fb1, fb1, g, l, M, fb2, fb2, g, l, M, cr, ci, dl, 1, -140.0, 3.5e9, false,
                                               nullptr, nullptr, nullptr, nullptr, irs_2);
    };
    arma::cube cr_1, ci_1, cr_2, ci_2;
    run(irs2, cr_1, ci_1), run(irs2, cr_2, ci_2);
    REQUIRE(cr_1.n_slices == 1);
    CHECK(arma::approx_equal(cr_1, cr_2, "absdiff", 0.0));
    CHECK(arma::approx_equal(ci_1, ci_2, "absdiff", 0.0));
    for (const auto *p : {&irs, &irs2})
    {
        const auto &c = (p == &irs) ? irs_copy : irs2_copy;
        CHECK(arma::approx_equal(p->e_theta_re, c.e_theta_re, "absdiff", 0.0));
        CHECK(arma::approx_equal(p->element_pos, c.element_pos, "absdiff", 0.0));
        CHECK(arma::approx_equal(p->coupling_re, c.coupling_re, "absdiff", 0.0));
    }

    // The same second IRS on a 5 degree grid
    auto irs3 = quadriga_lib::generate_arrayant_ula<double>(4, 3.5e9, 0.5, {}, 5.0);
    irs3.e_theta_re *= 0.5, irs3.coupling_re = irs2.coupling_re, irs3.coupling_im.reset(), irs3.element_pos = irs2.element_pos;
    arma::cube cr_3, ci_3;
    REQUIRE_NOTHROW(run(irs3, cr_3, ci_3));
    CHECK(arma::approx_equal(cr_3, cr_1, "absdiff", 1e-6));
    CHECK(arma::approx_equal(ci_3, ci_1, "absdiff", 1e-6));
}

TEST_CASE("Get Channels IRS - AVX2 in both segments agrees with GENERIC, auto is GENERIC")
{
    // The kernel reaches both segments: AVX2 agrees with GENERIC; auto selects GENERIC (identical results); invalid
    // values throw
    const auto tx = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, 5.0);
    const auto rx = quadriga_lib::generate_arrayant_xpol<double>(10.0);
    auto irs = quadriga_lib::generate_arrayant_omni<double>(10.0);
    quadriga_lib::arrayant_copy_element(irs, 0, {1, 2});
    irs.element_pos(1, 0) = 0.05, irs.element_pos(1, 2) = -0.05;
    irs.coupling_re.ones(3, 1), irs.coupling_im.reset();
    const arma::mat f1 = arma::randu<arma::mat>(3, 4) * 20.0, f2 = arma::randu<arma::mat>(3, 3) * 20.0 + 20.0;
    const arma::vec g1(4, arma::fill::ones), g2(3, arma::fill::ones), l1(4, arma::fill::zeros), l2(3, arma::fill::zeros);
    const arma::mat M1 = arma::randn<arma::mat>(8, 4), M2 = arma::randn<arma::mat>(8, 3);

    arma::cube cr[3], ci[3], dl[3];
    const bool avx2 = quadriga_lib::quadriga_lib_has_AVX2();
    for (int k : {0, 1, 2})
    {
        if (k == 2 && !avx2)
        {
            CHECK_THROWS_AS(quadriga_lib::get_channels_irs<double>(tx, rx, irs, {0, 0, 1}, {0, 0, 0}, {40, 0, 1}, {0, 0, 0}, {20, 0, 1}, {0, 0, 0},
                                                                   f1, f1, g1, l1, M1, f2, f2, g2, l2, M2, cr[k], ci[k], dl[k], 0, -200.0,
                                                                   3e9, false, nullptr, nullptr, nullptr, nullptr, {}, {}, nullptr, k),
                            std::invalid_argument);
            continue;
        }
        quadriga_lib::get_channels_irs<double>(tx, rx, irs, {0, 0, 1}, {0, 0, 0}, {40, 0, 1}, {0, 0, 0}, {20, 0, 1}, {0, 0, 0},
                                               f1, f1, g1, l1, M1, f2, f2, g2, l2, M2, cr[k], ci[k], dl[k], 0, -200.0,
                                               3e9, false, nullptr, nullptr, nullptr, nullptr, {}, {}, nullptr, k);
    }
    REQUIRE(cr[1].n_slices == 12);
    CHECK((arma::approx_equal(cr[0], cr[1], "absdiff", 0.0) && arma::approx_equal(ci[0], ci[1], "absdiff", 0.0)));
    if (avx2)
    {
        const double amax = arma::sqrt(arma::square(cr[1]) + arma::square(ci[1])).max();
        CHECK(arma::sqrt(arma::square(cr[2] - cr[1]) + arma::square(ci[2] - ci[1])).max() < 1e-5 * amax);
        CHECK(arma::approx_equal(dl[2], dl[1], "absdiff", 1e-12));
    }
    CHECK_THROWS_AS(quadriga_lib::get_channels_irs<double>(tx, rx, irs, {0, 0, 1}, {0, 0, 0}, {40, 0, 1}, {0, 0, 0}, {20, 0, 1}, {0, 0, 0},
                                                           f1, f1, g1, l1, M1, f2, f2, g2, l2, M2, cr[0], ci[0], dl[0], 0, -200.0,
                                                           3e9, false, nullptr, nullptr, nullptr, nullptr, {}, {}, nullptr, 3),
                    std::invalid_argument);
}
