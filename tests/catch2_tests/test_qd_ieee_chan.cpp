// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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
#include "ieee_channel_model_functions.hpp"
#include "quadriga_lib.hpp"

// Covered tests:
// - K-Factor model for Type A
// - Correctness of TX/RX positions and orientations
// - Directional antennas at TX and RX
// - LOS Steering matrix
// - NLOS Steering and correct application of offset angles
// - Manual setting of offset angles
// - Seed consistency

// TEST_CASE("IEEE Chan - A + 1user + Xpol")
// {
//     // Generate a cross-pol channel
//     auto ant = quadriga_lib::generate_arrayant_xpol<double>();
//     auto chan = quadriga_lib::get_channels_ieee_indoor(ant, ant, "A");

//     // Check dimensions of the channel object
//     REQUIRE(chan.size() == 1);
//     REQUIRE(chan[0].n_tx() == 2);
//     REQUIRE(chan[0].n_rx() == 2);
//     REQUIRE(arma::all(chan[0].n_path() == arma::uvec{2}));
//     REQUIRE(chan[0].n_snap() == 1);
//     REQUIRE(chan[0].path_gain[0].n_elem == 2);

//     CHECK(chan[0].center_frequency[0] == 5.25e9);

//     // Both paths should have equal power at KF = 0 dB
//     CHECK(std::abs(chan[0].path_gain[0][0] - chan[0].path_gain[0][1]) < 1e-20);

//     // Calculate path gain from coefficients and compare
//     arma::cube vc = chan[0].coeff_re[0] % chan[0].coeff_re[0] + chan[0].coeff_im[0] % chan[0].coeff_im[0];
//     arma::vec vv = arma::sum(arma::mat(vc.memptr(), 2 * 2, 2), 0).t();
//     vv *= 0.5;
//     CHECK(arma::all(arma::abs(chan[0].path_gain[0] - vv) < 1e-20));

//     // Tx should be at the origin, facing east
//     CHECK(arma::all(chan[0].tx_pos.col(0) == arma::vec{0.0, 0.0, 0.0}));
//     CHECK(arma::all(chan[0].tx_orientation.col(0) == arma::vec{0.0, 0.0, 0.0}));

//     // Rx should be 4.99 meters east of the Tx, facing west (so that LOS is at 0° from RX-local view)
//     CHECK(arma::all(chan[0].rx_pos.col(0) == arma::vec{4.99, 0.0, 0.0}));
//     CHECK(arma::all(arma::abs(chan[0].rx_orientation.col(0) - arma::vec{0.0, 0.0, 3.141592653589793}) < 1e-12));

//     // Off-diagonal elements of the LOS steering matrix should be 0 due to perfect Xpol isolation
//     CHECK(chan[0].coeff_re[0][1] == 0.0);
//     CHECK(chan[0].coeff_re[0][2] == 0.0);
//     CHECK(chan[0].coeff_im[0][1] == 0.0);
//     CHECK(chan[0].coeff_im[0][2] == 0.0);

//     // Main diagonal elements should have equal power
//     double p00 = chan[0].coeff_re[0][0] * chan[0].coeff_re[0][0] + chan[0].coeff_im[0][0] * chan[0].coeff_im[0][0];
//     double p01 = chan[0].coeff_re[0][3] * chan[0].coeff_re[0][3] + chan[0].coeff_im[0][3] * chan[0].coeff_im[0][3];
//     CHECK(p00 == p01);
// }

// TEST_CASE("IEEE Chan - A + 2user + Steering")
// {
//     // Generate a probe antenna with 10° steps
//     auto ant = quadriga_lib::generate_arrayant_custom(6.0, 10.0);
//     ant.copy_element(0, arma::regspace<arma::uvec>(1, 35));
//     for (int i = 1; i < 36; ++i)
//         ant.rotate_pattern(0.0, 0.0, double(i) * 10.0, 0, i);

//     auto chan = quadriga_lib::get_channels_ieee_indoor(ant, ant, "A", 2.4e9, 1e-8, 2);

//     // Check dimensions of the channel object
//     REQUIRE(chan.size() == 2);

//     REQUIRE(chan[0].n_tx() == ant.n_elements());
//     REQUIRE(chan[0].n_rx() == ant.n_elements());
//     REQUIRE(arma::all(chan[0].n_path() == arma::uvec{2}));
//     REQUIRE(chan[0].n_snap() == 1);
//     REQUIRE(chan[0].path_gain[0].n_elem == 2);

//     REQUIRE(chan[1].n_tx() == ant.n_elements());
//     REQUIRE(chan[1].n_rx() == ant.n_elements());
//     REQUIRE(arma::all(chan[1].n_path() == arma::uvec{2}));
//     REQUIRE(chan[1].n_snap() == 1);
//     REQUIRE(chan[1].path_gain[0].n_elem == 2);

//     CHECK(chan[0].center_frequency[0] == 2.4e9);
//     CHECK(chan[1].center_frequency[0] == 2.4e9);

//     // Check receiver positions and orientations
//     double aod_deg = -78.0189;  // From TGac doc.: IEEE 802.11-09/0308r12
//     double aoa_deg = -135.3011; // From TGac MATLAB code
//     double aod_rad = aod_deg * 1.745329251994330e-02;
//     double cx = 4.99 * std::cos(aod_rad);
//     double cy = 4.99 * std::sin(aod_rad);
//     double ori = aod_deg - aoa_deg - 180.0;
//     CHECK(arma::all(arma::abs(chan[0].rx_pos.col(0) - arma::vec{cx, cy, 0.0}) < 1e-5));
//     CHECK(std::abs(ori - chan[0].rx_orientation[2] * 57.29577951308232) < 1e-4);

//     aod_deg = -142.9707; // From TGac doc.: IEEE 802.11-09/0308r12
//     aoa_deg = 115.1550;  // From TGac MATLAB code
//     aod_rad = aod_deg * 1.745329251994330e-02;
//     cx = 4.99 * std::cos(aod_rad);
//     cy = 4.99 * std::sin(aod_rad);
//     ori = aod_deg - aoa_deg + 180.0; // -122.718
//     CHECK(arma::all(arma::abs(chan[1].rx_pos.col(0) - arma::vec{cx, cy, 0.0}) < 1e-5));
//     CHECK(std::abs(ori - chan[1].rx_orientation[2] * 57.29577951308232) < 1e-4);

//     // Check LOS steering angles
//     // Tx should see the rx at an steering angle of -78°
//     // Rx should see the tx at an steering angle of -135.3°
//     arma::mat S = chan[0].coeff_re[0].slice(0) % chan[0].coeff_re[0].slice(0);
//     S += chan[0].coeff_im[0].slice(0) % chan[0].coeff_im[0].slice(0);

//     // Max. directional power as seen from the TX and RX
//     arma::vec s_tx = arma::sum(S, 0).t();
//     arma::vec s_rx = arma::sum(S, 1);
//     CHECK(arma::index_max(s_tx) == 28ULL); // 280° = -80° ~ aod_los = -78°
//     CHECK(arma::index_max(s_rx) == 22ULL); // 220° = -140° ~ aoa_los = -135.3°

//     // Generate deterministic LOS and NLOS offsets of 100°
//     arma::mat offset_angles(4, 2, arma::fill::value(100.0));
//     chan = quadriga_lib::get_channels_ieee_indoor(ant, ant, "A", 2.4e9, 1e-8, 2, 0.0,
//                                                   0.001, 0.0, {1.99}, {0}, false, offset_angles, 100, 11);

//     aod_deg = 100.0; // From TGac doc.: IEEE 802.11-09/0308r12
//     aoa_deg = 100.0; // From TGac MATLAB code
//     aod_rad = aod_deg * 1.745329251994330e-02;
//     cx = 1.99 * std::cos(aod_rad);
//     cy = 1.99 * std::sin(aod_rad);
//     ori = aod_deg - aoa_deg + 180.0; // -122.718
//     CHECK(arma::all(arma::abs(chan[0].rx_pos.col(0) - arma::vec{cx, cy, 0.0}) < 1e-5));
//     CHECK(std::abs(ori - chan[0].rx_orientation[2] * 57.29577951308232) < 1e-4);
//     CHECK(arma::all(arma::abs(chan[1].rx_pos.col(0) - arma::vec{cx, cy, 0.0}) < 1e-5));
//     CHECK(std::abs(ori - chan[1].rx_orientation[2] * 57.29577951308232) < 1e-4);

//     // Check LOS steering angles
//     S = chan[0].coeff_re[0].slice(0) % chan[0].coeff_re[0].slice(0);
//     S += chan[0].coeff_im[0].slice(0) % chan[0].coeff_im[0].slice(0);
//     s_tx = arma::sum(S, 0).t();
//     s_rx = arma::sum(S, 1);
//     CHECK(arma::index_max(s_tx) == 10ULL);
//     CHECK(arma::index_max(s_rx) == 10ULL);

//     // Check NLOS steering angles, should be 145 degree (100 + 45 from TGn tables)
//     // with 40° AS, there is a lot of ambiguity, fix seed and check correct value
//     S = chan[0].coeff_re[0].slice(1) % chan[0].coeff_re[0].slice(1);
//     S += chan[0].coeff_im[0].slice(1) % chan[0].coeff_im[0].slice(1);
//     s_tx = arma::sum(S, 0).t();
//     s_rx = arma::sum(S, 1);
//     CHECK(arma::index_max(s_tx) == 15ULL);
//     CHECK(arma::index_max(s_rx) == 14ULL);

//     S = chan[1].coeff_re[0].slice(1) % chan[1].coeff_re[0].slice(1);
//     S += chan[1].coeff_im[0].slice(1) % chan[1].coeff_im[0].slice(1);
//     s_tx = arma::sum(S, 0).t();
//     s_rx = arma::sum(S, 1);
//     CHECK(arma::index_max(s_tx) == 13ULL);
//     CHECK(arma::index_max(s_rx) == 13ULL);
// }

TEST_CASE("IEEE Chan - B + 3user + Floors and Distances")
{
    auto ant = quadriga_lib::generate_arrayant_ula<double>(4, 2.4e9, 0.5, nullptr, 30.0);
    auto chan = quadriga_lib::get_channels_ieee_indoor(ant, ant, "B", ant.center_frequency,
                                                       1e-8, 3, 0.0, 0.001, 0.0, {4.0, 8.0, 2.0}, {0, 0, 1});

    // Transform function
    auto linear_2_dB = [](double x)
    { return 10.0 * std::log10(x); };

    // Check array sizes
    REQUIRE(chan.size() == 3);

    CHECK(chan[0].n_tx() == ant.n_elements());
    CHECK(chan[0].n_rx() == ant.n_elements());
    CHECK(arma::all(chan[0].n_path() == arma::uvec{13}));
    CHECK(chan[0].n_snap() == 1);
    CHECK(chan[0].path_gain[0].n_elem == 13); // LOS

    CHECK(arma::all(chan[1].n_path() == arma::uvec{12}));
    CHECK(chan[1].path_gain[0].n_elem == 12); // No LOS due to dist > d_BP

    CHECK(arma::all(chan[2].n_path() == arma::uvec{12}));
    CHECK(chan[2].path_gain[0].n_elem == 12); // No LOS due to n_floor = 1

    // Check relative path gains
    auto p = chan[0].path_gain[0];
    p /= (2.0 * p[0]);        // Normalize
    p.transform(linear_2_dB); // to dB

    CHECK(std::abs(p[0] + 3.01) < 0.01);  // LOS steering path
    CHECK(std::abs(p[1] + 3.01) < 0.01);  // p = 1, c = 1
    CHECK(std::abs(p[2] + 5.40) < 0.01);  // p = 2, c = 1
    CHECK(std::abs(p[3] + 10.8) < 0.01);  // p = 3, c = 1
    CHECK(std::abs(p[4] + 3.20) < 0.01);  // p = 3, c = 2
    CHECK(std::abs(p[5] + 16.2) < 0.01);  // p = 4, c = 1
    CHECK(std::abs(p[6] + 6.30) < 0.01);  // p = 4, c = 2
    CHECK(std::abs(p[7] + 21.7) < 0.01);  // p = 5, c = 1
    CHECK(std::abs(p[8] + 9.40) < 0.01);  // p = 5, c = 2
    CHECK(std::abs(p[9] + 12.5) < 0.01);  // p = 6, c = 2
    CHECK(std::abs(p[10] + 15.6) < 0.01); // p = 7, c = 2
    CHECK(std::abs(p[11] + 18.7) < 0.01); // p = 8, c = 2
    CHECK(std::abs(p[12] + 21.8) < 0.01); // p = 9, c = 2

    p = chan[1].path_gain[0];
    p /= p[0]; // Normalize
    p.transform(linear_2_dB);

    CHECK(std::abs(p[0] + 0.00) < 0.01); // p = 1, c = 1
    CHECK(std::abs(p[1] + 5.40) < 0.01); // p = 2, c = 1
    CHECK(std::abs(p[2] + 10.8) < 0.01); // p = 3, c = 1
    CHECK(std::abs(p[3] + 3.20) < 0.01); // p = 3, c = 2

    p = chan[2].path_gain[0];
    p /= p[0]; // Normalize
    p.transform(linear_2_dB);

    CHECK(std::abs(p[0] + 0.00) < 0.01); // p = 1, c = 1
    CHECK(std::abs(p[1] + 5.40) < 0.01); // p = 2, c = 1
    CHECK(std::abs(p[2] + 10.8) < 0.01); // p = 3, c = 1
    CHECK(std::abs(p[3] + 3.20) < 0.01); // p = 3, c = 2

    // Check LOS steering matrix (should only contain phasors multiplied by path gain)
    arma::cx_mat S;
    quadriga_lib::complex_cast(chan[0].coeff_re[0].slice(0), chan[0].coeff_im[0].slice(0), S);

    arma::mat A = arma::abs(S);
    A /= std::sqrt(chan[0].path_gain[0][0]);

    CHECK(arma::all(arma::abs(arma::vectorise(A) - 1.0) < 1e-12));

    // Check delays
    auto d = arma::mat(chan[0].delay[0].memptr(), 4 * 4, 13, false, true);
    d *= 1e9;
    d.print();
}