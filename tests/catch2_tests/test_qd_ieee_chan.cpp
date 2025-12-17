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
// - Doppler shift for LOS path
// - TGac badwidth extension
// - Delay, Powers and RMSDS for TGn and TGac
// - Doppler from moving vehicle in Model F
// - Uplink-downlink reciprocity for MIMO channels

static arma::mat calc_Doppler_profile(quadriga_lib::channel<double> chan, double update_rate_s, double BW = 100e6)
{
    // ----- Doppler axis ------------------------------------------------------
    arma::uword w = chan.n_snap();
    double Delta_t = update_rate_s;

    arma::vec Doppler_axis_Hz(w);
    for (arma::uword i = 0; i < w; ++i)
    {
        double x = static_cast<double>(i) / static_cast<double>(w) - 0.5;
        Doppler_axis_Hz(i) = x / Delta_t;
    }

    // ----- Delay–Doppler spectrum via 2D IFFT --------------------------------
    arma::vec pilot_grid = arma::regspace<arma::vec>(0.0, 0.01, 1.0);

    arma::cx_cube hmat;
    arma::cx_mat H(pilot_grid.n_elem, w);

    for (arma::uword i = 0; i < w; ++i)
    {
        quadriga_lib::baseband_freq_response<double>(
            &chan.coeff_re[i],
            &chan.coeff_im[i],
            &chan.delay[i],
            &pilot_grid,
            BW,
            nullptr,
            nullptr,
            &hmat);
        H.col(i) = arma::vectorise(hmat);
    }

    arma::cx_mat G = arma::ifft2(H);

    // fftshift along Doppler (columns)
    {
        arma::uword N = G.n_cols;
        arma::uword p = N / 2;

        arma::cx_mat tmp(G.n_rows, N);
        tmp.cols(0, N - p - 1) = G.cols(p, N - 1);
        tmp.cols(N - p, N - 1) = G.cols(0, p - 1);
        G = tmp;
    }

    arma::mat P = arma::square(arma::abs(G)); // power
    return P;
}

TEST_CASE("IEEE Chan - A + 1user + Xpol")
{
    // Generate a cross-pol channel
    auto ant = quadriga_lib::generate_arrayant_xpol<double>();
    auto chan = quadriga_lib::get_channels_ieee_indoor(ant, ant, "A");

    // Check dimensions of the channel object
    REQUIRE(chan.size() == 1);
    REQUIRE(chan[0].n_tx() == 2);
    REQUIRE(chan[0].n_rx() == 2);
    REQUIRE(arma::all(chan[0].n_path() == arma::uvec{2}));
    REQUIRE(chan[0].n_snap() == 1);
    REQUIRE(chan[0].path_gain[0].n_elem == 2);

    CHECK(chan[0].center_frequency[0] == 5.25e9);

    // Both paths should have equal power at KF = 0 dB
    CHECK(std::abs(chan[0].path_gain[0][0] - chan[0].path_gain[0][1]) < 1e-20);

    // Calculate path gain from coefficients and compare
    arma::cube vc = chan[0].coeff_re[0] % chan[0].coeff_re[0] + chan[0].coeff_im[0] % chan[0].coeff_im[0];
    arma::vec vv = arma::sum(arma::mat(vc.memptr(), 2 * 2, 2), 0).t();
    vv *= 0.5;
    CHECK(arma::all(arma::abs(chan[0].path_gain[0] - vv) < 1e-20));

    // Tx should be at the origin, facing east
    CHECK(arma::all(chan[0].tx_pos.col(0) == arma::vec{0.0, 0.0, 0.0}));
    CHECK(arma::all(chan[0].tx_orientation.col(0) == arma::vec{0.0, 0.0, 0.0}));

    // Rx should be 4.99 meters east of the Tx, facing west (so that LOS is at 0° from RX-local view)
    CHECK(arma::all(chan[0].rx_pos.col(0) == arma::vec{4.99, 0.0, 0.0}));
    CHECK(arma::all(arma::abs(chan[0].rx_orientation.col(0) - arma::vec{0.0, 0.0, 3.141592653589793}) < 1e-12));

    // Off-diagonal elements of the LOS steering matrix should be 0 due to perfect Xpol isolation
    CHECK(chan[0].coeff_re[0][1] == 0.0);
    CHECK(chan[0].coeff_re[0][2] == 0.0);
    CHECK(chan[0].coeff_im[0][1] == 0.0);
    CHECK(chan[0].coeff_im[0][2] == 0.0);

    // Main diagonal elements should have equal power
    double p00 = chan[0].coeff_re[0][0] * chan[0].coeff_re[0][0] + chan[0].coeff_im[0][0] * chan[0].coeff_im[0][0];
    double p01 = chan[0].coeff_re[0][3] * chan[0].coeff_re[0][3] + chan[0].coeff_im[0][3] * chan[0].coeff_im[0][3];
    CHECK(p00 == p01);
}

TEST_CASE("IEEE Chan - A + 2user + Steering")
{
    // Generate a probe antenna with 10° steps
    auto ant = quadriga_lib::generate_arrayant_custom(6.0, 10.0);
    ant.copy_element(0, arma::regspace<arma::uvec>(1, 35));
    for (int i = 1; i < 36; ++i)
        ant.rotate_pattern(0.0, 0.0, double(i) * 10.0, 0, i);

    auto chan = quadriga_lib::get_channels_ieee_indoor(ant, ant, "A", 2.4e9, 1e-8, 2);

    // Check dimensions of the channel object
    REQUIRE(chan.size() == 2);

    REQUIRE(chan[0].n_tx() == ant.n_elements());
    REQUIRE(chan[0].n_rx() == ant.n_elements());
    REQUIRE(arma::all(chan[0].n_path() == arma::uvec{2}));
    REQUIRE(chan[0].n_snap() == 1);
    REQUIRE(chan[0].path_gain[0].n_elem == 2);

    REQUIRE(chan[1].n_tx() == ant.n_elements());
    REQUIRE(chan[1].n_rx() == ant.n_elements());
    REQUIRE(arma::all(chan[1].n_path() == arma::uvec{2}));
    REQUIRE(chan[1].n_snap() == 1);
    REQUIRE(chan[1].path_gain[0].n_elem == 2);

    CHECK(chan[0].center_frequency[0] == 2.4e9);
    CHECK(chan[1].center_frequency[0] == 2.4e9);

    // Check receiver positions and orientations
    double aod_deg = -78.0189;  // From TGac doc.: IEEE 802.11-09/0308r12
    double aoa_deg = -135.3011; // From TGac MATLAB code
    double aod_rad = aod_deg * 1.745329251994330e-02;
    double cx = 4.99 * std::cos(aod_rad);
    double cy = 4.99 * std::sin(aod_rad);
    double ori = aod_deg - aoa_deg - 180.0;
    CHECK(arma::all(arma::abs(chan[0].rx_pos.col(0) - arma::vec{cx, cy, 0.0}) < 1e-5));
    CHECK(std::abs(ori - chan[0].rx_orientation[2] * 57.29577951308232) < 1e-4);

    aod_deg = -142.9707; // From TGac doc.: IEEE 802.11-09/0308r12
    aoa_deg = 115.1550;  // From TGac MATLAB code
    aod_rad = aod_deg * 1.745329251994330e-02;
    cx = 4.99 * std::cos(aod_rad);
    cy = 4.99 * std::sin(aod_rad);
    ori = aod_deg - aoa_deg + 180.0; // -122.718
    CHECK(arma::all(arma::abs(chan[1].rx_pos.col(0) - arma::vec{cx, cy, 0.0}) < 1e-5));
    CHECK(std::abs(ori - chan[1].rx_orientation[2] * 57.29577951308232) < 1e-4);

    // Check LOS steering angles
    // Tx should see the rx at an steering angle of -78°
    // Rx should see the tx at an steering angle of -135.3°
    arma::mat S = chan[0].coeff_re[0].slice(0) % chan[0].coeff_re[0].slice(0);
    S += chan[0].coeff_im[0].slice(0) % chan[0].coeff_im[0].slice(0);

    // Max. directional power as seen from the TX and RX
    arma::vec s_tx = arma::sum(S, 0).t();
    arma::vec s_rx = arma::sum(S, 1);
    CHECK(arma::index_max(s_tx) == 28ULL); // 280° = -80° ~ aod_los = -78°
    CHECK(arma::index_max(s_rx) == 22ULL); // 220° = -140° ~ aoa_los = -135.3°

    // Generate deterministic LOS and NLOS offsets of 100°
    arma::mat offset_angles(4, 2, arma::fill::value(100.0));
    chan = quadriga_lib::get_channels_ieee_indoor(ant, ant, "A", 2.4e9, 1e-8, 2, 0.0,
                                                  0.001, 0.0, 1.2, {1.99}, {0}, false, offset_angles, 100, true, 11);

    aod_deg = 100.0; // From TGac doc.: IEEE 802.11-09/0308r12
    aoa_deg = 100.0; // From TGac MATLAB code
    aod_rad = aod_deg * 1.745329251994330e-02;
    cx = 1.99 * std::cos(aod_rad);
    cy = 1.99 * std::sin(aod_rad);
    ori = aod_deg - aoa_deg + 180.0; // -122.718
    CHECK(arma::all(arma::abs(chan[0].rx_pos.col(0) - arma::vec{cx, cy, 0.0}) < 1e-5));
    CHECK(std::abs(ori - chan[0].rx_orientation[2] * 57.29577951308232) < 1e-4);
    CHECK(arma::all(arma::abs(chan[1].rx_pos.col(0) - arma::vec{cx, cy, 0.0}) < 1e-5));
    CHECK(std::abs(ori - chan[1].rx_orientation[2] * 57.29577951308232) < 1e-4);

    // Check LOS steering angles
    S = chan[0].coeff_re[0].slice(0) % chan[0].coeff_re[0].slice(0);
    S += chan[0].coeff_im[0].slice(0) % chan[0].coeff_im[0].slice(0);
    s_tx = arma::sum(S, 0).t();
    s_rx = arma::sum(S, 1);
    CHECK(arma::index_max(s_tx) == 10ULL);
    CHECK(arma::index_max(s_rx) == 10ULL);

    // Check NLOS steering angles, should be 145 degree (100 + 45 from TGn tables)
    // with 40° AS, there is a lot of ambiguity, fix seed and check correct value
    S = chan[0].coeff_re[0].slice(1) % chan[0].coeff_re[0].slice(1);
    S += chan[0].coeff_im[0].slice(1) % chan[0].coeff_im[0].slice(1);
    s_tx = arma::sum(S, 0).t();
    s_rx = arma::sum(S, 1);
    CHECK(arma::index_max(s_tx) == 15ULL);
    CHECK(arma::index_max(s_rx) == 14ULL);

    S = chan[1].coeff_re[0].slice(1) % chan[1].coeff_re[0].slice(1);
    S += chan[1].coeff_im[0].slice(1) % chan[1].coeff_im[0].slice(1);
    s_tx = arma::sum(S, 0).t();
    s_rx = arma::sum(S, 1);
    CHECK(arma::index_max(s_tx) == 13ULL);
    CHECK(arma::index_max(s_rx) == 13ULL);
}

TEST_CASE("IEEE Chan - B + 3user + Floors and Distances")
{
    double tap_spacing_s = 10e-9;
    arma::uword n_users = 3;
    double observation_time = 0.0;
    double update_rate = 0.001;
    double speed_station_kmh = 0.0;
    double speed_env_kmh = 1.2;
    arma::vec Dist_m = {4.0, 8.0, 2.0};
    arma::uvec n_floors = {0, 0, 1};

    auto ant = quadriga_lib::generate_arrayant_ula<double>(4, 2.4e9, 0.5, nullptr, 30.0);
    auto chan = quadriga_lib::get_channels_ieee_indoor(ant, ant, "B", ant.center_frequency,
                                                       tap_spacing_s, n_users, observation_time, update_rate,
                                                       speed_station_kmh, speed_env_kmh, Dist_m, n_floors);

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

    CHECK(arma::all(arma::abs(d.col(0) - 0.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(1) - 0.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(2) - 10.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(3) - 20.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(4) - 20.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(5) - 30.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(6) - 30.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(7) - 40.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(8) - 40.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(9) - 50.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(10) - 60.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(11) - 70.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(12) - 80.0) < 1.5));

    d = arma::mat(chan[1].delay[0].memptr(), 4 * 4, 13, false, true);
    d *= 1e9;

    CHECK(arma::all(arma::abs(d.col(0) - 0.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(1) - 10.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(2) - 20.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(3) - 20.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(4) - 30.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(5) - 30.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(6) - 40.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(7) - 40.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(8) - 50.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(9) - 60.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(10) - 70.0) < 1.5));
    CHECK(arma::all(arma::abs(d.col(11) - 80.0) < 1.5));

    // Check floor indicator
    CHECK(chan[0].rx_pos[2] == 0.0);
    CHECK(chan[1].rx_pos[2] == 0.0);
    CHECK(chan[2].rx_pos[2] == 3.0);
}

TEST_CASE("IEEE Chan - A + Doppler Shift")
{
    double fGHz = 2.4;
    double tap_spacing_ns = 10.0;
    double observation_time_s = 10.0;
    double update_rate_s = 0.0025;
    double speed_station_kmh = 50.0;
    double speed_env_kmh = 0.0;

    auto ant = quadriga_lib::generate_arrayant_omni<double>(30.0);

    auto chan = quadriga_lib::get_channels_ieee_indoor(ant, ant, "A", fGHz * 1e9,
                                                       tap_spacing_ns / 1e9, 1, observation_time_s, update_rate_s,
                                                       speed_station_kmh, speed_env_kmh);

    arma::uword w = chan[0].n_snap();
    double Delta_t = update_rate_s; // snapshot spacing

    arma::vec Doppler_axis_Hz(w);
    for (arma::uword i = 0; i < w; ++i)
    {
        double x = static_cast<double>(i) / static_cast<double>(w) - 0.5;
        Doppler_axis_Hz(i) = x / Delta_t;
    }

    arma::vec pilot_grid = arma::regspace<arma::vec>(0.0, 0.01, 1.0);
    double BW = 100e6;

    arma::cx_cube hmat;
    arma::cx_mat H(pilot_grid.n_elem, w);
    for (arma::uword i = 0; i < w; ++i)
    {
        quadriga_lib::baseband_freq_response<double>(&chan[0].coeff_re[i], &chan[0].coeff_im[i], &chan[0].delay[i], &pilot_grid, BW, nullptr, nullptr, &hmat);
        H.col(i) = arma::vectorise(hmat);
    }

    arma::cx_mat G = arma::ifft2(H); // 2D IFFT
    {                                // FFTshift
        arma::uword N = G.n_cols;
        arma::uword p = N / 2;

        arma::cx_mat tmp(G.n_rows, N);
        tmp.cols(0, N - p - 1) = G.cols(p, N - 1);
        tmp.cols(N - p, N - 1) = G.cols(0, p - 1);

        G = tmp;
    }

    arma::mat P = arma::square(arma::abs(G)); // power
    arma::rowvec DS_lin = arma::sum(P, 0);    // sum over delay (rows)
    arma::vec DS_dB = 10.0 * arma::log10(DS_lin.t() + 1e-12);

    // Simple sanity checks; adjust or extend as needed
    REQUIRE(DS_dB.n_elem == static_cast<arma::uword>(w));
    REQUIRE(Doppler_axis_Hz.n_elem == static_cast<arma::uword>(w));

    // Example: check that max is somewhere near the expected Doppler
    arma::uword idx_max;
    DS_dB.max(idx_max);
    double f_at_max = Doppler_axis_Hz(idx_max);

    // Expected Doppler magnitude
    double v = speed_station_kmh / 3.6;
    double c0 = 3.0e8;
    double fc = fGHz * 1.0e9;
    double fD_expected = v / c0 * fc;

    REQUIRE(std::abs(std::abs(f_at_max) - fD_expected) < 5.0); // loose check
}

TEST_CASE("IEEE Chan - C + TGn Bell Doppler integrated shape")
{
    double fGHz = 2.4;
    double tap_spacing_ns = 10.0;
    double observation_time_s = 20.0;
    double update_rate_s = 0.01;    // 100 Hz sampling -> 0.05 Hz resolution
    double speed_station_kmh = 0.0; // TGn: stations static
    double speed_env_kmh = 1.2;     // TGn environmental speed

    auto ant = quadriga_lib::generate_arrayant_omni<double>(30.0);

    auto chan = quadriga_lib::get_channels_ieee_indoor(
        ant, ant, "C",
        fGHz * 1e9,
        tap_spacing_ns / 1e9,
        1,
        observation_time_s,
        update_rate_s,
        speed_station_kmh,
        speed_env_kmh,
        {}, {}, false, {}, 100, 50.0, 1234);

    // ----- Doppler axis ------------------------------------------------------
    arma::uword w = chan[0].n_snap();
    double Delta_t = update_rate_s;

    arma::vec Doppler_axis_Hz(w);
    for (arma::uword i = 0; i < w; ++i)
    {
        double x = static_cast<double>(i) / static_cast<double>(w) - 0.5;
        Doppler_axis_Hz(i) = x / Delta_t;
    }

    // ----- Delay–Doppler spectrum via 2D IFFT --------------------------------
    arma::vec pilot_grid = arma::regspace<arma::vec>(0.0, 0.01, 1.0);
    double BW = 100e6;

    arma::cx_cube hmat;
    arma::cx_mat H(pilot_grid.n_elem, w);

    for (arma::uword i = 0; i < w; ++i)
    {
        quadriga_lib::baseband_freq_response<double>(
            &chan[0].coeff_re[i],
            &chan[0].coeff_im[i],
            &chan[0].delay[i],
            &pilot_grid,
            BW,
            nullptr,
            nullptr,
            &hmat);
        H.col(i) = arma::vectorise(hmat);
    }

    arma::cx_mat G = arma::ifft2(H);

    // fftshift along Doppler (columns)
    {
        arma::uword N = G.n_cols;
        arma::uword p = N / 2;

        arma::cx_mat tmp(G.n_rows, N);
        tmp.cols(0, N - p - 1) = G.cols(p, N - 1);
        tmp.cols(N - p, N - 1) = G.cols(0, p - 1);
        G = tmp;
    }

    arma::mat P = arma::square(arma::abs(G));  // power
    arma::rowvec DS_lin_row = arma::sum(P, 0); // sum over delay
    arma::vec DS_lin = DS_lin_row.t();

    // Basic sanity
    REQUIRE(DS_lin.n_elem == w);
    double total_power = arma::accu(DS_lin);
    REQUIRE(total_power > 0.0);

    // Normalise so that integral over all f is 1
    DS_lin /= total_power;

    // ----- Analytic TGn fractions R(F) ---------------------------------------
    double c0 = 3.0e8;
    double fc = fGHz * 1e9;
    double lambda = c0 / fc;

    double v_env = speed_env_kmh / 3.6;
    double f_d = v_env / lambda; // Doppler spread

    auto R_analytic = [&](double F) -> double
    {
        double x0 = F / f_d;
        return (2.0 / 3.141592653589793) * std::atan(3.0 * x0); // 2/pi * atan(3F/f_d)
    };

    double R1_ana = R_analytic(f_d);       // ≈ 0.80
    double R2_ana = R_analytic(2.0 * f_d); // ≈ 0.89

    // ----- Empirical fractions R_sim(F) from discrete spectrum ---------------
    auto R_sim = [&](double F) -> double
    {
        arma::uvec idx = arma::find(arma::abs(Doppler_axis_Hz) <= F);
        REQUIRE(idx.n_elem > 0);
        double num = arma::accu(DS_lin.elem(idx));
        return num; // denominator is 1 because DS_lin is normalised
    };

    double R1_sim = R_sim(f_d);
    double R2_sim = R_sim(2.0 * f_d);

    // The integrated fractions should be reasonably close to analytic values.
    // Allow some slack due to finite number of paths and finite observation time.
    CHECK(std::abs(R1_sim - R1_ana) < 0.10); // ±0.10 around ~0.80
    CHECK(std::abs(R2_sim - R2_ana) < 0.07); // ±0.07 around ~0.89

    // Also check monotonicity and sanity
    CHECK(R1_sim > 0.5);    // at least half the power within ±f_d
    CHECK(R2_sim > R1_sim); // more power within ±2f_d than within ±f_d
    CHECK(R2_sim < 1.0);    // not all power concentrated near DC
}

TEST_CASE("IEEE Chan - B + TGac Channel interpolation")
{
    arma::mat rx_pos, rx_orientation;
    std::vector<arma::mat> aod, aoa, pow;
    std::vector<arma::vec> delay;
    std::vector<arma::cube> M;

    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "C", 2.4e9, 5e-9);

    // Check IEEE 802.11-09/0569r0
    arma::vec delay_ns = delay[0] * 1e9;
    CHECK(delay_ns.n_elem == 35ULL);
    arma::vec T = {0, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80, 85, 85, 90, 90, 95, 110, 115, 140, 145, 170, 175, 200};
    CHECK(arma::approx_equal(delay_ns, T, "absdiff", 1e-6));

    arma::vec power_linear = arma::sum(pow[0], 0).t();

    auto linear_2_dB = [](double x)
    { return 10.0 * std::log10(x); };

    power_linear.transform(linear_2_dB);
    power_linear -= power_linear[0] + 3.0103; // KF = 0 dB for first tap

    T = {3.01, 3.01, 1.05, 2.1, 3.2, 4.3, 5.4, 6.5, 7.55, 8.6, 9.7, 10.8, 11.9, 13.0,
         5.0, 14.1, 6.1, 15.2, 7.2, 16.25, 8.25, 17.3, 9.3, 18.4, 10.4, 19.5, 11.5,
         12.05, 13.7, 14.05, 15.8, 16.17, 18.0, 18.37, 20.2};
    CHECK(arma::approx_equal(power_linear, -T, "absdiff", 0.01));
}

TEST_CASE("IEEE Chan - TGn and TGac delay spread")
{
    arma::mat rx_pos, rx_orientation;
    std::vector<arma::mat> aod, aoa, pow;
    std::vector<arma::vec> delay;
    std::vector<arma::cube> M;

    auto linear_2_dB = [](double x)
    { return 10.0 * std::log10(x); };

    auto rms_delay_spread = [](const arma::vec &delay_ns, const arma::vec &power_linear)
    {
        const arma::uword n = delay_ns.n_elem;
        if (power_linear.n_elem != n)
            throw std::invalid_argument("size mismatch");

        const double *d = delay_ns.memptr();
        const double *p = power_linear.memptr();

        double P = 0.0;
        for (arma::uword i = 0; i < n; ++i)
            P += p[i];
        if (!(P > 0.0))
            return 0.0;

        double sum1 = 0.0; // sum(p * d)
        double sum2 = 0.0; // sum(p * d^2)
        for (arma::uword i = 0; i < n; ++i)
        {
            const double di = d[i];
            const double pi = p[i];
            sum1 += pi * di;
            sum2 += pi * di * di;
        }

        const double mean = sum1 / P;
        const double mean_sq = sum2 / P;

        double var = mean_sq - mean * mean;
        if (var < 0.0)
            var = 0.0; // guard tiny negative from rounding
        return std::sqrt(var);
    };

    double drms;
    arma::vec power_linear, pow_log, delay_ns, T;

    // TGn, model B
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "B", 2.4e9, 10e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 15.648) < 0.1);

    pow_log = power_linear;
    pow_log[1] += pow_log[0];
    pow_log = pow_log.tail(pow_log.n_elem - 1);
    pow_log.transform(linear_2_dB);
    pow_log -= pow_log[0];

    T = {0, 10, 20, 20, 30, 30, 40, 40, 50, 60, 70, 80};
    CHECK(arma::approx_equal(delay_ns.tail(delay_ns.n_elem - 1), T, "absdiff", 0.01));

    T = {0, 5.4, 10.8, 3.2, 16.2, 6.3, 21.7, 9.4, 12.5, 15.6, 18.7, 21.8};
    CHECK(arma::approx_equal(pow_log, -T, "absdiff", 0.01));

    // TGn, model C
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "C", 2.4e9, 10e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 33.433) < 0.1);

    pow_log = power_linear;
    pow_log[1] += pow_log[0];
    pow_log = pow_log.tail(pow_log.n_elem - 1);
    pow_log.transform(linear_2_dB);
    pow_log -= pow_log[0];

    T = {0, 10, 20, 30, 40, 50, 60, 60, 70, 70, 80, 80, 90, 90, 110, 140, 170, 200};
    CHECK(arma::approx_equal(delay_ns.tail(delay_ns.n_elem - 1), T, "absdiff", 0.01));

    T = {0, 2.1, 4.3, 6.5, 8.6, 10.8, 13.0, 5.0, 15.2, 7.2, 17.3, 9.3, 19.5, 11.5, 13.7, 15.8, 18.0, 20.2};
    CHECK(arma::approx_equal(pow_log, -T, "absdiff", 0.01));

    // TGn, model D
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "D", 2.4e9, 10e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 49.953) < 0.25);

    pow_log = power_linear;
    pow_log[1] += pow_log[0];
    pow_log = pow_log.tail(pow_log.n_elem - 1);
    pow_log.transform(linear_2_dB);
    pow_log -= pow_log[0];

    T = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 110, 140, 140, 170, 170, 200, 200, 240, 240, 240, 290, 290, 290, 340, 340, 390};
    CHECK(arma::approx_equal(delay_ns.tail(delay_ns.n_elem - 1), T, "absdiff", 0.01));

    T = {0, 0.9, 1.7, 2.6, 3.5, 4.3, 5.2, 6.1, 6.9, 7.8, 9.0, 6.6, 11.1, 9.5, 13.7, 12.1, 16.3, 14.7, 19.3, 17.4, 18.8, 23.2, 21.9, 23.2, 25.5, 25.2, 26.7};
    CHECK(arma::approx_equal(pow_log, -T, "absdiff", 0.01));

    // TGn, model E
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "E", 2.4e9, 10e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 98.99) < 0.1);

    pow_log = power_linear;
    pow_log[1] += pow_log[0];
    pow_log = pow_log.tail(pow_log.n_elem - 1);
    pow_log.transform(linear_2_dB);
    pow_log -= pow_log[0] + 2.6;

    T = {0, 10, 20, 30, 50, 50, 80, 80, 110, 110, 140, 140, 180, 180, 180, 230, 230, 230, 280, 280, 280, 330,
         330, 330, 380, 380, 380, 430, 430, 430, 490, 490, 490, 490, 560, 560, 640, 730};
    CHECK(arma::approx_equal(delay_ns.tail(delay_ns.n_elem - 1), T, "absdiff", 0.01));

    T = {2.6, 3.0, 3.5, 3.9, 4.5, 1.8, 5.6, 3.2, 6.9, 4.5, 8.2, 5.8, 9.8, 7.1, 7.9, 11.7, 9.9, 9.6, 13.9, 10.3,
         14.2, 16.1, 14.3, 13.8, 18.3, 14.7, 18.6, 20.5, 18.7, 18.1, 22.9, 19.9, 22.8, 20.6, 22.4, 20.5, 20.7, 24.6};
    CHECK(arma::approx_equal(pow_log, -T, "absdiff", 0.01));

    // TGn, model F
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "F", 2.4e9, 10e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 148.92) < 0.15);

    pow_log = power_linear;
    pow_log[1] += pow_log[0];
    pow_log = pow_log.tail(pow_log.n_elem - 1);
    pow_log.transform(linear_2_dB);
    pow_log -= pow_log[0] + 3.3;

    T = {0, 10, 20, 30, 50, 50, 80, 80, 110, 110, 140, 140, 180, 180, 180, 230, 230, 230, 280, 280, 280, 330, 330, 330,
         400, 400, 400, 400, 490, 490, 490, 490, 600, 600, 600, 600, 600, 730, 730, 880, 1050};
    CHECK(arma::approx_equal(delay_ns.tail(delay_ns.n_elem - 1), T, "absdiff", 0.01));

    T = {3.3, 3.6, 3.9, 4.2, 4.6, 1.8, 5.3, 2.8, 6.2, 3.5, 7.1, 4.4, 8.2, 5.3, 5.7, 9.5, 7.4, 6.7, 11, 7, 10.4, 12.5, 10.3, 9.6,
         14.3, 10.4, 14.1, 8.8, 16.7, 13.8, 12.7, 13.3, 19.9, 15.7, 18.5, 18.7, 12.9, 19.9, 14.2, 16.3, 21.2};
    CHECK(arma::approx_equal(pow_log, -T, "absdiff", 0.01));

    // TGac, model B
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "B", 2.4e9, 5e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 15.933) < 0.1);

    // TGac, model C
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "C", 2.4e9, 5e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 33.278) < 0.1);

    // TGac, model D
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "D", 2.4e9, 5e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 49.402) < 0.25);

    // TGac, model E
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "E", 2.4e9, 5e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 97.249) < 0.1);

    // TGac, model F
    qd_ieee_indoor_param(rx_pos, rx_orientation, aod, aoa, pow, delay, M, "F", 2.4e9, 5e-9);
    delay_ns = delay[0] * 1e9;
    power_linear = arma::sum(pow[0], 0).t();
    drms = rms_delay_spread(delay_ns, power_linear);
    CHECK(std::abs(drms - 142.14) < 0.1);
}

TEST_CASE("IEEE Chan - Doppler from moving vehicle")
{
    double fGHz = 2.4;
    double tap_spacing_ns = 10;
    double observation_time_s = 2.0;
    double update_rate_s = 0.004;    // 100 Hz sampling -> 0.05 Hz resolution
    double speed_station_kmh = 0.0;  // TGn: stations static
    double speed_env_kmh = 0.0;      // TGn environmental speed
    double speed_vehicle_kmh = 40.0; // TGn environmental speed

    auto ant = quadriga_lib::generate_arrayant_omni<double>(30.0);

    auto chan = quadriga_lib::get_channels_ieee_indoor(
        ant, ant, "F",
        fGHz * 1e9,
        tap_spacing_ns / 1e9,
        1,
        observation_time_s,
        update_rate_s,
        speed_station_kmh,
        speed_env_kmh,
        {}, {}, false, {}, 100, speed_vehicle_kmh, 1234);

    auto P = calc_Doppler_profile(chan[0], update_rate_s);

    arma::rowvec DS_lin_row = arma::sum(P, 0); // sum over delay
    arma::vec DS_lin = DS_lin_row.t();

    auto linear_2_dB = [](double x)
    { return 10.0 * std::log10(x); };

    arma::vec dpp = arma::sum(P, 0).t();
    dpp.transform(linear_2_dB);

    double peak = arma::max(dpp);
    CHECK(std::abs(peak + 55.0) < 2.0);

    peak = dpp[429];
    CHECK(std::abs(peak + 77.0) < 2.0);
}

TEST_CASE("IEEE Chan - Reciprocity")
{
    double fGHz = 2.4;
    double tap_spacing_ns = 5;
    double observation_time_s = 0.0;
    double update_rate_s = 0.004;    // 100 Hz sampling -> 0.05 Hz resolution
    double speed_station_kmh = 0.0;  // TGn: stations static
    double speed_env_kmh = 0.0;      // TGn environmental speed
    double speed_vehicle_kmh = 40.0; // TGn environmental speed

    auto ant_AP = quadriga_lib::generate_arrayant_ula<double>(4, fGHz * 1e9, 0.5);
    auto ant_STA = quadriga_lib::generate_arrayant_ula<double>(2, fGHz * 1e9, 0.7);

    // Symmetric pilot grid
    arma::vec pilot_grid = arma::regspace<arma::vec>(-0.5, 0.01, 0.5);

    // Downlink channel
    auto chan_down = quadriga_lib::get_channels_ieee_indoor(
        ant_AP, ant_STA, "F",
        fGHz * 1e9,
        tap_spacing_ns / 1e9,
        2,
        observation_time_s,
        update_rate_s,
        speed_station_kmh,
        speed_env_kmh,
        {}, {}, false, {}, 100, speed_vehicle_kmh, 1234);

    auto chan_up = quadriga_lib::get_channels_ieee_indoor(
        ant_AP, ant_STA, "F",
        fGHz * 1e9,
        tap_spacing_ns / 1e9,
        2,
        observation_time_s,
        update_rate_s,
        speed_station_kmh,
        speed_env_kmh,
        {}, {}, true, {}, 100, speed_vehicle_kmh, 1234);

    REQUIRE(chan_down[0].n_rx() == 2);
    REQUIRE(chan_down[0].n_tx() == 4);

    REQUIRE(chan_up[0].n_rx() == 4);
    REQUIRE(chan_up[0].n_tx() == 2);

    arma::uvec n_path = chan_down[0].n_path();
    REQUIRE(arma::all(n_path == n_path[0]));

    // Compare in time domain
    double max_err = 0.0;
    for (arma::uword k = 0; k < n_path[0]; ++k)
    {
        arma::mat Hdl = chan_down[0].coeff_re[0].slice(k); // 2x4
        arma::mat Hul = chan_up[0].coeff_re[0].slice(k);   // 4x2
        arma::mat diff = Hul.t() - Hdl;

        double slice_max = arma::abs(diff).max();
        if (slice_max > max_err)
            max_err = slice_max;

        Hdl = chan_down[0].coeff_im[0].slice(k); // 2x4
        Hul = chan_up[0].coeff_im[0].slice(k);   // 4x2
        diff = Hul.t() + Hdl;                    // conjugate

        slice_max = arma::abs(diff).max();
        if (slice_max > max_err)
            max_err = slice_max;

        Hdl = chan_down[0].delay[0].slice(k); // 2x4
        Hul = chan_up[0].delay[0].slice(k);   // 4x2
        diff = Hul.t() - Hdl;

        slice_max = arma::abs(diff).max();
        if (slice_max > max_err)
            max_err = slice_max;
    }
    REQUIRE(max_err < 1e-10);

    // Convert to frequency domain
    arma::cx_cube hmat_down;
    quadriga_lib::baseband_freq_response<double>(
        &chan_down[0].coeff_re[0], &chan_down[0].coeff_im[0], &chan_down[0].delay[0],
        &pilot_grid, 100e6, nullptr, nullptr, &hmat_down);

    arma::cx_cube hmat_up;
    quadriga_lib::baseband_freq_response<double>(
        &chan_up[0].coeff_re[0], &chan_up[0].coeff_im[0], &chan_up[0].delay[0],
        &pilot_grid, 100e6, nullptr, nullptr, &hmat_up);

    // hmat_down: [n_rx_DL, n_tx_DL, n_f]  = [2,4, ...]
    // hmat_up:   [n_rx_UL, n_tx_UL, n_f]  = [4,2, ...]
    // Expect: H_up(f) ≈ H_down(f)^H  (conjugate transpose) for all f

    REQUIRE(hmat_down.n_rows == 2);
    REQUIRE(hmat_down.n_cols == 4);
    REQUIRE(hmat_up.n_rows == 4);
    REQUIRE(hmat_up.n_cols == 2);
    REQUIRE(hmat_down.n_slices == hmat_up.n_slices);

    // NOTE:
    // reciprocity in time domain implies reciprocity in frequency domain with a flipped frequency axis

    max_err = 0.0;
    arma::uword Nf = hmat_down.n_slices;

    for (arma::uword k = 0; k < Nf; ++k)
    {
        arma::uword k_mirror = Nf - 1 - k; // f_k ↔ -f_k (assuming symmetric grid)

        arma::cx_mat Hdl_neg = hmat_down.slice(k_mirror); // H_down(-f)
        arma::cx_mat Hul = hmat_up.slice(k);              // H_up(f)

        // Reciprocity: H_up(f) = H_down(-f)^H
        arma::cx_mat diff = Hul - Hdl_neg.t();

        double slice_max = arma::abs(diff).max();
        if (slice_max > max_err)
            max_err = slice_max;
    }

    REQUIRE(max_err < 1e-10);
}