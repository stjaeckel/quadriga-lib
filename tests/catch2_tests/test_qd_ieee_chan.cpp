// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

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
// - Uplink-downlink reciprocity for MIMO channels (plain transpose, moving station)
// - Doppler sign of a moving station against get_channels_spherical (downlink and uplink)
// - Floors add attenuation but no delay
// - Number of paths independent of distance, floors and walls
// - Invalid XPR, non-finite observation time and update rate
// - Wall penetration loss (default 5 dB, custom 7 dB, per-user vector)
// - TGax floor penetration loss formula (n=2,3,4 vs n=1 reference)

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

    // Delay: IFFT over frequency (rows). Doppler: FFT over time (columns), so that a coefficient
    // that rotates as exp(+j 2 pi f t) (approaching scatterer) appears at +f. Scaled like ifft2.
    arma::cx_mat G = arma::ifft(H);
    G = arma::cx_mat(arma::fft(arma::cx_mat(G.st())).st()) / double(w);

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
    arma::cx_mat S = arma::cx_mat(chan[0].coeff_re[0].slice(0), chan[0].coeff_im[0].slice(0));

    arma::mat A = arma::abs(S);
    A /= std::sqrt(chan[0].path_gain[0][0]);

    CHECK(arma::all(arma::abs(arma::vectorise(A) - 1.0) < 1e-12));

    // Check delays
    auto d = arma::mat(chan[0].delay[0].memptr(), 4 * 4, 13);
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

    d = arma::mat(chan[1].delay[0].memptr(), 4 * 4, 12);
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
    // The uplink is the plain transpose of the downlink for every snapshot and path: the reciprocal channel in
    // the baseband convention of the library (as get_channels_spherical with TX and RX swapped). A moving
    // station gives the same Doppler shift in both directions; a conjugate transpose would reverse it.
    double fc = 2.4e9;
    auto ant_AP = quadriga_lib::generate_arrayant_ula<double>(4, fc, 0.5);
    auto ant_STA = quadriga_lib::generate_arrayant_ula<double>(2, fc, 0.7);

    arma::vec pilot_grid = arma::regspace<arma::vec>(-0.5, 0.01, 0.5);

    for (std::string type : {"B", "F"})
    {
        auto chan_down = quadriga_lib::get_channels_ieee_indoor(ant_AP, ant_STA, type, fc, 5e-9, 2, 0.02, 0.004, 5.0, 1.2,
                                                                {}, {}, false, {}, 100, 40.0, 1234);
        auto chan_up = quadriga_lib::get_channels_ieee_indoor(ant_AP, ant_STA, type, fc, 5e-9, 2, 0.02, 0.004, 5.0, 1.2,
                                                              {}, {}, true, {}, 100, 40.0, 1234);
        REQUIRE(chan_down.size() == 2);
        REQUIRE(chan_up.size() == 2);

        for (size_t u = 0; u < 2; ++u)
        {
            const auto &dl = chan_down[u];
            const auto &ul = chan_up[u];

            REQUIRE(dl.n_rx() == 2);
            REQUIRE(dl.n_tx() == 4);
            REQUIRE(ul.n_rx() == 4);
            REQUIRE(ul.n_tx() == 2);
            REQUIRE(dl.n_snap() == 6);
            REQUIRE(ul.n_snap() == 6);

            // Positions and orientations are swapped; the moving station is the transmitter in the uplink
            CHECK(arma::approx_equal(ul.tx_pos, dl.rx_pos, "absdiff", 0.0));
            CHECK(arma::approx_equal(ul.rx_pos, dl.tx_pos, "absdiff", 0.0));
            CHECK(arma::approx_equal(ul.tx_orientation, dl.rx_orientation, "absdiff", 0.0));
            CHECK(arma::approx_equal(ul.rx_orientation, dl.tx_orientation, "absdiff", 0.0));
            CHECK(ul.tx_pos.n_cols == 6);

            double max_err = 0.0, max_err_conj = 0.0;
            for (arma::uword s = 0; s < 6; ++s)
            {
                REQUIRE(ul.n_path()[s] == dl.n_path()[s]);
                CHECK(arma::approx_equal(ul.path_gain[s], dl.path_gain[s], "absdiff", 0.0));

                for (arma::uword k = 0; k < dl.n_path()[s]; ++k)
                {
                    arma::mat Hdl_re = dl.coeff_re[s].slice(k), Hul_re = ul.coeff_re[s].slice(k);
                    arma::mat Hdl_im = dl.coeff_im[s].slice(k), Hul_im = ul.coeff_im[s].slice(k);
                    arma::mat Ddl = dl.delay[s].slice(k), Dul = ul.delay[s].slice(k);

                    max_err = std::max(max_err, arma::abs(Hul_re.t() - Hdl_re).max());
                    max_err = std::max(max_err, arma::abs(Hul_im.t() - Hdl_im).max());
                    max_err = std::max(max_err, arma::abs(Dul.t() - Ddl).max());
                    max_err_conj = std::max(max_err_conj, arma::abs(Hul_im.t() + Hdl_im).max());
                }

                // Frequency domain: H_up(f) = H_down(f)^T at every frequency
                arma::cx_cube hmat_down, hmat_up;
                quadriga_lib::baseband_freq_response<double>(&dl.coeff_re[s], &dl.coeff_im[s], &dl.delay[s],
                                                             &pilot_grid, 100e6, nullptr, nullptr, &hmat_down);
                quadriga_lib::baseband_freq_response<double>(&ul.coeff_re[s], &ul.coeff_im[s], &ul.delay[s],
                                                             &pilot_grid, 100e6, nullptr, nullptr, &hmat_up);
                REQUIRE(hmat_up.n_slices == hmat_down.n_slices);
                for (arma::uword f = 0; f < hmat_down.n_slices; ++f)
                    max_err = std::max(max_err, arma::abs(hmat_up.slice(f) - hmat_down.slice(f).st()).max());
            }
            CHECK(max_err < 1e-12);
            CHECK(max_err_conj > 1e-6); // the coefficients are complex, so a conjugate would differ
        }
    }
}

TEST_CASE("IEEE Chan - B + Wall loss")
{
    // Wall loss is a deterministic additive term in the path loss. With a fixed
    // seed and otherwise identical inputs, walls don't change path structure or
    // RNG draws, so the SF realisation is identical and the total power ratio
    // in dB equals -n_walls * wall_loss exactly (per user).

    auto ant = quadriga_lib::generate_arrayant_omni<double>(30.0);
    arma::sword seed = 42;

    // ---- Single user, default wall_loss = 5.0 dB ----------------------------
    auto chan_ref = quadriga_lib::get_channels_ieee_indoor(
        ant, ant, "B", 5.25e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2,
        {3.0}, {0}, false, {}, 20, 50.0, seed,
        NAN, NAN, NAN, NAN, NAN, {0}, 5.0);

    auto chan_w3 = quadriga_lib::get_channels_ieee_indoor(
        ant, ant, "B", 5.25e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2,
        {3.0}, {0}, false, {}, 20, 50.0, seed,
        NAN, NAN, NAN, NAN, NAN, {3}, 5.0);

    REQUIRE(arma::all(chan_ref[0].n_path() == chan_w3[0].n_path()));

    double P_ref = arma::accu(chan_ref[0].path_gain[0]);
    double P_w3 = arma::accu(chan_w3[0].path_gain[0]);
    CHECK(std::abs(10.0 * std::log10(P_w3 / P_ref) + 15.0) < 1e-10); // -3 * 5 dB

    // ---- Single user, custom wall_loss = 7.0 dB (alternative TGax value) ----
    auto chan_w2_7 = quadriga_lib::get_channels_ieee_indoor(
        ant, ant, "B", 5.25e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2,
        {3.0}, {0}, false, {}, 20, 50.0, seed,
        NAN, NAN, NAN, NAN, NAN, {2}, 7.0);

    REQUIRE(arma::all(chan_ref[0].n_path() == chan_w2_7[0].n_path()));
    double P_w2_7 = arma::accu(chan_w2_7[0].path_gain[0]);
    CHECK(std::abs(10.0 * std::log10(P_w2_7 / P_ref) + 14.0) < 1e-10); // -2 * 7 dB

    // ---- Per-user n_walls vector --------------------------------------------
    arma::vec Dist3 = {3.0, 3.0, 3.0};
    arma::uvec Floors3 = {0, 0, 0};
    arma::uvec Walls0 = {0, 0, 0};
    arma::uvec WallsN = {1, 2, 3};

    auto chan3_ref = quadriga_lib::get_channels_ieee_indoor(
        ant, ant, "B", 5.25e9, 10e-9, 3, 0.0, 1e-3, 0.0, 1.2,
        Dist3, Floors3, false, {}, 20, 50.0, seed,
        NAN, NAN, NAN, NAN, NAN, Walls0, 5.0);

    auto chan3_w = quadriga_lib::get_channels_ieee_indoor(
        ant, ant, "B", 5.25e9, 10e-9, 3, 0.0, 1e-3, 0.0, 1.2,
        Dist3, Floors3, false, {}, 20, 50.0, seed,
        NAN, NAN, NAN, NAN, NAN, WallsN, 5.0);

    REQUIRE(chan3_ref.size() == 3);
    for (arma::uword u = 0; u < 3; ++u)
    {
        REQUIRE(arma::all(chan3_ref[u].n_path() == chan3_w[u].n_path()));
        double pr = arma::accu(chan3_ref[u].path_gain[0]);
        double pw = arma::accu(chan3_w[u].path_gain[0]);
        double expected = -static_cast<double>(u + 1) * 5.0; // 1, 2, 3 walls
        CHECK(std::abs(10.0 * std::log10(pw / pr) - expected) < 1e-10);
    }
}

TEST_CASE("IEEE Chan - B + TGax floor loss formula")
{
    // For TGax (CarrierFreq >= 1 GHz) the per-floor penetration loss is
    //   FL(n) = 18.3 * n^((n+2)/(n+1) - 0.46)   [dB], for n_floors >= 1
    // (See IEEE 802.11-14/0882r4, Section 3.1.1.) Comparing two runs that both
    // have n_floors > 0 ensures both lose the LOS path, so the path structure
    // and SF realisation are identical at a fixed seed; the total power ratio
    // is then determined purely by the floor-loss term.

    auto ant = quadriga_lib::generate_arrayant_omni<double>(30.0);
    arma::sword seed = 7;

    auto FL = [](double n)
    {
        return 18.3 * std::pow(n, (n + 2.0) / (n + 1.0) - 0.46);
    };

    auto run = [&](arma::uword nf)
    {
        return quadriga_lib::get_channels_ieee_indoor(
            ant, ant, "B", 5.25e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2,
            {3.0}, arma::uvec{nf}, false, {}, 20, 50.0, seed);
    };

    auto chan1 = run(1);
    double P1 = arma::accu(chan1[0].path_gain[0]);

    for (arma::uword nf : {2u, 3u, 4u})
    {
        auto chan_n = run(nf);
        REQUIRE(arma::all(chan_n[0].n_path() == chan1[0].n_path()));
        double Pn = arma::accu(chan_n[0].path_gain[0]);
        double expected = -(FL(static_cast<double>(nf)) - FL(1.0));
        CHECK(std::abs(10.0 * std::log10(Pn / P1) - expected) < 1e-10);
    }
}
TEST_CASE("IEEE Chan - Doppler sign of a moving station matches get_channels_spherical")
{
    // A station moving along the track given in rx_pos (downlink) or tx_pos (uplink) must see the same phase
    // evolution as the deterministic channel of get_channels_spherical along that track: approaching the AP
    // shortens the path, the coefficient (proportional to exp(-j k d)) rotates as exp(+j 2 pi f_D t), f_D > 0.
    // The uplink (reciprocal channel) has the same Doppler shift as the downlink.

    double fc = 2.4e9, v_kmh = 5.0, dt = 1.0e-3;
    auto ap = quadriga_lib::generate_arrayant_ula<double>(2, fc, 0.5);
    auto sta = quadriga_lib::generate_arrayant_ula<double>(2, fc, 0.5);

    // One user; AoD LOS = 30 deg, AoA LOS = 50 deg: the station heads 50 deg off the direction to the AP
    arma::mat offsets(4, 1);
    offsets.col(0) = arma::vec({30.0, 0.0, 50.0, 0.0});
    arma::mat M_los = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0});

    double fD = v_kmh / 3.6 / (299792458.0 / fc) * std::cos(50.0 * arma::datum::pi / 180.0);
    double step_expected = 2.0 * arma::datum::pi * fD * dt; // > 0, approaching

    for (bool uplink : {false, true})
    {
        // Model A with a large K-factor: path 0 is the LOS path
        auto chan = quadriga_lib::get_channels_ieee_indoor(ap, sta, "A", fc, 10e-9, 1, 0.01, dt, v_kmh, 0.0, {50.0}, {0},
                                                           uplink, offsets, 20, 0.0, 11, 1.0e6);
        const auto &c = chan[0];
        REQUIRE(c.n_snap() == 11);

        arma::mat track = uplink ? c.tx_pos : c.rx_pos;
        arma::vec o_sta = uplink ? c.tx_orientation.col(0) : c.rx_orientation.col(0);
        REQUIRE(track.n_cols == 11);

        std::vector<arma::cx_mat> H_ieee, H_sph;
        for (arma::uword s = 0; s < 11; ++s)
        {
            arma::vec p = track.col(s);
            arma::mat sc = 0.5 * p;
            arma::vec pg = {1.0}, pl = {std::sqrt(p(0) * p(0) + p(1) * p(1) + p(2) * p(2))};
            arma::cube cr, ci, dly;
            if (uplink)
                quadriga_lib::get_channels_spherical<double>(&sta, &ap, p(0), p(1), p(2), o_sta(0), o_sta(1), o_sta(2),
                                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &sc, &sc, &pg, &pl, &M_los,
                                                             &cr, &ci, &dly, fc);
            else
                quadriga_lib::get_channels_spherical<double>(&ap, &sta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                             p(0), p(1), p(2), o_sta(0), o_sta(1), o_sta(2),
                                                             &sc, &sc, &pg, &pl, &M_los, &cr, &ci, &dly, fc);
            H_sph.push_back(arma::cx_mat(cr.slice(0), ci.slice(0)));
            H_ieee.push_back(arma::cx_mat(c.coeff_re[s].slice(0), c.coeff_im[s].slice(0)));
        }

        // Phase evolution relative to the first snapshot, per MIMO sub-link
        double max_err = 0.0;
        for (arma::uword s = 1; s < 11; ++s)
        {
            arma::cx_mat R_ieee = H_ieee[s] / H_ieee[0];
            arma::cx_mat R_sph = H_sph[s] / H_sph[0];
            max_err = std::max(max_err, arma::abs(R_ieee - R_sph).max());
        }
        CHECK(max_err < 1.0e-3);

        double step = std::arg(H_ieee[1](0, 0) / H_ieee[0](0, 0));
        CHECK(std::abs(step - step_expected) < 1.0e-6);
    }
}

TEST_CASE("IEEE Chan - Floors add attenuation, not delay")
{
    // n_floors adds floor penetration loss (and the TGah shadow fading below 1 GHz) but no delay: the station
    // is placed 3 m per floor above the AP, and the same 3-D distance is used for the path lengths and the
    // relative delays, so the first tap stays at 0 for any number of floors.
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    arma::vec T = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};

    for (double fc : {5.25e9, 0.9e9})
    {
        auto ref = quadriga_lib::get_channels_ieee_indoor(omni, omni, "B", fc, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {5.0}, {0},
                                                          false, {}, 20, 50.0, 7, NAN, NAN, 0.0, 0.0);
        REQUIRE(ref[0].n_path()[0] == 12);

        for (arma::uword nf = 1; nf <= 4; ++nf)
        {
            auto ch = quadriga_lib::get_channels_ieee_indoor(omni, omni, "B", fc, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {5.0}, {nf},
                                                             false, {}, 20, 50.0, 7, NAN, NAN, 0.0, 0.0);
            REQUIRE(ch[0].n_path()[0] == 12);

            CHECK(std::abs(ch[0].delay[0](0, 0, 0)) < 1.0e-15);
            arma::vec d = arma::vectorise(ch[0].delay[0]);
            CHECK(arma::approx_equal(d, arma::vectorise(ref[0].delay[0]), "absdiff", 1.0e-18));

            arma::vec du = arma::unique(d) * 1.0e9;
            REQUIRE(du.n_elem == T.n_elem);
            CHECK(arma::approx_equal(du, T, "absdiff", 1.0e-6));

            CHECK(ch[0].rx_pos(2, 0) == 3.0 * double(nf));

            // Floors attenuate
            CHECK(arma::accu(ch[0].path_gain[0]) < arma::accu(ref[0].path_gain[0]));
        }
    }
}

TEST_CASE("IEEE Chan - Number of paths does not depend on distance, floors and walls")
{
    // Zero-power taps are removed from the normalised power-delay profile, before path loss and shadow fading,
    // so the number of paths (and the delay spread) does not depend on the link budget.
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);

    for (double fc : {5.25e9, 0.9e9})
        for (double d : {10.0, 30.0, 40.0, 60.0, 100.0, 300.0})
            for (arma::uword nf : {0, 2, 4})
                for (arma::uword nw : {0, 6})
                {
                    std::vector<quadriga_lib::channel<double>> ch;
                    REQUIRE_NOTHROW(ch = quadriga_lib::get_channels_ieee_indoor(omni, omni, "F", fc, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2,
                                                                                {d}, {nf}, false, {}, 20, 0.0, 7,
                                                                                NAN, NAN, 0.0, 0.0, NAN, {nw}, 5.0));
                    arma::uword n_expected = (d < 30.0 && nf == 0) ? 42 : 41; // LOS only below dBP on the same floor
                    CHECK(ch[0].n_path()[0] == n_expected);
                    CHECK(ch[0].coeff_re[0].is_finite());
                    CHECK(ch[0].coeff_im[0].is_finite());
                    CHECK(arma::all(ch[0].path_gain[0] > 0.0));
                }
}

TEST_CASE("IEEE Chan - Invalid XPR_NLOS_linear")
{
    // XPR = 0 (no co-polar power) and XPR = Inf (no cross-polar power) give NaN coefficients; they and negative
    // values are rejected. NaN selects the model default (2, i.e. 3 dB).
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto run = [&](double xpr)
    {
        return quadriga_lib::get_channels_ieee_indoor(omni, omni, "B", 5.25e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {3.0}, {0},
                                                      false, {}, 20, 50.0, 7, NAN, xpr);
    };

    for (double xpr : {0.0, -1.0, -arma::datum::inf, arma::datum::inf})
        CHECK_THROWS_AS(run(xpr), std::invalid_argument);

    for (double xpr : {1.0e-6, 2.0, 1.0e6})
    {
        auto ch = run(xpr);
        CHECK(ch[0].coeff_re[0].is_finite());
        CHECK(ch[0].coeff_im[0].is_finite());
        CHECK(std::isfinite(arma::accu(ch[0].path_gain[0])));
    }

    auto ch_nan = run(NAN), ch_2 = run(2.0);
    CHECK(arma::approx_equal(ch_nan[0].coeff_re[0], ch_2[0].coeff_re[0], "absdiff", 0.0));
    CHECK(arma::approx_equal(ch_nan[0].coeff_im[0], ch_2[0].coeff_im[0], "absdiff", 0.0));
}

TEST_CASE("IEEE Chan - Non-finite observation time and update rate")
{
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto run = [&](double observation_time, double update_rate)
    {
        return quadriga_lib::get_channels_ieee_indoor(omni, omni, "A", 5.25e9, 10e-9, 1, observation_time, update_rate);
    };

    CHECK_THROWS_AS(run(INFINITY, 1e-3), std::invalid_argument);
    CHECK_THROWS_AS(run(NAN, 1e-3), std::invalid_argument);
    CHECK_THROWS_AS(run(0.1, INFINITY), std::invalid_argument);
    CHECK_THROWS_AS(run(0.1, NAN), std::invalid_argument);
    CHECK_THROWS_AS(run(1.0e30, 1e-3), std::invalid_argument); // snapshot count not representable
    CHECK_THROWS_AS(run(-1.0, 1e-3), std::invalid_argument);
    CHECK_THROWS_AS(run(0.1, 0.0), std::invalid_argument);

    CHECK(run(0.0, 1e-3)[0].n_snap() == 1);
    CHECK(run(0.01, 1e-3)[0].n_snap() == 11);
}

TEST_CASE("IEEE Chan - Tap spacing 10 ns / 2^k and non-finite carrier frequency, distance and speeds")
{
    // A tap spacing that is not 10 ns / 2^k (e.g. 7 ns, which rounded to a factor of 1 and was treated as 10 ns)
    // throws; so do NaN and Inf carrier frequency, distance and speeds
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto run = [&](double fc, double tap, double dist, double v_sta, double v_env)
    {
        return quadriga_lib::get_channels_ieee_indoor(omni, omni, "B", fc, tap, 1, 0.0, 1e-3, v_sta, v_env, {dist}, {0},
                                                      false, {}, 20, 50.0, 7);
    };
    const double fc = 5.25e9, d = 3.0;

    for (double tap : {7.0e-9, 12.0e-9, 4.0e-9, 3.0e-9, 5.000001e-9, 0.0, -10.0e-9, (double)NAN, (double)INFINITY, 1.0e-300})
    {
        INFO("tap " << tap);
        CHECK_THROWS_AS(run(fc, tap, d, 0.0, 1.2), std::invalid_argument);
    }
    for (int k = 0; k <= 6; ++k)
    {
        INFO("k " << k);
        const double tap = 10.0e-9 / std::pow(2.0, k);
        CHECK_NOTHROW(run(fc, tap, d, 0.0, 1.2));
        CHECK_NOTHROW(run(fc, tap * (1.0 + 5e-10), d, 0.0, 1.2)); // within the relative tolerance 1e-9
    }

    for (double bad : {(double)NAN, (double)INFINITY, -(double)INFINITY})
    {
        INFO("bad " << bad);
        CHECK_THROWS_AS(run(bad, 10e-9, d, 0.0, 1.2), std::invalid_argument);
        CHECK_THROWS_AS(run(fc, 10e-9, bad, 0.0, 1.2), std::invalid_argument);
        CHECK_THROWS_AS(run(fc, 10e-9, d, bad, 1.2), std::invalid_argument);
        CHECK_THROWS_AS(run(fc, 10e-9, d, 0.0, bad), std::invalid_argument);
    }
}
