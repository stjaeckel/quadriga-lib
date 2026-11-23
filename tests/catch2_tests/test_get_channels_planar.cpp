// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// get_channels_planar and get_channels_planar_vec: amplitudes, delays and Doppler of hand-placed paths between omni
// elements (exact by construction), the coupling matrix, the slot layout of add_fake_los_path, zero paths, the tilt
// sign, aliasing and shape checks; the AVX2 kernel and the vec form are compared with the GENERIC single call.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "quadriga_lib.hpp"

#include <iostream>
#include <string>

TEST_CASE("Get Channels Planar - Minimal test")
{

    float pi = 3.141592653589793f, pid180 = 0.017453292519943f;

    auto ant = quadriga_lib::generate_antenna<float>("omni", 1.0);
    quadriga_lib::copy_element(ant, 0, {1});
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

    arma::fcube coeff_re, coeff_im, delay;
    arma::fvec rx_Doppler;

    quadriga_lib::get_channels_planar<float>(ant, ant,
                                             {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                             {20.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                             aod, eod, aoa, eoa, path_gain, path_length, M,
                                             coeff_re, coeff_im, delay, 2997924580.0, true, false,
                                             &rx_Doppler);

    arma::fcube amp = arma::sqrt(coeff_re % coeff_re + coeff_im % coeff_im);
    CHECK(std::abs(amp.at(0, 0, 0) - 1.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(1, 0, 0) - 2.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(0, 1, 0) - 2.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(1, 1, 0) - 4.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(0, 0, 1) - 0.5f) < 1.0e-4f);
    CHECK(std::abs(amp.at(1, 0, 1) - 1.0f) < 1.0e-4f);
    CHECK(std::abs(amp.at(0, 1, 1) - 1.0f) < 1.0e-4f);
    CHECK(std::abs(amp.at(1, 1, 1) - 2.0f) < 1.0e-4f);

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

    float Doppler = std::cos(aoa.at(1)) * std::cos(eoa.at(1));
    REQUIRE(rx_Doppler.n_elem == 2);
    CHECK(std::abs(rx_Doppler.at(0) + 1.0f) < 1.0e-6f);
    CHECK(std::abs(rx_Doppler.at(1) - Doppler) < 1.0e-6f);
}

TEST_CASE("Get Channels Planar - Fake LOS")
{

    float pi = 3.141592653589793f, pid180 = 0.017453292519943f;
    auto ant = quadriga_lib::generate_antenna<float>("omni", 1.0);

    // The LOS path is the second input path

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

    arma::fcube coeff_re, coeff_im, delay;

    quadriga_lib::get_channels_planar<float>(ant, ant,
                                             {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                             {20.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                             aod, eod, aoa, eoa, path_gain, path_length, M,
                                             coeff_re, coeff_im, delay, 2997924580.0, true, true);

    REQUIRE(coeff_re.n_slices == 3);
    REQUIRE(coeff_im.n_slices == 3);
    REQUIRE(delay.n_slices == 3);

    arma::fcube amp = arma::sqrt(coeff_re % coeff_re + coeff_im % coeff_im);
    CHECK(std::abs(amp.at(0, 0, 0) - 1.0f) < 1.0e-6f);
    CHECK(std::abs(amp.at(0, 0, 1) - 0.5f) < 1.0e-6f);
    CHECK(std::abs(amp.at(0, 0, 2) - 0.0f) < 1.0e-6f);

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

    auto ant = quadriga_lib::generate_antenna<double>("omni", 1.0);
    quadriga_lib::copy_element(ant, 0, {2});
    ant.element_pos(1, 0) = 1.0;
    ant.element_pos(1, 2) = -1.0;

    arma::vec aod(1), eod(1), aoa(1), eoa(1), path_gain(1), path_length(1);
    aoa.at(0) = pi;
    path_length(0) = 20.0;
    path_gain(0) = 1;

    arma::mat M(8, 1);
    M(0, 0) = 1.0;
    M(6, 0) = -1.0;

    arma::cube coeff_re, coeff_im, delay;

    // Case 1 : No coupling (identity matrix)
    quadriga_lib::get_channels_planar<double>(ant, ant,
                                              {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                              {20.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                              aod, eod, aoa, eoa, path_gain, path_length, M,
                                              coeff_re, coeff_im, delay, 0.0, false, true);

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
    ant.port_weights_re.zeros();
    ant.port_weights_re.col(0).ones();
    ant.port_weights_im.reset();

    quadriga_lib::get_channels_planar<double>(ant, ant,
                                              {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                              {20.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                              aod, eod, aoa, eoa, path_gain, path_length, M,
                                              coeff_re, coeff_im, delay, 0.0, true, false);
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
    ant.port_weights_re.ones(3, 1);
    ant.port_weights_re *= 2.0;

    quadriga_lib::get_channels_planar<double>(ant, ant,
                                              {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                              {20.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                              aod, eod, aoa, eoa, path_gain, path_length, M,
                                              coeff_re, coeff_im, delay, 0.0, true, false);

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

TEST_CASE("Get Channels Planar - add_fake_los_path layout")
{
    // Always one extra output path; slot 0 holds the LOS path (zero power if none was found);
    // if a LOS path was present, its original slot holds a zero-power path.
    const double C = 299792458.0;
    auto omni = quadriga_lib::generate_antenna<double>("omni", 10.0);
    arma::vec aod = {0.5, 0.0, -0.4}, eod = {0.0, 0.0, 0.1}, aoa = {2.5, arma::datum::pi, -2.7}, eoa = {0.0, 0.0, 0.1};
    arma::vec G = {0.5, 1.0, 0.25};
    arma::mat M(8, 3, arma::fill::zeros);
    M.row(0).fill(1.0), M.row(6).fill(-1.0);
    arma::cube cr, ci, dl;

    // Case 1: LOS path at input index 1 (path_length = TX-RX distance 50 m)
    arma::vec PL = {60.0, 50.0, 70.0};
    quadriga_lib::get_channels_planar<double>(omni, omni, {0, 0, 0}, {0, 0, 0}, {50, 0, 0}, {0, 0, 0},
                                              aod, eod, aoa, eoa, G, PL, M, cr, ci, dl, 1e9, true, true);
    REQUIRE(cr.n_slices == 4);
    arma::vec p = arma::vectorise(arma::square(cr) + arma::square(ci));
    CHECK(std::abs(p(0) - 1.0) < 1e-9);
    CHECK(std::abs(p(1) - 0.5) < 1e-9);
    CHECK(p(2) == 0.0);
    CHECK(std::abs(p(3) - 0.25) < 1e-9);
    CHECK(std::abs(dl(0) * C - 50.0) < 1e-6);
    CHECK(std::abs(dl(2) * C - 50.0) < 1e-6);

    // Case 2: no LOS path
    PL(1) = 55.0;
    quadriga_lib::get_channels_planar<double>(omni, omni, {0, 0, 0}, {0, 0, 0}, {50, 0, 0}, {0, 0, 0},
                                              aod, eod, aoa, eoa, G, PL, M, cr, ci, dl, 1e9, true, true);
    REQUIRE(cr.n_slices == 4);
    p = arma::vectorise(arma::square(cr) + arma::square(ci));
    CHECK(p(0) == 0.0);
    CHECK(std::abs(p(1) - 0.5) < 1e-9);
    CHECK(std::abs(p(2) - 1.0) < 1e-9);
    CHECK(std::abs(p(3) - 0.25) < 1e-9);
    CHECK(std::abs(dl(0) * C - 50.0) < 1e-6);
    CHECK(std::abs(dl(2) * C - 55.0) < 1e-6);
}

TEST_CASE("Get Channels Planar - 0 paths with add_fake_los_path")
{
    // The output holds exactly the zero-power LOS path; its delays per antenna pair equal those of a real LOS path
    // with the same geometry (which is moved to slot 0); without add_fake_los_path, the outputs are empty
    const double C = 299792458.0;
    auto tx = quadriga_lib::generate_antenna<double>("omni", 10.0);
    quadriga_lib::copy_element(tx, 0, {1});
    tx.element_pos = {{0.3, -0.2}, {0.1, 0.4}, {0.0, 0.2}};
    auto rx = tx;
    rx.port_weights_re = arma::vec({1.0, 0.5}); // 2 elements, 1 port
    rx.port_weights_im = arma::vec({0.0, 0.2});

    const double Tx = 1, Ty = 2, Tz = 3, Rx = 11, Ry = -3, Rz = 5;
    const double dx = Rx - Tx, dy = Ry - Ty, dz = Rz - Tz, dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    arma::vec e, aod = {std::atan2(dy, dx)}, eod = {std::atan2(dz, std::sqrt(dx * dx + dy * dy))};
    arma::vec aoa = {std::atan2(-dy, -dx)}, eoa = {-eod(0)}, gain = {1.0}, len = {dist};
    arma::mat M0(8, 0), M = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0};
    M = M.t();

    for (bool abs_delays : {true, false})
    {
        arma::cube cr, ci, dl, crL, ciL, dlL;
        arma::vec dop, dopL;
        quadriga_lib::get_channels_planar<double>(tx, rx, {Tx, Ty, Tz}, {0, 0, 0.3}, {Rx, Ry, Rz}, {0, 0, -0.2},
                                                  e, e, e, e, e, e, M0, cr, ci, dl, 3e9, abs_delays, true, &dop);
        quadriga_lib::get_channels_planar<double>(tx, rx, {Tx, Ty, Tz}, {0, 0, 0.3}, {Rx, Ry, Rz}, {0, 0, -0.2},
                                                  aod, eod, aoa, eoa, gain, len, M, crL, ciL, dlL, 3e9, abs_delays, true, &dopL);
        REQUIRE(cr.n_rows == 1);
        REQUIRE(cr.n_cols == 2);
        REQUIRE(cr.n_slices == 1);
        REQUIRE(dl.n_slices == 1);
        REQUIRE(dop.n_elem == 1);
        CHECK(arma::all(arma::vectorise(cr) == 0.0));
        CHECK(arma::all(arma::vectorise(ci) == 0.0));
        CHECK(dop(0) == 0.0);
        CHECK(arma::approx_equal(dl.slice(0), dlL.slice(0), "absdiff", 1e-18));
        if (abs_delays)
            CHECK(std::abs(arma::mean(arma::vectorise(dl)) * C - dist) < 0.5);
        else
            CHECK(std::abs(arma::mean(arma::vectorise(dl)) * C) < 0.5);
    }

    // Single-element antennas: delay = TX-RX distance / c, or 0 for relative delays
    auto omni = quadriga_lib::generate_antenna<double>("omni", 10.0);
    arma::cube cr, ci, dl;
    quadriga_lib::get_channels_planar<double>(omni, omni, {0, 0, 0}, {0, 0, 0}, {10, 0, 0}, {0, 0, 0},
                                              e, e, e, e, e, e, M0, cr, ci, dl, 1e9, true, true);
    REQUIRE(dl.n_elem == 1);
    CHECK(std::abs(dl(0) - 10.0 / C) < 1e-20);
    CHECK(cr(0) == 0.0);
    CHECK(ci(0) == 0.0);
    quadriga_lib::get_channels_planar<double>(omni, omni, {0, 0, 0}, {0, 0, 0}, {10, 0, 0}, {0, 0, 0},
                                              e, e, e, e, e, e, M0, cr, ci, dl, 1e9, false, true);
    CHECK(dl(0) == 0.0);
    quadriga_lib::get_channels_planar<double>(omni, omni, {0, 0, 0}, {0, 0, 0}, {10, 0, 0}, {0, 0, 0},
                                              e, e, e, e, e, e, M0, cr, ci, dl, 1e9, true, false);
    CHECK(dl.n_slices == 0); // without add_fake_los_path: empty outputs
}

TEST_CASE("Get Channels Planar - 0 paths without add_fake_los_path")
{
    // Empty outputs of size [n_rx_ports, n_tx_ports, 0], also into outputs that held data before
    auto tx = quadriga_lib::generate_antenna<double>("omni", 10.0);
    quadriga_lib::copy_element(tx, 0, {1});
    auto rx = tx;
    rx.port_weights_re = arma::vec({1.0, 0.5}); // 2 elements, 1 port
    rx.port_weights_im = arma::vec({0.0, 0.2});
    arma::vec e, dop(3, arma::fill::ones);
    arma::mat M0(8, 0), Mx; // M may also be 0x0
    arma::cube cr(2, 2, 5, arma::fill::ones), ci, dl;
    for (const arma::mat *M : {&M0, &Mx})
    {
        quadriga_lib::get_channels_planar<double>(tx, rx, {0, 0, 0}, {0, 0, 0}, {10, 0, 0}, {0, 0, 0},
                                                  e, e, e, e, e, e, *M, cr, ci, dl, 1e9, true, false, &dop);
        for (const arma::cube *c : {&cr, &ci, &dl})
        {
            CHECK(c->n_rows == 1);
            CHECK(c->n_cols == 2);
            CHECK(c->n_slices == 0);
        }
        CHECK(dop.n_elem == 0);
    }
    // Inconsistent path counts remain an error
    arma::vec one = {1.0};
    CHECK_THROWS_AS(quadriga_lib::get_channels_planar<double>(tx, rx, {0, 0, 0}, {0, 0, 0}, {10, 0, 0}, {0, 0, 0},
                                                              one, e, e, e, e, e, M0, cr, ci, dl, 1e9, true, false),
                    std::invalid_argument);
}

TEST_CASE("Get Channels Planar - outputs that alias inputs")
{
    auto ant = quadriga_lib::generate_antenna<double>("omni", 10.0);
    arma::vec aod = {0.0}, eod = {0.0}, aoa = {arma::datum::pi}, eoa = {0.0}, gain = {1.0}, length = {20.0};
    arma::mat M = arma::mat({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}).t();
    arma::cube cr, ci, dl;
    arma::vec doppler;
    auto call = [&](arma::cube *c_re, arma::cube *d, arma::vec *dop, const arma::vec *g, bool fake_los)
    {
        quadriga_lib::get_channels_planar<double>(ant, ant, {0, 0, 0}, {0, 0, 0}, {20, 0, 0}, {0, 0, 0},
                                                  aod, eod, aoa, eoa, *g, length, M, *c_re, ci, *d,
                                                  1e9, false, fake_los, dop);
    };
    CHECK_NOTHROW(call(&cr, &dl, &doppler, &gain, true));

    SECTION("Output is an input")
    {
        CHECK_THROWS_AS(call(&cr, &dl, &gain, &gain, true), std::invalid_argument);
        CHECK_THROWS_AS(call(&cr, &dl, &aod, &gain, false), std::invalid_argument);
    }
    SECTION("Two outputs are the same object")
    {
        CHECK_THROWS_AS(call(&cr, &cr, &doppler, &gain, false), std::invalid_argument);
    }
    SECTION("Output shares memory with an input")
    {
        arma::vec gain_on_cr(cr.memptr(), 1, false, true); // cr is [1, 1, 2] from the baseline call
        CHECK_THROWS_AS(call(&cr, &dl, &doppler, &gain_on_cr, false), std::invalid_argument);
    }
}

TEST_CASE("get_channels_planar - a positive tilt points down (3GPP TR 38.901)")
{
    // The path leaves the TX 45 degree downward and arrives at the RX from 45 degree above, from the -x side. A beam
    // along x catches it only when turned toward it: TX tilt +pi/4 (down); RX heading pi and tilt -pi/4 (up).
    const double q = arma::datum::pi / 4.0;
    const auto beam = quadriga_lib::generate_antenna<double>("custom", 5.0, NAN, {40.0, 40.0});
    const auto omni = quadriga_lib::generate_antenna<double>("omni", 5.0);
    const arma::vec aod = {0.0}, eod = {-q}, aoa = {arma::datum::pi}, eoa = {q}, gain = {1.0}, length = {20.0};
    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0, M(6, 0) = -1.0;
    auto power = [&](const quadriga_lib::arrayant<double> &tx, const quadriga_lib::arrayant<double> &rx, double Tt, double Rt)
    {
        arma::cube cr, ci, dl;
        quadriga_lib::get_channels_planar<double>(tx, rx, {0.0, 0.0, 10.0}, {0.0, Tt, 0.0}, {10.0, 0.0, 0.0}, {0.0, Rt, arma::datum::pi},
                                                  aod, eod, aoa, eoa, gain, length, M, cr, ci, dl);
        return arma::accu(cr % cr + ci % ci);
    };
    CHECK(power(beam, omni, q, 0.0) > 100.0 * power(beam, omni, -q, 0.0));
    CHECK(power(omni, beam, 0.0, -q) > 100.0 * power(omni, beam, 0.0, q));
}

TEST_CASE("Get Channels Planar - Position and orientation shapes")
{
    // Positions and orientations: [3, 1] and [1, 3] give the same result; other sizes throw before any output is written
    auto ant = quadriga_lib::generate_antenna<double>("omni", 1.0);
    const arma::vec aod = {0.0}, eod = {0.0}, aoa = {arma::datum::pi}, eoa = {0.0}, gain = {1.0}, length = {30.0};
    const arma::mat M = arma::mat({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}).t();
    const arma::mat col[4] = {arma::vec{0.0, 0.0, 1.0}, arma::vec{0.1, 0.2, 0.3}, arma::vec{20.0, 0.0, 1.0}, arma::vec{0.0, -0.1, 0.5}};
    const char *name[4] = {"tx_pos", "tx_orientation", "rx_pos", "rx_orientation"};
    auto run = [&](const arma::mat *p, arma::cube &cr, arma::cube &ci, arma::cube &dl)
    { quadriga_lib::get_channels_planar(ant, ant, p[0], p[1], p[2], p[3], aod, eod, aoa, eoa, gain, length, M, cr, ci, dl, 1e9); };

    arma::cube cr, ci, dl, cr_row, ci_row, dl_row;
    const arma::mat row[4] = {col[0].t(), col[1].t(), col[2].t(), col[3].t()};
    run(col, cr, ci, dl), run(row, cr_row, ci_row, dl_row);
    CHECK(arma::approx_equal(cr, cr_row, "absdiff", 0.0));
    CHECK(arma::approx_equal(ci, ci_row, "absdiff", 0.0));
    CHECK(arma::approx_equal(dl, dl_row, "absdiff", 0.0));

    const arma::mat bad[] = {arma::mat(2, 1, arma::fill::zeros), arma::mat(4, 1, arma::fill::zeros), arma::mat(),
                             arma::mat(3, 3, arma::fill::zeros)};
    for (int k = 0; k < 4; ++k)
        for (const arma::mat &b : bad)
        {
            arma::mat p[4] = {col[0], col[1], col[2], col[3]};
            p[k] = b;
            arma::cube c_re = cr, c_im = ci, d = dl;
            CHECK_THROWS_WITH(run(p, c_re, c_im, d), Catch::Matchers::ContainsSubstring("Input '" + std::string(name[k]) + "' must have 3 elements"));
            CHECK(arma::approx_equal(c_re, cr, "absdiff", 0.0));
            CHECK(arma::approx_equal(d, dl, "absdiff", 0.0));
        }
}

TEST_CASE("Get Channels Planar - AVX2 interpolation agrees with GENERIC, auto is GENERIC")
{
    auto tx = quadriga_lib::generate_antenna<double>("custom", 5.0, NAN, {90.0, 90.0});
    quadriga_lib::copy_element(tx, 0, {1});
    tx.element_pos(1, 1) = 0.1;
    const auto rx = quadriga_lib::generate_antenna<double>("xpol", 10.0);
    const arma::uword n = 19;
    const arma::vec aod = arma::randu<arma::vec>(n) * 6.0 - 3.0, eod = arma::randu<arma::vec>(n) * 2.0 - 1.0;
    const arma::vec aoa = arma::randu<arma::vec>(n) * 6.0 - 3.0, eoa = arma::randu<arma::vec>(n) * 2.0 - 1.0;
    const arma::vec gain(n, arma::fill::ones), len = arma::randu<arma::vec>(n) * 50.0 + 60.0;
    const arma::mat M = arma::randn<arma::mat>(8, n);

    arma::cube cr[3], ci[3], dl[3];
    arma::vec dop[3];
    const bool avx2 = quadriga_lib::quadriga_lib_has_AVX2();
    for (int k : {0, 1, 2})
    {
        if (k == 2 && !avx2)
        {
            CHECK_THROWS_AS(quadriga_lib::get_channels_planar<double>(tx, rx, {0, 0, 10}, {0, 0.1, 0.5}, {50, 0, 1.5}, {0, 0, 2.0},
                                                                      aod, eod, aoa, eoa, gain, len, M, cr[k], ci[k], dl[k],
                                                                      3e9, false, false, &dop[k], k),
                            std::invalid_argument);
            continue;
        }
        quadriga_lib::get_channels_planar<double>(tx, rx, {0, 0, 10}, {0, 0.1, 0.5}, {50, 0, 1.5}, {0, 0, 2.0},
                                                  aod, eod, aoa, eoa, gain, len, M, cr[k], ci[k], dl[k], 3e9, false, false, &dop[k], k);
    }
    CHECK((arma::approx_equal(cr[0], cr[1], "absdiff", 0.0) && arma::approx_equal(ci[0], ci[1], "absdiff", 0.0)));
    CHECK(arma::approx_equal(dl[0], dl[1], "absdiff", 0.0));
    if (avx2)
    {
        const double amax = arma::sqrt(arma::square(cr[1]) + arma::square(ci[1])).max();
        CHECK(arma::sqrt(arma::square(cr[2] - cr[1]) + arma::square(ci[2] - ci[1])).max() < 1e-5 * amax);
        CHECK(arma::approx_equal(dl[2], dl[1], "absdiff", 1e-12));
        CHECK(arma::approx_equal(dop[2], dop[1], "absdiff", 1e-5));
    }

    CHECK_THROWS_AS(quadriga_lib::get_channels_planar<double>(tx, rx, {0, 0, 10}, {0, 0, 0}, {50, 0, 1.5}, {0, 0, 0},
                                                              aod, eod, aoa, eoa, gain, len, M, cr[0], ci[0], dl[0],
                                                              3e9, false, false, nullptr, 3),
                    std::invalid_argument);
}

TEST_CASE("Get Channels Planar - vec form equals single calls")
{
    // 40 snapshots (enough for the parallel region), shared or per-snapshot positions, snapshots without paths
    auto tx = quadriga_lib::generate_antenna<double>("omni", 10.0);
    quadriga_lib::copy_element(tx, 0, {1});
    tx.element_pos = {{0.3, -0.2}, {0.1, 0.4}, {0.0, 0.2}};
    tx.e_phi_re.slice(1) += 0.5;
    auto rx = tx;
    rx.port_weights_re = arma::vec({1.0, 0.5}); // 2 elements, 1 port
    rx.port_weights_im = arma::vec({0.0, 0.2});

    const arma::uword n_snap = 40;
    arma::arma_rng::set_seed(11);
    std::vector<arma::vec> aod(n_snap), eod(n_snap), aoa(n_snap), eoa(n_snap), gain(n_snap), len(n_snap);
    std::vector<arma::mat> M(n_snap);
    for (arma::uword s = 0; s < n_snap; ++s)
    {
        const arma::uword n_path = (s % 7 == 3) ? 0 : 1 + s % 5;
        aod[s] = arma::randu<arma::vec>(n_path) * 6.0 - 3.0, eod[s] = arma::randu<arma::vec>(n_path) - 0.5;
        aoa[s] = arma::randu<arma::vec>(n_path) * 6.0 - 3.0, eoa[s] = arma::randu<arma::vec>(n_path) - 0.5;
        gain[s] = arma::randu<arma::vec>(n_path), len[s] = arma::randu<arma::vec>(n_path) * 50.0 + 60.0;
        M[s] = arma::randn<arma::mat>(8, n_path);
    }
    const arma::mat tx_pos = arma::randu<arma::mat>(3, n_snap) * 10.0, tx_ori = arma::randu<arma::mat>(3, n_snap);
    const arma::mat rx_pos = arma::randu<arma::mat>(3, n_snap) * 10.0 + 30.0, rx_ori = arma::randu<arma::mat>(3, 1);

    for (bool fake_los : {false, true})
        for (bool shared : {false, true})
        {
            const arma::mat Tp = shared ? arma::mat(tx_pos.col(0)) : tx_pos;
            std::vector<arma::cube> cr, ci, dl, cr2(2), ci2(2), dl2;
            std::vector<arma::vec> dop, dop2;
            std::vector<arma::cx_cube> cx;
            quadriga_lib::get_channels_planar_vec<double>(tx, rx, Tp, tx_ori, rx_pos, rx_ori, aod, eod, aoa, eoa, gain, len, M,
                                                          cr, ci, dl, 3.5e9, false, fake_los, &dop);
            quadriga_lib::get_channels_planar_vec<double>(tx, rx, Tp, tx_ori, rx_pos, rx_ori, aod, eod, aoa, eoa, gain, len, M,
                                                          cr2, ci2, dl2, 3.5e9, false, fake_los, &dop2, &cx);
            REQUIRE(cr.size() == n_snap);
            REQUIRE(dop.size() == n_snap);
            REQUIRE(cx.size() == n_snap);
            CHECK(cr2.empty());
            CHECK(ci2.empty());
            for (arma::uword s = 0; s < n_snap; ++s)
            {
                arma::cube r_cr, r_ci, r_dl;
                arma::vec r_dop;
                quadriga_lib::get_channels_planar<double>(tx, rx, Tp.col(shared ? 0 : s), tx_ori.col(s), rx_pos.col(s), rx_ori,
                                                          aod[s], eod[s], aoa[s], eoa[s], gain[s], len[s], M[s], r_cr, r_ci, r_dl,
                                                          3.5e9, false, fake_los, &r_dop);
                REQUIRE(arma::size(cr[s]) == arma::size(r_cr));
                REQUIRE(arma::size(cx[s]) == arma::size(r_cr));
                CHECK(arma::all(arma::vectorise(cr[s] == r_cr)));
                CHECK(arma::all(arma::vectorise(ci[s] == r_ci)));
                CHECK(arma::all(arma::vectorise(dl[s] == r_dl)));
                CHECK(arma::all(dop[s] == r_dop));
                CHECK(arma::all(arma::vectorise(arma::real(cx[s]) == r_cr)));
                CHECK(arma::all(arma::vectorise(arma::imag(cx[s]) == r_ci)));
                CHECK(arma::all(arma::vectorise(dl2[s] == r_dl)));
                CHECK(arma::all(dop2[s] == r_dop));
            }
        }

    // Float; output entries of the right size are written in place
    auto txf = quadriga_lib::generate_antenna<float>("omni", 10.0);
    std::vector<arma::fvec> a1 = {arma::fvec({0.0f})}, g1 = {arma::fvec({1.0f})}, l1 = {arma::fvec({20.0f})};
    std::vector<arma::fmat> M1 = {arma::fmat({1, 0, 0, 0, 0, 0, -1, 0}).t()};
    std::vector<arma::fcube> cr(1, arma::fcube(1, 1, 1)), ci, dl;
    std::vector<arma::cx_fcube> cx(1, arma::cx_fcube(1, 1, 1));
    const float *p = cr[0].memptr();
    const std::complex<float> *pc = cx[0].memptr();
    const arma::fmat z(3, 1, arma::fill::zeros), r = arma::fmat({10.0f, 0.0f, 0.0f}).t();
    quadriga_lib::get_channels_planar_vec<float>(txf, txf, z, z, r, z, a1, a1, a1, a1, g1, l1, M1, cr, ci, dl, 1e9f);
    CHECK(cr[0].memptr() == p);
    CHECK(std::abs(dl[0](0) * 299792458.0f - 10.0f) < 1e-4f); // relative to LOS
    quadriga_lib::get_channels_planar_vec<float>(txf, txf, z, z, r, z, a1, a1, a1, a1, g1, l1, M1, cr, ci, dl, 1e9f,
                                                 false, false, nullptr, &cx);
    CHECK(cx[0].memptr() == pc);
    CHECK(cr.empty());

    // No snapshots: empty outputs
    std::vector<arma::fvec> e_v;
    std::vector<arma::fmat> e_m;
    quadriga_lib::get_channels_planar_vec<float>(txf, txf, z, z, r, z, e_v, e_v, e_v, e_v, e_v, e_v, e_m, cr, ci, dl);
    CHECK(cr.empty());
    CHECK(dl.empty());
}

TEST_CASE("Get Channels Planar - vec form errors")
{
    using Catch::Matchers::ContainsSubstring;
    auto ant = quadriga_lib::generate_antenna<double>("omni", 10.0);
    std::vector<arma::vec> a(3, arma::vec({0.0})), g(3, arma::vec({1.0})), l(3, arma::vec({20.0}));
    std::vector<arma::mat> M(3, arma::mat(8, 1, arma::fill::ones));
    const arma::mat p1(3, 1, arma::fill::zeros), p2(3, 2, arma::fill::zeros), r1 = arma::mat({10.0, 0.0, 0.0}).t();
    std::vector<arma::cube> cr, ci, dl;

    std::vector<arma::vec> bad_len(2, arma::vec({0.0}));
    CHECK_THROWS_WITH(quadriga_lib::get_channels_planar_vec<double>(ant, ant, p1, p1, r1, p1, a, a, a, a, g, bad_len, M, cr, ci, dl),
                      ContainsSubstring("must have the same length"));
    CHECK_THROWS_WITH(quadriga_lib::get_channels_planar_vec<double>(ant, ant, p2, p1, r1, p1, a, a, a, a, g, l, M, cr, ci, dl),
                      ContainsSubstring("1 or 'n_snap' columns"));
    CHECK_THROWS_WITH(quadriga_lib::get_channels_planar_vec<double>(ant, ant, p1, p1, arma::mat(r1.t()), p1, a, a, a, a, g, l, M, cr, ci, dl),
                      ContainsSubstring("must have 3 rows"));

    // Invalid antenna and kernel throw without the snapshot prefix, also without snapshots
    auto bad = ant;
    bad.e_theta_re.reset();
    std::vector<arma::vec> e_v;
    std::vector<arma::mat> e_m;
    CHECK_THROWS_WITH(quadriga_lib::get_channels_planar_vec<double>(bad, ant, p1, p1, r1, p1, e_v, e_v, e_v, e_v, e_v, e_v, e_m, cr, ci, dl),
                      ContainsSubstring("Input 'tx_array' is not valid: ") && !ContainsSubstring("Snapshot"));
    CHECK_THROWS_AS(quadriga_lib::get_channels_planar_vec<double>(ant, ant, p1, p1, r1, p1, e_v, e_v, e_v, e_v, e_v, e_v, e_m, cr, ci, dl,
                                                                  0.0, false, false, nullptr, nullptr, 7),
                    std::invalid_argument);

    // Errors of the single call name the lowest failing snapshot, also from the parallel region (40 snapshots)
    a.resize(40, a[0]), g.resize(40, g[0]), l.resize(40, l[0]), M.resize(40, M[0]);
    M[23] = arma::mat(7, 1, arma::fill::ones), M[5] = M[23];
    CHECK_THROWS_WITH(quadriga_lib::get_channels_planar_vec<double>(ant, ant, p1, p1, r1, p1, a, a, a, a, g, l, M, cr, ci, dl),
                      ContainsSubstring("Snapshot 5: "));
    std::vector<arma::cx_cube> cx;
    CHECK_THROWS_WITH(quadriga_lib::get_channels_planar_vec<double>(ant, ant, p1, p1, r1, p1, a, a, a, a, g, l, M, cr, ci, dl,
                                                                    0.0, false, false, nullptr, &cx),
                      ContainsSubstring("Snapshot 5: "));
}
