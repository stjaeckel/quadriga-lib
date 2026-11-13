// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_test_arrayant.hpp"

#include <iostream>
#include <string>

TEST_CASE("Arrayant combine pattern - Minimal test")
{
    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    quadriga_lib::arrayant_copy_element(ant, 0, {2});
    ant.coupling_re.ones(3, 1);
    ant.coupling_im.reset();

    ant = quadriga_lib::arrayant_combine_pattern(ant);

    arma::fcube T(181, 361, 1, arma::fill::value(2.0f));
    CHECK(arma::approx_equal(ant.e_theta_re, T, "absdiff", 1e-6));

    T.zeros();
    CHECK(arma::approx_equal(ant.e_theta_im, T, "absdiff", 1e-6));
    CHECK(arma::approx_equal(ant.e_phi_re, T, "absdiff", 1e-6));
    CHECK(arma::approx_equal(ant.e_phi_im, T, "absdiff", 1e-6));

    arma::fmat Q(3, 1);
    CHECK(arma::approx_equal(ant.element_pos, Q, "absdiff", 1e-6));

    Q.ones(1, 1);
    CHECK(arma::approx_equal(ant.coupling_re, Q, "absdiff", 1e-6));

    Q.zeros(1, 1);
    CHECK(arma::approx_equal(ant.coupling_im, Q, "absdiff", 1e-6));

    // Multiple copies
    quadriga_lib::arrayant_copy_element(ant, 0, {1,2});
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), ant.e_theta_re.slice(1), "absdiff", 1e-6));
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), ant.e_theta_re.slice(2), "absdiff", 1e-6));
}

TEST_CASE("Arrayant rotation - Minimal test")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(5.0, 5.0);

    arma::uword i = ant.e_theta_re.index_max();
    arma::uvec s = arma::ind2sub(arma::size(ant.e_theta_re), i);
    CHECK(s(0) == 90);
    CHECK(s(1) == 180);

    qd_rotate(ant, 0.0, 0.0, arma::datum::pi / 2.0);

    i = ant.e_theta_re.index_max();
    s = arma::ind2sub(arma::size(ant.e_theta_re), i);
    CHECK(s(0) == 90);
    CHECK(s(1) == 270);

    ant = quadriga_lib::generate_arrayant_custom<float>(5.0, 5.0);
    qd_rotate(ant, 0.0, -arma::datum::pi / 4.0);

    i = ant.e_theta_re.index_max();
    s = arma::ind2sub(arma::size(ant.e_theta_re), i);
    CHECK(s(0) == 135);
    CHECK(s(1) == 180);
}

TEST_CASE("Arrayant rotation - Full sphere, inplace")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(5.0, 5.0);
    arma::fmat T = {1.0, 2.0, 3.0};
    ant.element_pos = T.t();
    quadriga_lib::arrayant_copy_element(ant, 0, {1});
    ant.e_theta_re.slice(1) = ant.e_theta_re.slice(0) * 2.0f;
    ant.element_pos.col(1) = 10.0 * T.t();
    ant.read_only = true;

    arma::uword n_elem = ant.e_theta_re.n_elem;
    float val = ant.e_theta_re.at(90, 180, 0);

    // Check error message
    REQUIRE_THROWS_AS(qd_rotate(ant, 0.0, 0.0, arma::datum::pi / 2.0), std::invalid_argument);
    ant.read_only = false;

    // First element
    T = {{-2.0, 10.0}, {1, 20.0}, {3.0, 30.0}};
    qd_rotate(ant, 0.0, 0.0, arma::datum::pi / 2.0, 0, 0);
    CHECK(ant.e_theta_re.n_elem == n_elem);
    CHECK(ant.e_theta_im.n_elem == n_elem);
    CHECK(ant.e_phi_re.n_elem == n_elem);
    CHECK(ant.e_phi_im.n_elem == n_elem);
    CHECK(std::abs(ant.e_theta_re.at(90, 270, 0) - val) < 0.001);
    CHECK(std::abs(ant.e_theta_re.at(90, 180, 1) - 2.0 * val) < 0.001);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));

    // Second element
    T = {{-2.0, 20.0}, {1.0, -10.0}, {3.0, 30.0}};
    qd_rotate(ant, 0.0, 0.0, -arma::datum::pi / 2.0, 0, 1);
    CHECK(std::abs(ant.e_theta_re.at(90, 270, 0) - val) < 0.001);
    CHECK(std::abs(ant.e_theta_re.at(90, 90, 1) - 2.0 * val) < 0.001);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));

    // Both elements
    T = {{2.0, -20.0}, {-1.0, 10.0}, {3.0, 30.0}};
    qd_rotate(ant, 0.0, 0.0, arma::datum::pi, 0);
    CHECK(std::abs(ant.e_theta_re.at(90, 90, 0) - val) < 0.001);
    CHECK(std::abs(ant.e_theta_re.at(90, 270, 1) - 2.0 * val) < 0.001);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));

    // Polarization
    T = {{1.0, -20.0}, {2.0, 10.0}, {3.0, 30.0}};
    qd_rotate(ant, 0.0, 0.0, arma::datum::pi / 2.0, 0, 0);
    qd_rotate(ant, arma::datum::pi / 2.0, 0.0, 0.0, 2, 0);
    CHECK(std::abs(ant.e_phi_re.at(90, 180, 0) - val) < 0.001);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));
}

TEST_CASE("Arrayant rotation - Partial sphere, inplace")
{
    arma::fmat T = {1.0, 2.0, 3.0};

    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    ant.e_theta_re.zeros();
    ant.e_theta_re.at(90, 180, 0) = 1.0f;
    quadriga_lib::arrayant_remove_zeros(ant);
    ant.element_pos = T.t();
    quadriga_lib::arrayant_copy_element(ant, 0, {1});
    ant.e_theta_re.slice(1) = ant.e_theta_re.slice(0) * 2.0f;
    ant.element_pos.col(1) = 10.0 * T.t();
    // Check error message if trying to rotate single element
    REQUIRE_THROWS_AS(qd_rotate(ant, 0.0, 0.0, arma::datum::pi / 2.0, 0, 0), std::invalid_argument);

    // Both elements
    qd_rotate(ant, 0.0, 0.0, arma::datum::pi);

    T = {{-1.0, -10.0}, {-2.0, -20.0}, {3.0, 30.0}};
    arma::uword end = ant.n_azimuth() - 1;
    CHECK(ant.n_elevation() == 3);
    CHECK(ant.n_elements() == 2);
    CHECK(arma::approx_equal(ant.elevation_grid, ant.elevation_grid, "absdiff", 1e-5f));
    CHECK(std::abs(ant.azimuth_grid.at(0) + arma::fdatum::pi) < 1e-5f);
    CHECK(std::abs(ant.azimuth_grid.at(end) - arma::fdatum::pi) < 1e-5f);
    CHECK(std::abs(ant.e_theta_re.at(1, 0, 0) - 1.0f) < 1e-4f);
    CHECK(std::abs(ant.e_theta_re.at(1, end, 0) - 1.0f) < 1e-4f);
    CHECK(std::abs(ant.e_theta_re.at(1, 0, 1) - 2.0f) < 1e-4f);
    CHECK(std::abs(ant.e_theta_re.at(1, end, 1) - 2.0f) < 1e-4f);
    CHECK(arma::approx_equal(ant.element_pos, T, "absdiff", 1e-5));
}

// The position phase used the speed of light; acoustic arrays were combined without it
TEST_CASE("Arrayant combine pattern - Propagation speed")
{
    // Two omni elements 0.2 m apart on the y-axis at 1 kHz, summed into one port
    auto ant = quadriga_lib::generate_arrayant_omni<double>(10.0);
    quadriga_lib::arrayant_copy_element(ant, 0, {1});
    ant.element_pos = {{0.0, 0.0}, {-0.1, 0.1}, {0.0, 0.0}};
    ant.coupling_re = arma::mat(2, 1, arma::fill::ones);
    ant.coupling_im = arma::mat(2, 1, arma::fill::zeros);
    ant.center_frequency = 1000.0;

    const arma::uword ie = 9, ia0 = 18, ia90 = 27; // el = 0; az = 0 and 90 deg on the 10 deg grid
    auto amp = [](const quadriga_lib::arrayant<double> &a, arma::uword i, arma::uword j)
    { return std::hypot(a.e_theta_re(i, j, 0), a.e_theta_im(i, j, 0)); };

    // Default: speed of light, the 0.2 m spacing is negligible at 1 kHz
    auto em = quadriga_lib::arrayant_combine_pattern(ant);
    CHECK(std::abs(amp(em, ie, ia0) - 2.0) < 1e-6);
    CHECK(std::abs(amp(em, ie, ia90) - 2.0) < 1e-6);

    // Speed of sound
    const double c = 344.0, expected = 2.0 * std::abs(std::cos(arma::datum::pi * 1000.0 * 0.2 / c));
    auto ac = quadriga_lib::arrayant_combine_pattern<double>(ant, nullptr, nullptr, c);
    CHECK(std::abs(amp(ac, ie, ia0) - 2.0) < 1e-6);
    CHECK(std::abs(amp(ac, ie, ia90) - expected) < 1e-6);
    CHECK(std::abs(expected - 0.506) < 1e-3);

    // Explicit speed of light equals the default
    auto em2 = quadriga_lib::arrayant_combine_pattern<double>(ant, nullptr, nullptr, 299792458.0);
    CHECK(arma::approx_equal(em.e_theta_re, em2.e_theta_re, "absdiff", 0.0));
    CHECK(arma::approx_equal(em.e_theta_im, em2.e_theta_im, "absdiff", 0.0));

    // Float
    auto antf = quadriga_lib::generate_arrayant_omni<float>(10.0f);
    quadriga_lib::arrayant_copy_element(antf, 0, {1});
    antf.element_pos = {{0.0f, 0.0f}, {-0.1f, 0.1f}, {0.0f, 0.0f}};
    antf.coupling_re = arma::fmat(2, 1, arma::fill::ones);
    antf.coupling_im = arma::fmat(2, 1, arma::fill::zeros);
    antf.center_frequency = 1000.0f;
    auto acf = quadriga_lib::arrayant_combine_pattern<float>(antf, nullptr, nullptr, 344.0f);
    CHECK(std::abs(std::hypot(acf.e_theta_re(ie, ia90, 0), acf.e_theta_im(ie, ia90, 0)) - expected) < 1e-4);

    // Invalid speeds
    for (double v : {0.0, -344.0, arma::datum::nan, arma::datum::inf})
        CHECK_THROWS_AS(quadriga_lib::arrayant_combine_pattern<double>(ant, nullptr, nullptr, v), std::invalid_argument);
}
