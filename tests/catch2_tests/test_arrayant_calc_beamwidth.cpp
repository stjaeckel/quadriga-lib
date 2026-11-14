// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_test_arrayant.hpp"

#include <cmath>

TEST_CASE("Arrayant calc_beamwidth - Custom antenna, default 3dB threshold")
{
    // Custom antenna: az_3dB = 30°, el_3dB = 20°, no rear lobe, 1° sampling.
    // The pattern is constructed such that the measured FWHM equals the input.
    auto ant = quadriga_lib::generate_arrayant_custom<float>(30.0f, 20.0f);

    float bw_az = 0.0f, bw_el = 0.0f, az_pt = 999.0f, el_pt = 999.0f;
    qd_beamwidth(ant, 0, 3.0f, &bw_az, &bw_el, &az_pt, &el_pt);

    CHECK(std::abs(bw_az - 30.0f) < 0.1f);
    CHECK(std::abs(bw_el - 20.0f) < 0.1f);
    CHECK(std::abs(az_pt) < 0.05f);
    CHECK(std::abs(el_pt) < 0.05f);
}

TEST_CASE("Arrayant calc_beamwidth - Threshold dependence")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(30.0f, 30.0f);

    float bw1 = 0.0f, bw3 = 0.0f, bw6 = 0.0f;
    qd_beamwidth(ant, 0, 1.0f, &bw1, nullptr, nullptr, nullptr);
    qd_beamwidth(ant, 0, 3.0f, &bw3, nullptr, nullptr, nullptr);
    qd_beamwidth(ant, 0, 6.0f, &bw6, nullptr, nullptr, nullptr);

    CHECK(std::abs(bw3 - 30.0f) < 0.1f);
    CHECK(bw1 < bw3);          // narrower at 1dB
    CHECK(bw6 > bw3 + 5.0f);   // clearly wider at 6dB
}

TEST_CASE("Arrayant calc_beamwidth - Rotated antenna (heading)")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(20.0f, 20.0f);
    qd_rotate(ant, 0.0, 0.0, arma::datum::pi / 4.0); // heading +45° around z

    float bw_az = 0.0f, bw_el = 0.0f, az_pt = 0.0f, el_pt = 0.0f;
    qd_beamwidth(ant, 0, 3.0f, &bw_az, &bw_el, &az_pt, &el_pt);

    CHECK(std::abs(az_pt - 45.0f) < 0.1f);
    CHECK(std::abs(el_pt) < 0.5f);
    CHECK(std::abs(bw_az - 20.0f) < 0.1f);
    CHECK(std::abs(bw_el - 20.0f) < 0.1f);
}

TEST_CASE("Arrayant calc_beamwidth - Tilted antenna (elevation pointing)")
{
    // A tilt of -pi/6 (an uptilt, 3GPP TR 38.901) moves the main beam to elevation +30°
    auto ant = quadriga_lib::generate_arrayant_custom<float>(20.0f, 20.0f);
    qd_rotate(ant, 0.0, -arma::datum::pi / 6.0, 0.0);

    float az_pt = 999.0f, el_pt = 999.0f;
    qd_beamwidth(ant, 0, 3.0f, nullptr, nullptr, &az_pt, &el_pt);

    CHECK(std::abs(az_pt) < 0.1f);
    CHECK(std::abs(el_pt - 30.0f) < 0.1f);
}

TEST_CASE("Arrayant calc_beamwidth - Multi-element selection")
{
    // Three elements pointing at -60°, 0°, +60° respectively
    auto ant = quadriga_lib::generate_arrayant_custom<float>(20.0f, 20.0f);
    quadriga_lib::arrayant_copy_element(ant, 0, arma::uvec({1, 2}));
    qd_rotate(ant, 0.0, 0.0, arma::datum::pi / 3.0, 0, 1);
    qd_rotate(ant, 0.0, 0.0, -arma::datum::pi / 3.0, 0, 2);

    float az_pt = 0.0f;

    qd_beamwidth(ant, 0, 3.0f, nullptr, nullptr, &az_pt, nullptr);
    CHECK(std::abs(az_pt) < 0.5f);

    qd_beamwidth(ant, 1, 3.0f, nullptr, nullptr, &az_pt, nullptr);
    CHECK(std::abs(az_pt - 60.0f) < 0.1f);

    qd_beamwidth(ant, 2, 3.0f, nullptr, nullptr, &az_pt, nullptr);
    CHECK(std::abs(az_pt + 60.0f) < 0.1f);
}

TEST_CASE("Arrayant calc_beamwidth - Omni antenna")
{
    // Constant pattern: every direction is above any negative-dB threshold,
    // so the measured beamwidth equals the full grid extent.
    auto ant = quadriga_lib::generate_arrayant_omni<float>();

    float bw_az = 0.0f, bw_el = 0.0f;
    qd_beamwidth(ant, 0, 3.0f, &bw_az, &bw_el, nullptr, nullptr);

    CHECK(std::abs(bw_az - 360.0f) < 0.05f);
    CHECK(std::abs(bw_el - 180.0f) < 0.05f);
}

TEST_CASE("Arrayant calc_beamwidth - Selective output pointers")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(30.0f, 30.0f);

    // All-null call must be a no-op (no crash, early return).
    CHECK_NOTHROW(qd_beamwidth(ant, 0, 3.0f, nullptr, nullptr, nullptr, nullptr));

    // Request only azimuth beamwidth
    float bw_az = -1.0f;
    qd_beamwidth(ant, 0, 3.0f, &bw_az, nullptr, nullptr, nullptr);
    CHECK(std::abs(bw_az - 30.0f) < 0.1f);

    // Request only elevation pointing angle
    float el_pt = 999.0f;
    qd_beamwidth(ant, 0, 3.0f, nullptr, nullptr, nullptr, &el_pt);
    CHECK(std::abs(el_pt) < 0.1f);
}

TEST_CASE("Arrayant calc_beamwidth - Invalid element index")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>();
    float bw = 0.0f;
    REQUIRE_THROWS_AS(qd_beamwidth(ant, 5, 3.0f, &bw), std::invalid_argument);
}

// The beamwidth is measured on the main lobe, searched outwards from the maximum: circularly in azimuth and
// across the poles in elevation.
TEST_CASE("Arrayant calc_beamwidth - Main lobe across +-180 deg azimuth and at the poles")
{
    auto a = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, 1.0); // 90 deg FWHM in az and el
    double bw_az0 = 0.0, bw_el0 = 0.0, paz = 0.0, pel = 0.0;
    qd_beamwidth(a, 0, 3.0, &bw_az0, &bw_el0, &paz, &pel);
    CHECK(std::abs(bw_az0 - 90.0) < 1.0);
    CHECK(std::abs(bw_el0 - 90.0) < 1.0);

    for (double heading : {150.0, 180.0, -170.0})
    {
        auto b = a.copy();
        qd_rotate(b, 0.0, 0.0, heading * arma::datum::pi / 180.0, 3); // arrayant_calc_beamwidth returns degrees
        double bw_az = 0.0, bw_el = 0.0;
        qd_beamwidth(b, 0, 3.0, &bw_az, &bw_el, &paz, &pel);
        CHECK(std::abs(bw_az - bw_az0) < 1.0);
        CHECK(std::abs(bw_el - bw_el0) < 1.0);
        double d = std::fmod(paz - heading + 540.0, 360.0) - 180.0;
        CHECK(std::abs(d) < 0.5);
    }

    for (double tilt : {-90.0, 90.0, -80.0})
    {
        auto u = a.copy();
        qd_rotate(u, 0.0, tilt * arma::datum::pi / 180.0, 0.0, 3); // -90: zenith, +90: nadir
        double bw_el = 0.0;
        qd_beamwidth(u, 0, 3.0, nullptr, &bw_el, nullptr, &pel);
        CHECK(std::abs(bw_el - bw_el0) < 2.0);
        CHECK(std::abs(pel + tilt) < 0.5);
    }

    // Side lobes above the threshold are not part of the main lobe: two separate lobes at az = 0 and az = 90 deg
    auto c = quadriga_lib::generate_arrayant_custom<double>(20.0, 20.0, 0.0, 1.0);
    auto c2 = c.copy();
    qd_rotate(c2, 0.0, 0.0, arma::datum::pi / 2.0, 3);
    c.e_theta_re.slice(0) += 0.9 * c2.e_theta_re.slice(0);
    double bw_az = 0.0;
    qd_beamwidth(c, 0, 3.0, &bw_az, nullptr, &paz, nullptr);
    CHECK(std::abs(paz) < 0.5);
    CHECK(bw_az < 25.0);
}

// Vector forms: element selection, defaults and shapes; _multi: column i_freq equals the single-frequency
// function on entry i_freq. The forms are compared with a tolerance far below the resolution of the method (about
// 1/100 of the grid step), so that a different evaluation order of the same computation passes.
TEST_CASE("Arrayant calc_beamwidth and calc_directivity - Element lists and multi-frequency vectors")
{
    const double tol = 1.0e-10; // degree or dB
    auto a = quadriga_lib::generate_arrayant_custom<double>(40.0, 30.0, 0.0, 10.0);
    quadriga_lib::arrayant_copy_element(a, 0, arma::uvec{1, 2});
    qd_rotate(a, 0.0, 0.0, arma::datum::pi / 3.0, 0, 1);
    qd_rotate(a, 0.0, -arma::datum::pi / 9.0, 0.0, 0, 2);

    // nullptr and empty select all elements; duplicates and any order are allowed
    arma::vec bw_az, bw_el, p_az, p_el, d, d_all, d_empty;
    quadriga_lib::arrayant_calc_beamwidth(a, nullptr, 3.0, &bw_az, &bw_el, &p_az, &p_el);
    REQUIRE(bw_az.n_elem == 3);
    REQUIRE(p_el.n_elem == 3);
    for (arma::uword i = 0; i < 3; ++i)
    {
        double s_az = 0.0, s_el = 0.0, s_paz = 0.0, s_pel = 0.0;
        qd_beamwidth(a, i, 3.0, &s_az, &s_el, &s_paz, &s_pel);
        CHECK(std::abs(bw_az[i] - s_az) < tol);
        CHECK(std::abs(bw_el[i] - s_el) < tol);
        CHECK(std::abs(p_az[i] - s_paz) < tol);
        CHECK(std::abs(p_el[i] - s_pel) < tol);
    }
    CHECK(std::abs(p_az[1] - 60.0) < 0.1);
    CHECK(std::abs(p_el[2] - 20.0) < 0.1);

    const arma::uvec empty, sel = {2, 0, 2};
    quadriga_lib::arrayant_calc_directivity(a, d_all);
    quadriga_lib::arrayant_calc_directivity(a, d_empty, &empty);
    quadriga_lib::arrayant_calc_directivity(a, d, &sel);
    REQUIRE(d_all.n_elem == 3);
    CHECK(arma::approx_equal(d_all, d_empty, "absdiff", tol));
    REQUIRE(d.n_elem == 3);
    CHECK(std::abs(d[0] - d_all[2]) < tol);
    CHECK(std::abs(d[1] - d_all[0]) < tol);
    CHECK(std::abs(d[2] - d_all[2]) < tol);

    arma::vec bw_sel;
    quadriga_lib::arrayant_calc_beamwidth(a, &sel, 3.0, &bw_sel);
    REQUIRE(bw_sel.n_elem == 3);
    CHECK(std::abs(bw_sel[0] - bw_az[2]) < tol);

    // Multi-frequency: entries with different patterns, same grids and element count
    auto b = a;
    qd_rotate(b, 0.0, 0.0, -arma::datum::pi / 4.0);
    const std::vector<quadriga_lib::arrayant<double>> v = {a, b};
    arma::mat m_az, m_el, m_paz, m_pel, m_d;
    quadriga_lib::arrayant_calc_beamwidth_multi(v, &sel, 6.0, &m_az, &m_el, &m_paz, &m_pel);
    quadriga_lib::arrayant_calc_directivity_multi(v, m_d, &sel);
    REQUIRE(m_az.n_rows == 3);
    REQUIRE(m_az.n_cols == 2);
    REQUIRE(m_pel.n_rows == 3);
    REQUIRE(m_d.n_cols == 2);
    for (arma::uword f = 0; f < 2; ++f)
    {
        arma::vec s_az, s_el, s_paz, s_pel, s_d;
        quadriga_lib::arrayant_calc_beamwidth(v[f], &sel, 6.0, &s_az, &s_el, &s_paz, &s_pel);
        quadriga_lib::arrayant_calc_directivity(v[f], s_d, &sel);
        CHECK(arma::approx_equal(m_az.col(f), s_az, "absdiff", tol));
        CHECK(arma::approx_equal(m_el.col(f), s_el, "absdiff", tol));
        CHECK(arma::approx_equal(m_paz.col(f), s_paz, "absdiff", tol));
        CHECK(arma::approx_equal(m_pel.col(f), s_pel, "absdiff", tol));
        CHECK(arma::approx_equal(m_d.col(f), s_d, "absdiff", tol));
    }
    CHECK(std::abs(m_paz(1, 0)) < 0.1);        // element 0 of a points to az 0 (selection {2, 0, 2})
    CHECK(std::abs(m_paz(1, 1) + 45.0) < 5.0); // b: heading -45 deg, resampled on the 10 deg grid

    // A one-entry vector equals the single-frequency function
    arma::mat o_az;
    quadriga_lib::arrayant_calc_beamwidth_multi(std::vector<quadriga_lib::arrayant<double>>{a}, nullptr, 3.0, &o_az);
    REQUIRE(o_az.n_cols == 1);
    CHECK(arma::approx_equal(o_az.col(0), bw_az, "absdiff", tol));

    // float: _multi equals the single-frequency function
    auto af = quadriga_lib::generate_arrayant_custom<float>(40.0f, 30.0f, 0.0f, 10.0f);
    arma::fvec f_az, f_d;
    arma::fmat fm_az, fm_d;
    quadriga_lib::arrayant_calc_beamwidth(af, nullptr, 3.0f, &f_az);
    quadriga_lib::arrayant_calc_directivity(af, f_d);
    quadriga_lib::arrayant_calc_beamwidth_multi(std::vector<quadriga_lib::arrayant<float>>{af, af}, nullptr, 3.0f, &fm_az);
    quadriga_lib::arrayant_calc_directivity_multi(std::vector<quadriga_lib::arrayant<float>>{af, af}, fm_d);
    REQUIRE(fm_az.n_cols == 2);
    CHECK(arma::approx_equal(fm_az.col(1), f_az, "absdiff", 1.0e-4f));
    CHECK(arma::approx_equal(fm_d.col(0), f_d, "absdiff", 1.0e-4f));
    CHECK(std::abs(f_az[0] - 40.0f) < 1.0f);
}

TEST_CASE("Arrayant calc_beamwidth and calc_directivity - Input validation")
{
    auto a = quadriga_lib::generate_arrayant_custom<double>(40.0, 30.0, 0.0, 10.0);
    quadriga_lib::arrayant_copy_element(a, 0, arma::uvec{1, 2});
    const std::vector<quadriga_lib::arrayant<double>> v = {a, a};
    arma::vec bw, d;
    arma::mat m;

    SECTION("ant / arrayant_vec")
    {
        auto broken = a;
        broken.azimuth_grid.resize(2);
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_beamwidth(broken, nullptr, 3.0, &bw), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_directivity(broken, d), std::invalid_argument);

        const std::vector<quadriga_lib::arrayant<double>> v_empty, v_bad = {a, quadriga_lib::generate_arrayant_omni<double>()};
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_beamwidth_multi(v_empty, nullptr, 3.0, &m), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_directivity_multi(v_empty, m), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_beamwidth_multi(v_bad, nullptr, 3.0, &m), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_directivity_multi(v_bad, m), std::invalid_argument);
    }

    SECTION("i_element")
    {
        const arma::uvec bad = {0, 3};
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_beamwidth(a, &bad, 3.0, &bw), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_directivity(a, d, &bad), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_beamwidth_multi(v, &bad, 3.0, &m), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_calc_directivity_multi(v, m, &bad), std::invalid_argument);
    }

    SECTION("threshold_dB")
    {
        for (double t : {arma::datum::nan, arma::datum::inf, -arma::datum::inf, 0.0, -3.0})
        {
            CHECK_THROWS_AS(quadriga_lib::arrayant_calc_beamwidth(a, nullptr, t, &bw), std::invalid_argument);
            CHECK_THROWS_AS(quadriga_lib::arrayant_calc_beamwidth_multi(v, nullptr, t, &m), std::invalid_argument);
        }
    }
}
