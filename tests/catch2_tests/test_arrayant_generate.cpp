// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_test_arrayant.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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
    float directivity = qd_directivity(ant, 0);
    CHECK(std::abs(directivity - 1.760964f) < 0.0001f);
    REQUIRE_THROWS_AS(qd_directivity(ant, 1), std::invalid_argument);

    auto ant2 = quadriga_lib::generate_arrayant_dipole<double>(5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = qd_directivity(ant2, 0);
    CHECK(std::abs(directivity2 - 1.760964) < 0.01f);
}

TEST_CASE("Generate Arrayant - Minimal test Half-wave dipole")
{
    auto ant = quadriga_lib::generate_arrayant_half_wave_dipole<float>();
    float directivity = qd_directivity(ant, 0);
    CHECK(std::abs(directivity - 2.15f) < 0.001f);

    auto ant2 = quadriga_lib::generate_arrayant_half_wave_dipole<double>(5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = qd_directivity(ant2, 0);
    CHECK(std::abs(directivity2 - 2.15) < 0.01f);
}

TEST_CASE("Generate Arrayant - Custom")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(10.0, 10.0);
    float directivity = qd_directivity(ant, 0);
    CHECK(std::abs(directivity - 25.627f) < 0.001f);

    auto ant2 = quadriga_lib::generate_arrayant_custom<double>(10.0, 10.0, 0.0, 5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = qd_directivity(ant2, 0);
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
    quadriga_lib::arrayant_copy_element(pattern, 0, {1});
    quadriga_lib::arrayant_copy_element(pattern, 0, {2});

    float gain = pattern.e_theta_re.at(90, 180, 0);
    gain = std::sqrt(2.0 * gain * gain);

    ant = quadriga_lib::generate_arrayant_3GPP<float>(2, 2, pattern.center_freq,
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

TEST_CASE("Generate Arrayant - Multi-Beam")
{
    arma::vec az, el, weight;
    double freq = 3.75e9;

    // Generate a pattern with 2 weighted beams
    az = {20.0, 0.0};
    el = {-7.0, 30.0};
    weight = {2.0, 1.0};
    auto ant = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, weight, freq, 1, 0.4, 120.0, 120.0, 0.0, 15.0);

    CHECK(ant.n_elements() == 36);
    CHECK(ant.n_elevation() == 13);
    CHECK(ant.n_azimuth() == 25);
    CHECK(ant.n_ports() == 1);

    arma::vec azg = arma::regspace<arma::vec>(-180.0, 1.0, 180.0) * 0.017453292519943;
    arma::vec elg = arma::regspace<arma::vec>(-90.0, 1.0, 90.0) * 0.017453292519943;

    auto comb = quadriga_lib::arrayant_combine_pattern(ant, &azg, &elg);

    CHECK(comb.n_elements() == 1);
    CHECK(comb.n_elevation() == 181);
    CHECK(comb.n_azimuth() == 361);

    double a = comb.e_theta_re(83, 199, 0), b = comb.e_theta_im(83, 199, 0);
    double gain = 10.0 * std::log10(a * a + b * b);

    CHECK(std::abs(gain - 35.6992) < 0.0001);
    CHECK(std::abs(a + 34.7096) < 0.01);
    CHECK(std::abs(b + 50.0993) < 0.01);

    // Generate the same pattern from 3 unweighted beams
    az = {20.0, 0.0, 20.0};
    el = {-7.0, 30.0, -7.0};
    auto ant2 = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, {}, freq, 1, 0.4, 120.0, 120.0, 0.0, 15.0);
    auto comb2 = quadriga_lib::arrayant_combine_pattern(ant2, &azg, &elg);

    CHECK(arma::approx_equal(ant.coupling_re, ant2.coupling_re, "absdiff", 1e-6));
    CHECK(arma::approx_equal(ant.coupling_im, ant2.coupling_im, "absdiff", 1e-6));

    CHECK(arma::approx_equal(comb.e_theta_re, comb2.e_theta_re, "absdiff", 1e-6));
    CHECK(arma::approx_equal(comb.e_theta_im, comb2.e_theta_im, "absdiff", 1e-6));

    // Test if generating the patterns at 1 de resolution generates similar results as at 10 degree
    auto comb3 = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, {}, freq, 1, 0.4, 120.0, 120.0, 0.0, 1.0, false, true);

    CHECK(comb3.n_elements() == 1);
    CHECK(comb3.n_elevation() == 181);
    CHECK(comb3.n_azimuth() == 361);

    a = comb3.e_theta_re(83, 199, 0), b = comb3.e_theta_im(83, 199, 0);
    gain = 10.0 * std::log10(a * a + b * b);

    CHECK(std::abs(gain - 35.6992) < 0.1);
    CHECK(std::abs(a + 34.7096) < 0.5);
    CHECK(std::abs(b + 50.0993) < 0.5);

    // Test generation of separate beams
    auto ant4 = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, {}, freq, 1, 0.4, 120.0, 120.0, 0.0, 15.0, true);

    CHECK(ant4.n_elements() == 36);
    CHECK(ant4.n_elevation() == 13);
    CHECK(ant4.n_azimuth() == 25);
    CHECK(ant4.n_ports() == 3);

    // H/V Polarization
    auto ant5 = quadriga_lib::generate_arrayant_multibeam<double>(6, 6, az, el, {}, freq, 2, 0.4, 120.0, 120.0, 0.0, 10.0);

    CHECK(ant5.n_elements() == 72);
    CHECK(ant5.n_elevation() == 19);
    CHECK(ant5.n_azimuth() == 37);
    CHECK(ant5.n_ports() == 2);

    arma::mat x = ant5.element_pos.submat(0, 0, 2, 35);
    arma::mat y = ant5.element_pos.submat(0, 36, 2, 71);
    CHECK(arma::approx_equal(x, y, "absdiff", 1e-14));

    x = ant5.coupling_re.submat(0, 0, 35, 0);
    y = ant5.coupling_re.submat(36, 1, 71, 1);
    CHECK(arma::approx_equal(x, y, "absdiff", 1e-14));

    x.zeros();
    y = ant5.coupling_re.submat(0, 1, 35, 1);
    CHECK(arma::approx_equal(x, y, "absdiff", 1e-14));

    y = ant5.coupling_re.submat(36, 0, 71, 0);
    CHECK(arma::approx_equal(x, y, "absdiff", 1e-14));
}


// A beam towards a pole, where the element pattern is exactly 0, used to leave the beam weights unset:
// pol = 1 returned one port per element, pol = 2 and 3 crashed (null pointer in memcpy).
TEST_CASE("Generate Arrayant - Multi-Beam towards the poles")
{
    for (double el : {90.0, -90.0})
        for (unsigned pol : {1u, 2u, 3u})
        {
            quadriga_lib::arrayant<double> a;
            REQUIRE_NOTHROW(a = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, arma::vec{0.0}, arma::vec{el},
                                                                                 arma::vec{1.0}, 3e9, pol));
            const arma::uword n_el = pol == 1 ? 4 : 8;
            CHECK(a.n_elements() == n_el);
            CHECK(a.n_ports() == (pol == 1 ? 1 : 2));
            CHECK(a.is_valid().empty());
            CHECK(arma::is_finite(a.coupling_re));
            CHECK(arma::is_finite(a.coupling_im));
            CHECK(arma::accu(arma::abs(a.coupling_re)) + arma::accu(arma::abs(a.coupling_im)) > 0.0);
        }

    // Separate beams, one of them towards the zenith
    auto b = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, arma::vec{0.0, 30.0}, arma::vec{90.0, 0.0}, {}, 3e9, 2,
                                                               0.5, 120.0, 120.0, 0.0, 10.0, true);
    CHECK(b.n_elements() == 8);
    CHECK(b.n_ports() == 4);
    CHECK(b.is_valid().empty());
}

// The ULA and 3GPP generators use only the radiation pattern of a custom pattern; its element positions and
// coupling are ignored (as documented).
TEST_CASE("Generate Arrayant - ULA and 3GPP ignore positions and coupling of a custom pattern")
{
    auto p = quadriga_lib::generate_arrayant_omni<double>(10.0);
    quadriga_lib::arrayant_copy_element(p, 0, {1});
    p.element_pos = {{0.3, 0.3}, {0.0, 0.0}, {0.7, -0.7}};
    p.coupling_re = arma::mat(2, 1, arma::fill::ones); // 2 elements -> 1 port
    p.coupling_im = arma::mat(2, 1, arma::fill::zeros);
    p.center_freq = 1e9;

    const double lambda = 299792458.0 / 3e9;
    auto u = quadriga_lib::generate_arrayant_ula<double>(2, 3e9, 0.5, &p);
    CHECK(u.n_elements() == 4);
    CHECK(u.n_ports() == 4);
    CHECK(arma::approx_equal(u.coupling_re, arma::mat(4, 4, arma::fill::eye), "absdiff", 0.0));
    CHECK(arma::all(arma::vectorise(u.coupling_im) == 0.0));
    CHECK(arma::all(u.element_pos.row(0) == 0.0));
    CHECK(arma::all(u.element_pos.row(2) == 0.0));
    arma::rowvec y = {-0.25 * lambda, -0.25 * lambda, 0.25 * lambda, 0.25 * lambda};
    CHECK(arma::approx_equal(u.element_pos.row(1), y, "absdiff", 1e-12));
    CHECK(u.center_freq == 3e9);

    auto g = quadriga_lib::generate_arrayant_3GPP<double>(1, 1, 3e9, 1, 0.0, 0.5, 1, 1, 0.5, 0.5, &p);
    CHECK(g.n_elements() == 2);
    CHECK(g.n_ports() == 2);
    CHECK(arma::approx_equal(g.coupling_re, arma::mat(2, 2, arma::fill::eye), "absdiff", 0.0));
    CHECK(arma::all(arma::vectorise(g.element_pos) == 0.0));

    // Vertical stacking: z positions come from the array geometry only
    auto g2 = quadriga_lib::generate_arrayant_3GPP<double>(2, 1, 3e9, 1, 0.0, 0.5, 1, 1, 0.5, 0.5, &p);
    CHECK(g2.n_elements() == 4);
    CHECK(g2.n_ports() == 4);
    CHECK(arma::all(g2.element_pos.row(0) == 0.0));
    arma::rowvec z = {-0.25 * lambda, -0.25 * lambda, 0.25 * lambda, 0.25 * lambda};
    CHECK(arma::approx_equal(g2.element_pos.row(2), z, "absdiff", 1e-12));
}

// The header defaults were M = N = 2, the documentation and the MATLAB / Python wrappers say 1
TEST_CASE("Generate Arrayant - Multi-Beam defaults are 1 x 1")
{
    auto a = quadriga_lib::generate_arrayant_multibeam<double>();
    CHECK(a.n_elements() == 1);
    CHECK(a.n_ports() == 1);

    auto b = quadriga_lib::generate_arrayant_multibeam<float>();
    CHECK(b.n_elements() == 1);
    CHECK(b.n_ports() == 1);

    // Same as passing M = N = 1 explicitly
    auto c = quadriga_lib::generate_arrayant_multibeam<double>(1, 1);
    CHECK(arma::approx_equal(a.e_theta_re, c.e_theta_re, "absdiff", 0.0));
    CHECK(arma::approx_equal(a.coupling_re, c.coupling_re, "absdiff", 0.0));
}

TEST_CASE("Arrayant generation - Multibeam beamwidth <= 0 selects the default 120 deg")
{
    arma::vec az = {20.0, 0.0}, el = {-7.0, 30.0};
    auto ref = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, az, el, {}, 3.75e9, 1, 0.5, 120.0, 120.0, 0.0, 10.0);
    auto b90 = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, az, el, {}, 3.75e9, 1, 0.5, 90.0, 90.0, 0.0, 10.0);
    for (double bw : {0.0, -5.0})
    {
        auto a = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, az, el, {}, 3.75e9, 1, 0.5, bw, bw, 0.0, 10.0);
        CHECK(arma::approx_equal(a.e_theta_re, ref.e_theta_re, "absdiff", 0.0));
        CHECK(arma::approx_equal(a.e_theta_im, ref.e_theta_im, "absdiff", 0.0));
    }
    auto a = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, az, el, {}, 3.75e9, 1, 0.5, 0.0, 90.0, 0.0, 10.0);
    auto b = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, az, el, {}, 3.75e9, 1, 0.5, 120.0, 90.0, 0.0, 10.0);
    CHECK(arma::approx_equal(a.e_theta_re, b.e_theta_re, "absdiff", 0.0)); // each angle separately
    CHECK(arma::abs(ref.e_theta_re - b90.e_theta_re).max() > 0.01);
}

TEST_CASE("Generate Arrayant - Invalid resolution")
{
    // NaN, infinite and too fine resolutions throw; res <= 0 selects the default (1 deg)
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    for (double res : {nan, inf, -inf, 0.09, 1.0e-300})
    {
        REQUIRE_THROWS_AS(quadriga_lib::generate_arrayant_omni<double>(res), std::invalid_argument);
        REQUIRE_THROWS_AS(quadriga_lib::generate_arrayant_xpol<double>(res), std::invalid_argument);
        REQUIRE_THROWS_AS(quadriga_lib::generate_arrayant_dipole<double>(res), std::invalid_argument);
        REQUIRE_THROWS_AS(quadriga_lib::generate_arrayant_half_wave_dipole<double>(res), std::invalid_argument);
        REQUIRE_THROWS_AS(quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, res), std::invalid_argument);
    }
    REQUIRE_THROWS_AS(quadriga_lib::generate_arrayant_omni<float>(std::numeric_limits<float>::denorm_min()), std::invalid_argument);
    REQUIRE_THROWS_AS(quadriga_lib::generate_arrayant_ula<double>(2, 3.0e9, 0.5, nullptr, nan), std::invalid_argument);
    REQUIRE_THROWS_AS(quadriga_lib::generate_arrayant_3GPP<double>(1, 1, 3.0e9, 1, 0.0, 0.5, 1, 1, 0.5, 0.5, nullptr, nan), std::invalid_argument);

    CHECK(quadriga_lib::generate_arrayant_omni<double>(0.0).n_azimuth() == 361);
    CHECK(quadriga_lib::generate_arrayant_omni<double>(-5.0).n_azimuth() == 361);
}

TEST_CASE("generate_arrayant_3GPP - a positive electrical tilt points the beam down")
{
    // A positive electrical tilt is a downtilt (TR 38.901): 8 vertical elements combined with tilt 10 degree (pol 4),
    // the main beam at elevation -10 degree
    const auto ant = quadriga_lib::generate_arrayant_3GPP<double>(8, 1, 3.5e9, 4, 10.0);
    const arma::uword i = arma::index_max(arma::vectorise(arma::square(ant.e_theta_re.slice(0)) + arma::square(ant.e_theta_im.slice(0))));
    CHECK(std::abs(ant.elevation_grid(i % ant.n_elevation()) * 180.0 / arma::datum::pi + 10.0) < 1.01);
}

TEST_CASE("generate_arrayant_3GPP - turned polarizations follow TR 38.901 Model-1, also on the rotation axis")
{
    // pol = 2 and 3 turn the vertical element by the slant angle zeta (90 degree; +45 and -45 degree). Off the rotation
    // axis, TR 38.901 eq. 7.3-3: F_theta = sqrt(A) cos(psi), F_phi = sqrt(A) sin(psi) with
    // cos(psi) = (cos(zeta) sin(theta) + sin(zeta) sin(phi) cos(theta)) / w, sin(psi) = sin(zeta) cos(phi) / w and
    // w = sqrt(1 - (cos(zeta) cos(theta) - sin(zeta) sin(phi) sin(theta))^2), theta the zenith angle; sqrt(A) is the
    // vertical element (pol = 1). On the axis (w = 0) the polarization is not rotated, and float equals double there,
    // as everywhere else.
    const double deg = arma::datum::pi / 180.0;
    const auto v = quadriga_lib::generate_arrayant_3GPP<double>(1, 1, 3.5e9, 1);
    for (unsigned pol : {2u, 3u})
    {
        const auto d = quadriga_lib::generate_arrayant_3GPP<double>(1, 1, 3.5e9, pol);
        const auto f = quadriga_lib::generate_arrayant_3GPP<float>(1, 1, 3.5e9f, pol);
        const double zeta[2] = {pol == 2 ? 0.0 : 45.0 * deg, pol == 2 ? 90.0 * deg : -45.0 * deg};
        double d_model = 0.0, d_float = 0.0;
        for (arma::uword s = 0; s < 2; ++s)
            for (arma::uword j = 0; j < d.n_azimuth(); ++j)
                for (arma::uword i = 0; i < d.n_elevation(); ++i)
                {
                    const double th = 0.5 * arma::datum::pi - d.elevation_grid(i), ph = d.azimuth_grid(j);
                    const double cz = std::cos(zeta[s]), sz = std::sin(zeta[s]);
                    const double x = cz * std::cos(th) - sz * std::sin(ph) * std::sin(th), w = std::sqrt(1.0 - x * x);
                    if (w > 1e-3)
                    {
                        const double A = v.e_theta_re(i, j, 0);
                        const double cpsi = (cz * std::sin(th) + sz * std::sin(ph) * std::cos(th)) / w, spsi = sz * std::cos(ph) / w;
                        d_model = std::max({d_model, std::abs(d.e_theta_re(i, j, s) - A * cpsi), std::abs(d.e_phi_re(i, j, s) - A * spsi)});
                    }
                    d_float = std::max({d_float, std::abs((double)f.e_theta_re(i, j, s) - d.e_theta_re(i, j, s)),
                                        std::abs((double)f.e_phi_re(i, j, s) - d.e_phi_re(i, j, s))});
                }
        INFO("pol = " << pol);
        CHECK(d_model < 1e-12);
        CHECK(d_float < 1e-5); // Single-precision field tolerance
    }

    // H element of pol = 2 on its axis (elevation 0, azimuth -90 and +90 degree): equal to the vertical element
    const auto d = quadriga_lib::generate_arrayant_3GPP<double>(1, 1, 3.5e9, 2);
    for (arma::uword j : {90ULL, 270ULL})
    {
        CHECK(std::abs(d.e_theta_re(90, j, 1) - v.e_theta_re(90, j, 0)) < 1e-12);
        CHECK(std::abs(d.e_phi_re(90, j, 1)) < 1e-12);
    }
}

TEST_CASE("generate_arrayant_multibeam - turned polarizations: float equals double, also on the rotation axis")
{
    // The beams of pol = 2 and 3 combine turned elements; their rotation axes lie on the default 1 degree grid
    for (unsigned pol : {2u, 3u})
    {
        const auto d = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, {0.0}, {0.0}, {1.0}, 3.5e9, pol);
        const auto f = quadriga_lib::generate_arrayant_multibeam<float>(2, 2, {0.0f}, {0.0f}, {1.0f}, 3.5e9f, pol);
        const double peak = std::max(arma::abs(d.e_theta_re).max(), arma::abs(d.e_phi_re).max());
        const double d_float = std::max({arma::abs(arma::conv_to<arma::cube>::from(f.e_theta_re) - d.e_theta_re).max(),
                                         arma::abs(arma::conv_to<arma::cube>::from(f.e_phi_re) - d.e_phi_re).max(),
                                         arma::abs(arma::conv_to<arma::cube>::from(f.e_theta_im) - d.e_theta_im).max(),
                                         arma::abs(arma::conv_to<arma::cube>::from(f.e_phi_im) - d.e_phi_im).max()});
        INFO("pol = " << pol);
        CHECK(d_float < 2e-5 * peak); // Float rounding of the polarization angle grows as 1 / (distance from the axis)
    }
}

TEST_CASE("generate_arrayant_half_wave_dipole - pattern equals cos(pi/2 sin(el)) / cos(el), also at the poles")
{
    const auto d = quadriga_lib::generate_arrayant_half_wave_dipole<double>();
    const auto f = quadriga_lib::generate_arrayant_half_wave_dipole<float>();
    const double scale = 1.280968208215292, pih = 0.5 * arma::datum::pi;
    double d_model = 0.0, d_float = 0.0;
    for (arma::uword i = 1; i + 1 < d.n_elevation(); ++i)
    {
        const double el = d.elevation_grid(i);
        d_model = std::max(d_model, std::abs(d.e_theta_re(i, 0, 0) - scale * std::cos(pih * std::sin(el)) / std::cos(el)));
    }
    for (arma::uword i = 0; i < d.n_elevation(); ++i)
        d_float = std::max(d_float, std::abs((double)f.e_theta_re(i, 0, 0) - d.e_theta_re(i, 0, 0)));
    CHECK(d_model < 1e-12);
    CHECK(d_float < 1e-6);
    for (arma::uword i : {0ULL, 180ULL})
    {
        CHECK(std::abs(d.e_theta_re(i, 7, 0)) < 1e-12); // Limit 0 at the poles
        CHECK(std::abs(f.e_theta_re(i, 7, 0)) < 1e-6f);
    }
}
