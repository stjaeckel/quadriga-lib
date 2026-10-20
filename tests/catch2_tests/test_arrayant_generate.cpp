// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <iostream>
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
    float directivity = ant.calc_directivity_dBi(0);
    CHECK(std::abs(directivity - 1.760964f) < 0.0001f);
    REQUIRE_THROWS_AS(ant.calc_directivity_dBi(1), std::invalid_argument);

    auto ant2 = quadriga_lib::generate_arrayant_dipole<double>(5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = ant2.calc_directivity_dBi(0);
    CHECK(std::abs(directivity2 - 1.760964) < 0.01f);
}

TEST_CASE("Generate Arrayant - Minimal test Half-wave dipole")
{
    auto ant = quadriga_lib::generate_arrayant_half_wave_dipole<float>();
    float directivity = ant.calc_directivity_dBi(0);
    CHECK(std::abs(directivity - 2.15f) < 0.001f);

    auto ant2 = quadriga_lib::generate_arrayant_half_wave_dipole<double>(5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = ant2.calc_directivity_dBi(0);
    CHECK(std::abs(directivity2 - 2.15) < 0.01f);
}

TEST_CASE("Generate Arrayant - Custom")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(10.0, 10.0);
    float directivity = ant.calc_directivity_dBi(0);
    CHECK(std::abs(directivity - 25.627f) < 0.001f);

    auto ant2 = quadriga_lib::generate_arrayant_custom<double>(10.0, 10.0, 0.0, 5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = ant2.calc_directivity_dBi(0);
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
    pattern.copy_element(0, 1);
    pattern.copy_element(0, 2);

    float gain = pattern.e_theta_re.at(90, 180, 0);
    gain = std::sqrt(2.0 * gain * gain);

    ant = quadriga_lib::generate_arrayant_3GPP<float>(2, 2, pattern.center_frequency,
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

    auto comb = ant.combine_pattern(&azg, &elg);

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
    auto comb2 = ant2.combine_pattern(&azg, &elg);

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


// B-011: a beam towards a pole, where the element pattern is exactly 0, used to leave the beam weights unset:
// pol = 1 returned one port per element, pol = 2 and 3 crashed (null pointer in memcpy).
TEST_CASE("Generate Arrayant - Multi-Beam towards the poles (B-011)")
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
            CHECK(a.is_valid(false).empty());
            CHECK(arma::is_finite(a.coupling_re));
            CHECK(arma::is_finite(a.coupling_im));
            CHECK(arma::accu(arma::abs(a.coupling_re)) + arma::accu(arma::abs(a.coupling_im)) > 0.0);
        }

    // Separate beams, one of them towards the zenith
    auto b = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, arma::vec{0.0, 30.0}, arma::vec{90.0, 0.0}, {}, 3e9, 2,
                                                               0.5, 120.0, 120.0, 0.0, 10.0, true);
    CHECK(b.n_elements() == 8);
    CHECK(b.n_ports() == 4);
    CHECK(b.is_valid(false).empty());
}

// B-044: the ULA and 3GPP generators use only the radiation pattern of a custom pattern; its element positions and
// coupling are ignored (as documented).
TEST_CASE("Generate Arrayant - ULA and 3GPP ignore positions and coupling of a custom pattern (B-044)")
{
    auto p = quadriga_lib::generate_arrayant_omni<double>(10.0);
    p.copy_element(0, 1);
    p.element_pos = {{0.3, 0.3}, {0.0, 0.0}, {0.7, -0.7}};
    p.coupling_re = arma::mat(2, 1, arma::fill::ones); // 2 elements -> 1 port
    p.coupling_im = arma::mat(2, 1, arma::fill::zeros);
    p.center_frequency = 1e9;

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
    CHECK(u.center_frequency == 3e9);

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

// B-047: the header defaults were M = N = 2, the documentation and the MATLAB / Python wrappers say 1
TEST_CASE("Generate Arrayant - Multi-Beam defaults are 1 x 1 (B-047)")
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
