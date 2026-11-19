// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests that the scalar (acoustic) mode of calc_diffraction_gain, medium_gain, interface_gain, refractive_index and
// generate_diffraction_paths takes center_freq as the acoustic frequency. Oracle: the EM-mode result at the frequency
// of the same wavelength, f · c_light / 343 m/s. The diffraction values themselves are tested in
// test_calc_diffraction_gain.cpp and the material functions in their own suites.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <cmath>

TEST_CASE("calc_diffraction_gain - scalar mode takes the acoustic frequency")
{
    // A direct path of half an acoustic wavelength at 1 kHz (speed of sound 343 m/s) and a far-away triangle. In scalar
    // mode center_freq is the acoustic frequency, so the propagation phase (apply_phase) is -pi; EM mode at the
    // frequency of the same wavelength, 1 kHz * c_light / c_sound, gives the same phase.
    const double c_sound = 343.0, c_light = 299792458.0;
    arma::mat orig = {{0.0, 0.0, 0.0}}, dest = {{0.5 * c_sound / 1000.0, 0.0, 0.0}};
    arma::mat mesh = {{100.0, 100.0, 100.0, 101.0, 100.0, 100.0, 100.0, 101.0, 100.0}};
    arma::uvec mtl = {0};
    std::unordered_map<std::string, std::vector<double>> prop;

    arma::vec gain;
    arma::mat xpr;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl, prop, 1000.0, 0, &gain, &xpr,
                                                nullptr, 0, {}, 1, 0, true, 0.0, false, true);
    REQUIRE(xpr.n_rows == 2);
    CHECK(std::abs(gain(0) - 1.0) < 1e-6);
    CHECK(std::abs(std::abs(std::atan2(xpr(1, 0), xpr(0, 0))) - arma::datum::pi) < 1e-6);

    quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl, prop, 1000.0 * c_light / c_sound, 0, &gain, &xpr,
                                                nullptr, 0, {}, 1, 0, false, 0.0, false, true);
    REQUIRE(xpr.n_rows == 8);
    CHECK(std::abs(std::abs(std::atan2(xpr(1, 0), xpr(0, 0))) - arma::datum::pi) < 1e-6);
}

// In scalar mode the material functions and generate_diffraction_paths take the acoustic frequency and evaluate the
// material table at the EM-equivalent frequency of the same wavelength, 1 kHz * c_light / 343. A material whose every
// column depends on frequency gives the EM results at that frequency.
static const double su_f_ac = 1000.0, su_f_eq = su_f_ac * 299792458.0 / 343.0;
static const std::unordered_map<std::string, std::vector<double>> su_prop = {
    {"a", {4.0}}, {"b", {0.2}}, {"c", {0.05}}, {"d", {0.8}}, {"att", {3.0}}, {"attB", {0.5}}};

TEST_CASE("medium_gain - scalar mode takes the acoustic frequency")
{
    const double g = quadriga_lib::medium_gain<double>(su_prop, 1, 0.3, su_f_ac, true);
    CHECK(g < 0.99);
    CHECK(std::abs(g / quadriga_lib::medium_gain<double>(su_prop, 1, 0.3, su_f_eq) - 1.0) < 1e-12);
}

TEST_CASE("interface_gain - scalar mode takes the acoustic frequency")
{
    const double g = quadriga_lib::interface_gain<double>(su_prop, 1, su_f_ac, true);
    CHECK(g < 0.99);
    CHECK(std::abs(g / quadriga_lib::interface_gain<double>(su_prop, 1, su_f_eq) - 1.0) < 1e-12);
}

TEST_CASE("refractive_index - scalar mode takes the acoustic frequency")
{
    const double n = quadriga_lib::refractive_index<double>(su_prop, 1, su_f_ac, true);
    CHECK(std::abs(n / quadriga_lib::refractive_index<double>(su_prop, 1, su_f_eq) - 1.0) < 1e-12);
}

TEST_CASE("generate_diffraction_paths - scalar mode takes the acoustic frequency")
{
    arma::mat orig = {{0.0, 0.0, 0.0}}, dest = {{10.0, 0.0, 0.0}};
    arma::cube xs, ys, zs, ws, xe, ye, ze, we;
    quadriga_lib::generate_diffraction_paths<double>(orig, dest, su_f_ac, 2, xs, ys, zs, ws, true);
    quadriga_lib::generate_diffraction_paths<double>(orig, dest, su_f_eq, 2, xe, ye, ze, we);
    CHECK(arma::approx_equal(ys, ye, "absdiff", 1e-12));
    CHECK(arma::approx_equal(zs, ze, "absdiff", 1e-12));
    CHECK(arma::approx_equal(ws, we, "absdiff", 1e-12));
    CHECK(arma::abs(ys).max() > 0.1); // the ellipsoid of the acoustic wavelength (0.343 m), not of c_light / 1 kHz (300 km)
}
