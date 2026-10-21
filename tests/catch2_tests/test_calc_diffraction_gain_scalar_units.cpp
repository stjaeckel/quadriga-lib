// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <cmath>

TEST_CASE("calc_diffraction_gain - scalar mode takes the EM-equivalent frequency")
{
    // A direct path of half an acoustic wavelength at 1 kHz (speed of sound 342.76271 m/s) and a far-away triangle.
    // In scalar mode the documented frequency unit is f_eq = f_acoustic * c_light / c_sound, so that c_light / f_eq is
    // the acoustic wavelength: the propagation phase (apply_phase) is then -pi. Passing the acoustic frequency itself
    // gives a wavelength 874 636 times too long and a phase near 0.
    const double c_sound = 342.76271, c_light = 299792458.0;
    arma::mat orig = {{0.0, 0.0, 0.0}}, dest = {{0.5 * c_sound / 1000.0, 0.0, 0.0}};
    arma::mat mesh = {{100.0, 100.0, 100.0, 101.0, 100.0, 100.0, 100.0, 101.0, 100.0}};
    arma::uvec mtl = {0};
    std::unordered_map<std::string, std::vector<double>> prop;

    arma::vec gain;
    arma::mat xpr;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl, prop, 1000.0 * c_light / c_sound, 0, &gain, &xpr,
                                                nullptr, 0, nullptr, 1, 0, true, 0.0, false, true);
    REQUIRE(xpr.n_rows == 2);
    CHECK(std::abs(gain(0) - 1.0) < 1e-6);
    CHECK(std::abs(std::abs(std::atan2(xpr(1, 0), xpr(0, 0))) - arma::datum::pi) < 1e-6);

    quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl, prop, 1000.0, 0, &gain, &xpr,
                                                nullptr, 0, nullptr, 1, 0, true, 0.0, false, true);
    CHECK(std::abs(std::atan2(xpr(1, 0), xpr(0, 0))) < 1e-5);
}
