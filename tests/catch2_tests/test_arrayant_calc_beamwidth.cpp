// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <cmath>

TEST_CASE("Arrayant calc_beamwidth_deg - Custom antenna, default 3dB threshold")
{
    // Custom antenna: az_3dB = 30°, el_3dB = 20°, no rear lobe, 1° sampling.
    // The pattern is constructed such that the measured FWHM equals the input.
    auto ant = quadriga_lib::generate_arrayant_custom<float>(30.0f, 20.0f);

    float bw_az = 0.0f, bw_el = 0.0f, az_pt = 999.0f, el_pt = 999.0f;
    ant.calc_beamwidth_deg(0, 3.0f, &bw_az, &bw_el, &az_pt, &el_pt);

    CHECK(std::abs(bw_az - 30.0f) < 0.1f);
    CHECK(std::abs(bw_el - 20.0f) < 0.1f);
    CHECK(std::abs(az_pt) < 0.05f);
    CHECK(std::abs(el_pt) < 0.05f);
}

TEST_CASE("Arrayant calc_beamwidth_deg - Threshold dependence")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(30.0f, 30.0f);

    float bw1 = 0.0f, bw3 = 0.0f, bw6 = 0.0f;
    ant.calc_beamwidth_deg(0, 1.0f, &bw1, nullptr, nullptr, nullptr);
    ant.calc_beamwidth_deg(0, 3.0f, &bw3, nullptr, nullptr, nullptr);
    ant.calc_beamwidth_deg(0, 6.0f, &bw6, nullptr, nullptr, nullptr);

    CHECK(std::abs(bw3 - 30.0f) < 0.1f);
    CHECK(bw1 < bw3);          // narrower at 1dB
    CHECK(bw6 > bw3 + 5.0f);   // clearly wider at 6dB
}

TEST_CASE("Arrayant calc_beamwidth_deg - Rotated antenna (heading)")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(20.0f, 20.0f);
    ant.rotate_pattern(0.0f, 0.0f, 45.0f); // heading +45° around z

    float bw_az = 0.0f, bw_el = 0.0f, az_pt = 0.0f, el_pt = 0.0f;
    ant.calc_beamwidth_deg(0, 3.0f, &bw_az, &bw_el, &az_pt, &el_pt);

    CHECK(std::abs(az_pt - 45.0f) < 0.1f);
    CHECK(std::abs(el_pt) < 0.5f);
    CHECK(std::abs(bw_az - 20.0f) < 0.1f);
    CHECK(std::abs(bw_el - 20.0f) < 0.1f);
}

TEST_CASE("Arrayant calc_beamwidth_deg - Tilted antenna (elevation pointing)")
{
    // rotate_pattern(0, -30, 0) tilts the main beam to elevation +30°
    // (matches the convention used in test_arrayant_combine_pattern.cpp:
    //  rotate_pattern(0, -45) -> max moves to elevation index 135 = +45°)
    auto ant = quadriga_lib::generate_arrayant_custom<float>(20.0f, 20.0f);
    ant.rotate_pattern(0.0f, -30.0f, 0.0f);

    float az_pt = 999.0f, el_pt = 999.0f;
    ant.calc_beamwidth_deg(0, 3.0f, nullptr, nullptr, &az_pt, &el_pt);

    CHECK(std::abs(az_pt) < 0.1f);
    CHECK(std::abs(el_pt - 30.0f) < 0.1f);
}

TEST_CASE("Arrayant calc_beamwidth_deg - Multi-element selection")
{
    // Three elements pointing at -60°, 0°, +60° respectively
    auto ant = quadriga_lib::generate_arrayant_custom<float>(20.0f, 20.0f);
    ant.copy_element(0, arma::uvec({1, 2}));
    ant.rotate_pattern(0.0f, 0.0f,  60.0f, 0, 1);
    ant.rotate_pattern(0.0f, 0.0f, -60.0f, 0, 2);

    float az_pt = 0.0f;

    ant.calc_beamwidth_deg(0, 3.0f, nullptr, nullptr, &az_pt, nullptr);
    CHECK(std::abs(az_pt) < 0.5f);

    ant.calc_beamwidth_deg(1, 3.0f, nullptr, nullptr, &az_pt, nullptr);
    CHECK(std::abs(az_pt - 60.0f) < 0.1f);

    ant.calc_beamwidth_deg(2, 3.0f, nullptr, nullptr, &az_pt, nullptr);
    CHECK(std::abs(az_pt + 60.0f) < 0.1f);
}

TEST_CASE("Arrayant calc_beamwidth_deg - Omni antenna")
{
    // Constant pattern: every direction is above any negative-dB threshold,
    // so the measured beamwidth equals the full grid extent.
    auto ant = quadriga_lib::generate_arrayant_omni<float>();

    float bw_az = 0.0f, bw_el = 0.0f;
    ant.calc_beamwidth_deg(0, 3.0f, &bw_az, &bw_el, nullptr, nullptr);

    CHECK(std::abs(bw_az - 360.0f) < 0.05f);
    CHECK(std::abs(bw_el - 180.0f) < 0.05f);
}

TEST_CASE("Arrayant calc_beamwidth_deg - Selective output pointers")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(30.0f, 30.0f);

    // All-null call must be a no-op (no crash, early return).
    CHECK_NOTHROW(ant.calc_beamwidth_deg(0, 3.0f, nullptr, nullptr, nullptr, nullptr));

    // Request only azimuth beamwidth
    float bw_az = -1.0f;
    ant.calc_beamwidth_deg(0, 3.0f, &bw_az, nullptr, nullptr, nullptr);
    CHECK(std::abs(bw_az - 30.0f) < 0.1f);

    // Request only elevation pointing angle
    float el_pt = 999.0f;
    ant.calc_beamwidth_deg(0, 3.0f, nullptr, nullptr, nullptr, &el_pt);
    CHECK(std::abs(el_pt) < 0.1f);
}

TEST_CASE("Arrayant calc_beamwidth_deg - Invalid element index")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>();
    float bw = 0.0f;
    REQUIRE_THROWS_AS(ant.calc_beamwidth_deg(5, 3.0f, &bw), std::invalid_argument);
}
