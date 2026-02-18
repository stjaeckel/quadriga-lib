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

#include "quadriga_lib.hpp"

#include <cmath>
#include <string>
#include <vector>

// ================================================================================================
// Helper: Build a simple multi-frequency arrayant vector from hand-crafted patterns
// Each entry has 1 element, 1 elevation sample, 2 azimuth samples (0 and pi)
// Pattern values scale with frequency index for easy verification
// ================================================================================================
static std::vector<quadriga_lib::arrayant<double>>
build_simple_multi(const arma::vec &center_freqs, bool polarimetric = false)
{
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (arma::uword i = 0; i < center_freqs.n_elem; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double scale = (double)(i + 1); // 1, 2, 3, ...
        ant.e_theta_re.zeros(1, 2, 1);
        ant.e_theta_re.slice(0) = arma::mat({{scale, -scale}});
        ant.e_theta_im.zeros(1, 2, 1);
        ant.e_theta_im.slice(0) = arma::mat({{0.5 * scale, -0.5 * scale}});
        if (polarimetric)
        {
            ant.e_phi_re.zeros(1, 2, 1);
            ant.e_phi_re.slice(0) = arma::mat({{-0.5 * scale, 0.5 * scale}});
            ant.e_phi_im.zeros(1, 2, 1);
            ant.e_phi_im.slice(0) = arma::mat({{0.25 * scale, -0.25 * scale}});
        }
        else
        {
            ant.e_phi_re.zeros(1, 2, 1);
            ant.e_phi_im.zeros(1, 2, 1);
        }
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.center_frequency = center_freqs[i];
        ant.name = "entry_" + std::to_string(i);
        vec.push_back(ant);
    }
    return vec;
}

// ================================================================================================
// SECTION 1: Input validation
// ================================================================================================

TEST_CASE("Interp multi - Empty vector throws")
{
    std::vector<quadriga_lib::arrayant<double>> empty_vec;
    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi(empty_vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im),
        std::invalid_argument);
}

TEST_CASE("Interp multi - Null azimuth throws")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat el = {0.0};
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi<double>(vec, nullptr, &el, &freq, &V_re, &V_im, &H_re, &H_im),
        std::invalid_argument);
}

TEST_CASE("Interp multi - Null elevation throws")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0};
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi<double>(vec, &az, nullptr, &freq, &V_re, &V_im, &H_re, &H_im),
        std::invalid_argument);
}

TEST_CASE("Interp multi - Null frequency throws")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};
    arma::cube V_re, V_im, H_re, H_im;
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi<double>(vec, &az, &el, nullptr, &V_re, &V_im, &H_re, &H_im),
        std::invalid_argument);
}

TEST_CASE("Interp multi - Null output throws")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi<double>(vec, &az, &el, &freq, nullptr, &V_im, &H_re, &H_im),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi<double>(vec, &az, &el, &freq, &V_re, nullptr, &H_re, &H_im),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi<double>(vec, &az, &el, &freq, &V_re, &V_im, nullptr, &H_im),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi<double>(vec, &az, &el, &freq, &V_re, &V_im, &H_re, nullptr),
        std::invalid_argument);
}

TEST_CASE("Interp multi - Mismatched azimuth/elevation size throws")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az(1, 4, arma::fill::zeros);
    arma::mat el(1, 3, arma::fill::zeros); // Mismatch
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im),
        std::invalid_argument);
}

TEST_CASE("Interp multi - Element index out of range throws")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;
    arma::uvec i_elem = {5}; // Only 1 element exists
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im, i_elem),
        std::invalid_argument);
}

TEST_CASE("Interp multi - Invalid arrayant_vec caught by validation")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);
    vec[1].azimuth_grid = {0.0}; // Intentionally corrupt: different grid size
    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1500.0};
    arma::cube V_re, V_im, H_re, H_im;
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im),
        std::invalid_argument);
}

TEST_CASE("Interp multi - Validation can be skipped")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;
    // Should not throw even with validate_input = false
    REQUIRE_NOTHROW(
        quadriga_lib::arrayant_interpolate_multi<double>(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im,
                                                         {}, nullptr, nullptr, false));
}

// ================================================================================================
// SECTION 2: Output sizing
// ================================================================================================

TEST_CASE("Interp multi - Output cube dimensions correct")
{
    arma::vec freqs = {500.0, 1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az(1, 7, arma::fill::zeros); // 7 angles
    arma::mat el(1, 7, arma::fill::zeros);
    arma::vec freq = {600.0, 1500.0, 1800.0, 3000.0}; // 4 query frequencies
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // n_out=1 (1 element, all selected), n_ang=7, n_freq=4
    CHECK(V_re.n_rows == 1);
    CHECK(V_re.n_cols == 7);
    CHECK(V_re.n_slices == 4);
    CHECK(V_im.n_rows == 1);
    CHECK(V_im.n_cols == 7);
    CHECK(V_im.n_slices == 4);
    CHECK(H_re.n_rows == 1);
    CHECK(H_re.n_cols == 7);
    CHECK(H_re.n_slices == 4);
    CHECK(H_im.n_rows == 1);
    CHECK(H_im.n_cols == 7);
    CHECK(H_im.n_slices == 4);
}

// ================================================================================================
// SECTION 3: Single entry vector (always returns that entry)
// ================================================================================================

TEST_CASE("Interp multi - Single entry returns same result regardless of query freq")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};

    // Query at various frequencies: below, at, above
    arma::vec freq = {100.0, 1000.0, 50000.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // All 3 slices should be identical (scale=1, az=0 → value=1.0)
    for (arma::uword s = 0; s < 3; ++s)
    {
        CHECK(std::abs(V_re(0, 0, s) - 1.0) < 1e-12);
        CHECK(std::abs(V_im(0, 0, s) - 0.5) < 1e-12);
    }
}

// ================================================================================================
// SECTION 4: Exact frequency match — should recover single-frequency interpolation
// ================================================================================================

TEST_CASE("Interp multi - Exact frequency match recovers member function result")
{
    double pi = arma::datum::pi;
    arma::vec freqs = {500.0, 1000.0, 2000.0};
    auto vec = build_simple_multi(freqs, true); // Polarimetric

    arma::mat az = {0.0, pi / 4.0, pi / 2.0, 3.0 * pi / 4.0};
    arma::mat el(1, 4, arma::fill::zeros);

    // Query at the exact center frequencies
    arma::vec freq = {500.0, 1000.0, 2000.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Compare each slice against single-frequency member function result
    for (arma::uword f = 0; f < 3; ++f)
    {
        arma::mat v_re, v_im, h_re, h_im;
        vec[f].interpolate(&az, &el, &v_re, &v_im, &h_re, &h_im);

        arma::mat slice_Vr = V_re.slice(f);
        arma::mat slice_Vi = V_im.slice(f);
        arma::mat slice_Hr = H_re.slice(f);
        arma::mat slice_Hi = H_im.slice(f);
        CHECK(arma::approx_equal(slice_Vr, v_re, "absdiff", 1e-12));
        CHECK(arma::approx_equal(slice_Vi, v_im, "absdiff", 1e-12));
        CHECK(arma::approx_equal(slice_Hr, h_re, "absdiff", 1e-12));
        CHECK(arma::approx_equal(slice_Hi, h_im, "absdiff", 1e-12));
    }
}

// ================================================================================================
// SECTION 5: Extrapolation (clamping)
// ================================================================================================

TEST_CASE("Interp multi - Extrapolation below clamps to first entry")
{
    arma::vec freqs = {500.0, 1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};

    // Query way below the lowest center frequency
    arma::vec freq = {10.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Should match the first entry (scale=1)
    arma::mat v_re, v_im, h_re, h_im;
    vec[0].interpolate(&az, &el, &v_re, &v_im, &h_re, &h_im);

    CHECK(std::abs(V_re(0, 0, 0) - v_re(0, 0)) < 1e-12);
    CHECK(std::abs(V_im(0, 0, 0) - v_im(0, 0)) < 1e-12);
}

TEST_CASE("Interp multi - Extrapolation above clamps to last entry")
{
    arma::vec freqs = {500.0, 1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};

    // Query way above the highest center frequency
    arma::vec freq = {99999.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Should match the last entry (scale=3)
    arma::mat v_re, v_im, h_re, h_im;
    vec[2].interpolate(&az, &el, &v_re, &v_im, &h_re, &h_im);

    CHECK(std::abs(V_re(0, 0, 0) - v_re(0, 0)) < 1e-12);
    CHECK(std::abs(V_im(0, 0, 0) - v_im(0, 0)) < 1e-12);
}

// ================================================================================================
// SECTION 6: Frequency interpolation — midpoint between two linear-valued entries
// ================================================================================================

TEST_CASE("Interp multi - Midpoint frequency interpolation")
{
    // Two entries with constant patterns (no angular variation) → linear in frequency
    // Entry 0: e_theta_re = 1.0 at all angles, center_freq = 1000
    // Entry 1: e_theta_re = 3.0 at all angles, center_freq = 2000
    // At f=1500 (midpoint, w=0.5), amplitude slerp of two in-phase signals = linear blend = 2.0
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double val = (i == 0) ? 1.0 : 3.0;
        ant.e_theta_re.ones(1, 2, 1) * val;
        ant.e_theta_re.fill(val);
        ant.e_theta_im.zeros(1, 2, 1);
        ant.e_phi_re.zeros(1, 2, 1);
        ant.e_phi_im.zeros(1, 2, 1);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1500.0}; // Exact midpoint

    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Both entries are in-phase (positive reals), so slerp = linear blend of amplitudes
    // w=0.5: amp = 0.5*1.0 + 0.5*3.0 = 2.0, phase = 0 → result = 2.0
    CHECK(std::abs(V_re(0, 0, 0) - 2.0) < 1e-10);
    CHECK(std::abs(V_im(0, 0, 0)) < 1e-12);
}

TEST_CASE("Interp multi - Quarter-point frequency interpolation")
{
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double val = (i == 0) ? 2.0 : 6.0;
        ant.e_theta_re.zeros(1, 2, 1);
        ant.e_theta_re.fill(val);
        ant.e_theta_im.zeros(1, 2, 1);
        ant.e_phi_re.zeros(1, 2, 1);
        ant.e_phi_im.zeros(1, 2, 1);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};

    // w=0.25 → amp = 0.75*2 + 0.25*6 = 3.0
    arma::vec freq = {1250.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);
    CHECK(std::abs(V_re(0, 0, 0) - 3.0) < 1e-10);

    // w=0.75 → amp = 0.25*2 + 0.75*6 = 5.0
    freq = {1750.0};
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);
    CHECK(std::abs(V_re(0, 0, 0) - 5.0) < 1e-10);
}

// ================================================================================================
// SECTION 7: Spherical interpolation across frequency — phase rotation
// ================================================================================================

TEST_CASE("Interp multi - Slerp with 90-degree phase shift across frequency")
{
    // Entry 0: value = (1, 0) → amp=1, phase=0
    // Entry 1: value = (0, 1) → amp=1, phase=pi/2
    // At midpoint: slerp should give phase=pi/4, amp=1 → (cos(pi/4), sin(pi/4))
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        ant.e_theta_re.zeros(1, 1, 1);
        ant.e_theta_im.zeros(1, 1, 1);
        ant.e_phi_re.zeros(1, 1, 1);
        ant.e_phi_im.zeros(1, 1, 1);
        ant.e_theta_re(0, 0, 0) = (i == 0) ? 1.0 : 0.0;
        ant.e_theta_im(0, 0, 0) = (i == 0) ? 0.0 : 1.0;
        ant.azimuth_grid = {0.0};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1500.0}; // midpoint, w=0.5

    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Slerp: amp = 0.5*1 + 0.5*1 = 1; phase interpolated to pi/4
    double rs2 = 1.0 / std::sqrt(2.0);
    CHECK(std::abs(V_re(0, 0, 0) - rs2) < 1e-10);
    CHECK(std::abs(V_im(0, 0, 0) - rs2) < 1e-10);
}

TEST_CASE("Interp multi - Slerp with amplitude ramp across frequency")
{
    // Entry 0: value = (2, 0) → amp=2, phase=0
    // Entry 1: value = (0, 4) → amp=4, phase=pi/2
    // At midpoint w=0.5: amp = 0.5*2 + 0.5*4 = 3; phase = pi/4
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        ant.e_theta_re.zeros(1, 1, 1);
        ant.e_theta_im.zeros(1, 1, 1);
        ant.e_phi_re.zeros(1, 1, 1);
        ant.e_phi_im.zeros(1, 1, 1);
        ant.e_theta_re(0, 0, 0) = (i == 0) ? 2.0 : 0.0;
        ant.e_theta_im(0, 0, 0) = (i == 0) ? 0.0 : 4.0;
        ant.azimuth_grid = {0.0};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1500.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    double rs2 = 1.0 / std::sqrt(2.0);
    double amp = 3.0; // 0.5*2 + 0.5*4
    CHECK(std::abs(V_re(0, 0, 0) - amp * rs2) < 1e-10);
    CHECK(std::abs(V_im(0, 0, 0) - amp * rs2) < 1e-10);
}

// ================================================================================================
// SECTION 8: Polarimetric responses — both V and H components
// ================================================================================================

TEST_CASE("Interp multi - Polarimetric exact match preserves both V and H")
{
    arma::vec freqs = {500.0, 1000.0, 2000.0};
    auto vec = build_simple_multi(freqs, true);

    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1000.0}; // Exact match → entry 1 (scale=2)
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // At az=0: e_theta_re = scale=2, e_theta_im = 0.5*scale=1, e_phi_re = -0.5*scale=-1, e_phi_im = 0.25*scale=0.5
    CHECK(std::abs(V_re(0, 0, 0) - 2.0) < 1e-12);
    CHECK(std::abs(V_im(0, 0, 0) - 1.0) < 1e-12);
    CHECK(std::abs(H_re(0, 0, 0) - (-1.0)) < 1e-12);
    CHECK(std::abs(H_im(0, 0, 0) - 0.5) < 1e-12);
}

TEST_CASE("Interp multi - Polarimetric frequency interpolation blends both V and H")
{
    // Two entries with in-phase patterns, different amplitudes for V and H
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double sv = (i == 0) ? 1.0 : 3.0; // V amplitude
        double sh = (i == 0) ? 4.0 : 2.0; // H amplitude (decreasing)
        ant.e_theta_re.zeros(1, 1, 1);
        ant.e_theta_re(0, 0, 0) = sv;
        ant.e_theta_im.zeros(1, 1, 1);
        ant.e_phi_re.zeros(1, 1, 1);
        ant.e_phi_re(0, 0, 0) = sh;
        ant.e_phi_im.zeros(1, 1, 1);
        ant.azimuth_grid = {0.0};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1500.0}; // w=0.5
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // In-phase positive reals → slerp = linear amplitude blend
    CHECK(std::abs(V_re(0, 0, 0) - 2.0) < 1e-10); // 0.5*1 + 0.5*3
    CHECK(std::abs(H_re(0, 0, 0) - 3.0) < 1e-10); // 0.5*4 + 0.5*2
}

// ================================================================================================
// SECTION 9: EM antenna in GHz domain
// ================================================================================================

TEST_CASE("Interp multi - GHz frequency domain for EM antennas")
{
    // Simulate a dipole-like pattern at different GHz frequencies
    double pi = arma::datum::pi;
    arma::vec center_freqs = {0.9e9, 1.8e9, 2.4e9, 5.0e9}; // 900 MHz to 5 GHz
    auto vec = build_simple_multi(center_freqs);

    arma::mat az = {0.0}, el = {0.0};

    // Query at exact GHz frequencies
    arma::vec freq = {0.9e9, 1.8e9, 2.4e9, 5.0e9};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    for (arma::uword f = 0; f < 4; ++f)
    {
        arma::mat v_re, v_im, h_re, h_im;
        vec[f].interpolate(&az, &el, &v_re, &v_im, &h_re, &h_im);
        CHECK(std::abs(V_re(0, 0, f) - v_re(0, 0)) < 1e-12);
    }

    // Query between GHz frequencies
    arma::vec freq_mid = {1.35e9}; // Between 900M and 1.8G
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq_mid, &V_re, &V_im, &H_re, &H_im);

    // Should be between entry 0 (scale=1) and entry 1 (scale=2) results
    arma::mat v0_re, v0_im, h0_re, h0_im, v1_re, v1_im, h1_re, h1_im;
    vec[0].interpolate(&az, &el, &v0_re, &v0_im, &h0_re, &h0_im);
    vec[1].interpolate(&az, &el, &v1_re, &v1_im, &h1_re, &h1_im);
    double result = V_re(0, 0, 0);
    double lo = v0_re(0, 0), hi = v1_re(0, 0);
    double val_min = std::min(lo, hi), val_max = std::max(lo, hi);
    CHECK(result >= val_min - 1e-10);
    CHECK(result <= val_max + 1e-10);
}

// ================================================================================================
// SECTION 10: Speaker Hz domain with generate_speaker
// ================================================================================================

TEST_CASE("Interp multi - Speaker Hz domain with generate_speaker")
{
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
    auto spk = quadriga_lib::generate_speaker<double>(
        "piston", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "hemisphere",
        0.0, 0.0, 0.0, 0.15, 0.25, freqs, 10.0);

    arma::mat az = {0.0, 1.5708, -1.5708}; // Front, left, right
    arma::mat el(1, 3, arma::fill::zeros);

    // Query at exact frequencies
    arma::vec freq_exact = {100.0, 1000.0, 10000.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(spk, &az, &el, &freq_exact, &V_re, &V_im, &H_re, &H_im);

    CHECK(V_re.n_rows == 1);
    CHECK(V_re.n_cols == 3);
    CHECK(V_re.n_slices == 3);

    // Verify exact match against single-frequency interpolation
    // entry 0 → 100 Hz, entry 2 → 1000 Hz, entry 4 → 10000 Hz
    arma::uword idx_map[3] = {0, 2, 4};
    for (int f = 0; f < 3; ++f)
    {
        arma::mat v_re, v_im, h_re, h_im;
        spk[idx_map[f]].interpolate(&az, &el, &v_re, &v_im, &h_re, &h_im);
        arma::mat slice_Vr = V_re.slice(f);
        CHECK(arma::approx_equal(slice_Vr, v_re, "absdiff", 1e-10));
    }

    // Query between frequencies — should produce smooth interpolation
    arma::vec freq_between = {750.0}; // Between 500 and 1000
    quadriga_lib::arrayant_interpolate_multi(spk, &az, &el, &freq_between, &V_re, &V_im, &H_re, &H_im);
    CHECK(V_re.n_slices == 1);
    CHECK(V_re.is_finite());
    CHECK(!V_re.has_nan());
}

// ================================================================================================
// SECTION 11: Multi-element antennas
// ================================================================================================

TEST_CASE("Interp multi - Two-element antenna with element selection")
{
    double pi = arma::datum::pi;
    arma::vec freqs = {1000.0, 2000.0};

    // Build 2-element patterns manually
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (arma::uword i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double s = (double)(i + 1);
        ant.e_theta_re.zeros(1, 2, 2);
        ant.e_theta_re(0, 0, 0) = s;        // Element 0, az=0
        ant.e_theta_re(0, 1, 0) = -s;       // Element 0, az=pi
        ant.e_theta_re(0, 0, 1) = 2.0 * s;  // Element 1, az=0
        ant.e_theta_re(0, 1, 1) = -2.0 * s; // Element 1, az=pi
        ant.e_theta_im.zeros(1, 2, 2);
        ant.e_phi_re.zeros(1, 2, 2);
        ant.e_phi_im.zeros(1, 2, 2);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.center_frequency = freqs[i];
        ant.element_pos.zeros(3, 2);
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};

    // Select only element 1
    arma::uvec i_elem = {1};
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im, i_elem);

    CHECK(V_re.n_rows == 1);                      // 1 selected element
    CHECK(std::abs(V_re(0, 0, 0) - 2.0) < 1e-12); // Element 1, entry 0, scale=1, factor=2

    // Select both elements
    arma::uvec i_both = {0, 1};
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im, i_both);
    CHECK(V_re.n_rows == 2);
    CHECK(std::abs(V_re(0, 0, 0) - 1.0) < 1e-12); // Element 0
    CHECK(std::abs(V_re(1, 0, 0) - 2.0) < 1e-12); // Element 1
}

// ================================================================================================
// SECTION 12: 2-way speaker (concat) with frequency interpolation
// ================================================================================================

TEST_CASE("Interp multi - 2-way speaker documentation example")
{
    arma::vec entry_freqs = {100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 16000.0};
    auto woofer = quadriga_lib::generate_speaker<double>(
        "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, entry_freqs, 10.0);
    auto tweeter = quadriga_lib::generate_speaker<double>(
        "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, entry_freqs, 10.0);
    auto speaker = quadriga_lib::arrayant_concat_multi(woofer, tweeter);

    arma::mat az = {0.0, 1.5708, -1.5708, 3.14159};
    arma::mat el(1, 4, arma::fill::zeros);

    arma::vec query_freqs = {150.0, 750.0, 3000.0, 12000.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(speaker, &az, &el, &query_freqs,
                                             &V_re, &V_im, &H_re, &H_im);

    // [n_elements=2, n_angles=4, n_frequencies=4]
    CHECK(V_re.n_rows == 2);
    CHECK(V_re.n_cols == 4);
    CHECK(V_re.n_slices == 4);
    CHECK(V_re.is_finite());
    CHECK(!V_re.has_nan());

    // At 150 Hz (between 100 and 200), woofer should dominate, tweeter should be very small
    // Tweeter low-pass cutoff is 1500 Hz, so at 150 Hz it's well below passband
    double woofer_150 = std::abs(V_re(0, 0, 0));  // Element 0 (woofer), on-axis, 150 Hz
    double tweeter_150 = std::abs(V_re(1, 0, 0)); // Element 1 (tweeter), on-axis, 150 Hz
    CHECK(woofer_150 > tweeter_150 * 5.0);        // Woofer should be much stronger at 150 Hz

    // At 12 kHz (above last entry 10kHz, clamped to 16kHz entry), tweeter should dominate
    double woofer_12k = std::abs(V_re(0, 0, 3));
    double tweeter_12k = std::abs(V_re(1, 0, 3));
    CHECK(tweeter_12k > woofer_12k * 2.0); // Tweeter should be stronger at 12 kHz
}

// ================================================================================================
// SECTION 13: Orientation support
// ================================================================================================

TEST_CASE("Interp multi - Orientation rotates pattern consistently across frequency")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);
    double pi = arma::datum::pi;

    arma::mat az = {0.0, pi / 4.0, pi / 2.0};
    arma::mat el(1, 3, arma::fill::zeros);

    // Without orientation
    arma::cube V_re_no, V_im_no, H_re_no, H_im_no;
    arma::vec freq = {1000.0};
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq,
                                             &V_re_no, &V_im_no, &H_re_no, &H_im_no);

    // With z-rotation
    arma::Cube<double> orientation(3, 1, 1, arma::fill::zeros);
    orientation(2, 0, 0) = pi / 8.0; // 22.5° heading

    arma::cube V_re_rot, V_im_rot, H_re_rot, H_im_rot;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq,
                                             &V_re_rot, &V_im_rot, &H_re_rot, &H_im_rot,
                                             {}, &orientation);

    // Results should differ due to rotation
    arma::mat slice_no = V_re_no.slice(0);
    arma::mat slice_rot = V_re_rot.slice(0);
    CHECK(!arma::approx_equal(slice_no, slice_rot, "absdiff", 1e-6));
}

// ================================================================================================
// SECTION 14: Alternative element positions
// ================================================================================================

TEST_CASE("Interp multi - Alternative element positions used")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1000.0};

    // Default positions (zeros from build_simple_multi)
    arma::cube V_re1, V_im1, H_re1, H_im1;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re1, &V_im1, &H_re1, &H_im1);

    // With alternative element positions (should not affect V_re since no dist calculation,
    // but the internal element_pos_local is different)
    arma::mat alt_pos = arma::mat({0.5, 0.0, 0.0}).t();
    arma::cube V_re2, V_im2, H_re2, H_im2;
    quadriga_lib::arrayant_interpolate_multi<double>(vec, &az, &el, &freq, &V_re2, &V_im2, &H_re2, &H_im2,
                                                     {}, nullptr, &alt_pos);

    // V_re should be the same since dist is not requested and orientation is identity
    arma::mat s1 = V_re1.slice(0), s2 = V_re2.slice(0);
    CHECK(arma::approx_equal(s1, s2, "absdiff", 1e-12));
}

// ================================================================================================
// SECTION 15: Multiple query frequencies spanning full range
// ================================================================================================

TEST_CASE("Interp multi - Dense frequency sweep produces smooth output")
{
    arma::vec entry_freqs = {500.0, 1000.0, 2000.0, 5000.0};
    auto vec = build_simple_multi(entry_freqs);

    arma::mat az = {0.0}, el = {0.0};

    // Sweep from 100 Hz to 10 kHz in 50 steps
    arma::vec freq = arma::linspace(100.0, 10000.0, 50);
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    CHECK(V_re.n_slices == 50);
    CHECK(V_re.is_finite());
    CHECK(!V_re.has_nan());

    // Values should be monotonically non-decreasing (all entries are in-phase, amplitudes increase)
    for (arma::uword i = 1; i < 50; ++i)
        CHECK(V_re(0, 0, i) >= V_re(0, 0, i - 1) - 1e-10);
}

// ================================================================================================
// SECTION 16: Caching behavior — repeated bracket boundary
// ================================================================================================

TEST_CASE("Interp multi - Caching: adjacent frequencies in same bracket")
{
    arma::vec entry_freqs = {1000.0, 2000.0, 3000.0};
    auto vec = build_simple_multi(entry_freqs);
    arma::mat az = {0.0}, el = {0.0};

    // Three frequencies in the same bracket [1000, 2000]
    arma::vec freq = {1100.0, 1500.0, 1900.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Values should be monotonically increasing
    CHECK(V_re(0, 0, 0) < V_re(0, 0, 1));
    CHECK(V_re(0, 0, 1) < V_re(0, 0, 2));
}

// ================================================================================================
// SECTION 17: Per-element angles (spherical wave)
// ================================================================================================

TEST_CASE("Interp multi - Per-element azimuth angles")
{
    double pi = arma::datum::pi;
    arma::vec freqs = {1000.0, 2000.0};

    // 2-element antenna
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (arma::uword i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double s = (double)(i + 1);
        ant.e_theta_re.zeros(1, 2, 2);
        ant.e_theta_re.fill(s); // Constant pattern
        ant.e_theta_im.zeros(1, 2, 2);
        ant.e_phi_re.zeros(1, 2, 2);
        ant.e_phi_im.zeros(1, 2, 2);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.center_frequency = freqs[i];
        ant.element_pos.zeros(3, 2);
        vec.push_back(ant);
    }

    // Per-element angles: [n_out=2, n_ang=1]
    arma::mat az = arma::mat({0.0, pi / 4.0}).t(); // Different az for each element
    arma::mat el = arma::mat({0.0, 0.0}).t();

    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;
    arma::uvec i_elem = {0, 1};
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im, i_elem);

    CHECK(V_re.n_rows == 2);
    CHECK(V_re.n_cols == 1);
}

// ================================================================================================
// SECTION 18: EM polarimetric antenna with x-rotation (cross-pol test)
// ================================================================================================

TEST_CASE("Interp multi - Polarimetric EM antenna with orientation")
{
    // Build a vertically polarized antenna at two frequencies
    double pi = arma::datum::pi;
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double s = (double)(i + 1);
        ant.e_theta_re.zeros(1, 2, 1);
        ant.e_theta_re.fill(s); // Only V-pol
        ant.e_theta_im.zeros(1, 2, 1);
        ant.e_phi_re.zeros(1, 2, 1);
        ant.e_phi_im.zeros(1, 2, 1);
        ant.azimuth_grid = {0.0, pi};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1.0e9 : 2.0e9;
        vec.push_back(ant);
    }

    arma::mat az(1, 1, arma::fill::zeros);
    arma::mat el(1, 1, arma::fill::zeros);

    // Apply 45° bank rotation → should mix V into H equally
    arma::Cube<double> orientation(3, 1, 1, arma::fill::zeros);
    orientation(0, 0, 0) = pi / 4.0; // Bank angle = 45°

    arma::vec freq = {1.0e9};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im,
                                             {}, &orientation);

    double rs2 = 1.0 / std::sqrt(2.0);
    // V-pol input of 1.0 rotated 45° → V_re ≈ cos(45°), H_re ≈ sin(45°)
    CHECK(std::abs(V_re(0, 0, 0) - rs2) < 1e-10);
    CHECK(std::abs(H_re(0, 0, 0) - rs2) < 1e-10);
}

// ================================================================================================
// SECTION 19: Two entries with zero at one frequency (amplitude crossover)
// ================================================================================================

TEST_CASE("Interp multi - Zero amplitude at one frequency triggers linear fallback")
{
    // Entry 0: value = (1, 0) → amp=1
    // Entry 1: value = (0, 0) → amp=0
    // slerp with zero should fall back to linear interpolation
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        ant.e_theta_re.zeros(1, 1, 1);
        ant.e_theta_im.zeros(1, 1, 1);
        ant.e_phi_re.zeros(1, 1, 1);
        ant.e_phi_im.zeros(1, 1, 1);
        ant.e_theta_re(0, 0, 0) = (i == 0) ? 1.0 : 0.0;
        ant.azimuth_grid = {0.0};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1500.0}; // w=0.5
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Linear fallback: 0.5*1.0 + 0.5*0.0 = 0.5
    CHECK(std::abs(V_re(0, 0, 0) - 0.5) < 1e-10);
    CHECK(V_re.is_finite());
}

// ================================================================================================
// SECTION 20: Both entries zero → output zero
// ================================================================================================

TEST_CASE("Interp multi - Both entries zero produces zero output")
{
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        ant.e_theta_re.zeros(1, 1, 1);
        ant.e_theta_im.zeros(1, 1, 1);
        ant.e_phi_re.zeros(1, 1, 1);
        ant.e_phi_im.zeros(1, 1, 1);
        ant.azimuth_grid = {0.0};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1500.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    CHECK(std::abs(V_re(0, 0, 0)) < 1e-14);
    CHECK(std::abs(V_im(0, 0, 0)) < 1e-14);
    CHECK(std::abs(H_re(0, 0, 0)) < 1e-14);
    CHECK(std::abs(H_im(0, 0, 0)) < 1e-14);
}

// ================================================================================================
// SECTION 21: 180-degree phase shift (anti-phase) across frequency
// ================================================================================================

TEST_CASE("Interp multi - Anti-phase entries use linear fallback")
{
    // Entry 0: value = (1, 0) → amp=1, phase=0
    // Entry 1: value = (-1, 0) → amp=1, phase=pi (anti-phase)
    // cos(phase_diff) = -1 → fully linear interpolation
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        ant.e_theta_re.zeros(1, 1, 1);
        ant.e_theta_im.zeros(1, 1, 1);
        ant.e_phi_re.zeros(1, 1, 1);
        ant.e_phi_im.zeros(1, 1, 1);
        ant.e_theta_re(0, 0, 0) = (i == 0) ? 1.0 : -1.0;
        ant.azimuth_grid = {0.0};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1500.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Linear: 0.5*1.0 + 0.5*(-1.0) = 0.0
    CHECK(std::abs(V_re(0, 0, 0)) < 1e-10);
    CHECK(V_re.is_finite());
}

// ================================================================================================
// SECTION 22: Many frequency entries (dense sampling)
// ================================================================================================

TEST_CASE("Interp multi - Many entries with dense query frequencies")
{
    // 20 entries from 100 Hz to 10 kHz
    arma::vec entry_freqs = arma::linspace(100.0, 10000.0, 20);
    auto vec = build_simple_multi(entry_freqs);

    arma::mat az = {0.0}, el = {0.0};

    // Query at 100 frequencies spanning the full range
    arma::vec freq = arma::linspace(50.0, 15000.0, 100);
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    CHECK(V_re.n_slices == 100);
    CHECK(V_re.is_finite());
    CHECK(!V_re.has_nan());
}

// ================================================================================================
// SECTION 23: Float template
// ================================================================================================

TEST_CASE("Interp multi - Float template")
{
    float pi = (float)arma::datum::pi;

    std::vector<quadriga_lib::arrayant<float>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<float> ant;
        float val = (float)(i + 1);
        ant.e_theta_re.zeros(1, 2, 1);
        ant.e_theta_re.fill(val);
        ant.e_theta_im.zeros(1, 2, 1);
        ant.e_phi_re.zeros(1, 2, 1);
        ant.e_phi_im.zeros(1, 2, 1);
        ant.azimuth_grid = {0.0f, pi};
        ant.elevation_grid = {0.0f};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::fmat az = {0.0f}, el = {0.0f};
    arma::fvec freq = {1500.0f};
    arma::fcube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    CHECK(std::abs(V_re(0, 0, 0) - 1.5f) < 0.01f);
}

// ================================================================================================
// SECTION 24: Multiple angles and multiple frequencies simultaneously
// ================================================================================================

TEST_CASE("Interp multi - Multiple angles × multiple frequencies grid")
{
    double pi = arma::datum::pi;
    arma::vec entry_freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(entry_freqs, true); // Polarimetric

    // 5 azimuth angles
    arma::mat az = arma::linspace<arma::mat>(-pi, pi, 5).t();
    arma::mat el(1, 5, arma::fill::zeros);

    // 3 query frequencies
    arma::vec freq = {500.0, 1500.0, 3000.0};

    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    CHECK(V_re.n_rows == 1);
    CHECK(V_re.n_cols == 5);
    CHECK(V_re.n_slices == 3);

    // Slice 0 (500 Hz, clamped to entry 0) should match slice for exact freq 1000
    // Slice 2 (3000 Hz, clamped to entry 1) should match entry 1 directly
    arma::vec freq_exact = {1000.0, 2000.0};
    arma::cube V_re_ex, V_im_ex, H_re_ex, H_im_ex;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq_exact, &V_re_ex, &V_im_ex, &H_re_ex, &H_im_ex);

    arma::mat s_clamp_lo = V_re.slice(0), s_exact_lo = V_re_ex.slice(0);
    arma::mat s_clamp_hi = V_re.slice(2), s_exact_hi = V_re_ex.slice(1);
    CHECK(arma::approx_equal(s_clamp_lo, s_exact_lo, "absdiff", 1e-12));
    CHECK(arma::approx_equal(s_clamp_hi, s_exact_hi, "absdiff", 1e-12));
}

// ================================================================================================
// SECTION 25: Frequency-dependent directivity with generate_speaker
// ================================================================================================

TEST_CASE("Interp multi - Piston directivity narrows with frequency")
{
    // A piston speaker becomes more directional at higher frequencies
    arma::vec entry_freqs = {200.0, 1000.0, 5000.0, 15000.0};
    auto spk = quadriga_lib::generate_speaker<double>(
        "piston", 0.05, 80.0, 20000.0, 12.0, 12.0, 85.0, "hemisphere",
        0.0, 0.0, 0.0, 0.15, 0.25, entry_freqs, 10.0);

    double pi = arma::datum::pi;
    arma::mat az = {0.0, pi / 2.0}; // On-axis and 90° off-axis
    arma::mat el(1, 2, arma::fill::zeros);

    arma::vec freq = {200.0, 5000.0, 15000.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(spk, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // At higher frequencies the ratio on-axis/off-axis should increase
    double ratio_200 = std::abs(V_re(0, 0, 0)) / (std::abs(V_re(0, 1, 0)) + 1e-20);
    double ratio_5k = std::abs(V_re(0, 0, 1)) / (std::abs(V_re(0, 1, 1)) + 1e-20);
    double ratio_15k = std::abs(V_re(0, 0, 2)) / (std::abs(V_re(0, 1, 2)) + 1e-20);

    CHECK(ratio_5k > ratio_200);
    CHECK(ratio_15k > ratio_5k);
}

// ================================================================================================
// SECTION 26: Continuity at bracket boundaries
// ================================================================================================

TEST_CASE("Interp multi - Continuity at frequency entry points")
{
    arma::vec entry_freqs = {1000.0, 2000.0, 3000.0};
    auto vec = build_simple_multi(entry_freqs);
    arma::mat az = {0.0}, el = {0.0};

    // Query slightly below and above the middle entry (2000 Hz)
    arma::vec freq = {1999.0, 2000.0, 2001.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // All three should be very close (continuous)
    CHECK(std::abs(V_re(0, 0, 0) - V_re(0, 0, 1)) < 0.01);
    CHECK(std::abs(V_re(0, 0, 1) - V_re(0, 0, 2)) < 0.01);
}

// ================================================================================================
// SECTION 27: Power conservation during frequency slerp
// ================================================================================================

TEST_CASE("Interp multi - Total power preserved during frequency interpolation")
{
    // Two entries with same amplitude, different phase → slerp preserves amplitude
    std::vector<quadriga_lib::arrayant<double>> vec;
    for (int i = 0; i < 2; ++i)
    {
        quadriga_lib::arrayant<double> ant;
        double phase = (i == 0) ? 0.0 : arma::datum::pi / 3.0; // 0° and 60°
        ant.e_theta_re.zeros(1, 1, 1);
        ant.e_theta_im.zeros(1, 1, 1);
        ant.e_theta_re(0, 0, 0) = 2.0 * std::cos(phase);
        ant.e_theta_im(0, 0, 0) = 2.0 * std::sin(phase);
        ant.e_phi_re.zeros(1, 1, 1);
        ant.e_phi_im.zeros(1, 1, 1);
        ant.azimuth_grid = {0.0};
        ant.elevation_grid = {0.0};
        ant.center_frequency = (i == 0) ? 1000.0 : 2000.0;
        vec.push_back(ant);
    }

    arma::mat az = {0.0}, el = {0.0};

    // Query at several points between entries
    arma::vec freq = arma::linspace(1000.0, 2000.0, 11);
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Amplitude should be constant = 2.0 at every frequency (slerp preserves amplitude for equal-amp signals)
    for (arma::uword f = 0; f < freq.n_elem; ++f)
    {
        double amp = std::sqrt(V_re(0, 0, f) * V_re(0, 0, f) + V_im(0, 0, f) * V_im(0, 0, f));
        CHECK(std::abs(amp - 2.0) < 0.01);
    }
}

// ================================================================================================
// SECTION 28: Duplicated element indices
// ================================================================================================

TEST_CASE("Interp multi - Duplicated element indices")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1000.0};

    arma::uvec i_elem = {0, 0, 0}; // Same element three times
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im, i_elem);

    CHECK(V_re.n_rows == 3);
    // All rows should be identical
    CHECK(std::abs(V_re(0, 0, 0) - V_re(1, 0, 0)) < 1e-14);
    CHECK(std::abs(V_re(0, 0, 0) - V_re(2, 0, 0)) < 1e-14);
}

// ================================================================================================
// SECTION 29: Single query frequency
// ================================================================================================

TEST_CASE("Interp multi - Single query frequency output has 1 slice")
{
    arma::vec freqs = {1000.0, 2000.0, 3000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0, 1.0, 2.0};
    arma::mat el(1, 3, arma::fill::zeros);
    arma::vec freq = {1500.0};

    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    CHECK(V_re.n_slices == 1);
    CHECK(V_re.n_cols == 3);
}

// ================================================================================================
// SECTION 30: GHz antenna full workflow — generate, concat, interpolate
// ================================================================================================

TEST_CASE("Interp multi - GHz antenna workflow: generate, rotate, interpolate")
{
    // Build a dual-pol patch antenna at multiple GHz frequencies
    double pi = arma::datum::pi;
    arma::vec ghz_freqs = {2.4e9, 2.45e9, 2.5e9};

    std::vector<quadriga_lib::arrayant<double>> vec;
    for (arma::uword i = 0; i < ghz_freqs.n_elem; ++i)
    {
        // Create a simple V-pol and H-pol pattern at each frequency
        auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
        ant.center_frequency = ghz_freqs[i];
        // Scale the pattern slightly with frequency to simulate frequency-dependent gain
        double scale = 1.0 + 0.1 * (double)i;
        ant.e_theta_re *= scale;
        ant.e_theta_im *= scale;
        vec.push_back(ant);
    }

    arma::mat az = {0.0, pi / 4.0, pi / 2.0};
    arma::mat el(1, 3, arma::fill::zeros);

    // Query at WiFi channel center frequencies
    arma::vec freq = {2.412e9, 2.437e9, 2.462e9, 2.484e9};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    CHECK(V_re.n_slices == 4);
    CHECK(V_re.is_finite());
    CHECK(!V_re.has_nan());

    // On-axis gain should increase with frequency (due to scale factor)
    CHECK(std::abs(V_re(0, 0, 0)) <= std::abs(V_re(0, 0, 3)) + 0.01);
}

// ================================================================================================
// SECTION 31: Incorrect orientation dimensions throw
// ================================================================================================

TEST_CASE("Interp multi - Bad orientation dimensions throw")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;

    // Wrong number of rows (should be 3)
    arma::Cube<double> bad_orient(2, 1, 1, arma::fill::zeros);
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im,
                                                 {}, &bad_orient),
        std::invalid_argument);
}

TEST_CASE("Interp multi - Bad element_pos_i dimensions throw")
{
    arma::vec freqs = {1000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};
    arma::vec freq = {1000.0};
    arma::cube V_re, V_im, H_re, H_im;

    // Wrong number of rows (should be 3)
    arma::mat bad_pos(2, 1, arma::fill::zeros);
    REQUIRE_THROWS_AS(
        quadriga_lib::arrayant_interpolate_multi<double>(vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im,
                                                         {}, nullptr, &bad_pos),
        std::invalid_argument);
}

// ================================================================================================
// SECTION 32: Near-exact frequency match (within tolerance)
// ================================================================================================

TEST_CASE("Interp multi - Near-exact match within 1e-6 tolerance")
{
    arma::vec freqs = {1000.0, 2000.0};
    auto vec = build_simple_multi(freqs);
    arma::mat az = {0.0}, el = {0.0};

    // Query at 1000.0 + tiny offset (within 1e-6 relative tolerance)
    arma::vec freq = {1000.0000001};
    arma::cube V_re1, V_im1, H_re1, H_im1;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq, &V_re1, &V_im1, &H_re1, &H_im1);

    // Should match exact 1000.0 query
    arma::vec freq_exact = {1000.0};
    arma::cube V_re2, V_im2, H_re2, H_im2;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq_exact, &V_re2, &V_im2, &H_re2, &H_im2);

    CHECK(std::abs(V_re1(0, 0, 0) - V_re2(0, 0, 0)) < 1e-10);
}

// ================================================================================================
// SECTION 33: Subwoofer extrapolation — pattern at high frequency clamped
// ================================================================================================

TEST_CASE("Interp multi - Subwoofer model clamped at high frequency")
{
    // Generate a subwoofer that only covers low frequencies
    arma::vec entry_freqs = {30.0, 50.0, 100.0, 200.0, 500.0};
    auto subwoofer = quadriga_lib::generate_speaker<double>(
        "piston", 0.20, 30.0, 500.0, 12.0, 24.0, 90.0, "hemisphere",
        0.0, 0.0, 0.0, 0.15, 0.25, entry_freqs, 10.0);

    arma::mat az = {0.0}, el = {0.0};

    // Query at 12 kHz — way above subwoofer range
    arma::vec freq = {12000.0};
    arma::cube V_re, V_im, H_re, H_im;
    quadriga_lib::arrayant_interpolate_multi(subwoofer, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im);

    // Should be clamped to the 500 Hz entry
    arma::mat v_re, v_im, h_re, h_im;
    subwoofer.back().interpolate(&az, &el, &v_re, &v_im, &h_re, &h_im);

    CHECK(std::abs(V_re(0, 0, 0) - v_re(0, 0)) < 1e-12);
    CHECK(V_re.is_finite());
}

// ================================================================================================
// SECTION 34: Reversed query frequency order still works
// ================================================================================================

TEST_CASE("Interp multi - Reversed query frequency order")
{
    arma::vec entry_freqs = {1000.0, 2000.0, 3000.0};
    auto vec = build_simple_multi(entry_freqs);
    arma::mat az = {0.0}, el = {0.0};

    // Forward order
    arma::vec freq_fwd = {1200.0, 1800.0, 2500.0};
    arma::cube V_fwd, Vi_fwd, Hf, Hi_f;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq_fwd, &V_fwd, &Vi_fwd, &Hf, &Hi_f);

    // Reversed order
    arma::vec freq_rev = {2500.0, 1800.0, 1200.0};
    arma::cube V_rev, Vi_rev, Hr, Hi_r;
    quadriga_lib::arrayant_interpolate_multi(vec, &az, &el, &freq_rev, &V_rev, &Vi_rev, &Hr, &Hi_r);

    // Results should match (just different slice order)
    CHECK(std::abs(V_fwd(0, 0, 0) - V_rev(0, 0, 2)) < 1e-12);
    CHECK(std::abs(V_fwd(0, 0, 1) - V_rev(0, 0, 1)) < 1e-12);
    CHECK(std::abs(V_fwd(0, 0, 2) - V_rev(0, 0, 0)) < 1e-12);
}
