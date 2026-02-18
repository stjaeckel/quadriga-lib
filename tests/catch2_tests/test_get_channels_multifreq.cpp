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
// Helper: Wrap a single-frequency arrayant into a 1-element vector
// ================================================================================================
template <typename dtype>
static std::vector<quadriga_lib::arrayant<dtype>> wrap_single(const quadriga_lib::arrayant<dtype> &ant)
{
    return {ant};
}

// ================================================================================================
// SECTION 1: Input validation
// ================================================================================================

TEST_CASE("Multifreq - Empty TX array throws")
{
    std::vector<quadriga_lib::arrayant<double>> tx, rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {1.0};
    arma::cube M(8, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr, ci, dl;
    REQUIRE_THROWS_AS(
        quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl),
        std::invalid_argument);
}

TEST_CASE("Multifreq - Empty RX array throws")
{
    std::vector<quadriga_lib::arrayant<double>> rx;
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {1.0};
    arma::cube M(8, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr, ci, dl;
    REQUIRE_THROWS_AS(
        quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl),
        std::invalid_argument);
}

TEST_CASE("Multifreq - Empty freq_in throws")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {1.0};
    arma::cube M(8, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi, fo = {1e9}; // fi empty
    std::vector<arma::cube> cr, ci, dl;
    REQUIRE_THROWS_AS(
        quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl),
        std::invalid_argument);
}

TEST_CASE("Multifreq - Empty freq_out throws")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {1.0};
    arma::cube M(8, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9}, fo; // fo empty
    std::vector<arma::cube> cr, ci, dl;
    REQUIRE_THROWS_AS(
        quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl),
        std::invalid_argument);
}

TEST_CASE("Multifreq - Mismatched path_gain size throws")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 2, arma::fill::zeros), lbs(3, 2, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    fbs(0, 1) = 10.0; lbs(0, 1) = 10.0;
    arma::mat pg(3, 1, arma::fill::ones); // Wrong: 3 rows but only 2 paths
    arma::vec pl = {10.0, 10.0};
    arma::cube M(8, 2, 1, arma::fill::zeros); M(0, 0, 0) = 1.0; M(0, 1, 0) = 1.0;
    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr, ci, dl;
    REQUIRE_THROWS_AS(
        quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl),
        std::invalid_argument);
}

TEST_CASE("Multifreq - M with wrong row count throws")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};
    arma::cube M(4, 1, 1, arma::fill::zeros); // Wrong: neither 8 nor 2
    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr, ci, dl;
    REQUIRE_THROWS_AS(
        quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl),
        std::invalid_argument);
}

TEST_CASE("Multifreq - Non-positive propagation speed throws")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};
    arma::cube M(8, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr, ci, dl;
    REQUIRE_THROWS_AS(
        quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl,
                                              false, false, -1.0),
        std::invalid_argument);
}

// ================================================================================================
// SECTION 2: Output sizing
// ================================================================================================

TEST_CASE("Multifreq - Output vector and cube dimensions")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 3, arma::fill::zeros), lbs(3, 3, arma::fill::zeros);
    fbs(0, 0) = 10.0; fbs(0, 1) = 10.0; fbs(0, 2) = 10.0;
    lbs(0, 0) = 10.0; lbs(0, 1) = 10.0; lbs(0, 2) = 10.0;
    arma::mat pg(3, 2, arma::fill::ones);
    arma::vec pl = {10.0, 10.0, 10.0};
    arma::cube M(8, 3, 2, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(0, 1, 0) = 1.0; M(0, 2, 0) = 1.0;
    M(0, 0, 1) = 1.0; M(0, 1, 1) = 1.0; M(0, 2, 1) = 1.0;
    arma::vec fi = {1e9, 2e9}, fo = {1e9, 1.5e9, 2e9, 3e9};
    std::vector<arma::cube> cr, ci, dl;

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl);

    REQUIRE(cr.size() == 4); // n_freq_out = 4
    REQUIRE(ci.size() == 4);
    REQUIRE(dl.size() == 4);
    for (int f = 0; f < 4; ++f)
    {
        CHECK(cr[f].n_rows == 1);    // n_rx_ports = 1
        CHECK(cr[f].n_cols == 1);    // n_tx_ports = 1
        CHECK(cr[f].n_slices == 3);  // n_path = 3
    }
}

TEST_CASE("Multifreq - Fake LOS adds extra path slice")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 5.0; fbs(1, 0) = 5.0; // NLOS path
    lbs(0, 0) = 5.0; lbs(1, 0) = 5.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {15.0};
    arma::cube M(8, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr, ci, dl;

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl,
                                          false, true); // add_fake_los_path = true

    REQUIRE(cr[0].n_slices == 2); // 1 path + 1 fake LOS = 2
}

// ================================================================================================
// SECTION 3: Match single-frequency get_channels_spherical
// ================================================================================================

TEST_CASE("Multifreq - Matches get_channels_spherical for single freq, single entry")
{
    double C = 299792458.0;
    double fc = 2997924580.0; // 10 * C (wavelength = 0.1 m)

    // Create the antenna
    auto ant_single = quadriga_lib::generate_arrayant_omni<double>();
    ant_single.copy_element(0, 1);
    ant_single.element_pos(1, 0) = 1.0;
    ant_single.element_pos(1, 1) = -1.0;
    ant_single.e_theta_re.slice(1) *= 2;

    auto ant_vec = wrap_single(ant_single);

    arma::mat fbs(3, 2, arma::fill::zeros), lbs(3, 2, arma::fill::zeros);
    fbs(0, 0) = 10.0; fbs(2, 0) = 1.0;   // NLOS scatterer at (10, 0, 2)
    fbs(0, 1) = 20.0; fbs(2, 1) = 1.0;   // LOS at RX position
    lbs = fbs;

    arma::vec pg_vec = {1.0, 0.25}, pl_vec = {0.0, 0.0};
    arma::mat pg_mat(2, 1);
    pg_mat.col(0) = pg_vec;

    arma::mat M_single(8, 2, arma::fill::zeros);
    M_single(0, 0) = 1.0; M_single(6, 0) = -1.0;
    M_single(0, 1) = 1.0; M_single(6, 1) = -1.0;
    arma::cube M_multi(8, 2, 1);
    M_multi.slice(0) = M_single;

    arma::vec fi = {fc}, fo = {fc};

    // Single-frequency reference
    arma::cube cr_ref, ci_ref, dl_ref;
    quadriga_lib::get_channels_spherical(&ant_single, &ant_single,
                                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          &fbs, &fbs, &pg_vec, &pl_vec, &M_single,
                                          &cr_ref, &ci_ref, &dl_ref, fc, true);

    // Multi-frequency version
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(ant_vec, ant_vec,
                                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          fbs, fbs, pg_mat, pl_vec, M_multi,
                                          fi, fo, cr, ci, dl, true);

    REQUIRE(cr.size() == 1);
    CHECK(arma::approx_equal(cr[0], cr_ref, "absdiff", 1e-10));
    CHECK(arma::approx_equal(ci[0], ci_ref, "absdiff", 1e-10));
    CHECK(arma::approx_equal(dl[0], dl_ref, "absdiff", 1e-14));
}

// ================================================================================================
// SECTION 4: Delay calculation with custom propagation speed
// ================================================================================================

TEST_CASE("Multifreq - Speed of sound produces correct acoustic delays")
{
    double c_sound = 343.0; // m/s

    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());

    // LOS path: TX at origin, RX at (5, 0, 0)
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 5.0; lbs(0, 0) = 5.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {5.0};
    arma::cube M(2, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0; // Scalar pressure
    arma::vec fi = {1000.0}, fo = {1000.0};
    std::vector<arma::cube> cr, ci, dl;

    // Absolute delays
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl,
                                          true, false, c_sound);

    double expected_delay = 5.0 / c_sound; // ~14.577 ms
    CHECK(std::abs(dl[0](0, 0, 0) - expected_delay) < 1e-12);

    // Relative delays (LOS delay subtracted)
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl,
                                          false, false, c_sound);

    CHECK(std::abs(dl[0](0, 0, 0)) < 1e-14); // LOS relative delay = 0
}

TEST_CASE("Multifreq - Radio vs acoustic delays differ by speed ratio")
{
    double c_radio = 299792458.0;
    double c_sound = 343.0;
    double dist = 10.0;

    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = dist; lbs(0, 0) = dist;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {dist};
    arma::cube M(2, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr_r, ci_r, dl_r, cr_a, ci_a, dl_a;

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          dist, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr_r, ci_r, dl_r,
                                          true, false, c_radio);

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          dist, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr_a, ci_a, dl_a,
                                          true, false, c_sound);

    double ratio = dl_a[0](0, 0, 0) / dl_r[0](0, 0, 0);
    CHECK(std::abs(ratio - c_radio / c_sound) < 1e-6);
}

// ================================================================================================
// SECTION 5: Scalar pressure Jones matrix (M with 2 rows)
// ================================================================================================

TEST_CASE("Multifreq - Scalar M (2 rows) produces same VV result as full M (8 rows)")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};

    // Full Jones: only VV is set
    arma::cube M_full(8, 1, 1, arma::fill::zeros);
    M_full(0, 0, 0) = 0.8;  // ReVV
    M_full(1, 0, 0) = 0.3;  // ImVV

    // Scalar: identical VV
    arma::cube M_scalar(2, 1, 1, arma::fill::zeros);
    M_scalar(0, 0, 0) = 0.8;
    M_scalar(1, 0, 0) = 0.3;

    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr_f, ci_f, dl_f, cr_s, ci_s, dl_s;

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M_full, fi, fo, cr_f, ci_f, dl_f);

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M_scalar, fi, fo, cr_s, ci_s, dl_s);

    CHECK(arma::approx_equal(cr_f[0], cr_s[0], "absdiff", 1e-12));
    CHECK(arma::approx_equal(ci_f[0], ci_s[0], "absdiff", 1e-12));
    CHECK(arma::approx_equal(dl_f[0], dl_s[0], "absdiff", 1e-14));
}

// ================================================================================================
// SECTION 6: Frequency interpolation of path_gain and M
// ================================================================================================

TEST_CASE("Multifreq - Path gain linearly interpolated across frequency")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::vec pl = {10.0};

    // Path gain: 1.0 at 1 GHz, 4.0 at 2 GHz
    arma::mat pg = {{1.0, 4.0}};
    arma::cube M(2, 1, 2, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(0, 0, 1) = 1.0;
    arma::vec fi = {1e9, 2e9};

    // Query at midpoint → path_gain should be 2.5 → amplitude = sqrt(2.5)
    arma::vec fo = {1.5e9};
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl,
                                          true, false, 299792458.0);

    double amp = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    double expected_amp = std::sqrt(2.5); // Linear interpolation of gain, then sqrt
    CHECK(std::abs(amp - expected_amp) < 0.01);
}

TEST_CASE("Multifreq - M interpolated with SLERP (phase rotation)")
{
    // M rotates from (1,0) to (0,1) across frequency → 90° phase shift
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::vec pl = {10.0};
    arma::mat pg(1, 2, arma::fill::ones);

    arma::cube M(2, 1, 2, arma::fill::zeros);
    M(0, 0, 0) = 1.0; // ReVV = 1 at freq_in[0]
    M(1, 0, 1) = 1.0; // ImVV = 1 at freq_in[1] (90° phase)

    arma::vec fi = {1e9, 2e9}, fo = {1.5e9}; // midpoint
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl,
                                          true, false, 299792458.0);

    // SLERP at midpoint: phase = 45° → both re and im components should be similar magnitude
    double amp = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    CHECK(amp > 0.5); // Should be ~1.0 (amplitude preserved by slerp)
    CHECK(amp < 1.5);
}

// ================================================================================================
// SECTION 7: Extrapolation (clamping)
// ================================================================================================

TEST_CASE("Multifreq - Extrapolation clamps path gain and M")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::vec pl = {10.0};
    arma::mat pg = {{1.0, 4.0}};
    arma::cube M(2, 1, 2, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(0, 0, 1) = 1.0;
    arma::vec fi = {1e9, 2e9};

    // Query below range → clamp to first
    arma::vec fo_lo = {100.0};
    std::vector<arma::cube> cr_lo, ci_lo, dl_lo;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo_lo, cr_lo, ci_lo, dl_lo);

    // Query at exact first entry
    arma::vec fo_ex = {1e9};
    std::vector<arma::cube> cr_ex, ci_ex, dl_ex;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo_ex, cr_ex, ci_ex, dl_ex);

    // Clamped below should use gain=1.0 (first entry), same as exact first entry
    // Note: coefficients may differ slightly due to different wave number, but amplitude should match
    double amp_lo = std::sqrt(cr_lo[0](0, 0, 0) * cr_lo[0](0, 0, 0) + ci_lo[0](0, 0, 0) * ci_lo[0](0, 0, 0));
    double amp_ex = std::sqrt(cr_ex[0](0, 0, 0) * cr_ex[0](0, 0, 0) + ci_ex[0](0, 0, 0) * ci_ex[0](0, 0, 0));
    CHECK(std::abs(amp_lo - amp_ex) < 1e-10);
}

// ================================================================================================
// SECTION 8: Multiple output frequencies
// ================================================================================================

TEST_CASE("Multifreq - Multiple output frequencies produce correct count and finite values")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};
    arma::cube M(2, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9};

    arma::vec fo = arma::linspace(500e6, 3e9, 20);
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl);

    REQUIRE(cr.size() == 20);
    for (arma::uword f = 0; f < 20; ++f)
    {
        CHECK(cr[f].is_finite());
        CHECK(ci[f].is_finite());
        CHECK(dl[f].is_finite());
    }
}

// ================================================================================================
// SECTION 9: EM antenna GHz domain
// ================================================================================================

TEST_CASE("Multifreq - GHz antenna with frequency-dependent pattern")
{
    // Build multi-frequency antenna (pattern changes with frequency)
    arma::vec ghz_freqs = {1.0e9, 2.0e9, 3.0e9};
    std::vector<quadriga_lib::arrayant<double>> tx_vec;
    for (arma::uword i = 0; i < ghz_freqs.n_elem; ++i)
    {
        auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0);
        double scale = 1.0 + 0.5 * (double)i;
        ant.e_theta_re *= scale;
        ant.center_frequency = ghz_freqs[i];
        tx_vec.push_back(ant);
    }
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());

    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};
    arma::cube M(8, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0; M(6, 0, 0) = -1.0;
    arma::vec fi = {1e9}, fo = {1e9, 2e9, 3e9};
    std::vector<arma::cube> cr, ci, dl;

    quadriga_lib::get_channels_multifreq(tx_vec, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl);

    REQUIRE(cr.size() == 3);
    // Amplitude should increase with frequency (pattern scales up)
    double a0 = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    double a1 = std::sqrt(cr[1](0, 0, 0) * cr[1](0, 0, 0) + ci[1](0, 0, 0) * ci[1](0, 0, 0));
    double a2 = std::sqrt(cr[2](0, 0, 0) * cr[2](0, 0, 0) + ci[2](0, 0, 0) * ci[2](0, 0, 0));
    CHECK(a1 > a0);
    CHECK(a2 > a1);
}

// ================================================================================================
// SECTION 10: Acoustic speaker simulation
// ================================================================================================

TEST_CASE("Multifreq - Acoustic 2-way speaker with omni microphone (documentation example)")
{
    // Build a 2-way speaker as TX (source)
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
    auto tx_woofer = quadriga_lib::generate_speaker<double>(
        "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
    auto tx_tweeter = quadriga_lib::generate_speaker<double>(
        "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
    auto tx = quadriga_lib::arrayant_concat_multi(tx_woofer, tx_tweeter);

    // Omnidirectional microphone as RX (single entry, clamped for all frequencies)
    std::vector<quadriga_lib::arrayant<double>> rx = {quadriga_lib::generate_arrayant_omni<double>()};

    // Simple LOS path setup
    arma::mat fbs = arma::mat({0.5, 0.0, 0.0}).t();
    arma::mat lbs = arma::mat({0.5, 0.0, 0.0}).t();
    arma::vec path_length = {1.0};

    // Frequency-flat path gain and scalar Jones matrix
    arma::vec freq_in = {100.0, 10000.0};
    arma::mat path_gain_mat(1, 2, arma::fill::ones);
    arma::cube M_cube(2, 1, 2, arma::fill::zeros);
    M_cube(0, 0, 0) = 1.0; M_cube(0, 0, 1) = 1.0;

    arma::vec freq_out = {200.0, 1000.0, 5000.0};
    std::vector<arma::cube> coeff_re, coeff_im, delays;

    quadriga_lib::get_channels_multifreq(tx, rx,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, path_gain_mat, path_length, M_cube,
                                          freq_in, freq_out, coeff_re, coeff_im, delays,
                                          false, false, 343.0);

    REQUIRE(coeff_re.size() == 3);
    // TX has 2 elements (woofer + tweeter), RX has 1 element → [1, 2, 1]
    CHECK(coeff_re[0].n_rows == 1);
    CHECK(coeff_re[0].n_cols == 2);
    CHECK(coeff_re[0].n_slices == 1);

    // All outputs should be finite
    for (int f = 0; f < 3; ++f)
    {
        CHECK(coeff_re[f].is_finite());
        CHECK(!coeff_re[f].has_nan());
        CHECK(delays[f].is_finite());
    }

    // Delays should all be zero (relative, LOS only)
    for (int f = 0; f < 3; ++f)
        CHECK(std::abs(delays[f](0, 0, 0)) < 1e-10);
}

TEST_CASE("Multifreq - Acoustic woofer dominates at low freq, tweeter at high freq")
{
    arma::vec freqs = {100.0, 500.0, 1000.0, 5000.0, 10000.0};
    auto tx_woofer = quadriga_lib::generate_speaker<double>(
        "piston", 0.083, 50.0, 3000.0, 12.0, 24.0, 87.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
    auto tx_tweeter = quadriga_lib::generate_speaker<double>(
        "piston", 0.013, 1500.0, 20000.0, 24.0, 12.0, 90.0, "hemisphere",
        0.0, 0.0, 0.0, 0.20, 0.30, freqs, 10.0);
    auto tx = quadriga_lib::arrayant_concat_multi(tx_woofer, tx_tweeter);
    std::vector<quadriga_lib::arrayant<double>> rx = {quadriga_lib::generate_arrayant_omni<double>()};

    arma::mat fbs = arma::mat({2.0, 0.0, 0.0}).t();
    arma::mat lbs = arma::mat({2.0, 0.0, 0.0}).t();
    arma::vec pl = {2.0};
    arma::vec fi = {100.0, 10000.0};
    arma::mat pg(1, 2, arma::fill::ones);
    arma::cube M(2, 1, 2, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(0, 0, 1) = 1.0;

    // Query at low and high frequency
    arma::vec fo = {200.0, 8000.0};
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl,
                                          true, false, 343.0);

    // At 200 Hz: woofer (col 0) should dominate over tweeter (col 1)
    double woof_200 = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    double tweet_200 = std::sqrt(cr[0](0, 1, 0) * cr[0](0, 1, 0) + ci[0](0, 1, 0) * ci[0](0, 1, 0));
    CHECK(woof_200 > tweet_200 * 3.0);

    // At 8000 Hz: tweeter (col 1) should dominate over woofer (col 0)
    double woof_8k = std::sqrt(cr[1](0, 0, 0) * cr[1](0, 0, 0) + ci[1](0, 0, 0) * ci[1](0, 0, 0));
    double tweet_8k = std::sqrt(cr[1](0, 1, 0) * cr[1](0, 1, 0) + ci[1](0, 1, 0) * ci[1](0, 1, 0));
    CHECK(tweet_8k > woof_8k * 2.0);
}

// ================================================================================================
// SECTION 11: Full polarimetric Jones matrix
// ================================================================================================

TEST_CASE("Multifreq - Full polarimetric Jones with cross-pol coupling")
{
    // Create a cross-pol probe as RX (has both V and H response)
    auto probe = quadriga_lib::generate_arrayant_xpol<double>();
    auto tx_ant = quadriga_lib::generate_arrayant_omni<double>();
    auto tx = wrap_single(tx_ant);
    auto rx = wrap_single(probe);

    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};

    // Jones with VH cross-pol term
    arma::cube M(8, 1, 1, arma::fill::zeros);
    M(0, 0, 0) = 1.0;  // ReVV
    M(2, 0, 0) = 0.5;  // ReVH (cross-pol)
    M(6, 0, 0) = -1.0; // ReHH

    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr, ci, dl;

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl);

    // RX probe has 2 elements (V+H) → output should be [2, 1, 1]
    REQUIRE(cr[0].n_rows == 2);
    REQUIRE(cr[0].n_cols == 1);

    // Both rows should have non-zero response due to cross-pol
    double amp_v = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    double amp_h = std::sqrt(cr[0](1, 0, 0) * cr[0](1, 0, 0) + ci[0](1, 0, 0) * ci[0](1, 0, 0));
    CHECK(amp_v > 0.1);
    CHECK(amp_h > 0.1);
}

// ================================================================================================
// SECTION 12: TX and RX orientation
// ================================================================================================

TEST_CASE("Multifreq - TX orientation rotates pattern")
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    ant.copy_element(0, 1);
    ant.element_pos(1, 0) = 15.0;
    ant.element_pos(1, 1) = -15.0;
    ant.e_theta_re.slice(1) *= 2;

    auto probe = quadriga_lib::generate_arrayant_xpol<double>();
    auto tx = wrap_single(ant);
    auto rx = wrap_single(probe);

    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; fbs(2, 0) = 1.0;
    lbs = fbs;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl(1, arma::fill::zeros);
    arma::cube M(8, 1, 1, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(6, 0, 0) = -1.0;
    arma::vec fi = {2997924580.0}, fo = {2997924580.0};

    double pi = arma::datum::pi;

    // With -90° bank rotation
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx,
                                          0.0, 0.0, 1.0, -pi / 2.0, 0.0, 0.0,
                                          20.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl, false);

    REQUIRE(cr.size() == 1);
    CHECK(cr[0].n_rows == 2);    // xpol probe has 2 elements
    CHECK(cr[0].n_cols == 2);    // TX has 2 elements
    CHECK(cr[0].is_finite());
}

// ================================================================================================
// SECTION 13: Frequency-dependent phase from wave number
// ================================================================================================

TEST_CASE("Multifreq - Different frequencies produce different phases")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};
    arma::cube M(2, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9};

    // Two different frequencies
    arma::vec fo = {1.0e9, 1.5e9};
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl, true);

    // Amplitudes should be the same (same gain, same antenna)
    double a0 = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    double a1 = std::sqrt(cr[1](0, 0, 0) * cr[1](0, 0, 0) + ci[1](0, 0, 0) * ci[1](0, 0, 0));
    CHECK(std::abs(a0 - a1) < 1e-10);

    // Phases should differ (different wave number)
    double p0 = std::atan2(-ci[0](0, 0, 0), cr[0](0, 0, 0));
    double p1 = std::atan2(-ci[1](0, 0, 0), cr[1](0, 0, 0));
    CHECK(std::abs(p0 - p1) > 1e-6);

    // Delays should be the same (geometry doesn't change)
    CHECK(std::abs(dl[0](0, 0, 0) - dl[1](0, 0, 0)) < 1e-14);
}

// ================================================================================================
// SECTION 14: NLOS paths
// ================================================================================================

TEST_CASE("Multifreq - LOS and NLOS paths have different delays")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    double C = 299792458.0;

    // Path 0: LOS (FBS = LBS = RX position)
    // Path 1: NLOS via scatterer at (5, 5, 0)
    arma::mat fbs(3, 2, arma::fill::zeros), lbs(3, 2, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;        // LOS path
    fbs(0, 1) = 5.0;  fbs(1, 1) = 5.0;         // NLOS FBS
    lbs(0, 1) = 5.0;  lbs(1, 1) = 5.0;         // NLOS LBS (same as FBS = single bounce)

    double d_nlos = std::sqrt(25.0 + 25.0) + std::sqrt(25.0 + 25.0); // TX→FBS + LBS→RX
    arma::vec pl = {10.0, d_nlos};
    arma::mat pg(2, 1, arma::fill::ones);
    arma::cube M(8, 2, 1, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(0, 1, 0) = 1.0;
    M(6, 0, 0) = -1.0; M(6, 1, 0) = -1.0;
    arma::vec fi = {1e9}, fo = {1e9};

    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl, true);

    double delay_los = dl[0](0, 0, 0);
    double delay_nlos = dl[0](0, 0, 1);
    CHECK(delay_los < delay_nlos); // LOS should be shorter
    CHECK(std::abs(delay_los - 10.0 / C) < 1e-12);
    CHECK(std::abs(delay_nlos - d_nlos / C) < 1e-10);
}

// ================================================================================================
// SECTION 15: Multi-element MIMO antenna
// ================================================================================================

TEST_CASE("Multifreq - 2x2 MIMO output dimensions and element scaling")
{
    auto tx_ant = quadriga_lib::generate_arrayant_omni<double>();
    tx_ant.copy_element(0, 1);
    tx_ant.element_pos(1, 0) = 0.5;
    tx_ant.element_pos(1, 1) = -0.5;
    tx_ant.e_theta_re.slice(1) *= 2.0;

    auto rx_ant = quadriga_lib::generate_arrayant_omni<double>();
    rx_ant.copy_element(0, 1);

    auto tx = wrap_single(tx_ant);
    auto rx = wrap_single(rx_ant);

    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 20.0; lbs(0, 0) = 20.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {20.0};
    arma::cube M(8, 1, 1, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(6, 0, 0) = -1.0;
    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr, ci, dl;

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          20.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl);

    REQUIRE(cr[0].n_rows == 2);  // 2 RX elements
    REQUIRE(cr[0].n_cols == 2);  // 2 TX elements
    REQUIRE(cr[0].n_slices == 1);

    // TX element 1 has 2x gain → its column should have ~2x amplitude
    double a00 = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    double a01 = std::sqrt(cr[0](0, 1, 0) * cr[0](0, 1, 0) + ci[0](0, 1, 0) * ci[0](0, 1, 0));
    CHECK(std::abs(a01 / a00 - 2.0) < 0.1);
}

// ================================================================================================
// SECTION 16: Float template
// ================================================================================================

TEST_CASE("Multifreq - Float template compiles and runs")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<float>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<float>());
    arma::fmat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0f; lbs(0, 0) = 10.0f;
    arma::fmat pg(1, 1, arma::fill::ones);
    arma::fvec pl = {10.0f};
    arma::fcube M(2, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0f;
    arma::fvec fi = {1e9f}, fo = {1e9f, 2e9f};
    std::vector<arma::fcube> cr, ci, dl;

    REQUIRE_NOTHROW(
        quadriga_lib::get_channels_multifreq(tx, rx, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                              10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                              fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl));
    REQUIRE(cr.size() == 2);
    CHECK(cr[0].is_finite());
    CHECK(cr[1].is_finite());
}

// ================================================================================================
// SECTION 17: Zero center_frequency (phase disabled)
// ================================================================================================

TEST_CASE("Multifreq - Zero output frequency disables phase rotation")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};
    arma::cube M(2, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;

    // freq_out = 0 should disable phase (wave_number = 0)
    arma::vec fi = {1e9}, fo = {0.0};
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl, true);

    // With zero wave number: phase = 0, so coeff_im should be zero
    CHECK(std::abs(ci[0](0, 0, 0)) < 1e-14);
    // coeff_re should be +1.0 (unit gain, no phase rotation)
    CHECK(std::abs(cr[0](0, 0, 0) - 1.0) < 1e-10);
}

// ================================================================================================
// SECTION 18: Multiple freq_in entries
// ================================================================================================

TEST_CASE("Multifreq - Multiple freq_in entries with varying path gain")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::vec pl = {10.0};

    // 3 freq_in: gain increases linearly
    arma::vec fi = {1e9, 2e9, 3e9};
    arma::mat pg = {{1.0, 4.0, 9.0}};
    arma::cube M(2, 1, 3, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(0, 0, 1) = 1.0; M(0, 0, 2) = 1.0;

    arma::vec fo = {1e9, 2e9, 3e9};
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl, true);

    // At exact freq_in entries: amplitude = sqrt(gain)
    double a0 = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    double a1 = std::sqrt(cr[1](0, 0, 0) * cr[1](0, 0, 0) + ci[1](0, 0, 0) * ci[1](0, 0, 0));
    double a2 = std::sqrt(cr[2](0, 0, 0) * cr[2](0, 0, 0) + ci[2](0, 0, 0) * ci[2](0, 0, 0));
    CHECK(std::abs(a0 - 1.0) < 0.01);
    CHECK(std::abs(a1 - 2.0) < 0.01);
    CHECK(std::abs(a2 - 3.0) < 0.01);
}

// ================================================================================================
// SECTION 19: Absolute vs relative delays
// ================================================================================================

TEST_CASE("Multifreq - Absolute delay includes LOS, relative subtracts it")
{
    double C = 299792458.0;
    double dist = 20.0;

    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = dist; lbs(0, 0) = dist;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {dist};
    arma::cube M(2, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9}, fo = {1e9};
    std::vector<arma::cube> cr_a, ci_a, dl_a, cr_r, ci_r, dl_r;

    // Absolute
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          dist, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr_a, ci_a, dl_a, true);
    // Relative
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          dist, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr_r, ci_r, dl_r, false);

    CHECK(std::abs(dl_a[0](0, 0, 0) - dist / C) < 1e-12);
    CHECK(std::abs(dl_r[0](0, 0, 0)) < 1e-14); // LOS relative = 0
}

// ================================================================================================
// SECTION 20: Consistency across freq_out (amplitude stable, phase varies)
// ================================================================================================

TEST_CASE("Multifreq - Sweep: amplitude stable, phase varies with frequency")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};
    arma::cube M(2, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9};

    arma::vec fo = arma::linspace(1e9, 3e9, 10);
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl, true);

    // All amplitudes should be identical (same antenna, same gain)
    double amp_ref = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    for (arma::uword f = 1; f < fo.n_elem; ++f)
    {
        double amp = std::sqrt(cr[f](0, 0, 0) * cr[f](0, 0, 0) + ci[f](0, 0, 0) * ci[f](0, 0, 0));
        CHECK(std::abs(amp - amp_ref) < 1e-10);
    }
}

// ================================================================================================
// SECTION 21: Output memory reuse (pre-allocated)
// ================================================================================================

TEST_CASE("Multifreq - Pre-allocated output vectors are reused")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};
    arma::cube M(2, 1, 1, arma::fill::zeros); M(0, 0, 0) = 1.0;
    arma::vec fi = {1e9}, fo = {1e9, 2e9};

    // Pre-allocate with correct sizes
    std::vector<arma::cube> cr(2), ci(2), dl(2);
    for (int f = 0; f < 2; ++f)
    {
        cr[f].set_size(1, 1, 1);
        ci[f].set_size(1, 1, 1);
        dl[f].set_size(1, 1, 1);
    }

    // Remember pointers
    double *p0 = cr[0].memptr();
    double *p1 = cr[1].memptr();

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl);

    // Memory should not have been reallocated (same pointers)
    CHECK(cr[0].memptr() == p0);
    CHECK(cr[1].memptr() == p1);
}

// ================================================================================================
// SECTION 22: Acoustic with single-bounce reflection
// ================================================================================================

TEST_CASE("Multifreq - Acoustic single-bounce produces NLOS delay > LOS")
{
    double c_sound = 343.0;
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());

    // TX at origin, RX at (3, 0, 0), scatterer on wall at (1.5, 2, 0)
    arma::mat fbs(3, 2, arma::fill::zeros), lbs(3, 2, arma::fill::zeros);
    fbs(0, 0) = 3.0; lbs(0, 0) = 3.0;                          // LOS
    fbs(0, 1) = 1.5; fbs(1, 1) = 2.0; lbs(0, 1) = 1.5; lbs(1, 1) = 2.0; // Wall reflection

    double d_los = 3.0;
    double d_nlos = std::sqrt(1.5 * 1.5 + 4.0) + std::sqrt(1.5 * 1.5 + 4.0);
    arma::vec pl = {d_los, d_nlos};
    arma::mat pg(2, 1);
    pg(0, 0) = 1.0; pg(1, 0) = 0.5;
    arma::cube M(2, 2, 1, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(0, 1, 0) = 1.0;
    arma::vec fi = {500.0}, fo = {500.0};
    std::vector<arma::cube> cr, ci, dl;

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl,
                                          true, false, c_sound);

    CHECK(dl[0](0, 0, 0) < dl[0](0, 0, 1));
    CHECK(std::abs(dl[0](0, 0, 0) - d_los / c_sound) < 1e-10);
    CHECK(std::abs(dl[0](0, 0, 1) - d_nlos / c_sound) < 1e-10);
}

// ================================================================================================
// SECTION 23: Frequency-dependent path gain with acoustic simulation
// ================================================================================================

TEST_CASE("Multifreq - Air absorption increases with frequency (acoustic)")
{
    double c_sound = 343.0;
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 5.0; lbs(0, 0) = 5.0;
    arma::vec pl = {5.0};

    // Simulate air absorption: higher frequency → lower gain
    arma::vec fi = {200.0, 1000.0, 5000.0, 10000.0};
    arma::mat pg(1, 4);
    pg(0, 0) = 1.0; pg(0, 1) = 0.8; pg(0, 2) = 0.4; pg(0, 3) = 0.1;
    arma::cube M(2, 1, 4, arma::fill::zeros);
    for (arma::uword s = 0; s < 4; ++s) M(0, 0, s) = 1.0;

    arma::vec fo = {200.0, 5000.0, 10000.0};
    std::vector<arma::cube> cr, ci, dl;
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl,
                                          true, false, c_sound);

    // Amplitude should decrease with frequency
    double a_lo = std::sqrt(cr[0](0, 0, 0) * cr[0](0, 0, 0) + ci[0](0, 0, 0) * ci[0](0, 0, 0));
    double a_mid = std::sqrt(cr[1](0, 0, 0) * cr[1](0, 0, 0) + ci[1](0, 0, 0) * ci[1](0, 0, 0));
    double a_hi = std::sqrt(cr[2](0, 0, 0) * cr[2](0, 0, 0) + ci[2](0, 0, 0) * ci[2](0, 0, 0));
    CHECK(a_lo > a_mid);
    CHECK(a_mid > a_hi);
}

// ================================================================================================
// SECTION 24: Coupling interpolation across frequency
// ================================================================================================

TEST_CASE("Multifreq - Frequency-dependent coupling produces different coefficients")
{
    // Build 2-element antenna with different coupling at two frequencies
    std::vector<quadriga_lib::arrayant<double>> tx_vec;
    for (int i = 0; i < 2; ++i)
    {
        auto ant = quadriga_lib::generate_arrayant_omni<double>();
        ant.copy_element(0, 1);
        ant.center_frequency = (i == 0) ? 1e9 : 2e9;

        // Coupling: mix elements differently at each frequency
        // At 1 GHz: port 0 = elem0 + 0.5*elem1, port 1 = 0.5*elem0 + elem1
        // At 2 GHz: port 0 = elem0, port 1 = elem1 (identity)
        if (i == 0)
        {
            ant.coupling_re = {{1.0, 0.5}, {0.5, 1.0}};
            ant.coupling_im.zeros(2, 2);
        }
        // i==1: no coupling set → identity
        tx_vec.push_back(ant);
    }
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());

    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg(1, 1, arma::fill::ones);
    arma::vec pl = {10.0};
    arma::cube M(8, 1, 1, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(6, 0, 0) = -1.0;
    arma::vec fi = {1e9}, fo = {1e9, 2e9};
    std::vector<arma::cube> cr, ci, dl;

    quadriga_lib::get_channels_multifreq(tx_vec, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo, cr, ci, dl);

    REQUIRE(cr.size() == 2);
    // At 1 GHz (with coupling), ports should have mixed signals
    // At 2 GHz (identity coupling), ports should equal elements directly
    // The two frequency outputs should differ
    CHECK(!arma::approx_equal(cr[0], cr[1], "absdiff", 1e-6));
}

// ================================================================================================
// SECTION 25: Reversed freq_out order
// ================================================================================================

TEST_CASE("Multifreq - Reversed freq_out order gives consistent results")
{
    auto tx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    auto rx = wrap_single(quadriga_lib::generate_arrayant_omni<double>());
    arma::mat fbs(3, 1, arma::fill::zeros), lbs(3, 1, arma::fill::zeros);
    fbs(0, 0) = 10.0; lbs(0, 0) = 10.0;
    arma::mat pg = {{1.0, 4.0}};
    arma::vec pl = {10.0};
    arma::cube M(2, 1, 2, arma::fill::zeros);
    M(0, 0, 0) = 1.0; M(0, 0, 1) = 1.0;
    arma::vec fi = {1e9, 2e9};

    arma::vec fo_fwd = {1.2e9, 1.8e9};
    arma::vec fo_rev = {1.8e9, 1.2e9};
    std::vector<arma::cube> cr_f, ci_f, dl_f, cr_r, ci_r, dl_r;

    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo_fwd, cr_f, ci_f, dl_f, true);
    quadriga_lib::get_channels_multifreq(tx, rx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          fbs, lbs, pg, pl, M, fi, fo_rev, cr_r, ci_r, dl_r, true);

    // Forward[0] = Reverse[1] and vice versa
    CHECK(arma::approx_equal(cr_f[0], cr_r[1], "absdiff", 1e-12));
    CHECK(arma::approx_equal(cr_f[1], cr_r[0], "absdiff", 1e-12));
    CHECK(arma::approx_equal(dl_f[0], dl_r[1], "absdiff", 1e-14));
    CHECK(arma::approx_equal(dl_f[1], dl_r[0], "absdiff", 1e-14));
}
