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
#include "quadriga_channel.hpp"

#include <cmath>
#include <vector>

TEST_CASE("quantize_delays - Input validation")
{
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(2, 2, 3);
    cre[0].randn();
    cim[0].set_size(2, 2, 3);
    cim[0].randn();
    dl[0].set_size(2, 2, 3);
    dl[0].randu();
    dl[0] *= 100e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;

    // Null input pointers
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(nullptr, &cim, &dl, &cre_q, &cim_q, &dl_q),
                    std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, nullptr, &dl, &cre_q, &cim_q, &dl_q),
                    std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim, nullptr, &cre_q, &cim_q, &dl_q),
                    std::invalid_argument);

    // Null output pointers
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim, &dl, nullptr, &cim_q, &dl_q),
                    std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim, &dl, &cre_q, nullptr, &dl_q),
                    std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim, &dl, &cre_q, &cim_q, nullptr),
                    std::invalid_argument);

    // Empty input
    std::vector<arma::Cube<double>> empty_vec;
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&empty_vec, &cim, &dl, &cre_q, &cim_q, &dl_q),
                    std::invalid_argument);

    // Mismatched vector sizes
    std::vector<arma::Cube<double>> cim_short;
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim_short, &dl, &cre_q, &cim_q, &dl_q),
                    std::invalid_argument);

    // Invalid tap_spacing
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 0.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, -1.0e-9),
                    std::invalid_argument);

    // Invalid power_exponent
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 0.0),
                    std::invalid_argument);

    // Invalid fix_taps
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, -1),
                    std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 4),
                    std::invalid_argument);

    // Mismatched n_rx/n_tx across snapshots
    std::vector<arma::Cube<double>> cre2(2), cim2(2), dl2(2);
    cre2[0].set_size(2, 2, 3); cre2[0].randn();
    cim2[0].set_size(2, 2, 3); cim2[0].randn();
    dl2[0].set_size(2, 2, 3); dl2[0].randu(); dl2[0] *= 100e-9;
    cre2[1].set_size(3, 2, 3); cre2[1].randn(); // Different n_rx
    cim2[1].set_size(3, 2, 3); cim2[1].randn();
    dl2[1].set_size(3, 2, 3); dl2[1].randu(); dl2[1] *= 100e-9;
    CHECK_THROWS_AS(quadriga_lib::quantize_delays<double>(&cre2, &cim2, &dl2, &cre_q, &cim_q, &dl_q),
                    std::invalid_argument);
}

TEST_CASE("quantize_delays - Single path at exact tap boundary")
{
    // A single path whose delay is exactly on a tap boundary should pass through unchanged
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 1);
    cim[0].set_size(1, 1, 1);
    dl[0].set_size(1, 1, 1);

    cre[0](0, 0, 0) = 2.0;
    cim[0](0, 0, 0) = 3.0;
    dl[0](0, 0, 0) = 15.0e-9; // Exactly 3 * 5ns

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(cre_q.size() == 1);
    REQUIRE(cre_q[0].n_rows == 1);
    REQUIRE(cre_q[0].n_cols == 1);
    REQUIRE(cre_q[0].n_slices >= 1);

    // The coefficient should be unchanged at the tap corresponding to 15 ns
    bool found = false;
    for (arma::uword k = 0; k < cre_q[0].n_slices; ++k) {
        if (std::abs(cre_q[0](0, 0, k)) > 1e-10 || std::abs(cim_q[0](0, 0, k)) > 1e-10) {
            CHECK(std::abs(cre_q[0](0, 0, k) - 2.0) < 1e-12);
            CHECK(std::abs(cim_q[0](0, 0, k) - 3.0) < 1e-12);
            CHECK(std::abs(dl_q[0](0, 0, k) - 15.0e-9) < 1e-20);
            found = true;
        }
    }
    REQUIRE(found);
}

TEST_CASE("quantize_delays - Single path at half-tap offset with linear exponent")
{
    // A single path at exactly half a tap: delta = 0.5
    // With alpha=1.0: weights are (1-0.5)^1=0.5 and 0.5^1=0.5
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 1);
    cim[0].set_size(1, 1, 1);
    dl[0].set_size(1, 1, 1);

    cre[0](0, 0, 0) = 4.0;
    cim[0](0, 0, 0) = 0.0;
    dl[0](0, 0, 0) = 12.5e-9; // Half way between 10ns and 15ns

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(cre_q.size() == 1);

    // Expect two taps: one at 10ns, one at 15ns, each with coefficient 0.5 * 4.0 = 2.0
    double sum_re = 0.0;
    int n_nonzero = 0;
    for (arma::uword k = 0; k < cre_q[0].n_slices; ++k) {
        double v = cre_q[0](0, 0, k);
        if (std::abs(v) > 1e-10) {
            CHECK(std::abs(v - 2.0) < 1e-12);
            sum_re += v;
            n_nonzero++;
        }
    }
    CHECK(n_nonzero == 2);
    CHECK(std::abs(sum_re - 4.0) < 1e-12);
}

TEST_CASE("quantize_delays - Single path at half-tap offset with sqrt exponent")
{
    // alpha=0.5: weights are (1-0.5)^0.5 = sqrt(0.5) and 0.5^0.5 = sqrt(0.5)
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 1);
    cim[0].set_size(1, 1, 1);
    dl[0].set_size(1, 1, 1);

    cre[0](0, 0, 0) = 4.0;
    cim[0](0, 0, 0) = 0.0;
    dl[0](0, 0, 0) = 12.5e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 0.5, 0);

    REQUIRE(cre_q.size() == 1);

    double expected_coeff = std::sqrt(0.5) * 4.0;
    double sum_power = 0.0;
    int n_nonzero = 0;
    for (arma::uword k = 0; k < cre_q[0].n_slices; ++k) {
        double v = cre_q[0](0, 0, k);
        if (std::abs(v) > 1e-10) {
            CHECK(std::abs(v - expected_coeff) < 1e-12);
            sum_power += v * v;
            n_nonzero++;
        }
    }
    CHECK(n_nonzero == 2);
    // With alpha=0.5, sum of squared coefficients should equal original squared: 16
    CHECK(std::abs(sum_power - 16.0) < 1e-10);
}

TEST_CASE("quantize_delays - Multiple paths, basic correctness")
{
    // 3 paths: one on grid, one off grid, one far off
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 3);
    cim[0].set_size(1, 1, 3);
    dl[0].set_size(1, 1, 3);

    cre[0](0, 0, 0) = 1.0; cim[0](0, 0, 0) = 0.0; dl[0](0, 0, 0) = 0.0;
    cre[0](0, 0, 1) = 1.0; cim[0](0, 0, 1) = 0.0; dl[0](0, 0, 1) = 10.0e-9;
    cre[0](0, 0, 2) = 1.0; cim[0](0, 0, 2) = 0.0; dl[0](0, 0, 2) = 33.0e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(cre_q.size() == 1);

    // Count non-zero taps
    int n_nonzero = 0;
    for (arma::uword k = 0; k < cre_q[0].n_slices; ++k) {
        if (std::abs(cre_q[0](0, 0, k)) > 1e-10 || std::abs(cim_q[0](0, 0, k)) > 1e-10)
            n_nonzero++;
    }
    // Path 0: 1 tap (on grid), Path 1: 1 tap (on grid), Path 2: 2 taps (interpolated)
    CHECK(n_nonzero == 4);

    // Verify total coherent power (sum of real parts) is preserved for linear interpolation
    double sum_re = 0.0;
    for (arma::uword k = 0; k < cre_q[0].n_slices; ++k)
        sum_re += cre_q[0](0, 0, k);
    CHECK(std::abs(sum_re - 3.0) < 1e-10);
}

TEST_CASE("quantize_delays - Already quantized input passes through")
{
    // All delays are exactly on the grid
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 4);
    cim[0].set_size(1, 1, 4);
    dl[0].set_size(1, 1, 4);

    cre[0](0, 0, 0) = 1.0; cim[0](0, 0, 0) = 0.5; dl[0](0, 0, 0) = 0.0;
    cre[0](0, 0, 1) = 2.0; cim[0](0, 0, 1) = 1.0; dl[0](0, 0, 1) = 5.0e-9;
    cre[0](0, 0, 2) = 0.5; cim[0](0, 0, 2) = 0.3; dl[0](0, 0, 2) = 20.0e-9;
    cre[0](0, 0, 3) = 0.1; cim[0](0, 0, 3) = 0.0; dl[0](0, 0, 3) = 50.0e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(cre_q.size() == 1);
    REQUIRE(cre_q[0].n_slices >= 4);

    for (arma::uword k = 0; k < cre_q[0].n_slices; ++k) {
        double d = dl_q[0](0, 0, k);
        double re = cre_q[0](0, 0, k);
        double im = cim_q[0](0, 0, k);

        if (std::abs(d - 0.0) < 1e-20) {
            CHECK(std::abs(re - 1.0) < 1e-12);
            CHECK(std::abs(im - 0.5) < 1e-12);
        } else if (std::abs(d - 5.0e-9) < 1e-20) {
            CHECK(std::abs(re - 2.0) < 1e-12);
            CHECK(std::abs(im - 1.0) < 1e-12);
        } else if (std::abs(d - 20.0e-9) < 1e-20) {
            CHECK(std::abs(re - 0.5) < 1e-12);
            CHECK(std::abs(im - 0.3) < 1e-12);
        } else if (std::abs(d - 50.0e-9) < 1e-20) {
            CHECK(std::abs(re - 0.1) < 1e-12);
            CHECK(std::abs(im - 0.0) < 1e-12);
        }
    }
}

TEST_CASE("quantize_delays - max_no_taps limits output")
{
    // 5 paths on grid, limit to 3 taps — weakest should be dropped
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 5);
    cim[0].set_size(1, 1, 5);
    dl[0].set_size(1, 1, 5);

    cre[0](0, 0, 0) = 5.0; cim[0](0, 0, 0) = 0.0; dl[0](0, 0, 0) = 0.0;
    cre[0](0, 0, 1) = 1.0; cim[0](0, 0, 1) = 0.0; dl[0](0, 0, 1) = 5.0e-9;
    cre[0](0, 0, 2) = 3.0; cim[0](0, 0, 2) = 0.0; dl[0](0, 0, 2) = 10.0e-9;
    cre[0](0, 0, 3) = 0.5; cim[0](0, 0, 3) = 0.0; dl[0](0, 0, 3) = 15.0e-9;
    cre[0](0, 0, 4) = 4.0; cim[0](0, 0, 4) = 0.0; dl[0](0, 0, 4) = 20.0e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 3, 1.0, 0);

    REQUIRE(cre_q[0].n_slices <= 3);

    int n_nonzero = 0;
    for (arma::uword k = 0; k < cre_q[0].n_slices; ++k) {
        if (std::abs(cre_q[0](0, 0, k)) > 1e-10)
            n_nonzero++;
    }
    CHECK(n_nonzero == 3);
}

TEST_CASE("quantize_delays - Shared delays [1,1,n_path] with fix_taps=2")
{
    std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);

    for (int s = 0; s < 2; ++s) {
        cre[s].set_size(2, 2, 3);
        cre[s].randn();
        cim[s].set_size(2, 2, 3);
        cim[s].randn();
        dl[s].set_size(1, 1, 3);
        dl[s](0, 0, 0) = 0.0;
        dl[s](0, 0, 1) = 12.5e-9;
        dl[s](0, 0, 2) = 30.0e-9;
    }

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 2);

    REQUIRE(dl_q.size() == 2);
    CHECK(dl_q[0].n_rows == 1);
    CHECK(dl_q[0].n_cols == 1);
    CHECK(dl_q[1].n_rows == 1);
    CHECK(dl_q[1].n_cols == 1);
    CHECK(cre_q[0].n_rows == 2);
    CHECK(cre_q[0].n_cols == 2);
}

TEST_CASE("quantize_delays - Shared delays with fix_taps=0 produces per-antenna output")
{
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(2, 2, 2);
    cre[0].randn();
    cim[0].set_size(2, 2, 2);
    cim[0].randn();
    dl[0].set_size(1, 1, 2);
    dl[0](0, 0, 0) = 0.0;
    dl[0](0, 0, 1) = 17.5e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(dl_q.size() == 1);
    CHECK(dl_q[0].n_rows == 2);
    CHECK(dl_q[0].n_cols == 2);
}

TEST_CASE("quantize_delays - Per-antenna delays")
{
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(2, 1, 2);
    cim[0].set_size(2, 1, 2);
    dl[0].set_size(2, 1, 2);

    cre[0](0, 0, 0) = 1.0; cim[0](0, 0, 0) = 0.0; dl[0](0, 0, 0) = 0.0;
    cre[0](0, 0, 1) = 1.0; cim[0](0, 0, 1) = 0.0; dl[0](0, 0, 1) = 12.5e-9;
    cre[0](1, 0, 0) = 1.0; cim[0](1, 0, 0) = 0.0; dl[0](1, 0, 0) = 5.0e-9;
    cre[0](1, 0, 1) = 1.0; cim[0](1, 0, 1) = 0.0; dl[0](1, 0, 1) = 27.5e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(dl_q.size() == 1);
    CHECK(dl_q[0].n_rows == 2);
    CHECK(dl_q[0].n_cols == 1);
}

TEST_CASE("quantize_delays - Multiple snapshots with same n_path")
{
    arma::uword n_snap = 3;
    std::vector<arma::Cube<double>> cre(n_snap), cim(n_snap), dl(n_snap);

    for (arma::uword s = 0; s < n_snap; ++s) {
        cre[s].set_size(1, 1, 2);
        cim[s].set_size(1, 1, 2);
        dl[s].set_size(1, 1, 2);

        cre[s](0, 0, 0) = 1.0; cim[s](0, 0, 0) = 0.0;
        dl[s](0, 0, 0) = 0.0;

        cre[s](0, 0, 1) = 1.0; cim[s](0, 0, 1) = 0.0;
        dl[s](0, 0, 1) = (double)(s + 1) * 7.5e-9;
    }

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(cre_q.size() == n_snap);
    for (arma::uword s = 0; s < n_snap; ++s)
        CHECK(cre_q[s].n_slices >= 1);
}

TEST_CASE("quantize_delays - Variable n_path across snapshots")
{
    // Snapshot 0: 3 paths (one on-grid, one off-grid, one on-grid)
    // Snapshot 1: 1 path (on-grid)
    // Snapshot 2: 2 paths (both off-grid)
    std::vector<arma::Cube<double>> cre(3), cim(3), dl(3);

    cre[0].set_size(1, 1, 3); cim[0].set_size(1, 1, 3); dl[0].set_size(1, 1, 3);
    cre[0](0, 0, 0) = 1.0; cim[0](0, 0, 0) = 0.0; dl[0](0, 0, 0) = 0.0;
    cre[0](0, 0, 1) = 2.0; cim[0](0, 0, 1) = 0.0; dl[0](0, 0, 1) = 12.5e-9;
    cre[0](0, 0, 2) = 0.5; cim[0](0, 0, 2) = 0.0; dl[0](0, 0, 2) = 20.0e-9;

    cre[1].set_size(1, 1, 1); cim[1].set_size(1, 1, 1); dl[1].set_size(1, 1, 1);
    cre[1](0, 0, 0) = 3.0; cim[1](0, 0, 0) = 0.0; dl[1](0, 0, 0) = 5.0e-9;

    cre[2].set_size(1, 1, 2); cim[2].set_size(1, 1, 2); dl[2].set_size(1, 1, 2);
    cre[2](0, 0, 0) = 1.5; cim[2](0, 0, 0) = 0.0; dl[2](0, 0, 0) = 7.5e-9;
    cre[2](0, 0, 1) = 0.8; cim[2](0, 0, 1) = 0.0; dl[2](0, 0, 1) = 17.5e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(cre_q.size() == 3);
    // All output cubes must have same n_rx, n_tx and n_taps (n_taps = max across all grids)
    arma::uword n_taps = cre_q[0].n_slices;
    CHECK(cre_q[1].n_slices == n_taps);
    CHECK(cre_q[2].n_slices == n_taps);
    CHECK(cre_q[0].n_rows == 1);
    CHECK(cre_q[1].n_rows == 1);
    CHECK(cre_q[2].n_rows == 1);

    // Snapshot 1 has only 1 path at tap 1, so most output taps should be zero
    int nz_snap1 = 0;
    for (arma::uword k = 0; k < n_taps; ++k)
        if (std::abs(cre_q[1](0, 0, k)) > 1e-10) nz_snap1++;
    CHECK(nz_snap1 == 1);
    // Verify snapshot 1 coefficient
    bool found = false;
    for (arma::uword k = 0; k < n_taps; ++k) {
        if (std::abs(dl_q[1](0, 0, k) - 5.0e-9) < 1e-20) {
            CHECK(std::abs(cre_q[1](0, 0, k) - 3.0) < 1e-12);
            found = true;
        }
    }
    REQUIRE(found);
}

TEST_CASE("quantize_delays - Zero-path snapshot")
{
    // Snapshot 0 has paths, snapshot 1 has zero paths (e.g. ray tracing found nothing)
    std::vector<arma::Cube<double>> cre(3), cim(3), dl(3);

    cre[0].set_size(1, 1, 2); cim[0].set_size(1, 1, 2); dl[0].set_size(1, 1, 2);
    cre[0](0, 0, 0) = 1.0; cim[0](0, 0, 0) = 0.0; dl[0](0, 0, 0) = 0.0;
    cre[0](0, 0, 1) = 2.0; cim[0](0, 0, 1) = 0.0; dl[0](0, 0, 1) = 10.0e-9;

    cre[1].set_size(1, 1, 0); cim[1].set_size(1, 1, 0); dl[1].set_size(1, 1, 0);

    cre[2].set_size(1, 1, 1); cim[2].set_size(1, 1, 1); dl[2].set_size(1, 1, 1);
    cre[2](0, 0, 0) = 3.0; cim[2](0, 0, 0) = 0.0; dl[2](0, 0, 0) = 5.0e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(cre_q.size() == 3);
    arma::uword n_taps = cre_q[0].n_slices;
    CHECK(cre_q[1].n_slices == n_taps);
    CHECK(cre_q[2].n_slices == n_taps);

    // Snapshot 1 output should be all zeros
    bool all_zero = true;
    for (arma::uword k = 0; k < n_taps; ++k)
        if (cre_q[1](0, 0, k) != 0.0 || cim_q[1](0, 0, k) != 0.0)
            all_zero = false;
    CHECK(all_zero);

    // Snapshot 0 and 2 should have non-zero data
    int nz0 = 0, nz2 = 0;
    for (arma::uword k = 0; k < n_taps; ++k) {
        if (std::abs(cre_q[0](0, 0, k)) > 1e-10) nz0++;
        if (std::abs(cre_q[2](0, 0, k)) > 1e-10) nz2++;
    }
    CHECK(nz0 == 2);
    CHECK(nz2 == 1);
}

TEST_CASE("quantize_delays - All snapshots zero paths")
{
    // Edge case: all snapshots have zero paths
    std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
    cre[0].set_size(2, 1, 0); cim[0].set_size(2, 1, 0); dl[0].set_size(2, 1, 0);
    cre[1].set_size(2, 1, 0); cim[1].set_size(2, 1, 0); dl[1].set_size(2, 1, 0);

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(cre_q.size() == 2);
    CHECK(cre_q[0].n_rows == 2);
    CHECK(cre_q[0].n_cols == 1);
    CHECK(cre_q[0].n_slices >= 1);
    // All output should be zero
    for (arma::uword s = 0; s < 2; ++s)
        for (arma::uword k = 0; k < cre_q[s].n_slices; ++k)
            for (arma::uword a = 0; a < 2; ++a)
                CHECK(cre_q[s](a, 0, k) == 0.0);
}

TEST_CASE("quantize_delays - Variable n_path with fix_taps=1")
{
    // 2 snapshots with different n_path, fix_taps=1 uses a single grid for all
    std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);

    cre[0].set_size(1, 1, 3); cim[0].set_size(1, 1, 3); dl[0].set_size(1, 1, 3);
    cre[0](0, 0, 0) = 1.0; cim[0](0, 0, 0) = 0.0; dl[0](0, 0, 0) = 0.0;
    cre[0](0, 0, 1) = 1.0; cim[0](0, 0, 1) = 0.0; dl[0](0, 0, 1) = 10.0e-9;
    cre[0](0, 0, 2) = 1.0; cim[0](0, 0, 2) = 0.0; dl[0](0, 0, 2) = 25.0e-9;

    cre[1].set_size(1, 1, 1); cim[1].set_size(1, 1, 1); dl[1].set_size(1, 1, 1);
    cre[1](0, 0, 0) = 2.0; cim[1](0, 0, 0) = 0.0; dl[1](0, 0, 0) = 15.0e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 1);

    REQUIRE(cre_q.size() == 2);
    // Both snapshots should have identical delay grids
    arma::uword n_taps = dl_q[0].n_slices;
    REQUIRE(dl_q[1].n_slices == n_taps);
    for (arma::uword k = 0; k < n_taps; ++k)
        CHECK(std::abs(dl_q[0](0, 0, k) - dl_q[1](0, 0, k)) < 1e-20);
}

TEST_CASE("quantize_delays - Variable n_path with shared delays and fix_taps=2")
{
    // 2 snapshots, shared delays [1,1,n_p], different n_path
    std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);

    cre[0].set_size(2, 1, 3); cim[0].set_size(2, 1, 3);
    dl[0].set_size(1, 1, 3);
    cre[0].ones(); cim[0].zeros();
    dl[0](0, 0, 0) = 0.0; dl[0](0, 0, 1) = 12.5e-9; dl[0](0, 0, 2) = 30.0e-9;

    cre[1].set_size(2, 1, 2); cim[1].set_size(2, 1, 2);
    dl[1].set_size(1, 1, 2);
    cre[1].ones(); cim[1].zeros();
    dl[1](0, 0, 0) = 5.0e-9; dl[1](0, 0, 1) = 20.0e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 2);

    REQUIRE(dl_q.size() == 2);
    // Output delays should be shared [1,1,n_taps]
    CHECK(dl_q[0].n_rows == 1);
    CHECK(dl_q[0].n_cols == 1);
    CHECK(dl_q[1].n_rows == 1);
    CHECK(dl_q[1].n_cols == 1);
    // Coefficients are per-antenna
    CHECK(cre_q[0].n_rows == 2);
    CHECK(cre_q[1].n_rows == 2);
}

TEST_CASE("quantize_delays - fix_taps=1 produces uniform grid")
{
    std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);

    for (int s = 0; s < 2; ++s) {
        cre[s].set_size(2, 1, 2);
        cre[s].ones();
        cim[s].set_size(2, 1, 2);
        cim[s].zeros();
        dl[s].set_size(2, 1, 2);
        dl[s](0, 0, 0) = 0.0;
        dl[s](0, 0, 1) = 12.5e-9;
        dl[s](1, 0, 0) = 5.0e-9;
        dl[s](1, 0, 1) = 27.5e-9;
    }

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 1);

    REQUIRE(dl_q.size() == 2);
    arma::uword n_taps = dl_q[0].n_slices;

    for (arma::uword k = 0; k < n_taps; ++k)
        CHECK(std::abs(dl_q[0](0, 0, k) - dl_q[1](0, 0, k)) < 1e-20);
}

TEST_CASE("quantize_delays - fix_taps=3 same delays across snapshots")
{
    std::vector<arma::Cube<double>> cre(3), cim(3), dl(3);

    for (int s = 0; s < 3; ++s) {
        cre[s].set_size(1, 1, 2);
        cre[s].ones();
        cim[s].set_size(1, 1, 2);
        cim[s].zeros();
        dl[s].set_size(1, 1, 2);
        dl[s](0, 0, 0) = 0.0;
        dl[s](0, 0, 1) = 12.5e-9 + (double)s * 0.1e-9;
    }

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 3);

    REQUIRE(dl_q.size() == 3);
    arma::uword n_taps = dl_q[0].n_slices;
    for (arma::uword s = 1; s < 3; ++s) {
        REQUIRE(dl_q[s].n_slices == n_taps);
        for (arma::uword k = 0; k < n_taps; ++k)
            CHECK(std::abs(dl_q[s](0, 0, k) - dl_q[0](0, 0, k)) < 1e-20);
    }
}

TEST_CASE("quantize_delays - Float precision")
{
    std::vector<arma::Cube<float>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 2);
    cim[0].set_size(1, 1, 2);
    dl[0].set_size(1, 1, 2);

    cre[0](0, 0, 0) = 1.0f; cim[0](0, 0, 0) = 0.5f; dl[0](0, 0, 0) = 0.0f;
    cre[0](0, 0, 1) = 2.0f; cim[0](0, 0, 1) = 1.0f; dl[0](0, 0, 1) = 12.5e-9f;

    std::vector<arma::Cube<float>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9f, 48, 1.0f, 0);

    REQUIRE(cre_q.size() == 1);
    CHECK(cre_q[0].n_slices >= 2);
}

TEST_CASE("quantize_delays - Paths combining at same tap")
{
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 2);
    cim[0].set_size(1, 1, 2);
    dl[0].set_size(1, 1, 2);

    cre[0](0, 0, 0) = 1.0; cim[0](0, 0, 0) = 0.0; dl[0](0, 0, 0) = 10.0e-9;
    cre[0](0, 0, 1) = 2.0; cim[0](0, 0, 1) = 0.0; dl[0](0, 0, 1) = 10.0e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    bool found = false;
    for (arma::uword k = 0; k < cre_q[0].n_slices; ++k) {
        if (std::abs(dl_q[0](0, 0, k) - 10.0e-9) < 1e-20) {
            CHECK(std::abs(cre_q[0](0, 0, k) - 3.0) < 1e-12);
            found = true;
        }
    }
    REQUIRE(found);
}

TEST_CASE("quantize_delays - Zero delay (tap 0)")
{
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 1);
    cim[0].set_size(1, 1, 1);
    dl[0].set_size(1, 1, 1);

    cre[0](0, 0, 0) = 1.0;
    cim[0](0, 0, 0) = -1.0;
    dl[0](0, 0, 0) = 0.0;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);

    REQUIRE(cre_q.size() == 1);
    CHECK(std::abs(cre_q[0](0, 0, 0) - 1.0) < 1e-12);
    CHECK(std::abs(cim_q[0](0, 0, 0) + 1.0) < 1e-12);
    CHECK(std::abs(dl_q[0](0, 0, 0) - 0.0) < 1e-20);
}

TEST_CASE("quantize_delays - Unlimited taps (max_no_taps=0)")
{
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1);
    cre[0].set_size(1, 1, 3);
    cim[0].set_size(1, 1, 3);
    dl[0].set_size(1, 1, 3);

    cre[0](0, 0, 0) = 1.0; cim[0](0, 0, 0) = 0.0; dl[0](0, 0, 0) = 0.0;
    cre[0](0, 0, 1) = 1.0; cim[0](0, 0, 1) = 0.0; dl[0](0, 0, 1) = 12.5e-9;
    cre[0](0, 0, 2) = 1.0; cim[0](0, 0, 2) = 0.0; dl[0](0, 0, 2) = 100.0e-9;

    std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 0, 1.0, 0);

    REQUIRE(cre_q.size() == 1);
    int n_nonzero = 0;
    for (arma::uword k = 0; k < cre_q[0].n_slices; ++k)
        if (std::abs(cre_q[0](0, 0, k)) > 1e-10)
            n_nonzero++;
    CHECK(n_nonzero == 4);
}
