// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_channel.hpp"
#include "quadriga_lib.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <string>
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

// Helper: element-wise identity of two quantize_delays results
template <typename dtype>
static bool qd_same(const std::vector<arma::Cube<dtype>> &a, const std::vector<arma::Cube<dtype>> &b, double tol = 0.0)
{
    if (a.size() != b.size())
        return false;
    for (size_t s = 0; s < a.size(); ++s)
    {
        if (a[s].n_rows != b[s].n_rows || a[s].n_cols != b[s].n_cols || a[s].n_slices != b[s].n_slices)
            return false;
        for (arma::uword i = 0; i < a[s].n_elem; ++i)
            if (std::abs((double)a[s](i) - (double)b[s](i)) > tol)
                return false;
    }
    return true;
}

TEST_CASE("quantize_delays - Relative delays of get_channels_spherical (B-002)")
{
    // get_channels_spherical with relative delays (default) returns slightly negative delays for elements that are
    // closer than the phase centre; all of their power must go to tap 0
    auto tx = quadriga_lib::generate_arrayant_ula<double>(8, 3.5e9, 0.5);
    auto rx = quadriga_lib::generate_arrayant_ula<double>(4, 3.5e9, 0.5);
    arma::mat fbs = {{50.0, 50.0}, {0.0, 30.0}, {10.0, 2.0}}, lbs = fbs; // path 0: LOS, path 1: NLOS
    arma::vec gain = {1.0, 0.5}, len = {0.0, 0.0};
    arma::mat M(8, 2, arma::fill::zeros);
    M.row(0).fill(1.0), M.row(6).fill(-1.0);
    std::vector<arma::cube> cre(1), cim(1), dl(1), cre_q, cim_q, dl_q;
    quadriga_lib::get_channels_spherical<double>(&tx, &rx, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 100.0, 0.0, 10.0, 0.0, 0.0, 0.3,
                                                 &fbs, &lbs, &gain, &len, &M, &cre[0], &cim[0], &dl[0], 3.5e9);
    REQUIRE(dl[0].min() < 0.0);
    REQUIRE(arma::accu(dl[0] < 0.0) > 0);

    for (int fix_taps = 0; fix_taps < 4; ++fix_taps)
    {
        REQUIRE_NOTHROW(quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, fix_taps));
        REQUIRE(cre_q.size() == 1);
        REQUIRE(cre_q[0].n_rows == 4);
        REQUIRE(cre_q[0].n_cols == 8);
        REQUIRE(dl_q[0].n_rows == 4);
        REQUIRE(dl_q[0].n_cols == 8);
        for (arma::uword t = 0; t < 8; ++t)
            for (arma::uword r = 0; r < 4; ++r)
                if (dl[0](r, t, 0) < 0.0) // LOS path with a negative relative delay: unchanged coefficient at tap 0
                {
                    CHECK(dl_q[0](r, t, 0) == 0.0);
                    CHECK(std::abs(cre_q[0](r, t, 0) - cre[0](r, t, 0)) < 1e-12);
                    CHECK(std::abs(cim_q[0](r, t, 0) - cim[0](r, t, 0)) < 1e-12);
                }
    }
}

TEST_CASE("quantize_delays - Small negative delays are clamped to tap 0 (B-002)")
{
    const double ts = 5.0e-9;
    const double half = 0.5 * ts;

    // Reference: first delay exactly 0
    std::vector<arma::Cube<double>> cre(1), cim(1), dl0(1), dl(1), cre_r, cim_r, dl_r, cre_q, cim_q, dl_q;
    cre[0] = arma::cube(1, 1, 3, arma::fill::ones);
    cim[0] = arma::cube(1, 1, 3, arma::fill::zeros);
    cre[0](0, 0, 1) = 0.5, cim[0](0, 0, 2) = -0.25;
    dl0[0] = arma::cube(1, 1, 3);
    dl0[0](0, 0, 0) = 0.0, dl0[0](0, 0, 1) = 12.5e-9, dl0[0](0, 0, 2) = 33.4e-9;

    for (double d0 : {-1.0e-20, -1.0e-12, -5.0e-11, -2.4e-9, std::nextafter(-half, 0.0)})
        for (int fix_taps = 0; fix_taps < 4; ++fix_taps)
        {
            dl = dl0;
            dl[0](0, 0, 0) = d0;
            quadriga_lib::quantize_delays(&cre, &cim, &dl0, &cre_r, &cim_r, &dl_r, ts, 48, 1.0, fix_taps);
            REQUIRE_NOTHROW(quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, ts, 48, 1.0, fix_taps));
            CHECK(qd_same(cre_q, cre_r));
            CHECK(qd_same(cim_q, cim_r));
            CHECK(qd_same(dl_q, dl_r));
            CHECK(dl_q[0](0, 0, 0) == 0.0);
            CHECK(cre_q[0](0, 0, 0) == 1.0); // all power of path 0 in tap 0
        }

    // Per-antenna delays, one pair negative
    std::vector<arma::Cube<double>> cre2(1), cim2(1), dl2(1);
    cre2[0] = arma::cube(2, 2, 2, arma::fill::ones);
    cim2[0] = arma::cube(2, 2, 2, arma::fill::zeros);
    dl2[0] = arma::cube(2, 2, 2);
    dl2[0].slice(0).fill(0.0);
    dl2[0].slice(1).fill(20.0e-9);
    dl2[0](1, 0, 0) = -1.0e-9;
    REQUIRE_NOTHROW(quadriga_lib::quantize_delays(&cre2, &cim2, &dl2, &cre_q, &cim_q, &dl_q, ts, 48, 1.0, 0));
    CHECK(dl_q[0](1, 0, 0) == 0.0);
    CHECK(cre_q[0](1, 0, 0) == 1.0);

    // At or below -tap_spacing / 2: error
    for (double d0 : {-half, -5.0e-9, -1.0})
    {
        dl = dl0;
        dl[0](0, 0, 0) = d0;
        CHECK_THROWS_AS(quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, ts, 48, 1.0, 0),
                        std::invalid_argument);
    }

    // Single precision: the same rule
    std::vector<arma::Cube<float>> fre(1), fim(1), fdl(1), fre_q, fim_q, fdl_q;
    fre[0] = arma::fcube(1, 1, 2, arma::fill::ones);
    fim[0] = arma::fcube(1, 1, 2, arma::fill::zeros);
    fdl[0] = arma::fcube(1, 1, 2);
    fdl[0](0, 0, 0) = -1.0e-10f, fdl[0](0, 0, 1) = 12.5e-9f;
    REQUIRE_NOTHROW(quadriga_lib::quantize_delays(&fre, &fim, &fdl, &fre_q, &fim_q, &fdl_q, 5.0e-9f, 48, 1.0f, 0));
    CHECK(fdl_q[0](0, 0, 0) == 0.0f);
    CHECK(fre_q[0](0, 0, 0) == 1.0f);
    fdl[0](0, 0, 0) = -0.5f * 5.0e-9f;
    CHECK_THROWS_AS(quadriga_lib::quantize_delays(&fre, &fim, &fdl, &fre_q, &fim_q, &fdl_q, 5.0e-9f, 48, 1.0f, 0),
                    std::invalid_argument);
}

TEST_CASE("quantize_delays - Non-finite and too large delays throw (B-002)")
{
    const double ts = 5.0e-9;
    std::vector<arma::Cube<double>> cre(1), cim(1), dl0(1), dl(1), cre_q, cim_q, dl_q;
    cre[0] = arma::cube(2, 2, 3, arma::fill::ones);
    cim[0] = arma::cube(2, 2, 3, arma::fill::zeros);
    dl0[0] = arma::cube(2, 2, 3);
    dl0[0].slice(0).fill(0.0), dl0[0].slice(1).fill(12.5e-9), dl0[0].slice(2).fill(33.4e-9);

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double big = 2147483649.0 * ts; // 2^31 + 1 taps
    for (double bad : {nan, inf, -inf, big, 21.4748, 1.0e300})
        for (int fix_taps = 0; fix_taps < 4; ++fix_taps)
        {
            dl = dl0;
            dl[0](1, 0, 1) = bad;
            CHECK_THROWS_AS(quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, ts, 48, 1.0, fix_taps),
                            std::invalid_argument);
        }

    // Shared delays
    std::vector<arma::Cube<double>> dls(1);
    dls[0] = arma::cube(1, 1, 3);
    dls[0](0, 0, 0) = 0.0, dls[0](0, 0, 1) = nan, dls[0](0, 0, 2) = 20.0e-9;
    CHECK_THROWS_AS(quadriga_lib::quantize_delays(&cre, &cim, &dls, &cre_q, &cim_q, &dl_q, ts, 48, 1.0, 1),
                    std::invalid_argument);

    // Non-finite tap spacing
    CHECK_THROWS_AS(quadriga_lib::quantize_delays(&cre, &cim, &dl0, &cre_q, &cim_q, &dl_q, nan),
                    std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays(&cre, &cim, &dl0, &cre_q, &cim_q, &dl_q, inf),
                    std::invalid_argument);

    // Just below the bound: a single tap far out, no dense allocation over the delay range
    std::vector<arma::Cube<double>> c1(1), i1(1), d1(1);
    c1[0] = arma::cube(1, 1, 2, arma::fill::ones);
    i1[0] = arma::cube(1, 1, 2, arma::fill::zeros);
    d1[0] = arma::cube(1, 1, 2);
    d1[0](0, 0, 0) = 0.0;
    d1[0](0, 0, 1) = 2147483000.0 * ts; // integer tap index below 2^31
    for (arma::uword max_no_taps : {0, 48})
    {
        REQUIRE_NOTHROW(quadriga_lib::quantize_delays(&c1, &i1, &d1, &cre_q, &cim_q, &dl_q, ts, max_no_taps, 1.0, 0));
        REQUIRE(dl_q[0].n_elem == 2);
        CHECK(dl_q[0](0, 0, 0) == 0.0);
        CHECK(std::abs(dl_q[0](0, 0, 1) - 2147483000.0 * ts) < 1e-6 * ts);
        CHECK(cre_q[0](0, 0, 0) == 1.0);
        CHECK(cre_q[0](0, 0, 1) == 1.0);
    }

    // Acoustic tap spacing: the bound is in taps, not in seconds (48 kHz, delays of several seconds)
    d1[0](0, 0, 0) = 0.5, d1[0](0, 0, 1) = 30.0;
    REQUIRE_NOTHROW(quadriga_lib::quantize_delays(&c1, &i1, &d1, &cre_q, &cim_q, &dl_q, 1.0 / 48000.0, 48, 1.0, 0));
    CHECK(std::abs(dl_q[0](0, 0, 1) - 30.0) < 1e-9);
}

TEST_CASE("quantize_delays - Delay offset gives the same taps (B-054)")
{
    // Tap spacing, delays and offset are binary fractions, so adding the offset is exact and the taps shift by an
    // integer number. The results must be identical apart from the offset in the output delays.
    const double ts = std::ldexp(1.0, -28);  // 3.7 ns
    const double off = std::ldexp(1.0, -6);  // 15.6 ms = 2^22 taps
    arma::arma_rng::set_seed(17);
    std::vector<arma::Cube<double>> cre(3), cim(3), dl(3), dls(3), dl_off(3), dls_off(3);
    for (int s = 0; s < 3; ++s)
    {
        cre[s] = arma::cube(4, 3, 20, arma::fill::randn);
        cim[s] = arma::cube(4, 3, 20, arma::fill::randn);
        dl[s] = arma::floor(arma::cube(4, 3, 20, arma::fill::randu) * 65536.0) * std::ldexp(1.0, -36); // 0 .. 0.95 us
        dls[s] = arma::floor(arma::cube(1, 1, 20, arma::fill::randu) * 65536.0) * std::ldexp(1.0, -36);
        dl_off[s] = dl[s] + off;
        dls_off[s] = dls[s] + off;
    }
    for (int shared = 0; shared < 2; ++shared)
        for (int fix_taps = 0; fix_taps < 4; ++fix_taps)
            for (arma::uword max_no_taps : {8, 48})
                for (double pe : {1.0, 0.5})
                {
                    std::vector<arma::Cube<double>> re0, im0, d0, re1, im1, d1;
                    quadriga_lib::quantize_delays(&cre, &cim, shared ? &dls : &dl, &re0, &im0, &d0, ts, max_no_taps, pe, fix_taps);
                    quadriga_lib::quantize_delays(&cre, &cim, shared ? &dls_off : &dl_off, &re1, &im1, &d1, ts, max_no_taps, pe, fix_taps);
                    CHECK(qd_same(re0, re1));
                    CHECK(qd_same(im0, im1));
                    REQUIRE(d0.size() == d1.size());
                    for (size_t s = 0; s < d0.size(); ++s)
                    {
                        REQUIRE(arma::size(d0[s]) == arma::size(d1[s]));
                        // Output delays: unused (zero-padded) slots stay 0, used ones carry the offset
                        for (arma::uword i = 0; i < d0[s].n_elem; ++i)
                            if (d1[s](i) != 0.0)
                                CHECK(d1[s](i) - off == d0[s](i));
                            else
                                CHECK(d0[s](i) == 0.0);
                    }
                }

    // Timing guard (generous): with the old dense vectors over the absolute delay range, a 20 ms offset on 16x16
    // antennas took about 4 s; now it is a few ms
    std::vector<arma::Cube<double>> c2(1), i2(1), d2(1), re_q, im_q, dl_q;
    c2[0] = arma::cube(16, 16, 20, arma::fill::randn);
    i2[0] = arma::cube(16, 16, 20, arma::fill::randn);
    d2[0] = arma::cube(16, 16, 20, arma::fill::randu) * 2.0e-6 + 20.0e-3;
    auto t0 = std::chrono::steady_clock::now();
    quadriga_lib::quantize_delays(&c2, &i2, &d2, &re_q, &im_q, &dl_q, 5.0e-9, 48, 1.0, 0);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(sec < 1.0);
}

TEST_CASE("quantize_delays - Outputs may alias the inputs (B-055)")
{
    // Delay modes: 0 = random, 1 = on the tap grid (no interpolation), 2 = random with max_no_taps = 3 (fallback to
    // the nearest tap), 3 = the case of md/bugs.md (path at delay 0, interpolation for the others)
    arma::arma_rng::set_seed(5);
    for (int mode = 0; mode < 4; ++mode)
    for (int shared = 0; shared < 2; ++shared)
        for (int fix_taps = 0; fix_taps < 4; ++fix_taps)
        {
            arma::uword max_no_taps = (mode == 2) ? 3 : 48;
            std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
            for (int s = 0; s < 2; ++s)
            {
                cre[s] = arma::cube(2, 3, 5, arma::fill::randn);
                cim[s] = arma::cube(2, 3, 5, arma::fill::randn);
                dl[s] = shared ? arma::cube(1, 1, 5, arma::fill::randu) * 100.0e-9
                               : arma::cube(2, 3, 5, arma::fill::randu) * 100.0e-9;
                if (mode == 1)
                    dl[s] = arma::round(dl[s] / 5.0e-9) * 5.0e-9;
                if (mode == 3)
                    for (arma::uword p = 0; p < 5; ++p)
                        dl[s].slice(p).fill(p == 0 ? 0.0 : 12.5e-9 + 4.2e-9 * (double)p);
            }
            std::vector<arma::Cube<double>> re0, im0, d0;
            quadriga_lib::quantize_delays(&cre, &cim, &dl, &re0, &im0, &d0, 5.0e-9, max_no_taps, 1.0, fix_taps);

            // In place: each output aliases its own input
            auto a_re = cre, a_im = cim, a_dl = dl;
            quadriga_lib::quantize_delays(&a_re, &a_im, &a_dl, &a_re, &a_im, &a_dl, 5.0e-9, max_no_taps, 1.0, fix_taps);
            CHECK(qd_same(a_re, re0));
            CHECK(qd_same(a_im, im0));
            CHECK(qd_same(a_dl, d0));

            // Across arguments: delay_quant aliases coeff_re, coeff_re_quant aliases delay
            auto b_re = cre, b_im = cim, b_dl = dl;
            std::vector<arma::Cube<double>> b_imq;
            quadriga_lib::quantize_delays(&b_re, &b_im, &b_dl, &b_dl, &b_imq, &b_re, 5.0e-9, max_no_taps, 1.0, fix_taps);
            CHECK(qd_same(b_dl, re0));
            CHECK(qd_same(b_imq, im0));
            CHECK(qd_same(b_re, d0));

            // Rotated: coeff_re_quant aliases coeff_im, coeff_im_quant aliases delay, delay_quant aliases coeff_re
            auto c_re = cre, c_im = cim, c_dl = dl;
            quadriga_lib::quantize_delays(&c_re, &c_im, &c_dl, &c_im, &c_dl, &c_re, 5.0e-9, max_no_taps, 1.0, fix_taps);
            CHECK(qd_same(c_im, re0));
            CHECK(qd_same(c_dl, im0));
            CHECK(qd_same(c_re, d0));
        }

    // Two outputs pointing to the same vector cannot both be returned: error
    std::vector<arma::Cube<double>> cre(1), cim(1), dl(1), o1, o2;
    cre[0] = arma::cube(1, 1, 2, arma::fill::ones);
    cim[0] = arma::cube(1, 1, 2, arma::fill::zeros);
    dl[0] = arma::cube(1, 1, 2, arma::fill::zeros);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays(&cre, &cim, &dl, &o1, &o1, &o2), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays(&cre, &cim, &dl, &o1, &o2, &o1), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::quantize_delays(&cre, &cim, &dl, &o2, &o1, &o1), std::invalid_argument);
}

TEST_CASE("quantize_delays - Output delay shapes (B-103)")
{
    // [1, 1, n_taps] output delays only for shared [1, 1, n_path] input with fix_taps 1 or 2
    std::vector<arma::Cube<double>> cre(1), cim(1), dl_ant(1), dl_shr(1), cre_q, cim_q, dl_q;
    cre[0] = arma::cube(2, 2, 3, arma::fill::ones);
    cim[0] = arma::cube(2, 2, 3, arma::fill::zeros);
    dl_ant[0] = arma::cube(2, 2, 3);
    dl_shr[0] = arma::cube(1, 1, 3);
    for (arma::uword p = 0; p < 3; ++p)
        dl_ant[0].slice(p).fill(10.0e-9 * (double)p + 2.0e-9), dl_shr[0](0, 0, p) = 10.0e-9 * (double)p + 2.0e-9;

    for (int fix_taps = 0; fix_taps < 4; ++fix_taps)
    {
        quadriga_lib::quantize_delays(&cre, &cim, &dl_ant, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, fix_taps);
        CHECK(dl_q[0].n_rows == 2);
        CHECK(dl_q[0].n_cols == 2);
        CHECK(dl_q[0].n_slices == cre_q[0].n_slices);

        quadriga_lib::quantize_delays(&cre, &cim, &dl_shr, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, fix_taps);
        arma::uword n = (fix_taps == 1 || fix_taps == 2) ? 1 : 2;
        CHECK(dl_q[0].n_rows == n);
        CHECK(dl_q[0].n_cols == n);
        CHECK(dl_q[0].n_slices == cre_q[0].n_slices);
        CHECK(cre_q[0].n_rows == 2);
        CHECK(cre_q[0].n_cols == 2);
    }
}

TEST_CASE("quantize_delays - Relative delays of an acoustic array (B-002)")
{
    // get_channels_multifreq, speed of sound, 48 kHz taps: a 5 cm microphone pair turned by 0.3 rad already has
    // relative delays below -tap_spacing / 2 (about -1 tap). Relative delays throw with a message that points to
    // absolute delays; absolute delays quantize.
    const double c = 343.0, f = 1000.0, ts = 1.0 / 48000.0;
    auto mic = quadriga_lib::generate_arrayant_ula<double>(2, f, 0.5);
    mic.element_pos.col(0) = arma::vec{0.0, -0.025, 0.0};
    mic.element_pos.col(1) = arma::vec{0.0, 0.025, 0.0};
    std::vector<quadriga_lib::arrayant<double>> rx = {mic}, tx = {quadriga_lib::generate_arrayant_omni<double>()};
    rx[0].center_frequency = f;
    arma::mat fbs = {{2.5, 2.5}, {0.0, 1.5}, {1.5, 0.2}}; // path 0 on the TX-RX line (LOS)
    arma::mat pg(2, 1);
    pg(0, 0) = 1.0, pg(1, 0) = 0.5;
    arma::vec len = {0.0, 0.0}, fin = {f}, fout = {f};
    arma::cube M(2, 2, 1, arma::fill::zeros);
    M.row(0).fill(1.0);

    for (bool absolute : {false, true})
    {
        std::vector<arma::cube> cre, cim, dl, cre_q, cim_q, dl_q;
        quadriga_lib::get_channels_multifreq<double>(tx, rx, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 5.0, 0.0, 1.5, 0.0, 0.0, 0.3,
                                                     fbs, fbs, pg, len, M, fin, fout, cre, cim, dl, absolute, false, c);
        REQUIRE(dl.size() == 1);
        if (!absolute)
        {
            REQUIRE(dl[0].min() <= -0.5 * ts);
            std::string msg;
            try
            {
                quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, ts);
            }
            catch (const std::invalid_argument &e)
            {
                msg = e.what();
            }
            CHECK(msg.find("at or below -tap_spacing / 2") != std::string::npos);
            CHECK(msg.find("use_absolute_delays = true") != std::string::npos);
        }
        else
        {
            REQUIRE(dl[0].min() > 0.0);
            REQUIRE_NOTHROW(quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, ts));
            REQUIRE(cre_q.size() == 1);
            CHECK(cre_q[0].n_rows == 2);
            CHECK(arma::accu(arma::square(cre_q[0]) + arma::square(cim_q[0])) > 0.0);
        }
    }
}
