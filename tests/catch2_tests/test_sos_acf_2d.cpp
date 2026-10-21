// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_SOS=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_SOS=1 ./build_linux/test_bin "sos_acf_2d*"

// Verification suite for quadriga_lib::sos_acf_2d.
//
// Pins down: the grid axis, the radial interpolation, the cut-off at the last sample point, the symmetry,
// sizes, float and double, input validation.
// Oracle: values computed by hand for a 3-point table with 1 m spacing. The grid then has 9 points from
// -4 m to 4 m and the center is element (4, 4).

#include <catch2/catch_test_macros.hpp>

#include "quadriga_sos.hpp"

#include <cmath>
#include <stdexcept>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_SOS")

TEST_CASE("sos_acf_2d - grid values match the hand-computed radial interpolation")
{
    arma::vec acf = {1.0, 0.5, 0.25}, dist = {0.0, 1.0, 2.0};
    arma::mat a2;
    arma::vec axis;
    quadriga_lib::sos_acf_2d(acf, dist, a2, &axis);
    REQUIRE(a2.n_rows == 9);
    REQUIRE(a2.n_cols == 9);
    REQUIRE(axis.n_elem == 9);

    arma::vec ref_axis = {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
    CHECK(arma::approx_equal(axis, ref_axis, "absdiff", 1e-14));

    const double r2 = std::sqrt(2.0);
    QD_DIAG("\n# diag: sos_acf_2d hand values\n");
    QD_DIAG("row,col,radius,val,ref\n");
    struct Probe { arma::uword r, c; double radius, ref; };
    Probe probes[] = {
        {4, 4, 0.0, 1.0},                          // Center
        {4, 5, 1.0, 0.5},                          // On a sample point
        {3, 4, 1.0, 0.5},
        {5, 5, r2, 0.5 - (r2 - 1.0) * 0.25},       // Between samples 1 and 2
        {4, 6, 2.0, 0.25},                         // Last sample point: still inside
        {6, 4, 2.0, 0.25},
        {5, 6, std::sqrt(5.0), 0.0},               // Beyond the last sample point
        {4, 7, 3.0, 0.0},
        {0, 0, 4.0 * r2, 0.0}};
    for (const Probe &p : probes)
    {
        QD_DIAG("%llu,%llu,%.4f,%.9f,%.9f\n", (unsigned long long)p.r, (unsigned long long)p.c, p.radius, a2(p.r, p.c), p.ref);
        INFO("element " << p.r << "," << p.c);
        CHECK(std::abs(a2(p.r, p.c) - p.ref) < 1e-14);
    }

    // Symmetric in both axes and under transposition
    CHECK(arma::approx_equal(a2, a2.t(), "absdiff", 1e-14));
    CHECK(arma::approx_equal(a2, arma::flipud(a2), "absdiff", 1e-14));
    CHECK(arma::approx_equal(a2, arma::fliplr(a2), "absdiff", 1e-14));
}

TEST_CASE("sos_acf_2d - optional axis, pre-sized output and float")
{
    arma::vec acf = {1.0, 0.5, 0.25}, dist = {0.0, 1.0, 2.0};
    arma::mat ref;
    quadriga_lib::sos_acf_2d(acf, dist, ref);
    REQUIRE(ref.n_rows == 9);
    CHECK(std::abs(ref(4, 5) - 0.5) < 1e-14);

    arma::mat sized(9, 9);
    sized.fill(9.0);
    const double *ptr = sized.memptr();
    arma::vec axis(3);
    quadriga_lib::sos_acf_2d(acf, dist, sized, &axis);
    CHECK(sized.memptr() == ptr);
    CHECK(arma::approx_equal(sized, ref, "absdiff", 0.0));
    CHECK(axis.n_elem == 9);

    // Spacing other than 1: the axis scales, the values do not
    arma::fvec af = {1.0f, 0.5f, 0.25f}, df = {0.0f, 2.5f, 5.0f}, axf;
    arma::fmat a2f;
    quadriga_lib::sos_acf_2d(af, df, a2f, &axf);
    REQUIRE(a2f.n_rows == 9);
    CHECK(std::abs(axf(0) + 10.0f) < 1e-5f);
    CHECK(std::abs(axf(8) - 10.0f) < 1e-5f);
    CHECK(arma::abs(arma::conv_to<arma::mat>::from(a2f) - ref).max() < 1e-6);
}

TEST_CASE("sos_acf_2d - input validation")
{
    arma::vec acf = {1.0, 0.5, 0.25}, dist = {0.0, 1.0, 2.0};
    arma::mat a2;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_acf_2d(acf, dist, a2));
    }
    SECTION("acf")
    {
        arma::vec one = {1.0};
        CHECK_THROWS_AS(quadriga_lib::sos_acf_2d(one, dist, a2), std::invalid_argument);
    }
    SECTION("dist")
    {
        arma::vec size = {0.0, 1.0}, start = {1.0, 2.0, 3.0}, uneven = {0.0, 1.0, 3.0};
        CHECK_THROWS_AS(quadriga_lib::sos_acf_2d(acf, size, a2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_2d(acf, start, a2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_2d(acf, uneven, a2), std::invalid_argument);
    }
    SECTION("non-finite table")
    {
        arma::vec inf_dist = {0.0, 1.0, arma::datum::inf}, nan_dist = {0.0, std::nan(""), 2.0}, nan_acf = {1.0, std::nan(""), 0.25};
        CHECK_THROWS_AS(quadriga_lib::sos_acf_2d(acf, inf_dist, a2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_2d(acf, nan_dist, a2), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_2d(nan_acf, dist, a2), std::invalid_argument);
    }
    SECTION("a failed call leaves the outputs untouched")
    {
        arma::vec uneven = {0.0, 1.0, 3.0}, axis(2);
        axis.fill(7.0);
        a2.set_size(2, 2);
        a2.fill(7.0);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_2d(acf, uneven, a2, &axis), std::invalid_argument);
        CHECK(a2.n_rows == 2);
        CHECK(a2(1, 1) == 7.0);
        CHECK(axis.n_elem == 2);
    }
}

TEST_CASE("sos_acf_2d - long tables in single precision count as equally spaced")
{
    // 5001 samples with 0.01 m spacing. In float the sample points near 50 m are 3.8e-6 m apart, which is
    // more than 1e-4 of the step: the rounding of the sample points must not make the table "unequal".
    // Checked on the axis only (the full grid has 4e8 points): n = 501 gives a grid of 2001 x 2001.
    for (arma::uword n : {501, 2001, 5001})
    {
        arma::fvec dist(n), acf(n);
        for (arma::uword i = 0; i < n; ++i)
            dist(i) = (float)(0.01 * (double)i), acf(i) = (float)std::exp(-0.01 * (double)i / 10.0);
        arma::vec dist_d(n), acf_d(n);
        for (arma::uword i = 0; i < n; ++i)
            dist_d(i) = 0.01 * (double)i, acf_d(i) = std::exp(-0.01 * (double)i / 10.0);

        // sos_calc_mse and sos_acf_approx use the same test and are cheap for one sinusoid in 1-D
        arma::fmat freq(1, 1);
        freq(0, 0) = 0.01f;
        arma::mat freq_d(1, 1);
        freq_d(0, 0) = (double)freq(0, 0);
        INFO("n_acf " << n);
        float mse = 0.0f;
        CHECK_NOTHROW(mse = quadriga_lib::sos_calc_mse(freq, acf, dist));
        double mse_d = quadriga_lib::sos_calc_mse(freq_d, acf_d, dist_d);
        CHECK(std::abs((double)mse - mse_d) < 1e-3);
        arma::fcube approx;
        CHECK_NOTHROW(quadriga_lib::sos_acf_approx(freq, dist, approx));
        CHECK(approx.n_rows == 4 * n - 3);
        if (n == 501)
        {
            arma::fmat a2;
            arma::fvec axis;
            CHECK_NOTHROW(quadriga_lib::sos_acf_2d(acf, dist, a2, &axis));
            REQUIRE(a2.n_rows == 2001);
            CHECK(std::abs(a2(1000, 1500) - acf(500)) < 1e-6f);
            CHECK(std::abs(axis(2000) - 10.0f) < 1e-5f);
        }
    }

    // A table that is unequal by more than the rounding is still rejected, in float and in double
    arma::fvec bad(5001), acf(5001, arma::fill::ones);
    for (arma::uword i = 0; i < 5001; ++i)
        bad(i) = (float)(0.01 * (double)i);
    bad(2500) += 0.003f;
    arma::fmat freq(1, 1, arma::fill::ones);
    CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(freq, acf, bad), std::invalid_argument);
}

TEST_CASE("sos_acf_2d - the axis output may be the same object as an input")
{
    arma::vec acf = {1.0, 0.5, 0.25}, dist = {0.0, 1.5, 3.0}, ref_axis;
    arma::mat ref, a, b;
    quadriga_lib::sos_acf_2d(acf, dist, ref, &ref_axis);
    REQUIRE(ref_axis.n_elem == 9);

    arma::vec d = dist, c = acf;
    quadriga_lib::sos_acf_2d(acf, d, a, &d); // dist_2d is 'dist'
    quadriga_lib::sos_acf_2d(c, dist, b, &c); // dist_2d is 'acf'
    CHECK(arma::approx_equal(a, ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(b, ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(d, ref_axis, "absdiff", 0.0));
    CHECK(arma::approx_equal(c, ref_axis, "absdiff", 0.0));
    CHECK(std::abs(ref_axis(8) - 6.0) < 1e-14);
}
