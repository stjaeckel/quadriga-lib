// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

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

// Diagnostic output. Comment out to silence the tables. The assertions run either way.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

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
}
