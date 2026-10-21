// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_SOS=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_SOS=1 ./build_linux/test_bin "sos_acf_approx*"

// Verification suite for quadriga_lib::sos_acf_approx.
//
// Pins down: the closed form of the ACF of the sinusoids, the assignment of rows and columns to the axes
// of each plane, the three output shapes, the grid axis, float and double, input validation.
// Oracle: R(dx) = (1/N) · sum_n cos( 2·pi · f_n·dx ), evaluated in scalar double math at the grid point.
// For a single sinusoid this is one cosine in closed form.
// Conventions: frequencies with three different components, so that a swapped axis is visible; the
// table has 4 samples with 2 m spacing, which gives a grid of 13 points from -12 m to 12 m, center 6.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_sos.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_SOS")

namespace
{
    const double pi = 3.141592653589793;

    // Oracle: ACF of the sinusoids for the displacement a along axis ia and b along axis ib
    double ref_acf(const arma::mat &freq, arma::uword ia, double a, arma::uword ib, double b)
    {
        double acc = 0.0;
        for (arma::uword n = 0; n < freq.n_rows; ++n)
            acc += std::cos(2.0 * pi * (freq(n, ia) * a + (ib < freq.n_cols ? freq(n, ib) * b : 0.0)));
        return acc / (double)freq.n_rows;
    }
}

TEST_CASE("sos_acf_approx - a single 3-D sinusoid gives one cosine per plane with the documented axes")
{
    arma::mat freq = {{0.03, 0.05, -0.02}};
    arma::vec dist = {0.0, 2.0, 4.0, 6.0};
    arma::cube ro;
    arma::vec axis;
    quadriga_lib::sos_acf_approx(freq, dist, ro, &axis);
    REQUIRE(ro.n_rows == 13);
    REQUIRE(ro.n_cols == 13);
    REQUIRE(ro.n_slices == 3);
    REQUIRE(axis.n_elem == 13);
    CHECK(std::abs(axis(0) + 12.0) < 1e-13);
    CHECK(std::abs(axis(6)) < 1e-13);
    CHECK(std::abs(axis(12) - 12.0) < 1e-13);

    // Slice 0: x-y, slice 1: x-z, slice 2: y-z; column = first axis, row = second axis
    const arma::uword first[3] = {0, 0, 1}, second[3] = {1, 2, 2};
    double max_err = 0.0;
    QD_DIAG("\n# diag: sos_acf_approx single sinusoid\n");
    QD_DIAG("plane,row,col,val,ref\n");
    for (arma::uword s = 0; s < 3; ++s)
        for (arma::uword c = 0; c < 13; ++c)
            for (arma::uword r = 0; r < 13; ++r)
            {
                double ref = std::cos(2.0 * pi * (freq(0, first[s]) * axis(c) + freq(0, second[s]) * axis(r)));
                if (r == 2 && c == 9)
                    QD_DIAG("%llu,%llu,%llu,%.9f,%.9f\n", (unsigned long long)s, (unsigned long long)r, (unsigned long long)c, ro(r, c, s), ref);
                max_err = std::max(max_err, std::abs(ro(r, c, s) - ref));
            }
    CHECK(max_err < 1e-12);
    for (arma::uword s = 0; s < 3; ++s)
        CHECK(std::abs(ro(6, 6, s) - 1.0) < 1e-14); // Zero displacement
    CHECK(std::abs(ro(2, 9, 0) - ro(9, 2, 0)) > 0.05); // The planes are not symmetric under transposition
}

TEST_CASE("sos_acf_approx - several sinusoids average, in one, two and three dimensions")
{
    arma::mat freq = {{0.03, 0.05, -0.02}, {-0.07, 0.01, 0.04}, {0.002, -0.09, 0.06}, {0.11, 0.08, 0.0}};
    arma::vec dist = {0.0, 2.0, 4.0, 6.0};
    arma::cube ro;
    arma::vec axis;

    // 3-D
    quadriga_lib::sos_acf_approx(freq, dist, ro, &axis);
    REQUIRE(ro.n_slices == 3);
    CHECK(std::abs(ro(1, 10, 0) - ref_acf(freq, 0, axis(10), 1, axis(1))) < 1e-12);
    CHECK(std::abs(ro(1, 10, 1) - ref_acf(freq, 0, axis(10), 2, axis(1))) < 1e-12);
    CHECK(std::abs(ro(1, 10, 2) - ref_acf(freq, 1, axis(10), 2, axis(1))) < 1e-12);
    // Point symmetry of an ACF: R(-dx) = R(dx)
    CHECK(std::abs(ro(1, 10, 0) - ro(11, 2, 0)) < 1e-12);
    CHECK(std::abs(ro(1, 10, 0)) < 0.99);

    // 2-D
    arma::mat f2 = freq.cols(0, 1);
    quadriga_lib::sos_acf_approx(f2, dist, ro);
    REQUIRE(ro.n_rows == 13);
    REQUIRE(ro.n_cols == 13);
    REQUIRE(ro.n_slices == 1);
    double max_err = 0.0;
    for (arma::uword c = 0; c < 13; ++c)
        for (arma::uword r = 0; r < 13; ++r)
            max_err = std::max(max_err, std::abs(ro(r, c, 0) - ref_acf(f2, 0, axis(c), 1, axis(r))));
    CHECK(max_err < 1e-12);

    // 1-D
    arma::mat f1 = freq.col(0);
    quadriga_lib::sos_acf_approx(f1, dist, ro);
    REQUIRE(ro.n_rows == 13);
    REQUIRE(ro.n_cols == 1);
    REQUIRE(ro.n_slices == 1);
    max_err = 0.0;
    for (arma::uword r = 0; r < 13; ++r)
        max_err = std::max(max_err, std::abs(ro(r, 0, 0) - ref_acf(f1, 0, axis(r), 3, 0.0)));
    CHECK(max_err < 1e-12);
    CHECK(std::abs(ro(6, 0, 0) - 1.0) < 1e-14);
    CHECK(std::abs(ro(0, 0, 0)) < 0.99);
}

TEST_CASE("sos_acf_approx - pre-sized output and float")
{
    arma::mat freq = {{0.03, 0.05, -0.02}, {-0.07, 0.01, 0.04}};
    arma::vec dist = {0.0, 2.0, 4.0, 6.0};
    arma::cube ref;
    quadriga_lib::sos_acf_approx(freq, dist, ref);
    REQUIRE(ref.n_slices == 3);
    CHECK(std::abs(ref(6, 6, 1) - 1.0) < 1e-14);

    arma::cube sized(13, 13, 3);
    sized.fill(9.0);
    const double *ptr = sized.memptr();
    quadriga_lib::sos_acf_approx(freq, dist, sized);
    CHECK(sized.memptr() == ptr);
    CHECK(arma::approx_equal(sized, ref, "absdiff", 0.0));

    arma::cube wrong(2, 2, 2);
    quadriga_lib::sos_acf_approx(freq, dist, wrong);
    CHECK(arma::approx_equal(wrong, ref, "absdiff", 0.0));

    arma::fmat ff = arma::conv_to<arma::fmat>::from(freq);
    arma::fvec df = arma::conv_to<arma::fvec>::from(dist), axf;
    arma::fcube rf;
    quadriga_lib::sos_acf_approx(ff, df, rf, &axf);
    REQUIRE(rf.n_slices == 3);
    REQUIRE(axf.n_elem == 13);
    // Arguments up to about 12 rad from float inputs: 12 · 1.2e-7 per term
    CHECK(arma::abs(arma::conv_to<arma::cube>::from(rf) - ref).max() < 1e-5);
}

TEST_CASE("sos_acf_approx - input validation")
{
    arma::mat freq = {{0.03, 0.05, -0.02}};
    arma::vec dist = {0.0, 2.0, 4.0, 6.0};
    arma::cube ro;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_acf_approx(freq, dist, ro));
    }
    SECTION("freq")
    {
        arma::mat empty, four(2, 4, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(empty, dist, ro), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(four, dist, ro), std::invalid_argument);
    }
    SECTION("dist")
    {
        arma::vec one = {0.0}, start = {1.0, 2.0, 3.0}, uneven = {0.0, 1.0, 3.0};
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(freq, one, ro), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(freq, start, ro), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(freq, uneven, ro), std::invalid_argument);
    }
    SECTION("non-finite dist")
    {
        // A NaN passes every comparison of the spacing test, an Inf at the end gives an infinite step
        arma::vec nan_mid = {0.0, std::nan(""), 4.0, 6.0}, nan_end = {0.0, 2.0, 4.0, std::nan("")}, inf_end = {0.0, 2.0, 4.0, arma::datum::inf};
        arma::vec two_nan = {0.0, std::nan("")}, two_inf = {0.0, arma::datum::inf};
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(freq, nan_mid, ro), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(freq, nan_end, ro), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(freq, inf_end, ro), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(freq, two_nan, ro), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(freq, two_inf, ro), std::invalid_argument);
    }
    SECTION("a failed call leaves the outputs untouched")
    {
        arma::vec uneven = {0.0, 1.0, 3.0}, axis(2);
        axis.fill(7.0);
        ro.set_size(2, 2, 2);
        ro.fill(7.0);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_approx(freq, uneven, ro, &axis), std::invalid_argument);
        CHECK(ro.n_slices == 2);
        CHECK(ro(1, 1, 1) == 7.0);
        CHECK(axis.n_elem == 2);
    }
}

TEST_CASE("sos_acf_approx - the axis output may be the same object as dist")
{
    arma::mat freq = {{0.03, 0.05, -0.02}, {-0.07, 0.01, 0.04}};
    arma::vec dist = {0.0, 2.0, 4.0, 6.0}, ref_axis, d = dist;
    arma::cube ref, ro;
    quadriga_lib::sos_acf_approx(freq, dist, ref, &ref_axis);
    quadriga_lib::sos_acf_approx(freq, d, ro, &d);
    REQUIRE(d.n_elem == 13);
    CHECK(arma::approx_equal(d, ref_axis, "absdiff", 0.0));
    CHECK(arma::approx_equal(ro, ref, "absdiff", 0.0));
}

TEST_CASE("sos_acf_approx - the planes are point-symmetric to the last bit")
{
    // R(-dx) = R(dx). The half of each plane beyond the center column is the mirror image of the other
    // half, so the symmetry is exact, and the mirrored values still match the oracle.
    arma::mat freq = {{0.031, 0.052, -0.023}, {-0.071, 0.013, 0.044}, {0.017, -0.066, 0.029}, {0.11, 0.07, -0.13}};
    arma::vec dist = arma::regspace(0.0, 0.7, 6.3), axis;
    for (arma::uword n_dim = 2; n_dim <= 3; ++n_dim)
    {
        arma::mat f = freq.cols(0, n_dim - 1);
        arma::cube ro;
        quadriga_lib::sos_acf_approx(f, dist, ro, &axis);
        const arma::uword n = ro.n_rows;
        REQUIRE(n == 37);
        REQUIRE(ro.n_cols == 37);
        arma::uword n_asym = 0;
        double max_err = 0.0;
        const arma::uword first[3] = {0, 0, 1}, second[3] = {1, 2, 2};
        for (arma::uword s = 0; s < ro.n_slices; ++s)
            for (arma::uword c = 0; c < n; ++c)
                for (arma::uword r = 0; r < n; ++r)
                {
                    n_asym += (ro(r, c, s) == ro(n - 1 - r, n - 1 - c, s)) ? 0 : 1;
                    max_err = std::max(max_err, std::abs(ro(r, c, s) - ref_acf(f, first[s], axis(c), second[s], axis(r))));
                }
        INFO("n_dim " << n_dim);
        CHECK(n_asym == 0);
        CHECK(max_err < 1e-12);
        CHECK(std::abs(ro(3, 30, 0) - ro(3, 6, 0)) > 1e-3); // Point symmetry is not a mirror along one axis
    }
}
