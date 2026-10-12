// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::sos_acfi.
//
// Pins down: linear interpolation on an unequally spaced table, the clipping beyond the last sample
// point, sizes, float and double, input validation.
// Oracle: values computed by hand for a 3-point table, acf = {1, 0.5, 0.2} at dist = {0, 1, 3} m.

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

TEST_CASE("sos_acfi - linear interpolation and clipping match the hand-computed values")
{
    arma::vec acf = {1.0, 0.5, 0.2}, dist = {0.0, 1.0, 3.0};

    // Segment 1: 1 - 0.5·d; segment 2: 0.5 - 0.15·(d - 1); beyond 3 m: 0.2
    arma::vec d = {0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.0001, 10.0, 1e9};
    arma::vec ref = {1.0, 0.75, 0.5, 0.35, 0.275, 0.2, 0.2, 0.2, 0.2};

    arma::vec val;
    quadriga_lib::sos_acfi(acf, dist, d, val);
    REQUIRE(val.n_elem == d.n_elem);

    QD_DIAG("\n# diag: sos_acfi hand table\n");
    QD_DIAG("d,val,ref,diff\n");
    for (arma::uword i = 0; i < d.n_elem; ++i)
        QD_DIAG("%g,%.12f,%.12f,%.3e\n", d(i), val(i), ref(i), val(i) - ref(i));

    CHECK(arma::approx_equal(val, ref, "absdiff", 1e-14));
}

TEST_CASE("sos_acfi - sizes, empty query and pre-sized output")
{
    arma::vec acf = {1.0, 0.5, 0.2}, dist = {0.0, 1.0, 3.0};

    arma::vec val(4);
    val.fill(9.0);
    arma::vec none;
    quadriga_lib::sos_acfi(acf, dist, none, val);
    CHECK(val.n_elem == 0);

    arma::vec d = {2.0, 0.5};
    arma::vec sized(2);
    sized.fill(9.0);
    const double *ptr = sized.memptr();
    quadriga_lib::sos_acfi(acf, dist, d, sized);
    CHECK(sized.memptr() == ptr);
    CHECK(std::abs(sized(0) - 0.35) < 1e-14);
    CHECK(std::abs(sized(1) - 0.75) < 1e-14);
}

TEST_CASE("sos_acfi - float agrees with double")
{
    arma::fvec acf = {1.0f, 0.5f, 0.2f}, dist = {0.0f, 1.0f, 3.0f}, d = {0.0f, 0.5f, 2.0f, 7.0f}, val;
    quadriga_lib::sos_acfi(acf, dist, d, val);
    REQUIRE(val.n_elem == 4);
    arma::fvec ref = {1.0f, 0.75f, 0.35f, 0.2f};
    CHECK(arma::approx_equal(val, ref, "absdiff", 1e-6f));
}

TEST_CASE("sos_acfi - input validation")
{
    arma::vec acf = {1.0, 0.5, 0.2}, dist = {0.0, 1.0, 3.0}, d = {0.5}, val;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_acfi(acf, dist, d, val));
    }
    SECTION("acf")
    {
        arma::vec bad = {1.0};
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(bad, dist, d, val), std::invalid_argument);
        arma::vec empty;
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(empty, dist, d, val), std::invalid_argument);
    }
    SECTION("dist")
    {
        arma::vec size = {0.0, 1.0};
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(acf, size, d, val), std::invalid_argument);
        arma::vec start = {0.5, 1.0, 3.0};
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(acf, start, d, val), std::invalid_argument);
        arma::vec order = {0.0, 3.0, 1.0};
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(acf, order, d, val), std::invalid_argument);
        arma::vec twice = {0.0, 1.0, 1.0};
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(acf, twice, d, val), std::invalid_argument);
    }
    SECTION("d")
    {
        arma::vec neg = {0.5, -0.1}, nan = {0.5, std::nan("")};
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(acf, dist, neg, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(acf, dist, nan, val), std::invalid_argument);
    }
    SECTION("non-finite table")
    {
        arma::vec inf_dist = {0.0, 1.0, arma::datum::inf}, nan_dist = {0.0, std::nan(""), 3.0}, nan_acf = {1.0, std::nan(""), 0.2};
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(acf, inf_dist, d, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(acf, nan_dist, d, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acfi(nan_acf, dist, d, val), std::invalid_argument);
    }
}

TEST_CASE("sos_acfi - the output may be the same object as an input")
{
    // 40 table samples, 4000 distances: the output is resized when it is the table, and must then
    // still be evaluated with the table as it was given
    arma::vec dist = arma::regspace(0.0, 0.5, 19.5), acf = arma::exp(-dist / 4.0);
    arma::vec d(4000);
    for (arma::uword i = 0; i < d.n_elem; ++i)
        d(i) = 0.006 * (double)i; // 0 to 24 m, beyond the table
    arma::vec ref;
    quadriga_lib::sos_acfi(acf, dist, d, ref);
    REQUIRE(ref.n_elem == 4000);
    CHECK(std::abs(ref(1000) - std::exp(-6.0 / 4.0)) < 1e-12); // 6 m is a sample point
    CHECK(std::abs(ref(3999) - acf(39)) < 1e-14);

    arma::vec a = acf, b = dist, c = d;
    quadriga_lib::sos_acfi(a, dist, d, a); // val is 'acf'
    quadriga_lib::sos_acfi(acf, b, d, b);  // val is 'dist'
    quadriga_lib::sos_acfi(acf, dist, c, c); // val is 'd'
    REQUIRE(a.n_elem == 4000);
    REQUIRE(b.n_elem == 4000);
    REQUIRE(c.n_elem == 4000);
    CHECK(arma::approx_equal(a, ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(b, ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(c, ref, "absdiff", 0.0));
}
