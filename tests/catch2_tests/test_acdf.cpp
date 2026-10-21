// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_tools.hpp"

#include <cmath>

TEST_CASE("ACDF - Basic single column")
{
    // Simple data: sorted values 0 to 9
    arma::vec data_col = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    arma::mat data(10, 1);
    data.col(0) = data_col;

    arma::vec bins;
    arma::mat Sh;
    arma::vec Sc, mu, sig;

    quadriga_lib::acdf(data, &bins, &Sh, &Sc, &mu, &sig);

    // Bins should be auto-generated with 201 points from 0 to 9
    REQUIRE(bins.n_elem == 201);
    CHECK(std::abs(bins(0) - 0.0) < 1e-10);
    CHECK(std::abs(bins(200) - 9.0) < 1e-10);

    // Sh should be [201, 1]
    REQUIRE(Sh.n_rows == 201);
    REQUIRE(Sh.n_cols == 1);

    // CDF should start near 0.1 (first bin contains value 0) and end at 1.0
    CHECK(Sh(0, 0) > 0.0);
    CHECK(std::abs(Sh(200, 0) - 1.0) < 1e-10);

    // CDF should be monotonically non-decreasing
    for (arma::uword i = 1; i < 201; ++i)
        CHECK(Sh(i, 0) >= Sh(i - 1, 0));

    // Sc should equal Sh for single column
    REQUIRE(Sc.n_elem == 201);
    CHECK(arma::approx_equal(Sc, Sh.col(0), "absdiff", 1e-10));

    // mu should have 9 elements
    REQUIRE(mu.n_elem == 9);

    // sig should be all zeros for single column
    REQUIRE(sig.n_elem == 9);
    CHECK(arma::approx_equal(sig, arma::vec(9, arma::fill::zeros), "absdiff", 1e-10));

    // mu(4) should be approximately the median (4.5)
    CHECK(std::abs(mu(4) - 4.5) < 1.0);
}

TEST_CASE("ACDF - Multiple columns")
{
    // Two identical columns should give the same result as one column
    arma::mat data(100, 2);
    for (arma::uword i = 0; i < 100; ++i)
    {
        data(i, 0) = (double)i;
        data(i, 1) = (double)i;
    }

    arma::vec bins;
    arma::mat Sh;
    arma::vec Sc, mu, sig;

    quadriga_lib::acdf(data, &bins, &Sh, &Sc, &mu, &sig);

    REQUIRE(Sh.n_rows == 201);
    REQUIRE(Sh.n_cols == 2);

    // Both columns should have the same CDF
    CHECK(arma::approx_equal(Sh.col(0), Sh.col(1), "absdiff", 1e-10));

    // Averaged CDF should be close to 1.0 at the last bin
    // (quantile grid 0..0.999 means mapped CDF tops at ~0.99)
    REQUIRE(Sc.n_elem == 201);
    CHECK(std::abs(Sc(200) - 1.0) < 0.02);

    // sig reflects variation within the quantile averaging window, not across sets;
    // for identical columns, it is small but nonzero due to the ±5 sample window
    REQUIRE(sig.n_elem == 9);
    for (arma::uword i = 0; i < 9; ++i)
        CHECK(sig(i) < 1.0);

    REQUIRE(mu.n_elem == 9);
}

TEST_CASE("ACDF - Custom bins")
{
    arma::mat data(100, 1);
    for (arma::uword i = 0; i < 100; ++i)
        data(i, 0) = (double)i;

    // Provide custom bins
    arma::vec bins = {0, 25, 50, 75, 99};
    arma::mat Sh;

    quadriga_lib::acdf(data, &bins, &Sh);

    REQUIRE(bins.n_elem == 5);
    REQUIRE(Sh.n_rows == 5);
    REQUIRE(Sh.n_cols == 1);

    // CDF at last bin should be 1.0
    CHECK(std::abs(Sh(4, 0) - 1.0) < 1e-10);

    // CDF should be non-decreasing
    for (arma::uword i = 1; i < 5; ++i)
        CHECK(Sh(i, 0) >= Sh(i - 1, 0));
}

TEST_CASE("ACDF - Handles Inf and NaN")
{
    arma::mat data(12, 1);
    data.col(0) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   arma::datum::inf, arma::datum::nan};

    arma::vec bins;
    arma::mat Sh;

    quadriga_lib::acdf(data, &bins, &Sh);

    // Should have 201 bins spanning 0 to 9 (Inf/NaN excluded from range)
    REQUIRE(bins.n_elem == 201);
    CHECK(std::abs(bins(0) - 0.0) < 1e-10);
    CHECK(std::abs(bins(200) - 9.0) < 1e-10);

    // CDF should sum to 1.0 at the last bin (only 10 valid samples)
    CHECK(std::abs(Sh(200, 0) - 1.0) < 1e-10);
}

TEST_CASE("ACDF - Float precision")
{
    arma::fmat data(50, 2);
    for (arma::uword i = 0; i < 50; ++i)
    {
        data(i, 0) = (float)i;
        data(i, 1) = (float)(i + 1);
    }

    arma::fvec bins;
    arma::fmat Sh;
    arma::fvec Sc, mu, sig;

    quadriga_lib::acdf(data, &bins, &Sh, &Sc, &mu, &sig);

    REQUIRE(bins.n_elem == 201);
    REQUIRE(Sh.n_rows == 201);
    REQUIRE(Sh.n_cols == 2);
    REQUIRE(Sc.n_elem == 201);
    REQUIRE(mu.n_elem == 9);
    REQUIRE(sig.n_elem == 9);

    CHECK(std::abs(Sh(200, 0) - 1.0f) < 1e-5f);
    CHECK(std::abs(Sh(200, 1) - 1.0f) < 1e-5f);
}

TEST_CASE("ACDF - Optional outputs")
{
    arma::mat data(100, 3);
    for (arma::uword i = 0; i < 100; ++i)
        for (arma::uword j = 0; j < 3; ++j)
            data(i, j) = (double)i + (double)j;

    // Only request bins and Sh
    arma::vec bins;
    arma::mat Sh;

    quadriga_lib::acdf(data, &bins, &Sh);

    REQUIRE(bins.n_elem == 201);
    REQUIRE(Sh.n_rows == 201);
    REQUIRE(Sh.n_cols == 3);

    // Request only Sc
    arma::vec Sc;
    quadriga_lib::acdf(data, (arma::vec *)nullptr, (arma::mat *)nullptr, &Sc);

    REQUIRE(Sc.n_elem == 201);
    CHECK(std::abs(Sc(200) - 1.0) < 1e-6);
}

TEST_CASE("ACDF - Custom n_bins")
{
    arma::mat data(100, 1);
    for (arma::uword i = 0; i < 100; ++i)
        data(i, 0) = (double)i;

    arma::vec bins;
    arma::mat Sh;

    quadriga_lib::acdf(data, &bins, &Sh, (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 51);

    REQUIRE(bins.n_elem == 51);
    REQUIRE(Sh.n_rows == 51);
}

TEST_CASE("ACDF - Error cases")
{
    // Empty data
    arma::mat empty_data;
    CHECK_THROWS_AS(quadriga_lib::acdf(empty_data), std::invalid_argument);

    // n_bins too small
    arma::mat data(10, 1);
    data.col(0) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    CHECK_THROWS_AS(quadriga_lib::acdf(data, (arma::vec *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1),
                    std::invalid_argument);

    // All Inf/NaN data
    arma::mat bad_data(5, 1);
    bad_data.fill(arma::datum::nan);
    CHECK_THROWS_AS(quadriga_lib::acdf(bad_data), std::invalid_argument);
}

TEST_CASE("ACDF - Constant data")
{
    // All values are the same
    arma::mat data(100, 1);
    data.fill(5.0);

    arma::vec bins;
    arma::mat Sh;

    quadriga_lib::acdf(data, &bins, &Sh);

    REQUIRE(bins.n_elem == 201);
    // All data should be in one bin, CDF should jump from 0 to 1
    CHECK(std::abs(Sh(200, 0) - 1.0) < 1e-10);
}

TEST_CASE("ACDF - Quantile correctness")
{
    // Uniform data from 0 to 999
    arma::mat data(1000, 1);
    for (arma::uword i = 0; i < 1000; ++i)
        data(i, 0) = (double)i;

    arma::vec bins;
    arma::vec mu;

    quadriga_lib::acdf(data, &bins, (arma::mat *)nullptr, (arma::vec *)nullptr, &mu);

    REQUIRE(mu.n_elem == 9);

    // For uniform [0, 999], the p-th quantile should be approximately p * 999
    for (arma::uword q = 0; q < 9; ++q)
    {
        double expected = (double)(q + 1) * 0.1 * 999.0;
        CHECK(std::abs(mu(q) - expected) < 10.0); // Allow some binning tolerance
    }
}

TEST_CASE("ACDF - strictly increasing")
{
    auto check_monotonic = [](const arma::vec &b) {
        REQUIRE(b.n_elem == 201);
        for (arma::uword i = 1; i < b.n_elem; ++i)
            REQUIRE(b(i) > b(i - 1));
    };

    SECTION("Constant negative data")
    {
        arma::mat data(100, 3, arma::fill::value(-5.0));
        arma::vec bins;
        quadriga_lib::acdf(data, &bins);
        check_monotonic(bins);
        REQUIRE(bins.front() < -5.0);
        REQUIRE(bins.back()  > -5.0);
    }

    SECTION("Constant zero data")
    {
        arma::mat data(100, 2, arma::fill::zeros);
        arma::vec bins;
        quadriga_lib::acdf(data, &bins);
        check_monotonic(bins);
        REQUIRE(bins.front() < 0.0);
        REQUIRE(bins.back()  > 0.0);
    }

    SECTION("Constant positive data")
    {
        arma::mat data(100, 2, arma::fill::value(7.5));
        arma::vec bins;
        quadriga_lib::acdf(data, &bins);
        check_monotonic(bins);
        REQUIRE(bins.front() < 7.5);
        REQUIRE(bins.back()  > 7.5);
    }

    SECTION("Normal random data")
    {
        arma::arma_rng::set_seed(42);
        arma::mat data = arma::randn<arma::mat>(1000, 4);
        arma::vec bins;
        quadriga_lib::acdf(data, &bins);
        check_monotonic(bins);
    }
}

TEST_CASE("ACDF - Data sets without finite values are excluded from the averages")
{
    arma::mat data(100, 2);
    for (int i = 0; i < 100; ++i)
        data(i, 0) = i, data(i, 1) = NAN; // the second set has no finite value
    arma::vec bins, ca, mu, sig;
    arma::mat cps;
    quadriga_lib::acdf<double>(data, &bins, &cps, &ca, &mu, &sig);

    arma::mat d1 = data.col(0);
    arma::vec b1 = bins, ca1, mu1, sig1;
    arma::mat cps1;
    quadriga_lib::acdf<double>(d1, &b1, &cps1, &ca1, &mu1, &sig1);

    CHECK(std::abs(mu1(4) - 49.005) < 1e-12);
    CHECK(arma::approx_equal(mu, mu1, "absdiff", 0.0)); // median 49.005 (74.2275 with the bias)
    CHECK(arma::approx_equal(sig, sig1, "absdiff", 0.0));
    CHECK(arma::approx_equal(ca, ca1, "absdiff", 0.0));
    CHECK(arma::approx_equal(cps.col(0), cps1.col(0), "absdiff", 0.0));
    CHECK(cps.col(1).has_nan());
    CHECK(arma::all(arma::vec(cps.col(1)) != arma::vec(cps.col(1)))); // all NaN

    // Three valid sets and two empty ones (Inf counts as not finite) average like the three valid sets alone
    arma::mat d5 = arma::randn(200, 5), d3(200, 3);
    d5.col(1).fill(NAN), d5.col(3).fill(INFINITY);
    d3.col(0) = d5.col(0), d3.col(1) = d5.col(2), d3.col(2) = d5.col(4);
    arma::vec b5 = arma::linspace(-3.0, 3.0, 101), b3 = b5, ca5, mu5, sig5, ca3, mu3, sig3;
    arma::mat c5, c3;
    quadriga_lib::acdf<double>(d5, &b5, &c5, &ca5, &mu5, &sig5);
    quadriga_lib::acdf<double>(d3, &b3, &c3, &ca3, &mu3, &sig3);
    CHECK(arma::approx_equal(ca5, ca3, "absdiff", 0.0));
    CHECK(arma::approx_equal(mu5, mu3, "absdiff", 0.0));
    CHECK(arma::approx_equal(sig5, sig3, "absdiff", 0.0));
    CHECK(arma::approx_equal(c5.col(2), c3.col(1), "absdiff", 0.0));
    CHECK(c5.col(3).has_nan());

    // No set has data (only possible with user bins): the averaged outputs are NaN
    for (arma::uword n_sets : {1ULL, 3ULL})
    {
        arma::mat dn(10, n_sets);
        dn.fill(NAN);
        arma::vec bn = {0.0, 1.0, 2.0}, can, mun, sign;
        arma::mat cn;
        quadriga_lib::acdf<double>(dn, &bn, &cn, &can, &mun, &sign);
        INFO("n_sets " << n_sets);
        REQUIRE(cn.n_rows == 3);
        REQUIRE(cn.n_cols == n_sets);
        REQUIRE(can.n_elem == 3);
        REQUIRE(mun.n_elem == 9);
        REQUIRE(sign.n_elem == 9);
        CHECK(arma::all(arma::vectorise(cn) != arma::vectorise(cn)));
        CHECK(arma::all(can != can));
        CHECK(arma::all(mun != mun));
        CHECK(arma::all(sign != sign));
    }
}

TEST_CASE("ACDF - User bins must be finite and strictly ascending")
{
    arma::mat data(100, 1);
    for (int i = 0; i < 100; ++i)
        data(i, 0) = i;
    arma::mat cdf;
    for (arma::vec ub : {arma::vec({50.0, 0.0, 99.0, 25.0}), arma::vec({0.0, 25.0, 25.0, 99.0}), arma::vec({0.0, NAN, 99.0}),
                         arma::vec({0.0, 50.0, INFINITY}), arma::vec({-INFINITY, 50.0})})
    {
        INFO("bins " << ub.t());
        CHECK_THROWS_AS(quadriga_lib::acdf<double>(data, &ub, &cdf), std::invalid_argument);
    }
    arma::vec ok = {0.0, 25.0, 50.0, 99.0}, one = {10.0};
    CHECK_NOTHROW(quadriga_lib::acdf<double>(data, &ok, &cdf));
    CHECK_NOTHROW(quadriga_lib::acdf<double>(data, &one, &cdf));
    CHECK(cdf(0) == 1.0);
}
