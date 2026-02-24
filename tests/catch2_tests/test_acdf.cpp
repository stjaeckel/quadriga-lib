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
