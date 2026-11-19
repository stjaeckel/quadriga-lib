// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_SOS=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_SOS=1 ./build_linux/test_bin "sos_acfi*"

// Verification suite for quadriga_lib::sos_acfi.
//
// Pins down: linear interpolation on an unequally spaced table, the clipping beyond the last sample
// point, sizes, float and double, input validation.
// Oracle: values computed by hand for a 3-point table, acf = {1, 0.5, 0.2} at dist = {0, 1, 3} m.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_sos.hpp"

#include <cmath>
#include <stdexcept>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_SOS")

TEST_CASE("sos_acfi - linear interpolation and clipping match the hand-computed values")
{
    arma::vec acf = {1.0, 0.5, 0.2}, dist = {0.0, 1.0, 3.0};

    // Segment 1: 1 - 0.5·d; segment 2: 0.5 - 0.15·(d - 1); beyond 3 m: 0.2
    arma::vec d = {0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.0001, 10.0, 1e9};
    arma::vec ref = {1.0, 0.75, 0.5, 0.35, 0.275, 0.2, 0.2, 0.2, 0.2};

    arma::vec val;
    val = quadriga_lib::sos_acfi(acf, dist, d);
    REQUIRE(val.n_elem == d.n_elem);

    QD_DIAG("\n# diag: sos_acfi hand table\n");
    QD_DIAG("d,val,ref,diff\n");
    for (arma::uword i = 0; i < d.n_elem; ++i)
        QD_DIAG("%g,%.12f,%.12f,%.3e\n", d(i), val(i), ref(i), val(i) - ref(i));

    CHECK(arma::approx_equal(val, ref, "absdiff", 1e-14));
}

TEST_CASE("sos_acfi - sizes and empty query")
{
    arma::vec acf = {1.0, 0.5, 0.2}, dist = {0.0, 1.0, 3.0};

    arma::vec none;
    const arma::vec empty = quadriga_lib::sos_acfi(acf, dist, none);
    CHECK(empty.n_elem == 0);

    arma::vec d = {2.0, 0.5};
    const arma::vec val = quadriga_lib::sos_acfi(acf, dist, d);
    REQUIRE(val.n_elem == 2);
    CHECK(std::abs(val(0) - 0.35) < 1e-14);
    CHECK(std::abs(val(1) - 0.75) < 1e-14);
}

TEST_CASE("sos_acfi - float agrees with double")
{
    arma::fvec acf = {1.0f, 0.5f, 0.2f}, dist = {0.0f, 1.0f, 3.0f}, d = {0.0f, 0.5f, 2.0f, 7.0f}, val;
    val = quadriga_lib::sos_acfi(acf, dist, d);
    REQUIRE(val.n_elem == 4);
    arma::fvec ref = {1.0f, 0.75f, 0.35f, 0.2f};
    CHECK(arma::approx_equal(val, ref, "absdiff", 1e-6f));
}

TEST_CASE("sos_acfi - input validation")
{
    arma::vec acf = {1.0, 0.5, 0.2}, dist = {0.0, 1.0, 3.0}, d = {0.5}, val;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(val = quadriga_lib::sos_acfi(acf, dist, d));
    }
    SECTION("acf")
    {
        arma::vec bad = {1.0};
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(bad, dist, d), std::invalid_argument);
        arma::vec empty;
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(empty, dist, d), std::invalid_argument);
    }
    SECTION("dist")
    {
        arma::vec size = {0.0, 1.0};
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(acf, size, d), std::invalid_argument);
        arma::vec start = {0.5, 1.0, 3.0};
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(acf, start, d), std::invalid_argument);
        arma::vec order = {0.0, 3.0, 1.0};
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(acf, order, d), std::invalid_argument);
        arma::vec twice = {0.0, 1.0, 1.0};
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(acf, twice, d), std::invalid_argument);
    }
    SECTION("d")
    {
        arma::vec neg = {0.5, -0.1}, nan = {0.5, std::nan("")};
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(acf, dist, neg), std::invalid_argument);
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(acf, dist, nan), std::invalid_argument);
    }
    SECTION("non-finite table")
    {
        arma::vec inf_dist = {0.0, 1.0, arma::datum::inf}, nan_dist = {0.0, std::nan(""), 3.0}, nan_acf = {1.0, std::nan(""), 0.2};
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(acf, inf_dist, d), std::invalid_argument);
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(acf, nan_dist, d), std::invalid_argument);
        CHECK_THROWS_AS(val = quadriga_lib::sos_acfi(nan_acf, dist, d), std::invalid_argument);
    }
}
