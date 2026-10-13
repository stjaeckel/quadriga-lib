// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::sos_init_phases.
//
// Pins down: the documented random stream (so that seeds are portable), the value range, the
// column-wise fill order, sizes, float and double, the seed -1 path, input validation.
// Oracle: the stream is specified as std::mt19937_64 with the upper 53 bits of each output, which is
// reproduced here with the standard generator (its output sequence is fixed by the C++ standard).
// Statistics use the moments of the uniform distribution on [-pi, pi): mean 0, variance pi^2/3.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_sos.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

// Diagnostic output. Comment out to silence the tables. The assertions run either way.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

namespace
{
    const double pi = 3.141592653589793;

    // Oracle: phase = 2·pi · ( u - 0.5 ), u = ( output >> 11 ) / 2^53, filled column by column
    arma::mat ref_phases(arma::uword n_coeff, arma::uword n_gen, unsigned long long seed)
    {
        std::mt19937_64 rng(seed);
        arma::mat p(n_coeff, n_gen);
        for (arma::uword i = 0; i < p.n_elem; ++i)
            p(i) = 2.0 * pi * ((double)(rng() >> 11) / 9007199254740992.0 - 0.5);
        return p;
    }
}

TEST_CASE("sos_init_phases - phases follow the documented random stream")
{
    arma::mat phase;
    quadriga_lib::sos_init_phases(7, 3, phase, 42);
    REQUIRE(phase.n_rows == 7);
    REQUIRE(phase.n_cols == 3);

    arma::mat ref = ref_phases(7, 3, 42);
    QD_DIAG("\n# diag: sos_init_phases, seed 42, first column\n");
    QD_DIAG("row,phase,ref,diff\n");
    for (arma::uword i = 0; i < 7; ++i)
        QD_DIAG("%llu,%.15f,%.15f,%.3e\n", (unsigned long long)i, phase(i, 0), ref(i, 0), phase(i, 0) - ref(i, 0));

    // Same arithmetic in double, one multiplication and one subtraction: exact up to the last bit
    CHECK(arma::approx_equal(phase, ref, "absdiff", 1e-15));
    CHECK(arma::abs(ref).max() > 1.0); // The oracle itself is not degenerate
}

TEST_CASE("sos_init_phases - the first columns do not depend on the number of generators")
{
    arma::mat a, b;
    quadriga_lib::sos_init_phases(11, 3, a, 7);
    quadriga_lib::sos_init_phases(11, 5, b, 7);
    REQUIRE(b.n_cols == 5);
    CHECK(arma::approx_equal(a, b.cols(0, 2), "absdiff", 0.0));
    CHECK(arma::abs(b.col(3) - b.col(0)).max() > 0.1); // Later columns are new values
}

TEST_CASE("sos_init_phases - a seed is reproducible and different seeds differ")
{
    arma::mat a, b, c;
    quadriga_lib::sos_init_phases(50, 4, a, 1);
    quadriga_lib::sos_init_phases(50, 4, b, 1);
    quadriga_lib::sos_init_phases(50, 4, c, 2);
    CHECK(arma::approx_equal(a, b, "absdiff", 0.0));
    CHECK(arma::abs(a - c).max() > 0.1);

    // Seed 0 is a regular seed
    arma::mat z;
    quadriga_lib::sos_init_phases(50, 4, z, 0);
    CHECK(arma::approx_equal(z, ref_phases(50, 4, 0), "absdiff", 1e-15));
}

TEST_CASE("sos_init_phases - seed -1 draws from the system random device")
{
    arma::mat a, b;
    quadriga_lib::sos_init_phases(64, 2, a);
    quadriga_lib::sos_init_phases(64, 2, b);
    REQUIRE(a.n_elem == 128);
    CHECK(arma::abs(a - b).max() > 1e-3);
    CHECK(a.min() >= -pi);
    CHECK(a.max() < pi);
}

TEST_CASE("sos_init_phases - values are uniform in [-pi, pi)")
{
    const arma::uword n = 200000;
    arma::mat phase;
    quadriga_lib::sos_init_phases(1000, 200, phase, 3);
    REQUIRE(phase.n_elem == n);

    CHECK(phase.min() >= -pi);
    CHECK(phase.max() < pi);

    double mean = arma::accu(phase) / (double)n;
    double var = 0.0;
    for (arma::uword i = 0; i < n; ++i)
        var += phase(i) * phase(i) / (double)n;
    var -= mean * mean;

    // Standard deviation of the sample mean: (2·pi/sqrt(12)) / sqrt(n) = 0.0041; of the sample variance: 0.0066
    QD_DIAG("\n# diag: sos_init_phases moments, n = %llu\n", (unsigned long long)n);
    QD_DIAG("quantity,value,expected,sigma\n");
    QD_DIAG("mean,%.5f,0,%.5f\n", mean, 2.0 * pi / std::sqrt(12.0 * (double)n));
    QD_DIAG("variance,%.5f,%.5f,%.5f\n", var, pi * pi / 3.0, 0.0066);
    CHECK(std::abs(mean) < 5.0 * 0.0041);
    CHECK(std::abs(var - pi * pi / 3.0) < 5.0 * 0.0066);

    // 10 bins, expected share 0.1, standard deviation sqrt(0.1·0.9/n) = 0.00067
    arma::vec hist(10, arma::fill::zeros);
    for (arma::uword i = 0; i < n; ++i)
    {
        arma::uword b = (arma::uword)std::floor((phase(i) + pi) / (2.0 * pi) * 10.0);
        hist(b < 10 ? b : 9) += 1.0 / (double)n;
    }
    QD_DIAG("# diag: sos_init_phases histogram\nbin,share\n");
    for (arma::uword b = 0; b < 10; ++b)
    {
        QD_DIAG("%llu,%.5f\n", (unsigned long long)b, hist(b));
        INFO("bin " << b);
        CHECK(std::abs(hist(b) - 0.1) < 5.0 * 0.00067);
    }
}

TEST_CASE("sos_init_phases - a pre-sized output is filled and a wrong size is replaced")
{
    arma::mat ref = ref_phases(5, 2, 9);

    arma::mat sized(5, 2);
    sized.fill(99.0);
    const double *ptr = sized.memptr();
    quadriga_lib::sos_init_phases(5, 2, sized, 9);
    CHECK(sized.memptr() == ptr); // Written in place
    CHECK(arma::approx_equal(sized, ref, "absdiff", 1e-15));

    arma::mat wrong(3, 7);
    wrong.fill(99.0);
    quadriga_lib::sos_init_phases(5, 2, wrong, 9);
    REQUIRE(wrong.n_rows == 5);
    REQUIRE(wrong.n_cols == 2);
    CHECK(arma::approx_equal(wrong, ref, "absdiff", 1e-15));
}

TEST_CASE("sos_init_phases - float values are the rounded double values and stay below pi")
{
    arma::fmat pf;
    arma::mat pd;
    quadriga_lib::sos_init_phases(1000, 100, pf, 5);
    quadriga_lib::sos_init_phases(1000, 100, pd, 5);
    REQUIRE(pf.n_rows == 1000);
    REQUIRE(pf.n_cols == 100);

    // Rounding to float changes a value of magnitude up to pi by at most half a float spacing, 1.2e-7
    CHECK(arma::abs(arma::conv_to<arma::mat>::from(pf) - pd).max() < 2.5e-7);
    CHECK(pf.min() >= -(float)pi);
    CHECK(pf.max() < (float)pi);
    CHECK(arma::abs(pf).max() > 3.0f);
}

TEST_CASE("sos_init_phases - input validation")
{
    arma::mat phase;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_init_phases(1, 1, phase, 0));
        CHECK(phase.n_elem == 1);
        CHECK_NOTHROW(quadriga_lib::sos_init_phases(4, 2, phase));
    }
    SECTION("n_coeff")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_init_phases(0, 1, phase, 0), std::invalid_argument);
    }
    SECTION("n_gen")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_init_phases(4, 0, phase, 0), std::invalid_argument);
    }
    SECTION("seed")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_init_phases(4, 2, phase, -2), std::invalid_argument);
    }
}
