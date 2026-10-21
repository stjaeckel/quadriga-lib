// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_SOS=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_SOS=1 ./build_linux/test_bin "sos_generate*"

// Verification suite for quadriga_lib::sos_generate.
//
// Pins down: that the optimiser recovers a target it can represent exactly, that the reported error is
// the error of the returned table, that the error never increases, the frequency limits, the
// directions of the sinusoids, reproducibility, sizes, float and double, input validation.
// It leaves the quality of the shipped tables to test_sos_preset.
// Oracles, written in scalar double math in this file:
//  - recovery: the 1-D ACF of two known frequencies must be reproduced by two sinusoids
//  - the mean squared error over the documented test directions, evaluated from the returned table
//  - invariants: error sequence non-increasing, frequency magnitudes inside the limits

#include <catch2/catch_test_macros.hpp>

#include "quadriga_sos.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_SOS")

namespace
{
    const double pi = 3.141592653589793;

    // Uniform random matrix in [lo, hi), own generator so that the fixtures do not depend on the library
    arma::mat rnd(arma::uword n_rows, arma::uword n_cols, double lo, double hi, unsigned seed)
    {
        std::mt19937_64 rng(seed);
        arma::mat m(n_rows, n_cols);
        for (arma::uword i = 0; i < m.n_elem; ++i)
            m(i) = lo + (hi - lo) * ((double)(rng() >> 11) / 9007199254740992.0);
        return m;
    }

    // Oracle: documented test directions as unit vectors, [n_dir, 3]
    // n_dim = 3: sin(elevation) = (i + 0.5) / n_dir, azimuth = i · pi · (3 - sqrt(5))
    // n_dim = 2: azimuth = pi · i / n_dir; n_dim = 1: the x-axis
    arma::mat ref_dirs(arma::uword n_dir, arma::uword n_dim)
    {
        arma::uword n = (n_dim == 1) ? 1 : n_dir;
        arma::mat u(n, 3, arma::fill::zeros);
        for (arma::uword i = 0; i < n; ++i)
        {
            double az = 0.0, el = 0.0;
            if (n_dim == 3)
                el = std::asin(((double)i + 0.5) / (double)n), az = (double)i * pi * (3.0 - std::sqrt(5.0));
            else if (n_dim == 2)
                az = pi * (double)i / (double)n;
            u(i, 0) = std::cos(az) * std::cos(el), u(i, 1) = std::sin(az) * std::cos(el), u(i, 2) = std::sin(el);
        }
        return u;
    }

    // Oracle: ACF of the sinusoids at distance d along the unit vector in row t of u
    double ref_acf_dir(const arma::mat &freq, const arma::mat &u, arma::uword t, double d)
    {
        double acc = 0.0;
        for (arma::uword n = 0; n < freq.n_rows; ++n)
        {
            double fu = 0.0;
            for (arma::uword c = 0; c < freq.n_cols; ++c)
                fu += freq(n, c) * u(t, c);
            acc += std::cos(2.0 * pi * fu * d);
        }
        return acc / (double)freq.n_rows;
    }

    // Oracle: mean squared error along the test directions on the sample points, linear
    double ref_mse_dir(const arma::mat &freq, const arma::vec &acf, const arma::vec &dist, arma::uword n_dir)
    {
        arma::mat u = ref_dirs(n_dir, freq.n_cols);
        double acc = 0.0;
        for (arma::uword t = 0; t < u.n_rows; ++t)
            for (arma::uword s = 0; s < dist.n_elem; ++s)
            {
                double e = acf(s) - ref_acf_dir(freq, u, t, dist(s));
                acc += e * e;
            }
        return acc / (double)(u.n_rows * dist.n_elem);
    }
}

TEST_CASE("sos_generate - a 1-D target made of two sinusoids is recovered")
{
    // R(d) = 0.5 · ( cos( 2·pi · 0.04 · d ) + cos( 2·pi · 0.13 · d ) ), 2 and 6.5 periods inside 50 m
    arma::vec dist = arma::regspace(0.0, 0.5, 50.0), acf(dist.n_elem);
    for (arma::uword s = 0; s < dist.n_elem; ++s)
        acf(s) = 0.5 * (std::cos(2.0 * pi * 0.04 * dist(s)) + std::cos(2.0 * pi * 0.13 * dist(s)));

    QD_DIAG("\n# diag: sos_generate 1-D recovery\nseed,random_init,f_low,f_high,final_ase,passes\n");
    for (bool random_init : {true, false})
        for (arma::sword seed = 0; seed < 4; ++seed)
        {
            arma::mat freq, test_dir(2, 2);
            arma::vec ase;
            quadriga_lib::sos_generate(acf, dist, 2, freq, &ase, &test_dir, 1, true, 10, random_init, 0.0, 0.0, seed);
            REQUIRE(freq.n_rows == 2);
            REQUIRE(freq.n_cols == 1);
            REQUIRE(ase.n_elem >= 1);
            CHECK(test_dir.n_elem == 0);

            double f_low = std::min(std::abs(freq(0, 0)), std::abs(freq(1, 0)));
            double f_high = std::max(std::abs(freq(0, 0)), std::abs(freq(1, 0)));
            QD_DIAG("%lld,%d,%.7f,%.7f,%.3e,%llu\n", (long long)seed, (int)random_init, f_low, f_high, ase(ase.n_elem - 1), (unsigned long long)ase.n_elem);

            // The step refinement stops at a step of 1e-6 periods per 50 m, i.e. 2e-8 / m. The tolerance of
            // 1e-5 / m leaves room for the last accepted step; the error is then below 1e-6.
            INFO("seed " << seed << ", random_init " << random_init);
            CHECK(std::abs(f_low - 0.04) < 1e-5);
            CHECK(std::abs(f_high - 0.13) < 1e-5);
            CHECK(ase(ase.n_elem - 1) < 1e-6);
        }
}

TEST_CASE("sos_generate - the reported error is the error of the returned table and never increases")
{
    arma::vec dist = arma::regspace(0.0, 1.0, 40.0), acf = arma::exp(-dist / 10.0);

    QD_DIAG("\n# diag: sos_generate reported error\nn_dim,uniform_smp,random_init,passes,first_ase,final_ase,oracle\n");
    for (arma::uword n_dim = 1; n_dim <= 3; ++n_dim)
        for (bool uniform_smp : {true, false})
            for (bool random_init : {true, false})
            {
                arma::mat freq, test_dir;
                arma::vec ase;
                quadriga_lib::sos_generate(acf, dist, 24, freq, &ase, &test_dir, n_dim, uniform_smp, 7, random_init, 0.0, 0.0, 3);
                REQUIRE(freq.n_rows == 24);
                REQUIRE(freq.n_cols == n_dim);
                REQUIRE(ase.n_elem >= 1);

                double ref = ref_mse_dir(freq, acf, dist, 7), last = ase(ase.n_elem - 1);
                QD_DIAG("%llu,%d,%d,%llu,%.5e,%.5e,%.5e\n", (unsigned long long)n_dim, (int)uniform_smp, (int)random_init,
                        (unsigned long long)ase.n_elem, ase(0), last, ref);
                INFO("n_dim " << n_dim << ", uniform_smp " << uniform_smp << ", random_init " << random_init);
                CHECK(std::abs(last - ref) < 1e-10);
                CHECK(arma::abs(freq).max() > 1e-3);

                // 24 sinusoids reach a mean squared error below 1e-2 for an exponential ACF (MATLAB: 50
                // sinusoids within 0.1 to 0.15 absolute error)
                CHECK(last < 1e-2);
                for (arma::uword i = 1; i < ase.n_elem; ++i)
                    CHECK(ase(i) <= ase(i - 1));

                REQUIRE(test_dir.n_rows == (n_dim == 1 ? 0 : 7));
                REQUIRE(test_dir.n_cols == (n_dim == 1 ? 0 : n_dim - 1));
                if (n_dim > 1)
                {
                    arma::mat u = ref_dirs(7, n_dim);
                    double az = test_dir(4, 0), el = (n_dim == 3) ? test_dir(4, 1) : 0.0;
                    CHECK(std::abs(std::cos(az) * std::cos(el) - u(4, 0)) < 1e-12);
                    CHECK(std::abs(std::sin(el) - u(4, 2)) < 1e-12);
                }
            }
}

TEST_CASE("sos_generate - regular directions are the documented lattice")
{
    arma::vec dist = arma::regspace(0.0, 1.0, 40.0), acf = arma::exp(-dist / 10.0);

    for (arma::uword n_dim = 2; n_dim <= 3; ++n_dim)
    {
        arma::mat freq;
        quadriga_lib::sos_generate(acf, dist, 16, freq, (arma::vec *)nullptr, (arma::mat *)nullptr, n_dim, true, 5, true, 0.0, 0.0, 1);
        REQUIRE(freq.n_rows == 16);
        arma::mat u = ref_dirs(16, n_dim);
        double max_err = 0.0, min_len = 1e9;
        for (arma::uword n = 0; n < 16; ++n)
        {
            // Each row is a signed multiple of its direction: the cross product with the direction vanishes
            double f[3] = {freq(n, 0), freq(n, 1), n_dim == 3 ? freq(n, 2) : 0.0};
            double cx = f[1] * u(n, 2) - f[2] * u(n, 1), cy = f[2] * u(n, 0) - f[0] * u(n, 2), cz = f[0] * u(n, 1) - f[1] * u(n, 0);
            max_err = std::max(max_err, std::sqrt(cx * cx + cy * cy + cz * cz));
            min_len = std::min(min_len, std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]));
        }
        INFO("n_dim " << n_dim);
        CHECK(max_err < 1e-12);
        CHECK(arma::abs(freq).max() > 1e-3);
        (void)min_len;
    }
}

TEST_CASE("sos_generate - frequency limits are respected")
{
    arma::vec dist = arma::regspace(0.0, 1.0, 40.0), acf = arma::exp(-dist / 10.0);
    const double f_min = 0.02, f_max = 0.2; // Default upper limit: 15 / 40 = 0.375 / m

    for (bool random_init : {true, false})
    {
        arma::mat freq, def;
        quadriga_lib::sos_generate(acf, dist, 32, freq, (arma::vec *)nullptr, (arma::mat *)nullptr, 3, true, 7, random_init, f_min, f_max, 2);
        quadriga_lib::sos_generate(acf, dist, 32, def, (arma::vec *)nullptr, (arma::mat *)nullptr, 3, true, 7, random_init, 0.0, 0.0, 2);
        REQUIRE(freq.n_rows == 32);
        REQUIRE(def.n_rows == 32);
        double lo = 1e9, hi = 0.0, def_lo = 1e9, def_hi = 0.0;
        for (arma::uword n = 0; n < 32; ++n)
        {
            double len = std::sqrt(freq(n, 0) * freq(n, 0) + freq(n, 1) * freq(n, 1) + freq(n, 2) * freq(n, 2));
            double dl = std::sqrt(def(n, 0) * def(n, 0) + def(n, 1) * def(n, 1) + def(n, 2) * def(n, 2));
            lo = std::min(lo, len), hi = std::max(hi, len), def_lo = std::min(def_lo, dl), def_hi = std::max(def_hi, dl);
        }
        QD_DIAG("\n# diag: sos_generate frequency magnitudes, random_init %d: limited %.4f to %.4f, default %.4f to %.4f\n",
                (int)random_init, lo, hi, def_lo, def_hi);
        INFO("random_init " << random_init);
        CHECK(lo >= f_min - 1e-12);
        CHECK(hi <= f_max + 1e-12);
        CHECK(def_hi <= 0.375 + 1e-12);
        CHECK(def_lo < f_min); // Without the limit the exponential ACF draws frequencies towards zero
    }
}

TEST_CASE("sos_generate - a seed is reproducible, outputs are sized, float agrees with double")
{
    arma::vec dist = arma::regspace(0.0, 1.0, 40.0), acf = arma::exp(-dist / 10.0);

    arma::mat a, b, c;
    arma::vec ase_a, ase_b(7);
    quadriga_lib::sos_generate(acf, dist, 20, a, &ase_a, (arma::mat *)nullptr, 3, false, 7, true, 0.0, 0.0, 11);
    quadriga_lib::sos_generate(acf, dist, 20, b, &ase_b, (arma::mat *)nullptr, 3, false, 7, true, 0.0, 0.0, 11);
    quadriga_lib::sos_generate(acf, dist, 20, c, (arma::vec *)nullptr, (arma::mat *)nullptr, 3, false, 7, true, 0.0, 0.0, 12);
    REQUIRE(a.n_rows == 20);
    CHECK(arma::abs(a).max() > 1e-3);
    CHECK(arma::approx_equal(a, b, "absdiff", 0.0));
    CHECK(arma::approx_equal(ase_a, ase_b, "absdiff", 0.0));
    CHECK(arma::abs(a - c).max() > 1e-4);

    arma::mat sized(20, 3);
    sized.fill(9.0);
    const double *ptr = sized.memptr();
    quadriga_lib::sos_generate(acf, dist, 20, sized, (arma::vec *)nullptr, (arma::mat *)nullptr, 3, false, 7, true, 0.0, 0.0, 11);
    CHECK(sized.memptr() == ptr);
    CHECK(arma::approx_equal(sized, a, "absdiff", 0.0));

    // The computation is done in double for both types: the float result is the rounded double result
    arma::fvec af = arma::conv_to<arma::fvec>::from(acf), df = arma::conv_to<arma::fvec>::from(dist), asf;
    arma::fmat ff, tdf;
    quadriga_lib::sos_generate(af, df, 20, ff, &asf, &tdf, 3, false, 7, true, 0.0f, 0.0f, 11);
    REQUIRE(ff.n_rows == 20);
    REQUIRE(ff.n_cols == 3);
    REQUIRE(tdf.n_rows == 7);
    REQUIRE(asf.n_elem >= 1);
    // The float ACF differs from the double one by 6e-8, which can change the path of the search, so only
    // the quality of the result is compared: both reach a comparable error
    double e_f = ref_mse_dir(arma::conv_to<arma::mat>::from(ff), acf, dist, 7), e_d = ref_mse_dir(a, acf, dist, 7);
    QD_DIAG("\n# diag: sos_generate float %.4e, double %.4e\n", e_f, e_d);
    CHECK(e_f < 1e-2);
    CHECK(e_d < 1e-2);
    CHECK(std::abs((double)asf(asf.n_elem - 1) - e_f) < 1e-6);
}

TEST_CASE("sos_generate - input validation")
{
    arma::vec dist = arma::regspace(0.0, 1.0, 20.0), acf = arma::exp(-dist / 10.0);
    arma::mat freq;
    arma::vec *no_vec = nullptr;
    arma::mat *no_mat = nullptr;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 3, true, 5, true, 0.0, 0.0, 1));
        CHECK_NOTHROW(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 1, true, 5, true, 0.01, 0.3, 1));
        CHECK_NOTHROW(quadriga_lib::sos_generate(acf, dist, 2, freq)); // Defaults, random seed
        CHECK(freq.n_rows == 2);
        CHECK(freq.n_cols == 3);
    }
    SECTION("acf and dist")
    {
        arma::vec one = {1.0}, start = dist + 1.0, order = dist;
        order(5) = order(4);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(one, dist, 4, freq), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, start, 4, freq), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, order, 4, freq), std::invalid_argument);
        // Non-finite table: before the fix an Inf sample point ended in std::length_error
        arma::vec inf_dist = dist, nan_acf = acf;
        inf_dist(20) = arma::datum::inf, nan_acf(3) = std::nan("");
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, inf_dist, 4, freq, no_vec, no_mat, 3, true, 5, true, 0.0, 1.0, 1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, inf_dist, 4, freq), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(nan_acf, dist, 4, freq), std::invalid_argument);
    }
    SECTION("n_coeff")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 0, freq), std::invalid_argument);
    }
    SECTION("n_dim")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 4), std::invalid_argument);
    }
    SECTION("n_dir")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 3, true, 0), std::invalid_argument);
    }
    SECTION("freq_min and freq_max")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 3, true, 5, true, -0.1, 0.0, 1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 3, true, 5, true, 0.0, -0.1, 1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 3, true, 5, true, 0.2, 0.1, 1), std::invalid_argument);
        // The lower limit must also stay below the default upper limit of 15 / 20 = 0.75 / m
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 3, true, 5, true, 0.8, 0.0, 1), std::invalid_argument);
    }
    SECTION("freq_max: the search grid is limited to 100000 points")
    {
        // The grid has ( freq_max - freq_min ) · max(dist) / 0.1 + 1 points; max(dist) = 20 m.
        // 499.99 / m gives 99999 points and is accepted (one sinusoid in 1-D keeps it short), 500.01 / m is rejected.
        arma::vec d3 = {0.0, 10.0, 20.0}, a3 = {1.0, 0.5, 0.2};
        CHECK_NOTHROW(quadriga_lib::sos_generate(a3, d3, 1, freq, no_vec, no_mat, 1, true, 1, true, 0.0, 499.99, 1));
        CHECK_THROWS_AS(quadriga_lib::sos_generate(a3, d3, 1, freq, no_vec, no_mat, 1, true, 1, true, 0.0, 500.01, 1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(a3, d3, 1, freq, no_vec, no_mat, 1, true, 1, true, 0.0, 1.0e6, 1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(a3, d3, 1, freq, no_vec, no_mat, 1, true, 1, true, 0.0, 1.0e300, 1), std::invalid_argument);
        // The lower limit counts: 600 / m is fine with a lower limit of 200 / m
        CHECK_NOTHROW(quadriga_lib::sos_generate(a3, d3, 1, freq, no_vec, no_mat, 1, true, 1, true, 200.0, 600.0, 1));
    }
    SECTION("a failed call leaves the outputs untouched")
    {
        arma::mat keep(2, 2);
        keep.fill(7.0);
        arma::vec ase(3);
        ase.fill(7.0);
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 0, keep, &ase), std::invalid_argument);
        CHECK(keep.n_rows == 2);
        CHECK(keep(1, 1) == 7.0);
        CHECK(ase.n_elem == 3);
    }
    SECTION("seed")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_generate(acf, dist, 4, freq, no_vec, no_mat, 3, true, 5, true, 0.0, 0.0, -2), std::invalid_argument);
    }
}

TEST_CASE("sos_generate - with unequal sample spacing the reported error is the error of the returned table")
{
    // Sample points 0, 0.52, 1.08, ... 49.92 m with a growing step. The search then evaluates one cosine per
    // sample point instead of advancing by a rotation, which is valid for equal steps only; using the
    // rotation here would report an error that the returned table does not have.
    arma::vec dist(40), acf(40);
    for (arma::uword i = 0; i < 40; ++i)
        dist(i) = 0.02 * (double)(i * i) + 0.5 * (double)i, acf(i) = std::exp(-dist(i) / 10.0);

    QD_DIAG("\n# diag: sos_generate unequal sample spacing\nn_dim,random_init,passes,final_ase,oracle,diff\n");
    for (arma::uword n_dim = 1; n_dim <= 3; ++n_dim)
        for (bool random_init : {true, false})
        {
            arma::mat freq;
            arma::vec ase;
            quadriga_lib::sos_generate(acf, dist, 16, freq, &ase, (arma::mat *)nullptr, n_dim, true, 7, random_init, 0.0, 0.0, 4);
            REQUIRE(freq.n_rows == 16);
            REQUIRE(freq.n_cols == n_dim);
            REQUIRE(ase.n_elem >= 1);
            double ref = ref_mse_dir(freq, acf, dist, 7), last = ase(ase.n_elem - 1);
            QD_DIAG("%llu,%d,%llu,%.6e,%.6e,%.2e\n", (unsigned long long)n_dim, (int)random_init, (unsigned long long)ase.n_elem, last, ref, last - ref);
            INFO("n_dim " << n_dim << ", random_init " << random_init);
            // Cosines evaluated directly on both sides: equal to the rounding of sums of about 300 terms
            CHECK(std::abs(last - ref) < 1e-13);
            CHECK(last < 3e-2);
            CHECK(arma::abs(freq).max() > 1e-3);
            for (arma::uword i = 1; i < ase.n_elem; ++i)
                CHECK(ase(i) <= ase(i - 1));
        }
}

TEST_CASE("sos_generate - the iteration stops by the documented rules")
{
    // Documented: stop after a pass that changes nothing or lowers the error by less than a factor of 1e-6,
    // at the latest after 1000 passes. Stated on the error sequence: every pass but the last lowers the
    // error by at least 1e-6 of its value before the pass; unless the limit of 1000 passes ended the run,
    // the last pass lowers it by less than that. This holds on every platform, whichever rule ends a run.
    arma::vec dist = arma::regspace(0.0, 1.0, 40.0), acf = arma::exp(-dist / 10.0);

    arma::uword n_limit = 0, n_gain = 0, n_nothing = 0;
    QD_DIAG("\n# diag: sos_generate termination\nn_dim,random_init,n_coeff,passes,last_gain_rel,ended_by\n");
    for (arma::uword n_dim = 1; n_dim <= 3; ++n_dim)
        for (bool random_init : {true, false})
            for (arma::uword n_coeff : {4, 24})
            {
                arma::mat freq;
                arma::vec ase;
                quadriga_lib::sos_generate(acf, dist, n_coeff, freq, &ase, (arma::mat *)nullptr, n_dim, true, 7, random_init, 0.0, 0.0, 3);
                const arma::uword n = ase.n_elem;
                REQUIRE(n >= 1);
                INFO("n_dim " << n_dim << ", random_init " << random_init << ", n_coeff " << n_coeff << ", passes " << n);
                CHECK(n <= 1000);
                for (arma::uword i = 1; i + 1 < n; ++i)
                    CHECK(ase(i - 1) - ase(i) >= 1.0e-6 * ase(i - 1));
                double last_gain = (n > 1) ? (ase(n - 2) - ase(n - 1)) / ase(n - 2) : 0.0;
                if (n > 1 && n < 1000)
                    CHECK(last_gain < 1.0e-6);
                const char *ended = (n == 1000) ? "limit" : (last_gain > 0.0 ? "gain below 1e-6" : "nothing changed");
                n_limit += (n == 1000) ? 1 : 0, n_gain += (n < 1000 && last_gain > 0.0) ? 1 : 0, n_nothing += (n < 1000 && last_gain == 0.0) ? 1 : 0;
                QD_DIAG("%llu,%d,%llu,%llu,%.3e,%s\n", (unsigned long long)n_dim, (int)random_init, (unsigned long long)n_coeff, (unsigned long long)n, last_gain, ended);
            }
    // On the platform this was written on the 12 runs end 2 times at the limit, 2 times by the gain rule and
    // 8 times without a change. Which rule ends a run can differ between math libraries, so only the
    // presence of the two regular endings is asserted.
    QD_DIAG("ended by: limit %llu, gain %llu, nothing changed %llu\n", (unsigned long long)n_limit, (unsigned long long)n_gain, (unsigned long long)n_nothing);
    CHECK(n_gain >= 1);
    CHECK(n_nothing >= 1);
}

TEST_CASE("sos_generate - the smallest problem runs: one sinusoid, two samples, one test direction")
{
    arma::vec acf = {1.0, 0.5}, dist = {0.0, 10.0}, ase;
    arma::mat freq, test_dir;
    quadriga_lib::sos_generate(acf, dist, 1, freq, &ase, &test_dir, 3, true, 1, true, 0.0, 0.0, 1);
    REQUIRE(freq.n_rows == 1);
    REQUIRE(freq.n_cols == 3);
    REQUIRE(ase.n_elem >= 1);
    REQUIRE(test_dir.n_rows == 1);
    REQUIRE(test_dir.n_cols == 2);
    double ref = ref_mse_dir(freq, acf, dist, 1), last = ase(ase.n_elem - 1);
    QD_DIAG("\n# diag: sos_generate smallest problem: %llu passes, error %.3e, oracle %.3e\n", (unsigned long long)ase.n_elem, last, ref);
    CHECK(std::abs(last - ref) < 1e-13);
    // One cosine can meet the single value 0.5 at 10 m exactly; the step refinement stops at 1e-6 periods
    CHECK(last < 1e-9);
    CHECK(freq.is_finite());
    CHECK(arma::abs(freq).max() > 1e-3);
}

TEST_CASE("sos_generate - with few test directions the result fits those directions and is worse in others")
{
    // 3 test directions in 3-D. The error on these 3 directions is what the search minimises; on 101 other
    // directions the same table is worse. Measured ratios for the three seeds: 3.4 to 3.9.
    arma::vec dist = arma::regspace(0.0, 1.0, 40.0), acf = arma::exp(-dist / 10.0);
    QD_DIAG("\n# diag: sos_generate fitted with 3 test directions\nseed,error_3_dir,error_101_dir,ratio\n");
    for (arma::sword seed = 1; seed <= 3; ++seed)
    {
        arma::mat freq;
        arma::vec ase;
        quadriga_lib::sos_generate(acf, dist, 24, freq, &ase, (arma::mat *)nullptr, 3, true, 3, true, 0.0, 0.0, seed);
        REQUIRE(freq.n_rows == 24);
        double fit = ref_mse_dir(freq, acf, dist, 3), other = ref_mse_dir(freq, acf, dist, 101);
        QD_DIAG("%lld,%.3e,%.3e,%.2f\n", (long long)seed, fit, other, other / fit);
        INFO("seed " << seed);
        CHECK(std::abs(ase(ase.n_elem - 1) - fit) < 1e-10);
        CHECK(other > 1.5 * fit);
    }
}

TEST_CASE("sos_generate - an output may be the same object as an input")
{
    arma::vec dist = arma::regspace(0.0, 1.0, 40.0), acf = arma::exp(-dist / 10.0);
    arma::mat ref, freq;
    arma::vec ref_ase;
    quadriga_lib::sos_generate(acf, dist, 12, ref, &ref_ase, (arma::mat *)nullptr, 3, true, 7, true, 0.0, 0.0, 5);
    REQUIRE(ref.n_rows == 12);
    CHECK(arma::abs(ref).max() > 1e-3);

    // ase is 'acf' (41 elements, resized to the number of passes); then ase is 'dist'
    arma::vec a = acf, d = dist;
    quadriga_lib::sos_generate(a, dist, 12, freq, &a, (arma::mat *)nullptr, 3, true, 7, true, 0.0, 0.0, 5);
    CHECK(arma::approx_equal(freq, ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(a, ref_ase, "absdiff", 0.0));
    quadriga_lib::sos_generate(acf, d, 12, freq, &d, (arma::mat *)nullptr, 3, true, 7, true, 0.0, 0.0, 5);
    CHECK(arma::approx_equal(freq, ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(d, ref_ase, "absdiff", 0.0));
}
