// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::sos_calc_mse.
//
// Pins down: the three error measures and the tail measure, the test directions, the optional outputs,
// float and double, input validation.
// Oracles, written in scalar double math in this file:
//  - the documented test directions (half circle, spiral lattice on the hemisphere)
//  - the error along the test directions: mean over directions and sample points of ( R - R_sos )^2
//  - the grid errors: R_sos on the planes from its closed form, the target from a radial interpolation
//    that is zero beyond the last sample point, the core mask as index radius <= n_acf
//  - limiting case: a 1-D target that is the ACF of the sinusoids themselves has no error
// Conventions: small tables (6 samples, 2 m spacing, grid of 21 points) so that the brute-force grids
// are cheap; frequencies with distinct components.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_sos.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Diagnostic tables print with QD_TEST_DIAG=1 in the environment; the assertions run either way.
#include "qd_test_diag.hpp"

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

    // Oracle: grid errors [core, all] in dB for n_dim = 2 or 3
    void ref_mse_grid(const arma::mat &freq, const arma::vec &acf, const arma::vec &dist, double &core_dB, double &all_dB)
    {
        const long long S = (long long)acf.n_elem, half = 2 * S - 2;
        const double step = dist(1) - dist(0), d_max = dist(dist.n_elem - 1);
        const arma::uword n_plane = (freq.n_cols == 3) ? 3 : 1;
        const arma::uword first[3] = {0, 0, 1}, second[3] = {1, 2, 2};
        double core = 0.0, all = 0.0;
        for (arma::uword pl = 0; pl < n_plane; ++pl)
        {
            double sc = 0.0, sa = 0.0, nc = 0.0, na = 0.0;
            for (long long ic = -half; ic <= half; ++ic)
                for (long long ir = -half; ir <= half; ++ir)
                {
                    double a = (double)ic * step, b = (double)ir * step, r = std::sqrt(a * a + b * b), target = 0.0;
                    if (r <= d_max)
                    {
                        arma::uword i = (arma::uword)std::floor(r / step);
                        if (i >= acf.n_elem - 1)
                            i = acf.n_elem - 2;
                        target = acf(i) + (r - dist(i)) / step * (acf(i + 1) - acf(i));
                    }
                    double approx = 0.0;
                    for (arma::uword n = 0; n < freq.n_rows; ++n)
                        approx += std::cos(2.0 * pi * (freq(n, first[pl]) * a + freq(n, second[pl]) * b)) / (double)freq.n_rows;
                    double e2 = (approx - target) * (approx - target);
                    sa += e2, na += 1.0;
                    if (std::sqrt((double)(ic * ic + ir * ir)) <= (double)S)
                        sc += e2, nc += 1.0;
                }
            core += sc / nc / (double)n_plane, all += sa / na / (double)n_plane;
        }
        core_dB = -10.0 * std::log10(core), all_dB = -10.0 * std::log10(all);
    }
}

TEST_CASE("sos_calc_mse - the directional error matches the brute-force evaluation in 2-D and 3-D")
{
    arma::vec dist = arma::regspace(0.0, 2.0, 10.0), acf = arma::exp(-dist / 4.0);

    for (arma::uword n_dim = 2; n_dim <= 3; ++n_dim)
        for (arma::uword n_dir : {1, 7, 10})
        {
            arma::mat freq = rnd(12, n_dim, -0.15, 0.15, 10 + (unsigned)n_dim);
            arma::mat test_dir;
            double mse = quadriga_lib::sos_calc_mse(freq, acf, dist, (double *)nullptr, (double *)nullptr,
                                                    (arma::vec *)nullptr, &test_dir, n_dir);
            double ref = -10.0 * std::log10(ref_mse_dir(freq, acf, dist, n_dir));
            QD_DIAG("\n# diag: sos_calc_mse directional error, n_dim %llu, n_dir %llu: %.6f dB, oracle %.6f dB\n",
                    (unsigned long long)n_dim, (unsigned long long)n_dir, mse, ref);
            INFO("n_dim " << n_dim << ", n_dir " << n_dir);
            CHECK(std::abs(mse - ref) < 1e-9);
            CHECK(ref > 1.0);
            CHECK(ref < 40.0);

            // Test directions as documented
            REQUIRE(test_dir.n_rows == n_dir);
            REQUIRE(test_dir.n_cols == n_dim - 1);
            arma::mat u = ref_dirs(n_dir, n_dim);
            for (arma::uword i = 0; i < n_dir; ++i)
            {
                double az = test_dir(i, 0), el = (n_dim == 3) ? test_dir(i, 1) : 0.0;
                CHECK(std::abs(std::cos(az) * std::cos(el) - u(i, 0)) < 1e-12);
                CHECK(std::abs(std::sin(az) * std::cos(el) - u(i, 1)) < 1e-12);
                CHECK(std::abs(std::sin(el) - u(i, 2)) < 1e-12);
            }
        }
}

TEST_CASE("sos_calc_mse - the grid errors match the brute-force evaluation in 2-D and 3-D")
{
    arma::vec dist = arma::regspace(0.0, 2.0, 10.0), acf = arma::exp(-dist / 4.0);

    for (arma::uword n_dim = 2; n_dim <= 3; ++n_dim)
    {
        arma::mat freq = rnd(12, n_dim, -0.15, 0.15, 20 + (unsigned)n_dim);
        double core = 0.0, all = 0.0, ref_core = 0.0, ref_all = 0.0;
        quadriga_lib::sos_calc_mse(freq, acf, dist, &core, &all);
        ref_mse_grid(freq, acf, dist, ref_core, ref_all);
        QD_DIAG("\n# diag: sos_calc_mse grid errors, n_dim %llu\n", (unsigned long long)n_dim);
        QD_DIAG("measure,value_dB,oracle_dB\ncore,%.6f,%.6f\nall,%.6f,%.6f\n", core, ref_core, all, ref_all);
        INFO("n_dim " << n_dim);
        CHECK(std::abs(core - ref_core) < 1e-9);
        CHECK(std::abs(all - ref_all) < 1e-9);
        CHECK(ref_core > 1.0);
        CHECK(std::abs(ref_core - ref_all) > 1e-3); // The two measures differ
    }
}

TEST_CASE("sos_calc_mse - one dimension: an exact target has no error inside the fitted range")
{
    // Target = ACF of the two sinusoids. Inside the range the error vanishes; on the axis up to 2·Dmax the
    // target is 0 beyond Dmax while the sinusoids continue, which gives a finite error.
    arma::mat freq(2, 1);
    freq(0, 0) = 0.04, freq(1, 0) = 0.13;
    arma::vec dist = arma::regspace(0.0, 0.5, 20.0), acf(dist.n_elem);
    for (arma::uword s = 0; s < dist.n_elem; ++s)
        acf(s) = 0.5 * (std::cos(2.0 * pi * 0.04 * dist(s)) + std::cos(2.0 * pi * 0.13 * dist(s)));

    double core = 0.0, all = 0.0;
    arma::mat test_dir(3, 3);
    double mse = quadriga_lib::sos_calc_mse(freq, acf, dist, &core, &all, (arma::vec *)nullptr, &test_dir, 5);

    // Oracle for the complete axis: 4·S - 3 points, error only on the 2·(S - 1) points beyond Dmax
    const arma::uword S = dist.n_elem;
    double acc = 0.0;
    for (arma::uword k = 1; k < S; ++k)
    {
        double d = dist(S - 1) + dist(k), r = 0.5 * (std::cos(2.0 * pi * 0.04 * d) + std::cos(2.0 * pi * 0.13 * d));
        acc += 2.0 * r * r;
    }
    double ref_all = -10.0 * std::log10(acc / (double)(4 * S - 3));

    QD_DIAG("\n# diag: sos_calc_mse 1-D exact target\nmeasure,value_dB,oracle_dB\nmse,%.2f,>250\ncore,%.2f,>250\nall,%.6f,%.6f\n", mse, core, all, ref_all);
    CHECK(mse > 250.0); // Rounding errors of 1e-16 give about 300 dB
    CHECK(core > 250.0);
    CHECK(mse == core);
    CHECK(std::abs(all - ref_all) < 1e-9);
    CHECK(test_dir.n_elem == 0);

    // A wrong target has a finite error that matches the oracle
    arma::vec other = arma::exp(-dist / 5.0);
    mse = quadriga_lib::sos_calc_mse(freq, other, dist);
    CHECK(std::abs(mse + 10.0 * std::log10(ref_mse_dir(freq, other, dist, 1))) < 1e-9);
    CHECK(mse < 30.0);
}

TEST_CASE("sos_calc_mse - the tail measure is the directional average beyond the fitted range")
{
    arma::vec dist = arma::regspace(0.0, 2.0, 10.0), acf = arma::exp(-dist / 4.0);
    arma::mat freq = rnd(40, 3, -0.2, 0.2, 30);
    freq.rows(0, 9).zeros(); // 10 of 40 sinusoids at zero frequency

    arma::vec tail;
    quadriga_lib::sos_calc_mse(freq, acf, dist, (double *)nullptr, (double *)nullptr, &tail, (arma::mat *)nullptr, 50);
    REQUIRE(tail.n_elem == 4);

    arma::mat u = ref_dirs(50, 3);
    const double mult[4] = {2.0, 4.0, 10.0, 20.0};
    QD_DIAG("\n# diag: sos_calc_mse tail, 10 of 40 sinusoids at zero frequency\nmultiple,tail,oracle\n");
    for (arma::uword k = 0; k < 4; ++k)
    {
        double ref = 0.0;
        for (arma::uword t = 0; t < 50; ++t)
            ref += ref_acf_dir(freq, u, t, mult[k] * 10.0) / 50.0;
        QD_DIAG("%.0f,%.6f,%.6f\n", mult[k], tail(k), ref);
        INFO("multiple " << mult[k]);
        CHECK(std::abs(tail(k) - ref) < 1e-12);
        // Plateau of 10/40 plus the average of 30 unrelated cosines over 50 directions: within 0.25 +- 0.1
        CHECK(std::abs(tail(k) - 0.25) < 0.1);
    }

    // All sinusoids at zero frequency: the ACF is 1 everywhere
    arma::mat zero(5, 3, arma::fill::zeros);
    quadriga_lib::sos_calc_mse(zero, acf, dist, (double *)nullptr, (double *)nullptr, &tail);
    CHECK(arma::approx_equal(tail, arma::vec({1.0, 1.0, 1.0, 1.0}), "absdiff", 1e-14));
}

TEST_CASE("sos_calc_mse - optional outputs do not change each other, float agrees with double")
{
    arma::vec dist = arma::regspace(0.0, 2.0, 10.0), acf = arma::exp(-dist / 4.0);
    arma::mat freq = rnd(12, 3, -0.15, 0.15, 40);

    double core = 0.0, all = 0.0, core1 = 0.0, all1 = 0.0;
    arma::vec tail(9), tail1;
    tail.fill(9.0);
    arma::mat dir, dir1;
    double m_all = quadriga_lib::sos_calc_mse(freq, acf, dist, &core, &all, &tail, &dir);
    double m_none = quadriga_lib::sos_calc_mse(freq, acf, dist);
    CHECK(m_all == m_none);
    CHECK(m_all > 1.0);
    CHECK(quadriga_lib::sos_calc_mse(freq, acf, dist, &core1) == m_none);
    CHECK(core1 == core);
    CHECK(quadriga_lib::sos_calc_mse(freq, acf, dist, (double *)nullptr, &all1) == m_none);
    CHECK(all1 == all);
    quadriga_lib::sos_calc_mse(freq, acf, dist, (double *)nullptr, (double *)nullptr, &tail1);
    REQUIRE(tail.n_elem == 4);
    CHECK(arma::approx_equal(tail, tail1, "absdiff", 0.0));
    quadriga_lib::sos_calc_mse(freq, acf, dist, (double *)nullptr, (double *)nullptr, (arma::vec *)nullptr, &dir1);
    REQUIRE(dir.n_rows == 10);
    CHECK(arma::approx_equal(dir, dir1, "absdiff", 0.0));

    arma::fmat ff = arma::conv_to<arma::fmat>::from(freq);
    arma::fvec af = arma::conv_to<arma::fvec>::from(acf), df = arma::conv_to<arma::fvec>::from(dist), tf;
    float cf = 0.0f, alf = 0.0f;
    float mf = quadriga_lib::sos_calc_mse(ff, af, df, &cf, &alf, &tf);
    // Float inputs change the ACF by about 1e-6, which is 1e-4 dB at an error level of 10 to 20 dB
    CHECK(std::abs((double)mf - m_all) < 1e-3);
    CHECK(std::abs((double)cf - core) < 1e-3);
    CHECK(std::abs((double)alf - all) < 1e-3);
    REQUIRE(tf.n_elem == 4);
    CHECK(arma::abs(arma::conv_to<arma::vec>::from(tf) - tail).max() < 1e-3);
}

TEST_CASE("sos_calc_mse - input validation")
{
    arma::vec dist = arma::regspace(0.0, 2.0, 10.0), acf = arma::exp(-dist / 4.0);
    arma::mat freq = rnd(12, 3, -0.15, 0.15, 50);

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_calc_mse(freq, acf, dist));
    }
    SECTION("freq")
    {
        arma::mat empty, four(3, 4, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(empty, acf, dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(four, acf, dist), std::invalid_argument);
    }
    SECTION("acf and dist")
    {
        arma::vec short_acf = {1.0, 0.5}, uneven = dist, start = dist + 1.0;
        uneven(3) += 0.5;
        CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(freq, short_acf, dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(freq, acf, uneven), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(freq, acf, start), std::invalid_argument);
        // Non-finite table
        arma::vec inf_dist = dist, nan_dist = dist, nan_acf = acf;
        inf_dist(5) = arma::datum::inf, nan_dist(2) = std::nan(""), nan_acf(2) = std::nan("");
        CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(freq, acf, inf_dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(freq, acf, nan_dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(freq, nan_acf, dist), std::invalid_argument);
    }
    SECTION("n_dir")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_calc_mse(freq, acf, dist, (double *)nullptr, (double *)nullptr, (arma::vec *)nullptr, (arma::mat *)nullptr, 0), std::invalid_argument);
    }
}

TEST_CASE("sos_calc_mse - a perfect fit returns +Inf")
{
    // Sinusoids at zero frequency have the ACF 1 at all distances; with a target of 1 the error along the
    // test directions is exactly 0 and -10·log10(0) is +Inf. This is the documented result, not a cap.
    arma::vec dist = arma::regspace(0.0, 2.0, 10.0), acf(dist.n_elem, arma::fill::ones);
    for (arma::uword n_dim = 1; n_dim <= 3; ++n_dim)
    {
        arma::mat freq(4, n_dim, arma::fill::zeros);
        double core = 0.0, all = 0.0;
        double mse = quadriga_lib::sos_calc_mse(freq, acf, dist, &core, &all);
        QD_DIAG("\n# diag: sos_calc_mse perfect fit, n_dim %llu: mse %f, core %f, all %f\n", (unsigned long long)n_dim, mse, core, all);
        INFO("n_dim " << n_dim);
        CHECK(std::isinf(mse));
        CHECK(mse > 0.0);
        // The grids reach beyond the last sample point, where the target is 0 and the sinusoids are 1
        CHECK(std::isfinite(all));
        if (n_dim == 1)
            CHECK((std::isinf(core) && core > 0.0)); // Equals the return value in 1-D
        else
            CHECK(std::isfinite(core)); // The core radius of n_acf steps reaches one step beyond the table
    }

    arma::fvec df = arma::conv_to<arma::fvec>::from(dist), af(dist.n_elem, arma::fill::ones);
    arma::fmat ff(4, 3, arma::fill::zeros);
    float mf = quadriga_lib::sos_calc_mse(ff, af, df);
    CHECK((std::isinf(mf) && mf > 0.0f));
}

TEST_CASE("sos_calc_mse - an output may be the same object as an input")
{
    arma::vec dist = arma::regspace(0.0, 2.0, 10.0), acf = arma::exp(-dist / 4.0);
    arma::mat freq = rnd(12, 3, -0.15, 0.15, 60);

    double core = 0.0, all = 0.0, core1 = 0.0, all1 = 0.0;
    arma::vec tail;
    arma::mat dir;
    double ref = quadriga_lib::sos_calc_mse(freq, acf, dist, &core, &all, &tail, &dir, 7);

    // acf_tail is 'acf' (6 elements, resized to 4), test_dir is 'freq'
    arma::vec a = acf;
    arma::mat f = freq;
    double mse = quadriga_lib::sos_calc_mse(f, a, dist, &core1, &all1, &a, &f, 7);
    CHECK(mse == ref);
    CHECK(core1 == core);
    CHECK(all1 == all);
    REQUIRE(a.n_elem == 4);
    CHECK(arma::approx_equal(a, tail, "absdiff", 0.0));
    REQUIRE(f.n_rows == 7);
    CHECK(arma::approx_equal(f, dir, "absdiff", 0.0));

    // acf_tail is 'dist'
    arma::vec d = dist;
    mse = quadriga_lib::sos_calc_mse(freq, acf, d, &core1, &all1, &d, (arma::mat *)nullptr, 7);
    CHECK(mse == ref);
    CHECK(core1 == core);
    CHECK(arma::approx_equal(d, tail, "absdiff", 0.0));
}

TEST_CASE("sos_calc_mse - results do not depend on the number of threads")
{
    // The test directions are evaluated in parallel and summed in a fixed order
    arma::vec dist = arma::regspace(0.0, 2.0, 10.0), acf = arma::exp(-dist / 4.0);
    arma::mat freq = rnd(40, 3, -0.15, 0.15, 70);

    double mse[2], core[2], all[2];
    arma::vec tail[2];
    for (int run = 0; run < 2; ++run)
    {
#ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        if (run == 1)
            omp_set_num_threads(1);
#endif
        mse[run] = quadriga_lib::sos_calc_mse(freq, acf, dist, &core[run], &all[run], &tail[run], (arma::mat *)nullptr, 97);
#ifdef _OPENMP
        omp_set_num_threads(n_threads);
#endif
    }
    CHECK(mse[0] > 1.0);
    CHECK(mse[0] == mse[1]);
    CHECK(core[0] == core[1]);
    CHECK(all[0] == all[1]);
    CHECK(arma::approx_equal(tail[0], tail[1], "absdiff", 0.0));
    CHECK(std::abs(mse[0] + 10.0 * std::log10(ref_mse_dir(freq, acf, dist, 97))) < 1e-9);
}
