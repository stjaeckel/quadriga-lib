// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::sos_acf_estimate.
//
// Pins down: the documented procedure (positions, bins, pair selection, correlation coefficient) for one
// and for two devices, the NaN rule for sparse bins, that the estimate follows the ACF of the sinusoids,
// reproducibility, float and double, input validation.
// Oracles:
//  - exact: the procedure is documented down to the random stream of the positions, so it is repeated
//    here step by step in scalar double math. The values of the generator come from sos_val, which has
//    its own suite; everything else (positions, distances, bins, correlation) is independent code.
//  - statistical: for frequencies drawn from an isotropic Normal distribution with standard deviation
//    sigma per axis, the ACF is exp( -2·pi^2·sigma^2·d^2 ) for many sinusoids.
// Conventions: positions are drawn as a matrix [2·n_pos, 3], filled column by column (all x, then y,
// then z) with ( 2·u - 1 ) · 0.4 · max(bins); rows 0 ... n_pos-1 are the first device.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_sos.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

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

    arma::mat rnd(arma::uword n_rows, arma::uword n_cols, double lo, double hi, unsigned seed)
    {
        std::mt19937_64 rng(seed);
        arma::mat m(n_rows, n_cols);
        for (arma::uword i = 0; i < m.n_elem; ++i)
            m(i) = lo + (hi - lo) * ((double)(rng() >> 11) / 9007199254740992.0);
        return m;
    }

    // Running sums for a correlation coefficient
    struct Corr
    {
        double n = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0, sd = 0.0;
        void add(double x, double y, double d) { n += 1.0, sx += x, sy += y, sxx += x * x, syy += y * y, sxy += x * y, sd += d; }
        double coeff() const { return (sxy / n - sx * sy / (n * n)) / std::sqrt((sxx / n - sx * sx / (n * n)) * (syy / n - sy * sy / (n * n))); }
    };

    // Oracle: the documented procedure. Returns [n_bin, 3] = {acf_est, dist_est, acf_est_dual}
    arma::mat ref_estimate(const arma::mat &freq, const arma::vec &phase, const arma::vec &bins, arma::uword n_pos,
                           const arma::vec &acf, const arma::vec &dist, unsigned long long seed)
    {
        const arma::uword n_bin = bins.n_elem, n2 = 2 * n_pos;
        const double half_edge = 0.4 * bins(n_bin - 1);

        std::mt19937_64 rng(seed);
        arma::mat pos(n2, 3);
        for (arma::uword i = 0; i < pos.n_elem; ++i)
            pos(i) = (2.0 * ((double)(rng() >> 11) / 9007199254740992.0) - 1.0) * half_edge;

        // Bin limits: lower excluded, upper included
        arma::vec lo(n_bin), hi(n_bin);
        for (arma::uword b = 0; b < n_bin; ++b)
        {
            lo(b) = (b == 0) ? 1e-5 : 0.5 * (bins(b - 1) + bins(b));
            hi(b) = (b == n_bin - 1) ? bins(b) + 0.5 * (bins(n_bin - 1) - bins(0)) / (double)(n_bin - 1) : 0.5 * (bins(b) + bins(b + 1));
        }

        arma::mat ph = phase, v3, v6, pa = pos.rows(0, n_pos - 1), pb = pos.rows(n_pos, n2 - 1);
        quadriga_lib::sos_val(freq, ph, pos, v3, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
        quadriga_lib::sos_val(freq, ph, pa, v6, &pb, (arma::mat *)nullptr, &acf, &dist, 1.0, false, 1);

        auto distance = [&](arma::uword i, arma::uword j)
        { return std::sqrt(std::pow(pos(i, 0) - pos(j, 0), 2) + std::pow(pos(i, 1) - pos(j, 1), 2) + std::pow(pos(i, 2) - pos(j, 2), 2)); };

        std::vector<Corr> c3(n_bin), c6(n_bin);
        for (arma::uword i = 0; i < n2; ++i)
            for (arma::uword j = i + 1; j < n2; ++j)
            {
                double d = distance(i, j);
                for (arma::uword b = 0; b < n_bin; ++b)
                    if (d > lo(b) && d <= hi(b))
                        c3[b].add(v3(i, 0), v3(j, 0), d);
            }
        for (arma::uword i = 0; i < n_pos; ++i)
            for (arma::uword j = i + 1; j < n_pos; ++j)
            {
                double da = distance(i, j), db = distance(n_pos + i, n_pos + j);
                for (arma::uword b = 0; b < n_bin; ++b)
                    if (da > lo(b) && da <= hi(b) && db <= hi(b))
                        c6[b].add(v6(i, 0), v6(j, 0), da);
            }

        arma::mat out(n_bin, 3);
        out.fill(arma::datum::nan);
        for (arma::uword b = 0; b < n_bin; ++b)
        {
            if (c3[b].n > 5.5)
                out(b, 0) = c3[b].coeff(), out(b, 1) = c3[b].sd / c3[b].n;
            if (c6[b].n > 5.5)
                out(b, 2) = c6[b].coeff();
        }
        return out;
    }

    bool same_or_both_nan(double a, double b, double tol) { return (std::isnan(a) && std::isnan(b)) || std::abs(a - b) < tol; }
}

TEST_CASE("sos_acf_estimate - the estimate equals the documented procedure for one and two devices")
{
    arma::mat freq = rnd(40, 3, -0.08, 0.08, 1);
    arma::vec phase = rnd(40, 1, -pi, pi, 2), bins = {2.0, 5.0, 10.0, 15.0, 22.0, 30.0};
    arma::vec dist = arma::regspace(0.0, 1.0, 30.0), acf = arma::exp(-dist / 8.0);

    arma::vec est, dest, dual;
    quadriga_lib::sos_acf_estimate(freq, phase, bins, 60, est, &dest, &dual, &acf, &dist, 5);
    REQUIRE(est.n_elem == 6);
    REQUIRE(dest.n_elem == 6);
    REQUIRE(dual.n_elem == 6);

    arma::mat ref = ref_estimate(freq, phase, bins, 60, acf, dist, 5);
    QD_DIAG("\n# diag: sos_acf_estimate against the documented procedure, n_pos = 60\n");
    QD_DIAG("bin,est,ref,dist_est,ref,dual,ref\n");
    arma::uword n_finite = 0, n_dual = 0;
    for (arma::uword b = 0; b < 6; ++b)
    {
        QD_DIAG("%.0f,%.6f,%.6f,%.4f,%.4f,%.6f,%.6f\n", bins(b), est(b), ref(b, 0), dest(b), ref(b, 1), dual(b), ref(b, 2));
        INFO("bin " << b);
        CHECK(same_or_both_nan(est(b), ref(b, 0), 1e-10));
        CHECK(same_or_both_nan(dest(b), ref(b, 1), 1e-10));
        CHECK(same_or_both_nan(dual(b), ref(b, 2), 1e-10));
        n_finite += std::isfinite(ref(b, 0)) ? 1 : 0, n_dual += std::isfinite(ref(b, 2)) ? 1 : 0;
    }
    CHECK(n_finite == 6); // The oracle is populated in all bins
    CHECK(n_dual >= 3);
    CHECK(arma::abs(ref.col(0)).max() > 0.05);
}

TEST_CASE("sos_acf_estimate - the estimate follows the ACF of the sinusoids")
{
    // Frequencies from an isotropic Normal distribution (Box-Muller on the fixture generator),
    // sigma = 0.03 / m: R(d) = exp( -2·pi^2·sigma^2·d^2 ), which is 0.41 at 7 m and 0.03 at 14 m.
    const arma::uword n_coeff = 400;
    const double sigma = 0.03;
    arma::mat u1 = rnd(n_coeff, 3, 1e-12, 1.0, 3), u2 = rnd(n_coeff, 3, 0.0, 1.0, 4), freq(n_coeff, 3);
    for (arma::uword i = 0; i < freq.n_elem; ++i)
        freq(i) = sigma * std::sqrt(-2.0 * std::log(u1(i))) * std::cos(2.0 * pi * u2(i));
    arma::vec phase = rnd(n_coeff, 1, -pi, pi, 5), bins = arma::regspace(2.0, 4.0, 62.0);

    arma::vec est, dest;
    quadriga_lib::sos_acf_estimate(freq, phase, bins, 600, est, &dest, (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1);
    REQUIRE(est.n_elem == bins.n_elem);

    // One realisation in a cube of 50 m edge with a correlation length of about 7 m holds a few hundred
    // independent patches, and the table has 400 sinusoids: the estimate scatters by about 0.05 to 0.1
    // around the ideal ACF. The limit is 0.2.
    QD_DIAG("\n# diag: sos_acf_estimate against the Gaussian ACF\nbin,dist_est,est,ideal\n");
    for (arma::uword b = 0; b < bins.n_elem; ++b)
    {
        double ideal = std::exp(-2.0 * pi * pi * sigma * sigma * dest(b) * dest(b));
        QD_DIAG("%.0f,%.3f,%.4f,%.4f\n", bins(b), dest(b), est(b), ideal);
        INFO("bin " << bins(b));
        CHECK(std::abs(est(b) - ideal) < 0.2);
        CHECK(std::abs(dest(b) - bins(b)) <= 2.0); // Inside the bin
    }
    CHECK(est(0) > 0.7);
}

TEST_CASE("sos_acf_estimate - bins with fewer than 6 pairs are NaN")
{
    arma::mat freq = rnd(20, 3, -0.08, 0.08, 6);
    arma::vec phase = rnd(20, 1, -pi, pi, 7), bins = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    arma::vec dist = arma::regspace(0.0, 1.0, 10.0), acf = arma::exp(-dist / 3.0);

    // 4 positions give 6 pairs in total, spread over 8 bins; one pair of links for the two-device estimate
    arma::vec est, dest, dual;
    quadriga_lib::sos_acf_estimate(freq, phase, bins, 2, est, &dest, &dual, &acf, &dist, 3);
    REQUIRE(est.n_elem == 8);
    arma::mat ref = ref_estimate(freq, phase, bins, 2, acf, dist, 3);
    arma::uword n_nan = 0;
    for (arma::uword b = 0; b < 8; ++b)
    {
        CHECK(same_or_both_nan(est(b), ref(b, 0), 1e-10));
        CHECK(std::isnan(dest(b)) == std::isnan(est(b)));
        CHECK(std::isnan(dual(b)));
        n_nan += std::isnan(est(b)) ? 1 : 0;
    }
    CHECK(n_nan >= 7);
}

TEST_CASE("sos_acf_estimate - seeds, optional outputs, pre-sized outputs and float")
{
    arma::mat freq = rnd(40, 3, -0.08, 0.08, 8);
    arma::vec phase = rnd(40, 1, -pi, pi, 9), bins = {2.0, 5.0, 10.0, 15.0, 22.0, 30.0};
    arma::vec dist = arma::regspace(0.0, 1.0, 30.0), acf = arma::exp(-dist / 8.0);

    arma::vec a, b, c, d1, d2, dual;
    quadriga_lib::sos_acf_estimate(freq, phase, bins, 80, a, &d1, &dual, &acf, &dist, 1);
    quadriga_lib::sos_acf_estimate(freq, phase, bins, 80, b, (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1);
    quadriga_lib::sos_acf_estimate(freq, phase, bins, 80, c, &d2, (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 2);
    REQUIRE(a.n_elem == 6);
    CHECK(arma::abs(a).max() > 0.05);
    CHECK(arma::approx_equal(a, b, "absdiff", 0.0)); // Requesting more outputs does not change the estimate
    CHECK(arma::abs(a - c).max() > 1e-3);            // Another seed gives other positions

    arma::vec sized(6);
    sized.fill(9.0);
    const double *ptr = sized.memptr();
    quadriga_lib::sos_acf_estimate(freq, phase, bins, 80, sized, (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1);
    CHECK(sized.memptr() == ptr);
    CHECK(arma::approx_equal(sized, a, "absdiff", 0.0));

    arma::fmat ff = arma::conv_to<arma::fmat>::from(freq);
    arma::fvec pf = arma::conv_to<arma::fvec>::from(phase), bf = arma::conv_to<arma::fvec>::from(bins), ef, df;
    quadriga_lib::sos_acf_estimate(ff, pf, bf, 80, ef, &df, (arma::fvec *)nullptr, (arma::fvec *)nullptr, (arma::fvec *)nullptr, 1);
    REQUIRE(ef.n_elem == 6);
    // Float inputs move the generator values by about 1e-5; a pair exactly on a bin limit is unlikely
    CHECK(arma::abs(arma::conv_to<arma::vec>::from(ef) - a).max() < 1e-3);
    CHECK(arma::abs(arma::conv_to<arma::vec>::from(df) - d1).max() < 1e-3);
}

TEST_CASE("sos_acf_estimate - input validation")
{
    arma::mat freq = rnd(10, 3, -0.08, 0.08, 10);
    arma::vec phase = rnd(10, 1, -pi, pi, 11), bins = {2.0, 5.0, 10.0}, est, dual;
    arma::vec dist = arma::regspace(0.0, 1.0, 10.0), acf = arma::exp(-dist / 3.0);
    arma::vec *no_vec = nullptr;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_acf_estimate(freq, phase, bins, 10, est));
        CHECK_NOTHROW(quadriga_lib::sos_acf_estimate(freq, phase, bins, 10, est, no_vec, &dual, &acf, &dist, 0));
    }
    SECTION("freq")
    {
        arma::mat empty, two = freq.cols(0, 1);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(empty, phase, bins, 10, est), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(two, phase, bins, 10, est), std::invalid_argument);
    }
    SECTION("phase")
    {
        arma::vec bad(9, arma::fill::zeros);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(freq, bad, bins, 10, est), std::invalid_argument);
    }
    SECTION("bins")
    {
        arma::vec one = {2.0}, order = {2.0, 10.0, 5.0}, neg = {-1.0, 2.0, 3.0};
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(freq, phase, one, 10, est), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(freq, phase, order, 10, est), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(freq, phase, neg, 10, est), std::invalid_argument);
    }
    SECTION("n_pos")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(freq, phase, bins, 1, est), std::invalid_argument);
    }
    SECTION("acf and dist for the two-device estimate")
    {
        arma::vec bad_dist = dist + 1.0;
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(freq, phase, bins, 10, est, no_vec, &dual), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(freq, phase, bins, 10, est, no_vec, &dual, &acf, &bad_dist), std::invalid_argument);
    }
    SECTION("seed")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_acf_estimate(freq, phase, bins, 10, est, no_vec, no_vec, no_vec, no_vec, -2), std::invalid_argument);
    }
}
