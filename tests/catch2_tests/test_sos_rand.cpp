// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_SOS=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_SOS=1 ./build_linux/test_bin "sos_rand*"

// Verification suite for quadriga_lib::sos_randn, sos_rand and sos_randi.
//
// Pins down: the documented composition (preset, phases, evaluation) including the assignment of phase
// columns and the second phase stream of a device on its own map, the replacement of a zero
// decorrelation distance, the choice between one map and two maps, the mapping to Uniform and integer
// values, the distributions, seeds and what a seed does and does not fix, sizes, float and double, input
// validation.
// It leaves the values of the generator, the tables and the phase stream to their own suites.
// Oracles:
//  - composition: the same result built step by step from sos_preset, sos_init_phases and sos_val, as the
//    documentation describes it. These functions have their own suites; what is tested here is that the
//    convenience functions wire them together as documented.
//  - mappings: Uniform = 0.5 · erfc( -Normal / sqrt(2) ), integer = ceil( Uniform · imax )
//  - distributions over 10000 positions spread over 10 km with a decorrelation distance of 10 m, i.e.
//    practically independent samples: Normal and Uniform CDF, equal integer counts
// Conventions: positions are rows {x, y, z} in m.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

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

    // Oracle: the documented composition for generator g
    arma::vec ref_randn(double d_a, double d_b, const arma::mat &pos, const arma::mat *pos_b, const std::string &preset,
                        arma::sword seed, arma::uword g, arma::uword n_gen, bool uniform = false)
    {
        d_a = d_a == 0.0 ? 0.1 : d_a, d_b = d_b == 0.0 ? 0.1 : d_b;
        arma::mat freq, phase, val;
        arma::vec acf, dist;
        quadriga_lib::sos_preset(preset, freq, &acf, &dist, d_a);
        // Phases: one stream per device, the second one seeded with seed + 2^62 (bit 62 flipped)
        arma::mat phase_b;
        quadriga_lib::sos_init_phases(freq.n_rows, n_gen, phase, seed);
        quadriga_lib::sos_init_phases(freq.n_rows, n_gen, phase_b, seed + ((arma::sword)1 << 62));
        arma::mat ph_a = phase.col(g), ph_b = phase_b.col(g);
        if (pos_b == nullptr)
            quadriga_lib::sos_val(freq, ph_a, pos, val, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, uniform);
        else if (d_a == d_b)
            quadriga_lib::sos_val(freq, ph_a, pos, val, pos_b, (arma::mat *)nullptr, &acf, &dist, 1.0, uniform);
        else
            quadriga_lib::sos_val(freq, ph_a, pos, val, pos_b, &ph_b, (arma::vec *)nullptr, (arma::vec *)nullptr, d_a / d_b, uniform);
        return val.col(0);
    }
}

TEST_CASE("sos_randn - one device: each generator is the documented composition")
{
    // Repeated and zero decorrelation distances, 11 positions
    arma::vec dd = {5.0, 20.0, 5.0, 0.0, 37.5};
    arma::mat pos = rnd(11, 3, -100.0, 100.0, 1);

    arma::mat val;
    quadriga_lib::sos_randn(dd, pos, val, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 7);
    REQUIRE(val.n_rows == 11);
    REQUIRE(val.n_cols == 5);

    QD_DIAG("\n# diag: sos_randn one device against the composition\ngen,dist_decorr,val_0,ref_0,max_abs_diff\n");
    for (arma::uword g = 0; g < 5; ++g)
    {
        arma::vec ref = ref_randn(dd(g), dd(g), pos, nullptr, "Comb256", 7, g, 5);
        double mx = arma::abs(val.col(g) - ref).max();
        QD_DIAG("%llu,%.1f,%.9f,%.9f,%.2e\n", (unsigned long long)g, dd(g), val(0, g), ref(0), mx);
        INFO("generator " << g);
        // Same kernel on both sides and values that do not depend on the batch: identical up to the
        // rounding of the frequency scaling
        CHECK(mx < 1e-12);
    }
    CHECK(arma::abs(val).max() > 0.5);
    CHECK(arma::abs(val.col(0) - val.col(2)).max() > 0.05); // Equal distances are still independent generators

    // Another preset gives other values
    arma::mat other;
    quadriga_lib::sos_randn(dd, pos, other, (arma::mat *)nullptr, (arma::vec *)nullptr, "Exp128", 7);
    CHECK(arma::abs(other.col(1) - ref_randn(20.0, 20.0, pos, nullptr, "Exp128", 7, 1, 5)).max() < 1e-12);
    CHECK(arma::abs(other - val).max() > 0.05);
}

TEST_CASE("sos_randn - two devices: same map for equal distances, two maps for different distances")
{
    arma::vec dd = {5.0, 20.0, 12.0, 0.0}, db = {5.0, 8.0, 0.0, 0.0};
    arma::mat pa = rnd(9, 3, -60.0, 60.0, 2), pb = rnd(9, 3, -60.0, 60.0, 3);

    // Without dist_decorr_b: all generators on one map
    arma::mat same, swapped;
    quadriga_lib::sos_randn(dd, pa, same, &pb, (arma::vec *)nullptr, "Comb256", 4);
    quadriga_lib::sos_randn(dd, pb, swapped, &pa, (arma::vec *)nullptr, "Comb256", 4);
    REQUIRE(same.n_rows == 9);
    REQUIRE(same.n_cols == 4);
    for (arma::uword g = 0; g < 4; ++g)
    {
        INFO("same map, generator " << g);
        CHECK(arma::abs(same.col(g) - ref_randn(dd(g), dd(g), pa, &pb, "Comb256", 4, g, 4)).max() < 1e-12);
    }
    CHECK(arma::abs(same).max() > 0.3);
    CHECK(arma::approx_equal(same, swapped, "absdiff", 1e-12)); // Symmetric under a swap of the devices

    // With dist_decorr_b: generators 0 and 3 have equal distances (one map), 1 and 2 differ (two maps)
    arma::mat mixed;
    quadriga_lib::sos_randn(dd, pa, mixed, &pb, &db, "Comb256", 4);
    REQUIRE(mixed.n_cols == 4);
    QD_DIAG("\n# diag: sos_randn two devices\ngen,d_a,d_b,max_abs_diff_to_composition\n");
    for (arma::uword g = 0; g < 4; ++g)
    {
        double mx = arma::abs(mixed.col(g) - ref_randn(dd(g), db(g), pa, &pb, "Comb256", 4, g, 4)).max();
        QD_DIAG("%llu,%.1f,%.1f,%.2e\n", (unsigned long long)g, dd(g), db(g), mx);
        INFO("mixed, generator " << g);
        CHECK(mx < 1e-12);
    }
    CHECK(arma::approx_equal(mixed.col(0), same.col(0), "absdiff", 1e-12));
    CHECK(arma::approx_equal(mixed.col(3), same.col(3), "absdiff", 1e-12));
    CHECK(arma::abs(mixed.col(1) - same.col(1)).max() > 0.05);
}

TEST_CASE("sos_rand and sos_randi - mappings of the Normal values")
{
    arma::vec dd = {5.0, 20.0, 0.0}, db = {5.0, 8.0, 3.0};
    arma::mat pa = rnd(200, 3, -300.0, 300.0, 5), pb = rnd(200, 3, -300.0, 300.0, 6);

    arma::mat nrm, uni, itg, nrm2, uni2, itg2;
    quadriga_lib::sos_randn(dd, pa, nrm, (arma::mat *)nullptr, (arma::vec *)nullptr, "Gauss256", 9);
    quadriga_lib::sos_rand(dd, pa, uni, (arma::mat *)nullptr, (arma::vec *)nullptr, "Gauss256", 9);
    quadriga_lib::sos_randi(dd, pa, 6, itg, (arma::mat *)nullptr, (arma::vec *)nullptr, "Gauss256", 9);
    quadriga_lib::sos_randn(dd, pa, nrm2, &pb, &db, "Gauss256", 9);
    quadriga_lib::sos_rand(dd, pa, uni2, &pb, &db, "Gauss256", 9);
    quadriga_lib::sos_randi(dd, pa, 6, itg2, &pb, &db, "Gauss256", 9);
    REQUIRE(uni.n_rows == 200);
    REQUIRE(uni.n_cols == 3);
    REQUIRE(itg.n_rows == 200);
    REQUIRE(itg.n_cols == 3);
    REQUIRE(itg2.n_elem == 600);

    double max_u = 0.0, max_i = 0.0, max_u2 = 0.0, max_i2 = 0.0;
    for (arma::uword i = 0; i < 600; ++i)
    {
        max_u = std::max(max_u, std::abs(uni(i) - 0.5 * std::erfc(-nrm(i) / std::sqrt(2.0))));
        max_i = std::max(max_i, std::abs(itg(i) - std::ceil(uni(i) * 6.0)));
        max_u2 = std::max(max_u2, std::abs(uni2(i) - 0.5 * std::erfc(-nrm2(i) / std::sqrt(2.0))));
        max_i2 = std::max(max_i2, std::abs(itg2(i) - std::ceil(uni2(i) * 6.0)));
    }
    CHECK(max_u < 1e-14);
    CHECK(max_i == 0.0);
    CHECK(max_u2 < 1e-14);
    CHECK(max_i2 == 0.0);
    CHECK(uni.min() > 0.0);
    CHECK(uni.max() < 1.0);
    CHECK(uni.max() - uni.min() > 0.8);
    CHECK(itg.min() == 1.0);
    CHECK(itg.max() == 6.0);

    // imax = 1 leaves a single value
    quadriga_lib::sos_randi(dd, pa, 1, itg);
    CHECK(itg.min() == 1.0);
    CHECK(itg.max() == 1.0);
}

TEST_CASE("sos_randn, sos_rand, sos_randi - distributions over widely spread positions")
{
    // 10000 positions in a cube of 10 km, decorrelation distance 10 m: samples are practically independent.
    // Monte-Carlo error of an empirical CDF with 10000 samples: at most 0.005. A sum of 256 sinusoids
    // deviates from the Normal CDF by less than 0.002. Limit 0.02 (the MATLAB tests use 0.02 to 0.05).
    const arma::uword n = 10000;
    arma::mat pos = rnd(n, 3, 0.0, 10000.0, 7), pos_b = rnd(n, 3, 0.0, 10000.0, 8);
    arma::vec dd = {10.0}, db = {25.0};
    arma::mat nrm, uni, itg, dual_same, dual_two;
    quadriga_lib::sos_randn(dd, pos, nrm, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 1);
    quadriga_lib::sos_rand(dd, pos, uni, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 1);
    quadriga_lib::sos_randi(dd, pos, 10, itg, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 1);
    quadriga_lib::sos_randn(dd, pos, dual_same, &pos, (arma::vec *)nullptr, "Comb256", 1); // As the MATLAB test: both devices at the same place
    quadriga_lib::sos_randn(dd, pos, dual_two, &pos_b, &db, "Comb256", 1);
    REQUIRE(nrm.n_elem == n);
    REQUIRE(uni.n_elem == n);
    REQUIRE(itg.n_elem == n);
    REQUIRE(dual_same.n_elem == n);
    REQUIRE(dual_two.n_elem == n);

    QD_DIAG("\n# diag: sos_randn / sos_rand empirical CDF against the ideal CDF, 10000 positions\n");
    QD_DIAG("x,cdf_randn,cdf_same_map,cdf_two_maps,normal_cdf,u,cdf_rand\n");
    double worst_n = 0.0, worst_u = 0.0, worst_s = 0.0, worst_t = 0.0;
    for (int k = -6; k <= 6; ++k)
    {
        double x = 0.5 * (double)k, u = 0.5 + (double)k / 13.0, cn = 0.0, cs = 0.0, ct = 0.0, cu = 0.0;
        for (arma::uword i = 0; i < n; ++i)
        {
            cn += nrm(i) <= x ? 1.0 / (double)n : 0.0, cs += dual_same(i) <= x ? 1.0 / (double)n : 0.0;
            ct += dual_two(i) <= x ? 1.0 / (double)n : 0.0, cu += uni(i) <= u ? 1.0 / (double)n : 0.0;
        }
        double ideal = 0.5 * std::erfc(-x / std::sqrt(2.0));
        QD_DIAG("%.1f,%.4f,%.4f,%.4f,%.4f,%.3f,%.4f\n", x, cn, cs, ct, ideal, u, cu);
        worst_n = std::max(worst_n, std::abs(cn - ideal)), worst_s = std::max(worst_s, std::abs(cs - ideal));
        worst_t = std::max(worst_t, std::abs(ct - ideal)), worst_u = std::max(worst_u, std::abs(cu - u));
    }
    CHECK(worst_n < 0.02);
    CHECK(worst_s < 0.02);
    CHECK(worst_t < 0.02);
    CHECK(worst_u < 0.02);

    // Integer values: every value from 1 to 10 occurs with a share of 0.1; standard deviation of a count
    // is 30, the limit of 150 is 5 sigma
    arma::vec count(10, arma::fill::zeros);
    double total = 0.0;
    for (arma::uword i = 0; i < n; ++i)
    {
        REQUIRE(itg(i) >= 1.0);
        REQUIRE(itg(i) <= 10.0);
        REQUIRE(itg(i) == std::floor(itg(i)));
        count((arma::uword)itg(i) - 1) += 1.0, total += 1.0;
    }
    QD_DIAG("# diag: sos_randi counts for imax = 10\nvalue,count\n");
    for (arma::uword v = 0; v < 10; ++v)
    {
        QD_DIAG("%llu,%.0f\n", (unsigned long long)(v + 1), count(v));
        INFO("value " << v + 1);
        CHECK(std::abs(count(v) - 1000.0) < 150.0);
    }
    CHECK(total == (double)n);
}

TEST_CASE("sos_randn - seeds, pre-sized output, positions with fewer columns, float")
{
    arma::vec dd = {5.0, 20.0};
    arma::mat pos = rnd(13, 3, -50.0, 50.0, 10);

    arma::mat a, b, c, r1, r2;
    quadriga_lib::sos_randn(dd, pos, a, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 3);
    quadriga_lib::sos_randn(dd, pos, b, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 3);
    quadriga_lib::sos_randn(dd, pos, c, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 4);
    quadriga_lib::sos_randn(dd, pos, r1);
    quadriga_lib::sos_randn(dd, pos, r2);
    REQUIRE(a.n_rows == 13);
    CHECK(arma::abs(a).max() > 0.3);
    CHECK(arma::approx_equal(a, b, "absdiff", 0.0));
    CHECK(arma::abs(a - c).max() > 0.05);
    CHECK(arma::abs(r1 - r2).max() > 0.05); // Seed -1: a new realisation per call
    REQUIRE(r1.n_cols == 2);

    // The same seed gives the same field in separate calls: a subset of the positions gets the same values
    arma::mat sub = pos.rows(3, 7), vs;
    quadriga_lib::sos_randn(dd, sub, vs, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 3);
    CHECK(arma::approx_equal(vs, a.rows(3, 7), "absdiff", 0.0));

    // The first generator does not depend on the number of generators
    arma::vec d1 = {5.0};
    arma::mat v1;
    quadriga_lib::sos_randn(d1, pos, v1, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 3);
    CHECK(arma::approx_equal(v1.col(0), a.col(0), "absdiff", 0.0));

    arma::mat sized(13, 2);
    sized.fill(9.0);
    const double *ptr = sized.memptr();
    quadriga_lib::sos_randn(dd, pos, sized, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 3);
    CHECK(sized.memptr() == ptr);
    CHECK(arma::approx_equal(sized, a, "absdiff", 0.0));

    // Two columns: z = 0
    arma::mat p2 = pos.cols(0, 1), p3 = pos, v2, v3;
    p3.col(2).zeros();
    quadriga_lib::sos_randn(dd, p2, v2, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 3);
    quadriga_lib::sos_randn(dd, p3, v3, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 3);
    CHECK(arma::approx_equal(v2, v3, "absdiff", 0.0));

    arma::fvec df = {5.0f, 20.0f};
    arma::fmat pf = arma::conv_to<arma::fmat>::from(pos), vf, uf, jf;
    quadriga_lib::sos_randn(df, pf, vf, (arma::fmat *)nullptr, (arma::fvec *)nullptr, "Comb256", 3);
    quadriga_lib::sos_rand(df, pf, uf, (arma::fmat *)nullptr, (arma::fvec *)nullptr, "Comb256", 3);
    quadriga_lib::sos_randi(df, pf, 4, jf, (arma::fmat *)nullptr, (arma::fvec *)nullptr, "Comb256", 3);
    REQUIRE(vf.n_rows == 13);
    REQUIRE(vf.n_cols == 2);
    // Positions rounded to float (up to 50 m, 4e-6 m) at frequencies up to 0.15 / m: below 1e-4
    CHECK(arma::abs(arma::conv_to<arma::mat>::from(vf) - a).max() < 1e-4);
    CHECK(uf.min() > 0.0f);
    CHECK(uf.max() < 1.0f);
    CHECK(jf.min() >= 1.0f);
    CHECK(jf.max() <= 4.0f);
}

TEST_CASE("sos_randn, sos_rand, sos_randi - input validation")
{
    arma::vec dd = {5.0, 20.0}, db = {5.0, 8.0};
    arma::mat pos = rnd(6, 3, -50.0, 50.0, 11), pos_b = rnd(6, 3, -50.0, 50.0, 12), val;
    arma::mat *no_mat = nullptr;
    arma::vec *no_vec = nullptr;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_randn(dd, pos, val));
        CHECK_NOTHROW(quadriga_lib::sos_rand(dd, pos, val, &pos_b, &db, "Exp128", 0));
        CHECK_NOTHROW(quadriga_lib::sos_randi(dd, pos, 3, val, &pos_b));
        arma::mat empty;
        arma::vec vempty;
        CHECK_NOTHROW(quadriga_lib::sos_randn(dd, pos, val, &empty, &vempty)); // Empty means omitted
    }
    SECTION("dist_decorr")
    {
        arma::vec empty, neg = {5.0, -1.0};
        CHECK_THROWS_AS(quadriga_lib::sos_randn(empty, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(neg, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_rand(neg, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randi(neg, pos, 3, val), std::invalid_argument);
        arma::vec nan = {5.0, std::nan("")}, inf = {arma::datum::inf, 20.0};
        CHECK_THROWS_AS(quadriga_lib::sos_randn(nan, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(inf, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_rand(inf, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randi(nan, pos, 3, val), std::invalid_argument);
    }
    SECTION("pos")
    {
        arma::mat empty, four(6, 4, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, empty, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, four, val), std::invalid_argument);
    }
    SECTION("pos_b")
    {
        arma::mat rows(5, 3, arma::fill::ones), four(6, 4, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, val, &rows), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, val, &four), std::invalid_argument);
    }
    SECTION("dist_decorr_b")
    {
        arma::vec size = {5.0}, neg = {5.0, -8.0};
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, val, no_mat, &db), std::invalid_argument); // Needs pos_b
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, val, &pos_b, &size), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, val, &pos_b, &neg), std::invalid_argument);
        arma::vec nan = {5.0, std::nan("")}, inf = {5.0, arma::datum::inf};
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, val, &pos_b, &nan), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, val, &pos_b, &inf), std::invalid_argument);
    }
    SECTION("preset")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, val, no_mat, no_vec, "Comb300"), std::invalid_argument);
    }
    SECTION("seed")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, val, no_mat, no_vec, "Comb256", -2), std::invalid_argument);
    }
    SECTION("imax")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_randi(dd, pos, 0, val), std::invalid_argument);
    }
    SECTION("a failed call leaves the output untouched")
    {
        arma::mat keep(2, 2), empty;
        keep.fill(7.0);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, empty, keep), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randn(dd, pos, keep, no_mat, no_vec, "Comb300"), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_randi(dd, pos, 0, keep), std::invalid_argument);
        CHECK(keep.n_rows == 2);
        CHECK(keep(1, 1) == 7.0);
    }
}

TEST_CASE("sos_randn, sos_rand - nearby positions get similar values in separate calls with the same seed")
{
    // Second call 0.1 m away from the first, decorrelation distance 10 m. The difference of the field over a
    // displacement dx has the variance 2·( 1 - R(dx) ); for the Comb ACF R(0.1 m) = exp( -1e-4 ), which gives a
    // standard deviation of 0.014. The limit is 7 standard deviations. With another seed the two calls are
    // independent fields and differ by the order of 1.
    arma::vec dd = {10.0, 10.0, 10.0};
    arma::mat pos = rnd(200, 3, -300.0, 300.0, 30), near = pos, a, b, c, ua, ub;
    near.col(0) += 0.06, near.col(1) -= 0.08; // 0.1 m

    quadriga_lib::sos_randn(dd, pos, a, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 5);
    quadriga_lib::sos_randn(dd, near, b, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 5);
    quadriga_lib::sos_randn(dd, near, c, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 6);
    quadriga_lib::sos_rand(dd, pos, ua, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 5);
    quadriga_lib::sos_rand(dd, near, ub, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 5);
    REQUIRE(a.n_rows == 200);
    REQUIRE(b.n_cols == 3);

    double same = arma::abs(a - b).max(), other = arma::abs(a - c).max(), uni = arma::abs(ua - ub).max(), rms = 0.0;
    for (arma::uword i = 0; i < a.n_elem; ++i)
        rms += (a(i) - b(i)) * (a(i) - b(i)) / (double)a.n_elem;
    rms = std::sqrt(rms);
    QD_DIAG("\n# diag: sos_randn at positions 0.1 m apart: same seed max %.4f, rms %.4f (expected 0.014), other seed max %.3f, Uniform max %.4f\n", same, rms, other, uni);
    CHECK(same < 0.1);
    CHECK(rms < 0.03);
    CHECK(rms > 0.005); // The positions did move
    CHECK(other > 1.0);
    CHECK(uni < 0.04); // The Normal density is at most 0.4, so the Uniform values differ by at most 0.4 times as much
    CHECK(arma::abs(a).max() > 1.0);
}


TEST_CASE("sos_randn, sos_rand, sos_randi - on two maps the value of a generator does not depend on the number of generators")
{
    // The second device draws its phases from its own stream, sos_init_phases( n_coeff, n_gen, seed + 2^62 ),
    // so generator g has the same phases for both devices whatever follows it in the call
    arma::vec dd4 = {5.0, 20.0, 12.0, 7.0}, db4 = {9.0, 8.0, 12.0, 30.0}; // Generator 2 stays on one map
    arma::mat pa = rnd(40, 3, -60.0, 60.0, 40), pb = rnd(40, 3, -60.0, 60.0, 41);

    arma::mat n4, u4, i4;
    quadriga_lib::sos_randn(dd4, pa, n4, &pb, &db4, "Comb256", 9);
    quadriga_lib::sos_rand(dd4, pa, u4, &pb, &db4, "Comb256", 9);
    quadriga_lib::sos_randi(dd4, pa, 1000, i4, &pb, &db4, "Comb256", 9);
    REQUIRE(n4.n_cols == 4);
    CHECK(arma::abs(n4).max() > 0.5);

    for (arma::uword n_gen = 1; n_gen <= 3; ++n_gen)
    {
        arma::vec dd = dd4.head(n_gen), db = db4.head(n_gen);
        arma::mat n, u, i;
        quadriga_lib::sos_randn(dd, pa, n, &pb, &db, "Comb256", 9);
        quadriga_lib::sos_rand(dd, pa, u, &pb, &db, "Comb256", 9);
        quadriga_lib::sos_randi(dd, pa, 1000, i, &pb, &db, "Comb256", 9);
        REQUIRE(n.n_cols == n_gen);
        INFO("n_gen " << n_gen);
        // The generators of a call are evaluated in groups of equal decorrelation distances and each value
        // is independent of its group, so the columns are bit-identical
        CHECK(arma::approx_equal(n, n4.cols(0, n_gen - 1), "absdiff", 0.0));
        CHECK(arma::approx_equal(u, u4.cols(0, n_gen - 1), "absdiff", 0.0));
        CHECK(arma::approx_equal(i, i4.cols(0, n_gen - 1), "absdiff", 0.0));
    }

    // The two streams differ: a generator on two maps with equal positions is not twice the same field.
    // With the same phases for both devices the value would be sqrt(2) times the single-device value.
    arma::vec d1 = {10.0}, d1b = {10.5};
    arma::mat one, two;
    quadriga_lib::sos_randn(d1, pa, one, (arma::mat *)nullptr, (arma::vec *)nullptr, "Comb256", 9);
    quadriga_lib::sos_randn(d1, pa, two, &pa, &d1b, "Comb256", 9);
    CHECK(arma::abs(two - std::sqrt(2.0) * one).max() > 0.5);

    // Without a seed both streams are random
    arma::mat r1, r2;
    quadriga_lib::sos_randn(dd4, pa, r1, &pb, &db4);
    quadriga_lib::sos_randn(dd4, pa, r2, &pb, &db4);
    CHECK(arma::abs(r1.col(0) - r2.col(0)).max() > 0.05);
}

TEST_CASE("sos_randn, sos_rand, sos_randi - the output may be the same object as a position matrix")
{
    // 3 generators and 3 position columns: the output has the size of the input and is not reallocated,
    // 5 generators: it is resized. Both must give the values of a call with a separate output.
    arma::mat pos = rnd(50, 3, -60.0, 60.0, 50), pos_b = rnd(50, 3, -60.0, 60.0, 51);
    for (arma::uword n_gen : {3, 5})
    {
        arma::vec dd(n_gen), db(n_gen);
        for (arma::uword g = 0; g < n_gen; ++g)
            dd(g) = 5.0 + 3.0 * (double)g, db(g) = (g == 1) ? 30.0 : dd(g); // Generator 1 on two maps
        arma::mat ref_n, ref_u, ref_i;
        quadriga_lib::sos_randn(dd, pos, ref_n, &pos_b, &db, "Comb128", 3);
        quadriga_lib::sos_rand(dd, pos, ref_u, &pos_b, &db, "Comb128", 3);
        quadriga_lib::sos_randi(dd, pos, 9, ref_i, &pos_b, &db, "Comb128", 3);
        REQUIRE(ref_n.n_cols == n_gen);
        CHECK(arma::abs(ref_n).max() > 0.5);

        INFO("n_gen " << n_gen);
        arma::mat a = pos, b = pos_b, c = pos;
        quadriga_lib::sos_randn(dd, a, a, &pos_b, &db, "Comb128", 3); // val is 'pos'
        CHECK(arma::approx_equal(a, ref_n, "absdiff", 0.0));
        quadriga_lib::sos_rand(dd, pos, b, &b, &db, "Comb128", 3); // val is 'pos_b'
        CHECK(arma::approx_equal(b, ref_u, "absdiff", 0.0));
        quadriga_lib::sos_randi(dd, c, 9, c, &pos_b, &db, "Comb128", 3);
        CHECK(arma::approx_equal(c, ref_i, "absdiff", 0.0));
    }
}
