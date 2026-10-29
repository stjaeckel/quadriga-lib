// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "ssf_*"

// Verification suite for the phase sets of the small-scale model: quadriga_lib::ssf_init_phases and
// quadriga_lib::ssf_draw (test cases "ssf_init_phases - ..." and "ssf_draw - ...").
//
// ssf_init_phases pins down: the size of the shared cluster set (9·N generators) and of a per-frequency set (N
// generators) with N = NumClusters - 1, including the empty sets of NumClusters = 1; the seed rule (slice 0 equals
// sos_init_phases with the seed derived by SplitMix64 from the large-scale seed and the set); the mirror relation of
// slice 1 (blocks AOD <-> AOA and ZOD <-> ZOA of the signs and the offsets, delay and shadowing themselves); the
// append-only property of the phases; the independence of the sets from each other and from the large-scale set;
// determinism and seed -1; the input validation.
// ssf_draw pins down: the values of both set types in single mobility, the Uniform map of the delay block, the use
// of SC_lambda for every generator (0 used as 0.1 m), the two-device formulas of dual mobility, the swap
// reciprocity (departure and arrival blocks exchanged), the independence of a column from the phases of the other
// columns, floors as lsf_draw, 2D and 3D mode, the statistics (marginals, signs, ACF at the decorrelation distance,
// zero correlation of two floors at equal positions), sizes, empty input, pre-sized outputs and the input validation.
// Oracles, independent of the functions:
//  - the seed rule, typed here: splitmix64(x) with the constants 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9,
//    0x94D049BB133111EB; seed_set = splitmix64(splitmix64(seed) ^ tag) >> 1 with tag 1 for the shared set and 2 + k
//    for the set of frequency k; checked first against fixed values
//  - the column layout c = b·N + n and the mirror table of the blocks: 0 -> 0, 1 <-> 2, 3 <-> 4, 5 <-> 6, 7 <-> 8
//  - ref_sos(F, phi, x) = sqrt(2/N) · sum_n cos( 2·pi · F(n,:)·x + phi(n) + off(n) ), typed here, with the
//    frequencies of sos_preset at the decorrelation distance SC_lambda
//  - dual mobility: (ref_sos(F, P0_c, R) + ref_sos(F, P0_c, T)) / sqrt(2 + 2·A(D)) for self-mirrored columns,
//    (ref_sos(F, P0_c, R) + ref_sos(F, P0_m, T)) / sqrt(2) for the pairs, A the ACF table of sos_preset
//    interpolated linearly by sos_acfi, D the horizontal Tx-Rx distance
//  - the Uniform map 0.5·erfc(-v/sqrt(2)); floor offsets 2·pi·k·n/N + psi_b(n) as in the large-scale tests
//  - statistics: Normal(0,1), Uniform(0,1), the ACF table of sos_preset; for Uniform columns the Pearson
//    correlation (6/pi)·asin(rho/2) of two Uniform values whose Normal values have the correlation rho
// Tolerances of ssf_draw values: the sinusoids are evaluated with the AVX2 kernel where available (deviation below
// 4e-7·sqrt(2·n_coeff) = 9e-6 for 256 sinusoids): Normal columns 2e-5, Uniform columns 1e-5.
// Conventions: positions are rows {x, y, z} in meters; the scenarios set the decorrelation distances of the
// large-scale set to values different from SC_lambda, so that a use of the wrong distance changes the result; the
// phase cubes of the ssf_draw cases are raw phases of sos_init_phases with the typed mirror in slice 1.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_sos.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    const double pi = 3.141592653589793;

    // Mirror block of each block of the shared set
    const arma::uword mirror_b[9] = {0, 2, 1, 4, 3, 6, 5, 8, 7};

    // Scenario with NumClusters and SC_lambda; the large-scale distances differ from SC_lambda
    std::unordered_map<std::string, double> scen(double num_clusters, double sc_lambda)
    {
        return {{"NumClusters", num_clusters}, {"SC_lambda", sc_lambda}, {"DS_lambda", 11.0}, {"KF_lambda", 12.0},
                {"SF_lambda", 13.0}, {"AS_D_lambda", 14.0}, {"AS_A_lambda", 16.0}, {"ES_D_lambda", 17.0},
                {"ES_A_lambda", 18.0}, {"XPR_lambda", 19.0}, {"absTOA_lambda", 21.0}};
    }

    std::uint64_t splitmix64(std::uint64_t x)
    {
        std::uint64_t z = x + 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // Seed of a set: tag 1 for the shared set (set_id = -1), 2 + k for frequency k
    arma::sword seed_set(arma::sword seed, arma::sword set_id)
    {
        const std::uint64_t tag = (set_id < 0) ? 1ULL : 2ULL + (std::uint64_t)set_id;
        return (arma::sword)(splitmix64(splitmix64((std::uint64_t)seed) ^ tag) >> 1);
    }

    arma::uword n_gen_of(arma::uword N, bool shared) { return shared ? 9 * N : N; }

    // Mirror column of column c (layout c = b·N + n)
    arma::uword mirror_col(arma::uword c, arma::uword N, bool shared)
    {
        return shared ? mirror_b[c / N] * N + c % N : c;
    }

    // True if column c is in the Uniform block (block 0 of the shared set)
    bool is_uniform(arma::uword c, arma::uword N, bool shared) { return shared && c < N; }

    // Phase cube: raw phases in slice 0, typed mirror in slice 1
    arma::cube make_phase(arma::uword n_coeff, arma::uword N, bool shared, arma::sword seed)
    {
        const arma::uword n_gen = n_gen_of(N, shared);
        arma::mat P0;
        quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, P0, seed);
        arma::cube phase(n_coeff, n_gen, 2);
        phase.slice(0) = P0;
        for (arma::uword c = 0; c < n_gen; ++c)
            phase.slice(1).col(c) = P0.col(mirror_col(c, N, shared));
        return phase;
    }

    // Oracle: sum of sinusoids sqrt(2/N) · sum_n cos( 2·pi · (F(n,0)·x + F(n,1)·y + F(n,2)·z) + phi(n) + off(n) )
    double ref_sos(const arma::mat &F, const double *phi, double x, double y, double z, const arma::vec *off = nullptr)
    {
        double s = 0.0;
        for (arma::uword n = 0; n < F.n_rows; ++n)
            s += std::cos(2.0 * pi * (F(n, 0) * x + F(n, 1) * y + F(n, 2) * z) + phi[n] + (off ? (*off)(n) : 0.0));
        return std::sqrt(2.0 / (double)F.n_rows) * s;
    }

    double unif(double v) { return 0.5 * std::erfc(-v / std::sqrt(2.0)); }

    struct preset_tab
    {
        arma::mat F;
        arma::vec acf, dist;
    };

    preset_tab get_preset(const std::string &preset, double d)
    {
        preset_tab t;
        quadriga_lib::sos_preset<double>(preset, t.F, &t.acf, &t.dist, d);
        return t;
    }

    double acf_at(const preset_tab &t, double D)
    {
        arma::vec v;
        quadriga_lib::sos_acfi<double>(t.acf, t.dist, arma::vec{D}, v);
        return v(0);
    }

    // Floor offsets typed from the rule: 2·pi·k·n/N + psi_b(n)
    arma::vec floor_offset(arma::uword N, arma::uword q)
    {
        const arma::uword k = q % N, b = q / N;
        arma::vec d(N);
        std::mt19937_64 gen(b);
        for (arma::uword n = 0; n < N; ++n)
        {
            d(n) = 2.0 * pi * (double)k * (double)n / (double)N;
            if (b > 0)
                d(n) += 2.0 * pi * ((double)(gen() >> 11) / 9007199254740992.0);
        }
        return d;
    }

    // Regular grid of nx · ny positions with the given spacing at height z, offset from the axes
    arma::mat grid(arma::uword nx, arma::uword ny, double spacing, double z)
    {
        arma::mat p(nx * ny, 3);
        for (arma::uword i = 0; i < nx; ++i)
            for (arma::uword j = 0; j < ny; ++j)
                p(i * ny + j, 0) = 3.7 + spacing * (double)i, p(i * ny + j, 1) = -2.1 + spacing * (double)j, p(i * ny + j, 2) = z;
        return p;
    }

    double corr(const arma::vec &a, const arma::vec &b)
    {
        double ma = arma::mean(a), mb = arma::mean(b), sab = 0.0, saa = 0.0, sbb = 0.0;
        for (arma::uword i = 0; i < a.n_elem; ++i)
            sab += (a(i) - ma) * (b(i) - mb), saa += (a(i) - ma) * (a(i) - ma), sbb += (b(i) - mb) * (b(i) - mb);
        return sab / std::sqrt(saa * sbb);
    }

    double stddev(const arma::vec &a)
    {
        double m = arma::mean(a), s = 0.0;
        for (arma::uword i = 0; i < a.n_elem; ++i)
            s += (a(i) - m) * (a(i) - m);
        return std::sqrt(s / (double)(a.n_elem - 1));
    }

    bool equal(const arma::mat &a, const arma::mat &b)
    {
        return a.n_rows == b.n_rows && a.n_cols == b.n_cols && arma::all(arma::vectorise(a == b));
    }

    bool equal(const arma::cube &a, const arma::cube &b)
    {
        return a.n_rows == b.n_rows && a.n_cols == b.n_cols && a.n_slices == b.n_slices && arma::all(arma::vectorise(a == b));
    }

    // 5 links with distinct coordinates, off the axes, z != 0
    arma::mat rx5()
    {
        return {{12.5, -40.0, 1.5}, {-120.0, 33.0, 4.5}, {250.0, 300.0, 22.0}, {-7.25, -299.0, 1.8}, {88.0, 0.5, 10.0}};
    }
}

// ------------------------------------------------------------------------------------------------------------------
// ssf_init_phases
// ------------------------------------------------------------------------------------------------------------------

TEST_CASE("ssf_init_phases - output size follows the set and the scenario")
{
    const double num_clusters[4] = {1.0, 2.0, 13.0, 26.0};
    const char *presets[2] = {"Comb256", "Exp512"};
    const arma::uword n_coeff[2] = {256, 512};
    for (int p = 0; p < 2; ++p)
        for (double nc : num_clusters)
        {
            const arma::uword N = (arma::uword)nc - 1;
            const auto sp = scen(nc, 10.0);
            arma::cube shared, perf;
            quadriga_lib::ssf_init_phases(sp, shared, 5, -1, presets[p]);
            quadriga_lib::ssf_init_phases(sp, perf, 5, 3, presets[p]);
            INFO("preset " << presets[p] << ", NumClusters " << nc);
            CHECK(shared.n_rows == n_coeff[p]);
            CHECK(shared.n_cols == 9 * N);
            CHECK(shared.n_slices == 2);
            CHECK(perf.n_rows == n_coeff[p]);
            CHECK(perf.n_cols == N);
            CHECK(perf.n_slices == 2);
        }

    // A pre-sized output with stale content is written in place and completely
    const auto sp = scen(13.0, 10.0);
    arma::cube ref, pre(256, 108, 2);
    pre.fill(99.0);
    const double *mem = pre.memptr();
    quadriga_lib::ssf_init_phases(sp, ref, 3);
    quadriga_lib::ssf_init_phases(sp, pre, 3);
    CHECK(pre.memptr() == mem);
    CHECK(equal(pre, ref));
}

TEST_CASE("ssf_init_phases - slice 0 equals sos_init_phases with the derived seed")
{
    // The typed seed rule against fixed values
    CHECK(seed_set(42, -1) == (arma::sword)8766244108781592946LL);
    CHECK(seed_set(42, 0) == (arma::sword)4119046106699552547LL);
    CHECK(seed_set(42, 1) == (arma::sword)9018399064009245349LL);
    CHECK(seed_set(0, -1) == (arma::sword)313702574736366215LL);
    CHECK(seed_set(0, 0) == (arma::sword)7774984521324166428LL);

    const auto sp = scen(13.0, 10.0);
    const arma::sword seeds[2] = {42, 0};
    const arma::sword sets[3] = {-1, 0, 1};
    for (arma::sword seed : seeds)
        for (arma::sword set : sets)
        {
            const arma::uword n_gen = n_gen_of(12, set < 0);
            arma::cube phase;
            quadriga_lib::ssf_init_phases(sp, phase, seed, set);
            arma::mat P0;
            quadriga_lib::sos_init_phases<double>(256, n_gen, P0, seed_set(seed, set));
            INFO("seed " << seed << ", set " << set);
            REQUIRE(phase.n_cols == n_gen);
            CHECK(equal(arma::mat(phase.slice(0)), P0));
        }
}

TEST_CASE("ssf_init_phases - slice 1 is the mirror of slice 0")
{
    const auto sp = scen(13.0, 10.0);
    const arma::uword N = 12;
    for (int s = 0; s < 2; ++s)
    {
        const bool shared = s == 0;
        arma::cube phase;
        quadriga_lib::ssf_init_phases(sp, phase, 17, shared ? -1 : 2);
        REQUIRE(phase.n_cols == n_gen_of(N, shared));
        INFO("shared " << shared);
        for (arma::uword c = 0; c < phase.n_cols; ++c)
        {
            INFO("column " << c);
            CHECK(arma::all(phase.slice(1).col(c) == phase.slice(0).col(mirror_col(c, N, shared))));
        }
        if (shared) // the pairs are not self-mirrored: slice 1 differs from slice 0 there
            CHECK(arma::any(phase.slice(1).col(N) != phase.slice(0).col(N)));
    }
}

TEST_CASE("ssf_init_phases - appending generators keeps the existing columns")
{
    const auto sp = scen(13.0, 10.0);
    const arma::sword sets[2] = {-1, 0};
    for (arma::sword set : sets)
    {
        arma::cube phase;
        quadriga_lib::ssf_init_phases(sp, phase, 42, set);
        const arma::uword n_gen = phase.n_cols;
        arma::mat P;
        quadriga_lib::sos_init_phases<double>(256, n_gen + 9, P, seed_set(42, set));
        INFO("set " << set);
        CHECK(equal(arma::mat(phase.slice(0)), arma::mat(P.cols(0, n_gen - 1))));
    }
}

TEST_CASE("ssf_init_phases - the sets are distinct from each other and from the large-scale set")
{
    const auto sp = scen(13.0, 10.0);
    arma::cube lsf, shared, f0, f1;
    quadriga_lib::lsf_init_phases(sp, lsf, 42, nullptr, nullptr, false, false);
    quadriga_lib::ssf_init_phases(sp, shared, 42, -1);
    quadriga_lib::ssf_init_phases(sp, f0, 42, 0);
    quadriga_lib::ssf_init_phases(sp, f1, 42, 1);

    auto any_equal_col = [](const arma::cube &a, const arma::cube &b)
    {
        for (arma::uword i = 0; i < a.n_cols; ++i)
            for (arma::uword j = 0; j < b.n_cols; ++j)
                if (arma::all(a.slice(0).col(i) == b.slice(0).col(j)))
                    return true;
        return false;
    };
    CHECK_FALSE(any_equal_col(shared, lsf));
    CHECK_FALSE(any_equal_col(shared, f0));
    CHECK_FALSE(any_equal_col(shared, f1));
    CHECK_FALSE(any_equal_col(f0, f1));
    CHECK_FALSE(any_equal_col(f0, lsf));

    // Sample correlation of the phase values of the two per-frequency sets: n = 256·12, bound 5/sqrt(n) = 0.090
    const arma::vec a = arma::vectorise(f0.slice(0)), b = arma::vectorise(f1.slice(0));
    const double c = corr(a, b), bound = 5.0 / std::sqrt((double)a.n_elem);
    QD_DIAG("\n# diag: ssf_init_phases correlation of the phases of frequency sets 0 and 1\nn,corr,bound\n%llu,%.4f,%.4f\n",
            (unsigned long long)a.n_elem, c, bound);
    CHECK(std::abs(c) < bound);
}

TEST_CASE("ssf_init_phases - the same seed gives the same phases, seed -1 new ones")
{
    const auto sp = scen(13.0, 10.0);
    arma::cube a, b, c, d;
    quadriga_lib::ssf_init_phases(sp, a, 7);
    quadriga_lib::ssf_init_phases(sp, b, 7);
    CHECK(equal(a, b));
    quadriga_lib::ssf_init_phases(sp, c, -1);
    quadriga_lib::ssf_init_phases(sp, d, -1);
    REQUIRE(c.n_cols == 108);
    CHECK(arma::any(arma::vectorise(c != d)));
    CHECK(arma::any(arma::vectorise(c != a)));
}

TEST_CASE("ssf_init_phases - input validation")
{
    const auto sp = scen(13.0, 10.0);
    arma::cube phase;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::ssf_init_phases(sp, phase, 1, -1));
        CHECK_NOTHROW(quadriga_lib::ssf_init_phases(sp, phase, -1, 4, "Exp512"));
        CHECK_NOTHROW(quadriga_lib::ssf_init_phases({}, phase, 1)); // empty scenpar: all defaults
    }
    SECTION("seed")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_init_phases(sp, phase, -2), std::invalid_argument,
                             Message("Input 'seed' must be >= -1."));
    }
    SECTION("set_id")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_init_phases(sp, phase, 1, -2), std::invalid_argument,
                             Message("Input 'set_id' must be >= -1."));
    }
    SECTION("preset")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_init_phases(sp, phase, 1, -1, "Comb999"), std::invalid_argument,
                             Message("Input 'preset' is not a known preset."));
    }
    SECTION("NumClusters 0")
    {
        auto bad = sp;
        bad["NumClusters"] = 0.0;
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_init_phases(bad, phase, 1), std::invalid_argument,
                             Message("Value of 'NumClusters' must be an integer from 1 to 1000."));
    }
    SECTION("scenpar value")
    {
        auto bad = sp;
        bad["DS_lambda"] = -1.0; // a key not used here throws as well
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_init_phases(bad, phase, 1), std::invalid_argument,
                             Message("Value of 'DS_lambda' must be finite and >= 0."));
    }
    SECTION("the output is untouched on error")
    {
        arma::cube keep(2, 2, 2, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::ssf_init_phases(sp, keep, 1, -2), std::invalid_argument);
        CHECK(keep.n_rows == 2);
        CHECK(arma::all(arma::vectorise(keep) == 1.0));
    }
}

// ------------------------------------------------------------------------------------------------------------------
// ssf_draw
// ------------------------------------------------------------------------------------------------------------------

TEST_CASE("ssf_draw - single mobility values equal the sum of sinusoids")
{
    const auto sp = scen(4.0, 15.0); // N = 3
    const arma::uword N = 3;
    const arma::mat rx = rx5(), tx = {{10.0, 20.0, 25.0}};
    const arma::mat F = get_preset("Comb256", 15.0).F;

    QD_DIAG("\n# diag: ssf_draw single mobility (2D) against ref_sos, max abs difference per column\n");
    QD_DIAG("set,column,max_diff\n");
    for (int s = 0; s < 2; ++s)
    {
        const bool shared = s == 0;
        const arma::sword set = shared ? -1 : 1;
        const arma::cube phase = make_phase(256, N, shared, 51 + s);
        arma::mat rnd, rnd2;
        quadriga_lib::ssf_draw(sp, phase, tx, rx, rnd, set);
        REQUIRE(rnd.n_rows == 5);
        REQUIRE(rnd.n_cols == n_gen_of(N, shared));
        for (arma::uword c = 0; c < rnd.n_cols; ++c)
        {
            const bool u = is_uniform(c, N, shared);
            double max_d = 0.0;
            for (arma::uword l = 0; l < 5; ++l)
            {
                double v = ref_sos(F, phase.slice(0).colptr(c), rx(l, 0), rx(l, 1), 0.0);
                if (u)
                    v = unif(v);
                max_d = std::max(max_d, std::abs(rnd(l, c) - v));
            }
            QD_DIAG("%lld,%llu,%.3e\n", (long long)set, (unsigned long long)c, max_d);
            INFO("set " << set << ", column " << c);
            CHECK(max_d < (u ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
        }
        quadriga_lib::ssf_draw(sp, phase, tx, rx, rnd2, set);
        CHECK(equal(rnd, rnd2)); // determinism
    }
}

TEST_CASE("ssf_draw - all generators use SC_lambda, 0 is used as 0.1 m")
{
    const arma::mat rx = rx5(), tx = {{10.0, 20.0, 25.0}};
    const arma::cube phase = make_phase(256, 3, true, 53);
    arma::mat a, b;
    quadriga_lib::ssf_draw(scen(4.0, 0.0), phase, tx, rx, a);
    quadriga_lib::ssf_draw(scen(4.0, 0.1), phase, tx, rx, b);
    CHECK(equal(a, b));

    // The large-scale distances do not enter: changing them leaves the values unchanged
    auto sp = scen(4.0, 15.0);
    quadriga_lib::ssf_draw(sp, phase, tx, rx, a);
    sp["DS_lambda"] = 40.0, sp["AS_A_lambda"] = 3.0, sp["SF_lambda"] = 7.0;
    quadriga_lib::ssf_draw(sp, phase, tx, rx, b);
    CHECK(equal(a, b));
}

TEST_CASE("ssf_draw - dual mobility values follow the two-device formulas")
{
    const auto sp = scen(3.0, 15.0); // N = 2
    const arma::uword N = 2;
    const preset_tab t = get_preset("Comb256", 15.0);
    // 5 links, Tx per link; link 2 has Tx = Rx, link 4 a distance of more than 5 decorrelation distances
    const arma::mat rx = {{10.0, -20.0, 1.5}, {-55.0, 30.0, 3.0}, {100.0, 100.0, 1.5}, {-5.0, -5.0, 12.0}, {0.3, 0.7, 1.5}};
    const arma::mat tx = {{12.0, -15.0, 25.0}, {-40.0, 10.0, 30.0}, {100.0, 100.0, 1.5}, {-25.0, 3.0, 12.0}, {150.0, -80.0, 10.0}};
    const arma::uvec fl = {3, 0, 259, 3, 0};
    const arma::vec off3 = floor_offset(256, 3), off259 = floor_offset(256, 259);

    QD_DIAG("\n# diag: ssf_draw dual mobility against the two-device formulas, max abs difference\n");
    QD_DIAG("set,floors,max_diff\n");
    for (int s = 0; s < 2; ++s)
        for (int f = 0; f < 2; ++f)
        {
            const bool shared = s == 0;
            const arma::cube phase = make_phase(256, N, shared, 55 + s);
            arma::mat rnd;
            quadriga_lib::ssf_draw(sp, phase, tx, rx, rnd, shared ? -1 : 0, f ? &fl : nullptr, true);
            REQUIRE(rnd.n_rows == 5);
            REQUIRE(rnd.n_cols == n_gen_of(N, shared));
            double max_d = 0.0;
            for (arma::uword l = 0; l < 5; ++l)
                for (arma::uword c = 0; c < rnd.n_cols; ++c)
                {
                    const arma::vec *off = nullptr;
                    if (f && fl(l) == 3)
                        off = &off3;
                    if (f && fl(l) == 259)
                        off = &off259;
                    const arma::uword m = mirror_col(c, N, shared);
                    const double a = ref_sos(t.F, phase.slice(0).colptr(c), rx(l, 0), rx(l, 1), 0.0, off);
                    const double b = ref_sos(t.F, phase.slice(0).colptr(m), tx(l, 0), tx(l, 1), 0.0, off);
                    const double D = std::sqrt((rx(l, 0) - tx(l, 0)) * (rx(l, 0) - tx(l, 0)) + (rx(l, 1) - tx(l, 1)) * (rx(l, 1) - tx(l, 1)));
                    double v = (m == c) ? (a + b) / std::sqrt(2.0 + 2.0 * acf_at(t, D)) : (a + b) / std::sqrt(2.0);
                    const bool u = is_uniform(c, N, shared);
                    if (u)
                        v = unif(v);
                    max_d = std::max(max_d, std::abs(rnd(l, c) - v));
                    INFO("shared " << shared << ", floors " << f << ", link " << l << ", column " << c);
                    CHECK(std::abs(rnd(l, c) - v) < (u ? 1e-5 : 2e-5));
                }
            QD_DIAG("%d,%d,%.3e\n", shared ? -1 : 0, f, max_d);
        }
}

TEST_CASE("ssf_draw - swapping Tx and Rx exchanges the departure and arrival blocks")
{
    const auto sp = scen(4.0, 12.0); // N = 3
    const arma::uword N = 3;
    const arma::mat A = rx5();
    arma::mat B = rx5() * 0.7 + 20.0;
    B.col(2).fill(25.0);
    const arma::uvec fl = {0, 2, 2, 300, 0};

    for (int s = 0; s < 2; ++s)
        for (int f = 0; f < 2; ++f)
        {
            const bool shared = s == 0;
            const arma::sword set = shared ? -1 : 2;
            arma::cube phase;
            quadriga_lib::ssf_init_phases(sp, phase, 61, set);
            arma::mat ab, ba;
            quadriga_lib::ssf_draw(sp, phase, B, A, ab, set, f ? &fl : nullptr, true); // tx = B, rx = A
            quadriga_lib::ssf_draw(sp, phase, A, B, ba, set, f ? &fl : nullptr, true); // tx = A, rx = B
            REQUIRE(ab.n_cols == n_gen_of(N, shared));
            REQUIRE(ba.n_cols == n_gen_of(N, shared));
            for (arma::uword c = 0; c < ab.n_cols; ++c)
            {
                INFO("shared " << shared << ", floors " << f << ", column " << c);
                CHECK(arma::all(ab.col(c) == ba.col(mirror_col(c, N, shared))));
            }
        }
}

TEST_CASE("ssf_draw - a column does not depend on the phases of the other columns")
{
    const auto sp = scen(4.0, 12.0); // N = 3, 27 columns
    const arma::uword N = 3, n_gen = 27;
    const arma::mat rx = rx5();
    arma::mat tx = rx5() + 7.0;
    tx.col(2).fill(25.0);
    const arma::cube phase = make_phase(256, N, true, 63);
    const arma::uword cols[3] = {1, 4, 20}; // delay block, sign AOA, offset ZOD

    arma::mat ref1, ref2;
    quadriga_lib::ssf_draw(sp, phase, tx, rx, ref1);
    quadriga_lib::ssf_draw(sp, phase, tx, rx, ref2, -1, nullptr, true);
    for (arma::uword c : cols)
    {
        // New raw phases for all columns except c and its mirror (both enter the dual-mobility value of c), slice 1
        // mirrored again
        const arma::uword m = mirror_col(c, N, true);
        arma::mat P0;
        quadriga_lib::sos_init_phases<double>(256, n_gen, P0, 1000 + (arma::sword)c);
        P0.col(c) = phase.slice(0).col(c), P0.col(m) = phase.slice(0).col(m);
        arma::cube other(256, n_gen, 2);
        other.slice(0) = P0;
        for (arma::uword g = 0; g < n_gen; ++g)
            other.slice(1).col(g) = P0.col(mirror_col(g, N, true));

        arma::mat v1, v2;
        quadriga_lib::ssf_draw(sp, other, tx, rx, v1);
        quadriga_lib::ssf_draw(sp, other, tx, rx, v2, -1, nullptr, true);
        INFO("column " << c);
        CHECK(arma::all(v1.col(c) == ref1.col(c)));
        CHECK(arma::all(v2.col(c) == ref2.col(c)));
        CHECK(arma::all(v2.col(m) == ref2.col(m)));
        CHECK(arma::any(arma::vectorise(v1 != ref1))); // the other columns did change
    }
}

TEST_CASE("ssf_draw - floors as in lsf_draw")
{
    const arma::mat tx = {{0.0, 0.0, 25.0}};
    {
        const auto sp = scen(4.0, 12.0);
        const arma::cube phase = make_phase(256, 3, true, 65);
        const arma::mat rx = rx5();
        const arma::uvec fl0(5, arma::fill::zeros);
        arma::mat a, b;
        quadriga_lib::ssf_draw(sp, phase, tx, rx, a);
        quadriga_lib::ssf_draw(sp, phase, tx, rx, b, -1, &fl0);
        CHECK(equal(a, b));
    }

    // Floors 0 and 1 at equal positions: n = 10^4 links spaced 5 decorrelation distances, bound 5/sqrt(n) = 0.05
    const auto sp = scen(3.0, 10.0); // N = 2, 18 columns
    const arma::cube phase = make_phase(256, 2, true, 66);
    const arma::mat g0 = grid(100, 100, 50.0, 1.5);
    const arma::uword n = g0.n_rows;
    const arma::mat rx = arma::join_cols(g0, g0);
    arma::uvec fl(2 * n, arma::fill::zeros);
    fl.subvec(n, 2 * n - 1).fill(1);
    arma::mat rnd;
    quadriga_lib::ssf_draw(sp, phase, tx, rx, rnd, -1, &fl);
    REQUIRE(rnd.n_rows == 2 * n);
    QD_DIAG("\n# diag: ssf_draw correlation between floors 0 and 1 at equal positions\ncolumn,corr\n");
    for (arma::uword c = 0; c < rnd.n_cols; ++c)
    {
        const double r = corr(rnd.col(c).rows(0, n - 1), rnd.col(c).rows(n, 2 * n - 1));
        QD_DIAG("%llu,%.4f\n", (unsigned long long)c, r);
        INFO("column " << c);
        CHECK(std::abs(r) < 0.05);
    }
}

TEST_CASE("ssf_draw - 2D mode does not depend on z, 3D mode does")
{
    const auto sp = scen(4.0, 12.0);
    const arma::cube phase = make_phase(256, 3, true, 67);
    const arma::mat rx = rx5();
    arma::mat tx = rx5() + 9.0, rx_up = rx, tx_up = tx;
    rx_up.col(2) += 3.0, tx_up.col(2) += 3.0;

    for (int dual = 0; dual < 2; ++dual)
    {
        arma::mat a, b;
        quadriga_lib::ssf_draw(sp, phase, tx, rx, a, -1, nullptr, dual == 1);
        quadriga_lib::ssf_draw(sp, phase, tx_up, rx_up, b, -1, nullptr, dual == 1);
        INFO("dual mobility " << dual);
        CHECK(equal(a, b));
        quadriga_lib::ssf_draw(sp, phase, tx, rx, a, -1, nullptr, dual == 1, "Comb256", false);
        quadriga_lib::ssf_draw(sp, phase, tx_up, rx_up, b, -1, nullptr, dual == 1, "Comb256", false);
        REQUIRE(a.n_cols == 27);
        for (arma::uword c = 0; c < 27; ++c)
        {
            INFO("3D, column " << c);
            CHECK(arma::all(a.col(c) != b.col(c)));
        }
    }
}

TEST_CASE("ssf_draw - marginal distributions")
{
    const auto sp = scen(13.0, 10.0); // N = 12, 108 columns
    const arma::uword N = 12;
    const arma::cube phase = make_phase(256, N, true, 68);
    const arma::mat rx = grid(100, 100, 50.0, 1.5), tx = {{0.0, 0.0, 25.0}};
    arma::mat rnd;
    quadriga_lib::ssf_draw(sp, phase, tx, rx, rnd);
    REQUIRE(rnd.n_cols == 9 * N);

    // 5 standard errors with n = 10^4: Normal mean 0.05, std 0.05; Uniform mean 0.015 (0.0029), fraction below 0.1
    // 0.015 (0.003); sign fraction 0.025 (0.005)
    QD_DIAG("\n# diag: ssf_draw marginal statistics, 10^4 links\ncolumn,mean,std,frac_below_0.1,frac_nonneg\n");
    for (arma::uword c = 0; c < rnd.n_cols; ++c)
    {
        const arma::vec v = rnd.col(c);
        const double n = (double)v.n_elem, m = arma::mean(v), s = stddev(v);
        const double below = (double)arma::accu(v < 0.1) / n, nonneg = (double)arma::accu(v >= 0.0) / n;
        QD_DIAG("%llu,%.4f,%.4f,%.4f,%.4f\n", (unsigned long long)c, m, s, below, nonneg);
        INFO("column " << c);
        if (c < N)
        {
            CHECK(v.min() >= 0.0);
            CHECK(v.max() <= 1.0);
            CHECK(std::abs(m - 0.5) < 0.015);
            CHECK(std::abs(below - 0.1) < 0.015);
        }
        else
        {
            CHECK(std::abs(m) < 0.05);
            CHECK(std::abs(s - 1.0) < 0.05);
            if (c < 5 * N) // sign blocks
                CHECK(std::abs(nonneg - 0.5) < 0.025);
        }
    }

    // Per-frequency set
    const arma::cube pf = make_phase(256, N, false, 69);
    quadriga_lib::ssf_draw(sp, pf, tx, rx, rnd, 0);
    REQUIRE(rnd.n_cols == N);
    for (arma::uword c = 0; c < N; ++c)
    {
        INFO("per-frequency column " << c);
        CHECK(std::abs(arma::mean(rnd.col(c))) < 0.05);
        CHECK(std::abs(stddev(rnd.col(c)) - 1.0) < 0.05);
    }
}

TEST_CASE("ssf_draw - the correlation at the decorrelation distance follows the preset ACF")
{
    // Tolerance 0.1 = 5 standard errors (0.05) + directional error of the SOS approximation (as the large-scale case)
    const double lambda = 10.0;
    const auto sp = scen(5.0, lambda); // N = 4, 36 columns
    const arma::uword N = 4;
    const arma::cube phase = make_phase(256, N, true, 70);
    const arma::mat base = grid(100, 100, 50.0, 1.5), tx = {{0.0, 0.0, 25.0}};
    const double rho = acf_at(get_preset("Comb256", lambda), lambda);
    const double rho_u = 6.0 / pi * std::asin(0.5 * rho); // Pearson correlation of the Uniform map
    const double az[4] = {0.0, 0.25 * pi, 0.5 * pi, 0.75 * pi};

    arma::mat a;
    quadriga_lib::ssf_draw(sp, phase, tx, base, a);
    QD_DIAG("\n# diag: ssf_draw sample correlation at the offset lambda (column 0 Uniform, column N Normal)\n");
    QD_DIAG("azimuth_deg,corr_c0,ref_uniform,corr_cN,ref_normal\n");
    for (double phi : az)
    {
        arma::mat moved = base, b;
        moved.col(0) += lambda * std::cos(phi), moved.col(1) += lambda * std::sin(phi);
        quadriga_lib::ssf_draw(sp, phase, tx, moved, b);
        QD_DIAG("%.0f,%.4f,%.4f,%.4f,%.4f\n", phi * 180.0 / pi, corr(a.col(0), b.col(0)), rho_u, corr(a.col(N), b.col(N)), rho);
        for (arma::uword c = 0; c < a.n_cols; ++c)
        {
            INFO("azimuth " << phi << ", column " << c);
            CHECK(std::abs(corr(a.col(c), b.col(c)) - (c < N ? rho_u : rho)) < 0.1);
        }
    }
}

TEST_CASE("ssf_draw - sizes, empty input and pre-sized output")
{
    const auto sp = scen(4.0, 12.0);
    const arma::cube phase = make_phase(256, 3, true, 71);
    const arma::mat rx0(0, 3), tx1 = {{0.0, 0.0, 25.0}};
    arma::mat rnd(4, 4, arma::fill::ones);

    CHECK_NOTHROW(quadriga_lib::ssf_draw(sp, phase, tx1, rx0, rnd));
    CHECK(rnd.n_rows == 0);
    CHECK(rnd.n_cols == 27);
    rnd.ones(4, 4);
    CHECK_NOTHROW(quadriga_lib::ssf_draw(sp, phase, rx0, rx0, rnd, -1, nullptr, true));
    CHECK(rnd.n_rows == 0);
    CHECK(rnd.n_cols == 27);

    // NumClusters = 1: empty sets; the phase cube of ssf_init_phases and any other empty cube are accepted
    const auto sp1 = scen(1.0, 12.0);
    const arma::mat rx = rx5(), tx = rx5() + 5.0;
    arma::cube ph0, ph0f, empty;
    quadriga_lib::ssf_init_phases(sp1, ph0, 3, -1);
    quadriga_lib::ssf_init_phases(sp1, ph0f, 3, 0);
    rnd.ones(4, 4);
    quadriga_lib::ssf_draw(sp1, ph0, tx, rx, rnd);
    CHECK(rnd.n_rows == 5);
    CHECK(rnd.n_cols == 0);
    rnd.ones(4, 4);
    quadriga_lib::ssf_draw(sp1, ph0f, tx, rx, rnd, 0, nullptr, true);
    CHECK(rnd.n_rows == 5);
    CHECK(rnd.n_cols == 0);
    rnd.ones(4, 4);
    quadriga_lib::ssf_draw(sp1, empty, tx, rx, rnd);
    CHECK(rnd.n_rows == 5);
    CHECK(rnd.n_cols == 0);

    // Pre-sized output with stale content: written in place (same memory) and completely
    const arma::uvec fl = {0, 3, 3, 259, 1};
    arma::mat ref1, ref2;
    quadriga_lib::ssf_draw(sp, phase, tx, rx, ref1, -1, &fl);
    quadriga_lib::ssf_draw(sp, phase, tx, rx, ref2, -1, nullptr, true);
    arma::mat pre(5, 27);
    pre.fill(99.0);
    const double *mem = pre.memptr();
    quadriga_lib::ssf_draw(sp, phase, tx, rx, pre, -1, &fl);
    CHECK(pre.memptr() == mem);
    CHECK(equal(pre, ref1));
    pre.fill(99.0);
    quadriga_lib::ssf_draw(sp, phase, tx, rx, pre, -1, nullptr, true);
    CHECK(pre.memptr() == mem);
    CHECK(equal(pre, ref2));

    // Wrong size: resized
    arma::mat wrong(3, 3, arma::fill::ones);
    quadriga_lib::ssf_draw(sp, phase, tx, rx, wrong, -1, nullptr, true);
    CHECK(equal(wrong, ref2));
}

TEST_CASE("ssf_draw - input validation")
{
    const auto sp = scen(4.0, 12.0); // N = 3
    const arma::cube shared = make_phase(256, 3, true, 72), perf = make_phase(256, 3, false, 73);
    const arma::mat rx = rx5().rows(0, 2), tx = {{0.0, 0.0, 25.0}};
    const arma::uvec fl = {0, 1, 2};
    arma::mat rnd;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::ssf_draw(sp, shared, tx, rx, rnd));
        CHECK_NOTHROW(quadriga_lib::ssf_draw(sp, perf, tx, rx, rnd, 5, &fl, true));
    }
    SECTION("positions")
    {
        arma::mat bad(3, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, shared, tx, bad, rnd), std::invalid_argument,
                             Message("Input 'rx_pos' must have 3 columns."));
        bad = tx;
        bad(0, 1) = INFINITY;
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, shared, bad, rx, rnd), std::invalid_argument,
                             Message("Input 'tx_pos' must be finite."));
        bad.zeros(2, 3);
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, shared, bad, rx, rnd), std::invalid_argument,
                             Message("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'."));
    }
    SECTION("set_id")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, shared, tx, rx, rnd, -2), std::invalid_argument,
                             Message("Input 'set_id' must be >= -1."));
    }
    SECTION("preset")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, shared, tx, rx, rnd, -1, nullptr, false, "Comb999"), std::invalid_argument,
                             Message("Input 'preset' is not a known preset."));
    }
    SECTION("scenpar")
    {
        auto bad = sp;
        bad["SC_lambda"] = -1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(bad, shared, tx, rx, rnd), std::invalid_argument,
                             Message("Value of 'SC_lambda' must be finite and >= 0."));
        bad = sp;
        bad["NumClusters"] = 0.0;
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(bad, shared, tx, rx, rnd), std::invalid_argument,
                             Message("Value of 'NumClusters' must be an integer from 1 to 1000."));
    }
    SECTION("phase size")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, shared, tx, rx, rnd, 0), std::invalid_argument,
                             Message("Input 'phase' must have size [256, 3, 2] for preset 'Comb256' and this set."));
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, perf, tx, rx, rnd, -1), std::invalid_argument,
                             Message("Input 'phase' must have size [256, 27, 2] for preset 'Comb256' and this set."));
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, shared, tx, rx, rnd, -1, nullptr, false, "Exp512"), std::invalid_argument,
                             Message("Input 'phase' must have size [512, 27, 2] for preset 'Exp512' and this set."));
        arma::cube bad(256, 27, 1, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, bad, tx, rx, rnd), std::invalid_argument,
                             Message("Input 'phase' must have size [256, 27, 2] for preset 'Comb256' and this set."));
        // NumClusters = 1: a non-empty cube is rejected
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(scen(1.0, 12.0), perf, tx, rx, rnd, 0), std::invalid_argument,
                             Message("Input 'phase' must have size [256, 0, 2] for preset 'Comb256' and this set."));
    }
    SECTION("non-finite phase")
    {
        arma::cube bad = shared;
        bad(5, 3, 1) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, bad, tx, rx, rnd), std::invalid_argument,
                             Message("Input 'phase' must be finite."));
    }
    SECTION("floor_id")
    {
        const arma::uvec bad = {0, 1};
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, shared, tx, rx, rnd, -1, &bad), std::invalid_argument,
                             Message("Number of elements in 'floor_id' does not match number of rows in 'rx_pos'."));
        arma::mat keep(2, 2, arma::fill::ones);
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_draw(sp, shared, tx, rx, keep, -1, &fl, false, "Comb256", false), std::invalid_argument,
                             Message("Input 'floor_id' cannot be used with 'horizontal = false': in 3D mode the height separates the floors."));
        CHECK(keep.n_rows == 2);
        CHECK(arma::all(arma::vectorise(keep) == 1.0));
    }
}
