// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way.

// Verification suite for the "ray" sets of quadriga_lib::gbsm_init_phases and quadriga_lib::gbsm_draw (test cases
// "gbsm_init_phases - ray: ..." and "gbsm_draw - ray: ...").
//
// gbsm_init_phases ("ray") pins down: n_gen = 14·N·M + N (N = NumClusters - 1, M = NumSubPaths), empty for N = 0;
// the seed rule (slice 0 equals sos_init_phases with the derived seed); the mirror map of slice 1; the independence of
// the sets from each other and from the "cluster" sets; the input validation.
// gbsm_draw ("ray") pins down: the values in single mobility, the decorrelation distances (SC_lambda for blocks 0 to
// 13 and for block 14 of a per-frequency set, lambda_tau for block 14 of the shared set, also with SC_lambda = 0,
// and the guards of lambda_tau), the Uniform map of block 14, the swap reciprocity, the marginal distributions, the
// use of center_frequency, sizes and the input validation.
// Oracles, independent of the functions:
//  - the seed rule, typed here: splitmix64 as in test_gbsm_phases_cluster.cpp; seed_set = splitmix64(splitmix64(seed)
//    ^ tag) >> 1 with tag 2^63 + 2^32 + set_id + 2; checked first against values of an independent Python script
//  - the column layout: block b <= 12 at b·N·M + n·M + m, block 13 at 13·N·M + n, block 14 at 13·N·M + N + n·M + m;
//    mirror blocks (1, 0, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12) for b <= 12, blocks 13 and 14 self-mirrored
//  - the sum of sinusoids qd_ref_sos and the Uniform map of qd_test_sos_oracle.hpp with the frequencies of sos_preset
//  - lambda_tau = min(max(1.6·c·cDS/cASA, 0.1 m), 10 km) of TR 38.901 Table 7.5-6, typed here: cDS the mean over the
//    frequencies of max(PerClusterDS_min, PerClusterDS + PerClusterDS_gamma·lg(max(f/1e9, PerClusterDS_fc_min))) ns
//    (3.91 ns for PerClusterDS = 0), cASA = PerClusterAS_A in rad; cDS = 0 gives 0.1 m, cASA = 0 gives 10 km
// Tolerances: as test_gbsm_phases_cluster.cpp (AVX2 kernel): Normal columns 2e-5, Uniform columns 1e-5; lambda_tau
// against the 4-decimal values 5e-5 m.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_sos.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "qd_test_diag.hpp"
#include "qd_test_sos_oracle.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    const double pi = 3.141592653589793;
    const arma::uword mirror_b[13] = {1, 0, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12};

    // Scenario with N + 1 clusters, M rays and SC_lambda; cluster DS 11 ns and cluster ASA 22 deg (TR 38.901 Table
    // 7.5-6, UMi NLOS: lambda_tau = 13.7415 m at any frequency)
    std::unordered_map<std::string, double> scen(double N, double M, double sc_lambda)
    {
        return {{"NumClusters", N + 1.0}, {"NumClusters_min", 1.0}, {"NumSubPaths", M}, {"SC_lambda", sc_lambda},
                {"PerClusterDS", 11.0}, {"PerClusterAS_A", 22.0}, {"DS_lambda", 11.0}, {"SF_lambda", 13.0}};
    }

    std::uint64_t splitmix64(std::uint64_t x)
    {
        std::uint64_t z = x + 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    arma::sword seed_set(arma::sword seed, arma::sword set_id)
    {
        const std::uint64_t tag = (1ULL << 63) + (1ULL << 32) + (std::uint64_t)(set_id + 2);
        return (arma::sword)(splitmix64(splitmix64((std::uint64_t)seed) ^ tag) >> 1);
    }

    arma::uword n_gen_of(arma::uword N, arma::uword M) { return 14 * N * M + N; }

    arma::uword mirror_col(arma::uword g, arma::uword N, arma::uword M)
    {
        const arma::uword NM = N * M;
        return (g < 13 * NM) ? mirror_b[g / NM] * NM + g % NM : g;
    }

    bool is_uniform(arma::uword g, arma::uword N, arma::uword M) { return g >= 13 * N * M + N; }

    // Raw phases in slice 0, typed mirror in slice 1
    arma::cube make_phase(arma::uword N, arma::uword M, arma::sword seed)
    {
        const arma::uword n_gen = n_gen_of(N, M);
        arma::mat P0;
        quadriga_lib::sos_init_phases<double>(256, n_gen, P0, seed);
        arma::cube phase(256, n_gen, 2);
        phase.slice(0) = P0;
        for (arma::uword g = 0; g < n_gen; ++g)
            phase.slice(1).col(g) = P0.col(mirror_col(g, N, M));
        return phase;
    }

    // lambda_tau of TR 38.901 Table 7.5-6 for a scenario and a list of frequencies (closed form, see header)
    double lambda_tau(std::unordered_map<std::string, double> sp, const arma::vec &fc)
    {
        quadriga_lib::scenpar_validate(sp); // supporting: completes the defaults
        double cds = 0.0;
        for (double f : fc)
            cds += (sp.at("PerClusterDS") == 0.0) ? 3.91e-9
                                                  : 1e-9 * std::max(sp.at("PerClusterDS_min"),
                                                                    sp.at("PerClusterDS") + sp.at("PerClusterDS_gamma") * std::log10(std::max(f / 1e9, sp.at("PerClusterDS_fc_min"))));
        cds /= (double)fc.n_elem;
        const double casa = sp.at("PerClusterAS_A") * pi / 180.0;
        if (cds <= 0.0)
            return 0.1;
        if (casa <= 0.0)
            return 1e4;
        return std::min(std::max(1.6 * 299792458.0 * cds / casa, 0.1), 1e4);
    }

    // Largest deviation of columns [g0, g1) of 'rnd' (single mobility, 2D) from the sum of sinusoids at distance d
    double max_dev(const arma::mat &rnd, const arma::cube &phase, const arma::mat &rx, double d, arma::uword g0,
                   arma::uword g1, bool uniform)
    {
        const arma::mat F = qd_preset("Comb256", d).F;
        double m = 0.0;
        for (arma::uword g = g0; g < g1; ++g)
            for (arma::uword l = 0; l < rx.n_rows; ++l)
            {
                double v = qd_ref_sos(F, phase.slice(0).colptr(g), rx(l, 0), rx(l, 1), 0.0);
                m = std::max(m, std::abs(rnd(l, g) - (uniform ? qd_unif(v) : v)));
            }
        return m;
    }

    arma::mat rx5()
    {
        return {{12.5, -40.0, 1.5}, {-120.0, 33.0, 4.5}, {250.0, 300.0, 22.0}, {-7.25, -299.0, 1.8}, {88.0, 0.5, 10.0}};
    }
}

// ------------------------------------------------------------------------------------------------------------------
// gbsm_init_phases ("ray")
// ------------------------------------------------------------------------------------------------------------------

TEST_CASE("gbsm_init_phases - ray: n_gen is 14·N·M + N, empty for N = 0")
{
    const double NM[4][2] = {{19.0, 20.0}, {4.0, 1.0}, {3.0, 7.0}, {0.0, 20.0}};
    const arma::uword n_gen[4] = {5339, 60, 297, 0};
    for (int i = 0; i < 4; ++i)
        for (arma::sword set : {-1, 3})
        {
            arma::cube phase;
            quadriga_lib::gbsm_init_phases(scen(NM[i][0], NM[i][1], 10.0), "ray", phase, 5, set);
            INFO("N " << NM[i][0] << ", M " << NM[i][1] << ", set " << set);
            CHECK(n_gen_of((arma::uword)NM[i][0], (arma::uword)NM[i][1]) == n_gen[i]);
            CHECK(phase.n_rows == 256);
            CHECK(phase.n_cols == n_gen[i]);
            CHECK(phase.n_slices == 2);
        }
    arma::cube p512;
    quadriga_lib::gbsm_init_phases(scen(3.0, 7.0, 10.0), "ray", p512, 5, -1, "Exp512");
    CHECK(p512.n_rows == 512);
    CHECK(p512.n_cols == 297);

    // A pre-sized output with stale content is written in place and completely
    const auto sp = scen(3.0, 7.0, 10.0);
    arma::cube ref, pre(256, 297, 2);
    pre.fill(99.0);
    const double *mem = pre.memptr();
    quadriga_lib::gbsm_init_phases(sp, "ray", ref, 3);
    quadriga_lib::gbsm_init_phases(sp, "ray", pre, 3);
    CHECK(pre.memptr() == mem);
    CHECK(qd_equal(pre, ref));
}

TEST_CASE("gbsm_init_phases - ray: slice 0 equals sos_init_phases with the derived seed")
{
    // The typed seed rule against values of an independent Python transcription
    CHECK(seed_set(42, -1) == (arma::sword)5932692203561315960LL);
    CHECK(seed_set(42, 0) == (arma::sword)3454368648287109626LL);
    CHECK(seed_set(42, 3) == (arma::sword)7215045512090331949LL);
    CHECK(seed_set(0, -1) == (arma::sword)423928532642501001LL);
    CHECK(seed_set(0, 3) == (arma::sword)7915430664638101148LL);

    const auto sp = scen(3.0, 4.0, 10.0);
    for (arma::sword seed : {42, 0})
        for (arma::sword set : {-1, 0, 3})
        {
            arma::cube phase;
            quadriga_lib::gbsm_init_phases(sp, "ray", phase, seed, set);
            arma::mat P0;
            quadriga_lib::sos_init_phases<double>(256, n_gen_of(3, 4), P0, seed_set(seed, set));
            INFO("seed " << seed << ", set " << set);
            CHECK(qd_equal(arma::mat(phase.slice(0)), P0));
        }
}

TEST_CASE("gbsm_init_phases - ray: slice 1 is the mirror of slice 0")
{
    const arma::uword N = 3, M = 2;
    for (arma::sword set : {-1, 2})
    {
        arma::cube phase;
        quadriga_lib::gbsm_init_phases(scen(3.0, 2.0, 10.0), "ray", phase, 17, set);
        REQUIRE(phase.n_cols == n_gen_of(N, M));
        for (arma::uword g = 0; g < phase.n_cols; ++g)
        {
            INFO("set " << set << ", column " << g);
            CHECK(arma::all(phase.slice(1).col(g) == phase.slice(0).col(mirror_col(g, N, M))));
        }
        CHECK(arma::any(phase.slice(1).col(0) != phase.slice(0).col(0))); // block 0 is a pair
    }
}

TEST_CASE("gbsm_init_phases - ray sets are distinct from each other and from the cluster sets")
{
    const auto sp = scen(2.0, 3.0, 10.0);
    arma::cube r_shared, r0, r1, c_shared, c0;
    quadriga_lib::gbsm_init_phases(sp, "ray", r_shared, 42);
    quadriga_lib::gbsm_init_phases(sp, "ray", r0, 42, 0);
    quadriga_lib::gbsm_init_phases(sp, "ray", r1, 42, 1);
    quadriga_lib::gbsm_init_phases(sp, "cluster", c_shared, 42);
    quadriga_lib::gbsm_init_phases(sp, "cluster", c0, 42, 0);

    auto any_equal_col = [](const arma::cube &a, const arma::cube &b)
    {
        for (arma::uword i = 0; i < a.n_cols; ++i)
            for (arma::uword j = 0; j < b.n_cols; ++j)
                if (arma::all(a.slice(0).col(i) == b.slice(0).col(j)))
                    return true;
        return false;
    };
    CHECK_FALSE(any_equal_col(r_shared, r0));
    CHECK_FALSE(any_equal_col(r_shared, r1));
    CHECK_FALSE(any_equal_col(r0, r1));
    for (const arma::cube *c : {&c_shared, &c0})
    {
        CHECK_FALSE(any_equal_col(r_shared, *c));
        CHECK_FALSE(any_equal_col(r0, *c));
        CHECK_FALSE(any_equal_col(r1, *c));
    }
}

TEST_CASE("gbsm_init_phases - ray: input validation")
{
    const auto sp = scen(2.0, 3.0, 10.0);
    const arma::mat pos = {{1.0, 2.0, 1.5}};
    arma::cube phase;
    CHECK_NOTHROW(quadriga_lib::gbsm_init_phases(sp, "ray", phase, 1, 4, "Exp512"));
    CHECK_NOTHROW(quadriga_lib::gbsm_init_phases({}, "ray", phase, 1)); // empty scenpar: all defaults
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_init_phases(sp, "ray", phase, 1, -2), std::invalid_argument,
                         Message("Input 'set_id' must be >= -1."));
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_init_phases(sp, "ray", phase, 1, -1, "Comb999"), std::invalid_argument,
                         Message("Input 'preset' is not a known preset."));
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_init_phases(sp, "ray", phase, 1, -1, "Comb256", &pos), std::invalid_argument,
                         Message("Inputs 'tx_pos' and 'rx_pos' are used for set type 'lsf' only."));
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_init_phases(sp, "ray", phase, 1, -1, "Comb256", nullptr, &pos), std::invalid_argument,
                         Message("Inputs 'tx_pos' and 'rx_pos' are used for set type 'lsf' only."));
    auto bad = sp;
    bad["NumSubPaths"] = 0.0;
    CHECK_THROWS_AS(quadriga_lib::gbsm_init_phases(bad, "ray", phase, 1), std::invalid_argument);

    arma::cube keep(2, 2, 2, arma::fill::ones);
    CHECK_THROWS_AS(quadriga_lib::gbsm_init_phases(sp, "ray", keep, -2), std::invalid_argument);
    CHECK(keep.n_rows == 2);
    CHECK(arma::all(arma::vectorise(keep) == 1.0));
}

// ------------------------------------------------------------------------------------------------------------------
// gbsm_draw ("ray")
// ------------------------------------------------------------------------------------------------------------------

TEST_CASE("gbsm_draw - ray: single mobility values equal the sum of sinusoids")
{
    const arma::uword N = 2, M = 3, NM = 6, n_gen = 86;
    const auto sp = scen(2.0, 3.0, 15.0);
    const arma::mat rx = rx5(), tx = {{10.0, 20.0, 25.0}};
    const arma::vec fc = {3.5e9};
    const double lt = lambda_tau(sp, fc);

    QD_DIAG("\n# diag: gbsm_draw ray single mobility (2D) against ref_sos, max abs difference\nset,blocks_0_13,block_14\n");
    for (arma::sword set : {-1, 1})
    {
        const arma::cube phase = make_phase(N, M, 51 + set);
        arma::mat rnd, rnd2;
        quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, rnd, set, "Comb256", nullptr, &fc);
        REQUIRE(rnd.n_rows == 5);
        REQUIRE(rnd.n_cols == n_gen);
        const double d_normal = max_dev(rnd, phase, rx, 15.0, 0, 13 * NM + N, false);
        const double d_uniform = max_dev(rnd, phase, rx, set < 0 ? lt : 15.0, 13 * NM + N, n_gen, true);
        QD_DIAG("%lld,%.3e,%.3e\n", (long long)set, d_normal, d_uniform);
        INFO("set " << set);
        CHECK(d_normal < 2e-5);
        CHECK(d_uniform < 1e-5);
        for (arma::uword g = 0; g < n_gen; ++g)
            if (is_uniform(g, N, M))
                CHECK((rnd.col(g).min() > 0.0 && rnd.col(g).max() < 1.0));
        quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, rnd2, set, "Comb256", nullptr, &fc);
        CHECK(qd_equal(rnd, rnd2)); // determinism
    }
}

TEST_CASE("gbsm_draw - ray: block 14 of the shared set uses lambda_tau")
{
    const arma::mat rx = rx5(), tx = {{10.0, 20.0, 25.0}};
    const arma::uword N = 2, M = 2, NM = 4, b14 = 13 * NM + N, n_gen = 58;

    // Shared block 14 at lambda_tau and blocks 0 to 13 at SC_lambda; per-frequency block 14 at SC_lambda
    auto check_set = [&](const std::unordered_map<std::string, double> &sp, const arma::vec &fc, double lt, double sc)
    {
        for (arma::sword set : {-1, 0})
        {
            const arma::cube phase = make_phase(N, M, 81 + set);
            arma::mat rnd;
            quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, rnd, set, "Comb256", nullptr, &fc);
            REQUIRE(rnd.n_cols == n_gen);
            INFO("set " << set << ", lambda_tau " << lt << ", SC_lambda " << sc);
            CHECK(max_dev(rnd, phase, rx, sc, 0, b14, false) < 2e-5);
            CHECK(max_dev(rnd, phase, rx, set < 0 ? lt : sc, b14, n_gen, true) < 1e-5);
        }
    };

    // Values of TR 38.901 v19 conf files; UMa NLOS with and without PerClusterDS_fc_min = 6 (Table 7.5-6 note 6)
    const char *files[3] = {"UMi_NLOS", "UMa_NLOS", "UMa_NLOS"};
    const arma::vec freqs[3] = {{3.5e9}, {3.5e9, 28e9}, {3.5e9, 28e9}};
    const double expected[3] = {13.7415, 5.0749, 5.8058};
    QD_DIAG("\n# diag: gbsm_draw ray lambda_tau from the conf files\nfile,fc_min,lambda_tau\n");
    for (int i = 0; i < 3; ++i)
    {
        std::unordered_map<std::string, double> sp;
        quadriga_lib::conf_file_read(std::string("data/config/3GPP_38.901_v19_") + files[i] + ".conf", sp);
        sp["NumClusters"] = 3.0, sp["NumClusters_min"] = 1.0, sp["NumSubPaths"] = 2.0;
        if (i == 2)
            sp["PerClusterDS_fc_min"] = 0.0;
        const double lt = lambda_tau(sp, freqs[i]);
        QD_DIAG("%s,%g,%.6f\n", files[i], sp.at("PerClusterDS_fc_min"), lt);
        INFO("file " << files[i] << ", PerClusterDS_fc_min " << sp.at("PerClusterDS_fc_min"));
        CHECK(std::abs(lt - expected[i]) < 5e-5);
        check_set(sp, freqs[i], lt, sp.at("SC_lambda"));
    }

    // SC_lambda = 0: blocks 0 to 13 and per-frequency block 14 at 0.1 m, shared block 14 at lambda_tau
    const arma::vec f35 = {3.5e9};
    auto sp = scen(2.0, 2.0, 0.0);
    check_set(sp, f35, lambda_tau(sp, f35), 0.1);

    // Guards: cASA = 0 gives 10 km; cDS = 0 (1 ns - 1 ns·lg(10 GHz)) gives 0.1 m
    sp = scen(2.0, 2.0, 12.0);
    sp["PerClusterAS_A"] = 0.0;
    CHECK(lambda_tau(sp, f35) == 1e4);
    check_set(sp, f35, 1e4, 12.0);
    sp = scen(2.0, 2.0, 12.0);
    sp["PerClusterDS"] = 1.0, sp["PerClusterDS_gamma"] = -1.0, sp["PerClusterDS_min"] = 0.0;
    const arma::vec f10 = {10e9};
    CHECK(lambda_tau(sp, f10) == 0.1);
    check_set(sp, f10, 0.1, 12.0);

    // Clamps of 1.6·c·cDS/cASA at cDS = 11 ns: cASA = 1e4 deg gives 0.030 m (-> 0.1 m), 1e-4 deg 3.0e6 m (-> 10 km)
    sp = scen(2.0, 2.0, 12.0);
    sp["PerClusterAS_A"] = 1e4;
    CHECK(lambda_tau(sp, f35) == 0.1);
    check_set(sp, f35, 0.1, 12.0);
    sp["PerClusterAS_A"] = 1e-4;
    CHECK(lambda_tau(sp, f35) == 1e4);
    check_set(sp, f35, 1e4, 12.0);
}

TEST_CASE("gbsm_draw - ray: swapping Tx and Rx exchanges the mirror blocks")
{
    const arma::uword N = 2, M = 3;
    const auto sp = scen(2.0, 3.0, 12.0);
    const arma::vec fc = {3.5e9};
    const arma::mat A = rx5();
    arma::mat B = rx5() * 0.7 + 20.0;
    B.col(2).fill(25.0);
    const arma::uvec fl = {0, 2, 2, 300, 0};

    for (arma::sword set : {-1, 2})
        for (int f = 0; f < 2; ++f)
        {
            arma::cube phase;
            quadriga_lib::gbsm_init_phases(sp, "ray", phase, 61, set);
            arma::mat ab, ba;
            quadriga_lib::gbsm_draw(sp, "ray", phase, B, A, ab, set, "Comb256", f ? &fl : nullptr, &fc, true);
            quadriga_lib::gbsm_draw(sp, "ray", phase, A, B, ba, set, "Comb256", f ? &fl : nullptr, &fc, true);
            REQUIRE(ab.n_cols == n_gen_of(N, M));
            REQUIRE(ba.n_cols == n_gen_of(N, M));
            for (arma::uword g = 0; g < ab.n_cols; ++g)
            {
                INFO("set " << set << ", floors " << f << ", column " << g);
                CHECK(arma::all(ab.col(g) == ba.col(mirror_col(g, N, M))));
            }
        }
}

TEST_CASE("gbsm_draw - ray: marginal distributions")
{
    // Cluster DS and ASA of UMi NLOS at 3.5 GHz: lambda_tau = 13.7415 m; SC_lambda 15 m (both <= 40 m on the 50 m grid)
    const arma::uword N = 2, M = 3, b14 = 13 * N * M + N;
    const auto sp = scen(2.0, 3.0, 15.0);
    const arma::vec fc = {3.5e9};
    REQUIRE(std::abs(lambda_tau(sp, fc) - 13.7415) < 5e-5);
    const arma::mat rx = qd_grid(100, 100, 50.0, 1.5, 3.7, -2.1), tx = {{0.0, 0.0, 25.0}};

    // 5 standard errors with n = 10^4: Normal mean 0.05, std 0.05; Uniform mean 0.015, fraction below 0.1 0.015
    QD_DIAG("\n# diag: gbsm_draw ray marginal statistics, 10^4 links\nset,column,mean,std,frac_below_0.1\n");
    for (arma::sword set : {-1, 0})
    {
        arma::cube phase;
        quadriga_lib::gbsm_init_phases(sp, "ray", phase, 68, set);
        arma::mat rnd;
        quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, rnd, set, "Comb256", nullptr, &fc);
        REQUIRE(rnd.n_cols == n_gen_of(N, M));
        for (arma::uword g = 0; g < rnd.n_cols; ++g)
        {
            const arma::vec v = rnd.col(g);
            const double m = arma::mean(v), s = arma::stddev(v), below = (double)arma::accu(v < 0.1) / (double)v.n_elem;
            QD_DIAG("%lld,%llu,%.4f,%.4f,%.4f\n", (long long)set, (unsigned long long)g, m, s, below);
            INFO("set " << set << ", column " << g);
            if (g >= b14)
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
            }
        }
    }
}

TEST_CASE("gbsm_draw - ray: center_frequency is required by the shared set and ignored otherwise")
{
    const arma::uword N = 2, M = 3, b14 = 13 * N * M + N;
    auto sp = scen(2.0, 3.0, 12.0);
    sp["PerClusterDS_gamma"] = -2.0; // cluster DS depends on the frequency
    const arma::mat rx = rx5(), tx = {{10.0, 20.0, 25.0}};
    const arma::cube phase = make_phase(N, M, 91);

    arma::mat a, b;
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, a), std::invalid_argument,
                         Message("Input 'center_frequency' is required for the shared 'ray' set."));

    // Shared set: another frequency list changes block 14 only
    const arma::vec f1 = {3.5e9}, f2 = {3.5e9, 28e9};
    quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, a, -1, "Comb256", nullptr, &f1);
    quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, b, -1, "Comb256", nullptr, &f2);
    CHECK(lambda_tau(sp, f1) != lambda_tau(sp, f2));
    CHECK(qd_equal(arma::mat(a.cols(0, b14 - 1)), arma::mat(b.cols(0, b14 - 1))));
    for (arma::uword g = b14; g < a.n_cols; ++g)
        CHECK(arma::all(a.col(g) != b.col(g)));

    // Per-frequency set: not read, not checked
    arma::mat ref;
    quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, ref, 4);
    for (const arma::vec &fc : {arma::vec{}, arma::vec{-1.0, NAN}, arma::vec{3.5e9}, arma::vec{28e9}})
    {
        arma::mat c;
        CHECK_NOTHROW(quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, c, 4, "Comb256", nullptr, &fc));
        CHECK(qd_equal(c, ref));
    }
}

TEST_CASE("gbsm_draw - ray: n_link = 0 and N = 0")
{
    const auto sp = scen(2.0, 3.0, 12.0);
    const arma::cube phase = make_phase(2, 3, 93);
    const arma::mat rx0(0, 3), tx1 = {{0.0, 0.0, 25.0}}, rx = rx5();
    const arma::vec fc = {3.5e9};
    arma::mat rnd(4, 4, arma::fill::ones);

    quadriga_lib::gbsm_draw(sp, "ray", phase, tx1, rx0, rnd, -1, "Comb256", nullptr, &fc);
    CHECK(rnd.n_rows == 0);
    CHECK(rnd.n_cols == 86);
    rnd.ones(4, 4);
    quadriga_lib::gbsm_draw(sp, "ray", phase, rx0, rx0, rnd, 1, "Comb256", nullptr, nullptr, true);
    CHECK(rnd.n_rows == 0);
    CHECK(rnd.n_cols == 86);

    // N = 0: empty sets; the shared set still requires center_frequency
    const auto sp0 = scen(0.0, 3.0, 12.0);
    arma::cube ph0, empty;
    quadriga_lib::gbsm_init_phases(sp0, "ray", ph0, 3);
    for (const arma::cube *p : {&ph0, &empty})
    {
        rnd.ones(4, 4);
        quadriga_lib::gbsm_draw(sp0, "ray", *p, tx1, rx, rnd, -1, "Comb256", nullptr, &fc);
        CHECK(rnd.n_rows == 5);
        CHECK(rnd.n_cols == 0);
        rnd.ones(4, 4);
        quadriga_lib::gbsm_draw(sp0, "ray", *p, tx1, rx, rnd, 0);
        CHECK(rnd.n_rows == 5);
        CHECK(rnd.n_cols == 0);
    }
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp0, "ray", ph0, tx1, rx, rnd), std::invalid_argument,
                         Message("Input 'center_frequency' is required for the shared 'ray' set."));
}

TEST_CASE("gbsm_draw - ray: input validation")
{
    const auto sp = scen(2.0, 3.0, 12.0);
    const arma::cube phase = make_phase(2, 3, 95);
    const arma::mat rx = rx5().rows(0, 2), tx = {{0.0, 0.0, 25.0}};
    const arma::vec fc = {3.5e9};
    const arma::uvec fl = {0, 1, 2}, fl_bad = {0, 1};
    arma::mat rnd;

    CHECK_NOTHROW(quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, rnd, -1, "Comb256", &fl, &fc, true));
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, rnd, -2, "Comb256", nullptr, &fc),
                         std::invalid_argument, Message("Input 'set_id' must be >= -1."));

    // Checks 6 to 9 come before the frequency checks 10 and 11
    const arma::cube small = phase.cols(0, 84);
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "ray", small, tx, rx, rnd), std::invalid_argument,
                         Message("Input 'phase' must have size [256, 86, 2] for preset 'Comb256' and this set."));
    arma::cube bad = phase;
    bad(3, 80, 1) = NAN;
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "ray", bad, tx, rx, rnd), std::invalid_argument,
                         Message("Input 'phase' must be finite."));
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, rnd, -1, "Comb256", &fl_bad), std::invalid_argument,
                         Message("Number of elements in 'floor_id' does not match number of rows in 'rx_pos'."));
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, rnd, -1, "Comb256", &fl, nullptr, false, false),
                         std::invalid_argument,
                         Message("Input 'floor_id' cannot be used with 'horizontal = false': in 3D mode the height separates the floors."));

    // Checks 10 and 11; the output is untouched on error
    arma::mat keep(2, 2, arma::fill::ones);
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, keep), std::invalid_argument,
                         Message("Input 'center_frequency' is required for the shared 'ray' set."));
    const arma::vec f_empty;
    CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, keep, -1, "Comb256", nullptr, &f_empty),
                         std::invalid_argument, Message("Input 'center_frequency' cannot be empty."));
    for (const arma::vec &f : {arma::vec{3.5e9, 0.0}, arma::vec{-1.0}, arma::vec{NAN}, arma::vec{INFINITY}})
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "ray", phase, tx, rx, keep, -1, "Comb256", nullptr, &f),
                             std::invalid_argument, Message("Input 'center_frequency' must be finite and > 0."));
    CHECK(keep.n_rows == 2);
    CHECK(arma::all(arma::vectorise(keep) == 1.0));
}
