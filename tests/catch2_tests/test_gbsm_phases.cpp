// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "lsf_*"

// Verification suite for the phase sets of the large-scale model: quadriga_lib::lsf_init_phases and
// quadriga_lib::lsf_draw (test cases "lsf_init_phases - ..." and "lsf_draw - ...").
//
// lsf_init_phases pins down: the size of the phase cube per preset, the raw phases (slice 0 equals sos_init_phases
// with the same seed), the mirror relation of slice 1 (ASD <-> ASA, ESD <-> ESA, all others themselves), the seed,
// the bias removal (zero sample mean of every Normal field over the drop positions in single and dual mobility, 2D
// and 3D, Uniform generators 9 and 10 untouched, no correction below 3 positions), the replacement of a
// decorrelation distance of 0 by 0.1 m, and the input validation.
// lsf_draw pins down: the values of the 11 generators in single mobility (each generator with its own decorrelation
// distance, 2D and 3D mode, three presets), the Uniform map of columns 9 and 10, the two-device formulas of dual
// mobility (self-mirrored generators on one map, the pairs ASD/ASA and ESD/ESA on two maps without further mixing),
// also with floor IDs, unit variance and the pair correlation in dual mobility, the swap reciprocity, the floor phase
// offsets, the statistics (marginals, ACF, zero correlation of different floors at equal positions), the
// independence of z in 2D mode, of tx_pos in single mobility and of the other links, sizes, empty input, pre-sized
// outputs and the input validation.
// Oracles, independent of the functions:
//  - the sum of sinusoids qd_ref_sos, the Uniform map and the floor offsets of qd_test_sos_oracle.hpp, with the
//    frequencies of sos_preset at the decorrelation distance of each generator
//  - the mirror table and the raw phases of sos_init_phases (documented contract)
//  - dual mobility: (qd_ref_sos(F_g, P0_g, R) + qd_ref_sos(F_g, P0_g, T)) / sqrt(2 + 2·A_g(D)) for the self-mirrored
//    generators, (qd_ref_sos(F_g, P0_g, R) + qd_ref_sos(F_m, P0_m, T)) / sqrt(2) for the pairs, A_g the ACF table of
//    sos_preset interpolated linearly by sos_acfi, D the Tx-Rx distance (horizontal in 2D mode)
//  - statistics: Normal(0,1), Uniform(0,1), the ACF table of sos_preset
// Tolerances of lsf_draw values: lsf_draw evaluates the sinusoids with the AVX2 kernel where available (sine and
// cosine in single precision, accumulation in double, deviation below 4e-7·sqrt(2·n_coeff) = 9e-6 for 256
// sinusoids). Normal columns 2e-5 for 256 sinusoids, scaled with sqrt(n_coeff/256) for other presets, Uniform
// columns 1e-5 (the slope of the Uniform map is at most 0.4). The bias removal is closed-form: 1e-12.
// Conventions: positions are rows {x, y, z} in meters; most cases use distinct decorrelation distances (DS 11,
// KF 12, SF 13, ASD 14, ASA 15, ESD 16, ESA 17, XPR 18, absTOA 19, SC 20 m; generator 10 uses SF_lambda) so that a
// mix-up of generators changes the result; the phase cubes of the lsf_draw cases are raw phases of sos_init_phases
// with the Tx side mirrored (ASD <-> ASA, ESD <-> ESA).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_sos.hpp"

#include <cmath>
#include <random>
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

    // Decorrelation distance and mirror partner of each generator for the scenario of scen_distinct()
    const double lambda_g[11] = {11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 13.0};
    const arma::uword mirror_g[11] = {0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10};

    std::unordered_map<std::string, double> scen_distinct()
    {
        return {{"DS_lambda", 11.0}, {"KF_lambda", 12.0}, {"SF_lambda", 13.0}, {"AS_D_lambda", 14.0}, {"AS_A_lambda", 15.0},
                {"ES_D_lambda", 16.0}, {"ES_A_lambda", 17.0}, {"XPR_lambda", 18.0}, {"absTOA_lambda", 19.0}, {"SC_lambda", 20.0}};
    }

    // All 10 decorrelation distances equal
    std::unordered_map<std::string, double> scen_equal(double d)
    {
        return {{"DS_lambda", d}, {"KF_lambda", d}, {"SF_lambda", d}, {"AS_D_lambda", d}, {"AS_A_lambda", d},
                {"ES_D_lambda", d}, {"ES_A_lambda", d}, {"XPR_lambda", d}, {"absTOA_lambda", d}, {"SC_lambda", d}};
    }

    // Uniform random matrix in [lo, hi), own generator so that the fixtures do not depend on the library
    arma::mat urand(arma::uword n_rows, arma::uword n_cols, double lo, double hi, unsigned seed)
    {
        std::mt19937_64 rng(seed);
        arma::mat m(n_rows, n_cols);
        for (arma::uword i = 0; i < m.n_elem; ++i)
            m(i) = lo + (hi - lo) * ((double)(rng() >> 11) / 9007199254740992.0);
        return m;
    }

    // Drop positions: x, y uniform in [0, edge), z uniform in [1.5, 30)
    arma::mat drop(arma::uword n, double edge, unsigned seed)
    {
        arma::mat p = urand(n, 3, 0.0, edge, seed);
        p.col(2) = urand(n, 1, 1.5, 30.0, seed + 1000);
        return p;
    }

    // Raw phases of the documented contract: sos_init_phases(n_coeff, 11, seed)
    arma::mat raw_phases(arma::uword n_coeff, arma::sword seed)
    {
        arma::mat P;
        quadriga_lib::sos_init_phases<double>(n_coeff, 11, P, seed);
        return P;
    }

    // Phase cube: raw phases in slice 0, mirror in slice 1
    arma::cube make_phase(arma::uword n_coeff, arma::sword seed)
    {
        const arma::mat P0 = raw_phases(n_coeff, seed);
        arma::cube phase(n_coeff, 11, 2);
        phase.slice(0) = P0;
        for (arma::uword g = 0; g < 11; ++g)
            phase.slice(1).col(g) = P0.col(mirror_g[g]);
        return phase;
    }

    // Check that slice 1 is the mirror of slice 0
    bool is_mirror(const arma::cube &phase)
    {
        for (arma::uword g = 0; g < 11; ++g)
            if (arma::any(phase.slice(1).col(g) != phase.slice(0).col(mirror_g[g])))
                return false;
        return true;
    }

    // Mean of qd_ref_sos over the rows of U; use_z = false evaluates at z = 0
    double mean_ref(const arma::mat &F, const double *phi, const arma::mat &U, bool use_z)
    {
        double s = 0.0;
        for (arma::uword i = 0; i < U.n_rows; ++i)
            s += qd_ref_sos(F, phi, U(i, 0), U(i, 1), use_z ? U(i, 2) : 0.0);
        return s / (double)U.n_rows;
    }

    // 7 links with distinct coordinates, negative values, up to 300 m, z != 0
    arma::mat rx7()
    {
        return {{12.5, -40.0, 1.5}, {-120.0, 33.0, 4.5}, {250.0, 300.0, 22.0}, {-7.25, -299.0, 1.8},
                {88.0, 0.5, 10.0}, {0.0, 160.0, 31.5}, {-210.0, -95.5, 7.0}};
    }
}

TEST_CASE("lsf_init_phases - output size follows the preset")
{
    const auto sp = scen_distinct();
    const char *presets[3] = {"Comb256", "Exp128", "Gauss1024"};
    const arma::uword n_coeff[3] = {256, 128, 1024};
    for (int i = 0; i < 3; ++i)
    {
        INFO("preset " << presets[i]);
        arma::cube phase;
        quadriga_lib::lsf_init_phases(sp, phase, 3, nullptr, nullptr, false, true, presets[i]);
        CHECK(phase.n_rows == n_coeff[i]);
        CHECK(phase.n_cols == 11);
        CHECK(phase.n_slices == 2);
    }

    // A pre-sized output with stale content is written in place and completely
    arma::cube ref, pre(256, 11, 2);
    pre.fill(99.0);
    const double *mem = pre.memptr();
    quadriga_lib::lsf_init_phases(sp, ref, 3);
    quadriga_lib::lsf_init_phases(sp, pre, 3);
    CHECK(pre.memptr() == mem);
    CHECK(arma::all(arma::vectorise(pre == ref)));
}

TEST_CASE("lsf_init_phases - slice 0 equals sos_init_phases with the same seed")
{
    const auto sp = scen_distinct();
    arma::cube phase;
    quadriga_lib::lsf_init_phases(sp, phase, 42);
    REQUIRE(phase.n_rows == 256);
    REQUIRE(phase.n_cols == 11);
    REQUIRE(phase.n_slices == 2);
    CHECK(arma::all(arma::vectorise(phase.slice(0) == raw_phases(256, 42))));

    quadriga_lib::lsf_init_phases(sp, phase, 42, nullptr, nullptr, false, true, "Exp128");
    REQUIRE(phase.n_rows == 128);
    CHECK(arma::all(arma::vectorise(phase.slice(0) == raw_phases(128, 42))));
}

TEST_CASE("lsf_init_phases - slice 1 is the mirror of slice 0")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(50, 100.0, 11);
    arma::mat tx = rx;
    tx.col(0) += 30.0, tx.col(1) -= 20.0;
    const arma::mat tx1 = {{5.0, -7.0, 25.0}};

    arma::cube phase;
    quadriga_lib::lsf_init_phases(sp, phase, 5);
    CHECK(is_mirror(phase)); // no bias removal (no positions)

    quadriga_lib::lsf_init_phases(sp, phase, 5, &tx1, &rx); // single mobility, bias removal
    CHECK(is_mirror(phase));
    CHECK(arma::any(arma::vectorise(phase.slice(0) != raw_phases(256, 5)))); // the correction did act

    quadriga_lib::lsf_init_phases(sp, phase, 5, &tx, &rx, true); // dual mobility, bias removal
    CHECK(is_mirror(phase));

    quadriga_lib::lsf_init_phases(sp, phase, 5, &tx, &rx, true, false); // dual mobility, no bias removal
    CHECK(is_mirror(phase));
}

TEST_CASE("lsf_init_phases - the same seed gives the same phases")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(30, 100.0, 21);
    arma::cube a, b, c;
    quadriga_lib::lsf_init_phases(sp, a, 7, nullptr, &rx);
    quadriga_lib::lsf_init_phases(sp, b, 7, nullptr, &rx);
    quadriga_lib::lsf_init_phases(sp, c, 8, nullptr, &rx);
    CHECK(arma::all(arma::vectorise(a == b))); // determinism: repeated call, identical result
    CHECK(arma::any(arma::vectorise(a != c)));
}

TEST_CASE("lsf_init_phases - bias removal zeroes the drop mean in single mobility")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(200, 100.0, 31);
    const arma::mat tx = {{50.0, 50.0, 25.0}};

    arma::cube phase, phase_raw;
    quadriga_lib::lsf_init_phases(sp, phase, 9, &tx, &rx);
    quadriga_lib::lsf_init_phases(sp, phase_raw, 9, &tx, &rx, false, false);
    REQUIRE(phase.n_rows == 256);

    // Closed-form correction: the mean is zero up to the rounding of 200 O(1) values, 1e-12
    QD_DIAG("\n# diag: lsf_init_phases single mobility, mean of ref_sos over 200 positions (2D)\n");
    QD_DIAG("g,lambda,mean_raw,mean_corrected\n");
    for (arma::uword g = 0; g < 9; ++g)
    {
        const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
        double m = mean_ref(F, phase.slice(0).colptr(g), rx, false);
        double m0 = mean_ref(F, phase_raw.slice(0).colptr(g), rx, false);
        QD_DIAG("%llu,%.1f,%.6e,%.3e\n", (unsigned long long)g, lambda_g[g], m0, m);
        INFO("g " << g);
        CHECK(std::abs(m) < 1e-12);
    }
    // Uniform generators 9 and 10 are not corrected
    for (arma::uword g = 9; g < 11; ++g)
    {
        INFO("g " << g);
        CHECK(arma::all(phase.slice(0).col(g) == phase_raw.slice(0).col(g)));
    }
}

TEST_CASE("lsf_init_phases - bias removal in 3D mode uses the z coordinate")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(200, 100.0, 41);
    const arma::mat tx = {{50.0, 50.0, 25.0}};

    arma::cube phase;
    quadriga_lib::lsf_init_phases(sp, phase, 10, &tx, &rx, false, true, "Comb256", false);

    double max_2d = 0.0;
    QD_DIAG("\n# diag: lsf_init_phases 3D mode, mean of ref_sos with and without z\n");
    QD_DIAG("g,mean_3d,mean_2d\n");
    for (arma::uword g = 0; g < 9; ++g)
    {
        const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
        double m3 = mean_ref(F, phase.slice(0).colptr(g), rx, true);
        double m2 = mean_ref(F, phase.slice(0).colptr(g), rx, false);
        QD_DIAG("%llu,%.3e,%.6e\n", (unsigned long long)g, m3, m2);
        INFO("g " << g);
        CHECK(std::abs(m3) < 1e-12);
        max_2d = std::max(max_2d, std::abs(m2));
    }
    CHECK(max_2d > 1e-6);
}

TEST_CASE("lsf_init_phases - bias removal with dual mobility uses the Rx and Tx positions together")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(100, 100.0, 51);
    arma::mat tx = rx;
    tx.col(0) += 30.0, tx.col(1) -= 20.0, tx.col(2) += 5.0; // equal separation for every link

    arma::cube phase, phase_sw;
    quadriga_lib::lsf_init_phases(sp, phase, 12, &tx, &rx, true);
    quadriga_lib::lsf_init_phases(sp, phase_sw, 12, &rx, &tx, true); // swapped roles
    REQUIRE(phase.n_rows == 256);

    // Mean over the union [R; T] of the field of every Normal generator (2D: z = 0), closed form: 1e-12
    const arma::mat U = arma::join_cols(rx, tx);
    QD_DIAG("\n# diag: lsf_init_phases dual mobility, mean of ref_sos over the 200 Rx and Tx positions\n");
    QD_DIAG("g,mean_union,mean_rx_only\n");
    for (arma::uword g = 0; g < 9; ++g)
    {
        const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
        double m = mean_ref(F, phase.slice(0).colptr(g), U, false);
        QD_DIAG("%llu,%.3e,%.6e\n", (unsigned long long)g, m, mean_ref(F, phase.slice(0).colptr(g), rx, false));
        INFO("g " << g);
        CHECK(std::abs(m) < 1e-12);
    }

    // Swapped Tx and Rx: the same union, summed in another order; phases equal modulo 2·pi up to rounding
    double max_dphi = 0.0;
    for (arma::uword i = 0; i < phase.n_elem; ++i)
        max_dphi = std::max(max_dphi, std::abs(std::remainder(phase(i) - phase_sw(i), 2.0 * pi)));
    QD_DIAG("# diag: max phase difference after swapping Tx and Rx: %.3e\n", max_dphi);
    CHECK(max_dphi < 1e-12);

    // Supporting check through lsf_draw: with equal Tx-Rx distance for all links the normalisation
    // 1/sqrt(2 + 2·A(D)) is common, so the link mean of the self-mirrored Normal columns is 0.
    // Tolerance 2e-5: lsf_draw evaluates with the AVX2 kernel (single-precision sin/cos where available).
    arma::mat rnd;
    quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, nullptr, true);
    REQUIRE(rnd.n_rows == 100);
    REQUIRE(rnd.n_cols == 11);
    const arma::uword cols[5] = {0, 1, 2, 7, 8};
    for (arma::uword c : cols)
    {
        INFO("column " << c);
        CHECK(std::abs(arma::mean(rnd.col(c))) < 2e-5);
    }
}

TEST_CASE("lsf_init_phases - no bias removal for fewer than 3 positions, without positions or when switched off")
{
    const auto sp = scen_distinct();
    const arma::mat raw = raw_phases(256, 13);
    auto equals_raw = [&](const arma::cube &phase)
    {
        if (phase.n_rows != 256 || phase.n_cols != 11 || phase.n_slices != 2)
            return false;
        return arma::all(arma::vectorise(phase.slice(0) == raw)) && is_mirror(phase);
    };

    const arma::mat rx2 = drop(2, 100.0, 61), rx200 = drop(200, 100.0, 62), rx0(0, 3);
    const arma::mat tx = {{0.0, 0.0, 25.0}};
    arma::cube phase;

    quadriga_lib::lsf_init_phases(sp, phase, 13, &tx, &rx2);
    CHECK(equals_raw(phase));
    quadriga_lib::lsf_init_phases(sp, phase, 13);
    CHECK(equals_raw(phase));
    quadriga_lib::lsf_init_phases(sp, phase, 13, &tx, &rx200, false, false);
    CHECK(equals_raw(phase));
    CHECK_NOTHROW(quadriga_lib::lsf_init_phases(sp, phase, 13, &tx, &rx0));
    CHECK(equals_raw(phase));
}

TEST_CASE("lsf_init_phases - a decorrelation distance of 0 is used as 0.1 m")
{
    auto sp = scen_distinct();
    sp["DS_lambda"] = 0.0;
    // A 10 m square (100 decorrelation distances): the arguments of the sinusoids stay small enough for the
    // 1e-12 tolerance of the closed-form correction
    const arma::mat rx = drop(200, 10.0, 71);
    arma::cube phase;
    quadriga_lib::lsf_init_phases(sp, phase, 14, nullptr, &rx);

    const arma::mat F = qd_preset("Comb256", 0.1).F;
    double m = mean_ref(F, phase.slice(0).colptr(0), rx, false);
    QD_DIAG("\n# diag: lsf_init_phases DS_lambda = 0, mean with the 0.1 m table: %.3e\n", m);
    CHECK(std::abs(m) < 1e-12);
}

TEST_CASE("lsf_init_phases - input validation")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(5, 100.0, 81);
    const arma::mat tx = {{0.0, 0.0, 25.0}};
    arma::cube phase;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::lsf_init_phases(sp, phase, 1, &tx, &rx));
        CHECK_NOTHROW(quadriga_lib::lsf_init_phases(sp, phase));
        CHECK_NOTHROW(quadriga_lib::lsf_init_phases({}, phase, 1)); // empty scenpar: all defaults
        CHECK_NOTHROW(quadriga_lib::lsf_init_phases(sp, phase, 1, nullptr, &rx, true, false)); // dual mobility without tx_pos, no bias removal
    }
    SECTION("seed")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(sp, phase, -2, &tx, &rx), std::invalid_argument,
                             Message("Input 'seed' must be >= -1."));
    }
    SECTION("preset")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(sp, phase, 1, &tx, &rx, false, true, "Comb999"), std::invalid_argument,
                             Message("Input 'preset' is not a known preset."));
    }
    SECTION("scenpar value")
    {
        auto bad = sp;
        bad["DS_lambda"] = -1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(bad, phase, 1), std::invalid_argument,
                             Message("Value of 'DS_lambda' must be finite and >= 0."));
        bad = sp;
        bad["SC_lambda"] = INFINITY;
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(bad, phase, 1), std::invalid_argument,
                             Message("Value of 'SC_lambda' must be finite and >= 0."));
    }
    SECTION("scenpar unknown key")
    {
        auto bad = sp;
        bad["DS_lambdaX"] = 1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(bad, phase, 1), std::invalid_argument,
                             Message("Unknown key 'DS_lambdaX' in 'scenpar'."));
    }
    SECTION("rx_pos")
    {
        arma::mat bad(5, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(sp, phase, 1, &tx, &bad), std::invalid_argument,
                             Message("Input 'rx_pos' must have 3 columns."));
        bad = rx;
        bad(2, 1) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(sp, phase, 1, &tx, &bad), std::invalid_argument,
                             Message("Input 'rx_pos' must be finite."));
    }
    SECTION("tx_pos")
    {
        arma::mat bad(1, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(sp, phase, 1, &bad, &rx), std::invalid_argument,
                             Message("Input 'tx_pos' must have 3 columns."));
        bad.zeros(2, 3);
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(sp, phase, 1, &bad, &rx), std::invalid_argument,
                             Message("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'."));
    }
    SECTION("dual mobility without tx_pos")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_init_phases(sp, phase, 1, nullptr, &rx, true), std::invalid_argument,
                             Message("Input 'tx_pos' is required for the bias removal with dual mobility."));
    }
}

TEST_CASE("lsf_draw - single mobility values equal the sum of sinusoids of each generator")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 21);
    const arma::mat rx = rx7(), tx = {{10.0, 20.0, 25.0}};

    arma::mat rnd, rnd2;
    quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd);
    REQUIRE(rnd.n_rows == 7);
    REQUIRE(rnd.n_cols == 11);

    QD_DIAG("\n# diag: lsf_draw single mobility (2D) against ref_sos, max abs difference per generator\n");
    QD_DIAG("g,lambda,max_diff\n");
    for (arma::uword g = 0; g < 11; ++g)
    {
        const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
        double max_d = 0.0;
        for (arma::uword l = 0; l < 7; ++l)
        {
            double v = qd_ref_sos(F, phase.slice(0).colptr(g), rx(l, 0), rx(l, 1), 0.0);
            if (g >= 9)
                v = qd_unif(v);
            max_d = std::max(max_d, std::abs(rnd(l, g) - v));
        }
        QD_DIAG("%llu,%.1f,%.3e\n", (unsigned long long)g, lambda_g[g], max_d);
        INFO("g " << g);
        CHECK(max_d < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
    }

    // Determinism: a repeated call gives identical results
    quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd2);
    CHECK(qd_equal(rnd, rnd2));
}

TEST_CASE("lsf_draw - 3D mode uses the z coordinate")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 22);
    const arma::mat rx = rx7(), tx = {{10.0, 20.0, 25.0}};

    arma::mat rnd;
    quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, nullptr, false, "Comb256", false);
    REQUIRE(rnd.n_rows == 7);
    REQUIRE(rnd.n_cols == 11);
    for (arma::uword g = 0; g < 11; ++g)
    {
        const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
        for (arma::uword l = 0; l < 7; ++l)
        {
            double v = qd_ref_sos(F, phase.slice(0).colptr(g), rx(l, 0), rx(l, 1), rx(l, 2));
            if (g >= 9)
                v = qd_unif(v);
            INFO("g " << g << ", link " << l);
            CHECK(std::abs(rnd(l, g) - v) < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
        }
    }
}

TEST_CASE("lsf_draw - other presets")
{
    const auto sp = scen_distinct();
    const arma::mat rx = rx7(), tx = {{10.0, 20.0, 25.0}};
    const char *presets[2] = {"Exp128", "Gauss1024"};
    const arma::uword n_coeff[2] = {128, 1024};
    const double tol[2] = {1.5e-5, 4e-5}; // 2e-5 · sqrt(n_coeff / 256), AVX2 kernel tolerance
    for (int i = 0; i < 2; ++i)
    {
        const arma::cube phase = make_phase(n_coeff[i], 23);
        arma::mat rnd;
        quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, nullptr, false, presets[i]);
        REQUIRE(rnd.n_rows == 7);
        const arma::mat F = qd_preset(presets[i], lambda_g[0]).F;
        for (arma::uword l = 0; l < 7; ++l)
        {
            INFO("preset " << presets[i] << ", link " << l);
            CHECK(std::abs(rnd(l, 0) - qd_ref_sos(F, phase.slice(0).colptr(0), rx(l, 0), rx(l, 1), 0.0)) < tol[i]);
        }
    }
}

TEST_CASE("lsf_draw - 2D mode values do not depend on z")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 24);
    arma::mat rx = rx7(), tx = rx7() + 15.0;

    arma::mat a, b, c, d;
    quadriga_lib::lsf_draw(sp, phase, tx, rx, a);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, c, nullptr, true);
    rx.col(2) += 17.0, tx.col(2) -= 3.0;
    quadriga_lib::lsf_draw(sp, phase, tx, rx, b);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, d, nullptr, true);
    CHECK(qd_equal(a, b));
    CHECK(qd_equal(c, d)); // dual mobility as well
}

TEST_CASE("lsf_draw - single mobility ignores tx_pos")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 25);
    const arma::mat rx = rx7(), tx1 = {{10.0, 20.0, 25.0}}, tx2 = rx7() * 0.5 - 30.0;
    arma::mat a, b;
    quadriga_lib::lsf_draw(sp, phase, tx1, rx, a);
    quadriga_lib::lsf_draw(sp, phase, tx2, rx, b);
    CHECK(qd_equal(a, b));
}

TEST_CASE("lsf_draw - a link does not depend on the other links")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 26);
    const arma::mat rx = rx7(), tx = rx7() * 0.5 - 30.0;
    const arma::uvec fl = {0, 3, 3, 259, 0, 1, 259};

    for (int dual = 0; dual < 2; ++dual)
    {
        arma::mat all, one;
        quadriga_lib::lsf_draw(sp, phase, tx, rx, all, &fl, dual == 1);
        for (arma::uword l = 0; l < 7; ++l)
        {
            const arma::mat r = rx.row(l), t = tx.row(l);
            const arma::uvec f = {fl(l)};
            quadriga_lib::lsf_draw(sp, phase, t, r, one, &f, dual == 1);
            INFO("dual " << dual << ", link " << l);
            CHECK(qd_equal(one, all.row(l)));
        }
    }
}

TEST_CASE("lsf_draw - dual mobility values follow the two-device formulas")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 27);
    // 6 links; link 2 has Tx = Rx, link 4 a distance of more than 5 decorrelation distances (> 100 m)
    const arma::mat rx = {{10.0, -20.0, 1.5}, {-55.0, 30.0, 3.0}, {100.0, 100.0, 1.5}, {-5.0, -5.0, 12.0}, {0.0, 0.0, 1.5}, {33.0, 71.0, 2.0}};
    const arma::mat tx = {{12.0, -15.0, 25.0}, {-40.0, 10.0, 30.0}, {100.0, 100.0, 1.5}, {-25.0, 3.0, 12.0}, {150.0, -80.0, 10.0}, {30.0, 80.0, 1.5}};

    arma::mat rnd;
    quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, nullptr, true);
    REQUIRE(rnd.n_rows == 6);
    REQUIRE(rnd.n_cols == 11);

    // off: floor offsets of the link, added on both sides (both devices on the link's floor)
    auto oracle = [&](const arma::mat &R, const arma::mat &T, arma::uword l, arma::uword g, const arma::vec *off = nullptr)
    {
        const arma::uword m = mirror_g[g];
        const qd_preset_tab tg = qd_preset("Comb256", lambda_g[g]), tm = qd_preset("Comb256", lambda_g[m]);
        double a = qd_ref_sos(tg.F, phase.slice(0).colptr(g), R(l, 0), R(l, 1), 0.0, off);
        double b = qd_ref_sos(tm.F, phase.slice(0).colptr(m), T(l, 0), T(l, 1), 0.0, off);
        double D = std::sqrt((R(l, 0) - T(l, 0)) * (R(l, 0) - T(l, 0)) + (R(l, 1) - T(l, 1)) * (R(l, 1) - T(l, 1)));
        double x = (m == g) ? (a + b) / std::sqrt(2.0 + 2.0 * qd_acf_at(tg, D)) : (a + b) / std::sqrt(2.0);
        return g >= 9 ? qd_unif(x) : x;
    };

    QD_DIAG("\n# diag: lsf_draw dual mobility against the two-device formulas\n");
    QD_DIAG("link,g,value,oracle,diff\n");
    for (arma::uword l = 0; l < 6; ++l)
        for (arma::uword g = 0; g < 11; ++g)
        {
            double v = oracle(rx, tx, l, g);
            QD_DIAG("%llu,%llu,%.8f,%.8f,%.3e\n", (unsigned long long)l, (unsigned long long)g, rnd(l, g), v, rnd(l, g) - v);
            INFO("link " << l << ", g " << g);
            CHECK(std::abs(rnd(l, g) - v) < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
        }

    // One Tx row is used for every link
    const arma::mat tx1 = tx.row(4), tx_b = arma::repmat(tx1, 6, 1);
    quadriga_lib::lsf_draw(sp, phase, tx1, rx, rnd, nullptr, true);
    REQUIRE(rnd.n_rows == 6);
    for (arma::uword l = 0; l < 6; ++l)
        for (arma::uword g = 0; g < 11; ++g)
        {
            INFO("Tx broadcast, link " << l << ", g " << g);
            CHECK(std::abs(rnd(l, g) - oracle(rx, tx_b, l, g)) < (g >= 9 ? 1e-5 : 2e-5));
        }

    // Floors 3 (k = 3, b = 0) and 259 (k = 3, b = 1) mixed with floor 0: the offsets go to both sides
    const arma::uvec fl = {3, 259, 0, 259, 3, 0};
    const arma::vec off3 = qd_floor_offset(256, 3), off259 = qd_floor_offset(256, 259);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, &fl, true);
    REQUIRE(rnd.n_rows == 6);
    for (arma::uword l = 0; l < 6; ++l)
        for (arma::uword g = 0; g < 11; ++g)
        {
            const arma::vec *off = fl(l) == 3 ? &off3 : (fl(l) == 259 ? &off259 : nullptr);
            INFO("floor " << fl(l) << ", link " << l << ", g " << g);
            CHECK(std::abs(rnd(l, g) - oracle(rx, tx, l, g, off)) < (g >= 9 ? 1e-5 : 2e-5));
        }
}

TEST_CASE("lsf_draw - dual mobility keeps unit variance and correlates the pairs by the ACF at the Tx-Rx distance")
{
    auto sp = scen_distinct();
    sp["AS_D_lambda"] = 10.0, sp["AS_A_lambda"] = 15.0, sp["ES_D_lambda"] = 10.0, sp["ES_A_lambda"] = 20.0;
    const arma::cube phase = make_phase(256, 28);
    const arma::mat rx = qd_grid(100, 100, 50.0, 1.5);
    const qd_preset_tab t10 = qd_preset("Comb256", 10.0), t15 = qd_preset("Comb256", 15.0), t20 = qd_preset("Comb256", 20.0);

    // Variance: 5 standard errors of the sample variance, sqrt(2/n) = 0.014 -> 0.07
    // Correlation: 5 standard errors (<= 0.01) + approximation error of the SOS ACF (0.03) -> 0.08
    QD_DIAG("\n# diag: lsf_draw dual mobility, 10^4 links, Tx = Rx + (D, 0, 0)\n");
    QD_DIAG("D,var3,var4,var5,var6,corr34,ref34,corr56,ref56\n");
    const double Ds[3] = {5.0, 10.0, 20.0};
    for (double D : Ds)
    {
        arma::mat tx = rx;
        tx.col(0) += D;
        arma::mat rnd;
        quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, nullptr, true);
        REQUIRE(rnd.n_rows == rx.n_rows);
        double v[4] = {arma::var(rnd.col(3)), arma::var(rnd.col(4)), arma::var(rnd.col(5)), arma::var(rnd.col(6))};
        double c34 = qd_corr(rnd.col(3), rnd.col(4)), c56 = qd_corr(rnd.col(5), rnd.col(6));
        double r34 = 0.5 * (qd_acf_at(t10, D) + qd_acf_at(t15, D));
        double r56 = 0.5 * (qd_acf_at(t10, D) + qd_acf_at(t20, D));
        QD_DIAG("%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", D, v[0], v[1], v[2], v[3], c34, r34, c56, r56);
        INFO("D " << D);
        for (int i = 0; i < 4; ++i)
            CHECK(std::abs(v[i] - 1.0) < 0.07);
        CHECK(std::abs(c34 - r34) < 0.08);
        CHECK(std::abs(c56 - r56) < 0.08);
    }

    // D = 0: both pair members are the same sum
    arma::mat rnd;
    quadriga_lib::lsf_draw(sp, phase, rx, rx, rnd, nullptr, true);
    CHECK(arma::all(rnd.col(3) == rnd.col(4)));
    CHECK(arma::all(rnd.col(5) == rnd.col(6)));
}

TEST_CASE("lsf_draw - swapping Tx and Rx swaps ASD with ASA and ESD with ESA")
{
    const auto sp = scen_distinct(); // unequal distances of ASD/ASA and ESD/ESA
    arma::mat A = rx7(), B = rx7() * 0.7 + 20.0;
    B.col(2).fill(25.0);

    arma::cube ph_bias, ph_raw;
    quadriga_lib::lsf_init_phases(sp, ph_bias, 29, &B, &A, true); // computed once from the original positions
    quadriga_lib::lsf_init_phases(sp, ph_raw, 29, &B, &A, true, false);

    const arma::cube *phases[2] = {&ph_raw, &ph_bias};
    for (int i = 0; i < 2; ++i)
    {
        arma::mat ab, ba;
        quadriga_lib::lsf_draw(sp, *phases[i], B, A, ab, nullptr, true); // tx = B, rx = A
        quadriga_lib::lsf_draw(sp, *phases[i], A, B, ba, nullptr, true); // tx = A, rx = B
        REQUIRE(ab.n_cols == 11);
        REQUIRE(ba.n_cols == 11);
        INFO("bias removal " << i);
        const arma::uword same[7] = {0, 1, 2, 7, 8, 9, 10};
        for (arma::uword c : same)
            CHECK(arma::all(ab.col(c) == ba.col(c)));
        CHECK(arma::all(ab.col(3) == ba.col(4)));
        CHECK(arma::all(ab.col(4) == ba.col(3)));
        CHECK(arma::all(ab.col(5) == ba.col(6)));
        CHECK(arma::all(ab.col(6) == ba.col(5)));
    }
}

TEST_CASE("lsf_draw - floor 0 equals no floor ID and equal IDs share one field")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 30);
    const arma::mat rx = rx7(), tx = {{10.0, 20.0, 25.0}};

    arma::mat a, b;
    const arma::uvec fl0(7, arma::fill::zeros);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, a);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, b, &fl0);
    CHECK(qd_equal(a, b));

    // Floors 3 (k = 3, b = 0) and 259 (k = 3, b = 1) mixed with floor 0 in one call
    const arma::uvec fl = {3, 259, 3, 0, 259, 3, 259};
    quadriga_lib::lsf_draw(sp, phase, tx, rx, b, &fl);
    REQUIRE(b.n_rows == 7);
    const arma::vec off3 = qd_floor_offset(256, 3), off259 = qd_floor_offset(256, 259), off0(256, arma::fill::zeros);
    QD_DIAG("\n# diag: lsf_draw floors against ref_sos with typed offsets\n");
    QD_DIAG("link,floor,g,value,oracle,diff\n");
    for (arma::uword l = 0; l < 7; ++l)
        for (arma::uword g = 0; g < 11; ++g)
        {
            const arma::vec *off = fl(l) == 3 ? &off3 : (fl(l) == 259 ? &off259 : &off0);
            const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
            double v = qd_ref_sos(F, phase.slice(0).colptr(g), rx(l, 0), rx(l, 1), 0.0, off);
            if (g >= 9)
                v = qd_unif(v);
            QD_DIAG("%llu,%llu,%llu,%.8f,%.8f,%.3e\n", (unsigned long long)l, (unsigned long long)fl(l), (unsigned long long)g, b(l, g), v, b(l, g) - v);
            INFO("link " << l << ", floor " << fl(l) << ", g " << g);
            CHECK(std::abs(b(l, g) - v) < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
        }
}

TEST_CASE("lsf_draw - fields of different floors are uncorrelated at equal positions")
{
    const auto sp = scen_equal(10.0);
    const arma::cube phase = make_phase(256, 31);
    const arma::mat g0 = qd_grid(100, 100, 50.0, 1.5); // n = 10^4, spacing 5 decorrelation distances (ACF < 0.02)
    const arma::uword n = g0.n_rows;
    const arma::uword floors[5] = {0, 1, 5, 255, 256};

    arma::mat rx(5 * n, 3);
    arma::uvec fl(5 * n);
    for (arma::uword f = 0; f < 5; ++f)
        rx.rows(f * n, f * n + n - 1) = g0, fl.subvec(f * n, f * n + n - 1).fill(floors[f]);
    const arma::mat tx = {{0.0, 0.0, 25.0}};

    arma::mat rnd;
    quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, &fl);
    REQUIRE(rnd.n_rows == 5 * n);

    // Pairs (index into 'floors') and bounds: within a block of N = 256 floor IDs the correlation at zero
    // horizontal lag vanishes, 0.05 = 5 standard errors of 1/sqrt(n); floors 0 and 256 (different blocks) have
    // a random correlation with standard deviation 1/sqrt(2N) = 0.044, bound 0.25
    const int pairs[4][2] = {{0, 1}, {1, 2}, {0, 3}, {0, 4}};
    const double bound[4] = {0.05, 0.05, 0.05, 0.25};
    QD_DIAG("\n# diag: lsf_draw correlation between floors at equal positions\n");
    QD_DIAG("floor_a,floor_b,g,corr\n");
    for (int p = 0; p < 4; ++p)
        for (arma::uword g = 0; g < 9; ++g)
        {
            const arma::uword fa = (arma::uword)pairs[p][0], fb = (arma::uword)pairs[p][1];
            double c = qd_corr(rnd.col(g).rows(fa * n, fa * n + n - 1), rnd.col(g).rows(fb * n, fb * n + n - 1));
            QD_DIAG("%llu,%llu,%llu,%.4f\n", (unsigned long long)floors[fa], (unsigned long long)floors[fb], (unsigned long long)g, c);
            INFO("floors " << floors[fa] << ", " << floors[fb] << ", g " << g);
            CHECK(std::abs(c) < bound[p]);
        }
}

TEST_CASE("lsf_draw - marginal distributions")
{
    const auto sp = scen_equal(10.0);
    const arma::cube phase = make_phase(256, 32);
    const arma::mat rx = qd_grid(100, 100, 50.0, 1.5), tx = {{0.0, 0.0, 25.0}};
    arma::mat rnd;
    quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd);
    REQUIRE(rnd.n_cols == 11);

    // 5 standard errors with n = 10^4: mean 0.05, std 0.05 (0.01 each); Uniform mean 0.015 (0.003),
    // fraction below 0.1 within 0.015 (0.003)
    QD_DIAG("\n# diag: lsf_draw marginal statistics, 10^4 links\n");
    QD_DIAG("g,mean,std,frac_below_0.1\n");
    for (arma::uword g = 0; g < 11; ++g)
    {
        const arma::vec c = rnd.col(g);
        double m = arma::mean(c), s = std::sqrt(arma::var(c)), frac = (double)arma::accu(c < 0.1) / (double)c.n_elem;
        QD_DIAG("%llu,%.4f,%.4f,%.4f\n", (unsigned long long)g, m, s, frac);
        INFO("g " << g);
        if (g < 9)
        {
            CHECK(std::abs(m) < 0.05);
            CHECK(std::abs(s - 1.0) < 0.05);
        }
        else
        {
            CHECK(c.min() >= 0.0);
            CHECK(c.max() <= 1.0);
            CHECK(std::abs(m - 0.5) < 0.015);
            CHECK(std::abs(frac - 0.1) < 0.015);
        }
    }
}

TEST_CASE("lsf_draw - the correlation follows the preset ACF")
{
    // Tolerance 0.1 >= 5 standard errors (0.05) + directional error of the SOS approximation (up to 0.045):
    // the sample correlation converges to the ACF of the sinusoids along the offset direction, not to the table
    // ACF (e.g. Comb256 at 10 m: 0.408 along x, 0.393 along y, against 0.368; Exp256 at 5 m along y: 0.572
    // against 0.607)
    const auto sp = scen_equal(10.0);
    const arma::mat base = qd_grid(100, 100, 50.0, 1.5), tx = {{0.0, 0.0, 25.0}};
    const char *presets[2] = {"Comb256", "Exp256"};
    const double offsets[3] = {5.0, 10.0, 20.0};
    const double az[4] = {0.0, 0.25 * pi, 0.5 * pi, 0.75 * pi};

    QD_DIAG("\n# diag: lsf_draw sample correlation against the preset ACF (column 0 shown, all Normal columns checked)\n");
    QD_DIAG("preset,offset,azimuth_deg,corr_g0,acf\n");
    for (int p = 0; p < 2; ++p)
    {
        const arma::cube phase = make_phase(256, 33 + p);
        const qd_preset_tab t = qd_preset(presets[p], 10.0);
        arma::mat a;
        quadriga_lib::lsf_draw(sp, phase, tx, base, a, nullptr, false, presets[p]);
        for (double d : offsets)
            for (double phi : az)
            {
                arma::mat moved = base, b;
                moved.col(0) += d * std::cos(phi), moved.col(1) += d * std::sin(phi);
                quadriga_lib::lsf_draw(sp, phase, tx, moved, b, nullptr, false, presets[p]);
                double ref = qd_acf_at(t, d);
                QD_DIAG("%s,%.0f,%.0f,%.4f,%.4f\n", presets[p], d, phi * 180.0 / pi, qd_corr(a.col(0), b.col(0)), ref);
                for (arma::uword g = 0; g < 9; ++g)
                {
                    INFO(presets[p] << ", offset " << d << ", azimuth " << phi << ", g " << g);
                    CHECK(std::abs(qd_corr(a.col(g), b.col(g)) - ref) < 0.1);
                }
            }
    }
}

TEST_CASE("lsf_draw - sizes, empty input and pre-sized output")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 35);
    const arma::mat rx0(0, 3), tx1 = {{0.0, 0.0, 25.0}}, tx0(0, 3);
    arma::mat rnd(4, 4, arma::fill::ones);

    CHECK_NOTHROW(quadriga_lib::lsf_draw(sp, phase, tx1, rx0, rnd));
    CHECK(rnd.n_rows == 0);
    CHECK(rnd.n_cols == 11);
    rnd.ones(4, 4);
    CHECK_NOTHROW(quadriga_lib::lsf_draw(sp, phase, tx0, rx0, rnd, nullptr, true));
    CHECK(rnd.n_rows == 0);
    CHECK(rnd.n_cols == 11);

    // Pre-sized output with stale content: written in place (same memory) and completely, single and dual mobility
    const arma::mat rx = rx7(), tx = rx7() + 5.0;
    const arma::uvec fl = {0, 3, 3, 259, 0, 1, 259};
    arma::mat ref, ref1;
    quadriga_lib::lsf_draw(sp, phase, tx, rx, ref1, &fl);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, ref, nullptr, true);
    arma::mat pre(7, 11);
    pre.fill(99.0);
    const double *mem = pre.memptr();
    quadriga_lib::lsf_draw(sp, phase, tx, rx, pre, &fl);
    CHECK(pre.memptr() == mem);
    CHECK(qd_equal(pre, ref1));
    pre.fill(99.0);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, pre, nullptr, true);
    CHECK(pre.memptr() == mem);
    CHECK(qd_equal(pre, ref));

    // Wrong size: resized
    arma::mat wrong(3, 3, arma::fill::ones);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, wrong, nullptr, true);
    CHECK(qd_equal(wrong, ref));
}

TEST_CASE("lsf_draw - input validation")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 36);
    const arma::mat rx = rx7().rows(0, 2), tx = {{0.0, 0.0, 25.0}};
    const arma::uvec fl = {0, 1, 2};
    arma::mat rnd;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd));
        CHECK_NOTHROW(quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, &fl, true));
        CHECK_NOTHROW(quadriga_lib::lsf_draw({}, phase, tx, rx, rnd)); // empty scenpar: all defaults
    }
    SECTION("phase")
    {
        const std::string msg = "Input 'phase' must have size [256, 11, 2] for preset 'Comb256'.";
        arma::cube bad(128, 11, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, bad, tx, rx, rnd), std::invalid_argument, Message(msg));
        bad.zeros(256, 10, 2);
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, bad, tx, rx, rnd), std::invalid_argument, Message(msg));
        bad.zeros(256, 11, 1);
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, bad, tx, rx, rnd), std::invalid_argument, Message(msg));
        bad = phase;
        bad(5, 3, 1) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, bad, tx, rx, rnd), std::invalid_argument,
                             Message("Input 'phase' must be finite."));
    }
    SECTION("preset")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, nullptr, false, "Comb999"), std::invalid_argument,
                             Message("Input 'preset' is not a known preset."));
    }
    SECTION("rx_pos")
    {
        arma::mat bad(3, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, phase, tx, bad, rnd), std::invalid_argument,
                             Message("Input 'rx_pos' must have 3 columns."));
    }
    SECTION("tx_pos")
    {
        arma::mat bad = tx;
        bad(0, 1) = INFINITY;
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, phase, bad, rx, rnd), std::invalid_argument,
                             Message("Input 'tx_pos' must be finite."));
        bad.zeros(2, 3);
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, phase, bad, rx, rnd), std::invalid_argument,
                             Message("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'."));
    }
    SECTION("floor_id")
    {
        const arma::uvec bad = {0, 1};
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, &bad), std::invalid_argument,
                             Message("Number of elements in 'floor_id' does not match number of rows in 'rx_pos'."));

        // 3D mode: the height separates the floors, floor IDs are rejected (also all zero); the output is untouched
        const std::string msg = "Input 'floor_id' cannot be used with 'horizontal = false': in 3D mode the height separates the floors.";
        const arma::uvec fl0 = {0, 0, 0};
        arma::mat keep(2, 2, arma::fill::ones);
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, phase, tx, rx, keep, &fl, false, "Comb256", false),
                             std::invalid_argument, Message(msg));
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(sp, phase, tx, rx, keep, &fl0, true, "Comb256", false),
                             std::invalid_argument, Message(msg));
        CHECK(keep.n_rows == 2);
        CHECK(arma::all(arma::vectorise(keep) == 1.0));
        CHECK_NOTHROW(quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, nullptr, false, "Comb256", false));
    }
    SECTION("scenpar")
    {
        auto bad = sp;
        bad["XPR_lambda"] = -1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::lsf_draw(bad, phase, tx, rx, rnd), std::invalid_argument,
                             Message("Value of 'XPR_lambda' must be finite and >= 0."));
    }
}
