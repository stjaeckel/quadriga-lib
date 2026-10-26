// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "lsf_init_phases*"

// Verification suite for quadriga_lib::lsf_init_phases.
//
// Pins down: the size of the phase cube per preset, the raw phases (slice 0 equals sos_init_phases with the same
// seed), the mirror relation of slice 1 (ASD <-> ASA, ESD <-> ESA, all others themselves), the seed, the bias
// removal (zero sample mean of every Normal field over the drop positions in single and dual mobility, 2D and 3D,
// Uniform generators 9 and 10 untouched, no correction below 3 positions), the replacement of a decorrelation
// distance of 0 by 0.1 m, and the input validation.
// Left to test_lsf_draw.cpp: the evaluation of the fields (lsf_draw is used here only as a supporting function to
// show the zero link mean in dual mobility).
// Oracles, independent of the function:
//  - ref_sos(F, phi, x) = sqrt(2/N) · sum_n cos( 2·pi · F(n,:)·x + phi(n) ), the sum-of-sinusoids field typed here,
//    with the frequencies of sos_preset at the decorrelation distance of each generator
//  - the mirror table and the raw phases of sos_init_phases (documented contract)
// Conventions: positions are rows {x, y, z} in meters; the scenario sets all 10 decorrelation distances to distinct
// values (DS 11, KF 12, SF 13, ASD 14, ASA 15, ESD 16, ESA 17, XPR 18, absTOA 19, SC 20 m) so that a mix-up of
// generators changes the result; generator 10 (hE) uses SF_lambda.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_sos.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

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

    // Oracle: sum of sinusoids sqrt(2/N) · sum_n cos( 2·pi · (F(n,0)·x + F(n,1)·y + F(n,2)·z) + phi(n) )
    double ref_sos(const arma::mat &F, const double *phi, double x, double y, double z)
    {
        double s = 0.0;
        for (arma::uword n = 0; n < F.n_rows; ++n)
            s += std::cos(2.0 * pi * (F(n, 0) * x + F(n, 1) * y + F(n, 2) * z) + phi[n]);
        return std::sqrt(2.0 / (double)F.n_rows) * s;
    }

    // Mean of ref_sos over the rows of U; use_z = false evaluates at z = 0
    double mean_ref(const arma::mat &F, const double *phi, const arma::mat &U, bool use_z)
    {
        double s = 0.0;
        for (arma::uword i = 0; i < U.n_rows; ++i)
            s += ref_sos(F, phi, U(i, 0), U(i, 1), use_z ? U(i, 2) : 0.0);
        return s / (double)U.n_rows;
    }

    // Frequencies of the preset at decorrelation distance d
    arma::mat preset_freq(const std::string &preset, double d)
    {
        arma::mat F;
        quadriga_lib::sos_preset<double>(preset, F, nullptr, nullptr, d);
        return F;
    }

    // Raw phases of the documented contract: sos_init_phases(n_coeff, 11, seed)
    arma::mat raw_phases(arma::uword n_coeff, arma::sword seed)
    {
        arma::mat P;
        quadriga_lib::sos_init_phases<double>(n_coeff, 11, P, seed);
        return P;
    }

    // Message of the std::invalid_argument thrown by f, or a marker if none (or another type) was thrown
    std::string err_msg(const std::function<void()> &f)
    {
        try
        {
            f();
        }
        catch (const std::invalid_argument &e)
        {
            return e.what();
        }
        catch (...)
        {
            return "<other exception>";
        }
        return "<no exception>";
    }

    // Check that slice 1 is the mirror of slice 0
    bool is_mirror(const arma::cube &phase)
    {
        for (arma::uword g = 0; g < 11; ++g)
            if (arma::any(phase.slice(1).col(g) != phase.slice(0).col(mirror_g[g])))
                return false;
        return true;
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
        const arma::mat F = preset_freq("Comb256", lambda_g[g]);
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
        const arma::mat F = preset_freq("Comb256", lambda_g[g]);
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
        const arma::mat F = preset_freq("Comb256", lambda_g[g]);
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

    const arma::mat F = preset_freq("Comb256", 0.1);
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
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(sp, phase, -2, &tx, &rx); }) == "Input 'seed' must be >= -1.");
    }
    SECTION("preset")
    {
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(sp, phase, 1, &tx, &rx, false, true, "Comb999"); }) ==
              "Input 'preset' is not a known preset.");
    }
    SECTION("scenpar value")
    {
        auto bad = sp;
        bad["DS_lambda"] = -1.0;
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(bad, phase, 1); }) == "Value of 'DS_lambda' must be finite and >= 0.");
        bad = sp;
        bad["SC_lambda"] = INFINITY;
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(bad, phase, 1); }) == "Value of 'SC_lambda' must be finite and >= 0.");
    }
    SECTION("scenpar unknown key")
    {
        auto bad = sp;
        bad["DS_lambdaX"] = 1.0;
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(bad, phase, 1); }) == "Unknown key 'DS_lambdaX' in 'scenpar'.");
    }
    SECTION("rx_pos")
    {
        arma::mat bad(5, 2, arma::fill::zeros);
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(sp, phase, 1, &tx, &bad); }) == "Input 'rx_pos' must have 3 columns.");
        bad = rx;
        bad(2, 1) = NAN;
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(sp, phase, 1, &tx, &bad); }) == "Input 'rx_pos' must be finite.");
    }
    SECTION("tx_pos")
    {
        arma::mat bad(1, 2, arma::fill::zeros);
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(sp, phase, 1, &bad, &rx); }) == "Input 'tx_pos' must have 3 columns.");
        bad.zeros(2, 3);
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(sp, phase, 1, &bad, &rx); }) ==
              "Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'.");
    }
    SECTION("dual mobility without tx_pos")
    {
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_init_phases(sp, phase, 1, nullptr, &rx, true); }) ==
              "Input 'tx_pos' is required for the bias removal with dual mobility.");
    }
}
