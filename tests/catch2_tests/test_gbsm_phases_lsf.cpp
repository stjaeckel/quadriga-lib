// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "gbsm_*"

// Verification suite for the "lsf" set of quadriga_lib::gbsm_init_phases and quadriga_lib::gbsm_draw (test cases
// "gbsm_init_phases - lsf: ..." and "gbsm_draw - lsf: ...") and for the input validation of gbsm_init_phases
// ("gbsm_init_phases - input validation", all set types). Golden values that keep stored seeds reproducible:
// test_gbsm_phases_golden.cpp.
//
// gbsm_init_phases ("lsf") pins down: the size of the phase cube per preset, the raw phases (slice 0 equals
// sos_init_phases with the same seed), the mirror relation of slice 1 (ASD <-> ASA, ESD <-> ESA, all others
// themselves), the seed, the bias removal (zero sample mean of every Normal field over the drop positions in single and
// dual mobility, 2D and 3D, Uniform generators 9 and 10 untouched, no correction below 3 positions), the replacement of
// a decorrelation distance of 0 by 0.1 m, and the input validation.
// gbsm_draw ("lsf") pins down: the values of the 11 generators in single mobility (each generator with its own
// decorrelation distance, 2D and 3D mode, three presets), the Uniform map of columns 9 and 10, the two-device values
// of dual mobility (self-mirrored generators reciprocal, the pairs ASD/ASA and ESD/ESA one field with Tx and Rx
// exchanged), also with floor IDs, unit variance and the pair correlation in dual mobility, the decorrelation when one
// end moves away, the swap reciprocity, the
// floor phase offsets, the statistics (marginals, ACF, zero correlation of different floors at equal positions), the
// independence of z in 2D mode, of tx_pos in single mobility and of the other links, sizes, empty input, pre-sized
// outputs and the input validation.
// Oracles, independent of the functions:
//  - the sum of sinusoids qd_ref_sos, the Uniform map and the floor offsets of qd_test_sos_oracle.hpp, with the
//    frequencies of sos_preset at the decorrelation distance of each generator
//  - the mirror table and the raw phases of sos_init_phases (documented contract)
//  - dual mobility: qd_ref_dual, the two-device value of sos_val: self-mirrored generators reciprocal at (R, T) with
//    their own frequencies; a pair (g, m), owner o = min(g, m), the plain two-device sum of the owner with the Tx
//    frequencies scaled by lambda_o / lambda_m, at (R, T) for g = o and at (T, R) for the partner, phases of the owner
//  - statistics: Normal(0,1), Uniform(0,1), the ACF table of sos_preset
// Tolerances of gbsm_draw ("lsf") values: gbsm_draw ("lsf") evaluates the sinusoids with the AVX2 kernel where
// available (sine and cosine in single precision, accumulation in double, deviation below 4e-7·sqrt(2·n_coeff) = 9e-6
// for 256 sinusoids). Normal columns 2e-5 for 256 sinusoids, scaled with sqrt(n_coeff/256) for other presets, Uniform
// columns 1e-5 (the slope of the Uniform map is at most 0.4). The bias removal is closed-form: 1e-12.
// Conventions: positions are columns {x, y, z} in meters; most cases use distinct decorrelation distances (DS 11, KF 12,
// SF 13, ASD 14, ASA 15, ESD 16, ESA 17, XPR 18, absTOA 19, SC 20 m; generator 10 uses SF_lambda) so that a mix-up of
// generators changes the result; the phase cubes of the gbsm_draw ("lsf") cases are raw phases of sos_init_phases with
// the Tx side mirrored (ASD <-> ASA, ESD <-> ESA).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_sos.hpp"

#include <algorithm>
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
        return p.t();
    }

    // Raw phases of the documented contract: sos_init_phases(n_coeff, 11, seed)
    arma::mat raw_phases(arma::uword n_coeff, arma::sword seed)
    {
        arma::mat P;
        P = quadriga_lib::sos_init_phases<double>(n_coeff, 11, seed);
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

    // Mean of qd_ref_sos over the columns of U; use_z = false evaluates at z = 0
    double mean_ref(const arma::mat &F, const double *phi, const arma::mat &U, bool use_z)
    {
        double s = 0.0;
        for (arma::uword i = 0; i < U.n_cols; ++i)
            s += qd_ref_sos(F, phi, U(0, i), U(1, i), use_z ? U(2, i) : 0.0);
        return s / (double)U.n_cols;
    }

    // 7 links with distinct coordinates, negative values, up to 300 m, z != 0
    arma::mat rx7()
    {
        return arma::mat({{12.5, -40.0, 1.5}, {-120.0, 33.0, 4.5}, {250.0, 300.0, 22.0}, {-7.25, -299.0, 1.8},
                          {88.0, 0.5, 10.0}, {0.0, 160.0, 31.5}, {-210.0, -95.5, 7.0}}).t();
    }
}

TEST_CASE("gbsm_init_phases - lsf: output size follows the preset")
{
    const auto sp = scen_distinct();
    const char *presets[3] = {"Comb256", "Exp128", "Gauss1024"};
    const arma::uword n_coeff[3] = {256, 128, 1024};
    for (int i = 0; i < 3; ++i)
    {
        INFO("preset " << presets[i]);
        arma::cube phase;
        phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 3, -1, presets[i]);
        CHECK(phase.n_rows == n_coeff[i]);
        CHECK(phase.n_cols == 11);
        CHECK(phase.n_slices == 2);
    }

}

TEST_CASE("gbsm_init_phases - lsf: slice 0 equals sos_init_phases with the same seed")
{
    const auto sp = scen_distinct();
    arma::cube phase;
    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 42);
    REQUIRE(phase.n_rows == 256);
    REQUIRE(phase.n_cols == 11);
    REQUIRE(phase.n_slices == 2);
    CHECK(arma::all(arma::vectorise(phase.slice(0) == raw_phases(256, 42))));

    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 42, -1, "Exp128");
    REQUIRE(phase.n_rows == 128);
    CHECK(arma::all(arma::vectorise(phase.slice(0) == raw_phases(128, 42))));
}

TEST_CASE("gbsm_init_phases - lsf: slice 1 is the mirror of slice 0")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(50, 100.0, 11);
    arma::mat tx = rx;
    tx.row(0) += 30.0, tx.row(1) -= 20.0;
    const arma::mat tx1 = arma::vec{5.0, -7.0, 25.0};

    arma::cube phase;
    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 5);
    CHECK(is_mirror(phase)); // no bias removal (no positions)

    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 5, -1, "Comb256", tx1, rx); // single mobility, bias removal
    CHECK(is_mirror(phase));
    CHECK(arma::any(arma::vectorise(phase.slice(0) != raw_phases(256, 5)))); // the correction did act

    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 5, -1, "Comb256", tx, rx, true); // dual mobility, bias removal
    CHECK(is_mirror(phase));

    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 5, -1, "Comb256", tx, rx, true, false); // dual mobility, no bias removal
    CHECK(is_mirror(phase));
}

TEST_CASE("gbsm_init_phases - lsf: the same seed gives the same phases")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(30, 100.0, 21);
    arma::cube a, b, c;
    a = quadriga_lib::gbsm_init_phases(sp, "lsf", 7, -1, "Comb256", {}, rx);
    b = quadriga_lib::gbsm_init_phases(sp, "lsf", 7, -1, "Comb256", {}, rx);
    c = quadriga_lib::gbsm_init_phases(sp, "lsf", 8, -1, "Comb256", {}, rx);
    CHECK(arma::all(arma::vectorise(a == b))); // determinism: repeated call, identical result
    CHECK(arma::any(arma::vectorise(a != c)));
}

TEST_CASE("gbsm_init_phases - lsf: bias removal zeroes the drop mean in single mobility")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(200, 100.0, 31);
    const arma::mat tx = arma::vec{50.0, 50.0, 25.0};

    arma::cube phase, phase_raw;
    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 9, -1, "Comb256", tx, rx);
    phase_raw = quadriga_lib::gbsm_init_phases(sp, "lsf", 9, -1, "Comb256", tx, rx, false, false);
    REQUIRE(phase.n_rows == 256);

    // Closed-form correction: the mean is zero up to the rounding of 200 O(1) values, 1e-12
    QD_DIAG("\n# diag: gbsm_init_phases lsf single mobility, mean of ref_sos over 200 positions (2D)\n");
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

TEST_CASE("gbsm_init_phases - lsf: bias removal in 3D mode uses the z coordinate")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(200, 100.0, 41);
    const arma::mat tx = arma::vec{50.0, 50.0, 25.0};

    arma::cube phase;
    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 10, -1, "Comb256", tx, rx, false, true, false);

    double max_2d = 0.0;
    QD_DIAG("\n# diag: gbsm_init_phases lsf 3D mode, mean of ref_sos with and without z\n");
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

TEST_CASE("gbsm_init_phases - lsf: bias removal with dual mobility zeroes the link mean of the two-device values")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(100, 100.0, 51), tx = drop(100, 100.0, 52);

    arma::cube phase, phase_sw;
    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 12, -1, "Comb256", tx, rx, true);
    phase_sw = quadriga_lib::gbsm_init_phases(sp, "lsf", 12, -1, "Comb256", rx, tx, true); // swapped roles
    REQUIRE(phase.n_rows == 256);

    // Link mean of the two-device value of every owner (self-mirrored, ASD, ESD; 2D: z = 0), closed form: 1e-12
    const std::vector<arma::uword> p = qd_perm(256);
    const arma::uword owners[7] = {0, 1, 2, 3, 5, 7, 8};
    QD_DIAG("\n# diag: gbsm_init_phases lsf dual mobility, link mean of the two-device values\ng,mean\n");
    for (arma::uword g : owners)
    {
        const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
        const arma::uword m = mirror_g[g];
        double s = 0.0;
        for (arma::uword l = 0; l < 100; ++l)
        {
            const double a[3] = {rx(0, l), rx(1, l), 0.0}, b[3] = {tx(0, l), tx(1, l), 0.0};
            s += qd_ref_dual(F, phase.slice(0).colptr(g), a, b, p, lambda_g[g] / lambda_g[m], m == g) / 100.0;
        }
        QD_DIAG("%llu,%.3e\n", (unsigned long long)g, s);
        INFO("g " << g);
        CHECK(std::abs(s) < 1e-12);
    }

    // Swapped Tx and Rx: the self-mirrored fields are reciprocal, so the same values are summed in another order and
    // the phases are equal modulo 2·pi up to rounding
    double max_dphi = 0.0;
    for (arma::uword g : {0, 1, 2, 7, 8})
        for (arma::uword n = 0; n < 256; ++n)
            max_dphi = std::max(max_dphi, std::abs(std::remainder(phase(n, g, 0) - phase_sw(n, g, 0), 2.0 * pi)));
    QD_DIAG("# diag: max phase difference of the self-mirrored columns after swapping Tx and Rx: %.3e\n", max_dphi);
    CHECK(max_dphi < 1e-12);

    // Supporting check through gbsm_draw ("lsf"): the link mean of the owner columns is 0
    // Tolerance 2e-5: gbsm_draw ("lsf") evaluates with the AVX2 kernel (single-precision sin/cos where available).
    arma::mat rnd;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb256", {}, {}, true);
    REQUIRE(rnd.n_rows == 100);
    REQUIRE(rnd.n_cols == 11);
    for (arma::uword c : owners)
    {
        INFO("column " << c);
        CHECK(std::abs(arma::mean(rnd.col(c))) < 2e-5);
    }
}

TEST_CASE("gbsm_init_phases - lsf: no bias removal for fewer than 3 positions, without positions or when switched off")
{
    const auto sp = scen_distinct();
    const arma::mat raw = raw_phases(256, 13);
    auto equals_raw = [&](const arma::cube &phase)
    {
        if (phase.n_rows != 256 || phase.n_cols != 11 || phase.n_slices != 2)
            return false;
        return arma::all(arma::vectorise(phase.slice(0) == raw)) && is_mirror(phase);
    };

    const arma::mat rx2 = drop(2, 100.0, 61), rx200 = drop(200, 100.0, 62), rx0(3, 0);
    const arma::mat tx = arma::vec{0.0, 0.0, 25.0};
    arma::cube phase;

    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 13, -1, "Comb256", tx, rx2);
    CHECK(equals_raw(phase));
    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 13);
    CHECK(equals_raw(phase));
    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 13, -1, "Comb256", tx, rx200, false, false);
    CHECK(equals_raw(phase));
    CHECK_NOTHROW(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 13, -1, "Comb256", tx, rx0));
    CHECK(equals_raw(phase));
}

TEST_CASE("gbsm_init_phases - lsf: a decorrelation distance of 0 is used as 0.1 m")
{
    auto sp = scen_distinct();
    sp["DS_lambda"] = 0.0;
    // A 10 m square (100 decorrelation distances): the arguments of the sinusoids stay small enough for the
    // 1e-12 tolerance of the closed-form correction
    const arma::mat rx = drop(200, 10.0, 71);
    arma::cube phase;
    phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 14, -1, "Comb256", {}, rx);

    const arma::mat F = qd_preset("Comb256", 0.1).F;
    double m = mean_ref(F, phase.slice(0).colptr(0), rx, false);
    QD_DIAG("\n# diag: gbsm_init_phases lsf DS_lambda = 0, mean with the 0.1 m table: %.3e\n", m);
    CHECK(std::abs(m) < 1e-12);
}

TEST_CASE("gbsm_init_phases - input validation")
{
    const auto sp = scen_distinct();
    const arma::mat rx = drop(5, 100.0, 81);
    const arma::mat tx = arma::vec{0.0, 0.0, 25.0};
    arma::cube phase;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 1, -1, "Comb256", tx, rx));
        CHECK_NOTHROW(phase = quadriga_lib::gbsm_init_phases(sp, "lsf"));
        CHECK_NOTHROW(phase = quadriga_lib::gbsm_init_phases({}, "lsf", 1)); // empty scenpar: all defaults
        CHECK_NOTHROW(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 1, -1, "Comb256", {}, rx, true, false)); // dual mobility without tx_pos, no bias removal
    }

    SECTION("seed")
    {
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", -2, -1, "Comb256", tx, rx), std::invalid_argument,
                             Message("Input 'seed' must be >= -1."));
    }
    SECTION("preset")
    {
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 1, -1, "Comb999", tx, rx), std::invalid_argument,
                             Message("Input 'preset' is not a known preset."));
    }
    SECTION("scenpar value")
    {
        auto bad = sp;
        bad["DS_lambda"] = -1.0;
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(bad, "lsf", 1), std::invalid_argument,
                             Message("Value of 'DS_lambda' must be finite and >= 0."));
        bad = sp;
        bad["SC_lambda"] = INFINITY;
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(bad, "lsf", 1), std::invalid_argument,
                             Message("Value of 'SC_lambda' must be finite and >= 0."));
    }
    SECTION("scenpar unknown key")
    {
        auto bad = sp;
        bad["DS_lambdaX"] = 1.0;
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(bad, "lsf", 1), std::invalid_argument,
                             Message("Unknown key 'DS_lambdaX' in 'scenpar'."));
    }
    SECTION("rx_pos")
    {
        arma::mat bad(2, 5, arma::fill::zeros);
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 1, -1, "Comb256", tx, bad), std::invalid_argument,
                             Message("Input 'rx_pos' must have 3 rows."));
        bad = rx;
        bad(1, 2) = NAN;
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 1, -1, "Comb256", tx, bad), std::invalid_argument,
                             Message("Input 'rx_pos' must be finite."));
    }
    SECTION("tx_pos")
    {
        arma::mat bad(2, 1, arma::fill::zeros);
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 1, -1, "Comb256", bad, rx), std::invalid_argument,
                             Message("Input 'tx_pos' must have 3 rows."));
        bad.zeros(3, 2);
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 1, -1, "Comb256", bad, rx), std::invalid_argument,
                             Message("Number of columns in 'tx_pos' must be 1 or match the number of columns in 'rx_pos'."));
    }
    SECTION("dual mobility without tx_pos")
    {
        CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, "lsf", 1, -1, "Comb256", {}, rx, true), std::invalid_argument,
                             Message("Input 'tx_pos' is required for the bias removal with dual mobility."));
    }
    SECTION("set_type, checked first; the output is untouched on error")
    {
        arma::cube keep(2, 2, 2, arma::fill::ones);
        for (const char *t : {"LSF", "", "clusters", "rays"})
        {
            INFO("set_type " << t);
            CHECK_THROWS_MATCHES(keep = quadriga_lib::gbsm_init_phases(sp, t, -2, -5, "Comb999", tx, rx), std::invalid_argument,
                                 Message("Input 'set_type' must be 'lsf', 'cluster' or 'ray'."));
        }
        CHECK(keep.n_rows == 2);
        CHECK(arma::all(arma::vectorise(keep) == 1.0));
    }
    SECTION("set_id is ignored for lsf")
    {
        arma::cube a, b;
        a = quadriga_lib::gbsm_init_phases(sp, "lsf", 7, -1);
        CHECK_NOTHROW(b = quadriga_lib::gbsm_init_phases(sp, "lsf", 7, -5));
        CHECK(arma::approx_equal(a, b, "absdiff", 0.0));
    }
    SECTION("positions with another set type")
    {
        for (const char *t : {"cluster", "ray"})
        {
            INFO("set_type " << t);
            CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, t, 1, -1, "Comb256", tx, rx), std::invalid_argument,
                                 Message("Inputs 'tx_pos' and 'rx_pos' are used for set type 'lsf' only."));
            CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, t, 1, -1, "Comb256", {}, rx), std::invalid_argument,
                                 Message("Inputs 'tx_pos' and 'rx_pos' are used for set type 'lsf' only."));
            CHECK_THROWS_MATCHES(phase = quadriga_lib::gbsm_init_phases(sp, t, 1, -1, "Comb256", tx), std::invalid_argument,
                                 Message("Inputs 'tx_pos' and 'rx_pos' are used for set type 'lsf' only."));
        }
    }
}

TEST_CASE("gbsm_draw - lsf: single mobility values equal the sum of sinusoids of each generator")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 21);
    const arma::mat rx = rx7(), tx = arma::vec{10.0, 20.0, 25.0};

    arma::mat rnd, rnd2;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd);
    REQUIRE(rnd.n_rows == 7);
    REQUIRE(rnd.n_cols == 11);

    QD_DIAG("\n# diag: gbsm_draw lsf single mobility (2D) against ref_sos, max abs difference per generator\n");
    QD_DIAG("g,lambda,max_diff\n");
    for (arma::uword g = 0; g < 11; ++g)
    {
        const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
        double max_d = 0.0;
        for (arma::uword l = 0; l < 7; ++l)
        {
            double v = qd_ref_sos(F, phase.slice(0).colptr(g), rx(0, l), rx(1, l), 0.0);
            if (g >= 9)
                v = qd_unif(v);
            max_d = std::max(max_d, std::abs(rnd(l, g) - v));
        }
        QD_DIAG("%llu,%.1f,%.3e\n", (unsigned long long)g, lambda_g[g], max_d);
        INFO("g " << g);
        CHECK(max_d < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
    }

    // Determinism: a repeated call gives identical results
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd2);
    CHECK(qd_equal(rnd, rnd2));
}

TEST_CASE("gbsm_draw - lsf: 3D mode uses the z coordinate")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 22);
    const arma::mat rx = rx7(), tx = arma::vec{10.0, 20.0, 25.0};

    arma::mat rnd;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb256", {}, {}, false, false);
    REQUIRE(rnd.n_rows == 7);
    REQUIRE(rnd.n_cols == 11);
    for (arma::uword g = 0; g < 11; ++g)
    {
        const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
        for (arma::uword l = 0; l < 7; ++l)
        {
            double v = qd_ref_sos(F, phase.slice(0).colptr(g), rx(0, l), rx(1, l), rx(2, l));
            if (g >= 9)
                v = qd_unif(v);
            INFO("g " << g << ", link " << l);
            CHECK(std::abs(rnd(l, g) - v) < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
        }
    }
}

TEST_CASE("gbsm_draw - lsf: other presets")
{
    const auto sp = scen_distinct();
    const arma::mat rx = rx7(), tx = arma::vec{10.0, 20.0, 25.0};
    const char *presets[2] = {"Exp128", "Gauss1024"};
    const arma::uword n_coeff[2] = {128, 1024};
    const double tol[2] = {1.5e-5, 4e-5}; // 2e-5 · sqrt(n_coeff / 256), AVX2 kernel tolerance
    for (int i = 0; i < 2; ++i)
    {
        const arma::cube phase = make_phase(n_coeff[i], 23);
        arma::mat rnd;
        quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, presets[i]);
        REQUIRE(rnd.n_rows == 7);
        const arma::mat F = qd_preset(presets[i], lambda_g[0]).F;
        for (arma::uword l = 0; l < 7; ++l)
        {
            INFO("preset " << presets[i] << ", link " << l);
            CHECK(std::abs(rnd(l, 0) - qd_ref_sos(F, phase.slice(0).colptr(0), rx(0, l), rx(1, l), 0.0)) < tol[i]);
        }
    }
}

TEST_CASE("gbsm_draw - lsf: 2D mode values do not depend on z")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 24);
    arma::mat rx = rx7(), tx = rx7() + 15.0;

    arma::mat a, b, c, d;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, a);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, c, -1, "Comb256", {}, {}, true);
    rx.row(2) += 17.0, tx.row(2) -= 3.0;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, b);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, d, -1, "Comb256", {}, {}, true);
    CHECK(qd_equal(a, b));
    CHECK(qd_equal(c, d)); // dual mobility as well
}

TEST_CASE("gbsm_draw - lsf: single mobility ignores tx_pos")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 25);
    const arma::mat rx = rx7(), tx1 = arma::vec{10.0, 20.0, 25.0}, tx2 = rx7() * 0.5 - 30.0;
    arma::mat a, b;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx1, rx, a);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx2, rx, b);
    CHECK(qd_equal(a, b));
}

TEST_CASE("gbsm_draw - lsf: a link does not depend on the other links")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 26);
    const arma::mat rx = rx7(), tx = rx7() * 0.5 - 30.0;
    const arma::u32_vec fl = {0, 3, 3, 259, 0, 1, 259};

    for (int dual = 0; dual < 2; ++dual)
    {
        arma::mat all, one;
        quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, all, -1, "Comb256", fl, {}, dual == 1);
        for (arma::uword l = 0; l < 7; ++l)
        {
            const arma::mat r = rx.col(l), t = tx.col(l);
            const arma::u32_vec f = {fl(l)};
            quadriga_lib::gbsm_draw(sp, "lsf", phase, t, r, one, -1, "Comb256", f, {}, dual == 1);
            INFO("dual " << dual << ", link " << l);
            CHECK(qd_equal(one, all.row(l)));
        }
    }
}

TEST_CASE("gbsm_draw - lsf: dual mobility values follow the two-device formulas")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 27);
    // 6 links; link 2 has Tx = Rx, link 4 a distance of more than 5 decorrelation distances (> 100 m)
    const arma::mat rx = arma::mat({{10.0, -20.0, 1.5}, {-55.0, 30.0, 3.0}, {100.0, 100.0, 1.5}, {-5.0, -5.0, 12.0}, {0.0, 0.0, 1.5}, {33.0, 71.0, 2.0}}).t();
    const arma::mat tx = arma::mat({{12.0, -15.0, 25.0}, {-40.0, 10.0, 30.0}, {100.0, 100.0, 1.5}, {-25.0, 3.0, 12.0}, {150.0, -80.0, 10.0}, {30.0, 80.0, 1.5}}).t();

    arma::mat rnd;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb256", {}, {}, true);
    REQUIRE(rnd.n_rows == 6);
    REQUIRE(rnd.n_cols == 11);

    // off: floor offsets of the link, added to all phases (both devices on the link's floor)
    const std::vector<arma::uword> p = qd_perm(256);
    auto oracle = [&](const arma::mat &R, const arma::mat &T, arma::uword l, arma::uword g, const arma::vec *off = nullptr)
    {
        const arma::uword m = mirror_g[g], o = std::min(g, m);
        const arma::mat F = qd_preset("Comb256", lambda_g[o]).F;
        const double r[3] = {R(0, l), R(1, l), 0.0}, t[3] = {T(0, l), T(1, l), 0.0};
        const double scale = lambda_g[o] / lambda_g[mirror_g[o]];
        double x = (o == g) ? qd_ref_dual(F, phase.slice(0).colptr(o), r, t, p, scale, m == g, off)
                            : qd_ref_dual(F, phase.slice(0).colptr(o), t, r, p, scale, false, off);
        return g >= 9 ? qd_unif(x) : x;
    };

    QD_DIAG("\n# diag: gbsm_draw lsf dual mobility against the two-device formulas\n");
    QD_DIAG("link,g,value,oracle,diff\n");
    for (arma::uword l = 0; l < 6; ++l)
        for (arma::uword g = 0; g < 11; ++g)
        {
            double v = oracle(rx, tx, l, g);
            QD_DIAG("%llu,%llu,%.8f,%.8f,%.3e\n", (unsigned long long)l, (unsigned long long)g, rnd(l, g), v, rnd(l, g) - v);
            INFO("link " << l << ", g " << g);
            CHECK(std::abs(rnd(l, g) - v) < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
        }

    // One Tx column is used for every link
    const arma::mat tx1 = tx.col(4), tx_b = arma::repmat(tx1, 1, 6);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx1, rx, rnd, -1, "Comb256", {}, {}, true);
    REQUIRE(rnd.n_rows == 6);
    for (arma::uword l = 0; l < 6; ++l)
        for (arma::uword g = 0; g < 11; ++g)
        {
            INFO("Tx broadcast, link " << l << ", g " << g);
            CHECK(std::abs(rnd(l, g) - oracle(rx, tx_b, l, g)) < (g >= 9 ? 1e-5 : 2e-5));
        }

    // Floors 3 (k = 3, b = 0) and 259 (k = 3, b = 1) mixed with floor 0: the offsets go to both sides
    const arma::u32_vec fl = {3, 259, 0, 259, 3, 0};
    const arma::vec off3 = qd_floor_offset(256, 3), off259 = qd_floor_offset(256, 259);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb256", fl, {}, true);
    REQUIRE(rnd.n_rows == 6);
    for (arma::uword l = 0; l < 6; ++l)
        for (arma::uword g = 0; g < 11; ++g)
        {
            const arma::vec *off = fl(l) == 3 ? &off3 : (fl(l) == 259 ? &off259 : nullptr);
            INFO("floor " << fl(l) << ", link " << l << ", g " << g);
            CHECK(std::abs(rnd(l, g) - oracle(rx, tx, l, g, off)) < (g >= 9 ? 1e-5 : 2e-5));
        }
}

TEST_CASE("gbsm_draw - lsf: dual mobility keeps unit variance and correlates the pairs as one field with Tx and Rx exchanged")
{
    auto sp = scen_distinct();
    sp["AS_D_lambda"] = 10.0, sp["AS_A_lambda"] = 15.0, sp["ES_D_lambda"] = 10.0, sp["ES_A_lambda"] = 20.0;
    const arma::cube phase = make_phase(256, 28);
    const arma::mat rx = qd_grid(100, 100, 50.0, 1.5);
    const arma::mat F10 = qd_preset("Comb256", 10.0).F;
    const std::vector<arma::uword> p = qd_perm(256);
    // Correlation of a pair member with its partner at the same link, Tx = Rx + (D, 0, 0): the owner field at (R, T)
    // and at (T, R), (1/N) · sum_n cos( 2·pi · (f_n - b·f_p(n))·(R - T) ) with b = lambda_o / lambda_m
    auto pair_corr = [&](double D, double b)
    {
        double c = 0.0;
        for (arma::uword n = 0; n < 256; ++n)
            c += std::cos(2.0 * pi * (F10(n, 0) - b * F10(p[n], 0)) * D) / 256.0;
        return c;
    };

    // Variance: 5 standard errors of the sample variance, sqrt(2/n) = 0.014 -> 0.07
    // Correlation: 5 standard errors (<= 0.01) and the spread of one realisation of the phases (0.03) -> 0.08
    QD_DIAG("\n# diag: gbsm_draw lsf dual mobility, 10^4 links, Tx = Rx + (D, 0, 0)\n");
    QD_DIAG("D,var3,var4,var5,var6,corr34,ref34,corr56,ref56\n");
    const double Ds[3] = {5.0, 10.0, 20.0};
    for (double D : Ds)
    {
        arma::mat tx = rx;
        tx.row(0) += D;
        arma::mat rnd;
        quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb256", {}, {}, true);
        REQUIRE(rnd.n_rows == rx.n_cols);
        double v[4] = {arma::var(rnd.col(3)), arma::var(rnd.col(4)), arma::var(rnd.col(5)), arma::var(rnd.col(6))};
        double c34 = qd_corr(rnd.col(3), rnd.col(4)), c56 = qd_corr(rnd.col(5), rnd.col(6));
        double r34 = pair_corr(D, 10.0 / 15.0), r56 = pair_corr(D, 10.0 / 20.0);
        QD_DIAG("%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", D, v[0], v[1], v[2], v[3], c34, r34, c56, r56);
        INFO("D " << D);
        for (int i = 0; i < 4; ++i)
            CHECK(std::abs(v[i] - 1.0) < 0.07);
        CHECK(std::abs(c34 - r34) < 0.08);
        CHECK(std::abs(c56 - r56) < 0.08);
    }

    // D = 0: both pair members are the same sum at the same positions
    arma::mat rnd;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, rx, rx, rnd, -1, "Comb256", {}, {}, true);
    CHECK(arma::all(rnd.col(3) == rnd.col(4)));
    CHECK(arma::all(rnd.col(5) == rnd.col(6)));
}

TEST_CASE("gbsm_draw - lsf: swapping Tx and Rx swaps ASD with ASA and ESD with ESA")
{
    const auto sp = scen_distinct(); // unequal distances of ASD/ASA and ESD/ESA
    arma::mat A = rx7(), B = rx7() * 0.7 + 20.0;
    B.row(2).fill(25.0);

    arma::cube ph_bias, ph_raw;
    ph_bias = quadriga_lib::gbsm_init_phases(sp, "lsf", 29, -1, "Comb256", B, A, true); // computed once from the original positions
    ph_raw = quadriga_lib::gbsm_init_phases(sp, "lsf", 29, -1, "Comb256", B, A, true, false);

    const arma::cube *phases[2] = {&ph_raw, &ph_bias};
    for (int i = 0; i < 2; ++i)
    {
        arma::mat ab, ba;
        quadriga_lib::gbsm_draw(sp, "lsf", *phases[i], B, A, ab, -1, "Comb256", {}, {}, true); // tx = B, rx = A
        quadriga_lib::gbsm_draw(sp, "lsf", *phases[i], A, B, ba, -1, "Comb256", {}, {}, true); // tx = A, rx = B
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

TEST_CASE("gbsm_draw - lsf: floor 0 equals no floor ID and equal IDs share one field")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 30);
    const arma::mat rx = rx7(), tx = arma::vec{10.0, 20.0, 25.0};

    arma::mat a, b;
    const arma::u32_vec fl0(7, arma::fill::zeros);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, a);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, b, -1, "Comb256", fl0);
    CHECK(qd_equal(a, b));

    // Floors 3 (k = 3, b = 0) and 259 (k = 3, b = 1) mixed with floor 0 in one call
    const arma::u32_vec fl = {3, 259, 3, 0, 259, 3, 259};
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, b, -1, "Comb256", fl);
    REQUIRE(b.n_rows == 7);
    const arma::vec off3 = qd_floor_offset(256, 3), off259 = qd_floor_offset(256, 259), off0(256, arma::fill::zeros);
    QD_DIAG("\n# diag: gbsm_draw lsf floors against ref_sos with typed offsets\n");
    QD_DIAG("link,floor,g,value,oracle,diff\n");
    for (arma::uword l = 0; l < 7; ++l)
        for (arma::uword g = 0; g < 11; ++g)
        {
            const arma::vec *off = fl(l) == 3 ? &off3 : (fl(l) == 259 ? &off259 : &off0);
            const arma::mat F = qd_preset("Comb256", lambda_g[g]).F;
            double v = qd_ref_sos(F, phase.slice(0).colptr(g), rx(0, l), rx(1, l), 0.0, off);
            if (g >= 9)
                v = qd_unif(v);
            QD_DIAG("%llu,%llu,%llu,%.8f,%.8f,%.3e\n", (unsigned long long)l, (unsigned long long)fl(l), (unsigned long long)g, b(l, g), v, b(l, g) - v);
            INFO("link " << l << ", floor " << fl(l) << ", g " << g);
            CHECK(std::abs(b(l, g) - v) < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
        }
}

TEST_CASE("gbsm_draw - lsf: fields of different floors are uncorrelated at equal positions")
{
    const auto sp = scen_equal(10.0);
    const arma::cube phase = make_phase(256, 31);
    const arma::mat g0 = qd_grid(100, 100, 50.0, 1.5); // n = 10^4, spacing 5 decorrelation distances (ACF < 0.02)
    const arma::uword n = g0.n_cols;
    const arma::uword floors[5] = {0, 1, 5, 255, 256};

    arma::mat rx(3, 5 * n);
    arma::u32_vec fl(5 * n);
    for (arma::uword f = 0; f < 5; ++f)
        rx.cols(f * n, f * n + n - 1) = g0, fl.subvec(f * n, f * n + n - 1).fill(floors[f]);
    const arma::mat tx = arma::vec{0.0, 0.0, 25.0};

    arma::mat rnd;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb256", fl);
    REQUIRE(rnd.n_rows == 5 * n);

    // Pairs (index into 'floors') and bounds: within a block of N = 256 floor IDs the correlation at zero
    // horizontal lag vanishes, 0.05 = 5 standard errors of 1/sqrt(n); floors 0 and 256 (different blocks) have
    // a random correlation with standard deviation 1/sqrt(2N) = 0.044, bound 0.25
    const int pairs[4][2] = {{0, 1}, {1, 2}, {0, 3}, {0, 4}};
    const double bound[4] = {0.05, 0.05, 0.05, 0.25};
    QD_DIAG("\n# diag: gbsm_draw lsf correlation between floors at equal positions\n");
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

TEST_CASE("gbsm_draw - lsf: marginal distributions")
{
    const auto sp = scen_equal(10.0);
    const arma::cube phase = make_phase(256, 32);
    const arma::mat rx = qd_grid(100, 100, 50.0, 1.5), tx = arma::vec{0.0, 0.0, 25.0};
    arma::mat rnd;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd);
    REQUIRE(rnd.n_cols == 11);

    // 5 standard errors with n = 10^4: mean 0.05, std 0.05 (0.01 each); Uniform mean 0.015 (0.003),
    // fraction below 0.1 within 0.015 (0.003)
    QD_DIAG("\n# diag: gbsm_draw lsf marginal statistics, 10^4 links\n");
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

TEST_CASE("gbsm_draw - lsf: the correlation follows the preset ACF")
{
    // Tolerance 0.1 >= 5 standard errors (0.05) + directional error of the SOS approximation (up to 0.045):
    // the sample correlation converges to the ACF of the sinusoids along the offset direction, not to the table
    // ACF (e.g. Comb256 at 10 m: 0.408 along x, 0.393 along y, against 0.368; Exp256 at 5 m along y: 0.572
    // against 0.607)
    const auto sp = scen_equal(10.0);
    const arma::mat base = qd_grid(100, 100, 50.0, 1.5), tx = arma::vec{0.0, 0.0, 25.0};
    const char *presets[2] = {"Comb256", "Exp256"};
    const double offsets[3] = {5.0, 10.0, 20.0};
    const double az[4] = {0.0, 0.25 * pi, 0.5 * pi, 0.75 * pi};

    QD_DIAG("\n# diag: gbsm_draw lsf sample correlation against the preset ACF (column 0 shown, all Normal columns checked)\n");
    QD_DIAG("preset,offset,azimuth_deg,corr_g0,acf\n");
    for (int p = 0; p < 2; ++p)
    {
        const arma::cube phase = make_phase(256, 33 + p);
        const qd_preset_tab t = qd_preset(presets[p], 10.0);
        arma::mat a;
        quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, base, a, -1, presets[p]);
        for (double d : offsets)
            for (double phi : az)
            {
                arma::mat moved = base, b;
                moved.row(0) += d * std::cos(phi), moved.row(1) += d * std::sin(phi);
                quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, moved, b, -1, presets[p]);
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

TEST_CASE("gbsm_draw - lsf: sizes, empty input and pre-sized output")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 35);
    const arma::mat rx0(3, 0), tx1 = arma::vec{0.0, 0.0, 25.0}, tx0(3, 0);
    arma::mat rnd(4, 4, arma::fill::ones);

    CHECK_NOTHROW(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx1, rx0, rnd));
    CHECK(rnd.n_rows == 0);
    CHECK(rnd.n_cols == 11);
    rnd.ones(4, 4);
    CHECK_NOTHROW(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx0, rx0, rnd, -1, "Comb256", {}, {}, true));
    CHECK(rnd.n_rows == 0);
    CHECK(rnd.n_cols == 11);

    // Pre-sized output with stale content: written in place (same memory) and completely, single and dual mobility
    const arma::mat rx = rx7(), tx = rx7() + 5.0;
    const arma::u32_vec fl = {0, 3, 3, 259, 0, 1, 259};
    arma::mat ref, ref1;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, ref1, -1, "Comb256", fl);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, ref, -1, "Comb256", {}, {}, true);
    arma::mat pre(7, 11);
    pre.fill(99.0);
    const double *mem = pre.memptr();
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, pre, -1, "Comb256", fl);
    CHECK(pre.memptr() == mem);
    CHECK(qd_equal(pre, ref1));
    pre.fill(99.0);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, pre, -1, "Comb256", {}, {}, true);
    CHECK(pre.memptr() == mem);
    CHECK(qd_equal(pre, ref));

    // Wrong size: resized
    arma::mat wrong(3, 3, arma::fill::ones);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, wrong, -1, "Comb256", {}, {}, true);
    CHECK(qd_equal(wrong, ref));
}

TEST_CASE("gbsm_draw - lsf: input validation")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 36);
    const arma::mat rx = rx7().cols(0, 2), tx = arma::vec{0.0, 0.0, 25.0};
    const arma::u32_vec fl = {0, 1, 2};
    arma::mat rnd;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd));
        CHECK_NOTHROW(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb256", fl, {}, true));
        CHECK_NOTHROW(quadriga_lib::gbsm_draw({}, "lsf", phase, tx, rx, rnd)); // empty scenpar: all defaults
    }
    SECTION("phase")
    {
        const std::string msg = "Input 'phase' must have size [256, 11, 2] for preset 'Comb256'.";
        arma::cube bad(128, 11, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", bad, tx, rx, rnd), std::invalid_argument, Message(msg));
        bad.zeros(256, 10, 2);
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", bad, tx, rx, rnd), std::invalid_argument, Message(msg));
        bad.zeros(256, 11, 1);
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", bad, tx, rx, rnd), std::invalid_argument, Message(msg));
        bad = phase;
        bad(5, 3, 1) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", bad, tx, rx, rnd), std::invalid_argument,
                             Message("Input 'phase' must be finite."));
    }
    SECTION("outputs that alias inputs")
    {
        arma::mat r = rx, t = tx;
        arma::cube ph = phase;
        CHECK_THROWS_AS(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, r, r), std::invalid_argument); // rnd is 'rx_pos'
        CHECK_THROWS_AS(quadriga_lib::gbsm_draw(sp, "lsf", phase, t, rx, t), std::invalid_argument); // rnd is 'tx_pos'
        arma::mat on_phase(ph.memptr(), 3, 11, false, true);
        CHECK_THROWS_AS(quadriga_lib::gbsm_draw(sp, "lsf", ph, tx, rx, on_phase), std::invalid_argument);
        CHECK(arma::approx_equal(r, rx, "absdiff", 0.0));
        CHECK(arma::approx_equal(ph, phase, "absdiff", 0.0));
    }
    SECTION("preset")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb999"), std::invalid_argument,
                             Message("Input 'preset' is not a known preset."));
    }
    SECTION("rx_pos")
    {
        arma::mat bad(2, 3, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, bad, rnd), std::invalid_argument,
                             Message("Input 'rx_pos' must have 3 rows."));
    }
    SECTION("tx_pos")
    {
        arma::mat bad = tx;
        bad(1, 0) = INFINITY;
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", phase, bad, rx, rnd), std::invalid_argument,
                             Message("Input 'tx_pos' must be finite."));
        bad.zeros(3, 2);
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", phase, bad, rx, rnd), std::invalid_argument,
                             Message("Number of columns in 'tx_pos' must be 1 or match the number of columns in 'rx_pos'."));
    }
    SECTION("floor_id")
    {
        const arma::u32_vec bad = {0, 1};
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb256", bad), std::invalid_argument,
                             Message("Number of elements in 'floor_id' does not match number of columns in 'rx_pos'."));

        // 3D mode: the height separates the floors, floor IDs are rejected (also all zero); the output is untouched
        const std::string msg = "Input 'floor_id' cannot be used with 'horizontal = false': in 3D mode the height separates the floors.";
        const arma::u32_vec fl0 = {0, 0, 0};
        arma::mat keep(2, 2, arma::fill::ones);
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, keep, -1, "Comb256", fl, {},
                                                     false, false),
                             std::invalid_argument, Message(msg));
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, keep, -1, "Comb256", fl0, {},
                                                     true, false),
                             std::invalid_argument, Message(msg));
        CHECK(keep.n_rows == 2);
        CHECK(arma::all(arma::vectorise(keep) == 1.0));
        CHECK_NOTHROW(quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, rnd, -1, "Comb256", {}, {}, false,
                                              false));
    }
    SECTION("scenpar")
    {
        auto bad = sp;
        bad["XPR_lambda"] = -1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::gbsm_draw(bad, "lsf", phase, tx, rx, rnd), std::invalid_argument,
                             Message("Value of 'XPR_lambda' must be finite and >= 0."));
    }
}

TEST_CASE("gbsm_draw - lsf: in dual mobility a link decorrelates when one end moves away")
{
    // 2025 independent links on a 3 km grid (decorrelation distances 11 to 20 m): Tx at a grid point, Rx 100 m away.
    // Moving the Rx or the Tx by 1000 m must decorrelate every Normal column: all links have the same displacement, so
    // the residual of the sinusoids is one fixed value of rms 1/sqrt(2N) = 0.044 per column (18 values up to about 4
    // times that), plus a sampling error of 0.022, so |corr| < 0.25; the sum of two one-device fields keeps 0.5
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 5);
    const arma::uword n = 45, n_link = n * n;
    arma::mat tx(3, n_link), rx(3, n_link);
    for (arma::uword i = 0; i < n_link; ++i)
        tx(0, i) = 3000.0 * (double)(i % n), tx(1, i) = 3000.0 * (double)(i / n), tx(2, i) = 25.0;
    rx = tx, rx.row(0) += 100.0, rx.row(2).fill(1.5);
    arma::mat rx_m = rx, tx_m = tx;
    rx_m.row(0) += 1000.0, tx_m.row(1) += 1000.0;

    arma::mat v0, v_rx, v_tx;
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx, v0, -1, "Comb256", {}, {}, true);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx, rx_m, v_rx, -1, "Comb256", {}, {}, true);
    quadriga_lib::gbsm_draw(sp, "lsf", phase, tx_m, rx, v_tx, -1, "Comb256", {}, {}, true);
    QD_DIAG("\n# diag: gbsm_draw lsf dual mobility, correlation after a 1000 m move\ncolumn,rx_moves,tx_moves\n");
    for (arma::uword g = 0; g < 9; ++g)
    {
        const double c_rx = arma::as_scalar(arma::cor(v0.col(g), v_rx.col(g)));
        const double c_tx = arma::as_scalar(arma::cor(v0.col(g), v_tx.col(g)));
        QD_DIAG("%llu,%.3f,%.3f\n", (unsigned long long)g, c_rx, c_tx);
        INFO("column " << g);
        CHECK(std::abs(c_rx) < 0.25);
        CHECK(std::abs(c_tx) < 0.25);
    }
}
