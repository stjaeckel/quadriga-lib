// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "lsf_draw*"

// Verification suite for quadriga_lib::lsf_draw.
//
// Pins down: the values of the 11 generators in single mobility (each generator with its own decorrelation
// distance, 2D and 3D mode, three presets), the Uniform map of columns 9 and 10, the two-device formulas of dual
// mobility (self-mirrored generators on one map, the pairs ASD/ASA and ESD/ESA on two maps without further mixing),
// unit variance and the pair correlation in dual mobility, the swap reciprocity, the floor phase offsets, the
// statistics (marginals, ACF, zero correlation of different floors at equal positions), the independence of z in
// 2D mode, of tx_pos in single mobility and of the other links, sizes, empty input, pre-sized outputs and the
// input validation.
// Left to test_lsf_init_phases.cpp: the phases themselves (raw, mirrored, bias removal).
// Oracles, independent of the function:
//  - ref_sos(F, phi, x) = sqrt(2/N) · sum_n cos( 2·pi · F(n,:)·x + phi(n) ), typed here, with the frequencies of
//    sos_preset at the decorrelation distance of each generator
//  - dual mobility: (ref_sos(F_g, P0_g, R) + ref_sos(F_g, P0_g, T)) / sqrt(2 + 2·A_g(D)) for the self-mirrored
//    generators, (ref_sos(F_g, P0_g, R) + ref_sos(F_m, P0_m, T)) / sqrt(2) for the pairs, A_g the ACF table of
//    sos_preset interpolated linearly here, D the Tx-Rx distance (horizontal in 2D mode)
//  - the Uniform map 0.5·erfc(-v/sqrt(2))
//  - floor offsets 2·pi·k·n/N + psi_b(n), k = q mod N, b = q div N, psi_0 = 0, psi_b(n) = 2·pi·u_n with u_n from
//    the raw outputs x_n of std::mt19937_64 seeded with b: u_n = (x_n >> 11)·2^-53
//  - statistics: Normal(0,1), Uniform(0,1), the ACF table of sos_preset
// Tolerances of lsf_draw values: lsf_draw evaluates the sinusoids with the AVX2 kernel where available (sine and
// cosine in single precision, accumulation in double, deviation below 4e-7·sqrt(2·n_coeff) = 9e-6 for 256
// sinusoids). Normal columns 2e-5 for 256 sinusoids, scaled with sqrt(n_coeff/256) for other presets, Uniform
// columns 1e-5 (the slope of the Uniform map is at most 0.4).
// Conventions: positions are rows {x, y, z} in meters; most cases use distinct decorrelation distances (DS 11,
// KF 12, SF 13, ASD 14, ASA 15, ESD 16, ESA 17, XPR 18, absTOA 19, SC 20 m; generator 10 uses SF_lambda); the phase
// cubes are raw phases of sos_init_phases with the Tx side mirrored (ASD <-> ASA, ESD <-> ESA).

#include <catch2/catch_test_macros.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_sos.hpp"

#include <cmath>
#include <cstdint>
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

    // Phase cube: raw phases of sos_init_phases in slice 0, mirror in slice 1
    arma::cube make_phase(arma::uword n_coeff, arma::sword seed)
    {
        arma::mat P0;
        quadriga_lib::sos_init_phases<double>(n_coeff, 11, P0, seed);
        arma::cube phase(n_coeff, 11, 2);
        phase.slice(0) = P0;
        for (arma::uword g = 0; g < 11; ++g)
            phase.slice(1).col(g) = P0.col(mirror_g[g]);
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

    // Test-side linear interpolation of an ACF table, last value beyond the last sample point
    double acf_at(const arma::vec &acf, const arma::vec &dist, double D)
    {
        const arma::uword n = acf.n_elem;
        if (D >= dist(n - 1))
            return acf(n - 1);
        arma::uword i = 0;
        while (dist(i + 1) <= D)
            ++i;
        double w = (D - dist(i)) / (dist(i + 1) - dist(i));
        return (1.0 - w) * acf(i) + w * acf(i + 1);
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

    // Regular grid of nx · ny positions with the given spacing at height z
    arma::mat grid(arma::uword nx, arma::uword ny, double spacing, double z)
    {
        arma::mat p(nx * ny, 3);
        for (arma::uword i = 0; i < nx; ++i)
            for (arma::uword j = 0; j < ny; ++j)
                p(i * ny + j, 0) = spacing * (double)i, p(i * ny + j, 1) = spacing * (double)j, p(i * ny + j, 2) = z;
        return p;
    }

    double corr(const arma::vec &a, const arma::vec &b)
    {
        double ma = arma::mean(a), mb = arma::mean(b), sab = 0.0, saa = 0.0, sbb = 0.0;
        for (arma::uword i = 0; i < a.n_elem; ++i)
            sab += (a(i) - ma) * (b(i) - mb), saa += (a(i) - ma) * (a(i) - ma), sbb += (b(i) - mb) * (b(i) - mb);
        return sab / std::sqrt(saa * sbb);
    }

    double var(const arma::vec &a)
    {
        double m = arma::mean(a), s = 0.0;
        for (arma::uword i = 0; i < a.n_elem; ++i)
            s += (a(i) - m) * (a(i) - m);
        return s / (double)(a.n_elem - 1);
    }

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

    bool equal(const arma::mat &a, const arma::mat &b)
    {
        return a.n_rows == b.n_rows && a.n_cols == b.n_cols && arma::all(arma::vectorise(a == b));
    }

    // 7 links with distinct coordinates, negative values, up to 300 m, z != 0
    arma::mat rx7()
    {
        return {{12.5, -40.0, 1.5}, {-120.0, 33.0, 4.5}, {250.0, 300.0, 22.0}, {-7.25, -299.0, 1.8},
                {88.0, 0.5, 10.0}, {0.0, 160.0, 31.5}, {-210.0, -95.5, 7.0}};
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
        const arma::mat F = get_preset("Comb256", lambda_g[g]).F;
        double max_d = 0.0;
        for (arma::uword l = 0; l < 7; ++l)
        {
            double v = ref_sos(F, phase.slice(0).colptr(g), rx(l, 0), rx(l, 1), 0.0);
            if (g >= 9)
                v = unif(v);
            max_d = std::max(max_d, std::abs(rnd(l, g) - v));
        }
        QD_DIAG("%llu,%.1f,%.3e\n", (unsigned long long)g, lambda_g[g], max_d);
        INFO("g " << g);
        CHECK(max_d < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
    }

    // Determinism: a repeated call gives identical results
    quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd2);
    CHECK(equal(rnd, rnd2));
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
        const arma::mat F = get_preset("Comb256", lambda_g[g]).F;
        for (arma::uword l = 0; l < 7; ++l)
        {
            double v = ref_sos(F, phase.slice(0).colptr(g), rx(l, 0), rx(l, 1), rx(l, 2));
            if (g >= 9)
                v = unif(v);
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
        const arma::mat F = get_preset(presets[i], lambda_g[0]).F;
        for (arma::uword l = 0; l < 7; ++l)
        {
            INFO("preset " << presets[i] << ", link " << l);
            CHECK(std::abs(rnd(l, 0) - ref_sos(F, phase.slice(0).colptr(0), rx(l, 0), rx(l, 1), 0.0)) < tol[i]);
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
    CHECK(equal(a, b));
    CHECK(equal(c, d)); // dual mobility as well
}

TEST_CASE("lsf_draw - single mobility ignores tx_pos")
{
    const auto sp = scen_distinct();
    const arma::cube phase = make_phase(256, 25);
    const arma::mat rx = rx7(), tx1 = {{10.0, 20.0, 25.0}}, tx2 = rx7() * 0.5 - 30.0;
    arma::mat a, b;
    quadriga_lib::lsf_draw(sp, phase, tx1, rx, a);
    quadriga_lib::lsf_draw(sp, phase, tx2, rx, b);
    CHECK(equal(a, b));
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
            CHECK(equal(one, all.row(l)));
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

    auto oracle = [&](const arma::mat &R, const arma::mat &T, arma::uword l, arma::uword g)
    {
        const arma::uword m = mirror_g[g];
        const preset_tab tg = get_preset("Comb256", lambda_g[g]), tm = get_preset("Comb256", lambda_g[m]);
        double a = ref_sos(tg.F, phase.slice(0).colptr(g), R(l, 0), R(l, 1), 0.0);
        double b = ref_sos(tm.F, phase.slice(0).colptr(m), T(l, 0), T(l, 1), 0.0);
        double D = std::sqrt((R(l, 0) - T(l, 0)) * (R(l, 0) - T(l, 0)) + (R(l, 1) - T(l, 1)) * (R(l, 1) - T(l, 1)));
        double x = (m == g) ? (a + b) / std::sqrt(2.0 + 2.0 * acf_at(tg.acf, tg.dist, D)) : (a + b) / std::sqrt(2.0);
        return g >= 9 ? unif(x) : x;
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
}

TEST_CASE("lsf_draw - dual mobility keeps unit variance and correlates the pairs by the ACF at the Tx-Rx distance")
{
    auto sp = scen_distinct();
    sp["AS_D_lambda"] = 10.0, sp["AS_A_lambda"] = 15.0, sp["ES_D_lambda"] = 10.0, sp["ES_A_lambda"] = 20.0;
    const arma::cube phase = make_phase(256, 28);
    const arma::mat rx = grid(100, 100, 50.0, 1.5);
    const preset_tab t10 = get_preset("Comb256", 10.0), t15 = get_preset("Comb256", 15.0), t20 = get_preset("Comb256", 20.0);

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
        double v[4] = {var(rnd.col(3)), var(rnd.col(4)), var(rnd.col(5)), var(rnd.col(6))};
        double c34 = corr(rnd.col(3), rnd.col(4)), c56 = corr(rnd.col(5), rnd.col(6));
        double r34 = 0.5 * (acf_at(t10.acf, t10.dist, D) + acf_at(t15.acf, t15.dist, D));
        double r56 = 0.5 * (acf_at(t10.acf, t10.dist, D) + acf_at(t20.acf, t20.dist, D));
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
    CHECK(equal(a, b));

    // Floors 3 (k = 3, b = 0) and 259 (k = 3, b = 1) mixed with floor 0 in one call
    const arma::uvec fl = {3, 259, 3, 0, 259, 3, 259};
    quadriga_lib::lsf_draw(sp, phase, tx, rx, b, &fl);
    REQUIRE(b.n_rows == 7);
    const arma::vec off3 = floor_offset(256, 3), off259 = floor_offset(256, 259), off0(256, arma::fill::zeros);
    QD_DIAG("\n# diag: lsf_draw floors against ref_sos with typed offsets\n");
    QD_DIAG("link,floor,g,value,oracle,diff\n");
    for (arma::uword l = 0; l < 7; ++l)
        for (arma::uword g = 0; g < 11; ++g)
        {
            const arma::vec *off = fl(l) == 3 ? &off3 : (fl(l) == 259 ? &off259 : &off0);
            const arma::mat F = get_preset("Comb256", lambda_g[g]).F;
            double v = ref_sos(F, phase.slice(0).colptr(g), rx(l, 0), rx(l, 1), 0.0, off);
            if (g >= 9)
                v = unif(v);
            QD_DIAG("%llu,%llu,%llu,%.8f,%.8f,%.3e\n", (unsigned long long)l, (unsigned long long)fl(l), (unsigned long long)g, b(l, g), v, b(l, g) - v);
            INFO("link " << l << ", floor " << fl(l) << ", g " << g);
            CHECK(std::abs(b(l, g) - v) < (g >= 9 ? 1e-5 : 2e-5)); // AVX2 kernel tolerance, see header
        }
}

TEST_CASE("lsf_draw - fields of different floors are uncorrelated at equal positions")
{
    const auto sp = scen_equal(10.0);
    const arma::cube phase = make_phase(256, 31);
    const arma::mat g0 = grid(100, 100, 50.0, 1.5); // n = 10^4, spacing 5 decorrelation distances (ACF < 0.02)
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
            double c = corr(rnd.col(g).rows(fa * n, fa * n + n - 1), rnd.col(g).rows(fb * n, fb * n + n - 1));
            QD_DIAG("%llu,%llu,%llu,%.4f\n", (unsigned long long)floors[fa], (unsigned long long)floors[fb], (unsigned long long)g, c);
            INFO("floors " << floors[fa] << ", " << floors[fb] << ", g " << g);
            CHECK(std::abs(c) < bound[p]);
        }
}

TEST_CASE("lsf_draw - marginal distributions")
{
    const auto sp = scen_equal(10.0);
    const arma::cube phase = make_phase(256, 32);
    const arma::mat rx = grid(100, 100, 50.0, 1.5), tx = {{0.0, 0.0, 25.0}};
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
        double m = arma::mean(c), s = std::sqrt(var(c)), frac = (double)arma::accu(c < 0.1) / (double)c.n_elem;
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
    const arma::mat base = grid(100, 100, 50.0, 1.5), tx = {{0.0, 0.0, 25.0}};
    const char *presets[2] = {"Comb256", "Exp256"};
    const double offsets[3] = {5.0, 10.0, 20.0};
    const double az[4] = {0.0, 0.25 * pi, 0.5 * pi, 0.75 * pi};

    QD_DIAG("\n# diag: lsf_draw sample correlation against the preset ACF (column 0 shown, all Normal columns checked)\n");
    QD_DIAG("preset,offset,azimuth_deg,corr_g0,acf\n");
    for (int p = 0; p < 2; ++p)
    {
        const arma::cube phase = make_phase(256, 33 + p);
        const preset_tab t = get_preset(presets[p], 10.0);
        arma::mat a;
        quadriga_lib::lsf_draw(sp, phase, tx, base, a, nullptr, false, presets[p]);
        for (double d : offsets)
            for (double phi : az)
            {
                arma::mat moved = base, b;
                moved.col(0) += d * std::cos(phi), moved.col(1) += d * std::sin(phi);
                quadriga_lib::lsf_draw(sp, phase, tx, moved, b, nullptr, false, presets[p]);
                double ref = acf_at(t.acf, t.dist, d);
                QD_DIAG("%s,%.0f,%.0f,%.4f,%.4f\n", presets[p], d, phi * 180.0 / pi, corr(a.col(0), b.col(0)), ref);
                for (arma::uword g = 0; g < 9; ++g)
                {
                    INFO(presets[p] << ", offset " << d << ", azimuth " << phi << ", g " << g);
                    CHECK(std::abs(corr(a.col(g), b.col(g)) - ref) < 0.1);
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
    CHECK(equal(pre, ref1));
    pre.fill(99.0);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, pre, nullptr, true);
    CHECK(pre.memptr() == mem);
    CHECK(equal(pre, ref));

    // Wrong size: resized
    arma::mat wrong(3, 3, arma::fill::ones);
    quadriga_lib::lsf_draw(sp, phase, tx, rx, wrong, nullptr, true);
    CHECK(equal(wrong, ref));
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
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(sp, bad, tx, rx, rnd); }) == msg);
        bad.zeros(256, 10, 2);
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(sp, bad, tx, rx, rnd); }) == msg);
        bad.zeros(256, 11, 1);
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(sp, bad, tx, rx, rnd); }) == msg);
        bad = phase;
        bad(5, 3, 1) = NAN;
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(sp, bad, tx, rx, rnd); }) == "Input 'phase' must be finite.");
    }
    SECTION("preset")
    {
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, nullptr, false, "Comb999"); }) ==
              "Input 'preset' is not a known preset.");
    }
    SECTION("rx_pos")
    {
        arma::mat bad(3, 2, arma::fill::zeros);
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(sp, phase, tx, bad, rnd); }) == "Input 'rx_pos' must have 3 columns.");
    }
    SECTION("tx_pos")
    {
        arma::mat bad = tx;
        bad(0, 1) = INFINITY;
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(sp, phase, bad, rx, rnd); }) == "Input 'tx_pos' must be finite.");
        bad.zeros(2, 3);
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(sp, phase, bad, rx, rnd); }) ==
              "Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'.");
    }
    SECTION("floor_id")
    {
        const arma::uvec bad = {0, 1};
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(sp, phase, tx, rx, rnd, &bad); }) ==
              "Number of elements in 'floor_id' does not match number of rows in 'rx_pos'.");
    }
    SECTION("scenpar")
    {
        auto bad = sp;
        bad["XPR_lambda"] = -1.0;
        CHECK(err_msg([&]
                      { quadriga_lib::lsf_draw(bad, phase, tx, rx, rnd); }) == "Value of 'XPR_lambda' must be finite and >= 0.");
    }
}
