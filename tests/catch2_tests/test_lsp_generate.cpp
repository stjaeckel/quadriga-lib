// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "lsp_generate*"

// Verification suite for quadriga_lib::lsp_generate.
//
// Pins down: the LSP mean and standard deviation per link and frequency (every term of the formula, both limits at
// 0, LSP_fc_min, the ESD terms and floor, the V/S term of the delay spread), TR spot checks of four v19 files, the
// EoD offset, the transformation 10^(mu + sigma·x) of the eight LSPs, the root of the cross-correlation matrix
// (Cholesky in TR order, symmetric square root), the sample covariance of the log values, the clipping of the
// spreads, one random vector for all frequencies, the replacement of the SF std by 'sf_sigma', the absolute TOA,
// the ground permittivity, the optional outputs, sizes, pre-sized outputs and the input validation.
// Left to test_conf_files_v19.cpp: the key values of the 23 v19 files against TR 38.901 Tables 7.5-6 to 7.5-12.
// Oracles, independent of the function:
//  - the mean/std formula written out in ref_mu_sigma below
//  - TR 38.901 v19.5.0 Table 7.5-6 Parts 1, 3, 4, Tables 7.5-7, 7.5-8, 7.5-11, 7.5-12 and the SF standard deviation of
//    Table 7.4.1-1, typed; where the TR has "N/A" (KF of NLOS links) the convention of the files: mean -100 dB, std 0
//  - TR 38.901 clause 7.5 step 4 (Cholesky root in the order SF, K, DS, ASD, ASA, ZSD, ZSA; XPR appended last) with
//    a Cholesky factorization written in this file; the symmetric square root by its defining properties (X
//    symmetric, X·X = R, X positive definite: it is the unique SPD root)
//  - the absolute TOA rule 10^(mu + sigma·x), active for absTOA_mu > -30, bounded by 2·L/c with c = 299792458 m/s
//  - the ground permittivity model: linear interpolation between dry, medium and wet ground
// Random inputs of the statistical case: std::mt19937_64 with a fixed seed and a Box-Muller transform written here.
// Tolerances: 1e-12 for exact-by-construction double results (absolute for O(1) values, relative for linear LSPs);
// TR checks of rounded file values take their tolerance from the rounding bound (stated at the check).
// Conventions: positions are rows {x, y, z} in meters; TR reference geometry tx = (0, 0, hBS), rx = (d2D, 0, hUT);
// LSP order DS, KF, SF, ASD, ASA, ESD, ESA, XPR; x = lg(lsp) for DS, ASD, ASA, ESD, ESA, x = 10·lg(lsp) for KF,
// SF, XPR.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    typedef std::unordered_map<std::string, double> scen_t;

    const double pi = 3.141592653589793;
    const double c0 = 299792458.0;
    const char *lsp_name[8] = {"DS", "KF", "SF", "ASD", "ASA", "ESD", "ESA", "XPR"};
    const char *lsp_prefix[8] = {"DS", "KF", "SF", "AS_D", "AS_A", "ES_D", "ES_A", "XPR"};
    const bool lsp_db[8] = {false, true, true, false, false, false, false, true};

    // The 28 correlation keys and their (row, col) in the LSP order
    struct corr_key
    {
        const char *name;
        int r, c;
    };
    const corr_key corr_keys[28] = {
        {"ds_kf", 0, 1}, {"ds_sf", 0, 2}, {"asD_ds", 0, 3}, {"asA_ds", 0, 4}, {"esD_ds", 0, 5}, {"esA_ds", 0, 6},
        {"xpr_ds", 0, 7}, {"sf_kf", 1, 2}, {"asD_kf", 1, 3}, {"asA_kf", 1, 4}, {"esD_kf", 1, 5}, {"esA_kf", 1, 6},
        {"xpr_kf", 1, 7}, {"asD_sf", 2, 3}, {"asA_sf", 2, 4}, {"esD_sf", 2, 5}, {"esA_sf", 2, 6}, {"xpr_sf", 2, 7},
        {"asD_asA", 3, 4}, {"esD_asD", 3, 5}, {"esA_asD", 3, 6}, {"xpr_asd", 3, 7}, {"esD_asA", 4, 5},
        {"esA_asA", 4, 6}, {"xpr_asa", 4, 7}, {"esD_esA", 5, 6}, {"xpr_esd", 5, 7}, {"xpr_esa", 6, 7}};

    double getv(const scen_t &sp, const std::string &key, double def = 0.0)
    {
        auto it = sp.find(key);
        return it == sp.end() ? def : it->second;
    }

    scen_t conf(const std::string &name)
    {
        scen_t sp;
        quadriga_lib::conf_file_read("data/config/3GPP_38.901_v19_" + name + ".conf", sp);
        return sp;
    }

    // Oracle: LSP mean and std of one link at one frequency, written out
    //   d = max(d2D, 1e-4), h = max(hBS, 1e-4), a = atan(h/d), f' = max(fc/1e9, LSP_fc_min)
    //   mu    = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)
    //   sigma = max(0, max(0, P_sigma + P_delta·lg(P_omega + f')) + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))
    //   DS:  mu += lg(DS_VS_a·DS_VS + DS_VS_b) if DS_VS > 0
    //   ESD: mu += ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0),
    //        then max(mu, ES_D_mu_min)
    // Keys missing in 'sp' count as 0 (ES_D_mu_min as -Inf); SF has no mean keys
    void ref_mu_sigma(const scen_t &sp, double fc, const double *tx, const double *rx, double *mu, double *sg,
                      double *esd_raw = nullptr)
    {
        const double d = std::max(std::sqrt((rx[0] - tx[0]) * (rx[0] - tx[0]) + (rx[1] - tx[1]) * (rx[1] - tx[1])), 1e-4);
        const double h = std::max(tx[2], 1e-4), hut = rx[2];
        const double a = std::atan(h / d);
        const double fp = std::max(fc / 1e9, getv(sp, "LSP_fc_min"));
        for (int i = 0; i < 8; ++i)
        {
            const std::string p = lsp_prefix[i];
            const double lf = std::log10(getv(sp, p + "_omega") + fp);
            mu[i] = getv(sp, p + "_mu") + getv(sp, p + "_gamma") * lf + getv(sp, p + "_epsilon") * std::log10(d) +
                    getv(sp, p + "_zeta") * std::log10(h) + getv(sp, p + "_alpha") * std::log10(a);
            const double s = std::max(0.0, getv(sp, p + "_sigma") + getv(sp, p + "_delta") * lf);
            sg[i] = std::max(0.0, s + getv(sp, p + "_kappa") * std::log10(d) + getv(sp, p + "_tau") * std::log10(h) +
                                      getv(sp, p + "_beta") * std::log10(a));
        }
        if (getv(sp, "DS_VS") > 0.0)
            mu[0] += std::log10(getv(sp, "DS_VS_a") * getv(sp, "DS_VS") + getv(sp, "DS_VS_b"));
        mu[5] += getv(sp, "ES_D_mu_A") * d / 1000.0 + getv(sp, "ES_D_mu_B") * hut +
                 getv(sp, "ES_D_mu_C") * std::abs(hut - h) + getv(sp, "ES_D_mu_D") * std::max(hut - h, 0.0);
        if (esd_raw != nullptr)
            *esd_raw = mu[5];
        mu[5] = std::max(mu[5], getv(sp, "ES_D_mu_min", -INFINITY));
    }

    // x of LSP i from the linear value
    double lsp_to_x(double v, int i) { return lsp_db[i] ? 10.0 * std::log10(v) : std::log10(v); }

    // Matrix product written out (the tests link no BLAS)
    arma::mat mm(const arma::mat &A, const arma::mat &B)
    {
        arma::mat C(A.n_rows, B.n_cols, arma::fill::zeros);
        for (arma::uword j = 0; j < B.n_cols; ++j)
            for (arma::uword k = 0; k < A.n_cols; ++k)
                for (arma::uword i = 0; i < A.n_rows; ++i)
                    C(i, j) += A(i, k) * B(k, j);
        return C;
    }

    // Test-side Cholesky factorization (row by row) of a symmetric matrix; false if a pivot is not > 0
    bool test_chol(const arma::mat &A, arma::mat &L)
    {
        const arma::uword n = A.n_rows;
        L.zeros(n, n);
        for (arma::uword i = 0; i < n; ++i)
            for (arma::uword j = 0; j <= i; ++j)
            {
                double s = A(i, j);
                for (arma::uword k = 0; k < j; ++k)
                    s -= L(i, k) * L(j, k);
                if (i == j)
                {
                    if (!(s > 0.0))
                        return false;
                    L(i, i) = std::sqrt(s);
                }
                else
                    L(i, j) = s / L(j, j);
            }
        return true;
    }

    // Correlation matrix of the scenpar keys (missing keys 0)
    arma::mat corr_matrix(const scen_t &sp)
    {
        arma::mat R = arma::eye(8, 8);
        for (const auto &k : corr_keys)
            R(k.r, k.c) = R(k.c, k.r) = getv(sp, k.name);
        return R;
    }

    void set_corr(scen_t &sp, const arma::mat &R)
    {
        for (const auto &k : corr_keys)
            sp[k.name] = R(k.r, k.c);
    }

    // Custom positive definite correlation matrix with all 28 correlations non-zero and distinct:
    // C = B·B^T + diag(0.4 .. 1.1), normalized to unit diagonal
    arma::mat custom_corr()
    {
        arma::mat B(8, 3);
        for (arma::uword i = 0; i < 8; ++i)
            for (arma::uword j = 0; j < 3; ++j)
                B(i, j) = std::sin(1.3 * (double)i + 0.7 * (double)j + 0.4) + 0.1 * (double)j;
        arma::mat C = mm(B, B.t());
        for (arma::uword i = 0; i < 8; ++i)
            C(i, i) += 0.4 + 0.1 * (double)i;
        arma::mat R(8, 8);
        for (arma::uword i = 0; i < 8; ++i)
            for (arma::uword j = 0; j < 8; ++j)
                R(i, j) = (i == j) ? 1.0 : C(i, j) / std::sqrt(C(i, i) * C(j, j));
        return R;
    }

    // Mean 0, std 1 for all LSPs, no other terms (correlation keys added by the caller)
    scen_t unit_scen()
    {
        scen_t sp;
        for (int i = 0; i < 8; ++i)
        {
            if (i != 2)
                sp[std::string(lsp_prefix[i]) + "_mu"] = 0.0;
            sp[std::string(lsp_prefix[i]) + "_sigma"] = 1.0;
        }
        return sp;
    }

    // Uniform (0, 1] from the raw output of std::mt19937_64; Box-Muller pairs
    double unif01(std::mt19937_64 &g) { return 1.0 - (double)(g() >> 11) * (1.0 / 9007199254740992.0); }
    arma::mat box_muller(arma::uword n_row, arma::uword n_col, std::uint64_t seed)
    {
        std::mt19937_64 g(seed);
        arma::mat X(n_row, n_col);
        double *p = X.memptr();
        for (arma::uword i = 0; i < X.n_elem; i += 2)
        {
            const double r = std::sqrt(-2.0 * std::log(unif01(g))), t = 2.0 * pi * unif01(g);
            p[i] = r * std::cos(t);
            if (i + 1 < X.n_elem)
                p[i + 1] = r * std::sin(t);
        }
        return X;
    }

    // Root Q of the links with s = e_j (rows 0..7 of 'rnd'), read back from lsp with mu = 0, sigma = 1
    arma::mat root_from_lsp(const arma::cube &lsp)
    {
        arma::mat Q(8, 8);
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i)
                Q(i, j) = lsp_to_x(lsp(j, 0, i), i);
        return Q;
    }

    arma::mat unit_rnd()
    {
        arma::mat rnd(8, 11, arma::fill::zeros);
        for (arma::uword j = 0; j < 8; ++j)
            rnd(j, j) = 1.0;
        return rnd;
    }

    // All outputs of one call
    struct outputs
    {
        arma::cube lsp, mu, sg;
        arma::vec toa;
        arma::mat re, im, eod;
    };

    outputs run_all(const scen_t &sp, const arma::vec &fc, const arma::mat &tx, const arma::mat &rx, const arma::mat &rnd,
                    const arma::mat *sf = nullptr, bool chol = true, bool clip = true)
    {
        outputs o;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, o.lsp, sf, &o.toa, &o.re, &o.im, &o.eod, &o.mu, &o.sg, chol, clip);
        return o;
    }

    // Random values with columns 9 and 10 in (0, 1)
    arma::mat rnd_values(arma::uword n_link, std::uint64_t seed)
    {
        arma::mat rnd = box_muller(n_link, 11, seed);
        for (arma::uword l = 0; l < n_link; ++l)
        {
            rnd(l, 9) = 0.5 * std::erfc(-rnd(l, 9) / std::sqrt(2.0));
            rnd(l, 10) = 0.5 * std::erfc(-rnd(l, 10) / std::sqrt(2.0));
        }
        return rnd;
    }
}

TEST_CASE("lsp_generate - mean and std follow the formula with every term")
{
    // Every term non-zero and distinct; columns: mu, gamma, omega, epsilon, zeta, alpha, sigma, delta, kappa, tau, beta
    const double coef[8][11] = {
        {-7.1, -0.11, 0.3, 0.21, -0.13, 0.05, 0.31, 0.04, 0.012, -0.015, 0.021}, // DS
        {5.2, 1.3, 0.5, -1.7, 2.1, -0.9, 1.5, -1.2, 0.8, 0.3, -0.25},           // KF: inner limit at 28 GHz
        {0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 6.5, 1.1, -0.6, 0.4, 0.35},              // SF: no mean keys
        {1.05, -0.07, 1.0, 0.09, -0.04, 0.06, 0.28, 0.05, -0.1, 0.01, 0.02},    // ASD: outer limit at large d
        {1.62, -0.08, 1.2, -0.03, 0.07, -0.05, 0.22, 0.021, 0.03, -0.02, 0.015}, // ASA
        {0.92, -0.05, 0.8, 0.04, 0.06, 0.08, 0.33, -0.03, 0.025, 0.035, -0.01}, // ESD
        {1.21, -0.12, 0.6, 0.05, -0.08, 0.07, 0.19, 0.06, -0.02, 0.03, 0.04},   // ESA
        {8.5, 0.9, 0.4, -1.1, 0.6, 1.4, 3.3, -0.7, 0.5, -0.4, 0.45}};           // XPR
    const char *suffix[11] = {"_mu", "_gamma", "_omega", "_epsilon", "_zeta", "_alpha", "_sigma", "_delta", "_kappa", "_tau", "_beta"};

    scen_t sp;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 11; ++j)
            if (i != 2 || (j == 2 || j >= 6))
                sp[std::string(lsp_prefix[i]) + suffix[j]] = coef[i][j];
    sp["LSP_fc_min"] = 2.0;
    sp["ES_D_mu_A"] = -2.3;
    sp["ES_D_mu_B"] = -0.012;
    sp["ES_D_mu_C"] = 0.017;
    sp["ES_D_mu_D"] = 0.009;
    sp["ES_D_mu_min"] = 0.2;
    sp["DS_VS"] = 3.3;
    sp["DS_VS_a"] = 21.0;
    sp["DS_VS_b"] = 9.0;

    // Links: hUT above hBS; hUT below hBS; the ESD floor applies; d2D 5000 m where the ASD std is negative before
    // the outer limit
    const arma::mat tx = {{10.0, -5.0, 25.0}, {0.0, 0.0, 25.0}, {0.0, 0.0, 25.0}, {0.0, 0.0, 25.0}};
    const arma::mat rx = {{40.0, 35.0, 31.5}, {120.0, -90.0, 1.5}, {600.0, 800.0, 1.5}, {3000.0, 4000.0, 1.5}};
    const arma::vec fc = {1.5e9, 3.5e9, 28.0e9};
    const arma::mat rnd = rnd_values(4, 11);

    arma::cube lsp, mu, sg;
    quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, nullptr, &mu, &sg);
    REQUIRE(mu.n_rows == 4);
    REQUIRE(mu.n_cols == 3);
    REQUIRE(mu.n_slices == 8);
    REQUIRE(sg.n_rows == 4);
    REQUIRE(sg.n_cols == 3);
    REQUIRE(sg.n_slices == 8);

    QD_DIAG("\n# diag: lsp_generate mean and std with every term (mu, sigma per link, frequency and LSP)\n");
    QD_DIAG("link,fc_GHz,lsp,mu,mu_ref,sigma,sigma_ref\n");
    for (arma::uword l = 0; l < 4; ++l)
        for (arma::uword k = 0; k < 3; ++k)
        {
            double m[8], s[8], esd_raw;
            ref_mu_sigma(sp, fc(k), tx.row(l).eval().memptr(), rx.row(l).eval().memptr(), m, s, &esd_raw);
            for (int i = 0; i < 8; ++i)
            {
                QD_DIAG("%llu,%.1f,%s,%.15f,%.15f,%.15f,%.15f\n", (unsigned long long)l, fc(k) / 1e9, lsp_name[i],
                        mu(l, k, i), m[i], sg(l, k, i), s[i]);
                INFO("link " << l << ", fc " << fc(k) << ", LSP " << lsp_name[i]);
                CHECK(std::abs(mu(l, k, i) - m[i]) < 1e-12);
                CHECK(std::abs(sg(l, k, i) - s[i]) < 1e-12);
            }
            CHECK(mu(l, k, 2) == 0.0); // SF mean 0

            // Sanity of the fixture: the floor acts only for links 2 and 3, the outer limit of the ASD std for
            // links 2 and 3, the inner limit of the KF std only at 28 GHz
            INFO("link " << l << ", fc " << fc(k));
            CHECK((esd_raw < 0.2) == (l >= 2));
            const double fp = std::max(fc(k) / 1e9, 2.0);
            CHECK((1.5 - 1.2 * std::log10(0.5 + fp) < 0.0) == (k == 2));
        }
    CHECK(sg(3, 0, 3) == 0.0);
    CHECK(sg(1, 0, 3) > 0.05);

    // A std of 0 gives the median value 10^mu
    for (arma::uword k = 0; k < 3; ++k)
        CHECK(std::abs(lsp(3, k, 3) / std::pow(10.0, mu(3, k, 3)) - 1.0) < 1e-12);
}

TEST_CASE("lsp_generate - TR spot checks of four v19 files")
{
    // TR 38.901 v19.5.0, reference geometry tx = (0, 0, hBS), rx = (d2D, 0, hUT); fc in GHz
    // Expected mean and std per LSP: DS lg(s), KF dB, SF dB, ASD..ESA lg(deg), XPR dB
    typedef std::function<void(double fc, double hbs, double hut, double d, double *m, double *s)> tr_fun;
    auto lg = [](double x)
    { return std::log10(x); };

    struct spot
    {
        const char *file;
        double hbs, d;
        std::vector<double> hut, fc;
        tr_fun tr;
    };

    std::vector<spot> spots = {
        // UMa NLOS: Table 7.5-6 Part-1 (note 6: fc = 6 below 6 GHz), Table 7.5-7, SF std of Table 7.4.1-1
        {"UMa_NLOS", 25.0, 200.0, {1.5, 12.0}, {3.5, 28.0}, [&](double fc, double, double hut, double d, double *m, double *s)
         {
             const double f = std::max(fc, 6.0);
             m[0] = -6.47 - 0.134 * lg(f), s[0] = 0.39;
             m[1] = -100.0, s[1] = 0.0;
             m[2] = 0.0, s[2] = 6.0;
             m[3] = 1.09, s[3] = 0.44;
             m[4] = 2.04 - 0.25 * lg(f), s[4] = 0.17 - 0.03 * lg(f);
             m[5] = std::max(-0.5, -2.1 * d / 1000.0 - 0.01 * (hut - 1.5) + 0.9), s[5] = 0.49;
             m[6] = -0.2856 * lg(f) + 1.445, s[6] = 0.17;
             m[7] = 7.0, s[7] = 3.0;
         }},
        // UMi LOS: Table 7.5-6 Part-1 (note 7: fc = 2 below 2 GHz), Table 7.5-8, SF std of Table 7.4.1-1
        {"UMi_LOS", 10.0, 200.0, {1.5}, {1.5, 28.0}, [&](double fc, double hbs, double hut, double d, double *m, double *s)
         {
             const double f = std::max(fc, 2.0);
             m[0] = -0.18 * lg(1.0 + f) - 7.28, s[0] = 0.39;
             m[1] = 9.0, s[1] = 5.0;
             m[2] = 0.0, s[2] = 4.0;
             m[3] = -0.05 * lg(1.0 + f) + 1.21, s[3] = 0.08 * lg(1.0 + f) + 0.29;
             m[4] = -0.07 * lg(1.0 + f) + 1.66, s[4] = 0.021 * lg(1.0 + f) + 0.26;
             m[5] = std::max(-0.21, -14.8 * d / 1000.0 + 0.01 * std::abs(hut - hbs) + 0.83), s[5] = 0.35;
             m[6] = -0.11 * lg(1.0 + f) + 0.81, s[6] = -0.03 * lg(1.0 + f) + 0.29;
             m[7] = 9.0, s[7] = 3.0;
         }},
        // SMa NLOS: Table 7.5-6 Part-4, Table 7.5-12 (ZSD without distance or height terms), SF std of Table 7.4.1-1
        {"SMa_NLOS", 35.0, 200.0, {1.5, 4.5}, {3.5}, [&](double, double, double, double, double *m, double *s)
         {
             m[0] = -7.20, s[0] = 0.58;
             m[1] = -100.0, s[1] = 0.0;
             m[2] = 0.0, s[2] = 8.0;
             m[3] = 0.51, s[3] = 0.33;
             m[4] = 1.63, s[4] = 0.26;
             m[5] = 0.14, s[5] = 0.16;
             m[6] = 1.16, s[6] = 0.14;
             m[7] = 4.0, s[7] = 3.0;
         }},
        // InF NLOS DL: Table 7.5-6 Part-3, Table 7.5-11, SF std of Table 7.4.1-1; DS with V/S = 50/11 of the
        // calibration hall (checked separately below with its own tolerance)
        {"InF_NLOS_DL", 1.5, 30.0, {1.5}, {3.5, 28.0}, [&](double fc, double, double, double, double *m, double *s)
         {
             m[0] = lg(30.0 * 50.0 / 11.0 + 32.0) - 9.44, s[0] = 0.19;
             m[1] = -100.0, s[1] = 0.0;
             m[2] = 0.0, s[2] = 7.2;
             m[3] = 1.57, s[3] = 0.2;
             m[4] = 1.72, s[4] = 0.3;
             m[5] = 1.2, s[5] = 0.55;
             m[6] = -0.13 * lg(1.0 + fc) + 1.45, s[6] = 0.45;
             m[7] = 11.0, s[7] = 6.0;
         }}};

    QD_DIAG("\n# diag: lsp_generate TR spot checks (mean and std per LSP)\n");
    QD_DIAG("file,hUT,fc_GHz,lsp,mu,mu_TR,sigma,sigma_TR\n");
    for (const auto &sc : spots)
    {
        const scen_t sp = conf(sc.file);
        REQUIRE(sp.size() > 100);
        arma::vec fc(sc.fc.size());
        for (size_t k = 0; k < sc.fc.size(); ++k)
            fc(k) = sc.fc[k] * 1e9;
        arma::mat rx(sc.hut.size(), 3, arma::fill::zeros);
        for (size_t l = 0; l < sc.hut.size(); ++l)
            rx(l, 0) = sc.d, rx(l, 2) = sc.hut[l];
        const arma::mat tx = {{0.0, 0.0, sc.hbs}};
        const arma::mat rnd(rx.n_rows, 11, arma::fill::zeros);

        arma::cube lsp, mu, sg;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, nullptr, &mu, &sg);
        REQUIRE(mu.n_rows == rx.n_rows);

        for (arma::uword l = 0; l < rx.n_rows; ++l)
            for (arma::uword k = 0; k < fc.n_elem; ++k)
            {
                double m[8], s[8];
                sc.tr(sc.fc[k], sc.hbs, sc.hut[l], sc.d, m, s);
                for (int i = 0; i < 8; ++i)
                {
                    QD_DIAG("%s,%.1f,%.1f,%s,%.15f,%.15f,%.15f,%.15f\n", sc.file, sc.hut[l], sc.fc[k], lsp_name[i],
                            mu(l, k, i), m[i], sg(l, k, i), s[i]);
                    INFO(sc.file << ", hUT " << sc.hut[l] << ", fc " << sc.fc[k] << ", LSP " << lsp_name[i]);
                    // The file value of DS_VS is the hall's 50/11 rounded to 4.55: bound
                    // lg((30·4.55 + 32) / (30·50/11 + 32)) = 3.52e-4 in lg
                    const double tol = (i == 0 && std::string(sc.file) == "InF_NLOS_DL") ? 3.6e-4 : 1e-12;
                    CHECK(std::abs(mu(l, k, i) - m[i]) < tol);
                    CHECK(std::abs(sg(l, k, i) - s[i]) < 1e-12);
                }
            }
    }

    // InF NLOS DL with DS_VS set in a scenpar copy: TR formula lg(30·V/S + 32) - 9.44 with that V/S
    scen_t sp = conf("InF_NLOS_DL");
    const arma::mat tx = {{0.0, 0.0, 1.5}}, rx = {{30.0, 0.0, 1.5}};
    const arma::vec fc = {3.5e9};
    const arma::mat rnd(1, 11, arma::fill::zeros);
    for (double vs : {4.0, 50.0 / 11.0})
    {
        sp["DS_VS"] = vs;
        arma::cube lsp, mu;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, nullptr, &mu);
        const double ref = std::log10(30.0 * vs + 32.0) - 9.44;
        QD_DIAG("InF_NLOS_DL DS_VS %.6f: mu %.15f, TR %.15f\n", vs, mu(0, 0, 0), ref);
        INFO("DS_VS " << vs);
        CHECK(std::abs(mu(0, 0, 0) - ref) < 1e-12);
    }
}

TEST_CASE("lsp_generate - EoD offset formula and sign")
{
    // Formula with every term:
    //   e = e0 + e_gamma·lg f'
    //   b > 0: e += 10^( (a + a_gamma·lg f')·lg(max(b, d)) + c + c_gamma·lg f' + c_hUT·hUT )
    //   hcl > 0 and hUT < hcl: e += (atan((hcl - h)/d) - atan((hUT - h)/d))·180/pi
    // with f' = max(fc/1e9, LSP_fc_min), d = max(d2D, 1e-4), h = max(hBS, 1e-4)
    auto eod_ref = [](const scen_t &sp, double fc, double hbs, double hut, double d2d)
    {
        const double f = std::max(fc / 1e9, getv(sp, "LSP_fc_min")), lf = std::log10(f);
        const double d = std::max(d2d, 1e-4), h = std::max(hbs, 1e-4);
        double e = getv(sp, "EoD_offset_e") + getv(sp, "EoD_offset_e_gamma") * lf;
        const double b = getv(sp, "EoD_offset_b"), hcl = getv(sp, "EoD_offset_hcl");
        if (b > 0.0)
            e += std::pow(10.0, (getv(sp, "EoD_offset_a") + getv(sp, "EoD_offset_a_gamma") * lf) * std::log10(std::max(b, d)) +
                                    getv(sp, "EoD_offset_c") + getv(sp, "EoD_offset_c_gamma") * lf + getv(sp, "EoD_offset_c_hUT") * hut);
        if (hcl > 0.0 && hut < hcl)
            e += (std::atan((hcl - h) / d) - std::atan((hut - h) / d)) * 180.0 / pi;
        return e;
    };

    scen_t sp = {{"EoD_offset_e", 3.1}, {"EoD_offset_e_gamma", -2.2}, {"EoD_offset_a", -0.6}, {"EoD_offset_a_gamma", 0.15},
                 {"EoD_offset_b", 20.0}, {"EoD_offset_c", 1.9}, {"EoD_offset_c_gamma", -0.11}, {"EoD_offset_c_hUT", -0.05},
                 {"EoD_offset_hcl", 6.0}, {"LSP_fc_min", 4.0}};
    const double hbs = 20.0;
    const std::vector<double> dist = {5.0, 50.0, 500.0, 2000.0}, hut = {1.5, 9.0}; // hUT below and above hcl
    arma::mat rx(dist.size() * hut.size(), 3, arma::fill::zeros);
    for (size_t i = 0; i < dist.size(); ++i)
        for (size_t j = 0; j < hut.size(); ++j)
            rx(i * hut.size() + j, 0) = 0.6 * dist[i], rx(i * hut.size() + j, 1) = 0.8 * dist[i], rx(i * hut.size() + j, 2) = hut[j];
    const arma::mat tx = {{0.0, 0.0, hbs}};
    const arma::vec fc = {2.0e9, 10.0e9};
    const arma::mat rnd(rx.n_rows, 11, arma::fill::zeros);

    QD_DIAG("\n# diag: lsp_generate EoD offset with every term\n");
    QD_DIAG("b,d2D,hUT,fc_GHz,eod,eod_ref\n");
    for (double b : {20.0, 0.0})
    {
        sp["EoD_offset_b"] = b;
        arma::cube lsp;
        arma::mat eod;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, &eod);
        REQUIRE(eod.n_rows == rx.n_rows);
        REQUIRE(eod.n_cols == 2);
        for (size_t i = 0; i < dist.size(); ++i)
            for (size_t j = 0; j < hut.size(); ++j)
                for (arma::uword k = 0; k < 2; ++k)
                {
                    const arma::uword l = i * hut.size() + j;
                    const double ref = eod_ref(sp, fc(k), hbs, hut[j], dist[i]);
                    QD_DIAG("%.0f,%.0f,%.1f,%.0f,%.12f,%.12f\n", b, dist[i], hut[j], fc(k) / 1e9, eod(l, k), ref);
                    INFO("b " << b << ", d2D " << dist[i] << ", hUT " << hut[j] << ", fc " << fc(k));
                    CHECK(std::abs(eod(l, k) - ref) < 1e-10);
                }
    }

    // TR spot checks: EoD offset = -mu_offset,ZOD
    QD_DIAG("\n# diag: lsp_generate EoD offset against the TR\n");
    QD_DIAG("file,d2D,hUT,fc_GHz,eod,minus_mu_offset_TR\n");
    {
        // UMa NLOS, Table 7.5-7 with note 4 (fc = 6 below 6 GHz):
        //   mu_offset = e(fc) - 10^( a(fc)·lg(max(b(fc), d2D)) + c(fc) - 0.07·(hUT - 1.5) )
        //   a = 0.208·lg fc - 0.782, b = 25, c = -0.13·lg fc + 2.03, e = 7.66·lg fc - 5.96
        const scen_t uma = conf("UMa_NLOS");
        const arma::mat tx1 = {{0.0, 0.0, 25.0}};
        const arma::mat rx1 = {{10.0, 0.0, 1.5}, {200.0, 0.0, 1.5}, {10.0, 0.0, 12.0}, {200.0, 0.0, 12.0}};
        const arma::vec fc1 = {3.5e9, 28.0e9};
        arma::cube lsp;
        arma::mat eod;
        quadriga_lib::lsp_generate(uma, fc1, tx1, rx1, arma::mat(4, 11, arma::fill::zeros), lsp, nullptr, nullptr, nullptr, nullptr, &eod);
        for (arma::uword l = 0; l < 4; ++l)
            for (arma::uword k = 0; k < 2; ++k)
            {
                const double f = std::max(fc1(k) / 1e9, 6.0), lf = std::log10(f), d = rx1(l, 0), hu = rx1(l, 2);
                const double a = 0.208 * lf - 0.782, c = -0.13 * lf + 2.03, e = 7.66 * lf - 5.96;
                const double mu_off = e - std::pow(10.0, a * std::log10(std::max(25.0, d)) + c - 0.07 * (hu - 1.5));
                QD_DIAG("UMa_NLOS,%.0f,%.1f,%.1f,%.12f,%.12f\n", d, hu, fc1(k) / 1e9, eod(l, k), -mu_off);
                INFO("UMa NLOS, d2D " << d << ", hUT " << hu << ", fc " << fc1(k));
                CHECK(std::abs(eod(l, k) + mu_off) < 1e-10);
            }
    }
    {
        // SMa NLOS, Table 7.5-12: mu_offset = atan((hBS - hcl)/d2D) - atan((hBS - hUT)/d2D) for hUT < hcl, else 0;
        // hcl = 3.5 m, in degrees
        const scen_t sma = conf("SMa_NLOS");
        const arma::mat tx1 = {{0.0, 0.0, 35.0}};
        const arma::mat rx1 = {{50.0, 0.0, 1.5}, {200.0, 0.0, 1.5}, {50.0, 0.0, 4.5}, {200.0, 0.0, 4.5}};
        const arma::vec fc1 = {3.5e9};
        arma::cube lsp;
        arma::mat eod;
        quadriga_lib::lsp_generate(sma, fc1, tx1, rx1, arma::mat(4, 11, arma::fill::zeros), lsp, nullptr, nullptr, nullptr, nullptr, &eod);
        for (arma::uword l = 0; l < 4; ++l)
        {
            const double d = rx1(l, 0), hu = rx1(l, 2);
            const double mu_off = (hu < 3.5) ? (std::atan((35.0 - 3.5) / d) - std::atan((35.0 - hu) / d)) * 180.0 / pi : 0.0;
            QD_DIAG("SMa_NLOS,%.0f,%.1f,3.5,%.12f,%.12f\n", d, hu, eod(l, 0), -mu_off);
            INFO("SMa NLOS, d2D " << d << ", hUT " << hu);
            CHECK(std::abs(eod(l, 0) + mu_off) < 1e-10);
        }
        CHECK(eod(0, 0) > 0.0); // UT below the clutter: the departure is steered up
    }
    {
        // UMa LOS, Table 7.5-7: mu_offset = 0
        const scen_t uma = conf("UMa_LOS");
        const arma::mat tx1 = {{0.0, 0.0, 25.0}}, rx1 = {{10.0, 0.0, 1.5}, {200.0, 0.0, 22.5}};
        arma::cube lsp;
        arma::mat eod;
        quadriga_lib::lsp_generate(uma, arma::vec({3.5e9, 28e9}), tx1, rx1, arma::mat(2, 11, arma::fill::zeros), lsp,
                                   nullptr, nullptr, nullptr, nullptr, &eod);
        CHECK(arma::all(arma::vectorise(eod) == 0.0));
    }
}

TEST_CASE("lsp_generate - values are 10^(mu + sigma x) with x from the correlation root")
{
    // Identity correlation (all correlation keys 0): x = s; distinct mean and std per LSP
    const double m[8] = {-7.3, 6.5, 0.0, 1.21, 1.67, 0.83, 1.11, 9.4};
    const double s[8] = {0.37, 4.2, 5.5, 0.29, 0.23, 0.31, 0.19, 3.7};
    scen_t sp;
    for (int i = 0; i < 8; ++i)
    {
        if (i != 2)
            sp[std::string(lsp_prefix[i]) + "_mu"] = m[i];
        sp[std::string(lsp_prefix[i]) + "_sigma"] = s[i];
    }
    const arma::mat tx = {{0.0, 0.0, 25.0}};
    const arma::mat rx = {{100.0, 0.0, 1.5}, {-30.0, 70.0, 1.5}, {5.0, -300.0, 10.0}};
    const arma::mat rnd = {{0.3, -1.2, 0.7, 1.1, -0.4, 0.9, -1.5, 0.2, 0.0, 0.5, 0.5},
                           {-0.8, 0.6, -1.9, 0.1, 1.3, -0.2, 0.4, -1.1, 0.0, 0.5, 0.5},
                           {1.7, 0.05, 0.4, -0.6, -1.0, 1.2, 0.8, 2.1, 0.0, 0.5, 0.5}};
    const arma::vec fc = {3.5e9, 28e9};

    for (bool chol : {true, false})
    {
        arma::cube lsp;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, chol);
        REQUIRE(lsp.n_rows == 3);
        REQUIRE(lsp.n_cols == 2);
        REQUIRE(lsp.n_slices == 8);
        for (arma::uword l = 0; l < 3; ++l)
            for (arma::uword k = 0; k < 2; ++k)
                for (int i = 0; i < 8; ++i)
                {
                    const double v = m[i] + s[i] * rnd(l, i);
                    const double ref = lsp_db[i] ? std::pow(10.0, 0.1 * v) : std::pow(10.0, v);
                    INFO("cholesky " << chol << ", link " << l << ", fc " << fc(k) << ", LSP " << lsp_name[i]);
                    CHECK(std::abs(lsp(l, k, i) / ref - 1.0) < 1e-12);
                }
    }
}

TEST_CASE("lsp_generate - the Cholesky root follows the TR order")
{
    // TR 38.901 clause 7.5 step 4: Cholesky root of the correlation matrix in the order [SF, K, DS, ASD, ASA, ZSD,
    // ZSA], XPR last; position a of the TR vector holds our LSP p[a]
    const int p[8] = {2, 1, 0, 3, 4, 5, 6, 7};
    const arma::mat R = custom_corr();
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            REQUIRE(std::abs(R(i, j)) > 0.01);

    scen_t sp = unit_scen();
    set_corr(sp, R);
    const arma::mat Rs = corr_matrix(sp); // the exact doubles in scenpar

    arma::mat Rp(8, 8), L;
    for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b)
            Rp(a, b) = Rs(p[a], p[b]);
    REQUIRE(test_chol(Rp, L));
    arma::mat Q_ref(8, 8);
    for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b)
            Q_ref(p[a], p[b]) = L(a, b);

    const arma::mat tx = {{0.0, 0.0, 25.0}};
    arma::mat rx(8, 3, arma::fill::zeros);
    rx.col(0) = arma::linspace(10.0, 80.0, 8);
    arma::cube lsp;
    quadriga_lib::lsp_generate(sp, arma::vec({3.5e9}), tx, rx, unit_rnd(), lsp);
    const arma::mat Q = root_from_lsp(lsp);
    REQUIRE(Q.is_finite());

    QD_DIAG("\n# diag: lsp_generate Cholesky root (column j = links with s = e_j), max abs difference to the test-side root\n");
    QD_DIAG("max_abs_Q_minus_ref,max_abs_QQt_minus_R\n");
    QD_DIAG("%.3e,%.3e\n", arma::abs(Q - Q_ref).max(), arma::abs(mm(Q, Q.t()) - Rs).max());

    CHECK(arma::approx_equal(Q, Q_ref, "absdiff", 1e-12));
    CHECK(arma::approx_equal(mm(Q, Q.t()), Rs, "absdiff", 1e-12));
    for (int j = 0; j < 8; ++j) // x_SF = s_SF
        CHECK(std::abs(Q(2, j) - (j == 2 ? 1.0 : 0.0)) < 1e-12);
}

TEST_CASE("lsp_generate - the symmetric square root is the principal root of R")
{
    // The symmetric positive definite root of an SPD matrix is unique: X = X^T, X·X = R, X positive definite.
    // Tolerance 1e-12: the iteration ends at rounding level (residual about 2e-14 for the 0.999999 matrix).
    arma::mat R_near = arma::eye(8, 8);
    R_near(0, 1) = R_near(1, 0) = 0.999999; // smallest eigenvalue 1e-6
    const std::vector<std::pair<std::string, arma::mat>> cases = {
        {"custom", custom_corr()}, {"one correlation 0.999999", R_near}, {"identity", arma::eye(8, 8)}};

    const arma::mat tx = {{0.0, 0.0, 25.0}};
    arma::mat rx(8, 3, arma::fill::zeros);
    rx.col(0) = arma::linspace(10.0, 80.0, 8);

    QD_DIAG("\n# diag: lsp_generate symmetric square root\n");
    QD_DIAG("matrix,max_abs_XX_minus_R,max_abs_X_minus_Xt\n");
    for (const auto &c : cases)
    {
        scen_t sp = unit_scen();
        set_corr(sp, c.second);
        const arma::mat R = corr_matrix(sp);
        arma::cube lsp;
        quadriga_lib::lsp_generate(sp, arma::vec({3.5e9}), tx, rx, unit_rnd(), lsp, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, false);
        const arma::mat X = root_from_lsp(lsp);
        const double e_sq = arma::abs(mm(X, X) - R).max(), e_sym = arma::abs(X - X.t()).max();
        QD_DIAG("%s,%.3e,%.3e\n", c.first.c_str(), e_sq, e_sym);
        INFO(c.first);
        REQUIRE(X.is_finite()); // arma::max skips NaN
        CHECK(e_sq <= 1e-12);
        CHECK(e_sym <= 1e-12);
        arma::mat L;
        CHECK(test_chol(0.5 * (X + X.t()), L)); // positive definite
        if (c.first == "identity")
            CHECK(arma::abs(X - arma::eye(8, 8)).max() <= 1e-12);
    }
}

TEST_CASE("lsp_generate - sample covariance of the log values equals the correlation matrix")
{
    // 20000 links with Normal random values; mean 0 and std 1 for all LSPs, correlations of the UMa NLOS file; then
    // x = lg(lsp) (DS, AS, ES) and 10·lg(lsp) (KF, SF, XPR) has the covariance R.
    // Tolerance 0.05 = 5 standard errors: sqrt(2/n) = 0.010 on the diagonal, sqrt((1 + r²)/n) <= 0.010 off it.
    const arma::uword n = 20000;
    const scen_t uma = conf("UMa_NLOS");
    scen_t sp = unit_scen();
    for (const auto &k : corr_keys)
        sp[k.name] = uma.at(k.name);
    const arma::mat R = corr_matrix(sp);

    const arma::mat rnd = box_muller(n, 11, 4711);
    arma::mat rx(n, 3, arma::fill::zeros);
    rx.col(0) = arma::linspace(1.0, 2000.0, n);
    const arma::mat tx = {{0.0, 0.0, 25.0}};

    QD_DIAG("\n# diag: lsp_generate sample covariance of the log values vs R (UMa NLOS correlations)\n");
    QD_DIAG("root,max_abs_cov_minus_R\n");
    for (bool chol : {true, false})
    {
        arma::cube lsp;
        quadriga_lib::lsp_generate(sp, arma::vec({3.5e9}), tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, chol, false);
        arma::mat X(n, 8);
        for (arma::uword l = 0; l < n; ++l)
            for (int i = 0; i < 8; ++i)
                X(l, i) = lsp_to_x(lsp(l, 0, i), i);
        arma::mat Xc = X.each_row() - arma::mean(X, 0);
        const arma::mat C = mm(Xc.t(), Xc) / (double)(n - 1);
        QD_DIAG("%s,%.4f\n", chol ? "cholesky" : "symmetric", arma::abs(C - R).max());
        INFO("cholesky " << chol);
        REQUIRE(C.is_finite()); // arma::max skips NaN
        CHECK(arma::abs(C - R).max() < 0.05);
    }
}

TEST_CASE("lsp_generate - spreads are clipped at 104 and 52 degrees")
{
    scen_t sp = {{"AS_D_mu", 2.5}, {"AS_A_mu", 2.5}, {"ES_D_mu", 2.0}, {"ES_A_mu", 2.0}};
    const arma::mat tx = {{0.0, 0.0, 25.0}}, rx = {{100.0, 0.0, 1.5}, {200.0, 50.0, 1.5}};
    const arma::mat rnd(2, 11, arma::fill::zeros);
    const arma::vec fc = {3.5e9};
    arma::cube lsp_c, lsp_n, mu_c, mu_n;
    quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp_c, nullptr, nullptr, nullptr, nullptr, nullptr, &mu_c, nullptr, true, true);
    quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp_n, nullptr, nullptr, nullptr, nullptr, nullptr, &mu_n, nullptr, true, false);
    for (arma::uword l = 0; l < 2; ++l)
    {
        INFO("link " << l);
        CHECK(lsp_c(l, 0, 3) == 104.0);
        CHECK(lsp_c(l, 0, 4) == 104.0);
        CHECK(lsp_c(l, 0, 5) == 52.0);
        CHECK(lsp_c(l, 0, 6) == 52.0);
        CHECK(lsp_n(l, 0, 3) > 104.0);
        CHECK(lsp_n(l, 0, 4) > 104.0);
        CHECK(lsp_n(l, 0, 5) > 52.0);
        CHECK(lsp_n(l, 0, 6) > 52.0);
        for (int i : {0, 1, 2, 7})
            CHECK(lsp_c(l, 0, i) == lsp_n(l, 0, i));
    }
    CHECK(arma::approx_equal(mu_c, mu_n, "absdiff", 0.0));
}

TEST_CASE("lsp_generate - the same random values apply to all frequencies")
{
    // TR 38.901 clause 7.6.5: one realization x per link for all frequencies; (x_lsp - mu)/sigma is the same
    const scen_t sp = conf("UMa_LOS");
    const arma::mat tx = {{0.0, 0.0, 25.0}}, rx = {{100.0, 0.0, 1.5}, {-150.0, 80.0, 1.5}, {30.0, -400.0, 7.5}};
    const arma::mat rnd = rnd_values(3, 99);
    const arma::vec fc = {7.0e9, 15.0e9, 28.0e9};
    arma::cube lsp, mu, sg;
    quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, nullptr, &mu, &sg, true, false);
    REQUIRE(lsp.n_cols == 3);
    CHECK(std::abs(mu(0, 0, 0) - mu(0, 2, 0)) > 1e-3); // the DS mean depends on the frequency
    for (arma::uword l = 0; l < 3; ++l)
        for (int i = 0; i < 8; ++i)
        {
            if (sg(l, 0, i) <= 0.0)
                continue;
            const double z0 = (lsp_to_x(lsp(l, 0, i), i) - mu(l, 0, i)) / sg(l, 0, i);
            for (arma::uword k = 1; k < 3; ++k)
            {
                const double z = (lsp_to_x(lsp(l, k, i), i) - mu(l, k, i)) / sg(l, k, i);
                INFO("link " << l << ", LSP " << lsp_name[i] << ", fc " << fc(k));
                CHECK(std::abs(z - z0) < 1e-12);
            }
        }
}

TEST_CASE("lsp_generate - sf_sigma from path_loss replaces the SF standard deviation")
{
    const scen_t sp = conf("UMa_NLOS");
    const arma::mat tx = {{0.0, 0.0, 25.0}}, rx = {{100.0, 0.0, 1.5}, {-150.0, 80.0, 1.5}, {30.0, -400.0, 7.5}};
    const arma::mat rnd = rnd_values(3, 5);
    const arma::vec fc = {3.5e9, 28.0e9};
    arma::mat sf(3, 2);
    for (arma::uword l = 0; l < 3; ++l)
        for (arma::uword k = 0; k < 2; ++k)
            sf(l, k) = 3.1 + 0.7 * (double)l + 1.3 * (double)k;

    const outputs a = run_all(sp, fc, tx, rx, rnd);
    const outputs b = run_all(sp, fc, tx, rx, rnd, &sf);
    for (arma::uword l = 0; l < 3; ++l)
        for (arma::uword k = 0; k < 2; ++k)
        {
            INFO("link " << l << ", fc " << fc(k));
            // Cholesky root in TR order passes the SF field through: x_SF = s_SF; SF mean 0
            const double ref = std::pow(10.0, 0.1 * sf(l, k) * rnd(l, 2));
            CHECK(std::abs(b.lsp(l, k, 2) / ref - 1.0) < 1e-12);
            CHECK(b.sg(l, k, 2) == sf(l, k));
            CHECK(a.sg(l, k, 2) == 6.0);
            for (int i : {0, 1, 3, 4, 5, 6, 7})
            {
                CHECK(b.lsp(l, k, i) == a.lsp(l, k, i));
                CHECK(b.sg(l, k, i) == a.sg(l, k, i));
            }
        }
    CHECK(arma::approx_equal(b.mu, a.mu, "absdiff", 0.0));
}

TEST_CASE("lsp_generate - LSP_fc_min clamps the frequency")
{
    scen_t sp = conf("UMa_NLOS");
    sp["LSP_fc_min"] = 6.0;
    const arma::mat tx = {{0.0, 0.0, 25.0}}, rx = {{100.0, 0.0, 1.5}, {300.0, 80.0, 1.5}};
    const arma::vec fc = {2.0e9, 6.0e9, 10.0e9};
    const outputs o = run_all(sp, fc, tx, rx, rnd_values(2, 3));
    for (arma::uword l = 0; l < 2; ++l)
    {
        INFO("link " << l);
        for (int i = 0; i < 8; ++i)
        {
            CHECK(o.mu(l, 0, i) == o.mu(l, 1, i));
            CHECK(o.sg(l, 0, i) == o.sg(l, 1, i));
            CHECK(o.lsp(l, 0, i) == o.lsp(l, 1, i));
        }
        CHECK(o.eod(l, 0) == o.eod(l, 1));
        CHECK(std::abs(o.mu(l, 2, 0) - o.mu(l, 1, 0)) > 1e-6); // DS mean
        CHECK(std::abs(o.mu(l, 2, 4) - o.mu(l, 1, 4)) > 1e-6); // ASA mean
        CHECK(std::abs(o.sg(l, 2, 4) - o.sg(l, 1, 4)) > 1e-6); // ASA std
        CHECK(std::abs(o.mu(l, 2, 6) - o.mu(l, 1, 6)) > 1e-6); // ESA mean
        CHECK(std::abs(o.eod(l, 2) - o.eod(l, 1)) > 1e-6);
    }
}

TEST_CASE("lsp_generate - InF delay spread uses V/S")
{
    // TR 38.901 Table 7.5-6 Part-3, InF LOS: mu_lgDS = lg(26·(V/S) + 14) - 9.35
    scen_t sp = conf("InF_LOS");
    const arma::mat tx = {{0.0, 0.0, 8.0}}, rx = {{30.0, 0.0, 1.5}};
    const arma::vec fc = {3.5e9};
    const arma::mat rnd(1, 11, arma::fill::zeros);
    auto ds_mu = [&](const scen_t &s)
    {
        arma::cube lsp, mu;
        quadriga_lib::lsp_generate(s, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, nullptr, &mu);
        return mu(0, 0, 0);
    };

    // File default 4.27 for the hall's V/S = 47/11: rounding bound lg((26·47/11 + 14)/(26·4.27 + 14)) = 2.46e-4 in lg
    const double ref_hall = std::log10(26.0 * 47.0 / 11.0 + 14.0) - 9.35;
    const double v_file = ds_mu(sp);
    QD_DIAG("\n# diag: lsp_generate InF LOS DS mean\nDS_VS,mu,mu_ref\n");
    QD_DIAG("file,%.15f,%.15f\n", v_file, ref_hall);
    CHECK(std::abs(v_file - ref_hall) < 2.5e-4);

    for (double vs : {4.0, 50.0 / 11.0})
    {
        sp["DS_VS"] = vs;
        const double v = ds_mu(sp), ref = std::log10(26.0 * vs + 14.0) - 9.35;
        QD_DIAG("%.6f,%.15f,%.15f\n", vs, v, ref);
        INFO("DS_VS " << vs);
        CHECK(std::abs(v - ref) < 1e-12);
    }
    sp["DS_VS"] = 0.0; // no V/S term
    CHECK(std::abs(ds_mu(sp) + 9.35) < 1e-12);
}

TEST_CASE("lsp_generate - absolute TOA")
{
    // TR 38.901 clause 7.6.9: 10^(absTOA_mu + absTOA_sigma·x), active for absTOA_mu > -30, bounded by 2·L/c
    const arma::mat tx = {{0.0, 0.0, 10.0}}, rx = {{50.0, 0.0, 1.5}, {80.0, 20.0, 1.5}, {10.0, -60.0, 1.5}};
    arma::mat rnd(3, 11, arma::fill::zeros);
    rnd(0, 8) = 0.0, rnd(1, 8) = 4.0, rnd(2, 8) = -1.3;
    const arma::vec fc = {3.5e9};
    auto toa = [&](const scen_t &sp)
    {
        arma::cube lsp;
        arma::vec t;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, &t);
        return t;
    };

    arma::vec t = toa({{"absTOA_mu", -30.0}, {"absTOA_sigma", 0.5}});
    REQUIRE(t.n_elem == 3);
    CHECK(arma::all(t == 0.0));

    t = toa({{"absTOA_mu", -29.99}, {"absTOA_sigma", 0.5}});
    for (arma::uword l = 0; l < 3; ++l)
        CHECK(std::abs(t(l) / std::pow(10.0, -29.99 + 0.5 * rnd(l, 8)) - 1.0) < 1e-12);

    // Unbounded and bounded: 2·L/c = 2.0014e-6 s for L = 300 m, between 1e-7 s (link 0) and 1e-5 s (link 1)
    const double bound = 2.0 * 300.0 / c0;
    t = toa({{"absTOA_mu", -7.0}, {"absTOA_sigma", 0.5}});
    arma::vec tb = toa({{"absTOA_mu", -7.0}, {"absTOA_sigma", 0.5}, {"absTOA_Lmax", 300.0}});
    QD_DIAG("\n# diag: lsp_generate absolute TOA [s]\nlink,unbounded,bounded,2L_over_c\n");
    for (arma::uword l = 0; l < 3; ++l)
    {
        const double ref = std::pow(10.0, -7.0 + 0.5 * rnd(l, 8));
        QD_DIAG("%llu,%.6e,%.6e,%.6e\n", (unsigned long long)l, t(l), tb(l), bound);
        INFO("link " << l);
        CHECK(std::abs(t(l) / ref - 1.0) < 1e-12);
        CHECK(std::abs(tb(l) / std::min(ref, bound) - 1.0) < 1e-12);
    }
    CHECK(tb(0) < bound);
    CHECK(tb(1) == bound);
}

TEST_CASE("lsp_generate - ground permittivity interpolates dry, medium and wet ground")
{
    // dry = 3 + j·0.003·f^1.34, medium = 30.4·f^-0.47 + j·0.18·f^1.05, wet = 31.3·f^-0.48 + j·0.63·f^0.77 (f in GHz,
    // not clamped by LSP_fc_min); U <= 0.5: (1 - 2U)·dry + 2U·medium; U > 0.5: (2 - 2U)·medium + (2U - 1)·wet
    const scen_t sp = {{"LSP_fc_min", 20.0}};
    const double U[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    arma::mat rnd(5, 11, arma::fill::zeros);
    for (arma::uword l = 0; l < 5; ++l)
        rnd(l, 9) = U[l];
    const arma::mat tx = {{0.0, 0.0, 25.0}};
    arma::mat rx(5, 3, arma::fill::zeros);
    rx.col(0) = arma::linspace(20.0, 100.0, 5);
    const arma::vec fc = {1.0e9, 10.0e9, 28.0e9};
    arma::cube lsp;
    arma::mat re, im;
    quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, &re, &im);
    REQUIRE(re.n_rows == 5);
    REQUIRE(re.n_cols == 3);
    REQUIRE(im.n_rows == 5);
    REQUIRE(im.n_cols == 3);

    QD_DIAG("\n# diag: lsp_generate ground permittivity\nU,f_GHz,re,re_ref,im,im_ref\n");
    for (arma::uword k = 0; k < 3; ++k)
    {
        const double f = fc(k) / 1e9;
        const double dry_re = 3.0, dry_im = 0.003 * std::pow(f, 1.34);
        const double med_re = 30.4 * std::pow(f, -0.47), med_im = 0.18 * std::pow(f, 1.05);
        const double wet_re = 31.3 * std::pow(f, -0.48), wet_im = 0.63 * std::pow(f, 0.77);
        for (arma::uword l = 0; l < 5; ++l)
        {
            double r, i;
            if (U[l] <= 0.5)
                r = (1.0 - 2.0 * U[l]) * dry_re + 2.0 * U[l] * med_re, i = (1.0 - 2.0 * U[l]) * dry_im + 2.0 * U[l] * med_im;
            else
                r = (2.0 - 2.0 * U[l]) * med_re + (2.0 * U[l] - 1.0) * wet_re, i = (2.0 - 2.0 * U[l]) * med_im + (2.0 * U[l] - 1.0) * wet_im;
            QD_DIAG("%.2f,%.0f,%.12f,%.12f,%.12f,%.12f\n", U[l], f, re(l, k), r, im(l, k), i);
            INFO("U " << U[l] << ", f " << f);
            CHECK(std::abs(re(l, k) - r) < 1e-12);
            CHECK(std::abs(im(l, k) - i) < 1e-12);
            CHECK(im(l, k) >= 0.0);
        }
    }
}

TEST_CASE("lsp_generate - optional outputs alone and together give the same values")
{
    const scen_t sp = conf("UMa_NLOS");
    const arma::mat tx = {{0.0, 0.0, 25.0}};
    const arma::mat rx = {{100.0, 0.0, 1.5}, {-150.0, 80.0, 1.5}, {30.0, -400.0, 7.5}, {500.0, 500.0, 22.5}};
    const arma::mat rnd = rnd_values(4, 21);
    const arma::vec fc = {2.0e9, 28.0e9};
    const outputs all = run_all(sp, fc, tx, rx, rnd);

    REQUIRE(all.lsp.n_rows == 4);
    REQUIRE(all.lsp.n_cols == 2);
    REQUIRE(all.lsp.n_slices == 8);
    REQUIRE(all.toa.n_elem == 4);
    REQUIRE(all.re.n_rows == 4);
    REQUIRE(all.re.n_cols == 2);
    REQUIRE(all.im.n_rows == 4);
    REQUIRE(all.im.n_cols == 2);
    REQUIRE(all.eod.n_rows == 4);
    REQUIRE(all.eod.n_cols == 2);
    REQUIRE(all.mu.n_slices == 8);
    REQUIRE(all.sg.n_slices == 8);

    // Each optional output alone
    {
        arma::cube lsp;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp);
        CHECK(arma::approx_equal(lsp, all.lsp, "absdiff", 0.0));
        arma::vec toa;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, &toa);
        CHECK(arma::approx_equal(toa, all.toa, "absdiff", 0.0));
        arma::mat re, im, eod;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, &re);
        CHECK(arma::approx_equal(re, all.re, "absdiff", 0.0));
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, &im);
        CHECK(arma::approx_equal(im, all.im, "absdiff", 0.0));
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, &eod);
        CHECK(arma::approx_equal(eod, all.eod, "absdiff", 0.0));
        arma::cube mu, sg;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, nullptr, &mu);
        CHECK(arma::approx_equal(mu, all.mu, "absdiff", 0.0));
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &sg);
        CHECK(arma::approx_equal(sg, all.sg, "absdiff", 0.0));
    }

    // Determinism: a repeated call gives identical results
    {
        const outputs again = run_all(sp, fc, tx, rx, rnd);
        CHECK(arma::approx_equal(again.lsp, all.lsp, "absdiff", 0.0));
        CHECK(arma::approx_equal(again.toa, all.toa, "absdiff", 0.0));
        CHECK(arma::approx_equal(again.eod, all.eod, "absdiff", 0.0));
    }

    // Pre-sized outputs with stale content are overwritten in place; wrongly sized ones are resized
    {
        outputs o;
        o.lsp.set_size(4, 2, 8), o.lsp.fill(NAN);
        o.mu.set_size(4, 2, 8), o.mu.fill(NAN);
        o.sg.set_size(1, 1, 1), o.sg.fill(NAN);
        o.toa.set_size(4), o.toa.fill(NAN);
        o.re.set_size(4, 2), o.re.fill(NAN);
        o.im.set_size(3, 3), o.im.fill(NAN);
        o.eod.set_size(4, 2), o.eod.fill(NAN);
        const double *p_lsp = o.lsp.memptr(), *p_toa = o.toa.memptr(), *p_re = o.re.memptr(), *p_mu = o.mu.memptr();
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, o.lsp, nullptr, &o.toa, &o.re, &o.im, &o.eod, &o.mu, &o.sg);
        CHECK(o.lsp.memptr() == p_lsp);
        CHECK(o.toa.memptr() == p_toa);
        CHECK(o.re.memptr() == p_re);
        CHECK(o.mu.memptr() == p_mu);
        CHECK(arma::approx_equal(o.lsp, all.lsp, "absdiff", 0.0));
        CHECK(arma::approx_equal(o.mu, all.mu, "absdiff", 0.0));
        CHECK(arma::approx_equal(o.sg, all.sg, "absdiff", 0.0));
        CHECK(arma::approx_equal(o.toa, all.toa, "absdiff", 0.0));
        CHECK(arma::approx_equal(o.re, all.re, "absdiff", 0.0));
        CHECK(arma::approx_equal(o.im, all.im, "absdiff", 0.0));
        CHECK(arma::approx_equal(o.eod, all.eod, "absdiff", 0.0));
    }

    // One tx_pos row equals the repeated rows
    {
        const arma::mat tx_rep = arma::repmat(tx, 4, 1);
        const outputs rep = run_all(sp, fc, tx_rep, rx, rnd);
        CHECK(arma::approx_equal(rep.lsp, all.lsp, "absdiff", 0.0));
        CHECK(arma::approx_equal(rep.mu, all.mu, "absdiff", 0.0));
        CHECK(arma::approx_equal(rep.sg, all.sg, "absdiff", 0.0));
        CHECK(arma::approx_equal(rep.toa, all.toa, "absdiff", 0.0));
        CHECK(arma::approx_equal(rep.re, all.re, "absdiff", 0.0));
        CHECK(arma::approx_equal(rep.im, all.im, "absdiff", 0.0));
        CHECK(arma::approx_equal(rep.eod, all.eod, "absdiff", 0.0));
    }

    // n_link = 0: outputs with 0 rows; tx_pos with 1 or 0 rows
    for (arma::uword n_tx : {1, 0})
    {
        outputs o = run_all(sp, fc, arma::mat(n_tx, 3, arma::fill::zeros), arma::mat(0, 3), arma::mat(0, 11));
        INFO("tx rows " << n_tx);
        CHECK(o.lsp.n_rows == 0);
        CHECK(o.lsp.n_cols == 2);
        CHECK(o.lsp.n_slices == 8);
        CHECK(o.mu.n_rows == 0);
        CHECK(o.mu.n_cols == 2);
        CHECK(o.mu.n_slices == 8);
        CHECK(o.sg.n_rows == 0);
        CHECK(o.sg.n_slices == 8);
        CHECK(o.toa.n_elem == 0);
        CHECK(o.re.n_rows == 0);
        CHECK(o.re.n_cols == 2);
        CHECK(o.im.n_rows == 0);
        CHECK(o.im.n_cols == 2);
        CHECK(o.eod.n_rows == 0);
        CHECK(o.eod.n_cols == 2);
    }
}

TEST_CASE("lsp_generate - input validation")
{
    const scen_t sp = conf("UMa_NLOS");
    const arma::mat tx = {{0.0, 0.0, 25.0}}, rx = {{100.0, 0.0, 1.5}, {-150.0, 80.0, 1.5}, {30.0, -400.0, 7.5}};
    const arma::mat rnd = rnd_values(3, 8);
    const arma::vec fc = {3.5e9, 28.0e9};
    arma::cube lsp;
    arma::mat re, im;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(run_all(sp, fc, tx, rx, rnd));
        CHECK_NOTHROW(quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp));
        CHECK_NOTHROW(quadriga_lib::lsp_generate({}, fc, tx, rx, rnd, lsp)); // empty scenpar: all defaults
        arma::mat sf(3, 2, arma::fill::value(4.0));
        CHECK_NOTHROW(run_all(sp, fc, tx, rx, rnd, &sf, false, false));
    }
    SECTION("rx_pos")
    {
        arma::mat bad(3, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, bad, rnd, lsp), std::invalid_argument,
                             Message("Input 'rx_pos' must have 3 columns."));
        bad = rx;
        bad(1, 2) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, bad, rnd, lsp), std::invalid_argument,
                             Message("Input 'rx_pos' must be finite."));
    }
    SECTION("tx_pos")
    {
        arma::mat bad(1, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, bad, rx, rnd, lsp), std::invalid_argument,
                             Message("Input 'tx_pos' must have 3 columns."));
        bad.zeros(2, 3);
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, bad, rx, rnd, lsp), std::invalid_argument,
                             Message("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'."));
        bad = tx;
        bad(0, 0) = INFINITY;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, bad, rx, rnd, lsp), std::invalid_argument,
                             Message("Input 'tx_pos' must be finite."));
    }
    SECTION("fc")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, arma::vec(), tx, rx, rnd, lsp), std::invalid_argument,
                             Message("Input 'fc' cannot be empty."));
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, arma::vec({3.5e9, 0.0}), tx, rx, rnd, lsp), std::invalid_argument,
                             Message("Input 'fc' must be finite and > 0."));
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, arma::vec({NAN}), tx, rx, rnd, lsp), std::invalid_argument,
                             Message("Input 'fc' must be finite and > 0."));
    }
    SECTION("rnd")
    {
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd.cols(0, 9), lsp), std::invalid_argument,
                             Message("Input 'rnd' must have size [n_link, 11]."));
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd.rows(0, 1), lsp), std::invalid_argument,
                             Message("Input 'rnd' must have size [n_link, 11]."));
        arma::mat bad = rnd;
        bad(2, 4) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, rx, bad, lsp), std::invalid_argument,
                             Message("Input 'rnd' must be finite."));
    }
    SECTION("sf_sigma")
    {
        arma::mat sf(3, 1, arma::fill::value(4.0));
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, &sf), std::invalid_argument,
                             Message("Input 'sf_sigma' must have size [n_link, n_freq]."));
        sf.set_size(3, 2), sf.fill(4.0), sf(1, 1) = -1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, &sf), std::invalid_argument,
                             Message("Input 'sf_sigma' must be finite and >= 0."));
        sf(1, 1) = NAN;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, &sf), std::invalid_argument,
                             Message("Input 'sf_sigma' must be finite and >= 0."));
    }
    SECTION("scenpar")
    {
        scen_t bad = sp;
        bad["DS_VS"] = 4.0, bad["DS_VS_a"] = 0.0, bad["DS_VS_b"] = 0.0;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(bad, fc, tx, rx, rnd, lsp), std::invalid_argument,
                             Message("Values of 'DS_VS_a' and 'DS_VS_b' must not both be 0 if 'DS_VS' is > 0."));
        bad = sp;
        bad["LSP_fc_min"] = -1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(bad, fc, tx, rx, rnd, lsp), std::invalid_argument,
                             Message("Value of 'LSP_fc_min' must be finite and >= 0."));
        bad = sp;
        bad["DS_muX"] = 1.0;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(bad, fc, tx, rx, rnd, lsp), std::invalid_argument,
                             Message("Unknown key 'DS_muX' in 'scenpar'."));
    }
    SECTION("not positive definite")
    {
        scen_t bad = sp;
        bad["ds_kf"] = 0.9, bad["ds_sf"] = 0.9, bad["sf_kf"] = -0.9;
        for (bool chol : {true, false})
        {
            arma::cube out(3, 2, 8, arma::fill::value(-7.0));
            CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(bad, fc, tx, rx, rnd, out, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, chol), std::invalid_argument,
                                 Message("LSP cross-correlation matrix in 'scenpar' is not positive definite."));
            CHECK(arma::all(arma::vectorise(out) == -7.0)); // output untouched
        }
    }
    SECTION("column 9 of rnd")
    {
        arma::mat bad = rnd;
        bad(1, 9) = 1.5;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, rx, bad, lsp, nullptr, nullptr, &re), std::invalid_argument,
                             Message("Column 9 of 'rnd' must be in the range from 0 to 1."));
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, rx, bad, lsp, nullptr, nullptr, nullptr, &im), std::invalid_argument,
                             Message("Column 9 of 'rnd' must be in the range from 0 to 1."));
        bad(1, 9) = -0.1;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(sp, fc, tx, rx, bad, lsp, nullptr, nullptr, &re, &im), std::invalid_argument,
                             Message("Column 9 of 'rnd' must be in the range from 0 to 1."));
        // Not checked when the ground permittivity is not requested
        bad(1, 9) = 1.5;
        arma::vec toa;
        arma::mat eod;
        CHECK_NOTHROW(quadriga_lib::lsp_generate(sp, fc, tx, rx, bad, lsp, nullptr, &toa, nullptr, nullptr, &eod));
    }
    SECTION("frequency terms")
    {
        // An active term with P_omega + f' <= 0 would give a NaN mean and a std of 0; inactive terms are not checked
        arma::cube out(3, 2, 8, arma::fill::value(-7.0));
        scen_t bad = sp;
        bad["DS_omega"] = -10.0, bad["DS_gamma"] = 1.0, bad["DS_delta"] = 0.0;
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(bad, fc, tx, rx, rnd, out), std::invalid_argument,
                             Message("Scenario parameter 'DS_omega' plus the frequency in GHz must be > 0 where 'DS_gamma' or 'DS_delta' is not 0."));
        bad = sp;
        bad["SF_omega"] = -28.0, bad["SF_delta"] = 1.0; // f' = max(f, LSP_fc_min = 6): -22 and 0
        CHECK_THROWS_MATCHES(quadriga_lib::lsp_generate(bad, fc, tx, rx, rnd, out), std::invalid_argument,
                             Message("Scenario parameter 'SF_omega' plus the frequency in GHz must be > 0 where 'SF_gamma' or 'SF_delta' is not 0."));
        CHECK(arma::all(arma::vectorise(out) == -7.0)); // output untouched
        bad = sp;
        bad["XPR_omega"] = -10.0, bad["XPR_gamma"] = 0.0, bad["XPR_delta"] = 0.0;
        CHECK_NOTHROW(quadriga_lib::lsp_generate(bad, fc, tx, rx, rnd, out));
        CHECK(out.is_finite());
    }
}
