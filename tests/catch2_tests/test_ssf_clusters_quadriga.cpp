// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "*quadriga*"

// Verification suite for the method "quadriga" of quadriga_lib::ssf_clusters, the cluster generation of the paper
// Jaeckel, Raschkowski, Burkhardt, Thiele, "A Spatially Consistent Geometric D2D Small-Scale Fading Model for
// Multiple Frequencies" (2019, "the paper"), test cases "ssf_clusters - quadriga ...".
//
// Pins down: unitless exponential initial delays and uniform initial angles from the random values; the power
// product of paper eq. (6) with the fitted coefficients of eqs. (7) to (11) (several frequencies) or r_tau - 1, 0.56,
// 0.76 (one frequency); the K-factor split; the analytic delay scaling (with the ground reflection as a fixed path);
// the iterative angle scaling with the caps 3 (azimuth) and 1.5 (elevation); the scales averaged over the
// frequencies; the LOS direction applied additively (azimuth wrap, elevation fold at the poles); the EoD offset on
// the NLOS departures of NLOS scenarios; the absolute time of arrival in NLOS scenarios (required where the scenario
// defines it); no delay spread to scale gives delay 0; the coefficient powers with the ground reflection; exact delay
// and angular spreads; continuity along tracks; swap reciprocity with dual-mobility random values. Sizes, the LOS
// condition, the LOS-only case, extreme random values and the ignored inputs: test_ssf_clusters.cpp.
// Oracles, independent of the function and of the conf files:
//  - ref_qd: the method typed here in scalar double (linear power normalisation, the uniform map
//    0.5·erfc(-x/sqrt(2)), the delay scale as the root of the quadratic var(s·tau) = DS², the iteration of the angle
//    scale); r_tau of TR 38.901 Table 7.5-6 is typed here
//  - the LOS direction and the ground reflection of qd_test_ssf.hpp
//  - spreads of the output with calc_delay_spread and a typed linear angular spread
// Tolerances: angles 1e-9 relative + 1e-12 rad, delays and powers 1e-10 relative against ref_qd (the angle scale
// is found by an iteration that stops at 1e-4 dB; the fixture keeps away from that threshold); spreads 1e-10 (DS)
// and 1e-4 (AS, the iteration threshold) relative; values formed by the same operations bitwise.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_tools.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "qd_test_diag.hpp"
#include "qd_test_ssf.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

namespace
{
    // TR 38.901 Table 7.5-6: delay scaling factor r_tau
    const double r_UMi_LOS = 3.0, r_UMi_NLOS = 2.1;

    // Elevation folded at the poles: wrapped to [-pi, pi), then mirrored at +-pi/2
    double fold_el(double e)
    {
        e = wrap_rad(e);
        if (e > 0.5 * pi)
            return pi - e;
        if (e < -0.5 * pi)
            return -pi - e;
        return e;
    }

    // Oracle: the method for link l; outputs per cluster n < N_l, power[k·N_l + n] coefficient power
    struct ref_t
    {
        std::vector<double> delay, power, ang[4];
        double s_raw[4] = {0, 0, 0, 0}, eps_d = 0.0, eps_a = 0.0, delta = 0.0;
        int iters = 0;         // largest number of angle-scale iterations
        bool conv[4] = {true, true, true, true}; // the angle-scale iteration reached 1e-4 dB at every frequency
        bool clamped = false;  // the delay scale hit the lower limit of the ground reflection at one frequency
        int n_fold = 0;        // NLOS elevations folded at a pole (after the scaling or the centre)
        int n_fold_a[4] = {0, 0, 0, 0};
    };
    ref_t ref_qd(const fix_t &f, arma::uword l, double r_tau, arma::uword N_l, bool los, bool gr)
    {
        const arma::uword N = f.rnd.n_cols / 9, F = f.lsp.n_cols;
        const geo_t g = geometry(f, l);
        ref_t r;

        // Initial delays and angles (paper eqs. (1), (4), (5))
        std::vector<double> tt(N_l), at[4];
        for (arma::uword n = 0; n < N_l; ++n)
        {
            tt[n] = -std::log(f.rnd(l, n));
            for (int a = 0; a < 4; ++a)
                at[a].push_back((0.5 * std::erfc(-f.rnd(l, (5 + a) * N + n) / std::sqrt(2.0)) - 0.5) * pi);
        }

        // Centres and the offsets of the LOS and ground-reflection paths
        double om = 0.0;
        if (gr && los)
        {
            for (double v : g.R2)
                om += v / (double)F;
        }
        r.eps_d = (1.0 - om) * g.ang[2] + om * g.e_gr, r.eps_a = (1.0 - om) * g.ang[3] + om * g.e_gr;
        const double fo_los[4] = {0.0, 0.0, g.ang[2] - r.eps_d, g.ang[3] - r.eps_a};
        const double fo_gr[4] = {0.0, 0.0, g.e_gr - r.eps_d, g.e_gr - r.eps_a};

        // Power coefficients (paper eqs. (7) to (11))
        auto mx = [&](arma::uword s)
        { double m = 0.0; for (arma::uword k = 0; k < F; ++k) m = std::max(m, f.lsp(l, k, s)); return m; };
        auto mn = [&](arma::uword s)
        { double m = 1e300; for (arma::uword k = 0; k < F; ++k) m = std::min(m, f.lsp(l, k, s)); return m; };
        std::vector<double> gc(5 * F);
        for (arma::uword k = 0; k < F; ++k)
        {
            if (F == 1)
                gc[5 * k] = r_tau - 1.0, gc[5 * k + 1] = gc[5 * k + 2] = 0.56, gc[5 * k + 3] = gc[5 * k + 4] = 0.76;
            else
            {
                if (std::abs(mx(0) / mn(0) - 1.0) < 1e-6)
                    gc[5 * k] = r_tau - 1.0;
                else
                {
                    const double d = std::min(std::max(f.lsp(l, k, 0) / (mx(0) + mn(0)), 0.15), 0.85);
                    gc[5 * k] = -1.5 * std::log(1.2 * d - 0.15);
                }
                for (arma::uword a = 0; a < 4; ++a)
                {
                    const double b = std::max(0.75 * f.lsp(l, k, 3 + a) / mx(3 + a), 0.25);
                    gc[5 * k + 1 + a] = (a < 2) ? -2.2 * std::log(1.5 * b - 0.35) : -3.4 * std::log(1.2 * b - 0.1);
                }
            }
        }

        // Powers per frequency with the split of the LOS power, delay scale and angle scales
        r.power.assign(F * N_l, 0.0);
        double s_tau = 0.0, s_ang[4] = {0, 0, 0, 0};
        for (arma::uword k = 0; k < F; ++k)
        {
            const double K = los ? f.lsp(l, k, 1) : 0.0, R2 = gr ? g.R2[k] : 0.0;
            const double S_los = K / (1.0 + K) * (1.0 - R2), S_gr = K / (1.0 + K) * R2;
            std::vector<double> P(N_l);
            double sum = 0.0;
            for (arma::uword n = 0; n < N_l; ++n)
            {
                P[n] = std::exp(-tt[n] * gc[5 * k]) * std::exp(-at[0][n] * at[0][n] * gc[5 * k + 1]) *
                       std::exp(-at[1][n] * at[1][n] * gc[5 * k + 2]) * std::exp(-std::abs(at[2][n]) * gc[5 * k + 3]) *
                       std::exp(-std::abs(at[3][n]) * gc[5 * k + 4]);
                sum += P[n];
            }
            for (arma::uword n = 0; n < N_l; ++n)
                P[n] = P[n] / sum / (1.0 + K), r.power[k * N_l + n] = P[n];

            // Delay scale: var(s) = S_gr·tau_gr² + s²·c - (S_gr·tau_gr + s·e)² = DS², the larger root
            double e = 0.0, c = 0.0;
            for (arma::uword n = 0; n < N_l; ++n)
                e += P[n] * tt[n], c += P[n] * tt[n] * tt[n];
            const double DS = f.lsp(l, k, 0), B = S_gr * g.tau_gr * g.tau_gr, D = S_gr * g.tau_gr, den = c - e * e;
            double disc = D * D * e * e + den * (DS * DS - B + D * D);
            if (disc < 0.0)
                disc = 0.0, r.clamped = true;
            s_tau += ((den > 0.0) ? (D * e + std::sqrt(disc)) / den : 0.0) / (double)F;

            // Angle scales: wrapped spread of the initial angles, then the linear spread iterated
            for (int a = 0; a < 4; ++a)
            {
                std::vector<double> w = {S_los}, A = {fo_los[a]};
                if (gr)
                    w.push_back(S_gr), A.push_back(fo_gr[a]);
                const size_t n0 = w.size();
                for (arma::uword n = 0; n < N_l; ++n)
                    w.push_back(P[n]), A.push_back(at[a][n]);
                auto spread = [&](const std::vector<double> &x)
                {
                    double m1 = 0.0, m2 = 0.0;
                    for (size_t i = 0; i < x.size(); ++i)
                        m1 += w[i] * x[i], m2 += w[i] * x[i] * x[i];
                    return std::sqrt(std::max(m2 - m1 * m1, 0.0));
                };
                double ss = 0.0, cs = 0.0;
                for (size_t i = 0; i < A.size(); ++i)
                    ss += w[i] * std::sin(A[i]), cs += w[i] * std::cos(A[i]);
                std::vector<double> phi(A.size());
                for (size_t i = 0; i < A.size(); ++i)
                    phi[i] = wrap_rad(A[i] - std::atan2(ss, cs));
                const double S0 = spread(phi), AS = f.lsp(l, k, 3 + a) * pi / 180.0;
                double sc = 1.0;
                if (S0 > 0.0)
                {
                    double sL = 10.0 * std::log10(AS / S0);
                    int it = 0;
                    bool conv = false;
                    for (; it < 9; ++it)
                    {
                        std::vector<double> x = A;
                        for (size_t i = n0; i < x.size(); ++i)
                            x[i] = std::pow(10.0, 0.1 * sL) * A[i];
                        const double S1 = spread(x);
                        if (!(S1 > 0.0))
                            break;
                        const double sN = 10.0 * std::log10(AS / S1);
                        if (std::abs(sN) <= 1e-4)
                        {
                            conv = true;
                            break;
                        }
                        sL += sN;
                    }
                    r.conv[a] = r.conv[a] && conv;
                    r.iters = std::max(r.iters, it + 1);
                    sc = std::pow(10.0, 0.1 * sL);
                }
                s_ang[a] += sc / (double)F;
            }
        }

        // Scaled offsets, LOS direction applied additively, EoD offset in NLOS scenarios
        if (!los && f.use_eod)
        {
            for (arma::uword k = 0; k < F; ++k)
                r.delta += f.eod(l, k) * pi / 180.0 / (double)F;
        }
        const double cen[4] = {g.ang[0], g.ang[1], r.eps_d + r.delta, r.eps_a};
        for (int a = 0; a < 4; ++a)
        {
            r.s_raw[a] = s_ang[a];
            const double s = std::min(s_ang[a], a < 2 ? 3.0 : 1.5);
            for (arma::uword n = 0; n < N_l; ++n)
            {
                double b = s * at[a][n];
                if (a < 2)
                    r.ang[a].push_back(wrap_rad(cen[a] + b));
                else
                {
                    const bool fo = std::abs(b) > 0.5 * pi || std::abs(cen[a] + fold_el(b)) > 0.5 * pi;
                    r.n_fold += fo, r.n_fold_a[a] += fo;
                    r.ang[a].push_back(fold_el(cen[a] + fold_el(b)));
                }
            }
        }

        // Delays; absolute TOA in NLOS scenarios
        double t_min = 1e300;
        for (arma::uword n = 0; n < N_l; ++n)
            r.delay.push_back(s_tau * tt[n]), t_min = std::min(t_min, s_tau * tt[n]);
        if (!los && f.use_toa) // the v19 NLOS files define the absolute TOA
            for (double &d : r.delay)
                d = d - t_min + f.toa(l);
        return r;
    }

    // Compare link l of the output with the oracle; returns the largest errors (delay, power relative; angle rad)
    void check_link(const out_t &o, const fix_t &f, arma::uword l, const ref_t &r, bool los, bool gr,
                    arma::uword N_l, double *err)
    {
        const geo_t g = geometry(f, l);
        const arma::uword F = f.lsp.n_cols, N = f.rnd.n_cols / 9, c0s = gr ? 2 : 1;
        REQUIRE(o.delay.n_cols == c0s + N);
        INFO("link " << l << ", iterations " << r.iters);
        for (arma::uword s = 0; s < c0s + N; ++s)
        {
            const bool cl = s >= c0s, act = cl && s - c0s < N_l;
            const arma::uword n = cl ? s - c0s : 0;
            for (int a = 0; a < 4; ++a)
            {
                const double ref = act ? r.ang[a][n] : ((s == 1 && gr && a > 1) ? g.e_gr : g.ang[a]);
                const double ang = (a == 0) ? o.aod(l, s, 0) : (a == 1) ? o.aoa(l, s, 0) : (a == 2) ? o.eod(l, s, 0) : o.eoa(l, s, 0);
                const double e = std::abs(wrap_rad(ang - ref));
                err[2] = std::max(err[2], e);
                CHECK(e <= 1e-9 * std::abs(ref) + 1e-12);
            }
            const double dref = act ? r.delay[n] : ((s == 1 && gr) ? g.tau_gr : 0.0);
            err[0] = std::max(err[0], std::abs(o.delay(l, s, 0) - dref) / std::max(dref, 1e-300));
            CHECK(std::abs(o.delay(l, s, 0) - dref) <= 1e-10 * dref);
            for (arma::uword k = 0; k < F; ++k)
            {
                const double K = los ? f.lsp(l, k, 1) : 0.0;
                const double pref = act ? r.power[k * N_l + n] : (cl ? 0.0 : K / (K + 1.0) * (s == 1 ? g.R2[k] : 1.0));
                err[1] = std::max(err[1], std::abs(o.power(l, s, k) - pref) / std::max(pref, 1e-300));
                CHECK(std::abs(o.power(l, s, k) - pref) <= 1e-10 * pref);
            }
        }
    }

    // Linear power-weighted spread with normalised weights
    double lin_spread(const std::vector<double> &x, const std::vector<double> &w)
    {
        double sw = 0.0, m1 = 0.0, m2 = 0.0;
        for (size_t i = 0; i < x.size(); ++i)
            sw += w[i];
        for (size_t i = 0; i < x.size(); ++i)
            m1 += w[i] / sw * x[i], m2 += w[i] / sw * x[i] * x[i];
        return std::sqrt(std::max(m2 - m1 * m1, 0.0));
    }
}

TEST_CASE("ssf_clusters - quadriga matches the algorithm at one and several frequencies")
{
    // Configurations {F, GR, per-link cluster number}; several frequencies use the paper's power coefficients and
    // the averaged scales, F = 2 with equal DS (r_tau - 1 rule) and equal ASD at both frequencies
    const int cfg[6][3] = {{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {3, 1, 0}, {2, 1, 0}};
    double err[3] = {0, 0, 0};
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
        for (const auto &c : cfg)
        {
            const bool los = std::string(name) == "UMi_LOS", gr = c[1] == 1;
            const arma::uword N = los ? N_UMi_LOS : N_UMi_NLOS, F = (arma::uword)c[0];
            INFO(name << ", F = " << F << ", GR " << gr << ", per-link cluster number " << c[2]);
            spmap sp = read_v19(name);
            sp["GR_enabled"] = gr;
            fix_t f = make_fix(N, 4, F, 301 + 10 * (unsigned)F + (unsigned)gr, true);
            f.use_eod = f.use_toa = true, f.use_eps = gr, f.use_ncl = c[2] == 1;
            if (c[2] == 1)
                f.ncl = arma::uvec{N, N - 1, 6, 9};
            for (arma::uword l = 0; l < 4 && F == 2; ++l)
                f.lsp(l, 1, 0) = f.lsp(l, 0, 0), f.lsp(l, 1, 3) = f.lsp(l, 0, 3);
            const out_t o = run(sp, f, "quadriga");
            REQUIRE(o.power.n_slices == F);
            REQUIRE(o.delay.n_slices == 1);
            for (arma::uword l = 0; l < 4; ++l)
            {
                const ref_t r = ref_qd(f, l, los ? r_UMi_LOS : r_UMi_NLOS, f.ncl(l), los, gr);
                check_link(o, f, l, r, los, gr, f.ncl(l), err);
                CHECK(o.n_active(l) == f.ncl(l));
                CHECK((los || r.delta != 0.0));
            }
        }
    QD_DIAG("\n# diag: quadriga, largest error: delay %.2e, power %.2e (relative), angle %.2e rad\n", err[0], err[1],
            err[2]);
}

TEST_CASE("ssf_clusters - quadriga delay spread equals the input at one frequency")
{
    QD_DIAG("\n# diag: quadriga DS from the output / input, 200 links\nscenario,gr,median_slot_powers,"
            "max_error_split_powers,clamped\n");
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
        for (int gr = 0; gr < 2; ++gr)
        {
            const bool los = std::string(name) == "UMi_LOS";
            const arma::uword N = los ? N_UMi_LOS : N_UMi_NLOS, n_link = 200;
            INFO(name << ", GR " << gr);
            spmap sp = read_v19(name);
            sp["GR_enabled"] = gr;
            fix_t f = make_fix(N, n_link, 1, 321 + gr, true);
            f.use_eps = gr == 1, f.use_toa = !los;
            out_t o = run(sp, f, "quadriga");
            std::vector<arma::vec> d(n_link), p(n_link), ps(n_link);
            for (arma::uword l = 0; l < n_link; ++l)
            {
                d[l] = rv(o.delay, l), p[l] = rv(o.power, l), ps[l] = p[l];
                if (gr) // split powers of the scaling: LOS path minus the ground reflection
                    ps[l](0) -= ps[l](1);
            }
            const arma::vec ds = quadriga_lib::calc_delay_spread(d, p, 1000.0);
            const arma::vec ds_split = quadriga_lib::calc_delay_spread(d, ps, 1000.0);
            int n_clamped = 0;
            double e_max = 0.0;
            arma::vec ratio(n_link);
            for (arma::uword l = 0; l < n_link; ++l)
            {
                ratio(l) = ds(l) / f.lsp(l, 0, 0);
                if (ref_qd(f, l, los ? r_UMi_LOS : r_UMi_NLOS, N, los, gr == 1).clamped)
                {
                    ++n_clamped;
                    continue;
                }
                const double e = std::abs(ds_split(l) / f.lsp(l, 0, 0) - 1.0);
                e_max = std::max(e_max, e);
                CHECK(e <= 1e-10);
                if (!gr || !los)
                    CHECK(std::abs(ratio(l) - 1.0) <= 1e-10);
            }
            CHECK(n_clamped < 20);
            QD_DIAG("%s,%d,%.4f,%.2e,%d\n", name, gr, arma::median(ratio), e_max, n_clamped);
        }
}

TEST_CASE("ssf_clusters - quadriga angular spreads equal the input below the caps")
{
    QD_DIAG("\n# diag: quadriga linear AS of the offsets / input, 200 links, AS <= 20 deg, ES <= 10 deg\n"
            "scenario,gr,checked_ASD,checked_ASA,checked_ESD,checked_ESA,not_converged,max_rel_error\n");
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
        for (int gr = 0; gr < 2; ++gr)
        {
            const bool los = std::string(name) == "UMi_LOS";
            const arma::uword N = los ? N_UMi_LOS : N_UMi_NLOS, n_link = 200;
            INFO(name << ", GR " << gr);
            spmap sp = read_v19(name);
            sp["GR_enabled"] = gr;
            fix_t f = make_fix(N, n_link, 1, 331 + gr, true, 0.0, 10.0, 5.0, 20.0, 1.0, 10.0);
            f.use_eps = gr == 1, f.use_toa = true;
            out_t o = run(sp, f, "quadriga");
            int n_chk[4] = {0, 0, 0, 0}, n_nc = 0;
            double e_max = 0.0;
            for (arma::uword l = 0; l < n_link; ++l)
            {
                const ref_t r = ref_qd(f, l, los ? r_UMi_LOS : r_UMi_NLOS, N, los, gr == 1);
                const geo_t g = geometry(f, l);
                const double cen[4] = {g.ang[0], g.ang[1], r.eps_d, r.eps_a};
                const arma::cube *out[4] = {&o.aod, &o.aoa, &o.eod, &o.eoa};
                for (int a = 0; a < 4; ++a)
                {
                    if (r.n_fold_a[a] > 0 || r.s_raw[a] >= (a < 2 ? 2.0 : 1.5))
                        continue; // wrap, pole fold or cap
                    if (!r.conv[a])
                    {
                        ++n_nc; // the fixed LOS and ground-reflection elevations alone exceed the target spread
                        continue;
                    }
                    ++n_chk[a];
                    // Weights: LOS path (minus the ground reflection), ground reflection, clusters
                    std::vector<double> x, w;
                    const double p_gr = gr ? o.power(l, 1, 0) : 0.0;
                    x.push_back(a < 2 ? 0.0 : g.ang[a] - cen[a]), w.push_back(o.power(l, 0, 0) - p_gr);
                    if (gr)
                        x.push_back(a < 2 ? 0.0 : g.e_gr - cen[a]), w.push_back(p_gr);
                    for (arma::uword n = 0; n < N; ++n)
                        x.push_back(wrap_rad((*out[a])(l, 1 + gr + n, 0) - cen[a])), w.push_back(o.power(l, 1 + gr + n, 0));
                    const double e = std::abs(lin_spread(x, w) / (f.lsp(l, 0, 3 + a) * pi / 180.0) - 1.0);
                    e_max = std::max(e_max, e);
                    CHECK(e <= 1e-4);
                }
            }
            for (int a = 0; a < 4; ++a)
                CHECK(n_chk[a] >= (gr && los && a > 1 ? 10 : 100)); // with GR in LOS most elevation targets are unreachable
            QD_DIAG("%s,%d,%d,%d,%d,%d,%d,%.2e\n", name, gr, n_chk[0], n_chk[1], n_chk[2], n_chk[3], n_nc, e_max);
        }
}

TEST_CASE("ssf_clusters - quadriga limits the scale to 3 in azimuth and 1.5 in elevation")
{
    const spmap sp = read_v19("UMi_NLOS");
    const arma::uword N = N_UMi_NLOS;
    fix_t f = make_fix(N, 8, 1, 341, true);
    f.use_toa = true;
    for (arma::uword l = 0; l < 8; ++l)
        f.lsp(l, 0, 3) = f.lsp(l, 0, 4) = 300.0, f.lsp(l, 0, 5) = f.lsp(l, 0, 6) = 200.0;
    out_t o = run(sp, f, "quadriga");
    int n_el = 0;
    for (arma::uword l = 0; l < 8; ++l)
    {
        const ref_t r = ref_qd(f, l, r_UMi_NLOS, N, false, false);
        CHECK(r.s_raw[0] > 3.0);
        CHECK(r.s_raw[1] > 3.0);
        CHECK(r.s_raw[2] > 1.5);
        CHECK(r.s_raw[3] > 1.5);
        const geo_t g = geometry(f, l);
        const arma::cube *out[4] = {&o.aod, &o.aoa, &o.eod, &o.eoa};
        for (int a = 0; a < 4; ++a)
            for (arma::uword n = 0; n < N; ++n)
            {
                const double at = (0.5 * std::erfc(-f.rnd(l, (5 + a) * N + n) / std::sqrt(2.0)) - 0.5) * pi;
                const double rec = (*out[a])(l, 1 + n, 0) - g.ang[a];
                if (a < 2)
                    CHECK(std::abs(wrap_rad(rec - 3.0 * at)) <= 1e-12);
                else if (std::abs(1.5 * at) <= 0.5 * pi && std::abs(g.ang[a] + 1.5 * at) < 0.5 * pi) // no pole fold
                {
                    CHECK(std::abs(rec - 1.5 * at) <= 1e-12);
                    ++n_el;
                }
            }
    }
    CHECK(n_el >= 50);
}

TEST_CASE("ssf_clusters - quadriga applies the LOS direction additively")
{
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
    {
        const bool los = std::string(name) == "UMi_LOS";
        const arma::uword N = los ? N_UMi_LOS : N_UMi_NLOS;
        INFO(name);
        spmap sp = read_v19(name);
        sp["GR_enabled"] = 1.0;
        fix_t f = make_fix(N, 4, 1, 351, true);
        f.use_eps = f.use_toa = true;
        // Link 0 nearly below the BS: EoD -80 deg, EoA +80 deg; the NLOS elevations fold at the poles
        f.rx(0, 0) = f.tx(0, 0) + 3.0, f.rx(0, 1) = f.tx(0, 1) - 2.6, f.rx(0, 2) = 1.5;
        f.lsp(0, 0, 5) = f.lsp(0, 0, 6) = 25.0;
        out_t o = run(sp, f, "quadriga");
        double err[3] = {0, 0, 0};
        for (arma::uword l = 0; l < 4; ++l)
        {
            const ref_t r = ref_qd(f, l, los ? r_UMi_LOS : r_UMi_NLOS, N, los, true);
            check_link(o, f, l, r, los, true, N, err);
            if (l == 0)
                CHECK(r.n_fold > 0);
        }

        // Rotating the positions about the z-axis shifts every azimuth by the rotation angle and changes nothing else
        fix_t f2 = f;
        const double al = 0.7, ca = std::cos(al), sa = std::sin(al);
        for (arma::mat *m : {&f2.tx, &f2.rx})
            for (arma::uword i = 0; i < m->n_rows; ++i)
            {
                const double x = (*m)(i, 0), y = (*m)(i, 1);
                (*m)(i, 0) = ca * x - sa * y, (*m)(i, 1) = sa * x + ca * y;
            }
        out_t o2 = run(sp, f2, "quadriga");
        CHECK(arma::approx_equal(o2.power, o.power, "reldiff", 1e-12));
        CHECK(arma::approx_equal(o2.delay, o.delay, "absdiff", 1e-18));
        CHECK(arma::approx_equal(o2.eod, o.eod, "absdiff", 1e-12));
        CHECK(arma::approx_equal(o2.eoa, o.eoa, "absdiff", 1e-12));
        for (arma::uword i = 0; i < o.aod.n_elem; ++i)
        {
            CHECK(std::abs(wrap_rad(o2.aod(i) - o.aod(i) - al)) <= 1e-12);
            CHECK(std::abs(wrap_rad(o2.aoa(i) - o.aoa(i) - al)) <= 1e-12);
        }
    }
}

TEST_CASE("ssf_clusters - quadriga adds the EoD offset to the NLOS departures only")
{
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
    {
        const bool los = std::string(name) == "UMi_LOS";
        const arma::uword N = los ? N_UMi_LOS : N_UMi_NLOS;
        INFO(name);
        spmap sp = read_v19(name);
        sp["GR_enabled"] = 1.0;
        fix_t f = make_fix(N, 4, 2, 361, true);
        f.use_eps = f.use_toa = true;
        const out_t a = run(sp, f, "quadriga");
        f.use_eod = true;
        const out_t b = run(sp, f, "quadriga");
        if (los)
        {
            CHECK(equal(a, b));
            continue;
        }
        CHECK(eq(a.delay, b.delay));
        CHECK(eq(a.power, b.power));
        CHECK(eq(a.aod, b.aod));
        CHECK(eq(a.aoa, b.aoa));
        CHECK(eq(a.eoa, b.eoa));
        int n_chk = 0;
        for (arma::uword l = 0; l < 4; ++l)
        {
            const double delta = 0.5 * (f.eod(l, 0) + f.eod(l, 1)) * pi / 180.0;
            CHECK(a.eod(l, 0, 0) == b.eod(l, 0, 0)); // LOS and ground reflection unchanged
            CHECK(a.eod(l, 1, 0) == b.eod(l, 1, 0));
            for (arma::uword n = 0; n < N; ++n)
                if (std::abs(a.eod(l, 2 + n, 0)) < 0.5 * pi - 0.1) // no pole fold
                {
                    CHECK(std::abs(b.eod(l, 2 + n, 0) - a.eod(l, 2 + n, 0) - delta) <= 1e-12);
                    ++n_chk;
                }
        }
        CHECK(n_chk >= 50);
    }
}

TEST_CASE("ssf_clusters - quadriga outputs coefficient powers with the ground reflection")
{
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
    {
        const bool los = std::string(name) == "UMi_LOS";
        const arma::uword N = los ? N_UMi_LOS : N_UMi_NLOS;
        INFO(name);
        spmap sp = read_v19(name);
        sp["GR_enabled"] = 1.0;
        fix_t f = make_fix(N, 4, 2, 371, true);
        f.use_eps = f.use_toa = true;
        // Initial elevation offsets 0: the NLOS elevations are the centres
        for (arma::uword c = 7 * N; c < 9 * N; ++c)
            f.rnd.col(c).zeros();
        out_t o = run(sp, f, "quadriga");
        double err[3] = {0, 0, 0};
        for (arma::uword l = 0; l < 4; ++l)
        {
            const geo_t g = geometry(f, l);
            for (arma::uword k = 0; k < 2; ++k)
            {
                const double K = los ? f.lsp(l, k, 1) : 0.0;
                CHECK(std::abs(o.power(l, 0, k) - K / (1.0 + K)) <= 1e-12 * K / (1.0 + K));
                CHECK(std::abs(o.power(l, 1, k) - K / (1.0 + K) * g.R2[k]) <= 1e-12 * K / (1.0 + K) * g.R2[k]);
                double sum = 0.0;
                for (arma::uword n = 0; n < N; ++n)
                    sum += o.power(l, 2 + n, k);
                CHECK(std::abs(sum - 1.0 / (1.0 + K)) <= 1e-12);
            }
            const double om = los ? 0.5 * (g.R2[0] + g.R2[1]) : 0.0;
            for (arma::uword n = 0; n < N; ++n)
            {
                CHECK(std::abs(o.eod(l, 2 + n, 0) - ((1.0 - om) * g.ang[2] + om * g.e_gr)) <= 1e-12);
                CHECK(std::abs(o.eoa(l, 2 + n, 0) - ((1.0 - om) * g.ang[3] + om * g.e_gr)) <= 1e-12);
            }
            check_link(o, f, l, ref_qd(f, l, los ? r_UMi_LOS : r_UMi_NLOS, N, los, true), los, true, N, err);
        }
    }
}

TEST_CASE("ssf_clusters - quadriga places the first NLOS cluster at the absolute TOA in NLOS scenarios")
{
    // Scenarios that define the absolute TOA (absTOA_mu > -30): abs_toa is required, the NLOS delays start at
    // abs_toa(l), also where it is 0. Other scenarios (absTOA_mu <= -30): shifted only where abs_toa(l) > 0. LOS
    // scenarios: abs_toa is ignored
    const arma::uword N = N_UMi_NLOS;
    const spmap sp = read_v19("UMi_NLOS");
    spmap sl = sp;
    sl["absTOA_mu"] = -100.0;
    fix_t f = make_fix(N, 4, 1, 381, true);
    f.toa(3) = 0.0;
    CHECK_THROWS_AS(run(sp, f, "quadriga"), std::invalid_argument);
    const out_t a = run(sl, f, "quadriga"); // no shift
    f.use_toa = true;
    const out_t b = run(sp, f, "quadriga"), c = run(sl, f, "quadriga");
    CHECK(eq(a.power, b.power));
    CHECK(eq(a.aod, b.aod));
    CHECK(eq(a.eod, b.eod));
    CHECK(eq(b.power, c.power));
    for (arma::uword l = 0; l < 4; ++l)
    {
        INFO("link " << l);
        CHECK(b.delay(l, 0, 0) == 0.0);
        const arma::vec da = rv(a.delay, l).subvec(1, N), db = rv(b.delay, l).subvec(1, N);
        CHECK(db.min() == f.toa(l));
        for (arma::uword n = 0; n < N; ++n)
            CHECK(std::abs(db(n) - (da(n) - da.min() + f.toa(l))) <= 1e-15 * db.max());
        CHECK(arma::all(rv(c.delay, l) == rv(l < 3 ? b.delay : a.delay, l)));
    }

    const spmap so = read_v19("UMi_LOS");
    fix_t g = make_fix(N_UMi_LOS, 4, 1, 382, true);
    const out_t x = run(so, g, "quadriga");
    g.use_toa = true;
    CHECK(equal(run(so, g, "quadriga"), x));
}

TEST_CASE("ssf_clusters - quadriga gives delay 0 to clusters without delay spread")
{
    // One active NLOS cluster (RMa, NumClusters_min not set, n_cluster = 1) has no delay spread to scale: delay 0, or
    // abs_toa in an NLOS scenario; in LOS also with K = 0 and with the ground reflection
    for (const char *name : {"RMa_NLOS", "RMa_LOS"})
        for (int gr = 0; gr < 2; ++gr)
        {
            INFO(name << ", GR " << gr);
            const bool los = std::string(name) == "RMa_LOS";
            spmap sp = read_v19(name);
            sp["GR_enabled"] = gr;
            fix_t f = make_fix((arma::uword)sp["NumClusters"] - 1, 4, 2, 391, true);
            f.use_ncl = f.use_toa = true, f.use_eps = gr == 1;
            f.ncl.fill(1);
            if (los)
                f.lsp.slice(1).zeros();
            const out_t o = run(sp, f, "quadriga");
            check_finite(o);
            for (arma::uword l = 0; l < 4; ++l)
                CHECK(o.delay(l, 1 + gr, 0) == (los ? 0.0 : f.toa(l)));
            if (los)
                continue;
            sp["absTOA_mu"] = -100.0, f.use_toa = false;
            const out_t p = run(sp, f, "quadriga");
            CHECK(arma::all(slots(p.delay, 1 + gr, 1 + gr) == 0.0));
        }
}

TEST_CASE("ssf_clusters - quadriga is continuous along tracks")
{
    // 5 tracks of 20 m at 1 cm steps; LSPs and random values from the library at the track positions
    QD_DIAG("\n# diag: quadriga along 5 tracks x 20 m at 1 cm: events (angle > 10 deg, delay > DS/4, power > 3 dB) "
            "of clusters within 20 dB of the strongest\nscenario,angle,delay,power,cluster_samples\n");
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
    {
        const spmap sp = read_v19(name);
        const arma::uword n_pt = 2001;
        int ev[4] = {0, 0, 0, 0};
        for (int tr = 0; tr < 5; ++tr)
        {
            arma::mat tx = {{0.0, 0.0, 10.0}}, rx(n_pt, 3);
            const double a0 = 1.3 * tr - 2.0, x0 = 30.0 + 25.0 * tr, y0 = 15.0 - 9.0 * tr;
            for (arma::uword i = 0; i < n_pt; ++i)
                rx(i, 0) = x0 + 0.01 * i * std::cos(a0), rx(i, 1) = y0 + 0.01 * i * std::sin(a0), rx(i, 2) = 1.5;
            const lib_t in = lib_inputs(sp, tx, rx, 11 + tr);
            out_t o;
            quadriga_lib::ssf_clusters(sp, tx, rx, in.lsp, in.rnd, o.delay, o.power, o.aod, o.eod, o.aoa, o.eoa,
                                       "quadriga", nullptr, nullptr, &in.eod, &in.toa);
            track_events(o, in.lsp, ev);
        }
        QD_DIAG("%s,%d,%d,%d,%d\n", name, ev[0], ev[1], ev[2], ev[3]);
        CHECK(ev[0] == 0);
        CHECK(ev[1] == 0);
        CHECK(ev[2] == 0);
        CHECK(ev[3] > 10000);
    }
}

TEST_CASE("ssf_clusters - quadriga is swap-reciprocal with dual-mobility fields")
{
    double e_max = 0.0;
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
        for (double h : {1.5, 25.0})
        {
            const bool los = std::string(name) == "UMi_LOS";
            const arma::uword N = los ? N_UMi_LOS : N_UMi_NLOS;
            INFO(name << ", heights 1.5 and " << h << " m");
            spmap sp = read_v19(name);
            sp["GR_enabled"] = 1.0;
            fix_t f = make_fix(N, 6, 2, 401, true);
            f.use_eps = f.use_toa = true;
            f.tx.col(2).fill(h), f.rx.col(2).fill(1.5);
            arma::cube ph;
            quadriga_lib::ssf_init_phases(sp, ph, 23, -1);
            quadriga_lib::ssf_draw(sp, ph, f.tx, f.rx, f.rnd, -1, nullptr, true);
            fix_t s = f;
            s.tx = f.rx, s.rx = f.tx;
            quadriga_lib::ssf_draw(sp, ph, s.tx, s.rx, s.rnd, -1, nullptr, true);
            s.lsp.slice(3) = f.lsp.slice(4), s.lsp.slice(4) = f.lsp.slice(3);
            s.lsp.slice(5) = f.lsp.slice(6), s.lsp.slice(6) = f.lsp.slice(5);
            const out_t a = run(sp, f, "quadriga"), b = run(sp, s, "quadriga");
            CHECK(arma::approx_equal(a.delay, b.delay, "reldiff", 1e-10));
            CHECK(arma::approx_equal(a.power, b.power, "reldiff", 1e-10));
            for (arma::uword i = 0; i < a.aod.n_elem; ++i)
            {
                const double e = std::max({std::abs(wrap_rad(a.aod(i) - b.aoa(i))), std::abs(wrap_rad(a.aoa(i) - b.aod(i))),
                                           std::abs(a.eod(i) - b.eoa(i)), std::abs(a.eoa(i) - b.eod(i))});
                e_max = std::max(e_max, e);
                CHECK(e <= 1e-10);
            }
        }
    QD_DIAG("\n# diag: quadriga swap reciprocity, largest angle difference %.2e rad\n", e_max);
}
