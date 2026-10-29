// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "*ssf_clusters*"

// Verification suite for the cluster step of the small-scale model, quadriga_lib::ssf_clusters: the part shared by
// all methods and the method "38.901" (test cases "ssf_clusters - ..."; the other methods have their own files).
//
// Pins down, for all methods: the output sizes and the slot layout (slot 0 LOS path, slot 1 ground reflection if
// enabled, then the N = NumClusters - 1 NLOS clusters), the LOS geometry of slot 0, the ground-reflection slot (TR
// clause 7.6.8: delay, angles, coefficient power with the polarisation-averaged reflection power), the LOS condition
// of the scenario, the independence of the links, the LOS-only case NumClusters = 1, finite outputs for extreme
// random values, the inputs a method ignores, empty input, pre-sized outputs and the input validation.
// For "38.901": TR 38.901 clause 7.5 steps 5 to 7 in NLOS and LOS (delays with C_tau, powers with the per-cluster
// shadowing, the K-factor split of eq. (7.5-30), the angle mapping with C_phi and C_theta, signs, offsets, the
// alignment of the first cluster to the LOS direction, the ZOD offset, the O2I mean ZOA, the zenith fold); the
// K-factor limit at -5 dB in the C factors; the -25 dB removal without renormalisation; the per-link cluster number
// (TR clause 7.6.15) in the C factors and the inactive slots; the absolute time of arrival (TR clause 7.6.9, NLOS
// only); the reflection coefficients and the consumer rule of the reflection slot; swap reciprocity (also
// "38.901_MF"); the spread statistics over a drop with the library's random fields.
// Oracles, independent of the function and of the conf files (fixture and geometry: qd_test_ssf.hpp):
//  - ref_tr: TR 38.901 clause 7.5 steps 5 to 7 typed here in scalar double, in degrees, with the TR's own forms
//    (P'_n = exp(...)·10^(-Z_n/10), P_n/max(P_n), the removal in dB, the zenith wrap of the TR text); the TR
//    constants of the scenario (r_tau, zeta and N of Table 7.5-6, Tables 7.5-2 and 7.5-4 with linear
//    interpolation between their entries, the K polynomials of eqs. (7.5-3), (7.5-10), (7.5-15)) are typed here;
//    the K-factor in the polynomials is limited to >= -5 dB
//  - the absolute time of arrival of TR eq. (7.6-43): NLOS delays plus the offset, eq. (7.6-44): no offset in LOS
//  - statistics: the medians of the output/input spread ratios in a window of +-0.03 around the values of the method
//    (UMa NLOS, 6 GHz); +-0.03 is more than 5 standard errors of the median of 2000 independent links; the spreads
//    of the output are computed with calc_delay_spread and calc_angular_spreads_sphere
// Tolerances: delays 1e-12 relative to max(delay, DS); powers 1e-12 relative; angles 1e-11 rad (the oracle orders the
// operations differently and the angles reach a few thousand degrees before the wrap); values that the function
// forms by the same operations are compared bitwise.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_tools.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "qd_test_diag.hpp"
#include "qd_test_ssf.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    // TR 38.901 Table 7.5-6: delay scaling r_tau, per-cluster shadowing std zeta [dB], number of clusters N
    struct tr_const
    {
        double r_tau, zeta;
        arma::uword N;
    };
    const tr_const UMa_NLOS = {2.3, 3.0, 20}, UMa_LOS = {2.5, 3.0, 12}, UMi_O2I = {2.2, 4.0, 12},
                   InF_NLOS = {3.0, 3.0, 25};

    // TR 38.901 Tables 7.5-2 (C_phi) and 7.5-4 (C_theta), linear interpolation between the entries, constant beyond
    const double tab_n_phi[14] = {4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 19, 20, 25};
    const double tab_c_phi[14] = {0.779, 0.860, 0.921, 0.973, 1.018, 1.090, 1.123,
                                  1.146, 1.190, 1.211, 1.226, 1.273, 1.289, 1.358};
    const double tab_n_theta[12] = {6, 7, 8, 10, 11, 12, 14, 15, 16, 19, 20, 25};
    const double tab_c_theta[12] = {0.788, 0.847, 0.889, 0.957, 1.031, 1.104, 1.1072, 1.1088, 1.1276, 1.184, 1.178, 1.282};
    double c_tab(const double *n, const double *c, int m, double N)
    {
        if (N <= n[0])
            return c[0];
        for (int i = 0; i + 1 < m; ++i)
            if (N <= n[i + 1])
                return c[i] + (c[i + 1] - c[i]) * (N - n[i]) / (n[i + 1] - n[i]);
        return c[m - 1];
    }

    // Oracle: TR 38.901 clause 7.5 steps 5 to 7 for link l, one frequency; outputs per cluster n < N_l
    // - kstar_db: K-factor in [dB] for the polynomials; NaN = max(10·lg(K), -5)
    struct ref_t
    {
        std::vector<double> delay, power, aod, eod, aoa, eoa; // power: coefficient power, 0 if removed
        double p_los = 0.0;
        arma::uword n_removed = 0;
    };
    ref_t ref_tr(const fix_t &f, arma::uword l, const tr_const &c, arma::uword N_l, bool los, double eod_off = 0.0,
                 bool o2i = false, double kstar_db = NAN)
    {
        const arma::uword N = c.N;
        const double DS = f.lsp(l, 0, 0), K = los ? f.lsp(l, 0, 1) : 0.0;
        const double AS[4] = {f.lsp(l, 0, 3), f.lsp(l, 0, 4), f.lsp(l, 0, 5), f.lsp(l, 0, 6)}; // ASD, ASA, ZSD, ZSA
        const geo_t g = geometry(f, l);
        const double los_ang[4] = {g.phi_d, g.phi_d + 180.0, g.theta_d, 180.0 - g.theta_d}; // AOD, AOA, ZOD, ZOA

        // Step 5
        std::vector<double> tp(N_l), tau(N_l);
        double tmin = 1e300;
        arma::uword first = 0;
        for (arma::uword n = 0; n < N_l; ++n)
        {
            tp[n] = -c.r_tau * DS * std::log(f.rnd(l, n));
            if (tp[n] < tmin)
                tmin = tp[n], first = n;
        }
        const double Kdb = 10.0 * std::log10(K);
        const double Ks = std::isnan(kstar_db) ? std::max(Kdb, -5.0) : kstar_db;
        const double C_tau = 0.7705 - 0.0433 * Ks + 0.0002 * Ks * Ks + 0.000017 * Ks * Ks * Ks;
        ref_t r;
        for (arma::uword n = 0; n < N_l; ++n)
        {
            tau[n] = tp[n] - tmin;
            r.delay.push_back(los ? tau[n] / C_tau : tau[n]);
        }

        // Step 6
        std::vector<double> P(N_l), Pa(N_l);
        double sum = 0.0, pmax = 0.0, pamax = 0.0;
        for (arma::uword n = 0; n < N_l; ++n)
            P[n] = std::exp(-tau[n] * (c.r_tau - 1.0) / (c.r_tau * DS)) * std::pow(10.0, -c.zeta * f.rnd_freq(l, n, 0) / 10.0),
            sum += P[n];
        for (arma::uword n = 0; n < N_l; ++n)
            P[n] /= sum, pmax = std::max(pmax, P[n]);
        for (arma::uword n = 0; n < N_l; ++n)
        {
            Pa[n] = los ? P[n] / (K + 1.0) + (n == first ? K / (K + 1.0) : 0.0) : P[n];
            pamax = std::max(pamax, Pa[n]);
            const bool removed = 10.0 * std::log10(P[n] / pmax) < -25.0;
            r.n_removed += removed ? 1 : 0;
            r.power.push_back(removed ? 0.0 : P[n] / (K + 1.0));
        }
        r.p_los = K / (K + 1.0);

        // Step 7
        const double C_phi = c_tab(tab_n_phi, tab_c_phi, 14, (double)N_l) *
                             (los ? 1.1035 - 0.028 * Ks - 0.002 * Ks * Ks + 0.0001 * Ks * Ks * Ks : 1.0);
        const double C_theta = c_tab(tab_n_theta, tab_c_theta, 12, (double)N_l) *
                               (los ? 1.3086 + 0.0339 * Ks - 0.0077 * Ks * Ks + 0.0002 * Ks * Ks * Ks : 1.0);
        std::vector<double> *dst[4] = {&r.aod, &r.aoa, &r.eod, &r.eoa};
        for (int x = 0; x < 4; ++x)
        {
            std::vector<double> v(N_l);
            for (arma::uword n = 0; n < N_l; ++n)
            {
                const double lr = std::log(Pa[n] / pamax);
                const double a = (x < 2) ? 2.0 * (AS[x] / 1.4) * std::sqrt(-lr) / C_phi : -AS[x] * lr / C_theta;
                const double X = (f.rnd(l, (1 + x) * N + n) >= 0.0) ? 1.0 : -1.0;
                const double Y = f.rnd(l, (5 + x) * N + n) * AS[x] / 7.0;
                v[n] = X * a + Y;
            }
            for (arma::uword n = 0; n < N_l; ++n)
            {
                double ang;
                if (los)
                    ang = v[n] - (v[first] - los_ang[x]); // eqs. (7.5-12), (7.5-17)
                else if (x == 2)
                    ang = v[n] + los_ang[2] + (-eod_off); // eq. (7.5-19), mu_offset,ZOD = -EoD offset
                else if (x == 3)
                    ang = v[n] + (o2i ? 90.0 : los_ang[3]); // eq. (7.5-16)
                else
                    ang = v[n] + los_ang[x]; // eq. (7.5-11)
                dst[x]->push_back(x < 2 ? wrap_rad(ang * pi / 180.0) : zen2elev(ang));
            }
        }
        return r;
    }

    // Compares link l of the output with the oracle; the NLOS clusters start at slot 1
    void check_link(const out_t &o, const ref_t &r, const fix_t &f, arma::uword l, double *max_err = nullptr)
    {
        const double DS = f.lsp(l, 0, 0);
        double e_d = 0.0, e_p = 0.0, e_a = 0.0;
        for (arma::uword n = 0; n < r.delay.size(); ++n)
        {
            const arma::uword s = 1 + n;
            INFO("link " << l << ", cluster " << n);
            e_d = std::max(e_d, std::abs(o.delay(l, s, 0) - r.delay[n]) / std::max(r.delay[n], DS));
            CHECK(std::abs(o.delay(l, s, 0) - r.delay[n]) <= 1e-12 * std::max(r.delay[n], DS));
            if (r.power[n] == 0.0)
                CHECK(o.power(l, s, 0) == 0.0);
            else
            {
                e_p = std::max(e_p, std::abs(o.power(l, s, 0) / r.power[n] - 1.0));
                CHECK(std::abs(o.power(l, s, 0) - r.power[n]) <= 1e-12 * r.power[n]);
            }
            const double da[4] = {wrap_rad(o.aod(l, s, 0) - r.aod[n]), wrap_rad(o.aoa(l, s, 0) - r.aoa[n]),
                                  o.eod(l, s, 0) - r.eod[n], o.eoa(l, s, 0) - r.eoa[n]};
            for (double d : da)
            {
                e_a = std::max(e_a, std::abs(d));
                CHECK(std::abs(d) <= 1e-11);
            }
        }
        CHECK(std::abs(o.power(l, 0, 0) - r.p_los) <= 1e-12 * r.p_los);
        if (max_err != nullptr)
            max_err[0] = std::max(max_err[0], e_d), max_err[1] = std::max(max_err[1], e_p),
            max_err[2] = std::max(max_err[2], e_a);
    }

    // Keep the links in 'idx'
    fix_t subset(const fix_t &f, const arma::uvec &idx)
    {
        fix_t s = f;
        s.tx = (f.tx.n_rows == 1) ? f.tx : arma::mat(f.tx.rows(idx));
        s.rx = f.rx.rows(idx), s.rnd = f.rnd.rows(idx), s.eod = f.eod.rows(idx);
        s.eps_re = f.eps_re.rows(idx), s.eps_im = f.eps_im.rows(idx), s.toa = f.toa.elem(idx), s.ncl = f.ncl.elem(idx);
        s.lsp.set_size(idx.n_elem, f.lsp.n_cols, f.lsp.n_slices), s.anchor.set_size(idx.n_elem, 1, 8);
        s.rnd_freq.set_size(idx.n_elem, f.rnd_freq.n_cols, f.rnd_freq.n_slices);
        s.o2i.clear();
        for (arma::uword i = 0; i < idx.n_elem; ++i)
        {
            s.lsp.row(i) = f.lsp.row(idx(i)), s.anchor.row(i) = f.anchor.row(idx(i));
            s.rnd_freq.row(i) = f.rnd_freq.row(idx(i));
            s.o2i.push_back(f.o2i[idx(i)]);
        }
        return s;
    }
}

TEST_CASE("ssf_clusters - output sizes, slot layout and the ground-reflection slot")
{
    for (const char *m : qd_methods)
        for (const char *name : {"UMi_LOS", "UMi_NLOS"})
            for (int gr = 0; gr < 2; ++gr)
            {
                const bool los = std::string(name) == "UMi_LOS", mf = std::string(m) == "38.901_MF";
                const bool qd = std::string(m) == "quadriga";
                const arma::uword N = los ? N_UMi_LOS : N_UMi_NLOS, F = (std::string(m) == "38.901") ? 1 : 3;
                const arma::uword n_fd = mf ? F : 1;
                INFO(m << ", " << name << ", GR " << gr);
                spmap sp = read_v19(name);
                fix_t f = make_fix(N, 5, F, 11, gr == 0);
                f.use_toa = true, f.use_eps = gr == 1;
                const out_t o_off = run(sp, f, m);
                sp["GR_enabled"] = gr;
                const out_t o = run(sp, f, m);
                for (const arma::cube *c : {&o.delay, &o.power, &o.aod, &o.eod, &o.aoa, &o.eoa})
                {
                    CHECK(c->n_rows == 5);
                    CHECK(c->n_cols == N + 1 + gr);
                    CHECK(c->n_slices == (c == &o.power ? F : n_fd));
                }
                CHECK(o.n_active.n_elem == 5);
                check_finite(o);
                check_ranges(o);
                for (arma::uword l = 0; l < 5; ++l)
                {
                    const geo_t g = geometry(f, l);
                    for (arma::uword k = 0; k < n_fd; ++k)
                    {
                        CHECK(o.delay(l, 0, k) == 0.0);
                        CHECK(std::abs(wrap_rad(o.aod(l, 0, k) - g.ang[0])) <= 1e-12);
                        CHECK(std::abs(wrap_rad(o.aoa(l, 0, k) - g.ang[1])) <= 1e-12);
                        CHECK(std::abs(o.eod(l, 0, k) - g.ang[2]) <= 1e-12);
                        CHECK(std::abs(o.eoa(l, 0, k) - g.ang[3]) <= 1e-12);
                        if (gr == 0)
                            continue;
                        CHECK(std::abs(o.delay(l, 1, k) - g.tau_gr) <= 1e-12 * g.tau_gr); // eqs. (7.6-33), (7.6-34)
                        CHECK(o.aod(l, 1, k) == o.aod(l, 0, k));                       // eqs. (7.6-37a) to (7.6-37d)
                        CHECK(o.aoa(l, 1, k) == o.aoa(l, 0, k));
                        CHECK(std::abs(o.eod(l, 1, k) - g.e_gr) <= 1e-12);
                        CHECK(std::abs(o.eoa(l, 1, k) - g.e_gr) <= 1e-12);
                    }
                    for (arma::uword k = 0; k < F; ++k)
                    {
                        const double K = los ? f.lsp(l, k, 1) : 0.0, p_los = K / (K + 1.0);
                        CHECK(std::abs(o.power(l, 0, k) - p_los) <= 1e-15);
                        if (gr == 0)
                            continue;
                        // TR methods K/(K+1)·(d3D/d_GR)²·R² (eq. (7.6-32)), "quadriga" K/(K+1)·R²
                        const double p = p_los * (qd ? 1.0 : (g.d3D / g.d_gr) * (g.d3D / g.d_gr)) * g.R2[k];
                        CHECK(std::abs(o.power(l, 1, k) - p) <= 1e-12 * p);
                    }
                }
                // TR methods: the NLOS slots as without the reflection, shifted by one slot
                const arma::cube *on[6] = {&o.delay, &o.power, &o.aod, &o.eod, &o.aoa, &o.eoa},
                                 *off[6] = {&o_off.delay, &o_off.power, &o_off.aod, &o_off.eod, &o_off.aoa, &o_off.eoa};
                for (int i = 0; i < 6 && gr == 1 && !qd; ++i)
                    for (arma::uword k = 0; k < n_fd; ++k)
                        CHECK(arma::all(slots(*on[i], 2, N + 1, k) == slots(*off[i], 1, N, k)));
            }
}

TEST_CASE("ssf_clusters - 38.901 NLOS matches TR 7.5")
{
    const spmap sp = read_v19("UMa_NLOS");
    fix_t f = make_fix(UMa_NLOS.N, 4, 1, 21, true, -8.0, 20.0);
    f.use_eod = true;
    out_t o = run(sp, f);
    double err[3] = {0, 0, 0};
    for (arma::uword l = 0; l < 4; ++l)
    {
        ref_t r = ref_tr(f, l, UMa_NLOS, UMa_NLOS.N, false, f.eod(l, 0));
        check_link(o, r, f, l, err);
        CHECK(o.power(l, 0, 0) == 0.0);
        CHECK(o.n_active(l) == UMa_NLOS.N - r.n_removed);
    }
    QD_DIAG("# diag: 38.901 NLOS max error: delay %.2e rel, power %.2e rel, angle %.2e rad\n", err[0], err[1], err[2]);

    // The EoD offset moves the NLOS departure elevations only
    f.use_eod = false;
    out_t o0 = run(sp, f);
    CHECK(eq(o0.aod, o.aod));
    CHECK(eq(o0.eoa, o.eoa));
    CHECK(arma::any(arma::vectorise(o0.eod != o.eod)));
}

TEST_CASE("ssf_clusters - 38.901 LOS matches TR 7.5")
{
    const spmap sp = read_v19("UMa_LOS");
    double err[3] = {0, 0, 0};
    for (double kdb : {3.0, 9.0, 15.0})
    {
        fix_t f = make_fix(UMa_LOS.N, 4, 1, 31, true, kdb, kdb);
        f.use_eod = true; // no effect in LOS
        out_t o = run(sp, f);
        for (arma::uword l = 0; l < 4; ++l)
        {
            INFO("K = " << kdb << " dB");
            ref_t r = ref_tr(f, l, UMa_LOS, UMa_LOS.N, true);
            check_link(o, r, f, l, err);

            // The first cluster (smallest initial delay) is at delay 0 in the LOS direction
            arma::uword first = 0;
            for (arma::uword n = 1; n < UMa_LOS.N; ++n)
                if (f.rnd(l, n) > f.rnd(l, first))
                    first = n;
            CHECK(o.delay(l, 1 + first, 0) == 0.0);
            CHECK(std::abs(wrap_rad(o.aod(l, 1 + first, 0) - o.aod(l, 0, 0))) <= 1e-12);
            CHECK(std::abs(wrap_rad(o.aoa(l, 1 + first, 0) - o.aoa(l, 0, 0))) <= 1e-12);
            CHECK(std::abs(o.eod(l, 1 + first, 0) - o.eod(l, 0, 0)) <= 1e-12);
            CHECK(std::abs(o.eoa(l, 1 + first, 0) - o.eoa(l, 0, 0)) <= 1e-12);

            // NLOS powers sum to 1/(K+1) unless a cluster was removed
            const double K = f.lsp(l, 0, 1);
            const double s = arma::accu(o.power(arma::span(l), arma::span(1, UMa_LOS.N), arma::span(0)));
            if (r.n_removed == 0)
                CHECK(std::abs(s - 1.0 / (K + 1.0)) <= 1e-12);
        }
    }
    QD_DIAG("# diag: 38.901 LOS max error: delay %.2e rel, power %.2e rel, angle %.2e rad\n", err[0], err[1], err[2]);
}

TEST_CASE("ssf_clusters - 38.901 limits the K-factor to -5 dB in the C factors")
{
    const spmap sp = read_v19("UMa_LOS");
    fix_t f = make_fix(UMa_LOS.N, 4, 1, 41, true, -8.0, -8.0);
    out_t o = run(sp, f);
    for (arma::uword l = 0; l < 4; ++l)
    {
        // Polynomials at -5 dB, LOS power and the 7.5-8 split at -8 dB
        check_link(o, ref_tr(f, l, UMa_LOS, UMa_LOS.N, true, 0.0, false, -5.0), f, l);
        const double K = std::pow(10.0, -0.8);
        CHECK(std::abs(o.power(l, 0, 0) - K / (K + 1.0)) <= 1e-15);

        // Without the limit the delays would differ (C_tau at -8 dB)
        ref_t r8 = ref_tr(f, l, UMa_LOS, UMa_LOS.N, true, 0.0, false, -8.0);
        bool differs = false;
        for (arma::uword n = 0; n < UMa_LOS.N; ++n)
            differs = differs || std::abs(r8.delay[n] - o.delay(l, 1 + n, 0)) > 1e-3 * r8.delay[n];
        CHECK(differs);
    }
}

TEST_CASE("ssf_clusters - 38.901 removes clusters 25 dB below the strongest without renormalisation")
{
    // Delays from a narrow range (power ratio of the exponential PDP above -3.3 dB), z = 0 except two clusters with
    // z = 9 (27 dB per-cluster shadowing): exactly these two fall below -25 dB of the powers of eq. (7.5-6)
    for (const char *name : {"UMa_NLOS", "UMa_LOS"})
    {
        const bool los = std::string(name) == "UMa_LOS";
        const tr_const c = los ? UMa_LOS : UMa_NLOS;
        const spmap sp = read_v19(name);
        fix_t f = make_fix(c.N, 3, 1, 51, false, 10.0, 15.0);
        for (arma::uword l = 0; l < 3; ++l)
            for (arma::uword n = 0; n < c.N; ++n)
            {
                f.rnd(l, n) = 0.5 + 0.4 * (double)((n * 7 + l * 3) % c.N) / (double)c.N;
                f.rnd_freq(l, n, 0) = (n == 2 + l || n == 7 + l) ? 9.0 : 0.0;
            }
        out_t o = run(sp, f);
        for (arma::uword l = 0; l < 3; ++l)
        {
            INFO(name << ", link " << l);
            ref_t r = ref_tr(f, l, c, c.N, los);
            CHECK(r.n_removed == 2);
            check_link(o, r, f, l);
            CHECK(o.power(l, 1 + 2 + l, 0) == 0.0);
            CHECK(o.power(l, 1 + 7 + l, 0) == 0.0);
            CHECK(o.delay(l, 1 + 2 + l, 0) > 0.0); // delay and angles kept (oracle above)
            CHECK(o.n_active(l) == c.N - 2);
            const double s = arma::accu(o.power(arma::span(l), arma::span(1, c.N), arma::span(0)));
            const double K = los ? f.lsp(l, 0, 1) : 0.0;
            CHECK(s < 1.0 / (K + 1.0));
            CHECK(s > 0.99 / (K + 1.0));
        }
    }
}

TEST_CASE("ssf_clusters - 38.901 uses the per-link cluster number in the C factors")
{
    // Oracle interpolation against the examples C_phi(9) = 1.054, C_phi(13) = 1.168, C_theta(9) = 0.923
    CHECK(std::abs(c_tab(tab_n_phi, tab_c_phi, 14, 9.0) - 1.054) < 1e-12);
    CHECK(std::abs(c_tab(tab_n_phi, tab_c_phi, 14, 13.0) - 1.168) < 1e-12);
    CHECK(std::abs(c_tab(tab_n_theta, tab_c_theta, 12, 9.0) - 0.923) < 1e-12);
    CHECK(c_tab(tab_n_theta, tab_c_theta, 12, 30.0) == 1.282);

    spmap sp = read_v19("InF_NLOS_SL");
    fix_t f = make_fix(InF_NLOS.N, 5, 1, 61, true);
    f.use_ncl = true;
    f.ncl = {9, 13, 17, 18, 21};
    out_t o = run(sp, f);
    for (arma::uword l = 0; l < 5; ++l)
    {
        ref_t r = ref_tr(f, l, InF_NLOS, f.ncl(l), false);
        check_link(o, r, f, l);
        CHECK(o.n_active(l) == f.ncl(l) - r.n_removed);
        check_inactive(o, l, 1 + f.ncl(l));
    }

    // 30 clusters: C beyond the last table entry stays at the value for 25
    sp["NumClusters"] = 31.0;
    const tr_const c30 = {InF_NLOS.r_tau, InF_NLOS.zeta, 30};
    fix_t g = make_fix(30, 3, 1, 62, true);
    out_t o30 = run(sp, g);
    for (arma::uword l = 0; l < 3; ++l)
        check_link(o30, ref_tr(g, l, c30, 30, false), g, l);
}

TEST_CASE("ssf_clusters - 38.901 O2I links use 90 degrees as mean ZOA")
{
    const spmap sp = read_v19("UMi_NLOS_O2I");
    fix_t f = make_fix(UMi_O2I.N, 4, 1, 71, true);
    f.use_o2i = true;
    out_t o = run(sp, f);
    for (arma::uword l = 0; l < 4; ++l)
        check_link(o, ref_tr(f, l, UMi_O2I, UMi_O2I.N, false, 0.0, f.o2i[l]), f, l);

    // Without the flags all links use the LOS zenith
    f.use_o2i = false;
    out_t o0 = run(sp, f);
    CHECK(arma::all(rv(o0.eoa, 1) == rv(o.eoa, 1)));
    CHECK(arma::any(rv(o0.eoa, 0) != rv(o.eoa, 0)));
    CHECK(eq(o0.eod, o.eod));
}

TEST_CASE("ssf_clusters - 38.901 folds zenith angles at 0 and 180 degrees")
{
    // Offsets of +-25 standard deviations with ZSD, ZSA of 40 deg drive the zenith angles beyond 180 and below 0
    for (const char *name : {"UMa_NLOS", "UMa_LOS"})
    {
        const bool los = std::string(name) == "UMa_LOS";
        const tr_const c = los ? UMa_LOS : UMa_NLOS;
        const spmap sp = read_v19(name);
        fix_t f = make_fix(c.N, 3, 1, 81, true, 5.0, 10.0);
        for (arma::uword l = 0; l < 3; ++l)
        {
            f.lsp(l, 0, 5) = 40.0, f.lsp(l, 0, 6) = 40.0;
            for (arma::uword n = 0; n < c.N; ++n)
                f.rnd(l, 7 * c.N + n) = (n % 2 == 0) ? 25.0 : -25.0, f.rnd(l, 8 * c.N + n) = (n % 3 == 0) ? -25.0 : 25.0;
        }
        out_t o = run(sp, f);
        for (arma::uword l = 0; l < 3; ++l)
            check_link(o, ref_tr(f, l, c, c.N, los), f, l);
        check_ranges(o);
    }
}

TEST_CASE("ssf_clusters - 38.901 adds the absolute TOA to the NLOS delays of NLOS scenarios")
{
    // NLOS scenario: TR eq. (7.6-43), every NLOS delay plus the offset; the first cluster is at the offset
    const spmap sp = read_v19("UMa_NLOS");
    fix_t f = make_fix(UMa_NLOS.N, 3, 1, 91, true);
    f.toa = {120e-9, 37.5e-9, 0.0};
    f.use_ncl = true;
    f.ncl = {20, 17, 16};
    out_t o0 = run(sp, f);
    f.use_toa = true;
    out_t o = run(sp, f);
    for (arma::uword l = 0; l < 3; ++l)
    {
        INFO("link " << l);
        CHECK(o.delay(l, 0, 0) == 0.0);
        double dmin = 1e300;
        for (arma::uword n = 0; n < UMa_NLOS.N; ++n)
            if (n < f.ncl(l))
            {
                CHECK(o.delay(l, 1 + n, 0) == o0.delay(l, 1 + n, 0) + f.toa(l));
                dmin = std::min(dmin, o.delay(l, 1 + n, 0));
            }
            else
                CHECK(o.delay(l, 1 + n, 0) == 0.0);
        CHECK(dmin == f.toa(l));
    }
    CHECK(eq(o.power, o0.power));
    CHECK(eq(o.aod, o0.aod));

    // Zero offsets equal the omitted input
    f.toa.zeros();
    CHECK(equal(run(sp, f), o0));

    // LOS scenario: TR eq. (7.6-44), no offset
    const spmap sl = read_v19("UMa_LOS");
    fix_t g = make_fix(UMa_LOS.N, 3, 1, 92, true);
    out_t a = run(sl, g);
    g.use_toa = true;
    CHECK(equal(run(sl, g), a));
}

TEST_CASE("ssf_clusters - 38.901 ground reflection: reflection coefficients and coefficient rule")
{
    spmap sp = read_v19("UMa_LOS");
    sp["GR_enabled"] = 1.0;
    fix_t f = make_fix(UMa_LOS.N, 4, 1, 101, true, 3.0, 15.0);
    f.use_eps = true;
    out_t o = run(sp, f);
    double err_r = 0.0;
    for (arma::uword l = 0; l < 4; ++l)
    {
        INFO("link " << l);
        const geo_t g = geometry(f, l);
        const double K = f.lsp(l, 0, 1), R2 = g.R2[0];

        // R² is the same for both signs of the imaginary part
        std::complex<double> R_par, R_perp;
        ref_r(g.d2D, g.h_t, g.h_r, f.eps_re(l, 0), f.eps_im(l, 0), R_par, R_perp, 1.0);
        const double R2c = 0.5 * (std::norm(R_par) + std::norm(R_perp));
        err_r = std::max(err_r, std::abs(R2c / R2 - 1.0));
        CHECK(std::abs(R2c - R2) <= 1e-14 * R2);
        CHECK(R2 > 0.0);
        CHECK(R2 < 1.0);

        // Consumer rule: sqrt(slot power)·diag(R_par, -R_perp)/R equals the coefficient of eq. (7.6-32),
        // sqrt(K/(K+1))·(d3D/d_GR)·diag(R_par, -R_perp), for both polarisations
        ref_r(g.d2D, g.h_t, g.h_r, f.eps_re(l, 0), f.eps_im(l, 0), R_par, R_perp);
        const double amp = std::sqrt(o.power(l, 1, 0)) / std::sqrt(R2), amp_tr = std::sqrt(K / (K + 1.0)) * g.d3D / g.d_gr;
        CHECK(std::abs(amp * R_par - amp_tr * R_par) <= 1e-12 * std::abs(amp_tr * R_par));
        CHECK(std::abs(amp * R_perp - amp_tr * R_perp) <= 1e-12 * std::abs(amp_tr * R_perp));
        CHECK(arma::accu(rv(o.power, l)) > 1.0); // no renormalisation
    }
    QD_DIAG("# diag: GR R2 sign max difference %.2e rel\n", err_r);

    // NLOS scenario: zero reflection power, geometry unchanged
    spmap sn = read_v19("UMa_NLOS");
    sn["GR_enabled"] = 1.0;
    fix_t g = make_fix(UMa_NLOS.N, 4, 1, 102, true);
    g.use_eps = true;
    out_t on = run(sn, g);
    CHECK(arma::all(slots(on.power, 0, 1) == 0.0));
    CHECK(arma::all(slots(on.delay, 1, 1) > 0.0));

    // Both ends on the ground (grazing): eps = 1 gives R = 0, eps = 15 - 1j gives R_par = R_perp = -1
    fix_t h = make_fix(UMa_LOS.N, 2, 1, 103, true, 9.0, 9.0);
    h.use_eps = true;
    h.tx.col(2).zeros(), h.rx.col(2).zeros();
    h.eps_re.fill(1.0), h.eps_im.fill(0.0);
    out_t og = run(sp, h);
    check_finite(og);
    CHECK(og.power(0, 1, 0) == 0.0);
    CHECK(og.power(1, 1, 0) == 0.0);
    CHECK(og.delay(0, 1, 0) == 0.0);
    h.eps_re.fill(15.0), h.eps_im.fill(1.0);
    og = run(sp, h);
    check_finite(og);
    for (arma::uword l = 0; l < 2; ++l)
        CHECK(std::abs(og.power(l, 1, 0) - og.power(l, 0, 0)) <= 1e-15);
}

TEST_CASE("ssf_clusters - NLOS scenarios give zero LOS and ground-reflection power whatever the K-factor in lsp")
{
    spmap sp = read_v19("UMi_NLOS");
    sp["GR_enabled"] = 1.0;
    for (const char *m : qd_methods)
    {
        INFO(m);
        fix_t f = make_fix(N_UMi_NLOS, 4, std::string(m) == "38.901" ? 1 : 2, 111, true);
        f.use_eps = f.use_toa = true;
        f.lsp.slice(1).fill(1e-10);
        const out_t a = run(sp, f, m);
        f.lsp.slice(1).fill(5.0);
        CHECK(equal(run(sp, f, m), a));
        CHECK(arma::all(arma::vectorise(a.power.cols(0, 1)) == 0.0));
    }
}

TEST_CASE("ssf_clusters - 38.901 and 38.901_MF are swap-reciprocal for mirrored inputs")
{
    // Mirrored random values and LSPs, EoD offset 0 and no O2I link (38.901_MF ignores both)
    for (const char *m : {"38.901", "38.901_MF"})
        for (const char *name : {"UMa_LOS", "UMa_NLOS"})
        {
            const arma::uword N = (std::string(name) == "UMa_LOS") ? UMa_LOS.N : UMa_NLOS.N;
            const spmap sp = read_v19(name);
            const fix_t f = make_fix(N, 6, std::string(m) == "38.901" ? 1 : 2, 121, true);
            const out_t a = run(sp, f, m), b = run(sp, swapped(f), m);
            INFO(m << ", " << name);
            double e = 0.0;
            for (arma::uword i = 0; i < a.delay.n_elem; ++i)
            {
                CHECK(std::abs(a.delay(i) - b.delay(i)) <= 1e-12 * f.anchor(i % 6, 0, 0));
                CHECK(std::abs(a.power(i) - b.power(i)) <= 1e-12 * a.power(i));
                const double d[4] = {wrap_rad(a.aod(i) - b.aoa(i)), wrap_rad(a.aoa(i) - b.aod(i)), a.eod(i) - b.eoa(i),
                                     a.eoa(i) - b.eod(i)};
                for (double x : d)
                {
                    e = std::max(e, std::abs(x));
                    CHECK(std::abs(x) <= 1e-12);
                }
            }
            CHECK(arma::all(a.n_active == b.n_active));
            QD_DIAG("# diag: swap %s %s, max angle difference %.2e rad\n", m, name, e);
        }
}

TEST_CASE("ssf_clusters - a link does not depend on the other links")
{
    for (const char *m : qd_methods)
        for (const char *name : {"UMi_LOS", "UMi_NLOS"})
        {
            INFO(m << ", " << name);
            const arma::uword N = (std::string(name) == "UMi_LOS") ? N_UMi_LOS : N_UMi_NLOS;
            spmap sp = read_v19(name);
            sp["GR_enabled"] = 1.0;
            fix_t f = make_fix(N, 50, std::string(m) == "38.901" ? 1 : 2, 131, true);
            f.use_eod = f.use_toa = f.use_eps = f.use_ncl = f.use_o2i = true;
            for (arma::uword l = 0; l < 50; ++l)
                f.ncl(l) = 6 + l % (N - 5);
            const out_t all = run(sp, f, m);
            for (arma::uword l : {0, 17, 49})
            {
                const out_t one = run(sp, subset(f, arma::uvec{l}), m);
                const out_t row = {all.delay.rows(l, l), all.power.rows(l, l), all.aod.rows(l, l),
                                   all.eod.rows(l, l),   all.aoa.rows(l, l),   all.eoa.rows(l, l),
                                   all.n_active.rows(l, l)};
                CHECK(equal(one, row));
            }
        }
}

TEST_CASE("ssf_clusters - 38.901 spread statistics over a drop")
{
    // UMa NLOS at 6 GHz, 5 seeds x 400 UTs at d2D 35 to 300 m; all decorrelation distances 0 (fields at 0.1 m), so
    // the links are independent. Expected medians of output/input: DS [0.94, 1.00], ASD [0.99, 1.05], ASA
    // [0.96, 1.02], ESD [0.95, 1.01], ESA [0.96, 1.02] (each the value of the method +-0.03, more than 5 standard
    // errors of the median of 2000 independent links)
    spmap sp = read_v19("UMa_NLOS");
    for (const char *k : {"SC_lambda", "DS_lambda", "KF_lambda", "SF_lambda", "AS_D_lambda", "AS_A_lambda",
                          "ES_D_lambda", "ES_A_lambda", "XPR_lambda", "absTOA_lambda"})
        sp[k] = 0.0;
    std::vector<double> ratio[5];
    const arma::mat tx = {{0.0, 0.0, 25.0}};
    for (arma::sword seed = 1; seed <= 5; ++seed)
    {
        const arma::mat rx = drop_rx((unsigned)seed, 400);
        const lib_t in = lib_inputs(sp, tx, rx, seed);
        out_t o;
        quadriga_lib::ssf_clusters(sp, tx, rx, in.lsp, in.rnd, o.delay, o.power, o.aod, o.eod, o.aoa, o.eoa, "38.901",
                                   &in.rnd_freq, nullptr, &in.eod, &in.toa);
        spread_ratios(o, in.lsp, ratio);
    }
    const char *lbl[5] = {"DS", "ASD", "ASA", "ESD", "ESA"};
    const double lo[5] = {0.94, 0.99, 0.96, 0.95, 0.96}, hi[5] = {1.00, 1.05, 1.02, 1.01, 1.02};
    QD_DIAG("\n# diag: 38.901 UMa NLOS 6 GHz, 2000 links, median output/input spread\nspread,median,window_lo,window_hi\n");
    for (int i = 0; i < 5; ++i)
    {
        const double m = arma::median(arma::vec(ratio[i]));
        QD_DIAG("%s,%.4f,%.2f,%.2f\n", lbl[i], m, lo[i], hi[i]);
        INFO(lbl[i] << " median " << m);
        CHECK(m >= lo[i]);
        CHECK(m <= hi[i]);
    }
}

TEST_CASE("ssf_clusters - one cluster gives the LOS path only")
{
    for (const char *m : qd_methods)
        for (const char *name : {"UMa_LOS", "UMa_NLOS"})
            for (int gr = 0; gr < 2; ++gr)
            {
                INFO(m << ", " << name << ", GR " << gr);
                const bool qd = std::string(m) == "quadriga";
                const arma::uword F = std::string(m) == "38.901" ? 1 : 2, n_fd = std::string(m) == "38.901_MF" ? F : 1;
                spmap sp = read_v19(name);
                sp["NumClusters"] = 1.0, sp["NumClusters_min"] = 0.0, sp["GR_enabled"] = gr;
                fix_t f = make_fix(0, 4, F, 141, true);
                f.use_rf = f.use_anchor = false; // not needed without NLOS clusters
                f.use_eps = gr == 1, f.use_ncl = true;
                f.ncl.zeros();
                const out_t o = run(sp, f, m);
                for (const arma::cube *c : {&o.delay, &o.power, &o.aod, &o.eod, &o.aoa, &o.eoa})
                {
                    CHECK(c->n_cols == (arma::uword)(1 + gr));
                    CHECK(c->n_slices == (c == &o.power ? F : n_fd));
                }
                for (arma::uword l = 0; l < 4; ++l)
                {
                    const geo_t g = geometry(f, l);
                    CHECK(o.n_active(l) == 0);
                    for (arma::uword k = 0; k < n_fd; ++k)
                    {
                        CHECK(o.delay(l, 0, k) == 0.0);
                        CHECK(std::abs(wrap_rad(o.aod(l, 0, k) - g.ang[0])) <= 1e-12);
                        CHECK(std::abs(o.eoa(l, 0, k) - g.ang[3]) <= 1e-12);
                        if (gr)
                        {
                            CHECK(std::abs(o.delay(l, 1, k) - g.tau_gr) <= 1e-12 * g.tau_gr);
                            CHECK(std::abs(o.eod(l, 1, k) - g.e_gr) <= 1e-12);
                        }
                    }
                    for (arma::uword k = 0; k < F; ++k)
                    {
                        CHECK(o.power(l, 0, k) == 1.0); // power 1 in all scenarios
                        const double p = (qd ? 1.0 : (g.d3D / g.d_gr) * (g.d3D / g.d_gr)) * (gr ? g.R2[k] : 0.0);
                        if (gr)
                            CHECK(std::abs(o.power(l, 1, k) - p) <= 1e-12 * p);
                    }
                }
                // A 0x0 matrix is accepted as the empty set
                f.rnd.reset();
                CHECK(equal(run(sp, f, m), o));
            }
}

TEST_CASE("ssf_clusters - extreme finite random values give finite powers")
{
    for (const char *m : qd_methods)
        for (const char *name : {"UMa_LOS", "UMa_NLOS"})
        {
            const bool los = std::string(name) == "UMa_LOS", tr = std::string(m) == "38.901";
            const arma::uword N = los ? UMa_LOS.N : UMa_NLOS.N, F = tr ? 1 : 2;
            INFO(m << ", " << name);
            spmap sp = read_v19(name);
            sp["GR_enabled"] = 1.0;
            fix_t f = make_fix(N, 3, F, 151, true);
            f.use_eps = f.use_toa = f.use_eod = true;

            // z = -1100 on one cluster at the last frequency: 10^(330) overflows in the linear form of eqs. (7.5-5),
            // (7.6.30a); "quadriga" has no shadowing
            for (arma::uword l = 0; l < 3; ++l)
                f.rnd_freq(l, 3 + l, F - 1) = -1100.0;
            out_t o = run(sp, f, m);
            check_finite(o);
            for (arma::uword l = 0; l < 3 && std::string(m) != "quadriga"; ++l)
            {
                const double K = los ? f.lsp(l, F - 1, 1) : 0.0;
                CHECK(std::abs(o.power(l, 2 + 3 + l, F - 1) - 1.0 / (K + 1.0)) <= 1e-15);
            }

            // Normal values at +-40 (erfc underflows in the Laplacian term of 38.901_MF), delay values 0 and 1
            for (arma::uword c = 0; c < 9 * N; ++c)
                for (arma::uword l = 0; l < 3; ++l)
                    f.rnd(l, c) = (c < N) ? (double)((c + l) % 2) : ((c + l) % 2 ? 40.0 : -40.0);
            for (arma::uword i = 0; i < f.rnd_freq.n_elem; ++i)
                f.rnd_freq(i) = (i % 3 == 0) ? 40.0 : -40.0;
            o = run(sp, f, m);
            check_finite(o);
            CHECK(arma::all(arma::vectorise(o.power) >= 0.0));
            for (arma::uword l = 0; l < 3; ++l) // NLOS powers sum to 1/(K+1) (38.901: at most, -25 dB removal)
                for (arma::uword k = 0; k < F; ++k)
                {
                    const double K = los ? f.lsp(l, k, 1) : 0.0;
                    const double s = arma::accu(o.power(arma::span(l), arma::span(2, N + 1), arma::span(k)));
                    CHECK(s <= 1.0 / (1.0 + K) + 1e-12);
                    if (!tr)
                        CHECK(std::abs(s - 1.0 / (1.0 + K)) <= 1e-12);
                }
        }
}

TEST_CASE("ssf_clusters - each method ignores the inputs it does not use")
{
    // Unused inputs: 38.901 lsp_anchor; 38.901_MF eod_offset_deg, o2i; quadriga rnd_freq, lsp_anchor, o2i. Valid
    // values do not change the output, invalid sizes or values are not checked
    for (const char *m : qd_methods)
        for (const char *name : {"UMi_LOS", "UMi_NLOS"})
        {
            const std::string ms = m;
            const bool tr = ms == "38.901", mf = ms == "38.901_MF", qd = ms == "quadriga";
            INFO(m << ", " << name);
            spmap sp = read_v19(name);
            sp["GR_enabled"] = 1.0;
            fix_t f = make_fix(std::string(name) == "UMi_LOS" ? N_UMi_LOS : N_UMi_NLOS, 3, tr ? 1 : 2, 291, true);
            f.use_eps = f.use_toa = f.use_ncl = true;
            f.use_rf = !qd, f.use_anchor = mf, f.use_eod = !mf, f.use_o2i = tr;
            const out_t a = run(sp, f, m);
            fix_t g = f; // unused inputs, valid
            g.use_rf = g.use_anchor = g.use_eod = g.use_o2i = true;
            CHECK(equal(run(sp, g, m), a));
            if (!tr) // invalid
            {
                g.o2i.assign(5, false);
                if (mf)
                    g.eod.set_size(1, 7), g.eod.fill(NAN);
                if (qd)
                    g.rnd_freq.set_size(3, 2, 1), g.anchor.set_size(2, 2, 2);
            }
            else
                g.anchor.set_size(1, 1, 1), g.anchor.zeros();
            CHECK(equal(run(sp, g, m), a));
        }

    // NumClusters = 1: rnd_freq is ignored; GR off: the permittivity is ignored
    spmap s = read_v19("UMa_LOS");
    s["NumClusters"] = 1.0, s["NumClusters_min"] = 0.0;
    fix_t x = make_fix(0, 3, 1, 172, true);
    x.rnd_freq = arma::cube(5, 5, 5, arma::fill::value(NAN));
    x.use_eps = true;
    x.eps_re.fill(0.0);
    CHECK_NOTHROW(run(s, x));
}

TEST_CASE("ssf_clusters - empty input and pre-sized outputs")
{
    spmap sp = read_v19("UMa_LOS");
    sp["GR_enabled"] = 1.0;
    const arma::uword N = UMa_LOS.N;

    // No link
    fix_t e = make_fix(N, 0, 1, 161, false);
    e.use_eps = e.use_eod = e.use_toa = e.use_o2i = e.use_ncl = true;
    out_t o = run(sp, e);
    for (const arma::cube *c : {&o.delay, &o.power, &o.aod, &o.eod, &o.aoa, &o.eoa})
        CHECK(arma::size(*c) == arma::size(0, N + 2, 1));
    CHECK(o.n_active.n_elem == 0);

    // Pre-sized outputs with NaN are written completely; wrong sizes are resized
    fix_t f = make_fix(N, 4, 1, 162, true);
    f.use_eps = f.use_ncl = true;
    f.ncl = {N, 11, 12, 11};
    const out_t ref = run(sp, f);
    for (const arma::uword r : {4, 2})
    {
        out_t p;
        for (arma::cube *c : {&p.delay, &p.power, &p.aod, &p.eod, &p.aoa, &p.eoa})
            c->set_size(r, r == 4 ? N + 2 : 3, r == 4 ? 1 : 4), c->fill(NAN);
        p.n_active.set_size(r == 4 ? 4 : 7), p.n_active.fill(99);
        quadriga_lib::ssf_clusters(sp, f.tx, f.rx, f.lsp, f.rnd, p.delay, p.power, p.aod, p.eod, p.aoa, p.eoa,
                                   "38.901", &f.rnd_freq, nullptr, nullptr, nullptr, &f.eps_re, &f.eps_im, nullptr,
                                   &f.ncl, &p.n_active);
        CHECK(equal(p, ref));
    }
}

TEST_CASE("ssf_clusters - input validation")
{
    // UMa LOS: N = 12, NumClusters_min = 11, so N_l in [10, 12]. Each case changes the valid inputs (38.901, one
    // frequency, 3 links) and expects the message; positions and scenario parameters are checked by
    // qd_gbsm_check_links and scenpar_validate (one pass-through case each)
    const spmap sp = read_v19("UMa_LOS");
    const arma::uword N = UMa_LOS.N;
    const fix_t f0 = make_fix(N, 3, 1, 171, true);
    typedef std::function<void(spmap &, fix_t &, std::string &)> mut_t;
    auto gr = [](spmap &s, fix_t &x) { s["GR_enabled"] = 1.0, x.use_eps = true; };
    const mut_t nlos_qd = [](spmap &s, fix_t &x, std::string &m)
    { s = read_v19("UMa_NLOS"), x = make_fix(UMa_NLOS.N, 3, 1, 173, true), m = "quadriga"; };
    std::vector<std::pair<mut_t, const char *>> bad = {
        {[](spmap &, fix_t &, std::string &m) { m = "QuaDRiGa"; },
         "Input 'method' must be '38.901', '38.901_MF' or 'quadriga'."},
        {[](spmap &, fix_t &x, std::string &) { x.rx = arma::mat(3, 2, arma::fill::ones); },
         "Input 'rx_pos' must have 3 columns."},
        {[](spmap &s, fix_t &, std::string &) { s["NumClusters"] = 0.0; },
         "Value of 'NumClusters' must be an integer from 1 to 1000."},
        {[](spmap &, fix_t &x, std::string &) { x.lsp = arma::cube(3, 1, 7, arma::fill::ones); },
         "Input 'lsp' must have size [n_link, n_freq, 8]."},
        {[](spmap &, fix_t &x, std::string &) { x.lsp = arma::cube(2, 1, 8, arma::fill::ones); },
         "Input 'lsp' must have size [n_link, n_freq, 8]."},
        {[](spmap &, fix_t &x, std::string &) { x.lsp = arma::cube(3, 0, 8); },
         "Input 'lsp' must have size [n_link, n_freq, 8]."},
        {[](spmap &, fix_t &x, std::string &) { x.lsp(1, 0, 7) = INFINITY; }, "Input 'lsp' must be finite."},
        {[](spmap &, fix_t &x, std::string &) { x.lsp(0, 0, 1) = -1e-3; }, "K-factor in 'lsp' must be >= 0."},
        {[](spmap &, fix_t &x, std::string &) { x.lsp.resize(3, 2, 8), x.lsp.cols(1, 1) = x.lsp.cols(0, 0); },
         "Method '38.901' supports one frequency only; use '38.901_MF' or 'quadriga' for several frequencies."},
        {[N](spmap &, fix_t &x, std::string &) { x.rnd = x.rnd.cols(0, 9 * N - 2); },
         "Input 'rnd' must have size [n_link, 108]."},
        {[](spmap &, fix_t &x, std::string &) { x.rnd = x.rnd.rows(0, 1); },
         "Input 'rnd' must have size [n_link, 108]."},
        {[N](spmap &, fix_t &x, std::string &) { x.rnd(1, 5 * N) = NAN; }, "Input 'rnd' must be finite."},
        {[N](spmap &, fix_t &x, std::string &) { x.rnd(2, N - 1) = 1.0 + 1e-12; },
         "Columns 0 to N-1 of 'rnd' (cluster delays) must be in [0, 1]."},
        {[N](spmap &, fix_t &x, std::string &) { x.rnd(2, N - 1) = -1e-300; },
         "Columns 0 to N-1 of 'rnd' (cluster delays) must be in [0, 1]."},
        {[](spmap &, fix_t &x, std::string &) { x.use_rf = false; },
         "Input 'rnd_freq' is required for method '38.901'."},
        {[](spmap &, fix_t &x, std::string &m) { x.use_rf = false, m = "38.901_MF"; },
         "Input 'rnd_freq' is required for method '38.901_MF'."},
        {[N](spmap &, fix_t &x, std::string &) { x.rnd_freq = arma::cube(3, N, 2, arma::fill::zeros); },
         "Input 'rnd_freq' must have size [n_link, 12, n_freq]."},
        {[N](spmap &, fix_t &x, std::string &) { x.rnd_freq = arma::cube(3, N + 1, 1, arma::fill::zeros); },
         "Input 'rnd_freq' must have size [n_link, 12, n_freq]."},
        {[](spmap &, fix_t &x, std::string &) { x.rnd_freq(2, 3, 0) = NAN; }, "Input 'rnd_freq' must be finite."},
        {[](spmap &, fix_t &x, std::string &m) { x.use_anchor = false, m = "38.901_MF"; },
         "Input 'lsp_anchor' is required for method '38.901_MF'."},
        {[](spmap &, fix_t &x, std::string &m) { x.anchor = arma::cube(3, 2, 8, arma::fill::ones), m = "38.901_MF"; },
         "Input 'lsp_anchor' must have size [n_link, 1, 8]."},
        {[](spmap &, fix_t &x, std::string &m) { x.anchor(1, 0, 2) = NAN, m = "38.901_MF"; },
         "Input 'lsp_anchor' must be finite."},
        {[](spmap &, fix_t &x, std::string &) { x.use_eod = true, x.eod = arma::mat(3, 2, arma::fill::zeros); },
         "Input 'eod_offset_deg' must have size [n_link, n_freq]."},
        {[](spmap &, fix_t &x, std::string &) { x.use_eod = true, x.eod(1, 0) = NAN; },
         "Input 'eod_offset_deg' must be finite."},
        {[](spmap &, fix_t &x, std::string &) { x.use_toa = true, x.toa = arma::vec(2, arma::fill::zeros); },
         "Number of elements in 'abs_toa' does not match number of rows in 'rx_pos'."},
        {[](spmap &, fix_t &x, std::string &) { x.use_toa = true, x.toa(1) = -1e-12; },
         "Input 'abs_toa' must be finite and >= 0."},
        {[](spmap &, fix_t &x, std::string &) { x.use_toa = true, x.toa(1) = NAN; },
         "Input 'abs_toa' must be finite and >= 0."},
        {nlos_qd,
         "Method 'quadriga' requires 'abs_toa' in NLOS scenarios that define the absolute TOA ('absTOA_mu' > -30): the "
         "NLOS clusters start at the excess delay of TR 38.901 clause 7.6.9."},
        {[gr](spmap &s, fix_t &x, std::string &) { gr(s, x), x.use_eps = false; },
         "Inputs 'gr_eps_re' and 'gr_eps_im' are required when 'GR_enabled' is set."},
        {[gr](spmap &s, fix_t &x, std::string &) { gr(s, x), x.eps_im = arma::mat(3, 2, arma::fill::ones); },
         "Inputs 'gr_eps_re' and 'gr_eps_im' must have size [n_link, n_freq]."},
        {[gr](spmap &s, fix_t &x, std::string &) { gr(s, x), x.eps_re(0, 0) = INFINITY; },
         "Inputs 'gr_eps_re' and 'gr_eps_im' must be finite."},
        {[gr](spmap &s, fix_t &x, std::string &) { gr(s, x), x.eps_re(1, 0) = 0.999; },
         "Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0."},
        {[gr](spmap &s, fix_t &x, std::string &) { gr(s, x), x.eps_im(2, 0) = -1e-9; },
         "Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0."},
        {[gr](spmap &s, fix_t &x, std::string &) { gr(s, x), x.rx(2, 2) = -0.1; },
         "Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'."},
        {[gr](spmap &s, fix_t &x, std::string &) { gr(s, x), x.tx(0, 2) = -3.0; },
         "Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'."},
        {[](spmap &, fix_t &x, std::string &) { x.use_o2i = true, x.o2i = {true, false}; },
         "Number of elements in 'o2i' does not match number of rows in 'rx_pos'."},
        {[](spmap &, fix_t &x, std::string &) { x.use_ncl = true, x.ncl = {12, 12}; },
         "Number of elements in 'n_cluster' does not match number of rows in 'rx_pos'."},
        {[](spmap &, fix_t &x, std::string &) { x.use_ncl = true, x.ncl = {12, 9, 10}; },
         "Entries of 'n_cluster' must be in [10, 12]."},
        {[](spmap &, fix_t &x, std::string &) { x.use_ncl = true, x.ncl = {13, 10, 10}; },
         "Entries of 'n_cluster' must be in [10, 12]."},
        {[](spmap &s, fix_t &x, std::string &) { s["NumClusters_min"] = 0.0, x.use_ncl = true, x.ncl = {0, 1, 12}; },
         "Entries of 'n_cluster' must be in [1, 12]."}};
    for (arma::uword s : {0, 3, 4, 5, 6})
    {
        bad.push_back({[s](spmap &, fix_t &x, std::string &) { x.lsp(2, 0, s) = 0.0; },
                       "Delay and angular spreads in 'lsp' must be > 0."});
        bad.push_back({[s](spmap &, fix_t &x, std::string &m) { x.anchor(1, 0, s) = 0.0, m = "38.901_MF"; },
                       "Delay and angular spreads in 'lsp_anchor' must be > 0."});
    }
    const std::vector<mut_t> good = {
        [](spmap &, fix_t &, std::string &) {},
        [gr](spmap &s, fix_t &x, std::string &) { gr(s, x), x.use_eod = x.use_toa = x.use_o2i = x.use_ncl = true; },
        // Delay values at the bounds, block 1 is Normal; slices 2 and 7 of lsp are only checked for finiteness
        [N](spmap &, fix_t &x, std::string &) { x.rnd(2, N - 1) = 0.0, x.rnd(1, 0) = 1.0, x.rnd(0, N) = 5.0; },
        [](spmap &, fix_t &x, std::string &) { x.lsp(0, 0, 2) = -1.0, x.lsp(1, 0, 7) = 0.0; },
        [gr](spmap &s, fix_t &x, std::string &) { gr(s, x), x.eps_re(1, 0) = 1.0, x.eps_im(2, 0) = 0.0; }, // vacuum
        [](spmap &, fix_t &x, std::string &) { x.rx(2, 2) = -0.1, x.tx(0, 2) = -3.0; }, // z < 0 without the reflection
        [](spmap &, fix_t &x, std::string &) { x.use_ncl = true, x.ncl = {12, 10, 11}; },
        [](spmap &s, fix_t &x, std::string &) { s["NumClusters_min"] = 0.0, x.use_ncl = true, x.ncl = {1, 1, 12}; },
        [nlos_qd](spmap &s, fix_t &x, std::string &m) { nlos_qd(s, x, m), x.use_toa = true; },
        [nlos_qd](spmap &s, fix_t &x, std::string &m) { nlos_qd(s, x, m), s["absTOA_mu"] = -100.0; }};
    for (size_t i = 0; i < bad.size() + good.size(); ++i)
    {
        spmap s = sp;
        fix_t x = f0;
        std::string m = "38.901";
        (i < bad.size() ? bad[i].first : good[i - bad.size()])(s, x, m);
        INFO("case " << i << ", method " << m);
        if (i < bad.size())
            CHECK_THROWS_MATCHES(run(s, x, m), std::invalid_argument, Message(bad[i].second));
        else
            CHECK_NOTHROW(run(s, x, m));
    }

    // The outputs are untouched on error
    out_t o;
    o.delay = arma::cube(2, 2, 2, arma::fill::value(7.0)), o.power = o.delay;
    fix_t x = f0;
    x.lsp(0, 0, 0) = -1.0;
    CHECK_THROWS_AS(quadriga_lib::ssf_clusters(sp, x.tx, x.rx, x.lsp, x.rnd, o.delay, o.power, o.aod, o.eod, o.aoa,
                                               o.eoa, "38.901", &x.rnd_freq),
                    std::invalid_argument);
    CHECK(arma::all(arma::vectorise(o.delay) == 7.0));
    CHECK(arma::all(arma::vectorise(o.power) == 7.0));
}
