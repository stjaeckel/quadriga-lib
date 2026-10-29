// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "*ssf_clusters*"

// Verification suite for the cluster step of the small-scale model, quadriga_lib::ssf_clusters: the part shared by
// all methods and the method "38.901" (test cases "ssf_clusters - ..."; the other methods have their own files).
//
// Pins down: the output sizes and the slot layout (slot 0 LOS path, slot 1 ground reflection if enabled, then the
// N = NumClusters - 1 NLOS clusters); the LOS geometry of slot 0; the method "38.901" against TR 38.901 clause 7.5
// steps 5 to 7 in NLOS and LOS (delays with C_tau, powers with the per-cluster shadowing, the K-factor split of
// eq. (7.5-30), the angle mapping with C_phi and C_theta, signs, offsets, the alignment of the first cluster to the
// LOS direction, the ZOD offset, the O2I mean ZOA, the zenith fold); the K-factor limit at -5 dB in the C factors;
// the -25 dB removal without renormalisation; the per-link cluster number (TR clause 7.6.15) in the C factors and
// the inactive slots; the absolute time of arrival (TR clause 7.6.9, NLOS only); the ground reflection (TR clause
// 7.6.8: delay, angles, coefficient power with the polarisation-averaged reflection power, no renormalisation); the
// LOS condition of the scenario; swap reciprocity; the independence of the links; the spread statistics over a drop
// with the library's random fields; the LOS-only case NumClusters = 1; finite outputs for extreme random values;
// empty input, pre-sized outputs and the input validation.
// Oracles, independent of the function and of the conf files:
//  - ref_tr: TR 38.901 clause 7.5 steps 5 to 7 typed here in scalar double, in degrees, with the TR's own forms
//    (P'_n = exp(...)·10^(-Z_n/10), P_n/max(P_n), the removal in dB, the zenith wrap of the TR text); the TR
//    constants of the scenario (r_tau, zeta and N of Table 7.5-6, Tables 7.5-2 and 7.5-4 with linear
//    interpolation between their entries, the K polynomials of eqs. (7.5-3), (7.5-10), (7.5-15)) are typed here;
//    the K-factor in the polynomials is limited to >= -5 dB
//  - the LOS direction from the positions: azimuth atan2(dy, dx), zenith acos(dz / d3D) (TR clause 7.5 step 1c)
//  - the ground reflection of TR eqs. (7.6-32) to (7.6-40): delay (d_GR - d3D)/c, zenith 180 - atan(d2D/(h_t + h_r)),
//    R_par and R_perp with the TR's trigonometric forms, eps = eps_re - j·eps_im; slot power
//    K/(K+1)·(d3D/d_GR)²·(abs(R_par)² + abs(R_perp)²)/2
//  - the absolute time of arrival of TR eq. (7.6-43): NLOS delays plus the offset, eq. (7.6-44): no offset in LOS
//  - statistics: the medians of the output/input spread ratios of a comparison study of the method (UMa NLOS,
//    6 GHz) with a window of +-0.03; the spreads of the output are computed with calc_delay_spread and
//    calc_angular_spreads_sphere
// Tolerances: delays 1e-12 relative to max(delay, DS); powers 1e-12 relative; angles 1e-11 rad (the oracle orders the
// operations differently and the angles reach a few thousand degrees before the wrap); values that the function
// forms by the same operations are compared bitwise.
// Conventions: positions are rows {x, y, z} in meters; the input fixtures have distinct values per link and cluster,
// positions off the axes, and Tx positions per link where the case allows it.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_tools.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Message;
using Catch::Matchers::MessageMatches;

namespace
{
    typedef std::unordered_map<std::string, double> spmap;
    const double pi = 3.141592653589793;
    const double c0 = 299792458.0;

    spmap read_v19(const std::string &name)
    {
        spmap sp;
        quadriga_lib::conf_file_read("data/config/3GPP_38.901_v19_" + name + ".conf", sp);
        return sp;
    }

    // TR 38.901 Table 7.5-6: delay scaling r_tau, per-cluster shadowing std zeta [dB], number of clusters N
    struct tr_const
    {
        double r_tau, zeta;
        arma::uword N;
    };
    const tr_const UMa_NLOS = {2.3, 3.0, 20}, UMa_LOS = {2.5, 3.0, 12}, UMi_LOS = {3.0, 3.0, 12},
                   UMi_NLOS = {2.1, 3.0, 19}, UMi_O2I = {2.2, 4.0, 12}, InF_NLOS = {3.0, 3.0, 25};

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

    double wrap_deg(double x) { return x - 360.0 * std::floor((x + 180.0) / 360.0); }
    double wrap_rad(double x) { return x - 2.0 * pi * std::floor((x + pi) / (2.0 * pi)); }

    // Zenith angle in degrees to elevation in rad: wrapped to [0, 360], [180, 360] -> 360 - theta (TR 7.5 step 7)
    double zen2elev(double t)
    {
        t = t - 360.0 * std::floor(t / 360.0);
        if (t >= 180.0)
            t = 360.0 - t;
        return (90.0 - t) * pi / 180.0;
    }

    // Input fixture of one call
    struct fix_t
    {
        arma::mat tx, rx, rnd, eod, eps_re, eps_im;
        arma::cube lsp, rnd_freq;
        arma::vec toa;
        std::vector<bool> o2i;
        arma::uvec ncl;
        bool use_rf = true, use_eod = false, use_toa = false, use_eps = false, use_o2i = false, use_ncl = false;
    };

    // Distinct values: BS at 25 m, UTs 1.5 to 22.5 m, d2D 35 to 300 m; DS 40 to 400 ns, AS 5 to 60 deg, ZS 1 to
    // 20 deg, K in [kdb_lo, kdb_hi] dB; delay values in (0.02, 0.98), Normal values in (-2.5, 2.5)
    fix_t make_fix(arma::uword N, arma::uword n_link, unsigned seed, bool tx_per_link, double kdb_lo = -8.0,
                   double kdb_hi = 20.0)
    {
        std::mt19937 g(seed);
        auto U = [&](double a, double b)
        { return a + (b - a) * ((double)g() + 0.5) / 4294967296.0; };
        fix_t f;
        f.tx.set_size(tx_per_link ? n_link : 1, 3);
        for (arma::uword t = 0; t < f.tx.n_rows; ++t)
            f.tx(t, 0) = U(-40.0, 40.0), f.tx(t, 1) = U(-40.0, 40.0), f.tx(t, 2) = 25.0;
        f.rx.set_size(n_link, 3);
        f.lsp.zeros(n_link, 1, 8);
        for (arma::uword l = 0; l < n_link; ++l)
        {
            const arma::uword t = tx_per_link ? l : 0;
            const double d = U(35.0, 300.0), a = U(-pi, pi);
            f.rx(l, 0) = f.tx(t, 0) + d * std::cos(a), f.rx(l, 1) = f.tx(t, 1) + d * std::sin(a);
            f.rx(l, 2) = U(1.5, 22.5);
            f.lsp(l, 0, 0) = U(40e-9, 400e-9);
            f.lsp(l, 0, 1) = std::pow(10.0, 0.1 * U(kdb_lo, kdb_hi));
            f.lsp(l, 0, 2) = 1.0;
            f.lsp(l, 0, 3) = U(5.0, 60.0), f.lsp(l, 0, 4) = U(5.0, 60.0);
            f.lsp(l, 0, 5) = U(1.0, 20.0), f.lsp(l, 0, 6) = U(1.0, 20.0);
            f.lsp(l, 0, 7) = 10.0;
        }
        f.rnd.set_size(n_link, 9 * N);
        for (arma::uword c = 0; c < 9 * N; ++c)
            for (arma::uword l = 0; l < n_link; ++l)
                f.rnd(l, c) = (c < N) ? U(0.02, 0.98) : U(-2.5, 2.5);
        f.rnd_freq.set_size(n_link, N, 1);
        for (arma::uword i = 0; i < f.rnd_freq.n_elem; ++i)
            f.rnd_freq(i) = U(-2.5, 2.5);
        f.eod.set_size(n_link, 1), f.eps_re.set_size(n_link, 1), f.eps_im.set_size(n_link, 1);
        f.toa.set_size(n_link);
        for (arma::uword l = 0; l < n_link; ++l)
            f.eod(l, 0) = U(-3.0, 3.0), f.eps_re(l, 0) = U(3.0, 30.0), f.eps_im(l, 0) = U(0.1, 2.0),
                    f.toa(l) = U(10e-9, 300e-9);
        f.o2i.assign(n_link, false);
        f.ncl.set_size(n_link);
        f.ncl.fill(N);
        return f;
    }

    // Keep the links in 'idx'
    fix_t subset(const fix_t &f, const arma::uvec &idx)
    {
        fix_t s = f;
        s.tx = (f.tx.n_rows == 1) ? f.tx : arma::mat(f.tx.rows(idx));
        s.rx = f.rx.rows(idx), s.rnd = f.rnd.rows(idx), s.eod = f.eod.rows(idx);
        s.eps_re = f.eps_re.rows(idx), s.eps_im = f.eps_im.rows(idx), s.toa = f.toa.elem(idx), s.ncl = f.ncl.elem(idx);
        s.lsp.set_size(idx.n_elem, f.lsp.n_cols, f.lsp.n_slices);
        s.rnd_freq.set_size(idx.n_elem, f.rnd_freq.n_cols, f.rnd_freq.n_slices);
        s.o2i.clear();
        for (arma::uword i = 0; i < idx.n_elem; ++i)
        {
            s.lsp.tube(i, 0, i, f.lsp.n_cols - 1) = f.lsp.tube(idx(i), 0, idx(i), f.lsp.n_cols - 1);
            s.rnd_freq.tube(i, 0, i, f.rnd_freq.n_cols - 1) = f.rnd_freq.tube(idx(i), 0, idx(i), f.rnd_freq.n_cols - 1);
            s.o2i.push_back(f.o2i[idx(i)]);
        }
        return s;
    }

    struct out_t
    {
        arma::cube delay, power, aod, eod, aoa, eoa;
        arma::uvec n_active;
    };

    out_t run(const spmap &sp, const fix_t &f, const std::string &method = "38.901")
    {
        out_t o;
        quadriga_lib::ssf_clusters(sp, f.tx, f.rx, f.lsp, f.rnd, o.delay, o.power, o.aod, o.eod, o.aoa, o.eoa, method,
                                   f.use_rf ? &f.rnd_freq : nullptr, nullptr, f.use_eod ? &f.eod : nullptr,
                                   f.use_toa ? &f.toa : nullptr, f.use_eps ? &f.eps_re : nullptr,
                                   f.use_eps ? &f.eps_im : nullptr, f.use_o2i ? &f.o2i : nullptr,
                                   f.use_ncl ? &f.ncl : nullptr, &o.n_active);
        return o;
    }

    bool equal(const out_t &a, const out_t &b)
    {
        auto eq = [](const arma::cube &x, const arma::cube &y)
        { return arma::size(x) == arma::size(y) && arma::all(arma::vectorise(x) == arma::vectorise(y)); };
        return eq(a.delay, b.delay) && eq(a.power, b.power) && eq(a.aod, b.aod) && eq(a.eod, b.eod) &&
               eq(a.aoa, b.aoa) && eq(a.eoa, b.eoa) && arma::all(a.n_active == b.n_active);
    }

    // LOS direction of link l in degrees: azimuth of departure, zenith of departure (TR clause 7.5 step 1c)
    void los_dir(const fix_t &f, arma::uword l, double &phi_d, double &theta_d, double &d2D, double &d3D)
    {
        const arma::uword t = (f.tx.n_rows == 1) ? 0 : l;
        const double dx = f.rx(l, 0) - f.tx(t, 0), dy = f.rx(l, 1) - f.tx(t, 1), dz = f.rx(l, 2) - f.tx(t, 2);
        d2D = std::sqrt(dx * dx + dy * dy), d3D = std::sqrt(dx * dx + dy * dy + dz * dz);
        phi_d = std::atan2(dy, dx) * 180.0 / pi;
        theta_d = std::acos(dz / d3D) * 180.0 / pi;
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

        double phi_d, theta_d, d2D, d3D;
        los_dir(f, l, phi_d, theta_d, d2D, d3D);
        const double los_ang[4] = {phi_d, phi_d + 180.0, theta_d, 180.0 - theta_d}; // AOD, AOA, ZOD, ZOA

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

    // Compares link l of the output with the oracle; slots of the NLOS clusters start at 'first_slot'
    void check_link(const out_t &o, const ref_t &r, const fix_t &f, arma::uword l, arma::uword first_slot,
                    double *max_err = nullptr)
    {
        const double DS = f.lsp(l, 0, 0);
        double e_d = 0.0, e_p = 0.0, e_a = 0.0;
        for (arma::uword n = 0; n < r.delay.size(); ++n)
        {
            const arma::uword s = first_slot + n;
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

    // Slot 0 and the inactive slots: delay 0, LOS angles of TR clause 7.5 step 1c
    void check_slot0(const out_t &o, const fix_t &f, arma::uword l)
    {
        double phi_d, theta_d, d2D, d3D;
        los_dir(f, l, phi_d, theta_d, d2D, d3D);
        CHECK(o.delay(l, 0, 0) == 0.0);
        CHECK(std::abs(wrap_rad(o.aod(l, 0, 0) - phi_d * pi / 180.0)) <= 1e-12);
        CHECK(std::abs(wrap_rad(o.aoa(l, 0, 0) - (phi_d + 180.0) * pi / 180.0)) <= 1e-12);
        CHECK(std::abs(o.eod(l, 0, 0) - (90.0 - theta_d) * pi / 180.0) <= 1e-12);
        CHECK(std::abs(o.eoa(l, 0, 0) - (theta_d - 90.0) * pi / 180.0) <= 1e-12);
        CHECK(o.aod(l, 0, 0) >= -pi);
        CHECK(o.aod(l, 0, 0) < pi);
        CHECK(o.aoa(l, 0, 0) >= -pi);
        CHECK(o.aoa(l, 0, 0) < pi);
    }

    // Reflection coefficients of TR eqs. (7.6-37a), (7.6-38) to (7.6-40); 'sign' of the imaginary part: -1 = TR sign
    // eps_re - j·eps_im, +1 = conjugate
    void ref_r(double d2D, double h_t, double h_r, double eps_re, double eps_im, std::complex<double> &R_par,
               std::complex<double> &R_perp, double sign = -1.0)
    {
        const double th = (180.0 - std::atan(d2D / (h_r + h_t)) * 180.0 / pi) * pi / 180.0;
        const std::complex<double> eps(eps_re, sign * eps_im);
        const std::complex<double> q = std::sqrt(eps - std::sin(th) * std::sin(th));
        R_par = (eps * std::cos(th) + q) / (eps * std::cos(th) - q);
        R_perp = (std::cos(th) + q) / (std::cos(th) - q);
    }

    // Polarisation-averaged reflection power (abs(R_par)² + abs(R_perp)²)/2
    double ref_r2(double d2D, double h_t, double h_r, double eps_re, double eps_im, double sign = -1.0)
    {
        std::complex<double> R_par, R_perp;
        ref_r(d2D, h_t, h_r, eps_re, eps_im, R_par, R_perp, sign);
        return 0.5 * (std::norm(R_par) + std::norm(R_perp));
    }

    // Row l of a cube as a vector (slots, then slices)
    arma::vec rv(const arma::cube &c, arma::uword l)
    {
        arma::cube t = c.row(l);
        return arma::vectorise(t);
    }

    // Slots a to b of a cube
    arma::vec slots(const arma::cube &c, arma::uword a, arma::uword b)
    {
        arma::cube t = c.cols(a, b);
        return arma::vectorise(t);
    }

    void check_finite(const out_t &o)
    {
        CHECK(o.delay.is_finite());
        CHECK(o.power.is_finite());
        CHECK(o.aod.is_finite());
        CHECK(o.eod.is_finite());
        CHECK(o.aoa.is_finite());
        CHECK(o.eoa.is_finite());
    }
}

TEST_CASE("ssf_clusters - output sizes and slot layout")
{
    for (int gr = 0; gr < 2; ++gr)
        for (const char *name : {"UMa_LOS", "UMa_NLOS"})
        {
            spmap sp = read_v19(name);
            sp["GR_enabled"] = gr;
            const arma::uword N = (std::string(name) == "UMa_LOS") ? UMa_LOS.N : UMa_NLOS.N;
            fix_t f = make_fix(N, 5, 11, gr == 0);
            f.use_eps = gr == 1;
            INFO(name << ", GR " << gr);
            out_t o = run(sp, f);
            const arma::uword n_slot = N + 1 + gr;
            for (const arma::cube *c : {&o.delay, &o.power, &o.aod, &o.eod, &o.aoa, &o.eoa})
            {
                CHECK(c->n_rows == 5);
                CHECK(c->n_cols == n_slot);
                CHECK(c->n_slices == 1);
            }
            CHECK(o.n_active.n_elem == 5);
            for (arma::uword l = 0; l < 5; ++l)
                check_slot0(o, f, l);
            check_finite(o);
            CHECK(arma::all(arma::vectorise(o.eod) >= -pi / 2.0));
            CHECK(arma::all(arma::vectorise(o.eod) <= pi / 2.0));
            CHECK(arma::all(arma::vectorise(o.eoa) >= -pi / 2.0));
            CHECK(arma::all(arma::vectorise(o.eoa) <= pi / 2.0));
            CHECK(arma::all(arma::vectorise(o.aod) >= -pi));
            CHECK(arma::all(arma::vectorise(o.aod) < pi));
            CHECK(arma::all(arma::vectorise(o.aoa) >= -pi));
            CHECK(arma::all(arma::vectorise(o.aoa) < pi));
            CHECK(arma::all(arma::vectorise(o.power) >= 0.0));
            CHECK(arma::all(arma::vectorise(o.delay) >= 0.0));
        }
}

TEST_CASE("ssf_clusters - 38.901 NLOS matches TR 7.5")
{
    const spmap sp = read_v19("UMa_NLOS");
    fix_t f = make_fix(UMa_NLOS.N, 4, 21, true);
    f.use_eod = true;
    out_t o = run(sp, f);
    double err[3] = {0, 0, 0};
    for (arma::uword l = 0; l < 4; ++l)
    {
        ref_t r = ref_tr(f, l, UMa_NLOS, UMa_NLOS.N, false, f.eod(l, 0));
        check_link(o, r, f, l, 1, err);
        CHECK(o.power(l, 0, 0) == 0.0);
        CHECK(o.n_active(l) == UMa_NLOS.N - r.n_removed);
        double s = 0.0;
        for (double p : r.power)
            s += p;
        QD_DIAG("# diag: 38.901 NLOS link %d: removed %d, power sum %.6f\n", (int)l, (int)r.n_removed, s);
    }
    QD_DIAG("# diag: 38.901 NLOS max error: delay %.2e rel, power %.2e rel, angle %.2e rad\n", err[0], err[1], err[2]);

    // The EoD offset moves the NLOS departure elevations only
    f.use_eod = false;
    out_t o0 = run(sp, f);
    CHECK(arma::all(arma::vectorise(o0.aod == o.aod)));
    CHECK(arma::all(arma::vectorise(o0.eoa == o.eoa)));
    CHECK(arma::any(arma::vectorise(o0.eod != o.eod)));
}

TEST_CASE("ssf_clusters - 38.901 LOS matches TR 7.5")
{
    const spmap sp = read_v19("UMa_LOS");
    double err[3] = {0, 0, 0};
    for (double kdb : {3.0, 9.0, 15.0})
    {
        fix_t f = make_fix(UMa_LOS.N, 4, 31, true, kdb, kdb);
        f.use_eod = true; // no effect in LOS
        out_t o = run(sp, f);
        for (arma::uword l = 0; l < 4; ++l)
        {
            INFO("K = " << kdb << " dB");
            ref_t r = ref_tr(f, l, UMa_LOS, UMa_LOS.N, true);
            check_link(o, r, f, l, 1, err);
            const double K = f.lsp(l, 0, 1);
            CHECK(std::abs(o.power(l, 0, 0) - K / (K + 1.0)) <= 1e-15);

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
            double s = 0.0;
            for (arma::uword n = 0; n < UMa_LOS.N; ++n)
                s += o.power(l, 1 + n, 0);
            if (r.n_removed == 0)
                CHECK(std::abs(s - 1.0 / (K + 1.0)) <= 1e-12);
        }
    }
    QD_DIAG("# diag: 38.901 LOS max error: delay %.2e rel, power %.2e rel, angle %.2e rad\n", err[0], err[1], err[2]);
}

TEST_CASE("ssf_clusters - 38.901 limits the K-factor to -5 dB in the C factors")
{
    const spmap sp = read_v19("UMa_LOS");
    fix_t f = make_fix(UMa_LOS.N, 4, 41, true, -8.0, -8.0);
    out_t o = run(sp, f);
    for (arma::uword l = 0; l < 4; ++l)
    {
        // Polynomials at -5 dB, LOS power and the 7.5-8 split at -8 dB
        ref_t r = ref_tr(f, l, UMa_LOS, UMa_LOS.N, true, 0.0, false, -5.0);
        check_link(o, r, f, l, 1);
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
        fix_t f = make_fix(c.N, 3, 51, false, 10.0, 15.0);
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
            check_link(o, r, f, l, 1);
            CHECK(o.power(l, 1 + 2 + l, 0) == 0.0);
            CHECK(o.power(l, 1 + 7 + l, 0) == 0.0);
            CHECK(o.delay(l, 1 + 2 + l, 0) > 0.0); // delay and angles kept (oracle above)
            CHECK(o.n_active(l) == c.N - 2);
            double s = 0.0;
            for (arma::uword n = 0; n < c.N; ++n)
                s += o.power(l, 1 + n, 0);
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
    fix_t f = make_fix(InF_NLOS.N, 5, 61, true);
    f.use_ncl = true;
    const arma::uword ncl[5] = {9, 13, 17, 18, 21};
    for (arma::uword l = 0; l < 5; ++l)
        f.ncl(l) = ncl[l];
    out_t o = run(sp, f);
    for (arma::uword l = 0; l < 5; ++l)
    {
        ref_t r = ref_tr(f, l, InF_NLOS, ncl[l], false);
        check_link(o, r, f, l, 1);
        CHECK(o.n_active(l) == ncl[l] - r.n_removed);
        for (arma::uword s = 1 + ncl[l]; s < 1 + InF_NLOS.N; ++s)
        {
            CHECK(o.delay(l, s, 0) == 0.0);
            CHECK(o.power(l, s, 0) == 0.0);
            CHECK(o.aod(l, s, 0) == o.aod(l, 0, 0));
            CHECK(o.eod(l, s, 0) == o.eod(l, 0, 0));
            CHECK(o.aoa(l, s, 0) == o.aoa(l, 0, 0));
            CHECK(o.eoa(l, s, 0) == o.eoa(l, 0, 0));
        }
    }

    // 30 clusters: C beyond the last table entry stays at the value for 25
    sp["NumClusters"] = 31.0;
    const tr_const c30 = {InF_NLOS.r_tau, InF_NLOS.zeta, 30};
    fix_t g = make_fix(30, 3, 62, true);
    out_t o30 = run(sp, g);
    for (arma::uword l = 0; l < 3; ++l)
        check_link(o30, ref_tr(g, l, c30, 30, false), g, l, 1);
}

TEST_CASE("ssf_clusters - 38.901 O2I links use 90 degrees as mean ZOA")
{
    const spmap sp = read_v19("UMi_NLOS_O2I");
    fix_t f = make_fix(UMi_O2I.N, 4, 71, true);
    f.use_o2i = true;
    f.o2i = {true, false, true, false};
    out_t o = run(sp, f);
    for (arma::uword l = 0; l < 4; ++l)
        check_link(o, ref_tr(f, l, UMi_O2I, UMi_O2I.N, false, 0.0, f.o2i[l]), f, l, 1);

    // Without the flags all links use the LOS zenith
    f.use_o2i = false;
    out_t o0 = run(sp, f);
    CHECK(arma::all(rv(o0.eoa, 1) == rv(o.eoa, 1)));
    CHECK(arma::any(rv(o0.eoa, 0) != rv(o.eoa, 0)));
    CHECK(arma::all(arma::vectorise(o0.eod == o.eod)));
}

TEST_CASE("ssf_clusters - 38.901 folds zenith angles at 0 and 180 degrees")
{
    // Offsets of +-25 standard deviations with ZSD, ZSA of 40 deg drive the zenith angles beyond 180 and below 0
    for (const char *name : {"UMa_NLOS", "UMa_LOS"})
    {
        const bool los = std::string(name) == "UMa_LOS";
        const tr_const c = los ? UMa_LOS : UMa_NLOS;
        const spmap sp = read_v19(name);
        fix_t f = make_fix(c.N, 3, 81, true, 5.0, 10.0);
        for (arma::uword l = 0; l < 3; ++l)
        {
            f.lsp(l, 0, 5) = 40.0, f.lsp(l, 0, 6) = 40.0;
            for (arma::uword n = 0; n < c.N; ++n)
                f.rnd(l, 7 * c.N + n) = (n % 2 == 0) ? 25.0 : -25.0, f.rnd(l, 8 * c.N + n) = (n % 3 == 0) ? -25.0 : 25.0;
        }
        out_t o = run(sp, f);
        int n_beyond = 0;
        for (arma::uword l = 0; l < 3; ++l)
        {
            ref_t r = ref_tr(f, l, c, c.N, los);
            check_link(o, r, f, l, 1);
            for (arma::uword n = 0; n < c.N; ++n)
                n_beyond += (std::abs(f.rnd(l, 7 * c.N + n)) * 40.0 / 7.0 > 90.0) ? 1 : 0;
        }
        CHECK(n_beyond > 0);
        CHECK(arma::all(arma::vectorise(o.eod) >= -pi / 2.0));
        CHECK(arma::all(arma::vectorise(o.eod) <= pi / 2.0));
        CHECK(arma::all(arma::vectorise(o.eoa) >= -pi / 2.0));
        CHECK(arma::all(arma::vectorise(o.eoa) <= pi / 2.0));
    }
}

TEST_CASE("ssf_clusters - absolute TOA adds the offset to the NLOS delays of NLOS scenarios")
{
    // NLOS scenario: TR eq. (7.6-43), every NLOS delay plus the offset; the first cluster is at the offset
    const spmap sp = read_v19("UMa_NLOS");
    fix_t f = make_fix(UMa_NLOS.N, 3, 91, true);
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
    CHECK(arma::all(arma::vectorise(o.power == o0.power)));
    CHECK(arma::all(arma::vectorise(o.aod == o0.aod)));

    // Zero offsets equal the omitted input
    f.toa.zeros();
    CHECK(equal(run(sp, f), o0));

    // LOS scenario: TR eq. (7.6-44), no offset
    const spmap sl = read_v19("UMa_LOS");
    fix_t g = make_fix(UMa_LOS.N, 3, 92, true);
    out_t a = run(sl, g);
    g.use_toa = true;
    CHECK(equal(run(sl, g), a));
}

TEST_CASE("ssf_clusters - ground reflection follows TR 7.6.8")
{
    spmap sp = read_v19("UMa_LOS");
    fix_t f = make_fix(UMa_LOS.N, 4, 101, true, 3.0, 15.0);
    out_t o_off = run(sp, f);
    sp["GR_enabled"] = 1.0;
    f.use_eps = true;
    out_t o = run(sp, f);
    REQUIRE(o.power.n_cols == UMa_LOS.N + 2);
    double err_d = 0.0, err_p = 0.0, err_r = 0.0;
    for (arma::uword l = 0; l < 4; ++l)
    {
        INFO("link " << l);
        double phi_d, theta_d, d2D, d3D;
        los_dir(f, l, phi_d, theta_d, d2D, d3D);
        const double h_t = f.tx(l, 2), h_r = f.rx(l, 2);
        const double d_gr = std::sqrt(d2D * d2D + (h_t + h_r) * (h_t + h_r)); // eq. (7.6-33)
        const double tau = (d_gr - d3D) / c0;
        err_d = std::max(err_d, std::abs(o.delay(l, 1, 0) / tau - 1.0));
        CHECK(std::abs(o.delay(l, 1, 0) - tau) <= 1e-12 * tau);

        // Angles: eqs. (7.6-37a) to (7.6-37d), elevation = 90 deg - zenith
        const double el = (90.0 - (180.0 - std::atan(d2D / (h_r + h_t)) * 180.0 / pi)) * pi / 180.0;
        CHECK(o.aod(l, 1, 0) == o.aod(l, 0, 0));
        CHECK(o.aoa(l, 1, 0) == o.aoa(l, 0, 0));
        CHECK(std::abs(o.eod(l, 1, 0) - el) <= 1e-12);
        CHECK(std::abs(o.eoa(l, 1, 0) - el) <= 1e-12);

        // Power: K/(K+1)·(d3D/d_GR)²·R², R² the same for both signs of the imaginary part
        const double K = f.lsp(l, 0, 1);
        const double R2 = ref_r2(d2D, h_t, h_r, f.eps_re(l, 0), f.eps_im(l, 0));
        const double R2c = ref_r2(d2D, h_t, h_r, f.eps_re(l, 0), f.eps_im(l, 0), 1.0);
        err_r = std::max(err_r, std::abs(R2c / R2 - 1.0));
        CHECK(std::abs(R2c - R2) <= 1e-14 * R2);
        const double p_gr = K / (K + 1.0) * (d3D / d_gr) * (d3D / d_gr) * R2;
        err_p = std::max(err_p, std::abs(o.power(l, 1, 0) / p_gr - 1.0));
        CHECK(std::abs(o.power(l, 1, 0) - p_gr) <= 1e-12 * p_gr);
        CHECK(R2 > 0.0);
        CHECK(R2 < 1.0);

        // Consumer rule: sqrt(slot power)·diag(R_par, -R_perp)/R equals the coefficient of eq. (7.6-32),
        // sqrt(K/(K+1))·(d3D/d_GR)·diag(R_par, -R_perp), for both polarisations
        std::complex<double> R_par, R_perp;
        ref_r(d2D, h_t, h_r, f.eps_re(l, 0), f.eps_im(l, 0), R_par, R_perp);
        const double amp = std::sqrt(o.power(l, 1, 0)) / std::sqrt(R2), amp_tr = std::sqrt(K / (K + 1.0)) * d3D / d_gr;
        CHECK(std::abs(amp * R_par - amp_tr * R_par) <= 1e-12 * std::abs(amp_tr * R_par));
        CHECK(std::abs(amp * R_perp - amp_tr * R_perp) <= 1e-12 * std::abs(amp_tr * R_perp));

        // LOS slot and NLOS slots as without the reflection (shifted by one slot); total power above 1
        double total = 0.0;
        for (arma::uword s = 0; s < UMa_LOS.N + 2; ++s)
            total += o.power(l, s, 0);
        CHECK(total > 1.0);
        CHECK(o.power(l, 0, 0) == o_off.power(l, 0, 0));
    }
    CHECK(arma::all(slots(o.delay, 2, UMa_LOS.N + 1) == slots(o_off.delay, 1, UMa_LOS.N)));
    CHECK(arma::all(slots(o.power, 2, UMa_LOS.N + 1) == slots(o_off.power, 1, UMa_LOS.N)));
    CHECK(arma::all(slots(o.aod, 2, UMa_LOS.N + 1) == slots(o_off.aod, 1, UMa_LOS.N)));
    CHECK(arma::all(slots(o.eod, 2, UMa_LOS.N + 1) == slots(o_off.eod, 1, UMa_LOS.N)));
    CHECK(arma::all(slots(o.aoa, 2, UMa_LOS.N + 1) == slots(o_off.aoa, 1, UMa_LOS.N)));
    CHECK(arma::all(slots(o.eoa, 2, UMa_LOS.N + 1) == slots(o_off.eoa, 1, UMa_LOS.N)));
    CHECK(arma::all(o.n_active == o_off.n_active));
    QD_DIAG("# diag: GR max error: delay %.2e rel, power %.2e rel, R2 sign %.2e rel\n", err_d, err_p, err_r);

    // NLOS scenario: zero reflection power, geometry unchanged
    spmap sn = read_v19("UMa_NLOS");
    sn["GR_enabled"] = 1.0;
    fix_t g = make_fix(UMa_NLOS.N, 4, 102, true);
    g.use_eps = true;
    out_t on = run(sn, g);
    for (arma::uword l = 0; l < 4; ++l)
    {
        CHECK(on.power(l, 0, 0) == 0.0);
        CHECK(on.power(l, 1, 0) == 0.0);
        CHECK(on.delay(l, 1, 0) > 0.0);
    }

    // Both ends on the ground (grazing): eps = 1 gives R = 0, eps = 15 - 1j gives R_par = R_perp = -1
    fix_t h = make_fix(UMa_LOS.N, 2, 103, true, 9.0, 9.0);
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

TEST_CASE("ssf_clusters - NLOS scenarios give zero LOS power whatever the K-factor in lsp")
{
    const spmap sp = read_v19("UMi_NLOS");
    fix_t f = make_fix(UMi_NLOS.N, 4, 111, true);
    f.lsp.slice(1).fill(1e-10);
    out_t a = run(sp, f);
    f.lsp.slice(1).fill(5.0);
    out_t b = run(sp, f);
    CHECK(equal(a, b));
    CHECK(arma::all(arma::vectorise(a.power.col(0)) == 0.0));
}

TEST_CASE("ssf_clusters - 38.901 is swap-reciprocal for mirrored inputs")
{
    for (const char *name : {"UMa_LOS", "UMa_NLOS"})
    {
        const arma::uword N = (std::string(name) == "UMa_LOS") ? UMa_LOS.N : UMa_NLOS.N;
        const spmap sp = read_v19(name);
        fix_t f = make_fix(N, 6, 121, true);
        fix_t s = f;
        s.tx = f.rx, s.rx = f.tx;
        s.lsp.slice(3) = f.lsp.slice(4), s.lsp.slice(4) = f.lsp.slice(3);
        s.lsp.slice(5) = f.lsp.slice(6), s.lsp.slice(6) = f.lsp.slice(5);
        const arma::uword mb[9] = {0, 2, 1, 4, 3, 6, 5, 8, 7};
        for (arma::uword b = 0; b < 9; ++b)
            s.rnd.cols(b * N, b * N + N - 1) = f.rnd.cols(mb[b] * N, mb[b] * N + N - 1);
        out_t a = run(sp, f), b = run(sp, s);
        INFO(name);
        double e = 0.0;
        for (arma::uword i = 0; i < a.delay.n_elem; ++i)
        {
            const arma::uword l = i % 6;
            CHECK(std::abs(a.delay(i) - b.delay(i)) <= 1e-12 * f.lsp(l, 0, 0));
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
        QD_DIAG("# diag: swap %s, max angle difference %.2e rad\n", name, e);
    }
}

TEST_CASE("ssf_clusters - a link does not depend on the other links")
{
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
    {
        const arma::uword N = (std::string(name) == "UMi_LOS") ? UMi_LOS.N : UMi_NLOS.N;
        spmap sp = read_v19(name);
        sp["GR_enabled"] = 1.0;
        fix_t f = make_fix(N, 50, 131, true);
        f.use_eod = f.use_toa = f.use_eps = f.use_ncl = true;
        for (arma::uword l = 0; l < 50; ++l)
            f.ncl(l) = 6 + l % (N - 5);
        out_t all = run(sp, f);
        for (arma::uword l : {0, 17, 49})
        {
            out_t one = run(sp, subset(f, arma::uvec{l}));
            CHECK(arma::all(rv(one.delay, 0) == rv(all.delay, l)));
            CHECK(arma::all(rv(one.power, 0) == rv(all.power, l)));
            CHECK(arma::all(rv(one.aod, 0) == rv(all.aod, l)));
            CHECK(arma::all(rv(one.eod, 0) == rv(all.eod, l)));
            CHECK(arma::all(rv(one.aoa, 0) == rv(all.aoa, l)));
            CHECK(arma::all(rv(one.eoa, 0) == rv(all.eoa, l)));
            CHECK(one.n_active(0) == all.n_active(l));
        }
    }
}

TEST_CASE("ssf_clusters - 38.901 spread statistics over a drop")
{
    // UMa NLOS at 6 GHz, 5 seeds x 400 UTs at d2D 35 to 300 m; all decorrelation distances 0 (fields at 0.1 m), so
    // the links are independent. Window: medians of the comparison study +-0.03 (spread of the median over
    // repetitions of 2000 links about 0.005, i.e. more than 5 standard errors)
    spmap sp = read_v19("UMa_NLOS");
    for (const char *k : {"SC_lambda", "DS_lambda", "KF_lambda", "SF_lambda", "AS_D_lambda", "AS_A_lambda",
                          "ES_D_lambda", "ES_A_lambda", "XPR_lambda", "absTOA_lambda"})
        sp[k] = 0.0;
    const arma::uword N = UMa_NLOS.N, n_ut = 400;
    const arma::vec fc = {6e9};
    std::vector<double> ratio[5];
    arma::mat tx = {{0.0, 0.0, 25.0}};
    for (arma::sword seed = 1; seed <= 5; ++seed)
    {
        std::mt19937 g((unsigned)seed);
        arma::mat rx(n_ut, 3);
        for (arma::uword i = 0; i < n_ut; ++i)
        {
            const double d = 35.0 + 265.0 * ((double)g() + 0.5) / 4294967296.0;
            const double a = 2.0 * pi * ((double)g() + 0.5) / 4294967296.0;
            rx(i, 0) = d * std::cos(a), rx(i, 1) = d * std::sin(a), rx(i, 2) = 1.5;
        }
        arma::cube ph_l, ph_c, ph_f, lsp, rnd_freq;
        arma::mat rnd_l, rnd, rf, eod;
        arma::vec toa;
        quadriga_lib::lsf_init_phases(sp, ph_l, seed);
        quadriga_lib::lsf_draw(sp, ph_l, tx, rx, rnd_l);
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd_l, lsp, nullptr, &toa, nullptr, nullptr, &eod);
        quadriga_lib::ssf_init_phases(sp, ph_c, seed, -1);
        quadriga_lib::ssf_init_phases(sp, ph_f, seed, 0);
        quadriga_lib::ssf_draw(sp, ph_c, tx, rx, rnd, -1);
        quadriga_lib::ssf_draw(sp, ph_f, tx, rx, rf, 0);
        rnd_freq.set_size(n_ut, N, 1);
        rnd_freq.slice(0) = rf;
        arma::cube delay, power, aod, eod_o, aoa, eoa;
        quadriga_lib::ssf_clusters(sp, tx, rx, lsp, rnd, delay, power, aod, eod_o, aoa, eoa, "38.901", &rnd_freq,
                                   nullptr, &eod, &toa);
        std::vector<arma::vec> d(n_ut), p(n_ut), az_d(n_ut), el_d(n_ut), az_a(n_ut), el_a(n_ut);
        for (arma::uword i = 0; i < n_ut; ++i)
        {
            d[i] = rv(delay, i), p[i] = rv(power, i), az_d[i] = rv(aod, i), el_d[i] = rv(eod_o, i);
            az_a[i] = rv(aoa, i), el_a[i] = rv(eoa, i);
        }
        arma::vec ds = quadriga_lib::calc_delay_spread(d, p), asd, esd, asa, esa;
        quadriga_lib::calc_angular_spreads_sphere(az_d, el_d, p, &asd, &esd);
        quadriga_lib::calc_angular_spreads_sphere(az_a, el_a, p, &asa, &esa);
        const double r2d = 180.0 / pi;
        for (arma::uword i = 0; i < n_ut; ++i)
        {
            ratio[0].push_back(ds(i) / lsp(i, 0, 0));
            ratio[1].push_back(asd(i) * r2d / lsp(i, 0, 3)), ratio[2].push_back(asa(i) * r2d / lsp(i, 0, 4));
            ratio[3].push_back(esd(i) * r2d / lsp(i, 0, 5)), ratio[4].push_back(esa(i) * r2d / lsp(i, 0, 6));
        }
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
    for (const char *name : {"UMa_LOS", "UMa_NLOS"})
        for (int gr = 0; gr < 2; ++gr)
        {
            INFO(name << ", GR " << gr);
            spmap sp = read_v19(name);
            sp["NumClusters"] = 1.0, sp["NumClusters_min"] = 0.0, sp["GR_enabled"] = gr;
            fix_t f = make_fix(0, 4, 141, true, 3.0, 12.0);
            f.use_rf = false;
            f.use_eps = gr == 1;
            f.use_ncl = true;
            f.ncl.zeros();
            out_t o = run(sp, f);
            REQUIRE(o.power.n_cols == (arma::uword)(1 + gr));
            CHECK(o.power.n_slices == 1);
            for (arma::uword l = 0; l < 4; ++l)
            {
                check_slot0(o, f, l);
                CHECK(o.power(l, 0, 0) == 1.0);
                CHECK(o.n_active(l) == 0);
                if (gr == 1)
                {
                    double phi_d, theta_d, d2D, d3D;
                    los_dir(f, l, phi_d, theta_d, d2D, d3D);
                    const double h_t = f.tx(l, 2), h_r = f.rx(l, 2);
                    const double d_gr = std::sqrt(d2D * d2D + (h_t + h_r) * (h_t + h_r));
                    const double p = (d3D / d_gr) * (d3D / d_gr) * ref_r2(d2D, h_t, h_r, f.eps_re(l, 0), f.eps_im(l, 0));
                    CHECK(std::abs(o.power(l, 1, 0) - p) <= 1e-12 * p);
                    CHECK(std::abs(o.delay(l, 1, 0) - (d_gr - d3D) / c0) <= 1e-12 * (d_gr - d3D) / c0);
                }
            }
            // A 0x0 matrix is accepted as the empty set
            f.rnd.reset();
            CHECK(equal(run(sp, f), o));
        }
}

TEST_CASE("ssf_clusters - extreme finite random values give finite powers")
{
    for (const char *name : {"UMa_LOS", "UMa_NLOS"})
    {
        const bool los = std::string(name) == "UMa_LOS";
        const arma::uword N = los ? UMa_LOS.N : UMa_NLOS.N;
        INFO(name);
        spmap sp = read_v19(name);
        sp["GR_enabled"] = 1.0;
        fix_t f = make_fix(N, 3, 151, true, 0.0, 15.0);
        f.use_eps = true;

        // z = -1100 on one cluster: 10^(330) overflows in the linear form of eq. (7.5-5)
        for (arma::uword l = 0; l < 3; ++l)
            f.rnd_freq(l, 3 + l, 0) = -1100.0;
        out_t o = run(sp, f);
        check_finite(o);
        for (arma::uword l = 0; l < 3; ++l)
        {
            const double K = los ? f.lsp(l, 0, 1) : 0.0;
            CHECK(std::abs(o.power(l, 2 + 3 + l, 0) - 1.0 / (K + 1.0)) <= 1e-15);
            CHECK(arma::all(arma::vectorise(o.power.row(l)) >= 0.0));
        }

        // Normal values at +-40, delay values 0 and 1
        for (arma::uword c = 0; c < 9 * N; ++c)
            for (arma::uword l = 0; l < 3; ++l)
                f.rnd(l, c) = (c < N) ? (double)((c + l) % 2) : ((c + l) % 2 ? 40.0 : -40.0);
        for (arma::uword i = 0; i < f.rnd_freq.n_elem; ++i)
            f.rnd_freq(i) = (i % 3 == 0) ? 40.0 : -40.0;
        o = run(sp, f);
        check_finite(o);
    }
}

TEST_CASE("ssf_clusters - empty input and pre-sized outputs")
{
    spmap sp = read_v19("UMa_LOS");
    sp["GR_enabled"] = 1.0;
    const arma::uword N = UMa_LOS.N;

    // No link
    fix_t e = make_fix(N, 0, 161, false);
    e.use_eps = e.use_eod = e.use_toa = e.use_o2i = e.use_ncl = true;
    out_t o = run(sp, e);
    for (const arma::cube *c : {&o.delay, &o.power, &o.aod, &o.eod, &o.aoa, &o.eoa})
    {
        CHECK(c->n_rows == 0);
        CHECK(c->n_cols == N + 2);
        CHECK(c->n_slices == 1);
    }
    CHECK(o.n_active.n_elem == 0);

    // Pre-sized outputs with NaN are written completely; wrong sizes are resized
    fix_t f = make_fix(N, 4, 162, true);
    f.use_eps = f.use_ncl = true;
    f.ncl = {N, 11, 12, 11};
    out_t ref = run(sp, f);
    out_t p;
    for (arma::cube *c : {&p.delay, &p.power, &p.aod, &p.eod, &p.aoa, &p.eoa})
        c->set_size(4, N + 2, 1), c->fill(NAN);
    p.n_active.set_size(4), p.n_active.fill(99);
    quadriga_lib::ssf_clusters(sp, f.tx, f.rx, f.lsp, f.rnd, p.delay, p.power, p.aod, p.eod, p.aoa, p.eoa, "38.901",
                               &f.rnd_freq, nullptr, nullptr, nullptr, &f.eps_re, &f.eps_im, nullptr, &f.ncl, &p.n_active);
    CHECK(equal(p, ref));
    out_t w;
    for (arma::cube *c : {&w.delay, &w.power, &w.aod, &w.eod, &w.aoa, &w.eoa})
        c->set_size(2, 3, 4), c->fill(NAN);
    w.n_active.set_size(7);
    quadriga_lib::ssf_clusters(sp, f.tx, f.rx, f.lsp, f.rnd, w.delay, w.power, w.aod, w.eod, w.aoa, w.eoa, "38.901",
                               &f.rnd_freq, nullptr, nullptr, nullptr, &f.eps_re, &f.eps_im, nullptr, &f.ncl, &w.n_active);
    CHECK(equal(w, ref));
}

TEST_CASE("ssf_clusters - input validation")
{
    const spmap sp = read_v19("UMa_LOS");
    const arma::uword N = UMa_LOS.N;
    fix_t f = make_fix(N, 3, 171, true);
    arma::cube d, p, a1, e1, a2, e2;
    auto call = [&](const spmap &s, const fix_t &x, const std::string &method = "38.901",
                    const arma::cube *anchor = nullptr)
    {
        quadriga_lib::ssf_clusters(s, x.tx, x.rx, x.lsp, x.rnd, d, p, a1, e1, a2, e2, method,
                                   x.use_rf ? &x.rnd_freq : nullptr, anchor, x.use_eod ? &x.eod : nullptr,
                                   x.use_toa ? &x.toa : nullptr, x.use_eps ? &x.eps_re : nullptr,
                                   x.use_eps ? &x.eps_im : nullptr, x.use_o2i ? &x.o2i : nullptr,
                                   x.use_ncl ? &x.ncl : nullptr);
    };
    auto inv = [](const char *msg) { return Message(msg); };
    fix_t g = f; // all optional inputs given
    g.use_eod = g.use_toa = g.use_o2i = g.use_ncl = true;
    spmap sg = sp;
    sg["GR_enabled"] = 1.0;
    g.use_eps = true;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(call(sp, f));
        CHECK_NOTHROW(call(sg, g));
    }
    SECTION("method")
    {
        CHECK_THROWS_MATCHES(call(sp, f, "38.901 "), std::invalid_argument,
                             inv("Input 'method' must be '38.901', '38.901_MF' or 'quadriga'."));
        CHECK_THROWS_MATCHES(call(sp, f, "QuaDRiGa"), std::invalid_argument,
                             inv("Input 'method' must be '38.901', '38.901_MF' or 'quadriga'."));
    }
    SECTION("positions")
    {
        fix_t x = f;
        x.rx = arma::mat(3, 2, arma::fill::ones);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'rx_pos' must have 3 columns."));
        x = f;
        x.tx = arma::mat(2, 3, arma::fill::ones);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument,
                             inv("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'."));
        x = f;
        x.rx(1, 1) = NAN;
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'rx_pos' must be finite."));
    }
    SECTION("scenpar")
    {
        spmap s = sp;
        s["NumClusters"] = 0.0;
        CHECK_THROWS_MATCHES(call(s, f), std::invalid_argument,
                             inv("Value of 'NumClusters' must be an integer from 1 to 1000."));
        s = sp;
        s["r_DS"] = -1.0;
        CHECK_THROWS_MATCHES(call(s, f), std::invalid_argument, MessageMatches(ContainsSubstring("'r_DS'")));
        s = sp;
        s["NumClusters_min"] = 20.0;
        CHECK_THROWS_MATCHES(call(s, f), std::invalid_argument, MessageMatches(ContainsSubstring("NumClusters_min")));
    }
    SECTION("lsp")
    {
        fix_t x = f;
        x.lsp = arma::cube(3, 1, 7, arma::fill::ones);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'lsp' must have size [n_link, n_freq, 8]."));
        x.lsp = arma::cube(2, 1, 8, arma::fill::ones);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'lsp' must have size [n_link, n_freq, 8]."));
        x.lsp = arma::cube(3, 0, 8);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'lsp' must have size [n_link, n_freq, 8]."));
        x = f;
        x.lsp(1, 0, 7) = INFINITY;
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'lsp' must be finite."));
        for (arma::uword s : {0, 3, 4, 5, 6})
        {
            x = f;
            x.lsp(2, 0, s) = 0.0;
            CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument,
                                 inv("Delay and angular spreads in 'lsp' must be > 0."));
        }
        x = f;
        x.lsp(0, 0, 1) = -1e-3;
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("K-factor in 'lsp' must be >= 0."));
        x = f;
        x.lsp(0, 0, 2) = -1.0, x.lsp(1, 0, 7) = 0.0; // slices 2 and 7 are not checked beyond finiteness
        CHECK_NOTHROW(call(sp, x));
    }
    SECTION("38.901 with two frequencies")
    {
        fix_t x = f;
        x.lsp.set_size(3, 2, 8);
        for (arma::uword s = 0; s < 8; ++s)
            x.lsp.slice(s) = arma::join_rows(f.lsp.slice(s), f.lsp.slice(s));
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument,
                             inv("Method '38.901' supports one frequency only; use '38.901_MF' or 'quadriga' for "
                                 "several frequencies."));
    }
    SECTION("rnd")
    {
        fix_t x = f;
        x.rnd = f.rnd.cols(0, 9 * N - 2);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'rnd' must have size [n_link, 108]."));
        x.rnd = f.rnd.rows(0, 1);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'rnd' must have size [n_link, 108]."));
        x = f;
        x.rnd(1, 5 * N) = NAN;
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'rnd' must be finite."));
        x = f;
        x.rnd(2, N - 1) = 1.0 + 1e-12;
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument,
                             inv("Columns 0 to N-1 of 'rnd' (cluster delays) must be in [0, 1]."));
        x.rnd(2, N - 1) = -1e-300;
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument,
                             inv("Columns 0 to N-1 of 'rnd' (cluster delays) must be in [0, 1]."));
        x.rnd(2, N - 1) = 0.0, x.rnd(1, 0) = 1.0, x.rnd(0, N) = 5.0; // bounds included; block 1 is Normal
        CHECK_NOTHROW(call(sp, x));
    }
    SECTION("rnd_freq")
    {
        fix_t x = f;
        x.use_rf = false;
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'rnd_freq' is required for method '38.901'."));
        CHECK_THROWS_MATCHES(call(sp, x, "38.901_MF", &f.lsp), std::invalid_argument,
                             inv("Input 'rnd_freq' is required for method '38.901_MF'."));
        x = f;
        x.rnd_freq = arma::cube(3, N, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'rnd_freq' must have size [n_link, 12, n_freq]."));
        x.rnd_freq = arma::cube(3, N + 1, 1, arma::fill::zeros);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'rnd_freq' must have size [n_link, 12, n_freq]."));
        x = f;
        x.rnd_freq(2, 3, 0) = NAN;
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'rnd_freq' must be finite."));
    }
    SECTION("lsp_anchor (38.901_MF)")
    {
        CHECK_THROWS_MATCHES(call(sp, f, "38.901_MF"), std::invalid_argument,
                             inv("Input 'lsp_anchor' is required for method '38.901_MF'."));
        arma::cube an = arma::cube(3, 2, 8, arma::fill::ones);
        CHECK_THROWS_MATCHES(call(sp, f, "38.901_MF", &an), std::invalid_argument,
                             inv("Input 'lsp_anchor' must have size [n_link, 1, 8]."));
        an = f.lsp;
        an(1, 0, 2) = NAN;
        CHECK_THROWS_MATCHES(call(sp, f, "38.901_MF", &an), std::invalid_argument, inv("Input 'lsp_anchor' must be finite."));
        an = f.lsp;
        an(1, 0, 4) = 0.0;
        CHECK_THROWS_MATCHES(call(sp, f, "38.901_MF", &an), std::invalid_argument,
                             inv("Delay and angular spreads in 'lsp_anchor' must be > 0."));
    }
    SECTION("eod_offset_deg")
    {
        fix_t x = f;
        x.use_eod = true;
        x.eod = arma::mat(3, 2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument,
                             inv("Input 'eod_offset_deg' must have size [n_link, n_freq]."));
        x.eod = arma::mat(3, 1, arma::fill::zeros);
        x.eod(1, 0) = NAN;
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'eod_offset_deg' must be finite."));
    }
    SECTION("abs_toa")
    {
        fix_t x = f;
        x.use_toa = true;
        x.toa = arma::vec(2, arma::fill::zeros);
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument,
                             inv("Number of elements in 'abs_toa' does not match number of rows in 'rx_pos'."));
        x.toa = {1e-8, -1e-12, 0.0};
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'abs_toa' must be finite and >= 0."));
        x.toa = {1e-8, NAN, 0.0};
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Input 'abs_toa' must be finite and >= 0."));
    }
    SECTION("ground reflection")
    {
        fix_t x = g;
        x.use_eps = false;
        CHECK_THROWS_MATCHES(call(sg, x), std::invalid_argument,
                             inv("Inputs 'gr_eps_re' and 'gr_eps_im' are required when 'GR_enabled' is set."));
        x = g;
        x.eps_im = arma::mat(3, 2, arma::fill::ones);
        CHECK_THROWS_MATCHES(call(sg, x), std::invalid_argument,
                             inv("Inputs 'gr_eps_re' and 'gr_eps_im' must have size [n_link, n_freq]."));
        x = g;
        x.eps_re(0, 0) = INFINITY;
        CHECK_THROWS_MATCHES(call(sg, x), std::invalid_argument, inv("Inputs 'gr_eps_re' and 'gr_eps_im' must be finite."));
        x = g;
        x.eps_re(1, 0) = 0.999;
        CHECK_THROWS_MATCHES(call(sg, x), std::invalid_argument,
                             inv("Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0."));
        x = g;
        x.eps_im(2, 0) = -1e-9;
        CHECK_THROWS_MATCHES(call(sg, x), std::invalid_argument,
                             inv("Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0."));
        x = g;
        x.eps_re(1, 0) = 1.0, x.eps_im(2, 0) = 0.0; // the vacuum value is accepted
        CHECK_NOTHROW(call(sg, x));
        x = g;
        x.rx(2, 2) = -0.1;
        CHECK_THROWS_MATCHES(call(sg, x), std::invalid_argument,
                             inv("Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'."));
        x = g;
        x.tx(0, 2) = -3.0;
        CHECK_THROWS_MATCHES(call(sg, x), std::invalid_argument,
                             inv("Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'."));
        x.use_eps = false;
        CHECK_NOTHROW(call(sp, x)); // without the reflection neither is checked
    }
    SECTION("o2i")
    {
        fix_t x = f;
        x.use_o2i = true;
        x.o2i = {true, false};
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument,
                             inv("Number of elements in 'o2i' does not match number of rows in 'rx_pos'."));
    }
    SECTION("n_cluster")
    {
        // UMa LOS: NumClusters_min = 11, so N_l in [10, 12]
        fix_t x = f;
        x.use_ncl = true;
        x.ncl = {12, 12};
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument,
                             inv("Number of elements in 'n_cluster' does not match number of rows in 'rx_pos'."));
        x.ncl = {12, 9, 10};
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Entries of 'n_cluster' must be in [10, 12]."));
        x.ncl = {13, 10, 10};
        CHECK_THROWS_MATCHES(call(sp, x), std::invalid_argument, inv("Entries of 'n_cluster' must be in [10, 12]."));
        x.ncl = {12, 10, 11};
        CHECK_NOTHROW(call(sp, x));
        spmap s = sp;
        s["NumClusters_min"] = 0.0;
        x.ncl = {1, 1, 12};
        CHECK_NOTHROW(call(s, x));
        x.ncl = {0, 1, 12};
        CHECK_THROWS_MATCHES(call(s, x), std::invalid_argument, inv("Entries of 'n_cluster' must be in [1, 12]."));
    }
    SECTION("inputs not used by the method are not checked")
    {
        arma::cube an(1, 1, 1, arma::fill::zeros);
        CHECK_NOTHROW(call(sp, f, "38.901", &an));
        spmap s = sp;
        s["NumClusters"] = 1.0, s["NumClusters_min"] = 0.0;
        fix_t x = make_fix(0, 3, 172, true);
        x.use_rf = false;
        CHECK_NOTHROW(call(s, x));
        x.use_rf = true;
        x.rnd_freq = arma::cube(5, 5, 5, arma::fill::value(NAN));
        CHECK_NOTHROW(call(s, x)); // N = 0: rnd_freq ignored
        x.use_eps = true;
        x.eps_re.fill(0.0); // GR off: permittivity ignored
        CHECK_NOTHROW(call(s, x));
    }
    SECTION("the outputs are untouched on error")
    {
        d = arma::cube(2, 2, 2, arma::fill::value(7.0));
        p = d;
        fix_t x = f;
        x.lsp(0, 0, 0) = -1.0;
        CHECK_THROWS_AS(call(sp, x), std::invalid_argument);
        CHECK(arma::size(d) == arma::size(2, 2, 2));
        CHECK(arma::all(arma::vectorise(d) == 7.0));
        CHECK(arma::all(arma::vectorise(p) == 7.0));
    }
}

TEST_CASE("Diagnostic - ssf_clusters 38.901 jumps along a track")
{
    // UMi LOS at 6 GHz, 20 m track at 1 cm steps; signs and offsets frozen at the start of the track (the caller's
    // rule for spatial consistency), delays and shadowing along the track
    const spmap sp = read_v19("UMi_LOS");
    const arma::uword N = UMi_LOS.N, n_pt = 2001;
    arma::mat tx = {{0.0, 0.0, 10.0}}, rx(n_pt, 3);
    for (arma::uword i = 0; i < n_pt; ++i)
        rx(i, 0) = 40.0 + 0.01 * (double)i, rx(i, 1) = 23.0, rx(i, 2) = 1.5;
    arma::cube ph_l, ph_c, ph_f, lsp, rnd_freq(n_pt, N, 1);
    arma::mat rnd_l, rnd, rf;
    quadriga_lib::lsf_init_phases(sp, ph_l, 7);
    quadriga_lib::lsf_draw(sp, ph_l, tx, rx, rnd_l);
    quadriga_lib::lsp_generate(sp, arma::vec{6e9}, tx, rx, rnd_l, lsp);
    quadriga_lib::ssf_init_phases(sp, ph_c, 7, -1);
    quadriga_lib::ssf_init_phases(sp, ph_f, 7, 0);
    quadriga_lib::ssf_draw(sp, ph_c, tx, rx, rnd, -1);
    quadriga_lib::ssf_draw(sp, ph_f, tx, rx, rf, 0);
    rnd_freq.slice(0) = rf;
    for (arma::uword c = N; c < 9 * N; ++c)
        rnd.col(c).fill(rnd(0, c));
    arma::cube delay, power, aod, eod, aoa, eoa;
    quadriga_lib::ssf_clusters(sp, tx, rx, lsp, rnd, delay, power, aod, eod, aoa, eoa, "38.901", &rnd_freq);
    CHECK(delay.is_finite());
    CHECK(power.is_finite());
    CHECK(aod.is_finite());
    CHECK(eoa.is_finite());

    int ev[4] = {0, 0, 0, 0}; // angle > 10 deg, delay > DS/4, power > 3 dB, clusters within 20 dB
    for (arma::uword i = 1; i < n_pt; ++i)
    {
        double pmax = 0.0;
        for (arma::uword s = 1; s <= N; ++s)
            pmax = std::max(pmax, std::max(power(i, s, 0), power(i - 1, s, 0)));
        for (arma::uword s = 1; s <= N; ++s)
        {
            if (std::max(power(i, s, 0), power(i - 1, s, 0)) < 0.01 * pmax)
                continue;
            const double da = 180.0 / pi * std::max({std::abs(wrap_rad(aod(i, s, 0) - aod(i - 1, s, 0))),
                                                     std::abs(wrap_rad(aoa(i, s, 0) - aoa(i - 1, s, 0))),
                                                     std::abs(eod(i, s, 0) - eod(i - 1, s, 0)),
                                                     std::abs(eoa(i, s, 0) - eoa(i - 1, s, 0))});
            ev[0] += da > 10.0;
            ev[1] += std::abs(delay(i, s, 0) - delay(i - 1, s, 0)) > 0.25 * lsp(i, 0, 0);
            const double r = power(i, s, 0) / power(i - 1, s, 0);
            ev[2] += !(r < 2.0 && r > 0.5); // removal toggles count here (ratio 0 or infinite)
            ++ev[3];
        }
    }
    QD_DIAG("\n# diag: 38.901 UMi LOS, 20 m track at 1 cm: events per m: angle > 10 deg %.2f, delay > DS/4 %.2f, "
            "power > 3 dB %.2f (cluster samples within 20 dB: %d)\n",
            ev[0] / 20.0, ev[1] / 20.0, ev[2] / 20.0, ev[3]);
}
