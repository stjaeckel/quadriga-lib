// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "*38.901_MF*"

// Verification suite for the method "38.901_MF" of quadriga_lib::ssf_clusters, the multi-frequency method of
// TR 38.901 clause 7.6.5.1 (steps 5' to 7'), test cases "ssf_clusters - 38.901_MF ...".
//
// Pins down: nominal delays and angles from the anchor spreads (r = 1.5, wrapped Gaussian azimuths, wrapped
// Laplacian zeniths, no minimum subtraction, no sort); the powers per frequency with the g factors of eqs. (7.6.30b)
// to (7.6.30f) and the per-cluster shadowing per frequency; the LOS split of eq. (7.6.30h) into the LOS slot and the
// first cluster; the LOS scaling of eqs. (7.6.30i) to (7.6.30m) by the K-factor of each frequency, applied after the
// wrap of step 5'; the zenith fold of the final zenith angles; frequency-independent delays and angles in NLOS;
// vanishing g factors; the absolute time of arrival and the
// per-link cluster number. Sizes, ground reflection, swap reciprocity, the LOS-only case, extreme random values and the
// ignored inputs: test_ssf_clusters.cpp.
// Oracles, independent of the function and of the conf files:
//  - ref_mf: TR 38.901 clause 7.6.5.1 typed here in scalar double, in degrees, with the TR's own forms: the principal
//    value of arg(exp(-j·x)) taken from std::arg, the Laplacian term sgn(V - 0.5)·ln(1 - 2·abs(V - 0.5)) with
//    V = Phi(y) (all fixture values have abs(y) < 2.5), P'_n = exp(...)·10^(-Q_n/10) with Q_n = zeta·z_n and the
//    linear normalisation of eqs. (7.6.30g), (7.6.30h); r = 1.5 and zeta of TR Table 7.5-6 are typed here
//  - the LOS direction of qd_test_ssf.hpp; zenith angles are folded by the rule of TR clause 7.5 step 7 (wrap to
//    [0, 360], [180, 360] -> 360 - theta)
// Tolerances: delays 1e-12 relative to max(delay, DS0); powers 1e-12 relative; angles 1e-11 rad; values that the
// function forms by the same operations are compared bitwise.
// Fixture: anchor ASD, ASA 20 to 60 deg, ZSD, ZSA 3 to 20 deg; the spreads at frequency k > 0 a factor 0.5 to 1.2
// of the anchor (g factors > 0).

#include <catch2/catch_test_macros.hpp>

#include "quadriga_gbsm.hpp"

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
    const double r_mf = 1.5; // proportionality factor r of TR clause 7.6.5.1

    // TR 38.901 Table 7.5-6: per-cluster shadowing std zeta [dB] and number of clusters N
    const double zeta_UMi = 3.0;
    const arma::uword N_UMi_LOS = 12, N_UMi_NLOS = 19;

    // Principal value of arg(exp(-j·x)) in degrees, x in degrees (TR clause 7.6.5.1 step 5')
    double arg_deg(double x) { return std::arg(std::polar(1.0, -x * pi / 180.0)) * 180.0 / pi; }

    // Fixture of this file: anchor spreads 20 to 60 deg (azimuth) and 3 to 20 deg (zenith)
    fix_t make_fix_mf(arma::uword N, arma::uword n_link, arma::uword F, unsigned seed)
    {
        return make_fix(N, n_link, F, seed, true, 0.0, 15.0, 20.0, 60.0, 3.0, 20.0);
    }

    // Oracle: TR 38.901 clause 7.6.5.1 steps 5' to 7' for link l at frequency k; outputs per cluster n < N_l
    // (cluster 0 is the TR's cluster 1); power: coefficient power, P_n/(1+K) of eq. (7.6.30h) without the LOS term
    struct ref_t
    {
        std::vector<double> delay, power, ang[4]; // ang: AOD, AOA as azimuth [rad]; EOD, EOA as elevation [rad]
        double p_los = 0.0, p_first_tr = 0.0;     // K/(K+1); P_1 of eq. (7.6.30h)
    };
    ref_t ref_mf(const fix_t &f, arma::uword l, arma::uword k, double zeta, arma::uword N_l, bool los)
    {
        const arma::uword N = f.rnd.n_cols / 9;
        const double DS0 = f.anchor(l, 0, 0), DS = f.lsp(l, k, 0), K = los ? f.lsp(l, k, 1) : 0.0;
        const double AS0[4] = {f.anchor(l, 0, 3), f.anchor(l, 0, 4), f.anchor(l, 0, 5), f.anchor(l, 0, 6)};
        const double AS[4] = {f.lsp(l, k, 3), f.lsp(l, k, 4), f.lsp(l, k, 5), f.lsp(l, k, 6)};

        // Step 5': nominal delays and angles; Y, Z, V, W from the offset blocks 5 to 8
        std::vector<double> tp(N_l), ap[4];
        for (arma::uword n = 0; n < N_l; ++n)
        {
            tp[n] = -r_mf * DS0 * std::log(f.rnd(l, n));
            for (int x = 0; x < 4; ++x)
            {
                const double y = f.rnd(l, (5 + x) * N + n);
                double a;
                if (x < 2) // wrapped Gaussian
                    a = arg_deg(r_mf * AS0[x] * y);
                else // wrapped Laplacian with V = Phi(y)
                {
                    const double V = 0.5 * std::erfc(-y / std::sqrt(2.0));
                    const double sg = (V > 0.5) ? 1.0 : (V < 0.5 ? -1.0 : 0.0);
                    a = arg_deg(r_mf * AS0[x] * sg * std::log(1.0 - 2.0 * std::abs(V - 0.5)) / std::sqrt(2.0));
                }
                ap[x].push_back((los && n == 0) ? 0.0 : a);
            }
            if (los && n == 0)
                tp[n] = 0.0;
        }

        // Step 6': eqs. (7.6.30a) to (7.6.30h)
        const double g_ds = std::max(r_mf * DS0 - DS, 0.0) / (DS * r_mf * DS0);
        double g[4];
        for (int x = 0; x < 2; ++x)
            g[x] = std::sqrt(std::max(r_mf * AS0[x] * r_mf * AS0[x] - AS[x] * AS[x], 0.0)) / (AS[x] * r_mf * AS0[x]);
        for (int x = 2; x < 4; ++x)
            g[x] = std::max(r_mf * AS0[x] - AS[x], 0.0) / (AS[x] * r_mf * AS0[x]);
        std::vector<double> P(N_l);
        double sum = 0.0;
        for (arma::uword n = 0; n < N_l; ++n)
        {
            const double ea = ap[0][n] * g[0] / std::sqrt(2.0), eb = ap[1][n] * g[1] / std::sqrt(2.0);
            P[n] = std::exp(-tp[n] * g_ds - ea * ea - eb * eb - std::sqrt(2.0) * std::abs(ap[2][n]) * g[2] -
                            std::sqrt(2.0) * std::abs(ap[3][n]) * g[3]) *
                   std::pow(10.0, -zeta * f.rnd_freq(l, n, k) / 10.0);
            sum += P[n];
        }
        ref_t r;
        for (arma::uword n = 0; n < N_l; ++n)
            r.power.push_back(P[n] / sum / (1.0 + K));
        r.p_los = K / (1.0 + K);
        r.p_first_tr = P[0] / sum / (1.0 + K) + K / (1.0 + K);

        // Step 7': eqs. (7.6.30i) to (7.6.30m) without the ray terms
        const geo_t geo = geometry(f, l);
        const double los_ang[4] = {geo.phi_d, geo.phi_d + 180.0, geo.theta_d, 180.0 - geo.theta_d}; // AOD, AOA, ZOD, ZOA
        const double ct = los ? std::sqrt(1.0 + K / 2.0) : 1.0, ca = los ? std::sqrt(1.0 + K) : 1.0;
        for (arma::uword n = 0; n < N_l; ++n)
        {
            r.delay.push_back(ct * tp[n]);
            for (int x = 0; x < 4; ++x)
            {
                const double a = ca * ap[x][n] + los_ang[x];
                r.ang[x].push_back(x < 2 ? wrap_rad(a * pi / 180.0) : zen2elev(a));
            }
        }
        return r;
    }

    // Compares link l, frequency k of the output with the oracle; NLOS clusters start at 'first_slot'
    void check_link(const out_t &o, const ref_t &r, const fix_t &f, arma::uword l, arma::uword k,
                    arma::uword first_slot, double *max_err = nullptr)
    {
        const double DS0 = f.anchor(l, 0, 0);
        double e_d = 0.0, e_p = 0.0, e_a = 0.0;
        for (arma::uword n = 0; n < r.delay.size(); ++n)
        {
            const arma::uword s = first_slot + n;
            INFO("link " << l << ", frequency " << k << ", cluster " << n);
            e_d = std::max(e_d, std::abs(o.delay(l, s, k) - r.delay[n]) / std::max(r.delay[n], DS0));
            CHECK(std::abs(o.delay(l, s, k) - r.delay[n]) <= 1e-12 * std::max(r.delay[n], DS0));
            e_p = std::max(e_p, std::abs(o.power(l, s, k) / r.power[n] - 1.0));
            CHECK(std::abs(o.power(l, s, k) - r.power[n]) <= 1e-12 * r.power[n]);
            const double da[4] = {wrap_rad(o.aod(l, s, k) - r.ang[0][n]), wrap_rad(o.aoa(l, s, k) - r.ang[1][n]),
                                  o.eod(l, s, k) - r.ang[2][n], o.eoa(l, s, k) - r.ang[3][n]};
            for (double d : da)
            {
                e_a = std::max(e_a, std::abs(d));
                CHECK(std::abs(d) <= 1e-11);
            }
        }
        CHECK(std::abs(o.power(l, 0, k) - r.p_los) <= 1e-12 * r.p_los);
        if (max_err != nullptr)
            max_err[0] = std::max(max_err[0], e_d), max_err[1] = std::max(max_err[1], e_p),
            max_err[2] = std::max(max_err[2], e_a);
    }
}

TEST_CASE("ssf_clusters - 38.901_MF NLOS matches TR 7.6.5.1")
{
    const spmap sp = read_v19("UMi_NLOS");
    const arma::uword N = N_UMi_NLOS, F = 3;
    fix_t f = make_fix_mf(N, 4, F, 201);
    out_t o = run(sp, f, "38.901_MF");
    REQUIRE(o.delay.n_rows == 4);
    REQUIRE(o.delay.n_cols == N + 1);
    REQUIRE(o.delay.n_slices == F);
    double err[3] = {0, 0, 0};
    for (arma::uword l = 0; l < 4; ++l)
        for (arma::uword k = 0; k < F; ++k)
        {
            check_link(o, ref_mf(f, l, k, zeta_UMi, N, false), f, l, k, 1, err);
            CHECK(o.power(l, 0, k) == 0.0);
            CHECK(std::abs(arma::accu(o.power.slice(k)(arma::span(l), arma::span(1, N))) - 1.0) <= 1e-12);
        }
    CHECK(arma::all(o.n_active == N));
    QD_DIAG("# diag: 38.901_MF NLOS max error: delay %.2e rel, power %.2e rel, angle %.2e rad\n", err[0], err[1],
            err[2]);

    // No minimum subtraction: no NLOS cluster at delay 0
    const arma::cube d_nlos = o.delay.cols(1, N);
    CHECK(d_nlos.min() > 0.0);

    // The powers depend on the frequency (spreads and shadowing per frequency)
    CHECK(arma::any(slots(o.power, 1, N, 0) != slots(o.power, 1, N, 1)));

    // Absolute TOA (TR eq. (7.6-43)): added to every NLOS delay at every frequency, LOS slot stays at 0
    f.use_toa = true;
    out_t t = run(sp, f, "38.901_MF");
    for (arma::uword l = 0; l < 4; ++l)
        for (arma::uword k = 0; k < F; ++k)
        {
            CHECK(t.delay(l, 0, k) == 0.0);
            for (arma::uword s = 1; s <= N; ++s)
                CHECK(std::abs(t.delay(l, s, k) - o.delay(l, s, k) - f.toa(l)) <= 1e-15 * t.delay(l, s, k));
        }
    CHECK(eq(t.power, o.power));

    // Per-link cluster number (TR clause 7.6.15): normalisation over the active clusters, inactive slots with delay 0,
    // power 0 and the LOS angles at every frequency
    f.use_toa = false, f.use_ncl = true;
    f.ncl = {N, 6, 11, 14};
    out_t c = run(sp, f, "38.901_MF");
    for (arma::uword l = 0; l < 4; ++l)
    {
        CHECK(c.n_active(l) == f.ncl(l));
        for (arma::uword k = 0; k < F; ++k)
        {
            check_link(c, ref_mf(f, l, k, zeta_UMi, f.ncl(l), false), f, l, k, 1);
            check_inactive(c, l, 1 + f.ncl(l), k);
        }
    }
}

TEST_CASE("ssf_clusters - 38.901_MF LOS scales the wrapped angles by sqrt(1 + K)")
{
    const spmap sp = read_v19("UMi_LOS");
    const arma::uword N = N_UMi_LOS, F = 2;
    fix_t f = make_fix_mf(N, 4, F, 211);
    for (arma::uword l = 0; l < 4; ++l)
    {
        f.lsp(l, 0, 1) = std::pow(10.0, 0.1 * (2.0 + l));  // K 2 to 5 dB at frequency 0
        f.lsp(l, 1, 1) = std::pow(10.0, 0.1 * (11.0 + l)); // K 11 to 14 dB at frequency 1
        // AOA of cluster 2: r·ASA0·y = 1.5·60·2.4 = 216 deg is wrapped to -144 deg (Z_n of step 5')
        f.anchor(l, 0, 4) = 60.0, f.lsp(l, 0, 4) = 45.0 + l, f.lsp(l, 1, 4) = 40.0 + l;
        f.rnd(l, 6 * N + 2) = 2.4;
    }
    out_t o = run(sp, f, "38.901_MF");
    REQUIRE(o.delay.n_slices == F);
    double err[3] = {0, 0, 0}, jump = 1e300;
    for (arma::uword l = 0; l < 4; ++l)
    {
        const double phi_d = geometry(f, l).phi_d;
        for (arma::uword k = 0; k < F; ++k)
        {
            INFO("link " << l << ", frequency " << k);
            const ref_t r = ref_mf(f, l, k, zeta_UMi, N, true);
            check_link(o, r, f, l, k, 1, err);

            // LOS split of eq. (7.6.30h): LOS slot plus first cluster equal P_1 of the TR
            CHECK(std::abs(o.power(l, 0, k) + o.power(l, 1, k) - r.p_first_tr) <= 1e-12 * r.p_first_tr);

            // First cluster at delay 0 in the LOS direction
            CHECK(o.delay(l, 1, k) == 0.0);
            CHECK(std::abs(wrap_rad(o.aod(l, 1, k) - o.aod(l, 0, k))) <= 1e-12);
            CHECK(std::abs(wrap_rad(o.aoa(l, 1, k) - o.aoa(l, 0, k))) <= 1e-12);
            CHECK(std::abs(o.eod(l, 1, k) - o.eod(l, 0, k)) <= 1e-12);
            CHECK(std::abs(o.eoa(l, 1, k) - o.eoa(l, 0, k)) <= 1e-12);

            // TR order: wrap, then scale. Scaling first and wrapping last gives another AOA for cluster 2
            const double K = f.lsp(l, k, 1);
            const double alt = wrap_rad((phi_d + 180.0 - std::sqrt(1.0 + K) * 216.0) * pi / 180.0);
            jump = std::min(jump, std::abs(wrap_rad(o.aoa(l, 3, k) - alt)));
            CHECK(std::abs(wrap_rad(o.aoa(l, 3, k) - alt)) > 0.1);
        }

        // Delays scale with sqrt(1 + K/2) of each frequency (eq. (7.6.30i))
        const double q = std::sqrt(1.0 + f.lsp(l, 1, 1) / 2.0) / std::sqrt(1.0 + f.lsp(l, 0, 1) / 2.0);
        for (arma::uword s = 2; s <= N; ++s)
            CHECK(std::abs(o.delay(l, s, 1) - q * o.delay(l, s, 0)) <= 1e-12 * o.delay(l, s, 1));
    }
    QD_DIAG("# diag: 38.901_MF LOS max error: delay %.2e rel, power %.2e rel, angle %.2e rad; smallest difference "
            "to scale-then-wrap %.3f rad\n",
            err[0], err[1], err[2], jump);
}

TEST_CASE("ssf_clusters - 38.901_MF delays and angles are frequency-independent in NLOS")
{
    spmap sp = read_v19("UMi_NLOS");
    sp["GR_enabled"] = 1.0;
    const arma::uword N = N_UMi_NLOS, F = 3;
    fix_t f = make_fix_mf(N, 5, F, 221);
    f.use_eps = f.use_toa = true;
    out_t o = run(sp, f, "38.901_MF");
    REQUIRE(o.delay.n_slices == F);
    for (arma::uword k = 1; k < F; ++k)
    {
        CHECK(arma::all(arma::vectorise(o.delay.slice(k) == o.delay.slice(0))));
        CHECK(arma::all(arma::vectorise(o.aod.slice(k) == o.aod.slice(0))));
        CHECK(arma::all(arma::vectorise(o.eod.slice(k) == o.eod.slice(0))));
        CHECK(arma::all(arma::vectorise(o.aoa.slice(k) == o.aoa.slice(0))));
        CHECK(arma::all(arma::vectorise(o.eoa.slice(k) == o.eoa.slice(0))));
        CHECK(arma::any(arma::vectorise(o.power.slice(k) != o.power.slice(0))));
    }

    // LOS with the same K at all frequencies: delays and angles equal at all frequencies, too
    const spmap sl = read_v19("UMi_LOS");
    fix_t g = make_fix_mf(N_UMi_LOS, 5, F, 222);
    for (arma::uword k = 1; k < F; ++k)
        g.lsp.slice(1).col(k) = g.lsp.slice(1).col(0);
    out_t a = run(sl, g, "38.901_MF");
    for (arma::uword k = 1; k < F; ++k)
    {
        CHECK(arma::all(arma::vectorise(a.delay.slice(k) == a.delay.slice(0))));
        CHECK(arma::all(arma::vectorise(a.aod.slice(k) == a.aod.slice(0))));
        CHECK(arma::all(arma::vectorise(a.eoa.slice(k) == a.eoa.slice(0))));
    }
}

TEST_CASE("ssf_clusters - 38.901_MF g factors vanish when the spread exceeds r times the anchor spread")
{
    const spmap sp = read_v19("UMi_NLOS");
    const arma::uword N = N_UMi_NLOS, F = 2;
    fix_t f = make_fix_mf(N, 4, F, 231);
    for (arma::uword l = 0; l < 4; ++l)
        f.lsp(l, 0, 0) = 1.6 * f.anchor(l, 0, 0), f.lsp(l, 1, 0) = r_mf * f.anchor(l, 0, 0); // g_DS = 0
    out_t o = run(sp, f, "38.901_MF");
    for (arma::uword l = 0; l < 4; ++l)
        for (arma::uword k = 0; k < F; ++k)
            check_link(o, ref_mf(f, l, k, zeta_UMi, N, false), f, l, k, 1);

    // Other delays, same powers
    fix_t g = f;
    for (arma::uword n = 0; n < N; ++n)
        for (arma::uword l = 0; l < 4; ++l)
            g.rnd(l, n) = 1.0 - f.rnd(l, n);
    out_t b = run(sp, g, "38.901_MF");
    CHECK(arma::any(arma::vectorise(b.delay != o.delay)));
    CHECK(eq(b.power, o.power));

    // All spreads at r times the anchor or above: the powers follow the per-cluster shadowing only
    for (arma::uword l = 0; l < 4; ++l)
        for (arma::uword k = 0; k < F; ++k)
            for (arma::uword s : {3, 4, 5, 6})
                f.lsp(l, k, s) = (k == 0 ? 1.7 : r_mf) * f.anchor(l, 0, s);
    out_t c = run(sp, f, "38.901_MF");
    for (arma::uword l = 0; l < 4; ++l)
        for (arma::uword k = 0; k < F; ++k)
        {
            double sum = 0.0;
            for (arma::uword n = 0; n < N; ++n)
                sum += std::pow(10.0, -zeta_UMi * f.rnd_freq(l, n, k) / 10.0);
            for (arma::uword n = 0; n < N; ++n)
            {
                const double p = std::pow(10.0, -zeta_UMi * f.rnd_freq(l, n, k) / 10.0) / sum;
                CHECK(std::abs(c.power(l, 1 + n, k) - p) <= 1e-12 * p);
            }
        }
}
