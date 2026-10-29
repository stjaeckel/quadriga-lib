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
// vanishing g factors; the inputs the method ignores (EoD offset, O2I flag); the absolute time of arrival and the
// per-link cluster number; swap reciprocity; the output sizes with one delay and angle slice per frequency; the
// ground reflection per frequency (TR clause 7.6.8); the LOS-only case NumClusters = 1; finite outputs for extreme
// random values.
// Oracles, independent of the function and of the conf files:
//  - ref_mf: TR 38.901 clause 7.6.5.1 typed here in scalar double, in degrees, with the TR's own forms: the principal
//    value of arg(exp(-j·x)) taken from std::arg, the Laplacian term sgn(V - 0.5)·ln(1 - 2·abs(V - 0.5)) with
//    V = Phi(y) (all fixture values have abs(y) < 2.5), P'_n = exp(...)·10^(-Q_n/10) with Q_n = zeta·z_n and the
//    linear normalisation of eqs. (7.6.30g), (7.6.30h); r = 1.5 and zeta of TR Table 7.5-6 are typed here
//  - the LOS direction from the positions: azimuth atan2(dy, dx), zenith acos(dz / d3D) (TR clause 7.5 step 1c);
//    zenith angles are folded by the rule of TR clause 7.5 step 7 (wrap to [0, 360], [180, 360] -> 360 - theta)
//  - the ground reflection of TR eqs. (7.6-32) to (7.6-40) with eps = eps_re - j·eps_im, slot power
//    K/(K+1)·(d3D/d_GR)²·(abs(R_par)² + abs(R_perp)²)/2 per frequency
// Tolerances: delays 1e-12 relative to max(delay, DS0); powers 1e-12 relative; angles 1e-11 rad; values that the
// function forms by the same operations are compared bitwise.
// Conventions: positions are rows {x, y, z} in meters; the input fixtures have distinct values per link, cluster
// and frequency, positions off the axes, and Tx positions per link where the case allows it.

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
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

namespace
{
    typedef std::unordered_map<std::string, double> spmap;
    const double pi = 3.141592653589793;
    const double c0 = 299792458.0;
    const double r_mf = 1.5; // proportionality factor r of TR clause 7.6.5.1

    // TR 38.901 Table 7.5-6: per-cluster shadowing std zeta [dB] and number of clusters N
    const double zeta_UMi = 3.0;
    const arma::uword N_UMi_LOS = 12, N_UMi_NLOS = 19;

    spmap read_v19(const std::string &name)
    {
        spmap sp;
        quadriga_lib::conf_file_read("data/config/3GPP_38.901_v19_" + name + ".conf", sp);
        return sp;
    }

    double wrap_rad(double x) { return x - 2.0 * pi * std::floor((x + pi) / (2.0 * pi)); }

    // Zenith angle in degrees to elevation in rad: wrapped to [0, 360], [180, 360] -> 360 - theta (TR 7.5 step 7)
    double zen2elev(double t)
    {
        t = t - 360.0 * std::floor(t / 360.0);
        if (t >= 180.0)
            t = 360.0 - t;
        return (90.0 - t) * pi / 180.0;
    }

    // Principal value of arg(exp(-j·x)) in degrees, x in degrees (TR clause 7.6.5.1 step 5')
    double arg_deg(double x) { return std::arg(std::polar(1.0, -x * pi / 180.0)) * 180.0 / pi; }

    // Input fixture of one call; F frequencies
    struct fix_t
    {
        arma::mat tx, rx, rnd, eod, eps_re, eps_im;
        arma::cube lsp, anchor, rnd_freq;
        arma::vec toa;
        std::vector<bool> o2i;
        arma::uvec ncl;
        bool use_rf = true, use_anchor = true, use_eod = false, use_toa = false, use_eps = false, use_o2i = false,
             use_ncl = false;
    };

    // Distinct values: BS at 25 m, UTs 1.5 to 22.5 m, d2D 35 to 300 m; anchor DS0 40 to 400 ns, ASD0/ASA0 20 to
    // 60 deg, ZSD0/ZSA0 3 to 20 deg; spreads at the frequencies 0.5 to 1.2 times the anchor spread (g factors > 0);
    // K in [kdb_lo, kdb_hi] dB per frequency; delay values in (0.02, 0.98), Normal values in (-2.5, 2.5)
    fix_t make_fix(arma::uword N, arma::uword n_link, arma::uword F, unsigned seed, bool tx_per_link,
                   double kdb_lo = 0.0, double kdb_hi = 15.0)
    {
        std::mt19937 g(seed);
        auto U = [&](double a, double b)
        { return a + (b - a) * ((double)g() + 0.5) / 4294967296.0; };
        fix_t f;
        f.tx.set_size(tx_per_link ? n_link : 1, 3);
        for (arma::uword t = 0; t < f.tx.n_rows; ++t)
            f.tx(t, 0) = U(-40.0, 40.0), f.tx(t, 1) = U(-40.0, 40.0), f.tx(t, 2) = 25.0;
        f.rx.set_size(n_link, 3);
        f.lsp.zeros(n_link, F, 8), f.anchor.zeros(n_link, 1, 8);
        for (arma::uword l = 0; l < n_link; ++l)
        {
            const arma::uword t = tx_per_link ? l : 0;
            const double d = U(35.0, 300.0), a = U(-pi, pi);
            f.rx(l, 0) = f.tx(t, 0) + d * std::cos(a), f.rx(l, 1) = f.tx(t, 1) + d * std::sin(a);
            f.rx(l, 2) = U(1.5, 22.5);
            f.anchor(l, 0, 0) = U(40e-9, 400e-9), f.anchor(l, 0, 1) = 5.0, f.anchor(l, 0, 2) = 1.0;
            f.anchor(l, 0, 3) = U(20.0, 60.0), f.anchor(l, 0, 4) = U(20.0, 60.0);
            f.anchor(l, 0, 5) = U(3.0, 20.0), f.anchor(l, 0, 6) = U(3.0, 20.0), f.anchor(l, 0, 7) = 10.0;
            for (arma::uword k = 0; k < F; ++k)
            {
                for (arma::uword s : {0, 3, 4, 5, 6})
                    f.lsp(l, k, s) = f.anchor(l, 0, s) * U(0.5, 1.2);
                f.lsp(l, k, 1) = std::pow(10.0, 0.1 * U(kdb_lo, kdb_hi));
                f.lsp(l, k, 2) = 1.0, f.lsp(l, k, 7) = 10.0;
            }
        }
        f.rnd.set_size(n_link, 9 * N);
        for (arma::uword c = 0; c < 9 * N; ++c)
            for (arma::uword l = 0; l < n_link; ++l)
                f.rnd(l, c) = (c < N) ? U(0.02, 0.98) : U(-2.5, 2.5);
        f.rnd_freq.set_size(n_link, N, F);
        for (arma::uword i = 0; i < f.rnd_freq.n_elem; ++i)
            f.rnd_freq(i) = U(-2.5, 2.5);
        f.eod.set_size(n_link, F), f.eps_re.set_size(n_link, F), f.eps_im.set_size(n_link, F);
        for (arma::uword i = 0; i < f.eod.n_elem; ++i)
            f.eod(i) = U(-3.0, 3.0), f.eps_re(i) = U(3.0, 30.0), f.eps_im(i) = U(0.1, 2.0);
        f.toa.set_size(n_link);
        for (arma::uword l = 0; l < n_link; ++l)
            f.toa(l) = U(10e-9, 300e-9);
        f.o2i.assign(n_link, true);
        f.ncl.set_size(n_link);
        f.ncl.fill(N);
        return f;
    }

    struct out_t
    {
        arma::cube delay, power, aod, eod, aoa, eoa;
        arma::uvec n_active;
    };

    out_t run(const spmap &sp, const fix_t &f)
    {
        out_t o;
        quadriga_lib::ssf_clusters(sp, f.tx, f.rx, f.lsp, f.rnd, o.delay, o.power, o.aod, o.eod, o.aoa, o.eoa,
                                   "38.901_MF", f.use_rf ? &f.rnd_freq : nullptr, f.use_anchor ? &f.anchor : nullptr,
                                   f.use_eod ? &f.eod : nullptr, f.use_toa ? &f.toa : nullptr,
                                   f.use_eps ? &f.eps_re : nullptr, f.use_eps ? &f.eps_im : nullptr,
                                   f.use_o2i ? &f.o2i : nullptr, f.use_ncl ? &f.ncl : nullptr, &o.n_active);
        return o;
    }

    bool eq(const arma::cube &x, const arma::cube &y)
    {
        return arma::size(x) == arma::size(y) && arma::all(arma::vectorise(x) == arma::vectorise(y));
    }
    bool equal(const out_t &a, const out_t &b)
    {
        return eq(a.delay, b.delay) && eq(a.power, b.power) && eq(a.aod, b.aod) && eq(a.eod, b.eod) &&
               eq(a.aoa, b.aoa) && eq(a.eoa, b.eoa) && arma::all(a.n_active == b.n_active);
    }

    // LOS direction of link l in degrees: azimuth and zenith of departure (TR clause 7.5 step 1c)
    void los_dir(const fix_t &f, arma::uword l, double &phi_d, double &theta_d, double &d2D, double &d3D)
    {
        const arma::uword t = (f.tx.n_rows == 1) ? 0 : l;
        const double dx = f.rx(l, 0) - f.tx(t, 0), dy = f.rx(l, 1) - f.tx(t, 1), dz = f.rx(l, 2) - f.tx(t, 2);
        d2D = std::sqrt(dx * dx + dy * dy), d3D = std::sqrt(dx * dx + dy * dy + dz * dz);
        phi_d = std::atan2(dy, dx) * 180.0 / pi;
        theta_d = std::acos(dz / d3D) * 180.0 / pi;
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
        double phi_d, theta_d, d2D, d3D;
        los_dir(f, l, phi_d, theta_d, d2D, d3D);
        const double los_ang[4] = {phi_d, phi_d + 180.0, theta_d, 180.0 - theta_d}; // AOD, AOA, ZOD, ZOA
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

    // Polarisation-averaged reflection power of TR eqs. (7.6-37a), (7.6-38) to (7.6-40), eps = eps_re - j·eps_im
    double ref_r2(double d2D, double h_t, double h_r, double eps_re, double eps_im)
    {
        const double th = (180.0 - std::atan(d2D / (h_r + h_t)) * 180.0 / pi) * pi / 180.0;
        const std::complex<double> eps(eps_re, -eps_im);
        const std::complex<double> q = std::sqrt(eps - std::sin(th) * std::sin(th));
        const std::complex<double> R_par = (eps * std::cos(th) + q) / (eps * std::cos(th) - q);
        const std::complex<double> R_perp = (std::cos(th) + q) / (std::cos(th) - q);
        return 0.5 * (std::norm(R_par) + std::norm(R_perp));
    }

    // Slots a to b of slice k of a cube
    arma::vec slots(const arma::cube &c, arma::uword a, arma::uword b, arma::uword k)
    {
        arma::mat t = c.slice(k).cols(a, b);
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

TEST_CASE("ssf_clusters - 38.901_MF NLOS matches TR 7.6.5.1")
{
    const spmap sp = read_v19("UMi_NLOS");
    const arma::uword N = N_UMi_NLOS, F = 3;
    fix_t f = make_fix(N, 4, F, 201, true);
    out_t o = run(sp, f);
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
    out_t t = run(sp, f);
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
    out_t c = run(sp, f);
    for (arma::uword l = 0; l < 4; ++l)
    {
        CHECK(c.n_active(l) == f.ncl(l));
        for (arma::uword k = 0; k < F; ++k)
        {
            check_link(c, ref_mf(f, l, k, zeta_UMi, f.ncl(l), false), f, l, k, 1);
            for (arma::uword s = 1 + f.ncl(l); s <= N; ++s)
            {
                CHECK(c.delay(l, s, k) == 0.0);
                CHECK(c.power(l, s, k) == 0.0);
                CHECK(c.aod(l, s, k) == c.aod(l, 0, k));
                CHECK(c.eod(l, s, k) == c.eod(l, 0, k));
                CHECK(c.aoa(l, s, k) == c.aoa(l, 0, k));
                CHECK(c.eoa(l, s, k) == c.eoa(l, 0, k));
            }
        }
    }
}

TEST_CASE("ssf_clusters - 38.901_MF LOS scales the wrapped angles by sqrt(1 + K)")
{
    const spmap sp = read_v19("UMi_LOS");
    const arma::uword N = N_UMi_LOS, F = 2;
    fix_t f = make_fix(N, 4, F, 211, true);
    for (arma::uword l = 0; l < 4; ++l)
    {
        f.lsp(l, 0, 1) = std::pow(10.0, 0.1 * (2.0 + l));  // K 2 to 5 dB at frequency 0
        f.lsp(l, 1, 1) = std::pow(10.0, 0.1 * (11.0 + l)); // K 11 to 14 dB at frequency 1
        // AOA of cluster 2: r·ASA0·y = 1.5·60·2.4 = 216 deg is wrapped to -144 deg (Z_n of step 5')
        f.anchor(l, 0, 4) = 60.0, f.lsp(l, 0, 4) = 45.0 + l, f.lsp(l, 1, 4) = 40.0 + l;
        f.rnd(l, 6 * N + 2) = 2.4;
    }
    out_t o = run(sp, f);
    REQUIRE(o.delay.n_slices == F);
    double err[3] = {0, 0, 0}, jump = 1e300;
    for (arma::uword l = 0; l < 4; ++l)
    {
        double phi_d, theta_d, d2D, d3D;
        los_dir(f, l, phi_d, theta_d, d2D, d3D);
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
    fix_t f = make_fix(N, 5, F, 221, true);
    f.use_eps = f.use_toa = true;
    out_t o = run(sp, f);
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
    fix_t g = make_fix(N_UMi_LOS, 5, F, 222, true);
    for (arma::uword k = 1; k < F; ++k)
        g.lsp.slice(1).col(k) = g.lsp.slice(1).col(0);
    out_t a = run(sl, g);
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
    fix_t f = make_fix(N, 4, F, 231, true);
    for (arma::uword l = 0; l < 4; ++l)
        f.lsp(l, 0, 0) = 1.6 * f.anchor(l, 0, 0), f.lsp(l, 1, 0) = r_mf * f.anchor(l, 0, 0); // g_DS = 0
    out_t o = run(sp, f);
    for (arma::uword l = 0; l < 4; ++l)
        for (arma::uword k = 0; k < F; ++k)
            check_link(o, ref_mf(f, l, k, zeta_UMi, N, false), f, l, k, 1);

    // Other delays, same powers
    fix_t g = f;
    for (arma::uword n = 0; n < N; ++n)
        for (arma::uword l = 0; l < 4; ++l)
            g.rnd(l, n) = 1.0 - f.rnd(l, n);
    out_t b = run(sp, g);
    CHECK(arma::any(arma::vectorise(b.delay != o.delay)));
    CHECK(eq(b.power, o.power));

    // All spreads at r times the anchor or above: the powers follow the per-cluster shadowing only
    for (arma::uword l = 0; l < 4; ++l)
        for (arma::uword k = 0; k < F; ++k)
            for (arma::uword s : {3, 4, 5, 6})
                f.lsp(l, k, s) = (k == 0 ? 1.7 : r_mf) * f.anchor(l, 0, s);
    out_t c = run(sp, f);
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

TEST_CASE("ssf_clusters - 38.901_MF ignores the EoD offset and the O2I flag")
{
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
    {
        INFO(name);
        const spmap sp = read_v19(name);
        fix_t f = make_fix(std::string(name) == "UMi_LOS" ? N_UMi_LOS : N_UMi_NLOS, 4, 2, 241, true);
        out_t a = run(sp, f);
        f.use_eod = f.use_o2i = true;
        CHECK(equal(run(sp, f), a));
    }
}

TEST_CASE("ssf_clusters - 38.901_MF is swap-reciprocal for mirrored inputs")
{
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
    {
        const arma::uword N = (std::string(name) == "UMi_LOS") ? N_UMi_LOS : N_UMi_NLOS;
        const spmap sp = read_v19(name);
        fix_t f = make_fix(N, 6, 2, 251, true);
        fix_t s = f;
        s.tx = f.rx, s.rx = f.tx;
        for (arma::cube *c : {&s.lsp, &s.anchor})
        {
            const arma::cube &o = (c == &s.lsp) ? f.lsp : f.anchor;
            c->slice(3) = o.slice(4), c->slice(4) = o.slice(3), c->slice(5) = o.slice(6), c->slice(6) = o.slice(5);
        }
        const arma::uword mb[9] = {0, 2, 1, 4, 3, 6, 5, 8, 7};
        for (arma::uword b = 0; b < 9; ++b)
            s.rnd.cols(b * N, b * N + N - 1) = f.rnd.cols(mb[b] * N, mb[b] * N + N - 1);
        out_t a = run(sp, f), b = run(sp, s);
        INFO(name);
        double e = 0.0;
        for (arma::uword i = 0; i < a.delay.n_elem; ++i)
        {
            const arma::uword l = i % 6;
            CHECK(std::abs(a.delay(i) - b.delay(i)) <= 1e-12 * f.anchor(l, 0, 0));
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
        QD_DIAG("# diag: 38.901_MF swap %s, max angle difference %.2e rad\n", name, e);
    }
}

TEST_CASE("ssf_clusters - 38.901_MF output sizes")
{
    spmap sp = read_v19("UMi_LOS");
    const arma::uword N = N_UMi_LOS, F = 3;
    fix_t f = make_fix(N, 5, F, 261, false);
    out_t o_off = run(sp, f);
    sp["GR_enabled"] = 1.0;
    f.use_eps = true;
    out_t o = run(sp, f);
    for (const arma::cube *c : {&o.delay, &o.power, &o.aod, &o.eod, &o.aoa, &o.eoa})
    {
        CHECK(c->n_rows == 5);
        CHECK(c->n_cols == N + 2);
        CHECK(c->n_slices == F);
    }
    CHECK(o.n_active.n_elem == 5);
    CHECK(arma::all(arma::vectorise(o.eod) >= -pi / 2.0));
    CHECK(arma::all(arma::vectorise(o.eod) <= pi / 2.0));
    CHECK(arma::all(arma::vectorise(o.eoa) >= -pi / 2.0));
    CHECK(arma::all(arma::vectorise(o.eoa) <= pi / 2.0));
    CHECK(arma::all(arma::vectorise(o.aod) >= -pi));
    CHECK(arma::all(arma::vectorise(o.aod) < pi));
    CHECK(arma::all(arma::vectorise(o.aoa) >= -pi));
    CHECK(arma::all(arma::vectorise(o.aoa) < pi));

    // Ground reflection per frequency (TR clause 7.6.8): geometry at every frequency, power with R² of each frequency
    double err = 0.0;
    for (arma::uword l = 0; l < 5; ++l)
    {
        double phi_d, theta_d, d2D, d3D;
        los_dir(f, l, phi_d, theta_d, d2D, d3D);
        const double h_t = f.tx(0, 2), h_r = f.rx(l, 2);
        const double d_gr = std::sqrt(d2D * d2D + (h_t + h_r) * (h_t + h_r));
        const double el = (90.0 - (180.0 - std::atan(d2D / (h_r + h_t)) * 180.0 / pi)) * pi / 180.0;
        for (arma::uword k = 0; k < F; ++k)
        {
            INFO("link " << l << ", frequency " << k);
            CHECK(std::abs(o.delay(l, 1, k) - (d_gr - d3D) / c0) <= 1e-12 * (d_gr - d3D) / c0);
            CHECK(o.aod(l, 1, k) == o.aod(l, 0, k));
            CHECK(o.aoa(l, 1, k) == o.aoa(l, 0, k));
            CHECK(std::abs(o.eod(l, 1, k) - el) <= 1e-12);
            CHECK(std::abs(o.eoa(l, 1, k) - el) <= 1e-12);
            const double K = f.lsp(l, k, 1);
            const double p = K / (K + 1.0) * (d3D / d_gr) * (d3D / d_gr) *
                             ref_r2(d2D, h_t, h_r, f.eps_re(l, k), f.eps_im(l, k));
            err = std::max(err, std::abs(o.power(l, 1, k) / p - 1.0));
            CHECK(std::abs(o.power(l, 1, k) - p) <= 1e-12 * p);
            CHECK(o.power(l, 0, k) == o_off.power(l, 0, k));
        }
    }
    QD_DIAG("# diag: 38.901_MF GR power max error %.2e rel\n", err);

    // NLOS slots as without the reflection, shifted by one slot
    for (arma::uword k = 0; k < F; ++k)
    {
        CHECK(arma::all(slots(o.delay, 2, N + 1, k) == slots(o_off.delay, 1, N, k)));
        CHECK(arma::all(slots(o.power, 2, N + 1, k) == slots(o_off.power, 1, N, k)));
        CHECK(arma::all(slots(o.aod, 2, N + 1, k) == slots(o_off.aod, 1, N, k)));
        CHECK(arma::all(slots(o.eoa, 2, N + 1, k) == slots(o_off.eoa, 1, N, k)));
    }
}

TEST_CASE("ssf_clusters - 38.901_MF one cluster gives the LOS path only")
{
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
        for (int gr = 0; gr < 2; ++gr)
        {
            INFO(name << ", GR " << gr);
            spmap sp = read_v19(name);
            sp["NumClusters"] = 1.0, sp["NumClusters_min"] = 0.0, sp["GR_enabled"] = gr;
            fix_t f = make_fix(0, 4, 2, 271, true);
            f.use_rf = f.use_anchor = false;
            f.use_eps = gr == 1;
            out_t o = run(sp, f);
            for (const arma::cube *c : {&o.delay, &o.power, &o.aod, &o.eod, &o.aoa, &o.eoa})
            {
                CHECK(c->n_cols == (arma::uword)(1 + gr));
                CHECK(c->n_slices == 2);
            }
            for (arma::uword l = 0; l < 4; ++l)
            {
                CHECK(o.n_active(l) == 0);
                double phi_d, theta_d, d2D, d3D;
                los_dir(f, l, phi_d, theta_d, d2D, d3D);
                const double h_t = f.tx(l, 2), h_r = f.rx(l, 2);
                const double d_gr = std::sqrt(d2D * d2D + (h_t + h_r) * (h_t + h_r));
                for (arma::uword k = 0; k < 2; ++k)
                {
                    CHECK(o.power(l, 0, k) == 1.0);
                    CHECK(o.delay(l, 0, k) == 0.0);
                    CHECK(std::abs(wrap_rad(o.aod(l, 0, k) - phi_d * pi / 180.0)) <= 1e-12);
                    CHECK(std::abs(o.eod(l, 0, k) - (90.0 - theta_d) * pi / 180.0) <= 1e-12);
                    if (gr == 1)
                    {
                        const double p = (d3D / d_gr) * (d3D / d_gr) *
                                         ref_r2(d2D, h_t, h_r, f.eps_re(l, k), f.eps_im(l, k));
                        CHECK(std::abs(o.power(l, 1, k) - p) <= 1e-12 * p);
                    }
                }
            }
        }
}

TEST_CASE("ssf_clusters - 38.901_MF extreme finite random values give finite powers")
{
    for (const char *name : {"UMi_LOS", "UMi_NLOS"})
    {
        const bool los = std::string(name) == "UMi_LOS";
        const arma::uword N = los ? N_UMi_LOS : N_UMi_NLOS, F = 2;
        INFO(name);
        spmap sp = read_v19(name);
        sp["GR_enabled"] = 1.0;
        fix_t f = make_fix(N, 3, F, 281, true);
        f.use_eps = true;

        // z = -1100 on one cluster at frequency 1: 10^(330) overflows in the linear form of eq. (7.6.30a)
        for (arma::uword l = 0; l < 3; ++l)
            f.rnd_freq(l, 3 + l, 1) = -1100.0;
        out_t o = run(sp, f);
        check_finite(o);
        for (arma::uword l = 0; l < 3; ++l)
        {
            const double K = los ? f.lsp(l, 1, 1) : 0.0;
            CHECK(std::abs(o.power(l, 2 + 3 + l, 1) - 1.0 / (K + 1.0)) <= 1e-15);
            CHECK(arma::all(arma::vectorise(o.power.row(l)) >= 0.0));
        }

        // Normal values at +-40 (erfc underflows in the Laplacian term), delay values 0 and 1
        for (arma::uword c = 0; c < 9 * N; ++c)
            for (arma::uword l = 0; l < 3; ++l)
                f.rnd(l, c) = (c < N) ? (double)((c + l) % 2) : ((c + l) % 2 ? 40.0 : -40.0);
        for (arma::uword i = 0; i < f.rnd_freq.n_elem; ++i)
            f.rnd_freq(i) = (i % 3 == 0) ? 40.0 : -40.0;
        o = run(sp, f);
        check_finite(o);
        CHECK(arma::all(arma::vectorise(o.power) >= 0.0));
    }
}

TEST_CASE("ssf_clusters - 38.901_MF accepts valid inputs and ignores the inputs it does not use")
{
    spmap sp = read_v19("UMi_LOS");
    sp["GR_enabled"] = 1.0;
    fix_t f = make_fix(N_UMi_LOS, 3, 2, 291, true);
    f.use_eps = f.use_eod = f.use_toa = f.use_o2i = f.use_ncl = true;
    CHECK_NOTHROW(run(sp, f));

    // Wrong sizes of the EoD offset and of the O2I flags are not checked
    f.eod.set_size(1, 7), f.eod.fill(NAN);
    f.o2i.assign(5, false);
    CHECK_NOTHROW(run(sp, f));
}
