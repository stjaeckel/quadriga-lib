// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Fixture and geometry of the ssf_clusters tests (test_ssf_clusters.cpp, test_ssf_clusters_mf.cpp,
// test_ssf_clusters_quadriga.cpp), independent of the function under test:
//  - fix_t / make_fix: one call's inputs with distinct values per link, cluster and frequency: BS at 25 m, UTs 1.5 to
//    22.5 m, d2D 35 to 300 m, positions off the axes; DS 40 to 400 ns, AS in [as_lo, as_hi] deg, ES in [es_lo, es_hi]
//    deg at frequency 0 and the anchor, each spread at frequency k > 0 a factor 0.5 to 1.2 of it; K in
//    [kdb_lo, kdb_hi] dB per frequency; delay values in (0.02, 0.98), Normal values in (-2.5, 2.5); every optional
//    input filled, passed by run() where its use_* flag is set
//  - geometry: the LOS direction from the positions (TR 38.901 clause 7.5 step 1c: azimuth atan2(dy, dx), zenith
//    acos(dz / d3D); elevation atan(dz / d2D)) and the ground reflection of TR eqs. (7.6-33) to (7.6-40): delay
//    (d_GR - d3D)/c, elevation -atan((h_t + h_r)/d2D), R_par and R_perp with the TR's trigonometric forms and
//    eps = eps_re - j·eps_im, R² = (abs(R_par)² + abs(R_perp)²)/2 per frequency

#ifndef QD_TEST_SSF_HPP
#define QD_TEST_SSF_HPP

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

namespace
{
    typedef std::unordered_map<std::string, double> spmap;
    const double pi = arma::datum::pi;
    const double c0 = 299792458.0;
    const char *const qd_methods[3] = {"38.901", "38.901_MF", "quadriga"};
    const arma::uword N_UMi_LOS = 12, N_UMi_NLOS = 19; // TR 38.901 Table 7.5-6: NLOS clusters N = NumClusters - 1

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

    fix_t make_fix(arma::uword N, arma::uword n_link, arma::uword F, unsigned seed, bool tx_per_link,
                   double kdb_lo = 0.0, double kdb_hi = 15.0, double as_lo = 5.0, double as_hi = 60.0,
                   double es_lo = 1.0, double es_hi = 20.0)
    {
        std::mt19937 g(seed);
        auto U = [&](double a, double b)
        { return a + (b - a) * ((double)g() + 0.5) / 4294967296.0; };
        const arma::uword si[5] = {0, 3, 4, 5, 6}; // DS, ASD, ASA, ESD, ESA
        fix_t f;
        f.tx.set_size(tx_per_link ? n_link : 1, 3), f.rx.set_size(n_link, 3);
        f.lsp.zeros(n_link, F, 8), f.anchor.zeros(n_link, 1, 8);
        for (arma::uword t = 0; t < f.tx.n_rows; ++t)
            f.tx(t, 0) = U(-40.0, 40.0), f.tx(t, 1) = U(-40.0, 40.0), f.tx(t, 2) = 25.0;
        for (arma::uword l = 0; l < n_link; ++l)
        {
            const arma::uword t = tx_per_link ? l : 0;
            const double d = U(35.0, 300.0), a = U(-pi, pi);
            f.rx(l, 0) = f.tx(t, 0) + d * std::cos(a), f.rx(l, 1) = f.tx(t, 1) + d * std::sin(a);
            f.rx(l, 2) = U(1.5, 22.5);
            const double s0[5] = {U(40e-9, 400e-9), U(as_lo, as_hi), U(as_lo, as_hi), U(es_lo, es_hi), U(es_lo, es_hi)};
            for (arma::uword i = 0; i < 5; ++i)
                f.anchor(l, 0, si[i]) = s0[i];
            f.anchor(l, 0, 1) = 5.0, f.anchor(l, 0, 2) = 1.0, f.anchor(l, 0, 7) = 10.0;
            for (arma::uword k = 0; k < F; ++k)
            {
                for (arma::uword i = 0; i < 5; ++i)
                    f.lsp(l, k, si[i]) = s0[i] * (k == 0 ? 1.0 : U(0.5, 1.2));
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
            f.toa(l) = U(10e-9, 300e-9), f.o2i.push_back(l % 2 == 0);
        f.ncl.set_size(n_link);
        f.ncl.fill(N);
        return f;
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
                                   f.use_rf ? &f.rnd_freq : nullptr, f.use_anchor ? &f.anchor : nullptr,
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

    // Reflection coefficients of TR eqs. (7.6-37a), (7.6-38) to (7.6-40); 'sign' of the imaginary part: -1 = TR sign
    // eps_re - j·eps_im, +1 = conjugate
    void ref_r(double d2D, double h_t, double h_r, double eps_re, double eps_im, std::complex<double> &R_par,
               std::complex<double> &R_perp, double sign = -1.0)
    {
        const double th = pi - std::atan(d2D / (h_r + h_t)); // zenith of the ground reflection at the Tx
        const std::complex<double> eps(eps_re, sign * eps_im);
        const std::complex<double> q = std::sqrt(eps - std::sin(th) * std::sin(th));
        R_par = (eps * std::cos(th) + q) / (eps * std::cos(th) - q);
        R_perp = (std::cos(th) + q) / (std::cos(th) - q);
    }

    // Geometry of link l: LOS angles {AOD, AOA, EOD, EOA} [rad], azimuth and zenith of departure [deg]; ground
    // reflection with delay [s], elevation [rad] and R² per frequency
    struct geo_t
    {
        double ang[4], phi_d, theta_d, d2D, d3D, h_t, h_r, d_gr, tau_gr, e_gr;
        std::vector<double> R2;
    };
    geo_t geometry(const fix_t &f, arma::uword l)
    {
        geo_t g;
        const arma::uword t = (f.tx.n_rows == 1) ? 0 : l;
        const double dx = f.rx(l, 0) - f.tx(t, 0), dy = f.rx(l, 1) - f.tx(t, 1), dz = f.rx(l, 2) - f.tx(t, 2);
        g.d2D = std::sqrt(dx * dx + dy * dy), g.d3D = std::sqrt(dx * dx + dy * dy + dz * dz);
        g.ang[0] = std::atan2(dy, dx), g.ang[1] = wrap_rad(g.ang[0] + pi);
        g.ang[2] = std::atan(dz / g.d2D), g.ang[3] = -g.ang[2];
        g.phi_d = g.ang[0] * 180.0 / pi, g.theta_d = std::acos(dz / g.d3D) * 180.0 / pi;
        g.h_t = f.tx(t, 2), g.h_r = f.rx(l, 2);
        g.d_gr = std::sqrt(g.d2D * g.d2D + (g.h_t + g.h_r) * (g.h_t + g.h_r));
        g.tau_gr = (g.d_gr - g.d3D) / c0;
        g.e_gr = -std::atan((g.h_t + g.h_r) / g.d2D);
        for (arma::uword k = 0; k < f.eps_re.n_cols; ++k)
        {
            std::complex<double> R_par, R_perp;
            ref_r(g.d2D, g.h_t, g.h_r, f.eps_re(l, k), f.eps_im(l, k), R_par, R_perp);
            g.R2.push_back(0.5 * (std::norm(R_par) + std::norm(R_perp)));
        }
        return g;
    }

    // Row l of a cube as a vector (slots, then slices)
    arma::vec rv(const arma::cube &c, arma::uword l)
    {
        arma::cube t = c.row(l);
        return arma::vectorise(t);
    }

    // Slots a to b of slice k of a cube
    arma::vec slots(const arma::cube &c, arma::uword a, arma::uword b, arma::uword k = 0)
    {
        arma::mat t = c.slice(k).cols(a, b);
        return arma::vectorise(t);
    }

    void check_finite(const out_t &o)
    {
        for (const arma::cube *c : {&o.delay, &o.power, &o.aod, &o.eod, &o.aoa, &o.eoa})
            CHECK(c->is_finite());
    }

    // Inactive slots of link l at frequency k from slot s0 on: delay 0, power 0, the angles of slot 0
    void check_inactive(const out_t &o, arma::uword l, arma::uword s0, arma::uword k = 0)
    {
        for (arma::uword s = s0; s < o.delay.n_cols; ++s)
        {
            CHECK((o.delay(l, s, k) == 0.0 && o.power(l, s, k) == 0.0));
            CHECK((o.aod(l, s, k) == o.aod(l, 0, k) && o.eod(l, s, k) == o.eod(l, 0, k)));
            CHECK((o.aoa(l, s, k) == o.aoa(l, 0, k) && o.eoa(l, s, k) == o.eoa(l, 0, k)));
        }
    }

    // Ranges of the outputs: azimuths in [-pi, pi), elevations in [-pi/2, pi/2], powers and delays >= 0
    void check_ranges(const out_t &o)
    {
        for (const arma::cube *c : {&o.aod, &o.aoa})
            CHECK((arma::all(arma::vectorise(*c) >= -pi) && arma::all(arma::vectorise(*c) < pi)));
        for (const arma::cube *c : {&o.eod, &o.eoa})
            CHECK((arma::all(arma::vectorise(*c) >= -pi / 2.0) && arma::all(arma::vectorise(*c) <= pi / 2.0)));
        CHECK(arma::all(arma::vectorise(o.power) >= 0.0));
        CHECK(arma::all(arma::vectorise(o.delay) >= 0.0));
    }

    // Library inputs at the positions: LSPs at one frequency with the EoD offset and the absolute TOA, the shared
    // cluster set and the per-frequency set of frequency 0, all from the LSF seed 'seed'
    struct lib_t
    {
        arma::cube lsp, rnd_freq;
        arma::mat rnd, eod;
        arma::vec toa;
    };
    lib_t lib_inputs(const spmap &sp, const arma::mat &tx, const arma::mat &rx, arma::sword seed, double fc = 6e9)
    {
        lib_t r;
        arma::cube ph;
        arma::mat rnd_l;
        quadriga_lib::lsf_init_phases(sp, ph, seed);
        quadriga_lib::lsf_draw(sp, ph, tx, rx, rnd_l);
        quadriga_lib::lsp_generate(sp, arma::vec{fc}, tx, rx, rnd_l, r.lsp, nullptr, &r.toa, nullptr, nullptr, &r.eod);
        quadriga_lib::ssf_init_phases(sp, ph, seed, -1);
        quadriga_lib::ssf_draw(sp, ph, tx, rx, r.rnd, -1);
        quadriga_lib::ssf_init_phases(sp, ph, seed, 0);
        r.rnd_freq.set_size(rx.n_rows, (arma::uword)sp.at("NumClusters") - 1, 1);
        quadriga_lib::ssf_draw(sp, ph, tx, rx, r.rnd_freq.slice(0), 0);
        return r;
    }

    // UTs of a drop around the origin: d2D 35 to 300 m, height 1.5 m
    arma::mat drop_rx(unsigned seed, arma::uword n_ut)
    {
        std::mt19937 g(seed);
        arma::mat rx(n_ut, 3);
        for (arma::uword i = 0; i < n_ut; ++i)
        {
            const double d = 35.0 + 265.0 * ((double)g() + 0.5) / 4294967296.0;
            const double a = 2.0 * pi * ((double)g() + 0.5) / 4294967296.0;
            rx(i, 0) = d * std::cos(a), rx(i, 1) = d * std::sin(a), rx(i, 2) = 1.5;
        }
        return rx;
    }

    // Output/input spread ratios of each link appended to ratio[0 ... 4] (DS, ASD, ASA, ESD, ESA; frequency 0), with
    // calc_delay_spread and calc_angular_spreads_sphere
    void spread_ratios(const out_t &o, const arma::cube &lsp, std::vector<double> *ratio)
    {
        const arma::uword n = o.delay.n_rows;
        std::vector<arma::vec> d(n), p(n), az_d(n), el_d(n), az_a(n), el_a(n);
        for (arma::uword i = 0; i < n; ++i)
        {
            d[i] = rv(o.delay, i), p[i] = rv(o.power.slices(0, 0), i), az_d[i] = rv(o.aod, i), el_d[i] = rv(o.eod, i);
            az_a[i] = rv(o.aoa, i), el_a[i] = rv(o.eoa, i);
        }
        arma::vec ds = quadriga_lib::calc_delay_spread(d, p), as[4];
        quadriga_lib::calc_angular_spreads_sphere(az_d, el_d, p, &as[0], &as[2]);
        quadriga_lib::calc_angular_spreads_sphere(az_a, el_a, p, &as[1], &as[3]);
        for (arma::uword i = 0; i < n; ++i)
        {
            ratio[0].push_back(ds(i) / lsp(i, 0, 0));
            for (int a = 0; a < 4; ++a)
                ratio[1 + a].push_back(as[a](i) * 180.0 / pi / lsp(i, 0, 3 + a));
        }
    }

    // Events between neighbouring positions of a track (frequency 0) for the clusters within 20 dB of the strongest:
    // ev[0] angle step > 10 deg, ev[1] delay step > DS/4, ev[2] power step > 3 dB; ev[3] counts the cluster samples
    void track_events(const out_t &o, const arma::cube &lsp, int *ev)
    {
        const arma::uword n_slot = o.delay.n_cols;
        for (arma::uword i = 1; i < o.delay.n_rows; ++i)
        {
            double pmax = 0.0;
            for (arma::uword s = 1; s < n_slot; ++s)
                pmax = std::max(pmax, std::max(o.power(i, s, 0), o.power(i - 1, s, 0)));
            for (arma::uword s = 1; s < n_slot; ++s)
            {
                if (std::max(o.power(i, s, 0), o.power(i - 1, s, 0)) < 0.01 * pmax)
                    continue;
                const double da = 180.0 / pi * std::max({std::abs(wrap_rad(o.aod(i, s, 0) - o.aod(i - 1, s, 0))),
                                                         std::abs(wrap_rad(o.aoa(i, s, 0) - o.aoa(i - 1, s, 0))),
                                                         std::abs(o.eod(i, s, 0) - o.eod(i - 1, s, 0)),
                                                         std::abs(o.eoa(i, s, 0) - o.eoa(i - 1, s, 0))});
                ev[0] += da > 10.0;
                ev[1] += std::abs(o.delay(i, s, 0) - o.delay(i - 1, s, 0)) > 0.25 * lsp(i, 0, 0);
                const double r = o.power(i, s, 0) / o.power(i - 1, s, 0);
                ev[2] += !(r < 2.0 && r > 0.5); // a cluster switched on or off counts here (ratio 0 or infinite)
                ++ev[3];
            }
        }
    }

    // Swapped link ends with mirrored LSPs (also the anchor) and mirrored blocks of the shared set
    fix_t swapped(const fix_t &f)
    {
        const arma::uword N = f.rnd.n_cols / 9, mb[9] = {0, 2, 1, 4, 3, 6, 5, 8, 7};
        fix_t s = f;
        s.tx = f.rx, s.rx = f.tx;
        for (arma::cube *c : {&s.lsp, &s.anchor})
        {
            const arma::cube &o = (c == &s.lsp) ? f.lsp : f.anchor;
            c->slice(3) = o.slice(4), c->slice(4) = o.slice(3), c->slice(5) = o.slice(6), c->slice(6) = o.slice(5);
        }
        for (arma::uword b = 0; b < 9 && N > 0; ++b)
            s.rnd.cols(b * N, b * N + N - 1) = f.rnd.cols(mb[b] * N, mb[b] * N + N - 1);
        return s;
    }
}

#endif
