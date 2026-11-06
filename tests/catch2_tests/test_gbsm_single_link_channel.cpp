// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of gbsm_single_link_channel: channel coefficients per snapshot for the mobilities "phase" and "drift", the
// outputs "taps" and "rays", the frequency index, the validity report in 'par', the snapshot count, the QRT route,
// thread independence and validation.
// Oracles in this file: per snapshot the paths of ssf_paths at the snapshot positions, restricted to the rays with
// power > 0 at i_freq plus LOS and GR, passed to get_channels_planar (arrays at the ray-data positions) or
// get_channels_spherical (arrays at the snapshot positions); the tap sums with per-element delays weighted by the ray
// powers; b_med (lower weighted median) and b_eff = (sum P / sum (P / b²))^(1/2) from the scatterers; the limits
// sqrt(lambda·b_eff / 8) and 0.2·b_med and the displacement ratios. Fixture: make_rays of qd_test_rays.hpp (Tx (3,
// -4, 25), Rx (120, 45, 1.5)). Supporting tasks: ssf_paths, get_channels_planar, get_channels_spherical,
// generate_arrayant_*, gbsm_rays::solve_scatterers, gbsm_rays_from_paths, qrt_paths_from_scatterers, qrt_file_*.
// Units: [m], [s], [rad], [Hz].
// Tolerances: coefficients 1e-12 relative to the largest coefficient of the snapshot (sums over rays can cancel);
// delays 1e-12 relative to the largest delay; displacements up to 1 m, so that different evaluation orders of the
// displacement phase agree within 1e-12; QRT route: float32 storage, per path the length bound
// dL = 1e-7·(sum of the absolute coordinates of Tx, FBS, LBS, Rx), coefficients (k·dL + 1e-6)·abs(c).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_lib.hpp"
#include "quadriga_gbsm.hpp"
#include "qd_test_diag.hpp"
#include "qd_test_rays.hpp"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Catch::Matchers::Message;

QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

namespace
{
    using rays_t = quadriga_lib::gbsm_rays;
    using arr_t = quadriga_lib::arrayant<double>;
    using chan_t = quadriga_lib::channel<double>;
    const double nan_v = std::numeric_limits<double>::quiet_NaN();

    // Dual-polarised (+-45 deg) 2x2 panel, 8 ports, 5 deg pattern grid
    arr_t panel(double f)
    {
        return quadriga_lib::generate_arrayant_3GPP<double>(2, 2, f, 3, 0.0, 0.5, 1, 1, 0.5, 0.5, nullptr, 5.0);
    }

    arma::vec mirror(const arma::vec &x) { return arma::vec{x(0), x(1), -x(2)}; }

    // Arguments of one call; an empty matrix or vector is passed as nullptr
    struct args_t
    {
        const arr_t *tx, *rx;
        const rays_t *r;
        arma::vec lg = {0.0}, kf;
        std::string mob = "auto", out = "rays";
        arma::uword i_freq = 0;
        arma::mat T, R, To, Ro;
        bool o2 = false, abs = false;

        chan_t run() const
        {
            auto p = [](const arma::mat &m) { return m.empty() ? nullptr : &m; };
            return quadriga_lib::gbsm_single_link_channel(*tx, *rx, *r, lg, mob, out, i_freq, p(T), p(R), p(To),
                                                          p(Ro), kf.empty() ? nullptr : &kf, o2, abs);
        }
        arma::uword n_snap() const
        {
            return std::max({(arma::uword)1, T.n_rows, R.n_rows, To.n_rows, Ro.n_rows});
        }
        bool drift() const { return mob == "drift" || (mob == "auto" && r->method == "quadriga"); }
        arma::vec row(const arma::mat &m, arma::uword t, const arma::vec &d) const
        {
            return m.empty() ? d : arma::vec(m.row(m.n_rows == 1 ? 0 : t).t());
        }
    };

    // Expected rays of one snapshot: ssf_paths at the snapshot positions, kept rays at i_freq, get_channels_*
    struct snap_t
    {
        arma::cube cr, ci, dl;
        arma::vec gain, len;
        arma::mat M, ang, fbs, lbs;
        arma::uvec ind;
    };
    std::vector<snap_t> ref(const args_t &a)
    {
        const rays_t &r = *a.r;
        const arma::uword n_freq = r.n_freq(), f = a.i_freq;
        const arma::vec z3(3, arma::fill::zeros);
        std::vector<snap_t> out;
        for (arma::uword t = 0; t < a.n_snap(); ++t)
        {
            const arma::vec T = a.row(a.T, t, r.tx_pos), R = a.row(a.R, t, r.rx_pos);
            const arma::vec To = a.row(a.To, t, z3), Ro = a.row(a.Ro, t, z3);
            arma::vec lgv(n_freq, arma::fill::zeros), kfv(n_freq, arma::fill::ones);
            lgv(f) = a.lg(a.lg.n_elem == 1 ? 0 : t);
            if (!a.kf.empty())
                kfv(f) = a.kf(a.kf.n_elem == 1 ? 0 : t);
            arma::mat pg, fbs, lbs;
            arma::vec pl, aod, eod, aoa, eoa;
            arma::cube M;
            arma::uvec type, ind;
            quadriga_lib::ssf_paths(r, lgv, pg, pl, M, a.drift() ? "drift" : "phase", &T, &R,
                                    a.kf.empty() ? nullptr : &kfv, &aod, &eod, &aoa, &eoa, &fbs, &lbs, &type, &ind,
                                    a.o2);
            std::vector<arma::uword> v;
            for (arma::uword p = 0; p < ind.n_elem; ++p)
                if (type(p) != 2 || r.power(ind(p), f) > 0.0)
                    v.push_back(p);
            const arma::uvec sel(v);
            snap_t s;
            s.ind = ind.elem(sel), s.gain = pg.col(f), s.gain = s.gain.elem(sel), s.len = pl.elem(sel);
            s.M = M.slice(f).cols(sel);
            s.ang = arma::join_rows(arma::join_rows(aod.elem(sel), eod.elem(sel)),
                                    arma::join_rows(aoa.elem(sel), eoa.elem(sel)));
            const double fc = r.center_frequency(f);
            if (a.drift())
            {
                s.fbs = fbs.cols(sel), s.lbs = lbs.cols(sel);
                quadriga_lib::get_channels_spherical<double>(a.tx, a.rx, T(0), T(1), T(2), To(0), To(1), To(2), R(0),
                                                             R(1), R(2), Ro(0), Ro(1), Ro(2), &s.fbs, &s.lbs, &s.gain,
                                                             &s.len, &s.M, &s.cr, &s.ci, &s.dl, fc, a.abs);
            }
            else
            {
                const arma::vec &T0 = r.tx_pos, &R0 = r.rx_pos;
                const arma::vec a0 = s.ang.col(0), a1 = s.ang.col(1), a2 = s.ang.col(2), a3 = s.ang.col(3);
                quadriga_lib::get_channels_planar<double>(a.tx, a.rx, T0(0), T0(1), T0(2), To(0), To(1), To(2),
                                                          R0(0), R0(1), R0(2), Ro(0), Ro(1), Ro(2), &a0, &a1, &a2,
                                                          &a3, &s.gain, &s.len, &s.M, &s.cr, &s.ci, &s.dl, fc, a.abs);
            }
            out.push_back(s);
        }
        return out;
    }

    // Expected taps of one snapshot: sums per tap index, delays weighted by w (default: rays.power at i_freq; equal
    // weights in a tap whose weight sum is 0); taps ascending
    struct tap_t
    {
        arma::cube cr, ci, dl;
        arma::uvec tap, n;
    };
    tap_t ref_taps(const rays_t &r, arma::uword f, const snap_t &s, const arma::vec *w_in = nullptr)
    {
        const arma::uvec tp = r.tap.elem(s.ind);
        tap_t o;
        o.tap = arma::unique(tp), o.n.zeros(o.tap.n_elem);
        o.cr.zeros(s.cr.n_rows, s.cr.n_cols, o.tap.n_elem), o.ci = o.cr, o.dl = o.cr;
        for (arma::uword q = 0; q < o.tap.n_elem; ++q)
        {
            const arma::uvec in = arma::find(tp == o.tap(q));
            arma::vec w(in.n_elem);
            for (arma::uword j = 0; j < in.n_elem; ++j)
                w(j) = (w_in != nullptr) ? (*w_in)(in(j)) : r.power(s.ind(in(j)), f);
            if (arma::accu(w) == 0.0)
                w.ones();
            for (arma::uword j = 0; j < in.n_elem; ++j)
                o.cr.slice(q) += s.cr.slice(in(j)), o.ci.slice(q) += s.ci.slice(in(j)),
                    o.dl.slice(q) += w(j) * s.dl.slice(in(j));
            o.dl.slice(q) /= arma::accu(w), o.n(q) = in.n_elem;
        }
        return o;
    }

    // Largest abs(a - b) of complex cubes relative to the largest abs(b)
    double cdiff(const arma::cube &ar, const arma::cube &ai, const arma::cube &br, const arma::cube &bi)
    {
        if (arma::size(ar) != arma::size(br) || arma::size(ai) != arma::size(bi))
            return 1e300;
        if (br.empty())
            return 0.0;
        const arma::cube d = arma::sqrt(arma::square(ar - br) + arma::square(ai - bi));
        const double m = arma::sqrt(arma::square(br) + arma::square(bi)).max();
        return (m > 0.0) ? d.max() / m : d.max();
    }

    // Largest abs(a - b) relative to the largest abs(b), for delays, gains and lengths
    template <typename T>
    double rdiff(const T &a, const T &b)
    {
        if (arma::size(a) != arma::size(b))
            return 1e300;
        if (b.empty())
            return 0.0;
        const double m = arma::abs(b).max(), d = arma::abs(a - b).max();
        return (m > 0.0) ? d / m : d;
    }

    // par entry by name; throws if absent
    template <typename T>
    T par(const chan_t &c, const std::string &name)
    {
        for (size_t i = 0; i < c.par_names.size(); ++i)
            if (c.par_names[i] == name)
                return std::any_cast<T>(c.par_data[i]);
        throw std::runtime_error("par entry '" + name + "' not found");
    }
    bool has_par(const chan_t &c, const std::string &name)
    {
        return std::find(c.par_names.begin(), c.par_names.end(), name) != c.par_names.end();
    }

    // Channel of rays output against the oracle: coefficients, delays, path data, tap labels
    void check_rays(const chan_t &c, const args_t &a, const std::vector<snap_t> &e)
    {
        REQUIRE(c.n_snap() == e.size());
        QD_DIAG("\n# diag: rays output vs oracle, %s\nsnapshot,coeff_err,delay_err\n", a.drift() ? "drift" : "phase");
        for (arma::uword t = 0; t < e.size(); ++t)
        {
            INFO("snapshot " << t);
            const double dc = cdiff(c.coeff_re[t], c.coeff_im[t], e[t].cr, e[t].ci), dd = rdiff(c.delay[t], e[t].dl);
            QD_DIAG("%d,%.2e,%.2e\n", (int)t, dc, dd);
            CHECK(dc <= 1e-12);
            CHECK(dd <= 1e-12);
            CHECK(rdiff(c.path_gain[t], e[t].gain) <= 1e-12);
            CHECK(rdiff(c.path_length[t], e[t].len) <= 1e-12);
            CHECK(arma::approx_equal(c.path_polarization[t], e[t].M, "absdiff", 1e-12));
            CHECK(arma::approx_equal(c.path_angles[t], e[t].ang, "absdiff", 1e-12));
            if (a.drift())
                CHECK((arma::approx_equal(c.path_fbs_pos[t], e[t].fbs, "absdiff", 1e-12) &&
                       arma::approx_equal(c.path_lbs_pos[t], e[t].lbs, "absdiff", 1e-12)));
            else
                CHECK((c.path_fbs_pos.empty() && c.path_lbs_pos.empty()));
        }
        if (!e.empty())
            CHECK(arma::all(par<arma::uvec>(c, "tap") == a.r->tap.elem(e[0].ind)));
    }

    // Channel of taps output against the tap oracle
    void check_taps(const chan_t &c, const args_t &a, const std::vector<snap_t> &e)
    {
        REQUIRE(c.n_snap() == e.size());
        QD_DIAG("\n# diag: taps output vs oracle, %s\nsnapshot,coeff_err,delay_err\n", a.drift() ? "drift" : "phase");
        for (arma::uword t = 0; t < e.size(); ++t)
        {
            INFO("snapshot " << t);
            const tap_t x = ref_taps(*a.r, a.i_freq, e[t]);
            const double dc = cdiff(c.coeff_re[t], c.coeff_im[t], x.cr, x.ci), dd = rdiff(c.delay[t], x.dl);
            QD_DIAG("%d,%.2e,%.2e\n", (int)t, dc, dd);
            CHECK(dc <= 1e-12);
            CHECK(dd <= 1e-12);
            CHECK(arma::all(par<arma::uvec>(c, "tap") == x.tap));
            CHECK(arma::all(par<arma::uvec>(c, "n_ray_tap") == x.n));
            CHECK(c.path_gain.empty());
        }
    }

    // b_med (lower weighted median over the NLOS rays with P > 0) and b_eff (over all rays with P > 0) per end, at
    // the ray-data positions; NaN where undefined
    void ref_b(const rays_t &r, arma::uword f, double bm[2], double be[2])
    {
        const arma::vec &T = r.tx_pos, &R = r.rx_pos;
        const double d3D = len3(R - T), d_gr = len3(R - mirror(T));
        for (int e = 0; e < 2; ++e)
        {
            std::vector<std::pair<double, double>> nl; // (b, P), ray order
            double sP = 0.0, sQ = 0.0, sN = 0.0;
            for (arma::uword i = 0; i < r.n_ray(); ++i)
            {
                const double P = r.power(i, f);
                if (!(P > 0.0))
                    continue;
                const arma::uword y = r.path_type(i);
                const double b = (y == 0)   ? d3D
                                 : (y == 1) ? d_gr
                                            : len3(e == 0 ? arma::vec(r.fbs_pos.col(i) - T)
                                                          : arma::vec(r.lbs_pos.col(i) - R));
                sP += P, sQ += P / (b * b);
                if (y == 2)
                    nl.push_back({b, P}), sN += P;
            }
            be[e] = (sP > 0.0) ? std::sqrt(sP / sQ) : nan_v;
            std::stable_sort(nl.begin(), nl.end(), [](const auto &x, const auto &y) { return x.first < y.first; });
            bm[e] = nan_v;
            double acc = 0.0;
            for (const auto &x : nl)
                if ((acc += x.second) >= 0.5 * sN)
                {
                    bm[e] = x.first;
                    break;
                }
        }
    }

    // NaN-aware equality within a relative tolerance
    bool same(double a, double b, double tol = 1e-12)
    {
        return (std::isnan(a) && std::isnan(b)) || a == b || std::abs(a - b) <= tol * std::abs(b);
    }
}

TEST_CASE("gbsm_single_link_channel - rays output equals get_channels with the paths of ssf_paths")
{
    rays_t r = make_rays(51, true, true, 6, 1, "38.901", true);
    const arr_t tx = panel(3.5e9), rx = panel(3.5e9);
    args_t a{&tx, &rx, &r};
    a.T = arma::mat{{3.0, -4.0, 25.0}, {3.2, -4.1, 25.0}, {3.4, -4.3, 25.1}, {3.5, -4.6, 25.1}};
    a.R = arma::mat{{120.0, 45.0, 1.5}, {120.3, 45.2, 1.5}, {120.6, 45.3, 1.6}, {120.9, 45.5, 1.4}};
    a.To = arma::mat{{0.1, -0.2, 0.7}};
    a.Ro = arma::mat{{0.0, 0.3, -2.0}};
    a.lg = {-87.5};

    SECTION("static link gain") {}
    SECTION("oxygen at 60 GHz")
    {
        r.center_frequency(0) = 60e9, a.o2 = true;
    }
    SECTION("link gain and K ratio per snapshot")
    {
        a.lg = {-80.0, -80.0, -82.5, -79.0}, a.kf = {1.0, 1.0, 2.5, 0.0};
    }
    for (const char *mob : {"phase", "drift"})
    {
        INFO("mobility " << mob);
        a.mob = mob, a.out = "rays";
        const std::vector<snap_t> e = ref(a);
        REQUIRE(e[0].ind.n_elem == 7); // the zero-power ray is dropped
        check_rays(a.run(), a, e);
        a.out = "taps";
        check_taps(a.run(), a, e);
    }
}

TEST_CASE("gbsm_single_link_channel - taps sum the rays per tap with power-weighted per-element delays")
{
    rays_t r = make_rays(52, true, true, 6, 1, "38.901");
    const arr_t tx = panel(3.5e9), rx = panel(3.5e9);
    args_t a{&tx, &rx, &r};
    a.out = "taps", a.R = arma::mat{{120.4, 45.1, 1.5}};
    const arma::uword i0 = 2, i1 = 3, q6 = 2; // the rays of tap 6 (powers 1 : 4) and its output index
    REQUIRE((r.tap(i0) == 6 && r.tap(i1) == 6 && std::abs(r.power(i1, 0) - 4.0 * r.power(i0, 0)) < 1e-15));

    SECTION("power weights 1 : 4")
    {
        const std::vector<snap_t> e = ref(a);
        const chan_t c = a.run();
        check_taps(c, a, e);
        const tap_t x = ref_taps(r, 0, e[0]);
        const arma::vec eq(e[0].ind.n_elem, arma::fill::ones);
        const tap_t u = ref_taps(r, 0, e[0], &eq);
        CHECK(arma::abs(u.dl.slice(q6) - x.dl.slice(q6)).min() > 100.0 * 1e-12 * arma::abs(x.dl).max());
    }
    SECTION("LOS and GR taps of an NLOS scenario have their ray's delay")
    {
        r.power(0, 0) = 0.0, r.power(1, 0) = 0.0;
        const std::vector<snap_t> e = ref(a);
        const chan_t c = a.run();
        check_taps(c, a, e);
        for (arma::uword q = 0; q < 2; ++q)
        {
            CHECK(c.delay[0].slice(q).is_finite());
            CHECK(rdiff(arma::mat(c.delay[0].slice(q)), arma::mat(e[0].dl.slice(q))) <= 1e-12);
            CHECK(arma::abs(c.coeff_re[0].slice(q)).max() == 0.0);
        }
    }
    SECTION("oxygen: the weights are the ray powers, not the path gains")
    {
        r.center_frequency(0) = 60e9, r.delay(i0) = 50e-9, r.delay(i1) = 250e-9, a.o2 = true; // 60 m apart
        const std::vector<snap_t> e = ref(a);
        check_taps(a.run(), a, e);
        const tap_t x = ref_taps(r, 0, e[0]), g = ref_taps(r, 0, e[0], &e[0].gain);
        CHECK(arma::abs(g.dl.slice(q6) - x.dl.slice(q6)).min() > 100.0 * 1e-12 * arma::abs(x.dl).max());
    }
}

TEST_CASE("gbsm_single_link_channel - taps of data from paths use the default taps")
{
    rays_t s = make_rays(53, true, true, 6, 1, "quadriga", true);
    arma::mat pg, fbs, lbs;
    arma::vec pl;
    arma::cube M;
    arma::uvec type;
    quadriga_lib::ssf_paths(s, arma::vec{-90.0}, pg, pl, M, "drift", nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, &fbs, &lbs, &type);
    rays_t d;
    quadriga_lib::gbsm_rays_from_paths(s.center_frequency, s.tx_pos, s.rx_pos, fbs, lbs, pg, pl, M, d, "quadriga",
                                       &type, nullptr, &s.gr_eps_re, &s.gr_eps_im);
    const arr_t tx = panel(3.5e9), rx = panel(3.5e9);
    args_t a{&tx, &rx, &d};
    a.R = arma::mat{{120.2, 45.1, 1.5}, {120.4, 45.2, 1.5}};
    a.out = "taps";
    const chan_t c = a.run();
    REQUIRE(c.n_snap() == 2);
    const arma::uword n = pg.n_rows;
    CHECK(arma::all(par<arma::uvec>(c, "tap") == arma::regspace<arma::uvec>(0, 3, 3 * (n - 1))));
    CHECK(arma::all(par<arma::uvec>(c, "n_ray_tap") == 1));
    a.out = "rays";
    const chan_t cr = a.run();
    REQUIRE(cr.n_snap() == 2);
    for (arma::uword t = 0; t < 2; ++t)
        CHECK(cdiff(c.coeff_re[t], c.coeff_im[t], cr.coeff_re[t], cr.coeff_im[t]) == 0.0);
    check_rays(cr, a, ref(a));
}

TEST_CASE("gbsm_single_link_channel - the phase speed-up equals per-snapshot planar calls")
{
    const rays_t r = make_rays(54, true, true, 8, 1, "38.901");
    const arr_t tx = panel(3.5e9), rx = panel(3.5e9);
    args_t a{&tx, &rx, &r};
    a.mob = "phase";
    a.T = arma::mat{{3.0, -4.0, 25.0}, {3.1, -4.2, 25.0}, {3.3, -4.4, 25.2}, {3.5, -4.5, 25.2}, {3.6, -4.7, 25.3}};
    a.R = arma::mat{{120.1, 45.0, 1.5}, {120.3, 45.4, 1.5}, {120.5, 45.6, 1.7}, {120.7, 45.7, 1.5}, {120.9, 45.9, 1.5}};
    a.To = arma::mat{{0.0, 0.1, 0.5}, {0.0, 0.1, 0.5}, {0.0, 0.1, 0.9}, {0.0, 0.1, 0.9}, {0.2, 0.1, 0.9}};
    // Orientation changes: snapshot 1 Rx only, 2 Tx only, 4 both
    a.Ro = arma::mat{{0.0, 0.0, 1.0}, {0.0, 0.0, 1.4}, {0.0, 0.0, 1.4}, {0.0, 0.0, 1.4}, {0.0, 0.0, -2.5}};
    a.lg = {-70.0, -71.0, -71.0, -72.0, -72.0};
    const std::vector<snap_t> e = ref(a);
    check_rays(a.run(), a, e);
    a.out = "taps";
    check_taps(a.run(), a, e);
}

TEST_CASE("gbsm_single_link_channel - i_freq selects powers, M, kept rays, weights and frequency")
{
    rays_t r = make_rays(55, true, true, 6, 2, "quadriga", true);
    r.power(5, 1) = 0.0; // NLOS ray with power at frequency 0 only: dropped at i_freq = 1
    r.power(6, 0) = 0.0; // NLOS ray with power at frequency 1 only: kept
    r.power(3, 1) = 3.0 * r.power(2, 1);
    const arr_t tx = panel(8e9), rx = panel(8e9);
    args_t a{&tx, &rx, &r};
    a.i_freq = 1, a.lg = {-75.0}, a.kf = {0.7};
    a.R = arma::mat{{120.2, 45.3, 1.5}, {120.5, 45.4, 1.6}};
    for (const char *mob : {"auto", "phase"})
    {
        INFO("mobility " << mob);
        a.mob = mob, a.out = "rays";
        const std::vector<snap_t> e = ref(a);
        CHECK(arma::all(e[0].ind == arma::uvec{0, 1, 2, 3, 6, 7}));
        const chan_t c = a.run();
        check_rays(c, a, e);
        CHECK(c.center_frequency(0) == r.center_frequency(1));
        a.out = "taps";
        check_taps(a.run(), a, e);
    }
}

TEST_CASE("gbsm_single_link_channel - use_absolute_delays in phase and drift")
{
    const rays_t r = make_rays(56, true, true, 6, 1, "38.901", true);
    const arr_t tx = panel(3.5e9), rx = panel(3.5e9);
    args_t a{&tx, &rx, &r};
    a.R = arma::mat{{120.5, 45.2, 1.5}, {121.0, 45.4, 1.5}};
    for (const char *mob : {"phase", "drift"})
    {
        INFO("mobility " << mob);
        a.mob = mob, a.abs = false;
        const chan_t ca = a.run();
        a.abs = true;
        const chan_t cb = a.run();
        check_rays(cb, a, ref(a));
        if (a.mob == "phase") // LOS delay of the ray-data positions added
            for (arma::uword t = 0; t < 2; ++t)
                CHECK(rdiff(cb.delay[t], arma::cube(ca.delay[t] + len3(r.rx_pos - r.tx_pos) / c0)) <= 1e-12);
    }
}

TEST_CASE("gbsm_single_link_channel - n_ray = 0 and a single kept ray")
{
    const arr_t tx = panel(3.5e9), rx = quadriga_lib::generate_arrayant_xpol<double>(5.0);
    const rays_t e0 = make_rays(57, false, false, 0, 1, "38.901");
    rays_t one = make_rays(57, false, false, 3, 1, "38.901", true);
    one.power(1, 0) = 0.0; // rays 1 and 2 have power 0: only ray 0 is kept
    for (const char *mob : {"phase", "drift"})
        for (const char *out : {"taps", "rays"})
        {
            INFO(mob << ", " << out);
            args_t a{&tx, &rx, &e0};
            a.mob = mob, a.out = out, a.R = arma::mat{{120.0, 45.0, 1.5}, {120.1, 45.0, 1.5}};
            const chan_t c = a.run();
            REQUIRE(c.n_snap() == 2);
            CHECK(arma::size(c.coeff_re[1]) == arma::size(2, 8, 0));
            CHECK(arma::size(c.delay[1]) == arma::size(2, 8, 0));
            CHECK(par<arma::uvec>(c, "tap").n_elem == 0);
            CHECK(c.is_valid().empty());

            a.r = &one;
            const chan_t d = a.run();
            const std::vector<snap_t> e = ref(a);
            REQUIRE(e[0].ind.n_elem == 1);
            CHECK(d.coeff_re[1].n_slices == 1);
            CHECK(cdiff(d.coeff_re[1], d.coeff_im[1], e[1].cr, e[1].ci) <= 1e-12);
            CHECK(rdiff(d.delay[1], e[1].dl) <= 1e-12);
        }
}

TEST_CASE("gbsm_single_link_channel - LOS and GR taps are always present and taps are ascending")
{
    const auto omni = quadriga_lib::generate_arrayant_omni<double>();
    rays_t r = make_rays(58, true, true, 6, 1, "38.901");
    args_t a{&omni, &omni, &r};
    a.out = "taps";
    SECTION("NLOS scenario")
    {
        r.power(0, 0) = 0.0, r.power(1, 0) = 0.0;
        r.tap(2) = r.tap(3) = 20, r.tap(5) = 9, r.tap(6) = 15, r.tap(7) = 11; // labels out of order
        const chan_t c = a.run();
        REQUIRE(c.n_snap() == 1);
        CHECK(arma::all(par<arma::uvec>(c, "tap") == arma::uvec{0, 3, 9, 11, 15, 20}));
        CHECK((c.coeff_re[0](0, 0, 0) == 0.0 && c.coeff_im[0](0, 0, 1) == 0.0));
        check_taps(c, a, ref(a));
    }
    SECTION("TR LOS: the LOS tap and a tap of zero excess delay")
    {
        REQUIRE(r.delay(5) == 0.0); // make_rays: zero-excess ray in its own tap
        const chan_t c = a.run();
        REQUIRE(c.n_snap() == 1);
        const arma::uvec tp = par<arma::uvec>(c, "tap");
        CHECK(arma::all(tp == arma::uvec{0, 3, 6, 12, 15, 18}));
        CHECK((std::abs(c.delay[0](0, 0, 0)) < 1e-15 && std::abs(c.delay[0](0, 0, 3)) < 1e-15));
        CHECK(arma::accu(arma::abs(arma::vectorise(c.delay[0])) < 1e-15) == 2);
    }
}

TEST_CASE("gbsm_single_link_channel - par holds the validity report")
{
    const auto omni = quadriga_lib::generate_arrayant_omni<double>();
    rays_t s = make_rays(59, true, true, 6, 1, "38.901", true); // given scatterers, Tx and Rx distances differ
    const double lam = c0 / s.center_frequency(0);
    double bm[2], be[2];
    ref_b(s, 0, bm, be);
    REQUIRE(std::abs(bm[0] - bm[1]) > 1.0);

    // Positions of the snapshots: none at the ray-data positions
    args_t a{&omni, &omni, &s};
    a.T = arma::mat{{3.3, -4.0, 25.0}, {3.2, -4.4, 25.0}};
    a.R = arma::mat{{120.5, 45.0, 1.5}, {120.1, 45.0, 1.5}, {119.0, 45.0, 1.5}};
    a.T = arma::join_cols(a.T, a.T.row(1));
    a.lg = {-80.0, -81.0, -82.0};
    const double d_tx = len3(arma::vec(a.T.row(1).t()) - s.tx_pos), d_rx = len3(arma::vec(a.R.row(2).t()) - s.rx_pos);
    REQUIRE(std::abs(d_rx - len3(arma::vec(a.R.row(2).t() - a.R.row(0).t()))) > 0.1); // relative to snapshot 0 differs

    auto check = [&](const chan_t &c, const double b_med[2], const double b_eff[2], bool drift)
    {
        const arma::vec pm = par<arma::vec>(c, "b_med"), pe = par<arma::vec>(c, "b_eff");
        const arma::vec lim = par<arma::vec>(c, "limit"), rat = par<arma::vec>(c, "ratio");
        const double d[2] = {d_tx, d_rx};
        QD_DIAG("\n# diag: validity report, %s\nend,b_med,b_eff,limit,ratio\n", drift ? "drift" : "phase");
        for (int e = 0; e < 2; ++e)
        {
            INFO("end " << e);
            QD_DIAG("%s,%.4f,%.4f,%.4f,%.4f\n", e == 0 ? "Tx" : "Rx", pm(e), pe(e), lim(e), rat(e));
            CHECK(same(pm(e), b_med[e]));
            CHECK(same(pe(e), b_eff[e]));
            const double l = drift ? 0.2 * b_med[e] : std::sqrt(lam * b_eff[e] / 8.0);
            CHECK(same(lim(e), std::isnan(l) ? arma::datum::inf : l));
            CHECK(same(rat(e), std::isnan(l) ? 0.0 : d[e] / l));
        }
        CHECK(same(par<double>(c, "ratio_total"), std::sqrt(rat(0) * rat(0) + rat(1) * rat(1))));
        CHECK(arma::approx_equal(par<arma::vec>(c, "link_gain_dB"), a.lg, "absdiff", 0.0));
    };
    SECTION("phase and drift with given scatterers")
    {
        a.mob = "phase";
        check(a.run(), bm, be, false);
        a.mob = "drift";
        check(a.run(), bm, be, true);
    }
    SECTION("phase without scatterers: from a solved copy")
    {
        rays_t u = make_rays(59, true, true, 6, 1, "38.901"), v = u;
        v.solve_scatterers();
        ref_b(v, 0, bm, be);
        a.r = &u, a.mob = "phase";
        check(a.run(), bm, be, false);
    }
    SECTION("phase without scatterers, Tx or Rx below z = 0, with and without GR ray: b from a solved copy")
    {
        for (int e = 0; e < 2; ++e)
            for (bool gr : {true, false})
            {
                INFO("end " << e << " below z = 0, GR ray " << gr);
                rays_t u = make_rays(59, true, gr, 6, 1, "38.901");
                arma::vec &P0 = (e == 0) ? u.tx_pos : u.rx_pos;
                const double dz = -1.0 - P0(2);
                P0(2) = -1.0; // hand-made object: valid, but the GR ray has no specular point
                REQUIRE(u.is_valid().empty());

                // Oracle: the NLOS scatterers do not depend on the GR ray, whose scatterers b does not use (b of the
                // GR ray is its image-method length); the solver rejects the GR ray, so it is solved as an NLOS ray
                rays_t v = u;
                v.path_type.replace(1, 2);
                v.solve_scatterers();
                v.path_type = u.path_type;
                ref_b(v, 0, bm, be);
                REQUIRE((std::isfinite(bm[e]) && std::isfinite(be[e])));

                args_t b = a; // snapshot positions moved with the end, so that the displacements stay
                b.r = &u, b.mob = "phase";
                (e == 0 ? b.T : b.R).col(2) += dz;
                check(b.run(), bm, be, false);
            }
    }
    SECTION("LOS-only link, d3D = 100 m: limits 1.0347 m at 3.5 GHz and 0.3658 m at 28 GHz")
    {
        rays_t l = make_rays(59, true, false, 0, 2, "quadriga", true);
        l.rx_pos = l.tx_pos + arma::vec{60.0, 80.0, 0.0};
        l.fbs_pos.col(0) = l.tx_pos + arma::vec{30.0, 40.0, 0.0}, l.lbs_pos.col(0) = l.fbs_pos.col(0);
        l.center_frequency(1) = 28e9;
        args_t b{&omni, &omni, &l};
        b.mob = "phase", b.R = arma::mat{{l.rx_pos(0) + 0.5, l.rx_pos(1), l.rx_pos(2)}};
        const double expect[2] = {1.0347, 0.3658};
        for (arma::uword f = 0; f < 2; ++f)
        {
            b.i_freq = f;
            const chan_t c = b.run();
            const arma::vec lim = par<arma::vec>(c, "limit"), rat = par<arma::vec>(c, "ratio");
            CHECK(std::isnan(par<arma::vec>(c, "b_med")(0)));
            CHECK((std::abs(lim(0) - expect[f]) < 1e-4 && std::abs(lim(1) - expect[f]) < 1e-4));
            CHECK((rat(0) == 0.0 && std::abs(rat(1) - 0.5 / lim(1)) < 1e-12));
        }
        b.mob = "drift";
        const chan_t c = b.run();
        CHECK(arma::all(par<arma::vec>(c, "limit") == arma::datum::inf));
        CHECK((par<arma::vec>(c, "ratio")(1) == 0.0 && par<double>(c, "ratio_total") == 0.0));
    }
}

TEST_CASE("gbsm_single_link_channel - par entries are present by mode and output")
{
    const auto omni = quadriga_lib::generate_arrayant_omni<double>();
    const rays_t g = make_rays(60, true, true, 6, 1, "38.901", true);
    const rays_t n = make_rays(60, true, false, 6, 1, "38.901", true);
    for (const rays_t *r : {&g, &n})
        for (const char *mob : {"phase", "drift"})
            for (const char *out : {"taps", "rays"})
            {
                const bool drift = std::string(mob) == "drift", taps = std::string(out) == "taps", gr = r == &g;
                INFO(mob << ", " << out << ", GR " << gr);
                args_t a{&omni, &omni, r};
                a.mob = mob, a.out = out, a.R = arma::mat{{120.0, 45.0, 1.5}, {120.2, 45.0, 1.5}, {120.4, 45.0, 1.5}};
                const chan_t c = a.run();
                const arma::uword n_kept = r->n_ray() - 1, n_tap = gr ? 6 : 5;
                std::vector<std::string> names = {"tap"};
                if (taps)
                    names.push_back("n_ray_tap");
                if (drift && taps)
                    names.push_back("fbs_pos"), names.push_back("lbs_pos");
                if (drift && taps && gr)
                    names.push_back("gr_pos");
                for (const char *x : {"link_gain_dB", "b_med", "b_eff", "limit", "ratio", "ratio_total"})
                    names.push_back(x);
                CHECK(c.par_names == names);
                CHECK(par<arma::uvec>(c, "tap").n_elem == (taps ? n_tap : n_kept));
                if (taps)
                    CHECK(par<arma::uvec>(c, "n_ray_tap").n_elem == n_tap);
                if (drift && taps)
                {
                    const arma::uvec kk = gr ? arma::uvec{0, 1, 2, 3, 5, 6, 7} : arma::uvec{0, 1, 2, 4, 5, 6};
                    const arma::mat F = r->fbs_pos.cols(kk), L = r->lbs_pos.cols(kk);
                    CHECK(arma::approx_equal(par<arma::mat>(c, "fbs_pos"), F, "absdiff", 0.0));
                    CHECK(arma::approx_equal(par<arma::mat>(c, "lbs_pos"), L, "absdiff", 0.0));
                }
                if (drift && taps && gr)
                {
                    const arma::mat G = par<arma::mat>(c, "gr_pos");
                    REQUIRE(arma::size(G) == arma::size(3, 3));
                    for (arma::uword t = 0; t < 3; ++t)
                        CHECK(arma::approx_equal(G.col(t), specular_point(r->tx_pos, a.R.row(t).t()), "absdiff",
                                                 1e-12));
                }
                CHECK(par<arma::vec>(c, "link_gain_dB").n_elem == 3);
                for (const char *x : {"b_med", "b_eff", "limit", "ratio"})
                    CHECK(par<arma::vec>(c, x).n_elem == 2);
            }
}

TEST_CASE("gbsm_single_link_channel - the snapshot count follows the positions and orientations")
{
    const auto omni = quadriga_lib::generate_arrayant_omni<double>();
    const rays_t r = make_rays(61, true, true, 6, 1, "38.901");
    const arma::mat one = {{120.0, 45.0, 1.5}}, four = arma::repmat(one, 4, 1), ori4(4, 3, arma::fill::zeros);
    const arma::mat ori3(3, 3, arma::fill::zeros), two = arma::repmat(one, 2, 1);
    const arma::vec lg1 = {-80.0}, lg2 = {-80.0, -81.0}, lg4 = {-80.0, -81.0, -82.0, -83.0}, kf2 = {1.0, 2.0};
    const arma::vec kf4 = {1.0, 2.0, 0.5, 0.0};
    auto call = [&](const arma::mat *T, const arma::mat *R, const arma::mat *To, const arma::mat *Ro,
                    const arma::vec &lg, const arma::vec *kf = nullptr)
    { return quadriga_lib::gbsm_single_link_channel(omni, omni, r, lg, "phase", "taps", 0, T, R, To, Ro, kf); };
    CHECK(call(nullptr, nullptr, nullptr, nullptr, lg1).n_snap() == 1);
    CHECK(call(&one, &four, nullptr, nullptr, lg1).n_snap() == 4);
    CHECK(call(nullptr, nullptr, nullptr, &ori4, lg4).n_snap() == 4);
    CHECK(call(&one, nullptr, &ori4, nullptr, lg1, &kf4).n_snap() == 4);
    const chan_t c = call(&one, &four, nullptr, nullptr, lg4);
    CHECK(arma::approx_equal(c.tx_pos, arma::repmat(one.t(), 1, 4), "absdiff", 0.0));
    CHECK_THROWS_MATCHES(call(&two, &four, nullptr, nullptr, lg1), std::invalid_argument,
                         Message("Number of snapshots in 'tx_pos' must be 1 or 4."));
    CHECK_THROWS_MATCHES(call(nullptr, &four, &ori3, nullptr, lg1), std::invalid_argument,
                         Message("Number of snapshots in 'tx_orientation' must be 1 or 4."));
    CHECK_THROWS_MATCHES(call(nullptr, nullptr, nullptr, nullptr, lg2), std::invalid_argument,
                         Message("Number of snapshots in 'link_gain_dB' must be 1 or 1."));
    CHECK_THROWS_MATCHES(call(nullptr, &four, nullptr, nullptr, lg1, &kf2), std::invalid_argument,
                         Message("Number of snapshots in 'kf_ratio' must be 1 or 4."));
    const arma::vec v3 = r.tx_pos; // a [3] vector binds to const arma::mat *
    CHECK_THROWS_MATCHES(call(&v3, nullptr, nullptr, nullptr, lg1), std::invalid_argument,
                         Message("Input 'tx_pos' must have 3 columns."));
}

TEST_CASE("gbsm_single_link_channel - the QRT route equals the direct route in drift")
{
    rays_t r = make_rays(62, true, true, 6, 1, "quadriga");
    r.solve_scatterers();
    const arr_t a2 = panel(3.5e9);
    args_t a{&a2, &a2, &r};
    a.mob = "drift", a.out = "rays", a.lg = {-3.0};
    a.R = arma::mat{{120.0, 45.0, 1.5}, {120.4, 44.7, 1.6}, {120.8, 44.4, 1.7}};
    const chan_t cd = a.run();
    REQUIRE(cd.n_snap() == 3);
    const std::vector<snap_t> e = ref(a);
    REQUIRE(e[0].ind.n_elem == 7);

    const std::string fn = "test_gbsm_single_link_channel_qrt_route.qrt";
    const arma::vec f_GHz = r.center_frequency * 1e-9;
    quadriga_lib::qrt_file_init<double>(fn, f_GHz, a.R, arma::mat(3, 3, arma::fill::zeros));
    std::vector<quadriga_lib::path> all;
    for (arma::uword t = 0; t < 3; ++t)
    {
        const arma::vec R = a.R.row(t).t();
        arma::mat pg = e[t].gain;
        arma::cube M(8, e[t].ind.n_elem, 1);
        M.slice(0) = e[t].M;
        const arma::uvec type = r.path_type.elem(e[t].ind);
        const auto p = quadriga_lib::qrt_paths_from_scatterers(r.center_frequency, r.tx_pos, R, e[t].fbs, e[t].lbs,
                                                               pg, M, type, t);
        all.insert(all.end(), p.begin(), p.end());
    }
    quadriga_lib::qrt_file_append<double>(fn, all, r.tx_pos);

    const double k = 2.0 * pi * r.center_frequency(0) / c0;
    for (arma::uword t = 0; t < 3; ++t)
    {
        INFO("snapshot " << t);
        arma::vec cf, T, R, pl;
        arma::mat fbs, lbs, pg;
        arma::cube M;
        arma::u32_vec no_int;
        quadriga_lib::qrt_file_read<double>(fn, t, 0, true, &cf, &T, nullptr, &R, nullptr, &fbs, &lbs, &pg, &pl, &M,
                                            nullptr, nullptr, nullptr, nullptr, nullptr, 1, &no_int);
        const arma::uvec type = arma::conv_to<arma::uvec>::from(no_int), tap = r.tap.elem(e[t].ind);
        rays_t q;
        quadriga_lib::gbsm_rays_from_paths(cf, T, R, fbs, lbs, pg, pl, M, q, "quadriga", &type, &tap, &r.gr_eps_re,
                                           &r.gr_eps_im);
        const chan_t cq = quadriga_lib::gbsm_single_link_channel(a2, a2, q, arma::vec{0.0}, "drift", "rays");
        REQUIRE(cq.coeff_re[0].n_slices == e[t].ind.n_elem);
        for (arma::uword p = 0; p < e[t].ind.n_elem; ++p)
        {
            INFO("path " << p);
            const arma::vec F = e[t].fbs.col(p), L = e[t].lbs.col(p), Tt = r.tx_pos, Rt = a.R.row(t).t();
            const double dL = 1e-7 * (arma::accu(arma::abs(Tt)) + arma::accu(arma::abs(F)) +
                                      arma::accu(arma::abs(L)) + arma::accu(arma::abs(Rt)));
            const arma::cx_mat cx_d(cd.coeff_re[t].slice(p), cd.coeff_im[t].slice(p));
            const arma::cx_mat cx_q(cq.coeff_re[0].slice(p), cq.coeff_im[0].slice(p));
            CHECK(arma::all(arma::vectorise(arma::abs(cx_q - cx_d) <= (k * dL + 1e-6) * arma::abs(cx_d))));
            CHECK(std::abs(cq.path_length[0](p) - cd.path_length[t](p)) <= dL);
            CHECK(arma::abs(cq.delay[0].slice(p) - cd.delay[t].slice(p)).max() <= dL / c0);
        }
    }
    std::remove(fn.c_str());
}

TEST_CASE("gbsm_single_link_channel - results do not depend on the thread count")
{
#ifdef _OPENMP
    const rays_t r = make_rays(63, true, true, 8, 1, "38.901", true);
    const arr_t tx = panel(3.5e9), rx = panel(3.5e9);
    args_t a{&tx, &rx, &r};
    const arma::uword n_snap = 200; // n_snap · 9 kept rays · 64 element pairs = 115200
    a.R.set_size(n_snap, 3), a.To.zeros(n_snap, 3);
    for (arma::uword t = 0; t < n_snap; ++t)
    {
        a.R.row(t) = arma::rowvec{120.0 + 0.01 * (double)t, 45.0, 1.5};
        a.To(t, 2) = (t % 7 == 0) ? 0.01 * (double)t : 0.0; // orientation changes at every 7th snapshot
    }
    const int n_thr = omp_get_max_threads();
    for (const char *mob : {"phase", "drift"})
        for (const char *out : {"taps", "rays"})
        {
            INFO(mob << ", " << out);
            a.mob = mob, a.out = out;
            omp_set_num_threads(1);
            const chan_t c1 = a.run();
            omp_set_num_threads(n_thr);
            const chan_t cn = a.run();
            REQUIRE((c1.n_snap() == n_snap && cn.n_snap() == n_snap));
            for (arma::uword t = 0; t < n_snap; ++t)
                CHECK((arma::approx_equal(c1.coeff_re[t], cn.coeff_re[t], "absdiff", 0.0) &&
                       arma::approx_equal(c1.coeff_im[t], cn.coeff_im[t], "absdiff", 0.0) &&
                       arma::approx_equal(c1.delay[t], cn.delay[t], "absdiff", 0.0)));
        }
#endif
}

TEST_CASE("gbsm_single_link_channel - the channel is valid and stores the positions as columns")
{
    const arr_t tx = panel(3.5e9), rx = quadriga_lib::generate_arrayant_xpol<double>(5.0);
    rays_t r = make_rays(64, true, true, 6, 1, "38.901", true);
    args_t a{&tx, &rx, &r};
    a.R = arma::mat{{120.0, 45.0, 1.5}, {120.1, 45.0, 1.5}}, a.Ro = arma::mat{{0.1, 0.2, 0.3}};
    for (const char *mob : {"phase", "drift"})
        for (const char *out : {"taps", "rays"})
        {
            INFO(mob << ", " << out);
            a.mob = mob, a.out = out;
            const chan_t c = a.run();
            CHECK(c.is_valid().empty());
            CHECK(arma::approx_equal(c.tx_pos, arma::repmat(r.tx_pos, 1, 2), "absdiff", 0.0));
            CHECK(arma::approx_equal(c.rx_pos, arma::mat(a.R.t()), "absdiff", 0.0));
            CHECK(arma::approx_equal(c.tx_orientation, arma::mat(3, 2, arma::fill::zeros), "absdiff", 0.0));
            CHECK(arma::approx_equal(c.rx_orientation, arma::repmat(a.Ro.t(), 1, 2), "absdiff", 0.0));
            CHECK((c.center_frequency.n_elem == 1 && c.center_frequency(0) == r.center_frequency(0)));
            CHECK((c.n_rx() == 2 && c.n_tx() == 8 && c.initial_position == 0 && c.name == chan_t().name));
        }
}

TEST_CASE("gbsm_single_link_channel - input validation")
{
    const auto omni = quadriga_lib::generate_arrayant_omni<double>();
    rays_t r = make_rays(65, true, true, 6, 1, "quadriga", true), u = make_rays(65, true, true, 6, 1, "quadriga");
    const arma::vec lg = {-80.0};
    auto call = [&](const arr_t &tx, const arr_t &rx, const rays_t &x, const arma::vec &g, const std::string &mob,
                    const std::string &out, arma::uword f, const arma::mat *T, const arma::mat *R,
                    const arma::mat *To, const arma::mat *Ro, const arma::vec *kf)
    { return quadriga_lib::gbsm_single_link_channel(tx, rx, x, g, mob, out, f, T, R, To, Ro, kf); };
    auto msg = [&](const std::string &m, const arr_t &tx, const arr_t &rx, const rays_t &x, const arma::vec &g,
                   const std::string &mob, const std::string &out, arma::uword f, const arma::mat *T = nullptr,
                   const arma::mat *R = nullptr, const arma::mat *To = nullptr, const arma::mat *Ro = nullptr,
                   const arma::vec *kf = nullptr)
    { CHECK_THROWS_MATCHES(call(tx, rx, x, g, mob, out, f, T, R, To, Ro, kf), std::invalid_argument, Message(m)); };
    CHECK_NOTHROW(call(omni, omni, r, lg, "auto", "taps", 0, nullptr, nullptr, nullptr, nullptr, nullptr));

    SECTION("arrays and rays")
    {
        arr_t bad = omni;
        bad.e_theta_im.reset();
        const std::string m = bad.is_valid();
        REQUIRE(!m.empty());
        msg("Transmit antenna: " + m, bad, omni, r, lg, "auto", "taps", 0);
        msg("Receive antenna: " + m, omni, bad, r, lg, "auto", "taps", 0);
        rays_t x = r;
        x.method = "x";
        msg("Input 'rays.method' must be '38.901', '38.901_MF' or 'quadriga'.", omni, omni, x, lg, "auto", "taps", 0);
    }
    SECTION("options")
    {
        msg("Input 'mobility' must be 'phase', 'drift' or 'auto'.", omni, omni, r, lg, "x", "taps", 0);
        msg("Input 'output' must be 'taps' or 'rays'.", omni, omni, r, lg, "auto", "x", 0);
        msg("Input 'i_freq' is out of range for 1 frequencies.", omni, omni, r, lg, "auto", "taps", 1);
    }
    SECTION("link gain, positions, orientations, K ratio")
    {
        msg("Input 'link_gain_dB' cannot be empty.", omni, omni, r, arma::vec(), "auto", "taps", 0);
        msg("Input 'link_gain_dB' must be finite.", omni, omni, r, arma::vec{nan_v}, "auto", "taps", 0);
        const arma::mat m2(1, 2, arma::fill::zeros), mn = {{0.0, nan_v, 0.0}};
        const char *names[4] = {"tx_pos", "rx_pos", "tx_orientation", "rx_orientation"};
        for (int i = 0; i < 4; ++i)
        {
            const arma::mat *p[4] = {nullptr, nullptr, nullptr, nullptr};
            p[i] = &m2;
            msg("Input '" + std::string(names[i]) + "' must have 3 columns.", omni, omni, r, lg, "auto", "taps", 0,
                p[0], p[1], p[2], p[3]);
            p[i] = &mn;
            msg("Input '" + std::string(names[i]) + "' must be finite.", omni, omni, r, lg, "auto", "taps", 0, p[0],
                p[1], p[2], p[3]);
        }
        const arma::vec kn = {-0.1}, ki = {arma::datum::inf};
        for (const arma::vec *k : {&kn, &ki})
            msg("Input 'kf_ratio' must be finite and >= 0.", omni, omni, r, lg, "auto", "taps", 0, nullptr, nullptr,
                nullptr, nullptr, k);
        const arma::mat R2 = {{120.0, 45.0, 1.5}, {120.1, 45.0, 1.5}}, R3 = arma::join_cols(R2, R2.row(0));
        msg("Number of snapshots in 'rx_pos' must be 1 or 3.", omni, omni, r, lg, "auto", "taps", 0, &R3, &R2);
        msg("Number of snapshots in 'rx_orientation' must be 1 or 3.", omni, omni, r, lg, "auto", "taps", 0, &R3,
            nullptr, nullptr, &R2);
    }
    SECTION("drift")
    {
        msg("Mobility 'drift' needs 'rays.fbs_pos' and 'rays.lbs_pos'; solve the scatterers first.", omni, omni, u,
            lg, "auto", "taps", 0);
        CHECK_NOTHROW(call(omni, omni, u, lg, "phase", "taps", 0, nullptr, nullptr, nullptr, nullptr, nullptr));
        const arma::mat R2 = {{120.0, 45.0, 1.5}, {120.1, 45.0, 1.5}}, Rb = {{120.0, 45.0, 1.5}, {120.1, 45.0, -0.1}};
        rays_t x = r;
        x.gr_eps_re.reset(), x.gr_eps_im.reset();
        msg("Input 'rays.gr_eps_re' is required to recompute the ground reflection away from the positions of "
            "'rays'.", omni, omni, x, lg, "drift", "rays", 0, nullptr, &R2);
        msg("Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'.", omni, omni, r, lg, "drift", "rays", 0,
            nullptr, &Rb);
        CHECK_NOTHROW(call(omni, omni, r, lg, "phase", "rays", 0, nullptr, &Rb, nullptr, nullptr, nullptr));
    }
}
