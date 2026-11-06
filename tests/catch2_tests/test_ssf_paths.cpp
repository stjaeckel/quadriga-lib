// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file (see qd_test_diag.hpp).

// Tests of ssf_paths: kept rays, path gains, path lengths, angles, scatterers and Jones matrices for the mobilities
// "phase" (TR 38.901 eq. (7.5-22) phase rotation, delays and angles of the ray data) and "drift" (fixed scatterers,
// geometry recomputed, ground reflection recomputed at moved ends), the K-factor rescale, the link gain, the oxygen
// absorption of TR 38.901 clause 7.6.1, optional outputs, edge cases and validation. The scatterer solver is tested in
// test_gbsm_rays_solve_scatterers.cpp; channel coefficients of a segment in test_gbsm_single_link_channel.cpp.
// Oracles: make_rays and the geometry of qd_test_rays.hpp (polyline, specular point, Fresnel coefficients of TR eqs.
// (7.6-38) to (7.6-40)), and in this file: the displacement phase exp(j·2·pi·f/c·(r_rx·dr_rx + r_tx·dr_tx)), its
// second-order error bound against the exact polyline phase, the K-factor form of the rescale and the oxygen
// coefficients of TR Table 7.6.1-1 at the test frequencies. Supporting tasks: gbsm_rays::solve_scatterers,
// gbsm_rays_from_paths, get_channels_planar, get_channels_spherical, ssf_clusters, gbsm_draw and ssf_rays.
// Fixture: make_rays (Tx (3, -4, 25), Rx (120, 45, 1.5); frequencies 3.5 + 4.5·k GHz). Units: [m], [s], [rad], [Hz].
// Tolerances: values formed by the same operations exactly; transcribed products and phases 1e-12 (relative for
// gains); the displacement phase at 1e-12 only for displacements up to 1 m (other evaluation orders differ by up to
// 4.6e-13 at 3.5 GHz and 10 m); solver residuals 1e-9 m and 1e-10 rad; the K rescale 1e-14 relative.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_lib.hpp"
#include "quadriga_gbsm.hpp"
#include "qd_test_diag.hpp"
#include "qd_test_rays.hpp"

#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    using rays_t = quadriga_lib::gbsm_rays;
    typedef std::complex<double> cx;

    // All outputs of one ssf_paths call
    struct po_t
    {
        arma::mat gain, fbs, lbs;
        arma::vec len, aod, eod, aoa, eoa;
        arma::cube M;
        arma::uvec type, ind;
    };

    // ssf_paths with all optional outputs; link gain 'lg' in [dB] per frequency (omitted: 0 dB)
    po_t pth(const rays_t &r, const std::string &mob, const arma::vec *T = nullptr, const arma::vec *R = nullptr,
             const arma::vec *kf = nullptr, bool o2 = false, const arma::vec *lg = nullptr)
    {
        po_t o;
        const arma::vec g = (lg != nullptr) ? *lg : arma::vec(r.n_freq(), arma::fill::zeros);
        quadriga_lib::ssf_paths(r, g, o.gain, o.len, o.M, mob, T, R, kf, &o.aod, &o.eod, &o.aoa, &o.eoa, &o.fbs,
                                &o.lbs, &o.type, &o.ind, o2);
        return o;
    }

    // Entry e (0 tt, 1 pt, 2 tp, 3 pp) of the Jones matrix of path p at frequency k, 8-row layout
    cx jx(const arma::cube &M, arma::uword e, arma::uword p, arma::uword k)
    {
        return cx(M(2 * e, p, k), M(2 * e + 1, p, k));
    }

    // Direction from O to P as azimuth and elevation
    void to_dir(const arma::vec &O, const arma::vec &P, double &az, double &el)
    {
        const arma::vec d = P - O;
        az = std::atan2(d(1), d(0)), el = std::atan2(d(2), std::hypot(d(0), d(1)));
    }

    // Mirror image in the ground z = 0
    arma::vec mirror(const arma::vec &x) { return arma::vec{x(0), x(1), -x(2)}; }

    // Indices of the kept rays: LOS and GR always, NLOS with power > 0 at some frequency
    arma::uvec kept(const rays_t &r)
    {
        std::vector<arma::uword> v;
        for (arma::uword i = 0; i < r.n_ray(); ++i)
            if (r.path_type(i) != 2 || arma::any(r.power.row(i) > 0.0))
                v.push_back(i);
        return arma::uvec(v);
    }

    // Ground reflection at the ends T, R for frequency k of r: R_par, R_perp (TR eqs. (7.6-38) to (7.6-40)), their
    // rms Rn = sqrt((abs(R_par)² + abs(R_perp)²) / 2), and the power law g of the method: (d3D / d_GR)² for the TR
    // methods (TR eq. (7.6-32) power term), 1 for quadriga
    struct gr_t
    {
        cx par, perp;
        double Rn, g;
    };
    gr_t gr_at(const rays_t &r, arma::uword k, const arma::vec &T, const arma::vec &R)
    {
        gr_t o;
        const double d2D = std::hypot(R(0) - T(0), R(1) - T(1));
        fresnel(r.gr_eps_re(k), r.gr_eps_im(k), T(2), R(2), d2D, o.par, o.perp);
        o.Rn = std::sqrt(0.5 * (std::norm(o.par) + std::norm(o.perp)));
        const double d_gr = len3(R - mirror(T)), d3D = len3(R - T);
        o.g = (r.method == "quadriga") ? 1.0 : (d3D / d_gr) * (d3D / d_gr);
        return o;
    }

    // Second-order bound of the phase error of the plane-wave rotation against the exact phase of the polyline for
    // ray i and the end displacements dT, dR at wavenumber k: NLOS per moving end k·d²/(2·(b - d)) with b the distance
    // from the end to its scatterer; LOS and GR k·abs(dR - dT')²/(2·(d - abs(dR) - abs(dT))) with d = d3D (LOS, dT'
    // = dT) or the GR length (dT' = dT mirrored in z, the image of the Tx)
    double phase_bound(const rays_t &r, arma::uword i, const arma::vec &dT, const arma::vec &dR, double k)
    {
        const arma::vec &T = r.tx_pos, &R = r.rx_pos;
        const double nT = len3(dT), nR = len3(dR);
        if (r.path_type(i) == 2)
            return k * nT * nT / (2.0 * (len3(r.fbs_pos.col(i) - T) - nT)) +
                   k * nR * nR / (2.0 * (len3(r.lbs_pos.col(i) - R) - nR));
        const bool gr = r.path_type(i) == 1;
        const double d = gr ? len3(R - mirror(T)) : len3(R - T), q = len3(dR - (gr ? mirror(dT) : dT));
        return k * q * q / (2.0 * (d - nR - nT));
    }

    // Displacement phase k·(r_rx·dR + r_tx·dT) of ray i at wavenumber k (TR eq. (7.5-22)), with r_tx, r_rx the unit
    // vectors of the departure and arrival angles
    double disp_phase(const rays_t &r, arma::uword i, const arma::vec &dT, const arma::vec &dR, double k)
    {
        const arma::vec u_t = unit_vec(r.aod(i), r.eod(i)), u_r = unit_vec(r.aoa(i), r.eoa(i));
        return k * (u_r(0) * dR(0) + u_r(1) * dR(1) + u_r(2) * dR(2) + u_t(0) * dT(0) + u_t(1) * dT(1) +
                    u_t(2) * dT(2));
    }

    // Exact path length of ray i with ends T + dT, R + dR and fixed scatterers (LOS: direct line, GR: image method)
    double exact_len(const rays_t &r, arma::uword i, const arma::vec &dT, const arma::vec &dR)
    {
        const arma::vec T = r.tx_pos + dT, R = r.rx_pos + dR;
        if (r.path_type(i) == 0)
            return len3(R - T);
        if (r.path_type(i) == 1)
            return len3(R - mirror(T));
        return polyline(T, r.fbs_pos.col(i), r.lbs_pos.col(i), R);
    }

    // Coefficient of path p at frequency k for one omni element at each end: get_channels_planar at the positions
    // of r ("phase" outputs) or get_channels_spherical at T, R ("drift" outputs)
    arma::cx_vec coeff(const quadriga_lib::arrayant<double> &a, const rays_t &r, const po_t &o, arma::uword k,
                       bool spherical, const arma::vec &T, const arma::vec &R)
    {
        arma::cube cr, ci, dl;
        const arma::vec g = o.gain.col(k);
        const arma::mat M = o.M.slice(k);
        const double f = r.center_frequency(k);
        if (spherical)
            quadriga_lib::get_channels_spherical<double>(&a, &a, T(0), T(1), T(2), 0.0, 0.0, 0.0, R(0), R(1), R(2),
                                                         0.0, 0.0, 0.0, &o.fbs, &o.lbs, &g, &o.len, &M, &cr, &ci,
                                                         &dl, f);
        else
            quadriga_lib::get_channels_planar<double>(&a, &a, T(0), T(1), T(2), 0.0, 0.0, 0.0, R(0), R(1), R(2),
                                                      0.0, 0.0, 0.0, &o.aod, &o.eod, &o.aoa, &o.eoa, &g, &o.len, &M,
                                                      &cr, &ci, &dl, f);
        arma::cx_vec c(cr.n_slices);
        for (arma::uword p = 0; p < cr.n_slices; ++p)
            c(p) = cx(cr(0, 0, p), ci(0, 0, p));
        return c;
    }
}

TEST_CASE("ssf_paths - phase at the ray-data positions returns the ray data")
{
    const rays_t r = make_rays(11, true, true, 6, 2, "38.901");
    const arma::vec lg = {-87.3, -101.9}, T = r.tx_pos, R = r.rx_pos;
    const double d3D = len3(R - T);
    const po_t o = pth(r, "phase", nullptr, nullptr, nullptr, false, &lg);
    const po_t o2 = pth(r, "phase", &T, &R, nullptr, false, &lg);

    // The zero-power NLOS ray (index 4) is dropped; LOS and GR are kept
    const arma::uvec ind = {0, 1, 2, 3, 5, 6, 7}, type = {0, 1, 2, 2, 2, 2, 2};
    REQUIRE(o.ind.n_elem == 7);
    CHECK(arma::all(o.ind == ind));
    CHECK(arma::all(o.type == type));
    REQUIRE(o.gain.n_rows == 7);
    REQUIRE(o.gain.n_cols == 2);
    REQUIRE(arma::size(o.M) == arma::size(8, 7, 2));
    CHECK((o.fbs.n_rows == 3 && o.fbs.n_cols == 0 && o.lbs.n_rows == 3 && o.lbs.n_cols == 0));
    for (arma::uword p = 0; p < 7; ++p)
    {
        const arma::uword i = ind(p);
        INFO("path " << p);
        for (arma::uword k = 0; k < 2; ++k)
            CHECK(std::abs(o.gain(p, k) / (r.power(i, k) * std::pow(10.0, 0.1 * lg(k))) - 1.0) < 1e-12);
        CHECK(std::abs(o.len(p) - (d3D + c0 * r.delay(i))) < 1e-12 * o.len(p));
        CHECK((o.aod(p) == r.aod(i) && o.eod(p) == r.eod(i) && o.aoa(p) == r.aoa(i) && o.eoa(p) == r.eoa(i)));
        for (arma::uword k = 0; k < 2; ++k)
            for (arma::uword e = 0; e < 8; ++e)
                CHECK(std::abs(o.M(e, p, k) - r.M(e, i, k)) < 1e-14);
    }
    // Positions given and equal to those of the rays: the same values
    CHECK(arma::approx_equal(o2.M, o.M, "absdiff", 0.0));
    CHECK(arma::approx_equal(o2.gain, o.gain, "absdiff", 0.0));
    CHECK(arma::approx_equal(o2.len, o.len, "absdiff", 0.0));
}

TEST_CASE("ssf_paths - phase rotation follows TR 7.5-22 for Rx and Tx displacements")
{
    const rays_t r = make_rays(12, true, true, 6, 2, "38.901_MF", true);
    const arma::uvec ind = kept(r);
    const arma::vec z3(3, arma::fill::zeros);

    // Against the exact phase -k·(L(t) - L(0)) of the polyline with fixed scatterers: within the second-order bound
    // for displacements of a few cm; the opposite sign of the rotation would be off by 2·k·(r·dr), far above it
    QD_DIAG("\n# diag: phase rotation vs exact polyline phase (3 cm displacements)\n");
    QD_DIAG("case,path,freq,err_rad,bound_rad\n");
    const arma::vec sR = 0.03 * unit_vec(0.7, 0.3), sT = 0.03 * unit_vec(-2.1, -0.2);
    const std::vector<std::pair<arma::vec, arma::vec>> small = {{z3, sR}, {sT, z3}, {sT, sR}};
    for (arma::uword q = 0; q < small.size(); ++q)
    {
        const arma::vec dT2 = small[q].first, dR2 = small[q].second, T = r.tx_pos + dT2, R = r.rx_pos + dR2;
        const po_t o = pth(r, "phase", &T, &R);
        double max_first = 0.0, max_bound = 0.0;
        for (arma::uword p = 0; p < ind.n_elem; ++p)
            for (arma::uword k = 0; k < 2; ++k)
            {
                const arma::uword i = ind(p);
                const double kw = 2.0 * pi * r.center_frequency(k) / c0;
                const double dL = exact_len(r, i, dT2, dR2) - exact_len(r, i, z3, z3);
                const double bound = phase_bound(r, i, dT2, dR2, kw);
                CHECK(std::abs(disp_phase(r, i, dT2, dR2, kw) + kw * dL) <= bound); // oracles agree
                const cx ex = std::polar(1.0, -kw * dL);
                double err = 0.0;
                for (arma::uword e = 0; e < 4; ++e)
                    if (std::abs(jx(r.M, e, i, k)) > 0.0)
                        err = std::max(err, std::abs(jx(o.M, e, p, k) - jx(r.M, e, i, k) * ex) /
                                                std::abs(jx(r.M, e, i, k)));
                QD_DIAG("%d,%d,%d,%.3e,%.3e\n", (int)q, (int)p, (int)k, err, bound);
                CHECK(err <= bound + 1e-9);
                max_first = std::max(max_first, std::abs(kw * dL)), max_bound = std::max(max_bound, bound);
            }
        CHECK(max_first > 100.0 * max_bound); // the oracle sees the sign
    }

    // Displacements up to 1 m (the phase of the oracle at 1e-12), Rx only, Tx only, both
    const arma::vec dR = {0.31, -0.62, 0.05}, dT = {-0.4, 0.2, 0.45};
    const std::vector<std::pair<arma::vec, arma::vec>> cases = {{z3, dR}, {dT, z3}, {dT, dR}};
    for (const auto &c : cases)
    {
        const arma::vec T = r.tx_pos + c.first, R = r.rx_pos + c.second;
        const arma::vec eT = T - r.tx_pos, eR = R - r.rx_pos; // the displacements as the positions give them
        const po_t o = pth(r, "phase", &T, &R);
        REQUIRE(o.ind.n_elem == ind.n_elem);
        for (arma::uword p = 0; p < ind.n_elem; ++p)
            for (arma::uword k = 0; k < 2; ++k)
            {
                const arma::uword i = ind(p);
                const double kw = 2.0 * pi * r.center_frequency(k) / c0;
                const double ph = disp_phase(r, i, eT, eR, kw);
                INFO("path " << p << ", frequency " << k);
                for (arma::uword e = 0; e < 4; ++e)
                    CHECK(std::abs(jx(o.M, e, p, k) - jx(r.M, e, i, k) * std::polar(1.0, ph)) < 1e-12);
                CHECK(std::abs(o.len(p) - len3(r.rx_pos - r.tx_pos) - c0 * r.delay(i)) < 1e-12 * o.len(p));
                CHECK(o.gain(p, k) == r.power(i, k));
            }
    }
}

TEST_CASE("ssf_paths - drift at the ray-data positions reproduces the drop geometry")
{
    rays_t r = make_rays(13, true, true, 6, 2, "38.901");
    r.solve_scatterers(2.0);
    const arma::vec lg = {-90.0, -95.5}, &T = r.tx_pos, &R = r.rx_pos;
    const double d3D = len3(R - T);
    const po_t o = pth(r, "drift", nullptr, nullptr, nullptr, false, &lg);
    const arma::uvec ind = kept(r);
    REQUIRE(arma::all(o.ind == ind));
    REQUIRE(arma::size(o.fbs) == arma::size(3, ind.n_elem));
    REQUIRE(r.fallback(5) == 1); // the zero-excess ray
    for (arma::uword p = 0; p < ind.n_elem; ++p)
    {
        const arma::uword i = ind(p);
        INFO("path " << p << ", ray " << i);
        CHECK(arma::approx_equal(o.M.col(p), r.M.col(i), "absdiff", 0.0));
        for (arma::uword k = 0; k < 2; ++k)
            CHECK(std::abs(o.gain(p, k) / (r.power(i, k) * std::pow(10.0, 0.1 * lg(k))) - 1.0) < 1e-12);
        const arma::vec u_t = unit_vec(o.aod(p), o.eod(p)), u_r = unit_vec(o.aoa(p), o.eoa(p));
        if (r.path_type(i) == 2)
        {
            const double L = d3D + c0 * r.delay(i), poly = polyline(T, r.fbs_pos.col(i), r.lbs_pos.col(i), R);
            CHECK(std::abs(o.len(p) - poly) < 1e-9);
            if (r.fallback(i) == 1) // longer polyline, angles exact
                CHECK(o.len(p) > L + 1e-3);
            else
                CHECK(std::abs(o.len(p) - L) < 1e-9);
            CHECK(ang_dist(u_t, unit_vec(r.aod(i), r.eod(i))) < 1e-10);
            CHECK(ang_dist(u_r, unit_vec(r.aoa(i), r.eoa(i))) < 1e-10);
            CHECK(arma::approx_equal(o.fbs.col(p), r.fbs_pos.col(i), "absdiff", 0.0));
            CHECK(arma::approx_equal(o.lbs.col(p), r.lbs_pos.col(i), "absdiff", 0.0));
        }
        else if (r.path_type(i) == 0)
        {
            CHECK(std::abs(o.len(p) - d3D) < 1e-12 * d3D);
            CHECK(ang_dist(u_t, R - T) < 1e-12);
            CHECK(ang_dist(u_r, T - R) < 1e-12);
            CHECK(arma::approx_equal(o.fbs.col(p), 0.5 * (T + R), "absdiff", 1e-12));
            CHECK(arma::approx_equal(o.lbs.col(p), 0.5 * (T + R), "absdiff", 1e-12));
        }
        else
        {
            const arma::vec G = specular_point(T, R);
            CHECK(std::abs(o.len(p) - len3(R - mirror(T))) < 1e-12 * o.len(p));
            CHECK(ang_dist(u_t, mirror(R) - T) < 1e-12);
            CHECK(ang_dist(u_r, mirror(T) - R) < 1e-12);
            CHECK(arma::approx_equal(o.fbs.col(p), G, "absdiff", 1e-12));
            CHECK(arma::approx_equal(o.lbs.col(p), G, "absdiff", 1e-12));
        }
    }

    // LOS and GR with the Tx on the ground: GR angles towards the image points (the GR point equals the Tx, where a
    // rule "towards the next point" would flip the sign of the departure elevation)
    rays_t g = make_rays(14, true, true, 0, 1, "38.901");
    g.tx_pos = {0.0, 0.0, 0.0}, g.rx_pos = {80.0, -30.0, 1.5};
    g.solve_scatterers();
    const po_t og = pth(g, "drift");
    REQUIRE(og.ind.n_elem == 2);
    const double d2D = std::hypot(80.0, 30.0), el = std::atan2(-1.5, d2D);
    CHECK(std::abs(og.eod(1) - el) < 1e-12);
    CHECK(std::abs(og.eoa(1) - el) < 1e-12);
    CHECK(og.eod(1) < 0.0);
    CHECK(std::abs(og.aod(1) - std::atan2(-30.0, 80.0)) < 1e-12);
    CHECK(std::abs(og.aoa(1) - std::atan2(30.0, -80.0)) < 1e-12);
    CHECK(std::abs(og.len(1) - std::sqrt(d2D * d2D + 1.5 * 1.5)) < 1e-12);
}

TEST_CASE("ssf_paths - drift recomputes polylines, angles, LOS and GR at moved ends")
{
    const arma::vec dT = {0.7, -0.4, 0.3}, dR = {1.2, 0.5, -0.2}, lg = {-80.0, -91.0};
    const std::vector<cx> scale = {cx(1.0, 0.0), cx(1.9, 0.0), std::polar(1.0, 1.0)};
    for (const char *method : {"38.901", "quadriga"})
        for (const cx &s : scale)
        {
            rays_t r = make_rays(15, true, true, 6, 2, method, true);
            for (arma::uword k = 0; k < 2; ++k) // GR matrix M_0 = s·diag(R_par, -R_perp) / Rn at the ray-data ends
                for (arma::uword e = 0; e < 4; ++e)
                {
                    const cx m = jx(r.M, e, 1, k) * s;
                    r.M(2 * e, 1, k) = m.real(), r.M(2 * e + 1, 1, k) = m.imag();
                }
            const arma::vec T = r.tx_pos + dT, R = r.rx_pos + dR, G = specular_point(T, R);
            const po_t o = pth(r, "drift", &T, &R, nullptr, false, &lg);
            const arma::uvec ind = kept(r);
            REQUIRE(arma::all(o.ind == ind));
            for (arma::uword p = 0; p < ind.n_elem; ++p)
            {
                const arma::uword i = ind(p);
                INFO("method " << method << ", scale " << s << ", path " << p);
                const arma::vec u_t = unit_vec(o.aod(p), o.eod(p)), u_r = unit_vec(o.aoa(p), o.eoa(p));
                if (r.path_type(i) == 2)
                {
                    const arma::vec F = r.fbs_pos.col(i), B = r.lbs_pos.col(i);
                    CHECK(std::abs(o.len(p) - polyline(T, F, B, R)) < 1e-12 * o.len(p));
                    CHECK(ang_dist(u_t, F - T) < 1e-12);
                    CHECK(ang_dist(u_r, B - R) < 1e-12);
                    CHECK(arma::approx_equal(o.fbs.col(p), F, "absdiff", 0.0));
                    CHECK(arma::approx_equal(o.lbs.col(p), B, "absdiff", 0.0));
                }
                else if (r.path_type(i) == 0)
                {
                    CHECK(std::abs(o.len(p) - len3(R - T)) < 1e-12 * o.len(p));
                    CHECK(ang_dist(u_t, R - T) < 1e-12);
                    CHECK(ang_dist(u_r, T - R) < 1e-12);
                    CHECK(arma::approx_equal(o.fbs.col(p), 0.5 * (T + R), "absdiff", 1e-12));
                }
                else
                {
                    CHECK(std::abs(o.len(p) - (len3(G - T) + len3(R - G))) < 1e-12 * o.len(p));
                    CHECK(ang_dist(u_t, mirror(R) - T) < 1e-12);
                    CHECK(ang_dist(u_r, mirror(T) - R) < 1e-12);
                    CHECK(arma::approx_equal(o.fbs.col(p), G, "absdiff", 1e-12));
                    CHECK(arma::approx_equal(o.lbs.col(p), G, "absdiff", 1e-12));
                }
                for (arma::uword k = 0; k < 2; ++k)
                {
                    const double lin = std::pow(10.0, 0.1 * lg(k));
                    if (r.path_type(i) != 1)
                    {
                        CHECK(std::abs(o.gain(p, k) / (r.power(i, k) * lin) - 1.0) < 1e-12);
                        CHECK(arma::approx_equal(o.M.slice(k).col(p), r.M.slice(k).col(i), "absdiff", 0.0));
                        continue;
                    }
                    // GR: power × g(t) / g(0); M(t) = s·diag(R_par(t), -R_perp(t)) / Rn(0)
                    const gr_t g0 = gr_at(r, k, r.tx_pos, r.rx_pos), gt = gr_at(r, k, T, R);
                    CHECK(std::abs(o.gain(p, k) / (r.power(i, k) * gt.g / g0.g * lin) - 1.0) < 1e-12);
                    const cx ex[4] = {s * gt.par / g0.Rn, 0.0, 0.0, -s * gt.perp / g0.Rn};
                    for (arma::uword e = 0; e < 4; ++e)
                        CHECK(std::abs(jx(o.M, e, p, k) - ex[e]) < 1e-12);
                    if (std::string(method) == "38.901")
                        CHECK(std::abs(gt.g / g0.g - 1.0) > 1e-4); // the TR power law changes with the move
                }
            }

            // Continuous at the ray-data positions: a 1 µm move changes the GR matrix by far less than 1e-5
            const arma::vec R1 = r.rx_pos + arma::vec{1e-6, 0.0, 0.0};
            const po_t o1 = pth(r, "drift", nullptr, &R1);
            CHECK(arma::approx_equal(o1.M.col(1), r.M.col(1), "absdiff", 1e-5));
            CHECK(arma::abs(arma::vectorise(o1.M.col(1) - r.M.col(1))).max() > 0.0);
        }

    // Without permittivity at the ray-data positions: no recomputation needed, GR unchanged
    rays_t r = make_rays(16, true, true, 6, 2, "38.901", true);
    r.gr_eps_re.reset(), r.gr_eps_im.reset();
    const po_t o = pth(r, "drift");
    CHECK(arma::approx_equal(o.M.col(1), r.M.col(1), "absdiff", 0.0));
    CHECK(arma::approx_equal(o.gain.row(1), r.power.row(1), "absdiff", 0.0));

    // Rx moved to directly below the Tx (vertical LOS): EOD -pi/2 and EOA +pi/2
    const rays_t q = make_rays(16, true, true, 6, 2, "38.901", true);
    const arma::vec Rv = {q.tx_pos(0), q.tx_pos(1), 1.5};
    const po_t v = pth(q, "drift", nullptr, &Rv);
    REQUIRE(v.type(0) == 0);
    CHECK(std::abs(v.eod(0) + 0.5 * arma::datum::pi) < 1e-12);
    CHECK(std::abs(v.eoa(0) - 0.5 * arma::datum::pi) < 1e-12);
}

TEST_CASE("ssf_paths - drift path lengths change by at most the displacement")
{
    rays_t r = make_rays(17, true, true, 8, 1, "quadriga");
    r.solve_scatterers(2.0);
    const po_t o0 = pth(r, "drift");
    qd_uniform U(171);
    for (int t = 0; t < 30; ++t)
    {
        const arma::vec z3(3, arma::fill::zeros);
        const arma::vec dT = (t % 3 == 0) ? z3 : U(0.0, 3.0) * unit_vec(U(-pi, pi), U(-0.5, 0.5));
        const arma::vec dR = (t % 3 == 1) ? z3 : U(0.0, 3.0) * unit_vec(U(-pi, pi), U(-0.4, 0.4));
        const arma::vec T = r.tx_pos + dT, R = r.rx_pos + dR;
        const po_t o = pth(r, "drift", &T, &R);
        INFO("trial " << t);
        CHECK(arma::all(arma::abs(o.len - o0.len) <= len3(dT) + len3(dR) + 1e-9));
        CHECK(arma::abs(o.len - o0.len).max() > 0.1 * (len3(dT) + len3(dR))); // the lengths do change
    }
}

TEST_CASE("ssf_paths - kf_ratio rescales LOS and GR against NLOS as K' = r·K_c")
{
    const rays_t r = make_rays(18, true, true, 6, 3, "38.901");
    const arma::vec kf = {0.3, 2.5, 0.0}, ones(3, arma::fill::ones);
    const po_t o = pth(r, "phase", nullptr, nullptr, &kf), o1 = pth(r, "phase", nullptr, nullptr, &ones);
    const po_t on = pth(r, "phase");
    CHECK(arma::approx_equal(o1.gain, on.gain, "reldiff", 1e-15));
    const arma::uvec ind = kept(r);
    for (arma::uword k = 0; k < 3; ++k)
    {
        // TR coefficient powers with the K-factor K' = r·K_c, K_c = S_L / S_N: LOS K'/(K' + 1)·S, the NLOS rays share
        // S/(K' + 1) in proportion to their powers, GR scales with the LOS ray (S = S_L + S_N)
        double SL = 0.0, SN = 0.0;
        for (arma::uword i = 0; i < r.n_ray(); ++i)
            (r.path_type(i) == 0 ? SL : SN) += (r.path_type(i) == 1) ? 0.0 : r.power(i, k);
        const double S = SL + SN, Kp = kf(k) * SL / SN, PL = S * Kp / (Kp + 1.0);
        double sum = 0.0;
        for (arma::uword p = 0; p < ind.n_elem; ++p)
        {
            const arma::uword i = ind(p);
            const double P = r.power(i, k);
            const double ex = (r.path_type(i) == 2)   ? P * S / (SN * (Kp + 1.0))
                              : (r.path_type(i) == 0) ? PL
                                                      : P * PL / SL;
            sum += (r.path_type(i) == 1) ? 0.0 : ex;
            INFO("frequency " << k << ", path " << p);
            CHECK(std::abs(o.gain(p, k) - ex) <= 1e-14 * std::max(ex, 1e-300));
        }
        CHECK(std::abs(sum - S) < 1e-14 * S); // LOS + NLOS keep their sum
    }
    CHECK(o.gain(0, 2) == 0.0); // kf_ratio = 0: LOS and GR gain 0
    CHECK(o.gain(1, 2) == 0.0);

    // r·S_L + S_N = 0 (no NLOS power at frequency 1 and kf_ratio 0): LOS and GR 0, NLOS unchanged
    rays_t z = r;
    for (arma::uword i = 2; i < z.n_ray(); ++i)
        z.power(i, 1) = 0.0;
    const arma::vec kz = {1.0, 0.0, 1.0};
    const po_t oz = pth(z, "phase", nullptr, nullptr, &kz);
    CHECK(oz.gain.is_finite());
    CHECK((oz.gain(0, 1) == 0.0 && oz.gain(1, 1) == 0.0));
    CHECK(arma::all(oz.gain.col(1).subvec(2, oz.gain.n_rows - 1) == 0.0));
    CHECK(std::abs(oz.gain(0, 0) - z.power(0, 0)) < 1e-14 * z.power(0, 0));
}

TEST_CASE("ssf_paths - oxygen loss from each path's own length")
{
    // TR Table 7.6.1-1 interpolated at the test frequencies: 0.5, 15, 14.8, 8.65, 0.5, 0 dB/km
    rays_t r = make_rays(19, true, true, 6, 6, "38.901");
    r.center_frequency = {52.5e9, 60.0e9, 60.5e9, 63.5e9, 67.5e9, 70.0e9};
    const double alpha[6] = {0.5, 15.0, 14.8, 8.65, 0.5, 0.0};
    rays_t s = r;
    s.solve_scatterers();
    const arma::vec R = r.rx_pos + arma::vec{2.0, -1.0, 0.5};
    for (int mode = 0; mode < 2; ++mode)
    {
        const rays_t &x = (mode == 0) ? r : s;
        const std::string mob = (mode == 0) ? "phase" : "drift";
        const po_t a = pth(x, mob, nullptr, &R), b = pth(x, mob, nullptr, &R, nullptr, true);
        CHECK(arma::approx_equal(a.len, b.len, "absdiff", 0.0));
        CHECK(arma::approx_equal(a.M, b.M, "absdiff", 0.0));
        for (arma::uword p = 0; p < a.len.n_elem; ++p)
            for (arma::uword k = 0; k < 6; ++k)
            {
                INFO(mob << ", path " << p << ", frequency " << k);
                const double ex = a.gain(p, k) * std::pow(10.0, -alpha[k] * a.len(p) / 1e4);
                CHECK(std::abs(b.gain(p, k) - ex) <= 1e-12 * ex);
            }
        // The two rays of one tap (rays 2 and 3) have their own lengths and losses at 60 GHz
        const double l2 = b.gain(2, 1) / a.gain(2, 1), l3 = b.gain(3, 1) / a.gain(3, 1);
        CHECK(std::abs(10.0 * std::log10(l2 / l3)) > 1e-3);
    }

    // The table ends at 52 and 68 GHz (0 dB/km) and far outside the table (fc·1e-9 - 52 beyond the int range): no
    // loss, with and without oxygen
    rays_t h = r;
    h.center_frequency(0) = 52e9, h.center_frequency(4) = 68e9, h.center_frequency(5) = 1e300;
    const po_t c = pth(h, "phase", nullptr, &R), d = pth(h, "phase", nullptr, &R, nullptr, true);
    for (arma::uword k : {0, 4, 5})
        CHECK(arma::approx_equal(c.gain.col(k), d.gain.col(k), "absdiff", 0.0));
}

TEST_CASE("ssf_paths - auto mobility and optional outputs alone or together give the same values")
{
    const arma::vec dR = {0.4, 0.1, 0.0}, lg = {-77.0, -88.0};
    for (const char *method : {"38.901", "38.901_MF", "quadriga"})
    {
        rays_t r = make_rays(20, true, true, 6, 2, method);
        r.solve_scatterers();
        const arma::vec R = r.rx_pos + dR;
        const bool qd = std::string(method) == "quadriga";
        const po_t a = pth(r, "auto", nullptr, &R, nullptr, false, &lg);
        const po_t e = pth(r, qd ? "drift" : "phase", nullptr, &R, nullptr, false, &lg);
        INFO("method " << method);
        CHECK((a.gain.n_rows == e.gain.n_rows && arma::approx_equal(a.gain, e.gain, "absdiff", 0.0)));
        CHECK(arma::approx_equal(a.M, e.M, "absdiff", 0.0));
        CHECK(arma::approx_equal(a.len, e.len, "absdiff", 0.0));
        CHECK(a.fbs.n_cols == (qd ? a.len.n_elem : 0));

        // Each optional output alone and none at all: the same values
        arma::mat g, fb, lb;
        arma::vec l, v;
        arma::cube M;
        arma::uvec u;
        quadriga_lib::ssf_paths(r, lg, g, l, M, "drift", nullptr, &R);
        const po_t d = pth(r, "drift", nullptr, &R, nullptr, false, &lg);
        CHECK((arma::approx_equal(g, d.gain, "absdiff", 0.0) && arma::approx_equal(l, d.len, "absdiff", 0.0) &&
               arma::approx_equal(M, d.M, "absdiff", 0.0)));
        const arma::vec *ang[4] = {&d.aod, &d.eod, &d.aoa, &d.eoa};
        for (int q = 0; q < 4; ++q)
        {
            quadriga_lib::ssf_paths(r, lg, g, l, M, "drift", nullptr, &R, nullptr, q == 0 ? &v : nullptr,
                                    q == 1 ? &v : nullptr, q == 2 ? &v : nullptr, q == 3 ? &v : nullptr);
            CHECK(arma::approx_equal(v, *ang[q], "absdiff", 0.0));
        }
        quadriga_lib::ssf_paths(r, lg, g, l, M, "drift", nullptr, &R, nullptr, nullptr, nullptr, nullptr, nullptr, &fb);
        quadriga_lib::ssf_paths(r, lg, g, l, M, "drift", nullptr, &R, nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, &lb);
        CHECK((arma::approx_equal(fb, d.fbs, "absdiff", 0.0) && arma::approx_equal(lb, d.lbs, "absdiff", 0.0)));
        quadriga_lib::ssf_paths(r, lg, g, l, M, "drift", nullptr, &R, nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, &u);
        CHECK(arma::all(u == d.type));
        quadriga_lib::ssf_paths(r, lg, g, l, M, "drift", nullptr, &R, nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr, &u);
        CHECK(arma::all(u == d.ind));

        // Pre-sized outputs on strict external memory are written in place; stale outputs of another size resized
        const arma::uword n = d.len.n_elem;
        std::vector<double> b_g(2 * n, -7.0), b_l(n, -7.0), b_M(16 * n, -7.0), b_a(n, -7.0), b_f(3 * n, -7.0);
        std::vector<arma::uword> b_t(n, 9);
        arma::mat sg(b_g.data(), n, 2, false, true), sf(b_f.data(), 3, n, false, true);
        arma::vec sl(b_l.data(), n, false, true), sa(b_a.data(), n, false, true);
        arma::cube sM(b_M.data(), 8, n, 2, false, true);
        arma::uvec st(b_t.data(), n, false, true);
        arma::mat lb2(2, 2, arma::fill::value(5.0));
        arma::vec eo(n + 3, arma::fill::value(5.0)), aa(1, arma::fill::value(5.0)), ee;
        arma::uvec pt(1, arma::fill::value(5));
        quadriga_lib::ssf_paths(r, lg, sg, sl, sM, "drift", nullptr, &R, nullptr, &sa, &eo, &aa, &ee, &sf, &lb2, &pt,
                                &st);
        CHECK((sg.memptr() == b_g.data() && sl.memptr() == b_l.data() && sM.memptr() == b_M.data() &&
               sa.memptr() == b_a.data() && sf.memptr() == b_f.data() && st.memptr() == b_t.data()));
        CHECK((arma::approx_equal(sg, d.gain, "absdiff", 0.0) && arma::approx_equal(sl, d.len, "absdiff", 0.0) &&
               arma::approx_equal(sM, d.M, "absdiff", 0.0) && arma::approx_equal(sa, d.aod, "absdiff", 0.0) &&
               arma::approx_equal(sf, d.fbs, "absdiff", 0.0) && arma::all(st == d.ind)));
        CHECK((arma::approx_equal(eo, d.eod, "absdiff", 0.0) && arma::approx_equal(aa, d.aoa, "absdiff", 0.0) &&
               arma::approx_equal(ee, d.eoa, "absdiff", 0.0) && arma::approx_equal(lb2, d.lbs, "absdiff", 0.0) &&
               arma::all(pt == d.type)));
    }
}

TEST_CASE("ssf_paths - phase and drift agree for millimetre displacements")
{
    // One omni element at each end; "phase": get_channels_planar at the ray-data positions, "drift":
    // get_channels_spherical at the moved positions. LOS and NLOS (no fallback ray in this fixture): equal within
    // the second-order bound; GR: drift = phase × sqrt(g(t) / g(0))·R_par(t) / R_par(0) (vertical polarisation),
    // the change of the GR coefficient at the moved ends
    const auto omni = quadriga_lib::generate_arrayant_omni<double>();
    for (const char *method : {"38.901", "quadriga"})
    {
        const rays_t r = make_rays(21, true, true, 6, 2, method, true);
        const arma::vec z3(3, arma::fill::zeros), mR = 1e-3 * unit_vec(0.3, 0.2), mT = 1e-3 * unit_vec(-1.1, -0.4);
        const std::vector<std::pair<arma::vec, arma::vec>> cases = {{z3, mR}, {mT, z3}, {mT, mR}};
        QD_DIAG("\n# diag: phase vs drift coefficients, 1 mm, method %s\ncase,path,freq,err,bound\n", method);
        for (arma::uword q = 0; q < cases.size(); ++q)
        {
            const arma::vec dT = cases[q].first, dR = cases[q].second, T = r.tx_pos + dT, R = r.rx_pos + dR;
            const po_t op = pth(r, "phase", &T, &R), od = pth(r, "drift", &T, &R);
            REQUIRE(op.ind.n_elem == od.ind.n_elem);
            double max_first = 0.0, max_bound = 0.0;
            for (arma::uword k = 0; k < 2; ++k)
            {
                const arma::cx_vec cp = coeff(omni, r, op, k, false, r.tx_pos, r.rx_pos);
                const arma::cx_vec cd = coeff(omni, r, od, k, true, T, R);
                const double kw = 2.0 * pi * r.center_frequency(k) / c0;
                for (arma::uword p = 0; p < op.ind.n_elem; ++p)
                {
                    const arma::uword i = op.ind(p);
                    cx ex = cp(p);
                    if (r.path_type(i) == 1)
                    {
                        const gr_t g0 = gr_at(r, k, r.tx_pos, r.rx_pos), gt = gr_at(r, k, T, R);
                        ex *= std::sqrt(gt.g / g0.g) * gt.par / g0.par;
                    }
                    const double bound = phase_bound(r, i, dT, dR, kw), err = std::abs(cd(p) - ex) / std::abs(ex);
                    QD_DIAG("%d,%d,%d,%.3e,%.3e\n", (int)q, (int)p, (int)k, err, bound);
                    INFO("method " << method << ", case " << q << ", path " << p << ", frequency " << k);
                    CHECK(err <= bound + 1e-9);
                    max_bound = std::max(max_bound, bound);
                    max_first = std::max(max_first, std::abs(disp_phase(r, i, dT, dR, kw)));
                }
            }
            CHECK(max_first > 100.0 * max_bound); // the opposite sign of the rotation would fail
        }
    }
}

TEST_CASE("ssf_paths - n_ray = 0, NLOS scenario and data-from-paths rays")
{
    // n_ray = 0, both mobilities, solved and unsolved
    rays_t e = make_rays(22, false, false, 0, 2, "38.901");
    for (int solved = 0; solved < 2; ++solved)
    {
        if (solved == 1)
            e.solve_scatterers();
        for (const char *mob : {"phase", "drift"})
        {
            po_t o;
            CHECK_NOTHROW(o = pth(e, mob));
            CHECK((arma::size(o.gain) == arma::size(0, 2) && o.len.n_elem == 0 &&
                   arma::size(o.M) == arma::size(8, 0, 2)));
            CHECK((o.aod.n_elem == 0 && o.eoa.n_elem == 0 && o.type.n_elem == 0 && o.ind.n_elem == 0));
            CHECK((arma::size(o.fbs) == arma::size(3, 0) && arma::size(o.lbs) == arma::size(3, 0)));
        }
    }

    // NLOS scenario: LOS and GR at power 0 are kept with gain 0; a ray with power 0 at frequency 0 only is kept
    rays_t n = make_rays(23, true, true, 6, 2, "38.901");
    n.power.row(0).zeros(), n.power.row(1).zeros(), n.power(6, 0) = 0.0;
    n.solve_scatterers();
    const arma::vec kf = {2.0, 0.5};
    for (const char *mob : {"phase", "drift"})
    {
        const po_t o = pth(n, mob, nullptr, nullptr, &kf);
        INFO(mob);
        REQUIRE(o.ind.n_elem == 7);
        CHECK(arma::all(o.ind == arma::uvec{0, 1, 2, 3, 5, 6, 7}));
        CHECK(arma::all(arma::vectorise(o.gain.rows(0, 1)) == 0.0));
        CHECK((o.gain(5, 0) == 0.0 && o.gain(5, 1) > 0.0));
        CHECK(o.gain.is_finite());
    }

    // Data from paths: the drift outputs of a solved object back through gbsm_rays_from_paths, with link gain 0 dB,
    // give the same paths
    rays_t s = make_rays(24, true, true, 6, 2, "quadriga");
    s.solve_scatterers();
    const po_t o = pth(s, "drift");
    rays_t d;
    quadriga_lib::gbsm_rays_from_paths(s.center_frequency, s.tx_pos, s.rx_pos, o.fbs, o.lbs, o.gain, o.len, o.M, d,
                                       "quadriga", &o.type, nullptr, &s.gr_eps_re, &s.gr_eps_im);
    const po_t od = pth(d, "auto");
    REQUIRE(od.ind.n_elem == o.ind.n_elem);
    CHECK(arma::approx_equal(od.gain, o.gain, "reldiff", 1e-14));
    CHECK(arma::approx_equal(od.len, o.len, "absdiff", 1e-9));
    CHECK(arma::approx_equal(od.M, o.M, "absdiff", 0.0));
    CHECK(arma::approx_equal(od.fbs, o.fbs, "absdiff", 1e-12));
    CHECK(arma::all(od.type == o.type));
}

TEST_CASE("ssf_paths - input validation")
{
    rays_t r = make_rays(25, true, true, 4, 2, "38.901");
    rays_t s = r;
    s.solve_scatterers();
    const arma::vec lg = {-80.0, -90.0}, kf = {1.0, 2.0};
    arma::mat g(1, 1, arma::fill::value(-3.0));
    arma::vec l;
    arma::cube M;
    auto call = [&](const rays_t &x, const arma::vec &lgx, const std::string &mob, const arma::vec *T,
                    const arma::vec *R, const arma::vec *k)
    { quadriga_lib::ssf_paths(x, lgx, g, l, M, mob, T, R, k); };
    auto ok = [&](const rays_t &x, const arma::vec &lgx, const std::string &mob, const arma::vec *T,
                  const arma::vec *R, const arma::vec *k)
    {
        arma::mat g2;
        arma::vec l2;
        arma::cube M2;
        quadriga_lib::ssf_paths(x, lgx, g2, l2, M2, mob, T, R, k);
    };
    auto thr = [&](const std::function<void()> &f, const std::string &msg)
    {
        CHECK_THROWS_MATCHES(f(), std::invalid_argument, Message(msg));
        CHECK((g.n_elem == 1 && g(0) == -3.0)); // outputs untouched
    };
    const arma::vec T = r.tx_pos + arma::vec{1.0, 0.0, 0.0}, R = r.rx_pos;

    SECTION("baseline")
    {
        CHECK_NOTHROW(ok(r, lg, "phase", &T, &R, &kf));
        CHECK_NOTHROW(ok(s, lg, "drift", &T, &R, &kf));
        CHECK_NOTHROW(ok(s, lg, "auto", nullptr, nullptr, nullptr));
        rays_t q = s;
        q.method = "quadriga";
        CHECK_NOTHROW(ok(q, lg, "auto", &T, nullptr, nullptr));
    }
    SECTION("rays")
    {
        thr([&] { call(rays_t(), lg, "phase", nullptr, nullptr, nullptr); },
            "Input 'rays.tx_pos' must have 3 elements.");
        rays_t x = r;
        x.method = "x";
        thr([&] { call(x, lg, "phase", nullptr, nullptr, nullptr); },
            "Input 'rays.method' must be '38.901', '38.901_MF' or 'quadriga'.");
    }
    SECTION("mobility")
    {
        thr([&] { call(r, lg, "Phase", nullptr, nullptr, nullptr); },
            "Input 'mobility' must be 'phase', 'drift' or 'auto'.");
    }
    SECTION("link_gain_dB")
    {
        thr([&] { call(r, arma::vec{-80.0}, "phase", nullptr, nullptr, nullptr); },
            "Input 'link_gain_dB' must have n_freq elements.");
        thr([&] { call(r, arma::vec{-80.0, arma::datum::inf}, "phase", nullptr, nullptr, nullptr); },
            "Input 'link_gain_dB' must be finite.");
    }
    SECTION("tx_pos and rx_pos")
    {
        const arma::vec two = {1.0, 2.0}, nan3 = {1.0, arma::datum::nan, 2.0};
        thr([&] { call(r, lg, "phase", &two, nullptr, nullptr); }, "Input 'tx_pos' must have 3 elements.");
        thr([&] { call(r, lg, "phase", &nan3, nullptr, nullptr); }, "Input 'tx_pos' must be finite.");
        thr([&] { call(r, lg, "phase", nullptr, &two, nullptr); }, "Input 'rx_pos' must have 3 elements.");
        thr([&] { call(r, lg, "phase", nullptr, &nan3, nullptr); }, "Input 'rx_pos' must be finite.");
    }
    SECTION("kf_ratio")
    {
        const arma::vec one = {1.0}, neg = {1.0, -0.1}, nan2 = {arma::datum::nan, 1.0};
        thr([&] { call(r, lg, "phase", nullptr, nullptr, &one); }, "Input 'kf_ratio' must have n_freq elements.");
        thr([&] { call(r, lg, "phase", nullptr, nullptr, &neg); }, "Input 'kf_ratio' must be finite and >= 0.");
        thr([&] { call(r, lg, "phase", nullptr, nullptr, &nan2); }, "Input 'kf_ratio' must be finite and >= 0.");
    }
    SECTION("drift without scatterers")
    {
        const std::string msg = "Mobility 'drift' needs 'rays.fbs_pos' and 'rays.lbs_pos'; solve the scatterers first.";
        thr([&] { call(r, lg, "drift", nullptr, nullptr, nullptr); }, msg);
        rays_t q = r;
        q.method = "quadriga";
        thr([&] { call(q, lg, "auto", nullptr, nullptr, nullptr); }, msg);
    }
    SECTION("drift with a GR ray at moved ends")
    {
        rays_t x = s;
        x.gr_eps_re.reset(), x.gr_eps_im.reset();
        CHECK_NOTHROW(ok(x, lg, "drift", nullptr, nullptr, nullptr));
        CHECK_NOTHROW(ok(x, lg, "phase", &T, nullptr, nullptr));
        thr([&] { call(x, lg, "drift", &T, nullptr, nullptr); },
            "Input 'rays.gr_eps_re' is required to recompute the ground reflection away from the positions of 'rays'.");
        const arma::vec Tb = {3.0, -4.0, -0.5}, Rb = {120.0, 45.0, -1.0};
        CHECK_NOTHROW(ok(s, lg, "phase", &Tb, &Rb, nullptr));
        thr([&] { call(s, lg, "drift", &Tb, nullptr, nullptr); },
            "Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'.");
        thr([&] { call(s, lg, "drift", nullptr, &Rb, nullptr); },
            "Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'.");
    }
}

TEST_CASE("Diagnostic - ssf_paths fallback share and b_med on a library drop")
{
    // Library drop: UMi NLOS at 3.5 GHz, 200 UTs; rays of ssf_rays for "38.901" (no absolute TOA: the first cluster
    // has zero excess delay) and "quadriga" (absolute TOA), scatterers with d_min = 2 m, paths in drift at the drop
    // positions. Columns: share of fallback rays among the NLOS rays with power, median and p99 of the delay added
    // by the fallback in [ns], p50 of b_med (Tx, Rx) in [m], largest residual of the solved polylines in [m].
    // Read against the solver numbers in the docs of solve_scatterers (fallback 5.5 to 9.6 % for the TR methods
    // without absolute TOA, at most 2.3 % otherwise) and 0.2·b_med as the drift validity scale.
    const spmap sp = read_v19("UMi_NLOS");
    const arma::mat tx = {{0.0, 0.0, 10.0}}, rx = drop_rx(291, 200);
    const arma::vec fc = {3.5e9};
    const lib_t in = lib_inputs(sp, tx, rx, 292, 3.5e9);
    arma::cube mu(200, 1, 8, arma::fill::value(1.0)), sigma(200, 1, 8, arma::fill::value(3.0));
    mu.slice(7).fill(9.0);
    QD_DIAG("\n# diag: library drop UMi NLOS 3.5 GHz, 200 links, d_min 2 m\n");
    QD_DIAG("method,fallback_share,added_ns_p50,added_ns_p99,bmed_tx_p50,bmed_rx_p50,max_residual_m\n");
    for (const char *method : {"38.901", "quadriga"})
    {
        const bool qd = std::string(method) == "quadriga";
        out_t o;
        quadriga_lib::ssf_clusters(sp, tx, rx, in.lsp, in.rnd, o.delay, o.power, o.aod, o.eod, o.aoa, o.eoa, method,
                                   qd ? nullptr : &in.rnd_freq, nullptr, &in.eod, qd ? &in.toa : nullptr);
        arma::cube ph;
        arma::mat x;
        quadriga_lib::gbsm_init_phases(sp, "ray", ph, 292, qd ? -1 : 0);
        quadriga_lib::gbsm_draw(sp, "ray", ph, tx, rx, x, qd ? -1 : 0, "Comb256", nullptr, &fc);
        const arma::cube rnd(x.memptr(), x.n_rows, x.n_cols, 1);
        std::vector<rays_t> rays;
        quadriga_lib::ssf_rays(sp, tx, rx, fc, in.lsp, mu, sigma, o.delay, o.power, o.aod, o.eod, o.aoa, o.eoa, rnd,
                               rays, method);
        std::vector<double> added, bt, br;
        double n_on = 0.0, res = 0.0;
        for (rays_t &r : rays)
        {
            r.solve_scatterers(2.0);
            const po_t p = pth(r, "drift");
            const double d3D = len3(r.rx_pos - r.tx_pos);
            for (arma::uword q = 0; q < p.ind.n_elem; ++q)
            {
                const arma::uword i = p.ind(q);
                if (r.path_type(i) != 2)
                    continue;
                n_on += 1.0;
                const double L = d3D + c0 * r.delay(i);
                if (r.fallback(i) == 1)
                    added.push_back((p.len(q) - L) / c0 * 1e9);
                else
                    res = std::max(res, std::abs(p.len(q) - L));
            }
            const arma::vec b = r.b_med();
            bt.push_back(b(0)), br.push_back(b(1));
        }
        const arma::vec a = added.empty() ? arma::vec{0.0} : arma::vec(added);
        const arma::vec qa = arma::quantile(a, arma::vec{0.5, 0.99});
        const double mt = arma::median(arma::vec(bt)), mr = arma::median(arma::vec(br));
        QD_DIAG("%s,%.4f,%.3f,%.3f,%.2f,%.2f,%.2e\n", method, (double)added.size() / n_on, qa(0), qa(1), mt, mr, res);
        CHECK(n_on > 0.0);
        CHECK((qa.is_finite() && std::isfinite(mt) && std::isfinite(mr) && std::isfinite(res)));
    }
}
