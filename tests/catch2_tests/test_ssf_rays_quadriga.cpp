// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file (see qd_test_diag.hpp).

// Tests of ssf_rays with the method "quadriga" (one geometry for all frequencies, QuaDRiGa ray delays), the rotation
// angle model, the rotation polarisation model of QuaDRiGa-NG with its owner end and the polarisation power
// variability of TR 38.901 clause 7.6.16; swap reciprocity of both polarisation models with dual-mobility random
// values; the input checks of these options.
// Oracles: ref_rays_qd of qd_test_rays.hpp (typed from the model description), closed forms in the cases (zero-mean
// delays, great-circle distances, eta of eqs. (7.6-61), (7.6-62), unitarity). Inputs: clusters of ssf_clusters on
// make_fix (qd_test_ssf.hpp), make_lsp_stats and make_rnd_ray; the swap and diagnostic cases use the library's random
// sets (gbsm_init_phases, gbsm_draw).
// Tolerances: values formed by the same operations exactly; transcribed angles and Jones entries 1e-12 (other order
// of the operations), delays and powers 1e-12 relative; delay offsets 1e-18 s; swapped angles 1e-10 rad (the value
// of the cluster swap test).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"
#include "quadriga_tools.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "qd_test_diag.hpp"
#include "qd_test_rays.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    typedef std::complex<double> cx;

    // Entry e of the 8-row layout (0 tt, 1 pt, 2 tp, 3 pp) of ray i at frequency k
    cx jones(const quadriga_lib::gbsm_rays &r, arma::uword e, arma::uword i, arma::uword k)
    {
        return cx(r.M(2 * e, i, k), r.M(2 * e + 1, i, k));
    }

    // Link l of a "quadriga" call against ref_rays_qd at all frequencies
    void check_qd(const quadriga_lib::gbsm_rays &r, const rays_in_t &in, arma::uword l)
    {
        const std::vector<ref_rays_t> e = ref_rays_qd(in.sp, in.f, in.cl, in.st, in.rnd, in.fc, l, in.angle_model,
                                                      in.pol_model, in.pol_variability, in.random_phase);
        const arma::uword F = in.fc.n_elem;
        REQUIRE(r.n_ray() == e[0].delay.size());
        REQUIRE(r.n_freq() == F);
        const arma::vec *ang[4] = {&r.aod, &r.aoa, &r.eod, &r.eoa};
        double d_ang = 0.0, d_rel = 0.0, d_J = 0.0;
        for (arma::uword i = 0; i < r.n_ray(); ++i)
        {
            INFO("link " << l << ", ray " << i);
            CHECK(r.path_type(i) == e[0].path_type[i]);
            CHECK(r.tap(i) == e[0].tap[i]);
            d_rel = std::max(d_rel, std::abs(r.delay(i) - e[0].delay[i]) / std::max(e[0].delay[i], 1e-9));
            for (int q = 0; q < 4; ++q)
            {
                const double d = (*ang[q])(i)-e[0].ang[q][i];
                d_ang = std::max(d_ang, std::abs(q < 2 ? wrap_rad(d) : d));
            }
            for (arma::uword k = 0; k < F; ++k)
            {
                d_rel = std::max(d_rel, std::abs(r.power(i, k) - e[k].power[i]) / std::max(e[k].power[i], 1e-300));
                const cx J[4] = {e[k].J[4 * i], e[k].J[4 * i + 2], e[k].J[4 * i + 1], e[k].J[4 * i + 3]};
                for (arma::uword j = 0; j < 4; ++j)
                    d_J = std::max(d_J, std::abs(jones(r, j, i, k) - J[j]));
            }
        }
        QD_DIAG("%s,%s,%llu,%llu,%.3e,%.3e,%.3e\n", in.angle_model.c_str(), in.pol_model.c_str(),
                (unsigned long long)F, (unsigned long long)l, d_rel, d_ang, d_J);
        CHECK(d_rel < 1e-12);
        CHECK(d_ang < 1e-12);
        CHECK(d_J < 1e-12);
    }

    // Distinct XPR LSPs per link and frequency in (3, 30) (make_fix sets 10 everywhere)
    void set_xpr(rays_in_t &in, unsigned seed)
    {
        qd_uniform U(seed);
        for (arma::uword k = 0; k < in.fc.n_elem; ++k)
            for (arma::uword l = 0; l < in.f.lsp.n_rows; ++l)
                in.f.lsp(l, k, 7) = U(3.0, 30.0);
    }
}

TEST_CASE("ssf_rays - quadriga matches the algorithm at one and several frequencies")
{
    // UMi NLOS at 3.5 GHz; UMa NLOS (c_DS depends on the frequency) at 3.5 and 28 GHz with the ground reflection
    QD_DIAG("\n# diag: quadriga against the transcription (max relative delay/power error, angle, Jones)\n");
    QD_DIAG("angle_model,pol_model,n_freq,link,d_rel,d_ang,d_J\n");
    struct cfg_t
    {
        const char *name;
        arma::uword F;
        const char *angle, *pol;
        bool var, rp;
    };
    const cfg_t cfg[6] = {{"UMi_NLOS", 1, "additive", "38.901", false, true},
                          {"UMi_NLOS", 1, "rotation", "quadriga", false, true},
                          {"UMa_NLOS", 2, "additive", "38.901", true, true},
                          {"UMa_NLOS", 2, "rotation", "quadriga", false, true},
                          {"UMa_NLOS", 2, "rotation", "38.901", false, false},
                          {"UMa_NLOS", 2, "additive", "quadriga", false, false}};
    unsigned seed = 201;
    for (const cfg_t &c : cfg)
    {
        rays_in_t in = make_rays_in(scen(c.name, c.F == 2), 5, "quadriga", seed++, c.F);
        in.angle_model = c.angle, in.pol_model = c.pol, in.pol_variability = c.var, in.random_phase = c.rp;
        set_xpr(in, seed++);
        const auto rays = call(in);
        INFO(c.name << ", " << c.F << " frequencies, " << c.angle << ", " << c.pol << ", eta " << c.var);
        REQUIRE(rays.size() == 5);
        for (arma::uword l = 0; l < 5; ++l)
        {
            CHECK(arma::all(rays[l].center_frequency == in.fc));
            check_qd(rays[l], in, l);
        }
    }
}

TEST_CASE("ssf_rays - quadriga ray delays are zero-mean per cluster and clamped at 0")
{
    // UMi NLOS: c_DS = 11 ns, offsets 2·c_DS·u in (0.44, 21.56) ns before the mean is removed; cluster 0 of each link
    // moved to 2 ns, so that some of its rays fall below 0 and are clamped
    rays_in_t in = make_rays_in(scen("UMi_NLOS"), 4, "quadriga", 211);
    for (arma::uword l = 0; l < 4; ++l)
        in.cl.delay(l, 1, 0) = 2e-9;
    const auto rays = call(in);
    const arma::uword N = in.N, M = in.M, NM = N * M;
    arma::uword n_clamped = 0, n_free = 0;
    for (arma::uword l = 0; l < 4; ++l)
        for (arma::uword n = 0; n < N; ++n)
        {
            INFO("link " << l << ", cluster " << n);
            const double tau_s = in.cl.delay(l, 1 + n, 0);
            double u_mean = 0.0, off_mean = 0.0;
            for (arma::uword m = 0; m < M; ++m)
                u_mean += in.rnd(l, 13 * NM + N + n * M + m, 0) / (double)M;
            bool clamped = false;
            for (arma::uword m = 0; m < M; ++m)
            {
                const arma::uword i = 1 + n * M + m;
                const double t = tau_s + 2.0 * 11e-9 * (in.rnd(l, 13 * NM + N + n * M + m, 0) - u_mean);
                CHECK(rays[l].tap(i) == 3 * (1 + n));
                if (t < 0.0)
                {
                    CHECK(rays[l].delay(i) == 0.0);
                    clamped = true, ++n_clamped;
                }
                off_mean += (rays[l].delay(i) - tau_s) / (double)M;
            }
            if (!clamped)
            {
                CHECK(std::abs(off_mean) < 1e-18);
                ++n_free;
            }
        }
    CHECK(n_clamped > 0);
    CHECK(n_free > 0);

    // M = 1: the cluster delay; N = 0: the LOS and GR rays at both frequencies
    spmap sp = scen("UMi_NLOS");
    sp["NumSubPaths"] = 1.0;
    const rays_in_t in1 = make_rays_in(sp, 2, "quadriga", 212);
    const auto r1 = call(in1);
    for (arma::uword l = 0; l < 2; ++l)
        for (arma::uword n = 0; n < in1.N; ++n)
            CHECK(r1[l].delay(1 + n) == in1.cl.delay(l, 1 + n, 0));
    sp = scen("UMa_LOS", true);
    sp["NumClusters"] = sp["NumClusters_min"] = 1.0;
    const rays_in_t in0 = make_rays_in(sp, 2, "quadriga", 213, 2);
    const auto r0 = call(in0);
    for (arma::uword l = 0; l < 2; ++l)
    {
        CHECK(r0[l].is_valid() == "");
        CHECK(arma::size(r0[l].M) == arma::size(8, 2, 2));
        CHECK(arma::all(arma::vectorise(r0[l].power) == rv(in0.cl.power, l)));
    }
}

TEST_CASE("ssf_rays - quadriga shares the geometry and the phases across frequencies")
{
    // UMa NLOS with the ground reflection at 3.5 and 28 GHz
    rays_in_t in = make_rays_in(scen("UMa_NLOS", true), 4, "quadriga", 221, 2);
    set_xpr(in, 222);
    for (const char *pol : {"38.901", "quadriga"})
        for (bool var : {false, true})
        {
            if (var && std::string(pol) == "quadriga")
                continue;
            INFO("pol_model " << pol << ", eta " << var);
            in.pol_model = pol, in.pol_variability = var;
            const auto rays = call(in);

            // Inputs of frequency 1 equal to those of frequency 0: equal powers and Jones matrices
            rays_in_t eq = in;
            for (arma::cube *c : {&eq.cl.power, &eq.f.lsp, &eq.st.mu, &eq.st.sigma})
                if (c == &eq.cl.power)
                    c->slice(1) = c->slice(0);
                else
                    c->col(1) = c->col(0);
            eq.f.eps_re.col(1) = eq.f.eps_re.col(0), eq.f.eps_im.col(1) = eq.f.eps_im.col(0);
            const auto r_eq = call(eq);

            // Other powers, XPR and permittivity at frequency 1: the geometry and frequency 0 are unchanged
            rays_in_t ot = in;
            ot.cl.power.slice(1) *= 0.5;
            for (arma::uword l = 0; l < 4; ++l)
                ot.f.lsp(l, 1, 7) *= 2.0, ot.st.mu(l, 1, 7) += 1.0, ot.st.sigma(l, 1, 7) += 0.5;
            ot.f.eps_re.col(1) += 1.0;
            const auto r_ot = call(ot);

            for (arma::uword l = 0; l < 4; ++l)
            {
                INFO("link " << l);
                CHECK(arma::all(arma::vectorise(r_eq[l].M.slice(1) == r_eq[l].M.slice(0))));
                CHECK(arma::all(r_eq[l].power.col(1) == r_eq[l].power.col(0)));
                const quadriga_lib::gbsm_rays &a = rays[l], &b = r_ot[l];
                CHECK(arma::all(a.delay == b.delay));
                CHECK((arma::all(a.aod == b.aod) && arma::all(a.eod == b.eod)));
                CHECK((arma::all(a.aoa == b.aoa) && arma::all(a.eoa == b.eoa)));
                CHECK(arma::all(a.tap == b.tap));
                CHECK(arma::all(arma::vectorise(a.M.slice(0) == b.M.slice(0))));
                CHECK(arma::all(a.power.col(0) == b.power.col(0)));
                CHECK(arma::any(arma::vectorise(a.M.slice(1) != b.M.slice(1))));
                if (std::string(pol) == "38.901" && !var) // tt carries only the shared initial phase
                    for (arma::uword i = 2; i < a.n_ray(); ++i)
                        CHECK(jones(a, 0, i, 0) == jones(a, 0, i, 1));
            }
        }
}

TEST_CASE("ssf_rays - rotation angle model equals additive at the horizon and keeps the on-sphere distances")
{
    // Cluster elevations 0: both models equal. Cluster elevations 0.7 and -1.2 rad: the rotation model keeps the
    // great-circle distance of each ray to its cluster centre (that of elevation 0), the additive model does not
    for (const char *method : {"38.901", "quadriga"})
    {
        INFO("method " << method);
        rays_in_t in = make_rays_in(scen("UMi_NLOS"), 3, method, 231);
        auto with_el = [&](double el, const char *model)
        {
            rays_in_t x = in;
            x.cl.eod.fill(el), x.cl.eoa.fill(-el), x.angle_model = model;
            return call(x);
        };
        auto dist = [&](const quadriga_lib::gbsm_rays &r, arma::uword l, arma::uword i, bool arrival, double el)
        {
            const arma::uword s = 1 + (i - 1) / in.M;
            return arrival ? ang_dist(unit_vec(r.aoa(i), r.eoa(i)), unit_vec(in.cl.aoa(l, s, 0), -el))
                           : ang_dist(unit_vec(r.aod(i), r.eod(i)), unit_vec(in.cl.aod(l, s, 0), el));
        };
        const auto add0 = with_el(0.0, "additive"), rot0 = with_el(0.0, "rotation");
        for (arma::uword l = 0; l < 3; ++l)
        {
            CHECK(arma::approx_equal(add0[l].eod, rot0[l].eod, "absdiff", 1e-12));
            CHECK(arma::approx_equal(add0[l].eoa, rot0[l].eoa, "absdiff", 1e-12));
            for (arma::uword i = 0; i < add0[l].n_ray(); ++i)
            {
                CHECK(std::abs(wrap_rad(add0[l].aod(i) - rot0[l].aod(i))) < 1e-12);
                CHECK(std::abs(wrap_rad(add0[l].aoa(i) - rot0[l].aoa(i))) < 1e-12);
            }
        }
        for (double el : {0.7, -1.2})
        {
            const auto rot = with_el(el, "rotation"), add = with_el(el, "additive");
            double d_add = 0.0;
            for (arma::uword l = 0; l < 3; ++l)
                for (arma::uword i = 1; i < rot[l].n_ray(); ++i) // the NLOS rays
                    for (bool arr : {false, true})
                    {
                        INFO("elevation " << el << ", link " << l << ", ray " << i << ", arrival " << arr);
                        const double d0 = dist(rot0[l], l, i, arr, 0.0);
                        CHECK(std::abs(dist(rot[l], l, i, arr, el) - d0) < 1e-12);
                        d_add = std::max(d_add, std::abs(dist(add[l], l, i, arr, el) - d0));
                    }
            CHECK(d_add > 1e-3);
        }
    }
}

TEST_CASE("ssf_rays - rotation polarisation model is QuaDRiGa-NG's matrix with the owner-end rule")
{
    // One link per position pair, Rx = Tx + (dz, dx, dy); the owner end has the larger z, then x, then y, with ties
    // within 1 mm, and is the Tx if all three tie. NLOS rays: J = e^(j·Phi)·[[c·e^(-j·kappa), s·e^(j·kappa)],
    // [s·e^(-j·kappa), -c·e^(j·kappa)]] with the Tx as owner, the transpose with the Rx: so the entry
    // conj(J_tt)·J_pt (Tx owner) or conj(J_tt)·J_tp (Rx owner) is real, the other one is not
    struct pos_t
    {
        double dz, dx, dy;
        bool rx_owner;
    };
    const pos_t pos[12] = {{5.0, -3.0, 2.0, true},          // z decides
                           {-5.0, 3.0, -2.0, false},        //
                           {0.0009, 3.0, -7.0, true},       // z ties, x decides
                           {-0.0009, -3.0, 7.0, false},     //
                           {0.0009, -0.0009, 3.0, true},    // z and x tie, y decides
                           {-0.0009, 0.0009, -3.0, false},  //
                           {0.0009, -0.0009, 0.0009, false}, // all tie: Tx
                           {0.0, 0.0, 0.0, false},          //
                           {0.0011, -3.0, -3.0, true},      // 1.1 mm: z decides
                           {-0.0011, 3.0, 3.0, false},      //
                           {0.0009, -3.0, 0.0, false},      // 0.9 mm: z ties, x decides
                           {-0.0009, 3.0, 0.0, true}};
    rays_in_t in = make_rays_in(scen("UMi_NLOS"), 12, "quadriga", 241);
    in.pol_model = "quadriga";
    set_xpr(in, 242);
    for (arma::uword l = 0; l < 12; ++l)
    {
        in.f.tx(l, 0) = 10.0, in.f.tx(l, 1) = 20.0, in.f.tx(l, 2) = 25.0;
        in.f.rx(l, 0) = 10.0 + pos[l].dx, in.f.rx(l, 1) = 20.0 + pos[l].dy, in.f.rx(l, 2) = 25.0 + pos[l].dz;
    }
    const auto rays = call(in);
    for (arma::uword l = 0; l < 12; ++l)
    {
        INFO("link " << l);
        check_qd(rays[l], in, l);
        double im_own = 0.0, im_other = 1.0, unit = 0.0;
        for (arma::uword i = 1; i < rays[l].n_ray(); ++i)
        {
            const cx tt = jones(rays[l], 0, i, 0), pt = jones(rays[l], 1, i, 0), tp = jones(rays[l], 2, i, 0);
            const cx pp = jones(rays[l], 3, i, 0);
            const cx own = std::conj(tt) * (pos[l].rx_owner ? tp : pt);
            const cx other = std::conj(tt) * (pos[l].rx_owner ? pt : tp);
            im_own = std::max(im_own, std::abs(own.imag()));
            im_other = std::min(im_other, std::abs(other.imag()));
            // Unitarity: J·J^H = I
            unit = std::max({unit, std::abs(std::norm(tt) + std::norm(tp) - 1.0),
                             std::abs(std::norm(pt) + std::norm(pp) - 1.0),
                             std::abs(tt * std::conj(pt) + tp * std::conj(pp))});
        }
        CHECK(im_own < 1e-12);
        CHECK(im_other > 1e-3);
        CHECK(unit < 1e-12);
    }
}

TEST_CASE("ssf_rays - polarisation variability follows TR 7.6.16")
{
    // UMi NLOS with the ground reflection (PolVar_sigma = 3 dB). NLOS rays: each entry is sqrt(eta_b) times the
    // entry without variability, eta_b = eta'_b·(2 + 2/kappa)/(eta'_tt + eta'_pp + (eta'_tp + eta'_pt)/kappa) with
    // eta'_b = 10^(PolVar_sigma·x_b/10), x_b of blocks 9 (tt), 10 (tp), 11 (pt), 12 (pp); kappa = 10^(X/10),
    // X = mu + sigma·x_4 (eq. (7.5-21)); squared Frobenius norm 2 + 2/kappa. LOS and GR rays unchanged
    for (const char *method : {"38.901", "38.901_MF"})
    {
        const bool mf = std::string(method) == "38.901_MF";
        rays_in_t in = make_rays_in(scen("UMi_NLOS", true), 3, method, 251, mf ? 2 : 1);
        in.i_freq = mf ? 1 : 0;
        const arma::uword k = in.i_freq, N = in.N, M = in.M, NM = N * M, G = in.G;
        const double s_pv = in.sp.at("PolVar_sigma");
        const auto r0 = call(in);
        in.pol_variability = true;
        const auto r1 = call(in);
        double eta_dev = 0.0;
        for (arma::uword l = 0; l < 3; ++l)
        {
            INFO("method " << method << ", link " << l);
            for (arma::uword i = 0; i < 1 + G; ++i)
                CHECK(arma::all(r1[l].M.slice(0).col(i) == r0[l].M.slice(0).col(i)));
            double d_J = 0.0, d_fro = 0.0, d_pow = 0.0;
            for (arma::uword n = 0; n < N; ++n)
                for (arma::uword m = 0; m < M; ++m)
                {
                    const arma::uword i = 1 + G + n * M + m;
                    auto x = [&](arma::uword b) { return in.rnd(l, b * NM + n * M + m, k); };
                    const double kappa = std::pow(10.0, (in.st.mu(l, k, 7) + in.st.sigma(l, k, 7) * x(4)) / 10.0);
                    double ep[4], eta[4]; // tt, tp, pt, pp
                    for (int b = 0; b < 4; ++b)
                        ep[b] = std::pow(10.0, s_pv * x(9 + b) / 10.0);
                    const double nrm = (2.0 + 2.0 / kappa) / (ep[0] + ep[3] + (ep[1] + ep[2]) / kappa);
                    for (int b = 0; b < 4; ++b)
                        eta[b] = ep[b] * nrm, eta_dev = std::max(eta_dev, std::abs(eta[b] - 1.0));
                    const arma::uword e8[4] = {0, 2, 1, 3}; // layout entry of tt, tp, pt, pp
                    double fro = 0.0;
                    for (int b = 0; b < 4; ++b)
                    {
                        const cx j1 = jones(r1[l], e8[b], i, 0), j0 = jones(r0[l], e8[b], i, 0);
                        d_J = std::max(d_J, std::abs(j1 - std::sqrt(eta[b]) * j0));
                        d_pow = std::max(d_pow, std::abs(std::norm(j1) * (b == 1 || b == 2 ? kappa : 1.0) - eta[b]));
                        fro += std::norm(j1);
                    }
                    d_fro = std::max(d_fro, std::abs(fro - (2.0 + 2.0 / kappa)));
                }
            CHECK(d_J < 1e-12);
            CHECK(d_pow < 1e-12);
            CHECK(d_fro < 1e-12);
        }
        CHECK(eta_dev > 0.1);

        // PolVar_sigma = 0: eta = 1
        in.sp["PolVar_sigma"] = 0.0;
        const auto r2 = call(in);
        for (arma::uword l = 0; l < 3; ++l)
            CHECK(arma::approx_equal(r2[l].M, r0[l].M, "absdiff", 1e-12));
    }
}

TEST_CASE("ssf_rays - both polarisation models are swap-reciprocal for mirrored inputs")
{
    // UMi NLOS with the ground reflection and equal cluster spreads at both ends; the cluster and ray sets drawn with
    // dual mobility at the swapped positions, the LSPs mirrored. Swapping Tx and Rx exchanges departure and arrival
    // per ray, keeps delays and powers and transposes M. Link 0 has equal heights (the owner end follows from x)
    spmap sp = read_v19("UMi_NLOS");
    sp["GR_enabled"] = 1.0;
    sp["PerClusterAS_D"] = sp["PerClusterAS_A"] = 15.0;
    sp["PerClusterES_D"] = sp["PerClusterES_A"] = 5.0;
    sp["PerClusterES_D_scale"] = 0.0;
    const arma::vec fc = {3.5e9};
    for (const char *method : {"38.901", "quadriga"})
    {
        const bool qd = std::string(method) == "quadriga";
        fix_t a = make_fix(N_UMi_NLOS, 6, 1, 261, true);
        a.use_eps = true, a.use_toa = qd;
        a.rx(0, 2) = a.tx(0, 2);
        fix_t b = swapped(a);
        arma::cube ph, ra, rb;
        arma::mat x;
        quadriga_lib::gbsm_init_phases(sp, "cluster", ph, 23);
        quadriga_lib::gbsm_draw(sp, "cluster", ph, a.tx, a.rx, a.rnd, -1, "Comb256", nullptr, nullptr, true);
        quadriga_lib::gbsm_draw(sp, "cluster", ph, b.tx, b.rx, b.rnd, -1, "Comb256", nullptr, nullptr, true);
        quadriga_lib::gbsm_init_phases(sp, "cluster", ph, 23, 0);
        quadriga_lib::gbsm_draw(sp, "cluster", ph, a.tx, a.rx, x, 0, "Comb256", nullptr, nullptr, true);
        a.rnd_freq.slice(0) = x;
        quadriga_lib::gbsm_draw(sp, "cluster", ph, b.tx, b.rx, x, 0, "Comb256", nullptr, nullptr, true);
        b.rnd_freq.slice(0) = x;
        quadriga_lib::gbsm_init_phases(sp, "ray", ph, 23, qd ? -1 : 0);
        quadriga_lib::gbsm_draw(sp, "ray", ph, a.tx, a.rx, x, qd ? -1 : 0, "Comb256", nullptr, &fc, true);
        ra = arma::cube(x.memptr(), x.n_rows, x.n_cols, 1);
        quadriga_lib::gbsm_draw(sp, "ray", ph, b.tx, b.rx, x, qd ? -1 : 0, "Comb256", nullptr, &fc, true);
        rb = arma::cube(x.memptr(), x.n_rows, x.n_cols, 1);
        const out_t ca = run(sp, a, method), cb = run(sp, b, method);
        lsp_stats_t sa = make_lsp_stats(a, 262), sb = sa;
        for (arma::cube *c : {&sb.mu, &sb.sigma})
        {
            const arma::cube &o = (c == &sb.mu) ? sa.mu : sa.sigma;
            c->slice(3) = o.slice(4), c->slice(4) = o.slice(3), c->slice(5) = o.slice(6), c->slice(6) = o.slice(5);
        }
        for (const char *angle : {"additive", "rotation"})
            for (int p = 0; p < 3; ++p) // TR model, TR model with eta, rotation model
            {
                const std::string pol = p < 2 ? "38.901" : "quadriga";
                INFO(method << ", " << angle << ", " << pol << ", eta " << (p == 1));
                std::vector<quadriga_lib::gbsm_rays> A, B;
                quadriga_lib::ssf_rays(sp, a.tx, a.rx, fc, a.lsp, sa.mu, sa.sigma, ca.delay, ca.power, ca.aod, ca.eod,
                                       ca.aoa, ca.eoa, ra, A, method, 0, angle, pol, &a.eps_re, &a.eps_im, p == 1);
                quadriga_lib::ssf_rays(sp, b.tx, b.rx, fc, b.lsp, sb.mu, sb.sigma, cb.delay, cb.power, cb.aod, cb.eod,
                                       cb.aoa, cb.eoa, rb, B, method, 0, angle, pol, &b.eps_re, &b.eps_im, p == 1);
                double d_ang = 0.0, d_rel = 0.0, d_M = 0.0;
                for (arma::uword l = 0; l < 6; ++l)
                    for (arma::uword i = 0; i < A[l].n_ray(); ++i)
                    {
                        d_rel = std::max(d_rel, std::abs(A[l].delay(i) - B[l].delay(i)) /
                                                    std::max(A[l].delay(i), 1e-9));
                        d_rel = std::max(d_rel, std::abs(A[l].power(i, 0) - B[l].power(i, 0)) /
                                                    std::max(A[l].power(i, 0), 1e-300));
                        d_ang = std::max({d_ang, std::abs(wrap_rad(A[l].aod(i) - B[l].aoa(i))),
                                          std::abs(wrap_rad(A[l].aoa(i) - B[l].aod(i))),
                                          std::abs(A[l].eod(i) - B[l].eoa(i)), std::abs(A[l].eoa(i) - B[l].eod(i))});
                        const arma::uword tr[4] = {0, 2, 1, 3}; // layout entry of the transpose
                        for (arma::uword e = 0; e < 4; ++e)
                            d_M = std::max(d_M, std::abs(jones(B[l], e, i, 0) - jones(A[l], tr[e], i, 0)));
                    }
                QD_DIAG("# diag: swap %s %s %s eta %d: delay/power %.2e, angle %.2e rad, M %.2e\n", method, angle,
                        pol.c_str(), p == 1, d_rel, d_ang, d_M);
                CHECK(d_rel < 1e-12);
                CHECK(d_ang < 1e-10);
                CHECK(d_M < 1e-12);
            }
    }
}

TEST_CASE("ssf_rays - quadriga input validation")
{
    // UMa NLOS with the ground reflection at two frequencies; every frequency is used
    rays_in_t base = make_rays_in(scen("UMa_NLOS", true), 3, "quadriga", 271, 2);
    const arma::uword NM = base.N * base.M, b14 = 13 * NM + base.N;
    const std::vector<std::pair<void (*)(rays_in_t &), std::string>> bad = {
        {[](rays_in_t &x) { x.i_freq = 1; }, "Input 'i_freq' must select the first frequency for method 'quadriga'."},
        {[](rays_in_t &x) { x.pol_model = "quadriga", x.pol_variability = true; },
         "Input 'pol_variability' needs pol_model '38.901'."},
        {[](rays_in_t &x) { x.pol_model = "quadriga", x.f.lsp(2, 1, 7) = 0.0; }, "Input 'lsp' must be finite and > 0."},
        {[](rays_in_t &x) { x.pol_model = "quadriga", x.f.lsp(0, 0, 7) = NAN; },
         "Input 'lsp' must be finite and > 0."}};
    for (size_t i = 0; i < bad.size(); ++i)
    {
        rays_in_t x = base;
        bad[i].first(x);
        INFO("case " << i);
        CHECK_THROWS_MATCHES(call(x), std::invalid_argument, Message(bad[i].second));
    }
    for (double v : {-1e-12, 1.0 + 1e-12})
    {
        rays_in_t x = base;
        x.rnd(1, b14 + 5, 0) = v;
        CHECK_THROWS_MATCHES(call(x), std::invalid_argument,
                             Message("Values of block 14 of 'rnd_ray' must be in [0, 1]."));
    }

    // Accepted: block 14 at 0 and 1; the XPR LSP at 0 with the TR model; pol_variability with the TR model
    rays_in_t x = base;
    x.rnd(0, b14, 0) = 0.0, x.rnd(2, x.rnd.n_cols - 1, 0) = 1.0, x.f.lsp(1, 1, 7) = 0.0, x.pol_variability = true;
    CHECK_NOTHROW(call(x));
}

TEST_CASE("Diagnostic - ssf_rays quadriga ray delay spread relative to the slot delay spread")
{
    // Library drop: UMi NLOS at 3.5 GHz, 200 UTs; LSPs, clusters and the shared "ray" set from the library
    const spmap sp = read_v19("UMi_NLOS");
    const arma::mat tx = {{0.0, 0.0, 10.0}}, rx = drop_rx(281, 200);
    const arma::vec fc = {3.5e9};
    const lib_t in = lib_inputs(sp, tx, rx, 282, 3.5e9);
    out_t o;
    quadriga_lib::ssf_clusters(sp, tx, rx, in.lsp, in.rnd, o.delay, o.power, o.aod, o.eod, o.aoa, o.eoa, "quadriga",
                               nullptr, nullptr, &in.eod, &in.toa);
    arma::cube ph;
    arma::mat x;
    quadriga_lib::gbsm_init_phases(sp, "ray", ph, 282);
    quadriga_lib::gbsm_draw(sp, "ray", ph, tx, rx, x, -1, "Comb256", nullptr, &fc);
    const arma::cube rnd(x.memptr(), x.n_rows, x.n_cols, 1);
    arma::cube mu(200, 1, 8, arma::fill::value(1.0)), sigma(200, 1, 8, arma::fill::value(3.0));
    mu.slice(7).fill(9.0);
    std::vector<quadriga_lib::gbsm_rays> rays;
    quadriga_lib::ssf_rays(sp, tx, rx, fc, in.lsp, mu, sigma, o.delay, o.power, o.aod, o.eod, o.aoa, o.eoa, rnd, rays,
                           "quadriga");
    std::vector<arma::vec> d_slot(200), p_slot(200), d_ray(200), p_ray(200);
    for (arma::uword l = 0; l < 200; ++l)
        d_slot[l] = rv(o.delay, l), p_slot[l] = rv(o.power, l), d_ray[l] = rays[l].delay, p_ray[l] = rays[l].power;
    const arma::vec ratio =
        quadriga_lib::calc_delay_spread(d_ray, p_ray) / quadriga_lib::calc_delay_spread(d_slot, p_slot);
    const arma::vec q = arma::quantile(ratio, arma::vec{0.5, 0.9});
    QD_DIAG("\n# diag: quadriga ray DS / slot DS over 200 links: p50 %.4f, p90 %.4f\n", q(0), q(1));
    CHECK(ratio.is_finite());
}
