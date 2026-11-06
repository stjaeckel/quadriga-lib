// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Fixtures and oracles of the ray-level tests (gbsm_rays, ssf_rays, ssf_paths, gbsm_rays_from_paths,
// qrt_paths_from_scatterers, gbsm_single_link_channel), independent of the functions under test:
//  - make_rnd_ray: a "ray" set cube with distinct values, so that ssf_rays tests need no gbsm_draw; columns
//    b·N·M + n·M + m for blocks 0 to 12, 13·N·M + n for block 13, 13·N·M + N + n·M + m for block 14
//  - make_lsp_stats: lsp_mu, lsp_sigma with distinct finite values for a fixture of qd_test_ssf.hpp
//  - tr_offsets: the ray offsets of TR 38.901 Table 7.5-3 per unit cluster spread, extended to M != 20, and the
//    sub-cluster groups of Table 7.5-5
//  - geometry: unit_vec, ang_dist, polyline, specular_point, fresnel (TR eqs. (7.6-38) to (7.6-40))
//  - make_rays: a hand-made gbsm_rays object of one link (Tx (3, -4, 25), Rx (120, 45, 1.5)), so that the consumers
//    of ray data need no ssf_rays; with_scatterers builds the NLOS rays from hand-made scatterers
//  - ref_rays_tr, ref_cds: the expected rays of ssf_rays for the TR methods, typed from TR 38.901 clause 7.5
//  - rays_in_t, make_rays_in, call: the inputs of one ssf_rays call on make_fix, and the call; same: equality of ray
//    objects
//  - ref_rays_qd: the expected rays of ssf_rays for the method "quadriga" (QuaDRiGa ray delays, angle and
//    polarisation models; TR 38.901 clause 7.6.16)
// Units: positions in [m], angles in [rad], delays in [s], frequencies in [Hz].

#ifndef QD_TEST_RAYS_HPP
#define QD_TEST_RAYS_HPP

#include "qd_test_ssf.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <string>
#include <vector>

namespace
{
    // Uniform values in (a, b) from a 32-bit generator, as make_fix draws them
    struct qd_uniform
    {
        std::mt19937 g;
        explicit qd_uniform(unsigned seed) : g(seed) {}
        double operator()(double a, double b) { return a + (b - a) * ((double)g() + 0.5) / 4294967296.0; }
    };

    // "ray" set values [n_link, 14·N·M + N, n_set]: blocks 0 to 13 Normal-like in (-2.5, 2.5), block 14 (Uniform
    // field) in (0.02, 0.98); all values distinct
    arma::cube make_rnd_ray(arma::uword n_link, arma::uword N, arma::uword M, arma::uword n_set, unsigned seed)
    {
        qd_uniform U(seed);
        const arma::uword n_gen = 14 * N * M + N, b14 = 13 * N * M + N;
        arma::cube r(n_link, n_gen, n_set);
        for (arma::uword k = 0; k < n_set; ++k)
            for (arma::uword c = 0; c < n_gen; ++c)
                for (arma::uword l = 0; l < n_link; ++l)
                    r(l, c, k) = (c < b14) ? U(-2.5, 2.5) : U(0.02, 0.98);
        return r;
    }

    // Log-domain means and std [n_link, n_freq, 8] (sizes of fix.lsp): slice 5 (ESD) mean in (0.2, 1.2) lg(deg),
    // slice 7 (XPR) mean in (5, 12) dB and std in (2, 4) dB; other means in (-1, 1), other std in (0.1, 0.5)
    struct lsp_stats_t
    {
        arma::cube mu, sigma;
    };
    lsp_stats_t make_lsp_stats(const fix_t &fix, unsigned seed)
    {
        qd_uniform U(seed);
        lsp_stats_t s;
        s.mu.set_size(arma::size(fix.lsp)), s.sigma.set_size(arma::size(fix.lsp));
        for (arma::uword i = 0; i < 8; ++i)
            for (arma::uword e = 0; e < s.mu.n_rows * s.mu.n_cols; ++e)
            {
                s.mu.slice(i)(e) = (i == 5) ? U(0.2, 1.2) : (i == 7) ? U(5.0, 12.0) : U(-1.0, 1.0);
                s.sigma.slice(i)(e) = (i == 7) ? U(2.0, 4.0) : U(0.1, 0.5);
            }
        return s;
    }

    // Ray offsets per unit cluster spread (TR 38.901 Table 7.5-3 and its extension to M != 20), ascending
    // - M = 20: the 20 values of Table 7.5-3 (TR ray 2k - 1 = +a_k, ray 2k = -a_k)
    // - M < 20: o = the first M values of the reordered table of QuaDRiGa-NG, alpha = (o - mean(o)) / rms(o - mean(o));
    //   M = 1 gives 0
    // - M > 20: Laplacian quantiles q_k = -ln(2·(1 - j_k / (M + 1))), j_k = M - K + k, k = 1 ... K = floor(M / 2);
    //   the set {0 if M is odd, +-q_k} scaled to the rms of Table 7.5-3
    // - groups (M = 20 only): the offsets of the sub-clusters of Table 7.5-5 at the ray positions 0-9, 10-15, 16-19:
    //   TR rays (1-8, 19, 20), (9-12, 17, 18), (13-16), each ascending
    // - rms_table: rms of the 20 values of Table 7.5-3, computed in double
    struct offsets_t
    {
        std::vector<double> asc;
        std::vector<std::vector<double>> groups;
        double rms_table = 0.0;
    };
    offsets_t tr_offsets(arma::uword M)
    {
        const double a[10] = {0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195, 2.1551};
        const double ng[20] = {-0.8844, 1.1481, 0.6797, -1.1481, -0.6797, 1.5195, -0.0447, 0.3715, -1.5195, -0.2492,
                               -0.5129, 0.5129, 0.8844, -2.1551, -0.1413, 0.1413, 0.0447, 2.1551, -0.3715, 0.2492};
        offsets_t o;
        double s2 = 0.0;
        for (int r = 0; r < 20; ++r) // TR ray order, left to right
            s2 += a[r / 2] * a[r / 2];
        o.rms_table = std::sqrt(s2 / 20.0);

        auto rms = [](const std::vector<double> &x)
        {
            double s = 0.0;
            for (double v : x)
                s += v * v;
            return std::sqrt(s / (double)x.size());
        };
        if (M == 20)
        {
            for (int k = 0; k < 10; ++k)
                o.asc.push_back(a[k]), o.asc.push_back(-a[k]);
            const int grp[3][5] = {{0, 1, 2, 3, 9}, {4, 5, 8, -1, -1}, {6, 7, -1, -1, -1}};
            for (int i = 0; i < 3; ++i)
            {
                std::vector<double> g;
                for (int k : grp[i])
                    if (k >= 0)
                        g.push_back(a[k]), g.push_back(-a[k]);
                std::sort(g.begin(), g.end());
                o.groups.push_back(g);
            }
        }
        else if (M < 20)
        {
            double mean = 0.0;
            for (arma::uword m = 0; m < M; ++m)
                mean += ng[m] / (double)M;
            for (arma::uword m = 0; m < M; ++m)
                o.asc.push_back(ng[m] - mean);
            const double r = rms(o.asc);
            for (double &v : o.asc)
                v = (r > 0.0) ? v / r : 0.0;
        }
        else
        {
            const arma::uword K = M / 2;
            if (M % 2 == 1)
                o.asc.push_back(0.0);
            for (arma::uword k = 1; k <= K; ++k)
            {
                const double q = -std::log(2.0 * (1.0 - (double)(M - K + k) / (double)(M + 1)));
                o.asc.push_back(q), o.asc.push_back(-q);
            }
            const double r = o.rms_table / rms(o.asc);
            for (double &v : o.asc)
                v *= r;
        }
        std::sort(o.asc.begin(), o.asc.end());
        return o;
    }

    // Unit vector of azimuth 'az' and elevation 'el' in [rad]
    arma::vec unit_vec(double az, double el)
    {
        return arma::vec{std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)};
    }

    // Angle between two vectors in [rad] as atan2(abs(a x b), a·b), accurate also for small angles
    double ang_dist(const arma::vec &a, const arma::vec &b)
    {
        const double cx = a(1) * b(2) - a(2) * b(1), cy = a(2) * b(0) - a(0) * b(2), cz = a(0) * b(1) - a(1) * b(0);
        return std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), a(0) * b(0) + a(1) * b(1) + a(2) * b(2));
    }

    // Length of a vector, without BLAS
    double len3(const arma::vec &x) { return std::sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2)); }

    // Length of the polyline T -> FBS -> LBS -> R in [m]
    double polyline(const arma::vec &T, const arma::vec &FBS, const arma::vec &LBS, const arma::vec &R)
    {
        return len3(FBS - T) + len3(LBS - FBS) + len3(R - LBS);
    }

    // Specular point of the ground reflection on z = 0 for ends at z >= 0: T + h_t / (h_t + h_r)·(R' - T) with
    // R' = (R_x, R_y, -R_z); both ends on the ground: the midpoint
    arma::vec specular_point(const arma::vec &T, const arma::vec &R)
    {
        const double h = T(2) + R(2);
        if (h <= 0.0)
            return 0.5 * (T + R);
        const arma::vec Ri = {R(0), R(1), -R(2)};
        arma::vec G = T + T(2) / h * (Ri - T);
        G(2) = 0.0;
        return G;
    }

    // Reflection coefficients of the ground (TR 38.901 eqs. (7.6-38) to (7.6-40), eps = eps_re - j·eps_im), the
    // oracle of the ssf_clusters tests
    void fresnel(double eps_re, double eps_im, double h_t, double h_r, double d2D, std::complex<double> &R_par,
                 std::complex<double> &R_perp)
    {
        ref_r(d2D, h_t, h_r, eps_re, eps_im, R_par, R_perp);
    }

    // Hand-made ray data of one link: Tx (3, -4, 25), Rx (120, 45, 1.5); frequencies 3.5 + 4.5·k GHz
    // - LOS ray (has_los): delay 0, geometric angles, M = diag(1, -1), tap 0
    // - GR ray (has_gr): geometric delay and angles towards the image points, M = diag(R_par, -R_perp) / R with
    //   R² = (abs(R_par)² + abs(R_perp)²) / 2, permittivity in gr_eps_re, gr_eps_im; tap 3
    // - NLOS ray j (tap 3·(1 + G + n), n = 0 for j = 0, 1, else j - 1): j = 0, 1 share one tap with powers 1 : 4 and
    //   unequal delays; j = 2 has power 0 at all frequencies; j = 3 has zero excess delay with departure angles off
    //   the direct line (without scatterers) or both scatterers at 2 m from their end (with scatterers); delays in
    //   10 to 500 ns, M entries in (-1, 1)
    // - with_scatterers: the NLOS rays from hand-made FBS (5 to 80 m from the Tx) and LBS (3 to 60 m from the Rx):
    //   delay = (polyline - d3D) / c, angles towards the scatterers; LOS at the midpoint, GR at the specular point;
    //   fallback empty
    quadriga_lib::gbsm_rays make_rays(unsigned seed, bool has_los, bool has_gr, arma::uword n_nlos, arma::uword n_freq,
                                      const std::string &method, bool with_scatterers = false)
    {
        qd_uniform U(seed);
        const arma::vec T = {3.0, -4.0, 25.0}, R = {120.0, 45.0, 1.5}, D = R - T;
        const double d2D = std::sqrt(D(0) * D(0) + D(1) * D(1)), d3D = len3(D);
        const arma::uword G = has_gr ? 1 : 0, n_ray = (has_los ? 1 : 0) + G + n_nlos;

        quadriga_lib::gbsm_rays r;
        r.method = method, r.tx_pos = T, r.rx_pos = R;
        r.center_frequency.set_size(n_freq);
        for (arma::uword k = 0; k < n_freq; ++k)
            r.center_frequency(k) = 3.5e9 + 4.5e9 * (double)k;
        r.delay.zeros(n_ray), r.power.zeros(n_ray, n_freq), r.M.zeros(8, n_ray, n_freq);
        r.aod.zeros(n_ray), r.eod.zeros(n_ray), r.aoa.zeros(n_ray), r.eoa.zeros(n_ray);
        r.path_type.zeros(n_ray), r.tap.zeros(n_ray);
        if (with_scatterers)
            r.fbs_pos.zeros(3, n_ray), r.lbs_pos.zeros(3, n_ray);

        // Angles of a ray from its directions; scatterers if requested
        auto set_ray = [&](arma::uword i, const arma::vec &dep, const arma::vec &arr, const arma::vec *fbs,
                           const arma::vec *lbs)
        {
            r.aod(i) = std::atan2(dep(1), dep(0)), r.eod(i) = std::atan2(dep(2), std::hypot(dep(0), dep(1)));
            r.aoa(i) = std::atan2(arr(1), arr(0)), r.eoa(i) = std::atan2(arr(2), std::hypot(arr(0), arr(1)));
            if (with_scatterers)
                r.fbs_pos.col(i) = *fbs, r.lbs_pos.col(i) = *lbs;
        };

        arma::uword i = 0;
        if (has_los)
        {
            const arma::vec mid = 0.5 * (T + R);
            set_ray(i, D, -D, &mid, &mid);
            for (arma::uword k = 0; k < n_freq; ++k)
                r.power(i, k) = U(0.5, 1.0), r.M(0, i, k) = 1.0, r.M(6, i, k) = -1.0;
            ++i;
        }
        if (has_gr)
        {
            const arma::vec Ri = {R(0), R(1), -R(2)}, Ti = {T(0), T(1), -T(2)}, sp = specular_point(T, R);
            r.delay(i) = (len3(Ri - T) - d3D) / c0;
            set_ray(i, Ri - T, Ti - R, &sp, &sp);
            r.path_type(i) = 1, r.tap(i) = 3;
            r.gr_eps_re.set_size(n_freq), r.gr_eps_im.set_size(n_freq);
            for (arma::uword k = 0; k < n_freq; ++k)
            {
                r.gr_eps_re(k) = U(3.0, 30.0), r.gr_eps_im(k) = U(0.1, 2.0), r.power(i, k) = U(0.05, 0.3);
                std::complex<double> R_par, R_perp;
                fresnel(r.gr_eps_re(k), r.gr_eps_im(k), T(2), R(2), d2D, R_par, R_perp);
                const double Rn = std::sqrt(0.5 * (std::norm(R_par) + std::norm(R_perp)));
                r.M(0, i, k) = R_par.real() / Rn, r.M(1, i, k) = R_par.imag() / Rn;
                r.M(6, i, k) = -R_perp.real() / Rn, r.M(7, i, k) = -R_perp.imag() / Rn;
            }
            ++i;
        }
        const arma::uword i0 = i; // first NLOS ray
        for (arma::uword j = 0; j < n_nlos; ++j, ++i)
        {
            r.path_type(i) = 2, r.tap(i) = 3 * (1 + G + (j == 0 ? 0 : j - 1));
            if (with_scatterers)
            {
                const double b_t = (j == 3) ? 2.0 : U(5.0, 80.0), b_r = (j == 3) ? 2.0 : U(3.0, 60.0);
                const arma::vec fbs = T + b_t * unit_vec(U(-pi, pi), U(-0.3, 0.5));
                const arma::vec lbs = R + b_r * unit_vec(U(-pi, pi), U(-0.02, 0.6));
                r.delay(i) = (polyline(T, fbs, lbs, R) - d3D) / c0;
                set_ray(i, fbs - T, lbs - R, &fbs, &lbs);
            }
            else
            {
                const double az_los = std::atan2(D(1), D(0));
                r.delay(i) = (j == 3) ? 0.0 : U(10e-9, 500e-9);
                r.aod(i) = (j == 3) ? wrap_rad(az_los + U(0.3, 1.5)) : U(-pi, pi), r.eod(i) = U(-0.5, 0.5);
                r.aoa(i) = U(-pi, pi), r.eoa(i) = U(-0.5, 0.5);
            }
            for (arma::uword k = 0; k < n_freq; ++k)
            {
                r.power(i, k) = (j == 2) ? 0.0 : (j == 1) ? 4.0 * r.power(i0, k) : U(0.01, 1.0);
                for (arma::uword e = 0; e < 8; ++e)
                    r.M(e, i, k) = U(-1.0, 1.0);
            }
        }
        return r;
    }

    // Expected ray data of one link of ssf_rays with the TR methods, typed from TR 38.901 clause 7.5 steps 7 to 11
    // (antenna-free part): additive angle model (eqs. (7.5-13), (7.5-18), (7.5-20), zenith wrap of step 7 for ZOA and
    // ZOD), polarisation of eqs. (7.5-21), (7.5-28) without clause 7.6.16, LOS eq. (7.5-29), ground reflection eqs.
    // (7.6-38) to (7.6-40)
    // - Inputs as ssf_rays takes them; frequency k (index into fc, lsp_mu, lsp_sigma, the power slices and the slices
    //   of rnd), geometry slice g = k for 38.901_MF (mf), else 0
    // - Split (Table 7.5-5, M = 20 only): the clusters with power > 0 at k that have fewer than two clusters of higher
    //   power or of equal power and lower index; ray positions 0-9, 10-15, 16-19 with delay + 0, 1.28, 2.56 c_DS
    // - Coupling: in each group of positions, the ray whose rank-block value is the k-th smallest gets the k-th
    //   smallest offset of the group (rank blocks 0 AOD, 1 AOA, 2 ZOD, 3 ZOA; equal values: lower position first)
    // - c_DS of Table 7.5-6 (3.91 ns where the table has no value), c_ZSD = PerClusterES_D_scale·10^mu_lgZSD
    //   (eq. (7.5-20)) or PerClusterES_D; initial phases 2·pi·Phi(x) - pi of a Normal value x, or 0
    // - ang[q] in rank-block order: AOD, AOA, EOD, EOA; J holds 4 entries per ray: tt, tp, pt, pp (rows Rx, columns Tx)
    struct ref_rays_t
    {
        std::vector<double> delay, power, ang[4];
        std::vector<std::complex<double>> J;
        std::vector<arma::uword> path_type, tap;
    };
    double ref_cds(const spmap &sp, double fc) // c_DS in [s], Table 7.5-6 with the keys of a completed scenpar
    {
        const double ds = sp.at("PerClusterDS"), f = std::max(fc / 1e9, sp.at("PerClusterDS_fc_min"));
        return ds == 0.0 ? 3.91e-9
                         : 1e-9 * std::max(sp.at("PerClusterDS_min"), ds + sp.at("PerClusterDS_gamma") * std::log10(f));
    }
    ref_rays_t ref_rays_tr(const spmap &scenpar, const fix_t &f, const out_t &cl, const lsp_stats_t &st,
                           const arma::cube &rnd, const arma::vec &fc, arma::uword l, arma::uword k, bool mf,
                           bool random_phase = true)
    {
        spmap sp = scenpar;
        quadriga_lib::scenpar_validate(sp);
        const arma::uword N = (arma::uword)sp.at("NumClusters") - 1, M = (arma::uword)sp.at("NumSubPaths");
        const arma::uword G = sp.at("GR_enabled") > 0.0 ? 1 : 0, g = mf ? k : 0, NM = N * M, n_ray = 1 + G + NM;
        const double c_ds = ref_cds(sp, fc(k)), d2r = pi / 180.0;
        const double c[4] = {sp.at("PerClusterAS_D"), sp.at("PerClusterAS_A"),
                             sp.at("PerClusterES_D_scale") > 0.0
                                 ? sp.at("PerClusterES_D_scale") * std::pow(10.0, st.mu(l, k, 5))
                                 : sp.at("PerClusterES_D"),
                             sp.at("PerClusterES_A")};
        const arma::cube *cang[4] = {&cl.aod, &cl.aoa, &cl.eod, &cl.eoa};
        const offsets_t off = tr_offsets(M);

        ref_rays_t r;
        r.delay.assign(n_ray, 0.0), r.power.assign(n_ray, 0.0), r.J.assign(4 * n_ray, 0.0);
        r.path_type.assign(n_ray, 2), r.tap.assign(n_ray, 0);
        for (auto &a : r.ang)
            a.assign(n_ray, 0.0);
        for (arma::uword s = 0; s < 1 + G; ++s) // LOS and GR rays: the slot values
        {
            r.delay[s] = cl.delay(l, s, g), r.power[s] = cl.power(l, s, k), r.path_type[s] = s, r.tap[s] = 3 * s;
            for (int q = 0; q < 4; ++q)
                r.ang[q][s] = (*cang[q])(l, s, g);
        }
        r.J[0] = 1.0, r.J[3] = -1.0;
        if (G)
        {
            const arma::uword t = (f.tx.n_rows == 1) ? 0 : l;
            const double dx = f.rx(l, 0) - f.tx(t, 0), dy = f.rx(l, 1) - f.tx(t, 1);
            std::complex<double> R_par, R_perp;
            fresnel(f.eps_re(l, k), f.eps_im(l, k), f.tx(t, 2), f.rx(l, 2), std::sqrt(dx * dx + dy * dy), R_par,
                    R_perp);
            const double R = std::sqrt(0.5 * (std::norm(R_par) + std::norm(R_perp)));
            r.J[4] = R > 0.0 ? R_par / R : 0.0, r.J[7] = R > 0.0 ? -R_perp / R : 0.0;
        }

        const double dsub[3] = {0.0, 1.28, 2.56};
        for (arma::uword n = 0; n < N; ++n)
        {
            const arma::uword s = 1 + G + n;
            const double P = cl.power(l, s, k);
            arma::uword above = 0;
            for (arma::uword o = 0; o < N; ++o)
                above += (cl.power(l, 1 + G + o, k) > P || (cl.power(l, 1 + G + o, k) == P && o < n)) ? 1 : 0;
            const bool split = M == 20 && P > 0.0 && above < 2;
            const arma::uword n_grp = split ? 3 : 1, first[3] = {0, 10, 16}, size[3] = {split ? 10 : M, 6, 4};
            for (arma::uword i = 0; i < n_grp; ++i)
                for (arma::uword m = first[i]; m < first[i] + size[i]; ++m)
                {
                    const arma::uword ray = 1 + G + n * M + m;
                    auto x = [&](arma::uword b, arma::uword pos) { return rnd(l, b * NM + n * M + pos, k); };
                    r.delay[ray] = cl.delay(l, s, g) + dsub[i] * c_ds;
                    r.power[ray] = P / (double)M, r.tap[ray] = 3 * s + i;
                    for (arma::uword q = 0; q < 4; ++q)
                    {
                        arma::uword rank = 0;
                        for (arma::uword o = first[i]; o < first[i] + size[i]; ++o)
                            rank += (x(q, o) < x(q, m) || (x(q, o) == x(q, m) && o < m)) ? 1 : 0;
                        const double off_deg = c[q] * (split ? off.groups[i][rank] : off.asc[rank]);
                        const double centre_deg = (*cang[q])(l, s, g) / d2r;
                        r.ang[q][ray] = q < 2 ? wrap_rad((centre_deg + off_deg) * d2r)
                                              : zen2elev(90.0 - centre_deg + off_deg);
                    }
                    auto phase = [&](arma::uword b)
                    { return random_phase ? 2.0 * pi * 0.5 * std::erfc(-x(b, m) / std::sqrt(2.0)) - pi : 0.0; };
                    const double X = st.mu(l, k, 7) + st.sigma(l, k, 7) * x(4, m), ik = std::pow(10.0, -X / 10.0);
                    r.J[4 * ray] = std::polar(1.0, phase(5)), r.J[4 * ray + 1] = std::polar(std::sqrt(ik), phase(6));
                    r.J[4 * ray + 2] = std::polar(std::sqrt(ik), phase(7));
                    r.J[4 * ray + 3] = std::polar(1.0, phase(8));
                }
        }
        return r;
    }

    // All inputs of one ssf_rays call: scenario, fixture, clusters of ssf_clusters, LSP statistics, "ray" values,
    // frequencies and options
    struct rays_in_t
    {
        spmap sp;
        fix_t f;
        out_t cl;
        lsp_stats_t st;
        arma::cube rnd;
        arma::vec fc;
        std::string method = "38.901", angle_model = "additive", pol_model = "38.901";
        arma::uword i_freq = 0, N = 0, M = 0, G = 0;
        bool pol_variability = false, random_phase = true;
    };

    // Inputs for n_link links with F frequencies (3.5 and 28 GHz); the ground reflection as set in sp; "ray" values
    // with F slices for the TR methods and one for "quadriga"
    rays_in_t make_rays_in(const spmap &sp, arma::uword n_link, const std::string &method, unsigned seed,
                           arma::uword F = 1)
    {
        rays_in_t in;
        in.sp = sp, in.method = method;
        spmap c = sp;
        quadriga_lib::scenpar_validate(c);
        in.N = (arma::uword)c.at("NumClusters") - 1, in.M = (arma::uword)c.at("NumSubPaths");
        in.G = c.at("GR_enabled") > 0.0 ? 1 : 0;
        in.f = make_fix(in.N, n_link, F, seed, true);
        in.f.use_eps = in.G == 1, in.f.use_toa = method == "quadriga"; // NLOS scenarios: absolute TOA for "quadriga"
        in.cl = run(sp, in.f, method);
        in.st = make_lsp_stats(in.f, seed + 1);
        in.rnd = make_rnd_ray(n_link, in.N, in.M, method == "quadriga" ? 1 : F, seed + 2);
        in.fc = (F == 1) ? arma::vec{3.5e9} : arma::vec{3.5e9, 28e9};
        return in;
    }

    void call(const rays_in_t &in, std::vector<quadriga_lib::gbsm_rays> &rays)
    {
        const bool e = in.f.use_eps;
        quadriga_lib::ssf_rays(in.sp, in.f.tx, in.f.rx, in.fc, in.f.lsp, in.st.mu, in.st.sigma, in.cl.delay,
                               in.cl.power, in.cl.aod, in.cl.eod, in.cl.aoa, in.cl.eoa, in.rnd, rays, in.method,
                               in.i_freq, in.angle_model, in.pol_model, e ? &in.f.eps_re : nullptr,
                               e ? &in.f.eps_im : nullptr, in.pol_variability, in.random_phase);
    }

    std::vector<quadriga_lib::gbsm_rays> call(const rays_in_t &in)
    {
        std::vector<quadriga_lib::gbsm_rays> rays;
        call(in, rays);
        return rays;
    }

    spmap scen(const char *name, bool gr = false)
    {
        spmap sp = read_v19(name);
        if (gr)
            sp["GR_enabled"] = 1.0;
        return sp;
    }

    // Exact equality of two ray objects without scatterers
    bool same(const quadriga_lib::gbsm_rays &a, const quadriga_lib::gbsm_rays &b)
    {
        auto eq = [](const auto &x, const auto &y)
        { return arma::size(x) == arma::size(y) && arma::all(arma::vectorise(x) == arma::vectorise(y)); };
        return a.method == b.method && eq(a.tx_pos, b.tx_pos) && eq(a.rx_pos, b.rx_pos) &&
               eq(a.center_frequency, b.center_frequency) && eq(a.delay, b.delay) && eq(a.power, b.power) &&
               eq(a.aod, b.aod) && eq(a.eod, b.eod) && eq(a.aoa, b.aoa) && eq(a.eoa, b.eoa) && eq(a.M, b.M) &&
               eq(a.path_type, b.path_type) && eq(a.tap, b.tap) && eq(a.gr_eps_re, b.gr_eps_re) &&
               eq(a.gr_eps_im, b.gr_eps_im) && a.fbs_pos.empty() && b.fbs_pos.empty() && a.lbs_pos.empty() &&
               b.lbs_pos.empty() && a.fallback.empty() && b.fallback.empty();
    }

    // Expected ray data of link l of ssf_rays with the method "quadriga", one entry per frequency of fc; delays,
    // angles, path types and taps are equal in all entries (one geometry for all frequencies)
    // - Coupling and offsets as ref_rays_tr without sub-clusters, from the one slice of rnd; tap 3·slot
    // - Delays: tau'' = 2·c_DS'·u with u of block 14 and c_DS' the mean of c_DS (Table 7.5-6) over fc; tau' = tau''
    //   minus its mean over the rays of the cluster; tau = max(tau_slot + tau', 0)
    // - c_ZSD: the mean of eq. (7.5-20) over fc
    // - angle_model "additive": as ref_rays_tr; "rotation": the direction of azimuth c_az·alpha_az and elevation
    //   -c_el·alpha_el, rotated about the y-axis by -el_c and then about the z-axis by phi_c (cluster angles)
    // - pol_model "38.901": eqs. (7.5-21), (7.5-28) with the XPR mean and std of each frequency; pol_variability:
    //   eta of clause 7.6.16 from blocks 9 to 12 (eqs. (7.6-61), (7.6-62), std PolVar_sigma)
    // - pol_model "quadriga" (rotation model of QuaDRiGa-NG): X = 10·lg(XPR LSP) + 0.1·sigma·x with x of block 4
    //   (per ray) or 13 (per cluster); gamma = atan(10^(-X_ray/20)), kappa = atan(10^(-X_cluster/20)); J = e^(j·Phi)·
    //   [[cos(gamma)·e^(-j·kappa), sin(gamma)·e^(j·kappa)], [sin(gamma)·e^(-j·kappa), -cos(gamma)·e^(j·kappa)]],
    //   transposed when the Rx is the owner end: the end of larger z, then x, then y, ties within 1 mm; Tx if all tie
    std::vector<ref_rays_t> ref_rays_qd(const spmap &scenpar, const fix_t &f, const out_t &cl, const lsp_stats_t &st,
                                        const arma::cube &rnd, const arma::vec &fc, arma::uword l,
                                        const std::string &angle_model, const std::string &pol_model,
                                        bool pol_variability, bool random_phase = true)
    {
        spmap sp = scenpar;
        quadriga_lib::scenpar_validate(sp);
        const arma::uword N = (arma::uword)sp.at("NumClusters") - 1, M = (arma::uword)sp.at("NumSubPaths");
        const arma::uword G = sp.at("GR_enabled") > 0.0 ? 1 : 0, NM = N * M, n_ray = 1 + G + NM, F = fc.n_elem;
        const arma::uword t = (f.tx.n_rows == 1) ? 0 : l;
        const double d2r = pi / 180.0, s_pv = sp.at("PolVar_sigma");
        double c_ds = 0.0, c_zsd = 0.0;
        for (arma::uword k = 0; k < F; ++k)
        {
            c_ds += ref_cds(sp, fc(k)) / (double)F;
            const double scale = sp.at("PerClusterES_D_scale");
            c_zsd += (scale > 0.0 ? scale * std::pow(10.0, st.mu(l, k, 5)) : sp.at("PerClusterES_D")) / (double)F;
        }
        const double c[4] = {sp.at("PerClusterAS_D"), sp.at("PerClusterAS_A"), c_zsd, sp.at("PerClusterES_A")};
        const arma::cube *cang[4] = {&cl.aod, &cl.aoa, &cl.eod, &cl.eoa};
        const offsets_t off = tr_offsets(M);
        auto x = [&](arma::uword col) { return rnd(l, col, 0); };
        auto phase = [&](arma::uword b, arma::uword n, arma::uword m)
        { return random_phase ? 2.0 * pi * 0.5 * std::erfc(-x(b * NM + n * M + m) / std::sqrt(2.0)) - pi : 0.0; };

        // Owner end: the first coordinate (z, x, y) whose difference exceeds 1 mm decides
        bool rx_owner = false;
        for (arma::uword i : {2, 0, 1})
            if (std::abs(f.rx(l, i) - f.tx(t, i)) > 1e-3)
            {
                rx_owner = f.rx(l, i) > f.tx(t, i);
                break;
            }

        std::vector<ref_rays_t> out(F);
        ref_rays_t &r = out[0];
        r.delay.assign(n_ray, 0.0), r.path_type.assign(n_ray, 2), r.tap.assign(n_ray, 0);
        for (auto &a : r.ang)
            a.assign(n_ray, 0.0);
        for (arma::uword s = 0; s < 1 + G; ++s)
        {
            r.delay[s] = cl.delay(l, s, 0), r.path_type[s] = s, r.tap[s] = 3 * s;
            for (int q = 0; q < 4; ++q)
                r.ang[q][s] = (*cang[q])(l, s, 0);
        }
        for (arma::uword n = 0; n < N; ++n)
        {
            const arma::uword s = 1 + G + n;
            double u_mean = 0.0;
            for (arma::uword m = 0; m < M; ++m)
                u_mean += 2.0 * c_ds * x(13 * NM + N + n * M + m) / (double)M;
            for (arma::uword m = 0; m < M; ++m)
            {
                const arma::uword ray = 1 + G + n * M + m;
                r.tap[ray] = 3 * s;
                r.delay[ray] = std::max(cl.delay(l, s, 0) + (2.0 * c_ds * x(13 * NM + N + n * M + m) - u_mean), 0.0);
                double alpha[4];
                for (arma::uword q = 0; q < 4; ++q)
                {
                    arma::uword rank = 0;
                    for (arma::uword o = 0; o < M; ++o)
                        rank += (x(q * NM + n * M + o) < x(q * NM + n * M + m) ||
                                 (x(q * NM + n * M + o) == x(q * NM + n * M + m) && o < m))
                                    ? 1
                                    : 0;
                    alpha[q] = off.asc[rank];
                }
                for (arma::uword e = 0; e < 2; ++e) // e = 0 departure (AOD, ZOD), 1 arrival (AOA, ZOA)
                {
                    const double az_c = (*cang[e])(l, s, 0), el_c = (*cang[2 + e])(l, s, 0);
                    if (angle_model == "additive")
                    {
                        r.ang[e][ray] = wrap_rad(az_c + c[e] * alpha[e] * d2r);
                        r.ang[2 + e][ray] = zen2elev(90.0 - el_c / d2r + c[2 + e] * alpha[2 + e]);
                        continue;
                    }
                    const double a = c[e] * alpha[e] * d2r, b = -c[2 + e] * alpha[2 + e] * d2r;
                    const double v[3] = {std::cos(b) * std::cos(a), std::cos(b) * std::sin(a), std::sin(b)};
                    const double v1[3] = {std::cos(el_c) * v[0] - std::sin(el_c) * v[2], v[1],
                                          std::sin(el_c) * v[0] + std::cos(el_c) * v[2]}; // about y by -el_c
                    const double w[3] = {std::cos(az_c) * v1[0] - std::sin(az_c) * v1[1],
                                         std::sin(az_c) * v1[0] + std::cos(az_c) * v1[1], v1[2]}; // about z by az_c
                    r.ang[e][ray] = wrap_rad(std::atan2(w[1], w[0]));
                    r.ang[2 + e][ray] = std::atan2(w[2], std::sqrt(w[0] * w[0] + w[1] * w[1]));
                }
            }
        }

        for (arma::uword k = 0; k < F; ++k)
        {
            ref_rays_t &o = out[k];
            if (k > 0)
                o = out[0];
            o.power.assign(n_ray, 0.0), o.J.assign(4 * n_ray, 0.0);
            for (arma::uword s = 0; s < 1 + G; ++s)
                o.power[s] = cl.power(l, s, k);
            o.J[0] = 1.0, o.J[3] = -1.0;
            if (G)
            {
                const double dx = f.rx(l, 0) - f.tx(t, 0), dy = f.rx(l, 1) - f.tx(t, 1);
                std::complex<double> R_par, R_perp;
                fresnel(f.eps_re(l, k), f.eps_im(l, k), f.tx(t, 2), f.rx(l, 2), std::sqrt(dx * dx + dy * dy), R_par,
                        R_perp);
                const double R = std::sqrt(0.5 * (std::norm(R_par) + std::norm(R_perp)));
                o.J[4] = R > 0.0 ? R_par / R : 0.0, o.J[7] = R > 0.0 ? -R_perp / R : 0.0;
            }
            for (arma::uword n = 0; n < N; ++n)
                for (arma::uword m = 0; m < M; ++m)
                {
                    const arma::uword ray = 1 + G + n * M + m;
                    std::complex<double> *J = &o.J[4 * ray]; // tt, tp, pt, pp
                    o.power[ray] = cl.power(l, 1 + G + n, k) / (double)M;
                    if (pol_model == "38.901")
                    {
                        const double X = st.mu(l, k, 7) + st.sigma(l, k, 7) * x(4 * NM + n * M + m);
                        const double ik = std::pow(10.0, -X / 10.0);
                        double eta[4] = {1.0, 1.0, 1.0, 1.0}; // tt, tp, pt, pp: blocks 9, 10, 11, 12
                        if (pol_variability)
                        {
                            for (int b = 0; b < 4; ++b)
                                eta[b] = std::pow(10.0, s_pv * x((9 + b) * NM + n * M + m) / 10.0);
                            const double nrm = (2.0 + 2.0 * ik) / (eta[0] + eta[3] + (eta[1] + eta[2]) * ik);
                            for (double &e : eta)
                                e *= nrm;
                        }
                        J[0] = std::polar(std::sqrt(eta[0]), phase(5, n, m));
                        J[1] = std::polar(std::sqrt(eta[1] * ik), phase(6, n, m));
                        J[2] = std::polar(std::sqrt(eta[2] * ik), phase(7, n, m));
                        J[3] = std::polar(std::sqrt(eta[3]), phase(8, n, m));
                        continue;
                    }
                    const double mu = 10.0 * std::log10(f.lsp(l, k, 7)), sg = 0.1 * st.sigma(l, k, 7);
                    const double gam = std::atan(std::pow(10.0, -(mu + sg * x(4 * NM + n * M + m)) / 20.0));
                    const double kap = std::atan(std::pow(10.0, -(mu + sg * x(13 * NM + n)) / 20.0));
                    const std::complex<double> p = std::polar(1.0, phase(5, n, m));
                    J[0] = p * std::polar(std::cos(gam), -kap), J[3] = -p * std::polar(std::cos(gam), kap);
                    J[1] = p * std::polar(std::sin(gam), kap), J[2] = p * std::polar(std::sin(gam), -kap);
                    if (rx_owner)
                        std::swap(J[1], J[2]);
                }
        }
        return out;
    }
}

#endif
