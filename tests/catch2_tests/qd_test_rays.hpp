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
}

#endif
