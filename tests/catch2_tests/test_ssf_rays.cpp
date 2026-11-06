// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file (see qd_test_diag.hpp).

// Tests of ssf_rays with the TR methods "38.901" and "38.901_MF" (additive angle model, TR polarisation model): the
// transcription of TR 38.901 clause 7.5 steps 7 to 11 (antenna-free part) for all attributes of all rays, the
// sub-clusters of Table 7.5-5 with the c_DS of Table 7.5-6, the coupling of step 8, the offsets for NumSubPaths other
// than 20, the per-frequency slices of "38.901_MF", the LOS and ground-reflection rays, the delay spread, pre-sized
// outputs, zero initial phases, the edge cases N = 0, M = 1, no link, zero cluster spread and the zenith fold at the
// poles; with all three methods: the ray layout, batch independence, thread count and the input validation. The
// QuaDRiGa models and clause 7.6.16 are tested in test_ssf_rays_quadriga.cpp.
// Oracles: ref_rays_tr and tr_offsets of qd_test_rays.hpp (typed from the TR text), the Fresnel oracle of the
// ssf_clusters tests, closed forms in the cases. Inputs: clusters of ssf_clusters on make_fix (qd_test_ssf.hpp, v19
// UMi LOS, UMi NLOS, UMa LOS with the ground reflection), make_lsp_stats and make_rnd_ray (distinct values).
// Tolerances: values formed by the same operations are compared exactly; transcribed angles 1e-12 rad (other order
// of the operations), delays and powers 1e-12 relative, Jones entries 1e-12 absolute; offsets recovered from the
// angles 1e-9 (inversion of the angle formula) and their mean and rms 1e-12.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "qd_test_diag.hpp"
#include "qd_test_rays.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    // Link l against the transcription at frequency in.i_freq
    void check_link(const quadriga_lib::gbsm_rays &r, const rays_in_t &in, arma::uword l)
    {
        const ref_rays_t e = ref_rays_tr(in.sp, in.f, in.cl, in.st, in.rnd, in.fc, l, in.i_freq,
                                         in.method == "38.901_MF", in.random_phase);
        REQUIRE(r.n_ray() == e.delay.size());
        const arma::vec *ang[4] = {&r.aod, &r.aoa, &r.eod, &r.eoa};
        double d_ang = 0.0, d_rel = 0.0, d_J = 0.0;
        for (arma::uword i = 0; i < r.n_ray(); ++i)
        {
            INFO("link " << l << ", ray " << i);
            CHECK(r.path_type(i) == e.path_type[i]);
            CHECK(r.tap(i) == e.tap[i]);
            d_rel = std::max(d_rel, std::abs(r.delay(i) - e.delay[i]) / std::max(e.delay[i], 1e-9));
            d_rel = std::max(d_rel, std::abs(r.power(i, 0) - e.power[i]) / std::max(e.power[i], 1e-300));
            for (int q = 0; q < 4; ++q)
            {
                const double d = (*ang[q])(i)-e.ang[q][i];
                d_ang = std::max(d_ang, std::abs(q < 2 ? wrap_rad(d) : d));
            }
            const std::complex<double> J[4] = {e.J[4 * i], e.J[4 * i + 2], e.J[4 * i + 1], e.J[4 * i + 3]};
            for (int j = 0; j < 4; ++j) // rows tt, pt, tp, pp of the 8-row layout
                d_J = std::max(d_J, std::abs(std::complex<double>(r.M(2 * j, i, 0), r.M(2 * j + 1, i, 0)) - J[j]));
        }
        QD_DIAG("%s,%llu,%llu,%.3e,%.3e,%.3e\n", in.method.c_str(), (unsigned long long)l,
                (unsigned long long)r.n_ray(), d_rel, d_ang, d_J);
        CHECK(d_rel < 1e-12);
        CHECK(d_ang < 1e-12);
        CHECK(d_J < 1e-12);
    }

    // Inputs of the links 'rows' only
    rays_in_t subset(const rays_in_t &in, const arma::uvec &rows)
    {
        rays_in_t s = in;
        s.f.tx = in.f.tx.rows(rows), s.f.rx = in.f.rx.rows(rows);
        s.f.eps_re = in.f.eps_re.rows(rows), s.f.eps_im = in.f.eps_im.rows(rows);
        for (auto p : {std::make_pair(&s.f.lsp, &in.f.lsp), std::make_pair(&s.st.mu, &in.st.mu),
                       std::make_pair(&s.st.sigma, &in.st.sigma), std::make_pair(&s.cl.delay, &in.cl.delay),
                       std::make_pair(&s.cl.power, &in.cl.power), std::make_pair(&s.cl.aod, &in.cl.aod),
                       std::make_pair(&s.cl.eod, &in.cl.eod), std::make_pair(&s.cl.aoa, &in.cl.aoa),
                       std::make_pair(&s.cl.eoa, &in.cl.eoa), std::make_pair(&s.rnd, &in.rnd)})
        {
            p.first->set_size(rows.n_elem, p.second->n_cols, p.second->n_slices);
            for (arma::uword i = 0; i < rows.n_elem; ++i)
                p.first->row(i) = p.second->row(rows(i));
        }
        return s;
    }

    // Offset of ray i of NLOS cluster n per unit spread, recovered from the azimuth (q = 0 AOD, 1 AOA) or, without
    // a zenith fold, the elevation (q = 2 ZOD, 3 ZOA): alpha = (angle - centre) / c, in degrees
    double offset_of(const quadriga_lib::gbsm_rays &r, const rays_in_t &in, arma::uword l, arma::uword n, arma::uword i,
                     int q, double c)
    {
        const arma::uword s = 1 + in.G + n;
        const double d2r = pi / 180.0;
        switch (q)
        {
        case 0:
            return wrap_rad(r.aod(i) - in.cl.aod(l, s, 0)) / d2r / c;
        case 1:
            return wrap_rad(r.aoa(i) - in.cl.aoa(l, s, 0)) / d2r / c;
        case 2:
            return (in.cl.eod(l, s, 0) - r.eod(i)) / d2r / c;
        default:
            return (in.cl.eoa(l, s, 0) - r.eoa(i)) / d2r / c;
        }
    }
}

TEST_CASE("ssf_rays - output layout: entries, sizes, path types, taps and empty scatterers")
{
    // TR methods: frequency i_freq = n_freq - 1 only; "quadriga": all frequencies, no sub-clusters
    for (const char *method : qd_methods)
    {
        for (bool gr : {false, true})
        {
            const bool qd = std::string(method) == "quadriga";
            const arma::uword F = std::string(method) == "38.901" ? 1 : 2, Fo = qd ? F : 1;
            rays_in_t in = make_rays_in(scen("UMa_LOS", gr), 3, method, 11, F);
            in.i_freq = qd ? 0 : F - 1;
            std::vector<quadriga_lib::gbsm_rays> rays(7); // resized to n_link
            call(in, rays);
            INFO("method " << method << ", GR " << gr);
            REQUIRE(rays.size() == 3);
            const arma::uword N = in.N, M = in.M, G = in.G, n_ray = 1 + G + N * M;
            for (arma::uword l = 0; l < 3; ++l)
            {
                const quadriga_lib::gbsm_rays &r = rays[l];
                CHECK(r.is_valid() == "");
                CHECK(r.method == method);
                CHECK(arma::all(r.tx_pos == in.f.tx.row(l).t()));
                CHECK(arma::all(r.rx_pos == in.f.rx.row(l).t()));
                CHECK(arma::all(r.center_frequency == (qd ? in.fc : arma::vec{in.fc(F - 1)})));
                CHECK(r.n_ray() == n_ray);
                CHECK(arma::size(r.power) == arma::size(n_ray, Fo));
                CHECK(arma::size(r.M) == arma::size(8, n_ray, Fo));
                for (const arma::vec *a : {&r.aod, &r.eod, &r.aoa, &r.eoa})
                    CHECK(a->n_elem == n_ray);
                REQUIRE((r.path_type.n_elem == n_ray && r.tap.n_elem == n_ray));
                CHECK(r.path_type(0) == 0);
                CHECK(r.tap(0) == 0);
                if (G)
                {
                    CHECK(r.path_type(1) == 1);
                    CHECK(r.tap(1) == 3);
                    const arma::uword k0 = qd ? 0 : F - 1;
                    CHECK(arma::all(r.gr_eps_re == in.f.eps_re(arma::span(l), arma::span(k0, F - 1)).t()));
                    CHECK(arma::all(r.gr_eps_im == in.f.eps_im(arma::span(l), arma::span(k0, F - 1)).t()));
                }
                CHECK(r.gr_eps_re.n_elem == G * Fo);
                CHECK(r.gr_eps_im.n_elem == G * Fo);
                for (arma::uword n = 0; n < N; ++n)
                    for (arma::uword m = 0; m < M; ++m)
                    {
                        const arma::uword i = 1 + G + n * M + m, base = 3 * (1 + G + n);
                        CHECK(r.path_type(i) == 2);
                        CHECK((r.tap(i) >= base && r.tap(i) <= base + (qd ? 0 : 2)));
                        CHECK((m == 0 || r.tap(i) >= r.tap(i - 1))); // every tap a contiguous block
                    }
                CHECK(r.fbs_pos.empty());
                CHECK(r.lbs_pos.empty());
                CHECK(r.fallback.empty());
            }
        }
    }
}

TEST_CASE("ssf_rays - 38.901 matches TR 7.5 steps 7 to 10")
{
    QD_DIAG("\n# diag: ssf_rays against the transcription (max relative delay/power error, angle, Jones)\n");
    QD_DIAG("method,link,n_ray,d_rel,d_ang,d_J\n");
    const std::pair<const char *, bool> cases[3] = {{"UMi_NLOS", false}, {"UMi_LOS", false}, {"UMa_LOS", true}};
    unsigned seed = 21;
    for (const auto &c : cases)
    {
        INFO("scenario " << c.first);
        const rays_in_t in = make_rays_in(scen(c.first, c.second), 6, "38.901", seed++);
        const auto rays = call(in);
        REQUIRE(rays.size() == 6);
        for (arma::uword l = 0; l < 6; ++l)
            check_link(rays[l], in, l);
    }
}

TEST_CASE("ssf_rays - 38.901 splits the two strongest clusters into the sub-clusters of Table 7.5-5")
{
    // Expected split set: given clusters, checked by delays (slot delay + 0, 1.28, 2.56 c_DS at the positions 0-9,
    // 10-15, 16-19), taps (3·slot + sub-cluster) and the unsplit clusters (slot delay, 3·slot)
    auto check = [](const quadriga_lib::gbsm_rays &r, const rays_in_t &in, arma::uword l,
                    std::vector<arma::uword> split, double c_ds)
    {
        const arma::uword g = in.method == "38.901_MF" ? in.i_freq : 0;
        for (arma::uword n = 0; n < in.N; ++n)
        {
            const bool sp = std::find(split.begin(), split.end(), n) != split.end();
            for (arma::uword m = 0; m < in.M; ++m)
            {
                const arma::uword i = 1 + in.G + n * in.M + m, s = 1 + in.G + n;
                const arma::uword sub = sp ? (m < 10 ? 0 : m < 16 ? 1 : 2) : 0;
                INFO("link " << l << ", cluster " << n << ", ray " << m);
                CHECK(r.tap(i) == 3 * s + sub);
                CHECK(std::abs(r.delay(i) - in.cl.delay(l, s, g) - 1.28 * (double)sub * c_ds) < 1e-12 * 1e-6);
            }
        }
    };

    // c_DS of Table 7.5-6 against the values of the TR (UMi NLOS 11 ns; UMa 3.91 ns at 3.5 GHz with the 6 GHz
    // limit of note 6)
    spmap umi = scen("UMi_NLOS"), uma = scen("UMa_LOS"), def;
    for (spmap *s : {&umi, &uma, &def})
        quadriga_lib::scenpar_validate(*s);
    CHECK(std::abs(ref_cds(umi, 3.5e9) - 11e-9) < 1e-15);
    CHECK(std::abs(ref_cds(uma, 3.5e9) - 3.9099e-9) < 1e-13);

    SECTION("two strongest by coefficient power")
    {
        const rays_in_t in = make_rays_in(scen("UMi_NLOS"), 4, "38.901", 31);
        const auto rays = call(in);
        for (arma::uword l = 0; l < 4; ++l)
        {
            arma::vec P = arma::vec(in.N);
            for (arma::uword n = 0; n < in.N; ++n)
                P(n) = in.cl.power(l, 1 + in.G + n, 0);
            const arma::uvec o = arma::sort_index(P, "descend");
            check(rays[l], in, l, {o(0), o(1)}, 11e-9);
        }
    }
    SECTION("equal powers: lower index first; one active cluster; no active cluster")
    {
        rays_in_t in = make_rays_in(scen("UMi_NLOS"), 3, "38.901", 32);
        for (arma::uword n = 0; n < in.N; ++n)
        {
            const arma::uword s = 1 + in.G + n;
            in.cl.power(0, s, 0) = (n == 2 || n == 5 || n == 9) ? 0.2 : 0.01;
            in.cl.power(1, s, 0) = (n == 4) ? 0.3 : 0.0;
            in.cl.power(2, s, 0) = 0.0;
        }
        const auto rays = call(in);
        check(rays[0], in, 0, {2, 5}, 11e-9);
        check(rays[1], in, 1, {4}, 11e-9);
        check(rays[2], in, 2, {}, 11e-9);
        for (arma::uword i = 1; i < rays[2].n_ray(); ++i)
            CHECK(rays[2].power(i, 0) == 0.0); // zero-power clusters keep their rays at power 0
    }
    SECTION("NumSubPaths other than 20: no split")
    {
        spmap sp = scen("UMi_NLOS");
        sp["NumSubPaths"] = 19.0;
        const rays_in_t in = make_rays_in(sp, 2, "38.901", 33);
        const auto rays = call(in);
        for (arma::uword l = 0; l < 2; ++l)
            check(rays[l], in, l, {}, 11e-9);
    }
    SECTION("PerClusterDS = 0 uses 3.91 ns")
    {
        spmap sp = scen("UMi_NLOS");
        sp["PerClusterDS"] = 0.0;
        const rays_in_t in = make_rays_in(sp, 2, "38.901", 34);
        const auto rays = call(in);
        for (arma::uword l = 0; l < 2; ++l)
        {
            arma::vec P = arma::vec(in.N);
            for (arma::uword n = 0; n < in.N; ++n)
                P(n) = in.cl.power(l, 1 + in.G + n, 0);
            const arma::uvec o = arma::sort_index(P, "descend");
            check(rays[l], in, l, {o(0), o(1)}, 3.91e-9);
        }
    }
    SECTION("UMa: c_DS at 3.5 and 28 GHz, split per frequency")
    {
        rays_in_t in = make_rays_in(scen("UMa_LOS"), 3, "38.901_MF", 35, 2);
        for (arma::uword k = 0; k < 2; ++k)
        {
            in.i_freq = k;
            const auto rays = call(in);
            for (arma::uword l = 0; l < 3; ++l)
            {
                arma::vec P = arma::vec(in.N);
                for (arma::uword n = 0; n < in.N; ++n)
                    P(n) = in.cl.power(l, 1 + in.G + n, k);
                const arma::uvec o = arma::sort_index(P, "descend");
                check(rays[l], in, l, {o(0), o(1)}, ref_cds(uma, in.fc(k)));
            }
        }
    }
}

TEST_CASE("ssf_rays - coupling gives each quantity a permutation of the offsets within each group")
{
    // Cluster elevations 0.1 rad, so that no zenith fold occurs and the offsets follow from the angles. Per group and
    // quantity: each recovered offset is one of the group's offsets (1e-9), each offset is used once, and the k-th
    // smallest value of the rank block gets the k-th smallest offset (exact)
    rays_in_t in = make_rays_in(scen("UMi_NLOS"), 3, "38.901", 41);
    in.cl.eod.fill(0.1), in.cl.eoa.fill(0.1);
    const auto rays = call(in);
    spmap sp = in.sp;
    quadriga_lib::scenpar_validate(sp);
    const offsets_t off = tr_offsets(in.M);
    arma::uword n_split = 0;
    for (arma::uword l = 0; l < 3; ++l)
    {
        const double c[4] = {sp.at("PerClusterAS_D"), sp.at("PerClusterAS_A"),
                             sp.at("PerClusterES_D_scale") * std::pow(10.0, in.st.mu(l, 0, 5)),
                             sp.at("PerClusterES_A")};
        for (arma::uword n = 0; n < in.N; ++n)
        {
            const arma::uword i0 = 1 + in.G + n * in.M;
            const bool split = rays[l].tap(i0 + in.M - 1) != rays[l].tap(i0);
            n_split += split ? 1 : 0;
            const arma::uword n_grp = split ? 3 : 1, first[3] = {0, 10, 16}, size[3] = {split ? 10 : in.M, 6, 4};
            for (arma::uword gi = 0; gi < n_grp; ++gi)
            {
                const std::vector<double> &A = split ? off.groups[gi] : off.asc;
                for (int q = 0; q < 4; ++q)
                {
                    INFO("link " << l << ", cluster " << n << ", group " << gi << ", quantity " << q);
                    std::vector<arma::uword> used(A.size(), 0), idx(size[gi]);
                    arma::vec x(size[gi]);
                    for (arma::uword j = 0; j < size[gi]; ++j)
                    {
                        const arma::uword m = first[gi] + j;
                        const double a = offset_of(rays[l], in, l, n, i0 + m, q, c[q]);
                        arma::uword best = 0;
                        for (arma::uword o = 1; o < A.size(); ++o)
                            best = std::abs(A[o] - a) < std::abs(A[best] - a) ? o : best;
                        CHECK(std::abs(A[best] - a) < 1e-9);
                        used[best] += 1, idx[j] = best;
                        x(j) = in.rnd(l, q * in.N * in.M + n * in.M + m, 0);
                    }
                    CHECK(std::all_of(used.begin(), used.end(), [](arma::uword u) { return u == 1; }));
                    const arma::uvec order = arma::sort_index(x);
                    for (arma::uword k = 0; k < size[gi]; ++k)
                        CHECK(idx[order(k)] == k);
                }
            }
        }
    }
    CHECK(n_split == 6);
}

TEST_CASE("ssf_rays - ray offsets for M other than 20")
{
    // Offsets of the AOA (the rank block 1 orders them), against the 6-decimal values of the rule for M < 20 and the
    // Laplacian quantiles for M > 20; mean 0 and rms 1 (M < 20) or the rms of Table 7.5-3 (M > 20)
    const std::vector<std::vector<double>> quoted = {
        {0.0},
        {-1.0, 1.0},
        {-1.379551, 0.420279, 0.959273},
        {-1.115204, -0.847110, 0.743054, 1.219260},
        {-1.197040, -0.945308, -0.749898, 0.547806, 0.994948, 1.349492},
        {-2.179417, -1.549424, -1.180902, -0.919431, -0.716619, -0.550909, -0.410804, -0.289439, -0.182387,
         -0.086626, 0.0},
        {-2.288001, -1.678661, -1.322221, -1.069322, -0.873159, -0.712881, -0.577369, -0.459983, -0.356441,
         -0.263819, -0.180033, -0.103542, -0.033177}};
    const arma::uword Ms[7] = {1, 2, 3, 4, 6, 21, 26};
    for (int t = 0; t < 7; ++t)
    {
        const arma::uword M = Ms[t];
        std::vector<double> asc = quoted[t];
        for (size_t j = asc.size(); M > 6 && j-- > 0;) // the positive half of the symmetric sets
            if (asc[j] < 0.0)
                asc.push_back(-asc[j]);
        REQUIRE(asc.size() == M);
        spmap sp = scen("UMi_NLOS");
        sp["NumClusters"] = sp["NumClusters_min"] = 4.0, sp["NumSubPaths"] = (double)M;
        const rays_in_t in = make_rays_in(sp, 2, "38.901", 50 + t);
        const auto rays = call(in);
        const double rms_target = M < 20 ? 1.0 : tr_offsets(20).rms_table;
        for (arma::uword l = 0; l < 2; ++l)
            for (arma::uword n = 0; n < in.N; ++n)
            {
                INFO("M " << M << ", link " << l << ", cluster " << n);
                const arma::uword i0 = 1 + in.G + n * M;
                arma::vec a(M), x(M);
                for (arma::uword m = 0; m < M; ++m)
                {
                    a(m) = offset_of(rays[l], in, l, n, i0 + m, 1, 22.0); // UMi NLOS: c_ASA = 22 deg
                    x(m) = in.rnd(l, in.N * M + n * M + m, 0);
                    CHECK(rays[l].tap(i0 + m) == 3 * (1 + in.G + n));
                    CHECK(rays[l].delay(i0 + m) == in.cl.delay(l, 1 + in.G + n, 0));
                }
                const arma::vec r = a(arma::sort_index(x));
                double mean = 0.0, ms = 0.0;
                for (arma::uword m = 0; m < M; ++m)
                {
                    CHECK(std::abs(r(m) - asc[m]) < 1e-6);
                    mean += r(m) / (double)M, ms += r(m) * r(m) / (double)M;
                }
                CHECK(std::abs(mean) < 1e-12);
                CHECK(std::abs(std::sqrt(ms) - (M == 1 ? 0.0 : rms_target)) < 1e-12);
                if (M == 21)
                    CHECK(std::abs(r(10)) < 1e-12); // the middle rank of an odd M > 20
            }
    }
}

TEST_CASE("ssf_rays - 38.901_MF computes frequency i_freq with its own slices")
{
    // UMa NLOS at 3.5 and 28 GHz with the ground reflection; frequency 1 equals a call with only the slices and
    // columns of frequency 1, and the split clusters differ between the frequencies
    rays_in_t in = make_rays_in(scen("UMa_NLOS", true), 4, "38.901_MF", 61, 2);
    in.i_freq = 1;
    const auto r1 = call(in);
    rays_in_t one = in;
    one.fc = arma::vec{28e9}, one.i_freq = 0;
    for (auto p : {std::make_pair(&one.f.lsp, &in.f.lsp), std::make_pair(&one.st.mu, &in.st.mu),
                   std::make_pair(&one.st.sigma, &in.st.sigma)})
        *p.first = p.second->cols(1, 1);
    for (auto p : {std::make_pair(&one.cl.delay, &in.cl.delay), std::make_pair(&one.cl.power, &in.cl.power),
                   std::make_pair(&one.cl.aod, &in.cl.aod), std::make_pair(&one.cl.eod, &in.cl.eod),
                   std::make_pair(&one.cl.aoa, &in.cl.aoa), std::make_pair(&one.cl.eoa, &in.cl.eoa),
                   std::make_pair(&one.rnd, &in.rnd)})
        *p.first = p.second->slices(1, 1);
    one.f.eps_re = in.f.eps_re.col(1), one.f.eps_im = in.f.eps_im.col(1);
    const auto r1b = call(one);
    in.i_freq = 0;
    const auto r0 = call(in);
    arma::uword n_diff = 0;
    for (arma::uword l = 0; l < 4; ++l)
    {
        INFO("link " << l);
        CHECK(same(r1[l], r1b[l]));
        CHECK(r1[l].center_frequency(0) == 28e9);
        CHECK(r0[l].center_frequency(0) == 3.5e9);
        n_diff += arma::any(r0[l].tap != r1[l].tap) ? 1 : 0;
        check_link(r0[l], in, l);
    }
    in.i_freq = 1;
    for (arma::uword l = 0; l < 4; ++l)
        check_link(r1[l], in, l);
    CHECK(n_diff > 0);
}

TEST_CASE("ssf_rays - LOS and GR rays carry the slot values and the TR Jones matrices")
{
    for (const char *name : {"UMa_LOS", "UMi_NLOS"})
    {
        const rays_in_t in = make_rays_in(scen(name, true), 4, "38.901", 71);
        const auto rays = call(in);
        const bool los = std::string(name) == "UMa_LOS";
        for (arma::uword l = 0; l < 4; ++l)
        {
            INFO(name << ", link " << l);
            const quadriga_lib::gbsm_rays &r = rays[l];
            REQUIRE(r.n_ray() == 2 + in.N * in.M);
            for (arma::uword s = 0; s < 2; ++s)
            {
                CHECK(r.delay(s) == in.cl.delay(l, s, 0));
                CHECK(r.power(s, 0) == in.cl.power(l, s, 0));
                CHECK(r.aod(s) == in.cl.aod(l, s, 0));
                CHECK(r.eod(s) == in.cl.eod(l, s, 0));
                CHECK(r.aoa(s) == in.cl.aoa(l, s, 0));
                CHECK(r.eoa(s) == in.cl.eoa(l, s, 0));
                CHECK((los ? r.power(s, 0) > 0.0 : r.power(s, 0) == 0.0)); // NLOS: kept at power 0
            }
            CHECK(arma::all(r.M.slice(0).col(0) == arma::vec{1, 0, 0, 0, 0, 0, -1, 0}));
            const geo_t geo = geometry(in.f, l);
            std::complex<double> R_par, R_perp;
            fresnel(in.f.eps_re(l, 0), in.f.eps_im(l, 0), geo.h_t, geo.h_r, geo.d2D, R_par, R_perp);
            const double R = std::sqrt(geo.R2[0]);
            const arma::vec e = {R_par.real() / R, R_par.imag() / R, 0, 0, 0, 0, -R_perp.real() / R,
                                 -R_perp.imag() / R};
            CHECK(arma::approx_equal(r.M.slice(0).col(1), e, "absdiff", 1e-12));
        }
    }
}

TEST_CASE("ssf_rays - ray delay spread with sub-clusters equals the analytic value")
{
    // Analytic: each split cluster as three paths of power 10/20, 6/20, 4/20 of P_n at delays tau_n + 0, 1.28, 2.56
    // c_DS (UMi LOS 3.5 GHz: c_DS = 5 ns), the other slots as one path each
    const rays_in_t in = make_rays_in(scen("UMi_LOS"), 6, "38.901", 81);
    const auto rays = call(in);
    auto ds = [](const std::vector<double> &p, const std::vector<double> &t)
    {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0;
        for (size_t i = 0; i < p.size(); ++i)
            s0 += p[i], s1 += p[i] * t[i], s2 += p[i] * t[i] * t[i];
        return std::sqrt(s2 / s0 - (s1 / s0) * (s1 / s0));
    };
    for (arma::uword l = 0; l < 6; ++l)
    {
        std::vector<double> p, t, pr, tr;
        arma::vec P(in.N);
        for (arma::uword n = 0; n < in.N; ++n)
            P(n) = in.cl.power(l, 1 + n, 0);
        const arma::uvec o = arma::sort_index(P, "descend");
        for (arma::uword s = 0; s < 1 + in.N; ++s)
        {
            const bool split = s > 0 && (s - 1 == o(0) || s - 1 == o(1));
            for (int i = 0; i < (split ? 3 : 1); ++i)
                p.push_back(in.cl.power(l, s, 0) * (split ? (i == 0 ? 0.5 : i == 1 ? 0.3 : 0.2) : 1.0)),
                    t.push_back(in.cl.delay(l, s, 0) + 1.28 * i * 5e-9);
        }
        for (arma::uword i = 0; i < rays[l].n_ray(); ++i)
            pr.push_back(rays[l].power(i, 0)), tr.push_back(rays[l].delay(i));
        INFO("link " << l);
        CHECK(std::abs(ds(pr, tr) / ds(p, t) - 1.0) < 1e-12);
    }
}

TEST_CASE("ssf_rays - a link does not depend on the other links")
{
    for (const char *method : qd_methods)
    {
        const arma::uword F = std::string(method) == "38.901" ? 1 : 2;
        rays_in_t in = make_rays_in(scen("UMa_LOS", true), 5, method, 91, F);
        in.i_freq = std::string(method) == "quadriga" ? 0 : F - 1;
        const auto all = call(in);
        const auto sub = call(subset(in, arma::uvec{3, 1}));
        INFO("method " << method);
        CHECK(same(sub[0], all[3]));
        CHECK(same(sub[1], all[1]));
    }
}

TEST_CASE("ssf_rays - pre-sized entries are written in place")
{
    // Every attribute on strict external memory of the right size, filled with NaN: written completely, no
    // reallocation; entries of other sizes are resized
    const rays_in_t in = make_rays_in(scen("UMa_LOS", true), 3, "38.901", 101);
    const auto ref = call(in);
    const arma::uword n_ray = ref[0].n_ray();
    std::vector<double> buf(3 * (3 + 3 + 1 + 6 * n_ray + 8 * n_ray + 2), NAN);
    std::vector<arma::uword> ubuf(3 * 2 * n_ray, 99);
    std::vector<quadriga_lib::gbsm_rays> rays(3);
    std::vector<const void *> ptr;
    double *p = buf.data();
    arma::uword *u = ubuf.data();
    auto vec = [&](arma::vec &x, arma::uword n)
    { x.~Col(), new (&x) arma::vec(p, n, false, true), ptr.push_back(p), p += n; };
    auto uvec = [&](arma::uvec &x, arma::uword n)
    { x.~Col(), new (&x) arma::uvec(u, n, false, true), ptr.push_back(u), u += n; };
    for (auto &r : rays)
    {
        vec(r.tx_pos, 3), vec(r.rx_pos, 3), vec(r.center_frequency, 1), vec(r.delay, n_ray);
        r.power.~Mat(), new (&r.power) arma::mat(p, n_ray, 1, false, true), ptr.push_back(p), p += n_ray;
        vec(r.aod, n_ray), vec(r.eod, n_ray), vec(r.aoa, n_ray), vec(r.eoa, n_ray);
        r.M.~Cube(), new (&r.M) arma::cube(p, 8, n_ray, 1, false, true), ptr.push_back(p), p += 8 * n_ray;
        vec(r.gr_eps_re, 1), vec(r.gr_eps_im, 1), uvec(r.path_type, n_ray), uvec(r.tap, n_ray);
    }
    CHECK_NOTHROW(call(in, rays));
    size_t j = 0;
    for (arma::uword l = 0; l < 3; ++l)
    {
        const auto &r = rays[l];
        for (const void *q : {(const void *)r.tx_pos.memptr(), (const void *)r.rx_pos.memptr(),
                              (const void *)r.center_frequency.memptr(), (const void *)r.delay.memptr(),
                              (const void *)r.power.memptr(), (const void *)r.aod.memptr(),
                              (const void *)r.eod.memptr(), (const void *)r.aoa.memptr(), (const void *)r.eoa.memptr(),
                              (const void *)r.M.memptr(), (const void *)r.gr_eps_re.memptr(),
                              (const void *)r.gr_eps_im.memptr(), (const void *)r.path_type.memptr(),
                              (const void *)r.tap.memptr()})
            CHECK(q == ptr[j++]);
        CHECK(same(r, ref[l]));
    }

    // Wrong sizes and stale scatterers are replaced
    std::vector<quadriga_lib::gbsm_rays> other(2);
    other[0] = ref[2], other[0].delay.zeros(5), other[0].M.zeros(8, 2, 3), other[0].fbs_pos.zeros(3, n_ray);
    other[0].lbs_pos.zeros(3, n_ray), other[0].fallback.zeros(n_ray), other[0].gr_eps_re.reset();
    call(in, other);
    REQUIRE(other.size() == 3);
    for (arma::uword l = 0; l < 3; ++l)
        CHECK(same(other[l], ref[l]));
}

TEST_CASE("ssf_rays - random_phase false gives zero initial phases")
{
    rays_in_t in = make_rays_in(scen("UMi_NLOS", true), 3, "38.901", 111);
    in.random_phase = false;
    const auto rays = call(in);
    for (arma::uword l = 0; l < 3; ++l)
    {
        const quadriga_lib::gbsm_rays &r = rays[l];
        for (arma::uword i = 2; i < r.n_ray(); ++i)
        {
            INFO("link " << l << ", ray " << i);
            CHECK((r.M(1, i, 0) == 0.0 && r.M(3, i, 0) == 0.0 && r.M(5, i, 0) == 0.0 && r.M(7, i, 0) == 0.0));
            CHECK((r.M(0, i, 0) == 1.0 && r.M(6, i, 0) == 1.0));
            CHECK((r.M(2, i, 0) > 0.0 && r.M(2, i, 0) == r.M(4, i, 0)));
        }
        check_link(r, in, l);
    }
}

TEST_CASE("ssf_rays - N = 0, M = 1 and n_link = 0")
{
    // N = 0: the LOS and GR rays only
    spmap sp = scen("UMa_LOS", true);
    sp["NumClusters"] = sp["NumClusters_min"] = 1.0;
    const rays_in_t in0 = make_rays_in(sp, 2, "38.901", 121);
    REQUIRE(in0.rnd.n_cols == 0);
    const auto r0 = call(in0);
    REQUIRE(r0.size() == 2);
    for (arma::uword l = 0; l < 2; ++l)
    {
        CHECK(r0[l].n_ray() == 2);
        CHECK(r0[l].is_valid() == "");
        CHECK(r0[l].power(0, 0) == in0.cl.power(l, 0, 0));
        CHECK(arma::all(r0[l].tap == arma::uvec{0, 3}));
    }

    // M = 1: one ray per cluster at the cluster angles, delay and power
    sp = scen("UMi_LOS");
    sp["NumSubPaths"] = 1.0;
    const rays_in_t in1 = make_rays_in(sp, 2, "38.901", 122);
    const auto r1 = call(in1);
    for (arma::uword l = 0; l < 2; ++l)
    {
        REQUIRE(r1[l].n_ray() == 1 + in1.N);
        for (arma::uword n = 0; n < in1.N; ++n)
        {
            const arma::uword s = 1 + n;
            INFO("link " << l << ", cluster " << n);
            CHECK(r1[l].delay(s) == in1.cl.delay(l, s, 0));
            CHECK(r1[l].power(s, 0) == in1.cl.power(l, s, 0));
            CHECK(r1[l].tap(s) == 3 * s);
            CHECK(r1[l].aod(s) == in1.cl.aod(l, s, 0));
            CHECK(r1[l].aoa(s) == in1.cl.aoa(l, s, 0));
            CHECK(std::abs(r1[l].eod(s) - in1.cl.eod(l, s, 0)) < 1e-12);
            CHECK(std::abs(r1[l].eoa(s) - in1.cl.eoa(l, s, 0)) < 1e-12);
        }
    }

    // No link: the vector is emptied
    const rays_in_t e = make_rays_in(scen("UMi_LOS"), 0, "38.901", 123);
    std::vector<quadriga_lib::gbsm_rays> rays(3);
    call(e, rays);
    CHECK(rays.empty());
}

TEST_CASE("ssf_rays - c_ASA = 0 gives AOA offsets 0")
{
    spmap sp = scen("UMi_NLOS");
    sp["PerClusterAS_A"] = 0.0;
    const rays_in_t in = make_rays_in(sp, 3, "38.901", 131);
    const auto rays = call(in);
    for (arma::uword l = 0; l < 3; ++l)
        for (arma::uword i = 1; i < rays[l].n_ray(); ++i)
            CHECK(rays[l].aoa(i) == in.cl.aoa(l, 1 + (i - 1) / in.M, 0));
}

TEST_CASE("ssf_rays - results do not depend on the thread count")
{
#ifdef _OPENMP
    for (const char *method : qd_methods)
    {
        const arma::uword F = std::string(method) == "38.901" ? 1 : 2;
        const rays_in_t in = make_rays_in(scen("UMa_LOS", true), 24, method, 141, F);
        const int n_threads = omp_get_max_threads();
        omp_set_num_threads(1);
        const auto serial = call(in);
        omp_set_num_threads(n_threads);
        const auto parallel = call(in);
        for (arma::uword l = 0; l < 24; ++l)
            CHECK(same(serial[l], parallel[l]));
    }
#endif
}

TEST_CASE("ssf_rays - zenith fold at the poles")
{
    // Cluster elevations +-89 deg (ZOA and ZOD of opposite sign): offsets up to 15 deg (c_ZSA = 7 deg) cross the
    // pole. All rays against the transcription; in addition, per cluster, the zenith angles of arrival form the set
    // fold(90 - el_c + c_ZSA·alpha) over the 20 offsets of Table 7.5-3 (the sub-cluster groups together hold all 20),
    // with fold(t) = t mod 360, then 360 - t above 180 deg
    rays_in_t in = make_rays_in(scen("UMi_NLOS"), 2, "38.901", 151);
    for (arma::uword s = 1; s < 1 + in.N; ++s)
        for (arma::uword l = 0; l < 2; ++l)
            in.cl.eoa(l, s, 0) = (s % 2 ? 89.0 : -89.0) * pi / 180.0, in.cl.eod(l, s, 0) = -in.cl.eoa(l, s, 0);
    const auto rays = call(in);
    const std::vector<double> A = tr_offsets(20).asc;
    arma::uword n_fold = 0;
    for (arma::uword l = 0; l < 2; ++l)
    {
        check_link(rays[l], in, l);
        for (arma::uword n = 0; n < in.N; ++n)
        {
            const double el_c = in.cl.eoa(l, 1 + n, 0) * 180.0 / pi;
            arma::vec z(20), e(20);
            for (arma::uword m = 0; m < 20; ++m)
            {
                double t = 90.0 - el_c + 7.0 * A[m];
                n_fold += (t < 0.0 || t > 180.0) ? 1 : 0;
                t -= 360.0 * std::floor(t / 360.0);
                e(m) = t >= 180.0 ? 360.0 - t : t;
                z(m) = 90.0 - rays[l].eoa(1 + n * 20 + m) * 180.0 / pi;
            }
            INFO("link " << l << ", cluster " << n);
            CHECK(arma::approx_equal(arma::sort(z), arma::sort(e), "absdiff", 1e-9));
        }
    }
    CHECK(n_fold > 0);
}

TEST_CASE("ssf_rays - input validation")
{
    // UMa LOS (N = 12, M = 20, n_gen = 3372) with the ground reflection, 3 links; "38.901" with one frequency,
    // "38.901_MF" with two and i_freq = 1, "quadriga" with two. Each case changes one thing and expects the message
    typedef std::function<void(rays_in_t &)> mut_t;
    for (const char *method : qd_methods)
    {
        const bool mf = std::string(method) == "38.901_MF", is_qd = std::string(method) == "quadriga";
        rays_in_t base = make_rays_in(scen("UMa_LOS", true), 3, method, 161, std::string(method) == "38.901" ? 1 : 2);
        base.i_freq = mf ? 1 : 0;
        auto qd = [](rays_in_t &x) { x.method = "quadriga", x.i_freq = 0, x.rnd = x.rnd.slices(0, 0); };
        std::vector<std::pair<mut_t, std::string>> bad = {
            {[](rays_in_t &x) { x.method = "QuaDRiGa"; },
             "Input 'method' must be '38.901', '38.901_MF' or 'quadriga'."},
            {[](rays_in_t &x) { x.angle_model = "Additive"; }, "Input 'angle_model' must be 'additive' or 'rotation'."},
            {[](rays_in_t &x) { x.pol_model = "TR"; }, "Input 'pol_model' must be '38.901' or 'quadriga'."},
            {[](rays_in_t &x) { x.pol_model = "quadriga", x.pol_variability = true; },
             "Input 'pol_variability' needs pol_model '38.901'."},
            {[](rays_in_t &x) { x.sp["NumSubPaths"] = 0.0; },
             "Value of 'NumSubPaths' must be an integer from 1 to 1000."},
            {[](rays_in_t &x) { x.f.rx = x.f.rx.cols(0, 1); }, "Input 'rx_pos' must have 3 columns."},
            {[](rays_in_t &x) { x.f.tx = x.f.tx.rows(0, 1); },
             "Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'."},
            {[](rays_in_t &x) { x.fc.reset(); }, "Input 'center_frequency' cannot be empty."},
            {[](rays_in_t &x) { x.fc(0) = 0.0; }, "Input 'center_frequency' must be finite and > 0."},
            {[](rays_in_t &x) { x.i_freq = x.fc.n_elem; },
             is_qd ? "Input 'i_freq' must select the first frequency for method 'quadriga'."
                   : "Input 'i_freq' is out of range for " + std::to_string(base.fc.n_elem) + " frequencies."},
            {[qd](rays_in_t &x) { qd(x), x.i_freq = 1, x.fc = arma::vec{3.5e9, 28e9}; },
             "Input 'i_freq' must select the first frequency for method 'quadriga'."},
            {[](rays_in_t &x) { x.f.lsp = x.f.lsp.rows(0, 1); }, "Input 'lsp' must have size [n_link, n_freq, 8]."},
            {[](rays_in_t &x) { x.st.mu = x.st.mu.slices(0, 6); },
             "Input 'lsp_mu' must have size [n_link, n_freq, 8]."},
            {[](rays_in_t &x) { x.st.sigma = x.st.sigma.rows(1, 2); },
             "Input 'lsp_sigma' must have size [n_link, n_freq, 8]."},
            {[](rays_in_t &x) { x.cl.power = x.cl.power.cols(0, 5); },
             "Input 'power' must have size [n_link, n_slot, n_freq]."},
            {[](rays_in_t &x) { x.rnd = x.rnd.cols(0, 3370); },
             "Input 'rnd_ray' must have size [n_link, 3372, n_freq] for the TR methods or [n_link, 3372, 1] for "
             "'quadriga'."},
            {[](rays_in_t &x) { x.f.use_eps = false; },
             "Inputs 'gr_eps_re' and 'gr_eps_im' are required when 'GR_enabled' is set."},
            {[](rays_in_t &x) { x.f.eps_im = x.f.eps_im.rows(0, 1); },
             "Inputs 'gr_eps_re' and 'gr_eps_im' must have size [n_link, n_freq]."},
            {[](rays_in_t &x) { x.f.eps_re(0, 0) = NAN; }, "Inputs 'gr_eps_re' and 'gr_eps_im' must be finite."},
            {[](rays_in_t &x) { x.f.eps_re(1, 0) = 0.99; },
             "Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0."},
            {[](rays_in_t &x) { x.f.eps_im(2, 0) = -1e-9; },
             "Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0."},
            {[](rays_in_t &x) { x.f.rx(1, 2) = -0.1; }, "Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'."},
            {[](rays_in_t &x) { x.f.tx(1, 2) = -0.1; }, "Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'."},
            {[](rays_in_t &x) { x.cl.power(1, 4, x.i_freq) = -1e-12; }, "Input 'power' must be finite and >= 0."},
            {[](rays_in_t &x) { x.st.mu(2, x.i_freq, 5) = NAN; }, "Input 'lsp_mu' must be finite."},
            {[](rays_in_t &x) { x.st.mu(0, x.i_freq, 7) = INFINITY; }, "Input 'lsp_mu' must be finite."},
            {[](rays_in_t &x) { x.st.sigma(1, x.i_freq, 7) = -0.1; }, "Input 'lsp_sigma' must be finite and >= 0."},
            {[](rays_in_t &x) { x.pol_model = "quadriga", x.f.lsp(1, x.i_freq, 7) = 0.0; },
             "Input 'lsp' must be finite and > 0."},
            {[](rays_in_t &x) { x.rnd(2, 1000, x.i_freq) = NAN; }, "Input 'rnd_ray' must be finite."}};
        const char *geo[5] = {"delay", "aod", "eod", "aoa", "eoa"};
        for (int a = 0; a < 5; ++a)
        {
            bad.push_back({[a](rays_in_t &x)
                           {
                               arma::cube *c[5] = {&x.cl.delay, &x.cl.aod, &x.cl.eod, &x.cl.aoa, &x.cl.eoa};
                               *c[a] = c[a]->cols(0, 5);
                           },
                           "Input '" + std::string(geo[a]) +
                               "' must have size [n_link, n_slot, n_freq] for method '38.901_MF', else [n_link, "
                               "n_slot, 1]."});
            bad.push_back({[a](rays_in_t &x)
                           {
                               arma::cube *c[5] = {&x.cl.delay, &x.cl.aod, &x.cl.eod, &x.cl.aoa, &x.cl.eoa};
                               (*c[a])(2, 7, x.method == "38.901_MF" ? x.i_freq : 0) = NAN;
                           },
                           "Input '" + std::string(geo[a]) + "' must be finite."});
        }
        if (mf)
            bad.push_back({[](rays_in_t &x) { x.cl.delay = x.cl.delay.slices(0, 0); },
                           "Input 'delay' must have size [n_link, n_slot, n_freq] for method '38.901_MF', else "
                           "[n_link, n_slot, 1]."});
        else if (!is_qd) // "quadriga" takes the one-frequency inputs of the "38.901" base as they are
        {
            bad.push_back({[](rays_in_t &x) { x.fc = arma::vec{3.5e9, 28e9}; },
                           "Method '38.901' supports one frequency only; use '38.901_MF' or 'quadriga' for several "
                           "frequencies."});
            bad.push_back({[qd](rays_in_t &x) { qd(x), x.rnd(0, 3371, 0) = 1.5; },
                           "Values of block 14 of 'rnd_ray' must be in [0, 1]."});
        }
        else // all frequencies are used
        {
            bad.push_back({[](rays_in_t &x) { x.rnd = arma::join_slices(x.rnd, x.rnd); },
                           "Input 'rnd_ray' must have size [n_link, 3372, n_freq] for the TR methods or [n_link, "
                           "3372, 1] for 'quadriga'."});
            bad.push_back({[](rays_in_t &x) { x.cl.power(1, 4, 1) = NAN; }, "Input 'power' must be finite and >= 0."});
            bad.push_back({[](rays_in_t &x) { x.st.mu(2, 1, 5) = NAN; }, "Input 'lsp_mu' must be finite."});
            bad.push_back({[](rays_in_t &x) { x.st.sigma(1, 1, 7) = -1.0; },
                           "Input 'lsp_sigma' must be finite and >= 0."});
        }

        // Values that are not used: other frequencies and block 14 for the TR methods, other LSP slices, heights and
        // permittivity without the ground reflection
        const std::vector<mut_t> good = {
            [](rays_in_t &) {},
            [](rays_in_t &x)
            {
                const arma::uword k = 1 - x.i_freq;
                if (x.fc.n_elem == 2 && x.method != "quadriga")
                    x.cl.power(1, 4, k) = NAN, x.st.mu(2, k, 5) = NAN, x.st.sigma(1, k, 7) = -1.0,
                    x.rnd(2, 1000, k) = NAN, x.cl.delay(2, 7, k) = NAN;
            },
            [](rays_in_t &x)
            { x.st.mu(0, x.i_freq, 0) = NAN, x.st.sigma(0, x.i_freq, 3) = NAN, x.f.lsp(1, 0, 7) = NAN; },
            [](rays_in_t &x) { x.rnd(0, 3371, x.i_freq) = (x.method == "quadriga") ? 1.0 : 1.5; },
            [](rays_in_t &x) { x.sp["GR_enabled"] = 0.0, x.f.use_eps = false, x.f.rx(1, 2) = -0.1; },
            [](rays_in_t &x) { x.sp["GR_enabled"] = 0.0, x.f.eps_re = arma::mat(1, 1, arma::fill::zeros); }};
        for (size_t i = 0; i < bad.size() + good.size(); ++i)
        {
            rays_in_t x = base;
            if (i < bad.size())
                bad[i].first(x);
            else
                good[i - bad.size()](x);
            INFO("method " << method << ", case " << i);
            if (i < bad.size())
            {
                std::vector<quadriga_lib::gbsm_rays> rays;
                CHECK_THROWS_MATCHES(call(x, rays), std::invalid_argument, Message(bad[i].second));
            }
            else
            {
                if (x.sp["GR_enabled"] == 0.0) // the slots of ssf_clusters change with the ground reflection
                    x.cl = run(x.sp, x.f, x.method);
                CHECK_NOTHROW(call(x));
            }
        }

        // The output is untouched on error
        std::vector<quadriga_lib::gbsm_rays> rays(2);
        rays[0].delay = arma::vec(4, arma::fill::value(7.0));
        rays_in_t x = base;
        x.cl.power(0, 0, x.i_freq) = -1.0;
        CHECK_THROWS_AS(call(x, rays), std::invalid_argument);
        CHECK(rays.size() == 2);
        CHECK(arma::all(rays[0].delay == 7.0));
    }
}
