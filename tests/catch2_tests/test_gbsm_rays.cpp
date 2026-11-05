// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of the class gbsm_rays: sizes, validation with one message per failed check, and the self-tests of the
// shared ray fixtures of qd_test_rays.hpp (make_rays, tr_offsets), which the tests of ssf_rays, ssf_paths,
// gbsm_rays_from_paths, qrt_paths_from_scatterers and gbsm_single_link_channel build on. The scatterer solver and
// b_med, b_eff are tested in test_gbsm_rays_solve_scatterers.cpp and in cases of this file added with them.
// Oracles: the hand-made objects of make_rays, the TR 38.901 Table 7.5-3 values and the 6-decimal offsets for
// M != 20 (rule of tr_offsets) computed independently with NumPy.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_gbsm.hpp"
#include "qd_test_diag.hpp"
#include "qd_test_rays.hpp"

#include <cmath>
#include <functional>
#include <string>
#include <vector>

QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

TEST_CASE("gbsm_rays - is_valid accepts a valid object and one with n_ray = 0; a default-constructed object reports check 2")
{
    for (bool sc : {false, true})
    {
        INFO("with_scatterers " << sc);
        CHECK(make_rays(7, true, true, 5, 2, "38.901", sc).is_valid().empty());
        const quadriga_lib::gbsm_rays r0 = make_rays(7, false, false, 0, 2, "quadriga", sc);
        REQUIRE(r0.n_ray() == 0);
        CHECK(r0.power.n_rows == 0);
        CHECK(r0.power.n_cols == 2);
        CHECK(r0.is_valid().empty());
    }
    const quadriga_lib::gbsm_rays d;
    CHECK(d.is_valid() == "Input 'rays.tx_pos' must have 3 elements.");
}

TEST_CASE("gbsm_rays - is_valid reports each failed check")
{
    quadriga_lib::gbsm_rays base = make_rays(11, true, true, 5, 2, "quadriga", true);
    const arma::uword n_ray = base.n_ray();
    base.fallback.zeros(n_ray), base.fallback(5) = 1;
    REQUIRE(base.is_valid().empty());
    const double nan = arma::datum::nan, inf = arma::datum::inf;

    auto fails = [&](const std::function<void(quadriga_lib::gbsm_rays &)> &edit, const std::string &msg)
    {
        quadriga_lib::gbsm_rays r = base;
        edit(r);
        CHECK(r.is_valid() == msg);
    };
    using R = quadriga_lib::gbsm_rays;

    SECTION("1 method")
    fails([](R &r) { r.method = "38.901 "; }, "Input 'rays.method' must be '38.901', '38.901_MF' or 'quadriga'.");
    SECTION("2 tx_pos size")
    fails([](R &r) { r.tx_pos.resize(4); }, "Input 'rays.tx_pos' must have 3 elements.");
    SECTION("2 tx_pos finite")
    fails([&](R &r) { r.tx_pos(1) = nan; }, "Input 'rays.tx_pos' must be finite.");
    SECTION("2 rx_pos size")
    fails([](R &r) { r.rx_pos.resize(2); }, "Input 'rays.rx_pos' must have 3 elements.");
    SECTION("2 rx_pos finite")
    fails([&](R &r) { r.rx_pos(2) = inf; }, "Input 'rays.rx_pos' must be finite.");
    SECTION("3 center_frequency empty")
    fails([](R &r) { r.center_frequency.reset(); }, "Input 'rays.center_frequency' cannot be empty.");
    SECTION("3 center_frequency value")
    fails([](R &r) { r.center_frequency(1) = 0.0; }, "Input 'rays.center_frequency' must be finite and > 0.");
    SECTION("4 delay finite")
    fails([&](R &r) { r.delay(3) = nan; }, "Input 'rays.delay' must be finite.");
    SECTION("5 power size")
    fails([&](R &r) { r.power.zeros(n_ray, 3); }, "Input 'rays.power' must have size [n_ray, n_freq].");
    SECTION("5 power value")
    fails([](R &r) { r.power(4, 1) = -1e-9; }, "Input 'rays.power' must be finite and >= 0.");
    SECTION("6 aod size")
    fails([&](R &r) { r.aod.resize(n_ray - 1); }, "Input 'rays.aod' must have n_ray elements.");
    SECTION("6 aod finite")
    fails([&](R &r) { r.aod(0) = nan; }, "Input 'rays.aod' must be finite.");
    SECTION("6 eod size")
    fails([&](R &r) { r.eod.resize(n_ray + 1); }, "Input 'rays.eod' must have n_ray elements.");
    SECTION("6 eod finite")
    fails([&](R &r) { r.eod(2) = inf; }, "Input 'rays.eod' must be finite.");
    SECTION("6 aoa size")
    fails([](R &r) { r.aoa.reset(); }, "Input 'rays.aoa' must have n_ray elements.");
    SECTION("6 aoa finite")
    fails([&](R &r) { r.aoa(4) = nan; }, "Input 'rays.aoa' must be finite.");
    SECTION("6 eoa size")
    fails([&](R &r) { r.eoa.resize(n_ray - 1); }, "Input 'rays.eoa' must have n_ray elements.");
    SECTION("6 eoa finite")
    fails([&](R &r) { r.eoa(6) = -inf; }, "Input 'rays.eoa' must be finite.");
    SECTION("7 M size")
    fails([&](R &r) { r.M.zeros(8, n_ray, 1); }, "Input 'rays.M' must have size [8, n_ray, n_freq].");
    SECTION("7 M finite")
    fails([&](R &r) { r.M(7, 2, 1) = nan; }, "Input 'rays.M' must be finite.");
    SECTION("8 path_type size")
    fails([&](R &r) { r.path_type.resize(n_ray - 1); }, "Input 'rays.path_type' must have n_ray elements.");
    SECTION("8 path_type value")
    fails([](R &r) { r.path_type(3) = 3; }, "Values of 'rays.path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS).");
    SECTION("8 path_type two LOS rays")
    fails([](R &r) { r.path_type(4) = 0; }, "Input 'rays.path_type' can hold at most one LOS and one GR ray.");
    SECTION("9 tap size")
    fails([&](R &r) { r.tap.resize(n_ray + 1); }, "Input 'rays.tap' must have n_ray elements.");
    SECTION("10 permittivity sizes")
    fails([](R &r) { r.gr_eps_im.reset(); },
          "Inputs 'rays.gr_eps_re' and 'rays.gr_eps_im' must both be empty or both have n_freq elements.");
    SECTION("10 gr_eps_re value")
    fails([](R &r) { r.gr_eps_re(1) = 0.99; }, "Input 'rays.gr_eps_re' must be finite and >= 1.");
    SECTION("10 gr_eps_im value")
    fails([](R &r) { r.gr_eps_im(0) = -0.01; }, "Input 'rays.gr_eps_im' must be finite and >= 0.");
    SECTION("11 scatterer sizes")
    fails([](R &r) { r.lbs_pos.reset(); },
          "Inputs 'rays.fbs_pos' and 'rays.lbs_pos' must both be empty or both have size [3, n_ray].");
    SECTION("11 fbs_pos finite")
    fails([&](R &r) { r.fbs_pos(2, 4) = nan; }, "Input 'rays.fbs_pos' must be finite.");
    SECTION("11 lbs_pos finite")
    fails([&](R &r) { r.lbs_pos(0, 1) = inf; }, "Input 'rays.lbs_pos' must be finite.");
    SECTION("12 fallback size")
    fails([&](R &r) { r.fallback.resize(n_ray - 1); }, "Input 'rays.fallback' must be empty or have n_ray elements.");
    SECTION("12 fallback value")
    fails([](R &r) { r.fallback(2) = 2; }, "Values of 'rays.fallback' must be 0 or 1.");
}

TEST_CASE("gbsm_rays - n_ray, n_freq and empty")
{
    const quadriga_lib::gbsm_rays r = make_rays(3, true, true, 4, 3, "38.901_MF");
    CHECK(r.n_ray() == 6);
    CHECK(r.n_freq() == 3);
    CHECK(!r.empty());
    const quadriga_lib::gbsm_rays r0 = make_rays(3, false, false, 0, 2, "38.901");
    CHECK(r0.n_ray() == 0);
    CHECK(r0.n_freq() == 2);
    CHECK(r0.empty());
    const quadriga_lib::gbsm_rays d;
    CHECK(d.n_ray() == 0);
    CHECK(d.n_freq() == 0);
    CHECK(d.empty());
}

TEST_CASE("qd_test_rays - make_rays gives valid objects with the stated properties")
{
    for (bool sc : {false, true})
        for (bool los : {false, true})
            for (bool gr : {false, true})
                for (arma::uword n_nlos : {0, 1, 4, 7})
                    for (arma::uword n_freq : {1, 3})
                    {
                        INFO("scatterers " << sc << ", los " << los << ", gr " << gr << ", n_nlos " << n_nlos);
                        const quadriga_lib::gbsm_rays r = make_rays(5 + (unsigned)n_nlos, los, gr, n_nlos, n_freq,
                                                                    qd_methods[n_nlos % 3], sc);
                        CHECK(r.is_valid().empty());
                        REQUIRE(r.n_ray() == (los ? 1 : 0) + (gr ? 1 : 0) + n_nlos);
                        CHECK(r.gr_eps_re.n_elem == (gr ? n_freq : 0));
                        CHECK(r.gr_eps_im.n_elem == (gr ? n_freq : 0));
                        CHECK(r.fbs_pos.n_cols == (sc ? r.n_ray() : 0));
                        CHECK(r.fallback.n_elem == 0);
                    }

    const arma::vec T = {3.0, -4.0, 25.0}, R = {120.0, 45.0, 1.5}, D = R - T;
    for (bool sc : {false, true})
    {
        INFO("with_scatterers " << sc);
        const quadriga_lib::gbsm_rays r = make_rays(21, true, true, 7, 2, "quadriga", sc);
        CHECK(arma::approx_equal(r.tx_pos, T, "absdiff", 0.0));
        CHECK(arma::approx_equal(r.rx_pos, R, "absdiff", 0.0));
        CHECK(arma::all(r.path_type == arma::uvec({0, 1, 2, 2, 2, 2, 2, 2, 2})));
        CHECK(arma::all(r.tap == arma::uvec({0, 3, 6, 6, 9, 12, 15, 18, 21})));

        // The 1 : 4 tap pair (rays 2, 3), the zero-power ray (4), the zero-excess ray (5, without scatterers)
        CHECK(r.delay(2) != r.delay(3));
        for (arma::uword k = 0; k < 2; ++k)
        {
            CHECK(r.power(2, k) > 0.0);
            CHECK(std::abs(r.power(3, k) / r.power(2, k) - 4.0) < 1e-12);
            CHECK(r.power(4, k) == 0.0);
        }
        for (arma::uword i = 0; i < r.n_ray(); ++i)
            CHECK((r.power(i, 0) > 0.0) == (i != 4));
        CHECK((r.delay(5) == 0.0) == !sc);
        if (!sc)
            CHECK(ang_dist(unit_vec(r.aod(5), r.eod(5)), D) > 0.1);
        if (!sc)
            continue;

        // Scatterers: delay from the polyline, angles towards the first and last scatterer, LOS at the midpoint,
        // GR at the specular point (on z = 0 and on the line from the Tx to the image of the Rx)
        const double d3D = len3(D);
        std::vector<double> b_t, b_r;
        int n_2m = 0;
        for (arma::uword i = 0; i < r.n_ray(); ++i)
        {
            INFO("ray " << i);
            const arma::vec F = r.fbs_pos.col(i), L = r.lbs_pos.col(i);
            const double len = polyline(T, F, L, R);
            CHECK(std::abs(r.delay(i) * c0 - (len - d3D)) < 1e-12 * len);
            CHECK(ang_dist(unit_vec(r.aod(i), r.eod(i)), F - T) < 1e-12);
            CHECK(ang_dist(unit_vec(r.aoa(i), r.eoa(i)), L - R) < 1e-12);
            if (r.path_type(i) == 2)
                b_t.push_back(len3(F - T)), b_r.push_back(len3(L - R));
            n_2m += (std::abs(len3(F - T) - 2.0) < 1e-12 && std::abs(len3(L - R) - 2.0) < 1e-12);
        }
        CHECK(n_2m == 1);
        std::sort(b_t.begin(), b_t.end()), std::sort(b_r.begin(), b_r.end());
        double d_max = 0.0;
        for (size_t i = 0; i < b_t.size(); ++i)
            d_max = std::max(d_max, std::abs(b_t[i] - b_r[i]));
        CHECK(d_max > 1.0);
        CHECK(arma::approx_equal(r.fbs_pos.col(0), 0.5 * (T + R), "absdiff", 1e-12));
        CHECK(arma::approx_equal(r.lbs_pos.col(0), 0.5 * (T + R), "absdiff", 1e-12));
        const arma::vec Gp = r.fbs_pos.col(1), Ri = {R(0), R(1), -R(2)};
        CHECK(std::abs(Gp(2)) < 1e-12);
        CHECK(ang_dist(Gp - T, Ri - T) < 1e-12);
        CHECK(arma::approx_equal(r.lbs_pos.col(1), Gp, "absdiff", 0.0));
    }
}

TEST_CASE("qd_test_rays - tr_offsets has the properties of the ray offsets")
{
    // 6-decimal values in ascending order (rule of tr_offsets evaluated independently with NumPy)
    const std::vector<std::pair<arma::uword, std::vector<double>>> quoted = {
        {2, {-1.0, 1.0}},
        {3, {-1.379551, 0.420279, 0.959273}},
        {4, {-1.115204, -0.847110, 0.743054, 1.219260}},
        {6, {-1.197040, -0.945308, -0.749898, 0.547806, 0.994948, 1.349492}},
        {21, {-2.179417, -1.549424, -1.180902, -0.919431, -0.716619, -0.550909, -0.410804, -0.289439, -0.182387,
              -0.086626, 0.0}},
        {26, {-2.288001, -1.678661, -1.322221, -1.069322, -0.873159, -0.712881, -0.577369, -0.459983, -0.356441,
              -0.263819, -0.180033, -0.103542, -0.033177}}};
    const double rms_table = tr_offsets(20).rms_table;
    CHECK(std::abs(rms_table - 1.000038419262) < 1e-12);

    for (arma::uword M = 1; M <= 26; ++M)
    {
        INFO("M " << M);
        const std::vector<double> a = tr_offsets(M).asc;
        REQUIRE(a.size() == M);
        double mean = 0.0, ms = 0.0;
        for (arma::uword m = 0; m < M; ++m)
            mean += a[m] / (double)M, ms += a[m] * a[m] / (double)M;
        CHECK(std::abs(mean) < 1e-12);
        CHECK(std::abs(std::sqrt(ms) - (M == 1 ? 0.0 : M < 20 ? 1.0 : rms_table)) < 1e-12);
        for (arma::uword m = 1; m < M; ++m)
            CHECK(a[m] > a[m - 1]);
        if (M > 20) // symmetric
            for (arma::uword m = 0; m < M; ++m)
                CHECK(std::abs(a[m] + a[M - 1 - m]) < 1e-12);
        for (const auto &q : quoted)
            if (q.first == M)
                for (size_t m = 0; m < q.second.size(); ++m) // M = 21 and 26: the negative half and 0
                    CHECK(std::abs(a[m] - q.second[m]) < 1e-6);
        QD_DIAG("# diag: tr_offsets M = %u: largest %.6f\n", (unsigned)M, a[M - 1]);
    }

    // Sub-cluster groups of Table 7.5-5 at the ray positions 0-9, 10-15, 16-19; together the 20 values
    const offsets_t o = tr_offsets(20);
    REQUIRE(o.groups.size() == 3);
    const std::vector<double> A1 = {-2.1551, -0.3715, -0.2492, -0.1413, -0.0447,
                                    0.0447, 0.1413, 0.2492, 0.3715, 2.1551};
    const std::vector<double> A2 = {-1.5195, -0.6797, -0.5129, 0.5129, 0.6797, 1.5195};
    const std::vector<double> A3 = {-1.1481, -0.8844, 0.8844, 1.1481};
    CHECK(o.groups[0] == A1);
    CHECK(o.groups[1] == A2);
    CHECK(o.groups[2] == A3);
    std::vector<double> all = A1;
    all.insert(all.end(), A2.begin(), A2.end()), all.insert(all.end(), A3.begin(), A3.end());
    std::sort(all.begin(), all.end());
    CHECK(all == o.asc);
    CHECK(tr_offsets(19).groups.empty());
}
