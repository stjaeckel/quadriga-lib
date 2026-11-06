// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of the class gbsm_rays: sizes, validation with one message per failed check, the scatterer distance scales
// b_med and b_eff, and the self-tests of the shared ray fixtures of qd_test_rays.hpp (make_rays, tr_offsets), which
// the tests of ssf_rays, ssf_paths, gbsm_rays_from_paths, qrt_paths_from_scatterers and gbsm_single_link_channel
// build on. The scatterer solver is tested in test_gbsm_rays_solve_scatterers.cpp.
// Oracles: the hand-made objects of make_rays, the TR 38.901 Table 7.5-3 values and the 6-decimal offsets for
// M != 20 (rule of tr_offsets) computed independently with NumPy, a sorted cumulative sum (weighted median) and the
// closed form of b_eff in this file.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"
#include "qd_test_diag.hpp"
#include "qd_test_rays.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
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
    SECTION("5 power rows")
    fails([&](R &r) { r.power.zeros(n_ray - 1, 2); }, "Input 'rays.power' must have size [n_ray, n_freq].");
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
    SECTION("8 path_type two GR rays")
    fails([](R &r) { r.path_type(4) = 1; }, "Input 'rays.path_type' can hold at most one LOS and one GR ray.");
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
    SECTION("11 scatterer sizes [0, n_ray]: no elements, but n_ray columns")
    {
        const std::string m = "Inputs 'rays.fbs_pos' and 'rays.lbs_pos' must both be empty or both have size [3, n_ray].";
        fails([&](R &r) { r.fbs_pos.set_size(0, n_ray), r.lbs_pos.set_size(0, n_ray); }, m);
        R r = base;
        r.fbs_pos.set_size(0, n_ray), r.lbs_pos.set_size(0, n_ray);
        arma::mat pg;
        arma::vec pl;
        arma::cube M;
        CHECK_THROWS_MATCHES(quadriga_lib::ssf_paths(r, arma::vec(2, arma::fill::zeros), pg, pl, M, "drift"),
                             std::invalid_argument, Catch::Matchers::Message(m));
    }
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
        CHECK(arma::all(r.path_type == arma::u32_vec({0, 1, 2, 2, 2, 2, 2, 2, 2})));
        CHECK(arma::all(r.tap == arma::u32_vec({0, 3, 6, 6, 9, 12, 15, 18, 21})));

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

namespace
{
    // Distance of ray i from end e (0 Tx, 1 Rx): NLOS end to FBS (Tx) or LBS (Rx), LOS d3D, GR length via the image
    double end_dist(const quadriga_lib::gbsm_rays &r, arma::uword i, int e)
    {
        const arma::vec Ri = {r.rx_pos(0), r.rx_pos(1), -r.rx_pos(2)};
        if (r.path_type(i) == 0)
            return len3(r.rx_pos - r.tx_pos);
        if (r.path_type(i) == 1)
            return len3(Ri - r.tx_pos);
        return e == 0 ? len3(r.fbs_pos.col(i) - r.tx_pos) : len3(r.lbs_pos.col(i) - r.rx_pos);
    }

    // Weighted median of (value, weight) pairs: sorted by value (ascending, or descending for the upper median;
    // ties keep the ray order), the first value whose cumulative weight reaches half the total; NaN without pairs
    double wmedian(std::vector<std::pair<double, double>> v, bool upper = false)
    {
        std::stable_sort(v.begin(), v.end(), [upper](const auto &a, const auto &b)
                         { return upper ? a.first > b.first : a.first < b.first; });
        double total = 0.0, cum = 0.0;
        for (const auto &x : v)
            total += x.second;
        for (const auto &x : v)
            if (2.0 * (cum += x.second) >= total)
                return x.first;
        return arma::datum::nan;
    }

    // b_med oracle: NLOS rays with power > 0 at k (ray 'skip' left out); 'mode' 0 lower, 1 upper, 2 unweighted
    double oracle_b_med(const quadriga_lib::gbsm_rays &r, arma::uword k, int e, int mode = 0, arma::uword skip = 999)
    {
        std::vector<std::pair<double, double>> v;
        for (arma::uword i = 0; i < r.n_ray(); ++i)
            if (r.path_type(i) == 2 && r.power(i, k) > 0.0 && i != skip)
                v.push_back({end_dist(r, i, e), mode == 2 ? 1.0 : r.power(i, k)});
        return wmedian(v, mode == 1);
    }

    // b_eff oracle: (sum P / sum (P / b^p))^(1/p) over the rays with P > 0 at k; p = 2 is b_eff, p = -2 the
    // power-weighted quadratic mean, p = -1 the power-weighted arithmetic mean
    double oracle_b_eff(const quadriga_lib::gbsm_rays &r, arma::uword k, int e, double p = 2.0)
    {
        double sp = 0.0, sq = 0.0;
        for (arma::uword i = 0; i < r.n_ray(); ++i)
            if (r.power(i, k) > 0.0)
                sp += r.power(i, k), sq += r.power(i, k) / std::pow(end_dist(r, i, e), p);
        return sp > 0.0 ? std::pow(sp / sq, 1.0 / p) : arma::datum::nan;
    }

    bool both_nan(const arma::vec &b) { return b.n_elem == 2 && std::isnan(b(0)) && std::isnan(b(1)); }
}

TEST_CASE("gbsm_rays - b_med is the lower weighted median of the NLOS end distances")
{
    // Rays: 0 LOS, 1 GR, 2 to 7 NLOS with ray 4 at zero power; frequency 0 with the powers of the fixture,
    // frequency 1 with the powers 1, 1, 1, 1, 4 on the kept NLOS rays in ascending Tx distance
    quadriga_lib::gbsm_rays r = make_rays(31, true, true, 6, 2, "38.901", true);
    std::vector<arma::uword> kept = {2, 3, 5, 6, 7};
    std::sort(kept.begin(), kept.end(), [&](arma::uword a, arma::uword b)
              { return end_dist(r, a, 0) < end_dist(r, b, 0); });
    const double pw[5] = {1.0, 1.0, 1.0, 1.0, 4.0};
    for (int q = 0; q < 5; ++q)
        r.power(kept[q], 1) = pw[q];
    r.fallback.zeros(r.n_ray());
    r.fallback(kept[0]) = 1; // the nearest ray at the Tx is a fallback ray; it counts

    // Properties of the oracle: Tx and Rx differ; lower, upper and unweighted median differ; leaving out the
    // fallback ray changes the lower median
    for (arma::uword k = 0; k < 2; ++k)
        CHECK(std::abs(oracle_b_med(r, k, 0) - oracle_b_med(r, k, 1)) > 1.0);
    const double lo = oracle_b_med(r, 1, 0), up = oracle_b_med(r, 1, 0, 1), uw = oracle_b_med(r, 1, 0, 2);
    CHECK(lo == end_dist(r, kept[3], 0));
    CHECK(up == end_dist(r, kept[4], 0));
    CHECK(uw == end_dist(r, kept[2], 0));
    CHECK(oracle_b_med(r, 1, 0, 0, kept[0]) != lo);

    for (arma::uword k = 0; k < 2; ++k)
    {
        INFO("i_freq " << k);
        const arma::vec b = r.b_med(k);
        REQUIRE(b.n_elem == 2);
        CHECK(std::abs(b(0) - oracle_b_med(r, k, 0)) < 1e-12);
        CHECK(std::abs(b(1) - oracle_b_med(r, k, 1)) < 1e-12);
        QD_DIAG("# diag: b_med i_freq %u: Tx %.6f (oracle %.6f), Rx %.6f (oracle %.6f)\n", (unsigned)k, b(0),
                oracle_b_med(r, k, 0), b(1), oracle_b_med(r, k, 1));
    }
    CHECK(arma::approx_equal(r.b_med(), r.b_med(0), "absdiff", 0.0));

    // LOS and GR do not count
    quadriga_lib::gbsm_rays q = r;
    q.power(0, 0) = 100.0, q.power(1, 0) = 50.0;
    CHECK(arma::approx_equal(q.b_med(0), r.b_med(0), "absdiff", 0.0));

    // Cumulative power at exactly half: powers 1, 1 give the nearer ray at each end
    q.power.col(1).zeros(), q.power(3, 1) = 1.0, q.power(6, 1) = 1.0;
    for (int e = 0; e < 2; ++e)
    {
        const double b3 = end_dist(q, 3, e), b6 = end_dist(q, 6, e);
        REQUIRE(b3 != b6);
        CHECK(oracle_b_med(q, 1, e) == std::min(b3, b6));
        CHECK(std::abs(q.b_med(1)(e) - std::min(b3, b6)) < 1e-12);
    }

    // NaN: without NLOS power at i_freq, without scatterers, n_ray = 0
    q.power.col(1).zeros(), q.power(0, 1) = 1.0;
    CHECK(both_nan(q.b_med(1)));
    CHECK(both_nan(make_rays(31, true, true, 6, 2, "38.901").b_med(0)));
    CHECK(both_nan(make_rays(31, false, false, 0, 2, "38.901", true).b_med(1)));
}

TEST_CASE("gbsm_rays - b_eff is the power-weighted inverse-square mean of the end distances")
{
    const quadriga_lib::gbsm_rays r = make_rays(32, true, true, 6, 2, "quadriga", true);
    REQUIRE(r.power(0, 0) > 0.0);
    REQUIRE(r.power(1, 1) > 0.0);
    for (arma::uword k = 0; k < 2; ++k)
        for (int e = 0; e < 2; ++e)
        {
            INFO("i_freq " << k << ", end " << e);
            const double o = oracle_b_eff(r, k, e);
            CHECK(std::abs(o - oracle_b_eff(r, k, e, -2.0)) > 0.01 * o);
            CHECK(std::abs(o - oracle_b_eff(r, k, e, -1.0)) > 0.01 * o);
            CHECK(std::abs(r.b_eff(k)(e) - o) < 1e-12 * o);
            QD_DIAG("# diag: b_eff i_freq %u end %d: %.6f (oracle %.6f, quadratic mean %.6f)\n", (unsigned)k, e,
                    r.b_eff(k)(e), o, oracle_b_eff(r, k, e, -2.0));
        }
    CHECK(std::abs(oracle_b_eff(r, 0, 0) - oracle_b_eff(r, 1, 0)) > 1e-3); // i_freq selects the powers
    CHECK(arma::approx_equal(r.b_eff(), r.b_eff(0), "absdiff", 0.0));

    // LOS only: d3D at both ends; GR only: the GR length
    quadriga_lib::gbsm_rays q = r;
    q.power.zeros(), q.power(0, 0) = 0.7, q.power(1, 1) = 0.2;
    const double d3D = len3(r.rx_pos - r.tx_pos), d_gr = end_dist(r, 1, 0);
    CHECK(arma::approx_equal(q.b_eff(0), arma::vec{d3D, d3D}, "reldiff", 1e-12));
    CHECK(arma::approx_equal(q.b_eff(1), arma::vec{d_gr, d_gr}, "reldiff", 1e-12));

    // NaN: no power at i_freq, without scatterers, n_ray = 0
    q.power.col(1).zeros();
    CHECK(both_nan(q.b_eff(1)));
    CHECK(both_nan(make_rays(32, true, true, 6, 2, "quadriga").b_eff(0)));
    CHECK(both_nan(make_rays(32, false, false, 0, 2, "quadriga", true).b_eff(0)));
}

TEST_CASE("gbsm_rays - b_med and b_eff validate the object and i_freq")
{
    quadriga_lib::gbsm_rays r = make_rays(33, true, true, 5, 2, "38.901_MF", true);
    CHECK_NOTHROW(r.b_med(1));
    CHECK_NOTHROW(r.b_eff(1));
    auto msg = [](const std::function<void()> &f)
    {
        try
        {
            f();
        }
        catch (const std::invalid_argument &e)
        {
            return std::string(e.what());
        }
        return std::string("no throw");
    };
    const std::string m_range = "Input 'i_freq' is out of range for 2 frequencies.";
    CHECK(msg([&] { r.b_med(2); }) == m_range);
    CHECK(msg([&] { r.b_eff(2); }) == m_range);
    r.lbs_pos.reset();
    const std::string m_obj =
        "Inputs 'rays.fbs_pos' and 'rays.lbs_pos' must both be empty or both have size [3, n_ray].";
    CHECK(msg([&] { r.b_med(0); }) == m_obj);
    CHECK(msg([&] { r.b_eff(0); }) == m_obj);
    CHECK(msg([] { quadriga_lib::gbsm_rays().b_med(0); }) == "Input 'rays.tx_pos' must have 3 elements.");
}
