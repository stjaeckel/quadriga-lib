// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of gbsm_rays::solve_scatterers (two-bounce solver): scatterers with exact angles and equal distances from
// both ends, the fallback along the ray's own directions with both scatterers at d_min, LOS, GR and zero-power rays,
// degenerate geometry, in-place writing and validation. b_med and b_eff are tested in test_gbsm_rays.cpp.
// Oracles: the polyline geometry of qd_test_rays.hpp (length, angles, specular point by the image method) and a
// bisection of the balanced condition 2·s + abs(D + s·w) = L in this file (D = T - R, w = u_d - u_a,
// L = abs(R - T) + c·delay), independent of the closed form of the library.
// Fixture: make_rays of qd_test_rays.hpp (Tx (3, -4, 25), Rx (120, 45, 1.5)), with positions, delays and angles
// overwritten where a case needs other geometry. Units: [m], [s], [rad].

#include <catch2/catch_test_macros.hpp>

#include "quadriga_gbsm.hpp"
#include "qd_test_diag.hpp"
#include "qd_test_rays.hpp"

#include <cmath>
#include <string>

QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

namespace
{
    using rays_t = quadriga_lib::gbsm_rays;

    // Balanced distance s of a ray: the root of g(s) = 2·s + abs(D + s·w) = L on [0, L/2] for L > abs(D); g is
    // non-decreasing (abs(w) <= 2) with g(0) = abs(D); 200 halvings reach the resolution of double for any L
    double bisect_s(const arma::vec &T, const arma::vec &R, const arma::vec &ud, const arma::vec &ua, double L)
    {
        const arma::vec D = T - R, w = ud - ua;
        double lo = 0.0, hi = 0.5 * L;
        for (int it = 0; it < 200; ++it)
        {
            const double mid = 0.5 * (lo + hi);
            (2.0 * mid + len3(D + mid * w) < L ? lo : hi) = mid;
        }
        return lo;
    }

    // Ray length L = abs(R - T) + c·delay of ray i
    double ray_len(const rays_t &r, arma::uword i) { return len3(r.rx_pos - r.tx_pos) + c0 * r.delay(i); }

    // Balanced distance of ray i by bisection
    double ray_s(const rays_t &r, arma::uword i)
    {
        return bisect_s(r.tx_pos, r.rx_pos, unit_vec(r.aod(i), r.eod(i)), unit_vec(r.aoa(i), r.eoa(i)), ray_len(r, i));
    }

    // Exact equality of two objects; the scatterer attributes (fbs_pos, lbs_pos, fallback) only if 'with_scatterers'
    bool same(const rays_t &a, const rays_t &b, bool with_scatterers)
    {
        auto eq = [](const auto &x, const auto &y)
        { return arma::size(x) == arma::size(y) && arma::all(arma::vectorise(x == y)); };
        bool ok = a.method == b.method && eq(a.tx_pos, b.tx_pos) && eq(a.rx_pos, b.rx_pos) &&
                  eq(a.center_frequency, b.center_frequency) && eq(a.delay, b.delay) && eq(a.power, b.power) &&
                  eq(a.aod, b.aod) && eq(a.eod, b.eod) && eq(a.aoa, b.aoa) && eq(a.eoa, b.eoa) && eq(a.M, b.M) &&
                  eq(a.path_type, b.path_type) && eq(a.tap, b.tap) && eq(a.gr_eps_re, b.gr_eps_re) &&
                  eq(a.gr_eps_im, b.gr_eps_im);
        if (with_scatterers)
            ok = ok && eq(a.fbs_pos, b.fbs_pos) && eq(a.lbs_pos, b.lbs_pos) && eq(a.fallback, b.fallback);
        return ok;
    }

    // Checks of a solved NLOS ray: polyline = L, angles towards the scatterers, equal distances from both ends
    void check_solved(const rays_t &r, arma::uword i)
    {
        const arma::vec F = r.fbs_pos.col(i), B = r.lbs_pos.col(i);
        const double L = ray_len(r, i), dL = polyline(r.tx_pos, F, B, r.rx_pos) - L;
        const double db = len3(F - r.tx_pos) - len3(B - r.rx_pos);
        CHECK(r.fallback(i) == 0);
        CHECK(std::abs(dL) < 1e-9);
        CHECK(ang_dist(unit_vec(r.aod(i), r.eod(i)), F - r.tx_pos) < 1e-10);
        CHECK(ang_dist(unit_vec(r.aoa(i), r.eoa(i)), B - r.rx_pos) < 1e-10);
        CHECK(std::abs(db) < 1e-9);
        QD_DIAG("%u,%.6f,%.3e,%.3e\n", (unsigned)i, L, dL, db);
    }

    // Checks of a fallback ray: both scatterers at d_min along the ray's own directions, ray lengthened
    void check_fallback(const rays_t &r, arma::uword i, double d_min)
    {
        const arma::vec F = r.fbs_pos.col(i), B = r.lbs_pos.col(i);
        CHECK(r.fallback(i) == 1);
        CHECK(arma::approx_equal(F, r.tx_pos + d_min * unit_vec(r.aod(i), r.eod(i)), "absdiff", 1e-12));
        CHECK(arma::approx_equal(B, r.rx_pos + d_min * unit_vec(r.aoa(i), r.eoa(i)), "absdiff", 1e-12));
        CHECK(polyline(r.tx_pos, F, B, r.rx_pos) > ray_len(r, i));
    }
}

TEST_CASE("gbsm_rays::solve_scatterers - solved rays keep their delay and angles exactly")
{
    QD_DIAG("\n# diag: solved rays, d_min = 2 m\nray,L,poly_minus_L,b_tx_minus_b_rx\n");
    for (unsigned seed : {1u, 2u, 3u})
        for (const std::string method : qd_methods)
        {
            INFO("seed " << seed << ", method " << method);
            rays_t r = make_rays(seed, true, true, 8, 2, method);
            const rays_t r0 = r;
            r.solve_scatterers();
            REQUIRE(r.fbs_pos.n_cols == r.n_ray());
            REQUIRE(r.fallback.n_elem == r.n_ray());
            int n_solved = 0;
            for (arma::uword i = 0; i < r.n_ray(); ++i)
            {
                INFO("ray " << i);
                if (r.path_type(i) != 2 || r.power(i, 0) == 0.0 || r.delay(i) == 0.0) // zero excess: next case
                    continue;
                REQUIRE(ray_s(r, i) > 2.0);
                check_solved(r, i), ++n_solved;
            }
            CHECK(n_solved == 6);
            CHECK(same(r, r0, false));
        }

    // Scatterers at equal distances from both ends are restored (the 2 m pair of the fixture, d_min below 2 m)
    rays_t s = make_rays(4, true, true, 6, 1, "38.901", true);
    const arma::mat fbs = s.fbs_pos, lbs = s.lbs_pos;
    s.solve_scatterers(1.0);
    const arma::uword i2 = 2 + 3; // NLOS ray j = 3
    REQUIRE(std::abs(len3(fbs.col(i2) - s.tx_pos) - 2.0) < 1e-12);
    check_solved(s, i2);
    CHECK(arma::approx_equal(s.fbs_pos.col(i2), fbs.col(i2), "absdiff", 1e-9));
    CHECK(arma::approx_equal(s.lbs_pos.col(i2), lbs.col(i2), "absdiff", 1e-9));
}

TEST_CASE("gbsm_rays::solve_scatterers - rays without a solution take the minimum-distance fallback")
{
    for (double d_min : {0.5, 2.0, 5.0})
    {
        INFO("d_min " << d_min);
        rays_t r = make_rays(5, false, false, 6, 1, "38.901");
        // Ray 3: zero excess (fixture); rays 0, 1, 4: small excess (3 mm, 3 cm, 1.5 m); ray 5 keeps its delay
        r.delay(0) = 1e-11, r.delay(1) = 1e-10, r.delay(4) = 5e-9;
        r.solve_scatterers(d_min);
        int n_fb = 0;
        for (arma::uword i : {0, 1, 3, 4, 5})
        {
            INFO("ray " << i);
            if (i == 3 || ray_s(r, i) < d_min) // bisection oracle: no solution with both distances >= d_min
                check_fallback(r, i, d_min), ++n_fb;
            else
                check_solved(r, i);
        }
        CHECK(r.fallback(0) == 1); // 3 mm excess with angles off the direct line
        CHECK(n_fb >= 3);
    }
}

TEST_CASE("gbsm_rays::solve_scatterers - the closed form agrees with a bisection of the polyline")
{
    // Random ends (every tenth geometry with d3D = 0), random directions (every seventh ray with u_a = -u_d, where
    // the quadratic degenerates), excess lengths 0.1 mm to 1 km; d_min = 1 mm, so that nearly all rays are solved.
    // Tolerance 1e-12 relative to L plus the coordinate size, whose rounding enters FBS - T.
    qd_uniform U(77);
    double w_max = 0.0;
    int n_cmp = 0;
    for (int g = 0; g < 40; ++g)
    {
        INFO("geometry " << g);
        rays_t r = make_rays(100 + (unsigned)g, false, false, 30, 1, "38.901");
        r.tx_pos = {U(-200, 200), U(-200, 200), U(0, 50)};
        r.rx_pos = (g % 10 == 9) ? arma::vec(r.tx_pos) : arma::vec{U(-200, 200), U(-200, 200), U(0, 50)};
        for (arma::uword i = 0; i < r.n_ray(); ++i)
        {
            r.aod(i) = U(-pi, pi), r.eod(i) = U(-1.5, 1.5);
            r.aoa(i) = (i % 7 == 6) ? wrap_rad(r.aod(i) + pi) : U(-pi, pi);
            r.eoa(i) = (i % 7 == 6) ? -r.eod(i) : U(-1.5, 1.5);
            r.delay(i) = std::pow(10.0, U(-4.0, 3.0)) / c0;
            r.power(i, 0) = 1.0;
        }
        r.solve_scatterers(1e-3);
        const double scale = len3(r.tx_pos) + len3(r.rx_pos);
        for (arma::uword i = 0; i < r.n_ray(); ++i)
        {
            INFO("ray " << i);
            const double L = ray_len(r, i), s = ray_s(r, i);
            CHECK(r.fallback(i) == (s < 1e-3 ? 1u : 0u));
            if (s < 1e-3)
                continue;
            const double d = std::abs(len3(r.fbs_pos.col(i) - r.tx_pos) - s) / (L + scale);
            CHECK(d < 1e-12);
            CHECK(std::abs(len3(r.lbs_pos.col(i) - r.rx_pos) - s) / (L + scale) < 1e-12);
            w_max = std::max(w_max, d), ++n_cmp;
        }
    }
    CHECK(n_cmp > 900); // excess below about 2 mm takes the fallback
    QD_DIAG("\n# diag: closed form vs bisection: %d rays, max relative difference %.2e\n", n_cmp, w_max);
}

TEST_CASE("gbsm_rays::solve_scatterers - LOS, GR and zero-power rays")
{
    for (double z : {-1.0, 25.0, 0.0}) // Tx height; -1: both ends on the ground (h_t + h_r = 0)
    {
        INFO("Tx height " << z);
        rays_t r = make_rays(6, true, true, 4, 2, "38.901");
        if (z < 0.0)
            r.tx_pos(2) = 0.0, r.rx_pos(2) = 0.0;
        else
            r.tx_pos(2) = z;
        r.solve_scatterers();
        const arma::vec mid = 0.5 * (r.tx_pos + r.rx_pos), G = specular_point(r.tx_pos, r.rx_pos);
        for (arma::uword i : {0, 4}) // LOS, zero-power NLOS
        {
            INFO("ray " << i);
            CHECK(arma::approx_equal(r.fbs_pos.col(i), mid, "absdiff", 1e-12));
            CHECK(arma::approx_equal(r.lbs_pos.col(i), mid, "absdiff", 1e-12));
            CHECK(r.fallback(i) == 0);
        }
        CHECK(arma::approx_equal(r.fbs_pos.col(1), G, "absdiff", 1e-12));
        CHECK(arma::approx_equal(r.lbs_pos.col(1), G, "absdiff", 1e-12));
        CHECK(r.fbs_pos(2, 1) == 0.0);
        CHECK(r.fallback(1) == 0);
        // Image method: the GR point lies on the line from the Tx to the image of the Rx
        const arma::vec Ri = {r.rx_pos(0), r.rx_pos(1), -r.rx_pos(2)};
        if (z > 0.0)
            CHECK(ang_dist(r.fbs_pos.col(1) - r.tx_pos, Ri - r.tx_pos) < 1e-12);
    }
}

TEST_CASE("gbsm_rays::solve_scatterers - only the scatterer attributes change and given scatterers are overwritten")
{
    rays_t r = make_rays(8, true, true, 7, 3, "quadriga", true);
    r.fallback.ones(r.n_ray());
    const rays_t r0 = r;
    r.solve_scatterers(1.0);
    CHECK(same(r, r0, false));
    CHECK(arma::all(r.fallback == 0));
    // NLOS rays with power and unequal end distances move to the balanced points
    int n_moved = 0;
    for (arma::uword i = 0; i < r.n_ray(); ++i)
        if (r.path_type(i) == 2 && r.power(i, 0) > 0.0 &&
            std::abs(len3(r0.fbs_pos.col(i) - r.tx_pos) - len3(r0.lbs_pos.col(i) - r.rx_pos)) > 1e-3)
            n_moved += len3(r.fbs_pos.col(i) - r0.fbs_pos.col(i)) > 1e-3;
    CHECK(n_moved == 5);
    // A second call with the same d_min gives the same result
    rays_t r2 = r;
    r2.solve_scatterers(1.0);
    CHECK(same(r2, r, true));
}

TEST_CASE("gbsm_rays::solve_scatterers - d3D = 0, negative excess and n_ray = 0")
{
    rays_t r = make_rays(9, false, false, 5, 1, "38.901");
    r.rx_pos = r.tx_pos;                                                          // d3D = 0
    r.delay(3) = 30e-9, r.aoa(3) = wrap_rad(r.aod(3) + pi), r.eoa(3) = -r.eod(3); // u_a = -u_d, abs(w) = 2
    r.delay(4) = -2e-9, r.aoa(4) = wrap_rad(r.aod(4) + pi), r.eoa(4) = -r.eod(4); // negative L, u_a = -u_d
    r.solve_scatterers(0.1);
    for (arma::uword i : {0, 1, 3})
    {
        INFO("ray " << i);
        // D = 0: 2·s + s·abs(w) = L
        const double L = c0 * r.delay(i);
        const double s = L / (2.0 + len3(unit_vec(r.aod(i), r.eod(i)) - unit_vec(r.aoa(i), r.eoa(i))));
        CHECK(std::abs(len3(r.fbs_pos.col(i) - r.tx_pos) - s) < 1e-12 * (L + len3(r.tx_pos)));
        check_solved(r, i);
    }
    CHECK(r.fbs_pos.is_finite());
    CHECK(r.lbs_pos.is_finite());
    check_fallback(r, 4, 0.1);

    // Negative excess with d3D > 0
    rays_t q = make_rays(9, false, false, 2, 1, "38.901");
    q.delay(1) = -1e-9;
    q.solve_scatterers(0.5);
    CHECK(q.fbs_pos.is_finite());
    check_fallback(q, 1, 0.5);

    rays_t e = make_rays(9, false, false, 0, 2, "quadriga");
    e.solve_scatterers();
    CHECK(e.fbs_pos.n_rows == 3);
    CHECK(e.fbs_pos.n_cols == 0);
    CHECK(e.lbs_pos.n_rows == 3);
    CHECK(e.lbs_pos.n_cols == 0);
    CHECK(e.fallback.n_elem == 0);
    CHECK(e.is_valid().empty());
}

TEST_CASE("gbsm_rays::solve_scatterers - pre-sized scatterer attributes are written in place and the object is unchanged after a throw")
{
    rays_t r = make_rays(10, true, true, 5, 2, "38.901_MF");
    const arma::uword n = r.n_ray();
    arma::mat F(3, n, arma::fill::value(-7.0)), B(3, n, arma::fill::value(-7.0));
    arma::uvec fb(n, arma::fill::value(1));
    r.fbs_pos.~Mat(), ::new (&r.fbs_pos) arma::mat(F.memptr(), 3, n, false, true); // strict external memory
    r.lbs_pos.~Mat(), ::new (&r.lbs_pos) arma::mat(B.memptr(), 3, n, false, true);
    r.fallback.~Col(), ::new (&r.fallback) arma::uvec(fb.memptr(), n, false, true);
    r.solve_scatterers();
    CHECK(r.fbs_pos.memptr() == F.memptr());
    CHECK(r.lbs_pos.memptr() == B.memptr());
    CHECK(r.fallback.memptr() == fb.memptr());
    rays_t ref = make_rays(10, true, true, 5, 2, "38.901_MF");
    ref.solve_scatterers();
    CHECK(same(r, ref, true));

    // Throws leave the object unchanged: invalid d_min, invalid attribute, GR with an end below ground
    rays_t a = ref;
    CHECK_THROWS_AS(a.solve_scatterers(0.0), std::invalid_argument);
    CHECK(same(a, ref, true));
    a.power(2, 1) = -1.0;
    const rays_t a0 = a;
    CHECK_THROWS_AS(a.solve_scatterers(), std::invalid_argument);
    CHECK(same(a, a0, true));
    rays_t g = ref;
    g.tx_pos(2) = -0.5;
    const rays_t g0 = g;
    CHECK_THROWS_AS(g.solve_scatterers(), std::invalid_argument);
    CHECK(same(g, g0, true));
}

TEST_CASE("gbsm_rays::solve_scatterers - a ray with power 0 at frequency 0 and > 0 at another is solved")
{
    rays_t r = make_rays(11, false, false, 4, 2, "38.901_MF");
    r.power(1, 0) = 0.0;
    REQUIRE(r.power(1, 1) > 0.0);
    r.solve_scatterers();
    CHECK(len3(r.fbs_pos.col(1) - 0.5 * (r.tx_pos + r.rx_pos)) > 1.0);
    check_solved(r, 1);
}

TEST_CASE("gbsm_rays::solve_scatterers - input validation")
{
    const rays_t base = make_rays(12, true, true, 5, 2, "38.901");
    rays_t ok = base;
    CHECK_NOTHROW(ok.solve_scatterers(2.0));

    auto msg = [](rays_t r, double d_min)
    {
        try
        {
            r.solve_scatterers(d_min);
        }
        catch (const std::invalid_argument &e)
        {
            return std::string(e.what());
        }
        return std::string("no throw");
    };

    SECTION("object")
    {
        rays_t r = base;
        r.method = "quadriga2";
        CHECK(msg(r, 2.0) == "Input 'rays.method' must be '38.901', '38.901_MF' or 'quadriga'.");
        r = base, r.M.zeros(8, 2, 2);
        CHECK(msg(r, 2.0) == "Input 'rays.M' must have size [8, n_ray, n_freq].");
        CHECK(msg(rays_t(), 2.0) == "Input 'rays.tx_pos' must have 3 elements.");
    }
    SECTION("d_min")
    {
        for (double d : {0.0, -1.0, arma::datum::nan, arma::datum::inf})
        {
            INFO("d_min " << d);
            CHECK(msg(base, d) == "Input 'd_min' must be finite and > 0.");
        }
        CHECK(msg(base, 1e-9) == "no throw");
    }
    SECTION("ground reflection with an end below z = 0")
    {
        const std::string m = "Ground reflection requires z >= 0 in 'rays.tx_pos' and 'rays.rx_pos'.";
        rays_t r = base;
        r.tx_pos(2) = -1e-6;
        CHECK(msg(r, 2.0) == m);
        r = base, r.rx_pos(2) = -3.0;
        CHECK(msg(r, 2.0) == m);
        rays_t q = make_rays(12, true, false, 5, 2, "38.901"); // without a GR ray an end below ground is accepted
        q.rx_pos(2) = -3.0;
        CHECK(msg(q, 2.0) == "no throw");
    }
}
