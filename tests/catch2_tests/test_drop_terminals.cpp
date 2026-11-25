// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// drop_terminals: the UT drop into the cells of any Tx side.
//
// Oracles: the cell of each UT by brute force (the nearest site over the sites and wrap images, then the sector wedge
// written out per site in the test); the areas of the cells by counting the points of a fine grid, for the shares of a
// drop in total; the rules of the scenario families typed from the TR tables (as in test_generate_layout). The layouts
// are hand-built with library calls, or Tx sides of generate_layout. Statistics use fixed seeds and bounds of five
// standard deviations of the sample size.

#include <catch2/catch_test_macros.hpp>

#include "qd_test_layout.hpp"
#include "quadriga_layout.hpp"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
const double D = arma::datum::pi / 180.0;
const double NaN = arma::datum::nan;
using opts = std::unordered_map<std::string, std::vector<double>>;

bool has(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }

// Tx side: BS1 at (0, 0) with three sectors (30, 150, 270 deg), BS2 at (230, 40) with one, BS3 at (60, 210) with two
// (0 and 180 deg), BS4 at (-150, 120) with three (0, 120, 240 deg); heights differ
quadriga_lib::layout hand_built()
{
    quadriga_lib::layout L;
    auto &T = L.tx;
    T.name = {"BS1", "BS1", "BS1", "BS2", "BS3", "BS3", "BS4", "BS4", "BS4"};
    T.sector = {0, 1, 2, 0, 0, 1, 0, 1, 2};
    T.i_antenna.zeros(9);
    T.offset = arma::regspace<arma::uvec>(0, 8);
    const double x[] = {0.0, 230.0, 60.0, -150.0}, y[] = {0.0, 40.0, 210.0, 120.0}, z[] = {25.0, 12.0, 30.0, 8.0};
    const int site[] = {0, 0, 0, 1, 2, 2, 3, 3, 3};
    const double head[] = {30.0, 150.0, 270.0, 0.0, 0.0, 180.0, 0.0, 120.0, 240.0};
    T.pos.set_size(3, 9);
    T.orientation.zeros(3, 9);
    for (int i = 0; i < 9; ++i)
    {
        T.pos(0, i) = x[site[i]], T.pos(1, i) = y[site[i]], T.pos(2, i) = z[site[i]];
        T.orientation(2, i) = head[i] * D;
    }
    return L;
}

// Node of the cell of a point of the hand-built layout: nearest site; BS1 and BS4: the 120 deg wedge around the
// boresight; BS3: the half plane in front of each sector; BS2: all
long long hand_cell(const quadriga_lib::layout &L, double x, double y)
{
    const auto S = qd_layout_sites(L.tx);
    arma::uword s = 0, k = 0;
    qd_layout_nearest(S, qd_layout_images(L.wrap), x, y, s, k);
    if (k != 0)
        return -1;
    const double az = std::atan2(y - S.xy(1, s), x - S.xy(0, s));
    const double half = s == 2 ? 90.0 * D : 60.0 * D;
    for (arma::uword n : S.nodes[s])
        if (S.nodes[s].size() == 1 || std::abs(qd_layout_wrap(az - L.tx.orientation(2, n))) < half)
            return (long long)n;
    return -1;
}

std::vector<arma::uword> count(const quadriga_lib::layout &L)
{
    std::vector<arma::uword> c(L.tx.n_node() + 1, 0);
    for (arma::uword i = 0; i < L.rx.n_node(); ++i)
    {
        const long long n = hand_cell(L, L.rx.pos(0, i), L.rx.pos(1, i));
        ++c[n < 0 ? L.tx.n_node() : (arma::uword)n];
    }
    return c;
}

const arma::vec BOUNDS = {-300.0, 400.0, -200.0, 350.0};
} // namespace

TEST_CASE("drop_terminals - input validation")
{
    const auto L0 = hand_built();
    auto T = [&](quadriga_lib::layout L, arma::uword n_ut, const arma::vec &bounds, const arma::vec &ut = {}, const arma::vec &indoor = {},
                 const opts &o = {}, arma::sword seed = 1)
    {
        const std::string e = qd_layout_throws([&] { quadriga_lib::drop_terminals(L, n_ut, bounds, ut, indoor, o, true, seed); });
        return e;
    };
    CHECK(T(L0, 2, BOUNDS) == "");

    SECTION("layout")
    {
        CHECK(has(T(quadriga_lib::layout(), 2, BOUNDS), "Tx"));
        auto M = L0; // The sectors of one site with different scenarios: layout::is_valid rejects it
        M.tx.scenario = {"UMa", "InF_SL", "UMa", "", "", "", "", "", ""};
        CHECK(has(T(M, 2, BOUNDS), "'layout'"));
        auto L = L0;
        L.tx.offset[1] = 5;
        CHECK(has(T(L, 2, BOUNDS), "'layout'"));
        L = L0;
        L.wrap = arma::mat{{100.0, 200.0}, {0.0, 0.0}}; // Dependent vectors
        CHECK(has(T(L, 2, {}), "'wrap'"));
    }
    SECTION("bounds")
    {
        CHECK(has(T(L0, 2, {}), "'bounds'"));
        CHECK(has(T(L0, 2, {0.0, 1.0, 0.0}), "'bounds'"));
        CHECK(has(T(L0, 2, {10.0, -10.0, 0.0, 100.0}), "'bounds'"));
        CHECK(has(T(L0, 2, {0.0, NaN, 0.0, 100.0}), "'bounds'"));
        auto L = L0;
        L.wrap = arma::mat(arma::vec{500.0, 0.0}); // One wrap vector still needs bounds
        CHECK(has(T(L, 2, {}), "'bounds'"));
        CHECK(T(L, 2, BOUNDS) == "");
    }
    SECTION("arguments the scenarios do not use")
    {
        CHECK(has(T(L0, 2, BOUNDS, {}, {0.5}), "'indoor ratio'"));
        CHECK(has(T(L0, 2, BOUNDS, {}, {}, {{"p_residential", {0.5}}}), "'p_residential'"));
        CHECK(has(T(L0, 2, BOUNDS, {}, {}, {{"xyz", {0.5}}}), "unknown"));
        CHECK(has(T(L0, 2, BOUNDS, {1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}), "'ut'"));
        CHECK(has(T(L0, 2, BOUNDS, {NaN, -1.0}), "'min distance'"));
    }
    SECTION("sectors of a site with the same heading")
    {
        auto L = L0;
        L.tx.orientation(2, 1) = L.tx.orientation(2, 0) + 1.0e-9;
        for (bool per_cell : {true, false})
        {
            const std::string e = qd_layout_throws([&] { quadriga_lib::drop_terminals(L, 2, BOUNDS, {}, {}, {}, per_cell, 1); });
            CHECK(has(e, "'BS1'"));
            CHECK(has(e, "same heading"));
        }
    }
    SECTION("sizes")
    {
        CHECK(has(T(L0, 2000000, BOUNDS), "'n_ut'"));                // 9 cells: 1.8e7 UTs
        CHECK(has(T(L0, 2049638230412172402ULL, BOUNDS), "'n_ut'")); // 9 x n_ut wraps around in 64 bits
    }
    SECTION("no room")
    {
        // The minimum distance excludes the whole drop area
        CHECK(has(T(L0, 2, {-10.0, 10.0, -10.0, 10.0}, {NaN, 100.0}), "room"));
    }
    SECTION("seed")
    {
        CHECK(has(T(L0, 2, BOUNDS, {}, {}, {}, -2), "'seed'"));
    }
    SECTION("a failed call leaves the layout unchanged")
    {
        auto L = quadriga_lib::generate_layout("38.901_UMi", 1, {}, {1.0}, {}, {}, {}, {}, {}, true, 3);
        const auto L1 = L;
        CHECK(has(qd_layout_throws([&] { quadriga_lib::drop_terminals(L, 2, {}, {}, {}, {{"p_residential", {0.5}}}); }), "'p_residential'"));
        CHECK(qd_layout_diff(L, L1) == "");
    }
}

TEST_CASE("drop_terminals - Voronoi cells and sector wedges")
{
    auto L = hand_built();
    quadriga_lib::drop_terminals(L, 6, BOUNDS, {}, {}, {}, true, 5);
    REQUIRE(L.rx.n_node() == 54);
    auto c = count(L);
    CHECK(c.back() == 0);
    c.pop_back();
    CHECK(std::all_of(c.begin(), c.end(), [](arma::uword v) { return v == 6; }));
    bool in = true;
    for (arma::uword i = 0; i < 54; ++i)
        in = in && L.rx.pos(0, i) >= BOUNDS[0] && L.rx.pos(0, i) <= BOUNDS[1] && L.rx.pos(1, i) >= BOUNDS[2] && L.rx.pos(1, i) <= BOUNDS[3];
    CHECK(in);
    // No scenario: UT height 1.5 m, no speed, no minimum distance, no O2I state
    CHECK(arma::all(L.rx.pos.row(2) == 1.5));
    CHECK(arma::all(L.rx.speed == 0.0));
    CHECK(L.rx.indoor.n_elem == 0);
    CHECK(L.is_valid() == "");
}

TEST_CASE("drop_terminals - uniform over the union of the cells")
{
    // Shares of the sites in a drop of 3000 UTs in total against the areas of their Voronoi cells in the bounds, and of the
    // part of one cell near its site (uniform within the cell), from a 1 m grid; bound: five standard deviations of the
    // binomial count
    auto L = hand_built();
    const arma::uword n = 3000;
    quadriga_lib::drop_terminals(L, n, BOUNDS, {}, {}, {}, false, 8);
    REQUIRE(L.rx.n_node() == n);
    const auto S = qd_layout_sites(L.tx);
    const arma::mat I = qd_layout_images(L.wrap);
    // Index 4: the part of the cell of BS2 within 60 m of it, for the uniformity within a cell
    std::vector<double> area(5, 0.0), got(5, 0.0);
    arma::uword s = 0, k = 0;
    for (double x = BOUNDS[0] + 0.5; x < BOUNDS[1]; x += 1.0)
        for (double y = BOUNDS[2] + 0.5; y < BOUNDS[3]; y += 1.0)
        {
            const double dist = qd_layout_nearest(S, I, x, y, s, k);
            area[s] += 1.0;
            area[4] += s == 1 && dist < 60.0 ? 1.0 : 0.0;
        }
    for (arma::uword i = 0; i < n; ++i)
    {
        const double dist = qd_layout_nearest(S, I, L.rx.pos(0, i), L.rx.pos(1, i), s, k);
        got[s] += 1.0;
        got[4] += s == 1 && dist < 60.0 ? 1.0 : 0.0;
    }
    const double total = (BOUNDS[1] - BOUNDS[0]) * (BOUNDS[3] - BOUNDS[2]);
    for (int j = 0; j < 5; ++j)
    {
        const double p = area[(size_t)j] / total;
        CHECK(std::abs(got[(size_t)j] / (double)n - p) < 5.0 * std::sqrt(p * (1.0 - p) / (double)n));
    }
}

TEST_CASE("drop_terminals - wrap vectors bound the drop")
{
    // The Tx side of a hexagonal layout with one ring (7 sites, 21 sectors): without bounds the wrapped cluster is the
    // drop area, every UT is nearest to image 0 of its site, 4 UTs per sector
    auto L = quadriga_lib::generate_layout("38.901_UMi", 0, {}, {1.0}, {}, {}, {}, {}, {}, true, 2);
    quadriga_lib::drop_terminals(L, 4, {}, {}, {}, {}, true, 3);
    REQUIRE(L.rx.n_node() == 84);
    const auto S = qd_layout_sites(L.tx);
    const arma::mat I = qd_layout_images(L.wrap);
    std::vector<arma::uword> c(21, 0);
    bool image0 = true;
    for (arma::uword i = 0; i < 84; ++i)
    {
        arma::uword s = 0, k = 0;
        qd_layout_nearest(S, I, L.rx.pos(0, i), L.rx.pos(1, i), s, k);
        image0 = image0 && k == 0;
        const double az = std::atan2(L.rx.pos(1, i) - S.xy(1, s), L.rx.pos(0, i) - S.xy(0, s));
        for (arma::uword n : S.nodes[s])
            if (std::abs(qd_layout_wrap(az - L.tx.orientation(2, n))) < 60.0 * D)
                ++c[n];
    }
    CHECK(image0);
    CHECK(std::all_of(c.begin(), c.end(), [](arma::uword v) { return v == 4; }));

    // With bounds as well: the part of the cluster in the bounds
    const arma::vec b = {-150.0, 50.0, -100.0, 120.0};
    quadriga_lib::drop_terminals(L, 100, b, {}, {}, {}, false, 3);
    REQUIRE(L.rx.n_node() == 100);
    bool in = true;
    for (arma::uword i = 0; i < 100; ++i)
        in = in && L.rx.pos(0, i) >= b[0] && L.rx.pos(0, i) <= b[1] && L.rx.pos(1, i) >= b[2] && L.rx.pos(1, i) <= b[3];
    CHECK(in);
}

TEST_CASE("drop_terminals - the Rx side is replaced")
{
    auto L = quadriga_lib::generate_layout("38.901_UMa", 2, {}, {1.0}, {}, {}, {}, {}, {}, true, 21);
    L.pairing = arma::umat{{0, 1}, {0, 1}};
    auto &K = L.links; // One valid link
    K.i_tx = {0}, K.i_rx = {0}, K.i_segment = {0}, K.image = {0}, K.i_conf = {0}, K.los = {1}, K.floor_id = {0};
    K.indoor = {0}, K.d2d_in = {0.0}, K.gbsm_seed = {5}, K.conf_names = {"UMa_LOS"};
    REQUIRE(L.is_valid() == "");
    const auto tx = L.tx;
    auto M = L;
    quadriga_lib::drop_terminals(L, 1, {}, {}, {}, {}, true, -1); // Seed -1: the layout seed
    CHECK(L.rx.n_node() == 21);
    CHECK(L.rx.name[20] == "UT21");
    CHECK(L.pairing.n_elem == 0);
    CHECK(L.links.empty());
    CHECK(L.seed == 21);
    CHECK(qd_layout_diff_nodes(L.tx, tx) == "");
    quadriga_lib::drop_terminals(M, 1, {}, {}, {}, {}, true, 21);
    CHECK(qd_layout_diff(L, M) == "");
    quadriga_lib::drop_terminals(M, 1, {}, {}, {}, {}, true, 22);
    CHECK(!arma::approx_equal(L.rx.pos, M.rx.pos, "absdiff", 1.0));

    // No layout seed and seed -1: the system random device; the layout seed stays unset
    L.seed = -1;
    quadriga_lib::drop_terminals(L, 1, {}, {}, {}, {}, true, -1);
    CHECK(L.seed == -1);
    CHECK(L.rx.n_node() == 21);

    // n_ut = 0 empties the Rx side
    quadriga_lib::drop_terminals(L, 0, {}, {}, {}, {}, true, 1);
    CHECK(L.rx.n_node() == 0);
}

TEST_CASE("drop_terminals - rules from the scenarios of the sites")
{
    // BS1 and BS2 InF-SL (min 1 m, hUT 1.5 m, 3 km/h; TR 38.901 Table 7.8-7), BS3 UMa (min 35 m to d2D-out, floors, O2I),
    // BS4 no scenario
    auto L = hand_built();
    L.tx.scenario = {"InF_SL", "InF_SL", "InF_SL", "InF_SL", "UMa", "UMa", "", "", ""};
    quadriga_lib::drop_terminals(L, 5, BOUNDS, {}, {NaN, 5.0, 5.0}, {}, true, 4);
    REQUIRE(L.rx.n_node() == 45);
    REQUIRE(L.rx.indoor.n_elem == 45);
    auto c = count(L);
    c.pop_back();
    CHECK(std::all_of(c.begin(), c.end(), [](arma::uword v) { return v == 5; }));
    const auto S = qd_layout_sites(L.tx);
    const arma::mat I = qd_layout_images(L.wrap);
    bool ok = true;
    for (arma::uword i = 0; i < 45; ++i)
    {
        arma::uword s = 0, k = 0;
        const double d = qd_layout_nearest(S, I, L.rx.pos(0, i), L.rx.pos(1, i), s, k);
        if (s <= 1) // InF
            ok = ok && d >= 1.0 - 1.0e-9 && L.rx.pos(2, i) == 1.5 && std::abs(L.rx.speed[i] - 3.0 / 3.6) < 1.0e-12 && L.rx.indoor[i] == -1;
        else if (s == 2) // UMa, building floors fixed to N_fl = 5
        {
            const bool b = L.rx.indoor[i] == 1;
            ok = ok && (b || L.rx.indoor[i] == 0) && d - L.rx.d2d_in[i] >= 35.0 - 1.0e-9 && L.rx.floor[i] >= 1 && L.rx.floor[i] <= (b ? 5 : 1);
            ok = ok && std::abs(L.rx.pos(2, i) - (3.0 * (L.rx.floor[i] - 1) + 1.5)) < 1.0e-12 && std::abs(L.rx.speed[i] - 3.0 / 3.6) < 1.0e-12;
        }
        else // No scenario
            ok = ok && L.rx.pos(2, i) == 1.5 && L.rx.speed[i] == 0.0 && L.rx.indoor[i] == -1 && L.rx.floor[i] == -1 && std::isnan(L.rx.d2d_in[i]);
    }
    CHECK(ok);
    CHECK(L.is_valid() == "");
}
