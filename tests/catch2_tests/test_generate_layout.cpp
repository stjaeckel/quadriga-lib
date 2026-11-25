// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// generate_layout: the hexagonal and indoor presets of 3GPP TR 38.901, the UT drop with its O2I states, and the
// internal family table and O2I draws that the drop shares with the link states.
//
// Oracles: the hexagonal grid in axial coordinates (site count 3n^2 + 3n + 1, the ring of a site from its coordinates,
// the azimuth order, nearest neighbours at the ISD in the directions 30 deg + k 60 deg); the tiling of the wrap lattice by
// brute force (the translates of the cluster hit every grid point of a window exactly once); the preset values typed
// from TR 38.901 Tables 7.2-1 to 7.2-5, 7.4.3-2, 7.8-1, 7.8-1A, 7.8-7 and TR 36.873 Table 6-1; closed-form InH and InF
// BS positions and hall parameters; the cell of each UT by brute force over the sites and wrap images; the O2I fields
// recomputed with sos_rand / sos_randn at the documented stream seeds (FNV-1a oracle of the fixture); the floor draw
// recomputed from its documented algorithm. Statistics use fixed seeds: per-UT draws (floors) with bounds of five
// standard deviations of the sample size; the shares come from fields with 50 m correlation, which lowers the effective
// sample size, so their bounds include the mean correlation over the area (stated at the helper).

#include <catch2/catch_test_macros.hpp>

#include "qd_layout_internal.hpp"
#include "qd_test_layout.hpp"
#include "quadriga_layout.hpp"
#include "quadriga_sos.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
const double D = arma::datum::pi / 180.0;
const double NaN = arma::datum::nan;
const double KMH = 1.0 / 3.6;
using opts = std::unordered_map<std::string, std::vector<double>>;

bool has(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }

// Axial coordinates (q, r) of a point of the hexagonal grid with u = isd (sqrt(3)/2, 1/2), v = isd (0, 1)
bool axial(double x, double y, double isd, int &q, int &r)
{
    const double qf = x / (isd * std::sqrt(3.0) / 2.0), rf = y / isd - qf / 2.0;
    q = (int)std::lround(qf), r = (int)std::lround(rf);
    return std::abs(qf - q) < 1.0e-9 && std::abs(rf - r) < 1.0e-9;
}

int ring_of(int q, int r) { return std::max(std::max(std::abs(q), std::abs(r)), std::abs(q + r)); }

// Tx node of the cell of a point: the nearest site over all sites and wrap images (which must be image 0), then the
// sector whose boresight is within 180 deg / n_sector of the azimuth from the site; -1 outside the wrapped cluster
long long cell_oracle(const quadriga_lib::layout &L, const qd_layout_site_list &S, const arma::mat &I, double x, double y)
{
    arma::uword s = 0, k = 0;
    qd_layout_nearest(S, I, x, y, s, k);
    if (k != 0)
        return -1;
    const auto &nodes = S.nodes[s];
    const double az = std::atan2(y - S.xy(1, s), x - S.xy(0, s)), half = arma::datum::pi / (double)nodes.size();
    for (arma::uword n : nodes)
        if (nodes.size() == 1 || std::abs(qd_layout_wrap(az - L.tx.orientation(2, L.tx.offset[n]))) < half)
            return (long long)n;
    return -1;
}

// Number of UTs per Tx node by the cell oracle; UTs outside the cluster count at the end
std::vector<arma::uword> count_cells(const quadriga_lib::layout &L)
{
    const auto S = qd_layout_sites(L.tx);
    const arma::mat I = qd_layout_images(L.wrap);
    std::vector<arma::uword> c(L.tx.n_node() + 1, 0);
    for (arma::uword i = 0; i < L.rx.n_node(); ++i)
    {
        const long long n = cell_oracle(L, S, I, L.rx.pos(0, i), L.rx.pos(1, i));
        ++c[n < 0 ? L.tx.n_node() : (arma::uword)n];
    }
    return c;
}

// Distance of each UT to the nearest image of the nearest site
arma::vec site_distance(const quadriga_lib::layout &L)
{
    const auto S = qd_layout_sites(L.tx);
    const arma::mat I = qd_layout_images(L.wrap);
    arma::vec d(L.rx.n_node());
    arma::uword s = 0, k = 0;
    for (arma::uword i = 0; i < L.rx.n_node(); ++i)
        d[i] = qd_layout_nearest(S, I, L.rx.pos(0, i), L.rx.pos(1, i), s, k);
    return d;
}

// The O2I field of a stream at the UT positions, recomputed with the library function and the documented seed
arma::vec field(const quadriga_lib::layout &L, const std::string &key, double dist, bool normal = false)
{
    arma::mat xy = L.rx.pos.rows(0, 1).t(), val;
    const long long seed = qd_layout_oracle_seed(L.seed, key);
    if (normal)
        quadriga_lib::sos_randn<double>(arma::vec{dist}, xy, val, arma::mat(), arma::vec(), "Exp256", seed);
    else
        quadriga_lib::sos_rand<double>(arma::vec{dist}, xy, val, arma::mat(), arma::vec(), "Exp256", seed);
    return val.col(0);
}

quadriga_lib::layout gen(const std::string &type, arma::uword n_ut, const arma::vec &spacing = {}, const arma::vec &grid = {},
                         const arma::vec &area = {}, const arma::vec &bs = {}, const arma::vec &ut = {}, const arma::vec &indoor = {},
                         const opts &o = {}, bool per_cell = true, arma::sword seed = 7)
{
    return quadriga_lib::generate_layout(type, n_ut, spacing, grid, area, bs, ut, indoor, o, per_cell, seed);
}

// Bound of five standard deviations for a share p of n UTs whose states come from one field with 50 m correlation over
// the area A: the variance of the sample mean of a field with the correlation exp(-r / 50 m) is p (1 - p) (1 / n + rho),
// with the mean correlation of two uniform points rho = 2 pi 50^2 / A (A much larger than the correlation area); the
// correlation of the thresholded values is lower than that of the field, so the bound is an upper bound
double share_bound(double p, double n, double A) { return 5.0 * std::sqrt(p * (1.0 - p) * (1.0 / n + 2.0 * arma::datum::pi * 2500.0 / A)); }

// Area of a hexagonal cluster
double hex_area(double n_site, double isd) { return n_site * std::sqrt(3.0) / 2.0 * isd * isd; }

double share_of(const quadriga_lib::layout &L, int state)
{
    double n = 0.0;
    for (arma::uword i = 0; i < L.rx.n_node(); ++i)
        n += L.rx.indoor[i] == state ? 1.0 : 0.0;
    return n / (double)L.rx.n_node();
}
} // namespace

TEST_CASE("generate_layout - input validation")
{
    auto G = [](const std::string &type, const arma::vec &spacing = {}, const arma::vec &grid = {}, const arma::vec &area = {},
                const arma::vec &bs = {}, const arma::vec &ut = {}, const arma::vec &indoor = {}, const opts &o = {}, arma::sword seed = 1)
    { return qd_layout_throws([&] { quadriga_lib::generate_layout(type, 1, spacing, grid, area, bs, ut, indoor, o, true, seed); }); };

    CHECK(G("38.901_UMa", {}, {1.0}) == "");
    CHECK(G("38.901_uma", {NaN, NaN}, {1.0, NaN, NaN}, {NaN, NaN, NaN}, {NaN, NaN, NaN}, {NaN, NaN, NaN, NaN, NaN, NaN}, {NaN, NaN, NaN, NaN}) == "");
    CHECK(G("38.901_InH_Open", {20.0, 20.0}, {6.0, 2.0, 1.0}, {120.0, 50.0}) == "");
    CHECK(G("38.901_SMa", {}, {0.0}, {}, {}, {}, {0.5, NaN, NaN, 3.5}, {{"p_residential", {0.5}}}) == "");

    SECTION("type")
    {
        CHECK(has(G("38.901_XYZ"), "'type'"));
        CHECK(has(G(""), "'type'"));
    }
    SECTION("tuple sizes")
    {
        CHECK(has(G("38.901_UMa", {500.0, NaN, 1.0}), "'spacing'"));
        CHECK(has(G("38.901_UMa", {}, {1.0, NaN, 3.0, 1.0}), "'grid'"));
        CHECK(has(G("38.901_InH_Open", {}, {}, {120.0, 50.0, NaN, 1.0}), "'area'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {25.0, 0.0, 0.0, 1.0}), "'bs'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}), "'ut'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {}, {0.8, 4.0, 8.0, 3.0, 1.0}), "'indoor'"));
    }
    SECTION("entries the preset does not use")
    {
        CHECK(has(G("38.901_UMa", {500.0, 500.0}, {1.0}), "'d_y'"));
        CHECK(has(G("38.901_UMa", {}, {1.0, 2.0}), "'n_y'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {100.0}), "'length'"));
        CHECK(has(G("38.901_InH_Open", {}, {}, {120.0, 50.0, 3.0}), "'height'"));
        CHECK(has(G("38.901_RMa", {}, {0.0}, {}, {}, {}, {NaN, 4.0}), "'min floors'"));
        CHECK(has(G("38.901_SMa", {}, {0.0}, {}, {}, {}, {NaN, NaN, 5.0}), "'max floors'"));
        CHECK(has(G("38.901_RMa", {}, {0.0}, {}, {}, {}, {NaN, NaN, NaN, 3.5}), "'floor height'"));
        CHECK(has(G("38.901_InF_SL", {}, {}, {}, {}, {}, {0.5}), "'indoor ratio'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {}, {}, {{"p_residential", {0.5}}}), "'p_residential'"));
    }
    SECTION("values")
    {
        CHECK(has(G("38.901_UMa", {0.0}), "'d_x'"));
        CHECK(has(G("38.901_UMa", {NaN}, {1.5}), "'n_x'"));
        CHECK(has(G("38.901_UMa", {}, {-1.0}), "'n_x'"));
        CHECK(has(G("38.901_UMa", {}, {1.0, NaN, 0.0}), "'n_sector'"));
        CHECK(has(G("38.901_UMa", {}, {1.0, NaN, 2.5}), "'n_sector'"));
        CHECK(has(G("38.901_InH_Open", {20.0, -1.0}), "'d_y'"));
        CHECK(has(G("38.901_InH_Open", {}, {0.0}), "'n_x'"));
        CHECK(has(G("38.901_InH_Open", {}, {}, {120.0, 0.0}), "'width'"));
        CHECK(has(G("38.901_InH_Open", {}, {8.0}), "'grid'"));          // 8 BSs at 20 m do not fit into 120 m
        CHECK(has(G("38.901_InF_DH", {25.0}, {}, {}), "'spacing'"));    // 6 BSs at 25 m do not fit into 120 m
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {arma::datum::inf}), "'height'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {NaN, NaN, NaN}, {NaN, -1.0}), "'min distance'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {NaN, NaN, -1.0}), "'speed'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {NaN, NaN, NaN, NaN, NaN, arma::datum::inf}), "'heading'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {}, {1.5}), "'indoor ratio'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {}, {NaN, 0.5}), "'min floors'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {}, {NaN, 9.0}), "'min floors'")); // Above the max floors 8 of the preset
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {}, {NaN, NaN, NaN, 0.0}), "'floor height'"));
    }
    SECTION("options")
    {
        CHECK(has(G("38.901_SMa", {}, {0.0}, {}, {}, {}, {}, {{"abc", {1.0}}}), "unknown"));
        CHECK(has(G("38.901_SMa", {}, {0.0}, {}, {}, {}, {}, {{"p_residential", {0.5, 0.5}}}), "'p_residential'"));
        CHECK(has(G("38.901_SMa", {}, {0.0}, {}, {}, {}, {}, {{"p_residential", {1.5}}}), "'p_residential'"));
    }
    SECTION("seed")
    {
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {}, {}, {}, -2), "'seed'"));
    }
    SECTION("sizes")
    {
        // UTs: at most 1e7 in total, also when n_ut times the number of cells (57) wraps around in 64 bits
        auto N = [](arma::uword n_ut, bool per_cell)
        { return qd_layout_throws([&] { quadriga_lib::generate_layout("38.901_UMa", n_ut, {}, {}, {}, {}, {}, {}, {}, per_cell, 1); }); };
        CHECK(has(N(323627089012448274ULL, true), "'n_ut'"));
        CHECK(has(N(200000, true), "'n_ut'"));
        CHECK(has(N(10000001, false), "'n_ut'"));
        // Grid sizes
        CHECK(has(G("38.901_UMa", {}, {51.0}), "'n_x'"));
        CHECK(has(G("38.901_UMa", {}, {1.0, NaN, 13.0}), "'n_sector'"));
        CHECK(has(G("38.901_InH_Open", {1.0, 1.0}, {101.0, 2.0}), "'n_x'"));
        CHECK(has(G("38.901_InH_Open", {1.0, 1.0}, {6.0, 101.0}), "'n_y'"));
    }
}

TEST_CASE("generate_layout - hexagonal sites, order and wrap lattice")
{
    const double isd = 100.0, s3 = std::sqrt(3.0);
    for (int n = 0; n <= 3; ++n)
    {
        const auto L = gen("38.901_UMa", 0, {isd}, {(double)n, NaN, 1.0});
        const auto S = qd_layout_sites(L.tx);
        const size_t n_site = (size_t)(3 * n * n + 3 * n + 1);
        REQUIRE(S.name.size() == n_site);
        CHECK(L.tx.n_node() == n_site);
        CHECK(L.rx.n_node() == 0);

        // Grid points, each once, all with ring <= n (so all of them); order by ring, then azimuth in (-30, 330] deg
        std::set<std::pair<int, int>> seen;
        int prev_ring = -1;
        double prev_az = -1.0e9;
        bool on_grid = true, ordered = true;
        for (size_t i = 0; i < n_site; ++i)
        {
            CHECK(S.name[i] == "BS" + std::to_string(i + 1));
            int q = 0, r = 0;
            on_grid = on_grid && axial(S.xy(0, i), S.xy(1, i), isd, q, r) && ring_of(q, r) <= n;
            seen.insert({q, r});
            double az = std::atan2(S.xy(1, i), S.xy(0, i)) / D;
            if (az <= -30.0 + 1.0e-6)
                az += 360.0;
            const int ring = ring_of(q, r);
            if (ring == prev_ring)
                ordered = ordered && az > prev_az + 1.0e-6;
            else
                ordered = ordered && ring == prev_ring + 1;
            prev_ring = ring, prev_az = az;
        }
        CHECK(on_grid);
        CHECK(ordered);
        CHECK(seen.size() == n_site);

        // Nearest neighbour of every site at the ISD; the six neighbours of the centre at 30 deg + k 60 deg, in this order
        if (n >= 1)
        {
            double worst = 0.0;
            for (size_t i = 0; i < n_site; ++i)
            {
                double dmin = arma::datum::inf;
                for (size_t j = 0; j < n_site; ++j)
                    if (j != i)
                        dmin = std::min(dmin, std::hypot(S.xy(0, i) - S.xy(0, j), S.xy(1, i) - S.xy(1, j)));
                worst = std::max(worst, std::abs(dmin - isd));
            }
            CHECK(worst < 1.0e-9);
            for (int k = 0; k < 6; ++k)
            {
                CHECK(std::abs(S.xy(0, (size_t)k + 1) - isd * std::cos((30.0 + 60.0 * k) * D)) < 1.0e-9);
                CHECK(std::abs(S.xy(1, (size_t)k + 1) - isd * std::sin((30.0 + 60.0 * k) * D)) < 1.0e-9);
            }
        }

        // Lattice: a1 = n u + (n + 1) v, a2 = a1 rotated by +60 deg
        REQUIRE(L.wrap.n_rows == 2);
        REQUIRE(L.wrap.n_cols == 2);
        const double a1x = n * isd * s3 / 2.0, a1y = n * isd / 2.0 + (n + 1) * isd;
        CHECK(std::abs(L.wrap(0, 0) - a1x) < 1.0e-9);
        CHECK(std::abs(L.wrap(1, 0) - a1y) < 1.0e-9);
        CHECK(std::abs(L.wrap(0, 1) - (0.5 * a1x - s3 / 2.0 * a1y)) < 1.0e-9);
        CHECK(std::abs(L.wrap(1, 1) - (s3 / 2.0 * a1x + 0.5 * a1y)) < 1.0e-9);
        if (n == 2) // Two rings, by hand: a1 = (sqrt(3), 4) D, a2 = (-3 sqrt(3) / 2, 7 / 2) D
        {
            CHECK(std::abs(L.wrap(0, 0) - s3 * isd) < 1.0e-9);
            CHECK(std::abs(L.wrap(1, 0) - 4.0 * isd) < 1.0e-9);
            CHECK(std::abs(L.wrap(0, 1) + 1.5 * s3 * isd) < 1.0e-9);
            CHECK(std::abs(L.wrap(1, 1) - 3.5 * isd) < 1.0e-9);
        }

        // Tiling: the translates s + i a1 + j a2 hit every grid point of the window (ring <= 2n + 2) exactly once
        std::map<std::pair<int, int>, int> hits;
        bool translates_on_grid = true;
        for (int i = -6; i <= 6; ++i)
            for (int j = -6; j <= 6; ++j)
                for (size_t s = 0; s < n_site; ++s)
                {
                    int q = 0, r = 0;
                    const double x = S.xy(0, s) + i * L.wrap(0, 0) + j * L.wrap(0, 1), y = S.xy(1, s) + i * L.wrap(1, 0) + j * L.wrap(1, 1);
                    translates_on_grid = translates_on_grid && axial(x, y, isd, q, r);
                    if (ring_of(q, r) <= 2 * n + 2)
                        ++hits[{q, r}];
                }
        CHECK(translates_on_grid);
        const int w = 2 * n + 2;
        CHECK(hits.size() == (size_t)(3 * w * w + 3 * w + 1));
        bool once = true;
        for (const auto &h : hits)
            once = once && h.second == 1;
        CHECK(once);
    }
}

TEST_CASE("generate_layout - sectors and BS orientation")
{
    auto L = gen("38.901_UMa", 0, {}, {1.0});
    REQUIRE(L.tx.n_node() == 21);
    CHECK(L.tx.n_row() == 21);
    bool ok = true;
    for (arma::uword i = 0; i < 21; ++i)
    {
        const arma::uword r = L.tx.offset[i];
        ok = ok && L.tx.name[i] == "BS" + std::to_string(i / 3 + 1) && L.tx.sector[i] == i % 3 && L.tx.i_antenna[i] == 0;
        ok = ok && std::abs(qd_layout_wrap(L.tx.orientation(2, r) - (30.0 + 120.0 * (double)(i % 3)) * D)) < 1.0e-12;
        ok = ok && L.tx.orientation(0, r) == 0.0 && L.tx.orientation(1, r) == 0.0 && L.tx.pos(2, r) == 25.0;
        ok = ok && L.tx.scenario[r] == "UMa";
    }
    CHECK(ok);
    CHECK(L.center_freq.n_elem == 0);
    CHECK(L.antenna_args.empty());
    CHECK(L.is_valid() == "");

    // Height, mechanical tilt (positive down) and azimuth offset of the sectors
    L = gen("38.901_UMa", 0, {}, {1.0}, {}, {20.0, 7.0, 10.0});
    ok = true;
    for (arma::uword i = 0; i < 21; ++i)
    {
        ok = ok && std::abs(qd_layout_wrap(L.tx.orientation(2, i) - (40.0 + 120.0 * (double)(i % 3)) * D)) < 1.0e-12;
        ok = ok && std::abs(L.tx.orientation(1, i) - 7.0 * D) < 1.0e-15 && L.tx.pos(2, i) == 20.0;
    }
    CHECK(ok);

    // One and four sectors: boresights 30 deg + k 360 deg / n
    L = gen("38.901_UMa", 0, {}, {1.0, NaN, 1.0});
    CHECK(L.tx.n_node() == 7);
    CHECK(std::abs(L.tx.orientation(2, 0) - 30.0 * D) < 1.0e-12);
    L = gen("38.901_UMa", 0, {}, {0.0, NaN, 4.0});
    REQUIRE(L.tx.n_node() == 4);
    for (arma::uword k = 0; k < 4; ++k)
        CHECK(std::abs(qd_layout_wrap(L.tx.orientation(2, k) - (30.0 + 90.0 * (double)k) * D)) < 1.0e-12);
}

TEST_CASE("generate_layout - TR 38.901 presets")
{
    // Hexagonal presets: 19 sites, 3 sectors, wrap. TR 38.901 Table 7.2-1 (UMi: ISD 200 m, hBS 10 m, min 10 m, 80 %
    // indoor; UMa: ISD 500 m, hBS 25 m, min 35 m, 80 % indoor; both 3 km/h), Table 7.2-3 (RMa: ISD 1732 m, hBS 35 m, hUT
    // 1.5 m, 50 % indoor and 50 % in car, min 35 m), Table 7.2-5 (SMa: ISD 1299 m, hBS 35 m, 80 % indoor, min 35 m, 3 km/h
    // indoors, 40 km/h in cars), Table 7.8-1A (SMa at ISD 1299 m: mechanical downtilt 95 deg = 5 deg down). Car speed of
    // RMa: the TR gives none; the library uses the SMa value
    struct hex_row
    {
        const char *preset, *family;
        double isd, h_bs, tilt, d_min, v_ped, v_car;
    };
    const hex_row hex[] = {{"38.901_UMi", "UMi", 200.0, 10.0, 0.0, 10.0, 3.0, NaN},
                           {"38.901_UMa", "UMa", 500.0, 25.0, 0.0, 35.0, 3.0, NaN},
                           {"38.901_RMa", "RMa", 1732.0, 35.0, 0.0, 35.0, 3.0, 40.0},
                           {"38.901_SMa", "SMa", 1299.0, 35.0, 5.0, 35.0, 3.0, 40.0}};
    for (const auto &p : hex)
    {
        INFO(p.preset);
        const auto L = gen(p.preset, 1);
        const auto S = qd_layout_sites(L.tx);
        REQUIRE(S.name.size() == 19);
        REQUIRE(L.tx.n_node() == 57);
        double dmin = arma::datum::inf;
        for (arma::uword j = 1; j < 19; ++j)
            dmin = std::min(dmin, std::hypot(S.xy(0, j) - S.xy(0, 0), S.xy(1, j) - S.xy(1, 0)));
        CHECK(std::abs(dmin - p.isd) < 1.0e-9);
        bool ok = true;
        for (arma::uword i = 0; i < 57; ++i)
            ok = ok && L.tx.pos(2, i) == p.h_bs && std::abs(L.tx.orientation(1, i) - p.tilt * D) < 1.0e-15 && L.tx.scenario[i] == p.family &&
                 std::abs(qd_layout_wrap(L.tx.orientation(2, i) - (30.0 + 120.0 * (double)(i % 3)) * D)) < 1.0e-12;
        CHECK(ok);
        CHECK(L.wrap.n_cols == 2);
        CHECK(L.is_valid() == "");

        // UTs: one per cell; the minimum distance; speed of pedestrian and car UTs; car and outdoor UTs at 1.5 m
        REQUIRE(L.rx.n_node() == 57);
        const arma::vec d = site_distance(L);
        ok = true;
        for (arma::uword i = 0; i < 57; ++i)
        {
            const int s = L.rx.indoor[i];
            ok = ok && d[i] >= p.d_min + (s == 1 || s == 3 ? (std::string(p.family) == "UMi" || std::string(p.family) == "UMa" ? L.rx.d2d_in[i] : 0.0) : 0.0) - 1.0e-9;
            ok = ok && std::abs(L.rx.speed[i] - (s == 2 ? p.v_car : p.v_ped) * KMH) < 1.0e-12;
            if (s == 0 || s == 2)
                ok = ok && L.rx.pos(2, i) == 1.5 && L.rx.floor[i] == 1;
        }
        CHECK(ok);
    }

    // Indoor presets, centred at the origin: TR 38.901 Table 7.2-2 (InH: 120 m x 50 m x 3 m, ISD 20 m, hBS 3 m, hUT 1 m,
    // min 0, 3 km/h; 12 BSs) and Table 7.8-7 (InF: halls 120 m x 60 m and 300 m x 150 m, 10 m high; 18 BSs on a square
    // lattice with spacing D = 20 m or 50 m, D/2 from the walls; hBS 1.5 m (SL, DL) or 8 m (SH, DH); hUT 1.5 m; min 1 m).
    // InF-HH has no calibration layout in the TR: the big hall with hBS 8 m and the library UT height 3 m (above the
    // clutter height 2 m of the low clutter density); InF speed: the TR gives none, 3 km/h as InH
    struct in_row
    {
        const char *preset, *family;
        double len, wid, hgt, d;
        arma::uword nx, ny;
        double h_bs, h_ut, d_min, l_max;
    };
    const in_row in[] = {{"38.901_InH_Open", "InH_Open", 120.0, 50.0, NaN, 20.0, 6, 2, 3.0, 1.0, 0.0, NaN},
                         {"38.901_InH_Mixed", "InH_Mixed", 120.0, 50.0, NaN, 20.0, 6, 2, 3.0, 1.0, 0.0, NaN},
                         {"38.901_InF_SL", "InF_SL", 120.0, 60.0, 10.0, 20.0, 6, 3, 1.5, 1.5, 1.0, 120.0},
                         {"38.901_InF_DL", "InF_DL", 300.0, 150.0, 10.0, 50.0, 6, 3, 1.5, 1.5, 1.0, 300.0},
                         {"38.901_InF_SH", "InF_SH", 300.0, 150.0, 10.0, 50.0, 6, 3, 8.0, 1.5, 1.0, 300.0},
                         {"38.901_InF_DH", "InF_DH", 120.0, 60.0, 10.0, 20.0, 6, 3, 8.0, 1.5, 1.0, 120.0},
                         {"38.901_InF_HH", "InF_HH", 300.0, 150.0, 10.0, 50.0, 6, 3, 8.0, 3.0, 1.0, 300.0}};
    for (const auto &p : in)
    {
        INFO(p.preset);
        const auto L = gen(p.preset, 2);
        REQUIRE(L.tx.n_node() == p.nx * p.ny);
        CHECK(L.wrap.n_elem == 0);
        // Positions: the lattice with spacing D centred in the hall; InH: x = -50:20:50, y = +-10 (Figure 7.2-1); InF: D/2
        // from the walls
        std::set<std::pair<long, long>> want, got;
        for (arma::uword i = 0; i < p.nx; ++i)
            for (arma::uword j = 0; j < p.ny; ++j)
                want.insert({std::lround(1000.0 * p.d * ((double)i - 0.5 * (double)(p.nx - 1))), std::lround(1000.0 * p.d * ((double)j - 0.5 * (double)(p.ny - 1)))});
        if (!std::isnan(p.hgt))
        {
            CHECK(want.begin()->first == std::lround(1000.0 * (-p.len / 2.0 + p.d / 2.0)));
            CHECK(want.begin()->second == std::lround(1000.0 * (-p.wid / 2.0 + p.d / 2.0)));
            CHECK(want.rbegin()->second == std::lround(1000.0 * (p.wid / 2.0 - p.d / 2.0)));
        }
        bool ok = true;
        for (arma::uword i = 0; i < L.tx.n_node(); ++i)
        {
            got.insert({std::lround(1000.0 * L.tx.pos(0, i)), std::lround(1000.0 * L.tx.pos(1, i))});
            ok = ok && L.tx.pos(2, i) == p.h_bs && L.tx.sector[i] == 0 && L.tx.name[i] == "BS" + std::to_string(i + 1) &&
                 L.tx.scenario[i] == p.family && L.tx.orientation(0, i) == 0.0 && L.tx.orientation(1, i) == 0.0 && L.tx.orientation(2, i) == 0.0;
        }
        CHECK(ok);
        CHECK(got == want);
        if (p.nx == 6 && p.ny == 2)
        {
            CHECK(want.count({-50000, -10000}) == 1);
            CHECK(want.count({50000, 10000}) == 1);
        }

        // InF hall parameters: V/S (TR Table 7.5-6 Part-3, DS of InF) and the largest dimension (clause 7.6.9)
        if (!std::isnan(p.hgt))
        {
            REQUIRE(L.tx.scenpar_override.count("DS_VS") == 1);
            REQUIRE(L.tx.scenpar_override.count("absTOA_Lmax") == 1);
            const double vs = p.len * p.wid * p.hgt / (2.0 * (p.len * p.wid + p.len * p.hgt + p.wid * p.hgt));
            CHECK(arma::approx_equal(L.tx.scenpar_override.at("DS_VS"), arma::vec(L.tx.n_node(), arma::fill::value(vs)), "absdiff", 1.0e-12));
            CHECK(arma::all(L.tx.scenpar_override.at("absTOA_Lmax") == p.l_max));
        }
        else
            CHECK(L.tx.scenpar_override.empty());
        CHECK(L.is_valid() == "");

        // UTs: two per BS, in the hall, at hUT, the minimum distance, 3 km/h, no O2I state
        REQUIRE(L.rx.n_node() == 2 * p.nx * p.ny);
        const arma::vec d = site_distance(L);
        ok = true;
        for (arma::uword i = 0; i < L.rx.n_node(); ++i)
            ok = ok && std::abs(L.rx.pos(0, i)) <= p.len / 2.0 && std::abs(L.rx.pos(1, i)) <= p.wid / 2.0 && L.rx.pos(2, i) == p.h_ut &&
                 d[i] >= p.d_min - 1.0e-9 && std::abs(L.rx.speed[i] - 3.0 * KMH) < 1.0e-12;
        CHECK(ok);
        CHECK(L.rx.indoor.n_elem == 0);
        CHECK(L.rx.floor.n_elem == 0);
        CHECK(L.rx.d2d_in.n_elem == 0);
    }

    // A hall whose largest dimension is the width: 60 m x 80 m x 10 m, 6 x 3 BSs at 10 m and 20 m
    const auto W = gen("38.901_InF_SL", 0, {10.0, 20.0}, {}, {60.0, 80.0, 10.0});
    REQUIRE(W.tx.scenpar_override.count("absTOA_Lmax") == 1);
    CHECK(arma::all(W.tx.scenpar_override.at("absTOA_Lmax") == 80.0));
    CHECK(arma::all(arma::abs(W.tx.scenpar_override.at("DS_VS") - 48000.0 / 12400.0) < 1.0e-12));
}

TEST_CASE("generate_layout - exactly n_ut per cell")
{
    // UMi with one ring: 21 sector cells, three UTs each
    auto L = gen("38.901_UMi", 3, {}, {1.0});
    REQUIRE(L.rx.n_node() == 63);
    auto c = count_cells(L);
    CHECK(c.back() == 0);
    c.pop_back();
    CHECK(std::all_of(c.begin(), c.end(), [](arma::uword v) { return v == 3; }));
    for (arma::uword i = 0; i < 63; ++i)
        CHECK(L.rx.name[i] == "UT" + std::to_string(i + 1));

    // Without sectors: the site hexagon
    L = gen("38.901_UMi", 2, {}, {1.0, NaN, 1.0});
    c = count_cells(L);
    CHECK(c.back() == 0);
    c.pop_back();
    CHECK(std::all_of(c.begin(), c.end(), [](arma::uword v) { return v == 2; }));

    // In total: 50 UTs in the cluster
    L = gen("38.901_UMi", 50, {}, {1.0}, {}, {}, {}, {}, {}, false);
    CHECK(L.rx.n_node() == 50);
    CHECK(count_cells(L).back() == 0);

    // Indoor: the lattice cells of the hall (InH: 20 m x 25 m around each BS)
    L = gen("38.901_InH_Mixed", 4);
    REQUIRE(L.rx.n_node() == 48);
    c = count_cells(L);
    c.pop_back();
    CHECK(std::all_of(c.begin(), c.end(), [](arma::uword v) { return v == 4; }));
}

TEST_CASE("generate_layout - minimum distance, O2I states and floors")
{
    // UMa, one ring, 20 UTs per cell: 420 UTs
    const auto L = gen("38.901_UMa", 20, {}, {1.0});
    REQUIRE(L.rx.n_node() == 420);
    REQUIRE(L.rx.indoor.n_elem == 420);
    REQUIRE(L.rx.floor.n_elem == 420);
    REQUIRE(L.rx.d2d_in.n_elem == 420);
    const arma::vec d = site_distance(L);

    // The O2I states and indoor distances from the fields at the documented seeds: indoor state 50 m (TR Table
    // 7.6.3.1-2), indoor distance min(U1, U2) 25 m with 25 m correlation (clause 7.6.3.3, Table 7.4.3-2)
    const arma::vec u_io = field(L, "io|UMa", 50.0), u1 = field(L, "d2din1|UMa", 25.0), u2 = field(L, "d2din2|UMa", 25.0);
    bool states = true, dist = true, heights = true;
    arma::uword n_in = 0, n_in_near = 0;
    for (arma::uword i = 0; i < 420; ++i)
    {
        const bool building = u_io[i] < 0.8;
        states = states && L.rx.indoor[i] == (building ? 1 : 0);
        const double din = building ? std::min(u1[i], u2[i]) * 25.0 : 0.0;
        states = states && std::abs(L.rx.d2d_in[i] - din) < 1.0e-12;
        // Minimum distance 35 m: to d2D outdoors, to d2D-out = d2D - d2D-in indoors (TR 36.873 Table 6-1, note 1)
        dist = dist && d[i] - din >= 35.0 - 1.0e-9;
        n_in += building ? 1 : 0;
        n_in_near += building && d[i] < 35.0 + din + 1.0e-9 ? 1 : 0;
        // Floors: N_fl ~ U(4, 8), n_fl ~ U(1, N_fl), hUT = 3 (n_fl - 1) + 1.5 (TR 36.873 Table 6-1); outdoors floor 1
        const int f = L.rx.floor[i];
        heights = heights && f >= 1 && f <= (building ? 8 : 1) && std::abs(L.rx.pos(2, i) - (3.0 * (f - 1) + 1.5)) < 1.0e-12;
    }
    CHECK(states);
    CHECK(dist);
    CHECK(heights);
    CHECK(n_in_near == 0);
    // Indoor share 0.8 (Table 7.2-1)
    CHECK(std::abs((double)n_in / 420.0 - 0.8) < share_bound(0.8, 420.0, hex_area(7.0, 500.0)));

    // Floor statistics of the building UTs (drawn per UT, independent): mean 3.5 with variance 3.58 / n; floors >= 5 with
    // probability (1/5)(1/5 + 2/6 + 3/7 + 4/8) = 0.2924
    double mean = 0.0, high = 0.0;
    for (arma::uword i = 0; i < 420; ++i)
        if (L.rx.indoor[i] == 1)
            mean += L.rx.floor[i], high += L.rx.floor[i] >= 5 ? 1.0 : 0.0;
    mean /= (double)n_in, high /= (double)n_in;
    CHECK(std::abs(mean - 3.5) < 5.0 * std::sqrt(3.58 / (double)n_in));
    CHECK(std::abs(high - 0.2924) < 5.0 * std::sqrt(0.2924 * 0.7076 / (double)n_in));

    // A given UT height sets the floor of building UTs from the height and the floor height
    const auto H = gen("38.901_UMa", 5, {}, {1.0}, {}, {}, {7.5}, {NaN, NaN, NaN, 2.0});
    bool fh = true;
    for (arma::uword i = 0; i < H.rx.n_node(); ++i)
        fh = fh && H.rx.pos(2, i) == 7.5 && H.rx.floor[i] == (H.rx.indoor[i] == 1 ? 4 : 1);
    CHECK(fh);

    // Indoor ratio, floor range and floor height given
    const auto R = gen("38.901_UMi", 20, {}, {1.0}, {}, {}, {}, {0.3, 2.0, 2.0, 4.0});
    const arma::vec r_io = field(R, "io|UMi", 50.0);
    bool fr = true;
    for (arma::uword i = 0; i < R.rx.n_node(); ++i)
        fr = fr && R.rx.indoor[i] == (r_io[i] < 0.3 ? 1 : 0) && (R.rx.indoor[i] == 1 ? (R.rx.floor[i] == 1 || R.rx.floor[i] == 2) : R.rx.floor[i] == 1) &&
             R.rx.pos(2, i) == 4.0 * (R.rx.floor[i] - 1) + 1.5;
    CHECK(fr);
    CHECK(std::abs(share_of(R, 1) - 0.3) < share_bound(0.3, 420.0, hex_area(7.0, 200.0)));
}

TEST_CASE("generate_layout - RMa and SMa car UTs and building classes")
{
    // RMa: 50 % in buildings (low-loss model, d2D-in up to 10 m), 50 % in cars (Table 7.2-3, clause 7.4.3.1); 1.5 m
    const auto R = gen("38.901_RMa", 10, {}, {1.0});
    REQUIRE(R.rx.n_node() == 210);
    const arma::vec io = field(R, "io|RMa", 50.0), u1 = field(R, "d2din1|RMa", 25.0), u2 = field(R, "d2din2|RMa", 25.0);
    const arma::vec d = site_distance(R);
    bool ok = true;
    for (arma::uword i = 0; i < 210; ++i)
    {
        const bool b = io[i] < 0.5;
        ok = ok && R.rx.indoor[i] == (b ? 1 : 2) && std::abs(R.rx.d2d_in[i] - (b ? std::min(u1[i], u2[i]) * 10.0 : 0.0)) < 1.0e-12;
        ok = ok && R.rx.pos(2, i) == 1.5 && R.rx.floor[i] == 1 && d[i] >= 35.0 - 1.0e-9;
        ok = ok && std::abs(R.rx.speed[i] - (b ? 3.0 : 40.0) * KMH) < 1.0e-12;
    }
    CHECK(ok);
    CHECK(std::abs(share_of(R, 1) - 0.5) < share_bound(0.5, 210.0, hex_area(7.0, 1732.0)));

    // SMa: 80 % in buildings, 90 % of them residential (up to 2 floors, d2D-in up to 10 m), the others commercial (up to
    // 5 floors, 25 m), 20 % in cars (Tables 7.2-5, 7.8-1A, clause 7.4.3.1)
    const auto S = gen("38.901_SMa", 10, {}, {1.0});
    REQUIRE(S.rx.n_node() == 210);
    const arma::vec s_io = field(S, "io|SMa", 50.0), s_cl = field(S, "o2i_class|SMa", 50.0), s1 = field(S, "d2din1|SMa", 25.0),
                    s2 = field(S, "d2din2|SMa", 25.0);
    ok = true;
    double n_b = 0.0, n_res = 0.0, n_f2 = 0.0;
    for (arma::uword i = 0; i < 210; ++i)
    {
        const bool b = s_io[i] < 0.8, res = s_cl[i] < 0.9;
        const int want = b ? (res ? 3 : 1) : 2;
        ok = ok && S.rx.indoor[i] == want;
        ok = ok && std::abs(S.rx.d2d_in[i] - (b ? std::min(s1[i], s2[i]) * (res ? 10.0 : 25.0) : 0.0)) < 1.0e-12;
        const int f = S.rx.floor[i];
        ok = ok && f >= 1 && f <= (want == 3 ? 2 : want == 1 ? 5 : 1) && std::abs(S.rx.pos(2, i) - (3.0 * (f - 1) + 1.5)) < 1.0e-12;
        ok = ok && std::abs(S.rx.speed[i] - (b ? 3.0 : 40.0) * KMH) < 1.0e-12;
        n_b += b ? 1.0 : 0.0, n_res += want == 3 ? 1.0 : 0.0, n_f2 += want == 3 && f == 2 ? 1.0 : 0.0;
    }
    CHECK(ok);
    CHECK(std::abs(n_b / 210.0 - 0.8) < share_bound(0.8, 210.0, hex_area(7.0, 1299.0)));
    CHECK(std::abs(n_res / n_b - 0.9) < share_bound(0.9, n_b, hex_area(7.0, 1299.0)));
    // Residential floors U{1, 2}, per UT: five standard deviations
    CHECK(std::abs(n_f2 / n_res - 0.5) < 5.0 * std::sqrt(0.25 / n_res));

    // p_residential = 0: all building UTs commercial
    const auto C = gen("38.901_SMa", 3, {}, {1.0}, {}, {}, {}, {}, {{"p_residential", {0.0}}});
    CHECK(share_of(C, 3) == 0.0);
}

TEST_CASE("generate_layout - SMa tilt follows the ISD")
{
    // TR Tables 7.8-1A, 7.8-2A: mechanical downtilt 95 deg at ISD 1299 m, 92 deg at 1732 m (zenith angles: 5 and 2 deg
    // down); other ISDs take the value of the nearer one
    const double isd[] = {1299.0, 1732.0, 1400.0, 1600.0, 1200.0, 1800.0};
    const double tilt[] = {5.0, 2.0, 5.0, 2.0, 5.0, 2.0};
    for (int k = 0; k < 6; ++k)
    {
        const auto L = gen("38.901_SMa", 0, {isd[k]}, {0.0});
        CHECK(std::abs(L.tx.orientation(1, 0) - tilt[k] * D) < 1.0e-15);
    }
    const auto L = gen("38.901_SMa", 0, {1732.0}, {0.0}, {}, {NaN, 3.0});
    CHECK(std::abs(L.tx.orientation(1, 0) - 3.0 * D) < 1.0e-15);
}

TEST_CASE("generate_layout - orientation, speed and direction of the UTs")
{
    // Default: point drops, heading and direction uniform and independent, tilt and bank 0, elevation 0
    auto L = gen("38.901_UMi", 4, {}, {1.0});
    const arma::uword n = L.rx.n_node();
    REQUIRE(n == 84);
    REQUIRE(L.rx.n_row() == 84);
    REQUIRE(L.rx.direction.n_cols == 84);
    bool ok = true;
    arma::uword same = 0;
    for (arma::uword i = 0; i < n; ++i)
    {
        ok = ok && L.rx.offset[i] == i && L.rx.orientation(0, i) == 0.0 && L.rx.orientation(1, i) == 0.0 && L.rx.direction(1, i) == 0.0;
        ok = ok && std::abs(L.rx.speed[i] - 3.0 * KMH) < 1.0e-12 && L.rx.sector[i] == 0 && L.rx.i_antenna[i] == 0;
        same += std::abs(qd_layout_wrap(L.rx.orientation(2, i) - L.rx.direction(0, i))) < 1.0e-9 ? 1 : 0;
    }
    CHECK(ok);
    CHECK(same == 0);
    CHECK(L.rx.is_valid() == "");

    // Uniform in azimuth: the mean resultant length R of n uniform angles has 2 n R^2 ~ chi^2 with 2 degrees of freedom,
    // so R < 4 / sqrt(n) fails with the probability exp(-16); also for the difference of heading and direction
    // (independent draws). 420 UTs
    const auto U = gen("38.901_UMi", 20, {}, {1.0});
    const arma::uword nu = U.rx.n_node();
    REQUIRE(nu == 420);
    auto resultant = [](const arma::rowvec &a) { return std::hypot(arma::mean(arma::cos(a)), arma::mean(arma::sin(a))); };
    const double bound = 4.0 / std::sqrt((double)nu);
    CHECK(resultant(U.rx.orientation.row(2)) < bound);
    CHECK(resultant(U.rx.direction.row(0)) < bound);
    CHECK(resultant(U.rx.orientation.row(2) - U.rx.direction.row(0)) < bound);
    CHECK(resultant(2.0 * U.rx.orientation.row(2)) < bound); // Second harmonic: not two opposite clusters

    // Calibration orientation of TR Table 7.8-2 (tilt 90 deg, heading random) and Table 7.8-2A config D (all 0), speed given
    L = gen("38.901_UMi", 2, {}, {1.0}, {}, {}, {NaN, NaN, 10.0, NaN, 90.0});
    ok = true;
    for (arma::uword i = 0; i < L.rx.n_node(); ++i)
        ok = ok && std::abs(L.rx.orientation(1, i) - 90.0 * D) < 1.0e-15 && L.rx.orientation(0, i) == 0.0 && L.rx.speed[i] == 10.0;
    CHECK(ok);
    CHECK(L.rx.orientation.row(2).max() > L.rx.orientation.row(2).min());
    L = gen("38.901_UMi", 2, {}, {1.0}, {}, {}, {NaN, NaN, NaN, 5.0, 0.0, 0.0});
    CHECK(arma::all(L.rx.orientation.row(2) == 0.0));
    CHECK(arma::all(arma::abs(L.rx.orientation.row(0) - 5.0 * D) < 1.0e-15));

    // Minimum distance given
    L = gen("38.901_InH_Open", 5, {}, {}, {}, {}, {NaN, 6.0});
    CHECK(site_distance(L).min() >= 6.0 - 1.0e-9);
}

TEST_CASE("generate_layout - seeds and determinism")
{
    const auto A = gen("38.901_UMa", 3, {}, {1.0}, {}, {}, {}, {}, {}, true, 11);
    const auto B = gen("38.901_UMa", 3, {}, {1.0}, {}, {}, {}, {}, {}, true, 11);
    const auto C = gen("38.901_UMa", 3, {}, {1.0}, {}, {}, {}, {}, {}, true, 12);
    CHECK(A.seed == 11);
    CHECK(qd_layout_diff(A, B) == "");
    CHECK(!arma::approx_equal(A.rx.pos, C.rx.pos, "absdiff", 1.0));

    // Seed -1: a drawn seed is stored and reproduces the layout
    const auto R = gen("38.901_UMa", 3, {}, {1.0}, {}, {}, {}, {}, {}, true, -1);
    REQUIRE(R.seed >= 0);
    CHECK(qd_layout_diff(R, gen("38.901_UMa", 3, {}, {1.0}, {}, {}, {}, {}, {}, true, R.seed)) == "");

    // The CSV files hold the generated layout
    qd_layout_files F("generate_layout");
    const auto I = gen("38.901_InF_SH", 1);
    quadriga_lib::layout_csv_write(F.path("tx.csv"), F.path("rx.csv"), A);
    const auto A2 = quadriga_lib::layout_csv_read(F.path("tx.csv"), F.path("rx.csv"), {}, false);
    CHECK(qd_layout_diff(A, A2, 1.0e-9) == "");
    quadriga_lib::layout_csv_write(F.path("tx.csv"), F.path("rx.csv"), I);
    const auto I2 = quadriga_lib::layout_csv_read(F.path("tx.csv"), F.path("rx.csv"), {}, false);
    CHECK(qd_layout_diff(I, I2, 1.0e-9) == "");
}

TEST_CASE("generate_layout - family table and O2I draws")
{
    // The families: conf names (shipped files), LOS models, building O2I models with weights (TR 38.901 Table 7.8-1: 50 %
    // low, 50 % high loss; clause 7.4.3.1: RMa low loss only; Table 7.8-1A: SMa low-loss A), maximum d2D-in (clause
    // 7.4.3.1: 25 m UMa, UMi, SMa commercial; 10 m RMa, SMa residential), building shares (Tables 7.2-1, 7.2-3, 7.2-5),
    // residential share of SMa (Table 7.2-5: 90 %)
    struct row
    {
        const char *name, *los, *nlos, *los_o2i, *nlos_o2i, *model;
        std::vector<std::string> models;
        std::vector<double> weights;
        double din, din_c, share, p_res;
        bool car;
    };
    const std::vector<row> want = {
        {"UMi", "UMi_LOS", "UMi_NLOS", "UMi_LOS_O2I", "UMi_NLOS_O2I", "38.901_UMi", {"low", "high"}, {0.5, 0.5}, 25.0, 25.0, 0.8, 0.0, false},
        {"UMa", "UMa_LOS", "UMa_NLOS", "UMa_LOS_O2I", "UMa_NLOS_O2I", "38.901_UMa", {"low", "high"}, {0.5, 0.5}, 25.0, 25.0, 0.8, 0.0, false},
        {"RMa", "RMa_LOS", "RMa_NLOS", "RMa_LOS_O2I", "RMa_NLOS_O2I", "38.901_RMa", {"low"}, {1.0}, 10.0, 10.0, 0.5, 0.0, true},
        {"SMa", "SMa_LOS", "SMa_NLOS", "SMa_LOS_O2I", "SMa_NLOS_O2I", "38.901_SMa", {"low_A"}, {1.0}, 10.0, 25.0, 0.8, 0.9, true},
        {"InH_Open", "Indoor_LOS", "Indoor_NLOS", "", "", "38.901_InH_Open", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InH_Mixed", "Indoor_LOS", "Indoor_NLOS", "", "", "38.901_InH_Mixed", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_SL", "InF_LOS", "InF_NLOS_SL", "", "", "38.901_InF_SL", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_DL", "InF_LOS", "InF_NLOS_DL", "", "", "38.901_InF_DL", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_SH", "InF_LOS", "InF_NLOS_SH", "", "", "38.901_InF_SH", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_DH", "InF_LOS", "InF_NLOS_DH", "", "", "38.901_InF_DH", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_HH", "InF_LOS", "", "", "", "38.901_InF_HH", {}, {}, 0.0, 0.0, 0.0, 0.0, false}};
    const auto &T = qd_layout::families();
    REQUIRE(T.size() == want.size());
    for (size_t i = 0; i < want.size(); ++i)
    {
        const auto &f = T[i];
        const auto &w = want[i];
        INFO(w.name);
        CHECK(f.name == w.name);
        CHECK(f.los_conf == w.los);
        CHECK(f.nlos_conf == w.nlos);
        CHECK(f.los_o2i_conf == w.los_o2i);
        CHECK(f.nlos_o2i_conf == w.nlos_o2i);
        CHECK(f.los_model == w.model);
        CHECK(f.o2i_models == w.models);
        CHECK(f.o2i_weights == w.weights);
        CHECK(f.d2d_in_max == w.din);
        CHECK(f.d2d_in_max_commercial == w.din_c);
        CHECK(f.building_share == w.share);
        CHECK(f.p_residential == w.p_res);
        CHECK(f.car == w.car);
        for (const std::string &c : {f.los_conf, f.nlos_conf, f.los_o2i_conf, f.nlos_o2i_conf})
            if (!c.empty())
                CHECK(std::filesystem::exists("data/config/3GPP_38.901_v19_" + c + ".conf"));
        CHECK(qd_layout::find_family(w.name) == &f);
        // The model is known to los_probability
        CHECK_NOTHROW(quadriga_lib::los_probability(f.los_model, arma::vec{50.0}, arma::vec{10.0}, arma::vec{1.5}));
    }
    CHECK(qd_layout::find_family("UMa_LOS") == nullptr);
    CHECK(qd_layout::find_family("") == nullptr);
    CHECK(qd_layout::find_family("uma") == nullptr);

    // The fields at the documented seeds and decorrelation distances (TR clause 7.6.3.3, Table 7.6.3.1-2)
    const arma::mat xy = {{10.0, -20.0}, {35.0, 4.0}, {-300.0, 120.0}, {12.5, 77.0}, {1000.0, -1500.0}};
    const char *purpose[] = {"io", "o2i_class", "o2i_model", "d2din1", "d2din2", "o2i_dev"};
    const double dist[] = {50.0, 50.0, 50.0, 25.0, 25.0, 10.0};
    arma::mat rnd_want(5, 4);
    for (int s = 0; s < 6; ++s)
    {
        arma::mat val;
        const long long seed = qd_layout_oracle_seed(42, std::string(purpose[s]) + "|SMa");
        if (s == 5)
            quadriga_lib::sos_randn<double>(arma::vec{dist[s]}, xy, val, arma::mat(), arma::vec(), "Exp256", seed);
        else
            quadriga_lib::sos_rand<double>(arma::vec{dist[s]}, xy, val, arma::mat(), arma::vec(), "Exp256", seed);
        CHECK(arma::approx_equal(qd_layout::o2i_field(42, "SMa", s, xy), val.col(0), "absdiff", 1.0e-14));
        if (s >= 2)
            rnd_want.col((arma::uword)s - 2) = val.col(0);
    }
    const arma::mat rnd = qd_layout::o2i_rnd(42, "SMa", xy);
    CHECK(arma::approx_equal(rnd, rnd_want, "absdiff", 1.0e-14));

    // The indoor distance agrees with o2i_loss
    arma::vec din;
    quadriga_lib::o2i_loss(arma::vec{3.5e9}, rnd, {"low", "high"}, arma::vec{0.5, 0.5}, 25.0, &din);
    for (arma::uword i = 0; i < 5; ++i)
        CHECK(std::abs(qd_layout::d2d_in_of(rnd(i, 1), rnd(i, 2), 25.0) - din[i]) < 1.0e-12);

    // Floors: the documented draw (SplitMix64 from the stream seed), recomputed
    for (int k = 0; k < 40; ++k)
    {
        const std::string t = "UT" + std::to_string(k + 1);
        unsigned long long s = (unsigned long long)qd_layout_oracle_seed(9, "floor|UMa|" + t);
        auto splitmix = [&s]()
        {
            unsigned long long z = (s += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return (double)((z ^ (z >> 31)) >> 11) / 9007199254740992.0;
        };
        const double v1 = splitmix(), v2 = splitmix();
        const int n_fl = 4 + (int)std::floor(v1 * 5.0), f = 1 + (int)std::floor(v2 * n_fl);
        CHECK(qd_layout::draw_floor(9, "UMa", t, 4, 8) == f);
        const int g = qd_layout::draw_floor(9, "SMa", t, 2, 2);
        CHECK((g == 1 || g == 2));
    }

    // Floor from the height
    CHECK(qd_layout::floor_from_z(1.5) == 1);
    CHECK(qd_layout::floor_from_z(4.5) == 2);
    CHECK(qd_layout::floor_from_z(13.5) == 5);
    CHECK(qd_layout::floor_from_z(0.0) == 1);
    CHECK(qd_layout::floor_from_z(5.9) == 2);
    CHECK(qd_layout::floor_from_z(6.1) == 3);
}
