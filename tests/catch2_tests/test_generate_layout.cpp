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

// Lower end of the building share p after the minimum-distance rule to d2D-out: a building UT can only be rejected
// where d2D - d2D-in < d_min is possible, the fraction a of the area where outdoor UTs are accepted, and a rejected UT
// is redrawn with its position and state. The accepted building mass is at least p (1 - a), the outdoor mass is 1 - p,
// so the share lies in [p (1 - a) / (1 - p a), p]
double share_low(double p, double a) { return p * (1.0 - a) / (1.0 - p * a); }

// The fraction a of share_low for one hexagonal site with ISD 'isd' (inradius above d_min + d2d_in_max): the annulus
// [d_min, d_min + d2d_in_max) over the hexagon minus the disk of radius d_min
double annulus_share(double isd, double d_min, double din)
{
    const double pi = arma::datum::pi, r = d_min + din;
    return pi * (r * r - d_min * d_min) / (std::sqrt(3.0) / 2.0 * isd * isd - pi * d_min * d_min);
}

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
    SECTION("ITU-R M.2412 options")
    {
        CHECK(G("M.2412_UrbanMacro", {}, {}, {}, {}, {}, {}, {{"config", {2.0}}, {"service", {1.0}}}) == "");
        CHECK(has(G("M.2412_InH", {}, {}, {}, {}, {}, {}, {{"config", {4.0}}}), "'config'"));
        CHECK(has(G("M.2412_InH", {}, {}, {}, {}, {}, {}, {{"config", {1.5}}}), "'config'"));
        CHECK(has(G("M.2412_InH", {}, {}, {}, {}, {}, {}, {{"config", {1.0, 2.0}}}), "'config'"));
        CHECK(has(G("M.2412_UrbanMacro", {}, {}, {}, {}, {}, {}, {{"config", {3.0}}}), "'config'"));
        CHECK(has(G("M.2412_UrbanMacro", {}, {}, {}, {}, {}, {}, {{"service", {3.0}}}), "'service'"));
        CHECK(has(G("M.2412_Rural", {}, {}, {}, {}, {}, {}, {{"service", {1.0}}}), "'service'"));
        CHECK(has(G("M.2412_DenseUrban", {}, {}, {}, {}, {}, {}, {{"h_micro", {10.0}}}), "'h_micro'"));
        CHECK(has(G("M.2412_DenseUrban", {}, {}, {}, {}, {}, {}, {{"config", {3.0}}, {"h_micro", {NaN}}}), "'h_micro'"));
        CHECK(has(G("M.2412_DenseUrban", {}, {}, {}, {}, {}, {}, {{"config", {3.0}}, {"h_micro", {0.0}}}), "'h_micro'"));
        CHECK(has(G("M.2412_DenseUrban", {}, {}, {}, {}, {}, {}, {{"config", {3.0}}, {"h_micro", {-5.0}}}), "'h_micro'"));
        CHECK(has(G("38.901_UMa", {}, {1.0}, {}, {}, {}, {}, {{"config", {1.0}}}), "'config'"));
        CHECK(has(G("M.2412_InH", {}, {}, {}, {}, {}, {}, {{"lane_width", {3.0}}}), "'lane_width'"));
        CHECK(has(G("M.2412_InH", {}, {}, {}, {}, {}, {}, {{"xyz", {3.0}}}), "unknown"));
        CHECK(has(G("M.2412_InH", {}, {}, {}, {}, {}, {0.5}), "'indoor ratio'"));
        CHECK(has(G("M.2412_UrbanMacro", {}, {}, {}, {}, {}, {NaN, 4.0}), "'min floors'"));
    }
    SECTION("TR 37.885 roads")
    {
        auto V = [](const std::string &type, arma::uword n_ut, const arma::vec &spacing = {}, const arma::vec &grid = {}, const arma::vec &area = {},
                    const arma::vec &bs = {}, const arma::vec &ut = {}, const arma::vec &indoor = {}, const opts &o = {})
        { return qd_layout_throws([&] { quadriga_lib::generate_layout(type, n_ut, spacing, grid, area, bs, ut, indoor, o, true, 1); }); };
        CHECK(V("37.885_Highway", 0) == "");
        CHECK(V("37.885_Urban", 0, {433.0, 250.0}, {}, {1299.0, 750.0}, {}, {1.5, NaN, 10.0, 0.0, 0.0, 0.0}, {}, {{"p_truck", {0.5}}}) == "");
        CHECK(has(V("37.885_Highway", 1), "'n_ut'"));
        CHECK(has(V("37.885_Urban", 5), "'n_ut'"));
        CHECK(has(V("37.885_Highway", 0, {100.0}), "'d_x'"));
        CHECK(has(V("37.885_Urban", 0, {}, {1.0}), "'n_x'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {2000.0, 30.0}), "'width'"));
        CHECK(has(V("37.885_Urban", 0, {}, {}, {NaN, NaN, 5.0}), "'height'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {25.0}), "'bs'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {}, {NaN, 10.0}), "'min distance'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {}, {NaN, NaN, -1.0}), "'speed'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {}, {}, {0.5}), "'indoor ratio'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {0.0}), "'length'"));
        CHECK(has(V("37.885_Urban", 0, {15.0, 250.0}), "(20 m wide")); // Narrower than the street (20 m)
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {{"p_truck", {1.5}}}), "'p_truck'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {{"n_lane", {0.0}}}), "'n_lane'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {{"n_lane", {1.5}}}), "'n_lane'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {{"lane_width", {0.0}}}), "'lane_width'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {{"config", {1.0}}}), "'config'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {{"p_residential", {0.5}}}), "'p_residential'"));
        CHECK(has(V("37.885_Highway", 0, {}, {}, {1.0e9}), "vehicles")); // More than 10^7 vehicles
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
    const in_row in[] = {{"38.901_InH_Open", "InH_Open", 120.0, 50.0, NaN, 20.0, 6, 2, 3.0, 1.0, 0.0, 120.0},
                         {"38.901_InH_Mixed", "InH_Mixed", 120.0, 50.0, NaN, 20.0, 6, 2, 3.0, 1.0, 0.0, 120.0},
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

        // Hall and office parameters: V/S of the InF hall (TR Table 7.5-6 Part-3, DS of InF) and the largest dimension of
        // the hall or office (clause 7.6.9; InH: 120 m x 50 m x 3 m, L = 120 m)
        if (!std::isnan(p.hgt))
        {
            REQUIRE(L.tx.scenpar_override.count("DS_VS") == 1);
            const double vs = p.len * p.wid * p.hgt / (2.0 * (p.len * p.wid + p.len * p.hgt + p.wid * p.hgt));
            CHECK(arma::approx_equal(L.tx.scenpar_override.at("DS_VS"), arma::vec(L.tx.n_node(), arma::fill::value(vs)), "absdiff", 1.0e-12));
        }
        else
            CHECK(L.tx.scenpar_override.count("DS_VS") == 0);
        CHECK(L.tx.scenpar_override.size() == (std::isnan(p.hgt) ? 1 : 2));
        REQUIRE(L.tx.scenpar_override.count("absTOA_Lmax") == 1);
        CHECK(arma::all(L.tx.scenpar_override.at("absTOA_Lmax") == p.l_max));
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

    // An office whose largest dimension is the width: 40 m x 70 m, 6 x 2 BSs at 5 m and 20 m
    const auto O = gen("38.901_InH_Mixed", 0, {5.0, 20.0}, {}, {40.0, 70.0});
    REQUIRE(O.tx.scenpar_override.count("absTOA_Lmax") == 1);
    CHECK(arma::all(O.tx.scenpar_override.at("absTOA_Lmax") == 70.0));
    CHECK(O.tx.scenpar_override.count("DS_VS") == 0);
    // An office smaller than its height of 3 m (Table 7.2-2): L = 3 m; one BS
    const auto Q = gen("38.901_InH_Open", 0, {1.0, 1.0}, {1.0, 1.0}, {2.0, 2.5});
    REQUIRE(Q.tx.scenpar_override.count("absTOA_Lmax") == 1);
    CHECK(arma::all(Q.tx.scenpar_override.at("absTOA_Lmax") == 3.0));
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
    // Indoor share 0.8 (Table 7.2-1), lowered by the d2D-out rule (share_low)
    const double b_in = share_bound(0.8, 420.0, hex_area(7.0, 500.0));
    CHECK((double)n_in / 420.0 > share_low(0.8, annulus_share(500.0, 35.0, 25.0)) - b_in);
    CHECK((double)n_in / 420.0 < 0.8 + b_in);

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

TEST_CASE("generate_layout - ITU-R M.2412 presets")
{
    // Report ITU-R M.2412-0, Table 5: b) Dense Urban-eMBB, configs A and B (macro ISD 200 m, hBS 25 m; 80 % indoor, 20 %
    // outdoor in cars; 3 km/h indoors, 30 km/h in cars; hUT 1.5 m outdoors, 3 (n_fl - 1) + 1.5 indoors with
    // n_fl ~ U(1, N_fl), N_fl ~ U(4, 8)); c) Rural-eMBB (hBS 35 m, hUT 1.5 m; configs A and B: ISD 1732 m, 50 % indoor,
    // 50 % in cars at 120 km/h; config C: ISD 6000 m, 40 % indoor, 40 % pedestrian, 20 % in cars at 30 km/h; indoor and
    // pedestrian 3 km/h); d) Urban Macro-mMTC (ISD 500 m in config A, 1732 m in config B; hBS 25 m; 80 % indoor, 20 %
    // outdoor, 3 km/h; hUT 1.5 m); e) Urban Macro-URLLC (configs A and B: ISD 500 m, hBS 25 m; 80 % outdoor at 30 km/h,
    // 20 % indoor at 3 km/h; hUT 1.5 m). Section 8.3: 19 sites with 3 TRxPs, wrap-around. Annex 1 Table A1-1 maps the
    // environments to the UMa and RMa models; the drop uses their O2I fields and indoor distances (Annex 1 Tables A1-7
    // and A1-8: the minimum of two uniforms up to 25 m UMa, 10 m RMa). The Report gives no minimum distance: the library
    // takes the TR 38.901 value of
    // the mapped scenario (Tables 7.2-1, 7.2-3: 35 m; UMa: to d2D-out for indoor UTs, TR 36.873 Table 6-1, note 1)
    struct row
    {
        const char *preset;
        double config, service, isd, h_bs, p_in, p_car, v_in, v_ped, v_car;
        bool floors;
        const char *fam;
        double din;
        bool d_out;
    };
    const row rows[] = {{"M.2412_DenseUrban", 1.0, NaN, 200.0, 25.0, 0.8, 0.2, 3.0, NaN, 30.0, true, "UMa", 25.0, true},
                        {"M.2412_DenseUrban", 2.0, NaN, 200.0, 25.0, 0.8, 0.2, 3.0, NaN, 30.0, true, "UMa", 25.0, true},
                        {"M.2412_Rural", 1.0, NaN, 1732.0, 35.0, 0.5, 0.5, 3.0, NaN, 120.0, false, "RMa", 10.0, false},
                        {"M.2412_Rural", 2.0, NaN, 1732.0, 35.0, 0.5, 0.5, 3.0, NaN, 120.0, false, "RMa", 10.0, false},
                        {"M.2412_Rural", 3.0, NaN, 6000.0, 35.0, 0.4, 0.2, 3.0, 3.0, 30.0, false, "RMa", 10.0, false},
                        {"M.2412_UrbanMacro", 1.0, 1.0, 500.0, 25.0, 0.8, 0.0, 3.0, 3.0, NaN, false, "UMa", 25.0, true},
                        {"M.2412_UrbanMacro", 2.0, 1.0, 1732.0, 25.0, 0.8, 0.0, 3.0, 3.0, NaN, false, "UMa", 25.0, true},
                        {"M.2412_UrbanMacro", 1.0, 2.0, 500.0, 25.0, 0.2, 0.0, 3.0, 30.0, NaN, false, "UMa", 25.0, true},
                        {"M.2412_UrbanMacro", 2.0, 2.0, 500.0, 25.0, 0.2, 0.0, 3.0, 30.0, NaN, false, "UMa", 25.0, true}};
    for (const auto &p : rows)
    {
        INFO(p.preset << " config " << p.config << " service " << p.service);
        opts o = {{"config", {p.config}}};
        if (!std::isnan(p.service))
            o["service"] = std::vector<double>(1, p.service);
        const auto L = gen(p.preset, 2, {}, {}, {}, {}, {}, {}, o);
        const auto S = qd_layout_sites(L.tx);
        REQUIRE(S.name.size() == 19);
        REQUIRE(L.tx.n_node() == 57);
        double dmin = arma::datum::inf;
        for (arma::uword j = 1; j < 19; ++j)
            dmin = std::min(dmin, std::hypot(S.xy(0, j) - S.xy(0, 0), S.xy(1, j) - S.xy(1, 0)));
        CHECK(std::abs(dmin - p.isd) < 1.0e-9);
        bool ok = true;
        for (arma::uword i = 0; i < 57; ++i)
            ok = ok && L.tx.pos(2, i) == p.h_bs && L.tx.orientation(1, i) == 0.0 &&
                 std::abs(qd_layout_wrap(L.tx.orientation(2, i) - (30.0 + 120.0 * (double)(i % 3)) * D)) < 1.0e-12;
        CHECK(ok);
        CHECK(L.tx.scenario.empty()); // The user sets the scenario
        CHECK(L.wrap.n_cols == 2);
        CHECK(L.is_valid() == "");

        // UT states from the O2I fields of the mapped model at the documented seeds: building if U < p_in, in a car if
        // U < p_in + p_car, else outdoors
        REQUIRE(L.rx.n_node() == 114);
        REQUIRE(L.rx.indoor.n_elem == 114);
        const std::string fam = p.fam;
        const arma::vec io = field(L, "io|" + fam, 50.0), u1 = field(L, "d2din1|" + fam, 25.0), u2 = field(L, "d2din2|" + fam, 25.0);
        const arma::vec d = site_distance(L);
        ok = true;
        for (arma::uword i = 0; i < 114; ++i)
        {
            const int s = io[i] < p.p_in ? 1 : io[i] < p.p_in + p.p_car ? 2 : 0;
            const double din = s == 1 ? std::min(u1[i], u2[i]) * p.din : 0.0;
            ok = ok && L.rx.indoor[i] == s && std::abs(L.rx.d2d_in[i] - din) < 1.0e-12;
            ok = ok && std::abs(L.rx.speed[i] - (s == 1 ? p.v_in : s == 2 ? p.v_car : p.v_ped) * KMH) < 1.0e-12;
            ok = ok && d[i] - (p.d_out ? din : 0.0) >= 35.0 - 1.0e-9;
            const int f = L.rx.floor[i];
            if (p.floors && s == 1)
                ok = ok && f >= 1 && f <= 8 && std::abs(L.rx.pos(2, i) - (3.0 * (f - 1) + 1.5)) < 1.0e-12;
            else
                ok = ok && f == 1 && L.rx.pos(2, i) == 1.5;
        }
        CHECK(ok);
    }

    // Shares with 10 UTs per TRxP (570 UTs): Rural config C and URLLC
    const auto R = gen("M.2412_Rural", 10, {}, {}, {}, {}, {}, {}, {{"config", {3.0}}});
    CHECK(std::abs(share_of(R, 1) - 0.4) < share_bound(0.4, 570.0, hex_area(19.0, 6000.0)));
    CHECK(std::abs(share_of(R, 2) - 0.2) < share_bound(0.2, 570.0, hex_area(19.0, 6000.0)));
    CHECK(std::abs(share_of(R, 0) - 0.4) < share_bound(0.4, 570.0, hex_area(19.0, 6000.0)));
    // URLLC and Dense Urban config A: the d2D-out rule lowers the building share (share_low); RMa has no such rule
    const auto U = gen("M.2412_UrbanMacro", 10, {}, {}, {}, {}, {}, {}, {{"service", {2.0}}});
    const double b_u = share_bound(0.2, 570.0, hex_area(19.0, 500.0));
    CHECK(share_of(U, 1) > share_low(0.2, annulus_share(500.0, 35.0, 25.0)) - b_u);
    CHECK(share_of(U, 1) < 0.2 + b_u);
    CHECK(share_of(U, 2) == 0.0);
    const auto DA = gen("M.2412_DenseUrban", 10);
    const double b_d = share_bound(0.8, 570.0, hex_area(19.0, 200.0));
    CHECK(share_of(DA, 1) > share_low(0.8, annulus_share(200.0, 35.0, 25.0)) - b_d);
    CHECK(share_of(DA, 1) < 0.8 + b_d);
    CHECK(std::abs(share_of(DA, 1) + share_of(DA, 2) - 1.0) < 1.0e-12);

    // Defaults: config A (1), service mMTC (1)
    auto first_isd = [](const quadriga_lib::layout &L) { return std::hypot(L.tx.pos(0, 3) - L.tx.pos(0, 0), L.tx.pos(1, 3) - L.tx.pos(1, 0)); };
    CHECK(std::abs(first_isd(gen("M.2412_UrbanMacro", 0)) - 500.0) < 1.0e-9);
    CHECK(std::abs(first_isd(gen("M.2412_UrbanMacro", 0, {}, {}, {}, {}, {}, {}, {{"config", {2.0}}})) - 1732.0) < 1.0e-9);
    CHECK(std::abs(first_isd(gen("M.2412_Rural", 0)) - 1732.0) < 1.0e-9);
    CHECK(gen("M.2412_DenseUrban", 0).tx.n_node() == 57);

    // Indoor Hotspot-eMBB (Table 5 a), section 8.3.1, Figure 1): one floor of 120 m x 50 m, 12 BSs at 20 m spacing, 10 m
    // from the short and 15 m from the long walls, i.e. x = -50:20:50, y = +-10 m; hBS 3 m, hUT 1.5 m, 100 % indoor at
    // 3 km/h; the configurations differ only in frequency
    for (double c : {1.0, 2.0, 3.0})
    {
        INFO("InH config " << c);
        const auto H = gen("M.2412_InH", 2, {}, {}, {}, {}, {}, {}, {{"config", {c}}});
        REQUIRE(H.tx.n_node() == 12);
        std::set<std::pair<long, long>> want, got;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 2; ++j)
                want.insert({std::lround(1000.0 * (-50.0 + 20.0 * i)), std::lround(1000.0 * (-10.0 + 20.0 * j))});
        bool ok = true;
        for (arma::uword i = 0; i < 12; ++i)
        {
            got.insert({std::lround(1000.0 * H.tx.pos(0, i)), std::lround(1000.0 * H.tx.pos(1, i))});
            ok = ok && H.tx.pos(2, i) == 3.0 && H.tx.name[i] == "BS" + std::to_string(i + 1);
        }
        CHECK(ok);
        CHECK(got == want);
        CHECK(H.tx.scenario.empty());
        CHECK(H.wrap.n_elem == 0);
        CHECK(H.tx.scenpar_override.empty());
        REQUIRE(H.rx.n_node() == 24);
        ok = true;
        for (arma::uword i = 0; i < 24; ++i)
            ok = ok && std::abs(H.rx.pos(0, i)) <= 60.0 && std::abs(H.rx.pos(1, i)) <= 25.0 && H.rx.pos(2, i) == 1.5 &&
                 std::abs(H.rx.speed[i] - 3.0 * KMH) < 1.0e-12;
        CHECK(ok);
        CHECK(H.rx.indoor.n_elem == 0);
        CHECK(H.is_valid() == "");
    }
}

TEST_CASE("generate_layout - M.2412 Dense Urban micro layer")
{
    // Report ITU-R M.2412-0 section 8.3.2: 3 micro sites randomly dropped in each macro TRxP area (config C only);
    // Table 5 b): micro sites 10 m high, 10 UEs per TRxP, uniform. The library drops the micro sites uniformly in the
    // macro TRxP cell, each with one TRxP, at least 35 m from the macro sites (the UMa minimum distance) and 20 m from each
    // other (twice the UMi minimum distance, TR 38.901 Table 7.2-1). The micro sites form no cells: a macro TRxP cell
    // gets n_ut UTs for its own and for each of its micro TRxPs, and the UTs keep 10 m from the micro sites (to d2D-out
    // indoors)
    const auto L = gen("M.2412_DenseUrban", 1, {}, {}, {}, {}, {}, {}, {{"config", {3.0}}});
    const auto S = qd_layout_sites(L.tx);
    REQUIRE(S.name.size() == 19 + 171);
    REQUIRE(L.tx.n_node() == 57 + 171);
    CHECK(L.is_valid() == "");
    CHECK(L.tx.scenario.empty());

    // Macro TRxP cells of the macro layer alone, by brute force
    const auto M = gen("M.2412_DenseUrban", 0);
    const auto SM = qd_layout_sites(M.tx);
    const arma::mat I = qd_layout_images(M.wrap);
    std::vector<int> per_cell(57, 0);
    bool ok = true, order = true;
    for (arma::uword j = 0; j < 171; ++j)
    {
        const arma::uword n = 57 + j, r = L.tx.offset[n];
        ok = ok && L.tx.name[n] == "BS" + std::to_string(20 + j) && L.tx.sector[n] == 0 && L.tx.pos(2, r) == 10.0 &&
             L.tx.orientation(0, r) == 0.0 && L.tx.orientation(1, r) == 0.0 && L.tx.orientation(2, r) == 0.0;
        const long long c = cell_oracle(M, SM, I, L.tx.pos(0, r), L.tx.pos(1, r));
        ok = ok && c >= 0;
        if (c >= 0)
            ++per_cell[(size_t)c], order = order && (arma::uword)c == j / 3;
    }
    CHECK(ok);
    CHECK(order);
    CHECK(std::all_of(per_cell.begin(), per_cell.end(), [](int v) { return v == 3; }));
    // The macro layer is that of configs A and B
    CHECK(arma::approx_equal(L.tx.pos.cols(0, 56), M.tx.pos, "absdiff", 1.0e-12));

    // Distance of each micro site to the nearest image of every other site
    const arma::mat IL = qd_layout_images(L.wrap);
    double to_macro = arma::datum::inf, to_micro = arma::datum::inf;
    for (arma::uword a = 19; a < 190; ++a)
        for (arma::uword b = 0; b < 190; ++b)
            for (arma::uword k = 0; k < IL.n_cols; ++k)
                if (a != b || k != 0)
                {
                    const double d = std::hypot(S.xy(0, a) - S.xy(0, b) - IL(0, k), S.xy(1, a) - S.xy(1, b) - IL(1, k));
                    (b < 19 ? to_macro : to_micro) = std::min(b < 19 ? to_macro : to_micro, d);
                }
    CHECK(to_macro >= 35.0 - 1.0e-9);
    CHECK(to_micro >= 20.0 - 1.0e-9);

    // UTs: one per TRxP, i.e. 4 in each macro TRxP cell; 35 m from the macro sites and 10 m from the micro sites (indoors
    // to d2D-out)
    REQUIRE(L.rx.n_node() == 228);
    std::vector<int> in_cell(58, 0);
    for (arma::uword i = 0; i < 228; ++i)
    {
        const long long n = cell_oracle(M, SM, I, L.rx.pos(0, i), L.rx.pos(1, i));
        ++in_cell[n < 0 ? 57 : (size_t)n];
    }
    CHECK(in_cell[57] == 0);
    CHECK(std::all_of(in_cell.begin(), in_cell.end() - 1, [](int v) { return v == 4; }));
    ok = true;
    for (arma::uword i = 0; i < 228; ++i)
    {
        const double din = L.rx.indoor[i] == 1 ? L.rx.d2d_in[i] : 0.0;
        for (arma::uword b = 0; b < 190; ++b)
            for (arma::uword k = 0; k < IL.n_cols; ++k)
                ok = ok && std::hypot(L.rx.pos(0, i) - S.xy(0, b) - IL(0, k), L.rx.pos(1, i) - S.xy(1, b) - IL(1, k)) - din >= (b < 19 ? 35.0 : 10.0) - 1.0e-9;
    }
    CHECK(ok);
    // In total: 50 UTs over the macro cells
    CHECK(gen("M.2412_DenseUrban", 50, {}, {}, {}, {}, {}, {}, {{"config", {3.0}}}, false).rx.n_node() == 50);

    // UT states (Table 5 b): 80 % in buildings with floors 3 (n_fl - 1) + 1.5, N_fl ~ U(4, 8), at 3 km/h; 20 % in cars at
    // 30 km/h and 1.5 m. 600 UTs in total, uniform over the cluster. The d2D-out rule lowers the building share
    // (share_low); its area fraction a is counted on a 10 m grid over the cluster: E = points >= 35 m from the macro and
    // >= 10 m from the micro sites (all images), R = points of E within 35 + 25 m of a macro or 10 + 25 m of a micro site
    const auto P = gen("M.2412_DenseUrban", 600, {}, {}, {}, {}, {}, {}, {{"config", {3.0}}}, false, 3);
    REQUIRE(P.rx.n_node() == 600);
    const auto SP = qd_layout_sites(P.tx);
    double nE = 0.0, nR = 0.0;
    for (double x = -700.0; x <= 700.0; x += 10.0)
        for (double y = -700.0; y <= 700.0; y += 10.0)
        {
            double d_mac = arma::datum::inf, d_mic = arma::datum::inf; // Squared distances
            arma::uword best_k = 0;
            for (arma::uword b = 0; b < 190; ++b)
                for (arma::uword k = 0; k < IL.n_cols; ++k)
                {
                    const double dx = x - SP.xy(0, b) - IL(0, k), dy = y - SP.xy(1, b) - IL(1, k), d = dx * dx + dy * dy;
                    if (b < 19 && d < d_mac)
                        d_mac = d, best_k = k;
                    if (b >= 19)
                        d_mic = std::min(d_mic, d);
                }
            if (best_k != 0 || d_mac < 35.0 * 35.0 || d_mic < 10.0 * 10.0) // Outside the cluster or no UT at all
                continue;
            nE += 1.0;
            nR += d_mac < 60.0 * 60.0 || d_mic < 35.0 * 35.0 ? 1.0 : 0.0;
        }
    const double b_p = share_bound(0.8, 600.0, hex_area(19.0, 200.0)), p_in = share_of(P, 1);
    CHECK(p_in > share_low(0.8, nR / nE) - b_p);
    CHECK(p_in < 0.8 + b_p);
    CHECK(std::abs(p_in + share_of(P, 2) - 1.0) < 1.0e-12); // All other UTs in cars
    ok = true;
    for (arma::uword i = 0; i < 600; ++i)
    {
        const int st = P.rx.indoor[i], f = P.rx.floor[i];
        ok = ok && std::abs(P.rx.speed[i] - (st == 1 ? 3.0 : 30.0) * KMH) < 1.0e-12;
        if (st == 1)
            ok = ok && f >= 1 && f <= 8 && std::abs(P.rx.pos(2, i) - (3.0 * (f - 1) + 1.5)) < 1.0e-12 && P.rx.d2d_in[i] <= 25.0;
        else
            ok = ok && st == 2 && f == 1 && P.rx.pos(2, i) == 1.5 && P.rx.d2d_in[i] == 0.0;
    }
    CHECK(ok);

    // Seeds: equal seeds give equal layouts, other seeds other micro sites
    CHECK(qd_layout_diff(L, gen("M.2412_DenseUrban", 1, {}, {}, {}, {}, {}, {}, {{"config", {3.0}}})) == "");
    CHECK(!arma::approx_equal(L.tx.pos, gen("M.2412_DenseUrban", 1, {}, {}, {}, {}, {}, {}, {{"config", {3.0}}}, true, 8).tx.pos, "absdiff", 1.0));

    // A small ISD: the cells of 100 m hold their sites (a cell whose sites block each other is placed again)
    for (arma::sword sd = 1; sd <= 2; ++sd)
    {
        const auto Q = gen("M.2412_DenseUrban", 0, {100.0}, {}, {}, {}, {}, {}, {{"config", {3.0}}}, true, sd);
        CHECK(Q.tx.n_node() == 228);
    }

    // drop_terminals on the layout as returned (no scenario): every TRxP is a cell, here with exactly one UT
    auto D2 = gen("M.2412_DenseUrban", 0, {}, {}, {}, {}, {}, {}, {{"config", {3.0}}});
    quadriga_lib::drop_terminals(D2, 1, {}, {}, {}, {}, true, 4);
    REQUIRE(D2.rx.n_node() == 228);
    auto c2 = count_cells(D2);
    CHECK(c2.back() == 0);
    c2.pop_back();
    CHECK(std::all_of(c2.begin(), c2.end(), [](arma::uword v) { return v == 1; }));

    // Option h_micro
    const auto H = gen("M.2412_DenseUrban", 0, {}, {}, {}, {}, {}, {}, {{"config", {3.0}}, {"h_micro", {6.0}}});
    CHECK(arma::all(H.tx.pos.row(2).cols(57, 227) == 6.0));
    CHECK(arma::all(H.tx.pos.row(2).cols(0, 56) == 25.0));
}

namespace
{
// Vehicle length from the antenna height: TR 37.885 clause 6.1.2, type 2 (5 m, antenna 1.6 m), type 3 (13 m, 3 m)
double vehicle_length(double h) { return h == 3.0 ? 13.0 : 5.0; }

// The vehicles of a road layout by lane: key (lane coordinate in mm, direction in mdeg), value (position along the
// lane in the direction of travel, length), sorted by position
std::map<std::pair<long, long>, std::vector<std::pair<double, double>>> lanes_of(const quadriga_lib::layout &L)
{
    std::map<std::pair<long, long>, std::vector<std::pair<double, double>>> M;
    for (arma::uword i = 0; i < L.rx.n_node(); ++i)
    {
        const double az = L.rx.direction(0, i), c = std::cos(az), s = std::sin(az);
        const bool along_x = std::abs(c) > 0.5;
        const double lane = along_x ? L.rx.pos(1, i) : L.rx.pos(0, i);
        const double u = along_x ? L.rx.pos(0, i) * (c > 0.0 ? 1.0 : -1.0) : L.rx.pos(1, i) * (s > 0.0 ? 1.0 : -1.0);
        M[{std::lround(1000.0 * lane), std::lround(1000.0 * az / D)}].push_back({u, vehicle_length(L.rx.pos(2, i))});
    }
    for (auto &kv : M)
        std::sort(kv.second.begin(), kv.second.end());
    return M;
}

// Bumper-to-bumper gaps of the vehicles of a lane; 'loop' > 0: the lane is a loop of this length (the last gap closes it)
std::vector<double> gaps_of(const std::vector<std::pair<double, double>> &v, double loop)
{
    std::vector<double> g;
    for (size_t k = 1; k < v.size(); ++k)
        g.push_back(v[k].first - v[k - 1].first - 0.5 * (v[k].second + v[k - 1].second));
    if (loop > 0.0 && v.size() > 1)
        g.push_back(v[0].first + loop - v.back().first - 0.5 * (v[0].second + v.back().second));
    return g;
}
} // namespace

TEST_CASE("generate_layout - TR 37.885 road layouts")
{
    // TR 37.885 Table A-1: highway with 3 lanes per direction, lane width 4 m, length >= 2000 m, wrap-around; urban grid
    // with 2 lanes per direction, lane width 3.5 m, 3 m sidewalk, road grid 433 m x 250 m, area at least 1299 m x 750 m.
    // Clause 6.1.2: bumper-to-bumper gap max(2 m, Exp(mean = speed x 2 s)); option A: 100 % type 2 (5 m long, antenna at
    // 1.6 m), highway 140 km/h, urban 60 km/h. No BSs. Lanes: right-hand traffic (the TR does not say)
    auto L = gen("37.885_Highway", 0);
    CHECK(L.tx.n_node() == 0);
    CHECK(L.tx.scenario.empty());
    REQUIRE(L.wrap.n_rows == 2);
    REQUIRE(L.wrap.n_cols == 1);
    CHECK(L.wrap(0, 0) == 2000.0);
    CHECK(L.wrap(1, 0) == 0.0);
    CHECK(L.is_valid() == "");
    const arma::uword n = L.rx.n_node();
    REQUIRE(n > 100);
    bool ok = true;
    for (arma::uword i = 0; i < n; ++i)
    {
        const double y = L.rx.pos(1, i), az = L.rx.direction(0, i);
        const bool lane = std::abs(std::abs(y) - 2.0) < 1.0e-12 || std::abs(std::abs(y) - 6.0) < 1.0e-12 || std::abs(std::abs(y) - 10.0) < 1.0e-12;
        ok = ok && lane && std::abs(qd_layout_wrap(az - (y < 0.0 ? 0.0 : arma::datum::pi))) < 1.0e-12;
        ok = ok && L.rx.name[i] == "UT" + std::to_string(i + 1) && L.rx.pos(2, i) == 1.6 && std::abs(L.rx.pos(0, i)) <= 1000.0;
        ok = ok && std::abs(L.rx.speed[i] - 140.0 * KMH) < 1.0e-12 && L.rx.direction(1, i) == 0.0;
        ok = ok && L.rx.orientation(0, i) == 0.0 && L.rx.orientation(1, i) == 0.0 && std::abs(qd_layout_wrap(L.rx.orientation(2, i) - az)) < 1.0e-12;
    }
    CHECK(ok);
    CHECK(L.rx.indoor.n_elem == 0);
    auto lanes = lanes_of(L);
    CHECK(lanes.size() == 6);
    double g_min = arma::datum::inf;
    for (const auto &kv : lanes)
        for (double g : gaps_of(kv.second, 2000.0))
            g_min = std::min(g_min, g);
    CHECK(g_min >= 2.0 - 1.0e-9);

    // Speed 0: every gap is 2 m, so the vehicles follow at 7 m (gap plus length); a 2000 m loop holds 285 of them per lane
    // (7 n <= 2000), the gap that closes the loop is 2000 - 284 x 7 - 5 = 7 m, at a random place of the loop
    L = gen("37.885_Highway", 0, {}, {}, {}, {}, {NaN, NaN, 0.0});
    lanes = lanes_of(L);
    REQUIRE(lanes.size() == 6);
    for (const auto &kv : lanes)
    {
        CHECK(kv.second.size() == 285);
        const auto g = gaps_of(kv.second, 2000.0);
        CHECK(std::count_if(g.begin(), g.end(), [](double v) { return std::abs(v - 2.0) < 1.0e-9; }) == 284);
        CHECK(std::count_if(g.begin(), g.end(), [](double v) { return std::abs(v - 7.0) < 1.0e-9; }) == 1);
    }
    CHECK(arma::all(L.rx.speed == 0.0));
    // Trucks only: 15 m steps, 133 per lane (15 n <= 2000), antenna at 3 m
    L = gen("37.885_Highway", 0, {}, {}, {}, {}, {NaN, NaN, 0.0}, {}, {{"p_truck", {1.0}}});
    lanes = lanes_of(L);
    for (const auto &kv : lanes)
        CHECK(kv.second.size() == 133);
    CHECK(arma::all(L.rx.pos.row(2) == 3.0));

    // Gap statistics on a 20 km highway: E[max(2, X)] = 2 + m exp(-2 / m) and E[max(2, X)^2] = 4 (1 - e) + e (4 + 4 m +
    // 2 m^2) with e = exp(-2 / m), m = 140 km/h x 2 s; P(gap = 2 m) = 1 - e. The mean within four standard deviations of
    // its own sample, P(gap = 2 m) within five of the binomial; the gap across x = +-L/2 is left out, the one gap per lane
    // that closes the loop (shorter, at a random place) stays in and is negligible against the bound
    L = gen("37.885_Highway", 0, {}, {}, {20000.0}, {}, {}, {}, {{"p_truck", {0.3}}});
    CHECK(L.wrap(0, 0) == 20000.0);
    lanes = lanes_of(L);
    std::vector<double> all;
    for (const auto &kv : lanes)
    {
        const auto g = gaps_of(kv.second, 0.0);
        all.insert(all.end(), g.begin(), g.end());
    }
    const double m = 140.0 * KMH * 2.0, e = std::exp(-2.0 / m), ng = (double)all.size();
    const double mu = 2.0 + m * e, var = 4.0 * (1.0 - e) + e * (4.0 + 4.0 * m + 2.0 * m * m) - mu * mu;
    REQUIRE(ng > 1000.0);
    double mean = 0.0, n2 = 0.0, sq = 0.0;
    for (double g : all)
        mean += g / ng, n2 += std::abs(g - 2.0) < 1.0e-9 ? 1.0 : 0.0;
    for (double g : all)
        sq += (g - mean) * (g - mean) / (ng - 1.0);
    CHECK(std::abs(sq - var) < 0.2 * var);
    CHECK(std::abs(mean - mu) < 4.0 * std::sqrt(sq / ng));
    CHECK(std::abs(n2 / ng - (1.0 - e)) < 5.0 * std::sqrt((1.0 - e) * e / ng));
    CHECK(*std::min_element(all.begin(), all.end()) >= 2.0 - 1.0e-9);
    // Truck share 0.3, per vehicle
    const double nv = (double)L.rx.n_node(), nt = (double)arma::accu(L.rx.pos.row(2) == 3.0);
    CHECK(std::abs(nt / nv - 0.3) < 5.0 * std::sqrt(0.21 / nv));

    // Options: 2 lanes per direction of 3 m: y = +-1.5, +-4.5
    L = gen("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {{"lane_width", {3.0}}, {"n_lane", {2.0}}});
    std::set<long> ys;
    for (arma::uword i = 0; i < L.rx.n_node(); ++i)
        ys.insert(std::lround(1000.0 * L.rx.pos(1, i)));
    CHECK(ys == std::set<long>{-4500, -1500, 1500, 4500});

    // Urban grid: streets at x = -433, 0, 433 and y = -250, 0, 250 (the road grid centred in the area); lanes 1.75 m and
    // 5.25 m from the centre line; eastbound and northbound on the right side; 60 km/h; vehicles inside the area
    L = gen("37.885_Urban", 0);
    CHECK(L.tx.n_node() == 0);
    CHECK(L.wrap.n_elem == 0);
    CHECK(L.is_valid() == "");
    std::set<std::pair<long, long>> want;
    for (double c : {-433.0, 0.0, 433.0})
        for (double o : {1.75, 5.25})
            want.insert({std::lround(1000.0 * (c + o)), 90000}), want.insert({std::lround(1000.0 * (c - o)), -90000});
    for (double c : {-250.0, 0.0, 250.0})
        for (double o : {1.75, 5.25})
            want.insert({std::lround(1000.0 * (c - o)), 0}), want.insert({std::lround(1000.0 * (c + o)), 180000});
    lanes = lanes_of(L);
    std::set<std::pair<long, long>> got;
    for (const auto &kv : lanes)
        got.insert({kv.first.first, std::abs(kv.first.second) == 180000 ? 180000 : kv.first.second});
    CHECK(got == want);
    ok = true;
    for (arma::uword i = 0; i < L.rx.n_node(); ++i)
        ok = ok && std::abs(L.rx.speed[i] - 60.0 * KMH) < 1.0e-12 && L.rx.pos(2, i) == 1.6 && std::abs(L.rx.pos(0, i)) <= 649.5 - 2.5 + 1.0e-9 &&
             std::abs(L.rx.pos(1, i)) <= 375.0 - 2.5 + 1.0e-9;
    CHECK(ok);
    for (const auto &kv : lanes)
        for (double g : gaps_of(kv.second, 0.0))
            CHECK(g >= 2.0 - 1.0e-9);

    // Speed 0 on the urban grid: 7 m steps; the free length at the two ends of a lane is below two steps
    L = gen("37.885_Urban", 0, {}, {}, {}, {}, {NaN, NaN, 0.0});
    for (const auto &kv : lanes_of(L))
    {
        const double len = std::abs(kv.first.second) == 90000 ? 750.0 : 1299.0;
        const auto &v = kv.second;
        const auto g = gaps_of(v, 0.0);
        CHECK(std::count_if(g.begin(), g.end(), [](double x) { return std::abs(x - 2.0) < 1.0e-9; }) == (long)g.size());
        const double free = (v.front().first - 2.5 + len / 2.0) + (len / 2.0 - v.back().first - 2.5);
        CHECK(free >= -1.0e-9);
        CHECK(free < 14.0);
    }

    // Spacing and area: 2 x 2 streets at +-250 m and +-150 m; one lane per direction of 3 m
    L = gen("37.885_Urban", 0, {500.0, 300.0}, {}, {1000.0, 600.0}, {}, {}, {}, {{"n_lane", {1.0}}, {"lane_width", {3.0}}});
    std::set<std::pair<long, long>> want2;
    for (double c : {-250.0, 250.0})
        want2.insert({std::lround(1000.0 * (c + 1.5)), 90000}), want2.insert({std::lround(1000.0 * (c - 1.5)), -90000});
    for (double c : {-150.0, 150.0})
        want2.insert({std::lround(1000.0 * (c - 1.5)), 0}), want2.insert({std::lround(1000.0 * (c + 1.5)), 180000});
    got.clear();
    for (const auto &kv : lanes_of(L))
        got.insert({kv.first.first, std::abs(kv.first.second) == 180000 ? 180000 : kv.first.second});
    CHECK(got == want2);

    // Determinism
    CHECK(qd_layout_diff(gen("37.885_Urban", 0, {}, {}, {}, {}, {}, {}, {}, true, 5), gen("37.885_Urban", 0, {}, {}, {}, {}, {}, {}, {}, true, 5)) == "");
    CHECK(qd_layout_diff(gen("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {}, true, 5), gen("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {}, true, 5)) == "");
    const auto H5 = gen("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {}, true, 5), H6 = gen("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {}, true, 6);
    CHECK((H5.rx.n_node() != H6.rx.n_node() || !arma::approx_equal(H5.rx.pos, H6.rx.pos, "absdiff", 1.0)));
}

TEST_CASE("generate_layout - TR 37.885 vehicles are stationary along the lanes")
{
    // A stationary process has the same mean density everywhere on the lane, also at its start and, on the highway, at the
    // wrap seam. Counts over many seeds in bins at the ends of the lanes against the mean density of the same layouts;
    // the counts of a renewal process with gaps of coefficient of variation below 1 vary less than Poisson counts, so four
    // Poisson standard deviations bound them. Highway 2000 m at 140 km/h (mean step 82.8 m), 200 seeds, 100 m bins at both
    // ends of the loop (the seam); urban grid at 60 km/h (mean step 38 m), 100 seeds, 50 m bins at both ends of each lane,
    // by the rear at the start and by the front at the end (a vehicle lies on its lane)
    double n_all = 0.0, n_first = 0.0, n_last = 0.0;
    for (arma::sword sd = 1; sd <= 200; ++sd)
    {
        const auto L = gen("37.885_Highway", 0, {}, {}, {}, {}, {}, {}, {}, true, sd);
        for (arma::uword i = 0; i < L.rx.n_node(); ++i)
        {
            const double u = L.rx.pos(0, i) * (std::cos(L.rx.direction(0, i)) > 0.0 ? 1.0 : -1.0) + 1000.0; // From the lane start
            n_all += 1.0, n_first += u < 100.0 ? 1.0 : 0.0, n_last += u >= 1900.0 ? 1.0 : 0.0;
        }
    }
    const double e_bin = n_all * 100.0 / 2000.0;
    CHECK(std::abs(n_first - e_bin) < 4.0 * std::sqrt(e_bin));
    CHECK(std::abs(n_last - e_bin) < 4.0 * std::sqrt(e_bin));

    double e_end = 0.0, u_first = 0.0, u_last = 0.0;
    for (arma::sword sd = 1; sd <= 100; ++sd)
        for (const auto &kv : lanes_of(gen("37.885_Urban", 0, {}, {}, {}, {}, {}, {}, {}, true, sd)))
        {
            const double len = std::abs(kv.first.second) == 90000 ? 750.0 : 1299.0;
            // Rears lie in [0, len - 5 m]: first bin by the rear, last bin by the front (position along the lane from its start)
            e_end += (double)kv.second.size() * 50.0 / (len - 5.0);
            for (const auto &v : kv.second)
                u_first += v.first + len / 2.0 - 2.5 < 50.0 ? 1.0 : 0.0, u_last += v.first + len / 2.0 + 2.5 > len - 50.0 ? 1.0 : 0.0;
        }
    CHECK(std::abs(u_first - e_end) < 4.0 * std::sqrt(e_end));
    CHECK(std::abs(u_last - e_end) < 4.0 * std::sqrt(e_end));
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
