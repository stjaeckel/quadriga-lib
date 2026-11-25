// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of get_channels_layout, the one call from a layout to channel coefficients, and of the rules that tie the
// stages together: the manual path (init_link_states, generate_rays, get_gbsm_channels) as oracle of the one call, the
// stale-link hash, edited link states, pairing, dual mobility, per-band antennas and layouts read back from CSV files.
// Oracles: the manual path; the exact names of the conf files (an edited LOS state equals the state fixed by name);
// the Doppler phase -2 pi (v_rx - v_tx) t / lambda of a LOS path between two ends that move along the LOS direction.
// Fixtures: qd_test_layout.hpp.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_layout.hpp"
#include "qd_test_layout.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <complex>
#include <string>
#include <vector>

namespace
{
// BS1 with sectors 0 and 1 at (0, 0, 25), BS2 at (400, 30, 20), UMa on the Tx side with 4 NLOS clusters of 4 rays (small
// random sets keep the suite fast); four outdoor UTs; 3.5 GHz; seed 4711
quadriga_lib::layout small()
{
    quadriga_lib::layout L;
    L.tx = qd_layout_points({"BS1", "BS1", "BS2"}, {0, 1, 0}, arma::mat{{0.0, 0.0, 400.0}, {0.0, 0.0, 30.0}, {25.0, 25.0, 20.0}},
                            arma::vec{0.0, 2.0943951, 3.14159265});
    L.tx.scenario = {"UMa", "UMa", "UMa"};
    L.tx.scenpar_override["NumClusters"] = {5.0, 5.0, 5.0};
    L.tx.scenpar_override["NumSubPaths"] = {4.0, 4.0, 4.0};
    L.tx.scenpar_override["NumClusters_min"] = {0.0, 0.0, 0.0};
    L.rx = qd_layout_points({"UT1", "UT2", "UT3", "UT4"}, {0, 0, 0, 0},
                            arma::mat{{120.0, -90.0, 260.0, 350.0}, {40.0, 150.0, -70.0, 160.0}, {1.5, 1.5, 1.5, 1.5}});
    L.rx.indoor = {0, 0, 0, 0};
    L.center_freq = {3.5e9};
    L.seed = 4711;
    return L;
}

// First difference of two channel lists ("" = equal): names, coefficients and delays of all snapshots, within 'tol' times
// the largest magnitude of each array (the paths may differ in the batches of the GBSM functions and in FMA)
std::string diff(const std::vector<quadriga_lib::channel<double>> &a, const std::vector<quadriga_lib::channel<double>> &b, double tol = 1e-12)
{
    if (a.size() != b.size())
        return "size " + std::to_string(a.size()) + " " + std::to_string(b.size());
    for (size_t c = 0; c < a.size(); ++c)
    {
        if (a[c].name != b[c].name)
            return "name " + a[c].name + " " + b[c].name;
        if (a[c].coeff_re.size() != b[c].coeff_re.size())
            return "n_snap of " + a[c].name;
        for (size_t s = 0; s < a[c].coeff_re.size(); ++s)
            if (!qd_layout_same_data(a[c].coeff_re[s], b[c].coeff_re[s], tol) || !qd_layout_same_data(a[c].coeff_im[s], b[c].coeff_im[s], tol) ||
                !qd_layout_same_data(a[c].delay[s], b[c].delay[s], tol))
                return "coefficients of " + a[c].name;
    }
    return "";
}

// The manual path on a copy of the layout
std::vector<quadriga_lib::channel<double>> manual(quadriga_lib::layout L, const arma::vec &times = {}, const std::string &method = "38.901",
                                                  arma::umat *index = nullptr)
{
    if (L.links.empty())
        quadriga_lib::init_link_states(L, method);
    return quadriga_lib::get_gbsm_channels(L, quadriga_lib::generate_rays(L), {}, times, "taps", index);
}

const std::string stale = "The layout has changed since 'init_link_states': call it again, or 'links.reset()' to discard the links.";
} // namespace

TEST_CASE("get_channels_layout - input validation")
{
    quadriga_lib::layout L = small();
    auto thr = [&](const std::string &output, const std::string &method, const std::string &mobility, double overlap, arma::sword seed,
                   const std::string &msg)
    { CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_layout(L, {}, output, method, nullptr, mobility, overlap, seed); }) == msg); };
    CHECK_NOTHROW(quadriga_lib::get_channels_layout(L));
    thr("tap", "", "auto", 0.5, -1, "Input 'output' must be 'taps' or 'rays'.");
    thr("taps", "", "fast", 0.5, -1, "Input 'mobility' must be 'phase', 'drift' or 'auto'.");
    thr("taps", "", "auto", -0.1, -1, "Input 'overlap' must be in [0, 1].");
    thr("taps", "38.9", "auto", 0.5, -1, "Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.");
    L.center_freq.reset();
    thr("taps", "", "auto", 0.5, -1, "Member 'center_freq' of 'layout' must not be empty; set the simulation frequencies.");
    L = small();
    quadriga_lib::init_link_states(L);
    thr("taps", "quadriga", "auto", 0.5, -1, "Input 'method' is 'quadriga', but the links of the layout were drawn for '38.901'.");
    thr("taps", "", "auto", 0.5, 5, "Input 'seed' must be -1 when the links of the layout are filled.");
    CHECK_NOTHROW(quadriga_lib::get_channels_layout(L, {}, "taps", "38.901"));
}

TEST_CASE("get_channels_layout - empty links: the result of the manual path, the layout unchanged")
{
    quadriga_lib::layout L = small();
    L.pairing = arma::umat{{0, 1, 1}, {0, 2, 3}};
    const quadriga_lib::layout keep = L;
    const arma::vec times = {0.0, 0.25};
    for (const char *method : {"38.901", "quadriga"})
    {
        arma::umat idx, idx_m;
        const auto ch = quadriga_lib::get_channels_layout(L, times, "taps", method, &idx);
        CHECK(diff(ch, manual(L, times, method, &idx_m)) == "");
        CHECK(qd_layout_same_int(idx, idx_m));
        CHECK(ch.size() == 4); // pairing selects 2 + 1 + 1 channel links
        CHECK(L.links.empty());
        CHECK(L.seed == 4711);
        CHECK(qd_layout_diff(L, keep) == "");
    }
    // A seed argument: the same as a layout with that seed
    quadriga_lib::layout M = L;
    M.seed = 17;
    CHECK(diff(quadriga_lib::get_channels_layout(L, {}, "taps", "", nullptr, "auto", 0.5, 17), manual(M)) == "");
    CHECK(diff(quadriga_lib::get_channels_layout(L, {}, "taps", "", nullptr, "auto", 0.5, 17), manual(L)) != "");
}

TEST_CASE("get_channels_layout - an empty Tx or Rx side gives empty results")
{
    for (int side = 0; side < 2; ++side)
    {
        quadriga_lib::layout L = small();
        if (side == 0)
            L.tx = quadriga_lib::layout_nodes(), L.rx.scenario = {"UMa", "UMa", "UMa", "UMa"}; // default sides: no nodes
        else
            L.rx = quadriga_lib::layout_nodes();
        CHECK(L.is_valid().empty());
        arma::umat idx;
        CHECK(quadriga_lib::get_channels_layout(L, {}, "taps", "", &idx).empty());
        CHECK(idx.n_rows == 3);
        CHECK(idx.n_cols == 0);
        quadriga_lib::init_link_states(L);
        CHECK(L.links.n_link() == 0);
        const auto rays = quadriga_lib::generate_rays(L);
        CHECK(rays.empty());
        CHECK(quadriga_lib::get_gbsm_channels(L, rays).empty());
        CHECK(quadriga_lib::get_channels_layout(L).empty());
    }
}

TEST_CASE("get_channels_layout - filled and edited links are used")
{
    // An edited LOS state gives the conf and the hashed seed of the new state: the same rays and channel as a layout
    // whose UT has that state fixed by the exact conf name
    quadriga_lib::layout L = small();
    quadriga_lib::init_link_states(L);
    const auto before = quadriga_lib::get_channels_layout(L);
    const arma::uword los = L.links.los[0];
    L.links.los[0] = 1 - los;
    const auto after = quadriga_lib::get_channels_layout(L);
    CHECK(diff(after, manual(L)) == "");
    CHECK(diff(after, before) != "");
    quadriga_lib::layout M = small();
    M.rx.scenario = {los ? "UMa_NLOS" : "UMa_LOS", "", "", ""};
    quadriga_lib::init_link_states(M);
    const auto fixed = quadriga_lib::get_channels_layout(M);
    REQUIRE(fixed.size() == after.size());
    for (size_t c = 0; c < 2; ++c) // BS1 sectors to UT1 (link 0)
        CHECK(diff({after[c]}, {fixed[c]}) == "");
    // An edited seed is used while the conf is unchanged
    L.links.los[0] = los;
    L.links.gbsm_seed[0] += 1;
    const auto seeded = quadriga_lib::get_channels_layout(L);
    CHECK(diff({seeded[0]}, {before[0]}) != "");
    CHECK(diff({seeded[2]}, {before[2]}) == ""); // BS2 to UT1: other link
    L.links.gbsm_seed[0] -= 1;
    CHECK(diff(quadriga_lib::get_channels_layout(L), before) == "");
}

TEST_CASE("get_channels_layout - stale links throw after each hashed member changes")
{
    using fn = void (*)(quadriga_lib::layout &);
    const std::vector<std::pair<const char *, fn>> hashed = {
        {"tx.pos", [](quadriga_lib::layout &L) { L.tx.pos(1, 2) += 0.5; }},
        {"rx.pos", [](quadriga_lib::layout &L) { L.rx.pos(2, 3) += 0.5; }},
        {"tx.name", [](quadriga_lib::layout &L) { L.tx.name[2] = "BS3"; }},
        {"rx.name", [](quadriga_lib::layout &L) { L.rx.name[1] = "UT9"; }},
        {"tx.sector", [](quadriga_lib::layout &L) { L.tx.sector[2] = 4; }},
        {"rx.sector", [](quadriga_lib::layout &L) { L.rx.sector[0] = 1; }},
        {"center_freq", [](quadriga_lib::layout &L) { L.center_freq = {3.6e9}; }},
        {"tx.seed", [](quadriga_lib::layout &L) { L.tx.seed = {-1, -1, 5}; }},
        {"tx.speed", [](quadriga_lib::layout &L) { L.tx.speed = {0.0, 0.0, 1.0}; }},
        {"tx.scenario", [](quadriga_lib::layout &L) { L.tx.scenario = {"UMi", "UMi", "UMa"}; }},
        {"rx.scenario", [](quadriga_lib::layout &L) { L.rx.scenario = {"", "UMa", "", ""}; }},
        {"rx.indoor", [](quadriga_lib::layout &L) { L.rx.indoor = {0, 1, 0, 0}; }},
        {"rx.floor", [](quadriga_lib::layout &L) { L.rx.floor = {2, -1, -1, -1}; }},
        {"rx.d2d_in", [](quadriga_lib::layout &L) { L.rx.d2d_in = {5.0, arma::datum::nan, arma::datum::nan, arma::datum::nan}; }},
        {"tx.clutter", [](quadriga_lib::layout &L) { L.tx.clutter = arma::mat(3, 3, arma::fill::value(arma::datum::nan)), L.tx.clutter(0, 2) = 5.0; }},
        {"tx.scenpar_override", [](quadriga_lib::layout &L) { L.tx.scenpar_override["SF_sigma"] = {arma::datum::nan, arma::datum::nan, 3.0}; }},
        {"wrap", [](quadriga_lib::layout &L) { L.wrap = arma::mat(arma::vec{2000.0, 0.0}); }}};
    quadriga_lib::layout L0 = small();
    quadriga_lib::init_link_states(L0);
    for (const auto &h : hashed)
    {
        quadriga_lib::layout L = L0;
        h.second(L);
        INFO(h.first);
        REQUIRE(L.is_valid() == "");
        CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_layout(L); }) == stale);
        L.links.reset();
        CHECK_NOTHROW(quadriga_lib::init_link_states(L));
    }
    {
        quadriga_lib::layout L = L0;
        hashed[0].second(L);
        L.links.reset(); // the links are drawn for the call
        CHECK(quadriga_lib::get_channels_layout(L).size() == 12);
    }
    // Not hashed: the layout seed (the links hold their master seed), orientations, pairing and antennas
    quadriga_lib::layout L = L0;
    L.seed = 4712;
    L.tx.orientation(2, 0) += 0.3;
    L.pairing = arma::umat(2, 1, arma::fill::zeros);
    quadriga_lib::set_antenna(L, "omni", {}, true);
    CHECK(quadriga_lib::get_channels_layout(L).size() == 2);
}

TEST_CASE("get_channels_layout - drift requests scatterers")
{
    quadriga_lib::layout L = small();
    const auto ch = quadriga_lib::get_channels_layout(L, arma::vec{0.0, 0.1}, "taps", "38.901", nullptr, "drift");
    REQUIRE(ch.size() == 12);
    CHECK(qd_layout_par<arma::fmat>(ch[0], "fbs_pos").n_cols > 0);
    quadriga_lib::init_link_states(L);
    CHECK(qd_layout_throws([&] { quadriga_lib::get_gbsm_channels(L, quadriga_lib::generate_rays(L), {}, {}, "taps", nullptr, "drift"); })
              .find("solve the scatterers first") != std::string::npos);
}

TEST_CASE("get_channels_layout - dual mobility gives the relative Doppler of both ends")
{
    // The Tx moves at 5 m/s along +y, the Rx at 3 m/s at 30 deg azimuth and 0.2 rad elevation; in the phase mobility the
    // coefficient of each path turns by 2 pi / lambda (r_tx . dTx + r_rx . dRx) per snapshot, with r_tx and r_rx the
    // departure and arrival directions of its ray (TR 38.901 eqs. 7.6-45, 7.6-46 without moving scatterers); NLOS, and
    // LOS with the wrap image (200, 0), which is nearer to the Rx than the site
    const double pi = arma::datum::pi, lambda = 299792458.0 / 3.5e9, dt = 2e-4;
    for (int los = 0; los < 2; ++los)
    {
        quadriga_lib::layout L;
        L.tx = qd_layout_points({"BS"}, {0}, qd_layout_pt(0.0, 0.0, 25.0));
        L.rx = qd_layout_points({"UT"}, {0}, qd_layout_pt(120.0, 60.0, 1.5));
        L.rx.scenario = {los ? "UMa_LOS" : "UMa_NLOS"};
        L.tx.scenpar_override["NumClusters"] = {5.0};
        L.tx.scenpar_override["NumSubPaths"] = {4.0};
        L.tx.scenpar_override["NumClusters_min"] = {0.0};
        L.tx.speed = {5.0}, L.rx.speed = {3.0};
        L.tx.direction = arma::mat(arma::vec{pi / 2.0, 0.0}), L.rx.direction = arma::mat(arma::vec{pi / 6.0, 0.2});
        if (los)
            L.wrap = arma::mat(arma::vec{200.0, 0.0});
        L.center_freq = {3.5e9};
        L.seed = 7;
        quadriga_lib::init_link_states(L);
        CHECK(L.links.image[0] == (los ? 1u : 0u));
        const auto rays = quadriga_lib::generate_rays(L);
        const quadriga_lib::gbsm_rays &y = rays[0][0];
        const auto ch = quadriga_lib::get_gbsm_channels(L, rays, {}, arma::vec{0.0, dt}, "rays", nullptr, "phase");
        REQUIRE(ch.size() == 1);
        const auto &c = ch[0];
        CHECK(std::abs(c.tx_pos(0, 0) - (los ? 200.0 : 0.0)) < 1e-12); // the virtual site
        const arma::vec dtx = c.tx_pos.col(1) - c.tx_pos.col(0), drx = c.rx_pos.col(1) - c.rx_pos.col(0);
        CHECK(std::abs(dtx(1) - 5.0 * dt) < 1e-12);
        CHECK(std::abs(arma::norm(drx) - 3.0 * dt) < 1e-12);
        arma::uword p = 0;
        double worst = 0.0;
        for (arma::uword i = 0; i < y.n_ray(); ++i)
        {
            if (!(y.power(i, 0) > 0.0 || y.path_type(i) <= 1))
                continue;
            const arma::vec rt = {std::cos(y.eod(i)) * std::cos(y.aod(i)), std::cos(y.eod(i)) * std::sin(y.aod(i)), std::sin(y.eod(i))};
            const arma::vec rr = {std::cos(y.eoa(i)) * std::cos(y.aoa(i)), std::cos(y.eoa(i)) * std::sin(y.aoa(i)), std::sin(y.eoa(i))};
            const double expect = 2.0 * pi / lambda * (arma::dot(rt, dtx) + arma::dot(rr, drx));
            const std::complex<double> a(c.coeff_re[0](0, 0, p), c.coeff_im[0](0, 0, p)), b(c.coeff_re[1](0, 0, p), c.coeff_im[1](0, 0, p));
            if (std::abs(a) > 0.0)
                worst = std::max(worst, std::abs(std::remainder(std::arg(b / a) - expect, 2.0 * pi)));
            ++p;
        }
        CHECK(p == c.coeff_re[0].n_slices);
        CHECK(p > 10);
        CHECK(worst < 1e-9);
    }
}

TEST_CASE("get_channels_layout - a restored copy of the links is used")
{
    // A copy of layout.links stays valid after another realisation (the hash does not contain the layout seed)
    quadriga_lib::layout L = small();
    quadriga_lib::init_link_states(L);
    const quadriga_lib::layout_links A = L.links;
    const auto ra = quadriga_lib::generate_rays(L);
    quadriga_lib::init_link_states(L, "38.901", {}, false, 2e9, "", 99);
    CHECK(L.seed == 99);
    CHECK(qd_layout_diff_rays(quadriga_lib::generate_rays(L, 0)[0][0], ra[0][0], 1e-12) != "");
    L.links = A;
    const auto rb = quadriga_lib::generate_rays(L);
    REQUIRE(rb.size() == ra.size());
    for (size_t l = 0; l < ra.size(); ++l)
        CHECK(qd_layout_diff_rays(rb[l][0], ra[l][0], 1e-12) == "");
}

TEST_CASE("get_channels_layout - a layout written with its seeds gives the same link states after reading")
{
    quadriga_lib::layout L = small();
    L.rx.indoor.reset(); // drawn states
    L.tx.seed = {-1, -1, 77};
    quadriga_lib::init_link_states(L);
    qd_layout_files F("get_channels_layout_csv");
    quadriga_lib::layout_csv_write(F.path("tx.csv"), F.path("rx.csv"), L);
    quadriga_lib::layout M = quadriga_lib::layout_csv_read(F.path("tx.csv"), F.path("rx.csv"));
    quadriga_lib::init_link_states(M);
    REQUIRE(M.links.n_link() == L.links.n_link());
    CHECK(arma::all(M.links.los == L.links.los));
    CHECK(arma::all(M.links.indoor == L.links.indoor));
    CHECK(arma::all(M.links.gbsm_seed == L.links.gbsm_seed));
    CHECK(arma::approx_equal(M.links.d2d_in, L.links.d2d_in, "absdiff", 1e-12));
    CHECK(M.links.layout_hash == L.links.layout_hash);
}

TEST_CASE("get_channels_layout - per-band arrays with different ports reach the channel")
{
    quadriga_lib::layout L = small();
    L.center_freq = {3.5e9, 28e9};
    quadriga_lib::set_antenna(L, "3gpp", {{"panel", {2, 1, 1, 1}}}, false, {}, 3.5e9);
    quadriga_lib::set_antenna(L, "3gpp", {{"panel", {4, 1, 1, 1}}}, false, {}, 28e9);
    arma::umat idx;
    const auto ch = quadriga_lib::get_channels_layout(L, {}, "taps", "", &idx);
    REQUIRE(ch.size() == 24);
    for (size_t c = 0; c < ch.size(); ++c)
    {
        CHECK(ch[c].n_tx() == (idx(2, c) == 0 ? 2 : 4));
        CHECK(ch[c].n_rx() == 2); // generic antenna
    }
}

TEST_CASE("get_channels_layout - parallel and serial runs agree")
{
    // 8 links in at least 4 groups (exact conf names per UT), 12 channels: above the size limits of all parallel loops
    quadriga_lib::layout L = small();
    L.rx.scenario = {"UMa_LOS", "UMa_NLOS", "UMa_LOS_O2I", "UMa_NLOS_O2I"};
    L.rx.indoor.reset();
    L.rx.speed = {0.0, 1.0, 0.0, 2.0};
    const arma::vec times = {0.0, 0.5};
    const auto par = quadriga_lib::get_channels_layout(L, times);
#ifdef _OPENMP
    const int n = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto ser = quadriga_lib::get_channels_layout(L, times);
    omp_set_num_threads(n);
    CHECK(diff(par, ser, 1e-12) == "");
#endif
    CHECK(par.size() == 12);
}
