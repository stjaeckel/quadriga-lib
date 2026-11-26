// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of get_channels_tr38901, the one call from a TR 38.901 scenario to the channels of the serving links: the call
// against the manual path (generate_layout, set_antenna, init_link_states, attach_terminals, get_channels_layout), the
// layout tuples, the snapshots, forced LOS states, the RSRP rule, the uplink, several frequencies, the seed and
// the input checks.
// Oracles: the manual path with the same seed; the distance v T a UT moves; generate_layout with the same tuples; the
// transpose of the downlink. Fixtures: qd_test_layout.hpp.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_layout.hpp"
#include "qd_test_layout.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace
{
// 2x1 single-polarized BS panel and a cross-polarized UT at 10 deg resolution, so that the suite stays fast
const quadriga_lib::arrayant<double> &bs()
{
    static const auto a = quadriga_lib::generate_antenna<double>("3gpp", 10.0, 3.5e9, {}, {2, 1, 1, 1});
    return a;
}
const quadriga_lib::arrayant<double> &ut()
{
    static const auto a = quadriga_lib::generate_antenna<double>("xpol", 10.0, 3.5e9);
    return a;
}

// First difference of two channel lists ("" = equal) within 'tol' times the largest magnitude of each array
bool same(const arma::cube &a, const arma::cube &b, double tol)
{
    if (arma::size(a) != arma::size(b))
        return false;
    const double m = a.is_empty() ? 0.0 : arma::abs(a).max();
    return arma::approx_equal(a, b, "absdiff", tol * m + 1e-300);
}
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
            if (!same(a[c].coeff_re[s], b[c].coeff_re[s], tol) || !same(a[c].coeff_im[s], b[c].coeff_im[s], tol) ||
                !same(a[c].delay[s], b[c].delay[s], tol))
                return "coefficients of " + a[c].name;
    }
    return "";
}

// The channels of the serving nodes from the manual path on a layout with links and pairing
std::vector<quadriga_lib::channel<double>> serving(const quadriga_lib::layout &L, const arma::uvec &node, const arma::vec &times = {},
                                                   const std::string &output = "taps")
{
    arma::umat idx;
    auto all = quadriga_lib::get_channels_layout(L, times, output, "", &idx);
    std::vector<quadriga_lib::channel<double>> out;
    for (arma::uword c = 0; c < idx.n_cols; ++c)
        if (idx(0, c) == node[idx(1, c)])
            out.push_back(all[c]);
    return out;
}
} // namespace

TEST_CASE("get_channels_tr38901 - the manual path with the same seed")
{
    const arma::uword n_ut = 2;
    arma::mat gain;
    quadriga_lib::layout Lo;
    arma::umat index;
    const auto ch = quadriga_lib::get_channels_tr38901(bs(), ut(), "UMa", {3.5e9}, n_ut, 7, {}, {0.0}, {}, {}, {}, {}, {}, "auto", "taps",
                                                       "large_scale", "serving", true, false, NAN, {}, &gain, &Lo, &index);

    quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", n_ut, {3.5e9}, {}, {0.0}, {}, {}, {}, {}, {}, true, 7);
    quadriga_lib::set_antenna(L, bs(), false);
    quadriga_lib::set_antenna(L, ut(), true);
    quadriga_lib::init_link_states(L, "38.901");
    arma::uvec site;
    arma::mat g;
    const arma::uvec node = quadriga_lib::attach_terminals(L, "large_scale", {}, &site, &g);
    L.pairing = arma::join_cols(site.t(), arma::regspace<arma::urowvec>(0, site.n_elem - 1));

    REQUIRE(node.n_elem == 3 * n_ut);
    CHECK(diff(ch, serving(L, node)) == ""); // Only the serving sectors are computed, each as in the full set
    CHECK(arma::approx_equal(gain, g, "absdiff", 1e-12));
    CHECK(qd_layout_diff(Lo, L) == "");
    CHECK(Lo.links.n_link() == L.links.n_link());
    REQUIRE(index.n_rows == 3);
    REQUIRE(index.n_cols == ch.size());
    for (arma::uword c = 0; c < index.n_cols; ++c)
    {
        CHECK(index(0, c) == node[c]);
        CHECK(index(1, c) == c);
        CHECK(index(2, c) == 0);
    }
}

TEST_CASE("get_channels_tr38901 - the default drop: 19 sites with wrap-around, 57 cells")
{
    arma::mat gain;
    arma::umat index;
    const auto ch = quadriga_lib::get_channels_tr38901(bs(), ut(), "UMa", {3.5e9}, 1, 31, {}, {}, {}, {}, {}, {}, {}, "auto", "taps",
                                                       "large_scale", "serving", true, false, NAN, {}, &gain, nullptr, &index);
    REQUIRE(ch.size() == 57);
    REQUIRE(gain.n_rows == 57);
    REQUIRE(gain.n_cols == 57);
    for (arma::uword c = 0; c < 57; ++c)
    {
        CHECK(index(0, c) == gain.row(c).index_max());
        CHECK(index(1, c) == c);
    }
}

TEST_CASE("get_channels_tr38901 - the layout tuples reach generate_layout")
{
    // ISD 200 m, one site with one sector, BS at 30 m, all UTs outdoors at 1.8 m
    arma::mat gain;
    quadriga_lib::layout Lo;
    const auto ch = quadriga_lib::get_channels_tr38901(bs(), ut(), "UMa", {3.5e9}, 3, 5, {200.0}, {0.0, NAN, 1.0}, {}, {30.0}, {1.8}, {0.0},
                                                       {}, "auto", "taps", "large_scale", "serving", true, false, NAN, {}, &gain, &Lo);
    const quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 3, {3.5e9}, {200.0}, {0.0, NAN, 1.0}, {}, {30.0}, {1.8}, {0.0},
                                                                 {}, true, 5);
    REQUIRE(ch.size() == 3);
    CHECK(gain.n_cols == 1);
    CHECK(arma::approx_equal(Lo.tx.pos, L.tx.pos, "absdiff", 0.0));
    CHECK(arma::approx_equal(Lo.rx.pos, L.rx.pos, "absdiff", 0.0));
    CHECK(arma::approx_equal(Lo.wrap, L.wrap, "absdiff", 0.0));
    CHECK(Lo.tx.pos(2, 0) == 30.0);
    CHECK(arma::all(Lo.rx.pos.row(2) == 1.8));
    CHECK(arma::all(Lo.rx.indoor == 0));
    CHECK(std::abs(arma::norm(Lo.wrap.col(0)) - 200.0) < 1e-9 * 200.0); // one site repeated at the ISD of 200 m
}

TEST_CASE("get_channels_tr38901 - links: the serving sector, the sectors of its site, all sectors")
{
    // 7 sites of 2 sectors (one ring), one UT per cell: 14 UTs
    auto call = [](const std::string &links, quadriga_lib::layout *Lo, arma::umat *idx)
    {
        return quadriga_lib::get_channels_tr38901(bs(), ut(), "UMa", {3.5e9}, 1, 19, {}, {1.0, NAN, 2.0}, {}, {}, {}, {}, {}, "auto", "taps",
                                                  "large_scale", links, true, false, NAN, {}, nullptr, Lo, idx);
    };
    quadriga_lib::layout L1, L2, L3;
    arma::umat i1, i2, i3;
    const auto serv = call("serving", &L1, &i1), site = call("site", &L2, &i2), all = call("all", &L3, &i3);
    REQUIRE(serv.size() == 14);
    REQUIRE(site.size() == 28);
    REQUIRE(all.size() == 196);

    // "all": the channels of get_channels_layout of the returned layout (empty pairing), BS node fastest, then UT
    CHECK(L3.pairing.is_empty());
    arma::umat ia;
    CHECK(diff(all, quadriga_lib::get_channels_layout(L3, {}, "taps", "", &ia)) == "");
    CHECK(arma::all(arma::vectorise(i3 == ia)));
    for (arma::uword u = 0; u < 14; ++u) // The serving channels are among them
    {
        const arma::uword c = u * 14 + i1(0, u);
        CHECK(i3(0, c) == i1(0, u));
        CHECK(i3(1, c) == u);
        CHECK(diff({serv[u]}, {all[c]}) == "");
    }

    // "site": the 2 sectors of the serving site of each UT, the serving one among them
    for (arma::uword c = 0; c < 28; ++c)
        CHECK(L2.tx.name[i2(0, c)] == L1.tx.name[i1(0, i2(1, c))]);
    for (arma::uword u = 0; u < 14; ++u)
        CHECK(arma::any(i2.row(0).cols(2 * u, 2 * u + 1) == i1(0, u)));
}

TEST_CASE("get_channels_tr38901 - snapshots of the time grid, UTs moving at the given speed")
{
    const auto ch = quadriga_lib::get_channels_tr38901(bs(), ut(), "UMi", {3.5e9}, 1, 11, {}, {0.0}, {}, {}, {NAN, NAN, 10.0}, {},
                                                       {0.0105, 1e-3});
    REQUIRE(ch.size() == 3);
    for (const auto &c : ch)
    {
        REQUIRE(c.n_snap() == 11);
        CHECK(std::abs(arma::norm(c.rx_pos.col(10) - c.rx_pos.col(0)) - 0.1) < 1e-9); // 10 m/s for 10 ms
        CHECK(arma::norm(c.tx_pos.col(10) - c.tx_pos.col(0)) == 0.0);
    }
    // One entry: an interval of 1 ms; the preset speed of indoor UTs is 3 km/h, of those in cars 40 km/h; none is still
    const auto ch2 = quadriga_lib::get_channels_tr38901(bs(), ut(), "UMi", {3.5e9}, 1, 11, {}, {0.0}, {}, {}, {}, {}, {0.01});
    for (const auto &c : ch2)
    {
        REQUIRE(c.n_snap() == 11);
        CHECK(arma::norm(c.rx_pos.col(10) - c.rx_pos.col(0)) > 0.008 - 1e-9);
    }
    // A ratio that rounds below an integer keeps its last snapshot: 0.3 / 0.1 = 2.9999999999999996
    CHECK(quadriga_lib::get_channels_tr38901(bs(), ut(), "UMi", {3.5e9}, 1, 11, {}, {0.0}, {}, {}, {}, {}, {0.3, 0.1})[0].n_snap() == 4);
}

TEST_CASE("get_channels_tr38901 - forced LOS and NLOS states")
{
    for (const std::string los : {"los", "nlos"})
    {
        // Outdoor UTs: a LOS link has a path of type 0 (LOS) with power, an NLOS link none
        quadriga_lib::layout Lo;
        const auto ch = quadriga_lib::get_channels_tr38901(bs(), ut(), "UMi", {3.5e9}, 3, 21, {}, {0.0}, {}, {}, {}, {0.0}, {}, los, "rays",
                                                           "large_scale", "serving", true, false, NAN, {}, nullptr, &Lo);
        REQUIRE(Lo.links.n_link() > 0);
        CHECK(arma::all(Lo.links.los == (los == "los" ? 1u : 0u)));
        REQUIRE(ch.size() == 9);
        for (const auto &c : ch)
        {
            const arma::u32_vec type = qd_layout_par<arma::u32_vec>(c, "path_type");
            double p_los = 0.0;
            for (arma::uword p = 0; p < type.n_elem; ++p)
                if (type[p] == 0)
                    p_los += c.path_gain[0][p];
            CHECK((p_los > 0.0) == (los == "los"));
        }
    }
}

TEST_CASE("get_channels_tr38901 - the RSRP rule as attach_terminals with the rays of all links")
{
    arma::mat gain;
    quadriga_lib::layout Lo;
    const auto ch = quadriga_lib::get_channels_tr38901(bs(), ut(), "UMa", {3.5e9}, 2, 3, {}, {0.0}, {}, {}, {}, {}, {}, "auto", "taps", "rsrp", "serving",
                                                       true, false, NAN, {}, &gain, &Lo);
    arma::mat g;
    const arma::uvec node = quadriga_lib::attach_terminals(Lo, "rsrp", quadriga_lib::generate_rays(Lo), nullptr, &g);
    CHECK(arma::approx_equal(gain, g, "absdiff", 1e-12));
    CHECK(diff(ch, serving(Lo, node)) == "");
}

TEST_CASE("get_channels_tr38901 - link_gain false, the output rays and the uplink")
{
    auto call = [](bool link_gain, bool uplink)
    {
        return quadriga_lib::get_channels_tr38901(bs(), ut(), "UMa", {3.5e9}, 1, 9, {}, {0.0}, {}, {}, {}, {}, {0.002, 1e-3}, "auto", "rays",
                                                  "large_scale", "serving", link_gain, uplink);
    };
    const auto dn = call(true, false), off = call(false, false), up = call(true, true);
    REQUIRE(dn.size() == 3);
    REQUIRE(off.size() == 3);
    REQUIRE(up.size() == 3);
    for (size_t c = 0; c < dn.size(); ++c)
    {
        const arma::vec g = qd_layout_par<arma::vec>(dn[c], "link_gain_dB");
        CHECK(arma::all(qd_layout_par<arma::vec>(off[c], "link_gain_dB") == 0.0));
        CHECK(arma::approx_equal(up[c].tx_pos, dn[c].rx_pos, "absdiff", 0.0));
        CHECK(arma::approx_equal(up[c].rx_pos, dn[c].tx_pos, "absdiff", 0.0));
        CHECK(arma::approx_equal(up[c].tx_orientation, dn[c].rx_orientation, "absdiff", 0.0));
        CHECK(arma::approx_equal(up[c].rx_orientation, dn[c].tx_orientation, "absdiff", 0.0));
        const size_t u = dn[c].name.find('_');
        CHECK(up[c].name == dn[c].name.substr(u + 1) + "_" + dn[c].name.substr(0, u));
        CHECK(arma::all(qd_layout_par<arma::vec>(up[c], "limit") == arma::flipud(qd_layout_par<arma::vec>(dn[c], "limit"))));
        REQUIRE(up[c].n_snap() == 3);
        for (size_t s = 0; s < 3; ++s)
        {
            CHECK(same(off[c].coeff_re[s] * std::pow(10.0, g[s] / 20.0), dn[c].coeff_re[s], 1e-12));
            REQUIRE(up[c].coeff_re[s].n_slices == dn[c].coeff_re[s].n_slices);
            for (arma::uword p = 0; p < dn[c].coeff_re[s].n_slices; ++p)
            {
                CHECK(arma::approx_equal(up[c].coeff_re[s].slice(p), dn[c].coeff_re[s].slice(p).t(), "absdiff", 0.0));
                CHECK(arma::approx_equal(up[c].coeff_im[s].slice(p), dn[c].coeff_im[s].slice(p).t(), "absdiff", 0.0));
                CHECK(arma::approx_equal(up[c].delay[s].slice(p), dn[c].delay[s].slice(p).t(), "absdiff", 0.0));
            }
            const arma::mat &A = dn[c].path_angles[s], &B = up[c].path_angles[s];
            CHECK(arma::approx_equal(B.cols(0, 1), A.cols(2, 3), "absdiff", 0.0));
            CHECK(arma::approx_equal(B.cols(2, 3), A.cols(0, 1), "absdiff", 0.0));
            // The Jones matrix of the reverse direction is the transpose: (H rx, V tx) and (V rx, H tx) exchange
            const arma::mat &P = dn[c].polmat[s], &Q = up[c].polmat[s];
            CHECK(arma::approx_equal(Q.rows(0, 1), P.rows(0, 1), "absdiff", 0.0));
            CHECK(arma::approx_equal(Q.rows(2, 3), P.rows(4, 5), "absdiff", 0.0));
            CHECK(arma::approx_equal(Q.rows(4, 5), P.rows(2, 3), "absdiff", 0.0));
            CHECK(arma::approx_equal(Q.rows(6, 7), P.rows(6, 7), "absdiff", 0.0));
        }
    }
}

TEST_CASE("get_channels_tr38901 - several frequencies use the method 38.901_MF")
{
    quadriga_lib::layout Lo;
    arma::umat index;
    const auto ch = quadriga_lib::get_channels_tr38901(bs(), ut(), "UMa", {3.5e9, 28.0e9}, 1, 13, {}, {0.0}, {}, {}, {}, {}, {}, "auto",
                                                       "taps", "large_scale", "serving", true, false, NAN, {}, nullptr, &Lo, &index);
    CHECK(Lo.links.method == "38.901_MF");
    REQUIRE(ch.size() == 6);
    for (arma::uword c = 0; c < 6; ++c)
    {
        CHECK(index(1, c) == c % 3);
        CHECK(index(2, c) == c / 3);
        CHECK(ch[c].center_freq[0] == (c < 3 ? 3.5e9 : 28.0e9));
    }
}

TEST_CASE("get_channels_tr38901 - seeds repeat a call; indoor presets and preset names")
{
    auto call = [](arma::sword seed, quadriga_lib::layout *Lo)
    {
        return quadriga_lib::get_channels_tr38901(bs(), ut(), "InH_Open", {3.5e9}, 1, seed, {}, {}, {}, {}, {}, {}, {}, "auto", "taps",
                                                  "large_scale", "serving", true, false, NAN, {}, nullptr, Lo);
    };
    quadriga_lib::layout L1, L2;
    const auto a = call(17, nullptr), b = call(17, nullptr);
    CHECK(a.size() == 12);
    CHECK(diff(a, b, 0.0) == "");
    const auto c = call(-1, &L1), d = call(L1.seed, &L2); // The seed in layout_out repeats a call with seed -1
    CHECK(diff(c, d, 0.0) == "");
    CHECK(qd_layout_diff(L1, L2) == "");
    CHECK(quadriga_lib::get_channels_tr38901(bs(), ut(), "38.901_RMa", {3.5e9}, 1, 2, {}, {0.0}).size() == 3);
}

TEST_CASE("get_channels_tr38901 - input validation")
{
    auto thr = [](const std::string &scenario, arma::uword n_ut, const arma::vec &grid, const arma::vec &indoor, const arma::vec &time_grid,
                  const std::string &los, const std::string &output, const std::string &attach, double o2i)
    {
        return qd_layout_throws([&]
                                { quadriga_lib::get_channels_tr38901(bs(), ut(), scenario, {3.5e9}, n_ut, 1, {}, grid, {}, {}, {}, indoor,
                                                                     time_grid, los, output, attach, "serving", true, false, o2i); });
    };
    const arma::vec g0 = {0.0};
    CHECK(thr("UMa", 1, g0, {}, {}, "auto", "taps", "large_scale", NAN) == "");
    CHECK(thr("UMa", 0, g0, {}, {}, "auto", "taps", "large_scale", NAN) == "Input 'n_ut' must be at least 1.");
    CHECK(thr("UMa", 1, g0, {}, {0.1, 1e-3, 1.0}, "auto", "taps", "large_scale", NAN) ==
          "Input 'time_grid' must have at most 2 entries: [observation_time, update_interval].");
    const std::string tg = "Input 'time_grid' must hold a finite observation time >= 0 and a finite update interval > 0.";
    CHECK(thr("UMa", 1, g0, {}, {-1.0}, "auto", "taps", "large_scale", NAN) == tg);
    CHECK(thr("UMa", 1, g0, {}, {0.1, 0.0}, "auto", "taps", "large_scale", NAN) == tg);
    CHECK(thr("UMa", 1, g0, {}, {NAN}, "auto", "taps", "large_scale", NAN) == tg);
    CHECK(thr("UMa", 1, g0, {}, {1e9, 1e-6}, "auto", "taps", "large_scale", NAN) ==
          "Input 'time_grid' must give observation_time / update_interval below 2^32 - 1.");
    CHECK(thr("UMa", 1, g0, {}, {}, "LOS", "taps", "large_scale", NAN) == "Input 'los' must be 'auto', 'los' or 'nlos'.");
    CHECK(thr("UMa", 1, g0, {}, {}, "auto", "tap", "large_scale", NAN) == "Input 'output' must be 'taps' or 'rays'.");
    CHECK(thr("UMa", 1, g0, {}, {}, "auto", "taps", "best", NAN) == "Input 'attach' must be 'large_scale' or 'rsrp'.");
    CHECK(qd_layout_throws([] { quadriga_lib::get_channels_tr38901(bs(), ut(), "UMa", {3.5e9}, 1, 1, {}, {0.0}, {}, {}, {}, {}, {}, "auto", "taps",
                                                                   "large_scale", "sector"); }) == "Input 'links' must be 'serving', 'site' or 'all'.");
    CHECK(thr("UMa", 1, g0, {}, {}, "auto", "taps", "large_scale", -0.5) == "Input 'o2i_low' must be in [0, 1], or NaN.");
    const std::string fam = "Input 'scenario' must be 'UMa', 'UMi', 'RMa', 'SMa', 'InH_Open', 'InH_Mixed', 'InF_SL', 'InF_DL', 'InF_SH', "
                            "'InF_DH' or 'InF_HH', also with the prefix '38.901_'.";
    CHECK(thr("M.2412_DenseUrban", 1, {}, {}, {}, "auto", "taps", "large_scale", NAN) == fam);
    CHECK(thr("UMx", 1, g0, {}, {}, "auto", "taps", "large_scale", NAN) == fam);
    CHECK(thr("InF_SL", 1, {}, {}, {}, "auto", "taps", "large_scale", 0.5) == "Input 'o2i_low' must be NaN for the indoor preset '38.901_InF_SL'.");
    CHECK(thr("InF_HH", 1, {}, {}, {}, "nlos", "taps", "large_scale", NAN) == "Input 'los' cannot be 'nlos' for 'InF_HH', which is always LOS.");
    // The tuples are checked by generate_layout
    CHECK(thr("InH_Open", 1, {}, {0.5}, {}, "auto", "taps", "large_scale", NAN).find("'indoor'") != std::string::npos);
    CHECK(qd_layout_throws([] { quadriga_lib::get_channels_tr38901(bs(), ut(), "UMa", arma::vec()); }) == "Input 'center_freq' must not be empty.");
    CHECK(qd_layout_throws([] { quadriga_lib::get_channels_tr38901(bs(), quadriga_lib::arrayant<double>(), "UMa"); }).rfind("Input 'ut_array': ", 0) == 0);
}
