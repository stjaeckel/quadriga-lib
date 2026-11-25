// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of the channel pipeline of a layout, stage by stage: init_link_states (link states), generate_rays (rays) and
// get_channels_from_rays (channels), on small hand-built layouts of point drops and tracks.
// Oracles, all computed in this file from the documented rules and the public functions they name:
//  - seeds: the 64-bit FNV-1a hash of the base seed and the stream key (qd_layout_oracle_seed of the fixture)
//  - LOS state: sos_rand (Exp256) seeded by "los|<site>|<image>|<family>" at the first (x, y) of the terminal, at the
//    LOS_state_lambda of the NLOS conf file, below los_probability at d2d_out = max(d2D - d2d_in, 0)
//  - O2I state: the fields "io", "o2i_model", "d2din1", "d2din2", "o2i_dev" (sos_rand at 50, 50, 25, 25 m, sos_randn at
//    10 m) and the building shares of TR 38.901 Table 7.2-1 (0.8 in UMa, UMi), Table 7.2-3 (0.5 in RMa, the others in cars)
//    and Table 7.8-1A (0.8 in SMa)
//  - edited link states: the same terminal with the state fixed by the layout columns
//  - wrap image: brute force over the documented image order, strict minimum of the 2D distance
//  - rays: the GBSM chain of the section "Geometry-based stochastic model" called directly with the hashed seeds
//    (gbsm_init_phases, gbsm_draw, path_loss, lsp_generate, ssf_clusters, ssf_rays, ssf_scatterers)
//  - link gain: -PL - O2I + 10 lg(SF) from path_loss, o2i_loss and lsp_generate at the snapshot positions, with the
//    random hE (column 10 of the "lsf" set) held at its value at the start of the track
// Fixtures: qd_test_layout.hpp (point drops, seed hash, wrap images, comparison of rays, par_data access).

#include <catch2/catch_test_macros.hpp>

#include "quadriga_layout.hpp"
#include "quadriga_sos.hpp"
#include "qd_test_layout.hpp"

#include <cmath>
#include <complex>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
using spmap = std::unordered_map<std::string, double>;
using plmap = std::unordered_map<std::string, std::vector<double>>;
const std::string v19 = "3GPP_38.901_v19_";
const char *const qd_methods_all[3] = {"38.901", "38.901_MF", "quadriga"};

struct conf_t
{
    spmap sp;
    std::string pl;
    plmap pp;
};

// Overrides of the Tx sites in all layouts of this file: 4 NLOS clusters of 4 rays keep the random sets small (the
// "ray" set has 14·N·M + N generators), so that the suite runs fast; the oracles apply the same overrides
const spmap small_sets = {{"NumClusters", 5.0}, {"NumSubPaths", 4.0}, {"NumClusters_min", 0.0}};

// A shipped conf file by its name without the prefix, with the overrides
conf_t conf(const std::string &name)
{
    conf_t c;
    c.sp = quadriga_lib::conf_file_read("data/config/" + v19 + name + ".conf", &c.pl, &c.pp);
    for (const auto &kv : small_sets)
        c.sp[kv.first] = kv.second;
    return c;
}

arma::mat col3(double x, double y, double z) { return qd_layout_pt(x, y, z); }

// Two sites: BS1 with sectors 0 and 1 at (0, 0, 25), BS2 at (400, 30, 20); scenario 'scen' on the Tx side; the small
// random sets; terminals as point drops at the columns of 'ut' [3, n], named UT1, UT2, ...; one frequency 3.5 GHz; seed 4711
quadriga_lib::layout two_sites(const arma::mat &ut, const std::string &scen = "UMa")
{
    quadriga_lib::layout L;
    L.tx = qd_layout_points({"BS1", "BS1", "BS2"}, {0, 1, 0}, arma::mat{{0.0, 0.0, 400.0}, {0.0, 0.0, 30.0}, {25.0, 25.0, 20.0}},
                            arma::vec{0.0, 2.0943951, 3.14159265});
    L.tx.scenario = {scen, scen, scen};
    for (const auto &kv : small_sets)
        L.tx.scenpar_override[kv.first] = arma::vec(3, arma::fill::value(kv.second));
    std::vector<std::string> names;
    std::vector<unsigned> sec;
    for (arma::uword i = 0; i < ut.n_cols; ++i)
        names.push_back("UT" + std::to_string(i + 1)), sec.push_back(0);
    L.rx = qd_layout_points(names, sec, ut);
    L.center_freq = {3.5e9};
    L.seed = 4711;
    return L;
}

// Four outdoor UTs
arma::mat four_uts() { return arma::mat{{120.0, -90.0, 260.0, 350.0}, {40.0, 150.0, -70.0, 160.0}, {1.5, 1.5, 1.5, 1.5}}; }

// Link of site s and terminal t, or n_link if there is none
arma::uword link_of(const quadriga_lib::layout_links &K, arma::uword s, arma::uword t)
{
    for (arma::uword l = 0; l < K.n_link(); ++l)
        if (K.i_tx[l] == s && K.i_rx[l] == t)
            return l;
    return K.n_link();
}

// One value of a uniform (or normal) SOS field of one generator at (x, y)
double field(double dist, double x, double y, long long seed, bool normal = false)
{
    arma::mat xy = {{x, y}}, val;
    if (normal)
        quadriga_lib::sos_randn<double>(arma::vec{dist}, xy, val, arma::mat(), arma::vec(), "Exp256", seed);
    else
        quadriga_lib::sos_rand<double>(arma::vec{dist}, xy, val, arma::mat(), arma::vec(), "Exp256", seed);
    return val(0, 0);
}

// Wrap image of site position (sx, sy) for a point: strict minimum over the documented order
arma::uword nearest_image(const arma::mat &images, double sx, double sy, double x, double y)
{
    arma::uword best = 0;
    double d = arma::datum::inf;
    for (arma::uword k = 0; k < images.n_cols; ++k)
    {
        const double e = std::hypot(x - sx - images(0, k), y - sy - images(1, k));
        if (e < d)
            d = e, best = k;
    }
    return best;
}

// The GBSM chain for the links at 'rx' [3, n] of one site at 'tx' [3, 1], one conf and one seed; 'rx_all' are the
// positions of all links of the group (bias removal of "quadriga"); rays [n][n_freq] (TR methods) or [n][1]
std::vector<std::vector<quadriga_lib::gbsm_rays>> chain(const std::string &method, const conf_t &c, long long seed, const arma::mat &tx,
                                                        const arma::mat &rx, const arma::u32_vec &fl, const arma::vec &d2d,
                                                        const std::vector<bool> &o2i, const arma::vec &fc, const arma::mat &rx_all,
                                                        bool scat = false, double anchor = 2.0e9, bool polvar = false, bool dual = false)
{
    const bool qd = method == "quadriga", mf = method == "38.901_MF";
    const std::string preset = qd ? "Comb256" : "Exp256", angle = qd ? "rotation" : "additive", pol = qd ? "quadriga" : "38.901";
    const spmap &sp = c.sp;
    const arma::uword n = rx.n_cols, F = fc.n_elem, N = (arma::uword)sp.at("NumClusters") - 1;
    arma::cube ph = quadriga_lib::gbsm_init_phases(sp, "lsf", seed, -1, preset, qd ? tx : arma::mat(), qd ? rx_all : arma::mat(), dual, qd, true);
    arma::mat rnd, sf, eod, rc, gre, gim;
    quadriga_lib::gbsm_draw(sp, "lsf", ph, tx, rx, rnd, -1, preset, fl, {}, dual);
    quadriga_lib::path_loss(c.pl, c.pp, sp, fc, tx, rx, &sf, qd ? d2d : arma::vec(), arma::vec(rnd.col(10)));
    arma::cube lsp, mu, sg;
    arma::vec toa;
    const bool gr = sp.at("GR_enabled") != 0.0;
    quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, sf, &toa, gr ? &gre : nullptr, gr ? &gim : nullptr, &eod, &mu, &sg);
    ph = quadriga_lib::gbsm_init_phases(sp, "cluster", seed, -1, preset);
    quadriga_lib::gbsm_draw(sp, "cluster", ph, tx, rx, rc, -1, preset, fl, {}, dual);
    std::vector<std::vector<quadriga_lib::gbsm_rays>> out(n);
    arma::cube D, P, AOD, EOD, AOA, EOA;
    std::vector<quadriga_lib::gbsm_rays> rays;
    if (qd)
    {
        quadriga_lib::ssf_clusters(sp, tx, rx, lsp, rc, D, P, AOD, EOD, AOA, EOA, method, {}, {}, eod, toa, gre, gim, o2i);
        arma::mat rr;
        ph = quadriga_lib::gbsm_init_phases(sp, "ray", seed, -1, preset);
        quadriga_lib::gbsm_draw(sp, "ray", ph, tx, rx, rr, -1, preset, fl, fc, dual);
        const arma::cube rnd_ray(rr.memptr(), rr.n_rows, rr.n_cols, 1);
        quadriga_lib::ssf_rays(sp, tx, rx, fc, lsp, mu, sg, D, P, AOD, EOD, AOA, EOA, rnd_ray, rays, method, 0, angle, pol, gre, gim);
        for (arma::uword i = 0; i < n; ++i)
            out[i].push_back(rays[i]);
    }
    else
    {
        arma::cube rf(n, N, F), rr;
        for (arma::uword k = 0; k < F; ++k)
        {
            ph = quadriga_lib::gbsm_init_phases(sp, "cluster", seed, (arma::sword)k, preset);
            arma::mat x;
            quadriga_lib::gbsm_draw(sp, "cluster", ph, tx, rx, x, (arma::sword)k, preset, fl, {}, dual);
            rf.slice(k) = x;
            ph = quadriga_lib::gbsm_init_phases(sp, "ray", seed, (arma::sword)k, preset);
            quadriga_lib::gbsm_draw(sp, "ray", ph, tx, rx, x, (arma::sword)k, preset, fl, {}, dual);
            if (k == 0)
                rr.set_size(n, x.n_cols, F);
            rr.slice(k) = x;
        }
        if (mf)
        {
            arma::cube la;
            quadriga_lib::lsp_generate(sp, arma::vec{anchor}, tx, rx, rnd, la);
            quadriga_lib::ssf_clusters(sp, tx, rx, lsp, rc, D, P, AOD, EOD, AOA, EOA, method, rf, la, eod, toa, gre, gim, o2i);
            for (arma::uword k = 0; k < F; ++k)
            {
                quadriga_lib::ssf_rays(sp, tx, rx, fc, lsp, mu, sg, D, P, AOD, EOD, AOA, EOA, rr, rays, method, k, angle, pol, gre, gim, polvar);
                for (arma::uword i = 0; i < n; ++i)
                    out[i].push_back(rays[i]);
            }
        }
        else
            for (arma::uword k = 0; k < F; ++k)
            {
                const arma::cube lk = lsp.cols(k, k), mk = mu.cols(k, k), sk = sg.cols(k, k), rfk = rf.slices(k, k), rrk = rr.slices(k, k);
                const arma::mat grk = gr ? arma::mat(gre.col(k)) : arma::mat(), gik = gr ? arma::mat(gim.col(k)) : arma::mat();
                quadriga_lib::ssf_clusters(sp, tx, rx, lk, rc, D, P, AOD, EOD, AOA, EOA, method, rfk, {}, arma::mat(eod.col(k)), toa, grk, gik, o2i);
                quadriga_lib::ssf_rays(sp, tx, rx, arma::vec{fc(k)}, lk, mk, sk, D, P, AOD, EOD, AOA, EOA, rrk, rays, method, 0, angle, pol, grk, gik, polvar);
                for (arma::uword i = 0; i < n; ++i)
                    out[i].push_back(rays[i]);
            }
    }
    if (scat || qd)
        for (auto &v : out)
            for (auto &r : v)
                quadriga_lib::ssf_scatterers(r);
    return out;
}

// Link gain -PL - O2I + 10 lg(SF) at frequency k for the snapshot positions T, R [3, n] of a link whose segment starts at
// (t0, r0), and optionally the K-factor ratio K(position) / K(start); 'o2i' [4] the inputs of o2i_loss drawn at 'o2i_at' (empty: the start; column 3 is recomputed at other positions from the field
// "o2i_dev|<family>" with base seed 'base'); 'model' empty = no O2I loss; 'hold' = false draws hE at each position; 'dual': dual mobility
arma::vec gain(const std::string &method, const conf_t &c, long long seed, const arma::mat &t0, const arma::mat &r0, const arma::mat &T,
               const arma::mat &R, double floor_id, double d2d, const arma::vec &fc, arma::uword k, const std::string &model = "",
               const arma::rowvec &o2i = {}, double dmax = 25.0, long long base = 0, const std::string &family = "", bool hold = true,
               const arma::mat &rx_all = {}, arma::vec *kf_ratio = nullptr, bool dual = false,
               const arma::mat &o2i_at = {})
{
    const bool qd = method == "quadriga";
    const std::string preset = qd ? "Comb256" : "Exp256";
    const arma::uword n = R.n_cols;
    const arma::cube ph = quadriga_lib::gbsm_init_phases(c.sp, "lsf", seed, -1, preset, qd ? t0 : arma::mat(), qd ? rx_all : arma::mat(), dual, qd, true);
    arma::mat rnd0, rnd, sf, sf0;
    const arma::u32_vec fl0(1, arma::fill::value((arma::u32)floor_id)), fl(n, arma::fill::value((arma::u32)floor_id));
    quadriga_lib::gbsm_draw(c.sp, "lsf", ph, t0, r0, rnd0, -1, preset, fl0, {}, dual);
    quadriga_lib::gbsm_draw(c.sp, "lsf", ph, T, R, rnd, -1, preset, fl, {}, dual);
    const arma::vec he = hold ? arma::vec(n, arma::fill::value(rnd0(0, 10))) : arma::vec(rnd.col(10));
    const arma::mat pl = quadriga_lib::path_loss(c.pl, c.pp, c.sp, fc, T, R, &sf, qd ? arma::vec(n, arma::fill::value(d2d)) : arma::vec(), he);
    arma::cube lsp;
    quadriga_lib::lsp_generate(c.sp, fc, T, R, rnd, lsp, sf);
    if (kf_ratio != nullptr) // K-factor at the positions over the K-factor at the start
    {
        arma::cube lsp0;
        quadriga_lib::path_loss(c.pl, c.pp, c.sp, fc, t0, r0, &sf0, qd ? arma::vec{d2d} : arma::vec(), arma::vec{rnd0(0, 10)});
        quadriga_lib::lsp_generate(c.sp, fc, t0, r0, rnd0, lsp0, sf0);
        kf_ratio->set_size(n);
        for (arma::uword i = 0; i < n; ++i)
            (*kf_ratio)(i) = lsp(i, k, 1) / lsp0(0, k, 1);
    }
    arma::vec g(n);
    for (arma::uword i = 0; i < n; ++i)
        g(i) = -pl(i, k) + 10.0 * std::log10(lsp(i, k, 2));
    if (!model.empty())
    {
        arma::mat x(n, 4);
        for (arma::uword i = 0; i < n; ++i)
        {
            x.row(i) = o2i;
            const arma::mat &at = o2i_at.is_empty() ? r0 : o2i_at; // where the O2I values were drawn
            if (R(0, i) != at(0, 0) || R(1, i) != at(1, 0))
                x(i, 3) = field(10.0, R(0, i), R(1, i), qd_layout_oracle_seed(base, "o2i_dev|" + family), true);
        }
        const arma::mat loss = quadriga_lib::o2i_loss(fc, x, {model}, arma::vec{1.0}, dmax);
        g -= loss.col(k);
    }
    return g;
}
} // namespace

TEST_CASE("init_link_states - input validation")
{
    auto ok = [] { return two_sites(four_uts()); };
    auto thr = [](quadriga_lib::layout L, const std::string &method, const arma::vec &o2i_low, bool pv, double anchor,
                  const std::string &preset, arma::sword seed, const std::string &msg)
    {
        const std::string e = qd_layout_throws([&] { quadriga_lib::init_link_states(L, method, o2i_low, pv, anchor, preset, seed); });
        CHECK(e == msg);
        CHECK(L.links.empty()); // a failed call leaves the layout unchanged
        CHECK(L.seed == 4711);
    };
    {
        quadriga_lib::layout L = ok();
        CHECK_NOTHROW(quadriga_lib::init_link_states(L));
        CHECK(L.links.n_link() == 8);
    }
    SECTION("method and options")
    {
        thr(ok(), "38.902", {}, false, 2e9, "", -1, "Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.");
        thr(ok(), "38.901", arma::vec{0.5, 0.5}, false, 2e9, "", -1, "Input 'o2i_low' must be empty or one value in [0, 1].");
        thr(ok(), "38.901", arma::vec{1.5}, false, 2e9, "", -1, "Input 'o2i_low' must be empty or one value in [0, 1].");
        thr(ok(), "quadriga", {}, true, 2e9, "", -1, "Input 'pol_variability' applies to the TR methods only.");
        thr(ok(), "38.901_MF", {}, false, 0.0, "", -1, "Input 'anchor_freq' must be positive and finite.");
        thr(ok(), "38.901", {}, false, 2e9, "Comb999", -1, "Input 'sos_preset' is not a known preset ('Comb999').");
        thr(ok(), "38.901", {}, false, 2e9, "", -2, "Input 'seed' must be >= -1.");
    }
    SECTION("layout")
    {
        quadriga_lib::layout L = ok();
        L.center_freq.reset();
        thr(L, "38.901", {}, false, 2e9, "", -1, "Member 'center_freq' of 'layout' must not be empty; set the simulation frequencies.");
        L = ok();
        L.tx.pos(0, 1) = 5.0; // sectors of a site at different positions
        const std::string e = qd_layout_throws([&] { quadriga_lib::init_link_states(L); });
        CHECK(e.rfind("Input 'layout' is not valid: tx: ", 0) == 0);
        L = ok();
        L.tx.scenario.clear();
        thr(L, "38.901", {}, false, 2e9, "", -1, "No scenario for site 'BS1' and terminal 'UT1': set 'scenario' on the Tx or the Rx side.");
        L = ok();
        L.rx.scenario = {"", "no_such_scenario", "", ""};
        thr(L, "38.901", {}, false, 2e9, "", -1,
            "Terminal 'UT2': scenario 'no_such_scenario' is neither a scenario family nor a conf file in 'config_dirs' or the shipped folder.");
        L = ok();
        L.rx.indoor = {1, 1, 1, 1};
        L.rx.d2d_in = {5.0, 30.0, 5.0, 5.0};
        thr(L, "38.901", {}, false, 2e9, "", -1, "Terminal 'UT2': 'd2d_in' of 30 m is above the maximum indoor distance 25 m of the family 'UMa'.");
    }
    SECTION("segments")
    {
        // A segment that starts at a site position, and indoor distances that differ between segment starts
        quadriga_lib::layout L = ok();
        L.rx = qd_layout_points({"UT1"}, {0}, arma::mat{{10.0, 0.0}, {100.0, 0.0}, {1.5, 25.0}});
        L.rx.scenario = {"UMa", "UMa_NLOS"};
        thr(L, "38.901", {}, false, 2e9, "", -1, "Terminal 'UT1': a later segment starts at the position of site 'BS1'; a segment start needs a LOS path of non-zero length.");
        L.rx = qd_layout_points({"UT1"}, {0}, arma::mat{{10.0, 20.0, 30.0}, {100.0, 100.0, 100.0}, {1.5, 1.5, 1.5}});
        L.rx.indoor = {1, 1, 3};
        L.rx.d2d_in = {5.0, 5.0, 8.0};
        thr(L, "38.901", {}, false, 2e9, "", -1, "Terminal 'UT1': 'd2d_in' is 5 m and 8 m at the starts of its segments; a terminal has one indoor distance.");
    }
    SECTION("rules of the pipeline")
    {
        // A lattice basis at 120 degrees describes the same lattice, but +-(a1 - a2) are not its nearest images
        quadriga_lib::layout L = ok();
        L.wrap = arma::mat{{500.0, -250.0}, {0.0, 433.0}};
        CHECK(qd_layout_throws([&] { quadriga_lib::init_link_states(L); })
                  .rfind("Input 'layout' is not valid: Member 'wrap' must have columns a1, a2 with 0 <= a1.a2 <= min(|a1|^2, |a2|^2) / 2", 0) == 0);
        L.wrap = arma::mat{{500.0, 250.0}, {0.0, 433.0}}; // the reduced basis of the same lattice
        CHECK_NOTHROW(quadriga_lib::init_link_states(L));
        // The panels of a terminal move together
        L = ok();
        L.rx = qd_layout_points({"UT1", "UT1"}, {0, 1}, arma::mat{{120.0, 120.0}, {40.0, 40.0}, {1.5, 1.5}});
        L.rx.speed = {0.0, 10.0};
        thr(L, "38.901", {}, false, 2e9, "", -1,
            "Input 'layout' is not valid: rx: Nodes 'UT1' (sectors 0 and 1): must have equal 'speed', 'time' and 'direction' in each row.");
        // An exact conf name and a fixed indoor state that contradict each other
        L = ok();
        L.rx.scenario = {"UMa_LOS", "", "", ""};
        L.rx.indoor = {1, -1, -1, -1};
        thr(L, "38.901", {}, false, 2e9, "", -1, "Terminal 'UT1': the conf 'UMa_LOS' fixes an outdoor state, but 'indoor' is 1.");
        L.rx.scenario = {"UMa_NLOS_O2I", "", "", ""};
        L.rx.indoor = {0, -1, -1, -1};
        thr(L, "38.901", {}, false, 2e9, "", -1, "Terminal 'UT1': the conf 'UMa_NLOS_O2I' fixes a building state, but 'indoor' is 0.");
    }
    SECTION("links that are not valid are replaced")
    {
        quadriga_lib::layout L = ok();
        L.links.i_tx = {7}; // not valid: one member has an entry, the others none
        CHECK_NOTHROW(quadriga_lib::init_link_states(L));
        CHECK(L.links.is_valid().empty());
    }
}

TEST_CASE("init_link_states - family to conf resolution, exact names and user conf files")
{
    // Families give all four confs; an exact name on the Rx side wins over the family of the site and fixes LOS and
    // building state; a full file stem is accepted; a name that is no family and no table conf is a user conf file
    quadriga_lib::layout L = two_sites(four_uts());
    L.rx.indoor = {0, 1, 2, -1};
    L.rx.d2d_in = {arma::datum::nan, 8.0, arma::datum::nan, arma::datum::nan};
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    for (const char *n : {"UMa_LOS", "UMa_NLOS", "UMa_LOS_O2I", "UMa_NLOS_O2I"})
        CHECK(std::count(K.conf_names.begin(), K.conf_names.end(), std::string(n)) == 1);
    REQUIRE(K.conf_files.size() == K.conf_names.size());
    for (size_t i = 0; i < K.conf_names.size(); ++i)
    {
        CHECK(std::filesystem::path(K.conf_files[i]).filename().string() == v19 + K.conf_names[i] + ".conf");
        CHECK(std::filesystem::exists(K.conf_files[i]));
    }
    REQUIRE(K.n_link() == 8);
    for (arma::uword l = 0; l < 8; ++l)
    {
        const arma::uword t = K.i_rx[l];
        CHECK(K.i_tx[l] == l % 2); // terminal-major order
        CHECK(t == l / 2);
        const bool building = K.indoor[l] == 1 || K.indoor[l] == 3;
        const std::string expect = std::string("UMa_") + (K.los[l] ? "LOS" : "NLOS") + (building ? "_O2I" : "");
        CHECK(K.conf_names[K.i_conf[l]] == expect);
        if (t < 3)
            CHECK(K.indoor[l] == L.rx.indoor[t]);
        if (t != 1)
            CHECK((K.d2d_in[l] == 0.0) == !building); // car and outdoor UTs: 0
        else
            CHECK(K.d2d_in[l] == 8.0);
        CHECK(K.i_segment[l] == 0);
        CHECK(K.image[l] == 0);
    }

    L.rx.scenario = {"UMi_NLOS", v19 + "UMa_LOS_O2I", "my_scen", ""};
    L.rx.indoor.reset(), L.rx.d2d_in.reset();
    qd_layout_files F("init_link_states_conf");
    std::filesystem::copy_file("data/config/" + v19 + "UMa_LOS.conf", F.path("my_scen.conf"));
    quadriga_lib::init_link_states(L, "38.901", {}, false, 2e9, "", -1, {F.dir});
    for (arma::uword l = 0; l < 2; ++l)
    {
        CHECK(K.conf_names[K.i_conf[l]] == "UMi_NLOS");
        CHECK(K.los[l] == 0);
        CHECK(K.indoor[l] == 0);
        CHECK(K.conf_names[K.i_conf[l + 2]] == "UMa_LOS_O2I");
        CHECK(K.los[l + 2] == 1);
        CHECK(K.indoor[l + 2] == 1);
        CHECK(K.d2d_in[l + 2] > 0.0);
        CHECK(K.conf_names[K.i_conf[l + 4]] == "my_scen");
        CHECK(K.indoor[l + 4] == 0);
    }
    for (const char *n : {"UMi_LOS", "UMi_NLOS", "UMi_LOS_O2I", "UMi_NLOS_O2I", "my_scen"})
        CHECK(std::count(K.conf_names.begin(), K.conf_names.end(), std::string(n)) == 1);
    const auto it = std::find(K.conf_names.begin(), K.conf_names.end(), std::string("my_scen"));
    CHECK(K.conf_files[(size_t)(it - K.conf_names.begin())] == F.path("my_scen.conf"));

    // A conf that several families share belongs to the first of them: Indoor_LOS to InH_Open, with its NLOS conf
    L.rx.scenario = {"Indoor_LOS", "", "", ""};
    quadriga_lib::init_link_states(L);
    CHECK(K.conf_names[K.i_conf[0]] == "Indoor_LOS");
    CHECK(K.los[0] == 1);
    CHECK(std::count(K.conf_names.begin(), K.conf_names.end(), std::string("Indoor_NLOS")) == 1);
}

TEST_CASE("init_link_states - LOS and O2I states follow the documented draws")
{
    // 30 UTs at 1.5 m and 7.5 m (floor 3) on a spiral around BS1; all states drawn; the oracle evaluates the fields
    // one point at a time, so the states cannot depend on the other terminals
    arma::mat ut(3, 30);
    for (arma::uword i = 0; i < 30; ++i)
    {
        const double r = 40.0 + 14.0 * (double)i, a = 0.7 * (double)i;
        ut(0, i) = r * std::cos(a), ut(1, i) = r * std::sin(a), ut(2, i) = (i % 3 == 0) ? 7.5 : 1.5;
    }
    quadriga_lib::layout L = two_sites(ut);
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    REQUIRE(K.n_link() == 60);
    REQUIRE(K.o2i_rnd.n_rows == 30);
    const double lambda = conf("UMa_NLOS").sp.at("LOS_state_lambda");
    CHECK(lambda == 50.0);
    const char *site[2] = {"BS1", "BS2"};
    const double sx[2] = {0.0, 400.0}, sy[2] = {0.0, 30.0}, sz[2] = {25.0, 20.0};
    arma::uword n_building = 0, n_los = 0;
    for (arma::uword t = 0; t < 30; ++t)
    {
        const double x = ut(0, t), y = ut(1, t), z = ut(2, t);
        const bool building = field(50.0, x, y, qd_layout_oracle_seed(4711, "io|UMa")) < 0.8;
        const double c0 = field(50.0, x, y, qd_layout_oracle_seed(4711, "o2i_model|UMa"));
        const double c1 = field(25.0, x, y, qd_layout_oracle_seed(4711, "d2din1|UMa"));
        const double c2 = field(25.0, x, y, qd_layout_oracle_seed(4711, "d2din2|UMa"));
        const double c3 = field(10.0, x, y, qd_layout_oracle_seed(4711, "o2i_dev|UMa"), true);
        CHECK(std::abs(K.o2i_rnd(t, 0) - c0) < 1e-12);
        CHECK(std::abs(K.o2i_rnd(t, 1) - c1) < 1e-12);
        CHECK(std::abs(K.o2i_rnd(t, 2) - c2) < 1e-12);
        CHECK(std::abs(K.o2i_rnd(t, 3) - c3) < 1e-12);
        CHECK(K.o2i_model[t] == (c0 < 0.5 ? 0u : 1u)); // stored for every terminal, also outdoors (edited states)
        n_building += building;
        const int floor = std::max((int)std::floor((z - 1.5) / 3.0 + 0.5) + 1, 1);
        for (arma::uword s = 0; s < 2; ++s)
        {
            const arma::uword l = link_of(K, s, t);
            REQUIRE(l < K.n_link());
            CHECK(K.indoor[l] == (building ? 1 : 0));
            const double d2d = building ? std::min(c1, c2) * 25.0 : 0.0;
            CHECK(std::abs(K.d2d_in[l] - d2d) < 1e-9);
            CHECK(K.floor_id[l] == (building ? (arma::u32)(floor - 1) : 0u));
            const double d2D = std::hypot(x - sx[s], y - sy[s]);
            const arma::vec p = quadriga_lib::los_probability("38.901_UMa", arma::vec{std::max(d2D - d2d, 0.0)}, arma::vec{sz[s]}, arma::vec{z});
            const double u = field(lambda, x, y, qd_layout_oracle_seed(4711, std::string("los|") + site[s] + "|0|UMa"));
            CHECK(K.los[l] == (u < p(0) ? 1u : 0u));
            n_los += K.los[l];
            const std::string cname = std::string("UMa_") + (K.los[l] ? "LOS" : "NLOS") + (building ? "_O2I" : "");
            CHECK(K.conf_names[K.i_conf[l]] == cname);
            CHECK(K.gbsm_seed[l] == qd_layout_oracle_seed(4711, std::string("gbsm|") + site[s] + "|0|" + cname));
        }
    }
    CHECK(n_building > 5); // the layout has both states, so the checks above see both branches
    CHECK(n_building < 30);
    CHECK(n_los > 2);
    CHECK(n_los < 58);
    CHECK(K.seed == 4711);
    CHECK(K.method == "38.901");
    CHECK(K.sos_preset == "Exp256");

    // InF with d_clutter: the LOS state is correlated over d_clutter / 2, and the clutter reaches los_probability
    quadriga_lib::layout M = two_sites(ut * 0.2, "InF_SL");
    M.tx.pos.row(2).fill(8.0);
    M.tx.clutter = arma::mat(3, 3, arma::fill::value(arma::datum::nan));
    M.tx.clutter.row(0).fill(20.0), M.tx.clutter.row(1).fill(0.4);
    quadriga_lib::init_link_states(M);
    arma::uword n_inf = 0;
    for (arma::uword l = 0; l < M.links.n_link(); ++l)
    {
        const arma::uword s = M.links.i_tx[l], t = M.links.i_rx[l];
        const double x = M.rx.pos(0, t), y = M.rx.pos(1, t);
        const double d2D = std::hypot(x - M.tx.pos(0, 2 * s), y - M.tx.pos(1, 2 * s));
        const arma::vec p = quadriga_lib::los_probability("38.901_InF_SL", arma::vec{d2D}, arma::vec{8.0}, arma::vec{M.rx.pos(2, t)}, 20.0, 0.4);
        const double u = field(10.0, x, y, qd_layout_oracle_seed(4711, std::string("los|") + site[s] + "|0|InF_SL"));
        CHECK(M.links.los[l] == (u < p(0) ? 1u : 0u));
        n_inf += M.links.los[l];
    }
    CHECK(n_inf > 0);
    CHECK(n_inf < M.links.n_link());
}

TEST_CASE("init_link_states - wrap image equals the brute-force nearest image and enters the seeds")
{
    // Lattice a1 = (500, 0), a2 = (250, 433); UTs inside and far outside the cluster of the two sites
    quadriga_lib::layout L = two_sites(arma::mat{{120.0, 700.0, -380.0, 1300.0, 20.0, 260.0},
                                                 {40.0, 120.0, -300.0, 900.0, 400.0, -330.0},
                                                 {1.5, 1.5, 1.5, 1.5, 1.5, 1.5}});
    L.wrap = arma::mat{{500.0, 250.0}, {0.0, 433.0127}};
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    const arma::mat img = qd_layout_images(L.wrap);
    REQUIRE(img.n_cols == 7);
    const char *site[2] = {"BS1", "BS2"};
    const double sx[2] = {0.0, 400.0}, sy[2] = {0.0, 30.0}, sz[2] = {25.0, 20.0};
    const double lambda = conf("UMa_NLOS").sp.at("LOS_state_lambda");
    arma::uword n_wrapped = 0;
    for (arma::uword l = 0; l < K.n_link(); ++l)
    {
        const arma::uword s = K.i_tx[l], t = K.i_rx[l];
        const double x = L.rx.pos(0, t), y = L.rx.pos(1, t);
        const arma::uword k = nearest_image(img, sx[s], sy[s], x, y);
        CHECK(K.image[l] == k);
        n_wrapped += k != 0;
        const std::string cname = K.conf_names[K.i_conf[l]];
        CHECK(K.gbsm_seed[l] == qd_layout_oracle_seed(4711, std::string("gbsm|") + site[s] + "|" + std::to_string(k) + "|" + cname));
        // The LOS state is drawn at the distance to the virtual site, from the field of that image
        const double d2d = K.d2d_in[l];
        const double d2D = std::hypot(x - sx[s] - img(0, k), y - sy[s] - img(1, k));
        const arma::vec p = quadriga_lib::los_probability("38.901_UMa", arma::vec{std::max(d2D - d2d, 0.0)}, arma::vec{sz[s]}, arma::vec{1.5});
        const double u = field(lambda, x, y, qd_layout_oracle_seed(4711, std::string("los|") + site[s] + "|" + std::to_string(k) + "|UMa"));
        CHECK(K.los[l] == (u < p(0) ? 1u : 0u));
    }
    CHECK(n_wrapped >= 4);
    // Rays and channels refer to the virtual site position
    const auto rays = quadriga_lib::generate_rays(L);
    arma::umat idx;
    const auto ch = quadriga_lib::get_channels_from_rays(L, rays, {}, {}, "taps", &idx);
    for (arma::uword l = 0; l < K.n_link(); ++l)
    {
        const arma::uword s = K.i_tx[l], k = K.image[l];
        const arma::mat v = col3(sx[s] + img(0, k), sy[s] + img(1, k), sz[s]);
        CHECK(arma::approx_equal(rays[l][0].tx_pos, v, "absdiff", 1e-9));
    }
    for (size_t c = 0; c < ch.size(); ++c)
    {
        const arma::uword s = idx(0, c) == 2 ? 1 : 0, l = link_of(K, s, idx(1, c)), k = K.image[l];
        CHECK(std::abs(ch[c].tx_pos(0, 0) - sx[s] - img(0, k)) < 1e-9);
        CHECK(std::abs(ch[c].tx_pos(1, 0) - sy[s] - img(1, k)) < 1e-9);
    }
    // Two links of one site with equal conf and different images have different seeds
    for (arma::uword a = 0; a < K.n_link(); ++a)
        for (arma::uword b = a + 1; b < K.n_link(); ++b)
            if (K.i_tx[a] == K.i_tx[b] && K.i_conf[a] == K.i_conf[b])
                CHECK((K.image[a] == K.image[b]) == (K.gbsm_seed[a] == K.gbsm_seed[b]));
}

TEST_CASE("init_link_states - seeds, reproducibility and an added UT")
{
    quadriga_lib::layout L = two_sites(four_uts());
    quadriga_lib::init_link_states(L);
    const quadriga_lib::layout_links A = L.links;
    quadriga_lib::init_link_states(L);
    CHECK(L.links.los.n_elem == A.los.n_elem);
    CHECK(arma::all(L.links.los == A.los));
    CHECK(arma::all(L.links.gbsm_seed == A.gbsm_seed));
    CHECK(L.links.layout_hash == A.layout_hash);

    // A seed argument replaces the layout seed and is written to it
    quadriga_lib::init_link_states(L, "38.901", {}, false, 2e9, "", 99);
    CHECK(L.seed == 99);
    CHECK(L.links.seed == 99);
    const quadriga_lib::layout_links B = L.links;
    quadriga_lib::init_link_states(L);
    CHECK(arma::all(L.links.gbsm_seed == B.gbsm_seed));

    // A site seed replaces the layout seed in the streams of that site only
    L.seed = 4711;
    L.tx.seed = {-1, -1, 321};
    quadriga_lib::init_link_states(L);
    for (arma::uword l = 0; l < L.links.n_link(); ++l)
    {
        const std::string cname = L.links.conf_names[L.links.i_conf[l]];
        const long long base = L.links.i_tx[l] == 1 ? 321 : 4711;
        CHECK(L.links.gbsm_seed[l] == qd_layout_oracle_seed(base, std::string("gbsm|") + (L.links.i_tx[l] == 1 ? "BS2" : "BS1") + "|0|" + cname));
    }

    // No seed anywhere: one draw of the system random device, stored in the layout
    L.tx.seed.reset();
    L.seed = -1;
    quadriga_lib::init_link_states(L);
    CHECK(L.seed >= 0);
    CHECK(L.links.seed == L.seed);

    // Adding a UT does not change the states of the others
    L.seed = 4711;
    quadriga_lib::init_link_states(L);
    const quadriga_lib::layout_links C = L.links;
    quadriga_lib::layout M = two_sites(arma::join_rows(four_uts(), col3(-30.0, -200.0, 1.5)));
    quadriga_lib::init_link_states(M);
    REQUIRE(M.links.n_link() == C.n_link() + 2);
    for (arma::uword l = 0; l < C.n_link(); ++l)
    {
        CHECK(M.links.los[l] == C.los[l]);
        CHECK(M.links.indoor[l] == C.indoor[l]);
        CHECK(M.links.d2d_in[l] == C.d2d_in[l]);
        CHECK(M.links.gbsm_seed[l] == C.gbsm_seed[l]);
        CHECK(M.links.conf_names[M.links.i_conf[l]] == C.conf_names[C.i_conf[l]]);
    }
}

TEST_CASE("init_link_states - O2I: fixed d2d_in, car UTs, InH and InF")
{
    // A fixed d2d_in sets the distance columns of the O2I inputs
    quadriga_lib::layout L = two_sites(four_uts());
    L.rx.indoor = {1, 3, 0, 2};
    L.rx.d2d_in = {12.0, 12.0, 12.0, 12.0};
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    CHECK(std::abs(K.o2i_rnd(0, 1) - 12.0 / 25.0) < 1e-15);
    CHECK(std::abs(K.o2i_rnd(0, 2) - 12.0 / 25.0) < 1e-15);
    for (arma::uword l = 0; l < K.n_link(); ++l)
    {
        const arma::uword t = K.i_rx[l];
        CHECK(K.d2d_in[l] == (t < 2 ? 12.0 : 0.0));
        CHECK(K.indoor[l] == L.rx.indoor[t]);
        const bool b = t < 2;
        CHECK(K.conf_names[K.i_conf[l]] == std::string("UMa_") + (K.los[l] ? "LOS" : "NLOS") + (b ? "_O2I" : ""));
    }

    // RMa: the drawn non-building UTs are in cars and use the outdoor confs
    arma::mat ut(3, 24);
    for (arma::uword i = 0; i < 24; ++i)
        ut(0, i) = 300.0 * std::cos(0.9 * (double)i) + 150.0, ut(1, i) = 280.0 * std::sin(0.9 * (double)i), ut(2, i) = 1.5;
    L = two_sites(ut, "RMa");
    quadriga_lib::init_link_states(L);
    arma::uword n_car = 0;
    for (arma::uword l = 0; l < L.links.n_link(); ++l)
    {
        const arma::uword t = L.links.i_rx[l];
        const bool building = field(50.0, ut(0, t), ut(1, t), qd_layout_oracle_seed(4711, "io|RMa")) < 0.5;
        CHECK(L.links.indoor[l] == (building ? 1 : 2));
        n_car += !building;
        const std::string cname = L.links.conf_names[L.links.i_conf[l]];
        CHECK(cname == std::string("RMa_") + (L.links.los[l] ? "LOS" : "NLOS") + (building ? "_O2I" : ""));
        CHECK(L.links.d2d_in[l] <= (building ? 10.0 : 0.0));
    }
    CHECK(n_car > 4);
    CHECK(n_car < 44);

    // The share of the low-loss model: 1 = all low, 0 = all high (UMa)
    L = two_sites(four_uts());
    L.rx.indoor = {1, 1, 1, 1};
    quadriga_lib::init_link_states(L, "38.901", arma::vec{1.0});
    CHECK(arma::all(L.links.o2i_model == 0));
    quadriga_lib::init_link_states(L, "38.901", arma::vec{0.0});
    CHECK(arma::all(L.links.o2i_model == 1));

    // SMa: residential (0.9, d2d_in up to 10 m) and commercial (up to 25 m) buildings, cars
    L = two_sites(ut, "SMa");
    quadriga_lib::init_link_states(L);
    arma::uword n_res = 0;
    for (arma::uword l = 0; l < L.links.n_link(); ++l)
    {
        const arma::uword t = L.links.i_rx[l];
        const bool building = field(50.0, ut(0, t), ut(1, t), qd_layout_oracle_seed(4711, "io|SMa")) < 0.8;
        const bool res = field(50.0, ut(0, t), ut(1, t), qd_layout_oracle_seed(4711, "o2i_class|SMa")) < 0.9;
        CHECK(L.links.indoor[l] == (building ? (res ? 3 : 1) : 2));
        CHECK(L.links.d2d_in[l] <= (building ? (res ? 10.0 : 25.0) : 0.0));
        n_res += building && res;
    }
    CHECK(n_res > 2);

    // InH and InF: no O2I; InF-HH is always LOS
    for (const char *fam : {"InH_Open", "InF_HH", "InF_DH"})
    {
        L = two_sites(four_uts() * 0.1, fam);
        L.tx.pos.row(2).fill(8.0);
        L.rx.indoor = {1, 1, 1, 1};
        quadriga_lib::init_link_states(L);
        CHECK(arma::all(L.links.indoor == 0));
        CHECK(arma::all(L.links.d2d_in == 0.0));
        CHECK(arma::all(L.links.floor_id == 0));
        if (std::string(fam) == "InF_HH")
            CHECK(arma::all(L.links.los == 1));
        for (const std::string &c : L.links.conf_names)
            CHECK(c.find("O2I") == std::string::npos);
    }
}

TEST_CASE("generate_rays - one link equals a hand-wired GBSM chain for all three methods")
{
    // Four UTs fixed to UMa NLOS (one group of four links for the bias removal of quadriga), one building UT on floor 3
    // with d2d_in 7 m (own group, O2I flag, legacy distance of quadriga), two frequencies; "38.901" also with the
    // polarization power variability (PolVar_sigma = 3 dB in the UMa files)
    arma::mat ut = arma::join_rows(four_uts(), col3(-150.0, -60.0, 7.5));
    quadriga_lib::layout L = two_sites(ut);
    L.tx.pos.cols(2, 2) = col3(-60.0, 380.0, 20.0); // BS2 in NLOS conf range of all UTs
    L.rx.scenario = {"UMa_NLOS", "UMa_NLOS", "UMa_NLOS", "UMa_NLOS", "UMa_LOS_O2I"};
    L.rx.floor = {-1, -1, -1, -1, 3};
    L.rx.d2d_in = {arma::datum::nan, arma::datum::nan, arma::datum::nan, arma::datum::nan, 7.0};
    L.center_freq = {3.5e9, 6.0e9};
    const conf_t nlos = conf("UMa_NLOS"), o2i = conf("UMa_LOS_O2I");
    for (int run = 0; run < 4; ++run)
    {
        const char *method = run == 3 ? "38.901" : qd_methods_all[run];
        const bool pv = run == 3;
        quadriga_lib::init_link_states(L, method, {}, pv, 2.5e9);
        const auto rays = quadriga_lib::generate_rays(L);
        REQUIRE(rays.size() == L.links.n_link());
        const bool qd = std::string(method) == "quadriga";
        for (arma::uword s = 0; s < 2; ++s)
        {
            const arma::mat tx = s == 0 ? col3(0.0, 0.0, 25.0) : col3(-60.0, 380.0, 20.0);
            const std::string sname = s == 0 ? "BS1" : "BS2";
            const auto a = chain(method, nlos, qd_layout_oracle_seed(4711, "gbsm|" + sname + "|0|UMa_NLOS"), tx, ut.cols(0, 3),
                                 arma::u32_vec(4, arma::fill::zeros), arma::vec(4, arma::fill::zeros), std::vector<bool>(4, false),
                                 L.center_freq, ut.cols(0, 3), false, 2.5e9, pv);
            const auto b = chain(method, o2i, qd_layout_oracle_seed(4711, "gbsm|" + sname + "|0|UMa_LOS_O2I"), tx, ut.cols(4, 4),
                                 arma::u32_vec{2}, arma::vec{7.0}, std::vector<bool>{true}, L.center_freq, ut.cols(4, 4), false, 2.5e9, pv);
            for (arma::uword t = 0; t < 5; ++t)
            {
                const arma::uword l = link_of(L.links, s, t);
                REQUIRE(l < L.links.n_link());
                const auto &ref = t < 4 ? a[t] : b[0];
                REQUIRE(rays[l].size() == (qd ? 1 : 2));
                REQUIRE(ref.size() == rays[l].size());
                for (size_t k = 0; k < ref.size(); ++k)
                    CHECK(qd_layout_diff_rays(rays[l][k], ref[k], 1e-12) == "");
            }
        }
    }
}

TEST_CASE("generate_rays - subsets and batches give the values of the full call")
{
    quadriga_lib::layout L = two_sites(arma::join_rows(four_uts(), col3(-150.0, -60.0, 1.5)));
    L.rx.scenario = {"UMa_NLOS", "UMa_NLOS", "UMa_NLOS", "UMa_NLOS", "UMa_NLOS"};
    for (const char *method : {"38.901", "quadriga"})
    {
        quadriga_lib::init_link_states(L, method);
        const auto all = quadriga_lib::generate_rays(L);
        const auto sub = quadriga_lib::generate_rays(L, {7, 2});
        REQUIRE(sub.size() == 2);
        CHECK(qd_layout_diff_rays(sub[0][0], all[7][0], 1e-12) == "");
        CHECK(qd_layout_diff_rays(sub[1][0], all[2][0], 1e-12) == "");
        for (arma::uword l = 0; l < L.links.n_link(); l += 3)
            CHECK(qd_layout_diff_rays(quadriga_lib::generate_rays(L, l)[0][0], all[l][0], 1e-12) == "");
    }
    // Scatterers on request for the TR methods; quadriga has them always
    quadriga_lib::init_link_states(L);
    CHECK(quadriga_lib::generate_rays(L, 0)[0][0].fbs_pos.is_empty());
    CHECK(quadriga_lib::generate_rays(L, 0, true)[0][0].fbs_pos.n_cols == quadriga_lib::generate_rays(L, 0)[0][0].n_ray());
}

TEST_CASE("generate_rays - input validation")
{
    quadriga_lib::layout L = two_sites(four_uts());
    auto thr = [&](const quadriga_lib::uint_list &i, const std::string &msg)
    { CHECK(qd_layout_throws([&] { quadriga_lib::generate_rays(L, i); }) == msg); };
    thr({}, "Call 'init_link_states' first.");
    quadriga_lib::init_link_states(L);
    CHECK_NOTHROW(quadriga_lib::generate_rays(L, {1, 2}));
    thr({8}, "Input 'i_link' has an index outside the links.");
    thr({1, 1}, "Input 'i_link' must not repeat a link.");
    quadriga_lib::layout M = L;
    M.rx.pos(0, 0) += 1.0;
    CHECK(qd_layout_throws([&] { quadriga_lib::generate_rays(M); }) ==
          "The layout has changed since 'init_link_states': call 'init_link_states' again.");
    M = L;
    M.links.los[0] = 2;
    CHECK(qd_layout_throws([&] { quadriga_lib::generate_rays(M); }) == "Member 'links' of 'layout' is not valid: Member 'los' must be 0 or 1.");
    // Members edited to a wrong size after stage 1 throw instead of being read out of range
    M = L;
    M.rx.scenario = {"UMa"};
    CHECK(qd_layout_throws([&] { quadriga_lib::generate_rays(M); }).rfind("Input 'layout' is not valid: rx: ", 0) == 0);
    CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_layout(M); }).rfind("Input 'layout' is not valid: rx: ", 0) == 0);
    M = L;
    M.tx.clutter.zeros(3, 1);
    CHECK(qd_layout_throws([&] { quadriga_lib::generate_rays(M); }).rfind("Input 'layout' is not valid: tx: ", 0) == 0);
    M = L;
    quadriga_lib::set_antenna(M, "omni");
    quadriga_lib::init_link_states(M);
    const auto rays = quadriga_lib::generate_rays(M);
    M.antenna_args.push_back(M.antenna_args[0]); // a second description without an array
    M.tx.i_antenna[2] = 2;
    CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_from_rays(M, rays); }) ==
          "Input 'layout' is not valid: Members 'antenna' and 'antenna_args' must have the same length.");
    M = L;
    M.links.conf_names = {"UMa_LOS", "UMa_NLOS"}; // the derived conf of a building link is not in the list
    M.links.conf_files.resize(2);
    M.links.i_conf.zeros();
    M.links.indoor[0] = 1;
    CHECK(qd_layout_throws([&] { quadriga_lib::generate_rays(M); }).find("is not in 'links.conf_names'") != std::string::npos);
}

TEST_CASE("get_channels_from_rays - link gain equals path loss, O2I loss and shadow fading")
{
    // UT1 outdoor, UT2 building (fixed d2d_in 20 m, so that the TR geometric distance and the legacy outdoor distance
    // give different path loss), UT3 in a car, UT4 moving 60 m (beyond the SF decorrelation distance); snapshots over
    // 0 to 6 s
    quadriga_lib::layout L = two_sites(four_uts());
    L.rx.indoor = {0, 1, 2, 1};
    L.rx.d2d_in = {arma::datum::nan, 20.0, arma::datum::nan, 4.0};
    L.rx.speed = {0.0, 0.0, 0.0, 10.0};
    L.rx.direction = arma::mat{{arma::datum::nan, arma::datum::nan, arma::datum::nan, 0.5}, {0.0, 0.0, 0.0, 0.0}};
    const arma::vec times = arma::linspace(0.0, 6.0, 4);
    const arma::mat moved = arma::mat{{0.0, 1.0, 2.0, 3.0}} * 2.0 * 10.0; // 0, 20, 40, 60 m
    for (const char *method : {"38.901", "quadriga"})
    {
        const bool qd = std::string(method) == "quadriga";
        quadriga_lib::init_link_states(L, method);
        const auto &K = L.links;
        const auto rays = quadriga_lib::generate_rays(L);
        arma::umat idx;
        const auto ch = quadriga_lib::get_channels_from_rays(L, rays, {}, times, "taps", &idx);
        REQUIRE(ch.size() == 3 * 4);
        for (size_t c = 0; c < ch.size(); ++c)
        {
            const arma::uword i_tx = idx(0, c), t = idx(1, c), s = i_tx == 2 ? 1 : 0, l = link_of(K, s, t);
            const arma::mat t0 = L.tx.pos.col(i_tx), r0 = L.rx.pos.col(t);
            arma::mat R = arma::repmat(r0, 1, 4);
            if (t == 3)
                for (arma::uword i = 0; i < 4; ++i)
                    R(0, i) += moved(i) * std::cos(0.5), R(1, i) += moved(i) * std::sin(0.5);
            const conf_t cf = conf(K.conf_names[K.i_conf[l]]);
            std::string model;
            double dmax = 25.0;
            if (t == 1 || t == 3)
                model = K.o2i_model[t] == 0 ? "low" : "high";
            if (t == 2)
                model = "car";
            arma::mat rx_all; // positions of the links of the group (bias removal of quadriga)
            for (arma::uword m = 0; m < K.n_link(); ++m)
                if (K.i_tx[m] == s && K.i_conf[m] == K.i_conf[l] && K.image[m] == K.image[l])
                    rx_all = arma::join_rows(rx_all, L.rx.pos.col(K.i_rx[m]));
            const arma::vec g = gain(method, cf, K.gbsm_seed[l], t0, r0, arma::repmat(t0, 1, 4), R, K.floor_id[l], K.d2d_in[l],
                                     L.center_freq, 0, model, K.o2i_rnd.row(t), dmax, 4711, "UMa", true, rx_all);
            const arma::vec lg = qd_layout_par<arma::vec>(ch[c], "link_gain_dB");
            REQUIRE(lg.n_elem == 4);
            CHECK(arma::approx_equal(lg, g, "absdiff", 1e-9));
            if (t == 1 && !qd) // TR methods: the geometric distance; the legacy outdoor distance differs by more than 0.3 dB
            {
                arma::mat sf;
                const arma::mat a = quadriga_lib::path_loss(cf.pl, cf.pp, cf.sp, L.center_freq, t0, r0, &sf);
                const arma::mat b = quadriga_lib::path_loss(cf.pl, cf.pp, cf.sp, L.center_freq, t0, r0, &sf, arma::vec{20.0});
                CHECK(std::abs(a(0, 0) - b(0, 0)) > 0.3);
            }
            if (t == 3) // the O2I deviation and the SF move with the UT
                CHECK(arma::stddev(lg) > 0.1);
        }
    }
}

TEST_CASE("get_channels_from_rays - the random environment height is held along a track")
{
    // UMa LOS at 1000 m and 19.5 m UT height: the loss beyond the breakpoint depends on hE; the UT moves 100 m
    quadriga_lib::layout L = two_sites(col3(1000.0, 0.0, 19.5));
    L.rx.scenario = {"UMa_LOS"};
    L.rx.speed = {10.0};
    L.rx.direction = arma::mat(arma::vec{1.5707963267948966, 0.0});
    L.tx.pos.col(2) = col3(-5000.0, 0.0, 25.0);
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    const arma::vec times = arma::linspace(0.0, 10.0, 6);
    arma::umat idx;
    const auto ch = quadriga_lib::get_channels_from_rays(L, quadriga_lib::generate_rays(L), {}, times, "taps", &idx);
    const arma::uword l = link_of(K, 0, 0);
    const conf_t cf = conf("UMa_LOS");
    arma::mat R = arma::repmat(col3(1000.0, 0.0, 19.5), 1, 6);
    R.row(1) = (times * 10.0).t();
    const arma::mat t0 = col3(0.0, 0.0, 25.0), T = arma::repmat(t0, 1, 6);
    arma::vec kfr;
    const arma::vec held = gain("38.901", cf, K.gbsm_seed[l], t0, R.col(0), T, R, 0.0, 0.0, L.center_freq, 0, "", {}, 25.0, 0, "", true, {}, &kfr);
    const arma::vec free = gain("38.901", cf, K.gbsm_seed[l], t0, R.col(0), T, R, 0.0, 0.0, L.center_freq, 0, "", {}, 25.0, 0, "", false);
    CHECK(arma::abs(held - free).max() > 0.5); // the test can tell the two apart
    CHECK(arma::approx_equal(qd_layout_par<arma::vec>(ch[0], "link_gain_dB"), held, "absdiff", 1e-9));

    // The K-factor of the coefficient powers follows the K-factor at the positions: LOS (tap 0) over NLOS power
    const auto cr = quadriga_lib::get_channels_from_rays(L, quadriga_lib::generate_rays(L, l), l, times, "rays");
    const arma::u32_vec tap = qd_layout_par<arma::u32_vec>(cr[0], "tap");
    arma::vec kc(6);
    for (arma::uword s = 0; s < 6; ++s)
    {
        double los = 0.0, nlos = 0.0;
        for (arma::uword p = 0; p < tap.n_elem; ++p)
            (tap(p) == 0 ? los : nlos) += cr[0].path_gain[s](p);
        kc(s) = los / nlos;
    }
    CHECK(arma::stddev(kfr) > 0.01); // the K-factor changes along the track
    CHECK(arma::approx_equal(kc / kc(0), kfr, "reldiff", 1e-9));
}

TEST_CASE("get_channels_from_rays - output order, index, names and pairing")
{
    // UT1 has two panels (sectors 0 and 1), two frequencies; the pair BS2-UT2 is not linked
    quadriga_lib::layout L = two_sites(four_uts().cols(0, 1));
    L.rx = qd_layout_points({"UT1", "UT1", "UT2"}, {0, 1, 0}, arma::mat{{120.0, 120.0, -90.0}, {40.0, 40.0, 150.0}, {1.5, 1.5, 1.5}});
    L.center_freq = {3.5e9, 28e9};
    L.pairing = arma::umat{{0, 0, 1}, {0, 1, 0}};
    quadriga_lib::init_link_states(L);
    CHECK(L.links.n_link() == 4); // links hold all pairs, pairing selects at the output
    arma::umat idx;
    const auto ch = quadriga_lib::get_channels_from_rays(L, quadriga_lib::generate_rays(L), {}, {}, "taps", &idx);
    const arma::umat expect = {{0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1},
                               {0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2},
                               {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}};
    REQUIRE(ch.size() == 16);
    CHECK(qd_layout_same_int(idx, expect));
    CHECK(ch[0].name == "BS1:0_UT1:0");
    CHECK(ch[2].name == "BS2:0_UT1:0");
    CHECK(ch[4].name == "BS1:1_UT1:1");
    CHECK(ch[7].name == "BS1:1_UT2:0");
    for (size_t c = 0; c < ch.size(); ++c)
        CHECK(ch[c].center_freq(0) == L.center_freq(idx(2, c)));
    // The selection of propagation links: only channels whose link is in i_link
    const arma::uword l = link_of(L.links, 1, 0);
    const auto one = quadriga_lib::get_channels_from_rays(L, quadriga_lib::generate_rays(L, l), l, {}, "taps", &idx);
    REQUIRE(one.size() == 4);
    CHECK(one[0].name == "BS2:0_UT1:0");
    CHECK(one[1].name == "BS2:0_UT1:1");
    // Without sectors on the Rx side, the Rx part of the name has none
    L.rx = qd_layout_points({"UT1", "UT2"}, {0, 0}, four_uts().cols(0, 1));
    L.pairing.reset();
    quadriga_lib::init_link_states(L);
    CHECK(quadriga_lib::get_channels_from_rays(L, quadriga_lib::generate_rays(L))[1].name == "BS1:1_UT1");
}

TEST_CASE("get_channels_from_rays - the generic antenna gives the polarization matrix as coefficients")
{
    // xpol at both ends, orientation 0, the drop position: the coefficient of each kept ray (power > 0, LOS and GR) and
    // element pair is sqrt(power · 10^(link gain / 10)) times the entry of the polarization matrix of the ray data times
    // exp(-j 2 pi L / lambda) with L = |rx - tx| + c · delay, with V = element 0 and H = element 1; oracle from the
    // output of generate_rays, not from the channel
    quadriga_lib::layout L = two_sites(four_uts().cols(0, 0));
    L.tx.orientation.zeros();
    quadriga_lib::init_link_states(L);
    const auto rays = quadriga_lib::generate_rays(L, 0);
    const quadriga_lib::gbsm_rays &y = rays[0][0];
    const auto ch = quadriga_lib::get_channels_from_rays(L, rays, 0, {}, "rays");
    REQUIRE(ch.size() == 2);
    const auto &c = ch[0];
    REQUIRE(c.coeff_re[0].n_rows == 2);
    REQUIRE(c.coeff_re[0].n_cols == 2);
    const double lambda = 299792458.0 / 3.5e9, d3 = arma::norm(y.rx_pos - y.tx_pos);
    const double g = std::pow(10.0, 0.1 * qd_layout_par<arma::vec>(c, "link_gain_dB")(0));
    arma::uword p = 0;
    double worst = 0.0;
    for (arma::uword i = 0; i < y.n_ray(); ++i)
    {
        if (!(y.power(i, 0) > 0.0 || y.path_type(i) <= 1))
            continue;
        REQUIRE(p < c.coeff_re[0].n_slices);
        const double amp = std::sqrt(y.power(i, 0) * g);
        const std::complex<double> ph = std::polar(amp, -2.0 * arma::datum::pi * (d3 + 299792458.0 * y.delay(i)) / lambda);
        for (arma::uword r = 0; r < 2; ++r)
            for (arma::uword t = 0; t < 2; ++t)
            {
                const arma::uword q = 2 * (2 * t + r); // Re/Im of VV, HV, VH, HH
                const std::complex<double> e = ph * std::complex<double>(y.polmat(q, i, 0), y.polmat(q + 1, i, 0));
                const std::complex<double> a(c.coeff_re[0](r, t, p), c.coeff_im[0](r, t, p));
                worst = std::max(worst, std::abs(a - e) / std::max(amp, 1e-300));
            }
        ++p;
    }
    CHECK(p == c.coeff_re[0].n_slices);
    CHECK(worst < 1e-6);
}

TEST_CASE("get_channels_from_rays - sectors share rays and differ by antennas")
{
    quadriga_lib::layout L = two_sites(four_uts().cols(0, 0));
    quadriga_lib::set_antenna(L, "3gpp", {{"panel", {2, 1, 1, 1}}}, false, {0});
    quadriga_lib::init_link_states(L);
    const auto ch = quadriga_lib::get_channels_from_rays(L, quadriga_lib::generate_rays(L, 0), 0, {}, "rays");
    REQUIRE(ch.size() == 2);
    CHECK(ch[0].coeff_re[0].n_cols == 2);  // 3gpp panel, 2 elements
    CHECK(ch[1].coeff_re[0].n_cols == 2);  // xpol
    CHECK(arma::approx_equal(ch[0].path_length[0], ch[1].path_length[0], "absdiff", 1e-12));
    CHECK(arma::approx_equal(ch[0].path_gain[0], ch[1].path_gain[0], "absdiff", 1e-20));
    CHECK(!arma::approx_equal(ch[0].coeff_re[0], ch[1].coeff_re[0], "absdiff", 1e-9));
}

TEST_CASE("get_channels_from_rays - a moving site leaves the values of other sites unchanged")
{
    quadriga_lib::layout L = two_sites(four_uts());
    const arma::vec times = arma::linspace(0.0, 1.0, 3);
    quadriga_lib::init_link_states(L);
    const auto r0 = quadriga_lib::generate_rays(L);
    const auto c0 = quadriga_lib::get_channels_from_rays(L, r0, {}, times);
    const auto K0 = L.links;
    L.tx.speed = {2.0, 2.0, 0.0}; // BS1 moves: dual mobility for BS1 only
    L.tx.direction = arma::mat{{0.3, 0.3, arma::datum::nan}, {0.0, 0.0, 0.0}};
    quadriga_lib::init_link_states(L);
    const auto r1 = quadriga_lib::generate_rays(L);
    const auto c1 = quadriga_lib::get_channels_from_rays(L, r1, {}, times);
    REQUIRE(c1.size() == c0.size());
    arma::uword n_diff = 0;
    for (arma::uword l = 0; l < L.links.n_link(); ++l)
    {
        CHECK(L.links.los[l] == K0.los[l]);
        CHECK(L.links.gbsm_seed[l] == K0.gbsm_seed[l]);
        if (L.links.i_tx[l] == 1)
            CHECK(qd_layout_diff_rays(r1[l][0], r0[l][0], 1e-12) == "");
        else
            n_diff += qd_layout_diff_rays(r1[l][0], r0[l][0], 1e-12) != "";
    }
    CHECK(n_diff > 0); // the moving site uses dual mobility
    for (size_t c = 0; c < c0.size(); ++c)
        if (c % 3 == 2) // BS2
        {
            CHECK(qd_layout_same_data(c1[c].coeff_re[2], c0[c].coeff_re[2], 1e-12));
            CHECK(arma::approx_equal(qd_layout_par<arma::vec>(c1[c], "link_gain_dB"), qd_layout_par<arma::vec>(c0[c], "link_gain_dB"), "absdiff", 1e-9));
        }
}

TEST_CASE("get_channels_from_rays - input validation")
{
    quadriga_lib::layout L = two_sites(four_uts().cols(0, 1));
    const std::vector<std::vector<quadriga_lib::gbsm_rays>> none;
    CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_from_rays(L, none); }) == "Call 'init_link_states' first.");
    quadriga_lib::init_link_states(L);
    const auto rays = quadriga_lib::generate_rays(L);
    CHECK_NOTHROW(quadriga_lib::get_channels_from_rays(L, rays));
    auto thr = [&](const std::vector<std::vector<quadriga_lib::gbsm_rays>> &r, const quadriga_lib::uint_list &i, const arma::vec &times,
                   const std::string &output, const std::string &mobility, double overlap, const std::string &msg)
    { CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_from_rays(L, r, i, times, output, nullptr, mobility, false, false, overlap); }) == msg); };
    thr(rays, {0, 1}, {}, "taps", "auto", 0.5, "Input 'rays' must have one entry per link of 'i_link'.");
    thr(rays, {}, {}, "tap", "auto", 0.5, "Input 'output' must be 'taps' or 'rays'.");
    thr(rays, {}, {}, "taps", "drift2", 0.5, "Input 'mobility' must be 'phase', 'drift' or 'auto'.");
    thr(rays, {}, {}, "taps", "auto", 1.5, "Input 'overlap' must be in [0, 1].");
    thr(rays, {}, arma::vec{0.0, arma::datum::nan}, "taps", "auto", 0.5, "Input 'times' must be finite.");
    thr(rays, {}, {}, "taps", "drift", 0.5,
        "Channel 'BS1:0_UT1': Mobility 'drift' needs 'rays.fbs_pos' and 'rays.lbs_pos'; solve the scatterers first.");
    auto r2 = rays;
    r2[1].push_back(r2[1][0]);
    thr(r2, {}, {}, "taps", "auto", 0.5, "Entry 1: Input 'rays' must have 1 or n_freq objects per link.");
    r2 = rays;
    r2[2][0].center_freq = {2.0e9};
    thr(r2, {}, {}, "taps", "auto", 0.5, "Entry 2: The frequencies of 'rays' do not match 'center_freq' of the layout.");
    // Members outside the hash are checked where they address memory
    quadriga_lib::layout M = L;
    M.pairing = arma::umat{{0, 2}, {0, 0}};
    CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_from_rays(M, rays); }) ==
          "Input 'layout' is not valid: member 'pairing' must have 2 rows with indices of sites and terminals.");
    CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_layout(M); }) ==
          "Input 'layout' is not valid: member 'pairing' must have 2 rows with indices of sites and terminals.");
    M = L;
    M.rx.i_antenna[0] = 3;
    CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_from_rays(M, rays); }) ==
          "Input 'layout' is not valid: rx: Member 'i_antenna' is larger than the number of antennas.");
    // Two tracks without times need equal row counts
    L.rx = qd_layout_points({"UT1"}, {0}, arma::mat{{10.0, 20.0, 30.0}, {100.0, 100.0, 100.0}, {1.5, 1.5, 1.5}});
    L.tx = qd_layout_points({"BS1"}, {0}, arma::mat{{0.0, 1.0}, {0.0, 0.0}, {25.0, 25.0}});
    L.tx.scenario = {"UMa", "UMa"};
    quadriga_lib::init_link_states(L);
    const auto r3 = quadriga_lib::generate_rays(L);
    CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_from_rays(L, r3); }) ==
          "Channel 'BS1_UT1': the Tx and the Rx track have 2 and 3 rows; give 'times' or tracks with equal row counts.");
}

TEST_CASE("get_channels_from_rays - default quadriga run")
{
    quadriga_lib::layout L = two_sites(four_uts().cols(0, 1));
    L.center_freq = {3.5e9, 3.6e9};
    quadriga_lib::init_link_states(L, "quadriga");
    CHECK(L.links.sos_preset == "Comb256");
    const auto rays = quadriga_lib::generate_rays(L);
    REQUIRE(rays[0].size() == 1);
    CHECK(rays[0][0].n_freq() == 2);
    CHECK(rays[0][0].fbs_pos.n_cols == rays[0][0].n_ray());
    const auto ch = quadriga_lib::get_channels_from_rays(L, rays, {}, arma::vec{0.0, 0.5});
    REQUIRE(ch.size() == 3 * 2 * 2);
    for (const auto &c : ch)
    {
        CHECK(c.coeff_re.size() == 2);
        CHECK(c.coeff_re[0].is_finite());
        CHECK(arma::accu(arma::square(c.coeff_re[0])) > 0.0);
    }
}

TEST_CASE("generate_rays - an edited indoor state equals the state fixed by the indoor column")
{
    // UT1 at 7.5 m (floor 3): its indoor state is edited in the links of one layout and fixed by the column in another,
    // both with NLOS links; the rays and the link gain of its links agree, in both directions (building to outdoor and
    // outdoor to building), so that the floor, the indoor distance and the O2I model follow the edited state
    arma::mat ut = four_uts();
    ut(2, 0) = 7.5;
    for (const char *method : {"38.901", "quadriga"})
        for (int to_building = 0; to_building < 2; ++to_building)
        {
            INFO(method << (to_building ? ", outdoor to building" : ", building to outdoor"));
            quadriga_lib::layout A = two_sites(ut), B = two_sites(ut);
            A.rx.indoor = {to_building ? 0 : 1, 0, 0, 0};
            B.rx.indoor = {to_building ? 1 : 0, 0, 0, 0};
            quadriga_lib::init_link_states(A, method);
            quadriga_lib::init_link_states(B, method);
            std::vector<arma::uword> ls;
            for (arma::uword l = 0; l < A.links.n_link(); ++l)
            {
                REQUIRE(A.links.i_rx[l] == B.links.i_rx[l]);
                A.links.los[l] = B.links.los[l]; // the LOS draw uses the drawn indoor distance; equal states for the comparison
                if (A.links.i_rx[l] == 0)
                    A.links.indoor[l] = to_building ? 1 : 0, A.links.los[l] = B.links.los[l] = 0, ls.push_back(l);
            }
            CHECK(A.links.floor_id[ls[0]] == (to_building ? 0u : 2u)); // the stored values are those of the drawn state
            const auto ra = quadriga_lib::generate_rays(A, ls), rb = quadriga_lib::generate_rays(B, ls);
            for (size_t j = 0; j < ls.size(); ++j)
                CHECK(qd_layout_diff_rays(ra[j][0], rb[j][0], 1e-12) == "");
            const auto ca = quadriga_lib::get_channels_from_rays(A, ra, ls), cb = quadriga_lib::get_channels_from_rays(B, rb, ls);
            REQUIRE(ca.size() == cb.size());
            for (size_t c = 0; c < ca.size(); ++c)
                CHECK(arma::approx_equal(qd_layout_par<arma::vec>(ca[c], "link_gain_dB"), qd_layout_par<arma::vec>(cb[c], "link_gain_dB"), "absdiff", 1e-9));
        }
}

TEST_CASE("init_link_states - a conf that several families list belongs to the family of the site")
{
    // InF_LOS is listed by all InF families; in a DH hall an edited NLOS state gets InF_NLOS_DH, as the exact name does
    quadriga_lib::layout L = two_sites(four_uts() * 0.1, "InF_DH"), M = L;
    L.tx.pos.row(2).fill(8.0), M.tx.pos.row(2).fill(8.0);
    L.rx.scenario = {"InF_LOS", "", "", ""};
    M.rx.scenario = {"InF_NLOS_DH", "", "", ""};
    quadriga_lib::init_link_states(L);
    quadriga_lib::init_link_states(M);
    const auto &C = L.links.conf_names;
    CHECK(std::count(C.begin(), C.end(), std::string("InF_NLOS_DH")) == 1);
    CHECK(std::count(C.begin(), C.end(), std::string("InF_NLOS_SL")) == 0);
    CHECK(L.links.los[0] == 1);
    L.links.los[0] = 0;
    // An exact name fixes the state, so the edit has no effect; the conf and seed stay those of InF_LOS
    CHECK(qd_layout_diff_rays(quadriga_lib::generate_rays(L, 0)[0][0], quadriga_lib::generate_rays(M, 0)[0][0], 1e-12) != "");
    L.rx.scenario = {"", "", "", ""}; // the family: the edit gives the NLOS conf of the DH hall
    quadriga_lib::init_link_states(L);
    L.links.los[0] = 0;
    CHECK(qd_layout_diff_rays(quadriga_lib::generate_rays(L, 0)[0][0], quadriga_lib::generate_rays(M, 0)[0][0], 1e-12) == "");
}

TEST_CASE("generate_rays - the ground reflection")
{
    // GR_enabled through the overrides of the sites; the chain with the ground permittivity of lsp_generate
    quadriga_lib::layout L = two_sites(four_uts());
    L.tx.scenpar_override["GR_enabled"] = {1.0, 1.0, 1.0};
    L.rx.scenario = {"UMa_NLOS", "UMa_NLOS", "UMa_NLOS", "UMa_NLOS"};
    conf_t nlos = conf("UMa_NLOS");
    nlos.sp["GR_enabled"] = 1.0;
    const arma::mat ut = four_uts();
    for (const char *method : {"38.901", "quadriga"})
    {
        quadriga_lib::init_link_states(L, method);
        const auto rays = quadriga_lib::generate_rays(L);
        const auto a = chain(method, nlos, qd_layout_oracle_seed(4711, "gbsm|BS1|0|UMa_NLOS"), col3(0.0, 0.0, 25.0), ut, arma::u32_vec(4, arma::fill::zeros),
                             arma::vec(4, arma::fill::zeros), std::vector<bool>(4, false), L.center_freq, ut);
        for (arma::uword t = 0; t < 4; ++t)
        {
            const arma::uword l = link_of(L.links, 0, t);
            CHECK(qd_layout_diff_rays(rays[l][0], a[t][0], 1e-12) == "");
            CHECK(arma::accu(rays[l][0].path_type == 1) == 1); // one GR ray
            CHECK(rays[l][0].gr_eps_re.n_elem == 1);
        }
    }
}

TEST_CASE("get_channels_from_rays - antennas that are not evaluated")
{
    // An antenna that is not evaluated uses the generic antenna, except one spaced in wavelengths, whose size in metres is
    // not defined
    quadriga_lib::layout L = two_sites(four_uts().cols(0, 0));
    quadriga_lib::init_link_states(L);
    const auto rays = quadriga_lib::generate_rays(L);
    quadriga_lib::set_antenna(L, "dipole", {}, false, {2}, NAN, false);
    auto ch = quadriga_lib::get_channels_from_rays(L, rays);
    REQUIRE(ch.size() == 3);
    CHECK(ch[2].n_tx() == 2);
    quadriga_lib::set_antenna(L, "3gpp", {{"panel", {4, 1, 1, 1}}}, false, {2}, NAN, false);
    CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_from_rays(L, rays); }) ==
          "Tx node 'BS2' (sector 0): its antenna is spaced in wavelengths and not evaluated; call 'eval_antennas' of the layout.");
    L.eval_antennas();
    ch = quadriga_lib::get_channels_from_rays(L, rays);
    CHECK(ch[2].n_tx() == 4);
}

TEST_CASE("get_channels_from_rays - a layout without propagation links gives empty results")
{
    // The only pair is co-located: stage 1 gives an empty link table, the other stages and the one call empty results
    quadriga_lib::layout L = two_sites(col3(0.0, 0.0, 25.0));
    L.tx = qd_layout_points({"BS1"}, {0}, col3(0.0, 0.0, 25.0));
    L.tx.scenario = {"UMa"};
    arma::umat idx(1, 1);
    CHECK(quadriga_lib::get_channels_layout(L, {}, "taps", "", &idx).empty());
    CHECK(idx.n_rows == 3);
    CHECK(idx.n_cols == 0);
    quadriga_lib::init_link_states(L);
    CHECK(L.links.n_link() == 0);
    const auto rays = quadriga_lib::generate_rays(L);
    CHECK(rays.empty());
    CHECK(quadriga_lib::get_channels_from_rays(L, rays, {}, {}, "taps", &idx).empty());
    CHECK(idx.n_cols == 0);
    CHECK(quadriga_lib::get_channels_layout(L).empty());
}

// A track of 6 rows along +x from (120, 40) in steps of 50 m (rows 0-1 the scenario of the site, UMa; rows 2-3 UMa_NLOS;
// rows 4-5 UMa: three segments) and a track of 3 rows from (-90, 150) with indoor 0, 1, 1 (two segments)
quadriga_lib::layout segment_layout()
{
    quadriga_lib::layout L = two_sites(four_uts().cols(0, 1));
    arma::mat p1(3, 6), p2 = {{-90.0, -80.0, -70.0}, {150.0, 150.0, 150.0}, {1.5, 1.5, 1.5}};
    for (arma::uword i = 0; i < 6; ++i)
        p1(0, i) = 120.0 + 50.0 * (double)i, p1(1, i) = 40.0, p1(2, i) = 1.5;
    L.rx = qd_layout_nodes_from({p1, p2});
    L.rx.scenario = {"", "", "UMa_NLOS", "UMa_NLOS", "", "", "", "", ""};
    L.rx.indoor = {0, 0, 0, 0, 0, 0, 0, 1, 1};
    return L;
}

TEST_CASE("init_link_states - a change of scenario or indoor starts a segment, the wrap image is kept")
{
    quadriga_lib::layout L = segment_layout();
    L.wrap = arma::mat{{500.0, 250.0}, {0.0, 433.0}};
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    REQUIRE(K.n_link() == 3 * 2 + 2 * 2);
    // Order: terminal, site, segment
    CHECK(qd_layout_same_int(K.i_rx, arma::u32_vec{0, 0, 0, 0, 0, 0, 1, 1, 1, 1}));
    CHECK(qd_layout_same_int(K.i_tx, arma::u32_vec{0, 0, 0, 1, 1, 1, 0, 0, 1, 1}));
    CHECK(qd_layout_same_int(K.i_segment, arma::u32_vec{0, 1, 2, 0, 1, 2, 0, 1, 0, 1}));
    const arma::uword start[2][3] = {{0, 2, 4}, {6, 7, 0}};
    const arma::mat images = qd_layout_images(L.wrap);
    const char *site[2] = {"BS1", "BS2"};
    const double sx[2] = {0.0, 400.0}, sy[2] = {0.0, 30.0}, sz[2] = {25.0, 20.0};
    const double lambda = conf("UMa_NLOS").sp.at("LOS_state_lambda");
    arma::uword n_moved = 0; // links whose nearest image at the segment start is not that of the first row
    for (arma::uword l = 0; l < K.n_link(); ++l)
    {
        const arma::uword s = K.i_tx[l], t = K.i_rx[l], k = K.i_segment[l], r0 = start[t][0], r = start[t][k];
        const arma::uword img = nearest_image(images, sx[s], sy[s], L.rx.pos(0, r0), L.rx.pos(1, r0));
        CHECK(K.image[l] == img);
        n_moved += nearest_image(images, sx[s], sy[s], L.rx.pos(0, r), L.rx.pos(1, r)) != img;
        const double x = L.rx.pos(0, r), y = L.rx.pos(1, r), vx = sx[s] + images(0, img), vy = sy[s] + images(1, img);
        const std::string img_s = std::to_string(img);
        if (t == 0 && k == 1) // the exact name fixes NLOS
            CHECK(K.los[l] == 0);
        else // drawn at the segment start from the field of the site and image
        {
            const double d2d = K.d2d_in[l];
            const arma::vec p = quadriga_lib::los_probability("38.901_UMa", arma::vec{std::max(std::hypot(x - vx, y - vy) - d2d, 0.0)}, arma::vec{sz[s]}, arma::vec{1.5});
            const double u = field(lambda, x, y, qd_layout_oracle_seed(4711, std::string("los|") + site[s] + "|" + img_s + "|UMa"));
            CHECK(K.los[l] == (u < p(0) ? 1u : 0u));
        }
        const bool building = t == 1 && k == 1;
        CHECK(K.indoor[l] == (building ? 1 : 0));
        if (building) // the O2I values of the first position of the terminal
        {
            const double c1 = field(25.0, -90.0, 150.0, qd_layout_oracle_seed(4711, "d2din1|UMa")), c2 = field(25.0, -90.0, 150.0, qd_layout_oracle_seed(4711, "d2din2|UMa"));
            CHECK(std::abs(K.o2i_rnd(1, 1) - c1) < 1e-12);
            CHECK(std::abs(K.d2d_in[l] - std::min(c1, c2) * 25.0) < 1e-9);
            CHECK(K.floor_id[l] == 0);
        }
        else
            CHECK(K.d2d_in[l] == 0.0);
        const std::string cname = std::string("UMa_") + (K.los[l] ? "LOS" : "NLOS") + (building ? "_O2I" : "");
        CHECK(K.conf_names[K.i_conf[l]] == cname);
        CHECK(K.gbsm_seed[l] == qd_layout_oracle_seed(4711, std::string("gbsm|") + site[s] + "|" + img_s + "|" + cname));
    }
    CHECK(n_moved > 0); // the kept image differs from the nearest one at some segment start
}

TEST_CASE("generate_rays - the rays of segment k refer to its first row")
{
    // Oracle: the same terminal as a point drop at the first row of the segment, with the scenario and indoor state of
    // the segment and the indoor distance of the link
    quadriga_lib::layout L = segment_layout();
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    const auto rays = quadriga_lib::generate_rays(L);
    const arma::uword start[2][3] = {{0, 2, 4}, {6, 7, 0}};
    for (arma::uword l = 0; l < K.n_link(); ++l)
    {
        const arma::uword s = K.i_tx[l], t = K.i_rx[l], r = start[t][K.i_segment[l]];
        CHECK(arma::approx_equal(arma::vectorise(rays[l][0].rx_pos), L.rx.pos.col(r), "absdiff", 0.0));
        quadriga_lib::layout M = two_sites(L.rx.pos.col(r));
        M.rx.scenario = {L.rx.scenario[r]};
        M.rx.indoor = {L.rx.indoor[r]};
        if (K.indoor[l] == 1)
            M.rx.d2d_in = {K.d2d_in[l]};
        quadriga_lib::init_link_states(M);
        const arma::uword m = link_of(M.links, s, 0);
        CHECK(M.links.los[m] == K.los[l]);
        CHECK(qd_layout_diff_rays(rays[l][0], quadriga_lib::generate_rays(M, m)[0][0], 1e-12) == "");
    }
}

TEST_CASE("get_channels_from_rays - the cross-fade of a track with two segments is continuous in power")
{
    // One site, a track of 12 rows in steps of 1 m: rows 0-5 UMa_LOS, rows 6-11 UMa_NLOS. Oracle: the channel of each
    // segment from gbsm_single_link_channel with the link gain and K-factor ratio of its first row (segment 1 also over
    // rows 0-5), each with delays relative to its own LOS ("phase"), joined by crossfade_segments (a ramp of
    // round(0.5 * 6) = 3 snapshots)
    quadriga_lib::layout L = two_sites(col3(100.0, 20.0, 1.5));
    L.tx = qd_layout_points({"BS1"}, {0}, col3(0.0, 0.0, 25.0));
    L.tx.scenario = {"UMa"};
    for (const auto &kv : small_sets)
        L.tx.scenpar_override[kv.first] = arma::vec{kv.second};
    arma::mat R(3, 12);
    for (arma::uword i = 0; i < 12; ++i)
        R(0, i) = 100.0 + (double)i, R(1, i) = 20.0, R(2, i) = 1.5;
    L.rx = qd_layout_nodes_from({R});
    L.rx.scenario = {"UMa_LOS", "UMa_LOS", "UMa_LOS", "UMa_LOS", "UMa_LOS", "UMa_LOS", "UMa_NLOS", "UMa_NLOS", "UMa_NLOS", "UMa_NLOS", "UMa_NLOS", "UMa_NLOS"};
    L.rx.speed = arma::vec(12, arma::fill::ones);
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    REQUIRE(K.n_link() == 2);
    const auto rays = quadriga_lib::generate_rays(L);
    const arma::mat t0 = col3(0.0, 0.0, 25.0), T = arma::repmat(t0, 1, 12), O(3, 1, arma::fill::zeros);
    const quadriga_lib::arrayant<double> xpol = quadriga_lib::generate_antenna<double>("xpol", NAN, 3.5e9);
    std::vector<double> w;
    for (arma::uword i = 1; i <= 3; ++i)
        w.push_back(0.5 * (1.0 + std::sin(((double)i / 4.0 - 0.5) * 3.14159265358979323846)));
    for (const char *output : {"taps", "rays"})
    {
        std::vector<quadriga_lib::channel<double>> part(2);
        for (arma::uword k = 0; k < 2; ++k)
        {
            const arma::uword a = 0, b = k == 0 ? 5 : 11, r0 = k == 0 ? 0 : 6;
            arma::vec kf;
            const arma::vec g = gain("38.901", conf(k == 0 ? "UMa_LOS" : "UMa_NLOS"), K.gbsm_seed[k], t0, R.col(r0), T.cols(a, b), R.cols(a, b), 0.0, 0.0,
                                     L.center_freq, 0, "", {}, 25.0, 0, "", true, {}, &kf);
            part[k] = quadriga_lib::gbsm_single_link_channel(xpol, xpol, rays[k][0], g, "auto", output, 0, T.cols(a, b), R.cols(a, b), O, O, kf);
        }
        const quadriga_lib::channel<double> X = quadriga_lib::crossfade_segments(part, arma::uvec{0, 6}, 0.5);
        const auto C = quadriga_lib::get_channels_from_rays(L, rays, {}, {}, output);
        REQUIRE(C.size() == 1);
        REQUIRE(C[0].n_snap() == 12);
        for (arma::uword s = 0; s < 12; ++s)
        {
            REQUIRE(arma::size(C[0].coeff_re[s]) == arma::size(X.coeff_re[s]));
            const double m = arma::abs(X.coeff_re[s]).max() + arma::abs(X.coeff_im[s]).max();
            CHECK(arma::approx_equal(C[0].coeff_re[s], X.coeff_re[s], "absdiff", 1e-8 * m));
            CHECK(arma::approx_equal(C[0].coeff_im[s], X.coeff_im[s], "absdiff", 1e-8 * m));
            CHECK(arma::approx_equal(C[0].delay[s], X.delay[s], "absdiff", 1e-15));
        }
        if (std::string(output) == "rays") // the total path gain follows the ramp between the two segments
        {
            auto pg = [](const quadriga_lib::channel<double> &c, arma::uword s) { return arma::accu(c.path_gain[s]); };
            for (arma::uword s = 0; s < 12; ++s)
            {
                const double expect = s < 3 ? pg(part[0], s) : (s < 6 ? (1.0 - w[s - 3]) * pg(part[0], s) + w[s - 3] * pg(part[1], s) : pg(part[1], s));
                CHECK(std::abs(pg(C[0], s) - expect) <= 1e-8 * expect);
            }
            CHECK(std::abs(10.0 * std::log10(pg(part[0], 5) / pg(part[1], 6))) > 1.0); // a hard switch would step
        }

        // Snapshot times at the rows (speed 1 m/s) give the same channel; the one call gives that of the manual path
        const auto Ct = quadriga_lib::get_channels_from_rays(L, rays, {}, arma::regspace(0.0, 11.0), output);
        const auto C1 = quadriga_lib::get_channels_layout(L, {}, output);
        for (arma::uword s = 0; s < 12; ++s)
        {
            CHECK(qd_layout_same_data(Ct[0].coeff_re[s], C[0].coeff_re[s], 1e-12));
            CHECK(qd_layout_same_data(C1[0].coeff_re[s], C[0].coeff_re[s], 1e-12));
            CHECK(qd_layout_same_data(C1[0].delay[s], C[0].delay[s], 1e-12));
        }
    }

    // A selected segment alone gives its own snapshots; non-consecutive selected segments throw
    const auto C0 = quadriga_lib::get_channels_from_rays(L, {rays[1]}, arma::uvec{1}, {}, "rays");
    CHECK(C0[0].n_snap() == 6);
    const arma::vec back = {0.0, 1.0, 8.0, 2.0};
    CHECK(qd_layout_throws([&] { quadriga_lib::get_channels_from_rays(L, rays, {}, back, "rays"); }) ==
          "Terminal 'N0': the snapshots of a segment are not consecutive; a track with several segments needs increasing 'times'.");
}

TEST_CASE("attach_terminals - a track with segments is attached by its first segment")
{
    // The same layout with the scenario of the first segment on all rows has the same links of segment 0
    quadriga_lib::layout L = segment_layout(), M = L;
    M.rx.scenario = {"", "", "", "", "", "", "", "", ""};
    M.rx.indoor = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    quadriga_lib::init_link_states(L), quadriga_lib::init_link_states(M);
    for (const char *rule : {"large_scale", "rsrp"})
    {
        arma::mat ga, gb;
        const arma::uvec a = quadriga_lib::attach_terminals(L, rule, quadriga_lib::generate_rays(L), nullptr, &ga);
        const arma::uvec b = quadriga_lib::attach_terminals(M, rule, quadriga_lib::generate_rays(M), nullptr, &gb);
        CHECK(qd_layout_same_int(a, b));
        CHECK(arma::approx_equal(ga, gb, "absdiff", 1e-9));
    }
}

// One UMa site at (0, 0, 25) with the small random sets; a track of n rows along +x from (x0, 20, 1.5) in steps of
// 'step' m at 'step' m/s (row i at i s), the conf names 'scen' per row, 'indoor' per row (empty: not set)
quadriga_lib::layout track_layout(arma::uword n, double step, const std::vector<std::string> &scen, const std::vector<int> &indoor = {}, double x0 = 100.0)
{
    quadriga_lib::layout L = two_sites(col3(100.0, 20.0, 1.5));
    L.tx = qd_layout_points({"BS1"}, {0}, col3(0.0, 0.0, 25.0));
    L.tx.scenario = {"UMa"};
    for (const auto &kv : small_sets)
        L.tx.scenpar_override[kv.first] = arma::vec{kv.second};
    arma::mat R(3, n);
    for (arma::uword i = 0; i < n; ++i)
        R(0, i) = x0 + step * (double)i, R(1, i) = 20.0, R(2, i) = 1.5;
    L.rx = qd_layout_nodes_from({R});
    L.rx.scenario = scen;
    if (!indoor.empty())
    {
        L.rx.indoor.set_size(n);
        for (arma::uword i = 0; i < n; ++i)
            L.rx.indoor[i] = indoor[i];
    }
    L.rx.speed = arma::vec(n, arma::fill::value(step));
    return L;
}

// n1 rows of conf a, then n2 rows of conf b
std::vector<std::string> two_confs(arma::uword n1, const std::string &a, arma::uword n2, const std::string &b)
{
    std::vector<std::string> s(n1, a);
    s.insert(s.end(), n2, b);
    return s;
}

TEST_CASE("get_channels_from_rays - delays in the ramp are those of the paths' own segments")
{
    // A track away from the site, 20 rows 10 m apart, UMa_LOS then UMa_NLOS: the LOS distance grows by 100 m over the
    // old segment. In "phase" each segment's paths keep their delays relative to its own LOS, in "drift" to the LOS at
    // the snapshot; so in the ramp the old paths have the delays of the old segment alone, no powered path has a
    // negative relative delay, no delay (also of an empty tap slot) is negative, and in "drift" the absolute LOS delay
    // changes by at most the step of the Rx (10 m / c) from snapshot to snapshot
    quadriga_lib::layout L = track_layout(20, 10.0, two_confs(10, "UMa_LOS", 10, "UMa_NLOS"));
    quadriga_lib::init_link_states(L);
    REQUIRE(L.links.n_link() == 2);
    const auto rays = quadriga_lib::generate_rays(L, {}, true);
    const double c0 = 299792458.0;
    for (const char *mob : {"phase", "drift"})
        for (const char *out : {"taps", "rays"})
        {
            const auto C = quadriga_lib::get_channels_from_rays(L, rays, {}, {}, out, nullptr, mob);
            const auto A = quadriga_lib::get_channels_from_rays(L, {rays[0]}, arma::uvec{0}, {}, out, nullptr, mob); // old segment alone
            REQUIRE(C[0].n_snap() == 20);
            REQUIRE(A[0].n_snap() == 10);
            for (arma::uword s = 0; s < 20; ++s)
            {
                const arma::cube &d = C[0].delay[s];
                CHECK(d.min() >= -1e-12);
                for (arma::uword p = 0; p < d.n_slices; ++p)
                    if (arma::accu(arma::square(C[0].coeff_re[s].slice(p)) + arma::square(C[0].coeff_im[s].slice(p))) > 0.0)
                        CHECK(d.slice(p).min() >= -1e-12);
            }
            if (std::string(out) == "rays") // the old NLOS paths of the ramp (5 snapshots) follow LOS and GR in old order
                for (arma::uword s = 5; s < 10; ++s)
                    for (arma::uword p = 1; p < A[0].delay[s].n_slices; ++p)
                        CHECK(std::abs(C[0].delay[s](0, 0, p) - A[0].delay[s](0, 0, p)) <= 1e-15);
        }
    const auto D = quadriga_lib::get_channels_from_rays(L, rays, {}, {}, "rays", nullptr, "drift", false, true);
    for (arma::uword s = 1; s < 20; ++s)
        CHECK(std::abs(D[0].delay[s](0, 0, 0) - D[0].delay[s - 1](0, 0, 0)) <= 10.0 / c0 + 1e-15);
}

TEST_CASE("get_channels_from_rays - the O2I loss has no step at the first row of an indoor segment")
{
    // 200 rows 0.1 m apart, rows 0-99 outdoor (UMa_LOS), rows 100-199 in a building (UMa_LOS_O2I, indoor 1). The O2I
    // values are drawn at the first row of the track; the loss deviation follows the 10 m field at every other row.
    // Oracle: the link gain of segment 1 alone against path_loss, lsp_generate and o2i_loss with the deviation at each
    // position; and the total path gain changes by less than 3 dB per 0.1 m in the indoor segment (the deviation
    // decorrelates over 10 m: the difference over 0.1 m has a standard deviation of about 0.14 sigma, below 1 dB)
    std::vector<int> io(200, 0);
    std::fill(io.begin() + 100, io.end(), 1);
    quadriga_lib::layout L = track_layout(200, 0.1, two_confs(100, "UMa_LOS", 100, "UMa_LOS_O2I"), io);
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    REQUIRE(K.n_link() == 2);
    REQUIRE(K.indoor[1] == 1);
    const auto rays = quadriga_lib::generate_rays(L);
    const auto S = quadriga_lib::get_channels_from_rays(L, {rays[1]}, arma::uvec{1}, {}, "taps");
    REQUIRE(S[0].n_snap() == 100);
    const arma::mat R = L.rx.pos.cols(100, 199), T = arma::repmat(col3(0.0, 0.0, 25.0), 1, 100);
    const std::string model = K.o2i_model[0] == 0 ? "low" : "high";
    const arma::vec g = gain("38.901", conf("UMa_LOS_O2I"), K.gbsm_seed[1], col3(0.0, 0.0, 25.0), R.col(0), T, R, K.floor_id[1], K.d2d_in[1], L.center_freq, 0,
                             model, K.o2i_rnd.row(0), 25.0, 4711, "UMa", true, {}, nullptr, false, L.rx.pos.col(0));
    CHECK(arma::approx_equal(qd_layout_par<arma::vec>(S[0], "link_gain_dB"), g, "absdiff", 1e-9));
    const auto C = quadriga_lib::get_channels_from_rays(L, rays, {}, {}, "rays");
    for (arma::uword s = 100; s + 1 < 200; ++s)
        CHECK(std::abs(10.0 * std::log10(arma::accu(C[0].path_gain[s]) / arma::accu(C[0].path_gain[s + 1]))) < 3.0);
}

TEST_CASE("get_channels_from_rays - times that skip a short segment join its neighbours")
{
    // 11 rows at 1 s, a one-row middle segment (row 5); snapshots every 2 s hold segments 0 and 2 only
    std::vector<std::string> sc(11, "UMa_LOS");
    sc[5] = "UMa_NLOS";
    quadriga_lib::layout L = track_layout(11, 1.0, sc);
    quadriga_lib::init_link_states(L);
    REQUIRE(L.links.n_link() == 3);
    const auto rays = quadriga_lib::generate_rays(L);
    const arma::vec times = arma::regspace(0.0, 2.0, 10.0);
    std::vector<quadriga_lib::channel<double>> C;
    CHECK_NOTHROW(C = quadriga_lib::get_channels_from_rays(L, rays, {}, times, "rays"));
    REQUIRE(C.size() == 1);
    CHECK(C[0].n_snap() == 6);
    const auto A = quadriga_lib::get_channels_from_rays(L, {rays[0]}, arma::uvec{0}, times, "rays"); // segment 0 alone: t = 0, 2, 4
    REQUIRE(A[0].n_snap() == 3);
    CHECK(qd_layout_same_data(C[0].coeff_re[0], A[0].coeff_re[0], 1e-12));
}

TEST_CASE("generate_rays - with a moving site, segment k pairs its first row with the site at the same time")
{
    // BS1 moves at 2 m/s along azimuth 0.3; the Rx track (12 rows, 1 m at 1 m/s) changes from UMa_LOS to UMa_NLOS at
    // row 6, i.e. at 6 s. Oracle: the GBSM chain (dual mobility) at the Tx position of 6 s and the Rx at row 6; the link
    // gain of segment 1 alone at the times 6 ... 11 s with that drop
    quadriga_lib::layout L = track_layout(12, 1.0, two_confs(6, "UMa_LOS", 6, "UMa_NLOS"));
    L.tx.speed = {2.0};
    L.tx.direction = arma::mat(arma::vec{0.3, 0.0});
    quadriga_lib::init_link_states(L);
    const auto &K = L.links;
    REQUIRE(K.n_link() == 2);
    const auto rays = quadriga_lib::generate_rays(L);
    const arma::mat dir = col3(std::cos(0.3), std::sin(0.3), 0.0);
    for (arma::uword k = 0; k < 2; ++k)
    {
        const double t = 6.0 * (double)k;
        const arma::mat tx = col3(0.0, 0.0, 25.0) + 2.0 * t * dir, rx = L.rx.pos.col(6 * k);
        CHECK(arma::approx_equal(arma::vectorise(rays[k][0].tx_pos), arma::vectorise(tx), "absdiff", 1e-12));
        const auto ref = chain("38.901", conf(k == 0 ? "UMa_LOS" : "UMa_NLOS"), K.gbsm_seed[k], tx, rx, arma::u32_vec{0}, arma::vec{0.0}, {false}, L.center_freq,
                               {}, false, 2.0e9, false, true);
        CHECK(qd_layout_diff_rays(rays[k][0], ref[0][0], 1e-12) == "");
    }
    const arma::vec times = arma::regspace(6.0, 11.0);
    const auto S = quadriga_lib::get_channels_from_rays(L, {rays[1]}, arma::uvec{1}, times, "taps");
    REQUIRE(S[0].n_snap() == 6);
    arma::mat T(3, 6);
    for (arma::uword i = 0; i < 6; ++i)
        T.col(i) = col3(0.0, 0.0, 25.0) + 2.0 * times[i] * dir;
    const arma::vec g = gain("38.901", conf("UMa_NLOS"), K.gbsm_seed[1], T.col(0), L.rx.pos.col(6), T, L.rx.pos.cols(6, 11), 0.0, 0.0, L.center_freq, 0, "", {}, 25.0,
                             0, "", true, {}, nullptr, true);
    CHECK(arma::approx_equal(qd_layout_par<arma::vec>(S[0], "link_gain_dB"), g, "absdiff", 1e-9));
}
