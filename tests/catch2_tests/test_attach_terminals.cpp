// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of attach_terminals: the serving Tx node of each terminal by the large-scale gain ("pathloss considering LOS
// angle", TR 38.901 Table 7.8-1) and by RSRP (TR 36.873 eq. (8.1-1)), on small hand-built layouts of two sites.
// Oracles, all computed in this file with public library functions and the link states of init_link_states:
//  - link gain -PL - O2I + 10 lg(SF) at the drop: conf_file_read, gbsm_init_phases and gbsm_draw of the "lsf" set with
//    the link seed, path_loss, lsp_generate and o2i_loss with the stored O2I inputs and model, at the wrap image
//  - LOS antenna gain in closed form: the element of TR 38.901 Table 7.3-1 (65 deg, 30 dB, 8 dBi) at the local angles
//    times the array factor of a vertical column with 0.5 lambda spacing steered to the tilt; isotropic elements 1 (co-
//    polarized) or 0; the mean over the Rx ports
//  - RSRP: for the generic antennas (isotropic V and H elements at both ends) the sum over the rays of power times link
//    gain times |entry of the polarization matrix|^2, the mean over the two Rx ports; for directional antennas with
//    orientations, the per-path coefficients of get_channels_planar for the rays of the link; rays edited to cancel
//    pairwise in their taps show that the tap power differs
//  - serving node: brute-force maximum over the gains, ties to the lowest index
// Tolerances: 4.4e-12 dB (1e-12 relative of the linear value) where the oracle takes the same arithmetic; the closed
// forms of the 3GPP element are compared within 0.01 dB for the linear interpolation of the 1 deg pattern grid (largest
// error found 0.0013 dB).
// Fixtures: qd_test_layout.hpp (point drops, sites of a side, wrap images, message of a throw).

#include <catch2/catch_test_macros.hpp>

#include "quadriga_layout.hpp"
#include "qd_test_layout.hpp"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
using spmap = std::unordered_map<std::string, double>;
using plmap = std::unordered_map<std::string, std::vector<double>>;
using rays_t = std::vector<std::vector<quadriga_lib::gbsm_rays>>;
const double deg = 0.017453292519943295, pi = 3.14159265358979323846, c0 = 299792458.0;
const double tol12 = 4.4e-12; // 1e-12 relative of a linear value, in dB
const double tol_grid = 0.01; // interpolation of the 1 deg grid of the 3GPP element, in dB

// Overrides of all sites: 4 NLOS clusters of 4 rays keep the random sets small; the oracles apply the same
const spmap small_sets = {{"NumClusters", 5.0}, {"NumSubPaths", 4.0}, {"NumClusters_min", 0.0}};

struct conf_t
{
    spmap sp;
    std::string pl;
    plmap pp;
};

conf_t conf(const std::string &name)
{
    conf_t c;
    c.sp = quadriga_lib::conf_file_read("data/config/3GPP_38.901_v19_" + name + ".conf", &c.pl, &c.pp);
    for (const auto &kv : small_sets)
        c.sp[kv.first] = kv.second;
    return c;
}

// Site BS1 with sectors 0 (heading 0) and 1 (heading 'h1') at (0, 0, 25), site BS2 at (500, 0, 25) heading 180 deg;
// UMa; terminals UT1, UT2, ... as point drops at the columns of 'ut' [3, n]; frequencies 'fc'; seed 4711
quadriga_lib::layout two_sites(const arma::mat &ut, const arma::vec &fc = {3.5e9}, double h1 = 180.0 * deg)
{
    quadriga_lib::layout L;
    L.tx = qd_layout_points({"BS1", "BS1", "BS2"}, {0, 1, 0}, arma::mat{{0.0, 0.0, 500.0}, {0.0, 0.0, 0.0}, {25.0, 25.0, 25.0}},
                            arma::vec{0.0, h1, 180.0 * deg});
    L.tx.scenario = {"UMa", "UMa", "UMa"};
    for (const auto &kv : small_sets)
        L.tx.scenpar_override[kv.first] = arma::vec(3, arma::fill::value(kv.second));
    std::vector<std::string> names;
    std::vector<unsigned> sec;
    for (arma::uword i = 0; i < ut.n_cols; ++i)
        names.push_back("UT" + std::to_string(i + 1)), sec.push_back(0);
    L.rx = qd_layout_points(names, sec, ut);
    L.center_freq = fc;
    L.seed = 4711;
    return L;
}

const arma::uvec site_of_node = {0, 0, 1}; // of two_sites

arma::uword link_of(const quadriga_lib::layout_links &K, arma::uword s, arma::uword t)
{
    for (arma::uword l = 0; l < K.n_link(); ++l)
        if (K.i_tx[l] == s && K.i_rx[l] == t)
            return l;
    return K.n_link();
}

// Virtual position of the site of link l (first row plus the wrap image), [3, 1]
arma::mat site_pos(const quadriga_lib::layout &L, arma::uword l)
{
    const qd_layout_site_list S = qd_layout_sites(L.tx);
    arma::mat tx = L.tx.pos.col(L.tx.offset[S.nodes[L.links.i_tx[l]][0]]);
    const arma::mat img = qd_layout_images(L.wrap);
    tx(0, 0) += img(0, L.links.image[l]), tx(1, 0) += img(1, L.links.image[l]);
    return tx;
}

// Link gain -PL - O2I + 10 lg(SF) of link l at the drop, frequency k (UMa: building models "low" and "high", 25 m)
double link_gain(const quadriga_lib::layout &L, arma::uword l, arma::uword k)
{
    const quadriga_lib::layout_links &K = L.links;
    const bool qd = K.method == "quadriga";
    const qd_layout_site_list T = qd_layout_sites(L.rx);
    const conf_t c = conf(K.conf_names[K.i_conf[l]]);
    const arma::mat tx = site_pos(L, l), rx = L.rx.pos.col(L.rx.offset[T.nodes[K.i_rx[l]][0]]);
    arma::mat rx_all; // the terminals of the group (bias removal of quadriga)
    for (arma::uword m = 0; m < K.n_link(); ++m)
        if (K.i_tx[m] == K.i_tx[l] && K.image[m] == K.image[l] && K.i_conf[m] == K.i_conf[l])
            rx_all = arma::join_rows(rx_all, L.rx.pos.col(L.rx.offset[T.nodes[K.i_rx[m]][0]]));
    const arma::cube ph = quadriga_lib::gbsm_init_phases(c.sp, "lsf", K.gbsm_seed[l], -1, K.sos_preset, qd ? tx : arma::mat(),
                                                         qd ? rx_all : arma::mat(), false, qd, true);
    arma::mat rnd, sf;
    quadriga_lib::gbsm_draw(c.sp, "lsf", ph, tx, rx, rnd, -1, K.sos_preset, arma::u32_vec{K.floor_id[l]});
    const arma::mat pl = quadriga_lib::path_loss(c.pl, c.pp, c.sp, L.center_freq, tx, rx, &sf, qd ? arma::vec{K.d2d_in[l]} : arma::vec(),
                                                 arma::vec{rnd(0, 10)});
    arma::cube lsp;
    quadriga_lib::lsp_generate(c.sp, L.center_freq, tx, rx, rnd, lsp, sf);
    double g = -pl(0, k) + 10.0 * std::log10(lsp(0, k, 2));
    const int io = K.indoor[l];
    if (io == 1 || io == 3)
    {
        const arma::mat x = K.o2i_rnd.row(K.i_rx[l]);
        const arma::mat loss = quadriga_lib::o2i_loss(L.center_freq, x, {K.o2i_model[K.i_rx[l]] == 0 ? "low" : "high"}, arma::vec{1.0}, 25.0);
        g -= loss(0, k);
    }
    return g;
}

// Power gain of the element of TR 38.901 Table 7.3-1 with the defaults of generate_antenna (65 deg, SLA_V = A_max =
// 30 dB, G_max = 8 dBi) at the local elevation el and azimuth az in [rad]
double element(double el, double az)
{
    const double v = std::min(12.0 * std::pow(el / deg / 65.0, 2.0), 30.0);
    const double h = std::min(12.0 * std::pow(std::remainder(az, 2.0 * pi) / deg / 65.0, 2.0), 30.0);
    return std::pow(10.0, 0.1 * (8.0 - std::min(v + h, 30.0)));
}

// Array factor of M elements along z with 0.5 lambda spacing and the weights of an electrical downtilt 'tilt' [rad]:
// |sum_m exp(j pi m (sin el + sin tilt))|^2 / M
double column(arma::uword M, double el, double tilt)
{
    const double psi = pi * (std::sin(el) + std::sin(tilt));
    double re = 0.0, im = 0.0;
    for (arma::uword m = 0; m < M; ++m)
        re += std::cos(psi * (double)m), im += std::sin(psi * (double)m);
    return (re * re + im * im) / (double)M;
}

// Direction (azimuth, elevation) of b seen from a, [rad]
void direction(const arma::mat &a, const arma::mat &b, double &az, double &el)
{
    const double dx = b(0, 0) - a(0, 0), dy = b(1, 0) - a(1, 0), dz = b(2, 0) - a(2, 0);
    az = std::atan2(dy, dx), el = std::atan2(dz, std::hypot(dx, dy));
}

// RSRP oracle for any antennas: the per-path coefficients of get_channels_planar for the rays 'y' (frequency f, path
// gain = power times the link gain lg [dB], path length |rx - tx| + c delay), per Rx port the sum of |h|^2 of Tx port
// p over the paths, the mean over the Rx ports; in [dB]
double rsrp_planar(const quadriga_lib::arrayant<double> &ta, const quadriga_lib::arrayant<double> &ra, const arma::mat &to, const arma::mat &ro,
                   const quadriga_lib::gbsm_rays &y, arma::uword f, double lg, arma::uword p)
{
    const arma::vec len = arma::norm(y.rx_pos - y.tx_pos) + c0 * y.delay;
    const arma::vec gain = y.power.col(f) * std::pow(10.0, 0.1 * lg);
    arma::cube re, im, dl;
    quadriga_lib::get_channels_planar<double>(ta, ra, y.tx_pos, to, y.rx_pos, ro, y.aod, y.eod, y.aoa, y.eoa, gain, len, y.polmat.slice(f), re, im, dl,
                                              y.center_freq(f));
    double s = 0.0;
    for (arma::uword k = 0; k < re.n_slices; ++k)
        for (arma::uword r = 0; r < re.n_rows; ++r)
            s += re(r, p, k) * re(r, p, k) + im(r, p, k) * im(r, p, k);
    return 10.0 * std::log10(s / (double)re.n_rows);
}

// Serving node per row of gain_dB: the maximum, ties to the lowest index
arma::uvec argmax_rows(const arma::mat &g)
{
    arma::uvec s(g.n_rows, arma::fill::zeros);
    for (arma::uword t = 0; t < g.n_rows; ++t)
        for (arma::uword i = 1; i < g.n_cols; ++i)
            if (g(t, i) > g(t, s(t)))
                s(t) = i;
    return s;
}

arma::mat uts() { return arma::mat{{120.0, 300.0, -150.0, 500.0}, {40.0, -60.0, 80.0, 0.0}, {1.5, 1.5, 1.5, 25.0}}; }
} // namespace

TEST_CASE("attach_terminals - the large-scale gain is the link gain plus the LOS antenna gain")
{
    // UT2 in a building (O2I loss), UT4 at the position of BS2 (no link: -Inf); the generic antennas (xpol at both ends)
    // give the LOS gain 1/2 of Tx port 0 (V) averaged over the Rx ports V and H. Two frequencies, both evaluated
    quadriga_lib::layout L = two_sites(uts(), {3.5e9, 28.0e9});
    L.rx.indoor = {0, 1, 0, 0};
    L.rx.d2d_in = {arma::datum::nan, 10.0, arma::datum::nan, arma::datum::nan};
    for (const char *method : {"38.901", "quadriga"})
    {
        quadriga_lib::init_link_states(L, method);
        REQUIRE(L.links.indoor[link_of(L.links, 0, 1)] == 1);
        for (arma::uword k = 0; k < 2; ++k)
        {
            arma::uvec site;
            arma::mat g;
            const arma::uvec srv = quadriga_lib::attach_terminals(L, "large_scale", {}, &site, &g, k);
            REQUIRE(g.n_rows == 4);
            REQUIRE(g.n_cols == 3);
            REQUIRE(srv.n_elem == 4);
            REQUIRE(site.n_elem == 4);
            for (arma::uword t = 0; t < 4; ++t)
                for (arma::uword i = 0; i < 3; ++i)
                {
                    const arma::uword l = link_of(L.links, site_of_node(i), t);
                    if (l == L.links.n_link())
                    {
                        CHECK(t == 3);
                        CHECK(i == 2);
                        CHECK(std::isinf(g(t, i)));
                        CHECK(g(t, i) < 0.0);
                        continue;
                    }
                    CHECK(std::abs(g(t, i) - (link_gain(L, l, k) + 10.0 * std::log10(0.5))) < tol12);
                }
            CHECK(qd_layout_same_int(srv, argmax_rows(g)));
            for (arma::uword t = 0; t < 4; ++t)
                CHECK(site(t) == site_of_node(srv(t)));
            CHECK(srv(3) < 2); // the co-located site does not serve

            // The optional outputs alone, and pre-sized
            arma::uvec site2(4);
            arma::mat g2(4, 3);
            const arma::uword *ps = site2.memptr();
            const double *pg = g2.memptr();
            CHECK(qd_layout_same_int(quadriga_lib::attach_terminals(L, "large_scale", {}, nullptr, nullptr, k), srv));
            CHECK(qd_layout_same_int(quadriga_lib::attach_terminals(L, "large_scale", {}, &site2, nullptr, k), srv));
            CHECK(qd_layout_same_int(site2, site));
            quadriga_lib::attach_terminals(L, "large_scale", {}, nullptr, &g2, k);
            CHECK(qd_layout_same_mat(g2, g, 1e-12)); // also -Inf
            CHECK(site2.memptr() == ps);
            CHECK(g2.memptr() == pg);
        }
    }
}

TEST_CASE("attach_terminals - the large-scale gain includes the port coupling and the electrical tilt")
{
    // BS1 sector 0 faces UT1, which is 10 deg below the horizon of BS1 (23.5 m below at 133.27 m); sector 1 faces away;
    // UT2 is in front of sector 1. Tx: 3GPP column of M = 4 V elements (0.5 lambda), one port, 10 deg electrical
    // downtilt; Rx: omni (V). Oracle: element times array factor at the local angles of each pair
    const double d = 23.5 / std::tan(10.0 * deg);
    quadriga_lib::layout L = two_sites(arma::mat{{d, -200.0}, {0.0, 30.0}, {1.5, 1.5}});
    quadriga_lib::set_antenna(L, "3gpp", {{"panel", arma::mat{4.0, 1.0}}, {"ports", arma::mat{1.0, 1.0}}, {"steer", arma::mat{10.0}}});
    quadriga_lib::set_antenna(L, "omni", {}, true);
    quadriga_lib::init_link_states(L);
    arma::uvec site;
    arma::mat g;
    const arma::uvec srv = quadriga_lib::attach_terminals(L, "large_scale", {}, &site, &g);
    for (arma::uword t = 0; t < 2; ++t)
        for (arma::uword i = 0; i < 3; ++i)
        {
            const arma::uword l = link_of(L.links, site_of_node(i), t);
            double az, el;
            direction(L.tx.pos.col(i), L.rx.pos.col(t), az, el);
            const double a = element(el, az - L.tx.orientation(2, i)) * column(4, el, 10.0 * deg);
            CHECK(std::abs(g(t, i) - (link_gain(L, l, 0) + 10.0 * std::log10(a))) < tol_grid);
        }
    CHECK(srv(0) == 0); // the facing sectors serve; sector 1 shows that the lowest index is not a default
    CHECK(srv(1) == 1);
    CHECK(qd_layout_same_int(site, arma::uvec{0, 0}));

    // At the steering direction: G_max + 10 lg(M) - 12 (10 / 65)^2 dB; without the tilt more than 1 dB less
    const arma::uword l = link_of(L.links, 0, 0);
    const double ant = g(0, 0) - link_gain(L, l, 0);
    CHECK(std::abs(ant - (8.0 + 10.0 * std::log10(4.0) - 12.0 * std::pow(10.0 / 65.0, 2.0))) < tol_grid);
    CHECK(ant - 10.0 * std::log10(element(-10.0 * deg, 0.0) * column(4, -10.0 * deg, 0.0)) > 1.0);

    // Tx port 1 of a V/H column (pol 2) is H: zero gain to the V omni; port 0 is the untilted V column
    quadriga_lib::set_antenna(L, "3gpp", {{"panel", arma::mat{4.0, 1.0}}, {"ports", arma::mat{1.0, 1.0}}, {"pol", arma::mat{2.0}}});
    arma::mat g0, g1;
    quadriga_lib::attach_terminals(L, "large_scale", {}, nullptr, &g0, 0, 0);
    quadriga_lib::attach_terminals(L, "large_scale", {}, nullptr, &g1, 0, 1);
    CHECK(std::abs(g0(0, 0) - (link_gain(L, l, 0) + 10.0 * std::log10(element(-10.0 * deg, 0.0) * column(4, -10.0 * deg, 0.0)))) < tol_grid);
    CHECK(g1(0, 0) < g0(0, 0) - 100.0);
}

TEST_CASE("attach_terminals - RSRP equals the sum of the ray powers and differs from the tap power")
{
    // Generic antennas (xpol at both ends, orientation 0): the coefficient of a ray at the Rx port r and Tx port 0 (V) is
    // sqrt(power · link gain) times the entry (r, V) of its polarization matrix. Outdoor confs at 3.5 and 28 GHz, so that
    // the ray powers of both frequencies differ (UMa uses 6 GHz below 6 GHz; the O2I confs have no frequency terms).
    // UT1 is NLOS to BS1; its rays are edited so that the rays of each tap cancel in pairs (equal geometry, opposite
    // polarization matrix): the rays still carry their power, the taps do not. Frequency 1 of two
    quadriga_lib::layout L = two_sites(uts().cols(0, 2), {3.5e9, 28.0e9});
    L.rx.scenario = {"UMa_NLOS", "UMa_LOS", "UMa_NLOS"};
    const arma::uword k = 1;
    for (const char *method : {"38.901", "quadriga"})
    {
        const bool qd = std::string(method) == "quadriga";
        quadriga_lib::init_link_states(L, method);
        auto rays = quadriga_lib::generate_rays(L);
        if (qd)
            CHECK(arma::abs(rays[0][0].power.col(0) - rays[0][0].power.col(1)).max() > 1e-3 * rays[0][0].power.max());
        const arma::uword le = link_of(L.links, 0, 0);
        {
            quadriga_lib::gbsm_rays &y = rays[le][qd ? 0 : k];
            std::vector<arma::uword> open(y.tap.max() + 1, y.n_ray());
            for (arma::uword i = 0; i < y.n_ray(); ++i)
            {
                if (y.path_type(i) != 2)
                    continue;
                arma::uword &a = open[y.tap(i)];
                if (a == y.n_ray())
                {
                    a = i;
                    continue;
                }
                y.delay(i) = y.delay(a), y.aod(i) = y.aod(a), y.eod(i) = y.eod(a), y.aoa(i) = y.aoa(a), y.eoa(i) = y.eoa(a);
                y.power.row(i) = y.power.row(a);
                for (arma::uword f = 0; f < y.polmat.n_slices; ++f)
                    y.polmat.slice(f).col(i) = -y.polmat.slice(f).col(a);
                if (!y.fbs_pos.is_empty())
                    y.fbs_pos.col(i) = y.fbs_pos.col(a), y.lbs_pos.col(i) = y.lbs_pos.col(a), y.fallback(i) = y.fallback(a);
                a = y.n_ray();
            }
        }
        arma::uvec site;
        arma::mat g;
        const arma::uvec srv = quadriga_lib::attach_terminals(L, "rsrp", rays, &site, &g, k);
        REQUIRE(g.n_rows == 3);
        REQUIRE(g.n_cols == 3);
        for (arma::uword t = 0; t < 3; ++t)
            for (arma::uword i = 0; i < 3; ++i)
            {
                const arma::uword l = link_of(L.links, site_of_node(i), t);
                const quadriga_lib::gbsm_rays &y = rays[l][qd ? 0 : k];
                const arma::uword f = qd ? k : 0;
                double s = 0.0;
                for (arma::uword r = 0; r < y.n_ray(); ++r)
                {
                    const arma::mat &M = y.polmat.slice(f);
                    s += y.power(r, f) * (M(0, r) * M(0, r) + M(1, r) * M(1, r) + M(2, r) * M(2, r) + M(3, r) * M(3, r));
                }
                CHECK(std::abs(g(t, i) - (link_gain(L, l, k) + 10.0 * std::log10(0.5 * s))) < tol12);
            }
        CHECK(qd_layout_same_int(srv, argmax_rows(g)));
        for (arma::uword t = 0; t < 3; ++t)
            CHECK(site(t) == site_of_node(srv(t)));

        // The taps of the edited link: mean over the Rx ports of the sum of |h|^2 over the taps
        const auto ch = quadriga_lib::get_channels_from_rays(L, rays_t{rays[le]}, le, {}, "taps", nullptr, "phase");
        REQUIRE(ch.size() == 4);
        const auto &c = ch[2]; // BS1 sector 0, frequency 1
        REQUIRE(c.name == "BS1:0_UT1");
        double tap = 0.0;
        for (arma::uword p = 0; p < c.coeff_re[0].n_slices; ++p)
            for (arma::uword r = 0; r < 2; ++r)
                tap += std::pow(c.coeff_re[0](r, 0, p), 2.0) + std::pow(c.coeff_im[0](r, 0, p), 2.0);
        CHECK(g(0, 0) - 10.0 * std::log10(0.5 * tap) > 3.0);
    }
}

TEST_CASE("attach_terminals - RSRP includes the antennas, their orientations and the Tx port")
{
    // Tx: 3GPP column of 2 V/H element pairs (pol 2), one port per polarization, 6 deg electrical tilt; the sectors are
    // turned in bank, tilt and heading. Rx: one 3GPP element, turned. Ports 0 (V) and 1 (H), both by value; frequency 1
    // of 3.5 and 28 GHz, both methods
    quadriga_lib::layout L = two_sites(uts().cols(0, 2), {3.5e9, 28.0e9});
    L.rx.scenario = {"UMa_NLOS", "UMa_LOS", "UMa_NLOS"};
    quadriga_lib::set_antenna(L, "3gpp", {{"panel", arma::mat{2.0, 1.0}}, {"ports", arma::mat{1.0, 1.0}}, {"pol", arma::mat{2.0}}, {"steer", arma::mat{6.0}}});
    quadriga_lib::set_antenna(L, "3gpp", {}, true);
    L.tx.orientation.row(0) = arma::rowvec{0.05, -0.1, 0.2};
    L.tx.orientation.row(1) = arma::rowvec{0.1, 0.15, -0.05};
    L.tx.orientation.row(2) += arma::rowvec{0.3, -0.4, 0.2};
    L.rx.orientation.row(0) = arma::rowvec{0.1, -0.2, 0.3};
    L.rx.orientation.row(1) = arma::rowvec{-0.15, 0.1, 0.2};
    L.rx.orientation.row(2) = arma::rowvec{1.2, 2.9, -1.4};
    const arma::uword k = 1;
    const auto ta = quadriga_lib::generate_antenna<double>("3gpp", NAN, 28.0e9, {}, {2, 1}, {}, {2}, {1, 1}, arma::mat{6.0});
    const auto ra = quadriga_lib::generate_antenna<double>("3gpp", NAN, 28.0e9);
    for (const char *method : {"38.901", "quadriga"})
    {
        const bool qd = std::string(method) == "quadriga";
        quadriga_lib::init_link_states(L, method);
        const auto rays = quadriga_lib::generate_rays(L);
        double moved = 0.0;
        for (arma::uword p = 0; p < 2; ++p)
        {
            arma::mat g;
            const arma::uvec srv = quadriga_lib::attach_terminals(L, "rsrp", rays, nullptr, &g, k, p);
            for (arma::uword t = 0; t < 3; ++t)
                for (arma::uword i = 0; i < 3; ++i)
                {
                    const arma::uword l = link_of(L.links, site_of_node(i), t);
                    const double o = rsrp_planar(ta, ra, L.tx.orientation.col(i), L.rx.orientation.col(t), rays[l][qd ? 0 : k], qd ? k : 0,
                                                 link_gain(L, l, k), p);
                    CHECK(std::abs(g(t, i) - o) < tol12);
                }
            CHECK(qd_layout_same_int(srv, argmax_rows(g)));

            // The orientations matter
            quadriga_lib::layout M = L;
            M.tx.orientation.zeros(), M.rx.orientation.zeros();
            arma::mat h;
            quadriga_lib::attach_terminals(M, "rsrp", rays, nullptr, &h, k, p);
            moved = std::max(moved, arma::abs(h - g).max());
        }
        CHECK(moved > 1.0);
    }
}

TEST_CASE("attach_terminals - the wrap image is the position of the site")
{
    // Wrap vector (1000, 0): UT1 at (900, 20) is 102 m from the image +a1 of BS1 at (1000, 0), which it sees in the -x
    // direction, so sector 1 (heading 180 deg) serves, not sector 0 that faces the real BS1 position. Tx: one 3GPP
    // element per sector; Rx: generic antenna (LOS gain of the V element averaged over V and H). UT1 LOS
    quadriga_lib::layout L = two_sites(arma::mat(arma::vec{900.0, 20.0, 1.5}));
    L.wrap = arma::mat(arma::vec{1000.0, 0.0});
    L.rx.scenario = {"UMa_LOS"};
    quadriga_lib::set_antenna(L, "3gpp");
    quadriga_lib::init_link_states(L);
    const arma::uword l = link_of(L.links, 0, 0);
    REQUIRE(L.links.image[l] == 1);
    const arma::mat v = site_pos(L, l);
    CHECK(v(0, 0) == 1000.0);
    const auto rays = quadriga_lib::generate_rays(L);
    CHECK(arma::approx_equal(rays[l][0].tx_pos, v, "absdiff", 1e-12));

    arma::uvec site;
    arma::mat g;
    CHECK(quadriga_lib::attach_terminals(L, "large_scale", {}, &site, &g)(0) == 1);
    CHECK(site(0) == 0);
    for (arma::uword i = 0; i < 2; ++i)
    {
        double az, el;
        direction(v, L.rx.pos.col(0), az, el);
        CHECK(std::abs(g(0, i) - (link_gain(L, l, 0) + 10.0 * std::log10(0.5 * element(el, az - L.tx.orientation(2, i))))) < tol_grid);
    }
    CHECK(quadriga_lib::attach_terminals(L, "rsrp", rays, nullptr, &g)(0) == 1);
    const auto ta = quadriga_lib::generate_antenna<double>("3gpp", NAN, 3.5e9);
    const auto ra = quadriga_lib::generate_antenna<double>("xpol", NAN, 3.5e9);
    for (arma::uword i = 0; i < 3; ++i)
    {
        const arma::uword m = link_of(L.links, site_of_node(i), 0);
        CHECK(std::abs(g(0, i) - rsrp_planar(ta, ra, L.tx.orientation.col(i), L.rx.orientation.col(0), rays[m][0], 0, link_gain(L, m, 0), 0)) < tol12);
    }
}

TEST_CASE("attach_terminals - a terminal with several panels takes its best panel, ties take the lowest node")
{
    // UT1 has panel 0 (omni) and panel 1 (one 3GPP element facing BS2); BS1 has two equal sectors (heading 0, the
    // generic antenna), which tie exactly for UT2 next to BS1
    quadriga_lib::layout L = two_sites(arma::mat{{200.0, 30.0}, {20.0, 10.0}, {1.5, 1.5}}, {3.5e9}, 0.0);
    L.rx = qd_layout_points({"UT1", "UT1", "UT2"}, {0, 1, 0}, arma::mat{{200.0, 200.0, 30.0}, {20.0, 20.0, 10.0}, {1.5, 1.5, 1.5}},
                            arma::vec{0.0, -0.06, 0.0});
    quadriga_lib::set_antenna(L, "omni", {}, true, {0, 2});
    quadriga_lib::set_antenna(L, "3gpp", {}, true, {1});
    quadriga_lib::init_link_states(L);
    arma::mat g;
    const arma::uvec srv = quadriga_lib::attach_terminals(L, "large_scale", {}, nullptr, &g);
    REQUIRE(g.n_rows == 2);
    for (arma::uword i = 0; i < 3; ++i)
    {
        const arma::uword l = link_of(L.links, site_of_node(i), 0);
        double az, el;
        direction(L.rx.pos.col(1), L.tx.pos.col(i), az, el);
        const double b = element(el, az - L.rx.orientation(2, 1)); // panel 1; panel 0 gives 1
        CHECK(std::abs(g(0, i) - (link_gain(L, l, 0) + 10.0 * std::log10(std::max(1.0, b)))) < (b > 1.0 ? tol_grid : tol12));
        CHECK((b > 1.0) == (i == 2)); // the element panel gives the value toward BS2
    }
    CHECK(g(1, 0) == g(1, 1)); // the same computation for both sectors: an exact tie
    CHECK(srv(1) == 0);
}

TEST_CASE("attach_terminals - pairing is ignored and the result stored as pairing links the serving sites")
{
    quadriga_lib::layout L = two_sites(uts().cols(0, 2));
    quadriga_lib::init_link_states(L);
    arma::uvec site;
    arma::mat g;
    const arma::uvec srv = quadriga_lib::attach_terminals(L, "large_scale", {}, &site, &g);
    for (const arma::umat &pairing : {arma::umat(arma::uvec{1, 0}), arma::umat(arma::uvec{5, 0}), arma::umat(arma::uvec{0, 0, 0})})
    {
        L.pairing = pairing; // a pair, an index outside the sites, a wrong row count
        arma::uvec site2;
        arma::mat g2;
        CHECK(qd_layout_same_int(quadriga_lib::attach_terminals(L, "large_scale", {}, &site2, &g2), srv));
        CHECK(qd_layout_same_int(site2, site));
        CHECK(qd_layout_same_mat(g2, g, 1e-12));
    }

    L.pairing = arma::join_cols(site.t(), arma::regspace<arma::urowvec>(0, 2));
    arma::umat idx;
    quadriga_lib::get_channels_from_rays(L, quadriga_lib::generate_rays(L), {}, {}, "taps", &idx);
    arma::uword n = 0;
    for (arma::uword t = 0; t < 3; ++t)
        n += site(t) == 0 ? 2 : 1;
    REQUIRE(idx.n_cols == n);
    for (arma::uword c = 0; c < idx.n_cols; ++c)
        CHECK(site_of_node(idx(0, c)) == site(idx(1, c)));
}

TEST_CASE("attach_terminals - an empty Rx side gives empty outputs")
{
    quadriga_lib::layout L = two_sites(uts().cols(0, 0));
    L.rx = quadriga_lib::layout_nodes();
    quadriga_lib::init_link_states(L);
    REQUIRE(L.links.n_link() == 0);
    arma::uvec site(2);
    arma::mat g(2, 2);
    CHECK(quadriga_lib::attach_terminals(L, "large_scale", {}, &site, &g).n_elem == 0);
    CHECK(site.n_elem == 0);
    CHECK(g.n_rows == 0);
    CHECK(g.n_cols == 3);
    CHECK(quadriga_lib::attach_terminals(L, "rsrp", {}, &site, &g).n_elem == 0);
}

TEST_CASE("attach_terminals - parallel and serial runs agree")
{
    // 3 x 12 links and 3 x 12 node pairs, above the size limits of the parallel loops
    arma::mat ut(3, 12);
    for (arma::uword i = 0; i < 12; ++i)
        ut(0, i) = -300.0 + 70.0 * (double)i, ut(1, i) = 150.0 * std::sin(1.3 * (double)i), ut(2, i) = 1.5;
    quadriga_lib::layout L = two_sites(ut);
    quadriga_lib::init_link_states(L);
    const auto rays = quadriga_lib::generate_rays(L);
    arma::mat a, b, c, d;
    const arma::uvec sa = quadriga_lib::attach_terminals(L, "large_scale", {}, nullptr, &a);
    const arma::uvec sc = quadriga_lib::attach_terminals(L, "rsrp", rays, nullptr, &c);
#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const arma::uvec sb = quadriga_lib::attach_terminals(L, "large_scale", {}, nullptr, &b);
    const arma::uvec sd = quadriga_lib::attach_terminals(L, "rsrp", rays, nullptr, &d);
#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#endif
    CHECK(qd_layout_same_int(sa, sb));
    CHECK(qd_layout_same_int(sc, sd));
    CHECK(arma::approx_equal(a, b, "absdiff", 1e-12));
    CHECK(arma::approx_equal(c, d, "absdiff", 1e-12));
}

TEST_CASE("attach_terminals - input validation")
{
    quadriga_lib::layout L = two_sites(uts().cols(0, 1), {3.5e9, 28.0e9});
    quadriga_lib::init_link_states(L);
    const auto rays = quadriga_lib::generate_rays(L);
    CHECK_NOTHROW(quadriga_lib::attach_terminals(L));
    CHECK_NOTHROW(quadriga_lib::attach_terminals(L, "rsrp", rays, nullptr, nullptr, 1, 1));
    auto msg = [](const quadriga_lib::layout &M, const std::string &rule, const rays_t &r, arma::uword k, arma::uword p)
    { return qd_layout_throws([&] { quadriga_lib::attach_terminals(M, rule, r, nullptr, nullptr, k, p); }); };
    SECTION("rule, frequency and port")
    {
        CHECK(msg(L, "RSRP", {}, 0, 0) == "Input 'rule' must be 'large_scale' or 'rsrp'.");
        CHECK(msg(L, "large_scale", {}, 2, 0) == "Input 'i_freq' must select one of the 2 frequencies of the layout.");
        CHECK(msg(L, "large_scale", {}, 0, 2) == "Input 'tx_port' must select one of the 2 ports of the antenna of Tx node 'BS1' (sector 0).");
    }
    SECTION("rays")
    {
        CHECK(msg(L, "rsrp", {}, 0, 0) == "Input 'rays' must hold the rays of all links of the layout (from 'generate_rays') for the rule 'rsrp'.");
        auto r = rays;
        r.pop_back();
        CHECK(msg(L, "rsrp", r, 0, 0) == "Input 'rays' must hold the rays of all links of the layout (from 'generate_rays') for the rule 'rsrp'.");
        r = rays;
        r[1].pop_back();
        r[1].push_back(r[1][0]); // frequency 0 in place of 1
        CHECK(msg(L, "rsrp", r, 0, 0) == "Entry 1: The frequencies of 'rays' do not match 'center_freq' of the layout.");
    }
    SECTION("links")
    {
        quadriga_lib::layout M = L;
        M.links.reset();
        CHECK(msg(M, "large_scale", {}, 0, 0) == "Call 'init_link_states' first.");
        M = L;
        M.rx.pos(0, 0) += 1.0;
        CHECK(msg(M, "large_scale", {}, 0, 0) == "The layout has changed since 'init_link_states': call 'init_link_states' again.");
        M = L;
        M.tx = quadriga_lib::layout_nodes(); // a default side: no nodes
        M.rx.scenario = {"UMa", "UMa"};
        quadriga_lib::init_link_states(M);
        CHECK(msg(M, "large_scale", {}, 0, 0) == "Input 'layout' has no Tx nodes.");
    }
}
