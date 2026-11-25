// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of export_rays_qrt: the file layout (origins, destinations, CIRs, local positions and the returned shift), the
// round trip through qrt_file_read, gbsm_rays_from_paths and gbsm_single_link_channel, pairing, given rays, parallel and
// serial runs, input validation.
// Oracles:
//  - the channels of stage 3 (get_gbsm_channels with output "rays" and mobility "drift", rays from generate_rays with
//    scatterers) at each row of the terminals: a CIR read back, shifted by the returned shift and evaluated with link
//    gain 0 at the read positions gives these channels path by path
//  - a moved UT: gbsm_single_link_channel with the rays of generate_rays, the link gain of the drop and the moved Rx
//    position; the link gain of the drop is read from the stage-3 channel at the drop (ratio of its power to that of
//    the same rays at 0 dB)
//  - the shift: the centroid of the site and CIR positions in x and y, brute force; positions and orientations of the
//    file from the rows of the layout
// Tolerances: the file stores float32. Path lengths 1e-7 times the sum of the absolute local coordinates of the points
// of a path (the bound of the QRT round-trip tests), which gives the delay bound (over c) and a phase bound (times
// 2 pi f / c); amplitudes 1e-5 relative to the largest entry of the path, for float32 Jones matrices and the angles
// from float32 scatterer positions through the 3GPP element pattern.
// Fixtures: qd_test_layout.hpp (work folder, point drops, message of a throw). Test layouts use 4 NLOS clusters of 4 rays.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_layout.hpp"
#include "qd_test_layout.hpp"
#include "qd_test_diag.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

namespace
{
using rays_t = std::vector<std::vector<quadriga_lib::gbsm_rays>>;
using chan_t = quadriga_lib::channel<double>;
const double deg = 0.017453292519943295, pi = 3.14159265358979323846, c0 = 299792458.0;

// Tx: site BS1 with sectors 0 and 1 at (0, 0, 25), site BS2 at (500, 0, 25), turned; one 3GPP element each. Rx: UT1 with
// two panels (headings 0 and 180 deg, the first tilted) at (120, 40, 1.5), UT2 a track of three rows from (300, -60, 1.5),
// UT3 at (-150, 80, 1.5); generic antennas. Scenarios: UT1 UMa_NLOS, UT2 UMa_LOS, UT3 UMa_NLOS_O2I; 3.5 GHz, seed 4711
quadriga_lib::layout make_layout(const arma::vec &fc = {3.5e9})
{
    quadriga_lib::layout L;
    L.tx = qd_layout_points({"BS1", "BS1", "BS2"}, {0, 1, 0}, arma::mat{{0.0, 0.0, 500.0}, {0.0, 0.0, 0.0}, {25.0, 25.0, 25.0}});
    L.tx.orientation = arma::mat{{0.05, 0.1, 0.3}, {-0.1, 0.15, -0.2}, {0.2, 180.0 * deg, 170.0 * deg}};
    L.tx.scenario = {"UMa", "UMa", "UMa"};
    for (const auto &kv : std::unordered_map<std::string, double>{{"NumClusters", 5.0}, {"NumSubPaths", 4.0}, {"NumClusters_min", 0.0}})
        L.tx.scenpar_override[kv.first] = arma::vec(3, arma::fill::value(kv.second));
    quadriga_lib::layout_nodes &R = L.rx;
    R.name = {"UT1", "UT1", "UT2", "UT3"};
    R.sector = {0, 1, 0, 0};
    R.i_antenna = {0, 0, 0, 0};
    R.offset = {0, 1, 2, 5};
    R.pos = arma::mat{{120.0, 40.0, 1.5}, {120.0, 40.0, 1.5}, {300.0, -60.0, 1.5}, {306.0, -60.0, 1.5}, {306.0, -52.0, 1.5}, {-150.0, 80.0, 1.5}}.t();
    R.orientation.zeros(3, 6);
    R.orientation(1, 0) = 0.2, R.orientation(2, 1) = 180.0 * deg, R.orientation(2, 2) = 0.4, R.orientation(0, 5) = -0.3;
    R.scenario = {"UMa_NLOS", "UMa_NLOS", "UMa_LOS", "UMa_LOS", "UMa_LOS", "UMa_NLOS_O2I"};
    L.center_freq = fc;
    L.seed = 4711;
    quadriga_lib::set_antenna(L, "3gpp");
    return L;
}

// The array of node i of a side at frequency k: its antenna, or the generic antenna
quadriga_lib::arrayant<double> array_of(const quadriga_lib::layout &L, const quadriga_lib::layout_nodes &N, arma::uword i, arma::uword k)
{
    if (N.i_antenna[i] == 0)
        return quadriga_lib::generate_antenna<double>("xpol", NAN, L.center_freq[k]);
    const auto &v = L.antenna[N.i_antenna[i] - 1];
    return v.size() == 1 ? v[0] : v[k];
}

// One CIR of a file, shifted back to the coordinates of the layout, as rays of 'method'; e_len = bound of the path
// lengths of the CIR from the float32 coordinates (largest over its paths)
struct cir_t
{
    quadriga_lib::gbsm_rays rays;
    arma::mat tx, rx, tx_o, rx_o;
    arma::uword n_path = 0;
    double e_len = 0.0;
};

cir_t read_cir(const std::string &fn, arma::uword i_orig, arma::uword i_cir, const arma::vec &shift, const std::string &method)
{
    cir_t c;
    arma::vec cf;
    std::vector<arma::mat> fbs, lbs, pg;
    std::vector<arma::vec> pl;
    std::vector<arma::cube> M;
    std::vector<arma::u32_vec> ni;
    quadriga_lib::qrt_file_read<double>(fn, {i_cir}, i_orig, false, &cf, &c.tx, &c.tx_o, &c.rx, &c.rx_o, &fbs, &lbs, &pg, &pl, &M, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, 1, &ni);
    c.n_path = pl[0].n_elem;
    for (arma::uword p = 0; p < c.n_path; ++p)
        c.e_len = std::max(c.e_len, 1e-7 * (arma::accu(arma::abs(c.tx)) + arma::accu(arma::abs(c.rx)) + arma::accu(arma::abs(fbs[0].col(p))) +
                                            arma::accu(arma::abs(lbs[0].col(p)))));
    for (arma::mat *X : {&c.tx, &c.rx, &fbs[0], &lbs[0]})
        if (X->n_cols > 0)
            X->row(0) += shift[0], X->row(1) += shift[1];
    if (c.n_path > 0)
        quadriga_lib::gbsm_rays_from_paths(cf, c.tx, c.rx, fbs[0], lbs[0], pg[0], pl[0], M[0], c.rays, method, ni[0]);
    return c;
}

// Error of the channel f (snapshot 0) against d (snapshot s), relative to the bounds: per path the largest coefficient
// error over |h|max (1e-5 + 2 pi fc / c e_len), the power (sum of |h|^2 over the ports) |P_f / P_d - 1| over 1e-4,
// and the delays over e_len / c; > 1 fails, Inf for unequal sizes. The power is not affected by the phase errors of
// the float32 positions; its bound covers the float32 path gains, Jones matrices and angles (largest value seen 1.3e-5)
double rel_error(const chan_t &f, const chan_t &d, arma::uword s, double fc, double e_len)
{
    const arma::cube &fr = f.coeff_re[0], &fi = f.coeff_im[0], &dr = d.coeff_re[s], &di = d.coeff_im[s];
    if (arma::size(fr) != arma::size(dr) || arma::size(f.delay[0]) != arma::size(d.delay[s]))
        return arma::datum::inf;
    double worst = 0.0;
    for (arma::uword p = 0; p < dr.n_slices; ++p)
    {
        double amax = 0.0, err = 0.0, pf = 0.0, pd = 0.0;
        for (arma::uword i = 0; i < dr.n_rows * dr.n_cols; ++i)
        {
            const arma::uword r = i % dr.n_rows, t = i / dr.n_rows;
            amax = std::max(amax, std::hypot(dr(r, t, p), di(r, t, p)));
            err = std::max(err, std::hypot(fr(r, t, p) - dr(r, t, p), fi(r, t, p) - di(r, t, p)));
            pf += fr(r, t, p) * fr(r, t, p) + fi(r, t, p) * fi(r, t, p), pd += dr(r, t, p) * dr(r, t, p) + di(r, t, p) * di(r, t, p);
        }
        if (amax > 0.0)
            worst = std::max({worst, err / (amax * (1e-5 + 2.0 * pi * fc / c0 * e_len)), std::abs(pf / pd - 1.0) / 1e-4});
    }
    const double e_tau = arma::abs(f.delay[0] - d.delay[s]).max();
    return std::max(worst, e_tau / (e_len / c0 + 1e-15));
}

// Channel of the CIR c for Tx node i and Rx node r of L at frequency k: link gain 0, the read positions
chan_t from_file(const quadriga_lib::layout &L, const cir_t &c, arma::uword i, arma::uword r, arma::uword k, const arma::mat &rx_pos = {})
{
    const arma::uword i_freq = c.rays.n_freq() == 1 ? 0 : k;
    return quadriga_lib::gbsm_single_link_channel(array_of(L, L.tx, i, k), array_of(L, L.rx, r, k), c.rays, arma::vec{0.0}, "drift", "rays", i_freq, {},
                                                  rx_pos, L.tx.orientation.col(L.tx.offset[i]), L.rx.orientation.col(L.rx.offset[r]));
}

// The first CIR of each terminal and the terminal of each Rx node of make_layout
const arma::uvec cir_first = {0, 1, 4}, term_of_rx = {0, 0, 1, 2}, site_of_tx = {0, 0, 1};

// Every CIR of the file against the stage-3 channels of L (from 'rays'); the largest relative error; counts the
// compared channels and the CIRs without paths of unpaired site-terminal pairs
double compare_all(const std::string &fn, const quadriga_lib::layout &L, const arma::vec &shift, const rays_t &rays, size_t &n_compared,
                   size_t &n_empty)
{
    arma::umat idx;
    const std::vector<chan_t> C = quadriga_lib::get_gbsm_channels(L, rays, {}, {}, "rays", &idx, "drift");
    double worst = 0.0;
    n_compared = 0, n_empty = 0;
    QD_DIAG("\n# diag: export_rays_qrt, file CIR vs stage 3 (error over the bound)\n");
    QD_DIAG("method,tx_node,rx_node,freq,row,n_path,e_len_m,rel_error\n");
    for (arma::uword s = 0; s < 2; ++s)
        for (arma::uword t = 0; t < 3; ++t)
        {
            const arma::uword n_row = t == 1 ? 3 : 1;
            for (arma::uword row = 0; row < n_row; ++row)
            {
                const cir_t c = read_cir(fn, s, cir_first[t] + row, shift, L.links.method);
                bool any = false;
                for (arma::uword j = 0; j < C.size(); ++j)
                {
                    const arma::uword i = idx(0, j), r = idx(1, j), k = idx(2, j);
                    if (site_of_tx[i] != s || term_of_rx[r] != t)
                        continue;
                    any = true;
                    REQUIRE(c.n_path > 0);
                    const double e = rel_error(from_file(L, c, i, r, k), C[j], row, L.center_freq[k], c.e_len);
                    QD_DIAG("%s,%d,%d,%d,%d,%d,%.3e,%.3e\n", L.links.method.c_str(), (int)i, (int)r, (int)k, (int)row, (int)c.n_path, c.e_len, e);
                    worst = std::max(worst, e);
                    ++n_compared;
                }
                if (!any)
                {
                    CHECK(c.n_path == 0);
                    ++n_empty;
                }
            }
        }
    return worst;
}

// All path data of all CIRs of origin o, for comparisons of two files
std::vector<arma::mat> file_data(const std::string &fn, arma::uword o)
{
    std::vector<arma::mat> fbs, lbs, pg;
    std::vector<arma::vec> pl;
    std::vector<arma::cube> M;
    quadriga_lib::qrt_file_read<double>(fn, {}, o, false, nullptr, nullptr, nullptr, nullptr, nullptr, &fbs, &lbs, &pg, &pl, &M);
    std::vector<arma::mat> out;
    for (size_t c = 0; c < fbs.size(); ++c)
        out.push_back(fbs[c]), out.push_back(lbs[c]), out.push_back(pg[c]), out.push_back(pl[c]), out.push_back(arma::vectorise(M[c]));
    return out;
}

// Two files hold the same path data within 1e-12 of the largest value of each member (rounding of other code paths)
bool same_data(const std::string &a, const std::string &b, arma::uword n_orig = 2)
{
    for (arma::uword o = 0; o < n_orig; ++o)
    {
        const std::vector<arma::mat> x = file_data(a, o), y = file_data(b, o);
        if (x.size() != y.size())
            return false;
        for (size_t i = 0; i < x.size(); ++i)
            if (!qd_layout_same_data(x[i], y[i], 1e-12))
                return false;
    }
    return true;
}
} // namespace

TEST_CASE("export_rays_qrt - the file has one origin per site, one destination per terminal and local positions")
{
    quadriga_lib::layout L = make_layout();
    quadriga_lib::init_link_states(L);
    const qd_layout_files F("export_rays_qrt_layout");
    const std::string fn = F.path("a.qrt");
    const arma::vec shift = quadriga_lib::export_rays_qrt(fn, L);

    // The shift: centroid of the 2 sites and the 5 CIR positions (rows of the first nodes of the terminals)
    const arma::uvec cir_rows = {0, 2, 3, 4, 5};
    arma::mat P = arma::join_rows(L.tx.pos.cols(arma::uvec{0, 2}), L.rx.pos.cols(cir_rows));
    REQUIRE(shift.n_elem == 2);
    CHECK(std::abs(shift[0] - arma::mean(P.row(0))) < 1e-12);
    CHECK(std::abs(shift[1] - arma::mean(P.row(1))) < 1e-12);

    arma::uword n_cir = 0, n_orig = 0, n_dest = 0, n_freq = 0;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    arma::fvec cf;
    arma::fmat cir_pos, cir_ori, orig_pos, orig_ori;
    quadriga_lib::qrt_file_parse(fn, &n_cir, &n_orig, &n_dest, &n_freq, &cir_offset, &orig_names, &dest_names, nullptr, &cf, &cir_pos, &cir_ori,
                                 &orig_pos, &orig_ori);
    CHECK(n_cir == 5);
    CHECK(n_orig == 2);
    CHECK(n_dest == 3);
    CHECK(n_freq == 1);
    CHECK(qd_layout_same_int(cir_offset, cir_first));
    CHECK(orig_names == std::vector<std::string>{"BS1", "BS2"});
    CHECK(dest_names == std::vector<std::string>{"UT1", "UT2", "UT3"});
    REQUIRE(cf.n_elem == 1);
    CHECK(std::abs((double)cf[0] - 3.5e9) < 1e-6 * 3.5e9);

    // Local positions (float32: 1e-7 relative to 500 m), z unchanged; orientations of the first node of each terminal
    // and of sector 0 of each site
    arma::mat local = P;
    local.row(0) -= shift[0], local.row(1) -= shift[1];
    const arma::mat cp = arma::conv_to<arma::mat>::from(cir_pos).t(), op = arma::conv_to<arma::mat>::from(orig_pos).t();
    CHECK(arma::approx_equal(cp, local.cols(2, 6), "absdiff", 1e-4));
    CHECK(arma::approx_equal(op, local.cols(0, 1), "absdiff", 1e-4));
    CHECK(arma::approx_equal(arma::conv_to<arma::mat>::from(cir_ori).t(), L.rx.orientation.cols(cir_rows), "absdiff", 1e-6));
    CHECK(arma::approx_equal(arma::conv_to<arma::mat>::from(orig_ori).t(), L.tx.orientation.cols(arma::uvec{0, 2}), "absdiff", 1e-6));

    // A site whose sector 0 is not its first node: the orientation of sector 0
    quadriga_lib::layout M = L;
    M.tx.sector = {1, 0, 0};
    quadriga_lib::init_link_states(M);
    quadriga_lib::export_rays_qrt(fn, M);
    quadriga_lib::qrt_file_parse(fn, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 &orig_ori);
    CHECK(arma::approx_equal(arma::conv_to<arma::mat>::from(orig_ori).t(), L.tx.orientation.cols(arma::uvec{1, 2}), "absdiff", 1e-6));
}

TEST_CASE("export_rays_qrt - the CIRs read back give the channels of stage 3 at the rows of the terminals")
{
    // UT1 NLOS with two panels, UT2 a LOS track of three rows (link gain and K-factor change along it), UT3 in a building
    quadriga_lib::layout L = make_layout();
    quadriga_lib::init_link_states(L);
    REQUIRE(L.links.n_link() == 6);
    const qd_layout_files F("export_rays_qrt_round_trip");
    const std::string fn = F.path("a.qrt");
    const arma::vec shift = quadriga_lib::export_rays_qrt(fn, L);
    size_t n_compared = 0, n_empty = 0;
    CHECK(compare_all(fn, L, shift, quadriga_lib::generate_rays(L, {}, true), n_compared, n_empty) <= 1.0);
    CHECK(n_compared == 3 * 2 + 3 * 3 * 1 + 3 * 1); // UT1: 2 panels x 3 Tx nodes; UT2: 3 rows x 3; UT3: 3
    CHECK(n_empty == 0);
}

TEST_CASE("export_rays_qrt - a moved UT from a file has the channel of its rays at the link gain of the drop")
{
    // quadriga, two frequencies; UT1 a point drop with speed: one CIR, its motion applied at the evaluation
    quadriga_lib::layout L = make_layout({3.5e9, 28.0e9});
    L.rx.speed = {1.5, 1.5, 0.0, 0.0, 0.0, 0.0};
    L.rx.direction = arma::mat(2, 6, arma::fill::zeros);
    L.rx.direction.row(0).fill(arma::datum::nan);
    L.rx.direction(0, 0) = L.rx.direction(0, 1) = 30.0 * deg;
    quadriga_lib::init_link_states(L, "quadriga");
    const qd_layout_files F("export_rays_qrt_moved");
    const std::string fn = F.path("a.qrt");
    const arma::vec shift = quadriga_lib::export_rays_qrt(fn, L);
    const rays_t rays = quadriga_lib::generate_rays(L);

    // At the drop: the channels of stage 3 for both frequencies
    size_t n_compared = 0, n_empty = 0;
    CHECK(compare_all(fn, L, shift, rays, n_compared, n_empty) <= 1.0);
    CHECK(n_compared == 2 * (3 * 2 + 3 * 3 + 3));

    // Moved by 0.5 m and 2 m: the rays of the link with the link gain of the drop
    QD_DIAG("\n# diag: export_rays_qrt, moved UT from the file vs the rays at the drop link gain\n");
    QD_DIAG("tx_node,rx_node,freq,distance_m,link_gain_dB,rel_error\n");
    arma::umat idx;
    const std::vector<chan_t> C = quadriga_lib::get_gbsm_channels(L, rays, {}, {}, "rays", &idx, "drift");
    size_t n_moved = 0;
    for (arma::uword j = 0; j < C.size(); ++j)
    {
        const arma::uword i = idx(0, j), r = idx(1, j), k = idx(2, j);
        if (r > 1)
            continue;
        const arma::uword l = site_of_tx[i] == 0 ? 0 : 1; // links of UT1: terminal-major, BS1 then BS2
        REQUIRE(L.links.i_rx[l] == 0);
        REQUIRE(L.links.i_tx[l] == site_of_tx[i]);
        const auto ta = array_of(L, L.tx, i, k), ra = array_of(L, L.rx, r, k);
        const arma::mat to = L.tx.orientation.col(i), ro = L.rx.orientation.col(L.rx.offset[r]);
        const chan_t u = quadriga_lib::gbsm_single_link_channel(ta, ra, rays[l][0], arma::vec{0.0}, "drift", "rays", k, {}, {}, to, ro);
        double p3 = 0.0, pu = 0.0;
        for (arma::uword p = 0; p < C[j].coeff_re[0].n_elem; ++p)
            p3 += std::pow(C[j].coeff_re[0][p], 2) + std::pow(C[j].coeff_im[0][p], 2),
                pu += std::pow(u.coeff_re[0][p], 2) + std::pow(u.coeff_im[0][p], 2);
        const double lg = 10.0 * std::log10(p3 / pu);
        const cir_t c = read_cir(fn, site_of_tx[i], 0, shift, "quadriga");
        for (double d : {0.5, 2.0})
        {
            const arma::mat rx = L.rx.pos.col(0) + d * arma::vec{std::cos(30.0 * deg), std::sin(30.0 * deg), 0.0};
            const chan_t o = quadriga_lib::gbsm_single_link_channel(ta, ra, rays[l][0], arma::vec{lg}, "drift", "rays", k, {}, rx, to, ro);
            const double e = rel_error(from_file(L, c, i, r, k, rx), o, 0, L.center_freq[k], c.e_len);
            QD_DIAG("%d,%d,%d,%.1f,%.6f,%.3e\n", (int)i, (int)r, (int)k, d, lg, e);
            CHECK(e <= 1.0);
            ++n_moved;
        }
    }
    CHECK(n_moved == 2 * 3 * 2 * 2);
}

TEST_CASE("export_rays_qrt - a ground reflection is kept at the drop, moving it from a file throws")
{
    // The file stores no ground permittivity: the GR path of a CIR is exact at its position and cannot be recomputed
    quadriga_lib::layout L = make_layout();
    L.tx.scenpar_override["GR_enabled"] = arma::vec(3, arma::fill::ones);
    quadriga_lib::init_link_states(L);
    const qd_layout_files F("export_rays_qrt_gr");
    const std::string fn = F.path("a.qrt");
    const arma::vec shift = quadriga_lib::export_rays_qrt(fn, L);
    const rays_t rays = quadriga_lib::generate_rays(L, {}, true);
    REQUIRE(arma::accu(rays[0][0].path_type == 1) == 1);
    size_t n_compared = 0, n_empty = 0;
    CHECK(compare_all(fn, L, shift, rays, n_compared, n_empty) <= 1.0);
    CHECK(n_compared == 18);
    const cir_t c = read_cir(fn, 0, 0, shift, "38.901");
    CHECK(arma::accu(c.rays.path_type == 1) == 1);
    const std::string m = qd_layout_throws([&] { from_file(L, c, 0, 0, 0, c.rx + arma::vec{1.0, 0.0, 0.0}); });
    INFO(m);
    CHECK(m == "Input 'rays.gr_eps_re' is required to recompute the ground reflection away from the positions of 'rays'.");
}

TEST_CASE("export_rays_qrt - pairing exports the paired pairs, other CIRs of an origin are empty")
{
    quadriga_lib::layout L = make_layout();
    L.pairing = {{0, 1, 0, 1}, {0, 1, 2, 2}}; // BS1-UT1, BS2-UT2, both sites with UT3
    quadriga_lib::init_link_states(L);
    const qd_layout_files F("export_rays_qrt_pairing");
    const std::string fn = F.path("a.qrt");
    const arma::vec shift = quadriga_lib::export_rays_qrt(fn, L);
    size_t n_compared = 0, n_empty = 0;
    CHECK(compare_all(fn, L, shift, quadriga_lib::generate_rays(L, {}, true), n_compared, n_empty) <= 1.0);
    CHECK(n_compared == 2 * 2 + 3 + 3); // BS1 (2 sectors) with the 2 panels of UT1, BS2 with 3 rows of UT2, 3 Tx nodes with UT3
    CHECK(n_empty == 3 + 1); // BS1: the 3 rows of UT2; BS2: UT1

    // A site without paired terminals, first in the order: its origin holds only empty CIRs
    L.pairing = {{1, 1, 1}, {0, 1, 2}};
    quadriga_lib::init_link_states(L);
    const arma::vec s2 = quadriga_lib::export_rays_qrt(fn, L);
    CHECK(compare_all(fn, L, s2, quadriga_lib::generate_rays(L, {}, true), n_compared, n_empty) <= 1.0);
    CHECK(n_compared == 2 + 3 + 1);
    CHECK(n_empty == 5);
}

TEST_CASE("export_rays_qrt - given rays with or without scatterers give the file of the computed rays")
{
    quadriga_lib::layout L = make_layout();
    quadriga_lib::init_link_states(L);
    const qd_layout_files F("export_rays_qrt_given");
    const std::string a = F.path("a.qrt"), b = F.path("b.qrt"), c = F.path("c.qrt");
    const arma::vec s = quadriga_lib::export_rays_qrt(a, L);
    CHECK(qd_layout_same_data(quadriga_lib::export_rays_qrt(b, L, quadriga_lib::generate_rays(L, {}, true)), s, 1e-12));
    CHECK(qd_layout_same_data(quadriga_lib::export_rays_qrt(c, L, quadriga_lib::generate_rays(L)), s, 1e-12));
    CHECK(same_data(a, b));
    CHECK(same_data(a, c));

    // Edited rays are what the file holds: the path gains of one link scaled by 4
    rays_t r = quadriga_lib::generate_rays(L, {}, true);
    r[0][0].power *= 4.0;
    quadriga_lib::export_rays_qrt(b, L, r);
    std::vector<arma::mat> pa, pb;
    quadriga_lib::qrt_file_read<double>(a, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &pa);
    quadriga_lib::qrt_file_read<double>(b, {0}, 0, false, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &pb);
    CHECK(arma::approx_equal(pb[0], 4.0 * pa[0], "reldiff", 1e-6));
}

TEST_CASE("export_rays_qrt - sites in several batches of generated rays give the file of the given rays")
{
    // 2 sites with 130 terminals each: 260 links, more than the 256 of one batch, so each site is a batch of its own
    quadriga_lib::layout L = make_layout();
    arma::mat ut(3, 130);
    for (arma::uword i = 0; i < 130; ++i)
        ut(0, i) = -200.0 + 6.0 * (double)i, ut(1, i) = 60.0 + 150.0 * std::sin(0.7 * (double)i), ut(2, i) = 1.5;
    std::vector<std::string> names;
    for (arma::uword i = 0; i < 130; ++i)
        names.push_back("UT" + std::to_string(i));
    L.rx = qd_layout_points(names, std::vector<unsigned>(130, 0), ut);
    L.rx.scenario.assign(130, "UMa_NLOS");
    quadriga_lib::init_link_states(L);
    REQUIRE(L.links.n_link() == 260);
    const qd_layout_files F("export_rays_qrt_batches");
    const std::string a = F.path("a.qrt"), b = F.path("b.qrt");
    const arma::vec s = quadriga_lib::export_rays_qrt(a, L);
    CHECK(qd_layout_same_data(quadriga_lib::export_rays_qrt(b, L, quadriga_lib::generate_rays(L, {}, true)), s, 1e-12));
    CHECK(same_data(a, b));
}

TEST_CASE("export_rays_qrt - parallel and serial runs agree")
{
    // 9 terminals: 9 links per site, above the size limit of the parallel loop over the links of a site
    quadriga_lib::layout L = make_layout();
    arma::mat ut(3, 9);
    std::vector<std::string> names;
    for (arma::uword i = 0; i < 9; ++i)
        ut(0, i) = -100.0 + 70.0 * (double)i, ut(1, i) = 80.0 * std::cos(1.1 * (double)i), ut(2, i) = 1.5, names.push_back("UT" + std::to_string(i));
    L.rx = qd_layout_points(names, std::vector<unsigned>(9, 0), ut);
    L.rx.scenario.assign(9, "UMa_NLOS");
    quadriga_lib::init_link_states(L);
    const qd_layout_files F("export_rays_qrt_parallel");
    const std::string a = F.path("a.qrt"), b = F.path("b.qrt");
    quadriga_lib::export_rays_qrt(a, L);
#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    quadriga_lib::export_rays_qrt(b, L);
#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#endif
    CHECK(same_data(a, b));
}

TEST_CASE("export_rays_qrt - input validation")
{
    quadriga_lib::layout L = make_layout();
    quadriga_lib::init_link_states(L);
    const qd_layout_files F("export_rays_qrt_validation");
    const std::string old = F.write("old.qrt", "kept");
    auto msg = [&](const quadriga_lib::layout &M, const rays_t &r = {})
    {
        const std::string m = qd_layout_throws([&] { quadriga_lib::export_rays_qrt(old, M, r); });
        CHECK(F.read("old.qrt") == "kept"); // a failed call leaves an existing file unchanged
        return m;
    };
    SECTION("valid input")
    {
        CHECK_NOTHROW(quadriga_lib::export_rays_qrt(F.path("a.qrt"), L));
    }
    SECTION("links")
    {
        quadriga_lib::layout M = L;
        M.links.reset();
        CHECK(msg(M) == "Call 'init_link_states' first.");
        M = L;
        M.rx.pos(0, 5) += 1.0;
        CHECK(msg(M) == "The layout has changed since 'init_link_states': call it again, or 'links.reset()' to discard the links.");
    }
    SECTION("frequencies")
    {
        quadriga_lib::layout M = make_layout({3.5e9, 28.0e9});
        quadriga_lib::init_link_states(M);
        CHECK(msg(M) == "The method '38.901' exports one frequency, but 'center_freq' of the layout has 2; use one layout per frequency, or 'quadriga'.");
        M = make_layout(arma::linspace(1.0e9, 128.0e9, 128));
        quadriga_lib::init_link_states(M, "quadriga");
        CHECK(msg(M) == "A QRT file holds at most 127 frequencies, but 'center_freq' of the layout has 128.");
    }
    SECTION("moving Tx nodes")
    {
        quadriga_lib::layout M = L;
        M.tx.speed = {0.0, 0.0, 1.0};
        quadriga_lib::init_link_states(M);
        CHECK(msg(M) == "Tx node 'BS2' (sector 0): a QRT origin has one position; it must have one row and no speed.");
        M = L;
        M.tx.pos = arma::join_rows(M.tx.pos, arma::vec{510.0, 0.0, 25.0});
        M.tx.orientation = arma::join_rows(M.tx.orientation, M.tx.orientation.col(2));
        M.tx.scenario.push_back("UMa");
        quadriga_lib::init_link_states(M);
        CHECK(msg(M) == "Tx node 'BS2' (sector 0): a QRT origin has one position; it must have one row and no speed.");
    }
    SECTION("pairing")
    {
        quadriga_lib::layout M = L;
        M.pairing = {{0, 1}, {0, 2}};
        CHECK(msg(M) == "Terminal 'UT2' has no site in 'pairing': each QRT destination needs the CIRs of at least one origin.");
    }
    SECTION("wrap-around")
    {
        quadriga_lib::layout M = L;
        M.wrap = arma::mat(2, 1, arma::fill::zeros);
        M.wrap(0, 0) = 2000.0;
        quadriga_lib::init_link_states(M);
        CHECK(msg(M) == "Input 'layout' has wrap-around, which a QRT file cannot hold (an origin has one position, the wrap images of a site differ "
                        "per terminal); clear 'wrap' and call 'init_link_states' again.");
    }
    SECTION("terminal at a site position")
    {
        quadriga_lib::layout M = L;
        M.rx.pos.col(5) = arma::vec{500.0, 0.0, 25.0}; // UT3 at BS2: no link
        quadriga_lib::init_link_states(M);
        CHECK(msg(M) == "Terminal 'UT3' has a row at the position of site 'BS2'; a CIR needs a LOS path of non-zero length.");
        M = L;
        M.rx.pos.col(4) = arma::vec{500.0, 0.0, 25.0}; // the last row of the track of UT2
        quadriga_lib::init_link_states(M);
        CHECK(msg(M) == "Terminal 'UT2' has a row at the position of site 'BS2'; a CIR needs a LOS path of non-zero length.");
    }
    SECTION("rays")
    {
        rays_t r = quadriga_lib::generate_rays(L);
        r.pop_back();
        CHECK(msg(L, r) == "Input 'rays' must be empty or hold the rays of all links of the layout (from 'generate_rays').");
        quadriga_lib::layout M = make_layout({3.5e9, 28.0e9});
        quadriga_lib::init_link_states(M, "quadriga");
        r = quadriga_lib::generate_rays(M);
        r[2] = {r[2][0], r[2][0]};
        CHECK(msg(M, r) == "Entry 2: Input 'rays' must hold one object with all frequencies per link.");
        r = quadriga_lib::generate_rays(M);
        r[1][0].center_freq[1] = 30.0e9;
        CHECK(msg(M, r) == "Entry 1: The frequencies of 'rays' do not match 'center_freq' of the layout.");
    }
    SECTION("sides")
    {
        quadriga_lib::layout M = L;
        M.rx = quadriga_lib::layout_nodes();
        quadriga_lib::init_link_states(M);
        CHECK(msg(M) == "Input 'layout' has no Rx nodes.");
        M = L;
        M.tx = quadriga_lib::layout_nodes();
        M.rx.scenario.assign(6, "UMa");
        quadriga_lib::init_link_states(M);
        CHECK(msg(M) == "Input 'layout' has no Tx nodes.");
    }
    SECTION("file")
    {
        CHECK(qd_layout_throws([&] { quadriga_lib::export_rays_qrt(F.path("no_folder/a.qrt"), L); }).find("no_folder/a.qrt") != std::string::npos);
        CHECK_THROWS_AS(quadriga_lib::export_rays_qrt("", L), std::invalid_argument);
        CHECK(qd_layout_throws([&] { quadriga_lib::export_rays_qrt("", L); }) == "Input 'fn' must not be empty.");
        CHECK_THROWS_AS(quadriga_lib::export_rays_qrt(F.dir, L), std::invalid_argument);
        CHECK(qd_layout_throws([&] { quadriga_lib::export_rays_qrt(F.dir, L); }) == "Input 'fn' is a folder; give the name of a file.");
    }
}
