// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of crossfade_segments on small synthetic "rays" channels (2 x 2 element pairs, paths with constant powers and
// delays, phases that turn from snapshot to snapshot, tap indices of ssf_rays: LOS 0, GR 3, NLOS from 6) and on TR
// channels of single-segment layouts. Oracles: the ramp w = 0.5 (1 + sin((x - 0.5) pi)), x = i / (n_r + 1), on the
// n_r = n_overlap shared snapshots, the level P_old^(1 - w) P_new^w, and the bounds of the rule (no cluster more than
// 10 dB above its own power, monotone exchange, both output forms from the same weighted paths).

#include <catch2/catch_test_macros.hpp>

#include "quadriga_channel.hpp"
#include "quadriga_layout.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace
{
using ch_t = quadriga_lib::channel<double>;
const double pi = 3.14159265358979323846;

// A channel of n snapshots with the paths 'tap' of power 'pow' per element pair and delay 'dly'; the phase of path p at
// element pair e and snapshot s is ph + 0.3 e + 0.1 (p + 1) s; Rx at x = x0 + s, Tx fixed; rays: path data, else taps
ch_t synth(const arma::u32_vec &tap, const arma::vec &pow, const arma::vec &dly, arma::uword n, bool rays, double ph = 0.0, double x0 = 0.0)
{
    ch_t c;
    const arma::uword np = tap.n_elem;
    c.name = "seg";
    c.center_freq = {3.5e9};
    c.tx_pos = arma::mat(arma::vec{0.0, 0.0, 25.0});
    c.rx_pos.set_size(3, n);
    c.tx_orientation.zeros(3, 1);
    c.rx_orientation.zeros(3, n);
    for (arma::uword s = 0; s < n; ++s)
    {
        c.rx_pos(0, s) = x0 + (double)s, c.rx_pos(1, s) = 10.0, c.rx_pos(2, s) = 1.5;
        c.rx_orientation(2, s) = 0.01 * (double)s;
        arma::cube re(2, 2, np), im(2, 2, np), dl(1, 1, np);
        arma::vec pg(np), len(np);
        arma::mat pm(8, np, arma::fill::zeros), ang(np, 4), fbs(3, np), lbs(3, np);
        for (arma::uword p = 0; p < np; ++p)
        {
            for (arma::uword e = 0; e < 4; ++e)
            {
                const double a = ph + 0.3 * (double)e + 0.1 * (double)(p + 1) * (double)s;
                re(e % 2, e / 2, p) = std::sqrt(pow(p)) * std::cos(a), im(e % 2, e / 2, p) = std::sqrt(pow(p)) * std::sin(a);
            }
            dl(0, 0, p) = dly(p), pg(p) = pow(p), len(p) = dly(p) * 3.0e8;
            pm(0, p) = 1.0, pm(6, p) = -1.0;
            ang(p, 0) = 0.1 * (double)p, ang(p, 1) = 0.0, ang(p, 2) = -0.1 * (double)p, ang(p, 3) = 0.0;
            fbs.col(p).fill((double)p), lbs.col(p).fill((double)p + 0.5);
        }
        c.coeff_re.push_back(re), c.coeff_im.push_back(im), c.delay.push_back(dl);
        if (rays)
            c.path_gain.push_back(pg), c.path_length.push_back(len), c.polmat.push_back(pm), c.path_angles.push_back(ang), c.fbs_pos.push_back(fbs),
                c.lbs_pos.push_back(lbs);
    }
    c.par_names = {"tap"}, c.par_data = {tap};
    return c;
}

// Snapshots [a, b) of a channel
ch_t cut(const ch_t &c, arma::uword a, arma::uword b)
{
    ch_t x = c;
    for (arma::mat *m : {&x.rx_pos, &x.rx_orientation, &x.tx_pos, &x.tx_orientation})
        if (m->n_cols == c.n_snap())
            *m = m->cols(a, b - 1);
    auto sub = [a, b](auto &v)
    {
        if (!v.empty())
            v.assign(v.begin() + (std::ptrdiff_t)a, v.begin() + (std::ptrdiff_t)b);
    };
    sub(x.coeff_re), sub(x.coeff_im), sub(x.delay), sub(x.path_gain), sub(x.path_length), sub(x.polmat), sub(x.path_angles), sub(x.fbs_pos), sub(x.lbs_pos);
    return x;
}

// Total power of snapshot s over paths and element pairs
double total(const ch_t &c, arma::uword s) { return arma::accu(arma::square(c.coeff_re[s]) + arma::square(c.coeff_im[s])); }

// The documented ramp
std::vector<double> ramp(arma::uword n_r)
{
    std::vector<double> w;
    for (arma::uword i = 1; i <= n_r; ++i)
        w.push_back(0.5 * (1.0 + std::sin(((double)i / (double)(n_r + 1) - 0.5) * pi)));
    return w;
}

// Two different segments: LOS + 3 NLOS (old) and LOS + 4 NLOS (new), 2 x 2 pairs; 'los_new' is the new LOS power
ch_t seg_old(arma::uword n, bool rays) { return synth({0, 6, 9, 12}, {1.0, 0.3, 0.2, 0.1}, {100e-9, 150e-9, 220e-9, 400e-9}, n, rays, 0.0, 0.0); }
ch_t seg_new(arma::uword n, bool rays, double los_new = 0.5, double x0 = 6.0)
{
    return synth({0, 6, 9, 12, 15}, {los_new, 0.05, 0.4, 0.25, 0.15}, {100e-9, 120e-9, 300e-9, 500e-9, 700e-9}, n, rays, 1.0, x0);
}

// Sum of the path gains of the paths with tap index 0 (los = true) or the others
double gain_of(const ch_t &c, arma::uword s, const arma::u32_vec &tap, bool los)
{
    double g = 0.0;
    for (arma::uword p = 0; p < tap.n_elem; ++p)
        if ((tap(p) == 0) == los)
            g += c.path_gain[s](p);
    return g;
}
} // namespace

namespace
{
// Path gain sum of snapshot s
double pg_sum(const ch_t &c, arma::uword s) { return arma::accu(c.path_gain[s]); }

// "rays" or "taps" channel of a single-segment layout: one BS at 25 m, a track of n rows 0.1 m apart from x = 100 m
ch_t tr_channel(const char *scen, arma::sword seed, arma::uword n, const char *output, std::map<std::string, double> over = {})
{
    quadriga_lib::layout L;
    L.tx.name = {"BS1"}, L.tx.sector = {0}, L.tx.i_antenna = {0}, L.tx.offset = {0};
    L.tx.pos = arma::mat(arma::vec{0.0, 0.0, 25.0}), L.tx.orientation.zeros(3, 1), L.tx.scenario = {scen};
    for (const auto &kv : over)
        L.tx.scenpar_override[kv.first] = arma::vec{kv.second};
    L.rx.name = {"UT1"}, L.rx.sector = {0}, L.rx.i_antenna = {0}, L.rx.offset = {0};
    L.rx.pos.set_size(3, n), L.rx.orientation.zeros(3, n);
    for (arma::uword i = 0; i < n; ++i)
        L.rx.pos(0, i) = 100.0 + 0.1 * (double)i, L.rx.pos(1, i) = 20.0, L.rx.pos(2, i) = 1.5;
    L.center_freq = {3.5e9}, L.seed = seed;
    quadriga_lib::init_link_states(L);
    return quadriga_lib::get_channels_from_rays(L, quadriga_lib::generate_rays(L), {}, {}, output)[0];
}

// An entry of par_data
arma::u32_vec entry(const ch_t &c, const std::string &name)
{
    for (size_t i = 0; i < c.par_names.size(); ++i)
        if (c.par_names[i] == name)
            return std::any_cast<arma::u32_vec>(c.par_data[i]);
    return arma::u32_vec();
}

// Power weight of each output path of ramp snapshot s relative to its source path (matched by delay and angles) in the
// old segment A at snapshot ia or the new one B at ib; cluster = tap / 3 of the source, side 0 old, 1 new
struct weight_t
{
    double w;
    unsigned cluster, side, type;
};
std::vector<weight_t> weights(const ch_t &c, arma::uword s, const ch_t &A, arma::uword ia, const ch_t &B, arma::uword ib)
{
    std::vector<weight_t> v;
    for (arma::uword p = 0; p < c.path_gain[s].n_elem; ++p)
    {
        bool found = false;
        for (unsigned side = 0; side < 2 && !found; ++side)
        {
            const ch_t &X = side == 0 ? A : B;
            const arma::uword i = side == 0 ? ia : ib;
            const arma::u32_vec tap = entry(X, "tap"), type = entry(X, "path_type");
            for (arma::uword q = 0; q < X.path_gain[i].n_elem && !found; ++q)
                if (X.delay[i](0, 0, q) == c.delay[s](0, 0, p) && arma::approx_equal(X.path_angles[i].row(q), c.path_angles[s].row(p), "absdiff", 0.0))
                    found = true, v.push_back({c.path_gain[s][p] / X.path_gain[i][q], tap[q] / 3, side, type[q]});
        }
        REQUIRE(found);
    }
    return v;
}
} // namespace

TEST_CASE("crossfade_segments - one segment reproduces the input, its taps are those of gbsm_single_link_channel")
{
    const ch_t X = tr_channel("UMa_LOS", 4711, 6, "rays"), T = tr_channel("UMa_LOS", 4711, 6, "taps");
    const ch_t r = quadriga_lib::crossfade_segments({X}, arma::uvec{0}, "rays"), t = quadriga_lib::crossfade_segments({X}, arma::uvec{0});
    REQUIRE(r.n_snap() == 6);
    REQUIRE(t.n_snap() == 6);
    for (arma::uword s = 0; s < 6; ++s)
    {
        CHECK(arma::approx_equal(r.coeff_re[s], X.coeff_re[s], "absdiff", 0.0));
        CHECK(arma::approx_equal(r.path_gain[s], X.path_gain[s], "absdiff", 0.0));
        REQUIRE(arma::size(t.coeff_re[s]) == arma::size(T.coeff_re[s]));
        const double m = arma::abs(T.coeff_re[s]).max() + arma::abs(T.coeff_im[s]).max();
        CHECK(arma::approx_equal(t.coeff_re[s], T.coeff_re[s], "absdiff", 1e-12 * m));
        CHECK(arma::approx_equal(t.coeff_im[s], T.coeff_im[s], "absdiff", 1e-12 * m));
        CHECK(arma::approx_equal(t.delay[s], T.delay[s], "absdiff", 1e-18));
    }
}

TEST_CASE("crossfade_segments - the level in the ramp is interpolated in dB")
{
    // UMa_LOS (old) and UMa_NLOS (new) of single-segment layouts over the same track of 200 rows; rows 50-99 are the ramp:
    // at every ramp snapshot the summed path gain is P_old^(1 - w) P_new^w, with one LOS path
    const ch_t a = tr_channel("UMa_LOS", 4711, 200, "rays"), b = tr_channel("UMa_NLOS", 4711, 200, "rays");
    const ch_t A = cut(a, 0, 100), B = cut(b, 50, 200);
    const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 50}, "rays");
    REQUIRE(c.n_snap() == 200);
    const std::vector<double> w = ramp(50);
    double spread = 0.0;
    for (arma::uword r = 0; r < 50; ++r)
    {
        const arma::uword s = 50 + r;
        const double target = std::pow(pg_sum(A, s), 1.0 - w[r]) * std::pow(pg_sum(B, r), w[r]);
        spread = std::max(spread, std::abs(10.0 * std::log10(pg_sum(A, s) / pg_sum(B, r))));
        INFO("ramp snapshot " << r);
        CHECK(std::abs(pg_sum(c, s) - target) <= 1e-9 * target);
        arma::uword n_los = 0;
        for (const weight_t &x : weights(c, s, A, s, B, r))
            n_los += x.type == 0 ? 1 : 0;
        CHECK(n_los == 1);
    }
    CHECK(spread > 3.0);
    for (arma::uword s : {0, 49, 100, 199}) // outside the ramp: one segment
        CHECK(pg_sum(c, s) == (s < 50 ? pg_sum(A, s) : pg_sum(B, s - 50)));
}

TEST_CASE("crossfade_segments - clusters are exchanged one after another within 10 dB of their own power")
{
    // TR transitions LOS to NLOS and NLOS to O2I (levels 9 and 17 dB apart). Per cluster (tap / 3), relative to its own
    // power: one weight for all its paths, at most 10 dB (measured on 8 such transitions: up to 7.8 dB); the present part
    // (weight times P_own / G) falls for the old and rises for the new clusters. The taps above -25 dB of the total stay
    // at least those of the weaker-structured segment (measured: 1.0 to 1.06 of the smaller count), one less allowed
    for (int pair = 0; pair < 2; ++pair)
    {
        INFO("pair " << pair);
        const char *s_old = pair == 0 ? "UMa_LOS" : "UMa_NLOS", *s_new = pair == 0 ? "UMa_NLOS" : "UMa_NLOS_O2I";
        const ch_t A = cut(tr_channel(s_old, 4711, 100, "rays"), 0, 100), B = cut(tr_channel(s_new, 4718, 200, "rays"), 50, 200);
        const ch_t TA = tr_channel(s_old, 4711, 100, "taps"), TB = cut(tr_channel(s_new, 4718, 200, "taps"), 50, 200);
        const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 50}, "rays"), t = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 50});
        const std::vector<double> w = ramp(50);
        std::map<std::pair<unsigned, unsigned>, std::vector<double>> part; // [side, cluster] -> present part per ramp snapshot
        double excess = 0.0;
        for (arma::uword r = 0; r < 50; ++r)
        {
            const double G = std::pow(pg_sum(A, 50 + r), 1.0 - w[r]) * std::pow(pg_sum(B, r), w[r]);
            std::map<std::pair<unsigned, unsigned>, double> wc;
            for (const weight_t &x : weights(c, 50 + r, A, 50 + r, B, r))
            {
                if (x.type != 2)
                    continue;
                const auto key = std::make_pair(x.side, x.cluster);
                if (wc.count(key))
                    CHECK(std::abs(wc[key] - x.w) <= 1e-12 * x.w);
                wc[key] = x.w, excess = std::max(excess, 10.0 * std::log10(x.w));
            }
            for (const auto &kv : wc)
            {
                auto &v = part[kv.first];
                v.resize(50, 0.0);
                v[r] = kv.second * (kv.first.first == 0 ? pg_sum(A, 50 + r) : pg_sum(B, r)) / G;
            }
            auto sig = [](const ch_t &x, arma::uword s)
            {
                arma::vec p(x.coeff_re[s].n_slices);
                for (arma::uword q = 0; q < p.n_elem; ++q)
                    p[q] = arma::accu(arma::square(x.coeff_re[s].slice(q)) + arma::square(x.coeff_im[s].slice(q)));
                return (arma::uword)arma::accu(p > arma::accu(p) * std::pow(10.0, -2.5));
            };
            CHECK(sig(t, 50 + r) + 1 >= std::min(sig(TA, 50 + r), sig(TB, r)));
        }
        CHECK(excess <= 10.0 + 1e-9);
        CHECK(excess > 1.0);
        // Staggered exchange: per side and ramp snapshot, at most 3/4 of the clusters are partly present (measured: up to
        // 0.6; all of them with rho = 1)
        for (arma::uword r = 0; r < 50; ++r)
            for (unsigned side = 0; side < 2; ++side)
            {
                arma::uword n = 0, partial = 0;
                for (const auto &kv : part)
                    if (kv.first.first == side)
                        ++n, partial += kv.second[r] > 1e-12 && kv.second[r] < 1.0 - 1e-12 ? 1 : 0;
                CHECK(4 * partial <= 3 * n);
            }
        for (const auto &kv : part)
            for (arma::uword r = 1; r < 50; ++r)
                CHECK((kv.first.first == 0 ? kv.second[r] <= kv.second[r - 1] + 1e-12 : kv.second[r] >= kv.second[r - 1] - 1e-12));
    }
}

TEST_CASE("crossfade_segments - both output forms carry the same weighted paths")
{
    // Per snapshot and element pair, the sum of the tap coefficients equals that of the ray coefficients
    const ch_t A = tr_channel("UMa_LOS", 815, 40, "rays"), B = cut(tr_channel("UMa_NLOS", 822, 60, "rays"), 20, 60);
    const ch_t r = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 20}, "rays"), t = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 20}, "taps");
    REQUIRE(t.n_snap() == 60);
    for (arma::uword s = 0; s < 60; ++s)
    {
        const arma::mat dr = arma::sum(t.coeff_re[s], 2) - arma::sum(r.coeff_re[s], 2), di = arma::sum(t.coeff_im[s], 2) - arma::sum(r.coeff_im[s], 2);
        const double m = std::sqrt(total(r, s));
        CHECK(arma::abs(dr).max() <= 1e-12 * m);
        CHECK(arma::abs(di).max() <= 1e-12 * m);
    }
}

TEST_CASE("crossfade_segments - the fade order does not follow rounding at near-ties")
{
    // One ray per cluster and no GR (scenpar_override), a ramp of one snapshot: all fade orders cost about the same. A
    // relative change of 1e-13 in the coefficients and path gains of the new segment leaves the tap slots and the paths
    std::vector<ch_t> seg;
    for (arma::sword seed : {4711, 815})
        seg.push_back(tr_channel("UMa_LOS", seed, 4, "rays", {{"NumSubPaths", 1.0}, {"GR_enabled", 0.0}}));
    const arma::u32_vec tap = entry(seg[0], "tap"), type = entry(seg[0], "path_type");
    CHECK(arma::accu(tap == 3) == 1); // tap 3 is the first NLOS cluster, not a GR
    CHECK(arma::all(type.elem(arma::find(tap == 3)) == 2));
    for (arma::uword n_ov : {1, 2})
    {
        const ch_t t0 = quadriga_lib::crossfade_segments(seg, arma::uvec{0, n_ov}), r0 = quadriga_lib::crossfade_segments(seg, arma::uvec{0, n_ov}, "rays");
        for (double f : {1.0 + 1e-13, 1.0 - 1e-13})
        {
            std::vector<ch_t> p = seg;
            for (arma::uword s = 0; s < p[1].n_snap(); ++s)
                p[1].coeff_re[s] *= std::sqrt(f), p[1].coeff_im[s] *= std::sqrt(f), p[1].path_gain[s] *= f;
            const ch_t t1 = quadriga_lib::crossfade_segments(p, arma::uvec{0, n_ov}), r1 = quadriga_lib::crossfade_segments(p, arma::uvec{0, n_ov}, "rays");
            for (arma::uword s = 0; s < t0.n_snap(); ++s)
            {
                REQUIRE(t1.coeff_re[s].n_slices == t0.coeff_re[s].n_slices);
                REQUIRE(r1.coeff_re[s].n_slices == r0.coeff_re[s].n_slices);
                CHECK(arma::approx_equal(t1.coeff_re[s], t0.coeff_re[s], "absdiff", 1e-9 * std::sqrt(total(t0, s))));
            }
        }
    }
}

TEST_CASE("crossfade_segments - LOS and GR are one path each, marked by path_type or by the tap rule")
{
    // Ramp of 2 snapshots (w = 0.25, 0.75), levels within 10 dB, so the exchanged share is w: the LOS path holds
    // G ((1 - w) g_old / P_old + w g_new / P_new). Untyped, the single path of tap 3 is a GR; typed NLOS, it fades as a cluster
    auto with_type = [](ch_t c, const arma::u32_vec &t)
    {
        c.par_names.push_back("path_type"), c.par_data.push_back(t);
        return c;
    };
    const ch_t A = synth({0, 3, 6}, {1.0, 0.3, 0.2}, {100e-9, 150e-9, 220e-9}, 4, true);
    const ch_t B = synth({0, 3, 6, 9}, {0.5, 0.4, 0.25, 0.15}, {100e-9, 120e-9, 300e-9, 500e-9}, 4, true, 1.0, 2.0);
    const std::vector<double> w = ramp(2);
    auto count = [](const ch_t &c, arma::uword s, double d)
    {
        arma::uword n = 0;
        for (arma::uword p = 0; p < c.delay[s].n_slices; ++p)
            n += std::abs(c.delay[s](0, 0, p) - d) < 1e-15 ? 1 : 0;
        return n;
    };
    for (bool typed : {false, true})
    {
        INFO("typed " << typed);
        const ch_t a = typed ? with_type(A, {0, 2, 2}) : A, b = typed ? with_type(B, {0, 2, 2, 2}) : B;
        const ch_t c = quadriga_lib::crossfade_segments({a, b}, arma::uvec{0, 2}, "rays");
        REQUIRE(c.n_snap() == 6);
        for (arma::uword r = 0; r < 2; ++r)
        {
            const arma::uword s = 2 + r;
            const double Pa = pg_sum(A, s), Pb = pg_sum(B, r), G = std::pow(Pa, 1.0 - w[r]) * std::pow(Pb, w[r]);
            CHECK(count(c, s, 100e-9) == 1);
            double g_los = 0.0;
            for (arma::uword p = 0; p < c.delay[s].n_slices; ++p)
                g_los += std::abs(c.delay[s](0, 0, p) - 100e-9) < 1e-15 ? c.path_gain[s][p] : 0.0;
            CHECK(std::abs(g_los - G * ((1.0 - w[r]) * 1.0 / Pa + w[r] * 0.5 / Pb)) <= 1e-12 * G);
            CHECK(std::abs(pg_sum(c, s) - G) <= 1e-12 * G);
            // Untyped: one GR path (old delay 150 ns at w = 0.25: from the new segment, 120 ns); typed: no merge
            CHECK(count(c, s, 120e-9) + count(c, s, 150e-9) == (typed ? (arma::uword)(count(c, s, 120e-9) > 0) + (arma::uword)(count(c, s, 150e-9) > 0) : 1));
        }
    }

    // Invalid entries
    auto msg = [&](const ch_t &a, const ch_t &b)
    {
        try
        {
            quadriga_lib::crossfade_segments({a, b}, arma::uvec{0, 2});
        }
        catch (const std::invalid_argument &e)
        {
            return std::string(e.what());
        }
        return std::string();
    };
    CHECK(msg(with_type(A, {0, 3, 2}), B) == "Entry 0: the par_data entry 'path_type' of 'segments' must hold 0 (LOS), 1 (GR) or 2 (NLOS).");
    CHECK(msg(A, with_type(B, {0, 2, 2})) == "Entry 1: the par_data entry 'path_type' of 'segments' must have one value per path in every snapshot.");
}

TEST_CASE("crossfade_segments - hard switch, segments without power, positions, three segments")
{
    const ch_t A = seg_old(8, true), B = seg_new(12, true);
    {
        const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 0}, "rays");
        REQUIRE(c.n_snap() == 20);
        for (arma::uword s = 0; s < 20; ++s)
            CHECK(total(c, s) == (s < 8 ? total(A, s) : total(B, s - 8)));
    }
    {
        // A new segment without power: the union at own levels, (1 - w) P_old
        ch_t Z = B;
        for (arma::uword s = 0; s < Z.n_snap(); ++s)
            Z.coeff_re[s].zeros(), Z.coeff_im[s].zeros(), Z.path_gain[s].zeros();
        const ch_t c = quadriga_lib::crossfade_segments({A, Z}, arma::uvec{0, 4}, "rays");
        const std::vector<double> w = ramp(4);
        for (arma::uword r = 0; r < 4; ++r)
            CHECK(std::abs(pg_sum(c, 4 + r) - (1.0 - w[r]) * pg_sum(A, 4 + r)) <= 1e-12 * pg_sum(A, 4 + r));
    }
    {
        // Positions and orientations in the ramp are those of the old segment; three segments
        const ch_t C = synth({0, 6, 9, 12}, {0.2, 0.3, 0.3, 0.2}, {100e-9, 130e-9, 260e-9, 330e-9}, 10, true, 3.0, 12.0);
        const ch_t c = quadriga_lib::crossfade_segments({A, B, C}, arma::uvec{0, 6, 6}, "taps");
        REQUIRE(c.n_snap() == 8 + 12 + 10 - 12);
        for (arma::uword s = 2; s < 8; ++s)
        {
            CHECK(arma::approx_equal(c.rx_pos.col(s), A.rx_pos.col(s), "absdiff", 0.0));
            CHECK(arma::approx_equal(c.rx_orientation.col(s), A.rx_orientation.col(s), "absdiff", 0.0));
        }
        CHECK(arma::approx_equal(c.rx_pos.col(17), C.rx_pos.col(9), "absdiff", 0.0));
    }
}

TEST_CASE("crossfade_segments - input validation")
{
    const ch_t A = seg_old(4, true), B = seg_new(4, true);
    auto msg = [](const std::vector<ch_t> &v, const arma::uvec &n, const std::string &out)
    {
        try
        {
            quadriga_lib::crossfade_segments(v, n, out);
        }
        catch (const std::invalid_argument &e)
        {
            return std::string(e.what());
        }
        return std::string();
    };
    ch_t T = A;
    T.path_gain.clear(), T.path_length.clear(), T.polmat.clear(), T.path_angles.clear(), T.fbs_pos.clear(), T.lbs_pos.clear();
    ch_t V = A;
    V.coeff_re[1].shed_slice(3), V.coeff_im[1].shed_slice(3), V.delay[1].shed_slice(3), V.path_gain[1].shed_row(3), V.path_length[1].shed_row(3);
    V.polmat[1].shed_col(3), V.path_angles[1].shed_row(3), V.fbs_pos[1].shed_col(3), V.lbs_pos[1].shed_col(3);
    V.par_names.clear(), V.par_data.clear();
    CHECK(msg({}, {}, "taps") == "Input 'segments' must not be empty.");
    CHECK(msg({A, B}, {0}, "taps") == "Input 'n_overlap' must have one entry per segment.");
    CHECK(msg({A, B}, {1, 0}, "taps") == "The first entry of 'n_overlap' must be 0.");
    CHECK(msg({A, B}, {0, 2}, "paths") == "Input 'output' must be 'taps' or 'rays'.");
    CHECK(msg({A, T}, {0, 2}, "taps") == "Entry 1: 'segments' must have path gains (output \"rays\").");
    CHECK(msg({V, B}, {0, 2}, "taps") == "Entry 0: 'segments' must have the same number of paths in every snapshot.");
    CHECK(msg({A, B}, {0, 5}, "taps") == "Entry 1: 'n_overlap' must be at most the snapshots of the preceding segment after its own shared ones (4).");
    CHECK(msg({A, cut(B, 0, 2)}, {0, 3}, "taps") == "Entry 1: 'n_overlap' must be at most the snapshots of its own segment.");
}

TEST_CASE("crossfade_segments - the fade order keeps the delay and arrival spreads close to the superposition")
{
    // UMa_LOS to UMa_NLOS with levels within 10 dB (exchanged share = w): the delay spread and the azimuth arrival spread
    // of the output at each ramp snapshot against those of the superposition (1 - w) A / P_A + w B / P_B; normalized
    // mean squared deviation over the ramp (measured: DS 0.0012, ASA 0.0083; with the index or the reversed index order
    // instead of the search: DS 0.012, ASA 0.016 to 0.020)
    const ch_t A = tr_channel("UMa_LOS", 4711, 100, "rays"), B = cut(tr_channel("UMa_NLOS", 4718, 200, "rays"), 50, 200);
    const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 50}, "rays");
    const std::vector<double> w = ramp(50);
    auto mom = [](const ch_t &x, arma::uword s, double f, arma::vec &m)
    {
        for (arma::uword p = 0; p < x.path_gain[s].n_elem; ++p)
        {
            const double g = f * x.path_gain[s][p], d = x.delay[s](0, 0, p), a = x.path_angles[s](p, 2);
            m += arma::vec{g, g * d, g * d * d, g * std::cos(a), g * std::sin(a)};
        }
    };
    auto spreads = [](const arma::vec &m) { return arma::vec{std::sqrt(std::max(m[2] / m[0] - m[1] * m[1] / m[0] / m[0], 0.0)), std::sqrt(-2.0 * std::log(std::hypot(m[3], m[4]) / m[0]))}; };
    arma::vec err(2, arma::fill::zeros), ref(2, arma::fill::zeros);
    for (arma::uword r = 0; r < 50; ++r)
    {
        REQUIRE(std::abs(10.0 * std::log10(pg_sum(A, 50 + r) / pg_sum(B, r))) < 10.0);
        arma::vec mt(5, arma::fill::zeros), my(5, arma::fill::zeros);
        mom(A, 50 + r, (1.0 - w[r]) / pg_sum(A, 50 + r), mt), mom(B, r, w[r] / pg_sum(B, r), mt), mom(c, 50 + r, 1.0, my);
        const arma::vec T = spreads(mt), Y = spreads(my);
        err += arma::square(Y - T), ref += arma::square(T);
    }
    const arma::vec e = err / ref;
    CHECK(e[0] < 0.004);
    CHECK(e[1] < 0.012);
}

TEST_CASE("crossfade_segments - of near-equal costs the fade order keeps the lower cluster index")
{
    // Two old clusters (taps 6 and 9) with the same delay, angles and power up to a relative 1e-13: both orders cost the
    // same within rounding, and of near-equal costs the lower position wins. Moving tap 9 (the second unit) to position 0
    // is such a tie, so tap 9 fades out first, whatever the sign of the change
    for (double f : {1.0 + 1e-13, 1.0 - 1e-13})
    {
        ch_t A = synth({0, 6, 9}, {1.0, 0.3, 0.3 * f}, {100e-9, 200e-9, 200e-9}, 8, true);
        for (arma::uword s = 0; s < 8; ++s)
            A.path_angles[s].row(2) = A.path_angles[s].row(1);
        const ch_t B = synth({0, 6, 9}, {0.8, 0.25, 0.2}, {100e-9, 150e-9, 400e-9}, 8, true, 1.0, 2.0);
        const ch_t t = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 6});
        // Slots: LOS 0, old taps 6 and 9 at 1 and 2; at the middle of the ramp the one that fades first is weaker
        const arma::uword s = 2 + 3;
        auto slot_power = [&](arma::uword q) { return arma::accu(arma::square(t.coeff_re[s].slice(q)) + arma::square(t.coeff_im[s].slice(q))); };
        INFO("f - 1 = " << f - 1.0);
        CHECK(slot_power(2) < 0.9 * slot_power(1));
    }
}

TEST_CASE("crossfade_segments - without n_overlap the shared snapshots follow from the positions")
{
    // Old: Rx at x = 0 ... 7, new: x = 6 ... 17 (2 shared snapshots); new at x = 20 ... (none shared: a switch); ends that
    // do not move repeat their positions and need n_overlap
    const ch_t A = seg_old(8, true), B = seg_new(12, true), F = seg_new(12, true, 0.5, 20.0);
    for (const char *out : {"taps", "rays"})
    {
        const ch_t c = quadriga_lib::crossfade_segments({A, B}, {}, out), e = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 2}, out);
        REQUIRE(c.n_snap() == 18);
        for (arma::uword s = 0; s < 18; ++s)
            CHECK(arma::approx_equal(c.coeff_re[s], e.coeff_re[s], "absdiff", 0.0));
        CHECK(quadriga_lib::crossfade_segments({A, F}, {}, out).n_snap() == 20);
    }
    ch_t As = A, Bs = B;
    As.rx_pos = As.rx_pos.col(0), Bs.rx_pos = As.rx_pos;
    try
    {
        quadriga_lib::crossfade_segments({As, Bs});
        FAIL("no exception");
    }
    catch (const std::invalid_argument &x)
    {
        CHECK(std::string(x.what()) == "Entry 1: the positions of 'segments' do not tell the shared snapshots (they repeat); give 'n_overlap'.");
    }
    CHECK(quadriga_lib::crossfade_segments({As, Bs}, arma::uvec{0, 3}).n_snap() == 17);
}
