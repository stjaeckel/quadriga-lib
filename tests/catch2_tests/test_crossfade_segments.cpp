// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Tests of crossfade_segments on small synthetic channels: 2 x 2 element pairs, paths with constant powers and delays,
// phases that turn from snapshot to snapshot, tap indices of ssf_rays (LOS 0, GR 3, NLOS from 6, or from 3 without GR)
// and the path types of the entry path_type, the forms "rays"
// (with path data) and "taps". Oracles: the ramp w = 0.5 (1 + sin((x - 0.5) pi)), x = i / (n_r + 1), on
// n_r = round(overlap n_overlap) snapshots, and the power rule (1 - w) P_old + w P_new.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_channel.hpp"
#include "quadriga_layout.hpp"

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
    if (!rays)
        c.par_names.push_back("n_ray_tap"), c.par_data.push_back(arma::u32_vec(np, arma::fill::ones));
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

TEST_CASE("crossfade_segments - one segment and equal segments reproduce the input")
{
    for (bool rays : {true, false})
    {
        const ch_t X = synth({0, 3, 6, 9}, {1.0, 0.4, 0.2, 0.1}, {1e-7, 1.2e-7, 2e-7, 3e-7}, 10, rays);

        // One segment: the members that are carried, unchanged
        const ch_t one = quadriga_lib::crossfade_segments({X}, arma::uvec{0});
        CHECK(one.is_valid().empty());
        REQUIRE(one.n_snap() == 10);
        CHECK(one.name == X.name);
        CHECK(arma::approx_equal(one.rx_pos, X.rx_pos, "absdiff", 0.0));
        CHECK(arma::approx_equal(one.rx_orientation, X.rx_orientation, "absdiff", 0.0));
        CHECK(one.par_data.empty());
        for (arma::uword s = 0; s < 10; ++s)
        {
            CHECK(arma::approx_equal(one.coeff_re[s], X.coeff_re[s], "absdiff", 0.0));
            CHECK(arma::approx_equal(one.coeff_im[s], X.coeff_im[s], "absdiff", 0.0));
            CHECK(arma::approx_equal(one.delay[s], X.delay[s], "absdiff", 0.0));
            if (rays)
                CHECK(arma::approx_equal(one.path_gain[s], X.path_gain[s], "absdiff", 0.0));
        }

        // Equal segments (cuts of one track, snapshots 0-5 and 2-9, 4 shared): the snapshots outside the ramp are those
        // of the track, the ramp keeps the power and the LOS; with overlap = 0 the output is the track
        const std::vector<ch_t> seg = {cut(X, 0, 6), cut(X, 2, 10)};
        for (double ov : {0.0, 0.5, 1.0})
        {
            const ch_t c = quadriga_lib::crossfade_segments(seg, arma::uvec{0, 4}, ov);
            CHECK(c.is_valid().empty());
            REQUIRE(c.n_snap() == 10);
            const arma::uword n_r = (arma::uword)std::floor(ov * 4.0 + 0.5);
            for (arma::uword s = 0; s < 10; ++s)
            {
                const bool in_ramp = s >= 6 - n_r && s < 6;
                CHECK(std::abs(total(c, s) - total(X, s)) <= 1e-12 * total(X, s));
                CHECK(arma::approx_equal(c.rx_pos.col(s), X.rx_pos.col(s), "absdiff", 1e-12));
                if (!in_ramp && rays)
                {
                    CHECK(arma::approx_equal(c.coeff_re[s], X.coeff_re[s], "absdiff", 0.0));
                    CHECK(arma::approx_equal(c.coeff_im[s], X.coeff_im[s], "absdiff", 0.0));
                }
                if (in_ramp && rays) // LOS first, at the power of the track
                    CHECK(std::abs(c.path_gain[s](0) - 1.0) <= 1e-12);
            }
            if (ov == 0.0 && !rays) // taps: the same taps in the same slots
                for (arma::uword s = 0; s < 10; ++s)
                {
                    CHECK(arma::approx_equal(c.coeff_re[s], X.coeff_re[s], "absdiff", 0.0));
                    CHECK(arma::approx_equal(c.delay[s], X.delay[s], "absdiff", 0.0));
                }
        }
    }
}

TEST_CASE("crossfade_segments - power continuity across the ramp")
{
    // Old: 8 snapshots, new: 12 snapshots of which 6 are shared; tap and ray output give the interpolated total power
    for (bool rays : {true, false})
        for (double ov : {0.5, 1.0})
        {
            const ch_t A = seg_old(8, rays), B = seg_new(12, rays);
            const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 6}, ov);
            CHECK(c.is_valid().empty());
            REQUIRE(c.n_snap() == 8 + 12 - 6);
            const arma::uword n_r = (arma::uword)std::floor(ov * 6.0 + 0.5), r0 = 8 - n_r;
            const std::vector<double> w = ramp(n_r);
            for (arma::uword s = 0; s < c.n_snap(); ++s)
            {
                double expect;
                if (s < r0)
                    expect = total(A, s);
                else if (s < 8)
                    expect = (1.0 - w[s - r0]) * total(A, s) + w[s - r0] * total(B, s - 2);
                else
                    expect = total(B, s - 2);
                CHECK(std::abs(total(c, s) - expect) <= 1e-12 * expect);
            }
            if (!rays) // the tap count is fixed over all snapshots
                for (arma::uword s = 1; s < c.n_snap(); ++s)
                    CHECK(c.coeff_re[s].n_slices == c.coeff_re[0].n_slices);
        }
}

TEST_CASE("crossfade_segments - hard switch at overlap = 0")
{
    for (bool rays : {true, false})
    {
        const ch_t A = seg_old(8, rays), B = seg_new(12, rays);
        const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 6}, 0.0);
        REQUIRE(c.n_snap() == 14);
        for (arma::uword s = 0; s < 14; ++s)
        {
            const ch_t &X = s < 8 ? A : B;
            const arma::uword i = s < 8 ? s : s - 2;
            CHECK(std::abs(total(c, s) - total(X, i)) <= 1e-12 * total(X, i));
            CHECK(arma::approx_equal(c.rx_pos.col(s), X.rx_pos.col(i), "absdiff", 0.0));
            if (rays)
            {
                CHECK(arma::approx_equal(c.coeff_re[s], X.coeff_re[i], "absdiff", 0.0));
                CHECK(arma::approx_equal(c.path_gain[s], X.path_gain[i], "absdiff", 0.0));
            }
            else // the taps of the segment, with zero taps in the slots it does not use
            {
                arma::vec d = arma::sort(arma::vectorise(c.delay[s].elem(arma::find(arma::sum(arma::sum(arma::square(c.coeff_re[s]), 0), 1) > 0.0))));
                CHECK(arma::approx_equal(d, arma::sort(arma::vectorise(X.delay[i])), "absdiff", 1e-20));
            }
        }
    }
}

TEST_CASE("crossfade_segments - one LOS ray, NLOS ray power and the path union")
{
    // Ray output: at each ramp snapshot one LOS path with the power (1 - w) g_old + w g_new and the phase of the new
    // segment; all NLOS rays of both segments with their path data, the power of each scaled by the weight
    const ch_t A = seg_old(8, true), B = seg_new(12, true);
    const arma::u32_vec ta = {0, 6, 9, 12}, tb = {0, 6, 9, 12, 15};
    const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 6}, 1.0);
    const std::vector<double> w = ramp(6);
    for (arma::uword r = 0; r < 6; ++r)
    {
        const arma::uword s = 2 + r, i = 2 + r, j = r; // output, old and new snapshot
        REQUIRE(c.path_gain[s].n_elem == 1 + 3 + 4);   // the union of the paths
        const double g = (1.0 - w[r]) * gain_of(A, i, ta, true) + w[r] * gain_of(B, j, tb, true);
        CHECK(std::abs(c.path_gain[s](0) - g) <= 1e-14);
        const std::complex<double> h(c.coeff_re[s](0, 0, 0), c.coeff_im[s](0, 0, 0)), hb(B.coeff_re[j](0, 0, 0), B.coeff_im[j](0, 0, 0));
        CHECK(std::abs(std::arg(h / hb)) <= 1e-12); // phase of the new segment
        CHECK(std::abs(std::norm(h) - g) <= 1e-12);
        double nlos = 0.0;
        for (arma::uword p = 1; p < 8; ++p)
            nlos += c.path_gain[s](p);
        CHECK(std::abs(nlos - ((1.0 - w[r]) * gain_of(A, i, ta, false) + w[r] * gain_of(B, j, tb, false))) <= 1e-14);
        for (arma::uword p = 1; p < 4; ++p) // old rays: path data kept, amplitude sqrt(1 - w)
        {
            CHECK(c.delay[s](0, 0, p) == A.delay[i](0, 0, p));
            CHECK(c.path_length[s](p) == A.path_length[i](p));
            CHECK(arma::approx_equal(c.path_angles[s].row(p), A.path_angles[i].row(p), "absdiff", 0.0));
            CHECK(arma::approx_equal(c.fbs_pos[s].col(p), A.fbs_pos[i].col(p), "absdiff", 0.0));
            CHECK(arma::approx_equal(c.coeff_re[s].slice(p), std::sqrt(1.0 - w[r]) * A.coeff_re[i].slice(p), "absdiff", 1e-15));
            CHECK(arma::approx_equal(c.polmat[s].col(p), std::sqrt(1.0 - w[r]) * A.polmat[i].col(p), "absdiff", 1e-15));
        }
        for (arma::uword p = 1; p < 5; ++p) // new rays: amplitude sqrt(w)
        {
            CHECK(c.delay[s](0, 0, 3 + p) == B.delay[j](0, 0, p));
            CHECK(arma::approx_equal(c.lbs_pos[s].col(3 + p), B.lbs_pos[j].col(p), "absdiff", 0.0));
            CHECK(std::abs(c.path_gain[s](3 + p) - w[r] * B.path_gain[j](p)) <= 1e-15);
        }
    }

    // A new segment without LOS power keeps the phase and data of the old LOS, which fades out
    const ch_t B0 = seg_new(12, true, 0.0);
    const ch_t c0 = quadriga_lib::crossfade_segments({A, B0}, arma::uvec{0, 6}, 1.0);
    for (arma::uword r = 0; r < 6; ++r)
    {
        const arma::uword s = 2 + r;
        const std::complex<double> h(c0.coeff_re[s](1, 1, 0), c0.coeff_im[s](1, 1, 0)), ha(A.coeff_re[s](1, 1, 0), A.coeff_im[s](1, 1, 0));
        CHECK(std::abs(std::arg(h / ha)) <= 1e-12);
        CHECK(std::abs(c0.path_gain[s](0) - (1.0 - w[r]) * 1.0) <= 1e-14);
    }

    // GR (tap 3, one ray): both segments, the LOS rule; one segment only, a fade
    const ch_t G1 = synth({0, 3, 6}, {1.0, 0.5, 0.2}, {1e-7, 1.1e-7, 2e-7}, 8, true);
    const ch_t G2 = synth({0, 3, 6}, {0.8, 0.1, 0.2}, {1e-7, 1.05e-7, 3e-7}, 12, true, 2.0, 6.0);
    const ch_t N2 = synth({0, 6}, {0.8, 0.2}, {1e-7, 3e-7}, 12, true, 2.0, 6.0);
    const ch_t cg = quadriga_lib::crossfade_segments({G1, G2}, arma::uvec{0, 6}, 1.0), cn = quadriga_lib::crossfade_segments({G1, N2}, arma::uvec{0, 6}, 1.0);
    for (arma::uword r = 0; r < 6; ++r)
    {
        const arma::uword s = 2 + r;
        REQUIRE(cg.path_gain[s].n_elem == 4); // LOS, GR, 1 + 1 NLOS
        CHECK(std::abs(cg.path_gain[s](1) - ((1.0 - w[r]) * 0.5 + w[r] * 0.1)) <= 1e-14);
        CHECK(cg.delay[s](0, 0, 1) == 1.05e-7);
        REQUIRE(cn.path_gain[s].n_elem == 4); // LOS, the old GR, 1 + 1 NLOS
        CHECK(std::abs(cn.path_gain[s](1) - (1.0 - w[r]) * 0.5) <= 1e-14);
        CHECK(cn.delay[s](0, 0, 1) == 1.1e-7);
    }
}

TEST_CASE("crossfade_segments - tap and ray output give the same total power at each snapshot")
{
    // Taps of one ray each: the two forms hold the same paths, so the totals agree at every snapshot, also with taps
    // of very different power (a fade-out order that keeps the power per snapshot)
    for (double ov : {0.3, 0.5, 1.0})
    {
        const ch_t ra = quadriga_lib::crossfade_segments({seg_old(8, true), seg_new(12, true)}, arma::uvec{0, 6}, ov);
        const ch_t ta = quadriga_lib::crossfade_segments({seg_old(8, false), seg_new(12, false)}, arma::uvec{0, 6}, ov);
        REQUIRE(ra.n_snap() == ta.n_snap());
        for (arma::uword s = 0; s < ra.n_snap(); ++s)
            CHECK(std::abs(total(ra, s) - total(ta, s)) <= 1e-12 * total(ra, s));
    }

    // Three segments with two transitions; the new taps take the slots of faded taps: fewer slots than the union of all
    // taps (4 + 5 + 4)
    const ch_t C = synth({0, 6, 9, 12}, {0.2, 0.3, 0.3, 0.2}, {100e-9, 130e-9, 260e-9, 330e-9}, 10, false, 3.0, 12.0);
    const ch_t t3 = quadriga_lib::crossfade_segments({seg_old(8, false), seg_new(12, false), C}, arma::uvec{0, 6, 6}, 0.5);
    const ch_t r3 = quadriga_lib::crossfade_segments({seg_old(8, true), seg_new(12, true), synth({0, 6, 9, 12}, {0.2, 0.3, 0.3, 0.2}, {100e-9, 130e-9, 260e-9, 330e-9}, 10, true, 3.0, 12.0)},
                                                     arma::uvec{0, 6, 6}, 0.5);
    REQUIRE(t3.n_snap() == 8 + 6 + 4);
    CHECK(t3.coeff_re[0].n_slices < 1 + 3 + 4 + 3);
    for (arma::uword s = 0; s < t3.n_snap(); ++s)
    {
        CHECK(t3.coeff_re[s].n_slices == t3.coeff_re[0].n_slices);
        CHECK(std::abs(total(t3, s) - total(r3, s)) <= 1e-12 * total(r3, s));
    }
}

TEST_CASE("crossfade_segments - taps fade one after another and reuse the slots of faded taps")
{
    // The taps of both segments have distinct delays, so each slot of a snapshot names its tap. At each ramp snapshot at
    // most one old and one new NLOS tap are partly faded; the old taps hold (1 - w) and the new ones w of their power;
    // a new tap enters a slot only after its old tap is gone, so the output has fewer slots than the union of the taps
    const ch_t A = seg_old(8, false), B = seg_new(12, false);
    const arma::vec da = {150e-9, 220e-9, 400e-9}, pa = {0.3, 0.2, 0.1}, db = {120e-9, 300e-9, 500e-9, 700e-9}, pb = {0.05, 0.4, 0.25, 0.15};
    const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 6}, 1.0);
    const std::vector<double> w = ramp(6);
    CHECK(c.coeff_re[0].n_slices < 1 + 3 + 4);
    for (arma::uword r = 0; r < 6; ++r)
    {
        const arma::uword s = 2 + r;
        int part_a = 0, part_b = 0;
        double sum_a = 0.0, sum_b = 0.0;
        for (arma::uword q = 1; q < c.coeff_re[s].n_slices; ++q)
        {
            const double p = (arma::accu(arma::square(c.coeff_re[s].slice(q))) + arma::accu(arma::square(c.coeff_im[s].slice(q)))) / 4.0;
            if (p == 0.0)
                continue;
            const double d = c.delay[s](0, 0, q);
            const arma::uvec ia = arma::find(arma::abs(da - d) < 1e-15), ib = arma::find(arma::abs(db - d) < 1e-15);
            REQUIRE(ia.n_elem + ib.n_elem == 1);
            const double f = ia.n_elem ? p / pa(ia(0)) : p / pb(ib(0));
            CHECK(f <= 1.0 + 1e-12);
            if (f < 1.0 - 1e-12)
                (ia.n_elem ? part_a : part_b) += 1;
            (ia.n_elem ? sum_a : sum_b) += p;
        }
        CHECK(part_a <= 1);
        CHECK(part_b <= 1);
        CHECK(c.delay[s].min() >= 100e-9); // slots without power take a delay of the snapshot's taps (all >= 100 ns), not 0
        CHECK(std::abs(sum_a - (1.0 - w[r]) * 0.6) <= 1e-12);
        CHECK(std::abs(sum_b - w[r] * 0.85) <= 1e-12);
    }
}

TEST_CASE("crossfade_segments - orientations are interpolated, not the positions (NG reads tx_position for tx_orientation)")
{
    // A moving Tx with positions far from its orientations; the heading crosses +-180 degrees in the ramp
    ch_t A = seg_old(8, true), B = seg_new(12, true);
    const arma::vec tp = {50.0, 60.0, 70.0};
    A.tx_pos = arma::repmat(tp, 1, 8), B.tx_pos = arma::repmat(tp, 1, 12);
    A.tx_orientation = arma::repmat(arma::vec{0.1, 0.2, 179.0 * pi / 180.0}, 1, 8);
    B.tx_orientation = arma::repmat(arma::vec{0.1, 0.2, -179.0 * pi / 180.0}, 1, 12);
    const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 6}, 1.0);
    const std::vector<double> w = ramp(6);
    for (arma::uword r = 0; r < 6; ++r)
    {
        const arma::uword s = 2 + r;
        CHECK(arma::approx_equal(c.tx_pos.col(s), arma::vec{50.0, 60.0, 70.0}, "absdiff", 1e-12));
        CHECK(std::abs(c.tx_orientation(0, s) - 0.1) <= 1e-14);
        CHECK(std::abs(c.tx_orientation(1, s) - 0.2) <= 1e-14);
        const double h = (179.0 + 2.0 * w[r]) * pi / 180.0; // along the shorter way through 180 degrees
        CHECK(std::abs(std::remainder(c.tx_orientation(2, s) - h, 2.0 * pi)) <= 1e-12);
        const double x = (1.0 - w[r]) * A.rx_pos(0, s) + w[r] * B.rx_pos(0, r);
        CHECK(std::abs(c.rx_pos(0, s) - x) <= 1e-12);
    }
}

TEST_CASE("crossfade_segments - input validation")
{
    const ch_t A = seg_old(8, true), B = seg_new(12, true);
    auto msg = [&](const std::vector<ch_t> &v, const arma::uvec &n, double ov = 0.5)
    {
        try
        {
            quadriga_lib::crossfade_segments(v, n, ov);
        }
        catch (const std::invalid_argument &e)
        {
            return std::string(e.what());
        }
        return std::string();
    };
    CHECK(msg({A, B}, arma::uvec{0, 6}) == ""); // baseline
    SECTION("segments")
    {
        CHECK(msg({}, arma::uvec{}) == "Input 'segments' must not be empty.");
        ch_t bad = B;
        bad.coeff_im.pop_back();
        CHECK(msg({A, bad}, arma::uvec{0, 6}).rfind("Entry 1: 'segments' is not valid: ", 0) == 0);
        CHECK(msg({ch_t()}, arma::uvec{0}) == "Entry 0: 'segments' must have snapshots with channel coefficients.");
        ch_t ports = B;
        for (auto &x : ports.coeff_re)
            x = x.rows(0, 0);
        for (auto &x : ports.coeff_im)
            x = x.rows(0, 0);
        CHECK(msg({A, ports}, arma::uvec{0, 6}) == "All entries of 'segments' must have the same number of Rx and Tx ports.");
        CHECK(msg({A, seg_new(12, false)}, arma::uvec{0, 6}) == "All entries of 'segments' must have path gains (output \"rays\") or none (output \"taps\").");
        ch_t freq = B;
        freq.center_freq = {28e9};
        CHECK(msg({A, freq}, arma::uvec{0, 6}) == "All entries of 'segments' must have the same 'center_freq' with at most one value.");
        ch_t tap = B;
        tap.par_data[0] = arma::u32_vec{0, 6};
        CHECK(msg({A, tap}, arma::uvec{0, 6}) == "Entry 1: the par_data entry 'tap' of 'segments' must have one value per path in every snapshot.");
        tap.par_data[0] = arma::vec{0.0, 6.0};
        CHECK(msg({A, tap}, arma::uvec{0, 6}) == "Entry 1: the par_data entry 'tap' of 'segments' must be an arma::u32_vec.");
        ch_t var = seg_new(12, false);
        var.par_names.clear(), var.par_data.clear();
        var.coeff_re[3] = var.coeff_re[3].slices(0, 1), var.coeff_im[3] = var.coeff_im[3].slices(0, 1), var.delay[3] = var.delay[3].slices(0, 1);
        CHECK(msg({seg_old(8, false), var}, arma::uvec{0, 6}) == "Entry 1: 'segments' without path gains (taps) need the same number of taps in every snapshot.");
    }
    SECTION("n_overlap")
    {
        CHECK(msg({A, B}, arma::uvec{0}) == "Input 'n_overlap' must have one entry per segment.");
        CHECK(msg({A, B}, arma::uvec{1, 6}) == "The first entry of 'n_overlap' must be 0.");
        CHECK(msg({A, B}, arma::uvec{0, 9}) == "Entry 1: 'n_overlap' must be at most the snapshots of the preceding segment after its own shared ones (8).");
        CHECK(msg({A, cut(B, 0, 4)}, arma::uvec{0, 6}) == "Entry 1: 'n_overlap' must be at most the snapshots of its own segment.");
        CHECK(msg({A, B, A}, arma::uvec{0, 8, 5}) == "Entry 2: 'n_overlap' must be at most the snapshots of the preceding segment after its own shared ones (4).");
    }
    SECTION("overlap")
    {
        CHECK(msg({A, B}, arma::uvec{0, 6}, 1.5) == "Input 'overlap' must be in [0, 1].");
        CHECK(msg({A, B}, arma::uvec{0, 6}, -0.1) == "Input 'overlap' must be in [0, 1].");
        CHECK(msg({A, B}, arma::uvec{0, 6}, arma::datum::nan) == "Input 'overlap' must be in [0, 1].");
    }
}

TEST_CASE("crossfade_segments - fading taps of real channels follow the power ramp")
{
    // Taps of TR channels (full UMa ray sets, 2 x 2 generic antennas) of two single-segment layouts over the same track
    // of 200 rows, 0.1 m apart: UMa_LOS for the old, UMa_NLOS for the new segment, 100 shared snapshots. A tap sums the
    // rays of its cluster coherently, so its power fades with the snapshot, while the fade weights follow the mean tap
    // powers over the ramp: the total meets (1 - w) P_old + w P_new only on average. Bounds from measurements on such
    // channels (0.12 to 0.73 dB per snapshot, 0.05 dB for the mean over the ramp, overlap 0.2 to 1): 1 dB per snapshot,
    // 0.1 dB for the ratio of the sums over the ramp. For scale: the taps of one segment alone change by up to 2.5 dB
    // between neighbouring snapshots
    std::vector<ch_t> seg;
    for (const char *scen : {"UMa_LOS", "UMa_NLOS"})
    {
        quadriga_lib::layout L;
        L.tx.name = {"BS1"}, L.tx.sector = {0}, L.tx.i_antenna = {0}, L.tx.offset = {0};
        L.tx.pos = arma::mat(arma::vec{0.0, 0.0, 25.0}), L.tx.orientation.zeros(3, 1), L.tx.scenario = {scen};
        L.rx.name = {"UT1"}, L.rx.sector = {0}, L.rx.i_antenna = {0}, L.rx.offset = {0};
        L.rx.pos.set_size(3, 200), L.rx.orientation.zeros(3, 200);
        for (arma::uword i = 0; i < 200; ++i)
            L.rx.pos(0, i) = 100.0 + 0.1 * (double)i, L.rx.pos(1, i) = 20.0, L.rx.pos(2, i) = 1.5;
        L.center_freq = {3.5e9}, L.seed = 4711;
        quadriga_lib::init_link_states(L);
        seg.push_back(quadriga_lib::get_gbsm_channels(L, quadriga_lib::generate_rays(L), {}, {}, "taps")[0]);
    }
    const ch_t A = cut(seg[0], 0, 100), &B = seg[1];
    for (double ov : {0.2, 0.5, 1.0})
    {
        const ch_t c = quadriga_lib::crossfade_segments({A, B}, arma::uvec{0, 100}, ov);
        const arma::uword n_r = (arma::uword)std::floor(ov * 100.0 + 0.5);
        const std::vector<double> w = ramp(n_r);
        double sum_c = 0.0, sum_t = 0.0, worst = 0.0;
        for (arma::uword r = 0; r < n_r; ++r)
        {
            const arma::uword s = 100 - n_r + r;
            const double target = (1.0 - w[r]) * total(A, s) + w[r] * total(B, s);
            worst = std::max(worst, std::abs(10.0 * std::log10(total(c, s) / target)));
            sum_c += total(c, s), sum_t += target;
        }
        CHECK(worst < 1.0);
        CHECK(std::abs(10.0 * std::log10(sum_c / sum_t)) < 0.1);
    }
}

TEST_CASE("crossfade_segments - the par_data entry path_type marks the LOS, GR and NLOS paths")
{
    // "rays", one ramp snapshot (n_overlap 2, overlap 0.5): output snapshot 3, old snapshot 3, new snapshot 1. The merged
    // LOS and GR are one path each; the NLOS paths of both segments are side by side. "taps": the slot counts
    auto with_type = [](ch_t c, const arma::u32_vec &t)
    {
        c.par_names.push_back("path_type"), c.par_data.push_back(t);
        return c;
    };
    auto n_ramp = [](const ch_t &a, const ch_t &b) { return quadriga_lib::crossfade_segments({a, b}, arma::uvec{0, 2}, 0.5).coeff_re[3].n_slices; };
    for (bool rays : {true, false})
    {
        INFO("rays " << rays);
        // NLOS taps from index 3 (clusters numbered without a GR slot): the tap rule takes the single ray of tap 3 for a
        // GR ray (merged: LOS, GR, NLOS 6 old, NLOS 6, 9 new = 5 paths); with path_type it is NLOS (6 paths)
        const ch_t A = synth({0, 3, 6}, {1.0, 0.3, 0.2}, {100e-9, 150e-9, 220e-9}, 4, rays);
        const ch_t B = synth({0, 3, 6, 9}, {0.5, 0.4, 0.25, 0.15}, {100e-9, 120e-9, 300e-9, 500e-9}, 4, rays, 1.0, 2.0);
        const ch_t At = with_type(A, {0, 2, 2}), Bt = with_type(B, {0, 2, 2, 2});
        const ch_t c = quadriga_lib::crossfade_segments({At, Bt}, arma::uvec{0, 2}, 0.5);
        CHECK(std::abs(total(c, 3) - 0.5 * (total(A, 3) + total(B, 1))) < 1e-12 * total(c, 3)); // Power rule at w = 0.5
        if (!rays)
        {
            // Taps: one slot set for all snapshots, LOS, GR (untyped: tap 3), NLOS slots that fading taps share; regression
            // values of a traced run: the untyped layout has the GR slot and 3 NLOS slots, the typed one 3 NLOS slots
            CHECK(n_ramp(A, B) == 5);
            CHECK(n_ramp(At, Bt) == 4);
        }
        if (rays)
        {
            CHECK(n_ramp(A, B) == 5);
            CHECK(n_ramp(At, Bt) == 6);
            // A GR at another tap index: NLOS for the tap rule (6 paths), GR with path_type (5 paths)
            const ch_t C = synth({0, 5, 6}, {1.0, 0.3, 0.2}, {100e-9, 150e-9, 220e-9}, 4, rays);
            const ch_t D = synth({0, 5, 6, 9}, {0.5, 0.4, 0.25, 0.15}, {100e-9, 120e-9, 300e-9, 500e-9}, 4, rays, 1.0, 2.0);
            CHECK(n_ramp(C, D) == 6);
            CHECK(n_ramp(with_type(C, {0, 1, 2}), with_type(D, {0, 1, 2, 2})) == 5);
        }

        // Invalid entries
        auto msg = [&](const ch_t &a, const ch_t &b)
        {
            try
            {
                n_ramp(a, b);
            }
            catch (const std::invalid_argument &e)
            {
                return std::string(e.what());
            }
            return std::string();
        };
        CHECK(msg(with_type(A, {0, 3, 2}), Bt) == "Entry 0: the par_data entry 'path_type' of 'segments' must hold 0 (LOS), 1 (GR) or 2 (NLOS).");
        CHECK(msg(At, with_type(B, {0, 2, 2})) == "Entry 1: the par_data entry 'path_type' of 'segments' must have one value per path in every snapshot.");
    }
}

TEST_CASE("crossfade_segments - channels with one ray per cluster and no GR keep the NLOS tap 3")
{
    // UMa_LOS with NumSubPaths = 1 and GR_enabled = 0 (scenpar_override): the first NLOS cluster has the tap index 3 and
    // one ray, which the tap rule takes for a GR ray. gbsm_single_link_channel marks it NLOS in path_type, so its old and
    // new paths fade side by side: the ramp snapshot has n_old + n_new - 1 paths (one merged LOS); without the entry
    // path_type, the tap rule merges tap 3 too (n_old + n_new - 2)
    std::vector<ch_t> seg, segt; // "rays" and "taps"
    for (arma::sword seed : {4711, 815})
    {
        quadriga_lib::layout L;
        L.tx.name = {"BS1"}, L.tx.sector = {0}, L.tx.i_antenna = {0}, L.tx.offset = {0};
        L.tx.pos = arma::mat(arma::vec{0.0, 0.0, 25.0}), L.tx.orientation.zeros(3, 1), L.tx.scenario = {"UMa_LOS"};
        L.tx.scenpar_override["NumSubPaths"] = {1.0}, L.tx.scenpar_override["GR_enabled"] = {0.0};
        L.rx.name = {"UT1"}, L.rx.sector = {0}, L.rx.i_antenna = {0}, L.rx.offset = {0};
        L.rx.pos.set_size(3, 4), L.rx.orientation.zeros(3, 4);
        for (arma::uword i = 0; i < 4; ++i)
            L.rx.pos(0, i) = 100.0 + 0.1 * (double)i, L.rx.pos(1, i) = 20.0, L.rx.pos(2, i) = 1.5;
        L.center_freq = {3.5e9}, L.seed = seed;
        quadriga_lib::init_link_states(L);
        seg.push_back(quadriga_lib::get_gbsm_channels(L, quadriga_lib::generate_rays(L), {}, {}, "rays")[0]);
        segt.push_back(quadriga_lib::get_gbsm_channels(L, quadriga_lib::generate_rays(L), {}, {}, "taps")[0]);
    }
    const arma::uword na = seg[0].coeff_re[0].n_slices, nb = seg[1].coeff_re[0].n_slices;
    for (const ch_t &s : seg)
    {
        auto entry = [&](const std::string &name)
        {
            for (size_t i = 0; i < s.par_names.size(); ++i)
                if (s.par_names[i] == name)
                    return std::any_cast<arma::u32_vec>(s.par_data[i]);
            return arma::u32_vec();
        };
        const arma::u32_vec tap = entry("tap"), type = entry("path_type");
        REQUIRE(tap.n_elem == s.coeff_re[0].n_slices);
        REQUIRE(type.n_elem == tap.n_elem);
        CHECK(arma::accu(tap == 3) == 1);
        CHECK(arma::all(type.elem(arma::find(tap == 3)) == 2));
        CHECK(arma::all(type.elem(arma::find(tap == 0)) == 0));
    }
    CHECK(quadriga_lib::crossfade_segments(seg, arma::uvec{0, 2}, 0.5).coeff_re[3].n_slices == na + nb - 1);
    std::vector<ch_t> untyped = seg;
    for (ch_t &s : untyped)
        for (size_t i = 0; i < s.par_names.size(); ++i)
            if (s.par_names[i] == "path_type")
                s.par_names.erase(s.par_names.begin() + (std::ptrdiff_t)i), s.par_data.erase(s.par_data.begin() + (std::ptrdiff_t)i);
    CHECK(quadriga_lib::crossfade_segments(untyped, arma::uvec{0, 2}, 0.5).coeff_re[3].n_slices == na + nb - 2);
    {
        std::vector<ch_t> ut = segt;
        for (ch_t &s : ut)
            for (size_t i = 0; i < s.par_names.size(); ++i)
                if (s.par_names[i] == "path_type")
                    s.par_names.erase(s.par_names.begin() + (std::ptrdiff_t)i), s.par_data.erase(s.par_data.begin() + (std::ptrdiff_t)i);
        // Taps (13 per segment): slot counts of a traced run, also found by an independent probe; with path_type, tap 3 of
        // both segments fades as an NLOS tap instead of one merged GR slot
        REQUIRE(segt[0].coeff_re[0].n_slices == 13);
        REQUIRE(segt[1].coeff_re[0].n_slices == 13);
        CHECK(quadriga_lib::crossfade_segments(segt, arma::uvec{0, 2}, 0.5).coeff_re[3].n_slices == 17);
        CHECK(quadriga_lib::crossfade_segments(ut, arma::uvec{0, 2}, 0.5).coeff_re[3].n_slices == 15);
    }
}
