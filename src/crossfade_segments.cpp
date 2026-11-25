// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# crossfade_segments
Joins the channels of consecutive segments of one link by a power cross-fade

- Port of the channel merger of QuaDRiGa: segment `k` starts with `n_overlap[k]` snapshots that are also the last ones of segment `k − 1`; the output has `Σ (n_snap[k] − n_overlap[k])` snapshots
- In the last `n_r = round(overlap · n_overlap[k])` shared snapshots (the ramp) the old paths fade out with `1 − w` and the new ones fade in with `w`, in power, `w = 0.5 (1 + sin((i / (n_r + 1) − 0.5) π))`, `i = 1 … n_r`; before the ramp the snapshots are those of the old segment
- LOS (tap index 0 of the `par_data` entry `tap`) and GR (tap index 3 held by one ray, without index 4 or 5, as [[ssf_rays]] numbers them): one path with the data of the new segment (of the old one where the new one has no power), scaled to `|h|² = (1 − w) |h_old|² + w |h_new|²` by the path gain (rays) or the tap power (taps)
- `"rays"` (channels with `path_gain`): the NLOS paths of both segments side by side, amplitudes `sqrt(1 − w)` and `sqrt(w)`; `path_length`, `path_angles`, `fbs_pos` and `lbs_pos` are kept where all segments have them
- `"taps"` (without `path_gain`, one tap count per segment): one set of tap slots for all snapshots (LOS, GR, NLOS); the NLOS taps of the old segment hold `1 − w` and those of the new segment `w` of their mean power over the ramp; slots without power hold zero coefficients and the smallest delay of the snapshot
- Delays as given (one reference for all segments); positions interpolated in the ramp, orientations along the shorter way; `name` and `center_freq` of segment 0; `par_data`, `n_interact` and `interact_coord` are not carried

## Declaration:
```
channel<double> quadriga_lib::crossfade_segments(
    const std::vector<channel<double>> &segments,
    const arma::uvec &n_overlap,
    double overlap = 0.5);
```

## Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients, equal `n_rx`, `n_tx`, form and `center_freq` (at most one value); `[n_segment]`
- **`n_overlap`** — Snapshots shared with the preceding segment, the first entry is 0, at most the snapshots of the preceding segment after its own shared ones; `[n_segment]`
- **`overlap`** *(optional)* — Part of the shared snapshots used for the ramp, in [0, 1]

## Returns:
- The joined channel, see [[channel]]

## See also:
- [[get_gbsm_channels]] (joins the segments of tracks with this function)
- [[gbsm_single_link_channel]] (channels of one segment)
MD!*/

namespace
{
using qd_ch = quadriga_lib::channel<double>;

// LOS and GR path of a segment (-1 = none) and its NLOS paths; 'tagged': the segment has the entry "tap"
struct qd_xf_seg
{
    long long los = -1, gr = -1;
    std::vector<arma::uword> nlos;
    bool tagged = false;
};

// An entry of par_data as arma::u32_vec, or nullptr
const arma::u32_vec *qd_xf_par(const qd_ch &c, const char *name, arma::uword k)
{
    for (size_t i = 0; i < c.par_names.size() && i < c.par_data.size(); ++i)
        if (c.par_names[i] == name)
        {
            const arma::u32_vec *v = std::any_cast<arma::u32_vec>(&c.par_data[i]);
            if (v == nullptr)
                throw std::invalid_argument("Entry " + std::to_string(k) + ": the par_data entry '" + name + "' of 'segments' must be an arma::u32_vec.");
            return v;
        }
    return nullptr;
}

// LOS, GR and NLOS paths of segment k from its tap indices (all NLOS without the entry "tap")
qd_xf_seg qd_xf_classify(const qd_ch &c, arma::uword k, bool taps)
{
    qd_xf_seg g;
    const arma::u32_vec *tap = qd_xf_par(c, "tap", k);
    const arma::uword n = c.coeff_re[0].n_slices;
    if (tap == nullptr)
    {
        for (arma::uword p = 0; p < n; ++p)
            g.nlos.push_back(p);
        return g;
    }
    g.tagged = true;
    for (const arma::cube &x : c.coeff_re)
        if (x.n_slices != tap->n_elem)
            throw std::invalid_argument("Entry " + std::to_string(k) + ": the par_data entry 'tap' of 'segments' must have one value per path in every snapshot.");
    arma::uword n3 = 0;
    bool sub3 = false;
    for (arma::uword p = 0; p < n; ++p)
        n3 += (*tap)[p] == 3 ? 1 : 0, sub3 = sub3 || (*tap)[p] == 4 || (*tap)[p] == 5;
    const arma::u32_vec *rays_per_tap = taps ? qd_xf_par(c, "n_ray_tap", k) : nullptr;
    for (arma::uword p = 0; p < n; ++p)
    {
        const unsigned t = (*tap)[p];
        if (t == 0 && g.los < 0)
            g.los = (long long)p;
        else if (t == 3 && g.gr < 0 && n3 == 1 && !sub3 &&
                 (rays_per_tap == nullptr || rays_per_tap->n_elem != n || (*rays_per_tap)[p] == 1))
            g.gr = (long long)p;
        else
            g.nlos.push_back(p);
    }
    return g;
}

// Power of path p of a coefficient snapshot, summed over the element pairs
double qd_xf_power(const arma::cube &re, const arma::cube &im, arma::uword p)
{
    double s = 0.0;
    const double *a = re.slice_memptr(p), *b = im.slice_memptr(p);
    for (arma::uword e = 0; e < re.n_rows * re.n_cols; ++e)
        s += a[e] * a[e] + b[e] * b[e];
    return s;
}

// Mean delay of path p of a delay snapshot over the element pairs
double qd_xf_delay(const arma::cube &d, arma::uword p)
{
    return arma::accu(d.slice(p)) / (double)(d.n_rows * d.n_cols);
}

// Order in which the NLOS taps of the old (a) and new (b) segment fade, as in QuaDRiGa's merger: the ramp is split into
// n_sub = min(n_a, n_b, n_ramp) sub-intervals, in each one old tap fades out and one new tap fades in; the order
// minimises the deviation of the delay spread from a linear change between the segments (in dB) plus the largest power
// deviation from the ramp (in dB); two passes of moving each tap to its best position. pd: [power, delay] of the
// deterministic (LOS, GR) and NLOS taps, means over the ramp
void qd_xf_order(const std::vector<double> &pa_det, const std::vector<double> &da_det, const std::vector<double> &pa, const std::vector<double> &da,
                 const std::vector<double> &pb_det, const std::vector<double> &db_det, const std::vector<double> &pb, const std::vector<double> &db,
                 arma::uword n_ramp, std::vector<arma::uword> &oa, std::vector<arma::uword> &ob)
{
    const size_t na = pa.size(), nb = pb.size(), n_sub = std::min({na, nb, (size_t)n_ramp});
    oa.resize(na), ob.resize(nb);
    for (size_t i = 0; i < na; ++i)
        oa[i] = i;
    for (size_t i = 0; i < nb; ++i)
        ob[i] = i;
    if (n_sub == 0)
        return;
    auto spread = [](const std::vector<double> &p1, const std::vector<double> &d1, const std::vector<double> &p2, const std::vector<double> &d2)
    {
        double s = 0.0, m = 0.0, q = 0.0;
        for (size_t i = 0; i < p1.size(); ++i)
            s += p1[i], m += p1[i] * d1[i], q += p1[i] * d1[i] * d1[i];
        for (size_t i = 0; i < p2.size(); ++i)
            s += p2[i], m += p2[i] * d2[i], q += p2[i] * d2[i] * d2[i];
        return s > 0.0 ? std::sqrt(std::max(q / s - (m / s) * (m / s), 0.0)) : 0.0;
    };
    const double ds1 = spread(pa_det, da_det, pa, da), ds2 = spread(pb_det, db_det, pb, db);
    double P1 = 0.0, P2 = 0.0;
    for (double x : pa_det)
        P1 += x;
    for (double x : pa)
        P1 += x;
    for (double x : pb_det)
        P2 += x;
    for (double x : pb)
        P2 += x;
    std::vector<double> ro(n_sub), ds_t(n_sub);
    const double pi = arma::datum::pi;
    for (size_t i = 0; i < n_sub; ++i)
    {
        const double x = (2.0 * (double)i + 1.0) / (2.0 * (double)n_sub);
        ro[i] = 0.5 * (1.0 + std::sin((x - 0.5) * pi)), ds_t[i] = ds1 + (ds2 - ds1) * x;
    }
    auto cost = [&](const std::vector<arma::uword> &A, const std::vector<arma::uword> &B)
    {
        double num = 0.0, den = 0.0, pw = 0.0;
        for (size_t i = 0; i < n_sub; ++i)
        {
            double s = 0.0, m = 0.0, q = 0.0;
            auto add = [&](double wt, double p, double d)
            {
                const double x = wt * p;
                s += x, m += x * d, q += x * d * d;
            };
            for (size_t j = 0; j < pa_det.size(); ++j)
                add(1.0 - ro[i], pa_det[j], da_det[j]);
            for (size_t j = 0; j < pb_det.size(); ++j)
                add(ro[i], pb_det[j], db_det[j]);
            for (size_t j = 0; j < na; ++j)
                add(j < n_sub ? (j > i ? 1.0 : (j == i ? 0.5 : 0.0)) : 1.0 - ro[i], pa[A[j]], da[A[j]]);
            for (size_t j = 0; j < nb; ++j)
                add(j < n_sub ? (i > j ? 1.0 : (i == j ? 0.5 : 0.0)) : ro[i], pb[B[j]], db[B[j]]);
            const double ds = s > 0.0 ? std::sqrt(std::max(q / s - (m / s) * (m / s), 0.0)) : 0.0;
            num += (ds_t[i] - ds) * (ds_t[i] - ds), den += ds_t[i] * ds_t[i];
            const double target = P1 - (P1 - P2) * ro[i];
            if (s > 0.0 && target > 0.0)
                pw = std::max(pw, std::abs(10.0 * std::log10(target / s)));
        }
        return (den > 0.0 ? 10.0 * std::log10(num / den) : 0.0) + pw;
    };
    auto improve = [&](std::vector<arma::uword> &O, bool first)
    {
        const size_t n = O.size();
        for (arma::uword c = 0; c < n; ++c)
        {
            std::vector<arma::uword> rest;
            for (arma::uword x : O)
                if (x != c)
                    rest.push_back(x);
            double best = std::numeric_limits<double>::infinity();
            size_t at = 0;
            for (size_t pos = 0; pos < n; ++pos)
            {
                std::vector<arma::uword> T(rest.begin(), rest.begin() + (std::ptrdiff_t)pos);
                T.push_back(c);
                T.insert(T.end(), rest.begin() + (std::ptrdiff_t)pos, rest.end());
                const double v = first ? cost(T, ob) : cost(oa, T);
                if (v < best)
                    best = v, at = pos;
            }
            O.assign(rest.begin(), rest.begin() + (std::ptrdiff_t)at);
            O.push_back(c);
            O.insert(O.end(), rest.begin() + (std::ptrdiff_t)at, rest.end());
        }
    };
    for (int pass = 0; pass < 2; ++pass)
        improve(oa, true), improve(ob, false);
}

// Weight of each tap at each ramp snapshot, [n_tap][n_ramp]: the taps fade in the order O one after another so that
// the weighted sum of the powers p is f[r] times their sum (f = 1 - w for the old, w for the new segment); 'last'
// reverses the order (old taps: the first in O fades first)
std::vector<std::vector<double>> qd_xf_weights(const std::vector<double> &p, const std::vector<arma::uword> &O, const std::vector<double> &f, bool fade_out)
{
    const size_t n = p.size(), nr = f.size();
    std::vector<std::vector<double>> W(n, std::vector<double>(nr));
    double P = 0.0;
    for (double x : p)
        P += x;
    std::vector<double> before(n); // Power of the taps that are present longer than tap O[m]
    double acc = 0.0;
    if (fade_out)
        for (size_t m = n; m-- > 0;)
            before[m] = acc, acc += p[O[m]];
    else
        for (size_t m = 0; m < n; ++m)
            before[m] = acc, acc += p[O[m]];
    for (size_t m = 0; m < n; ++m)
        for (size_t r = 0; r < nr; ++r)
        {
            const double q = p[O[m]], x = f[r] * P - before[m];
            W[O[m]][r] = P <= 0.0 ? f[r] : (q > 0.0 ? std::min(std::max(x / q, 0.0), 1.0) : (x > 0.0 ? 1.0 : 0.0));
        }
    return W;
}

// One output snapshot: a snapshot of segment a, or a ramp snapshot between a and b = a + 1 with weight w
struct qd_xf_snap
{
    arma::uword a, ia;
    long long b;
    arma::uword ib, r;
    double w;
};

// Column s of a [3, 1 or n] matrix, zeros if empty
arma::vec qd_xf_col(const arma::mat &M, arma::uword s)
{
    if (M.is_empty())
        return arma::vec(3, arma::fill::zeros);
    return M.col(M.n_cols == 1 ? 0 : s);
}
} // namespace

quadriga_lib::channel<double> quadriga_lib::crossfade_segments(const std::vector<channel<double>> &segments, const arma::uvec &n_overlap, double overlap)
{
    const arma::uword n_seg = (arma::uword)segments.size();
    if (n_seg == 0)
        throw std::invalid_argument("Input 'segments' must not be empty.");
    if (n_overlap.n_elem != n_seg)
        throw std::invalid_argument("Input 'n_overlap' must have one entry per segment.");
    if (n_overlap[0] != 0)
        throw std::invalid_argument("The first entry of 'n_overlap' must be 0.");
    if (!(overlap >= 0.0 && overlap <= 1.0))
        throw std::invalid_argument("Input 'overlap' must be in [0, 1].");
    for (arma::uword k = 0; k < n_seg; ++k)
    {
        const qd_ch &c = segments[k];
        const std::string msg = c.is_valid();
        if (!msg.empty())
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'segments' is not valid: " + msg);
        if (c.n_snap() == 0 || c.coeff_re.empty())
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'segments' must have snapshots with channel coefficients.");
        const qd_ch &c0 = segments[0];
        if (c.n_rx() != c0.n_rx() || c.n_tx() != c0.n_tx())
            throw std::invalid_argument("All entries of 'segments' must have the same number of Rx and Tx ports.");
        if (c.path_gain.empty() != c0.path_gain.empty())
            throw std::invalid_argument("All entries of 'segments' must have path gains (output \"rays\") or none (output \"taps\").");
        if (c.center_freq.n_elem > 1 || c.center_freq.n_elem != c0.center_freq.n_elem || (c.center_freq.n_elem == 1 && c.center_freq[0] != c0.center_freq[0]))
            throw std::invalid_argument("All entries of 'segments' must have the same 'center_freq' with at most one value.");
        if (k > 0 && n_overlap[k] > segments[k - 1].n_snap() - n_overlap[k - 1])
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'n_overlap' must be at most the snapshots of the preceding segment after its own shared ones (" + std::to_string(segments[k - 1].n_snap() - n_overlap[k - 1]) + ").");
        if (n_overlap[k] > c.n_snap())
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'n_overlap' must be at most the snapshots of its own segment.");
    }
    const bool rays = !segments[0].path_gain.empty(), taps = !rays;
    std::vector<qd_xf_seg> info(n_seg);
    for (arma::uword k = 0; k < n_seg; ++k)
    {
        if (taps)
            for (const arma::cube &x : segments[k].coeff_re)
                if (x.n_slices != segments[k].coeff_re[0].n_slices)
                    throw std::invalid_argument("Entry " + std::to_string(k) + ": 'segments' without path gains (taps) need the same number of taps in every snapshot.");
        info[k] = qd_xf_classify(segments[k], k, taps);
    }

    // Output snapshots; ramp of transition k (from k - 1 to k): the last n_ramp[k] shared snapshots
    arma::uvec n_ramp(n_seg, arma::fill::zeros);
    std::vector<std::vector<double>> ramp_w(n_seg);
    const double pi = arma::datum::pi;
    for (arma::uword k = 1; k < n_seg; ++k)
    {
        n_ramp[k] = (arma::uword)std::floor(overlap * (double)n_overlap[k] + 0.5);
        for (arma::uword r = 0; r < n_ramp[k]; ++r)
        {
            const double x = (double)(r + 1) / (double)(n_ramp[k] + 1);
            ramp_w[k].push_back(0.5 * (1.0 + std::sin((x - 0.5) * pi)));
        }
    }
    std::vector<qd_xf_snap> out;
    for (arma::uword k = 0; k < n_seg; ++k)
    {
        const arma::uword S = segments[k].n_snap(), nr = k + 1 < n_seg ? n_ramp[k + 1] : 0;
        for (arma::uword i = n_overlap[k]; i < S - nr; ++i)
            out.push_back({k, i, -1, 0, 0, 0.0});
        for (arma::uword r = 0; r < nr; ++r)
            out.push_back({k, S - nr + r, (long long)k + 1, n_overlap[k + 1] - nr + r, r, ramp_w[k + 1][r]});
    }
    const arma::uword n_out = (arma::uword)out.size();

    // Taps: slots of the NLOS taps of each segment, the fade weights of the old and new taps in each ramp. The NLOS taps
    // fade one after another in the order of QuaDRiGa's merger (chosen for a delay spread close to a linear change), each
    // over the part of the ramp that matches its share of the mean power; a new tap takes the slot of a faded one
    std::vector<std::vector<arma::uword>> slot(n_seg);
    std::vector<std::vector<std::vector<double>>> w_out(n_seg), w_in(n_seg); // [k][tap][r]: fade-out in ramp k+1, fade-in in ramp k
    arma::uword n_slot = 0;
    if (taps)
    {
        const arma::uword n0 = (arma::uword)info[0].nlos.size();
        for (arma::uword j = 0; j < n0; ++j)
            slot[0].push_back(j);
        n_slot = n0;
        for (arma::uword k = 1; k < n_seg; ++k)
        {
            const qd_ch &A = segments[k - 1], &B = segments[k];
            const qd_xf_seg &ia = info[k - 1], &ib = info[k];
            const arma::uword nr = n_ramp[k], na = (arma::uword)ia.nlos.size(), nb = (arma::uword)ib.nlos.size();
            const arma::uword a0 = A.n_snap() - nr, b0 = n_overlap[k] - nr; // first ramp snapshot in A and B
            std::vector<arma::uword> oa, ob;
            std::vector<arma::uword> exit_r(na, nr), entry_r(nb, nr);
            if (nr > 0)
            {
                auto means = [nr](const qd_ch &c, arma::uword s0, const std::vector<arma::uword> &ps, std::vector<double> &p, std::vector<double> &d)
                {
                    p.assign(ps.size(), 0.0), d.assign(ps.size(), 0.0);
                    for (size_t j = 0; j < ps.size(); ++j)
                        for (arma::uword s = s0; s < s0 + nr; ++s)
                            p[j] += qd_xf_power(c.coeff_re[s], c.coeff_im[s], ps[j]) / (double)(nr * c.n_rx() * c.n_tx()),
                                d[j] += qd_xf_delay(c.delay[s], ps[j]) / (double)nr;
                };
                auto det = [](const qd_xf_seg &g)
                {
                    std::vector<arma::uword> v;
                    if (g.los >= 0)
                        v.push_back((arma::uword)g.los);
                    if (g.gr >= 0)
                        v.push_back((arma::uword)g.gr);
                    return v;
                };
                std::vector<double> pa, da, pb, db, pad, dad, pbd, dbd;
                means(A, a0, ia.nlos, pa, da), means(B, b0, ib.nlos, pb, db);
                means(A, a0, det(ia), pad, dad), means(B, b0, det(ib), pbd, dbd);
                qd_xf_order(pad, dad, pa, da, pbd, dbd, pb, db, nr, oa, ob);
                std::vector<double> f_out(nr), f_in(nr);
                for (arma::uword r = 0; r < nr; ++r)
                    f_out[r] = 1.0 - ramp_w[k][r], f_in[r] = ramp_w[k][r];
                w_out[k - 1] = qd_xf_weights(pa, oa, f_out, true);
                w_in[k] = qd_xf_weights(pb, ob, f_in, false);
                for (arma::uword m = 0; m < na; ++m)
                    for (arma::uword r = nr; r-- > 0;)
                        if (w_out[k - 1][m][r] == 0.0)
                            exit_r[m] = r;
                        else
                            break;
                for (arma::uword j = 0; j < nb; ++j)
                    for (arma::uword r = 0; r < nr; ++r)
                        if (w_in[k][j][r] > 0.0)
                        {
                            entry_r[j] = r;
                            break;
                        }
            }
            else
            {
                for (arma::uword j = 0; j < nb; ++j)
                    ob.push_back(j);
            }
            // A new tap takes the slot of an old tap that has faded out before it enters (the lowest such slot), else
            // a slot that the old segment does not use, else a new slot
            std::vector<char> used(n_slot, 0), taken(n_slot, 0);
            for (arma::uword m = 0; m < na; ++m)
                used[slot[k - 1][m]] = 1;
            slot[k].assign(nb, 0);
            for (arma::uword j : ob)
            {
                long long best = -1;
                for (arma::uword m = 0; m < na; ++m)
                    if (!taken[slot[k - 1][m]] && exit_r[m] <= entry_r[j] && (best < 0 || slot[k - 1][m] < (arma::uword)best))
                        best = (long long)slot[k - 1][m];
                for (arma::uword s = 0; best < 0 && s < n_slot; ++s)
                    if (!used[s] && !taken[s])
                        best = (long long)s;
                if (best < 0)
                {
                    best = (long long)n_slot++;
                    used.push_back(0), taken.push_back(0);
                }
                taken[(arma::uword)best] = 1, slot[k][j] = (arma::uword)best;
            }
        }
    }

    // Output object: positions and orientations
    qd_ch c;
    c.name = segments[0].name;
    c.center_freq = segments[0].center_freq;
    bool has_to = false, has_ro = false;
    for (const qd_ch &s : segments)
        has_to = has_to || !s.tx_orientation.is_empty(), has_ro = has_ro || !s.rx_orientation.is_empty();
    c.tx_pos.set_size(3, n_out), c.rx_pos.set_size(3, n_out);
    if (has_to)
        c.tx_orientation.set_size(3, n_out);
    if (has_ro)
        c.rx_orientation.set_size(3, n_out);
    for (arma::uword s = 0; s < n_out; ++s)
    {
        const qd_xf_snap &o = out[s];
        const qd_ch &A = segments[o.a];
        arma::mat *dst[4] = {&c.tx_pos, &c.rx_pos, &c.tx_orientation, &c.rx_orientation};
        const arma::mat *sa[4] = {&A.tx_pos, &A.rx_pos, &A.tx_orientation, &A.rx_orientation};
        for (int m = 0; m < 4; ++m)
        {
            if (dst[m]->is_empty())
                continue;
            arma::vec x = qd_xf_col(*sa[m], o.ia);
            if (o.b >= 0)
            {
                const qd_ch &B = segments[(arma::uword)o.b];
                const arma::mat *sb[4] = {&B.tx_pos, &B.rx_pos, &B.tx_orientation, &B.rx_orientation};
                arma::vec y = qd_xf_col(*sb[m], o.ib);
                for (arma::uword i = 0; i < 3; ++i) // positions linear, angles along the shorter way
                    x[i] += o.w * (m < 2 ? y[i] - x[i] : std::remainder(y[i] - x[i], 2.0 * pi));
            }
            dst[m]->col(s) = x;
        }
    }

    // Path data kept where all segments have them
    bool has_len = rays, has_pm = rays, has_ang = rays, has_fbs = rays, has_lbs = rays;
    for (const qd_ch &s : segments)
    {
        const size_t n = (size_t)s.n_snap();
        has_len = has_len && s.path_length.size() == n, has_pm = has_pm && s.polmat.size() == n;
        has_ang = has_ang && s.path_angles.size() == n, has_fbs = has_fbs && s.fbs_pos.size() == n, has_lbs = has_lbs && s.lbs_pos.size() == n;
    }
    bool flat_delay = true; // all delays [1, 1, n_path]
    for (const qd_ch &s : segments)
        for (const arma::cube &d : s.delay)
            flat_delay = flat_delay && d.n_rows == 1 && d.n_cols == 1;
    const arma::uword n_rx = segments[0].n_rx(), n_tx = segments[0].n_tx();
    c.coeff_re.resize(n_out), c.coeff_im.resize(n_out), c.delay.resize(n_out);
    if (rays)
    {
        c.path_gain.resize(n_out);
        if (has_len)
            c.path_length.resize(n_out);
        if (has_pm)
            c.polmat.resize(n_out);
        if (has_ang)
            c.path_angles.resize(n_out);
        if (has_fbs)
            c.fbs_pos.resize(n_out);
        if (has_lbs)
            c.lbs_pos.resize(n_out);
    }

    // A path in the output: segment, snapshot and path of the source, amplitude scale
    struct src
    {
        const qd_ch *c;
        arma::uword s, p;
        double a;
    };
    // The merged LOS or GR of a ramp snapshot: power (1 - w) P_old + w P_new with the data of the new path (of the old
    // where the new one has no power); P from the path gain (rays) or the coefficients (taps)
    auto merged = [&](const qd_xf_snap &o, long long pa, long long pb, std::vector<src> &v)
    {
        if (pa < 0 && pb < 0)
            return;
        const qd_ch &A = segments[o.a], &B = segments[(arma::uword)o.b];
        auto power = [&](const qd_ch &x, arma::uword s, long long p)
        { return p < 0 ? 0.0 : (rays ? x.path_gain[s][(arma::uword)p] : qd_xf_power(x.coeff_re[s], x.coeff_im[s], (arma::uword)p)); };
        const double g_a = power(A, o.ia, pa), g_b = power(B, o.ib, pb), target = (1.0 - o.w) * g_a + o.w * g_b;
        const bool from_b = pb >= 0 && (g_b > 0.0 || pa < 0);
        const double g = from_b ? g_b : g_a;
        v.push_back({from_b ? &B : &A, from_b ? o.ib : o.ia, (arma::uword)(from_b ? pb : pa), g > 0.0 ? std::sqrt(target / g) : 0.0});
    };

    if (rays)
        for (arma::uword s = 0; s < n_out; ++s)
        {
            const qd_xf_snap &o = out[s];
            const qd_ch &A = segments[o.a];
            if (o.b < 0) // a snapshot of one segment
            {
                c.coeff_re[s] = A.coeff_re[o.ia], c.coeff_im[s] = A.coeff_im[o.ia], c.delay[s] = A.delay[o.ia], c.path_gain[s] = A.path_gain[o.ia];
                if (has_len)
                    c.path_length[s] = A.path_length[o.ia];
                if (has_pm)
                    c.polmat[s] = A.polmat[o.ia];
                if (has_ang)
                    c.path_angles[s] = A.path_angles[o.ia];
                if (has_fbs)
                    c.fbs_pos[s] = A.fbs_pos[o.ia];
                if (has_lbs)
                    c.lbs_pos[s] = A.lbs_pos[o.ia];
                continue;
            }
            std::vector<src> v;
            {
                const qd_ch &B = segments[(arma::uword)o.b];
                const qd_xf_seg &ia = info[o.a], &ib = info[(arma::uword)o.b];
                merged(o, ia.los, ib.los, v);
                merged(o, ia.gr, ib.gr, v);
                auto nlos = [](const qd_xf_seg &g, const qd_ch &x, arma::uword i)
                {
                    if (g.tagged)
                        return g.nlos;
                    std::vector<arma::uword> all(x.coeff_re[i].n_slices);
                    for (arma::uword p = 0; p < all.size(); ++p)
                        all[p] = p;
                    return all;
                };
                for (arma::uword p : nlos(ia, A, o.ia))
                    v.push_back({&A, o.ia, p, std::sqrt(1.0 - o.w)});
                for (arma::uword p : nlos(ib, B, o.ib))
                    v.push_back({&B, o.ib, p, std::sqrt(o.w)});
            }
            const arma::uword n = (arma::uword)v.size();
            bool flat = true;
            for (const src &x : v)
                flat = flat && x.c->delay[x.s].n_rows == 1 && x.c->delay[x.s].n_cols == 1;
            arma::cube re(n_rx, n_tx, n), im(n_rx, n_tx, n), dl(flat ? 1 : n_rx, flat ? 1 : n_tx, n);
            arma::vec pg(n), len(has_len ? n : 0);
            arma::mat pm(has_pm ? 8 : 0, has_pm ? n : 0), ang(has_ang ? n : 0, has_ang ? 4 : 0), fbs(has_fbs ? 3 : 0, has_fbs ? n : 0),
                lbs(has_lbs ? 3 : 0, has_lbs ? n : 0);
            const arma::uword n_e = n_rx * n_tx, n_d = flat ? 1 : n_e;
            for (arma::uword q = 0; q < n; ++q) // raw copies: a ramp snapshot of "rays" has some hundred paths
            {
                const src &x = v[q];
                const double *a = x.c->coeff_re[x.s].slice_memptr(x.p), *b = x.c->coeff_im[x.s].slice_memptr(x.p);
                double *y = re.slice_memptr(q), *z = im.slice_memptr(q), *u = dl.slice_memptr(q);
                for (arma::uword e = 0; e < n_e; ++e)
                    y[e] = x.a * a[e], z[e] = x.a * b[e];
                const arma::cube &d = x.c->delay[x.s];
                const bool full = d.n_rows == n_rx && d.n_cols == n_tx;
                for (arma::uword e = 0; e < n_d; ++e)
                    u[e] = d.slice_memptr(x.p)[full ? e : 0];
                pg[q] = x.a * x.a * x.c->path_gain[x.s].at(x.p);
                if (has_len)
                    len[q] = x.c->path_length[x.s].at(x.p);
                for (arma::uword m = 0; has_pm && m < 8; ++m)
                    pm.at(m, q) = x.a * x.c->polmat[x.s].at(m, x.p);
                for (arma::uword m = 0; has_ang && m < 4; ++m)
                    ang.at(q, m) = x.c->path_angles[x.s].at(x.p, m);
                for (arma::uword m = 0; m < 3; ++m)
                {
                    if (has_fbs)
                        fbs.at(m, q) = x.c->fbs_pos[x.s].at(m, x.p);
                    if (has_lbs)
                        lbs.at(m, q) = x.c->lbs_pos[x.s].at(m, x.p);
                }
            }
            c.coeff_re[s] = std::move(re), c.coeff_im[s] = std::move(im), c.delay[s] = std::move(dl), c.path_gain[s] = std::move(pg);
            if (has_len)
                c.path_length[s] = std::move(len);
            if (has_pm)
                c.polmat[s] = std::move(pm);
            if (has_ang)
                c.path_angles[s] = std::move(ang);
            if (has_fbs)
                c.fbs_pos[s] = std::move(fbs);
            if (has_lbs)
                c.lbs_pos[s] = std::move(lbs);
        }
    else
    {
        // Slots: LOS, GR (if a segment has them), then the NLOS slots; NLOS slots without power are removed
        bool any_los = false, any_gr = false;
        for (const qd_xf_seg &g : info)
            any_los = any_los || g.los >= 0, any_gr = any_gr || g.gr >= 0;
        const arma::uword s_gr = any_los ? 1 : 0, n_det = s_gr + (any_gr ? 1 : 0), n_all = n_det + n_slot;
        std::vector<char> power(n_slot, 0);
        for (arma::uword s = 0; s < n_out; ++s)
        {
            const qd_xf_snap &o = out[s];
            arma::cube re(n_rx, n_tx, n_all, arma::fill::zeros), im = re, dl(flat_delay ? 1 : n_rx, flat_delay ? 1 : n_tx, n_all, arma::fill::zeros);
            auto put = [&](arma::uword slot_out, const src &x)
            {
                if (x.a == 0.0)
                    return;
                re.slice(slot_out) = x.a * x.c->coeff_re[x.s].slice(x.p), im.slice(slot_out) = x.a * x.c->coeff_im[x.s].slice(x.p);
                const arma::cube &d = x.c->delay[x.s];
                if (flat_delay || (d.n_rows == n_rx && d.n_cols == n_tx))
                    dl.slice(slot_out) = d.slice(x.p);
                else
                    dl.slice(slot_out).fill(d(0, 0, x.p));
            };
            const qd_ch &A = segments[o.a];
            const qd_xf_seg &ia = info[o.a];
            if (o.b < 0)
            {
                if (ia.los >= 0)
                    put(0, {&A, o.ia, (arma::uword)ia.los, 1.0});
                if (ia.gr >= 0)
                    put(s_gr, {&A, o.ia, (arma::uword)ia.gr, 1.0});
                for (size_t j = 0; j < ia.nlos.size(); ++j)
                    put(n_det + slot[o.a][j], {&A, o.ia, ia.nlos[j], 1.0});
            }
            else
            {
                const arma::uword kb = (arma::uword)o.b;
                const qd_ch &B = segments[kb];
                const qd_xf_seg &ib = info[kb];
                std::vector<src> v;
                merged(o, ia.los, ib.los, v);
                if (!v.empty())
                    put(0, v.back());
                v.clear();
                merged(o, ia.gr, ib.gr, v);
                if (!v.empty())
                    put(s_gr, v.back());
                for (size_t j = 0; j < ia.nlos.size(); ++j)
                    put(n_det + slot[o.a][j], {&A, o.ia, ia.nlos[j], std::sqrt(w_out[o.a][j][o.r])});
                for (size_t j = 0; j < ib.nlos.size(); ++j)
                    put(n_det + slot[kb][j], {&B, o.ib, ib.nlos[j], std::sqrt(w_in[kb][j][o.r])});
            }
            // Slots without power take the smallest delay of the taps with power, a delay valid in any delay form
            std::vector<char> on(n_all, 0);
            for (arma::uword q = 0; q < n_all; ++q)
            {
                const double *a = re.slice_memptr(q), *b = im.slice_memptr(q);
                for (arma::uword e = 0; e < n_rx * n_tx && !on[q]; ++e)
                    on[q] = a[e] != 0.0 || b[e] != 0.0;
                if (q >= n_det && on[q])
                    power[q - n_det] = 1;
            }
            arma::mat d_min(dl.n_rows, dl.n_cols, arma::fill::value(arma::datum::inf));
            for (arma::uword q = 0; q < n_all; ++q)
                if (on[q])
                    d_min = arma::min(d_min, dl.slice(q));
            if (d_min.is_finite())
                for (arma::uword q = 0; q < n_all; ++q)
                    if (!on[q])
                        dl.slice(q) = d_min;
            c.coeff_re[s] = std::move(re), c.coeff_im[s] = std::move(im), c.delay[s] = std::move(dl);
        }
        std::vector<arma::uword> keep;
        for (arma::uword q = 0; q < n_all; ++q)
            if (q < n_det || power[q - n_det])
                keep.push_back(q);
        if (keep.size() != n_all)
            for (arma::uword s = 0; s < n_out; ++s)
            {
                arma::cube *x[3] = {&c.coeff_re[s], &c.coeff_im[s], &c.delay[s]};
                for (arma::cube *y : x)
                {
                    arma::cube z(y->n_rows, y->n_cols, keep.size());
                    for (size_t q = 0; q < keep.size(); ++q)
                        z.slice(q) = y->slice(keep[q]);
                    *y = std::move(z);
                }
            }
    }
    return c;
}
