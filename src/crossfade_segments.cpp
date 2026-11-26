// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"
#include "qd_helpers_channel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# crossfade_segments
Join the channels of consecutive segments of one link by a cross-fade

- Joins the "rays" channels of the segments of one link, e.g. of a track whose scenario or indoor state changes, into
  one channel; consecutive segments share the snapshots of a ramp, over which the clusters of the old segment are
  exchanged for those of the new one
- [[get_channels_from_rays]] joins the segments of tracks with this function, the ramp centred on the row of the change

## Declaration:
```
channel<double> quadriga_lib::crossfade_segments(
    const std::vector<channel<double>> &segments,
    const arma::uvec &n_overlap = {},
    const std::string &output = "taps");
```

## Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients and path gains (`"rays"`), the same
  number of paths in every snapshot, equal `n_rx`, `n_tx` and `center_freq` (at most one value); `[n_segment]`
- **`n_overlap`** *(optional)* — Number of snapshots at the start of each segment that are also the last snapshots of
  the preceding segment (computed at the same positions); they form the ramp. The first entry is 0, at most the
  snapshots of the preceding segment after its own shared ones; omitted: the snapshots whose Tx and Rx positions match
  (within 1 µm), an error where the positions repeat (e.g. ends that do not move); `[n_segment]`
- **`output`** *(optional)* — `"taps"` (the paths summed per tap index) or `"rays"`

## Returns:
- The joined channel, see [[channel]]

## Notes:
- **Snapshots**: segment $k$ starts with the `n_overlap[k]` snapshots of the ramp, which are also the last ones of
  segment $k - 1$; the output has $\sum_k (n_{\mathrm{snap},k} - n_{\mathrm{overlap},k})$ snapshots. Outside the
  ramps the snapshots are those of one segment
- **Paths**: `par_data` entries of [[gbsm_single_link_channel]]: `path_type` gives the LOS and GR paths (without it, tap
  index 0 of `tap` and a tap index 3 held by one path, without index 4 or 5, as [[ssf_rays]] numbers them), `cluster`
  the clusters, whose paths fade together (without it: the taps, without `tap`: each path)
- **Level**: at ramp snapshot $i$ of $n_r$, with $P_a$, $P_b$ the summed path gains of the old and new segment,
  $$G_i = P_{a,i}^{1 - w_i} P_{b,i}^{w_i}, \quad w_i = \frac{1}{2} \left( 1 + \sin\left( \left( \frac{i}{n_r + 1} - \frac{1}{2} \right) \pi \right) \right)$$
- **Exchange**: both cluster sets are scaled to $G$; the clusters of the old segment fade out and those of the new one
  in, one after another, each over a quarter of the exchange, in an order that keeps the delay and angular spreads close
  to those of the superposition of both sets (a segment with more than 32 clusters keeps its cluster order). The
  exchange takes the part of the ramp in which $G$ is within 10 dB of the weaker segment, so that no cluster exceeds its
  own power by more than 10 dB
- **LOS and GR**: one path each, with the data of the new segment (of the old one where the new one has no power),
  scaled to $G_i \left( (1 - q_i) g_a / P_a + q_i g_b / P_b \right)$, $g$ the path gain of the path and $q_i$ the
  exchanged share
- **A segment without power** at a ramp snapshot: all paths of the transition fade with $1 - w$ (old) and $w$ (new)
  in power
- **Taps**: the weighted paths summed per tap index, one set of tap slots for all snapshots (LOS, GR, NLOS); a new tap
  takes the slot of an old tap that has faded out before it enters. Slots without power hold zero coefficients and the
  smallest delay of the snapshot. The delay of a tap is the mean of its paths' delays weighted by their path gains
- **Other data**: delays as given (one reference for all segments); positions and orientations of the old segment;
  `name` and `center_freq` of segment 0; `"rays"` keep `path_length`, `polmat`, `path_angles`, `fbs_pos` and
  `lbs_pos` where all segments have them; `par_data`, `n_interact` and `interact_coord` are not carried

## See also:
- [[get_channels_from_rays]] (joins the segments of tracks with this function)
- [[gbsm_single_link_channel]] (channels of one segment)
MD!*/

namespace
{
using qd_ch = quadriga_lib::channel<double>;
using qd_cx = std::complex<double>;
constexpr double qd_xf_rho = 0.25;     // Part of the exchange over which one cluster fades
constexpr double qd_xf_excess = 10.0;  // Largest power of a cluster above its own power in [dB]
constexpr double qd_xf_tie = 1e-9;     // Relative cost difference below which the earlier position of the order search stays
constexpr arma::uword qd_xf_n_eval = 16; // Evaluation points of the order search
constexpr size_t qd_xf_max_search = 32; // Segments with more clusters keep their cluster order (the search grows with n^3)

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

// Paths of a segment: LOS and GR (-1 = none), the NLOS paths and their cluster (index into 'keys', the sorted clusters)
struct qd_xf_seg
{
    long long los = -1, gr = -1;
    std::vector<arma::uword> nlos, cluster;
    std::vector<unsigned> keys;
    arma::uword n_path = 0;
};

// Classifies the paths of segment k: types from "path_type", else from "tap" (LOS tap 0; GR a tap 3 held by one path
// without the taps 4 and 5), else all NLOS; clusters from "cluster", else "tap", else one per path
qd_xf_seg qd_xf_classify(const qd_ch &c, arma::uword k)
{
    qd_xf_seg g;
    const arma::uword n = c.coeff_re[0].n_slices;
    g.n_path = n;
    for (const arma::cube &x : c.coeff_re)
        if (x.n_slices != n)
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'segments' must have the same number of paths in every snapshot.");
    const arma::u32_vec *type = qd_xf_par(c, "path_type", k), *tap = qd_xf_par(c, "tap", k), *cl = qd_xf_par(c, "cluster", k);
    for (const auto &e : {std::make_pair("path_type", type), std::make_pair("tap", tap), std::make_pair("cluster", cl)})
        if (e.second != nullptr && e.second->n_elem != n)
            throw std::invalid_argument("Entry " + std::to_string(k) + ": the par_data entry '" + e.first + "' of 'segments' must have one value per path in every snapshot.");
    if (type != nullptr && arma::any(*type > 2))
        throw std::invalid_argument("Entry " + std::to_string(k) + ": the par_data entry 'path_type' of 'segments' must hold 0 (LOS), 1 (GR) or 2 (NLOS).");
    arma::uword n3 = 0;
    bool sub3 = false;
    if (type == nullptr && tap != nullptr)
        for (arma::uword p = 0; p < n; ++p)
            n3 += (*tap)[p] == 3 ? 1 : 0, sub3 = sub3 || (*tap)[p] == 4 || (*tap)[p] == 5;
    std::vector<unsigned> key;
    for (arma::uword p = 0; p < n; ++p)
    {
        const unsigned t = type != nullptr ? (*type)[p] : (tap == nullptr ? 2u : ((*tap)[p] == 0 ? 0u : ((*tap)[p] == 3 && n3 == 1 && !sub3 ? 1u : 2u)));
        if (t == 0 && g.los < 0)
            g.los = (long long)p;
        else if (t == 1 && g.gr < 0)
            g.gr = (long long)p;
        else
            g.nlos.push_back(p), key.push_back(cl != nullptr ? (*cl)[p] : (tap != nullptr ? (*tap)[p] : (unsigned)p));
    }
    g.keys = key;
    std::sort(g.keys.begin(), g.keys.end());
    g.keys.erase(std::unique(g.keys.begin(), g.keys.end()), g.keys.end());
    for (unsigned x : key)
        g.cluster.push_back((arma::uword)(std::lower_bound(g.keys.begin(), g.keys.end(), x) - g.keys.begin()));
    return g;
}

// Mean delay of path p of a delay snapshot over the element pairs
double qd_xf_delay(const arma::cube &d, arma::uword p)
{
    return arma::accu(d.slice(p)) / (double)(d.n_rows * d.n_cols);
}

// Additive moments: power, power x delay, power x delay^2, power x exp(j angle) for AOD, EOD, AOA, EOA
struct qd_xf_m
{
    double m0 = 0.0, m1 = 0.0, m2 = 0.0;
    std::array<qd_cx, 4> z{};
    void add(const qd_xf_m &x, double w)
    {
        m0 += w * x.m0, m1 += w * x.m1, m2 += w * x.m2;
        for (int i = 0; i < 4; ++i)
            z[i] += w * x.z[i];
    }
};

// Ray statistics of one side of a transition: per ramp snapshot the level P and the path gains of LOS and GR; the moments
// of each cluster (of the data segment) and of LOS + GR, means over the ramp divided by the mean of P
struct qd_xf_stat
{
    std::vector<double> P, g_los, g_gr;
    std::vector<qd_xf_m> cluster;
    qd_xf_m det;
    bool angles = false;
};

// Statistics of the snapshots [s0, s0 + n_r) of the "rays" channel c for the clusters 'keys' of the data segment
qd_xf_stat qd_xf_stats(const qd_ch &c, arma::uword k, arma::uword s0, arma::uword n_r, const std::vector<unsigned> &keys)
{
    if (c.path_gain.size() != c.n_snap() || s0 + n_r > c.n_snap())
        throw std::invalid_argument("Entry " + std::to_string(k) + ": the ray statistics of 'segments' do not cover the ramp.");
    const qd_xf_seg g = qd_xf_classify(c, k);
    qd_xf_stat st;
    st.angles = c.path_angles.size() == c.n_snap();
    st.P.assign(n_r, 0.0), st.g_los.assign(n_r, 0.0), st.g_gr.assign(n_r, 0.0);
    st.cluster.assign(keys.size(), qd_xf_m());
    std::vector<long long> c_of(g.nlos.size(), -1); // cluster of each NLOS path of c in 'keys'
    for (size_t j = 0; j < g.nlos.size(); ++j)
    {
        const unsigned key = g.keys[g.cluster[j]];
        const auto it = std::lower_bound(keys.begin(), keys.end(), key);
        if (it != keys.end() && *it == key)
            c_of[j] = (long long)(it - keys.begin());
    }
    auto moment = [&](arma::uword s, arma::uword p)
    {
        qd_xf_m m;
        const double pg = c.path_gain[s][p], d = qd_xf_delay(c.delay[s], p);
        m.m0 = pg, m.m1 = pg * d, m.m2 = pg * d * d;
        for (int i = 0; st.angles && i < 4; ++i)
            m.z[i] = pg * std::polar(1.0, c.path_angles[s](p, (arma::uword)i));
        return m;
    };
    for (arma::uword r = 0; r < n_r; ++r)
    {
        const arma::uword s = s0 + r;
        st.P[r] = arma::accu(c.path_gain[s]);
        if (g.los >= 0)
            st.g_los[r] = c.path_gain[s][(arma::uword)g.los], st.det.add(moment(s, (arma::uword)g.los), 1.0);
        if (g.gr >= 0)
            st.g_gr[r] = c.path_gain[s][(arma::uword)g.gr], st.det.add(moment(s, (arma::uword)g.gr), 1.0);
        for (size_t j = 0; j < g.nlos.size(); ++j)
            if (c_of[j] >= 0)
                st.cluster[(size_t)c_of[j]].add(moment(s, g.nlos[j]), 1.0);
    }
    double Pm = 0.0;
    for (double x : st.P)
        Pm += x / (double)n_r;
    const double f = Pm > 0.0 ? 1.0 / (Pm * (double)n_r) : 0.0;
    for (qd_xf_m &m : st.cluster)
    {
        qd_xf_m x;
        x.add(m, f), m = x;
    }
    qd_xf_m x;
    x.add(st.det, f), st.det = x;
    return st;
}

// Exchanged part e of each cluster for the shares s (in fade order) at the exchanged share t: cluster i of the order fades over
// the window [(1 - rho) F0, (1 - rho) F1 + rho] of a parameter v (F0, F1: cumulative shares before and with it), v such
// that the sum of s e is t; e[i] in the order of s
void qd_xf_exchange(const std::vector<double> &s, double t, std::vector<double> &e)
{
    const size_t n = s.size();
    e.assign(n, 0.0);
    if (n == 0)
        return;
    std::vector<double> lo(n), hi(n), v{0.0, 1.0};
    double F = 0.0;
    for (size_t i = 0; i < n; ++i)
        lo[i] = (1.0 - qd_xf_rho) * F, F += s[i], hi[i] = (1.0 - qd_xf_rho) * F + qd_xf_rho, v.push_back(lo[i]), v.push_back(hi[i]);
    auto part = [&](size_t i, double x) { return std::min(std::max((x - lo[i]) / (hi[i] - lo[i]), 0.0), 1.0); };
    auto E = [&](double x)
    {
        double y = 0.0;
        for (size_t i = 0; i < n; ++i)
            y += s[i] * part(i, x);
        return y;
    };
    double x = 0.0;
    if (t >= 1.0)
        x = 1.0;
    else if (t > 0.0) // E is non-decreasing and linear between the breakpoints: the first breakpoint with E >= t
    {
        std::sort(v.begin(), v.end());
        size_t a = 0, b = v.size() - 1; // E(v[a]) < t <= E(v[b]) = 1
        while (b - a > 1)
        {
            const size_t m = (a + b) / 2;
            (E(v[m]) >= t ? b : a) = m;
        }
        const double E0 = E(v[a]), E1 = E(v[b]);
        x = E1 > E0 ? v[a] + (t - E0) / (E1 - E0) * (v[b] - v[a]) : v[a];
    }
    for (size_t i = 0; i < n; ++i)
        e[i] = part(i, x);
}

// Delay spread and the four circular angular spreads (TR 38.901 Annex A.1) of summed moments; n_m = 5 or 1 (DS only)
std::array<double, 5> qd_xf_spreads(const qd_xf_m &m, int n_m)
{
    std::array<double, 5> y{};
    const double s = std::max(m.m0, 1e-300);
    y[0] = std::sqrt(std::max(m.m2 / s - (m.m1 / s) * (m.m1 / s), 0.0));
    for (int i = 1; i < n_m; ++i)
        y[(size_t)i] = std::sqrt(-2.0 * std::log(std::min(std::max(std::abs(m.z[(size_t)i - 1]) / s, 1e-300), 1.0)));
    return y;
}

// Exchanged share of the new segment at the ramp value w: the part f of the ramp in which the level is within X dB of the
// weaker segment, at its end (new segment weaker) or start
double qd_xf_share(double w, double f, bool new_weaker)
{
    return std::min(std::max(new_weaker ? (w - (1.0 - f)) / f : w / f, 0.0), 1.0);
}

// Part (u) of each cluster present at the share t for the fade order O and the cluster powers m0: old clusters 1 - e, new ones e
void qd_xf_present(const std::vector<qd_xf_m> &m, const std::vector<arma::uword> &O, double t, bool old, std::vector<double> &u)
{
    const size_t n = O.size();
    double P = 0.0;
    for (const qd_xf_m &x : m)
        P += x.m0;
    u.assign(n, old ? 1.0 - t : t);
    if (P <= 0.0)
        return;
    std::vector<double> s(n), e;
    for (size_t i = 0; i < n; ++i)
        s[i] = m[O[i]].m0 / P;
    qd_xf_exchange(s, t, e);
    for (size_t i = 0; i < n; ++i)
        u[O[i]] = old ? 1.0 - e[i] : e[i];
}

// Fade orders of the old (a) and new (b) clusters: keeps the spreads of the present clusters close to those of the
// superposition of both sets at 16 points of the ramp; start in cluster order, two passes of moving each cluster to its best
// position, of positions within the relative tie tolerance the earlier one; a segment with more than 32 clusters keeps
// the cluster order
void qd_xf_order(const qd_xf_stat &a, const qd_xf_stat &b, double f, bool new_weaker, std::vector<arma::uword> &oa, std::vector<arma::uword> &ob)
{
    const size_t na = a.cluster.size(), nb = b.cluster.size();
    oa.resize(na), ob.resize(nb);
    for (size_t i = 0; i < na; ++i)
        oa[i] = i;
    for (size_t i = 0; i < nb; ++i)
        ob[i] = i;
    if (na < 2 && nb < 2)
        return;
    const int n_m = a.angles && b.angles ? 5 : 1;
    qd_xf_m Sa = a.det, Sb = b.det;
    for (const qd_xf_m &x : a.cluster)
        Sa.add(x, 1.0);
    for (const qd_xf_m &x : b.cluster)
        Sb.add(x, 1.0);
    std::vector<double> q(qd_xf_n_eval);
    std::vector<std::array<double, 5>> T(qd_xf_n_eval);
    std::array<double, 5> T2{};
    const double pi = arma::datum::pi;
    for (arma::uword i = 0; i < qd_xf_n_eval; ++i)
    {
        const double x = ((double)i + 0.5) / (double)qd_xf_n_eval, w = 0.5 * (1.0 + std::sin((x - 0.5) * pi));
        q[i] = qd_xf_share(w, f, new_weaker);
        qd_xf_m M;
        M.add(Sa, 1.0 - q[i]), M.add(Sb, q[i]);
        T[i] = qd_xf_spreads(M, n_m);
        for (int j = 0; j < n_m; ++j)
            T2[(size_t)j] += T[i][(size_t)j] * T[i][(size_t)j] / (double)qd_xf_n_eval;
    }
    std::vector<double> ua, ub;
    auto cost = [&](const std::vector<arma::uword> &A, const std::vector<arma::uword> &B)
    {
        std::array<double, 5> err{};
        for (arma::uword i = 0; i < qd_xf_n_eval; ++i)
        {
            qd_xf_present(a.cluster, A, q[i], true, ua), qd_xf_present(b.cluster, B, q[i], false, ub);
            qd_xf_m M;
            M.add(a.det, 1.0 - q[i]), M.add(b.det, q[i]);
            for (size_t j = 0; j < na; ++j)
                M.add(a.cluster[j], ua[j]);
            for (size_t j = 0; j < nb; ++j)
                M.add(b.cluster[j], ub[j]);
            const std::array<double, 5> Y = qd_xf_spreads(M, n_m);
            for (int j = 0; j < n_m; ++j)
                err[(size_t)j] += (Y[(size_t)j] - T[i][(size_t)j]) * (Y[(size_t)j] - T[i][(size_t)j]) / (double)qd_xf_n_eval;
        }
        double c = 0.0;
        for (int j = 0; j < n_m; ++j)
            c += err[(size_t)j] / std::max(T2[(size_t)j], 1e-300);
        return c;
    };
    auto improve = [&](std::vector<arma::uword> &O, bool old)
    {
        const size_t n = O.size();
        if (n < 2 || n > qd_xf_max_search)
            return;
        for (arma::uword c = 0; c < n; ++c)
        {
            std::vector<arma::uword> rest;
            for (arma::uword x : O)
                if (x != c)
                    rest.push_back(x);
            double best = 0.0;
            size_t at = 0;
            for (size_t pos = 0; pos < n; ++pos)
            {
                std::vector<arma::uword> X(rest.begin(), rest.begin() + (std::ptrdiff_t)pos);
                X.push_back(c);
                X.insert(X.end(), rest.begin() + (std::ptrdiff_t)pos, rest.end());
                const double v = old ? cost(X, ob) : cost(oa, X);
                if (pos == 0 || v < best - qd_xf_tie * best)
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

// Schedule of one transition: power weights of the clusters of the old (wa) and new (wb) segment, [cluster][r]; LOS and GR:
// source (new segment) and amplitude scale per ramp snapshot
struct qd_xf_sched
{
    std::vector<std::vector<double>> wa, wb;
    std::vector<char> los_b, gr_b;
    std::vector<double> los_a, gr_a;
};

// The schedule from the statistics of the two sides (sa, sb) and those for the fade order (oa_s, ob_s); da, db: the data
// segments; w: the ramp
qd_xf_sched qd_xf_schedule(const qd_xf_stat &sa, const qd_xf_stat &sb, const qd_xf_stat &oa_s, const qd_xf_stat &ob_s,
                           const qd_xf_seg &da, const qd_xf_seg &db, const std::vector<double> &w)
{
    const size_t n_r = w.size(), na = da.keys.size(), nb = db.keys.size();
    qd_xf_sched S;
    S.wa.assign(na, std::vector<double>(n_r)), S.wb.assign(nb, std::vector<double>(n_r));
    S.los_b.assign(n_r, 0), S.gr_b.assign(n_r, 0), S.los_a.assign(n_r, 0.0), S.gr_a.assign(n_r, 0.0);
    // Exchange range from the levels of two sides: false if a side has no power at a ramp snapshot, else the part f of the
    // ramp in which the level is within X dB of the weaker segment and whether that is the new one
    auto range = [n_r](const qd_xf_stat &x, const qd_xf_stat &y, double &f, bool &new_weaker)
    {
        double d_max = 0.0, d_mean = 0.0;
        for (size_t r = 0; r < n_r; ++r)
        {
            if (!(x.P[r] > 0.0) || !(y.P[r] > 0.0))
                return false;
            const double d = 10.0 * std::log10(y.P[r] / x.P[r]);
            d_max = std::max(d_max, std::abs(d)), d_mean += d / (double)n_r;
        }
        f = d_max <= qd_xf_excess ? 1.0 : qd_xf_excess / d_max, new_weaker = d_mean < 0.0;
        return true;
    };
    double f = 1.0, f_o = 1.0;
    bool new_weaker = false, new_weaker_o = false;
    const bool fallback = !range(sa, sb, f, new_weaker);
    std::vector<double> G(n_r), q(n_r);
    std::vector<arma::uword> oa, ob;
    if (!fallback) // the order from the statistics for the order (of frequency 0 for ray sets shared by all frequencies)
        range(oa_s, ob_s, f_o, new_weaker_o) ? qd_xf_order(oa_s, ob_s, f_o, new_weaker_o, oa, ob) : qd_xf_order(sa, sb, f, new_weaker, oa, ob);
    std::vector<double> ua, ub;
    for (size_t r = 0; r < n_r; ++r)
    {
        if (fallback) // the union at own levels
        {
            for (size_t j = 0; j < na; ++j)
                S.wa[j][r] = 1.0 - w[r];
            for (size_t j = 0; j < nb; ++j)
                S.wb[j][r] = w[r];
        }
        else
        {
            G[r] = std::pow(sa.P[r], 1.0 - w[r]) * std::pow(sb.P[r], w[r]), q[r] = qd_xf_share(w[r], f, new_weaker);
            qd_xf_present(sa.cluster, oa, q[r], true, ua), qd_xf_present(sb.cluster, ob, q[r], false, ub);
            for (size_t j = 0; j < na; ++j)
                S.wa[j][r] = G[r] / sa.P[r] * ua[j];
            for (size_t j = 0; j < nb; ++j)
                S.wb[j][r] = G[r] / sb.P[r] * ub[j];
        }
        // LOS and GR: one path with the data of the new segment where it has power, scaled to the target power
        auto det = [&](long long pa, long long pb, double ga, double gb, char &from_b, double &amp_)
        {
            if (pa < 0 && pb < 0)
                return;
            const double target = fallback ? (1.0 - w[r]) * ga + w[r] * gb : G[r] * ((1.0 - q[r]) * ga / sa.P[r] + q[r] * gb / sb.P[r]);
            from_b = pb >= 0 && (gb > 0.0 || pa < 0);
            const double g = from_b ? gb : ga;
            amp_ = g > 0.0 ? std::sqrt(target / g) : 0.0;
        };
        det(da.los, db.los, sa.g_los[r], sb.g_los[r], S.los_b[r], S.los_a[r]);
        det(da.gr, db.gr, sa.g_gr[r], sb.g_gr[r], S.gr_b[r], S.gr_a[r]);
    }
    return S;
}

// "taps" of a "rays" channel: the paths summed per tap index (ascending; without "tap": one tap per path), the delay per
// element pair the mean of the paths' delays weighted by their path gains (equal weights where these sum to 0); par_data:
// tap, path_type (the type of its paths, 2 if they differ), cluster (of its first path)
qd_ch qd_xf_taps(const qd_ch &c, arma::uword k)
{
    const qd_xf_seg g = qd_xf_classify(c, k);
    const arma::uword n = g.n_path;
    const arma::u32_vec *tp = qd_xf_par(c, "tap", k), *cl = qd_xf_par(c, "cluster", k);
    arma::u32_vec tap = tp != nullptr ? *tp : arma::regspace<arma::u32_vec>(0, (unsigned)n - 1);
    arma::u32_vec type(n, arma::fill::value(2)), clus(n);
    if (g.los >= 0)
        type[(arma::uword)g.los] = 0;
    if (g.gr >= 0)
        type[(arma::uword)g.gr] = 1;
    for (arma::uword p = 0; p < n; ++p)
        clus[p] = cl != nullptr ? (*cl)[p] : tap[p];
    const arma::u32_vec val = arma::unique(tap);
    const arma::uword nt = val.n_elem;
    arma::uvec q_of(n);
    arma::u32_vec t_type(nt), t_clus(nt), cnt(nt, arma::fill::zeros);
    for (arma::uword p = 0; p < n; ++p)
    {
        const arma::uword q = (arma::uword)(std::lower_bound(val.begin(), val.end(), tap[p]) - val.begin());
        q_of[p] = q;
        t_type[q] = cnt[q] == 0 || t_type[q] == type[p] ? type[p] : 2;
        if (cnt[q] == 0)
            t_clus[q] = clus[p];
        ++cnt[q];
    }
    qd_ch t;
    t.name = c.name, t.center_freq = c.center_freq;
    t.tx_pos = c.tx_pos, t.rx_pos = c.rx_pos, t.tx_orientation = c.tx_orientation, t.rx_orientation = c.rx_orientation;
    const arma::uword n_rx = c.n_rx(), n_tx = c.n_tx();
    for (arma::uword s = 0; s < c.n_snap(); ++s)
    {
        const arma::cube &d = c.delay[s];
        arma::cube re(n_rx, n_tx, nt, arma::fill::zeros), im = re, dl(d.n_rows, d.n_cols, nt, arma::fill::zeros);
        arma::vec ws(nt, arma::fill::zeros);
        for (arma::uword p = 0; p < n; ++p)
            ws[q_of[p]] += c.path_gain[s][p];
        for (arma::uword p = 0; p < n; ++p)
        {
            const arma::uword q = q_of[p];
            re.slice(q) += c.coeff_re[s].slice(p), im.slice(q) += c.coeff_im[s].slice(p);
            dl.slice(q) += (ws[q] > 0.0 ? c.path_gain[s][p] / ws[q] : 1.0 / (double)cnt[q]) * d.slice(p);
        }
        t.coeff_re.push_back(std::move(re)), t.coeff_im.push_back(std::move(im)), t.delay.push_back(std::move(dl));
    }
    t.par_names = {"tap", "path_type", "cluster"}, t.par_data = {val, t_type, t_clus};
    return t;
}

// Column s of a [3, 1 or n] matrix, zeros if empty
arma::vec qd_xf_col(const arma::mat &M, arma::uword s)
{
    if (M.is_empty())
        return arma::vec(3, arma::fill::zeros);
    return M.col(M.n_cols == 1 ? 0 : s);
}

// One output snapshot: snapshot ia of segment a, or ramp snapshot r between a and b = a + 1 (snapshot ib of b)
struct qd_xf_snap
{
    arma::uword a, ia;
    long long b;
    arma::uword ib, r;
};
} // namespace

quadriga_lib::channel<double> qd_crossfade(const std::vector<const quadriga_lib::channel<double> *> &data,
                                           const std::vector<const quadriga_lib::channel<double> *> &stats,
                                           const std::vector<const quadriga_lib::channel<double> *> *order_stats,
                                           const arma::uvec &trail0, const arma::uvec &order_trail0, const arma::uvec &n_overlap, bool taps)
{
    const arma::uword n_seg = (arma::uword)data.size();
    const double pi = arma::datum::pi;

    // Data in the output form ("taps" from "rays" by summing per tap), path classes
    std::vector<qd_ch> own;
    std::vector<const qd_ch *> D(data);
    if (taps)
        for (arma::uword k = 0; k < n_seg; ++k)
            if (!data[k]->path_gain.empty())
                own.reserve(n_seg), own.push_back(qd_xf_taps(*data[k], k)), D[k] = &own.back();
    std::vector<qd_xf_seg> info(n_seg);
    for (arma::uword k = 0; k < n_seg; ++k)
        info[k] = qd_xf_classify(*D[k], k);

    // Schedules of the transitions (k from k - 1 to k)
    std::vector<qd_xf_sched> sched(n_seg);
    std::vector<std::vector<double>> ramp_w(n_seg);
    for (arma::uword k = 1; k < n_seg; ++k)
    {
        const arma::uword n_r = n_overlap[k];
        if (n_r == 0)
            continue;
        for (arma::uword r = 0; r < n_r; ++r)
            ramp_w[k].push_back(0.5 * (1.0 + std::sin(((double)(r + 1) / (double)(n_r + 1) - 0.5) * pi)));
        const qd_xf_stat sa = qd_xf_stats(*stats[k - 1], k - 1, trail0[k - 1], n_r, info[k - 1].keys), sb = qd_xf_stats(*stats[k], k, 0, n_r, info[k].keys);
        if (order_stats == nullptr)
            sched[k] = qd_xf_schedule(sa, sb, sa, sb, info[k - 1], info[k], ramp_w[k]);
        else
        {
            const qd_xf_stat oa = qd_xf_stats(*(*order_stats)[k - 1], k - 1, order_trail0[k - 1], n_r, info[k - 1].keys),
                             ob = qd_xf_stats(*(*order_stats)[k], k, 0, n_r, info[k].keys);
            sched[k] = qd_xf_schedule(sa, sb, oa, ob, info[k - 1], info[k], ramp_w[k]);
        }
    }

    // Output snapshots
    std::vector<qd_xf_snap> out;
    for (arma::uword k = 0; k < n_seg; ++k)
    {
        const arma::uword S = D[k]->n_snap(), nr = k + 1 < n_seg ? n_overlap[k + 1] : 0;
        for (arma::uword i = n_overlap[k]; i < S - nr; ++i)
            out.push_back({k, i, -1, 0, 0});
        for (arma::uword r = 0; r < nr; ++r)
            out.push_back({k, S - nr + r, (long long)k + 1, r, r});
    }
    const arma::uword n_out = (arma::uword)out.size();

    // Output object: positions and orientations of the old segment
    qd_ch c;
    c.name = data[0]->name;
    c.center_freq = data[0]->center_freq;
    bool has_to = false, has_ro = false;
    for (const qd_ch *s : D)
        has_to = has_to || !s->tx_orientation.is_empty(), has_ro = has_ro || !s->rx_orientation.is_empty();
    c.tx_pos.set_size(3, n_out), c.rx_pos.set_size(3, n_out);
    if (has_to)
        c.tx_orientation.set_size(3, n_out);
    if (has_ro)
        c.rx_orientation.set_size(3, n_out);
    for (arma::uword s = 0; s < n_out; ++s)
    {
        const qd_ch &A = *D[out[s].a];
        c.tx_pos.col(s) = qd_xf_col(A.tx_pos, out[s].ia), c.rx_pos.col(s) = qd_xf_col(A.rx_pos, out[s].ia);
        if (has_to)
            c.tx_orientation.col(s) = qd_xf_col(A.tx_orientation, out[s].ia);
        if (has_ro)
            c.rx_orientation.col(s) = qd_xf_col(A.rx_orientation, out[s].ia);
    }
    const arma::uword n_rx = D[0]->n_rx(), n_tx = D[0]->n_tx();
    c.coeff_re.resize(n_out), c.coeff_im.resize(n_out), c.delay.resize(n_out);

    // A path of the output: segment, snapshot, path and amplitude scale
    struct src
    {
        const qd_ch *c;
        arma::uword s, p;
        double a;
    };
    // The paths of output snapshot o with their scales (LOS, GR, NLOS of the old segment, NLOS of the new one); a slot per
    // path from 'slot' (taps), weight-0 NLOS paths left out (rays)
    auto paths = [&](const qd_xf_snap &o, std::vector<src> &v, std::vector<long long> &kind)
    {
        v.clear(), kind.clear();
        const qd_ch &A = *D[o.a];
        const qd_xf_seg &ia = info[o.a];
        if (o.b < 0)
        {
            if (ia.los >= 0)
                v.push_back({&A, o.ia, (arma::uword)ia.los, 1.0}), kind.push_back(-1);
            if (ia.gr >= 0)
                v.push_back({&A, o.ia, (arma::uword)ia.gr, 1.0}), kind.push_back(-2);
            for (size_t j = 0; j < ia.nlos.size(); ++j)
                v.push_back({&A, o.ia, ia.nlos[j], 1.0}), kind.push_back((long long)j);
            return;
        }
        const arma::uword kb = (arma::uword)o.b;
        const qd_ch &B = *D[kb];
        const qd_xf_seg &ib = info[kb];
        const qd_xf_sched &S = sched[kb];
        if (ia.los >= 0 || ib.los >= 0)
            v.push_back(S.los_b[o.r] ? src{&B, o.ib, (arma::uword)ib.los, S.los_a[o.r]} : src{&A, o.ia, (arma::uword)ia.los, S.los_a[o.r]}), kind.push_back(-1);
        if (ia.gr >= 0 || ib.gr >= 0)
            v.push_back(S.gr_b[o.r] ? src{&B, o.ib, (arma::uword)ib.gr, S.gr_a[o.r]} : src{&A, o.ia, (arma::uword)ia.gr, S.gr_a[o.r]}), kind.push_back(-2);
        for (size_t j = 0; j < ia.nlos.size(); ++j)
            v.push_back({&A, o.ia, ia.nlos[j], std::sqrt(S.wa[ia.cluster[j]][o.r])}), kind.push_back((long long)j);
        for (size_t j = 0; j < ib.nlos.size(); ++j)
            v.push_back({&B, o.ib, ib.nlos[j], std::sqrt(S.wb[ib.cluster[j]][o.r])}), kind.push_back((long long)(ia.nlos.size() + j));
    };
    std::vector<src> v;
    std::vector<long long> kind;

    if (!taps)
    {
        bool has_len = true, has_pm = true, has_ang = true, has_fbs = true, has_lbs = true;
        for (const qd_ch *s : D)
        {
            const size_t n = (size_t)s->n_snap();
            has_len = has_len && s->path_length.size() == n, has_pm = has_pm && s->polmat.size() == n;
            has_ang = has_ang && s->path_angles.size() == n, has_fbs = has_fbs && s->fbs_pos.size() == n, has_lbs = has_lbs && s->lbs_pos.size() == n;
        }
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
        for (arma::uword s = 0; s < n_out; ++s)
        {
            const qd_xf_snap &o = out[s];
            if (o.b < 0) // a snapshot of one segment
            {
                const qd_ch &A = *D[o.a];
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
            paths(o, v, kind);
            std::vector<src> u;
            for (size_t i = 0; i < v.size(); ++i)
                if (v[i].a > 0.0 || kind[i] < 0) // LOS and GR are kept, also at zero power
                    u.push_back(v[i]);
            const arma::uword n = (arma::uword)u.size();
            bool flat = true;
            for (const src &x : u)
                flat = flat && x.c->delay[x.s].n_rows == 1 && x.c->delay[x.s].n_cols == 1;
            arma::cube re(n_rx, n_tx, n), im(n_rx, n_tx, n), dl(flat ? 1 : n_rx, flat ? 1 : n_tx, n);
            arma::vec pg(n), len(has_len ? n : 0);
            arma::mat pm(has_pm ? 8 : 0, has_pm ? n : 0), ang(has_ang ? n : 0, has_ang ? 4 : 0), fbs(has_fbs ? 3 : 0, has_fbs ? n : 0),
                lbs(has_lbs ? 3 : 0, has_lbs ? n : 0);
            const arma::uword n_e = n_rx * n_tx, n_d = flat ? 1 : n_e;
            for (arma::uword q = 0; q < n; ++q) // raw copies: a ramp snapshot of "rays" has some hundred paths
            {
                const src &x = u[q];
                const double *a = x.c->coeff_re[x.s].slice_memptr(x.p), *b = x.c->coeff_im[x.s].slice_memptr(x.p);
                double *y = re.slice_memptr(q), *z = im.slice_memptr(q), *w = dl.slice_memptr(q);
                for (arma::uword e = 0; e < n_e; ++e)
                    y[e] = x.a * a[e], z[e] = x.a * b[e];
                const arma::cube &d = x.c->delay[x.s];
                const bool full = d.n_rows == n_rx && d.n_cols == n_tx;
                for (arma::uword e = 0; e < n_d; ++e)
                    w[e] = d.slice_memptr(x.p)[full ? e : 0];
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
        return c;
    }

    // Taps: slots of the NLOS taps of each segment. A new tap takes the lowest slot of an old tap whose weight is 0 from
    // the first ramp snapshot on at which the new tap has a weight > 0, else a slot that the old segment does not use, else
    // a new slot; new taps in the order of their entry (then of their index)
    std::vector<std::vector<arma::uword>> slot(n_seg);
    arma::uword n_slot = (arma::uword)info[0].nlos.size();
    for (arma::uword j = 0; j < n_slot; ++j)
        slot[0].push_back(j);
    for (arma::uword k = 1; k < n_seg; ++k)
    {
        const qd_xf_seg &ia = info[k - 1], &ib = info[k];
        const arma::uword nr = n_overlap[k], na = (arma::uword)ia.nlos.size(), nb = (arma::uword)ib.nlos.size();
        std::vector<arma::uword> exit_r(na, nr), entry_r(nb, 0);
        for (arma::uword m = 0; m < na; ++m)
            for (arma::uword r = nr; r-- > 0;)
                if (sched[k].wa[ia.cluster[m]][r] == 0.0)
                    exit_r[m] = r;
                else
                    break;
        for (arma::uword j = 0; j < nb; ++j)
        {
            entry_r[j] = nr;
            for (arma::uword r = 0; r < nr; ++r)
                if (sched[k].wb[ib.cluster[j]][r] > 0.0)
                {
                    entry_r[j] = r;
                    break;
                }
        }
        std::vector<arma::uword> ord(nb);
        for (arma::uword j = 0; j < nb; ++j)
            ord[j] = j;
        std::stable_sort(ord.begin(), ord.end(), [&](arma::uword x, arma::uword y) { return entry_r[x] < entry_r[y]; });
        std::vector<char> used(n_slot, 0), taken(n_slot, 0);
        for (arma::uword m = 0; m < na; ++m)
            used[slot[k - 1][m]] = 1;
        slot[k].assign(nb, 0);
        for (arma::uword j : ord)
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

    // Slots: LOS, GR (if a segment has them), then the NLOS slots; NLOS slots without power are removed
    bool any_los = false, any_gr = false, flat_delay = true;
    for (const qd_xf_seg &g : info)
        any_los = any_los || g.los >= 0, any_gr = any_gr || g.gr >= 0;
    for (const qd_ch *s : D)
        for (const arma::cube &d : s->delay)
            flat_delay = flat_delay && d.n_rows == 1 && d.n_cols == 1;
    const arma::uword s_gr = any_los ? 1 : 0, n_det = s_gr + (any_gr ? 1 : 0), n_all = n_det + n_slot;
    std::vector<char> power(n_slot, 0);
    for (arma::uword s = 0; s < n_out; ++s)
    {
        const qd_xf_snap &o = out[s];
        arma::cube re(n_rx, n_tx, n_all, arma::fill::zeros), im = re, dl(flat_delay ? 1 : n_rx, flat_delay ? 1 : n_tx, n_all, arma::fill::zeros);
        paths(o, v, kind);
        const arma::uword na = (arma::uword)info[o.a].nlos.size();
        for (size_t i = 0; i < v.size(); ++i)
        {
            const src &x = v[i];
            if (x.a == 0.0)
                continue;
            const arma::uword q = kind[i] == -1 ? 0 : (kind[i] == -2 ? s_gr : n_det + ((arma::uword)kind[i] < na ? slot[o.a][(arma::uword)kind[i]] : slot[(arma::uword)o.b][(arma::uword)kind[i] - na]));
            re.slice(q) = x.a * x.c->coeff_re[x.s].slice(x.p), im.slice(q) = x.a * x.c->coeff_im[x.s].slice(x.p);
            const arma::cube &d = x.c->delay[x.s];
            if (flat_delay || (d.n_rows == n_rx && d.n_cols == n_tx))
                dl.slice(q) = d.slice(x.p);
            else
                dl.slice(q).fill(d(0, 0, x.p));
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
    return c;
}

quadriga_lib::channel<double> quadriga_lib::crossfade_segments(const std::vector<channel<double>> &segments, const arma::uvec &n_overlap_in, const std::string &output)
{
    const arma::uword n_seg = (arma::uword)segments.size();
    if (n_seg == 0)
        throw std::invalid_argument("Input 'segments' must not be empty.");
    arma::uvec n_overlap = n_overlap_in;
    if (n_overlap.is_empty()) // inferred: the last snapshots of k - 1 at the positions of the first ones of k
    {
        n_overlap.zeros(n_seg);
        for (arma::uword k = 1; k < n_seg; ++k)
        {
            const qd_ch &A = segments[k - 1], &B = segments[k];
            if (A.n_snap() < n_overlap[k - 1])
                break; // the checks below report it
            const arma::uword n_max = std::min(A.n_snap() - n_overlap[k - 1], B.n_snap());
            auto same = [&](arma::uword n)
            {
                for (arma::uword i = 0; i < n; ++i)
                {
                    const arma::uword ia = A.n_snap() - n + i;
                    if (arma::norm(qd_xf_col(A.tx_pos, ia) - qd_xf_col(B.tx_pos, i)) > 1e-6 || arma::norm(qd_xf_col(A.rx_pos, ia) - qd_xf_col(B.rx_pos, i)) > 1e-6)
                        return false;
                }
                return true;
            };
            arma::uword n_found = 0, n_match = 0;
            for (arma::uword n = n_max; n > 0; --n)
                if (same(n))
                    n_found = n_found == 0 ? n : n_found, ++n_match;
            if (n_match > 1)
                throw std::invalid_argument("Entry " + std::to_string(k) + ": the positions of 'segments' do not tell the shared snapshots (they repeat); give 'n_overlap'.");
            n_overlap[k] = n_found;
        }
    }
    if (n_overlap.n_elem != n_seg)
        throw std::invalid_argument("Input 'n_overlap' must have one entry per segment.");
    if (n_overlap[0] != 0)
        throw std::invalid_argument("The first entry of 'n_overlap' must be 0.");
    if (output != "taps" && output != "rays")
        throw std::invalid_argument("Input 'output' must be 'taps' or 'rays'.");
    for (arma::uword k = 0; k < n_seg; ++k)
    {
        const qd_ch &c = segments[k];
        const std::string msg = c.is_valid();
        if (!msg.empty())
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'segments' is not valid: " + msg);
        if (c.n_snap() == 0 || c.coeff_re.empty())
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'segments' must have snapshots with channel coefficients.");
        if (c.path_gain.empty())
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'segments' must have path gains (output \"rays\").");
        const qd_ch &c0 = segments[0];
        if (c.n_rx() != c0.n_rx() || c.n_tx() != c0.n_tx())
            throw std::invalid_argument("All entries of 'segments' must have the same number of Rx and Tx ports.");
        if (c.center_freq.n_elem > 1 || c.center_freq.n_elem != c0.center_freq.n_elem || (c.center_freq.n_elem == 1 && c.center_freq[0] != c0.center_freq[0]))
            throw std::invalid_argument("All entries of 'segments' must have the same 'center_freq' with at most one value.");
        if (k > 0 && n_overlap[k] > segments[k - 1].n_snap() - n_overlap[k - 1])
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'n_overlap' must be at most the snapshots of the preceding segment after its own shared ones (" + std::to_string(segments[k - 1].n_snap() - n_overlap[k - 1]) + ").");
        if (n_overlap[k] > c.n_snap())
            throw std::invalid_argument("Entry " + std::to_string(k) + ": 'n_overlap' must be at most the snapshots of its own segment.");
    }
    std::vector<const qd_ch *> p(n_seg);
    arma::uvec trail0(n_seg, arma::fill::zeros);
    for (arma::uword k = 0; k < n_seg; ++k)
        p[k] = &segments[k], trail0[k] = segments[k].n_snap() - (k + 1 < n_seg ? n_overlap[k + 1] : 0);
    return qd_crossfade(p, p, nullptr, trail0, trail0, n_overlap, output == "taps");
}
