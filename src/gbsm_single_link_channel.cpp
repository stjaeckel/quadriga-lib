// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"
#include "quadriga_gbsm.hpp"

#include <algorithm>
#include <any>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// OpenMP threshold of "phase" on n_snap · n_kept · n_rx · n_tx (measured, see the snapshot loop)
static constexpr arma::uword L_chan = 8192;

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm_single_link_channel
Channel coefficients of one link at one frequency over one segment (geometry-based stochastic model)

- Snapshot t: the paths of [[ssf_paths]] at the positions of t, at `rays.center_frequency(i_freq)` with the link gain
  and K ratio of t, for the rays with power > 0 at `i_freq` plus LOS and GR
- `"phase"`: [[get_channels_planar]] with the arrays at the positions of `rays` and the orientations of t; delays,
  path lengths and angles stay those of the ray-data positions. `"drift"`: [[get_channels_spherical]] with the arrays
  at the positions and orientations of t
- `"taps"`: the rays summed per tap index, taps in ascending index, one tap per distinct index of the kept rays (fixed
  per call); LOS and GR taps are present whenever `rays` holds them, also at zero power. The delay of a tap per
  element pair is the mean of the rays' delays weighted by `rays.power` at `i_freq` (equal weights where they sum to
  0), `[n_rx, n_tx, n_tap]`; reduce it to `[1, 1, n_tap]` where needed. TR 38.901 LOS gives two taps at delay 0
- `"taps"` is exact at the phase centre for the TR methods (the rays of a tap share one delay); for `"quadriga"` only
  at the carrier: in UMi NLOS a tap varies over the band by 0.18, 0.52 and 0.79 (in-band selectivity) for 20, 100 and
  400 MHz. `"rays"`: one path per kept ray in ray order, exact at any bandwidth, the form of the QRT route; the path
  data are the outputs of [[ssf_paths]] at t (in `"phase"` with the displacement phase in `M`, for the arrays at the
  ray-data positions)
- `n_snap` is the largest row count of the positions and orientations (1 if all are omitted); each has 1 or `n_snap`
  rows, `link_gain_dB` and `kf_ratio` 1 or `n_snap` entries; one row or entry applies to all snapshots. One position
  is a `[1, 3]` row; a `[3]` vector is rejected. A static link with a gain per snapshot repeats its position
- `use_absolute_delays` in `"phase"` adds the LOS delay of the ray-data positions
- No power correction and no initial position; validity limits and their meaning: see [[ssf_paths]]
- Errors: an invalid array (message after `Transmit antenna: ` or `Receive antenna: `), the errors of [[ssf_paths]]
  in `"drift"`, a row or entry count other than 1 or `n_snap`

## Entries of par:

| Name | Size | Content |
| --- | --- | --- |
| `tap` | `[n_tap]` or `[n_path]` | Tap index per output tap or per path, `arma::uvec` |
| `n_ray_tap` | `[n_tap]` | Number of kept rays per tap, `arma::uvec`; `"taps"` only |
| `fbs_pos`, `lbs_pos` | `[3, n_kept]` | Scatterers of the kept rays at the ray-data positions; `"drift"` with `"taps"` only |
| `gr_pos` | `[3, n_snap]` | GR point per snapshot; `"drift"` with `"taps"` and a GR ray only |
| `link_gain_dB` | `[n_snap]` | Link gain per snapshot |
| `b_med`, `b_eff` | `[2]` | Of [[gbsm_rays]] at `i_freq`, Tx and Rx; without scatterers from a solved copy (`d_min` = 2 m) |
| `limit` | `[2]` | Validity limit per end, $\sqrt{\lambda b_{\mathrm{eff}} / 8}$ (`"phase"`) or $0.2 b_{\mathrm{med}}$ (`"drift"`); Inf where b is NaN |
| `ratio` | `[2]` | Largest distance of the end from its ray-data position over the snapshots, divided by `limit` |
| `ratio_total` | scalar | $\sqrt{r_{\mathrm{Tx}}^2 + r_{\mathrm{Rx}}^2}$ of the two ratios; valid up to 1; `"drift"`: new segment from 2.5 |

## Declaration:
```
channel<double> quadriga_lib::gbsm_single_link_channel(
    const arrayant<double> &tx_array,
    const arrayant<double> &rx_array,
    const gbsm_rays &rays,
    const arma::vec &link_gain_dB,
    const std::string &mobility = "auto",
    const std::string &output = "taps",
    arma::uword i_freq = 0,
    const arma::mat *tx_pos = nullptr,
    const arma::mat *rx_pos = nullptr,
    const arma::mat *tx_orientation = nullptr,
    const arma::mat *rx_orientation = nullptr,
    const arma::vec *kf_ratio = nullptr,
    bool oxygen = false,
    bool use_absolute_delays = false);
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array antenna with `n_tx` and `n_rx` ports
- **`rays`** — Ray data of the link, see [[gbsm_rays]]
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at `rays.center_frequency(i_freq)` in [dB]; `[n_snap]` or `[1]`
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"` (phase for the TR methods, drift for `"quadriga"`)
- **`output`** *(optional)* — `"taps"` or `"rays"`
- **`i_freq`** *(optional)* — Frequency index in `rays.center_frequency`, 0-based
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx positions in [m]; omitted: the positions of `rays`;
  `[n_snap, 3]` or `[1, 3]`
- **`tx_orientation`**, **`rx_orientation`** *(optional)* — Euler angles (bank, tilt, heading) in [rad]; omitted: 0;
  `[n_snap, 3]` or `[1, 3]`
- **`kf_ratio`** *(optional)* — K(snapshot) / K(drop), >= 0; omitted: 1; `[n_snap]` or `[1]`
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1, see [[ssf_paths]]
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays

## Returns:
- Channel object: `center_frequency` `[1]`; `tx_pos`, `rx_pos` (the snapshot positions), `tx_orientation`,
  `rx_orientation` `[3, n_snap]`; per snapshot `coeff_re`, `coeff_im`, `delay` `[n_rx, n_tx, n_tap]` (`"taps"`) or
  `[n_rx, n_tx, n_path]` (`"rays"`); `"rays"` also `path_gain`, `path_length`, `path_polarization`, `path_angles`
  and in `"drift"` `path_fbs_pos`, `path_lbs_pos`; `par_names`, `par_data` as above

## See also:
- [[ssf_paths]] (paths of one link at one position, validity limits)
- [[gbsm_rays]] (input data, `b_med` and `b_eff`)
- [[gbsm_rays_from_paths]] (ray data from path data, e.g. of a QRT file; taps 0, 3, 6, ... without `tap`)
MD!*/

quadriga_lib::channel<double> quadriga_lib::gbsm_single_link_channel(const arrayant<double> &tx_array,
                                                                     const arrayant<double> &rx_array,
                                                                     const gbsm_rays &rays,
                                                                     const arma::vec &link_gain_dB,
                                                                     const std::string &mobility,
                                                                     const std::string &output,
                                                                     arma::uword i_freq,
                                                                     const arma::mat *tx_pos,
                                                                     const arma::mat *rx_pos,
                                                                     const arma::mat *tx_orientation,
                                                                     const arma::mat *rx_orientation,
                                                                     const arma::vec *kf_ratio,
                                                                     bool oxygen,
                                                                     bool use_absolute_delays)
{
    // Validation; the ground-reflection checks of ssf_paths for "drift" follow in the snapshot loop
    for (int e = 0; e < 2; ++e)
    {
        const std::string msg = (e == 0) ? tx_array.is_valid() : rx_array.is_valid();
        if (!msg.empty())
            throw std::invalid_argument((e == 0 ? "Transmit antenna: " : "Receive antenna: ") + msg);
    }
    const std::string err = rays.is_valid();
    if (!err.empty())
        throw std::invalid_argument(err);
    if (mobility != "phase" && mobility != "drift" && mobility != "auto")
        throw std::invalid_argument("Input 'mobility' must be 'phase', 'drift' or 'auto'.");
    if (output != "taps" && output != "rays")
        throw std::invalid_argument("Input 'output' must be 'taps' or 'rays'.");
    if (i_freq >= rays.n_freq())
        throw std::invalid_argument("Input 'i_freq' is out of range for " + std::to_string(rays.n_freq()) +
                                    " frequencies.");
    if (link_gain_dB.empty())
        throw std::invalid_argument("Input 'link_gain_dB' cannot be empty.");
    if (!link_gain_dB.is_finite())
        throw std::invalid_argument("Input 'link_gain_dB' must be finite.");
    const std::pair<std::string, const arma::mat *> pm[4] = {
        {"tx_pos", tx_pos}, {"rx_pos", rx_pos}, {"tx_orientation", tx_orientation}, {"rx_orientation", rx_orientation}};
    arma::uword n_snap = 1;
    for (const auto &p : pm)
        if (p.second != nullptr)
        {
            if (p.second->n_cols != 3)
                throw std::invalid_argument("Input '" + p.first + "' must have 3 columns.");
            if (!p.second->is_finite())
                throw std::invalid_argument("Input '" + p.first + "' must be finite.");
            n_snap = std::max(n_snap, p.second->n_rows);
        }
    if (kf_ratio != nullptr && (!kf_ratio->is_finite() || arma::any(*kf_ratio < 0.0)))
        throw std::invalid_argument("Input 'kf_ratio' must be finite and >= 0.");
    const std::pair<std::string, arma::uword> cnt[6] = {
        {"tx_pos", tx_pos ? tx_pos->n_rows : 1}, {"rx_pos", rx_pos ? rx_pos->n_rows : 1},
        {"tx_orientation", tx_orientation ? tx_orientation->n_rows : 1},
        {"rx_orientation", rx_orientation ? rx_orientation->n_rows : 1}, {"link_gain_dB", link_gain_dB.n_elem},
        {"kf_ratio", kf_ratio ? kf_ratio->n_elem : 1}};
    for (const auto &c : cnt)
        if (c.second != 1 && c.second != n_snap)
            throw std::invalid_argument("Number of snapshots in '" + c.first + "' must be 1 or " +
                                        std::to_string(n_snap) + ".");

    const bool drift = mobility == "drift" || (mobility == "auto" && rays.method == "quadriga");
    if (drift && rays.n_ray() > 0 && rays.fbs_pos.n_cols != rays.n_ray()) // also when no ray is kept at i_freq
        throw std::invalid_argument("Mobility 'drift' needs 'rays.fbs_pos' and 'rays.lbs_pos'; solve the scatterers "
                                    "first.");
    const bool taps = output == "taps";
    const double c0 = 299792458.0, fc = rays.center_frequency(i_freq), k0 = 6.283185307179586 * fc / c0;

    // The rays kept at i_freq (LOS, GR, NLOS with power > 0 there) as ray data of one frequency: ssf_paths keeps
    // all of them in ray order and returns the same values as for all frequencies (the dropped rays add 0 to the
    // power sums of the K rescale); the solver works per ray, so b_med and b_eff equal those of 'rays' at i_freq
    std::vector<arma::uword> kv;
    for (arma::uword i = 0; i < rays.n_ray(); ++i)
        if (rays.path_type(i) != 2 || rays.power(i, i_freq) > 0.0)
            kv.push_back(i);
    const arma::uvec kept(kv), fi = {i_freq};
    const arma::uword n_kept = kept.n_elem;
    gbsm_rays rr;
    rr.method = rays.method, rr.tx_pos = rays.tx_pos, rr.rx_pos = rays.rx_pos, rr.center_frequency = {fc};
    rr.delay = rays.delay.elem(kept), rr.power = rays.power.submat(kept, fi);
    rr.aod = rays.aod.elem(kept), rr.eod = rays.eod.elem(kept), rr.aoa = rays.aoa.elem(kept);
    rr.eoa = rays.eoa.elem(kept), rr.path_type = rays.path_type.elem(kept), rr.tap = rays.tap.elem(kept);
    rr.M.set_size(8, n_kept, 1), rr.M.slice(0) = rays.M.slice(i_freq).cols(kept);
    if (!rays.gr_eps_re.empty())
        rr.gr_eps_re = rays.gr_eps_re.elem(fi), rr.gr_eps_im = rays.gr_eps_im.elem(fi);
    if (rays.fbs_pos.n_cols == rays.n_ray() && n_kept > 0)
        rr.fbs_pos = rays.fbs_pos.cols(kept), rr.lbs_pos = rays.lbs_pos.cols(kept);
    if (!rays.fallback.empty())
        rr.fallback = rays.fallback.elem(kept);

    // Output object: positions and orientations as columns [3, n_snap]
    channel<double> c;
    c.center_frequency = {fc};
    const std::pair<arma::mat *, const arma::mat *> col[4] = {{&c.tx_pos, tx_pos},
                                                              {&c.rx_pos, rx_pos},
                                                              {&c.tx_orientation, tx_orientation},
                                                              {&c.rx_orientation, rx_orientation}};
    for (int j = 0; j < 4; ++j)
    {
        const arma::vec d = (j == 0) ? rays.tx_pos : (j == 1) ? rays.rx_pos : arma::vec(3, arma::fill::zeros);
        *col[j].first = (col[j].second == nullptr)       ? arma::repmat(d, 1, n_snap)
                        : (col[j].second->n_rows == 1) ? arma::repmat(col[j].second->t(), 1, n_snap)
                                                       : arma::mat(col[j].second->t());
    }
    arma::vec lg(n_snap), kf(n_snap);
    for (arma::uword t = 0; t < n_snap; ++t)
        lg(t) = link_gain_dB(link_gain_dB.n_elem == 1 ? 0 : t),
        kf(t) = (kf_ratio == nullptr) ? 1.0 : (*kf_ratio)(kf_ratio->n_elem == 1 ? 0 : t);

    // Taps: index of each kept ray's tap (ascending tap values) and its delay weight w / sum(w) with w the ray
    // power at i_freq (equal weights where the sum is 0)
    const arma::uvec tap_val = arma::unique(rr.tap);
    const arma::uword n_tap = tap_val.n_elem, n_rx = rx_array.n_ports(), n_tx = tx_array.n_ports();
    arma::uvec q_of(n_kept), n_ray_tap(n_tap, arma::fill::zeros);
    arma::vec w(n_kept), w_sum(n_tap, arma::fill::zeros);
    for (arma::uword r = 0; r < n_kept; ++r)
    {
        q_of(r) = (arma::uword)(std::lower_bound(tap_val.begin(), tap_val.end(), rr.tap(r)) - tap_val.begin());
        w_sum(q_of(r)) += rr.power(r, 0), n_ray_tap(q_of(r)) += 1;
    }
    for (arma::uword r = 0; r < n_kept; ++r)
        w(r) = (w_sum(q_of(r)) > 0.0) ? rr.power(r, 0) / w_sum(q_of(r)) : 1.0 / (double)n_ray_tap(q_of(r));

    // Runs f(i) for i < n, in parallel when 'par'. The serial branch opens no parallel region, so that
    // get_channels_* keep their own threads (an inactive region would make them start new nested teams). The first
    // exception of the parallel branch is rethrown after the loop with its type: OMP_SAFE_CALL keeps only the
    // message, and the "drift" errors of ssf_paths must stay std::invalid_argument.
    auto loop = [](int n, bool par, const auto &f)
    {
        if (!par)
        {
            for (int i = 0; i < n; ++i)
                f((arma::uword)i);
            return;
        }
        std::exception_ptr err;
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            try
            {
                f((arma::uword)i);
            }
            catch (...)
            {
#pragma omp critical
                if (!err)
                    err = std::current_exception();
            }
        }
        if (err)
            std::rethrow_exception(err);
    };

    // "phase" (TR 38.901 eq. (7.5-22)): c_r(t) = c_r(unit gain, orientations of t) · sqrt(g_r(t)) · exp(j·phi_r(t))
    // by the linearity of get_channels_planar in M; the unit coefficients are computed once per run of equal
    // orientations and the path gains once per run of equal link gain and K ratio (exact comparisons)
    std::vector<arma::uword> o_run(n_snap), g_run(n_snap);
    std::vector<int> o_start;
    std::vector<arma::cube> u_re(n_snap), u_im(n_snap), u_dl(n_snap);
    std::vector<arma::vec> g_ph(n_snap);
    arma::vec L_ph;
    arma::mat r_tx(3, n_kept), r_rx(3, n_kept);
    if (!drift)
    {
        for (arma::uword t = 0; t < n_snap; ++t)
        {
            const bool same_o = t > 0 && arma::all(c.tx_orientation.col(t) == c.tx_orientation.col(t - 1)) &&
                                arma::all(c.rx_orientation.col(t) == c.rx_orientation.col(t - 1));
            const bool same_g = t > 0 && lg(t) == lg(t - 1) && kf(t) == kf(t - 1);
            o_run[t] = same_o ? o_run[t - 1] : t, g_run[t] = same_g ? g_run[t - 1] : t;
            if (!same_o)
                o_start.push_back((int)t);
            if (g_run[t] == t)
            {
                arma::mat pg;
                arma::cube pM;
                const arma::vec lg_t = {lg(t)}, kf_t = {kf(t)};
                ssf_paths(rr, lg_t, pg, L_ph, pM, "phase", nullptr, nullptr, kf_ratio ? &kf_t : nullptr, nullptr,
                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, oxygen);
                if (n_kept > 0) // col(0) of a [0, 1] matrix would bind a null reference
                    g_ph[t] = pg.col(0);
            }
        }
        for (arma::uword r = 0; r < n_kept; ++r)
        {
            const double ce = std::cos(rr.eod(r)), ca = std::cos(rr.eoa(r));
            r_tx(0, r) = ce * std::cos(rr.aod(r)), r_tx(1, r) = ce * std::sin(rr.aod(r));
            r_rx(0, r) = ca * std::cos(rr.aoa(r)), r_rx(1, r) = ca * std::sin(rr.aoa(r));
            r_tx(2, r) = std::sin(rr.eod(r)), r_rx(2, r) = std::sin(rr.eoa(r));
        }
        const arma::vec ones(n_kept, arma::fill::ones);
        const arma::mat &M0 = rr.M.slice(0);
        const double *T0 = rays.tx_pos.memptr(), *R0 = rays.rx_pos.memptr();
        auto unit = [&](arma::uword t)
        {
            const double *To = c.tx_orientation.colptr(t), *Ro = c.rx_orientation.colptr(t);
            get_channels_planar(&tx_array, &rx_array, T0[0], T0[1], T0[2], To[0], To[1], To[2], R0[0], R0[1], R0[2],
                                Ro[0], Ro[1], Ro[2], &rr.aod, &rr.eod, &rr.aoa, &rr.eoa, &ones, &L_ph, &M0, &u_re[t],
                                &u_im[t], &u_dl[t], fc, use_absolute_delays);
        };
        const int n_o = (int)o_start.size();
        loop(n_o, n_o > 1 && (arma::uword)n_o * n_kept * n_rx * n_tx >= L_chan,
             [&](arma::uword s) { unit((arma::uword)o_start[s]); });
    }

    // Snapshot loop. OpenMP pays from 2 snapshots in "drift" (about 1 µs per element pair and path) and from
    // n_snap · n_kept · n_rx · n_tx = L_chan in "phase" (5 to 50 ns per unit; measured with 1 to 16 element pairs and
    // 10 to 380 rays).
    c.coeff_re.resize(n_snap), c.coeff_im.resize(n_snap), c.delay.resize(n_snap);
    if (!taps)
    {
        c.path_gain.resize(n_snap), c.path_length.resize(n_snap), c.path_polarization.resize(n_snap);
        c.path_angles.resize(n_snap);
        if (drift)
            c.path_fbs_pos.resize(n_snap), c.path_lbs_pos.resize(n_snap);
    }
    const arma::uvec i_gr = arma::find(rr.path_type == 1);
    const bool with_gr = drift && taps && i_gr.n_elem > 0;
    arma::mat gr_pos(3, with_gr ? n_snap : 0);
    auto snap = [&](arma::uword t)
    {
        arma::cube cr, ci, dl;
        arma::mat pg, M, fbs, lbs;
        arma::vec pl, aod, eod, aoa, eoa;
        if (!drift)
        {
            const arma::uword o = o_run[t];
            const arma::vec &g = g_ph[g_run[t]];
            cr = u_re[o], ci = u_im[o], dl = u_dl[o];
            M = arma::mat(rr.M.slice_memptr(0), 8, n_kept), pg = g, pl = L_ph;
            double dT[3], dR[3];
            for (int j = 0; j < 3; ++j)
                dT[j] = c.tx_pos(j, t) - rays.tx_pos(j), dR[j] = c.rx_pos(j, t) - rays.rx_pos(j);
            for (arma::uword r = 0; r < n_kept; ++r)
            {
                const double ph = k0 * (r_rx(0, r) * dR[0] + r_rx(1, r) * dR[1] + r_rx(2, r) * dR[2] +
                                        r_tx(0, r) * dT[0] + r_tx(1, r) * dT[1] + r_tx(2, r) * dT[2]);
                const double s = std::sqrt(g(r)), cp = std::cos(ph), sp = std::sin(ph);
                double *a = cr.slice_memptr(r), *b = ci.slice_memptr(r), *m = M.colptr(r);
                for (arma::uword e = 0; e < n_rx * n_tx; ++e)
                {
                    const double x = a[e];
                    a[e] = s * (x * cp - b[e] * sp), b[e] = s * (x * sp + b[e] * cp);
                }
                for (int e = 0; e < 8; e += 2)
                {
                    const double x = m[e];
                    m[e] = x * cp - m[e + 1] * sp, m[e + 1] = x * sp + m[e + 1] * cp;
                }
            }
            aod = rr.aod, eod = rr.eod, aoa = rr.aoa, eoa = rr.eoa;
        }
        else
        {
            arma::cube pM;
            const arma::vec T = c.tx_pos.col(t), R = c.rx_pos.col(t), lg_t = {lg(t)}, kf_t = {kf(t)};
            ssf_paths(rr, lg_t, pg, pl, pM, "drift", &T, &R, kf_ratio ? &kf_t : nullptr, &aod, &eod, &aoa, &eoa,
                      &fbs, &lbs, nullptr, nullptr, oxygen);
            M = arma::mat(pM.slice_memptr(0), 8, n_kept);
            const arma::vec g = (n_kept > 0) ? arma::vec(pg.col(0)) : arma::vec();
            const double *To = c.tx_orientation.colptr(t), *Ro = c.rx_orientation.colptr(t);
            get_channels_spherical(&tx_array, &rx_array, T(0), T(1), T(2), To[0], To[1], To[2], R(0), R(1), R(2),
                                   Ro[0], Ro[1], Ro[2], &fbs, &lbs, &g, &pl, &M, &cr, &ci, &dl, fc,
                                   use_absolute_delays);
            if (with_gr)
                gr_pos.col(t) = fbs.col(i_gr(0));
        }
        if (taps) // sums per tap, power-weighted per-element delays; local cubes and raw pointers (Cube::slice()
        {         // takes a lock, and writes to neighbouring cube objects of 'c' share cache lines)
            arma::cube tr(n_rx, n_tx, n_tap, arma::fill::zeros), ti = tr, td = tr;
            const arma::uword n_e = n_rx * n_tx;
            for (arma::uword r = 0; r < n_kept; ++r)
            {
                const double *a = cr.memptr() + r * n_e, *b = ci.memptr() + r * n_e, *d = dl.memptr() + r * n_e;
                double *x = tr.memptr() + q_of(r) * n_e, *y = ti.memptr() + q_of(r) * n_e;
                double *z = td.memptr() + q_of(r) * n_e, wr = w(r);
                for (arma::uword e = 0; e < n_e; ++e)
                    x[e] += a[e], y[e] += b[e], z[e] += wr * d[e];
            }
            c.coeff_re[t] = std::move(tr), c.coeff_im[t] = std::move(ti), c.delay[t] = std::move(td);
        }
        else
        {
            c.coeff_re[t] = std::move(cr), c.coeff_im[t] = std::move(ci), c.delay[t] = std::move(dl);
            c.path_gain[t] = arma::vectorise(pg), c.path_length[t] = pl, c.path_polarization[t] = M;
            c.path_angles[t] = arma::join_rows(arma::join_rows(aod, eod), arma::join_rows(aoa, eoa));
            if (drift)
                c.path_fbs_pos[t] = fbs, c.path_lbs_pos[t] = lbs;
        }
    };
    loop((int)n_snap, n_snap > 1 && (drift || n_snap * n_kept * n_rx * n_tx >= L_chan), snap);

    // Validity report: b_med, b_eff at i_freq (without scatterers from the solved copy), limits per end
    // ("phase" sqrt(lambda · b_eff / 8), "drift" 0.2 · b_med, Inf where b is NaN) and the largest displacement
    // from the ray-data positions relative to the limit. A GR ray with an end below z = 0 (possible in "phase") has
    // no specular point, which the solver needs: it is solved as an NLOS ray, since b uses no GR scatterer (b of the
    // GR ray is its length by the image method).
    if (rr.fbs_pos.n_cols != n_kept)
    {
        const arma::uvec type = rr.path_type;
        if (rr.tx_pos(2) < 0.0 || rr.rx_pos(2) < 0.0)
            rr.path_type.replace(1, 2);
        rr.solve_scatterers();
        rr.path_type = type;
    }
    const arma::vec b_med = rr.b_med(0), b_eff = rr.b_eff(0);
    arma::vec limit(2), ratio(2, arma::fill::zeros);
    for (int e = 0; e < 2; ++e)
    {
        const double l = drift ? 0.2 * b_med(e) : std::sqrt(c0 / fc * b_eff(e) / 8.0);
        limit(e) = std::isnan(l) ? arma::datum::inf : l;
        const arma::mat &P = (e == 0) ? c.tx_pos : c.rx_pos;
        const arma::vec &P0 = (e == 0) ? rays.tx_pos : rays.rx_pos;
        for (arma::uword t = 0; t < n_snap; ++t)
        {
            const double d = std::hypot(std::hypot(P(0, t) - P0(0), P(1, t) - P0(1)), P(2, t) - P0(2));
            ratio(e) = (d > 0.0) ? std::max(ratio(e), d / limit(e)) : ratio(e);
        }
    }

    c.par_names = {"tap"}, c.par_data = {taps ? tap_val : rr.tap};
    if (taps)
        c.par_names.push_back("n_ray_tap"), c.par_data.push_back(n_ray_tap);
    if (drift && taps)
    {
        c.par_names.push_back("fbs_pos"), c.par_data.push_back(arma::mat(arma::reshape(rr.fbs_pos, 3, n_kept)));
        c.par_names.push_back("lbs_pos"), c.par_data.push_back(arma::mat(arma::reshape(rr.lbs_pos, 3, n_kept)));
    }
    if (with_gr)
        c.par_names.push_back("gr_pos"), c.par_data.push_back(gr_pos);
    const std::pair<std::string, arma::vec> rep[5] = {
        {"link_gain_dB", lg}, {"b_med", b_med}, {"b_eff", b_eff}, {"limit", limit}, {"ratio", ratio}};
    for (const auto &x : rep)
        c.par_names.push_back(x.first), c.par_data.push_back(x.second);
    c.par_names.push_back("ratio_total"), c.par_data.push_back(std::sqrt(ratio(0) * ratio(0) + ratio(1) * ratio(1)));
    return c;
}
