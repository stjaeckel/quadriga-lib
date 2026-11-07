// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Members of the class gbsm_rays: sizes, validation, the scatterer distance scales b_med and b_eff, and the
// two-bounce scatterer solver.

// b_med and b_eff: validation and the distance b of ray i from end e (0 Tx, 1 Rx) at the ray-data positions:
// NLOS end to FBS (Tx) or LBS (Rx), LOS d3D, GR the length via the specular point (image method)
static void qd_rays_check_freq(const quadriga_lib::gbsm_rays &r, arma::uword i_freq)
{
    const std::string err = r.is_valid();
    if (!err.empty())
        throw std::invalid_argument(err);
    if (i_freq >= r.n_freq())
        throw std::invalid_argument("Input 'i_freq' is out of range for " + std::to_string(r.n_freq()) +
                                    " frequencies.");
}

static double qd_rays_end_dist(const quadriga_lib::gbsm_rays &r, arma::uword i, int e)
{
    const double *T = r.tx_pos.memptr(), *R = r.rx_pos.memptr();
    const double *P = (r.path_type(i) == 2) ? (e == 0 ? r.fbs_pos.colptr(i) : r.lbs_pos.colptr(i)) : nullptr;
    const double R_img[3] = {R[0], R[1], -R[2]};
    if (P == nullptr) // LOS, GR
        return qd_gbsm_dist(T, (r.path_type(i) == 1) ? R_img : R);
    return qd_gbsm_dist(P, (e == 0) ? T : R);
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm_rays
Class for the ray data of one link (geometry-based stochastic model)

- Antenna-free rays of one link at one drop: delays, powers, angles and polarization transfer matrices, with the
  positions the data refer to; written by [[ssf_rays]] and [[gbsm_rays_from_paths]], completed by
  .[[solve_scatterers]], read by [[ssf_paths]] and [[gbsm_single_link_channel]], which call `is_valid()` first
- Plain data: the class does not normalize or reorder; later versions append attributes, so set them by name
- An object has scatterers when `fbs_pos` has `n_ray` columns (an empty object counts as having them)
- Rays added from another source take tap indices the object does not use; at most one LOS and one GR ray (checked by
  `is_valid()`) and one `method` for all rays. `.solve_scatterers` overwrites given scatterers, so solve rays without
  scatterers in a separate object and concatenate the attributes
- Extreme but finite inputs (e.g. an XPR of 1000 dB, `kf_ratio` or positions near the range of double) can give
  non-finite attributes or outputs; the consumer's `is_valid()` message then names the attribute, not the input

## Attributes:
| Attribute | Type, size | Meaning | Set by |
| --- | --- | --- | --- |
| `method` | `std::string`, default `"38.901"` | `"38.901"`, `"38.901_MF"` or `"quadriga"`; GR power law and default mobility | all |
| `tx_pos`, `rx_pos` | `arma::vec`, `[3]` | Tx and Rx position the ray data refer to in [m] | all |
| `center_frequency` | `arma::vec`, `[n_freq]` | Carrier frequencies in [Hz] | all |
| `delay` | `arma::vec`, `[n_ray]` | Delays in [s] relative to the LOS delay d3D / c, d3D the Tx-Rx distance | all |
| `power` | `arma::mat`, `[n_ray, n_freq]` | Coefficient powers (linear) without link gain; data from paths: path gain | all |
| `aod`, `eod`, `aoa`, `eoa` | `arma::vec`, `[n_ray]` | Departure and arrival azimuth and elevation in [rad], GCS | all |
| `M` | `arma::cube`, `[8, n_ray, n_freq]` | Polarization transfer matrices incl. initial phases, rows as in [[get_channels_planar]] | all |
| `path_type` | `arma::u32_vec`, `[n_ray]` | 0 = LOS, 1 = GR, 2 = NLOS | all |
| `tap` | `arma::u32_vec`, `[n_ray]` | Tap index; rays with equal index form one tap | all |
| `gr_eps_re`, `gr_eps_im` | `arma::vec`, `[n_freq]` or empty | Ground permittivity `gr_eps_re - j·gr_eps_im`; needed where a GR ray is recomputed | [[ssf_rays]], [[gbsm_rays_from_paths]] |
| `fbs_pos`, `lbs_pos` | `arma::mat`, `[3, n_ray]` or empty | First- and last-bounce scatterer positions in [m] | .[[solve_scatterers]], [[gbsm_rays_from_paths]] |
| `fallback` | `arma::u32_vec`, `[n_ray]` or empty | 1 = ray lengthened by the solver fallback | .[[solve_scatterers]] |

## Member functions:
| Member | Returns |
| --- | --- |
| `n_ray()` | Number of rays, length of `delay` |
| `n_freq()` | Number of frequencies, length of `center_frequency` |
| `empty()` | `true` if `n_ray() == 0` |
| `is_valid()` | Empty string if the object is valid, else the message of the first failed check |
| `b_med(i_freq = 0)` | `arma::vec` `[2]`, Tx and Rx in [m]: lower power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS rays with power > 0 at frequency `i_freq` (0-based), fallback rays included; validity scale of the mobility `drift`; NaN without scatterers or without NLOS power |
| `b_eff(i_freq = 0)` | `arma::vec` `[2]`, Tx and Rx in [m]: `sqrt(sum P / sum(P / b²))` over the rays with power `P > 0` at frequency `i_freq`, with `b` the end-to-scatterer distance (NLOS), d3D (LOS) or the GR path length (GR); validity scale of the mobility `phase`; NaN without scatterers or power |
| `solve_scatterers(d_min = 2.0)` | `void`; writes `fbs_pos`, `lbs_pos` and `fallback`, see .[[solve_scatterers]] |

- `b_med` and `b_eff` throw the `is_valid()` message for an invalid object and an error for `i_freq >= n_freq()`

## See also:
- [[ssf_rays]] (ray data of a set of links)
- [[gbsm_rays_from_paths]] (ray data from path data)
- .[[solve_scatterers]] (scatterers for the mobility `drift`)
- [[ssf_paths]] (paths of one link at one position)
- [[gbsm_single_link_channel]] (channel of the link)
MD!*/

arma::uword quadriga_lib::gbsm_rays::n_ray() const
{
    return delay.n_elem;
}

arma::uword quadriga_lib::gbsm_rays::n_freq() const
{
    return center_frequency.n_elem;
}

bool quadriga_lib::gbsm_rays::empty() const
{
    return n_ray() == 0;
}

std::string quadriga_lib::gbsm_rays::is_valid() const
{
    const arma::uword n = n_ray(), nf = n_freq();

    if (method != "38.901" && method != "38.901_MF" && method != "quadriga")
        return "Input 'rays.method' must be '38.901', '38.901_MF' or 'quadriga'.";
    const std::pair<std::string, const arma::vec *> pos[2] = {{"tx_pos", &tx_pos}, {"rx_pos", &rx_pos}};
    for (const auto &p : pos)
    {
        if (p.second->n_elem != 3)
            return "Input 'rays." + p.first + "' must have 3 elements.";
        if (!p.second->is_finite())
            return "Input 'rays." + p.first + "' must be finite.";
    }
    if (nf == 0)
        return "Input 'rays.center_frequency' cannot be empty.";
    if (!center_frequency.is_finite() || arma::any(center_frequency <= 0.0))
        return "Input 'rays.center_frequency' must be finite and > 0.";
    if (!delay.is_finite())
        return "Input 'rays.delay' must be finite.";
    if (power.n_rows != n || power.n_cols != nf)
        return "Input 'rays.power' must have size [n_ray, n_freq].";
    if (!power.is_finite() || arma::any(arma::vectorise(power) < 0.0))
        return "Input 'rays.power' must be finite and >= 0.";
    const std::pair<std::string, const arma::vec *> ang[4] = {{"aod", &aod}, {"eod", &eod}, {"aoa", &aoa},
                                                              {"eoa", &eoa}};
    for (const auto &p : ang)
    {
        if (p.second->n_elem != n)
            return "Input 'rays." + p.first + "' must have n_ray elements.";
        if (!p.second->is_finite())
            return "Input 'rays." + p.first + "' must be finite.";
    }
    if (M.n_rows != 8 || M.n_cols != n || M.n_slices != nf)
        return "Input 'rays.M' must have size [8, n_ray, n_freq].";
    if (!M.is_finite())
        return "Input 'rays.M' must be finite.";
    if (path_type.n_elem != n)
        return "Input 'rays.path_type' must have n_ray elements.";
    if (arma::any(path_type > 2))
        return "Values of 'rays.path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS).";
    if (arma::accu(path_type == 0) > 1 || arma::accu(path_type == 1) > 1)
        return "Input 'rays.path_type' can hold at most one LOS and one GR ray.";
    if (tap.n_elem != n)
        return "Input 'rays.tap' must have n_ray elements.";

    // Permittivity: both empty or both [n_freq]
    if (!((gr_eps_re.n_elem == 0 && gr_eps_im.n_elem == 0) || (gr_eps_re.n_elem == nf && gr_eps_im.n_elem == nf)))
        return "Inputs 'rays.gr_eps_re' and 'rays.gr_eps_im' must both be empty or both have n_freq elements.";
    if (!gr_eps_re.is_finite() || arma::any(gr_eps_re < 1.0))
        return "Input 'rays.gr_eps_re' must be finite and >= 1.";
    if (!gr_eps_im.is_finite() || arma::any(gr_eps_im < 0.0))
        return "Input 'rays.gr_eps_im' must be finite and >= 0.";

    // Scatterers: both [3, n_ray] or both without elements; a matrix without elements but with n_ray > 0 columns
    // would count as "has scatterers" and is rejected
    auto sized = [n](const arma::mat &x)
    { return x.n_rows == 3 && x.n_cols == n; };
    auto none = [n](const arma::mat &x)
    { return x.n_elem == 0 && (n == 0 || x.n_cols != n); };
    if (!((sized(fbs_pos) && sized(lbs_pos)) || (none(fbs_pos) && none(lbs_pos))))
        return "Inputs 'rays.fbs_pos' and 'rays.lbs_pos' must both be empty or both have size [3, n_ray].";
    if (!fbs_pos.is_finite())
        return "Input 'rays.fbs_pos' must be finite.";
    if (!lbs_pos.is_finite())
        return "Input 'rays.lbs_pos' must be finite.";
    if (fallback.n_elem != 0 && fallback.n_elem != n)
        return "Input 'rays.fallback' must be empty or have n_ray elements.";
    if (arma::any(fallback > 1))
        return "Values of 'rays.fallback' must be 0 or 1.";
    return "";
}

arma::vec quadriga_lib::gbsm_rays::b_med(arma::uword i_freq) const
{
    qd_rays_check_freq(*this, i_freq);
    arma::vec b(2, arma::fill::value(arma::datum::nan));
    if (fbs_pos.n_cols != n_ray())
        return b;

    // Lower weighted median: NLOS rays with power > 0 sorted by distance (stable: equal distances keep the ray
    // order), the first distance whose cumulative power reaches half the total
    for (int e = 0; e < 2; ++e)
    {
        std::vector<std::pair<double, double>> v;
        double total = 0.0, cum = 0.0;
        for (arma::uword i = 0; i < n_ray(); ++i)
            if (path_type(i) == 2 && power(i, i_freq) > 0.0)
                v.push_back({qd_rays_end_dist(*this, i, e), power(i, i_freq)}), total += power(i, i_freq);
        std::stable_sort(v.begin(), v.end(), [](const std::pair<double, double> &x, const std::pair<double, double> &y)
                         { return x.first < y.first; });
        for (const auto &x : v)
            if (2.0 * (cum += x.second) >= total)
            {
                b(e) = x.first;
                break;
            }
        if (!v.empty() && std::isnan(b(e))) // rounding of the sum: the largest distance
            b(e) = v.back().first;
    }
    return b;
}

arma::vec quadriga_lib::gbsm_rays::b_eff(arma::uword i_freq) const
{
    qd_rays_check_freq(*this, i_freq);
    arma::vec b(2, arma::fill::value(arma::datum::nan));
    if (fbs_pos.n_cols != n_ray())
        return b;

    // (sum P / sum (P / b^2))^(1/2) over the rays with power > 0
    for (int e = 0; e < 2; ++e)
    {
        double sp = 0.0, sq = 0.0;
        for (arma::uword i = 0; i < n_ray(); ++i)
            if (power(i, i_freq) > 0.0)
            {
                const double d = qd_rays_end_dist(*this, i, e);
                sp += power(i, i_freq), sq += power(i, i_freq) / (d * d);
            }
        if (sp > 0.0)
            b(e) = std::sqrt(sp / sq);
    }
    return b;
}

/*!MD
# .solve_scatterers
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

- Member of [[gbsm_rays]]: places the scatterers that the mobility `drift` needs, in place
- NLOS rays with power > 0 at some frequency: FBS on the departure ray and LBS on the arrival ray at equal
  distances from their end, so that the polyline Tx, FBS, LBS, Rx has the length d3D + c·delay (d3D the Tx-Rx
  distance) and the angles are exact
- Where no such pair exists with both distances at least `d_min`, both scatterers are placed at `d_min` along the ray's
  own directions and `fallback` is 1: the polyline is longer than the ray, `delay` keeps its value. With `d_min` = 2 m
  this applies to up to about 23 % of the rays (most in the TR methods without absolute TOA) and lengthens them by up to
  25 ns (median below 5 ns), the DS of a link by up to 3 ns; `b_med` can then equal `d_min`
- LOS ray and NLOS rays with zero power at all frequencies: the midpoint of the direct line; GR ray: the specular
  point on z = 0, which requires both ends at z >= 0
- Writes `fbs_pos`, `lbs_pos` and `fallback` of all rays and overwrites given scatterers; nothing else changes;
  afterwards `b_med()` and `b_eff()` give the validity scales of [[ssf_paths]]
- One link per call; batch callers loop over the objects of [[ssf_rays]]. Throws `std::invalid_argument` with the
  `is_valid()` message for an invalid object, which stays unchanged

## Declaration:
```
void quadriga_lib::gbsm_rays::solve_scatterers(double d_min = 2.0);
```

## Input:
- **`d_min`** *(optional)* — Minimum distance of the scatterers to both ends in [m], finite and > 0

## See also:
- [[gbsm_rays]] (attributes and members)
- [[ssf_paths]] (mobility `drift` needs the scatterers)
MD!*/

void quadriga_lib::gbsm_rays::solve_scatterers(double d_min)
{
    const std::string err = is_valid();
    if (!err.empty())
        throw std::invalid_argument(err);
    if (!std::isfinite(d_min) || d_min <= 0.0)
        throw std::invalid_argument("Input 'd_min' must be finite and > 0.");
    if (arma::any(path_type == 1) && (tx_pos(2) < 0.0 || rx_pos(2) < 0.0))
        throw std::invalid_argument("Ground reflection requires z >= 0 in 'rays.tx_pos' and 'rays.rx_pos'.");

    const arma::uword n = n_ray();
    if (fbs_pos.n_rows != 3 || fbs_pos.n_cols != n)
        fbs_pos.set_size(3, n);
    if (lbs_pos.n_rows != 3 || lbs_pos.n_cols != n)
        lbs_pos.set_size(3, n);
    if (fallback.n_elem != n)
        fallback.set_size(n);

    const double *T = tx_pos.memptr(), *R = rx_pos.memptr();
    const double D[3] = {T[0] - R[0], T[1] - R[1], T[2] - R[2]};
    const double DD = D[0] * D[0] + D[1] * D[1] + D[2] * D[2], d3D = std::sqrt(DD);
    const double mid[3] = {0.5 * (T[0] + R[0]), 0.5 * (T[1] + R[1]), 0.5 * (T[2] + R[2])};

    double G[3];
    qd_gbsm_specular_point(T, R, G);

    for (arma::uword i = 0; i < n; ++i)
    {
        double *F = fbs_pos.colptr(i), *B = lbs_pos.colptr(i);
        fallback(i) = 0;
        if (path_type(i) == 1)
        {
            for (int j = 0; j < 3; ++j)
                F[j] = G[j], B[j] = G[j];
            continue;
        }
        bool on = false; // power > 0 at some frequency
        for (arma::uword k = 0; k < power.n_cols; ++k)
            on = on || power(i, k) > 0.0;
        if (path_type(i) == 0 || !on)
        {
            for (int j = 0; j < 3; ++j)
                F[j] = mid[j], B[j] = mid[j];
            continue;
        }

        // Balanced rule FBS = T + s·u_d, LBS = R + s·u_a with 2·s + abs(D + s·w) = L, w = u_d - u_a: the smallest
        // root of (4 - w·w)·s² - (4·L + 2·D·w)·s + (L² - D·D) = 0 in the cancellation-free form. g(s) = 2·s +
        // abs(D + s·w) is non-decreasing (abs(w) <= 2), and so is a + b + abs(D + a·u_d - b·u_a) in a and b: a
        // solution with both distances >= d_min exists iff s >= d_min. Otherwise (also for L <= abs(D)) both
        // scatterers go to d_min along the ray's own directions, which lengthens the ray.
        const double ce = std::cos(eod(i)), ca = std::cos(eoa(i));
        const double ud[3] = {ce * std::cos(aod(i)), ce * std::sin(aod(i)), std::sin(eod(i))};
        const double ua[3] = {ca * std::cos(aoa(i)), ca * std::sin(aoa(i)), std::sin(eoa(i))};
        const double w[3] = {ud[0] - ua[0], ud[1] - ua[1], ud[2] - ua[2]};
        const double L = d3D + 299792458.0 * delay(i);
        double s = 0.0;
        if (L > d3D) // zero or negative excess length: fallback (also avoids the 0 / 0 of L = -d3D)
        {
            const double A = std::max(4.0 - (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]), 0.0);
            const double Bq = 4.0 * L + 2.0 * (D[0] * w[0] + D[1] * w[1] + D[2] * w[2]), C = L * L - DD;
            s = 2.0 * C / (Bq + std::sqrt(std::max(Bq * Bq - 4.0 * A * C, 0.0)));
            s = std::isfinite(s) ? s : 0.0;
        }
        if (s < d_min)
            s = d_min, fallback(i) = 1;
        for (int j = 0; j < 3; ++j)
            F[j] = T[j] + s * ud[j], B[j] = R[j] + s * ua[j];
    }
}
