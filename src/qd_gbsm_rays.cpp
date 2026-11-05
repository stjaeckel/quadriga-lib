// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"

#include <cmath>
#include <string>
#include <utility>

// Members of the class gbsm_rays: sizes, validation, the scatterer distance scales b_med and b_eff, and the
// two-bounce scatterer solver.

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm_rays
Ray data of one link (geometry-based stochastic model)

- Antenna-free rays of one link at one drop: delays, powers, angles and polarization transfer matrices, with the
  positions the data refer to; written by [[ssf_rays]] and [[gbsm_rays_from_paths]], completed by
  .[[solve_scatterers]], read by [[ssf_paths]] and [[gbsm_single_link_channel]], which call `is_valid()` first
- Plain data: the class does not normalize or reorder; later versions append attributes, so set them by name
- An object has scatterers when `fbs_pos` has `n_ray` columns; every object with `n_ray = 0` has them
- Rays from several sources in one object: at most one LOS and one GR ray, with tap index 0 and 3; the other rays
  take any other tap indices; one `method` for all rays; `.solve_scatterers` overwrites given scatterers, so solve
  rays without scatterers in a separate object and concatenate the attributes

## Attributes:
| Attribute | Size | Meaning | Set by |
| --- | --- | --- | --- |
| `method` | string | `"38.901"`, `"38.901_MF"` or `"quadriga"`; GR power law and default mobility | all |
| `tx_pos`, `rx_pos` | `[3]` | Tx and Rx position the ray data refer to in [m] | all |
| `center_frequency` | `[n_freq]` | Carrier frequencies in [Hz] | all |
| `delay` | `[n_ray]` | Delays in [s] relative to the LOS delay `abs(rx_pos - tx_pos) / c` | all |
| `power` | `[n_ray, n_freq]` | Coefficient powers (linear) without link gain; data from paths: path gain | all |
| `aod`, `eod`, `aoa`, `eoa` | `[n_ray]` | Departure and arrival azimuth and elevation in [rad], GCS | all |
| `M` | `[8, n_ray, n_freq]` | Polarization transfer matrices incl. initial phases, rows as in [[get_channels_planar]] | all |
| `path_type` | `[n_ray]` | 0 = LOS, 1 = GR, 2 = NLOS | all |
| `tap` | `[n_ray]` | Tap index; rays with equal index form one tap | all |
| `gr_eps_re`, `gr_eps_im` | `[n_freq]` or empty | Ground permittivity `gr_eps_re - j·gr_eps_im`; needed where a GR ray is recomputed | [[ssf_rays]], [[gbsm_rays_from_paths]] |
| `fbs_pos`, `lbs_pos` | `[3, n_ray]` or empty | First- and last-bounce scatterer positions in [m] | .[[solve_scatterers]], [[gbsm_rays_from_paths]] |
| `fallback` | `[n_ray]` or empty | 1 = ray lengthened by the solver fallback | .[[solve_scatterers]] |

## Member functions:
| Member | Returns |
| --- | --- |
| `n_ray()` | Number of rays, length of `delay` |
| `n_freq()` | Number of frequencies, length of `center_frequency` |
| `empty()` | `true` if `n_ray() == 0` |
| `is_valid()` | Empty string if the object is valid, else the message of the first failed check |
| `b_med(i_freq)` | `[2]`, Tx and Rx in [m]: lower power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS rays with power > 0 at frequency `i_freq` (0-based); NaN without scatterers or without NLOS power |
| `b_eff(i_freq)` | `[2]`, Tx and Rx in [m]: $b_\text{eff} = \left(\sum P / \sum (P / b^2)\right)^{1/2}$ over the rays with power $P > 0$, with $b$ the end-to-scatterer distance (NLOS), d3D (LOS) or the GR path length (GR); NaN without scatterers or power |

## See also:
- [[ssf_rays]] (ray data of a set of links)
- [[ssf_paths]] (paths of one link at one position)
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
    // STUB: no computation yet
    (void)i_freq;
    return arma::vec(2, arma::fill::value(arma::datum::nan));
}

arma::vec quadriga_lib::gbsm_rays::b_eff(arma::uword i_freq) const
{
    // STUB: no computation yet
    (void)i_freq;
    return arma::vec(2, arma::fill::value(arma::datum::nan));
}

/*!MD
# .solve_scatterers
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

- NLOS rays: exact departure and arrival angles where a solution exists, equal distances from both ends;
  otherwise the shortest polyline along the ray's own directions with both scatterers at least `d_min` away
  (`fallback` = 1; the ray is lengthened, `delay` keeps its value)
- LOS ray on the direct line, GR ray at the specular point on z = 0, rays with zero power at all frequencies at
  the midpoint of the direct line
- Writes `fbs_pos`, `lbs_pos` and `fallback` of all rays and overwrites given scatterers; nothing else changes;
  afterwards `b_med()` and `b_eff()` give the validity scales of [[ssf_paths]]

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
    // STUB: no computation yet
    (void)d_min;
    const arma::uword n = n_ray();
    if (fbs_pos.n_rows != 3 || fbs_pos.n_cols != n)
        fbs_pos.set_size(3, n);
    if (lbs_pos.n_rows != 3 || lbs_pos.n_cols != n)
        lbs_pos.set_size(3, n);
    if (fallback.n_elem != n)
        fallback.set_size(n);
    fbs_pos.zeros(), lbs_pos.zeros(), fallback.zeros();
}
