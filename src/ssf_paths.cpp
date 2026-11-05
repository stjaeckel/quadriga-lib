// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"

#include <string>

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_paths
Antenna-free paths of one link at one position

- `"phase"`: use the outputs with [[get_channels_planar]] and the arrays at the positions of `rays` (the
  displacement phase is in `M`); `"drift"`: inputs of [[get_channels_spherical]], [[get_channels_multifreq]] and
  [[qrt_paths_from_scatterers]]
- Rays with zero power at all frequencies are dropped, LOS and GR rays are kept
- Validity: `"phase"` up to about `sqrt(lambda · b_eff / 8)` per moving end, `"drift"` up to about `0.2 · b_med`
  per moving end (see [[gbsm_rays]])

## Declaration:
```
void quadriga_lib::ssf_paths(
    const gbsm_rays &rays,
    const arma::vec &link_gain_dB,
    arma::mat &path_gain,
    arma::vec &path_length,
    arma::cube &M,
    const std::string &mobility = "auto",
    const arma::vec *tx_pos = nullptr,
    const arma::vec *rx_pos = nullptr,
    const arma::vec *kf_ratio = nullptr,
    arma::vec *aod = nullptr,
    arma::vec *eod = nullptr,
    arma::vec *aoa = nullptr,
    arma::vec *eoa = nullptr,
    arma::mat *fbs_pos = nullptr,
    arma::mat *lbs_pos = nullptr,
    arma::uvec *path_type = nullptr,
    arma::uvec *ray_ind = nullptr,
    bool oxygen = false);
```

## Inputs:
- **`rays`** — Ray data of the link, see [[gbsm_rays]]
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at the position in [dB]; 0 for data from paths; `[n_freq]`
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"` (phase for the TR methods, drift for quadriga)
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx position in [m]; omitted: the positions of `rays`; `[3]`
- **`kf_ratio`** *(optional)* — K(position) / K(drop); omitted: 1; `[n_freq]`
- **`oxygen`** *(optional)* — TR 7.6.1 oxygen absorption from each path's own length

## Outputs:
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`M`** — Polarization transfer matrices; `[8, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** *(optional)* — Departure and arrival angles in [rad]; `[n_path]`
- **`fbs_pos`**, **`lbs_pos`** *(optional)* — Scatterer positions in [m]; `[3, n_path]`, `"phase"`: `[3, 0]`
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS; `[n_path]`
- **`ray_ind`** *(optional)* — Index of the ray in `rays`, 0-based; `[n_path]`

## See also:
- [[gbsm_rays]] (input data, validity scales)
- [[gbsm_single_link_channel]] (channel coefficients of one link)
MD!*/

void quadriga_lib::ssf_paths(const gbsm_rays &rays,
                             const arma::vec &link_gain_dB,
                             arma::mat &path_gain,
                             arma::vec &path_length,
                             arma::cube &M,
                             const std::string &mobility,
                             const arma::vec *tx_pos,
                             const arma::vec *rx_pos,
                             const arma::vec *kf_ratio,
                             arma::vec *aod,
                             arma::vec *eod,
                             arma::vec *aoa,
                             arma::vec *eoa,
                             arma::mat *fbs_pos,
                             arma::mat *lbs_pos,
                             arma::uvec *path_type,
                             arma::uvec *ray_ind,
                             bool oxygen)
{
    // STUB: no computation yet; outputs sized with zeros
    (void)link_gain_dB, (void)mobility, (void)tx_pos, (void)rx_pos, (void)kf_ratio, (void)oxygen;

    const arma::uword n_path = rays.n_ray(), n_freq = rays.n_freq();
    path_gain.zeros(n_path, n_freq), path_length.zeros(n_path), M.zeros(8, n_path, n_freq);
    for (arma::vec *v : {aod, eod, aoa, eoa})
        if (v != nullptr)
            v->zeros(n_path);
    for (arma::mat *m : {fbs_pos, lbs_pos})
        if (m != nullptr)
            m->zeros(3, n_path);
    for (arma::uvec *u : {path_type, ray_ind})
        if (u != nullptr)
            u->zeros(n_path);
}
