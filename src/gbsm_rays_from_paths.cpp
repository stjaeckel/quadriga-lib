// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"

#include <string>

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm_rays_from_paths
Ray data of one link from spherical path data (e.g. read with qrt_file_read)

- Delays from the path lengths, angles from the scatterer positions, power = path gain
- Fills a pre-initialised object in place; use the result with `link_gain_dB = 0` in [[ssf_paths]] and
  [[gbsm_single_link_channel]]

## Declaration:
```
void quadriga_lib::gbsm_rays_from_paths(
    const arma::vec &center_frequency,
    const arma::vec &tx_pos,
    const arma::vec &rx_pos,
    const arma::mat &fbs_pos,
    const arma::mat &lbs_pos,
    const arma::mat &path_gain,
    const arma::vec &path_length,
    const arma::cube &M,
    gbsm_rays &rays,
    const std::string &method = "quadriga",
    const arma::uvec *path_type = nullptr,
    const arma::uvec *tap = nullptr,
    const arma::vec *gr_eps_re = nullptr,
    const arma::vec *gr_eps_im = nullptr);
```

## Inputs:
- **`center_frequency`** — Frequencies in [Hz]; `[n_freq]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in [m]; `[3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in [m]; `[3, n_path]`
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`M`** — Polarization transfer matrices; `[8, n_path, n_freq]`
- **`method`** *(optional)* — Value of `rays.method`: GR power law and default mobility
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS; omitted: LOS where FBS = LBS on the direct
  line, else NLOS; `[n_path]`
- **`tap`** *(optional)* — Tap index; omitted: one tap per path; `[n_path]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity; needed where a GR path is recomputed;
  `[n_freq]`

## Output:
- **`rays`** — Ray data with `n_ray = n_path`, see [[gbsm_rays]]

## See also:
- [[qrt_file_read]] (reads path data from QRT files)
- [[ssf_paths]] (paths of one link at one position)
MD!*/

void quadriga_lib::gbsm_rays_from_paths(const arma::vec &center_frequency,
                                        const arma::vec &tx_pos,
                                        const arma::vec &rx_pos,
                                        const arma::mat &fbs_pos,
                                        const arma::mat &lbs_pos,
                                        const arma::mat &path_gain,
                                        const arma::vec &path_length,
                                        const arma::cube &M,
                                        gbsm_rays &rays,
                                        const std::string &method,
                                        const arma::uvec *path_type,
                                        const arma::uvec *tap,
                                        const arma::vec *gr_eps_re,
                                        const arma::vec *gr_eps_im)
{
    // STUB: no computation yet; outputs sized with zeros
    (void)tx_pos, (void)rx_pos, (void)lbs_pos, (void)path_gain, (void)path_length, (void)M, (void)path_type;
    (void)tap, (void)gr_eps_re, (void)gr_eps_im;

    const arma::uword n_path = fbs_pos.n_cols, n_freq = center_frequency.n_elem;
    rays.method = method;
    rays.tx_pos.zeros(3), rays.rx_pos.zeros(3), rays.center_frequency.zeros(n_freq);
    rays.delay.zeros(n_path), rays.power.zeros(n_path, n_freq);
    rays.aod.zeros(n_path), rays.eod.zeros(n_path), rays.aoa.zeros(n_path), rays.eoa.zeros(n_path);
    rays.M.zeros(8, n_path, n_freq), rays.path_type.zeros(n_path), rays.tap.zeros(n_path);
    rays.fbs_pos.zeros(3, n_path), rays.lbs_pos.zeros(3, n_path);
    if (!rays.gr_eps_re.empty())
        rays.gr_eps_re.reset();
    if (!rays.gr_eps_im.empty())
        rays.gr_eps_im.reset();
    if (!rays.fallback.empty())
        rays.fallback.reset();
}
