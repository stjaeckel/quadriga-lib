// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"

#include <string>

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# gbsm_single_link_channel
Channel coefficients of one link at one frequency over one segment (geometry-based stochastic model)

- Per snapshot [[get_channels_planar]] (`"phase"`, arrays at the positions of `rays`) or
  [[get_channels_spherical]] (`"drift"`) with the paths of [[ssf_paths]]
- `"taps"`: rays summed per tap, LOS and GR taps always present, tap count fixed per call; `"rays"`: one path per
  ray
- Validity: see [[ssf_paths]]; the channel's `par` reports the largest displacement relative to the limit

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
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"` (phase for the TR methods, drift for quadriga)
- **`output`** *(optional)* — `"taps"` or `"rays"`
- **`i_freq`** *(optional)* — Frequency index in `rays.center_frequency`, 0-based
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx positions in [m]; omitted: the positions of `rays`;
  `[n_snap, 3]` or `[1, 3]`
- **`tx_orientation`**, **`rx_orientation`** *(optional)* — Euler angles (bank, tilt, heading) in [rad]; omitted: 0;
  `[n_snap, 3]` or `[1, 3]`
- **`kf_ratio`** *(optional)* — K(snapshot) / K(drop); omitted: 1; `[n_snap]` or `[1]`
- **`oxygen`** *(optional)* — TR 7.6.1 oxygen absorption from each path's own length
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays

## Returns:
- Channel object with `n_snap` snapshots

## See also:
- [[ssf_paths]] (paths of one link at one position)
- [[gbsm_rays]] (input data)
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
    // STUB: no computation yet; returns an empty channel
    (void)tx_array, (void)rx_array, (void)rays, (void)link_gain_dB, (void)mobility, (void)output, (void)i_freq;
    (void)tx_pos, (void)rx_pos, (void)tx_orientation, (void)rx_orientation, (void)kf_ratio, (void)oxygen;
    (void)use_absolute_delays;
    return quadriga_lib::channel<double>();
}
