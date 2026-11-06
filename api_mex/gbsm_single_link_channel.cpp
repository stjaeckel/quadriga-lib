// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# GBSM_SINGLE_LINK_CHANNEL
Channel coefficients of one link at one frequency over one segment (geometry-based stochastic model)

- Snapshot t: the paths of [[ssf_paths]] at the positions of t, at `rays.center_frequency(i_freq)` with the link
  gain and K ratio of t, for the rays with power > 0 at `i_freq` plus LOS and GR
- `'phase'`: [[get_channels_planar]] with the arrays at the positions of `rays` and the orientations of t; delays,
  path lengths and angles stay those of the ray-data positions. `'drift'`: [[get_channels_spherical]] with the
  arrays at the positions and orientations of t
- `'taps'`: the rays summed per tap index, taps in ascending index, one tap per distinct index of the kept rays
  (fixed per call); LOS and GR taps are present whenever `rays` holds them, also at zero power. The delay of a tap
  per element pair is the mean of the rays' delays weighted by `rays.power` at `i_freq` (equal weights where they sum
  to 0), `[n_rx, n_tx, n_tap]`; reduce it to `[1, 1, n_tap]` where needed. TR 38.901 LOS gives two taps at delay 0
- `'taps'` is exact at the phase centre for the TR methods (the rays of a tap share one delay); for `'quadriga'` only
  at the carrier: in UMi NLOS a tap varies over the band by 0.18, 0.52 and 0.79 (in-band selectivity) for 20, 100 and
  400 MHz. `'rays'`: one path per kept ray in ray order, exact at any bandwidth; the path data are the outputs of
  [[ssf_paths]] at t (in `'phase'` with the displacement phase in `M`, for the arrays at the ray-data positions)
- `n_snap` is the largest row count of the positions and orientations (1 if all are omitted); each has 1 or `n_snap`
  rows, `link_gain_dB` and `kf_ratio` 1 or `n_snap` entries; one row or entry applies to all snapshots. One position
  is a `[1, 3]` row; a `[3, 1]` column is rejected. A static link with a gain per snapshot repeats its position
- `use_absolute_delays` in `'phase'` adds the LOS delay of the ray-data positions
- No power correction and no initial position; validity limits and their meaning: see [[ssf_paths]]
- Errors: an invalid array (message after `Transmit antenna: ` or `Receive antenna: `), the errors of [[ssf_paths]]
  in `'drift'`, a row or entry count other than 1 or `n_snap`
- `chan` and `par` are copies of the library result

## Fields of par:

| Name | Size | Content |
| --- | --- | --- |
| `tap` | `[n_tap, 1]` or `[n_path, 1]` | Tap index per output tap or per path (a code, not shifted); uint32 |
| `n_ray_tap` | `[n_tap, 1]` | Number of kept rays per tap; uint32; `'taps'` only |
| `fbs_pos`, `lbs_pos` | `[3, n_kept]` | Scatterers of the kept rays at the ray-data positions; `'drift'` with `'taps'` only; single, the precision of path data in HDF5 files |
| `gr_pos` | `[3, n_snap]` | GR point per snapshot; `'drift'` with `'taps'` and a GR ray only; single |
| `link_gain_dB` | `[n_snap, 1]` | Link gain per snapshot |
| `b_med`, `b_eff` | `[2, 1]` | Validity scales of [[ssf_scatterers]] at `i_freq`, Tx and Rx; without scatterers from a solved copy (`d_min` = 2 m) |
| `limit` | `[2, 1]` | Validity limit per end, $\sqrt{\lambda b_{\mathrm{eff}} / 8}$ (`'phase'`) or $0.2 b_{\mathrm{med}}$ (`'drift'`); Inf where b is NaN |
| `ratio` | `[2, 1]` | Largest distance of the end from its ray-data position over the snapshots, divided by `limit` |
| `ratio_total` | scalar | $\sqrt{r_{\mathrm{Tx}}^2 + r_{\mathrm{Rx}}^2}$ of the two ratios; valid up to 1; `'drift'`: new segment from 2.5 |

## Usage:
```
[ chan, par ] = quadriga_lib.gbsm_single_link_channel( tx_array, rx_array, rays, link_gain_dB, mobility, ...
    output, i_freq, tx_pos, rx_pos, tx_orientation, rx_orientation, kf_ratio, oxygen, use_absolute_delays );
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Structs containing the arrayant data with `n_tx` and `n_rx` ports; field layout
  as documented in [[arrayant_generate]]
- **`rays`** — Ray struct of the link, see [[ssf_rays]]; `[1, 1]` struct
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at `rays.center_frequency(i_freq)` in [dB]; `[n_snap]` or `[1]`
- **`mobility`** *(optional)* — `'phase'`, `'drift'` or `'auto'` (default; phase for the TR methods, drift for
  `'quadriga'`)
- **`output`** *(optional)* — `'taps'` (default) or `'rays'`
- **`i_freq`** *(optional)* — Frequency index in `rays.center_frequency`, 1-based; default: 1
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx positions in [m]; omitted: the positions of `rays`;
  `[n_snap, 3]` or `[1, 3]`
- **`tx_orientation`**, **`rx_orientation`** *(optional)* — Euler angles (bank, tilt, heading) in [rad]; omitted:
  0; `[n_snap, 3]` or `[1, 3]`
- **`kf_ratio`** *(optional)* — K(snapshot) / K(drop), >= 0; omitted: 1; `[n_snap]` or `[1]`
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1, see [[ssf_paths]]; default: `false`
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays; default: `false`

## Outputs:
- **`chan`** — Channel struct, layout as documented in [[hdf5_read_channel]]: `center_frequency` scalar;
  `tx_position`, `rx_position` (the snapshot positions), `tx_orientation`, `rx_orientation` `[3, n_snap]`;
  `coeff_re`, `coeff_im`, `delay` `[n_rx, n_tx, n_tap, n_snap]` (`'taps'`) or `[n_rx, n_tx, n_path, n_snap]`
  (`'rays'`); `'rays'` also `path_gain`, `path_length`, `path_polarization`, `path_angles` and in `'drift'`
  `fbs_pos`, `lbs_pos`
- **`par`** — Struct with the fields above

## See also:
- [[ssf_paths]] (paths of one link at one position, validity limits)
- [[ssf_scatterers]] (scatterers, `b_med` and `b_eff`)
- [[gbsm_rays_from_paths]] (ray data from path data, e.g. of a QRT file; taps 0, 3, 6, ... without `tap`)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4 || nrhs > 14)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    if (!mxIsStruct(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'rays' must be a 1x1 struct.");

    // Read inputs; the library validates the arrays
    const auto tx_array = qd_mex_struct2arrayant(prhs[0]);
    const auto rx_array = qd_mex_struct2arrayant(prhs[1]);
    const auto rays = qd_mex_struct2gbsm_rays(prhs[2], 0, false, "rays");
    const arma::vec link_gain_dB = qd_mex_get_Col<double>(prhs[3], false, "link_gain_dB");
    const std::string mobility = (nrhs < 5) ? std::string("auto") : qd_mex_get_string(prhs[4], "auto");
    const std::string output = (nrhs < 6) ? std::string("taps") : qd_mex_get_string(prhs[5], "taps");
    arma::uword i_freq = (nrhs < 7) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[6], "i_freq", 1);
    const arma::mat tx_pos = (nrhs < 8) ? arma::mat() : qd_mex_get_Mat<double>(prhs[7], false, "tx_pos");
    const arma::mat rx_pos = (nrhs < 9) ? arma::mat() : qd_mex_get_Mat<double>(prhs[8], false, "rx_pos");
    const arma::mat tx_ori = (nrhs < 10) ? arma::mat() : qd_mex_get_Mat<double>(prhs[9], false, "tx_orientation");
    const arma::mat rx_ori = (nrhs < 11) ? arma::mat() : qd_mex_get_Mat<double>(prhs[10], false, "rx_orientation");
    const arma::vec kf_ratio = (nrhs < 12) ? arma::vec() : qd_mex_get_Col<double>(prhs[11], false, "kf_ratio");
    const bool oxygen = (nrhs < 13) ? false : qd_mex_get_scalar<bool>(prhs[12], "oxygen", false);
    const bool use_absolute_delays = (nrhs < 14) ? false : qd_mex_get_scalar<bool>(prhs[13], "use_absolute_delays", false);

    if (i_freq == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'i_freq' cannot be 0 (1-based index).");
    i_freq -= 1; // 1-based (MATLAB) -> 0-based (C++)

    // The result is returned by value and copied
    quadriga_lib::channel<double> chan;
    CALL_QD(chan = quadriga_lib::gbsm_single_link_channel(tx_array, rx_array, rays, link_gain_dB, mobility, output,
                                                          i_freq, tx_pos.empty() ? nullptr : &tx_pos,
                                                          rx_pos.empty() ? nullptr : &rx_pos,
                                                          tx_ori.empty() ? nullptr : &tx_ori,
                                                          rx_ori.empty() ? nullptr : &rx_ori,
                                                          kf_ratio.empty() ? nullptr : &kf_ratio, oxygen,
                                                          use_absolute_delays));

    if (nlhs > 0)
        plhs[0] = qd_mex_channel2struct({chan});
    if (nlhs > 1)
    {
        plhs[1] = qd_mex_make_struct(chan.par_names, 1);
        for (size_t n = 0; n < chan.par_names.size(); ++n)
            qd_mex_set_field(plhs[1], chan.par_names[n], qd_mex_any2matlab(chan.par_data[n])); // 'tap': codes, not shifted
    }
}
