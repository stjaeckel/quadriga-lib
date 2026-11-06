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
# SSF_PATHS
Antenna-free paths of one link at one position

- `'phase'`: use the outputs with [[get_channels_planar]] and the arrays at the positions of `rays`; the
  displacement phase of TR 38.901 eq. (7.5-22) is in `M`. With the arrays at the new positions the LOS path is no
  longer identified and the displacement phase counts twice, without an error
- `'drift'`: use the outputs at the new positions with [[get_channels_spherical]] or [[get_channels_multifreq]];
  needs the scatterers of [[ssf_scatterers]]
- Kept rays: LOS and GR rays always (also at zero power), NLOS rays with power > 0 at some frequency; paths in ray
  order, `ray_ind` maps them to the rays (e.g. `rays.tap(ray_ind)`, `rays.fallback(ray_ind)`)

| Quantity | `'phase'` | `'drift'` |
| --- | --- | --- |
| Path length | Tx-Rx distance of `rays` + c·delay | Polyline Tx, FBS, LBS, Rx; LOS direct line; GR via the specular point |
| Angles | Ray data | Geometry: towards FBS and LBS, LOS direct line, GR towards the image points |
| `M` | Ray `M` times the displacement phase | Ray `M`; GR recomputed when an end moved |
| `fbs_pos`, `lbs_pos` | `[3, 0]` | Scatterers of the rays; LOS midpoint of the direct line; GR specular point |

- `'drift'` with an end moved (any coordinate differs from `rays`): GR power by the power law of `rays.method`
  (`(d3D / d_GR)²` for the TR methods, 1 for `'quadriga'`), GR `M` the scalar multiple of `diag(R_par, -R_perp)` (TR
  38.901 eqs. (7.6-38), (7.6-39)) that best fits the stored matrix, which replaces a matrix of another form (e.g. a
  conjugated one). Fallback rays have their longer polyline, also at the positions of `rays`
- Path gain per frequency: the coefficient power; with `kf_ratio` LOS and GR are rescaled against NLOS so that the
  K-factor of the coefficient powers changes by `kf_ratio`, LOS + NLOS keep their sum; times
  `10^(link_gain_dB / 10)`; with `oxygen` times the absorption of TR 38.901 clause 7.6.1 (Table 7.6.1-1 at each
  frequency) over each path's own length. Deviations: τ_Δ is not added (at 60 GHz: 0.017 dB median and up to 1.28 dB
  at p99 more loss on NLOS rays of the TR methods without absolute TOA; sub-cluster rays up to 0.13 dB), eq. (7.6-2)
  is not applied across the band
- Validity per moving end: `'phase'` up to about `sqrt(λ · b_eff / 8)` (`b_eff` of [[ssf_scatterers]]; p50 over the
  links for an Rx move: TR NLOS 0.27 to 0.36 m at 3.5 GHz and 0.12 to 0.13 m at 28 GHz, `'quadriga'` NLOS 0.24 to
  0.77 m and 0.24 to 0.31 m, LOS 0.91 to 1.33 m and 0.33 to 0.40 m); `'drift'` up to about `0.2 · b_med` (p50 1.5 to
  5.6 m LOS, 7.8 to 12.4 m NLOS, about 20 % less when both ends move; new segment from `0.5 · b_med`); in `'drift'`
  each path length changes by at most the displacement
- Errors in `'drift'`: no scatterers; a GR ray at moved ends without permittivity in `rays` or with an end below z = 0

## Usage:
```
[ path_gain, path_length, M, aod, eod, aoa, eoa, fbs_pos, lbs_pos, path_type, ray_ind ] = ...
    quadriga_lib.ssf_paths( rays, link_gain_dB, mobility, tx_pos, rx_pos, kf_ratio, oxygen );
```

## Inputs:
- **`rays`** — Ray struct of the link, see [[ssf_rays]]; `[1, 1]` struct
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at the position in [dB]; 0 for data from paths; `[n_freq]`
- **`mobility`** *(optional)* — `'phase'`, `'drift'` or `'auto'` (default; phase for the TR methods, drift for
  `'quadriga'`)
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx position in [m]; omitted: the positions of `rays`; `[3]`
- **`kf_ratio`** *(optional)* — K(position) / K(drop), >= 0; omitted: 1; `[n_freq]`
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1; default: `false`

## Outputs:
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path, 1]`
- **`M`** — Polarization transfer matrices, rows as in [[get_channels_planar]]; `[8, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Departure and arrival angles in [rad], GCS; `[n_path, 1]`
- **`fbs_pos`**, **`lbs_pos`** — Scatterer positions in [m]; `[3, n_path]`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS (a code, not shifted); uint32; `[n_path, 1]`
- **`ray_ind`** — Index of the ray in `rays`, 1-based; uint64; `[n_path, 1]`

## See also:
- [[ssf_scatterers]] (scatterers, validity scales `b_eff` and `b_med`)
- [[gbsm_single_link_channel]] (channel coefficients of one link over a segment)
- [[gbsm_rays_from_paths]] (ray data from path data, e.g. of a QRT file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    if (!mxIsStruct(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'rays' must be a 1x1 struct.");

    // Read inputs
    const auto rays = qd_mex_struct2gbsm_rays(prhs[0], 0, false, "rays");
    const arma::vec link_gain_dB = qd_mex_get_Col<double>(prhs[1], false, "link_gain_dB");
    const std::string mobility = (nrhs < 3) ? std::string("auto") : qd_mex_get_string(prhs[2], "auto");
    const arma::vec tx_pos = (nrhs < 4) ? arma::vec() : qd_mex_get_Col<double>(prhs[3], false, "tx_pos");
    const arma::vec rx_pos = (nrhs < 5) ? arma::vec() : qd_mex_get_Col<double>(prhs[4], false, "rx_pos");
    const arma::vec kf_ratio = (nrhs < 6) ? arma::vec() : qd_mex_get_Col<double>(prhs[5], false, "kf_ratio");
    const bool oxygen = (nrhs < 7) ? false : qd_mex_get_scalar<bool>(prhs[6], "oxygen", false);

    // Outputs of unknown size (n_path), copied after the call; optional outputs only when requested
    arma::mat path_gain, fbs_pos, lbs_pos;
    arma::vec path_length, aod, eod, aoa, eoa;
    arma::cube M;
    arma::u32_vec path_type;
    arma::uvec ray_ind;
    CALL_QD(quadriga_lib::ssf_paths(rays, link_gain_dB, path_gain, path_length, M, mobility,
                                    tx_pos.empty() ? nullptr : &tx_pos, rx_pos.empty() ? nullptr : &rx_pos,
                                    kf_ratio.empty() ? nullptr : &kf_ratio, (nlhs > 3) ? &aod : nullptr,
                                    (nlhs > 4) ? &eod : nullptr, (nlhs > 5) ? &aoa : nullptr,
                                    (nlhs > 6) ? &eoa : nullptr, (nlhs > 7) ? &fbs_pos : nullptr,
                                    (nlhs > 8) ? &lbs_pos : nullptr, (nlhs > 9) ? &path_type : nullptr,
                                    (nlhs > 10) ? &ray_ind : nullptr, oxygen));

    // Empty outputs keep their size: matrices as returned (e.g. 'fbs_pos' [3, 0] in "phase"), vectors [0, 1], 'M'
    // [8, 0, n_freq]
    auto mat = [](const arma::mat &x)
    { return x.empty() ? mxCreateNumericMatrix(x.n_rows, x.n_cols, mxDOUBLE_CLASS, mxREAL) : qd_mex_copy2matlab(&x); };
    auto col = [](const auto &x)
    {
        using T = typename std::decay_t<decltype(x)>::elem_type;
        return x.empty() ? mxCreateNumericMatrix(0, 1, qd_mex_class_id<T>(), mxREAL) : qd_mex_copy2matlab(&x);
    };
    if (nlhs > 0)
        plhs[0] = mat(path_gain);
    if (nlhs > 1)
        plhs[1] = col(path_length);
    if (nlhs > 2)
    {
        const mwSize d[3] = {8, 0, (mwSize)M.n_slices};
        plhs[2] = M.empty() ? mxCreateNumericArray(3, d, mxDOUBLE_CLASS, mxREAL) : qd_mex_copy2matlab(&M);
    }
    arma::vec *ang[4] = {&aod, &eod, &aoa, &eoa};
    for (int k = 3; k < 7 && k < nlhs; ++k)
        plhs[k] = col(*ang[k - 3]);
    if (nlhs > 7)
        plhs[7] = mat(fbs_pos);
    if (nlhs > 8)
        plhs[8] = mat(lbs_pos);
    if (nlhs > 9)
        plhs[9] = col(path_type); // Codes, not shifted
    if (nlhs > 10)
    {
        ray_ind += 1; // 0-based (C++) -> 1-based (MATLAB)
        plhs[10] = col(ray_ind);
    }
}
