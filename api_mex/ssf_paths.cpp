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

- Computes the antenna-free paths (gains, lengths, polarization matrices, angles and scatterers) of the kept rays of
  `rays` (see Terms) at new Tx and Rx positions (default: the ray-data positions), LOS and GR also at zero power; paths
  in ray order, `ray_ind` maps them to the rays (e.g. `rays.tap(ray_ind)`)
- `'phase'`: use the outputs with [[get_channels_planar]] and the arrays at the positions of `rays`; the displacement
  phase of TR 38.901 eq. (7.5-22) is in `M`. With the arrays at the new positions the LOS path is no longer identified
  and the displacement phase counts twice, without an error
- `'drift'`: use the outputs at the new positions with [[get_channels_spherical]] or [[get_channels_multifreq]]; needs
  the scatterers of [[ssf_scatterers]]

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
  K-factor of the coefficient powers changes by `kf_ratio`, LOS + NLOS keep their sum; times `10^(link_gain_dB / 10)`;
  with `oxygen` times the absorption of TR 38.901 clause 7.6.1 (Table 7.6.1-1 at each frequency) over each path's own
  length, without the delay offset τ_Δ of eq. (7.6-1) and at the carrier only (no in-band variation of eq. (7.6-2))
- Validity per moving end, with `b_eff` and `b_med` of [[ssf_scatterers]]: `'phase'` up to about `sqrt(λ · b_eff / 8)`,
  typically 0.1 to 1 m; `'drift'` up to about `0.2 · b_med`, typically 1.5 to 12 m, new segment from `0.5 · b_med`; in
  `'drift'` each path length changes by at most the displacement. [[gbsm_single_link_channel]] reports the
  displacement relative to the limit
- An error is raised for invalid input; in `'drift'` also without scatterers, and for a GR ray at moved ends without
  permittivity in `rays` or with an end below z = 0

## Usage:
```
[ path_gain, path_length, M, aod, eod, aoa, eoa, fbs_pos, lbs_pos, path_type, ray_ind ] = ...
    quadriga_lib.ssf_paths( rays, link_gain_dB, mobility, tx_pos, rx_pos, kf_ratio, oxygen );
```

## Inputs:
- **`rays`** — Ray struct of the link, see [[ssf_rays]]; `[1, 1]` struct
- **`link_gain_dB`** — Link gain (see Terms) at the position in [dB], per frequency; 0 for data from paths; `[n_freq]`
- **`mobility`** — `'phase'`, `'drift'` or `'auto'` (see Terms; `'auto'` is `'drift'` for `'quadriga'` rays, which then
  need [[ssf_scatterers]]); default: `'auto'`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in [m]; `[3]`; default: the positions of `rays`
- **`kf_ratio`** — KF(position) / KF(drop), >= 0, per frequency; `[n_freq]`; default: 1
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `false`

## Outputs:
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path, 1]`
- **`M`** — Polarization transfer matrices, rows as in [[get_channels_planar]]; `[8, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Departure and arrival angles in [rad], GCS; `[n_path, 1]`
- **`fbs_pos`**, **`lbs_pos`** — Scatterer positions in [m]; `[3, n_path]`, `[3, 0]` in `'phase'`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS; uint32; `[n_path, 1]`
- **`ray_ind`** — Index of the ray in `rays`, 1-based; uint64; `[n_path, 1]`

## See also:
- [[ssf_scatterers]] (scatterers for `'drift'`, validity scales `b_eff` and `b_med`)
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
