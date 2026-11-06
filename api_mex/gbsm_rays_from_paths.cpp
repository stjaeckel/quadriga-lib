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
# GBSM_RAYS_FROM_PATHS
Ray data of one link from spherical path data (e.g. read with qrt_file_read)

- Returns the ray struct of [[ssf_rays]] (`fallback` empty) from path data in the names and order of
  [[qrt_file_read]], read with `downlink = true` (an uplink read swaps the ends and conjugates `M`)
- `delay` = (path length - Tx-Rx distance) / c, `power` = path gain. Angles point from the Tx to the first of
  `fbs_pos`, `lbs_pos`, `rx_pos` that differs from it and from the Rx to the first of `lbs_pos`, `fbs_pos`, `tx_pos`;
  GR angles to the image of the other end (z mirrored); 0 if `tx_pos` = `rx_pos`
- QRT stores no path types or taps. Without `path_type`: LOS where FBS = LBS on the direct line (0.1 mm), else
  NLOS; a GR path needs `path_type`
- The GR point (`lbs_pos`) must be within 1 mm of z = 0; permittivity only for `drift` away from `tx_pos`, `rx_pos`
- Use in [[ssf_paths]] and [[gbsm_single_link_channel]] with `link_gain_dB = 0` and without `kf_ratio`
- Messages name the argument `center_freq` as `'center_frequency'`

## Usage:
```
rays = quadriga_lib.gbsm_rays_from_paths( center_freq, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, ...
    path_length, M, method, path_type, tap, gr_eps_re, gr_eps_im );
```

## Inputs:
- **`center_freq`** — Frequencies in [Hz]; `[n_freq]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in [m]; `[3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in [m]; `[3, n_path]`
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`M`** — Polarization transfer matrices; `[8, n_path, n_freq]`
- **`method`** *(optional)* — Value of `rays.method`; sets the GR power law and the default mobility; default:
  `'quadriga'`
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS (a code, not shifted), at most one LOS and
  one GR; `[n_path]`
- **`tap`** *(optional)* — Tap index (a code, not shifted); omitted: LOS 0, GR 3, the others 6, 9, ... in path
  order; `[n_path]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity, given together; `[n_freq]`

## Output:
- **`rays`** — Ray struct with `n_ray = n_path`, see [[ssf_rays]]

## See also:
- [[qrt_file_read]] (source of the path data)
- [[ssf_paths]] (paths at other positions)
- [[gbsm_single_link_channel]] (channel coefficients of the link)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 8 || nrhs > 13)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::vec center_freq = qd_mex_get_Col<double>(prhs[0], false, "center_freq");
    const arma::vec tx_pos = qd_mex_get_Col<double>(prhs[1], false, "tx_pos");
    const arma::vec rx_pos = qd_mex_get_Col<double>(prhs[2], false, "rx_pos");
    // Empty arrays keep their dimensions, so that a link without paths ([3, 0], [0, n_freq], [8, 0, n_freq]) passes
    auto mat = [&](const mxArray *a, const char *name)
    { return mxIsEmpty(a) ? arma::mat(qd_mex_dim(a, 0), qd_mex_dim(a, 1)) : qd_mex_get_Mat<double>(a, false, name); };
    const arma::mat fbs_pos = mat(prhs[3], "fbs_pos"), lbs_pos = mat(prhs[4], "lbs_pos");
    const arma::mat path_gain = mat(prhs[5], "path_gain");
    const arma::vec path_length = qd_mex_get_Col<double>(prhs[6], false, "path_length");
    const mxArray *m = prhs[7];
    const arma::cube M = mxIsEmpty(m) ? arma::cube(qd_mex_dim(m, 0), qd_mex_dim(m, 1), qd_mex_dim(m, 2))
                                      : qd_mex_get_Cube<double>(m, false, "M");
    const std::string method = (nrhs < 9) ? std::string("quadriga") : qd_mex_get_string(prhs[8], "quadriga");
    const arma::uvec path_type = (nrhs < 10) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[9], false, "path_type");
    const arma::uvec tap = (nrhs < 11) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[10], false, "tap");
    const arma::vec gr_eps_re = (nrhs < 12) ? arma::vec() : qd_mex_get_Col<double>(prhs[11], false, "gr_eps_re");
    const arma::vec gr_eps_im = (nrhs < 13) ? arma::vec() : qd_mex_get_Col<double>(prhs[12], false, "gr_eps_im");

    // Output struct in MATLAB memory (n_path from 'fbs_pos'; wrong sizes throw in the library call before any write);
    // codes 'path_type' and 'tap' are not shifted
    std::vector<quadriga_lib::gbsm_rays> rays;
    const bool has_gr = !gr_eps_re.empty() && !gr_eps_im.empty();
    plhs[0] = qd_mex_init_gbsm_rays(1, fbs_pos.n_cols, center_freq.n_elem, has_gr, true, rays);

    CALL_QD(quadriga_lib::gbsm_rays_from_paths(center_freq, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, path_length,
                                               M, rays[0], method, path_type.empty() ? nullptr : &path_type,
                                               tap.empty() ? nullptr : &tap, gr_eps_re.empty() ? nullptr : &gr_eps_re,
                                               gr_eps_im.empty() ? nullptr : &gr_eps_im));

    mxSetField(plhs[0], 0, "method", mxCreateString(rays[0].method.c_str()));
}
