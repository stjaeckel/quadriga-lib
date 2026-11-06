// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# LOS_PROBABILITY
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

- Evaluates the probability that a link is in line-of-sight (LOS) from its 2D distance and the BS
  and UT heights, and, when `rnd_g` and `center_freq` are given, the soft LOS state of TR 38.901 eq.
  (7.6-18); the LOS state itself (a draw against `p_los`) and its spatial correlation are the
  caller's
- `38.901_*`: TR 38.901 v19.5.0 Table 7.4.2-1 (`InH_Mixed`, `InH_Open`: mixed and open office).
  Where the table leaves cases open: all results are limited to [0, 1]; UMa uses `C' = 1` for `hUT >
  23` m; InF-SH and InF-DH give 1 for `hUT >= h_clutter` and the rule of InF-SL and InF-DL for `hBS
  <= hUT`; SMa gives 1 for `hBS <= hUT` and uses `max(h_x - hUT, 0)` in all three terms (the TR only
  in the residential term; equal for `hUT < 15` m)
- `38.811_*`: TR 38.811 Table 6.6.1-1 over the elevation `atan2(hBS, d2d_out)`, linearly
  interpolated; `Suburban` and `Rural` share one table
- `QuaDRiGa_Industrial`: `min(1, exp(-(d2d_out + 20)/80)·exp(min(hBS, 8)/5 - 1))`; `LOS`: 1; `NLOS`:
  0
- The `LOS_state_lambda` of the InF conf files is `d_clutter/2` of the TR defaults; with another
  `d_clutter` pass a matching correlation distance

## Usage:
```
[ p_los, los_soft ] = quadriga_lib.los_probability( model, d2d_out, h_bs, h_ut, ...
    d_clutter, r_clutter, h_clutter, rnd_g, center_freq );
```

## Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`,
  `38.901_InH_Open`, `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`,
  `38.901_InF_HH`, `38.901_SMa`, `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`,
  `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in [m], finite and >= 0: `d2D - d2d_in` for O2I UTs, clamped
  at 0 by the caller; the 2D distance in indoor scenarios; `[n_link]`
- **`h_bs`** — BS height in [m], finite; `[n_link]` or `[1]` for all links
- **`h_ut`** — UT height in [m], finite; `[n_link]` or `[1]` for all links
- **`d_clutter`** — Clutter size in [m] of the InF and SMa models, > 0; NaN or empty = TR default
  (InF-SL and InF-SH 10 m, InF-DL and InF-DH 2 m, SMa 30 m); default: NaN
- **`r_clutter`** — Clutter density of the InF models (default 0.2 for SL and SH, 0.6 for DL and
  DH), vegetation density of SMa (default 0; 0.1 sparse, 0.2 dense); in [0, 1); NaN or empty =
  default; default: NaN
- **`h_clutter`** — Clutter height in [m] of InF-SH (default 2 m) and InF-DH (6 m), >= 0; NaN or
  empty = default. The other models ignore the clutter arguments, which are validated for every
  model; default: NaN
- **`rnd_g`** *(optional)* — Normal(0,1) value per link, G of eq. (7.6-18); `[n_link]`
- **`center_freq`** *(optional)* — Carrier frequencies in [Hz], finite and > 0 (messages name it
  `center_frequency`); `[n_freq]`

## Outputs:
- **`p_los`** — LOS probability in [0, 1]; `[n_link, 1]`
- **`los_soft`** — Soft LOS state in [0, 1]; `> 0.5` exactly when `rnd_g > -F(p_los)` of eq.
  (7.6-18), which has the probability `p_los`; `[n_link, n_freq]`; `[]` if `rnd_g` or `center_freq`
  is omitted or empty

## See also:
- [[o2i_loss]] (indoor distance for `d2d_out`)
- [[sos_rand]], [[sos_randn]] (spatially consistent LOS-state draw and `rnd_g`)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 4 || nrhs > 9)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string model = qd_mex_get_string(prhs[0]);
    const arma::vec d2d_out = qd_mex_get_Col<double>(prhs[1]);
    const arma::vec h_bs = qd_mex_get_Col<double>(prhs[2]);
    const arma::vec h_ut = qd_mex_get_Col<double>(prhs[3]);
    const double d_clutter = (nrhs < 5) ? NAN : qd_mex_get_scalar<double>(prhs[4], "d_clutter", NAN);
    const double r_clutter = (nrhs < 6) ? NAN : qd_mex_get_scalar<double>(prhs[5], "r_clutter", NAN);
    const double h_clutter = (nrhs < 7) ? NAN : qd_mex_get_scalar<double>(prhs[6], "h_clutter", NAN);
    const arma::vec rnd_g = (nrhs < 8) ? arma::vec() : qd_mex_get_Col<double>(prhs[7]);
    const arma::vec center_freq = (nrhs < 9) ? arma::vec() : qd_mex_get_Col<double>(prhs[8]);

    // Output allocation; the soft LOS state needs both 'rnd_g' and 'center_freq', else it is returned as []
    const bool soft = nlhs > 1 && !rnd_g.empty() && !center_freq.empty();
    arma::vec p_los;
    arma::mat los_soft;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&p_los, d2d_out.n_elem);
    if (soft)
        plhs[1] = qd_mex_init_output(&los_soft, d2d_out.n_elem, center_freq.n_elem);
    else if (nlhs > 1)
        plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);

    // Wrap optional pointers
    const arma::vec *p_rnd_g = rnd_g.empty() ? nullptr : &rnd_g;
    const arma::vec *p_center_freq = center_freq.empty() ? nullptr : &center_freq;
    arma::mat *p_los_soft = soft ? &los_soft : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::los_probability(model, d2d_out, h_bs, h_ut, p_los, d_clutter, r_clutter, h_clutter,
                                          p_rnd_g, p_center_freq, p_los_soft));
}
