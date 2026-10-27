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

- Evaluates the probability that a link is in line-of-sight (LOS) for each link, from its 2D
  distance and the BS and UT heights, and optionally the soft LOS state of TR 38.901 eq. (7.6-18)
- Notation: `d = d2d_out(l)`, `hB = h_bs(l)`, `hU = h_ut(l)` in m, `lg = log10`, `ln` the natural
  logarithm. All results are limited to [0, 1]
- TR 38.901 v19.5.0 Table 7.4.2-1:
  - `38.901_RMa`: 1 for `d <= 10`, else `exp(-(d - 10)/1000)`
  - `38.901_UMi`: 1 for `d <= 18`, else `18/d + exp(-d/36)·(1 - 18/d)`
  - `38.901_UMa`: 1 for `d <= 18`, else
    `(18/d + exp(-d/63)·(1 - 18/d))·(1 + C'·5/4·(d/100)³·exp(-d/150))` with `C' = 0` for `hU <= 13`,
    `((hU - 13)/10)^1.5` for `13 < hU <= 23` and 1 for `hU > 23`; for `hU > 13` the formula exceeds
    1 slightly just above 18 m (e.g. up to 1.0060 for `hU = 22.5`), where the limit gives 1
  - `38.901_InH_Mixed` (mixed office): 1 for `d <= 1.2`, `exp(-(d - 1.2)/4.7)` for `1.2 < d < 6.5`,
    `exp(-(d - 6.5)/32.6)·0.32` for `d >= 6.5`
  - `38.901_InH_Open` (open office): 1 for `d <= 5`, `exp(-(d - 5)/70.8)` for `5 < d <= 49`,
    `exp(-(d - 49)/211.7)·0.54` for `d > 49`
  - `38.901_InF_SL`, `38.901_InF_DL`: `exp(-d·kinv)` with `kinv = -ln(1 - r)/d_clutter`
  - `38.901_InF_SH`, `38.901_InF_DH`: with `kinv = -ln(1 - r)/d_clutter`: 1 for `hU >= h_c` (UT
    above the clutter); `exp(-d·kinv·(h_c - hU)/(hB - hU))` for `hB > hU` (TR:
    `k = -d_clutter/ln(1 - r)·(hB - hU)/(h_c - hU)`); `exp(-d·kinv)` (the rule of SL and DL) for
    `hB <= hU < h_c`, where the TR formula has a BS above the UT and would give `k <= 0`. `r = 0`
    gives 1
  - `38.901_InF_HH`: 1
  - `38.901_SMa` (TR v19): 1 for `d <= 10` or `hB <= hU`, else `exp(-d·(k_com + k_res + k_veg))`
    with `k_x = -ln(1 - r_x)/d_clutter·max(h_x - hU, 0)/(hB - hU)`, heights `h_com = 20`,
    `h_res = 8`, `h_veg = 15` m, densities `r_com = 0.02`, `r_res = 0.18`, `r_veg = r_clutter`; the
    TR writes `max(., 0)` only for the residential term, which gives the same values for `hU < 15` m
    (the TR range)
- Defaults of the clutter arguments (NaN or empty), TR 38.901 Table 7.2-4 and Table 7.8-7 for InF,
  Table 7.4.2-1 for SMa:
  - `InF_SL`: `r = 0.2`, `d_clutter = 10` m; `InF_DL`: `r = 0.6`, `d_clutter = 2` m
  - `InF_SH`: `r = 0.2`, `d_clutter = 10` m, `h_c = 2` m; `InF_DH`: `r = 0.6`, `d_clutter = 2` m,
    `h_c = 6` m
  - `SMa`: `d_clutter = 30` m, `r_veg = 0` (no vegetation; 0.1 sparse, 0.2 dense vegetation);
    `h_clutter` is not used
  - The other models ignore the clutter arguments (they are validated for every model)
- TR 38.811 (via QuaDRiGa), `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`:
  the LOS probability in percent over the elevation `atan2(hB, d)` in degrees (absolute BS height),
  linear interpolation over the grid -90, 0, 10, ..., 90 deg, times 0.01:
  - DenseUrban: 0, 0, 28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1
  - Urban: 0, 0, 24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2
  - Suburban and Rural: 0, 0, 78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8
- QuaDRiGa, `QuaDRiGa_Industrial`: `min(1, exp(-(d + 20)/80)·exp(min(hB, 8)/5 - 1))`
- `LOS`: 1, `NLOS`: 0
- Soft LOS state (TR 38.901 eq. 7.6-18), computed when `rnd_g` and `fc` are both given and
  `los_soft` is requested: `los_soft(l, k) = 1/2 + atan(sqrt(20/lambda_k)·(rnd_g(l) +
  F(p_los(l))))/pi`, `lambda_k = c/fc(k)` with `c = 299792458` m/s and
  `F(p) = sqrt(2)·erfinv(2p - 1)` the inverse of the standard normal CDF (accurate to
  `1e-10·max(1, abs(F))` for `p` in [1e-12, 1 - 1e-12]). `p = 1` gives 1, `p = 0` gives 0.
  `los_soft > 0.5` exactly when `rnd_g > -F(p)`, which has the probability `p` for `rnd_g` from
  N(0,1)
- The LOS state itself (a draw against `p_los`) and its spatial correlation are the caller's: e.g. a
  spatially consistent uniform value from [[sos_rand]], or `rnd_g` from [[sos_randn]], at the
  decorrelation distance of the scenario field `LOS_state_lambda` (TR 38.901 Table 7.6.3.1-2). The
  `LOS_state_lambda` of the InF conf files is `d_clutter / 2` of the TR defaults; it does not follow
  a `d_clutter` argument, so a caller who changes `d_clutter` passes its own correlation distance
- O2I UTs: Table 7.4.2-1 uses the outdoor distance `d2d_out = d2D - d2D-in` with `d2D-in` from
  [[o2i_loss]], which therefore runs first; the caller clamps `d2d_out` at 0 (negative distances are
  rejected). Indoor scenarios (InH, InF) pass the 2D distance
- Validation, in this order: the model name; `d2d_out` finite and >= 0; `h_bs` and `h_ut` with 1 or
  `n_link` elements and finite; `d_clutter` NaN or finite and > 0, `r_clutter` NaN or in [0, 1),
  `h_clutter` NaN or finite and >= 0 (for every model); if `los_soft` is computed: `rnd_g` with
  `n_link` finite elements, `fc` finite and > 0. Without `los_soft`, `rnd_g` and `fc` are ignored

## Usage:
```
[ p_los, los_soft ] = quadriga_lib.los_probability( model, d2d_out, h_bs, h_ut, ...
    d_clutter, r_clutter, h_clutter, rnd_g, fc );
```

## Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`,
  `38.901_InH_Open`, `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`,
  `38.901_InF_HH`, `38.901_SMa`, `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`,
  `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`; string
- **`d2d_out`** — 2D outdoor distance in [m] (indoor scenarios: 2D distance), finite and >= 0;
  `[n_link]`
- **`h_bs`** — BS height in [m]; one element is used for all links; `[n_link]` or `[1]`
- **`h_ut`** — UT height in [m]; one element is used for all links; `[n_link]` or `[1]`
- **`d_clutter`** — Clutter size in [m] of the InF and SMa models; default: NaN = TR default of the
  model
- **`r_clutter`** — Clutter density of the InF models, vegetation density `r_veg` of SMa, in [0, 1);
  default: NaN = TR default of the model
- **`h_clutter`** — Clutter height in [m] of InF-SH and InF-DH; default: NaN = TR default of the
  model
- **`rnd_g`** *(optional)* — Normal(0,1) value per link for the soft LOS state (G of eq. 7.6-18);
  `[n_link]`
- **`fc`** *(optional)* — Carrier frequencies in [Hz] for the soft LOS state; `[n_freq]`

## Outputs:
- **`p_los`** — LOS probability in [0, 1]; `[n_link, 1]`
- **`los_soft`** — Soft LOS state in [0, 1]; `[n_link, n_freq]`; `[]` if `rnd_g` or `fc` is omitted
  or empty

## See also:
- [[o2i_loss]] (indoor distance for `d2d_out`)
- [[sos_rand]], [[sos_randn]] (spatially consistent LOS-state draw and `rnd_g`)
- [[conf_file_read]] (field `LOS_state_lambda`)
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
    const arma::vec fc = (nrhs < 9) ? arma::vec() : qd_mex_get_Col<double>(prhs[8]);

    // Output allocation; the soft LOS state needs both 'rnd_g' and 'fc', else it is returned as []
    const bool soft = nlhs > 1 && !rnd_g.empty() && !fc.empty();
    arma::vec p_los;
    arma::mat los_soft;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&p_los, d2d_out.n_elem);
    if (soft)
        plhs[1] = qd_mex_init_output(&los_soft, d2d_out.n_elem, fc.n_elem);
    else if (nlhs > 1)
        plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);

    // Wrap optional pointers
    const arma::vec *p_rnd_g = rnd_g.empty() ? nullptr : &rnd_g;
    const arma::vec *p_fc = fc.empty() ? nullptr : &fc;
    arma::mat *p_los_soft = soft ? &los_soft : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::los_probability(model, d2d_out, h_bs, h_ut, p_los, d_clutter, r_clutter, h_clutter,
                                          p_rnd_g, p_fc, p_los_soft));
}
