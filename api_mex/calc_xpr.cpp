// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_tools.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# CALC_XPR
Calculate the cross-polarization ratio (XPR) of each CIR in the linear and the circular polarization basis

- Total-power ratio: the co- and cross-polar powers of the paths of a CIR, weighted by the path powers $p_l$, are
  summed, and the XPR is their ratio; LOS paths are left out unless `include_los` is set
- Entries of the polarization transfer matrix $M$ of a path, first index receive, second transmit polarization:
  rows 1-2 of `polmat` = $M_{vv}$, 3-4 = $M_{hv}$ (H rx, V tx), 5-6 = $M_{vh}$ (V rx, H tx), 7-8 = $M_{hh}$, each as
  real and imaginary part
- Circular basis: $M_{rt} = \mathbf{r}^T M \, \mathbf{t}$ (plain transpose), as [[get_channels_spherical]]
  couples antenna patterns, with the same vectors at both ends, each in the $(e_\theta, e_\phi)$ frame of its antenna:
  $\mathbf{L} = (1, +j) / \sqrt{2}$ (LHCP) and $\mathbf{R} = (1, -j) / \sqrt{2}$ (RHCP) in (V, H), which gives

  $$M_{LL}, M_{RR} = \frac{1}{2} \left( M_{vv} - M_{hh} \pm j (M_{vh} + M_{hv}) \right)$$

  $$M_{LR}, M_{RL} = \frac{1}{2} \left( M_{vv} + M_{hh} \pm j (M_{hv} - M_{vh}) \right)$$

  (upper sign for the first entry)

- The names L and R follow the IEEE definition (rotation seen along the propagation direction) for the time
  convention $e^{+j 2\pi f t}$, which the coefficients $e^{-j 2\pi f \tau}$ of the library imply, with
  $e_\theta \times e_\phi$ the outward direction of the pattern
- Co-polar means the same circular vector at both ends: the LOS matrix $\mathrm{diag}(1, -1)$ of
  [[get_channels_spherical]] is co-polar (handedness kept), $\mathrm{diag}(1, 1)$ (one mirrored component, as
  by a single reflection) is cross-polar

## Usage:
```
[ xpr, pg ] = quadriga_lib.calc_xpr( powers, polmat, path_length, tx_pos, rx_pos, include_los, ...
    window_size );
```

## Inputs:
- **`powers`** — Path powers $p_l$, linear; `[n_path, n_cir]`
- **`polmat`** — Polarization transfer matrices, magnitudes below about 1e18; `[8, n_path, n_cir]`
- **`path_length`** — Path lengths in m; `[n_path, n_cir]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, 1]` (fixed) or `[3, n_cir]`
- **`include_los`** — Include the LOS paths in the XPR; default: false
- **`window_size`** — LOS window in m, >= 0: paths with `path_length` <= $d_\mathrm{TR}$ +
  `window_size` are LOS, $d_\mathrm{TR}$ the Tx-Rx distance; default: 0.01

## Outputs:
- **`xpr`** — XPR, linear; `[n_cir, 6]`; columns:

  | Column | XPR |
  | ------ | --- |
  | 1 | Linear: $(P_{vv} + P_{hh}) / (P_{vh} + P_{hv})$ with $P_{rt} = \sum_l p_l \lvert M_{rt} \rvert^2$ |
  | 2 | V (receive V): $P_{vv} / P_{vh}$ |
  | 3 | H (receive H): $P_{hh} / P_{hv}$ |
  | 4 | Circular: $(P_{LL} + P_{RR}) / (P_{LR} + P_{RL})$ |
  | 5 | LHCP (receive L): $P_{LL} / P_{LR}$ |
  | 6 | RHCP (receive R): $P_{RR} / P_{RL}$ |

- **`pg`** — Path gain of all paths, LOS included,
  $\frac{1}{2} \sum_l p_l \sum_{r,t \in \{v,h\}} \lvert M_{rt} \rvert^2$; `[n_cir, 1]`

## Notes:
- A ratio with zero cross-polar power is infinite, or 0 if the co-polar power is zero too (also for a CIR without
  paths, which has `pg` = 0)
- A scaling of `polmat` does not change the XPR, but `pg`

## See also:
- [[calc_rician_k_factor]] (LOS and NLOS power with the same LOS window)
- [[get_channels_spherical]] (polarization transfer matrices of the paths)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 5 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[0], 1, "powers");
    const std::vector<arma::mat> polmat = qd_mex_matlab2vector_Mat<double>(prhs[1], 2, "polmat");
    const std::vector<arma::vec> path_length = qd_mex_matlab2vector_Col<double>(prhs[2], 1, "path_length");
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[3], false, "tx_pos");
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[4], false, "rx_pos");

    const bool include_los = (nrhs < 6) ? false : qd_mex_get_scalar<bool>(prhs[5], "include_los", false);
    const double window_size = (nrhs < 7) ? 0.01 : qd_mex_get_scalar<double>(prhs[6], "window_size", 0.01);

    arma::mat xpr;
    arma::vec pg;
    arma::vec *p_pg = (nlhs > 1) ? &pg : nullptr;

    CALL_QD(quadriga_lib::calc_xpr<double>(
        powers, polmat, path_length, tx_pos, rx_pos, &xpr, p_pg, include_los, window_size));

    plhs[0] = qd_mex_copy2matlab(&xpr);

    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&pg);
}
