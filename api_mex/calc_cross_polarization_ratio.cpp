// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# CALC_CROSS_POLARIZATION_RATIO
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method: co-pol 
  and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis via the Jones matrix transform `M_circ = T * M_lin * T^-1`
- LOS paths are identified by comparing path length against the direct TX-RX distance `dTR`; paths with 
  `path_length < dTR + window_size` are excluded by default
- Polarization transfer matrix `M` is stored column-major with interleaved real/imaginary parts, 8 rows per path:
  `[Re(M_vv); Im(M_vv); Re(M_vh); Im(M_vh); Re(M_hv); Im(M_hv); Re(M_hh); Im(M_hh)]`
- Normalization of `M` does not affect XPR (cancels in the ratio) but does affect `pg`
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR is set to 0
- TX/RX positions may be fixed `[3, 1]` or mobile `[3, n_cir]`

## Usage:
```
[ xpr, pg ] = quadriga_lib.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, include_los, window_size );
```

## Inputs:
- **`powers`** — Path powers in [W]; `[n_path, n_cir]`
- **`M`** — Polarization transfer matrices with interleaved real/imag parts; `[8, n_path, n_cir]`
- **`path_length`** — Absolute TX-to-RX path lengths; `[n_path, n_cir]`
- **`tx_pos`** — Transmitter position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`include_los`** — If true, includes LOS and near-LOS paths in the XPR calculation; default: false
- **`window_size`** — LOS exclusion window; paths within `dTR + window_size` are excluded when `include_los = false`; default: 0.01

## Outputs:
- **`xpr`** — XPR on linear scale; `[n_cir, 6]`; columns:<br><br>
   | Col | Description                                                     |
   | :-: | --------------------------------------------------------------- |
   | 1   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 2   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_hv)^2)                      |
   | 3   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_vh)^2)                      |
   | 4   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 5   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_RL)^2)                   |
   | 6   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_LR)^2)                   |
- **`pg`** — Total path gain summed over all paths (including LOS) as 
  `0.5 * sum(powers * (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `[n_cir]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 5 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[0], 1);
    const std::vector<arma::mat> M = qd_mex_matlab2vector_Mat<double>(prhs[1], 2);
    const std::vector<arma::vec> path_length = qd_mex_matlab2vector_Col<double>(prhs[2], 1);
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[3]);
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[4]);

    const bool include_los = (nrhs < 6) ? false : qd_mex_get_scalar<bool>(prhs[5], "include_los", false);
    const double window_size = (nrhs < 7) ? 0.01 : qd_mex_get_scalar<double>(prhs[6], "window_size", 0.01);

    // Declare outputs
    arma::mat xpr;
    arma::vec pg;

    arma::mat *p_xpr = (nlhs > 0) ? &xpr : nullptr;
    arma::vec *p_pg = (nlhs > 1) ? &pg : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::calc_cross_polarization_ratio<double>(
        powers, M, path_length, tx_pos, rx_pos, p_xpr, p_pg, include_los, window_size));

    // Write outputs
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&xpr);

    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&pg);
}
