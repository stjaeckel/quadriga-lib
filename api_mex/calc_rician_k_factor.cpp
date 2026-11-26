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
# CALC_RICIAN_K_FACTOR
Calculate the Rician K-factor of each CIR, the ratio of the LOS to the NLOS power

- $K = P_\mathrm{LOS} / P_\mathrm{NLOS}$ with the sums of the path powers; LOS paths are those with a length
  `path_length` <= $d_\mathrm{TR}$ + `window_size`, $d_\mathrm{TR}$ the Tx-Rx distance, all others are NLOS
- Without NLOS power, $K$ is infinite, or 0 if the LOS power is zero too (also for a CIR without paths)

## Usage:
```
[ kf, pg ] = quadriga_lib.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size );
```

## Inputs:
- **`powers`** — Path powers, linear, one column per CIR; zero-power entries pad CIRs with fewer paths;
  `[n_path, n_cir]`
- **`path_length`** — Path lengths in m; `[n_path, n_cir]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, 1]` (fixed) or `[3, n_cir]`
- **`window_size`** — LOS window in m, >= 0; default: 0.01

## Outputs:
- **`kf`** — K-factor, linear; `[n_cir, 1]`
- **`pg`** — Sum of all path powers, linear; `[n_cir, 1]`

## See also:
- [[calc_xpr]] (XPR with the same LOS window)
- [[calc_delay_spread]] (delay spread of the same paths)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[0], 1, "powers");
    const std::vector<arma::vec> path_length = qd_mex_matlab2vector_Col<double>(prhs[1], 1, "path_length");
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[2], false, "tx_pos");
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[3], false, "rx_pos");
    const double window_size = (nrhs < 5) ? 0.01 : qd_mex_get_scalar<double>(prhs[4], "window_size", 0.01);

    arma::vec kf, pg;

    arma::vec *p_pg = (nlhs > 1) ? &pg : nullptr;

    CALL_QD(quadriga_lib::calc_rician_k_factor<double>(powers, path_length, tx_pos, rx_pos, &kf, p_pg, window_size));

    plhs[0] = qd_mex_copy2matlab(&kf);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&pg);
}
