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
# CALC_RICIAN_K_FACTOR
Calculate the Rician K-Factor from channel impulse response data

- KF = LOS power / NLOS power; LOS paths are those with length ≤ `dTR + window_size`, where
  `dTR` is the direct TX-RX distance
- If total NLOS power is zero, KF is set to infinity; if total LOS power is zero, KF is
  set to 0
- TX/RX positions may be fixed `[3, 1]` (reused for all snapshots) or mobile `[3, n_cir]`

## Usage:
```
[ kf, pg ] = quadriga_lib.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size );
```

## Inputs:
- **`powers`** — Path powers in [W]; `[n_path, n_cir]`
- **`path_length`** — Absolute TX-to-RX path lengths; `[n_path, n_cir]`
- **`tx_pos`** — Transmitter position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `[3, 1]` (fixed) or `[3, n_cir]` (mobile)
- **`window_size`** — LOS window; paths with length ≤ `dTR + window_size` are treated as LOS; default: 0.01

## Outputs:
- **`kf`** — Rician K-Factor on linear scale; `[n_cir]`
- **`pg`** — Total path gain (sum of all path powers) in [W]; `[n_cir]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[0], 1);
    const std::vector<arma::vec> path_length = qd_mex_matlab2vector_Col<double>(prhs[1], 1);
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[2]);
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[3]);
    const double window_size = (nrhs < 5) ? 0.01 : qd_mex_get_scalar<double>(prhs[4], "window_size", 0.01);

    // Declare outputs
    arma::vec kf, pg;

    // Set up optional output pointers based on nlhs
    arma::vec *p_kf = (nlhs > 0) ? &kf : nullptr;
    arma::vec *p_pg = (nlhs > 1) ? &pg : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::calc_rician_k_factor<double>(powers, path_length, tx_pos, rx_pos, p_kf, p_pg, window_size));

    // Write outputs to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&kf);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&pg);
}
