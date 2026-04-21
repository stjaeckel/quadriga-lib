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
# CALC_DELAY_SPREAD
Calculates RMS delay spread from per-CIR delays and linear-scale powers

- Paths with power below `p_max / 10^(0.1 * threshold)` are excluded; the default threshold
  of 100 dB effectively includes all paths
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have
  their powers summed before computing the spread; function recurses on the binned profile

## Usage:
```
[ ds, mean_delay ] = quadriga_lib.calc_delay_spread( delays, powers, threshold, granularity );
```

## Inputs:
- **`delays`** — Delays in [s] per CIR; `[n_path, n_cir]`
- **`powers`** — Path powers on linear scale in [W]; `[n_path, n_cir]`
- **`threshold`** *(optional)* — Power threshold in [dB] relative to strongest path; paths
  below threshold are excluded; default: 100
- **`granularity`** *(optional)* — Bin width in [s] for grouping paths in the delay domain;
  default: 0 (no grouping)

## Outputs:
- **`ds`** — RMS delay spread in [s] per CIR; `[n_cir]`
- **`mean_delay`** *(optional)* — Mean delay in [s] per CIR; `[n_cir]`

## See also:
- [[quantize_delays]] (for mapping delays to a fixed tap grid)
- [[calc_rician_k_factor]] (for calculating K-factor)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::vector<arma::vec> delays = qd_mex_matlab2vector_Col<double>(prhs[0], 1);
    const std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[1], 1);
    const double threshold = (nrhs < 3) ? 100.0 : qd_mex_get_scalar<double>(prhs[2], "threshold", 100.0);
    const double granularity = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "granularity", 0.0);

    // Declare outputs
    arma::vec ds;
    arma::vec mean_delay;

    // Set up optional output pointer
    arma::vec *p_mean_delay = (nlhs > 1) ? &mean_delay : nullptr;

    // Call library function
    CALL_QD(ds = quadriga_lib::calc_delay_spread<double>(delays, powers, threshold, granularity, p_mean_delay));

    // Write outputs
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&ds);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&mean_delay);
}
