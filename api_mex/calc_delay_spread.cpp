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
# CALC_DELAY_SPREAD
Calculate the RMS delay spread of each CIR from its path delays and powers

- With the path powers $p_l$ normalized to sum 1 over the paths of a CIR, the mean delay and the RMS delay spread are

  $$\bar\tau = \sum_l p_l \, \tau_l, \qquad \sigma_\tau = \sqrt{\sum_l p_l \, (\tau_l - \bar\tau)^2}$$

- Paths more than `threshold_dB` below the strongest path of their CIR are left out
- With `granularity` > 0, the delays are first rounded to the nearest multiple of `granularity`, and the powers of
  the paths with the same rounded delay are summed
- A CIR without power gives a spread and a mean delay of 0

## Usage:
```
[ ds, mean_delay ] = quadriga_lib.calc_delay_spread( delays, powers, threshold_dB, granularity );
```

## Inputs:
- **`delays`** — Path delays in s, one column per CIR; `[n_path, n_cir]`
- **`powers`** — Path powers, linear; zero-power entries pad CIRs with fewer paths; `[n_path, n_cir]`
- **`threshold_dB`** — Power range in dB below the strongest path that is kept, > 0; default: 100
- **`granularity`** — Spacing of the delay grid in s, finite and >= 0; 0 = no grid; with a grid, all
  delays must be finite and their span must be below $2^{53}$ grid steps; default: 0

## Outputs:
- **`ds`** — RMS delay spread $\sigma_\tau$ in s; `[n_cir, 1]`
- **`mean_delay`** — Mean delay $\bar\tau$ in s; `[n_cir, 1]`

## See also:
- [[quantize_delays]] (delays on a tap grid with interpolation)
- [[calc_angular_spread]] (angular spreads)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::vector<arma::vec> delays = qd_mex_matlab2vector_Col<double>(prhs[0], 1, "delays");
    const std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[1], 1, "powers");
    const double threshold_dB = (nrhs < 3) ? 100.0 : qd_mex_get_scalar<double>(prhs[2], "threshold_dB", 100.0);
    const double granularity = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "granularity", 0.0);

    arma::vec ds;
    arma::vec mean_delay;

    arma::vec *p_mean_delay = (nlhs > 1) ? &mean_delay : nullptr;

    CALL_QD(ds = quadriga_lib::calc_delay_spread<double>(delays, powers, threshold_dB, granularity, p_mean_delay));

    plhs[0] = qd_mex_copy2matlab(&ds);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&mean_delay);
}
