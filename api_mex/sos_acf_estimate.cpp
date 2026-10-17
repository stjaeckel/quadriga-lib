// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

#include <cmath>
#include <string>

// Count argument: read as double, so that a negative, non-finite or non-integer value is rejected with the name of
// the argument instead of wrapping around in the conversion to an unsigned type.
static arma::uword qd_mex_sos_count(const mxArray *input, const char *name, double default_value)
{
    const double v = qd_mex_get_scalar<double>(input, name, default_value);
    if (!(v >= 1.0) || !(v < 9.0e18) || v != std::floor(v))
    {
        const std::string msg = std::string("Input '") + name + "' must be an integer >= 1.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", msg.c_str());
    }
    return (arma::uword)v;
}

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# SOS_ACF_ESTIMATE
Monte-Carlo estimate of the ACF of one generator

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at
  random positions
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin,
  evaluates the generator there, and for each distance bin computes the correlation coefficient
  between the two values of all position pairs whose distance falls into the bin
- Positions: a matrix `[2·n_pos, 3]` is filled column by column (all x, then all y, then all z)
  with `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in
  [[sos_init_phases]]
- Bin `n` covers the distances from the midpoint between `bins(n-1)` and `bins(n)` (excluded) to
  the midpoint between `bins(n)` and `bins(n+1)` (included). The first bin starts at 1e-5 m, the
  last bin ends half a mean bin spacing above `bins(end)`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices on the same map. The first `n_pos` positions are the
  first device, the second `n_pos` positions the second device, and the value of a link is
  [[sos_val]] with both positions. Two links form a pair if the distance between their first
  devices falls into the bin and the distance between their second devices does not exceed the
  upper limit of the bin; requires `acf` and `dist`
- The estimate describes one realisation; it scatters around the ACF of the sinusoids
- Run time grows with `n_pos^2`; the result does not depend on the number of threads

## Usage:
```
[ acf_est, dist_est, acf_est_dual ] = quadriga_lib.sos_acf_estimate( freq, phase, bins, n_pos, ...
    acf, dist, seed );
```

## Inputs:
- **`freq`** — Frequencies; `[n_coeff, 3]`
- **`phase`** — Phases of the generator; `[n_coeff]`
- **`bins`** — Bin centers; must be >= 0, increase strictly and be finite; at least 2
  elements; `[n_bin]`
- **`n_pos`** — Number of random positions per device; must be > 1
- **`acf`** *(optional)* — Target ACF; required for `acf_est_dual`, must then be > -1;
  `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`seed`** — Seed for the positions; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

## Outputs:
- **`acf_est`** — Estimated ACF; column vector; `[n_bin]`
- **`dist_est`** — Mean distance of the pairs in each bin; column vector; `[n_bin]`
- **`acf_est_dual`** — Estimated ACF for two devices on the same map; column vector; `[n_bin]`

## See also:
- [[sos_acf_approx]] (ACF of the sinusoids in closed form)
- [[sos_val]] (evaluation of the generator)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 4 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::mat freq = qd_mex_get_Mat<double>(prhs[0]);
    const arma::vec phase = qd_mex_get_Col<double>(prhs[1]);
    const arma::vec bins = qd_mex_get_Col<double>(prhs[2]);
    const arma::uword n_pos = qd_mex_sos_count(prhs[3], "n_pos", 0.0);
    const arma::vec acf = (nrhs < 5) ? arma::vec() : qd_mex_get_Col<double>(prhs[4]);
    const arma::vec dist = (nrhs < 6) ? arma::vec() : qd_mex_get_Col<double>(prhs[5]);
    const arma::sword seed = (nrhs < 7) ? -1 : qd_mex_get_seed(prhs[6]);

    // Output allocation
    arma::vec acf_est, dist_est, acf_est_dual;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&acf_est, bins.n_elem);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&dist_est, bins.n_elem);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&acf_est_dual, bins.n_elem);

    // Wrap optional pointers
    arma::vec *p_dist_est = dist_est.empty() ? nullptr : &dist_est;
    arma::vec *p_acf_est_dual = acf_est_dual.empty() ? nullptr : &acf_est_dual;
    const arma::vec *p_acf = acf.empty() ? nullptr : &acf;
    const arma::vec *p_dist = dist.empty() ? nullptr : &dist;

    // Call library function
    CALL_QD(quadriga_lib::sos_acf_estimate<double>(freq, phase, bins, n_pos, acf_est, p_dist_est, p_acf_est_dual, p_acf, p_dist, seed));
}
