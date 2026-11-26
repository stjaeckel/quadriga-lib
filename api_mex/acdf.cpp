// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_math.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# ACDF
Empirical cumulative distribution functions (CDF) of several data sets, their average and quantile statistics

- The CDF of each data set (column of `data`) at the bin centers: each finite value counts for the nearest bin (values
  outside the bins for the first or last one), and the CDF is the cumulative count divided by the number of finite
  values of the set
- Average `cdf_avg` in quantile space: for the probabilities $P = 0, 0.001, \dots, 0.999$, the bins at which the
  CDFs of the sets reach $P$ are averaged over the sets, and the averaged quantile function is mapped back to the
  bins; this averages the position of the distributions instead of their CDF values
- `mu` and `sig`: mean and standard deviation (normalized by $N - 1$) of the quantiles at
  $P = 0.1, 0.2, \dots, 0.9$ over the sets, each taken over the 11 probabilities within $\pm 0.005$; for one set its
  quantiles and 0
- Non-finite values are left out; a set without finite values gets a NaN CDF and is left out of `cdf_avg`, `mu` and
  `sig`, which are NaN if no set has finite values (possible only with given bins)
- Generated bins: `n_bins` equally spaced bins from the smallest to the largest finite value of `data` (±10 % of the
  value, or ±1 for 0, if all are equal); `data` without finite values then throws

## Usage:
```
[ cdf_per_set, bins_out, cdf_avg, mu, sig ] = quadriga_lib.acdf( data, bins_in, n_bins );
```

## Inputs:
- **`data`** — Data sets, one per column, magnitudes below about 1e18; `[n_samples, n_sets]`
- **`bins_in`** — Bin centers, finite and strictly ascending; `[n_bins_in, 1]`; default: `[]` (generated bins)
- **`n_bins`** — Number of generated bins, >= 2; not used with `bins_in`; default: 201

## Outputs:
- **`cdf_per_set`** — CDF of each set; `[n_bins_out, n_sets]`
- **`bins_out`** — Bin centers, `bins_in` or the generated ones; `[n_bins_out, 1]`
- **`cdf_avg`** — Average CDF; `[n_bins_out, 1]`
- **`mu`** — Mean of the quantiles at 0.1, ..., 0.9; `[9, 1]`
- **`sig`** — Standard deviation of these quantiles; `[9, 1]`

## See also:
- [[calc_delay_spread]] (delay spreads of many CIRs, a typical input)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const arma::mat data = qd_mex_get_Mat<double>(prhs[0], false, "data");
    const arma::vec bins_in = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1], false, "bins_in");
    const arma::uword n_bins = (nrhs < 3) ? 201 : qd_mex_get_scalar<arma::uword>(prhs[2], "n_bins", 201);

    arma::uword n_bins_out = bins_in.empty() ? n_bins : bins_in.n_elem;
    arma::uword n_sets = data.n_cols;

    arma::mat cdf_per_set;
    arma::vec cdf_avg, bins_out, mu, sig;

    plhs[0] = qd_mex_init_output(&cdf_per_set, n_bins_out, n_sets);

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&cdf_avg, n_bins_out);

    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&mu, 9);

    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&sig, 9);

    if (!bins_in.empty())
        bins_out = bins_in;

    CALL_QD(quadriga_lib::acdf<double>(data, &bins_out, &cdf_per_set,
                                       (nlhs > 2) ? &cdf_avg : nullptr,
                                       (nlhs > 3) ? &mu : nullptr,
                                       (nlhs > 4) ? &sig : nullptr, n_bins));

    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&bins_out);
}
