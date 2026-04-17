// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# ACDF
Calculate the empirical averaged cumulative distribution function (CDF)

## Description:
- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative
  sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from
  each column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- If `bins` is empty, equally spaced bins spanning the data range are generated

## Usage:
```
[ Sh, bins_out, Sc, mu, sig ] = quadriga_lib.acdf( data, bins_in, n_bins );
```

## Input Arguments:
- **`data`** — Input data matrix; each column is one independent data set, `[n_samples, n_sets]`
- **`bins_in`** *(optional)* — Bin centers; used as-is if non-empty, `[n_bins_in]`
- **`n_bins`** *(optional)* — Number of bins when auto-generating; must be >= 2; ignored when
  non-empty `bins_in` are provided

## Output Arguments:
- **`Sh`** *(optional)* — Individual CDFs, one per column of data, `[n_bins_out, n_sets]`
- **`bins_out`** *(optional)* — Auto-generated bins; copy of `bins_in` when
  non-empty `bins_in` are provided, `[n_bins_out = n_bins]` or `[n_bins_out = n_bins_in]`
- **`Sc`** *(optional)* — Averaged CDF via quantile-space averaging across data sets, `[n_bins]`
- **`mu`** *(optional)* — Mean of the 0.1–0.9 quantiles across data sets, `[9]`
- **`sig`** *(optional)* — Standard deviation of the 0.1–0.9 quantiles across data sets, `[9]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const arma::mat data = qd_mex_get_Mat<double>(prhs[0]);
    const arma::vec bins_in = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1]);
    const arma::uword n_bins = (nrhs < 3) ? 201 : qd_mex_get_scalar<arma::uword>(prhs[2], "n_bins", 201);

    arma::uword n_bins_out = bins_in.empty() ? n_bins : bins_in.n_elem;
    arma::uword n_sets = data.n_cols;

    // Output allocation
    arma::mat Sh;
    arma::vec Sc, bins_out, mu, sig;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&Sh, n_bins_out, n_sets);

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&Sc, n_bins_out);

    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&mu, 9);

    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&sig, 9);

    // Special case for bins
    if (!bins_in.empty())
        bins_out = bins_in;

    // Wrap optional pointers
    arma::mat *p_Sh = Sh.empty() ? nullptr : &Sh;
    arma::vec *p_Sc = Sc.empty() ? nullptr : &Sc;
    arma::vec *p_mu = mu.empty() ? nullptr : &mu;
    arma::vec *p_sig = sig.empty() ? nullptr : &sig;

    // Call library function
    CALL_QD(quadriga_lib::acdf<double>(data, &bins_out, p_Sh, p_Sc, p_mu, p_sig, n_bins));

    // Copy to MATLAB
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&bins_out);
}
