// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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
# acdf
Calculate the empirical averaged cumulative distribution function (CDF)

## Description:
- Calculates the empirical CDF from the given data matrix, where each column represents an
  independent data set (e.g., repeated experiment runs).
- Individual CDFs are computed per column and an averaged CDF is obtained by interpolation in
  quantile space.
- `Inf` and `NaN` values in the data are excluded from the computation.
- If `bins` is empty or not provided, 201 equally spaced bins spanning the data range are generated.

## Usage:
```
[ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data );
[ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data, bins );
[ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data, bins, n_bins );
```

## Input arguments:
- `**data**` (input)<br>
  Input data matrix. Size `[n_samples, n_sets]`. Each column is one data set.

- `**bins** = []` (optional input)<br>
  Bin centers for the histogram. Length `[n_bins]`. If empty, bins are auto-generated.

- `**n_bins** = 201` (optional input)<br>
  Number of bins to generate when bins are auto-generated. Must be at least 2. Ignored when
  non-empty bins are provided.

## Output arguments:
- `double **Sh**` (output)<br>
  Individual CDFs. Size `[n_bins, n_sets]`.

- `double **bins**` (output)<br>
  Bin centers. Length `[n_bins]`.

- `double **Sc**` (output)<br>
  Averaged CDF. Length `[n_bins]`.

- `double **mu**` (output)<br>
  Mean of the 0.1, 0.2, ..., 0.9 quantiles across data sets. Length `[9]`.

- `double **sig**` (output)<br>
  Standard deviation of the 0.1, 0.2, ..., 0.9 quantiles across data sets. Length `[9]`.

## Example:
```
data = randn(10000, 5);
[ Sh, bins, Sc, mu, sig ] = quadriga_lib.acdf( data );
% bins has 201 elements, Sh is [201, 5], Sc is [201, 1], mu and sig are [9, 1]
```
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    arma::mat data = qd_mex_get_double_Mat(prhs[0]);

    // Read optional bins
    arma::vec bins;
    if (nrhs >= 2 && !mxIsEmpty(prhs[1]))
        bins = qd_mex_get_double_Col(prhs[1]);

    // Read optional n_bins
    arma::uword n_bins = (nrhs < 3) ? 201 : qd_mex_get_scalar<arma::uword>(prhs[2], "n_bins", 201);

    // Set up output pointers based on nlhs
    arma::mat Sh;
    arma::vec Sc, mu, sig;

    arma::mat *p_Sh = (nlhs > 0) ? &Sh : nullptr;
    arma::vec *p_bins = (nlhs > 1 || bins.n_elem == 0) ? &bins : nullptr;
    arma::vec *p_Sc = (nlhs > 2) ? &Sc : nullptr;
    arma::vec *p_mu = (nlhs > 3) ? &mu : nullptr;
    arma::vec *p_sig = (nlhs > 4) ? &sig : nullptr;

    // Call library function (double precision)
    CALL_QD(quadriga_lib::acdf<double>(data, p_bins, p_Sh, p_Sc, p_mu, p_sig, n_bins));

    // Write outputs
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&Sh);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&bins);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&Sc);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&mu);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&sig);
}
