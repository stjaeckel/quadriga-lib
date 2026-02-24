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
# calc_delay_spread
Calculate the RMS delay spread in [s]

## Description:
- Computes the root-mean-square (RMS) delay spread from a given set of delays and corresponding
  linear-scale powers for each channel impulse response (CIR).
- An optional power threshold in [dB] relative to the strongest path can be applied. Paths with
  power below `p_max(dB) - threshold` are excluded.
- An optional granularity parameter in [s] groups paths in the delay domain.
- Optionally returns the mean delay for each CIR.

## Usage:
```
ds = quadriga_lib.calc_delay_spread( delays, powers );
ds = quadriga_lib.calc_delay_spread( delays, powers, threshold );
ds = quadriga_lib.calc_delay_spread( delays, powers, threshold, granularity );
[ ds, mean_delay ] = quadriga_lib.calc_delay_spread( delays, powers, threshold, granularity );
```

## Arguments:
- `**delays**` (input)<br>
  Delays in [s]. A 2D matrix of size `[n_cir, n_path]`. Each row is one CIR. Rows may be
  zero-padded if CIRs have different numbers of paths.

- `**powers**` (input)<br>
  Path powers on a linear scale [W]. Same size as `delays`.

- `**threshold** = 100.0` (input)<br>
  Power threshold in [dB] relative to the strongest path. Default: 100 dB.

- `**granularity** = 0.0` (input)<br>
  Window size in [s] for grouping paths in the delay domain. Default: 0.

## Returns:
- `double **ds**` (output)<br>
  RMS delay spread in [s] for each CIR. Size `[n_cir, 1]`.

- `double **mean_delay**` (optional output)<br>
  Mean delay in [s] for each CIR. Size `[n_cir, 1]`.

## Example:
```
delays = [0, 1e-6, 2e-6];
powers = [1.0, 0.5, 0.25];
[ds, mean_delay] = quadriga_lib.calc_delay_spread( delays, powers );
```
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // --- Validate argument counts ---
    if (nrhs < 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // --- Read inputs ---

    // delays: [n_cir, n_path] matrix -> split rows into vector of column vectors
    arma::mat delays_mat = qd_mex_get_double_Mat(prhs[0]);
    arma::mat powers_mat = qd_mex_get_double_Mat(prhs[1]);

    arma::uword n_cir = delays_mat.n_rows;
    arma::uword n_path = delays_mat.n_cols;

    if (powers_mat.n_rows != n_cir || powers_mat.n_cols != n_path)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Delays and powers must have the same size.");

    // Convert rows to vector of column vectors
    std::vector<arma::vec> delays(n_cir);
    std::vector<arma::vec> powers(n_cir);
    for (arma::uword i = 0; i < n_cir; ++i)
    {
        delays[i] = delays_mat.row(i).t();
        powers[i] = powers_mat.row(i).t();
    }

    // Optional scalar inputs
    double threshold = (nrhs < 3) ? 100.0 : qd_mex_get_scalar<double>(prhs[2], "threshold", 100.0);
    double granularity = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "granularity", 0.0);

    // --- Declare outputs ---
    arma::vec ds;
    arma::vec mean_delay;

    // Set up optional output pointer
    arma::vec *p_mean_delay = (nlhs > 1) ? &mean_delay : nullptr;

    // --- Call library function ---
    CALL_QD(ds = quadriga_lib::calc_delay_spread<double>(delays, powers, threshold, granularity, p_mean_delay));

    // --- Write outputs ---
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&ds);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&mean_delay);
}
