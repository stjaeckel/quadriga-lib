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
# calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

## Description:
- Computes the aggregate cross-polarization ratio (XPR) from the polarization transfer matrices
  of all channel impulse responses (CIRs) using the total-power-ratio method.
- For each CIR, the total co-polarized and cross-polarized received powers are accumulated
  across all qualifying paths, and the XPR is obtained as a single ratio of the totals.
- In addition to the linear V/H basis, the XPR is also computed in the circular LHCP/RHCP basis.
- The LOS path is identified by comparing each path's absolute length against the direct
  TX-RX distance. All paths with `path_length < dTR + window_size` are excluded by default.
- If the total cross-polarized power is zero, the XPR is set to 0 (undefined).

## Usage:
```
[ xpr, pg ] = quadriga_lib.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos )
[ xpr, pg ] = quadriga_lib.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, include_los )
[ xpr, pg ] = quadriga_lib.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, include_los, window_size )
```

## Input Arguments:
- **`powers`** (required)<br>
  Path powers in Watts. A 2D matrix of size `[n_path_max, n_cir]` where columns are zero-padded
  if CIRs have different numbers of paths. Alternatively, for a single CIR, a column vector of
  length `[n_path]`.

- **`M`** (required)<br>
  Polarization transfer matrices. A 3D array of size `[8, n_path_max, n_cir]` with interleaved
  real/imaginary parts in column-major order.

- **`path_length`** (required)<br>
  Absolute path length from TX to RX in meters. A 2D matrix of size `[n_path_max, n_cir]`.

- **`tx_pos`** (required)<br>
  Transmitter position in Cartesian coordinates. Size `[3, 1]` (fixed) or `[3, n_cir]` (mobile).

- **`rx_pos`** (required)<br>
  Receiver position in Cartesian coordinates. Size `[3, 1]` (fixed) or `[3, n_cir]` (mobile).

- **`include_los`** (optional)<br>
  Logical flag. If `true`, include LOS paths in XPR calculation. Default: `false`.

- **`window_size`** (optional)<br>
  LOS window size in meters. Default: `0.01`.

## Output Arguments:
- **`xpr`** (optional)<br>
  Cross-polarization ratio in linear scale. Size `[n_cir, 6]` (double).<br>
  Columns: 1=aggregate linear, 2=V-XPR, 3=H-XPR, 4=aggregate circular, 5=LHCP, 6=RHCP.

- **`pg`** (optional)<br>
  Total path gain over all paths (including LOS). Column vector of length `[n_cir]` (double).
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // --- Validate argument counts ---
    if (nrhs < 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // --- Read required inputs ---
    // powers: split 2D matrix [n_path_max, n_cir] into vector of columns
    std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[0], 1);

    // M: split 3D array [8, n_path_max, n_cir] into vector of matrices
    std::vector<arma::mat> M = qd_mex_matlab2vector_Mat<double>(prhs[1], 2);

    // path_length: split 2D matrix [n_path_max, n_cir] into vector of columns
    std::vector<arma::vec> path_length = qd_mex_matlab2vector_Col<double>(prhs[2], 1);

    // tx_pos, rx_pos
    arma::mat tx_pos = qd_mex_get_double_Mat(prhs[3]);
    arma::mat rx_pos = qd_mex_get_double_Mat(prhs[4]);

    // --- Read optional inputs ---
    bool include_los = (nrhs < 6) ? false : qd_mex_get_scalar<bool>(prhs[5], "include_los", false);
    double window_size = (nrhs < 7) ? 0.01 : qd_mex_get_scalar<double>(prhs[6], "window_size", 0.01);

    // --- Declare outputs ---
    arma::mat xpr;
    arma::vec pg;

    arma::mat *p_xpr = (nlhs > 0) ? &xpr : nullptr;
    arma::vec *p_pg = (nlhs > 1) ? &pg : nullptr;

    // --- Call library function ---
    CALL_QD(quadriga_lib::calc_cross_polarization_ratio<double>(
        powers, M, path_length, tx_pos, rx_pos, p_xpr, p_pg, include_los, window_size));

    // --- Write outputs ---
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&xpr);

    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&pg);
}
