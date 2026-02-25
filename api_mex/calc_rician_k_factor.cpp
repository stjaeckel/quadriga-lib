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
# calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

## Description:
- The Rician K-Factor (KF) is defined as the ratio of signal power in the dominant line-of-sight
  (LOS) path to the power in the scattered (non-line-of-sight, NLOS) paths.
- The LOS path is identified by matching the absolute path length with the direct distance between
  TX and RX positions (`dTR`).
- All paths arriving within `dTR + window_size` are considered LOS and their power is summed.
- Paths arriving after `dTR + window_size` are considered NLOS and their power is summed.
- If the total NLOS power is zero, the K-Factor is set to infinity.
- If the total LOS power is zero, the K-Factor is set to zero.
- The transmitter and receiver positions can be fixed (size `[3, 1]`) or mobile (size `[3, n_cir]`).
  Fixed positions are reused for all channel snapshots.
- Optional output `pg` returns the total path gain (sum of all path powers) for each snapshot.

## Usage:
```
[ kf, pg ] = quadriga_lib.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size );
```

## Input Arguments:
- **`powers`** (required)<br>
  Path powers in Watts [W]. A 2D matrix of size `[n_path_max, n_cir]` where columns correspond to
  channel snapshots. Shorter paths are zero-padded.

- **`path_length`** (required)<br>
  Absolute path lengths from TX to RX phase center in meters. A 2D matrix of size
  `[n_path_max, n_cir]` matching `powers`.

- **`tx_pos`** (required)<br>
  Transmitter position in Cartesian coordinates. Size `[3, 1]` for fixed TX or `[3, n_cir]` for
  mobile TX. Type: double.

- **`rx_pos`** (required)<br>
  Receiver position in Cartesian coordinates. Size `[3, 1]` for fixed RX or `[3, n_cir]` for
  mobile RX. Type: double.

- **`window_size`** (optional)<br>
  LOS window size in meters. Default: `0.01`. Paths with length ≤ `dTR + window_size` are
  considered LOS.

## Output Arguments:
- **`kf`** (optional)<br>
  Rician K-Factor on linear scale. Size `[n_cir, 1]`, type: double.

- **`pg`** (optional)<br>
  Total path gain (sum of path powers). Size `[n_cir, 1]`, type: double.
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs: powers and path_length as vectors of column vectors
    std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[0], 1);
    std::vector<arma::vec> path_length = qd_mex_matlab2vector_Col<double>(prhs[1], 1);

    // Read TX and RX positions
    arma::mat tx_pos = qd_mex_get_double_Mat(prhs[2]);
    arma::mat rx_pos = qd_mex_get_double_Mat(prhs[3]);

    // Read optional window_size
    double window_size = (nrhs < 5) ? 0.01 : qd_mex_get_scalar<double>(prhs[4], "window_size", 0.01);

    // Declare output variables
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
