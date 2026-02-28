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
Channel functions
SECTION!*/

/*!MD
# quantize_delays
Fixes the path delays to a grid of delay bins

## Description:
- For channel emulation with finite delay resolution, path delays must be mapped to a fixed grid
  of delay bins (taps). This function approximates each path delay using two adjacent taps with
  power-weighted coefficients, producing smooth transitions in the frequency domain.
- For a path at fractional offset &delta; between tap indices, two taps are created with complex
  coefficients scaled by (1&minus;&delta;)^&alpha; and &delta;^&alpha;, where &alpha; is the power
  exponent.
- Input delays may be per-antenna `[n_rx, n_tx, n_path, n_snap]` or shared `[1, 1, n_path, n_snap]`.

## Usage:
```
[ coeff_re_q, coeff_im_q, delay_q ] = quadriga_lib.quantize_delays( coeff_re, coeff_im, delay, ...
    tap_spacing, max_no_taps, power_exponent, fix_taps );
```

## Input Arguments:
- **`coeff_re`** (required)<br>
  Channel coefficients, real part. 4D array of size `[n_rx, n_tx, n_path, n_snap]` (double).

- **`coeff_im`** (required)<br>
  Channel coefficients, imaginary part. 4D array of size `[n_rx, n_tx, n_path, n_snap]` (double).

- **`delay`** (required)<br>
  Path delays in seconds. 4D array of size `[n_rx, n_tx, n_path, n_snap]` or
  `[1, 1, n_path, n_snap]` (double).

- **`tap_spacing`** (optional)<br>
  Spacing of the delay bins in seconds. Scalar double. Default: 5e-9

- **`max_no_taps`** (optional)<br>
  Maximum number of output taps. Scalar integer. 0 = unlimited. Default: 48

- **`power_exponent`** (optional)<br>
  Interpolation exponent. Scalar double. Default: 1.0

- **`fix_taps`** (optional)<br>
  Delay sharing mode. Scalar integer (0-3). Default: 0<br>
  0 = per tx-rx pair and snapshot, 1 = single grid for all,
  2 = per snapshot, 3 = per tx-rx pair.

## Output Arguments:
- **`coeff_re_q`**<br>
  Output coefficients, real part. 4D array of size `[n_rx, n_tx, n_taps, n_snap]` (double).

- **`coeff_im_q`**<br>
  Output coefficients, imaginary part. 4D array of size `[n_rx, n_tx, n_taps, n_snap]` (double).

- **`delay_q`**<br>
  Output delays in seconds. 4D array of size `[n_rx, n_tx, n_taps, n_snap]` or
  `[1, 1, n_taps, n_snap]` (double).
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs: 4D arrays are split along the 4th dimension into std::vector<arma::Cube>
    std::vector<arma::Cube<double>> coeff_re = qd_mex_matlab2vector_Cube<double>(prhs[0], 3);
    std::vector<arma::Cube<double>> coeff_im = qd_mex_matlab2vector_Cube<double>(prhs[1], 3);
    std::vector<arma::Cube<double>> delay = qd_mex_matlab2vector_Cube<double>(prhs[2], 3);

    // Optional scalar parameters
    double tap_spacing = (nrhs < 4) ? 5.0e-9 : qd_mex_get_scalar<double>(prhs[3], "tap_spacing", 5.0e-9);
    arma::uword max_no_taps = (nrhs < 5) ? 48 : qd_mex_get_scalar<arma::uword>(prhs[4], "max_no_taps", 48);
    double power_exponent = (nrhs < 6) ? 1.0 : qd_mex_get_scalar<double>(prhs[5], "power_exponent", 1.0);
    int fix_taps = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "fix_taps", 0);

    // Declare output vectors
    std::vector<arma::Cube<double>> coeff_re_q, coeff_im_q, delay_q;

    // Call library function
    CALL_QD(quadriga_lib::quantize_delays<double>(
        &coeff_re, &coeff_im, &delay,
        &coeff_re_q, &coeff_im_q, &delay_q,
        tap_spacing, max_no_taps, power_exponent, fix_taps));

    // Write outputs — convert std::vector<Cube> back to 4D MATLAB arrays
    if (nlhs > 0)
        plhs[0] = qd_mex_vector2matlab(&coeff_re_q);
    if (nlhs > 1)
        plhs[1] = qd_mex_vector2matlab(&coeff_im_q);
    if (nlhs > 2)
        plhs[2] = qd_mex_vector2matlab(&delay_q);
}
