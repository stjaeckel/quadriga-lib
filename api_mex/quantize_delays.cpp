// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# QUANTIZE_DELAYS
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1−δ)^α
  and δ^α, where δ is the fractional offset within the bin and α is `power_exponent`
- Two-tap interpolation avoids discontinuities when delays cross tap boundaries
- Use `power_exponent = 1.0` for narrowband (linear interpolation) or `0.5` for wideband
  (incoherent power preservation)
- If all fractional offsets are below 0.01 or above 0.99, weight computation is skipped but
  tap-selection logic still applies
- Input `delay` may be per-antenna `[n_rx, n_tx, n_path, n_snap]` or shared
  `[1, 1, n_path, n_snap]`; shared delays are expanded internally when `fix_taps` is 0 or 3
- Output arrays are zero-padded along the tap dimension so that all snapshots share the same `n_taps`

## Usage:
```
[ coeff_re_q, coeff_im_q, delay_q ] = quadriga_lib.quantize_delays( coeff_re, coeff_im, delay, ...
    tap_spacing, max_no_taps, power_exponent, fix_taps );
```

## Inputs:
- **`coeff_re`** — Channel coefficients, real part; `[n_rx, n_tx, n_path, n_snap]`
- **`coeff_im`** — Channel coefficients, imaginary part; `[n_rx, n_tx, n_path, n_snap]`
- **`delay`** — Path delays in seconds; `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`
- **`tap_spacing`** *(optional)* — Delay bin spacing in seconds; 5 ns corresponds to 200 MHz
  sampling rate; default: 5e-9
- **`max_no_taps`** *(optional)* — Maximum number of output taps; 0 = unlimited; default: 48
- **`power_exponent`** *(optional)* — Interpolation exponent α; default: 1.0
- **`fix_taps`** *(optional)* — Delay grid sharing mode; default: 0<br><br>
  | Value | Meaning                                                                                         |
  | ----- | ----------------------------------------------------------------------------------------------- |
  | 0     | Per tx-rx pair and snapshot; output delays `[n_rx, n_tx, n_taps, n_snap]`                       |
  | 1     | Single shared grid across all snapshots and tx-rx pairs; output delays `[1, 1, n_taps, n_snap]` |
  | 2     | Per snapshot; output delays `[1, 1, n_taps, n_snap]`, each snapshot independent                 |
  | 3     | Per tx-rx pair across all snapshots; output delays `[n_rx, n_tx, n_taps, n_snap]`               |

## Outputs:
- **`coeff_re_q`** — Output coefficients, real part; `[n_rx, n_tx, n_taps, n_snap]`
- **`coeff_im_q`** — Output coefficients, imaginary part; `[n_rx, n_tx, n_taps, n_snap]`
- **`delay_q`** — Output delays in seconds; `[n_rx, n_tx, n_taps, n_snap]` or `[1, 1, n_taps, n_snap]` depending on `fix_taps`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs < 3 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs: 4D arrays are split along the 4th dimension into std::vector<arma::Cube>
    const auto coeff_re = qd_mex_matlab2vector_Cube<double>(prhs[0], 3);
    const auto coeff_im = qd_mex_matlab2vector_Cube<double>(prhs[1], 3);
    const auto delay = qd_mex_matlab2vector_Cube<double>(prhs[2], 3);

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
