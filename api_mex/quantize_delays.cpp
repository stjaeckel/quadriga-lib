// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# QUANTIZE_DELAYS
Map the path delays to a tap grid by two-tap interpolation

- For time-domain processing with a tapped delay line: a path with delay $\tau$ and coefficient $a$ goes to the two
  neighbouring taps $\lfloor \tau / \Delta \rfloor$ and $\lfloor \tau / \Delta \rfloor + 1$ of the grid with the
  spacing $\Delta$ = `tap_spacing`, with the coefficients $a \, (1 - \delta)^\alpha$ and $a \, \delta^\alpha$, where
  $\delta$ is the fractional offset of $\tau / \Delta$ and $\alpha$ = `power_exponent`; the coefficients change smoothly
  when a delay crosses a tap
- $\alpha = 1$ interpolates linearly (narrowband), $\alpha = 0.5$ keeps the power of each path (wideband)
- A grid holds the taps with power, at most `max_taps`: all taps of the interpolation if they fit, else the nearest
  taps of the paths, strongest first, and the remaining places for the strongest other taps. A path whose two taps
  are not both in the grid goes to its nearest tap with its full coefficient, or is dropped if that tap is not in the
  grid either
- If every delay is within 1 % of the spacing from a tap, each path goes to its nearest tap with its full coefficient
- All snapshots of the output have the same number of taps `n_taps`, the size of the largest grid; the taps beyond a
  shorter grid have zero coefficients and delay 0, taps that are zero in all snapshots at the end are removed

## Usage:
```
[ coeff_q, delay_q ] = quadriga_lib.quantize_delays( coeff, delay, tap_spacing, max_taps, ...
    power_exponent, fix_taps );
```

## Inputs:
- **`coeff`** — Complex coefficients, the paths of a snapshot zero-padded to `n_path`; `[n_rx, n_tx, n_path, n_snap]`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`; finite, above
  `-tap_spacing / 2` and below `2^31` · `tap_spacing`
- **`tap_spacing`** — Tap spacing in s, > 0 and finite; 5 ns is a sampling rate of 200 MHz; default: 5e-9
- **`max_taps`** — Largest number of taps of a grid; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent $\alpha$, > 0; default: 1
- **`fix_taps`** — Which coefficients share a grid; default: 0

  | Value | Grid |
  | ----- | ---- |
  | 0 | One grid per antenna pair and snapshot |
  | 1 | One grid for all antenna pairs and snapshots |
  | 2 | One grid per snapshot for all its antenna pairs; the grids of the snapshots differ |
  | 3 | One grid per antenna pair for all snapshots |

## Outputs:
- **`coeff_q`** — Complex coefficients per tap; `[n_rx, n_tx, n_taps, n_snap]`
- **`delay_q`** — Tap delays in s; `[n_rx, n_tx, n_taps, n_snap]`, or `[1, 1, n_taps, n_snap]` for delays
  `[1, 1, n_path, n_snap]` with `fix_taps` 1 or 2

## Notes:
- Negative delays: delays in `(-tap_spacing / 2, 0)` count as 0, all their power goes to tap 0 (the error is below
  `tap_spacing / 2`, as for rounding to the nearest tap)
- Relative delays (`use_absolute_delays = false` in [[get_channels_spherical]], [[get_channels_planar]] and
  [[get_channels_multifreq]]) are negative for element pairs that are closer to each other than the phase centres,
  by up to (half the Tx array size + half the Rx array size) / propagation speed. They stay above
  `-tap_spacing / 2` only while these half sizes add up to less than about 0.75 m for radio at 5 ns taps, or 3.6 mm
  for sound at 48 kHz taps, i.e. practically never for acoustic arrays: use absolute delays then
- The upper limit of `2^31` taps is e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz; run time and memory depend on the
  number of paths and taps, not on the delay values

## See also:
- [[freq_response]] (frequency response of the paths or taps)
- [[calc_delay_spread]] (delay spread, with an optional delay granularity)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs < 2 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // 4D arrays are split along the 4th dimension; the coefficients into real and imaginary part
    std::vector<arma::cube> coeff_re, coeff_im;
    qd_mex_get_complex_CubeVec<double>(prhs[0], "coeff", coeff_re, coeff_im);
    const auto delay = qd_mex_matlab2vector_Cube<double>(prhs[1], 3, "delay");

    double tap_spacing = (nrhs < 3) ? 5.0e-9 : qd_mex_get_scalar<double>(prhs[2], "tap_spacing", 5.0e-9);
    arma::uword max_taps = (nrhs < 4) ? 48 : qd_mex_get_scalar<arma::uword>(prhs[3], "max_taps", 48);
    double power_exponent = (nrhs < 5) ? 1.0 : qd_mex_get_scalar<double>(prhs[4], "power_exponent", 1.0);
    int fix_taps = (nrhs < 6) ? 0 : qd_mex_get_scalar<int>(prhs[5], "fix_taps", 0);

    std::vector<arma::Cube<double>> coeff_re_q, coeff_im_q, delay_q;

    CALL_QD(quadriga_lib::quantize_delays<double>(
        coeff_re, coeff_im, delay,
        coeff_re_q, coeff_im_q, delay_q,
        tap_spacing, max_taps, power_exponent, fix_taps));

    plhs[0] = qd_mex_vector2matlab(&coeff_re_q, &coeff_im_q);
    if (nlhs > 1)
        plhs[1] = qd_mex_vector2matlab(&delay_q);
}
