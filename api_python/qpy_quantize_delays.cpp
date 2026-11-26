// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# quantize_delays
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
coeff_q, delay_q = quadriga_lib.channel.quantize_delays( coeff, delay, *, tap_spacing, max_taps, \
    power_exponent, fix_taps, stack )
```

## Inputs:
- **`coeff`** — Complex coefficients; list of `n_snap` arrays `(n_rx, n_tx, n_path_s)`, at least one; `n_path_s` may
  differ between the snapshots, `n_rx` and `n_tx` (>= 1) not
- **`delay`** — Delays in s; list of `n_snap` arrays `(n_rx, n_tx, n_path_s)` or `(1, 1, n_path_s)`, one of the two
  forms for all snapshots; finite, above `-tap_spacing / 2` and below `2^31` · `tap_spacing`
- **`tap_spacing`** — Tap spacing in s, > 0 and finite; 5 ns is a sampling rate of 200 MHz; default: 5e-9
- **`max_taps`** — Largest number of taps of a grid; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent $\alpha$, > 0; default: 1.0
- **`fix_taps`** — Which coefficients share a grid; default: 0

  | Value | Grid |
  | ----- | ---- |
  | 0 | One grid per antenna pair and snapshot |
  | 1 | One grid for all antenna pairs and snapshots |
  | 2 | One grid per snapshot for all its antenna pairs; the grids of the snapshots differ |
  | 3 | One grid per antenna pair for all snapshots |

- **`stack`** — `True`: the outputs as 4D arrays with the snapshots on the last axis; `False`: as lists with one array
  per snapshot; default: `False`

## Outputs:
- **`coeff_q`** — Complex coefficients per tap; complex128; list of `n_snap` arrays `(n_rx, n_tx, n_taps)`, or
  `(n_rx, n_tx, n_taps, n_snap)` with `stack`
- **`delay_q`** — Tap delays in s; list of `n_snap` arrays `(n_rx, n_tx, n_taps)`, or `(n_rx, n_tx, n_taps, n_snap)`
  with `stack`; `(1, 1, n_taps)` and `(1, 1, n_taps, n_snap)` for delays `(1, 1, n_path_s)` with `fix_taps` 1 or 2

## Notes:
- Negative delays: delays in `(-tap_spacing / 2, 0)` count as 0, all their power goes to tap 0 (the error is below
  `tap_spacing / 2`, as for rounding to the nearest tap)
- Relative delays (`use_absolute_delays = False` in [[channel.spherical]], [[channel.planar]] and
  [[channel.multifreq]]) are negative for element pairs that are closer to each other than the phase centres, by up
  to (half the Tx array size + half the Rx array size) / propagation speed. They stay above `-tap_spacing / 2` only
  while these half sizes add up to less than about 0.75 m for radio at 5 ns taps, or 3.6 mm for sound at 48 kHz
  taps, i.e. practically never for acoustic arrays: use absolute delays then
- The upper limit of `2^31` taps is e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz; run time and memory depend on the
  number of paths and taps, not on the delay values

## See also:
- [[freq_response]] (frequency response of the paths or taps)
- [[calc_delay_spread]] (delay spread, with an optional delay granularity)
MD!*/

py::tuple quantize_delays(const py::handle &coeff,
                          const py::handle &delay,
                          double tap_spacing,
                          arma::uword max_taps,
                          double power_exponent,
                          int fix_taps,
                          py::handle stack)
{
    const bool stack_b = qd_python_flag(stack, false, "stack");

    // Convert the complex coefficient list to split real / imaginary cubes (length = n_snap)
    if (!py::isinstance<py::list>(coeff))
        throw std::invalid_argument("Input 'coeff' must be a list.");
    std::vector<arma::Cube<double>> coeff_re_a, coeff_im_a;
    qd_python_list2vector_Cube_Cplx<double>(py::cast<py::list>(coeff), coeff_re_a, coeff_im_a);

    // Convert delay list (ragged path counts allowed)
    if (!py::isinstance<py::list>(delay))
        throw std::invalid_argument("Input 'delay' must be a list.");
    auto delay_a = qd_python_list2vector_Cube<double>(py::cast<py::list>(delay));

    std::vector<arma::Cube<double>> coeff_re_q, coeff_im_q, delay_q;

    quadriga_lib::quantize_delays<double>(coeff_re_a, coeff_im_a, delay_a,
                                          coeff_re_q, coeff_im_q, delay_q,
                                          tap_spacing, max_taps, power_exponent, fix_taps);

    // Delay output (always real); list (stack=False) or 4D array (stack=True)
    py::object delay_q_py;
    if (stack_b)
        delay_q_py = qd_python_stack2numpy(&delay_q);
    else
        delay_q_py = qd_python_copy2list(&delay_q);

    // Complex coefficient output, independent of the input form
    py::object coeff_q_py;
    if (stack_b)
        coeff_q_py = qd_python_stack2numpy<double, std::complex<double>>(&coeff_re_q, &coeff_im_q);
    else
        coeff_q_py = qd_python_copy2list<arma::Cube<double>, std::complex<double>>(&coeff_re_q, &coeff_im_q);
    return py::make_tuple(coeff_q_py, delay_q_py);
}
