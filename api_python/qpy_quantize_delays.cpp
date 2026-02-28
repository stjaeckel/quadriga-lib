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

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

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
- Input delays may be per-antenna `[n_rx, n_tx, n_path_s]` or shared `[1, 1, n_path_s]`. Output
  delay shape depends on `fix_taps` mode.
- The number of paths `n_path_s` may differ across snapshots.

## Usage:
```
import quadriga_lib
coeff_re_q, coeff_im_q, delay_q = quadriga_lib.channel.quantize_delays(
    coeff_re, coeff_im, delay,
    tap_spacing=5e-9, max_no_taps=48, power_exponent=1.0, fix_taps=0)
```

## Arguments:
- `list ***coeff_re**` (input)<br>
  Channel coefficients, real part. List of `n_snap` numpy arrays, each of shape
  `[n_rx, n_tx, n_path_s]`. The number of paths may differ per snapshot.

- `list ***coeff_im**` (input)<br>
  Channel coefficients, imaginary part. Same shapes as `coeff_re`.

- `list ***delay**` (input)<br>
  Path delays in seconds. List of `n_snap` numpy arrays, each of shape
  `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`.

- `float **tap_spacing** = 5e-9` (input)<br>
  Spacing of the delay bins in seconds.

- `int **max_no_taps** = 48` (input)<br>
  Maximum number of output taps. 0 means unlimited.

- `float **power_exponent** = 1.0` (input)<br>
  Interpolation exponent. Use 1.0 for narrowband or 0.5 for wideband.

- `int **fix_taps** = 0` (input)<br>
  Delay sharing mode: 0 = per tx-rx pair and snapshot, 1 = single grid for all,
  2 = per snapshot, 3 = per tx-rx pair.

## Returns:
- `np.ndarray **coeff_re_q**` (output)<br>
  Output coefficients, real part. 4D array of shape `[n_rx, n_tx, n_taps, n_snap]`.

- `np.ndarray **coeff_im_q**` (output)<br>
  Output coefficients, imaginary part. 4D array of shape `[n_rx, n_tx, n_taps, n_snap]`.

- `np.ndarray **delay_q**` (output)<br>
  Output delays in seconds. 4D array of shape `[n_rx, n_tx, n_taps, n_snap]` or
  `[1, 1, n_taps, n_snap]`.
MD!*/

py::tuple quantize_delays(py::list coeff_re_list,
                          py::list coeff_im_list,
                          py::list delay_list,
                          double tap_spacing,
                          unsigned long long max_no_taps,
                          double power_exponent,
                          int fix_taps)
{
    // Convert Python lists of 3D arrays to std::vector<arma::Cube<double>>
    auto coeff_re = qd_python_list2vector_Cube<double>(coeff_re_list);
    auto coeff_im = qd_python_list2vector_Cube<double>(coeff_im_list);
    auto delay = qd_python_list2vector_Cube<double>(delay_list);

    // Declare output vectors
    std::vector<arma::Cube<double>> coeff_re_q, coeff_im_q, delay_q;

    // Call C++ library
    quadriga_lib::quantize_delays<double>(
        &coeff_re, &coeff_im, &delay,
        &coeff_re_q, &coeff_im_q, &delay_q,
        tap_spacing, (arma::uword)max_no_taps, power_exponent, fix_taps);

    // Convert outputs to 4D numpy arrays (all output cubes have uniform size)
    auto coeff_re_q_p = qd_python_copy2numpy(coeff_re_q);
    auto coeff_im_q_p = qd_python_copy2numpy(coeff_im_q);
    auto delay_q_p = qd_python_copy2numpy(delay_q);

    return py::make_tuple(coeff_re_q_p, coeff_im_q_p, delay_q_p);
}

