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
- The transmitter and receiver positions can be fixed (shape `(3,)` or `(3, 1)`) or mobile
  (shape `(3, n_cir)`). Fixed positions are reused for all channel snapshots.
- Output `pg` returns the total path gain (sum of all path powers) for each snapshot.

## Usage:
```
import quadriga_lib
kf, pg = quadriga_lib.tools.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size=0.01 )
```

## Arguments:
- `list of ndarray **powers**` (input)<br>
  Path powers in Watts [W]. List of length `n_cir`, where each element is a 1D numpy array of
  length `n_path`.

- `list of ndarray **path_length**` (input)<br>
  Absolute path lengths from TX to RX phase center in meters. List of length `n_cir`, where each
  element is a 1D numpy array of length `n_path` matching the corresponding entry in `powers`.

- `ndarray **tx_pos**` (input)<br>
  Transmitter position in Cartesian coordinates. Shape `(3,)` or `(3, 1)` for fixed TX, or
  `(3, n_cir)` for mobile TX.

- `ndarray **rx_pos**` (input)<br>
  Receiver position in Cartesian coordinates. Shape `(3,)` or `(3, 1)` for fixed RX, or
  `(3, n_cir)` for mobile RX.

- `float **window_size** = 0.01` (input)<br>
  LOS window size in meters. Paths with length ≤ `dTR + window_size` are considered LOS.

## Returns:
- `ndarray **kf**` (output)<br>
  Rician K-Factor on linear scale. Shape `(n_cir,)`.

- `ndarray **pg**` (output)<br>
  Total path gain (sum of path powers). Shape `(n_cir,)`.
MD!*/

py::tuple calc_rician_k_factor(py::list powers_list, py::list path_length_list,
                               py::array_t<double> tx_pos_arr, py::array_t<double> rx_pos_arr,
                               double window_size)
{
    // Convert Python inputs to C++ types
    std::vector<arma::vec> powers = qd_python_list2vector_Col<double>(powers_list);
    std::vector<arma::vec> path_length = qd_python_list2vector_Col<double>(path_length_list);

    arma::mat tx_pos = qd_python_numpy2arma_Mat<double>(tx_pos_arr);
    arma::mat rx_pos = qd_python_numpy2arma_Mat<double>(rx_pos_arr);

    // Declare outputs
    arma::vec kf, pg;

    // Call C++ library (always compute all outputs for Python)
    quadriga_lib::calc_rician_k_factor<double>(powers, path_length, tx_pos, rx_pos, &kf, &pg, window_size);

    // Convert to Python
    auto kf_p = qd_python_copy2numpy(kf);
    auto pg_p = qd_python_copy2numpy(pg);

    return py::make_tuple(kf_p, pg_p);
}