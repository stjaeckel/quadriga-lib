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
# calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

## Description:
- Computes the aggregate cross-polarization ratio (XPR) from the polarization transfer matrices
  of all channel impulse responses (CIRs) using the total-power-ratio method.
- For each CIR, the total co-polarized and cross-polarized received powers are accumulated
  across all qualifying paths, and the XPR is obtained as a single ratio of the totals.
- In addition to the linear V/H basis, the XPR is also computed in the circular LHCP/RHCP basis.
- The LOS path is identified by comparing each path's absolute length against the direct
  TX-RX distance. All paths with `path_length < dTR + window_size` are excluded from
  the XPR calculation by default (controlled by `include_los`).
- If the total cross-polarized power is zero, the XPR is set to 0 (undefined).

## Usage:
```
import quadriga_lib
xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos )
xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, include_los, window_size )
```

## Arguments:
- `list of np.ndarray **powers**` (input)<br>
  Path powers in Watts. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `list of np.ndarray **M**` (input)<br>
  Polarization transfer matrices. List of length `n_cir`, each element is a 2D array of size `[8, n_path]`.

- `list of np.ndarray **path_length**` (input)<br>
  Absolute path length from TX to RX in meters. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `np.ndarray **tx_pos**` (input)<br>
  Transmitter position. Size `[3, 1]` (fixed) or `[3, n_cir]` (mobile).

- `np.ndarray **rx_pos**` (input)<br>
  Receiver position. Size `[3, 1]` (fixed) or `[3, n_cir]` (mobile).

- `bool **include_los** = False` (input)<br>
  If `True`, include LOS paths in the XPR calculation.

- `float **window_size** = 0.01` (input)<br>
  LOS window size in meters.

## Returns:
- `np.ndarray **xpr**` (output)<br>
  Cross-polarization ratio, linear scale. Size `[n_cir, 6]`.
  Columns: 0=aggregate linear, 1=V-XPR, 2=H-XPR, 3=aggregate circular, 4=LHCP, 5=RHCP.

- `np.ndarray **pg**` (output)<br>
  Total path gain over all paths. 1D array of length `[n_cir]`.
MD!*/

py::tuple calc_cross_polarization_ratio(py::list powers_py, py::list M_py, py::list path_length_py,
                                        py::array_t<double> tx_pos_py, py::array_t<double> rx_pos_py,
                                        bool include_los, double window_size)
{
    // Convert inputs
    std::vector<arma::vec> powers = qd_python_list2vector_Col<double>(powers_py);
    std::vector<arma::mat> M = qd_python_list2vector_Mat<double>(M_py);
    std::vector<arma::vec> path_length = qd_python_list2vector_Col<double>(path_length_py);
    arma::mat tx_pos = qd_python_numpy2arma_Mat<double>(tx_pos_py);
    arma::mat rx_pos = qd_python_numpy2arma_Mat<double>(rx_pos_py);

    // Declare outputs
    arma::mat xpr;
    arma::vec pg;

    // Call library
    quadriga_lib::calc_cross_polarization_ratio<double>(powers, M, path_length, tx_pos, rx_pos,
                                                        &xpr, &pg, include_los, window_size);

    // Convert outputs
    auto xpr_py = qd_python_copy2numpy(xpr);
    auto pg_py = qd_python_copy2numpy(pg);

    return py::make_tuple(xpr_py, pg_py);
}
