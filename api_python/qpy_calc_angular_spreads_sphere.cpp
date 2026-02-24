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
# calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads with spherical wrapping

## Description:
- Calculates the RMS azimuth and elevation angular spreads from a set of power-weighted angles.
- Uses spherical coordinate wrapping to avoid the pole singularity: the power-weighted mean
  direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies
  on the equator before computing spreads.
- Without spherical wrapping, azimuth spread near the poles is inflated (large azimuth spread
  despite energy being focused into a small solid angle). This method corrects for that.
- Optionally computes an optimal bank (roll) angle that maximizes azimuth spread and minimizes
  elevation spread, corresponding to the principal axes of the angular power distribution.
- If `pow` has only 1 row but `az` has `n_ang` rows, the power vector is replicated.
- If `el` has only 1 row but `az` has `n_ang` rows, elevation is assumed zero.

## Usage:
```
import quadriga_lib
as_spread, es_spread, orientation, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, pw)
as_spread, es_spread, orientation, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(
    az, el, pw, calc_bank_angle=True, quantize=0.0)
```

## Arguments:
- `np.ndarray **az**` (input)<br>
  Azimuth angles in [rad], ranging from -pi to pi. Shape `(n_ang, n_path)`.

- `np.ndarray **el**` (input)<br>
  Elevation angles in [rad], ranging from -pi/2 to pi/2. Shape `(n_ang, n_path)` or `(1, n_path)`.

- `np.ndarray **pw**` (input)<br>
  Path powers in [W]. Shape `(n_ang, n_path)` or `(1, n_path)`.

- `bool **calc_bank_angle** = True` (input)<br>
  If True, the optimal bank angle is computed analytically.

- `float **quantize** = 0.0` (input)<br>
  Angular quantization step in [deg]. Set to 0 to treat all paths independently.

## Returns:
- `np.ndarray **azimuth_spread**` (output)<br>
  RMS azimuth angular spread in [rad]. Shape `(n_ang,)`.

- `np.ndarray **elevation_spread**` (output)<br>
  RMS elevation angular spread in [rad]. Shape `(n_ang,)`.

- `np.ndarray **orientation**` (output)<br>
  Mean-angle orientation [bank, tilt, heading] in [rad]. Shape `(3, n_ang)`.

- `np.ndarray **phi**` (output)<br>
  Rotated azimuth angles in [rad]. Shape `(n_ang, n_path)`.

- `np.ndarray **theta**` (output)<br>
  Rotated elevation angles in [rad]. Shape `(n_ang, n_path)`.

## Example:
```
import numpy as np
import quadriga_lib

az = np.array([[0.1, 0.2, -0.1, 0.3]])
el = np.array([[0.0, 0.05, -0.05, 0.02]])
pw = np.array([[1.0, 2.0, 1.5, 0.5]])
as_spread, es_spread, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, pw)
```
MD!*/

static py::tuple calc_angular_spreads_sphere(py::array_t<double> az_py,
                                             py::array_t<double> el_py,
                                             py::array_t<double> pw_py,
                                             bool calc_bank_angle = true,
                                             double quantize = 0.0)
{
    // Convert inputs to Armadillo
    arma::mat az = qd_python_numpy2arma_Mat<double>(az_py);
    arma::mat el = qd_python_numpy2arma_Mat<double>(el_py);
    arma::mat pw = qd_python_numpy2arma_Mat<double>(pw_py);

    // Declare outputs — always compute all (no nullptr optimization in Python)
    arma::vec azimuth_spread, elevation_spread;
    arma::mat orientation, phi, theta;

    // Call library function
    quadriga_lib::calc_angular_spreads_sphere<double>(
        az, el, pw,
        &azimuth_spread, &elevation_spread, &orientation, &phi, &theta,
        calc_bank_angle, quantize);

    // Convert to Python
    auto as_py = qd_python_copy2numpy(azimuth_spread);
    auto es_py = qd_python_copy2numpy(elevation_spread);
    auto orient_py = qd_python_copy2numpy(orientation);
    auto phi_py = qd_python_copy2numpy(phi);
    auto theta_py = qd_python_copy2numpy(theta);

    return py::make_tuple(as_py, es_py, orient_py, phi_py, theta_py);
}