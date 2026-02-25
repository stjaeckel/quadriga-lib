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
- Inputs use lists of 1D numpy arrays so that each CIR can have a different number of paths.
- Uses spherical coordinate wrapping to avoid the pole singularity: the power-weighted mean
  direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies
  on the equator before computing spreads.
- Without spherical wrapping, azimuth spread near the poles is inflated (large azimuth spread
  despite energy being focused into a small solid angle). This method corrects for that.
- Optionally computes an optimal bank (roll) angle that maximizes azimuth spread and minimizes
  elevation spread, corresponding to the principal axes of the angular power distribution.
- Setting `disable_wrapping` to True skips the rotation and computes spreads directly from
  raw angles.

## Usage:
```
import quadriga_lib
as_spread, es_spread, orientation, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(
    az, el, powers)
as_spread, es_spread, orientation, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(
    az, el, powers, disable_wrapping=False, calc_bank_angle=True, quantize=0.0)
```

## Arguments:
- `list of np.ndarray **az**` (input)<br>
  Azimuth angles in [rad]. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `list of np.ndarray **el**` (input)<br>
  Elevation angles in [rad]. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `list of np.ndarray **powers**` (input)<br>
  Path powers in [W]. List of length `n_cir`, each element is a 1D array of length `n_path`.

- `bool **disable_wrapping** = False` (input)<br>
  If True, skip spherical rotation and compute spreads from raw angles.

- `bool **calc_bank_angle** = True` (input)<br>
  If True, compute the optimal bank angle analytically.

- `float **quantize** = 0.0` (input)<br>
  Angular quantization step in [deg]. Set to 0 for no quantization.

## Returns:
- `np.ndarray **azimuth_spread**` (output)<br>
  RMS azimuth angular spread in [rad]. Shape `(n_cir,)`.

- `np.ndarray **elevation_spread**` (output)<br>
  RMS elevation angular spread in [rad]. Shape `(n_cir,)`.

- `np.ndarray **orientation**` (output)<br>
  Mean-angle orientation [bank, tilt, heading] in [rad]. Shape `(3, n_cir)`.

- `list of np.ndarray **phi**` (output)<br>
  Rotated azimuth angles in [rad]. List of length `n_cir`.

- `list of np.ndarray **theta**` (output)<br>
  Rotated elevation angles in [rad]. List of length `n_cir`.

## Example:
```
import numpy as np
import quadriga_lib

az = [np.array([0.1, -0.1, 0.05]), np.array([0.2, -0.2, 0.1, -0.1])]
el = [np.array([0.0, 0.0, 0.0]), np.array([0.05, -0.05, 0.0, 0.0])]
powers = [np.array([1.0, 1.0, 0.5]), np.array([2.0, 1.0, 1.5, 0.5])]
as_spread, es_spread, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(
    az, el, powers)
```
MD!*/

static py::tuple calc_angular_spreads_sphere(py::list az_py,
                                             py::list el_py,
                                             py::list powers_py,
                                             bool disable_wrapping = false,
                                             bool calc_bank_angle = true,
                                             double quantize = 0.0)
{
    // Convert inputs to std::vector<arma::vec>
    std::vector<arma::vec> az = qd_python_list2vector_Col<double>(az_py);
    std::vector<arma::vec> el = qd_python_list2vector_Col<double>(el_py);
    std::vector<arma::vec> powers = qd_python_list2vector_Col<double>(powers_py);

    // Declare outputs
    arma::vec azimuth_spread, elevation_spread;
    arma::mat orientation;
    std::vector<arma::vec> phi, theta;

    // Call library function
    quadriga_lib::calc_angular_spreads_sphere<double>(
        az, el, powers,
        &azimuth_spread, &elevation_spread, &orientation, &phi, &theta,
        disable_wrapping, calc_bank_angle, quantize);

    // Convert to Python
    auto as_py = qd_python_copy2numpy(azimuth_spread);
    auto es_py = qd_python_copy2numpy(elevation_spread);
    auto orient_py = qd_python_copy2numpy(orientation);
    auto phi_py = qd_python_copy2numpy(phi);
    auto theta_py = qd_python_copy2numpy(theta);

    return py::make_tuple(as_py, es_py, orient_py, phi_py, theta_py);
}

// pybind11 module declaration (to be placed in the appropriate module init)
// m.def("calc_angular_spreads_sphere", &calc_angular_spreads_sphere,
//       py::arg("az"),
//       py::arg("el"),
//       py::arg("powers"),
//       py::arg("disable_wrapping") = false,
//       py::arg("calc_bank_angle") = true,
//       py::arg("quantize") = 0.0);
