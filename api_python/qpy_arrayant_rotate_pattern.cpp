// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ROTATE_PATTERN
Rotates antenna patterns

## Description:
This function transforms the radiation patterns of array antenna elements, allowing for
precise rotations around the three principal axes (x, y, z) of the local Cartesian coordinate system.
The 3 rotations are applies in the order: 1. rotation around the x-axis (bank angle);
2. rotation around the y-axis (tilt angle), 3. rotation around the z-axis (heading angle)

## Usage:

```
from quadriga_lib import arrayant
arrayant_out = arrayant.rotate_pattern(arrayant, bank, tilt, head, usage, element);
```

## Input Arguments:
- **`arrayant`**<br>
  Dictionary containing the arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements]`
  `coupling_re`    | Coupling matrix, real part, optional                  | Size: `[n_elements, n_ports]`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Size: `[n_elements, n_ports]`
  `center_freq`    | Center frequency in [Hz], optional                    | Scalar
  `name`           | Name of the array antenna object, optional            | String

- **`x_deg`** (optional)<br>
  The rotation angle around x-axis (bank angle) in [degrees]

- **`y_deg`** (optional)<br>
  The rotation angle around y-axis (tilt angle) in [degrees]

- **`z_deg`** (optional)<br>
  The rotation angle around z-axis (heading angle) in [degrees]

- **`usage`** (optional)<br>
  The optional parameter 'usage' can limit the rotation procedure either to the pattern or polarization.
  `usage = 0` | Rotate both, pattern and polarization, adjusts sampling grid (default)
  `usage = 1` | Rotate only pattern, adjusts sampling grid
  `usage = 2` | Rotate only polarization
  `usage = 3` | Rotate both, but do not adjust the sampling grid

- **`element`** (optional)<br>
  The element indices for which the pattern should be transformed. Optional parameter. Values must
  be between 1 and n_elements. If this parameter is not provided (or an empty array is passed),
  all elements will be rotated by the same angles. Size: `[1, n_elements]` or empty `[]`

## Output Arguments:
- **`arrayant_out`**<br>
  Dictionary containing the arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted             | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions                     | Size: `[3, n_elements]`
  `coupling_re`    | Coupling matrix, real part                            | Size: `[n_elements, n_ports]`
  `coupling_im`    | Coupling matrix, imaginary part                       | Size: `[n_elements, n_ports]`
  `center_freq`    | Center frequency in [Hz], default = 0.3 GHz           | Scalar
  `name`           | Name of the array antenna object                      | String
MD!*/

py::dict arrayant_rotate_pattern(const py::dict &arrayant,
                                 double x_deg,
                                 double y_deg,
                                 double z_deg,
                                 unsigned usage,
                                 const py::array_t<unsigned> &element)
{
    auto ant = qd_python_dict2arrayant(arrayant, false); // Copy
    const auto element_ind = qd_python_numpy2arma_Col(element, true);

    if (element_ind.n_elem == 0)
        ant.rotate_pattern(x_deg, y_deg, z_deg, usage);
    else
        for (auto el : element_ind)
            ant.rotate_pattern(x_deg, y_deg, z_deg, usage, el);

    return qd_python_arrayant2dict(ant);
}
