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
# combine_pattern
Calculate effective radiation patterns for array antennas

## Description:
An array antenna consists of multiple individual elements. Each element occupies a specific position
relative to the array's phase-center, its local origin. Elements can also be inter-coupled,
represented by a coupling matrix. By integrating the element radiation patterns, their positions,
and the coupling weights, one can determine an effective radiation pattern observable by a receiver
in the antenna's far field. Leveraging these effective patterns is especially beneficial in antenna
design, beamforming applications such as in 5G systems, and in planning wireless communication
networks in complex environments like urban areas. This streamlined approach offers a significant
boost in computation speed when calculating MIMO channel coefficients, as it reduces the number of
necessary operations. The function `arrayant_combine_pattern` is designed to compute these effective
radiation patterns.

## Usage:
```
from quadriga_lib import arrayant

# Minimal example
arrayant_out = arrayant.combine_pattern(arrayant)

# Optional inputs: freq, azimuth_grid, elevation_grid
arrayant_out = arrayant.combine_pattern(arrayant, freq, azimuth_grid, elevation_grid)
```

## Input Arguments:
- **`arrayant`**<br>
  Dictionary containing the arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_elevation, n_azimuth, n_elements)`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions                     | Shape: `(3, n_elements)`
  `coupling_re`    | Coupling matrix, real part                            | Shape: `(n_elements, n_ports)`
  `coupling_im`    | Coupling matrix, imaginary part                       | Shape: `(n_elements, n_ports)`
  `center_freq`    | Center frequency in [Hz], optional                    | Scalar
  `name`           | Name of the array antenna object, optional            | String

- **`freq`** (optional)<br>
  An alternative value for the center frequency. Overwrites the value given in `arrayant_in`. If
  neither `freq` not `arrayant_in["center_freq")` are given, an error is thrown.

- **`azimuth_grid`** (optional)<br>
  Alternative azimuth angles for the output in [rad], -pi to pi, sorted, Shape: `(n_azimuth_out)`,
  If not given, `arrayant_in["azimuth_grid")` is used instead.

- **`elevation_grid`** (optional)<br>
  Alternative elevation angles for the output in [rad], -pi/2 to pi/2, sorted, Shape: `(n_elevation_out)`,
  If not given, `arrayant_in["elevation_grid")` is used instead.

## Output Arguments:
- **`arrayant_out`**<br>
  Dictionary containing the arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_elevation_out, n_azimuth_out, n_ports)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_azimuth_out, n_azimuth_out, n_ports)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_azimuth_out, n_azimuth_out, n_ports)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_azimuth_out, n_azimuth_out, n_ports)`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Shape: `(n_azimuth_out)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_azimuth_out)`
  `element_pos`    | Antenna element (x,y,z) positions, set to 0           | Shape: `(3, n_ports)`
  `coupling_re`    | Coupling matrix, real part, identity matrix           | Shape: `(n_ports, n_ports)`
  `coupling_im`    | Coupling matrix, imaginary part, zero matrix          | Shape: `(n_ports, n_ports)`
  `center_freq`    | Center frequency in [Hz]                              | Scalar
  `name`           | Name of the array antenna object, same as input       | String
MD!*/

py::dict arrayant_combine_pattern(const py::dict &arrayant,                  // Input data
                                  double freq,                               // The center frequency in [Hz]
                                  const py::array_t<double> &azimuth_grid,   // Optional alternative azimuth_grid
                                  const py::array_t<double> &elevation_grid) // Optional alternative elevation_grid)
{
    auto ant = qd_python_dict2arrayant(arrayant, true);

    if (freq > 0.0)
        ant.center_frequency = freq;

    const auto az = qd_python_numpy2arma_Col(azimuth_grid, true);
    const auto el = qd_python_numpy2arma_Col(elevation_grid, true);

    auto arrayant_out = ant.combine_pattern(&az, &el);
    arrayant_out.center_frequency = freq;

    return qd_python_arrayant2dict(arrayant_out);
}
