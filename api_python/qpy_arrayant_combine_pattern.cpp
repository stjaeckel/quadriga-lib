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

#include "quadriga_python_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_COMBINE_PATTERN
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
# Minimal example
arrayant_out = quadriga_lib.arrayant_combine_pattern(arrayant_in)

# Optional inputs: freq, azimuth_grid, elevation_grid
arrayant_out = quadriga_lib.arrayant_combine_pattern(arrayant_in, freq, azimuth_grid, elevation_grid)
```

## Input Arguments:
- **`arrayant_in`**<br>
  Dictionary containing the arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions                     | Size: `[3, n_elements]`
  `coupling_re`    | Coupling matrix, real part                            | Size: `[n_elements, n_ports]`
  `coupling_im`    | Coupling matrix, imaginary part                       | Size: `[n_elements, n_ports]`
  `center_freq`    | Center frequency in [Hz], optional                    | Scalar
  `name`           | Name of the array antenna object, optional            | String

- **`freq`** (optional)<br>
  An alternative value for the center frequency. Overwrites the value given in `arrayant_in`. If 
  neither `freq` not `arrayant_in["center_freq"]` are given, an error is thrown.alignas

- **`azimuth_grid`** (optional)<br>
  Alternative azimuth angles for the output in [rad], -pi to pi, sorted, Size: `[n_azimuth_out]`, 
  If not given, `arrayant_in["azimuth_grid"]` is used instead.

- **`elevation_grid`** (optional)<br>
  Alternative elevation angles for the output in [rad], -pi/2 to pi/2, sorted, Size: `[n_elevation_out]`, 
  If not given, `arrayant_in["elevation_grid"]` is used instead.

## Output Arguments:
- **`arrayant_out`**<br>
  Dictionary containing the arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation_out, n_azimuth_out, n_ports]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_azimuth_out, n_azimuth_out, n_ports]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_azimuth_out, n_azimuth_out, n_ports]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_azimuth_out, n_azimuth_out, n_ports]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth_out]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_azimuth_out]`
  `element_pos`    | Antenna element (x,y,z) positions, set to 0           | Size: `[3, n_ports]`
  `coupling_re`    | Coupling matrix, real part, identity matrix           | Size: `[n_ports, n_ports]`
  `coupling_im`    | Coupling matrix, imaginary part, zero matrix          | Size: `[n_ports, n_ports]`
  `center_freq`    | Center frequency in [Hz]                              | Scalar
  `name`           | Name of the array antenna object, same as input       | String
MD!*/

py::dict arrayant_combine_pattern(const py::dict &arrayant,                  // Input data
                                  double freq,                               // The center frequency in [Hz]
                                  const py::array_t<double> &azimuth_grid,   // Optional new azimuth_grid
                                  const py::array_t<double> &elevation_grid, // Optional new elevation_grid
                                  bool fast_access)                          // Enforces fast memory access
{
    // Assemble array antenna object
    auto ant = quadriga_lib::arrayant<double>();
    ant.e_theta_re = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_re"], true, fast_access);
    ant.e_theta_im = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_im"], true, fast_access);
    ant.e_phi_re = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_re"], true, fast_access);
    ant.e_phi_im = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_im"], true, fast_access);
    ant.azimuth_grid = qd_python_numpy2arma_Col<double>(arrayant["azimuth_grid"], true, fast_access);
    ant.elevation_grid = qd_python_numpy2arma_Col<double>(arrayant["elevation_grid"], true, fast_access);
    ant.element_pos = qd_python_numpy2arma_Mat<double>(arrayant["element_pos"], true, fast_access);
    ant.coupling_re = qd_python_numpy2arma_Cube<double>(arrayant["coupling_re"], true);
    ant.coupling_im = qd_python_numpy2arma_Cube<double>(arrayant["coupling_im"], true);

    if (freq > 0.0)
        ant.center_frequency = freq;
    else if (arrayant.contains("center_freq"))
        ant.center_frequency = arrayant["center_freq"].cast<double>();
    else
        throw std::invalid_argument("Center frequency is missing.");

    if (arrayant.contains("name"))
        ant.name = arrayant["name"].cast<std::string>();

    // Parse grid
    const auto az = qd_python_numpy2arma_Col(azimuth_grid, true);
    const auto el = qd_python_numpy2arma_Col(elevation_grid, true);

    // Call member function
    auto arrayant_out = ant.combine_pattern(&az, &el);

    // Return to python
    py::dict output;
    output["e_theta_re"] = qd_python_copy2numpy(arrayant_out.e_theta_re);
    output["e_theta_im"] = qd_python_copy2numpy(arrayant_out.e_theta_im);
    output["e_phi_re"] = qd_python_copy2numpy(arrayant_out.e_phi_re);
    output["e_phi_im"] = qd_python_copy2numpy(arrayant_out.e_phi_im);
    output["azimuth_grid"] = qd_python_copy2numpy(arrayant_out.azimuth_grid);
    output["elevation_grid"] = qd_python_copy2numpy(arrayant_out.elevation_grid);
    output["element_pos"] = qd_python_copy2numpy(arrayant_out.element_pos);
    output["coupling_re"] = qd_python_copy2numpy(arrayant_out.coupling_re);
    output["coupling_im"] = qd_python_copy2numpy(arrayant_out.coupling_im);
    output["center_freq"] = freq;
    output["name"] = arrayant_out.name;
    return output;
}
