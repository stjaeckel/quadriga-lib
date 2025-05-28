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
# ARRAYANT_COPY_ELEMENT
Create copies of array antenna elements

## Usage:

```
arrayant_out = quadriga_lib.arrayant_copy_element(arrayant, source_element, dest_element);
```

## Input Arguments:
- **`arrayant`** [1] (required)<br>
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

- **`source_element`** [2] (required)<br>
  Index of the source elements (0-based), scalar or vector

- **`dest_element`** [3] (optional)<br>
  Index of the destination elements (0-based), either as a vector or as a scalar. If `source_element`
  is also a vector, `dest_element` must have the same length.

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

py::dict arrayant_copy_element(const py::dict &arrayant,
                               const py::array_t<arma::uword> &source_element,
                               const py::array_t<arma::uword> &dest_element)
{
    auto ant = qd_python_dict2arrayant(arrayant, false); // Copy
    const auto source = qd_python_numpy2arma_Col(source_element);
    const auto dest = qd_python_numpy2arma_Col(dest_element);

    if (source.n_elem == 1)
        ant.copy_element(source.at(0), dest);
    else if (source.n_elem == dest.n_elem)
    {
        for (arma::uword i = 0; i < source.n_elem; ++i)
            ant.copy_element(source.at(i), dest.at(i));
    }
    else
        throw std::invalid_argument("Copy element: when copying multiple elements, source and dest must be of same length.");

    return qd_python_arrayant2dict(ant);
}