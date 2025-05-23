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
# ARRAYANT_EXPORT_OBJ_FILE
Creates a Wavefront OBJ file for visualizing the shape of the antenna pattern

## Usage:

```
quadriga_lib.arrayant_export_obj_file( fn, arrayant, directivity_range, colormap,
                object_radius, icosphere_n_div, i_element )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the OBJ file, string

- **`arrayant`**<br>
  Dictionary containing array antenna data with at least the following keys:
  `e_theta_re`     | Real part of e-theta field component             | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_theta_im`     | Imaginary part of e-theta field component        | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_phi_re`       | Real part of e-phi field component               | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_phi_im`       | Imaginary part of e-phi field component          | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions, optional      | Size: `[3, n_elements]`
  `name`           | Name of the array antenna object                 | String

- **`directivity_range`**<br>
  Directivity range of the antenna pattern visualization in dB

- **`colormap`**<br>
  Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
  'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', Optional, default = 'jet'

- **`object_radius`**<br>
  Radius in meters of the exported object

- **`icosphere_n_div`**<br>
  Map pattern to an Icosphere with given number of subdivisions

- **`i_element`**<br>
  Antenna element indices, 0-based, empty = export all
MD!*/

void arrayant_export_obj_file(const std::string fn,
                              const py::dict arrayant,
                              double directivity_range,
                              const std::string colormap,
                              double object_radius,
                              unsigned icosphere_n_div,
                              const py::array_t<arma::uword> element)
{
    auto ant = quadriga_lib::arrayant<double>();

    ant.e_theta_re = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_re"], true);
    ant.e_theta_im = qd_python_numpy2arma_Cube<double>(arrayant["e_theta_im"], true);
    ant.e_phi_re = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_re"], true);
    ant.e_phi_im = qd_python_numpy2arma_Cube<double>(arrayant["e_phi_im"], true);
    ant.azimuth_grid = qd_python_numpy2arma_Col<double>(arrayant["azimuth_grid"], true);
    ant.elevation_grid = qd_python_numpy2arma_Col<double>(arrayant["elevation_grid"], true);
    ant.element_pos = qd_python_numpy2arma_Mat<double>(arrayant["element_pos"], true);

    if (arrayant.contains("name"))
        ant.name = arrayant["name"].cast<std::string>();

    arma::uvec i_element_a = qd_python_numpy2arma_Col(element);
    ant.export_obj_file(fn, directivity_range, colormap, object_radius, icosphere_n_div, i_element_a);
}