// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "quadriga_lib.hpp"

#include "python_helpers.cpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_EXPORT_OBJ_FILE
Creates a Wavefront OBJ file for visualizing the shape of the antenna pattern

## Usage:

```
quadriga_lib.arrayant_export_obj_file( fn, directivity_range, colormap, object_radius, icosphere_n_div,
    e_theta, e_phi, azimuth_grid, elevation_grid, element_pos, name, i_element )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the OBJ file, string

- **`directivity_range`**<br>
  Directivity range of the antenna pattern visualization in dB

- **`colormap`**<br>
  Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
  'copper', 'spring', 'cool', 'gray', 'autumn', 'summer'

- **`object_radius`**<br>
  Radius in meters of the exported object

- **`icosphere_n_div`**<br>
  Map pattern to an Icosphere with given number of subdivisions

- **`e_theta`**<br>
  e-theta field component, complex-valued, Size: `[n_elevation, n_azimuth, n_elements]`

- **`e_phi`**<br>
  e-phi field component, complex-valued, Size: `[n_elevation, n_azimuth, n_elements]`

- **`azimuth_grid`**<br>
  Azimuth angles in [rad] -pi to pi, sorted, Size: `[n_azimuth]`

- **`elevation_grid`**<br>
  Elevation angles in [rad], -pi/2 to pi/2, sorted, Size: `[n_elevation]`

- **`element_pos`**<br>
  Antenna element (x,y,z) positions, Size: `[3, n_elements]`

- **`name`**<br>
  Name of the array antenna object

- **`i_element`**<br>
  Antenna element indices, 0-based, empty = export all
MD!*/

void arrayant_export_obj_file(const std::string fn,
                              double directivity_range,
                              const std::string colormap,
                              double object_radius,
                              unsigned icosphere_n_div,
                              const pybind11::array_t<std::complex<double>> e_theta,
                              const pybind11::array_t<std::complex<double>> e_phi,
                              const pybind11::array_t<double> azimuth_grid,
                              const pybind11::array_t<double> elevation_grid,
                              const pybind11::array_t<double> element_pos,
                              const std::string name,
                              const pybind11::array_t<arma::uword> i_element)
{
    auto ant = quadriga_lib::arrayant<double>();
    ant.name = name;

    qd_python_complexNPArray_to_2Cubes(&e_theta, &ant.e_theta_re, &ant.e_theta_im);
    qd_python_complexNPArray_to_2Cubes(&e_phi, &ant.e_phi_re, &ant.e_phi_im);

    ant.azimuth_grid = qd_python_NPArray_to_Col(&azimuth_grid);
    ant.elevation_grid = qd_python_NPArray_to_Col(&elevation_grid);
    ant.element_pos = qd_python_NPArray_to_Mat(&element_pos);

    arma::uvec i_element_a = qd_python_NPArray_to_Col(&i_element);

    ant.export_obj_file(fn, directivity_range, colormap, object_radius, icosphere_n_div, i_element_a);
}