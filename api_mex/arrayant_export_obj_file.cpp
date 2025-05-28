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

#include "mex.h"
#include "quadriga_arrayant.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_EXPORT_OBJ_FILE
Creates a Wavefront OBJ file for visualizing the shape of the antenna pattern

## Usage:

```
quadriga_lib.arrayant_export_obj_file( fn, arrayant, directivity_range, colormap, ...
                object_radius, icosphere_n_div, i_element );
```

## Input Arguments:
- **`fn`** [1]<br>
  Filename of the OBJ file, string

- **`arrayant`** [2]<br>
  Struct containing the arrayant data with the following fields:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements]`
  `name`           | Name of the array antenna object, optional            | String

- **`directivity_range`** [3]<br>
  Directivity range of the antenna pattern visualization in dB

- **`colormap`** [4]<br>
  Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
  'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', Optional, default = 'jet'

- **`object_radius`** [5]<br>
  Radius in meters of the exported object

- **`icosphere_n_div`** [6]<br>
  Map pattern to an Icosphere with given number of subdivisions

- **`element`** [7]<br>
  Antenna element indices, 1-based, empty = export all
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    // Parse inputs
    std::string fn = qd_mex_get_string(prhs[0]);
    double directivity_range = (nrhs < 3) ? 30.0 : qd_mex_get_scalar<double>(prhs[2], "directivity_range", 30.0);
    std::string colormap = (nrhs < 4) ? "jet" : qd_mex_get_string(prhs[3]);
    double object_radius = (nrhs < 5) ? 1.0 : qd_mex_get_scalar<double>(prhs[4], "object_radius", 1.0);
    unsigned icosphere_n_div = (nrhs < 6) ? 4 : qd_mex_get_scalar<unsigned>(prhs[5], "icosphere_n_div", 4);
    const arma::uvec element_ind = (nrhs < 7) ? arma::uvec() : qd_mex_typecast_Col<arma::uword>(prhs[6]) - 1;

    // Assemble array antenna object (copy input data)
    auto ant = quadriga_lib::arrayant<double>();
    ant.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_theta_re"), true);
    ant.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_theta_im"), true);
    ant.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_phi_re"), true);
    ant.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_phi_im"), true);
    ant.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[1], "azimuth_grid"), true);
    ant.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[1], "elevation_grid"), true);

    if (qd_mex_has_field(prhs[1], "element_pos"))
        ant.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "element_pos"), true);

    if (qd_mex_has_field(prhs[1], "name"))
        ant.name = qd_mex_get_string(qd_mex_get_field(prhs[1], "name"));

    CALL_QD(ant.export_obj_file(fn, directivity_range, colormap, object_radius, icosphere_n_div, element_ind));

    double out = 1.0;
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&out);
}
