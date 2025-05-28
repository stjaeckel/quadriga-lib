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
# ARRAYANT_COPY_ELEMENT
Create copies of array antenna elements

## Usage:

```
arrayant_out = quadriga_lib.arrayant_copy_element(arrayant_in, source_element, dest_element);
```

## Input Arguments:
- **`arrayant_in`** [1] (required)<br>
  Struct containing the arrayant data with the following fields:
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
  Index of the source elements (1-based), scalar or vector

- **`dest_element`** [3] (optional)<br>
  Index of the destination elements (1-based), either as a vector or as a scalar. If `source_element`
  is also a vector, `dest_element` must have the same length.

## Output Arguments:
- **`arrayant_out`**<br>
  Struct containing the arrayant data with the following fields:
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs == 0)
        return;

    // Assemble array antenna object (copy input data)
    auto ant = quadriga_lib::arrayant<double>();
    ant.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_re"), true);
    ant.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_im"), true);
    ant.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_re"), true);
    ant.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_im"), true);
    ant.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "azimuth_grid"), true);
    ant.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "elevation_grid"), true);
    if (qd_mex_has_field(prhs[0], "element_pos"))
        ant.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "element_pos"), true);
    if (qd_mex_has_field(prhs[0], "coupling_re"))
        ant.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_re"), true);
    if (qd_mex_has_field(prhs[0], "coupling_im"))
        ant.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_im"), true);
    if (qd_mex_has_field(prhs[0], "center_frequency"))
        ant.center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(prhs[0], "center_frequency"));
    if (qd_mex_has_field(prhs[0], "name"))
        ant.name = qd_mex_get_string(qd_mex_get_field(prhs[0], "name"));

    arma::uvec source = qd_mex_typecast_Col<arma::uword>(prhs[1]) - 1;
    arma::uvec dest = qd_mex_typecast_Col<arma::uword>(prhs[2]) - 1;

    if (source.n_elem == 0 || dest.n_elem == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Copy element: source and dest cannot be empty.");

    if (source.n_elem == 1)
        CALL_QD(ant.copy_element(source.at(0), dest));
    else if (source.n_elem == dest.n_elem)
    {
        for (arma::uword i = 0; i < source.n_elem; ++i)
            CALL_QD(ant.copy_element(source.at(i), dest.at(i)));
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Copy element: when copying multiple elements, source and dest must be of same length.");

    if (nlhs == 1) // Output as struct
    {
        std::vector<std::string> fields = {"e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im",
                                           "azimuth_grid", "elevation_grid", "element_pos",
                                           "coupling_re", "coupling_im", "center_freq", "name"};

        plhs[0] = qd_mex_make_struct(fields);
        qd_mex_set_field(plhs[0], fields[0], qd_mex_copy2matlab(&ant.e_theta_re));
        qd_mex_set_field(plhs[0], fields[1], qd_mex_copy2matlab(&ant.e_theta_im));
        qd_mex_set_field(plhs[0], fields[2], qd_mex_copy2matlab(&ant.e_phi_re));
        qd_mex_set_field(plhs[0], fields[3], qd_mex_copy2matlab(&ant.e_phi_im));
        qd_mex_set_field(plhs[0], fields[4], qd_mex_copy2matlab(&ant.azimuth_grid, true));
        qd_mex_set_field(plhs[0], fields[5], qd_mex_copy2matlab(&ant.elevation_grid, true));
        qd_mex_set_field(plhs[0], fields[6], qd_mex_copy2matlab(&ant.element_pos));
        qd_mex_set_field(plhs[0], fields[7], qd_mex_copy2matlab(&ant.coupling_re));
        qd_mex_set_field(plhs[0], fields[8], qd_mex_copy2matlab(&ant.coupling_im));
        qd_mex_set_field(plhs[0], fields[9], qd_mex_copy2matlab(&ant.center_frequency));
        qd_mex_set_field(plhs[0], fields[10], mxCreateString(ant.name.c_str()));
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
}