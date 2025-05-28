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
# ARRAYANT_CALC_DIRECTIVITY
Calculates the directivity (in dBi) of array antenna elements

## Description:
Directivity is a parameter of an antenna or which measures the degree to which the radiation emitted
is concentrated in a single direction. It is the ratio of the radiation intensity in a given direction
from the antenna to the radiation intensity averaged over all directions. Therefore, the directivity
of a hypothetical isotropic radiator is 1, or 0 dBi.<br>

## Usage:

```
% Input as struct
directivity = quadriga_lib.arrayant_calc_directivity(arrayant);
directivity = quadriga_lib.arrayant_calc_directivity(arrayant, i_element);

% Separate inputs
directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid);

directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid, i_element);
```

## Examples:
```
% Generate dipole antenna
ant = quadriga_lib.arrayant_generate('dipole');

% Calculate directivity
directivity = quadriga_lib.arrayant_calc_directivity(ant);
```

## Input arguments for struct mode:
- **`arrayant`** [1]<br>
  Struct containing a array antenna pattern with at least the following fields:
  `e_theta_re`     | Real part of e-theta field component             | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | Imaginary part of e-theta field component        | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | Real part of e-phi field component               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | Imaginary part of e-phi field component          | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation]`

- **`i_element`** [2] (optional)<br>
  Element index, 1-based. If not provided or empty, the directivity is calculated for all elements in the
  array antenna. <br>Size: `[n_out]` or empty<br>

## Input arguments for separate inputs:
- **`e_theta_re`** [1]<br>
  Real part of e-theta field component, Size: `[n_elevation, n_azimuth, n_elements]`

- **`e_theta_im`** [2]<br>
  Imaginary part of e-theta field component, Size: `[n_elevation, n_azimuth, n_elements]`

- **`e_phi_re`** [3]<br>
  Real part of e-phi field component, Size: `[n_elevation, n_azimuth, n_elements]`

- **`e_phi_im`** [4]<br>
  Imaginary part of e-phi field component, Size: `[n_elevation, n_azimuth, n_elements]`

- **`azimuth_grid`** [5]<br>
  Azimuth angles in [rad] -pi to pi, sorted, Size: `[n_azimuth]`

- **`elevation_grid`** [6]<br>
  Elevation angles in [rad], -pi/2 to pi/2, sorted, Size: `[n_elevation]`

- **`i_element`** [7] (optional)<br>
  Element index, 1-based. If not provided or empty, the directivity is calculated for all elements in the
  array antenna. <br>Size: `[n_out]` or empty<br>

## Output Argument:
- **`directivity`**<br>
  Directivity of the antenna pattern in dBi, double precision, <br>Size: `[n_out]` or `[n_elements]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (!(nrhs == 1 || nrhs == 2 || nrhs == 6 || nrhs == 7))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    if (nlhs == 0)
        return;

    // Assemble array antenna object
    auto ant = quadriga_lib::arrayant<double>();
    if (nrhs <= 2) // Struct
    {
        ant.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_re"));
        ant.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_im"));
        ant.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_re"));
        ant.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_im"));
        ant.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "azimuth_grid"));
        ant.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "elevation_grid"));
    }
    else // Separate
    {
        ant.e_theta_re = qd_mex_get_double_Cube(prhs[0]);
        ant.e_theta_im = qd_mex_get_double_Cube(prhs[1]);
        ant.e_phi_re = qd_mex_get_double_Cube(prhs[2]);
        ant.e_phi_im = qd_mex_get_double_Cube(prhs[3]);
        ant.azimuth_grid = qd_mex_get_double_Col(prhs[4]);
        ant.elevation_grid = qd_mex_get_double_Col(prhs[5]);
    }

    arma::uvec element_ind;
    if (nrhs == 2)
        element_ind = qd_mex_typecast_Col<arma::uword>(prhs[1], "i_element") - 1;
    else if (nrhs == 7)
        element_ind = qd_mex_typecast_Col<arma::uword>(prhs[6], "i_element") - 1;
    else
        element_ind = arma::regspace<arma::uvec>(0, ant.e_theta_re.n_slices - 1);

    arma::vec directivity;
    plhs[0] = qd_mex_init_output(&directivity, element_ind.n_elem);

    auto *p_directivity = directivity.memptr();
    for (auto el : element_ind)
        CALL_QD(*p_directivity++ = ant.calc_directivity_dBi(el));
}
