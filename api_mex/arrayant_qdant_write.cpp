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
# ARRAYANT_QDANT_WRITE
Writes array antenna data to QDANT files

## Description:
The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna pattern
data in XML. This function writes pattern data to the specified file.

## Usage:

```
% Arrayant as struct
id_in_file = quadriga_lib.arrayant_qdant_write( fn, arrayant, id, layout);

% Arrayant as separate inputs
id_in_file = quadriga_lib.arrayant_qdant_write( fn, [], id, layout, e_theta_re, e_theta_im, e_phi_re,
    e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name);
```

## Caveat:
- Multiple array antennas can be stored in the same file using the `id` parameter.
- If writing to an exisiting file without specifying an `id`, the data gests appended at the end.
  The output `id_in_file` identifies the location inside the file.
- An optional storage `layout` can be provided to organize data inside the file.

## Input Arguments:
- **`fn`** [1]<br>
  Filename of the QDANT file, string

- **`arrayant`** [2] (optional)<br>
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
  `center_freq`    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
  `name`           | Name of the array antenna object, optional            | String
  If an empty array is passed, array antenna data is provided as separate inputs (Inputs 5-15)<br><br>

- **`id`** [3] (optional)<br>
  ID of the antenna to be written to the file, optional, Default: Max-ID in existing file + 1

- **`layout`** [4] (optional)<br>
  Layout of multiple array antennas. Must only contain element ids that are present in the file. optional

## Output Argument:
- **`id_in_file`**<br>
  ID of the antenna in the file after writing
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::string fn = qd_mex_get_string(prhs[0]);
    unsigned id = (nrhs < 3) ? 0 : qd_mex_get_scalar<unsigned>(prhs[2], "id", 0);
    arma::u32_mat layout = (nrhs < 4) ? arma::u32_mat() : qd_mex_typecast_Mat<unsigned>(prhs[3], "layout");

    // Assemble array antenna object
    auto ant = quadriga_lib::arrayant<double>();
    if (nrhs <= 4) // Struct
    {
        ant.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_theta_re"));
        ant.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_theta_im"));
        ant.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_phi_re"));
        ant.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_phi_im"));
        ant.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[1], "azimuth_grid"));
        ant.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[1], "elevation_grid"));

        if (qd_mex_has_field(prhs[1], "element_pos"))
            ant.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "element_pos"));

        if (qd_mex_has_field(prhs[1], "coupling_re"))
            ant.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "coupling_re"));

        if (qd_mex_has_field(prhs[1], "coupling_im"))
            ant.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "coupling_im"));

        if (qd_mex_has_field(prhs[1], "center_freq"))
            ant.center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(prhs[1], "center_freq"));

        if (qd_mex_has_field(prhs[1], "name"))
            ant.name = qd_mex_get_string(qd_mex_get_field(prhs[1], "name"));
    }
    else if (nrhs >= 10) // Separate
    {
        ant.e_theta_re = qd_mex_get_double_Cube(prhs[4]);
        ant.e_theta_im = qd_mex_get_double_Cube(prhs[5]);
        ant.e_phi_re = qd_mex_get_double_Cube(prhs[6]);
        ant.e_phi_im = qd_mex_get_double_Cube(prhs[7]);
        ant.azimuth_grid = qd_mex_get_double_Col(prhs[8]);
        ant.elevation_grid = qd_mex_get_double_Col(prhs[9]);

        if (nrhs > 10)
            ant.element_pos = qd_mex_get_double_Mat(prhs[10]);

        if (nrhs > 11)
            ant.coupling_re = qd_mex_get_double_Mat(prhs[11]);

        if (nrhs > 12)
            ant.coupling_im = qd_mex_get_double_Mat(prhs[12]);

        if (nrhs > 13)
            ant.center_frequency = qd_mex_get_scalar<double>(prhs[13]);

        if (nrhs > 14)
            ant.name = qd_mex_get_string(prhs[14]);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    // Write to file
    CALL_QD(id = ant.qdant_write(fn, id, layout));

    if (nlhs == 1)
        plhs[0] = qd_mex_copy2matlab(&id);
}
