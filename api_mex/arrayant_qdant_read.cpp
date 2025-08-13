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
# ARRAYANT_QDANT_READ
Reads array antenna data from QDANT files

## Description:
The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna pattern
data in XML. This function reads pattern data from the specified file.

## Usage:

```
% Read as struct
[ ant, layout ] = quadriga_lib.arrayant_qdant_read( fn, id );

% Read as separate fields
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re,
   coupling_im, center_freq, name, layout ] = quadriga_lib.arrayant_qdant_read( fn, id );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the QDANT file, string

- **`id`** (optional)<br>
  ID of the antenna to be read from the file, optional, Default: Read first

## Output Arguments:
- **`ant`**<br>
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
  `center_freq`    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
  `name`           | Name of the array antenna object                      | String

- **`layout`**<br>
  Layout of multiple array antennas. Contain element ids that are present in the file

## See also:
- [[arrayant_qdant_write]] (for writing QDANT data)
- QuaDRiGa Array Antenna Exchange Format  (<a href="formats.html#6cab4884">QDANT</a>)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs == 0)
        return;

    std::string fn = qd_mex_get_string(prhs[0]);
    unsigned id = (nrhs < 2) ? 1 : qd_mex_get_scalar<unsigned>(prhs[1], "id");

    // Read data from file
    quadriga_lib::arrayant<double> arrayant;
    arma::Mat<unsigned> layout;
    CALL_QD(arrayant = quadriga_lib::qdant_read<double>(fn, id, &layout));

    if (nlhs == 1 || nlhs == 2) // Output as struct
    {
        std::vector<std::string> fields = {"e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im",
                                           "azimuth_grid", "elevation_grid", "element_pos",
                                           "coupling_re", "coupling_im", "center_freq", "name"};

        plhs[0] = qd_mex_make_struct(fields);
        qd_mex_set_field(plhs[0], fields[0], qd_mex_copy2matlab(&arrayant.e_theta_re));
        qd_mex_set_field(plhs[0], fields[1], qd_mex_copy2matlab(&arrayant.e_theta_im));
        qd_mex_set_field(plhs[0], fields[2], qd_mex_copy2matlab(&arrayant.e_phi_re));
        qd_mex_set_field(plhs[0], fields[3], qd_mex_copy2matlab(&arrayant.e_phi_im));
        qd_mex_set_field(plhs[0], fields[4], qd_mex_copy2matlab(&arrayant.azimuth_grid, true));
        qd_mex_set_field(plhs[0], fields[5], qd_mex_copy2matlab(&arrayant.elevation_grid, true));
        qd_mex_set_field(plhs[0], fields[6], qd_mex_copy2matlab(&arrayant.element_pos));
        qd_mex_set_field(plhs[0], fields[7], qd_mex_copy2matlab(&arrayant.coupling_re));
        qd_mex_set_field(plhs[0], fields[8], qd_mex_copy2matlab(&arrayant.coupling_im));
        qd_mex_set_field(plhs[0], fields[9], qd_mex_copy2matlab(&arrayant.center_frequency));
        qd_mex_set_field(plhs[0], fields[10], mxCreateString(arrayant.name.c_str()));

        if (nlhs == 2)
            plhs[1] = qd_mex_copy2matlab(&layout);
    }
    else if (nlhs == 11 || nlhs == 12) // Separate outputs
    {
        plhs[0] = qd_mex_copy2matlab(&arrayant.e_theta_re);
        plhs[1] = qd_mex_copy2matlab(&arrayant.e_theta_im);
        plhs[2] = qd_mex_copy2matlab(&arrayant.e_phi_re);
        plhs[3] = qd_mex_copy2matlab(&arrayant.e_phi_im);
        plhs[4] = qd_mex_copy2matlab(&arrayant.azimuth_grid, true);
        plhs[5] = qd_mex_copy2matlab(&arrayant.elevation_grid, true);
        plhs[6] = qd_mex_copy2matlab(&arrayant.element_pos);
        plhs[7] = qd_mex_copy2matlab(&arrayant.coupling_re);
        plhs[8] = qd_mex_copy2matlab(&arrayant.coupling_im);
        plhs[9] = qd_mex_copy2matlab(&arrayant.center_frequency);
        plhs[10] = mxCreateString(arrayant.name.c_str());

        if (nlhs == 12)
            plhs[11] = qd_mex_copy2matlab(&layout);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
}
