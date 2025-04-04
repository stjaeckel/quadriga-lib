// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re,
   coupling_im, center_freq, name, layout ] = quadriga_lib.arrayant_qdant_read( fn, id, use_single );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the QDANT file, string

- **`id`** (optional)<br>
  ID of the antenna to be read from the file, optional, Default: Read first

- **`use_single`** (optional)<br>
  Indicator if results should be returned in single precision, default = 0, returned in double precision

## Output Arguments:
- **Antenna data:** (outputs 1-11, single or double)
  `e_theta_re`     | Real part of e-theta field component                  | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | Imaginary part of e-theta field component             | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | Real part of e-phi field component                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | Imaginary part of e-phi field component               | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted             | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements]` or `[]`
  `coupling_re`    | Real part of coupling matrix, optional                | Size: `[n_elements, n_ports]` or `[]`
  `coupling_im`    | Imaginary part of coupling matrix, optional           | Size: `[n_elements, n_ports]` or `[]`
  `center_freq`    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
  `name`           | Name of the array antenna object                      | String

- **`layout`**<br>
  Layout of multiple array antennas. Contain element ids that are present in the file
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the QDANT file
    //  1 - id              ID of the antenna to be read from the file (optional, default: 1)
    //  2 - use_single      Indicator if results should be returned in single precision (optional, default: 0, double)

    // Outputs:
    //  0 - e_theta_re      Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
    //  1 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
    //  2 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
    //  3 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
    //  4 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
    //  5 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
    //  6 - element_pos     Element positions                                               Size [3, n_elements]
    //  7 - coupling_re     Coupling matrix, real part                                      Size [n_elements, n_ports]
    //  8 - coupling_im     Coupling matrix, imaginary part                                 Size [n_elements, n_ports]
    //  9 - center_frequency   Center frequency in [Hz]                                     Scalar
    // 10 - name            Name of the array antenna object, string
    // 11 - layout          Layout of multiple array antennas (optional), uint32            Matrix

    // Number of in and outputs
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "Filename not given.");

    if (nlhs > 12)
        mexErrMsgIdAndTxt("quadriga_lib:qdant_read:no_input", "Too many output arguments.");

    // Read file name
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "Input 'fn' must be a string.");

    auto mx_fn = mxArrayToString(prhs[0]);
    std::string fn = std::string(mx_fn);
    mxFree(mx_fn);

    // Read scalar variables
    unsigned id = nrhs < 2 ? 1 : qd_mex_get_scalar<unsigned>(prhs[1], "id");
    bool use_single = nrhs < 3 ? false : qd_mex_get_scalar<bool>(prhs[2], "use_single");

    // Read from file
    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    arma::Mat<unsigned> layout;

    try
    {
        if (use_single)
            arrayant_single = quadriga_lib::qdant_read<float>(fn, id, &layout);
        else
            arrayant_double = quadriga_lib::qdant_read<double>(fn, id, &layout);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:qdant_read:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:qdant_read:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    if (use_single)
    {
        if (nlhs > 0)
            plhs[0] = qd_mex_copy2matlab(&arrayant_single.e_theta_re);
        if (nlhs > 1)
            plhs[1] = qd_mex_copy2matlab(&arrayant_single.e_theta_im);
        if (nlhs > 2)
            plhs[2] = qd_mex_copy2matlab(&arrayant_single.e_phi_re);
        if (nlhs > 3)
            plhs[3] = qd_mex_copy2matlab(&arrayant_single.e_phi_im);
        if (nlhs > 4)
            plhs[4] = qd_mex_copy2matlab(&arrayant_single.azimuth_grid, true);
        if (nlhs > 5)
            plhs[5] = qd_mex_copy2matlab(&arrayant_single.elevation_grid, true);
        if (nlhs > 6)
            plhs[6] = qd_mex_copy2matlab(&arrayant_single.element_pos);
        if (nlhs > 7)
            plhs[7] = qd_mex_copy2matlab(&arrayant_single.coupling_re);
        if (nlhs > 8)
            plhs[8] = qd_mex_copy2matlab(&arrayant_single.coupling_im);
        if (nlhs > 9)
            plhs[9] = qd_mex_copy2matlab(&arrayant_single.center_frequency);
    }
    else
    {
        if (nlhs > 0)
            plhs[0] = qd_mex_copy2matlab(&arrayant_double.e_theta_re);
        if (nlhs > 1)
            plhs[1] = qd_mex_copy2matlab(&arrayant_double.e_theta_im);
        if (nlhs > 2)
            plhs[2] = qd_mex_copy2matlab(&arrayant_double.e_phi_re);
        if (nlhs > 3)
            plhs[3] = qd_mex_copy2matlab(&arrayant_double.e_phi_im);
        if (nlhs > 4)
            plhs[4] = qd_mex_copy2matlab(&arrayant_double.azimuth_grid, true);
        if (nlhs > 5)
            plhs[5] = qd_mex_copy2matlab(&arrayant_double.elevation_grid, true);
        if (nlhs > 6)
            plhs[6] = qd_mex_copy2matlab(&arrayant_double.element_pos);
        if (nlhs > 7)
            plhs[7] = qd_mex_copy2matlab(&arrayant_double.coupling_re);
        if (nlhs > 8)
            plhs[8] = qd_mex_copy2matlab(&arrayant_double.coupling_im);
        if (nlhs > 9)
            plhs[9] = qd_mex_copy2matlab(&arrayant_double.center_frequency);
    }

    if (nlhs > 10)
        plhs[10] = use_single ? mxCreateString(arrayant_single.name.c_str()) : mxCreateString(arrayant_double.name.c_str());

    if (nlhs == 12)
        plhs[11] = qd_mex_copy2matlab(&layout);
}
