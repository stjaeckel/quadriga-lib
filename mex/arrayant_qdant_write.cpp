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
#include "mex_helper_functions.cpp"

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
id_in_file = quadriga_lib.arrayant_qdant_write( fn, e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name, id, layout);
```

## Caveat:
- Inputs can be single or double precision, but type must match for all inputs
- Multiple array antennas can be stored in the same file using the `id` parameter.
- If writing to an exisiting file without specifying an `id`, the data gests appended at the end.  
  The output `id_in_file` identifies the location inside the file.
- An optional storage `layout` can be provided to organize data inside the file.

## Input Arguments:
- **`fn`**<br>
  Filename of the QDANT file, string

- **Antenna data:** (inputs 2-12, single or double)
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

- **`id`** (optional)<br>
  ID of the antenna to be written to the file, optional, Default: Max-ID in existing file + 1

- **`layout`** (optional)<br>
  Layout of multiple array antennas. Must only contain element ids that are present in the file. optional

## Output Argument:
- **`id_in_file`**<br>
  ID of the antenna in the file after writing
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the QDANT file, string
    //  1 - e_theta_re      Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
    //  2 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
    //  3 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
    //  4 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
    //  5 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
    //  6 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
    //  7 - element_pos     Element positions                                               Size [3, n_elements]
    //  8 - coupling_re     Coupling matrix, real part                                      Size [n_elements, n_ports]
    //  9 - coupling_im     Coupling matrix, imaginary part                                 Size [n_elements, n_ports]
    // 10 - center_frequency   Center frequency in [Hz]                                     Scalar
    // 11 - name            Name of the array antenna object, string
    // 12 - id              ID of the antenna to be written to the file, optional, Default: Max-ID in existing file + 1
    // 13 - layout          Layout of multiple array antennas, optional

    // Output:
    //  0 - id_in_file      ID of the antenna in the file

    // Number of in and outputs
    if (nrhs < 7)
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "Need at least 7 inputs.");

    if (nrhs > 14)
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "Cannot have more than 14 inputs.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "Cannot have more than 1 output.");

    // Read file name
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "Input 'fn' must be a string.");

    auto mx_fn = mxArrayToString(prhs[0]);
    std::string fn = std::string(mx_fn);
    mxFree(mx_fn);

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[1]) || mxIsDouble(prhs[1]))
        use_single = mxIsSingle(prhs[1]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 2; i < 10; ++i)
        if (nrhs > i)
            if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
                mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "All floating-point inputs must have the same type: 'single' or 'double' precision.");

    // Create arrayant object and validate the input
    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    if (use_single)
    {
        arrayant_single.e_theta_re = qd_mex_reinterpret_Cube<float>(prhs[1]);
        arrayant_single.e_theta_im = qd_mex_reinterpret_Cube<float>(prhs[2]);
        arrayant_single.e_phi_re = qd_mex_reinterpret_Cube<float>(prhs[3]);
        arrayant_single.e_phi_im = qd_mex_reinterpret_Cube<float>(prhs[4]);
        arrayant_single.azimuth_grid = qd_mex_reinterpret_Col<float>(prhs[5]);
        arrayant_single.elevation_grid = qd_mex_reinterpret_Col<float>(prhs[6]);
        if (nrhs > 7)
            arrayant_single.element_pos = qd_mex_reinterpret_Mat<float>(prhs[7]);
        if (nrhs > 8)
            arrayant_single.coupling_re = qd_mex_reinterpret_Mat<float>(prhs[8]);
        if (nrhs > 9)
            arrayant_single.coupling_im = qd_mex_reinterpret_Mat<float>(prhs[9]);
        if (nrhs > 10)
            arrayant_single.center_frequency = qd_mex_get_scalar<float>(prhs[10], "center_frequency");
        arrayant_single.read_only = true;
    }
    else
    {
        arrayant_double.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[1]);
        arrayant_double.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[2]);
        arrayant_double.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[3]);
        arrayant_double.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[4]);
        arrayant_double.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[5]);
        arrayant_double.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[6]);
        if (nrhs > 7)
            arrayant_double.element_pos = qd_mex_reinterpret_Mat<double>(prhs[7]);
        if (nrhs > 8)
            arrayant_double.coupling_re = qd_mex_reinterpret_Mat<double>(prhs[8]);
        if (nrhs > 9)
            arrayant_double.coupling_im = qd_mex_reinterpret_Mat<double>(prhs[9]);
        if (nrhs > 10)
            arrayant_double.center_frequency = qd_mex_get_scalar<double>(prhs[10], "center_frequency");
        arrayant_double.read_only = true;
    }
    
    // Read arrayant name
     std::string name = "QDArrayant";
    if (nrhs > 11 && !mxIsClass(prhs[11], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "Input 'name' must be a string.");
    else if (nrhs > 11)
    {
        auto mx_name = mxArrayToString(prhs[11]);
        name = std::string(mx_name);
        mxFree(mx_name);
    }
    if (use_single)
        arrayant_single.name = name;
    else
        arrayant_double.name = name;

    // Validate the data integrity
    std::string error_message = use_single ? arrayant_single.validate() : arrayant_double.validate();
    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", error_message.c_str());

    // Read id variable - indicate that ID is not given by passing a value 0
    unsigned id = nrhs < 13 ? 0 : qd_mex_get_scalar<unsigned>(prhs[12], "id", 0);

    // Read layout variable - if layout is not given, pass empty matrix
    arma::Mat<unsigned> layout;
    if (nrhs < 14 || mxGetNumberOfElements(prhs[13]) == 0)
        layout = arma::Mat<unsigned>();
    else
        layout = qd_mex_typecast_Mat<unsigned>(prhs[13], "layout");

    // Call library function
    try
    {
        if (use_single)
            id = arrayant_single.qdant_write(fn, id, layout);
        else
            id = arrayant_double.qdant_write(fn, id, layout);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Create output variable
    if (nlhs > 0)
    {
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT32_CLASS, mxREAL);
        unsigned *id_in_file = (unsigned *)mxGetData(plhs[0]);
        *id_in_file = id;
    }
}
