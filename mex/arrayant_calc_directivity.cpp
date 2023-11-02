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
# ARRAYANT_CALC_DIRECTIVITY
Calculates the directivity (in dBi) of array antenna elements

## Description:
Directivity is a parameter of an antenna or which measures the degree to which the radiation emitted 
is concentrated in a single direction. It is the ratio of the radiation intensity in a given direction 
from the antenna to the radiation intensity averaged over all directions. Therefore, the directivity 
of a hypothetical isotropic radiator is 1, or 0 dBi. [Wikipedia]<br>

## Usage:

```
directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid);

directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid, i_element);
```

## Examples:

```
% Generate dipole antenna
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid] = ...
    <a href="#arrayant_generate">quadriga_lib.arrayant_generate</a>('dipole');

% Calculate directivity 
directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid);
```

## Input Arguments:
- **`e_theta_re`**<br>
  Real part of the e-theta component (vertical component) of the far field of each antenna element in 
  the array antenna. Single or double precision. <br>Size: `[n_elevation, n_azimuth, n_elements]`
  
- **`e_theta_im`**<br>
  Imaginary part of the e-theta component of the electric field. Single or double precision, 
  <br>Size: `[n_elevation, n_azimuth, n_elements]`<br>
  
- **`e_phi_re`**<br>
  Real part of the e-phi component (horizontal component) of the far field of each antenna element in 
  the array antenna. Single or double precision, <br>Size: `[n_elevation, n_azimuth, n_elements]`<br>
  
- **`e_phi_im`**<br>
  Imaginary part of the e-phi component of the electric field. Single or double precision, 
  <br>Size: `[n_elevation, n_azimuth, n_elements]`<br>
  
- **`azimuth_grid`**<br>
  Azimuth angles (theta) in [rad] were samples of the field patterns are provided. Values must be between 
  -pi and pi, sorted in ascending order. Single or double precision, <br>Size: `[n_azimuth]`<br>
  
- **`elevation_grid`**<br>
  Elevation angles (phi) in [rad] where samples of the field patterns are provided. Values must be between 
  -pi/2 and pi/2, sorted in ascending order. Single or double precision, <br>Size: `[n_elevation]`<br>
  
- **`i_element`** (optional)<br> 
  Element index. If ot is not provided or empty, the directivity is calculated for all elements in the 
  array antenna. <br>Size: `[n_out]` or empty<br>

## Output Argument:
- **`directivity`**<br>
  Directivity of the antenna pattern in dBi, double precision, <br>Size: `[n_out]` or `[n_elements]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - e_theta_re      Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
    //  1 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
    //  2 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
    //  3 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
    //  4 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
    //  5 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
    //  6 - i_element       Element index, scalar, 1-based (optional, default: 1)

    // Output:
    //  0 - directivity     Directivity of the antenna pattern in dBi

    if (nrhs < 6) 
        mexErrMsgIdAndTxt("quadriga_lib:calc_directivity:IO_error", "Need at least 6 inputs.");

    if (nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:calc_directivity:IO_error", "Can have at most 7 inputs.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:calc_directivity:IO_error", "Wrong number of output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:calc_directivity:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 1; i < 6; ++i)
        if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
            mexErrMsgIdAndTxt("quadriga_lib:calc_directivity:wrong_type", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    // Create arrayant object and validate the input
    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    if (use_single)
        arrayant_single.e_theta_re = qd_mex_reinterpret_Cube<float>(prhs[0]),
        arrayant_single.e_theta_im = qd_mex_reinterpret_Cube<float>(prhs[1]),
        arrayant_single.e_phi_re = qd_mex_reinterpret_Cube<float>(prhs[2]),
        arrayant_single.e_phi_im = qd_mex_reinterpret_Cube<float>(prhs[3]),
        arrayant_single.azimuth_grid = qd_mex_reinterpret_Col<float>(prhs[4]),
        arrayant_single.elevation_grid = qd_mex_reinterpret_Col<float>(prhs[5]);
    else
        arrayant_double.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[0]),
        arrayant_double.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[1]),
        arrayant_double.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[2]),
        arrayant_double.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[3]),
        arrayant_double.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[4]),
        arrayant_double.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[5]);

    unsigned long long n_element = (use_single) ? arrayant_single.n_elements() : arrayant_double.n_elements();

    // Validate the data integrity
    std::string error_message = use_single ? arrayant_single.validate() : arrayant_double.validate();
    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:calc_directivity:IO_error", error_message.c_str());

    // Read i_element
    arma::u32_vec i_element;
    if (nrhs < 7)
    {
        i_element.set_size(n_element);
        auto p_element = i_element.memptr();
        for (unsigned i = 0; i < (unsigned)n_element; ++i)
            p_element[i] = i + 1; // 1-based
    }
    else
        i_element = qd_mex_typecast_Col<unsigned>(prhs[6], "i_element");

    // Generate output
    if (nlhs > 0)
    {
        arma::vec directivity;
        plhs[0] = qd_mex_init_output(&directivity, i_element.n_elem);

        // Call library function
        try
        {
            auto p_directivity = directivity.memptr();
            for (auto &el : i_element)
            {
                if (use_single)
                    *p_directivity++ = (double)arrayant_single.calc_directivity_dBi(el - 1);
                else
                    *p_directivity++ = arrayant_double.calc_directivity_dBi(el - 1);
            }
        }
        catch (const std::invalid_argument &ex)
        {
            mexErrMsgIdAndTxt("quadriga_lib:calc_directivity:unknown_error", ex.what());
        }
        catch (...)
        {
            mexErrMsgIdAndTxt("quadriga_lib:calc_directivity:unknown_error", "Unknown failure occurred. Possible memory corruption!");
        }
    }
}
