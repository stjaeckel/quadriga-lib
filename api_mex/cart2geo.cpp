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
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# CART2GEO
Transform Cartesian (x,y,z) coordinates to Geographic (az, el, length) coordinates

## Description:
This function transforms Cartesian (x,y,z) coordinates to Geographic (azimuth, elevation, length)
coordinates. A geographic coordinate system is a three-dimensional reference system that locates
points on the surface of a sphere. A point has three coordinate values: azimuth, elevation and length
where azimuth and elevation measure angles. In the geographic coordinate system, the elevation angle
θ = 90° points to the zenith and θ = 0° points to the horizon.

## Usage:

```
[ azimuth, elevation, length ] = arrayant_lib.cart2geo( cart )
```

## Input Argument:
- **`cart`**<br>
  Cartesian coordinates (x,y,z)
  Single or double precision, Size: `[3, n_row, n_col]`

## Output Arguments:
- **`azimuth`**<br>
  Azimuth angles in [rad], values between -pi and pi.
  Single or double precision (same as input), Size `[n_row, n_col]`

- **`elevation`**<br>
  Elevation angles in [rad], values between -pi/2 and pi/2.
  Single or double precision (same as input), Size `[n_row, n_col]`

- **`length`**<br>
  Vector length, i.e. the distance from the origin to the point defined by x,y,z.
  Single or double precision (same as input), Size `[n_row, n_col]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Input:           cart            Cartesian coordinates,                  Size [3, n_row, n_col]
    // Outputs:         azimuth         Azimuth angles in [rad],                Size [n_row, n_col]
    //                  elevation       Elevation angles in [rad],              Size [n_row, n_col]
    //                  length          Length of the vector, optional,         Size [n_row, n_col]

    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:no_input", "Cartesian coordinates not given.");

    if (nrhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:no_input", "Too many input arguments.");

    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:no_output", "Too many output arguments.");

    if (mxIsSingle(prhs[0]))
    {
        arma::fcube cart = qd_mex_reinterpret_Cube<float>(prhs[0]);
        arma::fcube geo;
        CALL_QD(geo = quadriga_lib::cart2geo(cart));
        if (nlhs > 0)
        {
            arma::fmat sub_matrix(geo.slice_memptr(0), geo.n_rows, geo.n_cols, false, true);
            plhs[0] = qd_mex_copy2matlab(&sub_matrix);
        }
        if (nlhs > 1)
        {
            arma::fmat sub_matrix(geo.slice_memptr(1), geo.n_rows, geo.n_cols, false, true);
            plhs[1] = qd_mex_copy2matlab(&sub_matrix);
        }
        if (nlhs > 2)
        {
            arma::fmat sub_matrix(geo.slice_memptr(2), geo.n_rows, geo.n_cols, false, true);
            plhs[2] = qd_mex_copy2matlab(&sub_matrix);
        }
    }
    else
    {
        arma::cube cart = mxIsDouble(prhs[0]) ? qd_mex_reinterpret_Cube<double>(prhs[0]) : qd_mex_typecast_Cube<double>(prhs[0]);
        arma::cube geo;
        CALL_QD(geo = quadriga_lib::cart2geo(cart));
        if (nlhs > 0)
        {
            arma::mat sub_matrix(geo.slice_memptr(0), geo.n_rows, geo.n_cols, false, true);
            plhs[0] = qd_mex_copy2matlab(&sub_matrix);
        }
        if (nlhs > 1)
        {
            arma::mat sub_matrix(geo.slice_memptr(1), geo.n_rows, geo.n_cols, false, true);
            plhs[1] = qd_mex_copy2matlab(&sub_matrix);
        }
        if (nlhs > 2)
        {
            arma::mat sub_matrix(geo.slice_memptr(2), geo.n_rows, geo.n_cols, false, true);
            plhs[2] = qd_mex_copy2matlab(&sub_matrix);
        }
    }
}
