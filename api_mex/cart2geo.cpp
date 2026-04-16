// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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
#include "quadriga_math.hpp"
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

- **kernel**<br>
  Kernel selection: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2 
  (throws if AVX2 unavailable); default `0`

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

    // Validate argument counts
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    arma::cube data = qd_mex_get_double_Cube(prhs[0]);
    int kernel = (nrhs < 2) ? 1 : qd_mex_get_scalar<int>(prhs[1], "kernel", 1);

    if (data.n_elem == 0 || data.n_rows != 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input must have 3 rows.");

    // Split data
    arma::uword n_val = data.n_cols * data.n_slices;
    arma::vec x(n_val, arma::fill::none), y(n_val, arma::fill::none), z(n_val, arma::fill::none);
    const double *pd = data.memptr();
    double *px = x.memptr(), *py = y.memptr(), *pz = z.memptr();
    for (arma::uword i = 0; i < n_val; ++i)
        px[i] = pd[3 * i], py[i] = pd[3 * i + 1], pz[i] = pd[3 * i + 2];

    // Allocate outputs
    arma::mat az, el, len;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&az, data.n_cols, data.n_slices);
    else
        az.set_size(data.n_cols, data.n_slices);

    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&el, data.n_cols, data.n_slices);
    else
        el.set_size(data.n_cols, data.n_slices);

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&len, data.n_cols, data.n_slices);

    // Map to vectors
    arma::vec az_vec(az.memptr(), n_val, false, true);
    arma::vec el_vec(el.memptr(), n_val, false, true);
    arma::vec len_vec = len.empty() ? arma::vec() : arma::vec(len.memptr(), n_val, false, true);

    arma::vec *p_len_vec = (nlhs > 2) ? &len_vec : nullptr;

    // Call library function (double precision)
    CALL_QD(quadriga_lib::fast_cart2geo<double>(x, y, z, az_vec, el_vec, p_len_vec, kernel));
}
