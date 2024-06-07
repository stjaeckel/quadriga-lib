// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "quadriga_lib.hpp"

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
geo_coords = quadriga_lib.cart2geo(cart_coords)
```

## Input Argument:
- **`cart_coords`**<br>
  Cartesian coordinates (x,y,z), Size: `[3, n_row, n_col]`

## Output Arguments:
- **`geo_coords`**<br>
  Geographic coordinates, Size: `[3, n_row, n_col]`<br>
  First row: Azimuth angles in [rad], values between -pi and pi.<br>
  Second row: Elevation angles in [rad], values between -pi/2 and pi/2.<br>
  Third row: Vector length, i.e. the distance from the origin to the point defined by x,y,z.
MD!*/

pybind11::array_t<double> cart2geo(pybind11::array_t<double> cart)
{
    pybind11::buffer_info buf = cart.request();
    if (buf.ndim != 3)
        throw std::invalid_argument("Number of dimensions must be 3"); 

    arma::cube cart_arma(reinterpret_cast<double *>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2], false, true);
    arma::cube result = quadriga_lib::cart2geo(cart_arma);
    return pybind11::array_t<double>({result.n_rows, result.n_cols, result.n_slices}, result.memptr());
}
