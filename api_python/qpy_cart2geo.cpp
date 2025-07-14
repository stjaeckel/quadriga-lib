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

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-Specific Simulation Tools
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
from quadriga_lib import RTtools
geo_coords = RTtools.cart2geo(cart_coords)
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

// #include <chrono>

// std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(); // Start time
// std::cout << "Start CPP" << std::endl;

// std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now(); // Current time
// double ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
// std::cout << "Input data converted, t = " << std::round(ms/100.0)/10.0 << std::endl;

py::array_t<double> cart2geo(const py::array_t<double> &cart)
{
    const auto cart_arma = qd_python_numpy2arma_Cube(cart, true);

    arma::cube geo_arma;
    auto geo = qd_python_init_output(cart_arma.n_cols, cart_arma.n_slices, cart_arma.n_rows, &geo_arma);

    quadriga_lib::cart2geo(cart_arma, geo_arma);

    return geo;
}
