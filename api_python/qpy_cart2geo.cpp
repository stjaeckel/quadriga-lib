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

#include "python_arma_adapter.hpp"
#include "quadriga_math.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# cart2geo
Transform Cartesian (x,y,z) coordinates to Geographic (az, el, length) coordinates

## Description:
This function transforms Cartesian (x,y,z) coordinates to Geographic (azimuth, elevation, length)
coordinates. A geographic coordinate system is a three-dimensional reference system that locates
points on the surface of a sphere. A point has three coordinate values: azimuth, elevation and length
where azimuth and elevation measure angles. In the geographic coordinate system, the elevation angle
θ = 90° points to the zenith and θ = 0° points to the horizon.

## Usage:
```
import quadriga_lib
geo_coords = quadriga_lib.tools.cart2geo(cart_coords)
```

## Input Argument:
- **`cart_coords`**<br>
  Cartesian coordinates (x,y,z), Shape: `(3, n_row, n_col)`

## Output Arguments:
- **`geo_coords`**<br>
  Geographic coordinates, Shape: `(3, n_row, n_col)`<br>
  First row: Azimuth angles in [rad], values between -pi and pi.<br>
  Second row: Elevation angles in [rad], values between -pi/2 and pi/2.<br>
  Third row: Vector length, i.e. the distance from the origin to the point defined by x,y,z.
MD!*/

py::array_t<double> cart2geo(const py::array_t<double> &cart)
{
    const auto data = qd_python_numpy2arma_Cube(cart, true);

    if (data.n_elem == 0 || data.n_rows != 3)
        throw std::invalid_argument("Input must have 3 rows.");

    // Split data
    arma::uword n_val = data.n_cols * data.n_slices;
    arma::vec x(n_val, arma::fill::none), y(n_val, arma::fill::none), z(n_val, arma::fill::none);
    const double *pd = data.memptr();
    double *px = x.memptr(), *py = y.memptr(), *pz = z.memptr();
    for (arma::uword i = 0; i < n_val; ++i)
        px[i] = pd[3 * i], py[i] = pd[3 * i + 1], pz[i] = pd[3 * i + 2];

    // Call library function (double precision)
    arma::vec az, el, len;
    quadriga_lib::fast_cart2geo<double>(x, y, z, az, el, &len, 1);

    // Combine outputs
    arma::cube geo_arma;
    auto geo = qd_python_init_output(3, data.n_cols, data.n_slices, &geo_arma);

    double *pa = az.memptr(), *pe = el.memptr(), *pl = len.memptr(), *po = geo_arma.memptr();
    for (arma::uword i = 0; i < n_val; ++i)
        po[3 * i] = pa[i], po[3 * i + 1] = pe[i], po[3 * i + 2] = pl[i];

    return geo;
}
