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

#include "quadriga_tools.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Miscellaneous tools
SECTION!*/

/*!MD
# geo2cart
Transform geographic (azimuth, elevation, length) to Cartesian coordinates

## Description:
- Converts azimuth and elevation angles (in radians) into 3D Cartesian coordinates.
- Optional radial distance (`length`) can be provided; otherwise, unit vectors are returned.
- Useful for converting spherical direction data into vector representations.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
arma::Cube<dtype> quadriga_lib::geo2cart(
                const arma::Mat<dtype> &azimuth,
                const arma::Mat<dtype> &elevation,
                const arma::Mat<dtype> &length = {})
```

## Arguments:
- `const arma::Mat<dtype> **azimuth**` (input)<br>
  Azimuth angles in radians. Size `[n_row, n_col]`.

- `const arma::Mat<dtype> **elevation**` (input)<br>
  Elevation angles in radians. Size `[n_row, n_col]`.

- `const arma::Mat<dtype> **length** = {}` (optional input)<br>
  Radial distance (length). Same size as azimuth/elevation or empty for unit vectors. Size `[n_row, n_col]` or `[0, 0]`.

## Returns:
- `arma::Cube<dtype>` (output)<br>
  Cartesian coordinates with dimensions `[3, n_row, n_col]`, representing (x, y, z) for each input direction.

## Example:
```
arma::mat az(2, 2), el(2, 2), len(2, 2, arma::fill::ones);
auto cart = quadriga_lib::geo2cart(az, el, len);
```
MD!*/

// FUNCTION: Transform from geographic coordinates to Cartesian coordinates
template <typename dtype>
arma::Cube<dtype> quadriga_lib::geo2cart(const arma::Mat<dtype> &azimuth, const arma::Mat<dtype> &elevation, const arma::Mat<dtype> &length)
{
    if (azimuth.n_elem == 0 || elevation.n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (elevation.n_rows != azimuth.n_rows || elevation.n_cols != azimuth.n_cols)
        throw std::invalid_argument("Inputs must have the same size.");

    arma::uword n_row = azimuth.n_rows, n_col = azimuth.n_cols;
    arma::Cube<dtype> cart(3, n_row, n_col, arma::fill::none);

    if (length.n_elem == 0)
    {
        qd_geo2cart_interleaved(n_row * n_col, cart.memptr(), azimuth.memptr(), elevation.memptr());
        return cart;
    }

    if (length.n_rows != azimuth.n_rows || length.n_cols != azimuth.n_cols)
        throw std::invalid_argument("Inputs must have the same size.");

    qd_geo2cart_interleaved(n_row * n_col, cart.memptr(), azimuth.memptr(), elevation.memptr(), length.memptr());
    return cart;
}
template arma::Cube<float> quadriga_lib::geo2cart(const arma::Mat<float> &azimuth, const arma::Mat<float> &elevation, const arma::Mat<float> &length);

template arma::Cube<double> quadriga_lib::geo2cart(const arma::Mat<double> &azimuth, const arma::Mat<double> &elevation, const arma::Mat<double> &length);
