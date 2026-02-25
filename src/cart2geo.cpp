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
Miscellaneous / Tools
SECTION!*/

/*!MD
# cart2geo
Convert Cartesian coordinates to geographic coordinates (azimuth, elevation, distance)

## Description:
- Transforms 3D Cartesian coordinates `(x, y, z)` into geographic coordinates:
  - Azimuth angle [rad]
  - Elevation angle [rad]
  - Distance (vector norm)
- Azimuth is measured in the x-y plane from the x-axis; elevation is from the x-y plane upward.
- Allowed datatypes (`dtype`): `float` or `double`

## Declaration:
```
arma::Cube<dtype> quadriga_lib::cart2geo(const arma::Cube<dtype> &cart);

arma::Mat<dtype> quadriga_lib::cart2geo(const arma::Mat<dtype> &cart);

arma::Col<dtype> quadriga_lib::cart2geo(const arma::Col<dtype> &cart);
```

## Arguments:
- `const arma::Cube<dtype> ***cart**` or `const arma::Mat<dtype> ***cart**` or `const arma::Col<dtype> **cart**` (input)<br>
  Cartesian coordinate vectors (x, y, z), Size `[3, n_row, n_col]` or `[3, n_row]` or `[3]`.

## Returns:
- `arma::Cube<dtype>` or `arma::Mat<dtype>` or `arma::Col<dtype>`<br>
  Geographic coordinate vectors `(azimuth, elevation, distance)`, Size `[n_row, n_col, 3]` or `[n_row, 3]` or `[3]`.

## Example:
```
arma::vec cart = {1.0, 1.0, 1.0};
auto geo = quadriga_lib::cart2geo(cart);
```
MD!*/

// FUNCTION: Transform from Cartesian coordinates to geographic coordinates
template <typename dtype>
void quadriga_lib::cart2geo(const arma::Cube<dtype> &cart, arma::Cube<dtype> &geo)
{
    if (cart.n_elem == 0)
        throw std::invalid_argument("Input cannot be empty.");
    if (cart.n_rows != 3)
        throw std::invalid_argument("Input must have 3 rows.");

    arma::uword n_row = cart.n_cols, n_col = cart.n_slices;
    if (geo.n_rows != n_row || geo.n_cols != n_col || geo.n_slices != 3)
        geo.set_size(n_row, n_col, 3);

    qd_cart2geo_interleaved(n_row * n_col, cart.memptr(), geo.slice_memptr(0), geo.slice_memptr(1), geo.slice_memptr(2));
}

template <typename dtype>
void quadriga_lib::cart2geo(const arma::Mat<dtype> &cart, arma::Mat<dtype> &geo)
{
    if (cart.n_elem == 0)
        throw std::invalid_argument("Input cannot be empty.");
    if (cart.n_rows != 3)
        throw std::invalid_argument("Input must have 3 rows.");

    arma::uword n_row = cart.n_cols;
    if (geo.n_rows != n_row || geo.n_cols != 3)
        geo.set_size(n_row, 3);

    qd_cart2geo_interleaved(n_row, cart.memptr(), geo.colptr(0), geo.colptr(1), geo.colptr(2));
}

template <typename dtype>
void quadriga_lib::cart2geo(const arma::Col<dtype> &cart, arma::Col<dtype> &geo)
{
    if (cart.n_elem != 3)
        throw std::invalid_argument("Input must have 3 elements.");

    if (geo.n_elem != 3)
        geo.set_size(3);

    dtype *p = geo.memptr();
    qd_cart2geo_interleaved(1, cart.memptr(), &p[0], &p[1], &p[2]);
}

template <typename arma_type>
arma_type quadriga_lib::cart2geo(const arma_type &cart)
{
    arma_type geo;
    quadriga_lib::cart2geo(cart, geo);
    return geo;
}

template void quadriga_lib::cart2geo(const arma::Cube<float> &cart, arma::Cube<float> &geo);

template void quadriga_lib::cart2geo(const arma::Cube<double> &cart, arma::Cube<double> &geo);

template void quadriga_lib::cart2geo(const arma::Mat<float> &cart, arma::Mat<float> &geo);

template void quadriga_lib::cart2geo(const arma::Mat<double> &cart, arma::Mat<double> &geo);

template void quadriga_lib::cart2geo(const arma::Col<float> &cart, arma::Col<float> &geo);

template void quadriga_lib::cart2geo(const arma::Col<double> &cart, arma::Col<double> &geo);

template arma::Cube<float> quadriga_lib::cart2geo(const arma::Cube<float> &cart);

template arma::Cube<double> quadriga_lib::cart2geo(const arma::Cube<double> &cart);

template arma::Mat<float> quadriga_lib::cart2geo(const arma::Mat<float> &cart);

template arma::Mat<double> quadriga_lib::cart2geo(const arma::Mat<double> &cart);

template arma::Col<float> quadriga_lib::cart2geo(const arma::Col<float> &cart);

template arma::Col<double> quadriga_lib::cart2geo(const arma::Col<double> &cart);
