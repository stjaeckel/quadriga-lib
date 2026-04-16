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

#include "quadriga_tools.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# calc_rotation_matrix
Calculate rotation matrices from Euler angles

## Description:
- Computes 3×3 rotation matrices from Euler angles (bank, tilt, head) in column-major order (9 elements per orientation)
- Internally uses double precision regardless of `dtype`
- Allowed datatypes: `float` or `double`

## Declaration:
```
arma::Cube<dtype> quadriga_lib::calc_rotation_matrix(const arma::Cube<dtype> &orientation,
                bool invert_y_axis = false, bool transposeR = false);

arma::Mat<dtype> quadriga_lib::calc_rotation_matrix(const arma::Mat<dtype> &orientation,
                bool invert_y_axis = false, bool transposeR = false);

arma::Col<dtype> quadriga_lib::calc_rotation_matrix(const arma::Col<dtype> &orientation,
                bool invert_y_axis = false, bool transposeR = false);
```

## Input Arguments:
- **`orientation`** — Euler angles (bank, tilt, head) in radians; `[3, n_row, n_col]` or `[3, n_mat]` or `[3]`
- **`invert_y_axis`** *(optional)* — Inverts the y-axis of the rotation
- **`transposeR`** *(optional)* — Returns the transpose of the rotation matrix

## Returns:
- Rotation matrices in column-major order; `[9, n_row, n_col]` or `[9, n_mat]` or `[9]`
MD!*/

template <typename dtype>
arma::Cube<dtype> quadriga_lib::calc_rotation_matrix(const arma::Cube<dtype> &orientation, bool invert_y_axis, bool transposeR)
{
    if (orientation.n_elem == 0)
        throw std::invalid_argument("Input cannot be empty.");
    if (orientation.n_rows != 3)
        throw std::invalid_argument("Input must have 3 rows.");

    arma::uword n_row = orientation.n_cols, n_col = orientation.n_slices;
    arma::Cube<dtype> rotation(9, n_row, n_col, arma::fill::none);
    qd_rotation_matrix(orientation.memptr(), rotation.memptr(), n_row * n_col, invert_y_axis, transposeR);
    return rotation;
}

template <typename dtype>
arma::Mat<dtype> quadriga_lib::calc_rotation_matrix(const arma::Mat<dtype> &orientation, bool invert_y_axis, bool transposeR)
{
    if (orientation.n_elem == 0)
        throw std::invalid_argument("Input cannot be empty.");
    if (orientation.n_rows != 3)
        throw std::invalid_argument("Input must have 3 rows.");

    arma::uword n_row = orientation.n_cols;
    arma::Mat<dtype> rotation(9, n_row, arma::fill::none);
    qd_rotation_matrix(orientation.memptr(), rotation.memptr(), n_row, invert_y_axis, transposeR);
    return rotation;
}

template <typename dtype>
arma::Col<dtype> quadriga_lib::calc_rotation_matrix(const arma::Col<dtype> &orientation, bool invert_y_axis, bool transposeR)
{
    if (orientation.n_elem != 3)
        throw std::invalid_argument("Input must have 3 elements.");

    arma::Col<dtype> rotation(9, arma::fill::none);
    qd_rotation_matrix(orientation.memptr(), rotation.memptr(), 1ULL, invert_y_axis, transposeR);
    return rotation;
}

template arma::Cube<float> quadriga_lib::calc_rotation_matrix(const arma::Cube<float> &orientation, bool invert_y_axis, bool transposeR);

template arma::Cube<double> quadriga_lib::calc_rotation_matrix(const arma::Cube<double> &orientation, bool invert_y_axis, bool transposeR);

template arma::Mat<float> quadriga_lib::calc_rotation_matrix(const arma::Mat<float> &orientation, bool invert_y_axis, bool transposeR);

template arma::Mat<double> quadriga_lib::calc_rotation_matrix(const arma::Mat<double> &orientation, bool invert_y_axis, bool transposeR);

template arma::Col<float> quadriga_lib::calc_rotation_matrix(const arma::Col<float> &orientation, bool invert_y_axis, bool transposeR);

template arma::Col<double> quadriga_lib::calc_rotation_matrix(const arma::Col<double> &orientation, bool invert_y_axis, bool transposeR);
