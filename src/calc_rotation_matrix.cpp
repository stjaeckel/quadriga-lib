// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_math.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# calc_rotation_matrix
Calculate rotation matrices from Euler angles

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, head) in column-major order (9 elements per orientation)
- R = Rz(heading)·Ry(tilt)·Rx(bank), right-handed as in 3GPP TR 38.901; a positive tilt turns the x-axis down
- Internally uses double precision regardless of `dtype`
- A matrix of orientations `[3, n]` is the cube `[3, n, 1]`; wrap it without a copy:
  `arma::Cube<dtype>(const_cast<dtype *>(mat.memptr()), 3, n, 1, false, true)`

## Declaration:
```
void quadriga_lib::calc_rotation_matrix(
    const arma::Cube<dtype> &orientation,
    arma::Cube<dtype> &rotation,
    bool transposed = false);
```

## Inputs:
- **`orientation`** — Euler angles (bank, tilt, head) in [rad]; `[3, n_row, n_col]`
- **`transposed`** *(optional)* — Returns the transpose of the rotation matrix

## Output:
- **`rotation`** — Rotation matrices in column-major order; `[9, n_row, n_col]`
MD!*/

template <typename dtype>
void quadriga_lib::calc_rotation_matrix(const arma::Cube<dtype> &orientation, arma::Cube<dtype> &rotation, bool transposed)
{
    if (orientation.n_elem == 0)
        throw std::invalid_argument("Input 'orientation' cannot be empty.");
    if (orientation.n_rows != 3)
        throw std::invalid_argument("Input 'orientation' must have 3 rows.");
    qd_check_alias(&rotation, "rotation", &orientation);

    arma::uword n_row = orientation.n_cols, n_col = orientation.n_slices;
    if (rotation.n_rows != 9 || rotation.n_cols != n_row || rotation.n_slices != n_col)
        rotation.set_size(9, n_row, n_col);
    qd_rotation_matrix(orientation.memptr(), rotation.memptr(), n_row * n_col, transposed);
}

template void quadriga_lib::calc_rotation_matrix(const arma::Cube<float> &orientation, arma::Cube<float> &rotation, bool transposed);

template void quadriga_lib::calc_rotation_matrix(const arma::Cube<double> &orientation, arma::Cube<double> &rotation, bool transposed);
