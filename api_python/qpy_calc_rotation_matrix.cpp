// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# calc_rotation_matrix
Calculate rotation matrices from Euler angles

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, head) in column-major order (9 elements per orientation)
- R = Rz(heading)·Ry(tilt)·Rx(bank), right-handed as in 3GPP TR 38.901; a positive tilt turns the x-axis down

## Usage:
```
rotation = quadriga_lib.tools.calc_rotation_matrix( orientation, *, transposed )
```

## Inputs:
- **`orientation`** — Euler angles (bank, tilt, head) in rad; `(3,)`, `(3, n)` or `(3, n_row, n_col)`
- **`transposed`** — Returns the transpose of the rotation matrix; default: `False`

## Output:
- **`rotation`** — Rotation matrices in column-major order; same rank as `orientation`: `(9,)`, `(9, n)` or
  `(9, n_row, n_col)`
MD!*/

py::array_t<double> calc_rotation_matrix(const py::array_t<double> &orientation,
                                         py::handle transposed)
{
    const auto orientation_a = qd_python_numpy2arma_Cube(orientation, true);
    const bool transposed_b = qd_python_flag(transposed, false, "transposed");

    // Output [9, n_row, n_col] in numpy memory; for a 1D or 2D input the trailing axes of size 1 are dropped (a view)
    arma::cube rotation;
    auto rotation_py = qd_python_init_output((arma::uword)9, orientation_a.n_cols, orientation_a.n_slices, &rotation);
    quadriga_lib::calc_rotation_matrix<double>(orientation_a, rotation, transposed_b);

    if (orientation.ndim() == 1)
        return py::array_t<double>(rotation_py.reshape({(py::ssize_t)9}));
    if (orientation.ndim() == 2)
        return py::array_t<double>(rotation_py.reshape({(py::ssize_t)9, (py::ssize_t)orientation_a.n_cols}));
    return rotation_py;
}
