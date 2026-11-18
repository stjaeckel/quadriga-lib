// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# interp
Perform linear interpolation (1D or 2D) on multiple data sets

- Interpolates given input data at specified output points.
- Outside the range of the input sample points, the nearest end value is returned (constant extrapolation);
  output sample points that are NaN give NaN.
- The input sample points need not be sorted; repeated values are allowed; they must be finite.
- `y = None` (or empty) selects the 1D case.

## Usage:
```
# 2D case
output = quadriga_lib.tools.interp( x, y, input, x_new, y_new )

# 1D case
output = quadriga_lib.tools.interp( x, None, input, x_new )
```

## Inputs:
- **`x`** — Input x-axis sample points; `(nx,)`
- **`y`** — Input y-axis sample points; `(ny,)`; `None` for 1D
- **`input`** — Input data; 2D: `(ny, nx)` or `(ny, nx, ne)`; 1D: `(nx,)` or `(nx, ne)`; `ne` = number of data sets
- **`x_new`** — Output x-axis sample points; `(nx_new,)`
- **`y_new`** — Output y-axis sample points, required in 2D, `None` or empty in 1D; `(ny_new,)`; default: `None`

## Output:
- **`output`** — Interpolated data, same rank as `input`; 2D: `(ny_new, nx_new)` or `(ny_new, nx_new, ne)`;
  1D: `(nx_new,)` or `(nx_new, ne)`
MD!*/

py::array_t<double> interp(const py::array_t<double> &x,
                           py::handle y,
                           const py::array_t<double> &input,
                           const py::array_t<double> &x_new,
                           py::handle y_new)
{
    const auto x_a = qd_python_numpy2arma_Col(x, true);
    const auto y_a = qd_python_numpy2arma_Col<double>(y, true);
    const auto x_new_a = qd_python_numpy2arma_Col(x_new, true);
    const auto y_new_a = qd_python_numpy2arma_Col<double>(y_new, true);
    const auto nd = input.ndim();

    if (y_a.is_empty()) // 1D: interp_1D on the columns of input
    {
        if (nd < 1 || nd > 2)
            throw std::invalid_argument("Input 'input' must have shape (nx,) or (nx, ne) in the 1D case.");
        if (!y_new_a.is_empty())
            throw std::invalid_argument("Input 'y_new' must be None or empty in the 1D case.");

        const auto input_a = qd_python_numpy2arma_Mat(input, true);
        arma::mat output;
        auto output_py = qd_python_init_output(x_new_a.n_elem, input_a.n_cols, &output);
        quadriga_lib::interp_1D<double>(x_a, input_a, x_new_a, output);

        // A 1D input gives a 1D output: dropping the axis of size 1 is a view
        if (nd == 1)
            return py::array_t<double>(output_py.reshape({(py::ssize_t)x_new_a.n_elem}));
        return output_py;
    }

    // 2D: interp_2D
    if (nd < 2 || nd > 3)
        throw std::invalid_argument("Input 'input' must have shape (ny, nx) or (ny, nx, ne) in the 2D case.");
    if (y_new.is_none())
        throw std::invalid_argument("Input 'y_new' is required in the 2D case.");

    const auto input_a = qd_python_numpy2arma_Cube(input, true);
    arma::cube output;
    auto output_py = qd_python_init_output(y_new_a.n_elem, x_new_a.n_elem, input_a.n_slices, &output);
    quadriga_lib::interp_2D<double>(x_a, y_a, input_a, x_new_a, y_new_a, output);

    // A 2D input gives a 2D output: dropping the axis of size 1 is a view
    if (nd == 2)
        return py::array_t<double>(output_py.reshape({(py::ssize_t)y_new_a.n_elem, (py::ssize_t)x_new_a.n_elem}));
    return output_py;
}
