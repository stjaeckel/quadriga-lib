// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_acf_approx
ACF produced by a set of sinusoids

- Autocorrelation function (ACF) of the values of [[sos_val]] over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [[sos_acf_2d]]: from `-2·Dmax` to `2·Dmax` with the spacing of
  `dist`, `n_grid = 4·n_acf - 3`
- The output is always 3-dimensional
- `n_dim = 1`: along the axis; `(n_grid, 1, 1)`
- `n_dim = 2`: on the x-y plane; `(n_grid, n_grid, 1)`; element `[iy, ix, 0]`
- `n_dim = 3`: on the x-y, x-z and y-z plane (index 0, 1, 2 of the last axis); `(n_grid, n_grid, 3)`;
  the row index belongs to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values

## Usage:
```
acf_approx, dist_2d = quadriga_lib.sos.acf_approx( freq, dist )
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`dist`** — Sample points of the ACF; must start at 0 and be equally spaced; at least 2
  elements; `(n_acf,)`

## Outputs:
- **`acf_approx`** — Approximated ACF; shape depends on `n_dim`, see above
- **`dist_2d`** — Grid axis; `(n_grid,)`

## See also:
- [[sos_acf_2d]] (target ACF on the same grid)
- [[sos_calc_mse]] (difference between the two)
MD!*/

py::tuple sos_acf_approx(const py::array_t<double> &freq,
                         const py::array_t<double> &dist)
{
    // Read inputs
    const auto freq_a = qd_python_numpy2arma_Mat<double>(freq, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Output allocation (invalid sizes are left to the library to reject)
    const arma::uword n_dim = freq_a.n_cols, n_grid = (dist_a.n_elem < 2) ? 0 : 4 * dist_a.n_elem - 3;
    arma::cube acf_approx;
    arma::vec dist_2d;
    auto acf_approx_py = qd_python_init_output(n_grid, (n_dim == 1) ? 1 : n_grid, (n_dim == 3) ? 3 : 1, &acf_approx);
    auto dist_2d_py = qd_python_init_output(n_grid, &dist_2d);

    // Call library function
    quadriga_lib::sos_acf_approx<double>(freq_a, dist_a, acf_approx, &dist_2d);

    return py::make_tuple(acf_approx_py, dist_2d_py);
}

// pybind11 declaration:
// m.def("acf_approx", &sos_acf_approx,
//       py::arg("freq"),
//       py::arg("dist"));
