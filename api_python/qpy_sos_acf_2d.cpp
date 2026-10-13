// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos.acf_2d
Target ACF on a 2-D grid

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the
  distance, on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample
  point; `n_grid = 4·n_acf - 3`, the center is at index `2·n_acf - 2`
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [[sos.acf_approx]]
- Equally spaced means that every sample point lies within 1e-4 of the step from its place on
  the regular axis

## Usage:
```
acf_2d, dist_2d = quadriga_lib.sos.acf_2d( acf, dist )
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `(n_acf,)`

## Outputs:
- **`acf_2d`** — ACF on the grid; `(n_grid, n_grid)`
- **`dist_2d`** — Grid axis; `(n_grid,)`

## See also:
- [[sos.acf_approx]] (ACF produced by a set of sinusoids on the same grid)
- [[sos.calc_mse]] (difference between the two)
MD!*/

py::tuple sos_acf_2d(const py::array_t<double> &acf,
                     const py::array_t<double> &dist)
{
    // Read inputs
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Output allocation (a table with fewer than 2 samples is left to the library to reject)
    const arma::uword n_grid = (acf_a.n_elem < 2) ? 0 : 4 * acf_a.n_elem - 3;
    arma::mat acf_2d;
    arma::vec dist_2d;
    auto acf_2d_py = qd_python_init_output(n_grid, n_grid, &acf_2d);
    auto dist_2d_py = qd_python_init_output(n_grid, &dist_2d);

    // Call library function
    quadriga_lib::sos_acf_2d<double>(acf_a, dist_a, acf_2d, &dist_2d);

    return py::make_tuple(acf_2d_py, dist_2d_py);
}

// pybind11 declaration:
// m.def("acf_2d", &sos_acf_2d,
//       py::arg("acf").none(false),
//       py::arg("dist").none(false));
