// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos.calc_mse
Approximation error of the ACF of a set of sinusoids

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- `mse`: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`, with `i` counted
  from 0. A direction and its opposite give the same ACF
- `mse_core`: error on the grid of [[sos.acf_approx]] against [[sos.acf_2d]], for the grid points
  within a radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error
  is averaged over the three planes
- `mse_all`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond
  `Dmax`
- For `n_dim = 1`, `mse_core` is evaluated on the sample points `dist`, `mse_all` on the axis from
  `-2·Dmax` to `2·Dmax`, and `mse` equals `mse_core`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`.
  The target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- All outputs are always computed, including the 2-D grids behind `mse_core` and `mse_all`
- A perfect fit has a mean squared error of 0 and is returned as `inf`; this happens for a target
  that the sinusoids reproduce exactly at all sample points
- Equally spaced has the meaning given in [[sos.acf_2d]]

## Usage:
```
mse, mse_core, mse_all, acf_tail, test_dir = quadriga_lib.sos.calc_mse( freq, acf, dist, n_dir )
```

## Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `(n_acf,)`
- **`n_dir`** — Number of test directions; `int`; must be > 0; ignored for `n_dim = 1`; default: 10

## Outputs:
- **`mse`** — Error along the test directions in [dB]; `float`
- **`mse_core`** — Error within the core radius in [dB]; `float`
- **`mse_all`** — Error on the complete grid in [dB]; `float`
- **`acf_tail`** — Approximated ACF beyond the fitted range; `(4,)`
- **`test_dir`** — Test directions; `(n_dir, 2)` with columns `(azimuth, elevation)` for
  `n_dim = 3`; `(n_dir, 1)` with the azimuth for `n_dim = 2`; empty `(0, 0)` for `n_dim = 1`

## See also:
- [[sos.generate]] (finds the frequencies; minimises the error returned here)
- [[sos.acf_approx]], [[sos.acf_2d]] (the two grids that are compared)
MD!*/

py::tuple sos_calc_mse(const py::array_t<double> &freq,
                       const py::array_t<double> &acf,
                       const py::array_t<double> &dist,
                       arma::uword n_dir)
{
    // Read inputs
    const auto freq_a = qd_python_numpy2arma_Mat<double>(freq, true);
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Output allocation; the size of test_dir depends on the number of dimensions
    double mse_core = 0.0, mse_all = 0.0;
    arma::vec acf_tail;
    arma::mat test_dir;
    auto acf_tail_py = qd_python_init_output((arma::uword)4, &acf_tail);

    // Call library function
    double mse = quadriga_lib::sos_calc_mse<double>(freq_a, acf_a, dist_a, &mse_core, &mse_all, &acf_tail, &test_dir, n_dir);

    // Copy to python
    auto test_dir_py = qd_python_copy2numpy(&test_dir);

    return py::make_tuple(mse, mse_core, mse_all, acf_tail_py, test_dir_py);
}

// pybind11 declaration:
// m.def("calc_mse", &sos_calc_mse,
//       py::arg("freq").none(false),
//       py::arg("acf").none(false),
//       py::arg("dist").none(false),
//       py::arg("n_dir") = 10);
