// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"
#include "qd_helpers_python.hpp"



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

    // Output allocation of the optional output (a table with fewer than 2 samples is left to the library to reject)
    const arma::uword n_grid = (acf_a.n_elem < 2) ? 0 : 4 * acf_a.n_elem - 3;
    arma::vec dist_2d;
    auto dist_2d_py = qd_python_init_output(n_grid, &dist_2d);

    // Call library function
    const arma::mat acf_2d = quadriga_lib::sos_acf_2d<double>(acf_a, dist_a, &dist_2d);

    return py::make_tuple(qd_python_copy2numpy(&acf_2d), dist_2d_py);
}

/*!MD
# sos.acf_approx
ACF produced by a set of sinusoids

- Autocorrelation function (ACF) of the values of [[sos.val]] over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [[sos.acf_2d]]: from `-2·Dmax` to `2·Dmax` with the spacing of
  `dist`, `n_grid = 4·n_acf - 3`
- The output is always 3-dimensional
- `n_dim = 1`: along the axis; `(n_grid, 1, 1)`
- `n_dim = 2`: on the x-y plane; `(n_grid, n_grid, 1)`; element `[iy, ix, 0]`
- `n_dim = 3`: on the x-y, x-z and y-z plane (index 0, 1, 2 of the last axis); `(n_grid, n_grid, 3)`;
  the row index belongs to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values
- `R(-dx) = R(dx)`: each plane is point-symmetric about its center, exactly
- Equally spaced has the meaning given in [[sos.acf_2d]]

## Usage:
```
acf_approx, dist_2d = quadriga_lib.sos.acf_approx( spatial_freq, dist )
```

## Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; at least
  2 elements; `(n_acf,)`

## Outputs:
- **`acf_approx`** — Approximated ACF; shape depends on `n_dim`, see above
- **`dist_2d`** — Grid axis; `(n_grid,)`

## See also:
- [[sos.acf_2d]] (target ACF on the same grid)
- [[sos.calc_mse]] (difference between the two)
MD!*/

py::tuple sos_acf_approx(const py::array_t<double> &spatial_freq,
                         const py::array_t<double> &dist)
{
    // Read inputs
    const auto freq_a = qd_python_numpy2arma_Mat<double>(spatial_freq, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Output allocation of the optional output (invalid sizes are left to the library to reject)
    const arma::uword n_grid = (dist_a.n_elem < 2) ? 0 : 4 * dist_a.n_elem - 3;
    arma::vec dist_2d;
    auto dist_2d_py = qd_python_init_output(n_grid, &dist_2d);

    // Call library function
    const arma::cube acf_approx = quadriga_lib::sos_acf_approx<double>(freq_a, dist_a, &dist_2d);

    return py::make_tuple(qd_python_copy2numpy(&acf_approx), dist_2d_py);
}

/*!MD
# sos.acf_estimate
Monte-Carlo estimate of the ACF of one generator

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at
  random positions
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin,
  evaluates the generator there, and for each distance bin computes the correlation coefficient
  between the two values of all position pairs whose distance falls into the bin
- Positions: an array `(2·n_pos, 3)` is filled column by column (all x, then all y, then all z)
  with `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in
  [[sos.init_phases]]
- Bin `n` covers the distances from the midpoint between `bins[n-1]` and `bins[n]` (excluded) to
  the midpoint between `bins[n]` and `bins[n+1]` (included). The first bin starts at 1e-5 m, the
  last bin ends half a mean bin spacing above `bins[-1]`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices on the same map. The first `n_pos` positions are the
  first device, the second `n_pos` positions the second device, and the value of a link is
  [[sos.val]] with both positions. Two links form a pair if the distance between their first
  devices falls into the bin and the distance between their second devices does not exceed the
  upper limit of the bin; requires `acf` and `dist`
- `acf_est_dual` is computed if `acf` or `dist` is given, and is `None` if both are omitted; giving
  only one of the two is an error
- The estimate describes one realisation; it scatters around the ACF of the sinusoids
- Run time grows with `n_pos^2`; the result does not depend on the number of threads

## Usage:
```
acf_est, dist_est, acf_est_dual = quadriga_lib.sos.acf_estimate( spatial_freq, phase, bins, n_pos, \
    *, acf, dist, seed )
```

## Inputs:
- **`spatial_freq`** — Spatial frequencies; `(n_coeff, 3)`
- **`phase`** — Phases of the generator; `(n_coeff,)` or `(n_coeff, 1)`
- **`bins`** — Bin centers; must be >= 0, increase strictly and be finite; at least 2
  elements; `(n_bin,)`
- **`n_pos`** — Number of random positions per device; `int`; must be > 1
- **`acf`** — Target ACF; required for `acf_est_dual`, must then be > -1;
  `(n_acf,)`; default: `None`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`;
  default: `None`
- **`seed`** — Seed for the positions; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

## Outputs:
- **`acf_est`** — Estimated ACF; `(n_bin,)`
- **`dist_est`** — Mean distance of the pairs in each bin; `(n_bin,)`
- **`acf_est_dual`** — Estimated ACF for two devices on the same map; `(n_bin,)`; `None` if `acf`
  and `dist` were not given

## See also:
- [[sos.acf_approx]] (ACF of the sinusoids in closed form)
- [[sos.val]] (evaluation of the generator)
MD!*/

py::tuple sos_acf_estimate(const py::array_t<double> &spatial_freq,
                           const py::array_t<double> &phase,
                           const py::array_t<double> &bins,
                           arma::uword n_pos,
                           py::handle acf,
                           py::handle dist,
                           arma::sword seed)
{
    // Read inputs
    const auto freq_a = qd_python_numpy2arma_Mat<double>(spatial_freq, true);
    const auto phase_a = qd_python_numpy2arma_Col<double>(phase, true);
    const auto bins_a = qd_python_numpy2arma_Col<double>(bins, true);
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Output allocation; the two-device estimate is requested only with an ACF table
    arma::vec dist_est, acf_est_dual;
    auto dist_est_py = qd_python_init_output(bins_a.n_elem, &dist_est);

    py::object acf_est_dual_py = py::none();
    arma::vec *p_acf_est_dual = nullptr;
    if (!acf_a.is_empty() || !dist_a.is_empty())
    {
        acf_est_dual_py = qd_python_init_output(bins_a.n_elem, &acf_est_dual);
        p_acf_est_dual = &acf_est_dual;
    }

    // Call library function
    const arma::vec acf_est = quadriga_lib::sos_acf_estimate<double>(freq_a, phase_a, bins_a, n_pos, &dist_est, p_acf_est_dual,
                                                                     acf_a, dist_a, seed);

    return py::make_tuple(qd_python_copy2numpy(&acf_est), dist_est_py, acf_est_dual_py);
}

/*!MD
# sos.acfi
Target ACF at given distances

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table
- The same rule is applied by [[sos.val]] and [[sos.remove_bias]] for two devices on the same map

## Usage:
```
val = quadriga_lib.sos.acfi( acf, dist, d )
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`
- **`d`** — Distances; must be >= 0; `(n_d,)`

## Output:
- **`val`** — ACF at the distances `d`; `(n_d,)`

## See also:
- [[sos.preset]] (ACF tables of the predefined models)
MD!*/

py::array_t<double> sos_acfi(const py::array_t<double> &acf,
                             const py::array_t<double> &dist,
                             const py::array_t<double> &d)
{
    // Read inputs
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);
    const auto d_a = qd_python_numpy2arma_Col<double>(d, true);

    // Call library function
    const arma::vec val = quadriga_lib::sos_acfi<double>(acf_a, dist_a, d_a);
    return qd_python_copy2numpy(&val);
}
