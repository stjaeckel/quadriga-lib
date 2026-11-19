// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_sos.hpp"

#include <cmath>
#include <stdexcept>

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_acf_2d
Target ACF on a 2-D grid

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the distance,
  on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample point;
  `n_grid = 4·n_acf - 3`, the center is element `2·n_acf - 2` (0-based)
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [[sos_acf_approx]]
- Equally spaced means that every sample point lies within 1e-4 of the step from its place on the regular
  axis; for long single-precision tables the rounding of the sample points themselves is allowed for

## Declaration:
```
arma::Mat<dtype> quadriga_lib::sos_acf_2d(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    arma::Col<dtype> *dist_2d = nullptr);
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, be equally spaced and finite; `[n_acf]`

## Output:
- **`dist_2d`** *(optional)* — Grid axis in [m]; `[n_grid]`

## Returns:
- ACF `acf_2d` on the grid; `[n_grid, n_grid]`

## See also:
- [[sos_acf_approx]] (ACF produced by a set of sinusoids on the same grid)
- [[sos_calc_mse]] (difference between the two)
MD!*/

template <typename dtype>
arma::Mat<dtype> quadriga_lib::sos_acf_2d(const arma::Col<dtype> &acf,
                                          const arma::Col<dtype> &dist,
                                          arma::Col<dtype> *dist_2d)
{
    qd_sos_check_acf(acf, dist);
    qd_check_alias(dist_2d, "dist_2d", &acf, &dist);
    if (!qd_sos_equally_spaced(dist))
        throw std::invalid_argument("Input 'dist' must be equally spaced.");

    const arma::uword n_acf = acf.n_elem, n_grid = 4 * n_acf - 3;

    const double step = (double)dist.at(n_acf - 1) / (double)(n_acf - 1);
    const long long half = 2 * (long long)n_acf - 2, i_max = (long long)n_acf - 1;

    const arma::vec acf_d = arma::conv_to<arma::vec>::from(acf);
    const double *pa = acf_d.memptr();

    arma::Mat<dtype> acf_2d(n_grid, n_grid, arma::fill::none);

    // Radial distance in grid steps; the table is equally spaced, so the step index is the table index
    for (long long ic = -half; ic <= half; ++ic)
    {
        dtype *col = acf_2d.colptr((arma::uword)(ic + half));
        for (long long ir = -half; ir <= half; ++ir)
        {
            double r = std::sqrt((double)(ic * ic + ir * ir)), v = 0.0;
            if (r <= (double)i_max)
            {
                long long i = (long long)std::floor(r);
                if (i >= i_max)
                    v = pa[i_max];
                else
                    v = pa[i] + (r - (double)i) * (pa[i + 1] - pa[i]);
            }
            col[ir + half] = (dtype)v;
        }
    }

    if (dist_2d != nullptr)
    {
        if (dist_2d->n_elem != n_grid)
            dist_2d->set_size(n_grid);
        for (long long i = -half; i <= half; ++i)
            dist_2d->at((arma::uword)(i + half)) = (dtype)((double)i * step);
    }
    return acf_2d;
}

template arma::Mat<float> quadriga_lib::sos_acf_2d(const arma::Col<float> &acf, const arma::Col<float> &dist, arma::Col<float> *dist_2d);
template arma::Mat<double> quadriga_lib::sos_acf_2d(const arma::Col<double> &acf, const arma::Col<double> &dist, arma::Col<double> *dist_2d);
