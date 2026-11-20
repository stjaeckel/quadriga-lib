// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# SOS_ACF_APPROX
ACF produced by a set of sinusoids

- Autocorrelation function (ACF) of the values of [[sos_val]] over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [[sos_acf_2d]]: from `-2·Dmax` to `2·Dmax` with the spacing of
  `dist`, `n_grid = 4·n_acf - 3`
- `n_dim = 1`: along the axis; `[n_grid, 1]`
- `n_dim = 2`: on the x-y plane; `[n_grid, n_grid]`; element `(iy, ix)`
- `n_dim = 3`: on the x-y, x-z and y-z plane (pages 1, 2, 3); `[n_grid, n_grid, 3]`; the row index
  belongs to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values
- `R(-dx) = R(dx)`: each plane is point-symmetric about its center, exactly
- Equally spaced has the meaning given in [[sos_acf_2d]]

## Usage:
```
[ acf_approx, dist_2d ] = quadriga_lib.sos_acf_approx( spatial_freq, dist );
```

## Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; at least
  2 elements; `[n_acf]`

## Outputs:
- **`acf_approx`** — Approximated ACF; size depends on `n_dim`, see above
- **`dist_2d`** — Grid axis; `[n_grid, 1]`

## See also:
- [[sos_acf_2d]] (target ACF on the same grid)
- [[sos_calc_mse]] (difference between the two)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const arma::mat spatial_freq = qd_mex_get_Mat<double>(prhs[0], false, "spatial_freq");
    const arma::vec dist = qd_mex_get_Col<double>(prhs[1], false, "dist");

    // Output allocation (invalid sizes are left to the library to reject)
    const arma::uword n_dim = spatial_freq.n_cols, n_grid = (dist.n_elem < 2) ? 0 : 4 * dist.n_elem - 3;
    const bool valid = n_grid != 0 && n_dim >= 1 && n_dim <= 3;
    arma::cube acf_approx;
    arma::vec dist_2d;
    if (nlhs > 1 && valid)
        plhs[1] = qd_mex_init_output(&dist_2d, n_grid);

    arma::vec *p_dist_2d = dist_2d.empty() ? nullptr : &dist_2d;

    CALL_QD(acf_approx = quadriga_lib::sos_acf_approx<double>(spatial_freq, dist, p_dist_2d));
    plhs[0] = qd_mex_copy2matlab(&acf_approx);
}
