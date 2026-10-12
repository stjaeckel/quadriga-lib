// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# SOS_ACF_2D
Target ACF on a 2-D grid

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the
  distance, on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample
  point; `n_grid = 4·n_acf - 3`, the center is element `2·n_acf - 1`
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [[sos_acf_approx]]

## Usage:
```
[ acf_2d, dist_2d ] = quadriga_lib.sos_acf_2d( acf, dist );
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0 and be equally spaced; `[n_acf]`

## Outputs:
- **`acf_2d`** — ACF on the grid; `[n_grid, n_grid]`
- **`dist_2d`** — Grid axis; column vector; `[n_grid]`

## See also:
- [[sos_acf_approx]] (ACF produced by a set of sinusoids on the same grid)
- [[sos_calc_mse]] (difference between the two)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 2 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::vec acf = qd_mex_get_Col<double>(prhs[0]);
    const arma::vec dist = qd_mex_get_Col<double>(prhs[1]);

    // Output allocation (a table with fewer than 2 samples is left to the library to reject)
    const arma::uword n_grid = (acf.n_elem < 2) ? 0 : 4 * acf.n_elem - 3;
    arma::mat acf_2d;
    arma::vec dist_2d;
    if (nlhs > 0 && n_grid != 0)
        plhs[0] = qd_mex_init_output(&acf_2d, n_grid, n_grid);
    if (nlhs > 1 && n_grid != 0)
        plhs[1] = qd_mex_init_output(&dist_2d, n_grid);

    // Wrap optional pointers
    arma::vec *p_dist_2d = dist_2d.empty() ? nullptr : &dist_2d;

    // Call library function
    CALL_QD(quadriga_lib::sos_acf_2d<double>(acf, dist, acf_2d, p_dist_2d));
}
