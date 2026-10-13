// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

#include <cmath>
#include <string>

// Count argument: read as double, so that a negative or non-finite value is rejected with the name of the
// argument instead of wrapping around in the conversion to an unsigned type. Non-integer values are truncated.
static arma::uword qd_mex_sos_count(const mxArray *input, const char *name, double default_value)
{
    const double v = qd_mex_get_scalar<double>(input, name, default_value);
    if (!(v >= 1.0) || !(v < 9.0e18))
    {
        const std::string msg = std::string("Input '") + name + "' must be a finite number >= 1.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", msg.c_str());
    }
    return (arma::uword)v;
}

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# SOS_CALC_MSE
Approximation error of the ACF of a set of sinusoids

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- `mse`: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`, with `i` counted
  from 0. A direction and its opposite give the same ACF
- `mse_core`: error on the grid of [[sos_acf_approx]] against [[sos_acf_2d]], for the grid points
  within a radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error
  is averaged over the three planes
- `mse_all`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond
  `Dmax`
- For `n_dim = 1`, `mse_core` is evaluated on the sample points `dist`, `mse_all` on the axis from
  `-2·Dmax` to `2·Dmax`, and `mse` equals `mse_core`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`.
  The target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- `mse_core` and `mse_all` need the 2-D grids and are only computed when requested
- A perfect fit has a mean squared error of 0 and is returned as `Inf`; this happens for a target
  that the sinusoids reproduce exactly at all sample points
- Equally spaced has the meaning given in [[sos_acf_2d]]

## Usage:
```
[ mse, mse_core, mse_all, acf_tail, test_dir ] = quadriga_lib.sos_calc_mse( freq, acf, dist, ...
    n_dir );
```

## Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `[n_acf]`
- **`n_dir`** — Number of test directions; must be > 0; ignored for `n_dim = 1`; default: 10

## Outputs:
- **`mse`** — Error along the test directions in [dB]; scalar
- **`mse_core`** — Error within the core radius in [dB]; scalar
- **`mse_all`** — Error on the complete grid in [dB]; scalar
- **`acf_tail`** — Approximated ACF beyond the fitted range; column vector; `[4]`
- **`test_dir`** — Test directions; `[n_dir, 2]` with columns `{azimuth, elevation}` for
  `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

## See also:
- [[sos_generate]] (finds the frequencies; minimises the error returned here)
- [[sos_acf_approx]], [[sos_acf_2d]] (the two grids that are compared)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 3 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::mat freq = qd_mex_get_Mat<double>(prhs[0]);
    const arma::vec acf = qd_mex_get_Col<double>(prhs[1]);
    const arma::vec dist = qd_mex_get_Col<double>(prhs[2]);
    const arma::uword n_dir = (nrhs < 4) ? 10 : qd_mex_sos_count(prhs[3], "n_dir", 10.0);

    // Wrap optional pointers
    double mse = 0.0, mse_core = 0.0, mse_all = 0.0;
    arma::vec acf_tail;
    arma::mat test_dir;
    double *p_mse_core = (nlhs > 1) ? &mse_core : nullptr;
    double *p_mse_all = (nlhs > 2) ? &mse_all : nullptr;
    arma::vec *p_acf_tail = (nlhs > 3) ? &acf_tail : nullptr;
    arma::mat *p_test_dir = (nlhs > 4) ? &test_dir : nullptr;

    // Call library function
    CALL_QD(mse = quadriga_lib::sos_calc_mse<double>(freq, acf, dist, p_mse_core, p_mse_all, p_acf_tail, p_test_dir, n_dir));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&mse);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&mse_core);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&mse_all);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&acf_tail);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&test_dir);
}
