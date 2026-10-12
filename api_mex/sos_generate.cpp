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
# SOS_GENERATE
Find the frequencies of a sum-of-sinusoids generator for a given ACF

- Searches `n_coeff` frequencies such that the autocorrelation function (ACF) of the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, approximates the target ACF in all directions
- Each sinusoid has a fixed direction and a signed frequency along it. `n_dim = 1`: no direction;
  `n_dim = 2`: directions on the half circle; `n_dim = 3`: directions on the upper hemisphere.
  `uniform_smp = true` places them regularly (equally spaced azimuth, spiral lattice, as the test
  directions of [[sos_calc_mse]]), `false` draws them at random
- Iteration: in each pass every sinusoid is re-fitted once. Its frequency is chosen to fit the
  residual between the target and the contribution of all other sinusoids along its own direction:
  coarse search over a grid with a spacing of `0.1 / max(dist)`, then a step refinement. The new
  frequency is kept only if the mean squared error over the test directions decreases. The
  iteration stops after a pass that changes nothing or lowers the error by less than a factor of
  1e-6, or after 1000 passes
- `random_init = true` starts from uniformly random frequencies; `false` builds the set up
  sinusoid by sinusoid in the first pass
- The fit covers the range of `dist` only. The ACF of the result beyond the last sample point is
  not controlled; sinusoids with a frequency near zero keep it above zero at all larger distances
  (see `acf_tail` of [[sos_calc_mse]]). To prevent this, set `freq_min`
- `freq_min` and `freq_max` limit the magnitude of the frequency along the direction of each
  sinusoid; 0 selects the default: no lower limit, and `15 / max(dist)` as upper limit
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse
  in others; the predefined tables of [[sos_preset]] were made with 300
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject
  decisions, so a difference in the last bit of the input or of the math library can lead to
  another table of equivalent quality

## Usage:
```
[ freq, ase, test_dir ] = quadriga_lib.sos_generate( acf, dist, n_coeff, n_dim, uniform_smp, ...
    n_dir, random_init, freq_min, freq_max, seed );
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0 and increase strictly; `[n_acf]`
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_dim`** — Number of dimensions; 1, 2 or 3; default: 3
- **`uniform_smp`** — Regular (true) or random (false) directions of the sinusoids; default: true
- **`n_dir`** — Number of test directions; must be > 0; ignored for `n_dim = 1`; default: 10
- **`random_init`** — Start from random frequencies (true) or build up sinusoid by sinusoid
  (false); default: true
- **`freq_min`** — Lower limit of the frequency magnitude in [1/m]; must be >= 0; 0 = none;
  default: 0
- **`freq_max`** — Upper limit of the frequency magnitude in [1/m]; must be >= 0 and, if not 0,
  larger than `freq_min`; 0 = `15 / max(dist)`; default: 0
- **`seed`** — Seed; must be >= -1; -1 seeds from the system random device; default: -1

## Outputs:
- **`freq`** — Frequencies in [1/m]; `[n_coeff, n_dim]`
- **`ase`** — Mean squared error over the test directions after each pass, linear scale; column
  vector; `[n_pass]`
- **`test_dir`** — Test directions in [rad]; `[n_dir, 2]` with columns `{azimuth, elevation}` for
  `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

## See also:
- [[sos_calc_mse]] (error of a frequency table)
- [[sos_preset]] (precomputed tables)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 3 || nrhs > 10)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::vec acf = qd_mex_get_Col<double>(prhs[0]);
    const arma::vec dist = qd_mex_get_Col<double>(prhs[1]);
    const arma::uword n_coeff = qd_mex_get_scalar<arma::uword>(prhs[2], "n_coeff", 0);
    const arma::uword n_dim = (nrhs < 4) ? 3 : qd_mex_get_scalar<arma::uword>(prhs[3], "n_dim", 3);
    const bool uniform_smp = (nrhs < 5) ? true : qd_mex_get_scalar<bool>(prhs[4], "uniform_smp", true);
    const arma::uword n_dir = (nrhs < 6) ? 10 : qd_mex_get_scalar<arma::uword>(prhs[5], "n_dir", 10);
    const bool random_init = (nrhs < 7) ? true : qd_mex_get_scalar<bool>(prhs[6], "random_init", true);
    const double freq_min = (nrhs < 8) ? 0.0 : qd_mex_get_scalar<double>(prhs[7], "freq_min", 0.0);
    const double freq_max = (nrhs < 9) ? 0.0 : qd_mex_get_scalar<double>(prhs[8], "freq_max", 0.0);
    const arma::sword seed = (nrhs < 10) ? -1 : qd_mex_get_scalar<arma::sword>(prhs[9], "seed", -1);

    // Wrap optional pointers; the number of passes is known only after the call
    arma::mat freq, test_dir;
    arma::vec ase;
    arma::vec *p_ase = (nlhs > 1) ? &ase : nullptr;
    arma::mat *p_test_dir = (nlhs > 2) ? &test_dir : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::sos_generate<double>(acf, dist, n_coeff, freq, p_ase, p_test_dir, n_dim, uniform_smp, n_dir, random_init, freq_min, freq_max, seed));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&freq);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&ase);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&test_dir);
}
