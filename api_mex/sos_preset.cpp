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
# SOS_PRESET
Frequencies and ACF of a predefined sum-of-sinusoids generator

- Returns a precomputed frequency table for [[sos_val]] together with the autocorrelation
  function (ACF) it was fitted to
- Presets: 'Exp128', 'Exp256', 'Exp512', 'Exp1024', 'Gauss128', 'Gauss256', 'Gauss512',
  'Gauss1024', 'Comb128', 'Comb256', 'Comb512', 'Comb1024'. The number is the number of
  sinusoids, `n_coeff`. The name is matched exactly, including upper and lower case
- ACF types, with the distance `D` and the decorrelation distance `d`: `Exp`: `R = exp( -D/d )`;
  `Gauss`: `R = exp( -(D/d)^2 )`; `Comb`: `Gauss` for `D <= d`, `Exp` beyond.
  All three have `R(d) = exp(-1)`
- The ACF is returned as a table with 200 samples from 0 to `4.975·d` in steps of `0.025·d`
- The tables are stored for `d = 10 m`. For another decorrelation distance the distances are
  multiplied by `d/10` and the frequencies by `10/d`
- All tables are 3-dimensional and can be used with positions of 1, 2 or 3 columns
- No sinusoid has a wavelength above 20 decorrelation distances. The ACF of the tables stays near
  zero beyond the fitted range, and the mean of the generator over a large area is not offset by
  sinusoids of nearly zero frequency

## Usage:
```
[ freq, acf, dist ] = quadriga_lib.sos_preset( name, dist_decorr );
```

## Inputs:
- **`name`** — Name of the preset, e.g. 'Comb256'; string
- **`dist_decorr`** — Decorrelation distance; must be > 0; default: 10

## Outputs:
- **`freq`** — Frequencies; `[n_coeff, 3]`
- **`acf`** — Target ACF; `[200]`
- **`dist`** — Sample points of the ACF; `[200]`

## See also:
- [[sos_init_phases]] (random phases for the generator)
- [[sos_val]] (evaluation)
- [[sos_generate]] (frequencies for other ACFs)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string name = qd_mex_get_string(prhs[0]);
    const double dist_decorr = (nrhs < 2) ? 10.0 : qd_mex_get_scalar<double>(prhs[1], "dist_decorr", 10.0);

    // Output allocation; the number of sinusoids is known only after the call
    arma::mat freq;
    arma::vec acf, dist;
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&acf, 200);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&dist, 200);

    // Wrap optional pointers
    arma::vec *p_acf = acf.empty() ? nullptr : &acf;
    arma::vec *p_dist = dist.empty() ? nullptr : &dist;

    // Call library function
    CALL_QD(quadriga_lib::sos_preset<double>(name, freq, p_acf, p_dist, dist_decorr));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&freq);
}
