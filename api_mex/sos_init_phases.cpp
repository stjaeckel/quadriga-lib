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
# SOS_INIT_PHASES
Draw random phases for a set of sum-of-sinusoids generators

- Each column of `phase` is one realisation of the random field of a generator; see [[sos_val]]
- Values are uniform in the interval from -pi (included) to pi (excluded)
- Random stream: 64-bit Mersenne Twister (`std::mt19937_64`) seeded with `seed`; each value is
  `2·pi · ( u - 0.5 )`, where `u` is the upper 53 bits of one generator output divided by 2^53.
  A seed >= 0 gives the same phases on all platforms; MATLAB's own random stream is not used
- The matrix is filled column by column from one stream: for the same seed, column `g` does not
  depend on `n_gen`

## Usage:
```
phase = quadriga_lib.sos_init_phases( n_coeff, n_gen, seed );
```

## Inputs:
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_gen`** — Number of generators; must be > 0; default: 1
- **`seed`** — Seed of the random stream; must be >= -1; -1 seeds from the system random device and
  gives different phases on each call; default: -1

## Output:
- **`phase`** — Phases in [rad]; `[n_coeff, n_gen]`

## See also:
- [[sos_val]] (evaluates the generators)
- [[sos_preset]] (frequencies that belong to the phases)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::uword n_coeff = qd_mex_get_scalar<arma::uword>(prhs[0], "n_coeff", 0);
    const arma::uword n_gen = (nrhs < 2) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[1], "n_gen", 1);
    const arma::sword seed = (nrhs < 3) ? -1 : qd_mex_get_scalar<arma::sword>(prhs[2], "seed", -1);

    // Output allocation (a zero size is left to the library to reject)
    arma::mat phase;
    if (nlhs > 0 && n_coeff != 0 && n_gen != 0)
        plhs[0] = qd_mex_init_output(&phase, n_coeff, n_gen);

    // Call library function
    CALL_QD(quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, phase, seed));
}
