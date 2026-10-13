// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

#include <cmath>
#include <string>

// Seed: a floating-point value is checked before the conversion to an integer type, so that NaN and Inf are
// rejected; non-integer values are truncated. Integer classes are read as they are (seeds above 2^53 are
// exact only as int64 or uint64). Values below -1 are left to the library to reject.
static arma::sword qd_mex_sos_seed(const mxArray *input)
{
    if (mxIsDouble(input) || mxIsSingle(input))
    {
        const double v = qd_mex_get_scalar<double>(input, "seed", -1.0);
        if (!(v > -9.0e18) || !(v < 9.0e18))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'seed' must be a finite number >= -1.");
        return (arma::sword)v;
    }
    return qd_mex_get_scalar<arma::sword>(input, "seed", -1);
}

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# SOS_RAND
Spatially correlated Uniform distributed random numbers

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [[sos_val]] for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second
  device on its own map uses column `g` of a second stream,
  `sos_init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62 flipped, i.e.
  `seed + 2^62` for seeds below 2^62 (exact only as `int64`); for `seed = -1` both streams are
  random. The phases of a generator therefore depend neither on the number of generators nor on
  whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr`
  for that generator): both devices are on the same map, the value does not change when they
  swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the
  sum of both maps divided by `sqrt(2)`
- The Normal distributed value `s` of the generator is mapped to `0.5 · erfc( -s / sqrt(2) )`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [[sos_val]]. MATLAB's own random stream is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent

## Usage:
```
val = quadriga_lib.sos_rand( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed );
```

## Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device; requires
  `pos_b`; must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** — Name of the preset, see [[sos_preset]]; string; default: 'Comb256'
- **`seed`** — Seed of the phases; must be >= -1; -1 seeds from the system random device;
  default: -1

## Output:
- **`val`** — Values, approximately Uniform distributed between 0 and 1; one row per position, one
  column per generator; `[n_pos, n_gen]`

## See also:
- [[sos_val]] (evaluation with given frequencies and phases)
- [[sos_preset]], [[sos_init_phases]] (the two steps this function combines)
- [[sos_randn]], [[sos_randi]] (Normal and integer values)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 2 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::vec dist_decorr = qd_mex_get_Col<double>(prhs[0]);
    const arma::mat pos = qd_mex_get_Mat<double>(prhs[1]);
    const arma::mat pos_b = (nrhs < 3) ? arma::mat() : qd_mex_get_Mat<double>(prhs[2]);
    const arma::vec dist_decorr_b = (nrhs < 4) ? arma::vec() : qd_mex_get_Col<double>(prhs[3]);
    const std::string preset = (nrhs < 5) ? std::string("Comb256") : qd_mex_get_string(prhs[4], "Comb256");
    const arma::sword seed = (nrhs < 6) ? -1 : qd_mex_sos_seed(prhs[5]);

    // Output allocation
    arma::mat val;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&val, pos.n_rows, dist_decorr.n_elem);

    // Wrap optional pointers
    const arma::mat *p_pos_b = pos_b.empty() ? nullptr : &pos_b;
    const arma::vec *p_dist_decorr_b = dist_decorr_b.empty() ? nullptr : &dist_decorr_b;

    // Call library function
    CALL_QD(quadriga_lib::sos_rand<double>(dist_decorr, pos, val, p_pos_b, p_dist_decorr_b, preset, seed));
}
