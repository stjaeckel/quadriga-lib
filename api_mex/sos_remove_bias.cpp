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
# SOS_REMOVE_BIAS
Remove the sample mean of sum-of-sinusoids generators over a set of positions

- One realisation of a spatially correlated random field has a non-zero mean over a finite set of
  positions. Its variance over the realisations is the mean of the autocorrelation function (ACF)
  over all pairs of positions, which is much larger than `1 / n_pos` when the positions lie within
  a few decorrelation distances of each other. This is a property of any correlated field, not an
  error of the generator
- This function adds one offset to all phases of a generator such that the mean of the Normal
  distributed values of [[sos_val]] over the given positions becomes zero
- Closed form, no iteration: with `z_n = mean over the positions of exp( j·2·pi · f_n·x )` and
  `Z = sum_n z_n · exp( j·phi_n )`, the mean is `sqrt(2/n_coeff) · Re( exp( j·offset ) · Z )`,
  which vanishes for `offset = pi/2 - arg(Z)` and for that value minus pi; the one with the
  smaller magnitude is applied, so the offset lies between -pi/2 and pi/2
- One offset per generator; with two devices (`pos_b` given) it zeroes the mean of the two-device value
  of [[sos_val]] with the same `spatial_freq_scale_b` and `reciprocal`
- Two reciprocal devices at a position where the normalization of [[sos_val]] is singular (`c = -1`) throw
- Effect on the statistics: the result is the field conditioned on a zero mean over the positions.
  The variance at a position drops from 1 to `1 - c^2 / v`, where `v` is the variance of the
  sample mean and `c` the mean ACF between that position and all positions; on average over the
  positions the loss is at least `v`. It is small only if the positions spread over many
  decorrelation distances. For positions within about one decorrelation distance most of the
  variance is removed
- The zero mean holds for exactly these positions and frequencies. It is lost when the phases are
  used with other frequencies (another decorrelation distance) or positions
- Rule of thumb, measured for 200 positions in a square with the edge `L` and the decorrelation
  distance `d` (`Comb256`): the variance drops by 74 % for `L = d`, 44 % for `L = 2·d`, 14 % for
  `L = 5·d`, 5 % for `L = 10·d`, 2.5 % for `L = 20·d` and less than 1 % from `L = 50·d` on
- Phases are returned in the interval from -pi (included) to pi (excluded). A generator whose
  `abs(Z)` is below `1e-14 · n_coeff` already has a zero mean for every offset and is returned
  unchanged, also if its phases lie outside that interval
- The C++ in/out argument `phase` is split into `phase_in` and `phase_out`, so that the input stays
  unchanged; C++ messages name it `phase`

## Usage:
```
[ phase_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( spatial_freq, phase_in, pos, pos_b, ...
    spatial_freq_scale_b, reciprocal );
```

## Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase_in`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `spatial_freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`
- **`spatial_freq_scale_b`** — Spatial-frequency scale of the second device as in [[sos_val]]; must be > 0;
  other than 1 requires `pos_b` and `reciprocal = false`; default: 1
- **`reciprocal`** — Two devices: the mode of [[sos_val]]; default: true

## Outputs:
- **`phase_out`** — Corrected phases; `[n_coeff, n_gen]`
- **`offset`** — Offsets added to the phases; `[n_gen, 1]`
- **`mean_before`** — Sample mean of the values of [[sos_val]] for the given arguments before the
  correction; `[n_gen, 1]`

## See also:
- [[sos_val]] (the values whose mean is removed)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; the phases are modified by the library and therefore copied
    const arma::mat spatial_freq = qd_mex_get_Mat<double>(prhs[0], false, "spatial_freq");
    arma::mat phase = qd_mex_get_Mat<double>(prhs[1], true, "phase_in");
    const arma::mat pos = qd_mex_get_Mat<double>(prhs[2], false, "pos");
    const arma::mat pos_b = (nrhs < 4) ? arma::mat() : qd_mex_get_Mat<double>(prhs[3], false, "pos_b");
    const double spatial_freq_scale_b = (nrhs < 5) ? 1.0 : qd_mex_get_scalar<double>(prhs[4], "spatial_freq_scale_b", 1.0);
    const bool reciprocal = (nrhs < 6) ? true : qd_mex_get_scalar<bool>(prhs[5], "reciprocal", true);

    arma::mat offset, mean_before;
    arma::mat *p_offset = (nlhs > 1) ? &offset : nullptr;
    arma::mat *p_mean_before = (nlhs > 2) ? &mean_before : nullptr;

    CALL_QD(quadriga_lib::sos_remove_bias<double>(spatial_freq, phase, pos, pos_b, spatial_freq_scale_b, reciprocal, p_offset, p_mean_before));

    plhs[0] = qd_mex_copy2matlab(&phase);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&offset);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&mean_before);
}
