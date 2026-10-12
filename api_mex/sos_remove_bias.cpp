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
- One device (`pos_b` omitted): one offset per generator
- Two devices on the same map (`pos_b` given, `phase_b_in` omitted): one offset per generator that
  zeroes the mean of the combined, normalised value; `acf` and `dist` are required
- Two devices on different maps (`pos_b` and `phase_b_in` given): each device is treated on its
  own, `phase_in` with `pos` and `phase_b_in` with `pos_b` and the frequencies
  `freq · freq_scale_b`; two offsets per generator. The mean of the combined value is then zero
  as well
- Effect on the statistics: the result is the field conditioned on a zero mean over the positions.
  The variance at a position drops from 1 to `1 - c^2 / v`, where `v` is the variance of the
  sample mean and `c` the mean ACF between that position and all positions; on average over the
  positions the loss is at least `v`. It is small only if the positions spread over many
  decorrelation distances. For positions within about one decorrelation distance most of the
  variance is removed
- The zero mean holds for exactly these positions and frequencies. It is lost when the phases are
  used with other frequencies (another decorrelation distance) or positions
- Phases are returned in the interval from -pi (included) to pi (excluded)

## Usage:
```
[ phase_out, phase_b_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( freq, phase_in, ...
    pos, pos_b, phase_b_in, acf, dist, freq_scale_b );
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase_in`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos_b, 1]`, `[n_pos_b, 2]` or `[n_pos_b, 3]` as for `pos`; `n_pos_b = n_pos` for the same map,
  any number for different maps
- **`phase_b_in`** *(optional)* — Phases of the second device in [rad]; requires `pos_b`; omitted
  or empty = same map; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF; must start at 0 and increase strictly;
  `[n_acf]`
- **`freq_scale_b`** — Frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; other than 1 requires `phase_b_in`;
  default: 1

## Outputs:
- **`phase_out`** — Corrected phases in [rad]; `[n_coeff, n_gen]`
- **`phase_b_out`** — Corrected phases of the second device in [rad]; `[n_coeff, n_gen]`; empty if
  `phase_b_in` was not given
- **`offset`** — Offsets added to the phases in [rad]; `[n_gen, 1]`, or `[n_gen, 2]` with
  `phase_b_in` (column 1: `phase_in`, column 2: `phase_b_in`)
- **`mean_before`** — Sample mean before the correction; `[n_gen, 1]`: mean of the values of
  [[sos_val]] for the given arguments; with `phase_b_in` `[n_gen, 2]`: mean of the field of each
  device on its own positions, the mean of the combined value being their sum divided by `sqrt(2)`

## See also:
- [[sos_val]] (the values whose mean is removed)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 3 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; the phases are modified by the library and therefore copied
    const arma::mat freq = qd_mex_get_Mat<double>(prhs[0]);
    arma::mat phase = qd_mex_get_Mat<double>(prhs[1], true);
    const arma::mat pos = qd_mex_get_Mat<double>(prhs[2]);
    const arma::mat pos_b = (nrhs < 4) ? arma::mat() : qd_mex_get_Mat<double>(prhs[3]);
    arma::mat phase_b = (nrhs < 5) ? arma::mat() : qd_mex_get_Mat<double>(prhs[4], true);
    const arma::vec acf = (nrhs < 6) ? arma::vec() : qd_mex_get_Col<double>(prhs[5]);
    const arma::vec dist = (nrhs < 7) ? arma::vec() : qd_mex_get_Col<double>(prhs[6]);
    const double freq_scale_b = (nrhs < 8) ? 1.0 : qd_mex_get_scalar<double>(prhs[7], "freq_scale_b", 1.0);

    // Wrap optional pointers
    arma::mat offset, mean_before;
    const arma::mat *p_pos_b = pos_b.empty() ? nullptr : &pos_b;
    arma::mat *p_phase_b = phase_b.empty() ? nullptr : &phase_b;
    const arma::vec *p_acf = acf.empty() ? nullptr : &acf;
    const arma::vec *p_dist = dist.empty() ? nullptr : &dist;
    arma::mat *p_offset = (nlhs > 2) ? &offset : nullptr;
    arma::mat *p_mean_before = (nlhs > 3) ? &mean_before : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::sos_remove_bias<double>(freq, phase, pos, p_pos_b, p_phase_b, p_acf, p_dist, p_offset, p_mean_before, freq_scale_b));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&phase);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&phase_b);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&offset);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&mean_before);
}
