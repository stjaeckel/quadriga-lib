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
# SOS_VAL
Spatially correlated random values at given positions

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of
  `freq`) and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function
  (ACF) `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the
  frequencies were fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The
  sines and cosines of the positions are computed once and reused for all generators
- Positions are given as one row per position with the columns `{x, y, z}`; they may have fewer
  columns than `freq`; missing coordinates are 0
- One device (`pos_b` omitted): `val = s(pos)`
- Two devices on the same map (`pos_b` given, `phase_b` omitted): both devices use `phase`;
  `val = ( s(pos) + s(pos_b) ) / sqrt( 2 + 2·A(D) )`, where `D` is the distance between `pos` and
  `pos_b` and `A` the target ACF interpolated as in [[sos_acfi]]; `acf` and `dist` are required.
  The value does not change when the two devices swap their positions
- Two devices on different maps (`pos_b` and `phase_b` given):
  `val = ( s_a(pos) + s_b(pos_b) ) / sqrt(2)`; `s_b` uses `phase_b` and the frequencies
  `freq · freq_scale_b`; `acf` and `dist` are ignored
- The mode follows from the arguments only; phase values are never compared. Passing the same
  matrix as `phase` and `phase_b` selects the two-map formula
- `freq_scale_b` other than 1 requires `phase_b`: one map with two decorrelation distances has no
  consistent ACF
- `uniform = true` maps the values to `0.5 · erfc( -val / sqrt(2) )`
- Kernels: GENERIC computes in double precision. AVX2 evaluates sine and cosine in single
  precision; its values differ from GENERIC by less than `4e-7 · sqrt( 2·n_coeff )`, at any
  distance from the origin
- The value at a position does not depend on the other positions or generators of the call

## Usage:
```
val = quadriga_lib.sos_val( freq, phase, pos, pos_b, phase_b, acf, dist, freq_scale_b, ...
    uniform, use_kernel );
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`phase_b`** *(optional)* — Phases of the second device in [rad]; requires `pos_b`; omitted or
  empty = same map; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF; must start at 0 and increase strictly;
  `[n_acf]`
- **`freq_scale_b`** — Frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; default: 1
- **`uniform`** — Output distribution: false = Normal(0,1), true = Uniform between 0 and 1;
  default: false
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  default: 0

## Output:
- **`val`** — Values; one row per position, one column per generator; `[n_pos, n_gen]`

## See also:
- [[sos_preset]] (frequencies and ACF of the predefined models)
- [[sos_init_phases]] (random phases)
- [[sos_map]] (evaluation on a grid)
- [[sos_randn]] (generator set-up and evaluation in one call)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 3 || nrhs > 10)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::mat freq = qd_mex_get_Mat<double>(prhs[0]);
    const arma::mat phase = qd_mex_get_Mat<double>(prhs[1]);
    const arma::mat pos = qd_mex_get_Mat<double>(prhs[2]);
    const arma::mat pos_b = (nrhs < 4) ? arma::mat() : qd_mex_get_Mat<double>(prhs[3]);
    const arma::mat phase_b = (nrhs < 5) ? arma::mat() : qd_mex_get_Mat<double>(prhs[4]);
    const arma::vec acf = (nrhs < 6) ? arma::vec() : qd_mex_get_Col<double>(prhs[5]);
    const arma::vec dist = (nrhs < 7) ? arma::vec() : qd_mex_get_Col<double>(prhs[6]);
    const double freq_scale_b = (nrhs < 8) ? 1.0 : qd_mex_get_scalar<double>(prhs[7], "freq_scale_b", 1.0);
    const bool uniform = (nrhs < 9) ? false : qd_mex_get_scalar<bool>(prhs[8], "uniform", false);
    const int use_kernel = (nrhs < 10) ? 0 : qd_mex_get_scalar<int>(prhs[9], "use_kernel", 0);

    // Output allocation
    arma::mat val;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&val, pos.n_rows, phase.n_cols);

    // Wrap optional pointers
    const arma::mat *p_pos_b = pos_b.empty() ? nullptr : &pos_b;
    const arma::mat *p_phase_b = phase_b.empty() ? nullptr : &phase_b;
    const arma::vec *p_acf = acf.empty() ? nullptr : &acf;
    const arma::vec *p_dist = dist.empty() ? nullptr : &dist;

    // Call library function
    CALL_QD(quadriga_lib::sos_val<double>(freq, phase, pos, val, p_pos_b, p_phase_b, p_acf, p_dist, freq_scale_b, uniform, use_kernel));
}
