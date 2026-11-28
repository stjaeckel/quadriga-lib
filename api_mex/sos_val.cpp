// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!SECTION_DESC
These functions generate **spatially correlated random numbers**: values that are random, but similar at
nearby positions. Two positions that are much closer than the *decorrelation distance* get almost the same
value, two positions that are far apart get independent values. The decorrelation distance is the distance
at which the correlation has fallen to `exp(-1)`, about 0.37. Typical uses are shadow fading maps and the
spatially consistent parameters of the 3GPP TR 38.901 channel model.<br><br>

The values are a sum of sinusoids, `s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n )`. The frequencies
`f_n` set how the correlation falls with the distance, i.e. the autocorrelation function (ACF). The random
phases `phi_n` select one realisation of the random field. The generator needs no map and no grid:
positions can be anywhere, in 1, 2 or 3 dimensions, and the same frequencies and phases always give the
same field.<br><br>

**Quick start**<br>
[[sos_randn]] sets up the generators, draws the phases and returns the values in one call. [[sos_rand]]
and [[sos_randi]] do the same for Uniform and integer values.

```
% 1000 positions; one row per position, columns are [x, y, z] in meters
pos = rand(1000, 3) * 100;

% One independent generator per decorrelation distance: 10 m and 50 m
dist_decorr = [10, 50];

% Normal distributed values with zero mean and unit variance, size [1000, 2]
val = quadriga_lib.sos_randn(dist_decorr, pos, [], [], 'Comb256', 42);

% Uniform values between 0 and 1, and integers from 1 to 6
uni  = quadriga_lib.sos_rand(dist_decorr, pos, [], [], 'Comb256', 42);
dice = quadriga_lib.sos_randi(dist_decorr, pos, 6, [], [], 'Comb256', 42);
```

The last argument is the seed. With the same seed, decorrelation distances and preset, a later call returns the same
field, also at other positions. Without a seed every call gives a new field. The functions have their own random
stream; the state of `rand` and `randn` in MATLAB or Octave has no effect on them.<br><br>

A seed selects a set of phases, not a field. Calls with the same seed but another decorrelation distance or
another preset return related fields: they are equal at the coordinate origin and correlated around it.
Quantities that must be independent of each other, such as shadow fading and delay spread, therefore need
different seeds, or one call with one entry of `dist_decorr` per quantity: the generators of one call are
independent.<br><br>
**Keeping a generator**<br>
A generator is two matrices: the frequencies from [[sos_preset]] and the phases from [[sos_init_phases]].
[[sos_val]] evaluates them. Each column of the phase matrix is one independent generator. Evaluating many
generators in one call is much cheaper than one call per generator, because the sines and cosines of the
positions are shared: 20 generators take less than twice as long as one.

```
% Frequencies of a predefined generator for a decorrelation distance of 10 m
[spatial_freq, acf, dist] = quadriga_lib.sos_preset('Comb256', 10);

% Phases of 20 generators; each column is one independent random field
phase = quadriga_lib.sos_init_phases(size(spatial_freq, 1), 20, 42);

% Values of all 20 generators at the positions, size [1000, 20]
val_all = quadriga_lib.sos_val(spatial_freq, phase, pos);

% The same fields at other positions, at any later time
pos_new = pos + 2.5;
val_new = quadriga_lib.sos_val(spatial_freq, phase, pos_new);

% Uniform distributed output
val_uni = quadriga_lib.sos_val(spatial_freq, phase, pos, [], [], [], true);

% Map on a regular grid, size [n_y, n_x, 1, 20]
x = 0:0.5:250;  y = 0:0.5:100;
grid = quadriga_lib.sos_map(spatial_freq, phase, x, y);
imagesc(x, y, grid(:, :, 1, 1));  axis xy;
```

- **Positions:** one row per position, `[n_pos, 3]`. Results have one row per position and one column per
  generator, `[n_pos, n_gen]`.
- **Presets:** `Exp`, `Gauss` and `Comb`, each with 128, 256, 512 or 1024 sinusoids, e.g. `'Exp512'`. `Exp`
  is the exponential ACF `exp(-d/d0)`, `Gauss` is `exp(-(d/d0)^2)`, and `Comb` follows `Gauss` up to the
  decorrelation distance `d0` and `Exp` beyond it. Each doubling of the number of sinusoids doubles the
  computing time and lowers the error of the ACF by 3 to 4 dB: [[sos_calc_mse]] reports 28, 31, 34 and
  38 dB for `Exp128` to `Exp1024` and 31, 35, 39 and 42 dB for `Comb128` to `Comb1024`. `'Comb256'` is a
  good default; take a larger table if the shape of the ACF matters, a smaller one if speed does.
- **Both ends mobile:** if a value belongs to a link between two moving devices, pass both positions.
  The value then changes smoothly when either device moves, keeps the correlation of one device while the
  other stays, and does not change when the two ends swap.

```
% Positions of the two ends of 1000 links; one value per link and generator, size [1000, 20]
pos_tx = pos;
pos_rx = rand(1000, 3) * 100;
val_link = quadriga_lib.sos_val(spatial_freq, phase, pos_tx, pos_rx);
```

- **Mean over an area:** the values at positions that lie within a few decorrelation distances of each
  other are similar, so their average is not zero for a single realisation. This is how a correlated
  field behaves. [[sos_remove_bias]] can force the average to zero if an application needs it. This
  removes variance: for 200 positions in a square with an edge of 20 decorrelation distances 2.5 %, for
  an edge of 5 decorrelation distances 14 %, for 2 already 44 %. It is meant for positions that spread
  over many decorrelation distances.
- **Coordinates:** use local coordinates. The argument `f_n·x` of a sinusoid is computed in double
  precision and is known to about 1e-16 of its size, so the values lose accuracy with the distance from
  the origin counted in wavelengths. Up to 1e9 wavelengths (more than 1e9 m for a decorrelation distance
  of 10 m) the results stay within the tolerance stated in [[sos_val]].
- **Threads and memory:** the functions that take noticeable time use the OpenMP threads of the process,
  except [[sos_generate]], which runs on one thread. No result depends on the number of threads,
  except the last digits of the errors of [[sos_calc_mse]].
  [[sos_val]] computes in its output: it needs no further memory of that size for one device and one more
  array of that size for two devices.

**Own autocorrelation functions**<br>
[[sos_generate]] finds the frequencies for any ACF that is given as a table over the distance, and
[[sos_calc_mse]] reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```
% Target ACF from 0 to 5 decorrelation distances, here a Gaussian shape with 5 m
dist_own = 0:0.125:24.875;
acf_own  = exp(-(dist_own / 5).^2);

% 256 sinusoids in 3-D, 300 test directions, wavelengths up to 20 decorrelation distances (100 m)
freq_own = quadriga_lib.sos_generate(acf_own, dist_own, 256, 3, true, 300, true, 0.01, 0, 1);

% Approximation error in dB, larger is better; about 35 dB for this example
mse = quadriga_lib.sos_calc_mse(freq_own, acf_own, dist_own);
```

The result is used like a preset: draw phases with [[sos_init_phases]] and evaluate with [[sos_val]]. To
change the decorrelation distance afterwards, divide the frequencies and multiply the distances by the
same factor.<br><br>

**Reference**<br>
S. Jaeckel, L. Raschkowski, F. Burkhardt, L. Thiele; "Efficient Sum-of-Sinusoids-Based Spatial Consistency
for the 3GPP New-Radio Channel Model"; 2018 IEEE Globecom Workshops (GC Wkshps); 2018; Available:
<a target="_blank" rel="noopener noreferrer" href="https://ieeexplore.ieee.org/document/8644265">IEEE Xplore</a> (Original version),
<a target="_blank" rel="noopener noreferrer" href="publications/2018_sum_of_sinusoids.pdf">PDF version</a>,
<a target="_blank" rel="noopener noreferrer" href="publications/2018_sum_of_sinusoids.md">Markdown version</a><br><br>
SECTION_DESC!*/

/*!MD
# SOS_VAL
Spatially correlated random values at given positions

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of
  `spatial_freq`, in cycles per meter) and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function
  (ACF) `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the
  frequencies were fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The
  sines and cosines of the positions are computed once and reused for all generators
- Positions are given as one row per position with the columns `{x, y, z}`; they may have fewer
  columns than `spatial_freq`; missing coordinates are 0
- One device (`pos_b` omitted): `val = s(pos)`
- Two devices (`pos_b` given): one generator over the positions of both,
  `s(x, y) = sqrt(2/n_coeff) · sum_n cos( 2·pi · (f_n·x + b·f_p(n)·y) + phi_n )`, with
  `b = spatial_freq_scale_b` and `p` a fixed pseudo-random permutation of the sinusoids that depends only on
  `n_coeff` and pairs no sinusoid with itself. The correlation of two links approximates the product of the
  ACFs of the two moves; while one device stays at its position, it is the ACF of one device (with
  `reciprocal = true` for devices more than about one decorrelation distance apart)
- `reciprocal = true` (requires `b = 1`): `val = ( s(pos, pos_b) + s(pos_b, pos) ) / sqrt( 2 + 2·c )` with
  `c = (1/n_coeff) · sum_n cos( 2·pi · (f_n - f_p(n))·(pos - pos_b) )`. The value does not change when the
  two devices swap their positions; for devices closer than about one decorrelation distance the
  correlation of two links contains the term this symmetry requires
- `reciprocal = false`: `val = s(pos, pos_b)`. With `pos` and `pos_b` exchanged, the same phases give a
  second field with the same statistics, for example the arrival value of a link whose departure value
  is `val`
- For large moves of both devices the correlation keeps a residual with rms `1/sqrt(2·n_coeff)`
- With `reciprocal = true`, a position where `c = -1` (possible only for tables with few sinusoids) gives NaN
- `uniform = true` maps the values to `0.5 · erfc( -val / sqrt(2) )`, which is uniform between 0 and
  1 as far as `val` is Normal distributed. The limits 0 and 1 are included: a value far out in the
  tail is rounded to them
- Kernels: GENERIC computes in double precision. AVX2 evaluates sine and cosine in single
  precision; its values differ from GENERIC by less than `4e-7 · sqrt( 2·n_coeff )`, typically by
  less than 1e-6; with two devices and `reciprocal = true` the difference is scaled by
  `2 / sqrt( 2 + 2·c )`, at most about 1.7 for the presets
- Accuracy far from the origin: the argument `f_n·x` is known to about 1e-16 of its size. The
  bound between the kernels holds up to about 1e9 periods of the sinusoids; beyond that both
  kernels lose accuracy at the same rate
- A NaN or infinite coordinate gives NaN at that position and does not affect other positions
- The value at a position does not depend on the other positions or generators of the call, nor
  on the number of threads

## Usage:
```
val = quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b, spatial_freq_scale_b, reciprocal, ...
    uniform, use_kernel );
```

## Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `spatial_freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`spatial_freq_scale_b`** — Spatial-frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; other than 1 requires `pos_b` and
  `reciprocal = false`; default: 1
- **`reciprocal`** — Two devices: true = the value does not change when the devices swap positions;
  default: true
- **`uniform`** — Output distribution: false = Normal(0,1), true = Uniform between 0 and 1;
  default: false
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2 (error if
  unavailable); default: 0

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
    if (nrhs < 3 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const arma::mat spatial_freq = qd_mex_get_Mat<double>(prhs[0], false, "spatial_freq");
    const arma::mat phase = qd_mex_get_Mat<double>(prhs[1], false, "phase");
    const arma::mat pos = qd_mex_get_Mat<double>(prhs[2], false, "pos");
    const arma::mat pos_b = (nrhs < 4) ? arma::mat() : qd_mex_get_Mat<double>(prhs[3], false, "pos_b");
    const double spatial_freq_scale_b = (nrhs < 5) ? 1.0 : qd_mex_get_scalar<double>(prhs[4], "spatial_freq_scale_b", 1.0);
    const bool reciprocal = (nrhs < 6) ? true : qd_mex_get_scalar<bool>(prhs[5], "reciprocal", true);
    const bool uniform = (nrhs < 7) ? false : qd_mex_get_scalar<bool>(prhs[6], "uniform", false);
    const int use_kernel = (nrhs < 8) ? 0 : qd_mex_get_scalar<int>(prhs[7], "use_kernel", 0);

    arma::mat val;
    plhs[0] = qd_mex_init_output(&val, pos.n_rows, phase.n_cols);

    CALL_QD(quadriga_lib::sos_val<double>(spatial_freq, phase, pos, val, pos_b, spatial_freq_scale_b, reciprocal, uniform, use_kernel));
}
