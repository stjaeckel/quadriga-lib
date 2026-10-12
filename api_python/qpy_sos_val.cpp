// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

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

The functions are in the submodule `quadriga_lib.sos` and are called without the prefix `sos_` that their
names have in this documentation: [[sos_val]] is `quadriga_lib.sos.val`. In the examples, `np` is
numpy.<br><br>

**Quick start**<br>
[[sos_randn]] sets up the generators, draws the phases and returns the values in one call. [[sos_rand]]
and [[sos_randi]] do the same for Uniform and integer values.

```
# 1000 positions; one row per position, columns are (x, y, z) in meters
pos = np.random.default_rng(0).random((1000, 3)) * 100

# One independent generator per decorrelation distance: 10 m and 50 m
dist_decorr = [10, 50]

# Normal distributed values with zero mean and unit variance, shape (1000, 2)
val = quadriga_lib.sos.randn(dist_decorr, pos, seed=42)

# Uniform values between 0 and 1, and integers from 1 to 6
uni = quadriga_lib.sos.rand(dist_decorr, pos, seed=42)
dice = quadriga_lib.sos.randi(dist_decorr, pos, 6, seed=42)
```

The argument `seed` is the seed. With the same seed, decorrelation distances and preset, a later call returns the same
field, also at other positions. Without a seed every call gives a new field. The functions have their own
random stream; the random state of numpy has no effect on them.<br><br>

A seed selects a set of phases, not a field. Calls with the same seed but another decorrelation distance or
another preset return related fields: they are equal at the coordinate origin and correlated around it.
Quantities that must be independent of each other, such as shadow fading and delay spread, therefore need
different seeds, or one call with one entry of `dist_decorr` per quantity: the generators of one call are
independent.<br><br>
**Keeping a generator**<br>
A generator is two arrays: the frequencies from [[sos_preset]] and the phases from [[sos_init_phases]].
[[sos_val]] evaluates them. Each column of the phase array is one independent generator. Evaluating many
generators in one call is much cheaper than one call per generator, because the sines and cosines of the
positions are shared: 20 generators take less than twice as long as one.

```
# Frequencies of a predefined generator for a decorrelation distance of 10 m
freq, acf, dist = quadriga_lib.sos.preset("Comb256", 10)

# Phases of 20 generators; each column is one independent random field
phase = quadriga_lib.sos.init_phases(freq.shape[0], 20, 42)

# Values of all 20 generators at the positions, shape (1000, 20)
val_all = quadriga_lib.sos.val(freq, phase, pos)

# The same fields at other positions, at any later time
pos_new = pos + 2.5
val_new = quadriga_lib.sos.val(freq, phase, pos_new)

# Uniform distributed output
val_uni = quadriga_lib.sos.val(freq, phase, pos, uniform=True)

# Map on a regular grid, shape (n_y, n_x, 1, 20); the first generator is saved as an image
x = np.arange(0, 250.5, 0.5)
y = np.arange(0, 100.5, 0.5)
grid = quadriga_lib.sos.map(freq, phase, x, y)
quadriga_lib.tools.write_png("map.png", grid[:, :, 0, 0])
```

- **Positions:** one row per position, `(n_pos, 3)`. Results have one row per position and one column per
  generator, `(n_pos, n_gen)`. A one-dimensional array is one coordinate per position: `[x, y, z]` is three
  positions on the x-axis; one position is a two-dimensional array with one row.
- **Presets:** `Exp`, `Gauss` and `Comb`, each with 128, 256, 512 or 1024 sinusoids, e.g. `"Exp512"`. `Exp`
  is the exponential ACF `exp(-d/d0)`, `Gauss` is `exp(-(d/d0)^2)`, and `Comb` follows `Gauss` up to the
  decorrelation distance `d0` and `Exp` beyond it. Each doubling of the number of sinusoids doubles the
  computing time and lowers the error of the ACF by 3 to 4 dB: [[sos_calc_mse]] reports 28, 31, 34 and
  38 dB for `Exp128` to `Exp1024` and 31, 35, 39 and 42 dB for `Comb128` to `Comb1024`. `"Comb256"` is a
  good default; take a larger table if the shape of the ACF matters, a smaller one if speed does.
- **Both ends mobile:** if a value belongs to a link between two moving devices, pass both positions.
  The value then changes smoothly when either device moves.

```
# Positions of the two ends of 1000 links; one value per link and generator, shape (1000, 20)
pos_tx = pos
pos_rx = np.random.default_rng(1).random((1000, 3)) * 100
val_link = quadriga_lib.sos.val(freq, phase, pos_tx, pos_rx, acf=acf, dist=dist)
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
  except [[sos_generate]], which runs on one thread. No result depends on the number of threads.
  [[sos_val]] computes in its output: it needs no further memory of that size for one device and one more
  array of that size for two devices.

**Own autocorrelation functions**<br>
[[sos_generate]] finds the frequencies for any ACF that is given as a table over the distance, and
[[sos_calc_mse]] reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```
# Target ACF from 0 to 5 decorrelation distances, here a Gaussian shape with 5 m
dist_own = np.arange(200) * 0.125
acf_own = np.exp(-np.square(dist_own / 5))

# 256 sinusoids in 3-D, 300 test directions, wavelengths up to 20 decorrelation distances (100 m)
freq_own, ase, test_dir = quadriga_lib.sos.generate(acf_own, dist_own, 256, n_dir=300, freq_min=0.01, seed=1)

# Approximation error in dB, larger is better; about 35 dB for this example
mse, mse_core, mse_all, acf_tail, test_dir = quadriga_lib.sos.calc_mse(freq_own, acf_own, dist_own)
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
# sos_val
Spatially correlated random values at given positions

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of
  `freq`, in cycles per meter) and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function
  (ACF) `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the
  frequencies were fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The
  sines and cosines of the positions are computed once and reused for all generators
- Positions are given as one row per position with the columns `(x, y, z)`; they may have fewer
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
  array as `phase` and `phase_b` selects the two-map formula
- `freq_scale_b` other than 1 requires `phase_b`: one map with two decorrelation distances has no
  consistent ACF
- Two devices on the same map need an ACF above -1 at all sample points, because the value is
  divided by `sqrt( 2 + 2·A(D) )`
- `uniform = True` maps the values to `0.5 · erfc( -val / sqrt(2) )`, which is uniform between 0 and
  1 as far as `val` is Normal distributed. The limits 0 and 1 are included: a value far out in the
  tail is rounded to them
- Kernels: GENERIC computes in double precision. AVX2 evaluates sine and cosine in single
  precision; its values differ from GENERIC by less than `4e-7 · sqrt( 2·n_coeff )`, typically by
  less than 1e-6
- Accuracy far from the origin: the argument `f_n·x` is known to about 1e-16 of its size. The
  bound between the kernels holds up to about 1e9 periods of the sinusoids; beyond that both
  kernels lose accuracy at the same rate
- A NaN or infinite coordinate gives NaN at that position and does not affect other positions
- The value at a position does not depend on the other positions or generators of the call, nor
  on the number of threads
- An optional array that is `None` or empty counts as not given

## Usage:
```
val = quadriga_lib.sos.val( freq, phase, pos, pos_b, phase_b, acf, dist, freq_scale_b, \
    uniform, use_kernel )
```

## Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`phase`** — Phases; one column per generator; `(n_coeff, n_gen)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; not more columns than `freq`; a 1-D array is one column,
  i.e. one x-coordinate per position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos, 1)`, `(n_pos, 2)` or
  `(n_pos, 3)` as for `pos`; the number of columns may differ from `pos`; default: `None`
- **`phase_b`** — Phases of the second device; requires `pos_b`; `None` = same map;
  `(n_coeff, n_gen)`; default: `None`
- **`acf`** — Target ACF; required for two devices on the same map, must then be > -1; `(n_acf,)`;
  default: `None`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite;
  `(n_acf,)`; default: `None`
- **`freq_scale_b`** — Frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; default: 1.0
- **`uniform`** — Output distribution: `False` = Normal(0,1), `True` = Uniform between 0 and 1;
  default: `False`
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2 (throws if
  unavailable); `int`; default: 0

## Output:
- **`val`** — Values; one row per position, one column per generator; `(n_pos, n_gen)`

## See also:
- [[sos_preset]] (frequencies and ACF of the predefined models)
- [[sos_init_phases]] (random phases)
- [[sos_map]] (evaluation on a grid)
- [[sos_randn]] (generator set-up and evaluation in one call)
MD!*/

py::array_t<double> sos_val(const py::array_t<double> &freq,
                            const py::array_t<double> &phase,
                            const py::array_t<double> &pos,
                            py::handle pos_b,
                            py::handle phase_b,
                            py::handle acf,
                            py::handle dist,
                            double freq_scale_b,
                            bool uniform,
                            int use_kernel)
{
    // Read inputs
    const auto freq_a = qd_python_numpy2arma_Mat<double>(freq, true);
    const auto phase_a = qd_python_numpy2arma_Mat<double>(phase, true);
    const auto pos_a = qd_python_numpy2arma_Mat<double>(pos, true);
    const auto pos_b_a = qd_python_numpy2arma_Mat<double>(pos_b, true);
    const auto phase_b_a = qd_python_numpy2arma_Mat<double>(phase_b, true);
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Output allocation
    arma::mat val;
    auto val_py = qd_python_init_output(pos_a.n_rows, phase_a.n_cols, &val);

    // Wrap optional pointers
    const arma::mat *p_pos_b = pos_b_a.is_empty() ? nullptr : &pos_b_a;
    const arma::mat *p_phase_b = phase_b_a.is_empty() ? nullptr : &phase_b_a;
    const arma::vec *p_acf = acf_a.is_empty() ? nullptr : &acf_a;
    const arma::vec *p_dist = dist_a.is_empty() ? nullptr : &dist_a;

    // Call library function
    quadriga_lib::sos_val<double>(freq_a, phase_a, pos_a, val, p_pos_b, p_phase_b, p_acf, p_dist,
                                  freq_scale_b, uniform, use_kernel);

    return val_py;
}

// pybind11 declaration:
// m.def("val", &sos_val,
//       py::arg("freq").none(false),
//       py::arg("phase").none(false),
//       py::arg("pos").none(false),
//       py::arg("pos_b") = py::none(),
//       py::arg("phase_b") = py::none(),
//       py::arg("acf") = py::none(),
//       py::arg("dist") = py::none(),
//       py::arg("freq_scale_b") = 1.0,
//       py::arg("uniform") = false,
//       py::arg("use_kernel") = 0);
