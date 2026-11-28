# Sum-of-sinusoids random number generator

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-sum-of-sinusoids-random-number-generator

| Function | Summary |
| --- | --- |
| [sos_acf_2d](#sos_acf_2d) | Target ACF on a 2-D grid |
| [sos_acf_approx](#sos_acf_approx) | ACF produced by a set of sinusoids |
| [sos_acf_estimate](#sos_acf_estimate) | Monte-Carlo estimate of the ACF of one generator |
| [sos_acfi](#sos_acfi) | Target ACF at given distances |
| [sos_calc_mse](#sos_calc_mse) | Approximation error of the ACF of a set of sinusoids |
| [sos_generate](#sos_generate) | Find the spatial frequencies of a sum-of-sinusoids generator for a given ACF |
| [sos_init_phases](#sos_init_phases) | Draw random phases for a set of sum-of-sinusoids generators |
| [sos_map](#sos_map) | Spatially correlated random values on a regular grid |
| [sos_preset](#sos_preset) | Spatial frequencies and ACF of a predefined sum-of-sinusoids generator |
| [sos_rand](#sos_rand) | Spatially correlated Uniform distributed random numbers |
| [sos_randi](#sos_randi) | Spatially correlated integer random numbers |
| [sos_randn](#sos_randn) | Spatially correlated Normal distributed random numbers |
| [sos_remove_bias](#sos_remove_bias) | Remove the sample mean of sum-of-sinusoids generators over a set of positions |
| [sos_val](#sos_val) | Spatially correlated random values at given positions |

These functions generate **spatially correlated random numbers**: values that are random, but similar at
nearby positions. Two positions that are much closer than the *decorrelation distance* get almost the same
value, two positions that are far apart get independent values. The decorrelation distance is the distance
at which the correlation has fallen to `exp(-1)`, about 0.37. Typical uses are shadow fading maps and the
spatially consistent parameters of the 3GPP TR 38.901 channel model.

The values are a sum of sinusoids, `s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n )`. The frequencies
`f_n` set how the correlation falls with the distance, i.e. the autocorrelation function (ACF). The random
phases `phi_n` select one realisation of the random field. The generator needs no map and no grid:
positions can be anywhere, in 1, 2 or 3 dimensions, and the same frequencies and phases always give the
same field.

**Quick start**\
[sos_randn](#sos_randn) sets up the generators, draws the phases and returns the values in one call. [sos_rand](#sos_rand)
and [sos_randi](#sos_randi) do the same for Uniform and integer values.

```matlab
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
stream; the state of `rand` and `randn` in MATLAB or Octave has no effect on them.

A seed selects a set of phases, not a field. Calls with the same seed but another decorrelation distance or
another preset return related fields: they are equal at the coordinate origin and correlated around it.
Quantities that must be independent of each other, such as shadow fading and delay spread, therefore need
different seeds, or one call with one entry of `dist_decorr` per quantity: the generators of one call are
independent.\
**Keeping a generator**\
A generator is two matrices: the frequencies from [sos_preset](#sos_preset) and the phases from [sos_init_phases](#sos_init_phases).
[sos_val](#sos_val) evaluates them. Each column of the phase matrix is one independent generator. Evaluating many
generators in one call is much cheaper than one call per generator, because the sines and cosines of the
positions are shared: 20 generators take less than twice as long as one.

```matlab
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
  computing time and lowers the error of the ACF by 3 to 4 dB: [sos_calc_mse](#sos_calc_mse) reports 28, 31, 34 and
  38 dB for `Exp128` to `Exp1024` and 31, 35, 39 and 42 dB for `Comb128` to `Comb1024`. `'Comb256'` is a
  good default; take a larger table if the shape of the ACF matters, a smaller one if speed does.
- **Both ends mobile:** if a value belongs to a link between two moving devices, pass both positions.
  The value then changes smoothly when either device moves, keeps the correlation of one device while the
  other stays, and does not change when the two ends swap.

```matlab
% Positions of the two ends of 1000 links; one value per link and generator, size [1000, 20]
pos_tx = pos;
pos_rx = rand(1000, 3) * 100;
val_link = quadriga_lib.sos_val(spatial_freq, phase, pos_tx, pos_rx);
```

- **Mean over an area:** the values at positions that lie within a few decorrelation distances of each
  other are similar, so their average is not zero for a single realisation. This is how a correlated
  field behaves. [sos_remove_bias](#sos_remove_bias) can force the average to zero if an application needs it. This
  removes variance: for 200 positions in a square with an edge of 20 decorrelation distances 2.5 %, for
  an edge of 5 decorrelation distances 14 %, for 2 already 44 %. It is meant for positions that spread
  over many decorrelation distances.
- **Coordinates:** use local coordinates. The argument `f_n·x` of a sinusoid is computed in double
  precision and is known to about 1e-16 of its size, so the values lose accuracy with the distance from
  the origin counted in wavelengths. Up to 1e9 wavelengths (more than 1e9 m for a decorrelation distance
  of 10 m) the results stay within the tolerance stated in [sos_val](#sos_val).
- **Threads and memory:** the functions that take noticeable time use the OpenMP threads of the process,
  except [sos_generate](#sos_generate), which runs on one thread. No result depends on the number of threads,
  except the last digits of the errors of [sos_calc_mse](#sos_calc_mse).
  [sos_val](#sos_val) computes in its output: it needs no further memory of that size for one device and one more
  array of that size for two devices.

**Own autocorrelation functions**\
[sos_generate](#sos_generate) finds the frequencies for any ACF that is given as a table over the distance, and
[sos_calc_mse](#sos_calc_mse) reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```matlab
% Target ACF from 0 to 5 decorrelation distances, here a Gaussian shape with 5 m
dist_own = 0:0.125:24.875;
acf_own  = exp(-(dist_own / 5).^2);

% 256 sinusoids in 3-D, 300 test directions, wavelengths up to 20 decorrelation distances (100 m)
freq_own = quadriga_lib.sos_generate(acf_own, dist_own, 256, 3, true, 300, true, 0.01, 0, 1);

% Approximation error in dB, larger is better; about 35 dB for this example
mse = quadriga_lib.sos_calc_mse(freq_own, acf_own, dist_own);
```

The result is used like a preset: draw phases with [sos_init_phases](#sos_init_phases) and evaluate with [sos_val](#sos_val). To
change the decorrelation distance afterwards, divide the frequencies and multiply the distances by the
same factor.

**Reference**\
S. Jaeckel, L. Raschkowski, F. Burkhardt, L. Thiele; "Efficient Sum-of-Sinusoids-Based Spatial Consistency
for the 3GPP New-Radio Channel Model"; 2018 IEEE Globecom Workshops (GC Wkshps); 2018; Available:
[IEEE Xplore](https://ieeexplore.ieee.org/document/8644265) (Original version),
[PDF version](http://quadriga-lib.org/publications/2018_sum_of_sinusoids.pdf),
[Markdown version](http://quadriga-lib.org/publications/2018_sum_of_sinusoids.md)

---
<a id="sos_acf_2d"></a>
## sos_acf_2d
Target ACF on a 2-D grid

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_acf_2d) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.acf_2d)

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the
  distance, on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample
  point; `n_grid = 4·n_acf - 3`, the center is element `2·n_acf - 1`
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [sos_acf_approx](#sos_acf_approx)
- Equally spaced means that every sample point lies within 1e-4 of the step from its place on
  the regular axis

### Usage:
```matlab
[ acf_2d, dist_2d ] = quadriga_lib.sos_acf_2d( acf, dist );
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `[n_acf]`

### Outputs:
- **`acf_2d`** — ACF on the grid; `[n_grid, n_grid]`
- **`dist_2d`** — Grid axis; `[n_grid, 1]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF produced by a set of sinusoids on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
<a id="sos_acf_approx"></a>
## sos_acf_approx
ACF produced by a set of sinusoids

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_acf_approx) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.acf_approx)

- Autocorrelation function (ACF) of the values of [sos_val](#sos_val) over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [sos_acf_2d](#sos_acf_2d): from `-2·Dmax` to `2·Dmax` with the spacing of
  `dist`, `n_grid = 4·n_acf - 3`
- `n_dim = 1`: along the axis; `[n_grid, 1]`
- `n_dim = 2`: on the x-y plane; `[n_grid, n_grid]`; element `(iy, ix)`
- `n_dim = 3`: on the x-y, x-z and y-z plane (pages 1, 2, 3); `[n_grid, n_grid, 3]`; the row index
  belongs to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values
- `R(-dx) = R(dx)`: each plane is point-symmetric about its center, exactly
- Equally spaced has the meaning given in [sos_acf_2d](#sos_acf_2d)

### Usage:
```matlab
[ acf_approx, dist_2d ] = quadriga_lib.sos_acf_approx( spatial_freq, dist );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; at least
  2 elements; `[n_acf]`

### Outputs:
- **`acf_approx`** — Approximated ACF; size depends on `n_dim`, see above
- **`dist_2d`** — Grid axis; `[n_grid, 1]`

### See also:
- [sos_acf_2d](#sos_acf_2d) (target ACF on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
<a id="sos_acf_estimate"></a>
## sos_acf_estimate
Monte-Carlo estimate of the ACF of one generator

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_acf_estimate) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.acf_estimate)

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at
  random positions
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin,
  evaluates the generator there, and for each distance bin computes the correlation coefficient
  between the two values of all position pairs whose distance falls into the bin
- Positions: a matrix `[2·n_pos, 3]` is filled column by column (all x, then all y, then all z)
  with `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in
  [sos_init_phases](#sos_init_phases)
- Bin `n` covers the distances from the midpoint between `bins(n-1)` and `bins(n)` (excluded) to
  the midpoint between `bins(n)` and `bins(n+1)` (included). The first bin starts at 1e-5 m, the
  last bin ends half a mean bin spacing above `bins(end)`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices, reciprocal as in [sos_val](#sos_val). The first `n_pos` positions are the
  first device, the second `n_pos` positions the second device, and the value of a link is [sos_val](#sos_val) with both
  positions. Two links form a pair if the distances between their first devices and between their second devices
  fall into the same bin; for the two-device generator of [sos_val](#sos_val) the estimate approximates the square of
  the ACF
- The estimate describes one realisation; it scatters around the ACF of the sinusoids
- Run time grows with `n_pos^2`; the result does not depend on the number of threads

### Usage:
```matlab
[ acf_est, dist_est, acf_est_dual ] = quadriga_lib.sos_acf_estimate( spatial_freq, phase, bins, n_pos, seed );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `[n_coeff, 3]`
- **`phase`** — Phases of the generator; `[n_coeff]`
- **`bins`** — Bin centers; must be >= 0, increase strictly and be finite; at least 2
  elements; `[n_bin]`
- **`n_pos`** — Number of random positions per device; must be > 1
- **`seed`** — Seed for the positions; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

### Outputs:
- **`acf_est`** — Estimated ACF; `[n_bin, 1]`
- **`dist_est`** — Mean distance of the pairs in each bin; `[n_bin, 1]`
- **`acf_est_dual`** — Estimated ACF for two devices; `[n_bin, 1]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF of the sinusoids in closed form)
- [sos_val](#sos_val) (evaluation of the generator)

---
<a id="sos_acfi"></a>
## sos_acfi
Target ACF at given distances

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_acfi) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.acfi)

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table

### Usage:
```matlab
val = quadriga_lib.sos_acfi( acf, dist, d );
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `[n_acf]`
- **`d`** — Distances; must be >= 0; `[n_d]`

### Output:
- **`val`** — ACF at the distances `d`; `[n_d, 1]`

### See also:
- [sos_preset](#sos_preset) (ACF tables of the predefined models)

---
<a id="sos_calc_mse"></a>
## sos_calc_mse
Approximation error of the ACF of a set of sinusoids

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_calc_mse) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.calc_mse)

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- `mse_dB`: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`, with `i` counted
  from 0. A direction and its opposite give the same ACF
- `mse_core_dB`: error on the grid of [sos_acf_approx](#sos_acf_approx) against [sos_acf_2d](#sos_acf_2d), for the grid points
  within a radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error
  is averaged over the three planes
- `mse_all_dB`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond
  `Dmax`
- For `n_dim = 1`, `mse_core_dB` is evaluated on the sample points `dist`, `mse_all_dB` on the axis from
  `-2·Dmax` to `2·Dmax`, and `mse_dB` equals `mse_core_dB`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`.
  The target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- `mse_core_dB` and `mse_all_dB` need the 2-D grids and are only computed when requested
- A perfect fit has a mean squared error of 0 and is returned as `Inf`; this happens for a target
  that the sinusoids reproduce exactly at all sample points
- Equally spaced has the meaning given in [sos_acf_2d](#sos_acf_2d)

### Usage:
```matlab
[ mse_dB, mse_core_dB, mse_all_dB, acf_tail, test_dir ] = quadriga_lib.sos_calc_mse( spatial_freq, acf, dist, ...
    n_dir );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `[n_acf]`
- **`n_dir`** — Number of test directions; must be > 0; ignored for `n_dim = 1`; default: 10

### Outputs:
- **`mse_dB`** — Error along the test directions in [dB]; scalar
- **`mse_core_dB`** — Error within the core radius in [dB]; scalar
- **`mse_all_dB`** — Error on the complete grid in [dB]; scalar
- **`acf_tail`** — Approximated ACF beyond the fitted range; `[4, 1]`
- **`test_dir`** — Test directions; `[n_dir, 2]` with columns `{azimuth, elevation}` for
  `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

### See also:
- [sos_generate](#sos_generate) (finds the frequencies; minimises the error returned here)
- [sos_acf_approx](#sos_acf_approx), [sos_acf_2d](#sos_acf_2d) (the two grids that are compared)

---
<a id="sos_generate"></a>
## sos_generate
Find the spatial frequencies of a sum-of-sinusoids generator for a given ACF

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_generate) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.generate)

- Searches `n_coeff` frequencies such that the autocorrelation function (ACF) of the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, approximates the target ACF in all directions
- Each sinusoid has a fixed direction and a signed frequency along it. `n_dim = 1`: no direction;
  `n_dim = 2`: directions on the half circle; `n_dim = 3`: directions on the upper hemisphere.
  `uniform_smp = true` places them regularly (equally spaced azimuth, spiral lattice, as the test
  directions of [sos_calc_mse](#sos_calc_mse)), `false` draws them at random
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
  (see `acf_tail` of [sos_calc_mse](#sos_calc_mse)). To prevent this, set `spatial_freq_min`
- `spatial_freq_min` and `spatial_freq_max` limit the magnitude of the frequency along the direction of each
  sinusoid; by default there is no lower limit, and the upper limit is `15 / max(dist)`
- The search grid has `( spatial_freq_max - spatial_freq_min ) · max(dist) / 0.1 + 1` points, and its cosines are
  stored for all sample points. It must not have more than 100000 points (the default has 151),
  which limits `spatial_freq_max - spatial_freq_min` to less than `10000 / max(dist)`
- The function runs on one thread
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse
  in others; the predefined tables of [sos_preset](#sos_preset) were made with 300
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject
  decisions, so a difference in the last bit of the input or of the math library can lead to
  another table of equivalent quality

### Usage:
```matlab
[ spatial_freq, ase, test_dir ] = quadriga_lib.sos_generate( acf, dist, n_coeff, n_dim, uniform_smp, ...
    n_dir, random_init, spatial_freq_min, spatial_freq_max, seed );
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `[n_acf]`
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_dim`** — Number of dimensions; 1, 2 or 3; default: 3
- **`uniform_smp`** — Regular (true) or random (false) directions of the sinusoids; default: true
- **`n_dir`** — Number of test directions; must be > 0; ignored for `n_dim = 1`; default: 10
- **`random_init`** — Start from random frequencies (true) or build up sinusoid by sinusoid
  (false); default: true
- **`spatial_freq_min`** — Lower limit of the frequency magnitude; must be >= 0; 0 = none;
  default: 0
- **`spatial_freq_max`** — Upper limit of the frequency magnitude; NaN or <= 0: default `15 / max(dist)`;
  must be larger than `spatial_freq_min`; limited by the size of the search grid, see above;
  default: NaN
- **`seed`** — Seed; integer from -1 to 2^63-1; -1 seeds from the system random device; default: -1

### Outputs:
- **`spatial_freq`** — Spatial frequencies; `[n_coeff, n_dim]`
- **`ase`** — Mean squared error over the test directions after each pass, linear scale; `[n_pass, 1]`
- **`test_dir`** — Test directions; `[n_dir, 2]` with columns `{azimuth, elevation}` for
  `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

### See also:
- [sos_calc_mse](#sos_calc_mse) (error of a frequency table)
- [sos_preset](#sos_preset) (precomputed tables)

---
<a id="sos_init_phases"></a>
## sos_init_phases
Draw random phases for a set of sum-of-sinusoids generators

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_init_phases) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.init_phases)

- Each column of `phase` is one realisation of the random field of a generator; see [sos_val](#sos_val)
- Values are uniform in the interval from -pi (included) to pi (excluded)
- Random stream: 64-bit Mersenne Twister (`std::mt19937_64`) seeded with `seed`; each value is
  `2·pi · ( u - 0.5 )`, where `u` is the upper 53 bits of one generator output divided by 2^53.
  A seed >= 0 gives the same phases on all platforms; MATLAB's own random stream is not used
- The matrix is filled column by column from one stream: for the same seed, column `g` does not
  depend on `n_gen`
- A seed stands for a set of phases. Generators that use phases of the same seed with different
  frequencies (another decorrelation distance or preset) produce related fields, which are equal
  at the coordinate origin and correlated around it; draw independent quantities as columns of
  one call or with different seeds

### Usage:
```matlab
phase = quadriga_lib.sos_init_phases( n_coeff, n_gen, seed );
```

### Inputs:
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_gen`** — Number of generators; must be > 0
- **`seed`** — Seed of the random stream; integer from -1 to 2^63-1; -1 seeds from the system random
  device and gives different phases on each call; default: -1

### Output:
- **`phase`** — Phases; `[n_coeff, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluates the generators)
- [sos_preset](#sos_preset) (frequencies that belong to the phases)

---
<a id="sos_map"></a>
## sos_map
Spatially correlated random values on a regular grid

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_map) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.map)

- Evaluates the generators of [sos_val](#sos_val) for one device at all combinations of the grid vectors
- Output in meshgrid order: element `(iy, ix, iz, g)` belongs to the position
  `{x(ix), y(iy), z(iz)}` and to generator `g`, i.e. column `g` of `phase`
- The number of grid vectors given must not exceed the number of columns of `spatial_freq`; an omitted or
  empty `y` or `z` means a single sample at 0 on that axis
- `z` can be given without `y` if `spatial_freq` has 3 columns
- The values are those of [sos_val](#sos_val) at the grid points, bit for bit; kernels, accuracy and the
  Uniform mapping are described there

### Usage:
```matlab
map = quadriga_lib.sos_map( spatial_freq, phase, x, y, z, uniform, use_kernel );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`x`** — Grid points on the x-axis; `[nx]`
- **`y`** *(optional)* — Grid points on the y-axis; requires `n_dim >= 2`; omitted or empty = one
  sample at 0, `ny = 1`; `[ny]`
- **`z`** *(optional)* — Grid points on the z-axis; requires `n_dim = 3`; omitted or empty = one
  sample at 0, `nz = 1`; `[nz]`
- **`uniform`** — Output distribution: false = Normal(0,1), true = Uniform between 0 and 1;
  default: false
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  default: 0

### Output:
- **`map`** — Values; `[ny, nx, nz, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation at arbitrary positions, two devices)

---
<a id="sos_preset"></a>
## sos_preset
Spatial frequencies and ACF of a predefined sum-of-sinusoids generator

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_preset) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.preset)

- Returns a precomputed frequency table for [sos_val](#sos_val) together with the autocorrelation
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

### Usage:
```matlab
[ spatial_freq, acf, dist ] = quadriga_lib.sos_preset( name, dist_decorr );
```

### Inputs:
- **`name`** — Name of the preset, e.g. 'Comb256'; string
- **`dist_decorr`** — Decorrelation distance; must be > 0; default: 10

### Outputs:
- **`spatial_freq`** — Spatial frequencies; `[n_coeff, 3]`
- **`acf`** — Target ACF; `[200, 1]`
- **`dist`** — Sample points of the ACF; `[200, 1]`

### See also:
- [sos_init_phases](#sos_init_phases) (random phases for the generator)
- [sos_val](#sos_val) (evaluation)
- [sos_generate](#sos_generate) (frequencies for other ACFs)

---
<a id="sos_rand"></a>
## sos_rand
Spatially correlated Uniform distributed random numbers

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_rand) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.rand)

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`, with or without a
  second device; the phases of a generator therefore do not depend on the number of generators
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices (`pos_b` given): the two-device generator of [sos_val](#sos_val); with the same decorrelation distance
  (`dist_decorr_b` omitted, or equal to `dist_decorr` for that generator) `reciprocal = true`, the value does
  not change when the devices swap positions; with different ones `reciprocal = false` and
  `spatial_freq_scale_b = dist_decorr / dist_decorr_b`
- The Normal distributed value `s` of the generator is mapped to `0.5 · erfc( -s / sqrt(2) )`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [sos_val](#sos_val). MATLAB's own random stream is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent

### Usage:
```matlab
val = quadriga_lib.sos_rand( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed );
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device; requires
  `pos_b`; must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** — Name of the preset, see [sos_preset](#sos_preset); string; default: 'Comb256'
- **`seed`** — Seed of the phases; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

### Output:
- **`val`** — Values, approximately Uniform distributed between 0 and 1; one row per position, one
  column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_randn](#sos_randn), [sos_randi](#sos_randi) (Normal and integer values)

---
<a id="sos_randi"></a>
## sos_randi
Spatially correlated integer random numbers

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_randi) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.randi)

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`, with or without a
  second device; the phases of a generator therefore do not depend on the number of generators
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices (`pos_b` given): the two-device generator of [sos_val](#sos_val); with the same decorrelation distance
  (`dist_decorr_b` omitted, or equal to `dist_decorr` for that generator) `reciprocal = true`, the value does
  not change when the devices swap positions; with different ones `reciprocal = false` and
  `spatial_freq_scale_b = dist_decorr / dist_decorr_b`
- The Uniform value `u` of [sos_rand](#sos_rand) is mapped to `ceil( u · imax )`, limited to 1 ... `imax`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [sos_val](#sos_val), and an integer can then differ by 1
  where the Uniform value lies at a step. MATLAB's own random stream is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent

### Usage:
```matlab
val = quadriga_lib.sos_randi( dist_decorr, pos, imax, pos_b, dist_decorr_b, preset, seed );
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`imax`** — Largest value; must be > 0
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device; requires
  `pos_b`; must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** — Name of the preset, see [sos_preset](#sos_preset); string; default: 'Comb256'
- **`seed`** — Seed of the phases; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

### Output:
- **`val`** — Integer values from 1 to `imax`, each with approximately the same probability;
  returned as double; one row per position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_randn](#sos_randn), [sos_rand](#sos_rand) (Normal and Uniform values)

---
<a id="sos_randn"></a>
## sos_randn
Spatially correlated Normal distributed random numbers

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_randn) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.randn)

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`, with or without a
  second device; the phases of a generator therefore do not depend on the number of generators
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices (`pos_b` given): the two-device generator of [sos_val](#sos_val); with the same decorrelation distance
  (`dist_decorr_b` omitted, or equal to `dist_decorr` for that generator) `reciprocal = true`, the value does
  not change when the devices swap positions; with different ones `reciprocal = false` and
  `spatial_freq_scale_b = dist_decorr / dist_decorr_b`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [sos_val](#sos_val). MATLAB's own random stream is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent

### Usage:
```matlab
val = quadriga_lib.sos_randn( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed );
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from
  `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device; requires
  `pos_b`; must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** — Name of the preset, see [sos_preset](#sos_preset); string; default: 'Comb256'
- **`seed`** — Seed of the phases; integer from -1 to 2^63-1; -1 seeds from the system random
  device; default: -1

### Output:
- **`val`** — Values, approximately Normal distributed with zero mean and unit variance; one row
  per position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_rand](#sos_rand), [sos_randi](#sos_randi) (Uniform and integer values)

---
<a id="sos_remove_bias"></a>
## sos_remove_bias
Remove the sample mean of sum-of-sinusoids generators over a set of positions

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_remove_bias) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.remove_bias)

- One realisation of a spatially correlated random field has a non-zero mean over a finite set of
  positions. Its variance over the realisations is the mean of the autocorrelation function (ACF)
  over all pairs of positions, which is much larger than `1 / n_pos` when the positions lie within
  a few decorrelation distances of each other. This is a property of any correlated field, not an
  error of the generator
- This function adds one offset to all phases of a generator such that the mean of the Normal
  distributed values of [sos_val](#sos_val) over the given positions becomes zero
- Closed form, no iteration: with `z_n = mean over the positions of exp( j·2·pi · f_n·x )` and
  `Z = sum_n z_n · exp( j·phi_n )`, the mean is `sqrt(2/n_coeff) · Re( exp( j·offset ) · Z )`,
  which vanishes for `offset = pi/2 - arg(Z)` and for that value minus pi; the one with the
  smaller magnitude is applied, so the offset lies between -pi/2 and pi/2
- One offset per generator; with two devices (`pos_b` given) it zeroes the mean of the two-device value
  of [sos_val](#sos_val) with the same `spatial_freq_scale_b` and `reciprocal`
- Two reciprocal devices at a position where the normalization of [sos_val](#sos_val) is singular (`c = -1`) throw
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

### Usage:
```matlab
[ phase_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( spatial_freq, phase_in, pos, pos_b, ...
    spatial_freq_scale_b, reciprocal );
```

### Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase_in`** — Phases; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `spatial_freq`
- **`pos_b`** *(optional)* — Positions of the second device; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`
- **`spatial_freq_scale_b`** — Spatial-frequency scale of the second device as in [sos_val](#sos_val); must be > 0;
  other than 1 requires `pos_b` and `reciprocal = false`; default: 1
- **`reciprocal`** — Two devices: the mode of [sos_val](#sos_val); default: true

### Outputs:
- **`phase_out`** — Corrected phases; `[n_coeff, n_gen]`
- **`offset`** — Offsets added to the phases; `[n_gen, 1]`
- **`mean_before`** — Sample mean of the values of [sos_val](#sos_val) for the given arguments before the
  correction; `[n_gen, 1]`

### See also:
- [sos_val](#sos_val) (the values whose mean is removed)

---
<a id="sos_val"></a>
## sos_val
Spatially correlated random values at given positions

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_val) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.val)

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

### Usage:
```matlab
val = quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b, spatial_freq_scale_b, reciprocal, ...
    uniform, use_kernel );
```

### Inputs:
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

### Output:
- **`val`** — Values; one row per position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_preset](#sos_preset) (frequencies and ACF of the predefined models)
- [sos_init_phases](#sos_init_phases) (random phases)
- [sos_map](#sos_map) (evaluation on a grid)
- [sos_randn](#sos_randn) (generator set-up and evaluation in one call)
