# Sum-of-sinusoids random number generator

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-sum-of-sinusoids-random-number-generator

| Function | Summary |
| --- | --- |
| [sos.acf_2d](#sos.acf_2d) | Target ACF on a 2-D grid |
| [sos.acf_approx](#sos.acf_approx) | ACF produced by a set of sinusoids |
| [sos.acf_estimate](#sos.acf_estimate) | Monte-Carlo estimate of the ACF of one generator |
| [sos.acfi](#sos.acfi) | Target ACF at given distances |
| [sos.calc_mse](#sos.calc_mse) | Approximation error of the ACF of a set of sinusoids |
| [sos.generate](#sos.generate) | Find the frequencies of a sum-of-sinusoids generator for a given ACF |
| [sos.init_phases](#sos.init_phases) | Draw random phases for a set of sum-of-sinusoids generators |
| [sos.map](#sos.map) | Spatially correlated random values on a regular grid |
| [sos.preset](#sos.preset) | Frequencies and ACF of a predefined sum-of-sinusoids generator |
| [sos.rand](#sos.rand) | Spatially correlated Uniform distributed random numbers |
| [sos.randi](#sos.randi) | Spatially correlated integer random numbers |
| [sos.randn](#sos.randn) | Spatially correlated Normal distributed random numbers |
| [sos.remove_bias](#sos.remove_bias) | Remove the sample mean of sum-of-sinusoids generators over a set of positions |
| [sos.val](#sos.val) | Spatially correlated random values at given positions |

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

The functions are in the submodule `quadriga_lib.sos`: [sos.val](#sos.val) is called as `quadriga_lib.sos.val`. In the
examples, `np` is numpy.

**Quick start**\
[sos.randn](#sos.randn) sets up the generators, draws the phases and returns the values in one call. [sos.rand](#sos.rand)
and [sos.randi](#sos.randi) do the same for Uniform and integer values.

```python
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
random stream; the random state of numpy has no effect on them.

A seed selects a set of phases, not a field. Calls with the same seed but another decorrelation distance or
another preset return related fields: they are equal at the coordinate origin and correlated around it.
Quantities that must be independent of each other, such as shadow fading and delay spread, therefore need
different seeds, or one call with one entry of `dist_decorr` per quantity: the generators of one call are
independent.\
**Keeping a generator**\
A generator is two arrays: the frequencies from [sos.preset](#sos.preset) and the phases from [sos.init_phases](#sos.init_phases).
[sos.val](#sos.val) evaluates them. Each column of the phase array is one independent generator. Evaluating many
generators in one call is much cheaper than one call per generator, because the sines and cosines of the
positions are shared: 20 generators take less than twice as long as one.

```python
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
  computing time and lowers the error of the ACF by 3 to 4 dB: [sos.calc_mse](#sos.calc_mse) reports 28, 31, 34 and
  38 dB for `Exp128` to `Exp1024` and 31, 35, 39 and 42 dB for `Comb128` to `Comb1024`. `"Comb256"` is a
  good default; take a larger table if the shape of the ACF matters, a smaller one if speed does.
- **Both ends mobile:** if a value belongs to a link between two moving devices, pass both positions.
  The value then changes smoothly when either device moves.

```python
# Positions of the two ends of 1000 links; one value per link and generator, shape (1000, 20)
pos_tx = pos
pos_rx = np.random.default_rng(1).random((1000, 3)) * 100
val_link = quadriga_lib.sos.val(freq, phase, pos_tx, pos_rx, acf=acf, dist=dist)
```

- **Mean over an area:** the values at positions that lie within a few decorrelation distances of each
  other are similar, so their average is not zero for a single realisation. This is how a correlated
  field behaves. [sos.remove_bias](#sos.remove_bias) can force the average to zero if an application needs it. This
  removes variance: for 200 positions in a square with an edge of 20 decorrelation distances 2.5 %, for
  an edge of 5 decorrelation distances 14 %, for 2 already 44 %. It is meant for positions that spread
  over many decorrelation distances.
- **Coordinates:** use local coordinates. The argument `f_n·x` of a sinusoid is computed in double
  precision and is known to about 1e-16 of its size, so the values lose accuracy with the distance from
  the origin counted in wavelengths. Up to 1e9 wavelengths (more than 1e9 m for a decorrelation distance
  of 10 m) the results stay within the tolerance stated in [sos.val](#sos.val).
- **Threads and memory:** the functions that take noticeable time use the OpenMP threads of the process,
  except [sos.generate](#sos.generate), which runs on one thread. No result depends on the number of threads.
  [sos.val](#sos.val) computes in its output: it needs no further memory of that size for one device and one more
  array of that size for two devices.

**Own autocorrelation functions**\
[sos.generate](#sos.generate) finds the frequencies for any ACF that is given as a table over the distance, and
[sos.calc_mse](#sos.calc_mse) reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```python
# Target ACF from 0 to 5 decorrelation distances, here a Gaussian shape with 5 m
dist_own = np.arange(200) * 0.125
acf_own = np.exp(-np.square(dist_own / 5))

# 256 sinusoids in 3-D, 300 test directions, wavelengths up to 20 decorrelation distances (100 m)
freq_own, ase, test_dir = quadriga_lib.sos.generate(acf_own, dist_own, 256, n_dir=300, freq_min=0.01, seed=1)

# Approximation error in dB, larger is better; about 35 dB for this example
mse, mse_core, mse_all, acf_tail, test_dir = quadriga_lib.sos.calc_mse(freq_own, acf_own, dist_own)
```

The result is used like a preset: draw phases with [sos.init_phases](#sos.init_phases) and evaluate with [sos.val](#sos.val). To
change the decorrelation distance afterwards, divide the frequencies and multiply the distances by the
same factor.

**Reference**\
S. Jaeckel, L. Raschkowski, F. Burkhardt, L. Thiele; "Efficient Sum-of-Sinusoids-Based Spatial Consistency
for the 3GPP New-Radio Channel Model"; 2018 IEEE Globecom Workshops (GC Wkshps); 2018; Available:
[IEEE Xplore](https://ieeexplore.ieee.org/document/8644265) (Original version),
[PDF version](http://quadriga-lib.org/publications/2018_sum_of_sinusoids.pdf),
[Markdown version](http://quadriga-lib.org/publications/2018_sum_of_sinusoids.md)

---
<a id="sos.acf_2d"></a>
## sos.acf_2d
Target ACF on a 2-D grid

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_acf_2d) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_acf_2d)

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the
  distance, on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample
  point; `n_grid = 4·n_acf - 3`, the center is at index `2·n_acf - 2`
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [sos.acf_approx](#sos.acf_approx)
- Equally spaced means that every sample point lies within 1e-4 of the step from its place on
  the regular axis

### Usage:
```python
acf_2d, dist_2d = quadriga_lib.sos.acf_2d( acf, dist )
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `(n_acf,)`

### Outputs:
- **`acf_2d`** — ACF on the grid; `(n_grid, n_grid)`
- **`dist_2d`** — Grid axis; `(n_grid,)`

### See also:
- [sos.acf_approx](#sos.acf_approx) (ACF produced by a set of sinusoids on the same grid)
- [sos.calc_mse](#sos.calc_mse) (difference between the two)

---
<a id="sos.acf_approx"></a>
## sos.acf_approx
ACF produced by a set of sinusoids

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_acf_approx) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_acf_approx)

- Autocorrelation function (ACF) of the values of [sos.val](#sos.val) over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [sos.acf_2d](#sos.acf_2d): from `-2·Dmax` to `2·Dmax` with the spacing of
  `dist`, `n_grid = 4·n_acf - 3`
- The output is always 3-dimensional
- `n_dim = 1`: along the axis; `(n_grid, 1, 1)`
- `n_dim = 2`: on the x-y plane; `(n_grid, n_grid, 1)`; element `[iy, ix, 0]`
- `n_dim = 3`: on the x-y, x-z and y-z plane (index 0, 1, 2 of the last axis); `(n_grid, n_grid, 3)`;
  the row index belongs to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values
- `R(-dx) = R(dx)`: each plane is point-symmetric about its center, exactly
- Equally spaced has the meaning given in [sos.acf_2d](#sos.acf_2d)

### Usage:
```python
acf_approx, dist_2d = quadriga_lib.sos.acf_approx( freq, dist )
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; at least
  2 elements; `(n_acf,)`

### Outputs:
- **`acf_approx`** — Approximated ACF; shape depends on `n_dim`, see above
- **`dist_2d`** — Grid axis; `(n_grid,)`

### See also:
- [sos.acf_2d](#sos.acf_2d) (target ACF on the same grid)
- [sos.calc_mse](#sos.calc_mse) (difference between the two)

---
<a id="sos.acf_estimate"></a>
## sos.acf_estimate
Monte-Carlo estimate of the ACF of one generator

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_acf_estimate) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_acf_estimate)

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at
  random positions
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin,
  evaluates the generator there, and for each distance bin computes the correlation coefficient
  between the two values of all position pairs whose distance falls into the bin
- Positions: an array `(2·n_pos, 3)` is filled column by column (all x, then all y, then all z)
  with `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in
  [sos.init_phases](#sos.init_phases)
- Bin `n` covers the distances from the midpoint between `bins[n-1]` and `bins[n]` (excluded) to
  the midpoint between `bins[n]` and `bins[n+1]` (included). The first bin starts at 1e-5 m, the
  last bin ends half a mean bin spacing above `bins[-1]`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices on the same map. The first `n_pos` positions are the
  first device, the second `n_pos` positions the second device, and the value of a link is
  [sos.val](#sos.val) with both positions. Two links form a pair if the distance between their first
  devices falls into the bin and the distance between their second devices does not exceed the
  upper limit of the bin; requires `acf` and `dist`
- `acf_est_dual` is computed if `acf` or `dist` is given, and is `None` if both are omitted; giving
  only one of the two is an error
- The estimate describes one realisation; it scatters around the ACF of the sinusoids
- Run time grows with `n_pos^2`; the result does not depend on the number of threads

### Usage:
```python
acf_est, dist_est, acf_est_dual = quadriga_lib.sos.acf_estimate( freq, phase, bins, n_pos, \
    acf, dist, seed )
```

### Inputs:
- **`freq`** — Frequencies; `(n_coeff, 3)`
- **`phase`** — Phases of the generator; `(n_coeff,)` or `(n_coeff, 1)`
- **`bins`** — Bin centers; must be >= 0, increase strictly and be finite; at least 2
  elements; `(n_bin,)`
- **`n_pos`** — Number of random positions per device; `int`; must be > 1
- **`acf`** — Target ACF; required for `acf_est_dual`, must then be > -1;
  `(n_acf,)`; default: `None`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`;
  default: `None`
- **`seed`** — Seed for the positions; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

### Outputs:
- **`acf_est`** — Estimated ACF; `(n_bin,)`
- **`dist_est`** — Mean distance of the pairs in each bin; `(n_bin,)`
- **`acf_est_dual`** — Estimated ACF for two devices on the same map; `(n_bin,)`; `None` if `acf`
  and `dist` were not given

### See also:
- [sos.acf_approx](#sos.acf_approx) (ACF of the sinusoids in closed form)
- [sos.val](#sos.val) (evaluation of the generator)

---
<a id="sos.acfi"></a>
## sos.acfi
Target ACF at given distances

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_acfi) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_acfi)

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table
- The same rule is applied by [sos.val](#sos.val) and [sos.remove_bias](#sos.remove_bias) for two devices on the same map

### Usage:
```python
val = quadriga_lib.sos.acfi( acf, dist, d )
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`
- **`d`** — Distances; must be >= 0; `(n_d,)`

### Output:
- **`val`** — ACF at the distances `d`; `(n_d,)`

### See also:
- [sos.preset](#sos.preset) (ACF tables of the predefined models)

---
<a id="sos.calc_mse"></a>
## sos.calc_mse
Approximation error of the ACF of a set of sinusoids

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_calc_mse) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_calc_mse)

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- `mse`: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`, with `i` counted
  from 0. A direction and its opposite give the same ACF
- `mse_core`: error on the grid of [sos.acf_approx](#sos.acf_approx) against [sos.acf_2d](#sos.acf_2d), for the grid points
  within a radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error
  is averaged over the three planes
- `mse_all`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond
  `Dmax`
- For `n_dim = 1`, `mse_core` is evaluated on the sample points `dist`, `mse_all` on the axis from
  `-2·Dmax` to `2·Dmax`, and `mse` equals `mse_core`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`.
  The target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- All outputs are always computed, including the 2-D grids behind `mse_core` and `mse_all`
- A perfect fit has a mean squared error of 0 and is returned as `inf`; this happens for a target
  that the sinusoids reproduce exactly at all sample points
- Equally spaced has the meaning given in [sos.acf_2d](#sos.acf_2d)

### Usage:
```python
mse, mse_core, mse_all, acf_tail, test_dir = quadriga_lib.sos.calc_mse( freq, acf, dist, n_dir )
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `(n_acf,)`
- **`n_dir`** — Number of test directions; `int`; must be > 0; ignored for `n_dim = 1`; default: 10

### Outputs:
- **`mse`** — Error along the test directions in [dB]; `float`
- **`mse_core`** — Error within the core radius in [dB]; `float`
- **`mse_all`** — Error on the complete grid in [dB]; `float`
- **`acf_tail`** — Approximated ACF beyond the fitted range; `(4,)`
- **`test_dir`** — Test directions; `(n_dir, 2)` with columns `(azimuth, elevation)` for
  `n_dim = 3`; `(n_dir, 1)` with the azimuth for `n_dim = 2`; empty `(0, 0)` for `n_dim = 1`

### See also:
- [sos.generate](#sos.generate) (finds the frequencies; minimises the error returned here)
- [sos.acf_approx](#sos.acf_approx), [sos.acf_2d](#sos.acf_2d) (the two grids that are compared)

---
<a id="sos.generate"></a>
## sos.generate
Find the frequencies of a sum-of-sinusoids generator for a given ACF

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_generate) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_generate)

- Searches `n_coeff` frequencies such that the autocorrelation function (ACF) of the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, approximates the target ACF in all directions
- Each sinusoid has a fixed direction and a signed frequency along it. `n_dim = 1`: no direction;
  `n_dim = 2`: directions on the half circle; `n_dim = 3`: directions on the upper hemisphere.
  `uniform_smp = True` places them regularly (equally spaced azimuth, spiral lattice, as the test
  directions of [sos.calc_mse](#sos.calc_mse)), `False` draws them at random
- Iteration: in each pass every sinusoid is re-fitted once. Its frequency is chosen to fit the
  residual between the target and the contribution of all other sinusoids along its own direction:
  coarse search over a grid with a spacing of `0.1 / max(dist)`, then a step refinement. The new
  frequency is kept only if the mean squared error over the test directions decreases. The
  iteration stops after a pass that changes nothing or lowers the error by less than a factor of
  1e-6, or after 1000 passes
- `random_init = True` starts from uniformly random frequencies; `False` builds the set up
  sinusoid by sinusoid in the first pass
- The fit covers the range of `dist` only. The ACF of the result beyond the last sample point is
  not controlled; sinusoids with a frequency near zero keep it above zero at all larger distances
  (see `acf_tail` of [sos.calc_mse](#sos.calc_mse)). To prevent this, set `freq_min`
- `freq_min` and `freq_max` limit the magnitude of the frequency along the direction of each
  sinusoid; 0 selects the default: no lower limit, and `15 / max(dist)` as upper limit
- The search grid has `( freq_max - freq_min ) · max(dist) / 0.1 + 1` points, and its cosines are
  stored for all sample points. It must not have more than 100000 points (the default has 151),
  which limits `freq_max - freq_min` to less than `10000 / max(dist)`
- The function runs on one thread
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse
  in others; the predefined tables of [sos.preset](#sos.preset) were made with 300
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject
  decisions, so a difference in the last bit of the input or of the math library can lead to
  another table of equivalent quality

### Usage:
```python
freq, ase, test_dir = quadriga_lib.sos.generate( acf, dist, n_coeff, n_dim, uniform_smp, n_dir, \
    random_init, freq_min, freq_max, seed )
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`
- **`n_coeff`** — Number of sinusoids; `int`; must be > 0
- **`n_dim`** — Number of dimensions; `int`; 1, 2 or 3; default: 3
- **`uniform_smp`** — Regular (`True`) or random (`False`) directions of the sinusoids;
  default: `True`
- **`n_dir`** — Number of test directions; `int`; must be > 0; ignored for `n_dim = 1`; default: 10
- **`random_init`** — Start from random frequencies (`True`) or build up sinusoid by sinusoid
  (`False`); default: `True`
- **`freq_min`** — Lower limit of the frequency magnitude; must be >= 0; 0 = none;
  default: 0.0
- **`freq_max`** — Upper limit of the frequency magnitude; must be >= 0 and, if not 0,
  larger than `freq_min`; 0 = `15 / max(dist)`; limited by the size of the search
  grid, see above; default: 0.0
- **`seed`** — Seed; `int`; must be >= -1; -1 seeds from the system random device; default: -1

### Outputs:
- **`freq`** — Frequencies; `(n_coeff, n_dim)`
- **`ase`** — Mean squared error over the test directions after each pass, linear scale;
  `(n_pass,)`
- **`test_dir`** — Test directions; `(n_dir, 2)` with columns `(azimuth, elevation)` for
  `n_dim = 3`; `(n_dir, 1)` with the azimuth for `n_dim = 2`; empty `(0, 0)` for `n_dim = 1`

### See also:
- [sos.calc_mse](#sos.calc_mse) (error of a frequency table)
- [sos.preset](#sos.preset) (precomputed tables)

---
<a id="sos.init_phases"></a>
## sos.init_phases
Draw random phases for a set of sum-of-sinusoids generators

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_init_phases) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_init_phases)

- Each column of `phase` is one realisation of the random field of a generator; see [sos.val](#sos.val)
- Values are uniform in the interval from -pi (included) to pi (excluded)
- Random stream: 64-bit Mersenne Twister (`std::mt19937_64`) seeded with `seed`; each value is
  `2·pi · ( u - 0.5 )`, where `u` is the upper 53 bits of one generator output divided by 2^53.
  A seed >= 0 gives the same phases on all platforms; the random state of numpy is not used
- The matrix is filled column by column from one stream: for the same seed, column `g` does not
  depend on `n_gen`
- A seed stands for a set of phases. Generators that use phases of the same seed with different
  frequencies (another decorrelation distance or preset) produce related fields, which are equal
  at the coordinate origin and correlated around it; draw independent quantities as columns of
  one call or with different seeds

### Usage:
```python
phase = quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )
```

### Inputs:
- **`n_coeff`** — Number of sinusoids; `int`; must be > 0
- **`n_gen`** — Number of generators; `int`; must be > 0; default: 1
- **`seed`** — Seed of the random stream; `int`; must be >= -1; -1 seeds from the system random
  device and gives different phases on each call; default: -1

### Output:
- **`phase`** — Phases; `(n_coeff, n_gen)`

### See also:
- [sos.val](#sos.val) (evaluates the generators)
- [sos.preset](#sos.preset) (frequencies that belong to the phases)

---
<a id="sos.map"></a>
## sos.map
Spatially correlated random values on a regular grid

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_map) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_map)

- Evaluates the generators of [sos.val](#sos.val) for one device at all combinations of the grid vectors
- Output in meshgrid order: element `[iy, ix, iz, g]` belongs to the position
  `(x[ix], y[iy], z[iz])` and to generator `g`, i.e. column `g` of `phase`
- The number of grid vectors given must not exceed the number of columns of `freq`; a `y` or `z`
  that is `None` or empty means a single sample at 0 on that axis
- `z` can be given without `y` if `freq` has 3 columns
- The values are those of [sos.val](#sos.val) at the grid points, bit for bit; kernels, accuracy and the
  Uniform mapping are described there

### Usage:
```python
map = quadriga_lib.sos.map( freq, phase, x, y, z, uniform, use_kernel )
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`phase`** — Phases; one column per generator; `(n_coeff, n_gen)`
- **`x`** — Grid points on the x-axis; `(nx,)`
- **`y`** — Grid points on the y-axis; requires `n_dim >= 2`; `None` = one sample at 0, `ny = 1`;
  `(ny,)`; default: `None`
- **`z`** — Grid points on the z-axis; requires `n_dim = 3`; `None` = one sample at 0, `nz = 1`;
  `(nz,)`; default: `None`
- **`uniform`** — Output distribution: `False` = Normal(0,1), `True` = Uniform between 0 and 1;
  default: `False`
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2 (throws if
  unavailable); `int`; default: 0

### Output:
- **`map`** — Values; `(ny, nx, nz, n_gen)`

### See also:
- [sos.val](#sos.val) (evaluation at arbitrary positions, two devices)

---
<a id="sos.preset"></a>
## sos.preset
Frequencies and ACF of a predefined sum-of-sinusoids generator

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_preset) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_preset)

- Returns a precomputed frequency table for [sos.val](#sos.val) together with the autocorrelation
  function (ACF) it was fitted to
- Presets: "Exp128", "Exp256", "Exp512", "Exp1024", "Gauss128", "Gauss256", "Gauss512",
  "Gauss1024", "Comb128", "Comb256", "Comb512", "Comb1024". The number is the number of
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
```python
freq, acf, dist = quadriga_lib.sos.preset( name, dist_decorr )
```

### Inputs:
- **`name`** — Name of the preset, e.g. "Comb256"; `str`
- **`dist_decorr`** — Decorrelation distance; must be > 0; default: 10.0

### Outputs:
- **`freq`** — Frequencies; `(n_coeff, 3)`
- **`acf`** — Target ACF; `(200,)`
- **`dist`** — Sample points of the ACF; `(200,)`

### See also:
- [sos.init_phases](#sos.init_phases) (random phases for the generator)
- [sos.val](#sos.val) (evaluation)
- [sos.generate](#sos.generate) (frequencies for other ACFs)

---
<a id="sos.rand"></a>
## sos.rand
Spatially correlated Uniform distributed random numbers

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_rand) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_rand)

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos.val](#sos.val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )`.
  A second device on its own map uses column `g` of a second stream,
  `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62
  flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both streams are random. The
  phases of a generator therefore depend neither on the number of generators nor on whether a
  second device is given
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
  without AVX2 differ by the tolerance stated in [sos.val](#sos.val). The random state of numpy is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent
- An optional array that is `None` or empty counts as not given

### Usage:
```python
val = quadriga_lib.sos.rand( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed )
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `(n_gen,)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; a 1-D array is one column, i.e. one x-coordinate per
  position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos, 1)`, `(n_pos, 2)` or
  `(n_pos, 3)` as for `pos`; the number of columns may differ from `pos`; default: `None`
- **`dist_decorr_b`** — Decorrelation distances of the second device; requires `pos_b`; must
  be >= 0; `None` = same as `dist_decorr`; `(n_gen,)`; default: `None`
- **`preset`** — Name of the preset, see [sos.preset](#sos.preset); `str`; default: "Comb256"
- **`seed`** — Seed of the phases; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

### Output:
- **`val`** — Values, approximately Uniform distributed between 0 and 1; one row per position, one
  column per generator; `(n_pos, n_gen)`

### See also:
- [sos.val](#sos.val) (evaluation with given frequencies and phases)
- [sos.preset](#sos.preset), [sos.init_phases](#sos.init_phases) (the two steps this function combines)
- [sos.randn](#sos.randn), [sos.randi](#sos.randi) (Normal and integer values)

---
<a id="sos.randi"></a>
## sos.randi
Spatially correlated integer random numbers

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_randi) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_randi)

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos.val](#sos.val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )`.
  A second device on its own map uses column `g` of a second stream,
  `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62
  flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both streams are random. The
  phases of a generator therefore depend neither on the number of generators nor on whether a
  second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr`
  for that generator): both devices are on the same map, the value does not change when they
  swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the
  sum of both maps divided by `sqrt(2)`
- The Uniform value `u` of [sos.rand](#sos.rand) is mapped to `ceil( u · imax )`, limited to 1 ... `imax`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [sos.val](#sos.val), and an integer can then differ by 1
  where the Uniform value lies at a step. The random state of numpy is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent
- An optional array that is `None` or empty counts as not given

### Usage:
```python
val = quadriga_lib.sos.randi( dist_decorr, pos, imax, pos_b, dist_decorr_b, preset, seed )
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `(n_gen,)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; a 1-D array is one column, i.e. one x-coordinate per
  position
- **`imax`** — Largest value; `int`; must be > 0
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos, 1)`, `(n_pos, 2)` or
  `(n_pos, 3)` as for `pos`; the number of columns may differ from `pos`; default: `None`
- **`dist_decorr_b`** — Decorrelation distances of the second device; requires `pos_b`; must
  be >= 0; `None` = same as `dist_decorr`; `(n_gen,)`; default: `None`
- **`preset`** — Name of the preset, see [sos.preset](#sos.preset); `str`; default: "Comb256"
- **`seed`** — Seed of the phases; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

### Output:
- **`val`** — Integer values from 1 to `imax`, each with approximately the same probability;
  returned as `float64`; one row per position, one column per generator; `(n_pos, n_gen)`

### See also:
- [sos.val](#sos.val) (evaluation with given frequencies and phases)
- [sos.preset](#sos.preset), [sos.init_phases](#sos.init_phases) (the two steps this function combines)
- [sos.randn](#sos.randn), [sos.rand](#sos.rand) (Normal and Uniform values)

---
<a id="sos.randn"></a>
## sos.randn
Spatially correlated Normal distributed random numbers

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_randn) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_randn)

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a
  preset, draws its phases and evaluates it at the given positions; see [sos.val](#sos.val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )`.
  A second device on its own map uses column `g` of a second stream,
  `quadriga_lib.sos.init_phases( n_coeff, n_gen, seed_b )`, where `seed_b` is the seed with bit 62
  flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both streams are random. The
  phases of a generator therefore depend neither on the number of generators nor on whether a
  second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr`
  for that generator): both devices are on the same map, the value does not change when they
  swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the
  sum of both maps divided by `sqrt(2)`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on
  all platforms: the same positions give the same values and nearby positions similar values,
  also in separate calls. The kernel is chosen automatically, so values from machines with and
  without AVX2 differ by the tolerance stated in [sos.val](#sos.val). The random state of numpy is not used
- A seed selects the phases, not the field: calls with the same seed but other decorrelation
  distances or another preset return related fields, equal at the coordinate origin and
  correlated around it. Use different seeds, or several entries of `dist_decorr` in one call, for
  quantities that must be independent
- An optional array that is `None` or empty counts as not given

### Usage:
```python
val = quadriga_lib.sos.randn( dist_decorr, pos, pos_b, dist_decorr_b, preset, seed )
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances; one generator per entry; must be >= 0; `(n_gen,)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; a 1-D array is one column, i.e. one x-coordinate per
  position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos, 1)`, `(n_pos, 2)` or
  `(n_pos, 3)` as for `pos`; the number of columns may differ from `pos`; default: `None`
- **`dist_decorr_b`** — Decorrelation distances of the second device; requires `pos_b`; must
  be >= 0; `None` = same as `dist_decorr`; `(n_gen,)`; default: `None`
- **`preset`** — Name of the preset, see [sos.preset](#sos.preset); `str`; default: "Comb256"
- **`seed`** — Seed of the phases; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

### Output:
- **`val`** — Values, approximately Normal distributed with zero mean and unit variance; one row
  per position, one column per generator; `(n_pos, n_gen)`

### See also:
- [sos.val](#sos.val) (evaluation with given frequencies and phases)
- [sos.preset](#sos.preset), [sos.init_phases](#sos.init_phases) (the two steps this function combines)
- [sos.rand](#sos.rand), [sos.randi](#sos.randi) (Uniform and integer values)

---
<a id="sos.remove_bias"></a>
## sos.remove_bias
Remove the sample mean of sum-of-sinusoids generators over a set of positions

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_remove_bias) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_remove_bias)

- One realisation of a spatially correlated random field has a non-zero mean over a finite set of
  positions. Its variance over the realisations is the mean of the autocorrelation function (ACF)
  over all pairs of positions, which is much larger than `1 / n_pos` when the positions lie within
  a few decorrelation distances of each other. This is a property of any correlated field, not an
  error of the generator
- This function adds one offset to all phases of a generator such that the mean of the Normal
  distributed values of [sos.val](#sos.val) over the given positions becomes zero
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
- Rule of thumb, measured for 200 positions in a square with the edge `L` and the decorrelation
  distance `d` (`Comb256`): the variance drops by 74 % for `L = d`, 44 % for `L = 2·d`, 14 % for
  `L = 5·d`, 5 % for `L = 10·d`, 2.5 % for `L = 20·d` and less than 1 % from `L = 50·d` on
- Phases are returned in the interval from -pi (included) to pi (excluded). A generator whose
  `abs(Z)` is below `1e-14 · n_coeff` already has a zero mean for every offset and is returned
  unchanged, also if its phases lie outside that interval
- Two devices on the same map need an ACF above -1 at all sample points, as in [sos.val](#sos.val)
- The input arrays `phase_in` and `phase_b_in` are not modified; the corrected phases are new arrays
- An optional array that is `None` or empty counts as not given

### Usage:
```python
phase_out, phase_b_out, offset, mean_before = quadriga_lib.sos.remove_bias( freq, phase_in, pos, \
    pos_b, phase_b_in, acf, dist, freq_scale_b )
```

### Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`phase_in`** — Phases; one column per generator; `(n_coeff, n_gen)`
- **`pos`** — Positions of the first device; one row per position; the number of columns sets the
  number of dimensions: `(n_pos, 1)` = `(x)`, `(n_pos, 2)` = `(x, y)` or `(n_pos, 3)` = `(x, y, z)`;
  coordinates that are not given are 0; not more columns than `freq`; a 1-D array is one column,
  i.e. one x-coordinate per position
- **`pos_b`** — Positions of the second device; `None` = one device; `(n_pos_b, 1)`, `(n_pos_b, 2)`
  or `(n_pos_b, 3)` as for `pos`; `n_pos_b = n_pos` for the same map, any number for different
  maps; default: `None`
- **`phase_b_in`** — Phases of the second device; requires `pos_b`; `None` = same map;
  `(n_coeff, n_gen)`; default: `None`
- **`acf`** — Target ACF; required for two devices on the same map, must then be > -1; `(n_acf,)`;
  default: `None`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`;
  default: `None`
- **`freq_scale_b`** — Frequency scale of the second device, i.e. decorrelation distance of the
  first device divided by that of the second; must be > 0; other than 1 requires `phase_b_in`;
  default: 1.0

### Outputs:
- **`phase_out`** — Corrected phases; `(n_coeff, n_gen)`
- **`phase_b_out`** — Corrected phases of the second device; `(n_coeff, n_gen)`; `None` if
  `phase_b_in` was not given
- **`offset`** — Offsets added to the phases; `(n_gen, 1)`, or `(n_gen, 2)` with
  `phase_b_in` (column 0: `phase_in`, column 1: `phase_b_in`)
- **`mean_before`** — Sample mean before the correction; `(n_gen, 1)`: mean of the values of
  [sos.val](#sos.val) for the given arguments; with `phase_b_in` `(n_gen, 2)`: mean of the field of each
  device on its own positions, the mean of the combined value being their sum divided by `sqrt(2)`

### See also:
- [sos.val](#sos.val) (the values whose mean is removed)

---
<a id="sos.val"></a>
## sos.val
Spatially correlated random values at given positions

Also in: [C++](../cpp/sum-of-sinusoids-random-number-generator.md#sos_val) · [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_val)

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
  `pos_b` and `A` the target ACF interpolated as in [sos.acfi](#sos.acfi); `acf` and `dist` are required.
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

### Usage:
```python
val = quadriga_lib.sos.val( freq, phase, pos, pos_b, phase_b, acf, dist, freq_scale_b, \
    uniform, use_kernel )
```

### Inputs:
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

### Output:
- **`val`** — Values; one row per position, one column per generator; `(n_pos, n_gen)`

### See also:
- [sos.preset](#sos.preset) (frequencies and ACF of the predefined models)
- [sos.init_phases](#sos.init_phases) (random phases)
- [sos.map](#sos.map) (evaluation on a grid)
- [sos.randn](#sos.randn) (generator set-up and evaluation in one call)
