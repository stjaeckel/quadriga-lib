# Sum-of-sinusoids random number generator

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-sum-of-sinusoids-random-number-generator

| Function | Summary |
| --- | --- |
| [sos_acf_2d](#sos_acf_2d) | Target ACF on a 2-D grid |
| [sos_acf_approx](#sos_acf_approx) | ACF produced by a set of sinusoids |
| [sos_acf_estimate](#sos_acf_estimate) | Monte-Carlo estimate of the ACF of one generator |
| [sos_acfi](#sos_acfi) | Target ACF at given distances |
| [sos_calc_mse](#sos_calc_mse) | Approximation error of the ACF of a set of sinusoids |
| [sos_generate](#sos_generate) | Find the frequencies of a sum-of-sinusoids generator for a given ACF |
| [sos_init_phases](#sos_init_phases) | Draw random phases for a set of sum-of-sinusoids generators |
| [sos_map](#sos_map) | Spatially correlated random values on a regular grid |
| [sos_preset](#sos_preset) | Frequencies and ACF of a predefined sum-of-sinusoids generator |
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

```cpp
#include "quadriga_lib.hpp"

// 1000 positions; one row per position, columns are {x, y, z} in meters
arma::mat pos(1000, 3, arma::fill::randu);
pos *= 100.0;

// One independent generator per decorrelation distance: 10 m and 50 m
arma::vec dist_decorr = {10.0, 50.0};

// Normal distributed values with zero mean and unit variance, size [1000, 2]
arma::mat val;
quadriga_lib::sos_randn<double>(dist_decorr, pos, val, nullptr, nullptr, "Comb256", 42);

// Uniform values between 0 and 1, and integers from 1 to 6
arma::mat uni, dice;
quadriga_lib::sos_rand<double>(dist_decorr, pos, uni, nullptr, nullptr, "Comb256", 42);
quadriga_lib::sos_randi<double>(dist_decorr, pos, 6, dice, nullptr, nullptr, "Comb256", 42);
```

The last argument is the seed. With the same seed, decorrelation distances and preset, a later call returns the same
field, also at other positions. Without a seed every call gives a new field.

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

```cpp
// Frequencies of a predefined generator for a decorrelation distance of 10 m
arma::mat freq;
arma::vec acf, dist;
quadriga_lib::sos_preset<double>("Comb256", freq, &acf, &dist, 10.0);

// Phases of 20 generators; each column is one independent random field
arma::mat phase;
quadriga_lib::sos_init_phases<double>(freq.n_rows, 20, phase, 42);

// Values of all 20 generators at the positions, size [1000, 20]
arma::mat val_all;
quadriga_lib::sos_val(freq, phase, pos, val_all);

// The same fields at other positions, at any later time
arma::mat pos_new = pos + 2.5, val_new;
quadriga_lib::sos_val(freq, phase, pos_new, val_new);

// Uniform distributed output
arma::mat val_uni;
quadriga_lib::sos_val<double>(freq, phase, pos, val_uni, nullptr, nullptr, nullptr, nullptr, 1.0, true);

// Map on a regular grid, one cube of size [n_y, n_x, 1] per generator
arma::vec x = arma::regspace(0.0, 0.5, 250.0), y = arma::regspace(0.0, 0.5, 100.0);
std::vector<arma::cube> grid;
quadriga_lib::sos_map<double>(freq, phase, x, grid, &y);
```

- **Presets:** `Exp`, `Gauss` and `Comb`, each with 128, 256, 512 or 1024 sinusoids, e.g. `"Exp512"`. `Exp`
  is the exponential ACF `exp(-d/d0)`, `Gauss` is `exp(-(d/d0)^2)`, and `Comb` follows `Gauss` up to the
  decorrelation distance `d0` and `Exp` beyond it. Each doubling of the number of sinusoids doubles the
  computing time and lowers the error of the ACF by 3 to 4 dB: [sos_calc_mse](#sos_calc_mse) reports 28, 31, 34 and
  38 dB for `Exp128` to `Exp1024` and 31, 35, 39 and 42 dB for `Comb128` to `Comb1024`. `"Comb256"` is a
  good default; take a larger table if the shape of the ACF matters, a smaller one if speed does.
- **Both ends mobile:** if a value belongs to a link between two moving devices, pass both positions.
  The value then changes smoothly when either device moves.

```cpp
// Positions of the two ends of 1000 links; one value per link and generator, size [1000, 20]
arma::mat pos_tx = pos, pos_rx(1000, 3, arma::fill::randu), val_link;
pos_rx *= 100.0;
quadriga_lib::sos_val<double>(freq, phase, pos_tx, val_link, &pos_rx, nullptr, &acf, &dist);
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
  of 10 m) the results stay within the tolerance stated in [sos_val](#sos_val). The `float` versions of the
  functions read positions in single precision, i.e. rounded to about 1 mm at a distance of 10 km.
- **Threads and memory:** the functions that take noticeable time use the OpenMP threads of the process,
  except [sos_generate](#sos_generate), which runs on one thread. No result depends on the number of threads.
  [sos_val](#sos_val) computes in its output: for `double` it needs no further memory of that size for one device
  and one more array of that size for two devices; for `float` one array in double precision is added.

**Own autocorrelation functions**\
[sos_generate](#sos_generate) finds the frequencies for any ACF that is given as a table over the distance, and
[sos_calc_mse](#sos_calc_mse) reports how well the result fits. The table should reach to about 5 decorrelation
distances.

```cpp
// Target ACF from 0 to 5 decorrelation distances, here a Gaussian shape with 5 m
arma::vec dist_own = arma::regspace(0.0, 0.125, 24.875);
arma::vec acf_own = arma::exp(-arma::square(dist_own / 5.0));

// 256 sinusoids in 3-D, 300 test directions, wavelengths up to 20 decorrelation distances (100 m)
arma::mat freq_own;
quadriga_lib::sos_generate<double>(acf_own, dist_own, 256, freq_own, nullptr, nullptr, 3, true, 300, true,
                                   0.01, 0.0, 1);

// Approximation error in dB, larger is better; about 35 dB for this example
double mse = quadriga_lib::sos_calc_mse(freq_own, acf_own, dist_own);
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

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_acf_2d) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.acf_2d)

- Evaluates a rotationally symmetric autocorrelation function (ACF), given as a table over the distance,
  on a square grid
- Grid axis: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`, where `Dmax` is the last sample point;
  `n_grid = 4·n_acf - 3`, the center is element `2·n_acf - 2` (0-based)
- Linear interpolation of the table at the radial distance; 0 at radial distances above `Dmax`
- The result is the reference for [sos_acf_approx](#sos_acf_approx)
- Equally spaced means that every sample point lies within 1e-4 of the step from its place on the regular
  axis; for long single-precision tables the rounding of the sample points themselves is allowed for
- `dist_2d` may be the same object as `acf` or `dist`

### Declaration:
```cpp
void quadriga_lib::sos_acf_2d(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    arma::Mat<dtype> &acf_2d,
    arma::Col<dtype> *dist_2d = nullptr);
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, be equally spaced and finite; `[n_acf]`

### Outputs:
- **`acf_2d`** — ACF on the grid; `[n_grid, n_grid]`
- **`dist_2d`** *(optional)* — Grid axis in [m]; `[n_grid]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF produced by a set of sinusoids on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
<a id="sos_acf_approx"></a>
## sos_acf_approx
ACF produced by a set of sinusoids

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_acf_approx) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.acf_approx)

- Autocorrelation function (ACF) of the values of [sos_val](#sos_val) over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [sos_acf_2d](#sos_acf_2d): from `-2·Dmax` to `2·Dmax` with the spacing of `dist`,
  `n_grid = 4·n_acf - 3`
- `n_dim = 1`: along the axis; `[n_grid, 1, 1]`
- `n_dim = 2`: on the x-y plane; `[n_grid, n_grid, 1]`; element `(iy, ix)`
- `n_dim = 3`: on the x-y, x-z and y-z plane (slices 0, 1, 2); `[n_grid, n_grid, 3]`; the row index belongs
  to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values
- `R(-dx) = R(dx)`: each plane is point-symmetric about its center. One half is computed and the other
  half is its mirror image, so the symmetry is exact
- Equally spaced has the meaning given in [sos_acf_2d](#sos_acf_2d)
- `dist_2d` may be the same object as `dist`
- `acf_approx` cannot share memory with `freq` or `dist`; such a call throws before anything is written

### Declaration:
```cpp
void quadriga_lib::sos_acf_approx(
    const arma::Mat<dtype> &freq,
    const arma::Col<dtype> &dist,
    arma::Cube<dtype> &acf_approx,
    arma::Col<dtype> *dist_2d = nullptr);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, be equally spaced and finite; at least
  2 elements; `[n_acf]`

### Outputs:
- **`acf_approx`** — Approximated ACF; size depends on `n_dim`, see above
- **`dist_2d`** *(optional)* — Grid axis in [m]; `[n_grid]`

### See also:
- [sos_acf_2d](#sos_acf_2d) (target ACF on the same grid)
- [sos_calc_mse](#sos_calc_mse) (difference between the two)

---
<a id="sos_acf_estimate"></a>
## sos_acf_estimate
Monte-Carlo estimate of the ACF of one generator

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_acf_estimate) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.acf_estimate)

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at random
  positions, without using the frequencies other than through [sos_val](#sos_val)
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin, evaluates
  the generator there, and for each distance bin computes the correlation coefficient between the two values
  of all position pairs whose distance falls into the bin
- Positions: a matrix `[2·n_pos, 3]` is filled column by column (all x, then all y, then all z) with
  `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in [sos_init_phases](#sos_init_phases)
- Bin `n` covers the distances from the midpoint between `bins(n-1)` and `bins(n)` (excluded) to the midpoint
  between `bins(n)` and `bins(n+1)` (included). The first bin starts at 1e-5 m, the last bin ends half a mean
  bin spacing above `bins(end)`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices on the same map. The first `n_pos` positions are the first device,
  the second `n_pos` positions the second device, and the value of a link is [sos_val](#sos_val) with both positions.
  Two links form a pair if the distance between their first devices falls into the bin and the distance
  between their second devices does not exceed the upper limit of the bin; requires `acf` and `dist`
- The estimate describes one realisation; it scatters around the ACF of the sinusoids
- The pairs are searched in parallel and added in a fixed order; the result does not depend on the number
  of threads. Run time and memory: `2·n_pos^2` pairs for one device, at most 36 MB for the pairs of a block
  of rows
- The outputs may be the same objects as inputs

### Declaration:
```cpp
void quadriga_lib::sos_acf_estimate(
    const arma::Mat<dtype> &freq,
    const arma::Col<dtype> &phase,
    const arma::Col<dtype> &bins,
    arma::uword n_pos,
    arma::Col<dtype> &acf_est,
    arma::Col<dtype> *dist_est = nullptr,
    arma::Col<dtype> *acf_est_dual = nullptr,
    const arma::Col<dtype> *acf = nullptr,
    const arma::Col<dtype> *dist = nullptr,
    arma::sword seed = -1);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `[n_coeff, 3]`
- **`phase`** — Phases of the generator in [rad]; `[n_coeff]`
- **`bins`** — Bin centers in [m]; must be >= 0, increase strictly and be finite; at least 2 elements;
  `[n_bin]`
- **`n_pos`** — Number of random positions per device; must be > 1
- **`acf`** *(optional)* — Target ACF; required for `acf_est_dual`, must then be > -1; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`seed`** *(optional)* — Seed for the positions; must be >= -1; `-1` seeds from the system random device

### Outputs:
- **`acf_est`** — Estimated ACF; `[n_bin]`
- **`dist_est`** *(optional)* — Mean distance of the pairs in each bin in [m]; `[n_bin]`
- **`acf_est_dual`** *(optional)* — Estimated ACF for two devices on the same map; `[n_bin]`

### See also:
- [sos_acf_approx](#sos_acf_approx) (ACF of the sinusoids in closed form)
- [sos_val](#sos_val) (evaluation of the generator)

---
<a id="sos_acfi"></a>
## sos_acfi
Target ACF at given distances

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_acfi) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.acfi)

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table
- The same rule is applied by [sos_val](#sos_val) and [sos_remove_bias](#sos_remove_bias) for two devices on the same map
- An empty `d` returns an empty `val`
- `val` may be the same object as `acf`, `dist` or `d`

### Declaration:
```cpp
void quadriga_lib::sos_acfi(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    const arma::Col<dtype> &d,
    arma::Col<dtype> &val);
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, increase strictly and be finite; `[n_acf]`
- **`d`** — Distances in [m]; must be >= 0; `[n_d]`

### Output:
- **`val`** — ACF at the distances `d`; `[n_d]`

### See also:
- [sos_preset](#sos_preset) (ACF tables of the predefined models)

---
<a id="sos_calc_mse"></a>
## sos_calc_mse
Approximation error of the ACF of a set of sinusoids

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_calc_mse) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.calc_mse)

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- Return value: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`. A direction and its opposite
  give the same ACF
- `mse_core`: error on the grid of [sos_acf_approx](#sos_acf_approx) against [sos_acf_2d](#sos_acf_2d), for the grid points within a
  radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error is averaged over
  the three planes
- `mse_all`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond `Dmax`
- For `n_dim = 1`, `mse_core` is evaluated on the sample points `dist`, `mse_all` on the axis from
  `-2·Dmax` to `2·Dmax`, and the return value equals `mse_core`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`. The
  target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- `mse_core` and `mse_all` need the 2-D grids and are only computed when requested
- A perfect fit has a mean squared error of 0 and is returned as +Inf dB; this happens for a target that
  the sinusoids reproduce exactly at all sample points, e.g. sinusoids at zero frequency and a target of 1
- The test directions are evaluated in parallel; the result does not depend on the number of threads
- Equally spaced has the meaning given in [sos_acf_2d](#sos_acf_2d)
- `acf_tail` and `test_dir` may be the same objects as inputs

### Declaration:
```cpp
dtype quadriga_lib::sos_calc_mse(
    const arma::Mat<dtype> &freq,
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    dtype *mse_core = nullptr,
    dtype *mse_all = nullptr,
    arma::Col<dtype> *acf_tail = nullptr,
    arma::Mat<dtype> *test_dir = nullptr,
    arma::uword n_dir = 10);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, be equally spaced and finite; `[n_acf]`
- **`n_dir`** *(optional)* — Number of test directions; must be > 0; ignored for `n_dim = 1`

### Outputs:
- **`mse_core`** *(optional)* — Error within the core radius in [dB]
- **`mse_all`** *(optional)* — Error on the complete grid in [dB]
- **`acf_tail`** *(optional)* — Approximated ACF beyond the fitted range; `[4]`
- **`test_dir`** *(optional)* — Test directions in [rad]; `[n_dir, 2]` with columns `{azimuth, elevation}`
  for `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

### Returns:
- Error along the test directions in [dB]

### See also:
- [sos_generate](#sos_generate) (finds the frequencies; minimises the error returned here)
- [sos_acf_approx](#sos_acf_approx), [sos_acf_2d](#sos_acf_2d) (the two grids that are compared)

---
<a id="sos_generate"></a>
## sos_generate
Find the frequencies of a sum-of-sinusoids generator for a given ACF

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_generate) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.generate)

- Searches `n_coeff` frequencies such that the autocorrelation function (ACF) of the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, approximates the target ACF in all directions
- Each sinusoid has a fixed direction and a signed frequency along it. `n_dim = 1`: no direction;
  `n_dim = 2`: directions on the half circle; `n_dim = 3`: directions on the upper hemisphere.
  `uniform_smp = true` places them regularly (equally spaced azimuth, spiral lattice, as the test directions
  of [sos_calc_mse](#sos_calc_mse)), `false` draws them at random
- Iteration: in each pass every sinusoid is re-fitted once. Its frequency is chosen to fit the residual
  between the target and the contribution of all other sinusoids along its own direction: coarse search
  over a grid with a spacing of `0.1 / max(dist)`, then a step refinement. The
  new frequency is kept only if the mean squared error over the test directions decreases. The iteration
  stops after a pass that changes nothing or lowers the error by less than a factor of 1e-6, or after
  1000 passes
- `random_init = true` starts from uniformly random frequencies; `false` builds the set up sinusoid by
  sinusoid in the first pass
- The fit covers the range of `dist` only. The ACF of the result beyond the last sample point is not
  controlled; sinusoids with a frequency near zero keep it above zero at all larger distances
  (see `acf_tail` of [sos_calc_mse](#sos_calc_mse)). To prevent this, set `freq_min`; the tables of [sos_preset](#sos_preset) use
  `0.05 / d` with the decorrelation distance `d`
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse in
  others; the tables of [sos_preset](#sos_preset) were made with 300
- `freq_min` and `freq_max` limit the magnitude of the frequency along the direction of each sinusoid;
  0 selects the default: no lower limit, and `15 / max(dist)` as upper limit
- The search grid has `( freq_max - freq_min ) · max(dist) / 0.1 + 1` points, and its cosines are stored for
  all sample points. It must not have more than 100000 points (the default has 151), which limits
  `freq_max - freq_min` to less than `10000 / max(dist)`
- The function runs on one thread
- `ase` and `test_dir` may be the same objects as `acf` or `dist`
- `test_dir` and the reported error use the same test directions as [sos_calc_mse](#sos_calc_mse) with the same `n_dir`
- All computations are done in double precision for both instantiations
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject decisions, so a
  difference in the last bit of the input or of the math library can lead to another table of equivalent
  quality

### Declaration:
```cpp
void quadriga_lib::sos_generate(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    arma::uword n_coeff,
    arma::Mat<dtype> &freq,
    arma::Col<dtype> *ase = nullptr,
    arma::Mat<dtype> *test_dir = nullptr,
    arma::uword n_dim = 3,
    bool uniform_smp = true,
    arma::uword n_dir = 10,
    bool random_init = true,
    dtype freq_min = 0.0,
    dtype freq_max = 0.0,
    arma::sword seed = -1);
```

### Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, increase strictly and be finite; `[n_acf]`
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_dim`** *(optional)* — Number of dimensions; 1, 2 or 3
- **`uniform_smp`** *(optional)* — Regular (`true`) or random (`false`) directions of the sinusoids
- **`n_dir`** *(optional)* — Number of test directions; must be > 0; ignored for `n_dim = 1`
- **`random_init`** *(optional)* — Start from random frequencies (`true`) or build up sinusoid by sinusoid
  (`false`)
- **`freq_min`** *(optional)* — Lower limit of the frequency magnitude in [1/m]; must be >= 0; 0 = none
- **`freq_max`** *(optional)* — Upper limit of the frequency magnitude in [1/m]; must be >= 0 and, if not 0,
  larger than `freq_min`; 0 = `15 / max(dist)`; limited by the size of the search grid, see above
- **`seed`** *(optional)* — Seed; must be >= -1; `-1` seeds from the system random device

### Outputs:
- **`freq`** — Frequencies in [1/m]; `[n_coeff, n_dim]`
- **`ase`** *(optional)* — Mean squared error over the test directions after each pass, linear scale;
  `[n_pass]`
- **`test_dir`** *(optional)* — Test directions in [rad]; `[n_dir, 2]` with columns `{azimuth, elevation}`
  for `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

### See also:
- [sos_calc_mse](#sos_calc_mse) (error of a frequency table)
- [sos_preset](#sos_preset) (precomputed tables)

---
<a id="sos_init_phases"></a>
## sos_init_phases
Draw random phases for a set of sum-of-sinusoids generators

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_init_phases) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.init_phases)

- Each column of `phase` is one realisation of the random field of a generator; see [sos_val](#sos_val)
- Values are uniform in the interval from -pi (included) to pi (excluded)
- Random stream: `std::mt19937_64` seeded with `seed`; each value is `2·pi · ( u - 0.5 )`, where `u` is the
  upper 53 bits of one generator output divided by 2^53; no distribution class of the standard library is
  used, so a seed >= 0 gives the same phases on all platforms
- The matrix is filled column by column from one stream: for the same seed, column `g` does not depend on
  `n_gen`
- For `dtype = float` the values are rounded from double; a value that rounds to pi is returned as -pi
- A seed stands for a set of phases. Generators that use phases of the same seed with different
  frequencies (another decorrelation distance or preset) produce related fields, which are equal at the
  coordinate origin and correlated around it; draw independent quantities as columns of one call or with
  different seeds

### Declaration:
```cpp
void quadriga_lib::sos_init_phases(
    arma::uword n_coeff,
    arma::uword n_gen,
    arma::Mat<dtype> &phase,
    arma::sword seed = -1);
```

### Inputs:
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_gen`** — Number of generators; must be > 0
- **`seed`** *(optional)* — Seed of the random stream; must be >= -1; `-1` seeds from the system random
  device and gives different phases on each call

### Output:
- **`phase`** — Phases in [rad]; `[n_coeff, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluates the generators)
- [sos_preset](#sos_preset) (frequencies that belong to the phases)

---
<a id="sos_map"></a>
## sos_map
Spatially correlated random values on a regular grid

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_map) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.map)

- Evaluates the generators of [sos_val](#sos_val) for one device at all combinations of the grid vectors
- Output in meshgrid order: element `(iy, ix, iz)` of a cube belongs to the position `{x(ix), y(iy), z(iz)}`
- One cube per generator, i.e. per column of `phase`
- The number of grid vectors given must not exceed the number of columns of `freq`; an omitted or empty
  `y` or `z` means a single sample at 0 on that axis
- `z` can be given without `y` if `freq` has 3 columns
- The values are those of [sos_val](#sos_val) at the grid points, bit for bit; kernels, accuracy and the Uniform
  mapping are described there
- Cubes of the correct size keep their memory. If they lie one after the other in memory (one block of
  `n_gen` cubes, as the MATLAB and Python interfaces allocate it), the values are computed in place;
  otherwise they are computed in a matrix `[ny·nx·nz, n_gen]` and copied
- A cube that shares memory with `freq` or `phase` is an error, before anything is written

### Declaration:
```cpp
void quadriga_lib::sos_map(
    const arma::Mat<dtype> &freq,
    const arma::Mat<dtype> &phase,
    const arma::Col<dtype> &x,
    std::vector<arma::Cube<dtype>> &map,
    const arma::Col<dtype> *y = nullptr,
    const arma::Col<dtype> *z = nullptr,
    bool uniform = false,
    int use_kernel = 0);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`x`** — Grid points on the x-axis in [m]; `[nx]`
- **`y`** *(optional)* — Grid points on the y-axis in [m]; requires `n_dim >= 2`; omitted = one sample at 0,
  `ny = 1`; `[ny]`
- **`z`** *(optional)* — Grid points on the z-axis in [m]; requires `n_dim = 3`; omitted = one sample at 0,
  `nz = 1`; `[nz]`
- **`uniform`** *(optional)* — Output distribution: `false` = Normal(0,1), `true` = Uniform between 0 and 1
- **`use_kernel`** *(optional)* — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  throws if the requested kernel is unavailable

### Output:
- **`map`** — Values; vector of length `n_gen` with one cube of size `[ny, nx, nz]` per generator

### See also:
- [sos_val](#sos_val) (evaluation at arbitrary positions, two devices)

---
<a id="sos_preset"></a>
## sos_preset
Frequencies and ACF of a predefined sum-of-sinusoids generator

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_preset) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.preset)

- Returns a precomputed frequency table for [sos_val](#sos_val) together with the autocorrelation function (ACF)
  it was fitted to
- Presets: `Exp128`, `Exp256`, `Exp512`, `Exp1024`, `Gauss128`, `Gauss256`, `Gauss512`, `Gauss1024`,
  `Comb128`, `Comb256`, `Comb512`, `Comb1024`. The number is the number of sinusoids, `n_coeff`. The name is
  matched exactly, including upper and lower case
- ACF types, with the distance `D` and the decorrelation distance `d`: `Exp`: `R = exp( -D/d )`;
  `Gauss`: `R = exp( -(D/d)^2 )`; `Comb`: `Gauss` for `D <= d`, `Exp` beyond. All three have `R(d) = exp(-1)`
- The ACF is returned as a table with 200 samples from 0 to `4.975·d` in steps of `0.025·d`
- The tables are stored for `d = 10 m`. For another decorrelation distance the distances are multiplied by
  `d/10` and the frequencies by `10/d`
- All tables are 3-dimensional and can be used with positions of 1, 2 or 3 columns
- The outputs cannot be the same object or share memory with each other; such a call throws before anything
  is written
- How the tables were made: [sos_generate](#sos_generate) with the 200-sample ACF for 10 m, `n_dim = 3`,
  `uniform_smp = true`, `n_dir = 300`, `random_init = true`, `freq_min = 0.005`, `freq_max = 15 / 49.75`,
  and the seeds 39, 3, 2, 5 (Exp), 13, 32, 6, 21 (Gauss), 13, 25, 37, 47 (Comb) for 128, 256, 512 and 1024
  sinusoids. Sinusoid `n` points in direction `n` of the spiral lattice described in [sos_calc_mse](#sos_calc_mse)
- The lower frequency limit means that no sinusoid has a wavelength above 20 decorrelation distances. The
  ACF of the tables therefore stays near zero beyond the fitted range (magnitude below 0.015 at 10, 20, 50
  and 100 decorrelation distances), and the mean of the generator over a large area is not offset by
  sinusoids of nearly zero frequency

### Declaration:
```cpp
void quadriga_lib::sos_preset(
    const std::string &name,
    arma::Mat<dtype> &freq,
    arma::Col<dtype> *acf = nullptr,
    arma::Col<dtype> *dist = nullptr,
    dtype dist_decorr = 10.0);
```

### Inputs:
- **`name`** — Name of the preset, e.g. `"Comb256"`
- **`dist_decorr`** *(optional)* — Decorrelation distance in [m]; must be > 0

### Outputs:
- **`freq`** — Frequencies in [1/m]; `[n_coeff, 3]`
- **`acf`** *(optional)* — Target ACF; `[200]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; `[200]`

### See also:
- [sos_init_phases](#sos_init_phases) (random phases for the generator)
- [sos_val](#sos_val) (evaluation)
- [sos_generate](#sos_generate) (frequencies for other ACFs)

---
<a id="sos_rand"></a>
## sos_rand
Spatially correlated Uniform distributed random numbers

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_rand) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.rand)

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a preset,
  draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second device on
  its own map uses column `g` of a second stream, `sos_init_phases( n_coeff, n_gen, seed_b )`, where
  `seed_b` is the seed with bit 62 flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both
  streams are random. The phases of a generator therefore depend neither on the number of generators nor
  on whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr` for that
  generator): both devices are on the same map, the value does not change when they swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the sum of both
  maps divided by `sqrt(2)`
- The Normal distributed value `s` of the generator is mapped to `0.5 · erfc( -s / sqrt(2) )`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on all
  platforms: the same positions give the same values and nearby positions similar values, also in
  separate calls. The kernel is chosen automatically, so values from machines with and without AVX2
  differ by the tolerance stated in [sos_val](#sos_val)
- A seed selects the phases, not the field: calls with the same seed but other decorrelation distances
  or another preset return related fields, equal at the coordinate origin and correlated around it. Use
  different seeds, or several entries of `dist_decorr` in one call, for quantities that must be independent
- `val` may be the same object as `pos` or `pos_b`

### Declaration:
```cpp
void quadriga_lib::sos_rand(
    const arma::Col<dtype> &dist_decorr,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Col<dtype> *dist_decorr_b = nullptr,
    const std::string &preset = "Comb256",
    arma::sword seed = -1);
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances in [m]; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device in [m]; requires `pos_b`;
  must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** *(optional)* — Name of the preset, see [sos_preset](#sos_preset)
- **`seed`** *(optional)* — Seed of the phases; must be >= -1; `-1` seeds from the system random device

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

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_randi) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.randi)

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a preset,
  draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second device on
  its own map uses column `g` of a second stream, `sos_init_phases( n_coeff, n_gen, seed_b )`, where
  `seed_b` is the seed with bit 62 flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both
  streams are random. The phases of a generator therefore depend neither on the number of generators nor
  on whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr` for that
  generator): both devices are on the same map, the value does not change when they swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the sum of both
  maps divided by `sqrt(2)`
- The Uniform value `u` of [sos_rand](#sos_rand) is mapped to `ceil( u · imax )`, limited to 1 ... `imax`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on all
  platforms: the same positions give the same values and nearby positions similar values, also in
  separate calls. The kernel is chosen automatically, so values from machines with and without AVX2
  differ by the tolerance stated in [sos_val](#sos_val), and
  an integer can then differ by 1 where the Uniform value lies at a step
- A seed selects the phases, not the field: calls with the same seed but other decorrelation distances
  or another preset return related fields, equal at the coordinate origin and correlated around it. Use
  different seeds, or several entries of `dist_decorr` in one call, for quantities that must be independent
- `val` may be the same object as `pos` or `pos_b`

### Declaration:
```cpp
void quadriga_lib::sos_randi(
    const arma::Col<dtype> &dist_decorr,
    const arma::Mat<dtype> &pos,
    arma::uword imax,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Col<dtype> *dist_decorr_b = nullptr,
    const std::string &preset = "Comb256",
    arma::sword seed = -1);
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances in [m]; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`imax`** — Largest value; must be > 0
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device in [m]; requires `pos_b`;
  must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** *(optional)* — Name of the preset, see [sos_preset](#sos_preset)
- **`seed`** *(optional)* — Seed of the phases; must be >= -1; `-1` seeds from the system random device

### Output:
- **`val`** — Integer values from 1 to `imax`, each with approximately the same probability; one row per
  position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_randn](#sos_randn), [sos_rand](#sos_rand) (Normal and Uniform values)

---
<a id="sos_randn"></a>
## sos_randn
Spatially correlated Normal distributed random numbers

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_randn) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.randn)

- Convenience function: sets up one sum-of-sinusoids generator per entry of `dist_decorr` from a preset,
  draws its phases and evaluates it at the given positions; see [sos_val](#sos_val) for the model
- The generators are independent of each other and may have different decorrelation distances
- Phases: generator `g` uses column `g` of `sos_init_phases( n_coeff, n_gen, seed )`. A second device on
  its own map uses column `g` of a second stream, `sos_init_phases( n_coeff, n_gen, seed_b )`, where
  `seed_b` is the seed with bit 62 flipped, i.e. `seed + 2^62` for seeds below 2^62; for `seed = -1` both
  streams are random. The phases of a generator therefore depend neither on the number of generators nor
  on whether a second device is given
- A decorrelation distance of 0 is replaced by 0.1 m
- One device (`pos_b` omitted): values of the generator at `pos`
- Two devices, same decorrelation distance (`dist_decorr_b` omitted, or equal to `dist_decorr` for that
  generator): both devices are on the same map, the value does not change when they swap positions
- Two devices, different decorrelation distances: each device has its own map; the value is the sum of both
  maps divided by `sqrt(2)`
- Each call with `seed = -1` gives a new realisation. With a seed >= 0 the phases are the same on all
  platforms: the same positions give the same values and nearby positions similar values, also in
  separate calls. The kernel is chosen automatically, so values from machines with and without AVX2
  differ by the tolerance stated in [sos_val](#sos_val)
- A seed selects the phases, not the field: calls with the same seed but other decorrelation distances
  or another preset return related fields, equal at the coordinate origin and correlated around it. Use
  different seeds, or several entries of `dist_decorr` in one call, for quantities that must be independent
- `val` may be the same object as `pos` or `pos_b`

### Declaration:
```cpp
void quadriga_lib::sos_randn(
    const arma::Col<dtype> &dist_decorr,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Col<dtype> *dist_decorr_b = nullptr,
    const std::string &preset = "Comb256",
    arma::sword seed = -1);
```

### Inputs:
- **`dist_decorr`** — Decorrelation distances in [m]; one generator per entry; must be >= 0; `[n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`dist_decorr_b`** *(optional)* — Decorrelation distances of the second device in [m]; requires `pos_b`;
  must be >= 0; omitted or empty = same as `dist_decorr`; `[n_gen]`
- **`preset`** *(optional)* — Name of the preset, see [sos_preset](#sos_preset)
- **`seed`** *(optional)* — Seed of the phases; must be >= -1; `-1` seeds from the system random device

### Output:
- **`val`** — Values, approximately Normal distributed with zero mean and unit variance; one row per
  position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_val](#sos_val) (evaluation with given frequencies and phases)
- [sos_preset](#sos_preset), [sos_init_phases](#sos_init_phases) (the two steps this function combines)
- [sos_rand](#sos_rand), [sos_randi](#sos_randi) (Uniform and integer values)

---
<a id="sos_remove_bias"></a>
## sos_remove_bias
Remove the sample mean of sum-of-sinusoids generators over a set of positions

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_remove_bias) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.remove_bias)

- One realisation of a spatially correlated random field has a non-zero mean over a finite set of
  positions. Its variance over the realisations is the mean of the autocorrelation function (ACF) over all
  pairs of positions, which is much larger than `1 / n_pos` when the positions lie within a few
  decorrelation distances of each other. This is a property of any correlated field, not an error of the
  generator
- This function adds one offset to all phases of a generator such that the mean of the Normal distributed
  values of [sos_val](#sos_val) over the given positions becomes zero
- Closed form, no iteration: with `z_n = mean over the positions of exp( j·2·pi · f_n·x )` and
  `Z = sum_n z_n · exp( j·phi_n )`, the mean is `sqrt(2/n_coeff) · Re( exp( j·offset ) · Z )`, which
  vanishes for `offset = pi/2 - arg(Z)` and for that value minus pi; the one with the smaller magnitude is
  applied, so the offset lies between -pi/2 and pi/2
- One device (`pos_b` omitted): one offset per generator
- Two devices on the same map (`pos_b` given, `phase_b` omitted): one offset per generator that zeroes the
  mean of the combined, normalised value; `acf` and `dist` are required
- Two devices on different maps (`pos_b` and `phase_b` given): each device is treated on its own, `phase`
  with `pos` and `phase_b` with `pos_b` and the frequencies `freq · freq_scale_b`; two offsets per generator.
  The mean of the combined value is then zero as well. `phase_b` cannot share memory with `phase`: one
  matrix cannot hold two corrections
- The arguments that are written, `phase`, `phase_b`, `offset` and `mean_before`, cannot share memory with
  another argument, not even in part, and `offset` and `mean_before` cannot be the same object. Such a call
  throws and leaves all arguments unchanged
- Effect on the statistics: the result is the field conditioned on a zero mean over the positions. The
  variance at a position drops from 1 to `1 - c^2 / v`, where `v` is the variance of the sample mean and
  `c` the mean ACF between that position and all positions; on average over the positions the loss is at
  least `v`. It is small only if the positions spread over many decorrelation distances. For positions
  within about one decorrelation distance most of the variance is removed
- Rule of thumb, measured for 200 positions in a square with the edge `L` and the decorrelation distance
  `d` (`Comb256`): the variance drops by 74 % for `L = d`, 44 % for `L = 2·d`, 14 % for `L = 5·d`, 5 % for
  `L = 10·d`, 2.5 % for `L = 20·d` and less than 1 % from `L = 50·d` on
- The zero mean holds for exactly these positions and frequencies. It is lost when the phases are used
  with other frequencies (another decorrelation distance) or positions
- The mean of the Uniform output of [sos_val](#sos_val) is not exactly 0.5 afterwards
- Phases are returned in the interval from -pi (included) to pi (excluded). A generator whose `abs(Z)` is
  below `1e-14 · n_coeff` already has a zero mean for every offset and is left unchanged, also if its
  phases lie outside that interval
- Two devices on the same map need an ACF above -1 at all sample points, as in [sos_val](#sos_val)
- The sums over the positions are computed in parallel over the sinusoids; the result does not depend
  on the number of threads

### Declaration:
```cpp
void quadriga_lib::sos_remove_bias(
    const arma::Mat<dtype> &freq,
    arma::Mat<dtype> &phase,
    const arma::Mat<dtype> &pos,
    const arma::Mat<dtype> *pos_b = nullptr,
    arma::Mat<dtype> *phase_b = nullptr,
    const arma::Col<dtype> *acf = nullptr,
    const arma::Col<dtype> *dist = nullptr,
    arma::Mat<dtype> *offset = nullptr,
    arma::Mat<dtype> *mean_before = nullptr,
    dtype freq_scale_b = 1.0);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; updated in-place; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos_b, 1]`, `[n_pos_b, 2]` or `[n_pos_b, 3]` as for `pos`; `n_pos_b = n_pos` for the same map, any
  number for different maps
- **`phase_b`** *(optional)* — Phases of the second device in [rad]; requires `pos_b`; omitted or empty =
  same map; updated in-place; no memory shared with `phase`; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map, must then be > -1;
  `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`freq_scale_b`** *(optional)* — Frequency scale of the second device, i.e. decorrelation distance of
  the first device divided by that of the second; must be > 0; other than 1 requires `phase_b`

### Outputs:
- **`offset`** *(optional)* — Offsets added to the phases in [rad]; `[n_gen, 1]`, or `[n_gen, 2]` with
  `phase_b` (column 0: `phase`, column 1: `phase_b`)
- **`mean_before`** *(optional)* — Sample mean before the correction; `[n_gen, 1]`: mean of the values of
  [sos_val](#sos_val) for the given arguments; with `phase_b` `[n_gen, 2]`: mean of the field of each device on its
  own positions, the mean of the combined value being their sum divided by `sqrt(2)`

### See also:
- [sos_val](#sos_val) (the values whose mean is removed)

---
<a id="sos_val"></a>
## sos_val
Spatially correlated random values at given positions

Also in: [MATLAB](../mex/sum-of-sinusoids-random-number-generator.md#sos_val) · [Python](../python/sum-of-sinusoids-random-number-generator.md#sos.val)

- Sum-of-sinusoids generator: at position `x` a generator with `n_coeff` sinusoids produces
  `s(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )`, with the frequencies `f_n` (rows of `freq`)
  and the phases `phi_n` (one column of `phase`)
- For uniformly random phases, `s` has zero mean, unit variance and the autocorrelation function (ACF)
  `(1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, which approximates the target ACF the frequencies were
  fitted to; with many sinusoids the values are approximately Normal distributed
- All generators of one call share the frequencies; each column of `phase` is one generator. The sines and
  cosines of the positions are computed once and reused for all generators
- Positions may have fewer columns than `freq`; missing coordinates are 0
- One device (`pos_b` omitted): `val = s(pos)`
- Two devices on the same map (`pos_b` given, `phase_b` omitted): both devices use `phase`;
  `val = ( s(pos) + s(pos_b) ) / sqrt( 2 + 2·A(D) )`, where `D` is the distance between `pos` and `pos_b`
  and `A` the target ACF interpolated as in [sos_acfi](#sos_acfi); `acf` and `dist` are required. The value does
  not change when the two devices swap their positions
- Two devices on different maps (`pos_b` and `phase_b` given):
  `val = ( s_a(pos) + s_b(pos_b) ) / sqrt(2)`; `s_b` uses `phase_b` and the frequencies
  `freq · freq_scale_b`; `acf` and `dist` are ignored
- The mode follows from the arguments only; phase values are never compared. Passing the same matrix as
  `phase` and `phase_b` selects the two-map formula
- `freq_scale_b` other than 1 requires `phase_b`: one map with two decorrelation distances has no
  consistent ACF
- Two devices on the same map need an ACF above -1 at all sample points, because the value is divided by
  `sqrt( 2 + 2·A(D) )`
- `uniform = true` maps the values to `0.5 · erfc( -val / sqrt(2) )`, which is uniform between 0 and 1
  as far as `val` is Normal distributed. The limits 0 and 1 are included: a value far out in the tail is
  rounded to them, for `dtype = float` every value above 5.4 gives 1
- Kernels: GENERIC computes in double precision. AVX2 reduces the argument `f_n·x` to one period in
  double precision, evaluates sine and cosine in single precision and accumulates in double precision;
  its values differ from GENERIC by less than `4e-7 · sqrt( 2·n_coeff )`, typically by less than 1e-6
- Accuracy far from the origin: the argument `f_n·x` is known to about 1e-16 of its size. The bound
  between the kernels holds up to about 1e9 periods of the sinusoids; beyond that both kernels lose
  accuracy at the same rate, and from 2^53 periods on every position returns the value at the origin
- A NaN or infinite coordinate gives NaN at that position and does not affect other positions
- The value at a position does not depend on the other positions or generators of the call, nor on the
  number of threads
- For `dtype = float` the inputs are converted to double, the computation is the same and the result is
  rounded to single precision
- Memory: for `dtype = double` the values are computed in `val`; two devices need one more array of the
  size of `val`, and `dtype = float` one array of that size in double precision
- `val` cannot be the same object as an input, nor use the memory of one: it is resized and written while
  the inputs are read. Such a call throws and leaves `val` unchanged

### Declaration:
```cpp
void quadriga_lib::sos_val(
    const arma::Mat<dtype> &freq,
    const arma::Mat<dtype> &phase,
    const arma::Mat<dtype> &pos,
    arma::Mat<dtype> &val,
    const arma::Mat<dtype> *pos_b = nullptr,
    const arma::Mat<dtype> *phase_b = nullptr,
    const arma::Col<dtype> *acf = nullptr,
    const arma::Col<dtype> *dist = nullptr,
    dtype freq_scale_b = 1.0,
    bool uniform = false,
    int use_kernel = 0);
```

### Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`; the number of columns may differ from `pos`
- **`phase_b`** *(optional)* — Phases of the second device in [rad]; requires `pos_b`; omitted or empty =
  same map; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map, must then be > -1;
  `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`freq_scale_b`** *(optional)* — Frequency scale of the second device, i.e. decorrelation distance of
  the first device divided by that of the second; must be > 0
- **`uniform`** *(optional)* — Output distribution: `false` = Normal(0,1), `true` = Uniform between 0 and 1
- **`use_kernel`** *(optional)* — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  throws if the requested kernel is unavailable

### Output:
- **`val`** — Values; one row per position, one column per generator; `[n_pos, n_gen]`

### See also:
- [sos_preset](#sos_preset) (frequencies and ACF of the predefined models)
- [sos_init_phases](#sos_init_phases) (random phases)
- [sos_map](#sos_map) (evaluation on a grid)
- [sos_randn](#sos_randn) (generator set-up and evaluation in one call)
