// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos.preset
Spatial frequencies and ACF of a predefined sum-of-sinusoids generator

- Returns a precomputed frequency table for [[sos.val]] together with the autocorrelation
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

## Usage:
```
spatial_freq, acf, dist = quadriga_lib.sos.preset( name, dist_decorr )
```

## Inputs:
- **`name`** — Name of the preset, e.g. "Comb256"; `str`
- **`dist_decorr`** — Decorrelation distance; must be > 0; default: 10.0

## Outputs:
- **`spatial_freq`** — Spatial frequencies; `(n_coeff, 3)`
- **`acf`** — Target ACF; `(200,)`
- **`dist`** — Sample points of the ACF; `(200,)`

## See also:
- [[sos.init_phases]] (random phases for the generator)
- [[sos.val]] (evaluation)
- [[sos.generate]] (frequencies for other ACFs)
MD!*/

py::tuple sos_preset(const std::string &name,
                     double dist_decorr)
{
    // Output allocation; the number of sinusoids is known only after the call
    arma::vec acf, dist;
    auto acf_py = qd_python_init_output((arma::uword)200, &acf);
    auto dist_py = qd_python_init_output((arma::uword)200, &dist);

    const arma::mat spatial_freq = quadriga_lib::sos_preset<double>(name, &acf, &dist, dist_decorr);

    auto freq_py = qd_python_copy2numpy(&spatial_freq);

    return py::make_tuple(freq_py, acf_py, dist_py);
}

/*!MD
# sos.init_phases
Draw random phases for a set of sum-of-sinusoids generators

- Each column of `phase` is one realisation of the random field of a generator; see [[sos.val]]
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

## Usage:
```
phase = quadriga_lib.sos.init_phases( n_coeff, n_gen, *, seed )
```

## Inputs:
- **`n_coeff`** — Number of sinusoids; `int`; must be > 0
- **`n_gen`** — Number of generators; `int`; must be > 0
- **`seed`** — Seed of the random stream; `int`; must be >= -1; -1 seeds from the system random
  device and gives different phases on each call; default: -1

## Output:
- **`phase`** — Phases; `(n_coeff, n_gen)`

## See also:
- [[sos.val]] (evaluates the generators)
- [[sos.preset]] (frequencies that belong to the phases)
MD!*/

py::array_t<double> sos_init_phases(arma::uword n_coeff,
                                    arma::uword n_gen,
                                    arma::sword seed)
{
    qd_python_check_output_size({n_coeff, n_gen}, sizeof(double)); // ValueError before the C++ allocation
    const arma::mat phase = quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, seed);
    return qd_python_copy2numpy(&phase);
}

/*!MD
# sos.generate
Find the spatial frequencies of a sum-of-sinusoids generator for a given ACF

- Searches `n_coeff` frequencies such that the autocorrelation function (ACF) of the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, approximates the target ACF in all directions
- Each sinusoid has a fixed direction and a signed frequency along it. `n_dim = 1`: no direction;
  `n_dim = 2`: directions on the half circle; `n_dim = 3`: directions on the upper hemisphere.
  `uniform_smp = True` places them regularly (equally spaced azimuth, spiral lattice, as the test
  directions of [[sos.calc_mse]]), `False` draws them at random
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
  (see `acf_tail` of [[sos.calc_mse]]). To prevent this, set `spatial_freq_min`
- `spatial_freq_min` and `spatial_freq_max` limit the magnitude of the frequency along the direction of each
  sinusoid; by default there is no lower limit, and the upper limit is `15 / max(dist)`
- The search grid has `( spatial_freq_max - spatial_freq_min ) · max(dist) / 0.1 + 1` points, and its cosines are
  stored for all sample points. It must not have more than 100000 points (the default has 151),
  which limits `spatial_freq_max - spatial_freq_min` to less than `10000 / max(dist)`
- The function runs on one thread
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse
  in others; the predefined tables of [[sos.preset]] were made with 300
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject
  decisions, so a difference in the last bit of the input or of the math library can lead to
  another table of equivalent quality

## Usage:
```
spatial_freq, ase, test_dir = quadriga_lib.sos.generate( acf, dist, n_coeff, n_dim, *, uniform_smp, \
    n_dir, random_init, spatial_freq_min, spatial_freq_max, seed )
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `(n_acf,)`
- **`n_coeff`** — Number of sinusoids; `int`; must be > 0
- **`n_dim`** — Number of dimensions; `int`; 1, 2 or 3; default: 3
- **`uniform_smp`** — Regular (`True`) or random (`False`) directions of the sinusoids;
  default: `True`
- **`n_dir`** — Number of test directions; `int`; must be > 0; ignored for `n_dim = 1`; default: 10
- **`random_init`** — Start from random frequencies (`True`) or build up sinusoid by sinusoid
  (`False`); default: `True`
- **`spatial_freq_min`** — Lower limit of the frequency magnitude; must be >= 0; 0 = none;
  default: 0.0
- **`spatial_freq_max`** — Upper limit of the frequency magnitude; NaN or <= 0: default `15 / max(dist)`;
  must be larger than `spatial_freq_min`; limited by the size of the search grid, see above;
  default: NaN
- **`seed`** — Seed; `int`; must be >= -1; -1 seeds from the system random device; default: -1

## Outputs:
- **`spatial_freq`** — Spatial frequencies; `(n_coeff, n_dim)`
- **`ase`** — Mean squared error over the test directions after each pass, linear scale;
  `(n_pass,)`
- **`test_dir`** — Test directions; `(n_dir, 2)` with columns `(azimuth, elevation)` for
  `n_dim = 3`; `(n_dir, 1)` with the azimuth for `n_dim = 2`; empty `(0, 0)` for `n_dim = 1`

## See also:
- [[sos.calc_mse]] (error of a frequency table)
- [[sos.preset]] (precomputed tables)
MD!*/

py::tuple sos_generate(const py::array_t<double> &acf,
                       const py::array_t<double> &dist,
                       arma::uword n_coeff,
                       arma::uword n_dim,
                       py::handle uniform_smp,
                       arma::uword n_dir,
                       py::handle random_init,
                       double spatial_freq_min,
                       double spatial_freq_max,
                       arma::sword seed)
{
    const bool uniform_smp_b = qd_python_flag(uniform_smp, true, "uniform_smp");
    const bool random_init_b = qd_python_flag(random_init, true, "random_init");

    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Outputs; the number of passes and the size of test_dir are known only after the call
    arma::mat test_dir;
    arma::vec ase;

    const arma::mat spatial_freq = quadriga_lib::sos_generate<double>(acf_a, dist_a, n_coeff, &ase, &test_dir, n_dim,
                                                                      uniform_smp_b, n_dir, random_init_b,
                                                                      spatial_freq_min, spatial_freq_max, seed);

    auto freq_py = qd_python_copy2numpy(&spatial_freq);
    auto ase_py = qd_python_copy2numpy(&ase);
    auto test_dir_py = qd_python_copy2numpy(&test_dir);

    return py::make_tuple(freq_py, ase_py, test_dir_py);
}

/*!MD
# sos.calc_mse
Approximation error of the ACF of a set of sinusoids

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- `mse_dB`: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`, with `i` counted
  from 0. A direction and its opposite give the same ACF
- `mse_core_dB`: error on the grid of [[sos.acf_approx]] against [[sos.acf_2d]], for the grid points
  within a radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error
  is averaged over the three planes
- `mse_all_dB`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond
  `Dmax`
- For `n_dim = 1`, `mse_core_dB` is evaluated on the sample points `dist`, `mse_all_dB` on the axis from
  `-2·Dmax` to `2·Dmax`, and `mse_dB` equals `mse_core_dB`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`.
  The target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- All outputs are always computed, including the 2-D grids behind `mse_core_dB` and `mse_all_dB`
- A perfect fit has a mean squared error of 0 and is returned as `inf`; this happens for a target
  that the sinusoids reproduce exactly at all sample points
- Equally spaced has the meaning given in [[sos.acf_2d]]

## Usage:
```
mse_dB, mse_core_dB, mse_all_dB, acf_tail, test_dir = quadriga_lib.sos.calc_mse( spatial_freq, acf, \
    dist, *, n_dir )
```

## Inputs:
- **`spatial_freq`** — Spatial frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0, be equally spaced and finite; `(n_acf,)`
- **`n_dir`** — Number of test directions; `int`; must be > 0; ignored for `n_dim = 1`; default: 10

## Outputs:
- **`mse_dB`** — Error along the test directions in [dB]; `float`
- **`mse_core_dB`** — Error within the core radius in [dB]; `float`
- **`mse_all_dB`** — Error on the complete grid in [dB]; `float`
- **`acf_tail`** — Approximated ACF beyond the fitted range; `(4,)`
- **`test_dir`** — Test directions; `(n_dir, 2)` with columns `(azimuth, elevation)` for
  `n_dim = 3`; `(n_dir, 1)` with the azimuth for `n_dim = 2`; empty `(0, 0)` for `n_dim = 1`

## See also:
- [[sos.generate]] (finds the frequencies; minimises the error returned here)
- [[sos.acf_approx]], [[sos.acf_2d]] (the two grids that are compared)
MD!*/

py::tuple sos_calc_mse(const py::array_t<double> &spatial_freq,
                       const py::array_t<double> &acf,
                       const py::array_t<double> &dist,
                       arma::uword n_dir)
{
    const auto freq_a = qd_python_numpy2arma_Mat<double>(spatial_freq, true);
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Output allocation; the size of test_dir depends on the number of dimensions
    double mse_core_dB = 0.0, mse_all_dB = 0.0;
    arma::vec acf_tail;
    arma::mat test_dir;
    auto acf_tail_py = qd_python_init_output((arma::uword)4, &acf_tail);

    double mse_dB = quadriga_lib::sos_calc_mse<double>(freq_a, acf_a, dist_a, &mse_core_dB, &mse_all_dB, &acf_tail, &test_dir, n_dir);

    auto test_dir_py = qd_python_copy2numpy(&test_dir);

    return py::make_tuple(mse_dB, mse_core_dB, mse_all_dB, acf_tail_py, test_dir_py);
}
