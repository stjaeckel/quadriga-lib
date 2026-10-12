// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_generate
Find the frequencies of a sum-of-sinusoids generator for a given ACF

- Searches `n_coeff` frequencies such that the autocorrelation function (ACF) of the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, approximates the target ACF in all directions
- Each sinusoid has a fixed direction and a signed frequency along it. `n_dim = 1`: no direction;
  `n_dim = 2`: directions on the half circle; `n_dim = 3`: directions on the upper hemisphere.
  `uniform_smp = True` places them regularly (equally spaced azimuth, spiral lattice, as the test
  directions of [[sos_calc_mse]]), `False` draws them at random
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
  (see `acf_tail` of [[sos_calc_mse]]). To prevent this, set `freq_min`
- `freq_min` and `freq_max` limit the magnitude of the frequency along the direction of each
  sinusoid; 0 selects the default: no lower limit, and `15 / max(dist)` as upper limit
- Use enough test directions: with few (e.g. 10) the result fits those directions and is worse
  in others; the predefined tables of [[sos_preset]] were made with 300
- A seed >= 0 makes the result repeatable on one platform. The search takes accept/reject
  decisions, so a difference in the last bit of the input or of the math library can lead to
  another table of equivalent quality

## Usage:
```
freq, ase, test_dir = quadriga_lib.sos.generate( acf, dist, n_coeff, n_dim, uniform_smp, n_dir, \
    random_init, freq_min, freq_max, seed )
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0 and increase strictly; `(n_acf,)`
- **`n_coeff`** — Number of sinusoids; `int`; must be > 0
- **`n_dim`** — Number of dimensions; `int`; 1, 2 or 3; default: 3
- **`uniform_smp`** — Regular (`True`) or random (`False`) directions of the sinusoids;
  default: `True`
- **`n_dir`** — Number of test directions; `int`; must be > 0; ignored for `n_dim = 1`; default: 10
- **`random_init`** — Start from random frequencies (`True`) or build up sinusoid by sinusoid
  (`False`); default: `True`
- **`freq_min`** — Lower limit of the frequency magnitude in [1/m]; must be >= 0; 0 = none;
  default: 0.0
- **`freq_max`** — Upper limit of the frequency magnitude in [1/m]; must be >= 0 and, if not 0,
  larger than `freq_min`; 0 = `15 / max(dist)`; default: 0.0
- **`seed`** — Seed; `int`; must be >= -1; -1 seeds from the system random device; default: -1

## Outputs:
- **`freq`** — Frequencies in [1/m]; `(n_coeff, n_dim)`
- **`ase`** — Mean squared error over the test directions after each pass, linear scale;
  `(n_pass,)`
- **`test_dir`** — Test directions in [rad]; `(n_dir, 2)` with columns `(azimuth, elevation)` for
  `n_dim = 3`; `(n_dir, 1)` with the azimuth for `n_dim = 2`; empty `(0, 0)` for `n_dim = 1`

## See also:
- [[sos_calc_mse]] (error of a frequency table)
- [[sos_preset]] (precomputed tables)
MD!*/

py::tuple sos_generate(const py::array_t<double> &acf,
                       const py::array_t<double> &dist,
                       arma::uword n_coeff,
                       arma::uword n_dim,
                       bool uniform_smp,
                       arma::uword n_dir,
                       bool random_init,
                       double freq_min,
                       double freq_max,
                       arma::sword seed)
{
    // Read inputs
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Outputs; the number of passes and the size of test_dir are known only after the call
    arma::mat freq, test_dir;
    arma::vec ase;

    // Call library function
    quadriga_lib::sos_generate<double>(acf_a, dist_a, n_coeff, freq, &ase, &test_dir, n_dim, uniform_smp,
                                       n_dir, random_init, freq_min, freq_max, seed);

    // Copy to python
    auto freq_py = qd_python_copy2numpy(&freq);
    auto ase_py = qd_python_copy2numpy(&ase);
    auto test_dir_py = qd_python_copy2numpy(&test_dir);

    return py::make_tuple(freq_py, ase_py, test_dir_py);
}

// pybind11 declaration:
// m.def("generate", &sos_generate,
//       py::arg("acf"),
//       py::arg("dist"),
//       py::arg("n_coeff"),
//       py::arg("n_dim") = 3,
//       py::arg("uniform_smp") = true,
//       py::arg("n_dir") = 10,
//       py::arg("random_init") = true,
//       py::arg("freq_min") = 0.0,
//       py::arg("freq_max") = 0.0,
//       py::arg("seed") = -1);
