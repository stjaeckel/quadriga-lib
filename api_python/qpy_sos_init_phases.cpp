// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_init_phases
Draw random phases for a set of sum-of-sinusoids generators

- Each column of `phase` is one realisation of the random field of a generator; see [[sos_val]]
- Values are uniform in the interval from -pi (included) to pi (excluded)
- Random stream: 64-bit Mersenne Twister (`std::mt19937_64`) seeded with `seed`; each value is
  `2·pi · ( u - 0.5 )`, where `u` is the upper 53 bits of one generator output divided by 2^53.
  A seed >= 0 gives the same phases on all platforms; the random state of numpy is not used
- The matrix is filled column by column from one stream: for the same seed, column `g` does not
  depend on `n_gen`

## Usage:
```
phase = quadriga_lib.sos.init_phases( n_coeff, n_gen, seed )
```

## Inputs:
- **`n_coeff`** — Number of sinusoids; `int`; must be > 0
- **`n_gen`** — Number of generators; `int`; must be > 0; default: 1
- **`seed`** — Seed of the random stream; `int`; must be >= -1; -1 seeds from the system random
  device and gives different phases on each call; default: -1

## Output:
- **`phase`** — Phases in [rad]; `(n_coeff, n_gen)`

## See also:
- [[sos_val]] (evaluates the generators)
- [[sos_preset]] (frequencies that belong to the phases)
MD!*/

py::array_t<double> sos_init_phases(arma::uword n_coeff,
                                    arma::uword n_gen,
                                    arma::sword seed)
{
    // Output allocation
    arma::mat phase;
    auto phase_py = qd_python_init_output(n_coeff, n_gen, &phase);

    // Call library function
    quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, phase, seed);

    return phase_py;
}

// pybind11 declaration:
// m.def("init_phases", &sos_init_phases,
//       py::arg("n_coeff"),
//       py::arg("n_gen") = 1,
//       py::arg("seed") = -1);
