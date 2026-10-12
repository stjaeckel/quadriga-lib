// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <random>
#include <stdexcept>

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_init_phases
Draw random phases for a set of sum-of-sinusoids generators

- Each column of `phase` is one realisation of the random field of a generator; see [[sos_val]]
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

## Declaration:
```
void quadriga_lib::sos_init_phases(
    arma::uword n_coeff,
    arma::uword n_gen,
    arma::Mat<dtype> &phase,
    arma::sword seed = -1);
```

## Inputs:
- **`n_coeff`** — Number of sinusoids; must be > 0
- **`n_gen`** — Number of generators; must be > 0
- **`seed`** *(optional)* — Seed of the random stream; must be >= -1; `-1` seeds from the system random
  device and gives different phases on each call

## Output:
- **`phase`** — Phases in [rad]; `[n_coeff, n_gen]`

## See also:
- [[sos_val]] (evaluates the generators)
- [[sos_preset]] (frequencies that belong to the phases)
MD!*/

template <typename dtype>
void quadriga_lib::sos_init_phases(arma::uword n_coeff,
                                   arma::uword n_gen,
                                   arma::Mat<dtype> &phase,
                                   arma::sword seed)
{
    if (n_coeff == 0)
        throw std::invalid_argument("Input 'n_coeff' cannot be 0.");
    if (n_gen == 0)
        throw std::invalid_argument("Input 'n_gen' cannot be 0.");
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");

    if (phase.n_rows != n_coeff || phase.n_cols != n_gen)
        phase.set_size(n_coeff, n_gen);

    const double two_pi = 6.283185307179586;
    const dtype pi = (dtype)3.141592653589793;

    std::mt19937_64 rng = qd_sos_rng(seed);
    dtype *p = phase.memptr();
    for (arma::uword i = 0; i < n_coeff * n_gen; ++i)
    {
        dtype v = (dtype)(two_pi * (qd_sos_uniform(rng) - 0.5));
        p[i] = (v >= pi) ? -pi : v; // A double just below pi can round up to pi in single precision
    }
}

template void quadriga_lib::sos_init_phases(arma::uword n_coeff, arma::uword n_gen, arma::Mat<float> &phase, arma::sword seed);
template void quadriga_lib::sos_init_phases(arma::uword n_coeff, arma::uword n_gen, arma::Mat<double> &phase, arma::sword seed);
