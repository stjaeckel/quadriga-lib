// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <cmath>
#include <stdexcept>

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_acfi
Target ACF at given distances

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table
- The same rule is applied by [[sos_val]] and [[sos_remove_bias]] for two devices on the same map
- An empty `d` returns an empty `val`
- `val` may be the same object as `acf`, `dist` or `d`

## Declaration:
```
void quadriga_lib::sos_acfi(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    const arma::Col<dtype> &d,
    arma::Col<dtype> &val);
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, increase strictly and be finite; `[n_acf]`
- **`d`** — Distances in [m]; must be >= 0; `[n_d]`

## Output:
- **`val`** — ACF at the distances `d`; `[n_d]`

## See also:
- [[sos_preset]] (ACF tables of the predefined models)
MD!*/

template <typename dtype>
void quadriga_lib::sos_acfi(const arma::Col<dtype> &acf,
                            const arma::Col<dtype> &dist,
                            const arma::Col<dtype> &d,
                            arma::Col<dtype> &val)
{
    qd_sos_check_acf(acf, dist);
    for (arma::uword i = 0; i < d.n_elem; ++i)
        if (!(d.at(i) >= (dtype)0.0))
            throw std::invalid_argument("Input 'd' must be >= 0.");

    // The table is copied before the output is resized, so that 'val' may be the same object as an input
    const arma::vec acf_d = arma::conv_to<arma::vec>::from(acf), dist_d = arma::conv_to<arma::vec>::from(dist);

    if (val.n_elem != d.n_elem)
        val.set_size(d.n_elem);

    for (arma::uword i = 0; i < d.n_elem; ++i)
        val.at(i) = (dtype)qd_interp_linear(acf_d.memptr(), dist_d.memptr(), acf_d.n_elem, (double)d.at(i));
}

template void quadriga_lib::sos_acfi(const arma::Col<float> &acf, const arma::Col<float> &dist, const arma::Col<float> &d, arma::Col<float> &val);
template void quadriga_lib::sos_acfi(const arma::Col<double> &acf, const arma::Col<double> &dist, const arma::Col<double> &d, arma::Col<double> &val);
