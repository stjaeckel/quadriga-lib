// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_sos.hpp"

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

## Declaration:
```
arma::Col<dtype> quadriga_lib::sos_acfi(
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    const arma::Col<dtype> &d);
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, increase strictly and be finite; `[n_acf]`
- **`d`** — Distances in [m]; must be >= 0; `[n_d]`

## Returns:
- ACF `val` at the distances `d`; `[n_d]`

## See also:
- [[sos_preset]] (ACF tables of the predefined models)
MD!*/

template <typename dtype>
arma::Col<dtype> quadriga_lib::sos_acfi(const arma::Col<dtype> &acf,
                                        const arma::Col<dtype> &dist,
                                        const arma::Col<dtype> &d)
{
    qd_sos_check_acf(acf, dist);
    for (arma::uword i = 0; i < d.n_elem; ++i)
        if (!(d.at(i) >= (dtype)0.0))
            throw std::invalid_argument("Input 'd' must be >= 0.");

    const arma::vec acf_d = arma::conv_to<arma::vec>::from(acf), dist_d = arma::conv_to<arma::vec>::from(dist);

    arma::Col<dtype> val(d.n_elem, arma::fill::none);
    for (arma::uword i = 0; i < d.n_elem; ++i)
        val.at(i) = (dtype)qd_interp_linear(acf_d.memptr(), dist_d.memptr(), acf_d.n_elem, (double)d.at(i));
    return val;
}

template arma::Col<float> quadriga_lib::sos_acfi(const arma::Col<float> &acf, const arma::Col<float> &dist, const arma::Col<float> &d);
template arma::Col<double> quadriga_lib::sos_acfi(const arma::Col<double> &acf, const arma::Col<double> &dist, const arma::Col<double> &d);
