// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_acfi
Target ACF at given distances

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table
- The same rule is applied by [[sos_val]] and [[sos_remove_bias]] for two devices on the same map

## Usage:
```
val = quadriga_lib.sos.acfi( acf, dist, d )
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `(n_acf,)`
- **`dist`** — Sample points of the ACF; must start at 0 and increase strictly; `(n_acf,)`
- **`d`** — Distances; must be >= 0; `(n_d,)`

## Output:
- **`val`** — ACF at the distances `d`; `(n_d,)`

## See also:
- [[sos_preset]] (ACF tables of the predefined models)
MD!*/

py::array_t<double> sos_acfi(const py::array_t<double> &acf,
                             const py::array_t<double> &dist,
                             const py::array_t<double> &d)
{
    // Read inputs
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);
    const auto d_a = qd_python_numpy2arma_Col<double>(d, true);

    // Output allocation
    arma::vec val;
    auto val_py = qd_python_init_output(d_a.n_elem, &val);

    // Call library function
    quadriga_lib::sos_acfi<double>(acf_a, dist_a, d_a, val);

    return val_py;
}

// pybind11 declaration:
// m.def("acfi", &sos_acfi,
//       py::arg("acf"),
//       py::arg("dist"),
//       py::arg("d"));
