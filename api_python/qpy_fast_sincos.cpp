// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# fast_sincos
Compute elementwise approximate sine and cosine of a vector

- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for |x| <= 1e10
  (the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; inf and NaN give NaN
- Outputs are single precision: the fast_* functions are meant for very large arrays; double-precision callers use
  `numpy.sin` / `numpy.cos`

## Usage:
```
sin_x, cos_x = quadriga_lib.tools.fast_sincos( x, *, use_kernel )
```

## Inputs:
- **`x`** — Input angles in rad; any shape
- **`use_kernel`** — Kernel: `0` = auto (AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2; throws if
  unavailable; default: `0`

## Outputs:
- **`sin_x`** — sin(x); float32; shape of `x`
- **`cos_x`** — cos(x); float32; shape of `x`
MD!*/

py::tuple fast_sincos(const py::array_t<double> &x,
                      int use_kernel)
{
    const auto x_a = qd_python_numpy2arma_Col(x, true); // Column-major order

    arma::fvec sin_x, cos_x;
    auto sin_x_py = qd_python_init_output(x_a.n_elem, &sin_x);
    auto cos_x_py = qd_python_init_output(x_a.n_elem, &cos_x);

    quadriga_lib::fast_sincos<double>(x_a, &sin_x, &cos_x, use_kernel);

    // Outputs in the shape of x; a column-major reshape of a 1D array is a view
    const auto shape = x.attr("shape");
    return py::make_tuple(sin_x_py.attr("reshape")(shape, py::arg("order") = "F"),
                          cos_x_py.attr("reshape")(shape, py::arg("order") = "F"));
}
