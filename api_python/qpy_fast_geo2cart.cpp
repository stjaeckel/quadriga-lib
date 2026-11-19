// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_math.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# fast_geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

- Conversion: x = cos(el)·cos(az)·len, y = cos(el)·sin(az)·len, z = sin(el)·len
- `el` and `len` have the same shape as `az` (up to 3 dimensions); the outputs have that shape; omit `len` for unit
  length; empty input gives empty outputs
- Inverse of `fast_cart2geo`
- The AVX2 kernel computes sine and cosine in single precision (angles reduced in double precision first; max. error
  about 1.3e-7 for angles up to 1e10); use `use_kernel = 1` for full double precision

## Usage:
```
x, y, z = quadriga_lib.tools.fast_geo2cart( az, el, len, *, use_kernel )
```

## Inputs:
- **`az`** — Azimuth angles; up to 3 dimensions
- **`el`** — Elevation angles; shape of `az`
- **`len`** — Vector length; shape of `az`; default: None (unit length)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws
  if unavailable); other values raise; default: 0

## Outputs:
- **`x`** — X-coordinates; shape of `az`
- **`y`** — Y-coordinates; shape of `az`
- **`z`** — Z-coordinates; shape of `az`
MD!*/

py::tuple fast_geo2cart(const py::array_t<double> &az,
                        const py::array_t<double> &el,
                        py::handle len,
                        int use_kernel)
{
    const auto shape = az.attr("shape");
    if (!shape.equal(el.attr("shape")))
        throw std::invalid_argument("Input 'el' must have the same shape as 'az'.");
    const bool have_len = !len.is_none();
    const auto len_np = have_len ? py::array_t<double>::ensure(len) : py::array_t<double>();
    if (have_len && (!len_np || !shape.equal(len_np.attr("shape"))))
        throw std::invalid_argument("Input 'len' must have the same shape as 'az'.");

    // Flat views of the inputs in column-major order where the memory layout allows it, else copies
    const auto az_a = qd_python_numpy2arma_Col(az, true);
    const auto el_a = qd_python_numpy2arma_Col(el, true);
    const auto len_a = have_len ? qd_python_numpy2arma_Col(len_np, true) : arma::vec();

    arma::vec x, y, z;
    auto x_py = qd_python_init_output(az_a.n_elem, &x);
    auto y_py = qd_python_init_output(az_a.n_elem, &y);
    auto z_py = qd_python_init_output(az_a.n_elem, &z);

    quadriga_lib::fast_geo2cart<double>(az_a, el_a, x, y, z, nullptr, nullptr, nullptr, nullptr,
                                        have_len ? &len_a : nullptr, use_kernel);

    // Outputs in the shape of az; a column-major reshape of a 1D array is a view
    return py::make_tuple(x_py.attr("reshape")(shape, py::arg("order") = "F"),
                          y_py.attr("reshape")(shape, py::arg("order") = "F"),
                          z_py.attr("reshape")(shape, py::arg("order") = "F"));
}
