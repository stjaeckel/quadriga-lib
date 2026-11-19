// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_math.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# fast_cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

- Computes: length = sqrt(x² + y² + z²), az = atan2(y, x), el = asin(clamp(z / length, -1, 1))
- Inputs are arbitrary 3D vectors (not required to be unit length); length returns the Euclidean norm
- z / length is clamped to [-1, 1] before asin to guard against length == 0 and rounding artifacts
- Azimuth is in [-pi, pi], elevation in [-pi/2, pi/2]; elevation pi/2 points to the zenith, 0 to the horizon
- Special values of atan2 as in numpy (e.g. atan2(0, -0) = pi); NaN input gives NaN in az, el and length
- `x`, `y` and `z` have the same shape (up to 3 dimensions); the outputs have that shape; empty input gives empty outputs
- The AVX2 kernel computes the angles in single precision (length and z / length in double precision); use
  `use_kernel = 1` for full double precision

## Usage:
```
az, el, length = quadriga_lib.tools.fast_cart2geo( x, y, z, *, use_kernel )
```

## Inputs:
- **`x`** — X-coordinates; up to 3 dimensions
- **`y`** — Y-coordinates; shape of `x`
- **`z`** — Z-coordinates; shape of `x`
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws if unavailable); other values raise; default: 0

## Outputs:
- **`az`** — Azimuth angles; shape of `x`
- **`el`** — Elevation angles; shape of `x`
- **`length`** — Vector length; shape of `x`
MD!*/

py::tuple fast_cart2geo(const py::array_t<double> &x,
                        const py::array_t<double> &y,
                        const py::array_t<double> &z,
                        int use_kernel)
{
    const auto shape = x.attr("shape");
    if (!shape.equal(y.attr("shape")))
        throw std::invalid_argument("Input 'y' must have the same shape as 'x'.");
    if (!shape.equal(z.attr("shape")))
        throw std::invalid_argument("Input 'z' must have the same shape as 'x'.");

    // Flat views of the inputs in column-major order where the memory layout allows it, else copies
    const auto x_a = qd_python_numpy2arma_Col(x, true);
    const auto y_a = qd_python_numpy2arma_Col(y, true);
    const auto z_a = qd_python_numpy2arma_Col(z, true);

    arma::vec az, el, length;
    auto az_py = qd_python_init_output(x_a.n_elem, &az);
    auto el_py = qd_python_init_output(x_a.n_elem, &el);
    auto length_py = qd_python_init_output(x_a.n_elem, &length);

    quadriga_lib::fast_cart2geo<double>(x_a, y_a, z_a, az, el, &length, use_kernel);

    // Outputs in the shape of x; a column-major reshape of a 1D array is a view
    return py::make_tuple(az_py.attr("reshape")(shape, py::arg("order") = "F"),
                          el_py.attr("reshape")(shape, py::arg("order") = "F"),
                          length_py.attr("reshape")(shape, py::arg("order") = "F"));
}
