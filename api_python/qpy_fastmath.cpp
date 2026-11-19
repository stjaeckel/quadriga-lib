// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python.hpp"
#include "quadriga_lib.hpp"

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
                                        len_a, use_kernel);

    // Outputs in the shape of az; a column-major reshape of a 1D array is a view
    return py::make_tuple(x_py.attr("reshape")(shape, py::arg("order") = "F"),
                          y_py.attr("reshape")(shape, py::arg("order") = "F"),
                          z_py.attr("reshape")(shape, py::arg("order") = "F"));
}

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
