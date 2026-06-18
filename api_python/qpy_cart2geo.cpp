// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_math.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# cart2geo
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

- Computes: length = sqrt(x² + y² + z²), az = atan2(y, x), el = asin(clamp(z / length, -1, 1))
- Inputs are arbitrary 3D vectors (not required to be unit length); length returns the Euclidean norm
- z / length is clamped to [-1, 1] before asin to guard against length == 0 and rounding artifacts
- Azimuth is in [-pi, pi], elevation in [-pi/2, pi/2]; elevation pi/2 points to the zenith, 0 to the horizon
- Two mutually exclusive input forms: a combined array `cart` of shape (3, n, m), or separate `x`, `y`, `z` of shape (n, m)
- The AVX2 kernel computes internally in single precision; use `use_kernel = 1` for full double precision

## Usage:
```
geo_coords     = quadriga_lib.tools.cart2geo( cart, y, z, combine, use_kernel )
az, el, length = quadriga_lib.tools.cart2geo( cart, y, z, combine=False )
```

## Inputs:
- **`cart`** — Combined Cartesian coordinates `(3, n, m)`; or, when `y` and `z` are given, the x-coordinates `(n, m)`
- **`y`** — Y-coordinates `(n, m)`; provide together with `z` for separate inputs; default: None
- **`z`** — Z-coordinates `(n, m)`; provide together with `y`; default: None
- **`combine`** — If True, return a single `(3, n, m)` array; if False, return separate az, el, length arrays; default: True
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws if unavailable); default: 1

## Outputs:
- **`geo_coords`** — Combined geographic coordinates `(3, n, m)`; row 0 = azimuth, row 1 = elevation, row 2 = vector length; returned when `combine` is True
- **`az`** — Azimuth angles `(n, m)`; returned when `combine` is False
- **`el`** — Elevation angles `(n, m)`; returned when `combine` is False
- **`length`** — Vector length `(n, m)`; returned when `combine` is False
MD!*/

py::object cart2geo(const py::array_t<double> &cart,
                    py::handle y,
                    py::handle z,
                    bool combine,
                    int use_kernel)
{
    // Read inputs into flat coordinate vectors
    arma::vec cx, cy, cz;
    arma::uword n = 0, m = 0, n_val = 0;

    if (y.is_none() && z.is_none()) // Combined input: cart is (3, n, m)
    {
        const auto cart_a = qd_python_numpy2arma_Cube<double>(cart, true);
        if (cart_a.n_elem == 0 || cart_a.n_rows != 3)
            throw std::invalid_argument("Combined input 'cart' must have shape (3, n, m).");

        n = cart_a.n_cols, m = cart_a.n_slices, n_val = n * m;
        cx.set_size(n_val), cy.set_size(n_val), cz.set_size(n_val);

        const double *pc = cart_a.memptr();
        double *wx = cx.memptr(), *wy = cy.memptr(), *wz = cz.memptr();
        for (arma::uword i = 0; i < n_val; ++i)
            wx[i] = pc[3 * i], wy[i] = pc[3 * i + 1], wz[i] = pc[3 * i + 2];
    }
    else // Separate inputs: cart is x, plus y and z
    {
        auto cart_a = qd_python_numpy2arma_Mat<double>(cart, true);
        auto y_a = qd_python_numpy2arma_Mat<double>(y, true);
        auto z_a = qd_python_numpy2arma_Mat<double>(z, true);

        if (cart_a.n_elem == 0 || y_a.n_elem != cart_a.n_elem || z_a.n_elem != cart_a.n_elem)
            throw std::invalid_argument("Separate inputs 'x', 'y', 'z' must be non-empty and equal in size.");

        n = cart_a.n_rows, m = cart_a.n_cols, n_val = cart_a.n_elem;
        cx = arma::vec(cart_a.memptr(), n_val, false, true);
        cy = arma::vec(y_a.memptr(), n_val, false, true);
        cz = arma::vec(z_a.memptr(), n_val, false, true);
    }

    if (combine) // Single (3, n, m) output
    {
        arma::vec az, el, length;
        quadriga_lib::fast_cart2geo<double>(cx, cy, cz, az, el, &length, use_kernel);

        arma::cube geo_coords;
        auto geo_coords_py = qd_python_init_output(3, n, m, &geo_coords);

        const double *pa = az.memptr(), *pe = el.memptr(), *pl = length.memptr();
        double *pg = geo_coords.memptr();
        for (arma::uword i = 0; i < n_val; ++i)
            pg[3 * i] = pa[i], pg[3 * i + 1] = pe[i], pg[3 * i + 2] = pl[i];

        return geo_coords_py;
    }

    // Separate (n, m) outputs — written in place via strict views (zero-copy)
    arma::mat az, el, length;
    auto az_py = qd_python_init_output(n, m, &az);
    auto el_py = qd_python_init_output(n, m, &el);
    auto length_py = qd_python_init_output(n, m, &length);

    arma::vec az_v(az.memptr(), n_val, false, true);
    arma::vec el_v(el.memptr(), n_val, false, true);
    arma::vec length_v(length.memptr(), n_val, false, true);

    quadriga_lib::fast_cart2geo<double>(cx, cy, cz, az_v, el_v, &length_v, use_kernel);

    return py::make_tuple(az_py, el_py, length_py);
}

// pybind11 declaration:
// m.def("cart2geo", &cart2geo,
//       py::arg("cart"),
//       py::arg("y") = py::none(),
//       py::arg("z") = py::none(),
//       py::arg("combine") = true,
//       py::arg("use_kernel") = 1);
