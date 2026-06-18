// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_math.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# geo2cart
Convert elementwise azimuth/elevation angles to Cartesian coordinates

- Conversion: x = cos(el)·cos(az)·len, y = cos(el)·sin(az)·len, z = sin(el)·len
- Two mutually exclusive input forms: a combined array `az` holding geographic coordinates
  (3, n, m), or separate `az`, `el`, `len` of shape (n, m)
- For combined input, row 0 = azimuth, row 1 = elevation, row 2 = length
- `len` is optional in separate mode; omit it for unit length
- Inverse of `cart2geo`; round-trips in both the combined and separate forms
- The AVX2 kernel computes internally in single precision; use `use_kernel = 1` for full double precision

## Usage:
```
cart    = quadriga_lib.tools.geo2cart( az, el, len, combine, use_kernel )
x, y, z = quadriga_lib.tools.geo2cart( az, el, len, combine=False )
```

## Inputs:
- **`az`** — Azimuth angles `(n, m)`; or, when `el` is omitted, combined geographic coordinates
  `(3, n, m)` (row 0 = azimuth, row 1 = elevation, row 2 = length)
- **`el`** — Elevation angles `(n, m)`; omit for combined input; default: None
- **`len`** — Vector length `(n, m)`; omit for unit length; ignored for combined input; default: None
- **`combine`** — If True, return a single `(3, n, m)` array; if False, return separate x, y, z  arrays; default: True
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (throws
  if unavailable); default: 1

## Outputs:
- **`cart`** — Combined Cartesian coordinates `(3, n, m)`; row 0 = x, row 1 = y, row 2 = z; returned when `combine` is True
- **`x`** — X-coordinates `(n, m)`; returned when `combine` is False
- **`y`** — Y-coordinates `(n, m)`; returned when `combine` is False
- **`z`** — Z-coordinates `(n, m)`; returned when `combine` is False
MD!*/

py::object geo2cart(const py::array_t<double> &az,
                    py::handle el,
                    py::handle len,
                    bool combine,
                    int use_kernel)
{
    // Read inputs into flat angle/length vectors
    arma::vec az_v, el_v, len_v;
    arma::uword n = 0, m = 0, n_val = 0;
    bool have_len = false;

    if (el.is_none()) // Combined input: az is geo_coords (3, n, m)
    {
        const auto geo_a = qd_python_numpy2arma_Cube<double>(az, true);
        if (geo_a.n_elem == 0 || geo_a.n_rows != 3)
            throw std::invalid_argument("Combined input 'az' (geo_coords) must have shape (3, n, m).");

        n = geo_a.n_cols, m = geo_a.n_slices, n_val = n * m;
        az_v.set_size(n_val), el_v.set_size(n_val), len_v.set_size(n_val);

        const double *pg = geo_a.memptr();
        double *wa = az_v.memptr(), *we = el_v.memptr(), *wl = len_v.memptr();
        for (arma::uword i = 0; i < n_val; ++i)
            wa[i] = pg[3 * i], we[i] = pg[3 * i + 1], wl[i] = pg[3 * i + 2];

        have_len = true;
    }
    else // Separate inputs: az, el, and optional len
    {
        auto az_a = qd_python_numpy2arma_Mat<double>(az, true);
        auto el_a = qd_python_numpy2arma_Mat<double>(el, true);
        auto len_a = qd_python_numpy2arma_Mat<double>(len, true); // None -> empty

        if (az_a.n_elem == 0 || el_a.n_rows != az_a.n_rows || el_a.n_cols != az_a.n_cols)
            throw std::invalid_argument("Separate inputs 'az' and 'el' must be non-empty and have the same shape.");
        if (!len_a.empty() && (len_a.n_rows != az_a.n_rows || len_a.n_cols != az_a.n_cols))
            throw std::invalid_argument("'len' must have the same shape as 'az'.");

        n = az_a.n_rows, m = az_a.n_cols, n_val = az_a.n_elem;
        az_v = arma::vec(az_a.memptr(), n_val, false, true);
        el_v = arma::vec(el_a.memptr(), n_val, false, true);
        if (!len_a.empty())
        {
            len_v = arma::vec(len_a.memptr(), n_val, false, true);
            have_len = true;
        }
    }

    const arma::vec *p_len = have_len ? &len_v : nullptr;

    if (combine) // Single (3, n, m) output
    {
        arma::vec x, y, z;
        quadriga_lib::fast_geo2cart<double>(az_v, el_v, x, y, z,
                                            nullptr, nullptr, nullptr, nullptr,
                                            p_len, use_kernel);

        arma::cube cart;
        auto cart_py = qd_python_init_output(3, n, m, &cart);

        const double *px = x.memptr(), *py_ = y.memptr(), *pz = z.memptr();
        double *pc = cart.memptr();
        for (arma::uword i = 0; i < n_val; ++i)
            pc[3 * i] = px[i], pc[3 * i + 1] = py_[i], pc[3 * i + 2] = pz[i];

        return cart_py;
    }

    // Separate (n, m) outputs — written in place via strict views (zero-copy)
    arma::mat x, y, z;
    auto x_py = qd_python_init_output(n, m, &x);
    auto y_py = qd_python_init_output(n, m, &y);
    auto z_py = qd_python_init_output(n, m, &z);

    arma::vec x_v(x.memptr(), n_val, false, true);
    arma::vec y_v(y.memptr(), n_val, false, true);
    arma::vec z_v(z.memptr(), n_val, false, true);

    quadriga_lib::fast_geo2cart<double>(az_v, el_v, x_v, y_v, z_v,
                                        nullptr, nullptr, nullptr, nullptr,
                                        p_len, use_kernel);

    return py::make_tuple(x_py, y_py, z_py);
}

// pybind11 declaration:
// m.def("geo2cart", &geo2cart,
//       py::arg("az"),
//       py::arg("el") = py::none(),
//       py::arg("len") = py::none(),
//       py::arg("combine") = true,
//       py::arg("use_kernel") = 1);
