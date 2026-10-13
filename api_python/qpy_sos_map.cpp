// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos.map
Spatially correlated random values on a regular grid

- Evaluates the generators of [[sos.val]] for one device at all combinations of the grid vectors
- Output in meshgrid order: element `[iy, ix, iz, g]` belongs to the position
  `(x[ix], y[iy], z[iz])` and to generator `g`, i.e. column `g` of `phase`
- The number of grid vectors given must not exceed the number of columns of `freq`; a `y` or `z`
  that is `None` or empty means a single sample at 0 on that axis
- `z` can be given without `y` if `freq` has 3 columns
- The values are those of [[sos.val]] at the grid points, bit for bit; kernels, accuracy and the
  Uniform mapping are described there

## Usage:
```
map = quadriga_lib.sos.map( freq, phase, x, y, z, uniform, use_kernel )
```

## Inputs:
- **`freq`** — Frequencies; `n_dim` = 1, 2 or 3; `(n_coeff, n_dim)`
- **`phase`** — Phases; one column per generator; `(n_coeff, n_gen)`
- **`x`** — Grid points on the x-axis; `(nx,)`
- **`y`** — Grid points on the y-axis; requires `n_dim >= 2`; `None` = one sample at 0, `ny = 1`;
  `(ny,)`; default: `None`
- **`z`** — Grid points on the z-axis; requires `n_dim = 3`; `None` = one sample at 0, `nz = 1`;
  `(nz,)`; default: `None`
- **`uniform`** — Output distribution: `False` = Normal(0,1), `True` = Uniform between 0 and 1;
  default: `False`
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2 (throws if
  unavailable); `int`; default: 0

## Output:
- **`map`** — Values; `(ny, nx, nz, n_gen)`

## See also:
- [[sos.val]] (evaluation at arbitrary positions, two devices)
MD!*/

py::array_t<double> sos_map(const py::array_t<double> &freq,
                            const py::array_t<double> &phase,
                            const py::array_t<double> &x,
                            py::handle y,
                            py::handle z,
                            bool uniform,
                            int use_kernel)
{
    // Read inputs
    const auto freq_a = qd_python_numpy2arma_Mat<double>(freq, true);
    const auto phase_a = qd_python_numpy2arma_Mat<double>(phase, true);
    const auto x_a = qd_python_numpy2arma_Col<double>(x, true);
    const auto y_a = qd_python_numpy2arma_Col<double>(y, true);
    const auto z_a = qd_python_numpy2arma_Col<double>(z, true);

    const arma::uword nx = x_a.n_elem, ny = y_a.is_empty() ? 1 : y_a.n_elem;
    const arma::uword nz = z_a.is_empty() ? 1 : z_a.n_elem, n_gen = phase_a.n_cols;

    // Output allocation
    std::vector<arma::cube> map;
    auto map_py = qd_python_init_output(ny, nx, nz, n_gen, &map);

    // Wrap optional pointers
    const arma::vec *p_y = y_a.is_empty() ? nullptr : &y_a;
    const arma::vec *p_z = z_a.is_empty() ? nullptr : &z_a;

    // Call library function
    quadriga_lib::sos_map<double>(freq_a, phase_a, x_a, map, p_y, p_z, uniform, use_kernel);

    return map_py;
}

// pybind11 declaration:
// m.def("map", &sos_map,
//       py::arg("freq").none(false),
//       py::arg("phase").none(false),
//       py::arg("x").none(false),
//       py::arg("y") = py::none(),
//       py::arg("z") = py::none(),
//       py::arg("uniform") = false,
//       py::arg("use_kernel") = 0);
