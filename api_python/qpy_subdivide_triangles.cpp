// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# subdivide_triangles
Subdivide triangles into smaller triangles

- Uniformly subdivides each input triangle into `n_div x n_div` smaller triangles
- Output count: `n_triangles_out = n_triangles_in x n_div x n_div`
- Material indices are duplicated from the parent triangle to all sub-triangles

## Usage:
```
triangles_out, mtl_ind_out = quadriga_lib.RTtools.subdivide_triangles( n_div, triangles_in, mtl_ind )
```

## Inputs:
- **`n_div`** — Number of subdivisions per edge, > 0
- **`triangles_in`** — Mesh vertices as `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `(n_triangles_in, 9)`
- **`mtl_ind`** — Material indices per triangle (the `csv_ind` output of [[obj_file_read]]), passed through unchanged;
  `(n_triangles_in,)`; default: `None`

## Outputs:
- **`triangles_out`** — Subdivided mesh vertices, same column layout as `triangles_in`; `(n_triangles_out, 9)`
- **`mtl_ind_out`** — Material indices for subdivided triangles; int64; `(n_triangles_out,)`; empty when `mtl_ind` is
  `None`
MD!*/

py::tuple subdivide_triangles(arma::uword n_div,
                              const py::array_t<double> &triangles_in,
                              py::handle mtl_ind)
{
    const auto triangles_in_a = qd_python_numpy2arma_Mat(triangles_in, true);
    const auto mtl_ind_a = qd_python_numpy2arma_Col<arma::uword>(mtl_ind, true);

    // triangles_out is written in numpy memory; for an n_div too large for 64 bits the size wraps and C++ throws
    arma::mat triangles_out;
    arma::uvec mtl_ind_out;
    auto triangles_out_py = qd_python_init_output(triangles_in_a.n_rows * n_div * n_div, (arma::uword)9, &triangles_out);

    quadriga_lib::subdivide_triangles<double>(n_div, &triangles_in_a, &triangles_out, &mtl_ind_a, &mtl_ind_out);

    // Copied to int64, the dtype of the material indices from obj_file_read
    return py::make_tuple(triangles_out_py, qd_python_copy2numpy<arma::uword, py::ssize_t>(&mtl_ind_out));
}
