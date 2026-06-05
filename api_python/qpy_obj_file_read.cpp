// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# obj_file_read
Read a Wavefront `.obj` file and extract geometry, visual materials, and EM/acoustic materials

- Parses a triangulated `.obj`; quads and n-gons are rejected
- Parses a triangulated `.obj`; quads and n-gons are rejected. Two independent material systems are returned:
  - Visual side, from the companion `.mtl`: `mtl_ind`, `mtl_names` (raw `usemtl` names), and `bsdf`.
  - EM/acoustic side, from a material table (`fn_csv`, or a built-in ITU-R P.2040 default): `csv_ind`,`csv_names`, `csv_prop`.
- A face's `usemtl` name is matched to the table by exact name, then by the base name (everything
  before the first dot, so Blender sub-materials like `concrete.gray` map to `concrete`)
- Unmatched names raise an error when `csv_strict = True`; otherwise they map to row 0 of the table
  (the transparent fallback)
- With an empty `fn`, geometry and `.mtl` outputs are empty and only the table (`csv_names`,
  `csv_prop`) is populated; if `fn_csv` is also empty, the built-in default table is returned
- For a detailed description of the material model see
  <a href="http://quadriga-lib.org/formats.html">Data Formats</a> section

## Usage:
```
mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, csv_ind, csv_names, csv_prop = \
    quadriga_lib.RTtools.obj_file_read( fn, fn_csv, csv_strict )
```

## Inputs:
- **`fn`** — Path to the `.obj` file; empty loads only the material table
- **`fn_csv`** — Path to an EM/acoustic material CSV; must contain a `name` column, and row 0 is the
  fallback material (should be transparent, e.g. air); empty uses the built-in default table;
  default: `""`
- **`csv_strict`** — If `True`, raise when a `usemtl` material is absent from the table; otherwise
  map to row 0; default: `False`

## Outputs:
- **`mesh`** — Triangle vertex coordinates `[X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3]` per row; `(n_mesh, 9)`
- **`vert_list`** — All vertex positions in the file; `(n_vert, 3)`
- **`face_ind`** — 0-based vertex indices into `vert_list` per triangle; `(n_mesh, 3)`
- **`obj_ind`** — 0-based object index per triangle; `(n_mesh,)`
- **`obj_names`** — Object names; list of `str`; length `max(obj_ind) + 1`
- **`mtl_ind`** — 0-based visual-material index per triangle; `(n_mesh,)`
- **`mtl_names`** — Visual material names (raw `usemtl`); list of `str`; length `no_mtl`
- **`bsdf`** — Principled BSDF values from the `.mtl`; `(no_mtl, 17)`
- **`csv_ind`** — 0-based EM/acoustic-material index per triangle; `(n_mesh,)`
- **`csv_names`** — Material names from the table; list of `str`; length `n_csv`
- **`csv_prop`** — Material properties as a `dict`; each key is one CSV column (excluding `name`)
  mapping to a 1D array of length `n_csv`

## See also:
- [[obj_file_write]] (for writing OBJ files)
- [[triangle_mesh_segmentation]] (used to calculate indexed mesh for faster processing)
- [[ray_mesh_interact]] (calculating interactions between rays and the triangular mesh)
MD!*/

py::tuple obj_file_read(const std::string &fn,
                        const std::string &fn_csv,
                        bool csv_strict)
{
    arma::mat mesh, vert_list, bsdf;
    arma::umat face_ind;
    arma::uvec obj_ind, mtl_ind, csv_ind;
    std::vector<std::string> obj_names, mtl_names, csv_names;
    std::unordered_map<std::string, std::vector<double>> csv_prop;

    quadriga_lib::obj_file_read<double>(fn, &mesh, &vert_list, &face_ind, &obj_ind, &obj_names,
                                        &mtl_ind, &mtl_names, &bsdf,
                                        fn_csv, &csv_ind, &csv_names, &csv_prop, csv_strict);

    // Stable dict key order: standard EM columns first, then any extra CSV columns
    std::vector<std::string> order = {"a", "b", "c", "d", "e", "f", "g", "h",
                                      "att", "attB", "alpha", "alphaB", "fRef", "m",
                                      "resF", "resQ", "resS", "coiF", "coiQ", "coiA", "tf", "tfB"};

    return py::make_tuple(
        qd_python_copy2numpy(&mesh),
        qd_python_copy2numpy(&vert_list),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&face_ind),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&obj_ind),
        qd_python_copy2list(obj_names),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&mtl_ind),
        qd_python_copy2list(mtl_names),
        qd_python_copy2numpy(&bsdf),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&csv_ind),
        qd_python_copy2list(csv_names),
        qd_python_map2dict(csv_prop, order));
}

// pybind11 declaration:
// m.def("obj_file_read", &obj_file_read,
//       py::arg("fn"),
//       py::arg("fn_csv") = std::string(""),
//       py::arg("csv_strict") = false);