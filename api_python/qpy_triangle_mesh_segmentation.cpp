// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more
  than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with
  zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; `mesh_index` uses 0 to mark
  padding entries
- If `mtl_ind` is provided, material indices are reordered and padded in the same way; padding
  entries get index 0

## Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.triangle_mesh_segmentation( triangles, target_size, vec_size, mtl_ind )

# Unpacked outputs
triangles_out, sub_mesh_index, mesh_index, mtl_ind_out = \
    quadriga_lib.RTtools.triangle_mesh_segmentation( triangles, target_size, vec_size, mtl_ind )
```

## Inputs:
- **`triangles`** — Triangle vertices, each row `[x1,y1,z1,x2,y2,z2,x3,y3,z3]`; `(n_mesh, 9)`
- **`target_size`** — Target triangle count per sub-mesh; for best performance set near sqrt(n_mesh); default: 1024
- **`vec_size`** — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each sub-mesh row count
  is rounded up to a multiple of this value; default: 1
- **`mtl_ind`** — 0-based material index per face (the `csv_ind` output of [[obj_file_read]]);
  `(n_mesh,)` or `None`; default: `None`

## Outputs:
- **`triangles_out`** — Reordered and padded triangle vertices; `(n_triangles_out, 9)`
- **`sub_mesh_index`** — 0-based start indices of sub-meshes in `triangles_out`; uint32; `(n_sub,)`
- **`mesh_index`** — 0-based mapping from original to reorganized mesh (0 = padding); uint32; `(n_triangles_out,)`
- **`mtl_ind_out`** — Reordered and padded material indices; `(n_triangles_out,)`; empty if `mtl_ind` is not provided
MD!*/

py::tuple triangle_mesh_segmentation(const py::array_t<double> &triangles,
                                     arma::uword target_size,
                                     arma::uword vec_size,
                                     const py::handle &mtl_ind)
{
    const auto triangles_a = qd_python_numpy2arma_Mat<double>(triangles, true);
    const auto mtl_ind_a = qd_python_numpy2arma_Col<arma::uword>(mtl_ind, true);

    arma::mat triangles_out;
    arma::uvec mtl_ind_out;
    arma::u32_vec sub_mesh_index, mesh_index;

    const arma::uvec *p_mtl_ind = mtl_ind_a.is_empty() ? nullptr : &mtl_ind_a;
    arma::uvec *p_mtl_ind_out = mtl_ind_a.is_empty() ? nullptr : &mtl_ind_out;

    quadriga_lib::triangle_mesh_segmentation<double>(&triangles_a, &triangles_out, &sub_mesh_index,
                                                     target_size, vec_size,
                                                     p_mtl_ind, p_mtl_ind_out, &mesh_index);

    return py::make_tuple(
        qd_python_copy2numpy(&triangles_out),
        qd_python_copy2numpy<unsigned, py::ssize_t>(&sub_mesh_index),
        qd_python_copy2numpy<unsigned, py::ssize_t>(&mesh_index),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&mtl_ind_out));
}

// pybind11 declaration:
// m.def("triangle_mesh_segmentation", &triangle_mesh_segmentation,
//       py::arg("triangles"),
//       py::arg("target_size") = 1024,
//       py::arg("vec_size") = 1,
//       py::arg("mtl_ind") = py::none());