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

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; `mesh_index` uses 0 to mark padding entries
- If `mtl_prop` is provided, material rows are reordered and padded in the same way

## Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.triangle_mesh_segmentation( triangles, target_size, vec_size, mtl_prop )

# Unpacked outputs
triangles_out, sub_mesh_index, mesh_index, mtl_propR = 
    quadriga_lib.RTtools.triangle_mesh_segmentation( triangles, target_size, vec_size, mtl_prop )
```

## Inputs:
- **`triangles`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `(n_mesh, 9)`
- **`target_size`** *(optional)* — Target triangle count per sub-mesh; for best performance set
  near sqrt(n_mesh); default: 1024
- **`vec_size`** *(optional)* — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each
  sub-mesh row count is rounded up to a multiple of this value; default: 1
- **`mtl_prop`** *(optional)* — Material properties; see [[obj_file_read]]; `(n_mesh, 9)`

## Outputs:
- **`triangles_out`** — Reordered and padded triangle vertices; `(n_triangles_out, 9)`
- **`sub_mesh_index`** — 0-based start indices of sub-meshes in `triangles_out`; uint32; `(n_sub,)`
- **`mesh_index`** — 1-based mapping from original to reorganized mesh (0 = padding); uint32; `(n_triangles_out, )`
- **`mtl_prop_out`** — Reordered and padded material properties; `(n_triangles_out, 9)`
MD!*/

py::tuple triangle_mesh_segmentation(const py::array_t<double> &triangles, // Faces of the triangular mesh (input), Size: [ n_mesh, 9 ]
                                     arma::uword target_size,              // Target value for the sub-mesh size
                                     arma::uword vec_size,                 // Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA)
                                     const py::array_t<double> &mtl_prop)  // Material properties (input), Size: [ n_mesh, 5 ], optional
{
    const auto triangles_arma = qd_python_numpy2arma_Mat(triangles, true);
    const auto mtl_prop_arma = qd_python_numpy2arma_Mat(mtl_prop, true);

    arma::mat triangles_out_arma, mtl_prop_out_arma;
    arma::u32_vec sub_mesh_index, mesh_index;

    quadriga_lib::triangle_mesh_segmentation(&triangles_arma, &triangles_out_arma,
                                             &sub_mesh_index, target_size, vec_size,
                                             &mtl_prop_arma, &mtl_prop_out_arma, &mesh_index);

    auto triangles_p = qd_python_copy2numpy(&triangles_out_arma);
    auto sub_mesh_index_p = qd_python_copy2numpy(&sub_mesh_index);
    auto mesh_index_p = qd_python_copy2numpy(&mesh_index);
    auto mtl_prop_p = qd_python_copy2numpy(&mtl_prop_out_arma);

    return py::make_tuple(triangles_p, sub_mesh_index_p, mesh_index_p, mtl_prop_p);
}