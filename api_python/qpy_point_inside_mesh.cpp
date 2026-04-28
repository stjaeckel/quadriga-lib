// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

- Always casts 4 rays per point in near-tetrahedral directions (rotated regular tetrahedron,
  scaled to 1000 m) for inside/outside detection
- When `distance > 0`, adds icosphere-sampled rays at subdivision level ⌈distance⌉ + 1
  (e.g. subdiv 2 for distance ≤ 1 m, subdiv 3 for ≤ 2 m), substantially increasing ray count
- A point is inside if any ray hits a face with a negative incidence angle, or if the ray
  thickness at FBS is below 1 mm (surface proximity)
- Mesh must be watertight with all normals pointing outward
- If `obj_ind` is provided, returns the 1-based enclosing object index instead of binary 0/1

## Usage:
```
result = quadriga_lib.RTtools.point_inside_mesh( points, mesh, obj_ind, distance )
```

## Input Arguments:
- **`points`** — 3D coordinates of test points; `(n_points, 3)`
- **`mesh`** — Triangle faces in row-major vertex format `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `(n_mesh, 9)`
- **`obj_ind`** *(optional)* — 1-based object index per mesh element; enables per-object output; `(n_mesh,)`
- **`distance`** *(optional)* — Surface proximity threshold; points within this distance
  of the mesh surface are classified as inside; increases ray count to 4 + N_icosphere(⌈distance⌉ + 1);
  range: 0–20 m; Default: 0

## Output Arguments:
- `**result**`— Indicator: `0` = outside, `1` = inside any object (no `obj_ind`), or 1-based object
  index (with `obj_ind`); uint32; size `(n_points,)`

See also:
- [[obj_file_read]] (for reading `mesh` and `obj_ind` from an .obj file)
MD!*/

py::array_t<py::ssize_t> point_inside_mesh(const py::array_t<double> &points,
                                           const py::array_t<double> &mesh,
                                           const py::array_t<unsigned> &obj_ind,
                                           double distance)
{
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);
    const auto mesh_arma = qd_python_numpy2arma_Mat(mesh, true);
    const auto obj_ind_arma = qd_python_numpy2arma_Col(obj_ind, true);

    auto res = quadriga_lib::point_inside_mesh(&points_arma, &mesh_arma, &obj_ind_arma, distance);

    return qd_python_copy2numpy(res);
}