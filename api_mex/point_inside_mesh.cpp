// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# POINT_INSIDE_MESH
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
result = quadriga_lib.point_inside_mesh( points, mesh, obj_ind, distance );
```

## Inputs:
- **`points`** — 3D coordinates of test points; `[n_points, 3]`
- **`mesh`** — Triangle faces in row-major vertex format `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`obj_ind`** — Object index per mesh element; enables per-object output; `[n_mesh]`
- **`distance`** — Surface proximity threshold; points within this distance of the mesh surface are
  classified as inside; increases ray count to 4 + N_icosphere(⌈distance⌉ + 1); range: 0–20 m; Default: 0

## Output:
- `**result**`— Indicator: `0` = outside, `1` = inside any object (no `obj_ind`), or 1-based object
  index (with `obj_ind`); size `[n_points]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto points_arma = qd_mex_get_Mat<double>(prhs[0]);
    const auto mesh_arma = qd_mex_get_Mat<double>(prhs[1]);
    double distance = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "distance", 0.0);

    // Material/object index: MATLAB 1-based -> C++ 0-based (copy so we don't mutate caller memory)
    arma::uvec obj_ind_arma = (nrhs < 3) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[2], true);
    if (!obj_ind_arma.is_empty())
        obj_ind_arma -= 1;

    const arma::uvec *p_obj_ind = obj_ind_arma.is_empty() ? nullptr : &obj_ind_arma;

    arma::uvec res;
    CALL_QD(res = quadriga_lib::point_inside_mesh(&points_arma, &mesh_arma, p_obj_ind, distance));

    if (nlhs == 1)
        plhs[0] = qd_mex_copy2matlab(&res);
}