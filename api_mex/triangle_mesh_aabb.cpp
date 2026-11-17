// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# TRIANGLE_MESH_AABB
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding
  non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

## Usage:
```
aabb = quadriga_lib.triangle_mesh_aabb( mesh, sub_mesh_ind, vec_size );
```

## Inputs:
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; further columns are ignored; `[n_triangles, 9]`
- **`sub_mesh_ind`** — Start indices of sub-meshes in `mesh`; use [[triangle_mesh_segmentation]] to generate; 
  `[n_sub]` or empty; default: `[]` (returns AABB of the entire mesh)
- **`vec_size`** — Alignment size for SIMD/CUDA padding (e.g., 8 for AVX2); default: 1

## Output:
- **`aabb`** — Axis-aligned bounding boxes, one row per sub-mesh; `[n_sub_aligned, 6]`

## See also:
- [[triangle_mesh_segmentation]] (for calculating sub-meshes)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const arma::mat mesh = qd_mex_get_Mat<double>(prhs[0]);
    arma::u32_vec sub_mesh_ind = (nrhs < 2) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[1], true, "sub_mesh_ind");
    const arma::uword vec_size = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "vec_size", 1);

    // Convert sub_mesh_ind to 0-based
    for (unsigned &val : sub_mesh_ind)
    {
        if (val == 0)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Values in 'sub_mesh_ind' cannot be 0.");
        --val;
    }

    // Wrap optional input pointer
    const arma::u32_vec *p_sub_mesh_index = sub_mesh_ind.empty() ? nullptr : &sub_mesh_ind;

    // Output (size determined at runtime)
    arma::mat aabb;

    // Call library function
    CALL_QD(quadriga_lib::triangle_mesh_aabb<double>(mesh, aabb, p_sub_mesh_index, vec_size));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&aabb);
}