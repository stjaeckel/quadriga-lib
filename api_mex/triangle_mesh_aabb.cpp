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
aabb = quadriga_lib.triangle_mesh_aabb( mesh, sub_mesh_index, vec_size );
```

## Inputs:
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; `[n_triangles, 9]`
- **`sub_mesh_index`** *(optional)* — 1-based start indices of sub-meshes; if omitted, the AABB
  of the entire mesh is returned; uint32; `[n_sub]`
- **`vec_size`** *(optional)* — Alignment size for SIMD/CUDA padding (e.g., `8` for AVX2, `32`
  for CUDA); default: 1

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
    arma::u32_vec sub_mesh_index = (nrhs < 2) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[1], true);
    const arma::uword vec_size = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "vec_size", 1);

    // Convert sub_mesh_index to 0-based
    for (unsigned &val : sub_mesh_index)
    {
        if (val == 0)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Values in 'sub_mesh_index' cannot be 0.");
        --val;
    }

    // Wrap optional input pointer
    const arma::u32_vec *p_sub_mesh_index = sub_mesh_index.empty() ? nullptr : &sub_mesh_index;

    // Output (returned by value, size determined at runtime)
    arma::mat aabb;

    // Call library function
    CALL_QD(aabb = quadriga_lib::triangle_mesh_aabb<double>(&mesh, p_sub_mesh_index, vec_size));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&aabb);
}