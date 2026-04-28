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
# TRIANGLE_MESH_SEGMENTATION
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more
  than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with
  zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; `mesh_index` uses 0 to mark
  padding entries
- If `mtl_prop_in` is provided, material rows are reordered and padded in the same way

## Usage:
```
[ triangles_out, sub_mesh_index, mesh_index, mtl_prop_out ] = ...
    quadriga_lib.triangle_mesh_segmentation( triangles_in, target_size, vec_size, mtl_prop_in );
```

## Inputs:
- **`triangles_in`** — Triangle vertices, each row [v1x,v1y,v1z, v2x,v2y,v2z, v3x,v3y,v3z]; `[n_mesh, 9]`
- **`target_size`** *(optional)* — Target triangle count per sub-mesh; for best performance set
  near sqrt(n_mesh); default: 1024
- **`vec_size`** *(optional)* — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each
  sub-mesh row count is rounded up to a multiple of this value; default: 1
- **`mtl_prop_in`** *(optional)* — Material properties; see obj_file_read; `[n_mesh, 9]`

## Outputs:
- **`triangles_out`** — Reordered and padded triangle vertices; `[n_meshR, 9]`
- **`sub_mesh_index`** — 1-based start indices of sub-meshes in `triangles_out`; `[n_sub]`
- **`mesh_index`** *(optional)* — 1-based mapping from original to reorganized mesh
  (0 = padding); `[n_meshR]`
- **`mtl_prop_out`** *(optional)* — Reordered and padded material properties; `[n_meshR, 9]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const arma::mat mesh = qd_mex_get_Mat<double>(prhs[0]);
    const arma::uword target_size = (nrhs < 2) ? 1024 : qd_mex_get_scalar<arma::uword>(prhs[1], "target_size", 1024);
    const arma::uword vec_size = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "vec_size", 1);
    const arma::mat mtl_prop = (nrhs < 4) ? arma::mat() : qd_mex_get_Mat<double>(prhs[3]);

    // Output containers (sizes are determined inside the C++ call)
    arma::mat meshR;
    arma::u32_vec sub_mesh_index;
    arma::mat mtl_propR;
    arma::u32_vec mesh_index;

    // Wrap optional pointers
    const arma::mat *p_mtl_prop = mtl_prop.empty() ? nullptr : &mtl_prop;
    arma::mat *p_mtl_propR = (nlhs > 3) ? &mtl_propR : nullptr;
    arma::u32_vec *p_mesh_index = (nlhs > 2) ? &mesh_index : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::triangle_mesh_segmentation<double>(&mesh, &meshR, &sub_mesh_index,
                                                             target_size, vec_size, p_mtl_prop, p_mtl_propR, p_mesh_index));

    // Convert sub_mesh_index from 0-based (C++) to 1-based (MATLAB)
    sub_mesh_index += 1U;

    // Copy outputs to MATLAB (sizes only known after the C++ call)
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&meshR);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&sub_mesh_index);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&mesh_index);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&mtl_propR);
}