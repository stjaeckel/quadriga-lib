// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-Specific Simulations
SECTION!*/

/*!MD
# SUBDIVIDE_TRIANGLES
Subdivide triangles into smaller triangles

- Uniformly subdivides each input triangle into `n_div x n_div` smaller triangles
- Output count: `n_triangles_out = n_triangles_in · n_div · n_div`
- Material properties are duplicated from parent triangle to all sub-triangles

## Usage:
```
[ triangles_out, mtl_prop_out ] = quadriga_lib.subdivide_triangles( triangles_in, n_div, mtl_prop_in );
```

## Inputs:
- **`triangles_in`** — Mesh vertices as `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; `[n_triangles_in, 9]`
- **`n_div`** — Number of subdivisions per edge
- **`mtl_prop`** *(optional)* — Material properties; see `obj_file_read`; `[n_triangles_in, 9]`

## Outputs:
- **`triangles_out`** — Subdivided mesh vertices, same column layout as `triangles_in`; `[n_triangles_out, 9]`
- **`mtl_prop_out`** *(optional)* — Material properties for subdivided triangles; `[n_triangles_out, 9]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const arma::mat triangles_in = qd_mex_get_Mat<double>(prhs[0]);
    const arma::uword n_div = (nrhs < 2) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[1], "n_div", 1);
    const arma::mat mtl_prop = (nrhs < 3) ? arma::mat() : qd_mex_get_Mat<double>(prhs[2]);

    // Compute output sizes
    arma::uword n_triangles_in = triangles_in.n_rows;
    arma::uword n_triangles_out = n_triangles_in * n_div * n_div;

    // Output allocation
    arma::mat triangles_out, mtl_prop_out;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&triangles_out, n_triangles_out, 9);

    if (nlhs > 1)
    {
        if (!mtl_prop.empty())
            plhs[1] = qd_mex_init_output(&mtl_prop_out, n_triangles_out, 9);
        else
            plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
    }

    // Wrap optional pointers
    const arma::mat *p_mtl_prop = mtl_prop.empty() ? nullptr : &mtl_prop;
    arma::mat *p_mtl_prop_out = mtl_prop_out.empty() ? nullptr : &mtl_prop_out;

    // Call library function
    CALL_QD(quadriga_lib::subdivide_triangles<double>(n_div, &triangles_in, &triangles_out,
                                                      p_mtl_prop, p_mtl_prop_out));
}