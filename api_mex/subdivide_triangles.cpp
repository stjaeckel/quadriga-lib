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
# SUBDIVIDE_TRIANGLES
Subdivide triangles into smaller triangles

- Uniformly subdivides each input triangle into `n_div x n_div` smaller triangles
- Output count: `n_triangles_out = n_triangles_in · n_div · n_div`
- Material indices are duplicated from the parent triangle to all sub-triangles

## Usage:
```
[ triangles_out, mtl_ind_out ] = quadriga_lib.subdivide_triangles( triangles_in, n_div, mtl_ind_in );
```

## Inputs:
- **`triangles_in`** — Mesh vertices as `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; `[n_triangles_in, 9]`
- **`n_div`** — Number of subdivisions per edge
- **`mtl_ind_in`** — 1-based material index per triangle (the `csv_ind` output of [[obj_file_read]]);
  0 = no material; `[n_triangles_in]`; passed through unchanged; default: `[]`

## Outputs:
- **`triangles_out`** — Subdivided mesh vertices, same column layout as `triangles_in`; `[n_triangles_out, 9]`
- **`mtl_ind_out`** — Material indices for the subdivided triangles, 1-based; uint64; `[n_triangles_out]`; only
  populated if `mtl_ind_in` is given
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

    // Material index: 1-based in C++ as well (0 = no material), no shift intended
    const arma::uvec mtl_ind = (nrhs < 3) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[2], false, "mtl_ind_in");

    // Compute output sizes; the output (n_triangles_in * n_div^2 rows, 9 values each) must fit into 64 bits,
    // checked before the outputs are allocated (same check and message as in C++)
    arma::uword n_triangles_in = triangles_in.n_rows;
    const arma::uword uword_max = std::numeric_limits<arma::uword>::max();
    if (n_div != 0 && (n_div > uword_max / n_div || n_triangles_in > uword_max / 9 / (n_div * n_div)))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'n_div' is too large.");
    arma::uword n_triangles_out = n_triangles_in * n_div * n_div;

    // Output allocation
    arma::mat triangles_out;
    arma::uvec mtl_ind_out;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&triangles_out, n_triangles_out, 9);

    if (nlhs > 1)
    {
        if (!mtl_ind.is_empty())
            plhs[1] = qd_mex_init_output(&mtl_ind_out, n_triangles_out);
        else
            plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
    }

    // Wrap optional pointers
    // Wrap optional pointers
    const arma::uvec *p_mtl_ind = mtl_ind.is_empty() ? nullptr : &mtl_ind;
    arma::uvec *p_mtl_ind_out = mtl_ind_out.is_empty() ? nullptr : &mtl_ind_out;

    // Call library function
    CALL_QD(quadriga_lib::subdivide_triangles<double>(n_div, &triangles_in, &triangles_out, p_mtl_ind, p_mtl_ind_out));
}