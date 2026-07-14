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
# CUBE
Construct a triangulated cube mesh

## Description:
- Generates a Blender-style cube: a 2 x 2 x 2 box centered at the origin (vertices at +/-1)
- Each of the 6 faces is split into 2 triangles, yielding 12 triangles at n_div = 1
- Optional uniform subdivision produces 12 · n_div^2 triangles
- Triangle winding is consistent (outward-facing normals), compatible with obj_file_write
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- An odd number of negative scale components flips the winding (inward-facing normals)

## Usage:
```
mesh = quadriga_lib.cube( scale, rotation, location, n_div );
```

## Inputs:
- **`scale`** *(optional)* — Length 1 scales all axes uniformly; length 3 scales {x,y,z}
  independently; empty or omitted = 1 (no scaling)
- **`rotation`** *(optional)* — Euler angles about {x,y,z}, applied as R = Rz·Ry·Rx (Blender
  XYZ); length 3; empty or omitted = no rotation
- **`location`** *(optional)* — Translation {x,y,z}; length 3; empty or omitted = origin
- **`n_div`** *(optional)* — Number of subdivisions per edge; yields 12 · n_div^2 triangles;
  default: 1

## Outputs:
- **`mesh`** — Triangle mesh; each row holds {x1,y1,z1,x2,y2,z2,x3,y3,z3}; `[12 · n_div^2, 9]`

## See also:
- [[icosphere]]
- [[obj_file_write]]
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs (empty = use C++ defaults)
    const arma::vec scale = (nrhs < 1) ? arma::vec() : qd_mex_get_Col<double>(prhs[0]);
    const arma::vec rotation = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1]);
    const arma::vec location = (nrhs < 3) ? arma::vec() : qd_mex_get_Col<double>(prhs[2]);
    const arma::uword n_div = (nrhs < 4) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[3], "n_div", 1);

    // Call library function (returns mesh by value)
    arma::mat mesh;
    CALL_QD(mesh = quadriga_lib::cube<double>(scale, rotation, location, n_div));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&mesh);
}