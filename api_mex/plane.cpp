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
# PLANE
Construct a triangulated plane mesh

## Description:
- Generates a Blender-style plane: a 2 x 2 quad in the xy-plane centered at the origin (vertices
  at +/-1, z = 0)
- The quad is split into 2 triangles, yielding 2 triangles at n_div = 1
- Optional uniform subdivision produces 2 · n_div^2 triangles
- Triangle winding is consistent (normals point in +z direction), compatible with obj_file_write
- The plane has no thickness in z-direction; scaling therefore only affects x and y
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- An odd number of negative scale components flips the winding (normals point in -z direction)

## Usage:
```
mesh = quadriga_lib.plane( scale, rotation, location, n_div );
```

## Inputs:
- **`scale`** *(optional)* — Length 1 scales x and y uniformly; length 2 scales {x,y}
  independently; length 3 is accepted for compatibility with `cube`, but the third element is
  ignored; empty or omitted = 1 (no scaling)
- **`rotation`** *(optional)* — Euler angles about {x,y,z}, applied as R = Rz·Ry·Rx (Blender
  XYZ); length 3; empty or omitted = no rotation
- **`location`** *(optional)* — Translation {x,y,z}; length 3; empty or omitted = origin
- **`n_div`** *(optional)* — Number of subdivisions per edge; yields 2 · n_div^2 triangles;
  default: 1

## Outputs:
- **`mesh`** — Triangle mesh; each row holds {x1,y1,z1,x2,y2,z2,x3,y3,z3}; `[2 · n_div^2, 9]`

## See also:
- [[cube]]
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
    CALL_QD(mesh = quadriga_lib::plane<double>(scale, rotation, location, n_div));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&mesh);
}