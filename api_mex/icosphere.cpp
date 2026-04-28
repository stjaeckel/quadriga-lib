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
# ICOSPHERE
Construct a geodesic polyhedron from recursive icosahedron subdivision

- Produces 20 · n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)

## Usage:
```
[ center, length, vert, direction ] = quadriga_lib.icosphere( n_div, radius, direction_xyz );
```

## Inputs:
- **`n_div`** — Number of subdivisions; generates 20 · n_div² faces; default: 1
- **`radius`** — Radius of icosphere in meters; default: 1
- **`direction_xyz`** *(optional)* — Output directions in Cartesian (true) or spherical
  azimuth/elevation (false); default: false

## Outputs:
- **`center`** — Face center coordinates in Cartesian space; each vector points radially outward
  from origin with magnitude equal to the inradius of the face; `[n_faces, 3]`
- **`length`** *(optional)* — Distance from origin to face plane; equals the magnitude of each
  `center` vector; `[n_faces]`
- **`vert`** *(optional)* — Vertex offsets from face center [x1,y1,z1,x2,y2,z2,x3,y3,z3]; `[n_faces, 9]`
- **`direction`** *(optional)* — Edge directions; spherical [az1,el1,az2,el2,az3,el3] or Cartesian
  [x1,y1,z1,x2,y2,z2,x3,y3,z3] per `direction_xyz` flag; `[n_faces, 6]` or `[n_faces, 9]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input arguments
    const arma::uword n_div = (nrhs < 1) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[0], "n_div", 1);
    const double radius = (nrhs < 2) ? 1.0 : qd_mex_get_scalar<double>(prhs[1], "radius", 1.0);
    const bool direction_xyz = (nrhs < 3) ? false : qd_mex_get_scalar<bool>(prhs[2], "direction_xyz", false);

    // Output sizes are known up front
    const arma::uword n_faces = 20 * n_div * n_div;
    const arma::uword n_dir_cols = direction_xyz ? 9 : 6;

    // Output allocation
    arma::mat center, vert, direction;
    arma::vec length;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&center, n_faces, 3);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&length, n_faces);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&vert, n_faces, 9);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&direction, n_faces, n_dir_cols);

    // Wrap optional output pointers (center is required by C++, always pass &center)
    arma::vec *p_length = length.empty() ? nullptr : &length;
    arma::mat *p_vert = vert.empty() ? nullptr : &vert;
    arma::mat *p_direction = direction.empty() ? nullptr : &direction;

    // Call library function
    CALL_QD(quadriga_lib::icosphere<double>(n_div, radius, &center, p_length, p_vert, p_direction, direction_xyz));
}