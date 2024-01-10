// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.cpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# ICOSPHERE
Construct a geodesic polyhedron (icosphere), a convex polyhedron made from triangles

## Description:
An icosphere is constructed by subdividing faces of an icosahedron, a polyhedron with 20 faces,
12 vertices and 30 edges, and then projecting the new vertices onto the surface of a sphere. The
resulting mesh has 6 triangles at each vertex, except for 12 vertices which have 5 triangles.
The approximate equilateral triangles have roughly the same edge length and surface area.

## Usage:

```
[ center, length, vert, direction ] = quadriga_lib.icosphere( no_div, radius );
```

## Input Arguments:
- **`no_div`**<br>
  Number of divisions per edge of the generating icosahedron. The resulting number of faces is
  equal to `no_face = 20 · no_div^2`

- **`radius`**<br>
  Radius of the sphere in meters


## Output Arguments:
- **`center`**<br>
  Position of the center point of each triangle; Size: `[ no_face, 3 ]`

- **`length`**<br>
  Length of the vector pointing from the origin to the center point. This number is smaller than
  1 since the triangles are located inside the unit sphere; <br>Size: `[ no_face ]`

- **`vert`**<br>
  The 3 vectors pointing from the center point to the vertices of each triangle; the values are
  in the order `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; <br>Size: `[ no_face, 9 ]`

- **`direction`**<br>
  The directions of the vertex-rays in geographic coordinates (azimuth and elevation angle in
  rad); the values are in the order `[ v1az, v1el, v2az, v2el, v3az, v3el ]`; <br>Size: `[ no_face, 6 ]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - n_div           Number of sub-segments per edge, results in n_faces = 20 * n_div^2 elements
    //  1 - radius          Radius of the icosphere in meters

    // Output:
    //  0 - center          Pointing vector from the origin to the center of the triangle, matrix of size [no_faces, 3]
    //  1 - length          Length of the pointing vector "center" (slightly smaller than 1), vector of length [no_faces]
    //  2 - vert            Vectors pointing from "center" to the vertices of the triangle, matrix of size [no_ray, 9], [x1 y1 z1 x2 y2 z3 x3 y3 z3]
    //  3 - direction       Directions of the vertex-rays in rad; matrix of size [no_ray, 6], the values are in the order [ v1az, v1el, v2az, v2el, v3az, v3el ]

    if (nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:icosphere:io_error", "Too many input arguments.");

    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:icosphere:io_error", "Too many output arguments.");

    // Read inputs
    unsigned long long n_div = nrhs < 1 ? 1ULL : qd_mex_get_scalar<unsigned long long>(prhs[0], "n_div", 1ULL);
    double radius = nrhs < 2 ? 1.0 : qd_mex_get_scalar<double>(prhs[1], "radius", 1.0);

    // Calculate number of rows in the output
    mwSize n_rows = 20 * (mwSize)n_div * (mwSize)n_div;

    // Reserve memory for the output
    arma::mat center, vert, direction;
    arma::vec length;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&center, n_rows, 3ULL);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&length, n_rows);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&vert, n_rows, 9ULL);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&direction, n_rows, 6ULL);

    // Call the quadriga-lib function
    try
    {
        if (nlhs > 3)
            quadriga_lib::icosphere<double>(n_div, radius, &center, &length, &vert, &direction);
        else if (nlhs > 2)
            quadriga_lib::icosphere<double>(n_div, radius, &center, &length, &vert);
        else if (nlhs > 1)
            quadriga_lib::icosphere<double>(n_div, radius, &center, &length);
        else if (nlhs > 0)
            quadriga_lib::icosphere<double>(n_div, radius, &center);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:icosphere:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:icosphere:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }
}