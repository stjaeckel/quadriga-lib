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
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# RAY_TRIANGLE_INTERSECT
Calculates the intersection of rays and triangles in three dimensions

## Description:
- This function implements the Möller–Trumbore ray-triangle intersection algorithm, known for its
  efficiency in calculating the intersection of a ray and a triangle in three-dimensional space.
  This method achieves its speed by eliminating the need for precomputed plane equations of the plane
  containing the triangle.

- For further information, refer to [Wikipedia: <a href="https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm">Möller–Trumbore intersection algorithm</a>].

- The algorithm defines the ray using two points: an origin and a destination. Similarly, the triangle
  is specified by its three vertices. 
  
- To enhance performance, this implementation leverages AVX2 intrinsic functions and OpenMP, when 
  available, to speed up the computational process.

## Usage:

```
[ fbs, sbs, no_interact, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( orig, dest, mesh );
```

## Input Arguments:
- **`orig`**<br>
  Ray origins in 3D Cartesian coordinates; Size: `[ no_ray, 3 ]`

- **`dest`**<br>
  Ray destinations in 3D Cartesian coordinates; Size: `[ no_ray, 3 ]`

- **`mesh`**<br>
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3 points
  in 3D-space. Hence, a face has 9 values in the order [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; <br>
  Size: `[ no_mesh, 9 ]`

## Output Arguments:
- **`fbs`**<br>
  First interaction point between the rays and the triangular mesh. If no interaction was found, the
  FBS location is equal to `dest`. Size: `[ no_ray, 3 ]`

- **`sbs`**<br>
  Second interaction point between the rays and the triangular mesh. If no interaction was found, the
  SBS location is equal to `dest`. Size: `[ no_ray, 3 ]`

- **`no_interact`**<br>
  Total number of interactions between the origin point and the destination; uint32; Length: `[ no_ray ]`

- **`fbs_ind`**<br>
  Index of the triangle that was hit by the ray at the FBS location; 1-based; uint32; Length: `[ no_ray ]`

- **`sbs_ind`**<br>
  Index of the triangle that was hit by the ray at the SBS location; 1-based; uint32; Length: `[ no_ray ]`

## Caveat:
- `orig`, `dest`, and `mesh` can be provided in single or double precision; `fbs` and `lbs` will have
  the same type.
- All internal computation are done in single precision to achieve an additional 2x improvement in 
  speed compared to double precision when using AVX2 intrinsic instructions
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - orig        Ray origin points in GCS, Size [ n_ray, 3 ]
    //  1 - dest        Ray destination points in GCS, Size [ n_ray, 3 ]
    //  3 - mesh        Faces of the triangular mesh, Size: [ n_mesh, 9 ]

    // Outputs:
    //  0 - fbs         First interaction points in GCS, Size [ n_ray, 3 ]
    //  1 - sbs         Second interaction points in GCS, Size [ n_ray, 3 ]
    //  2 - no_interact   Number of mesh between orig and dest, Size [ n_ray, 1 ]
    //  3 - fbs_ind     Index of first hit mesh element, 1-based, 0 = no hit, Size [ n_ray ]
    //  4 - sbs_ind     Index of second hit mesh element, 1-based, 0 = no hit, Size [ n_ray ]

    if (nrhs != 3)
        mexErrMsgIdAndTxt("quadriga_lib:ray_triangle_intersect:IO_error", "Need exactly 3 input arguments: orig, dest and mesh.");

    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:ray_triangle_intersect:IO_error", "Too many output arguments.");

    // Read inputs
    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:ray_triangle_intersect:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 1; i < 3; ++i)
        if (nrhs > i)
            if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
                mexErrMsgIdAndTxt("quadriga_lib:ray_triangle_intersect:IO_error", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    arma::fmat orig_single, dest_single, mesh_single, fbs_single, sbs_single;
    arma::mat orig_double, dest_double, mesh_double, fbs_double, sbs_double;
    arma::u32_vec no_interact, fbs_ind, sbs_ind;

    if (use_single)
    {
        orig_single = qd_mex_reinterpret_Mat<float>(prhs[0]);
        dest_single = qd_mex_reinterpret_Mat<float>(prhs[1]);
        mesh_single = qd_mex_reinterpret_Mat<float>(prhs[2]);
    }
    else
    {
        orig_double = qd_mex_reinterpret_Mat<double>(prhs[0]);
        dest_double = qd_mex_reinterpret_Mat<double>(prhs[1]);
        mesh_double = qd_mex_reinterpret_Mat<double>(prhs[2]);
    }

    // Number of rays
    arma::uword n_rays = use_single ? orig_single.n_rows : orig_double.n_rows;

    // Initialize output memory
    if (nlhs > 0 && use_single)
        plhs[0] = qd_mex_init_output(&fbs_single, n_rays, 3);
    else if (nlhs > 0) // double
        plhs[0] = qd_mex_init_output(&fbs_double, n_rays, 3);

    if (nlhs > 1 && use_single)
        plhs[1] = qd_mex_init_output(&sbs_single, n_rays, 3);
    else if (nlhs > 1) // double
        plhs[1] = qd_mex_init_output(&sbs_double, n_rays, 3);

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&no_interact, n_rays);

    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&fbs_ind, n_rays);

    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&sbs_ind, n_rays);

    // Call library function
    try
    {
        if (use_single)
        {
            if (nlhs == 1)
                quadriga_lib::ray_triangle_intersect(&orig_single, &dest_single, &mesh_single, &fbs_single);
            else if (nlhs == 2)
                quadriga_lib::ray_triangle_intersect(&orig_single, &dest_single, &mesh_single, &fbs_single, &sbs_single);
            else if (nlhs == 3)
                quadriga_lib::ray_triangle_intersect(&orig_single, &dest_single, &mesh_single, &fbs_single, &sbs_single, &no_interact);
            else if (nlhs == 4)
                quadriga_lib::ray_triangle_intersect(&orig_single, &dest_single, &mesh_single, &fbs_single, &sbs_single, &no_interact, &fbs_ind);
            else if (nlhs == 5)
                quadriga_lib::ray_triangle_intersect(&orig_single, &dest_single, &mesh_single, &fbs_single, &sbs_single, &no_interact, &fbs_ind, &sbs_ind);
        }
        else // double
        {
            if (nlhs == 1)
                quadriga_lib::ray_triangle_intersect(&orig_double, &dest_double, &mesh_double, &fbs_double);
            else if (nlhs == 2)
                quadriga_lib::ray_triangle_intersect(&orig_double, &dest_double, &mesh_double, &fbs_double, &sbs_double);
            else if (nlhs == 3)
                quadriga_lib::ray_triangle_intersect(&orig_double, &dest_double, &mesh_double, &fbs_double, &sbs_double, &no_interact);
            else if (nlhs == 4)
                quadriga_lib::ray_triangle_intersect(&orig_double, &dest_double, &mesh_double, &fbs_double, &sbs_double, &no_interact, &fbs_ind);
            else if (nlhs == 5)
                quadriga_lib::ray_triangle_intersect(&orig_double, &dest_double, &mesh_double, &fbs_double, &sbs_double, &no_interact, &fbs_ind, &sbs_ind);
        }
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:ray_triangle_intersect:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:ray_triangle_intersect:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }
}