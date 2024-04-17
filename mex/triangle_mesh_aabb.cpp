// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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
# TRIANGLE_MESH_AABB
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

## Description:
The axis-aligned minimum bounding box (or AABB) for a given set of triangles is its minimum 
bounding box subject to the constraint that the edges of the box are parallel to the (Cartesian) 
coordinate axes. Axis-aligned bounding boxes are used as an approximate location of the set of 
triangles. In order to find intersections with the triangles (e.g. using ray tracing), the 
initial check is the intersections between the rays and the AABBs. Since it is usually a much 
less expensive operation than the check of the actual intersection (because it only requires 
comparisons of coordinates), it allows quickly excluding checks of the pairs that are far apart. 

## Usage:

```
aabb = quadriga_lib.subdivide_triangles( triangle_mesh, sub_mesh_index, vec_size );
```

## Input Arguments:

- **`triangles`**<br>
  Vertices of the triangle mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; single or double precision;
  Size: `[ n_triangles, 9 ]`

- **`sub_mesh_index`** (optional)<br>
  Start indices of the sub-meshes in 0-based notation. If this parameter is not given, the AABB of 
  the entire triangle mesh is returned. Type: uint32; Vector of length `[ n_sub_mesh ]`

- **`vec_size`** (optional)<br>
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1. For values > 1,
  the number of rows in the output is increased to a multiple of `vec_size`, padded with zeros. 

## Output Argument:

- **`aabb`**<br>
  Axis-aligned bounding box of each sub-mesh. Each box is described by 6 values: 
  `[ x_min, x_max, y_min, y_max, z_min, z_max ]`; Size: `[ n_sub_mesh, 6 ]`



MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - triangle_mesh   Faces of the triangular mesh (input), Size: [ n_mesh_in, 9 ]
    //  1 - sub_mesh_index  Sub-mesh index, 0-based, Length: [ n_sub ]
    //  2 - vec_size        Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA), default = 1

    // Output:
    //  0 - aabb            Axis-aligned bounding box (AABB) of the 3D mesh

    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_aabb:IO_error", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_aabb:IO_error", "Too many output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_aabb:IO_error",
                          "Input 'triangle_mesh' must be provided in 'single' or 'double' precision.");

    arma::fmat mesh_in_single;
    arma::mat mesh_in_double;
    if (use_single)
        mesh_in_single = qd_mex_reinterpret_Mat<float>(prhs[0]);
    else
        mesh_in_double = qd_mex_reinterpret_Mat<double>(prhs[0]);

    arma::u32_vec sub_mesh_index;
    if (nrhs > 1)
    {
        if (!mxIsUint32(prhs[1]))
            mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_aabb:IO_error", "Input 'sub_mesh_index' must be provided as 'uint32'.");

        sub_mesh_index = qd_mex_reinterpret_Col<unsigned>(prhs[1]);
    }

    size_t vec_size = (nrhs < 3) ? 1 : qd_mex_get_scalar<size_t>(prhs[2], "vec_size", 1);

    // Call the quadriga-lib function
    try
    {
        if (use_single)
        {
            arma::fmat aabb = quadriga_lib::triangle_mesh_aabb(&mesh_in_single, &sub_mesh_index, vec_size);
            plhs[0] = qd_mex_copy2matlab(&aabb);
        }
        else
        {
            arma::mat aabb = quadriga_lib::triangle_mesh_aabb(&mesh_in_double, &sub_mesh_index, vec_size);
            plhs[0] = qd_mex_copy2matlab(&aabb);
        }
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_aabb:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_aabb:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }
}