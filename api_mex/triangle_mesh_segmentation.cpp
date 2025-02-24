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
# TRIANGLE_MESH_SEGMENTATION
Rearranges elements of a triangle mesh into smaller sub-meshes

## Description:
This function processes the elements of a large triangle mesh by clustering those that are 
closely spaced. The resulting mesh retains the same elements but rearranges their order. 
The function aims to minimize the size of the axis-aligned bounding box around each cluster, 
referred to as a sub-mesh, while striving to maintain a specific number of elements within 
each cluster.

This approach is particularly useful in computer graphics and simulation applications where 
managing computational resources efficiently is crucial. By organizing the mesh elements into 
compact clusters, the function enhances rendering performance and accelerates computational 
tasks, such as collision detection and physics simulations. It allows for quicker processing 
and reduced memory usage, making it an essential technique in both real-time graphics rendering 
and complex simulation environments.

## Usage:

```
[ triangles_out, sub_mesh_index, mesh_index ] = quadriga_lib.triangle_mesh_segmentation( ...
    triangles_in, target_size, vec_size );

[ triangles_out, sub_mesh_index, mesh_index, mtl_prop_out ] = ...
     quadriga_lib.triangle_mesh_segmentation( triangles_in, target_size, vec_size, mtl_prop_in );

```

## Input Arguments:

- **`triangles_in`**<br>
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; single or double precision;
  Size: `[ n_triangles_in, 9 ]`

- **`target_size`** (optional)<br>
  The target number of elements of each sub-mesh. Default value = 1024. For best performance, the 
  value should be around sgrt( n_triangles_in )

- **`vec_size`** (optional)<br>
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1. 
  For values > 1,the number of rows for each sub-mesh in the output is increased to a multiple 
  of `vec_size`. For padding, zero-sized triangles are placed at the center of the AABB of 
  the corresponding sub-mesh.
  
- **`mtl_prop_in`** (optional)<br>
  Material properties of each mesh element; Size: `[ n_triangles_in, 5 ]`

## Output Arguments:

- **`triangles_out`**<br>
  Vertices of the clustered mesh in global Cartesian coordinates; singe or double precision;
  Size: `[ n_triangles_out, 9 ]`

- **`sub_mesh_index`**<br>
  Start indices of the sub-meshes in 0-based notation. Type: uint32; Vector of length `[ n_sub_mesh ]`

- **`mesh_index`**<br>
  Indices for mapping elements of "triangles_in" to "triangles_out"; 1-based; 
  Length: `[ n_triangles_out ]`; For `vec_size > 1`, the added elements not contained in the input
  are indicated by zeros.

- **`mtl_prop_out`**<br>
  Material properties for the sub-divided triangle mesh elements. The values for the new faces are 
  copied from `mtl_prop_in`; Size: `[ n_triangles_out, 5 ]`; For `vec_size > 1`, the added elements
  will contain the vacuum / air material.


MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - triangles_in    Faces of the triangle mesh (input), Size: [ n_mesh_in, 9 ]
    //  1 - target_size     Target value for the sub-mesh size, default = 1024
    //  2 - vec_size        Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA), default = 1
    //  3 - mtl_prop_in     Material properties (input), Size: [ n_mesh_in, 5 ], optional (can be empty)

    // Output:
    //  0 - triangles_out   Reorganized mesh (output), Size: [ n_mesh_out, 9 ]
    //  1 - sub_mesh_index  Sub-mesh index, 0-based, Length: [ n_sub ]
    //  2 - mesh_index      Index mapping elements of "mesh" to "meshR", 1-based, Length: [ n_mesh_out ]
    //  3 - mtl_prop_out    Material properties (output), Size: [ n_mesh_out, 5 ], Empty if input "mtl_prop" is empty

    if (nrhs < 1 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_segmentation:IO_error", "Wrong number of input arguments.");

    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_segmentation:IO_error", "Too many output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_segmentation:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    if (nrhs > 3)
        if ((use_single && !mxIsSingle(prhs[3])) || (!use_single && !mxIsDouble(prhs[3])))
            mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_segmentation:IO_error", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    // Read inputs
    arma::fmat mesh_in_single, mtl_prop_in_single;
    arma::mat mesh_in_double, mtl_prop_in_double;

    if (use_single)
        mesh_in_single = qd_mex_reinterpret_Mat<float>(prhs[0]);
    else
        mesh_in_double = qd_mex_reinterpret_Mat<double>(prhs[0]);

    size_t target_size = (nrhs < 2) ? 1024 : qd_mex_get_scalar<size_t>(prhs[1], "target_size", 1024);
    size_t vec_size = (nrhs < 3) ? 1 : qd_mex_get_scalar<size_t>(prhs[2], "vec_size", 1);

    if (nrhs > 3 && !mxIsEmpty(prhs[3]))
    {
        if (use_single)
            mtl_prop_in_single = qd_mex_reinterpret_Mat<float>(prhs[3]);
        else
            mtl_prop_in_double = qd_mex_reinterpret_Mat<double>(prhs[3]);
    }

    // Reserve memory for the output
    arma::fmat mesh_out_single, mtl_prop_out_single;
    arma::mat mesh_out_double, mtl_prop_out_double;
    arma::u32_vec sub_mesh_index, mesh_index;

    // Call the quadriga-lib function
    try
    {
        if (use_single)
            quadriga_lib::triangle_mesh_segmentation(&mesh_in_single, &mesh_out_single,
                                                     &sub_mesh_index, target_size, vec_size,
                                                     &mtl_prop_in_single, &mtl_prop_out_single, &mesh_index);
        else
            quadriga_lib::triangle_mesh_segmentation(&mesh_in_double, &mesh_out_double,
                                                     &sub_mesh_index, target_size, vec_size,
                                                     &mtl_prop_in_double, &mtl_prop_out_double, &mesh_index);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_segmentation:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:triangle_mesh_segmentation:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Copy data to MATLAB / Octave
    if (nlhs > 0 && use_single)
        plhs[0] = qd_mex_copy2matlab(&mesh_out_single);
    else if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&mesh_out_double);

    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&sub_mesh_index);

    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&mesh_index);

    if (nlhs > 3 && use_single)
        plhs[3] = qd_mex_copy2matlab(&mtl_prop_out_single);
    else if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&mtl_prop_out_double);
}