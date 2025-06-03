// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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
#include "mex_helper_functions.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

## Description:
- Uses raycasting to determine whether each 3D point lies inside a triangle mesh.
- Requires that the mesh is watertight and all normals are pointing outwards.
- For each point, multiple rays are cast in various directions.
- If any ray intersects a mesh element with a negative incidence angle, the point is classified as **inside**.
- Output can be binary (0 = outside, 1 = inside) or labeled with object indices.

## Usage:
```
result = quadriga_lib.point_inside_mesh( points, mesh, obj_ind, distance )
```

## Input Arguments:
- `**points**` (input)<br>
  3D point coordinates to test, size `[n_points, 3]`.

- `**mesh**` (input)<br>
  Triangular mesh faces. Each row represents a triangle using 3 vertices in row-major format
  (x1,y1,z1,x2,y2,z2,x3,y3,z3), size `[n_mesh, 9]`.

- `**obj_ind**` (optional input)<br>
  Optional object index for each mesh element (1-based), size `[n_mesh]`. If provided, the return
  vector will contain the index of the enclosing object instead of binary values.

- `**distance**` (optional input)<br>
  Optional distance in meters from objects that should be considered as *inside* the object.
  Possible range: 0 - 20 m. Using this parameter significantly increases computation time.

## Output Arguments:
- `**result**`<br>
  For each point: Returns `0` if the point is outside the mesh (or all objects), `1` if inside
  (or close to) any mesh object (if `obj_ind` not given), or returns the **1-based object index**
  if `obj_ind` is provided. Size: `[n_points]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto points_arma = qd_mex_get_double_Mat(prhs[0]);
    const auto mesh_arma = qd_mex_get_double_Mat(prhs[1]);
    const auto obj_ind_arma = (nrhs < 3) ? arma::u32_vec() : qd_mex_typecast_Col<unsigned>(prhs[2]);
    double distance = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "distance", 0.0);

    arma::u32_vec res;
    CALL_QD(res = quadriga_lib::point_inside_mesh(&points_arma, &mesh_arma, &obj_ind_arma, distance));

    if (nlhs == 1)
        plhs[0] = qd_mex_copy2matlab(&res);
}