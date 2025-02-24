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
# OBJ_FILE_READ
Reads a triangulated 3D polygon mesh from a Wavefront OBJ file

## Description:
The function imports a polygon mesh from an OBJ file. The OBJ file format is a straightforward data
format, exclusively representing 3D geometry. It details the position of each vertex and defines
polygons as lists of these vertices. By default, vertices are arranged in a counter-clockwise order,
eliminating the need for explicit declaration of face normals. When exporting the mesh from software
like Blender, it's essential to triangulate the mesh and include material definitions. If the
material name exists in the material database, the function loads the corresponding properties.
Otherwise, it defaults to using standard properties.

## Usage:

```
[ mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names ] = ...
    quadriga_lib.obj_file_read( fn, use_single );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the OBJ file, string

- **`use_single`** (optional)<br>
  Indicator if results should be returned in single precision, default = 0, returned in double precision

## Output Arguments:
- **`mesh`**<br>
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3 points
  in 3D-space. Hence, a face has 9 values in the order [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; <br>
  Size: `[ no_mesh, 9 ]`

- **`mtl_prop`**<br>
  Material properties of each mesh element; If no material is defined for an object, the properties 
  for vacuum are used. Size: `[ no_mesh, 5 ]`

- **`vert_list`**<br>
  List of vertices found in the OBJ file; Size: `[ no_vert, 3 ]`

- **`face_ind`**<br>
  Triangular faces are defined by three vertices. Vertex indices match the corresponding vertex elements
  of the previously defined `vert_list` (1-based indexing). <br>
  uint32; Size: `[ no_mesh, 3 ]`

- **`obj_id`**<br>
  Mesh elements in the OBJ file can be grouped into objects (e.g. 12 triangles define the walls of a
  cube). Each object is identified by a unique ID (1-based index). <br>
  uint32; Size: `[ no_mesh, 1 ]`

- **`mtl_id`**<br>
  Each mesh element gets assigned a material and each unique material gets assigned an ID. Different
  faces of an object can have different materials. If no material is defined in the OBJ file, the 
  id is set to `0` and no entry is made in `mtl_names`. <br>
  uint32; Size: `[ no_mesh, 1 ]`

- **`obj_names`**<br>
  Names of the objects in the OBJ file; Cell array of strings

- **`mtl_names`**<br>
  Names of the materials in the OBJ file; Cell array of strings

## Material properties:
Each material is defined by its electrical properties. Radio waves that interact with a building will
produce losses that depend on the electrical properties of the building materials, the material
structure and the frequency of the radio wave. The fundamental quantities of interest are the electrical
permittivity (ϵ) and the conductivity (σ). A simple regression model for the frequency dependence is
obtained by fitting measured values of the permittivity and the conductivity at a number of frequencies.
The five parameters returned in `mtl_prop` then are:

- Real part of relative permittivity at f = 1 GHz (a)
- Frequency dependence of rel. permittivity (b) such that ϵ = a · f^b
- Conductivity at f = 1 GHz (c)
- Frequency dependence of conductivity (d) such that σ = c· f^d
- Fixed attenuation in dB applied to each transition

A more detailed explanation together with a derivation can be found in ITU-R P.2040. The following
list of material is currently supported and the material can be selected by using the `usemtl` tag
in the OBJ file. When using Blender, the simply assign a material with that name to an object or face.
In addition, custom properties can be set by assigning adding the 5 properties after the material
name, separated by `:`, e.g.:

```
usemtl custom::2.1:0.1:0.1:0.5:20
```

The following materials are defined by default:

Name                  |         a |        b  |         c |         d |       Att |  max fGHz |
----------------------|-----------|-----------|-----------|-----------|-----------|-----------|
vacuum / air          |       1.0 |       0.0 |       0.0 |       0.0 |       0.0 |       100 |
textiles              |       1.5 |       0.0 |      5e-5 |      0.62 |       0.0 |       100 |
plastic               |      2.44 |       0.0 |   2.33e-5 |       1.0 |       0.0 |       100 |
ceramic               |       6.5 |       0.0 |    0.0023 |      1.32 |       0.0 |       100 |
sea_water             |      80.0 |     -0.25 |       4.0 |      0.58 |       0.0 |       100 |
sea_ice               |       3.2 |    -0.022 |       1.1 |       1.5 |       0.0 |       100 |
water                 |      80.0 |     -0.18 |       0.6 |      1.52 |       0.0 |        20 |
water_ice             |      3.17 |    -0.005 |    5.6e-5 |       1.7 |       0.0 |        20 |
itu_concrete          |      5.24 |       0.0 |    0.0462 |    0.7822 |       0.0 |       100 |
itu_brick             |      3.91 |       0.0 |    0.0238 |      0.16 |       0.0 |        40 |
itu_plasterboard      |      2.73 |       0.0 |    0.0085 |    0.9395 |       0.0 |       100 |
itu_wood              |      1.99 |       0.0 |    0.0047 |    1.0718 |       0.0 |       100 |
itu_glass             |      6.31 |       0.0 |    0.0036 |    1.3394 |       0.0 |       100 |
itu_ceiling_board     |      1.48 |       0.0 |    0.0011 |     1.075 |       0.0 |       100 |
itu_chipboard         |      2.58 |       0.0 |    0.0217 |      0.78 |       0.0 |       100 |
itu_plywood           |      2.71 |       0.0 |      0.33 |       0.0 |       0.0 |        40 |
itu_marble            |     7.074 |       0.0 |    0.0055 |    0.9262 |       0.0 |        60 |
itu_floorboard        |      3.66 |       0.0 |    0.0044 |    1.3515 |       0.0 |       100 |
itu_metal             |       1.0 |       0.0 |     1.0e7 |       0.0 |       0.0 |       100 |
itu_very_dry_ground   |       3.0 |       0.0 |   0.00015 |      2.52 |       0.0 |        10 |
itu_medium_dry_ground |      15.0 |      -0.1 |     0.035 |      1.63 |       0.0 |        10 |
itu_wet_ground        |      30.0 |      -0.4 |      0.15 |       1.3 |       0.0 |        10 |
itu_vegetation        |       1.0 |       0.0 |    1.0e-4 |       1.1 |       0.0 |       100 |
irr_glass             |      6.27 |       0.0 |    0.0043 |    1.1925 |      23.0 |       100 |

MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the OBJ file (including path); string
    //  1 - use_single      Switch for single precision, default = 0 (double precision)

    // Output:
    //  0 - mesh            Vertices of the triangular mesh, Size [ no_mesh, 9 ]
    //  1 - mtl_prop        Material properties of each mesh element; Size [ no_mesh, 5 ]
    //  2 - vert_list       List of vertices found in the OBJ file; Size [ no_vert, 3 ]
    //  3 - face_ind        Face indices (=entries in vert_list); uint32; 1-based; Size [ no_mesh, 3 ]
    //  4 - obj_id          Object index; uint32; Size [ no_mesh, 1 ]
    //  5 - mtl_id          Material index; uint32; Size [ no_mesh, 1 ]
    //  6 - obj_names       Names of the objects in the OBJ file; Cell array of strings
    //  7 - mtl_names       Names of the materials in the OBJ file; Cell array of strings

    // Notes:
    // Material values are:
    //     * Real part of relative permittivity at f = 1 GHz (a)
    //     * Frequency dependence of rel. permittivity (b) such that ϵ = a · f^b
    //     * Conductivity at f = 1 GHz (c)
    //     * Frequency dependence of conductivity (d) such that σ = c· f^d
    //     * Fixed attenuation in dB

    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:io_error", "Filename is missing.");

    if (nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:io_error", "Too many input arguments.");

    if (nlhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:io_error", "Too many output arguments.");

    // Read filename
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:qdant_write:IO_error", "Input 'fn' must be a string.");

    auto mx_fn = mxArrayToString(prhs[0]);
    std::string fn = std::string(mx_fn);
    mxFree(mx_fn);

    // Read "use_single"
    bool use_single = nrhs < 2 ? false : qd_mex_get_scalar<bool>(prhs[1], "use_single", false);

    // Declare Armadillo variables
    arma::Mat<double> mesh_double, mtl_prop_double, vert_list_double;
    arma::Mat<float> mesh_single, mtl_prop_single, vert_list_single;
    arma::Mat<unsigned> face_ind;
    arma::Col<unsigned> obj_ind, mtl_ind;
    std::vector<std::string> obj_names, mtl_names;

    // Read data from file
    try
    {
        if (use_single)
            quadriga_lib::obj_file_read<float>(fn, &mesh_single, &mtl_prop_single, &vert_list_single, &face_ind, &obj_ind, &mtl_ind, &obj_names, &mtl_names);
        else
            quadriga_lib::obj_file_read<double>(fn, &mesh_double, &mtl_prop_double, &vert_list_double, &face_ind, &obj_ind, &mtl_ind, &obj_names, &mtl_names);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Convert 0-based indexing to 1-based indexing
    face_ind = face_ind + 1;

    // Write to MATLAB / Octave
    if (nlhs > 0 && use_single)
        plhs[0] = qd_mex_copy2matlab(&mesh_single);
    else if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&mesh_double);

    if (nlhs > 1 && use_single)
        plhs[1] = qd_mex_copy2matlab(&mtl_prop_single);
    else if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&mtl_prop_double);

    if (nlhs > 2 && use_single)
        plhs[2] = qd_mex_copy2matlab(&vert_list_single);
    else if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&vert_list_double);

    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&face_ind);

    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&obj_ind);

    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&mtl_ind);

    if (nlhs > 6)
    {
        size_t n_obj = obj_names.size();
        auto *cellArray = mxCreateCellMatrix((mwSize)n_obj, 1);
        for (size_t i = 0; i < n_obj; ++i)
        {
            auto *mxStr = mxCreateString(obj_names[i].c_str());
            mxSetCell(cellArray, i, mxStr);
        }
        plhs[6] = cellArray;
    }

    if (nlhs > 7)
    {
        size_t n_mtl = mtl_names.size();
        auto *cellArray = mxCreateCellMatrix((mwSize)n_mtl, 1);
        for (size_t i = 0; i < n_mtl; ++i)
        {
            auto *mxStr = mxCreateString(mtl_names[i].c_str());
            mxSetCell(cellArray, i, mxStr);
        }
        plhs[7] = cellArray;
    }
}
