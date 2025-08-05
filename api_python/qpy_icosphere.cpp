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

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-Specific Simulation Tools
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
from quadriga_lib import RTtools
center, length, vert, direction = RTtools.icosphere( no_div, radius, direction_xyz )
```

## Input Arguments:
- **`no_div`**<br>
  Number of divisions per edge of the generating icosahedron. The resulting number of faces is
  equal to `no_face = 20 · no_div^2`

- **`radius`**<br>
  Radius of the sphere in meters

- **`direction_xyz`**<br>
  Direction format indicator: 0 = Spherical (default), 1 = Cartesian


## Output Arguments (tuple containing 4 values):
- **`center`**<br>
  Position of the center point of each triangle; Shape: `( no_face, 3 )`

- **`length`**<br>
  Length of the vector pointing from the origin to the center point. This number is smaller than
  1 since the triangles are located inside the unit sphere; Shape: `( no_face )`

- **`vert`**<br>
  The 3 vectors pointing from the center point to the vertices of each triangle; the values are
  in the order `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Shape: `( no_face, 9 )`

- **`direction`**<br>
  The directions of the vertex-rays. If the format indicator `direction_xyz` is set to `0`, the
  output is in geographic coordinates (azimuth and elevation angle in rad); the values are in the
  order `( v1az, v1el, v2az, v2el, v3az, v3el ]`;Shape: `( no_face, 6 )` If the format indicator
  `direction_xyz` is set to `1`, the output is in Cartesian coordinates and the values are in the
  order `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z  ]`; Shape: `( no_face, 9 )`
MD!*/

py::tuple icosphere(unsigned long long n_div, double radius, bool direction_xyz)
{
    arma::mat center, vert, direction;
    arma::vec length;
    unsigned long long n_faces = quadriga_lib::icosphere<double>(n_div, radius, &center, &length, &vert, &direction, direction_xyz);

    return py::make_tuple(py::array_t<double>({n_faces, 3ULL}, center.memptr()),
                          py::array_t<double>(n_faces, length.memptr()),
                          py::array_t<double>({n_faces, 9ULL}, vert.memptr()),
                          py::array_t<double>({n_faces, (unsigned long long)direction.n_cols}, direction.memptr()));
}