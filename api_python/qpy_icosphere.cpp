// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# icosphere
Construct a geodesic polyhedron from recursive icosahedron subdivision

- Produces 20 · n_div² triangular faces, each pointing outward from origin
- All vertices lie on a sphere of specified radius
- Suitable for uniform angular sampling (ray tracing, antenna patterns, spatial grids)

## Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.icosphere( no_div, radius, direction_xyz )

# Unpacked outputs
center, length, vert, direction = quadriga_lib.RTtools.icosphere( no_div, radius, direction_xyz )
```

## Inputs:
- **`n_div`** — Number of subdivisions; generates 20 · n_div² faces; default: 1
- **`radius`** — Radius of icosphere in meters; default: 1
- **`direction_xyz`** *(optional)* — Output directions in Cartesian (true) or spherical
  azimuth/elevation (false); default: false

## Outputs:
- **`center`** — Face center coordinates in Cartesian space; each vector points radially outward
  from origin with magnitude equal to the inradius of the face; `(n_faces, 3)`
- **`length`** — Distance from origin to face plane; equals the magnitude of each
  `center` vector; `(n_faces,)`
- **`vert`** — Vertex offsets from face center `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `(n_faces, 9)`
- **`direction`** — Edge directions; spherical `{az1,el1,az2,el2,az3,el3}` or Cartesian
  `{x1,y1,z1,x2,y2,z2,x3,y3,z3}` per `direction_xyz` flag; `(n_faces, 6)` or `[n_faces, 9)`
MD!*/

py::tuple icosphere(arma::uword n_div, double radius, bool direction_xyz)
{
    const arma::uword n_faces = 20 * n_div * n_div;
    const arma::uword n_dir_cols = direction_xyz ? 9 : 6;

    arma::mat center, vert, direction;
    arma::vec length;

    auto center_p = qd_python_init_output(n_faces, 3, &center);
    auto length_p = qd_python_init_output(n_faces, &length);
    auto vert_p = qd_python_init_output(n_faces, 9, &vert);
    auto cdirection_p = qd_python_init_output(n_faces, n_dir_cols, &direction);

    quadriga_lib::icosphere<double>(n_div, radius, &center, &length, &vert, &direction, direction_xyz);

    return py::make_tuple(center_p, length_p, vert_p, cdirection_p);
}