// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# cube
Construct a triangulated cube mesh

- Generates a Blender-style cube: a 2 x 2 x 2 box centered at the origin (vertices at +/-1) before scaling
- Each of the 6 faces is split into 2 triangles, yielding 12 triangles at n_div = 1
- Optional uniform subdivision produces 12 * n_div^2 triangles
- Triangle winding is consistent (outward-facing normals), compatible with [[obj_file_write]]
- Scale, Euler rotation, and translation are applied in that order (scale -> rotate -> translate)

## Declaration:
```
arma::Mat<dtype> quadriga_lib::cube(
    const arma::vec &scale = {1.0},
    const arma::vec &rotation = {0.0, 0.0, 0.0},
    const arma::vec &location = {0.0, 0.0, 0.0},
    const arma::uword n_div = 1);
```

## Inputs:
- **`scale`** — Length 1 scales all axes uniformly; length 3 scales `{x,y,z}` independently; empty = 1 (no scaling)
- **`rotation`** — Euler angles in [rad] about {x,y,z}, applied as R = Rz*Ry*Rx (Blender XYZ); length 3 or empty (no rotation)
- **`location`** — Translation {x,y,z} in [m]; length 3 or empty (origin)
- **`n_div`** — Number of subdivisions per edge; results in `12 * n_div^2` triangles

## Returns:
- Triangle mesh; each row holds `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[12 * n_div^2, 9]`

## See also:
- [[icosphere]]
- [[subdivide_triangles]]
- [[obj_file_write]]
MD!*/

template <typename dtype>
arma::Mat<dtype> quadriga_lib::cube(const arma::vec &scale,
                                    const arma::vec &rotation,
                                    const arma::vec &location,
                                    const arma::uword n_div)
{
    if (n_div == 0)
        throw std::invalid_argument("Input 'n_div' cannot be 0.");

    // Resolve scale: empty = uniform 1, length 1 = uniform, length 3 = per-axis
    double sx = 1.0, sy = 1.0, sz = 1.0;
    if (scale.n_elem == 1)
        sx = sy = sz = scale.at(0);
    else if (scale.n_elem == 3)
        sx = scale.at(0), sy = scale.at(1), sz = scale.at(2);
    else if (scale.n_elem != 0)
        throw std::invalid_argument("Input 'scale' must have 0, 1, or 3 elements.");

    // Resolve rotation: empty = none, length 3 = Euler angles [rad] about x, y, z
    double rx = 0.0, ry = 0.0, rz = 0.0;
    if (rotation.n_elem == 3)
        rx = rotation.at(0), ry = rotation.at(1), rz = rotation.at(2);
    else if (rotation.n_elem != 0)
        throw std::invalid_argument("Input 'rotation' must have 0 or 3 elements.");

    // Resolve location: empty = origin, length 3 = translation
    double lx = 0.0, ly = 0.0, lz = 0.0;
    if (location.n_elem == 3)
        lx = location.at(0), ly = location.at(1), lz = location.at(2);
    else if (location.n_elem != 0)
        throw std::invalid_argument("Input 'location' must have 0 or 3 elements.");

    // Default cube: 6 faces split into 12 triangles, vertices at +/-1 (edge length 2)
    arma::Mat<dtype> base = {
        {-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},
        {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0},
        {-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0},
        {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},
        {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0},
        {-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0},
        {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0},
        {1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0},
        {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0},
        {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0},
        {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0},
        {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0}};

    // Subdivide each triangle into n_div^2 smaller triangles
    arma::Mat<dtype> mesh;
    quadriga_lib::subdivide_triangles(n_div, &base, &mesh);

    // Rotation matrix R = Rz * Ry * Rx (Blender Euler XYZ convention)
    double ca = std::cos(rx), sa = std::sin(rx);
    double cb = std::cos(ry), sb = std::sin(ry);
    double cc = std::cos(rz), sc = std::sin(rz);

    double R00 = cc * cb, R01 = cc * sb * sa - sc * ca, R02 = cc * sb * ca + sc * sa;
    double R10 = sc * cb, R11 = sc * sb * sa + cc * ca, R12 = sc * sb * ca - cc * sa;
    double R20 = -sb, R21 = cb * sa, R22 = cb * ca;

    // Apply scale -> rotate -> translate to all vertices (column-major, per corner)
    const arma::uword n_faces = mesh.n_rows;
    dtype *pm = mesh.memptr();
    for (arma::uword v = 0; v < 3; ++v) // three triangle corners
    {
        dtype *px = pm + (3 * v) * n_faces;
        dtype *py = pm + (3 * v + 1) * n_faces;
        dtype *pz = pm + (3 * v + 2) * n_faces;
        for (arma::uword n = 0; n < n_faces; ++n)
        {
            double x = sx * (double)px[n];
            double y = sy * (double)py[n];
            double z = sz * (double)pz[n];
            px[n] = dtype(R00 * x + R01 * y + R02 * z + lx);
            py[n] = dtype(R10 * x + R11 * y + R12 * z + ly);
            pz[n] = dtype(R20 * x + R21 * y + R22 * z + lz);
        }
    }

    return mesh;
}

template arma::fmat quadriga_lib::cube(const arma::vec &scale, const arma::vec &rotation, const arma::vec &location, const arma::uword n_div);
template arma::mat quadriga_lib::cube(const arma::vec &scale, const arma::vec &rotation, const arma::vec &location, const arma::uword n_div);
