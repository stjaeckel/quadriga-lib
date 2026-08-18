// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# plane
Construct a triangulated plane mesh

- Generates a Blender-style plane: a 2 x 2 quad in the xy-plane centered at the origin (vertices at +/-1, z = 0)
- The quad is split into 2 triangles, yielding 2 triangles at n_div = 1
- Optional uniform subdivision produces 2 · n_div^2 triangles
- Triangle winding is consistent (normals point in +z direction), compatible with obj_file_write
- The plane has no thickness in z-direction; scaling therefore only affects x and y
- Scale, rotation, and translation are applied in that order (scale -> rotate -> translate)
- An odd number of negative scale components flips the winding (normals point in -z direction)

## Usage:
```
mesh = quadriga_lib.RTtools.plane( scale, rotation, location, n_div )
```

## Inputs:
- **`scale`** — Length 1 scales x and y uniformly; length 2 scales (x, y) independently; length 3 is
  accepted for compatibility with `cube`, but the third element is ignored; `None` or empty = 1
  (no scaling); default: `None`
- **`rotation`** — Euler angles about (x, y, z), applied as R = Rz·Ry·Rx (Blender XYZ); shape `(3,)`;
  `None` or empty = no rotation; default: `None`
- **`location`** — Translation (x, y, z); shape `(3,)`; `None` or empty = origin; default: `None`
- **`n_div`** — Number of subdivisions per edge; yields 2 · n_div^2 triangles; default: 1

## Outputs:
- **`mesh`** — Triangle mesh; each row holds (x1, y1, z1, x2, y2, z2, x3, y3, z3); `(2 · n_div^2, 9)`
MD!*/

py::array_t<double> plane(py::handle scale,
                          py::handle rotation,
                          py::handle location,
                          arma::uword n_div)
{
    // Read optional inputs (None or empty -> empty arma vec -> C++ defaults)
    const auto scale_a = qd_python_numpy2arma_Col<double>(scale, true);
    const auto rotation_a = qd_python_numpy2arma_Col<double>(rotation, true);
    const auto location_a = qd_python_numpy2arma_Col<double>(location, true);

    // Call library function (returns the mesh by value)
    arma::mat mesh = quadriga_lib::plane<double>(scale_a, rotation_a, location_a, n_div);

    // Copy to numpy
    return qd_python_copy2numpy(&mesh);
}

// pybind11 declaration:
// m.def("plane", &plane,
//       py::arg("scale") = py::none(),
//       py::arg("rotation") = py::none(),
//       py::arg("location") = py::none(),
//       py::arg("n_div") = 1);