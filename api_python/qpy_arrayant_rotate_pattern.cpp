// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# rotate_pattern
Rotate antenna radiation patterns around the principal axes using Euler rotations

- Rotates the pattern and/or polarization of array antenna elements around the x (bank), y (tilt), and z (heading) axes
- Rotations are applied in the order x, y, z, composed as Rz·Ry·Rx (intrinsic Tait-Bryan)
- Single-frequency input (3D pattern fields): `usage` 0 and 1 adjust the sampling grid for non-uniformly sampled antennas
- Frequency-dependent input (4D pattern fields): the rotation is applied to every entry and the grid is never adjusted,
  since all entries must share one grid
- For scalar acoustic fields (pressure stored in `e_theta_re` only) use `usage = 1` to avoid spurious polarization effects

## Usage:
```
# Rotate all elements by 45 deg bank
arrayant_out = quadriga_lib.arrayant.rotate_pattern( arrayant, x_deg=45.0 )

# Rotate only elements 0 and 1 by 90 deg heading
arrayant_out = quadriga_lib.arrayant.rotate_pattern( arrayant, z_deg=90.0, element=[0, 1] )

# Frequency-dependent input (4D patterns) — same interface
arrayant_out = quadriga_lib.arrayant.rotate_pattern( arrayant, y_deg=10.0 )
```

## Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as in [[generate]]; pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`x_deg`** — Rotation around the x-axis (bank) in degrees; default: 0.0
- **`y_deg`** — Rotation around the y-axis (tilt) in degrees; default: 0.0
- **`z_deg`** — Rotation around the z-axis (heading) in degrees; default: 0.0
- **`usage`** — Rotation mode; default: 0<br><br>
   | Mode | Pattern | Polarization | Grid adj. |
   | :--: | :-----: | :----------: | :-------: |
   | 0    | Yes     | Yes          | Yes       |
   | 1    | Yes     | No           | Yes       |
   | 2    | No      | Yes          | No        |
   | 3    | Yes     | Yes          | No        |
   | 4    | Yes     | No           | No        |

   for 4D input the grid is never adjusted, so `0`/`3` and `1`/`4` are equivalent
- **`element`** — Element indices to rotate; 1D list or array of int; `None` or empty rotates all
  elements; default: `None`

## Outputs:
- **`arrayant_out`** — Dict with the rotated array antenna data; same keys and layout as
  `arrayant`; single-frequency dict for 3D input, frequency-dependent dict for 4D input
MD!*/

py::dict arrayant_rotate_pattern(const py::dict &arrayant,
                                 double x_deg,
                                 double y_deg,
                                 double z_deg,
                                 unsigned usage,
                                 py::handle element)
{
    // Element indices (0-based); empty = rotate all elements
    const arma::uvec element_ind = qd_python_numpy2arma_Col<arma::uword>(element, true);

    // Parse the (possibly frequency-dependent) antenna
    auto ant_vec = qd_python_dict2arrayant_multi(arrayant, false, false, true);

    if (ant_vec.size() > 1) // Frequency-dependent: rotate every entry
    {
        unsigned usage_multi = (usage == 3) ? 0 : ((usage == 4) ? 1 : usage);
        quadriga_lib::arrayant_rotate_pattern_multi(ant_vec, x_deg, y_deg, z_deg, usage_multi, element_ind);
    }
    else // Single-frequency: usage 0/1 adjust the sampling grid.
    {
        auto &ant = ant_vec[0];
        if (element_ind.n_elem == 0)
            ant.rotate_pattern(x_deg, y_deg, z_deg, usage);
        else
            for (arma::uword el : element_ind)
                ant.rotate_pattern(x_deg, y_deg, z_deg, usage, (unsigned)el);
    }

    return qd_python_arrayant2dict_multi(ant_vec);
}

// pybind11 declaration (register under the `arrayant` submodule in python_main.cpp):
// m.def("rotate_pattern", &arrayant_rotate_pattern,
//       py::arg("arrayant"),
//       py::arg("x_deg") = 0.0,
//       py::arg("y_deg") = 0.0,
//       py::arg("z_deg") = 0.0,
//       py::arg("usage") = 0,
//       py::arg("element") = py::none());