// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# rotate_pattern
Rotate antenna radiation patterns using Euler rotations

- Rotates the pattern and/or polarization by the orientation (bank, tilt, heading); the rotated pattern equals what
  [[interpolate]] returns for the unrotated pattern with the same `orientation`
- A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
- Element positions are rotated with the pattern (usage 0, 1, 3 and 4); coupling, `center_freq` and `name` are kept
- Single-frequency input (3D pattern fields): `usage` 0 and 1 adjust the sampling grid for non-uniformly sampled
  antennas; this needs all elements, so a subset in `i_element` raises an error when the grid must be adjusted
- Frequency-dependent input (4D pattern fields): the rotation is applied to every entry and the grid is never adjusted,
  since all entries must share one grid
- For scalar acoustic fields (pressure stored in `e_theta_re` only) use `usage = 1` to avoid spurious polarization effects

## Usage:
```
# 10 degree mechanical downtilt (0.1745 rad) of all elements
arrayant_out = quadriga_lib.arrayant.rotate_pattern(arrayant, [0.0, 0.1745, 0.0])

# Elements 0 and 1 only, 90 degree heading (1.5708 rad)
arrayant_out = quadriga_lib.arrayant.rotate_pattern(arrayant, [0.0, 0.0, 1.5708], i_element=[0, 1])
```

## Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as in [[generate]]; pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`orientation`** — Euler angles (bank, tilt, heading) in rad; 3 elements
- **`usage`** — Rotation mode; default: 0<br><br>
   | Mode | Pattern | Polarization | Grid adj. |
   | :--: | :-----: | :----------: | :-------: |
   | 0    | Yes     | Yes          | Yes       |
   | 1    | Yes     | No           | Yes       |
   | 2    | No      | Yes          | No        |
   | 3    | Yes     | Yes          | No        |
   | 4    | Yes     | No           | No        |

   for 4D input the grid is never adjusted, so `0`/`3` and `1`/`4` are equivalent
- **`i_element`** — Element indices to rotate, 0-based, no duplicates; `None` or empty rotates all elements;
  default: `None`

## Outputs:
- **`arrayant_out`** — Dict with the rotated array antenna data; same keys and layout as
  `arrayant`; single-frequency dict for 3D input, frequency-dependent dict for 4D input
MD!*/

py::dict arrayant_rotate_pattern(const py::dict &arrayant,
                                 const py::handle &orientation,
                                 unsigned usage,
                                 const py::handle &i_element)
{
    const arma::vec ori = qd_python_numpy2arma_Col<double>(orientation, true);
    if (ori.n_elem != 3)
        throw std::invalid_argument("Input 'orientation' must have 3 elements.");

    // Element indices (0-based); empty = rotate all elements
    const arma::uvec i_el = qd_python_numpy2arma_Col<arma::uword>(i_element, true);

    // Parse the (possibly frequency-dependent) antenna (copy, the user's arrays are not changed)
    auto ant_vec = qd_python_dict2arrayant_multi(arrayant, false, false, false, false);

    if (ant_vec.size() > 1) // Frequency-dependent: rotate every entry
    {
        const unsigned usage_multi = (usage == 3) ? 0 : ((usage == 4) ? 1 : usage);
        quadriga_lib::arrayant_rotate_pattern_multi(ant_vec, ori, usage_multi, &i_el);
    }
    else // Single-frequency: usage 0/1 adjust the sampling grid
        quadriga_lib::arrayant_rotate_pattern(ant_vec[0], ori, usage, &i_el);

    return qd_python_arrayant2dict_multi(ant_vec);
}

// pybind11 declaration (register under the `arrayant` submodule in python_main.cpp):
// m.def("rotate_pattern", &arrayant_rotate_pattern,
//       py::arg("arrayant"),
//       py::arg("orientation"),
//       py::arg("usage") = 0,
//       py::arg("i_element") = py::none());
