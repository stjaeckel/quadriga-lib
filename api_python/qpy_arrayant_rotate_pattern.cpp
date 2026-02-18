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

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# rotate_pattern
Rotates antenna patterns

## Description:
- This function transforms the radiation patterns of array antenna elements, allowing for
  precise rotations around the three principal axes (x, y, z) of the local Cartesian coordinate system.
  The 3 rotations are applied in the order: 1. rotation around the x-axis (bank angle);
  2. rotation around the y-axis (tilt angle), 3. rotation around the z-axis (heading angle).
- Supports both single-frequency arrayants (3D pattern fields) and multi-frequency arrayants
  (4D pattern fields). For multi-frequency inputs, the rotation is applied consistently across all
  frequency entries. The function auto-detects the format by inspecting the dimensionality of
  `e_theta_re` (3D = single, 4D = multi).
- **Note on usage modes for multi-frequency:** Grid adjustment (usage 0 and 1) is not supported for
  multi-frequency arrayants because all frequency entries must share the same angular grid. The multi-
  frequency path automatically maps usage 0 → 3 (pattern + polarization, no grid adjust) and
  usage 1 → 4 (pattern only, no grid adjust). Usage 2 (polarization only) works identically in both
  paths.

## Usage:

```
from quadriga_lib import arrayant

# Single-frequency: rotate all elements by 45 deg bank
arrayant_out = arrayant.rotate_pattern(ant, x_deg=45.0)

# Single-frequency: rotate only elements 0 and 1 (0-based)
arrayant_out = arrayant.rotate_pattern(ant, z_deg=90.0, element=[0, 1])

# Multi-frequency (4D patterns): same interface
speaker_out = arrayant.rotate_pattern(speaker, y_deg=10.0)
```

## Input Arguments:
- **`arrayant`**<br>
  Dictionary containing the arrayant data. Pattern fields may be 3D (single-frequency) or
  4D (multi-frequency, 4th dimension = frequency). The following keys are expected:
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_el, n_az, n_elem)` or `(n_el, n_az, n_elem, n_freq)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_el, n_az, n_elem)` or `(n_el, n_az, n_elem, n_freq)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_el, n_az, n_elem)` or `(n_el, n_az, n_elem, n_freq)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_el, n_az, n_elem)` or `(n_el, n_az, n_elem, n_freq)`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Shape: `(3, n_elements)`
  `coupling_re`    | Coupling matrix, real part, optional                  | Shape: `(n_elem, n_ports)` or `(n_elem, n_ports, n_freq)`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Shape: `(n_elem, n_ports)` or `(n_elem, n_ports, n_freq)`
  `center_freq`    | Center frequency in [Hz], optional                    | Scalar or 1D array `(n_freq)`
  `name`           | Name of the array antenna object, optional            | String

- **`x_deg`** (optional)<br>
  The rotation angle around x-axis (bank angle) in [degrees]. Default: `0.0`

- **`y_deg`** (optional)<br>
  The rotation angle around y-axis (tilt angle) in [degrees]. Default: `0.0`

- **`z_deg`** (optional)<br>
  The rotation angle around z-axis (heading angle) in [degrees]. Default: `0.0`

- **`usage`** (optional)<br>
  The optional parameter 'usage' can limit the rotation procedure either to the pattern or polarization.
  `usage = 0` | Rotate both, pattern and polarization, adjusts sampling grid (default; multi-freq: no grid adjust)
  `usage = 1` | Rotate only pattern, adjusts sampling grid (multi-freq: no grid adjust)
  `usage = 2` | Rotate only polarization
  `usage = 3` | Rotate both, but do not adjust the sampling grid
  `usage = 4` | Rotate only pattern, do not adjust the sampling grid

- **`element`** (optional)<br>
  The element indices for which the pattern should be transformed. Optional parameter. Values must
  be between 0 and n_elements-1 (0-based). If this parameter is not provided (or an empty array is
  passed), all elements will be rotated by the same angles. Shape: `(n_elements)` or empty `()`

## Output Arguments:
- **`arrayant_out`**<br>
  Dictionary containing the arrayant data with the rotated patterns. Output format matches the
  input format (3D for single-frequency, 4D for multi-frequency).
MD!*/

py::dict arrayant_rotate_pattern(const py::dict &arrayant,
                                 double x_deg,
                                 double y_deg,
                                 double z_deg,
                                 unsigned usage,
                                 const py::array_t<unsigned> &element)
{
    const auto element_ind = qd_python_numpy2arma_Col(element, true);

    // Detect single vs multi-frequency from e_theta_re dimensionality
    py::array e_theta_re_arr = py::cast<py::array>(arrayant["e_theta_re"]);
    int nd = (int)e_theta_re_arr.request().ndim;

    if (nd == 4) // Multi-frequency path
    {
        auto ant_vec = qd_python_dict2arrayant_multi(arrayant, false);

        // Validate multi-frequency consistency
        std::string err = quadriga_lib::arrayant_is_valid_multi(ant_vec, true);
        if (!err.empty())
            throw std::invalid_argument(err);

        // Element indices are 0-based in Python; arrayant_rotate_pattern_multi also uses 0-based
        arma::uvec i_element;
        if (element_ind.n_elem > 0)
        {
            i_element.set_size(element_ind.n_elem);
            for (arma::uword i = 0; i < element_ind.n_elem; ++i)
                i_element[i] = (arma::uword)(element_ind[i]);
        }

        usage = (usage == 3) ? 0 : usage;
        usage = (usage == 4) ? 1 : usage;

        quadriga_lib::arrayant_rotate_pattern_multi(ant_vec, x_deg, y_deg, z_deg, usage, i_element);

        return qd_python_arrayant2dict_multi(ant_vec);
    }
    else // Single-frequency path
    {
        auto ant = qd_python_dict2arrayant(arrayant, false); // Copy

        if (element_ind.n_elem == 0)
            ant.rotate_pattern(x_deg, y_deg, z_deg, usage);
        else
            for (auto el : element_ind)
                ant.rotate_pattern(x_deg, y_deg, z_deg, usage, el);

        return qd_python_arrayant2dict(ant);
    }
}
