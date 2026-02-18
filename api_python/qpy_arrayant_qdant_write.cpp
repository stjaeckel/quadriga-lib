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
# qdant_write
Writes array antenna data to QDANT files

## Description:
- The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna pattern
  data in XML. This function writes pattern data to the specified file.
- Supports both single-frequency arrayants (3D pattern fields) and multi-frequency arrayants
  (4D pattern fields). The function auto-detects the format by inspecting the dimensionality of
  `e_theta_re` (3D = single, 4D = multi).
- **Single-frequency:** Writes one antenna entry to the file. The `id` parameter controls where the
  entry is placed. Multiple antennas can be stored in the same file by calling this function
  repeatedly with different IDs.
- **Multi-frequency:** Writes all frequency entries as sequential IDs (1-based) to the file. The file
  is overwritten if it already exists. A layout matrix is created automatically. The `id` and `layout`
  parameters are ignored for multi-frequency inputs.

## Usage:

```
from quadriga_lib import arrayant

# Single-frequency: write with optional ID
id_in_file = arrayant.qdant_write('antenna.qdant', ant)
id_in_file = arrayant.qdant_write('antenna.qdant', ant, id=2)

# Multi-frequency (4D patterns): writes all frequencies sequentially
arrayant.qdant_write('speaker.qdant', speaker)
```

## Input Arguments:
- **`fn`** [1]<br>
  Filename of the QDANT file, string

- **`arrayant`** [2] (optional)<br>
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

- **`id`** [3] (optional, single-frequency only)<br>
  ID of the antenna to be written to the file, optional, Default: Max-ID in existing file + 1.
  Ignored for multi-frequency inputs.

- **`layout`** [4] (optional, single-frequency only)<br>
  Layout of multiple array antennas. Must only contain element ids that are present in the file.
  Ignored for multi-frequency inputs.

## Output Argument:
- **`id_in_file`**<br>
  For single-frequency: ID of the antenna in the file after writing.
  For multi-frequency: always returns 0 (all entries are written sequentially starting at ID 1).

# See also:
- [[qdant_read]] (for reading QDANT data)
- QuaDRiGa Array Antenna Exchange Format  (<a href="formats.html#6cab4884">QDANT</a>)
MD!*/

ssize_t arrayant_qdant_write(const std::string &fn,
                             const py::dict &arrayant,
                             const unsigned id,
                             const py::array_t<unsigned> &layout)
{
    // Detect single vs multi-frequency from e_theta_re dimensionality
    py::array e_theta_re_arr = py::cast<py::array>(arrayant["e_theta_re"]);
    int nd = (int)e_theta_re_arr.request().ndim;

    if (nd == 4) // Multi-frequency path
    {
        auto ant_vec = qd_python_dict2arrayant_multi(arrayant, false);
        quadriga_lib::qdant_write_multi(fn, ant_vec);
        return 0;
    }
    else // Single-frequency path
    {
        const auto ant = qd_python_dict2arrayant(arrayant, true);
        const auto layout_a = qd_python_numpy2arma_Mat(layout, true);
        return (ssize_t)ant.qdant_write(fn, id, layout_a);
    }
}