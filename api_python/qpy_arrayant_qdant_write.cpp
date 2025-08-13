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
The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna pattern
data in XML. This function writes pattern data to the specified file.

## Usage:

```
from quadriga_lib import arrayant
id_in_file = arrayant.qdant_write( fn, arrayant, id, layout);
```

## Caveat:
- Multiple array antennas can be stored in the same file using the `id` parameter.
- If writing to an exisiting file without specifying an `id`, the data gests appended at the end.
  The output `id_in_file` identifies the location inside the file.
- An optional storage `layout` can be provided to organize data inside the file.

## Input Arguments:
- **`fn`** [1]<br>
  Filename of the QDANT file, string

- **`arrayant`** [2] (optional)<br>
  Dictionary containing the arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_elevation, n_azimuth, n_elements)`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Shape: `(3, n_elements)`
  `coupling_re`    | Coupling matrix, real part, optional                  | Shape: `(n_elements, n_ports)`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Shape: `(n_elements, n_ports)`
  `center_freq`    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
  `name`           | Name of the array antenna object, optional            | String

- **`id`** [3] (optional)<br>
  ID of the antenna to be written to the file, optional, Default: Max-ID in existing file + 1

- **`layout`** [4] (optional)<br>
  Layout of multiple array antennas. Must only contain element ids that are present in the file. optional

## Output Argument:
- **`id_in_file`**<br>
  ID of the antenna in the file after writing

# See also:
- [[qdant_read]] (for reading QDANT data)
- QuaDRiGa Array Antenna Exchange Format  (<a href="formats.html#6cab4884">QDANT</a>)
MD!*/

ssize_t arrayant_qdant_write(const std::string &fn,
                             const py::dict &arrayant,
                             const unsigned id,
                             const py::array_t<unsigned> &layout)
{
    const auto ant = qd_python_dict2arrayant(arrayant, true);
    const auto layout_a = qd_python_numpy2arma_Mat(layout, true);
    return (ssize_t)ant.qdant_write(fn, id, layout_a);
}