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
# qdant_read
Reads array antenna data from QDANT files

## Description:
- The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna pattern
  data in XML. This function reads pattern data from the specified file.
- Supports both single-entry and multi-entry (multi-frequency) reading:
- **Single-entry (id &ge; 1):** Reads one antenna entry from the file. Pattern fields are returned as
  3D arrays `(n_el, n_az, n_elem)`. This is the default behavior.
- **Multi-entry (id = 0):** Reads all antenna entries from the file and returns them as a single
  multi-frequency arrayant dict. Pattern fields are returned as 4D arrays
  `(n_el, n_az, n_elem, n_freq)`, center frequencies as a 1D array, and coupling matrices as 3D
  arrays if they vary across entries. This is the inverse of `qdant_write` with 4D input.

## Usage:
```
from quadriga_lib import arrayant

# Read a single antenna entry (default: first entry)
data = arrayant.qdant_read('antenna.qdant')
data = arrayant.qdant_read('antenna.qdant', id=2)

# Read all entries as multi-frequency arrayant (4D patterns)
data = arrayant.qdant_read('speaker.qdant', id=0)
```

## Input Arguments:
- **`fn`**<br>
  Filename of the QDANT file, string

- **`id`** (optional)<br>
  ID of the antenna to be read from the file. Default: `1` (read first entry).
  Set to `0` to read all entries as a multi-frequency arrayant.

## Output Arguments:
- **`data`**<br>
  Dictionary containing the data in the QDANT file with the following keys:

  For single-entry (id &ge; 1):
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_el, n_az, n_elem)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_el, n_az, n_elem)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_el, n_az, n_elem)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_el, n_az, n_elem)`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions                     | Shape: `(3, n_elements)`
  `coupling_re`    | Coupling matrix, real part                            | Shape: `(n_elements, n_ports)`
  `coupling_im`    | Coupling matrix, imaginary part                       | Shape: `(n_elements, n_ports)`
  `center_freq`    | Center frequency in [Hz]                              | Scalar
  `name`           | Name of the array antenna object                      | String
  `layout`         | Layout of multiple array antennas                     | Matrix

  For multi-entry (id = 0):
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_el, n_az, n_elem, n_freq)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_el, n_az, n_elem, n_freq)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_el, n_az, n_elem, n_freq)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_el, n_az, n_elem, n_freq)`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions                     | Shape: `(3, n_elements)`
  `coupling_re`    | Coupling matrix, real part                            | Shape: `(n_elem, n_ports)` or `(n_elem, n_ports, n_freq)`
  `coupling_im`    | Coupling matrix, imaginary part                       | Shape: `(n_elem, n_ports)` or `(n_elem, n_ports, n_freq)`
  `center_freq`    | Center frequencies in [Hz]                            | Shape: `(n_freq)`
  `name`           | Name of the array antenna object                      | String
  `layout`         | Layout of multiple array antennas                     | Matrix

## See also:
- [[qdant_write]] (for writing QDANT data)
- QuaDRiGa Array Antenna Exchange Format  (<a href="formats.html#6cab4884">QDANT</a>)
MD!*/

py::dict arrayant_qdant_read(const std::string fn, unsigned id)
{
    arma::Mat<unsigned> layout;

    if (id == 0) // Multi-entry: read all entries as multi-frequency arrayant
    {
        auto ant_vec = quadriga_lib::qdant_read_multi<double>(fn, &layout);

        auto output = qd_python_arrayant2dict_multi(ant_vec);
        output["layout"] = qd_python_copy2numpy(layout);
        return output;
    }
    else // Single-entry: read one antenna by ID
    {
        auto arrayant = quadriga_lib::qdant_read<double>(fn, id, &layout);

        auto output = qd_python_arrayant2dict(arrayant);
        output["layout"] = qd_python_copy2numpy(layout);
        return output;
    }
}