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
The QuaDRiGa array antenna exchange format (QDANT) is a file format used to store antenna pattern
data in XML. This function reads pattern data from the specified file.

## Usage:

```
from quadriga_lib import arrayant
data = arrayant.qdant_read( fn, id )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the QDANT file, string

- **`id`** (optional)<br>
  ID of the antenna to be read from the file, optional, Default: Read first

## Output Arguments:
- **`data`**<br>
  Dictionary containing the data in the QDANT file with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_elevation, n_azimuth, n_elements)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_elevation, n_azimuth, n_elements)`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted             | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Shape: `(3, n_elements)`
  `coupling_re`    | Coupling matrix, real part                            | Shape: `(n_elements, n_ports)`
  `coupling_im`    | Coupling matrix, imaginary part                       | Shape: `(n_elements, n_ports)`
  `center_freq`    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
  `name`           | Name of the array antenna object                      | String
  `layout`         | Layout of multiple array antennas.                    | Matrix

## See also:
- [[qdant_write]] (for writing QDANT data)
- QuaDRiGa Array Antenna Exchange Format  (<a href="formats.html#6cab4884">QDANT</a>)
MD!*/

py::dict arrayant_qdant_read(const std::string fn, unsigned id)
{
    arma::Mat<unsigned> layout;
    auto arrayant = quadriga_lib::qdant_read<double>(fn, id, &layout);

    auto output = qd_python_arrayant2dict(arrayant);
    output["layout"] = qd_python_copy2numpy(layout);
    return output;
}