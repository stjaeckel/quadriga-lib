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
# ARRAYANT_CALC_DIRECTIVITY
Calculates the directivity (in dBi) of array antenna elements

## Description:
Directivity is a parameter of an antenna or which measures the degree to which the radiation emitted
is concentrated in a single direction. It is the ratio of the radiation intensity in a given direction
from the antenna to the radiation intensity averaged over all directions. Therefore, the directivity
of a hypothetical isotropic radiator is 1, or 0 dBi.

## Usage:

```
directivity = quadriga_lib.arrayant_calc_directivity(arrayant, element);
```

## Input Arguments:
- **`arrayant_in`**<br>
  Dictionary containing the arrayant data with the following keys:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`

- **`element`** (optional)<br>
  Element index, 0-based. If not provided or empty, the directivity is calculated for all elements in the
  array antenna. Size: `[n_out]` or empty

## Output Argument:
- **`directivity`**<br>
  Directivity of the antenna pattern in dBi, Size: `[n_out]` or `[n_elements]`
MD!*/

py::array_t<double> arrayant_calc_directivity(const py::dict &arrayant,                // Input data
                                              const py::array_t<arma::uword> &element) // Antenna element indices, 0-based
{
    const auto ant = qd_python_dict2arrayant(arrayant, true);

    arma::uvec element_ind = (element.size() == 0) ? arma::regspace<arma::uvec>(0, ant.e_theta_re.n_slices - 1)
                                                   : qd_python_numpy2arma_Col(element);

    arma::vec directivity;
    auto output = qd_python_init_output<double>(element_ind.n_elem, &directivity);

    auto *p_directivity = directivity.memptr();
    for (auto el : element_ind)
        *p_directivity++ = ant.calc_directivity_dBi(el);

    return output;
}
