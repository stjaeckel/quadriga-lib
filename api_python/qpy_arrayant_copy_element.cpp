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
# copy_element
Create copies of array antenna elements

## Description:
Copies one or more antenna elements to new positions within the arrayant, expanding the element
count if necessary. Supports both single-frequency arrayants (3D pattern fields) and multi-frequency
arrayants (4D pattern fields). For multi-frequency inputs, element copying is applied consistently
across all frequency entries.

The function auto-detects whether the input is single-frequency or multi-frequency by inspecting
the dimensionality of the `e_theta_re` field (3D = single, 4D = multi).

## Usage:

```
from quadriga_lib import arrayant

# Single-frequency: copy element 0 to position 1
arrayant_out = arrayant.copy_element(ant, source_element=[0], dest_element=[1])

# Single-frequency: copy element 0 to positions 2 and 3
arrayant_out = arrayant.copy_element(ant, source_element=[0], dest_element=[2, 3])

# Multi-frequency (4D patterns): same interface
speaker_out = arrayant.copy_element(speaker, source_element=[0], dest_element=[1])

# Copy multiple sources to multiple destinations (must be same length)
arrayant_out = arrayant.copy_element(ant, source_element=[0, 1], dest_element=[2, 3])
```

## Input Arguments:
- **`arrayant`** [1] (required)<br>
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

- **`source_element`** [2] (required)<br>
  Index of the source elements (0-based), scalar or vector

- **`dest_element`** [3] (optional)<br>
  Index of the destination elements (0-based), either as a vector or as a scalar. If `source_element`
  is also a vector, `dest_element` must have the same length.

## Output Arguments:
- **`arrayant_out`**<br>
  Dictionary containing the arrayant data with the copied elements. Output format matches the
  input format (3D for single-frequency, 4D for multi-frequency).
MD!*/

py::dict arrayant_copy_element(const py::dict &arrayant,
                               const py::array_t<arma::uword> &source_element,
                               const py::array_t<arma::uword> &dest_element)
{
    const auto source = qd_python_numpy2arma_Col(source_element);
    const auto dest = qd_python_numpy2arma_Col(dest_element);

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

        if (source.n_elem == 1)
            quadriga_lib::arrayant_copy_element_multi(ant_vec, source.at(0), dest);
        else if (source.n_elem == dest.n_elem)
        {
            for (arma::uword i = 0; i < source.n_elem; ++i)
                quadriga_lib::arrayant_copy_element_multi(ant_vec, source.at(i), arma::uvec{dest.at(i)});
        }
        else
            throw std::invalid_argument("Copy element: when copying multiple elements, source and dest must be of same length.");

        return qd_python_arrayant2dict_multi(ant_vec);
    }
    else // Single-frequency path
    {
        auto ant = qd_python_dict2arrayant(arrayant, false); // Copy

        if (source.n_elem == 1)
            ant.copy_element(source.at(0), dest);
        else if (source.n_elem == dest.n_elem)
        {
            for (arma::uword i = 0; i < source.n_elem; ++i)
                ant.copy_element(source.at(i), dest.at(i));
        }
        else
            throw std::invalid_argument("Copy element: when copying multiple elements, source and dest must be of same length.");

        return qd_python_arrayant2dict(ant);
    }
}
