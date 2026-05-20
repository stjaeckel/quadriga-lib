// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# concat
Concatenate two array antennas into a single one

- Appends all elements of `arrayant2` onto `arrayant1` along the element dimension; the `element_pos`
  matrices are joined horizontally
- Both inputs must share identical azimuth and elevation sampling grids
- Coupling is assembled block-diagonally: `arrayant1`'s elements connect only to its own ports,
  `arrayant2`'s elements only to its own ports
- `center_freq` and `name` are taken from `arrayant1`
- Frequency-dependent input (4D pattern fields): both inputs must have the same number of
  frequency entries with matching `center_freq` at each index; concatenation is done per entry
- Output format matches the input: single-frequency dict for 3D input, frequency-dependent dict
  for 4D input

## Usage:
```
arrayant_out = quadriga_lib.arrayant.concat( arrayant1, arrayant2 )
```

## Inputs:
- **`arrayant1`** — Dict with the first array antenna; keys as in [[generate]]; pattern fields
  may be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`arrayant2`** — Dict with the second array antenna; must use the same azimuth and elevation
  grids as `arrayant1`, and for 4D input the same number of frequency entries with matching `center_freq`

## Outputs:
- **`arrayant_out`** — Dict with the concatenated array antenna; same keys and layout as the
  inputs; single-frequency dict for 3D input, frequency-dependent dict for 4D input
MD!*/

py::dict arrayant_concat(const py::dict &arrayant1,
                         const py::dict &arrayant2)
{
    // Parse both inputs via the unified multi-freq reader
    auto ant1 = qd_python_dict2arrayant_multi(arrayant1, false, false, true);
    auto ant2 = qd_python_dict2arrayant_multi(arrayant2, true, false, true);

    if (ant1.empty() || ant2.empty())
        throw std::invalid_argument("'arrayant1' and 'arrayant2' cannot be empty.");

    if (ant1.size() != ant2.size())
        throw std::invalid_argument("'arrayant1' and 'arrayant2' must have the same number of frequency entries.");

    // Frequency-dependent: concatenate per entry (requires matching center_freq at each index)
    if (ant1.size() > 1)
    {
        auto out = quadriga_lib::arrayant_concat_multi<double>(ant1, ant2);
        return qd_python_arrayant2dict_multi(out);
    }

    // Single-frequency: append the elements of in2 onto in1
    auto out = ant1[0].append(&ant2[0]);
    return qd_python_arrayant2dict(out);
}

// pybind11 declaration (register under the `arrayant` submodule in python_main.cpp):
// m.def("concat", &arrayant_concat,
//       py::arg("arrayant1"),
//       py::arg("arrayant2"));