// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# copy_element
Create copies of array antenna elements

- Copies a source element to one or more destination slots within an arrayant
- The antenna is resized when a destination index exceeds the current number of elements
- Coupling-matrix entries for newly added elements are set to identity; existing coupling is kept
- Works on single-frequency dicts (3D pattern fields) and frequency-dependent dicts (4D pattern
  fields); the same copy is applied to every frequency and a matching dict is returned
- If `source_element` has one entry, that element is copied to every index in `dest_element`
- If `source_element` has several entries, `dest_element` must have the same length and copies
  are performed pairwise as `source_element[i]` to `dest_element[i]`

## Usage:
```
# Copy element 0 to position 1
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, 0, 1 )

# Copy element 0 to several positions
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, 0, [2, 3] )

# Pairwise copy — source_element and dest_element must have equal length
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, [0, 1], [2, 3] )
```

## Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as in [[generate]]; pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`source_element`** — Index of the source element(s); scalar int or 1D list or array of int; `(n_copy,)`
- **`dest_element`** — Index of the destination element(s); scalar int  1D list or array of int; `(n_copy,)`;
  must have the same length as `source_element` unless `source_element` is a single index

## Outputs:
- **`arrayant_out`** — Dict with the modified array antenna data; same keys and layout as
  `arrayant`; single-frequency dict for 3D input, frequency-dependent dict for 4D input
MD!*/

py::dict arrayant_copy_element(const py::dict &arrayant,
                               const py::handle &source_element,
                               const py::handle &dest_element)
{
    // Index vectors (0-based)
    const auto source = qd_python_numpy2arma_Col<arma::uword>(source_element, true);
    const auto dest = qd_python_numpy2arma_Col<arma::uword>(dest_element, true);

    if (source.n_elem == 0 || dest.n_elem == 0)
        throw std::invalid_argument("'source_element' and 'dest_element' cannot be empty.");

    if (source.n_elem > 1 && source.n_elem != dest.n_elem)
        throw std::invalid_argument("When copying multiple elements, 'source_element' and 'dest_element' must have the same length.");

    // Parse the (possibly frequency-dependent) input antenna (copy)
    auto ant = qd_python_dict2arrayant_multi(arrayant, false, false, true);

    // Apply the copy to every frequency entry
    if (source.n_elem == 1)
        quadriga_lib::arrayant_copy_element_multi(ant, source.at(0), dest);
    else
        for (arma::uword i = 0; i < source.n_elem; ++i)
            quadriga_lib::arrayant_copy_element_multi(ant, source.at(i), dest.at(i));

    return qd_python_arrayant2dict_multi(ant);
}
