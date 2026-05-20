// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# combine_pattern
Combine element patterns, positions, and coupling weights into effective radiation patterns

- Integrates the element field patterns, element positions, and coupling weights into one effective
  pattern per port (column of the coupling matrix)
- The result behaves as a virtual array with one element per port, zeroed element positions, and an
  identity coupling matrix
- Speeds up MIMO channel computation; useful for beamforming in 5G systems and network planning
- Accepts a frequency-dependent antenna (4D pattern fields); the pattern is then combined per frequency
- `freq` may recompute (interpolate) the combined pattern at one or more requested frequencies

## Usage:
```
# Single frequency
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant )
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, freq, azimuth_grid, elevation_grid )

# Multiple frequencies (freq as a 1D array, or a frequency-dependent input antenna)
arrayant_out = quadriga_lib.arrayant.combine_pattern( arrayant, freq, azimuth_grid, elevation_grid )
```

## Inputs:
- **`arrayant`** — Dict with the array antenna data; keys as documented in [[generate]]
- **`freq`** — Alternative center frequency in Hz; scalar or 1D array; if given, the pattern is
  recomputed at each value; if `None`, each input entry's `center_freq` is used; a scalar `<= 0`
  is treated as not given; default: `None`
- **`azimuth_grid`** — Alternative output azimuth grid in rad; -pi to pi; sorted; `(n_azimuth_out,)`;
  if `None`, the input grid is used; default: `None`
- **`elevation_grid`** — Alternative output elevation grid in rad; -pi/2 to pi/2; sorted;
  `(n_elevation_out,)`; if `None`, the input grid is used; default: `None`

## Outputs:
- **`arrayant_out`** — Dict with the combined array antenna data; keys as in [[generate]]

## See also:
- [[generate]] (for field layout in the arrayant struct)
- [[rotate_pattern]] (for changing the orientation of elements before combining)
- [[calc_beamwidth]] (calculates the beam width of array antennas)
- [[calc_directivity]] (directivity in dBi of array antenna elements)
MD!*/

py::dict arrayant_combine_pattern(const py::dict &arrayant,
                                  py::handle freq,
                                  py::handle azimuth_grid,
                                  py::handle elevation_grid)
{
    // Optional output grids (empty Col = keep the input grid)
    const auto azimuth_grid_a = qd_python_numpy2arma_Col<double>(azimuth_grid, true);
    const auto elevation_grid_a = qd_python_numpy2arma_Col<double>(elevation_grid, true);

    // Optional frequency override: accepts a scalar or a 1D array.
    arma::vec freq_a = qd_python_numpy2arma_Col<double>(freq, true);
    if (freq_a.n_elem == 1 && freq_a[0] <= 0.0)
        freq_a.reset();

    // Parse the (possibly frequency-dependent) input antenna; length 1 for a single-freq dict.
    auto ant = qd_python_dict2arrayant_multi(arrayant, true, false, true);

    // Multi-frequency branch: frequency-dependent input, or a freq vector with > 1 entry
    if (ant.size() > 1 || freq_a.n_elem > 1)
    {
        auto out = quadriga_lib::arrayant_combine_pattern_multi(ant, &azimuth_grid_a, &elevation_grid_a, &freq_a);
        return qd_python_arrayant2dict_multi(out);
    }

    // Single-frequency branch
    if (freq_a.n_elem == 1)
        ant[0].center_frequency = freq_a[0];

    auto arrayant_out = ant[0].combine_pattern(&azimuth_grid_a, &elevation_grid_a);
    return qd_python_arrayant2dict(arrayant_out);
}

// pybind11 declaration (register under the `arrayant` submodule in python_main.cpp):
// m.def("combine_pattern", &arrayant_combine_pattern,
//       py::arg("arrayant"),
//       py::arg("freq") = py::none(),
//       py::arg("azimuth_grid") = py::none(),
//       py::arg("elevation_grid") = py::none());