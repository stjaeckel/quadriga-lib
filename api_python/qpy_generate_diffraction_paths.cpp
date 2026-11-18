// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# generate_diffraction_paths
Generate elliptic propagation paths and weights for diffraction gain estimation

- Generates inputs required by [[calc_diffraction_gain]]: elliptic-arc paths sampling the Fresnel ellipsoid volume
  between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so
  `sum(prod(weight, axis=2), axis=1) = 1`

## Usage:
```
ray_x, ray_y, ray_z, weight = quadriga_lib.RTtools.generate_diffraction_paths( orig, dest, \
    center_freq, lod, *, scalar_mode )
```

## Inputs:
- **`orig`** — TX positions; `(n_pos, 3)`
- **`dest`** — RX positions; `(n_pos, 3)`
- **`center_freq`** — Center frequency in Hz; in scalar mode the acoustic frequency
- **`lod`** — Level of detail; controls `n_path` and `n_seg`:<br><br>
   | `lod` | `n_path` | `n_seg` | Note  |
   | ----- | -------- | ------- | ----- |
   | 1     | 7        | 3       | -     |
   | 2     | 19       | 3       | -     |
   | 3     | 37       | 4       | -     |
   | 4     | 61       | 5       | -     |
   | 5     | 1        | 2       | debug |
   | 6     | 2        | 2       | debug |
   | 7     | 1        | 10      | direct line in 10 segments |

- **`scalar_mode`** — `True` = acoustic: the Fresnel ellipsoids use the wavelength `343 / center_freq`;
  default: `False` (EM, `299792458 / center_freq`)

## Outputs:
- **`ray_x`** — x-coordinates of path waypoints (excluding endpoints); `(n_pos, n_path, n_seg-1)`
- **`ray_y`** — y-coordinates of path waypoints (excluding endpoints); `(n_pos, n_path, n_seg-1)`
- **`ray_z`** — z-coordinates of path waypoints (excluding endpoints); `(n_pos, n_path, n_seg-1)`
- **`weight`** — Per-segment weights; `(n_pos, n_path, n_seg)`

## See also:
- [[calc_diffraction_gain]] (consumes the output of this function)
MD!*/

py::tuple generate_diffraction_paths(const py::array_t<double> &orig,
                                     const py::array_t<double> &dest,
                                     double center_freq,
                                     int lod,
                                     py::handle scalar_mode)
{
    const auto orig_a = qd_python_numpy2arma_Mat(orig, true);
    const auto dest_a = qd_python_numpy2arma_Mat(dest, true);
    const bool scalar_mode_b = qd_python_flag(scalar_mode, false, "scalar_mode");

    // Output size follows from the lod table in C++; copied after the call
    arma::cube ray_x, ray_y, ray_z, weight;
    quadriga_lib::generate_diffraction_paths<double>(orig_a, dest_a, center_freq, lod,
                                                     ray_x, ray_y, ray_z, weight, scalar_mode_b);

    return py::make_tuple(qd_python_copy2numpy(&ray_x), qd_python_copy2numpy(&ray_y),
                          qd_python_copy2numpy(&ray_z), qd_python_copy2numpy(&weight));
}
