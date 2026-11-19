// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# channel.export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g. for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min_dB` are excluded; `max_paths` caps the total number of exported paths
- Paths with non-finite coordinates are skipped

## Usage:
```
quadriga_lib.channel.export_obj_file( fn, chan, *, max_paths, gain_max_dB, gain_min_dB, colormap, i_snap, \
    radius_max, radius_min, n_edges )
```

## Inputs:
- **`fn`** — Output `.obj` file path
- **`chan`** — Channel; dict with the keys of [[hdf5_read_channel]], list or stacked layout; needs `rx_pos`,
  `tx_pos`, `center_freq`, `n_interact`, `interact_coord` and `coeff` with `delay`, or `polmat`; in the stacked
  layout, columns of `interact_coord` beyond `sum(n_interact)` are ignored
- **`max_paths`** — Max paths to export; 0 includes all paths above `gain_min_dB`; default: 0
- **`gain_max_dB`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min_dB`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`i_snap`** — Snapshot indices to include; 0-based; `None` or empty exports all; default: `None`
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5

## Outputs:
- None; the OBJ file is written directly to disk
MD!*/

void channel_export_obj_file(const py::handle &fn,
                             const py::handle &chan,
                             size_t max_paths,
                             double gain_max_dB,
                             double gain_min_dB,
                             const std::string &colormap,
                             const py::handle &i_snap,
                             double radius_max,
                             double radius_min,
                             size_t n_edges)
{
    const std::string fn_s = qd_python_path(fn, "fn");
    if (!py::isinstance<py::dict>(chan))
        throw std::invalid_argument("Input 'chan' must be a dict.");
    const auto c = qd_python_dict2channel(py::reinterpret_borrow<py::dict>(chan)); // C++ validates
    const arma::uvec i_snap_a = qd_python_numpy2arma_Col<arma::uword>(i_snap, true, false, "i_snap");

    quadriga_lib::channel_export_obj_file(fn_s, c, max_paths, gain_max_dB, gain_min_dB, colormap, i_snap_a,
                                          radius_max, radius_min, n_edges);
}
