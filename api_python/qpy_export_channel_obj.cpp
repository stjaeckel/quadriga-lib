// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# export_channel_obj
Export propagation paths to a Wavefront OBJ file for 3D visualization

- Writes the paths of ray-traced channels as tubes to an OBJ file and their colors to an MTL file of the same name
  (`.mtl` instead of `.obj`), e.g. for Blender; existing files are overwritten
- Path gain in dB: with coefficients the power summed over all antenna pairs, else from `polmat` (sum of the squared
  entries) with the free-space gain of the length of the drawn path
- The gain, clipped to `[gain_min_dB, gain_max_dB]`, sets the color (64 steps of `colormap`) and the radius (linear
  from 0 at `gain_min_dB` to `radius_max` at `gain_max_dB`, at least `radius_min`); a path with a radius up to
  0.1 mm is written as a polyline
- Paths below `gain_min_dB` and paths with non-finite coordinates are left out

## Usage:
```
quadriga_lib.channel.export_channel_obj( fn, chan, *, max_paths, gain_max_dB, gain_min_dB, colormap, \
    i_snap, radius_max, radius_min, n_edges )
```

## Inputs:
- **`fn`** — Path of the OBJ file, ending in `.obj`; str or path-like
- **`chan`** — Channel; dict with the keys of [[hdf5_read_channel]], list or stacked layout; needs `rx_pos`,
  `tx_pos`, `center_freq`, `n_interact`, `interact_coord` and `coeff` with `delay`, or `polmat`; in the stacked
  layout, columns of `interact_coord` beyond `sum(n_interact)` are ignored
- **`max_paths`** — Number of the strongest paths per snapshot to export; 0 = all; default: 0
- **`gain_max_dB`** — Gain in dB of the largest radius and the last color, finite; default: -60.0
- **`gain_min_dB`** — Gain in dB of the first color, finite and below `gain_max_dB`; weaker paths are left out;
  default: -140.0
- **`colormap`** — Name of the colormap: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer;
  default: `"jet"`
- **`i_snap`** — 0-based indices of the snapshots to export; `(n,)`; default: `None` (all snapshots)
- **`radius_max`** — Tube radius in m at `gain_max_dB`, >= 0; default: 0.05
- **`radius_min`** — Smallest tube radius in m, >= 0; at most `radius_max` (a larger value is reduced to it);
  default: 0.01
- **`n_edges`** — Vertices of the tube cross-section, >= 3; default: 5
MD!*/

void export_channel_obj(const py::handle &fn,
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

    quadriga_lib::export_channel_obj(fn_s, c, max_paths, gain_max_dB, gain_min_dB, colormap, i_snap_a,
                                     radius_max, radius_min, n_edges);
}
