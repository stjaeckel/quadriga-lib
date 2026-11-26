// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# EXPORT_CHANNEL_OBJ
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
quadriga_lib.export_channel_obj( fn, chan, max_paths, gain_max_dB, gain_min_dB, colormap, i_snap, ...
    radius_max, radius_min, n_edges );
```

## Inputs:
- **`fn`** — Path of the OBJ file, ending in `.obj`
- **`chan`** — Channel; 1x1 struct with the fields of [[hdf5_read_channel]]; needs `rx_pos`, `tx_pos`,
  `center_freq`, `n_interact`, `interact_coord` and `coeff_re` / `coeff_im` with `delay`, or `polmat`; columns of
  `interact_coord` beyond `sum(n_interact)` are ignored
- **`max_paths`** — Number of the strongest paths per snapshot to export; 0 = all; default: 0
- **`gain_max_dB`** — Gain in dB of the largest radius and the last color, finite; default: -60
- **`gain_min_dB`** — Gain in dB of the first color, finite and below `gain_max_dB`; weaker paths are left out;
  default: -140
- **`colormap`** — Name of the colormap: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer;
  default: `'jet'`
- **`i_snap`** — 1-based indices of the snapshots to export; default: `[]` (all snapshots)
- **`radius_max`** — Tube radius in m at `gain_max_dB`, >= 0; default: 0.05
- **`radius_min`** — Smallest tube radius in m, >= 0; at most `radius_max` (a larger value is reduced to it);
  default: 0.01
- **`n_edges`** — Vertices of the tube cross-section, >= 3; default: 5
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs; // No outputs

    if (nrhs < 2 || nrhs > 10)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto fn = qd_mex_get_string(prhs[0], "", "fn");

    qd_mex_check_struct(prhs[1], "chan", true);
    const auto chan = qd_mex_struct2channel(prhs[1], false, false, "chan"); // C++ validates

    const auto max_paths = (nrhs < 3) ? arma::uword(0) : qd_mex_get_scalar<arma::uword>(prhs[2], "max_paths", 0);
    const auto gain_max_dB = (nrhs < 4) ? -60.0 : qd_mex_get_scalar<double>(prhs[3], "gain_max_dB", -60.0);
    const auto gain_min_dB = (nrhs < 5) ? -140.0 : qd_mex_get_scalar<double>(prhs[4], "gain_min_dB", -140.0);
    const auto colormap = (nrhs < 6) ? std::string("jet") : qd_mex_get_string(prhs[5], "jet", "colormap");
    const arma::uvec i_snap = (nrhs < 7) ? arma::uvec() : qd_mex_get_index_Col(prhs[6], "i_snap"); // 1-based -> 0-based
    const auto radius_max = (nrhs < 8) ? 0.05 : qd_mex_get_scalar<double>(prhs[7], "radius_max", 0.05);
    const auto radius_min = (nrhs < 9) ? 0.01 : qd_mex_get_scalar<double>(prhs[8], "radius_min", 0.01);
    const auto n_edges = (nrhs < 10) ? arma::uword(5) : qd_mex_get_scalar<arma::uword>(prhs[9], "n_edges", 5);

    CALL_QD(quadriga_lib::export_channel_obj(fn, chan[0], max_paths, gain_max_dB, gain_min_dB, colormap,
                                             i_snap, radius_max, radius_min, n_edges));
}
