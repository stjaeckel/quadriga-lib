// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# CHANNEL_EXPORT_OBJ_FILE
Export propagation paths to a Wavefront OBJ file for 3D visualization

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min_dB` are excluded; `max_paths` limits the total number of exported paths
- Paths with non-finite coordinates are skipped

## Usage:
```
quadriga_lib.channel_export_obj_file( fn, chan, max_paths, gain_max_dB, gain_min_dB, colormap, i_snap, ...
    radius_max, radius_min, n_edges );
```

## Inputs:
- **`fn`** — Output `.obj` file path
- **`chan`** — Channel; 1x1 struct with the fields of [[hdf5_read_channel]]; needs `rx_pos`, `tx_pos`,
  `center_freq`, `n_interact`, `interact_coord` and `coeff_re` / `coeff_im` with `delay`, or `polmat`;
  columns of `interact_coord` beyond `sum(n_interact)` are ignored
- **`max_paths`** — Max paths to export; 0 includes all paths above `gain_min_dB`; default: 0
- **`gain_max_dB`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min_dB`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`i_snap`** — Snapshot indices to include; 1-based; range [1 ... n_snap]; empty exports all
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5

## Outputs:
- This function writes the OBJ file directly to disk and does not return any data
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs; // No outputs

    if (nrhs < 2 || nrhs > 10)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto fn = qd_mex_get_string(prhs[0], "", "fn");

    if (!mxIsStruct(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'chan' must be a 1x1 struct.");
    const auto chan = qd_mex_struct2channel(prhs[1], false, false, "chan"); // C++ validates

    const auto max_paths = (nrhs < 3) ? arma::uword(0) : qd_mex_get_scalar<arma::uword>(prhs[2], "max_paths", 0);
    const auto gain_max_dB = (nrhs < 4) ? -60.0 : qd_mex_get_scalar<double>(prhs[3], "gain_max_dB", -60.0);
    const auto gain_min_dB = (nrhs < 5) ? -140.0 : qd_mex_get_scalar<double>(prhs[4], "gain_min_dB", -140.0);
    const auto colormap = (nrhs < 6) ? std::string("jet") : qd_mex_get_string(prhs[5], "jet", "colormap");
    const arma::uvec i_snap = (nrhs < 7) ? arma::uvec() : qd_mex_get_index_Col(prhs[6], "i_snap"); // 1-based -> 0-based
    const auto radius_max = (nrhs < 8) ? 0.05 : qd_mex_get_scalar<double>(prhs[7], "radius_max", 0.05);
    const auto radius_min = (nrhs < 9) ? 0.01 : qd_mex_get_scalar<double>(prhs[8], "radius_min", 0.01);
    const auto n_edges = (nrhs < 10) ? arma::uword(5) : qd_mex_get_scalar<arma::uword>(prhs[9], "n_edges", 5);

    CALL_QD(quadriga_lib::channel_export_obj_file(fn, chan[0], max_paths, gain_max_dB, gain_min_dB, colormap,
                                                  i_snap, radius_max, radius_min, n_edges));
}
