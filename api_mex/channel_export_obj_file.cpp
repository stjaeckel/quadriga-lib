// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# CHANNEL_EXPORT_OBJ_FILE
Export propagation paths to a Wavefront OBJ file for 3D visualization

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` limits the total number of exported paths
- The function takes raw channel data fields directly; no MATLAB channel struct is needed

## Usage:
```
quadriga_lib.channel_export_obj_file( fn, max_no_paths, gain_max, gain_min, colormap, radius_max, ...
    radius_min, n_edges, rx_position, tx_position, no_interact, interact_coord, center_freq, ...
    coeff_re, coeff_im, i_snap );
```

## Inputs:
- **`fn`** — Output `.obj` file path
- **`max_no_paths`** *(optional)* — Max paths to export; 0 includes all paths above `gain_min`; default: 0
- **`gain_max`** *(optional)* — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min`** *(optional)* — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** *(optional)* — Colormap name; supported: jet, parula, winter, hot, turbo, copper, 
  spring, cool, gray, autumn, summer; default: jet
- **`radius_max`** *(optional)* — Tube radius at maximum gain; default: 0.05
- **`radius_min`** *(optional)* — Tube radius at minimum gain; default: 0.01
- **`n_edges`** *(optional)* — Vertices per tube cross-section; must be >= 3; default: 5
- **`rx_position`** — Receiver positions; `[3, n_snap]` or `[3, 1]`
- **`tx_position`** — Transmitter positions; `[3, n_snap]` or `[3, 1]`
- **`no_interact`** — Number of interaction points of paths with the environment; uint32; `[n_path, n_snap]`
- **`interact_coord`** — Interaction coordinates; `[3, max(sum(no_interact)), n_snap]`
- **`center_freq`** — Center frequency; unit: Hz; `[n_snap]` or scalar
- **`coeff_re`** — Channel coefficients, real part; `[n_rx, n_tx, n_path, n_snap]`
- **`coeff_im`** — Channel coefficients, imaginary part; `[n_rx, n_tx, n_path, n_snap]`
- **`i_snap`** *(optional)* — 1-based snapshot indices to include; range [1 ... n_snap]; empty exports all

## Outputs:
- This function writes the OBJ file directly to disk and does not return any data
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 15 || nrhs > 16)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const auto fn = qd_mex_get_string(prhs[0]);

    const auto max_no_paths = qd_mex_get_scalar<arma::uword>(prhs[1], "max_no_paths", 0);
    const auto gain_max = qd_mex_get_scalar<double>(prhs[2], "gain_max", -60.0);
    const auto gain_min = qd_mex_get_scalar<double>(prhs[3], "gain_min", -140.0);
    const auto colormap = qd_mex_get_string(prhs[4], "jet");
    const auto radius_max = qd_mex_get_scalar<double>(prhs[5], "radius_max", 0.05);
    const auto radius_min = qd_mex_get_scalar<double>(prhs[6], "radius_min", 0.01);
    const auto n_edges = qd_mex_get_scalar<arma::uword>(prhs[7], "n_edges", 5);

    // Construct channel object from input data
    auto c = quadriga_lib::channel<double>();

    c.rx_pos = qd_mex_get_Mat<double>(prhs[8]);
    c.tx_pos = qd_mex_get_Mat<double>(prhs[9]);
    c.no_interact = qd_mex_matlab2vector_Col<unsigned>(prhs[10], 1);
    c.interact_coord = qd_mex_matlab2vector_Mat<double>(prhs[11], 2);
    c.center_frequency = qd_mex_get_Col<double>(prhs[12]);
    c.coeff_re = qd_mex_matlab2vector_Cube<double>(prhs[13], 3);
    c.coeff_im = qd_mex_matlab2vector_Cube<double>(prhs[14], 3);

    arma::uvec i_snap;
    if (nrhs > 15)
    {
        i_snap = qd_mex_get_Col<arma::uword>(prhs[15]);
        if (!i_snap.is_empty())
            i_snap = i_snap - 1; // Convert from 1-based to 0-based
    }

    if (c.coeff_re.size() > c.interact_coord.size())
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Number of snapshots in interact_coord must match coefficients.");

    for (size_t i_snap_a = 0; i_snap_a < c.coeff_re.size(); ++i_snap_a)
    {
        // Add a zero-power delay matrix (write_paths_to_obj_file expects delays to be present)
        arma::cube delays(c.coeff_re[i_snap_a].n_rows,
                          c.coeff_re[i_snap_a].n_cols,
                          c.coeff_re[i_snap_a].n_slices);
        c.delay.push_back(delays);

        // Remove trailing zeros from interact_coord
        unsigned sum_no_int = arma::sum(c.no_interact[i_snap_a]);
        c.interact_coord[i_snap_a] = arma::resize(c.interact_coord[i_snap_a], 3, sum_no_int);
    }

    // Call quadriga-lib function
    CALL_QD(c.write_paths_to_obj_file(fn, max_no_paths, gain_max, gain_min, colormap,
                                      i_snap, radius_max, radius_min, n_edges));

    // Dummy output for backward compatibility
    double out = 1.0;
    if (nlhs == 1)
        plhs[0] = qd_mex_copy2matlab(&out);
}