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

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# CHANNEL_EXPORT_OBJ_FILE
Export path data to a Wavefront OBJ file for visualization in Blender

## Description:
This function exports path data to a Wavefront OBJ file, which can be used for visualization in 3D
software such as Blender. It supports various colormaps for color-coding the paths based on their
gain values. In addition, the function allows you to control the maximum number of paths displayed,
set gain thresholds for color-coding and selection.

## Usage:

```
quadriga_lib.channel_export_obj_file( fn, max_no_paths, gain_max, gain_min, colormap, radius_max,
    radius_min, n_edges, rx_position, tx_position, no_interact, interact_coord, center_freq,
    coeff_re, coeff_im, i_snap )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the OBJ file, string, required

- **`max_no_paths`** (optional)<br>
  Maximum number of paths to be shown, optional, default: 0 = export all above `gain_min`

- **`gain_max`** (optional)<br>
  Maximum path gain in dB (only for color-coding), optional, default = -60.0

- **`gain_min`** (optional)<br>
  Minimum path gain in dB (for color-coding and path selection), optional, default = -140.0

- **`colormap`** (optional)<br>
  Colormap for the visualization, string, supported are 'jet', 'parula', 'winter', 'hot', 'turbo',
  'copper', 'spring', 'cool', 'gray', 'autumn', 'summer', optional, default = 'jet'

- **`radius_max`** (optional)<br>
  Maximum tube radius in meters, optional, default = 0.05

- **`radius_min`** (optional)<br>
  Minimum tube radius in meters, optional, default = 0.01

- **`n_edges`** (optional)<br>
  Number of vertices in the circle building the tube, must be >= 3, optional, default = 5

- **`rx_position`**<br>
  Receiver positions, required, size `[3, n_snap]` or `[3, 1]`

- **`tx_position`**<br>
  Transmitter positions, required, size `[3, n_snap]` or `[3, 1]`

- **`no_interact`**<br>
  Number interaction points of paths with the environment, required, uint32, Size `[n_path, n_snap]`

- **`interact_coord`**<br>
  Interaction coordinates, required, Size `[3, max(sum(no_interact)), n_snap]`

- **`center_freq`**<br>
  Center frequency in [Hz], required, Size `[n_snap, 1]` or scalar

- **`coeff_re`**<br>
  Channel coefficients, real part, Size: `[ n_rx, n_tx, n_path, n_snap ]`

- **`coeff_im`**<br>
  Channel coefficients, imaginary part, Size: `[ n_rx, n_tx, n_path, n_snap ]`

- **`i_snap`**<br> (optional)
  Snapshot indices, optional, 1-based, range [1 ... n_snap]

## Output Argument:
This function does not return a value. It writes the OBJ file directly to disk.

MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Number of in and outputs
  if (nrhs < 15 || nrhs > 16)
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Incorrect number of input arguments.");

  if (nlhs > 1)
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Incorrect number of output arguments.");

  // Read inputs
  std::string fn = qd_mex_get_string(prhs[0]);

  arma::uword max_no_paths = qd_mex_get_scalar<arma::uword>(prhs[1], "max_no_paths", 0ULL);
  double gain_max = qd_mex_get_scalar<double>(prhs[2], "gain_max", -60.0);
  double gain_min = qd_mex_get_scalar<double>(prhs[3], "gain_min", -140.0);

  std::string colormap = qd_mex_get_string(prhs[4], "jet");

  double radius_max = qd_mex_get_scalar<double>(prhs[5], "radius_max", 0.05);
  double radius_min = qd_mex_get_scalar<double>(prhs[6], "radius_min", 0.01);
  arma::uword n_edges = qd_mex_get_scalar<arma::uword>(prhs[7], "n_edges", 5ULL);

  // Construct channel object from input data
  auto c = quadriga_lib::channel<double>();

  c.rx_pos = qd_mex_typecast_Mat<double>(prhs[8], "rx_position");
  c.tx_pos = qd_mex_typecast_Mat<double>(prhs[9], "tx_position");
  c.no_interact = qd_mex_matlab2vector_Col<unsigned>(prhs[10], 1);
  c.interact_coord = qd_mex_matlab2vector_Mat<double>(prhs[11], 2);
  c.center_frequency = qd_mex_typecast_Col<double>(prhs[12], "center_freq");
  c.coeff_re = qd_mex_matlab2vector_Cube<double>(prhs[13], 3);
  c.coeff_im = qd_mex_matlab2vector_Cube<double>(prhs[14], 3);

  arma::uvec i_snap;
  if (nrhs > 15)
  {
    i_snap = qd_mex_typecast_Col<arma::uword>(prhs[15], "i_snap");
    i_snap = i_snap - 1ULL; // Convert to 0-based
  }

  if (c.coeff_re.size() > c.interact_coord.size())
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Number of snapshots in interact_coord must match coefficients.");

  for (size_t i_snap_a = 0ULL; i_snap_a < c.coeff_re.size(); ++i_snap_a)
  {
    // Add a zero-power delay matrix
    arma::cube delays(c.coeff_re[i_snap_a].n_rows, c.coeff_re[i_snap_a].n_cols, c.coeff_re[i_snap_a].n_slices);
    c.delay.push_back(delays);

    // Remove tailing zeros from interact_coord
    unsigned sum_no_int = arma::sum(c.no_interact[i_snap_a]);
    c.interact_coord[i_snap_a] = arma::resize(c.interact_coord[i_snap_a], 3, sum_no_int);
  }

  // Call Quadriga-Lib function
  CALL_QD(c.write_paths_to_obj_file(fn, max_no_paths, gain_max, gain_min, colormap, i_snap, radius_max, radius_min, n_edges));

  // Dummy output
  if (nlhs == 1)
  {
    double tmp = 1.0;
    plhs[0] = qd_mex_copy2matlab(&tmp);
  }
}
