// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "quadriga_python_adapter.hpp"
#include "quadriga_lib.hpp"

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
    coeff, i_snap )
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

- **`coeff`**<br>
  Channel coefficients, complex valued, required only if `path_polarization` is not given,
  Size `[n_rx, n_tx, n_path, n_snap]`

- **`i_snap`**<br> (optional)
  Snapshot indices, optional, 0-based, range [0 ... n_snap - 1]
MD!*/

void channel_export_obj_file(const std::string fn,
                             size_t max_no_paths,
                             double gain_max,
                             double gain_min,
                             const std::string colormap,
                             double radius_max,
                             double radius_min,
                             size_t n_edges,
                             const py::array_t<double> rx_pos,
                             const py::array_t<double> tx_pos,
                             const py::list no_interact,
                             const py::list interact_coord,
                             const py::array_t<double> center_freq,
                             const py::list coeff,
                             const py::array_t<arma::uword> snap)
{
    // Construct channel object from input data
    auto c = quadriga_lib::channel<double>();

    c.rx_pos = qd_python_numpy2arma_Mat(rx_pos, true);
    c.tx_pos = qd_python_numpy2arma_Mat(tx_pos, true);
    c.no_interact = qd_python_list2vector_Col<unsigned>(no_interact);
    c.interact_coord = qd_python_list2vector_Mat<double>(interact_coord);
    c.center_frequency = qd_python_numpy2arma_Col(center_freq, true, true);
    qd_python_list2vector_Cube_Cplx(coeff, c.coeff_re, c.coeff_im);

    // Add a zero-power delay matrix
    for (size_t i = 0; i < c.coeff_re.size(); ++i)
    {
        arma::cube delays(c.coeff_re[i].n_rows, c.coeff_re[i].n_cols, c.coeff_re[i].n_slices);
        c.delay.push_back(delays);
    }

    arma::uvec snap_arma = qd_python_numpy2arma_Col(snap, true);
    c.write_paths_to_obj_file(fn, max_no_paths, gain_max, gain_min, colormap, snap_arma, radius_max, radius_min, n_edges);
}
