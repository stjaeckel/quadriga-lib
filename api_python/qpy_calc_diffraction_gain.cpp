// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# calc_diffraction_gain
Calculate diffraction gain for multiple transmit and receive positions using a 3D triangular mesh

## Description:
- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction from mesh geometry. The wave
  propagation between each TX-RX pair is divided into `n_path` elliptic-arc paths (controlled by `lod`),
  each approximated by `n_seg` line segments.
- Individual segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients,
  generalized to arbitrary 3D shapes.
- Optional sub-mesh indexing (see [[triangle_mesh_segmentation]]) accelerates computation by skipping
  geometry whose bounding box does not intersect the TX-RX path.

## Usage:
```
from quadriga_lib import RTtools

# Output as tuple
data = RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_prop, center_frequency, lod, verbose, sub_mesh_index, use_kernel, gpu_id )

# Unpacked outputs
gain, coord = RTtools.calc_diffraction_gain( orig, dest, mesh, mtl_prop, center_frequency, lod, verbose, sub_mesh_index, use_kernel, gpu_id )
```

## Input Arguments:
- ndarray **`orig`**<br>
  TX positions in global coordinate system (GCS); dtype: float64; shape: `(n_pos, 3)`.

- ndarray **`dest`**<br>
  RX positions in global coordinate system (GCS); dtype: float64; shape: `(n_pos, 3)`.

- ndarray **`mesh`**<br>
  Triangle vertices, each row `[X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3]`; dtype: float64; shape: `(n_mesh, 9)`.

- ndarray **`mtl_prop`**<br>
  Material properties per triangle; dtype: float64; shape: `(n_mesh, 5)`. See [[obj_file_read]].

- float **`center_frequency`** (optional)<br>
  Center frequency in Hz. Default: `1e9`.

- int **`lod`** (optional)<br>
  Level of detail (0–6). Controls `n_path` and `n_seg`. Default: `2`

- int **`verbose`** (optional)<br>
  Verbosity level. Default: `0`.

- ndarray **`sub_mesh_index`** (optional)<br>
  Sub-mesh index for acceleration, 0-based; dtype: uint32; shape: `(n_mesh,)`. Pass empty array to skip.
  See [[triangle_mesh_segmentation]].

- int **`use_kernel`** (optional)<br>
  Selects the compute kernel: 0 = auto (default), 1 = GENERIC (scalar CPU), 2 = AVX2 (SIMD),
  3 = CUDA (GPU). An error is thrown if the requested kernel is not available at runtime.

- int **`gpu_id`** (optional)<br>
  GPU device ID for CUDA kernel (0-based). Default: `0`. Ignored for non-CUDA kernels.

## Output Arguments (tuple):
- ndarray **`gain`**<br>
  Diffraction gain per TX-RX pair, linear scale; dtype: float64; shape: `(n_pos,)`.

- ndarray **`coord`**<br>
  Diffracted path coordinates (excluding endpoints); dtype: float64; shape: `(3, n_seg-1, n_pos)`.

## See also:
- [[triangle_mesh_segmentation]]
- [[obj_file_read]]
MD!*/

py::tuple calc_diffraction_gain(const py::array_t<double> &orig,             // Ray origin points in GCS, Size [ n_pos, 3 ]
                                const py::array_t<double> &dest,             // Ray destination points in GCS, Size [ n_pos, 3 ]
                                const py::array_t<double> &mesh,             // Faces of the triangular mesh (input), Size: [ n_mesh, 9 ]
                                const py::array_t<double> &mtl_prop,         // Material properties; Size: [ n_mesh, 5 ]
                                double center_freq,                          // Center frequency in [Hz]
                                int lod,                                     // Level of detail, 0-6
                                int verbose,                                 // Verbosity level
                                const py::array_t<unsigned> &sub_mesh_index, // Start indices of the sub-meshes in 0-based notation
                                int use_kernel,                              // Kernel selector: 0=auto, 1=GENERIC, 2=AVX2, 3=CUDA
                                int gpu_id)                                  // GPU device ID for CUDA kernel
{
    const auto orig_arma = qd_python_numpy2arma_Mat(orig, true);
    const auto dest_arma = qd_python_numpy2arma_Mat(dest, true);
    const auto mesh_arma = qd_python_numpy2arma_Mat(mesh, true);
    const auto mtl_prop_arma = qd_python_numpy2arma_Mat(mtl_prop, true);
    const auto sub_mesh_index_arma = qd_python_numpy2arma_Col(sub_mesh_index, true);

    arma::uword n_pos = orig_arma.n_rows;
    arma::uword n_seg = 0;
    if (lod == 1 || lod == 2)
        n_seg = 2;
    else if (lod == 3)
        n_seg = 3;
    else if (lod == 4)
        n_seg = 4;
    else if (lod == 5 || lod == 6)
        n_seg = 1;

    // Pre-allocate outputs in Python memory and map Armadillo wrappers to them
    arma::vec gain;
    arma::cube coord;
    auto gain_p = qd_python_init_output(n_pos, &gain);
    auto coord_p = qd_python_init_output(3, n_seg, n_pos, &coord);

    // Resolve optional pointers
    const arma::u32_vec *p_sub_mesh_index = sub_mesh_index_arma.empty() ? nullptr : &sub_mesh_index_arma;

    quadriga_lib::calc_diffraction_gain<double>(&orig_arma, &dest_arma, &mesh_arma, &mtl_prop_arma,
                                                center_freq, lod, &gain, &coord, verbose,
                                                p_sub_mesh_index, use_kernel, gpu_id);

    return py::make_tuple(gain_p, coord_p);
}