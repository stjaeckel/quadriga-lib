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

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# CALC_DIFFRACTION_GAIN
Calculate diffraction gain for multiple transmit and receive positions using a 3D triangular mesh

## Description:
- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction from mesh geometry. The wave
  propagation between each TX-RX pair is divided into `n_path` elliptic-arc paths (controlled by `lod`),
  each approximated by `n_seg` line segments. 
- Individual segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients, 
  generalized to arbitrary 3D shapes.
- Optional sub-mesh indexing (see [[triangle_mesh_segmentation]]) accelerates computation by skipping
  geometry whose bounding box does not intersect the TX-RX path.
- All inputs are cast to double precision internally.

## Usage:
```
[ gain, coord ] = quadriga_lib.calc_diffraction_gain( orig, dest, mesh, mtl_prop, ...
    center_frequency, lod, verbose, sub_mesh_index, use_kernel, gpu_id );
```

## Input Arguments:
- `**orig**` (input)<br>
  TX positions, size `[n_pos, 3]`, numeric.

- `**dest**` (input)<br>
  RX positions, size `[n_pos, 3]`, numeric.

- `**mesh**` (input)<br>
  Triangle vertices, each row `[X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3]`, size `[n_mesh, 9]`, numeric.

- `**mtl_prop**` (input)<br>
  Material properties per triangle, size `[n_mesh, 5]`, numeric. See [[obj_file_read]].

- `**center_frequency**` (input)<br>
  Center frequency in Hz, scalar.

- `**lod**` (optional input)<br>
  Level of detail (0–6), scalar. Default: `2`. See [[generate_diffraction_paths]].

- `**verbose**` (optional input)<br>
  Verbosity level, scalar. Default: `0`.

- `**sub_mesh_index**` (optional input)<br>
  Sub-mesh index for acceleration, 0-based, `uint32` vector of length `[n_mesh]`. Pass `[]` to skip.
  See [[triangle_mesh_segmentation]].

- `**use_kernel**` (optional input)<br>
  Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA. Default: `0`. Error if unavailable.

- `**gpu_id**` (optional input)<br>
  CUDA device ID (0-based). Default: `0`. Ignored for non-CUDA kernels.

## Output Arguments:
- `**gain**`<br>
  Diffraction gain per TX-RX pair, linear scale, size `[n_pos, 1]`.

- `**coord**`<br>
  Diffracted path coordinates (excluding endpoints), size `[3, n_seg-1, n_pos]`.

## See also:
- [[generate_diffraction_paths]]
- [[triangle_mesh_segmentation]]
- [[obj_file_read]]
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 5 || nrhs > 10)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many output arguments.");

    // Load inputs (cast to double if needed)
    arma::mat orig = qd_mex_get_double_Mat(prhs[0]);
    arma::mat dest = qd_mex_get_double_Mat(prhs[1]);
    arma::mat mesh = qd_mex_get_double_Mat(prhs[2]);
    arma::mat mtl_prop = qd_mex_get_double_Mat(prhs[3]);

    double center_freq = qd_mex_get_scalar<double>(prhs[4], "center_frequency", 1.0e9);
    int lod = (nrhs < 6) ? 2 : qd_mex_get_scalar<int>(prhs[5], "lod", 2);
    int verbose = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "verbose", 0);

    // Optional: sub_mesh_index
    arma::u32_vec sub_mesh_index;
    if (nrhs > 7 && !mxIsEmpty(prhs[7]))
    {
        if (mxIsUint32(prhs[7]))
            sub_mesh_index = qd_mex_reinterpret_Col<unsigned>(prhs[7]);
        else
            sub_mesh_index = qd_mex_typecast_Col<unsigned>(prhs[7]);
    }

    int use_kernel = (nrhs < 9) ? 0 : qd_mex_get_scalar<int>(prhs[8], "use_kernel", 0);
    int gpu_id = (nrhs < 10) ? 0 : qd_mex_get_scalar<int>(prhs[9], "gpu_id", 0);

    arma::uword n_pos = orig.n_rows;
    arma::uword n_seg = 0;
    if (lod == 1 || lod == 2)
        n_seg = 2;
    else if (lod == 3)
        n_seg = 3;
    else if (lod == 4)
        n_seg = 4;
    else if (lod == 5 || lod == 6)
        n_seg = 1;

    // Initialize output containers
    arma::vec gain;
    arma::cube coord;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&gain, n_pos);

    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&coord, 3, n_seg, n_pos);

    arma::vec *p_gain = gain.empty() ? nullptr : &gain;
    arma::cube *p_coord = coord.empty() ? nullptr : &coord;
    arma::u32_vec *p_sub_mesh_index = sub_mesh_index.empty() ? nullptr : &sub_mesh_index;

    CALL_QD(quadriga_lib::calc_diffraction_gain(&orig, &dest, &mesh, &mtl_prop,
                                                center_freq, lod, p_gain, p_coord, verbose,
                                                p_sub_mesh_index, use_kernel, gpu_id));
}