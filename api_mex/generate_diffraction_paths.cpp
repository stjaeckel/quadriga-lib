// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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
#include "mex_helper_functions.cpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# GENERATE_DIFFRACTION_PATHS
Generate propagation paths for estimating the diffraction gain

## Description:
Diffraction refers to the phenomenon where waves bend or interfere around the edges of an obstacle,
extending into the region that would otherwise be in the obstacle's geometrical shadow. The object
causing the diffraction acts as a secondary source for the wave's propagation. A specific example of
this is the knife-edge effect, or knife-edge diffraction, where a sharp, well-defined obstacle—like
a mountain range or a building wall—partially truncates the incident radiation.<br><br>

To estimate the diffraction gain in a three-dimensional space, one can assess the extent to which the
Fresnel ellipsoid is obstructed by objects, and then evaluate the impact of this obstruction on the
received power. This method presupposes that diffracted waves travel along slightly varied paths
before arriving at a receiver. These waves may reach the receiver out of phase with the primary wave
due to their different travel lengths, leading to either constructive or destructive interference.<br><br>

The process of estimating the gain involves dividing the wave propagation from a transmitter to a
receiver into `n_path` paths. These paths are represented by elliptic arcs, which are further
approximated using `n_seg` line segments. Each segment can be individually blocked or attenuated
by environmental objects. To determine the overall diffraction gain, a weighted sum of these
individual path contributions is calculated. The weighting is adjusted to align with the uniform
theory of diffraction (UTD) coefficients in two dimensions, but the methodology is adapted for
any 3D object shape. This function generates the elliptic propagation paths and corresponding weights
necessary for this calculation.

## Caveat:
- Each ellipsoid consists of `n_path` diffraction paths. The number of paths is determined by the
  level of detail (`lod`).
- All diffraction paths of an ellipsoid originate at `orig` and arrive at `dest`
- Each diffraction path has `n_seg` segments
- Points `orig` and `dest` lay on the semi-major axis of the ellipsoid
- The generated rays sample the volume of the ellipsoid
- Weights are calculated from the Knife-edge diffraction model when parts of the ellipsoid are shadowed
- Initial weights are normalized such that `sum(prod(weights,3),2) = 1`
- Inputs `orig` and `dest` may be provided as double or single precision

## Usage:

```
[ rays, weights ] = quadriga_lib.generate_diffraction_paths( orig, dest, center_frequency, lod );
```

## Input Arguments:
- **`orig`**<br>
  Origin point of the propagation ellipsoid (e.g. transmitter positions). Size: `[ n_pos, 3 ]`

- **`dest`**<br>
  Destination point of the propagation ellipsoid (e.g. receiver positions). Size: `[ n_pos, 3 ]`

- **`center_freq`**<br>
  The center frequency in [Hz], scalar, default = 299792458 Hz

- **`lod`**<br>
  Level of detail, scalar value
  `lod = 1` | results in `n_path = 7` and `n_seg = 3`
  `lod = 2` | results in `n_path = 19` and `n_seg = 3`
  `lod = 3` | results in `n_path = 37` and `n_seg = 4`
  `lod = 4` | results in `n_path = 61` and `n_seg = 5`
  `lod = 5` | results in `n_path = 1` and `n_seg = 2` (for debugging)
  `lod = 6` | results in `n_path = 2` and `n_seg = 2` (for debugging)

## Output Arguments:
- **`rays`**<br>
  Coordinates of the generated rays; Size: `[ n_pos, n_path, n_seg-1, 3 ]`

- **`weights`**<br>
  Weights; Size: `[ n_pos, n_path, n_seg ]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2)
        mexErrMsgIdAndTxt("quadriga_lib:generate_diffraction_paths:io_error", "Need at least 2 input arguments.");

    if (nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:generate_diffraction_paths:io_error", "Too many input arguments.");

    if (nlhs != 2)
        mexErrMsgIdAndTxt("quadriga_lib:generate_diffraction_paths:io_error", "Need exactly 2 output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:generate_diffraction_paths:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    if ((use_single && !mxIsSingle(prhs[1])) || (!use_single && !mxIsDouble(prhs[1])))
        mexErrMsgIdAndTxt("quadriga_lib:generate_diffraction_paths:wrong_type", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    // Read inputs
    arma::mat orig_double, dest_double;
    arma::fmat orig_single, dest_single;
    if (use_single)
        orig_single = qd_mex_reinterpret_Mat<float>(prhs[0]),
        dest_single = qd_mex_reinterpret_Mat<float>(prhs[1]);
    else
        orig_double = qd_mex_reinterpret_Mat<double>(prhs[0]),
        dest_double = qd_mex_reinterpret_Mat<double>(prhs[1]);

    double center_freq = nrhs < 3 ? 299792458.0 : qd_mex_get_scalar<double>(prhs[2], "center_freq", 299792458.0);
    int lod = nrhs < 4 ? 1 : qd_mex_get_scalar<int>(prhs[3], "lod", 1);

    // Array dimensions
    arma::uword n_pos = use_single ? orig_single.n_rows : orig_double.n_rows;

    arma::uword n_path = 0, n_seg = 0;
    if (lod == 1)
        n_seg = 2, n_path = 7;
    else if (lod == 2)
        n_seg = 2, n_path = 19;
    else if (lod == 3)
        n_seg = 3, n_path = 37;
    else if (lod == 4)
        n_seg = 4, n_path = 61;
    else if (lod == 5)
        n_seg = 1, n_path = 1;
    else if (lod == 6)
        n_seg = 1, n_path = 2;

    // Reserve memory for the output
    arma::cube ray_x_double, ray_y_double, ray_z_double, weight_double;
    arma::fcube ray_x_single, ray_y_single, ray_z_single, weight_single;

    mwSize dims[4] = {(mwSize)n_pos, (mwSize)n_path, (mwSize)n_seg, 3};
    size_t offset = size_t(n_pos * n_path * n_seg);
    if (use_single)
    {
        plhs[0] = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        float *data = (float *)mxGetData(plhs[0]);

        ray_x_single = arma::Cube<float>(data, n_pos, n_path, n_seg, false, true);
        ray_y_single = arma::Cube<float>(&data[offset], n_pos, n_path, n_seg, false, true);
        ray_z_single = arma::Cube<float>(&data[2 * offset], n_pos, n_path, n_seg, false, true);

        plhs[1] = qd_mex_init_output(&weight_single, n_pos, n_path, n_seg+1);
    }
    else // double
    {
        plhs[0] = mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL);
        double *data = (double *)mxGetData(plhs[0]);

        ray_x_double = arma::Cube<double>(data, n_pos, n_path, n_seg, false, true);
        ray_y_double = arma::Cube<double>(&data[offset], n_pos, n_path, n_seg, false, true);
        ray_z_double = arma::Cube<double>(&data[2 * offset], n_pos, n_path, n_seg, false, true);

        plhs[1] = qd_mex_init_output(&weight_double, n_pos, n_path, n_seg+1);
    }

    // Call the quadriga-lib function
    try
    {
        if (use_single)
            quadriga_lib::generate_diffraction_paths(&orig_single, &dest_single, (float)center_freq, lod,
                                                     &ray_x_single, &ray_y_single, &ray_z_single, &weight_single);
        else
            quadriga_lib::generate_diffraction_paths(&orig_double, &dest_double, center_freq, lod,
                                                     &ray_x_double, &ray_y_double, &ray_z_double, &weight_double);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:generate_diffraction_paths:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:generate_diffraction_paths:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }
}