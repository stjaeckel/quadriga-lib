// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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
Miscellaneous / Tools
SECTION!*/

/*!MD
# calc_angular_spreads_sphere
Calculate azimuth and elevation angular spreads with spherical wrapping

## Description:
- Calculates the RMS azimuth and elevation angular spreads from a set of power-weighted angles.
- Inputs and outputs for angles and powers are provided as 2D matrices where each column
  represents a CIR (internally converted to vectors of column vectors to allow variable path
  counts per CIR when called from C++).
- Uses spherical coordinate wrapping to avoid the pole singularity: the power-weighted mean
  direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies
  on the equator before computing spreads.
- Without spherical wrapping, azimuth spread near the poles is inflated (large azimuth spread
  despite energy being focused into a small solid angle). This method corrects for that.
- Optionally computes an optimal bank (roll) angle that maximizes azimuth spread and minimizes
  elevation spread, corresponding to the principal axes of the angular power distribution.
- Setting `disable_wrapping` to true skips the rotation and computes spreads from raw angles.

## Usage:
```
[ as, es, orientation, phi, theta ] = quadriga_lib.calc_angular_spreads_sphere( az, el, powers );
[ as, es, orientation, phi, theta ] = quadriga_lib.calc_angular_spreads_sphere( az, el, powers, disable_wrapping, calc_bank_angle, quantize );
```

## Input Arguments:
- **`az`**<br>
  Azimuth angles in [rad], ranging from -pi to pi. Size `[n_path, n_cir]` (each column is one CIR).

- **`el`**<br>
  Elevation angles in [rad], ranging from -pi/2 to pi/2. Size `[n_path, n_cir]`.

- **`powers`**<br>
  Path powers in [W]. Size `[n_path, n_cir]`.

- **`disable_wrapping`** (optional)<br>
  Logical. If true, skip spherical rotation and compute spreads from raw angles. Default: false.

- **`calc_bank_angle`** (optional)<br>
  Logical. If true (default), compute the optimal bank angle analytically. Only used when
  `disable_wrapping` is false.

- **`quantize`** (optional)<br>
  Angular quantization step in [deg]. Default: 0 (no quantization).

## Output Arguments:
- **`as`**<br>
  RMS azimuth angular spread in [rad]. Type: `double`. Size `[n_cir, 1]`.

- **`es`**<br>
  RMS elevation angular spread in [rad]. Type: `double`. Size `[n_cir, 1]`.

- **`orientation`**<br>
  Power-weighted mean-angle orientation: row 1 = bank angle, row 2 = tilt angle, row 3 = heading
  angle, all in [rad]. Type: `double`. Size `[3, n_cir]`.

- **`phi`**<br>
  Rotated azimuth angles in [rad]. Type: `double`. Size `[n_path, n_cir]`.

- **`theta`**<br>
  Rotated elevation angles in [rad]. Type: `double`. Size `[n_path, n_cir]`.

## Example:
```
az = [0.1, 0.2; -0.1, -0.2; 0.05, 0.0];
el = [0.0, 0.05; 0.0, -0.05; 0.0, 0.0];
powers = [1.0, 2.0; 1.0, 1.0; 0.5, 1.5];
[as, es, orient] = quadriga_lib.calc_angular_spreads_sphere(az, el, powers);
```
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // --- Validate argument counts ---
    if (nrhs < 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments (need at least 3: az, el, powers).");
    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many output arguments (max 5).");

    // --- Read inputs: convert MATLAB matrices to std::vector<arma::vec> ---
    std::vector<arma::vec> az = qd_mex_matlab2vector_Col<double>(prhs[0], 1);
    std::vector<arma::vec> el = qd_mex_matlab2vector_Col<double>(prhs[1], 1);
    std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[2], 1);

    bool disable_wrapping = (nrhs < 4) ? false : qd_mex_get_scalar<bool>(prhs[3], "disable_wrapping", false);
    bool calc_bank_angle = (nrhs < 5) ? true : qd_mex_get_scalar<bool>(prhs[4], "calc_bank_angle", true);
    double quantize = (nrhs < 6) ? 0.0 : qd_mex_get_scalar<double>(prhs[5], "quantize", 0.0);

    // --- Declare output variables ---
    arma::vec azimuth_spread, elevation_spread;
    arma::mat orientation;
    std::vector<arma::vec> phi, theta;

    // --- Set up optional output pointers based on nlhs ---
    arma::vec *p_as = (nlhs > 0) ? &azimuth_spread : nullptr;
    arma::vec *p_es = (nlhs > 1) ? &elevation_spread : nullptr;
    arma::mat *p_orient = (nlhs > 2) ? &orientation : nullptr;
    std::vector<arma::vec> *p_phi = (nlhs > 3) ? &phi : nullptr;
    std::vector<arma::vec> *p_theta = (nlhs > 4) ? &theta : nullptr;

    // --- Call library function (double precision only) ---
    CALL_QD(quadriga_lib::calc_angular_spreads_sphere<double>(
        az, el, powers, p_as, p_es, p_orient, p_phi, p_theta,
        disable_wrapping, calc_bank_angle, quantize));

    // --- Write outputs to MATLAB ---
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&azimuth_spread);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&elevation_spread);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&orientation);
    if (nlhs > 3)
        plhs[3] = qd_mex_vector2matlab(&phi);
    if (nlhs > 4)
        plhs[4] = qd_mex_vector2matlab(&theta);
}
