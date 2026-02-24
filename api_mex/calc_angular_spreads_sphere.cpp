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
- Uses spherical coordinate wrapping to avoid the pole singularity: the power-weighted mean
  direction is computed in Cartesian coordinates and all paths are rotated so the centroid lies
  on the equator before computing spreads.
- Without spherical wrapping, azimuth spread near the poles is inflated (large azimuth spread
  despite energy being focused into a small solid angle). This method corrects for that.
- Optionally computes an optimal bank (roll) angle that maximizes azimuth spread and minimizes
  elevation spread, corresponding to the principal axes of the angular power distribution.
- If `pow` has only 1 row but `az` has `n_ang` rows, the power vector is replicated.
- If `el` has only 1 row but `az` has `n_ang` rows, elevation is assumed zero.

## Usage:
```
[ as, es, orientation, phi, theta ] = quadriga_lib.calc_angular_spreads_sphere( az, el, pw );
[ as, es, orientation, phi, theta ] = quadriga_lib.calc_angular_spreads_sphere( az, el, pw, calc_bank_angle, quantize );
```

## Input Arguments:
- **`az`**<br>
  Azimuth angles in [rad], ranging from -pi to pi. Size `[n_ang, n_path]`.

- **`el`**<br>
  Elevation angles in [rad], ranging from -pi/2 to pi/2. Size `[n_ang, n_path]` or `[1, n_path]`.

- **`pw`**<br>
  Path powers in [W]. Size `[n_ang, n_path]` or `[1, n_path]`.

- **`calc_bank_angle`** (optional)<br>
  Logical. If true (default), the optimal bank angle is computed analytically. If false, bank is
  set to zero.

- **`quantize`** (optional)<br>
  Angular quantization step in [deg]. Paths within this angular distance are grouped and their
  powers summed before computing the spread. Default: 0 (no quantization).

## Output Arguments:
- **`as`**<br>
  RMS azimuth angular spread in [rad]. Type: `double`. Size `[n_ang, 1]`.

- **`es`**<br>
  RMS elevation angular spread in [rad]. Type: `double`. Size `[n_ang, 1]`.

- **`orientation`**<br>
  Power-weighted mean-angle orientation: row 1 = bank angle, row 2 = tilt angle, row 3 = heading
  angle, all in [rad]. Type: `double`. Size `[3, n_ang]`.

- **`phi`**<br>
  Rotated azimuth angles in [rad]. Type: `double`. Size `[n_ang, n_path]`.

- **`theta`**<br>
  Rotated elevation angles in [rad]. Type: `double`. Size `[n_ang, n_path]`.

## Example:
```
az = [0.1, 0.2, -0.1, 0.3];
el = [0.0, 0.05, -0.05, 0.02];
pw = [1.0, 2.0, 1.5, 0.5];
[as, es, orient] = quadriga_lib.calc_angular_spreads_sphere(az, el, pw);
```
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // --- Validate argument counts ---
    if (nrhs < 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments (need at least 3: az, el, pw).");
    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many output arguments (max 5).");

    // --- Read inputs ---
    arma::mat az = qd_mex_get_double_Mat(prhs[0]);
    arma::mat el = qd_mex_get_double_Mat(prhs[1]);
    arma::mat pw = qd_mex_get_double_Mat(prhs[2]);

    bool calc_bank_angle = (nrhs < 4) ? true : qd_mex_get_scalar<bool>(prhs[3], "calc_bank_angle", true);
    double quantize = (nrhs < 5) ? 0.0 : qd_mex_get_scalar<double>(prhs[4], "quantize", 0.0);

    // --- Declare output variables ---
    arma::vec azimuth_spread, elevation_spread;
    arma::mat orientation, phi, theta;

    // --- Set up optional output pointers based on nlhs ---
    arma::vec *p_as = (nlhs > 0) ? &azimuth_spread : nullptr;
    arma::vec *p_es = (nlhs > 1) ? &elevation_spread : nullptr;
    arma::mat *p_orient = (nlhs > 2) ? &orientation : nullptr;
    arma::mat *p_phi = (nlhs > 3) ? &phi : nullptr;
    arma::mat *p_theta = (nlhs > 4) ? &theta : nullptr;

    // --- Call library function (double precision only) ---
    CALL_QD(quadriga_lib::calc_angular_spreads_sphere<double>(
        az, el, pw, p_as, p_es, p_orient, p_phi, p_theta,
        calc_bank_angle, quantize));

    // --- Write outputs to MATLAB ---
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&azimuth_spread);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&elevation_spread);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&orientation);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&phi);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&theta);
}
