// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# CALC_ANGULAR_SPREAD
Calculate azimuth and elevation angular spreads with spherical wrapping

- Computes RMS azimuth and elevation angular spreads from power-weighted angles; each column
  of `az`/`el`/`powers` is one CIR; zero-power paths can be used to pad CIRs with fewer paths
- RMS spread formula: `sqrt(sum(pw .* d^2))` where `d` are wrapped deviations from the
  circular mean (3GPP TR 38.901 second-moment definition)
- When `wrapping = true`, the power-weighted mean direction is computed in Cartesian
  coordinates and all paths are rotated so the centroid lies on the equator before computing
  spreads, avoiding pole singularity artifacts
- When `wrapping = false`, spreads are computed directly from raw angles; `orientation` is
  zero and `phi`/`theta` equal the input `az`/`el`
- When `calc_bank_angle = true`, an optimal bank angle maximizing azimuth spread is derived
  analytically from eigenvectors of the 2x2 power-weighted covariance matrix of centered
  angles; only used when `wrapping = true`
- When `quantize > 0`, paths within that angular distance are grouped and their powers
  summed before computing spreads

## Usage:
```
[ as, es, orientation, phi, theta ] = quadriga_lib.calc_angular_spread( az, el, powers, ...
    wrapping, calc_bank_angle, quantize );
```

## Inputs:
- **`az`** — Azimuth angles; range -pi to pi; `[n_path, n_cir]`
- **`el`** — Elevation angles; range -pi/2 to pi/2; `[n_path, n_cir]`
- **`powers`** — Path powers in [W]; `[n_path, n_cir]`
- **`wrapping`** — If true, enables spherical rotation; default: false
- **`calc_bank_angle`** — If true, computes optimal bank angle analytically; only used when `wrapping = true`; default: false
- **`quantize`** — Angular quantization step in [deg]; paths within this distance are grouped; default: 0 (no quantization)

## Outputs:
- **`as`** — RMS azimuth angular spread; `[n_cir]`
- **`es`** — RMS elevation angular spread; `[n_cir]`
- **`orientation`** — Power-weighted mean orientation in Euler angles [bank; tilt; heading]; `[3, n_cir]`
- **`phi`** — Rotated azimuth angles; `[n_path, n_cir]`
- **`theta`** — Rotated elevation angles; `[n_path, n_cir]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 3 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many output arguments (max 5).");

    // Read inputs: convert MATLAB matrices to std::vector<arma::vec>
    const std::vector<arma::vec> az = qd_mex_matlab2vector_Col<double>(prhs[0], 1);
    const std::vector<arma::vec> el = qd_mex_matlab2vector_Col<double>(prhs[1], 1);
    const std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[2], 1);

    const bool wrapping = (nrhs < 4) ? false : qd_mex_get_scalar<bool>(prhs[3], "wrapping", false);
    const bool calc_bank_angle = (nrhs < 5) ? false : qd_mex_get_scalar<bool>(prhs[4], "calc_bank_angle", false);
    const double quantize = (nrhs < 6) ? 0.0 : qd_mex_get_scalar<double>(prhs[5], "quantize", 0.0);

    // Declare output variables
    arma::vec azimuth_spread, elevation_spread;
    arma::mat orientation;
    std::vector<arma::vec> phi, theta;

    // Set up optional output pointers based on nlhs
    arma::vec *p_as = (nlhs > 0) ? &azimuth_spread : nullptr;
    arma::vec *p_es = (nlhs > 1) ? &elevation_spread : nullptr;
    arma::mat *p_orient = (nlhs > 2) ? &orientation : nullptr;
    std::vector<arma::vec> *p_phi = (nlhs > 3) ? &phi : nullptr;
    std::vector<arma::vec> *p_theta = (nlhs > 4) ? &theta : nullptr;

    // Call library function (double precision only)
    CALL_QD(quadriga_lib::calc_angular_spreads_sphere<double>(
        az, el, powers, p_as, p_es, p_orient, p_phi, p_theta,
        !wrapping, calc_bank_angle, quantize));

    // Write outputs to MATLAB
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
