// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_tools.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# CALC_ANGULAR_SPREAD
Calculate the RMS azimuth and elevation angular spreads of each CIR, optionally around the mean direction of its paths

- For one angle, with the path powers $p_l$ normalized to sum 1 over the paths of a CIR (the second-moment definition
  of 3GPP TR 38.901):

  $$\sigma = \sqrt{\sum_l p_l \, \Delta_l^2}, \quad \Delta_l = \arg e^{j (\phi_l - \bar\phi)}$$

  with the circular mean $\bar\phi = \arg \sum_l p_l \, e^{j \phi_l}$
- `wrapping = false`: the spreads of the given azimuth and elevation angles; near the poles, the azimuth spread
  overstates the angular spread
- `wrapping = true`: the paths are first rotated so that their power-weighted mean direction (the mean of the unit
  vectors) lies at azimuth 0 on the equator; `calc_bank_angle` then also rotates them about that direction so that
  the azimuth spread is largest (principal axis of the power-weighted covariance of the rotated angles); `orientation`
  is the direction of an antenna that sees the paths at the rotated angles `phi` and `theta`
- `quantize_deg` > 0: the deviations $\Delta_l$ are rounded to multiples of `quantize_deg` and the powers of equal
  deviations summed, for azimuth and elevation separately

## Usage:
```
[ azimuth_spread, elevation_spread, orientation, phi, theta ] = quadriga_lib.calc_angular_spread( ...
    az, el, powers, wrapping, calc_bank_angle, quantize_deg );
```

## Inputs:
- **`az`** — Path azimuths in rad, one column per CIR; `[n_path, n_cir]`
- **`el`** — Path elevations in rad; `[n_path, n_cir]`
- **`powers`** — Path powers, linear, with a positive sum per CIR; zero-power entries pad CIRs with fewer paths;
  `[n_path, n_cir]`
- **`wrapping`** — Rotate the paths to the equator before computing the spreads; default: false
- **`calc_bank_angle`** — Rotate them also about the mean direction, for the largest azimuth
  spread; only with `wrapping`; default: false
- **`quantize_deg`** — Quantization step of the angle deviations in deg; 0 = none, else finite and >= 1e-3; default: 0

## Outputs:
- **`azimuth_spread`** — RMS azimuth spread in rad; `[n_cir, 1]`
- **`elevation_spread`** — RMS elevation spread in rad; `[n_cir, 1]`
- **`orientation`** — Euler angles (bank, tilt, heading) in rad of an antenna pointing at the mean
  direction (a positive tilt points down), zero without `wrapping`; `[3, n_cir]`
- **`phi`** — Path azimuths in rad in the frame of that antenna (as [[interpolate_pattern]]
  returns them), the input `az` without `wrapping`; `[n_path, n_cir]`
- **`theta`** — Path elevations in rad in that frame, the input `el` without `wrapping`;
  `[n_path, n_cir]`

## See also:
- [[calc_delay_spread]] (delay spread of the same paths)
- [[interpolate_pattern]] (antenna response in the rotated frame)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::vector<arma::vec> az = qd_mex_matlab2vector_Col<double>(prhs[0], 1, "az");
    const std::vector<arma::vec> el = qd_mex_matlab2vector_Col<double>(prhs[1], 1, "el");
    const std::vector<arma::vec> powers = qd_mex_matlab2vector_Col<double>(prhs[2], 1, "powers");

    const bool wrapping = (nrhs < 4) ? false : qd_mex_get_scalar<bool>(prhs[3], "wrapping", false);
    const bool calc_bank_angle = (nrhs < 5) ? false : qd_mex_get_scalar<bool>(prhs[4], "calc_bank_angle", false);
    const double quantize_deg = (nrhs < 6) ? 0.0 : qd_mex_get_scalar<double>(prhs[5], "quantize_deg", 0.0);

    arma::vec azimuth_spread, elevation_spread;
    arma::mat orientation;
    std::vector<arma::vec> phi, theta;

    arma::vec *p_es = (nlhs > 1) ? &elevation_spread : nullptr;
    arma::mat *p_orient = (nlhs > 2) ? &orientation : nullptr;
    std::vector<arma::vec> *p_phi = (nlhs > 3) ? &phi : nullptr;
    std::vector<arma::vec> *p_theta = (nlhs > 4) ? &theta : nullptr;

    CALL_QD(quadriga_lib::calc_angular_spread<double>(
        az, el, powers, &azimuth_spread, p_es, p_orient, p_phi, p_theta,
        wrapping, calc_bank_angle, quantize_deg));

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
