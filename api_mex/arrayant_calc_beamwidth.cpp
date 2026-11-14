// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_CALC_BEAMWIDTH
Calculate the beamwidth and pointing angles of array antenna elements in degrees

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern (≈100x finer grid in
  each direction than the antenna sampling grid)
- The width is that of the main lobe, searched outwards from the maximum; side lobes above the threshold
  are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full
  circle; 360° when the whole cut is above the threshold (e.g. a beam pointing to a pole)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles;
  at most 180°
- Calculated per element, not per port; ignores element coupling

## Usage:
```
[ beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg ] = ...
    quadriga_lib.arrayant_calc_beamwidth( arrayant, i_element, threshold_dB );
```

## Inputs:
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array may contain a frequency-dependent model
- **`i_element`** — Element indices, 1-based, duplicates allowed; uint64; `[n_out]`; default: all elements
- **`threshold_dB`** — Threshold in dB, positive and finite; default: 3 (equivalent to FWHM)

## Outputs:
- **`beamwidth_az_deg`** — Azimuth beamwidth in degree; `[n_out, n_freq]`; with `n_out = n_elements` when `i_element` is omitted/empty
- **`beamwidth_el_deg`** — Elevation beamwidth in degree; `[n_out, n_freq]`
- **`az_point_ang_deg`** — Azimuth pointing angle of the main beam in degree; `[n_out, n_freq]`
- **`el_point_ang_deg`** — Elevation pointing angle of the main beam in degree; `[n_out, n_freq]`

## See also:
- [[arrayant_combine_pattern]] (to apply element coupling before calculating the beamwidth)
- [[arrayant_calc_directivity]] (directivity in dBi of array antenna elements)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    if (!mxIsStruct(prhs[0]) || mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant' must be an arrayant struct or struct array.");

    // Element indices: 1-based MATLAB -> 0-based C++ (empty = all elements)
    arma::uvec i_element = (nrhs < 2) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[1], true, "i_element");
    if (arma::any(i_element == 0))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'i_element' cannot be 0 (1-based index).");
    i_element -= 1;
    const double threshold_dB = (nrhs < 3) ? 3.0 : qd_mex_get_scalar<double>(prhs[2], "threshold_dB", 3.0);

    // A single struct is a one-entry vector; the outputs are [n_out, n_freq] either way. The first output is also
    // computed without output argument (it goes to ans).
    const int n_out = (nlhs == 0) ? 1 : nlhs;
    arma::mat out[4];
    const auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0]); // Validated by the C++ function
    CALL_QD(quadriga_lib::arrayant_calc_beamwidth_multi(ant_multi, &i_element, threshold_dB, &out[0],
                                                        n_out > 1 ? &out[1] : nullptr, n_out > 2 ? &out[2] : nullptr,
                                                        n_out > 3 ? &out[3] : nullptr));
    for (int k = 0; k < n_out; ++k)
        plhs[k] = qd_mex_copy2matlab(&out[k]);
}
