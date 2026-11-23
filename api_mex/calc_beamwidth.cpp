// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# CALC_BEAMWIDTH
Calculate the beamwidth and pointing angles of the elements of an antenna array in degrees

- Measures the width of the main lobe of each element in azimuth and elevation at a threshold below its peak, and the
  direction of the peak
- Calculated per element, not per port: the port weights are ignored

## Usage:
```
[ beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg ] = ...
    quadriga_lib.calc_beamwidth( arrayant, i_element, threshold_dB );
```

## Inputs:
- **`arrayant`** — Arrayant struct ([[generate_antenna]]), or 1xN struct array for a multi-frequency array with `n_freq`
  entries
- **`i_element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements
- **`threshold_dB`** — Threshold below the peak in dB, positive and finite; 3 dB is the half-power beamwidth;
  loudspeakers state the coverage angle between the -6 dB points, use 6; default: 3

## Outputs:
- **`beamwidth_az_deg`** — Azimuth beamwidth in degrees; `[n_out, n_freq]`
- **`beamwidth_el_deg`** — Elevation beamwidth in degrees; `[n_out, n_freq]`
- **`az_point_ang_deg`** — Azimuth pointing angle of the main beam in degrees; `[n_out, n_freq]`
- **`el_point_ang_deg`** — Elevation pointing angle of the main beam in degrees; `[n_out, n_freq]`

## Notes:
- **Main lobe**: starting at the maximum, the search goes outward until the power drops below the threshold; side lobes
  above the threshold are not included
- **Resolution**: the cuts are interpolated as in [[interpolate_pattern]], about 100 times finer than the sampling grid
- **Azimuth cut**: at the elevation of the maximum, circular across ±180° when the grid covers the full circle; 360°
  when the whole cut is above the threshold (e.g. omni patterns or a beam pointing to a pole, where the cut is a point)
- **Elevation cut**: along the great circle through the maximum, so a lobe can extend across the poles (requires samples
  at ±90°); the result is at most 180°

## See also:
- [[calc_directivity]] (directivity in dBi of the elements)
- [[apply_port_weights]] (combines the elements into ports, to measure the beams of the ports)
- [[generate_antenna]] (creates elements with a given beamwidth)
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
    const arma::uvec i_element = (nrhs < 2) ? arma::uvec() : qd_mex_get_index_Col(prhs[1], "i_element");
    const double threshold_dB = (nrhs < 3) ? 3.0 : qd_mex_get_scalar<double>(prhs[2], "threshold_dB", 3.0);

    // A single struct is a one-entry vector; the outputs are [n_out, n_freq] either way
    const int n_out = (nlhs == 0) ? 1 : nlhs;
    arma::mat out[4];
    const auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0], false, false, "arrayant"); // Validated by the C++ function
    CALL_QD(quadriga_lib::calc_beamwidth(ant_multi, i_element, threshold_dB, &out[0],
                                                        n_out > 1 ? &out[1] : nullptr, n_out > 2 ? &out[2] : nullptr,
                                                        n_out > 3 ? &out[3] : nullptr));
    for (int k = 0; k < n_out; ++k)
        plhs[k] = qd_mex_copy2matlab(&out[k]);
}