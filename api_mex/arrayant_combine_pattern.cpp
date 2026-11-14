// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_COMBINE_PATTERN
Combine element patterns, positions, and coupling weights into effective radiation patterns

- Integrates the element field patterns, element positions, and coupling weights into one effective
  pattern per port (column of the coupling matrix)
- The result behaves as a virtual array with one element per port, zeroed element positions, and an
  identity coupling matrix
- Speeds up MIMO channel computation; useful for beamforming in 5G systems and network planning
- The element positions add a phase that depends on frequency and `propagation_speed`; the default speed of light
  applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [[generate_speaker]])

## Usage:
```
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in, center_freq_new, azimuth_grid_new, ...
    elevation_grid_new, propagation_speed );
```

## Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array may contain a frequency-dependent model
- **`center_freq_new`** — Alternative frequency (grid) in Hz; if provided, the combined pattern is 
  recomputed (interpolated) at each requested frequency. If omitted or empty, the function uses each input 
   entry's `center_freq` unchanged.
- **`azimuth_grid_new`** — Alternative azimuth grid in rad; in [-pi, pi]; strictly increasing; defaults to input grid
- **`elevation_grid_new`** — Alternative elevation grid in rad; in [-pi/2, pi/2]; strictly increasing; defaults to input grid
- **`propagation_speed`** — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics;
  default: 299792458

## Outputs:
- **`arrayant_out`** — Arrayant struct (single-frequency result) or struct array (multi-frequency
  result); field layout as documented in [[arrayant_generate]]. Single struct when both input and
  `center_freq_new` describe a single frequency; struct array of size `numel(center_freq_new)` (or
  `numel(arrayant_in)` when `center_freq_new` is omitted) otherwise.

## See also:
- [[arrayant_generate]] (for field layout in the arrayant struct)
- [[arrayant_rotate_pattern]] (for changing the orientation of elements before combining)
- [[arrayant_calc_beamwidth]] (calculates the beam width of array antennas)
- [[arrayant_calc_directivity]] (directivity in dBi of array antenna elements)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    if (!mxIsStruct(prhs[0]) || mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant_in' must be an arrayant struct or struct array.");

    const arma::vec freq = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1]);
    const arma::vec az = (nrhs < 3) ? arma::vec() : qd_mex_get_Col<double>(prhs[2]);
    const arma::vec el = (nrhs < 4) ? arma::vec() : qd_mex_get_Col<double>(prhs[3]);
    const double propagation_speed = (nrhs < 5) ? 299792458.0 : qd_mex_get_scalar<double>(prhs[4], "propagation_speed", 299792458.0);

    // The output is also set without output argument (it goes to ans)
    if (freq.n_elem > 1 || mxGetNumberOfElements(prhs[0]) > 1) // Multi-frequency
    {
        const auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0]); // Validated by the C++ function
        std::vector<quadriga_lib::arrayant<double>> out;
        CALL_QD(out = quadriga_lib::arrayant_combine_pattern_multi(ant_multi, &az, &el, &freq, propagation_speed));
        plhs[0] = (out.size() == 1) ? qd_mex_arrayant2struct(out[0]) : qd_mex_arrayant2struct_multi(out);
        return;
    }

    auto ant = qd_mex_struct2arrayant(prhs[0]); // Validated by the C++ function
    if (freq.n_elem == 1) // Scalar frequency override
        ant.center_frequency = freq[0];

    quadriga_lib::arrayant<double> arrayant_out;
    CALL_QD(arrayant_out = quadriga_lib::arrayant_combine_pattern(ant, &az, &el, propagation_speed));
    plhs[0] = qd_mex_arrayant2struct(arrayant_out);
}
