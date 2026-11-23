// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# APPLY_PORT_WEIGHTS
Apply the port weights of an antenna array: one pattern per port

- Evaluates the element patterns on the output grid, applies the phase shift of each element position (which
  moves every pattern to the array origin) and sums the elements with the port weights
- The result has one element per port, identity port weights and all element positions at the origin
- Speeds up MIMO channel computation; useful for beamforming in 5G systems and network planning
- The element positions add a phase that depends on frequency and `propagation_speed`; the default speed of light
  applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [[generate_speaker]])

## Usage:
```
arrayant_out = quadriga_lib.apply_port_weights( arrayant_in, freq_new, azimuth_grid_new, ...
    elevation_grid_new, propagation_speed );
```

## Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [[generate_antenna]];
  a struct array may contain a frequency-dependent model
- **`freq_new`** — Alternative frequency (grid) in Hz; if provided, the combined pattern is
  recomputed (interpolated) at each requested frequency; default: `[]` (each input entry's `center_freq` unchanged)
- **`azimuth_grid_new`** — Alternative azimuth grid in rad; in [-pi, pi]; strictly increasing; default: `[]` (input grid)
- **`elevation_grid_new`** — Alternative elevation grid in rad; in [-pi/2, pi/2]; strictly increasing; default: `[]` (input grid)
- **`propagation_speed`** — Wave propagation speed in m/s; positive and finite; use about 343 for acoustics;
  default: 299792458

## Output:
- **`arrayant_out`** — Arrayant struct (single-frequency result) or struct array (multi-frequency
  result); field layout as documented in [[generate_antenna]]. Single struct when both input and
  `freq_new` describe a single frequency; struct array of size `numel(freq_new)` (or
  `numel(arrayant_in)` when `freq_new` is omitted) otherwise.

## See also:
- [[generate_antenna]] (for field layout in the arrayant struct)
- [[rotate_pattern]] (for changing the orientation of elements before combining)
- [[calc_beamwidth]] (calculates the beam width of antenna arrays)
- [[calc_directivity]] (directivity in dBi of antenna array elements)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    if (!mxIsStruct(prhs[0]) || mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant_in' must be an arrayant struct or struct array.");

    const arma::vec freq_new = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1], false, "freq_new");
    const arma::vec az = (nrhs < 3) ? arma::vec() : qd_mex_get_Col<double>(prhs[2], false, "azimuth_grid_new");
    const arma::vec el = (nrhs < 4) ? arma::vec() : qd_mex_get_Col<double>(prhs[3], false, "elevation_grid_new");
    const double propagation_speed = (nrhs < 5) ? 299792458.0 : qd_mex_get_scalar<double>(prhs[4], "propagation_speed", 299792458.0);

    if (freq_new.n_elem > 1 || mxGetNumberOfElements(prhs[0]) > 1) // Multi-frequency
    {
        const auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0], false, false, "arrayant_in"); // Validated by the C++ function
        std::vector<quadriga_lib::arrayant<double>> out;
        CALL_QD(out = quadriga_lib::apply_port_weights(ant_multi, az, el, freq_new, propagation_speed));
        plhs[0] = (out.size() == 1) ? qd_mex_arrayant2struct(out[0]) : qd_mex_arrayant2struct_multi(out);
        return;
    }

    auto ant = qd_mex_struct2arrayant(prhs[0], false, false, "arrayant_in"); // Validated by the C++ function
    if (freq_new.n_elem == 1) // Scalar frequency override
        ant.center_freq = freq_new[0];

    quadriga_lib::arrayant<double> arrayant_out;
    CALL_QD(arrayant_out = quadriga_lib::apply_port_weights(ant, az, el, propagation_speed));
    plhs[0] = qd_mex_arrayant2struct(arrayant_out);
}
