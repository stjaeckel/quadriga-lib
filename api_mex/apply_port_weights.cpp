// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!SECTION_DESC
The functions of this section create, inspect, transform and store antenna and acoustic arrays. They share one data
structure, the arrayant struct (fields in [[generate_antenna]]): complex polarimetric patterns of the elements on an
azimuth and elevation grid, the element positions and the port weights. The channel functions take it as transmit and
receive antenna.<br><br>

**Terms** (full list in the [Glossary](cpp_api.html#glossary)):

- **Element**: one radiator or sensor (antenna element, loudspeaker driver, microphone capsule) with a pattern, a
  position and one row of the port weights
- **Grid**: azimuth φ in [-π, π], measured from +x toward +y, and elevation θ in [-π/2, π/2], measured from the x-y
  plane (TR 38.901 uses the zenith angle 90° - θ), both in rad and strictly increasing. Boresight is +x
- **Pattern**: the direction-dependent far field of an element: F_θ (`e_theta_re/im`) along
  `(sin θ cos φ, sin θ sin φ, -cos θ)`, the zenith-angle unit vector of TR 38.901, which points toward decreasing
  elevation, and F_φ (`e_phi_re/im`) along `(-sin φ, cos φ, 0)`. For an antenna, `|F_θ|² + |F_φ|²` is the linear gain
  over an isotropic radiator
- **Port**: one signal of the array, one column of the port weights
- **Port weights**: the fixed complex feed network `w(i,p)` from element i to port p, `[n_elements, n_ports]`. It is not
  the mutual coupling between elements. Empty means identity (one port per element)
- **Element position**: local Cartesian coordinates in m, `[3, n_elements]`; empty means all at the origin
- **Multi-frequency array**: a 1xN struct array, one entry per `center_freq`, with equal grids, element positions and
  port weight sizes. [[interpolate_pattern]] and [[apply_port_weights]] interpolate over frequency between the entries

The field of port p in a direction is the weighted sum of the element fields, each with the phase of its position,

`F_p = Σ_i w(i,p) · F_i · exp(-j 2π f d_i / c)`

where `d_i = -r · x_i` is the distance of element i at position x_i from the wavefront plane through the origin,
negative for elements closer to the source, r the unit vector toward the source, f the frequency and c the propagation
speed.<br><br>

**Functions by task:**

| Task | Functions |
| ---- | --------- |
| Create | [[generate_antenna]], [[generate_speaker]], [[qdant_read]] |
| Evaluate | [[interpolate_pattern]], [[calc_directivity]], [[calc_beamwidth]] |
| Transform | [[apply_port_weights]], [[rotate_pattern]] |
| Edit and combine | [[join_arrays]], [[copy_element]] |
| Store, show | [[qdant_write]], [[export_pattern_obj]] |

Arrays are stored in the [QDANT file format](formats.html#qdant).
SECTION_DESC!*/

/*!MD
# APPLY_PORT_WEIGHTS
Apply the port weights of an antenna array: one pattern per port

- Combines the elements of an array into one pattern per port, for beamforming and to speed up MIMO channel computation
- Each port pattern is the weighted sum of the element fields from the section overview, evaluated on the output grid
  and including the phase of each element position
- The result has one element per port, identity port weights and all element positions at the origin

## Usage:
```
arrayant_out = quadriga_lib.apply_port_weights( arrayant_in, freq_new, azimuth_grid_new, ...
    elevation_grid_new, propagation_speed );
```

## Inputs:
- **`arrayant_in`** — Arrayant struct ([[generate_antenna]]), or 1xN struct array for a multi-frequency array
- **`freq_new`** — Output frequencies in Hz, positive and finite; a 1x1 struct has one pattern, so that only the phase
  of the element positions follows `freq_new`; `[n_freq_new]`; default: the `center_freq` of the entries, one output per
  entry
- **`azimuth_grid_new`** — Output azimuth grid in rad; default: the input grid (of the first entry)
- **`elevation_grid_new`** — Output elevation grid in rad; default: the input grid (of the first entry)
- **`propagation_speed`** — Wave propagation speed in m/s, positive and finite; 299792458 is the speed of light for
  radio waves, use about 343 for acoustic arrays (e.g. from [[generate_speaker]]); default: 299792458

## Output:
- **`arrayant_out`** — Arrayant struct with `n_ports` elements, each holding the pattern of that port; `name` is kept,
  `center_freq` is the output frequency; a 1x1 struct for one output frequency, otherwise a 1xN struct array with one
  entry per output frequency

## Notes:
- **Frequency interpolation** (struct array): the fields are interpolated over frequency as in [[interpolate_pattern]],
  the port weights spherically between the two entries that bracket the output frequency. Output frequencies outside the
  entries take the nearest entry. The position phase uses the output frequency

## See also:
- [[generate_antenna]] (creates arrays with port weights)
- [[interpolate_pattern]] (evaluates the element patterns without the port weights)
- [[calc_directivity]], [[calc_beamwidth]] (work per element, so they measure the port patterns of the result)
- [[rotate_pattern]] (turns the combined pattern)
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