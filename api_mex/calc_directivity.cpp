// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# CALC_DIRECTIVITY
Calculate the directivity in dBi of the elements of an antenna array

- Measures how strongly an element concentrates its power: `D = 10 log10(P_max / P_mean)` with the peak radiation
  intensity `P_max` and its mean `P_mean` over the sphere; an isotropic radiator has 0 dBi
- Calculated per element, not per port: the port weights are ignored

## Usage:
```
directivity_dBi = quadriga_lib.calc_directivity( arrayant, i_element );
```

## Inputs:
- **`arrayant`** — Arrayant struct ([[generate_antenna]]), or 1xN struct array for a multi-frequency array with `n_freq`
  entries
- **`i_element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements

## Output:
- **`directivity_dBi`** — Directivity in dBi; `[n_out, n_freq]`

## Notes:
- **Scale**: the result does not depend on the scale of the pattern; an element whose pattern is zero everywhere gives 0
  dBi

## See also:
- [[calc_beamwidth]] (beamwidth and pointing angles of the elements)
- [[apply_port_weights]] (combines the elements into ports, to measure the directivity of the ports)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    if (!mxIsStruct(prhs[0]) || mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant' must be an arrayant struct or struct array.");

    // Element indices: 1-based MATLAB -> 0-based C++ (empty = all elements)
    const arma::uvec i_element = (nrhs < 2) ? arma::uvec() : qd_mex_get_index_Col(prhs[1], "i_element");

    // A single struct is a one-entry vector; the output is [n_out, n_freq] either way
    const auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0], false, false, "arrayant"); // Validated by the C++ function
    arma::mat directivity_dBi;
    CALL_QD(directivity_dBi = quadriga_lib::calc_directivity(ant_multi, i_element));
    plhs[0] = qd_mex_copy2matlab(&directivity_dBi);
}