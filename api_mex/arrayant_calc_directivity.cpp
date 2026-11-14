// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_CALC_DIRECTIVITY
Calculates the directivity in dBi of array antenna elements

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Calculated per element, not per port; ignores element coupling
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi

## Usage:
```
directivity_dBi = quadriga_lib.arrayant_calc_directivity( arrayant, i_element );
```

## Inputs:
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array may contain a frequency-dependent model
- **`i_element`** — Element indices, 1-based, duplicates allowed; uint64; `[n_out]`; default: all elements

## Output:
- **`directivity_dBi`** — Directivity in dBi; `[n_out, n_freq]`; with `n_out = n_elements` when `i_element` is omitted/empty

## See also:
- [[arrayant_combine_pattern]] (to apply element coupling before calculating directivity)
- [[arrayant_calc_beamwidth]] (calculates the beam width of array antennas)
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
    arma::uvec i_element = (nrhs < 2) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[1], true, "i_element");
    if (arma::any(i_element == 0))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'i_element' cannot be 0 (1-based index).");
    i_element -= 1;

    // A single struct is a one-entry vector; the output is [n_out, n_freq] either way. The output is also set
    // without output argument (it goes to ans).
    const auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0]); // Validated by the C++ function
    arma::mat directivity_dBi;
    CALL_QD(quadriga_lib::arrayant_calc_directivity_multi(ant_multi, directivity_dBi, &i_element));
    plhs[0] = qd_mex_copy2matlab(&directivity_dBi);
}
