// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_CONCAT
Concatenate two arrayant structs into a single one

- Concatenates all elements from `arrayant_in2` onto `arrayant_in1` along the element dimension; 
  `element_pos` matrices are joined horizontally
- Both inputs must share identical azimuth and elevation sampling grids
- Coupling is assembled block-diagonally: elements from `arrayant_in1` connect only to ports from 
  `arrayant_in1`, elements from `arrayant_in2` only to ports from `arrayant_in2`
- `center_freq` and `name` are inherited from `arrayant_in1`
- Supports multi-frequency arrayant models: when both inputs are struct arrays, they must have the 
  same number of entries and matching `center_freq` at each index; concatenation is performed per 
  entry and a struct array of equal size is returned
- Output struct shape matches the input shape (scalar struct -> scalar struct, struct array -> struct array)

## Usage:
```
arrayant_out = quadriga_lib.arrayant_concat( arrayant_in1, arrayant_in2 );
```

## Inputs:
- **`arrayant_in1`** — Struct containing the first arrayant data; field layout as documented in 
  [[arrayant_generate]]; a struct array may contain a frequency-dependent  model
- **`arrayant_in2`** — Struct containing the second arrayant data; must match the sampling grids of
  `arrayant_in1`, and for multi-frequency the entry count and `center_freq` per entry

## Outputs:
- **`arrayant_out`** — Struct containing the combined arrayant data; same field layout as the inputs; 
  a struct array of equal size is returned for multi-frequency input
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    if (!mxIsStruct(prhs[0]))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant_in1' must be a struct.");
        
    if (!mxIsStruct(prhs[1]))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant_in2' must be a struct.");

    size_t n_freq1 = (size_t)mxGetNumberOfElements(prhs[0]);
    size_t n_freq2 = (size_t)mxGetNumberOfElements(prhs[1]);

    if (n_freq1 == 0 || n_freq2 == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "'arrayant_in1' and 'arrayant_in2' cannot be empty.");

    if (n_freq1 != n_freq2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "'arrayant_in1' and 'arrayant_in2' must have the same number of entries.");

    // Assemble array antenna objects (single or multi-frequency)
    auto ant1 = quadriga_lib::arrayant<double>();
    auto ant2 = quadriga_lib::arrayant<double>();
    auto ant1_multi = std::vector<quadriga_lib::arrayant<double>>();
    auto ant2_multi = std::vector<quadriga_lib::arrayant<double>>();

    if (n_freq1 > 1)
    {
        ant1_multi = qd_mex_struct2arrayant_multi(prhs[0], false);
        ant2_multi = qd_mex_struct2arrayant_multi(prhs[1], false);
    }
    else
    {
        ant1 = qd_mex_struct2arrayant(prhs[0], false);
        ant2 = qd_mex_struct2arrayant(prhs[1], false);
    }

    // Output containers
    auto ant_out = quadriga_lib::arrayant<double>();
    auto ant_out_multi = std::vector<quadriga_lib::arrayant<double>>();

    // Dispatch
    if (n_freq1 > 1)
        CALL_QD(ant_out_multi = quadriga_lib::arrayant_concat_multi<double>(ant1_multi, ant2_multi));
    else
        CALL_QD(ant_out = ant1.append(&ant2));

    // Output as struct (or struct array for multi-frequency)
    if (nlhs == 1)
        plhs[0] = (n_freq1 > 1) ? qd_mex_arrayant2struct_multi(ant_out_multi) : qd_mex_arrayant2struct(ant_out);
}
