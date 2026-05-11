// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_COPY_ELEMENT
Create copies of array antenna elements

- Copies a source element to one or more destination slots within an arrayant
- Array is resized if any destination index exceeds the current number of elements
- Coupling matrix entries for newly added elements are set to identity; existing coupling is preserved
- Supports multi-frequency arrayant models: when `arrayant_in` is a struct array, the same copy is
  applied to every entry and a struct array of equal size is returned
- If `source_element` is a vector, `dest_element` must have the same length; copies are performed
  pairwise as `source_element(i)` to `dest_element(i)`

## Usage:
```
arrayant_out = quadriga_lib.arrayant_copy_element( arrayant_in, source_element, dest_element );
```

## Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in
  [[arrayant_generate]]; a struct array may contain a frequency-dependent model
- **`source_element`** — Index of the source element(s); 1-based; uint64; scalar or `[n_copy]`
- **`dest_element`** — Index of the destination element(s); 1-based; uint64; scalar or `[n_copy]`;
  if `source_element` is a vector, must have the same length

## Output Arguments:
- **`arrayant_out`** — Struct containing the modified arrayant data; same field layout as
  `arrayant_in`; a struct array of equal size is returned for multi-frequency input
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs != 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    if (!mxIsStruct(prhs[0]))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant_in' must be a struct.");

    // Assemble array antenna object (single or multi-frequency)
    auto ant = quadriga_lib::arrayant<double>();
    auto ant_multi = std::vector<quadriga_lib::arrayant<double>>();
    size_t n_freq = (size_t)mxGetNumberOfElements(prhs[0]);

    if (n_freq > 1)
        ant_multi = qd_mex_struct2arrayant_multi(prhs[0], true, true);
    else
        ant = qd_mex_struct2arrayant(prhs[0], true, true);

    // Read source and destination indices (1-based -> 0-based)
    arma::uvec source = qd_mex_get_Col<arma::uword>(prhs[1], true);
    arma::uvec dest = qd_mex_get_Col<arma::uword>(prhs[2], true);

    if (source.n_elem == 0 || dest.n_elem == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "'source_element' and 'dest_element' cannot be empty.");

    if (arma::any(source == 0) || arma::any(dest == 0))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'source_element' / 'dest_element' cannot be 0 (1-based index).");

    source -= 1;
    dest -= 1;

    if (source.n_elem > 1 && source.n_elem != dest.n_elem)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "When copying multiple elements, 'source_element' and 'dest_element' must have the same length.");

    // Dispatch
    if (n_freq > 1)
    {
        if (source.n_elem == 1)
            CALL_QD(quadriga_lib::arrayant_copy_element_multi(ant_multi, source.at(0), dest));
        else
            for (arma::uword i = 0; i < source.n_elem; ++i)
                CALL_QD(quadriga_lib::arrayant_copy_element_multi(ant_multi, source.at(i), dest.at(i)));
    }
    else
    {
        if (source.n_elem == 1)
            CALL_QD(ant.copy_element(source.at(0), dest));
        else
            for (arma::uword i = 0; i < source.n_elem; ++i)
                CALL_QD(ant.copy_element(source.at(i), dest.at(i)));
    }

    // Output as struct (or struct array for multi-frequency)
    if (nlhs == 1)
        plhs[0] = (n_freq > 1) ? qd_mex_arrayant2struct_multi(ant_multi) : qd_mex_arrayant2struct(ant);
}
