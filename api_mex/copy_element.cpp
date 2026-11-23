// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# COPY_ELEMENT
Create copies of antenna array elements

- Copies a source element to one or more destination slots within an arrayant
- Array is resized if any destination index exceeds the current number of elements
- Port weights of newly added elements are set to identity; existing port weights are preserved
- Supports multi-frequency arrayant models: when `arrayant_in` is a struct array, the same copy is
  applied to every entry and a struct array of equal size is returned
- If `i_source` is a vector, `i_target` must have the same length; copies are performed
  pairwise as `i_source(i)` to `i_target(i)`

## Usage:
```
arrayant_out = quadriga_lib.copy_element( arrayant_in, i_source, i_target );
```

## Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in
  [[generate_antenna]]; a struct array may contain a frequency-dependent model
- **`i_source`** — Index of the source element(s); 1-based; scalar or `[n_copy]`
- **`i_target`** — Index of the destination element(s); 1-based; scalar or `[n_copy]`;
  if `i_source` is a vector, must have the same length

## Output:
- **`arrayant_out`** — Struct containing the modified arrayant data; same field layout as
  `arrayant_in`; a struct array of equal size is returned for multi-frequency input
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    qd_mex_check_struct(prhs[0], "arrayant_in");

    auto ant = quadriga_lib::arrayant<double>();
    auto ant_multi = std::vector<quadriga_lib::arrayant<double>>();
    size_t n_freq = (size_t)mxGetNumberOfElements(prhs[0]);

    if (n_freq > 1)
        ant_multi = qd_mex_struct2arrayant_multi(prhs[0], true, true, "arrayant_in");
    else
        ant = qd_mex_struct2arrayant(prhs[0], true, true, "arrayant_in");

    // Read source and destination indices (1-based -> 0-based)
    const arma::uvec source = qd_mex_get_index_Col(prhs[1], "i_source");
    const arma::uvec dest = qd_mex_get_index_Col(prhs[2], "i_target");

    if (source.n_elem == 0 || dest.n_elem == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'i_source' and 'i_target' cannot be empty.");

    if (source.n_elem > 1 && source.n_elem != dest.n_elem)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'i_source' and 'i_target' must have the same length when 'i_source' has more than one entry.");

    if (n_freq > 1)
    {
        if (source.n_elem == 1)
            CALL_QD(quadriga_lib::copy_element(ant_multi, source.at(0), dest));
        else
            for (arma::uword i = 0; i < source.n_elem; ++i)
                CALL_QD(quadriga_lib::copy_element(ant_multi, source.at(i), arma::uvec{dest.at(i)}));
    }
    else
    {
        if (source.n_elem == 1)
            CALL_QD(quadriga_lib::copy_element(ant, source.at(0), dest));
        else
            for (arma::uword i = 0; i < source.n_elem; ++i)
                CALL_QD(quadriga_lib::copy_element(ant, source.at(i), arma::uvec{dest.at(i)}));
    }

    plhs[0] = (n_freq > 1) ? qd_mex_arrayant2struct_multi(ant_multi) : qd_mex_arrayant2struct(ant);
}
