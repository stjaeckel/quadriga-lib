// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# COPY_ELEMENT
Copy a single element of an antenna array to one or more destination slots

- Copies the pattern and the position of an element, e.g. to build an array from a single element; the port weights of
  existing elements do not change
- A destination beyond the last element enlarges the array. Each added element gets its own new port with weight 1;
  added elements that are not a destination have a zero pattern and position

## Usage:
```
arrayant_out = quadriga_lib.copy_element( arrayant_in, i_source, i_target );
```

## Inputs:
- **`arrayant_in`** — Arrayant struct ([[generate_antenna]]), or 1xN struct array for a multi-frequency array; the same
  copy is made in every entry
- **`i_source`** — Index of the element to copy, 1-based, at most `n_elements`; scalar, or `[n_copy]` for pairwise
  copies from `i_source(k)` to `i_target(k)`
- **`i_target`** — Destination indices, 1-based, not empty; a scalar `i_source` is copied to each of them, a vector
  `i_source` needs the same length; `[n_copy]`

## Output:
- **`arrayant_out`** — Arrayant with the copies, in the form of `arrayant_in`

## See also:
- [[join_arrays]] (joins two arrays)
- [[generate_antenna]] (creates the element to copy)
- [[rotate_pattern]] (turns single copies, through `i_element`)
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