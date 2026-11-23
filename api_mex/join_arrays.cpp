// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# JOIN_ARRAYS
Join the elements of two antenna arrays into a new array

- Combines two arrays with identical sampling grids, e.g. the drivers of a loudspeaker; `name` and `center_freq` are
  taken from the first array
- The ports stay separate: elements of the first array connect only to its ports, and likewise for the second:<br><br>
   | Element \ Port        | P1…Pp1 (first) | Pp1+1…Pp1+p2 (second) |
   | --------------------- | :------------: | :-------------------: |
   | E1…En1 (first)        |    W1 block    |           0           |
   | En1+1…En1+n2 (second) |       0        |       W2 block        |

## Usage:
```
arrayant_out = quadriga_lib.join_arrays( arrayant_in1, arrayant_in2 );
```

## Inputs:
- **`arrayant_in1`** — First array; arrayant struct ([[generate_antenna]]), or 1xN struct array for a multi-frequency
  array; its elements come first in the output
- **`arrayant_in2`** — Second array with the same sampling grids; a struct array needs as many entries as the first and
  matching `center_freq` values at each index

## Output:
- **`arrayant_out`** — New arrayant with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports, in the form of
  the inputs

## See also:
- [[copy_element]] (duplicates an element within one array)
- [[generate_speaker]] (creates the driver models to join)
- [[qdant_write]] (stores the joined model)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    qd_mex_check_struct(prhs[0], "arrayant_in1");
    qd_mex_check_struct(prhs[1], "arrayant_in2");

    size_t n_freq1 = (size_t)mxGetNumberOfElements(prhs[0]);
    size_t n_freq2 = (size_t)mxGetNumberOfElements(prhs[1]);

    if (n_freq1 == 0 || n_freq2 == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'arrayant_in1' and 'arrayant_in2' cannot be empty.");

    if (n_freq1 != n_freq2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'arrayant_in1' and 'arrayant_in2' must have the same number of frequency entries.");

    auto ant1 = quadriga_lib::arrayant<double>();
    auto ant2 = quadriga_lib::arrayant<double>();
    auto ant1_multi = std::vector<quadriga_lib::arrayant<double>>();
    auto ant2_multi = std::vector<quadriga_lib::arrayant<double>>();

    if (n_freq1 > 1)
    {
        ant1_multi = qd_mex_struct2arrayant_multi(prhs[0], false, false, "arrayant_in1");
        ant2_multi = qd_mex_struct2arrayant_multi(prhs[1], false, false, "arrayant_in2");
    }
    else
    {
        ant1 = qd_mex_struct2arrayant(prhs[0], false, false, "arrayant_in1");
        ant2 = qd_mex_struct2arrayant(prhs[1], false, false, "arrayant_in2");
    }

    auto ant_out = quadriga_lib::arrayant<double>();
    auto ant_out_multi = std::vector<quadriga_lib::arrayant<double>>();

    if (n_freq1 > 1)
        CALL_QD(ant_out_multi = quadriga_lib::join_arrays<double>(ant1_multi, ant2_multi));
    else
        CALL_QD(ant_out = quadriga_lib::join_arrays(ant1, ant2));

    plhs[0] = (n_freq1 > 1) ? qd_mex_arrayant2struct_multi(ant_out_multi) : qd_mex_arrayant2struct(ant_out);
}