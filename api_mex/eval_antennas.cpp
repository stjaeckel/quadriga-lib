// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_mex_layout.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# EVAL_ANTENNAS
Generate the antenna arrays of a layout from the antenna lines

- Generates the arrays of `antenna` that are not evaluated yet from `antenna_args`; a line without `center_freq` once per simulation frequency, so that an array spaced in wavelengths keeps its spacing in each band (co-located elements give one array for all bands), a line of a band at its frequency, a `qdant` line by reading its entry as stored
- A frequency list `center_freq` replaces `layout_in.center_freq` and regenerates all lines; arrays without a line are kept and must fit the new list
- Raises an error for a band without description and for an array spaced in wavelengths without a frequency; the message names the node
- This is the member function `eval_antennas` of the C++ layout class; its in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## Usage:
```
layout_out = quadriga_lib.eval_antennas( layout_in, center_freq );
```

## Inputs:
- **`layout_in`** — Layout struct, fields in [[layout_csv_read]]
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `[]` = keep `layout_in.center_freq`; default: `[]`; `[n_freq]`

## Outputs:
- **`layout_out`** — The layout with evaluated `antenna`

## See also:
- [[layout_csv_read]] (reads a layout, optionally with evaluated antennas)
- [[set_antenna]] (sets the antennas of nodes)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], true, "layout_in");
    const arma::vec center_freq = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1], false, "center_freq");

    CALL_QD(layout.eval_antennas(center_freq));

    plhs[0] = qd_mex_layout2struct(layout);
}
