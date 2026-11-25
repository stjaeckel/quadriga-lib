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

- Generates the arrays `antenna` of a layout from its antenna descriptions `antenna_args`, for the entries that are not
  evaluated yet, e.g. after [[layout_csv_read]] or [[set_antenna]] with the evaluation turned off
- With `center_freq`, sets new simulation frequencies and generates all antennas for them

## Usage:
```
layout_out = quadriga_lib.eval_antennas( layout_in, center_freq );
```

## Inputs:
- **`layout_in`** — Layout struct, fields in [[layout_csv_read]]
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; replaces `layout_in.center_freq`
  and regenerates all descriptions; `[]` = keep `layout_in.center_freq`; default: `[]`; `[n_freq]`

## Outputs:
- **`layout_out`** — The layout with evaluated `antenna`

## Notes:
- **Antenna descriptions**: a line of `antenna_args` holds the cells of the antenna columns `antenna_cols` as in the
  [Layout CSV files](formats.html#layout-csv-antennas). Nodes with equal lines share one antenna; `i_antenna` of a node
  is its index, 0 the generic antenna. No line or an empty line is an array without description (set with
  [[set_antenna_array]]); a line without antenna type is a band that is not set yet
- **Evaluation**: a line for all bands is generated once per band at its frequency, so that an array spaced in
  wavelengths keeps its spacing (an array of co-located elements is stored once for all bands); a line of a band at the
  frequency of that band; a `qdant` line is read as stored. Until an antenna is evaluated (its entry of `antenna` holds
  no arrays, or an array without elements), its nodes use the generic antenna
- **New frequencies**: arrays without a description are kept and must fit the new list
- **Errors**: a band without description and an array spaced in wavelengths in a layout without frequencies raise an
  error; the message names the node
- This is the member function `eval_antennas` of the C++ layout class; its in/out argument `layout` is split into
  `layout_in` / `layout_out`, so that the input stays unchanged

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
