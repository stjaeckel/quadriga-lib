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
# LAYOUT_CSV_WRITE
Write a layout to the Tx and Rx CSV files

- Writes only columns with content other than the default, in the order of the format, unknown columns last; angles in degrees, numbers with up to 15 significant digits (16 or 17 where 15 do not read back); reading the files gives the same layout within these digits, except for `name` and `links`, which are not stored
- With `compact` the cells that the reader repeats from the previous row of the node (held values) or sets to the default (fixed values, which are written on the first row only) are left empty; an extra sector (sector above 0 of a site that has a sector 0) has all its cells in its first row, because an empty cell would be the value of sector 0
- Antennas are written as their CSV lines, with each `qdant_file` relative to the folder of its CSV file. If no antenna line carries the frequencies of the layout, they are declared on the first node: `center_freq` on its first row and one band row per further frequency
- All files are composed first and written to temporary files next to the targets; each target is replaced by a rename only after all files were composed and written, so an error in the layout, the composition or the writing leaves all files unchanged (a failing rename of the second file is not undone for the first)
- Antenna bands that are evaluated arrays without a line (set with [[set_antenna_array]]) are written to QDANT files next to the first written CSV file that uses the antenna, named `<stem>_<k>.qdant` (`<stem>` is the name of the CSV file without `.csv`, k = 1, 2, ... per CSV file, skipping names that `qdant` lines of the layout use), one file per antenna with one entry per exported band, and become `qdant` lines. Afterwards, the files `<stem>_<k>.qdant` next to a written CSV file that the new files do not reference are deleted; files of other names are never touched. These arrays must be evaluated: a layout whose `antenna` does not hold them raises an error
- The `seed`, `layout_seed` and `wrap` of the layout are stored in the Tx file; a Tx file for a layout without Tx nodes that has them raises an error. Not expressible in the format and therefore rejected: a scenario that returns to none after a scenario was set, a terminal without a pair in `pairing` (an empty cell means all sites). An extra sector without antenna next to a sector 0 with one is written with the antenna `none`

## Usage:
```
quadriga_lib.layout_csv_write( fn_tx, fn_rx, layout, compact );
```

## Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; `''` = not written; different files (also if the paths are spelled differently)
- **`layout`** — Layout struct, must be valid; fields in [[layout_csv_read]]
- **`compact`** — `true`: empty cells where the format allows; `false`: every cell is written; default: `true`

## See also:
- [[layout_csv_read]] (reads a layout)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs; // No outputs

    if (nrhs < 3 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn_tx = qd_mex_get_string(prhs[0], "", "fn_tx");
    const std::string fn_rx = qd_mex_get_string(prhs[1], "", "fn_rx");
    const quadriga_lib::layout layout = qd_mex_struct2layout(prhs[2], false, "layout");
    const bool compact = (nrhs < 4) ? true : qd_mex_get_scalar<bool>(prhs[3], "compact", true);

    CALL_QD(quadriga_lib::layout_csv_write(fn_tx, fn_rx, layout, compact));
}
