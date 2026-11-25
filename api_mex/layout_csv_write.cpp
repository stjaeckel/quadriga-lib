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

- Writes the Tx nodes to the Tx file and the Rx nodes to the Rx file in the
  [Layout CSV format](formats.html#layout_csv); reading the files back gives the same layout within the written digits
  (9 significant digits, positions 1 µm, times 1 ns), except for `name` and `links`, which are not stored
- Antennas without a description, e.g. a measured array set with [[set_antenna_array]], are written to QDANT files next
  to the CSV file
- A failed call leaves the existing files unchanged

## Usage:
```
quadriga_lib.layout_csv_write( fn_tx, fn_rx, layout, compact );
```

## Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; `''` = not written; different files (also if the paths are spelled
  differently)
- **`layout`** — Layout struct, must be valid; fields in [[layout_csv_read]]
- **`compact`** — `true`: cells that the reader fills in are left empty; `false`: every cell is written; default: `true`

## Notes:
- **Rules of the writer**: the columns written, the number format, the compact form, the names of the QDANT files and
  what cannot be written are described in [Writing](formats.html#layout-csv-writing)
- **QDANT files**: one file `<stem>_<k>.qdant` per antenna without description, with one entry per band. Files of this
  name pattern next to a written CSV file that the new files do not refer to are deleted. The arrays must be evaluated:
  a layout whose `antenna` does not hold them raises an error. An antenna set from a QDANT file ([[set_antenna]]) is
  written as a reference to its file, not exported

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
