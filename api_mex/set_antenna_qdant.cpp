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
# SET_ANTENNA_QDANT
Set the antenna of layout nodes to an entry of a QDANT file

- Sets the antenna of Tx nodes or Rx nodes to an array stored in a QDANT file, e.g. a measured or simulated pattern
- The layout refers to the file: [[layout_csv_write]] writes the reference (the columns `antenna` = `qdant`,
  `qdant_file`, `qdant_id`), relative to the CSV file, and exports no copy of the array
- The array is used as stored, for all bands; the rules for bands and the antenna lists are those of [[set_antenna]]

## Usage:
```
layout_out = quadriga_lib.set_antenna_qdant( layout_in, qdant_file, qdant_id, rx, i_node, center_freq, evaluate );
```

## Inputs:
- **`layout_in`** — Layout struct, fields in [[layout_csv_read]]; its antenna lists must fit its nodes and frequencies
- **`qdant_file`** — Existing QDANT file, absolute or relative to the current folder
- **`qdant_id`** — 1-based entry of the file; default: 1
- **`rx`** — `false`: set the antenna of Tx nodes (`layout_in.tx`); `true`: of Rx nodes (`layout_in.rx`); default:
  `false`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx`, 1-based; `[]` = all of them; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN
- **`evaluate`** — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none; default:
  `true`

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna_qdant( layout, 'measured/handset.qdant', 2, true );
```

## Notes:
- **Path**: the file is stored as a normalised absolute path
- **Errors**: a missing or unreadable file raises an error; without evaluation, an entry that the file does not have
  raises an error when the antennas are evaluated
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## See also:
- [[set_antenna]] (sets the antenna from the arguments of generate_antenna)
- [[set_antenna_array]] (sets an array)
- [[qdant_read]] (reads the entries of a QDANT file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], true, "layout_in");
    const std::string qdant_file = qd_mex_get_string(prhs[1], "", "qdant_file");
    const arma::uword qdant_id = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "qdant_id", 1); // A file ID, not converted
    const bool rx = (nrhs < 4) ? false : qd_mex_get_scalar<bool>(prhs[3], "rx", false);
    const arma::uvec i_node = (nrhs < 5) ? arma::uvec() : qd_mex_get_index_Col<arma::uword>(prhs[4], "i_node"); // 1-based -> 0-based
    const double center_freq = (nrhs < 6) ? NAN : qd_mex_get_scalar<double>(prhs[5], "center_freq", NAN);
    const bool evaluate = (nrhs < 7) ? true : qd_mex_get_scalar<bool>(prhs[6], "evaluate", true);

    CALL_QD(quadriga_lib::set_antenna_qdant(layout, qdant_file, qdant_id, rx, i_node, center_freq, evaluate));

    plhs[0] = qd_mex_layout2struct(layout);
}
