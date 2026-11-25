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
# SET_ANTENNA_ARRAY
Set the antenna of layout nodes from an evaluated array

- Stores an array, e.g. a measurement read with [[qdant_read]], which [[layout_csv_write]] exports to a QDANT file; for the antenna lists of the layout, the bands and the renumbering of `i_antenna` see [[set_antenna]]
- Without `center_freq`, the array applies to all bands. With `center_freq`, only that band of the nodes' antenna is set: a frequency that is not in `layout_in.center_freq` is added, the other bands of the nodes keep their antenna
- The C++ function `set_antenna` with an array argument has this name in MATLAB, because MATLAB does not overload by type; the in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## Usage:
```
layout_out = quadriga_lib.set_antenna_array( layout_in, array, rx_side, i_node, center_freq );
```

## Inputs:
- **`layout_in`** — Layout struct, fields in [[layout_csv_read]]; its antenna lists must fit its nodes and frequencies
- **`array`** — Valid single-frequency array, 1x1 struct, fields as in [[generate_antenna]]
- **`rx_side`** — `false`: nodes of `layout_in.tx`; `true`: nodes of `layout_in.rx`; default: `false`
- **`i_node`** — Indices of the nodes of that side, 1-based; `[]` = all nodes of the side; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = the same array for all bands; default: NaN

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna_array( layout, quadriga_lib.qdant_read( 'handset.qdant' ), true );
```

## See also:
- [[set_antenna]] (sets the antenna from the arguments of generate_antenna)
- [[qdant_read]] (reads an array from a QDANT file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], true, "layout_in");
    qd_mex_check_struct(prhs[1], "array", true); // A multi-frequency array is not accepted
    const auto array = qd_mex_struct2arrayant(prhs[1], false, false, "array");   // The library validates the array
    const bool rx_side = (nrhs < 3) ? false : qd_mex_get_scalar<bool>(prhs[2], "rx_side", false);
    const arma::uvec i_node = (nrhs < 4) ? arma::uvec() : qd_mex_get_index_Col<arma::uword>(prhs[3], "i_node"); // 1-based -> 0-based
    const double center_freq = (nrhs < 5) ? NAN : qd_mex_get_scalar<double>(prhs[4], "center_freq", NAN);

    CALL_QD(quadriga_lib::set_antenna(layout, array, rx_side, i_node, center_freq));

    plhs[0] = qd_mex_layout2struct(layout);
}
