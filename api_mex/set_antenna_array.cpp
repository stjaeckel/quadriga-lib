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

- Sets the antenna of Tx nodes or Rx nodes from an array, e.g. a measured pattern read with [[qdant_read]]
- Without `center_freq`, the array applies to all bands; with `center_freq`, it is the antenna of that band
- Stores the array, which [[layout_csv_write]] exports to a QDANT file

## Usage:
```
layout_out = quadriga_lib.set_antenna_array( layout_in, array, rx, i_node, center_freq );
```

## Inputs:
- **`layout_in`** — Layout struct, fields in [[layout_csv_read]]; its antenna lists must fit its nodes and frequencies
- **`array`** — Valid single-frequency array, 1x1 struct, fields as in [[generate_antenna]]
- **`rx`** — `false`: set the antenna of Tx nodes (`layout_in.tx`); `true`: of Rx nodes (`layout_in.rx`); default:
  `false`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx`, 1-based; `[]` = all of them; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna_array( layout, quadriga_lib.qdant_read( 'handset.qdant' ), true );
```

## Notes:
- **Bands and antenna lists**: as in [[set_antenna]]
- This is the second form of the C++ function `set_antenna`, which takes an array; it has its own name in MATLAB,
  because MATLAB does not overload by type. The C++ in/out argument `layout` is split into `layout_in` / `layout_out`,
  so that the input stays unchanged

## See also:
- [[set_antenna]] (sets the antenna from the arguments of generate_antenna)
- [[qdant_read]] (reads an array from a QDANT file)
- [[layout_csv_write]] (exports the array)
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
    const bool rx = (nrhs < 3) ? false : qd_mex_get_scalar<bool>(prhs[2], "rx", false);
    const arma::uvec i_node = (nrhs < 4) ? arma::uvec() : qd_mex_get_index_Col<arma::uword>(prhs[3], "i_node"); // 1-based -> 0-based
    const double center_freq = (nrhs < 5) ? NAN : qd_mex_get_scalar<double>(prhs[4], "center_freq", NAN);

    CALL_QD(quadriga_lib::set_antenna(layout, array, rx, i_node, center_freq));

    plhs[0] = qd_mex_layout2struct(layout);
}
