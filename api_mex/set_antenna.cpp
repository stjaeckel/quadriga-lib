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
# SET_ANTENNA
Set the antenna of layout nodes from the arguments of generate_antenna

- This form stores the arguments of [[generate_antenna]] (see there for their meaning) as an antenna line of the layout, which [[layout_csv_write]] writes as CSV cells; an array set with [[set_antenna_array]] is exported to a QDANT file by [[layout_csv_write]]
- Without `center_freq`, the antenna applies to all bands. With `center_freq`, only that band of the nodes' antenna is set: a frequency that is not in `layout.center_freq` is added, the other bands of the nodes keep their antenna, and nodes that shared the old entry keep it; until all bands of a node are set, the layout is not valid and the functions that check it name the node
- A node set band by band keeps one line per band, also when the lines are equal, so a band added later is reported as not set for it; in a layout with one band, the line of a node set for that band also covers bands added later
- The antenna lists are encoded again after the call as [[layout_csv_read]] composes them: equal lines share one entry, entries are numbered in order of first use (Tx nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- An array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- A failed call leaves the layout unchanged
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## Usage:
```
layout_out = quadriga_lib.set_antenna( layout_in, type, args, rx_side, i_node, center_freq, evaluate );
```

## Inputs:
- **`layout_in`** — Layout struct, fields in [[layout_csv_read]]; its antenna lists must fit its nodes and frequencies
- **`type`** — Antenna type of [[generate_antenna]], e.g. `'3GPP'`, `'dipole'`, or `'none'` (no antenna, for all bands, without `args`)
- **`args`** — Struct with the arguments of [[generate_antenna]] as fields: `res`, `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers, vectors or matrices (`steer`, `beams` row by row as there), `separate_beams` and `apply_weights` as 0 or 1; a field that the type does not use raises an error; `pattern` is not available (an array built with it is set with [[set_antenna_array]]); the field `center_freq` is the same as the argument; e.g. `struct( 'panel', [4 4 1 2], 'pol', 3 )`; default: `[]` (none)
- **`rx_side`** — `false`: nodes of `layout_in.tx`; `true`: nodes of `layout_in.rx`; default: `false`
- **`i_node`** — Indices of the nodes of that side, 1-based; `[]` = all nodes of the side; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN
- **`evaluate`** — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none, the nodes of new antennas use the generic antenna until [[eval_antennas]] is called; default: `true`

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna( layout, '3GPP', struct( 'panel', [4 4 1 2], 'pol', 3 ), false, [], 3.5e9 );
```

## See also:
- [[set_antenna_array]] (sets an evaluated array)
- [[generate_antenna]] (the meaning of the arguments)
- [[layout_csv_write]] (writes the lines and exports the arrays)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], true, "layout_in");
    const std::string type = qd_mex_get_string(prhs[1], "", "type");
    const auto args = (nrhs < 3) ? std::unordered_map<std::string, arma::mat>() : qd_mex_layout_args(prhs[2], "args");
    const bool rx_side = (nrhs < 4) ? false : qd_mex_get_scalar<bool>(prhs[3], "rx_side", false);
    const arma::uvec i_node = (nrhs < 5) ? arma::uvec() : qd_mex_get_index_Col<arma::uword>(prhs[4], "i_node"); // 1-based -> 0-based
    const double center_freq = (nrhs < 6) ? NAN : qd_mex_get_scalar<double>(prhs[5], "center_freq", NAN);
    const bool evaluate = (nrhs < 7) ? true : qd_mex_get_scalar<bool>(prhs[6], "evaluate", true);

    CALL_QD(quadriga_lib::set_antenna(layout, type, args, rx_side, i_node, center_freq, evaluate));

    plhs[0] = qd_mex_layout2struct(layout);
}
