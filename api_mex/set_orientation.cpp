// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex_types.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# SET_ORIENTATION
Set the orientation of nodes from their motion, or to a given value

- Turns the antennas of tracks along their motion, e.g. for a UT held in the direction of travel, or sets all rows of
  the nodes to a fixed orientation
- `force = false`: tracks get bank 0 and the tilt and heading of the steps between their rows, plus `orientation` as an
  offset; point drops keep their orientation plus the offset
- `force = true`: every row of the nodes gets exactly `orientation`

## Usage:
```
nodes_out = quadriga_lib.set_orientation( nodes_in, orientation, i_node, force );
```

## Inputs:
- **`nodes_in`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`); fields in [[layout_csv_read]]
- **`orientation`** — Bank, tilt, heading in [rad], 3 finite values; `[]` = 0; default: `[]`
- **`i_node`** — Node indices, 1-based; `[]` = all nodes; default: `[]`
- **`force`** — `false`: orientation from the motion of tracks, plus `orientation` as offset; `true`: `orientation` on
  all rows; default: `false`

## Outputs:
- **`nodes_out`** — The nodes with the new orientations

## Example:
```
layout.rx = quadriga_lib.set_orientation( layout.rx );                    % The antennas look along the motion
layout.rx = quadriga_lib.set_orientation( layout.rx, [0 0 0.5] );         % The same, turned by 0.5 rad
```

## Notes:
- **Tracks** (`force = false`): forward differences between the rows, the last step repeated, or for a closed track the
  step to the first row, as `qd_track.calc_orientation` of the NG library does; the angles are wrapped to [-π, π]. A
  step of zero length keeps the previous orientation, at the start the next one that exists. A vertical step has the
  tilt -90° (up) or 90° (down) and keeps the previous heading (a deliberate deviation from NG, where it is 0). A rising
  track has a negative tilt; a track that does not move keeps its orientation (plus the offset)
- **Point drops** (`force = false`): without an offset, the orientation is not touched; random orientations are not
  generated here
- **Mounted antennas**: `force = false` leaves nodes with `relative = 1` unchanged, offset included, since their
  orientation is relative to the motion, which [[sample_track]] applies at each snapshot; `force = true` sets them too
- The C++ in/out argument `nodes` is split into `nodes_in` / `nodes_out`, so that the input stays unchanged

## See also:
- [[generate_track]] (creates tracks with this orientation)
- [[sample_track]] (applies the motion to mounted antennas)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    quadriga_lib::layout_nodes nodes = qd_mex_struct2nodes(prhs[0], "nodes_in");
    const arma::vec orientation = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1], false, "orientation");
    const arma::uvec i_node = (nrhs < 3) ? arma::uvec() : qd_mex_get_index_Col<arma::uword>(prhs[2], "i_node"); // 1-based -> 0-based
    const bool force = (nrhs < 4) ? false : qd_mex_get_scalar<bool>(prhs[3], "force", false);

    CALL_QD(quadriga_lib::set_orientation(nodes, orientation, i_node, force));

    plhs[0] = qd_mex_nodes2struct(nodes);
}
