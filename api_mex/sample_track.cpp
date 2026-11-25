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
# SAMPLE_TRACK
Positions and orientations of a node at snapshot times

- A **snapshot** is one sample of the channel in time and space; a **row** is a point stored in the layout. Without `times`, each row of the node is one snapshot. With `times`, a point drop gives as many snapshots as times, and a track is interpolated between its rows
- Point drop (one row): the position is `p0 + v d t` with the speed `v` of the row and the unit vector `d` of its direction (azimuth, elevation; an elevation of NaN is 0). Without direction (NaN azimuth), `d = [cos(tilt) cos(heading), cos(tilt) sin(heading), -sin(tilt)]` from the orientation; bank has no effect. The orientation is constant
- Track (several rows): the time of each row is `time`, else it follows from the arc length and the speed of the row, `t(0) = 0`, `t(i+1) = t(i) + |Δp| / speed(i)`; a step of zero length with speed 0 takes no time. The position is interpolated linearly in time, bank and heading of the orientation along the shorter way and the tilt linearly; before the first and after the last time, the first and the last row are held. If several rows have the same time, the later row holds at that time. The times in `times` need not increase. A track with neither `time` nor a speed raises an error if `times` is given; a moving step with speed 0 raises an error
- Mounted antennas (`relative = 1`): the orientation is relative to the direction of motion, and the output is `R(motion) R(orientation)` in the GCS, with bank 0 in `R(motion)`. For a point drop, the motion is its direction, which must be given. For a track, it is the direction of the step of the row segment that holds the snapshot (the step from a row to the next one, the last step repeated at the end); a step of zero length takes the previous direction, at the start the next one that exists. The stored orientation is interpolated first, then rotated by the motion of the segment
- Members of `nodes` with sizes that do not fit, and a `direction` on a track raise an error; the other rules of a valid layout (see [[layout_csv_read]]) are not checked

## Usage:
```
[ pos, orientation, i_row ] = quadriga_lib.sample_track( nodes, i_node, times );
```

## Inputs:
- **`nodes`** — Nodes of one side of a layout, e.g. `layout.rx`; fields in [[layout_csv_read]]
- **`i_node`** — Node index, 1-based
- **`times`** — Snapshot times in [s], finite; `[]` = one snapshot per row; `[n_snap]`

## Outputs:
- **`pos`** — Positions in [m]; `[3, n_snap]`
- **`orientation`** — Orientations (bank, tilt, heading) in [rad] in the GCS, including the mounting rotation; `[3, n_snap]`
- **`i_row`** — Row at or before each snapshot, 1-based, index into the arrays of `nodes` (not into the rows of the node); uint64; `[n_snap, 1]`

## Example:
```
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
[ pos, ori ] = quadriga_lib.sample_track( layout.rx, 1, 0:0.1:1 );
```

## See also:
- [[generate_track]] (creates tracks)
- [[set_orientation]] (orientation from the motion)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const quadriga_lib::layout_nodes nodes = qd_mex_struct2nodes(prhs[0], "nodes");
    if (mxIsEmpty(prhs[1]))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'i_node' must not be empty.");
    const arma::uword i_node = qd_mex_get_index(prhs[1], "i_node"); // 1-based -> 0-based
    const arma::vec times = qd_mex_get_Col<double>(prhs[2], false, "times");

    arma::mat pos, orientation;
    arma::uvec i_row;
    arma::uvec *p_i_row = (nlhs > 2) ? &i_row : nullptr;
    CALL_QD(quadriga_lib::sample_track(nodes, i_node, times, pos, orientation, p_i_row));

    plhs[0] = qd_mex_copy2matlab(&pos);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&orientation);
    if (nlhs > 2)
    {
        i_row += 1; // 0-based -> 1-based
        plhs[2] = qd_mex_copy2matlab(&i_row);
    }
}
