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
# GENERATE_TRACK
Replace the rows of nodes by a generated track or a point drop, or append a node with one

- Gives nodes a straight, circular or street-like track, or makes them point drops with a speed and a direction, the
  short-term mobility of 3GPP TR 38.901
- All nodes with the name of a given node (the sectors of a site, the panels of a terminal) get the same track. It
  starts at the first position of the node and is horizontal at the height of the first row
- The antennas look along the motion, as [[set_orientation]] turns them; [[sample_track]] gives the positions at
  snapshot times
- An empty `i_node` appends a new node at `start` and gives it the track (see Notes)

## Usage:
```
nodes_out = quadriga_lib.generate_track( nodes_in, i_node, type, length, speed, direction, spacing, ...
    options, seed, name, start );
```

## Inputs:
- **`nodes_in`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`); fields in [[layout_csv_read]]
- **`i_node`** — Node indices, 1-based; `[]` = append a new node
- **`type`** — Type of the track, not case sensitive:
  - `point`: a point drop with `speed` and a horizontal `direction`; the rows after the first are removed; `length` is
    not used
  - `linear`: a straight track of `length` (default 1 m) in the start direction
  - `circular`: a closed circle of circumference `length` (default 20π m, radius 10 m); the first and the last point are
    equal
  - `street`: the route of a car through a city grid of straight streets and crossings, of `length` (default 1000 m),
    see Notes
- **`length`** — Length in [m], finite and >= 0; 0 = default of the type; default: 0
- **`speed`** — Speed in [m/s], finite and >= 0, and > 0 for tracks; NaN = the speed of the first row of the node if it
  is valid (> 0 for a track); without one, `point` gets 0 and the other types raise an error; default: NaN
- **`direction`** — Start direction (azimuth) in [rad], finite; one value for all nodes, or one per entry of `i_node`
  (nodes of one name need equal values); `[]` = uniform random in [0, 2π); default: `[]`
- **`spacing`** — Point spacing in [m], > 0; NaN = two points (`linear`), 129 points (`circular`), 1 m (`street`);
  default: NaN
- **`options`** — Struct of options of `street`, one value per field: `street_length_min` = 50, `street_length_mu` =
  187, `street_length_std` = 83 (all in [m], >= 0; with a std of 0, mu must be >= min and > 0), `curve_radius` = 10 [m],
  `turn_probability` = 0.5 (in [0, 1]); other fields, and these fields for the other types, raise an error; default:
  `[]`
- **`seed`** — Seed, >= 0 for reproducible draws; -1 = system random device; below -1 raises an error; default: -1
- **`name`** — Name of the new node (`i_node = []` only); `''` = the next free name; default: `''`
- **`start`** — First position `[x, y, z]` of the new node in [m] (`i_node = []` only); `[]` = the origin; default: `[]`

## Outputs:
- **`nodes_out`** — The nodes with the generated rows

## Example:
```
layout = quadriga_lib.layout_csv_read( '', 'rx.csv' );
layout.rx = quadriga_lib.generate_track( layout.rx, 1, 'street', 500, 10, [], NaN, struct( 'turn_probability', 0.3 ), 42 );
layout.rx = quadriga_lib.generate_track( layout.rx, [], 'linear', 100, 1.5, 0, NaN, [], 42, 'walker', [10, 20, 1.5] );
```

## Notes:
- **Points**: `linear` has two points, or `ceil(length / spacing) + 1` equidistant points; `circular` 129 points, or
  `max(3, ceil(length / spacing)) + 1`; `street` is sampled with the point spacing `spacing`, and the chords between the
  points are slightly shorter than the route at the corners
- **Direction of a circle**: the angle of the start point seen from the centre of the circle. A positive value runs
  counter-clockwise, a negative one clockwise: 0 starts in the east of the circle and runs north, -π/2 starts in the
  south and runs west. A random direction is drawn in [0, 2π), so random circles run counter-clockwise
- **Street route** (as `qd_track.generate` of the NG library): straight streets of random length (normal distribution,
  drawn again below `street_length_min`) are separated by crossings. At a crossing the car turns with the probability
  `turn_probability` by about 90° (standard deviation 15°) to the left or right, cutting the corner along a path of
  length `2 · r · cos(φ / 2)` for the turn angle φ and the `curve_radius` r. The direction stays within 105° of the
  start direction, so that the car does not drive in circles
- **Values of the new rows**: `linear`, `circular` and `street` give tracks of equal `speed` without `direction`,
  oriented along the motion as [[set_orientation]] with `force = false`; nodes with `relative = 1` keep the orientation
  of their first row on all rows. `point` keeps the orientation of the node. `scenario`, `indoor`, `floor`, `d2d_in` and
  the extra columns take the values of the first row; `time` is cleared
- **Reproducibility**: a seed >= 0 gives the same track on all platforms. Each name (all its nodes) has its own random
  stream, so the track does not depend on the other nodes or the order in `i_node`
- **Append mode**: with an empty `i_node`, a new node is appended with one row at `start` and the default values (sector
  0, the generic antenna, orientation 0, speed 0, no scenario), and gets the track; a track type therefore needs
  `speed`. It is the last node of `nodes_out`. The default name is the common prefix of the names before their number
  with the next number (`UT3` after `UT1`, `UT2`), else `N<n>`. `name` or `start` with a non-empty `i_node` raises an
  error
- The C++ in/out argument `nodes` is split into `nodes_in` / `nodes_out`, so that the input stays unchanged

## See also:
- [[sample_track]] (snapshots of a track)
- [[set_orientation]] (orientation from the motion)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    quadriga_lib::layout_nodes nodes = qd_mex_struct2nodes(prhs[0], "nodes_in");
    const arma::uvec i_node = qd_mex_get_index_Col<arma::uword>(prhs[1], "i_node"); // 1-based -> 0-based
    const std::string type = qd_mex_get_string(prhs[2], "", "type");
    const double length = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "length", 0.0);
    const double speed = (nrhs < 5) ? NAN : qd_mex_get_scalar<double>(prhs[4], "speed", NAN);
    const arma::vec direction = (nrhs < 6) ? arma::vec() : qd_mex_get_Col<double>(prhs[5], false, "direction");
    const double spacing = (nrhs < 7) ? NAN : qd_mex_get_scalar<double>(prhs[6], "spacing", NAN);
    const auto options = (nrhs < 8) ? std::unordered_map<std::string, std::vector<double>>() : qd_mex_layout_options(prhs[7], "options");
    const arma::sword seed = (nrhs < 9) ? -1 : (arma::sword)qd_mex_get_seed(prhs[8], "seed");
    const std::string name = (nrhs < 10) ? std::string() : qd_mex_get_string(prhs[9], "", "name");
    const arma::vec start = (nrhs < 11) ? arma::vec() : qd_mex_get_Col<double>(prhs[10], false, "start");

    CALL_QD(quadriga_lib::generate_track(nodes, i_node, type, length, speed, direction, spacing, options, seed, name, start));

    plhs[0] = qd_mex_nodes2struct(nodes);
}
