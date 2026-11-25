// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# generate_track
Replace the rows of nodes by a generated track or a point drop

- Gives nodes a straight, circular or street-like track, or makes them point drops with a speed and a direction, the
  short-term mobility of 3GPP TR 38.901
- All nodes with the name of a given node (the sectors of a site, the panels of a terminal) get the same track. It
  starts at the first position of the node and is horizontal at the height of the first row
- The antennas look along the motion, as [[set_orientation]] turns them; [[sample_track]] gives the positions at
  snapshot times

## Usage:
```
nodes_out = quadriga_lib.layout.generate_track( nodes_in, i_node, type, *, length, speed, direction, spacing, \
    options, seed )
```

## Inputs:
- **`nodes_in`** — The Tx nodes or the Rx nodes of a layout (`layout["tx"]` or `layout["rx"]`); dict, keys in
  [[layout.csv_read]]
- **`i_node`** — Node indices, 0-based, not empty; int or `(n,)`
- **`type`** — Type of the track, str, not case sensitive:
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
- **`direction`** — Start direction (azimuth) in [rad], finite; float, one value for all nodes, or `(n,)` with one per
  entry of `i_node` (nodes of one name need equal values); `None` = uniform random in [0, 2π); default: `None`
- **`spacing`** — Point spacing in [m], > 0; NaN = two points (`linear`), 129 points (`circular`), 1 m (`street`);
  default: NaN
- **`options`** — Dict of options of `street`, one value per key: `street_length_min` = 50, `street_length_mu` = 187,
  `street_length_std` = 83 (all in [m], >= 0; with a std of 0, mu must be >= min and > 0), `curve_radius` = 10 [m],
  `turn_probability` = 0.5 (in [0, 1]); other keys, and these keys for the other types, raise an error; default: `None`
- **`seed`** — Seed, >= 0 for reproducible draws; -1 = system random device; below -1 raises an error; default: -1

## Outputs:
- **`nodes_out`** — The nodes with the generated rows; dict

## Example:
```
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "", "rx.csv" )
layout["rx"] = qd.layout.generate_track( layout["rx"], 0, "street", length=500, speed=10, \
    options={"turn_probability": 0.3}, seed=42 )
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
  length $2 r \cos(\varphi / 2)$ for the turn angle $\varphi$ and the `curve_radius` $r$. The direction stays within
  105° of the start direction, so that the car does not drive in circles
- **Values of the new rows**: `linear`, `circular` and `street` give tracks of equal `speed` without `direction`,
  oriented along the motion as [[set_orientation]] with `force = False`; nodes with `relative = 1` keep the orientation
  of their first row on all rows. `point` keeps the orientation of the node. `scenario`, `indoor`, `floor`, `d2d_in` and
  the extra columns take the values of the first row; `time` is cleared
- **Reproducibility**: a seed >= 0 gives the same track on all platforms. Each name (all its nodes) has its own random
  stream, so the track does not depend on the other nodes or the order in `i_node`
- **`nodes_in` / `nodes_out`**: the C++ in/out argument `nodes` is split, so that the input stays unchanged

## See also:
- [[sample_track]] (snapshots of a track)
- [[set_orientation]] (orientation from the motion)
- [[layout.csv_read]] (the keys of a node dict)
MD!*/
py::dict generate_track(const py::dict &nodes_in,
                        const py::handle &i_node,
                        const std::string &type,
                        double length,
                        double speed,
                        py::handle direction,
                        double spacing,
                        py::handle options,
                        long long seed)
{
    auto nodes = qd_python_dict2nodes(nodes_in, false, "nodes_in");
    const auto i_node_a = qd_python_numpy2arma_Col<arma::uword>(i_node, true, false, "i_node");
    const auto direction_a = qd_python_numpy2arma_Col<double>(direction, true);
    const auto options_m = qd_pl_options(options, "options");

    quadriga_lib::generate_track(nodes, i_node_a, type, length, speed, direction_a, spacing, options_m, (arma::sword)seed);
    return qd_python_nodes2dict(nodes);
}

/*!MD
# sample_track
Positions and orientations of a node at snapshot times

- Returns the position and the orientation of one node at each snapshot time, for point drops and tracks;
  [[get_channels_from_rays]] samples both ends of each link this way
- Without `times`, each row of the node is one snapshot
- The orientation includes the rotation of mounted antennas (`relative = 1`), so it is the orientation in the GCS

## Usage:
```
pos, orientation, i_row = quadriga_lib.layout.sample_track( nodes, i_node, times )
```

## Inputs:
- **`nodes`** — The Tx nodes or the Rx nodes of a layout (`layout["tx"]` or `layout["rx"]`); dict, keys in
  [[layout.csv_read]]
- **`i_node`** — Node index, 0-based; int
- **`times`** — Snapshot times in [s], finite; `(n_snap,)`; empty = one snapshot per row

## Outputs:
- **`pos`** — Positions in [m]; `(3, n_snap)`
- **`orientation`** — Orientations (bank, tilt, heading) in [rad] in the GCS, including the mounting rotation;
  `(3, n_snap)`
- **`i_row`** — Row at or before each snapshot, 0-based, index into the arrays of `nodes` (not into the rows of the
  node); int64; `(n_snap,)`

## Example:
```
import numpy as np, quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
pos, ori, i_row = qd.layout.sample_track( layout["rx"], 0, np.arange(0, 1.01, 0.1) )
```

## Notes:
- **Point drop** (one row): $p(t) = p_0 + v t \hat{d}$ with the speed $v$ of the row and the unit vector $\hat{d}$ of
  its direction (azimuth, elevation; an elevation of NaN is 0). Without a direction (NaN azimuth),
  $\hat{d} = (\cos\beta \cos\alpha, \cos\beta \sin\alpha, -\sin\beta)$ with the tilt $\beta$ and the heading $\alpha$ of
  the orientation; the bank has no effect. The orientation is constant
- **Track** (several rows): the time of row $i$ is `time`, else it follows from the speed $v_i$ of the row, $t_0 = 0$,
  $t_{i+1} = t_i + \lvert p_{i+1} - p_i \rvert / v_i$; a step of zero length with speed 0 takes no time. The position is
  interpolated linearly in time, bank and heading along the shorter way and the tilt linearly; before the first and
  after the last time, the first and the last row hold. If several rows have the same time, the later row holds at that
  time. The times in `times` need not increase. A track with neither `time` nor a speed raises an error if `times` is
  given; a moving step with speed 0 raises an error
- **Mounted antennas** (`relative = 1`): the orientation is relative to the direction of motion, and the output is
  $R_\mathrm{motion} R_\mathrm{orientation}$, with bank 0 in $R_\mathrm{motion}$. For a point drop, the motion is its
  direction, which must be given. For a track, it is the direction of the step from the row at or before the snapshot to
  the next row (the last step repeated at the end); a step of zero length takes the previous direction, at the start the
  next one that exists. The stored orientation is interpolated first, then rotated by the motion
- **Checks**: members of `nodes` with sizes that do not fit, and a `direction` on a track, raise a `ValueError`;
  the other rules of a valid layout ([[layout.csv_read]]) are not checked

## See also:
- [[generate_track]] (creates tracks)
- [[set_orientation]] (orientation from the motion)
- [[layout.csv_read]] (the keys of a node dict)
MD!*/
py::tuple sample_track(const py::dict &nodes,
                       arma::uword i_node,
                       const py::array_t<double> &times)
{
    const auto nodes_c = qd_python_dict2nodes(nodes, true, "nodes");
    const auto times_a = qd_python_numpy2arma_Col<double>(times, true);

    arma::mat pos, orientation;
    arma::uvec i_row;
    quadriga_lib::sample_track(nodes_c, i_node, times_a, pos, orientation, &i_row);
    return py::make_tuple(qd_python_copy2numpy(&pos), qd_python_copy2numpy(&orientation),
                          qd_python_copy2numpy<arma::uword, py::ssize_t>(&i_row));
}

/*!MD
# set_orientation
Set the orientation of nodes from their motion, or to a given value

- Turns the antennas of tracks along their motion, e.g. for a UT held in the direction of travel, or sets all rows of
  the nodes to a fixed orientation
- `force = False`: tracks get bank 0 and the tilt and heading of the steps between their rows, plus `orientation` as an
  offset; point drops keep their orientation plus the offset
- `force = True`: every row of the nodes gets exactly `orientation`

## Usage:
```
nodes_out = quadriga_lib.layout.set_orientation( nodes_in, *, orientation, i_node, force )
```

## Inputs:
- **`nodes_in`** — The Tx nodes or the Rx nodes of a layout (`layout["tx"]` or `layout["rx"]`); dict, keys in
  [[layout.csv_read]]
- **`orientation`** — Bank, tilt, heading in [rad], 3 finite values, `(3,)`; `None` = 0; default: None
- **`i_node`** — Node indices, 0-based; int or `(n,)`; `None` = all nodes; default: None
- **`force`** — `False`: orientation from the motion of tracks, plus `orientation` as offset; `True`: `orientation` on
  all rows; default: `False`

## Outputs:
- **`nodes_out`** — The nodes with the new orientations; dict

## Example:
```
import quadriga_lib as qd
layout["rx"] = qd.layout.set_orientation( layout["rx"] )                          # The antennas look along the motion
layout["rx"] = qd.layout.set_orientation( layout["rx"], orientation=[0, 0, 0.5] )  # The same, turned by 0.5 rad
```

## Notes:
- **Tracks** (`force = False`): forward differences between the rows, the last step repeated, or for a closed track the
  step to the first row, as `qd_track.calc_orientation` of the NG library does; the angles are wrapped to [-π, π]. A
  step of zero length keeps the previous orientation, at the start the next one that exists. A vertical step has the
  tilt -90° (up) or 90° (down) and keeps the previous heading (a deliberate deviation from NG, where it is 0). A rising
  track has a negative tilt; a track that does not move keeps its orientation (plus the offset)
- **Point drops** (`force = False`): without an offset, the orientation is not touched; random orientations are not
  generated here
- **Mounted antennas**: `force = False` leaves nodes with `relative = 1` unchanged, offset included, since their
  orientation is relative to the motion, which [[sample_track]] applies at each snapshot; `force = True` sets them too
- **`nodes_in` / `nodes_out`**: the C++ in/out argument `nodes` is split, so that the input stays unchanged

## See also:
- [[generate_track]] (creates tracks with this orientation)
- [[sample_track]] (applies the motion to mounted antennas)
MD!*/
py::dict set_orientation(const py::dict &nodes_in,
                         py::handle orientation,
                         py::handle i_node,
                         py::handle force)
{
    auto nodes = qd_python_dict2nodes(nodes_in, false, "nodes_in");
    const auto orientation_a = qd_python_numpy2arma_Col<double>(orientation, true);
    const auto i_node_a = qd_python_numpy2arma_Col<arma::uword>(i_node, true, false, "i_node");

    quadriga_lib::set_orientation(nodes, orientation_a, i_node_a, qd_python_flag(force, false, "force"));
    return qd_python_nodes2dict(nodes);
}
