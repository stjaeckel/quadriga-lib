// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_mex_layout.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!SECTION_DESC
A **layout** describes a radio network for channel simulation: the transmitters (Tx) and receivers (Rx), where they are
and how they move, which antenna each one has and which scenario applies. The functions of this section create layouts,
store them in files, move the nodes along tracks and compute the channels of all Tx-Rx pairs.

**What a layout holds** (a struct, fields in [[layout_csv_read]]):

- The **Tx nodes** in `layout.tx` (base stations, access points) and the **Rx nodes** in `layout.rx` (user terminals,
  stations), each a struct of flat arrays. A node has a name, a sector number, an antenna and one or more rows:
  positions with orientation, scenario and indoor state
- The **antennas** of the nodes and the simulation frequencies `layout.center_freq`
- The **link states** `layout.links` of one realisation: LOS and indoor state, scenario and seeds of each Tx-Rx pair
  ([[init_link_states]])

**Typical workflow:**

1. Create a layout: [[generate_layout]] from a preset of 3GPP TR 38.901, ITU-R M.2412 or 3GPP TR 37.885, or
   [[layout_csv_read]] from two CSV files; [[drop_terminals]] drops new UTs into the cells of the Tx nodes
2. Set the frequencies `layout.center_freq` and the antennas ([[set_antenna]], [[set_antenna_array]]); [[eval_antennas]]
   generates the antennas for new frequencies
3. Optionally, move the nodes: [[generate_track]] creates tracks, [[set_orientation]] turns the antennas along the
   motion, [[sample_track]] gives the positions at given times
4. Compute the channels with [[get_channels_layout]]
5. Store the layout with [[layout_csv_write]]

For more control, step 4 runs in three stages: [[init_link_states]] draws the link states, [[generate_rays]] the rays of
the geometry-based stochastic model (GBSM), and [[get_gbsm_channels]] applies the antennas, the motion and the link
gain. Between the stages, link states and rays can be inspected or edited, [[attach_terminals]] selects the serving cell
of each terminal, and [[export_rays_qrt]] stores the rays in a QRT file.

**Antennas.** Each node has one antenna. [[set_antenna]] sets it from the arguments of [[generate_antenna]] or from an
entry of a QDANT file, [[set_antenna_array]] from an array; in the CSV files, antenna columns describe it.
[[eval_antennas]] generates the arrays from these descriptions. A node has one antenna for all bands or one per band. A
node without an antenna uses the generic antenna: two co-located isotropic elements with vertical and horizontal
polarization.

**Terms** (full list in the [Glossary](cpp_api.html#glossary)):

- **Tx nodes**, **Rx nodes**: the two node structs of a layout, `layout.tx` and `layout.rx`. A function that works on
  one of them takes the struct itself or `rx` (`false`: Tx nodes, `true`: Rx nodes)
- **Node**: one transmitter or receiver, identified by its `name` and `sector`, with one antenna and one track
- **Site**: all Tx nodes of one name, e.g. the three sectors of a base station; they share position and scenario
- **Terminal**: all Rx nodes of one name, e.g. the antenna panels of one device; they share their positions
- **Row**: one position and orientation of a node, with the values that apply from there on (speed, scenario, indoor
  state)
- **Point drop**: a node with one row; it stays where it is or, with a speed, moves along its direction
- **Track**: a node with several rows; it moves along its positions
- **Snapshot**: one sample of the channel in time; without snapshot times, each row is one snapshot
- **Segment**: a part of a track with one scenario and indoor state; a new segment starts where one of them changes
- **Propagation link**: one site, one terminal and one segment, with its own LOS and indoor state, large-scale
  parameters and rays
- **Channel link**: one Tx node and one Rx node; it applies their antennas to the rays of its propagation link, which
  the sectors of a site share
- **Band**: one of the simulation frequencies `layout.center_freq`
- **Scenario**: a family (e.g. `UMa`), which draws the LOS and indoor state of each link, or a conf name (e.g.
  `UMa_NLOS`), which fixes them
- **Wrap-around**: copies of the sites, displaced by the lattice vectors `layout.wrap`, so that a cluster of cells tiles
  the plane; each link uses the copy (*image*) nearest to the terminal
- **Orientation**: bank, tilt and heading of the antenna; a positive tilt points down. **Direction**: azimuth and
  elevation of the motion of a point drop

**Conventions.** Angles are in radians in the structs and in degrees in the CSV files. Indices are 1-based: the index
arguments, the returned indices and the index fields `offset`, `pairing` and, in `layout.links`, `i_tx`, `i_rx`,
`i_segment`, `i_conf` and `o2i_model`. `i_antenna`, the antenna of a node, is 0 for the generic antenna and 1 for the
first antenna of the layout; `sector`, `image`, `floor_id`, `indoor`, `floor` and the seeds are values, not indices.
Random values follow from the seeds and the positions, not from the order of evaluation, the batches or the number of
threads. The file format is described in [Layout CSV files](formats.html#layout_csv).
SECTION_DESC!*/

/*!MD
# LAYOUT_CSV_READ
Read the Tx and Rx CSV files of a layout

- Reads the Tx file and the Rx file of a layout in the [Layout CSV format](formats.html#layout_csv) and returns the
  layout struct, with angles in rad
- Generates the antennas of the nodes from their antenna columns, unless `eval_antennas` is `false`
- Keeps unknown columns and returns their names, so that a typo in a column name is visible

## Usage:
```
[ layout, extra_tx, extra_rx ] = quadriga_lib.layout_csv_read( fn_tx, fn_rx, center_freq, eval_antennas );
```

## Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; `''` = no Tx nodes or no Rx nodes
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `[]` = from the files; default:
  `[]`; `[n_freq]`
- **`eval_antennas`** — `true`: generate the antennas; `false`: `antenna` stays empty, `antenna_args` and `i_antenna`
  are kept, and the nodes use the generic antenna until [[eval_antennas]] is called; default: `true`

## Outputs:
- **`layout`** — The layout, a 1x1 struct, which all functions of this section take and return; `name` and `links` are
  empty. Angles in rad; `offset`, `pairing`, `i_tx`, `i_rx`, `i_segment`, `i_conf` and `o2i_model` are 1-based. A field
  that is missing or empty in an input struct takes the value of an empty member; unknown fields are rejected.

  Fields of `layout`:
  - `name` — Name of the layout; not stored in the CSV files; String
  - `tx`, `rx` — Tx nodes (BS, AP) and Rx nodes (UT, STA), fields below; Struct
  - `links` — Link states of the current realisation, fields below; `[]` = none; Struct
  - `seed` — Seed of the layout, -1 = not set; int64, scalar
  - `center_freq` — Simulation frequencies (bands) in [Hz], strictly ascending; `[n_freq, 1]`
  - `antenna_cols` — Names of the antenna columns of the lines in `antenna_args`; Cell of strings
  - `antenna_args` — Antenna descriptions, one line for all bands or one per band (see [[eval_antennas]]); Cell of cells
    of strings
  - `antenna` — Evaluated arrays, entry `k` a 1xN struct array as in [[generate_antenna]], N = 0 (not evaluated), 1 (all
    bands) or `n_freq`; Cell `[n_antenna, 1]`
  - `wrap` — Lattice vectors of the wrap-around in [m], one per column; two vectors form a reduced basis,
    `0 <= a1·a2 <= min(abs(a1)^2, abs(a2)^2) / 2`, so that ±a1, ±a2 and ±(a1 - a2) are the nearest images; empty = none;
    `[2, 1]` or `[2, 2]`
  - `pairing` — Site index and terminal index of the linked pairs; empty = all pairs; uint64, `[2, n_pair]`

  Fields of the node structs `layout.tx` and `layout.rx`:
  - `name` — Site or terminal name, 1 to 255 bytes, without comma, semicolon, quote and line break, and without leading
    or trailing blank; Cell `[n_node, 1]`
  - `sector` — Sector number; (`name`, `sector`) is unique; uint32, `[n_node, 1]`
  - `i_antenna` — Antenna: index into the antenna lists of the layout, 0 = generic antenna; uint32, `[n_node, 1]`
  - `offset` — First row of each node; the rows of node `i` are `offset(i)` to `offset(i+1) - 1` (the last node to the
    end); uint64, `[n_node, 1]`
  - `pos` — Positions in [m]; `[3, n_row]`
  - `orientation` — Bank, tilt, heading in [rad]; `[3, n_row]`
  - `direction` — Azimuth, elevation of travel in [rad]; NaN azimuth = along the orientation; NaN on all rows of a
    track; empty = all NaN; `[2, n_row]`
  - `speed` — Speed in [m/s] from this row on; empty = 0; `[n_row, 1]`
  - `time` — Time in [s]; NaN on all rows of a node = not given; empty = none; `[n_row, 1]`
  - `scenario` — Scenario family or conf name, `''` = none; empty = none; Cell `[n_row, 1]`
  - `indoor` — -1 = draw, 0 outdoor, 1 building, 2 car, 3 residential building; empty = all -1; int32, `[n_row, 1]`
  - `floor` — Floor number from 1, -1 = from z; empty = all -1; int32, `[n_row, 1]`
  - `d2d_in` — Indoor 2D distance in [m], NaN = draw; empty = all NaN; `[n_row, 1]`
  - `clutter` — `d_clutter`, `r_clutter` and `h_clutter`, NaN = model default; empty = all NaN; `[3, n_node]`
  - `scenpar_override` — Overrides of scenario parameters, one field per key, NaN = none; Struct of `[n_node, 1]`
  - `seed` — Seed of the site, -1 = derived from the seed of the layout; empty = all -1; int64, `[n_node, 1]`
  - `relative` — 1 = orientation relative to the direction of motion; empty = all 0; uint32, `[n_node, 1]`
  - `cell` — Tx nodes only: 0 = the site has no drop cells of its own ([[drop_terminals]]); empty = all 1; uint32,
    `[n_node, 1]`
  - `extra_names`, `extra` — Names of unknown CSV columns and their cells as text; Cell `[n_extra, 1]` of cells
    `[n_row, 1]`

  Fields of `layout.links`:
  - `i_tx`, `i_rx`, `i_segment` — Site, terminal and segment of each link; uint32, `[n_link, 1]`
  - `image` — Wrap image of the site, 0 = the real site; uint32, `[n_link, 1]`
  - `i_conf`, `conf_names` — Index into `conf_names`, from scenario, LOS and indoor state; resolved conf names; uint32,
    `[n_link, 1]`; cell
  - `los`, `indoor`, `floor_id`, `d2d_in` — 1 = line of sight; indoor state at the start of the segment, as in the
    nodes; floor number - 1 (0 outdoors) and indoor 2D distance in [m] of the drawn state; `[n_link, 1]`
  - `o2i_rnd`, `o2i_model` — Random inputs of [[o2i_loss]]; building O2I model, index into the list of the family, for
    every terminal of a family with O2I, also outdoors; `[n_terminal, 4]`; uint32, `[n_terminal, 1]`
  - `gbsm_seed`, `seed` — Seed of the random phases, equal for one site, image and conf; master seed, the base of the
    random values of all stages, -1 = not set; int64, `[n_link, 1]`; scalar
  - `method`, `sos_preset`, `anchor_freq`, `pol_variability` — Method and SOS preset as used (an empty `method` means
    not filled); anchor frequency of the multi-frequency model in [Hz]; polarization power variability; Strings, scalar,
    logical
  - `conf_files`, `layout_hash` — Resolved paths of the conf files, empty = not resolved; hash of the layout fields that
    determine the links (see [[init_link_states]]); Cell; uint64
- **`extra_tx`**, **`extra_rx`** — Names of the unknown columns of the Tx and Rx file, in file order; cell array of
  strings; `[n_extra, 1]`

## Notes:
- **Errors**: a violation of the format and a layout that is not valid raise an error with the file name, the line and
  the column (see [Reading](formats.html#layout-csv-reading)); with `eval_antennas`, so do a line that
  [[generate_antenna]] rejects, a QDANT file that cannot be read and an array spaced in wavelengths in a layout without
  frequencies
- **Validity**: a valid layout, which the functions of this section check, has node structs with fitting sizes, a
  fitting `offset`, finite positions and orientations, valid names and values in range, and follows the rules of the
  [Layout CSV files](formats.html#layout-csv-columns) for `time`, `speed`, `direction`, `floor` and `d2d_in` and for the
  values that the nodes of one site or terminal share ([Nodes and rows](formats.html#layout-csv-rows)); the antenna
  lists (a node with a band that is not set is reported), the frequencies, `wrap`, `pairing` and `links` are checked too
- **Antennas**: equal descriptions share one entry of the antenna lists, Tx nodes first; a relative `qdant_file` is
  stored as a normalised absolute path
- **Memory**: large arrays on the default 1° grid need much memory ([Antennas](formats.html#layout-csv-antennas)); give
  a `res` column or read with `eval_antennas = false`

## See also:
- [[layout_csv_write]] (stores a layout)
- [[generate_antenna]] (the antenna types and arguments of the antenna columns)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn_tx = qd_mex_get_string(prhs[0], "", "fn_tx");
    const std::string fn_rx = qd_mex_get_string(prhs[1], "", "fn_rx");
    const arma::vec center_freq = (nrhs < 3) ? arma::vec() : qd_mex_get_Col<double>(prhs[2], false, "center_freq");
    const bool eval_antennas = (nrhs < 4) ? true : qd_mex_get_scalar<bool>(prhs[3], "eval_antennas", true);

    std::vector<std::string> extra_tx, extra_rx;
    std::vector<std::string> *p_extra_tx = (nlhs > 1) ? &extra_tx : nullptr;
    std::vector<std::string> *p_extra_rx = (nlhs > 2) ? &extra_rx : nullptr;

    quadriga_lib::layout layout;
    CALL_QD(layout = quadriga_lib::layout_csv_read(fn_tx, fn_rx, center_freq, eval_antennas, p_extra_tx, p_extra_rx));

    plhs[0] = qd_mex_layout2struct(layout);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&extra_tx);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&extra_rx);
}
