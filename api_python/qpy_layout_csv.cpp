// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!SECTION_DESC
A **layout** describes a radio network for channel simulation: the transmitters (Tx) and receivers (Rx), where they are
and how they move, which antenna each one has and which scenario applies. The functions of this section create layouts,
store them in files, move the nodes along tracks and compute the channels of all Tx-Rx pairs.

**What a layout holds** (keys in [[layout.csv_read]]):

- The **Tx nodes** in `layout["tx"]` (base stations, access points) and the **Rx nodes** in `layout["rx"]` (user
  terminals, stations), each a dict of flat arrays. A node has a name, a sector number, an antenna and one or more rows:
  positions with orientation, scenario and indoor state
- The **antennas** of the nodes and the simulation frequencies `layout["center_freq"]`
- The **link states** `layout["links"]` of one realisation: LOS and indoor state, scenario and seeds of each Tx-Rx pair
  ([[init_link_states]])

**In Python:**

- The functions are in the submodule `quadriga_lib.layout`, e.g. `quadriga_lib.layout.generate_layout`;
  `layout_csv_read` and `layout_csv_write` are `layout.csv_read` and `layout.csv_write`. [[crossfade_segments]] is in
  `quadriga_lib.channel`
- A layout is a dict with the member names of the C++ class `layout`: `tx`, `rx` and `links` are nested dicts,
  `scenpar_override` is a dict with one array per scenario key
- `antenna` is a list with one entry per antenna, each a list of arrayant dicts, one per band (the 4D multi-frequency
  dict of [[generate_antenna]] cannot hold per-band geometry); rays are lists of ray dicts, see [[ssf_rays]]

**Typical workflow:**

1. Create a layout: [[generate_layout]] from a preset of 3GPP TR 38.901, ITU-R M.2412 or 3GPP TR 37.885, or
   [[layout.csv_read]] from two CSV files; [[drop_terminals]] drops new UTs into the cells of the Tx nodes
2. Set the frequencies `layout["center_freq"]` and the antennas ([[set_antenna]], [[set_antenna_array]]);
   [[eval_antennas]] generates the antennas that are not evaluated yet, or all of them for new frequencies
3. Optionally, move the nodes: [[generate_track]] creates tracks, [[set_orientation]] turns the antennas along the
   motion, [[sample_track]] gives the positions at given times
4. Compute the channels with [[get_channels_layout]]
5. Store the layout with [[layout.csv_write]]

For more control, step 4 runs in three stages: [[init_link_states]] draws the link states, [[generate_rays]] the rays of
the geometry-based stochastic model (GBSM), and [[get_gbsm_channels]] applies the antennas, the motion and the link
gain. Between the stages, link states and rays can be inspected or edited, [[attach_terminals]] selects the serving cell
of each terminal, and [[export_rays_qrt]] stores the rays in a QRT file.

**Antennas.** Each node has one antenna. [[set_antenna]] sets it from the arguments of [[generate_antenna]], from an
entry of a QDANT file or by the index of an antenna that the layout already has, [[set_antenna_array]] from an array; in
the CSV files, antenna columns describe it. [[eval_antennas]] generates the arrays from the descriptions. A node has one
antenna for all bands or one per band. A node without an antenna uses the generic antenna: two co-located isotropic
elements with vertical and horizontal polarization.

**Terms** (full list in the [Glossary](cpp_api.html#glossary)):

- **Tx nodes**, **Rx nodes**: the two node tables of a layout, `layout["tx"]` and `layout["rx"]`. A function that works
  on one of them takes the node dict itself or `rx` (`False`: Tx nodes, `True`: Rx nodes)
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
- **Band**: one of the simulation frequencies `layout["center_freq"]`
- **Scenario**: a family (e.g. `UMa`), which draws the LOS and indoor state of each link, or a conf name (e.g.
  `UMa_NLOS`), which fixes them
- **Wrap-around**: copies of the sites, displaced by the lattice vectors `layout["wrap"]`, so that a cluster of cells
  tiles the plane; each link uses the copy (*image*) nearest to the terminal
- **Orientation**: bank, tilt and heading of the antenna; a positive tilt points down. **Direction**: azimuth and
  elevation of the motion of a point drop

**Conventions.** Angles are in radians in the dicts and in degrees in the CSV files. Indices are 0-based, as in C++,
except `i_antenna`, the antenna of a node, where 0 is the generic antenna and 1 the first antenna of the layout. Random
values follow from the seeds and the positions, not from the order of evaluation, the batches or the number of threads.
The file format is described in [Layout CSV files](formats.html#layout_csv).
SECTION_DESC!*/

/*!MD
# layout.csv_read
Read the Tx and Rx CSV files of a layout

- Reads the Tx file and the Rx file of a layout in the [Layout CSV format](formats.html#layout_csv) and returns the
  layout, with angles in rad
- Generates the antennas of the nodes from their antenna columns, unless `eval_antennas` is `False`
- Keeps unknown columns and returns their names, so that a typo in a column name is visible

## Usage:
```
layout, extra_tx, extra_rx = quadriga_lib.layout.csv_read( fn_tx, fn_rx, *, center_freq, eval_antennas )
```

## Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files, str or path-like; `""` = no Tx nodes or no Rx nodes
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `None` = from the files;
  `(n_freq,)`; default: None
- **`eval_antennas`** — `True`: generate the antennas; `False`: `antenna` stays empty, `antenna_args` and `i_antenna`
  are kept, and the nodes use the generic antenna until [[eval_antennas]] is called; default: `True`

## Outputs:
- **`layout`** — The layout dict, which all functions of this section take and return; `name` and `links` are empty.
  Keys of `layout`, of the node dicts `layout["tx"]` and `layout["rx"]` and of `layout["links"]` (a key that is missing
  or None in an input dict takes the value of an empty member, unknown keys are rejected). In the node dicts, per-node
  keys have one entry per node and per-row keys one per row; the rows of node `i` are `offset[i]` to `offset[i + 1] - 1`
  (the last node to the end); a key may be empty, with the meaning given:<br><br>
  | Key | Meaning | Type / Size |
  | --- | ------- | ----------- |
  | `name` | Name of the layout; not stored in the CSV files | str |
  | `tx`, `rx` | Tx nodes (BS, AP) and Rx nodes (UT, STA) | dict |
  | `links` | Link states of the current realisation, see [[init_link_states]]; `{}` = none | dict |
  | `seed` | Seed of the layout, -1 = not set | int |
  | `center_freq` | Simulation frequencies (bands) in [Hz], strictly ascending | `(n_freq,)` |
  | `antenna_cols` | Names of the antenna columns of the lines in `antenna_args` | list of str |
  | `antenna_args` | Antenna descriptions, one line for all bands or one per band (see [[eval_antennas]]) | list `[n_antenna]` of lists of str |
  | `antenna` | Evaluated arrays as arrayant dicts of [[generate_antenna]], one for all bands or one per band; empty = not evaluated | list `[n_antenna]` of lists |
  | `wrap` | Lattice vectors of the wrap-around in [m], one per column; two vectors form a reduced basis, $0 \le a_1 \cdot a_2 \le \min(\lvert a_1 \rvert^2, \lvert a_2 \rvert^2) / 2$, so that $\pm a_1$, $\pm a_2$ and $\pm(a_1 - a_2)$ are the nearest images; empty = none | `(2, 1)` or `(2, 2)` |
  | `pairing` | Site index and terminal index of the linked pairs, 0-based; empty = all pairs | int64, `(2, n_pair)` |
  | `name` (`tx`, `rx`) | Site or terminal name, 1 to 255 bytes, without comma, semicolon, quote and line break, and without leading or trailing blank | list of str `[n_node]` |
  | `sector` | Sector number; (`name`, `sector`) is unique | uint32, `(n_node,)` |
  | `i_antenna` | Antenna: 1-based index into the antenna lists of the layout, 0 = generic antenna | uint32, `(n_node,)` |
  | `offset` | First row of each node, 0-based | int64, `(n_node,)` |
  | `pos` | Positions in [m] | `(3, n_row)` |
  | `orientation` | Bank, tilt, heading in [rad] | `(3, n_row)` |
  | `direction` | Azimuth, elevation of travel in [rad]; NaN azimuth = along the orientation; NaN on all rows of a track; empty = all NaN | `(2, n_row)` |
  | `speed` | Speed in [m/s] from this row on; empty = 0 | `(n_row,)` |
  | `time` | Time in [s]; NaN on all rows of a node = not given; empty = none | `(n_row,)` |
  | `scenario` | Scenario family or conf name, `""` = none; empty = none | list of str `[n_row]` |
  | `indoor` | -1 = draw, 0 outdoor, 1 building, 2 car, 3 residential building; empty = all -1 | int32, `(n_row,)` |
  | `floor` | Floor number from 1, -1 = from z; empty = all -1 | int32, `(n_row,)` |
  | `d2d_in` | Indoor 2D distance in [m], NaN = draw; empty = all NaN | `(n_row,)` |
  | `clutter` | `d_clutter`, `r_clutter` and `h_clutter`, NaN = model default; empty = all NaN | `(3, n_node)` |
  | `scenpar_override` | Overrides of scenario parameters by key, NaN = none | dict of `(n_node,)` |
  | `seed` (`tx`, `rx`) | Seed of the site, -1 = derived from the seed of the layout; empty = all -1 | int64, `(n_node,)` |
  | `relative` | 1 = orientation relative to the direction of motion; empty = all 0 | uint32, `(n_node,)` |
  | `cell` | Tx nodes only: 0 = the site has no drop cells of its own ([[drop_terminals]]); empty = all 1 | uint32, `(n_node,)` |
  | `extra_names`, `extra` | Names of unknown CSV columns and their cells as text | list of str `[n_extra]`; list `[n_extra]` of lists `[n_row]` of str |
  | `i_tx`, `i_rx`, `i_segment` (`links`) | Site, terminal and segment of each link, 0-based | uint32, `(n_link,)` |
  | `image` | Wrap image of the site, 0 = the real site | uint32, `(n_link,)` |
  | `i_conf`, `conf_names` | Index into `conf_names`, from scenario, LOS and indoor state; resolved conf names | uint32, `(n_link,)`; list of str |
  | `los` | 1 = line of sight | uint32, `(n_link,)` |
  | `indoor` (`links`) | Indoor state at the start of the segment, as in the node dicts | int32, `(n_link,)` |
  | `floor_id`, `d2d_in` (`links`) | Floor number - 1, 0 outdoors, and indoor 2D distance in [m] of the drawn state | uint32, `(n_link,)`; `(n_link,)` |
  | `o2i_rnd` | Random inputs of [[o2i_loss]] | `(n_terminal, 4)` |
  | `o2i_model` | Building O2I model, index into the list of the family; for every terminal of a family with O2I, also outdoors | uint32, `(n_terminal,)` |
  | `gbsm_seed` | Seed of the random phases; equal for one site, image and conf | int64, `(n_link,)` |
  | `seed` (`links`) | Master seed, the base of the random values of all stages; -1 = not set | int |
  | `method`, `sos_preset` | Method and SOS preset as used; an empty `method` means not filled | str |
  | `anchor_freq` | Anchor frequency of the multi-frequency model in [Hz], default 2e9 | float |
  | `pol_variability` | Polarization power variability, default `False` | bool |
  | `conf_files` | Resolved paths of the conf files; empty = not resolved | list of str |
  | `layout_hash` | Hash of the layout members that determine the links (see [[init_link_states]]) | int |
- **`extra_tx`**, **`extra_rx`** — Names of the unknown columns of the Tx and Rx file, in file order; list of str

## Notes:
- **Errors**: a violation of the format and a layout that is not valid raise a `RuntimeError` with the file name, the
  line and the column (see [Reading](formats.html#layout-csv-reading)); with `eval_antennas`, so do a line that
  [[generate_antenna]] rejects, a QDANT file that cannot be read and an array spaced in wavelengths in a layout without
  frequencies
- **Validity**: the functions that need a valid layout check both node dicts (sizes, `offset`, finite positions and
  orientations, names, ranges and the rules of the [columns](formats.html#layout-csv-columns)) and `links`, the values
  that the nodes of one site or terminal share ([Nodes and rows](formats.html#layout-csv-rows)), the antenna lists (a
  node with a band that is not set is reported), the frequencies, `wrap` and `pairing`
- **Antennas**: equal descriptions share one entry of the antenna lists, Tx nodes first; a relative `qdant_file` is
  stored as a normalised absolute path
- **Memory**: large arrays on the default 1° grid need much memory ([Antennas](formats.html#layout-csv-antennas)); give
  a `res` column or read with `eval_antennas=False`

## See also:
- [[layout.csv_write]] (stores a layout)
- [[eval_antennas]] (generates the antennas later)
- [[generate_antenna]] (the antenna types and arguments of the antenna columns)
MD!*/
py::tuple layout_csv_read(const py::handle &fn_tx,
                          const py::handle &fn_rx,
                          py::handle center_freq,
                          py::handle eval_antennas)
{
    const std::string fn_tx_s = qd_python_path(fn_tx, "fn_tx");
    const std::string fn_rx_s = qd_python_path(fn_rx, "fn_rx");
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const bool eval_antennas_b = qd_python_flag(eval_antennas, true, "eval_antennas");

    std::vector<std::string> extra_tx, extra_rx;
    const auto lay = quadriga_lib::layout_csv_read(fn_tx_s, fn_rx_s, center_freq_a, eval_antennas_b, &extra_tx, &extra_rx);
    return py::make_tuple(qd_python_layout2dict(lay), qd_python_copy2list(extra_tx), qd_python_copy2list(extra_rx));
}

/*!MD
# layout.csv_write
Write a layout to the Tx and Rx CSV files

- Writes the Tx nodes to the Tx file and the Rx nodes to the Rx file in the
  [Layout CSV format](formats.html#layout_csv); reading the files back gives the same layout within 9 significant
  digits, except for `name` and `links`, which are not stored
- Antennas without a description, e.g. a measured array set with [[set_antenna_array]], are written to QDANT files next
  to the CSV file
- A failed call leaves the existing files unchanged

## Usage:
```
quadriga_lib.layout.csv_write( fn_tx, fn_rx, layout, *, compact )
```

## Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files, str or path-like; `""` = not written; different files (also if the
  paths are spelled differently)
- **`layout`** — Layout dict, must be valid; keys in [[layout.csv_read]]
- **`compact`** — `True`: cells that the reader fills in are left empty; `False`: every cell is written; default: `True`

## Notes:
- **Rules of the writer**: the columns written, the number format, the compact form, the names of the QDANT files and
  what cannot be written are described in [Writing](formats.html#layout-csv-writing)
- **QDANT files**: one file `<stem>_<k>.qdant` per antenna without description, with one entry per band. Files of this
  name pattern next to a written CSV file that the new files do not refer to are deleted. The arrays must be evaluated:
  a layout whose `antenna` does not hold them raises an error. An antenna set from a QDANT file ([[set_antenna]]) is
  written as a reference to its file, not exported

## See also:
- [[layout.csv_read]] (reads a layout)
MD!*/
void layout_csv_write(const py::handle &fn_tx,
                      const py::handle &fn_rx,
                      const py::dict &layout,
                      py::handle compact)
{
    const std::string fn_tx_s = qd_python_path(fn_tx, "fn_tx");
    const std::string fn_rx_s = qd_python_path(fn_rx, "fn_rx");
    const auto lay = qd_python_dict2layout(layout, true, "layout");
    quadriga_lib::layout_csv_write(fn_tx_s, fn_rx_s, lay, qd_python_flag(compact, true, "compact"));
}
