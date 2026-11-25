// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!SECTION_DESC
All functions of this section are in the submodule `quadriga_lib.layout`: `generate_layout` is called as `quadriga_lib.layout.generate_layout`; `layout_csv_read` and `layout_csv_write` are `layout.csv_read` and `layout.csv_write`. `crossfade_segments` is in `quadriga_lib.channel`. A layout is a dict with the member names of the C++ class `layout`: `tx`, `rx` and `links` are nested dicts, `scenpar_override` is a dict with one array per scenario key, `antenna` is a list with one entry per antenna, each a list of arrayant dicts, one per band (the 4D multi-frequency dict of [[generate_antenna]] cannot hold per-band geometry), and `rays` are lists of ray dicts, see [[ssf_rays]].<br><br>

A **layout** describes the nodes of a radio network and how they move: where each transmitter (Tx) and receiver (Rx) is, which scenario applies there and which antenna it has. It is stored in two flat tables, one per side, and in two CSV files with the same format.

**Terms.** A *node* is one (`name`, `sector`) pair on one side. It has one antenna and one *track*: its *rows*, each a position with orientation and further values that apply from that row on. A node with one row is a *point drop*; it stays at its position, or, with a speed, moves along its direction. A node with several rows is a track and moves along its positions. A *site* is all Tx nodes with one name; its sectors share position and scenario. A *terminal* is all Rx nodes with one name, e.g. the panels of one device. Angles are in degrees in the CSV files and in radians in the dicts. The orientation (bank, tilt, heading) is that of the antenna, a positive tilt points down; the optional direction (azimuth, elevation of travel) is that of the movement, and an unset direction follows the orientation.

**CSV files.** [[layout.csv_read]] and [[layout.csv_write]] read and write the files. The first line is the header; the columns are optional, in any order, except `name`, `x`, `y`, `z`. Cells follow the syntax of the other CSV files of the library: quoting with `"`, lines starting with `#` are comments, numbers are decimal (NaN and Inf in any case). Empty cells follow the class of the column:

| Class | Columns | An empty cell is |
| ----- | ------- | ---------------- |
| Identity | `name`, `sector` | `name` is required; an empty `sector` is 0 |
| Per row | `x`, `y`, `z`, `time` | an error; except for the one row of an extra sector of a static site, which takes the position of sector 0; `time` is given on all rows of a node or on none |
| Held | `bank`, `tilt`, `heading`, `direction_az`, `direction_el`, `speed`, `scenario`, `indoor`, `floor`, `d2d_in` | the value of the previous row of the node; on the first row the value of sector 0 for an extra sector, else the default; NaN is a value |
| Fixed | all others | the value of the first row, for an extra sector that of sector 0; later rows are empty or repeat it, and a different value is an error |

| Column | Unit | Meaning |
| ------ | ---- | ------- |
| `bank`, `tilt`, `heading` | degree | Antenna orientation, default 0 |
| `direction_az`, `direction_el` | degree | Direction of travel; empty or NaN azimuth follows the orientation; only for point drops |
| `speed` | m/s | Speed from this row on, default 0 |
| `time` | s | Time of the row, strictly increasing; a node has `time` or a speed |
| `scenario` | | Scenario family or conf file name, never a path |
| `indoor`, `floor`, `d2d_in` | | Indoor state (-1 or empty: draw, 0 outdoor, 1 building, 2 car, 3 residential building), floor from 1 (-1: from z), indoor 2D distance in m (NaN: draw); they change only where `scenario` or `indoor` changes |
| `d_clutter`, `r_clutter`, `h_clutter`, `scenpar.<key>` | | Tx side: clutter inputs and overrides of scenario parameters, equal for all sectors of a site |
| `seed`, `layout_seed`, `wrap` | | Tx side: seed of the site, seed of the layout, lattice vectors in m (`"ax,ay;bx,by"`) |
| `pairing` | | Rx side: names of the Tx sites linked to the terminal, empty: all |
| `relative` | | 1: the orientation is relative to the direction of motion |
| `center_freq`, `antenna`, `qdant_file`, `qdant_id` and the inputs of [[generate_antenna]] | | Antenna columns, see below |

**Antennas.** The antenna columns of a node (type `antenna` as in [[generate_antenna]], or `qdant` with the file `qdant_file` relative to the CSV file and its 1-based entry `qdant_id`) are kept as one text line per antenna in `antenna_args`, with the names of the columns in `antenna_cols`. Nodes with equal lines share one entry of the lists, Tx nodes first, and `i_antenna` is the 1-based index of it, 0 for the generic antenna with two co-located isotropic elements of vertical and horizontal polarization. A column that the type does not use is an error. An extra sector without antenna cells has the antenna of sector 0; the type `none` (no other antenna cells) gives a node no antenna, e.g. an extra sector next to a sector 0 with one. [[eval_antennas]] generates the arrays `antenna` from the lines: a line without `center_freq` once per simulation frequency, so that an array spaced in wavelengths keeps its spacing in each band (co-located elements give one array for all bands), a line of a band at its frequency, a `qdant` line by reading its entry as stored. [[set_antenna]] sets the antenna of nodes from the arguments of [[generate_antenna]], [[set_antenna_array]] from an evaluated array, which [[layout.csv_write]] writes to a QDANT file next to the CSV file; [[set_antenna_qdant]] sets an entry of a QDANT file, which the writer refers to.

**Bands.** A row with `center_freq` and no position adds a frequency, and optionally an antenna for it, to its node. The frequencies of the Tx side (else of the Rx side) are the simulation frequencies `center_freq` of the layout. A node without bands has all of them with one antenna description; a node with bands defines all of them. An empty antenna in a band repeats the antenna of the first band.

**Unknown columns** are kept as text in `extra_names` and `extra` and written back unchanged.

**Channels from a layout.** The channel coefficients are computed in three stages. [[init_link_states]] draws the discrete state of each *propagation link*, one site and one terminal: its wrap image, scenario and conf file, LOS and indoor state and seeds, stored in `layout["links"]`. [[generate_rays]] runs the chain of the section "Geometry-based stochastic model" for chosen links and returns their rays. [[get_gbsm_channels]] applies the antennas, the motion along the tracks and the link gain, for each *channel link*, one Tx node and one Rx node; the sectors of a site share the rays of their propagation link. [[get_channels_layout]] runs all three stages. [[attach_terminals]] chooses the serving Tx node and site of each terminal by the large-scale gain or by RSRP, for `layout["pairing"]`. [[export_rays_qrt]] writes the rays with their link gain to a QRT file, which gives channels for other antennas and positions without the GBSM chain. All random values are addressed: they follow from the seeds, a stream key and the position, so that the evaluation order, the batches and the thread count do not change them, and only the discrete states are stored. `layout["links"]` holds a hash of the layout members that determine the links (positions, names, sectors, frequencies, site seeds, scenarios, the O2I columns, clutter, scenario overrides, wrap vectors and whether a site moves); stages 2 and 3 throw when it no longer matches, and `layout["links"] = {}` discards the links. The links hold the master seed they were drawn with, so a copy of `layout["links"]` can be restored after another realisation. Editing `los` or `indoor` of the links between stage 1 and stage 2 forces a state; orientations, antennas and `pairing` may change without new links.

**Indices.** All indices are 0-based, as in C++: the members `offset`, `pairing` and, in `links`, `i_tx`, `i_rx`, `i_segment`, `i_conf` and `o2i_model`, the index arguments of the functions and the functions that return indices. `i_antenna` is the index into the antenna lists with 0 = generic antenna, as in the other APIs; `sector`, `image`, `floor_id`, `indoor`, `floor` and the seeds are values.
SECTION_DESC!*/

/*!MD
# layout.csv_read
Read the Tx and Rx CSV files of a layout

- The nodes of each file are created in order of first appearance; the rows of a node need not be consecutive in the file, and their order is the track order. Angles are converted from degrees to radians
- Columns, the meaning of empty cells, antenna columns and bands are described at the top of this section. Antenna columns are composed into `antenna_cols`, `antenna_args` and `i_antenna`; equal descriptions share one entry, Tx nodes first. A relative `qdant_file` is resolved against the folder of the CSV file and stored as a normalised absolute path
- Simulation frequencies: the argument, else the `center_freq` values of the Tx file, else those of the Rx file; a node with bands must define all of them, and the values in the files must match the argument within a relative 1e-6
- Unknown columns are kept and reported; a column in the wrong file (e.g. `pairing` in the Tx file), a value outside its range, an antenna column that the antenna type does not use and every violation of the rules above raise an error with the file name, line and column; so does a terminal whose empty `pairing` cell leaves it without a pair, a first band without antenna cells next to a later band with antenna cells, and a layout that is not valid, e.g. a name with a comma or a node with one row, `relative` = 1 and no direction
- With `eval_antennas`, the arrays `antenna` are generated as [[eval_antennas]] does; a line that [[generate_antenna]] rejects, a QDANT file that cannot be read and an array spaced in wavelengths in a layout without frequencies raise an error

## Usage:
```
layout, extra_tx, extra_rx = quadriga_lib.layout.csv_read( fn_tx, fn_rx, *, center_freq, eval_antennas )
```

## Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files, str or path-like; `""` = no nodes on that side
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `None` = from the files; `(n_freq,)`; default: None
- **`eval_antennas`** — `True`: generate the antennas; `False`: `antenna` stays empty, `antenna_args` and `i_antenna` are kept, and the nodes use the generic antenna until [[eval_antennas]] is called; default: `True`

## Outputs:
- **`layout`** — The layout, a dict with the members of the layout class, which all functions of this section take and return; `name` and `links` are empty; angles in rad, all indices 0-based unless noted. Keys of `layout`, of the node dicts `layout["tx"]` and `layout["rx"]` and of `layout["links"]` (a key that is missing or None in an input dict takes the value of an empty member, unknown keys are rejected):<br><br>
  | Key | Meaning | Type / Size |
  | --- | ------- | ----------- |
  | `name` | Name of the layout, not stored in the CSV files | str |
  | `tx`, `rx` | Nodes of the Tx (BS, AP) and Rx (UT, STA) side, see below | dict |
  | `links` | State of the propagation links of the current realisation, see below; `{}` = none | dict |
  | `seed` | Seed of the layout, -1 = not set | int |
  | `center_freq` | Simulation frequencies in [Hz], strictly ascending | `(n_freq,)` |
  | `antenna` | Evaluated antennas, entry `k` is a list of N arrayant dicts as in [[generate_antenna]], one per band, N = 0 (not evaluated), 1 (all bands) or `n_freq` | list `[n_antenna]` of lists |
  | `antenna_cols` | Names of the antenna columns of the CSV files | list of str |
  | `antenna_args` | CSV lines of the antenna columns of each antenna: none, one for all bands or one per band; `""` = an evaluated array without description | list of lists of str |
  | `wrap` | Lattice vectors of the wrap-around in [m], one per column; empty = none | `(2, 1)` or `(2, 2)` |
  | `pairing` | Tx site index and Rx terminal index of the linked pairs, 0-based; empty = all pairs | int64, `(2, n_pair)` |
  | `name` (`tx`, `rx`) | Site or terminal name of each node | list of str `[n_node]` |
  | `sector` | Sector index | uint32, `(n_node,)` |
  | `i_antenna` | Index into the antenna lists of the layout, 1-based, 0 = generic antenna of two co-located polarized isotropic elements | uint32, `(n_node,)` |
  | `offset` | First row of each node, 0-based | int64, `(n_node,)` |
  | `pos` | Positions in [m] | `(3, n_row)` |
  | `orientation` | Bank, tilt, heading in [rad]; a positive tilt points down | `(3, n_row)` |
  | `direction` | Azimuth, elevation of travel in [rad]; NaN azimuth = follow the orientation; empty = all NaN | `(2, n_row)` |
  | `speed` | Speed in [m/s] from this row on; empty = 0 | `(n_row,)` |
  | `time` | Time in [s]; NaN on all rows of a node = not given; empty = none | `(n_row,)` |
  | `scenario` | Scenario family or conf name, `""` = none; empty = none | list of str `[n_row]` |
  | `indoor` | -1 = draw, 0 = outdoor, 1 = building, 2 = car, 3 = residential building; empty = all -1 | int32, `(n_row,)` |
  | `floor` | Floor number from 1, -1 = from z; empty = all -1 | int32, `(n_row,)` |
  | `d2d_in` | Indoor 2D distance in [m], NaN = draw; empty = all NaN | `(n_row,)` |
  | `clutter` | `d_clutter`, `r_clutter`, `h_clutter`, NaN = model default; empty = all NaN | `(3, n_node)` |
  | `scenpar_override` | Overrides of scenario parameters, one key per parameter, NaN = none | dict of `(n_node,)` |
  | `seed` (`tx`, `rx`) | Seed of the site, -1 = derived from the layout seed; empty = all -1 | int64, `(n_node,)` |
  | `relative` | 1 = orientation relative to the direction of motion; empty = all 0 | uint32, `(n_node,)` |
  | `extra_names`, `extra` | Names of unknown CSV columns and their cells as text | list of str `[n_extra]`; list `[n_extra]` of lists `[n_row]` of str |
  | `i_tx`, `i_rx`, `i_segment` (`links`) | Site, terminal and segment of each propagation link, 0-based | uint32, `(n_link,)` |
  | `image` | Wrap-around image of the site, 0 = the real site | uint32, `(n_link,)` |
  | `i_conf`, `conf_names` | Index into `conf_names` for each link, 0-based; resolved conf names | uint32, `(n_link,)`; list of str |
  | `los`, `indoor`, `floor_id`, `d2d_in` | 1 = LOS; indoor state at the start of the segment; floor number - 1; indoor 2D distance in [m] | `(n_link,)` |
  | `o2i_rnd`, `o2i_model` | Random inputs of [[o2i_loss]] and the chosen building model (0-based) of each terminal | `(n_terminal, 4)`; uint32, `(n_terminal,)` |
  | `gbsm_seed`, `seed` (`links`) | Seed of the random phases of each link; master seed | int64, `(n_link,)`; int |
  | `method`, `sos_preset`, `anchor_freq`, `pol_variability` | Method and SOS preset as used, anchor frequency in [Hz], polarization power variability | str, str, float, bool |
  | `conf_files`, `layout_hash` | Resolved paths of the conf files; hash of the layout members that determine the links | list of str; int |
- **`extra_tx`**, **`extra_rx`** — Names of the unknown columns of the Tx and Rx file, in file order; list of str

## See also:
- [[layout.csv_write]] (stores a layout)
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

- Writes only columns with content other than the default, in the order of the format, unknown columns last; angles in degrees, numbers with up to 15 significant digits (16 or 17 where 15 do not read back); reading the files gives the same layout within these digits, except for `name` and `links`, which are not stored
- With `compact` the cells that the reader repeats from the previous row of the node (held values) or sets to the default (fixed values, which are written on the first row only) are left empty; an extra sector (sector above 0 of a site that has a sector 0) has all its cells in its first row, because an empty cell would be the value of sector 0
- Antennas are written as their CSV lines, with each `qdant_file` relative to the folder of its CSV file. If no antenna line carries the frequencies of the layout, they are declared on the first node: `center_freq` on its first row and one band row per further frequency
- All files are composed first and written to temporary files next to the targets; each target is replaced by a rename only after all files were composed and written, so an error in the layout, the composition or the writing leaves all files unchanged (a failing rename of the second file is not undone for the first)
- Antenna bands that are evaluated arrays without a line (set with [[set_antenna_array]]) are written to QDANT files next to the first written CSV file that uses the antenna, named `<stem>_<k>.qdant` (`<stem>` is the name of the CSV file without `.csv`, k = 1, 2, ... per CSV file, skipping names that `qdant` lines of the layout use), one file per antenna with one entry per exported band, and become `qdant` lines. Afterwards, the files `<stem>_<k>.qdant` next to a written CSV file that the new files do not reference are deleted; files of other names are never touched. These arrays must be evaluated: a layout whose `antenna` does not hold them raises an error
- A band set with [[set_antenna_qdant]] is a `qdant` line: it is written as a reference to its file, relative to the CSV file, and not exported
- The `seed`, `layout_seed` and `wrap` of the layout are stored in the Tx file; a Tx file for a layout without Tx nodes that has them raises an error. Not expressible in the format and therefore rejected: a scenario that returns to none after a scenario was set, a terminal without a pair in `pairing` (an empty cell means all sites). An extra sector without antenna next to a sector 0 with one is written with the antenna `none`

## Usage:
```
quadriga_lib.layout.csv_write( fn_tx, fn_rx, layout, *, compact )
```

## Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files, str or path-like; `""` = not written; different files (also if the paths are spelled differently)
- **`layout`** — Layout dict, must be valid; fields in [[layout.csv_read]]
- **`compact`** — `True`: empty cells where the format allows; `False`: every cell is written; default: `True`

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
