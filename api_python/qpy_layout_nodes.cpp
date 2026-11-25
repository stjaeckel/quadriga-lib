// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# set_antenna
Set the antenna of layout nodes from the arguments of generate_antenna

- Sets the antenna of Tx nodes or Rx nodes from the arguments of [[generate_antenna]]; [[set_antenna_array]] sets it
  from an array, e.g. a measured pattern read with [[qdant_read]]
- Without `center_freq`, the antenna applies to all bands; with `center_freq`, it is the antenna of that band
- Stores the arguments as an antenna description, which [[layout.csv_write]] writes as antenna columns

## Usage:
```
layout_out = quadriga_lib.layout.set_antenna( layout_in, type, args, *, rx_side, i_node, center_freq, evaluate )
```

## Inputs:
- **`layout_in`** — Layout dict, keys in [[layout.csv_read]]; its antenna lists must fit its nodes and frequencies
- **`type`** — Antenna type of [[generate_antenna]], str, e.g. `"3GPP"`, `"dipole"`, or `"none"` (no antenna, for all
  bands, without `args`)
- **`args`** — Dict with the arguments of [[generate_antenna]] as keys, with the meaning given there: `res`,
  `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers,
  vectors or matrices (`steer`, `beams` row by row as there), `separate_beams` and `apply_weights` as bool or 0 or 1; a
  key that the type does not use raises an error; `pattern` is not available (an array built with it is set with
  [[set_antenna_array]]); the key `center_freq` is the same as the argument; e.g. `{"panel": [4, 4, 1, 2], "pol": 3}`;
  default: None (none)
- **`rx_side`** — `False`: set the antenna of Tx nodes (`layout_in["tx"]`); `True`: of Rx nodes (`layout_in["rx"]`);
  default: `False`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx_side`, 0-based; int or `(n,)`; `None`: all of them;
  default: `None`
- **`center_freq`** — Band in [Hz], positive; NaN: one antenna for all bands; default: NaN
- **`evaluate`** — `True`: generate all antennas of the layout that are not evaluated yet; `False`: none, the nodes of
  new antennas use the generic antenna until [[eval_antennas]] is called; default: `True`

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
layout = qd.layout.set_antenna( layout, "3GPP", {"panel": [4, 4, 1, 2], "pol": 3}, center_freq=3.5e9 )
```

## Notes:
- **Bands**: with `center_freq`, only that band of the antenna of the nodes is set. A frequency that is not in
  `layout["center_freq"]` is added; the other bands of the nodes keep their antenna, and nodes that shared the old entry
  keep it. Until all bands of a node are set, the layout is not valid, and the functions that check it name the node
- **Bands set one by one**: such a node keeps one line per band, also when the lines are equal, so a band added later is
  reported as not set for it. In a layout with one band, the line of a node set for that band also covers bands added
  later
- **Antenna lists**: after the call, equal descriptions share one entry, entries are numbered in order of first use (Tx
  nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- **Deferred evaluation**: an array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in
  a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- **`layout_in` / `layout_out`**: the C++ in/out argument `layout` is split, so that the input stays unchanged

## See also:
- [[generate_antenna]] (the meaning of the arguments)
- [[eval_antennas]] (generates antennas that are not evaluated yet)
- [[layout.csv_write]] (writes the lines and exports the arrays)
- [[set_antenna_array]] (sets an evaluated array)
- [[set_antenna_qdant]] (sets an entry of a QDANT file)
MD!*/
py::dict set_antenna(const py::dict &layout_in,
                     const std::string &type,
                     py::handle args,
                     py::handle rx_side,
                     py::handle i_node,
                     double center_freq,
                     py::handle evaluate)
{
    auto lay = qd_python_dict2layout(layout_in, false, "layout_in");
    const auto args_m = qd_pl_args(args, "args");
    const auto i_node_a = qd_python_numpy2arma_Col<arma::uword>(i_node, true, false, "i_node");

    quadriga_lib::set_antenna(lay, type, args_m, qd_python_flag(rx_side, false, "rx_side"), i_node_a, center_freq,
                              qd_python_flag(evaluate, true, "evaluate"));
    return qd_python_layout2dict(lay);
}

/*!MD
# set_antenna_array
Set the antenna of layout nodes from an evaluated array

- Sets the antenna of Tx nodes or Rx nodes to an array, e.g. a measured pattern read with [[qdant_read]]
- Without `center_freq`, the array applies to all bands; with `center_freq`, it is the antenna of that band
- Stores the array without description, which [[layout.csv_write]] exports to a QDANT file

## Usage:
```
layout_out = quadriga_lib.layout.set_antenna_array( layout_in, array, *, rx_side, i_node, center_freq )
```

## Inputs:
- **`layout_in`** — Layout dict, keys in [[layout.csv_read]]; its antenna lists must fit its nodes and frequencies
- **`array`** — Valid single-frequency array, dict as in [[generate_antenna]]
- **`rx_side`** — `False`: set the antenna of Tx nodes (`layout_in["tx"]`); `True`: of Rx nodes (`layout_in["rx"]`);
  default: `False`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx_side`, 0-based; int or `(n,)`; `None`: all of them;
  default: `None`
- **`center_freq`** — Band in [Hz], positive; NaN: one antenna for all bands; default: NaN

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
array, _ = qd.arrayant.qdant_read( "handset.qdant" )
layout = qd.layout.set_antenna_array( layout, array, rx_side=True )
```

## Notes:
- **Bands and antenna lists**: the rules of [[set_antenna]] apply
- **Name**: this is the second form of the C++ function `set_antenna`, which takes an array; it has a name of its own
  because Python does not overload by type
- **`layout_in` / `layout_out`**: the C++ in/out argument `layout` is split, so that the input stays unchanged

## See also:
- [[set_antenna]] (sets the antenna from the arguments of generate_antenna)
- [[qdant_read]] (reads an array from a QDANT file)
- [[layout.csv_write]] (exports the array to a QDANT file)
MD!*/
py::dict set_antenna_array(const py::dict &layout_in,
                           const py::dict &array,
                           py::handle rx_side,
                           py::handle i_node,
                           double center_freq)
{
    auto lay = qd_python_dict2layout(layout_in, false, "layout_in");
    const auto array_a = qd_pl_dict2arrayant(array, true, "array"); // The library validates the array
    const auto i_node_a = qd_python_numpy2arma_Col<arma::uword>(i_node, true, false, "i_node");

    quadriga_lib::set_antenna(lay, array_a, qd_python_flag(rx_side, false, "rx_side"), i_node_a, center_freq);
    return qd_python_layout2dict(lay);
}

/*!MD
# set_antenna_qdant
Set the antenna of layout nodes to an entry of a QDANT file

- Sets the antenna of Tx nodes or Rx nodes to an array stored in a QDANT file, e.g. a measured or simulated pattern
- The layout refers to the file: [[layout.csv_write]] writes the reference (the columns `antenna` = `qdant`,
  `qdant_file`, `qdant_id`), relative to the CSV file, and exports no copy of the array
- The array is used as stored, for all bands; the rules for bands and the antenna lists are those of [[set_antenna]]

## Usage:
```
layout_out = quadriga_lib.layout.set_antenna_qdant( layout_in, qdant_file, qdant_id, *, rx_side, i_node, center_freq, evaluate )
```

## Inputs:
- **`layout_in`** — Layout dict, keys in [[layout.csv_read]]; its antenna lists must fit its nodes and frequencies
- **`qdant_file`** — Existing QDANT file, str or path-like, absolute or relative to the current folder
- **`qdant_id`** — 1-based entry of the file; int; default: 1
- **`rx_side`** — `False`: set the antenna of Tx nodes (`layout_in["tx"]`); `True`: of Rx nodes (`layout_in["rx"]`);
  default: `False`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx_side`, 0-based; int or `(n,)`; `None`: all of them;
  default: `None`
- **`center_freq`** — Band in [Hz], positive; NaN: one antenna for all bands; default: NaN
- **`evaluate`** — `True`: generate all antennas of the layout that are not evaluated yet; `False`: none; default:
  `True`

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
layout = qd.layout.set_antenna_qdant( layout, "measured/handset.qdant", 2, rx_side=True )
```

## Notes:
- **Path**: the file is stored as a normalised absolute path
- **Errors**: a missing or unreadable file raises a `RuntimeError`; without evaluation, an entry that the file does not
  have raises an error when the antennas are evaluated
- **`layout_in` / `layout_out`**: the C++ in/out argument `layout` is split, so that the input stays unchanged

## See also:
- [[set_antenna]] (sets the antenna from the arguments of generate_antenna)
- [[set_antenna_array]] (sets an evaluated array)
- [[qdant_read]] (reads the entries of a QDANT file)
MD!*/
py::dict set_antenna_qdant(const py::dict &layout_in,
                           py::handle qdant_file,
                           arma::uword qdant_id,
                           py::handle rx_side,
                           py::handle i_node,
                           double center_freq,
                           py::handle evaluate)
{
    auto lay = qd_python_dict2layout(layout_in, false, "layout_in");
    const std::string fn = qd_python_path(qdant_file, "qdant_file");
    const auto i_node_a = qd_python_numpy2arma_Col<arma::uword>(i_node, true, false, "i_node");

    quadriga_lib::set_antenna_qdant(lay, fn, qdant_id, qd_python_flag(rx_side, false, "rx_side"), i_node_a, center_freq,
                                    qd_python_flag(evaluate, true, "evaluate"));
    return qd_python_layout2dict(lay);
}

/*!MD
# eval_antennas
Generate the antenna arrays of a layout from the antenna lines

- Generates the arrays of `antenna` that are not evaluated yet from the antenna descriptions `antenna_args`, e.g. after
  [[layout.csv_read]] or [[set_antenna]] without evaluation
- A frequency list `center_freq` replaces `layout_in["center_freq"]` and regenerates all arrays; arrays without
  description are kept and must fit the new list
- This is the member function `eval_antennas` of the C++ layout class

## Usage:
```
layout_out = quadriga_lib.layout.eval_antennas( layout_in, *, center_freq )
```

## Inputs:
- **`layout_in`** — Layout dict, keys in [[layout.csv_read]]
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `None` = keep
  `layout_in["center_freq"]`; `(n_freq,)`; default: None

## Outputs:
- **`layout_out`** — The layout with evaluated `antenna`

## Notes:
- **Antenna descriptions**: a line of `antenna_args` holds the cells of the antenna columns `antenna_cols` as in the
  [Layout CSV files](formats.html#layout-csv-antennas). Nodes with equal lines share one antenna; `i_antenna` of a node
  is its 1-based index, 0 the generic antenna. No line or an empty line is an array without description (set with
  [[set_antenna_array]]); a line without antenna type is a band that is not set yet
- **Evaluation**: a line for all bands is generated once per band at its frequency, so that an array spaced in
  wavelengths keeps its spacing (an array of co-located elements is stored once for all bands); a line of a band at the
  frequency of that band; a `qdant` line is read as stored. Until an antenna is evaluated (its entry of `antenna` holds
  no arrays, or an array without elements), its nodes use the generic antenna
- **Errors**: a band without description and an array spaced in wavelengths in a layout without frequencies raise an
  error; the message names the node
- **`layout_in` / `layout_out`**: the C++ in/out argument `layout` is split, so that the input stays unchanged

## See also:
- [[layout.csv_read]] (reads a layout, optionally with evaluated antennas)
- [[set_antenna]] (sets the antennas of nodes)
MD!*/
py::dict eval_antennas(const py::dict &layout_in,
                       py::handle center_freq)
{
    auto lay = qd_python_dict2layout(layout_in, false, "layout_in");
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);

    lay.eval_antennas(center_freq_a);
    return qd_python_layout2dict(lay);
}
