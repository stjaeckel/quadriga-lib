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

- This form stores the arguments of [[generate_antenna]] (see there for their meaning) as an antenna line of the layout, which [[layout.csv_write]] writes as CSV cells; an array set with [[set_antenna_array]] is exported to a QDANT file by [[layout.csv_write]]
- Without `center_freq`, the antenna applies to all bands. With `center_freq`, only that band of the nodes' antenna is set: a frequency that is not in `layout["center_freq"]` is added, the other bands of the nodes keep their antenna, and nodes that shared the old entry keep it; until all bands of a node are set, the layout is not valid and the functions that check it name the node
- A node set band by band keeps one line per band, also when the lines are equal, so a band added later is reported as not set for it; in a layout with one band, the line of a node set for that band also covers bands added later
- The antenna lists are encoded again after the call as [[layout.csv_read]] composes them: equal lines share one entry, entries are numbered in order of first use (Tx nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- An array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- A failed call leaves the layout unchanged
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## Usage:
```
layout_out = quadriga_lib.layout.set_antenna( layout_in, type, args, *, rx_side, i_node, center_freq, evaluate )
```

## Inputs:
- **`layout_in`** — Layout dict, keys in [[layout.csv_read]]; its antenna lists must fit its nodes and frequencies
- **`type`** — str; antenna type of [[generate_antenna]], e.g. `"3GPP"`, `"dipole"`, or `"none"` (no antenna, for all bands, without `args`)
- **`args`** — Dict with the arguments of [[generate_antenna]] as keys: `res`, `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers, vectors or matrices (`steer`, `beams` as there), `separate_beams` and `apply_weights` as bool or 0 or 1; a key that the type does not use raises an error; `pattern` is not available (an array built with it is set with [[set_antenna_array]]); the key `center_freq` is the same as the argument; e.g. `{"panel": [4, 4, 1, 2], "pol": 3}`; default: None (none)
- **`rx_side`** — `False`: nodes of `layout_in["tx"]`; `True`: nodes of `layout_in["rx"]`; default: `False`
- **`i_node`** — Indices of the nodes of that side, 0-based; int or `(n,)`; `None` = all nodes of the side; default: `None`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN
- **`evaluate`** — `True`: generate all antennas of the layout that are not evaluated yet; `False`: none, the nodes of new antennas use the generic antenna until [[eval_antennas]] is called; default: `True`

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
layout = qd.layout.set_antenna( layout, "3GPP", {"panel": [4, 4, 1, 2], "pol": 3}, center_freq=3.5e9 )
```

## See also:
- [[set_antenna_array]] (sets an evaluated array)
- [[set_antenna_qdant]] (sets an entry of a QDANT file)
- [[generate_antenna]] (the meaning of the arguments)
- [[layout.csv_write]] (writes the lines and exports the arrays)
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

- Stores an array, e.g. a measurement read with [[qdant_read]], which [[layout.csv_write]] exports to a QDANT file; for the antenna lists of the layout, the bands and the renumbering of `i_antenna` see [[set_antenna]]
- Without `center_freq`, the array applies to all bands. With `center_freq`, only that band of the nodes' antenna is set: a frequency that is not in `layout_in["center_freq"]` is added, the other bands of the nodes keep their antenna
- The C++ function `set_antenna` with an array argument has this name in Python, because Python does not overload by type; the in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## Usage:
```
layout_out = quadriga_lib.layout.set_antenna_array( layout_in, array, *, rx_side, i_node, center_freq )
```

## Inputs:
- **`layout_in`** — Layout dict, keys in [[layout.csv_read]]; its antenna lists must fit its nodes and frequencies
- **`array`** — Valid single-frequency array, dict as in [[generate_antenna]]
- **`rx_side`** — `False`: nodes of `layout_in["tx"]`; `True`: nodes of `layout_in["rx"]`; default: `False`
- **`i_node`** — Indices of the nodes of that side, 0-based; int or `(n,)`; `None` = all nodes of the side; default: `None`
- **`center_freq`** — Band in [Hz], positive; NaN = the same array for all bands; default: NaN

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
array, _ = qd.arrayant.qdant_read( "handset.qdant" )
layout = qd.layout.set_antenna_array( layout, array, rx_side=True )
```

## See also:
- [[set_antenna]] (sets the antenna from the arguments of generate_antenna)
- [[qdant_read]] (reads an array from a QDANT file)
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

- Stores a `qdant` antenna line with the file as normalised absolute path and the entry `qdant_id`, as [[layout.csv_read]] composes it from the columns `antenna`, `qdant_file` and `qdant_id`; [[layout.csv_write]] writes this line with the path relative to the CSV file and exports no copy of the array
- The array is read as stored, for all bands; bands, the sharing of entries and the renumbering of `i_antenna` follow [[set_antenna]]
- A missing or unreadable file raises an error; without evaluation, an entry that the file does not have raises an error when the antennas are evaluated
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## Usage:
```
layout_out = quadriga_lib.layout.set_antenna_qdant( layout_in, qdant_file, qdant_id, *, rx_side, i_node, center_freq, evaluate )
```

## Inputs:
- **`layout_in`** — Layout dict, keys in [[layout.csv_read]]; its antenna lists must fit its nodes and frequencies
- **`qdant_file`** — str or path; existing QDANT file, absolute or relative to the current folder
- **`qdant_id`** — int; 1-based entry of the file; default: 1
- **`rx_side`** — `False`: nodes of `layout_in["tx"]`; `True`: nodes of `layout_in["rx"]`; default: `False`
- **`i_node`** — Indices of the nodes of that side, 0-based; int or `(n,)`; `None` = all nodes of the side; default: `None`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN
- **`evaluate`** — `True`: generate all antennas of the layout that are not evaluated yet; `False`: none; default: `True`

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
layout = qd.layout.set_antenna_qdant( layout, "measured/handset.qdant", 2, rx_side=True )
```

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

- Generates the arrays of `antenna` that are not evaluated yet from `antenna_args`; a line without `center_freq` once per simulation frequency, so that an array spaced in wavelengths keeps its spacing in each band (co-located elements give one array for all bands), a line of a band at its frequency, a `qdant` line by reading its entry as stored
- A frequency list `center_freq` replaces `layout_in["center_freq"]` and regenerates all lines; arrays without a line are kept and must fit the new list
- Raises an error for a band without description and for an array spaced in wavelengths without a frequency; the message names the node
- This is the member function `eval_antennas` of the C++ layout class; its in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## Usage:
```
layout_out = quadriga_lib.layout.eval_antennas( layout_in, *, center_freq )
```

## Inputs:
- **`layout_in`** — Layout dict, keys in [[layout.csv_read]]
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `None` = keep `layout_in["center_freq"]`; default: None; `(n_freq,)`

## Outputs:
- **`layout_out`** — The layout with evaluated `antenna`

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
