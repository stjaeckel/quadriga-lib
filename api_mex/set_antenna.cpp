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
Set the antenna of layout nodes from the arguments of generate_antenna, a QDANT file, or by index

- Sets the antenna of Tx nodes or Rx nodes from the arguments of [[generate_antenna]] or from an entry of a QDANT file,
  or, with a numeric second input, to antennas that the layout already has, by their index; [[set_antenna_array]] sets
  an array, e.g. a measured pattern read with [[qdant_read]]
- Without `center_freq`, the antenna applies to all bands; with `center_freq`, it is the antenna of that band
- Stores an antenna description, which [[layout_csv_write]] writes as antenna columns (a QDANT file as a reference,
  without a copy)

## Usage:
```
layout_out = quadriga_lib.set_antenna( layout_in, type, args, rx, i_node, center_freq, evaluate );
layout_out = quadriga_lib.set_antenna( layout_in, i_antenna, rx, i_node, center_freq );
```

## Inputs:
- **`layout_in`** — Layout struct, fields in [[layout_csv_read]]; its antenna lists must fit its nodes and frequencies
- **`type`** — Antenna type of [[generate_antenna]], e.g. `'3GPP'`, `'dipole'`; a QDANT file, i.e. a name ending in
  `.qdant` (any case), absolute or relative to the current folder; or `'none'` (no antenna, for all bands, without
  `args`)
- **`args`** — Struct with the arguments of [[generate_antenna]] as fields, with the meaning given there: `res`,
  `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers,
  vectors or matrices in the shapes given there (a `beams` vector is one beam), `separate_beams` and `apply_weights` as
  0 or 1; a field that the type does not use raises an error; `pattern` is not available (an array built with it is set
  with [[set_antenna_array]]); the field `center_freq` is the same as the argument; e.g.
  `struct( 'panel', [4 4 1 2], 'pol', 3 )`. For a QDANT file only `id`, the 1-based entry of the file (default 1).
  Default: `[]` (none)
- **`i_antenna`** — Antenna entries of the layout, 1-based as `i_antenna` of the nodes, 0 = generic antenna; one value
  for all chosen nodes or one per node in the order of `i_node`; `[]` = 0
- **`rx`** — `false`: set the antenna of Tx nodes (`layout_in.tx`); `true`: of Rx nodes (`layout_in.rx`); default:
  `false`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx`, 1-based; `[]` = all of them; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; with `i_antenna`, a band of the layout,
  which the nodes take from the given entries; default: NaN
- **`evaluate`** — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none, the nodes of
  new antennas use the generic antenna until [[eval_antennas]] is called; default: `true`

## Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

## Example:
```
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna( layout, '3GPP', struct( 'panel', [4 4 1 2], 'pol', 3 ), false, [], 3.5e9 );
layout = quadriga_lib.set_antenna( layout, 'measured/handset.qdant', struct( 'id', 2 ), true );
layout = quadriga_lib.set_antenna( layout, 2, true, [ 1, 2 ] );
```

## Notes:
- **Bands**: with `center_freq`, only that band of the antenna of the nodes is set. A frequency that is not in
  `layout_in.center_freq` is added; the other bands of the nodes keep their antenna, and nodes that shared the old entry
  keep it. Until all bands of a node are set, the layout is not valid, and the functions that check it name the node
- **Bands set one by one**: such a node keeps one line per band, also when the lines are equal, so a band added later is
  reported as not set for it. In a layout with one band, the line of a node set for that band also covers bands added
  later
- **Antenna lists**: after the call, equal descriptions share one entry, entries are numbered in order of first use (Tx
  nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- **Deferred evaluation**: an array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in
  a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- **QDANT file**: stored as a normalised absolute path; [[layout_csv_write]] writes it relative to the CSV file. The
  array is used as stored, not rescaled to the frequency of a band. A missing or unreadable file raises an error;
  without evaluation, an entry that the file does not have raises an error when the antennas are evaluated
- **Index form**: with `center_freq`, that band of the antenna of each node becomes the band of its entry, copied as in
  the other forms; the generic antenna (0) applies to all bands only. A node listed twice in `i_node` with different
  entries raises an error
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## See also:
- [[generate_antenna]] (the meaning of the arguments)
- [[qdant_read]] (reads the entries of a QDANT file)
- [[eval_antennas]] (generates the antennas)
- [[layout_csv_write]] (writes the descriptions and exports the arrays)
- [[set_antenna_array]] (sets an array)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], true, "layout_in");

    // Index form: numeric second input
    if (mxIsNumeric(prhs[1]) || mxIsLogical(prhs[1]))
    {
        if (nrhs > 5)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
        const arma::uvec i_antenna = qd_mex_get_Col<arma::uword>(prhs[1], true, "i_antenna"); // 1-based, 0 = generic antenna; no shift intended
        const bool rx = (nrhs < 3) ? false : qd_mex_get_scalar<bool>(prhs[2], "rx", false);
        const arma::uvec i_node = (nrhs < 4) ? arma::uvec() : qd_mex_get_index_Col<arma::uword>(prhs[3], "i_node"); // 1-based -> 0-based
        const double center_freq = (nrhs < 5) ? NAN : qd_mex_get_scalar<double>(prhs[4], "center_freq", NAN);
        CALL_QD(quadriga_lib::set_antenna(layout, quadriga_lib::uint_list(i_antenna), rx, i_node, center_freq));
        plhs[0] = qd_mex_layout2struct(layout);
        return;
    }

    const std::string type = qd_mex_get_string(prhs[1], "", "type");
    const auto args = (nrhs < 3) ? std::unordered_map<std::string, arma::mat>() : qd_mex_layout_args(prhs[2], "args");
    const bool rx = (nrhs < 4) ? false : qd_mex_get_scalar<bool>(prhs[3], "rx", false);
    const arma::uvec i_node = (nrhs < 5) ? arma::uvec() : qd_mex_get_index_Col<arma::uword>(prhs[4], "i_node"); // 1-based -> 0-based
    const double center_freq = (nrhs < 6) ? NAN : qd_mex_get_scalar<double>(prhs[5], "center_freq", NAN);
    const bool evaluate = (nrhs < 7) ? true : qd_mex_get_scalar<bool>(prhs[6], "evaluate", true);

    CALL_QD(quadriga_lib::set_antenna(layout, type, args, rx, i_node, center_freq, evaluate));

    plhs[0] = qd_mex_layout2struct(layout);
}
