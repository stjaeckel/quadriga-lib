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
# GET_CHANNELS_LAYOUT
Channel coefficients of a layout in one call

- Computes the channels in one call from the layout: one channel per Tx node, Rx node and frequency, for the pairs in
  `layout.pairing` (empty: all pairs); [[get_channels_from_rays]] is the third stage alone, from given rays
- Supports tracks (with times or speeds), point drops that move with their speed, tracks of several segments, joined by
  the cross-fade of [[crossfade_segments]], and both output forms, `'taps'` and `'rays'`
- Runs the stages [[init_link_states]], [[generate_rays]] and [[get_channels_from_rays]] with their default settings;
  the results are those of the stages called one by one
- Without link states in `layout.links`, draws them for this call only and leaves the layout unchanged; with link
  states, uses them, also when they were edited

## Usage:
```
[ channels, index ] = quadriga_lib.get_channels_layout( layout, times, output, method, mobility, ...
    overlap, seed, config_dirs );
```

## Inputs:
- **`layout`** — Layout struct with `center_freq`; fields in [[layout_csv_read]]; antennas as evaluated in
  `layout.antenna`, nodes without one use the generic antenna (see [[get_channels_from_rays]])
- **`times`** — Snapshot times in [s]; `[]` = one snapshot per row; default: `[]`; `[n_snap]`
- **`output`** — `'taps'` or `'rays'`; default: `'taps'`
- **`method`** — `'38.901'`, `'38.901_MF'` or `'quadriga'`; `''` = the method of filled links, else `'38.901'`; another
  method than that of filled links raises an error; default: `''`
- **`mobility`** — `'phase'`, `'drift'` or `'auto'`; default: `'auto'`
- **`overlap`** — Length of the cross-fade ramp between consecutive segments as a part of the shorter one, in [0, 1],
  centred on the first row of the new segment; 0 = a switch at that row, see [[crossfade_segments]]; default: 0.5
- **`seed`** — Master seed for empty links, as in [[init_link_states]]; with filled links only -1; default: -1
- **`config_dirs`** — Folders of user conf files for empty links, searched before the shipped folder; string or cell
  array of strings; default: `{}`

## Outputs:
- **`channels`** — Channels, Tx node fastest, then Rx node, then frequency; struct array, fields in
  [[hdf5_read_channel]]; `[1, n_channel]`
- **`index`** — Tx node, Rx node and frequency index of each channel, 1-based; uint64; `[3, n_channel]`

## Example:
```
layout = quadriga_lib.generate_layout( '38.901_UMa', 1, 3.5e9 );
channels = quadriga_lib.get_channels_layout( layout );
```

## Notes:
- **Memory**: the links are processed in batches of at most 256 propagation links (the segments of a track in one
  batch), so that the rays of one batch bound the memory; the results do not depend on the batches
- **Link states**: `layout.links` holds none when `layout.links.method` is empty, e.g. after `layout.links = []`. With
  link states, `seed` and `config_dirs` are not used. A layout without linked pairs gives an empty result
- **Scatterers**: the mobility `'drift'` (also `'auto'` with `'quadriga'`) solves the scatterers
- The C++ output `index` is the second output; the wrapper adds the folder of the shipped conf files to `config_dirs`

## See also:
- [[init_link_states]], [[generate_rays]], [[get_channels_from_rays]] (the three stages)
- [[generate_layout]] (layouts of the TR scenarios)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], false, "layout");
    const arma::vec times = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1], false, "times");
    const std::string output = (nrhs < 3) ? std::string("taps") : qd_mex_get_string(prhs[2], "taps", "output");
    const std::string method = (nrhs < 4) ? std::string() : qd_mex_get_string(prhs[3], "", "method");
    const std::string mobility = (nrhs < 5) ? std::string("auto") : qd_mex_get_string(prhs[4], "auto", "mobility");
    const double overlap = (nrhs < 6) ? 0.5 : qd_mex_get_scalar<double>(prhs[5], "overlap", 0.5);
    const arma::sword seed = (nrhs < 7) ? -1 : (arma::sword)qd_mex_get_seed(prhs[6], "seed");
    const std::vector<std::string> config_dirs = qd_mex_layout_config_dirs((nrhs < 8) ? nullptr : prhs[7]);

    std::vector<quadriga_lib::channel<double>> channels;
    arma::umat index;
    arma::umat *p_index = (nlhs > 1) ? &index : nullptr;
    CALL_QD(channels = quadriga_lib::get_channels_layout(layout, times, output, method, p_index, mobility, overlap, seed, config_dirs));

    plhs[0] = qd_mex_channel2struct(channels);
    if (nlhs > 1)
        plhs[1] = qd_mex_layout_index_out(index);
}
