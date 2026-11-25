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
# GET_GBSM_CHANNELS
Channel coefficients of the channel links of propagation links of a layout (stage 3 of the channel pipeline)

- One channel per Tx node, Rx node and frequency whose propagation link is in `i_link` and whose pair is in `layout.pairing` (empty: all); order: Tx node fastest, then Rx node, then frequency; name `<tx>:<sector>_<rx>:<sector>`, without the sector where all sectors of the side are 0
- Snapshots of both ends from [[sample_track]] over `times`; a node with one snapshot is repeated to the count of the other; two tracks without `times` need equal row counts
- Link gain per snapshot (see the GBSM terms): [[gbsm_draw]] of the `'lsf'` set at the snapshot, [[path_loss]] with the random environment height `rnd_he` held at the start of the segment, [[lsp_generate]] (SF and the K-factor ratio), [[o2i_loss]] for building and car UTs with the model of the link state, the deviation of a building UT at the snapshot, that of a car UT at the first row of the terminal for its whole track (TR 38.901 eq. 7.4-4: one value per UT); evaluated once for a link whose ends do not move
- [[gbsm_single_link_channel]] with the node antennas at the frequency (entry `i_freq` of the `1xN` struct array of the antenna in `layout.antenna`, or its one entry for all bands), the generic antenna with two co-located isotropic elements of vertical and horizontal polarization for nodes without one and for antennas that are not evaluated (an antenna spaced in wavelengths that is not evaluated raises an error), and the Tx positions moved by the wrap image of the link; rays with `n_freq` entries are used as entry `i_freq` (TR methods), rays with one entry at frequency `i_freq` (`'quadriga'`)
- Tracks with several segments: the snapshots of a segment are those whose row (see [[sample_track]]) is in it; each segment also covers the snapshots of the preceding one, with its link gain, rays and K-factor ratio referred to its first row; [[crossfade_segments]] joins them. Relative delays: in `'phase'` the paths of each segment relative to its own LOS, in `'drift'` relative to the LOS at the snapshot. The selected segments of a channel link must be consecutive (segments without snapshots are skipped), with increasing `times`
- The index output is the second output

## Usage:
```
[ channels, index ] = quadriga_lib.get_gbsm_channels( layout, rays, i_link, times, output, mobility, ...
    oxygen, use_absolute_delays, overlap );
```

## Inputs:
- **`layout`** — Layout struct with links from [[init_link_states]]; fields in [[layout_csv_read]]; raises an error if a member that determines the links changed since then
- **`rays`** — Rays of the links from [[generate_rays]], possibly edited, with the frequencies of the layout; cell array `[n_out, 1]` of `1xN` struct arrays, `N = 1` or `n_freq`
- **`i_link`** — The propagation links of `rays`, 1-based; `[]` = all links; default: `[]`; `[n_out]`
- **`times`** — Snapshot times in [s], finite, the same for all nodes; `[]` = one snapshot per row; default: `[]`; `[n_snap]`
- **`output`** — `'taps'` or `'rays'`, see [[gbsm_single_link_channel]]; default: `'taps'`
- **`mobility`** — `'phase'`, `'drift'` (needs rays with scatterers) or `'auto'` (phase for the TR methods, drift for `'quadriga'`); default: `'auto'`
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `false`
- **`use_absolute_delays`** — Include the LOS delay in all delays; default: `false`
- **`overlap`** — Part of the snapshots shared by consecutive segments that the cross-fade uses, in [0, 1], see [[crossfade_segments]]; default: 0.5

## Outputs:
- **`channels`** — Channels, struct array, fields in [[hdf5_read_channel]]; `[1, n_channel]`
- **`index`** — Tx node, Rx node and frequency index of each channel, 1-based; uint64; `[3, n_channel]`

## See also:
- [[generate_rays]] (stage 2)
- [[get_channels_layout]] (all stages in one call)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 9)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], false, "layout");
    const auto rays = qd_mex_cell2rays(prhs[1], "rays");
    const arma::uvec i_link = (nrhs < 3) ? arma::uvec() : qd_mex_get_index_Col<arma::uword>(prhs[2], "i_link"); // 1-based -> 0-based
    const arma::vec times = (nrhs < 4) ? arma::vec() : qd_mex_get_Col<double>(prhs[3], false, "times");
    const std::string output = (nrhs < 5) ? std::string("taps") : qd_mex_get_string(prhs[4], "taps", "output");
    const std::string mobility = (nrhs < 6) ? std::string("auto") : qd_mex_get_string(prhs[5], "auto", "mobility");
    const bool oxygen = (nrhs < 7) ? false : qd_mex_get_scalar<bool>(prhs[6], "oxygen", false);
    const bool use_absolute_delays = (nrhs < 8) ? false : qd_mex_get_scalar<bool>(prhs[7], "use_absolute_delays", false);
    const double overlap = (nrhs < 9) ? 0.5 : qd_mex_get_scalar<double>(prhs[8], "overlap", 0.5);

    std::vector<quadriga_lib::channel<double>> channels;
    arma::umat index;
    arma::umat *p_index = (nlhs > 1) ? &index : nullptr;
    CALL_QD(channels = quadriga_lib::get_gbsm_channels(layout, rays, i_link, times, output, p_index, mobility, oxygen, use_absolute_delays, overlap));

    plhs[0] = qd_mex_channel2struct(channels);
    if (nlhs > 1)
        plhs[1] = qd_mex_layout_index_out(index);
}
