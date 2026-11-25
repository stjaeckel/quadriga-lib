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
# ATTACH_TERMINALS
Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP

- Evaluates all Tx nodes for each terminal at the first rows of both nodes and `layout.center_freq[i_freq]`, with the link states of the first segment, the wrap image and the antennas as in [[get_gbsm_channels]]; `layout.pairing` is ignored
- `'large_scale'` (TR 38.901 Table 7.8-1): the link gain `−PL − O2I + 10 lg(SF)` plus the gain of one unit-gain LOS path from [[get_channels_planar]] (port weights, element positions, orientations); no rays
- `'rsrp'` (TR 36.873 eq. (8.1-1)): per Rx port the sum of `|h|²` over the rays of [[gbsm_single_link_channel]] (`output = 'rays'`, mobility `'phase'`), link gain included, transmit power 1
- Both rules take `|h|²` of Tx port `tx_port` and average it over the Rx ports, so a dual-polarized UT (such as the generic antenna) reads 3 dB below one co-polarized element; a calibration sets the UT antenna of the TR
- A terminal takes the highest value of its nodes; the serving node has the highest value, a tie the lowest index. A Tx node without a propagation link (co-located) has −Inf; a terminal without any link gets node 0
- `layout.pairing = [ site.'; 1:n_rx_terminal ]` links the sectors of the serving sites (1-based)

## Usage:
```
[ node, site, gain_dB ] = quadriga_lib.attach_terminals( layout, rule, rays, i_freq, tx_port );
```

## Inputs:
- **`layout`** — Layout struct with links from [[init_link_states]], Tx nodes required if there are terminals; fields in [[layout_csv_read]]
- **`rule`** — `'large_scale'` or `'rsrp'`; default: `'large_scale'`
- **`rays`** — `'rsrp'`: rays of all links from [[generate_rays]], possibly edited, required; cell array `[n_link, 1]` of `1xN` struct arrays, `N = 1` or `n_freq`; default: `{}`
- **`i_freq`** — Frequency index into `layout.center_freq`, 1-based; default: 1
- **`tx_port`** — Port of the Tx antennas, 1-based; default: 1

## Outputs:
- **`node`** — Serving Tx node per terminal, 1-based; uint64; `[n_rx_terminal, 1]`
- **`site`** — Serving site per terminal, 1-based; uint64; `[n_rx_terminal, 1]`
- **`gain_dB`** — Coupling gain or RSRP in [dB]; `[n_rx_terminal, n_tx_node]`

## See also:
- [[init_link_states]], [[generate_rays]] (link states and rays)
- [[get_gbsm_channels]] (channels of the linked pairs)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], false, "layout");
    const std::string rule = (nrhs < 2) ? std::string("large_scale") : qd_mex_get_string(prhs[1], "large_scale", "rule");
    const auto rays = (nrhs < 3 || mxIsEmpty(prhs[2])) ? std::vector<std::vector<quadriga_lib::gbsm_rays>>() : qd_mex_cell2rays(prhs[2], "rays");
    const arma::uword i_freq = (nrhs < 4) ? 0 : qd_mex_get_index(prhs[3], "i_freq");     // 1-based -> 0-based
    const arma::uword tx_port = (nrhs < 5) ? 0 : qd_mex_get_index(prhs[4], "tx_port");   // 1-based -> 0-based

    arma::uvec node, site;
    arma::mat gain_dB;
    arma::uvec *p_site = (nlhs > 1) ? &site : nullptr;
    arma::mat *p_gain_dB = (nlhs > 2) ? &gain_dB : nullptr;
    CALL_QD(node = quadriga_lib::attach_terminals(layout, rule, rays, p_site, p_gain_dB, i_freq, tx_port));

    node += 1; // 0-based -> 1-based
    plhs[0] = qd_mex_copy2matlab(&node);
    if (nlhs > 1)
    {
        site += 1;
        plhs[1] = qd_mex_copy2matlab(&site);
    }
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&gain_dB);
}
