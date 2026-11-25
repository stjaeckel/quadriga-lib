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
# DROP_TERMINALS
Drop UTs into the cells of the Tx side of a layout

- Cells: the Voronoi regions of the sites (the Tx nodes of one name, at the position of their first row), with the wrap images when `layout.wrap` is set. A site with several sectors is split into wedges by the nearest sector heading (±60° around each boresight for three sectors at equal spacing)
- Drop area: with two wrap vectors, the wrapped cluster, i.e. the points nearest to image 0 of a site among the images 0, ±a1, ±a2 and ±(a1 − a2); else `bounds`, which is then required. Given bounds limit the drop in both cases
- The drop of [[generate_layout]] applies in the cells of each site, with the rules of the scenario family in its `scenario` (minimum distance, heights and floors, speeds, O2I states). Sites without a family (no scenario, a conf name): height 1.5 m, no minimum distance, speed 0, no O2I state; their UTs get `indoor` -1, `floor` -1 and `d2d_in` NaN when other UTs have states
- Every site is a cell, also the micro sites of `M.2412_DenseUrban` config 3 (whose own drop counts them with their macro TRxP). A site closer than twice the minimum distance to another one can leave a cell too little room, e.g. micro sites with the scenario `UMa` or `UMi` set; such a cell raises an error
- The Rx side is replaced by the UTs `UT1`, `UT2`, ... (ordered by cell with `per_cell`) with the generic antenna; `pairing` and `links` are emptied. The Tx side, the antenna lists and `layout.seed` stay. A failed call leaves the layout unchanged
- Random streams: the positions use the key `drop`, the heading and direction of a UT `ut|<name>`, its floor `floor|<family>|<name>`, the O2I fields `<purpose>|<family>`; the base of all is the seed of the call
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## Usage:
```
layout_out = quadriga_lib.drop_terminals( layout_in, n_ut, bounds, ut, indoor, options, per_cell, seed );
```

## Inputs:
- **`layout_in`** — Valid layout struct, fields in [[layout_csv_read]]; the sectors of one site need different headings
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 10<sup>7</sup> in total; 0 = no UTs
- **`bounds`** — `[x_min, x_max, y_min, y_max]` in [m], finite, `x_min < x_max`, `y_min < y_max`; required unless `layout_in.wrap` has two vectors; default: `[]`
- **`ut`**, **`indoor`**, **`options`**, **`per_cell`** — As in [[generate_layout]]; an entry or option that no family of the sites uses raises an error; defaults: `[]`, `[]`, `[]`, `true`
- **`seed`** — Seed >= 0; -1 = `layout_in.seed` if it is set, else the system random device; default: -1

## Outputs:
- **`layout_out`** — The layout with the Rx side replaced by the dropped UTs

## See also:
- [[generate_layout]] (layouts from presets, with this drop)
- [[generate_track]] (tracks for the UTs)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], true, "layout_in");
    const arma::uword n_ut = qd_mex_get_scalar<arma::uword>(prhs[1], "n_ut", 0);
    const arma::vec bounds = qd_mex_layout_vec(nrhs, prhs, 2, "bounds"), ut = qd_mex_layout_vec(nrhs, prhs, 3, "ut");
    const arma::vec indoor = qd_mex_layout_vec(nrhs, prhs, 4, "indoor");
    const auto options = (nrhs < 6) ? std::unordered_map<std::string, std::vector<double>>() : qd_mex_layout_options(prhs[5], "options");
    const bool per_cell = (nrhs < 7) ? true : qd_mex_get_scalar<bool>(prhs[6], "per_cell", true);
    const arma::sword seed = (nrhs < 8) ? -1 : (arma::sword)qd_mex_get_seed(prhs[7], "seed");

    CALL_QD(quadriga_lib::drop_terminals(layout, n_ut, bounds, ut, indoor, options, per_cell, seed));

    plhs[0] = qd_mex_layout2struct(layout);
}
