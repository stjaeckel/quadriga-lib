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
# EXPORT_RAYS_QRT
Writes the rays of a layout with their link gain to a QRT file

- Separates the ray generation from the antennas: [[qrt_file_read]], [[gbsm_rays_from_paths]] (with the method of the links and the `n_interact` of the file as `path_type`) and [[gbsm_single_link_channel]] (link gain 0, mobility `'drift'`) turn a CIR into channels for any arrays, orientations and positions
- One origin per site at its position, with the orientation of sector 0 (else of the first node of the site) and the site name; one destination per terminal with its name and one CIR per row (the nodes of a terminal share their positions; the CIR has the orientation of the first node)
- Per CIR: the rays of the link and of the segment that holds the row (scatterers added where missing), the link gain and the K-factor ratio of [[get_gbsm_channels]] at the row, [[ssf_paths]] with `'drift'`. A CIR read back gives the channel of [[get_gbsm_channels]] (`output = 'rays'`, `mobility = 'drift'`) at the row within float32 accuracy
- A point drop with speed has one CIR; its motion is applied at the evaluation, at the link gain of the drop
- Only the pairs of `layout.pairing` (empty: all) are written; the other CIRs of an origin hold no paths
- Positions in the file are local: the returned shift, the centroid of the site and CIR positions in x and y, is subtracted (z unchanged); add it to the positions read back
- The QRT format stores no tap index and no ground permittivity: a CIR gives rays, not the taps of the TR, and moving a ground-reflection ray from a file raises an error
- The file is written to `fn + '.part'` and renamed at the end; a failed call leaves an existing file unchanged

## Usage:
```
shift = quadriga_lib.export_rays_qrt( fn, layout, rays );
```

## Inputs:
- **`fn`** — QRT file, replaced if it exists
- **`layout`** — Layout struct with links from [[init_link_states]]: one frequency for the TR methods (`'quadriga'`: 1 to 127), Tx nodes with one row and no speed, no wrap-around, each terminal paired with a site and no row of a terminal at a site position; fields in [[layout_csv_read]]
- **`rays`** — Rays of all links from [[generate_rays]], possibly edited, one object with all frequencies per link; `{}` = generated for batches of sites with up to 256 links, which bounds the memory; cell array `[n_link, 1]` of 1x1 structs; default: `{}`

## Outputs:
- **`shift`** — Shift of the local coordinates in x and y in [m]; `[2, 1]`

## See also:
- [[generate_rays]], [[get_gbsm_channels]] (the rays and the link gain)
- [[qrt_file_read]], [[gbsm_rays_from_paths]], [[gbsm_single_link_channel]] (evaluation of the file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const quadriga_lib::layout layout = qd_mex_struct2layout(prhs[1], false, "layout");
    const auto rays = (nrhs < 3 || mxIsEmpty(prhs[2])) ? std::vector<std::vector<quadriga_lib::gbsm_rays>>() : qd_mex_cell2rays(prhs[2], "rays");

    arma::vec shift;
    CALL_QD(shift = quadriga_lib::export_rays_qrt(fn, layout, rays));

    plhs[0] = qd_mex_copy2matlab(&shift);
}
