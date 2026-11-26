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
# EXPORT_RAYS_QRT
Write the rays of a layout with their link gain to a QRT file

- Stores the rays of the linked pairs with their link gain as CIRs: one origin per site, one destination per terminal
  and one CIR per row of the terminal
- Separates the ray generation from the antennas: [[qrt_file_read]], [[gbsm_rays_from_paths]] and
  [[gbsm_single_link_channel]] turn a CIR into channels for any arrays, orientations and positions, without the GBSM
  chain
- Positions in the file are local (see [QRT file format: precision](formats.html#qrt-precision)); add the returned shift
  to the positions read back

## Usage:
```
shift = quadriga_lib.export_rays_qrt( fn, layout, rays );
```

## Inputs:
- **`fn`** — QRT file, replaced if it exists
- **`layout`** — Layout struct with links from [[init_link_states]]: one frequency for the TR methods (`'quadriga'`: 1
  to 127), Tx nodes with one row and no speed, no wrap-around, each terminal paired with a site and no row of a terminal
  at a site position; fields in [[layout_csv_read]]
- **`rays`** — Rays of all links from [[generate_rays]], possibly edited, one object with all frequencies per link; `{}`
  = generated for batches of sites with up to 256 links, which bounds the memory; cell array `[n_link, 1]` of 1x1
  structs; default: `{}`

## Outputs:
- **`shift`** — Shift of the local coordinates in x and y in [m]; `[2, 1]`

## Notes:
- **Origins and destinations**: an origin at each site, with the orientation of sector 0 (else of the first node of the
  site) and the site name; a destination per terminal with its name and one CIR per row, with the orientation of the
  first node of the terminal (its nodes share their positions)
- **Content of a CIR**: the rays of the link and of the segment that holds the row (scatterers added where missing), the
  link gain and the K-factor ratio of [[get_channels_from_rays]] at the row, [[ssf_paths]] with `'drift'`. A CIR read
  back gives the channel of [[get_channels_from_rays]] (`output = 'rays'`, `mobility = 'drift'`) at the row within
  float32 accuracy
- **Reading back**: [[gbsm_rays_from_paths]] with the method of the links and the `n_interact` of the file as
  `path_type`, then [[gbsm_single_link_channel]] with link gain 0 and mobility `'drift'`
- **Point drops with speed**: one CIR; the motion is applied at the evaluation, at the link gain of the drop
- **Pairs**: only the pairs of `layout.pairing` (empty: all) are written; the other CIRs of an origin hold no paths
- **Shift**: the centroid of the site and CIR positions in x and y is subtracted; z is unchanged
- **Limits of the [QRT file format](formats.html#qrt)**: no tap index and no ground permittivity, so a CIR gives rays,
  not the taps of the TR, and moving a ground-reflection ray from a file raises an error
- The file is written to `fn + '.part'` and renamed at the end; a failed call leaves an existing file unchanged

## See also:
- [[generate_rays]], [[get_channels_from_rays]] (the rays and the link gain)
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
