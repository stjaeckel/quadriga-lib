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
# GENERATE_RAYS
Rays of the propagation links of a layout (stage 2 of the channel pipeline)

- Runs the chain of the section "Geometry-based stochastic model" for the chosen links, from the link states of
  [[init_link_states]]: large-scale parameters, clusters, rays and, on request, the scatterers
- The rays have no antennas; [[get_gbsm_channels]] applies them. Edit the rays in between to study a change of the
  propagation, or pass them to [[attach_terminals]] and [[export_rays_qrt]]
- The rays of a link do not depend on the other requested links

## Usage:
```
rays = quadriga_lib.generate_rays( layout, i_link, scatterers );
```

## Inputs:
- **`layout`** — Layout struct with links from [[init_link_states]]; fields in [[layout_csv_read]]; raises an error if a
  member that determines the links changed since then, or if the sizes of its members do not match
- **`i_link`** — Propagation links, 1-based, without repetition; `[]` = all (none for links without entries); default:
  `[]`; `[n_out]`
- **`scatterers`** — Solve the first- and last-bounce scatterers, needed for the mobility `'drift'`; default: `false`

## Outputs:
- **`rays`** — Rays per link and frequency, a cell array `[n_out, 1]` of `1xN` struct arrays with the fields of
  [[ssf_rays]], `N = n_freq` with one frequency each for the TR methods, `N = 1` with all frequencies for `'quadriga'`

## Notes:
- **Chain**: [[gbsm_init_phases]] and [[gbsm_draw]] of the random sets, [[path_loss]] (for `rnd_he` and the SF standard
  deviation), [[lsp_generate]], [[ssf_clusters]], [[ssf_rays]] and, with `scatterers`, [[ssf_scatterers]], once for the
  links of one site, image, conf and seed
- **Position**: the first row of the segment of the terminal and the site (moved by its wrap image) at the same time; a
  moving site is sampled at the time of the row, without times on the Rx track at the row of the same index. A site with
  more than one row or a speed uses dual mobility; the other sites are not affected
- **TR methods**: the SOS preset of the links (default `Exp256`), no bias removal, path loss at the geometric distance,
  `angle_model = 'additive'`, `pol_model = '38.901'`, one set of clusters and rays per frequency (`'38.901_MF'`:
  clusters at the anchor frequency, rays per frequency)
- **`'quadriga'`**: bias removal over the positions of all links of the group, the legacy outdoor distance,
  `'rotation'`, `'quadriga'`, one ray set for all frequencies, always with scatterers
- **Edited link states**: the conf and the seed of a link follow from its scenario and the `los` and `indoor` of
  `layout.links`; a changed conf uses its hashed seed, an edited `gbsm_seed` is used while the conf is unchanged.
  Building UTs are on the floor of the Rx nodes (else from z) at the distance of the stored O2I inputs, the others on
  floor 0 at 0 m

## See also:
- [[init_link_states]] (stage 1), [[get_gbsm_channels]] (stage 3)
- [[ssf_rays]] (the ray data)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], false, "layout");
    const arma::uvec i_link = (nrhs < 2) ? arma::uvec() : qd_mex_get_index_Col<arma::uword>(prhs[1], "i_link"); // 1-based -> 0-based
    const bool scatterers = (nrhs < 3) ? false : qd_mex_get_scalar<bool>(prhs[2], "scatterers", false);

    std::vector<std::vector<quadriga_lib::gbsm_rays>> rays;
    CALL_QD(rays = quadriga_lib::generate_rays(layout, i_link, scatterers));

    plhs[0] = qd_mex_rays2cell(rays);
}
