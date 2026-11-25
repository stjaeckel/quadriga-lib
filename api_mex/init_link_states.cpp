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
# INIT_LINK_STATES
Draw the discrete state of the propagation links of a layout (stage 1 of the channel pipeline)

- Draws for each propagation link (one site, one terminal, one segment) the state that the later stages need: the wrap
  image, the scenario and conf file, the LOS and indoor state and the seeds, stored in `layout_out.links` (fields in
  [[layout_csv_read]])
- The states follow from the seeds and the positions, so adding a terminal does not change the states of the others
- `layout.pairing` does not limit the links; it selects at the output of the channel functions

## Usage:
```
layout_out = quadriga_lib.init_link_states( layout_in, method, o2i_low, pol_variability, anchor_freq, ...
    sos_preset, seed, config_dirs );
```

## Inputs:
- **`layout_in`** — Layout struct, valid, with `center_freq`; fields in [[layout_csv_read]]
- **`method`** — `'38.901'`, `'38.901_MF'` (TR 38.901 clause 7.6.5.1) or `'quadriga'`, see [[generate_rays]]; default:
  `'38.901'`
- **`o2i_low`** — Share of the low-loss model among the building UTs of UMa and UMi, in [0, 1]; the other families have
  one building model; `[]` = the share of TR 38.901 Table 7.8-1 (0.5); default: `[]`
- **`pol_variability`** — Polarization power variability of TR 38.901 clause 7.6.16; TR methods only; default: `false`
- **`anchor_freq`** — Anchor frequency of `'38.901_MF'` in [Hz]; default: 2e9
- **`sos_preset`** — SOS preset of the GBSM random sets, see [[sos_preset]]; `''` = `Exp256` (TR methods), `Comb256`
  (`'quadriga'`); default: `''`
- **`seed`** — Master seed, >= -1; -1 = `layout_in.seed`, if that is -1 one draw of the system random device; default:
  -1
- **`config_dirs`** — Folders of user conf files, searched before the shipped folder; string or cell array of strings;
  default: `{}`

## Outputs:
- **`layout_out`** — The layout with `links` and `seed` (the seed used) set; the rest is unchanged

## Notes:
- **Links**: one per site, terminal and segment, for pairs farther apart than 1e-4 m at the first row, terminal-major
  (all sites of terminal 1 first). Co-located pairs have no link; a later segment that starts at the site raises an
  error. A layout without any pair farther apart gives links without entries, and the later stages return empty results
- **Scenario** of a link: the value of the terminal, else that of the site. A scenario family (e.g. `UMa`) draws the LOS
  and indoor state, a conf name of the family table (e.g. `UMa_NLOS_O2I`, also with the prefix `3GPP_38.901_v19_`) fixes
  them, and any other name is a user conf file, used in either state without O2I loss. A conf that several families list
  (`Indoor_LOS`, `InF_LOS`) belongs to the family of the site if that lists it, else to the first. Conf files are found
  with [[conf_file_list]], the folders of `config_dirs` first
- **Wrap image**: the image of the site (`layout.wrap`) with the shortest 2D distance to the first position of the
  terminal, ties to the lowest index in the order 0, +a1, -a1, +a2, -a2, +(a1 - a2), -(a1 - a2); all distances of the
  link use this virtual site position
- **O2I state** per terminal, from spatially correlated fields at its first position (TR 38.901 clause 7.6.3.3):
  building UTs with the share of TR 38.901 Table 7.2-1 (0.8 in UMa and UMi), Table 7.2-3 (0.5 in RMa) and Table 7.8-1A
  (0.8 in SMa), the other UTs outdoors or, in RMa and SMa, in cars; residential or commercial building in SMa (0.9); the
  O2I model and `d2d_in` as in [[o2i_loss]]. The values `indoor`, `floor` and `d2d_in` of the Rx nodes fix them; an
  exact conf name that contradicts a fixed `indoor` raises an error
- **One O2I state per terminal** (TR 38.901: all the links from a UT have the same indoor/outdoor state): a terminal
  linked to sites of several families has the state of the first family with O2I in site order, and an exact name on a
  site fixes it for the links to all sites; InH and InF have none
- **LOS state** per link: a uniform field per site, image and family at the first position of the segment, at the
  `LOS_state_lambda` of the NLOS conf (InF with `d_clutter`: `d_clutter / 2`), below [[los_probability]] at
  `d2d_out = max(d2d - d2d_in, 0)`; InF-HH is always LOS
- **Segments**: on the Rx nodes, a row whose `scenario` or `indoor` differs from the row before it starts a segment (an
  empty `scenario` differs from an explicit one). A segment has its own propagation links (order: terminal, site,
  segment), LOS state, indoor state, conf and seed, drawn at its first row, with the distances to a moving site at the
  time of the segment start. The wrap image and the O2I random values (`o2i_rnd`, `o2i_model`, a drawn indoor state) are
  those of the first row of the track; a fixed `d2d_in` must be equal at all segment starts. A segment keeps its LOS
  state along its length, so long tracks need explicit segments
- **Random values**: [[sos_rand]] with the preset `Exp256`; each field has its own seed, hashed from the site seed (else
  the layout seed) and a key of its purpose, site, image and family or conf. `gbsm_seed` is the seed of
  [[gbsm_init_phases]]
- **Frequencies**: the TR methods draw the cluster and ray sets of frequency `k` with the index `k` into
  `layout.center_freq`, so a frequency inserted below another changes the rays of the higher bands
- **Layout hash**: `links.layout_hash` covers the positions, names, sectors, frequencies, site seeds, scenarios, the O2I
  columns, clutter, scenario overrides, wrap vectors and whether a site moves. The later stages raise an error when it
  no longer matches the layout; `layout.links = []` discards the links. Orientations, antennas and `pairing` may change
  without new links
- **Forced states**: editing `los` or `indoor` of `layout.links` before [[generate_rays]] forces that state; the conf
  and its seed follow from it. `floor_id` and `d2d_in` stay those of the drawn state; the later stages derive the values
  of the forced state from `indoor`
- **Restoring**: the links hold the master seed they were drawn with, so a copy of `layout.links` can be restored after
  another realisation
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged; the
  wrapper adds the folder of the shipped conf files to `config_dirs`

## See also:
- [[generate_rays]], [[get_channels_from_rays]] (stages 2 and 3)
- [[get_channels_layout]] (all stages in one call)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    quadriga_lib::layout layout = qd_mex_struct2layout(prhs[0], true, "layout_in");
    const std::string method = (nrhs < 2) ? std::string("38.901") : qd_mex_get_string(prhs[1], "38.901", "method");
    const arma::vec o2i_low = (nrhs < 3) ? arma::vec() : qd_mex_get_Col<double>(prhs[2], false, "o2i_low");
    const bool pol_variability = (nrhs < 4) ? false : qd_mex_get_scalar<bool>(prhs[3], "pol_variability", false);
    const double anchor_freq = (nrhs < 5) ? 2.0e9 : qd_mex_get_scalar<double>(prhs[4], "anchor_freq", 2.0e9);
    const std::string sos_preset = (nrhs < 6) ? std::string() : qd_mex_get_string(prhs[5], "", "sos_preset");
    const arma::sword seed = (nrhs < 7) ? -1 : (arma::sword)qd_mex_get_seed(prhs[6], "seed");
    const std::vector<std::string> config_dirs = qd_mex_layout_config_dirs((nrhs < 8) ? nullptr : prhs[7]);

    CALL_QD(quadriga_lib::init_link_states(layout, method, o2i_low, pol_variability, anchor_freq, sos_preset, seed, config_dirs));

    plhs[0] = qd_mex_layout2struct(layout);
}
