// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# get_channels_layout
Channel coefficients of a layout in one call

- Computes one channel per Tx node, Rx node and frequency, for the pairs in `layout["pairing"]` (empty: all pairs)
- Runs the stages [[init_link_states]], [[generate_rays]] and [[get_gbsm_channels]] with their default settings; the
  results are those of the stages called one by one
- Without link states in `layout["links"]`, draws them for this call only and leaves the layout unchanged; with link
  states, uses them, also when they were edited

## Usage:
```
channels, index = quadriga_lib.layout.get_channels_layout( layout, *, times, output, method, mobility, overlap, \
    seed, config_dirs, stack )
```

## Inputs:
- **`layout`** — Layout dict with `center_freq`; keys in [[layout.csv_read]]; antennas as evaluated in
  `layout["antenna"]`, nodes without one use the generic antenna (see [[get_gbsm_channels]])
- **`times`** — Snapshot times in [s]; `None` = one snapshot per row; default: None; `(n_snap,)`
- **`output`** — `"taps"` or `"rays"`; default: `"taps"`
- **`method`** — `"38.901"`, `"38.901_MF"` or `"quadriga"`; `""` = the method of filled links, else `"38.901"`; another
  method than that of filled links raises an error; default: `""`
- **`mobility`** — `"phase"`, `"drift"` or `"auto"`; default: `"auto"`
- **`overlap`** — Overlap of consecutive segments, in [0, 1]; default: 0.5
- **`seed`** — Master seed for empty links, as in [[init_link_states]]; with filled links only -1; default: -1
- **`config_dirs`** — Folders of user conf files for empty links, searched before the shipped folder; str or list of
  str; default: None
- **`stack`** — Snapshot layout of the channels as in [[hdf5_read_channel]]; default: `False`

## Outputs:
- **`channels`** — Channels, Tx node fastest, then Rx node, then frequency; list of channel dicts, keys in
  [[hdf5_read_channel]]
- **`index`** — Tx node, Rx node and frequency index of each channel, 0-based; int64; `(3, n_channel)`

## Example:
```
import numpy as np, quadriga_lib as qd
layout = qd.layout.generate_layout( "38.901_UMa", 1 )
layout["center_freq"] = np.array([3.5e9])
channels, index = qd.layout.get_channels_layout( layout )
```

## Notes:
- **Memory**: the links are processed in batches of at most 256 propagation links (the segments of a track in one
  batch), so that the rays of one batch bound the memory; the results do not depend on the batches
- **Link states**: `layout["links"]` holds none when `layout["links"]["method"]` is empty, e.g. after
  `layout["links"] = {}`. With link states, `seed` and `config_dirs` are not used. A layout without linked pairs gives
  an empty result
- **Scatterers**: the mobility `"drift"` (also `"auto"` with `"quadriga"`) solves the scatterers

## See also:
- [[init_link_states]], [[generate_rays]], [[get_gbsm_channels]] (the three stages)
- [[generate_layout]] (layouts of the TR scenarios)
MD!*/
py::tuple get_channels_layout(const py::dict &layout,
                              py::handle times,
                              const std::string &output,
                              const std::string &method,
                              const std::string &mobility,
                              double overlap,
                              long long seed,
                              py::handle config_dirs,
                              py::handle stack)
{
    const auto lay = qd_python_dict2layout(layout, true, "layout");
    const auto times_a = qd_python_numpy2arma_Col<double>(times, true);
    const auto config_dirs_v = qd_pl_config_dirs(config_dirs);
    const bool stack_b = qd_python_flag(stack, false, "stack");

    arma::umat index;
    const auto chan = quadriga_lib::get_channels_layout(lay, times_a, output, method, &index, mobility, overlap,
                                                        (arma::sword)seed, config_dirs_v);
    return py::make_tuple(qd_python_channel2list(chan, false, stack_b), qd_python_copy2numpy<arma::uword, py::ssize_t>(&index));
}

/*!MD
# init_link_states
Draw the discrete state of the propagation links of a layout (stage 1 of the channel pipeline)

- Draws for each propagation link (one site, one terminal, one segment) the state that the later stages need: the wrap
  image, the scenario and conf file, the LOS and indoor state and the seeds, stored in `layout["links"]` (keys in
  [[layout.csv_read]])
- The states follow from the seeds and the positions, so adding a terminal does not change the states of the others
- `layout["pairing"]` does not limit the links; it selects at the output of the channel functions

## Usage:
```
layout_out = quadriga_lib.layout.init_link_states( layout_in, *, method, o2i_low, pol_variability, anchor_freq, \
    sos_preset, seed, config_dirs )
```

## Inputs:
- **`layout_in`** — Layout dict, valid, with `center_freq`; keys in [[layout.csv_read]]
- **`method`** — `"38.901"`, `"38.901_MF"` (TR 38.901 clause 7.6.5.1) or `"quadriga"`, see [[generate_rays]]; default:
  `"38.901"`
- **`o2i_low`** — Share of the low-loss model among the building UTs of UMa and UMi, in [0, 1], `(1,)`; the other
  families have one building model; `None` = the share of TR 38.901 Table 7.8-1 (0.5); default: None
- **`pol_variability`** — Polarization power variability of TR 38.901 clause 7.6.16; TR methods only; default: `False`
- **`anchor_freq`** — Anchor frequency of `"38.901_MF"` in [Hz]; default: 2e9
- **`sos_preset`** — SOS preset of the GBSM random sets, see [[sos.preset]]; `""` = `Exp256` (TR methods), `Comb256`
  (`"quadriga"`); default: `""`
- **`seed`** — Master seed, >= -1; -1 = `layout_in["seed"]`, if that is -1 one draw of the system random device. The
  seed used is written to `layout_out["seed"]`; default: -1
- **`config_dirs`** — Folders of user conf files, searched before the shipped folder; str or list of str; default: None

## Outputs:
- **`layout_out`** — The layout with `links` and `seed` set; the rest is unchanged

## Notes:
- **Links**: one per site, terminal and segment, for pairs farther apart than 1e-4 m at the first row, terminal-major
  (all sites of terminal 0 first). Co-located pairs have no link; a later segment that starts at the site raises an
  error. A layout without any pair farther apart gives links without entries, and the later stages return empty results
- **Scenario** of a link: the value of the terminal, else that of the site. A scenario family (e.g. `UMa`) draws the LOS
  and indoor state, a conf name of the family table (e.g. `UMa_NLOS_O2I`, also with the prefix `3GPP_38.901_v19_`) fixes
  them, and any other name is a user conf file, used in either state without O2I loss. A conf that several families list
  (`Indoor_LOS`, `InF_LOS`) belongs to the family of the site if that lists it, else to the first. Conf files are found
  with [[conf_file_list]], the folders of `config_dirs` first
- **Wrap image**: the image of the site (`layout["wrap"]`) with the shortest 2D distance to the first position of the
  terminal, ties to the lowest index in the order 0, $+a_1$, $-a_1$, $+a_2$, $-a_2$, $+(a_1 - a_2)$, $-(a_1 - a_2)$; all
  distances of the link use this virtual site position
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
  $d_\mathrm{2D,out} = \max(d_\mathrm{2D} - d_\mathrm{2D,in}, 0)$; InF-HH is always LOS
- **Segments**: on the Rx nodes, a row whose `scenario` or `indoor` differs from the row before it starts a segment (an
  empty `scenario` differs from an explicit one). A segment has its own propagation links (order: terminal, site,
  segment), LOS state, indoor state, conf and seed, drawn at its first row, with the distances to a moving site at the
  time of the segment start. The wrap image and the O2I random values (`o2i_rnd`, `o2i_model`, a drawn indoor state) are
  those of the first row of the track; a fixed `d2d_in` must be equal at all segment starts. A segment keeps its LOS
  state along its length, so long tracks need explicit segments
- **Random values**: [[sos.rand]] with the preset `Exp256`; each field has its own seed, hashed from the site seed (else
  the layout seed) and a key of its purpose, site, image and family or conf. `gbsm_seed` is the seed of
  [[gbsm.init_phases]]
- **Frequencies**: the TR methods draw the cluster and ray sets of frequency `k` with the index `k` into
  `layout["center_freq"]`, so a frequency inserted below another changes the rays of the higher bands
- **Layout hash**: covers the positions, names, sectors, frequencies, site seeds, scenarios, the O2I columns, clutter,
  scenario overrides, wrap vectors and whether a site moves. The later stages raise an error when it no longer matches
  the layout; `layout["links"] = {}` discards the links. Orientations, antennas and `pairing` may change without new
  links
- **Forced states**: editing `los` or `indoor` of `layout["links"]` before [[generate_rays]] forces that state; the conf
  and its seed follow from it. `floor_id` and `d2d_in` stay those of the drawn state; the later stages derive the values
  of the forced state from `indoor`
- **Restoring**: the links hold the master seed they were drawn with, so a copy of `layout["links"]` can be restored
  after another realisation
- **`layout_in` / `layout_out`**: the C++ in/out argument `layout` is split, so that the input stays unchanged

## See also:
- [[generate_rays]], [[get_gbsm_channels]] (stages 2 and 3)
- [[get_channels_layout]] (all stages in one call)
- [[layout.csv_read]] (the keys of `layout["links"]`)
MD!*/
py::dict init_link_states(const py::dict &layout_in,
                          const std::string &method,
                          py::handle o2i_low,
                          py::handle pol_variability,
                          double anchor_freq,
                          const std::string &sos_preset,
                          long long seed,
                          py::handle config_dirs)
{
    auto lay = qd_python_dict2layout(layout_in, false, "layout_in");
    const auto o2i_low_a = qd_python_numpy2arma_Col<double>(o2i_low, true);
    const auto config_dirs_v = qd_pl_config_dirs(config_dirs);

    quadriga_lib::init_link_states(lay, method, o2i_low_a, qd_python_flag(pol_variability, false, "pol_variability"),
                                   anchor_freq, sos_preset, (arma::sword)seed, config_dirs_v);
    return qd_python_layout2dict(lay);
}

/*!MD
# generate_rays
Rays of the propagation links of a layout (stage 2 of the channel pipeline)

- Runs the chain of the section "Geometry-based stochastic model" for the chosen links, from the link states of
  [[init_link_states]]: large-scale parameters, clusters, rays and, on request, the scatterers
- The rays have no antennas; [[get_gbsm_channels]] applies them. Edit the rays in between to study a change of the
  propagation, or pass them to [[attach_terminals]] and [[export_rays_qrt]]
- The rays of a link do not depend on the other requested links

## Usage:
```
rays = quadriga_lib.layout.generate_rays( layout, *, i_link, scatterers )
```

## Inputs:
- **`layout`** — Layout dict with links from [[init_link_states]]; keys in [[layout.csv_read]]; raises an error if a
  member that determines the links changed since then, or if the sizes of its members do not match
- **`i_link`** — Propagation links, 0-based, without repetition; `None` = all (none for links without entries); default:
  None; `(n_out,)`
- **`scatterers`** — Solve the first- and last-bounce scatterers, needed for the mobility `"drift"`; default: `False`

## Outputs:
- **`rays`** — Rays per link and frequency, a list of `n_out` lists of `N` ray dicts with the keys of [[ssf_rays]],
  `N = n_freq` with one frequency each for the TR methods, `N = 1` with all frequencies for `"quadriga"`

## Notes:
- **Chain**: [[gbsm.init_phases]] and [[gbsm.draw]] of the random sets, [[path_loss]] (for `rnd_he` and the SF standard
  deviation), [[lsp_generate]], [[ssf_clusters]], [[ssf_rays]] and, with `scatterers`, [[ssf_scatterers]], once for the
  links of one site, image, conf and seed
- **Position**: the first row of the segment of the terminal and the site (moved by its wrap image) at the same time; a
  moving site is sampled at the time of the row, without times on the Rx track at the row of the same index. A site with
  more than one row or a speed uses dual mobility; the other sites are not affected
- **TR methods**: the SOS preset of the links (default `Exp256`), no bias removal, path loss at the geometric distance,
  `angle_model = "additive"`, `pol_model = "38.901"`, one set of clusters and rays per frequency (`"38.901_MF"`:
  clusters at the anchor frequency, rays per frequency)
- **`"quadriga"`**: bias removal over the positions of all links of the group, the legacy outdoor distance,
  `"rotation"`, `"quadriga"`, one ray set for all frequencies, always with scatterers
- **Edited link states**: the conf and the seed of a link follow from its scenario and the `los` and `indoor` of
  `layout["links"]`; a changed conf uses its hashed seed, an edited `gbsm_seed` is used while the conf is unchanged.
  Building UTs are on the floor of the Rx nodes (else from z) at the distance of the stored O2I inputs, the others on
  floor 0 at 0 m

## See also:
- [[init_link_states]] (stage 1), [[get_gbsm_channels]] (stage 3)
- [[ssf_rays]] (the ray data)
MD!*/
py::list generate_rays(const py::dict &layout,
                       py::handle i_link,
                       py::handle scatterers)
{
    const auto lay = qd_python_dict2layout(layout, true, "layout");
    const auto i_link_a = qd_python_numpy2arma_Col<arma::uword>(i_link, true, false, "i_link");

    const auto rays = quadriga_lib::generate_rays(lay, i_link_a, qd_python_flag(scatterers, false, "scatterers"));
    return qd_pl_rays2list(rays);
}

/*!MD
# get_gbsm_channels
Channel coefficients of a layout from given rays (stage 3 of the channel pipeline)

- Applies the antennas of the nodes, the motion along their tracks and the link gain to the rays of [[generate_rays]],
  and returns one channel per Tx node, Rx node and frequency
- Only the channel links of the given propagation links and of the pairs in `layout["pairing"]` (empty: all) are
  computed; the sectors of a site share the rays of their propagation link
- Tracks with several segments are joined by [[crossfade_segments]]

## Usage:
```
channels, index = quadriga_lib.layout.get_gbsm_channels( layout, rays, *, i_link, times, output, mobility, oxygen, \
    use_absolute_delays, overlap, stack )
```

## Inputs:
- **`layout`** — Layout dict with links from [[init_link_states]]; keys in [[layout.csv_read]]; raises an error if a
  member that determines the links changed since then
- **`rays`** — Rays of the links from [[generate_rays]], possibly edited, with the frequencies of the layout; list of
  `n_out` lists of `N` ray dicts (a ray dict alone for a link with one), `N = 1` or `n_freq`
- **`i_link`** — The propagation links of `rays`, 0-based; `None` = all links; default: None; `(n_out,)`
- **`times`** — Snapshot times in [s], finite, the same for all nodes; `None` = one snapshot per row; default: None;
  `(n_snap,)`
- **`output`** — `"taps"` or `"rays"`, see [[gbsm.single_link_channel]]; default: `"taps"`
- **`mobility`** — `"phase"`, `"drift"` (needs rays with scatterers) or `"auto"` (phase for the TR methods, drift for
  `"quadriga"`); default: `"auto"`
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `False`
- **`use_absolute_delays`** — Include the LOS delay in all delays; default: `False`
- **`overlap`** — Part of the snapshots shared by consecutive segments that the cross-fade uses, in [0, 1], see
  [[crossfade_segments]]; default: 0.5
- **`stack`** — Snapshot layout of the channels as in [[hdf5_read_channel]]; default: `False`

## Outputs:
- **`channels`** — Channels; list of channel dicts, keys in [[hdf5_read_channel]]
- **`index`** — Tx node, Rx node and frequency index of each channel, 0-based; int64; `(3, n_channel)`

## Notes:
- **Order and names**: Tx node fastest, then Rx node, then frequency; name `<tx>:<sector>_<rx>:<sector>`, without the
  sector where all Tx nodes (or all Rx nodes) have sector 0
- **Snapshots** of both ends from [[sample_track]] over `times`; a node with one snapshot is repeated to the count of
  the other; two tracks without `times` need equal row counts
- **Link gain** per snapshot, $G = -\mathrm{PL} - L_\mathrm{O2I} + 10 \log_{10} \mathrm{SF}$ in dB: the path loss PL of
  [[path_loss]] with the random environment height `rnd_he` held at the start of the segment, the shadow fading SF and
  the K-factor ratio of [[lsp_generate]] from [[gbsm.draw]] of the `"lsf"` set at the snapshot, and the O2I loss of
  [[o2i_loss]] for building and car UTs with the model of the link state (the deviation of a building UT at the
  snapshot, that of a car UT at the first row of the terminal for its whole track, TR 38.901 eq. 7.4-4). A link whose
  ends do not move is evaluated once
- **Antennas**: [[gbsm.single_link_channel]] with the node antennas at the frequency (entry `i_freq` of the list of
  bands of the antenna in `layout["antenna"]`, or entry 0 for all bands) and the Tx positions moved by the wrap image of
  the link. Nodes without an antenna and antennas that are not evaluated use the generic antenna; an antenna spaced in
  wavelengths that is not evaluated raises an error
- **Frequencies**: rays with `n_freq` entries are used as entry `i_freq` (TR methods), rays with one entry at frequency
  `i_freq` (`"quadriga"`)
- **Segments**: the snapshots of a segment are those whose row (see [[sample_track]]) is in it; each segment also covers
  the snapshots of the preceding one, with its link gain, rays and K-factor ratio referred to its first row. The
  selected segments of a channel link must be consecutive (segments without snapshots are skipped), with increasing
  `times`
- **Relative delays**: in `"phase"` the paths of each segment relative to its own LOS, in `"drift"` relative to the LOS
  at the snapshot

## See also:
- [[generate_rays]] (stage 2)
- [[get_channels_layout]] (all stages in one call)
MD!*/
py::tuple get_gbsm_channels(const py::dict &layout,
                            const py::handle &rays,
                            py::handle i_link,
                            py::handle times,
                            const std::string &output,
                            const std::string &mobility,
                            py::handle oxygen,
                            py::handle use_absolute_delays,
                            double overlap,
                            py::handle stack)
{
    const auto lay = qd_python_dict2layout(layout, true, "layout");
    const auto rays_v = qd_pl_list2rays(rays, "rays");
    const auto i_link_a = qd_python_numpy2arma_Col<arma::uword>(i_link, true, false, "i_link");
    const auto times_a = qd_python_numpy2arma_Col<double>(times, true);
    const bool oxygen_b = qd_python_flag(oxygen, false, "oxygen");
    const bool use_absolute_delays_b = qd_python_flag(use_absolute_delays, false, "use_absolute_delays");
    const bool stack_b = qd_python_flag(stack, false, "stack");

    arma::umat index;
    const auto chan = quadriga_lib::get_gbsm_channels(lay, rays_v, i_link_a, times_a, output, &index, mobility, oxygen_b,
                                                      use_absolute_delays_b, overlap);
    return py::make_tuple(qd_python_channel2list(chan, false, stack_b), qd_python_copy2numpy<arma::uword, py::ssize_t>(&index));
}

/*!MD
# attach_terminals
Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP

- Evaluates every Tx node for each terminal and returns the one with the highest coupling gain (`"large_scale"`) or the
  highest RSRP (`"rsrp"`), with its site and the gains of all nodes
- Use it to link each terminal to its serving site in `layout["pairing"]` (see Notes), so that the channel functions
  compute only the serving links
- One call attaches at one frequency, `layout["center_freq"][i_freq]`, the first band by default; this suits offloading
  scenarios, where the higher bands serve as data pipes. The result is one serving Tx node per terminal

## Usage:
```
node, site, gain_dB = quadriga_lib.layout.attach_terminals( layout, *, rule, rays, i_freq, tx_port )
```

## Inputs:
- **`layout`** — Layout dict with links from [[init_link_states]], Tx nodes required if there are terminals; keys in
  [[layout.csv_read]]
- **`rule`** — `"large_scale"` or `"rsrp"`; default: `"large_scale"`
- **`rays`** — `"rsrp"`: rays of all links from [[generate_rays]], possibly edited, required; list of `n_link` lists of
  `N` ray dicts (a ray dict alone for a link with one), `N = 1` or `n_freq`; default: None
- **`i_freq`** — Frequency index into `layout["center_freq"]`, 0-based; default: 0
- **`tx_port`** — Port of the Tx antennas, 0-based; default: 0

## Outputs:
- **`node`** — Serving Tx node per terminal, 0-based; int64; `(n_rx_terminal,)`
- **`site`** — Serving site per terminal, 0-based; int64; `(n_rx_terminal,)`
- **`gain_dB`** — Coupling gain or RSRP in [dB]; `(n_rx_terminal, n_tx_node)`

## Notes:
- **Evaluation**: at the first rows of both nodes and `layout["center_freq"][i_freq]`, with the link states of the first
  segment, the wrap image and the antennas as in [[get_gbsm_channels]]; `layout["pairing"]` is ignored
- **`"large_scale"`** (TR 38.901 Table 7.8-1): the link gain of [[get_gbsm_channels]] plus the gain of one unit-gain LOS
  path through both antennas (port weights, element positions, orientations), as [[channel.planar]] gives it; no rays
- **`"rsrp"`** (TR 36.873 eq. (8.1-1)): per Rx port the sum of $\lvert h \rvert^2$ over the rays of
  [[gbsm.single_link_channel]] (`output = "rays"`, mobility `"phase"`), link gain included, transmit power 1
- **Ports**: both rules take $\lvert h \rvert^2$ of Tx port `tx_port` and average it over the Rx ports, so a
  dual-polarized UT (such as the generic antenna) reads 3 dB below one co-polarized element; a calibration sets the UT
  antenna of the TR
- **Choice**: a terminal takes the highest value of its nodes; ties go to the lowest index. A Tx node without a
  propagation link (co-located) has -Inf; a terminal without any link gets node 0
- **Pairing**: `layout["pairing"]` holds pairs of site and terminal, so
  `layout["pairing"] = np.vstack((site, np.arange(len(site))))` links all sectors of the serving site; for
  multi-connectivity, `layout["pairing"]` may list several sites per terminal, e.g. all sites within a margin of the
  best value in `gain_dB`, and the channel functions then compute the links to all of them

## See also:
- [[init_link_states]], [[generate_rays]] (link states and rays)
- [[get_gbsm_channels]] (channels of the linked pairs)
MD!*/
py::tuple attach_terminals(const py::dict &layout,
                           const std::string &rule,
                           py::handle rays,
                           arma::uword i_freq,
                           arma::uword tx_port)
{
    const auto lay = qd_python_dict2layout(layout, true, "layout");
    const auto rays_v = qd_pl_list2rays(rays, "rays");

    arma::uvec site;
    arma::mat gain_dB;
    const arma::uvec node = quadriga_lib::attach_terminals(lay, rule, rays_v, &site, &gain_dB, i_freq, tx_port);
    return py::make_tuple(qd_python_copy2numpy<arma::uword, py::ssize_t>(&node),
                          qd_python_copy2numpy<arma::uword, py::ssize_t>(&site), qd_python_copy2numpy(&gain_dB));
}

/*!MD
# export_rays_qrt
Write the rays of a layout with their link gain to a QRT file

- Stores the rays of the linked pairs with their link gain as CIRs: one origin per site, one destination per terminal
  and one CIR per row of the terminal
- Separates the ray generation from the antennas: [[qrt_file_read]], [[gbsm.rays_from_paths]] and
  [[gbsm.single_link_channel]] turn a CIR into channels for any arrays, orientations and positions, without the GBSM
  chain
- Positions in the file are local; add the returned shift to the positions read back

## Usage:
```
shift = quadriga_lib.layout.export_rays_qrt( fn, layout, *, rays )
```

## Inputs:
- **`fn`** — QRT file, str or path-like, replaced if it exists
- **`layout`** — Layout dict with links from [[init_link_states]]: one frequency for the TR methods (`"quadriga"`: 1 to
  127), Tx nodes with one row and no speed, no wrap-around, each terminal paired with a site and no row of a terminal at
  a site position; keys in [[layout.csv_read]]
- **`rays`** — Rays of all links from [[generate_rays]], possibly edited, one object with all frequencies per link;
  `None` = generated for batches of sites with up to 256 links, which bounds the memory; list of `n_link` lists of one
  ray dict (or the ray dicts alone); default: None

## Outputs:
- **`shift`** — Shift of the local coordinates in x and y in [m]; `(2,)`

## Notes:
- **Origins and destinations**: an origin at each site, with the orientation of sector 0 (else of the first node of the
  site) and the site name; a destination per terminal with its name and one CIR per row, with the orientation of the
  first node of the terminal (its nodes share their positions)
- **Content of a CIR**: the rays of the link and of the segment that holds the row (scatterers added where missing), the
  link gain and the K-factor ratio of [[get_gbsm_channels]] at the row, [[ssf_paths]] with `"drift"` and the C++
  function `qrt_paths_from_scatterers`. A CIR read back gives the channel of [[get_gbsm_channels]] (`output = "rays"`,
  `mobility = "drift"`) at the row within float32 accuracy
- **Reading back**: [[gbsm.rays_from_paths]] with the method of the links and the `n_interact` of the file as
  `path_type`, then [[gbsm.single_link_channel]] with link gain 0 and mobility `"drift"`
- **Point drops with speed**: one CIR; the motion is applied at the evaluation, at the link gain of the drop
- **Pairs**: only the pairs of `layout["pairing"]` (empty: all) are written; the other CIRs of an origin hold no paths
- **Shift**: the centroid of the site and CIR positions in x and y is subtracted; z is unchanged
- **Limits of the format**: no tap index and no ground permittivity, so a CIR gives rays, not the taps of the TR, and
  moving a ground-reflection ray from a file raises an error
- The file is written to `fn + ".part"` and renamed at the end; a failed call leaves an existing file unchanged

## See also:
- [[generate_rays]], [[get_gbsm_channels]] (the rays and the link gain)
- [[qrt_file_read]], [[gbsm.rays_from_paths]], [[gbsm.single_link_channel]] (evaluation of the file)
MD!*/
py::array_t<double> export_rays_qrt(const py::handle &fn,
                                    const py::dict &layout,
                                    py::handle rays)
{
    const std::string fn_s = qd_python_path(fn, "fn");
    const auto lay = qd_python_dict2layout(layout, true, "layout");
    const auto rays_v = qd_pl_list2rays(rays, "rays");

    const arma::vec shift = quadriga_lib::export_rays_qrt(fn_s, lay, rays_v);
    return qd_python_copy2numpy(&shift);
}
