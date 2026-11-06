// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!SECTION_DESC
A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the *scenario*
(e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The functions of this
section implement 3GPP TR 38.901 clause 7.5 steps 2 to 10 and the sub-cluster split of step 11 (LOS probability, path
loss, O2I penetration loss, large-scale parameters (LSPs), clusters and rays, without antennas) with the spatial
consistency of clause 7.6.3, oxygen absorption (clause 7.6.1) and polarization power variability (clause 7.6.16);
[[gbsm.single_link_channel]] computes the coefficients of step 11 with the antennas. They are in the submodule
`quadriga_lib.gbsm`.<br><br>

**Scenario parameters.** A scenario is a dict of numeric keys (`scenpar`), a path-loss dict (`plpar`: the entry `model`
and the model parameters) and optional short names, read from a conf file by [[conf_file_read]]. 23 files for TR 38.901
v19.5.0 come with the library, one per scenario and LOS and indoor state (e.g. `3GPP_38.901_v19_UMa_LOS`, `_NLOS`,
`_LOS_O2I`, `_NLOS_O2I`; InH is `Indoor`, InF NLOS has one file per clutter type), in `data/config` of the source tree
and in `share/quadriga-lib/config` of an installed package (for pip, below `sys.prefix`); [[conf_file_list]] finds
them. Their comments give the meaning and unit of each key; [[conf_file_write]] with `write_defaults = True` writes the
complete default set with these comments. Every function that takes `scenpar` accepts the dict of [[conf_file_read]] or
[[scenpar_validate]] or any subset of its keys (`None` = all defaults; a key `SubpathMethod` of QuaDRiGa's structs must
be `"legacy"`), completes a copy with the defaults of missing keys and validates it with [[scenpar_validate]], whose
messages pass through, also for keys the function does not use. Invalid input raises `ValueError`, a failure to read or
write a file `RuntimeError`.<br><br>

**Workflow** for one drop of a set of links:

1. [[conf_file_read]]: the scenarios, one per LOS and indoor state
2. Per link, by the caller with own SOS generators ([[sos.init_phases]], [[sos.rand]], [[sos.randn]]): the indoor
   state (uniform value at `IO_state_lambda` of the O2I files against an indoor ratio of the caller's choice, e.g. TR
   38.901 Table 7.2-1); for indoor UTs [[o2i_loss]] (loss `O2I`, indoor distance `d2d_in`); [[los_probability]] at
   `d2D - d2d_in` and the LOS state (uniform value at `LOS_state_lambda` of the NLOS file against `p_los`). The caller
   chooses the `los_probability` model and the `o2i_loss` models and weights (TR 38.901 Tables 7.4.2-1, 7.4.3-2). The
   states assign each link to a scenario and give the `o2i` flags of [[ssf_clusters]]; steps 3 to 6 run per scenario
3. Per BS site and scenario, once: [[gbsm.init_phases]] of the random sets in the table below; `"lsf"` with the link
   positions for the bias removal
4. [[gbsm.draw]] of each set at the links; [[path_loss]] (`PL`, `sf_sigma`; column 10 of the `"lsf"` set as
   `rnd_he`); [[lsp_generate]] with `sf_sigma`, with the outputs that step 5 takes
5. [[ssf_clusters]] (`n_cluster` is the caller's choice) and [[ssf_rays]] (one ray dict per link), called as in the
   table below
6. Per link and segment: [[ssf_scatterers]] if the mobility is `drift`; then [[gbsm.single_link_channel]] (once per
   ray dict; `"quadriga"`: per `i_freq`), or [[ssf_paths]] for other channel functions. Path data read with
   [[qrt_file_read]] become a ray dict with [[gbsm.rays_from_paths]]; other path sources fill a ray dict directly

| Method | Random sets (`set_id`) | Calls in step 5 |
| --- | --- | --- |
| `"38.901"` | `"lsf"`; `"cluster"` -1 and one per frequency; `"ray"` one per frequency | per frequency `k`: [[ssf_clusters]] and [[ssf_rays]] (`i_freq = 0`) with slice `k` of all per-frequency inputs (`center_freq`, `lsp`, `lsp_mu`, `lsp_sigma`, `rnd_freq`, `rnd_ray`, `eod_offset_deg`, `gr_eps`) |
| `"38.901_MF"` | as `"38.901"` | [[ssf_clusters]] once, with `lsp_anchor`; [[ssf_rays]] per frequency `k` with `i_freq = k` |
| `"quadriga"` | `"lsf"`; `"cluster"` -1; `"ray"` -1 | [[ssf_clusters]] and [[ssf_rays]] once for all frequencies |

For moved positions, evaluate [[gbsm.draw]] (`"lsf"`), [[path_loss]], [[lsp_generate]] and for indoor UTs [[o2i_loss]]
again with the same SOS phases: they give the link gain (see Terms) and `kf_ratio` = KF(position) / KF(drop) from
slice 1 of `lsp`.<br><br>

**Terms.** TR 38.901 names where they exist; QuaDRiGa names are marked:

| Term | Meaning |
| --- | --- |
| scenario | type of environment with its statistics: numeric keys `scenpar`, path-loss model, short names (one conf file) |
| link | one Tx-Rx pair of a batch (row `l` of `rx_pos`) |
| drop | one realization of LSPs, clusters and rays at the link positions; the positions the ray data refer to |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 38.901 clause 7.5 steps 5 to 7) |
| slot | index in the [[ssf_clusters]] outputs: 0 LOS, 1 GR (if enabled), then the N clusters |
| sub-cluster | one of the three parts of a split cluster (step 11, Table 7.5-5) |
| ray | one of the `NumSubPaths` rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) |
| path | a propagation path as `get_channels_*` take it and QRT files store it; [[ssf_paths]] makes one per kept ray (QuaDRiGa also says path for a cluster) |
| tap | rays summed in one output tap: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`"quadriga"`) |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed (QuaDRiGa) |
| segment | the snapshots of one drop over which the ray data are kept (QuaDRiGa) |
| mobility | how the paths follow moved ends within a segment: `phase` (phase rotation per ray, delays and angles fixed) or `drift` (scatterers fixed, geometry recomputed; QuaDRiGa drifting); `auto`: `phase` for the TR methods, `drift` for `"quadriga"` |
| scatterer | first-bounce (FBS) and last-bounce (LBS) interaction point of a ray; one point for LOS (direct line) and GR (ground) (QuaDRiGa) |
| coefficient power | power of a ray without link gain and oxygen loss (key `power` of the ray dict): the powers of TR 38.901 eq. (7.5-30) split over the rays, GR as [[ssf_clusters]] sets it; for data from paths the path gain |
| path gain | linear power gain of a path incl. link gain and oxygen loss |
| link gain | `-PL - O2I + 10·lg(SF)` in dB of a link at a position (`link_gain_dB`), with SF the linear shadow fading, slice 2 of `lsp` |
| random set | the spatially correlated random values of one `set_type` and `set_id` (table in [[gbsm.draw]]) |
| generator | one spatially correlated random field, a [sum-of-sinusoids (SOS)](python_api.html#sec-sum-of-sinusoids-random-number-generator) generator; one column of a random set |
| SOS phases | the random phases of the sinusoids of the generators of a random set (`phase`); evaluated at positions, they give its random values. "Phase" alone is the phase of a coefficient or ray |
| block | the generators of one random quantity in a set |

**Codes.** Path type: 0 LOS, 1 GR, 2 NLOS. Tap index: rays with equal index form one tap; [[ssf_rays]] gives 3·slot + i,
with i the sub-cluster (0, 1, 2) of a split cluster and 0 otherwise: LOS 0, GR 3 if enabled, NLOS cluster `n` (from 0)
3·(1 + G + n) + i with G = 1 if GR is enabled (without GR the first cluster has 3).<br><br>

**Model properties.** UTs closer than the decorrelation distance `SC_lambda` get correlated clusters and rays, so
separately dropped UTs a few wavelengths apart have unrealistically correlated fading: drop once and move the UTs with
[[ssf_paths]] or [[gbsm.single_link_channel]]. Across bands, `"quadriga"` uses one ray set, so the fading is correlated;
the TR methods draw coupling, XPR and phases per frequency, so it is independent, also with `"38.901_MF"`, which keeps
the delays and angles.<br><br>

**Conventions.** `N = NumClusters - 1` is the number of NLOS clusters: `NumClusters` counts the LOS slot, as in
QuaDRiGa. Positions are in m; in the batch functions one row (x, y, z) per link: `rx_pos` is `(n_link, 3)`, `tx_pos` is
`(1, 3)` (one Tx for all links) or `(n_link, 3)`; heights are the z coordinates. The one-link functions take a position
as a `(3,)` array, except [[gbsm.single_link_channel]], which takes one row per snapshot; scatterer positions are
`(3, n)`. Angles are elevations, not TR zenith angles (EOD, EOA, ESD for ZOD, ZOA, ZSD; also in the conf-file keys): in
rad in the GCS, azimuth in [-pi, pi), elevation in [-pi/2, pi/2] with +pi/2 up. Defaults follow TR 38.901 except two
deliberate library choices: the SOS preset `Comb256` with bias removal ([[gbsm.init_phases]], [[gbsm.draw]]) and
`method = "quadriga"` of [[gbsm.rays_from_paths]]; QuaDRiGa variants are options. Units: frequencies in Hz, path loss
in dB, DS in s, KF, SF and XPR as linear power ratios, spreads and `_deg` offsets in degrees.
SECTION_DESC!*/

/*!MD
# conf_file_read
Read and validate a QuaDRiGa scenario configuration file

- A conf file has one `Key = value` assignment per line; `%` starts a comment; blank lines, LF or CRLF line ends and a
  UTF-8 byte order mark at the start are accepted; keys are case-sensitive names of `A-Z`, `a-z`, `0-9` and `_`
- Three kinds of keys:
  - 171 numeric scenario parameters (see [[scenpar_validate]]), one number each; keys that are not in the file have
    their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the QuaDRiGa key `GR_epsilon`
  (the complex ground permittivity comes from [[lsp_generate]]). A key stated more than once has its last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of `SubpathMethod` raises
  `ValueError`, also with `validate = False`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`), read independent of
  the locale; `Inf`, `+Inf` and `-Inf` in any case; a number below the smallest subnormal reads as 0
- Raises `RuntimeError` if the file cannot be opened or is a directory
- Raises `ValueError` with file name and line number for a line that is not blank, not a comment and not a well-formed
  assignment (no `=`, not one name before the `=`, a missing value, a word where a number is required or the reverse,
  text after the value, `NaN`, a number beyond the range of a double, a `PL_` key without a name, a `ShortName` value
  stated twice), and with `validate = True` for a value that fails [[scenpar_validate]]

## Usage:
```
scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys = \
    quadriga_lib.gbsm.conf_file_read( fn, validate )
```

## Inputs:
- **`fn`** — Path to the conf file; str, bytes or path-like object (e.g. `pathlib.Path`)
- **`validate`** — `True`: apply the value checks of [[scenpar_validate]]; `False`: only the line
  grammar is checked; bool or `numpy.bool_`; default: True; `None` selects the default

## Outputs:
- **`scenpar`** — Scenario parameters; dict with 171 float entries in the canonical order: the 138
  numeric keys of QuaDRiGa, then the 33 new keys
- **`plpar`** — Path-loss parameters; dict with the entry `model` (str, only if the file states
  `PL_model`) first, then one entry per `PL_` key in file order, named without the prefix `PL_`; each
  is a 1D array with one or more values; `{}` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite; `False` is no error here, but [[lsp_generate]]
  raises `ValueError` for the scenario; bool
- **`short_names`** — Values of all `ShortName` lines in file order; list of str
- **`ignored_keys`** — Unknown keys, each once, in the order of their first occurrence; list of str

## See also:
- [[scenpar_validate]] (value checks and defaults)
- [[conf_file_list]] (find the conf files in a set of directories)
- [[conf_file_write]] (write a conf file)
MD!*/

py::tuple conf_file_read(py::object fn,
                         py::handle validate)
{
    // Read inputs
    const std::string fn_s = qd_python_path(fn, "fn");
    const bool validate_b = qd_python_flag(validate, true, "validate");

    // Output allocation
    std::unordered_map<std::string, double> scenpar;
    std::unordered_map<std::string, std::vector<double>> plpar;
    std::string pl_model;
    std::vector<std::string> key_order, pl_order, short_names, ignored_keys;
    arma::mat lsp_xcorr;

    auto lsp_xcorr_py = qd_python_init_output((arma::uword)8, (arma::uword)8, &lsp_xcorr);

    // Call library function
    const bool lsp_xcorr_chk = quadriga_lib::conf_file_read(fn_s, scenpar, &pl_model, &plpar,
                                                            &lsp_xcorr, &key_order, &pl_order, &short_names,
                                                            &ignored_keys, validate_b);

    // Scenario parameters: numeric keys in canonical order
    auto scenpar_py = qd_python_scenpar2dict(scenpar, key_order);

    // Path-loss parameters: model first, then the numeric keys in file order as 1D arrays
    auto plpar_py = qd_python_plpar2dict(pl_model, plpar, pl_order);

    auto short_names_py = qd_python_copy2list(short_names);
    auto ignored_keys_py = qd_python_copy2list(ignored_keys);

    return py::make_tuple(scenpar_py, plpar_py, lsp_xcorr_py, lsp_xcorr_chk, short_names_py, ignored_keys_py);
}

// pybind11 declaration:
// m.def("conf_file_read", &conf_file_read,
//       py::arg("fn"),
//       py::arg("validate") = true);
