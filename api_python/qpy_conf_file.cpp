// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# conf_file_list
List the scenarios defined by the conf files in a set of directories

- Lists the regular files and symbolic links to regular files named `<scenario>.conf` (case-sensitive, non-empty
  scenario name) in each directory, not recursive: the directories in the given order, the files of one directory in
  byte-wise sorted order of their names
- A file name found in an earlier directory is skipped, so a local folder can override the files of a shared one
- With `parse_shortnames = True` every line `ShortName = name` that is not a comment adds an entry right after its file;
  other lines are not examined
- Raises `RuntimeError` if a short name equals an earlier entry, if a file name equals an earlier short name, if a
  `ShortName` value is not one name, or if a listed file cannot be opened
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty string `""` is the
  current directory

## Usage:
```
scenarios, file_names, file_dirs = quadriga_lib.gbsm.conf_file_list( dirs, *, parse_shortnames )
```

## Inputs:
- **`dirs`** — Directories, searched in this order; one path or a list or tuple of paths, each a str,
  bytes or path-like object (e.g. `pathlib.Path`); an empty list or `None` lists nothing
- **`parse_shortnames`** — `True`: read the files and add their short names; `False`: list the files
  only, which is faster; default: True

## Outputs:
- **`scenarios`** — Scenario names and short names; list of str; length `n_scen`
- **`file_names`** — File name including `.conf` for each entry; list of str; length `n_scen`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs` (a path-like object
  converted with `os.fspath`); list of str; length `n_scen`

## See also:
- [[conf_file_read]] (read one of the files)
MD!*/

py::tuple conf_file_list(py::handle dirs,
                         py::handle parse_shortnames)
{
    // Read inputs; one path (str, bytes, os.PathLike) is one directory, also when it is empty (= current directory)
    std::vector<std::string> dirs_v;
    std::string path;
    if (qd_python_try_path(dirs, path))
        dirs_v.push_back(path);
    else if (py::isinstance<py::list>(dirs) || py::isinstance<py::tuple>(dirs))
        for (py::handle entry : py::reinterpret_borrow<py::sequence>(dirs))
        {
            if (!qd_python_try_path(entry, path))
                throw std::invalid_argument("Input 'dirs' must be a path, a list or tuple of paths, or None.");
            dirs_v.push_back(path);
        }
    else if (!dirs.is_none())
        throw std::invalid_argument("Input 'dirs' must be a path, a list or tuple of paths, or None.");
    const bool parse_shortnames_b = qd_python_flag(parse_shortnames, true, "parse_shortnames");

    std::vector<std::string> file_names, file_dirs;

    const auto scenarios = quadriga_lib::conf_file_list(dirs_v, &file_names, &file_dirs, parse_shortnames_b);

    auto scenarios_py = qd_python_copy2list(scenarios);
    auto file_names_py = qd_python_copy2list(file_names);
    auto file_dirs_py = qd_python_copy2list(file_dirs);

    return py::make_tuple(scenarios_py, file_names_py, file_dirs_py);
}

/*!SECTION_DESC
All functions of this section are in the submodule `quadriga_lib.gbsm`: `ssf_rays` is called as
`quadriga_lib.gbsm.ssf_rays`, `gbsm.draw` as `quadriga_lib.gbsm.draw`.<br><br>

A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the *scenario*
(e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The functions of this
section implement 3GPP TR 38.901 clause 7.5 steps 2 to 10 and the sub-cluster split of step 11 (LOS probability, path
loss, outdoor-to-indoor (O2I) penetration loss, large-scale parameters (LSPs), clusters and rays, without antennas)
with the spatial consistency of clause 7.6.3, oxygen absorption (clause 7.6.1) and polarization power variability
(clause 7.6.16); [[gbsm.single_link_channel]] computes the coefficients of step 11 with the antennas.<br><br>

**Scenario parameters.** A scenario is a dict of numeric keys (`scenpar`), a path-loss dict (`plpar`: the entry `model`
and the model parameters) and optional short names, read from a conf file by [[conf_file_read]]. 23 files for TR 38.901
v19.5.0 come with the library, one per scenario and LOS and indoor state (e.g. `3GPP_38.901_v19_UMa_LOS`, `_NLOS`,
`_LOS_O2I`, `_NLOS_O2I`; InH is `Indoor`, InF NLOS has one file per clutter type), in `data/config` of the source tree
and in `share/quadriga-lib/config` of an installed package (for pip,
`os.path.join(sys.prefix, "share", "quadriga-lib", "config")`); [[conf_file_list]] finds them. Their comments give the
meaning and unit of each key; [[conf_file_write]] with `write_defaults = True` writes the complete default set with
these comments. Every function that takes `scenpar` accepts the dict of [[conf_file_read]] or [[scenpar_validate]] or
any subset of its keys (`None` = all defaults), completes a copy with the defaults of missing keys and validates it with
[[scenpar_validate]], whose messages pass through, also for keys the function does not use. A key `SubpathMethod`
(QuaDRiGa's structs) must be `"legacy"`. Invalid input raises `ValueError`, a problem with a file's access or
content `RuntimeError`.<br><br>

**Workflow** for one drop of a set of links (terms: see Terms below):

1. [[conf_file_read]]: the scenarios, one per LOS and indoor state
2. Per link, by the caller with own SOS generators ([[sos.init_phases]], [[sos.rand]], [[sos.randn]]): the indoor
   state, a Uniform(0,1) value at the decorrelation distance `IO_state_lambda` of the O2I files below an indoor ratio
   of the caller's choice (e.g. TR 38.901 Table 7.2-1), and for indoor UTs [[o2i_loss]] (`loss_dB`, `d2d_in`); then the
   LOS state, a Uniform(0,1) value at `LOS_state_lambda` of the NLOS file below `p_los` of [[los_probability]] at
   `d2d_out = max(d2D - d2d_in, 0)`. The caller chooses the `los_probability` model and the `o2i_loss` models and
   weights (TR 38.901 Tables 7.4.2-1, 7.4.3-2). The states assign each link to a scenario and give the `o2i` flags of
   [[ssf_clusters]]; steps 3 to 6 run per scenario
3. Per BS site and scenario, once, with one seed for all sets: [[gbsm.init_phases]] of the random sets in the table
   below; `"lsf"` with the link positions for the bias removal
4. [[gbsm.draw]] of each set at the links (the shared `"ray"` set with `center_freq`, the list later passed to
   [[ssf_rays]]); [[path_loss]] (`pl_dB`, `sf_sigma_dB`; column 10 of the `"lsf"` set as `rnd_he`); [[lsp_generate]]
   with `sf_sigma_dB` (`lsp`, `abs_toa`, `gr_eps`, `eod_offset_deg`, `lsp_mu`, `lsp_sigma` for step 5)
5. [[ssf_clusters]] (`n_cluster` is the caller's choice) and [[ssf_rays]] (one ray dict per link), called as in the
   table below
6. Per link and segment: [[ssf_scatterers]] if the mobility is `drift`; then [[gbsm.single_link_channel]] with the
   link gain (see Terms) as `link_gain_dB` (TR methods: once per ray dict, which holds one frequency; `"quadriga"`:
   once per `i_freq`), or [[ssf_paths]] for other channel functions

| Method | Random sets (`set_id`) | Calls in step 5 |
| --- | --- | --- |
| `"38.901"` | `"lsf"`; `"cluster"` -1 and one per frequency; `"ray"` one per frequency | per frequency `k`: [[ssf_clusters]] and [[ssf_rays]] (`i_freq = 0`) with frequency `k` of all per-frequency inputs, frequency axis kept (`lsp[:, k:k+1]`, likewise `center_freq`, `lsp_mu`, `lsp_sigma`, `eod_offset_deg`, `gr_eps`), and the sets of frequency `k` (`rnd_freq`, `rnd_ray`) as [[gbsm.draw]] returns them |
| `"38.901_MF"` | as `"38.901"` | [[lsp_generate]] at the anchor frequency with the same `rnd` gives `lsp_anchor`; [[ssf_clusters]] once; [[ssf_rays]] per frequency `k` with `i_freq = k` |
| `"quadriga"` | `"lsf"`; `"cluster"` -1; `"ray"` -1 | [[ssf_clusters]] and [[ssf_rays]] once for all frequencies |

For moved positions, evaluate [[gbsm.draw]] (`"lsf"`), [[path_loss]], [[lsp_generate]] and for indoor UTs [[o2i_loss]]
again with the same SOS phases: they give the link gain (see Terms) and `kf_ratio` = KF(position) / KF(drop) from
slice 1 of `lsp`. Ray dicts can also come from path data, with [[gbsm.rays_from_paths]] (e.g. of [[qrt_file_read]]), or
be filled directly (keys in [[ssf_rays]]); continue with step 6.<br><br>

**Terms.** TR 38.901 names where they exist; QuaDRiGa names are marked:

| Term | Meaning |
| --- | --- |
| scenario | type of environment with its statistics: numeric keys `scenpar`, path-loss model, short names (one conf file) |
| link | one Tx-Rx pair of a batch (column `l` of `rx_pos`) |
| drop | one realization of LSPs, clusters and rays at the link positions; the positions the ray data refer to |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 38.901 clause 7.5 steps 5 to 7) |
| slot | index in the [[ssf_clusters]] outputs: 0 LOS, 1 ground reflection (GR, if enabled), then the N clusters |
| sub-cluster | one of the three parts of a split cluster (step 11, Table 7.5-5) |
| ray | one of the `NumSubPaths` rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) |
| kept ray | a LOS or GR ray, or an NLOS ray with power > 0 at some frequency (at `i_freq` in [[gbsm.single_link_channel]]) |
| path | a propagation path as `channel.spherical` / `channel.planar` take it and QRT files store it; [[ssf_paths]] makes one per kept ray (QuaDRiGa also says path for a cluster) |
| tap | rays summed in one output tap: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`"quadriga"`) |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed (QuaDRiGa) |
| segment | the snapshots of one drop over which the ray data are kept (QuaDRiGa) |
| mobility | how the paths follow moved ends within a segment: `phase` (phase rotation per ray, delays and angles fixed) or `drift` (scatterers fixed, geometry recomputed; QuaDRiGa drifting; validity in [[ssf_paths]]); `auto`: `phase` for the TR methods, `drift` for `"quadriga"` |
| scatterer | first-bounce (FBS) and last-bounce (LBS) interaction point of a ray; one point for LOS (direct line) and GR (ground) (QuaDRiGa) |
| coefficient power | power of a ray without link gain and oxygen loss (key `power` of the ray dict): the powers of TR 38.901 eq. (7.5-30) split over the rays, GR as [[ssf_clusters]] sets it; for data from paths the path gain |
| path gain | linear power gain of a path incl. link gain and oxygen loss |
| link gain | `-pl_dB - loss_dB + 10·lg(SF)` in dB of a link at a position and frequency (`link_gain_dB`), with `pl_dB` of [[path_loss]], `loss_dB` of [[o2i_loss]] (0 outdoors) and SF the linear shadow fading, slice 2 of `lsp` |
| random set | the spatially correlated random values of one `set_type` and `set_id` (table in [[gbsm.draw]]) |
| generator | one spatially correlated random field, a [sum-of-sinusoids (SOS)](python_api.html#sec-sum-of-sinusoids-random-number-generator) generator; one column of a random set |
| SOS phases | the random phases of the sinusoids of the generators of a random set (`phase`); evaluated at positions, they give its random values. "Phase" alone is the phase of a coefficient or ray |
| block | the generators of one random quantity in a set |

**Codes.** Path type: 0 LOS, 1 GR, 2 NLOS. Tap: rays with equal value form one tap; [[ssf_rays]] gives 3·s + i, with s
the slot and i the sub-cluster (0, 1, 2) of a split cluster, else 0 (LOS 0, GR 3, the first NLOS cluster 3 without and
6 with GR).<br><br>

**Model properties.** UTs closer than `SC_lambda` get correlated clusters and rays (TR 38.901 clause 7.6.3.1). In the TR
methods, the tap fading of separately dropped UTs then stays correlated over metres instead of a few wavelengths
(tap-power correlation 0.5 at about 0.3·`SC_lambda`, 0.1 at 0.8·`SC_lambda`); `"quadriga"` avoids this. In all methods,
the cluster delays and angles of separately dropped UTs change faster than for a moved UT, so their single-frequency
channel and Doppler are wrong at small distances: for close UTs, drop once and move them with [[ssf_paths]] or
[[gbsm.single_link_channel]]. Across bands, `"quadriga"` uses one ray set, so the fading is correlated; the TR methods
draw coupling, XPR and phases per frequency, so it is independent, also with `"38.901_MF"`, which keeps the delays and
angles.<br><br>

**Conventions.** `N = NumClusters - 1` is the number of NLOS clusters: `NumClusters` counts the LOS slot, as in
QuaDRiGa. Positions are in m; in the batch functions one column (x, y, z) per link: `rx_pos` is `(3, n_link)` (empty: no
links), `tx_pos` is `(3,)` (one Tx for all links) or `(3, n_link)`; heights are the z coordinates. The one-link
functions take a position as a `(3,)` array, and [[gbsm.single_link_channel]] one column per snapshot; scatterer
positions are `(3, n)`. Angles are elevations, not TR zenith angles (EOD, EOA, ESD for ZOD, ZOA, ZSD; also in the
conf-file keys): in rad in the GCS, azimuth in [-pi, pi), elevation in [-pi/2, pi/2] with +pi/2 up. Defaults follow TR
38.901 except two deliberate library choices: the SOS preset `Comb256` with bias removal ([[gbsm.init_phases]],
[[gbsm.draw]]) and `method = "quadriga"` of [[gbsm.rays_from_paths]]; QuaDRiGa variants are options. Units: frequencies
in Hz, path loss in dB, DS in s, KF, SF and XPR as linear power ratios, spreads and `_deg` offsets in degrees; conf-file
keys and `scenpar` keys have the units of the file comments (e.g. `PerClusterDS` in ns). Messages name the real and
imaginary part of `gr_eps` `gr_eps_re` and `gr_eps_im`.
SECTION_DESC!*/

/*!MD
# conf_file_read
Read and validate a QuaDRiGa scenario configuration file

- A conf file has one `Key = value` assignment per line; `%` starts a comment; blank lines, LF or CRLF line ends and a
  UTF-8 byte order mark at the start are accepted; keys are case-sensitive names of `A-Z`, `a-z`, `0-9` and `_`
- Three kinds of keys:
  - the numeric scenario parameters (see [[scenpar_validate]]), one number each; keys that are not in the file have
    their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the QuaDRiGa key `GR_epsilon`
  (the complex ground permittivity comes from [[lsp_generate]]). A key stated more than once has its last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of `SubpathMethod` raises
  `RuntimeError`, also with `validate = False`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`), read independent of
  the locale; `Inf`, `+Inf` and `-Inf` in any case
- Raises `RuntimeError` if the file cannot be opened or is a directory
- Raises `RuntimeError` with file name and line number for a line that is not blank, not a comment and not a well-formed
  assignment (no `=`, not one name before the `=`, a missing value, a word where a number is required or the reverse,
  text after the value, `NaN`, a number beyond the range of a double, a `PL_` key without a name, a `ShortName` value
  stated twice), and with `validate = True` for a value that fails [[scenpar_validate]]
- The outputs differ from C++: the path-loss model `pl_model` is the key `model` of `plpar`, the C++ outputs
  `key_order` and `pl_order` set the order of the keys, and `lsp_xcorr_pd` is called `lsp_xcorr_chk`

## Usage:
```
scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys = \
    quadriga_lib.gbsm.conf_file_read( fn, *, validate )
```

## Inputs:
- **`fn`** — Path to the conf file
- **`validate`** — `True`: apply the value checks of [[scenpar_validate]]; `False`: only the line
  grammar is checked; default: True

## Outputs:
- **`scenpar`** — Scenario parameters; dict with all scenario keys as floats in the canonical order of
  [[scenpar_validate]]
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

py::tuple conf_file_read(const py::handle &fn,
                         py::handle validate)
{
    const std::string fn_s = qd_python_path(fn, "fn");
    const bool validate_b = qd_python_flag(validate, true, "validate");

    std::unordered_map<std::string, std::vector<double>> plpar;
    std::string pl_model;
    std::vector<std::string> key_order, pl_order, short_names, ignored_keys;
    arma::mat lsp_xcorr;

    auto lsp_xcorr_py = qd_python_init_output((arma::uword)8, (arma::uword)8, &lsp_xcorr);

    bool lsp_xcorr_chk = false;
    const auto scenpar = quadriga_lib::conf_file_read(fn_s, &pl_model, &plpar, &lsp_xcorr, &lsp_xcorr_chk, &key_order,
                                                      &pl_order, &short_names, &ignored_keys, validate_b);

    // Scenario parameters: numeric keys in canonical order
    auto scenpar_py = qd_python_scenpar2dict(scenpar, key_order);

    // Path-loss parameters: model first, then the numeric keys in file order as 1D arrays
    auto plpar_py = qd_python_plpar2dict(pl_model, plpar, pl_order);

    auto short_names_py = qd_python_copy2list(short_names);
    auto ignored_keys_py = qd_python_copy2list(ignored_keys);

    return py::make_tuple(scenpar_py, plpar_py, lsp_xcorr_py, lsp_xcorr_chk, short_names_py, ignored_keys_py);
}

/*!MD
# conf_file_write
Write a QuaDRiGa scenario configuration file

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files of QuaDRiGa; every
  written file passes [[conf_file_read]] with `validate = True`, without ignored keys and with the same values bit for
  bit
- The header has the scenario name (the file name without `.conf`), `ref`, the validity range and the short names. Up to
  10 sections follow, each with an optional line `% See: ref; ref_sec` and, for the LSPs and the path-loss model, the
  formulas as comments; the `plpar` keys are written in dict order. Sections by entry of `ref_sec`: 0 large-scale
  distributions, 1 large-scale decorrelation distances, 2 inter-parameter correlations, 3 model parameters, 4
  small-scale decorrelation distance, 5 ground reflection, 6 absolute TOA, 7 path-loss model, 8 EoD offset, 9 optional
  modelling components; 0 to 7 are those of QuaDRiGa, so its 8 section references can be passed unchanged
- Numbers are written independent of the locale; `Inf`, `-Inf` and `-0` are kept
- `write_defaults = False`: only keys whose value differs from the default (bit for bit); a section without a written
  key is left out
- All inputs are validated before the file is opened: the checks of [[scenpar_validate]]; `model`, the other keys of
  `plpar` and the short names are names, the short names distinct; every `plpar` key has at least one value and no
  `NaN`; `fn`, `ref` and `ref_sec` have no line break
- A failed call creates no file and leaves an existing file unchanged; the directory must be writable

## Usage:
```
quadriga_lib.gbsm.conf_file_write( fn, scenpar, plpar, *, short_names, ref, ref_sec, valid_range, \
    write_defaults )
```

## Inputs:
- **`fn`** — Path of the conf file to write
- **`scenpar`** — Scenario parameters; dict with any subset of the keys returned by
  [[conf_file_read]]; real scalars (int, float, bool or a numpy value with one element); missing keys
  have their default; `None` or `{}` writes the default set
- **`plpar`** — Path-loss parameters; dict with the entry `model` (str) and one real scalar, list or
  1D array per further entry, named without the prefix `PL_`, as returned by [[conf_file_read]];
  written in dict order; `None` or `{}`: no path-loss section; default: None
- **`short_names`** — Short names of the scenario; list of str or one str; each one name of the
  characters `A-Z`, `a-z`, `0-9`, `_`; distinct; default: None
- **`ref`** — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; str; default: None (no reference)
- **`ref_sec`** — Reference per section, indexed as listed above; list of up to 10 str; an empty or
  missing entry writes no reference; default: None
- **`valid_range`** — Validity range, written as comment; rows: frequency in GHz, distance, BS
  height, UT height in m; columns: lower and upper end, one value is written if both are equal; a
  row with `NaN` is not written; `(4, 2)`; default: None
- **`write_defaults`** — `True`: write all keys; `False`: write only keys that differ from their
  default; default: False

## See also:
- [[conf_file_read]] (read the file back)
- [[scenpar_validate]] (value checks and defaults)
MD!*/

void conf_file_write(const py::handle &fn,
                     py::handle scenpar,
                     py::handle plpar,
                     py::handle short_names,
                     py::handle ref,
                     py::handle ref_sec,
                     py::handle valid_range,
                     py::handle write_defaults)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    // Scenario parameters: dict of real scalars
    const auto scenpar_m = qd_python_dict2scenpar(scenpar, "scenpar");

    // Path-loss parameters: 'model' and the numeric entries; the dict order is the write order
    std::string pl_model;
    std::vector<std::string> pl_order;
    const auto plpar_m = qd_python_dict2plpar(plpar, pl_model, pl_order, "plpar");

    const auto short_names_v = qd_python_list2vector_Strings(short_names);

    if (!ref.is_none() && !py::isinstance<py::str>(ref))
        throw std::invalid_argument("Input 'ref' must be a str or None.");
    const std::string ref_s = ref.is_none() ? std::string() : py::cast<std::string>(ref);

    // A single str would silently become the reference of section 0
    if (py::isinstance<py::str>(ref_sec))
        throw std::invalid_argument("Input 'ref_sec' must be a list of str or None.");
    const auto ref_sec_v = qd_python_list2vector_Strings(ref_sec);

    const auto valid_range_a = qd_python_numpy2arma_Mat<double>(valid_range, true);
    const bool write_defaults_b = qd_python_flag(write_defaults, false, "write_defaults");

    quadriga_lib::conf_file_write(fn_s, scenpar_m, pl_model, plpar_m, pl_order, short_names_v,
                                  ref_s, ref_sec_v, valid_range_a, write_defaults_b);
}

/*!MD
# scenpar_validate
Validate scenario parameters and complete them with default values

- Checks scenario parameters, e.g. ones built in code, and adds every missing key with its default value, so that
  `scenpar_out` has all 171 keys: the 138 numeric keys of QuaDRiGa and 33 keys for TR 38.901 v19.5.0, which have no
  effect at their default of 0. `None` or an empty dict gives the complete default set
- A name that is not one of the keys is an error, also with `validate = False`
- Value checks with `validate = True`; `NaN` is an error for every key:
  - Integer from 1 to 1000: `NumClusters`, `NumSubPaths`
  - Finite and > 0: `r_DS`
  - Finite and >= 0: the 8 decorrelation distances (`DS_lambda` ... `XPR_lambda`), `SC_lambda`, `absTOA_lambda`,
    `PerClusterDS`, `PerClusterDS_min`, `PerClusterAS_D`, `PerClusterAS_A`, `PerClusterES_D`, `PerClusterES_A`,
    `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`, `LSP_fc_min`, `EoD_offset_b`, `EoD_offset_hcl`,
    `PerClusterDS_fc_min`, `PerClusterES_D_scale`, `absTOA_Lmax`, `NF_alpha`, `NF_beta`, `SNS_sigma`, `SNS_A`, `SNS_B`,
    `SNS_R`, `SNS_delta_var`, `PolVar_sigma`, `LOS_state_lambda`, `IO_state_lambda`, `O2I_d2D_in_max`, `DS_VS`,
    `DS_VS_a`, `DS_VS_b`
  - Integer from 0 to 1000: `NumClusters_min`, `NF_k1`
  - In the range from 0 to 1: `SNS_mu`
  - In the range from -1 to 1: the 28 correlations (`asD_ds` to `xpr_esa`)
  - All other numeric keys accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `DS_VS_a` and `DS_VS_b` must not both be 0 if `DS_VS` is > 0
- `LOS_scatter_radius`, `NF_*` and `SNS_*` are checked but not used by any function of this version

## Usage:
```
scenpar_out, lsp_xcorr, lsp_xcorr_chk = quadriga_lib.gbsm.scenpar_validate( scenpar_in, *, validate )
```

## Inputs:
- **`scenpar_in`** — Scenario parameters; dict with any subset of the keys returned by
  [[conf_file_read]]; real scalars (int, float, bool or a numpy value with one element); `None` or
  `{}` returns the default set; default: None
- **`validate`** — `False`: only complete with defaults, no value checks (`NaN` is accepted); default: True

## Outputs:
- **`scenpar_out`** — Complete scenario parameters; given values are kept, missing keys have their
  default; dict with all keys as floats in the canonical order, QuaDRiGa's keys first
- **`lsp_xcorr`** — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite; `False` also if it contains `NaN`; `False` is no
  error here, but [[lsp_generate]] raises `ValueError` for such parameters; bool

## See also:
- [[conf_file_read]] (read the parameters from a conf file)
- [[conf_file_write]] (applies the same checks before writing)
MD!*/

py::tuple scenpar_validate(py::handle scenpar_in,
                           py::handle validate)
{
    // Read inputs: dict of real scalars
    auto scenpar = qd_python_dict2scenpar(scenpar_in, "scenpar_in");
    const bool validate_b = qd_python_flag(validate, true, "validate");

    std::vector<std::string> key_order;
    arma::mat lsp_xcorr;

    auto lsp_xcorr_py = qd_python_init_output((arma::uword)8, (arma::uword)8, &lsp_xcorr);

    // Completes 'scenpar' in place (a copy; the input dict is not changed)
    const bool lsp_xcorr_chk = quadriga_lib::scenpar_validate(scenpar, &lsp_xcorr, &key_order,
                                                              validate_b);

    // Scenario parameters: numeric keys in canonical order
    auto scenpar_out_py = qd_python_scenpar2dict(scenpar, key_order);

    return py::make_tuple(scenpar_out_py, lsp_xcorr_py, lsp_xcorr_chk);
}
