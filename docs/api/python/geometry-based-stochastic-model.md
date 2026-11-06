# Geometry-based stochastic model

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-geometry-based-stochastic-model

| Function | Summary |
| --- | --- |
| [conf_file_list](#conf_file_list) | List the scenarios defined by the conf files in a set of directories |
| [conf_file_read](#conf_file_read) | Read and validate a QuaDRiGa scenario configuration file |
| [conf_file_write](#conf_file_write) | Write a QuaDRiGa scenario configuration file |
| [gbsm.draw](#gbsm.draw) | Spatially correlated normalized random values of one random set at a set of links |
| [gbsm.init_phases](#gbsm.init_phases) | SOS phases of one random set of a BS site and scenario |
| [gbsm.rays_from_paths](#gbsm.rays_from_paths) | Ray data of one link from path data (positions, gains, lengths, polarization), e.g. read with qrt_file_read |
| [gbsm.single_link_channel](#gbsm.single_link_channel) | Channel coefficients of one link at one frequency over one segment |
| [los_probability](#los_probability) | LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models |
| [lsp_generate](#lsp_generate) | Large-scale parameters of a set of links from normalized random values |
| [o2i_loss](#o2i_loss) | O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3 |
| [path_loss](#path_loss) | Path loss and shadow-fading standard deviation of a set of links |
| [scenpar_validate](#scenpar_validate) | Validate scenario parameters and complete them with default values |
| [ssf_clusters](#ssf_clusters) | Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7) |
| [ssf_paths](#ssf_paths) | Antenna-free paths of one link at one position |
| [ssf_rays](#ssf_rays) | Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16) |
| [ssf_scatterers](#ssf_scatterers) | First- and last-bounce scatterers of the rays of one link (two-bounce solver) |

All functions of this section are in the submodule `quadriga_lib.gbsm`: `ssf_rays` is called as
`quadriga_lib.gbsm.ssf_rays`, `gbsm.draw` as `quadriga_lib.gbsm.draw`.

A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the *scenario*
(e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The functions of this
section implement 3GPP TR 38.901 clause 7.5 steps 2 to 10 and the sub-cluster split of step 11 (LOS probability, path
loss, outdoor-to-indoor (O2I) penetration loss, large-scale parameters (LSPs), clusters and rays, without antennas)
with the spatial consistency of clause 7.6.3, oxygen absorption (clause 7.6.1) and polarization power variability
(clause 7.6.16); [gbsm.single_link_channel](#gbsm.single_link_channel) computes the coefficients of step 11 with the antennas.

**Scenario parameters.** A scenario is a dict of numeric keys (`scenpar`), a path-loss dict (`plpar`: the entry `model`
and the model parameters) and optional short names, read from a conf file by [conf_file_read](#conf_file_read). 23 files for TR 38.901
v19.5.0 come with the library, one per scenario and LOS and indoor state (e.g. `3GPP_38.901_v19_UMa_LOS`, `_NLOS`,
`_LOS_O2I`, `_NLOS_O2I`; InH is `Indoor`, InF NLOS has one file per clutter type), in `data/config` of the source tree
and in `share/quadriga-lib/config` of an installed package (for pip,
`os.path.join(sys.prefix, "share", "quadriga-lib", "config")`); [conf_file_list](#conf_file_list) finds them. Their comments give the
meaning and unit of each key; [conf_file_write](#conf_file_write) with `write_defaults = True` writes the complete default set with
these comments. Every function that takes `scenpar` accepts the dict of [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate) or
any subset of its keys (`None` = all defaults), completes a copy with the defaults of missing keys and validates it with
[scenpar_validate](#scenpar_validate), whose messages pass through, also for keys the function does not use. A key `SubpathMethod`
(QuaDRiGa's structs) must be `"legacy"`. Invalid input raises `ValueError`, a failure to read or write a file
`RuntimeError`.

**Workflow** for one drop of a set of links (terms: see Terms below):

1. [conf_file_read](#conf_file_read): the scenarios, one per LOS and indoor state
2. Per link, by the caller with own SOS generators ([sos.init_phases](sum-of-sinusoids-random-number-generator.md#sos.init_phases), [sos.rand](sum-of-sinusoids-random-number-generator.md#sos.rand), [sos.randn](sum-of-sinusoids-random-number-generator.md#sos.randn)): the indoor
   state, a Uniform(0,1) value at the decorrelation distance `IO_state_lambda` of the O2I files below an indoor ratio
   of the caller's choice (e.g. TR 38.901 Table 7.2-1), and for indoor UTs [o2i_loss](#o2i_loss) (`loss`, `d2d_in`); then the
   LOS state, a Uniform(0,1) value at `LOS_state_lambda` of the NLOS file below `p_los` of [los_probability](#los_probability) at
   `d2d_out = max(d2D - d2d_in, 0)`. The caller chooses the `los_probability` model and the `o2i_loss` models and
   weights (TR 38.901 Tables 7.4.2-1, 7.4.3-2). The states assign each link to a scenario and give the `o2i` flags of
   [ssf_clusters](#ssf_clusters); steps 3 to 6 run per scenario
3. Per BS site and scenario, once, with one seed for all sets: [gbsm.init_phases](#gbsm.init_phases) of the random sets in the table
   below; `"lsf"` with the link positions for the bias removal
4. [gbsm.draw](#gbsm.draw) of each set at the links (the shared `"ray"` set with `center_freq`, the list later passed to
   [ssf_rays](#ssf_rays)); [path_loss](#path_loss) (`pl`, `sf_sigma`; column 10 of the `"lsf"` set as `rnd_he`); [lsp_generate](#lsp_generate) with
   `sf_sigma` (`lsp`, `abs_toa`, `gr_eps`, `eod_offset_deg`, `lsp_mu`, `lsp_sigma` for step 5)
5. [ssf_clusters](#ssf_clusters) (`n_cluster` is the caller's choice) and [ssf_rays](#ssf_rays) (one ray dict per link), called as in the
   table below
6. Per link and segment: [ssf_scatterers](#ssf_scatterers) if the mobility is `drift`; then [gbsm.single_link_channel](#gbsm.single_link_channel) with the
   link gain (see Terms) as `link_gain_dB` (TR methods: once per ray dict, which holds one frequency; `"quadriga"`:
   once per `i_freq`), or [ssf_paths](#ssf_paths) for other channel functions

| Method | Random sets (`set_id`) | Calls in step 5 |
| --- | --- | --- |
| `"38.901"` | `"lsf"`; `"cluster"` -1 and one per frequency; `"ray"` one per frequency | per frequency `k`: [ssf_clusters](#ssf_clusters) and [ssf_rays](#ssf_rays) (`i_freq = 0`) with frequency `k` of all per-frequency inputs, frequency axis kept (`lsp[:, k:k+1]`, likewise `center_freq`, `lsp_mu`, `lsp_sigma`, `eod_offset_deg`, `gr_eps`), and the sets of frequency `k` (`rnd_freq`, `rnd_ray`) as [gbsm.draw](#gbsm.draw) returns them |
| `"38.901_MF"` | as `"38.901"` | [lsp_generate](#lsp_generate) at the anchor frequency with the same `rnd` gives `lsp_anchor`; [ssf_clusters](#ssf_clusters) once; [ssf_rays](#ssf_rays) per frequency `k` with `i_freq = k` |
| `"quadriga"` | `"lsf"`; `"cluster"` -1; `"ray"` -1 | [ssf_clusters](#ssf_clusters) and [ssf_rays](#ssf_rays) once for all frequencies |

For moved positions, evaluate [gbsm.draw](#gbsm.draw) (`"lsf"`), [path_loss](#path_loss), [lsp_generate](#lsp_generate) and for indoor UTs [o2i_loss](#o2i_loss)
again with the same SOS phases: they give the link gain (see Terms) and `kf_ratio` = KF(position) / KF(drop) from
slice 1 of `lsp`. Ray dicts can also come from path data, with [gbsm.rays_from_paths](#gbsm.rays_from_paths) (e.g. of [qrt_file_read](channel-functions.md#qrt_file_read)), or
be filled directly (keys in [ssf_rays](#ssf_rays)); continue with step 6.

**Terms.** TR 38.901 names where they exist; QuaDRiGa names are marked:

| Term | Meaning |
| --- | --- |
| scenario | type of environment with its statistics: numeric keys `scenpar`, path-loss model, short names (one conf file) |
| link | one Tx-Rx pair of a batch (row `l` of `rx_pos`) |
| drop | one realization of LSPs, clusters and rays at the link positions; the positions the ray data refer to |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 38.901 clause 7.5 steps 5 to 7) |
| slot | index in the [ssf_clusters](#ssf_clusters) outputs: 0 LOS, 1 ground reflection (GR, if enabled), then the N clusters |
| sub-cluster | one of the three parts of a split cluster (step 11, Table 7.5-5) |
| ray | one of the `NumSubPaths` rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) |
| kept ray | a LOS or GR ray, or an NLOS ray with power > 0 at some frequency (at `i_freq` in [gbsm.single_link_channel](#gbsm.single_link_channel)) |
| path | a propagation path as `get_channels_*` take it and QRT files store it; [ssf_paths](#ssf_paths) makes one per kept ray (QuaDRiGa also says path for a cluster) |
| tap | rays summed in one output tap: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`"quadriga"`) |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed (QuaDRiGa) |
| segment | the snapshots of one drop over which the ray data are kept (QuaDRiGa) |
| mobility | how the paths follow moved ends within a segment: `phase` (phase rotation per ray, delays and angles fixed) or `drift` (scatterers fixed, geometry recomputed; QuaDRiGa drifting); `auto`: `phase` for the TR methods, `drift` for `"quadriga"` |
| scatterer | first-bounce (FBS) and last-bounce (LBS) interaction point of a ray; one point for LOS (direct line) and GR (ground) (QuaDRiGa) |
| coefficient power | power of a ray without link gain and oxygen loss (key `power` of the ray dict): the powers of TR 38.901 eq. (7.5-30) split over the rays, GR as [ssf_clusters](#ssf_clusters) sets it; for data from paths the path gain |
| path gain | linear power gain of a path incl. link gain and oxygen loss |
| link gain | `-pl - loss + 10·lg(SF)` in dB of a link at a position and frequency (`link_gain_dB`), with `pl` of [path_loss](#path_loss), `loss` of [o2i_loss](#o2i_loss) (0 outdoors) and SF the linear shadow fading, slice 2 of `lsp` |
| random set | the spatially correlated random values of one `set_type` and `set_id` (table in [gbsm.draw](#gbsm.draw)) |
| generator | one spatially correlated random field, a [sum-of-sinusoids (SOS)](sum-of-sinusoids-random-number-generator.md) generator; one column of a random set |
| SOS phases | the random phases of the sinusoids of the generators of a random set (`phase`); evaluated at positions, they give its random values. "Phase" alone is the phase of a coefficient or ray |
| block | the generators of one random quantity in a set |

**Codes.** Path type: 0 LOS, 1 GR, 2 NLOS. Tap: rays with equal value form one tap; [ssf_rays](#ssf_rays) gives 3·s + i, with s
the slot and i the sub-cluster (0, 1, 2) of a split cluster, else 0 (LOS 0, GR 3, the first NLOS cluster 3 without and
6 with GR).

**Model properties.** UTs closer than the decorrelation distance `SC_lambda` get correlated clusters and rays, so
separately dropped UTs a few wavelengths apart have unrealistically correlated fading: drop once and move the UTs with
[ssf_paths](#ssf_paths) or [gbsm.single_link_channel](#gbsm.single_link_channel). Across bands, `"quadriga"` uses one ray set, so the fading is correlated;
the TR methods draw coupling, XPR and phases per frequency, so it is independent, also with `"38.901_MF"`, which keeps
the delays and angles.

**Conventions.** `N = NumClusters - 1` is the number of NLOS clusters: `NumClusters` counts the LOS slot, as in
QuaDRiGa. Positions are in m; in the batch functions one row (x, y, z) per link: `rx_pos` is `(n_link, 3)`, `tx_pos` is
`(1, 3)` (one Tx for all links) or `(n_link, 3)`; heights are the z coordinates. The one-link functions take a position
as a `(3,)` array, except [gbsm.single_link_channel](#gbsm.single_link_channel), which takes one row per snapshot; scatterer positions are
`(3, n)`. Angles are elevations, not TR zenith angles (EOD, EOA, ESD for ZOD, ZOA, ZSD; also in the conf-file keys): in
rad in the GCS, azimuth in [-pi, pi), elevation in [-pi/2, pi/2] with +pi/2 up. Defaults follow TR 38.901 except two
deliberate library choices: the SOS preset `Comb256` with bias removal ([gbsm.init_phases](#gbsm.init_phases), [gbsm.draw](#gbsm.draw)) and
`method = "quadriga"` of [gbsm.rays_from_paths](#gbsm.rays_from_paths); QuaDRiGa variants are options. Units: frequencies in Hz, path loss
in dB, DS in s, KF, SF and XPR as linear power ratios, spreads and `_deg` offsets in degrees; conf-file keys and
`scenpar` keys have the units of the file comments (e.g. `PerClusterDS` in ns). Flags take a `bool` or `numpy.bool_`;
`None` selects the default, any other type raises `TypeError`. Messages name the argument `center_freq`
`center_frequency` and the real and imaginary part of `gr_eps` `gr_eps_re` and `gr_eps_im`.

---
<a id="conf_file_list"></a>
## conf_file_list
List the scenarios defined by the conf files in a set of directories

Also in: [C++](../cpp/geometry-based-stochastic-model.md#conf_file_list) · [MATLAB](../mex/geometry-based-stochastic-model.md#conf_file_list)

- Lists the regular files and symbolic links to regular files named `<scenario>.conf` (case-sensitive, non-empty
  scenario name) in each directory, not recursive: the directories in the given order, the files of one directory in
  byte-wise sorted order of their names
- A file name found in an earlier directory is skipped, so a local folder can override the files of a shared one
- With `parse_shortnames = True` every line `ShortName = name` that is not a comment adds an entry right after its file;
  other lines are not examined
- Raises `ValueError` if a short name equals an earlier entry, if a file name equals an earlier short name, or if a
  `ShortName` value is not one name; `RuntimeError` if a listed file cannot be opened
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty string `""` is the
  current directory

### Usage:
```python
scenarios, file_names, file_dirs = quadriga_lib.gbsm.conf_file_list( dirs, parse_shortnames )
```

### Inputs:
- **`dirs`** — Directories, searched in this order; one path or a list or tuple of paths, each a str,
  bytes or path-like object (e.g. `pathlib.Path`); an empty list or `None` lists nothing
- **`parse_shortnames`** — `True`: read the files and add their short names; `False`: list the files
  only, which is faster; default: True

### Outputs:
- **`scenarios`** — Scenario names and short names; list of str; length `n_scen`
- **`file_names`** — File name including `.conf` for each entry; list of str; length `n_scen`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs` (a path-like object
  converted with `os.fspath`); list of str; length `n_scen`

### See also:
- [conf_file_read](#conf_file_read) (read one of the files)

---
<a id="conf_file_read"></a>
## conf_file_read
Read and validate a QuaDRiGa scenario configuration file

Also in: [C++](../cpp/geometry-based-stochastic-model.md#conf_file_read) · [MATLAB](../mex/geometry-based-stochastic-model.md#conf_file_read)

- A conf file has one `Key = value` assignment per line; `%` starts a comment; blank lines, LF or CRLF line ends and a
  UTF-8 byte order mark at the start are accepted; keys are case-sensitive names of `A-Z`, `a-z`, `0-9` and `_`
- Three kinds of keys:
  - the numeric scenario parameters (see [scenpar_validate](#scenpar_validate)), one number each; keys that are not in the file have
    their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the QuaDRiGa key `GR_epsilon`
  (the complex ground permittivity comes from [lsp_generate](#lsp_generate)). A key stated more than once has its last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of `SubpathMethod` raises
  `ValueError`, also with `validate = False`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`), read independent of
  the locale; `Inf`, `+Inf` and `-Inf` in any case
- Raises `RuntimeError` if the file cannot be opened or is a directory
- Raises `ValueError` with file name and line number for a line that is not blank, not a comment and not a well-formed
  assignment (no `=`, not one name before the `=`, a missing value, a word where a number is required or the reverse,
  text after the value, `NaN`, a number beyond the range of a double, a `PL_` key without a name, a `ShortName` value
  stated twice), and with `validate = True` for a value that fails [scenpar_validate](#scenpar_validate)

### Usage:
```python
scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys = \
    quadriga_lib.gbsm.conf_file_read( fn, validate )
```

### Inputs:
- **`fn`** — Path to the conf file; str, bytes or path-like object (e.g. `pathlib.Path`)
- **`validate`** — `True`: apply the value checks of [scenpar_validate](#scenpar_validate); `False`: only the line
  grammar is checked; default: True

### Outputs:
- **`scenpar`** — Scenario parameters; dict with all scenario keys as floats in the canonical order of
  [scenpar_validate](#scenpar_validate)
- **`plpar`** — Path-loss parameters; dict with the entry `model` (str, only if the file states
  `PL_model`) first, then one entry per `PL_` key in file order, named without the prefix `PL_`; each
  is a 1D array with one or more values; `{}` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite; `False` is no error here, but [lsp_generate](#lsp_generate)
  raises `ValueError` for the scenario; bool
- **`short_names`** — Values of all `ShortName` lines in file order; list of str
- **`ignored_keys`** — Unknown keys, each once, in the order of their first occurrence; list of str

### See also:
- [scenpar_validate](#scenpar_validate) (value checks and defaults)
- [conf_file_list](#conf_file_list) (find the conf files in a set of directories)
- [conf_file_write](#conf_file_write) (write a conf file)

---
<a id="conf_file_write"></a>
## conf_file_write
Write a QuaDRiGa scenario configuration file

Also in: [C++](../cpp/geometry-based-stochastic-model.md#conf_file_write) · [MATLAB](../mex/geometry-based-stochastic-model.md#conf_file_write)

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files of QuaDRiGa; every
  written file passes [conf_file_read](#conf_file_read) with `validate = True`, without ignored keys and with the same values bit for
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
- All inputs are validated before the file is opened: the checks of [scenpar_validate](#scenpar_validate); `model`, the other keys of
  `plpar` and the short names are names, the short names distinct; every `plpar` key has at least one value and no
  `NaN`; `fn`, `ref` and `ref_sec` have no line break
- A failed call creates no file and leaves an existing file unchanged; the directory must be writable

### Usage:
```python
quadriga_lib.gbsm.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, \
    write_defaults )
```

### Inputs:
- **`fn`** — Path of the conf file to write; str, bytes or path-like object (e.g. `pathlib.Path`)
- **`scenpar`** — Scenario parameters; dict with any subset of the keys returned by
  [conf_file_read](#conf_file_read); real scalars (int, float, bool or a numpy value with one element); missing keys
  have their default; `None` or `{}` writes the default set
- **`plpar`** — Path-loss parameters; dict with the entry `model` (str) and one real scalar, list or
  1D array per further entry, named without the prefix `PL_`, as returned by [conf_file_read](#conf_file_read);
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

### See also:
- [conf_file_read](#conf_file_read) (read the file back)
- [scenpar_validate](#scenpar_validate) (value checks and defaults)

---
<a id="gbsm.draw"></a>
## gbsm.draw
Spatially correlated normalized random values of one random set at a set of links

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_draw) · [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_draw)

- Evaluates the SOS phases of [gbsm.init_phases](#gbsm.init_phases) at the links as [sos.val](sum-of-sinusoids-random-number-generator.md#sos.val), one column per generator. The value of a
  link depends neither on the other links of the call nor on the other columns; with `dual_mobility = False` not on
  `tx_pos`
- `dual_mobility = True`: the Tx is a second device on the same fields; its side of each generator uses the mirror
  generator (e.g. departure and arrival quantities exchange), and every column keeps its distribution
- 2D mode (`horizontal = True`, TR): the values are correlated in the x-y plane; links with equal `floor_id` share one
  field, fields of different floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6 note 3):
  exactly for IDs in the same group of `n_coeff` consecutive IDs (0 to `n_coeff - 1`, ...), otherwise up to a random
  residual of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- 3D mode (legacy QuaDRiGa): the height separates the floors, so UTs on different floors are correlated by the ACF of
  their height difference. The per-link statistics are the same in both modes
- An empty set (`NumClusters = 1`) gives `(n_link, 0)` and accepts any empty `phase`

Random sets, with `N = NumClusters - 1` and `M = NumSubPaths`; a set never changes size (new random quantities come in
new sets), and a decorrelation distance of 0 is used as 0.1 m:

| `set_type`, `set_id` | `n_gen` | Columns | Decorrelation distance | Used by |
| --- | --- | --- | --- | --- |
| `"lsf"` | 11 | 0 to 7 DS, KF, SF, ASD, ASA, ESD, ESA, XPR and 8 absolute TOA, Normal(0,1); 9 ground permittivity and 10 UMa environment height hE, Uniform(0,1) | `DS_lambda` ... `XPR_lambda` (0 to 7), `absTOA_lambda` (8), `SC_lambda` (9), `SF_lambda` (10) | [lsp_generate](#lsp_generate) `rnd`, [path_loss](#path_loss) `rnd_he` (column 10) |
| `"cluster"`, -1 | `9·N` | column `b·N + n` for block `b` and NLOS cluster `n`: block 0 delay, Uniform(0,1); blocks 1 to 8 Normal(0,1): 1 to 4 sign of AOD, AOA, EOD, EOA, 5 to 8 offset of the same | `SC_lambda` | [ssf_clusters](#ssf_clusters) `rnd` |
| `"cluster"`, `k` | `N` | cluster shadowing at frequency `k`, Normal(0,1) | `SC_lambda` | [ssf_clusters](#ssf_clusters) `rnd_freq`, slice `k` |
| `"ray"`, -1 or `k` | `14·N·M + N` | blocks 0 to 14, below | `SC_lambda`; block 14 of set -1: 1.6·c·c_DS / `PerClusterAS_A` (in rad), within 0.1 m to 10 km | [ssf_rays](#ssf_rays) `rnd_ray`: -1 for `"quadriga"`, `k` as slice `k` for the TR methods |

A `"ray"` set holds block `b` up to 12 for ray `m` of NLOS cluster `n` in column `b·N·M + n·M + m`: 0 to 3 coupling
ranks of AOD, AOA, EOD, EOA; 4 XPR; 5 to 8 initial phases θθ, θφ, φθ, φφ; 9 to 12 polarization power variability of
the same. Block 13, the XPR per cluster, is in column `13·N·M + n`; block 14, the ray delay of `"quadriga"`, in column
`13·N·M + N + n·M + m`. All blocks are Normal(0,1) except block 14, Uniform(0,1).

### Usage:
```python
rnd = quadriga_lib.gbsm.draw( scenpar, set_type, phase, tx_pos, rx_pos, set_id, preset, floor_id, \
    center_freq, dual_mobility, horizontal )
```

### Inputs:
- **`scenpar`** — Scenario parameters; decorrelation distances, `NumClusters`, `NumSubPaths`, `PerClusterDS*` and
  `PerClusterAS_A`
- **`set_type`** — Set type of [gbsm.init_phases](#gbsm.init_phases); str
- **`phase`** — SOS phases from [gbsm.init_phases](#gbsm.init_phases), finite; `(n_coeff, n_gen, 2)`
- **`tx_pos`** — Tx positions in m; used in dual mobility, always validated; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`, `n_link = 0` allowed
- **`set_id`** — Set of [gbsm.init_phases](#gbsm.init_phases): `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`;
  another set with the same `n_gen` is not detected; default: -1
- **`preset`** — Preset of [gbsm.init_phases](#gbsm.init_phases); another preset with the same `n_coeff` is not detected; default:
  `"Comb256"`
- **`floor_id`** — Floor number of the UT per link, ground floor 0 (a label; `n_fl - 1` for the UT heights of TR 38.901
  Table 7.2-1, 0 for outdoor UTs); int or float array; `None` or empty = floor 0; 2D mode only; `(n_link,)`;
  default: None
- **`center_freq`** — Carrier frequencies in Hz; required by the shared `"ray"` set, whose block 14 uses their mean
  c_DS ([ssf_rays](#ssf_rays)): pass the same list as to [ssf_rays](#ssf_rays) (another list gives other `"quadriga"` rays); ignored and
  not checked for all other sets; `(n_freq,)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move; default: False
- **`horizontal`** — `True`: 2D mode; `False`: 3D mode; default: True

### Output:
- **`rnd`** — Random values; `(n_link, n_gen)`

### See also:
- [gbsm.init_phases](#gbsm.init_phases) (the SOS phases)
- [lsp_generate](#lsp_generate) (the large-scale parameters from the `"lsf"` set)
- [path_loss](#path_loss) (column 10 of the `"lsf"` set as `rnd_he`)
- [ssf_clusters](#ssf_clusters) (the clusters from the `"cluster"` sets)
- [ssf_rays](#ssf_rays) (the rays from the `"ray"` sets)

---
<a id="gbsm.init_phases"></a>
## gbsm.init_phases
SOS phases of one random set of a BS site and scenario

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_init_phases) · [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_init_phases)

- Draws the SOS phases of one random set (set table in [gbsm.draw](#gbsm.draw)): the random phases of the sinusoids of its
  [sum-of-sinusoids (SOS)](sum-of-sinusoids-random-number-generator.md) generators, as
  [sos.init_phases](sum-of-sinusoids-random-number-generator.md#sos.init_phases); [gbsm.draw](#gbsm.draw) evaluates them at the links. Store `phase` or the seed to evaluate the same fields
  again along tracks or on maps
- The SOS phases are a fixed function of `seed`, `set_type` and `set_id`, equal on all platforms (bias-removed `"lsf"`
  sets within about 2e-14 rad); the sets are independent; each scenario of a site needs its own seed
- `"lsf"` bias removal (`remove_bias`): [sos.remove_bias](sum-of-sinusoids-random-number-generator.md#sos.remove_bias) sets the sample mean of columns 0 to 8 over the positions
  (Rx, and Tx in dual mobility; in 2D mode all on floor 0) to 0, so these means do not vary from drop to drop
- `NumClusters = 1`: all sets but `"lsf"` are empty
- TR 38.901 clause 7.4.4 literally: preset `Exp256` without bias removal

### Usage:
```python
phase = quadriga_lib.gbsm.init_phases( scenpar, set_type, seed, set_id, preset, tx_pos, rx_pos, \
    dual_mobility, remove_bias, horizontal )
```

### Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances, `NumClusters` and `NumSubPaths` are used
- **`set_type`** — `"lsf"`, `"cluster"` or `"ray"`; str
- **`seed`** — Seed of the site and scenario; int from -1 to 2^63-1; -1 = system random device, new SOS phases on each
  call; default: -1
- **`set_id`** — `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`; default: -1
- **`preset`** — Sum-of-sinusoids preset, see [sos.preset](sum-of-sinusoids-random-number-generator.md#sos.preset); sets `n_coeff`; default: `"Comb256"`
- **`tx_pos`**, **`rx_pos`** — Bias-removal positions in m as in [gbsm.draw](#gbsm.draw); `"lsf"` only, else `ValueError`;
  default: None
- **`dual_mobility`** — `"lsf"` with bias removal: Tx and Rx both move; as in [gbsm.draw](#gbsm.draw); default: False
- **`remove_bias`** — `"lsf"` only: remove the sample mean over the positions; needs `rx_pos` with at least 3 rows, else
  no effect; default: True
- **`horizontal`** — `"lsf"` with bias removal: `True` 2D, `False` 3D mode; as in [gbsm.draw](#gbsm.draw); default: True

### Output:
- **`phase`** — SOS phases in rad, in [-pi, pi), row = sinusoid, column = generator; slice 0 = Rx side, slice 1 = Tx
  side; `(n_coeff, n_gen, 2)`

### See also:
- [gbsm.draw](#gbsm.draw) (evaluates the SOS phases at the links)
- [sos.init_phases](sum-of-sinusoids-random-number-generator.md#sos.init_phases), [sos.val](sum-of-sinusoids-random-number-generator.md#sos.val) (SOS phases and evaluation of one generator), [sos.remove_bias](sum-of-sinusoids-random-number-generator.md#sos.remove_bias) (its bias removal)

---
<a id="gbsm.rays_from_paths"></a>
## gbsm.rays_from_paths
Ray data of one link from path data (positions, gains, lengths, polarization), e.g. read with qrt_file_read

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_rays_from_paths) · [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_rays_from_paths)

- Returns the ray dict of [ssf_rays](#ssf_rays) (`fallback` empty) from path data in the names and order of [qrt_file_read](channel-functions.md#qrt_file_read),
  read with `downlink = True` and `normalize_M = 1` (an uplink read gives swapped ends and a conjugated `M`, which this
  function does not undo; `normalize_M = 0` leaves only the free-space gain in `power`)
- `delay` = (path length - Tx-Rx distance) / c, `power` = path gain. Angles point from the Tx to the first of
  `fbs_pos`, `lbs_pos`, `rx_pos` that differs from it and from the Rx to the first of `lbs_pos`, `fbs_pos`, `tx_pos`;
  GR angles to the image of the other end (z mirrored); 0 if `tx_pos` = `rx_pos`
- QRT stores no path types or taps. Without `path_type`: LOS where FBS = LBS on the direct line (0.1 mm), else NLOS. A
  GR path needs `path_type`: for QRT files whose paths were built from GBSM scatterers pass the `no_int` of
  [qrt_file_read](channel-functions.md#qrt_file_read) (interaction count = path type); for ray-traced files the user sets it (values > 2 raise
  `ValueError`)
- The GR point (`lbs_pos`) must be within 1 mm of z = 0

### Usage:
```python
rays = quadriga_lib.gbsm.rays_from_paths( center_freq, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, \
    path_length, M, method, path_type, tap, gr_eps )
```

### Inputs:
- **`center_freq`** — Frequencies in Hz; `(n_freq,)`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m; `(3,)`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `(3, n_path)`
- **`path_gain`** — Path gains (linear); `(n_path, n_freq)`
- **`path_length`** — Path lengths in m; `(n_path,)`
- **`M`** — Polarization transfer matrices, real, rows as in [get_channels_planar](channel-generation-functions.md#get_channels_planar); `(8, n_path, n_freq)`
- **`method`** — Value of `rays["method"]`; sets the GR power law and the default mobility; default: `"quadriga"`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS, at most one LOS and one GR; `(n_path,)`; default: None
- **`tap`** — Tap index; `None`: LOS 0, GR 3, the others 6, 9, ... in path order (3 is kept for GR also without a GR
  path); `(n_path,)`; default: None
- **`gr_eps`** — Ground permittivity (complex), used as `real - j·imag`; needed only where a GR ray is recomputed, i.e.
  in the mobility `drift` at moved ends; `(n_freq,)`; default: None (also if empty)

### Output:
- **`rays`** — Ray dict with `n_ray = n_path`, see [ssf_rays](#ssf_rays); use with `link_gain_dB = 0` and without `kf_ratio`

### See also:
- [qrt_file_read](channel-functions.md#qrt_file_read) (source of the path data, `downlink = True`)
- [ssf_paths](#ssf_paths) (paths at moved positions)
- [gbsm.single_link_channel](#gbsm.single_link_channel) (channel of the link)

---
<a id="gbsm.single_link_channel"></a>
## gbsm.single_link_channel
Channel coefficients of one link at one frequency over one segment

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_single_link_channel) · [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_single_link_channel)

- Computes the MIMO coefficients of one link with its antennas over the `n_snap` snapshots of one segment at
  `rays["center_frequency"][i_freq]`: per snapshot, the paths of [ssf_paths](#ssf_paths) at the snapshot positions with the link
  gain and K-factor ratio of the snapshot, for the kept rays (see Terms)
- `"phase"`: [get_channels_planar](channel-generation-functions.md#get_channels_planar) with the arrays at the positions of `rays` and the orientations of the snapshot;
  delays, path lengths and angles stay those of the ray-data positions. `"drift"`: [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) with the
  arrays at the positions and orientations of the snapshot
- `"taps"`: the rays summed per tap index, taps in ascending index, one tap per distinct index of the kept rays (fixed
  per call); LOS and GR taps are present whenever `rays` holds them, also at zero power. The delay of a tap per
  element pair is the mean of the rays' delays weighted by `rays["power"]` at `i_freq` (equal weights where they sum
  to 0)
- `"taps"` is exact at the phase center for the TR methods (the rays of a tap share one delay); for `"quadriga"` only
  at the carrier: a tap sums rays of different delays and is frequency-selective within the band, so use `"rays"` for
  wide bands. `"rays"`: one path per kept ray in ray order, exact at any bandwidth; the path data are those of
  `ssf_paths` at the snapshot for these rays
- `n_snap` is the largest row count of the positions and orientations (1 if all are omitted); each has 1 or `n_snap`
  rows, `link_gain_dB` and `kf_ratio` 1 or `n_snap` entries (per snapshot at `i_freq`, while `ssf_paths` takes one
  per frequency); one row or entry applies to all snapshots. A static link with a gain per snapshot repeats its
  position
- Powers are not renormalized, and `initial_position` of `chan` stays 0. Validity limits: see `ssf_paths`
- Raises `ValueError` for an invalid array (message prefixed with `Transmit antenna: ` or `Receive antenna: `), invalid
  `rays`, `"drift"` without scatterers or with a GR ray at moved ends without permittivity or below z = 0, and a row or
  entry count other than 1 or `n_snap`

### Usage:
```python
chan, par = quadriga_lib.gbsm.single_link_channel( tx_array, rx_array, rays, link_gain_dB, mobility, \
    output, i_freq, tx_pos, rx_pos, tx_orientation, rx_orientation, kf_ratio, oxygen, \
    use_absolute_delays, stack )
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Dicts containing the arrayant data with `n_tx` and `n_rx` ports; layout as
  documented in [generate](array-antenna-functions.md#generate)
- **`rays`** — Ray dict of the link, see [ssf_rays](#ssf_rays)
- **`link_gain_dB`** — Link gain (see Terms) at frequency `i_freq` of `rays` in dB; float, `(1,)` or `(n_snap,)`
- **`mobility`** — `"phase"`, `"drift"` or `"auto"` (see Terms; `"auto"` is `"drift"` for `"quadriga"` rays, which then
  need [ssf_scatterers](#ssf_scatterers)); default: `"auto"`
- **`output`** — `"taps"` or `"rays"`; default: `"taps"`
- **`i_freq`** — Frequency index in `rays["center_frequency"]`, 0-based; ray data of the TR methods hold one frequency:
  0; default: 0
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `None`: the positions of `rays`; `(n_snap, 3)` or `(1, 3)`
  (a `(3,)` array is rejected); default: None
- **`tx_orientation`**, **`rx_orientation`** — Euler angles (bank, tilt, heading) in rad; `None`: 0; `(n_snap, 3)`
  or `(1, 3)`; default: None
- **`kf_ratio`** — KF(snapshot) / KF(drop) at `i_freq`, >= 0; `None`: 1; `(n_snap,)` or `(1,)`; default: None
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1, see `ssf_paths`; default: `False`
- **`use_absolute_delays`** — Include the LOS delay in all delays (`"phase"`: of the ray-data positions); default:
  `False`
- **`stack`** — Snapshot layout of `chan` as in [hdf5_read_channel](channel-functions.md#hdf5_read_channel); default: `False`

### Outputs:
- **`chan`** — Channel dict, layout as documented in [hdf5_read_channel](channel-functions.md#hdf5_read_channel): `center_frequency` `(1,)`;
  `tx_position`, `rx_position` (the snapshot positions), `tx_orientation`, `rx_orientation` `(3, n_snap)`; per
  snapshot `coeff`, `delay` `(n_rx, n_tx, n_tap)` (`"taps"`) or `(n_rx, n_tx, n_path)` (`"rays"`); `"rays"` also
  `path_gain`, `path_length`, `path_polarization`, `path_angles` and in `"drift"` `fbs_pos`, `lbs_pos`
- **`par`** — Dict with the keys below

### Keys of par:

| Name | Size | Content |
| --- | --- | --- |
| `tap` | `(n_tap,)` or `(n_path,)` | Tap index per output tap or per path; uint32 |
| `n_ray_tap` | `(n_tap,)` | Number of kept rays per tap; uint32; `"taps"` only |
| `fbs_pos`, `lbs_pos` | `(3, n_kept)` | Scatterers of the kept rays at the ray-data positions; `"drift"` with `"taps"` only; float32, the precision of path data in HDF5 files |
| `gr_pos` | `(3, n_snap)` | GR point per snapshot; `"drift"` with `"taps"` and a GR ray only; float32 |
| `link_gain_dB` | `(n_snap,)` | Link gain per snapshot |
| `b_med`, `b_eff` | `(2,)` | Of [ssf_scatterers](#ssf_scatterers) at `i_freq`, Tx and Rx; without scatterers from a solved copy (`d_min` = 2 m) |
| `limit` | `(2,)` | Validity limit per end in m (see [ssf_paths](#ssf_paths)): `sqrt(λ·b_eff/8)` in `"phase"`, `0.2·b_med` in `"drift"`; Inf where `b_eff` or `b_med` is NaN |
| `ratio` | `(2,)` | Largest distance of the end from its ray-data position over the snapshots, divided by `limit` |
| `ratio_total` | scalar | `sqrt(ratio[0]² + ratio[1]²)`; valid up to 1 |

### See also:
- [ssf_rays](#ssf_rays) (the ray dict)
- [ssf_paths](#ssf_paths) (paths of one link at one position, validity limits)
- [ssf_scatterers](#ssf_scatterers) (scatterers for `"drift"`, `b_med` and `b_eff`)
- [gbsm.rays_from_paths](#gbsm.rays_from_paths) (ray data from path data)

---
<a id="los_probability"></a>
## los_probability
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

Also in: [C++](../cpp/geometry-based-stochastic-model.md#los_probability) · [MATLAB](../mex/geometry-based-stochastic-model.md#los_probability)

- Evaluates the probability that a link is in line-of-sight (LOS) from its 2D distance and the BS and UT heights, and,
  when `rnd_g` and `center_freq` are given, the soft LOS state of TR 38.901 eq. (7.6-18); the LOS state itself (a draw
  against `p_los`) and its spatial correlation are the caller's
- `38.901_*`: TR 38.901 v19.5.0 Table 7.4.2-1 (`InH_Mixed`, `InH_Open`: mixed and open office). Where the table leaves
  cases open: all results are limited to [0, 1]; UMa uses `C' = 1` for `hUT > 23` m; InF-SH and InF-DH give 1 for
  `hUT >= h_clutter` and the rule of InF-SL and InF-DL for `hBS <= hUT`; SMa gives 1 for `hBS <= hUT` and uses
  `max(h_x - hUT, 0)` in all three terms (the TR only in the residential term; equal for `hUT < 15` m)
- `38.811_*`: TR 38.811 Table 6.6.1-1 over the elevation `atan2(hBS, d2d_out)`, linearly interpolated; `Suburban` and
  `Rural` share one table
- `QuaDRiGa_Industrial`: `min(1, exp(-(d2d_out + 20)/80)·exp(min(hBS, 8)/5 - 1))`; `LOS`: 1; `NLOS`: 0
- The `LOS_state_lambda` of the InF conf files is `d_clutter/2` of the TR defaults; with another `d_clutter` draw the
  LOS state at a matching decorrelation distance

### Usage:
```python
p_los, los_soft = quadriga_lib.gbsm.los_probability( model, d2d_out, h_bs, h_ut, d_clutter, r_clutter, \
    h_clutter, rnd_g, center_freq )
```

### Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`, `38.901_InH_Open`,
  `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`, `38.901_InF_HH`, `38.901_SMa`,
  `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in m, finite and >= 0: `d2D - d2d_in` for O2I UTs, clamped at 0 by the caller; the
  2D distance in indoor scenarios; `(n_link,)`
- **`h_bs`** — BS height in m, finite; float or `(n_link,)`
- **`h_ut`** — UT height in m, finite; float or `(n_link,)`
- **`d_clutter`** — Clutter size in m of the InF and SMa models, > 0; `None` or nan = TR default (InF-SL and InF-SH 10
  m, InF-DL and InF-DH 2 m, SMa 30 m); default: None
- **`r_clutter`** — Clutter density of the InF models (default 0.2 for SL and SH, 0.6 for DL and DH), vegetation density
  of SMa (default 0; 0.1 sparse, 0.2 dense); in [0, 1); `None` or nan = default; default: None
- **`h_clutter`** — Clutter height in m of InF-SH (default 2 m) and InF-DH (6 m), >= 0; `None` or nan = default. The
  other models ignore the clutter arguments, which are validated for every model; default: None
- **`rnd_g`** — Normal(0,1) value per link, G of eq. (7.6-18); `(n_link,)`; default: None
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`; default: None

### Outputs:
- **`p_los`** — LOS probability in [0, 1]; `(n_link,)`
- **`los_soft`** — Soft LOS state in [0, 1]; `> 0.5` exactly when `rnd_g > -F(p_los)` of eq. (7.6-18), which has the
  probability `p_los`; `(n_link, n_freq)`; `None` if `rnd_g` or `center_freq` is `None`

### See also:
- [o2i_loss](#o2i_loss) (indoor distance for `d2d_out`)
- [sos.rand](sum-of-sinusoids-random-number-generator.md#sos.rand), [sos.randn](sum-of-sinusoids-random-number-generator.md#sos.randn) (spatially consistent LOS-state draw and `rnd_g`)

---
<a id="lsp_generate"></a>
## lsp_generate
Large-scale parameters of a set of links from normalized random values

Also in: [C++](../cpp/geometry-based-stochastic-model.md#lsp_generate) · [MATLAB](../mex/geometry-based-stochastic-model.md#lsp_generate)

- Generates the 8 LSPs of TR 38.901 clause 7.5 step 4 per link and frequency from the random values of [gbsm.draw](#gbsm.draw)
  (`"lsf"`), and the absolute TOA offset, the ground permittivity and the EoD offset
- Mean `mu` and standard deviation `sigma` in the log domain (lg for DS and the angular spreads, dB for KF, SF and XPR),
  with the key prefix `P` (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), `d = max(d2D, 1e-4)`, `h = max(hBS,
  1e-4)`, `a = atan(h/d)` and `f' = max(center_freq/1e9, LSP_fc_min)` in GHz:
  - `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`
  - `sigma = max(0, max(0, P_sigma + P_delta·lg(P_omega + f')) + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))`
  - SF has the mean 0. DS adds `lg(DS_VS_a·DS_VS + DS_VS_b)` if `DS_VS > 0` (V/S of the InF hall, TR 38.901 Table 7.5-6
    Part-3). ESD adds `ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)` and is
    limited below by `ES_D_mu_min` (TR 38.901 Tables 7.5-7 to 7.5-12)
  - The multiplying coefficients must be finite; a frequency term with `P_gamma` or `P_delta` not 0 requires `P_omega +
    f' > 0`
- `lsp = 10^(mu + sigma·x)` for DS and the angular spreads, `10^(0.1·(mu + sigma·x))` for KF, SF and XPR; `x` are the
  columns 0 to 7 of `rnd`, cross-correlated by a root of the LSP cross-correlation matrix of `scenpar`, the same for all
  frequencies (TR 38.901 clause 7.6.5). `rnd = numpy.zeros((n_link, 11))` gives the median LSPs
- Absolute TOA offset (TR 38.901 clause 7.6.9), active if `absTOA_mu > -30`: `10^(absTOA_mu + absTOA_sigma·x)` with
  column 8 of `rnd`, bounded by `2·absTOA_Lmax/c` if `absTOA_Lmax > 0`
- The InF and InH NLOS files set `absTOA_Lmax`, and the InF files `DS_VS`, for the hall or office of their calibration
  scenario; for another hall set these keys in `scenpar`
- Ground permittivity: QuaDRiGa's fits for dry, medium and wet ground, interpolated linearly over column 9 of `rnd` (0
  dry, 0.5 medium, 1 wet) with `center_freq` not limited by `LSP_fc_min`; they are not the materials of TR 38.901 Table
  7.6.8-1
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e + EoD_offset_e_gamma·lg
  f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) + EoD_offset_c +
  EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if `EoD_offset_b > 0`, plus `(atan((EoD_offset_hcl - h)/d) -
  atan((hUT - h)/d))·180/pi` if `hUT < EoD_offset_hcl`
- `complex = True` returns the permittivity as one complex array `gr_eps = gr_eps_re + 1j·gr_eps_im`; `complex = False`
  returns the two real arrays `gr_eps_re`, `gr_eps_im` in its place (one more output)

### Usage:
```python
# complex=True (default)
lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( scenpar, \
    center_freq, tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads, complex=True )

# complex=False
lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, \
    lsp_sigma = quadriga_lib.gbsm.lsp_generate( ..., complex=False )
```

### Inputs:
- **`scenpar`** — Scenario parameters; the LSP cross-correlation matrix must be positive definite
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`tx_pos`** — Tx positions in m; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`rnd`** — Random values from [gbsm.draw](#gbsm.draw) (`"lsf"`), finite; column 9 in [0, 1] (the permittivity is always
  computed); column 10 is not used; `(n_link, 11)`
- **`sf_sigma`** — SF standard deviation in dB (e.g. from [path_loss](#path_loss)), finite and >= 0; replaces the value from
  `scenpar`, also in `lsp_sigma`; `None` or empty = the scenario value; `(n_link, n_freq)`; default: None
- **`cholesky`** — `True`: Cholesky root in the order of TR 38.901 clause 7.5 step 4 (SF, KF, DS, ASD, ASA, ESD, ESA;
  XPR last); `False`: symmetric square root (legacy QuaDRiGa); default: True
- **`clip_spreads`** — `True`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg (TR 38.901 clause 7.5 step 4);
  `lsp_mu` and `lsp_sigma` are not affected; default: True
- **`complex`** — See the description; default: True

### Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in s, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA in deg, 7 XPR; `(n_link,
  n_freq, 8)`
- **`abs_toa`** — Absolute TOA offset in s, 0 if not active; `(n_link,)`
- **`gr_eps`** — Relative permittivity of the ground (`complex=True`), imaginary part >= 0; the complex value of TR
  38.901 eq. (7.6-40) is `gr_eps.conj()`; complex128; `(n_link, n_freq)`
- **`gr_eps_re`**, **`gr_eps_im`** — Real and imaginary part instead of `gr_eps` (`complex=False`); `(n_link, n_freq)`
- **`eod_offset_deg`** — EoD offset in deg, positive = up; `(n_link, n_freq)`
- **`lsp_mu`** — Mean `mu` in the log domain; required by [ssf_rays](#ssf_rays); `(n_link, n_freq, 8)`
- **`lsp_sigma`** — Standard deviation `sigma` in the log domain, >= 0; required by [ssf_rays](#ssf_rays); `(n_link, n_freq, 8)`

### See also:
- [gbsm.draw](#gbsm.draw) (the random values `rnd`)
- [path_loss](#path_loss) (the SF standard deviation `sf_sigma`)
- [ssf_clusters](#ssf_clusters) (the clusters from the LSPs)
- [ssf_rays](#ssf_rays) (takes `lsp`, `lsp_mu` and `lsp_sigma`)

---
<a id="o2i_loss"></a>
## o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

Also in: [C++](../cpp/geometry-based-stochastic-model.md#o2i_loss) · [MATLAB](../mex/geometry-based-stochastic-model.md#o2i_loss)

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier frequency; the
  caller adds `loss` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table 7.4.3-1), `legacy_6GHz`
  (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz), `car` and `car_metal` (clause 7.4.3.2, metallized windows)
- Model of UT `u`: the first model `j` with `rnd(u, 0)` below the sum of the weights up to `j`, else the last model with
  a weight > 0; a model with weight 0 is never chosen
- Indoor distance: the minimum of columns 1 and 2 of `rnd` times `d2d_in_max` (`low`, `high`, `low_A`, clause 7.4.3.1),
  column 1 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [sos.rand](sum-of-sinusoids-random-number-generator.md#sos.rand) and [sos.randn](sum-of-sinusoids-random-number-generator.md#sos.randn) at the correlation distances of TR 38.901 clause 7.6.3.3: 50 m for
  column 0, 25 m for columns 1 and 2, 10 m for column 3
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`, SMa `low`, `high` and
  `low_A`. SMa has 90 % residential and 10 % commercial buildings (Table 7.2-5) with different maximum indoor distances:
  call `o2i_loss` once per building type

### Usage:
```python
loss, d2d_in, model_index = quadriga_lib.gbsm.o2i_loss( center_freq, rnd, models, weights, d2d_in_max )
```

### Inputs:
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`rnd`** — Random values per UT: columns 0 to 2 Uniform in [0, 1] (model choice, two indoor distances), column 3
  Normal(0,1), finite (deviation `sigma_P`, car loss); `(n_ut, 4)`, `n_ut = 0` allowed
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; list of str or one str;
  `n_model` entries
- **`weights`** — Probability of each model, finite and >= 0, sum 1 within 1e-6; `(n_model,)`
- **`d2d_in_max`** — Maximum indoor 2D distance in m, finite and >= 0 (`scenpar["O2I_d2D_in_max"]` of the O2I conf
  files; TR: 25 m for UMa, UMi and SMa commercial buildings, 10 m for RMa and SMa residential buildings); default: 25.0

### Outputs:
- **`loss`** — Penetration loss in dB; `(n_ut, n_freq)`
- **`d2d_in`** — Indoor 2D distance in m, 0 for the car models; `(n_ut,)`
- **`model_index`** — Chosen model per UT, 0-based index into `models`; uint64; `(n_ut,)`

### See also:
- [los_probability](#los_probability) (uses the outdoor distance `d2D - d2d_in`)
- [path_loss](#path_loss) (basic path loss, legacy outdoor distance from `d2d_in`)

---
<a id="path_loss"></a>
## path_loss
Path loss and shadow-fading standard deviation of a set of links

Also in: [C++](../cpp/geometry-based-stochastic-model.md#path_loss) · [MATLAB](../mex/geometry-based-stochastic-model.md#path_loss)

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and carrier frequency;
  the TR 38.901 v19.5.0 conf files of the library reproduce Table 7.4.1-1 with them. The O2I penetration loss is not
  included (see [o2i_loss](#o2i_loss)). The models are evaluated as written for any positive frequency, also outside their
  validity range
- Notation: `f = center_freq/1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance, at least 0.1
  m; `hBS`, `hUT` the z coordinates of Tx and Rx; `lg = log10`; `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`; `sf` the
  shadow-fading std. Keys of `plpar` in parentheses: the required ones, after the semicolon the optional ones (`X = v`:
  default)
- Models (the key `model` of `plpar`):
  - `constant` (A): `A`
  - `logdist` (A, B, C; SF): `A·lg d3 + B + C·lg f`; `sf = SF`
  - `logdist_simple` (A, B): `A·lg d3 + B`
  - `winner_los` (A1, A2 different, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb = max(hBS, 1.5)`, `hm =
    max(hUT, 1.5)`, `Gj = Bj + Cj·lg f + Dj·lg hb + Ej·lg hm`, breakpoint `bp = 10^((G2 - G1)/(A1 - A2))`; `Aj·lg d2 +
    Gj + Fj·hm` and `sf = sigj` with `j = 1` for `d2 <= bp`, else `j = 2`
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as `winner_los`; `(Aj + Ahj·lg
    hb)·lg d2 + Bj + Cj·lg f + Dj·lg hb + Ej·lg hm + Fj·hm` with `j = 1` for `f < 1.5`, 2 for `f < 2`, else 3
  - `winner_pathloss` (A, B, C, X; SF): `A·lg d3 + B + C·lg(f/5) + X`; `sf = SF`
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2): `hb = max(hBS, 1.1·hE, 0.1)`,
    `hm = max(hUT, 1.1·hE, 0.1)`, `dBP = E·f·(hb - hE)·(hm - hE)`, `dBP3 = sqrt(dBP² + (hb - hm)²)`; `PL1(d3)`, `sf =
    sig1` for `d2 <= dBP`, else `PL1(dBP3) + A2·lg(d3/dBP3)`, `sf = sig2`
  - `tripple_slope`, spelled as in QuaDRiGa (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0,
    hE2 = 0, sig1..sig3): the `dual_slope` rule with the breakpoints `dBP1` of `E1, hE1` (at most `dBP2`) and `dBP2`
    of `E2, hE2`, the slopes `A1`, `A2`, `A3` and `sf = sig1`, `sig2`, `sig3`
  - `nlos` (An, Bn; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, Ax, Bx, Cx, sig): `hb =
    max(hBS, 0.1)`, `hm = max(hUT, 0.1)`; `An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb + D2n·lg(hb)/hb² +
    D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; with
    `LOSmax != 0` (which requires the `dual_slope` keys) at least the `dual_slope` loss; with `Ax`, `Bx` or `Cx` at
    least `Ax·lg d3 + Bx + Cx·lg f` (e.g. the InF-SL term of InF-DL); `sf = sig`
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS/d3, 1)`; `+Inf` for `sa <= 0`, else `A·lg
    d3 + B + C·lg f + D·lg(asin(sa))`, plus the zenith attenuation by atmospheric gases over `sa` (1 to 100 GHz,
    constant beyond) if `usePLa != 0`
  - no model (`plpar` is `None`, `{}` or has no entry `model`): 0
- Where the model gives no `sf` (an `SF` or `sig` key not set, or a model without one): the SF std of [lsp_generate](#lsp_generate)
  at the geometric distance, so that passing `sf_sigma` to [lsp_generate](#lsp_generate) is neutral
- `hE_rand != 0` with `rnd_he` given: the random environment height of TR 38.901 Table 7.4.1-1 note 1, drawn from
  `rnd_he`, which makes it spatially consistent at the decorrelation distance `SF_lambda` (the TR draws it independently
  per link); the UT height is then clamped at `hE + 0.1` instead of `1.1·hE`. Along a track the loss beyond the
  breakpoint steps where hE changes; evaluate `rnd_he` once per track segment to avoid this
- Distances: the geometric distances of TR 38.901 eq. (7.4-1). With `d2d_in` (legacy QuaDRiGa), links with `d2d_in > 0`
  use the outdoor part of the distances instead, at least 10 m
- A required key that is missing is an error, naming `A1` for "A1 or A" and `E1` for "E1 or E"; every key that the model
  reads must have one finite value; the breakpoint factors `E`, `E1`, `E2` and `G2n` must be > 0

### Usage:
```python
pl, sf_sigma = quadriga_lib.gbsm.path_loss( plpar, scenpar, center_freq, tx_pos, rx_pos, d2d_in, rnd_he )
```

### Inputs:
- **`plpar`** — Path-loss parameters; dict as returned by [conf_file_read](#conf_file_read): the entry `model` (str) and one real
  scalar or 1D array per further parameter, named without "PL_"; keys not read are ignored; `None` or `{}`: no model
  (loss 0)
- **`scenpar`** — Scenario parameters; the SF std terms and `LSP_fc_min` are used
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`tx_pos`** — Tx positions in m; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`d2d_in`** — Indoor 2D distance in m of the UT of each link, legacy QuaDRiGa only (leave it `None` for TR 38.901,
  which uses the geometric distance); finite and >= 0; 0 = outdoor link; `(n_link,)`; default: None
- **`rnd_he`** — Uniform(0,1) value per link for the random environment height, column 10 of [gbsm.draw](#gbsm.draw) (`"lsf"`);
  `None` or empty = fixed `hE`; `(n_link,)`; default: None

### Outputs:
- **`pl`** — Path loss in dB; `(n_link, n_freq)`
- **`sf_sigma`** — Shadow-fading standard deviation in dB; `(n_link, n_freq)`

### See also:
- [lsp_generate](#lsp_generate) (takes `sf_sigma`)
- [o2i_loss](#o2i_loss) (penetration loss, added by the caller)

---
<a id="scenpar_validate"></a>
## scenpar_validate
Validate scenario parameters and complete them with default values

Also in: [C++](../cpp/geometry-based-stochastic-model.md#scenpar_validate) · [MATLAB](../mex/geometry-based-stochastic-model.md#scenpar_validate)

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

### Usage:
```python
scenpar_out, lsp_xcorr, lsp_xcorr_chk = quadriga_lib.gbsm.scenpar_validate( scenpar_in, validate )
```

### Inputs:
- **`scenpar_in`** — Scenario parameters; dict with any subset of the keys returned by
  [conf_file_read](#conf_file_read); real scalars (int, float, bool or a numpy value with one element); `None` or
  `{}` returns the default set; default: None
- **`validate`** — `False`: only complete with defaults, no value checks (`NaN` is accepted); default: True

### Outputs:
- **`scenpar_out`** — Complete scenario parameters; given values are kept, missing keys have their
  default; dict with all keys as floats in the canonical order, QuaDRiGa's keys first
- **`lsp_xcorr`** — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite; `False` also if it contains `NaN`; `False` is no
  error here, but [lsp_generate](#lsp_generate) raises `ValueError` for such parameters; bool

### See also:
- [conf_file_read](#conf_file_read) (read the parameters from a conf file)
- [conf_file_write](#conf_file_write) (applies the same checks before writing)

---
<a id="ssf_clusters"></a>
## ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

Also in: [C++](../cpp/geometry-based-stochastic-model.md#ssf_clusters) · [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_clusters)

- Turns the LSPs of [lsp_generate](#lsp_generate) and the cluster random values of [gbsm.draw](#gbsm.draw) into the clusters of each link, by
  one of three methods:
  - `"38.901"` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters
    25 dB below the strongest get power 0 (no renormalization); the K-factor in the scaling factors is
    limited to >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` at the drop
    or segment start and keep them along the track
  - `"38.901_MF"`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `"quadriga"`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links

| Property | `"38.901"` | `"38.901_MF"` | `"quadriga"` |
| --- | --- | --- | --- |
| Continuous along tracks | no | NLOS only | yes |
| Same delays and angles at all frequencies | n/a | NLOS, or K the same at all frequencies | yes |
| Output spreads equal the input spreads | no | no | DS, and AS below the caps, at one frequency without ground reflection |
| Per-cluster shadowing | yes | yes, per frequency | no |

- `n_slot = NumClusters + G` slots per link (see Terms), `G = 1` if `GR_enabled > 0`. Slot `1 + G + n` belongs to
  column `n` of each block of `rnd` (no sorting). Clusters `n >= n_cluster[l]` (TR clause 7.6.15) get delay 0, power 0
  and the LOS angles. Identify the LOS path by its slot: NLOS clusters of the TR methods can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever `lsp`
  holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in s relative to the LOS delay `d3D/c`; slot 0 holds the geometric LOS direction. Powers are the coefficient
  powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `"38.901"`, `"38.901_MF"` add `abs_toa[l]` to the NLOS delays;
  `"quadriga"` places the first NLOS cluster at `abs_toa[l]` (where `absTOA_mu <= -30`: only where `abs_toa[l] > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0: slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`"quadriga"`), with `R²` the mean of `abs(R_par)²` and `abs(R_perp)²` of TR eqs.
  (7.6-38), (7.6-39); the coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarization
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios, plus the ground reflection if enabled; `rnd` must be
  empty, `rnd_freq` and `lsp_anchor` are not used
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario; only their sizes are checked

### Usage:
```python
delay, power, aod, eod, aoa, eoa, n_active = quadriga_lib.gbsm.ssf_clusters( scenpar, tx_pos, rx_pos, \
    lsp, rnd, method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps, o2i, n_cluster )
```

### Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`, `KF_mu`, `absTOA_mu`, `r_DS`, `LNS_ksi`,
  `GR_enabled`
- **`tx_pos`** — Tx positions in m; z >= 0 with ground reflection; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; z >= 0 with ground reflection; `(n_link, 3)`
- **`lsp`** — LSPs from [lsp_generate](#lsp_generate); DS, ASD, ASA, ESD, ESA (slices 0, 3 to 6) > 0, KF (slice 1) >= 0;
  `"38.901"`: one frequency, call once per frequency; `(n_link, n_freq, 8)`
- **`rnd`** — Shared cluster set of [gbsm.draw](#gbsm.draw) (`"cluster"`, `set_id = -1`), columns 0 to N-1 in [0, 1];
  `(n_link, 9·N)`
- **`method`** — `"38.901"`, `"38.901_MF"` or `"quadriga"`; default: `"38.901"`
- **`rnd_freq`** — Per-frequency cluster sets of [gbsm.draw](#gbsm.draw), slice `k` from `set_id = k` (`"38.901"`: one slice,
  the set of the frequency in `lsp`, also as `(n_link, N)`); required by the TR methods; `(n_link, N, n_freq)`;
  default: None
- **`lsp_anchor`** — LSPs at the anchor frequency (e.g. 2 GHz) from [lsp_generate](#lsp_generate); required by `"38.901_MF"`;
  `(n_link, 1, 8)`; default: None
- **`eod_offset_deg`** — EoD offset in deg from [lsp_generate](#lsp_generate); used by `"38.901"` and `"quadriga"` in NLOS
  scenarios; `None` = 0; `(n_link, n_freq)`; default: None
- **`abs_toa`** — Absolute TOA offset in s from [lsp_generate](#lsp_generate), >= 0; required by `"quadriga"` in NLOS scenarios
  with `absTOA_mu > -30`; `None` = no offset; `(n_link,)`; default: None
- **`gr_eps`** — Ground permittivity as [lsp_generate](#lsp_generate) returns it (complex, real part >= 1, imaginary part >= 0,
  used as `real - j·imag`; a real array = imaginary part 0); required if `GR_enabled > 0`; `(n_link, n_freq)`;
  default: None
- **`o2i`** — O2I flag per link (mean arrival elevation 0), bool or numeric array (nonzero = O2I); used by `"38.901"`
  in NLOS scenarios; `None` = none; `(n_link,)`; default: None
- **`n_cluster`** — Active NLOS clusters per link in `[min(max(1, NumClusters_min - 1), N), N]`, int or float
  array of non-negative integers; `None` = N; `(n_link,)`; default: None

### Outputs:
- **`delay`** — Delays in s; `(n_link, n_slot, n_freq)` for `"38.901_MF"`, else `(n_link, n_slot, 1)`
- **`power`** — Coefficient powers (linear); `(n_link, n_slot, n_freq)`
- **`aod`**, **`eod`** — Azimuth in [-pi, pi) and elevation in [-pi/2, pi/2] of departure in rad; shapes as `delay`
- **`aoa`**, **`eoa`** — Azimuth and elevation of arrival in rad, ranges as departure; shapes as `delay`
- **`n_active`** — Number of NLOS clusters with power > 0; uint32; `(n_link,)`

### See also:
- [gbsm.draw](#gbsm.draw) (the random values `rnd` and `rnd_freq`)
- [lsp_generate](#lsp_generate) (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [ssf_rays](#ssf_rays) (the rays of the clusters)
- [calc_delay_spread](channel-statistics.md#calc_delay_spread), [calc_angular_spread](channel-statistics.md#calc_angular_spread) (spreads of the output)

---
<a id="ssf_paths"></a>
## ssf_paths
Antenna-free paths of one link at one position

Also in: [C++](../cpp/geometry-based-stochastic-model.md#ssf_paths) · [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_paths)

- Computes the antenna-free paths (gains, lengths, polarization matrices, angles and scatterers) of the kept rays of
  `rays` (see Terms) at new Tx and Rx positions (default: the ray-data positions), LOS and GR also at zero power; paths
  in ray order, `ray_ind` maps them to the rays (e.g. `rays["tap"][ray_ind]`)
- `"phase"`: use the outputs with [get_channels_planar](channel-generation-functions.md#get_channels_planar) and the arrays at the positions of `rays`; the displacement
  phase of TR 38.901 eq. (7.5-22) is in `M`. With the arrays at the new positions the LOS path is no longer identified
  and the displacement phase counts twice, without an error
- `"drift"`: use the outputs at the new positions with [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) or [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq); needs
  the scatterers of [ssf_scatterers](#ssf_scatterers)

| Quantity | `"phase"` | `"drift"` |
| --- | --- | --- |
| Path length | Tx-Rx distance of `rays` + c·delay | Polyline Tx, FBS, LBS, Rx; LOS direct line; GR via the specular point |
| Angles | Ray data | Geometry: towards FBS and LBS, LOS direct line, GR towards the image points |
| `M` | Ray `M` times the displacement phase | Ray `M`; GR recomputed when an end moved |
| `fbs_pos`, `lbs_pos` | `(3, 0)` | Scatterers of the rays; LOS midpoint of the direct line; GR specular point |

- `"drift"` with an end moved (any coordinate differs from `rays`): GR power by the power law of the method of `rays`
  (`(d3D / d_GR)²` for the TR methods, 1 for `"quadriga"`), GR `M` the scalar multiple of `diag(R_par, -R_perp)` (TR
  38.901 eqs. (7.6-38), (7.6-39)) that best fits the stored matrix, which replaces a matrix of another form (e.g. a
  conjugated one). Fallback rays have their longer polyline, also at the positions of `rays`
- Path gain per frequency: the coefficient power; with `kf_ratio` LOS and GR are rescaled against NLOS so that the
  K-factor of the coefficient powers changes by `kf_ratio`, LOS + NLOS keep their sum; times `10^(link_gain_dB / 10)`;
  with `oxygen` times the absorption of TR 38.901 clause 7.6.1 (Table 7.6.1-1 at each frequency) over each path's own
  length, without the delay offset τ_Δ of eq. (7.6-1) and at the carrier only (no in-band variation of eq. (7.6-2))
- Validity per moving end, with `b_eff` and `b_med` of [ssf_scatterers](#ssf_scatterers): `"phase"` up to about `sqrt(λ · b_eff / 8)`,
  typically 0.1 to 1 m; `"drift"` up to about `0.2 · b_med`, typically 1.5 to 12 m, new segment from `0.5 · b_med`; in
  `"drift"` each path length changes by at most the displacement. [gbsm.single_link_channel](#gbsm.single_link_channel) reports the
  displacement relative to the limit
- Raises `ValueError` for invalid input; in `"drift"` also without scatterers, and for a GR ray at moved ends without
  permittivity in `rays` or with an end below z = 0

### Usage:
```python
path_gain, path_length, M, aod, eod, aoa, eoa, fbs_pos, lbs_pos, path_type, ray_ind = \
    quadriga_lib.gbsm.ssf_paths( rays, link_gain_dB, mobility, tx_pos, rx_pos, kf_ratio, oxygen )
```

### Inputs:
- **`rays`** — Ray dict of the link, see [ssf_rays](#ssf_rays)
- **`link_gain_dB`** — Link gain (see Terms) at the position in dB, per frequency; 0 for data from paths; `(n_freq,)`
- **`mobility`** — `"phase"`, `"drift"` or `"auto"` (see Terms; `"auto"` is `"drift"` for `"quadriga"` rays, which then
  need [ssf_scatterers](#ssf_scatterers)); default: `"auto"`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m; `None`: the positions of `rays`; `(3,)`; default: None
- **`kf_ratio`** — KF(position) / KF(drop), >= 0, per frequency; `None`: 1; `(n_freq,)`; default: None
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `False`

### Outputs:
- **`path_gain`** — Path gains (linear); `(n_path, n_freq)`
- **`path_length`** — Path lengths in m; `(n_path,)`
- **`M`** — Polarization transfer matrices, real, rows as in [get_channels_planar](channel-generation-functions.md#get_channels_planar); `(8, n_path, n_freq)`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Departure and arrival angles in rad, GCS; `(n_path,)`
- **`fbs_pos`**, **`lbs_pos`** — Scatterer positions in m; `(3, n_path)`, `(3, 0)` in `"phase"`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS; int64; `(n_path,)`
- **`ray_ind`** — Index of the ray in `rays`, 0-based; int64; `(n_path,)`

### See also:
- [ssf_scatterers](#ssf_scatterers) (scatterers for `"drift"`, validity scales `b_eff` and `b_med`)
- [gbsm.single_link_channel](#gbsm.single_link_channel) (channel coefficients of one link over a segment)
- [gbsm.rays_from_paths](#gbsm.rays_from_paths) (ray data from path data, e.g. of a QRT file)

---
<a id="ssf_rays"></a>
## ssf_rays
Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)

Also in: [C++](../cpp/geometry-based-stochastic-model.md#ssf_rays) · [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_rays)

- Splits the clusters of [ssf_clusters](#ssf_clusters) into rays and returns one ray dict per link (keys below): delays, coefficient
  powers, angles, polarization transfer matrices with initial phases, path types and tap indices; `fbs_pos`, `lbs_pos`
  and `fallback` stay empty (see [ssf_scatterers](#ssf_scatterers))
- A pure function of the clusters, the LSPs and the random values, without a seed: a link does not depend on the other
  links of the call
- Ray layout per link, fixed per scenario: index 0 LOS, 1 GR (if `GR_enabled`), then ray `m` of NLOS cluster `n` at
  `1 + G + n·M + m` (`G` = 1 with GR, `M` = `NumSubPaths`); clusters at power 0 keep their rays at power 0
- LOS and GR rays: delay, angles and power of their slot; Jones matrix diag(1, -1) (LOS), GR as in [ssf_clusters](#ssf_clusters)
- NLOS rays: power P_n / `NumSubPaths`; offsets of Table 7.5-3 scaled by the cluster spreads `PerClusterAS_D`,
  `PerClusterAS_A`, `PerClusterES_A` and the cluster ESD of eq. (7.5-20), `PerClusterES_D_scale`·10^(`lsp_mu` of
  ESD), or `PerClusterES_D` if the scale is 0; coupling (step 8): within a cluster or sub-cluster, the ray with the k-th
  smallest value of a quantity's rank block gets the k-th smallest offset, separately for AOD, AOA, EOD and EOA
- c_DS, the cluster DS of Table 7.5-6: `max(PerClusterDS_min, PerClusterDS + PerClusterDS_gamma·lg f')` ns with
  `f' = max(center_freq/1e9, PerClusterDS_fc_min)`; 3.91 ns where `PerClusterDS` = 0
- `NumSubPaths` other than 20: the offsets of QuaDRiGa-NG (the MATLAB QuaDRiGa fork that uses this library) below 20
  (mean 0, rms 1), Laplacian quantiles at the rms of Table 7.5-3 above 20; no sub-clusters

| Method | `"38.901"`, `"38.901_MF"` | `"quadriga"` |
| --- | --- | --- |
| Frequencies per call | `i_freq` only; slice `i_freq` of `rnd_ray`; `"38.901"` takes one frequency | all; one geometry, per frequency powers, XPR and Jones matrices; the shared `"ray"` set |
| Ray delays | cluster delay; sub-clusters + 0, 1.28, 2.56 c_DS (Table 7.5-5) | cluster delay + 2·c_DS·(u - mean of u in the cluster), u of block 14; negative delays set to 0 |
| Taps | sub-cluster of the two strongest clusters (`NumSubPaths` = 20), else cluster | cluster |
| c_DS, cluster ESD | at frequency `i_freq` | mean over the frequencies |

| Option | Default (TR) | Alternative (QuaDRiGa-NG) |
| --- | --- | --- |
| `angle_model` | `"additive"`: offsets added to azimuth and elevation, elevation folded at the poles (step 7) | `"rotation"`: offset pattern rotated to the cluster direction; equal at elevation 0 |
| `pol_model` | `"38.901"`: XPR and four initial phases per ray (steps 9, 10) | `"quadriga"`: rotation polarization model of QuaDRiGa, one phase, XPR per ray and per cluster |
| `pol_variability` | `False` | `True`: clause 7.6.16 on the NLOS rays with std `PolVar_sigma` (key default 0: no effect); `pol_model = "38.901"` only |

- Sub-clusters (step 11): the two clusters of largest power at frequency `i_freq` (equal powers: lower index first),
  rays at the positions 0-9, 10-15 and 16-19 (TR rays 1-8, 19, 20; 9-12, 17, 18; 13-16); every tap is a contiguous
  block of rays
- `pol_model = "quadriga"`: the matrix refers to the end with the larger z, then x, then y (coordinates within 1 mm
  count as equal; the Tx if all tie), so UTs dropped separately at equal heights may get transposed matrices

### Ray dict:
One link at one drop. All keys are present on output (empty where empty). As input, missing optional keys are empty,
other missing or unknown keys raise `ValueError`; integer keys accept any real dtype. The dict has scatterers when
`fbs_pos` has `n_ray` columns (a dict with no rays counts as having them). Rays added from another source take tap
indices the dict does not use; at most one LOS and one GR ray and one `method` for all rays. [ssf_scatterers](#ssf_scatterers)
overwrites given scatterers, so solve rays without scatterers separately and concatenate the arrays.

| Key | Shape | Type | Meaning | Required |
| --- | --- | --- | --- | --- |
| `method` | — | `str` | `"38.901"`, `"38.901_MF"` or `"quadriga"`; GR power law and default mobility | yes |
| `tx_pos`, `rx_pos` | `(3,)` | float | Tx and Rx position the ray data refer to in m | yes |
| `center_frequency` | `(n_freq,)` | float | Carrier frequencies in Hz | yes |
| `delay` | `(n_ray,)` | float | Delays in s relative to the LOS delay d3D / c, d3D the Tx-Rx distance | yes |
| `power` | `(n_ray, n_freq)` | float | Coefficient powers (linear) without link gain; data from paths: path gain | yes |
| `aod`, `eod`, `aoa`, `eoa` | `(n_ray,)` | float | Departure and arrival azimuth and elevation in rad, GCS | yes |
| `M` | `(8, n_ray, n_freq)` | float | Polarization transfer matrices incl. initial phases, real, rows as in [get_channels_planar](channel-generation-functions.md#get_channels_planar) | yes |
| `path_type` | `(n_ray,)` | uint32 | 0 = LOS, 1 = GR, 2 = NLOS | yes |
| `tap` | `(n_ray,)` | uint32 | Tap index; rays with equal index form one tap | yes |
| `gr_eps` | `(n_freq,)` or empty | complex | Ground permittivity, used as `real - j·imag`; needed where a GR ray is recomputed | no |
| `fbs_pos`, `lbs_pos` | `(3, n_ray)` or empty | float | First- and last-bounce scatterer positions in m | no |
| `fallback` | `(n_ray,)` or empty | uint32 | 1 = ray lengthened by the solver fallback | no |

### Usage:
```python
rays = quadriga_lib.gbsm.ssf_rays( scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, \
    delay, power, aod, eod, aoa, eoa, rnd_ray, method, i_freq, angle_model, pol_model, gr_eps, \
    pol_variability, random_phase )
```

### Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumSubPaths`, the `PerCluster*` keys, `GR_enabled`,
  `PolVar_sigma`
- **`tx_pos`** — Tx positions in m; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`center_freq`** — Carrier frequencies in Hz; `(n_freq,)`
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [lsp_generate](#lsp_generate); used: the ESD mean
  and the XPR mean and std, the XPR LSP for `pol_model = "quadriga"`; `(n_link, n_freq, 8)`
- **`delay`**, **`power`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays, powers and angles, outputs 0 to
  5 of [ssf_clusters](#ssf_clusters) in this order (`n_slot` slots); `power` `(n_link, n_slot, n_freq)`, the others
  `(n_link, n_slot, n_freq)` for `"38.901_MF"`, else `(n_link, n_slot, 1)`
- **`rnd_ray`** — `"ray"` sets from [gbsm.draw](#gbsm.draw) (`n_gen` columns); TR methods: slice `k` = set `k`,
  `(n_link, n_gen, n_freq)`, one set also `(n_link, n_gen)`; `"quadriga"`: the shared set (`set_id = -1`),
  `(n_link, n_gen)` or `(n_link, n_gen, 1)`
- **`method`** — Method of [ssf_clusters](#ssf_clusters): `"38.901"`, `"38.901_MF"` or `"quadriga"`; default: `"38.901"`
- **`i_freq`** — TR methods: index into `center_freq`, 0-based; `"quadriga"`: must be 0; default: 0
- **`angle_model`**, **`pol_model`** — See the option table; default: `"additive"`, `"38.901"`
- **`gr_eps`** — Ground permittivity as [lsp_generate](#lsp_generate) returns it (complex); required if `GR_enabled`, else
  ignored; `(n_link, n_freq)`; default: None
- **`pol_variability`** — TR 7.6.16 polarization power variability, see the option table; default: `False`
- **`random_phase`** — `False`: all initial phases 0; default: `True`

### Output:
- **`rays`** — Ray dicts, one per link, TR methods at frequency `i_freq` only; `list` of `n_link` dicts

### See also:
- [ssf_clusters](#ssf_clusters) (the clusters)
- [ssf_scatterers](#ssf_scatterers) (scatterer positions and validity scales of one link)
- [ssf_paths](#ssf_paths) (paths of one link)
- [gbsm.single_link_channel](#gbsm.single_link_channel) (channel coefficients from a ray dict)
- [gbsm.rays_from_paths](#gbsm.rays_from_paths) (ray dict from path data)

---
<a id="ssf_scatterers"></a>
## ssf_scatterers
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

Also in: [C++](../cpp/geometry-based-stochastic-model.md#.solve_scatterers) · [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_scatterers)

- Places the scatterers that the mobility `drift` needs and returns the validity scales of [ssf_paths](#ssf_paths)
- NLOS rays with power > 0 at some frequency: FBS on the departure ray and LBS on the arrival ray at equal distances
  from their end, so that the polyline Tx, FBS, LBS, Rx has the length d3D + c·delay (d3D the Tx-Rx distance) and the
  angles are exact
- Where no such pair exists with both distances at least `d_min`, both scatterers are placed at `d_min` along the ray's
  own directions and `fallback` is 1: the polyline is longer than the ray, `delay` keeps its value. With `d_min` = 2 m
  this applies to up to about 23 % of the rays (Indoor LOS, RMa NLOS; most in the TR methods without absolute TOA)
  and lengthens them by a few ns
- LOS ray and NLOS rays with zero power at all frequencies: the midpoint of the direct line; GR ray: the specular point
  on z = 0, which requires both ends at z >= 0
- `rays_out` is a new dict with all keys; it shares the unchanged entries with `rays` as they are (no copy and no
  conversion: a list stays a list, and an in-place edit of `rays_out["delay"]` changes `rays["delay"]`) and has new
  arrays `fbs_pos`, `lbs_pos` and `fallback`; `rays` and its arrays are not changed, also when it already holds
  scatterers
- `b_med`, the scale of `drift`: the lower power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS
  rays with power > 0, fallback rays included; NaN without NLOS power. `b_eff`, the scale of `phase`:
  `sqrt(sum P / sum(P / b²))` over the rays with power `P > 0`, with `b` the end-to-scatterer distance (NLOS), d3D
  (LOS) or the GR path length (GR)
- One link per call: `rays[l], b_med, b_eff = quadriga_lib.gbsm.ssf_scatterers(rays[l])`

### Usage:
```python
rays_out, b_med, b_eff = quadriga_lib.gbsm.ssf_scatterers( rays, d_min )
```

### Inputs:
- **`rays`** — Ray dict of one link, see [ssf_rays](#ssf_rays)
- **`d_min`** — Minimum distance of the scatterers to both ends in m, finite and > 0; default: 2.0

### Outputs:
- **`rays_out`** — Ray dict with `fbs_pos`, `lbs_pos` `(3, n_ray)` and `fallback` `(n_ray,)` (uint32)
- **`b_med`**, **`b_eff`** — Validity scales in m, rows Tx and Rx, column `k` for frequency `k`; `(2, n_freq)`

### See also:
- [ssf_rays](#ssf_rays) (ray data of a set of links)
- [ssf_paths](#ssf_paths) (mobility `drift` needs the scatterers)
- [gbsm.single_link_channel](#gbsm.single_link_channel) (reports `b_med`, `b_eff` and the displacement ratios)
