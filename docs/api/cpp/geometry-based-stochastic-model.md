# Geometry-based stochastic model

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-geometry-based-stochastic-model

| Function | Summary |
| --- | --- |
| [conf_file_list](#conf_file_list) | List the scenarios defined by the conf files in a set of directories |
| [conf_file_read](#conf_file_read) | Read and validate a QuaDRiGa scenario configuration file |
| [conf_file_write](#conf_file_write) | Write a QuaDRiGa scenario configuration file |
| [gbsm_draw](#gbsm_draw) | Spatially correlated normalized random values of one random set at a set of links |
| [gbsm_init_phases](#gbsm_init_phases) | SOS phases of one random set of a BS site and scenario |
| [gbsm_rays](#gbsm_rays) | Class for the ray data of one link (geometry-based stochastic model) |
| [gbsm_rays_from_paths](#gbsm_rays_from_paths) | Ray data of one link from path data (positions, gains, lengths, polarization), e.g. read with qrt_file_read |
| [gbsm_single_link_channel](#gbsm_single_link_channel) | Channel coefficients of one link at one frequency over one segment |
| [los_probability](#los_probability) | LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models |
| [lsp_generate](#lsp_generate) | Large-scale parameters of a set of links from normalized random values |
| [o2i_loss](#o2i_loss) | O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3 |
| [path_loss](#path_loss) | Path loss and shadow-fading standard deviation of a set of links |
| [scenpar_validate](#scenpar_validate) | Validate scenario parameters and complete them with default values |
| [ssf_clusters](#ssf_clusters) | Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7) |
| [ssf_paths](#ssf_paths) | Antenna-free paths of one link at one position |
| [ssf_rays](#ssf_rays) | Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16) |
| [.solve_scatterers](#.solve_scatterers) | First- and last-bounce scatterers of the rays of one link (two-bounce solver) |

A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the
*scenario* (e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The
functions of this section implement 3GPP TR 38.901 clause 7.5 steps 2 to 10 and the sub-cluster split of step 11
(LOS probability, path loss, O2I penetration loss, large-scale parameters (LSPs), clusters and rays, without antennas)
with the spatial consistency of clause 7.6.3, oxygen absorption (clause 7.6.1) and polarization power variability
(clause 7.6.16); [gbsm_single_link_channel](#gbsm_single_link_channel) computes the coefficients of step 11 with the antennas.

**Scenario parameters.** A scenario is a set of numeric keys (`scenpar`), a path-loss model with its parameters and
optional short names, read from a conf file by [conf_file_read](#conf_file_read). 23 files for TR 38.901 v19.5.0 come with the
library, one per scenario and LOS and indoor state (e.g. `3GPP_38.901_v19_UMa_LOS`, `_NLOS`, `_LOS_O2I`,
`_NLOS_O2I`; InH is `Indoor`, InF NLOS has one file per clutter type), in `data/config` of the source tree and in
`share/quadriga-lib/config` of an installed package; [conf_file_list](#conf_file_list) finds them. Their comments give the meaning
and unit of each key; [conf_file_write](#conf_file_write) with `write_defaults = true` writes the complete default set with these
comments. The functions that read `scenpar` complete a copy with the defaults of missing keys and validate it with
[scenpar_validate](#scenpar_validate), whose messages pass through, also for keys the function does not use.

**Workflow** for one drop of a set of links (terms: see Terms below):

1. [conf_file_read](#conf_file_read): the scenarios, one per LOS and indoor state
2. Per link, by the caller with own SOS generators ([sos_init_phases](sum-of-sinusoids-random-number-generator.md#sos_init_phases), [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand), [sos_randn](sum-of-sinusoids-random-number-generator.md#sos_randn)): the indoor
   state, a Uniform(0,1) value at the decorrelation distance `IO_state_lambda` of the O2I files below an indoor ratio
   of the caller's choice (e.g. TR 38.901 Table 7.2-1), and for indoor UTs [o2i_loss](#o2i_loss) (`loss`, `d2d_in`); then the
   LOS state, a Uniform(0,1) value at `LOS_state_lambda` of the NLOS file below `p_los` of [los_probability](#los_probability) at
   `d2d_out = max(d2D - d2d_in, 0)`. The caller chooses the `los_probability` model and the `o2i_loss` models and
   weights (TR 38.901 Tables 7.4.2-1, 7.4.3-2). The states assign each link to a scenario and give the `o2i` flags of
   [ssf_clusters](#ssf_clusters); steps 3 to 6 run per scenario
3. Per BS site and scenario, once, with one seed for all sets: [gbsm_init_phases](#gbsm_init_phases) of the random sets in the table
   below; `"lsf"` with the link positions for the bias removal
4. [gbsm_draw](#gbsm_draw) of each set at the links (the shared `"ray"` set with `center_frequency`, the list later passed to
   [ssf_rays](#ssf_rays)); [path_loss](#path_loss) (`pl`, `sf_sigma`; column 10 of the `"lsf"` set as `rnd_he`); [lsp_generate](#lsp_generate) with
   `sf_sigma` (`lsp`, `abs_toa`, `gr_eps_re`, `gr_eps_im`, `eod_offset_deg`, `lsp_mu`, `lsp_sigma` for step 5)
5. [ssf_clusters](#ssf_clusters) (`n_cluster` is the caller's choice) and [ssf_rays](#ssf_rays) (one [gbsm_rays](#gbsm_rays) object per link), called
   as in the table below
6. Per link and segment: [.solve_scatterers](#.solve_scatterers) if the mobility is `drift`; then [gbsm_single_link_channel](#gbsm_single_link_channel) with the
   link gain (see Terms) as `link_gain_dB` (TR methods: once per object, which holds one frequency; `"quadriga"`:
   once per `i_freq`), or [ssf_paths](#ssf_paths) for other channel functions and the QRT route ([qrt_paths_from_scatterers](channel-functions.md#qrt_paths_from_scatterers),
   [qrt_file_append](channel-functions.md#qrt_file_append))

| Method | Random sets | Calls in step 5 |
| --- | --- | --- |
| `"38.901"` | `"lsf"`; `"cluster"` -1 and one per frequency; `"ray"` one per frequency | per frequency `k`: [ssf_clusters](#ssf_clusters) and [ssf_rays](#ssf_rays) (`i_freq = 0`) with slice `k` of all per-frequency inputs (`center_frequency`, `lsp`, `lsp_mu`, `lsp_sigma`, `rnd_freq`, `rnd_ray`, `eod_offset_deg`, `gr_eps_re`, `gr_eps_im`) |
| `"38.901_MF"` | as `"38.901"` | [lsp_generate](#lsp_generate) at the anchor frequency with the same `rnd` gives `lsp_anchor`; [ssf_clusters](#ssf_clusters) once; [ssf_rays](#ssf_rays) per frequency `k` with `i_freq = k` |
| `"quadriga"` | `"lsf"`; `"cluster"` -1; `"ray"` -1 | [ssf_clusters](#ssf_clusters) and [ssf_rays](#ssf_rays) once for all frequencies |

For moved positions, evaluate [gbsm_draw](#gbsm_draw) (`"lsf"`), [path_loss](#path_loss), [lsp_generate](#lsp_generate) and for indoor UTs
[o2i_loss](#o2i_loss) again with the same SOS phases: they give the link gain (see Terms) and `kf_ratio` = KF(position) /
KF(drop) from slice 1 of `lsp`. Ray data can also come from path data, with [gbsm_rays_from_paths](#gbsm_rays_from_paths) (e.g. of
[qrt_file_read](channel-functions.md#qrt_file_read)), or fill a [gbsm_rays](#gbsm_rays) object directly; continue with step 6.

**Terms.** TR 38.901 names where they exist; QuaDRiGa names are marked:

| Term | Meaning |
| --- | --- |
| scenario | type of environment with its statistics: numeric keys `scenpar`, path-loss model, short names (one conf file) |
| link | one Tx-Rx pair of a batch (row `l` of `rx_pos`) |
| drop | one realization of LSPs, clusters and rays at the link positions; the positions the ray data refer to |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 38.901 clause 7.5 steps 5 to 7) |
| slot | index in the [ssf_clusters](#ssf_clusters) outputs: 0 LOS, 1 GR (if enabled), then the N clusters |
| sub-cluster | one of the three parts of a split cluster (step 11, Table 7.5-5) |
| ray | one of the `NumSubPaths` rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) |
| path | a propagation path as `get_channels_*` and QRT take it; [ssf_paths](#ssf_paths) makes one per kept ray (QuaDRiGa also says path for a cluster) |
| tap | rays summed in one output tap: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`"quadriga"`) |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed (QuaDRiGa) |
| segment | the snapshots of one drop over which the ray data are kept (QuaDRiGa) |
| mobility | how the paths follow moved ends within a segment: `phase` (phase rotation per ray, delays and angles fixed) or `drift` (scatterers fixed, geometry recomputed; QuaDRiGa drifting; validity in [ssf_paths](#ssf_paths)); `auto`: `phase` for the TR methods, `drift` for `"quadriga"` |
| scatterer | first-bounce (FBS) and last-bounce (LBS) interaction point of a ray; one point for LOS (direct line) and GR (ground) (QuaDRiGa) |
| coefficient power | power of a ray without link gain and oxygen loss (`gbsm_rays.power`): the powers of TR 38.901 eq. (7.5-30) split over the rays, GR as [ssf_clusters](#ssf_clusters) sets it; for data from paths the path gain |
| path gain | linear power gain of a path incl. link gain and oxygen loss |
| link gain | `-pl - loss + 10·lg(SF)` in dB of a link at a position and frequency (`link_gain_dB`), with `pl` of [path_loss](#path_loss), `loss` of [o2i_loss](#o2i_loss) (0 outdoors) and SF the linear shadow fading, slice 2 of `lsp` |
| random set | the spatially correlated random values of one `set_type` and `set_id` (table in [gbsm_draw](#gbsm_draw)) |
| generator | one spatially correlated random field, a [sum-of-sinusoids (SOS)](sum-of-sinusoids-random-number-generator.md) generator; one column of a random set |
| SOS phases | the random phases of the sinusoids of the generators of a random set (`phase`); evaluated at positions, they give its random values. "Phase" alone is the phase of a coefficient or ray |
| block | the generators of one random quantity in a set |

**Codes.** Path type: 0 LOS, 1 GR, 2 NLOS. Tap index: rays with equal index form one tap; [ssf_rays](#ssf_rays) gives 3·slot +
i, with i the sub-cluster (0, 1, 2) of a split cluster and 0 otherwise: LOS 0, GR 3 if enabled, NLOS cluster `n` (from
0) 3·(1 + G + n) + i with G = 1 if GR is enabled (without GR the first cluster has 3).

**Model properties.** UTs closer than `SC_lambda` get correlated clusters and rays (TR 38.901 clause 7.6.3.1). In the
TR methods, the tap fading of separately dropped UTs then stays correlated over metres instead of a few wavelengths
(tap-power correlation 0.5 at about 0.3·`SC_lambda`, 0.1 at 0.8·`SC_lambda`); `"quadriga"` avoids this. In all
methods, the cluster delays and angles of separately dropped UTs change faster than for a moved UT, so their
single-frequency channel and Doppler are wrong at small distances: for close UTs, drop once and move them with
[ssf_paths](#ssf_paths) or [gbsm_single_link_channel](#gbsm_single_link_channel). Across bands, `"quadriga"` uses one ray set, so the fading is correlated;
the TR methods draw coupling, XPR and phases per frequency, so it is independent, also with `"38.901_MF"`, which keeps
the delays and angles.

**Conventions.** `N = NumClusters - 1` is the number of NLOS clusters: `NumClusters` counts the LOS slot, as in
QuaDRiGa. Positions are in m; in the batch functions one row (x, y, z) per link: `rx_pos` is `[n_link, 3]`, `tx_pos` is
`[1, 3]` (one Tx for all links) or `[n_link, 3]`; heights are the z coordinates. The one-link functions take a position
as a `[3]` vector, except [gbsm_single_link_channel](#gbsm_single_link_channel), which takes one row per snapshot; scatterer positions are
`[3, n]`. Angles are elevations, not TR zenith angles (EOD, EOA, ESD for ZOD, ZOA, ZSD; also in the conf-file keys): in
rad in the GCS, azimuth in [-pi, pi), elevation in [-pi/2, pi/2] with +pi/2 up. Defaults follow TR 38.901 except two
deliberate library choices: the SOS preset `Comb256` with bias removal ([gbsm_init_phases](#gbsm_init_phases), [gbsm_draw](#gbsm_draw)) and
`method = "quadriga"` of [gbsm_rays_from_paths](#gbsm_rays_from_paths); QuaDRiGa variants are options. Units: frequencies in Hz, path loss in dB, DS in s, KF, SF and XPR as
linear power ratios, spreads and `_deg` offsets in degrees.

---
<a id="conf_file_list"></a>
## conf_file_list
List the scenarios defined by the conf files in a set of directories

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#conf_file_list) · [Python](../python/geometry-based-stochastic-model.md#conf_file_list)

- Lists the regular files and symbolic links to regular files named `<scenario>.conf` (case-sensitive, non-empty
  scenario name) in each directory, not recursive: the directories in the given order, the files of one directory
  in byte-wise sorted order of their names
- A file name found in an earlier directory is skipped, so a local folder can override the files of a shared one
- With `parse_shortnames = true` every line `ShortName = name` that is not a comment adds an entry right after its
  file; other lines are not examined
- Throws `std::invalid_argument` if a short name equals an earlier entry, if a file name equals an earlier short
  name, or if a `ShortName` value is not one name; `std::runtime_error` if a listed file cannot be opened
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty string is the
  current directory. A failed call leaves the outputs untouched

### Declaration:
```cpp
arma::uword quadriga_lib::conf_file_list(
    const std::vector<std::string> &dirs,
    std::vector<std::string> &scenarios,
    std::vector<std::string> *file_names = nullptr,
    std::vector<std::string> *file_dirs = nullptr,
    bool parse_shortnames = true);
```

### Inputs:
- **`dirs`** — Directories, searched in this order; can be empty
- **`parse_shortnames`** *(optional)* — `true`: read the files and add their short names; `false`: list the
  files only, which is faster

### Outputs:
- **`scenarios`** — Scenario names and short names; previous content is discarded; `[n_scen]`
- **`file_names`** *(optional)* — File name including `.conf` for each entry; `[n_scen]`
- **`file_dirs`** *(optional)* — Directory of the file for each entry, exactly as given in `dirs`; `[n_scen]`

### Returns:
- Number of entries, `n_scen`

### See also:
- [conf_file_read](#conf_file_read) (read one of the files)

---
<a id="conf_file_read"></a>
## conf_file_read
Read and validate a QuaDRiGa scenario configuration file

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#conf_file_read) · [Python](../python/geometry-based-stochastic-model.md#conf_file_read)

- A conf file has one `Key = value` assignment per line; `%` starts a comment; blank lines, LF or CRLF line ends and
  a UTF-8 byte order mark at the start are accepted; keys are case-sensitive names of `A-Z`, `a-z`, `0-9` and `_`
- Three kinds of keys:
  - the `n_key` numeric scenario parameters of [scenpar_validate](#scenpar_validate), one number each; keys that are not in the file
    have their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the QuaDRiGa key
  `GR_epsilon` (the complex ground permittivity comes from [lsp_generate](#lsp_generate)). A key stated more than once has its
  last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of `SubpathMethod`
  throws, also with `validate = false`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`), read
  independent of the global locale; `Inf`, `+Inf` and `-Inf` in any case
- Throws `std::runtime_error` if the file cannot be opened or is a directory
- Throws `std::invalid_argument` with file name and line number for a line that is not blank, not a comment and not
  a well-formed assignment (no `=`, not one name before the `=`, a missing value, a word where a number is required
  or the reverse, text after the value, `NaN`, a number beyond the range of a double, a `PL_` key without a name, a
  `ShortName` value stated twice), and with `validate = true` for a value that fails [scenpar_validate](#scenpar_validate)
- A failed call leaves all outputs untouched

### Declaration:
```cpp
bool quadriga_lib::conf_file_read(
    const std::string &fn,
    std::unordered_map<std::string, double> &scenpar,
    std::string *pl_model = nullptr,
    std::unordered_map<std::string, std::vector<double>> *plpar = nullptr,
    arma::mat *lsp_xcorr = nullptr,
    std::vector<std::string> *key_order = nullptr,
    std::vector<std::string> *pl_order = nullptr,
    std::vector<std::string> *short_names = nullptr,
    std::vector<std::string> *ignored_keys = nullptr,
    bool validate = true);
```

### Inputs:
- **`fn`** — Path to the conf file
- **`validate`** *(optional)* — `true`: apply the value checks of [scenpar_validate](#scenpar_validate); `false`: only the line
  grammar is checked

### Outputs:
- **`scenpar`** — Numeric scenario parameters; previous content is discarded; `n_key` entries
- **`pl_model`** *(optional)* — Value of `PL_model`; empty if not in the file
- **`plpar`** *(optional)* — Path-loss parameters: the `PL_` keys other than `PL_model`, without the prefix, with
  one or more values each; `n_pl` entries
- **`lsp_xcorr`** *(optional)* — LSP cross-correlation matrix, see [scenpar_validate](#scenpar_validate); `[8, 8]`
- **`key_order`** *(optional)* — Canonical order of the keys in `scenpar`, see [scenpar_validate](#scenpar_validate); `[n_key]`
- **`pl_order`** *(optional)* — Keys of `plpar` in the order of their first occurrence in the file; `[n_pl]`
- **`short_names`** *(optional)* — Values of all `ShortName` lines in file order; `[n_short]`
- **`ignored_keys`** *(optional)* — Unknown keys, each once, in the order of their first occurrence; `[n_ignored]`

### Returns:
- `true` if `lsp_xcorr` is positive definite; `false` is no error here, but [lsp_generate](#lsp_generate) throws for the scenario

### See also:
- [scenpar_validate](#scenpar_validate) (value checks and defaults)
- [conf_file_list](#conf_file_list) (find the conf files in a set of directories)
- [conf_file_write](#conf_file_write) (write a conf file)

---
<a id="conf_file_write"></a>
## conf_file_write
Write a QuaDRiGa scenario configuration file

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#conf_file_write) · [Python](../python/geometry-based-stochastic-model.md#conf_file_write)

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files of QuaDRiGa;
  every written file passes [conf_file_read](#conf_file_read) with `validate = true`, without ignored keys and with the same
  values bit for bit
- The header has the scenario name (the file name without `.conf`), `ref`, the validity range and the short names.
  Up to 10 sections follow, each with an optional line `% See: ref; ref_sec` and, for the LSPs and the path-loss
  model, the formulas as comments. Sections by index in `ref_sec`: 0 large-scale distributions, 1 large-scale
  decorrelation distances, 2 inter-parameter correlations, 3 model parameters, 4 small-scale decorrelation distance,
  5 ground reflection, 6 absolute TOA, 7 path-loss model, 8 EoD offset, 9 optional modelling components; 0 to 7 are
  those of QuaDRiGa, so its 8 section references can be passed unchanged
- Numbers are written independent of the global locale; `Inf`, `-Inf` and `-0` are kept
- `write_defaults = false`: only keys whose value differs from the default (bit for bit); a section without a
  written key is left out
- All inputs are validated before the file is opened: the checks of [scenpar_validate](#scenpar_validate); `pl_model`, the keys of
  `plpar` and the short names are names, the short names distinct; `plpar` has no key `model`, and every key has at
  least one value and no `NaN`; `pl_order` names each key of `plpar` once; `fn`, `ref` and `ref_sec` have no line
  break
- The text is written to a temporary file in the same directory, which must be writable, and renamed to `fn`: a
  failed call creates no file and leaves an existing file unchanged; an existing file keeps its permissions, a
  symbolic link `fn` is written through
- Throws `std::invalid_argument` for invalid input; `std::runtime_error` if the file cannot be written, or if `fn`
  exists and is not a regular file that can be opened for writing

### Declaration:
```cpp
void quadriga_lib::conf_file_write(
    const std::string &fn,
    const std::unordered_map<std::string, double> &scenpar,
    const std::string &pl_model = "",
    const std::unordered_map<std::string, std::vector<double>> *plpar = nullptr,
    const std::vector<std::string> *pl_order = nullptr,
    const std::vector<std::string> *short_names = nullptr,
    const std::string &ref = "",
    const std::vector<std::string> *ref_sec = nullptr,
    const arma::mat *valid_range = nullptr,
    bool write_defaults = false);
```

### Inputs:
- **`fn`** — Path of the conf file to write
- **`scenpar`** — Numeric scenario parameters; any subset of the `n_key` keys; missing keys have their default
- **`pl_model`** *(optional)* — Value of `PL_model`; empty: not written
- **`plpar`** *(optional)* — Path-loss parameters; keys without the prefix `PL_`, one or more values each; `[n_pl]`
- **`pl_order`** *(optional)* — Order in which the keys of `plpar` are written; omitted: sorted by name; `[n_pl]`
- **`short_names`** *(optional)* — Values of the `ShortName` lines
- **`ref`** *(optional)* — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`
- **`ref_sec`** *(optional)* — Reference per section, indexed as listed above; an empty or missing entry writes no
  reference; up to 10 entries
- **`valid_range`** *(optional)* — Validity range, written as comment; rows: frequency in GHz, distance, BS
  height, UT height in m; columns: lower and upper end, one value is written if both are equal; a row with `NaN`
  is not written; empty: no validity range; `[4, 2]`
- **`write_defaults`** *(optional)* — `true`: write all keys; `false`: write only keys that differ from their default

### See also:
- [conf_file_read](#conf_file_read) (read the file back)

---
<a id="gbsm_draw"></a>
## gbsm_draw
Spatially correlated normalized random values of one random set at a set of links

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_draw) · [Python](../python/geometry-based-stochastic-model.md#gbsm.draw)

- Evaluates the SOS phases of [gbsm_init_phases](#gbsm_init_phases) at the links as [sos_val](sum-of-sinusoids-random-number-generator.md#sos_val), one column per generator. The value of a
  link depends neither on the other links nor on the other columns; with `dual_mobility = false` not on `tx_pos`
- `dual_mobility = true`: the Tx is a second device on the same fields; its side of each generator uses the mirror
  generator (e.g. departure and arrival quantities exchange), and every column keeps its distribution
- 2D mode (`horizontal = true`, TR): the values are correlated in the x-y plane; links with equal `floor_id` share one
  field, fields of different floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6 note 3):
  exactly for IDs in the same group of `n_coeff` consecutive IDs (0 to `n_coeff - 1`, ...), otherwise up to a random
  residual of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- 3D mode (legacy QuaDRiGa): the height separates the floors, so UTs on different floors are correlated by the ACF of
  their height difference. The per-link statistics are the same in both modes
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation
- An empty set (`NumClusters = 1`) accepts any empty `phase`

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

### Declaration:
```cpp
void quadriga_lib::gbsm_draw(
    const std::unordered_map<std::string, double> &scenpar,
    const std::string &set_type,
    const arma::cube &phase,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &rnd,
    arma::sword set_id = -1,
    const std::string &preset = "Comb256",
    const arma::u32_vec *floor_id = nullptr,
    const arma::vec *center_frequency = nullptr,
    bool dual_mobility = false,
    bool horizontal = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters; decorrelation distances, `NumClusters`, `NumSubPaths`, `PerClusterDS*` and
  `PerClusterAS_A`
- **`set_type`** — Set type of [gbsm_init_phases](#gbsm_init_phases)
- **`phase`** — SOS phases from [gbsm_init_phases](#gbsm_init_phases), finite; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility, always validated; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`, `n_link = 0` allowed
- **`set_id`** *(optional)* — Set of [gbsm_init_phases](#gbsm_init_phases); ignored for `"lsf"`
- **`preset`** *(optional)* — Preset of [gbsm_init_phases](#gbsm_init_phases); another preset with the same `n_coeff` is not detected
- **`floor_id`** *(optional)* — Floor number of the UT per link, ground floor 0 (`n_fl - 1` for the UT heights of
  TR 38.901 Table 7.2-1, 0 for outdoor UTs); omitted = floor 0; 2D mode only; `[n_link]`
- **`center_frequency`** *(optional)* — Carrier frequencies in [Hz]; required by the shared `"ray"` set, whose block 14
  uses their mean c_DS ([ssf_rays](#ssf_rays)): pass the same list as to [ssf_rays](#ssf_rays) (another list gives other `"quadriga"`
  rays); ignored and not checked for all other sets; `[n_freq]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move
- **`horizontal`** *(optional)* — `true`: 2D mode; `false`: 3D mode

### Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

### See also:
- [gbsm_init_phases](#gbsm_init_phases) (the SOS phases)
- [lsp_generate](#lsp_generate) (the large-scale parameters from the `"lsf"` set)
- [path_loss](#path_loss) (column 10 of the `"lsf"` set as `rnd_he`)
- [ssf_clusters](#ssf_clusters) (the clusters from the `"cluster"` sets)
- [ssf_rays](#ssf_rays) (the rays from the `"ray"` sets)

---
<a id="gbsm_init_phases"></a>
## gbsm_init_phases
SOS phases of one random set of a BS site and scenario

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_init_phases) · [Python](../python/geometry-based-stochastic-model.md#gbsm.init_phases)

- Draws the SOS phases of one random set (set table in [gbsm_draw](#gbsm_draw)): the random phases of the sinusoids of its
  [sum-of-sinusoids (SOS)](sum-of-sinusoids-random-number-generator.md) generators, as
  [sos_init_phases](sum-of-sinusoids-random-number-generator.md#sos_init_phases); [gbsm_draw](#gbsm_draw) evaluates them at the links. Store `phase` or the seed to evaluate the same fields
  again along tracks or on maps
- The SOS phases are a fixed function of `seed`, `set_type` and `set_id`, equal on all platforms (bias-removed
  `"lsf"` sets within about 2e-14 rad); the sets are independent; each scenario of a site needs its own seed
- `"lsf"` bias removal (`remove_bias`): [sos_remove_bias](sum-of-sinusoids-random-number-generator.md#sos_remove_bias) sets the sample mean of columns 0 to 8 over the positions
  (Rx, and Tx in dual mobility; in 2D mode all on floor 0) to 0, so these means do not vary from drop to drop
- `NumClusters = 1`: all sets but `"lsf"` are empty
- TR 38.901 clause 7.4.4 literally: preset `Exp256` without bias removal

### Declaration:
```cpp
void quadriga_lib::gbsm_init_phases(
    const std::unordered_map<std::string, double> &scenpar,
    const std::string &set_type,
    arma::cube &phase,
    arma::sword seed = -1,
    arma::sword set_id = -1,
    const std::string &preset = "Comb256",
    const arma::mat *tx_pos = nullptr,
    const arma::mat *rx_pos = nullptr,
    bool dual_mobility = false,
    bool remove_bias = true,
    bool horizontal = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances, `NumClusters` and `NumSubPaths` are used
- **`set_type`** — `"lsf"`, `"cluster"` or `"ray"`
- **`seed`** *(optional)* — Seed of the site and scenario, >= -1; -1 = system random device, new SOS phases on each call
- **`set_id`** *(optional)* — `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset); sets `n_coeff`
- **`tx_pos`**, **`rx_pos`** *(optional)* — Bias-removal positions in [m] as in [gbsm_draw](#gbsm_draw); `"lsf"` only, else throws
- **`dual_mobility`** *(optional)* — `"lsf"` with bias removal: Tx and Rx both move; as in [gbsm_draw](#gbsm_draw)
- **`remove_bias`** *(optional)* — `"lsf"` only: remove the sample mean over the positions; needs `rx_pos` with at
  least 3 rows, else no effect
- **`horizontal`** *(optional)* — `"lsf"` with bias removal: `true` 2D, `false` 3D mode; as in [gbsm_draw](#gbsm_draw)

### Output:
- **`phase`** — SOS phases in [rad], in [-pi, pi), row = sinusoid, column = generator; slice 0 = Rx side, slice 1 =
  Tx side; `[n_coeff, n_gen, 2]`

### See also:
- [gbsm_draw](#gbsm_draw) (evaluates the SOS phases at the links)
- [sos_init_phases](sum-of-sinusoids-random-number-generator.md#sos_init_phases), [sos_val](sum-of-sinusoids-random-number-generator.md#sos_val) (SOS phases and evaluation of one generator), [sos_remove_bias](sum-of-sinusoids-random-number-generator.md#sos_remove_bias) (its bias removal)

---
<a id="gbsm_rays"></a>
## gbsm_rays
Class for the ray data of one link (geometry-based stochastic model)

Not in: MATLAB, Python

- Antenna-free rays of one link at one drop: delays, powers, angles and polarization transfer matrices, with the
  positions the data refer to; written by [ssf_rays](#ssf_rays) and [gbsm_rays_from_paths](#gbsm_rays_from_paths), completed by
  [.solve_scatterers](#.solve_scatterers), read by [ssf_paths](#ssf_paths) and [gbsm_single_link_channel](#gbsm_single_link_channel), which call `is_valid()` first
- Plain data: the class does not normalize or reorder; later versions append attributes, so set them by name
- An object has scatterers when `fbs_pos` has `n_ray` columns (an empty object counts as having them)
- Rays added from another source take tap indices the object does not use; at most one LOS and one GR ray (checked by
  `is_valid()`) and one `method` for all rays. `.solve_scatterers` overwrites given scatterers, so solve rays without
  scatterers in a separate object and concatenate the attributes
- Extreme but finite inputs (e.g. an XPR of 1000 dB, `kf_ratio` or positions near the range of double) can give
  non-finite attributes or outputs; the consumer's `is_valid()` message then names the attribute, not the input

### Attributes:
| Attribute | Type, size | Meaning | Set by |
| --- | --- | --- | --- |
| `method` | `std::string`, default `"38.901"` | `"38.901"`, `"38.901_MF"` or `"quadriga"`; GR power law and default mobility | all |
| `tx_pos`, `rx_pos` | `arma::vec`, `[3]` | Tx and Rx position the ray data refer to in [m] | all |
| `center_frequency` | `arma::vec`, `[n_freq]` | Carrier frequencies in [Hz] | all |
| `delay` | `arma::vec`, `[n_ray]` | Delays in [s] relative to the LOS delay d3D / c, d3D the Tx-Rx distance | all |
| `power` | `arma::mat`, `[n_ray, n_freq]` | Coefficient powers (linear) without link gain; data from paths: path gain | all |
| `aod`, `eod`, `aoa`, `eoa` | `arma::vec`, `[n_ray]` | Departure and arrival azimuth and elevation in [rad], GCS | all |
| `M` | `arma::cube`, `[8, n_ray, n_freq]` | Polarization transfer matrices incl. initial phases, rows as in [get_channels_planar](channel-generation-functions.md#get_channels_planar) | all |
| `path_type` | `arma::u32_vec`, `[n_ray]` | 0 = LOS, 1 = GR, 2 = NLOS | all |
| `tap` | `arma::u32_vec`, `[n_ray]` | Tap index; rays with equal index form one tap | all |
| `gr_eps_re`, `gr_eps_im` | `arma::vec`, `[n_freq]` or empty | Ground permittivity `gr_eps_re - j·gr_eps_im`; needed where a GR ray is recomputed | [ssf_rays](#ssf_rays), [gbsm_rays_from_paths](#gbsm_rays_from_paths) |
| `fbs_pos`, `lbs_pos` | `arma::mat`, `[3, n_ray]` or empty | First- and last-bounce scatterer positions in [m] | [.solve_scatterers](#.solve_scatterers), [gbsm_rays_from_paths](#gbsm_rays_from_paths) |
| `fallback` | `arma::u32_vec`, `[n_ray]` or empty | 1 = ray lengthened by the solver fallback | [.solve_scatterers](#.solve_scatterers) |

### Member functions:
| Member | Returns |
| --- | --- |
| `n_ray()` | Number of rays, length of `delay` |
| `n_freq()` | Number of frequencies, length of `center_frequency` |
| `empty()` | `true` if `n_ray() == 0` |
| `is_valid()` | Empty string if the object is valid, else the message of the first failed check |
| `b_med(i_freq = 0)` | `arma::vec` `[2]`, Tx and Rx in [m]: lower power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS rays with power > 0 at frequency `i_freq` (0-based), fallback rays included; validity scale of the mobility `drift`; NaN without scatterers or without NLOS power |
| `b_eff(i_freq = 0)` | `arma::vec` `[2]`, Tx and Rx in [m]: `sqrt(sum P / sum(P / b²))` over the rays with power `P > 0` at frequency `i_freq`, with `b` the end-to-scatterer distance (NLOS), d3D (LOS) or the GR path length (GR); validity scale of the mobility `phase`; NaN without scatterers or power |
| `solve_scatterers(d_min = 2.0)` | `void`; writes `fbs_pos`, `lbs_pos` and `fallback`, see [.solve_scatterers](#.solve_scatterers) |

- `b_med` and `b_eff` throw the `is_valid()` message for an invalid object and an error for `i_freq >= n_freq()`

### See also:
- [ssf_rays](#ssf_rays) (ray data of a set of links)
- [gbsm_rays_from_paths](#gbsm_rays_from_paths) (ray data from path data)
- [.solve_scatterers](#.solve_scatterers) (scatterers for the mobility `drift`)
- [ssf_paths](#ssf_paths) (paths of one link at one position)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel of the link)

---
<a id="gbsm_rays_from_paths"></a>
## gbsm_rays_from_paths
Ray data of one link from path data (positions, gains, lengths, polarization), e.g. read with qrt_file_read

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_rays_from_paths) · [Python](../python/geometry-based-stochastic-model.md#gbsm.rays_from_paths)

- Fills a [gbsm_rays](#gbsm_rays) object (`fallback` cleared; members of the right size keep their memory) from path data in the names and order of
  [qrt_file_read](channel-functions.md#qrt_file_read), read as `qrt_file_read<double>` with `downlink = true` and `normalize_M = 1` (an uplink read gives
  swapped ends and a conjugated `M`, which this function does not undo; `normalize_M = 0` leaves only the free-space
  gain in `power`)
- `delay` = (path length - Tx-Rx distance) / c, `power` = path gain. Angles point from the Tx to the first of
  `fbs_pos`, `lbs_pos`, `rx_pos` that differs from it and from the Rx to the first of `lbs_pos`, `fbs_pos`, `tx_pos`;
  GR angles to the image of the other end (z mirrored); 0 if `tx_pos` = `rx_pos`
- QRT stores no path types or taps. Without `path_type`: LOS where FBS = LBS on the direct line (0.1 mm), else
  NLOS. A GR path needs `path_type`: for files of [qrt_paths_from_scatterers](channel-functions.md#qrt_paths_from_scatterers) pass the `no_int` of [qrt_file_read](channel-functions.md#qrt_file_read)
  (interaction count = path type); for ray-traced files the user sets it (values > 2 throw)
- The GR point (`lbs_pos`) must be within 1 mm of z = 0

### Declaration:
```cpp
void quadriga_lib::gbsm_rays_from_paths(
    const arma::vec &center_frequency,
    const arma::vec &tx_pos,
    const arma::vec &rx_pos,
    const arma::mat &fbs_pos,
    const arma::mat &lbs_pos,
    const arma::mat &path_gain,
    const arma::vec &path_length,
    const arma::cube &M,
    gbsm_rays &rays,
    const std::string &method = "quadriga",
    const arma::u32_vec *path_type = nullptr,
    const arma::u32_vec *tap = nullptr,
    const arma::vec *gr_eps_re = nullptr,
    const arma::vec *gr_eps_im = nullptr);
```

### Inputs:
- **`center_frequency`** — Frequencies in [Hz]; `[n_freq]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in [m]; `[3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in [m]; `[3, n_path]`
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`M`** — Polarization transfer matrices; `[8, n_path, n_freq]`
- **`method`** *(optional)* — Value of `rays.method`; sets the GR power law and the default mobility
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS, at most one LOS and one GR; `[n_path]`
- **`tap`** *(optional)* — Tap index; omitted: LOS 0, GR 3, the others 6, 9, ... in path order; `[n_path]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity, given together; needed only where a GR ray is
  recomputed, i.e. in the mobility `drift` at moved ends; `[n_freq]`

### Output:
- **`rays`** — Ray data with `n_ray = n_path`, see [gbsm_rays](#gbsm_rays); use with `link_gain_dB = 0` and without `kf_ratio`

### See also:
- [qrt_file_read](channel-functions.md#qrt_file_read) (source of the path data, `downlink = true`)
- [qrt_paths_from_scatterers](channel-functions.md#qrt_paths_from_scatterers) (write the paths of [ssf_paths](#ssf_paths) to a QRT file)
- [ssf_paths](#ssf_paths) (paths at moved positions)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel of the link)

---
<a id="gbsm_single_link_channel"></a>
## gbsm_single_link_channel
Channel coefficients of one link at one frequency over one segment

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_single_link_channel) · [Python](../python/geometry-based-stochastic-model.md#gbsm.single_link_channel)

- Computes the MIMO coefficients of one link with its antennas over the `n_snap` snapshots of one segment at
  `rays.center_frequency(i_freq)`: per snapshot, the paths of [ssf_paths](#ssf_paths) at the snapshot positions with the link
  gain and K-factor ratio of the snapshot, for the kept rays (power > 0 at `i_freq`, plus LOS and GR)
- `"phase"`: [get_channels_planar](channel-generation-functions.md#get_channels_planar) with the arrays at the positions of `rays` and the orientations of the snapshot;
  delays, path lengths and angles stay those of the ray-data positions. `"drift"`: [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) with the
  arrays at the positions and orientations of the snapshot
- `"taps"`: the rays summed per tap index, taps in ascending index, one tap per distinct index of the kept rays (fixed
  per call); LOS and GR taps are present whenever `rays` holds them, also at zero power. The delay of a tap per
  element pair is the mean of the rays' delays weighted by `rays.power` at `i_freq` (equal weights where they sum to
  0)
- `"taps"` is exact at the phase center for the TR methods (the rays of a tap share one delay); for `"quadriga"` only
  at the carrier: a tap sums rays of different delays and is frequency-selective within the band, so use `"rays"` for
  wide bands. `"rays"`: one path per kept ray in ray order, exact at any bandwidth, the form of the QRT route; the
  path data are those of `ssf_paths` at the snapshot for these rays
- `n_snap` is the largest row count of the positions and orientations (1 if all are omitted); each has 1 or `n_snap`
  rows, `link_gain_dB` and `kf_ratio` 1 or `n_snap` entries (per snapshot at `i_freq`, while `ssf_paths` takes one
  per frequency); one row or entry applies to all snapshots. A static link with a gain per snapshot repeats its
  position
- Powers are not renormalized, and `initial_position` of the result stays 0. Validity limits: see `ssf_paths`
- Throws `std::invalid_argument` for an invalid array (message after `Transmit antenna: ` or `Receive antenna: `),
  invalid `rays` (the `is_valid()` message), `"drift"` without scatterers or with a GR ray at moved ends without
  permittivity or below z = 0, and a row or entry count other than 1 or `n_snap`

### Declaration:
```cpp
channel<double> quadriga_lib::gbsm_single_link_channel(
    const arrayant<double> &tx_array,
    const arrayant<double> &rx_array,
    const gbsm_rays &rays,
    const arma::vec &link_gain_dB,
    const std::string &mobility = "auto",
    const std::string &output = "taps",
    arma::uword i_freq = 0,
    const arma::mat *tx_pos = nullptr,
    const arma::mat *rx_pos = nullptr,
    const arma::mat *tx_orientation = nullptr,
    const arma::mat *rx_orientation = nullptr,
    const arma::vec *kf_ratio = nullptr,
    bool oxygen = false,
    bool use_absolute_delays = false);
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array antenna with `n_tx` and `n_rx` ports
- **`rays`** — Ray data of the link, see [gbsm_rays](#gbsm_rays)
- **`link_gain_dB`** — Link gain (see Terms) at `rays.center_frequency(i_freq)` in [dB]; `[n_snap]` or `[1]`
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"` (see Terms)
- **`output`** *(optional)* — `"taps"` or `"rays"`
- **`i_freq`** *(optional)* — Frequency index in `rays.center_frequency`, 0-based; ray data of the TR methods hold
  one frequency: 0
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx positions in [m]; omitted: the positions of `rays`;
  `[n_snap, 3]` or `[1, 3]` (a `[3]` vector is rejected)
- **`tx_orientation`**, **`rx_orientation`** *(optional)* — Euler angles (bank, tilt, heading) in [rad]; omitted: 0;
  `[n_snap, 3]` or `[1, 3]`
- **`kf_ratio`** *(optional)* — KF(snapshot) / KF(drop) at `i_freq`, >= 0; omitted: 1; `[n_snap]` or `[1]`
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1, see `ssf_paths`
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays (`"phase"`: of the ray-data positions)

### Returns:
- Channel object: `center_frequency` `[1]`; `tx_pos`, `rx_pos` (the snapshot positions), `tx_orientation`,
  `rx_orientation` `[3, n_snap]`; per snapshot `coeff_re`, `coeff_im`, `delay` `[n_rx, n_tx, n_tap]` (`"taps"`) or
  `[n_rx, n_tx, n_path]` (`"rays"`); `"rays"` also `path_gain`, `path_length`, `path_polarization`, `path_angles`
  and in `"drift"` `path_fbs_pos`, `path_lbs_pos`; `par_names`, `par_data` with the entries below

### Entries of `par_names` / `par_data`:

| Name | Type, size | Content |
| --- | --- | --- |
| `tap` | `arma::u32_vec`, `[n_tap]` or `[n_path]` | Tap index per output tap or per path |
| `n_ray_tap` | `arma::u32_vec`, `[n_tap]` | Number of kept rays per tap; `"taps"` only |
| `fbs_pos`, `lbs_pos` | `arma::fmat`, `[3, n_kept]` | Scatterers of the kept rays at the ray-data positions; `"drift"` with `"taps"` only; float32, the precision of path data in HDF5 files |
| `gr_pos` | `arma::fmat`, `[3, n_snap]` | GR point per snapshot; `"drift"` with `"taps"` and a GR ray only |
| `link_gain_dB` | `arma::vec`, `[n_snap]` | Link gain per snapshot |
| `b_med`, `b_eff` | `arma::vec`, `[2]` | Of [gbsm_rays](#gbsm_rays) at `i_freq`, Tx and Rx; without scatterers from a solved copy (`d_min` = 2 m) |
| `limit` | `arma::vec`, `[2]` | Validity limit per end in [m] (see [ssf_paths](#ssf_paths)): `sqrt(λ·b_eff/8)` in `"phase"`, `0.2·b_med` in `"drift"`; Inf where `b_eff` or `b_med` is NaN |
| `ratio` | `arma::vec`, `[2]` | Largest distance of the end from its ray-data position over the snapshots, divided by `limit` |
| `ratio_total` | `double` | `sqrt(ratio(0)² + ratio(1)²)`; valid up to 1 |

### See also:
- [ssf_paths](#ssf_paths) (paths of one link at one position, validity limits)
- [gbsm_rays](#gbsm_rays) (input data, `b_med` and `b_eff`)
- [.solve_scatterers](#.solve_scatterers) (scatterers for `"drift"`)
- [gbsm_rays_from_paths](#gbsm_rays_from_paths) (ray data from path data)
- [channel](channel-class.md#channel) (the returned object)

---
<a id="los_probability"></a>
## los_probability
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#los_probability) · [Python](../python/geometry-based-stochastic-model.md#los_probability)

- Evaluates the probability that a link is in line-of-sight (LOS) from its 2D distance and the BS and UT heights,
  and on request the soft LOS state of TR 38.901 eq. (7.6-18); the LOS state itself (a draw against `p_los`) and
  its spatial correlation are the caller's
- `38.901_*`: TR 38.901 v19.5.0 Table 7.4.2-1 (`InH_Mixed`, `InH_Open`: mixed and open office). Where the table
  leaves cases open: all results are limited to [0, 1]; UMa uses `C' = 1` for `hUT > 23` m; InF-SH and InF-DH give 1
  for `hUT >= h_clutter` and the rule of InF-SL and InF-DL for `hBS <= hUT`; SMa gives 1 for `hBS <= hUT` and uses
  `max(h_x - hUT, 0)` in all three terms (the TR only in the residential term; equal for `hUT < 15` m)
- `38.811_*`: TR 38.811 Table 6.6.1-1 over the elevation `atan2(hBS, d2d_out)`, linearly interpolated;
  `Suburban` and `Rural` share one table
- `QuaDRiGa_Industrial`: `min(1, exp(-(d2d_out + 20)/80)·exp(min(hBS, 8)/5 - 1))`; `LOS`: 1; `NLOS`: 0
- The `LOS_state_lambda` of the InF conf files is `d_clutter/2` of the TR defaults; with another `d_clutter` pass a
  matching correlation distance

### Declaration:
```cpp
void quadriga_lib::los_probability(
    const std::string &model,
    const arma::vec &d2d_out,
    const arma::vec &h_bs,
    const arma::vec &h_ut,
    arma::vec &p_los,
    double d_clutter = NAN,
    double r_clutter = NAN,
    double h_clutter = NAN,
    const arma::vec *rnd_g = nullptr,
    const arma::vec *center_frequency = nullptr,
    arma::mat *los_soft = nullptr);
```

### Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`, `38.901_InH_Open`,
  `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`, `38.901_InF_HH`, `38.901_SMa`,
  `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in [m], finite and >= 0: `d2D - d2d_in` for O2I UTs, clamped at 0 by the
  caller; the 2D distance in indoor scenarios; `[n_link]`
- **`h_bs`** — BS height in [m], finite; `[n_link]` or `[1]` for all links
- **`h_ut`** — UT height in [m], finite; `[n_link]` or `[1]` for all links
- **`d_clutter`** *(optional)* — Clutter size in [m] of the InF and SMa models, > 0; NaN = TR default (InF-SL and
  InF-SH 10 m, InF-DL and InF-DH 2 m, SMa 30 m)
- **`r_clutter`** *(optional)* — Clutter density of the InF models (default 0.2 for SL and SH, 0.6 for DL and DH),
  vegetation density of SMa (default 0; 0.1 sparse, 0.2 dense); in [0, 1); NaN = default
- **`h_clutter`** *(optional)* — Clutter height in [m] of InF-SH (default 2 m) and InF-DH (6 m), >= 0; NaN =
  default. The other models ignore the clutter arguments, which are validated for every model
- **`rnd_g`** *(optional)* — Normal(0,1) value per link, G of eq. (7.6-18); required with `los_soft`; `[n_link]`
- **`center_frequency`** *(optional)* — Carrier frequencies in [Hz], finite and > 0; required with `los_soft`;
  `[n_freq]`

### Outputs:
- **`p_los`** — LOS probability in [0, 1]; `[n_link]`
- **`los_soft`** *(optional)* — Soft LOS state in [0, 1]; `> 0.5` exactly when `rnd_g > -F(p_los)` of eq. (7.6-18),
  which has the probability `p_los`; `[n_link, n_freq]`

### See also:
- [o2i_loss](#o2i_loss) (indoor distance for `d2d_out`)
- [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand), [sos_randn](sum-of-sinusoids-random-number-generator.md#sos_randn) (spatially consistent LOS-state draw and `rnd_g`)

---
<a id="lsp_generate"></a>
## lsp_generate
Large-scale parameters of a set of links from normalized random values

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#lsp_generate) · [Python](../python/geometry-based-stochastic-model.md#lsp_generate)

- Generates the 8 LSPs of TR 38.901 clause 7.5 step 4 per link and frequency from the random values of
  [gbsm_draw](#gbsm_draw) (`"lsf"`), and on request the absolute TOA offset, the ground permittivity and the EoD offset
- Mean `mu` and standard deviation `sigma` in the log domain (lg for DS and the angular spreads, dB for KF, SF and
  XPR), with the key prefix `P` (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), `d = max(d2D, 1e-4)`,
  `h = max(hBS, 1e-4)`, `a = atan(h/d)` and `f' = max(center_frequency/1e9, LSP_fc_min)` in GHz:
  - `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`
  - `sigma = max(0, max(0, P_sigma + P_delta·lg(P_omega + f')) + P_kappa·lg(d) + P_tau·lg(h) + P_beta·lg(a))`
  - SF has the mean 0. DS adds `lg(DS_VS_a·DS_VS + DS_VS_b)` if `DS_VS > 0` (V/S of the InF hall, TR 38.901 Table
    7.5-6 Part-3). ESD adds `ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)`
    and is limited below by `ES_D_mu_min` (TR 38.901 Tables 7.5-7 to 7.5-12)
  - The multiplying coefficients must be finite; a frequency term with `P_gamma` or `P_delta` not 0 requires
    `P_omega + f' > 0`
- `lsp = 10^(mu + sigma·x)` for DS and the angular spreads, `10^(0.1·(mu + sigma·x))` for KF, SF and XPR; `x` are
  the columns 0 to 7 of `rnd`, cross-correlated by a root of the LSP cross-correlation matrix of `scenpar`, the same
  for all frequencies (TR 38.901 clause 7.6.5). `rnd = 0` gives the median LSPs
- Absolute TOA offset (TR 38.901 clause 7.6.9), active if `absTOA_mu > -30`: `10^(absTOA_mu + absTOA_sigma·x)`
  with column 8 of `rnd`, bounded by `2·absTOA_Lmax/c` if `absTOA_Lmax > 0`
- The InF and InH NLOS files set `absTOA_Lmax`, and the InF files `DS_VS`, for the hall or office of their
  calibration scenario; for another hall set these keys in `scenpar`
- Ground permittivity: QuaDRiGa's fits for dry, medium and wet ground, interpolated linearly over column 9 of `rnd`
  (0 dry, 0.5 medium, 1 wet) with `center_frequency` not limited by `LSP_fc_min`; they are not the materials of TR
  38.901 Table 7.6.8-1
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e +
  EoD_offset_e_gamma·lg f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) +
  EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if `EoD_offset_b > 0`, plus
  `(atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi` if `hUT < EoD_offset_hcl`

### Declaration:
```cpp
void quadriga_lib::lsp_generate(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::vec &center_frequency,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::mat &rnd,
    arma::cube &lsp,
    const arma::mat *sf_sigma = nullptr,
    arma::vec *abs_toa = nullptr,
    arma::mat *gr_eps_re = nullptr,
    arma::mat *gr_eps_im = nullptr,
    arma::mat *eod_offset_deg = nullptr,
    arma::cube *lsp_mu = nullptr,
    arma::cube *lsp_sigma = nullptr,
    bool cholesky = true,
    bool clip_spreads = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters; the LSP cross-correlation matrix must be positive definite
- **`center_frequency`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`rnd`** — Random values from [gbsm_draw](#gbsm_draw) (`"lsf"`), finite; column 9 in [0, 1] when the ground permittivity
  is requested; column 10 is not used; `[n_link, 11]`
- **`sf_sigma`** *(optional)* — SF standard deviation in [dB] (e.g. from [path_loss](#path_loss)), finite and >= 0; replaces
  the value from `scenpar`, also in `lsp_sigma`; `[n_link, n_freq]`
- **`cholesky`** *(optional)* — `true`: Cholesky root in the order of TR 38.901 clause 7.5 step 4 (SF, KF, DS, ASD,
  ASA, ESD, ESA; XPR last); `false`: symmetric square root (legacy QuaDRiGa)
- **`clip_spreads`** *(optional)* — `true`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg (TR 38.901 clause 7.5
  step 4); `lsp_mu` and `lsp_sigma` are not affected

### Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in [s], 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA in [deg], 7 XPR;
  `[n_link, n_freq, 8]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s], 0 if not active; `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Relative permittivity of the ground, real part and imaginary
  part >= 0; the complex value of TR 38.901 eq. (7.6-40) is `gr_eps_re - j·gr_eps_im`; `[n_link, n_freq]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg], positive = up; `[n_link, n_freq]`
- **`lsp_mu`** *(optional)* — Mean `mu` in the log domain; required by [ssf_rays](#ssf_rays); `[n_link, n_freq, 8]`
- **`lsp_sigma`** *(optional)* — Standard deviation `sigma` in the log domain, >= 0; required by [ssf_rays](#ssf_rays);
  `[n_link, n_freq, 8]`

### See also:
- [gbsm_draw](#gbsm_draw) (the random values `rnd`)
- [path_loss](#path_loss) (the SF standard deviation `sf_sigma`)
- [ssf_clusters](#ssf_clusters) (the clusters from the LSPs)
- [ssf_rays](#ssf_rays) (takes `lsp`, `lsp_mu` and `lsp_sigma`)

---
<a id="o2i_loss"></a>
## o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#o2i_loss) · [Python](../python/geometry-based-stochastic-model.md#o2i_loss)

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier frequency; the
  caller adds `loss` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table 7.4.3-1),
  `legacy_6GHz` (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz), `car` and `car_metal` (clause 7.4.3.2,
  metallized windows)
- Model of UT `u`: the smallest `j` with `rnd(u, 0) < w_0 + ... + w_j`, else the last model with a weight > 0; a model
  with weight 0 is never chosen
- Indoor distance: the minimum of columns 1 and 2 of `rnd` times `d2d_in_max` (`low`, `high`, `low_A`, clause
  7.4.3.1), column 1 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand) and [sos_randn](sum-of-sinusoids-random-number-generator.md#sos_randn) at the correlation distances of TR 38.901 clause 7.6.3.3: 50 m
  for column 0, 25 m for columns 1 and 2, 10 m for column 3
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`, SMa `low`,
  `high` and `low_A`. SMa has 90 % residential and 10 % commercial buildings (Table 7.2-5) with different maximum
  indoor distances: call `o2i_loss` once per building type

### Declaration:
```cpp
void quadriga_lib::o2i_loss(
    const arma::vec &center_frequency,
    const arma::mat &rnd,
    const std::vector<std::string> &models,
    const arma::vec &weights,
    arma::mat &loss,
    double d2d_in_max = 25.0,
    arma::vec *d2d_in = nullptr,
    arma::uvec *model_index = nullptr);
```

### Inputs:
- **`center_frequency`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`rnd`** — Random values per UT: columns 0 to 2 Uniform in [0, 1] (model choice, two indoor distances), column
  3 Normal(0,1), finite (deviation `sigma_P`, car loss); `[n_ut, 4]`, `n_ut = 0` allowed
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; `[n_model]`
- **`weights`** — Probability of each model, finite and >= 0, sum 1 within 1e-6; `[n_model]`
- **`d2d_in_max`** *(optional)* — Maximum indoor 2D distance in [m], finite and >= 0 (key `O2I_d2D_in_max`; TR: 25 m
  for UMa, UMi and SMa commercial buildings, 10 m for RMa and SMa residential buildings)

### Outputs:
- **`loss`** — Penetration loss in [dB]; `[n_ut, n_freq]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m], 0 for the car models; `[n_ut]`
- **`model_index`** *(optional)* — Chosen model per UT, 0-based index into `models`; `[n_ut]`

### See also:
- [los_probability](#los_probability) (uses the outdoor distance `d2D - d2d_in`)
- [path_loss](#path_loss) (basic path loss, legacy outdoor distance from `d2d_in`)

---
<a id="path_loss"></a>
## path_loss
Path loss and shadow-fading standard deviation of a set of links

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#path_loss) · [Python](../python/geometry-based-stochastic-model.md#path_loss)

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and carrier
  frequency; the TR 38.901 v19.5.0 conf files of the library reproduce Table 7.4.1-1 with them. The O2I
  penetration loss is not included (see [o2i_loss](#o2i_loss)). The models are evaluated as written for any positive
  frequency, also outside their validity range
- Notation: `f = center_frequency/1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance, at
  least 0.1 m; `hBS`, `hUT` the z coordinates of Tx and Rx; `lg = log10`; `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`;
  `sf` the shadow-fading std. Keys in parentheses: the required ones, after the semicolon the optional ones (`X = v`:
  default)
- Models:
  - `constant` (A): `A`
  - `logdist` (A, B, C; SF): `A·lg d3 + B + C·lg f`; `sf = SF`
  - `logdist_simple` (A, B): `A·lg d3 + B`
  - `winner_los` (A1, A2 different, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb = max(hBS, 1.5)`,
    `hm = max(hUT, 1.5)`, `Gj = Bj + Cj·lg f + Dj·lg hb + Ej·lg hm`, breakpoint `bp = 10^((G2 - G1)/(A1 - A2))`;
    `Aj·lg d2 + Gj + Fj·hm` and `sf = sigj` with `j = 1` for `d2 <= bp`, else `j = 2`
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as `winner_los`;
    `(Aj + Ahj·lg hb)·lg d2 + Bj + Cj·lg f + Dj·lg hb + Ej·lg hm + Fj·hm` with `j = 1` for `f < 1.5`, 2 for
    `f < 2`, else 3
  - `winner_pathloss` (A, B, C, X; SF): `A·lg d3 + B + C·lg(f/5) + X`; `sf = SF`
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2): `hb = max(hBS, 1.1·hE,
    0.1)`, `hm = max(hUT, 1.1·hE, 0.1)`, `dBP = E·f·(hb - hE)·(hm - hE)`, `dBP3 = sqrt(dBP² + (hb - hm)²)`;
    `PL1(d3)`, `sf = sig1` for `d2 <= dBP`, else `PL1(dBP3) + A2·lg(d3/dBP3)`, `sf = sig2`
  - `tripple_slope`, spelled as in QuaDRiGa (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0,
    hE2 = 0, sig1..sig3): the `dual_slope` rule with the breakpoints `dBP1` of `E1, hE1` (at most `dBP2`) and `dBP2`
    of `E2, hE2`, the slopes `A1`, `A2`, `A3` and `sf = sig1`, `sig2`, `sig3`
  - `nlos` (An, Bn; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, Ax, Bx, Cx, sig):
    `hb = max(hBS, 0.1)`, `hm = max(hUT, 0.1)`; `An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb +
    D2n·lg(hb)/hb² + D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) +
    G1n·(lg(G2n·hm))²`; with `LOSmax != 0` (which requires the `dual_slope` keys) at least the `dual_slope` loss; with
    `Ax`, `Bx` or `Cx` at least `Ax·lg d3 + Bx + Cx·lg f` (e.g. the InF-SL term of InF-DL); `sf = sig`
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS/d3, 1)`; `+Inf` for `sa <= 0`, else
    `A·lg d3 + B + C·lg f + D·lg(asin(sa))`, plus the zenith attenuation by atmospheric gases over `sa` (1 to 100 GHz,
    constant beyond) if `usePLa != 0`
  - empty `pl_model`: 0
- Where the model gives no `sf` (an `SF` or `sig` key not set, or a model without one): the SF std of [lsp_generate](#lsp_generate)
  at the geometric distance, so that passing `sf_sigma` to [lsp_generate](#lsp_generate) is neutral
- `hE_rand != 0` with `rnd_he` given: the random environment height of TR 38.901 Table 7.4.1-1 note 1, drawn from
  `rnd_he`, which makes it spatially consistent at the decorrelation distance `SF_lambda` (the TR draws it
  independently per link); the UT height is then clamped at `hE + 0.1` instead of `1.1·hE`. Along a track the loss
  beyond the breakpoint steps where hE changes; evaluate `rnd_he` once per track segment to avoid this
- Distances: the geometric distances of TR 38.901 eq. (7.4-1). With `d2d_in` (legacy QuaDRiGa), links with
  `d2d_in > 0` use the outdoor part of the distances instead, at least 10 m
- A required key that is missing throws, naming `A1` for "A1 or A" and `E1` for "E1 or E"; every key that the model
  reads must have one finite value; the breakpoint factors `E`, `E1`, `E2` and `G2n` must be > 0

### Declaration:
```cpp
void quadriga_lib::path_loss(
    const std::string &pl_model,
    const std::unordered_map<std::string, std::vector<double>> &plpar,
    const std::unordered_map<std::string, double> &scenpar,
    const arma::vec &center_frequency,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &pl,
    arma::mat *sf_sigma = nullptr,
    const arma::vec *d2d_in = nullptr,
    const arma::vec *rnd_he = nullptr);
```

### Inputs:
- **`pl_model`** — Path-loss model, value of `PL_model` from [conf_file_read](#conf_file_read); empty = zero loss
- **`plpar`** — Path-loss parameters, keys without "PL_", from [conf_file_read](#conf_file_read); keys not read are ignored
- **`scenpar`** — Scenario parameters; the SF std terms and `LSP_fc_min` are used
- **`center_frequency`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m] of the UT of each link (e.g. from [o2i_loss](#o2i_loss)), finite
  and >= 0; given = legacy outdoor distance; 0 = outdoor link; `[n_link]`
- **`rnd_he`** *(optional)* — Uniform(0,1) value per link for the random environment height, column 10 of
  [gbsm_draw](#gbsm_draw) (`"lsf"`); `[n_link]`

### Outputs:
- **`pl`** — Path loss in [dB]; `[n_link, n_freq]`
- **`sf_sigma`** *(optional)* — Shadow-fading standard deviation in [dB]; `[n_link, n_freq]`

### See also:
- [lsp_generate](#lsp_generate) (takes `sf_sigma`)
- [o2i_loss](#o2i_loss) (penetration loss and indoor distance)

---
<a id="scenpar_validate"></a>
## scenpar_validate
Validate scenario parameters and complete them with default values

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#scenpar_validate) · [Python](../python/geometry-based-stochastic-model.md#scenpar_validate)

- Checks scenario parameters, e.g. ones built in code, and adds every missing key with its default value, so that
  `scenpar` has all `n_key = 171` keys on return: the 138 numeric keys of QuaDRiGa and 33 keys for TR 38.901 v19.5.0,
  which have no effect at their default of 0. An empty `scenpar` gives the complete default set
- A name that is not one of the keys is an error, also with `validate = false`
- Value checks with `validate = true`; `NaN` is an error for every key:
  - Integer from 1 to 1000: `NumClusters`, `NumSubPaths`
  - Finite and > 0: `r_DS`
  - Finite and >= 0: the 8 decorrelation distances (`DS_lambda` ... `XPR_lambda`), `SC_lambda`,
    `absTOA_lambda`, `PerClusterDS`, `PerClusterDS_min`, `PerClusterAS_D`, `PerClusterAS_A`, `PerClusterES_D`,
    `PerClusterES_A`, `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`, `LSP_fc_min`, `EoD_offset_b`,
    `EoD_offset_hcl`, `PerClusterDS_fc_min`, `PerClusterES_D_scale`, `absTOA_Lmax`, `NF_alpha`, `NF_beta`,
    `SNS_sigma`, `SNS_A`, `SNS_B`, `SNS_R`, `SNS_delta_var`, `PolVar_sigma`, `LOS_state_lambda`,
    `IO_state_lambda`, `O2I_d2D_in_max`, `DS_VS`, `DS_VS_a`, `DS_VS_b`
  - Integer from 0 to 1000: `NumClusters_min`, `NF_k1`
  - In the range from 0 to 1: `SNS_mu`
  - In the range from -1 to 1: the 28 correlations (`asD_ds` to `xpr_esa`)
  - All other keys accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `DS_VS_a` and `DS_VS_b` must not both be 0 if `DS_VS` is > 0
- `LOS_scatter_radius`, `NF_*` and `SNS_*` are checked but not used by any function of this version
- Throws `std::invalid_argument`; a failed call leaves `scenpar` and the outputs untouched

### Declaration:
```cpp
bool quadriga_lib::scenpar_validate(
    std::unordered_map<std::string, double> &scenpar,
    arma::mat *lsp_xcorr = nullptr,
    std::vector<std::string> *key_order = nullptr,
    bool validate = true);
```

### Inputs:
- **`scenpar`** — Numeric scenario parameters; updated in-place: given values are kept, missing keys are
  added with their default; `n_key` entries on return
- **`validate`** *(optional)* — `false`: only complete with defaults, no value checks (`NaN` is accepted)

### Outputs:
- **`lsp_xcorr`** *(optional)* — LSP cross-correlation matrix from the 28 correlation keys, symmetric with unit
  diagonal; order of rows and columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `[8, 8]`
- **`key_order`** *(optional)* — Canonical order of the keys in `scenpar`: the QuaDRiGa order, then the 33 new
  keys; `[n_key]`

### Returns:
- `true` if `lsp_xcorr` is positive definite (all pivots of its Cholesky factorization > 0); `false` also if it
  contains `NaN`; `false` is no error here, but [lsp_generate](#lsp_generate) throws for such parameters

### See also:
- [conf_file_read](#conf_file_read) (read the parameters from a conf file)
- [conf_file_write](#conf_file_write) (applies the same checks before writing)

---
<a id="ssf_clusters"></a>
## ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_clusters) · [Python](../python/geometry-based-stochastic-model.md#ssf_clusters)

- Turns the LSPs of [lsp_generate](#lsp_generate) and the cluster random values of [gbsm_draw](#gbsm_draw) into the clusters of each link,
  by one of three methods:
  - `"38.901"` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters
    25 dB below the strongest get power 0 (no renormalization); the K-factor in the scaling factors is limited
    to >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` at the drop or segment
    start and keep them along the track
  - `"38.901_MF"`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `"quadriga"`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links; delays and angles depend on the frequency list, so keep it fixed to reproduce a seed

| Property | `"38.901"` | `"38.901_MF"` | `"quadriga"` |
| --- | --- | --- | --- |
| Continuous along tracks | no | NLOS only | yes |
| Same delays and angles at all frequencies | n/a | NLOS, or K the same at all frequencies | yes |
| Output spreads equal the input spreads | no | no | DS, and AS below the caps, at one frequency without ground reflection |
| Per-cluster shadowing | yes | yes, per frequency | no |

- `n_slot = NumClusters + G` slots per link (see Terms), `G = 1` if `GR_enabled > 0`. Slot `1 + G + n` belongs to
  column `n` of each block of `rnd` (no sorting). Clusters `n >= n_cluster(l)` (TR clause 7.6.15) get delay 0, power
  0 and the LOS angles. Identify the LOS path by its slot: NLOS clusters of the TR methods can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever
  `lsp` holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; slot 0 holds the geometric LOS direction. Powers are the
  coefficient powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `"38.901"`, `"38.901_MF"` add `abs_toa(l)` to the NLOS
  delays; `"quadriga"` places the first NLOS cluster at `abs_toa(l)` (where `absTOA_mu <= -30`: only where
  `abs_toa(l) > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0: slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`"quadriga"`), with `R²` the mean of `abs(R_par)²` and `abs(R_perp)²` of TR
  eqs. (7.6-38), (7.6-39); the coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarization
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios, plus the ground reflection if enabled; `rnd` must
  be empty, `rnd_freq` and `lsp_anchor` are not used
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario; only their sizes are checked

### Declaration:
```cpp
void quadriga_lib::ssf_clusters(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::cube &lsp,
    const arma::mat &rnd,
    arma::cube &delay,
    arma::cube &power,
    arma::cube &aod,
    arma::cube &eod,
    arma::cube &aoa,
    arma::cube &eoa,
    const std::string &method = "38.901",
    const arma::cube *rnd_freq = nullptr,
    const arma::cube *lsp_anchor = nullptr,
    const arma::mat *eod_offset_deg = nullptr,
    const arma::vec *abs_toa = nullptr,
    const arma::mat *gr_eps_re = nullptr,
    const arma::mat *gr_eps_im = nullptr,
    const std::vector<bool> *o2i = nullptr,
    const arma::u32_vec *n_cluster = nullptr,
    arma::u32_vec *n_active = nullptr);
```

### Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`, `KF_mu`, `absTOA_mu`, `r_DS`,
  `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in [m]; z >= 0 with ground reflection; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[n_link, 3]`
- **`lsp`** — LSPs from [lsp_generate](#lsp_generate); DS, ASD, ASA, ESD, ESA (slices 0, 3 to 6) > 0, KF (slice 1) >= 0;
  `"38.901"`: one frequency, call once per frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [gbsm_draw](#gbsm_draw) (`"cluster"`, `set_id = -1`), columns 0 to N-1 in [0, 1];
  `[n_link, 9·N]`
- **`method`** *(optional)* — `"38.901"` (default), `"38.901_MF"` or `"quadriga"`
- **`rnd_freq`** *(optional)* — Per-frequency cluster sets of [gbsm_draw](#gbsm_draw), slice `k` from `set_id = k`
  (`"38.901"`: one slice, the set of the frequency in `lsp`); required by the TR methods; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz) from [lsp_generate](#lsp_generate); required by
  `"38.901_MF"`; `[n_link, 1, 8]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg] from [lsp_generate](#lsp_generate); used by `"38.901"` and
  `"quadriga"` in NLOS scenarios; omitted = 0; `[n_link, n_freq]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s] from [lsp_generate](#lsp_generate), >= 0; required by `"quadriga"` in
  NLOS scenarios with `absTOA_mu > -30`; omitted = no offset; `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im` from
  [lsp_generate](#lsp_generate), real part >= 1, imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** *(optional)* — O2I flag per link (mean arrival elevation 0); used by `"38.901"` in NLOS scenarios; omitted =
  none; `[n_link]`
- **`n_cluster`** *(optional)* — Active NLOS clusters per link in `[min(max(1, NumClusters_min - 1), N), N]`;
  omitted = N; `[n_link]`

### Outputs:
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_freq]` for `"38.901_MF"`, else `[n_link, n_slot, 1]`
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in [rad], in [-pi, pi); sizes as `delay`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in [rad], in [-pi/2, pi/2]; sizes as `delay`
- **`n_active`** *(optional)* — Number of NLOS clusters with power > 0; `[n_link]`

### See also:
- [gbsm_draw](#gbsm_draw) (the random values `rnd` and `rnd_freq`)
- [lsp_generate](#lsp_generate) (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [ssf_rays](#ssf_rays) (the rays of the clusters)
- [calc_delay_spread](channel-statistics.md#calc_delay_spread), [calc_angular_spreads_sphere](channel-statistics.md#calc_angular_spreads_sphere) (spreads of the output)

---
<a id="ssf_paths"></a>
## ssf_paths
Antenna-free paths of one link at one position

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_paths) · [Python](../python/geometry-based-stochastic-model.md#ssf_paths)

- Computes the antenna-free paths (gains, lengths, polarization matrices, optionally angles and scatterers) of the
  kept rays of `rays` at new Tx and Rx positions (default: the ray-data positions). Kept are LOS and GR rays always
  (also at zero power) and NLOS rays with power > 0 at some frequency; paths in ray order, `ray_ind` maps them to
  the rays (e.g. `rays.tap(ray_ind)`)
- `"phase"`: use the outputs with [get_channels_planar](channel-generation-functions.md#get_channels_planar) and the arrays at the positions of `rays`; the displacement
  phase of TR 38.901 eq. (7.5-22) is in `M`. With the arrays at the new positions the LOS path is no longer
  identified and the displacement phase counts twice, without an error
- `"drift"`: use the outputs at the new positions with [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) or [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq), or
  pass them to [qrt_paths_from_scatterers](channel-functions.md#qrt_paths_from_scatterers); needs the scatterers of [.solve_scatterers](#.solve_scatterers)

| Quantity | `"phase"` | `"drift"` |
| --- | --- | --- |
| Path length | Tx-Rx distance of `rays` + c·delay | Polyline Tx, FBS, LBS, Rx; LOS direct line; GR via the specular point |
| Angles | Ray data | Geometry: towards FBS and LBS, LOS direct line, GR towards the image points |
| `M` | Ray `M` times the displacement phase | Ray `M`; GR recomputed when an end moved |
| `fbs_pos`, `lbs_pos` | `[3, 0]` | Scatterers of the rays; LOS midpoint of the direct line; GR specular point |

- `"drift"` with an end moved (any coordinate differs from `rays`): GR power by the power law of `rays.method`
  (`(d3D / d_GR)²` for the TR methods, 1 for `"quadriga"`), GR `M` the scalar multiple of `diag(R_par, -R_perp)` (TR
  38.901 eqs. (7.6-38), (7.6-39)) that best fits the stored matrix, which replaces a matrix of another form (e.g. a
  conjugated one). Fallback rays have their longer polyline, also at the positions of `rays`
- Path gain per frequency: the coefficient power; with `kf_ratio` LOS and GR are rescaled against NLOS so that the
  K-factor of the coefficient powers changes by `kf_ratio`, LOS + NLOS keep their sum; times `10^(link_gain_dB / 10)`;
  with `oxygen` times the absorption of TR 38.901 clause 7.6.1 (Table 7.6.1-1 at each frequency) over each path's own
  length, without the delay offset τ_Δ of eq. (7.6-1) and at the carrier only (no in-band variation of eq. (7.6-2))
- Validity per moving end, with `b_eff` and `b_med` of [gbsm_rays](#gbsm_rays): `"phase"` up to about `sqrt(λ · b_eff / 8)`,
  typically 0.1 to 1 m at the carrier, about 0.15 to 0.35 m at the edges of a 400 MHz band; `"drift"` up to about
  `0.2 · b_med`, typically 1.5 to 12 m, new segment from `0.5 · b_med`; in `"drift"` each path length changes by at most
  the displacement. [gbsm_single_link_channel](#gbsm_single_link_channel) reports the displacement relative to the limit
- Throws `std::invalid_argument` for invalid input; in `"drift"` also without scatterers, and for a GR ray at moved
  ends without permittivity in `rays` or with an end below z = 0

### Declaration:
```cpp
void quadriga_lib::ssf_paths(
    const gbsm_rays &rays,
    const arma::vec &link_gain_dB,
    arma::mat &path_gain,
    arma::vec &path_length,
    arma::cube &M,
    const std::string &mobility = "auto",
    const arma::vec *tx_pos = nullptr,
    const arma::vec *rx_pos = nullptr,
    const arma::vec *kf_ratio = nullptr,
    arma::vec *aod = nullptr,
    arma::vec *eod = nullptr,
    arma::vec *aoa = nullptr,
    arma::vec *eoa = nullptr,
    arma::mat *fbs_pos = nullptr,
    arma::mat *lbs_pos = nullptr,
    arma::u32_vec *path_type = nullptr,
    arma::uvec *ray_ind = nullptr,
    bool oxygen = false);
```

### Inputs:
- **`rays`** — Ray data of the link
- **`link_gain_dB`** — Link gain (see Terms) at the position in [dB], per frequency; 0 for data from paths; `[n_freq]`
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"` (see Terms)
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx position in [m]; omitted: the positions of `rays`; `[3]`
- **`kf_ratio`** *(optional)* — KF(position) / KF(drop), >= 0, per frequency; omitted: 1; `[n_freq]`
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1

### Outputs:
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`M`** — Polarization transfer matrices, rows as in [get_channels_planar](channel-generation-functions.md#get_channels_planar); `[8, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** *(optional)* — Departure and arrival angles in [rad], GCS; `[n_path]`
- **`fbs_pos`**, **`lbs_pos`** *(optional)* — Scatterer positions in [m]; `[3, n_path]`, `[3, 0]` in `"phase"`
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS; `[n_path]`
- **`ray_ind`** *(optional)* — Index of the ray in `rays`, 0-based; `[n_path]`

### See also:
- [gbsm_rays](#gbsm_rays) (input data, validity scales `b_eff` and `b_med`)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel coefficients of one link over a segment)
- [gbsm_rays_from_paths](#gbsm_rays_from_paths) (ray data from path data, e.g. of a QRT file)
- [.solve_scatterers](#.solve_scatterers) (scatterers for `"drift"`)

---
<a id="ssf_rays"></a>
## ssf_rays
Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_rays) · [Python](../python/geometry-based-stochastic-model.md#ssf_rays)

- Splits the clusters of [ssf_clusters](#ssf_clusters) into rays and writes one [gbsm_rays](#gbsm_rays) object per link: delays, coefficient
  powers, angles, polarization transfer matrices with initial phases, path types and tap indices; `fbs_pos`, `lbs_pos`
  and `fallback` stay empty (see [.solve_scatterers](#.solve_scatterers))
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
  `f' = max(center_frequency/1e9, PerClusterDS_fc_min)`; 3.91 ns where `PerClusterDS` = 0
- `NumSubPaths` other than 20: the offsets of QuaDRiGa-NG (the MATLAB QuaDRiGa fork that uses this library) below 20
  (mean 0, rms 1), Laplacian quantiles at the rms of Table 7.5-3 above 20; no sub-clusters

| Method | `"38.901"`, `"38.901_MF"` | `"quadriga"` |
| --- | --- | --- |
| Frequencies per call | `i_freq` only; slice `i_freq` of `rnd_ray`; `"38.901"` takes one frequency | all; one geometry, per frequency powers, XPR and Jones matrices; the shared "ray" set |
| Ray delays | cluster delay; sub-clusters + 0, 1.28, 2.56 c_DS (Table 7.5-5) | cluster delay + 2·c_DS·(u - mean of u in the cluster), u of block 14; negative delays set to 0 |
| Taps | sub-cluster of the two strongest clusters (`NumSubPaths` = 20), else cluster | cluster |
| c_DS, cluster ESD | at frequency `i_freq` | mean over the frequencies |

| Option | Default (TR) | Alternative (QuaDRiGa-NG) |
| --- | --- | --- |
| `angle_model` | `"additive"`: offsets added to azimuth and elevation, elevation folded at the poles (step 7) | `"rotation"`: offset pattern rotated to the cluster direction; equal at elevation 0 |
| `pol_model` | `"38.901"`: XPR and four initial phases per ray (steps 9, 10) | `"quadriga"`: rotation polarization model of QuaDRiGa, one phase, XPR per ray and per cluster |
| `pol_variability` | `false` | `true`: clause 7.6.16 on the NLOS rays with std `PolVar_sigma` (key default 0: no effect); `pol_model = "38.901"` only |

- Sub-clusters (step 11): the two clusters of largest power at frequency `i_freq` (equal powers: lower index first),
  rays at the positions 0-9, 10-15 and 16-19 (TR rays 1-8, 19, 20; 9-12, 17, 18; 13-16); every tap is a contiguous
  block of rays
- `pol_model = "quadriga"`: the matrix refers to the end with the larger z, then x, then y (coordinates within 1 mm
  count as equal; the Tx if all tie), so UTs dropped separately at equal heights may get transposed matrices

### Declaration:
```cpp
void quadriga_lib::ssf_rays(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::vec &center_frequency,
    const arma::cube &lsp,
    const arma::cube &lsp_mu,
    const arma::cube &lsp_sigma,
    const arma::cube &delay,
    const arma::cube &power,
    const arma::cube &aod,
    const arma::cube &eod,
    const arma::cube &aoa,
    const arma::cube &eoa,
    const arma::cube &rnd_ray,
    std::vector<gbsm_rays> &rays,
    const std::string &method = "38.901",
    arma::uword i_freq = 0,
    const std::string &angle_model = "additive",
    const std::string &pol_model = "38.901",
    const arma::mat *gr_eps_re = nullptr,
    const arma::mat *gr_eps_im = nullptr,
    bool pol_variability = false,
    bool random_phase = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumSubPaths`, the `PerCluster*` keys, `GR_enabled`,
  `PolVar_sigma`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`center_frequency`** — Carrier frequencies in [Hz]; `[n_freq]`
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [lsp_generate](#lsp_generate); used: the ESD
  mean and the XPR mean and std, the XPR LSP for `pol_model = "quadriga"`; `[n_link, n_freq, 8]`
- **`delay`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays and angles from [ssf_clusters](#ssf_clusters)
  (`n_slot` slots); `[n_link, n_slot, n_freq]` for `"38.901_MF"`, else `[n_link, n_slot, 1]`
- **`power`** — Cluster powers from [ssf_clusters](#ssf_clusters); `[n_link, n_slot, n_freq]`
- **`rnd_ray`** — `"ray"` sets from [gbsm_draw](#gbsm_draw) (`n_gen` columns); TR methods: slice `k` = set `k`,
  `[n_link, n_gen, n_freq]`; `"quadriga"`: the shared set, `[n_link, n_gen, 1]`
- **`method`** *(optional)* — Method of [ssf_clusters](#ssf_clusters): `"38.901"`, `"38.901_MF"` or `"quadriga"`
- **`i_freq`** *(optional)* — TR methods: index into `center_frequency`, 0-based; `"quadriga"`: must be 0
- **`angle_model`**, **`pol_model`** *(optional)* — See the option table
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity from [lsp_generate](#lsp_generate); required if
  `GR_enabled`, else ignored; `[n_link, n_freq]`
- **`pol_variability`** *(optional)* — TR 7.6.16 polarization power variability, see the option table
- **`random_phase`** *(optional)* — `false`: all initial phases 0

### Output:
- **`rays`** — Ray data, one entry per link, TR methods at frequency `i_freq` only; resized to `n_link`, existing
  entries are overwritten (their memory is reused); `[n_link]`

### See also:
- [ssf_clusters](#ssf_clusters) (the clusters)
- [gbsm_rays](#gbsm_rays) (attributes of the output)
- [ssf_paths](#ssf_paths) (paths of one link)

---
<a id=".solve_scatterers"></a>
## .solve_scatterers
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_scatterers) · [Python](../python/geometry-based-stochastic-model.md#ssf_scatterers)

- Member of [gbsm_rays](#gbsm_rays): places the scatterers that the mobility `drift` needs, in place
- NLOS rays with power > 0 at some frequency: FBS on the departure ray and LBS on the arrival ray at equal
  distances from their end, so that the polyline Tx, FBS, LBS, Rx has the length d3D + c·delay (d3D the Tx-Rx
  distance) and the angles are exact
- Where no such pair exists with both distances at least `d_min`, both scatterers are placed at `d_min` along the ray's
  own directions and `fallback` is 1: the polyline is longer than the ray, `delay` keeps its value. With `d_min` = 2 m
  this applies to up to about 23 % of the rays (most in the TR methods without absolute TOA) and lengthens them by up to
  25 ns (median below 5 ns), the DS of a link by up to 3 ns; `b_med` can then equal `d_min`
- LOS ray and NLOS rays with zero power at all frequencies: the midpoint of the direct line; GR ray: the specular
  point on z = 0, which requires both ends at z >= 0
- Writes `fbs_pos`, `lbs_pos` and `fallback` of all rays and overwrites given scatterers; nothing else changes;
  afterwards `b_med()` and `b_eff()` give the validity scales of [ssf_paths](#ssf_paths)
- One link per call; batch callers loop over the objects of [ssf_rays](#ssf_rays). Throws `std::invalid_argument` with the
  `is_valid()` message for an invalid object, which stays unchanged

### Declaration:
```cpp
void quadriga_lib::gbsm_rays::solve_scatterers(double d_min = 2.0);
```

### Input:
- **`d_min`** *(optional)* — Minimum distance of the scatterers to both ends in [m], finite and > 0

### See also:
- [gbsm_rays](#gbsm_rays) (attributes and members)
- [ssf_paths](#ssf_paths) (mobility `drift` needs the scatterers)
