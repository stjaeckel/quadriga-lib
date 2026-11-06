# Geometry-based stochastic model

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-geometry-based-stochastic-model

| Function | Summary |
| --- | --- |
| [conf_file_list](#conf_file_list) | List the scenarios defined by the conf files in a set of directories |
| [conf_file_read](#conf_file_read) | Read and validate a QuaDRiGa scenario configuration file |
| [conf_file_write](#conf_file_write) | Write a QuaDRiGa scenario configuration file |
| [gbsm_draw](#gbsm_draw) | Spatially correlated normalized random values of one generator set at a set of links |
| [gbsm_init_phases](#gbsm_init_phases) | Draw the random phases of one generator set of a BS site and scenario |
| [gbsm_rays](#gbsm_rays) | Ray data of one link (geometry-based stochastic model) |
| [gbsm_rays_from_paths](#gbsm_rays_from_paths) | Ray data of one link from spherical path data (e.g. read with qrt_file_read) |
| [gbsm_single_link_channel](#gbsm_single_link_channel) | Channel coefficients of one link at one frequency over one segment (geometry-based stochastic model) |
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
functions of this section implement 3GPP TR 38.901 clause 7.5 steps 1 to 7 (path loss, LOS probability, O2I
penetration loss, large-scale parameters (LSPs) and clusters) with the spatial consistency of clause 7.6.3. The ray
level (steps 8 to 12) follows in a later version.

**Scenario parameters.** A scenario is a set of numeric keys (`scenpar`), a path-loss model with its parameters and
optional short names, read from a conf file by [conf_file_read](#conf_file_read). 23 files for TR 38.901 v19.5.0 come with the
library, in `data/config` of the source tree and in `share/quadriga-lib/config` of an installed package;
[conf_file_list](#conf_file_list) finds them and [conf_file_write](#conf_file_write) writes them. Every function that takes `scenpar` completes it
with the defaults of missing keys and validates it with [scenpar_validate](#scenpar_validate), whose messages pass through; an invalid
value of a key that the function does not use throws as well.

**Workflow.** Per drop: [o2i_loss](#o2i_loss) gives the penetration loss and the indoor distance of each indoor UT, then
[los_probability](#los_probability) the LOS probability at the outdoor distance `d2D - d2D-in`. The caller draws the LOS, indoor and
O2I states with [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand) / [sos_randn](sum-of-sinusoids-random-number-generator.md#sos_randn) at the correlation distances of TR 38.901 clause 7.6.3.3; the keys
`LOS_state_lambda` and `IO_state_lambda` apply to all links of a scenario and are stated in the NLOS and in the O2I
files (0 in the other files means "not stated here"). Per BS site and scenario: [gbsm_init_phases](#gbsm_init_phases) draws the
phases of each random set once (`"lsf"`; `"cluster"` and `"ray"` with `set_id = -1` and one set per carrier
frequency), and [gbsm_draw](#gbsm_draw) evaluates them at the links. Then [path_loss](#path_loss) gives the path loss and the
shadow-fading (SF) standard deviation, [lsp_generate](#lsp_generate) the LSPs of each link and frequency, and [ssf_clusters](#ssf_clusters)
turns the LSPs and the cluster random values into cluster delays, powers and angles.

**Random sets.** The spatially correlated random values come in sets of generators, one column of `rnd` each;
`N = NumClusters - 1`, column `b·N + n` of a `"cluster"` set belongs to block `b` and NLOS cluster `n`:

| `set_type`, `set_id` | `n_gen` | Columns and distributions | Decorrelation distance | Mirror pairs | Used by |
| --- | --- | --- | --- | --- | --- |
| `"lsf"` | 11 | 0 to 7 DS, KF, SF, ASD, ASA, ESD, ESA, XPR, Normal(0,1); 8 absolute TOA, Normal(0,1); 9 ground permittivity, 10 UMa environment height hE, Uniform(0,1) | `DS_lambda` ... `XPR_lambda` (0 to 7), `absTOA_lambda` (8), `SC_lambda` (9), `SF_lambda` (10) | 3 and 4, 5 and 6 | [lsp_generate](#lsp_generate) `rnd`, [path_loss](#path_loss) `rnd_he` (column 10) |
| `"cluster"`, -1 | `9·N` | block 0 cluster delay, Uniform(0,1); 1, 2 sign of AOD, AOA; 3, 4 sign of ZOD, ZOA; 5, 6 offset of AOD, AOA; 7, 8 offset of ZOD, ZOA, Normal(0,1) | `SC_lambda` | blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 | [ssf_clusters](#ssf_clusters) `rnd` |
| `"cluster"`, `k` | `N` | cluster shadowing of frequency `k`, Normal(0,1) | `SC_lambda` | none | [ssf_clusters](#ssf_clusters) `rnd_freq`, slice `k` |
| `"ray"`, -1 | `14·N·M + N` | blocks 0 to 14 (below) | `SC_lambda`; block 14: $\lambda_\tau$ | below | [ssf_rays](#ssf_rays) `rnd_ray`, method `"quadriga"` |
| `"ray"`, `k` | `14·N·M + N` | blocks 0 to 14 (below), block 14 unused | `SC_lambda` | below | [ssf_rays](#ssf_rays) `rnd_ray`, slice `k`, TR methods |

A `"ray"` set holds, with `M = NumSubPaths`, block `b` up to 12 for ray `m` of NLOS cluster `n` in column
`b·N·M + n·M + m`, block 13 in column `13·N·M + n` and block 14 in column `13·N·M + N + n·M + m`; all blocks are
Normal(0,1) except block 14, Uniform(0,1):

| Block | Random value | Mirror pair | Used by |
| --- | --- | --- | --- |
| 0, 1 | coupling rank of AOD, AOA (TR 38.901 clause 7.5 step 8) | 0 and 1 | all methods |
| 2, 3 | coupling rank of ZOD, ZOA | 2 and 3 | all methods |
| 4 | XPR per ray | none | both `pol_model` |
| 5 | initial phase θθ (`pol_model = "quadriga"`: its only phase) | none | both `pol_model` |
| 6, 7 | initial phases θφ, φθ | 6 and 7 | `pol_model = "38.901"` |
| 8 | initial phase φφ | none | `pol_model = "38.901"` |
| 9 | polarisation power variability θθ (TR 38.901 clause 7.6.16) | none | `pol_variability` |
| 10, 11 | the same, θφ and φθ | 10 and 11 | `pol_variability` |
| 12 | the same, φφ | none | `pol_variability` |
| 13 | XPR per cluster (`N` columns) | none | `pol_model = "quadriga"` |
| 14 | ray delay | none | method `"quadriga"` |

Block 14 of the shared `"ray"` set has the decorrelation distance
$\lambda_\tau = 1.6 \cdot c \cdot \bar{c}_\mathrm{DS} / c_\mathrm{ASA}$, limited to 0.1 m to 10 km, with
$\bar{c}_\mathrm{DS}$ the mean cluster delay spread (TR 38.901 Table 7.5-6) over the frequencies passed to
[gbsm_draw](#gbsm_draw) and $c_\mathrm{ASA}$ the cluster ASA in rad.

The Tx side of a phase cube (slice 1) exchanges the columns of each mirror pair, so that in dual mobility (Tx and Rx
both move) the departure quantities at the Tx are the arrival quantities of the Tx as a second device. A
decorrelation distance of 0 is used as 0.1 m, except for $\lambda_\tau$. Only the `"lsf"` columns 0 to 8 take part
in the bias removal. A set never changes size: new random quantities come in new sets (a new `set_type` or
`set_id`).

**Conventions.** Positions are in m with the rows `{x, y, z}`: `rx_pos` is `[n_link, 3]`, `tx_pos` is `[1, 3]`
(one Tx for all links) or `[n_link, 3]`; heights are the z coordinates. In 2D mode (`horizontal = true`, the TR
behaviour) the random values are correlated in the x-y plane, and `floor_id` separates the floors, which are
uncorrelated (TR 38.901 Table 7.5-6 note 3). In 3D mode (`horizontal = false`, legacy QuaDRiGa) the height separates
the floors, so UTs on different floors are correlated by the ACF of their height difference; the per-link statistics
are the same in both modes. Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default and QuaDRiGa's is
available through an argument where it is still wanted; where QuaDRiGa's behaviour was an error, only the corrected
rule is implemented. Outputs must not share memory with the inputs or with each other unless stated otherwise.
Units: frequencies in Hz, path loss in dB, DS in s, KF, SF and XPR as linear power ratios, spreads in degrees.

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
  independent of the global locale; `Inf`, `+Inf` and `-Inf` in any case; a number below the smallest subnormal
  reads as 0
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
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
- `true` if `lsp_xcorr` is positive definite; a matrix that is not is no error

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
- Numbers have up to 15 significant digits, or 16 or 17 where 15 do not read back to the same double, independent of
  the global locale; `Inf`, `-Inf` and `-0` are kept
- `write_defaults = false`: only keys whose value differs from the default (bit for bit); a section without a
  written key is left out
- All inputs are validated before the file is opened: the checks of [scenpar_validate](#scenpar_validate); `pl_model`, the keys of
  `plpar` (other than `model`) and the short names are names, the short names distinct; every `plpar` key has at
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
  height, MT height in m; columns: lower and upper end, one value is written if both are equal; a row with `NaN`
  is not written; empty: no validity range; `[4, 2]`
- **`write_defaults`** *(optional)* — `true`: write all keys; `false`: write only keys that differ from their default

### See also:
- [conf_file_read](#conf_file_read) (read the file back)

---
<a id="gbsm_draw"></a>
## gbsm_draw
Spatially correlated normalized random values of one generator set at a set of links

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_draw) · [Python](../python/geometry-based-stochastic-model.md#gbsm.draw)

- Evaluates the phases of [gbsm_init_phases](#gbsm_init_phases) at the links, one column per generator; columns, distributions and
  decorrelation distances: see the set table in the section description. The value of a link depends neither on the
  other links of the call nor on the other columns; in single mobility it does not depend on `tx_pos`
- Dual mobility: the Tx is a second device on the same maps, its side of each generator uses the mirror generator.
  Every column keeps its distribution; swapping `tx_pos` and `rx_pos` exchanges the columns of each mirror pair and
  keeps the self-mirrored columns, exactly. The ray delays (block 14 of the shared `"ray"` set) change 2 to 8 times
  faster with a Tx move than plane waves at the Tx would (one decorrelation distance, from the cluster ASA)
- Floors (2D mode): links with equal `floor_id` share one field; fields of different floors are uncorrelated at
  equal horizontal positions, exactly within a block of `n_coeff` consecutive IDs, otherwise up to a random residual
  of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation (AVX2 kernel of [sos_val](sum-of-sinusoids-random-number-generator.md#sos_val))
- A set never changes size. `rnd` may share memory with `tx_pos` or `rx_pos`; an empty set (`NumClusters = 1`) accepts
  any empty `phase`

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
    const arma::uvec *floor_id = nullptr,
    const arma::vec *center_frequency = nullptr,
    bool dual_mobility = false,
    bool horizontal = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters; decorrelation distances, `NumClusters`, `NumSubPaths`, cluster DS and ASA
- **`set_type`** — Set type of [gbsm_init_phases](#gbsm_init_phases)
- **`phase`** — Phases from [gbsm_init_phases](#gbsm_init_phases), finite; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility, always validated; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`, `n_link = 0` allowed
- **`set_id`** *(optional)* — Set of [gbsm_init_phases](#gbsm_init_phases); ignored for `"lsf"`
- **`preset`** *(optional)* — Preset of [gbsm_init_phases](#gbsm_init_phases); another preset with the same `n_coeff` is not detected
- **`floor_id`** *(optional)* — Floor index of the UT per link, 0-based (`n_fl - 1` for the UT heights of TR 38.901
  Table 7.2-1, 0 for outdoor UTs); omitted = floor 0; 2D mode only; `[n_link]`
- **`center_frequency`** *(optional)* — Carrier frequencies in [Hz]; required by the shared `"ray"` set, whose block 14
  uses their mean cluster delay spread: pass the same list as to [ssf_rays](#ssf_rays) (another list gives other `quadriga`
  rays); ignored and not checked for all other sets; `[n_freq]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move
- **`horizontal`** *(optional)* — `true`: 2D mode (TR 38.901); `false`: 3D mode (legacy QuaDRiGa)

### Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

### See also:
- [lsp_generate](#lsp_generate) (the large-scale parameters from the `"lsf"` set)
- [path_loss](#path_loss) (column 10 of the `"lsf"` set as `rnd_he`)
- [ssf_clusters](#ssf_clusters) (the clusters from the `"cluster"` sets)
- [ssf_rays](#ssf_rays) (the rays from the `"ray"` sets)

---
<a id="gbsm_init_phases"></a>
## gbsm_init_phases
Draw the random phases of one generator set of a BS site and scenario

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_init_phases) · [Python](../python/geometry-based-stochastic-model.md#gbsm.init_phases)

- Draws the phases of one random set (set table in the section description) for [gbsm_draw](#gbsm_draw); store `phase` or the
  seed to evaluate the same fields again along tracks or on maps
- The phases are a fixed function of `seed`, `set_type` and `set_id`, equal on all platforms (bias-removed `"lsf"`
  phases within about 2e-14 rad); the sets are independent; each scenario of a site needs its own seed
- `"lsf"` bias removal (`remove_bias`, at least 3 rows in `rx_pos`): [sos_remove_bias](sum-of-sinusoids-random-number-generator.md#sos_remove_bias) sets the sample mean of
  columns 0 to 8 over the positions to 0 (with `tx_pos` in dual mobility; 2D mode: one floor), removing variance
- TR 38.901 clause 7.4.4 literally: preset `Exp256` without bias removal; `NumClusters = 1`: all sets but `"lsf"` empty

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
- **`seed`** *(optional)* — Seed of the site and scenario, >= -1; -1 = system random device, new phases on each call
- **`set_id`** *(optional)* — `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset); sets `n_coeff`
- **`tx_pos`**, **`rx_pos`** *(optional)* — `"lsf"` only: positions in [m] for the bias removal, as in [gbsm_draw](#gbsm_draw)
- **`dual_mobility`**, **`remove_bias`**, **`horizontal`** *(optional)* — `"lsf"`: Tx and Rx both move; remove the
  sample mean; 2D bias removal (TR 38.901), `false`: 3D (legacy QuaDRiGa)

### Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 0 = Rx side, slice 1 = Tx side; `[n_coeff, n_gen, 2]`

### See also:
- [gbsm_draw](#gbsm_draw) (evaluates the phases at the links)
- [sos_remove_bias](sum-of-sinusoids-random-number-generator.md#sos_remove_bias) (the bias removal of one generator)

---
<a id="gbsm_rays"></a>
## gbsm_rays
Ray data of one link (geometry-based stochastic model)

Not in: MATLAB, Python

- Antenna-free rays of one link at one drop: delays, powers, angles and polarization transfer matrices, with the
  positions the data refer to; written by [ssf_rays](#ssf_rays) and [gbsm_rays_from_paths](#gbsm_rays_from_paths), completed by
  [.solve_scatterers](#.solve_scatterers), read by [ssf_paths](#ssf_paths) and [gbsm_single_link_channel](#gbsm_single_link_channel), which call `is_valid()` first
- Plain data: the class does not normalize or reorder; later versions append attributes, so set them by name
- An object has scatterers when `fbs_pos` has `n_ray` columns; every object with `n_ray = 0` has them
- Rays from several sources in one object: at most one LOS and one GR ray, with tap index 0 and 3; the other rays
  take any other tap indices; one `method` for all rays; `.solve_scatterers` overwrites given scatterers, so solve
  rays without scatterers in a separate object and concatenate the attributes

### Attributes:
| Attribute | Size | Meaning | Set by |
| --- | --- | --- | --- |
| `method` | string | `"38.901"`, `"38.901_MF"` or `"quadriga"`; GR power law and default mobility | all |
| `tx_pos`, `rx_pos` | `[3]` | Tx and Rx position the ray data refer to in [m] | all |
| `center_frequency` | `[n_freq]` | Carrier frequencies in [Hz] | all |
| `delay` | `[n_ray]` | Delays in [s] relative to the LOS delay `abs(rx_pos - tx_pos) / c` | all |
| `power` | `[n_ray, n_freq]` | Coefficient powers (linear) without link gain; data from paths: path gain | all |
| `aod`, `eod`, `aoa`, `eoa` | `[n_ray]` | Departure and arrival azimuth and elevation in [rad], GCS | all |
| `M` | `[8, n_ray, n_freq]` | Polarization transfer matrices incl. initial phases, rows as in [get_channels_planar](channel-generation-functions.md#get_channels_planar) | all |
| `path_type` | `[n_ray]` | 0 = LOS, 1 = GR, 2 = NLOS | all |
| `tap` | `[n_ray]` | Tap index; rays with equal index form one tap | all |
| `gr_eps_re`, `gr_eps_im` | `[n_freq]` or empty | Ground permittivity `gr_eps_re - j·gr_eps_im`; needed where a GR ray is recomputed | [ssf_rays](#ssf_rays), [gbsm_rays_from_paths](#gbsm_rays_from_paths) |
| `fbs_pos`, `lbs_pos` | `[3, n_ray]` or empty | First- and last-bounce scatterer positions in [m] | [.solve_scatterers](#.solve_scatterers), [gbsm_rays_from_paths](#gbsm_rays_from_paths) |
| `fallback` | `[n_ray]` or empty | 1 = ray lengthened by the solver fallback | [.solve_scatterers](#.solve_scatterers) |

### Member functions:
| Member | Returns |
| --- | --- |
| `n_ray()` | Number of rays, length of `delay` |
| `n_freq()` | Number of frequencies, length of `center_frequency` |
| `empty()` | `true` if `n_ray() == 0` |
| `is_valid()` | Empty string if the object is valid, else the message of the first failed check |
| `b_med(i_freq)` | `[2]`, Tx and Rx in [m]: lower power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS rays with power > 0 at frequency `i_freq` (0-based); NaN without scatterers or without NLOS power |
| `b_eff(i_freq)` | `[2]`, Tx and Rx in [m]: $b_\text{eff} = \left(\sum P / \sum (P / b^2)\right)^{1/2}$ over the rays with power $P > 0$, with $b$ the end-to-scatterer distance (NLOS), d3D (LOS) or the GR path length (GR); NaN without scatterers or power |

### See also:
- [ssf_rays](#ssf_rays) (ray data of a set of links)
- [ssf_paths](#ssf_paths) (paths of one link at one position)

---
<a id="gbsm_rays_from_paths"></a>
## gbsm_rays_from_paths
Ray data of one link from spherical path data (e.g. read with qrt_file_read)

Not in: MATLAB, Python

- Delays from the path lengths, angles from the scatterer positions, power = path gain
- Fills a pre-initialised object in place; use the result with `link_gain_dB = 0` in [ssf_paths](#ssf_paths) and
  [gbsm_single_link_channel](#gbsm_single_link_channel)

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
    const arma::uvec *path_type = nullptr,
    const arma::uvec *tap = nullptr,
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
- **`method`** *(optional)* — Value of `rays.method`: GR power law and default mobility
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS; omitted: LOS where FBS = LBS on the direct
  line, else NLOS; `[n_path]`
- **`tap`** *(optional)* — Tap index; omitted: one tap per path; `[n_path]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity; needed where a GR path is recomputed;
  `[n_freq]`

### Output:
- **`rays`** — Ray data with `n_ray = n_path`, see [gbsm_rays](#gbsm_rays)

### See also:
- [qrt_file_read](channel-functions.md#qrt_file_read) (reads path data from QRT files)
- [ssf_paths](#ssf_paths) (paths of one link at one position)

---
<a id="gbsm_single_link_channel"></a>
## gbsm_single_link_channel
Channel coefficients of one link at one frequency over one segment (geometry-based stochastic model)

Not in: MATLAB, Python

- Per snapshot [get_channels_planar](channel-generation-functions.md#get_channels_planar) (`"phase"`, arrays at the positions of `rays`) or
  [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (`"drift"`) with the paths of [ssf_paths](#ssf_paths)
- `"taps"`: rays summed per tap, LOS and GR taps always present, tap count fixed per call; `"rays"`: one path per
  ray
- Validity: see [ssf_paths](#ssf_paths); the channel's `par` reports the largest displacement relative to the limit

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
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at `rays.center_frequency(i_freq)` in [dB]; `[n_snap]` or `[1]`
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"` (phase for the TR methods, drift for quadriga)
- **`output`** *(optional)* — `"taps"` or `"rays"`
- **`i_freq`** *(optional)* — Frequency index in `rays.center_frequency`, 0-based
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx positions in [m]; omitted: the positions of `rays`;
  `[n_snap, 3]` or `[1, 3]`
- **`tx_orientation`**, **`rx_orientation`** *(optional)* — Euler angles (bank, tilt, heading) in [rad]; omitted: 0;
  `[n_snap, 3]` or `[1, 3]`
- **`kf_ratio`** *(optional)* — K(snapshot) / K(drop); omitted: 1; `[n_snap]` or `[1]`
- **`oxygen`** *(optional)* — TR 7.6.1 oxygen absorption from each path's own length
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays

### Returns:
- Channel object with `n_snap` snapshots

### See also:
- [ssf_paths](#ssf_paths) (paths of one link at one position)
- [gbsm_rays](#gbsm_rays) (input data)

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
- **`d2d_out`** — 2D outdoor distance in [m], finite and >= 0: `d2D - d2D-in` for O2I UTs, clamped at 0 by the
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
- **`lsp_mu`** *(optional)* — Mean `mu` in the log domain; `[n_link, n_freq, 8]`
- **`lsp_sigma`** *(optional)* — Standard deviation `sigma` in the log domain, >= 0; `[n_link, n_freq, 8]`

### See also:
- [gbsm_draw](#gbsm_draw) (the random values `rnd`)
- [path_loss](#path_loss) (the SF standard deviation `sf_sigma`)
- [ssf_clusters](#ssf_clusters) (the clusters from the LSPs)

---
<a id="o2i_loss"></a>
## o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#o2i_loss) · [Python](../python/geometry-based-stochastic-model.md#o2i_loss)

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier frequency; the
  caller adds `loss` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table 7.4.3-1),
  `legacy_6GHz` (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz; the O2I conf files of UMa and UMi have
  its `SF_sigma = 7`), `car` and `car_metal` (clause 7.4.3.2, metallized windows)
- Model of UT `u`: the smallest `j` with `rnd(u, 0) < w_0 + ... + w_j`, else the last model with a weight > 0; a model
  with weight 0 is never chosen
- Indoor distance: the minimum of columns 1 and 2 of `rnd` times `d2d_in_max` (`low`, `high`, `low_A`, clause
  7.4.3.1), column 1 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand) and [sos_randn](sum-of-sinusoids-random-number-generator.md#sos_randn) at the correlation distances of TR 38.901 clause 7.6.3.3: 50 m
  for column 0, 25 m for columns 1 and 2, 10 m for column 3
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`, SMa all three.
  SMa has 90 % residential and 10 % commercial buildings (Table 7.2-5) with different maximum indoor distances: call
  `o2i_loss` once per building type

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
  - `tripple_slope` (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0, sig1..sig3): the
    `dual_slope` rule with the breakpoints `dBP1 = min(dBP1, dBP2)` and `dBP2` of `E1, hE1` and `E2, hE2`, the
    slopes `A1`, `A2`, `A3` and `sf = sig1`, `sig2`, `sig3`
  - `nlos` (An, Bn; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, Ax, Bx, Cx, sig):
    `hb = max(hBS, 0.1)`, `hm = max(hUT, 0.1)`; `An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb +
    D2n·lg(hb)/hb² + D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) +
    G1n·(lg(G2n·hm))²`; with `LOSmax != 0` (which requires the `dual_slope` keys) at least the `dual_slope` loss; with
    `Ax`, `Bx` or `Cx` at least `Ax·lg d3 + Bx + Cx·lg f` (e.g. the InF-SL term of InF-DL); `sf = sig`
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS/d3, 1)`; `+Inf` for `sa <= 0`, else
    `A·lg d3 + B + C·lg f + D·lg(asin(sa))`, plus the zenith attenuation by atmospheric gases over `sa` (1 to 100 GHz,
    constant beyond) if `usePLa != 0`
  - empty `pl_model`: 0
- Where the model sets no `sf` (a missing `SF`, `sig` key or no such key): the SF std of [lsp_generate](#lsp_generate) at the geometric distance, so that
  passing `sf_sigma` to [lsp_generate](#lsp_generate) is neutral
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

- Checks scenario parameters that did not come from a conf file and adds every missing key with its default value,
  so that `scenpar` has all `n_key = 171` keys on return: the 138 numeric keys of QuaDRiGa and 33 keys for TR 38.901
  v19.5.0, which have no effect at their default of 0. An empty `scenpar` gives the complete default set
- A name that is not one of the keys is an error, also with `validate = false`; this includes the QuaDRiGa key
  `GR_epsilon`, which is not supported
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
  contains `NaN`; a matrix that is not positive definite is no error

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
    25 dB below the strongest get power 0 (no renormalisation); the K-factor in the scaling factors is limited
    to >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` at the drop or segment
    start and keep them along the track
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

- Slots per link: 0 = LOS path, 1 = ground reflection if `GR_enabled > 0` (`G = 1`), then the `N = NumClusters - 1`
  NLOS clusters; `n_slot = NumClusters + G`. Slot `1 + G + n` belongs to column `n` of each block of `rnd` (no
  sorting). Clusters `n >= n_cluster(l)` (TR clause 7.6.15) get delay 0, power 0 and the LOS angles. Identify the
  LOS path by its slot, never by delay or path length: NLOS clusters of the TR methods can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever
  `lsp` holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; angles in [rad], GCS, elevation convention; slot 0 holds the
  geometric LOS direction. Powers are the coefficient powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters
  `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `"38.901"`, `"38.901_MF"` add `abs_toa(l)` to the NLOS
  delays; `"quadriga"` places the first NLOS cluster at `abs_toa(l)` (where `absTOA_mu <= -30`: only where
  `abs_toa(l) > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0: slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`"quadriga"`), with `R²` the mean of `abs(R_par)²` and `abs(R_perp)²` of TR
  eqs. (7.6-38), (7.6-39); the coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarisation
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios, plus the ground reflection if enabled; `rnd` must
  be empty, `rnd_freq` and `lsp_anchor` are not used
- Swap reciprocity: with mirrored random values (dual mobility in [gbsm_draw](#gbsm_draw)) and mirrored LSPs, swapping `tx_pos`
  and `rx_pos` exchanges the departure and arrival angles (`"38.901"`: `eod_offset_deg` = 0 and no O2I link;
  `"quadriga"`: `eod_offset_deg` = 0)
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
    const arma::uvec *n_cluster = nullptr,
    arma::uvec *n_active = nullptr);
```

### Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`, `KF_mu`, `absTOA_mu`, `r_DS`,
  `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in [m]; z >= 0 with ground reflection; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[n_link, 3]`
- **`lsp`** — LSPs from [lsp_generate](#lsp_generate); DS, ASD, ASA, ESD, ESA (slices 0, 3 to 6) > 0, KF (slice 1) >= 0;
  `"38.901"`: one frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [gbsm_draw](#gbsm_draw) (`"cluster"`, `set_id = -1`), columns 0 to N-1 in [0, 1];
  `[n_link, 9·N]`
- **`method`** *(optional)* — `"38.901"` (default), `"38.901_MF"` or `"quadriga"`
- **`rnd_freq`** *(optional)* — Per-frequency cluster sets of [gbsm_draw](#gbsm_draw), slice `k` from `set_id = k`; required
  by `"38.901"` and `"38.901_MF"`; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz) from [lsp_generate](#lsp_generate); required by
  `"38.901_MF"`; `[n_link, 1, 8]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg] from [lsp_generate](#lsp_generate); used by `"38.901"` and
  `"quadriga"` in NLOS scenarios; omitted = 0; `[n_link, n_freq]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s] from [lsp_generate](#lsp_generate), >= 0; required by `"quadriga"` in
  NLOS scenarios with `absTOA_mu > -30`; omitted = no offset; `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im` from
  [lsp_generate](#lsp_generate), real part >= 1, imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** *(optional)* — O2I flag per link (mean ZOA 90 deg); used by `"38.901"` in NLOS scenarios; omitted =
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
- [calc_delay_spread](channel-statistics.md#calc_delay_spread), [calc_angular_spreads_sphere](channel-statistics.md#calc_angular_spreads_sphere) (spreads of the output)

---
<a id="ssf_paths"></a>
## ssf_paths
Antenna-free paths of one link at one position

Not in: MATLAB, Python

- `"phase"`: use the outputs with [get_channels_planar](channel-generation-functions.md#get_channels_planar) and the arrays at the positions of `rays` (the
  displacement phase is in `M`); `"drift"`: inputs of [get_channels_spherical](channel-generation-functions.md#get_channels_spherical), [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) and
  [qrt_paths_from_scatterers](channel-functions.md#qrt_paths_from_scatterers)
- Rays with zero power at all frequencies are dropped, LOS and GR rays are kept
- Validity: `"phase"` up to about `sqrt(lambda · b_eff / 8)` per moving end, `"drift"` up to about `0.2 · b_med`
  per moving end (see [gbsm_rays](#gbsm_rays))

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
    arma::uvec *path_type = nullptr,
    arma::uvec *ray_ind = nullptr,
    bool oxygen = false);
```

### Inputs:
- **`rays`** — Ray data of the link, see [gbsm_rays](#gbsm_rays)
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at the position in [dB]; 0 for data from paths; `[n_freq]`
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"` (phase for the TR methods, drift for quadriga)
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx position in [m]; omitted: the positions of `rays`; `[3]`
- **`kf_ratio`** *(optional)* — K(position) / K(drop); omitted: 1; `[n_freq]`
- **`oxygen`** *(optional)* — TR 7.6.1 oxygen absorption from each path's own length

### Outputs:
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`M`** — Polarization transfer matrices; `[8, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** *(optional)* — Departure and arrival angles in [rad]; `[n_path]`
- **`fbs_pos`**, **`lbs_pos`** *(optional)* — Scatterer positions in [m]; `[3, n_path]`, `"phase"`: `[3, 0]`
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS; `[n_path]`
- **`ray_ind`** *(optional)* — Index of the ray in `rays`, 0-based; `[n_path]`

### See also:
- [gbsm_rays](#gbsm_rays) (input data, validity scales)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel coefficients of one link)

---
<a id="ssf_rays"></a>
## ssf_rays
Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)

Not in: MATLAB, Python

- Splits the clusters of [ssf_clusters](#ssf_clusters) into rays and writes one [gbsm_rays](#gbsm_rays) object per link: ray delays,
  powers, angles, polarization transfer matrices with initial phases, path types and tap indices
- One geometry per call: the TR methods compute frequency `i_freq`, `"quadriga"` all frequencies
- Ray layout per link: index 0 LOS, 1 GR (if `GR_enabled`), then ray m of NLOS cluster n at `1 + G + n·M + m`

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
- **`scenpar`** — Scenario parameters
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`center_frequency`** — Carrier frequencies in [Hz]; `[n_freq]`
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [lsp_generate](#lsp_generate);
  `[n_link, n_freq, 8]`
- **`delay`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays and angles from [ssf_clusters](#ssf_clusters);
  `[n_link, n_slot, n_freq]` for `"38.901_MF"`, else `[n_link, n_slot, 1]`
- **`power`** — Cluster powers from [ssf_clusters](#ssf_clusters); `[n_link, n_slot, n_freq]`
- **`rnd_ray`** — "ray" sets from [gbsm_draw](#gbsm_draw); `[n_link, n_gen, n_freq]` (TR methods) or `[n_link, n_gen, 1]`
- **`method`** *(optional)* — Method of [ssf_clusters](#ssf_clusters): `"38.901"`, `"38.901_MF"` or `"quadriga"`
- **`i_freq`** *(optional)* — TR methods: index into `center_frequency`, 0-based; `"quadriga"`: 0
- **`angle_model`** *(optional)* — `"additive"` or `"rotation"`
- **`pol_model`** *(optional)* — `"38.901"` or `"quadriga"`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity from [lsp_generate](#lsp_generate); required if
  `GR_enabled`; `[n_link, n_freq]`
- **`pol_variability`** *(optional)* — TR 7.6.16 polarisation power variability
- **`random_phase`** *(optional)* — `false`: all initial phases 0

### Output:
- **`rays`** — Ray data, one entry per link; `fbs_pos`, `lbs_pos`, `fallback` empty; `[n_link]`

### See also:
- [gbsm_rays](#gbsm_rays) (attributes of the output)
- [ssf_paths](#ssf_paths) (paths of one link)

---
<a id=".solve_scatterers"></a>
## .solve_scatterers
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

Not in: MATLAB, Python

- NLOS rays: exact departure and arrival angles where a solution exists, equal distances from both ends;
  otherwise the shortest polyline along the ray's own directions with both scatterers at least `d_min` away
  (`fallback` = 1; the ray is lengthened, `delay` keeps its value)
- LOS ray on the direct line, GR ray at the specular point on z = 0, rays with zero power at all frequencies at
  the midpoint of the direct line
- Writes `fbs_pos`, `lbs_pos` and `fallback` of all rays and overwrites given scatterers; nothing else changes;
  afterwards `b_med()` and `b_eff()` give the validity scales of [ssf_paths](#ssf_paths)

### Declaration:
```cpp
void quadriga_lib::gbsm_rays::solve_scatterers(double d_min = 2.0);
```

### Input:
- **`d_min`** *(optional)* — Minimum distance of the scatterers to both ends in [m], finite and > 0

### See also:
- [gbsm_rays](#gbsm_rays) (attributes and members)
- [ssf_paths](#ssf_paths) (mobility `drift` needs the scatterers)
