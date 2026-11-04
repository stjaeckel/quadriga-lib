# Geometry-based stochastic model

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-geometry-based-stochastic-model

| Function | Summary |
| --- | --- |
| [conf_file_list](#conf_file_list) | List the scenarios defined by the conf files in a set of directories |
| [conf_file_read](#conf_file_read) | Read and validate a QuaDRiGa scenario configuration file |
| [conf_file_write](#conf_file_write) | Write a QuaDRiGa scenario configuration file |
| [los_probability](#los_probability) | LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models |
| [lsf_draw](#lsf_draw) | Spatially correlated normalized random values of the large-scale generators at a set of links |
| [lsf_init_phases](#lsf_init_phases) | Draw the random phases of the large-scale generators of one BS site and scenario |
| [lsp_generate](#lsp_generate) | Large-scale parameters of a set of links from normalized random values |
| [o2i_loss](#o2i_loss) | O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3 |
| [path_loss](#path_loss) | Path loss and shadow-fading standard deviation of a set of links |
| [scenpar_validate](#scenpar_validate) | Validate scenario parameters and complete them with default values |
| [ssf_clusters](#ssf_clusters) | Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7) |
| [ssf_draw](#ssf_draw) | Spatially correlated normalized random values of one small-scale fading generator set at a set of links |
| [ssf_init_phases](#ssf_init_phases) | Draw the random phases of one small-scale fading generator set of a BS site and scenario |

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
files (0 in the other files means "not stated here"). Per BS site and scenario: [lsf_init_phases](#lsf_init_phases) draws the phases
of the large-scale generators once, [lsf_draw](#lsf_draw) evaluates them at the links, [path_loss](#path_loss) gives the path loss and
the shadow-fading (SF) standard deviation, and [lsp_generate](#lsp_generate) the LSPs of each link and frequency. Small scale:
[ssf_init_phases](#ssf_init_phases) and [ssf_draw](#ssf_draw) do the same for the cluster random values, and [ssf_clusters](#ssf_clusters) turns the LSPs
and random values into cluster delays, powers and angles.

**Large-scale generators.** Columns of the phase cube of [lsf_init_phases](#lsf_init_phases) and of the random values of
[lsf_draw](#lsf_draw); a decorrelation distance of 0 is used as 0.1 m:

| Column | Generator | Distribution | Decorrelation distance |
| --- | --- | --- | --- |
| 0 to 7 | DS, KF, SF, ASD, ASA, ESD, ESA, XPR | Normal(0,1) | `DS_lambda`, `KF_lambda`, `SF_lambda`, `AS_D_lambda`, `AS_A_lambda`, `ES_D_lambda`, `ES_A_lambda`, `XPR_lambda` |
| 8 | absolute TOA | Normal(0,1) | `absTOA_lambda` |
| 9 | ground permittivity | Uniform(0,1) | `SC_lambda` |
| 10 | UMa environment height hE | Uniform(0,1) | `SF_lambda` |

The Tx side of a phase cube (slice 1) exchanges ASD and ASA, ESD and ESA, so that in dual mobility (Tx and Rx both
move) the departure spreads at the Tx are the arrival spreads of the Tx as a second device.

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
- Four kinds of keys:
  - the `n_key` numeric scenario parameters of [scenpar_validate](#scenpar_validate), one number each; keys that are not in the file
    have their default value
  - `SubpathMethod`: one word, default `legacy`
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the QuaDRiGa key
  `GR_epsilon` (the complex ground permittivity comes from [lsp_generate](#lsp_generate)). A key stated more than once has its
  last value
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
    std::string *subpath_method = nullptr,
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
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`
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
- `write_defaults = false`: only keys whose value differs from the default (bit for bit) and `SubpathMethod` if it
  is not `legacy`; a section without a written key is left out
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
    const std::string &subpath_method = "legacy",
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
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`
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
    const arma::vec *fc = nullptr,
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
- **`fc`** *(optional)* — Carrier frequencies in [Hz], finite and > 0; required with `los_soft`; `[n_freq]`

### Outputs:
- **`p_los`** — LOS probability in [0, 1]; `[n_link]`
- **`los_soft`** *(optional)* — Soft LOS state in [0, 1]; `> 0.5` exactly when `rnd_g > -F(p_los)` of eq. (7.6-18),
  which has the probability `p_los`; `[n_link, n_freq]`

### See also:
- [o2i_loss](#o2i_loss) (indoor distance for `d2d_out`)
- [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand), [sos_randn](sum-of-sinusoids-random-number-generator.md#sos_randn) (spatially consistent LOS-state draw and `rnd_g`)

---
<a id="lsf_draw"></a>
## lsf_draw
Spatially correlated normalized random values of the large-scale generators at a set of links

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#lsf_draw) · [Python](../python/geometry-based-stochastic-model.md#lsf_draw)

- Evaluates the phases of [lsf_init_phases](#lsf_init_phases) at the links, one column per generator (see the section
  description). The value of a link does not depend on the other links of the call; in single mobility it does not
  depend on `tx_pos`
- Dual mobility: the Tx is a second device on the same maps. Every column keeps its distribution; ASD and ASA, ESD
  and ESA of a link are correlated by the ACF at the Tx-Rx distance (equal at distance 0). Swapping `tx_pos` and
  `rx_pos` swaps the columns 3 and 4, 5 and 6 and leaves the others unchanged, exactly
- Floors (2D mode): links with equal `floor_id` share one field; fields of different floors are uncorrelated at
  equal horizontal positions, exactly within a block of `n_coeff` consecutive IDs, otherwise up to a random residual
  of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation (AVX2 kernel of [sos_val](sum-of-sinusoids-random-number-generator.md#sos_val))
- `rnd` may share memory with `tx_pos` or `rx_pos`

### Declaration:
```cpp
void quadriga_lib::lsf_draw(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::cube &phase,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &rnd,
    const arma::uvec *floor_id = nullptr,
    bool dual_mobility = false,
    const std::string &preset = "Comb256",
    bool horizontal = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances are used
- **`phase`** — Phases from [lsf_init_phases](#lsf_init_phases), finite; `[n_coeff, 11, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility, always validated; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`, `n_link = 0` allowed
- **`floor_id`** *(optional)* — Floor index of the UT per link, 0-based (`n_fl - 1` for the UT heights of TR 38.901
  Table 7.2-1, 0 for outdoor UTs); omitted = floor 0; 2D mode only; `[n_link]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move
- **`preset`** *(optional)* — Sum-of-sinusoids preset of [lsf_init_phases](#lsf_init_phases), see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset); a different
  preset with the same number of sinusoids is not detected
- **`horizontal`** *(optional)* — Mode of [lsf_init_phases](#lsf_init_phases)

### Output:
- **`rnd`** — Random values, columns as in the section description; `[n_link, 11]`

### See also:
- [lsf_init_phases](#lsf_init_phases) (draws the phases)
- [path_loss](#path_loss) (column 10 as `rnd_he`)
- [lsp_generate](#lsp_generate) (the large-scale parameters from `rnd`)

---
<a id="lsf_init_phases"></a>
## lsf_init_phases
Draw the random phases of the large-scale generators of one BS site and scenario

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#lsf_init_phases) · [Python](../python/geometry-based-stochastic-model.md#lsf_init_phases)

- Draws the phases of the 11 large-scale generators (see the section description); [lsf_draw](#lsf_draw) evaluates them at
  the links. The caller stores `phase` to evaluate the same fields again along tracks or on maps
- Without bias removal, slice 0 (Rx side) equals [sos_init_phases](sum-of-sinusoids-random-number-generator.md#sos_init_phases) with 11 generators and the `n_coeff` of
  `preset`; slice 1 (Tx side) is its mirror, with and without bias removal
- Bias removal (`remove_bias = true` and at least 3 rows in `rx_pos`): [sos_remove_bias](sum-of-sinusoids-random-number-generator.md#sos_remove_bias) sets the sample mean of
  each Normal field (columns 0 to 8) over the positions to 0; in dual mobility over `rx_pos` and `tx_pos` together.
  It removes variance unless the positions spread over many decorrelation distances
- In 2D mode the bias removal treats all positions as one floor: for a drop with UTs on several floors pass the
  positions of one floor or set `remove_bias = false`
- TR 38.901 clause 7.4.4 literally (exponential ACF, no bias removal): `preset = "Exp256"`, `remove_bias = false`.
  The default `Comb256` has the same decorrelation distance without the kink of the exponential ACF at zero
  distance, so the values evolve smoothly along a track

### Declaration:
```cpp
void quadriga_lib::lsf_init_phases(
    const std::unordered_map<std::string, double> &scenpar,
    arma::cube &phase,
    arma::sword seed = -1,
    const arma::mat *tx_pos = nullptr,
    const arma::mat *rx_pos = nullptr,
    bool dual_mobility = false,
    bool remove_bias = true,
    const std::string &preset = "Comb256",
    bool horizontal = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances are used
- **`seed`** *(optional)* — Seed of the phases, >= -1; -1 = system random device, new phases on each call
- **`tx_pos`** *(optional)* — Tx positions in [m]; required for the bias removal in dual mobility, otherwise not
  used; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** *(optional)* — Rx positions of the drop in [m] for the bias removal; omitted or fewer than 3 rows:
  no bias removal; `[n_link, 3]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move
- **`remove_bias`** *(optional)* — `true`: remove the sample mean of the Normal fields over the positions
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset); sets `n_coeff`
- **`horizontal`** *(optional)* — `true`: 2D mode (TR 38.901); `false`: 3D mode (legacy QuaDRiGa)

### Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 0 = Rx side, slice 1 = Tx side; `[n_coeff, 11, 2]`

### See also:
- [lsf_draw](#lsf_draw) (evaluates the phases at the links)
- [ssf_init_phases](#ssf_init_phases) (the small-scale sets, seeds derived from this seed)
- [sos_remove_bias](sum-of-sinusoids-random-number-generator.md#sos_remove_bias) (the bias removal of one generator)

---
<a id="lsp_generate"></a>
## lsp_generate
Large-scale parameters of a set of links from normalized random values

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#lsp_generate) · [Python](../python/geometry-based-stochastic-model.md#lsp_generate)

- Generates the 8 LSPs of TR 38.901 clause 7.5 step 4 per link and frequency from the random values of [lsf_draw](#lsf_draw),
  and on request the absolute TOA offset, the ground permittivity and the EoD offset
- Mean `mu` and standard deviation `sigma` in the log domain (lg for DS and the angular spreads, dB for KF, SF and
  XPR), with the key prefix `P` (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`, `XPR`), `d = max(d2D, 1e-4)`,
  `h = max(hBS, 1e-4)`, `a = atan(h/d)` and `f' = max(fc/1e9, LSP_fc_min)` in GHz:
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
  (0 dry, 0.5 medium, 1 wet) with `fc` not limited by `LSP_fc_min`; they are not the materials of TR 38.901 Table
  7.6.8-1
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e +
  EoD_offset_e_gamma·lg f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) +
  EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if `EoD_offset_b > 0`, plus
  `(atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi` if `hUT < EoD_offset_hcl`

### Declaration:
```cpp
void quadriga_lib::lsp_generate(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::vec &fc,
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
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`rnd`** — Random values from [lsf_draw](#lsf_draw), finite; column 9 in [0, 1] when the ground permittivity is requested;
  column 10 is not used; `[n_link, 11]`
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
- [lsf_draw](#lsf_draw) (the random values `rnd`)
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
    const arma::vec &fc,
    const arma::mat &rnd,
    const std::vector<std::string> &models,
    const arma::vec &weights,
    arma::mat &loss,
    double d2d_in_max = 25.0,
    arma::vec *d2d_in = nullptr,
    arma::uvec *model_index = nullptr);
```

### Inputs:
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
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
- Notation: `f = fc/1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance, at least 0.1 m;
  `hBS`, `hUT` the z coordinates of Tx and Rx; `lg = log10`; `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`; `sf` the
  shadow-fading std. Keys in parentheses: the required ones, after the semicolon the optional ones (`X = v`: default)
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
    const arma::vec &fc,
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
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m] of the UT of each link (e.g. from [o2i_loss](#o2i_loss)), finite
  and >= 0; given = legacy outdoor distance; 0 = outdoor link; `[n_link]`
- **`rnd_he`** *(optional)* — Uniform(0,1) value per link for the random environment height, column 10 of
  [lsf_draw](#lsf_draw); `[n_link]`

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
  - `subpath_method` must be `legacy`, `Laplacian` or `mmMAGIC`
- Throws `std::invalid_argument`; a failed call leaves `scenpar` and the outputs untouched

### Declaration:
```cpp
bool quadriga_lib::scenpar_validate(
    std::unordered_map<std::string, double> &scenpar,
    const std::string &subpath_method = "legacy",
    arma::mat *lsp_xcorr = nullptr,
    std::vector<std::string> *key_order = nullptr,
    bool validate = true);
```

### Inputs:
- **`scenpar`** — Numeric scenario parameters; updated in-place: given values are kept, missing keys are
  added with their default; `n_key` entries on return
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`, the only parameter that is not numeric
- **`validate`** *(optional)* — `false`: only complete with defaults, no value checks (`NaN` and any
  `subpath_method` are accepted)

### Outputs:
- **`lsp_xcorr`** *(optional)* — LSP cross-correlation matrix from the 28 correlation keys, symmetric with unit
  diagonal; order of rows and columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `[8, 8]`
- **`key_order`** *(optional)* — Canonical order of the keys in `scenpar`: the QuaDRiGa order without
  `SubpathMethod` (its position is between elements 105 and 106, 0-based), then the 33 new keys; `[n_key]`

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

- Turns the LSPs of [lsp_generate](#lsp_generate) and the random values of [ssf_draw](#ssf_draw) into the clusters of each link, by one of
  three methods:
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
- Swap reciprocity: with mirrored random values (dual mobility in [ssf_draw](#ssf_draw)) and mirrored LSPs, swapping `tx_pos`
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
- **`rnd`** — Shared cluster set of [ssf_draw](#ssf_draw) (`set_id = -1`), columns 0 to N-1 in [0, 1]; `[n_link, 9·N]`
- **`method`** *(optional)* — `"38.901"` (default), `"38.901_MF"` or `"quadriga"`
- **`rnd_freq`** *(optional)* — Per-frequency sets of [ssf_draw](#ssf_draw), slice `k` from `set_id = k`; required by
  `"38.901"` and `"38.901_MF"`; `[n_link, N, n_freq]`
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
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_fd]`, `n_fd = n_freq` for `"38.901_MF"`, else 1
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in [rad], in [-pi, pi); `[n_link, n_slot, n_fd]`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in [rad], in [-pi/2, pi/2]; `[n_link, n_slot, n_fd]`
- **`n_active`** *(optional)* — Number of NLOS clusters with power > 0; `[n_link]`

### See also:
- [ssf_draw](#ssf_draw) (the random values `rnd` and `rnd_freq`)
- [lsp_generate](#lsp_generate) (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [calc_delay_spread](channel-statistics.md#calc_delay_spread), [calc_angular_spreads_sphere](channel-statistics.md#calc_angular_spreads_sphere) (spreads of the output)

---
<a id="ssf_draw"></a>
## ssf_draw
Spatially correlated normalized random values of one small-scale fading generator set at a set of links

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_draw) · [Python](../python/geometry-based-stochastic-model.md#ssf_draw)

- Evaluates a set of [ssf_init_phases](#ssf_init_phases) at the links with the rules of [lsf_draw](#lsf_draw) (dual mobility, floors,
  accuracy); every generator uses the decorrelation distance `SC_lambda` (TR 38.901 Table 7.6.3.1-2; 0 = spatial
  consistency off)
- Columns, `N = NumClusters - 1`; column `b·N + n` belongs to block `b` and NLOS cluster `n`:
  - shared set (`9·N` columns): block 0 cluster delay, Uniform(0,1); blocks 1, 2 sign of AOD, AOA; 3, 4 sign of
    ZOD, ZOA; 5, 6 angle offset of AOD, AOA; 7, 8 angle offset of ZOD, ZOA; Normal(0,1)
  - per-frequency set (`N` columns): cluster shadowing, Normal(0,1)
- Dual mobility: swapping `tx_pos` and `rx_pos` exchanges the blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 and keeps
  the delay and shadowing blocks, exactly
- A column depends neither on the other columns nor on the other links; later versions append blocks and keep the
  values of the existing columns
- `NumClusters = 1` gives `[n_link, 0]` (any empty `phase` is accepted)

### Declaration:
```cpp
void quadriga_lib::ssf_draw(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::cube &phase,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &rnd,
    arma::sword set_id = -1,
    const arma::uvec *floor_id = nullptr,
    bool dual_mobility = false,
    const std::string &preset = "Comb256",
    bool horizontal = true);
```

### Inputs:
- **`scenpar`** — Scenario parameters; `NumClusters` and `SC_lambda` are used
- **`phase`** — Phases from [ssf_init_phases](#ssf_init_phases), finite; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`set_id`** *(optional)* — Set of [ssf_init_phases](#ssf_init_phases): `-1` = shared cluster set; `k >= 0` = per-frequency set
- **`floor_id`** *(optional)* — Floor index per link as in [lsf_draw](#lsf_draw); omitted = floor 0; 2D mode only; `[n_link]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move
- **`preset`** *(optional)* — Sum-of-sinusoids preset of [ssf_init_phases](#ssf_init_phases), see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset)
- **`horizontal`** *(optional)* — `true`: 2D mode (TR 38.901); `false`: 3D mode (legacy QuaDRiGa)

### Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

### See also:
- [ssf_init_phases](#ssf_init_phases) (draws the phases)
- [ssf_clusters](#ssf_clusters) (the clusters from the random values)

---
<a id="ssf_init_phases"></a>
## ssf_init_phases
Draw the random phases of one small-scale fading generator set of a BS site and scenario

Also in: [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_init_phases) · [Python](../python/geometry-based-stochastic-model.md#ssf_init_phases)

- The cluster random values (TR 38.901 clause 7.6.3.1) come in one shared cluster set (`set_id = -1`, `9·N`
  generators, `N = NumClusters - 1` NLOS clusters) and one set per carrier frequency (`set_id = k`, `N` generators,
  `k` the index in the caller's frequency list); [ssf_draw](#ssf_draw) evaluates them and lists the columns
- The phases are a fixed function of `seed` and `set_id`, equal on all platforms; the sets are independent of each
  other and of the large-scale phases. Keep an explicit seed (or the phase cubes) to reproduce the clusters
- Each scenario of a site (LOS, NLOS, O2I) needs its own large-scale seed: TR 38.901 clause 7.6.3.4 makes the link
  types uncorrelated
- `NumClusters = 1`: both sets are empty

### Declaration:
```cpp
void quadriga_lib::ssf_init_phases(
    const std::unordered_map<std::string, double> &scenpar,
    arma::cube &phase,
    arma::sword seed = -1,
    arma::sword set_id = -1,
    const std::string &preset = "Comb256");
```

### Inputs:
- **`scenpar`** — Scenario parameters; `NumClusters` sets `N`
- **`seed`** *(optional)* — Seed of the large-scale set of the same site and scenario ([lsf_init_phases](#lsf_init_phases)), >= -1;
  -1 = system random device, new phases on each call
- **`set_id`** *(optional)* — `-1` = shared cluster set; `k >= 0` = set of frequency `k`
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset); sets `n_coeff`

### Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 0 = Rx side, slice 1 = Tx side; `[n_coeff, n_gen, 2]`,
  `n_gen = 9·N` (shared set) or `N`

### See also:
- [ssf_draw](#ssf_draw) (evaluates the phases at the links)
- [lsf_init_phases](#lsf_init_phases) (the large-scale set and its seed)
