# Geometry-based stochastic model

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-geometry-based-stochastic-model

| Function | Summary |
| --- | --- |
| [conf_file_list](#conf_file_list) | List the scenarios defined by the conf files in a set of directories |
| [conf_file_read](#conf_file_read) | Read and validate a QuaDRiGa scenario configuration file |
| [conf_file_write](#conf_file_write) | Write a QuaDRiGa scenario configuration file |
| [gbsm.draw](#gbsm.draw) | Spatially correlated normalized random values of one generator set at a set of links |
| [gbsm.init_phases](#gbsm.init_phases) | Draw the random phases of one generator set of a BS site and scenario |
| [los_probability](#los_probability) | LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models |
| [lsp_generate](#lsp_generate) | Large-scale parameters of a set of links from normalized random values |
| [o2i_loss](#o2i_loss) | O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3 |
| [path_loss](#path_loss) | Path loss and shadow-fading standard deviation of a set of links |
| [scenpar_validate](#scenpar_validate) | Validate scenario parameters and complete them with default values |
| [ssf_clusters](#ssf_clusters) | Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7) |

A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the *scenario*
(e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The functions of this
section implement 3GPP TR 38.901 clause 7.5 steps 1 to 7 (path loss, LOS probability, O2I penetration loss, large-scale
parameters (LSPs) and clusters) with the spatial consistency of clause 7.6.3. The ray level (steps 8 to 12) follows in a
later version. They are in the submodule `quadriga_lib.gbsm`.

**Scenario parameters.** A scenario is a dict of numeric parameters (`scenpar`), a dict of path-loss parameters
(`plpar`) and optional short names, read from a conf file by [conf_file_read](#conf_file_read). 23 files for TR 38.901 v19.5.0 come
with the library, in `data/config` of the source tree and in `share/quadriga-lib/config` of an installed package (for
pip, below `sys.prefix`); [conf_file_list](#conf_file_list) finds them and [conf_file_write](#conf_file_write) writes them. Every function that takes
`scenpar` accepts the dict of [conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate) or any subset of its keys (`None` = all
defaults; a key `SubpathMethod` of QuaDRiGa's structs must be `"legacy"`), completes it with the defaults of missing
keys and validates it with [scenpar_validate](#scenpar_validate); an invalid value of a key that the function does not use is an error
as well. Invalid input raises
`ValueError`, a failure to read or write a file `RuntimeError`.

**Workflow.** Per drop: [o2i_loss](#o2i_loss) gives the penetration loss and the indoor distance of each indoor UT, then
[los_probability](#los_probability) the LOS probability at the outdoor distance `d2D - d2D-in`. The caller draws the LOS, indoor and O2I
states with [sos.rand](sum-of-sinusoids-random-number-generator.md#sos.rand) / [sos.randn](sum-of-sinusoids-random-number-generator.md#sos.randn) at the correlation distances of TR 38.901 clause 7.6.3.3; the keys
`LOS_state_lambda` and `IO_state_lambda` apply to all links of a scenario and are stated in the NLOS and in the O2I
files (0 in the other files means "not stated here"). Per BS site and scenario: [gbsm.init_phases](#gbsm.init_phases) draws the phases of
each random set once (`"lsf"`; `"cluster"` with `set_id = -1` and one set per carrier frequency), and [gbsm.draw](#gbsm.draw)
evaluates them at the links. Then [path_loss](#path_loss) gives the path loss and the shadow-fading (SF) standard deviation,
[lsp_generate](#lsp_generate) the LSPs of each link and frequency, and [ssf_clusters](#ssf_clusters) turns the LSPs and the cluster random values
into cluster delays, powers and angles.

**Random sets.** The spatially correlated random values come in sets of generators, one column of `rnd` each;
`N = NumClusters - 1`, column `b·N + n` of a `"cluster"` set belongs to block `b` and NLOS cluster `n`:

| `set_type`, `set_id` | `n_gen` | Columns and distributions | Decorrelation distance | Mirror pairs | Used by |
| --- | --- | --- | --- | --- | --- |
| `"lsf"` | 11 | 0 to 7 DS, KF, SF, ASD, ASA, ESD, ESA, XPR, Normal(0,1); 8 absolute TOA, Normal(0,1); 9 ground permittivity, 10 UMa environment height hE, Uniform(0,1) | `DS_lambda` ... `XPR_lambda` (0 to 7), `absTOA_lambda` (8), `SC_lambda` (9), `SF_lambda` (10) | 3 and 4, 5 and 6 | [lsp_generate](#lsp_generate) `rnd`, [path_loss](#path_loss) `rnd_he` (column 10) |
| `"cluster"`, -1 | `9·N` | block 0 cluster delay, Uniform(0,1); 1, 2 sign of AOD, AOA; 3, 4 sign of ZOD, ZOA; 5, 6 offset of AOD, AOA; 7, 8 offset of ZOD, ZOA, Normal(0,1) | `SC_lambda` | blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 | [ssf_clusters](#ssf_clusters) `rnd` |
| `"cluster"`, `k` | `N` | cluster shadowing of frequency `k`, Normal(0,1) | `SC_lambda` | none | [ssf_clusters](#ssf_clusters) `rnd_freq`, slice `k` |

The Tx side of a phase array (slice 1) exchanges the columns of each mirror pair, so that in dual mobility (Tx and Rx
both move) the departure quantities at the Tx are the arrival quantities of the Tx as a second device. A decorrelation
distance of 0 is used as 0.1 m. Only the `"lsf"` columns 0 to 8 take part in the bias removal. A set never changes size:
new random quantities come in new sets (a new `set_type` or `set_id`).

**Conventions.** Positions are in m with the columns x, y, z: `rx_pos` is `(n_link, 3)`, `tx_pos` is `(1, 3)` (one Tx
for all links) or `(n_link, 3)`; heights are the z coordinates. In 2D mode (`horizontal = True`, the TR behaviour) the
random values are correlated in the x-y plane, and `floor_id` separates the floors, which are uncorrelated (TR 38.901
Table 7.5-6 note 3). In 3D mode (`horizontal = False`, legacy QuaDRiGa) the height separates the floors, so UTs on
different floors are correlated by the ACF of their height difference; the per-link statistics are the same in both
modes. Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default and QuaDRiGa's is available through an
argument where it is still wanted; where QuaDRiGa's behaviour was an error, only the corrected rule is implemented.
Units: frequencies in Hz, path loss in dB, DS in s, KF, SF and XPR as linear power ratios, spreads in degrees.

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
  only, which is faster; bool or `numpy.bool_`; default: True; `None` selects the default

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
  - 171 numeric scenario parameters (see [scenpar_validate](#scenpar_validate)), one number each; keys that are not in the file have
    their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the QuaDRiGa key `GR_epsilon`
  (the complex ground permittivity comes from [lsp_generate](#lsp_generate)). A key stated more than once has its last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of `SubpathMethod` raises
  `ValueError`, also with `validate = False`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`), read independent of
  the locale; `Inf`, `+Inf` and `-Inf` in any case; a number below the smallest subnormal reads as 0
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
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
  grammar is checked; bool or `numpy.bool_`; default: True; `None` selects the default

### Outputs:
- **`scenpar`** — Scenario parameters; dict with 171 float entries in the canonical order: the 138
  numeric keys of QuaDRiGa, then the 33 new keys
- **`plpar`** — Path-loss parameters; dict with the entry `model` (str, only if the file states
  `PL_model`) first, then one entry per `PL_` key in file order, named without the prefix `PL_`; each
  is a 1D array with one or more values; `{}` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be
  generated from it; bool
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
- Numbers have up to 15 significant digits, or 16 or 17 where 15 do not read back to the same double, independent of the
  locale; `Inf`, `-Inf` and `-0` are kept
- `write_defaults = False`: only keys whose value differs from the default (bit for bit); a section without a written
  key is left out
- All inputs are validated before the file is opened: the checks of [scenpar_validate](#scenpar_validate); `model`, the other keys of
  `plpar` and the short names are names, the short names distinct; every `plpar` key has at least one value and no
  `NaN`; `fn`, `ref` and `ref_sec` have no line break
- The text is written to a temporary file in the same directory, which must be writable, and renamed to `fn`: a failed
  call creates no file and leaves an existing file unchanged; an existing file keeps its permissions, a symbolic link
  `fn` is written through

### Usage:
```python
quadriga_lib.gbsm.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, write_defaults )
```

### Inputs:
- **`fn`** — Path of the conf file to write; str, bytes or path-like object (e.g. `pathlib.Path`)
- **`scenpar`** — Scenario parameters; dict with any subset of the 171 keys returned by
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
  height, MT height in m; columns: lower and upper end, one value is written if both are equal; a
  row with `NaN` is not written; `(4, 2)`; default: None
- **`write_defaults`** — `True`: write all keys; `False`: write only keys that differ from their
  default; bool or `numpy.bool_`; default: False; `None` selects the default

### See also:
- [conf_file_read](#conf_file_read) (read the file back)
- [scenpar_validate](#scenpar_validate) (value checks and defaults)

---
<a id="gbsm.draw"></a>
## gbsm.draw
Spatially correlated normalized random values of one generator set at a set of links

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_draw) · [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_draw)

- Evaluates the phases of [gbsm.init_phases](#gbsm.init_phases) at the links, one column per generator; columns, distributions and
  decorrelation distances: see the set table in the section description. The value of a link does not depend on the
  other links of the call; in single mobility it does not depend on `tx_pos`
- Dual mobility: the Tx is a second device on the same maps, its side of each generator uses the mirror generator.
  Every column keeps its distribution; swapping `tx_pos` and `rx_pos` exchanges the columns of each mirror pair and
  keeps the self-mirrored columns, exactly
- Floors (2D mode): links with equal `floor_id` share one field; fields of different floors are uncorrelated at equal
  horizontal positions, exactly within a block of `n_coeff` consecutive IDs, otherwise up to a random residual of
  standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation
- A set never changes size: new random quantities come in new sets (a new `set_type` or `set_id`). A column depends
  neither on the other columns nor on the other links
- An empty set (`NumClusters = 1`) gives `(n_link, 0)` and accepts any empty `phase`
- The flags take a bool or `numpy.bool_`; `None` selects the default, any other type raises `TypeError`

### Usage:
```python
rnd = quadriga_lib.gbsm.draw( scenpar, set_type, phase, tx_pos, rx_pos, set_id, preset, floor_id, center_freq, \
    dual_mobility, horizontal )
```

### Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances and `NumClusters` are used
- **`set_type`** — Set type of [gbsm.init_phases](#gbsm.init_phases); str
- **`phase`** — Phases from [gbsm.init_phases](#gbsm.init_phases), finite; `(n_coeff, n_gen, 2)`
- **`tx_pos`** — Tx positions in m; used in dual mobility, always validated; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`, `n_link = 0` allowed
- **`set_id`** — Set of [gbsm.init_phases](#gbsm.init_phases): `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`;
  default: -1
- **`preset`** — Preset of [gbsm.init_phases](#gbsm.init_phases); another preset with the same `n_coeff` is not detected; default:
  `"Comb256"`
- **`floor_id`** — Floor index of the UT per link, 0-based (`n_fl - 1` for the UT heights of TR 38.901 Table 7.2-1, 0
  for outdoor UTs); int or float array; `None` or empty = floor 0; 2D mode only; `(n_link,)`; default: None
- **`center_freq`** — Carrier frequencies in Hz (messages name it `center_frequency`); ignored and not checked for the
  `"lsf"` and `"cluster"` sets; `(n_freq,)`; default: None
- **`dual_mobility`** — `True`: Tx and Rx both move; default: False
- **`horizontal`** — `True`: 2D mode (TR 38.901); `False`: 3D mode (legacy QuaDRiGa); default: True

### Output:
- **`rnd`** — Random values; `(n_link, n_gen)`

### See also:
- [lsp_generate](#lsp_generate) (the large-scale parameters from the `"lsf"` set)
- [path_loss](#path_loss) (column 10 of the `"lsf"` set as `rnd_he`)
- [ssf_clusters](#ssf_clusters) (the clusters from the `"cluster"` sets)

---
<a id="gbsm.init_phases"></a>
## gbsm.init_phases
Draw the random phases of one generator set of a BS site and scenario

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_init_phases) · [MATLAB](../mex/geometry-based-stochastic-model.md#gbsm_init_phases)

- Draws the phases of one random set (set table in the section description) for [gbsm.draw](#gbsm.draw); store `phase` or the
  seed to evaluate the same fields again along tracks or on maps
- The phases are a fixed function of `seed`, `set_type` and `set_id`, equal on all platforms (bias-removed `"lsf"`
  phases within about 2e-14 rad); the sets are independent; each scenario of a site needs its own seed
- `"lsf"` bias removal (`remove_bias`, at least 3 rows in `rx_pos`): [sos.remove_bias](sum-of-sinusoids-random-number-generator.md#sos.remove_bias) sets the sample mean of
  columns 0 to 8 over the positions to 0 (with `tx_pos` in dual mobility; 2D mode: one floor), removing variance
- TR 38.901 clause 7.4.4 literally: preset `Exp256` without bias removal; `NumClusters = 1`: empty `"cluster"` sets
- The flags take a bool or `numpy.bool_`; `None` selects the default, any other type raises `TypeError`

### Usage:
```python
phase = quadriga_lib.gbsm.init_phases( scenpar, set_type, seed, set_id, preset, tx_pos, rx_pos, dual_mobility, \
    remove_bias, horizontal )
```

### Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances and `NumClusters` are used
- **`set_type`** — `"lsf"` or `"cluster"`; `"ray"` is not available yet; str
- **`seed`** — Seed of the site and scenario; int from -1 to 2^63-1; -1 = system random device; default: -1
- **`set_id`** — `"cluster"`: `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`; default: -1
- **`preset`** — Sum-of-sinusoids preset, see [sos.preset](sum-of-sinusoids-random-number-generator.md#sos.preset); sets `n_coeff`; default: `"Comb256"`
- **`tx_pos`**, **`rx_pos`** — `"lsf"` only: positions in m for the bias removal, as in [gbsm.draw](#gbsm.draw); default: None
- **`dual_mobility`**, **`remove_bias`**, **`horizontal`** — `"lsf"`: Tx and Rx both move; remove the sample mean; 2D
  bias removal (TR 38.901), `False`: 3D (legacy QuaDRiGa); default: False, True, True

### Output:
- **`phase`** — Phases in rad, in [-pi, pi); slice 0 = Rx side, slice 1 = Tx side; `(n_coeff, n_gen, 2)`

### See also:
- [gbsm.draw](#gbsm.draw) (evaluates the phases at the links)
- [sos.remove_bias](sum-of-sinusoids-random-number-generator.md#sos.remove_bias) (the bias removal of one generator)

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
- The `LOS_state_lambda` of the InF conf files is `d_clutter/2` of the TR defaults; with another `d_clutter` pass a
  matching correlation distance

### Usage:
```python
p_los, los_soft = quadriga_lib.gbsm.los_probability( model, d2d_out, h_bs, h_ut, d_clutter, r_clutter, \
    h_clutter, rnd_g, center_freq )
```

### Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`, `38.901_InH_Open`,
  `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`, `38.901_InF_HH`, `38.901_SMa`,
  `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in m, finite and >= 0: `d2D - d2D-in` for O2I UTs, clamped at 0 by the caller; the
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
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0 (messages name it `center_frequency`); `(n_freq,)`;
  default: None

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
  7.6.8-1. The imaginary part is >= 0: the complex value of TR 38.901 eq. (7.6-40) is `gr_eps.conj()`
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e + EoD_offset_e_gamma·lg
  f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg f')·lg(max(EoD_offset_b, d)) + EoD_offset_c +
  EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if `EoD_offset_b > 0`, plus `(atan((EoD_offset_hcl - h)/d) -
  atan((hUT - h)/d))·180/pi` if `hUT < EoD_offset_hcl`
- `complex = True` returns the permittivity as one complex array `gr_eps = gr_eps_re + 1j·gr_eps_im`; `complex = False`
  returns the two real arrays `gr_eps_re`, `gr_eps_im` in its place (one more output)

### Usage:
```python
# complex=True (default)
lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( scenpar, center_freq, \
    tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads, complex=True )

# complex=False
lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( ..., \
    complex=False )
```

### Inputs:
- **`scenpar`** — Scenario parameters; the LSP cross-correlation matrix must be positive definite
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0 (messages name it `center_frequency`); `(n_freq,)`
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
- **`complex`** — See the description; bool or `numpy.bool_`, `None` selects the default, another type raises
  `TypeError`; default: True

### Outputs:
- **`lsp`** — Large-scale parameters; slices: 0 DS in s, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA in deg, 7 XPR; `(n_link,
  n_freq, 8)`
- **`abs_toa`** — Absolute TOA offset in s, 0 if not active; `(n_link,)`
- **`gr_eps`** — Relative permittivity of the ground (`complex=True`), imaginary part >= 0; complex128; `(n_link,
  n_freq)`
- **`gr_eps_re`**, **`gr_eps_im`** — Real and imaginary part instead of `gr_eps` (`complex=False`); `(n_link, n_freq)`
- **`eod_offset_deg`** — EoD offset in deg, positive = up; `(n_link, n_freq)`
- **`lsp_mu`** — Mean `mu` in the log domain; `(n_link, n_freq, 8)`
- **`lsp_sigma`** — Standard deviation `sigma` in the log domain, >= 0; `(n_link, n_freq, 8)`

### See also:
- [gbsm.draw](#gbsm.draw) (the random values `rnd`)
- [path_loss](#path_loss) (the SF standard deviation `sf_sigma`)
- [ssf_clusters](#ssf_clusters) (the clusters from the LSPs)

---
<a id="o2i_loss"></a>
## o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

Also in: [C++](../cpp/geometry-based-stochastic-model.md#o2i_loss) · [MATLAB](../mex/geometry-based-stochastic-model.md#o2i_loss)

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier frequency; the
  caller adds `loss` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table 7.4.3-1), `legacy_6GHz`
  (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz; the O2I conf files of UMa and UMi have its `SF_sigma = 7`),
  `car` and `car_metal` (clause 7.4.3.2, metallized windows)
- Model of UT `u`: the first model `j` with `rnd(u, 0)` below the sum of the weights up to `j`, else the last model with
  a weight > 0; a model with weight 0 is never chosen
- Indoor distance: the minimum of columns 1 and 2 of `rnd` times `d2d_in_max` (`low`, `high`, `low_A`, clause 7.4.3.1),
  column 1 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [sos.rand](sum-of-sinusoids-random-number-generator.md#sos.rand) and [sos.randn](sum-of-sinusoids-random-number-generator.md#sos.randn) at the correlation distances of TR 38.901 clause 7.6.3.3: 50 m for
  column 0, 25 m for columns 1 and 2, 10 m for column 3
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`, SMa all three. SMa
  has 90 % residential and 10 % commercial buildings (Table 7.2-5) with different maximum indoor distances: call
  `o2i_loss` once per building type

### Usage:
```python
loss, d2d_in, model_index = quadriga_lib.gbsm.o2i_loss( center_freq, rnd, models, weights, d2d_in_max )
```

### Inputs:
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0 (messages name it `center_frequency`); `(n_freq,)`
- **`rnd`** — Random values per UT: columns 0 to 2 Uniform in [0, 1] (model choice, two indoor distances), column 3
  Normal(0,1), finite (deviation `sigma_P`, car loss); `(n_ut, 4)`, `n_ut = 0` allowed
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; list of str or one str;
  `n_model` entries
- **`weights`** — Probability of each model, finite and >= 0, sum 1 within 1e-6; `(n_model,)`
- **`d2d_in_max`** — Maximum indoor 2D distance in m, finite and >= 0 (key `O2I_d2D_in_max`; TR: 25 m for UMa, UMi and
  SMa commercial buildings, 10 m for RMa and SMa residential buildings); default: 25.0

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
  - `tripple_slope` (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0, sig1..sig3): the
    `dual_slope` rule with the breakpoints `dBP1 = min(dBP1, dBP2)` and `dBP2` of `E1, hE1` and `E2, hE2`, the slopes
    `A1`, `A2`, `A3` and `sf = sig1`, `sig2`, `sig3`
  - `nlos` (An, Bn; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, Ax, Bx, Cx, sig): `hb =
    max(hBS, 0.1)`, `hm = max(hUT, 0.1)`; `An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb + D2n·lg(hb)/hb² +
    D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; with
    `LOSmax != 0` (which requires the `dual_slope` keys) at least the `dual_slope` loss; with `Ax`, `Bx` or `Cx` at
    least `Ax·lg d3 + Bx + Cx·lg f` (e.g. the InF-SL term of InF-DL); `sf = sig`
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS/d3, 1)`; `+Inf` for `sa <= 0`, else `A·lg
    d3 + B + C·lg f + D·lg(asin(sa))`, plus the zenith attenuation by atmospheric gases over `sa` (1 to 100 GHz,
    constant beyond) if `usePLa != 0`
  - no model (`plpar` is `None`, `{}` or has no entry `model`): 0
- Where the model sets no `sf` (a missing `SF`, `sig` key or no such key): the SF std of [lsp_generate](#lsp_generate) at the
  geometric distance, so that passing `sf_sigma` to [lsp_generate](#lsp_generate) is neutral
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
  scalar or 1D array per further parameter, named without "PL_"; keys not read are ignored
- **`scenpar`** — Scenario parameters; the SF std terms and `LSP_fc_min` are used
- **`center_freq`** — Carrier frequencies in Hz, finite and > 0 (messages name it `center_frequency`); `(n_freq,)`
- **`tx_pos`** — Tx positions in m; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`d2d_in`** — Indoor 2D distance in m of the UT of each link (e.g. from [o2i_loss](#o2i_loss)), finite and >= 0; given =
  legacy outdoor distance; 0 = outdoor link; `None` or empty = geometric distance; `(n_link,)`; default: None
- **`rnd_he`** — Uniform(0,1) value per link for the random environment height, column 10 of [gbsm.draw](#gbsm.draw) (`"lsf"`);
  `None` or empty = fixed `hE`; `(n_link,)`; default: None

### Outputs:
- **`pl`** — Path loss in dB; `(n_link, n_freq)`
- **`sf_sigma`** — Shadow-fading standard deviation in dB; `(n_link, n_freq)`

### See also:
- [lsp_generate](#lsp_generate) (takes `sf_sigma`)
- [o2i_loss](#o2i_loss) (penetration loss and indoor distance)

---
<a id="scenpar_validate"></a>
## scenpar_validate
Validate scenario parameters and complete them with default values

Also in: [C++](../cpp/geometry-based-stochastic-model.md#scenpar_validate) · [MATLAB](../mex/geometry-based-stochastic-model.md#scenpar_validate)

- Checks scenario parameters that did not come from a conf file and adds every missing key with its default value, so
  that `scenpar_out` has all 171 keys: the 138 numeric keys of QuaDRiGa and 33 keys for TR 38.901 v19.5.0, which have
  no effect at their default of 0. `None` or an empty dict gives the complete default set
- A name that is not one of the keys is an error, also with `validate = False`; this includes the QuaDRiGa key
  `GR_epsilon`, which is not supported
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
- With `validate = False` no value is checked: `NaN` is accepted

### Usage:
```python
scenpar_out, lsp_xcorr, lsp_xcorr_chk = quadriga_lib.gbsm.scenpar_validate( scenpar_in, validate )
```

### Inputs:
- **`scenpar_in`** — Scenario parameters; dict with any subset of the 171 keys returned by
  [conf_file_read](#conf_file_read); real scalars (int, float, bool or a numpy value with one element); `None` or
  `{}` returns the default set; default: None
- **`validate`** — `False`: only complete with defaults, no value checks; bool or `numpy.bool_`;
  default: True; `None` selects the default

### Outputs:
- **`scenpar_out`** — Complete scenario parameters; given values are kept, missing keys have their
  default; dict with 171 float entries in the canonical order: the 138 numeric keys of QuaDRiGa, then
  the 33 new keys
- **`lsp_xcorr`** — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite; `False` also if it contains
  `NaN`; bool

### See also:
- [conf_file_read](#conf_file_read) (read the parameters from a conf file)
- [conf_file_write](#conf_file_write) (applies the same checks before writing)

---
<a id="ssf_clusters"></a>
## ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

Also in: [C++](../cpp/geometry-based-stochastic-model.md#ssf_clusters) · [MATLAB](../mex/geometry-based-stochastic-model.md#ssf_clusters)

- Turns the LSPs of [lsp_generate](#lsp_generate) and the cluster random values of [gbsm.draw](#gbsm.draw) into the clusters of each link, by
  one of three methods (a method and property table is in the C++ documentation of `ssf_clusters`):
  - `"38.901"` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters 25 dB
    below the strongest get power 0 (no renormalisation); the K-factor in the scaling factors is limited to >= -5 dB.
    Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` at the drop or segment start and keep
    them along the track
  - `"38.901_MF"`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `"quadriga"`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links
- Slots per link: 0 = LOS path, 1 = ground reflection if `GR_enabled > 0` (`G = 1`), then the `N = NumClusters - 1` NLOS
  clusters; `n_slot = NumClusters + G`. Slot `1 + G + n` belongs to column `n` of each block of `rnd` (no sorting).
  Clusters `n >= n_cluster[l]` (TR clause 7.6.15) get delay 0, power 0 and the LOS angles. Identify the LOS path by its
  slot, never by delay or path length: NLOS clusters of the TR methods can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever `lsp`
  holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in s relative to the LOS delay `d3D/c`; angles in rad, GCS, elevation convention; slot 0 holds the geometric
  LOS direction. Powers are the coefficient powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters `P_n/(K+1)`, without
  path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `"38.901"`, `"38.901_MF"` add `abs_toa` to the NLOS delays;
  `"quadriga"` places the first NLOS cluster at `abs_toa` (where `absTOA_mu <= -30`: only where `abs_toa > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0: slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of the
  other slots) or `K/(K+1)·R²` (`"quadriga"`), with `R²` the mean of `abs(R_par)²` and `abs(R_perp)²` of TR eqs.
  (7.6-38), (7.6-39); the coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarisation
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios, plus the ground reflection if enabled; `rnd` must be
  empty, `rnd_freq` and `lsp_anchor` are not used
- Swap reciprocity: with mirrored random values (dual mobility in [gbsm.draw](#gbsm.draw)) and mirrored LSPs, swapping `tx_pos`
  and `rx_pos` exchanges the departure and arrival angles (`"38.901"`: `eod_offset_deg` = 0 and no O2I link;
  `"quadriga"`: `eod_offset_deg` = 0)
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario; only their sizes are checked

### Usage:
```python
delay, power, aod, eod, aoa, eoa, n_active = quadriga_lib.gbsm.ssf_clusters( scenpar, tx_pos, rx_pos, lsp, rnd, \
    method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps, o2i, n_cluster )
```

### Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`,
  `KF_mu`, `absTOA_mu`, `r_DS`, `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in m; z >= 0 with ground reflection; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; z >= 0 with ground reflection; `(n_link, 3)`
- **`lsp`** — LSPs from [lsp_generate](#lsp_generate); DS, ASD, ASA, ESD, ESA (slices 0, 3 to 6) > 0, KF (slice 1) >= 0;
  `"38.901"`: one frequency; `(n_link, n_freq, 8)`
- **`rnd`** — Shared cluster set of [gbsm.draw](#gbsm.draw) (`"cluster"`, `set_id = -1`), columns 0 to N-1 in [0, 1]; `(n_link,
  9·N)`
- **`method`** — `"38.901"`, `"38.901_MF"` or `"quadriga"`; default: `"38.901"`
- **`rnd_freq`** — Per-frequency cluster sets of [gbsm.draw](#gbsm.draw), slice `k` from `set_id = k`; required by `"38.901"` and
  `"38.901_MF"`; `(n_link, N, n_freq)`; default: None
- **`lsp_anchor`** — LSPs at the anchor frequency (e.g. 2 GHz) from [lsp_generate](#lsp_generate); required by `"38.901_MF"`;
  `(n_link, 1, 8)`; default: None
- **`eod_offset_deg`** — EoD offset in deg from [lsp_generate](#lsp_generate); used by `"38.901"` and `"quadriga"` in NLOS
  scenarios; `None` = 0; `(n_link, n_freq)`; default: None
- **`abs_toa`** — Absolute TOA offset in s from [lsp_generate](#lsp_generate), >= 0; required by `"quadriga"` in NLOS scenarios
  with `absTOA_mu > -30`; `None` = no offset; `(n_link,)`; default: None
- **`gr_eps`** — Ground permittivity as [lsp_generate](#lsp_generate) returns it (complex, real part >= 1, imaginary part >= 0,
  used as `real - j·imag`; a real array = imaginary part 0); required if `GR_enabled > 0`; `(n_link, n_freq)`;
  default: None
- **`o2i`** — O2I flag per link (mean ZOA 90 deg), bool or numeric array (nonzero = O2I); used by `"38.901"` in
  NLOS scenarios; `None` = none; `(n_link,)`; default: None
- **`n_cluster`** — Active NLOS clusters per link in `[min(max(1, NumClusters_min - 1), N), N]`, int or float
  array of non-negative integers; `None` = N; `(n_link,)`; default: None

### Outputs:
- **`delay`** — Delays in s; `(n_link, n_slot, n_freq)` for `"38.901_MF"`, else `(n_link, n_slot, 1)`
- **`power`** — Coefficient powers (linear); `(n_link, n_slot, n_freq)`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in rad, in [-pi, pi); shapes as `delay`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in rad, in [-pi/2, pi/2]; shapes as `delay`
- **`n_active`** — Number of NLOS clusters with power > 0; uint64; `(n_link,)`

### See also:
- [gbsm.draw](#gbsm.draw) (the random values `rnd` and `rnd_freq`)
- [lsp_generate](#lsp_generate) (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [calc_delay_spread](channel-statistics.md#calc_delay_spread), [calc_angular_spread](channel-statistics.md#calc_angular_spread) (spreads of the output)
