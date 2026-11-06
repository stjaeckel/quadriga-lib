# Geometry-based stochastic model

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-geometry-based-stochastic-model

| Function | Summary |
| --- | --- |
| [conf_file_list](#conf_file_list) | List the scenarios defined by the conf files in a set of directories |
| [conf_file_read](#conf_file_read) | Read and validate a QuaDRiGa scenario configuration file |
| [conf_file_write](#conf_file_write) | Write a QuaDRiGa scenario configuration file |
| [gbsm_draw](#gbsm_draw) | Spatially correlated normalized random values of one generator set at a set of links |
| [gbsm_init_phases](#gbsm_init_phases) | Draw the random phases of one generator set of a BS site and scenario |
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
| [ssf_scatterers](#ssf_scatterers) | First- and last-bounce scatterers of the rays of one link (two-bounce solver) |

A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of
environment, the *scenario* (e.g. urban macro-cell with or without line of sight), instead of from
the geometry of one site. The functions of this section implement 3GPP TR 38.901 clause 7.5 steps 1
to 11 (path loss, LOS probability, O2I penetration loss, large-scale parameters (LSPs), clusters and
rays, without antennas) with the spatial consistency of clause 7.6.3, oxygen absorption (clause
7.6.1) and polarisation power variability (clause 7.6.16); [gbsm_single_link_channel](#gbsm_single_link_channel) computes the
channel coefficients of a link with its antennas.

**Scenario parameters.** A scenario is a struct of numeric parameters (`scenpar`), a struct of
path-loss parameters (`plpar`) and optional short names, read from a conf file by
[conf_file_read](#conf_file_read). 23 files for TR 38.901 v19.5.0 come with the library, in `data/config` of the
source tree and in `share/quadriga-lib/config` of an installed package; [conf_file_list](#conf_file_list) finds
them and [conf_file_write](#conf_file_write) writes them. Every function that takes `scenpar` accepts the struct of
[conf_file_read](#conf_file_read) or [scenpar_validate](#scenpar_validate) or any subset of its fields (`[]` = all defaults; a field
`SubpathMethod` of QuaDRiGa's structs must be `'legacy'`), completes it with the defaults of missing
fields and validates it with [scenpar_validate](#scenpar_validate); an invalid value of a field that the function
does not use is an error as well.

**Workflow.** Per drop: [o2i_loss](#o2i_loss) gives the penetration loss and the indoor distance of each
indoor UT, then [los_probability](#los_probability) the LOS probability at the outdoor distance `d2D - d2D-in`. The
caller draws the LOS, indoor and O2I states with [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand) / [sos_randn](sum-of-sinusoids-random-number-generator.md#sos_randn) at the correlation
distances of TR 38.901 clause 7.6.3.3; the fields `LOS_state_lambda` and `IO_state_lambda` apply to
all links of a scenario and are stated in the NLOS and in the O2I files (0 in the other files means
"not stated here"). Per BS site and scenario: [gbsm_init_phases](#gbsm_init_phases) draws the phases of each random
set once (`'lsf'`; `'cluster'` with `set_id = 0` and one set per carrier frequency; `'ray'` with
`set_id = 0` for the method `'quadriga'`, one set per carrier frequency for the TR methods), and
[gbsm_draw](#gbsm_draw) evaluates them at the links. Then [path_loss](#path_loss) gives the path loss and the
shadow-fading (SF) standard deviation, [lsp_generate](#lsp_generate) the LSPs of each link and frequency,
[ssf_clusters](#ssf_clusters) the cluster delays, powers and angles, and [ssf_rays](#ssf_rays) the rays of each link as a
ray struct (one call for `'quadriga'`, one per frequency for the TR methods). For the mobility
`drift`, [ssf_scatterers](#ssf_scatterers) places the scatterers of each link and returns `b_med`, which sets the
segment length, and `b_eff`. Per link and segment, [gbsm_single_link_channel](#gbsm_single_link_channel) computes the
coefficients with `link_gain_dB` = -PL - O2I + 10·lg SF and `kf_ratio` = K(position) / K(drop) from
the large-scale functions; the sum and the ratio are the caller's. Path data read with
[qrt_file_read](channel-functions.md#qrt_file_read) become a ray struct with [gbsm_rays_from_paths](#gbsm_rays_from_paths); other path sources fill a ray
struct directly.

**Terms.** TR 38.901 names where they exist; QuaDRiGa names are marked:

| Term | Meaning |
| --- | --- |
| scenario | type of environment with its statistics: numeric fields `scenpar`, path-loss model, short names (one conf file) |
| link | one Tx-Rx pair of a batch (row `l` of `rx_pos`) |
| drop | one realisation of LSPs, clusters and rays at the link positions; the positions the ray data refer to |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 38.901 clause 7.5 steps 5 to 7) |
| slot | column of the [ssf_clusters](#ssf_clusters) outputs: 1 LOS, 2 GR (if enabled), then the N clusters |
| sub-cluster | one of the three parts of a split cluster (step 11, Table 7.5-5) |
| ray | one of the M rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) |
| path | a propagation path as `get_channels_*` take it; [ssf_paths](#ssf_paths) makes one per kept ray (QuaDRiGa also says path for a cluster) |
| tap | rays summed in the `'taps'` output: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`'quadriga'`) |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed (QuaDRiGa) |
| segment | the snapshots of one drop over which the ray data are kept (QuaDRiGa) |
| mobility | how the paths follow moved ends within a segment: `phase` (phase rotation per ray, delays and angles fixed) or `drift` (scatterers fixed, geometry recomputed; QuaDRiGa drifting) |
| scatterer | first-bounce (FBS) and last-bounce (LBS) interaction point of a ray; one point for LOS (direct line) and GR (ground) (QuaDRiGa) |
| coefficient power | power of a ray without link gain and oxygen loss (field `power` of the ray struct): the powers of TR 38.901 eq. (7.5-30) split over the rays, GR as [ssf_clusters](#ssf_clusters) sets it; for data from paths the path gain |
| path gain | linear power gain of a path incl. link gain and oxygen loss |
| link gain | -PL - O2I + SF in dB of a link at a position (`link_gain_dB`) |
| random set | the random values of one `set_type` and `set_id`, drawn from one phase array; never changes size |
| generator | one spatially consistent random field; one column of `rnd` |
| block | the generators of one random quantity in a set (N or N·M columns) |

**Codes.** Path type: 0 LOS, 1 GR, 2 NLOS. Tap index: 3·(s - 1) + i, with s the column of the
[ssf_clusters](#ssf_clusters) outputs and i the sub-cluster (0, 1, 2) of a split cluster and 0 otherwise, so LOS
0, GR 3 and NLOS cluster `n` (from 1) 3·(G + n) + i; `'taps'` lists the taps in ascending index.
Both are codes, not indices: the wrappers do not shift them.

**Model properties.**

- Separately dropped UTs keep correlated taps over 43 to 440 wavelengths, while measured fading
  decorrelates within about 0.4 to 1 wavelength: for UTs a few wavelengths apart, use one drop and
  move the UTs (see [ssf_paths](#ssf_paths))
- `'quadriga'`: the per-tap correlation across bands follows the intra-cluster delays of one random
  field; TR methods: independent per band (TR 38.901 clause 7.6.5)
- `pol_model = 'quadriga'` gives a nearly unitary 2x2 cluster matrix (singular-value ratio 0.10 to
  0.22 dB, TR model 7.6 to 8.3 dB)
- The validity of the mobilities `phase` and `drift` is stated in [ssf_paths](#ssf_paths)



**Random sets.** The spatially correlated random values come in sets of generators, one column of
`rnd` each; `N = NumClusters - 1`, columns `b·N + 1` to `(b + 1)·N` of a `'cluster'` set form block
`b`, one column per NLOS cluster:

| `set_type`, `set_id` | `n_gen` | Columns and distributions | Decorrelation distance | Mirror pairs | Used by |
| --- | --- | --- | --- | --- | --- |
| `'lsf'` | 11 | 1 to 8 DS, KF, SF, ASD, ASA, ESD, ESA, XPR, Normal(0,1); 9 absolute TOA, Normal(0,1); 10 ground permittivity, 11 UMa environment height hE, Uniform(0,1) | `DS_lambda` ... `XPR_lambda` (1 to 8), `absTOA_lambda` (9), `SC_lambda` (10), `SF_lambda` (11) | 4 and 5, 6 and 7 | [lsp_generate](#lsp_generate) `rnd`, [path_loss](#path_loss) `rnd_he` (column 11) |
| `'cluster'`, 0 | `9·N` | block 0 cluster delay, Uniform(0,1); 1, 2 sign of AOD, AOA; 3, 4 sign of ZOD, ZOA; 5, 6 offset of AOD, AOA; 7, 8 offset of ZOD, ZOA, Normal(0,1) | `SC_lambda` | blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 | [ssf_clusters](#ssf_clusters) `rnd` |
| `'cluster'`, `k` | `N` | cluster shadowing of frequency `k`, Normal(0,1) | `SC_lambda` | none | [ssf_clusters](#ssf_clusters) `rnd_freq`, slice `k` |
| `'ray'`, 0 | `14·N·M + N` | blocks 0 to 14 (below) | `SC_lambda`; block 14: $\lambda_\tau$ | below | [ssf_rays](#ssf_rays) `rnd_ray`, method `'quadriga'` |
| `'ray'`, `k` | `14·N·M + N` | blocks 0 to 14 (below), block 14 unused | `SC_lambda` | below | [ssf_rays](#ssf_rays) `rnd_ray`, slice `k`, TR methods |

A `'ray'` set holds, with `M = NumSubPaths` and ray `m` of NLOS cluster `n` (both from 1), block `b`
up to 12 in column `b·N·M + (n - 1)·M + m`, block 13 in column `13·N·M + n` and block 14 in column
`13·N·M + N + (n - 1)·M + m`; all blocks are Normal(0,1) except block 14, Uniform(0,1):

| Block | Random value | Mirror pair | Used by |
| --- | --- | --- | --- |
| 0, 1 | coupling rank of AOD, AOA (TR 38.901 clause 7.5 step 8) | 0 and 1 | all methods |
| 2, 3 | coupling rank of ZOD, ZOA | 2 and 3 | all methods |
| 4 | XPR per ray | none | both `pol_model` |
| 5 | initial phase θθ (`pol_model = 'quadriga'`: its only phase) | none | both `pol_model` |
| 6, 7 | initial phases θφ, φθ | 6 and 7 | `pol_model = '38.901'` |
| 8 | initial phase φφ | none | `pol_model = '38.901'` |
| 9 | polarisation power variability θθ (TR 38.901 clause 7.6.16) | none | `pol_variability` |
| 10, 11 | the same, θφ and φθ | 10 and 11 | `pol_variability` |
| 12 | the same, φφ | none | `pol_variability` |
| 13 | XPR per cluster (`N` columns) | none | `pol_model = 'quadriga'` |
| 14 | ray delay | none | method `'quadriga'` |

Block 14 of the shared `'ray'` set has the decorrelation distance
$\lambda_\tau = 1.6 \cdot c \cdot \bar{c}_\mathrm{DS} / c_\mathrm{ASA}$, limited to 0.1 m to
10 km, with $\bar{c}_\mathrm{DS}$ the mean cluster delay spread (TR 38.901 Table 7.5-6) over the
frequencies passed to [gbsm_draw](#gbsm_draw) and $c_\mathrm{ASA}$ the cluster ASA in rad.

The Tx side of a phase array (slice 2) exchanges the columns of each mirror pair, so that in dual
mobility (Tx and Rx both move) the departure quantities at the Tx are the arrival quantities of the
Tx as a second device. A decorrelation distance of 0 is used as 0.1 m, except for $\lambda_\tau$.
Only the `'lsf'` columns 1 to 9 take part in the bias removal. A set never changes size: new random
quantities come in new sets (a new `set_type` or `set_id`).

**Conventions.** Positions are in m with the columns x, y, z: `rx_pos` is `[n_link, 3]`, `tx_pos` is
`[1, 3]` (one Tx for all links) or `[n_link, 3]`; heights are the z coordinates. In 2D mode
(`horizontal = true`, the TR behaviour) the random values are correlated in the x-y plane, and
`floor_id` separates the floors, which are uncorrelated (TR 38.901 Table 7.5-6 note 3). In 3D mode
(`horizontal = false`, legacy QuaDRiGa) the height separates the floors, so UTs on different floors
are correlated by the ACF of their height difference; the per-link statistics are the same in both
modes. Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default and QuaDRiGa's is
available through an argument where it is still wanted; where QuaDRiGa's behaviour was an error,
only the corrected rule is implemented. Units: frequencies in Hz, path loss in dB, DS in s, KF, SF
and XPR as linear power ratios, spreads in degrees.

---
<a id="conf_file_list"></a>
## conf_file_list
List the scenarios defined by the conf files in a set of directories

Also in: [C++](../cpp/geometry-based-stochastic-model.md#conf_file_list) · [Python](../python/geometry-based-stochastic-model.md#conf_file_list)

- Lists the regular files and symbolic links to regular files named `<scenario>.conf`
  (case-sensitive, non-empty scenario name) in each directory, not recursive: the directories in the
  given order, the files of one directory in byte-wise sorted order of their names
- A file name found in an earlier directory is skipped, so a local folder can override the files of
  a shared one
- With `parse_shortnames = true` every line `ShortName = name` that is not a comment adds an entry
  right after its file; other lines are not examined
- An error is raised if a short name equals an earlier entry, if a file name equals an earlier short
  name, if a `ShortName` value is not one name, or if a listed file cannot be opened
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty
  string `''` is the current directory

### Usage:
```matlab
[ scenarios, file_names, file_dirs ] = quadriga_lib.conf_file_list( dirs, parse_shortnames );
```

### Inputs:
- **`dirs`** — Directories, searched in this order; one string or a cell array of strings; an empty
  cell array lists nothing
- **`parse_shortnames`** — `true`: read the files and add their short names; `false`: list the files
  only, which is faster; default: true

### Outputs:
- **`scenarios`** — Scenario names and short names; cell array of strings; `[n_scen, 1]`
- **`file_names`** — File name including `.conf` for each entry; cell array of strings;
  `[n_scen, 1]`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs`; cell array of
  strings; `[n_scen, 1]`

### See also:
- [conf_file_read](#conf_file_read) (read one of the files)

---
<a id="conf_file_read"></a>
## conf_file_read
Read and validate a QuaDRiGa scenario configuration file

Also in: [C++](../cpp/geometry-based-stochastic-model.md#conf_file_read) · [Python](../python/geometry-based-stochastic-model.md#conf_file_read)

- A conf file has one `Key = value` assignment per line; `%` starts a comment; blank lines, LF or
  CRLF line ends and a UTF-8 byte order mark at the start are accepted; keys are case-sensitive
  names of `A-Z`, `a-z`, `0-9` and `_`
- Three kinds of keys:
  - 171 numeric scenario parameters (see [scenpar_validate](#scenpar_validate)), one number each; keys that are not
    in the file have their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the
  QuaDRiGa key `GR_epsilon` (the complex ground permittivity comes from [lsp_generate](#lsp_generate)). A key
  stated more than once has its last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of
  `SubpathMethod` is an error, also with `validate = false`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`),
  read independent of the locale; `Inf`, `+Inf` and `-Inf` in any case; a number below the smallest
  subnormal reads as 0
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- An error is raised if the file cannot be opened or is a directory
- An error with file name and line number for a line that is not blank, not a comment and not a
  well-formed assignment (no `=`, not one name before the `=`, a missing value, a word where a
  number is required or the reverse, text after the value, `NaN`, a number beyond the range of a
  double, a `PL_` key without a name, a `ShortName` value stated twice), and with `validate = true`
  for a value that fails [scenpar_validate](#scenpar_validate)

### Usage:
```matlab
[ scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys ] = ...
    quadriga_lib.conf_file_read( fn, validate );
```

### Inputs:
- **`fn`** — Path to the conf file; string
- **`validate`** — `true`: apply the value checks of [scenpar_validate](#scenpar_validate); `false`: only the line
  grammar is checked; default: true

### Outputs:
- **`scenpar`** — Scenario parameters; struct with 171 scalar fields in the canonical order: the 138
  numeric keys of QuaDRiGa, then the 33 new keys
- **`plpar`** — Path-loss parameters; struct with the field `model` (string, only if the file states
  `PL_model`) followed by one field per `PL_` key in file order, named without the prefix `PL_`;
  each is a row vector with one or more values; `[]` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be
  generated from it; logical
- **`short_names`** — Values of all `ShortName` lines in file order; cell array of strings;
  `[n_short, 1]`
- **`ignored_keys`** — Unknown keys, each once, in the order of their first occurrence; cell array
  of strings; `[n_ignored, 1]`

### See also:
- [scenpar_validate](#scenpar_validate) (value checks and defaults)
- [conf_file_list](#conf_file_list) (find the conf files in a set of directories)
- [conf_file_write](#conf_file_write) (write a conf file)

---
<a id="conf_file_write"></a>
## conf_file_write
Write a QuaDRiGa scenario configuration file

Also in: [C++](../cpp/geometry-based-stochastic-model.md#conf_file_write) · [Python](../python/geometry-based-stochastic-model.md#conf_file_write)

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files
  of QuaDRiGa; every written file passes [conf_file_read](#conf_file_read) with `validate = true`, without ignored
  keys and with the same values bit for bit
- The header has the scenario name (the file name without `.conf`), `ref`, the validity range and
  the short names. Up to 10 sections follow, each with an optional line `% See: ref; ref_sec` and,
  for the LSPs and the path-loss model, the formulas as comments; the `plpar` fields are written in
  struct order. Sections by entry of `ref_sec`: 1 large-scale distributions, 2 large-scale
  decorrelation distances, 3 inter-parameter correlations, 4 model parameters, 5 small-scale
  decorrelation distance, 6 ground reflection, 7 absolute TOA, 8 path-loss model, 9 EoD offset, 10
  optional modelling components; 1 to 8 are those of QuaDRiGa, so its 8 section references can be
  passed unchanged
- Numbers have up to 15 significant digits, or 16 or 17 where 15 do not read back to the same
  double, independent of the locale; `Inf`, `-Inf` and `-0` are kept
- `write_defaults = false`: only fields whose value differs from the default (bit for bit); a
  section without a written field is left out
- All inputs are validated before the file is opened: the checks of [scenpar_validate](#scenpar_validate); `model`,
  the other fields of `plpar` and the short names are names, the short names distinct; every `plpar`
  field has at least one value and no `NaN`; `fn`, `ref` and `ref_sec` have no line break
- The text is written to a temporary file in the same directory, which must be writable, and renamed
  to `fn`: a failed call creates no file and leaves an existing file unchanged; an existing file
  keeps its permissions, a symbolic link `fn` is written through

### Usage:
```matlab
quadriga_lib.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, ...
    write_defaults );
```

### Inputs:
- **`fn`** — Path of the conf file to write; string
- **`scenpar`** — Scenario parameters; struct with any subset of the 171 fields returned by
  [conf_file_read](#conf_file_read); real scalars of any numeric class or logical; missing fields have their default; `[]` writes
  the default set
- **`plpar`** *(optional)* — Path-loss parameters; struct with the field `model` (string) and one real vector (row
  or column) of any numeric class per further field, named without the prefix `PL_`, as returned by
  [conf_file_read](#conf_file_read); a matrix is an error; written in struct order; if omitted or `[]`, no path-loss section
- **`short_names`** *(optional)* — Short names of the scenario; cell array of strings or one string;
  each one name of the characters `A-Z`, `a-z`, `0-9`, `_`; distinct
- **`ref`** — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; string; default: no reference
- **`ref_sec`** *(optional)* — Reference per section, numbered as listed above; cell array of up to
  10 strings; an empty or missing entry writes no reference
- **`valid_range`** *(optional)* — Validity range, written as comment; rows: frequency in GHz,
  distance, BS height, MT height in m; columns: lower and upper end, one value is written if both
  are equal; a row with `NaN` is not written; `[4, 2]`
- **`write_defaults`** — `true`: write all fields; `false`: write only fields that differ from their
  default; default: false

### See also:
- [conf_file_read](#conf_file_read) (read the file back)
- [scenpar_validate](#scenpar_validate) (value checks and defaults)

---
<a id="gbsm_draw"></a>
## gbsm_draw
Spatially correlated normalized random values of one generator set at a set of links

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_draw) · [Python](../python/geometry-based-stochastic-model.md#gbsm.draw)

- Evaluates the phases of [gbsm_init_phases](#gbsm_init_phases) at the links, one column per generator; columns,
  distributions and decorrelation distances: see the set table in the section description. The value
  of a link depends neither on the other links of the call nor on the other columns; in single
  mobility it does not depend on `tx_pos`
- Dual mobility: the Tx is a second device on the same maps, its side of each generator uses the
  mirror generator. Every column keeps its distribution; swapping `tx_pos` and `rx_pos` exchanges the
  columns of each mirror pair and keeps the self-mirrored columns, exactly. The ray delays (block 14
  of the shared `'ray'` set) change 2 to 8 times faster with a Tx move than plane waves at the Tx
  would (one decorrelation distance, from the cluster ASA)
- Floors (2D mode): links with equal `floor_id` share one field; fields of different floors are
  uncorrelated at equal horizontal positions, exactly within a block of `n_coeff` consecutive IDs,
  otherwise up to a random residual of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both
  devices of a link are on the link's floor
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation
- A set never changes size. An empty set (`NumClusters = 1`) gives `[n_link, 0]` and accepts any
  empty `phase`

### Usage:
```matlab
rnd = quadriga_lib.gbsm_draw( scenpar, set_type, phase, tx_pos, rx_pos, set_id, preset, floor_id, ...
    center_freq, dual_mobility, horizontal );
```

### Inputs:
- **`scenpar`** — Scenario parameters; decorrelation distances, `NumClusters`, `NumSubPaths`,
  cluster DS and ASA
- **`set_type`** — Set type of [gbsm_init_phases](#gbsm_init_phases); string
- **`phase`** — Phases from [gbsm_init_phases](#gbsm_init_phases), finite; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility, always validated; `[1, 3]` or
  `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`, `n_link = 0` allowed
- **`set_id`** — Set of [gbsm_init_phases](#gbsm_init_phases): `0` = shared set, `k >= 1` = set of frequency `k`;
  ignored for `'lsf'`; default: 0
- **`preset`** — Preset of [gbsm_init_phases](#gbsm_init_phases); another preset with the same `n_coeff` is not
  detected; default: `'Comb256'`
- **`floor_id`** *(optional)* — Floor index of the UT per link, 0-based (`n_fl - 1` for the UT
  heights of TR 38.901 Table 7.2-1, 0 for outdoor UTs); omitted or empty = floor 0; 2D mode only;
  `[n_link]`
- **`center_freq`** *(optional)* — Carrier frequencies in [Hz] (messages name it
  `center_frequency`); required by the shared `'ray'` set, whose block 14 uses their mean cluster
  delay spread: pass the same list as to [ssf_rays](#ssf_rays) (another list gives other `quadriga` rays);
  ignored and not checked for all other sets; `[n_freq]`
- **`dual_mobility`** — `true`: Tx and Rx both move; default: false
- **`horizontal`** — `true`: 2D mode (TR 38.901); `false`: 3D mode (legacy QuaDRiGa); default: true

### Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

### See also:
- [lsp_generate](#lsp_generate) (the large-scale parameters from the `'lsf'` set)
- [path_loss](#path_loss) (column 11 of the `'lsf'` set as `rnd_he`)
- [ssf_clusters](#ssf_clusters) (the clusters from the `'cluster'` sets)
- [ssf_rays](#ssf_rays) (the rays from the `'ray'` sets)

---
<a id="gbsm_init_phases"></a>
## gbsm_init_phases
Draw the random phases of one generator set of a BS site and scenario

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_init_phases) · [Python](../python/geometry-based-stochastic-model.md#gbsm.init_phases)

- Draws the phases of one random set (set table in the section description) for [gbsm_draw](#gbsm_draw);
  store `phase` or the seed to evaluate the same fields again along tracks or on maps
- The phases are a fixed function of `seed`, `set_type` and `set_id`, equal on all platforms
  (bias-removed `'lsf'` phases within about 2e-14 rad); the sets are independent; each scenario of a
  site needs its own seed
- `'lsf'` bias removal (`remove_bias`, at least 3 rows in `rx_pos`): [sos_remove_bias](sum-of-sinusoids-random-number-generator.md#sos_remove_bias) sets the
  sample mean of columns 1 to 9 over the positions to 0 (with `tx_pos` in dual mobility; 2D mode:
  one floor), removing variance
- TR 38.901 clause 7.4.4 literally: preset `Exp256` without bias removal; `NumClusters = 1`: all
  sets but `'lsf'` empty

### Usage:
```matlab
phase = quadriga_lib.gbsm_init_phases( scenpar, set_type, seed, set_id, preset, tx_pos, rx_pos, ...
    dual_mobility, remove_bias, horizontal );
```

### Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances, `NumClusters` and `NumSubPaths`
  are used
- **`set_type`** — `'lsf'`, `'cluster'` or `'ray'`; string
- **`seed`** — Seed of the site and scenario; integer from -1 to 2^63-1; -1 = system random device,
  new phases on each call; default: -1
- **`set_id`** — `0` = shared set, `k >= 1` = set of frequency `k`; ignored for `'lsf'`; default: 0
- **`preset`** — Sum-of-sinusoids preset, see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset); sets `n_coeff`; default: `'Comb256'`
- **`tx_pos`**, **`rx_pos`** *(optional)* — `'lsf'` only: positions in [m] for the bias removal, as
  in [gbsm_draw](#gbsm_draw)
- **`dual_mobility`**, **`remove_bias`**, **`horizontal`** — `'lsf'`: Tx and Rx both move; remove
  the sample mean; 2D bias removal (TR 38.901), `false`: 3D (legacy QuaDRiGa); default: false, true,
  true

### Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 1 = Rx side, slice 2 = Tx side;
  `[n_coeff, n_gen, 2]`

### See also:
- [gbsm_draw](#gbsm_draw) (evaluates the phases at the links)
- [sos_remove_bias](sum-of-sinusoids-random-number-generator.md#sos_remove_bias) (the bias removal of one generator)

---
<a id="gbsm_rays_from_paths"></a>
## gbsm_rays_from_paths
Ray data of one link from spherical path data (e.g. read with qrt_file_read)

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_rays_from_paths) · [Python](../python/geometry-based-stochastic-model.md#gbsm.rays_from_paths)

- Returns the ray struct of [ssf_rays](#ssf_rays) (`fallback` empty) from path data in the names and order of
  [qrt_file_read](channel-functions.md#qrt_file_read), read with `downlink = true` (an uplink read swaps the ends and conjugates `M`)
- `delay` = (path length - Tx-Rx distance) / c, `power` = path gain. Angles point from the Tx to the first of
  `fbs_pos`, `lbs_pos`, `rx_pos` that differs from it and from the Rx to the first of `lbs_pos`, `fbs_pos`, `tx_pos`;
  GR angles to the image of the other end (z mirrored); 0 if `tx_pos` = `rx_pos`
- QRT stores no path types or taps. Without `path_type`: LOS where FBS = LBS on the direct line (0.1 mm), else
  NLOS; a GR path needs `path_type`
- The GR point (`lbs_pos`) must be within 1 mm of z = 0; permittivity only for `drift` away from `tx_pos`, `rx_pos`
- Use in [ssf_paths](#ssf_paths) and [gbsm_single_link_channel](#gbsm_single_link_channel) with `link_gain_dB = 0` and without `kf_ratio`
- Messages name the argument `center_freq` as `'center_frequency'`

### Usage:
```matlab
rays = quadriga_lib.gbsm_rays_from_paths( center_freq, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, ...
    path_length, M, method, path_type, tap, gr_eps_re, gr_eps_im );
```

### Inputs:
- **`center_freq`** — Frequencies in [Hz]; `[n_freq]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in [m]; `[3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in [m]; `[3, n_path]`
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path]`
- **`M`** — Polarization transfer matrices; `[8, n_path, n_freq]`
- **`method`** *(optional)* — Value of `rays.method`; sets the GR power law and the default mobility; default:
  `'quadriga'`
- **`path_type`** *(optional)* — Path type, 0 = LOS, 1 = GR, 2 = NLOS (a code, not shifted), at most one LOS and
  one GR; `[n_path]`
- **`tap`** *(optional)* — Tap index (a code, not shifted); omitted: LOS 0, GR 3, the others 6, 9, ... in path
  order; `[n_path]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity, given together; `[n_freq]`

### Output:
- **`rays`** — Ray struct with `n_ray = n_path`, see [ssf_rays](#ssf_rays)

### See also:
- [qrt_file_read](channel-functions.md#qrt_file_read) (source of the path data)
- [ssf_paths](#ssf_paths) (paths at other positions)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel coefficients of the link)

---
<a id="gbsm_single_link_channel"></a>
## gbsm_single_link_channel
Channel coefficients of one link at one frequency over one segment (geometry-based stochastic model)

Also in: [C++](../cpp/geometry-based-stochastic-model.md#gbsm_single_link_channel) · [Python](../python/geometry-based-stochastic-model.md#gbsm.single_link_channel)

- Snapshot t: the paths of [ssf_paths](#ssf_paths) at the positions of t, at `rays.center_frequency(i_freq)` with the link
  gain and K ratio of t, for the rays with power > 0 at `i_freq` plus LOS and GR
- `'phase'`: [get_channels_planar](channel-generation-functions.md#get_channels_planar) with the arrays at the positions of `rays` and the orientations of t; delays,
  path lengths and angles stay those of the ray-data positions. `'drift'`: [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) with the
  arrays at the positions and orientations of t
- `'taps'`: the rays summed per tap index, taps in ascending index, one tap per distinct index of the kept rays
  (fixed per call); LOS and GR taps are present whenever `rays` holds them, also at zero power. The delay of a tap
  per element pair is the mean of the rays' delays weighted by `rays.power` at `i_freq` (equal weights where they sum
  to 0), `[n_rx, n_tx, n_tap]`; reduce it to `[1, 1, n_tap]` where needed. TR 38.901 LOS gives two taps at delay 0
- `'taps'` is exact at the phase centre for the TR methods (the rays of a tap share one delay); for `'quadriga'` only
  at the carrier: in UMi NLOS a tap varies over the band by 0.18, 0.52 and 0.79 (in-band selectivity) for 20, 100 and
  400 MHz. `'rays'`: one path per kept ray in ray order, exact at any bandwidth; the path data are the outputs of
  [ssf_paths](#ssf_paths) at t (in `'phase'` with the displacement phase in `M`, for the arrays at the ray-data positions)
- `n_snap` is the largest row count of the positions and orientations (1 if all are omitted); each has 1 or `n_snap`
  rows, `link_gain_dB` and `kf_ratio` 1 or `n_snap` entries; one row or entry applies to all snapshots. One position
  is a `[1, 3]` row; a `[3, 1]` column is rejected. A static link with a gain per snapshot repeats its position
- `use_absolute_delays` in `'phase'` adds the LOS delay of the ray-data positions
- No power correction and no initial position; validity limits and their meaning: see [ssf_paths](#ssf_paths)
- Errors: an invalid array (message after `Transmit antenna: ` or `Receive antenna: `), the errors of [ssf_paths](#ssf_paths)
  in `'drift'`, a row or entry count other than 1 or `n_snap`
- `chan` and `par` are copies of the library result

### Fields of par:

| Name | Size | Content |
| --- | --- | --- |
| `tap` | `[n_tap, 1]` or `[n_path, 1]` | Tap index per output tap or per path (a code, not shifted); uint64 |
| `n_ray_tap` | `[n_tap, 1]` | Number of kept rays per tap; uint64; `'taps'` only |
| `fbs_pos`, `lbs_pos` | `[3, n_kept]` | Scatterers of the kept rays at the ray-data positions; `'drift'` with `'taps'` only |
| `gr_pos` | `[3, n_snap]` | GR point per snapshot; `'drift'` with `'taps'` and a GR ray only |
| `link_gain_dB` | `[n_snap, 1]` | Link gain per snapshot |
| `b_med`, `b_eff` | `[2, 1]` | Validity scales of [ssf_scatterers](#ssf_scatterers) at `i_freq`, Tx and Rx; without scatterers from a solved copy (`d_min` = 2 m), NaN if a GR ray has an end below z = 0 |
| `limit` | `[2, 1]` | Validity limit per end, $\sqrt{\lambda b_{\mathrm{eff}} / 8}$ (`'phase'`) or $0.2 b_{\mathrm{med}}$ (`'drift'`); Inf where b is NaN |
| `ratio` | `[2, 1]` | Largest distance of the end from its ray-data position over the snapshots, divided by `limit` |
| `ratio_total` | scalar | $\sqrt{r_{\mathrm{Tx}}^2 + r_{\mathrm{Rx}}^2}$ of the two ratios; valid up to 1; `'drift'`: new segment from 2.5 |

### Usage:
```matlab
[ chan, par ] = quadriga_lib.gbsm_single_link_channel( tx_array, rx_array, rays, link_gain_dB, mobility, ...
    output, i_freq, tx_pos, rx_pos, tx_orientation, rx_orientation, kf_ratio, oxygen, use_absolute_delays );
```

### Inputs:
- **`tx_array`**, **`rx_array`** — Structs containing the arrayant data with `n_tx` and `n_rx` ports; field layout
  as documented in [arrayant_generate](array-antenna-functions.md#arrayant_generate)
- **`rays`** — Ray struct of the link, see [ssf_rays](#ssf_rays); `[1, 1]` struct
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at `rays.center_frequency(i_freq)` in [dB]; `[n_snap]` or `[1]`
- **`mobility`** *(optional)* — `'phase'`, `'drift'` or `'auto'` (default; phase for the TR methods, drift for
  `'quadriga'`)
- **`output`** *(optional)* — `'taps'` (default) or `'rays'`
- **`i_freq`** *(optional)* — Frequency index in `rays.center_frequency`, 1-based; default: 1
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx positions in [m]; omitted: the positions of `rays`;
  `[n_snap, 3]` or `[1, 3]`
- **`tx_orientation`**, **`rx_orientation`** *(optional)* — Euler angles (bank, tilt, heading) in [rad]; omitted:
  0; `[n_snap, 3]` or `[1, 3]`
- **`kf_ratio`** *(optional)* — K(snapshot) / K(drop), >= 0; omitted: 1; `[n_snap]` or `[1]`
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1, see [ssf_paths](#ssf_paths); default: `false`
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays; default: `false`

### Outputs:
- **`chan`** — Channel struct, layout as documented in [hdf5_read_channel](channel-functions.md#hdf5_read_channel): `center_frequency` scalar;
  `tx_position`, `rx_position` (the snapshot positions), `tx_orientation`, `rx_orientation` `[3, n_snap]`;
  `coeff_re`, `coeff_im`, `delay` `[n_rx, n_tx, n_tap, n_snap]` (`'taps'`) or `[n_rx, n_tx, n_path, n_snap]`
  (`'rays'`); `'rays'` also `path_gain`, `path_length`, `path_polarization`, `path_angles` and in `'drift'`
  `fbs_pos`, `lbs_pos`
- **`par`** — Struct with the fields above

### See also:
- [ssf_paths](#ssf_paths) (paths of one link at one position, validity limits)
- [ssf_scatterers](#ssf_scatterers) (scatterers, `b_med` and `b_eff`)
- [gbsm_rays_from_paths](#gbsm_rays_from_paths) (ray data from path data, e.g. of a QRT file; taps 0, 3, 6, ... without `tap`)

---
<a id="los_probability"></a>
## los_probability
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

Also in: [C++](../cpp/geometry-based-stochastic-model.md#los_probability) · [Python](../python/geometry-based-stochastic-model.md#los_probability)

- Evaluates the probability that a link is in line-of-sight (LOS) from its 2D distance and the BS
  and UT heights, and, when `rnd_g` and `center_freq` are given, the soft LOS state of TR 38.901 eq.
  (7.6-18); the LOS state itself (a draw against `p_los`) and its spatial correlation are the
  caller's
- `38.901_*`: TR 38.901 v19.5.0 Table 7.4.2-1 (`InH_Mixed`, `InH_Open`: mixed and open office).
  Where the table leaves cases open: all results are limited to [0, 1]; UMa uses `C' = 1` for `hUT >
  23` m; InF-SH and InF-DH give 1 for `hUT >= h_clutter` and the rule of InF-SL and InF-DL for `hBS
  <= hUT`; SMa gives 1 for `hBS <= hUT` and uses `max(h_x - hUT, 0)` in all three terms (the TR only
  in the residential term; equal for `hUT < 15` m)
- `38.811_*`: TR 38.811 Table 6.6.1-1 over the elevation `atan2(hBS, d2d_out)`, linearly
  interpolated; `Suburban` and `Rural` share one table
- `QuaDRiGa_Industrial`: `min(1, exp(-(d2d_out + 20)/80)·exp(min(hBS, 8)/5 - 1))`; `LOS`: 1; `NLOS`:
  0
- The `LOS_state_lambda` of the InF conf files is `d_clutter/2` of the TR defaults; with another
  `d_clutter` pass a matching correlation distance

### Usage:
```matlab
[ p_los, los_soft ] = quadriga_lib.los_probability( model, d2d_out, h_bs, h_ut, ...
    d_clutter, r_clutter, h_clutter, rnd_g, center_freq );
```

### Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`,
  `38.901_InH_Open`, `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`,
  `38.901_InF_HH`, `38.901_SMa`, `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`,
  `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in [m], finite and >= 0: `d2D - d2D-in` for O2I UTs, clamped
  at 0 by the caller; the 2D distance in indoor scenarios; `[n_link]`
- **`h_bs`** — BS height in [m], finite; `[n_link]` or `[1]` for all links
- **`h_ut`** — UT height in [m], finite; `[n_link]` or `[1]` for all links
- **`d_clutter`** — Clutter size in [m] of the InF and SMa models, > 0; NaN or empty = TR default
  (InF-SL and InF-SH 10 m, InF-DL and InF-DH 2 m, SMa 30 m); default: NaN
- **`r_clutter`** — Clutter density of the InF models (default 0.2 for SL and SH, 0.6 for DL and
  DH), vegetation density of SMa (default 0; 0.1 sparse, 0.2 dense); in [0, 1); NaN or empty =
  default; default: NaN
- **`h_clutter`** — Clutter height in [m] of InF-SH (default 2 m) and InF-DH (6 m), >= 0; NaN or
  empty = default. The other models ignore the clutter arguments, which are validated for every
  model; default: NaN
- **`rnd_g`** *(optional)* — Normal(0,1) value per link, G of eq. (7.6-18); `[n_link]`
- **`center_freq`** *(optional)* — Carrier frequencies in [Hz], finite and > 0 (messages name it
  `center_frequency`); `[n_freq]`

### Outputs:
- **`p_los`** — LOS probability in [0, 1]; `[n_link, 1]`
- **`los_soft`** — Soft LOS state in [0, 1]; `> 0.5` exactly when `rnd_g > -F(p_los)` of eq.
  (7.6-18), which has the probability `p_los`; `[n_link, n_freq]`; `[]` if `rnd_g` or `center_freq`
  is omitted or empty

### See also:
- [o2i_loss](#o2i_loss) (indoor distance for `d2d_out`)
- [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand), [sos_randn](sum-of-sinusoids-random-number-generator.md#sos_randn) (spatially consistent LOS-state draw and `rnd_g`)

---
<a id="lsp_generate"></a>
## lsp_generate
Large-scale parameters of a set of links from normalized random values

Also in: [C++](../cpp/geometry-based-stochastic-model.md#lsp_generate) · [Python](../python/geometry-based-stochastic-model.md#lsp_generate)

- Generates the 8 LSPs of TR 38.901 clause 7.5 step 4 per link and frequency from the random values
  of [gbsm_draw](#gbsm_draw) (`'lsf'`), and the absolute TOA offset, the ground permittivity and the EoD offset
- Mean `mu` and standard deviation `sigma` in the log domain (lg for DS and the angular spreads, dB
  for KF, SF and XPR), with the field prefix `P` (`DS`, `KF`, `SF`, `AS_D`, `AS_A`, `ES_D`, `ES_A`,
  `XPR`), `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`, `a = atan(h/d)` and `f' = max(center_freq/1e9,
  LSP_fc_min)` in GHz:
  - `mu = P_mu + P_gamma·lg(P_omega + f') + P_epsilon·lg(d) + P_zeta·lg(h) + P_alpha·lg(a)`
  - `sigma = max(0, max(0, P_sigma + P_delta·lg(P_omega + f')) + P_kappa·lg(d) + P_tau·lg(h) +
    P_beta·lg(a))`
  - SF has the mean 0. DS adds `lg(DS_VS_a·DS_VS + DS_VS_b)` if `DS_VS > 0` (V/S of the InF hall, TR
    38.901 Table 7.5-6 Part-3). ESD adds
    `ES_D_mu_A·d/1000 + ES_D_mu_B·hUT + ES_D_mu_C·abs(hUT - h) + ES_D_mu_D·max(hUT - h, 0)` and is
    limited below by `ES_D_mu_min` (TR 38.901 Tables 7.5-7 to 7.5-12)
  - The multiplying coefficients must be finite; a frequency term with `P_gamma` or `P_delta` not 0
    requires `P_omega + f' > 0`
- `lsp = 10^(mu + sigma·x)` for DS and the angular spreads, `10^(0.1·(mu + sigma·x))` for KF, SF and
  XPR; `x` are the columns 1 to 8 of `rnd`, cross-correlated by a root of the LSP cross-correlation
  matrix of `scenpar`, the same for all frequencies (TR 38.901 clause 7.6.5). `rnd = zeros(n_link,
  11)` gives the median LSPs
- Absolute TOA offset (TR 38.901 clause 7.6.9), active if `absTOA_mu > -30`: `10^(absTOA_mu +
  absTOA_sigma·x)` with column 9 of `rnd`, bounded by `2·absTOA_Lmax/c` if `absTOA_Lmax > 0`
- The InF and InH NLOS files set `absTOA_Lmax`, and the InF files `DS_VS`, for the hall or office of
  their calibration scenario; for another hall set these fields in `scenpar`
- Ground permittivity: QuaDRiGa's fits for dry, medium and wet ground, interpolated linearly over
  column 10 of `rnd` (0 dry, 0.5 medium, 1 wet) with `center_freq` not limited by `LSP_fc_min`; they
  are not the materials of TR 38.901 Table 7.6.8-1. The imaginary part is >= 0: the complex value of
  TR 38.901 eq. (7.6-40) is `gr_eps_re - j·gr_eps_im`
- EoD offset, the negative of the ZOD offset of TR 38.901 Tables 7.5-7 to 7.5-12: `EoD_offset_e +
  EoD_offset_e_gamma·lg f'`, plus `10^((EoD_offset_a + EoD_offset_a_gamma·lg
  f')·lg(max(EoD_offset_b, d)) + EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT·hUT)` if
  `EoD_offset_b > 0`, plus `(atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d))·180/pi` if `hUT <
  EoD_offset_hcl`
- Outputs that are not requested are not computed

### Usage:
```matlab
[ lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma ] = ...
    quadriga_lib.lsp_generate( scenpar, center_freq, tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads );
```

### Inputs:
- **`scenpar`** — Scenario parameters; the LSP cross-correlation matrix must be positive definite
- **`center_freq`** — Carrier frequencies in [Hz], finite and > 0 (messages name it
  `center_frequency`); `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`rnd`** — Random values from [gbsm_draw](#gbsm_draw) (`'lsf'`), finite; column 10 in [0, 1] when 3 or more
  outputs are requested (the error message counts the columns from 0); column 11 is not used;
  `[n_link, 11]`
- **`sf_sigma`** *(optional)* — SF standard deviation in [dB] (e.g. from [path_loss](#path_loss)), finite
  and >= 0; replaces the value from `scenpar`, also in `lsp_sigma`; omitted or empty = the scenario
  value; `[n_link, n_freq]`
- **`cholesky`** — `true`: Cholesky root in the order of TR 38.901 clause 7.5 step 4 (SF, KF, DS,
  ASD, ASA, ESD, ESA; XPR last); `false`: symmetric square root (legacy QuaDRiGa); default: true
- **`clip_spreads`** — `true`: limit ASD and ASA to 104 deg, ESD and ESA to 52 deg (TR 38.901 clause
  7.5 step 4); `lsp_mu` and `lsp_sigma` are not affected; default: true

### Outputs:
- **`lsp`** — Large-scale parameters; slices: 1 DS in [s], 2 KF, 3 SF, 4 ASD, 5 ASA, 6 ESD, 7 ESA in
  [deg], 8 XPR; `[n_link, n_freq, 8]`
- **`abs_toa`** — Absolute TOA offset in [s], 0 if not active; `[n_link, 1]`
- **`gr_eps_re`**, **`gr_eps_im`** — Relative permittivity of the ground, real part and imaginary
  part >= 0; `[n_link, n_freq]`
- **`eod_offset_deg`** — EoD offset in [deg], positive = up; `[n_link, n_freq]`
- **`lsp_mu`** — Mean `mu` in the log domain; `[n_link, n_freq, 8]`
- **`lsp_sigma`** — Standard deviation `sigma` in the log domain, >= 0; `[n_link, n_freq, 8]`

### See also:
- [gbsm_draw](#gbsm_draw) (the random values `rnd`)
- [path_loss](#path_loss) (the SF standard deviation `sf_sigma`)
- [ssf_clusters](#ssf_clusters) (the clusters from the LSPs)

---
<a id="o2i_loss"></a>
## o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

Also in: [C++](../cpp/geometry-based-stochastic-model.md#o2i_loss) · [Python](../python/geometry-based-stochastic-model.md#o2i_loss)

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier
  frequency; the caller adds `loss` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table
  7.4.3-1), `legacy_6GHz` (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz; the O2I conf
  files of UMa and UMi have its `SF_sigma = 7`), `car` and `car_metal` (clause 7.4.3.2, metallized
  windows)
- Model of UT `u`: the first model `j` with `rnd(u, 1)` below the sum of the weights up to `j`, else
  the last model with a weight > 0; a model with weight 0 is never chosen
- Indoor distance: the minimum of columns 2 and 3 of `rnd` times `d2d_in_max` (`low`, `high`,
  `low_A`, clause 7.4.3.1), column 2 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand) and [sos_randn](sum-of-sinusoids-random-number-generator.md#sos_randn) at the correlation distances of TR 38.901 clause
  7.6.3.3: 50 m for column 1, 25 m for columns 2 and 3, 10 m for column 4
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`,
  SMa all three. SMa has 90 % residential and 10 % commercial buildings (Table 7.2-5) with different
  maximum indoor distances: call `o2i_loss` once per building type

### Usage:
```matlab
[ loss, d2d_in, model_index ] = quadriga_lib.o2i_loss( center_freq, rnd, models, weights, d2d_in_max );
```

### Inputs:
- **`center_freq`** — Carrier frequencies in [Hz], finite and > 0 (messages name it
  `center_frequency`); `[n_freq]`
- **`rnd`** — Random values per UT: columns 1 to 3 Uniform in [0, 1] (model choice, two indoor
  distances), column 4 Normal(0,1), finite (deviation `sigma_P`, car loss); `[n_ut, 4]`, `n_ut = 0`
  allowed
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; cell
  array of strings or one string; `n_model` entries
- **`weights`** — Probability of each model, finite and >= 0, sum 1 within 1e-6; `[n_model]`
- **`d2d_in_max`** — Maximum indoor 2D distance in [m], finite and >= 0 (field `O2I_d2D_in_max`; TR:
  25 m for UMa, UMi and SMa commercial buildings, 10 m for RMa and SMa residential buildings);
  default: 25

### Outputs:
- **`loss`** — Penetration loss in [dB]; `[n_ut, n_freq]`
- **`d2d_in`** — Indoor 2D distance in [m], 0 for the car models; `[n_ut, 1]`
- **`model_index`** — Chosen model per UT, 1-based index into `models`; uint64; `[n_ut, 1]`

### See also:
- [los_probability](#los_probability) (uses the outdoor distance `d2D - d2d_in`)
- [path_loss](#path_loss) (basic path loss, legacy outdoor distance from `d2d_in`)

---
<a id="path_loss"></a>
## path_loss
Path loss and shadow-fading standard deviation of a set of links

Also in: [C++](../cpp/geometry-based-stochastic-model.md#path_loss) · [Python](../python/geometry-based-stochastic-model.md#path_loss)

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and
  carrier frequency; the TR 38.901 v19.5.0 conf files of the library reproduce Table 7.4.1-1 with
  them. The O2I penetration loss is not included (see [o2i_loss](#o2i_loss)). The models are evaluated as
  written for any positive frequency, also outside their validity range
- Notation: `f = center_freq/1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D
  distance, at least 0.1 m; `hBS`, `hUT` the z coordinates of Tx and Rx; `lg = log10`; `PL1(x) =
  A1·lg(x) + B + C·lg(f) + D·x`; `sf` the shadow-fading std. Fields of `plpar` in parentheses: the
  required ones, after the semicolon the optional ones (`X = v`: default)
- Models (the field `model` of `plpar`):
  - `constant` (A): `A`
  - `logdist` (A, B, C; SF): `A·lg d3 + B + C·lg f`; `sf = SF`
  - `logdist_simple` (A, B): `A·lg d3 + B`
  - `winner_los` (A1, A2 different, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb =
    max(hBS, 1.5)`, `hm = max(hUT, 1.5)`, `Gj = Bj + Cj·lg f + Dj·lg hb + Ej·lg hm`, breakpoint `bp
    = 10^((G2 - G1)/(A1 - A2))`; `Aj·lg d2 + Gj + Fj·hm` and `sf = sigj` with `j = 1` for `d2 <=
    bp`, else `j = 2`
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as
    `winner_los`; `(Aj + Ahj·lg hb)·lg d2 + Bj + Cj·lg f + Dj·lg hb + Ej·lg hm + Fj·hm` with `j = 1`
    for `f < 1.5`, 2 for `f < 2`, else 3
  - `winner_pathloss` (A, B, C, X; SF): `A·lg d3 + B + C·lg(f/5) + X`; `sf = SF`
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2): `hb =
    max(hBS, 1.1·hE, 0.1)`, `hm = max(hUT, 1.1·hE, 0.1)`, `dBP = E·f·(hb - hE)·(hm - hE)`, `dBP3 =
    sqrt(dBP² + (hb - hm)²)`; `PL1(d3)`, `sf = sig1` for `d2 <= dBP`, else `PL1(dBP3) +
    A2·lg(d3/dBP3)`, `sf = sig2`
  - `tripple_slope` (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0,
    sig1..sig3): the `dual_slope` rule with the breakpoints `dBP1 = min(dBP1, dBP2)` and `dBP2` of
    `E1, hE1` and `E2, hE2`, the slopes `A1`, `A2`, `A3` and `sf = sig1`, `sig2`, `sig3`
  - `nlos` (An, Bn; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0, G2n = 1, LOSmax = 1, Ax,
    Bx, Cx, sig): `hb = max(hBS, 0.1)`, `hm = max(hUT, 0.1)`; `An·lg d3 + Bn + Cn·lg f + Dn·lg hb +
    D1n·lg(hb)/hb + D2n·lg(hb)/hb² + D3n·hb + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm +
    Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; with `LOSmax != 0` (which requires the `dual_slope`
    fields) at least the `dual_slope` loss; with `Ax`, `Bx` or `Cx` at least `Ax·lg d3 + Bx + Cx·lg
    f` (e.g. the InF-SL term of InF-DL); `sf = sig`
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS/d3, 1)`; `+Inf` for
    `sa <= 0`, else `A·lg d3 + B + C·lg f + D·lg(asin(sa))`, plus the zenith attenuation by
    atmospheric gases over `sa` (1 to 100 GHz, constant beyond) if `usePLa != 0`
  - no model (`plpar` empty or without the field `model`): 0
- Where the model sets no `sf` (a missing `SF`, `sig` field or no such field): the SF std of
  [lsp_generate](#lsp_generate) at the geometric distance, so that passing `sf_sigma` to [lsp_generate](#lsp_generate) is
  neutral
- `hE_rand != 0` with `rnd_he` given: the random environment height of TR 38.901 Table 7.4.1-1 note
  1, drawn from `rnd_he`, which makes it spatially consistent at the decorrelation distance
  `SF_lambda` (the TR draws it independently per link); the UT height is then clamped at `hE + 0.1`
  instead of `1.1·hE`. Along a track the loss beyond the breakpoint steps where hE changes; evaluate
  `rnd_he` once per track segment to avoid this
- Distances: the geometric distances of TR 38.901 eq. (7.4-1). With `d2d_in` (legacy QuaDRiGa),
  links with `d2d_in > 0` use the outdoor part of the distances instead, at least 10 m
- A required field that is missing is an error, naming `A1` for "A1 or A" and `E1` for "E1 or E";
  every field that the model reads must have one finite value; the breakpoint factors `E`, `E1`,
  `E2` and `G2n` must be > 0

### Usage:
```matlab
[ pl, sf_sigma ] = quadriga_lib.path_loss( plpar, scenpar, center_freq, tx_pos, rx_pos, d2d_in, rnd_he );
```

### Inputs:
- **`plpar`** — Path-loss parameters; struct as returned by [conf_file_read](#conf_file_read): the field `model`
  (string) and one numeric field per parameter, named without "PL_"; fields not read are ignored
- **`scenpar`** — Scenario parameters; the SF std terms and `LSP_fc_min` are used
- **`center_freq`** — Carrier frequencies in [Hz], finite and > 0 (messages name it
  `center_frequency`); `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m] of the UT of each link (e.g. from
  [o2i_loss](#o2i_loss)), finite and >= 0; given = legacy outdoor distance; 0 = outdoor link; omitted or
  empty = geometric distance; `[n_link]`
- **`rnd_he`** *(optional)* — Uniform(0,1) value per link for the random environment height, column
  11 of [gbsm_draw](#gbsm_draw) (`'lsf'`); omitted or empty = fixed `hE`; `[n_link]`

### Outputs:
- **`pl`** — Path loss in [dB]; `[n_link, n_freq]`
- **`sf_sigma`** — Shadow-fading standard deviation in [dB]; `[n_link, n_freq]`

### See also:
- [lsp_generate](#lsp_generate) (takes `sf_sigma`)
- [o2i_loss](#o2i_loss) (penetration loss and indoor distance)

---
<a id="scenpar_validate"></a>
## scenpar_validate
Validate scenario parameters and complete them with default values

Also in: [C++](../cpp/geometry-based-stochastic-model.md#scenpar_validate) · [Python](../python/geometry-based-stochastic-model.md#scenpar_validate)

- Checks scenario parameters that did not come from a conf file and adds every missing field with
  its default value, so that `scenpar_out` has all 171 fields: the 138 numeric fields of QuaDRiGa
  and 33 fields for TR 38.901 v19.5.0, which have no effect at their default of 0.
  An empty input (`[]` or a struct without fields) gives the complete default set
- A name that is not one of the fields is an error, also with `validate = false`; this includes the
  QuaDRiGa field `GR_epsilon`, which is not supported
- Value checks with `validate = true`; `NaN` is an error for every field:
  - Integer from 1 to 1000: `NumClusters`, `NumSubPaths`
  - Finite and > 0: `r_DS`
  - Finite and >= 0: the 8 decorrelation distances (`DS_lambda` ... `XPR_lambda`), `SC_lambda`,
    `absTOA_lambda`, `PerClusterDS`, `PerClusterDS_min`, `PerClusterAS_D`, `PerClusterAS_A`,
    `PerClusterES_D`, `PerClusterES_A`, `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`, `LSP_fc_min`,
    `EoD_offset_b`, `EoD_offset_hcl`, `PerClusterDS_fc_min`, `PerClusterES_D_scale`, `absTOA_Lmax`,
    `NF_alpha`, `NF_beta`, `SNS_sigma`, `SNS_A`, `SNS_B`, `SNS_R`, `SNS_delta_var`, `PolVar_sigma`,
    `LOS_state_lambda`, `IO_state_lambda`, `O2I_d2D_in_max`, `DS_VS`, `DS_VS_a`, `DS_VS_b`
  - Integer from 0 to 1000: `NumClusters_min`, `NF_k1`
  - In the range from 0 to 1: `SNS_mu`
  - In the range from -1 to 1: the 28 correlations (`asD_ds` to `xpr_esa`)
  - All other numeric fields accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `DS_VS_a` and `DS_VS_b` must not both be 0 if `DS_VS` is > 0
- With `validate = false` no value is checked: `NaN` is accepted

### Usage:
```matlab
[ scenpar_out, lsp_xcorr, lsp_xcorr_chk ] = quadriga_lib.scenpar_validate( scenpar_in, validate );
```

### Inputs:
- **`scenpar_in`** *(optional)* — Scenario parameters; struct with any subset of the 171 fields;
  numeric fields are real scalars of any numeric class or logical; if omitted or empty, the default set is returned
- **`validate`** — `false`: only complete with defaults, no value checks; default: true

### Outputs:
- **`scenpar_out`** — Complete scenario parameters; given values are kept, missing fields have their
  default; struct with 171 double scalar fields in the canonical order: the 138 numeric keys of
  QuaDRiGa, then the 33 new keys
- **`lsp_xcorr`** — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite; `false` also if it contains
  `NaN`; logical

### See also:
- [conf_file_read](#conf_file_read) (read the parameters from a conf file)
- [conf_file_write](#conf_file_write) (applies the same checks before writing)

---
<a id="ssf_clusters"></a>
## ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

Also in: [C++](../cpp/geometry-based-stochastic-model.md#ssf_clusters) · [Python](../python/geometry-based-stochastic-model.md#ssf_clusters)

- Turns the LSPs of [lsp_generate](#lsp_generate) and the cluster random values of [gbsm_draw](#gbsm_draw) into the
  clusters of each link, by one of three methods (a method and property table is in the C++
  documentation of `ssf_clusters`):
  - `'38.901'` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based
    calibration. Clusters 25 dB below the strongest get power 0 (no renormalisation); the K-factor
    in the scaling factors is limited to >= -5 dB. Under spatial consistency (clause 7.6.3.1)
    evaluate the blocks 1 to 8 of `rnd` (columns N + 1 to 9·N) at the drop or segment start and keep
    them along the track
  - `'38.901_MF'`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the
    multi-frequency method of the TR is required
  - `'quadriga'`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model
    for Multiple Frequencies", 2019, with the LOS direction added to the cluster offsets; for
    tracks, several frequencies and device-to-device links
- Slots per link: 1 = LOS path, 2 = ground reflection if `GR_enabled > 0` (`G = 1`), then the `N =
  NumClusters - 1` NLOS clusters; `n_slot = NumClusters + G`. NLOS cluster `n = 1 ... N` is in slot
  `1 + G + n` and belongs to the columns `b·N + n` of `rnd` (no sorting). Clusters `n >
  n_cluster(l)` (TR clause 7.6.15) get delay 0, power 0 and the LOS angles. Identify the LOS path by
  its slot, never by delay or path length: NLOS clusters of the TR methods can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers
  are 0 whatever `lsp` holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30`
  run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; angles in [rad], GCS, elevation convention; slot
  1 holds the geometric LOS direction. Powers are the coefficient powers of TR eq. (7.5-30), LOS
  `K/(K+1)`, NLOS clusters `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `'38.901'`, `'38.901_MF'` add `abs_toa` to
  the NLOS delays; `'quadriga'` places the first NLOS cluster at `abs_toa` (where `absTOA_mu <=
  -30`: only where `abs_toa > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0: slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR
  methods, on top of the other slots) or `K/(K+1)·R²` (`'quadriga'`), with `R²` the mean of
  `abs(R_par)²` and `abs(R_perp)²` of TR eqs. (7.6-38), (7.6-39); the coefficient is
  `sqrt(power)·diag(R_par, -R_perp)/R`, without polarisation `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios, plus the ground reflection if
  enabled; `rnd` must be empty, `rnd_freq` and `lsp_anchor` are not used
- Swap reciprocity: with mirrored random values (dual mobility in [gbsm_draw](#gbsm_draw)) and mirrored LSPs,
  swapping `tx_pos` and `rx_pos` exchanges the departure and arrival angles (`'38.901'`:
  `eod_offset_deg` = 0 and no O2I link; `'quadriga'`: `eod_offset_deg` = 0)
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario; only their sizes are
  checked; at least one link is required

### Usage:
```matlab
[ delay, power, aod, eod, aoa, eoa, n_active ] = quadriga_lib.ssf_clusters( scenpar, tx_pos, rx_pos, ...
    lsp, rnd, method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps_re, gr_eps_im, o2i, n_cluster );
```

### Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`, `KF_mu`, `absTOA_mu`,
  `r_DS`, `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in [m]; z >= 0 with ground reflection; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[n_link, 3]`
- **`lsp`** — LSPs from [lsp_generate](#lsp_generate); DS, ASD, ASA, ESD, ESA (slices 1, 4 to 7) > 0, KF (slice
  2) >= 0; `'38.901'`: one frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [gbsm_draw](#gbsm_draw) (`'cluster'`, `set_id = 0`), columns 1 to N in [0,
  1]; `[n_link, 9·N]`
- **`method`** *(optional)* — `'38.901'` (default), `'38.901_MF'` or `'quadriga'`
- **`rnd_freq`** *(optional)* — Per-frequency cluster sets of [gbsm_draw](#gbsm_draw), slice `k` from `set_id
  = k`; required by `'38.901'` and `'38.901_MF'`; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz); required by
  `'38.901_MF'`; `[n_link, 1, 8]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg] from [lsp_generate](#lsp_generate); used by `'38.901'`
  and `'quadriga'` in NLOS scenarios; omitted or empty = 0; `[n_link, n_freq]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s] from [lsp_generate](#lsp_generate), >= 0; required by
  `'quadriga'` in NLOS scenarios with `absTOA_mu > -30`; omitted or empty = no offset; `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im`,
  real part >= 1, imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** *(optional)* — O2I flag per link (mean ZOA 90 deg), logical or numeric; used by
  `'38.901'` in NLOS scenarios; omitted or empty = none; `[n_link]`
- **`n_cluster`** *(optional)* — Active NLOS clusters per link (a count) in `[min(max(1,
  NumClusters_min - 1), N), N]`; omitted or empty = N; `[n_link]`

### Outputs:
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_freq]` for `'38.901_MF'`, else `[n_link, n_slot,
  1]`
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in [rad], in [-pi, pi); sizes as `delay`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in [rad], in [-pi/2, pi/2]; sizes as
  `delay`
- **`n_active`** — Number of NLOS clusters with power > 0; uint64; `[n_link, 1]`

### See also:
- [gbsm_draw](#gbsm_draw) (the random values `rnd` and `rnd_freq`)
- [lsp_generate](#lsp_generate) (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [calc_delay_spread](channel-statistics.md#calc_delay_spread), [calc_angular_spread](channel-statistics.md#calc_angular_spread) (spreads of the output)

---
<a id="ssf_paths"></a>
## ssf_paths
Antenna-free paths of one link at one position

Also in: [C++](../cpp/geometry-based-stochastic-model.md#ssf_paths) · [Python](../python/geometry-based-stochastic-model.md#ssf_paths)

- `'phase'`: use the outputs with [get_channels_planar](channel-generation-functions.md#get_channels_planar) and the arrays at the positions of `rays`; the
  displacement phase of TR 38.901 eq. (7.5-22) is in `M`. With the arrays at the new positions the LOS path is no
  longer identified and the displacement phase counts twice, without an error
- `'drift'`: use the outputs at the new positions with [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) or [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq);
  needs the scatterers of [ssf_scatterers](#ssf_scatterers)
- Kept rays: LOS and GR rays always (also at zero power), NLOS rays with power > 0 at some frequency; paths in ray
  order, `ray_ind` maps them to the rays (e.g. `rays.tap(ray_ind)`, `rays.fallback(ray_ind)`)

| Quantity | `'phase'` | `'drift'` |
| --- | --- | --- |
| Path length | Tx-Rx distance of `rays` + c·delay | Polyline Tx, FBS, LBS, Rx; LOS direct line; GR via the specular point |
| Angles | Ray data | Geometry: towards FBS and LBS, LOS direct line, GR towards the image points |
| `M` | Ray `M` times the displacement phase | Ray `M`; GR recomputed when an end moved |
| `fbs_pos`, `lbs_pos` | `[3, 0]` | Scatterers of the rays; LOS midpoint of the direct line; GR specular point |

- `'drift'` with an end moved (any coordinate differs from `rays`): GR power by the power law of `rays.method`
  (`(d3D / d_GR)²` for the TR methods, 1 for `'quadriga'`), GR `M` the scalar multiple of `diag(R_par, -R_perp)` (TR
  38.901 eqs. (7.6-38), (7.6-39)) that best fits the stored matrix, which replaces a matrix of another form (e.g. a
  conjugated one). Fallback rays have their longer polyline, also at the positions of `rays`
- Path gain per frequency: the coefficient power; with `kf_ratio` LOS and GR are rescaled against NLOS so that the
  K-factor of the coefficient powers changes by `kf_ratio`, LOS + NLOS keep their sum; times
  `10^(link_gain_dB / 10)`; with `oxygen` times the absorption of TR 38.901 clause 7.6.1 (Table 7.6.1-1 at each
  frequency) over each path's own length. Deviations: τ_Δ is not added (at 60 GHz: 0.017 dB median and up to 1.28 dB
  at p99 more loss on NLOS rays of the TR methods without absolute TOA; sub-cluster rays up to 0.13 dB), eq. (7.6-2)
  is not applied across the band
- Validity per moving end: `'phase'` up to about `sqrt(λ · b_eff / 8)` (`b_eff` of [ssf_scatterers](#ssf_scatterers); p50 over the
  links for an Rx move: TR NLOS 0.27 to 0.36 m at 3.5 GHz and 0.12 to 0.13 m at 28 GHz, `'quadriga'` NLOS 0.24 to
  0.77 m and 0.24 to 0.31 m, LOS 0.91 to 1.33 m and 0.33 to 0.40 m); `'drift'` up to about `0.2 · b_med` (p50 1.5 to
  5.6 m LOS, 7.8 to 12.4 m NLOS, about 20 % less when both ends move; new segment from `0.5 · b_med`); in `'drift'`
  each path length changes by at most the displacement
- Errors in `'drift'`: no scatterers; a GR ray at moved ends without permittivity in `rays` or with an end below z = 0

### Usage:
```matlab
[ path_gain, path_length, M, aod, eod, aoa, eoa, fbs_pos, lbs_pos, path_type, ray_ind ] = ...
    quadriga_lib.ssf_paths( rays, link_gain_dB, mobility, tx_pos, rx_pos, kf_ratio, oxygen );
```

### Inputs:
- **`rays`** — Ray struct of the link, see [ssf_rays](#ssf_rays); `[1, 1]` struct
- **`link_gain_dB`** — Link gain `-PL - O2I + SF` at the position in [dB]; 0 for data from paths; `[n_freq]`
- **`mobility`** *(optional)* — `'phase'`, `'drift'` or `'auto'` (default; phase for the TR methods, drift for
  `'quadriga'`)
- **`tx_pos`**, **`rx_pos`** *(optional)* — Tx and Rx position in [m]; omitted: the positions of `rays`; `[3]`
- **`kf_ratio`** *(optional)* — K(position) / K(drop), >= 0; omitted: 1; `[n_freq]`
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1; default: `false`

### Outputs:
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`path_length`** — Path lengths in [m]; `[n_path, 1]`
- **`M`** — Polarization transfer matrices, rows as in [get_channels_planar](channel-generation-functions.md#get_channels_planar); `[8, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Departure and arrival angles in [rad], GCS; `[n_path, 1]`
- **`fbs_pos`**, **`lbs_pos`** — Scatterer positions in [m]; `[3, n_path]`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS (a code, not shifted); uint64; `[n_path, 1]`
- **`ray_ind`** — Index of the ray in `rays`, 1-based; uint64; `[n_path, 1]`

### See also:
- [ssf_scatterers](#ssf_scatterers) (scatterers, validity scales `b_eff` and `b_med`)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel coefficients of one link over a segment)
- [gbsm_rays_from_paths](#gbsm_rays_from_paths) (ray data from path data, e.g. of a QRT file)

---
<a id="ssf_rays"></a>
## ssf_rays
Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)

Also in: [C++](../cpp/geometry-based-stochastic-model.md#ssf_rays) · [Python](../python/geometry-based-stochastic-model.md#ssf_rays)

- Splits the clusters of [ssf_clusters](#ssf_clusters) into rays and returns one ray struct per link (fields below): delays,
  coefficient powers, angles, polarization transfer matrices with initial phases, path types and tap indices;
  `fbs_pos`, `lbs_pos` and `fallback` stay empty (see [ssf_scatterers](#ssf_scatterers))
- A pure function of the clusters, the LSPs and the random values, without a seed: a link does not depend on the
  other links of the call
- Ray layout per link, fixed per scenario: ray 1 LOS, 2 GR (if `GR_enabled`), then ray `m` of NLOS cluster `n` (both
  from 1) at `1 + G + (n - 1)·M + m` (`G` = 1 with GR, `M` = `NumSubPaths`); clusters at power 0 keep their rays
- LOS and GR rays: delay, angles and power of their slot; Jones matrices diag(1, -1) and diag(R_par, -R_perp)/R of
  the ground reflection (TR eqs. (7.6-38) to (7.6-40))
- NLOS rays: power P_n/M; offsets of Table 7.5-3 scaled by the cluster spreads; coupling (step 8) by the rank blocks
  of `rnd_ray`; `M` other than 20: QuaDRiGa-NG's offsets below 20, Laplacian quantiles above 20, no sub-clusters

| Method | `'38.901'`, `'38.901_MF'` | `'quadriga'` |
| --- | --- | --- |
| Frequencies per call | `i_freq` only; slice `i_freq` of `rnd_ray` | all; one geometry, per frequency powers, XPR and Jones matrices; the shared "ray" set |
| Ray delays | cluster delay; sub-clusters + 0, 1.28, 2.56 c_DS (Table 7.5-6) | cluster delay + 2·c_DS·(u - mean of u in the cluster), u of block 14; negative delays set to 0 |
| Taps | sub-cluster of the two strongest clusters (`M` = 20), else cluster | cluster |
| c_DS, ZSD | at frequency `i_freq` | mean over the frequencies |

| Option | Default (TR) | Alternative (QuaDRiGa-NG) |
| --- | --- | --- |
| `angle_model` | `'additive'`: offsets added to azimuth and zenith, zenith folded (step 7) | `'rotation'`: offset pattern rotated to the cluster direction; equal at elevation 0 |
| `pol_model` | `'38.901'`: XPR and four initial phases per ray (steps 9, 10) | `'quadriga'`: rotation polarisation model of QuaDRiGa, one phase, XPR per ray and per cluster |
| `pol_variability` | `false` | `true`: clause 7.6.16 on the NLOS rays with std `PolVar_sigma`; `pol_model = '38.901'` only |

- Sub-clusters (step 11): the two clusters of largest power at frequency `i_freq`; every tap is a contiguous block
- `'quadriga'` sets 0.2 to 1.4 % of the active NLOS rays to delay 0 (at most 2.9 % of the NLOS power)
- `pol_model = 'quadriga'`: the matrix refers to the owner end, the end with the larger z, then x, then y (ends
  within 1 mm count as equal; the Tx if all tie); the Rx as owner transposes it
- Swap reciprocity: with mirrored inputs (dual mobility in [gbsm_draw](#gbsm_draw), mirrored LSPs and clusters,
  `PerClusterAS_D` = `PerClusterAS_A`, `PerClusterES_D` = `PerClusterES_A`, `PerClusterES_D_scale` = 0), swapping
  `tx_pos` and `rx_pos` exchanges departure and arrival per ray and transposes `M` (not for `pol_model = 'quadriga'`
  with ends within 1 mm of each other, where the Tx stays the owner)
- Messages name the argument `center_freq` as `'center_frequency'`

### Ray struct:
One link; all fields are present on output (empty where empty). As input, missing optional fields are empty, other
missing or unknown fields are an error; integer fields accept any real class. Rays from several sources in one
struct: at most one LOS and one GR ray, with tap index 0 and 3; the other rays take any other tap indices; one
`method`; [ssf_scatterers](#ssf_scatterers) overwrites given scatterers, so solve rays without scatterers separately and
concatenate the fields.

| Field | Size | Type | Meaning | Required |
| --- | --- | --- | --- | --- |
| `method` | string | char | `'38.901'`, `'38.901_MF'` or `'quadriga'`; GR power law and default mobility | yes |
| `tx_pos`, `rx_pos` | `[3, 1]` | double | Tx and Rx position the ray data refer to in [m] | yes |
| `center_frequency` | `[n_freq, 1]` | double | Carrier frequencies in [Hz] | yes |
| `delay` | `[n_ray, 1]` | double | Delays in [s] relative to the LOS delay d3D / c, d3D the Tx-Rx distance | yes |
| `power` | `[n_ray, n_freq]` | double | Coefficient powers (linear) without link gain; data from paths: path gain | yes |
| `aod`, `eod`, `aoa`, `eoa` | `[n_ray, 1]` | double | Departure and arrival azimuth and elevation in [rad], GCS | yes |
| `M` | `[8, n_ray, n_freq]` | double | Polarization transfer matrices incl. initial phases, rows as in [get_channels_planar](channel-generation-functions.md#get_channels_planar) | yes |
| `path_type` | `[n_ray, 1]` | uint64 | 0 = LOS, 1 = GR, 2 = NLOS (a code, not shifted) | yes |
| `tap` | `[n_ray, 1]` | uint64 | Tap index (a code, not shifted); rays with equal index form one tap | yes |
| `gr_eps_re`, `gr_eps_im` | `[n_freq, 1]` or empty | double | Ground permittivity `gr_eps_re - j·gr_eps_im`; needed where a GR ray is recomputed | no |
| `fbs_pos`, `lbs_pos` | `[3, n_ray]` or empty | double | First- and last-bounce scatterer positions in [m] | no |
| `fallback` | `[n_ray, 1]` or empty | uint64 | 1 = ray lengthened by the solver fallback | no |

### Usage:
```matlab
rays = quadriga_lib.ssf_rays( scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, delay, power, ...
    aod, eod, aoa, eoa, rnd_ray, method, i_freq, angle_model, pol_model, gr_eps_re, gr_eps_im, ...
    pol_variability, random_phase );
```

### Inputs:
- **`scenpar`** — Scenario parameters
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`center_freq`** — Carrier frequencies in [Hz]; `[n_freq]`
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [lsp_generate](#lsp_generate); `[n_link,
  n_freq, 8]`
- **`delay`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays and angles from [ssf_clusters](#ssf_clusters);
  `[n_link, n_slot, n_freq]` for `'38.901_MF'`, else `[n_link, n_slot, 1]`
- **`power`** — Cluster powers from [ssf_clusters](#ssf_clusters); `[n_link, n_slot, n_freq]`
- **`rnd_ray`** — "ray" sets from [gbsm_draw](#gbsm_draw), `n_gen` = 14·N·M + N; TR methods: slice `k` = set `k`, `[n_link,
  n_gen, n_freq]`; `'quadriga'`: the shared set (`set_id = 0`), `[n_link, n_gen]`
- **`method`** *(optional)* — Method of [ssf_clusters](#ssf_clusters): `'38.901'` (default), `'38.901_MF'` or `'quadriga'`
- **`i_freq`** *(optional)* — TR methods: index into `center_freq`, 1-based (`center_frequency` of the output then
  has one entry, so `i_freq` = 1 in [gbsm_single_link_channel](#gbsm_single_link_channel)); `'quadriga'`: 1; default: 1
- **`angle_model`** *(optional)* — `'additive'` (default) or `'rotation'`
- **`pol_model`** *(optional)* — `'38.901'` (default) or `'quadriga'`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity from [lsp_generate](#lsp_generate); required if
  `GR_enabled`, else ignored; `[n_link, n_freq]`
- **`pol_variability`** *(optional)* — TR 7.6.16 polarisation power variability; default: `false`
- **`random_phase`** *(optional)* — `false`: all initial phases 0; default: `true`

### Output:
- **`rays`** — Ray structs, one per link; `[1, n_link]` struct array

### See also:
- [ssf_scatterers](#ssf_scatterers) (scatterer positions and validity scales of one link)
- [ssf_paths](#ssf_paths) (paths of one link)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (channel coefficients of one link)

---
<a id="ssf_scatterers"></a>
## ssf_scatterers
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

Also in: [C++](../cpp/geometry-based-stochastic-model.md#.solve_scatterers) · [Python](../python/geometry-based-stochastic-model.md#ssf_scatterers)

- NLOS rays with power > 0 at some frequency: FBS on the departure ray and LBS on the arrival ray at equal
  distances from their end, so that the polyline Tx, FBS, LBS, Rx has the length d3D + c·delay (d3D the Tx-Rx
  distance) and the angles are exact
- Where no such pair exists with both distances at least `d_min`, both scatterers are placed at `d_min` along the
  ray's own directions and `fallback` is 1: the polyline is longer than the ray, `delay` keeps its value. With
  `d_min` = 2 m (UMi, UMa, RMa, InF and indoor scenarios) this applies to 5.5 to 13 % of the rays of the TR methods
  without absolute TOA, to 1.0 to 9.8 % in `'quadriga'` and to at most 2.3 % with absolute TOA; it adds a median
  delay of up to 5.5 ns (p99 up to 21.7 ns)
- LOS ray and NLOS rays with zero power at all frequencies: the midpoint of the direct line; GR ray: the specular
  point on z = 0, which requires both ends at z >= 0
- `rays_out` is a copy of `rays_in` (all fields, about 49 KB per link at 381 rays and one frequency; unchanged fields
  keep their class) with new `fbs_pos`, `lbs_pos` and `fallback`; `rays_in` is not changed, also when it already
  holds scatterers
- `b_med` and `b_eff` are the validity scales of [ssf_paths](#ssf_paths) for planning segments: `b_med` the lower
  power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS rays with power > 0 (`drift`, NaN
  without NLOS power); $b_\text{eff} = \left(\sum P / \sum (P / b^2)\right)^{1/2}$ over the rays with power
  $P > 0$, with $b$ the end-to-scatterer distance (NLOS), d3D (LOS) or the GR path length (GR) (`phase`)
- One link per call; loop over the elements of a struct array: `rays(l) = quadriga_lib.ssf_scatterers( rays(l) )`

### Usage:
```matlab
[ rays_out, b_med, b_eff ] = quadriga_lib.ssf_scatterers( rays_in, d_min );
```

### Inputs:
- **`rays_in`** — Ray struct of one link, see [ssf_rays](#ssf_rays); `[1, 1]` struct
- **`d_min`** *(optional)* — Minimum distance of the scatterers to both ends in [m], > 0; default: 2

### Outputs:
- **`rays_out`** — Ray struct with `fbs_pos`, `lbs_pos` `[3, n_ray]` and `fallback` `[n_ray, 1]` (uint64)
- **`b_med`**, **`b_eff`** — Validity scales in [m], rows Tx and Rx, column `k` for frequency `k`; `[2, n_freq]`

### See also:
- [ssf_rays](#ssf_rays) (ray data of a set of links)
- [ssf_paths](#ssf_paths) (mobility `drift` needs the scatterers)
- [gbsm_single_link_channel](#gbsm_single_link_channel) (reports `b_med`, `b_eff` and the displacement ratios)
