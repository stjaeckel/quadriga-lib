# Plan: port the scenario configuration file parser to quadriga-lib

Status: plan with all open points answered (sections 2 and 3), written 2026-10-12 in a QuaDRiGa-NG
session. Nothing is implemented yet.
No source file was changed while writing it.

This document is the brief for a session that works in the quadriga-lib repository and ports the
parsing and validation of QuaDRiGa scenario files (`*.conf`) to the C++ core. A second deliverable is
a set of new conf files that state the 3GPP TR 38.901 v19.5.0 parameters.

Read first, in this order:

1. This document.
2. `md/SPEC_CPP_API.md` in quadriga-lib. It defines the workflow (test first, review points A, B, C),
   file layout, style, tests and documentation blocks. This plan does not repeat it.
3. The MATLAB reference in `/sjc/QuaDRiGa-NG/quadriga_src/@qd_builder/`:
   `private/set_scenario_table.m`, `private/check_scenario_parameter_table.m`,
   `supported_scenarios.m`, and the property accessors `lsp_vals`, `lsp_xcorr`, `lsp_xcorr_chk` in
   `qd_builder.m`.
4. The 91 files in `/sjc/QuaDRiGa-NG/quadriga_src/config/`.
5. `/sjc/references/3gpp_38901/38901-v19.5.0.md`, clauses 7.4.1, 7.5 (Tables 7.5-6 to 7.5-12),
   7.6.3.1, 7.6.9, 7.6.13 to 7.6.16.

QuaDRiGa-NG and the 3GPP reference are read-only for the library session.


## 1. Goal and scope

In scope (C++ only):

- Three functions declared in `include/quadriga_tools.hpp`: read and validate one conf file, validate
  a parameter set that did not come from a file, and list the scenarios found in a set of directories.
- One key table as the single source of truth for key names, defaults and validation classes. It
  holds today's 140 keys and 27 new keys for 38.901 v19.5.0.
- Catch2 tests for the three functions.
- 23 new conf files `3GPP_38.901_v19_*.conf` with the v19.5.0 values.

Out of scope:

- MEX and Python wrappers. They are separate tasks, but the contract in section 6 is designed for them.
- Any change to QuaDRiGa-NG. Replacing the MATLAB parser is a later session there (section 9).
- The model code that consumes the new keys (EoD offset, near field, SNS and so on). The parser reads
  and validates them; nothing uses them yet.
- Path-loss evaluation (`get_pl.m`), LOS probability and O2I loss. Section 5.4 lists what v19.5.0
  changes there, for a later port.


## 2. Decisions

Answers given by Stephan on 2026-10-12. They are binding unless he changes them.

| # | Question | Answer |
|---|---|---|
| D1 | v19.5.0 values versus the 20 existing `3GPP_38.901_*` files | **New files, keep old.** Existing files stay byte-identical (they state v16.1.0). The v19.5.0 set is added under a new prefix. |
| D2 | How far does the key set go? | **Add all v19 keys now.** All candidate keys are parsed and validated in this port, with defaults that reproduce today's behaviour. |
| D3 | C++ result type | **`std::unordered_map`**, as in the material-property API (`mtl_prop`). |
| D4 | Scenario-name lookup | **File reader plus directory lister.** The caller supplies the directories. Conf files are not embedded in the library. |

Constraints from the task description:

- The functions belong to "tools" (`quadriga_tools.hpp`).
- Validation is folded into the reader.
- The result must serve as a drop-in for QuaDRiGa-NG: every one of the 91 shipped files must give the
  same `scenpar` and `plpar` values as the MATLAB parser gives today.


## 3. Decisions on the open points

Asked one by one and answered by Stephan on 2026-10-12. They are binding like section 2. Where an
answer differs from the recommendation of the first draft, the rest of this plan has been updated.

| # | Open point | Answer |
|---|---|---|
| O1 | Value type: fixed `double` or template `dtype`? | **Fixed `double`**, no template. |
| O2 | A third public function `scenpar_validate`, needed because `qd_builder.scenpar = struct` validates a parameter set without a file. | **Keep it.** Three public functions. |
| O3 | Documentation section. | **"Channel generation functions".** (The draft recommended "Miscellaneous / Tools".) The declarations stay in `quadriga_tools.hpp`, as required by the task. |
| O4 | New internal header `src/qd_scenpar.hpp` for the key table? | **No. One source file.** All three public functions and the key table (file-local, `static`) live in one `.cpp`. This is an approved exception to the one-function-per-file rule. File name chosen by the plan: `src/qd_scenpar.cpp`. |
| O5 | Where are the v19 conf files authored, and under which licence? | **`quadriga-lib/data/config/`, Apache-2.0**, written fresh from the TR. Copied to QuaDRiGa-NG in the integration session. The 91 existing files are not copied into quadriga-lib. |
| O6 | Strictness for malformed input (section 4.3). | **Clean grammar, errors.** Input that the MATLAB parser turns into `NaN` or a stray string is an error with file name and line number. |
| O7 | Do values in a conf file switch the optional v19 components on? | **The consumer decides.** The v19 files carry the table values. Whether near field, SNS, cluster-count variability and polarization variability are active is a simulation setting of the consumer. Absolute TOA keeps today's rule (`absTOA_mu > -30`), so it is active in the v19 files that state it. |
| O8 | EoD offset (TR: ZOD offset) for RMa: literal 35 m and 1.5 m, or actual heights? | **Actual heights.** One geometric form for RMa and SMa with the real `hBS` and `hUT` and `EoD_offset_hcl = 3.5`. Identical to the TR at the default heights. |
| O9 | Does `LSP_fc_min` also clamp the cluster delay spread formula? | **Separate key.** New key `PerClusterDS_fc_min`, default 0. (The draft recommended no clamp and no key.) The TR states no clamp for the cluster DS, so the v19 files leave it at 0. |
| O10 | Which v19 files get absolute TOA values? | **NLOS and NLOS_O2I.** Not LOS and LOS_O2I. |
| O11 | `PerClusterES_D` versus the TR rule `c_ZSD = 3/8 · 10^mu_lgZSD` (7.6-7; a spread, so `c_ESD = c_ZSD`). | **New key, dynamic.** New key `PerClusterES_D_scale`, default 0. (The draft recommended a fixed number.) See section 7.2. |
| O12 | Values QuaDRiGa needs that the TR does not give (`KF_lambda` in NLOS, `XPR_lambda`). | **Carry over from the v16 files.** SMa uses the UMa values. The file comments mark them as not from the TR. |

The key names `PerClusterDS_fc_min` and `PerClusterES_D_scale`, like all names in section 7.2, are
proposals. Review point A is the place to rename them.


## Record of later changes

Changes after this plan was written. The sections below are not rewritten; where they differ, this record
holds.

- `GR_epsilon` dropped (3690107): the key table has 138 + 27 = 165 numeric keys, the MEX struct and the Python
  dict 166 fields with `SubpathMethod`. In a file the key is ignored and reported in `ignored_keys`; in
  `scenpar_validate` and `conf_file_write` it is an unknown name and an error. Sections 4.2, 6, 7.1, WI-3 and 9
  still count it (`n_key = 166`, "grows from 140 to 167 fields").
- `conf_file_write` added (e525c7b), a fourth public function, with the decisions of its review point A: a key is
  written only if its value differs from the default (bit for bit), a section without a written key is left out,
  the sections "EoD offset" and "Optional modelling components" are new, numbers read back bit for bit.
- MEX wrappers of `conf_file_read`, `scenpar_validate` and `conf_file_list` (3690107) and of `conf_file_write`
  (ad7a9a4); Python wrappers of all four in `quadriga_lib.tools` (145f163).
- The parity test of WI-5 was removed (e525c7b), see the status note of WI-5.
- Decisions of 2026-10-14 on the review `md/REVIEW_CONF_FILE.md`, implemented by `md/PLAN_CONF_FILE_FIXES.md`:
  - B1: `conf_file_write` writes a temporary file in the same directory and renames it over the target; a
    failed call never changes an existing file and leaves no file behind; an existing file keeps its
    permissions. An existing target that is not a regular file (directory, device, FIFO) is an error.
  - B2: a line break in the file name of `conf_file_write` is an error.
  - B3: Python flags (`validate`, `parse_shortnames`, `write_defaults`): `None` selects the default, `bool` and
    `numpy.bool_` are accepted, anything else is a `TypeError`.
  - Q1: a UTF-8 byte order mark at the start of a file is skipped (reader and lister).
  - Q2: a number beyond the range of a double is the error "is out of range"; below the smallest subnormal it
    reads as the signed 0 that `strtod` gives in the C locale. The conversion stays a stream with the classic
    locale (the fallback of the plan): `std::from_chars` for `double` needs `GLIBCXX_3.4.29`, which the
    libstdc++ of MATLAB R2024a does not have, so the MEX files could not be loaded (decided 2026-10-14).
  - Q3: a `ShortName` value stated twice in one file is an error of the reader (both `validate` modes);
    `conf_file_write` rejects duplicate short names.
  - Q4: Python `fn` and `dirs` accept `str`, `bytes` and `os.PathLike`.
  - Q5: the shared MEX helpers raise `quadriga_lib:CPPerror` (was `MATLAB:unexpectedCPPexception`).
  - Q6: a MEX `plpar` field must be a vector.
  - Q7: kept; the scenario name of the lister is the file name without `.conf`, whatever characters it has.
  - Q8: kept; a directory that cannot be listed is skipped like a missing one.
  - Q10: the ESD overview comment of a written file shows `ES_D_mu_A` to `ES_D_mu_D` and the lower limit
    `ES_D_mu_min`; with `LSP_fc_min > 0` the frequency is written as `max( fGHz, fc_min )`.
  - Q11: `NumClusters` and `NumSubPaths` are integers from 1 to 1000, `NumClusters_min` and `NF_k1` from 0 to 1000.
  - Q12: test cases do not share temporary file names.


## 4. Background: the format and the MATLAB parser

### 4.1 File format

A conf file is a text file with one `Key = value` assignment per line and `%` comments.

```
DS_mu =               -6.28         % delay spread [log10(s)] @ 1 GHz
SubpathMethod =        legacy       % subpath mapping method (legacy, Laplacian or mmMAGIC)
PL_model =             nlos
PL_E3n =              -0.6          % RX height-dep. of the NLOS-PL in [dB/m]
ShortName =            Un           % You can use this short name to load the file
```

There are three kinds of keys:

- **Scenario parameters** (`scenpar`): 140 fixed keys with defaults. 139 are numeric scalars, one
  (`SubpathMethod`) is a string. A file states only the keys that differ from the default.
- **Path-loss parameters** (`plpar`): every key that starts with `PL_`. The set is open. `PL_model`
  is a string, all others are numeric. The prefix is removed, so `PL_A1` becomes field `A1`.
- **`ShortName`**: an alias for the scenario name. It is read by `supported_scenarios.m`, not by the
  parameter parser.

Facts from a survey of all 91 files:

- 164 distinct keys occur: 115 of the 140 scenario keys, 48 `PL_` keys and `ShortName`.
- `PL_model` takes 10 values: `constant`, `dual_slope`, `logdist`, `logdist_simple`, `nlos`,
  `satellite`, `tripple_slope`, `winner_los`, `winner_nlos`, `winner_pathloss`. `LOSonly.conf` has no
  `PL_` key at all.
- `SubpathMethod` is `legacy` (79 files) or `mmMAGIC` (6 files). The code also accepts `Laplacian`.
- All numbers are plain decimals except 16 values in the MIMOSA files, which use an exponent
  (`6.2e-07`). No file uses `Inf`, `NaN`, an upper-case exponent, a plus sign or more than one value.
- 5 files have CRLF line ends. 8 files (MIMOSA) contain non-ASCII bytes in comments.
- One file states a key twice (`SC_lambda` in `5G-ALLSTAR_DenseUrban_LOS.conf`, both 50).
- `PL_Exn` occurs in 4 files and is read by nothing in `get_pl.m`.
- Only 2 files have a `ShortName` (`BERLIN_UMa_LOS` = `Ul`, `BERLIN_UMa_NLOS` = `Un`).

### 4.2 What the MATLAB parser does, line by line

`set_scenario_table.m` reads the file with `fgetl` and applies this to each line:

1. Find the first `=`. No `=` means the line is skipped.
2. If a `%` stands before that `=`, the line is a comment and is skipped.
3. The key is the first `[A-Za-z0-9_]+` token before the `=`. No token means the line is skipped.
4. If the key contains `PL_` (anywhere, not only at the start), it is a path-loss key. The field name
   is the key without its first three characters.
   - `model`: the value is the first `[A-Za-z0-9_]+` token after the `=`. A missing value is an error.
   - Other fields: the text between the `=` and the first `%` is scanned for all `[0-9e.-]+` tokens.
     Each goes through `str2double`. The result is a row vector, so several values are possible.
5. Otherwise the key must match one of the 140 names exactly (case-sensitive). Unknown keys are
   skipped without a message. The value is the text after the `=` up to the first `%`.
   - If it contains a `[0-9.-]+` token, the first `[0-9Eeji.+-]+` token goes through `str2double`.
   - Otherwise the first `[A-Za-z]+` token is stored as a string. This applies to every key, not only
     to `SubpathMethod`.
   - Otherwise it is an error ("Could not understand value").
6. A key that occurs twice keeps its last value.
7. `plpar` is `[]` when the file has no `PL_` key. Otherwise it is a struct with fields in file order.

`check_scenario_parameter_table.m` then validates, unless the caller passed `check = false`:

| Class | Rule | Keys |
|---|---|---|
| count | exactly 140 fields (names are not checked) | all |
| int | integer and > 0 | `NumClusters`, `NumSubPaths` |
| positive | real and > 0 | `r_DS` |
| non-negative | real and >= 0 | `PerClusterAS_D`, `PerClusterAS_A`, `PerClusterES_D`, `PerClusterES_A`, `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`, `SC_lambda`, the 8 `*_lambda`, `PerClusterDS`, `PerClusterDS_min`, `absTOA_lambda` |
| real | real scalar | `*_mu`, `*_sigma`, `*_gamma`, `*_delta` of the 8 LSPs (where they exist), `ES_D_mu_A`, `ES_D_mu_min`, `PerClusterDS_gamma`, `absTOA_mu`, `absTOA_sigma` |
| correlation | real and in [-1, 1] | the 21 correlations without XPR (`asD_ds` to `esA_kf`) |
| not checked | none | `*_omega`, `*_epsilon`, `*_zeta`, `*_alpha`, `*_kappa`, `*_tau`, `*_beta` of all LSPs, `GR_epsilon`, `SubpathMethod`, the 7 `xpr_*` correlations |

Two derived quantities belong to the parameter set and are computed in `qd_builder.m`:

- `lsp_xcorr`: the symmetric 8 x 8 matrix in the order DS, KF, SF, ASD, ASA, ESD, ESA, XPR, with unit
  diagonal. Its 28 off-diagonal values are the 28 correlation keys. The mapping is the `get.lsp_xcorr`
  code and must be copied exactly.
- `lsp_xcorr_chk`: true if the Cholesky factorisation of `lsp_xcorr` succeeds. The validator does not
  test this. `gen_lsf_parameters` raises the error later.

`supported_scenarios.m` builds the scenario list:

- It lists `*.conf` in the current folder first, then in `quadriga_src/config`, which it finds by
  searching the MATLAB path. The scenario name is the file name without extension.
- With `parse_shortnames`, each file is scanned for `ShortName = <token>`. The short name is added
  right after its file. A short name that equals any earlier entry is an error ("ambiguous").
- `set_scenario_table` takes the first entry that matches, so a file in the current folder wins over
  a file of the same name in the config folder.

### 4.3 Quirks, and what the port does with each

| # | MATLAB behaviour | Port |
|---|---|---|
| Q1 | Unknown keys are skipped silently, including typos. | Keep. They are reported in the optional output `ignored_keys`. |
| Q2 | A key stated twice keeps the last value. | Keep. |
| Q3 | A `PL_` field can have several values. | Keep. `plpar` values are vectors. |
| Q4 | No `PL_` key gives an empty `plpar`. `PL_` keys without `PL_model` give a `plpar` without model. | Keep. Empty map and empty model string. |
| Q5 | Number tokens: `[0-9e.-]+` for PL, `[0-9Eeji.+-]+` for scenpar. `1E3` in a PL line gives `[1 3]`, `1e+3` gives `[NaN 3]`, `-Inf` gives `NaN`, `1+2i` gives a complex value. | Change. One number grammar for both: optional sign, decimal digits with optional point, optional exponent with `e` or `E` and optional sign, plus `Inf`, `+Inf`, `-Inf` (any case). Anything else is an error. `NaN` and complex values are errors. |
| Q6 | A numeric key with a word as value stores the word as a string (`DS_mu = abc`). A string key with a number stores the number. | Change. Both are errors. |
| Q7 | A key that contains `PL_` in the middle is treated as a PL key and loses its first three characters. | Change. Only keys that start with `PL_` are PL keys. `PL_` alone is an error. |
| Q8 | `PL_model = % foo` reads `foo` from the comment. | Change. The comment is removed first. A missing value is an error, as today. |
| Q9 | `ShortName` is matched anywhere in a line, also inside a comment. | Change. Only on lines that are not comments. |
| Q10 | A file copied from the config folder into the current folder makes its `ShortName` "ambiguous". | Change. A file whose name is shadowed by an earlier directory is skipped completely. |
| Q11 | The 140-field count is checked, the names are not. | Change. Unknown names are errors. Missing names get their default (this is what lets old 140-field sets pass after 27 keys are added). |
| Q12 | `NaN` passes class "real". | Change. `NaN` is an error for every key. `Inf` is allowed for class "real" and for unchecked keys (`ES_D_mu_min` defaults to `-Inf`). |
| Q13 | The 7 `xpr_*` correlations are not range-checked. | Change. Class "correlation" for all 28. |
| Q14 | `SubpathMethod` is not checked. A wrong value fails later in `gen_ssf_parameters`. | Change. Must be `legacy`, `Laplacian` or `mmMAGIC` (case-sensitive). |
| Q15 | A non-positive-definite correlation matrix is not an error at load time. | Keep. It is reported through the return value. |

Every change turns input that is wrong today into an error. None changes the result for a file that
is well formed. The parity test of WI-5 showed this for the 91 shipped files (since removed, see WI-5).

The parser works on bytes. It strips a trailing `\r`, treats space and tab as white space, and does
not decode comment text.


## 5. Analysis: 38.901 v19.5.0 against the QuaDRiGa sources

The existing `3GPP_38.901_*` files cite v16.1.0 (2019-12). Since then the TR received the Rel-19
changes for 7 to 24 GHz (CR 0026 and four corrections), the ISAC channel model (clause 7.9) and one
correction of the ZOD generation (v19.5.0).

### 5.0 Angle convention: elevation in QuaDRiGa, zenith in the TR

QuaDRiGa uses geographic elevation angles: +90 deg (pi/2) is up, 0 is the horizon, -90 deg is down.
TR 38.901 uses zenith angles: 0 is up, 90 deg is the horizon, 180 deg is down. The difference is
historical and deliberate. All keys, file comments and documentation of this port use the QuaDRiGa
names and the elevation convention. No key is named after a zenith quantity.

`elevation = 90 deg - zenith`. What this means for each kind of TR value:

| TR quantity | QuaDRiGa name | Conversion of the table value |
|---|---|---|
| ZSD, ZSA (rms spreads, mean and std of lg) | `ES_D_*`, `ES_A_*` | None. A spread does not change under `90 deg - x`. |
| Cluster ZSA `c_ZSA`, cluster ZSD `c_ZSD` | `PerClusterES_A`, `PerClusterES_D`, `PerClusterES_D_scale` | None. |
| Cross-correlations with ZSD or ZSA | `esD_*`, `esA_*`, `xpr_esd`, `xpr_esa` | None. Correlations of spreads. |
| Correlation distances of ZSD, ZSA | `ES_D_lambda`, `ES_A_lambda` | None. |
| ZOD offset `mu_offset,ZOD` | `EoD_offset_*` | **Sign change.** `EoD offset = - mu_offset,ZOD`. The keys are defined so that the formula of section 7.2 gives the elevation offset directly. |
| Mean ZOD and ZOA of the LOS direction, cluster angles ZOD, ZOA | `EoD`, `EoA` | Not conf data. The consumer computes them in elevation. |

Consequences for the consumer of the keys (not part of this port, noted for the later model port):
in (7.5-19) the terms `X_n · theta'` and `Y_n` are symmetric around zero, so they are the same in
elevation. Only the LOS direction and the offset carry a sign, and both are stated in elevation here.
The Laplacian zenith distribution of the TR is likewise symmetric around the mean.

The session checks every value it transcribes from the TR against this table. A test in WI-6
evaluates the EoD offset formula for each v19 NLOS file at one reference point and compares it with
`-mu_offset,ZOD` computed from the TR formula typed into the test.

I compared the LSP rows, cross-correlations, cluster parameters and correlation distances of Table
7.5-6 with the conf files. I did not re-derive every path-loss constant of the existing files. The
library session transcribes all v19 values from the TR itself and uses this section as a check list.

### 5.1 Values that changed (Table 7.5-6 Part-1, "note 8")

Only UMi and UMa changed. RMa, Indoor-Office and InF rows are unchanged.

| Scenario | Parameter | v16.1.0 conf | v19.5.0 |
|---|---|---|---|
| UMi LOS | DS mu | -7.14 - 0.24 lg(1+fc) | -7.28 - 0.18 lg(1+fc) |
| UMi LOS | DS sigma | 0.38 | 0.39 |
| UMi LOS | ASD sigma | 0.41 | 0.29 + 0.08 lg(1+fc) |
| UMi LOS | ASA mu, sigma | 1.73 - 0.08 lg, 0.28 + 0.014 lg | 1.66 - 0.07 lg, 0.26 + 0.021 lg |
| UMi LOS | ESA mu, sigma | 0.73 - 0.1 lg, 0.34 - 0.04 lg | 0.81 - 0.11 lg, 0.29 - 0.03 lg |
| UMi NLOS | DS mu, sigma | -6.83 - 0.24 lg, 0.28 + 0.16 lg | -6.87 - 0.22 lg, 0.22 + 0.19 lg |
| UMi NLOS | ASD mu, sigma | 1.53 - 0.23 lg, 0.33 + 0.11 lg | 1.54 - 0.24 lg, 0.33 + 0.10 lg |
| UMi NLOS | ASA mu, sigma | 1.81 - 0.08 lg, 0.30 + 0.05 lg | 1.76 - 0.07 lg, 0.27 + 0.05 lg |
| UMi NLOS | ESA mu, sigma | 0.92 - 0.04 lg, 0.41 - 0.07 lg | 0.92 - 0.03 lg, 0.35 - 0.05 lg |
| UMa LOS | DS mu, sigma | -6.955 - 0.0963 lg(fc), 0.66 | -7.067 - 0.0794 lg(fc), 0.57 + 0.026 lg(fc) |
| UMa LOS | ASD mu, sigma | 1.06 + 0.1114 lg(fc), 0.28 | 0.92, 0.31 |
| UMa LOS | ASA mu, sigma | 1.81, 0.20 | 1.76, 0.19 |
| UMa LOS | ESA mu, sigma | 0.95, 0.16 | 0.96, 0.15 |
| UMa LOS | cluster ASD | 5 | 3.58 |
| UMa NLOS | DS mu | -6.28 - 0.204 lg(fc) | -6.47 - 0.134 lg(fc) |
| UMa NLOS | ASD mu, sigma | 1.5 - 0.1144 lg(fc), 0.28 | 1.09, 0.44 |
| UMa NLOS | ASA mu, sigma | 2.08 - 0.27 lg(fc), 0.11 | 2.04 - 0.25 lg(fc), 0.17 - 0.03 lg(fc) |
| UMa NLOS | ESA mu, sigma | 1.512 - 0.3236 lg(fc), 0.16 | 1.445 - 0.2856 lg(fc), 0.17 |
| UMa NLOS | cluster ASD | 2 | 1.8 |
| UMa O2I | ASD mu, sigma | 1.25, 0.42 | 0.58, 0.7 |
| UMa O2I | cluster ASD | 5 | 1.8 |

All of these fit the existing keys (`*_mu`, `*_sigma`, `*_omega`, `*_gamma`, `*_delta`).

### 5.2 New in v19.5.0

- **SMa (suburban macro)**: Table 7.5-6 Part-4 (LOS, NLOS, O2I, valid up to 24 GHz), Table 7.5-12
  (ESD and EoD offset; TR: ZSD and ZOD offset), path loss and LOS probability rows, spatial consistency distances 40/50/15 m.
  The LSP table fits the existing keys.
- **Absolute time of arrival** (Table 7.6.9-1) now covers all scenarios, not only InF: InH -8.6/0.1/10,
  UMi -7.5/0.5/15, UMa -7.4/0.2/50, RMa -8.33/0.26/50, SMa -7.702/0.4/50 (mu, sigma, distance). It fits
  the existing `absTOA_*` keys. New is the upper bound `2L/c` for InF and InH.
- **Near-field model** (7.6.13): `k1` and Beta(alpha, beta) per scenario.
- **Spatial non-stationarity** (7.6.14), stochastic model at the TRP: `Pr_sns` mean and std, and
  `A`, `B`, `R`, `var(delta)` per scenario. The roll-off `C = 13` and the UT-side tables are the same
  for all scenarios and are not conf data.
- **Number-of-cluster variability** (7.6.15): a range [D1, D2] per scenario. D2 equals N.
- **Polarization power variability** (7.6.16): one constant, 3 dB, for all scenarios.

### 5.3 Parts of the TR that QuaDRiGa does not state today (already true for v16.1.0)

These are deviations of the existing files and code from the TR. They are why D2 adds keys.

| # | TR | QuaDRiGa today |
|---|---|---|
| G1 | Lower frequency clamp for LSPs: use fc = 6 GHz below 6 GHz for UMa and InH, fc = 2 GHz below 2 GHz for UMi (Table 7.5-6 notes 6, 7; ZSD tables 7.5-7 to 7.5-12, note 4). | Not modelled. `lsp_vals` evaluates the formulas at the actual frequency. |
| G2 | ESD (TR: ZSD) mean depends on the UT height: `-0.01 (hUT - 1.5)` for UMa and RMa, `+0.01 abs(hUT - hBS)` for UMi LOS, `+0.01 max(hUT - hBS, 0)` for UMi NLOS. | Only the distance term (`ES_D_mu_A`) and the floor (`ES_D_mu_min`) exist. |
| G3 | EoD offset (TR: `mu_offset,ZOD`, 7.5-19) in NLOS for UMa, UMi, RMa, SMa. | Not modelled. No key. |
| G4 | RMa break point `dBP = 2 pi hBS hUT fc / c` (note 5), no environment height. | The RMa files use `PL_E = 13.34`, `PL_hE = 1`, which is the UMa/UMi rule (note 1). |
| G5 | UMa environment height `hE` is random for UTs above 13 m. | Fixed `PL_hE = 1`. |
| G6 | InF-DL: `PL = max(PL_DL, PL_LOS, PL_SL)`. | The `nlos` model takes the maximum of two terms only. The SL term is missing. It matters below about 26 m. |
| G7 | InF LOS shadow fading std 4.3 dB. | `3GPP_38.901_InF_LOS.conf` has `SF_sigma = 4`. To be checked against v16.1.0. |
| G8 | InF delay spread depends on the hall, `lg(26 (V/S) + 14) - 9.35`. | Evaluated for the average of two halls and written as a number. |

G1 to G3 get keys in this port. G4 is fixed in the v19 files with existing keys. G5 and G6 need a
change of the path-loss model and stay open. G7 is fixed in the v19 file. G8 stays as it is.

### 5.4 Changes outside the conf files (not part of this port)

- `qd_layout.gen_o2i_loss`: IRR glass is now `25.4 + 0.11 f` (code: `23 + 0.3 f`). A third model
  ("low-loss A", glass and plywood) exists for SMa. Indoor distance ranges for SMa are new.
- `qd_layout.set_scenario`: needs scenario groups for the v19 file set, the SMa LOS probability
  (Table 7.4.2-1) and the SMa LOS/NLOS state distance (50 m).
- `qd_builder.get_pl`: SMa NLOS has no `max(LOS, NLOS)` in the TR. The `nlos` model always applies it.
  The v19 SMa NLOS file can switch the LOS term off with `PL_A1 = 0`, `PL_B = 0`, `PL_C = 0`. A flag in
  the model would be cleaner.
- ISAC (7.9), blockage (7.6.4) and the antenna changes (7.3) do not touch the scenario files.


## 6. Contract (to be confirmed at review point A)

The declarations go into `include/quadriga_tools.hpp` under a new group line
`// ---- Scenario parameters ----`. All three functions and the file-local key table are in one source
file, `src/qd_scenpar.cpp` (O4). Each function has its own documentation block there, in the
section "Channel generation functions" (O3). The value type is `double` (O1).

```cpp
// Read and validate a QuaDRiGa scenario configuration file
// - Returns true if the LSP cross-correlation matrix is positive definite
// - 'scenpar' always holds all n_key numeric keys; keys not in the file have their default value
bool conf_file_read(const std::string &fn,                                                  // Path to the .conf file
                    std::unordered_map<std::string, double> &scenpar,                       // Output: Numeric scenario parameters, n_key entries
                    std::string *subpath_method = nullptr,                                  // Optional output: Value of 'SubpathMethod'
                    std::string *pl_model = nullptr,                                        // Optional output: Value of 'PL_model', empty if not in file
                    std::unordered_map<std::string, std::vector<double>> *plpar = nullptr,  // Optional output: Path-loss parameters, keys without "PL_"
                    arma::mat *lsp_xcorr = nullptr,                                         // Optional output: LSP cross-correlation matrix, [8, 8]
                    std::vector<std::string> *key_order = nullptr,                          // Optional output: Canonical order of the scenpar keys, [n_key]
                    std::vector<std::string> *pl_order = nullptr,                           // Optional output: plpar keys in file order
                    std::vector<std::string> *short_names = nullptr,                        // Optional output: Values of 'ShortName' in file order
                    std::vector<std::string> *ignored_keys = nullptr,                       // Optional output: Keys in the file that are not known
                    bool validate = true);                                                  // Apply the value checks of 'scenpar_validate'

// Validate scenario parameters and complete them with default values
// - Returns true if the LSP cross-correlation matrix is positive definite
// - An empty 'scenpar' returns the complete default parameter set
bool scenpar_validate(std::unordered_map<std::string, double> &scenpar,                     // In/out: Missing keys are added with their default
                      const std::string &subpath_method = "legacy",                         // Value of 'SubpathMethod'
                      arma::mat *lsp_xcorr = nullptr,                                       // Optional output: LSP cross-correlation matrix, [8, 8]
                      std::vector<std::string> *key_order = nullptr,                        // Optional output: Canonical order of the scenpar keys, [n_key]
                      bool validate = true);                                                // false: only complete with defaults, no value checks

// List the scenarios defined by the conf files in a set of directories
// - Returns the number of entries, n_scen
arma::uword conf_file_list(const std::vector<std::string> &dirs,                            // Directories, searched in this order, not recursive
                           std::vector<std::string> &scenarios,                             // Output: Scenario names and short names, [n_scen]
                           std::vector<std::string> *file_names = nullptr,                  // Optional output: File name incl. ".conf" per entry, [n_scen]
                           std::vector<std::string> *file_dirs = nullptr,                   // Optional output: Directory per entry, as given in 'dirs', [n_scen]
                           bool parse_shortnames = true);                                   // Read the files and add their short names
```

Behaviour to settle with the contract:

- **Key order.** `key_order` lists the 140 legacy names in the order of the `struct(...)` call in
  `set_scenario_table.m` without `SubpathMethod`, then the 27 new keys in the order of section 7.
  `n_key` = 166. The wrappers use it to build the MATLAB struct and insert `SubpathMethod` at its
  legacy position (field 107).
- **Errors.** A file that cannot be opened: `std::runtime_error`. Malformed content and failed value
  checks: `std::invalid_argument`. Messages for file content name the file and the line number,
  for example `"3GPP_x.conf, line 12: Value of 'DS_mu' is not a number."`.
- **Validate first.** The file is parsed and checked completely before any output is written.
- **`validate = false`** skips the value checks only. Grammar errors (Q5 to Q8) are always errors,
  because without them the function could not return a number.
- **Unknown names.** A name in the `scenpar` given to `scenpar_validate` that is not in the key table
  is an error in both modes.
- **Positive-definite test.** Cholesky factorisation of the 8 x 8 matrix, written by hand (no
  LAPACK). The result is true if all 8 pivots are > 0. MATLAB `chol` is the reference.
- **`conf_file_list`.** Per directory, files that end in `.conf` in byte-wise sorted name order.
  For each file one entry with the name without extension, followed by one entry per `ShortName`.
  A file name that an earlier directory already defined is skipped. A short name that equals an
  earlier entry is an `std::invalid_argument`. A directory that does not exist is skipped. An empty
  string in `dirs` means the current directory, as in MATLAB.
- **Not in the contract.** Finding the QuaDRiGa config folder and caching the list are tasks of the
  wrapper or of QuaDRiGa-NG.


## 7. Key table

### 7.1 Legacy keys (140)

Names, order and defaults are copied from the `struct(...)` call in `set_scenario_table.m`, lines
57 to 197. Classes are those of section 4.2 with the changes Q11 to Q14.

For each of the LSPs DS, AS_D, AS_A, ES_D, ES_A, KF, XPR there are 12 keys (`mu`, `sigma`, `lambda`,
`omega`, `gamma`, `epsilon`, `zeta`, `alpha`, `delta`, `kappa`, `tau`, `beta`). SF has 7 (no `mu`,
`gamma`, `epsilon`, `zeta`, `alpha`). ES_D has two more (`mu_A`, `mu_min`). Then 19 model keys
(`NumClusters`, `NumSubPaths`, `r_DS`, the `PerCluster*` keys, `absTOA_*`, `SubpathMethod`,
`LOS_scatter_radius`, `SC_lambda`, `LNS_ksi`, `GR_enabled`, `GR_epsilon`) and 28 correlations.

### 7.2 New keys for 38.901 v19.5.0 (27)

All defaults reproduce today's behaviour. Names are proposals.

| Key | Default | Class | Meaning | TR |
|---|---|---|---|---|
| `LSP_fc_min` | 0 | non-negative | Lower clamp of fc in GHz for the frequency terms of the LSPs and of the EoD offset. 0 = none. | 7.5-6 notes 6, 7 |
| `ES_D_mu_B` | 0 | real | UT-height term of the ESD mean in lg(deg)/m, applied as `B · hUT`. | 7.5-7, 7.5-9 |
| `ES_D_mu_C` | 0 | real | Term `C · abs(hUT - hBS)` of the ESD mean. | 7.5-8 LOS |
| `ES_D_mu_D` | 0 | real | Term `D · max(hUT - hBS, 0)` of the ESD mean. | 7.5-8 NLOS |
| `EoD_offset_e` | 0 | real | Constant of the EoD offset in deg. Sign opposite to the TR value `e`. | 7.5-7 |
| `EoD_offset_e_gamma` | 0 | real | Frequency term of `e`, in deg/lg(GHz). Sign opposite to the TR. | 7.5-7 |
| `EoD_offset_a` | 0 | real | Distance exponent `a`, as in the TR. | 7.5-7, 7.5-8 |
| `EoD_offset_a_gamma` | 0 | real | Frequency term of `a`. | 7.5-7 |
| `EoD_offset_b` | 0 | non-negative | Minimum distance `b` in m. 0 switches the power term off. | 7.5-7, 7.5-8 |
| `EoD_offset_c` | 0 | real | Constant `c` of the exponent. | 7.5-7, 7.5-8 |
| `EoD_offset_c_gamma` | 0 | real | Frequency term of `c`. | 7.5-7 |
| `EoD_offset_c_hUT` | 0 | real | UT-height term of the exponent, per m. | 7.5-7 |
| `EoD_offset_hcl` | 0 | non-negative | Clutter height in m of the geometric offset. 0 switches it off. | 7.5-9, 7.5-12 |
| `PerClusterDS_fc_min` | 0 | non-negative | Lower clamp of fc in GHz for the cluster delay spread formula (`PerClusterDS`, `PerClusterDS_gamma`). 0 = none. Independent of `LSP_fc_min` (O9). | not in the TR |
| `PerClusterES_D_scale` | 0 | non-negative | If > 0, the cluster ESD of a link is `scale · 10^(ESD mean)` with the ESD mean of that link (formula below), and `PerClusterES_D` is not used. 0 = use `PerClusterES_D`. The TR value is 0.375 (O11). | (7.6-7) |
| `absTOA_Lmax` | 0 | non-negative | Largest room dimension L in m. The excess delay is bounded by 2L/c. 0 = no bound. | 7.6.9 |
| `NumClusters_min` | 0 | integer >= 0 | Lower end D1 of the cluster-count range, counted like `NumClusters`. 0 = fixed count. Must not exceed `NumClusters`. | 7.6.15-1 |
| `NF_k1` | 0 | integer >= 0 | Number of strongest clusters with scaling 1. | 7.6-13-1 |
| `NF_alpha` | 0 | non-negative | Beta distribution parameter alpha. 0 = no near-field data. | 7.6-13-1 |
| `NF_beta` | 0 | non-negative | Beta distribution parameter beta. | 7.6-13-1 |
| `SNS_mu` | 0 | in [0, 1] | Mean of `Pr_sns`. | 7.6.14.1.2-1 |
| `SNS_sigma` | 0 | non-negative | Std of `Pr_sns`. | 7.6.14.1.2-1 |
| `SNS_A` | 0 | non-negative | Visibility-region parameter A. | 7.6.14.1.2-2 |
| `SNS_B` | 0 | non-negative | Visibility-region parameter B. | 7.6.14.1.2-2 |
| `SNS_R` | 0 | non-negative | Visibility-region parameter R in dB. 0 where the TR says "NA". | 7.6.14.1.2-2 |
| `SNS_delta_var` | 0 | non-negative | Variance of delta. | 7.6.14.1.2-2 |
| `PolVar_sigma` | 0 | non-negative | Std in dB of the polarization variability powers. | 7.6.16 |

Formulas that the keys state (constants are folded, as the path-loss keys already do):

```
ESD mean  = max( ES_D_mu_min,
                 ES_D_mu + ES_D_mu_A · d2D/1000 + ES_D_mu_B · hUT
                         + ES_D_mu_C · abs(hUT - hBS) + ES_D_mu_D · max(hUT - hBS, 0)
                         + (existing omega, gamma, epsilon, zeta, alpha terms) )

EoD offset [deg, elevation, positive = upwards] =
      EoD_offset_e + EoD_offset_e_gamma · lg(fc)
    + 10^( (EoD_offset_a + EoD_offset_a_gamma · lg(fc)) · lg(max(EoD_offset_b, d2D))
           + EoD_offset_c + EoD_offset_c_gamma · lg(fc) + EoD_offset_c_hUT · hUT )      if EoD_offset_b > 0
    + atan((hcl - hBS)/d2D) - atan((hUT - hBS)/d2D)                                     if hcl > 0 and hUT < hcl
```

The EoD offset is the TR's `mu_offset,ZOD` with the opposite sign (section 5.0). Compared with the
TR formula, `e` and its frequency term change sign, the power term is added instead of subtracted
(its exponent parameters `a`, `b`, `c` are unchanged), and the geometric term is written with
elevation angles: it is the elevation of the clutter top seen from the BS minus the elevation of the
UT seen from the BS. A positive offset raises the departure direction. The UMi, RMa and SMa offsets
are always positive. The UMa offset can become negative at high frequency and large distance (for
example about -2.6 deg at 28 GHz and 1000 m).

Example, UMa NLOS: `ES_D_mu = 0.915` (0.9 + 0.01 · 1.5), `ES_D_mu_A = -2.1`, `ES_D_mu_B = -0.01`,
`ES_D_mu_min = -0.5`; `EoD_offset_e = 5.96` (TR: -5.96), `_e_gamma = -7.66` (TR: 7.66),
`_a = -0.782`, `_a_gamma = 0.208`, `_b = 25`, `_c = 2.135` (2.03 + 0.07 · 1.5), `_c_gamma = -0.13`,
`_c_hUT = -0.07`; `LSP_fc_min = 6`.

Example, UMi NLOS: TR `-10^(-1.5 lg(max(10, d2D)) + 3.3)` becomes `EoD_offset_a = -1.5`,
`EoD_offset_b = 10`, `EoD_offset_c = 3.3`, all other offset keys 0. At d2D = 100 m the EoD offset is
+2.0 deg.

Example, RMa and SMa NLOS: `EoD_offset_hcl = 3.5`. At hBS = 35 m, hUT = 1.5 m, d2D = 500 m the EoD
offset is +0.23 deg.


## 8. Work items

The workflow of `md/SPEC_CPP_API.md` applies to each function. Review point A covers all three
contracts at once, review point B all three test files.

### WI-0 Orientation

Read the sources named at the top. Confirm the survey of section 4.1 on the files. Report anything
that contradicts this plan.

### WI-1 Contract, then review point A

Write the declarations of section 6 into the header and draft the key table of section 7 (it will be
file-local in `src/qd_scenpar.cpp`). The open points O1 to O12 are answered (section 3). Present the
key names of section 7.2 and anything in the contract that is still unclear, and wait.

### WI-2 Stubs

`src/qd_scenpar.cpp` with the key table and three stubs. It compiles without warnings and the stubs
return the default parameter set.

### WI-3 Tests, then review point B

Test files `test_conf_file_read.cpp`, `test_scenpar_validate.cpp`, `test_conf_file_list.cpp`. The
tests write their own small conf files with distinct names and remove them. Expected values are
written by hand in the test, not taken from the parser.

- Defaults: an empty file gives all 166 defaults. The 140 legacy defaults are compared with a table
  typed from `set_scenario_table.m`.
- Grammar: comment before and after the `=`, tabs, CRLF, no space around `=`, exponent forms, `Inf`,
  duplicate key, unknown key in `ignored_keys`, non-ASCII bytes in a comment.
- Path loss: no `PL_` key, model only, several values on one line, file order in `pl_order`.
- `lsp_xcorr`: each of the 28 correlation keys set alone to a distinct value lands in exactly two
  symmetric positions. The expected positions are typed from `get.lsp_xcorr`.
- Positive-definite flag: identity (true), a matrix with one correlation of 1 (false), a known
  indefinite 3-parameter case (false), a shipped example (true).
- Validation: one section per class and per changed quirk Q5 to Q14, each next to a passing baseline.
- `scenpar_validate`: empty map, 140-key legacy map (passes and grows to 166), unknown key, each class.
- `conf_file_list`: two directories with a shadowed file, short names, the ambiguity error, a
  missing directory, `parse_shortnames = false`.
- Optional outputs: each requested alone and all together give the same values.

### WI-4 Implementation

Parser, validator, lister. No regular-expression library is needed: the grammar of section 4.3 is a
hand-written scanner over one line.

### WI-5 Parity with MATLAB on the 91 shipped files

Status (2026-10-14): done and removed. The parity test ran in the sessions of commits 338eaa0 and 3690107
and last passed on 2026-10-14 with the reader of 3690107: every value of 90 files equal to MATLAB, no ignored
keys, the same lister entries; `TwoRayGR.conf` is rejected by the reader (`GR_enabled = 1;`, clean
grammar). The test, the MATLAB script and the reference file were then deleted, because quadriga-lib must
not depend on QuaDRiGa-NG: no path to it in the sources and no data derived from its files. The rules below
describe what the removed test did.

This is the acceptance test for "drop-in". It is a test case named `"Diagnostic - ..."` that runs only
if the directory `/sjc/QuaDRiGa-NG/quadriga_src/config` exists and is skipped otherwise.

- Reference: a MATLAB or Octave script calls `qd_builder(name)` for every scenario of
  `qd_builder.supported_scenarios(0)` and writes `scenpar`, `plpar`, `lsp_xcorr` and `lsp_xcorr_chk`
  with 17 significant digits to one text file in `quadriga-lib/tests/data/`. The script only reads
  QuaDRiGa-NG.
- The test reads the same 91 files with `conf_file_read` and compares every value exactly
  (`strtod` and `str2double` both round correctly, so equality is expected; report any difference
  instead of loosening the comparison).
- It also checks that `ignored_keys` is empty for every file (`ShortName` is a known key and is not
  reported there) and that `conf_file_list` returns the same names as `supported_scenarios(1)`.

### WI-6 The v19.5.0 conf files

23 files, prefix `3GPP_38.901_v19_`:

- UMi, UMa, RMa, SMa: `LOS`, `NLOS`, `LOS_O2I`, `NLOS_O2I` each (16 files)
- Indoor: `LOS`, `NLOS` (2 files)
- InF: `LOS`, `NLOS_SL`, `NLOS_DL`, `NLOS_SH`, `NLOS_DH` (5 files)

Rules:

- Layout and comment style of the existing files, with clause and table references to v19.5.0.
- `NumClusters` is N for LOS files and N + 1 for NLOS and O2I files, as in the existing files
  (QuaDRiGa always keeps a LOS path, switched off by `KF_mu = -100`).
- O2I files use the O2I column for the LSPs. `LOS_O2I` and `NLOS_O2I` differ in the path loss and in
  the ESD and EoD offset rows (TR: ZSD and ZOD offset), which follow the outdoor state (TR tables
  7.5-7 to 7.5-12, note 5).
- Path loss with the existing models. RMa and SMa LOS use the break point of note 5:
  `PL_E = 20.944` (2 pi / 0.3), `PL_hE = 0`.
- SMa constants for h = 10 m, W = 10 m: LOS `PL_A1 = 21.574`, `PL_B = 30.133`, `PL_C = 20`,
  `PL_D = 0.002`, `PL_A2 = 40`, `PL_sig1 = 4`, `PL_sig2 = 6`; NLOS `PL_An = 43.42`, `PL_Bn = 36.15`,
  `PL_Cn = 20`, `PL_Dn = -15.07`, `PL_D2n = 370`, `PL_Fn = -3.1`, `PL_G1n = -3.2`, `PL_G2n = 11.75`.
  The session re-derives these from Table 7.4.1-1 and reports a mismatch.
- New keys of section 7.2 with the table values. `PerClusterES_D_scale = 0.375` in all v19 files;
  `PerClusterES_D` is also stated, evaluated at the constant part of the ESD mean, for consumers that
  do not know the new key. `PerClusterDS_fc_min` stays 0.
- Absolute TOA in NLOS and NLOS_O2I files only (O10).
- `KF_lambda` in NLOS files and `XPR_lambda` are carried over from the matching v16 file, SMa from
  UMa, with a comment that they are not from the TR (O12).
- The files go to `quadriga-lib/data/config/` with the Apache-2.0 header (O5).
- All comments in the files use the QuaDRiGa names (ESD, ESA, EoD) and the elevation convention of
  section 5.0. The EoD offset keys carry the converted signs.
- A test evaluates the EoD offset formula of section 7.2 for each v19 NLOS and NLOS_O2I file at one
  reference point (fc, d2D, hBS, hUT) and compares it with `-mu_offset,ZOD` from the TR formula typed
  into the test.
- A test reads all 23 files: no ignored keys, validation passes, the correlation matrix is positive
  definite or the file is reported.

### WI-7 Documentation and final report (review point C)

Documentation blocks for the three functions, section "Channel generation functions". The report lists test results, every quirk of section
4.3 with the test that pins it, the parity result of WI-5 and the open points that remain.


## 9. Hand-over for the QuaDRiGa-NG integration (later session)

- `set_scenario_table.m` and `check_scenario_parameter_table.m` are replaced by calls to the MEX
  wrappers. `supported_scenarios.m` keeps finding the config folder on the MATLAB path and passes
  `{'', config_folder}` to the lister.
- `scenpar` grows from 140 to 167 fields. Places that assume 140: the count check in
  `check_scenario_parameter_table.m`, `write_conf_file.m` (must write the new keys) and saved builder
  objects with a 140-field `Pscenpar`. The constructor path `qd_builder([], [], scenpar)` and
  `scenpar_nocheck` bypass validation; code that reads a new field must not assume it exists there, or
  these paths must complete the struct with `scenpar_validate(..., validate = false)`.
- `plpar` stays a struct with the model as a string field and numeric row vectors, `[]` when empty.
- Error identifiers change (`QuaDRiGa:qd_builder:WrongInput` becomes the MEX error). Tests that match
  on the identifier need a look.
- The v19 files are copied to `quadriga_src/config`. The old MATLAB parser already tolerates them,
  because it skips unknown keys.
- `qd_layout.set_scenario` needs groups such as `3GPP_38.901_v19_UMa`. See section 5.4 for the model
  changes that the v19 set needs before it is a complete implementation of the TR.
