# Code review round 1: large-scale model port

Reviewer: independent agent. Scope: `git diff 46dd28d..HEAD -- . ':!md'` at HEAD 00e19aa (branch `fixes`), 88
files, +11821 / -513 lines; no uncommitted code changes (only `md/prompts/code_review_r1.md` is modified).
Findings table: `md/gbsm_lsf_code_findings.md` (C1-01 to C1-20).

## 1. Method

1. Read the spec (`md/gbsm_lsf_spec.md`, all sections), SPEC_CPP_API, the relevant parts of SPEC_MATLAB_API and
   SPEC_PYTHON_API, and the "Decisions" and "Open points" of the run reports r1 to r4b.
2. Read every new or changed source line: the public header, the private header, the six new source files, the
   `qd_scenpar.cpp` diff, the six MEX and six Python wrappers, `python_gbsm.cpp`; the 23 v19 conf files (PL keys,
   new keys, comments); the Catch2 tests (helpers and oracles in full, case lists and tolerances of all files).
3. Checked formulas and constants against TR 38.901 v19.5.0 (Table 7.4.1-1 with notes 1 and 5, Table 7.4.2-1,
   clause 7.4.3 with Tables 7.4.3-1 to 7.4.3-3, clause 7.5 step 4, Table 7.5-6 Parts 1 and 3, Tables 7.5-7 to
   7.5-12, clause 7.6.3.3 with eq. 7.6-18, Table 7.6.3.1-2, clause 7.6.9, Table 7.8-7) and against QuaDRiGa-NG
   (`get_pl.m`, `gen_lsf_parameters.m`, `qd_builder.m` `lsp_vals`, `gen_o2i_loss.m`, `set_scenario.m`).
4. Rebuilt `build_linux` (up to date) and ran the Catch2 cases of the new and changed functions once.
5. Built a separate ASan/UBSan tree `build_asan` and ran the same cases plus `conf files v19*`.
6. Wrote probe programs in the scratchpad to confirm suspected bugs (C++ against `build_linux/libquadriga.a`;
   Python against the installed module `lib/quadriga_lib...so`, built from the current wrapper sources).
7. Grep for internal references (`md/`, spec sections, planning IDs, local paths, NG line numbers) over all files of
   the diff; compile of the six new sources with extra warnings (`-Wshadow -Wconversion -Wsign-conversion
   -Wfloat-equal`) for information.

Resource use: at most 12 threads, one build at a time; load average 2.7 to 4.6 during the runs (11 at the peak of
the ASan build, which was this build itself). No timing below is a benchmark; the two times in C1-02 are
indicative only (load 2.8).

## 2. Builds and test runs

| Run | Command | Result |
|---|---|---|
| Plain build | `cmake --build build_linux --parallel 12 --target test_bin` | up to date, 0 warnings |
| Plain tests | `build_linux/test_bin "lsf_*,lsp_generate*,path_loss*,los_probability*,o2i_loss*,conf_file_*,scenpar_validate*"` | All tests passed (12159 assertions in 112 test cases), 0.44 s, load 3.7 |
| ASan/UBSan build | `cmake -B build_asan -D CMAKE_CXX_COMPILER=g++-12 -D ENABLE_TESTS=ON -D ENABLE_AVX2=ON -D CMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" ...`, `--parallel 12` | built in 8 min 26 s, 0 warnings |
| ASan/UBSan tests | same filter plus `conf files v19*`, `ASAN_OPTIONS=detect_leaks=1:halt_on_error=0`, `UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0`, `OMP_NUM_THREADS=8` | All tests passed (14901 assertions in 119 test cases); no AddressSanitizer error, no leak; 8 UBSan reports at 3 code sites (C1-03, C1-04, C1-05) |

The UBSan reports are all "reference binding to null pointer" inside Armadillo's `colptr` / `col` on an empty
matrix (memptr null) for `n_link = 0`: `los_probability.cpp:385` (test_los_probability.cpp:633),
`lsp_generate.cpp:254` twice (test_lsp_generate.cpp:1074 via `run_all`), `o2i_loss.cpp:112`
(test_o2i_loss.cpp:323). Harmless on GCC, but undefined behaviour and noise that would hide a real report later.
No huge-allocation case exists in the new code (all allocations scale with the inputs), so no `ulimit -v` probe
under ASan was needed; the memory probe of C1-02 ran in the plain build under `ulimit -v 3000000`.

The full suite, the MEX suite and pytest were not run (the implementers' reports show them green, and the only
commit after Run 4b changed the `path_loss` documentation text).

## 3. Probes and results

All probes are in the scratchpad (`probes/`); none touches the repository.

| Probe | What | Result |
|---|---|---|
| `py_flags.py` | Python flags with `None` and a float | `remove_bias=None` equals `False` (default True); `horizontal=None` gives z-dependent (3D) values; `complex=None` returns the 7-tuple; `remove_bias=2.5` accepted; conf function `validate=None` = default, `validate=2.5` TypeError; `d_clutter=None` TypeError (C1-01, C1-20) |
| `probe1.cpp` A1 | `o2i_loss` with `loss` aliasing `rnd` | `std::logic_error` "Mat::operator(): index out of bounds" (C1-06) |
| `probe1.cpp` A2 | `lsp_generate` with `gr_eps_re` aliasing `rnd` | same `std::logic_error`, after `lsp` was written (C1-06) |
| `probe1.cpp` A3 | `path_loss` with `sf_sigma` aliasing `rx_pos` | path loss wrong by 28.4 dB, no error (C1-06) |
| `probe1.cpp` B1 | `los_probability` InF_SL, `d_clutter = 1e-310` | p_los = {NaN, 0} for d = {0, 5} (C1-09) |
| `probe1.cpp` C1 | `DS_omega -10, DS_gamma 1, DS_delta 0.1`; `SF_omega -10, SF_delta 1` | DS lsp NaN, mu NaN, sigma 0; default `sf_sigma` 0 (C1-08) |
| `probe1.cpp` D1 | symmetric root, correlation 1 - eps between ASD and ASA | max abs(X·X - R) 1.3e-15 (eps 1e-2), 4.4e-16 (1e-6), 1.6e-11 (1e-10), 1.1e-9 (1e-13); X exactly symmetric: the iteration converges as the spec states |
| `probe1.cpp` E1 | floor ID 2^64 - 1 | finite values, no overflow in the offset rule |
| `probe2.cpp` | `path_loss`, 1e6 links x 4 frequencies, with and without `sf_sigma` | peak RSS 59 MB vs 559 MB, 0.11 s vs 0.73 s (C1-02) |
| `probe3.cpp` | random hE with `rnd_he = 1`, UMa LOS, hUT 20 m, d2D 4/6/10 km | 132.29 / 139.34 dB / NaN (C1-07) |
| `py_probit.py` | soft LOS probit through `los_probability` (RMa, rnd_g 0, fc 1 kHz) against Python `NormalDist.inv_cdf` for p from 1e-300 to 1 - 1e-12 | worst relative error 1.4e-14 in [1e-12, 1 - 1e-12] (incl. the tan/atan round trip): the probit and both tails are correct |

## 4. What was verified and found correct

- Path loss: the 10 models against `get_pl.m` (distances 2D/3D per model, height clamps, breakpoint logic, the
  satellite elevation and zenith table, `nlos` terms and the maximum); the deliberate differences of the spec
  (`winner_los` with C2 and per-link heights, 3D breakpoint distance, random hE without the UT clamp, the legacy
  O2I distance rule of `gen_o2i_loss.m`) are implemented as specified. The UMa random hE matches note 1 (set
  {12, 15, ..., hUT - 1.5}, probability 1/(1 + C), C' for 13 <= hUT <= 23).
- v19 path-loss constants recomputed: RMa A1 20.478 -> 20.48, B 31.741 -> 31.74, D 0.0013979 -> 0.001398,
  Bn 31.7548 -> 31.75, Dn -15.07, D2n 92.5; SMa A1 21.5744 -> 21.574, B 30.1327 -> 30.13, Bn 36.15, D2n 370;
  E = 20.944 -> 20.94 and 40/3 -> 13.33; UMa NLOS Bn 14.44 / E3n -0.6, UMi NLOS Bn 22.85 / E3n -0.3, InH, InF
  (incl. the InF-DL term `Ax/Bx/Cx` = 25.5/33/20) and SMa NLOS without the LOS term all equal Table 7.4.1-1.
- New keys: LOS-state 60/50/50/50/10/d_clutter/2, IO-state 50, `O2I_d2D_in_max` 10/25, `absTOA_Lmax` 120/300/120
  (L = max dimension of the halls of Table 7.8-7 and the office), `DS_VS` 4 / 50/11 / 47/11 with the right hall per
  sub-scenario, `DS_VS_a/b` 26/14 and 30/32, DS_mu -9.35/-9.44: all as in the TR.
- LSP mean/std: identical to `lsp_vals` + `gen_lsf_parameters.m` plus the v19 terms; UMa/InH ZOD offset with the
  6 GHz clamp (Table 7.5-7 note 4) through `LSP_fc_min`; EoD offset sign (`-mu_offset,ZOD`) and constants of UMa
  NLOS (e 5.96, e_gamma -7.66, a -0.782, a_gamma 0.208, b 25, c 2.135, c_gamma -0.13, c_hUT -0.07) and UMi NLOS;
  SMa `hcl` term.
- Cholesky in the TR order [SF, K, DS, ASD, ASA, ZSD, ZSA] (+ XPR) with the index mapping Q(pi[a], pi[b]) = L(a, b);
  `x_SF = s_SF`; Newton-Schulz root as specified; abs TOA rule and bound; ground permittivity = MATLAB.
- LOS probability: every TR row of Table 7.4.2-1, the InF and SMa edge rules of the spec, the 38.811 tables and
  `QuaDRiGa_Industrial` = `set_scenario.m`; soft LOS = eq. 7.6-18.
- O2I: Tables 7.4.3-1 to 7.4.3-3 (v19 IRR glass), clause 7.4.3.2, the model choice with zero weights.
- LSF: generator table, mirror relations, bias removal over the union in dual mobility, Tx side as the mirror field
  with the mirror's frequency table, normalisations of the self-mirrored and paired generators, floor offsets.
  The tests pin these with independent `ref_sos` oracles, including bit-exact swap reciprocity.
- No internal references in produced files (grep over all 88 files: no `md/`, spec section, planning ID, local path
  or NG line number; the conf tests cite `qd_builder.m` of published QuaDRiGa, which is allowed).
- MSVC: no GCC extensions, no VLAs, no OpenMP; `NAN` defaults in the header have precedents in
  `quadriga_lib.hpp`/`quadriga_tools.hpp`; no BLAS/LAPACK calls (the 8x8 products are hand-written); no
  floating-point `std::to_chars`.

## 5. Findings rationale

### Majors

- **C1-01 Python flags.** The new wrappers follow SPEC_PYTHON_API (native `bool`), the conf wrappers of the same
  submodule do not (`qd_python_flag`). With pybind11's caster `None` becomes `False`; for `horizontal` this
  silently selects the legacy 3D correlation, for `remove_bias` it switches the bias removal off, for `cholesky` /
  `clip_spreads` it selects the legacy root / no clipping. Users who write `f(..., x=None)` for "default" get the
  legacy model without notice. Because the spec prescribes plain `bool`, the user decides; my recommendation is
  `qd_python_flag` for every flag of `gbsm` (it already exists, rejects non-bool, and makes the submodule uniform)
  and a line in SPEC_PYTHON_API.
- **C1-02 sf_sigma cost.** `qd_gbsm_lsp_mu_sigma` evaluates 8 LSPs x (mean, std) for every link and frequency to
  provide one std slice. For a coverage map this is 16x the needed work and 8x the output memory, and the Python
  wrapper requests `sf_sigma` on every call. A slice mask or an SF-only helper keeps the "one formula for both
  callers" requirement of the spec.

### Minors and nits

- **Sanitizer reports (C1-03 to C1-05)**: three one-line guards.
- **Aliasing (C1-06)**: SPEC_CPP_API requires the documentation to say so when aliasing is not supported. One
  probe gives a silently wrong path loss; the others throw `std::logic_error` (not `std::invalid_argument`) after
  outputs were changed, which also breaks "a failed call leaves the outputs untouched". A sentence per block fixes
  the contract; a check would make it safe.
- **NaN paths (C1-07, C1-08, C1-09)**: all three need inputs at the edge of the validated domain (`rnd_he = 1` with
  C below 1.1e-16, e.g. d2D beyond about 7.5 km at hUT 20 m; `*_omega + f' <= 0`; subnormal `d_clutter`), but each returns NaN or a
  silently wrong 0 from inputs the validation accepts. C1-08 is the most relevant: an invalid frequency term gives
  a NaN mean with a std of 0, i.e. the std looks valid.
- **C1-10 floors in dual mobility**: a modelling gap rather than a bug; the TR rule is for UTs on different floors,
  and in D2D both ends are UTs. Documenting the assumption is enough for now; an optional Tx floor ID would be the
  complete solution and would also serve the SSF port.
- **C1-11 RMa EoD offset**: the TR row is the SMa-type formula evaluated at the default heights. The library
  generalises it, which gives 0 for UTs above 3.5 m (RMa allows 1 to 10 m). Either is defensible; the TR wins
  unless a reason is stated, so the choice should be written down.
- **C1-12, C1-13, C1-16, C1-18**: lean-code items (duplicated validation, a double preset lookup, a dead return
  value, helpers that the user's scoping rule makes lambdas, duplicated test helpers, an extra cube copy).
- **C1-15**: the user's binding criterion (library calls for supporting tasks such as interpolation) conflicts with
  spec 5.0 (test-side ACF interpolation); `sos_acfi` and `interp_1D` are public, tested functions.
- **C1-17**: the only branch combination of `lsf_draw` without an independent oracle.
- **C1-19, C1-20**: wording and Python input conveniences.

### Spec points where the code is right and the spec is not

- Spec 5.4 asks for "the ESD height terms" of SMa NLOS; Table 7.5-12 has a constant ZSD (0.14). The test checks the
  absence of height terms (r2b decision 13); correct.
- Spec 5.3 lists TR points beyond the TR distance ranges and states SMa up to 10 km; the TR ends SMa at 5 km. The
  test restricts the points (r3a decision 1); correct. The 10 km bound for SMa in 4.4.2 is harmless but not TR.
- Spec 2.1 / 3.0.5 places the generic phase-set functions in two files and the LSF helpers in the private header;
  see the proposal below for a leaner layout that keeps the SSF requirement.

## 6. File structure proposal (C1-14)

Current: `lsf_init_phases.cpp` 200, `lsf_draw.cpp` 311, `lsp_generate.cpp` 378, `path_loss.cpp` 595,
`los_probability.cpp` 390, `o2i_loss.cpp` 220 lines, `quadriga_gbsm_helpers.hpp` 254 lines with six entries.

| Candidate | Shared code that would become file-local | Effect | Judgement |
|---|---|---|---|
| `lsf_init_phases` + `lsf_draw` | `qd_gbsm_gen`, `qd_gbsm_lsf_generators`, `qd_gbsm_freq`, `qd_gbsm_positions` (4 header entries, ~85 lines incl. the duplicated generator table) and the three generic functions | one file of ~520 lines that holds the whole phase-set machinery next to its two users; one preset lookup; shared validation; the two test files merge too (~80 duplicated test lines go) | worth it |
| `path_loss` + `lsp_generate` | `qd_gbsm_lsp_mu_sigma` | ~950 lines with two large, unrelated documentation blocks; `qd_gbsm_chol8` stays shared with `qd_scenpar.cpp` anyway | not worth it |
| `los_probability` + `o2i_loss` | none | ~600 lines, only grouping of the two layout functions | not worth it |

Target layout:
- `src/qd_lsf.cpp` (name in the style of `qd_scenpar.cpp`; alternatively keep `lsf_init_phases.cpp` as the file
  name): section statement, anonymous namespace with `qd_gbsm_gen`, `qd_gbsm_lsf_generators`, `qd_gbsm_freq`,
  `qd_gbsm_positions`, `qd_gbsm_floor_offset`, `qd_gbsm_eval`, `qd_gbsm_remove_bias` (signatures unchanged), then the
  documentation blocks and definitions of `lsf_init_phases` and `lsf_draw`.
- `src/quadriga_gbsm_helpers.hpp`: `qd_gbsm_lsp_mu_sigma` (path_loss, lsp_generate; with an LSP mask for C1-02),
  `qd_gbsm_chol8` (lsp_generate, qd_scenpar), `qd_gbsm_check_links`, `qd_gbsm_check_fc` (C1-12). Its documentation
  block shrinks to the users of each helper and the LSP order/units; the LSF generator index stays in the public
  header and in the merged file.
- `path_loss.cpp`, `lsp_generate.cpp`, `los_probability.cpp`, `o2i_loss.cpp` unchanged.
- Tests: `test_lsf_init_phases.cpp` + `test_lsf_draw.cpp` -> one file with one fixture block (the case names keep
  their `lsf_init_phases - ...` / `lsf_draw - ...` prefixes).

SSF effect: the SSF phase sets (cluster delays, angles, XPR, phases) are evaluated with the same machinery. With
the merge, the SSF functions that create and evaluate those sets are added to the same file and the machinery stays
file-local; only if the SSF functions must live elsewhere do the seven helpers move to the private header (as the
spec already plans for three of them). The signatures are generic today (driven by the generator table, one
`sos_val` / `sos_remove_bias` call per frequency table and side), so the merge neither helps nor hurts that
requirement; it removes the split of one mechanism over two source files and a header.

## 7. SSF readiness

`qd_gbsm_eval`, `qd_gbsm_floor_offset` and `qd_gbsm_remove_bias` have the signatures of spec 3.0.5, no LSF
constants, no captured state, and group calls by frequency table, so sets with hundreds of generators at one
decorrelation distance cost one call per table and side. `qd_gbsm_freq` and `qd_gbsm_positions` are generic;
`qd_gbsm_lsf_generators` is the only LSF-specific helper, as intended. `lsp_generate` provides every quantity of
spec section 8 (`lsp`, `lsp_mu` incl. the ESD floor for `c_ZSD`, `eod_offset_deg`, `abs_toa`, ground permittivity),
and `rnd = zeros` gives the deterministic values. Open for the SSF port: the floor of a Tx device in dual mobility
(C1-10) applies to cluster variables in the same way.

## 8. Untested areas

- `lsf_draw` in dual mobility with floor IDs: no oracle (C1-17).
- `lsp_generate`: the path where `scenpar_validate` accepts R but `qd_gbsm_chol8` fails on the permuted matrix
  (hard to construct; the message is shared with the tested non-PD path).
- Edge inputs at the validation boundary: `rnd_he = 1` at large distance, `*_omega + f' <= 0`, subnormal
  `d_clutter` (C1-07 to C1-09); aliasing of outputs and inputs (C1-06).
- MEX: zero links (accepted restriction, spec 6.1); `uint8` floor IDs (rejected by the helpers, user decision).
- Python: `None` for flags (C1-01) is not covered by the tests, which pass explicit booleans.
- `path_loss` performance for large maps (C1-02); no test can see it.

## 9. Overall judgement

The port is in good shape. I found no wrong formula, constant, unit, index or sign: path loss, LOS probability,
O2I loss, LSP statistics, the TR-order Cholesky, the symmetric root, the probit and the LSF field machinery agree
with TR 38.901 v19.5.0 and with QuaDRiGa-NG where the spec keeps its behaviour, and the 23 conf files carry the
right values. The tests are thorough, use independent oracles (typed TR formulas, `ref_sos`, test-side Cholesky,
bisection probit) and state their tolerances; they pass in the plain and in the ASan/UBSan build.

No blocker. Two majors: the Python flag semantics (C1-01, a decision) and the memory/time cost of the default
`sf_sigma` (C1-02, an easy fix). The minors are robustness at the validation boundary, three sanitizer reports
with one-line fixes, undocumented aliasing, two modelling decisions (floors in dual mobility, RMa EoD offset), lean
code, and the file layout. Recommended before the merge: C1-01 (after the decision), C1-02, C1-03 to C1-08, the
documentation sentences of C1-06 and C1-10; the rest can follow with the layout change of C1-14 if the user wants
it.
