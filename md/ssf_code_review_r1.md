# SSF code review, round 1 (cluster level, commits 408c10e … da9c787)

Scope: `git diff 26da9f93..HEAD -- . ':!md'` — `ssf_init_phases`, `ssf_draw` (src/qd_gbsm_phases.cpp),
`ssf_clusters` (src/ssf_clusters.cpp), header, six v19 LOS conf files, Catch2/MEX/Python tests, MEX/Python wrappers,
SECTION_DESC, html_docu. Reference: `md/ssf_spec.md`, TR 38.901 v19.5.0, coding rules `md/SPEC_*.md`. Appendix with
the proposed documentation texts: `md/ssf_code_review_r1_docs.md`.

How measured: library `/home/stephan/quadriga-lib-agents/build_ssf/libquadriga.a` (HEAD build) and the Python module
`lib/quadriga_lib.cpython-314-*.so` of the worktree; scratch programs in
`/home/stephan/quadriga-lib-agents/scratch/ssf_cr1/` (`probe1.cpp`, `probe2.cpp` fuzz, `bug1.py`, `bug1b.py`,
`bug1c.py`, `wrap.cpp`). Sanitizer build `/home/stephan/quadriga-lib-agents/build_ssf_asan` (library and the four SSF
test files plus `test_conf_files_v19.cpp`, `-fsanitize=address,undefined`). Load average 1–6 during all runs; no
timings are reported.

Counts: **1 bug, 3 major, 8 minor.**

---------------------------------------------------------------------------------------------------------------------

## A. Bugs

### B1. `quadriga`: delays in seconds of `-ln(u)` when the cluster delays have no spread

`src/ssf_clusters.cpp:253`: `s_tau += ((den > 0.0) ? ... : 1.0) / F`. The delay scale `s_k` has the unit [s]
(`delay = s_tau·tau~`, `tau~ = -ln(u)` unitless). When `den = c - e² = 0` (one active NLOS cluster, or equal `u`, in
an NLOS scenario or with K = 0) the fallback 1 gives delays of `-ln(u)` seconds. The spec (3.6 step 6) prescribes the
1, so this is a spec erratum carried into the code.

Evidence (measured):
- `bug1.py`: RMa NLOS (`NumClusters_min` absent, so `n_cluster = 1` is valid), library LSPs and fields, 3.5 GHz,
  `n_cluster = [1, 1]`: DS = 91.9 ns, 83.2 ns. `"38.901"`: delay of the active cluster 0 s, 0 s (expected).
  `"quadriga"`: **0.6226 s, 1.0822 s**.
- `bug1c.py`: RMa LOS with `lsp` K = 0 (valid, K >= 0), `n_cluster = 1`: `"quadriga"` **0.0179 s, 0.806 s**.
- `probe1.cpp`: `NumClusters = 2` (one NLOS cluster for every link, as the mmMAGIC legacy file) in an NLOS scenario:
  `"quadriga"` 0.693 s, 2.996 s; `"38.901"` 0, 0.
- `probe2.cpp` fuzz (section C): largest delay of all outputs **708.4 s** (`quadriga`, UMi NLOS, `u = 0`, one
  active cluster: `-ln(DBL_MIN)` seconds).

Expected: delay 0 for the active cluster(s) (no spread to scale; a single cluster's delay spread is 0 for any scale),
then the abs-TOA shift where it applies. Fix: fallback `0.0` instead of `1.0` at line 253 (and in spec 3.6 step 6).
With M1's fix the NLOS case is closed independently of the fallback; the LOS-with-K = 0 case needs the fallback fix.
Regression test (in the `quadriga` file): RMa NLOS and RMa LOS with K = 0, `n_cluster = 1`: active delay 0 (NLOS
with `abs_toa`: `abs_toa`).

## B. Major

### M1. `quadriga` in NLOS without absolute TOA: the delay origin `s·min(tau~)` is ill-conditioned for few clusters

`src/ssf_clusters.cpp:311` shifts the NLOS delays to `abs_toa` only `if toa > 0`; otherwise the first NLOS cluster
sits at `s_tau·min(tau~)` with `s_tau = DS/sqrt(var(tau~))`, which grows without bound when the few `tau~` are
close. Measured (`bug1b.py`, RMa NLOS, 2000 independent links, `SC_lambda = 0`, `abs_toa` omitted), first NLOS
delay / DS:

| `n_cluster` | median | p90 | max |
|---|---|---|---|
| 2 | 1.33 | 10.1 | 1.06e5 |
| 3 | 0.59 | 2.71 | 84.5 |
| 5 | 0.29 | 1.05 | 5.55 |
| 10 | 0.135 | 0.44 | 2.12 |

Affected: NLOS scenarios without abs TOA (legacy files with `absTOA_mu <= -30`, where `lsp_generate` returns 0; the
mmMAGIC file with N = 2) and small `n_cluster`. v19 NLOS files are not affected in practice (`lsp_generate` gives
Δτ > 0). Fix (maintainer decision, it changes the spec rule "shift only where Δτ > 0"): in NLOS scenarios always shift
the active delays so that the first is at `abs_toa(l)` (0 if omitted), i.e. drop `&& toa > 0.0` at line 311 and pass
`toa = 0` when `abs_toa` is omitted. The delay spread is unchanged (no LOS weight in NLOS); B1 then cannot occur in
NLOS. The NG difference "abs TOA shift only where Δτ > 0" (spec 10.3) becomes "NLOS delays always start at Δτ".

### M2. Documentation bloat (the maintainer's main concern)

Measured block sizes (lines between `/*!MD` and `MD!*/`):

| Block | C++ | MEX | Python |
|---|---|---|---|
| `ssf_init_phases` | 65 | 70 | 61 |
| `ssf_draw` | 84 | 86 | 81 |
| `ssf_clusters` | 272 | 283 | 246 |

Sum 1248 lines, plus 9 new SECTION_DESC lines in each of three files and 26 comment lines in the header. For
comparison: `lsp_generate` 124, `path_loss` 124, `lsf_draw` 93. Causes: implementation formulas of all three methods
in the `ssf_clusters` block (≈ 90 lines), measured study numbers (the 9-row table, the property bullets; unpublished,
cannot be cited), repetition (the random-value layout in four places: header, file comment, `ssf_init_phases`,
`ssf_draw`, plus the block list again in `ssf_clusters`; the input-use table and the argument list; GR power rule
three times; the SC rule of `38.901` in `ssf_draw` and `ssf_clusters`), the SplitMix64 formula, mobility/floor/
precision rules copied from `lsf_draw`, validation-order lines, and MEX/Python blocks that copy the C++ text.

Proposal (full texts in the appendix): C++ 65 → 41, 84 → 53, 272 → 98; MEX 70 → 36, 86 → 45, 283 → ~95; Python
61 → ~38, 81 → ~50, 246 → ~95; sum 1248 → ~551. SECTION_DESC paragraph 9 → 5 lines (×3). Header SSF comments 26 → 8.
Facts kept and where (each once) and facts moved to source comments: section D.

### M3. Test bloat: duplicated oracles and fixtures

- `tests/catch2_tests/test_ssf_phases.cpp:115–195` copies `ref_sos`, `preset_tab`/`get_preset`, `acf_at`,
  `floor_offset`, `grid`, `corr`, `equal` from `test_gbsm_phases.cpp:129–215` (≈ 80 lines; `diff` shows identical
  bodies except the grid offset from the axes and `stddev` instead of `var`). Fix: one
  private test header `tests/catch2_tests/qd_test_sos_oracle.hpp` (precedent: `qd_test_diag.hpp`) included by both
  files; −80 lines.
- The three `ssf_clusters` files repeat the fixture prologue: `read_v19`, `wrap_rad`, `zen2elev`, `fix_t`/`make_fix`,
  `out_t`/`run`, `eq`/`equal`, `los_dir`, `ref_r2`, `slots`, `check_finite` (first test case at line 412, 316, 430;
  ≈ 120–150 shared lines per extra file). The split into three files served parallel implementation runs only. Fix:
  a shared header `qd_test_ssf.hpp` with one fixture (F frequencies, anchor LSPs, all optional inputs) and
  `run(sp, f, method)`, or merge `_mf` and `_quadriga` into `test_ssf_clusters.cpp`; −250 to −300 lines.
- Per-method copies of the same assertion: "one cluster gives the LOS path only" (38 + 41 + 38 lines), "extreme
  finite random values" (35 + 36 + 29), "accepts valid inputs and ignores the inputs it does not use" (13 + 14), "zero
  LOS power in NLOS scenarios" (12 + 14): one case each, looping over the three methods; −≈140 lines.
- `ssf_clusters - input validation` (263 lines) re-tests the messages of `qd_gbsm_check_links` (section
  "positions") and `scenpar_validate` (three messages), which their own suites pin; keep one pass-through call each;
  the remaining sections can be table-driven (input mutation, expected message); −≈100 lines.
- Target: cluster tests 3119 → ≈ 2300 lines, phases tests 844 → ≈ 760; no coverage lost.

## C. Minor

- **m1. UBSan report** `src/ssf_clusters.cpp:698`: `arma::any(rx_pos.col(2) < 0.0) || arma::any(tx_pos.col(2) < 0.0)`
  on a 0-row `rx_pos` (n_link = 0 with `GR_enabled`) binds a reference to a null pointer inside Armadillo
  (`subview_meat.hpp:1413`, `access.hpp:26`; reached from `test_ssf_clusters.cpp:1022`). Harmless in practice, UB by
  the standard. Fix: `if (n_link > 0 && (...))`, or a row loop.
- **m2. Reference to an unpublished source** `tests/catch2_tests/test_ssf_clusters.cpp:34` and `:875`: "medians of
  the output/input spread ratios of a comparison study of the method", "medians of the comparison study". The study
  is a working document. Fix: state the window itself ("expected medians [0.94, 1.00] … ± 0.03; ±0.03 is > 5
  standard errors of the median for 2000 independent links").
- **m3.** `ssf_clusters.cpp:778`: `centre[4]` is computed for `"quadriga"` too, which does not use it; move it into
  the TR and MF branches.
- **m4.** `qd_ssf_tr` (`ssf_clusters.cpp:94`) computes `rho = ln(Pa/max Pa)` four times (inside the angle-type loop)
  and `std::pow(10.0, -2.5)` per cluster (line 76); compute `rho[n]` once after the power loop (one value per cluster,
  as eqs. (7.5-9), (7.5-14) read) and the threshold as a constant.
- **m5.** `qd_ssf_quadriga` (`ssf_clusters.cpp:259–262`) writes the GR weight and offset into `wt[1]`, `A[1]` also
  without GR and relies on the cluster loop to overwrite them; write the GR entry only `if (gr)`.
- **m6.** `ssf_clusters.cpp:20, 119`: `qd_pi` duplicates `arma::datum::pi` (used in `lsp_generate.cpp:359`), the
  `wrap180` lambda duplicates `qd_wrap` in degrees. Nit; use `arma::datum::pi`, keep `wrap180` or document why.
- **m7.** `api_python/qpy_ssf_clusters.cpp` (gr_eps conversion): a 3-D `gr_eps` is flattened to `[n, m·s]`, so an
  `(n_link, n_freq, 1)` array passes silently (by reading). Reject `ndim > 2`.
- **m8.** The non-negative-integer conversion loop (`n_cluster`, `floor_id`) is now in `api_mex/ssf_clusters.cpp`,
  `api_mex/ssf_draw.cpp`, `api_mex/lsf_draw.cpp` and the three Python counterparts (≈ 10 lines each). Optional: one
  helper in the MEX and Python helper headers.

## C'. Checks without finding

- **TR fidelity**, read equation by equation against TR 38.901 v19.5.0: `38.901` — (7.5-1) with the min subtraction
  of (7.5-2) (no sort, documented), (7.5-3)/(7.5-4) with K ≥ −5 dB (documented deviation), (7.5-5)/(7.5-6) with
  Z_n = ζ·z, (7.5-7)/(7.5-8) powers used only in (7.5-9)/(7.5-14), −25 dB removal against the (7.5-6) powers without
  renormalisation, (7.5-9)/(7.5-10) and (7.5-14)/(7.5-15) with the Tables 7.5-2/7.5-4 interpolated, X_n, Y_n with
  AS/7 per angle type, (7.5-11)/(7.5-12), (7.5-16)/(7.5-17) incl. O2I 90°, (7.5-19) with μ_offset,ZOD = −EoD offset
  (sign verified against `lsp_generate.cpp:354–367` and the conf comment "TR value with opposite sign"), ZOD in LOS by
  (7.5-17), zenith fold; `38.901_MF` — step 5' incl. the arg() principal value, the wrapped Laplacian through the
  erfc identity (exact), LOS cluster 1 at 0, (7.6.30a–h), (7.6.30i–m) without ray terms, wrap before √(1+K);
  7.6.8 — (7.6-32) power factor (d3D/d_GR)², (7.6-33/34) delay difference in cancellation-free form, (7.6-37a–d),
  (7.6-38/39/40) with ε = re − j·im; 7.6.9 — Δτ in NLOS only (7.6-43 vs 7.6-44); 7.6.15 — per-link N in the C
  factors. `quadriga`: matches spec 3.6 (except B1/M1). No deviation beyond the documented ones.
- **Fuzz** (`probe2.cpp`): 4 files (UMi LOS/NLOS, InF LOS, RMa NLOS) × 3 methods × GR off/on × 20 draws × 50 links,
  K −30…40 dB, spreads 0.01–200°, DS 0.3 ns–3 µs, `u` ∈ {0, 1} on 4 % of entries, Normal values ±40 on 5 %,
  co-located Tx/Rx, per-link `n_cluster`, O2I, 3 frequencies (MF, quadriga): 720,000 output elements, all finite,
  azimuths in [−π, π) (none equal to π; the wrap was also probed at the −π boundary), elevations in [−π/2, π/2],
  powers ≥ 0. Only anomaly: B1 (max delay 708 s).
- **Edge cases**: `NumClusters = 1`, zero links, per-link N, GR on/off, NLOS + GR, K = 0 in LOS (C factors at −5 dB),
  C factors positive for all K ≥ −5 dB (C_τ ≥ 0.078 at 25 dB). OK apart from B1/M1/m1.
- **OpenMP**: per-link state only, inputs read-only, signed `long long` loop variable (as 29 other loops in `src/`),
  no exception inside the region (all checks before). OK.
- **MSVC**: no VLAs, `__builtin_*`, designated initialisers, `M_PI`, `std::to_chars`; `INFINITY`/`NAN` from
  `<cmath>`; `std::complex<double> == double` is standard. OK. No BLAS in the tests (no matrix products, no `norm`).
- **Scoping and API**: kernels > 20 lines file-local above the doc block, single-use helpers as lambdas
  (`derive_seed`, `spreads_positive`, `efold`, `spread`), `qd_gbsm_gen_ssf` shared by two functions in one file, users
  list of `qd_gbsm_check_links` updated. Three public functions, no new key, no new file type. OK.
- **References**: no `md/` path, D/A/N/W/P label, run name or spec section in any produced file; the only finding is
  m2.
- **Wrappers**: MEX `set_id` shift, `n_cluster` count without shift, zero links rejected (documented); Python flags
  via `qd_python_flag`, `gr_eps` complex/real, `None` handling. OK apart from m7/m8.
- **ASan + UBSan**: `test_ssf_asan` (SSF and v19 conf cases), `OMP_NUM_THREADS=8`: **All tests passed (31574
  assertions in 71 test cases)**; 0 AddressSanitizer errors, 0 leaks; 2 UBSan reports, both m1.

---------------------------------------------------------------------------------------------------------------------

## D. Documentation: what stays, what moves, what goes

### D.1 Facts that must survive (each in one place)

| Fact | Place |
|---|---|
| Two set types, `n_gen`, `set_id` meaning, frequency index = caller's list order | `ssf_init_phases` |
| Seed derived from the LSF seed and `set_id`; platform-independent; −1 random; own LSF seed per scenario (TR 7.6.3.4) | `ssf_init_phases` |
| Column layout `b·N + n`, block meanings, distributions | `ssf_draw` (only) |
| Mirror/swap rule of the blocks (D2D reciprocity) | `ssf_draw` (only) |
| `SC_lambda` for all generators, 0 → 0.1 m | `ssf_draw` |
| Mobility, floors, 2D/3D, precision | by reference to `lsf_draw` |
| Append-only promise | `ssf_draw`, one clause |
| Methods, one bullet each with clause/paper and use | `ssf_clusters` |
| SC rule of `38.901` (freeze blocks 1–8 per segment) | `ssf_clusters` (only) |
| Slot layout, `n_slot`, slot ↔ column identity, inactive slots | `ssf_clusters` |
| LOS condition `KF_mu > −30`, NLOS → zero LOS/GR power, legacy-file consequence | `ssf_clusters` |
| Units and conventions, coefficient powers, no PL/SF | `ssf_clusters` |
| Abs TOA rule per method | `ssf_clusters` |
| GR: plane z = 0, power rule per method, how to form the coefficient (polarised and scalar) | `ssf_clusters` |
| `NumClusters = 1`; swap-reciprocity conditions; same-scenario inputs | `ssf_clusters` |
| Ranges, sizes, "required by", "used by" | argument lists |

### D.2 Facts that leave the MD blocks

Into source comments, next to the code (`src/ssf_clusters.cpp` unless stated):
- `38.901` formulas: already in the kernel comments with equation numbers (lines 44–104). Add at line 53 the reason
  of the K limit (C_θ of (7.5-15) falls to 0 near −10 dB and turns negative; far below −5 dB the ASA is ≈ 0.86 of
  the input) and at line 76 "removed clusters keep delay and angles and stay in max and ρ".
- `38.901_MF` TR-literal points (no min subtraction, sort, removal, C factors, μ_offset,ZOD, O2I rule; Q_n per
  frequency; zenith fold applied to the final angles because 7.6.5.1 states none; wrap-before-scale jumps in LOS):
  into the kernel header comment, line 107.
- `quadriga` algorithm summary and its changes against the paper/NG (additive LOS direction, EoD offset in NLOS, no
  satellite EoA fade, ω = 0 in NLOS, no fixed angles for one or two clusters) and the properties (DS exact without GR;
  with GR the scaling uses the split K/(K+1)·(1 − R², R²), so the DS from the slot powers is ≈ 0.85 of the input;
  AS exact on the linear definition below the caps; averaged scales with several frequencies): kernel header comment,
  line 173.
- GR geometry details (d_GR' = max(d_GR, 1e-5), zero-denominator rule at grazing with ε = 1, cancellation-free τ_GR,
  cos θ / sin² θ): the block at line 746, which already has most of it.
- SplitMix64 formula and tags: file comment of `src/qd_gbsm_phases.cpp` (appendix 12), the only place.
- Validation order lines: deleted (messages name the argument; ranges are in the argument lists).
- Measured study numbers (method × property table, ratio medians, jump rates, Spearman values): deleted from the
  docs; they stay in the working documents. The plan's "method × property table in the docs" is replaced by the
  qualitative clauses of the method bullets; a 7-line qualitative table is offered in the appendix if the maintainer
  wants to keep a table.
- `ssf_draw` dual-mobility formulas, floor offsets, precision bound: deleted (they are `lsf_draw`'s, referenced).
- "Outputs are resized only if their size differs", "rnd may share memory with tx_pos": deleted (general library
  behaviour). "The permittivity fit of `lsp_generate` falls below 1 above ≈ 1.3 THz": deleted (the range ≥ 1 stays).

### D.3 Header and file comments

`include/quadriga_gbsm.hpp:183–236`: the 14-line SSF index and the "see the documentation" lines go (appendix 11);
the trailing per-argument comments stay. `src/qd_gbsm_phases.cpp:34–66`: keep the generator index (it documents
`qd_gbsm_gen_ssf`), make it the single home of the seed rule, drop the paragraph on the machinery (appendix 12).

### D.4 SECTION_DESC

First paragraph: keep "steps 1 to 7", drop the repeated output list ("and the delays, powers and angles of the
clusters" → "and the clusters"). New paragraph 9 → 5 lines (appendix 10), the same in the three files.

---------------------------------------------------------------------------------------------------------------------

## E. NG integration readiness (spec section 10)

The outputs give NG what steps 5–7 of `gen_ssf_parameters` produce (`taus`, `pow`, `AoD`, `AoA`, `EoD`, `EoA`, per
frequency where `38.901_MF` needs it, `n_active`). Gaps:
- GR consumer contract: the consumer needs R∥ (its phase) and R = √R², which the library computes but does not return;
  NG recomputes them (spec 10.1 keeps NG's own R with the conjugate ε sign). Two implementations of (7.6-38/39) must
  then agree on the geometry (the d_GR' floor). Acceptable for this series; the ray step could return R∥, R⊥.
- `38.901` under spatial consistency: the caller splices blocks 1–8 of a segment-start `ssf_draw` call into the
  current `rnd`; no helper, one sentence in the docs (kept in the proposed text).
- M1 matters for NG's legacy NLOS files without abs TOA and the mmMAGIC file (N = 2); decide before integration.

## F. Verdict

A second round is needed, short and targeted: the documentation rewrite (nine blocks, SECTION_DESC, header) is the
largest change after this review and the maintainer's main concern, the test restructuring touches four files, and
B1/M1 change `quadriga` delays (spec text 3.6 steps 6 and 10 changes with them). The method kernels, the validation
and the wrappers need no re-review beyond the B1/M1 diff and m1–m7.
