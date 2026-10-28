# SSF port, series 1 (cluster level): final plan

Brief for the manager session and the spec writer of the first SSF series. Plan level, not a spec. Working
document: deleted before the merge; only a hand-off survives. Background, TR analysis, NG analysis and the decision
log: `md/ssf_analysis.md` (IDs A = TR ambiguity, N = NG finding). Literature: `md/ssf_literature.md`.

Branch `ssf` in `/home/stephan/quadriga-lib-ssf`, from v0.12.4 (`26da9f93`); local only, never pushed. The user
commits everything; commit messages carry no internal IDs and no `md/` references.

## 1. Goal and scope

Goal: the cluster level of TR 38.901 v19.5.0 clause 7.5 (steps 5–7) and the cluster-level optional components in
quadriga-lib, so that QuaDRiGa-NG replaces steps 5–7 of `gen_ssf_parameters` by library calls; a step towards a
complete 38.901 implementation in the library.

In scope:
- Steps 5–7: cluster delays, powers, angles, incl. C_τ, C_φ/C_θ, −25 dB rule, μ_offset,ZOD, O2I ZOA, LOS handling.
- Three methods for steps 5–7: `38.901` (7.5), `38.901_MF` (7.6.5.1), and the consistent method `quadriga`
  (redesign, D3). Procedure B (7.6.3.2) dropped after R0 (D1).
- Spatial consistency of the cluster random variables (7.6.3.1), multi-frequency (7.6.5, 7.6.5.1), dual mobility /
  D2D reciprocity, explicit ground reflection (7.6.8), absolute time of arrival (7.6.9), cluster-number variability
  (7.6.15).

Out of scope (later series; this series leaves the interface ready):
- Ray level: ray offsets, sub-clusters, coupling, XPR, initial phases, 7.6.2 large bandwidth, 7.6.10 mobile
  scatterers, 7.6.13 near field, 7.6.14 SNS, 7.6.16 polarisation variability. Needs its own analysis (time variation
  of rays, drifting, storing physical paths in QRT files before the antennas).
- Time evolution: Procedure A, drifting, scatterer geometry (`gen_fbs_lbs`).
- Calibration against 3GPP (needs antennas and the ray step).

## 2. Decisions (binding for the spec)

- **D1 Methods.** `38.901`, `38.901_MF`, `quadriga` (section 4.2). Default `38.901` (decided after R0,
  `md/ssf_r0_report.md`); revisit after the ray series with antenna-level metrics. Procedure B (`38.901_B`) dropped
  after R0: as written it gives 0.47–0.74 (NLOS) and 0.14–0.19 (LOS) of the input spreads with 1–2 dominant
  clusters; it can be added later (append-only random-field blocks keep room for it). 7.6.5.1 stays TR-literal
  (order "wrap, then scale by √(1+K)" and its LOS jumps stated in the spec).
- **D2 Feature combinations.** Mirrored random fields in all methods (D2D reciprocity of the random inputs);
  `38.901` single frequency only (error otherwise: the TR requires frequency-independent delays and angles, 7.5
  cannot give them); spatial consistency allowed with `38.901`, its jumps documented; TR BS-specific terms
  (μ_offset,ZOD, cluster ZSD, O2I ZOA = 90°) kept as written and documented; a method × property table in the docs
  (continuity, frequency-independent delays/angles, exact spreads, swap reciprocity, per-cluster shadowing).
- **D3 Consistent method.** Redesign only, named `quadriga` (decided after R0): NG's QuaDRiGa cluster generation
  (delays, power product with the paper's g fits, analytic DS and iterative AS scaling averaged over frequencies,
  caps 3 / 1.5, NG GR power split, no ζ) with three changes: LOS direction applied additively (azimuth wrap,
  elevation pole fold) instead of the 3-D rotation; μ_offset,ZOD added to the NLOS departure elevations; no satellite
  EoA fade. Not adopted: per-cluster ζ, Gaussian/Laplacian initial values, exact wrapped AS solve, exact
  per-frequency scaling. No faithful port, no NG parity test; NG output changes documented (R0 report 4.2).
- **D4 Cluster count.** `NumClusters = N + 1` in every file: slot 0 = LOS ray (zero power in NLOS), then N NLOS
  clusters; in TR LOS the first NLOS cluster is at τ = 0, aligned to LOS (7.5-12/-17). Change: +1 in `NumClusters`
  and `NumClusters_min` of the 8 v19 LOS files, comments updated. NG's old conf files keep their legacy meaning.
- **D5 TR gaps.** C_φ/C_θ by linear interpolation between table entries (exact at table points, clamped outside);
  −25 dB removal sets the power to 0, keeps the slot, no renormalisation, only in `38.901`; C_τ applied in
  `38.901` LOS; signs X_n and jitter Y_n spatially consistent, evaluated by the caller at the drop / segment-start
  positions and kept along the track (documented, like hE in the LSF); zenith wrap [180°, 360°] → 360° − θ for ZOA
  and ZOD.
- **D6 Random fields.** Blocks by variable type, append-only (existing columns never move), ray blocks reserved;
  one shared cluster set plus one set per frequency (Z_n and later the per-frequency ray blocks), seeds derived from
  the LSF seed by a fixed documented rule; no bias removal; 2D + floor IDs as the LSF, 3D legacy mode, `SC_lambda =
  0` → 0.1 m.
- **D7 Output.** Padded slots per link (0 LOS, 1 GR if enabled, then N), zero power for inactive entries (removal,
  smaller per-link N); delays in s relative to the LOS delay, NLOS + Δτ when abs TOA is on, frequency dimension only
  where a method needs it (7.6.5.1 LOS); powers per frequency as the coefficients use them (7.5-30: LOS K/(K+1),
  NLOS P_n/(K+1)); AoD, EoD, AoA, EoA in rad, GCS, elevation convention; active-cluster count per link. Inputs of the
  later ray step (cluster spreads, c_DS(f), per-link cluster ZSD, K) are read from scenpar / `lsp`, not duplicated.
- **D8 Scope and runs** as in sections 1 and 6.

## 3. Rules for the series (lessons of the LSF series)

Content and design:
- **TR first.** TR behaviour is the default of each TR method; legacy QuaDRiGa behaviour through flags or the
  `quadriga` method. NG is not authoritative (never bug-scanned): where NG conflicts with the TR, follow the TR
  unless NG gives correct statistics for a good reason; state each such case in the spec. Known NG findings to
  respect: N1 (no C_τ), N2 (baseline angles not the TR mapping), N3 (LOS cluster count), N4 (ZOD offset never
  applied), N5 (no −25 dB rule), N12/N13 (SSF fields 3D, seed not stored).
- **Deprecated:** TR 36.873 and mmMAGIC (no `mmMAGIC` sub-path method).
- **Random fields separate from deterministic models.** Every model function takes normalized values (U(0,1) or
  N(0,1)) and is deterministic; the fields come from the phase machinery. This makes exact test oracles possible.
- **Reuse** `src/qd_gbsm_phases.cpp` (generator table `qd_gbsm_gen`, `qd_gbsm_freq`, `qd_gbsm_positions`,
  `qd_gbsm_eval`, floors, mirror combination) and the shared helpers; no parallel machinery. The SSF init/draw
  functions go into that file (its header states the seed rule and the design for further sets).
- **Conf format fixed.** No new file types, no new string keys; new numeric keys allowed (none expected: all TR
  table values exist). Group-level values only in files where they are unique. Values readable: source precision,
  derived values with at most 4 significant digits. Simulation settings (anchor/reference frequency, …) are
  arguments, not conf keys. TR constants (Tables 7.5-2/-4, C_τ, r = 1.5, 50 m) live in the code.
- **`double` only.** SOS evaluation runs in single precision on AVX2: tolerances for random fields about 1e-5.

Code:
- Lean code and a lean public API that fits `html_docu/quadriga_lib_api_cpp.md`; the binding coding rules are
  `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`.
- Helper scoping: a helper used once is a lambda, except a self-contained algorithm over ~20 lines (named
  file-local function above its caller); helpers reused in one file are `static inline` or in an anonymous
  namespace; only cross-file helpers go into a documented private header (`src/quadriga_gbsm_helpers.hpp`).
- Related functions may share one cpp file; fewer, coherent files.
- No BLAS/LAPACK, also in tests (no Armadillo matrix products above 4×4). MSVC must compile. No `std::to_chars` for
  floating point; objdump-gate the MEX files (`GLIBCXX` ≤ 3.4.28).
- Python flags: `None` = default, a non-bool raises `TypeError` (`qd_python_flag`). MEX wrappers reject zero links;
  C++ and Python accept them.

Tests:
- Library calls for supporting tasks (geometry, interpolation, spreads), never re-implementations.
- Oracles are TR formulas typed into the test, independent of the conf mapping (tests code and conf files
  together). Tolerances from the arithmetic; TR-oracle points inside the TR applicability ranges.
- Random fields: marginals, ACF at λ, floor independence, swap reciprocity, layout stability (appending a block
  leaves existing columns bit-identical).

Documents:
- No produced file (code, tests, wrappers, conf files) references temporary `md/` documents or tracking IDs; only
  published material and the persistent `md/SPEC_*.md` may be cited.
- The spec stands on its own. Working documents are deleted before the merge; a hand-off survives (SSF section of
  the NG integration hand-off).

Process:
- Runs sized for one agent each (section 6). Effort per agent: manager `medium`; R0, spec writer, implementation
  runs R1–R3, spec review rounds 1–2 and code review rounds 1–2 `high`; the final (third) spec review round and the
  final code reviewers (code, TR compliance, NG readiness) `xhigh`; `max` never (fast pass, then thorough pass).
- The machine is shared: check `cat /proc/loadavg`; at most 12 threads, one build or test suite at a time.
- Separate final reviewers: code, TR compliance, NG integration readiness. Every bug claim needs a concrete input
  with wrong vs expected output. ASan/UBSan runs, never unbounded allocation probes under ASan.
- Decision style towards the user: what is decided, options with trade-offs and implications, what would be
  correct, recommendation.

## 4. Function set (plan level)

### 4.1 Random fields (`src/qd_gbsm_phases.cpp`)

- **`ssf_init_phases`** — phases of an SSF generator set of one BS site and scenario, [n_coeff, n_gen, 2], slice 1
  the mirror of slice 0 (as `lsf_init_phases`). Set type: shared cluster set or per-frequency set (frequency index).
  Generator table from scenpar (N_max from `NumClusters`, λ = `SC_lambda`). Seed derived from the LSF seed.
- **`ssf_draw`** — normalized values [n_link, n_gen] at the links (2D/3D, floors, dual mobility), as `lsf_draw`.

Cluster blocks (each N_max columns): delay, shadowing Z_n (per-frequency set), sign ×4, jitter / offset ×4 (method
dependent: Y_n, 7.6.5.1's Y/Z/V/W, `quadriga`'s uniforms; the spec fixes one layout that serves all methods or one
table per method), then reserved ray blocks. Mirror pairs: delay, Z_n self; AOD↔AOA and ZOD↔ZOA for signs and
offsets. The spec states the column layout as part of the public contract.

### 4.2 Deterministic model (new source file, e.g. `src/ssf_clusters.cpp`)

- **`ssf_clusters`** — steps 5–7 per link with `method` ∈ {`38.901` (default), `38.901_MF`, `quadriga`}. Inputs: positions; `lsp` of `lsp_generate` (+ anchor-frequency `lsp` for `38.901_MF`); `lsp_mu`
  (μ_lgZSD); `eod_offset_deg`; `abs_toa`; ground permittivity (GR); O2I flag per link; optional N per link (7.6.15);
  `rnd` from `ssf_draw` (shared + per-frequency sets). Output as D7.
- Optional small helpers only if a second user exists (e.g. the C_φ/C_θ interpolation as a documented function is
  not needed in the public API).

Public API: 3 functions, each with MEX and Python wrappers. No converter to `get_channels_*` in this series (the
ray step comes first).

## 5. R0: comparison study (first run, before the spec)

Purpose: measured, not guessed, trade-offs to decide the default method and D3. Scratch prototype (Python/NumPy or
MATLAB, not committed, in the scratchpad or `md/` scratch), cluster level only, all candidates fed with the same LSP
draws (from the library: `lsf_draw` + `lsp_generate` on v19 files, UMi and UMa LOS/NLOS at 6 and 30 GHz, plus one
multi-frequency case 2/6/30 GHz) and the same random inputs.

Candidates: `38.901` (with D5 rules), Procedure B, 7.6.5.1, faithful `quadriga` (NG's cluster generation: frequency-
averaged DS/AS scaling, iterative AS loop, satellite EoA fade, GR power split, no ZOD offset), and a redesigned
consistent method; redesign elements switchable one by one (per-cluster shadowing ζ; Gaussian/Laplacian initial
values mapped from Normal fields; ZOD offset as a rotation; TR GR; per-frequency exact scaling).

Metrics:
- spread fidelity: output DS, ASD, ASA, ESD, ESA per link against the LSF input (CDFs, bias, the K-dependent caps,
  ESD inflation);
- cluster power profile: sorted powers, clusters within 10/20/25 dB, effect of ζ;
- delay–power and angle–power relation, marginal angle distributions;
- continuity along a track and between nearby UTs (jumps per metre), frequency stability of delays and angles;
- D2D swap reciprocity;
- faithful vs redesign: distance from NG behaviour (how much NG's default output changes at integration), distance
  from the `38.901` statistics, ablation per redesign element, cost (lines, run time per link).

Deliverable: a short report in `md/` with figures/tables and a recommendation for the default and for D3; the user
decides before the spec is written. Risk: no consistent method reaches the baseline statistics → default stays
`38.901`, the consistent path stays an option.

## 6. Work breakdown

| Run | Content | Code + docs / tests (lines) |
|---|---|---|
| R0 | Comparison study (section 5) | prototype |
| — | Spec, three spec review rounds | — |
| R1 | `ssf_init_phases`, `ssf_draw`: tables, layout, mirror pairs, derived seeds; tests | 300 / 600 |
| R2a | `ssf_clusters` shared part + `38.901`: C factors, C_τ, −25 dB, ZOD offset, O2I ZOA, abs TOA, GR, per-link N; TR-oracle tests | 650 / 1,200 |
| R2b | `38.901_MF`; tests | 150 / 350 |
| R2c | `quadriga` (redesign, D3); property tests (exact spreads where promised, continuity, reciprocity); documented differences to NG's current output | 350 / 600 |
| R3 | MEX + Python wrappers, docs (`html_docu`), objdump gate, D4 conf change | 600 / 400 |
| — | Three code review rounds (separate reviewers: code, TR compliance, NG readiness) | — |
| — | NG integration: steps 5–7 of `gen_ssf_parameters` by library calls; store seed and SSF phase cubes; NG's MATLAB ray code stays | — |

Total about 2,100 lines of code and docs, 3,150 lines of tests (≈ 60 % of the LSF series). Order: R1 → R2a
gives a usable TR baseline; R2b/R2c independent of each other after R2a.

Risks:
- Random-field layout must already serve the ray series and Procedure A (reserved, append-only blocks).
- NG integration with the redesigned `quadriga` changes NG's default output (measured in R0: ESD, arrival and
  departure centres, D4).
- Open TR points (A1–A16 of the analysis) are resolved by policy; each must be stated in the spec.

## 7. NG integration notes (for the later hand-off)

- Replace delays/powers/angles of `gen_ssf_parameters` (both paths) by `ssf_clusters`; keep `xprmat`, `pin`,
  `subpath_coupling`, `get_subpath_angles`, `gen_fbs_lbs`, `get_channels` in MATLAB until the ray series.
- Builder stores the LSF seed (today drawn but not stored) and the SSF phase cubes instead of `path_sos`;
  `xpr_sos`, `pin_sos`, `clst_dl_sos` stay until the ray series (`clst_dl_sos` only for the deprecated mmMAGIC).
- Signs/jitter: evaluate at segment starts and reuse along the segment (D5).
- The D4 change affects NG users of the v19 LOS files (one more NLOS cluster); NG's legacy files unchanged.
- No parity test against NG's QuaDRiGa path (D3: redesign only); the measured output changes are documented instead.
