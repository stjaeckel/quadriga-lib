# Spec: small-scale fading, series 1 (cluster level, TR 38.901 v19.5.0 clause 7.5 steps 5–7)

Status: revised after review rounds 1 and 2 (2026-10-28; `md/ssf_spec_review_r1.md`, `md/ssf_spec_review_r2.md`). Branch `ssf` of `/home/stephan/quadriga-lib-ssf`, base 26da9f93 (v0.12.4).
Working document of the series (deleted before the merge, like `md/ssf_plan.md`, `md/ssf_analysis.md` and the R0
report). Labels: D = decision of the plan, A = TR ambiguity, N = QuaDRiGa-NG finding, W = choice of this spec that the
plan did not fix (listed in 1.4 for the maintainer).

This spec is self-contained. With the quadriga-lib sources, TR 38.901 v19.5.0 and the 2019 paper it cites
(Jaeckel, Raschkowski, Burkhardt, Thiele, "A Spatially Consistent Geometric D2D Small-Scale Fading Model for
Multiple Frequencies", "the paper" below; equation numbers of its published version), it contains everything the
implementation runs need. QuaDRiGa-NG ("NG", HEAD 9a2beeb) is quoted for the differences only; it is not a
reference to reproduce.

No D/A/N/W label, no section number of this spec, no `md/` path, no run name (R1 … R3), no local path and no NG
line number may appear in any produced source, header, test, wrapper, conf file or documentation block (9.1). Cite TR
clauses, tables and equations, the paper, or state the rule itself.

Process: the review points A, B and C of `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md`
are not stops for a human. The implementation agent decides within this spec, never against it, and records every
decision and deviation in its run report (8). Where this spec and a SPEC file disagree, this spec wins; the known
differences are in 1.6.

Contents
1. Scope and decisions
2. Public API and the random-field layout
3. Normative behaviour (shared definitions, the three functions, the three methods)
4. Conf change (D4)
5. Catch2 tests
6. MEX and Python wrappers
7. Documentation
8. Implementation runs R1, R2a, R2b, R2c, R3
9. Constraints (into every run)
10. NG integration hand-off (out of scope for the runs)


## 1. Scope and decisions

### 1.1 Goal and data flow

Goal: the cluster level of TR 38.901 v19.5.0 clause 7.5, steps 5 to 7 (cluster delays, powers, angles), with the
cluster-level optional components (7.6.3.1 spatial consistency, 7.6.5 and 7.6.5.1 multi-frequency, dual mobility
and D2D reciprocity, 7.6.8 ground reflection, 7.6.9 absolute time of arrival, 7.6.15 cluster-number variability), so
that NG can replace steps 5 to 7 of its `gen_ssf_parameters` by library calls. Three new public functions, each with
MEX and Python wrappers:

| Function | File | Purpose |
|---|---|---|
| `ssf_init_phases` | `src/qd_gbsm_phases.cpp` | phases of one SSF generator set of a BS site and scenario |
| `ssf_draw` | `src/qd_gbsm_phases.cpp` | normalized random values of a set at the links |
| `ssf_clusters` | `src/ssf_clusters.cpp` (new) | steps 5–7 per link, methods `38.901`, `38.901_MF`, `quadriga` |

Data flow per BS site and scenario (after the LSF step of the existing functions):

```
conf_file_read -> lsf_init_phases(seed) -> lsf_draw -> path_loss -> lsp_generate (lsp, abs_toa, gr_eps, eod_offset)
ssf_init_phases(scenpar, seed, set_id = -1)        -> phase_cl        (once per site and scenario)
ssf_init_phases(scenpar, seed, set_id = k)         -> phase_f[k]      (once per frequency k = 0 .. n_freq-1)
ssf_draw(phase_cl, set_id = -1) -> rnd                 [n_link, 9·N]  (at the links)
ssf_draw(phase_f[k], set_id = k) -> rnd_freq(:, :, k)  [n_link, N]    (at the links, per frequency)
ssf_clusters(scenpar, tx, rx, lsp, rnd, method, rnd_freq, ...) -> delay, power, aod, eod, aoa, eoa, n_active
```

`seed` is the seed of the LSF set of the same site and scenario; the SSF sets derive their seeds from it (2.2.4).
Every model function is deterministic in its normalized inputs; randomness enters only through the seed.

### 1.2 Decisions (binding, carried over from the plan)

| ID | Decision | Where |
|---|---|---|
| D1 | Methods `38.901` (default), `38.901_MF`, `quadriga` (redesign of NG's QuaDRiGa method). Procedure B (7.6.3.2) dropped after the comparison study: as written it gives 0.47–0.74 (NLOS) and 0.14–0.19 (LOS) of the input spreads with 1–2 dominant clusters; the append-only random-field layout keeps room for it. 7.6.5.1 stays TR-literal, including the order "wrap, then scale by √(1+K)" and the LOS jumps this causes. Default revisited after the ray series with antenna-level metrics. | 2.1, 3.4–3.6 |
| D2 | Mirrored random fields in all methods (D2D reciprocity of the random inputs). `38.901` single frequency only (error otherwise). Spatial consistency allowed with `38.901`, its jumps documented. TR BS-specific terms (μ_offset,ZOD, cluster ZSD, O2I ZOA = 90°) kept as written and documented. A method × property table in the docs. | 3.3, 3.7, 7 |
| D3 | `quadriga` = NG's QuaDRiGa cluster generation (−ln U delays, uniform initial angles, power product with the paper's g fits, analytic DS scaling, NG's iterative linear AS scaling averaged over frequencies, caps 3 / 1.5, NG ground-reflection power split, no per-cluster shadowing) with three changes: LOS direction applied additively (azimuth wrap, elevation pole fold) instead of the 3-D rotation; μ_offset,ZOD added to the NLOS departure elevations; no satellite EoA fade. Not adopted: per-cluster ζ, Gaussian/Laplacian initial values, exact wrapped AS solve, exact per-frequency scaling. No faithful port, no NG parity test; NG's output changes documented (10.3). | 3.6, 10 |
| D4 | `NumClusters = N + 1` in every file: slot 0 = LOS ray (zero power in NLOS), then N NLOS clusters; in TR LOS the first NLOS cluster is at τ = 0, aligned to LOS (7.5-12/-17). +1 in `NumClusters` and `NumClusters_min` of the v19 LOS files (6 files; the plan's count of 8 is corrected in 4); NG's legacy files keep their meaning. | 2.3, 4 |
| D5 | C_φ/C_θ by linear interpolation between table entries (exact at table points, clamped outside); −25 dB removal sets the power to 0, keeps the slot, no renormalisation, only in `38.901`; C_τ applied in `38.901` LOS; signs X_n and jitter Y_n spatially consistent, evaluated by the caller at the drop / segment-start positions and kept along the track; zenith wrap [180°, 360°] → 360° − θ for ZOA and ZOD. | 3.0, 3.4 |
| D6 | Random fields in blocks by variable type, append-only (existing columns never move), ray blocks reserved; one shared cluster set plus one set per frequency (Z_n now, the per-frequency ray blocks later); seeds derived from the LSF seed by a fixed documented rule; no bias removal; 2D + floor IDs as the LSF, 3D legacy mode; `SC_lambda = 0` → 0.1 m. | 2.2, 3.1, 3.2 |
| D7 | Padded slots per link (0 LOS, 1 GR if enabled, then N); zero power for inactive entries (removal, smaller per-link N); delays in s relative to the LOS delay, NLOS + Δτ with abs TOA; frequency dimension of delays and angles only where a method needs it (7.6.5.1 LOS); powers per frequency as the coefficients use them (7.5-30: LOS K/(K+1), NLOS P_n/(K+1)); AoD, EoD, AoA, EoA in rad, GCS, elevation convention; active-cluster count per link. Inputs of the later ray step (cluster spreads, c_DS(f), per-link cluster ZSD, K) are read from `scenpar` / `lsp` / `lsp_mu`, not duplicated. | 2.3, 3.3 |
| D8 | Scope and runs: cluster level only; runs R1, R2a, R2b, R2c, R3 (8). | 1.5, 8 |

### 1.3 Resolution of the TR ambiguities

| ID | Ambiguity | Resolution in this series |
|---|---|---|
| A1 | Signs and coupling: spatially consistent, yet "kept unchanged per drop" (7.6.3.1) | Sign (and, D5, jitter) fields are spatially consistent; the caller evaluates them at the drop or segment-start positions and reuses those values along the track (as the hE rule of the LSF). Coupling: ray series. |
| A2 | 7.5 is discontinuous under SC (sort, max-normalisation, signs, −25 dB) | Accepted for `38.901`; jump sources and rates documented (3.7); `quadriga` is the continuous method. The sort of 7.5-2 is a relabelling: slots keep the cluster identity of the field column (3.0.7). |
| A3 | 7.6.5 asks for frequency-independent delays and angles, 7.5 cannot give them | `38.901` accepts one frequency only (D2); several frequencies with `38.901_MF` or `quadriga`. |
| A4 | 7.6.5.1 LOS: √(1+K_R) uses K per frequency; no min subtraction, sort, −25 dB, C factors, μ_offset,ZOD, O2I rule; anchor "e.g. 2 GHz"; Q_n per frequency not stated | TR-literal: wrap in step 5', scale in step 7'; none of the 7.5 rules added; the anchor LSPs are an input (`lsp_anchor`); Q_n = the per-frequency shadowing field (7.6.5: Z_n independent per frequency); the zenith fold of 7.5 is applied to the final zenith angles (W4). |
| A5, A6 | Procedure B distances and labels; Procedure A distances | Out of scope (D1, 1.5). |
| A7 | Dual mobility and reciprocity undefined in the TR | Mirrored fields (2.2.3); every method is swap-reciprocal for mirrored LSPs when `eod_offset_deg` = 0 and no link is O2I (the BS terms of D2 break it by construction). |
| A8 | Tables 7.5-2 and 7.5-4 have gaps; 7.5-4 is not monotonic (19: 1.184, 20: 1.178) | Linear interpolation between the table entries, constant beyond both ends; the non-monotonic entries are used as given (D5). |
| A9 | The TR's LOS cluster 1 holds the LOS ray and diffuse power; 7.6.15 names Parts 1 and 2 only | D4 (N + 1 slots); `n_cluster` is accepted for every scenario within [max(1, `NumClusters_min` − 1), `NumClusters` − 1] (W11). |
| A10 | −25 dB removal: renormalisation, interaction with sub-clusters and SC | Zero power, slot kept, no renormalisation, criterion on the powers of 7.5-6, `38.901` only (D5); the two strongest clusters are chosen by the ray step from the output powers; the on/off toggles under SC are documented. |
| A11, A13, A15 | Sub-clusters, NF/SNS ranking, per-ray XPR | Ray series. |
| A12 | ZOD wrap stated for ZOA only | Same fold for ZOD (D5). |
| A14 | GR adds power on top of the LOS block, no renormalisation stated | All methods output coefficient powers (D7, W7): TR methods additive (7.6-32), no renormalisation (median +1.6 dB of link power in UMi LOS); `quadriga` LOS K/(1+K), GR K/(1+K)·R² (NG's coherent channel, 7.6-32 without d3D/d_GR); NG's power split stays internal to `quadriga`'s spread scaling (3.3.4). |
| A16 | Which LOS-case power is "the cluster power" | The output is the coefficient power of 7.5-30 (LOS K/(K+1), NLOS P_n/(K+1)); the powers of 7.5-8 are used only inside the angle mapping (3.4). |

NG findings resolved here: N1 (C_τ) applied; N2 (baseline angle mapping) replaced by the TR mapping; N3 (LOS cluster
count) by D4; N4 (ZOD offset never applied) applied in `38.901` and `quadriga`; N5 (−25 dB) applied in `38.901`;
N8 (multi-frequency in baseline mode) by `38.901_MF`/`quadriga`; N10 (frequency-averaged scaling) kept by design;
N11 (satellite fade) dropped; N12/N13 (SSF fields 3D, seed not stored) by the 2D + floor fields and the derived
seeds; N16 (GR power split) kept inside `quadriga`'s spread scaling. N6, N7, N9, N14, N15 belong to the ray series.

### 1.4 Choices of this spec (not fixed by the plan; for the maintainer)

| ID | Choice | Reason | Section |
|---|---|---|---|
| W1 | The LOS condition is a property of the scenario: LOS when `KF_mu > -30` (dB). In an NLOS scenario the LOS slot has zero power whatever KF `lsp` holds (1e-10 from `lsp_generate`). | NG uses one scenario (conf file) per builder and LOS state; the plan lists no per-link LOS input; the TR 7.5 formulas need a hard LOS/NLOS switch (C_τ at K = −100 dB is meaningless); a per-link K threshold would jump under SC. The threshold mirrors the `absTOA_mu > -30` rule; all v19 NLOS/O2I files have −100, all LOS files ≥ 7. Consequence (documented, 3.3.5, 10.2): legacy NG "NLOS" files with KF_mu > −30 run the LOS branch. Accepted by the maintainer. | 3.0.4 |
| W2 | In 7.5-3, 7.5-10, 7.5-15 the K-factor is limited to K ≥ −5 dB. | C_θ of 7.5-15 falls to 0 at K = −10 dB (−0.0004) and turns negative below (−1.61 at −15 dB), which inverts or explodes the zenith angles; at −5 dB C_τ = 0.990 and C_θ = 0.922, near their NLOS limit 1 (C_τ = 1 at −5.2 dB, C_θ = 1 at −4.4 dB), while the LOS factor of C_φ (7.5-10) is 1.181 (it crosses 1 only at −11.7 dB). Cost, measured by the reviewer for K ≤ −8 dB: medians DS 0.94–0.95, ASA 0.86–0.88, ZSA 1.03–1.09 of the input (NLOS limit DS 0.94, ASA 1.02, ZSA 0.95); without the limit ZSA 7.6 at −10 dB; with NG's rule (factor ≥ 1 for K < 0) DS 0.6–0.8. Links below −5 dB: InF LOS 6.7 %, SMa LOS 2.3 %, UMi LOS 0.26 %, InH/RMa LOS 0.13 %, UMa LOS < 0.01 %. Accepted by the maintainer; the ASA effect is documented. | 3.4 |
| W3 | Slots keep the cluster identity (field column) in every method; the "first" cluster of TR LOS is the one with the smallest initial delay (lowest index on ties). | The TR sort only relabels exchangeable clusters; identity keeps SC continuous and the field column of a cluster stable when N changes per link. NG sorts by power when SC is off; the ray step selects the strongest clusters by power. | 3.0.7 |
| W4 | The zenith fold of 7.5 (D5) is also applied to the final zenith angles of `38.901_MF`. | 7.6.5.1 gives zenith angles in (−180°, 360°) and states no rule; the fold is the only zenith convention of the TR; R0 used it. | 3.5 |
| W5 | `quadriga` with several frequencies adds the EoD offset averaged over the frequencies. | Delays and angles must be frequency-independent; the scales are averaged the same way. The v19 UMa NLOS offset depends on frequency. | 3.6 |
| W6 | `quadriga` ignores `o2i` (no 90° ZOA rule); `38.901_MF` ignores `o2i` and `eod_offset_deg`. | D3 lists exactly three changes to NG's method; 7.6.5.1 has neither term (A4). | 3.5, 3.6 |
| W7 | Ground reflection is switched on by the scenario key `GR_enabled` (> 0). Maintainer decision: every method outputs coefficient powers (D7) — LOS slot P_LOS = K/(K+1), GR slot TR methods P_LOS·(d3D/d_GR)²·R², `quadriga` P_LOS·R², with the polarisation-averaged R² = (abs(R∥)² + abs(R⊥)²)/2; NG's (1 − R², R²) split is internal to `quadriga`'s DS/AS scaling. One consumer contract for all methods (2.3): amplitude √(slot power), GR coefficient with diag(R∥, −R⊥)/R. | The key exists and NG uses it. Coefficient powers make one contract possible; NG's channel stays unchanged (NG drops its LOS factor and divides R out, 10.1; NumPy check: identical to 2.5e-16). Cost: the DS measured from the `quadriga` slot powers of a GR link is 0.85 of the input (3.6). | 2.3, 3.3.4, 3.6 |
| W8 | Seed rule: SplitMix64 of the LSF seed and a set tag (2.2.4); MATLAB `set_id` 0 = shared set, k = frequency k (1-based). | Fixed, platform-independent, independent streams per set; the column-of-one-stream variant would tie the per-frequency sets to the size of the shared set and break append-only. | 2.2.4, 6.1 |
| W9 | One layout serves all methods: delay block Uniform, sign and offset blocks Normal; `ssf_clusters` maps Normal → Uniform with Φ where a method needs a uniform. | Same field realisation for every method; the map is the one of `ssf_draw`. | 2.2 |
| W10 | `NumClusters >= 1`; `NumClusters = 1` (N = 0) gives the LOS path only (full power, K ignored) plus the GR slot if enabled; the random-field sets are then empty. Maintainer decision. | NG has LOS-only files (`Freespace`, `LOSonly`, `Null`: NumClusters 1) and a LOS + GR file (`TwoRayGR`: 2 with GR, read with NumClusters − 1, 10.2); one rule serves them in all methods. | 3.0.9 |
| W11 | `n_cluster` must lie in [min(max(1, `NumClusters_min` − 1), N), N], N = `NumClusters` − 1 (all 0 when N = 0). | Uses the TR range where the file states it (7.6.15), any count otherwise. | 3.0.6 |
| W12 | Slots beyond `n_cluster(l)` get delay 0, power 0 and the LOS angles. | Every output element is written; zero-power entries with valid angles are harmless for consumers. | 3.0.6 |
| W13 | Absolute TOA in NLOS scenarios only (TR 7.6-43/-44, all methods): TR methods add Δτ; `quadriga` shifts the NLOS delays to start at Δτ where `abs_toa(l) > 0`; `abs_toa` is ignored in LOS scenarios. | TR-literal; in a LOS scenario the shift would break the exact DS of `quadriga` (measured: DS ratio 1.00 → 1.10 for Δτ = 30 ns, K = 9 dB). NG applies the shift whenever an offset exists; `lsp_generate` returns 0 for the LOS files, so NG with v19 files is unaffected. | 3.3.3, 3.6 |
| W14 | No special case for 1 or 2 NLOS clusters in `quadriga` (NG: fixed ±40.5°). | Fields work for any N; only the deprecated mmMAGIC file has N = 2. | 3.6 |
| W15 | With GR the slot count grows by one; NG takes the GR slot from `NumClusters`. | D7 layout (0 LOS, 1 GR, then N): the number of NLOS clusters does not depend on the GR switch. | 2.3 |
| W16 | `lsp_mu` is not an input of `ssf_clusters` (the plan lists it). | μ_lgZSD enters only 7.5-20, the ray ZOD spread (ray series). | 2.1 |
| W17 | The D4 conf change is done in R2a, not R3 (plan). | R2a's LOS TR-oracle tests read the v19 LOS files with the TR N typed in; they need N + 1 in the files. | 8 |
| W18 | Numerical robustness: cluster powers normalised in the log domain in all three methods (3.0.10); ln(max(u, DBL_MIN)) for uniform delays; ln(max(P, DBL_MIN)) in the 7.5 angle mapping; 7.6.5.1's Laplacian term through the identity 1 − 2·abs(V − 0.5) = erfc(abs(x)/√2) with a DBL_MIN floor; d_GR floored at 1e-5 m; the GR delay in the cancellation-free form 4·h_t·h_r/((d_GR + d3D)·c); permittivity validated (real part ≥ 1, imaginary part ≥ 0) and the only possible 0/0 of the reflection coefficients (ε = 1 + 0j at grazing) set to 0, the value of ε = 1 at every other geometry; `quadriga` scale 1 where a spread is not > 0. | No NaN or Inf for any finite random values (also far outside the ±√(2·n_coeff) range of `ssf_draw`) and LSPs in the ranges `lsp_generate` produces; identical results wherever the floors do not act. | 3.0–3.6 |
| W19 | `ssf_clusters` tests are split into three files (one per method run). | R2b and R2c may run in either order without editing the same test file. | 5 |
| W20 | `quadriga` in an NLOS scenario uses w = 0 for the elevation centres (no pull towards the GR elevation). | The GR slot has zero power there; NG's K-independent weight would move the NLOS EoA centre down by 10–23° (UMa 25/1.5 m, d2D 35–100 m) for a reflection that does not exist. LOS behaviour (D3) unchanged. | 3.6 |
| W21 | The LOS, NLOS and O2I scenarios of a site need different LSF seeds (and hence different SSF streams). | TR 7.6.3.4 makes the link types uncorrelated; one seed would give identical SSF fields in the LOS and NLOS scenario. NG already draws one seed per builder. | 3.1, 10.1 |
| W22 | A consumer without polarisation forms the scalar GR coefficient √(GR slot power)·exp(j·arg(R∥)) (arg(0) = 0). | The maintainer asks that it keep the reflection phase (≈ π near grazing; dropping it reverses the two-ray interference, e.g. +5.8 dB instead of −8.6 dB at 10 km); R∥ is the coefficient of the vertical (θθ) polarisation, the usual base-station case; a consumer with a known polarisation uses its own entry of diag(R∥, −R⊥)/R. | 2.3 |

### 1.5 Out of scope, and how the interface stays ready

Out of scope (later series): the ray level (ray offsets of Table 7.5-3, sub-clusters of Table 7.5-5, coupling,
XPR, initial phases, 7.6.2 large bandwidth, 7.6.10 mobile scatterers, 7.6.13 near field, 7.6.14 SNS, 7.6.16
polarisation variability), Procedure A and B (7.6.3.2), drifting, scatterer geometry (NG's `gen_fbs_lbs`), path
storage in QRT files, calibration against 3GPP (needs the ray step and antennas), converters to `get_channels_*`.

Readiness:
- Random fields: append-only blocks (2.2.5). The shared set takes later frequency-independent cluster blocks (e.g.
  the Procedure A sign, Procedure B blocks at its own decorrelation distances, the 7.6.13 s_TRP draw, SNS x_n); the
  per-frequency sets take the ray blocks of 7.6.5 "steps 8–11 independently per frequency" (coupling, XPR, phases,
  η, 7.6.2.2 offsets, 7.6.10 α and D). Existing columns and their values never change; a stored seed reproduces the
  clusters of this version after a library update.
- Outputs: everything the ray step needs is in the cluster output (slot identity incl. LOS and GR, delays, powers
  per frequency, angles) or in existing outputs: K from `lsp`, μ_lgZSD for 7.5-20 from `lsp_mu` slice 5, cluster
  spreads, c_DS(f), M from `scenpar`, the GR polarisation from the permittivity under the consumer contract of 2.3
  (the slot powers are coefficient powers; the GR slot already contains R²; the ray step applies diag(R∥, −R⊥)/R). Procedure A needs the absolute
  cluster state at the segment start (this output plus d3D/c) and an appended sign block.
- `ssf_clusters` validates the exact sizes of this version's layout; a later version that appends blocks changes the
  size rule together with the layout.

### 1.6 Differences to the SPEC files (approved)

- Review points A, B, C are not stops (see the header).
- Functions are `double` only, without templates, like the rest of `quadriga_gbsm.hpp`; no float/double parity test.
- Argument order of `ssf_clusters`: required inputs, mandatory outputs, `method`, then optional inputs and the
  optional output (as `lsp_generate`, which puts mandatory outputs before optional pointers). `method` stands right
  after the outputs because it decides which optional inputs are needed.
- Test diagnostics use the existing environment switch of `tests/catch2_tests/qd_test_diag.hpp`
  (`QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")`) as the other GBSM test files, not the `#define QD_TEST_VERBOSE` pattern.
- `ssf_init_phases` and `ssf_draw` share `src/qd_gbsm_phases.cpp` with the LSF functions and are tested in one file
  `test_ssf_phases.cpp`; `ssf_clusters` is tested in three files (W19).


## 2. Public API and the random-field layout

### 2.1 Header `include/quadriga_gbsm.hpp` (text to append after the "Large-scale fading" group)

R1 adds the group comment and the first two declarations, R2a the third. No other public function.

```cpp
    // ---- Small-scale fading ----

    // Generators of the small-scale fading (SSF) phase sets, N = NumClusters - 1 NLOS clusters; column c = b·N + n of
    // 'phase' and of the random values belongs to block b and cluster n (0-based):
    //   Shared cluster set (set_id = -1), frequency-independent, n_gen = 9·N
    //     b = 0     cluster delay                Uniform(0,1)   mirror: itself
    //     b = 1, 2  sign of AOD, AOA             Normal(0,1)    mirror: each other
    //     b = 3, 4  sign of ZOD, ZOA             Normal(0,1)    mirror: each other
    //     b = 5, 6  angle offset of AOD, AOA     Normal(0,1)    mirror: each other
    //     b = 7, 8  angle offset of ZOD, ZOA     Normal(0,1)    mirror: each other
    //   Per-frequency set (set_id = k >= 0, the k-th frequency of the caller's list), n_gen = N
    //     b = 0     cluster shadowing            Normal(0,1)    mirror: itself
    // - All generators use SC_lambda (0 is used as 0.1 m); no bias removal; floors and 2D/3D as lsf_draw
    // - Later versions append blocks; existing columns and their values never change
    // - NumClusters = 1 (N = 0): both sets are empty, n_gen = 0
    // - The functions are documented in src/qd_gbsm_phases.cpp (ssf_init_phases, ssf_draw), src/ssf_clusters.cpp

    // Draw the random phases of one small-scale fading generator set of a BS site and scenario
    // - The seed of the set is derived from the seed of the site's large-scale set and 'set_id'
    void ssf_init_phases(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (NumClusters)
                         arma::cube &phase,                                      // Output: Phases in [rad], [n_coeff, n_gen, 2]; slice 0 = Rx side, slice 1 = Tx side
                         arma::sword seed = -1,                                  // Seed of the large-scale set (lsf_init_phases), -1 = system random device
                         arma::sword set_id = -1,                                // -1 = shared cluster set; k >= 0 = set of frequency k (0-based)
                         const std::string &preset = "Comb256");                 // SOS preset, see sos_preset

    // Spatially correlated normalized random values of one small-scale fading generator set at a set of links
    // - Generator order and reciprocity rules: see the SSF generator index above
    void ssf_draw(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (NumClusters, SC_lambda)
                  const arma::cube &phase,                                 // Phases from ssf_init_phases, [n_coeff, n_gen, 2]
                  const arma::mat &tx_pos,                                 // Tx positions, [1, 3] or [n_link, 3]
                  const arma::mat &rx_pos,                                 // Rx positions, [n_link, 3]
                  arma::mat &rnd,                                          // Output: Random values, [n_link, n_gen]
                  arma::sword set_id = -1,                                 // Set type, as in ssf_init_phases: -1 = shared, k >= 0 = per-frequency
                  const arma::uvec *floor_id = nullptr,                    // Optional: Floor index per link, [n_link]; nullptr = all on one floor; 2D only
                  bool dual_mobility = false,                              // Tx and Rx both move: the Tx side is a second device
                  const std::string &preset = "Comb256",                   // Must match ssf_init_phases
                  bool horizontal = true);                                 // true: 2D correlation (TR); false: 3D (legacy)

    // Cluster delays, powers and angles of a set of links, TR 38.901 clause 7.5 steps 5 to 7
    // - Methods: "38.901" (clause 7.5, one frequency), "38.901_MF" (clause 7.6.5.1), "quadriga" (spatially consistent,
    //   multi-frequency, swap-reciprocal; exact delay spread)
    // - Slots per link: 0 = LOS path (zero power in NLOS scenarios), 1 = ground reflection if GR_enabled, then the
    //   N = NumClusters - 1 NLOS clusters; n_slot = NumClusters + (GR ? 1 : 0); NumClusters = 1: LOS path only
    // - The ground-reflection power contains the polarisation-averaged reflection power; see the documentation
    // - Delays relative to the LOS delay; angles in GCS, elevation convention; powers as used by the coefficients
    void ssf_clusters(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters
                      const arma::mat &tx_pos,                                // Tx positions in [m], [1, 3] or [n_link, 3]
                      const arma::mat &rx_pos,                                // Rx positions in [m], [n_link, 3]
                      const arma::cube &lsp,                                  // LSPs from lsp_generate, [n_link, n_freq, 8]
                      const arma::mat &rnd,                                   // Shared cluster set from ssf_draw, [n_link, 9·N]
                      arma::cube &delay,                                      // Output: Delays in [s], [n_link, n_slot, n_fd]; n_fd = n_freq for 38.901_MF, else 1
                      arma::cube &power,                                      // Output: Powers (linear), [n_link, n_slot, n_freq]
                      arma::cube &aod,                                        // Output: Azimuth of departure in [rad], [n_link, n_slot, n_fd]
                      arma::cube &eod,                                        // Output: Elevation of departure in [rad], [n_link, n_slot, n_fd]
                      arma::cube &aoa,                                        // Output: Azimuth of arrival in [rad], [n_link, n_slot, n_fd]
                      arma::cube &eoa,                                        // Output: Elevation of arrival in [rad], [n_link, n_slot, n_fd]
                      const std::string &method = "38.901",                   // "38.901", "38.901_MF" or "quadriga"
                      const arma::cube *rnd_freq = nullptr,                   // Optional: Per-frequency sets from ssf_draw, [n_link, N, n_freq]; required by 38.901, 38.901_MF if N > 0
                      const arma::cube *lsp_anchor = nullptr,                 // Optional: LSPs at the anchor frequency, [n_link, 1, 8]; required by 38.901_MF if N > 0
                      const arma::mat *eod_offset_deg = nullptr,              // Optional: EoD offset in [deg] from lsp_generate, positive = up, [n_link, n_freq]; nullptr = 0
                      const arma::vec *abs_toa = nullptr,                     // Optional: Absolute TOA offset in [s] from lsp_generate, [n_link]; nullptr = 0
                      const arma::mat *gr_eps_re = nullptr,                   // Optional: Ground permittivity, real part, [n_link, n_freq]; required if GR_enabled
                      const arma::mat *gr_eps_im = nullptr,                   // Optional: Ground permittivity, imaginary part, [n_link, n_freq]; required if GR_enabled
                      const std::vector<bool> *o2i = nullptr,                 // Optional: O2I flag per link (mean ZOA 90 deg in 38.901), [n_link]; nullptr = none
                      const arma::uvec *n_cluster = nullptr,                  // Optional: Number of NLOS clusters per link (TR clause 7.6.15), [n_link]; nullptr = N
                      arma::uvec *n_active = nullptr);                        // Optional output: Number of NLOS clusters with power > 0, [n_link]
```

Interface notes (decided; not to be reopened by the runs):
- `std::vector<bool>` per-link flags exist in the library (`active_path` of `quadriga_tools.hpp`).
- The output size does not depend on `n_cluster`; it is padded to N (D7).
- No `lsp_mu` input (W16), no per-link LOS input (W1), no anchor frequency argument (the anchor LSPs come from
  `lsp_generate` with the anchor frequency added to `fc`; 7.6.5.1 takes them "according to Step 4", so
  `clip_spreads` applies).

### 2.2 Random-field layout (public contract)

#### 2.2.1 Sets and columns

N = `NumClusters` − 1. A set is one phase cube `[n_coeff, n_gen, 2]` and one `ssf_draw` output `[n_link, n_gen]`.
With `NumClusters = 1` (N = 0) both sets have n_gen = 0: `ssf_init_phases` returns `[n_coeff, 0, 2]` and `ssf_draw`
`[n_link, 0]` without calling a sum-of-sinusoids function (they reject zero generators).

| Set | `set_id` | n_gen | Block b (columns b·N … b·N + N − 1) | Distribution | Mirror block m(b) |
|---|---|---|---|---|---|
| shared cluster set | −1 | 9·N | 0 delay X_n | Uniform(0,1) | 0 |
| | | | 1 sign AOD, 2 sign AOA | Normal(0,1) | 2, 1 |
| | | | 3 sign ZOD, 4 sign ZOA | Normal(0,1) | 4, 3 |
| | | | 5 offset AOD, 6 offset AOA | Normal(0,1) | 6, 5 |
| | | | 7 offset ZOD, 8 offset ZOA | Normal(0,1) | 8, 7 |
| per-frequency set k | k ≥ 0 | N | 0 shadowing Z_n | Normal(0,1) | 0 |

Mirror of a generator: column `b·N + n` mirrors `m(b)·N + n` (same cluster n). Column `g` of slice 1 of the phase
cube is column `mirror(g)` of slice 0.

Use per method (3.4–3.6; W9): `38.901` uses all nine blocks and Z; `38.901_MF` uses block 0 (X_n), 5 (Y_n), 6 (Z_n
of 7.6.5.1), Φ of 7 (V_n), Φ of 8 (W_n) and the per-frequency Z as Q_n; `quadriga` uses block 0 and Φ of blocks 5
to 8 (initial angles), no per-frequency set. Φ(x) = 0.5·erfc(−x/√2), the map of `ssf_draw`.

#### 2.2.2 Decorrelation distance, floors, 2D/3D, bias

- Every generator of every set uses λ' = `SC_lambda` if > 0, else 0.1 m (TR Table 7.6.3.1-2 gives one distance for
  all cluster and ray variables; `SC_lambda = 0` means "spatial consistency off" and gives fields that decorrelate
  within 0.1 m).
- Floors, 2D/3D and dual mobility exactly as `lsf_draw` (TR 7.6.3.3: SSF variables uncorrelated across floors; the
  caller passes the same `floor_id` as to `lsf_draw`).
- No bias removal (`ssf_init_phases` takes no positions).

#### 2.2.3 Mirror and dual mobility

`ssf_draw` evaluates the set with the generic machinery of `lsf_draw` (`qd_gbsm_eval`): single mobility, column g
is the field of g at the Rx; dual mobility, self-mirrored columns `(s_g(rx) + s_g(tx))/sqrt(2 + 2·A(D))`, pair
columns `(s_g(rx) + s_m(tx))/sqrt(2)`; uniform columns mapped by Φ afterwards. Swapping `tx_pos` and `rx_pos` in dual
mobility leaves blocks 0 (shared) and 0 (per-frequency) unchanged and exchanges block 1 with 2, 3 with 4, 5 with 6,
7 with 8, bit for bit (paper eqs. (3)–(5) for delays and angles).

#### 2.2.4 Seeds

```
splitmix64(x):  z = x + 0x9E3779B97F4A7C15;  z = (z ^ (z >> 30))·0xBF58476D1CE4E5B9;
                z = (z ^ (z >> 27))·0x94D049BB133111EB;  return z ^ (z >> 31)        (all uint64, wrapping)
tag            = 1 for the shared set (set_id = -1),  2 + k for the per-frequency set k
seed_set       = splitmix64( splitmix64( (uint64) seed ) ^ tag ) >> 1               (a value in [0, 2^63))
slice 0        = sos_init_phases(n_coeff, n_gen, seed_set)
seed = -1      : the system random device (sos_init_phases with -1); no relation to any other set
```

Test vectors (computed for this spec): seed 42 → shared 8766244108781592946, k = 0 → 4119046106699552547, k = 1 →
9018399064009245349; seed 0 → shared 313702574736366215, k = 0 → 7774984521324166428. The shared set therefore
never repeats the LSF phases of the same seed, and the sets of different frequencies are independent streams.

#### 2.2.5 Append-only rule

`sos_init_phases` fills column g from one stream independently of `n_gen`: a later version that appends blocks to a
set keeps the phases, and `ssf_draw` (which evaluates each column with its own phases only) keeps the values of all
existing columns bit for bit. Blocks are never reordered or resized; a new block appended to the shared set has N
columns or a multiple of N (ray blocks N·M, cluster-major inside the block). The column layout depends on N: a
scenario with another `NumClusters` (or the D4 change) has another layout.

### 2.3 Slot layout of the `ssf_clusters` outputs

`G = 1` if `GR_enabled > 0`, else 0. `n_slot = NumClusters + G = 1 + G + N`.

| Slot | Content | Delay | Angles | Power |
|---|---|---|---|---|
| 0 | LOS path | 0 | geometric LOS (3.0.3) | LOS scenario: method rule; NLOS scenario: 0 |
| 1 (only if G = 1) | ground reflection (7.6.8) | τ_GR (3.0.5) | GR geometry | method rule (3.3.4); 0 in NLOS scenarios (except N = 0, below) |
| 1 + G + n, n = 0 … N − 1 | NLOS cluster n (field column n of each block) | method | method | method; 0 if removed (`38.901`) or n ≥ `n_cluster(l)` |

In a TR LOS scenario with `38.901` the first NLOS cluster (3.0.7) has delay 0 and the LOS angles, i.e. slots 0 and
the first NLOS slot together are the TR's cluster 1 (specular + diffuse part).

`NumClusters = 1` (N = 0, W10): `n_slot = 1 + G`; slot 0 carries the full power (K treated as infinite: K/(K+1) = 1,
also in an NLOS scenario), the GR slot follows the method rule of 3.3.4 with K/(K+1) = 1 (TR methods
(d3D/d_GR')²·R²_k, `quadriga` R²_k); no NLOS slot.

**Powers and the consumer contract (D7, W7, W22; the same for all methods).** Every slot power is the power its
coefficient carries (7.5-30, 7.6-32): LOS slot P_LOS,k = K_k/(K_k + 1); GR slot P_GR,k = P_LOS,k·(d3D/d_GR')²·R²_k
(TR methods) or P_LOS,k·R²_k (`quadriga`, NG's coherent channel: 7.6-32 without d3D/d_GR); NLOS slots their share of
1/(K_k + 1). R²_k = (abs(R∥)² + abs(R⊥)²)/2 is the polarisation-averaged reflection power of 3.0.5 (R∥, R⊥ with the
TR sign ε = `gr_eps_re` − j·`gr_eps_im`). A consumer (the later ray step, `get_channels_*` with `path_gain` = slot
power and the polarisation transfer matrix M of 8 rows, NG's `get_channels`) forms:
- LOS coefficient: amplitude √(P_LOS,k), polarisation diag(1, −1) of 7.5-29 (M = [1, 0, 0, 0, 0, 0, −1, 0]).
- GR coefficient: amplitude √(P_GR,k), polarisation diag(R∥, −R⊥)/R with R = √R²_k, i.e. M = [Re R∥/R, Im R∥/R, 0,
  0, 0, 0, −Re R⊥/R, −Im R⊥/R]; the reflection is then applied exactly once and the coherent channel is 7.6-32
  (`quadriga`: without d3D/d_GR). R²_k = 0 (only for ε = 1 + 0j) gives a zero GR coefficient.
- A consumer without polarisation (scalar channel) uses √(P_GR,k)·exp(j·arg(R∥)) (arg(0) = 0, W22): it keeps the
  reflection phase, which is ≈ π near grazing; √(slot power) alone reverses the two-ray interference.
- NLOS clusters: amplitude √(slot power); polarisation by the ray step (later series).


## 3. Normative behaviour

### 3.0 Shared definitions

#### 3.0.1 Notation

Per link l (0-based), frequency index k = 0 … F − 1 (F = `lsp.n_cols`), cluster index n = 0 … N − 1. Inputs:
`DS_k = lsp(l, k, 0)` [s], `KF_k = lsp(l, k, 1)` [linear], `ASD_k = lsp(l, k, 3)`, `ASA_k = lsp(l, k, 4)`,
`ZSD_k = lsp(l, k, 5)`, `ZSA_k = lsp(l, k, 6)` [deg] (ZSD/ZSA = ESD/ESA of `lsp_generate`). Shared set:
`u_n = rnd(l, n)`; for the four angle types x = 0 AOD, 1 AOA, 2 ZOD (EoD), 3 ZOA (EoA): sign field
`s^x_n = rnd(l, (1 + x)·N + n)`, offset field `y^x_n = rnd(l, (5 + x)·N + n)`. Per-frequency: `z_{n,k} =
rnd_freq(l, n, k)`. Scenario: `r_τ = r_DS`, `ζ = LNS_ksi` [dB], N, `N_min = NumClusters_min`. Constants: c =
299792458 m/s, π, DBL_MIN = 2.2250738585072014e-308.

Helpers (file-local in `src/ssf_clusters.cpp`, `static inline` or anonymous namespace):

```
wrap(x)    = x - 2π·floor((x + π)/(2π))                      in [-π, π)        (rad; wrap180 the same in deg)
zfold(t)   = t' = t - 360·floor(t/360);  t' > 180 ? 360 - t' : t'      zenith fold in deg, result in [0, 180]
pole(e)    = e > π/2 ? π - e : (e < -π/2 ? -π - e : e)          for e in [-π, π)
efold(e)   = pole(wrap(e))                                    elevation fold, result in [-π/2, π/2]
Φ(x)       = 0.5·erfc(-x/√2)
elev(t)    = (90 - zfold(t))·π/180                            zenith in deg -> elevation in rad
```

`zfold` is the TR rule "θ wrapped to [0°, 360°], θ ∈ [180°, 360°] → 360° − θ" (7.5 step 7, D5); at exactly 180°
both branches agree. `pole` mirrors an elevation beyond a pole at the pole without changing the azimuth (NG's rule).

#### 3.0.2 Validation of `ssf_clusters` (in this order, before any output is touched)

Messages are `std::invalid_argument` unless noted. Inputs that the selected method does not use (column "used by"
of 3.3.1) are ignored and not validated.

1. `method` ∈ {"38.901", "38.901_MF", "quadriga"}: `Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.`
2. Positions: `qd_gbsm_check_links(&tx_pos, &rx_pos)` (existing messages). n_link = `rx_pos.n_rows`.
3. `scenpar`: copy, `scenpar_validate(sp, "legacy", nullptr, nullptr, true)` (messages pass through; it already
   enforces `NumClusters >= 1` and `NumClusters_min <= NumClusters`).
4. `lsp`: `n_rows == n_link`, `n_cols >= 1`, `n_slices == 8`, else `Input 'lsp' must have size [n_link, n_freq, 8].`;
   finite, else `Input 'lsp' must be finite.`; slices 0, 3, 4, 5, 6 > 0, else `Delay and angular spreads in 'lsp' must
   be > 0.`; slice 1 >= 0, else `K-factor in 'lsp' must be >= 0.` (slices 2 and 7 are not used and not checked
   beyond finiteness). F = `lsp.n_cols`.
5. `38.901` and F > 1: `Method '38.901' supports one frequency only; use '38.901_MF' or 'quadriga' for several
   frequencies.`
6. `rnd`: size `[n_link, 9·N]` (N = 0: any empty matrix is accepted, as MATLAB passes `zeros(n, 0)` as 0×0), else `Input 'rnd' must have size [n_link, <9N>].` (the number written out); finite,
   else `Input 'rnd' must be finite.`; columns 0 … N − 1 in [0, 1], else `Columns 0 to N-1 of 'rnd' (cluster delays)
   must be in [0, 1].`
7. `rnd_freq` (38.901, 38.901_MF; only if N > 0): nullptr → `Input 'rnd_freq' is required for method '<method>'.`; size
   `[n_link, N, F]`, else `Input 'rnd_freq' must have size [n_link, <N>, n_freq].`; finite. (N = 0: not required,
   ignored.)
8. `lsp_anchor` (38.901_MF; only if N > 0): nullptr → `Input 'lsp_anchor' is required for method '38.901_MF'.`; size `[n_link, 1,
   8]`; finite; slices 0, 3, 4, 5, 6 > 0 (`Delay and angular spreads in 'lsp_anchor' must be > 0.`).
9. `eod_offset_deg` (38.901, quadriga; if given): size `[n_link, F]`, finite.
10. `abs_toa` (all; if given): `n_elem == n_link`, finite, >= 0, else `Input 'abs_toa' must be finite and >= 0.`
11. Ground reflection, if `GR_enabled > 0` (all methods): `gr_eps_re` and `gr_eps_im` given, else `Inputs 'gr_eps_re'
    and 'gr_eps_im' are required when 'GR_enabled' is set.`; each `[n_link, F]`, finite; `gr_eps_re >= 1` and
    `gr_eps_im >= 0`, else `Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.` (no ground is below
    the vacuum value; `lsp_generate` returns real parts 3.0–44 between 0.5 and 100 GHz and imaginary parts >= 0; its
    fit falls below 1 above about 1.3 THz, where `ssf_clusters` rejects it — to be documented); column 2 of `tx_pos` and of `rx_pos` >= 0, else `Ground
    reflection requires z >= 0 in 'tx_pos' and 'rx_pos'.` (the ground is the plane z = 0).
12. `o2i` (38.901; if given): `size() == n_link`, else `Number of elements in 'o2i' does not match number of rows in
    'rx_pos'.`
13. `n_cluster` (all; if given): `n_elem == n_link`; every entry in [N_lo, N] with N_lo = min(max(1, N_min − 1), N)
    (N = 0: all entries 0), else
    `Entries of 'n_cluster' must be in [<N_lo>, <N>].`

Size checks use n_link = 0 consistently (empty inputs with the right number of columns/slices pass).

#### 3.0.3 LOS geometry (all methods)

`t = tx_pos` row l (row 0 if `tx_pos` has one row), `r = rx_pos` row l, `v = r − t`:

```
d2D = hypot(v_x, v_y);  d3D = norm(v)
AoD_L = atan2(v_y, v_x);   EoD_L = atan2(v_z, d2D);   AoA_L = wrap(AoD_L + π);   EoA_L = -EoD_L      [rad]
ZOD_L = 90 - EoD_L·180/π;  ZOA_L = 90 - EoA_L·180/π;  φD_L = AoD_L·180/π;  φA_L = AoA_L·180/π       [deg]
```

Co-located Tx and Rx give atan2(0, 0) = 0 (no special case). Slot 0 always carries these angles and delay 0.

#### 3.0.4 LOS condition (W1)

`LOS = (KF_mu > -30)` for the whole call. `K_k = LOS ? KF_k : 0`. In an NLOS scenario every LOS and GR power is
exactly 0 and every method runs its NLOS branch (exception: N = 0, 3.0.9).

Consequence to document: legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run the LOS branch; measured in NG HEAD:
`BERLIN_UMi_Campus_NLOS` (−13.2), `DRESDEN_UMa_NLOS` (−10.4), `QuaDRiGa_UD2D_NLOS` (−8), `BERLIN_UMa_NLOS` and
`MIMOSA_*_NLOS` (−6.3), `BERLIN_UMi_Square_NLOS` (−5.5), `QuaDRiGa_Industrial_NLOS` (+4.22), the 3GPP 37.885 `NLOSv`
file (0). `quadriga` reproduces NG for them (LOS power K/(1+K)); `38.901` applies C_τ, the K-limited C factors and
the first-cluster alignment.

#### 3.0.5 Ground-reflection geometry (all methods, G = 1)

Heights h_t = t_z, h_r = r_z (>= 0, validated). Per link:

```
dGR   = sqrt(d2D² + (h_t + h_r)²);  dGR' = max(dGR, 1e-5)
τ_GR  = 4·h_t·h_r / ((dGR + d3D)·c)   (0 if dGR + d3D = 0)   [s], >= 0; equals (dGR - d3D)/c without the
                                                         cancellation (relative error of the difference form 1.6e-11
                                                         at 10 km, 1.1e-6 at 500 km)
E_GR  = -atan2(h_t + h_r, d2D)                          elevation of the GR at both ends [rad] (7.6-37a, -37c)
GR slot angles: AoD = AoD_L, AoA = AoA_L (7.6-37b, -37d), EoD = EoA = E_GR
Per frequency k, ε = gr_eps_re(l, k) - j·gr_eps_im(l, k)  (TR sign of 7.6-40; abs(R) does not depend on it),
cθ = -(h_t + h_r)/dGR',  s2 = (d2D/dGR')²,  q = sqrt(ε - s2)  (complex principal root):
R∥ = (ε·cθ + q)/(ε·cθ - q)   (7.6-38),   R⊥ = (cθ + q)/(cθ - q)   (7.6-39)
     a coefficient whose denominator is exactly 0 is set to 0 (only possible at cθ = 0 with ε = 1 + 0j, the edge
     of the validated range, where R∥ = R⊥ = 0 at every other geometry: vacuum does not reflect). For ε ≠ 1 the
     grazing case has no 0/0 and gives R∥ = R⊥ = -1 by the formulas.
R²_k = (abs(R∥)² + abs(R⊥)²)/2                          polarisation-averaged reflection power (W7)
```

cθ is cos θ_GR,ZOD and s2 is sin² θ_GR,ZOD of 7.6-37a. `std::complex<double>` is allowed inside the function. The
GR geometry (τ_GR, E_GR, d3D/d_GR') and R²_k are computed per link before the method kernel and passed to it
(`quadriga` needs them in steps 3, 5 and 6 of 3.6); only the writing of the slots follows the kernel (3.3.2).

#### 3.0.6 Per-link cluster number (7.6.15) and inactive slots

`N_l = n_cluster ? (*n_cluster)(l) : N`. Clusters n < N_l are active; slots of clusters n ≥ N_l get delay 0, power 0
at all frequencies and the angles of slot 0 (W12). Every normalisation, minimum, maximum, sum and scale of the
methods runs over the active clusters only. `n_active(l)` = number of active NLOS slots whose power is > 0 at one
frequency at least.

#### 3.0.7 Cluster identity (W3)

NLOS slot 1 + G + n always belongs to field column n. No method sorts. In TR LOS, the first cluster is
`f = argmin_{n < N_l} τ'_n` (lowest index on ties); it is the TR's cluster 1 after the sort of 7.5-2.

#### 3.0.8 Table interpolation (D5)

```
Table 7.5-2  N:     4      5      6      7      8      10     11     12     14     15     16     19     20     25
            C_φ:  0.779  0.860  0.921  0.973  1.018  1.090  1.123  1.146  1.190  1.211  1.226  1.273  1.289  1.358
Table 7.5-4  N:     6      7      8      10     11     12     14      15      16      19     20     25
            C_θ:  0.788  0.847  0.889  0.957  1.031  1.104  1.1072  1.1088  1.1276  1.184  1.178  1.282
```

`C^NLOS(N_l) = qd_interp_linear(values, entries, n_tab, (double)N_l)` (existing helper of
`src/quadriga_lib_helper_functions.hpp`: linear between entries, constant beyond both ends; n_tab = 14 or 12). Examples: C_φ(9) = 1.054, C_φ(13) = 1.168, C_θ(9) = 0.923, C_θ(30) = 1.282. N_l is the
per-link number (7.6.15 replaces N); the −25 dB removal does not change it (7.5 step 6: "the scaling factors need
not be changed").

#### 3.0.9 Cluster count; the LOS-only case (W10, maintainer decision)

`NumClusters >= 1` (the range of `scenpar_validate`). N = 0 is the LOS-only case (NG's `Freespace`, `LOSonly`,
`Null`; `TwoRayGR` with GR, 10.2):
- `ssf_init_phases`: `[n_coeff, 0, 2]`; `ssf_draw`: `[n_link, 0]`; no sum-of-sinusoids call (2.2.1).
- `ssf_clusters`: validation as usual with `rnd` `[n_link, 0]`; `rnd_freq` and `lsp_anchor` not required (ignored);
  `n_cluster` entries 0. Then every method short-circuits before its cluster formulas (no C tables, no spread
  scaling, no normalisation over an empty set): K/(K+1) := 1 and 1/(K+1) := 0 at every frequency, whatever the
  scenario and `lsp` K. Slot 0: delay 0, LOS angles, power 1. GR slot (if G = 1): GR geometry, power TR methods
  (d3D/d_GR')²·R²_k (7.6-32), `quadriga` R²_k (coefficient powers, total 1 + R²_k; NG's coherent channel of its
  `TwoRayGR` branch). `n_active` = 0. Output sizes as 3.3.2 with n_slot = 1 + G (`38.901_MF`: n_fd = F;
  `38.901` keeps its one-frequency rule).

#### 3.0.10 Power normalisation in the log domain (W18)

Every method forms its unnormalised cluster powers as logarithms and normalises with the largest one subtracted:
```
Λ_n = ln P'_n   (38.901: -τ_n·(r_τ - 1)/(r_τ·DS) - ζ·z_{n,0}·ln(10)/10;   38.901_MF: the exponent of 7.6.30a
                 - ζ·z_{n,k}·ln(10)/10;   quadriga: the exponent of step 5)
P_n = exp(Λ_n - max_m Λ_m) / Σ_m exp(Λ_m - max_m Λ_m)
```
Identical to P'_n/Σ P'_m wherever no overflow or underflow occurs; it removes the overflow of 10^(−ζ·z/10) for
extreme finite z and the all-underflow case Σ P' = 0 of 7.6.30a (anchor spread far above the spread at k).

### 3.1 `ssf_init_phases`

Source file `src/qd_gbsm_phases.cpp`, after `lsf_draw`, section "Geometry-based stochastic model".

Validation, in this order: `seed >= -1` (`Input 'seed' must be >= -1.`); `set_id >= -1` (`Input 'set_id' must be
>= -1.`); preset known (`qd_gbsm_n_coeff`); `scenpar` completed and validated (copy + `scenpar_validate`); 3.0.9.

Algorithm:
1. N = `NumClusters` − 1; generator table `qd_gbsm_gen_ssf(sp, set_id)` (declared below); n_gen = its size.
2. `seed_set` by 2.2.4 (seed −1 passes −1 on). `sos_init_phases<double>(n_coeff, n_gen, P0, seed_set)`.
3. `phase` `[n_coeff, n_gen, 2]` (resize only if the size differs): slice 0 = P0, slice 1 column g = P0 column
   `gen[g].mirror`. N = 0: `[n_coeff, 0, 2]` after the validation, step 2 skipped.

New file-local code in the anonymous namespace of `src/qd_gbsm_phases.cpp` (used by both SSF functions):

```cpp
// SSF generators of a completed scenpar: set_id = -1 shared cluster set (9·N), >= 0 per-frequency set (N)
std::vector<qd_gbsm_gen> qd_gbsm_gen_ssf(const std::unordered_map<std::string, double> &sp, arma::sword set_id);
```

The seed rule of 2.2.4 is a lambda inside `ssf_init_phases` (its only user; helper scoping rule).

`qd_gbsm_gen_ssf` sets `lambda = SC_lambda > 0 ? SC_lambda : 0.1`, `uniform = (shared && b == 0)`, `mirror` by
2.2.1, `bias = false`. The header comment of the file (its "Design for further phase sets" paragraph) is replaced
by the SSF index of 2.2.1, the seed rule and the append-only rule; the LSF index stays.

Documentation block must state: the two set types and their layout (table of 2.2.1 in words), mirror rule, seed
rule with the formula (SplitMix64 written out) and that seed −1 gives unrelated random phases, that the caller keeps
an explicit seed to reproduce the clusters, that each scenario (link type: LOS, NLOS, O2I) of a site needs its own
LSF seed because TR 38.901 clause 7.6.3.4 makes the link types uncorrelated (one seed would give the LOS and the NLOS
scenario identical cluster fields, W21), the N = 0 case (empty sets), the per-frequency set index belongs to the caller's frequency list
(same order in every call), the append-only rule, no bias removal, `SC_lambda` not used here (only by `ssf_draw`),
validation order. Output: `[n_coeff, n_gen, 2]`, values in [−π, π).

### 3.2 `ssf_draw`

Source file `src/qd_gbsm_phases.cpp`, after `ssf_init_phases`.

Validation, in this order (messages of `lsf_draw` where the check is the same): positions
(`qd_gbsm_check_links`); `set_id >= -1`; preset known; `scenpar` completed and validated; 3.0.9; `phase` size
`[n_coeff, n_gen, 2]`, else `Input 'phase' must have size [<n_coeff>, <n_gen>, 2] for preset '<preset>' and this
set.` (n_gen = 0: any empty `phase` is accepted, as MATLAB passes `zeros(n_coeff, 0, 2)` as 0×0×0); `phase` finite;
`floor_id` length; `floor_id` with `horizontal = false` (same message as `lsf_draw`).

Algorithm: `n_link = 0` or n_gen = 0 → `rnd` `[n_link, n_gen]`, return. Else `qd_gbsm_freq` (one table: all generators share λ'),
`qd_gbsm_positions`, `qd_gbsm_eval(gen, …, rnd)` — exactly the `lsf_draw` path with the SSF generator table. No new
machinery.

Documentation block must state: the columns (2.2.1), λ' = `SC_lambda` (0 → 0.1 m) for all, distributions and Φ
for block 0 of the shared set, single and dual mobility formulas, swap rule (2.2.3), floors as `lsf_draw` (same
`floor_id`), 2D/3D, precision (AVX2 sin/cos, below `4e-7·sqrt(2·n_coeff)`), `preset`, `horizontal` and the set type
must match `ssf_init_phases`, the value of a link does not depend on the other links and a column does not depend on
the phases of the other columns, the caller's rule for `38.901` under spatial consistency (signs and offsets
evaluated at the drop or segment start and reused along the track, D5), `n_link = 0`, validation order.

### 3.3 `ssf_clusters`: shared part

New file `src/ssf_clusters.cpp`, section "Geometry-based stochastic model"; it becomes a user of
`qd_gbsm_check_links`, so the users list of `src/quadriga_gbsm_helpers.hpp` (file comment and the comment above the
helper) adds `ssf_clusters.cpp` (R2a). Layout: includes; anonymous namespace
with the helpers of 3.0 and one file-local kernel per method (`qd_ssf_tr`, `qd_ssf_mf`, `qd_ssf_quadriga`, each a
self-contained algorithm of more than 20 lines, placed directly above the documentation block); documentation
block; definition.

#### 3.3.1 Inputs used per method

| Input | `38.901` | `38.901_MF` | `quadriga` |
|---|---|---|---|
| `rnd` blocks | 0–8 | 0, 5, 6, 7, 8 | 0, 5, 6, 7, 8 |
| `rnd_freq` | required (F = 1) | required | ignored |
| `lsp_anchor` | ignored | required | ignored |
| `eod_offset_deg` | NLOS scenarios (μ_offset,ZOD = −value) | ignored (W6) | NLOS scenarios: mean over k (W5) |
| `abs_toa` | NLOS scenarios: + Δτ | NLOS scenarios: + Δτ | NLOS scenarios: shift to Δτ if > 0 (W13) |
| `gr_eps_*` (`GR_enabled`) | TR 7.6.8 | TR 7.6.8 | NG split inside the scaling; output coefficient powers |
| `o2i` | NLOS: mean ZOA 90° | ignored (W6) | ignored (W6) |
| `n_cluster` | yes (also in C_φ, C_θ) | yes | yes |

#### 3.3.2 Outputs

After validation: N, G, n_slot = 1 + G + N, n_fd = (method == "38.901_MF") ? F : 1. N = 0: 3.0.9. `delay`, `aod`, `eod`, `aoa`,
`eoa` `[n_link, n_slot, n_fd]`, `power` `[n_link, n_slot, F]`, `n_active` (if requested) `[n_link]`; resize only if
the size differs, every element written. `n_link = 0`: outputs with 0 rows, return.

Per link: GR geometry and R²_k (3.0.5) before the kernel; after the kernel the slots are written: slot 0 (3.0.3,
delay 0, power `K_k/(K_k+1)` in all methods), slot 1 if G (power 3.3.4), inactive slots (3.0.6), `n_active`. For `38.901_MF` the frequency-independent
slots (0, 1, inactive, NLOS in an NLOS scenario) are written into every k of the n_fd dimension.

The link loop may use `#pragma omp parallel for` (signed loop variable; no exception inside: all checks precede it);
results do not depend on the number of threads (each link is computed alone).

#### 3.3.3 Absolute TOA (7.6.9)

TR methods, NLOS scenarios only: every active NLOS delay (all frequencies of n_fd) `+= abs_toa(l)` (7.6-43: τ_n +
Δτ; the d3D/c of the equation is the LOS delay to which all output delays are relative). In LOS scenarios `abs_toa`
is ignored (7.6-44/-44a have no Δτ; W13). `quadriga`: 3.6 step 10.

#### 3.3.4 Ground-reflection power (7.6.8)

GR power contract: 2.3. TR methods (7.6-32, no renormalisation, A14): `power(l, 1, k) = K_k/(K_k + 1)·(d3D/dGR')²·R²_k`; the LOS slot keeps
`K_k/(K_k + 1)` and the NLOS slots `P_n/(K_k + 1)`, so the total power is `1 + K_k/(K_k + 1)·(d3D/dGR')²·R²_k`
(UMi LOS 6 GHz median +1.6 dB, p90 +1.9 dB). `quadriga` (W7): `power(l, 1, k) = K_k/(K_k + 1)·R²_k`, LOS slot
`K_k/(K_k + 1)`, NLOS slots summing to `1/(K_k + 1)` (total `1 + K_k/(K_k + 1)·R²_k`); NG's split (1 − R², R²) of the
LOS power is used only inside the DS and AS scaling (3.6 steps 5–7). Consumer contract: 2.3.

#### 3.3.5 Documentation block (must state)

Purpose and the three methods in one bullet each with their TR clause; the slot layout of 2.3 and n_slot; LOS
condition (W1) and that NLOS gives zero LOS/GR power whatever `lsp` holds; units and conventions (delays relative to
the LOS delay, i.e. absolute delay = d3D/c + delay; angles GCS, rad, elevation; azimuth in [−π, π), elevation in
[−π/2, π/2]); powers are the coefficient powers of 7.5-30 (LOS K/(K+1), NLOS P_n/(K+1)), shadow fading and path loss
not included (step 12, caller); the inputs used per method (3.3.1); per-link N and the padding; `n_active`; the
random-field use (2.2.1) and the SC rule for `38.901` (signs and offsets of the drop / segment start, D5); GR (on
by `GR_enabled`, slot, power rule per method, no renormalisation in the TR methods, z = 0 is the ground, the
permittivity range (incl. the THz remark of 3.0.2 #11), and the consumer contract of 2.3 written out: coefficient
powers in all methods, LOS diag(1, −1), GR diag(R∥, −R⊥)/R, R² = 0 → zero GR coefficient, the scalar rule W22, and
for `quadriga` that the DS of a GR link measured from the slot powers is about 0.85 of the input (3.6)); the LOS-only case `NumClusters = 1` (3.0.9); the legacy-file
consequence of the LOS condition (3.0.4) and the ASA effect of the −5 dB K limit (W2: about 0.86 of the input for
links far below −5 dB); abs TOA rule per method; the method × property table of 3.7;
for each method its formulas in the compact form of 3.4–3.6 and the TR deviations (W2, W3, W4); swap reciprocity
conditions; `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario (`lsp_generate`, `ssf_draw`;
not checked beyond the sizes); `n_link = 0`; outputs must not share memory with the inputs; validation order.

### 3.4 Method `38.901` (TR 7.5 steps 5–7, one frequency, k = 0)

Computation in degrees (as the TR), conversion at the end. Per link with N_l active clusters (n, m < N_l):

Step 5 (7.5-1 to 7.5-4):
```
τ'_n = -r_τ·DS·ln(max(u_n, DBL_MIN));    τ_n = τ'_n - min_m τ'_m;    f = argmin_m τ'_m (3.0.7)
LOS:  K_dB = 10·lg(K) (K = KF_0; K = 0 gives -Inf),  K* = max(K_dB, -5)   (W2)
      C_τ = 0.7705 - 0.0433·K* + 0.0002·K*² + 0.000017·K*³
      output delay of cluster n: τ_n / C_τ     (7.5-4; not used for the powers)
NLOS: output delay τ_n
```

Step 6 (7.5-5 to 7.5-8, removal):
```
P'_n = exp(-τ_n·(r_τ - 1)/(r_τ·DS)) · 10^(-ζ·z_{n,0}/10);    P_n = P'_n / Σ_m P'_m   (7.5-5, 7.5-6; computed as 3.0.10)
removed_n = P_n < 10^(-2.5)·max_m P_m                                   (−25 dB against the 7.5-6 powers, D5)
angle-mapping powers: NLOS Pa_n = P_n;  LOS Pa_n = P_n/(K + 1) + (n == f ? K/(K + 1) : 0)    (7.5-8)
output power of cluster n: removed_n ? 0 : P_n/(K + 1)      (K = 0 in NLOS; 7.5-30); slot 0: K/(K + 1)
```

Step 7 (7.5-9 to 7.5-19), for x = AOD (AS = ASD), AOA (ASA), ZOD (ZSD), ZOA (ZSA):
```
ρ_n   = ln( max(Pa_n, DBL_MIN) / max_m Pa_m )                                   (<= 0; W18 floor)
C_φ   = C_φ^NLOS(N_l)·(LOS ? 1.1035 - 0.028·K* - 0.002·K*² + 0.0001·K*³ : 1)     (7.5-10)
C_θ   = C_θ^NLOS(N_l)·(LOS ? 1.3086 + 0.0339·K* - 0.0077·K*² + 0.0002·K*³ : 1)  (7.5-15)
a'_n  = azimuth: 2·(AS/1.4)·sqrt(-ρ_n)/C_φ        zenith: -AS·ρ_n/C_θ          (7.5-9, 7.5-14)
X_n   = s^x_n >= 0 ? +1 : -1;     Y_n = y^x_n·AS/7                             (Y_n ~ N(0, (AS/7)²))
v_n   = X_n·a'_n + Y_n
NLOS: AOD_n = φD_L + v_n;  AOA_n = φA_L + v_n;                                   (7.5-11)
      ZOD_n = ZOD_L - eod_offset_deg(l, 0) + v_n                                  (7.5-19; μ_offset,ZOD = -EoD offset)
      ZOA_n = (o2i(l) ? 90 : ZOA_L) + v_n                                         (7.5-16)
LOS:  angle_n = LOS angle + v_n - v_f   for all four (7.5-12, 7.5-17; ZOD in LOS uses 7.5-17, no offset)
output: aod/aoa = wrap(angle·π/180),  eod/eoa = elev(angle)   (zenith fold for ZOD and ZOA, D5)
```

Notes for the implementer and the documentation:
- Removed clusters keep their delay and angles (the slot stays, D5); they still enter max and ρ (max is never a
  removed cluster).
- The first cluster of LOS gets exactly delay 0 and the LOS angles; a removed first cluster (possible with large ζ)
  keeps them with zero power.
- No sorting (W3). TR's "C_φ for LOS" uses K in dB "as generated in Step 4" (W2 limits it).
- With `eod_offset_deg` omitted the offset is 0; with `o2i` omitted no link is O2I. In a LOS scenario `o2i` and
  the offset have no effect (7.5-17 aligns the first cluster to the LOS direction; the TR gives μ_offset,ZOD = 0 for
  LOS); the v19 O2I files are NLOS scenarios.
- A scratch NumPy transcription of this section agrees with the comparison-study prototype to 2e-15 (50 random
  links, LOS and NLOS, K ≥ 0 dB where the limit of W2 does not act).

### 3.5 Method `38.901_MF` (TR 7.6.5.1, TR-literal, D1)

Anchor spreads `DS0, ASD0, ASA0, ZSD0, ZSA0 = lsp_anchor(l, 0, {0, 3, 4, 5, 6})`, r = 1.5. Angles in degrees.

Step 5' (frequency-independent; n < N_l):
```
τ'_n     = -r·DS0·ln(max(u_n, DBL_MIN))                                   [s]
φ'_AOD,n = wrap180(-r·ASD0·y^AOD_n)          (arg(exp(-j·r·ASD0·Y_n)), Y_n = y^AOD_n)
φ'_AOA,n = wrap180(-r·ASA0·y^AOA_n)          (Z_n of 7.6.5.1 = y^AOA_n)
θ'_ZOD,n = wrap180(-r·ZSD0·sgn(y^ZOD_n)·ln(max(erfc(abs(y^ZOD_n)/√2), DBL_MIN))/√2)
θ'_ZOA,n = wrap180(-r·ZSA0·sgn(y^ZOA_n)·ln(max(erfc(abs(y^ZOA_n)/√2), DBL_MIN))/√2)
LOS: τ'_0 = 0 and all four angles of cluster 0 are 0
```
The TR writes the zenith terms with V_n = Φ(y^ZOD_n) as `sgn(V_n − 0.5)·ln(1 − 2·abs(V_n − 0.5))`; this equals
`sgn(y)·ln(erfc(abs(y)/√2))` exactly and keeps full precision near V = 0 or 1 (W18). sgn(0) = 0. The azimuths use
the TR's minus sign of `arg(exp(−j·…))` (the comparison-study prototype used +; statistically equivalent; a
transcription of this section was checked against the prototype with the sign flipped: equal to 5e-15).

Step 6' per frequency k (7.6.30a–h), with DS, ASD, ASA, ZSD, ZSA at k:
```
g_DS  = max(r·DS0 - DS, 0)/(DS·r·DS0)                         [1/s]
g_ASD = sqrt(max((r·ASD0)² - ASD², 0))/(ASD·r·ASD0)            [1/deg], g_ASA likewise
g_ZSD = max(r·ZSD0 - ZSD, 0)/(ZSD·r·ZSD0)                      [1/deg], g_ZSA likewise
P'_n  = exp(-τ'_n·g_DS - (φ'_AOD,n·g_ASD/√2)² - (φ'_AOA,n·g_ASA/√2)² - √2·abs(θ'_ZOD,n)·g_ZSD
            - √2·abs(θ'_ZOA,n)·g_ZSA) · 10^(-ζ·z_{n,k}/10)           (Q_n = ζ·z_{n,k}, 7.6.5: per frequency)
P_n   = P'_n/Σ_m P'_m (computed as 3.0.10);   output power: P_n/(K_k + 1);   slot 0: K_k/(K_k + 1)   (7.6.30g, h)
```

Step 7' per frequency k (7.6.30i–m), cluster level (the c·α_m ray terms are the ray step):
```
LOS: cτ = sqrt(1 + K_k/2),  ca = sqrt(1 + K_k);   NLOS: cτ = ca = 1
delay(n, k) = cτ·τ'_n  (+ abs_toa, 3.3.3)
aod = wrap((φD_L + ca·φ'_AOD,n)·π/180);   aoa = wrap((φA_L + ca·φ'_AOA,n)·π/180)
eod = elev(ZOD_L + ca·θ'_ZOD,n);          eoa = elev(ZOA_L + ca·θ'_ZOA,n)          (zenith fold, W4)
```

TR-literal points to document: the nominal angles are wrapped in step 5' and scaled by √(1+K_R) in step 7'
afterwards, so in LOS an angle that crosses ±180° in step 5' jumps by ca·360° before the final wrap (measured
0.05–0.07 jumps per metre along tracks in LOS); in LOS, delays and angles depend on frequency through K_k (they are
equal at all frequencies when K does not depend on frequency); no min subtraction of the NLOS delays (no cluster at
τ = 0 in NLOS), no sort, no −25 dB removal, no C factors, no μ_offset,ZOD (mean EoD 1.9–3.0° off the 7.5 statistics in
UMi/UMa NLOS), no O2I rule; per-cluster shadowing Q_n per frequency; the anchor is an input.

### 3.6 Method `quadriga` (redesign of NG's QuaDRiGa method, D3)

Per link; F frequencies; N_l active clusters; angles in rad. `AS_{x,k}` = lsp spreads at k in rad (ASD, ASA, ESD,
ESA for x = AOD, AOA, EOD, EOA). Source: NG `gen_ssf_parameters.m` (QuaDRiGa branch) and the paper; the three D3
changes are marked [D3].

1. Initial delays (paper eq. (1)): `τ̃_n = -ln(max(u_n, DBL_MIN))` (unitless).
2. Initial angles (paper eqs. (4), (5)): `ã^x_n = (Φ(y^x_n) - 0.5)·π` ∈ (−π/2, π/2), x = AOD, AOA, EOD (block 7),
   EOA (block 8).
3. Centres and fixed slots. Azimuth: `μ_D = AoD_L`, `μ_A = AoA_L`. Elevation without GR: `ε_D = EoD_L`,
   `ε_A = EoA_L`. With GR: `ω = (1/F)·Σ_k R²_k`, `ε_D = (1 − ω)·EoD_L + ω·E_GR`, `ε_A = (1 − ω)·EoA_L + ω·E_GR`
   (NG's linear mean of the fixed angles with weights 1 − ω, ω, independent of K); in an NLOS scenario ω = 0 (W20:
   the GR has zero power there). Fixed-slot offsets used in step
   7: azimuth 0; elevation `EoD_L − ε_D` (LOS) and `E_GR − ε_D` (GR) for departure, `EoA_L − ε_A`, `E_GR − ε_A` for
   arrival; without GR the LOS offset is 0.
4. Power coefficients (paper eqs. (7)–(11) with the clipping as NG, which corrects the paper's typos):
   ```
   F = 1:  g_DS = r_τ - 1;  g_AOD = g_AOA = 0.56;  g_EOD = g_EOA = 0.76
   F > 1:  if abs(max_k DS_k / min_k DS_k - 1) < 1e-6:  g_DS,k = r_τ - 1
           else  d_k = min(max(DS_k/(max_k DS_k + min_k DS_k), 0.15), 0.85);  g_DS,k = -1.5·ln(1.2·d_k - 0.15)
           azimuth:   b_k = max(0.75·AS_{x,k}/max_k AS_{x,k}, 0.25);  g_{x,k} = -2.2·ln(1.5·b_k - 0.35)
           elevation: b_k = max(0.75·AS_{x,k}/max_k AS_{x,k}, 0.25);  g_{x,k} = -3.4·ln(1.2·b_k - 0.1)
   ```
5. Scaling powers per k (paper eqs. (6), (12); NG GR split, D3; these drive steps 6 and 7 only):
   ```
   Λ_n     = -τ̃_n·g_DS,k - (ã^AOD_n)²·g_AOD,k - (ã^AOA_n)²·g_AOA,k - abs(ã^EOD_n)·g_EOD,k - abs(ã^EOA_n)·g_EOA,k
   S_LOS,k = K_k/(1 + K_k)·(1 - G·R²_k);   S_GR,k = G·K_k/(1 + K_k)·R²_k
   P_n,k   = (1/(1 + K_k))·exp(Λ_n - max Λ)/Σ_m exp(Λ_m - max Λ)      (3.0.10; S_LOS + S_GR + Σ P = 1; no ζ)
   ```
6. Delay scale per k (paper eq. (13), (14); NG closed form with GR), weights S_LOS, S_GR, P_n (sum 1):
   ```
   no GR:  b = Σ_n P_n,k·τ̃_n,  c = Σ_n P_n,k·τ̃_n²;  s_k = (c - b² > 0) ? DS_k/sqrt(c - b²) : 1
   GR:     a = DS_k², B = S_GR,k·τ_GR², D = S_GR,k·τ_GR, c as above and e = Σ_n P_n,k·τ̃_n;
           den = c - e²;  t = max(a·c - a·e² - B·c + B·e² + c·D², 0);  s_k = (den > 0) ? (sqrt(t) + D·e)/den : 1
   s_τ = (1/F)·Σ_k s_k;   τ_n = s_τ·τ̃_n   [s]
   ```
   (With GR the delay spread cannot be smaller than the one of LOS + GR alone; t is then clamped at 0, as NG.)
7. Angle scale per type x and frequency k (NG's iterative linear solve, D3). Angle set A = fixed-slot offsets of
   step 3 and the active ã^x_n; weights p̂ = (S_LOS,k, S_GR,k, P_n,k) (sum 1):
   ```
   wrapped spread:  μ = atan2(Σ p̂·sin A, Σ p̂·cos A);  φ = wrap(A - μ);  S0 = sqrt(max(Σ p̂·φ² - (Σ p̂·φ)², 0))
   linear spread:   S(A) = sqrt(max(Σ p̂·A² - (Σ p̂·A)², 0))
   if !(S0 > 0): s_{x,k} = 1
   else: sL = 10·lg(AS_{x,k}/S0)
         for i = 1 .. 9:  A_n = 10^(sL/10)·ã^x_n (fixed slots unchanged);  S1 = S(A);  if !(S1 > 0) break;
                          sN = 10·lg(AS_{x,k}/S1);  if abs(sN) <= 1e-4 break;  sL += sN
         s_{x,k} = 10^(sL/10)
   s_x = min((1/F)·Σ_k s_{x,k}, cap_x),   cap = 3 (AOD, AOA), 1.5 (EOD, EOA)        (paper eq. (18))
   ```
8. Scaled offsets: `b^x_n = s_x·ã^x_n`; elevation offsets folded at the poles: `b^EOD_n = efold(b^EOD_n)`,
   `b^EOA_n = efold(b^EOA_n)` (NG's mirror at the poles).
9. Angles of the NLOS clusters, LOS direction applied additively [D3, replaces paper eq. (25)]:
   ```
   aod = wrap(μ_D + b^AOD_n);    aoa = wrap(μ_A + b^AOA_n)
   eod = efold(ε_D + δ + b^EOD_n),  δ = LOS ? 0 : (1/F)·Σ_k eod_offset_deg(l, k)·π/180  (0 if omitted)  [D3; W5]
   (NLOS scenarios only, as the comparison-study redesign and TR 7.5-17: no μ_offset,ZOD in LOS)
   eoa = efold(ε_A + b^EOA_n)                                     [D3: no satellite EoA fade]
   ```
   Slot 0 and the GR slot keep their geometric angles (3.0.3, 3.0.5); NG's rotation gives the same for them.
10. Absolute TOA (W13), NLOS scenarios only: if `abs_toa` is given and `abs_toa(l) > 0`: `τ_n = τ_n − min_m τ_m +
    abs_toa(l)` over the active clusters; ignored in LOS scenarios.
11. Output: delays `[·, ·, 1]`: slot 0 → 0, GR → τ_GR, clusters → τ_n; coefficient powers per k (W7): slot 0
    K_k/(1 + K_k), GR slot K_k/(1 + K_k)·R²_k, clusters P_n,k; angles of step 9. No per-cluster shadowing (ζ unused), `rnd_freq` and `o2i` unused.

A scratch NumPy transcription of steps 1–9 (one frequency, no GR) agrees with the prototype's redesign to 2e-15 in
LOS; in NLOS to 4e-16 s in delay and 2e-10 rad, because the prototype keeps NG's K = 1e-10 where this spec uses 0.

Properties to document (measured in the comparison study on UMi/UMa, 6 and 30 GHz, 4000 links, unless "exact"):
- DS exact at one frequency: the delay spread of the output (power-weighted, LOS slot included) equals `DS_0`
  without GR and with abs TOA in NLOS scenarios. With GR the scaling targets NG's split (S_LOS, S_GR), so the DS from
  the output slot powers is smaller: measured 0.845 median (p10–p90 0.81–0.90) in UMi LOS and 0.849 (0.82–0.89) in
  UMa LOS at 6 GHz, medium ground ε = 13.5 − 1.18j, 1000 links each; the DS with the split powers (slot 0 power
  minus GR power, GR power, clusters) is exact unless t is clamped; several frequencies: the averaged scale
  leaves a per-frequency DS error of about ±15–20 % (p10–p90 0.85–1.23 at the outer frequency of 2/6/30 GHz).
- AS: below the cap and after convergence the linear power-weighted spread of the offsets around their mean equals
  the input (relative 2.3e-5); with the library's circular definition the medians are 1.00 for ASD, ESD, ESA; ASA is
  limited by the K-factor in LOS (median 0.86 UMi LOS, 0.59 UMa LOS; a LOS ray of K/(K+1) at one angle bounds the
  azimuth spread).
- Continuous along tracks and between nearby UTs (no jumps at 1 cm steps); delays and angles independent of
  frequency; swap-reciprocal (mirrored fields and LSPs, offset 0).
- Differs from 7.5 in the angle–power relation (Spearman ρ of angle offset and power −0.31…−0.34 vs −0.80…−0.89 in
  NLOS) and in the LOS ASA at moderate K (UMa LOS 0.61 vs 0.83).

### 3.7 Method × property table (for the `ssf_clusters` documentation, D2)

| Property | `38.901` | `38.901_MF` | `quadriga` |
|---|---|---|---|
| Statistics of TR 7.5 (output/input spread, median) | reference: DS 0.96–1.02, ASD 0.95–1.02, ASA 0.99 (NLOS) / 0.80–0.89 (LOS), ESD 0.97–1.04, ESA 0.97–1.03 | NLOS 0.88–0.94, LOS 0.44–0.77 | DS 1.00, ASD 1.00, ESD 1.00, ESA 1.00, ASA 1.00 (NLOS) / 0.59–0.86 (LOS) |
| Continuous along a track (spatial consistency) | no: in LOS the first cluster changes (0.03–0.11 jumps/m), the −25 dB rule switches clusters (0.03–0.19 /m), signs and offsets must be frozen per segment; nearby UTs see sign flips (0.5–1.5 jumps per m of separation) | NLOS yes; LOS no (0.05–0.07 /m, wrap before √(1+K)) | yes |
| Delays and angles equal at all frequencies | one frequency only | NLOS yes; LOS only if K is frequency-independent | yes |
| Exact spreads | no (C factors calibrated) | no | DS exact at one frequency; AS exact below the caps (linear definition); averaged scales with several frequencies |
| Swap-reciprocal (mirrored fields and LSPs) | yes, if `eod_offset_deg` = 0 and no O2I link | yes | yes, if `eod_offset_deg` = 0 |
| Per-cluster shadowing | yes (ζ) | yes (Q_n) | no |
| TR terms μ_offset,ZOD (NLOS), O2I ZOA 90° | both | neither | offset only |
| `NumClusters = 1` | LOS path only (+ GR, 7.6-32) | LOS path only (+ GR, 7.6-32) | LOS path only (+ GR, R²) |
| Ground reflection (coefficient powers in all methods) | TR 7.6.8: GR K/(K+1)·(d3D/d_GR)²·R² | TR 7.6.8 | GR K/(K+1)·R² (NG's channel); NG split inside the scaling |

The numbers are those of the comparison study for UMi and UMa at 6 and 30 GHz (they go into the documentation as
"typical values for UMi and UMa"). The documentation recommends `38.901` for drop-based calibration with one
frequency, `quadriga` for tracks, several frequencies or D2D, and `38.901_MF` where the TR's multi-frequency method
is required.


## 4. Conf change (D4, done in R2a, W17)

The v19 LOS files hold the TR N; NLOS and O2I files already hold N + 1 ("+ 1 for the disabled LOS path"). Six files
change (the plan said 8; the four `*_LOS_O2I` files are NLOS for the small-scale fading and already hold N + 1):

| File `data/config/3GPP_38.901_v19_…` | `NumClusters` | `NumClusters_min` |
|---|---|---|
| `UMa_LOS.conf` | 12 → 13 | 10 → 11 |
| `UMi_LOS.conf` | 12 → 13 | 6 → 7 |
| `Indoor_LOS.conf` | 15 → 16 | 7 → 8 |
| `RMa_LOS.conf` | 11 → 12 | (not present) |
| `SMa_LOS.conf` | 15 → 16 | (not present) |
| `InF_LOS.conf` | 25 → 26 | (not present) |

Comment of the `NumClusters` line in these six files (column layout unchanged): `% number of clusters (<N> NLOS
clusters + 1 for the LOS path)`, with N = 12, 12, 15, 11, 15, 25. The `NumClusters_min` comments stay (they already
say "counted like NumClusters (TR: D1 = …)"). Nothing else changes in the conf files; no key is added.

`tests/catch2_tests/test_conf_files_v19.cpp` (R2a): the table entries `UMi_LOS NumClusters 12 → 13`, `SMa_LOS 15 →
16`, `RMa_LOS 11 → 12`, `Indoor_LOS 15 → 16`, `InF_LOS 25 → 26`, `UMi_LOS NumClusters_min 6 → 7`, `UMa_LOS
NumClusters_min 10 → 11`, `Indoor_LOS NumClusters_min 7 → 8`; the comment "LOS files have a K-factor, all others
switch the LOS path off and carry one extra cluster" becomes "LOS files have a K-factor, all others switch the LOS
path off; every file carries the TR cluster number N plus one slot for the LOS path"; the comment above the table
entries (line 349, "Number of clusters: N for LOS, N + 1 otherwise; lower end of the range D1 likewise (Table
7.6.15-1)") becomes "Number of clusters: N + 1 in every file; lower end of the range D1 + 1 (Table 7.6.15-1)"; one
new check in that loop:
`NumClusters == N_TR + 1` with N_TR typed from TR Table 7.5-6 per file (UMi LOS 12, NLOS 19, O2I 12; UMa 12, 20, 12;
RMa 11, 10, 10; Indoor 15, 19; InF 25, 25 (all InF files); SMa 15, 14, 14 (LOS_O2I and NLOS_O2I as O2I)).


## 5. Catch2 tests

### 5.0 Conventions for the new test files

- Layout of the existing GBSM test files (`test_gbsm_phases.cpp`): license, diagnostic note, header comment (what the
  suite pins down, the oracles, the conventions), `QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")`, helpers in an anonymous
  namespace, test case names `"<function> - <statement>"`, one behaviour per case. Add the new file names to the
  `QD_DIAG_GBSM` line of `qd_test_diag.hpp`.
- Oracles are typed in the test file in scalar double from the formulas of this spec and the TR; they never call
  `ssf_clusters` or the file-local helpers. TR constants (r_τ, ζ, N of Table 7.5-6, the C tables, the K polynomials)
  are typed in the test, independent of the conf files, so that tests check code and conf files together.
- Supporting tasks use the library: `conf_file_read` / `scenpar_validate`, `sos_preset`, `sos_init_phases`,
  `sos_acfi`, `lsf_init_phases`, `lsf_draw`, `lsp_generate`, `ssf_init_phases`, `ssf_draw`, `calc_delay_spread`,
  `calc_angular_spreads_sphere`. The C-table interpolation of the oracles is typed in the test (no library helper).
- No Armadillo matrix products above 4×4, no `arma::norm`, `dot`, `accu(a % b)` (no BLAS in the test binary).
- Tolerances: exact-by-construction values 0 (bitwise, `==` on doubles where the same operations are repeated);
  typed TR oracles 1e-12 relative for delays and powers, 1e-11 rad for angles (the oracle may order operations
  differently; angles go through degrees and up to a few thousand degrees before the wrap); `ssf_draw` values 2e-5
  (Normal) and 1e-5 (Uniform) for 256 sinusoids, as `lsf_draw` (AVX2 single-precision sin/cos); statistics from the
  sample size (5 standard errors, stated per case).
- Input fixtures: distinctive values (different per link and cluster), positions not on axes, at least one link with
  `tx_pos` given per link.

### 5.1 `tests/catch2_tests/test_ssf_phases.cpp` (R1)

| Case | Assertion | Oracle / tolerance |
|---|---|---|
| `ssf_init_phases - output size follows the set and the scenario` | shared `[n_coeff, 9N, 2]`, per-frequency `[n_coeff, N, 2]` for NumClusters 1 (empty sets), 2, 13, 26 and presets Comb256, Exp512 | sizes, exact |
| `ssf_init_phases - slice 0 equals sos_init_phases with the derived seed` | seed 42 and 0, set −1, 0, 1 | SplitMix64 typed in the test; the test vectors of 2.2.4 checked first; then `sos_init_phases(n_coeff, n_gen, seed_set)` bitwise |
| `ssf_init_phases - slice 1 is the mirror of slice 0` | shared: blocks 1↔2, 3↔4, 5↔6, 7↔8 per cluster, block 0 itself; per-frequency: itself | bitwise |
| `ssf_init_phases - appending generators keeps the existing columns` | first n_gen columns of `sos_init_phases(n_coeff, n_gen + 9, seed_set)` equal slice 0 | bitwise (append-only contract) |
| `ssf_init_phases - the sets are distinct from each other and from the large-scale set` | for seed 42: no column of the shared set equals a column of `lsf_init_phases` (no bias removal) or of the per-frequency sets 0, 1; sample correlation of the phase values between set 0 and set 1 below 5/sqrt(n) | n = 256·N values |
| `ssf_init_phases - the same seed gives the same phases, seed -1 new ones` | two calls with 7 equal; two with −1 differ | bitwise / not equal |
| `ssf_init_phases - input validation` | baseline `CHECK_NOTHROW`; sections: seed −2, set_id −2, unknown preset, NumClusters 0, an invalid scenpar value | `std::invalid_argument` with the message substring |
| `ssf_draw - single mobility values equal the sum of sinusoids` | shared and per-frequency set, 5 links, 2D | typed `ref_sos` as in `test_gbsm_phases.cpp` with the frequencies of `sos_preset(preset, λ')`; Φ for block 0; 2e-5 / 1e-5 |
| `ssf_draw - all generators use SC_lambda, 0 is used as 0.1 m` | `SC_lambda = 0` equals `SC_lambda = 0.1` bitwise; λ' in the oracle above with SC_lambda 15 | bitwise |
| `ssf_draw - dual mobility values follow the two-device formulas` | self-mirrored and pair columns | typed formulas of 2.2.3 with `sos_acfi`; 2e-5 |
| `ssf_draw - swapping Tx and Rx exchanges the departure and arrival blocks` | dual mobility, shared and per-frequency set, also with floor IDs | bitwise |
| `ssf_draw - a column does not depend on the phases of the other columns` | change all other columns of both slices (keeping the mirror relation); the column is unchanged | bitwise (append-only on the draw side) |
| `ssf_draw - floors as in lsf_draw` | floor 0 equals no floor ID bitwise; floors 0 and 1 at equal positions uncorrelated | correlation over 10^4 links spaced 5λ' below 0.05 (5/sqrt(n)) |
| `ssf_draw - 2D mode does not depend on z, 3D mode does` | z changed by 3 m | bitwise / differs |
| `ssf_draw - marginal distributions` | 10^4 positions on a grid of spacing 5λ'; per column of a 9N = 108 column set: Normal mean, std; block 0 mean and fraction below 0.1; sign fraction (x >= 0) per sign column | 5 standard errors: Normal mean 0.05, std 0.05; Uniform mean 0.015, fraction 0.015; sign fraction 0.025 |
| `ssf_draw - the correlation at the decorrelation distance follows the preset ACF` | offsets λ' in 4 directions, all columns | `sos_preset` ACF at λ'; 0.1 (as the LSF case: 5 standard errors + directional SOS error) |
| `ssf_draw - sizes, empty input and pre-sized output` | `[0, n_gen]` for 0 links; `[n_link, 0]` for NumClusters 1; pre-sized output written completely; a wrong-size output resized | exact |
| `ssf_draw - input validation` | sections: positions, set_id −2, preset, scenpar, NumClusters 0, phase size (shared cube passed as per-frequency set and vice versa), non-finite phase, floor_id length, floor_id with 3D | `std::invalid_argument` |

### 5.2 `tests/catch2_tests/test_ssf_clusters.cpp` (R2a: shared part and `38.901`)

Fixture: scenpar from the v19 files (`conf_file_read`), links with distinct geometry (BS 25 m, UTs 1.5–22.5 m, d2D
35–300 m), synthetic `lsp` (DS 40–400 ns, AS 5–60°, ZS 1–20°, K −8…20 dB), synthetic `rnd`/`rnd_freq` with distinct
values (delays u in (0.02, 0.98), Normal values in (−2.5, 2.5)). The `38.901` oracle `ref_tr` is a typed scalar
implementation of 3.4 with the TR constants of the scenario typed in.

| Case | Assertion | Oracle / tolerance |
|---|---|---|
| `ssf_clusters - output sizes and slot layout` | n_slot = NumClusters (+1 with GR), n_fd = 1, `n_active` size; slot 0 delay 0 and LOS angles | 3.0.3 typed; exact / 1e-12 |
| `ssf_clusters - 38.901 NLOS matches TR 7.5` | UMa NLOS (r_τ 2.3, ζ 3, N 20, C_φ 1.289, C_θ 1.178), 4 links, with `eod_offset_deg` | `ref_tr`; 1e-12 rel / 1e-11 rad |
| `ssf_clusters - 38.901 LOS matches TR 7.5` | UMa LOS after D4 (N 12, r_τ 2.5, ζ 3), K = 3, 9, 15 dB: C_τ delays, first cluster at 0 delay and LOS angles, slot 0 power K/(K+1), NLOS powers P_n/(K+1) | `ref_tr`; same |
| `ssf_clusters - 38.901 limits the K-factor to -5 dB in the C factors` | K = −8 dB equals the result with the polynomials at −5 dB; the LOS power still uses K = −8 dB | `ref_tr` with K* typed; 1e-12 |
| `ssf_clusters - 38.901 removes clusters 25 dB below the strongest without renormalisation` | z chosen so that 2 clusters fall below −25 dB: their power 0, delay and angles kept; other powers unchanged against the oracle; sum of powers < 1; `n_active` = N − 2 | `ref_tr`; exact zeros |
| `ssf_clusters - 38.901 uses the per-link cluster number in the C factors` | `n_cluster` 9, 13, 17, 18, 21 on InF NLOS (N 25): C from the typed tables by linear interpolation; inactive slots delay 0, power 0, LOS angles; NumClusters 31 override for N_l = 30 (clamped C) | typed interpolation; 1e-12 |
| `ssf_clusters - 38.901 O2I links use 90 degrees as mean ZOA` | UMi NLOS O2I file, `o2i` true for 2 of 4 links | `ref_tr`; 1e-11 rad |
| `ssf_clusters - 38.901 folds zenith angles at 0 and 180 degrees` | large y values drive ZOA/ZOD beyond 180° and below 0° | `ref_tr` with zfold; elevations within [−π/2, π/2]; 1e-11 |
| `ssf_clusters - absolute TOA adds the offset to the NLOS delays of NLOS scenarios` | NLOS, `abs_toa` 3 values: NLOS delays + Δτ, slot 0 stays 0; omitted = 0; LOS scenario: `abs_toa` ignored | exact difference 1e-15·delay / bitwise |
| `ssf_clusters - ground reflection follows TR 7.6.8` | UMa LOS with `GR_enabled = 1`: GR slot delay (oracle (d_GR − d3D)/c at d2D ≤ 300 m, 1e-12 relative), angles, power (typed R∥, R⊥ of 7.6-38/-39 with ε = re − j·im and also with +j: same power); NLOS slots identical to GR off (shifted by one slot); total power > 1 (no renormalisation); NLOS scenario: GR power 0; grazing case h_t = h_r = 0 with ε = 1: finite, R = 0 (GR power 0); grazing h_t = h_r = 0 with ε = 15 − 1j: R∥ = R⊥ = −1 exactly | typed; 1e-12 |
| `ssf_clusters - NLOS scenarios give zero LOS power whatever the K-factor in lsp` | UMi NLOS with lsp K = 1e-10 and K = 5: identical outputs, slot 0 power 0 | bitwise |
| `ssf_clusters - 38.901 is swap-reciprocal for mirrored inputs` | LOS and NLOS, offset 0, no O2I: swap positions, LSP slices 3↔4, 5↔6, `rnd` blocks 1↔2, 3↔4, 5↔6, 7↔8: delays and powers equal, AoD↔AoA, EoD↔EoA | 1e-12 (angles compared after wrap of the difference) |
| `ssf_clusters - a link does not depend on the other links` | a link alone equals the same link inside a batch of 50 (the OpenMP loop) | bitwise |
| `ssf_clusters - 38.901 spread statistics over a drop` | UMa NLOS 6 GHz, 2000 links (5 seeds × 400 UTs, d2D 35–300 m), scenpar copy with `SC_lambda = 0` and all LSF decorrelation distances 0 (fields at 0.1 m: independent links), LSPs from `lsf_draw`/`lsp_generate`, fields from `ssf_draw`: median output/input ratio of DS, ASD, ASA, ESD, ESA (`calc_delay_spread`, `calc_angular_spreads_sphere` without wrapping) | expected medians (study, UMa NLOS 6 GHz) ± 0.03: DS [0.94, 1.00], ASD [0.99, 1.05], ASA [0.96, 1.02], ESD [0.95, 1.01], ESA [0.96, 1.02]. Measured spread of the median over 12 repetitions of 2000 independent links (38.901 transcription, typical UMa NLOS spreads): 0.0046, 0.0042, 0.0045, 0.0053, 0.0048 → ±0.03 ≥ 5.6 standard errors (with SC_lambda = 50 m the spread was 0.0089 for DS: too wide for 2000 links) |
| `ssf_clusters - one cluster gives the LOS path only` | `NumClusters = 1`, all three methods (R2a: `38.901`; R2b and R2c add a section in their file), LOS and NLOS scenario, F = 1 and (MF, quadriga) 2: `rnd` `[n_link, 0]`, no `rnd_freq`/`lsp_anchor` needed, n_slot = 1, slot 0 power 1, delay 0, LOS angles, `n_active` 0; with `GR_enabled`: n_slot 2, TR methods GR power (d3D/d_GR)²·R², `quadriga` (1, R²) | typed; 1e-12 |
| `ssf_clusters - extreme finite random values give finite powers` | `38.901`: z = −1100 on one cluster (10^(330) would overflow): that cluster carries all NLOS power (1/(K+1)), the others ≥ 0, no NaN; Normal blocks of `rnd` and `rnd_freq` at ±40 (delay block 0 and 1): all outputs finite, all three methods (R2b, R2c add their method) | 3.0.10; exact / finite |
| `ssf_clusters - empty input and pre-sized outputs` | 0 links: outputs `[0, n_slot, n_fd]`; pre-sized outputs written completely; wrong sizes resized | exact |
| `ssf_clusters - input validation` | baseline `CHECK_NOTHROW` for `38.901` (the baselines of the other methods are in their own files, 5.3, 5.4); one section per rule of 3.0.2 (method name, positions, scenpar (NumClusters 0, an invalid value), permittivity below 1 / negative imaginary part with GR, lsp size/finite/spread/K, 38.901 with 2 frequencies, rnd size/finite/delay range, rnd_freq missing/size, lsp_anchor missing/size/spread (with 38.901_MF), eod_offset size, abs_toa size/negative, GR inputs missing/size, negative z with GR, o2i length, n_cluster length/range); an input unused by the method is not checked (e.g. a wrong-size `lsp_anchor` with `38.901` passes) | `std::invalid_argument` with message substring |
| `Diagnostic - ssf_clusters 38.901 jumps along a track` | UMi LOS, 20 m track, 1 cm steps, signs and offsets frozen at the start: prints events per metre (angle > 10°, delay > DS/4, power > 3 dB, clusters within 20 dB) | sanity only (finite) |

### 5.3 `tests/catch2_tests/test_ssf_clusters_mf.cpp` (R2b)

Oracle `ref_mf`: typed scalar implementation of 3.5 with the TR form `sgn(V − 0.5)·ln(1 − 2·abs(V − 0.5))` for the
Laplacian (inputs with abs(y) < 2.5, so the TR form and the erfc identity agree to 1e-13).

| Case | Assertion | Oracle / tolerance |
|---|---|---|
| `ssf_clusters - 38.901_MF NLOS matches TR 7.6.5.1` | UMi NLOS (N 19, ζ 3), F = 3 (lsp at 3 frequencies, anchor separately), powers per frequency with Q_n per frequency, delays without min subtraction | `ref_mf`; 1e-12 rel / 1e-11 rad |
| `ssf_clusters - 38.901_MF LOS scales the wrapped angles by sqrt(1 + K)` | UMi LOS, an input with r·ASA0·y beyond 180° so that wrap-then-scale differs from scale-then-wrap; delays √(1 + K/2)·τ' per frequency with K differing by frequency; cluster 0 at delay 0 and LOS angles | `ref_mf`; checks that the TR order is used |
| `ssf_clusters - 38.901_MF delays and angles are frequency-independent in NLOS` | all n_fd slices equal | bitwise |
| `ssf_clusters - 38.901_MF g factors vanish when the spread exceeds r times the anchor spread` | DS > 1.5·DS0: power independent of delay | `ref_mf` |
| `ssf_clusters - 38.901_MF ignores the EoD offset and the O2I flag` | outputs with and without both are equal | bitwise |
| `ssf_clusters - 38.901_MF is swap-reciprocal for mirrored inputs` | LOS, NLOS (anchor mirrored too) | 1e-12 |
| `ssf_clusters - 38.901_MF output sizes` | n_fd = F, ground reflection per frequency (TR 7.6.8) | exact / typed |
| `ssf_clusters - 38.901_MF one cluster gives the LOS path only` | `NumClusters = 1`, F = 2, with and without GR: n_slot 1 / 2, n_fd = 2, slot 0 power 1 at both frequencies, GR (d3D/d_GR)²·R²_k per frequency; no `rnd_freq`, `lsp_anchor` | typed; 1e-12 |
| `ssf_clusters - 38.901_MF accepts valid inputs and ignores the inputs it does not use` | baseline `CHECK_NOTHROW` (the checks of its inputs are in the validation case of 5.2); `o2i` and `eod_offset_deg` of wrong size do not throw | no throw |

### 5.4 `tests/catch2_tests/test_ssf_clusters_quadriga.cpp` (R2c)

Oracle `ref_qd`: typed scalar implementation of 3.6 (including the iteration). Tolerance for the angles 1e-9
relative + 1e-12 rad, delays and powers 1e-10 relative: the iteration stops at a threshold, and a rounding
difference at the threshold changes the scale by up to 2.3e-5 relative; if the comparison fails at that level, the
diagnostic table prints the iteration counts (the oracle and the function must take the same branches for the
fixture; the fixture avoids values within 1e-6 dB of the threshold).

| Case | Assertion | Oracle / tolerance |
|---|---|---|
| `ssf_clusters - quadriga single frequency matches the algorithm` | UMi LOS and NLOS, 4 links, with `eod_offset_deg` (applied in NLOS only) | `ref_qd` |
| `ssf_clusters - quadriga several frequencies use the paper's power coefficients and averaged scales` | F = 3 with frequency-dependent DS and AS; and F = 2 with equal DS (r_τ − 1 rule) | `ref_qd` |
| `ssf_clusters - quadriga delay spread equals the input at one frequency` | 200 links, LOS and NLOS, GR on/off, abs TOA in NLOS; with GR the DS is computed from the split powers recovered from the output (slot 0 power − GR power, GR power, clusters) and the case prints the DS from the slot powers (expected about 0.85); the assertion covers the links where t of 3.6 step 6 is > 0 (computed by the oracle) and the case prints the number of clamped links (0 of 400 in the reviewer's fixture) | `calc_delay_spread` of the output (all slots) against DS: 1e-10 relative |
| `ssf_clusters - quadriga angular spreads equal the input below the caps` | 200 links with AS ≤ 20°, ES ≤ 10° (scale < 2, no wrap and no pole fold): offsets recovered as wrap(angle − centre) (EoD: minus δ); typed linear spread with the output powers | 1e-4 relative (iteration criterion 1e-4 dB = 2.3e-5) |
| `ssf_clusters - quadriga limits the scale to 3 in azimuth and 1.5 in elevation` | NLOS links with ASA 170° and ESA 100° in `lsp` (beyond the clipping of `lsp_generate`, so that the unlimited scale exceeds the caps: the initial spreads are about 42° and 44°): azimuth: wrap(angle − centre) equals wrap(3·ã) (3ã exceeds π for abs(ã) > π/3, so the ratio is not compared); elevation: recovered offset / initial offset = 1.5 on clusters where no pole fold acts | 1e-12 |
| `ssf_clusters - quadriga applies the LOS direction additively` | angles = wrap(centre + offset), efold for elevations, slot 0 and GR geometric; an EoD near the pole folds | `ref_qd`; 1e-12 |
| `ssf_clusters - quadriga adds the EoD offset to the NLOS departures only` | NLOS scenario: δ = mean over frequencies; EoA, azimuths, LOS and GR slots unchanged; LOS scenario: output with the offset equals the output without it | difference exact to 1e-12 / bitwise |
| `ssf_clusters - quadriga outputs coefficient powers with the ground reflection` | slot 0 K/(1+K), GR K/(1+K)·R², clusters summing to 1/(1+K) at every frequency; the scaling uses the split (checked through `ref_qd`); elevation centre (1 − ω)·EoD_L + ω·E_GR in LOS, EoD_L / EoA_L (ω = 0) in an NLOS scenario | typed; 1e-12 |
| `ssf_clusters - quadriga shifts the NLOS delays to the absolute TOA only where it is positive` | NLOS scenario: abs_toa > 0: min NLOS delay = Δτ; abs_toa = 0 and omitted: no shift; LOS scenario: `abs_toa` ignored (output bitwise equal to omitted) | exact / 1e-15 |
| `ssf_clusters - quadriga gives zero LOS power in NLOS scenarios` | K in lsp ignored | bitwise |
| `ssf_clusters - quadriga is continuous along tracks` | UMi LOS and NLOS, 5 tracks × 20 m at 1 cm steps, LSPs and fields from the library along the track (`lsf_draw`, `lsp_generate`, `ssf_draw`): no event (angle > 10°, delay > DS/4, power > 3 dB, clusters within 20 dB of the strongest) | count 0 (study: 0 events) |
| `ssf_clusters - quadriga is swap-reciprocal with dual-mobility fields` | `ssf_draw` with dual mobility at swapped positions, mirrored LSPs, offset 0, LOS and NLOS, heights 1.5/1.5 and 1.5/25 m | 1e-10 |
| `ssf_clusters - quadriga ignores rnd_freq and o2i` | equal outputs | bitwise |
| `ssf_clusters - quadriga one cluster gives the LOS path only` | `NumClusters = 1`, F = 2, with and without GR: slot 0 power 1, GR R²_k per frequency (NG's `TwoRayGR` channel), delays 0 / τ_GR | typed; 1e-12 |
| `ssf_clusters - quadriga accepts valid inputs and ignores the inputs it does not use` | baseline `CHECK_NOTHROW` without `rnd_freq`; `rnd_freq`, `lsp_anchor` and `o2i` of wrong size do not throw | no throw |
| `Diagnostic - ssf_clusters quadriga spread ratios over a drop` | UMi/UMa NLOS and LOS 6 GHz: medians of output/input per spread (library definitions) | sanity only |

### 5.5 Changes to existing tests

R2a: `test_conf_files_v19.cpp` as in 4. No other existing test changes (the LSF tests do not depend on
`NumClusters`; the MEX/Python conf tests write their own files).


## 6. MEX and Python wrappers (R3)

General: thin wrappers (conversion, index shift of `set_id` in MEX, output allocation, call); the documentation
blocks adapt the C++ blocks (MEX: 1-based columns and slices; Python: 0-based); `scenpar` as struct (MEX,
`qd_mex_struct2scenpar`) / dict (Python, `qd_python_dict2scenpar`), `SubpathMethod` ignored; outputs whose size
depends on `scenpar` (`phase`, `rnd`, the cluster outputs) are allocated after the call and copied
(`qd_mex_copy2matlab`, `qd_python_copy2numpy`) — no pre-sizing in the wrapper. MEX wrappers reject zero links (empty
`rx_pos` loses its 3 columns in `qd_mex_get_Mat`, and the C++ check fails); Python accepts `(0, 3)`.

### 6.1 MEX (`api_mex/<name>.cpp`, `tests/mex_tests/test_<name>.m`)

```
phase = quadriga_lib.ssf_init_phases( scenpar, seed, set_id, preset );
rnd = quadriga_lib.ssf_draw( scenpar, phase, tx_pos, rx_pos, set_id, floor_id, dual_mobility, preset, horizontal );
[ delay, power, aod, eod, aoa, eoa, n_active ] = quadriga_lib.ssf_clusters( scenpar, tx_pos, rx_pos, lsp, rnd, ...
    method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps_re, gr_eps_im, o2i, n_cluster );
```

- `set_id` (MATLAB): 0 = shared cluster set (default), k = per-frequency set of the k-th frequency (1-based);
  C++ = MATLAB − 1; negative or non-integer → `Input 'set_id' must be 0 (shared set) or a frequency index >= 1.`
- `seed` default −1; `preset` default `'Comb256'`; `horizontal` default true; `dual_mobility` default false;
  `floor_id` as in `lsf_draw` (labels, no shift, the same conversion code); `method` default `'38.901'`.
- `o2i`: logical or numeric, `qd_mex_matlab2vector_Bool`, empty = none. `n_cluster`: a count (no shift), read as
  double, non-negative integers checked as `floor_id`, empty = none. Empty optional arrays = `nullptr`.
- Outputs: double arrays; `n_active` uint64 `[n_link, 1]`. Slices/columns 1-based in the documentation (slot 1 =
  LOS, slot 2 = GR if enabled; `rnd` column `b·N + n + 1`).
- Tests (MOxUnit, per SPEC_MATLAB_API): sizes and types; a value check against the C++-documented identity (e.g.
  `ssf_init_phases` slice 2 = mirror of slice 1; `ssf_draw` swap; `ssf_clusters` slot 1 angles = LOS geometry, NLOS
  scenario zero LOS power);
  optional arguments omitted vs `[]`; errors (zero links, bad `set_id`, missing `rnd_freq` for `38.901`).
  The `set_id` mapping is checked by sizes (MATLAB 0 → `9·N` columns, 1 and 2 → `N` columns) and by equality (MATLAB
  `set_id` 1 and 2 differ, equal calls agree); the exact seed derivation is checked in C++ and Python (MATLAB cannot
  pass the 63-bit test seeds exactly as a double literal).

### 6.2 Python (`api_python/qpy_<name>.cpp`, registered in `python_gbsm.cpp`; `tests/python_tests/test_<name>.py`)

```
phase = quadriga_lib.gbsm.ssf_init_phases(scenpar, seed=-1, set_id=-1, preset="Comb256")
rnd = quadriga_lib.gbsm.ssf_draw(scenpar, phase, tx_pos, rx_pos, set_id=-1, floor_id=None, dual_mobility=False,
                                 preset="Comb256", horizontal=True)
delay, power, aod, eod, aoa, eoa, n_active = quadriga_lib.gbsm.ssf_clusters(scenpar, tx_pos, rx_pos, lsp, rnd,
    method="38.901", rnd_freq=None, lsp_anchor=None, eod_offset_deg=None, abs_toa=None, gr_eps=None, o2i=None,
    n_cluster=None)
```

- Flags (`dual_mobility`, `horizontal`) via `qd_python_flag`: `None` = default, non-bool → `TypeError`.
- `gr_eps`: one complex array `(n_link, n_freq)` (as `lsp_generate` returns it with `complex=True`), split into
  re/im in the wrapper (`py::array_t<std::complex<double>, py::array::forcecast>` from the handle, then
  `qd_python_copy2arma` or an element loop into two `arma::mat`); a real array is accepted (imaginary part 0). No
  split inputs.
- `o2i`: any numeric or bool array, nonzero = True, converted in the wrapper to `std::vector<bool>`; `n_cluster`:
  read as double, non-negative integers (as `floor_id`). `None` = `nullptr`.
- `set_id`, `seed`: Python ints (`arma::sword`).
- Outputs: float64; `n_active` uint64 `(n_link,)`.
- Tests per SPEC_PYTHON_API checklist (basic call, keywords, each optional input, memory layout, dtypes, inputs not
  modified, flags, errors from C++/wrapper/pybind11) plus: zero links accepted; seed rule end to end:
  `ssf_init_phases(sp, 42, 1)[:, :, 0]` equals `quadriga_lib.sos.init_phases(256, N, 9018399064009245349)`;
  `gr_eps` given as complex (as `lsp_generate` returns it) and as a real array with the same real part (imaginary
  part 0): both accepted, the outputs differ; a negative imaginary part raises `ValueError` with the C++ message
  (`'gr_eps_im' >= 0`).


## 7. Documentation

- C++ blocks: as 3.1, 3.2, 3.3.5 (written by R1 and R2a, extended by R2b and R2c for their method). Formulas without
  `*` or `|` in running text (use `·` and `abs()`); the method × property table of 3.7 is a markdown table in the
  `ssf_clusters` block (tables are used in other blocks).
- MEX and Python blocks: adapted from the C++ blocks (R3), same facts, 1-based for MEX.
- `SECTION_DESC` of the section "Geometry-based stochastic model" (C++ in `src/qd_scenpar.cpp`, MEX in
  `api_mex/conf_file_read.cpp`, Python in `api_python/qpy_conf_file_read.cpp`): one new paragraph (5–8 lines) after
  the LSF data flow: the small-scale part implements TR 38.901 clause 7.5 steps 5 to 7 (cluster level); per site and
  scenario [[ssf_init_phases]] draws the phases of the shared set and of one set per frequency from the LSF seed;
  [[ssf_draw]] evaluates them; [[ssf_clusters]] turns the LSPs and the random values into cluster delays, powers and
  angles with one of three methods (one line each); the ray level follows in a later version. The first sentence of
  the existing text ("implement the large-scale part … steps 1 to 4") is extended to "steps 1 to 7 (cluster level)".
- `See also` lines: `lsf_init_phases` → `ssf_init_phases`; `lsp_generate` → `ssf_clusters`; the three new functions
  name each other and `lsp_generate`, `lsf_draw`, `calc_delay_spread`, `calc_angular_spreads_sphere`.
- `make documentation` regenerates `html_docu/*.html` (tracked); R3 runs it and lists the regenerated files as a
  separate commit theme.


## 8. Implementation runs

Five runs, one agent each: R1 → R2a → (R2b, R2c in either order) → R3. A run starts after the previous run's report
was accepted and the maintainer committed. Every run reads 1, 2, 9 and the sections named below, plus
`md/SPEC_CPP_API.md` (wrappers: the wrapper SPEC file). Each run writes `md/ssf_<run>_report.md` (e.g.
`md/ssf_r2a_report.md`) with: **Summary** (5 lines); **Files** (created/changed); **Decisions and deviations** (every
choice within the spec and every deviation, with the spec section; the SPEC review points answered here);
**Test results** (commands, the actual summary lines: Catch2 test cases and assertions, MEX per MATLAB/Octave
version, pytest summary, objdump gate output, load average before heavy runs); **Warnings**; **Open points**;
**Suggested commit cut** (themes, file lists, a commit message in plain physics without internal labels; the agent
does not commit). The report is a working document and is not referenced by any produced file.

### R1 — random fields (`ssf_init_phases`, `ssf_draw`)

Read: 1, 2, 3.0.9, 3.1, 3.2, 5.0, 5.1, 9; `src/qd_gbsm_phases.cpp`, `src/sos_init_phases.cpp`,
`tests/catch2_tests/test_gbsm_phases.cpp` (patterns).

Files: `include/quadriga_gbsm.hpp` (group comment, two declarations), `src/qd_gbsm_phases.cpp` (header comment,
`qd_gbsm_gen_ssf`, two functions with documentation blocks; the seed rule as a lambda), `tests/catch2_tests/test_ssf_phases.cpp`
(new), `tests/catch2_tests/qd_test_diag.hpp` (file list).

Steps: declarations and stubs (sizes, zeros), build; tests of 5.1; run against the stubs (record which checks fail
for the right reason); implement; iterate; final documentation blocks; full suite.

Acceptance: all new cases pass; `make test` passes; no warnings. Size: about 300 lines code + docs, 600 test.

### R2a — `ssf_clusters`: shared part, method `38.901`, D4 conf change

Read: 1, 2, 3.0, 3.3, 3.4, 3.7, 4, 5.0, 5.2, 5.5, 9; `src/lsp_generate.cpp` (validation and documentation style),
`src/quadriga_gbsm_helpers.hpp`, TR 7.5 steps 5–7, 7.6.8, 7.6.9, 7.6.15.

Files: `include/quadriga_gbsm.hpp` (declaration), `src/ssf_clusters.cpp` (new), `src/quadriga_gbsm_helpers.hpp`
(users list only), six conf files (4),
`tests/catch2_tests/test_ssf_clusters.cpp` (new), `test_conf_files_v19.cpp`.

Steps: declaration and stub; the complete validation of 3.0.2 for all methods (the `38.901_MF` and `quadriga`
branches throw `std::runtime_error("Method '<method>' is not implemented yet.")` after validation, replaced by R2b and
R2c); conf change and `test_conf_files_v19.cpp`; tests of 5.2 against the stub; implement the shared part and
`qd_ssf_tr`; documentation block (3.3.5 for the shared part and `38.901`, the table of 3.7 complete); full suite.

Acceptance: all new cases and the conf tests pass; `make test` passes; no warnings. Size: about 650 code + docs,
1,200 test.

### R2b — method `38.901_MF`

Read: 1, 2, 3.0, 3.3, 3.5, 5.0, 5.2 (fixtures, validation case), 5.3, 9; TR 7.6.5, 7.6.5.1; `src/ssf_clusters.cpp` as left by R2a.

Files: `src/ssf_clusters.cpp` (`qd_ssf_mf`, dispatch, documentation), `tests/catch2_tests/test_ssf_clusters_mf.cpp`
(new).

Acceptance: new cases and `test_ssf_clusters.cpp` pass; `make test`; no warnings. Size: about 150 / 350.

### R2c — method `quadriga`

Read: 1, 2, 3.0, 3.3, 3.6, 5.0, 5.2 (fixtures, validation case), 5.4, 9, 10.3; the paper (section II); NG `gen_ssf_parameters.m` QuaDRiGa branch
(read-only, `git -C /sjc/QuaDRiGa-NG show HEAD:quadriga_src/@qd_builder/gen_ssf_parameters.m`) for orientation
only — this spec is normative where they differ.

Files: `src/ssf_clusters.cpp` (`qd_ssf_quadriga`, dispatch, documentation),
`tests/catch2_tests/test_ssf_clusters_quadriga.cpp` (new).

Acceptance: new cases and the other `ssf_clusters` files pass; `make test`; no warnings. The report lists the measured
spread medians of the diagnostic case next to the study values of 3.6. Size: about 350 / 600.

### R3 — wrappers, documentation, gates

Read: 1, 2, 6, 7, 9, the three C++ documentation blocks, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`, the
wrappers and tests of `lsf_init_phases`, `lsf_draw`, `lsp_generate` (patterns).

Files: `api_mex/ssf_init_phases.cpp`, `ssf_draw.cpp`, `ssf_clusters.cpp`; `tests/mex_tests/test_ssf_*.m` (3);
`api_python/qpy_ssf_*.cpp` (3), `python_gbsm.cpp`; `tests/python_tests/test_ssf_*.py` (3); `SECTION_DESC` in
`src/qd_scenpar.cpp`, `api_mex/conf_file_read.cpp`, `api_python/qpy_conf_file_read.cpp`; `See also` lines (7);
regenerated `html_docu/*.html`.

Steps: MEX wrappers and tests; Python wrappers, registration, tests; `SECTION_DESC` and `See also`; `make
documentation`; gates (MEX in Octave and MATLAB R2024a — R2026a if available —, objdump, Python, full Catch2).
Release-note text in the report: three new functions, the D4 change of the six v19 LOS files (one more NLOS cluster
slot), the method defaults.

Acceptance: MEX suite passes in Octave and MATLAB R2024a; objdump gate shows nothing above `GLIBCXX_3.4.28`;
`make python` and the Python suite pass; `make test` passes; no warnings in any build. Size: about 600 / 400.


## 9. Constraints (into every run)

### 9.1 Code

- `double` only; C++17; no warnings with `-Wall -Wextra -Wpedantic`; MSVC must compile (no GCC extensions, no VLAs);
  OpenMP 2.0 only (signed loop variable, no exception leaves a parallel region); results independent of the number
  of threads.
- No BLAS/LAPACK in the library or the tests: no `arma::chol`, `solve`, `inv`, `norm`, `dot`, `accu(a % b)`; no
  Armadillo matrix product above 4×4 in tests.
- No `std::to_chars` / `std::from_chars` for floating point (MATLAB R2024a ships libstdc++ up to `GLIBCXX_3.4.28`).
- Helper scoping (SPEC_CPP_API): a helper used once is a lambda, except a self-contained algorithm of more than about
  20 lines (named file-local function directly above its caller's documentation block); helpers reused within one
  file `static inline` or in the anonymous namespace; only helpers used by several files go to
  `src/quadriga_gbsm_helpers.hpp` (none expected in this series). Reuse `qd_gbsm_*` machinery, `qd_interp_linear`,
  `qd_gbsm_check_links`, `scenpar_validate`; no parallel machinery.
- Lean code and API: no public function beyond the three, no new conf key, no new file type.
- No internal references in produced files (code, tests, wrappers, conf files, documentation blocks): no D/A/N/W
  labels, run names, spec section numbers, `md/` paths, local paths, NG line numbers. Cite TR 38.901 (clause, table,
  equation), the paper, or state the rule; `md/SPEC_*.md` may be cited.
- quadriga-lib must not depend on NG paths or data. NG, the TR and everything under `/sjc` are read-only.
- Random numbers only through the explicit seed; no global mutable state.
- Agents do not commit, push or merge, and do not edit this spec (proposals go into the report).

### 9.2 Resources and commands

- Shared machine: check `cat /proc/loadavg` before heavy work; at most 12 threads (`cmake --build build_linux
  --parallel 12`, `CMAKE_BUILD_PARALLEL_LEVEL=12`, `OMP_NUM_THREADS=12` or fewer); never two builds or test suites
  at once; heavy tools once on the final code.
- Catch2: `make test`; iterate with `cmake --build build_linux --parallel 12 && build_linux/test_bin "ssf_*"`;
  diagnostics `QD_DIAG_GBSM=1 build_linux/test_bin "ssf_clusters*"`.
- MEX (R3): build in a scratch build directory outside the repository (e.g.
  `/home/stephan/quadriga-lib-agents/scratch/ssf_r3/build`) with `-DCMAKE_CXX_COMPILER=g++-12
  -DCMAKE_C_COMPILER=gcc-12` (MATLAB's libstdc++); gate `objdump -T +quadriga_lib/*.mexa64 | grep -o
  'GLIBCXX_3\.4\.[0-9]*' | sort -u`; MOxUnit from `external/MOxUnit-master/MOxUnit` in a scratch working directory.
- Python: `make python`, `python3 -m pytest tests/python_tests/test_ssf_*.py -x -s`, `PYTHONDONTWRITEBYTECODE=1`.
- Scratch files, probes and baselines outside the repository (`/home/stephan/quadriga-lib-agents/scratch/`);
  never under `/sjc`.


## 10. NG integration hand-off (out of scope for R1–R3)

### 10.1 What NG replaces

- Steps 5–7 of `gen_ssf_parameters` (both paths: `use_3GPP_baseline` → `38.901`, QuaDRiGa path → `quadriga`;
  `38.901_MF` as a new option) by `ssf_clusters`; `taus`, `pow`, `AoD`, `AoA`, `EoD`, `EoA` from its outputs (NG
  stores `taus` `[n_rx, L]`: take slice 0 of `delay`; for `38.901_MF` in LOS the delays and angles are per frequency,
  which needs NG's per-frequency builders of `split_multi_freq`).
- Keep in MATLAB until the ray series: `xprmat`, `pin`, `subpath_coupling`, `get_subpath_angles`, `gen_fbs_lbs`,
  `get_channels`. The sub-cluster split and the "two strongest" selection read the output powers (no sort in the
  library).
- Ground reflection (W7, consumer contract of 2.3). The stored `pow` holds coefficient powers in every method: LOS
  K/(1+K), GR K/(1+K)·(d3D/d_GR)²·R² (TR methods) or K/(1+K)·R² (`quadriga`). Today NG stores the split
  ((1 − R²)·K/(1+K), R²·K/(1+K)) and undoes it in `get_channels` (`Rsq = 2·P_GR/(P_LOS + P_GR)`, `gSv = sqrt(2/Rsq)·
  R_par`, `Sl = 1/sqrt(1 − Rsq/2)`). At integration NG sets `Sl = 1` for every method and uses diag(R∥, −R⊥)/R with
  R² taken from the permittivity (or `P_GR/P_LOS` for `quadriga`). NG's channel is then unchanged: a NumPy check of
  the narrowband LOS + GR coefficient (V and H polarisation, 4 geometries 40 m–10 km, K 0–15 dB, ε = 13.5 − 1.18j)
  gives the same complex values as today's `get_channels` formulas to 2.5e-16; only the stored `pow` arrays differ
  (LOS K/(1+K) instead of (1 − R²)·K/(1+K), sum above 1). NG computes R with ε = re + j·im (the conjugate of the TR
  sign the contract prescribes); keeping NG's own R keeps its channel bit-compatible, the phase difference is that of
  a conjugated small imaginary part.
- Seeds: one LSF seed per builder, i.e. per scenario and site (W21; TR 7.6.3.4); the SSF sets derive from it.
- Builder: store the LSF seed (drawn once with `floor(rand·2^52)` if the user gives none; today drawn but not
  stored) and the SSF phase cubes (shared set + one per frequency) instead of `path_sos`; `xpr_sos`, `pin_sos`,
  `clst_dl_sos` stay until the ray series (`clst_dl_sos` only for the deprecated mmMAGIC method).
- Per call: `rnd = ssf_draw(...)` at the builder positions with the builder's `dual_mobility`, `floor_id`,
  `horizontal`; `38.901` under spatial consistency: evaluate blocks 1–8 at the segment start and reuse them for all
  snapshots of the segment (D5); `abs_toa`, `eod_offset_deg`, `gr_eps` from `lsp_generate`; `o2i` true for the O2I
  builders; NLOS builders pass their own scenpar (LOS by `KF_mu`, W1).
- `use_3GPP_baseline` with `SC_lambda = 0` / `'Disable'`: the fields at 0.1 m replace NG's `rand` draws (different
  realisations, same statistics).

### 10.2 Conf files

The D4 change affects NG users of the v19 LOS files (one more NLOS cluster). NG's legacy files keep their meaning:
- Without GR, NG reads `NumClusters − 1` NLOS clusters, as the library: same count as today.
- With GR (`TwoRayGR`: 2, `3GPP_38.901_UMi_LOS_GR`: 13) NG counts the GR slot inside `NumClusters`, the library adds
  it on top (W15). At integration NG passes a scenpar copy with `NumClusters − 1` for legacy files with
  `GR_enabled` (TwoRayGR → 1, i.e. LOS + GR only, 3.0.9; UMi_LOS_GR → 12, i.e. 11 NLOS clusters as today). The v19
  rule (GR slot on top) stays.
- `Freespace`, `LOSonly`, `Null` (NumClusters 1): LOS path only (3.0.9), as NG's `n_clusters == 1` branch. For
  `TwoRayGR` with `quadriga` the library gives the coefficient powers (1, R²), NG's channel; with the baseline path
  (`38.901`) the TR form (1, (d3D/d_GR)²·R²) (NG's baseline channel has no d3D/d_GR factor).
- Legacy "NLOS" files with `KF_mu > -30` run the LOS branch (list in 3.0.4).

### 10.3 Changes of NG's default output (QuaDRiGa path → `quadriga`), measured in the comparison study

No parity test (D3). Delays, powers, DS, ASD, ASA unchanged except:
- D4 in the v19 LOS files: +1 NLOS cluster (UMi LOS: clusters within 10 dB 6 → 7, KS 0.005).
- Additive LOS application: ESD median ratio 1.22 → 1.00 (UMi LOS), 1.09 → 1.00 (UMi NLOS), ESA up to −4 %;
  multi-frequency ESD at 2 GHz 1.30 (p90 3.31) → 1.03 (p90 1.21); NLOS arrival centre up by 2.1° (UMi NLOS),
  4.5° (UMi LOS).
- No satellite fade: UMa NLOS arrival centre up by 7.3° in total (with the additive application); D2D reciprocity
  restored for height differences > 10 m in NLOS.
- EoD offset on NLOS departures: departure centre moves by the TR offset (median 1.9° UMi NLOS, 3.0° UMa NLOS).
- Per-link worst-spread change median 0.004–0.093 in lg (largest: ESD in UMi LOS); KS distance 0.006–0.058.
- GR in NLOS scenarios: no pull of the NLOS EoA centre towards the GR elevation (W20; NG moves it by 10–23° in UMa
  at 35–100 m).
- Smaller differences: GR adds a slot instead of taking one of `NumClusters` (W15, handled by the scenpar copy of
  10.2); no fixed angles for 1–2 NLOS
  clusters (W14); no power sort when SC is off (W3); abs TOA shift only where Δτ > 0 (W13); 2D fields with floors
  instead of 3D (new realisations); per-frequency EoD offset averaged (W5).

Baseline path (`use_3GPP_baseline` → `38.901`): the TR angle mapping replaces NG's alternative mapping (N2), C_τ
applied (N1), −25 dB removal (N5), ZOD offset (N4), N + 1 slots in LOS (N3), LOS power exactly 0 in NLOS; NG unit
tests that inspect the old baseline angles (`testBuilder_SSF_generation*`, `3GPP_baseline_multifreq`) are rewritten
against the library.

### 10.4 NG tests

`testBuilder_SSF_generation_dual` (exact swap reciprocity) must pass with `quadriga` and with `38.901` for offset 0;
`3GPP_baseline_multifreq` now errors for `38.901` with several frequencies (use `38.901_MF` or `quadriga`).
