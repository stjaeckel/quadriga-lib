# SSF series 1 (cluster level): maintainer notes

What is worth keeping from the first small-scale fading series (TR 38.901 v19.5.0 clause 7.5 steps 5–7) after its working documents (plan, analysis, literature scan, comparison study, spec, reviews, run reports) were deleted. Internal document; IDs: D = decision, A = TR ambiguity, N = finding on QuaDRiGa-NG (commit 9a2beeb), W = design choice of the implementation. Section numbers in parentheses refer to the deleted spec; the facts themselves live in the doc blocks and source comments of `src/qd_gbsm_phases.cpp` and `src/ssf_clusters.cpp`.

## 1. Status

- Built on v0.12.4. Public API: `ssf_init_phases`, `ssf_draw` (`src/qd_gbsm_phases.cpp`), `ssf_clusters` with methods `38.901` (default), `38.901_MF`, `quadriga` (`src/ssf_clusters.cpp`); MEX and Python wrappers; the six v19 LOS files hold N + 1 clusters.
- Process: R0 study → spec (2 review rounds, round 3 skipped) → runs R1, R2a, R2b, R2c, R3 → code review 1 (1 bug, 3 major, 8 minor) → fixes → full code review 2 (0 bugs, 2 major doc, 6 minor) → fixes. Gates at the end: Catch2, ASan/UBSan, Octave, MATLAB R2024a/R2026a, Python, objdump (≤ GLIBCXX_3.4.26).
- NG integration done (NG commits 6c30173 to bb594cd on e318034, section 6). Next: the ray series.

## 2. Comparison study (R0): key numbers

Prototype (NumPy, not kept), UMi/UMa LOS/NLOS at 6 and 30 GHz, 4000 links per case, multi-frequency UMi 2/6/30 GHz, tracks 30 × 40 m at 0.01 m steps, D2D 2000 pairs. Median output/input spread:

| Method | NLOS | LOS | Clusters within 10 dB (UMi NLOS, N = 19) | Continuity |
|---|---|---|---|---|
| `38.901` | 0.96–1.02 | ASA 0.80–0.89 (K cap), others ≈ 1 | 13 | LOS 0.03–0.11 jumps/m (first cluster changes); sign flips between UTs 0.5–1.5 per m; −25 dB toggles 0.03–0.19 /m |
| Procedure B (dropped) | 0.47–0.74 | 0.14–0.19 | 2 | continuous |
| `38.901_MF` | 0.83–0.95 | 0.42–0.77 | 8 | LOS 0.05–0.07 /m (wrap before √(1+K)) |
| NG QuaDRiGa (faithful) | DS exact, others ≈ 1; ESD 1.09–1.22 | UMa ASA 0.59 | 13 | nearly continuous |
| `quadriga` (redesign) | 1.00 | UMa ASA 0.59, UMi ASA 0.86 | 13 | 0 jumps |

- `38.901` per frequency moves delays by 244–313 ns between 2 and 30 GHz (why it is single-frequency only).
- Not adopted (measured worse): per-cluster ζ in `quadriga` (within 10 dB 13 → 10), Gaussian/Laplacian initial values (UMi LOS ASA 0.86 → 0.64), exact wrapped AS solve (jumps), exact per-frequency scaling (2↔30 GHz power correlation 0.86 → 0.67).
- `quadriga` vs `38.901` differences that remain: angle–power rank correlation −0.33 vs −0.84; UMa LOS ASA at moderate K 0.61 vs 0.83. Antenna-level effect unknown until the ray series.
- TR ground reflection adds +1.6 dB of link power (median) and gives `38.901` DS 0.85 in UMi LOS.
- Re-measured in NG (1000 links per case, 3.5/6/30 GHz; section 6): the `38.901` LOS ASA range does not hold for UMa LOS (0.65–0.67 with NG's legacy file, 0.78 with v19: large input ASA at K ≈ 9 dB), and NLOS DS reaches 0.95–0.97; an independent MATLAB TR 7.5 oracle gives the same values (library within ±0.02), so both are TR-literal.

## 3. Decisions

| ID | Decision |
|---|---|
| D1 | Methods `38.901` (default), `38.901_MF`, `quadriga` (redesign of NG's QuaDRiGa method). Procedure B (7.6.3.2) dropped after the comparison study: as written it gives 0.47–0.74 (NLOS) and 0.14–0.19 (LOS) of the input spreads with 1–2 dominant clusters; the append-only random-field layout keeps room for it. 7.6.5.1 stays TR-literal, including the order "wrap, then scale by √(1+K)" and the LOS jumps this causes. Default revisited after the ray series with antenna-level metrics. |
| D2 | Mirrored random fields in all methods (D2D reciprocity of the random inputs). `38.901` single frequency only (error otherwise). Spatial consistency allowed with `38.901`, its jumps documented. TR BS-specific terms (μ_offset,ZOD, cluster ZSD, O2I ZOA = 90°) kept as written and documented. A method × property table in the docs. |
| D3 | `quadriga` = NG's QuaDRiGa cluster generation (−ln U delays, uniform initial angles, power product with the paper's g fits, analytic DS scaling, NG's iterative linear AS scaling averaged over frequencies, caps 3 / 1.5, NG ground-reflection power split, no per-cluster shadowing) with three changes: LOS direction applied additively (azimuth wrap, elevation pole fold) instead of the 3-D rotation; μ_offset,ZOD added to the NLOS departure elevations; no satellite EoA fade. Not adopted: per-cluster ζ, Gaussian/Laplacian initial values, exact wrapped AS solve, exact per-frequency scaling. No faithful port, no NG parity test; NG's output changes documented (10.3). |
| D4 | `NumClusters = N + 1` in every file: slot 0 = LOS ray (zero power in NLOS), then N NLOS clusters; in TR LOS the first NLOS cluster is at τ = 0, aligned to LOS (7.5-12/-17). +1 in `NumClusters` and `NumClusters_min` of the v19 LOS files (6 files; the plan's count of 8 is corrected in 4); NG's legacy files keep their meaning. |
| D5 | C_φ/C_θ by linear interpolation between table entries (exact at table points, clamped outside); −25 dB removal sets the power to 0, keeps the slot, no renormalisation, only in `38.901`; C_τ applied in `38.901` LOS; signs X_n and jitter Y_n spatially consistent, evaluated by the caller at the drop / segment-start positions and kept along the track; zenith wrap [180°, 360°] → 360° − θ for ZOA and ZOD. |
| D6 | Random fields in blocks by variable type, append-only (existing columns never move), ray blocks reserved; one shared cluster set plus one set per frequency (Z_n now, the per-frequency ray blocks later); seeds derived from the LSF seed by a fixed documented rule; no bias removal; 2D + floor IDs as the LSF, 3D legacy mode; `SC_lambda = 0` → 0.1 m. |
| D7 | Padded slots per link (0 LOS, 1 GR if enabled, then N); zero power for inactive entries (removal, smaller per-link N); delays in s relative to the LOS delay, NLOS + Δτ with abs TOA; frequency dimension of delays and angles only where a method needs it (7.6.5.1 LOS); powers per frequency as the coefficients use them (7.5-30: LOS K/(K+1), NLOS P_n/(K+1)); AoD, EoD, AoA, EoA in rad, GCS, elevation convention; active-cluster count per link. Inputs of the later ray step (cluster spreads, c_DS(f), per-link cluster ZSD, K) are read from `scenpar` / `lsp` / `lsp_mu`, not duplicated. |
| D8 | Scope and runs: cluster level only; runs R1, R2a, R2b, R2c, R3 (8). |

## 4. TR ambiguities and NG findings

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

NG findings resolved here: N1 (C_τ) applied; N2 (baseline angle mapping) replaced by the TR mapping; N3 (LOS cluster count) by D4; N4 (ZOD offset never applied) applied in `38.901` and `quadriga`; N5 (−25 dB) applied in `38.901`; N8 (multi-frequency in baseline mode) by `38.901_MF`/`quadriga`; N10 (frequency-averaged scaling) kept by design; N11 (satellite fade) dropped; N12/N13 (SSF fields 3D, seed not stored) by the 2D + floor fields and the derived seeds; N16 (GR power split) kept inside `quadriga`'s spread scaling. N6, N7, N9, N14, N15 belong to the ray series.

## 5. Design choices (W)

| ID | Choice | Reason |
|---|---|---|
| W1 | The LOS condition is a property of the scenario: LOS when `KF_mu > -30` (dB). In an NLOS scenario the LOS slot has zero power whatever KF `lsp` holds (1e-10 from `lsp_generate`). | NG uses one scenario (conf file) per builder and LOS state; the plan lists no per-link LOS input; the TR 7.5 formulas need a hard LOS/NLOS switch (C_τ at K = −100 dB is meaningless); a per-link K threshold would jump under SC. The threshold mirrors the `absTOA_mu > -30` rule; all v19 NLOS/O2I files have −100, all LOS files ≥ 7. Consequence (documented, 3.3.5, 10.2): legacy NG "NLOS" files with KF_mu > −30 run the LOS branch. Accepted by the maintainer. |
| W2 | In 7.5-3, 7.5-10, 7.5-15 the K-factor is limited to K ≥ −5 dB. | C_θ of 7.5-15 falls to 0 at K = −10 dB (−0.0004) and turns negative below (−1.61 at −15 dB), which inverts or explodes the zenith angles; at −5 dB C_τ = 0.990 and C_θ = 0.922, near their NLOS limit 1 (C_τ = 1 at −5.2 dB, C_θ = 1 at −4.4 dB), while the LOS factor of C_φ (7.5-10) is 1.181 (it crosses 1 only at −11.7 dB). Cost, measured by the reviewer for K ≤ −8 dB: medians DS 0.94–0.95, ASA 0.86–0.88, ZSA 1.03–1.09 of the input (NLOS limit DS 0.94, ASA 1.02, ZSA 0.95); without the limit ZSA 7.6 at −10 dB; with NG's rule (factor ≥ 1 for K < 0) DS 0.6–0.8. Links below −5 dB: InF LOS 6.7 %, SMa LOS 2.3 %, UMi LOS 0.26 %, InH/RMa LOS 0.13 %, UMa LOS < 0.01 %. Accepted by the maintainer; the ASA effect is documented. |
| W3 | Slots keep the cluster identity (field column) in every method; the "first" cluster of TR LOS is the one with the smallest initial delay (lowest index on ties). | The TR sort only relabels exchangeable clusters; identity keeps SC continuous and the field column of a cluster stable when N changes per link. NG sorts by power when SC is off; the ray step selects the strongest clusters by power. |
| W4 | The zenith fold of 7.5 (D5) is also applied to the final zenith angles of `38.901_MF`. | 7.6.5.1 gives zenith angles in (−180°, 360°) and states no rule; the fold is the only zenith convention of the TR; R0 used it. |
| W5 | `quadriga` with several frequencies adds the EoD offset averaged over the frequencies. | Delays and angles must be frequency-independent; the scales are averaged the same way. The v19 UMa NLOS offset depends on frequency. |
| W6 | `quadriga` ignores `o2i` (no 90° ZOA rule); `38.901_MF` ignores `o2i` and `eod_offset_deg`. | D3 lists exactly three changes to NG's method; 7.6.5.1 has neither term (A4). |
| W7 | Ground reflection is switched on by the scenario key `GR_enabled` (> 0). Maintainer decision: every method outputs coefficient powers (D7) — LOS slot P_LOS = K/(K+1), GR slot TR methods P_LOS·(d3D/d_GR)²·R², `quadriga` P_LOS·R², with the polarisation-averaged R² = (abs(R∥)² + abs(R⊥)²)/2; NG's (1 − R², R²) split is internal to `quadriga`'s DS/AS scaling. One consumer contract for all methods (2.3): amplitude √(slot power), GR coefficient with diag(R∥, −R⊥)/R. | The key exists and NG uses it. Coefficient powers make one contract possible; NG's channel stays unchanged (NG drops its LOS factor and divides R out, 10.1; NumPy check: identical to 2.5e-16). Cost: the DS measured from the `quadriga` slot powers of a GR link is 0.85 of the input (3.6). |
| W8 | Seed rule: SplitMix64 of the LSF seed and a set tag (2.2.4); MATLAB `set_id` 0 = shared set, k = frequency k (1-based). | Fixed, platform-independent, independent streams per set; the column-of-one-stream variant would tie the per-frequency sets to the size of the shared set and break append-only. |
| W9 | One layout serves all methods: delay block Uniform, sign and offset blocks Normal; `ssf_clusters` maps Normal → Uniform with Φ where a method needs a uniform. | Same field realisation for every method; the map is the one of `ssf_draw`. |
| W10 | `NumClusters >= 1`; `NumClusters = 1` (N = 0) gives the LOS path only (full power, K ignored) plus the GR slot if enabled; the random-field sets are then empty. Maintainer decision. | NG has LOS-only files (`Freespace`, `LOSonly`, `Null`: NumClusters 1) and a LOS + GR file (`TwoRayGR`: 2 with GR, read with NumClusters − 1, 10.2); one rule serves them in all methods. |
| W11 | `n_cluster` must lie in [min(max(1, `NumClusters_min` − 1), N), N], N = `NumClusters` − 1 (all 0 when N = 0). | Uses the TR range where the file states it (7.6.15), any count otherwise. |
| W12 | Slots beyond `n_cluster(l)` get delay 0, power 0 and the LOS angles. | Every output element is written; zero-power entries with valid angles are harmless for consumers. |
| W13 | Absolute TOA in NLOS scenarios only (TR 7.6-43/-44, all methods): TR methods add Δτ; `quadriga` shifts the NLOS delays to start at Δτ. Changed after code review 1 (M1, maintainer decision): `quadriga` requires `abs_toa` in NLOS scenarios with `absTOA_mu > -30` (missing → `std::invalid_argument`, the NLOS clusters must start at the calibrated excess delay of TR 7.6.9, otherwise the origin `s·min(τ̃)` is arbitrary for few clusters and scatterers derived from the clusters can collapse onto the arrays) and then always shifts, also for Δτ = 0; NLOS scenarios without `absTOA_mu > -30` (legacy files) keep the shift only where `abs_toa(l) > 0`; `abs_toa` is ignored in LOS scenarios. TR methods unchanged (7.6.9 optional). | TR-literal; in a LOS scenario the shift would break the exact DS of `quadriga` (measured: DS ratio 1.00 → 1.10 for Δτ = 30 ns, K = 9 dB). NG applies the shift whenever an offset exists; `lsp_generate` returns 0 for the LOS files, so NG with v19 files is unaffected. |
| W14 | No special case for 1 or 2 NLOS clusters in `quadriga` (NG: fixed ±40.5°). | Fields work for any N; only the deprecated mmMAGIC file has N = 2. |
| W15 | With GR the slot count grows by one; NG takes the GR slot from `NumClusters`. | D7 layout (0 LOS, 1 GR, then N): the number of NLOS clusters does not depend on the GR switch. |
| W16 | `lsp_mu` is not an input of `ssf_clusters` (the plan lists it). | μ_lgZSD enters only 7.5-20, the ray ZOD spread (ray series). |
| W17 | The D4 conf change is done in R2a, not R3 (plan). | R2a's LOS TR-oracle tests read the v19 LOS files with the TR N typed in; they need N + 1 in the files. |
| W18 | Numerical robustness: cluster powers normalised in the log domain in all three methods (3.0.10); ln(max(u, DBL_MIN)) for uniform delays; ln(max(P, DBL_MIN)) in the 7.5 angle mapping; 7.6.5.1's Laplacian term through the identity 1 − 2·abs(V − 0.5) = erfc(abs(x)/√2) with a DBL_MIN floor; d_GR floored at 1e-5 m; the GR delay in the cancellation-free form 4·h_t·h_r/((d_GR + d3D)·c); permittivity validated (real part ≥ 1, imaginary part ≥ 0) and the only possible 0/0 of the reflection coefficients (ε = 1 + 0j at grazing) set to 0, the value of ε = 1 at every other geometry; `quadriga`: delay scale 0 where the cluster delays have no spread, angle scale 1 where an angular spread is not > 0. | No NaN or Inf for any finite random values (also far outside the ±√(2·n_coeff) range of `ssf_draw`) and LSPs in the ranges `lsp_generate` produces; identical results wherever the floors do not act. |
| W19 | `ssf_clusters` tests are split into three files (one per method run). | R2b and R2c may run in either order without editing the same test file. |
| W20 | `quadriga` in an NLOS scenario uses w = 0 for the elevation centres (no pull towards the GR elevation). | The GR slot has zero power there; NG's K-independent weight would move the NLOS EoA centre down by 10–23° (UMa 25/1.5 m, d2D 35–100 m) for a reflection that does not exist. LOS behaviour (D3) unchanged. |
| W21 | The LOS, NLOS and O2I scenarios of a site need different LSF seeds (and hence different SSF streams). | TR 7.6.3.4 makes the link types uncorrelated; one seed would give identical SSF fields in the LOS and NLOS scenario. NG already draws one seed per builder. |
| W22 | A consumer without polarisation forms the scalar GR coefficient √(GR slot power)·exp(j·arg(R∥)) (arg(0) = 0). | The maintainer asks that it keep the reflection phase (≈ π near grazing; dropping it reverses the two-ray interference, e.g. +5.8 dB instead of −8.6 dB at 10 km); R∥ is the coefficient of the vertical (θθ) polarisation, the usual base-station case; a consumer with a known polarisation uses its own entry of diag(R∥, −R⊥)/R. |

### Out of scope and interface readiness

Out of scope (later series): the ray level (ray offsets of Table 7.5-3, sub-clusters of Table 7.5-5, coupling, XPR, initial phases, 7.6.2 large bandwidth, 7.6.10 mobile scatterers, 7.6.13 near field, 7.6.14 SNS, 7.6.16 polarisation variability), Procedure A and B (7.6.3.2), drifting, scatterer geometry (NG's `gen_fbs_lbs`), path storage in QRT files, calibration against 3GPP (needs the ray step and antennas), converters to `get_channels_*`.

Readiness:
- Random fields: append-only blocks (2.2.5). The shared set takes later frequency-independent cluster blocks (e.g. the Procedure A sign, Procedure B blocks at its own decorrelation distances, the 7.6.13 s_TRP draw, SNS x_n); the per-frequency sets take the ray blocks of 7.6.5 "steps 8–11 independently per frequency" (coupling, XPR, phases, η, 7.6.2.2 offsets, 7.6.10 α and D). Existing columns and their values never change; a stored seed reproduces the clusters of this version after a library update.
- Outputs: everything the ray step needs is in the cluster output (slot identity incl. LOS and GR, delays, powers per frequency, angles) or in existing outputs: K from `lsp`, μ_lgZSD for 7.5-20 from `lsp_mu` slice 5, cluster spreads, c_DS(f), M from `scenpar`, the GR polarisation from the permittivity under the consumer contract of 2.3 (the slot powers are coefficient powers; the GR slot already contains R²; the ray step applies diag(R∥, −R⊥)/R). Procedure A needs the absolute cluster state at the segment start (this output plus d3D/c) and an appended sign block.
- `ssf_clusters` validates the exact sizes of this version's layout; a later version that appends blocks changes the size rule together with the layout.

## 6. NG integration (done)

NG commits on e318034: 6c30173 (generator state), 15c6590 (`quadriga`, coefficient powers, GR consumers), 0840170 (`38.901`), 15c24c5 (`38.901_MF`), bb594cd (review fixes). The commit messages list the behaviour changes with numbers. NG has no notes of its own; this section is the record.

### How NG uses the library

- `qd_builder.lsf_generator.seed` (stored now) and `qd_builder.ssf_generator` (`shared` `[n_coeff, 9·N, 2]`, `freq` `[n_coeff, N, 2, F]`, `set_id`); drawn in `init_sos`, carried by `copy`, `split_rx`, `split_multi_freq` (frequency f keeps set `set_id(f)`, also after a redraw), `add_paths`; `path_sos` removed; `xpr_sos`, `pin_sos`, `clst_dl_sos` stay until the ray series. Builders saved before the series without `seed` get a new seed for the SSF sets only.
- One `ssf_draw` per set and one `ssf_clusters` per builder (`private/ssf_eval.m`); `lsp` from the builder properties (user edits honoured), clamped at the call (K ≤ 1e10, spreads ≥ 1e-6°, DS ≥ 1e-12 s, non-finite SF/XPR → 1); `eod_offset_deg` from `lsp_generate`; `abs_toa` = `absTOA_offset` (empty → `absTOA_mu = -100` in the scenpar copy, no shift); `'Disable'` → `SC_lambda = 0`; `floor_id` not used (UTs at the same x-y share fields, as in the LSF port).
- Method: QuaDRiGa path `quadriga`; 3GPP baseline `qd_simulation_parameters.ssf_model` = `38.901` (default) or `38.901_MF`. `38.901` with several frequencies errors (builder and `qd_layout.get_channels`); `38.901_MF` runs per split builder (7.6.5.1 is exactly separable per frequency given the anchor and the frequency's set; measured difference 0) with anchor LSPs at 2 GHz from the builder's generators; an unsplit multi-frequency builder errors.
- Conventions: `NumClusters` counts LOS and GR, so NG passes `NumClusters − 1` whenever `GR_enabled` (every scenpar); O2I = scenario name ends in `_O2I` (all O2I files have `KF_mu = -100`; a `'Custom'` scenpar is not O2I); stored `pow` = coefficient powers (LOS + NLOS = 1, GR on top), `gen_ssf_parameters` writes `gain` directly, the `pow` setter normalises all slots except the GR slot (found by geometry); NLOS scenarios keep a zero-power LOS path.
- GR: polarisation matrix diag(R∥, −R⊥)/R in both paths; `get_channels` uses R(t)/R0 with R0 at the segment start and `p_los = pow(:,1)` in the KF scaling; `gen_lsf_from_ssf` inverts K, SF and ε for both power forms. NG keeps its own permittivity sign (ε = re + j·im, the conjugate of the TR sign; same |R|).
- v19 conf files: NG's CMake copies `share/quadriga-lib/config` to `quadriga_lib/config`, searched after NG's own `config` folder.

### Measured in NG (1000 links per case; e318034 vs bb594cd)

- Spherical GR channel unchanged (7e-16 on a 40 m track, V and H). Baseline GR/LOS coefficient = TR eq. (7.6-32) to 2e-15; before, NG's baseline had an inverted H sign and wrong magnitudes (pre-existing bug, fixed).
- `quadriga`: DS, ASD, ASA unchanged; 1-D ESD/ESA match the input exactly (before up to +11 % median, +167 % p90; the old excess shrinks with distance: UMi LOS 1.39/1.10/1.02 for 35–100/100–200/200–500 m, which explains the study's 1.22 for near links); NLOS EoA centre at the LOS elevation (±0.4°; UMa NLOS +6.6–7.0°, +20° at 35–100 m); EoD centre moves by the TR offset minus 0.4–0.7° (additive LOS application); multi-frequency ESD at 2 GHz 1.10 (p90 1.88) → 1.00 (p90 1.11). NG's spherical estimator `qf.calc_angular_spreads_sphere` moves the other way (ESD 1.00 → 1.05–1.13, p90 up to 4.9 at tiny input ESD; ESA up to 1.34 in UMa NLOS at 35–100 m).
- `38.901`: library = oracle within ±0.02; LOS DS 0.45 → 1.0 (C_τ); −25 dB removal on 21–60 % of links; D2D swap reciprocity exact for all three methods (before: baseline not reciprocal, QuaDRiGa path broken above 20 m height difference).
- `38.901_MF`: the TR LOS files have no frequency-dependent K, so LOS delays and angles are identical at all frequencies too (mmMAGIC UMi LOS with frequency-dependent K: median delay ratio 30/2 GHz 1.52).
- abs TOA: v19 NLOS clusters start exactly at Δτ; DS unchanged. "+1 cluster in the v19 LOS files" is no change in NG (NG first read the v19 files with N + 1 already).
- Statistics from one builder over a small area are dominated by the correlated cluster fields (now also in the baseline): one builder with 100 links within 200 m gave v19 UMa NLOS DS 0.90; 20 builders 0.96.

### NG items not fixed (pre-existing)

- Baseline `get_channels_seg` for part of a multi-segment track fails ("matrix cannot be indexed with ." at `par_tmp.pg`), also at one frequency.
- An NLOS scenario with `GR_enabled` makes `gen_lsf_from_ssf` clear `gr_epsilon_r`; the next `gen_ssf_parameters` fails (no shipped file has this combination).
- Changing `NumClusters` after `init_sos` under spatial consistency fails on the `xpr_sos` size check.
- NG's CMake relocate step is not idempotent: changing its commands after an install re-runs it without the install step and it deletes `+quadriga_lib`; remove the install and relocate stamps and rebuild.
- Builders saved in MATLAB before the series with GR carry the old LOS/GR power split (LOS lower by 1 − R² in the new `get_channels`); regenerate their SSF.
- The global MATLAB R2024a `pathdef.m` lists `/sjc/quadriga-lib`: when MEX files are built there, MATLAB uses them instead of NG's package (NG then sees 92 instead of 115 conf files); add NG's `quadriga_lib` first.

## 7. Open items for later series

- Revisit the default method with antenna-level metrics (coupling loss, singular values of 7.8.2) once the ray step exists; `quadriga`'s weak angle–power coupling and the UMa LOS ASA gap are the reasons it is not the default.
- Scatterer series: minimum-distance guard for clusters whose delay is short for their angles. Measured in NG's `gen_fbs_lbs` (unchanged by the integration): scatterers within 1 m of the Rx for 0.2–0.8 % of NLOS sub-paths in the legacy files (10–29 % of links have one), ≤ 0.04 % in v19 NLOS (abs TOA); all at the Rx in the single-bounce fallback, 73–97 % with excess path < 0.01·d3D and angle > 30° from the LOS.
- `quadriga` with only 2–4 active clusters in LOS can place a weak cluster far out (RMa: delay span up to ≈ 700·DS vs 150·DS for `38.901`); in the TR cluster-count ranges the tails match `38.901`. Recorded, no action.
- `quadriga` with GR in LOS: the ESD/ESA target is often unreachable because the fixed LOS and GR elevations already spread more than the target (NG behaviour, documented); with GR the slot-power DS is ≈ 0.85 of the input.
- Procedure B can be added later (append-only random-field blocks); as written it does not reproduce 7.5.
- Pre-existing, unrelated: MEX `test_ray_mesh_interact` line 113 fails (transmission gain 0.985 vs 0.696); the test and `src/ray_mesh_interact.cpp` are unchanged since v0.12.4.

## 8. Lessons learned

- Measure before deciding: R0 overturned several plan assumptions (Procedure B quality, ζ, Gaussian initial values) and the reviewers' measured inputs found the real bugs (B1: a unitless fallback scale of 1 gave delays in seconds; M1: NG's NLOS delay origin, which also explained a scatterer singularity downstream).
- Delay-origin rules must be checked against the downstream geometry step, not only against spread statistics: a cluster at delay 0 is harmless for antenna mapping (TR) but collapses scatterers onto the arrays (QuaDRiGa).
- Put the documentation rules (API only, each fact once, no formulas, size targets vs neighbouring functions) into the spec and every run brief from the start: the first implementation produced 1248 doc lines, cut to ≈ 500 later.
- Test-size estimates were exceeded 1.5–2× by every run; shared test headers and method-generic cases (all three methods in one case) should be specified up front.
- Two spec review rounds were enough when the TR transcription is checked numerically by each reviewer; the parts changed after the last review (GR contract, log-domain powers, abs-TOA rule) were caught by the implementation's TR-oracle tests.
- Effort: medium sufficed for the implementation runs with a detailed spec; high for the largest TR run (R2a) and the first code review.
- NG integration: three plan reviews were needed; the third, which built rough versions of the steps and ran NG's full suite against them, found no blocker or major — prototype runs belong in the first plan review. Check the brief's assumptions in the code first (NG had never read the v19 files; the `pow` setter renormalised).
- MOxUnit's `assertElementsAlmostEqual(..., 'relative', tol)` adds an absolute floor of sqrt(eps): relative checks on values near 1e-9 (path powers) pass vacuously; compare ratios to 1 with `'absolute'`. Mutant checks (revert a line, the test must fail) found two untested paths.
