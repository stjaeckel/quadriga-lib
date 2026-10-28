# SSF port: analysis of the current state (with the decision log)

Working document of the SSF series (branch `ssf`, from v0.12.4). Analysis and decision log; the final plan is
`md/ssf_plan.md`. Sections 6 to 9 below are the pre-decision draft; where they differ, section 0 and the plan win. Deleted before the merge.

Sources: TR 38.901 v19.5.0 (`/sjc/references/3gpp_38901/38901-v19.5.0.md`), TR 38.901 v16.1.0 (PDF, for the diff),
QuaDRiGa-NG HEAD `9a2beeb` (read with `git show`), the 2019 paper (`/sjc/QuaDRiGa/publications/2019_multifreq_d2d_ssf`),
quadriga-lib v0.12.4.

Labels: **[TR]** TR text, **[NG]** NG code, **[P]** paper, **[L]** library, **[I]** my inference or proposal.
IDs: A = TR ambiguity, N = NG deviation or suspected bug, P = policy proposal, D = decision for the user.

## 0. Decisions and scope changes so far

- **D1 decided: all four methods** for steps 5–7 (`38.901`, `38.901_B`, `38.901_MF`, `quadriga`), so they can be
  compared. The default is **not decided yet**: it follows from the comparison study below.
- **Framing (user):** a *baseline path* close to TR 7.5 without optional features (the calibration target of 7.8.2),
  and a *consistent path* that follows the paper and covers SC, MF and D2D together. Open question: how far the
  consistent path diverges from each optional TR feature, and whether it can be extended (e.g. per-cluster
  shadowing, Gaussian/Laplacian initial values, ZOD offset as a rotation) so that it reproduces the baseline
  statistics. If it does, it can become the default and the baseline formulas stay only for calibration.
- **New plan item: comparison study before the default is set** — a literature scan (papers, 3GPP contributions on
  SC × MF, Procedure B, 7.6.5.1, D2D) and a scratch prototype that compares the four methods on cluster-level
  metrics (section 8.1).
- **D2 decided:** mirrored random fields in all methods; `38.901` single frequency only (error otherwise); SC
  allowed with `38.901`, jumps documented; D2D reciprocity through the random fields, TR BS-specific terms (ZOD
  offset, cluster ZSD, O2I ZOA = 90°) kept as written and documented; a method × property table in the docs.
- **D3 deferred to R0, lean: redesign.** R0 prototypes both a faithful `quadriga` (NG's cluster generation incl.
  frequency-averaged scaling, iterative AS loop, satellite EoA fade, GR power split, no ZOD offset) and a
  redesigned consistent method (candidates: per-cluster shadowing ζ, Gaussian/Laplacian initial values mapped from
  Normal fields, ZOD offset as rotation, TR GR, per-frequency exact scaling). R0 measures the trade-offs; the user
  then picks one (or keeps both).
- **D4 decided: NumClusters = N + 1 everywhere.** +1 in `NumClusters` and `NumClusters_min` of the 8 v19 LOS files
  (UMa/UMi/RMa/SMa/Indoor/InF LOS, …); slot 0 = LOS ray (zero power in NLOS), then N NLOS clusters; in TR LOS the
  first NLOS cluster is at τ = 0 aligned to LOS. NG's old files keep their legacy meaning.
- **D5 decided (batch):** C_φ/C_θ by linear interpolation between table entries, clamped outside; −25 dB removal
  zeros the power, keeps the slot, no renormalisation, only in `38.901`; C_τ applied in `38.901` LOS; signs X_n and
  jitter Y_n spatially consistent, evaluated by the caller at the drop / segment-start positions and frozen along
  the track (documented rule); the [180°, 360°] zenith wrap also for ZOD.
- **D6 decided:** variable-type blocks, append-only, ray blocks reserved; one shared cluster set plus one set per
  frequency (Z_n …) with seeds derived from the LSF seed by a fixed rule; no bias removal; 2D + floors, 3D legacy,
  SC off → 0.1 m; Procedure B delay/AOD/ZOD distance 2c·10^(μlgDS+σlgDS) from scenpar at a reference frequency
  and distance given at init (multi-frequency: the frequency with the largest μ + σ), AOA/ZOA 50 m, Z_n at
  `SF_lambda`.
- **D7 decided:** padded slots (0 LOS, 1 GR if enabled, then N), zero power for inactive entries; delays [s]
  relative to LOS, NLOS + Δτ with abs TOA, frequency dimension only where a method needs it; coefficient powers
  per 7.5-30 per frequency; angles in rad, GCS, elevation convention; active-cluster count per link; ray-step inputs
  read from scenpar / lsp, not duplicated. The NG integration replaces steps 5–7 only; NG's MATLAB ray code stays
  until the ray series.
- **D8 decided:** scope and runs as in `md/ssf_plan.md` (cluster level; R0, spec + 3 reviews, R1, R2a, R2b, R2c,
  R3, 3 code reviews, NG integration).
- **Comparison study = first run of the series (R0)**, own agent, scratch prototype, before the spec; the default
  method is decided on its result. Decisions that do not depend on the default continue now.
- **Literature scan done** (`md/ssf_literature.md`; 3GPP Tdocs known only by their CR records, not read). Key
  points: the only SC × MF rule in the TR is Procedure B's "largest spread over all frequencies" (since v14.0.0,
  still in v19.5); 7.6.5.1 came from Ericsson + Fraunhofer HHI at RAN1#88 (2017, motivated by the 7.6.5 note);
  Procedures A/B date from 38.900 v14.0.0 (2016), last changed by CR 0019 (HHI, 2017); no CR since Rel-15 touched
  steps 5–7, 7.6.3.2 or 7.6.5; Rel-19 ISAC (7.9.5.1) adds a "link-correlated" SC type for UT-UT links but no
  dual-mobility or reciprocity rule. Sionna keeps sorting and max-normalisation (jumps remain), ns-3 implements
  Procedure A, MATLAB has dual mobility but no SC or MF. **No published comparison of the baseline statistics
  against Procedure B, 7.6.5.1 or the paper's method exists**; the 7.8.3 SC calibration metric depends on the
  sort order. The comparison study (8.1) therefore has to produce this evidence itself.
- **Scope narrowed to the cluster level.** The ray level (ray offsets, sub-clusters, coupling, XPR, initial phases,
  7.6.2.2, 7.6.10 mobile scatterers, 7.6.13 NF, 7.6.14 SNS, 7.6.16, time variation of rays, drifting, storing
  physical paths in QRT files before the antennas) gets its own analysis in a later series. `ssf_rays` and
  `sns_element_gain` (section 6.2) are removed from this series. This series must still leave the interface ready:
  the cluster output carries everything the ray step needs, and the random-field layout reserves the ray-level
  blocks (appended later, existing columns never move).
- **After R0 (`md/ssf_r0_report.md`):** default method `38.901`; D3 = redesign only, named `quadriga` (NG's
  cluster generation + additive LOS application + additive ZOD offset in NLOS + no satellite fade; no ζ, no
  Gaussian/Laplacian initial values, no exact wrapped solve, frequency-averaged scaling kept); Procedure B dropped
  from the series (spreads 0.14–0.74 of the input). Methods: `38.901`, `38.901_MF`, `quadriga`. Details in the plan.

---

## 1. What the TR describes

### 1.1 Steps 5 to 11 (clause 7.5)

Inputs per link from step 4: DS, ASD, ASA, ZSD, ZSA, K, SF; LOS angles from the geometry; the tables of 7.5-6
(r_τ, ζ, N, M, c_DS, c_ASD, c_ASA, c_ZSA, μ/σ of XPR) and 7.5-7 to 7.5-12 (μ_lgZSD, μ_offset,ZOD).

| Step | Content | Random variables |
|---|---|---|
| 5 | τ'_n = −r_τ·DS·ln X_n; τ_n = sort(τ'_n − min τ'_n); LOS: τ_n^LOS = τ_n / C_τ, C_τ = 0.7705 − 0.0433K + 0.0002K² + 0.000017K³ (K in dB), not used for the powers | X_n ~ U(0,1) per cluster |
| 6 | P'_n = exp(−τ_n (r_τ−1)/(r_τ DS))·10^(−Z_n/10); normalise; LOS: P_n = P'_n/ΣP'/(K_R+1) + δ(n−1)·K_R/(K_R+1), used only for the angle mapping (7.5-9, 7.5-14), not for the coefficients (7.5-22); remove clusters < −25 dB of the max (of 7.5-6), "scaling factors need not be changed"; ray power P_n/M | Z_n ~ N(0, ζ²) per cluster |
| 7 az | φ'_n = 2(AS/1.4)·sqrt(−ln(P_n/max P))/C_φ; C_φ = C_φ^NLOS(N) (Table 7.5-2), LOS ×(1.1035 − 0.028K − 0.002K² + 0.0001K³); φ_n = X_n φ'_n + Y_n + φ_LOS; LOS: shifted so cluster 1 is at φ_LOS (7.5-12); rays φ_n,m = φ_n + c_AS α_m (Table 7.5-3) | X_n ∈ {±1}, Y_n ~ N(0,(AS/7)²) per cluster and angle |
| 7 el | θ'_n = −ZSA ln(P_n/max P)/C_θ; C_θ^NLOS(N) (Table 7.5-4), LOS ×(1.3086 + 0.0339K − 0.0077K² + 0.0002K³); ZOA: + θ̄ (90° for O2I, else θ_LOS); ZOD: + θ_LOS,ZOD + μ_offset,ZOD; ray ZOD spread (3/8)·10^μlgZSD; ZOA in [180°, 360°] → 360° − θ | as above |
| 8 | random coupling AOD↔AOA, ZOD↔ZOA, AOD↔ZOD within a cluster, or within a sub-cluster for the two strongest | 3 permutations per cluster |
| 9 | κ_n,m = 10^(X/10), X ~ N(μ_XPR, σ_XPR²) per ray (XPR is not an LSP in the TR) | 1 per ray |
| 10 | Φ^θθ, Φ^θφ, Φ^φθ, Φ^φφ ~ U(−π, π) per ray | 4 per ray |
| 11 | NLOS: 7.5-22 (weak clusters) and 7.5-28 with sub-clusters for the 2 strongest: delays τ_n + {0, 1.28, 2.56}·c_DS, rays R_1 = {1..8,19,20}, R_2 = {9..12,17,18}, R_3 = {13..16} (Table 7.5-5); c_DS N/A → 3.91 ns; LOS: 7.5-29 with diag(1, −1) and exp(−j2π d3D/λ), combined by 7.5-30 | — |

Steps 1 to 9 are identical for all links from co-sited sectors to a UT [TR].

The table values of v19 are in the 23 v19 conf files [L]: `NumClusters`, `NumSubPaths`, `r_DS`, `LNS_ksi`,
`PerClusterDS` (+ `_gamma`, `_min` for the `max(0.25, 6.5622 − 3.4084 lg fc)` rule), `PerClusterAS_D/A`,
`PerClusterES_A`, `PerClusterES_D_scale = 0.375` (eq. 7.6-7) with `PerClusterES_D` as fallback, `XPR_mu/sigma`,
`SC_lambda` (Table 7.6.3.1-2), `NumClusters_min` (Table 7.6.15-1), `NF_*`, `SNS_*`, `PolVar_sigma`. Cluster count
convention: LOS files hold N (UMa LOS 12), NLOS and O2I files N+1 ("+1 for the disabled LOS path"); the same as
the old NG files.

### 1.2 In-scope components (clause 7.6)

| Clause | What it changes | New random variables | Inputs not in the conf files |
|---|---|---|---|
| 7.6.2.1 | per-ray TOA (7.6-3/-4); only for B > c/D | — | B, array size |
| 7.6.2.2 | optional: α_n,m ~ U(−2, 2) per ray and angle; τ'_n,m from U(0, 2c_DS), no sub-clusters; unequal ray powers (7.6-6) with c_ZSD = 3/8·10^μlgZSD; M = min(max(M_t M_AOD M_ZOD, M_min), M_max), k = 0.5 | 5 per ray | B, D_h, D_v, λ, M_min (20), M_max |
| 7.6.3.1 | cluster RVs (delay, shadowing, angle offsets, signs) and ray RVs (coupling, XPR, phases) spatially consistent, 2D, at the distances of Table 7.6.3.1-2; "before sorting"; signs, coupling and intra-cluster delays fixed per drop | — | — |
| 7.6.3.2 A | kinematic update of cluster delays and angles with v_rx, v_tx (7.6-9 to 7.6-14), step ≤ 1 m (≤ d3D/10 with abs TOA) | X_n ∈ {±1} per NLOS cluster, fixed per drop | velocities, Δt |
| 7.6.3.2 B | replaces steps 5–7: τ'_n ~ U(0, 2·10^(μlgDS+σlgDS)), angles ~ 2·10^(μ+σ)·U(−1,1); powers like 7.6.5.1 with Laplacian terms and Z_n; LOS: spreads ×sqrt(1+K_R/2) (DS) and ×sqrt(1+K_R) (AS) in the powers, first cluster at 0; multi-frequency: largest μ+σ of all frequencies | per cluster: delay, 4 angles, Z_n | — |
| 7.6.3.3 | LOS/indoor states, soft LOS (done in LSF); SSF RVs uncorrelated across floors | — | — |
| 7.6.3.4 | all SSF RVs site-specific | — | — |
| 7.6.5 | same LSP random values for all frequencies; cluster delays and angles the same for all frequencies; Z_n independent per frequency; steps 8–11 independent per frequency | per frequency: Z_n, coupling, XPR, phases | — |
| 7.6.5.1 | alternative steps 5'–7' with anchor spreads DS_0 … ZSA_0 (e.g. 2 GHz), r = 1.5, nominal values from exponential / wrapped Gaussian / wrapped Laplacian; powers by g_DS … g_ZSA (7.6.30a–f); LOS: τ ×sqrt(1+K_R/2), angles ×sqrt(1+K_R) | per cluster: X, Y, Z, V, W, Q_n | anchor frequency |
| 7.6.8 | LOS block = H_LOS + (d3D/d_GR)·H_GR, R_∥, R_⊥ (7.6-38/-39), angles from the flat-ground geometry; NLOS part delayed by τ_LOS | — | — |
| 7.6.9 | abs TOA: NLOS delays + d3D/c + Δτ (Δτ from LSF) | — | — |
| 7.6.10 | Doppler with v_rx and v_tx; NLOS rays + 2α_n,m D_n,m/λ (mobile scatterers) | α ~ Bernoulli(p), D ~ U(−v_scatt, v_scatt) per ray | p (0.2), v_scatt |
| 7.6.13 | element-wise spherical phase: d_1,n,m = s_TRP,n·(d3D + Δτ c + τ_n c), s_TRP = 1 for the k1 strongest, else Beta(α, β); optional UT side s_UE = 1 − s_TRP; LOS element-wise | per cluster: s_TRP | array geometry |
| 7.6.14.1.2 | SNS physical blocker model (blockers of 7.6.4.2, knife-edge per ray and element) | blockers | blocker set |
| 7.6.14.1.3 | SNS stochastic: Pr_sns ~ truncated N per UT; per cluster x_n < Pr_sns → visibility region a×b (V_n = A e^(−(Pmax−P_n)/R) + B + δ), random corner; element gain α_i,n (7.6-58), C = 13 | per UT 1, per cluster ~5 | array size W, H, element positions |
| 7.6.14.2 | SNS at the UT: usage scenario (58/29/13 %), 90 % of UTs, per-antenna attenuation table | per UT | antenna index, fc band |
| 7.6.15 | N per link in [D1, D2] (UMi, UMa, Indoor only), "selection determined by the user" | (user) | — |
| 7.6.16 | η per ray and polarisation: η' = 10^(Q/10), Q ~ N(0, 3²), normalised by 7.6-62; NLOS only | 4 per ray | — |

### 1.3 Changes from v16.1 to v19.5

The text of 7.5 steps 5–11, 7.6.2, 7.6.3.1, 7.6.3.2 (A and B), 7.6.5, 7.6.5.1 and 7.6.10 is unchanged (checked
against the v16.1.0 PDF; the CR list of Annex B has no SSF-procedure CR since CR 0019 of 2017). v19 changed:
- parameter values of Table 7.5-6 (note 8: DS, AS, ZSA, c_DS = max(0.25, 6.5622 − 3.4084 lg fc) for UMa,
  c_ASD for UMa, XPR, N for UMa NLOS = 20), new SMa tables, ZSD/ZOD offsets (CR 0043, in the LSF port);
- new optional components 7.6.13 (NF), 7.6.14 (SNS), 7.6.15 (cluster number), 7.6.16 (polarisation variability);
- the uplink rule in the 7.5 introduction (CR 0035): UL uses the DL CIR under ideal reciprocity.

**The multi-frequency / spatial-consistency divergence was not reconciled in v19.** The new components add
cluster- and ray-specific random variables (s_TRP, x_n, V_n, η, α, D) without saying whether they are spatially
consistent or frequency-shared.

### 1.4 Ambiguities and conflicts in the TR

- **A1 Signs and coupling.** 7.6.3.1 lists the signs X_n and the ray coupling among the spatially consistent RVs
  and in the same clause fixes them per drop "even if UT position changes". Read together: spatially consistent at
  the drop positions, then frozen along the track.
- **A2 Continuity of 7.5 under SC.** Sorting (7.5-2), the max-normalised angle mapping (7.5-9/14), the signs,
  the choice of the two strongest clusters, the −25 dB removal and the cluster renumbering after sorting are all
  discontinuous in the position. SC of the RVs does not give a continuous channel (the paper's argument).
- **A3 7.6.5 vs 7.5.** 7.6.5 asks for frequency-independent cluster delays and angles; 7.5 scales them with
  DS(f), AS(f). The TR notes this itself and offers 7.6.5.1.
- **A4 7.6.5.1 in LOS.** τ ×sqrt(1+K_R/2) and angles ×sqrt(1+K_R) use K at the frequency of interest, so delays
  and angles become frequency-dependent in LOS, against "reusing the delays and angles from Step 5'". Further:
  no min-subtraction of the NLOS delays (no zero-delay cluster in NLOS), no sorting, no −25 dB removal, no C
  factors, no μ_offset,ZOD, no O2I ZOA rule; anchor frequency only "e.g. 2 GHz"; Q_n per frequency not stated.
- **A5 Procedure B.** Delay/AOD/ZOD decorrelation distance "2c·10^(μlgDS+σlgDS)" (a length of the order of
  100–500 m) while AOA/ZOA use 50 m; Z_n at the SF distance; eq. 7.6-17f is labelled "ZOA" but defines the ZOD;
  no −25 dB removal; uniform ranges use μ, σ per link (continuous); multi-frequency rule only for the ranges.
- **A6 Procedure A** uses cluster decorrelation distances 60/15/50/10 m (RMa/UMi/UMa/Indoor) that differ from
  Table 7.6.3.1-2 (50/60/15, 12/15/15, 40/50/15, 10).
- **A7 Dual mobility.** 7.6.10 changes only the Doppler. How the SC RVs depend on a moving Tx, and whether swapping
  Tx and Rx must give the transposed channel, is not defined; the correlation type "site-specific" assumes a fixed
  site. D2D scenarios are not in the TR; the BS/UT asymmetric terms (μ_offset,ZOD, c_ZSD = 3/8·10^μlgZSD, ZOA = 90°
  for O2I) have no D2D meaning.
- **A8 C_φ, C_θ gaps.** Table 7.5-2 lacks N = 9, 13, 17, 18, 21–24; Table 7.5-4 lacks 4, 5, 9, 13, 17, 18 and is
  not monotonic (19: 1.184, 20: 1.178). 7.6.15 asks for N in 6–19 (UMi NLOS, Indoor NLOS), 7–15, 10–12, 15–20.
- **A9 Cluster count in LOS.** The TR's LOS cluster 1 holds the LOS ray and diffuse rays (7.5-8, 7.5-30); N
  counts it. 7.6.15 replaces N of Parts 1 and 2 only (no InF, SMa).
- **A10 −25 dB removal.** No renormalisation stated; interaction with the two-strongest split and with SC.
- **A11 Sub-clusters** assume M = 20; with 7.6.2.2 the split "shall not be applied".
- **A12 ZOD wrap.** The [180°, 360°] rule is stated for ZOA only; ZOD "follows the same procedure".
- **A13 NF and SNS rank clusters** ("k1 strongest", P_n relative to max) — same continuity issue as A2.
- **A14 Ground reflection** adds the GR power on top of the K/(K+1) block (total LOS-block power grows by
  (d3D/d_GR)²·|R|²); no renormalisation stated.
- **A15 XPR** is a per-ray distribution in the TR, but a cross-correlated LSP in QuaDRiGa (and in the library's
  LSF, generator 7). 7.6.5 makes the per-ray XPR independent per frequency.
- **A16 Powers in LOS.** 7.5-8 powers are used only for the angle mapping; the coefficients use the NLOS powers of
  7.5-6 scaled by 1/(K_R+1) (7.5-30). An output "cluster power" must say which one it is.

---

## 2. What QuaDRiGa-NG does

### 2.1 Structure

`gen_parameters` → `init_sos` (LSF: `lsf_init_phases`; SSF: `qd_sos` objects) → `gen_lsf_parameters` (library) →
`gen_ssf_parameters` (MATLAB) → `gen_fbs_lbs` (scatterers, not in baseline) → `get_channels`
(`get_channels_spherical` per snapshot, or `get_channels_planar` once + rotating Doppler in baseline).

`use_3GPP_baseline` (simpar) switches: the baseline SSF path below, planar waves, no scatterers, no SC, no
multi-frequency, the TR polarisation matrix, per-UT ray coupling, sorting and the two-strongest split. Default is
0 (QuaDRiGa mode). `lsf_model` ('38.901' / 'legacy') is the LSF switch of the LSF integration; there is no SSF
switch beyond `use_3GPP_baseline`, `SubpathMethod`, `SC_lambda`/`autocorrelation_function = 'Disable'`,
`use_random_initial_phase`, `use_absolute_delays`, `GR_enabled`.

### 2.2 Cluster convention

Slot 1 is always the LOS path (one sub-path, zero power in NLOS since KF = −100 dB), slot 2 the GR if
`GR_enabled`, then `NumClusters − 1 (− 1)` NLOS clusters. In LOS files N counts the LOS slot, so a TR LOS link has
N − 1 NLOS clusters plus the LOS ray: the diffuse part of the TR's cluster 1 is missing (**N3**).

### 2.3 Baseline path (`use_3GPP_baseline = 1`)

- Delays: 7.5-1/-2 with `rand`; **C_τ scaling not applied** (commented out) (**N1**).
- Powers: 7.5-5 with `LNS_ksi`, then slot 1 replaced by K/(K+1) and the rest normalised to 1/(K+1) (**N3**).
- Angles: `alt_AS_mapping = true` ("leading to correct calibration results"): uniform in ±1 rad, strongest set
  to 0, scaled to the exact AS by `calc_angular_spreads`, shifted to the LOS direction. **Not the TR mapping**
  7.5-9…7.5-17 (no C_φ/C_θ, no X_n/Y_n) (**N2**). The TR branch is dead code and indexes the C tables at N − 1 with
  inconsistent bounds.
- No −25 dB removal (**N5**); no μ_offset,ZOD — the library's `eod_offset_deg` is not used anywhere in NG, in
  either mode (**N4**).
- ZOA/ZOD wrap applied to both.
- Sort by power, split the two strongest into 3 sub-clusters with 1.28·c_DS steps, powers 10/6/4 of 20. Split
  only if `PerClusterDS ≠ 0` and M = 20, so RMa, InH, InF (c_DS N/A) are never split; TR: 3.91 ns (**N7**).
- Ray angles (`get_subpath_angles`): Table 7.5-3 offsets × `PerClusterAS_*`, coupling by sorting 4 uniform rows;
  sub-clusters use fixed re-normalised offset sets of 10/6/4 rays with unit RMS each, not the α_m subsets of Table
  7.5-5, and the coupling spans the whole cluster (**N7**); ZOD ray spread is the constant `PerClusterES_D`, not
  3/8·10^μlgZSD per link (`PerClusterES_D_scale` ignored) (**N6**).
- XPR per ray N(lsp_mu, lsp_sigma) (slice 8), 4 phases per ray, LOS diag(1, −1): TR.
- Multi-frequency: error (**N8**). `get_channels` disables GR in baseline although `gen_ssf_parameters` builds it.

### 2.4 QuaDRiGa path (default, the paper's model)

- Delays −ln U (unit exponential), angles (U − 0.5)·π, U from `path_sos` (5 Uniform `qd_sos` per cluster at
  `SC_lambda`, delays self-mirrored, ASD↔ASA and ESD↔ESA mirror pairs); without SC `rand`.
- Powers P = exp(−τ g_DS)·exp(−φ_D² g_ASD)·exp(−φ_A² g_ASA)·exp(−|θ_D| g_ESD)·exp(−|θ_A| g_ESA); single frequency:
  g_DS = r_DS − 1, g_AS = 0.56, g_ES = 0.76; multi-frequency: the paper's fits relative to max/min over the
  frequencies. No per-cluster shadowing (ζ unused). K applied per frequency; GR splits the LOS power into
  (1 − R²) and R² (**N16**).
- DS: delays scaled analytically to the exact DS (with GR: closed form), scale averaged over the frequencies
  (**N10**, by design). AS: iterative scaling (≤ 10 steps) to the exact AS incl. the LOS/GR paths, averaged over
  frequencies, limited to 3 (az) / 1.5 (el); elevations mirrored at the poles; LOS direction applied by a 3D
  rotation (paper eq. spherical rotation), NLOS EoA rotation faded out for 10–20 m height difference when KF ≈ 0
  (satellites; **N11**, QuaDRiGa only).
- abs TOA: NLOS delays shifted so that the minimum NLOS delay is Δτ.
- Sorting only without SC. With SC and `SubpathMethod = legacy/Laplacian`, **all** clusters are split into 3
  sub-clusters (QuaDRiGa's fix for A2). `mmMAGIC`: per-ray exponential delays from `clst_dl_sos` (deprecated).
- Polarisation: rotation matrix with a linear angle γ = acot(sqrt(XPR)) per ray and a circular phase κ per cluster,
  XPR mean = the per-link XPR LSP realization, std = XPR_sigma, so the per-ray XPR has about twice the TR variance
  (**N9**, to verify); one initial phase per ray (`pin`), independent per frequency (`xpr_sos`, `pin_sos` per
  frequency, Normal/Uniform).
- `subpath_coupling = rand(4, n_path, F)`: one draw per builder, the same coupling for all UTs (**N14**).
- SSF `qd_sos` sets are 3D, without floors and without bias removal, unlike the LSF set since the integration
  (**N12**); `init_lsf_generator` draws a seed but does not store it (**N13**).

### 2.5 Downstream in NG

- `gen_fbs_lbs`: ray angles = cluster angle ⊕ (offset × `PerClusterAS`, coupled), applied by `rotate_sphere` (on
  the sphere, not additive); path length = d3D + c·τ; multi-bounce solver, single bounce fallback; per frequency
  (coupling and cluster AS per frequency) → `fbs_pos`, `lbs_pos` [3, n_path, n_rx, n_freq].
- `get_channels`: per segment builder, `get_channels_spherical` per snapshot with fixed FBS/LBS (this is
  QuaDRiGa's drifting); `update_drifting` is dead code (**N15**); coefficient × exp(−j·pin); per-cluster power
  correction against the antenna-pattern gain; Laplacian weights for `SubpathMethod = Laplacian`; GR geometry and
  reflection coefficients recomputed per snapshot; SF/KF profiles along the track.
- `split_multi_freq`: AoD…EoA shared; `taus` shared unless c_DS is frequency-dependent; `pow`, `xprmat`, `pin`,
  `subpath_coupling`, FBS/LBS per frequency.
- Tests: `testBuilder_SSF_generation_dual` requires exact swap reciprocity (delays, powers equal; AoA↔AoD,
  EoA↔EoD) for symmetric parameters; further SSF tests listed in `xunit_new/tests` (`gen_subpaths`,
  `spatial_consistency`, `pol_rotation_dual`, `GroundReflection`, `3GPP_baseline_multifreq`, …).

### 2.6 What the builder stores (SSF state)

`NumClusters`, `NumSubPaths` [1, L]; `taus` [n_rx, L(, F)]; `gain`/`pow` [n_rx, L, F]; `AoD`, `AoA`, `EoD`, `EoA`
[n_rx, L] (cluster level, rad, elevation convention); `xprmat` [4, n_path, n_rx, F]; `pin` [n_rx, n_path, F];
`subpath_coupling` [4, n_path, F or n_rx]; `fbs_pos`, `lbs_pos` [3, n_path, n_rx, F]; generators `path_sos`,
`xpr_sos`, `pin_sos`, `clst_dl_sos`; LSF state `lsf_generator` (phase, preset, horizontal, lambda), `ds` … `xpr`,
`gr_epsilon_r`, `absTOA_offset`. Ray angles are not stored; consumers recompute them.

---

## 3. Where the paper fits

The paper (Jaeckel et al. 2019) replaces steps 5–7 and is what NG's QuaDRiGa path implements. Against the TR:
- It starts from 7.6.5.1's idea (frequency-independent delays and angles, frequency-dependent powers through a
  product of exponentials with exp(−φ²g) for azimuth and exp(−|θ|g) for elevation), but draws the initial angles
  uniformly instead of from Gaussian/Laplacian, derives the g coefficients from fits relative to the max/min spread
  over the frequencies (no anchor frequency), and then **rescales delays and angles so the output DS and AS equal
  the LSF input** (averaged over frequencies). 7.6.5.1 does not rescale ("similar but not identical").
- It is continuous by construction: no sorting, no max-normalised mapping, no signs, LOS via a 3D rotation, all
  RVs spatially consistent, and reciprocal for D2D through self-mirrored delay fields and mirrored D/A pairs. This
  is the structure Procedure B also has (uniform initial values, powers from all dimensions), but B uses fixed
  ranges from μ + σ and no rescaling.
- Its weak points (paper section III): the achievable AS is limited by K (≈ 57° at K = 10 dB), ASA output below
  input for large ASA, ESD nearly doubled for small ESD by the az/el mixing of the 3D rotation. No per-cluster
  shadowing, no ZOD offset, no cluster ESD rule.
- Typos (code is right): the DS clipping should read max(min(x, 0.85), 0.15), the AS normalisation max(·, 0.25).
- NG differs from the paper only by the GR handling, the satellite EoA fade, abs TOA, and the sub-cluster split
  under SC.

| Feature | TR 7.5 | TR 7.6.5.1 | TR Proc. B | Paper / NG QuaDRiGa |
|---|---|---|---|---|
| Continuous in position (SC + mobility) | no (A2) | yes, except the LOS K scaling | yes | yes |
| Delays/angles equal across frequencies | no (A3) | NLOS yes, LOS no (A4) | yes (max rule) | yes |
| Output spreads = LSF input | approx. (C factors, calibrated) | approx. | approx. | yes (avg over freq.) |
| D2D swap-reciprocal | no | possible with mirrored RVs | possible | yes (symmetric params) |
| Per-cluster shadowing | yes | yes (Q_n) | yes | no |
| TR distributions (Gauss/Laplace) | yes | yes | uniform | uniform |

---

## 4. LSF module: what exists, what the SSF port needs

Exists [L]: per link and frequency DS, KF, SF, ASD, ASA, ESD, ESA, XPR (`lsp`), `lsp_mu`/`lsp_sigma` (μ_lgZSD for
the cluster ZSD, μ/σ of XPR for step 9), `eod_offset_deg` (μ_offset,ZOD, elevation sign), `abs_toa` (Δτ, 0 if
disabled), ground permittivity; anchor-frequency LSPs for 7.6.5.1 by adding the anchor to `fc`; soft LOS state;
the phase machinery of `qd_gbsm_phases.cpp` (generator table, frequency grouping per λ, 2D/3D, floor offsets,
dual-mobility mirror combination, bias removal) and the seed rule; `calc_delay_spread`,
`calc_angular_spreads_sphere` (for the QuaDRiGa rescaling); `get_channels_planar` / `_spherical` /
`_multifreq` and `channel` as consumers; `sos_*` for any extra fields.

Needed on top:
- **Generator sets** for the SSF RVs (section 6.1), in `qd_gbsm_phases.cpp`, with their own public init/draw pair,
  mirror pairs for D2D, floors, 2D, per-frequency independence, blocks that are only drawn when used.
- **Deterministic model functions** for steps 5–7 (four methods) and for the ray level (steps 7–10 and the
  components).
- **Conf keys**: none required by the TR content (all table values exist). Simulation settings stay arguments: B,
  array size, M_max, anchor frequency, p and v_scatt (7.6.10), UT usage scenario (7.6.14.2). One value change is
  proposed (P3: `NumClusters` of the 8 v19 LOS files +1). Constants of the TR (r = 1.5, 3.91 ns, k = 0.5, M_min =
  20, C = 13, Table 7.5-3/-5, C_φ/C_θ, Table 7.6.14.2-2, Procedure B's 50 m) go into the code.

---

## 5. Interface boundary

### 5.1 What the SSF functions return [I]

Two levels, per link, one fixed layout per call (padded, zero power for inactive entries, so the arrays are not
ragged; this covers 7.6.15, −25 dB removal and 7.6.2.2 alike):

Cluster level, slots s = 0 … N_max: slot 0 = LOS path (zero power in NLOS), slot 1 = GR when enabled, then the NLOS
clusters.
- delay τ [n_link, n_slot(, n_freq)] in s, relative to the LOS delay, NLOS shifted by Δτ when abs TOA is on;
- power [n_link, n_slot, n_freq], the coefficient powers (7.5-30 scaling: LOS K/(K+1), NLOS P_n/(K+1); A16);
- AoD, EoD, AoA, EoA [n_link, n_slot(, n_freq)] in rad, GCS, elevation convention (as NG and `channel`);
- per link: number of active NLOS clusters; per cluster: NF scale s_TRP (and s_UE), SNS visibility parameters.

Ray level, rays r = 0 … n_ray − 1 (LOS ray, GR ray, then M rays per cluster; sub-clusters keep the ray count):
- cluster index per ray [n_ray];
- delay, power [n_link, n_ray, n_freq] (sub-cluster offsets or 7.6.2.2 delays included);
- AoD, EoD, AoA, EoA [n_link, n_ray, n_freq] (per frequency because the coupling is per frequency);
- Jones matrix [8, n_ray] per link and frequency, Re/Im interleaved as in `get_channels_*`, with XPR, the 4 initial
  phases and η (TR), or the QuaDRiGa rotation × exp(−j·pin);
- optional per ray: mobile-scatterer Doppler term 2αD (7.6.10), NF distances d_1, d_2.

### 5.2 How consumers use it

- **TR drop-based (`get_channels_planar`)**: per link and frequency: ray angles, `path_gain` = ray power × PL × SF
  (step 12, caller), `path_length` = d3D + c·τ, `M` = Jones. LOS is identified by path length = d3D (slot 0, τ =
  0). Summing rays per (sub-)cluster delay gives 7.5-27/-30. To be checked by an end-to-end test against
  7.5-22/-28/-29 typed into the test.
- **Near field (`get_channels_spherical`)**: FBS = Tx + d_1·r̂_tx, LBS = Rx + d_2·r̂_rx (or Rx side planar) with the
  same path length reproduces 7.6-47/-48; LOS element-wise (7.6-50) is what `get_channels_spherical` already does.
- **Geometry port (`gen_fbs_lbs`, later)**: ray angles + path lengths → FBS/LBS (multi-bounce solver) →
  `get_channels_spherical`. Needs the ray angles of the QuaDRiGa ray method (sphere rotation), so `ssf_rays`
  produces them directly instead of `get_subpath_angles` in the consumer.
- **Drifting (later)**: QuaDRiGa's is geometric (fixed scatterers per segment, positions move) and needs nothing
  more than the above at segment starts. Procedure A needs the cluster-level state (absolute τ̃, angles), the per-
  drop X_n of R_n (an extra column block, can be appended later, section 6.1) and the velocities: a later
  deterministic update function. Procedure B is re-evaluation at each position (needs nothing more).
- **SNS**: element gains need the array geometry: a deterministic evaluator from the visibility parameters; applying
  them per element and path is not supported by `get_channels_*` today (consumer work, later).
- **NG**: stores the cluster level as today (`taus`, `pow`, angles), replaces `xprmat`/`pin`/`subpath_coupling` by
  the ray-level output, stores the SSF phase cubes and the seed (N13).

---

## 6. Proposed function set (plan level)

All model functions are deterministic in their normalized random inputs; all randomness enters through the seed
and `*_draw`. Names provisional.

### 6.1 Random fields (in `qd_gbsm_phases.cpp`, reusing the file-local machinery)

1. **`ssf_init_phases`** — phases of the SSF generator set of one BS site and scenario: [n_coeff, n_gen, 2], slice 1
   the mirror of slice 0. Generator table from scenpar (N_max from `NumClusters`, M from `NumSubPaths`, λ =
   `SC_lambda`, 0 → 0.1 m) and the method (Procedure B has its own λ). Seed derived from the LSF seed by a fixed
   rule, with a set index for the per-frequency sets.
2. **`ssf_draw`** — normalized values [n_link, n_gen] at the links (2D/3D, floors, dual mobility), optionally only a
   subset of blocks (cost: a 20×20 set has ~3,500 ray-level generators per frequency).

Layout: blocks by variable (delay, shadowing, sign ×4, offset ×4, coupling ×3 or 4, XPR, phase ×4, then optional
blocks: 7.6.2.2 α ×4 and τ'', η ×4, mobile scatterers α and D, NF s, SNS, Procedure A X_n), each sized N_max or
N_max·M; new blocks are appended, existing columns never move. Mirror pairs: delay, shadowing, XPR, D self; AOD↔AOA,
ZOD↔ZOA for signs, offsets, coupling; Φ^θφ↔Φ^φθ (the UL channel is the transposed DL channel). Frequency-shared
blocks (cluster level) in one set; per-frequency blocks (Z_n, coupling, XPR, phases, η, α/D) in one set per
frequency.

### 6.2 Deterministic models (new source file(s), e.g. `src/ssf_generate.cpp`)

3. **`ssf_clusters`** — steps 5–7 per link: delays, powers per frequency, cluster angles, LOS/GR slots. Methods:
   `"38.901"` (7.5), `"38.901_B"` (Procedure B), `"38.901_MF"` (7.6.5.1, with anchor LSPs), `"quadriga"` (paper /
   NG). Inputs: positions, `lsp` (+ anchor `lsp`), `lsp_mu` (μ_lgZSD), `eod_offset_deg`, `abs_toa`, ground
   permittivity, O2I flag per link, optional N per link (7.6.15), `rnd`. Includes C_τ, C_φ/C_θ, −25 dB rule,
   μ_offset,ZOD, O2I ZOA, abs TOA, GR (7.6.8).
4. **`ssf_rays`** — ray level per link and frequency: offsets (Table 7.5-3, 7.6.2.2), coupling, sub-clusters
   (Table 7.5-5), ray delays and powers, XPR + phases → Jones (TR or QuaDRiGa rotation), η (7.6.16), mobile
   scatterers (7.6.10), NF scales (7.6.13), SNS visibility parameters (7.6.14.1.3), M by 7.6-8.
5. **`sns_element_gain`** — element-wise gains from the visibility parameters and element positions (7.6-58), and
   the UT-side table (7.6.14.2) [I: could also be two small functions].

Not in the public API: Procedure A update, scatterer geometry, drifting (later ports); a converter to
`get_channels_*` inputs (a documented recipe instead, tested end to end).

Public API size: 2 + 2 + 1 functions, MEX and Python wrappers for each.

---

## 7. Policy proposals

Each with options, trade-offs and a recommendation; decided in the grill round (section 9).

- **P1 Methods for steps 5–7.** (a) TR 7.5 + QuaDRiGa only; (b) all four (7.5, B, 7.6.5.1, QuaDRiGa); (c) one
  reconciled method. Rec.: (b), default `"38.901"`; each method is one branch with shared inputs/outputs. The TR is
  complete only with B and 7.6.5.1; NG needs `"quadriga"`.
- **P2 MF × SC × D2D.** v19 did not reconcile. (a) TR-literal per feature, combinations the TR leaves open
  documented and allowed (7.5 + SC accepts jumps; 7.6.5.1 + SC; B + MF); (b) make `"quadriga"` the method whenever
  two of the three are combined; (c) define a TR-based reconciled variant (e.g. 7.6.5.1 with the LOS K scaling
  moved into the powers as in B, mirrored RVs). Rec.: (a) with mirrored generators always (D2D reciprocity of the
  random fields for free) and a documented matrix of which method gives which property; no invented method.
- **P3 LOS cluster count.** (a) Keep files (LOS: N incl. the LOS slot); TR mode then lacks the diffuse part of
  cluster 1 (as NG, N3); (b) +1 in the 8 v19 LOS files (and `NumClusters_min`): rule "NumClusters = N + 1"
  everywhere, TR LOS = LOS ray + N NLOS clusters, the first at τ = 0 aligned to LOS; QuaDRiGa mode then also gets
  N NLOS clusters in LOS; (c) a rule in code that depends on the file type. Rec.: (b).
- **P4 C_φ/C_θ for untabulated N.** (a) linear interpolation between table entries, clamped; (b) nearest entry;
  (c) compute from the definition by simulation. Rec.: (a), exact at table points.
- **P5 −25 dB removal.** (a) zero the power, keep the slot, no renormalisation (TR literal, fixed layout);
  (b) omit (NG); (c) flag. Rec.: (a) in `"38.901"`, off in B / MF / quadriga (not part of those procedures).
- **P6 C_τ.** Apply in `"38.901"` LOS (TR; fixes N1).
- **P7 Sub-clusters under SC.** (a) two strongest (TR); (b) all clusters (NG under SC); (c) flag. Rec.: (a) default,
  (b) as flag for mobility; c_DS N/A → 3.91 ns in TR mode (fixes N7); sub-cluster rays and coupling by Table 7.5-5.
- **P8 Signs, coupling, intra-cluster delays (A1).** Drawn spatially consistent; the caller evaluates these blocks at
  the drop (segment start) positions and reuses them along the track (documented, like the hE rule of the LSF).
- **P9 Polarisation.** TR matrix (κ per ray from μ/σ, 4 phases, η) in TR methods; QuaDRiGa rotation in
  `"quadriga"`; decide whether the QuaDRiGa XPR keeps the per-link LSP mean (N9).
- **P10 Ground reflection.** (a) TR 7.6.8 (additive, d3D/d_GR); (b) NG power split; (c) both by method. Rec.: (c):
  TR in TR methods, NG in `"quadriga"`.
- **P11 Generator layout and seeds.** Blocks appended, never moved; per-frequency sets by derived seeds; no bias
  removal for SSF fields; SC disabled → λ = 0.1 m; 2D + floors as the LSF (fixes N12); store the seed (N13).
- **P12 `"quadriga"` fidelity.** (a) faithful port incl. its quirks (frequency-averaged scaling, iterative AS loop,
  satellite EoA fade, split-all) with a parity test against NG at integration; (b) cleaned re-design. Rec.: (a),
  with fixes only where a statistic is wrong (N9 if confirmed); `mmMAGIC` dropped (deprecated), `Laplacian` kept.
- **P13 ZOD offset in `"quadriga"`.** NG never applied it (N4). Rec.: apply as a rotation of the NLOS departure
  directions in TR methods always, in `"quadriga"` behind the legacy switch.
- **P14 Units.** Spreads in deg (as `lsp_generate`), output angles in rad (as `get_channels_*`), delays in s.

---

## 8. Work breakdown

Runs sized for one agent each (LSF needed 7). Sizes: code + docs / tests, rough lines.

| Run | Content | Size |
|---|---|---|
| R1 | `ssf_init_phases`, `ssf_draw`: generator tables, layout, mirror pairs, seed rule, blocks, per-frequency sets; tests (marginals, ACF at λ, floor independence, swap reciprocity, layout stability) | 500 / 500 |
| R2a | `ssf_clusters` `"38.901"`: steps 5–7, C factors, C_τ, removal, ZOD offset, O2I ZOA, abs TOA, GR, N per link; TR-oracle tests | 600 / 600 |
| R2b | `ssf_clusters` `"38.901_B"`, `"38.901_MF"`; tests | 400 / 400 |
| R2c | `ssf_clusters` `"quadriga"`; property tests (exact DS/AS, reciprocity), reference values for the NG parity | 500 / 400 |
| R3a | `ssf_rays` core: offsets, coupling, sub-clusters, XPR/phases/Jones (TR + QuaDRiGa), η; end-to-end test vs 7.5-22/-28/-29 with `get_channels_planar` | 600 / 600 |
| R3b | `ssf_rays` components: 7.6.2.2, 7.6.10, 7.6.13 (+ e2e with `get_channels_spherical`), SNS + `sns_element_gain` | 600 / 500 |
| R4a/b | MEX and Python wrappers, docs (`html_docu`), objdump gate, conf value change (P3) | 2 × 600 |

Order: R1 → R2a → R3a give a complete TR drop-based model (first useful milestone, calibratable against 7.8.2);
then R2b/R2c/R3b; R4 last (or wrappers per run). Then three code-review rounds and the NG integration (replace
`gen_ssf_parameters`; keep `gen_fbs_lbs`, `get_channels` in MATLAB; parity test of `"quadriga"`; full calibration
7.8.2 and the SC metrics of 7.8.3).

Deferrable without blocking the interface: Procedure A update, SNS physical blocker model, UT-side SNS, 7.6.10.

Risks:
- **Cost of the ray-level fields**: thousands of generators per link and frequency; mitigated by blocks, by
  drawing the drop-fixed blocks once per segment, and by interpolation within the coherence length (Procedure B
  note).
- **Output shape** (padded slots, per-frequency ray angles) must fit NG without copies.
- **`"quadriga"` parity**: the iterative AS loop and frequency averaging make exact oracles hard; parity by injected
  random values (as the LSF parity test).
- **Open TR points** (A1–A16) decided by policy rather than by the TR; document each in the spec.
- **Calibration** of the TR mode needs the full chain (antennas, `get_channels`), i.e. NG integration.

---

### 8.1 Comparison study (before the spec, after the literature scan) [I]

Scratch prototype (not committed), cluster level only, all four methods fed with the same LSP draws:
- spread fidelity: output DS, ASD, ASA, ESD, ESA per link against the LSF input (CDFs, bias, the K-dependent caps);
- cluster power profile: sorted powers, number of clusters within 10/20/25 dB, the effect of ζ;
- delay–power and angle–power relation (PDP slope ~ r_τ, power vs angle offset), marginal angle distributions;
- continuity along a track and between nearby UTs (jumps per metre), frequency stability of delays and angles;
- D2D swap reciprocity.
Faithful `quadriga` vs redesign (D3), measured on the same LSP draws and random inputs:
- distance from NG behaviour (how much NG's default output statistics would change at integration);
- distance from the `38.901` baseline statistics (the goal of the redesign);
- which redesign element buys what (ablation: each candidate switched on alone);
- AS/DS caps vs K, ESD inflation, per-frequency spread error;
- cost: lines of prototype code, run time per link.
Antenna-level metrics of 7.8.2 (coupling loss, singular values) need the ray step and NG; they come later.

### 8.2 Size estimate, cluster level, all four methods

Reference: the LSF series is about 2,500 lines of source (5 files + helpers, without `qd_scenpar.cpp`), 5,300
lines of tests, 930 lines of MEX wrappers (Python similar); tests ≈ 2× code.

| Part | Code + docs | Tests |
|---|---|---|
| SSF random fields (`ssf_init_phases`, `ssf_draw`; machinery exists) | 300 | 600 |
| `ssf_clusters` shared part (validation, LOS geometry, ZOD offset, abs TOA, GR, slots) | 450 | 700 |
| method `38.901` | 200 | 500 |
| method `38.901_B` | 120 | 300 |
| method `38.901_MF` | 150 | 350 |
| method `quadriga` | 350 | 600 |
| MEX + Python wrappers | 600 | 400 |
| **Total** | **≈ 2,200** | **≈ 3,450** |

About 5,500–6,000 lines, roughly 60 % of the LSF series. B and MF together cost about 270 lines of code and 650
of tests over a two-method variant. Runs: R1 random fields, R2a shared + `38.901`, R2b B + MF, R2c `quadriga`, R3
wrappers (5 runs).

## 9. Decisions needed

- **D1** Methods for steps 5–7 and the default (P1).
- **D2** The MF × SC × D2D policy (P2).
- **D3** Fidelity of the `"quadriga"` method, mmMAGIC/Laplacian (P12), its XPR (N9, P9), ZOD offset (P13), GR (P10).
- **D4** LOS cluster count and the conf change (P3).
- **D5** TR ambiguity resolutions: C factors (P4), −25 dB (P5), sub-clusters (P7), signs/coupling (P8).
- **D6** Generator layout, per-frequency sets, seeds, costs (P11).
- **D7** Output data model and the consumer recipe (section 5).
- **D8** Scope trims and run plan (section 8).
