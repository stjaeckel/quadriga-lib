# Spec review, round 1: `md/ssf_spec.md` (draft of 2026-10-28)

Scope: TR fidelity of `38.901` and `38.901_MF`, the `quadriga` definition against plan D3 and R0 4.2, plan
conformance, implementability, tests, the W-choices of spec 1.4. Line numbers are lines of `md/ssf_spec.md`.

Result: **0 blockers, 4 majors, 11 minors.** The TR transcription of 7.5 steps 5–7, 7.6.5.1, 7.6.8 (geometry and
reflection coefficients) and 7.6.15 is correct; the C tables, K polynomials, signs, units, zenith fold, −25 dB rule
and the first-cluster alignment match the TR. Most W-choices are sound; W10's reason is wrong (M4).

## How the numbers were obtained

Scratch scripts in `/home/stephan/quadriga-lib-agents/scratch/ssf_rev1/` (NumPy, one thread, no BLAS):

- `spec_ref.py`: an independent per-link transcription of spec 3.4, 3.5 and 3.6, written from the spec text alone.
  `cmp.py`, `cmp_mf.py`, `cmp_gr.py` compare it with the R0 prototype (`proto.py`, `m_tr`, `m_mf`, `m_qd(**RED)`)
  on 50 random links (N = 12, K 0–15 dB, F = 1 and 3):
  - `38.901` LOS and NLOS: delays equal, powers 1e-16, angles 3e-15 rad.
  - `38.901_MF` (prototype azimuth sign flipped as spec 3.5 says): delays equal, powers 3e-15, angles 1e-14 rad.
  - `quadriga` without GR, NLOS: 7e-16 s, powers 1e-10, angles 5e-10 rad (the prototype uses K = 1e-10). Three
    differences: EoD in **LOS** 5.9° (F = 1) and 5.4° (F = 3), see M1; EoD in NLOS with F = 3 3.4° (W5,
    intended).
  - `quadriga` with GR: LOS matches to 9e-14 (relative delay) and 8e-15 rad. In NLOS only the intended EoD offset
    differs (the comparison passed no offset).
  - DS with GR is exact: largest relative error 1.7e-15 on 400 links, no link clamped (`ds_gr.py`).
- `seeds.py`: the SplitMix64 test vectors of 2.2.4 are reproduced exactly (42 → 8766244108781592946,
  4119046106699552547, 9018399064009245349; 0 → 313702574736366215, 7774984521324166428).
- C-table interpolation: C_φ(9) = 1.054, C_φ(13) = 1.168, C_θ(9) = 0.923, C_θ(30) = 1.282 (as in 3.0.8); also
  C_θ(17) = 1.1464, C_θ(18) = 1.1652 for the test of 5.2.
- `kclamp.py`: K polynomials and the spread effect of the W2 clamp (see W2).
- `se_test.py`: standard error of the drop-statistics test of 5.2 under spatially consistent fields (M3).
- `marg.py`: SOS marginals on the 5.1 grid without bias removal (5 seeds × 108 columns): largest |mean| 0.012,
  largest |std − 1| 0.026, largest |sign fraction − 0.5| 0.010. The tolerances of 5.1 (0.05, 0.05, 0.025) hold.

## Majors

### M1 — `quadriga` adds the EoD offset in LOS scenarios as well; R0, D3 and the TR add it in NLOS only

- Spec: 3.6 step 9 (l. 698) `δ = (1/F)·Σ_k eod_offset_deg(l, k)·π/180 (0 if omitted)`, with no LOS condition. Table
  3.3.1 (l. 509) "mean over k". Test 5.4 (l. 868) pins it with "UMi LOS and NLOS, 4 links, with `eod_offset_deg`".
- Sources: the R0 redesign (`proto.py` `_apply_los`: `if side == 0 and o['zod_rot'] and not L['los']`) applies the
  offset only in NLOS scenarios. R0 4.2 and the ablation of 2.6 describe and measure the change as "NLOS departure
  centre moves by the TR offset". TR 7.5-17: in LOS the ZOD follows the ZOA procedure, with no μ_offset,ZOD.
  Spec 3.4 (l. 586, 596–598) does the same for `38.901`.
- Measured: a LOS link with a nonzero `eod_offset_deg` gets NLOS EoDs moved by the offset (5.9° in the comparison
  fixture). The R0 redesign leaves them unchanged.
- Impact: none with the v19 LOS files (they have no `EoD_offset_*` keys, so `lsp_generate` returns 0). A caller
  with a custom LOS scenario, or one who passes the offset of another scenario, gets an unmeasured behaviour. That
  behaviour differs from the TR and from `38.901`, and the R2c test locks it in.
- Fix: 3.6 step 9 `δ = LOS ? 0 : (1/F)·Σ_k …`. Table 3.3.1: "NLOS: mean over k (W5)". Test 5.4 "adds the EoD offset
  to the NLOS departures only": add "and not in LOS scenarios". 3.7 is unchanged.

### M2 — Ground-reflection slot power includes R², but the contract for the later polarisation step is not stated

- Spec: 3.0.5 / 3.3.4 / 3.6 step 5 put R²_k = (abs(R∥)² + abs(R⊥)²)/2 into the GR slot power. W7 (l. 113) says
  "the ray step applies diag(R∥, −R⊥) itself from the permittivity". 1.5 (l. 141–142) mentions "the GR
  polarisation from the permittivity". 2.3 and D7 say "powers as the coefficients use them".
- Source: TR 7.6-32/-35. The coefficient amplitude is √(K/(K+1))·d3D/dGR and the polarisation matrix carries
  R∥, R⊥, so the TR coefficient's "power" does not contain R². NG keeps R² in the path power and divides it out
  again when it builds the polarisation matrix (`get_channels.m`, `Rsq = 2·P_GR/(P_LOS+P_GR)`,
  `gSv = sqrt(2/Rsq)·R_par`).
- Impact: a consumer who follows "powers as the coefficients use them" and multiplies √power by diag(R∥, −R⊥)
  applies the reflection twice (the GR power is too low by R², median about 0.3 for the R0 permittivities, −5 dB).
- Fix: state the contract in 2.3 and in the `ssf_clusters` documentation block (3.3.5): "The GR slot power contains
  the polarisation-averaged reflection power R² = (abs(R∥)² + abs(R⊥)²)/2. A consumer that applies the reflection
  matrix uses diag(R∥, −R⊥)/R with R = √R², so that the slot keeps its power." Repeat it in 1.5 Readiness and in
  10.1.

### M3 — The drop-statistics test of 5.2 understates its standard error; the correct implementation fails it now and then

- Spec 5.2 (l. 837): UMa NLOS, 2000 links, fields from `ssf_draw` at 5 seeds, medians "within 0.05 of 1 (study:
  0.97–1.02; standard error of the median ≈ 0.006)". 5.0 asks for "5 standard errors".
- The ssf_draw fields have SC_lambda = 50 m. The UTs lie in d2D 35–300 m, about 110 decorrelation cells per seed,
  so the 2000 links are far from independent.
- Measured (`se_test.py`): the 38.901 transcription on 12 repetitions of 5 × 400 links on that ring, with typical
  UMa NLOS spreads. Median DS ratio 0.966; spread of the median over repetitions 0.0089 with λ' = 50 m, 0.0044 with
  λ' = 0.1 m. ASA: 0.993 ± 0.0063 (50 m), ± 0.0045 (0.1 m).
- So the DS limit 0.95 is 1.8 standard errors from the expected median. About 4 % of seed choices fail on DS
  alone. An implementation agent that hits such a seed is tempted to widen the tolerance or to doubt correct code.
- Fix: use `SC_lambda = 0` in the scenpar of this test (fields decorrelate within 0.1 m, so the links are
  independent; measured SE 0.0044, 3.6 SE margin for DS). Alternatively use 10^4 links. Write the tolerance as
  "expected median from the study ± 5 SE": DS [0.94, 1.02], the others 1 ± 0.05. Correct the stated SE.

### M4 — W10's reason is false: NG has scenarios with NumClusters 1 and 2 that `ssf_clusters` cannot serve; 10.2's legacy-count claim fails for GR files

- Spec W10 (l. 116): "no v19 or NG file has fewer than 3". 3.0.9 requires NumClusters ≥ 2. 10.2 (l. 1114–1116):
  "a legacy LOS file gives NumClusters − 1 NLOS clusters too, i.e. the same count as today".
- NG HEAD conf files: `Freespace.conf`, `LOSonly.conf`, `Null.conf` have NumClusters = 1 (KF_mu 0).
  `TwoRayGR.conf` has NumClusters = 2 with `GR_enabled = 1` (LOS + GR, no NLOS cluster).
  `3GPP_38.901_UMi_LOS_GR.conf` has 13 with GR: NG reads 11 NLOS clusters, the library 12. NG
  `gen_ssf_parameters.m` has explicit branches for these (`n_clusters == 1`: LOS power 1;
  `use_ground_reflection && n_clusters == 2`: powers (1 − R², R²) regardless of K).
- Behaviour:
  - LOSonly, Freespace, Null: `ssf_clusters` throws.
  - TwoRayGR with `quadriga`: 1 NLOS cluster with P = 1/(1+K) = 0.5 (KF 0 dB), LOS 0.5·(1 − R²), GR 0.5·R²,
    instead of NG's (1 − R², R²).
  - UMi_LOS_GR: one NLOS cluster more than today. 10.3 mentions this for `quadriga` (W15) but not for the baseline
    path.
- Fix:
  - Correct W10's reason.
  - Either (a) keep NumClusters ≥ 2 and state in 10.1 that NG keeps its LOS-only and GR-only branches
    (n_clusters = 1; GR with n_clusters = 2) in MATLAB, or (b) accept NumClusters = 1 (N = 0: slots 0 and GR only,
    LOS power 1 or NG split). (a) is less code.
  - Correct 10.2: legacy files with `GR_enabled` get one NLOS cluster more than in NG (W15), on both paths.

## Minors

1. **GR reflection gives NaN for finite inputs (W18 claim false).**
   - Spec 3.0.5 (l. 400–407) and W18 claim "No NaN or Inf for any finite input".
   - Measured: `tx_pos.z = rx_pos.z = 0`, d2D = 50 m, `gr_eps_re = 1`, `gr_eps_im = 0` gives cθ = 0 and q = 0, so
     R∥ = R⊥ = 0/0. The same happens for d2D = 0, h = 0, ε = 0. Validation 3.0.2 #11 admits both.
   - Fix: validate `gr_eps_re >= 1` (no ground has a relative permittivity below 1) and `gr_eps_im >= 0`
     (`lsp_generate` guarantees both). Or set R∥ = R⊥ = −1 when the denominator is 0, the grazing limit.
2. **τ_GR by cancellation.**
   - Spec l. 401: `τ_GR = (dGR − d3D)/c`.
   - Measured relative error against the stable form 4·h_t·h_r/(dGR + d3D): 7e-14 at 300 m, 1.6e-11 at 10 km,
     2e-9 at 100 km (h 35/1.5 m), 1.1e-6 at 500 km (h 1.5/1.5 m).
   - Fix: `τ_GR = 4·h_t·h_r/((dGR + d3D)·c)`. Identical in exact arithmetic, and the oracle can type either form at
     the test geometry.
3. **W2 reasoning omits C_φ.**
   - Measured (`kclamp.py`): the LOS factor of 7.5-10 is 1.181 at −5 dB; the spec mentions only C_τ = 0.990 and
     C_θ = 0.922. The factor crosses 1 at −11.7 dB; C_τ = 1 at −5.2 dB, C_θ = 1 at −4.4 dB.
   - Effect for K ≤ −8 dB (toy, N = 12, ASA 30°, ZSA 10°, DS 100 ns): the clamp gives medians DS 0.94–0.95,
     ASA 0.86–0.88, ZSA 1.03–1.09 of the input. The NLOS limit is DS 0.94, ASA 1.02, ZSA 0.95. Without a clamp: ZSA
     7.6 at −10 dB, DS 0.6 at −20 dB. With NG's rule (factor ≥ 1 for K < 0): DS 0.6–0.8.
   - The clamp is the best of these. State in W2 and in the docs that the clamp lowers ASA by about 14 % for links
     far below −5 dB.
4. **Absolute TOA in LOS scenarios (TR methods).**
   - Spec 3.3.3 (l. 530–532) adds Δτ to the NLOS delays whatever the LOS state. TR 7.6-44/-44a has no Δτ in LOS.
   - With a LOS scenario whose `absTOA_mu > −30` (custom file) or a user-supplied `abs_toa`, the NLOS clusters move
     away from the LOS ray by Δτ.
   - Fix: "TR methods: += abs_toa(l) in NLOS scenarios only (7.6-43); ignored in LOS (7.6-44)". One condition, and
     the TR becomes literal.
5. **W1: legacy NG NLOS files with KF_mu > −30 run the LOS branch.**
   - Measured: BERLIN_UMi_Campus_NLOS −13.2, DRESDEN_UMa_NLOS −10.4, QuaDRiGa_UD2D_NLOS −8, BERLIN_UMa_NLOS and
     MIMOSA_*_NLOS −6.3, BERLIN_UMi_Square_NLOS −5.5, QuaDRiGa_Industrial_NLOS +4.22, 3GPP_37.885 NLOSv 0.
   - For `quadriga` this reproduces NG (LOS power K/(1+K)). For `38.901` these files get C_τ, the K-clamped C
     factors and first-cluster alignment.
   - W1 itself is sound (see below). Name the consequence in the documentation and list the files in 10.2.
6. **Cap test of 5.4 (l. 872) cannot hold in azimuth.**
   - With s = 3 and ã ∈ (−π/2, π/2) the offset 3ã exceeds π for abs(ã) > π/3. `wrap(angle − centre)` then differs
     from 3ã by 2π, so "recovered / initial = 3" fails for those clusters.
   - Fix: compare `wrap(angle − centre)` with `wrap(3·ã)`, or restrict the check to clusters with abs(3ã) < π.
7. **Quadriga DS test with GR (5.4 l. 870).**
   - The 1e-10 equality holds only where t of 3.6 step 6 is not clamped (stated at l. 680 but not in the test).
   - Fix: the test asserts on links where the oracle's t > 0 and counts the others (0 of 400 in my fixture).
8. **TR 7.6.3.4 (Tables 7.6.3.4-1/-2): LOS, NLOS and O2I link types are uncorrelated.**
   - With one LSF seed used for the LOS and the NLOS scenario of a site, the derived SSF streams are identical. The
     LOS delay block (columns 0–11) equals the NLOS delay columns 0–11, and further blocks overlap shifted.
   - Fix: the `ssf_init_phases` documentation (3.1) states that each scenario (link type) of a site needs its own
     LSF seed, as the LSF set already does implicitly.
9. **Conf test comment.**
   - Spec 4 (l. 760–766) updates the comment at `test_conf_files_v19.cpp:211` but not the one at line 349: "Number
     of clusters: N for LOS, N + 1 otherwise; lower end of the range D1 likewise". It becomes false after D4.
   - Fix: add it to section 4 (new text: "Number of clusters: N + 1 in every file; lower end of the range D1 + 1").
10. **`quadriga` with GR in an NLOS scenario.**
    - The elevation centres use w = mean R² independent of K (3.6 step 3, NG). In an NLOS scenario the GR slot has
      zero power, yet the NLOS EoA centre moves down.
    - Measured, UMa 25/1.5 m, ε = 12.6 − 1.1j: −22.6° at d2D 35 m, −10.4° at 100 m. UMi 10/1.5 m at 30 m: −12.5°.
      R0 measured GR only in LOS (2.7).
    - Fix (one condition): w = 0 when the scenario is NLOS. LOS behaviour (D3) stays.
11. **Helper header bookkeeping.** `src/quadriga_gbsm_helpers.hpp` lists the users of `qd_gbsm_check_links` (l. 11,
    35). `src/ssf_clusters.cpp` becomes a user and must be added (R2a). Not mentioned in 3.3 or 9.1.

## TR fidelity: points checked and found correct

- 7.5-1/-2: −r_τ·DS·ln X, min subtraction. 7.5-3/-4: C_τ applied to the output delays only, not in 7.5-5.
- 7.5-5/-6: powers from the min-subtracted delays, ζ·z. 7.5-8: angle-mapping powers (LOS cluster f gets
  K/(K+1)).
- −25 dB rule against the 7.5-6 powers, C factors unchanged (l. 435–436, 570).
- 7.5-9/-10/-14/-15: C_φ, C_θ with the LOS K polynomials in dB. Tables 7.5-2/-4 typed correctly.
- X_n ∈ {±1}, Y_n ~ N(0, (AS/7)²) per angle type. 7.5-11/-12/-16/-17/-19: μ_offset,ZOD = −EoD offset in NLOS only.
  O2I ZOA 90° in NLOS. ZOD in LOS via 7.5-17.
- Zenith fold (7.5 step 7, D5) for ZOA and ZOD.
- 7.6.5.1: Y, Z (Normal) for the azimuths and V, W (Uniform) for the zeniths. The arg principal value, the LOS
  cluster 1 zeroing and the g factors 7.6.30b–f are typed correctly. The erfc identity equals the TR form exactly.
  7.6.30i–m: √(1+K/2) and √(1+K) with the TR's wrap-then-scale order.
- 7.6.8: cos θ_GR,ZOD = −(h_t+h_r)/dGR, sin² = (d2D/dGR)², 7.6-38/-39. abs(R) does not depend on the sign of
  Im ε. Angles 7.6-37a–d. Power K/(K+1)·(d3D/dGR)² (with R², see M2).
- 7.6.15: N_l replaces N in the C tables, range [D1, D2] from `NumClusters_min − 1`.
- Conf: the six LOS files and the counts of section 4 are correct. The `*_LOS_O2I` files already hold N + 1
  (UMa/UMi 13, RMa 11, SMa 15). The N_TR list for the new check matches Table 7.5-6 Parts 1–4.

## `quadriga` against D3 / R0 4.2 / NG

Matches NG `gen_ssf_parameters.m` (HEAD 9a2beeb, QuaDRiGa branch) element by element:

- −ln U delays; (U − ½)π angles; GR Rsq form (identical to TR 7.6-38/-39).
- `mu_fixed` = linear mean with weights (1 − mean R², mean R²), via `qf.calc_angular_spreads(·,·,0)`.
- gDS/gAS fits with clipping, and the r_τ − 1 rule for equal DS. Power split.
- Closed-form delay scale with and without GR. The iterative loop: start from the wrapped spread, at most 9
  passes, 1e-4 dB stop, linear spread.
- Frequency averaging, caps 3 / 1.5, pole mirror of the offsets.

The three D3 changes are as in R0, except M1 (offset in LOS) and the intended W5 (frequency mean of the offset). The
"not adopted" list (ζ, Gaussian/Laplacian initial values, wrapped solve, exact per-frequency scaling) is respected.

## W-choices (spec 1.4)

| ID | Verdict |
|---|---|
| W1 | **Sound.** A scenario-level state matches NG's one-builder-per-state design. It supports TR soft LOS (7.6-19 runs the LOS and the NLOS scenario for the same link). It avoids per-link K thresholds that would jump under SC. −30 separates all v19 files. Document the legacy NLOS files with KF_mu > −30 (minor 5). |
| W2 | **Sound; reasoning incomplete** (minor 3). Of the options measured (no clamp, clamp at 0 dB, NG's factor ≥ 1, clamp at −5 dB) the −5 dB clamp keeps DS and ZSA nearest the NLOS limit and avoids the C_θ sign change. Its cost is ASA about 0.86 of the input for K ≪ −5 dB; document this. |
| W3 | **Sound.** TR 7.6.3.1: "The procedure shall apply to each cluster before sorting the delay". The sort is a relabelling of exchangeable clusters, and the first cluster is the one with the smallest delay. |
| W4 | Sound (the only zenith convention of the TR; needed to output valid elevations). |
| W5 | Sound (delays and angles must be frequency-independent). Apply it only in NLOS (M1). |
| W6 | Sound (TR-literal for 7.6.5.1; D3 lists exactly three changes). |
| W7 | Sound, with the contract of M2 stated. |
| W8 | **Sound.** Test vectors reproduced. Independent streams per set; append-only safe. Seeds up to 2^52 (NG) map without loss. Add the per-link-type seed rule (minor 8). |
| W9 | **Sound.** One field realisation for all methods. Φ is the map of `ssf_draw`. |
| W10 | **Reason wrong** (M4). Either keep it and hand NG's LOS-only/GR-only cases back to MATLAB explicitly, or accept NumClusters = 1. |
| W11 | Sound. |
| W12 | Sound. |
| W13 | Sound. |
| W14 | Sound (NG's xunit `unit_test_conf.conf` also has N = 2; test-only). |
| W15 | Sound; extend 10.2 (M4). |
| W16 | **Sound.** μ_lgZSD enters only 7.5-20 (ray level). D7 already says ray-step inputs are read from `lsp_mu`, not duplicated. |
| W17 | **Sound.** R2a's LOS oracle tests need N + 1 in the files. No other existing test reads `NumClusters` of the v19 LOS files (checked: Catch2, MEX, Python). Include the second comment of minor 9. |
| W18 | Sound in intent; one hole (minor 1). |
| W19 | Sound. |

## Plan conformance and implementability (no further findings)

- D1–D8 carried over as written; Procedure B out; the D4 file count corrected with reason (6, not 8).
- Append-only layout with ray and Procedure A/B room (B blocks at their own λ fit the per-generator `lambda` of
  `qd_gbsm_gen`). No `md/`/ID references are prescribed for produced files (9.1).
- Machinery reuse: `qd_gbsm_gen` table, `qd_gbsm_freq`, `qd_gbsm_positions`, `qd_gbsm_eval` (signatures checked in
  `src/qd_gbsm_phases.cpp`). `qd_interp_linear(y, x, n, x0)` is exact at table points and clamps.
- Validation order, messages, sizes and `n_link = 0` are defined.
- Run reading lists cover what each run needs (R2a reads 3.3 for the validation of all methods).
- Wrapper helper names exist (`qd_mex_matlab2vector_Bool`, `qd_python_flag`, `qd_mex_get_seed`, …).
