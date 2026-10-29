# Spec review, round 2: `md/ssf_spec.md` (revision after round 1, 2026-10-28)

Scope: a full independent read of the spec, plus a check of every round-1 finding and of the new parts. The new parts
are the `NumClusters = 1` path, the GR power contract, W20, W21, the permittivity validation and N = 1 in the TR
methods. Line numbers refer to `md/ssf_spec.md`.

Result: **0 blockers, 1 major, 7 minors.** The TR transcription of 7.5 steps 5–7, 7.6.5.1, 7.6.8 geometry and
coefficients, 7.6.9 and 7.6.15 is still correct. `quadriga` matches plan D3 and NG's QuaDRiGa branch element by
element. Round-1 findings M1, M3, M4 and minors 1–11 are fixed. M2 is fixed only for the GR slot: the contract it
added is wrong for the LOS slot of `quadriga`, and the NG hand-off for the TR methods is incomplete (major R2-M1).

## How the numbers were obtained

Scratch scripts are in `/home/stephan/quadriga-lib-agents/scratch/ssf_rev2/` (Python, one thread, no BLAS). The
round-1 scripts in `../ssf_rev1/` were reused where they still apply.

- `gr_contract.py`: the narrowband LOS + GR channel for V-pol isotropic antennas at 6 GHz, BS 10 m, UT 1.5 m,
  ε = 13.1 − 1.18j (the medium ground of `lsp_generate` at 6 GHz), K/(K+1) = 1. It compares four channels: TR
  7.6-32, NG `get_channels` with NG's split powers, the spec contract applied to the `quadriga` slot powers, and the
  spec contract applied to the TR slot powers. It also runs NG's LOS rescaling `Sl` on TR powers, and the
  "non-polarised consumer" of 2.3.
- `gr_eps.py`: reflection coefficients at ε = 1 + 0j for several geometries. It also prints the real part of
  `lsp_generate`'s permittivity fit over frequency and scans 2·10^5 random ε with real part in (0, 1] for non-finite R.
- `w13.py`: the DS of `quadriga` LOS links before and after the abs-TOA shift (1000 links, N = 12, K = 9 dB).
  It also computes a 38.901 power normalisation with an extreme finite shadowing value.
- `seeds.py`: the SplitMix64 test vectors of 2.2.4 are reproduced exactly (42 → 8766244108781592946,
  4119046106699552547, 9018399064009245349; 0 → 313702574736366215, 7774984521324166428).
- NG HEAD 9a2beeb, read with `git show`: `gen_ssf_parameters.m` (lines 100–290, 470–830) and `get_channels.m`
  (lines 320–370).

## Major

### R2-M1 — The GR power contract is wrong for the LOS slot of `quadriga`; the NG hand-off for the TR methods omits NG's LOS rescaling

- **Spec.** 2.3 (l. 325–330): the GR slot contains R², and a consumer uses diag(R∥, −R⊥)/R with
  √(slot power) "so that the reflection is applied exactly once and the slot keeps its power". The spec says
  nothing about the LOS slot. W7 (l. 113): "NG divides R² out the same way in `get_channels`". 3.7 (l. 784) and
  A14 (l. 94): `quadriga` "NG split, unit power". 10.1 (l. 1155–1159): NG takes R² from the permittivity for the TR
  methods and applies diag/R "with the amplitude √(GR slot power)".
- **Source.** NG `get_channels.m` l. 341–357 computes `Rsq = 2·P_GR/(P_LOS + P_GR)` (= 2R² for the split),
  `gSv = sqrt(2/Rsq)·R_par` and `Sl = 1/sqrt(1 − Rsq/2)`, and sets `M(1,1) = Sl`. So NG divides R² out of the GR
  slot **and** divides (1 − R²) out of the LOS slot. NG's channel has LOS power K/(1+K) and GR amplitude
  √(K/(1+K))·R∥, i.e. TR 7.6-32 without the d3D/d_GR factor. NG's split exists only in the stored powers that feed
  the DS/AS scaling. D7 (plan, binding) asks for "powers per frequency as the coefficients use them". The
  `quadriga` LOS slot power K/(1+K)·(1 − R²) is not the power NG's coefficients use.
- **Measured (`gr_contract.py`, |h|² in dB).** The case is LOS + GR only, as `TwoRayGR` with `quadriga`.

  | d2D [m] | TR 7.6-32 | NG today | spec contract, `quadriga` powers | spec contract, TR powers |
  |---|---|---|---|---|
  | 100 | −4.12 | −4.13 | −9.85 | −4.12 |
  | 200 | −8.85 | −8.86 | −22.95 | −8.85 |
  | 1000 | 5.21 | 5.21 | 1.35 | 5.21 |
  | 10000 | −8.56 | −8.56 | **−0.90** | −8.56 |

  With the contract, a `quadriga` consumer gets an LOS amplitude of √(1 − R²). Beyond the break point this goes
  to 0 (R² = 0.99 at 10 km), so the two-ray fade disappears: −0.9 dB instead of −8.6 dB. NG integration would change
  NG's channel if NG followed the library contract.
- **NG hand-off for the TR methods.** With TR powers, NG's unchanged `Sl` gives `Sl² = 1 + (d3D/d_GR)²·R²`. The LOS
  power then rises by +1.1 dB (d2D 20 m), +1.4 dB (50 m), +1.8 dB (100 m) and +2.4 dB (300 m) for BS 10 m / UT 1.5 m.
  10.1 does not tell NG to drop `Sl` for these methods.
- **Non-polarised consumer.** l. 330 says such a consumer "uses √(slot power) as is". That drops the phase of R,
  which is ≈ π near grazing. Measured with TR powers: +5.8 dB instead of −8.6 dB at 10 km, +4.7 dB instead of
  −4.1 dB at 100 m.
- **Fix.** Do not change the code; correct the contract. Proposed text for 2.3 and the docs (3.3.5):
  - "The GR slot power contains R²_k. A consumer forms the GR coefficient as √(GR slot power)·diag(R∥, −R⊥)/R,
    R = √R²_k.
  - TR methods: the LOS coefficient is √(LOS slot power).
  - `quadriga`: the LOS and GR slot powers are NG's split for the spread scaling (sum 1). The LOS coefficient is
    √(LOS slot power/(1 − G·R²_k)), so the coherent channel has LOS power K/(K+1) and GR power K/(K+1)·R²_k
    (NG's `get_channels` factor 1/√(1 − R²)).
  - A consumer without polarisation must still apply a reflection phase, e.g. −R⊥/abs(R⊥); √(slot power) alone
    reverses the two-ray interference.
  - R² = 0 gives a zero GR coefficient."
- **Further corrections.**
  - 10.1: "for the TR methods NG takes R² from the permittivity, applies diag/R to √(GR slot power) and sets
    Sl = 1".
  - W7's reason: "NG divides R² out of the GR slot and (1 − R²) out of the LOS slot".
  - 3.7 and A14: "NG split of the stored powers (unit sum); coherent channel as NG".
- **Alternative for the maintainer (a code change, deviates from D3's "NG GR power split"):** `quadriga` stores the
  channel powers (LOS K/(1+K), GR K/(1+K)·R², NLOS normalised to 1/(1+K)) and uses them, normalised, as weights for
  the DS/AS scaling. D7 would then hold literally, and NG would set `Sl = 1` for every method. I recommend the
  contract fix (no code change, NG's output unchanged).

## Minors

1. **The Python test of `gr_eps` contradicts the validation.**
   - Spec 6.2 (l. 1001–1002): `gr_eps` "as its conjugate … the first two give equal outputs". Validation 3.0.2 #11
     (l. 387) rejects `gr_eps_im < 0`.
   - `lsp_generate` returns `re + 1j·im` with im ≥ 0 (`qpy_lsp_generate.cpp` l. 81). Its conjugate, e.g.
     13.1 − 1.18j, therefore throws `Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.` R3 will
     find the test impossible.
   - Fix: replace the conjugate case by "a negative imaginary part raises `ValueError` with the C++ message". Keep
     the real-array case.

2. **ε = 1 + 0j, the edge of the validated range, is degenerate.**
   - Spec 3.0.5 (l. 436–437): a zero denominator is set to −1 (grazing limit). W18.
   - At ε = 1 + 0j, R∥ = R⊥ = 0 for every geometry with h_t + h_r > 0. Measured R² = 6e-31 (25/1.5 m, 100 m),
     8e-24 (1.5/1.5 m, 1 km), 4e-16 (1 mm, 50 m). The 0/0 rule then gives R = −1 at h_t = h_r = 0, so the GR power
     jumps from 0 to K/(K+1)·(d3D/d_GR)² when the heights reach 0.
   - The consumer rule diag/R then divides by R ≈ 0. ε = 1 + 0j is vacuum, which reflects nothing.
   - Fix: set the 0/0 case to R∥ = R⊥ = 0, the value at every other geometry for ε = 1 + 0j. State in the contract
     that R² = 0 gives a zero GR coefficient (R2-M1). The test of 5.2 ("grazing case … ε = 1: finite, R = −1")
     changes to R = 0. A grazing test with ε ≠ 1 (h = 0, ε = 15 − 1j: R = −1 exactly, no 0/0) keeps the −1 limit
     covered.

3. **Permittivity range: the stated reason is wrong; rejecting a real part below 1 is defensible but not needed.**
   - Spec l. 388–389: "`lsp_generate` returns 3–31".
   - Measured (`gr_eps.py`): the real part of `lsp_generate`'s fit is 42.1 (medium) and 43.7 (wet) at 0.5 GHz, the
     lower end of TR 38.901. It falls below 1 above about 1.3 THz (wet 0.94, medium 0.98 at 1.5 THz). There
     `ssf_clusters` would reject the library's own output (the fit is far outside its range there).
   - Numerically, a real part > 0 and an imaginary part ≥ 0 suffice: 0 non-finite R in 2·10^5 random cases with
     real part in (0, 1] (`gr_eps.py`). The only zero denominator is cθ = 0 with ε = 1 (proof: the principal
     root has Re q ≥ 0 and cθ ≤ 0).
   - Verdict on the maintainer question: rejecting a real part below 1 is physically right for ground (ε_r ≥ 1;
     the TR's metal is 1 − j·huge) and harmless in the TR range (0.5–100 GHz: 3.0–44). Keep it.
   - Fix: correct the text to "`lsp_generate` returns 3–44 between 0.5 and 100 GHz". Add one documentation
     sentence: the permittivity fit of `lsp_generate` falls below 1 above about 1.3 THz, where `ssf_clusters`
     rejects it.

4. **W13: `quadriga` shifts the NLOS delays to Δτ in LOS scenarios as well.**
   - Spec 3.3.1 (l. 554), 3.6 step 10 (l. 751): the shift applies wherever `abs_toa(l) > 0`, with no scenario
     condition. The TR methods add Δτ in NLOS only (3.3.3, round-1 minor 4, TR 7.6-44/-44a).
   - Measured (`w13.py`): a LOS scenario (K = 9 dB, DS 36 ns, N = 12) with a caller-supplied Δτ = 30 ns. The DS
     ratio is 1.00 before the shift and 1.10 after it (median; p10 0.94, p90 1.17). The documented property "DS
     exact at one frequency" (l. 760) then fails.
   - The v19 LOS files have the TOA off, so only custom scenarios are affected. NG shifts whenever an offset
     exists, but NG never gets one in LOS from the TR files.
   - Fix: "`quadriga`: in NLOS scenarios, if `abs_toa(l) > 0`, shift …; `abs_toa` is ignored in LOS scenarios (TR
     7.6-44)". Update the W13 text, table 3.3.1 and the 5.4 abs-TOA case ("LOS scenario: ignored").

5. **W18 "No NaN or Inf for any valid input" is false for extreme finite random values.**
   - Spec W18 (l. 124). Validation 3.0.2 #6 and #7 checks `rnd` and `rnd_freq` only for finiteness.
   - Measured (`w13.py`): a 38.901 cluster with z = −1100, ζ = 3 overflows 10^(330) to Inf. The powers then
     become [NaN, 0, 0].
   - `ssf_draw` values are bounded by √(2·n_coeff) (22.6 for 256 sinusoids), so only hand-made inputs trigger
     this.
   - Fix (either):
     - (a) Narrow the claim to "for random values within ±√(2·n_coeff) of `ssf_draw`".
     - (b) Normalise the powers in the log domain in all three methods: L_n = ln P'_n, P_n = exp(L_n − max L) / Σ
       exp(L_m − max L). That is three lines, gives identical results wherever the floors do not act, and also
       removes the all-underflow case of 7.6.30a (Σ P' = 0 for an anchor DS far above DS).

     (b) closes the defect everywhere.

6. **R2b and R2c do not read the instruction to add their `NumClusters = 1` section.**
   - Spec 5.2 (l. 891): "R2b and R2c add a section in their file". The tables 5.3 and 5.4 have no such case. The
     reading lists of R2b (l. 1069) and R2c (l. 1078) include 5.3 / 5.4 but not 5.2.
   - Fix: add a row "`ssf_clusters - 38.901_MF one cluster gives the LOS path only`" to 5.3 and the same for
     `quadriga` to 5.4 (with GR: MF TR form per frequency; `quadriga` (1 − R², R²)). Alternatively add 5.2 to both
     reading lists.

7. **Consistency points that an implementer must resolve.**
   - (a) 3.3.2 (l. 565) computes "after the method kernel: … slot 1 if G (3.0.5, power 3.3.4)". `quadriga` needs
     R²_k, E_GR and τ_GR inside its kernel: w (step 3), P_GR (step 5) and B, D (step 6). Say that the GR geometry
     and R²_k are computed per link before the kernel and passed in. Only the slot writing follows the kernel.
   - (b) 3.6 uses w for the GR weight (step 3, l. 702) and again for the normalised power weights (step 7,
     l. 730). Rename the second to p̂.
   - (c) The 2.3 table (l. 316) says the GR power is "0 in NLOS scenarios". With N = 0 it is not 0 (3.0.9,
     3.0.4). Add "(except N = 0, below)".

## Round-1 findings: status

| Round 1 | Status |
|---|---|
| M1 EoD offset in LOS (`quadriga`) | Fixed: δ = LOS ? 0 : mean (l. 746–747), table 3.3.1, test 5.4. |
| M2 GR contract | Stated for the GR slot (2.3, 1.5, 3.3.5, 10.1). Incomplete for the LOS slot of `quadriga` and for NG's `Sl` (R2-M1). |
| M3 drop-statistics SE | Fixed: SC_lambda = 0 and LSF distances 0; the intervals are centred on the R0 UMa NLOS 6 GHz medians (0.97, 1.02, 0.99, 0.98, 0.99 in R0 2.1); ±0.03 ≥ 5.6 SE. |
| M4 NumClusters 1/2 | Fixed by the maintainer's N = 0 decision (3.0.9, W10, 10.2). NG's `n_clusters == 1` and `GR && n_clusters == 2` branches are reproduced for `quadriga` (NG l. 269–288). No conf file with GR has `NumClusters_min`, so the NumClusters − 1 copy of 10.2 cannot violate `NumClusters_min ≤ NumClusters`. |
| minor 1 GR NaN | Fixed. The new edge ε = 1 + 0j is minor 2. |
| minor 2 τ_GR | Fixed (stable form). |
| minor 3 W2 reasoning | Fixed. The Φ values of the K < −5 dB link fractions were rechecked: InF 6.7 %, SMa 2.3 %, UMi 0.26 %, InH/RMa 0.13 %, UMa 3e-5. |
| minor 4 abs TOA LOS (TR methods) | Fixed. For the parallel point in `quadriga` see minor 4. |
| minor 5 legacy NLOS files | Fixed. None of the listed NG files has `absTOA_mu`, `EoD_offset` or `GR_enabled`, so W1 adds no further consequence. |
| minor 6 cap test | Fixed (wrap(3ã)). |
| minor 7 DS test with GR | Fixed. |
| minor 8 seed per link type | Fixed (W21, 3.1 docs). |
| minor 9 conf comment | Fixed (both comments, l. 811–817). |
| minor 10 GR in NLOS (`quadriga`) | Fixed (W20). |
| minor 11 helper users list | Fixed (3.3). |

## The new parts

- **N = 0 path.**
  - `ssf_init_phases` and `ssf_draw` skip the SOS calls, which reject `n_gen = 0` (`sos_init_phases.cpp` l. 64).
  - The MEX `qd_mex_copy2matlab` returns an empty matrix or cube as 0×0 (`mex_helper_functions.hpp` l. 727,
    768). The "any empty" acceptance in 3.0.2 #6 and 3.2 is therefore needed and sufficient.
  - The `NumClusters_min ≤ NumClusters` check of `scenpar_validate` keeps `n_cluster` at 0.
  - Output sizes are defined for all three methods. NG's LOS-only and LOS + GR branches are reproduced for
    `quadriga` (powers 1, and (1 − R², R²) regardless of K).
  - Only the test instruction is incomplete (minor 6).
- **TR methods with N = 1 or N_l = 1.**
  - 7.5 gives τ = 0, P = 1, ρ = 0, so the angles are LOS + Y_n in NLOS and exactly the LOS angles in LOS (v − v_f
    = 0). C_φ and C_θ clamp to the N = 4 and N = 6 entries.
  - 7.6.5.1 is well defined. `quadriga` falls into the `!(den > 0)` and `!(S0 > 0)` guards in NLOS (single
    cluster, DS 0).
  - No NaN.
- **GR with the additive 1 + (d3D/d_GR)²·R²** is TR 7.6-32 for the TR methods, also for N = 0. Correct (TwoRayGR
  with `38.901` changes NG's baseline split, as 10.2 says).
- **W20, W21**: sound (below).
- **Permittivity validation**: minors 2 and 3.

## TR fidelity: rechecked and correct

- **7.5-1 to -8:**
  - min subtraction; C_τ on the output delays only.
  - P' with ζ·z in dB.
  - The −25 dB criterion on the 7.5-6 powers.
  - The angle-mapping powers of 7.5-8 with the LOS share on the first cluster.
- **7.5-9 to -19:**
  - Tables 7.5-2 and 7.5-4 are typed as in the TR.
  - The LOS polynomials of C_φ and C_θ (signs and coefficients) are correct.
  - X_n ∈ {±1}; Y_n ~ N(0, (AS/7)²) per angle type.
  - μ_offset,ZOD = −`eod_offset_deg`, NLOS only (`lsp_generate` l. 152: EoD offset = minus the ZOD offset).
  - The O2I ZOA of 90° in NLOS.
  - LOS ZOD via 7.5-17.
  - The zenith fold for ZOA and ZOD.
- **7.6.5.1:**
  - Y, Z (Normal) for the azimuths; V, W (Uniform via Φ, erfc identity exact) for the zeniths.
  - arg principal value with the TR's minus sign.
  - LOS cluster 1 zeroed.
  - 7.6.30a–h with Q_n per frequency.
  - 7.6.30i–m with √(1+K/2) and √(1+K), applied after the wrap.
- **7.6.8:**
  - θ_GR,ZOD = 180° − atan(d2D/(h_t+h_r)) gives cθ = −(h_t+h_r)/d_GR, s² = (d2D/d_GR)² and E_GR =
    −atan2(h_t+h_r, d2D).
  - φ_GR,AOA = φ_GR,AOD + 180°.
  - 7.6-38/-39 as written. They are identical to NG's grazing-angle form (`gen_ssf_parameters.m` l. 262–264, with
    sin θ_r = −cθ).
  - The power factor (d3D/d_GR)² is correct.
- **7.6.9:** + Δτ in NLOS (7.6-43), none in LOS (7.6-44/-44a).
- **7.6.15:** N_l replaces N, also in the C tables. The ranges [D1, D2] follow from `NumClusters_min − 1` after
  D4 (UMa LOS 11 − 1 = 10, UMa NLOS 16 − 1 = 15, UMi 7 − 1 = 6, Indoor NLOS 7 − 1 = 6, Indoor LOS 8 − 1 = 7).
- **Conf (4):**
  - The six LOS files and their values match `data/config` (UMa 12/10, UMi 12/6, Indoor 15/7, RMa 11, SMa 15,
    InF 25).
  - The O2I files already hold N + 1.
  - The test entries at `test_conf_files_v19.cpp` l. 346 and 352–358 exist as quoted.
- **The K polynomials never cross zero for K ≥ −5 dB.** Minima: C_τ 0.077 near 26 dB, the C_φ LOS factor 0.54 near
  20 dB, the C_θ LOS factor 0.46 near 24 dB. W2 needs no upper limit.

## `quadriga` against D3 and NG (rechecked against NG l. 478–747)

The spec matches NG step by step:
- −ln U delays.
- (U − ½)π angles from Φ of blocks 5–8.
- The fixed-slot centre as the linear mean with weights (1 − mean R², mean R²); w = 0 in NLOS is W20.
- The g fits with clipping, and the r_τ − 1 rule for equal DS.
- The split P_LOS = K/(1+K)·(1 − R²), P_GR = K/(1+K)·R², with NLOS normalised to the rest.
- The closed-form delay scale with and without GR. The discriminant was derived again and equals NG's `tmp`.
- The iteration: start from the wrapped spread, at most 9 passes, stop at 1e-4 dB, update in the last pass.
- The frequency mean, then the caps 3 / 1.5.
- The pole mirror of the offsets.

The D3 changes:
- The additive LOS application.
- The EoD offset in NLOS only.
- No satellite fade.

The "not adopted" list is respected. The guards (`!(S0 > 0)`, `den > 0`) only replace NG's NaN cases.

## W-choices (spec 1.4)

| ID | Verdict |
|---|---|
| W1 | Sound (round 1). |
| W2 | Sound. The polynomials are positive for all K ≥ −5 dB, so no upper limit is needed. |
| W3 | Sound. TR 7.6.3.1: "The procedure shall apply to each cluster before sorting the delay". |
| W4 | Sound. |
| W5 | Sound (NLOS only, after M1). |
| W6 | Sound. |
| W7 | **Not sound as written**: the contract misses the `quadriga` LOS slot, NG's `Sl`, and the phase for non-polarised consumers (R2-M1). |
| W8 | Sound (test vectors reproduced again). |
| W9 | Sound. |
| W10 | Sound (maintainer decision). It reproduces NG's branches for `quadriga`; TR methods use 7.6-32. |
| W11 | Sound. |
| W12 | Sound. |
| W13 | TR part sound. **`quadriga` part should be NLOS-only** (minor 4). |
| W14 | Sound. |
| W15 | Sound. |
| W16 | Sound. |
| W17 | Sound. |
| W18 | Floors sound. The claim is too broad (minor 5). Permittivity validation: keep real ≥ 1, but set the 0/0 case to 0 (minors 2, 3). |
| W19 | Sound. |
| W20 | **Sound.** A GR with zero power must not move the NLOS centre. LOS behaviour is unchanged (D3). |
| W21 | **Sound.** TR 7.6.3.4 Tables 7.6.3.4-1/-2: link types are uncorrelated, and correlation is site-specific. With one seed, the phase columns of the LOS and NLOS sets would be identical column by column (sos_init_phases fills column g independently of n_gen). Only the documentation is affected; NG already draws one seed per builder. |

## Plan conformance, layout, implementability, tests (no further findings)

- **Plan.** D1–D8 carried over. Deviations from the plan's letter are justified in W16 (no `lsp_mu`), W17 (conf
  change in R2a) and section 4 (6 files, not 8). No produced file is told to cite md/ documents or labels. The TR
  notation "D1" in the test comment is Table 7.6.15-1's.
- **Layout.** Blocks are appended in units of N (ray blocks N·M, cluster-major). Each generator has its own lambda in
  `qd_gbsm_gen`, which leaves room for Procedure B blocks at their own distances and for a Procedure A sign block.
  The per-frequency sets take the ray blocks of 7.6.5.
- **Reuse.** `qd_gbsm_gen`, `qd_gbsm_freq`, `qd_gbsm_positions`, `qd_gbsm_eval` and `qd_interp_linear(y, x, n, x0)`
  were checked against `src/qd_gbsm_phases.cpp` and `quadriga_lib_helper_functions.hpp` l. 567. The globs in
  `CMakeLists.txt` (l. 353, 507, 678) pick up the new source, MEX and test files. Python registration goes through
  `python_gbsm.cpp`.
- **Tests.**
  - Oracles are typed; helpers are library calls; no BLAS.
  - The statistical tolerances are ≥ 5 SE (5.1 marginals; 5.2 drop medians).
  - The `quadriga` iteration tolerance argument holds: powers and delays do not depend on the iteration, and
    angles differ only by rounding when the branches agree.
  - The swap tests hold by construction: the dual-mobility swap is bitwise; atan2(−y, −x) against
    wrap(atan2(y, x) + π) differs by 2π only on the branch cut, and the test compares after wrapping.
