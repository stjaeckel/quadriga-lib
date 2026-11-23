# TR 38.901 inventory: what quadriga-lib covers

## 1. Introduction

This document lists the features of 3GPP TR 38.901 v19.5.0 (2026-09) and states, for each one, whether the C++ library of quadriga-lib implements it, in which function, and how the implementation differs from the TR. It is meant for contributors and maintainers who want to know which parts of the TR the library reproduces today, where it follows the TR only through a different API, and where the results would not match the TR.

Scope and basis:

- **Library state:** main after the source cleanup of 2026-11-19. Line numbers refer to that state and move with later edits; the function names stay valid longer.
- **What counts:** only the C++ library (`src/`, `include/`) and the scenario files in `data/config/` (cited by file name and key). The MATLAB and Python wrappers expose the same functions but are not counted separately. QuaDRiGa-NG, which builds a full simulator on top of the library, is not part of this inventory.
- **Method:** the TR text and the code were read side by side. Nothing was run for this document. Where a finding rests on an existing record (a test, an API doc block, the maintainer notes in `md/ssf_notes.md` with their IDs D, A, W), the record is cited. Where it rests on reading the code only, the comment says "(from reading, not measured)". Section 6 proposes runs that would confirm these points.
- **TR reference:** the clause numbers, tables and equations of v19.5.0. The ISAC CRs of v19.1 to v19.5 and CR 0043 ("Correction on ZOD generation", v19.5) are newer than any 3GPP calibration campaign (see `md/3gpp_calibration_references.md`).

How to read the table:

- **Sections:** numbered after the TR. Clause 7.x is section 3.x, clause 8 is section 4 and Annex A is section 5, each titled with the TR heading. Where a clause holds several functional blocks (7.4, 7.6, 7.8, 7.9), sub-clause 7.x.y has its own section 3.x.y (for example 7.6.1 oxygen absorption is 3.6.1). The other clauses are one table each, and the TR ref column gives the sub-clause of each row.
- **Covered:** **Yes** = implemented, possibly with a different API. **Partial** = some parts exist, or only building blocks that the caller has to combine. **No** = not implemented. Two informative sub-clauses that define no model are marked n/a.
- **Function, location:** the public function with the line where its definition starts, followed by the line of the block that implements the feature (`ssf_clusters` (`src/ssf_clusters.cpp:440`; block `:651`)).
- **Comment:** API differences, then deviations. A deviation that changes the result compared with the TR is marked **Drift:**. API differences are allowed and are noted for orientation only. These conventions recur and are not repeated in every row:
  - Angles are **elevation angles** (EoD, EoA = 90° − ZOD, ZOA; +90° points up) in **radians** in the GCS. Spreads are the same in both conventions (ESD = ZSD, ESA = ZSA). The EoD offset is −μ_offset,ZOD.
  - Antenna orientations are **Euler angles (bank, tilt, heading)** in rad, corresponding to the TR's (γ, β, α) of clause 7.1; a positive tilt points down, as the TR β.
  - Motion is given as **positions per snapshot**, not as velocity vectors; the Doppler term uses the displacement form r̂·Δr.
  - The cluster output has **N + 1 slots** (slot 0 = LOS path, zero power in NLOS; an optional ground-reflection slot; then the N NLOS clusters, not sorted by delay) and holds **coefficient powers** (LOS K/(K+1), NLOS P_n/(K+1)) (ssf_notes D4, D7, W3).
  - `ssf_clusters` has three methods: `38.901` (default, clause 7.5), `38.901_MF` (clause 7.6.5.1) and `quadriga` (a redesign of the QuaDRiGa method, not TR). The TR rows refer to `38.901` unless they say otherwise.

### Maintaining this inventory

The inventory is useful only while it matches the code, so a change that adds, removes or alters a TR feature updates its rows in the same commit. The section numbers follow the TR: clause 7.x is section 3.x, clause 8 is section 4 and Annex A is section 5. Clauses that hold several functional blocks (7.4, 7.6, 7.8, 7.9) are split into sections 3.x.y that carry the TR sub-clause number (for example 7.6.1 oxygen absorption is 3.6.1); the other clauses are one table each. A new feature goes into the section of its TR clause or sub-clause, so it can be found from the TR number alone. Deeper sub-clauses (such as 7.6.3.2) and individual steps are rows, and a sub-clause that gains a second functional block gets a second row rather than a longer comment. When a one-table clause grows to several functional blocks, or a new TR version adds a sub-clause to a split clause, it gets sections with the TR numbers. The points below keep the document consistent:

- **Rows:** set Covered from what the library itself does, not from what a caller could assemble. Give the definition line of the public function and the line of the block that implements the feature. State API differences and drifts separately, and keep the conventions listed above out of the individual rows.
- **Evidence:** a drift that a test or a run has measured cites that test (file and line) and drops "(from reading, not measured)". Remove the matching item from section 6 once it has been run, or replace it with the result. A new drift found only by reading keeps the marker and gets a proposed run in section 6.
- **Line numbers:** they move with every edit to a cited file. When the document is refreshed, update the commit in "Library state" and recheck the cited lines against that commit (for example with `git show <commit>:src/<file>.cpp | sed -n <line>p`). A row whose feature did not change still needs its line numbers checked.
- **Summary:** after editing rows, recount the table in section 2 from the rows, and update the summary paragraphs and the drift list when a main gap closes or a drift is fixed.
- **New TR version:** update the version in the first paragraph and in "TR reference", read the new CRs in Annex B, recheck the scenario files against the parameter tables, and add rows for new sub-clauses.
- **Citations:** code, tests and scenario files do not cite this document; it cites them. The document is public, so it holds only technical content.

## 2. Summary

The large-scale model, the cluster and ray steps of clause 7.5, path loss, LOS probability and O2I loss, and most of the clause 7.6 components that act on a single link are implemented. The scenario files hold the v19.5.0 table values: no number in Tables 7.5-6 and 7.5-7 to 7.5-12 differs from the TR, and the path-loss, correlation-distance, absolute-TOA, cluster-count and polarisation-variability values match as well.

Not implemented: the link-level CDL and TDL models (7.7, only building blocks), blockage models A and B (7.6.4), mobility procedures A and B (7.6.3.2), the NLOS part of the near-field model (7.6.13), spatial non-stationarity (7.6.14), the 7.6.2.2 large-bandwidth ray model for the TR methods, mobile scatterers (7.6.10), the ISAC model (7.9), and the stochastic part of the map-based hybrid model (clause 8 steps 4–12). The deterministic part of clause 8 is covered by the library's ray tracer. Apart from sample directions of the handheld UT patterns (Table 7.8-2A, metric 5), no 3GPP calibration (7.8) has been reproduced with the library.

Main drifts in implemented features (details in the rows):

- The angular spread of `calc_angular_spread` is not the circular standard deviation of Annex A.1, although its doc calls it the TR definition. It agrees for small spreads only.
- The RMa ZOD offset uses the SMa form with the actual heights and the hUT < 3.5 m gate. The TR fixes hBS = 35 m and hUT = 1.5 m.
- The ground permittivity of the ground reflection uses QuaDRiGa's fits. Medium and wet ground differ from TR Table 7.6.8-1.
- Oxygen absorption omits τ_Δ of eq. 7.6-1 (too little loss in NLOS without absolute TOA) and the in-band variation of eq. 7.6-2.
- Defaults that differ from the TR unless the caller sets them: SOS preset `Comb256` with bias removal (TR: exponential ACF, `Exp256` without bias removal), InF LOS hall V/S = 4.27, RMa/SMa LOS `SF_sigma` = 5 dB when `sf_sigma_dB` from `path_loss` is not passed.
- Small numeric deviations: RMa/SMa PL2 with the 3D breakpoint distance (≤ 0.034 dB), the rounded constants 13.33 and 20.94 (≤ 2e-3 dB), and the K-factor limit K ≥ −5 dB in C_τ, C_φ and C_θ (ssf_notes W2).

Rows per clause (a "Yes" row may still carry a drift; see the comments):

| Clause (section) | Rows | Yes | Partial | No | n/a |
|---|---|---|---|---|---|
| 7.1 Coordinate system (3.1) | 6 | 6 | 0 | 0 | 0 |
| 7.2 Scenarios (3.2) | 5 | 0 | 5 | 0 | 0 |
| 7.3 Antenna modelling (3.3) | 8 | 8 | 0 | 0 | 0 |
| 7.4 Path loss, LOS probability, penetration (3.4) | 20 | 17 | 3 | 0 | 0 |
| 7.5 Fast fading model incl. parameter tables (3.5) | 46 | 43 | 2 | 1 | 0 |
| 7.6 Additional modelling components (3.6) | 36 | 17 | 4 | 14 | 1 |
| 7.7 Link-level models (3.7) | 16 | 1 | 8 | 7 | 0 |
| 7.8 Calibration (3.8) | 14 | 0 | 10 | 4 | 0 |
| 7.9 ISAC (3.9) | 23 | 0 | 8 | 15 | 0 |
| 8 Map-based hybrid model (4) | 23 | 9 | 5 | 8 | 1 |
| Annex A (5) | 6 | 1 | 4 | 1 | 0 |

## 3. Inventory: clause 7, channel models for 0.5–100 GHz

### 3.1 Coordinate system (7.1)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Spherical angles and field components | 7.1.1, Fig. 7.1.1 | Yes | `interpolate_pattern` (`src/interpolate_pattern.cpp:207`); kernel `qd_interpolate_pattern_GENERIC` (`src/interpolate_pattern_generic.cpp:14`; basis vectors `:95`) | Elevation instead of zenith; azimuth equals the TR φ. `e_theta`/`e_phi` are F_θ/F_φ. Patterns are sampled on an elevation grid [−π/2, π/2] and interpolated. |
| Local and global coordinate systems | 7.1.2 | Yes | `arrayant` class (`include/quadriga_arrayant.hpp`); `interpolate_pattern` (`src/interpolate_pattern.cpp:207`) | Patterns and `element_pos` are given in the LCS; the array position is the Tx/Rx position, its rotation the `orientation` input (per array, element or angle). |
| LCS→GCS rotation matrix | 7.1.3, eq. 7.1-2 to 7.1-5 | Yes | `calc_rotation_matrix` (`src/calc_rotation_matrix.cpp:39`); `qd_rotation_matrix` (`src/qd_helpers_shared.hpp:254`); call in `src/interpolate_pattern.cpp:80` | Same R = Rz·Ry·Rx. Euler angles (bank, tilt, heading) = (γ, β, α): a positive tilt points the antenna down, as the TR β, in all functions; pinned by the test cases "A positive tilt points down" and `tests/catch2_tests/test_interpolate_pattern.cpp:471`. |
| GCS→LCS angle mapping | 7.1.3, eq. 7.1-6 to 7.1-8 | Yes | kernel (`src/interpolate_pattern_generic.cpp:14`; block `:82`) | Cartesian R⁻¹·ρ̂ with atan2/asin, equivalent to 7.1-7/7.1-8. Local angles are an optional output. |
| Polarisation rotation ψ | 7.1.3, eq. 7.1-9 to 7.1-17 | Yes | kernel (`src/interpolate_pattern_generic.cpp:14`; block `:95`) | ψ from dot products of the rotated local θ̂ with the global θ̂, φ̂ (the approach of 7.1-12). Optional output `gamma`. |
| Downtilt-only transformation | 7.1.4, eq. 7.1-18 to 7.1-26 | Yes | as 7.1.3 | General rotation with bank = heading = 0; no separate code path. |

### 3.2 Scenarios (7.2)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| UMi-street canyon and UMa | 7.2, Table 7.2-1 | Partial | `3GPP_38.901_v19_UMi_*.conf`, `3GPP_38.901_v19_UMa_*.conf` | Propagation parameters only (LOS, NLOS, O2I files). The 19-site hexagonal layout, ISD, BS heights, the hUT floor rule, the 80 % indoor ratio, minimum distances and the UT drop are the caller's; the library has no layout or drop helper. |
| Indoor office (open, mixed) | 7.2, Table 7.2-2 | Partial | `3GPP_38.901_v19_Indoor_{LOS,NLOS}.conf`; `los_probability` `38.901_InH_Open`, `38.901_InH_Mixed` | One file pair for both offices; the TR says they differ only in LOS probability. Room layout and heights are the caller's. |
| RMa | 7.2, Table 7.2-3 | Partial | `3GPP_38.901_v19_RMa_*.conf` | Layout, heights and the indoor / in-car split are the caller's; in-car UTs use `o2i_loss` model `car`. |
| InF (SL, DL, SH, DH, HH) | 7.2, Table 7.2-4 | Partial | `3GPP_38.901_v19_InF_LOS.conf`, `_InF_NLOS_{SL,DL,SH,DH}.conf`; `los_probability` `38.901_InF_*` | No HH file: HH is LOS only, so the LOS file plus `38.901_InF_HH` (p = 1) covers it. Hall size and clutter parameters are inputs. |
| SMa (Rel-19) | 7.2, Table 7.2-5 | Partial | `3GPP_38.901_v19_SMa_*.conf` | Layout, the residential/commercial split, floor heights and the in-car share are the caller's. |

### 3.3 Antenna modelling (7.3)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| BS panel array (Mg, Ng, M, N, P) | 7.3.0, Fig. 7.3-1 | Yes | `generate_antenna` type `3GPP` (`src/generate_antenna.cpp:846`; tuples filled in `qd_panel_fill` `:43`, positions and port weights in `qd_arrayant_panel` `:378`) | Tuples `panel = [M, N, Mg, Ng]`, `spacing = [d_V, d_H, d_gV, d_gH]` in λ (d_H = d_V, panel spacing M·d_V and N·d_H by default; 0 allowed for co-located elements). Element order: polarisation, then m (upwards along z), n (along +y), Mg, Ng, as Fig. 7.3-1; positions centred. Tested against the configurations of Tables 7.8-2 (configs 1, 2), 7.8-2A (configs 3, 4), 7.8-6A (configs 1, 2) and 7.8-6C: counts, positions and weights in closed form ("TR 38.901 calibration configurations"). |
| BS element pattern | 7.3.0, Table 7.3-1 | Yes | `qd_tr38901_element` (`src/generate_antenna.cpp:281`), `element = [φ_3dB, θ_3dB, SLA_V, A_max, G_max]` | Default 65°, 65°, 30 dB, 30 dB, 8 dBi; G_max NaN normalises to the directivity (9.83 dBi for the default). Shape tested within 1e-10 dB for the Table 7.3-1 and the Table 7.3-2 constants, G_max NaN against the directivity within 0.01 dB. |
| UT antenna model (handheld, CPE: positions, device size, Ω_UT) | 7.3.0, Fig. 7.3-2 to 7.3-6 | Yes | `generate_antenna` types `3GPP-UT`, `3GPP-CPE` (`qd_generate_device_array`, `src/generate_antenna.cpp:474`), `device_size`, `locations` | Handheld: 8 locations on a 15 × 7 cm rectangle, in any order, repeats allowed; positions and bearings from the geometry (25.02° against 25° in A.6; the company results in R1-2506406 use both, the fields differ by at most 0.0016 at a peak of 1.84). CPE: 9 locations on a 20 × 20 cm square in the y-z plane, elements facing +x, Fig. 7.3-6 read as seen from the front (our reading). The TR gives no CPE element; the default is the UE element of TR 38.803 Table 5.2.3.3-1 (90°, 25 dB, 5 dBi). Ω_UT is the receiver orientation of the channel functions, not a generator input. |
| Handheld UT element pattern | 7.3.0, Table 7.3-2 | Yes | `generate_antenna` type `3GPP-UT`, `element` default `[125, 125, 22.5, 22.5, 5.3]` (`qd_tr38901_element`, `src/generate_antenna.cpp:281`) | Shape tested within 1e-10 dB. The antenna-imbalance note is not implemented (a per-port random loss, a channel-side draw). |
| Port mapping (TXRU sub-arrays with electrical tilt and pan) | 7.3.1, eq. 7.3-1 | Yes | `generate_antenna` type `3GPP`, `ports = [M_P, N_P]`, `steer = [tilt, pan]` per panel, `apply_weights` (`qd_arrayant_panel`, `src/generate_antenna.cpp:382`) | Eq. 7.3-1 per sub-array of M/M_P elements (1/√K, phase reference at the sub-array centre, equal to 7.3-1 up to a constant phase per port); horizontal weights for `pan` by analogy along y with d_H (a reading, not TR text). `tilt` is the electrical downtilt (θ_etilt = 90° + tilt). Weights in the coupling matrix by default; `apply_weights` combines each port into one pattern at the sub-array centre (equal within 1e-12). Tilt and pan signs pinned by tests ("a positive electrical tilt points the beam down", "a positive pan turns the beam towards +y"). |
| Polarised antenna Model-1 | 7.3.2, eq. 7.3-2 to 7.3-3b | Yes | `qd_apply_polarization` (`src/generate_antenna.cpp:310`), `pol = [2 or 3, 1]`; `rotate_pattern` usage 3 (`src/pattern_transform.cpp`) | The element is turned by ζ about its boresight, pattern and polarisation together, so F″ is evaluated at (θ″, φ″) as in eq. 7.3-3; also applied to a one-element custom `pattern`. Tested against eq. 7.3-3 with A″(θ″, φ″) on and off the rotation axis; on the axis the polarisation is not rotated (8551c68). |
| Polarised antenna Model-2 | 7.3.2, eq. 7.3-4, 7.3-5 | Yes | `qd_apply_polarization` (`src/generate_antenna.cpp:310`), `pol = [2 or 3, 2]` | F_θ = √A′ cos ζ, F_φ = √A′ sin ζ with A′ = \|F_θ\|² + \|F_φ\|² of the element; also for a one-element custom `pattern` (e.g. the isotropic element of the panel UT configs). Tested against eq. 7.3-4/7.3-5 within 1e-12. |
| Handheld UT polarised model (CR 0032) | 7.3.2, eq. 7.3-6 to 7.3-8, Fig. 7.3-7 to 7.3-10 | Yes | `generate_antenna` type `3GPP-UT`, `pol` `[1, 2]` and `[2, 2]`; `rotate_pattern` usage 3 | Each antenna is the reference element rotated by (γ_u, 0, α_u). Tested against eq. 7.3-6 to 7.3-8 for all 8 antennas with one and two field patterns (0.01 on a 5° grid, peak 1.84) and against 8 sample directions of the InterDigital results in R1-2506406, Table 7.8-2A metric 5 (1e-4 on a 1° grid). The other z'' direction of Fig. 7.3-7 is the same antenna with a negated port weight; other `pol` values are extensions. |

### 3.4 Pathloss, LOS probability and penetration modelling (7.4)

#### 3.4.1 Pathloss (7.4.1)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Distance definitions (d2D, d3D, indoor parts) | 7.4.1, Fig. 7.4.1-1/-2, eq. 7.4-1 | Yes | `path_loss` (`src/path_loss.cpp:181`; distances `:411`) | Geometric distances from the positions; d3D = d3D-out + d3D-in for indoor UTs; clamped at ≥ 0.1 m. The optional `d2d_in` input gives the legacy outdoor-only distance (not TR). |
| RMa LOS and NLOS | 7.4.1, Table 7.4.1-1 | Yes | `path_loss` (`src/path_loss.cpp:181`); `qd_pl_dual_slope` `:47` (breakpoint `:62`); NLOS block `:525`; `3GPP_38.901_v19_RMa_*.conf` | Constants are the TR formula at h = 5 m, W = 20 m (other h, W need new keys). NLOS = max(LOS, NLOS′). **Drift:** PL2 uses the 3D breakpoint distance where the TR uses d_BP, ≤ 0.034 dB (source comment `:40`, `tests/catch2_tests/test_path_loss.cpp:350`); E = 20.94 instead of 2π/0.3. Validity ranges are not enforced. |
| UMa LOS and NLOS, random h_E | 7.4.1, Table 7.4.1-1, note 1 | Yes | `path_loss` (`src/path_loss.cpp:181`); `qd_pl_dual_slope` `:47`; h_E draw `:388`; `3GPP_38.901_v19_UMa_*.conf` | h_E per note 1 (C′ = 1 above 23 m, which the TR leaves open). **Drift:** E = 13.33 instead of 40/3 (≤ 2.1e-3 dB, `test_path_loss.cpp:338`). h_E comes from the spatially consistent column 10 of `gbsm_draw("lsf")`; the TR draws it independently per link, so the loss of a moving UT steps where h_E changes (API doc `src/path_loss.cpp:135`, `test_path_loss.cpp:510`). Without `rnd_he`, h_E = 1 m. |
| UMi LOS and NLOS | 7.4.1, Table 7.4.1-1 | Yes | as UMa; `3GPP_38.901_v19_UMi_*.conf` | h_E = 1 m. **Drift:** E = 13.33 (≤ 2.1e-3 dB). |
| InH LOS and NLOS | 7.4.1, Table 7.4.1-1 | Yes | `path_loss` `logdist` (`src/path_loss.cpp:449`), NLOS (`:525`); `3GPP_38.901_v19_Indoor_*.conf` | Exact (≤ 1e-9 dB, `test_path_loss.cpp:335`). |
| InF LOS, SL, DL, SH, DH | 7.4.1, Table 7.4.1-1 | Yes | NLOS block (`src/path_loss.cpp:525`; extra term `:537`); `3GPP_38.901_v19_InF_*.conf` | Exact, including max(PL, PL_LOS, PL_InF-SL) for DL. HH uses the LOS file. |
| SMa LOS and NLOS (Rel-19) | 7.4.1, Table 7.4.1-1 | Yes | `qd_pl_dual_slope` / NLOS block; `3GPP_38.901_v19_SMa_*.conf` | Constants at h = 10 m, W = 10 m; NLOS without max with LOS, as in the TR. **Drift:** 3D breakpoint in PL2 (≤ 0.031 dB), E = 20.94. |
| Optional NLOS path loss (UMa, UMi, InH) | 7.4.1, Table 7.4.1-1 "Optional" | Partial | `path_loss` `logdist` (`src/path_loss.cpp:449`) | The formula can be expressed with `logdist` keys, but no file ships these values. |
| Shadow-fading std per scenario | 7.4.1, Table 7.4.1-1 | Yes | `path_loss` output `sf_sigma_dB` (`src/path_loss.cpp:557`); conf `SF_sigma`, `PL_sig1`, `PL_sig2` | RMa and SMa LOS switch 4/6 dB at d_BP; all other values match. The output must be passed on to `lsp_generate` (see step 4). |
| Frequency validity | 7.4.1, note 2 | Partial | `path_loss` API doc (`src/path_loss.cpp:101`) | Evaluated at any fc > 0; the TR ranges are stated only in the conf headers. The LSP fc floor (`LSP_fc_min`) is correctly not applied to path loss. |

#### 3.4.2 LOS probability (7.4.2)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| RMa, UMi, UMa | 7.4.2, Table 7.4.2-1 | Yes | `los_probability` (`src/los_probability.cpp:164`; RMa `:262`, UMi `:266`, UMa `:270`) | Exact. The caller passes d2D-out. UMa uses C′ = 1 above 23 m and clips to [0, 1] (the formula reaches 1.006 just above 18 m). Test `tests/catch2_tests/test_los_probability.cpp:191`. |
| InH mixed and open office | 7.4.2, Table 7.4.2-1 | Yes | `los_probability` (`:280`) | Exact, including the interval edges. |
| InF SL, DL, SH, DH, HH | 7.4.2, Table 7.4.2-1 | Yes | `los_probability` (defaults `:225`, SL/DL `:292`, SH/DH `:296`, HH `:306`) | Exact for hBS > hUT and hUT < h_c; the library defines the cases the TR leaves open (p = 1 for hUT ≥ h_c). Defaults are the Table 7.8-7 clutter values. |
| SMa (Rel-19) | 7.4.2, Table 7.4.2-1 | Yes | `los_probability` (parameters `:233`, block `:308`) | r_veg is the input `r_clutter` (default 0). **Drift (outside the TR range only):** max(h − hUT, 0) is applied to all three terms, the TR applies it to the residential term only; different only for hUT ≥ 15 m (`test_los_probability.cpp:381`). |

#### 3.4.3 O2I penetration loss (7.4.3)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Building loss: low, high, low-loss A | 7.4.3.1, eq. 7.4-2/-3, Tables 7.4.3-1/-2 | Yes | `o2i_loss` (`src/o2i_loss.cpp:71`; materials `:145`, σ_P `:151`, loss `:174`) | Values match, including the Rel-19 IRR glass (25.4 + 0.11f). One N(0,1) draw per UT for all frequencies. The model mix is chosen per UT by `weights`; random inputs come from the caller. Tests `tests/catch2_tests/test_o2i_loss.cpp:78`, `:211`. |
| General material mix, wood | 7.4.3.1, eq. 7.4-3, Table 7.4.3-1 | Partial | `o2i_loss` (`src/o2i_loss.cpp:71`) | Only the fixed mixes of Table 7.4.3-2; no user-defined mix, wood unused. |
| Indoor 2D distance | 7.4.3.1 | Yes | `o2i_loss` (`:176`); conf `O2I_d2D_in_max` | min(U1, U2)·d_max: 25 m (UMa, UMi, SMa), 10 m (RMa). SMa residential buildings (10 m) need a second call; the SMa files hold the commercial 25 m. |
| Backward-compatible O2I below 6 GHz | 7.4.3.1, Table 7.4.3-3 | Yes | `o2i_loss` model `legacy_6GHz` (`:181`) | PL_tw = 20 dB, σ_P = 0; σ_SF = 7 dB from the UMa/UMi O2I files. Test `test_o2i_loss.cpp:115`. |
| Car penetration loss | 7.4.3.2, eq. 7.4-4 | Yes | `o2i_loss` models `car`, `car_metal` (`:187`) | N(9, 5²) dB, metallised windows μ = 20 dB. Test `test_o2i_loss.cpp:141`. |

#### 3.4.4 Autocorrelation of shadow fading (7.4.4)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Exponential SF autocorrelation | 7.4.4, eq. 7.4-5 | Yes | `gbsm_init_phases` (`src/qd_gbsm_phases.cpp:431`), `gbsm_draw` (`:607`), "lsf" generators (`:82`); `sos_preset` (`src/sos_preset.cpp:1563`); conf `SF_lambda` | Sum-of-sinusoids fields in the horizontal plane, per floor; all `SF_lambda` values match. **Drift (default):** preset `Comb256` (Gaussian up to d_cor, exponential beyond) with bias removal; the TR-literal setting is `Exp256` without bias removal (API doc `:368`). |

### 3.5 Fast fading model (7.5)

#### Model steps

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Angle notation (LOS, cluster, ray) | 7.5, Table 7.5-1, Fig. 7.5-2 | Yes | `ssf_clusters` (`src/ssf_clusters.cpp:440`; LOS angles `:605`; zenith/elevation helper `:30`) | Elevation in rad, azimuth wrapped to [−π, π); N + 1 slots, not sorted (ssf_notes D4, W3). |
| Uplink by reciprocity (CR 0035) | 7.5 introduction | Partial | `ssf_clusters`, `ssf_rays` | No uplink switch; the caller generates the downlink and transposes it. A Tx/Rx swap with mirrored random fields is exactly reciprocal only for `eod_offset_deg` = 0 and no O2I link (ssf_notes A7, D2). |
| Step 1a: scenario, GCS | 7.5 step 1a | Yes | `conf_file_read` (`src/qd_scenpar.cpp:866`); 23 files `data/config/3GPP_38.901_v19_*.conf` | One file per scenario and LOS/O2I state (InH is `Indoor`; one InF NLOS file per clutter type). The TR frequency limits are stated in the file comments only. |
| Step 1b–c: positions, LOS angles | 7.5 step 1b, 1c | Yes | `ssf_clusters` (`src/ssf_clusters.cpp:440`; LOS geometry `:600`) | Tx is the BS (downlink). Wrap-around copies are the caller's. |
| Step 1d–e: patterns, arrays, orientations | 7.5 step 1d, 1e | Yes | `gbsm_single_link_channel` (`src/gbsm_single_link_channel.cpp:117`) | `arrayant` in the LCS, Euler orientations (see 7.1). |
| Step 1f: UT speed and direction | 7.5 step 1f | Yes | `gbsm_single_link_channel` (`src/gbsm_single_link_channel.cpp:117`) | Positions per snapshot (`mobility` `phase` or `drift`). |
| Step 1g: carrier frequency | 7.5 step 1g | Yes | `center_freq` [Hz] in all GBSM functions | The bandwidth matters only for 7.6.2. |
| Step 2: LOS/NLOS and indoor state | 7.5 step 2 | Partial | `los_probability` (`src/los_probability.cpp:164`); workflow in section doc `src/qd_scenpar.cpp:721` | The library gives p_LOS. The caller draws the states (spatially consistent with the SOS generators at `LOS_state_lambda`, `IO_state_lambda`) and picks the scenario file. The LOS condition is a property of the file: LOS when `KF_mu > −30` dB (ssf_notes W1). |
| Step 3: path loss | 7.5 step 3 | Yes | `path_loss` (`src/path_loss.cpp:181`) | See 3.4.1. |
| Step 4: log-normal LSPs, frequency terms, fc floors | 7.5 step 4, Table 7.5-6 notes | Yes | `lsp_generate` (`src/lsp_generate.cpp:177`; per-link loop `:317`); `qd_gbsm_lsp_stats` (`src/qd_helpers_gbsm.hpp:199`; fc floor `:226`) | μ = P_μ + P_γ·lg(P_ω + max(fc, `LSP_fc_min`)), floors UMa 6, UMi 2, InH 6 GHz as in the notes. Output `lsp`: DS in s, spreads in deg, K, SF, XPR linear; `lsp_mu`/`lsp_sigma` in the log domain. NLOS K "N/A" is `KF_mu` = −100 dB. |
| Step 4: SF standard deviation | 7.5 step 4 (σ_SF from Table 7.4.1-1) | Yes | `lsp_generate` input `sf_sigma_dB` (`src/lsp_generate.cpp:318`) | **Drift (default):** without `sf_sigma_dB` from `path_loss`, RMa and SMa LOS use the file value 5 dB instead of 4/6 dB before/after the breakpoint (conf comment; `tests/catch2_tests/test_lsp_generate.cpp:811`). |
| Step 4: cross-correlation | 7.5 step 4 | Yes | Cholesky `src/lsp_generate.cpp:239`, symmetric root `:253`; matrix `src/qd_scenpar.cpp:309` | Default Cholesky in TR order [SF, K, DS, ASD, ASA, ZSD, ZSA]. XPR is an 8th, uncorrelated LSP (QuaDRiGa extension) placed last, so it does not change the others. All 23 files are positive definite (`test_conf_files_v19.cpp:167`). `cholesky = false` is QuaDRiGa's symmetric root (not TR). |
| Step 4: spread clipping (104°, 52°) | 7.5 step 4 | Yes | limits `src/lsp_generate.cpp:292`, applied `:322` | Default on; `clip_spreads = false` is a QuaDRiGa option. |
| Step 4: co-sited sectors, different sites and floors | 7.5 step 4, Table 7.5-6 note 3 | Yes | `gbsm_init_phases` (`src/qd_gbsm_phases.cpp:431`); `gbsm_draw` floor offsets (`:244`) | One seed per site and scenario; sectors of a site share the LSPs; other sites need other seeds. **Drift:** floors are exactly uncorrelated only at equal horizontal positions; elsewhere a residual correlation up to about 0.16 remains (source comment `:234`, doc `:538`). |
| Step 4: LSP decorrelation distances | 7.5 step 4, Table 7.5-6 | Yes | `gbsm_draw` "lsf" generators (`src/qd_gbsm_phases.cpp:82`); conf `*_lambda` | All distances match. Default SOS preset as in 3.4.4. `XPR_lambda` has no TR source. |
| Step 4: ZSD mean vs d2D, hUT, hBS, fc | Tables 7.5-7 to 7.5-12 | Yes | `src/qd_helpers_gbsm.hpp:281`; conf `ES_D_mu_*` | Written as ESD; the TR constant 0.01·1.5 is folded into `ES_D_mu`. Test `test_conf_files_v19.cpp:294`. |
| Step 4: ZOD offset μ_offset,ZOD | Tables 7.5-7 to 7.5-12 | Yes | `lsp_generate` EoD offset block (`src/lsp_generate.cpp:332`); conf `EoD_offset_*` | Output `eod_offset_deg` = −μ_offset,ZOD. UMa NLOS uses the v19 form with the fc = 6 GHz floor; UMi, SMa, InH, InF follow the TR (`test_lsp_generate.cpp:530`). **Drift (RMa NLOS, O2I):** the library uses the SMa form with the actual hBS, hUT and the gate hUT < 3.5 m; the TR fixes hBS = 35 m, hUT = 1.5 m and has no gate. Equal at 35/1.5 m; 0 instead of about 1° for hUT ≥ 3.5 m (from reading, not measured). |
| Step 4: InF DS with hall V/S | Table 7.5-6 Part-3 | Yes | DS term `src/qd_helpers_gbsm.hpp:258`; conf `DS_VS`, `DS_VS_a`, `DS_VS_b` | NLOS files hold the V/S of their calibration hall (4 or 4.55). **Drift (default):** `InF_LOS.conf` uses the average 4.27 for all halls, about ±0.024 in μ_lgDS (55.8 ns instead of 52.7 or 59.0 ns median) unless the caller sets `DS_VS` (conf comment; from reading, not measured). |
| Step 5: exponential cluster delays | 7.5 step 5, eq. 7.5-1/-2 | Yes | `ssf_clusters` (`src/ssf_clusters.cpp:440`); kernel `qd_ssf_tr` (`:43`; delays `:47`) | No sort: the TR's first cluster is the one with the smallest delay (W3). X_n from a spatially consistent field (D6). Delays in s relative to the LOS delay. Method `38.901` accepts one frequency only (D2). |
| Step 5: uniform delay distribution | 7.5 step 5 | No | – | Only the exponential distribution, which Table 7.5-6 prescribes for all scenarios. |
| Step 5: LOS delay scaling C_τ | 7.5 step 5, eq. 7.5-3/-4 | Yes | `qd_ssf_tr` (`src/ssf_clusters.cpp:43`) | **Drift:** K limited to ≥ −5 dB in the polynomial (ssf_notes W2; `tests/catch2_tests/test_ssf_clusters.cpp:351`). LOS chosen per scenario file, not per link (W1). |
| Step 6: cluster powers with per-cluster shadowing | 7.5 step 6, eq. 7.5-5/-6 | Yes | `qd_ssf_tr` (`src/ssf_clusters.cpp:47`); conf `r_DS`, `LNS_ksi` | Normalised in the log domain (W18). Z_n from the per-frequency random set (D6). |
| Step 6: LOS power and LOS cluster powers | 7.5 step 6, eq. 7.5-7/-8 | Yes | `qd_ssf_tr` (`:77`); LOS slot `src/ssf_clusters.cpp:705` | Output = coefficient powers of 7.5-30; the 7.5-8 powers are used inside the angle mapping only (D7, A16). |
| Step 6: −25 dB cluster removal | 7.5 step 6 | Yes | `qd_ssf_tr` (`src/ssf_clusters.cpp:68`) | Criterion on the 7.5-6 powers; a removed cluster keeps its slot with power 0, no renormalisation (D5, A10; `test_ssf_clusters.cpp:372`). |
| Step 6: ray power P_n/M | 7.5 step 6 | Yes | `ssf_rays` (`src/ssf_rays.cpp:180`; `:494`) | M = `NumSubPaths` (20 in all files). |
| Step 7: AOA/AOD with C_φ | 7.5 step 7, eq. 7.5-9 to 7.5-12, Table 7.5-2 | Yes | `qd_ssf_tr` (`src/ssf_clusters.cpp:74`; mapping `:103`, LOS alignment `:109`) | C_φ interpolated between table entries, constant outside (D5, A8). **Drift:** the K ≥ −5 dB limit also enters the LOS factor of 7.5-10 (W2). C_φ uses the cluster count before the −25 dB removal; the TR does not say which count (from reading, not measured). |
| Step 7: ZOA with C_θ, O2I ZOA = 90° | 7.5 step 7, eq. 7.5-14 to 7.5-17, Table 7.5-4 | Yes | `qd_ssf_tr` (`:97`, `:103`); O2I centre `src/ssf_clusters.cpp:637` | C_θ as C_φ; non-monotonic entries used as given (A8). The caller passes the `o2i` flag per link; the O2I file alone does not set 90°. |
| Step 7: ZOD with μ_offset,ZOD (CR 0043) | 7.5 step 7, eq. 7.5-19 | Yes | `qd_ssf_tr`; offset applied `src/ssf_clusters.cpp:636` | `eod_offset_deg` from `lsp_generate`, passed by the caller (omitted = 0). Applied in NLOS only; LOS follows 7.5-17 without offset, as in v19.5. |
| Step 7: ray offset angles | 7.5 step 7, eq. 7.5-13, 7.5-18, 7.5-20, Table 7.5-3 | Yes | `ssf_rays` (`src/ssf_rays.cpp:180`; table `:21`, c_ZSD `:445`, angles `:497`); conf `PerClusterAS_*`, `PerClusterES_A`, `PerClusterES_D_scale` | c_ZSD = 3/8·10^μ_lgZSD from `lsp_mu` slice 5 (A18). Table 7.5-3 at M = 20; other M use a deterministic non-TR rule (W24). |
| Step 7: zenith wrap | 7.5 step 7 | Yes | `qd_gbsm_elev` (`src/qd_helpers_gbsm.hpp:71`); ray fold `src/ssf_rays.cpp:333` | **Drift (rare cases):** the same fold is applied to ZOD, which the TR states for ZOA only, and the azimuth is not flipped (ssf_notes A12, A23, D5; `test_ssf_clusters.cpp:454`). |
| Step 8: random coupling of rays | 7.5 step 8 | Yes | `ssf_rays` (`src/ssf_rays.cpp:455`) | Rank fields give independent random permutations within each cluster and sub-cluster, the TR's joint distribution (A20). Spatially consistent, kept per drop (D14). |
| Step 9: per-ray XPR | 7.5 step 9, eq. 7.5-21 | Yes | `ssf_rays` (`src/ssf_rays.cpp:534`) | μ, σ from `lsp_mu`/`lsp_sigma` slice 7 in dB. `pol_model = "quadriga"` is a non-TR option. |
| Steps 1–9 shared by co-sited sectors | 7.5 step 9 | Yes | `ssf_rays` | Ray data are antenna-free; sectors differ only in the channel call. |
| Step 10: initial phases | 7.5 step 10 | Yes | `ssf_rays` (`src/ssf_rays.cpp:506`); `gbsm_draw` (`src/qd_gbsm_phases.cpp:607`) | Uniform in (−π, π) from a spatially consistent field; stored as Jones entries (θθ, φθ, θφ, φφ). |
| Step 11: NLOS coefficient | 7.5 step 11, eq. 7.5-22 to 7.5-24 | Yes | `gbsm_single_link_channel` (`src/gbsm_single_link_channel.cpp:117`; planar call `:322`); `get_channels_planar` (`src/get_channels_planar.cpp:109`; patterns `:263`, F_rx^T·M·F_tx `:309`, phase `:325`) | Each path also gets the phase −2π(d3D + cτ)/λ, which has no statistical effect (from reading, not measured). **Drift (wideband only):** delays are per element pair including the array geometry (7.6.2.1 always on), 0.2–2.3 ns on large arrays, where 7.5 uses one delay per cluster (ssf_notes section 14, D27). |
| Step 11: Doppler term | 7.5 step 11, eq. 7.5-22, 7.5-25 | Yes | `ssf_paths` (`src/ssf_paths.cpp:103`; `:246`, `:275`) via `gbsm_single_link_channel` | Displacement form, equal to r̂·v·t while r̂ is fixed (A25, mobility `phase`). `drift` is a non-TR mode (D11). Test `tests/catch2_tests/test_ssf_paths.cpp:209`. |
| Step 11: sub-clusters of the two strongest clusters | 7.5 step 11, eq. 7.5-26 to 7.5-28, Table 7.5-5 | Yes | `ssf_rays` (selection `src/ssf_rays.cpp:412`, groups `:308`, delays `:492`); c_DS `qd_gbsm_cds` (`src/qd_helpers_gbsm.hpp:123`) | Strongest = largest coefficient power; LOS and GR never split; only for M = 20 and the TR methods (A17, D13). c_DS = 3.91 ns where the TR gives N/A. UMa c_DS floor: see the UMa parameters row in 3.5. |
| Step 11: LOS coefficient | 7.5 step 11, eq. 7.5-29 | Yes | `ssf_rays` (`src/ssf_rays.cpp:397`); `get_channels_planar` (`src/get_channels_planar.cpp:325`) | Jones matrix diag(1, −1), phase from d3D, geometric angles. |
| Step 11: K-factor combination | 7.5 step 11, eq. 7.5-30 | Yes | `ssf_clusters` (`src/ssf_clusters.cpp:705`); `ssf_paths` (`src/ssf_paths.cpp:196`) | Already in the coefficient powers (D7). LOS and the first sub-cluster are two taps at delay 0: one more tap than the TR, same sum (`test_gbsm_single_link_channel.cpp:513`). `kf_ratio` is a non-TR option. |
| Step 12: path loss and shadowing | 7.5 step 12 | Yes | `ssf_paths` (`src/ssf_paths.cpp:301`); `gbsm_single_link_channel` | The caller passes `link_gain_dB` = −PL − O2I + 10·lg SF per snapshot (D24). |

#### Parameter tables

The scenario files hold the values of Table 7.5-6 and Tables 7.5-7 to 7.5-12; the formulas that use them are rows of the model steps above (step 4 and later).

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| UMi parameters | Table 7.5-6 Part-1, Table 7.5-8 | Yes | `3GPP_38.901_v19_UMi_*.conf` | All values match. The NLOS ZOD offset is also used in `UMi_NLOS_O2I`; the TR note covers only the ZSD, so this is the library's reading. |
| UMa parameters | Table 7.5-6 Part-1, Table 7.5-7 | Yes | `3GPP_38.901_v19_UMa_*.conf` | All values match, including the Rel-19 DS, ASA, ZSA, c_ASD and ZOD offset. **Drift (interpretation):** `PerClusterDS_fc_min = 6` applies the fc floor of note 6 also to c_DS: 3.91 instead of 4.71 ns at 3.5 GHz (ssf_notes section 15). |
| RMa parameters | Table 7.5-6 Part-2, Table 7.5-9 | Yes | `3GPP_38.901_v19_RMa_*.conf` | All table values match. **Drifts:** ZOD offset formula, LOS `SF_sigma` fallback (rows above). |
| InH parameters | Table 7.5-6 Part-2, Table 7.5-10 | Yes | `3GPP_38.901_v19_Indoor_*.conf` | All values match; the 6 GHz floor also applies to the ZSD frequency terms (note 4 of 7.5-10). |
| InF parameters | Table 7.5-6 Part-3, Table 7.5-11 | Yes | `3GPP_38.901_v19_InF_*.conf` | All values match; σ_SF per sub-scenario from Table 7.4.1-1. **Drift:** InF LOS V/S default (row above). |
| SMa parameters | Table 7.5-6 Part-4, Table 7.5-12 | Yes | `3GPP_38.901_v19_SMa_*.conf` | All values match, including the ZOD offset (h_cl = 3.5 m) and c_DS without fc floor. **Drift:** LOS `SF_sigma` fallback. |

### 3.6 Additional modelling components (7.6)

#### 3.6.1 Oxygen absorption (7.6.1)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Oxygen loss per cluster | 7.6.1, eq. 7.6-1, Table 7.6.1-1 | Yes | `ssf_paths` (`src/ssf_paths.cpp:103`; table `:210`, applied `:302`); switch `oxygen` of `gbsm_single_link_channel` | Off by default. Table at 52–68 GHz, linear interpolation, 0 outside (W27). Loss from each path's own length (sub-cluster rays up to +0.13 dB, A24). **Drift:** τ_Δ = min τ′_n is not added: in NLOS without absolute TOA the loss is too low (60 GHz p99: UMi 0.71 dB, UMa 1.28 dB; ssf_notes A19, section 14). |
| Oxygen loss across a large bandwidth | 7.6.1, eq. 7.6-2 | No | – | Evaluated at each `center_freq` only (A24). |

#### 3.6.2 Large bandwidth and large antenna array (7.6.2)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Per-element, per-ray propagation delay | 7.6.2.1, eq. 7.6-3, 7.6-4 | Yes | `get_channels_planar` (`src/get_channels_planar.cpp:109`; `:325`); `gbsm_single_link_channel` output `rays` | Delay τ − r̂·d/c per element pair; the frequency dependence comes through a frequency response. Output `taps` uses one mean delay per tap. The B > c/D test is the caller's. |
| Random per-ray angle offsets | 7.6.2.2 item 1, eq. 7.6-5 | No | – | Only the fixed offsets of Table 7.5-3 (ssf_notes section 16: future work). |
| Intra-cluster ray delays | 7.6.2.2 item 2 | Partial | `ssf_rays` (`src/ssf_rays.cpp:470`) | Only in the non-TR method `quadriga`, where delays are zero-mean per cluster instead of minimum-subtracted and the field uses its own decorrelation distance (D15). The TR methods keep the sub-cluster split. |
| Unequal ray powers | 7.6.2.2 item 3, eq. 7.6-6, 7.6-7 | No | – | Ray power is always P_n/M. |
| Number of rays from bandwidth and aperture | 7.6.2.2 item 4, eq. 7.6-8 | No | – | M is the key `NumSubPaths`; formula 7.6-8 is not computed. |

#### 3.6.3 Spatial consistency (7.6.3)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Spatially consistent cluster and ray variables | 7.6.3.1 | Yes | `gbsm_init_phases` (`src/qd_gbsm_phases.cpp:431`), `gbsm_draw` (`:607`; cluster set `:100`, ray set `:121`) | SOS fields at `SC_lambda` for delays, shadowing, signs and offsets, coupling ranks, XPR, phases, η; slots keep the cluster identity (W3, A2). The 7.5 steps stay discontinuous under spatial consistency (documented jumps, ssf_notes section 2). **Drift:** the offsets Y_n are kept along a track together with the signs; the TR keeps only the signs (D5, A1). Separately dropped UTs keep correlated taps over 0.3–0.8·`SC_lambda` (TR-literal; issue #25). |
| Correlation distances | Table 7.6.3.1-2 | Yes | conf `SC_lambda`, `LOS_state_lambda`, `IO_state_lambda` | All values match. The LOS and indoor distances are read by the caller. |
| Procedure A (cluster updates for moving UT/BS) | 7.6.3.2, eq. 7.6-9 to 7.6-14 | No | – | Not implemented (ssf_notes D1, section 16). Instead: mobility `phase` (fixed delays and angles, displacement phase only) and `drift` (fixed scatterers, geometry per position, non-TR; D11). |
| Procedure B | 7.6.3.2, eq. 7.6-15 to 7.6-17f | No | – | Dropped after measurement: as written it gives 0.47–0.74 (NLOS) and 0.14–0.19 (LOS) of the input spreads (ssf_notes D1, section 2). |
| LOS/NLOS and indoor state | 7.6.3.3 | Partial | `los_probability` (`src/los_probability.cpp:164`); `sos_rand`, `sos_randn` | The library gives p_LOS and the generators; the draw and the comparison are the caller's. |
| Soft LOS state | 7.6.3.3, eq. 7.6-18, 7.6-19 | Partial | `los_probability` (`src/los_probability.cpp:338`) | 7.6-18 per frequency with the caller's G (`rnd_g`); the two channel runs and their combination (7.6-19) are the caller's. Test `test_los_probability.cpp:413`. |
| O2I parameters | 7.6.3.3 | Yes | `o2i_loss` (`src/o2i_loss.cpp:71`; `:162`) | The caller supplies spatially consistent `rnd` at the TR distances. |
| Same floor correlated, different floors not | 7.6.3.3 | Yes | `gbsm_draw` floor IDs (`src/qd_gbsm_phases.cpp:222`) | **Drift:** residual floor correlation, see 3.5 step 4. |
| Correlation among TRPs, uncorrelated states | 7.6.3.4, Tables 7.6.3.4-1/-2 | Yes | seed rule of `gbsm_init_phases` (`src/qd_gbsm_phases.cpp:431`) | One seed per site and scenario; LOS, NLOS and O2I scenarios need different seeds (W21). |

#### 3.6.4 Blockage (7.6.4)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Blockage model A | 7.6.4.1, eq. 7.6-20 to 7.6-28 | No | – | |
| Blockage model B | 7.6.4.2, eq. 7.6-29, 7.6-30 | No | – | No dedicated model. Screens could be meshed and evaluated with `calc_diffraction_gain` (one Tx–Rx line, not per ray) or the ray tracer. |

#### 3.6.5 Correlation modelling for multi-frequency simulations (7.6.5)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Multi-frequency rules | 7.6.5 | Yes | `lsp_generate`, `ssf_clusters`, `ssf_rays`; per-frequency random sets | Same LSP random values at all frequencies, Z_n and steps 8–11 per frequency (D14). Method `38.901` rejects more than one frequency because 7.5 cannot keep delays and angles frequency-independent (A3, D2); use `38.901_MF`. |
| Alternative generation method | 7.6.5.1, eq. 7.6-30a to 7.6-30m | Yes | `ssf_clusters` method `38.901_MF` (`src/ssf_clusters.cpp:440`; kernel `qd_ssf_mf` `:128`) | TR-literal, including "wrap, then scale by √(1+K)" with its LOS jumps; none of the 7.5 rules added (D1, A4). Anchor LSPs are an input (`lsp_anchor`). Added: the 7.5 zenith fold on the final zenith angles (W4). Tests `test_ssf_clusters_mf.cpp:169`. |

#### 3.6.6 Time-varying Doppler shift (7.6.6)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Time-varying Doppler | 7.6.6, eq. 7.6-31 | Yes | `ssf_paths` (`src/ssf_paths.cpp:244`, `:275`) | Displacement form, exact for any trajectory while r̂ is fixed (A25). |

#### 3.6.7 UT rotation (7.6.7)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| UT rotation | 7.6.7 | Yes | `gbsm_single_link_channel` `rx_orientation`, `tx_orientation` per snapshot | Euler angles (7.1). |

#### 3.6.8 Explicit ground reflection model (7.6.8)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Ground reflection path | 7.6.8, eq. 7.6-32 to 7.6-40 | Yes | `ssf_clusters` (GR block `src/ssf_clusters.cpp:592`); `ssf_rays` (`src/ssf_rays.cpp:385`); `ssf_paths` (`src/ssf_paths.cpp:223`); key `GR_enabled` | Off in all files (TR optional). GR slot power K/(K+1)·(d3D/d_GR)²·R², coefficient diag(R∥, −R⊥)/R, equal to the TR form (W7). Additive, no renormalisation, as written (median +1.6 dB link power in UMi LOS; A14). Test `test_ssf_clusters.cpp:517`. |
| Ground permittivity | 7.6.8, eq. 7.6-41, 7.6-42, Table 7.6.8-1 | Partial | `lsp_generate` (`src/lsp_generate.cpp:360`) | **Drift:** QuaDRiGa's dry/medium/wet fits; only very dry ground agrees with the table (3.5 GHz medium 16.87 + j0.67 vs 13.23 + j1.39, wet j1.65 vs j3.93; source comment). The caller can pass table values as `gr_eps_re`, `gr_eps_im`. |

#### 3.6.9 Absolute time of arrival (7.6.9)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Absolute time of arrival | 7.6.9, eq. 7.6-43, 7.6-44, Table 7.6.9-1 | Yes | `lsp_generate` (`src/lsp_generate.cpp:349`); `ssf_clusters` (`src/ssf_clusters.cpp:677`); conf `absTOA_*` | All values match, including the 2L/c bound. Spatially consistent, NLOS files only (including O2I; the TR does not say whether O2I counts as NLOS). Tests `test_ssf_clusters.cpp:476`, `test_lsp_generate.cpp:900`. |

#### 3.6.10 Dual mobility (7.6.10)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Doppler with Tx and Rx moving | 7.6.10, eq. 7.6-45 | Yes | `ssf_paths` (`src/ssf_paths.cpp:244`); `gbsm_draw` `dual_mobility` | Both ends in the displacement form; mirrored random fields for swap reciprocity (D2, A7, A26). |
| Mobile scatterers | 7.6.10, eq. 7.6-46 | No | – | No α, D term (ssf_notes section 16). |

#### 3.6.11 Sources of EM interference (7.6.11)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Sources of EM interference | 7.6.11 | n/a | – | The TR defines no model; interferer links use the existing scenarios. |

#### 3.6.12 Embedded devices (7.6.12)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Embedded devices | 7.6.12 | No | – | No L_embedded; the caller can add it to `link_gain_dB`. |

#### 3.6.13 Near-field channel model (7.6.13)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Near-field LOS (element-wise phase and patterns) | 7.6.13, eq. 7.6-50, 7.6-51 | Yes | `get_channels_spherical` (`src/get_channels_spherical.cpp:145`; elements `src/qd_helpers_channel.hpp:233`, paths `:256`) | Exact distance and angles per element pair. Through `gbsm_single_link_channel` only with mobility `drift` (non-TR for the TR methods, D11). |
| Near-field NLOS (s_TRP, s_UE) | 7.6.13, eq. 7.6-47 to 7.6-49, Table 7.6.13-1 | No | – | `NF_*` keys are in the files and match the table but are not read (`src/qd_scenpar.cpp:1030`). `ssf_scatterers` places scatterers by a different rule (D16). |

#### 3.6.14 Spatial non-stationarity channel model (7.6.14)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| SNS at the TRP, blocker-based | 7.6.14.1, eq. 7.6-52 | No | – | |
| SNS at the TRP, stochastic visibility regions | 7.6.14.1, eq. 7.6-53 to 7.6-58 | No | – | `SNS_*` keys match the tables but are not read (`src/qd_scenpar.cpp:1030`). |
| SNS at the UT (grip attenuation) | 7.6.14.2, Tables 7.6.14.2-1/-2 | No | – | Could be expressed as per-element gains of the UT `arrayant`. |

#### 3.6.15 Number of cluster variability model (7.6.15)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Number-of-cluster variability | 7.6.15, Table 7.6.15-1 | Yes | `ssf_clusters` `n_cluster` (`src/ssf_clusters.cpp:547`, `:598`); conf `NumClusters`, `NumClusters_min` | Files hold N + 1 in both keys (D4, A9); ranges match the table; the choice is the caller's (W11). Test `test_ssf_clusters.cpp:936`. |

#### 3.6.16 Polarization power variability model (7.6.16)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Polarisation power variability | 7.6.16, eq. 7.6-59 to 7.6-62 | Yes | `ssf_rays` (`src/ssf_rays.cpp:539`); key `PolVar_sigma` | 3 dB in all files; NLOS rays only, `pol_model = "38.901"`. **Drift:** η from spatially consistent fields at `SC_lambda` instead of independent draws (same marginals; A22). |

### 3.7 Channel models for link-level evaluations (7.7)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| CDL-A to CDL-E tables | 7.7.1, Tables 7.7.1-1 to 7.7.1-5 | No | – | No tables or presets. |
| CDL step 1: ray angles | 7.7.1 step 1, eq. 7.7-0a | Partial | `ssf_rays` (`src/ssf_rays.cpp:180`; offsets `:30`) | The caller can pass the tabulated clusters as cluster cubes (elevations, rad, linear powers, LOS in slot 0) and the CDL c_ASD, c_ASA, c_ZSA, c_ZSD as `PerCluster*` keys (ssf_notes D10). Untested route (from reading, not measured). |
| CDL step 2: random coupling | 7.7.1 step 2 | Partial | `ssf_rays` (`src/ssf_rays.cpp:455`) | **Drift:** with M = 20 and the TR methods the two strongest clusters are always split into sub-clusters (delays +1.28·c_DS, +2.56·c_DS, coupling within sub-clusters); CDL treats all clusters as unsplit. No option turns this off (from reading, not measured). |
| CDL step 3: per-ray XPR | 7.7.1 step 3, eq. 7.7-0b | Partial | `ssf_rays` (`src/ssf_rays.cpp:534`) | Fixed CDL XPR via μ = table value and σ = 0 supplied by the caller. |
| CDL step 4: coefficients | 7.7.1 step 4 | Partial | `gbsm_single_link_channel` (`src/gbsm_single_link_channel.cpp:117`) | Mobility `phase` with the ray data; the step-2 drift applies. |
| Brick-wall window | 7.7.1 | No | – | |
| TDL-A to TDL-E tables | 7.7.2, Tables 7.7.2-1 to 7.7.2-5 | No | – | `get_channels_ieee_indoor` is the IEEE TGn model; `quantize_delays` (`src/qd_channel_quantize_delays.cpp:234`) maps paths to a tap grid. Neither is TR TDL. |
| Jakes Doppler spectrum per tap, Ricean first tap | 7.7.2 | No | – | Doppler always comes from ray geometry. |
| Delay scaling | 7.7.3, eq. 7.7-1, Tables 7.7.3-1/-2 | No | – | The caller can scale cluster delays of the CDL route. |
| Spatial filter: isotropic | 7.7.4.1, eq. 7.7-2 | Yes | `generate_antenna` type `omni` (`src/generate_antenna.cpp:160`) | |
| Spatial filter: rectangular mask | 7.7.4.1, eq. 7.7-3 | Partial | `arrayant` class | No generator; the caller fills a 0/1 pattern. |
| Spatial filter: ITU-R M.2135 pattern | 7.7.4.1, Fig. 7.7.4-2 | Partial | `generate_antenna` type `3GPP` (`src/generate_antenna.cpp:281`) | The library element is the 7.3 pattern, not the M.2135 sector pattern (from reading, not measured). |
| Spatial filter: generation procedure | 7.7.4.2, eq. 7.7-4 | Partial | `interpolate_pattern` (`src/interpolate_pattern.cpp:207`); `rotate_pattern` (`src/pattern_transform.cpp`) | Building blocks only; no function applies 7.7-4 or picks the dominant direction. |
| CDL angle translation and scaling | 7.7.5.1, eq. 7.7-5 to 7.7-6b, Table 7.7.5.1-1 | No | – | |
| TDL correlation matrix | 7.7.5.2 | No | – | |
| K-factor of CDL/TDL-D and -E | 7.7.6, eq. 7.7.6-1, 7.7.6-2 | Partial | `gbsm_single_link_channel` `kf_ratio` → `ssf_paths` (`src/ssf_paths.cpp:196`) | API: ratio K_desired/K_model. **Drift:** the library keeps the LOS + NLOS sum where the TR keeps the LOS power (same K, different total gain), and the delay renormalisation to DS = 1 is missing (from reading, not measured). |

### 3.8 Channel model calibration (7.8)

No 3GPP calibration has been reproduced with the library, and no test compares with the reference curves. The reference numbers and a proposed test set are in `md/3gpp_calibration_references.md`. The rows state what the library can supply for each setup.

#### 3.8.1 Large scale calibration (7.8.1)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Large-scale setup | 7.8.1, Tables 7.8-1, 7.8-1A | Partial | `path_loss`, `lsp_generate` (SF), `los_probability`, `o2i_loss`, `generate_antenna` | Per-link loss, SF, LOS state, O2I and the tilted vertical array exist. Hexagonal layout, wrap-around, UT drop, attachment and the DFT virtualisation are the caller's. |
| Large-scale metrics: coupling loss, geometry | Table 7.8-1 | Partial | `acdf` (`src/qd_tools_acdf.cpp:59`) | CDFs only; coupling loss and SIR are the caller's. |
| Large-scale comparison with 3GPP results | 7.8.1 | No | – | |

#### 3.8.2 Full calibration (7.8.2)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Full calibration setup | 7.8.2, Tables 7.8-2, 7.8-2A | Partial | `lsp_generate`, `ssf_clusters`, `ssf_rays`, `gbsm_single_link_channel`, `generate_antenna` | Chain and BS arrays exist (the BS configurations of Tables 7.8-2 and 7.8-2A, including the config 1 port mapping, are `generate_antenna` tuples, see 3.3); the handheld UT configurations are missing (7.3). |
| Full metric: DS and AS CDFs | Table 7.8-2 | Partial | `calc_delay_spread` (`src/qd_tools_calc_delay_spread.cpp:54`); `calc_angular_spread` (`src/qd_tools_calc_angular_spread.cpp:150`) | **Drift:** the metric asks for the TR 25.996 circular AS; the library uses another definition (see Annex A.1). |
| Full metric: PRB singular values | Table 7.8-2 | Partial | `baseband_freq_response` (`src/baseband_freq_response.cpp:68`) | H per subcarrier; PRB averaging and eigenvalues are the caller's. |
| Full comparison with 3GPP results | 7.8.2 | No | – | Open item in ssf_notes section 7. |

#### 3.8.3 Calibration of additional features (7.8.3)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Oxygen absorption | Table 7.8-3 | Partial | `ssf_paths` `oxygen` | Drift of 7.6.1 applies. |
| Large bandwidth and large array | Table 7.8-4 | Partial | `ssf_rays` | **Drift:** `NumSubPaths` = 40 gives fixed offsets with equal powers, not the 7.6.2.2 random model. |
| Spatial consistency | Table 7.8-5 | Partial | `gbsm_init_phases`, `gbsm_draw` | Fields exist; Procedure A/B does not. |
| Blockage | Table 7.8-6 | No | – | |
| Near field, SNS at BS and UT | Tables 7.8-6A to 7.8-6C | No | – | Keys stored, not used. |

#### 3.8.4 Calibration of the indoor factory scenario (7.8.4)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| InF setup | 7.8.4, Table 7.8-7 | Partial | `los_probability` defaults; conf `DS_VS`, `absTOA_Lmax` | Clutter defaults and hall V/S and L match (`tests/catch2_tests/test_conf_files_v19.cpp:131`, `test_los_probability.cpp:219`). BS lattice and UT drop are the caller's. |
| InF metrics incl. first-path delay | Table 7.8-7 | Partial | `calc_angular_spread`, `calc_delay_spread`, `lsp_generate` `abs_toa` | AS definition drift (A.1). |

### 3.9 Channel model(s) for ISAC (7.9)

The library has no sensing-specific code (no target, RCS or monostatic model). The ISAC CRs of v19.1 to v19.4 (0029, 0031, 0033, 0038, 0039) are therefore not reflected. Partial rows mean that a link of the ISAC model can be generated as a clause-7 BS–UT link by treating the target as a UT.

#### 3.9.1 Scenarios (7.9.1)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Sensing scenarios and target dropping | 7.9.1, Tables 7.9.1-1 to 7.9.1-5 | No | – | Of the base scenarios only UMi, UMa, RMa, SMa, InH and InF exist; aerial, V2X, HST and indoor-room scenarios are missing. |

#### 3.9.2 Physical object model (7.9.2)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| RCS model 1 | 7.9.2.1, eq. 7.9.2-1/-2 | No | – | |
| RCS model 2 | 7.9.2.1, eq. 7.9.2-3, Tables 7.9.2.1-2 to -7 | No | – | |
| Target cross-polarisation matrix | 7.9.2.2, eq. 7.9.2-4/-5 | No | – | Path data hold generic Jones matrices, but nothing fills them from this model. |

#### 3.9.3 Reference channel models and required updates (7.9.3)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Reference models per link type | 7.9.3, Tables 7.9.3-1 to -3 | Partial | `path_loss`, `los_probability`, `lsp_generate`, `ssf_clusters`, `ssf_rays`; scenario files | Only the cases that reduce to clause-7 BS–UT links. BS–BS, UT–UT, aerial and V2X links are missing. |
| ISAC-specific updates (heights, d_BP height) | 7.9.3 | No | – | `path_loss` uses one UT height for both the loss and the breakpoint. |
| LOS probability for aerial UEs | 7.9.3, Tables 7.9.3-4/-5 | No | – | |

#### 3.9.4 Fast fading model (7.9.4)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Step 1: layout, scattering points | 7.9.4.0 | Partial | `get_channels_spherical` (`src/get_channels_spherical.cpp:145`) | Generic positions, patterns, orientations; no target object with scattering points. |
| Steps 2–4: LOS, PL, LSPs per link | 7.9.4.1 | Partial | `los_probability`, `path_loss`, `lsp_generate` | The monostatic rule (identical LSPs for both links of a point) is the caller's. |
| Steps 5–8: clusters and rays per link | 7.9.4.1 | Partial | `ssf_clusters`, `ssf_rays` | Monostatic equalities of angles and coupling are not available. |
| Step 9: coupling across the two links | 7.9.4.1 | Partial | `get_channels_irs` (`src/get_channels_irs.cpp:137`); `combine_irs_coord` (`src/combine_irs_coord.cpp:62`) | Building block only: the IRS cascade combines all path pairs (Option 1). Option 2 and the LOS-LOS rule are missing. |
| Step 10: path power with RCS, X dB pruning | 7.9.4.1, eq. 7.9.4-1 | No | – | |
| Step 11: absolute delay of the combined path | 7.9.4.1, eq. 7.9.4-2 | Partial | `get_channels_irs` | Sum of segment delays; Δτ rules are the caller's. |
| Steps 12–13: XPR and phases at the point | 7.9.4.1, eq. 7.9.4-3 | No | – | |
| Step 14: coefficient with target CPM and Doppler | 7.9.4.1, eq. 7.9.4-4 to 7.9.4-13 | No | – | |
| Step 15: PL, SF, RCS factor, sum over points | 7.9.4.1, eq. 7.9.4-14 | No | – | |
| Bistatic background channel | 7.9.4.2 | Partial | clause-7 chain | Only where the pair maps to a BS–UT link. |
| Monostatic background (reference points) | 7.9.4.2, eq. 7.9.4-15 | No | – | |
| Combining target and background | 7.9.4.3, eq. 7.9.4-16 | No | – | Nothing ISAC-specific; the caller adds the paths. |

#### 3.9.5 Additional modelling components (7.9.5)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Spatial consistency for ISAC links | 7.9.5.1, Table 7.9.5.1-1 | No | – | |
| Type-2 environment object (specular wall) | 7.9.5.2, eq. 7.9.5-1 to 7.9.5-13 | Partial | `plane` (`src/mesh_primitives.cpp:187`); ray tracer (`ray_progress` `src/ray_progress.cpp:301`, `ray_mesh_interact` `src/ray_mesh_interact.cpp:2707`; Fresnel `:848`) | A meshed wall gives the finite-face reflection with TE/TM Fresnel coefficients. Insertion into the target channel and the power rule are missing. Material drift as in section 4. |
| Power normalisation, mobile scatterers, low-power clusters, blockage | 7.9.5.3 to 7.9.5.6 | No | – | |

#### 3.9.6 Channel model calibration (7.9.6)

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| ISAC calibration | 7.9.6 | No | – | No tests or results; the 3GPP reference (R1-2509126) is recorded in `md/3gpp_calibration_references.md`. |

## 4. Inventory: clause 8, map-based hybrid channel model

The deterministic part of clause 8 maps onto the library's ray tracer: `obj_file_read` → `ray_init` → (`ray_progress`, `ray_commit`) → `calc_diffraction_gain` for the direct path → `qrt_file_append` or `get_channels_spherical`. The stochastic part (random clusters merged with the deterministic ones) does not exist.

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Coordinate system | 8.1 | Yes | `get_channels_spherical` (`src/get_channels_spherical.cpp:145`); `qrt_file_read` (`src/qd_channel_qrt.cpp:1447`) | As clause 7. |
| Scenarios | 8.2 | n/a | – | No maps are shipped; the caller supplies the scene. |
| Antenna modelling | 8.3 | Yes | `get_channels_spherical`; `get_channels_multifreq` (`src/get_channels_multifreq.cpp:212`) | Same `arrayant` model; frequency-dependent patterns per bin. |
| Step 1: digital map import | 8.4 step 1 | Yes | `obj_file_read` (`src/obj_file.cpp:831`); `mitsuba_xml_file_write` (`src/mitsuba_xml_file_write.cpp:122`) | Wavefront OBJ with a material name per face; thickness from solid geometry or a sheet `thk` value. Random small objects are the caller's. |
| Material properties (ITU-R P.2040) | 8.4 step 1, eq. 7.6-40 to 7.6-42, Table 7.6.8-1 | Yes | `ray_mesh_interact` (`src/ray_mesh_interact.cpp:2707`; `Material::eta` `:489`); table `src/obj_file.cpp:30`; `data/materials_rf.csv` | ε = a·f^b − j·17.98·c·f^d/f, as 7.6-40 to 7.6-42; extensions (permeability, Lorentz pole, sheets). **Drift:** the built-in values come from a later P.2040 revision than Table 7.6.8-1: concrete, brick and plasterboard differ (e.g. concrete 5.24/0/0.0462/0.7822 vs 5.31/0/0.0326/0.8095); wood, floorboard, metal and grounds match. A user CSV can supply the TR values (from reading, not measured). |
| Step 2: layout, antennas, motion | 8.4 step 2 | Yes | `get_channels_spherical` | Positions per snapshot. |
| Step 2f: frequency bins | 8.4 step 2f | Partial | `ray_progress` (`src/ray_progress.cpp:301`); `get_channels_multifreq` | Geometry traced once, Jones and gain per frequency (up to 127). The bin rule K_B ≥ ⌈B·D/c⌉ is the caller's. |
| Step 3: LOS path | 8.4 step 3, Table 8.4-1 | Yes | `calc_diffraction_gain` (`src/calc_diffraction_gain.cpp:118`); `ray_commit` (`src/ray_commit.cpp:134`; LOS polarisation `:394`) | Free-space loss from the path length. |
| Step 3: reflection | 8.4 step 3 | Yes | `ray_progress` (reflection pass `src/ray_progress.cpp:1045`); `ray_mesh_interact` (Fresnel `:804`) | Shooting and bouncing ray tubes from an icosphere (`ray_init` `src/ray_init.cpp:112`) instead of images, which the TR allows. Order 0–255 (default 10). |
| Step 3: penetration | 8.4 step 3 | Yes | `ray_progress` (transmission pass `src/ray_progress.cpp:1081`); `ray_mesh_interact` | Solids with refraction or thin sheets with the internal reflection series; order configurable (default 10, TR recommends 5). |
| Step 3: diffraction | 8.4 step 3 | Partial | `calc_diffraction_gain` (`src/calc_diffraction_gain.cpp:118`); `generate_diffraction_paths` (`src/generate_diffraction_paths.cpp:66`) | **Drift:** Fresnel-ellipsoid obstruction with knife-edge weights calibrated to 2D UTD, applied to the direct path only; no explicit edge paths, no reflection + diffraction paths, so values differ from a UTD tracer (from reading, not measured). |
| Step 3: scattering on small objects | 8.4 step 3 | No | – | "Scattering is not implemented" (`src/ray_mesh_interact.cpp:3335`). |
| Step 3: outputs | 8.4 step 3 | Partial | `qrt_file_append` (`src/qd_channel_qrt.cpp:510`), `qrt_file_read` (`:1447`); `coord2path` (`src/coord2path.cpp:52`) | Interaction points, length, per-frequency gain and Jones matrix, angles, interaction count. Missing: −25 dB relative pruning (`min_gain_dB` is absolute), per-path XPR κ_RT, path IDs across positions. Delays are relative to d3D/c, not to the first arrival (from reading, not measured). |
| Step 4: LSPs for random clusters | 8.4 step 4 | No | – | |
| Steps 5–6: random-cluster delays and powers | 8.4 steps 5–6, eq. 8.4-4 to 8.4-9 | No | – | |
| Step 7: random-cluster angles | 8.4 step 7, eq. 8.4-10 to 8.4-18 | No | – | |
| Step 8: merging of clusters | 8.4 step 8 | No | – | `qrt_paths_from_scatterers` (`src/qd_channel_qrt.cpp:1836`) can write GBSM paths next to traced paths, without the 8.4 rules (ssf_notes D25). |
| Steps 9–10: rays in deterministic and random clusters | 8.4 steps 9–10, eq. 8.4-19 to 8.4-25 | No | – | `gbsm_rays_from_paths` (`src/gbsm_rays_from_paths.cpp:74`) makes one ray per traced path. |
| Step 11: XPR of deterministic clusters | 8.4 step 11, eq. 8.4-26 | No | – | Traced paths keep their Jones matrix. |
| Step 12: initial phases | 8.4 step 12 | Partial | `get_channels_spherical` | Propagation phases of the traced paths; no random clusters. |
| Step 13: coefficients | 8.4 step 13, eq. 8.4-27 to 8.4-32 | Partial | `get_channels_spherical`, `get_channels_multifreq`; or `gbsm_rays_from_paths` → `gbsm_single_link_channel` | Deterministic clusters only; spherical waves per element instead of plane waves. |
| Step 13: oxygen absorption | 8.4 step 13, eq. 8.4-33 | Yes | `ssf_paths` (`src/ssf_paths.cpp:216`) via `gbsm_single_link_channel` | Per-path length as in 8.4-33; only through the GBSM route, not `get_channels_spherical`. |
| Step 13: blockage | 8.4 step 13 | No | – | |

## 5. Inventory: Annex A, further parameter definitions

| Feature | TR ref | Covered | Function, location | Comment |
|---|---|---|---|---|
| Angular spread | A.1, eq. A-1 | Partial | `calc_angular_spread` (`src/qd_tools_calc_angular_spread.cpp:150`; helper `:14`, RMS `:88`) | **Drift:** A.1 (v19.5) is the circular standard deviation √(−2 ln\|Σ P e^{jφ}/Σ P\|). The library computes √(Σ p·wrap(φ − μ)²) around the circular mean, which is neither A-1 nor the TR 25.996 minimum over shifts. All agree for small spreads; two equal paths at 0° and 120° give 60° vs 67.5° with A-1. The API doc and the comment at `:88` call this "the 3GPP TR 38.901 definition" (from reading, not measured). |
| Mean angle | A.2, eq. A-2 | Partial | `calc_angular_spread` (internal mean `:17`; `orientation` output) | A-2 is computed internally but not output; with wrapping on, `orientation` is a 3D mean direction, equal to A-2 only for equal elevations (from reading, not measured). |
| Cluster angular spread incl. LOS | A.3, eq. A-3 | Partial | `calc_angular_spread` | The caller passes cluster angles plus the LOS path; A.1 drift applies. |
| Cluster mean angle incl. LOS | A.4, eq. A-4 | Partial | as A.2 | |
| CDL AS scaling factor | A.5, eq. A-5, A-6 | No | – | |
| Handheld UT antenna positions and orientations | A.6, Tables A.6-1/-2 | Yes | `generate_antenna` type `3GPP-UT` | Positions tested within 1e-12 m, bearings within 0.05° of the tables (computed from the geometry for every device size). |

## 6. Proposed runs to confirm drifts and check open points

Nothing in this list has been run. Each item names the feature, the run and the expected outcome where known.

Large-scale model and path loss:

- **RMa ZOD offset (3.5):** `lsp_generate` with `RMa_NLOS` at 3.5 GHz, d2D = 100 m, hUT = 1.5, 3, 5, 10 m, and hBS = 35 and 25 m. TR: EoD offset +1.03° for all cases. Expected library: 1.03° only at 35/1.5 m, 0 for hUT ≥ 3.5 m, about 1.09° at hBS = 25 m.
- **InF LOS V/S default (3.5):** `lsp_generate` with `InF_LOS`, `rnd = 0`. TR median DS 52.7 ns (V/S 4) or 59.0 ns (V/S 4.55); expected default 55.8 ns.
- **SF std fallback (3.5):** RMa and SMa LOS with and without `sf_sigma_dB` from `path_loss`, below and above d_BP. Expected 5 dB without, 4/6 dB with.
- **UMa ZOD offset (3.5):** `lsp_generate` at 28 GHz, d2D = 100 m, hUT = 1.5 m: TR μ_offset,ZOD ≈ −2.46°, so EoD +2.46°; equal values at 3.5 and 6 GHz (floor). Probably covered by `test_conf_files_v19.cpp:242`; confirm only.
- **RMa/SMa breakpoint (3.4.1):** `path_loss` at 0.5 GHz, hUT = 1 m, d2D = 2·d_BP against the TR PL2 with the 2D d_BP. Expected ≈ 0.034 dB (SMa ≈ 0.031 dB).
- **UMa h_E (3.4.1):** 1e5 links at hUT = 22.5 m with independent `rnd_he`; compare the h_E histogram with p(1 m) = 1/(1 + C) ≈ 0.66 and a uniform split over {12, 15, 18, 21}. Then measure the h_E correlation between UTs 5 m apart under the spatially consistent draw.
- **SF autocorrelation (3.4.4):** `gbsm_draw("lsf")` column 2 on 2000 points at 1 m spacing; estimate R(Δx) for `Exp256` without bias removal and for the default, against exp(−Δx/37) (UMa LOS).
- **Floor residual (3.6.3):** "cluster" fields on two floors at horizontal lags 0 to 3·`SC_lambda`; expected 0 at lag 0, up to about 0.16 elsewhere.
- **SMa O2I mix (3.4.3):** `o2i_loss` with two calls (10 m and 25 m) for the 90/10 split, compared with one call at 25 m.

Antennas and coordinates:

- **LCS/GCS (3.1):** `interpolate_pattern` with (bank, tilt, heading) = (20°, −15°, 40°) at random angles; compare local angles and `gamma` with eq. 7.1-7, 7.1-8 and 7.1-15; repeat with downtilt only against 7.1-24 to 7.1-26.

Fast fading and additional components:

- **C_φ/C_θ cluster count (3.5):** `38.901`, UMa and UMi NLOS at 3.5 GHz, 2000 links; output ASA/ZSA with C from `n_cluster` against C from the count after −25 dB removal.
- **Zenith fold on ZOD (3.5):** count the NLOS rays whose zenith before folding is outside [0°, 180°] and compare the link ZSD with a fold that also flips the azimuth.
- **UMa c_DS floor (3.5):** `ssf_rays` UMa at 3.5 GHz with `PerClusterDS_fc_min` 6 and 0; sub-cluster offsets 5.0/10.0 ns vs 6.0/12.1 ns; report the ray DS change.
- **Per-element delays in taps (3.5):** UMi NLOS link, 8×8 dual-polarised array, ±50 and ±200 MHz; frequency response with per-element tap delays against one delay per tap; report the NMSE.
- **CR 0043 at cluster level (3.5):** UMa NLOS with and without `eod_offset_deg`; the mean cluster EoD should move by exactly the offset, UMa LOS not at all.
- **Oxygen τ_Δ (3.6.1):** NLOS drops at 60 GHz without absolute TOA against an oracle with τ_Δ = min τ′_n; expected p99 0.71 dB (UMi) and 1.28 dB (UMa).
- **Oxygen in-band (3.6.1):** L = 200 m, B = 2 GHz at 63 GHz; TR 2.86/2.10/1.36 dB at f_c − B/2, f_c, f_c + B/2.
- **Per-element delay 7.6.2.1 (3.6.2):** 64-element ULA with B > c/D; compare delays with τ − r̂·d/c and the band-edge phases with 7.6-3.
- **Ground permittivity (3.6.8):** GR slot power at 3.5 GHz in UMi LOS with the `lsp_generate` values and with Table 7.6.8-1 medium dry and wet ground.
- **Signs and offsets kept (3.6.3):** angle change per metre on a UT track with only the signs held (TR) versus signs and offsets held (library).
- **7.6.5.1 LOS jumps (3.6.5):** how often an angle crosses ±180° in step 5′ and jumps after the √(1+K) scaling at two frequencies with different K.
- **Cluster count range (3.6.15):** `n_cluster` at the range ends (UMa NLOS 15 and 20); output DS/AS ratios within the 7.5 tolerance.

Link level, calibration, Annex A:

- **Angular spread definition (5):** `calc_angular_spread` on two equal paths at 0° and 120° and on three equal paths at 0°, 170°, −170°, against A-1 and TR 25.996 (hand values: 60° / 67.5° / 60°; ≈104° / ≈86° / ≈85°). Then the p5/p50/p95 ASA/ASD of 1000 UMa/UMi NLOS links with all three definitions, against the 7.8.2 references.
- **Mean angle (5):** with wrapping on, compare the `orientation` heading with A-2 for paths at elevations 0° and 60°.
- **CDL route (3.7):** CDL-A through `ssf_rays` → `gbsm_single_link_channel` (`rays`); check the normalised DS and the effect of the forced sub-cluster split.
- **CDL K-factor (3.7):** CDL-D with `kf_ratio`; compare ray powers with eq. 7.7.6-1 (same K, total power kept instead of LOS power, delays not renormalised).
- **InF calibration (3.8.4):** InF-SL/DL/SH/DH with the Table 7.8-7 layout at 3.5 GHz; DS/AS p10/p50/p90 against R1-1909704, with the library AS and with A-1.
- **Full calibration (3.8.2):** the test set of `md/3gpp_calibration_references.md` section 6, using the TR 25.996 AS for metric 3.

Ray tracer and clause 8:

- **Materials (4):** reflection coefficient at 28 GHz, 45°, on `itu_concrete` and on a CSV row with the Table 7.6.8-1 concrete values; compare with eq. 7.6-38/39.
- **Specular wall (3.9.5):** one large `plane` as a type-2 object; compare delay, angles and Jones matrix of the traced path with eq. 7.9.5-2 to 7.9.5-13.
- **Diffraction (4):** `calc_diffraction_gain` for a knife edge, a wedge and a building corner at 3.5 and 28 GHz against UTD and ITU-R P.526; deviation in dB over shadow depth.
- **Step 3 delays (4):** NLOS-only scene with `use_absolute_delays = false`; the first delay should be τ′_min − d3D/c, not 0 as in the TR.
- **Penetration (4):** 0.2 m concrete slab at normal incidence, traced solid and `thk = 0.2`, against a closed-form slab with the TR concrete values.

## 7. Notes

- **CRs of v19.1 to v19.5:** Annex B gives only the subject of each CR, and the CR texts are not part of this review, so the mapping of a CR to individual table entries cannot be shown. The code and the scenario files match the v19.5.0 text in all reviewed clauses, which includes the effect of the 7–24 GHz corrections (CR 0028, 0036, 0040, 0042) and of CR 0043 (ZOD generation, tested at `tests/catch2_tests/test_lsp_generate.cpp:530`). CR 0032 (handheld UT polarised model) and the ISAC CRs (0029, 0031, 0033, 0038, 0039) are not implemented. CR 0035 (UL/DL reciprocity) is covered as described in 3.5.
- **Scenario keys stored but unused:** `NF_*`, `SNS_*` and `LOS_scatter_radius` are validated but read by no function of this version (`src/qd_scenpar.cpp:1030`); `LOS_state_lambda` and `IO_state_lambda` are read by the caller.
- **TR text oddities found in passing (no code impact):** TDL-E tap 14 has delay 20.6519 where CDL-E cluster 14 has 20.6419; Table 7.7.5.1-1 has a stray value in the CDL-E ASD-25 row; Annex A.1 (circular standard deviation) and 7.8.2 metric 3 (TR 25.996 definition) name two different angular-spread definitions.
- **Uncertain readings:** whether O2I links count as NLOS for 7.6.9; which cluster count enters C_φ and C_θ after the −25 dB removal.
