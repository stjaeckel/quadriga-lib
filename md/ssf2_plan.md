# SSF series 2: path generator (plan)

Plan for the second small-scale fading series: from the clusters of `ssf_clusters` to propagation paths that feed `get_channels_planar`, `get_channels_spherical` and the QRT writer, for TR 38.901 v19.5.0 clause 7.5 steps 7 (ray offsets) to 11 (antenna-free part) and the QuaDRiGa method. Working document; deleted before the merge, its outcome goes to `md/ssf_notes.md`. Builds on series 1 (`md/ssf_notes.md`: decisions D1–D8, W1–W22).

Status: plan v1 (step 1 of 7), decisions taken; open items for the plan review in section 9.

## 1. Goal and requirements

1. Consume the output of `ssf_clusters` (slots LOS, GR, N clusters; delays, powers per frequency, angles) for all three methods (`38.901`, `38.901_MF`, `quadriga`).
2. Generate everything `get_channels_planar` (angles) and `get_channels_spherical` (FBS/LBS) need: path gain, path length, polarisation transfer matrix M, angles or scatterer positions.
3. Implement the TR ray step stand-alone in quadriga-lib (no QuaDRiGa-NG needed), TR-literal for the TR methods.
4. Serve QuaDRiGa-NG so that its own path code can be retired later (integration is a separate project).
5. Feed the C++ QRT writer (`qrt_file_append`) with antenna-free paths, so that antennas and orientations can be added later at low cost.
6. Support mobility: the TR phase rotation and QuaDRiGa drifting, both driven by a real offset of Tx/Rx from the drop position. Tracks and segments stay in QuaDRiGa-NG.

Side quests (no implementation, the design must not block them): multi-frequency paths for `get_channels_multifreq` (section 7.1), TR Procedure A (7.2), TR 7.6.2.2 and 7.6.10 (7.3), a library-only TR 7.8.2 calibration (7.4).

## 2. Findings from the analysis

1. **The TR baseline is angle-based.** TR 7.5-22 uses plane waves, no positions, and models motion as a rotating phase per ray with delays and angles fixed. The only TR clause with positions is 7.6.13 (near field), whose distances do not match the path length. A TR drop can be turned into scatterer positions with the QuaDRiGa two-bounce solver (exact drop angles where a solution exists); its mobility is then drift, not the TR rotation. So a TR drop can be stored in QRT as geometry, not TR-literally.
2. **QRT stores interaction coordinates and one Jones matrix per frequency** (≤ 127 frequencies, shared coordinates, float32). Path length and gain are recomputed on read (gain = FSPL(length, f) · |M|²), so GBSM path power must be folded into M. Origin = static Tx; CIR = Rx snapshot; destination = group of CIRs. Dual mobility cannot be stored. float32 gives about 1 mm resolution 10 km from the origin; local coordinates are recommended.
3. **NG drifting lives inside `get_channels_spherical`** (`update_drifting.m` has no caller). NG keeps FBS/LBS from the drop, moves Tx/Rx, recomputes LOS and the GR point (ground z = 0, GR polarisation R(t)/R0) and passes `path_length = d(t) + τ·c`; the library uses max(path_length, polyline), so a path cannot become shorter than its drop excess delay. To be measured in R0.
4. **NG's TR baseline mobility** is one `get_channels_planar` call at the drop and a phase rotation per sub-path afterwards (straight track along the Rx heading). Its sub-cluster offsets are rescaled tables, not the TR ray mapping of Table 7.5-5.
5. **NG sums sub-paths per cluster and normalises each cluster's segment-average power with the antennas** (`p_correct`). Not portable to an antenna-free model; stays in NG.
6. **NG multi-frequency QuaDRiGa geometry is per frequency** (coupling and sub-cluster delays per frequency); a multi-frequency QRT file and `get_channels_multifreq` need one geometry.
7. **Sub-path methods in use:** `legacy` in all library conf files and 85 of 91 NG files; `mmMAGIC` in 6 deprecated NG files; `Laplacian` in none. `NumSubPaths` = 20 except one NG file with 10 and one with 26.
8. **Ray-level random fields are large** (TR: about 1900 SOS generators per frequency for N = 19, M = 20); cost to be measured in R0. Coupling is kept per drop (TR 7.6.3.1) and needs a seeded stream, not an SOS field.

## 3. Decisions

| ID | Decision |
|---|---|
| P1 | TR scope: steps 8–11 (antenna-free part) + 7.6.16 polarisation power variability + 7.6.1 oxygen absorption. 7.6.2.2, 7.6.10 not implemented; no-blocker checks (7.3). 7.6.4, 7.6.13, 7.6.14 out of scope. |
| P2 | Scatterer geometry is available for every cluster method. Mobility is a separate switch: `phase` (TR rotation) or `drift` (fixed scatterers). TR methods default to `phase`, `quadriga` to `drift`. A TR drop with `drift` is a documented non-TR combination; it is the way to store a TR drop in QRT. |
| P3 | Ray angle construction: both TR additive offsets (with zenith fold) and the 3-D rotation of the offset pattern around the cluster direction, with a switch. Defaults: additive for `38.901`/`38.901_MF`, rotation for `quadriga`. Impact measured in R0; the calibration step checks it again. |
| P4 | Sub-clusters: TR methods split the two strongest clusters per link (per frequency in `38.901_MF`) with the ray mapping of Table 7.5-5 and c_DS per frequency. `quadriga` splits all NLOS clusters always, same ray mapping, one c_DS for all frequencies (frequency average, as W5 of series 1). |
| P5 | Ray randomness across frequencies: `quadriga` shares coupling, sub-cluster delays, positions and initial phases across frequencies; XPR uses one normalised value per ray with frequency-dependent μ/σ. TR methods draw steps 8–11 per frequency (TR 7.6.5). |
| P6 | Ray XPR, phases and η come from spatially consistent SOS fields in all methods (TR 7.6.3.1), in new append-only generator sets; coupling per drop from a seeded stream (SplitMix64 of the LSF seed and a tag, as W8). Run time measured in R0; fallback: seeded stream when `SC_lambda = 0`, same layout. |
| P7 | `phase` mobility: TR-literal phase exp(j·2π·(r̂_rx·Δr_rx + r̂_tx·Δr_tx)/λ) per ray folded into M per frequency; delays and angles fixed; `get_channels_planar` called with the arrays at the drop position (orientations may change per offset, TR 7.6.7). Valid for short moves (documented). Procedure A later (7.2). |
| P8 | Large-scale changes along a track (PL + SF per frequency, K) are optional per-offset inputs of the offset assembly; the caller evaluates them with the existing LSF functions at the offset positions; omitted = drop values. The assembly applies the gain to all paths and rescales LOS/GR against NLOS to follow K, as NG does. |
| P9 | API: building blocks only (random sets, rays, scatterers, offset assembly, QRT conversion); the caller runs `get_channels_*`. A one-call coefficient function can wrap the blocks later. |
| P10 | Rays are separate paths; every ray carries its slot (cluster) index so that a caller can sum rays per cluster. NG's antenna-dependent `p_correct` is not ported; it stays in NG until the track logic is ported. Channels read from a QRT file do not contain it unless the caller applies it after reading. |
| P11 | `SubpathMethod`: `legacy` only; any `NumSubPaths` via the rescale rule of NG (TR Table 7.5-3 at 20); `Laplacian` and `mmMAGIC` raise an error. |
| P12 | QRT: generic C++ conversion of spherical path data (FBS, LBS, gain and M per frequency) into `quadriga_lib::path` objects for `qrt_file_append`. No MEX/Python QRT wrappers in this series. Segment overlaps are out of scope (a path-level merge is the bridge for the track project). |
| P13 | Scatterer solver: redesign with the intent of NG's `gen_fbs_lbs` (exact angles at both ends where a two-bounce solution exists), with a documented rule for the free distance, a minimum distance to both arrays (open item of series 1), a defined fallback and a direct solve; compared with NG in R0. No faithful port, no NG parity test. |
| P14 | Validation: Catch2 TR oracles and an R0 study in this series. The TR 7.8.2 calibration follows in a later library-only session; nothing in this series may require QuaDRiGa-NG for it (7.4). |
| P15 | `drift` path length = exact Tx-FBS-LBS-Rx polyline at the offset (pure geometry); NG's `d(t) + τ·c` with the max rule is not reproduced; NG's behaviour is measured once in R0 and recorded as a finding. |
| P16 | Polarisation model: both the TR model (κ per ray, four phases, 7.5-21/22) and the QuaDRiGa rotation model (linear XPR γ per ray, circular XPR κ per cluster, one phase per ray), with a switch. Defaults by method. Impact measured in R0; the calibration step checks it again. |

Choices taken in this plan without a separate question (open to the review): ground plane at z = 0 (as `ssf_clusters`); LOS ray in `drift` with FBS = LBS on the direct line; LOS Jones diag(1, −1), GR Jones diag(R∥, −R⊥)/R (W7, W22); padded batch layout with a fixed ray count per slot, zero-power rays dropped only by the offset assembly (W12); QuaDRiGa XPR mean = the link's XPR LSP and σ = 0.1·σ_XPR (NG).

## 4. Inventory

Legend: **new** = new code, **ext** = extension of an existing function, **NG** = replaces NG code.

### 4.1 Ray level

| # | Feature | Kind | Methods |
|---|---|---|---|
| A1 | Ray offsets α_m (Table 7.5-3) with c_ASA, c_ASD, c_ZSA and ZSD = 3/8·10^μlgZSD (`PerClusterES_D_scale`); any M via the rescale rule | new, NG | all |
| A2 | Additive construction with zenith fold (7.5-13/18/20) and 3-D rotation construction (P3) | new, NG | switch |
| A3 | Random coupling AOD↔AOA, ZOD↔ZOA, AOD↔ZOD (step 8), per drop | new, NG | all |
| A4 | Sub-clusters (7.5-26, Table 7.5-5, c_DS default 3.91 ns, `PerClusterDS_gamma`, `PerClusterDS_min`, `PerClusterDS_fc_min`) | new, NG | P4 |
| A5 | Polarisation: TR κ + 4 phases; QuaDRiGa γ/κ + 1 phase (P16); LOS and GR Jones | new, NG | switch |
| A6 | 7.6.16 polarisation power variability (η, normalisation 7.6-62) | new | TR model; QuaDRiGa model: see 9 |
| A7 | 7.6.1 oxygen absorption (per path and frequency, τ_Δ rule of 7.6.1) | new | all |
| A8 | Ray powers P_n/M per frequency; path lengths d3D + τ·c (Δτ already in τ, W13) | new | all |
| A9 | Link gain (PL + SF + O2I loss, per frequency) folded into the path gain | new | all |

### 4.2 Geometry

| # | Feature | Kind |
|---|---|---|
| B1 | Two-bounce solver from ray angles and length; documented free-distance rule; fallback | new, NG |
| B2 | Minimum distance of scatterers to both arrays | new |
| B3 | LOS on the direct line; GR point on z = 0 | new, NG |
| B4 | One geometry for all frequencies in `quadriga` (P5); per frequency in TR multi-frequency | new |

### 4.3 Mobility and offset assembly

| # | Feature | Kind |
|---|---|---|
| C1 | `phase`: per-ray phase rotation for Rx and Tx offsets, per frequency (7.5-22, 7.6-6, 7.6-45) | new |
| C2 | `drift`: fixed FBS/LBS, moved ends, polyline length (P15), LOS and GR recomputed, GR Jones R(t)/R0 | new, NG |
| C3 | Optional per-offset link gain and K (P8) | new, NG |
| C4 | Inactive rays dropped; slot index per output path (P10) | new |

### 4.4 Random fields

| # | Feature | Kind |
|---|---|---|
| F1 | New generator sets for ray variables, appended without changing the series-1 sets (stored seeds keep reproducing series-1 clusters) | ext (`ssf_init_phases`, `ssf_draw`) |
| F2 | Coupling permutations from a seeded stream per site and scenario | new |
| F3 | Reserved blocks: Procedure A sign X_n (shared), 7.6.2.2 offsets and ray delays, 7.6.10 α and D | layout only |

### 4.5 QRT and wrappers

| # | Feature | Kind |
|---|---|---|
| E1 | Spherical path data → `quadriga_lib::path` (0/1/2 interactions, gain folded into M per frequency, interaction type code for scattering) | new |
| E2 | MEX and Python wrappers for the new GBSM functions (not for QRT, P12) | new |

### 4.6 NG code that the library replaces (retired in the NG integration project)

`gen_fbs_lbs`, `private/solve_multi_bounce_opti`, `private/solve_cos_theorem`, `private/rotate_sphere`, `private/get_subpath_angles`, `private/update_drifting` (already unused), the sub-cluster split, XPR and phase parts of `gen_ssf_parameters`, `xpr_sos` / `pin_sos` / `clst_dl_sos` / `subpath_coupling` in `init_sos`, the path assembly and GR update inside `qd_builder.get_channels`. NG keeps tracks, segments, merge, layout, builder bookkeeping, sub-path summation and `p_correct`.

## 5. Draft contracts

All new functions in `include/quadriga_gbsm.hpp`, section "Small-scale fading" (after `ssf_clusters`), except E1 in `include/quadriga_channel.hpp`. Names, argument order and layouts are drafts for the review; the spec fixes them. Common notation: N = `NumClusters` − 1, M = `NumSubPaths`, G = 1 if `GR_enabled` else 0, n_ray = 1 + G + N·M (fixed per scenario; ray r belongs to slot `slot(r)`), n_fd = number of geometry sets (1, or n_freq for TR multi-frequency).

### 5.1 Random sets (extension, `src/qd_gbsm_phases.cpp`)

`ssf_init_phases` and `ssf_draw` get new set types for the ray variables, selected by an appended argument (existing calls unchanged): a shared ray set (frequency-independent ray variables: QuaDRiGa phases and normalised XPR values, X_n reserved) and a per-frequency ray set (TR κ, four phases, η; QuaDRiGa η). Seeds by the W8 rule with new tags. Column layout in blocks by variable type, append-only, documented in `ssf_draw`. `ssf_clusters` is not changed.

### 5.2 `ssf_rays` (new, `src/ssf_rays.cpp`)

Ray parameters of a set of links at the drop (TR steps 7 offsets, 8, 9, 10; 7.6.1; 7.6.16).

```
void ssf_rays(const std::unordered_map<std::string, double> &scenpar,
              const arma::mat &tx_pos,            // [1, 3] or [n_link, 3]
              const arma::mat &rx_pos,            // [n_link, 3]
              const arma::vec &fc,                // carrier frequencies in [Hz], [n_freq]
              const arma::cube &lsp,              // from lsp_generate, [n_link, n_freq, 8] (XPR)
              const arma::cube &lsp_mu,           // from lsp_generate, [n_link, n_freq, 8] (μ_lgZSD, XPR μ)
              const arma::cube &lsp_sigma,        // from lsp_generate, [n_link, n_freq, 8] (XPR σ)
              const arma::cube &cl_delay, const arma::cube &cl_power,                 // from ssf_clusters
              const arma::cube &cl_aod, const arma::cube &cl_eod,
              const arma::cube &cl_aoa, const arma::cube &cl_eoa,
              const arma::mat &rnd_ray,           // shared ray set from ssf_draw
              const arma::cube &rnd_ray_freq,     // per-frequency ray sets from ssf_draw
              arma::sword seed,                   // LSF seed (coupling stream)
              arma::cube &delay,                  // Output: ray delays in [s] rel. to the LOS delay, [n_ray, n_link, n_fd]
              arma::cube &power,                  // Output: ray powers (linear, incl. oxygen), [n_ray, n_link, n_freq]
              arma::cube &aod, arma::cube &eod,   // Output: ray angles in [rad], [n_ray, n_link, n_fd]
              arma::cube &aoa, arma::cube &eoa,
              std::vector<arma::cube> &jones,     // Output: M per frequency, n_freq × [8, n_ray, n_link]
              arma::uvec &slot,                   // Output: slot index per ray, [n_ray]
              const std::string &method = "38.901",          // cluster method that produced the clusters
              const std::string &angle_model = "",           // "additive", "rotation"; "" = default of the method
              const std::string &pol_model = "",             // "38.901", "quadriga"; "" = default of the method
              bool oxygen = false, bool pol_variability = false,
              const arma::mat *gr_eps_re = nullptr, const arma::mat *gr_eps_im = nullptr); // GR Jones
```

Properties: a pure function of the cluster parameters and the random inputs (Procedure A readiness, 7.2); the ray layout does not depend on the link (padding, W12); `quadriga` has n_fd = 1 and Jones per frequency with shared phases (P5).

### 5.3 `ssf_scatterers` (new, `src/ssf_scatterers.cpp`)

FBS/LBS positions of rays at the drop (P2, P13).

```
void ssf_scatterers(const arma::mat &tx_pos, const arma::mat &rx_pos,
                    const arma::cube &delay, const arma::cube &aod, const arma::cube &eod,
                    const arma::cube &aoa, const arma::cube &eoa,      // from ssf_rays
                    const arma::uvec &slot, bool gr,                   // LOS / GR identification
                    arma::field<arma::cube> or 2 × arma::cube &fbs_pos, &lbs_pos,   // [3, n_ray, n_link] per geometry set
                    arma::Cube<unsigned char> *valid = nullptr);      // Optional: two-bounce solution found
```

### 5.4 `ssf_paths` (new, `src/ssf_paths.cpp`)

Inputs of `get_channels_planar` / `get_channels_spherical` for one link at a list of offsets (P7, P8, P10, P15).

```
void ssf_paths(const arma::vec &tx_pos0, const arma::vec &rx_pos0,      // drop positions, [3]
               const arma::mat &tx_offset, const arma::mat &rx_offset,  // [n_off, 3] or [1, 3] / empty
               const arma::vec &fc,                                     // [n_freq]
               ray data of one link (delay, power, angles, jones, slot; fbs/lbs for drift),
               const std::string &mobility,                             // "phase" or "drift"
               std::vector<...> per offset: aod/eod/aoa/eoa or fbs/lbs, path_length [n_path],
                                 path_gain [n_path, n_freq], M [8, n_path, n_freq], slot [n_path],
               const arma::mat *gain_dB = nullptr,                      // optional PL+SF per offset, [n_off, n_freq]
               const arma::mat *kf = nullptr,                           // optional K (linear) per offset, [n_off, n_freq]
               const arma::mat *gr_eps_re = nullptr, const arma::mat *gr_eps_im = nullptr);
```

The output per offset is exactly the argument set of one `get_channels_*` call (one frequency: column `f` of gain, slice `f` of M).

### 5.5 `qrt_paths_from_scatterers` (new, generic, `src/qd_channel_qrt.cpp`)

```
std::vector<quadriga_lib::path> qrt_paths_from_scatterers(
    const arma::vec &tx_pos, const arma::vec &rx_pos,     // CIR end points
    const arma::mat &fbs_pos, const arma::mat &lbs_pos,   // [3, n_path]
    const arma::mat &path_gain, const arma::cube &M,      // [n_path, n_freq], [8, n_path, n_freq]
    const arma::vec &freq, unsigned i_cir);               // [n_freq] in Hz; CIR index (path::iC)
```

Folds the gain into M with FSPL of the polyline length; LOS → 0, FBS = LBS → 1, else 2 interactions. Round trip with `qrt_file_read` is the main test.

### 5.6 Wrappers

MEX and Python for 5.1 (extension), 5.2, 5.3, 5.4. Not for 5.5 (P12).

## 6. Data flow

Drop: `lsf_*` / `lsp_generate` / `path_loss` → `ssf_init_phases` + `ssf_draw` (cluster and ray sets) → `ssf_clusters` → `ssf_rays` → (`drift`:) `ssf_scatterers`. Per link and offset: `ssf_paths` → `get_channels_planar` (`phase`) or `get_channels_spherical` (`drift`) with the antennas. QRT: `ssf_paths` (`drift`) at the CIR positions → `qrt_paths_from_scatterers` → `qrt_file_append`.

## 7. Readiness checks (side quests)

Each check is a short paragraph in the spec that shows the contract allows the later addition without changing existing outputs or random-field columns; the spec review verifies it.

### 7.1 Multi-frequency paths (`get_channels_multifreq`)

Needed: one geometry, gain `[n_path, n_freq_in]` and M `[8, n_path, n_freq_in]` at sample frequencies; the function interpolates gain linearly and M by SLERP. Available after this series in `quadriga` (P5): shared coupling, sub-cluster delays, positions and phases; per-frequency powers, K, SF, PL and XPR from the same normalised random values. Gap for a later session: a "sample grid" mode (many closely spaced frequencies, cost of per-frequency evaluation), checks that powers and XPR vary smoothly between samples, and a validation of the frequency correlation. TR methods do not qualify (7.6.5 per-frequency steps; `38.901_MF` geometry per frequency). Feasible; the main limit is physical (one scatterer set over a very wide band), which `quadriga` already assumes.

### 7.2 Procedure A (TR 7.6.3.2)

Needs: absolute cluster delays and angles at the segment start (available: `ssf_clusters` + d3D), velocities per step (caller), the sign X_n per cluster (reserved block, F3), re-derivation of rays from updated clusters (`ssf_rays` is a pure function of the cluster parameters), the 7.6-10a delay normalisation over the whole mobility range (a later function over the step sequence). No contract change expected.

### 7.3 7.6.2.2 (large bandwidth) and 7.6.10 (mobile scatterers)

7.6.2.2: per-link M from 7.6-8, random offsets and ray delays, unequal ray powers; fits the padded layout with M_max = `NumSubPaths` upper bound and zero-power padding; random blocks reserved (F3). 7.6.10: adds a Doppler 2·α·D/λ per ray that needs time; `ssf_paths` would get an optional time vector per offset; α and D blocks reserved.

### 7.4 Library-only TR 7.8.2 calibration

Needs: many sites and sectors in one batch (per-link `tx_pos` is supported), rays shared by co-sited sectors (TR: steps 1–9 identical; the rays are antenna-free, sectors differ only in the `get_channels_*` call), per-site seeds (series 1), LOS and NLOS links of a layout handled by separate calls per scenario (W1: LOS is a scenario property) and joined by the caller, TR antenna models (`arrayant_generate`). Nothing in this series may require QuaDRiGa-NG; the R0 notes for P3 and P16 go to the calibration session.

## 8. Process (steps 2–7)

A manager session tracks plan, spec and code and starts agents; Stephan talks only to the manager. Working folders: repository `/home/stephan/quadriga-lib-ssf` (branch `ssf`), internals (briefs, reports, R0 data, review reports) in `/home/stephan/quadriga-lib-agents/ssf2/`. Coding rules: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`; produced code cites only published material and the SPEC files.

1. **Plan review (step 2):** Stephan + independent reviewers. Lesson of series 1: prototype runs belong in the first plan review. Reviewers check the TR transcription numerically and test the contracts against NG's use.
2. **R0 study (between plan review and spec):** prototype (NumPy or C++ scratch, not kept). Measures: P3 and P16 impact (spreads, XPR, rank with simple antennas); P5 frequency correlation of the coefficients vs. frequency spacing; P6 run time of the ray fields vs. the cluster step; P10 spread of per-segment cluster powers without `p_correct`; P13 solver vs. NG (exact-solution fraction, scatterer distance distribution, drift rates in °/m, Doppler spread, run time); P15 NG's length rule (Doppler asymmetry); drift continuity; TR oracle for steps 8–11. Results go into the plan before the spec.
3. **Spec (step 3), spec review (step 4):** one mandatory round, up to three if major issues are found; each reviewer checks the TR transcription numerically. Doc rules in the spec: API only, each fact once, no formulas, size targets set against `ssf_clusters`; doc blocks follow the block grammar and writing pitfalls of `docs_builder/README.md`, and each run regenerates `docs/` with `make docs` and commits the changed files with the function (`md/SPEC_*.md`).
4. **Implementation (step 5)**, test-first, in runs: R1 random sets (5.1); R2 `ssf_rays` TR methods; R3 `ssf_rays` QuaDRiGa models, 7.6.1, 7.6.16; R4 `ssf_scatterers`; R5 `ssf_paths`; R6 `qrt_paths_from_scatterers`; R7 wrappers and documentation. R4 and R6 touch separate files and may run in parallel with R2/R3. Shared test headers and method-generic test cases are specified up front (lesson: test sizes exceeded 1.5–2×).
5. **Code review (step 6):** up to three rounds, one mandatory, plus a security review (inputs, sizes, file output of 5.5).
6. **Cleanup and merge (step 7):** working documents deleted, `md/ssf_notes.md` updated, Stephan commits.
7. **NG integration (step 8):** separate project; retires the code of 4.6.

## 9. Open items for the plan review

- Layout of the ray outputs: `[n_ray, n_link, n_f]` (contiguous per link, direct slicing for `ssf_paths`) vs. `[n_link, n_ray, n_f]` (as `ssf_clusters`).
- Container for per-geometry-set positions (`arma::field<arma::cube>` vs. a 4-D layout in a cube).
- Selector for the new random sets in `ssf_init_phases` / `ssf_draw` (appended argument vs. new set ids).
- 7.6.16 with the QuaDRiGa polarisation model: apply η to the rotation matrix with the 7.6-62 normalisation, or TR model only.
- 7.6.1 τ_Δ with W13 (Δτ already in the NLOS delays).
- Free-distance rule of the solver (R0 sub-study).
- Function names (`ssf_rays`, `ssf_scatterers`, `ssf_paths`, `qrt_paths_from_scatterers`).
