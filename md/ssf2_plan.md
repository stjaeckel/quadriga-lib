# SSF series 2: path generator (plan)

Plan for the second small-scale fading series: from the clusters of `ssf_clusters` to propagation paths that feed `get_channels_planar`, `get_channels_spherical` and the QRT writer, for TR 38.901 v19.5.0 clause 7.5 steps 7 (ray offsets) to 11 (antenna-free part) and the QuaDRiGa method. Working document; deleted before the merge, its outcome goes to `md/ssf_notes.md`. Builds on series 1 (`md/ssf_notes.md`: decisions D1–D8, W1–W22).

Status: plan v4 (step 2 of 7). v3 folded in the first plan review (two independent reviews with prototype measurements): the antenna-free power correction withdrawn (P18), coupling from ranked random fields (P6), the ray angle and polarisation defaults follow the TR in every method (P3, P16), ray-level reciprocity (P19), layout-agnostic per-link functions (P20), a fallback for rays without a scatterer solution (P13). v4 adds the R0 study (a literature search and measurements on library clusters against a point-scatterer reference; findings 5 and 8–16): per-ray delays from their own random field in `quadriga` (P21), the solver rule and d_min (P13), the reciprocal rotation construction (P19), and confirms P3, P15, P16, P17, P18. Next: contract review, whose outcome (the full contracts of section 5) becomes plan v5 and gets a further review round before the spec.

## 1. Goal and requirements

1. Consume the output of `ssf_clusters` (slots LOS, GR, N clusters; delays, powers per frequency, angles) for all three methods (`38.901`, `38.901_MF`, `quadriga`).
2. Generate everything `get_channels_planar` (angles) and `get_channels_spherical` (FBS/LBS) need: path gain, path length, polarisation transfer matrix M, angles or scatterer positions.
3. Implement the TR ray step stand-alone in quadriga-lib (no QuaDRiGa-NG needed), TR-literal for the TR methods; the ray angle and polarisation defaults follow the TR in every method.
4. Serve QuaDRiGa-NG so that its own path code can be retired later (integration is a separate project); NG selects its non-TR variants by switches.
5. Feed the C++ QRT writer (`qrt_file_append`) with antenna-free paths, so that antennas and orientations can be added later at low cost.
6. Support mobility: the TR phase rotation and QuaDRiGa drifting, both driven by a real offset of Tx/Rx from the drop position. Tracks and segments stay in QuaDRiGa-NG.

Side quests (no implementation, the design must not block them): multi-frequency paths for `get_channels_multifreq` (section 7.1), TR Procedure A (7.2), TR 7.6.2.2 and the mobile-scatterer term of 7.6.10 (7.3), a library-only TR 7.8.2 calibration (7.4).

## 2. Findings from the analysis, the plan review and R0

1. **The TR baseline is angle-based.** TR 7.5-22 uses plane waves, no positions, and models motion as a rotating phase per ray with delays and angles fixed. The only TR clause with positions is 7.6.13 (near field), whose distances do not match the path length. A TR drop can be turned into scatterer positions with a two-bounce solver (exact drop angles where a solution exists); its mobility is then drift, not the TR rotation. So a TR drop can be stored in QRT as geometry, not TR-literally.
2. **QRT stores interaction coordinates and one Jones matrix per frequency** (≤ 127 frequencies, shared coordinates, float32). Path length and gain are recomputed on read (gain = FSPL(length, f) · |M|²), so GBSM path power must be folded into M. Origin = static Tx; CIR = Rx snapshot; destination = group of CIRs. Dual mobility cannot be stored. Measured round trip: gain·M within 4.7e-8; path length error 2.7 µm / 0.31 mm / 1.0 mm for coordinates up to 150 m / 5 km / 20 km from the origin, i.e. phase error 0.003 / 0.39 / 1.29 rad at 60 GHz. Local coordinates are required. QRT has no field for the ray-to-tap grouping.
3. **NG drifting lives inside `get_channels_spherical`** (`update_drifting.m` has no caller). NG keeps FBS/LBS from the drop, moves Tx/Rx, recomputes LOS and the GR point (ground z = 0, GR polarisation R(t)/R0) and passes `path_length = d(t) + τ·c`; the library uses max(path_length, polyline), so a path cannot become shorter than its drop excess delay. To be measured in R0.
4. **NG's TR baseline mobility** is one `get_channels_planar` call at the drop and a phase rotation per sub-path afterwards, with the scalar distance along the Rx heading (the library uses the displacement vector). Its sub-cluster offsets are rescaled tables, not the TR ray mapping of Table 7.5-5.
5. **NG sums sub-paths per cluster and normalises each cluster's segment-average power per antenna pair** (`p_correct`, in all methods including the TR baseline, also for single-snapshot drops). The library does not port it (P18 withdrawn). R0 on library clusters: an antenna-free correction stays below 0.9 dB at p90 from 10 m segments on (≤ 0.35 dB at 28 GHz from 5 m), does not narrow the per-segment power spread at elements off the phase centre, and at a single offset biases elements by up to +10 dB with factors up to 44 dB; without correction the mean power is unbiased within 0.05 dB. NG's `p_correct` at one snapshot removes the whole per-antenna tap fading (V-V tap power p10/p90 −9.6/+3.7 dB → 0 dB, also equal power on neighbouring elements). Removing it changes NG's segment tap powers by ±0.5–0.8 dB in the 3GPP baseline (100 snapshots, 5–20 m) and by −4…−5/+2 dB in QuaDRiGa mode even at 20 m (cause not isolated; checked in the NG integration).
6. **NG multi-frequency QuaDRiGa geometry is per frequency** (coupling and sub-cluster delays per frequency); a multi-frequency QRT file and `get_channels_multifreq` need one geometry.
7. **Sub-path methods in use:** `legacy` in all library conf files and 85 of 91 NG files; `mmMAGIC` in 6 deprecated NG files; `Laplacian` in none. `NumSubPaths` = 20 except one NG file with 10 and one with 26.
8. **Run time.** Ray-level random fields: 33 ns per value and link; the full TR per-frequency ray set (κ, four phases, η, coupling ranks, ≈ 4560 columns) ≈ 0.15 ms per link and frequency, below one `get_channels_planar` call with 381 paths (0.30 ms); no seeded-stream fallback needed. `ssf_init_phases` accepts at most 1000 generators per set, fewer than the TR ray set needs (contract item). Expected `gbsm_single_link_channel` cost (measured under load, provisional): `phase` mobility with constant orientation 0.035 / 0.075 / 1.0 ms per link and snapshot for 1×1 / 2×2 dual-pol / 8×8 dual-pol BS arrays, 11–24× faster than NG's baseline loop; `drift` dominated by `get_channels_spherical` (≈ 1.2 µs per element × path pair), about equal to NG at 8×8 and up to 2× faster for small arrays.
9. **Rays without a scatterer solution:** TR methods put the first NLOS cluster at τ = 0 (LOS scenarios, NLOS without 7.6.9); 4.2–6.4 % of NLOS rays (7.6–13.4 % of NLOS power) have zero excess length; `get_channels_spherical` treats paths at the LOS length as LOS. With the P13 rule at d_min = 2 m the fallback applies to 5.5–9.6 % of rays in the TR methods without absolute TOA and ≤ 2.3 % otherwise; it adds a median delay of 0.2–4.6 ns (p99 ≤ 21.7 ns) and changes the per-link DS by ≤ 4.1 % at p99. d_min 0.5–1 m leaves up to 313 TR LOS rays per 2000 links within 0.1 mm of the LOS length; 5 m doubles the fallback (LOS DS p99 +53 %); d_min = 0 puts zero-excess scatterers on the Rx.
10. **Ray phases between separately dropped UTs.** Measured fading at a UT decorrelates within about 0.4–1 λ (literature); the point-scatterer reference gives per-tap |ρ|² < 0.5 at 0.43–0.86 λ and < 0.1 at 1.5–4 λ. Phases from spatially consistent fields at the cluster distance keep the taps of two separately dropped UTs correlated over 43–440 λ (0.97 at 1 m in UMi NLOS at 3.5 GHz), because all rays of a tap share one length and the cluster change is a common phase; this is TR-literal (7.6.3.1). No phase field alone matches the reference: a λ/2 field decorrelates 3–5× too fast and makes bands independent at 1 MHz; a λ/2 field at one reference frequency is right only there; a geometric phase per ray is ruined by the spatially consistent angle drift (284–959 rad per metre). Separately, the cluster path length d3D + c·τ of the series-1 clusters changes by 2.9–4.1·d per moved distance d (a fixed scatterer allows ≤ d), so the narrowband channel of separate drops decorrelates within 0.02–0.03 λ in every ray variant. One drop plus offsets is the physical route for UTs a few wavelengths apart.
11. **Per-tap correlation across bands.** Measured small-scale fading is uncorrelated between widely separated bands (literature); the per-tap correlation is the squared Fourier transform of the intra-cluster delay profile (coherence bandwidth per tap 20–130 MHz with the TR c_DS, 4–40 MHz with measured intra-cluster spreads). One length per sub-cluster with phases shared across bands gives 1.000 up to 24.5 GHz; independent phases per band (TR 7.6.5, NG) give 0 already at 1 MHz where the reference gives 0.98–1.00.
12. **Per-ray delays serve both.** TR 7.6.2.2 item-2 ray delays U(0, 2·c_DS), shared across bands, give a per-tap cross-band correlation that follows sinc² (coherence bandwidth 1.7–1.8× the reference) with the DS unchanged (ratio ≈ 1.00). Drawn from their own spatially consistent field with decorrelation distance λ_τ = 1.6·c·c_DS/c_ASA (13.7 / 8.1 / 5.8 m in UMi NLOS / UMi LOS / UMa NLOS), they change the relative ray lengths at the plane-wave rate (0.117 vs 0.121, 0.090 vs 0.090, 0.080 vs 0.084 per metre); the constant follows from the two measured rates (field 0.49·c·c_DS/λ_τ, plane waves 0.31·c_ASA in rad). The per-tap spatial decorrelation then matches the reference within 0.03–0.17 λ at 3.5 and 28 GHz with one field, continuously (0.98–1.00 at 1 mm); not reproduced: the anisotropy (reference up to 30 % different between perpendicular and radial moves). Cost: 240–400 shared random columns per link.
13. **The QuaDRiGa rotation polarisation model gives a nearly unitary 2×2 cluster matrix:** co-located X-pol singular-value ratio 0.10–0.22 dB (TR model 7.6–8.3 dB) in all tested cases, instantaneous XPR within ±0.4–0.6 dB of the link XPR (TR model −13 … +10 dB); structural (a ten times larger γ spread gives 2 dB); NLOS coupling loss TR − rotation model +0.8…+1.0 dB. Its transpose is not of rotation form; a symmetric construction exists (P19).
14. **Scatterer solver.** Five closed-form free-distance rules find every ray that has a solution (test: polyline at a = b = d_min ≤ L). The balanced rule (equal distances from both ends) puts the fewest scatterers within 5 m of an array (0.2–6.0 %) and is the smoothest when a ray's length changes (p99 2.4–12 m/m, no jumps); minimum-distance rules put a scatterer at d_min for 24–100 % of rays (drift 7–28× faster). NG's iterative search misses 0.5–1.9 % of solvable rays (≤ 3.7 % of power), is not smooth (up to 19 000 m/m; the valid flag flips for 2.3 % of rays at a 1 cm length change), its single-bounce fallback turns the departure angle by a median of 7.7–129°, and it is 520–630× slower than the balanced rule in C++ (12.5 ns per ray). With d_min > 0 the drift Doppler equals the TR rotation Doppler at the drop, without jumps along tracks.
15. **NG's drift length rule** max(d(t) + τ·c, polyline) gives the LOS Doppler to 86–98 % of rays moving towards their scatterer (0.03–0.27 % of receding rays), Doppler spread 0.55–0.75 of the geometric value, delays lagging up to 115 ns after 20 m.
16. **Angle construction:** the 3-D rotation widens the GCS azimuth spread by +9 % at 20–30° and +19 % at 30–40° cluster elevation; link-level ASA/ASD/ZSD change ≤ 1 %, ZSA −1…−4 %.

## 3. Decisions

| ID | Decision |
|---|---|
| P1 | TR scope: steps 8–11 (antenna-free part) + 7.6.16 polarisation power variability (TR polarisation model only; error with the rotation model) + 7.6.1 oxygen absorption. 7.6.2.2 (large bandwidth) not implemented as a TR option (its item-2 ray delays serve `quadriga`, P21) and the mobile-scatterer term of 7.6.10 (7.6-46) not implemented; no-blocker checks (7.3). The dual-mobility term of 7.6.10 (Tx offset) is part of P7. 7.6.4, 7.6.13, 7.6.14 out of scope. |
| P2 | Scatterer geometry is available for every cluster method. Mobility is a separate switch: `phase` (TR rotation) or `drift` (fixed scatterers). TR methods default to `phase`, `quadriga` to `drift`. A TR drop with `drift` is a documented non-TR combination; it is the way to store a TR drop in QRT. |
| P3 | Ray angle construction: TR additive offsets (with zenith fold) and the 3-D rotation of the offset pattern around the cluster direction, with a switch. Default: additive for all methods; NG passes the rotation explicitly. The rotation keeps the on-sphere spread and widens the GCS azimuth spread by about 1/cos(elevation) (R0: +9 % at 20–30°, +19 % at 30–40° cluster elevation; link-level spreads ≤ 1 %, ZSA −1…−4 %, finding 16). The calibration step checks it again. |
| P4 | Sub-clusters: TR methods split the two strongest clusters per link (per frequency in `38.901_MF`) with the ray mapping of Table 7.5-5 and c_DS per frequency. "Strongest" by the coefficient powers (7.5-6 order; 7.5-8 is used only in the angle mapping 7.5-9/7.5-14); the LOS ray and the GR slot are never split; clusters at power 0 (removed, padding) are not candidates; with one active cluster only that one is split. `quadriga` does not split (P21). Split only at M = 20 (P11). |
| P5 | Ray randomness across frequencies: `quadriga` shares coupling, ray delays (P21), positions and initial phases across frequencies; XPR uses one normalised value per ray with frequency-dependent μ/σ; all intra-cluster spreads (c_DS, c_ZSD from μ_lgZSD, others) frequency-averaged. TR methods draw steps 8–11 per frequency (TR 7.6.5). |
| P6 | Ray XPR, phases and η come from spatially consistent SOS fields in all methods (TR 7.6.3.1), in new append-only generator sets; no seeded-stream fallback (finding 8). Coupling (step 8) = rank of SOS field values per ray and coupled quantity, ranked within each sub-cluster for split clusters and within the cluster otherwise, evaluated at the drop and kept (as the series-1 signs); batch-independent, consistent for nearby UTs, independent of which clusters are split. Phase fields at the cluster decorrelation distance in all methods, no λ-scale phase field (finding 10); in `quadriga` the λ-scale per-tap decorrelation comes from the ray delays (P21). The docs state the inter-UT behaviour of separate drops (finding 10) and recommend one drop plus offsets for UTs a few wavelengths apart. |
| P7 | `phase` mobility: TR-literal phase exp(j·2π·(r̂_rx·Δr_rx + r̂_tx·Δr_tx)/λ) per ray (7.5-22, 7.6-31, 7.6-45) folded into M per frequency; delays and angles fixed; `get_channels_planar` called with the arrays at the drop position (orientations may change per offset). Validity documented: about √(λ·d3D/8) for LOS-dominated links (1.0 m at 3.5 GHz, 0.37 m at 28 GHz for d3D = 100 m); `drift` beyond. Procedure A later (7.2). |
| P8 | Large-scale changes along a track: link gain (PL + SF + O2I, per frequency) is a mandatory per-offset input of the offset assembly (the caller evaluates the existing LSF functions at the drop or the offsets); K enters as the optional ratio K(t)/K(drop) per offset. The assembly rescales LOS/GR against NLOS on the oxygen-free coefficient powers (LOS + NLOS as given, GR on top) and then applies the oxygen loss from the path length at the offset. |
| P9 | API: building blocks (random sets, rays, scatterers, offset assembly, QRT conversion) plus one function on top, `gbsm_single_link_channel` (P17), that runs `get_channels_*` for one link, one frequency and one segment. Tracks, segment boundaries, merging and multi-frequency loops stay with the caller (NG now, a library layout class later). |
| P10 | The blocks keep rays as separate paths; every ray carries a path type (LOS, GR, NLOS) and its group index (TR methods: slot and sub-cluster, sub-clusters have different delays and are separate taps; `quadriga`: the cluster, P21). NG's antenna-dependent `p_correct` is not ported and not replaced (P18). |
| P11 | `SubpathMethod`: `legacy` only; any `NumSubPaths` via one deterministic rescale rule (TR Table 7.5-3 at 20; reordered-table rule with a defined extension above 20; no unseeded draws); sub-clusters only at M = 20 (NG behaviour), documented; `Laplacian` and `mmMAGIC` raise an error. |
| P12 | QRT: generic C++ conversion of spherical path data (FBS, LBS, path type, gain and M per frequency) into `quadriga_lib::path` objects for `qrt_file_append`. No MEX/Python QRT wrappers in this series. Segment overlaps are out of scope (a path-level merge is the bridge for the track project). QRT route = direct route for `drift`, static Tx, `rays` output and local coordinates: amplitude within float32, phase within 2π·f·δL/c (δL ≈ 1e-7 of the coordinate magnitudes); taps cannot be rebuilt from a QRT file (documented). |
| P13 | Scatterer solver: redesign with the intent of NG's `gen_fbs_lbs` (exact angles at both ends where a two-bounce solution exists), closed form with the balanced free-distance rule (equal scatterer distances from both ends; one rule, no switch; finding 14), a minimum distance d_min to both arrays (input, default 2 m, d_min ≤ 0 rejected; finding 9), and one fallback for all methods: a ray without a solution is lengthened to the shortest polyline with scatterers ≥ d_min from both arrays along its own departure and arrival directions (angles exact, delay grows; used length output so that `ssf_paths` and QRT agree; per-ray fallback flag). Padding rays are skipped. Spec note: about 2·10⁻⁶ of track samples in TR LOS come within 0.1 mm of the LOS length (LOS branch of `get_channels_spherical`). No faithful port, no NG parity test. |
| P14 | Validation: Catch2 TR oracles and an R0 study in this series. The TR 7.8.2 calibration follows in a later library-only session; nothing in this series may require QuaDRiGa-NG for it (7.4). |
| P15 | `drift` path length = exact Tx-FBS-LBS-Rx polyline at the offset (pure geometry); NG's `d(t) + τ·c` with the max rule is not reproduced; NG's rule measured in R0 (finding 15). |
| P16 | Polarisation model: both the TR model (κ per ray, four phases, 7.5-21/22) and the QuaDRiGa rotation model (linear XPR γ per ray, circular XPR κ per cluster, one phase per ray), with a switch. Default: TR model for all methods; NG passes the rotation model explicitly. The rotation model does not reproduce the X-pol singular-value statistics of the TR model (finding 13); it is fit as NG's own model only, in the reciprocal form of P19. |
| P17 | `gbsm_single_link_channel`: one link, one frequency, one segment (the list of Tx/Rx positions and orientations of one drop); inputs: the ray data of the link (any path set, P20), both arrays, positions and orientations per snapshot, link gain per snapshot or at the drop and optional K ratio (P8), ground permittivity and GR power law, initial position; output: a `channel` object. Output switch: `taps` (default; rays summed per group, mean delay; LOS tap and GR tap (if enabled) always present, zero power in NLOS; tap count fixed per call; planar delays per tap `[1, 1, n_tap]`) or `rays`. TR methods: a tap is a sub-cluster, its rays share one delay, so `taps` is exact at the phase centre (only the per-element delays across large arrays are averaged). `quadriga`: a tap is a cluster whose rays have different delays (P21), so `taps` is exact only at the carrier (in-band selectivity 0.18 / 0.52 / 0.79 over 20 / 100 / 400 MHz in UMi NLOS; the same approximation as NG's sub-path sum today) and is documented as narrowband per cluster; `rays` is exact at any bandwidth and is the form the QRT route stores. `par` holds the rays per tap, FBS/LBS (`drift`), the GR point and the link gain per snapshot. "phase" mobility with constant orientations computes the planar coefficients once and rotates them per snapshot. No power correction. MEX and Python wrappers. A later layout class (tracks, segments, multi-frequency, scenario assignment, antennas) calls it per segment. |
| P18 | Withdrawn in the plan review: no antenna-free power correction (finding 5); R0 confirmed the withdrawal (no long-segment effect above 1 dB at p90). Removing NG's `p_correct` is on the NG change list. |
| P19 | Swap reciprocity at ray level (as the series-1 clusters, D2): mirror pairs of the ray blocks: coupling ranks AOD↔AOA and ZOD↔ZOA; TR phases Φθφ↔Φφθ and ηθφ↔ηφθ; κ, Φθθ, Φφφ and the ray delays self-mirrored. The rotation model uses the symmetric construction M = e^{jΦ}·[[cosγ·e^{−jα}, sinγ], [sinγ, −cosγ·e^{jα}]] with sin α = sin κ / cos γ: exactly reciprocal, no new random columns, linear and circular XPR as NG's model from both ends, link XPR, X-pol singular values and 4×4 singular values within 0.03 dB of NG's model, slant XPR 0.5–1.2 dB lower per ray; where no α exists (X_lin·X_circ < 1, 0–2.4 % of rays on links with an XPR LSP below 0 dB) a clip rule defined in the spec. Catch2 swap test for both models. |
| P20 | `ssf_scatterers`, `ssf_paths`, `qrt_paths_from_scatterers` and `gbsm_single_link_channel` are layout-agnostic: any number of paths with per-path type and group index as inputs, no `scenpar`. Only `ssf_rays` knows the scenario layout. NG's other path sources (`add_sdc`, `gen_cdl_model`, `add_paths`, user scatterers) can use the solver and the offset assembly. |
| P21 | `quadriga` intra-cluster delays: TR 7.6.2.2 item-2 ray delays τ_n + τ'_n,m with τ'' ~ U(0, 2·c_DS) (minimum subtracted per cluster), shared across frequencies, replace the sub-cluster split; the delay values come from their own spatially consistent SOS set with decorrelation distance λ_τ = 1.6·c·c_DS/c_ASA (documented default; with frequency-averaged c_DS it depends on the frequency set); the group index is the cluster. Gives the λ-scale per-tap decorrelation between UTs and the per-tap cross-band correlation (finding 12) with one field. TR methods unchanged (sub-clusters, per-band draws). |

Choices taken in this plan without a separate question: ground plane at z = 0 (as `ssf_clusters`); LOS ray in `drift` with FBS = LBS on the direct line; LOS Jones diag(1, −1), GR Jones diag(R∥, −R⊥)/R (W7, W22), GR power factor of the TR methods (d3D/d_GR)²·R² recomputed at the offset in `drift`; padded batch layout with a fixed ray count per slot, zero-power rays dropped only by the offset assembly except the LOS and GR taps (W12, P17); QuaDRiGa XPR mean = the link's XPR LSP and σ = 0.1·σ_XPR (NG); a `random_phase` switch in `ssf_rays` (default on) for NG's option without random initial phases.

TR ambiguities recorded: "two strongest" powers (P4); ZOD ray spread in LOS (7.5-20 in LOS and NLOS, 7.6-7); τ_Δ of 7.6.1 with absolute TOA (5.1a).

## 4. Inventory

Legend: **new** = new code, **ext** = extension of an existing function, **NG** = replaces NG code.

### 4.1 Ray level

| # | Feature | Kind | Methods |
|---|---|---|---|
| A1 | Ray offsets α_m (Table 7.5-3) with c_ASA, c_ASD, c_ZSA and ZSD = 3/8·10^μlgZSD (`PerClusterES_D_scale`, LOS and NLOS); any M via the rescale rule (P11) | new, NG | all |
| A2 | Additive construction with zenith fold (7.5-13/18/20; default) and 3-D rotation construction (P3) | new, NG | switch |
| A3 | Random coupling AOD↔AOA, ZOD↔ZOA, AOD↔ZOD (step 8) by ranks of SOS field values, within sub-clusters (P6) | new, NG | all |
| A4 | Sub-clusters (7.5-26, Table 7.5-5, c_DS default 3.91 ns, `PerClusterDS_gamma`, `PerClusterDS_min`, `PerClusterDS_fc_min`), strongest by coefficient powers | new | TR methods |
| A10 | Ray delays of 7.6.2.2 item 2 from an own spatially consistent set at λ_τ = 1.6·c·c_DS/c_ASA, shared across frequencies, group = cluster (P21) | new, NG | `quadriga` |
| A5 | Polarisation: TR κ + 4 phases (default); QuaDRiGa rotation with a swap-symmetric construction (P16, P19); LOS and GR Jones | new, NG | switch |
| A6 | 7.6.16 polarisation power variability (η, normalisation 7.6-62; NLOS rays only, per frequency; XPR of the means 6.95 → 6.41 dB for XPR 8 ± 3 dB) | new | TR polarisation model |
| A7 | 7.6.1 oxygen absorption (per cluster and frequency, applied in the offset assembly; τ_Δ per 5.1a; GR with d_GR) | new | all |
| A8 | Ray powers P_n/M per frequency; path lengths d3D + τ·c (Δτ already in τ, W13) | new | all |
| A9 | Link gain (PL + SF + O2I, per frequency) as an input of the offset assembly | new | all |

### 4.2 Geometry

| # | Feature | Kind |
|---|---|---|
| B1 | Two-bounce solver from ray angles and length, closed form with the balanced free-distance rule; minimum excess length fallback (P13) | new, NG |
| B2 | Minimum distance d_min of scatterers to both arrays (default 2 m, > 0) | new |
| B3 | LOS on the direct line; GR point on z = 0 | new, NG |
| B4 | One geometry for all frequencies in `quadriga` (P5); per frequency in TR multi-frequency | new |

### 4.3 Mobility and offset assembly

| # | Feature | Kind |
|---|---|---|
| C1 | `phase`: per-ray phase rotation for Rx and Tx offsets, per frequency (7.5-22, 7.6-31, 7.6-45) | new |
| C2 | `drift`: fixed FBS/LBS, moved ends, polyline length (P15), LOS and GR recomputed, GR Jones R(t)/R0 and GR power factor of the method | new, NG |
| C3 | Link gain per offset, optional K ratio (P8) | new, NG |
| C4 | Inactive rays dropped (LOS and GR kept); path type and group index per output path (P10) | new |

### 4.3a Channels of one link

| # | Feature | Kind |
|---|---|---|
| H1 | `gbsm_single_link_channel`: per snapshot `get_channels_planar` or `get_channels_spherical` with the link's arrays, positions and orientations; `taps` / `rays` output; `channel` object with the `par` content of P17 | new, NG |
| H2 | Planar speed-up: one planar call per distinct orientation, phase rotation per snapshot | new |

### 4.4 Random fields

| # | Feature | Kind |
|---|---|---|
| F1 | New generator sets for ray variables (XPR, phases, η, coupling ranks; for `quadriga` the ray delays as an own set at λ_τ, P21), appended without changing the series-1 sets (stored seeds keep reproducing series-1 clusters); seed tags with bit 63 set; mirror table (P19); the limit of 1000 generators per set in `ssf_init_phases` raised or the ray sets split (contract item) | ext (`ssf_init_phases`, `ssf_draw`) |
| F2 | Reserved blocks: Procedure A sign X_n (shared), 7.6.2.2 offsets, ray delays and ray powers for the TR methods, 7.6.10 α and D | layout only |

### 4.5 QRT and wrappers

| # | Feature | Kind |
|---|---|---|
| E1 | Spherical path data → `quadriga_lib::path` (path type input; 0/1/2 interactions; gain folded into M per frequency; a scattering interaction code defined in the `path` doc) | new |
| E2 | MEX and Python wrappers for the new GBSM functions and `gbsm_single_link_channel` (not for QRT, P12) | new |

### 4.6 NG code that the library replaces (retired in the NG integration project)

`gen_fbs_lbs`, `private/solve_multi_bounce_opti`, `private/solve_cos_theorem` (all NG path sources through the layout-agnostic solver, P20), `private/update_drifting` (already unused), the sub-cluster split, XPR and phase parts of `gen_ssf_parameters`, `xpr_sos` / `pin_sos` / `subpath_coupling` in `init_sos` (also their users in `check_los`, `add_sdc` and the unmerged channel path), the interior of `qd_builder.get_channels` (path assembly, GR update, Doppler rotation, sub-path summation), which becomes one `gbsm_single_link_channel` call per mobile. NG keeps tracks, segments, merge, layout, builder bookkeeping, the LSF evaluation along the track, `p_correct` until the integration (then removed, NG change list), `private/rotate_sphere` and `private/get_subpath_angles` for the per-cluster spreads and ray counts of `add_sdc` and `gen_cdl_model`, and `clst_dl_sos` as long as the mmMAGIC files exist.

## 5. Draft contracts

All new functions in `include/quadriga_gbsm.hpp`, section "Small-scale fading" (after `ssf_clusters`), except E1 in `include/quadriga_channel.hpp` and `gbsm_single_link_channel` in `include/quadriga_lib.hpp` (next to `get_channels_ieee_indoor`; keeps `quadriga_gbsm.hpp` free of the arrayant and channel types). Names, argument order and layouts are drafts; the spec fixes them. Argument order follows the `ssf_clusters` precedent (outputs before string options and optional pointers). Common notation: N = `NumClusters` − 1, M = `NumSubPaths`, G = 1 if `GR_enabled` else 0, n_ray = 1 + G + N·M (fixed per scenario in `ssf_rays`), n_fd = number of geometry sets (1, or n_freq for TR multi-frequency), n_path = any path count (P20). Ray outputs are path-major `[n_ray, n_link, …]` (per-link data contiguous, matching the inputs of `get_channels_*`; transposed against `ssf_clusters`, documented).

### 5.1 Random sets (extension, `src/qd_gbsm_phases.cpp`)

`ssf_init_phases` and `ssf_draw` get new set types for the ray variables, selected by an appended set-type argument (default: the cluster sets; existing calls and set sizes unchanged): a shared ray set (frequency-independent ray variables: `quadriga` phases, normalised XPR values and coupling ranks; X_n reserved) and a per-frequency ray set (TR κ, four phases, η, coupling ranks). Seed tags with bit 63 set (disjoint from the series-1 tags 1 and 2 + k). Column layout in blocks by variable type, append-only, mirror pairs per P19, documented in `ssf_draw` with the mirror table. `ssf_clusters` keeps its outputs (5.1a).

### 5.1a `ssf_clusters` (extension)

An appended optional output: the delay offset τ_Δ = min(τ'_n) of step 5 per link (0 in LOS scenarios; `quadriga`: 0). Used by 7.6.1 in NLOS without absolute TOA; with absolute TOA the output delays already contain Δτ and τ_Δ is not added. Existing outputs and random columns unchanged.

### 5.2 `ssf_rays` (new, `src/ssf_rays.cpp`)

Ray parameters of a set of links at the drop (TR steps 7 offsets, 8, 9, 10; 7.6.16).

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
              arma::cube &delay,                  // Output: ray delays in [s] rel. to the LOS delay, [n_ray, n_link, n_fd]
              arma::cube &power,                  // Output: ray powers (linear, oxygen-free), [n_ray, n_link, n_freq]
              arma::cube &aod, arma::cube &eod,   // Output: ray angles in [rad], [n_ray, n_link, n_fd]
              arma::cube &aoa, arma::cube &eoa,
              std::vector<arma::cube> &jones,     // Output: M per frequency, n_freq × [8, n_ray, n_link]
              arma::ucube &group,                 // Output: group index per ray (TR: sub-cluster, quadriga: cluster), [n_ray, n_link, n_fd]
              arma::uvec &path_type,              // Output: LOS / GR / NLOS per ray, [n_ray]
              const std::string &method = "38.901",          // cluster method that produced the clusters
              const std::string &angle_model = "additive",   // "additive", "rotation"
              const std::string &pol_model = "38.901",       // "38.901", "quadriga"
              bool pol_variability = false, bool random_phase = true,
              const arma::mat *gr_eps_re = nullptr, const arma::mat *gr_eps_im = nullptr); // GR Jones
```

Properties: a pure function of the cluster parameters and the random inputs (Procedure A readiness, 7.2); `quadriga` ray delays include the 7.6.2.2 intra-cluster offsets (P21); no seed input (coupling from the random sets, P6); the ray layout does not depend on the link (padding, W12); `quadriga` has n_fd = 1 and Jones per frequency with shared phases (P5).

### 5.3 `ssf_scatterers` (new, `src/ssf_scatterers.cpp`)

FBS/LBS positions of paths at the drop (P2, P13, P20).

```
void ssf_scatterers(const arma::mat &tx_pos, const arma::mat &rx_pos,
                    const arma::cube &length, const arma::cube &aod, const arma::cube &eod,
                    const arma::cube &aoa, const arma::cube &eoa,      // [n_path, n_link, n_fd]
                    const arma::uvec &path_type,                       // LOS / GR / NLOS per path
                    std::vector<arma::cube> &fbs_pos, std::vector<arma::cube> &lbs_pos,  // n_fd × [3, n_path, n_link]
                    arma::cube &length_used,                           // Output: path length after the fallback
                    double d_min = 2.0,                                // [m], > 0
                    const arma::Cube<unsigned char> *active = nullptr, // paths to solve (padding skipped)
                    arma::Cube<unsigned char> *fallback = nullptr);    // Optional output: fallback applied
```

MEX/Python return the positions as one 4-D array `[3, n_path, n_link, n_fd]`.

### 5.4 `ssf_paths` (new, `src/ssf_paths.cpp`)

Inputs of `get_channels_planar` / `get_channels_spherical` for one link at a list of offsets (P7, P8, P10, P15, P20). Each offset is evaluated on its own.

```
void ssf_paths(const arma::vec &tx_pos0, const arma::vec &rx_pos0,      // drop positions, [3]
               const arma::mat &tx_offset, const arma::mat &rx_offset,  // [n_off, 3] or [1, 3] / empty
               const arma::vec &fc,                                     // [n_freq]
               path data of one link (length, power, angles, jones, path_type, group; fbs/lbs for drift),
               const arma::mat &gain_dB,                                // PL + SF + O2I, [n_off, n_freq] or [1, n_freq]
               const std::string &mobility,                             // "phase" or "drift"
               std::vector<...> per offset: aod/eod/aoa/eoa or fbs/lbs, path_length [n_path],
                                 path_gain [n_path, n_freq], M [8, n_path, n_freq], group [n_path],
               const arma::mat *kf_ratio = nullptr,                     // optional K(t)/K(drop), [n_off, n_freq]
               const std::string &gr_law = "38.901",                    // GR power factor: TR (d3D/d_GR)²·R² or quadriga R²
               const arma::mat *gr_eps_re = nullptr, const arma::mat *gr_eps_im = nullptr,
               bool oxygen = false, const arma::vec *delay_offset = nullptr);  // 7.6.1, τ_Δ from 5.1a
```

The output per offset is exactly the argument set of one `get_channels_*` call (one frequency: column `f` of gain, slice `f` of M). Zero-power paths are dropped except the LOS and GR paths.

### 5.5 `qrt_paths_from_scatterers` (new, generic, `src/qd_channel_qrt.cpp`)

```
std::vector<quadriga_lib::path> qrt_paths_from_scatterers(
    const arma::vec &tx_pos, const arma::vec &rx_pos,     // CIR end points (local coordinates)
    const arma::mat &fbs_pos, const arma::mat &lbs_pos,   // [3, n_path]
    const arma::uvec &path_type,                          // LOS / GR / NLOS
    const arma::mat &path_gain, const arma::cube &M,      // [n_path, n_freq], [8, n_path, n_freq]
    const arma::vec &freq, unsigned i_cir);               // [n_freq] in Hz; CIR index (path::iC)
```

Folds the gain into M with FSPL of the polyline length; LOS → 0, GR and FBS = LBS → 1 (GR with a reflection code), else 2 interactions (scattering code). Round trip with `qrt_file_read` is the main test, with local coordinates and the tolerance of P12. Writer memory: `qrt_file_append` takes all paths of one origin at once (about 3–4 GB for 1000 UTs × 100 snapshots × 381 rays at one frequency), documented.

### 5.6 `gbsm_single_link_channel` (new, `src/gbsm_single_link_channel.cpp`, `quadriga_lib.hpp`)

Channel coefficients of one link at one frequency over one segment (P17).

```
quadriga_lib::channel<double> gbsm_single_link_channel(
    const arrayant<double> &tx_array, const arrayant<double> &rx_array,
    const arma::mat &tx_pos, const arma::mat &tx_orientation,   // [n_snap, 3] or [1, 3]
    const arma::mat &rx_pos, const arma::mat &rx_orientation,   // [n_snap, 3] or [1, 3]
    path data of one link at one frequency (as for ssf_paths, incl. drop positions, path type and group),
    double fc,
    const arma::vec &gain_dB,                                    // [n_snap] or [1]
    const std::string &mobility,                                 // "phase" or "drift"
    const std::string &output = "taps",                          // "taps" or "rays"
    const arma::vec *kf_ratio = nullptr, const std::string &gr_law = "38.901",
    std::complex<double> gr_eps = ..., bool oxygen = false, double delay_offset = 0.0,
    const arma::vec *initial_position = nullptr, bool use_absolute_delays = false);
```

Internally: `ssf_paths` at the offsets (positions minus drop positions), then `get_channels_*` per snapshot (or once per orientation for planar), summation per group in `taps` mode. The `par` data hold the rays per tap, FBS/LBS (`drift`), the GR point and the link gain per snapshot. Multi-frequency: one call per frequency with the same path data.

### 5.7 Wrappers

MEX and Python for 5.1 (extension), 5.1a, 5.2, 5.3, 5.4, 5.6. Not for 5.5 (P12). The path data of one link are passed as one struct (MEX) or dict (Python) whose fields are the output names of `ssf_rays` / `ssf_scatterers` (about 12 positional inputs instead of 27 for `gbsm_single_link_channel`); C++ keeps flat arguments.

## 6. Data flow

Drop: `lsf_*` / `lsp_generate` / `path_loss` / `o2i_loss` → `ssf_init_phases` + `ssf_draw` (cluster and ray sets) → `ssf_clusters` → `ssf_rays` → (`drift`:) `ssf_scatterers`. Per link and segment: `gbsm_single_link_channel` (= `ssf_paths`, then `get_channels_planar` for `phase` or `get_channels_spherical` for `drift` with the antennas). Optional QRT route: `ssf_paths` (`drift`) at the CIR positions of a segment → `qrt_paths_from_scatterers` → `qrt_file_append`; later `qrt_file_read` → `get_channels_*` with any antennas gives the same channel in `rays` form within the tolerance of P12. Other path sources (NG's extra clusters, CDL, user scatterers) enter at `ssf_scatterers` / `ssf_paths` with their own path type and group (P20).

## 7. Readiness checks (side quests)

Each check is a short paragraph in the spec that shows the contract allows the later addition without changing existing outputs or random-field columns; the spec review verifies it.

### 7.1 Multi-frequency paths (`get_channels_multifreq`)

Needed: one geometry, gain `[n_path, n_freq_in]` and M `[8, n_path, n_freq_in]` at sample frequencies; the function interpolates gain linearly and M by SLERP. Available after this series in `quadriga` (P5): shared coupling, ray delays (P21), positions and phases; per-frequency powers, K, SF, PL and XPR from the same normalised random values. Gap for a later session: a "sample grid" mode (many closely spaced frequencies, cost of per-frequency evaluation; dense grid near the 60 GHz oxygen line), checks that powers and XPR vary smoothly between samples, and a validation of the frequency correlation (the per-tap behaviour is settled by P21, finding 12). TR methods do not qualify (7.6.5 per-frequency steps; `38.901_MF` geometry per frequency). Feasible; the main limit is physical (one scatterer set over a very wide band), which `quadriga` already assumes.

### 7.2 Procedure A (TR 7.6.3.2)

Needs: absolute cluster delays and angles at the segment start (available: `ssf_clusters` + d3D), velocities per step (caller), the sign X_n per cluster (reserved block, F2), re-derivation of rays from updated clusters (`ssf_rays` is a pure function of the cluster parameters and the drop random values; coupling ranks and phases from the drop), cluster powers from the updated delays (a step-6 function with the stored Z_n), ray data per step of ≤ 1 m (segments of that length or a per-snapshot ray input), the 7.6-10a delay normalisation over the whole mobility range (a later function over the step sequence). No contract change expected.

### 7.3 7.6.2.2 (large bandwidth) and the mobile-scatterer term of 7.6.10

7.6.2.2: M from 7.6-8 is ≥ 20 and depends on the bandwidth B, the array sizes D_h, D_v and λ (so per frequency); `NumSubPaths` becomes the user's M_max (100–1000 scales the N·M blocks and the ray layout); per-frequency ray counts through zero-power padding; inputs B, D_h, D_v added then; the sub-cluster mapping is replaced (group = cluster, as `quadriga` already has with P21); random offsets, ray delays and unequal ray powers in reserved blocks (F2) for the TR methods. 7.6.10 (7.6-46): adds a Doppler 2·α·D/λ per ray that needs time; `ssf_paths` would get an optional time vector per offset; α and D blocks reserved.

### 7.4 Library-only TR 7.8.2 calibration

Needs: many sites and sectors in one batch (per-link `tx_pos` is supported), rays shared by co-sited sectors (TR: steps 1–9 identical; the rays are antenna-free, sectors differ only in the `get_channels_*` call), per-site seeds (series 1), LOS and NLOS links of a layout handled by separate calls per scenario (W1: LOS is a scenario property; batch-independent coupling, P6) and joined by the caller, TR antenna models (`arrayant_generate`), and `gbsm_single_link_channel` as the per-link channel generator (one snapshot per drop, library defaults = TR). Nothing in this series may require QuaDRiGa-NG; the R0 notes for P3 and P16 go to the calibration session.

## 8. Process (steps 2–7)

A manager session tracks plan, spec and code and starts agents; the maintainer talks only to the manager. Working folders: repository `/home/stephan/quadriga-lib-ssf` (branch `ssf`), internals (briefs, reports, R0 data, review reports) in `/home/stephan/quadriga-lib-agents/ssf2/`. Coding rules: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`; produced code cites only published material and the SPEC files.

1. **Plan review (step 2):** round 1 done (two independent reviews with prototypes; v3). Next: contract review with the maintainer; the full contracts of section 5 become plan v5, which gets a further review round (physics and TR, NG fit, API design) before the spec.
2. **R0 study (done):** literature search (about 60 sources) and NumPy prototypes on library clusters, against a point-scatterer reference (each ray a point scatterer whose distance matches its delay, intra-cluster length spread c·c_DS, exact spherical field; it reproduces Clarke's narrowband decorrelation and the Fourier relation for the cross-band coherence). Results: findings 5 and 8–16; decisions P13, P19, P21; confirmations of P3, P15, P16, P17, P18. Run times were measured under load and stay provisional.
3. **Spec (step 3), spec review (step 4):** one mandatory round, up to three if major issues are found; each reviewer checks the TR transcription numerically. Doc rules in the spec: API only, each fact once, no formulas, size targets set against `ssf_clusters`; doc blocks follow the block grammar and writing pitfalls of `docs_builder/README.md`. The spec states the validity limits and model properties of findings 10–13 and P7, and the `taps` bandwidth limit of P17, for the docs.
4. **Implementation (step 5)**, test-first, in runs: R0' one preparatory change (all declarations, doc stubs and empty test files, so that later runs do not edit shared headers); R1 random sets (5.1, 5.1a); R2 `ssf_rays` TR methods; R3 `ssf_rays` QuaDRiGa models, 7.6.16; R4 `ssf_scatterers`; R5 `ssf_paths` (needs R2/R3 for the group index and the P13 fallback); R6 `qrt_paths_from_scatterers`; R7 `gbsm_single_link_channel` (QRT round-trip test against the direct route; needs R5 and R6); R8 wrappers and documentation. R4 and R6 may run in parallel with R2/R3 in separate worktrees (R6 tests write fixed-name `.qrt` files); `make docs` once per merge by the manager. Shared test headers and method-generic test cases are specified up front (lesson: test sizes exceeded 1.5–2×).
5. **Code review (step 6):** up to three rounds, one mandatory, plus a security review (inputs, sizes, file output of 5.5).
6. **Cleanup and merge (step 7):** working documents deleted, `md/ssf_notes.md` updated, the maintainer commits.
7. **NG integration (step 8):** separate project; retires the code of 4.6; change list includes the removal of `p_correct`, the displacement-vector Doppler, the rotation model's swap-symmetric construction, the TR ray mapping of the sub-clusters.

## 9. Open items

### 9.1 For the contract review (plan v5)

- Full C++ declarations, MEX and Python signatures for section 5; flat arguments or a small struct for the path data of one link in C++; extension points for the side quests of section 7 (only appended optional arguments).
- Random-set layout table: blocks, shared vs per-frequency sets, mirror pairs (P19), seed tags, the ray-delay set of `quadriga` (P21), and the generator limit of `ssf_init_phases` (finding 8).
- The `group` output now differs by method (sub-cluster in the TR methods, cluster in `quadriga`).

### 9.2 For the spec

- The clip rule of the reciprocal rotation construction (P19).
- The rescale rule for M ≠ 20 above 20 (P11).
- The interaction code for scattering in the `path` doc (E1).
- c_DS of UMa below 6 GHz: note 6 of Table 7.5-6 needs `PerClusterDS_fc_min`, which the v19 files do not set (4.71 vs 3.91 ns at 3.5 GHz).
- λ_τ with the frequency-averaged c_DS depends on the frequency set (P21); document it.
