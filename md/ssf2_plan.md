# SSF series 2: path generator (plan)

Plan for the second small-scale fading series: from the clusters of `ssf_clusters` to propagation paths that feed `get_channels_planar`, `get_channels_spherical` and the QRT writer, for TR 38.901 v19.5.0 clause 7.5 steps 7 (ray offsets) to 11 (antenna-free part) and the QuaDRiGa method. Working document; deleted before the merge, its outcome goes to `md/ssf_notes.md`. Builds on series 1 (`md/ssf_notes.md`: decisions D1–D8, W1–W22).

Status: plan v5 (step 2 of 7). v3 folded in the first plan review (two independent reviews with prototype measurements): the antenna-free power correction withdrawn (P18), coupling from ranked random fields (P6), the ray angle and polarisation defaults follow the TR in every method (P3, P16), ray-level reciprocity (P19), layout-agnostic per-link functions (P20), a fallback for rays without a scatterer solution (P13). v4 added the R0 study (a literature search and measurements on library clusters against a point-scatterer reference; findings 5 and 8–16): per-ray delays from their own random field in `quadriga` (P21), the solver rule and d_min (P13), the reciprocal rotation construction (P19). v5 folds in the contract review: the full C++ contracts (section 5) with the ray data of one link as the class `gbsm_rays`, the merged random-set functions (P22), one frequency name (P23), `SubpathMethod` removed from the API (P24), zero-copy wrappers (P25), the terms (P26, section 10), oxygen absorption from each path's own length (P1), the validity limits of both mobility modes (finding 17, P7, P15), the sub-path literature (finding 18) and a corrected finding 8. Next: a full review round (physics and TR, NG fit, API design) before the spec.

## 1. Goal and requirements

1. Consume the output of `ssf_clusters` (slots LOS, GR, N clusters; delays, powers per frequency, angles) for all three methods (`38.901`, `38.901_MF`, `quadriga`).
2. Generate everything `get_channels_planar` (angles) and `get_channels_spherical` (FBS/LBS) need: path gain, path length, polarisation transfer matrix M, angles or scatterer positions.
3. Implement the TR ray step stand-alone in quadriga-lib (no QuaDRiGa-NG needed), TR-literal for the TR methods; the ray angle and polarisation defaults follow the TR in every method.
4. Serve QuaDRiGa-NG so that its own path code can be retired later (integration is a separate project); NG selects its non-TR variants by switches.
5. Feed the C++ QRT writer (`qrt_file_append`) with antenna-free paths, so that antennas and orientations can be added later at low cost.
6. Support mobility: the TR phase rotation and QuaDRiGa drifting, both driven by real Tx/Rx positions relative to the drop. Tracks and segments stay in QuaDRiGa-NG.

Side quests (no implementation, the design must not block them): multi-frequency paths for `get_channels_multifreq` (section 7.1), TR Procedure A (7.2), TR 7.6.2.2 and the mobile-scatterer term of 7.6.10 (7.3), a library-only TR 7.8.2 calibration (7.4).

## 2. Findings from the analysis, the plan review and R0

1. **The TR baseline is angle-based.** TR 7.5-22 uses plane waves, no positions, and models motion as a rotating phase per ray with delays and angles fixed. The only TR clause with positions is 7.6.13 (near field), whose distances do not match the path length. A TR drop can be turned into scatterer positions with a two-bounce solver (exact drop angles where a solution exists); its mobility is then drift, not the TR rotation. So a TR drop can be stored in QRT as geometry, not TR-literally.
2. **QRT stores interaction coordinates and one Jones matrix per frequency** (≤ 127 frequencies, shared coordinates, float32). Path length and gain are recomputed on read (gain = FSPL(length, f) · |M|²), so GBSM path power must be folded into M. Origin = static Tx; CIR = Rx snapshot; destination = group of CIRs. Dual mobility cannot be stored. Measured round trip: gain·M within 4.7e-8; path length error 2.7 µm / 0.31 mm / 1.0 mm for coordinates up to 150 m / 5 km / 20 km from the origin, i.e. phase error 0.003 / 0.39 / 1.29 rad at 60 GHz. Local coordinates are required. QRT has no field for the ray-to-tap grouping.
3. **NG drifting lives inside `get_channels_spherical`** (`update_drifting.m` has no caller). NG keeps FBS/LBS from the drop, moves Tx/Rx, recomputes LOS and the GR point (ground z = 0, GR polarisation R(t)/R0) and passes `path_length = d(t) + τ·c`; the library uses max(path_length, polyline), so a path cannot become shorter than its drop excess delay (measured in R0, finding 15).
4. **NG's TR baseline mobility** is one `get_channels_planar` call at the drop and a phase rotation per sub-path afterwards, with the scalar distance along the Rx heading (the library uses the displacement vector). Its sub-cluster offsets are rescaled tables, not the TR ray mapping of Table 7.5-5.
5. **NG sums sub-paths per cluster and normalises each cluster's segment-average power per antenna pair** (`p_correct`, in all methods including the TR baseline, also for single-snapshot drops). The library does not port it (P18 withdrawn). R0 on library clusters: an antenna-free correction stays below 0.9 dB at p90 from 10 m segments on (≤ 0.35 dB at 28 GHz from 5 m), does not narrow the per-segment power spread at elements off the phase centre, and at a single offset biases elements by up to +10 dB with factors up to 44 dB; without correction the mean power is unbiased within 0.05 dB. NG's `p_correct` at one snapshot removes the whole per-antenna tap fading (V-V tap power p10/p90 −9.6/+3.7 dB → 0 dB, also equal power on neighbouring elements). Removing it changes NG's segment tap powers by ±0.5–0.8 dB in the 3GPP baseline (100 snapshots, 5–20 m) and by −4…−5/+2 dB in QuaDRiGa mode even at 20 m (cause not isolated; checked in the NG integration).
6. **NG multi-frequency QuaDRiGa geometry is per frequency** (coupling and sub-cluster delays per frequency); a multi-frequency QRT file and `get_channels_multifreq` need one geometry.
7. **Sub-path methods in use:** `legacy` in all 23 library conf files and 85 of 91 NG files; `mmMAGIC` in 6 deprecated NG files; `Laplacian` in none. `NumSubPaths` = 20 except one NG file with 10 and one with 26.
8. **Run time and set sizes.** Ray-level random fields: 33 ns per value and link; the full TR per-frequency ray set (κ, four phases, η, coupling ranks, ≈ 4560 columns) ≈ 0.15 ms per link and frequency, below one `get_channels_planar` call with 381 paths (0.30 ms); no seeded-stream fallback needed. There is no generator limit: the generator count of a set follows from its type (9·N or N for the cluster sets; N ≤ 999 only because `NumClusters` ≤ 1000); the 999 of plan v4 came from the R0 stand-in ray sets, which were built as per-frequency cluster sets with `NumClusters` = columns + 1. Expected `gbsm_single_link_channel` cost (measured under load, provisional): `phase` mobility with constant orientation 0.035 / 0.075 / 1.0 ms per link and snapshot for 1×1 / 2×2 dual-pol / 8×8 dual-pol BS arrays, 11–24× faster than NG's baseline loop; `drift` dominated by `get_channels_spherical` (≈ 1.2 µs per element × path pair), about equal to NG at 8×8 and up to 2× faster for small arrays.
9. **Rays without a scatterer solution:** TR methods put the first NLOS cluster at τ = 0 (LOS scenarios, NLOS without 7.6.9); 4.2–6.4 % of NLOS rays (7.6–13.4 % of NLOS power) have zero excess length; `get_channels_spherical` treats paths at the LOS length as LOS. With the P13 rule at d_min = 2 m the fallback applies to 5.5–9.6 % of rays in the TR methods without absolute TOA and ≤ 2.3 % otherwise; it adds a median delay of 0.2–4.6 ns (p99 ≤ 21.7 ns) and changes the per-link DS by ≤ 4.1 % at p99. d_min 0.5–1 m leaves up to 313 TR LOS rays per 2000 links within 0.1 mm of the LOS length; 5 m doubles the fallback (LOS DS p99 +53 %); d_min = 0 puts zero-excess scatterers on the Rx.
10. **Ray phases between separately dropped UTs.** Measured fading at a UT decorrelates within about 0.4–1 λ (literature); the point-scatterer reference gives per-tap |ρ|² < 0.5 at 0.43–0.86 λ and < 0.1 at 1.5–4 λ. Phases from spatially consistent fields at the cluster distance keep the taps of two separately dropped UTs correlated over 43–440 λ (0.97 at 1 m in UMi NLOS at 3.5 GHz), because all rays of a tap share one length and the cluster change is a common phase; this is TR-literal (7.6.3.1). No phase field alone matches the reference: a λ/2 field decorrelates 3–5× too fast and makes bands independent at 1 MHz; a λ/2 field at one reference frequency is right only there; a geometric phase per ray is ruined by the spatially consistent angle drift (284–959 rad per metre). Separately, the cluster path length d3D + c·τ of the series-1 clusters changes by 2.9–4.1·d per moved distance d (a fixed scatterer allows ≤ d), so the narrowband channel of separate drops decorrelates within 0.02–0.03 λ in every ray variant. One drop plus offsets is the physical route for UTs a few wavelengths apart.
11. **Per-tap correlation across bands.** Measured small-scale fading is uncorrelated between widely separated bands (literature); the per-tap correlation is the squared Fourier transform of the intra-cluster delay profile (coherence bandwidth per tap 20–130 MHz with the TR c_DS, 4–40 MHz with measured intra-cluster spreads). One length per sub-cluster with phases shared across bands gives 1.000 up to 24.5 GHz; independent phases per band (TR 7.6.5, NG) give 0 already at 1 MHz where the reference gives 0.98–1.00.
12. **Per-ray delays serve both.** TR 7.6.2.2 item-2 ray delays U(0, 2·c_DS), shared across bands, give a per-tap cross-band correlation that follows sinc² (coherence bandwidth 1.7–1.8× the reference) with the DS unchanged (ratio ≈ 1.00). Drawn from their own spatially consistent field with decorrelation distance λ_τ = 1.6·c·c_DS/c_ASA (c_ASA in rad; 13.7 / 8.1 / 5.8 m in UMi NLOS / UMi LOS / UMa NLOS), they change the relative ray lengths at the plane-wave rate (0.117 vs 0.121, 0.090 vs 0.090, 0.080 vs 0.084 per metre); the constant follows from the two measured rates (field 0.49·c·c_DS/λ_τ, plane waves 0.31·c_ASA in rad). The per-tap spatial decorrelation then matches the reference within 0.03–0.17 λ at 3.5 and 28 GHz with one field, continuously (0.98–1.00 at 1 mm); not reproduced: the anisotropy (reference up to 30 % different between perpendicular and radial moves). Cost: 240–400 shared random columns per link.
13. **The QuaDRiGa rotation polarisation model gives a nearly unitary 2×2 cluster matrix:** co-located X-pol singular-value ratio 0.10–0.22 dB (TR model 7.6–8.3 dB) in all tested cases, instantaneous XPR within ±0.4–0.6 dB of the link XPR (TR model −13 … +10 dB); structural (a ten times larger γ spread gives 2 dB); NLOS coupling loss TR − rotation model +0.8…+1.0 dB. Its transpose is not of rotation form; a symmetric construction exists (P19).
14. **Scatterer solver.** Five closed-form free-distance rules find every ray that has a solution (test: polyline at a = b = d_min ≤ L). The balanced rule (equal distances from both ends) puts the fewest scatterers within 5 m of an array (0.2–6.0 %) and is the smoothest when a ray's length changes (p99 2.4–12 m/m, no jumps); minimum-distance rules put a scatterer at d_min for 24–100 % of rays (drift 7–28× faster). NG's iterative search misses 0.5–1.9 % of solvable rays (≤ 3.7 % of power), is not smooth (up to 19 000 m/m; the valid flag flips for 2.3 % of rays at a 1 cm length change), its single-bounce fallback turns the departure angle by a median of 7.7–129°, and it is 520–630× slower than the balanced rule in C++ (12.5 ns per ray). With d_min > 0 the drift Doppler equals the TR rotation Doppler at the drop, without jumps along tracks.
15. **NG's drift length rule** max(d(t) + τ·c, polyline) gives the LOS Doppler to 86–98 % of rays moving towards their scatterer (0.03–0.27 % of receding rays), Doppler spread 0.55–0.75 of the geometric value, delays lagging up to 115 ns after 20 m.
16. **Angle construction:** the 3-D rotation widens the GCS azimuth spread by +9 % at 20–30° and +19 % at 30–40° cluster elevation; link-level ASA/ASD/ZSD change ≤ 1 %, ZSA −1…−4 %.
17. **Validity range of `drift`** (R0 S6: 1000 links per case, UMi/UMa LOS/NLOS, `quadriga` and `38.901`, 3.5 and 28 GHz, dual mobility, `SC_lambda` ×0.5–×2, d_min 1/2/5 m). The power-weighted mean change of the arrival angle reaches 10° at 0.17·b_med per link (p50 in every case, p10–p90 0.03–0.28), with b_med the power-weighted median distance of the moving end to its last scatterer: 1.5–5.6 m in LOS and 7.8–12.4 m in NLOS at p50. The limit scales with b_med (correlation 0.67–0.86) and not with frequency, d_min, `SC_lambda` or d3D (correlation ≤ 0.3); dual mobility shortens it by about 20 %. At 0.5·b_med the mean change is about 30° and the cluster set is half way to an independent drop. The `phase` limit √(λ·d3D/8) is 1.2–1.5 m at 3.5 GHz and 0.44–0.54 m at 28 GHz; `drift` is valid 1.2–3.7× (LOS) and 6–8× (NLOS) longer at 3.5 GHz, 3–10× and 18–23× at 28 GHz. Under `drift` |ΔL| ≤ d holds exactly and the DS changes by rms lg 0.003–0.03 at 10 m (fresh drops of the scenario 0.14–0.45); beyond about 0.4·b_med the ASA shrinks (scatterers recede, −0.10 lg at 20 m in UMi NLOS). Fresh spatially consistent drops move away from the drop 2–6× faster than the drifted geometry (half the independent-drop distance at about 0.2·`SC_lambda`), so a segment boundary jumps by the SC model's own change, not by drift error.
18. **Sub-path methods in the literature.** The only alternative ray method inside TR 38.901 is 7.6.2.2 (random per-ray offsets, per-ray delays, unequal powers, M from bandwidth, array size and M_max), an optional switch that depends on the simulation (bandwidth, aperture), not on the scenario; the map-based hybrid model (clause 8.4) repeats the same split, and 7.6.13, 7.6.14 and 7.9 reuse the Table 7.5-3 rays. Sionna offers Table 7.5-3 only (sub-clustering flag); MATLAB `nrCDLChannel` Table 7.5-3 with 20 rays (`NumStrongestClusters`, `ClusterDelaySpread`); ns-3 (3.49) adds 7.6.2.2 as the boolean `LargeBandwidthArrayModeling` with `ChannelBandwidth` and `MaxRaysPerCluster`. None ties a ray method to the scenario. `mmMAGIC` and `Laplacian` exist only in QuaDRiGa (no 3GPP or ITU text, no publication using them; `Laplacian` only inside `gen_cdl_model` for CDL/TDL-D, -E).

## 3. Decisions

| ID | Decision |
|---|---|
| P1 | TR scope: steps 8–11 (antenna-free part) + 7.6.16 polarisation power variability (TR polarisation model only; error with the rotation model) + 7.6.1 oxygen absorption from each path's own length at the position. τ_Δ of 7.6-1 is not added (documented deviation: 0.017 dB median, up to 0.71 dB at p99 at 60 GHz in TR NLOS without absolute TOA); `ssf_clusters` gets no τ_Δ output. 7.6.2.2 (large bandwidth) not implemented as a TR option (its item-2 ray delays serve `quadriga`, P21; later a simulation option, 7.3) and the mobile-scatterer term of 7.6.10 (7.6-46) not implemented; no-blocker checks (7.3). The dual-mobility term of 7.6.10 (Tx offset) is part of P7. 7.6.4, 7.6.13, 7.6.14 out of scope. |
| P2 | Scatterer geometry is available for every cluster method. Mobility is a separate switch: `phase` (TR rotation) or `drift` (fixed scatterers); default `"auto"`: `phase` for the TR methods, `drift` for `quadriga`, taken from `gbsm_rays.method`. A TR drop with `drift` is a documented non-TR combination; it is the way to store a TR drop in QRT. |
| P3 | Ray angle construction: TR additive offsets (with zenith fold) and the 3-D rotation of the offset pattern around the cluster direction, with a switch. Default: additive for all methods; NG passes the rotation explicitly. The rotation keeps the on-sphere spread and widens the GCS azimuth spread by about 1/cos(elevation) (R0: +9 % at 20–30°, +19 % at 30–40° cluster elevation; link-level spreads ≤ 1 %, ZSA −1…−4 %, finding 16). The calibration step checks it again. |
| P4 | Sub-clusters: TR methods split the two strongest clusters per link (per frequency in `38.901_MF`) with the ray mapping of Table 7.5-5 and c_DS per frequency. "Strongest" by the coefficient powers (7.5-6 order; 7.5-8 is used only in the angle mapping 7.5-9/7.5-14); the LOS ray and the GR slot are never split; clusters at power 0 (removed, padding) are not candidates; with one active cluster only that one is split. `quadriga` does not split (P21). Split only at M = 20 (P11). |
| P5 | Ray randomness across frequencies: `quadriga` shares coupling, ray delays (P21), positions and initial phases across frequencies; XPR uses one normalised value per ray with frequency-dependent μ/σ; all intra-cluster spreads (c_DS, c_ZSD from μ_lgZSD, others) frequency-averaged. TR methods draw steps 8–11 per frequency (TR 7.6.5). |
| P6 | Ray XPR, phases and η come from spatially consistent SOS fields in all methods (TR 7.6.3.1), in the new set type "ray" (P22); no seeded-stream fallback (finding 8). Coupling (step 8) = rank of SOS field values per ray and coupled quantity, ranked within each sub-cluster for split clusters and within the cluster otherwise, evaluated at the drop and kept (as the series-1 signs); batch-independent, consistent for nearby UTs, independent of which clusters are split. Phase fields at the cluster decorrelation distance in all methods, no λ-scale phase field (finding 10); in `quadriga` the λ-scale per-tap decorrelation comes from the ray delays (P21). The docs state the inter-UT behaviour of separate drops (finding 10) and recommend one drop plus offsets for UTs a few wavelengths apart. |
| P7 | `phase` mobility: TR-literal phase exp(j·2π·(r̂_rx·Δr_rx + r̂_tx·Δr_tx)/λ) per ray (7.5-22, 7.6-31, 7.6-45) folded into M per frequency, Δr = position − position of the ray data; delays and angles fixed; `get_channels_planar` called with the arrays at the positions of the ray data (orientations may change per snapshot). Valid up to about √(λ·d3D/8): 1.2–1.5 m at 3.5 GHz, 0.44–0.54 m at 28 GHz for typical d3D (finding 17); documented in `ssf_paths` and `gbsm_single_link_channel`, reported in the channel's `par`; `drift` beyond (P15). Procedure A later (7.2). |
| P8 | Large-scale changes along a track: one link gain `link_gain_dB` = −PL − O2I + SF in dB per frequency (0 for data from paths) is a mandatory per-position input of `ssf_paths` and `gbsm_single_link_channel` (the caller evaluates the existing LSF functions at the drop or the positions); K enters as the optional `kf_ratio` = K(position)/K(drop). The assembly rescales LOS/GR against NLOS on the oxygen-free coefficient powers (LOS + NLOS as given, GR on top), applies the link gain and then the oxygen loss from the path length at the position. |
| P9 | API: building blocks (random sets `gbsm_init_phases`/`gbsm_draw`, `ssf_rays`, `ssf_scatterers`, `ssf_paths`, the QRT pair of P12) plus one function on top, `gbsm_single_link_channel` (P17), that runs `get_channels_*` for one link, one frequency and one segment. The ray data of one link travel as one object of the class `gbsm_rays` (public Armadillo attributes as `channel` and `arrayant`; members `n_ray`, `n_freq`, `empty`, `is_valid`). Tracks, segment boundaries, merging and multi-frequency loops stay with the caller (NG now, a library layout class later). |
| P10 | The blocks keep rays as separate paths; every ray carries a path type (0 LOS, 1 GR, 2 NLOS) and a tap index, in all methods `tap` = 3·slot + sub-cluster (LOS 0, GR 3, NLOS cluster n: 3·(1 + G + n) + i, i = 0 unless split, always 0 in `quadriga`); taps in ascending index, so LOS first and GR second. NG's antenna-dependent `p_correct` is not ported and not replaced (P18). |
| P11 | Ray offsets: only the TR Table 7.5-3 method (QuaDRiGa `legacy`), without a scenario key (P24). Any `NumSubPaths` via one deterministic rescale rule (TR Table 7.5-3 at 20; reordered-table rule with a defined extension above 20; no unseeded draws); sub-clusters only at M = 20 (NG behaviour), documented. TR 7.6.2.2 can come later as a simulation option of `ssf_rays` (appended arguments, 7.3), not as a scenario key (finding 18). |
| P12 | QRT: a generic C++ pair: `qrt_paths_from_scatterers` (spherical path data of one CIR → `quadriga_lib::path` objects for `qrt_file_append`) and the reverse `gbsm_rays_from_paths` (flat path data as `qrt_file_read` returns them → `gbsm_rays`). No MEX/Python wrapper for `qrt_paths_from_scatterers`. Segment overlaps are out of scope (a path-level merge is the bridge for the track project). QRT route = direct route for `drift`, static Tx, `rays` output and local coordinates: amplitude within float32, phase within 2π·f·δL/c (δL ≈ 1e-7 of the coordinate magnitudes). QRT stores neither the tap index nor the GR type; `gbsm_rays_from_paths` takes both as optional inputs (documented). |
| P13 | Scatterer solver `ssf_scatterers`: redesign with the intent of NG's `gen_fbs_lbs` (exact angles at both ends where a two-bounce solution exists), closed form with the balanced free-distance rule (equal scatterer distances from both ends; one rule, no switch; finding 14), a minimum distance d_min to both ends (default 2 m, d_min ≤ 0 throws; finding 9), and one fallback for all methods: a ray without a solution is lengthened to the shortest polyline with scatterers ≥ d_min from both ends along its own departure and arrival directions (angles exact, delay grows; flag `fallback`). Every consumer uses the polyline (P15), so no used-length output is needed. LOS on the direct line, GR at the specular point on z = 0, rays with zero power at all frequencies at the midpoint of the direct line. Spec note: about 2·10⁻⁶ of track samples in TR LOS come within 0.1 mm of the LOS length (LOS branch of `get_channels_spherical`). No faithful port, no NG parity test. |
| P14 | Validation: Catch2 TR oracles and an R0 study in this series. The TR 7.8.2 calibration follows in a later library-only session; nothing in this series may require QuaDRiGa-NG for it (7.4). |
| P15 | `drift` path length = exact Tx-FBS-LBS-Rx polyline at the position (pure geometry); NG's `d(t) + τ·c` with the max rule is not reproduced (finding 15). Valid up to about 0.2·b_med per moving end (b_med = power-weighted median distance of the moving end to its last scatterer: Rx to LBS, Tx to FBS), a new segment from about 0.5·b_med (finding 17); documented in `ssf_paths` and `gbsm_single_link_channel`, reported in the channel's `par`. |
| P16 | Polarisation model: both the TR model (κ per ray, four phases, 7.5-21/22) and the QuaDRiGa rotation model (linear XPR γ per ray, circular XPR κ per cluster, one phase per ray), with a switch. Default: TR model for all methods; NG passes the rotation model explicitly. The rotation model does not reproduce the X-pol singular-value statistics of the TR model (finding 13); it is fit as NG's own model only, in the reciprocal form of P19. |
| P17 | `gbsm_single_link_channel`: one link, one frequency of the ray data, one segment (absolute Tx/Rx positions and orientations per snapshot; default the positions of the ray data); no initial-position input (the caller sets the class member). Output `channel<double>`, `taps` (default; rays summed per tap, mean delay; LOS and GR taps always present, zero power in NLOS; tap count fixed per call) or `rays` (one path per kept ray, with the path fields of the class); `par` with the tap data, the geometry, the link gain and the validity report of P7/P15. TR methods: a tap is a sub-cluster, its rays share one delay, so `taps` is exact at the phase centre (only the per-element delays across large arrays are averaged). `quadriga`: a tap is a cluster whose rays have different delays (P21), so `taps` is exact only at the carrier (in-band selectivity 0.18 / 0.52 / 0.79 over 20 / 100 / 400 MHz in UMi NLOS; the same approximation as NG's sub-path sum today) and is documented as narrowband per cluster; `rays` is exact at any bandwidth and is the form the QRT route stores. `phase` with constant orientations computes the planar coefficients once and rotates them per snapshot. No power correction. MEX and Python wrappers. A later layout class (tracks, segments, multi-frequency, scenario assignment, antennas) calls it per segment. |
| P18 | Withdrawn in the plan review: no antenna-free power correction (finding 5); R0 confirmed the withdrawal (no long-segment effect above 1 dB at p90). Removing NG's `p_correct` is on the NG change list. |
| P19 | Swap reciprocity at ray level (as the series-1 clusters, D2): mirror pairs of the "ray" set are the coupling ranks AOD↔AOA and ZOD↔ZOA, the TR phases Φθφ↔Φφθ and η θφ↔η φθ; XPR, Φθθ, Φφφ, η θθ, η φφ, the cluster κ and the ray delays are self-mirrored. The rotation model uses the symmetric construction M = e^{jΦ}·[[cosγ·e^{−jα}, sinγ], [sinγ, −cosγ·e^{jα}]] with sin α = sin κ / cos γ: exactly reciprocal, no new random columns, linear and circular XPR as NG's model from both ends, link XPR, X-pol singular values and 4×4 singular values within 0.03 dB of NG's model, slant XPR 0.5–1.2 dB lower per ray; where no α exists (X_lin·X_circ < 1, 0–2.4 % of rays on links with an XPR LSP below 0 dB) a clip rule (review round, section 9). Catch2 swap test for both models. |
| P20 | `ssf_scatterers`, `ssf_paths`, `qrt_paths_from_scatterers` and `gbsm_single_link_channel` are layout-agnostic: one `gbsm_rays` with any number of rays, each with its path type and tap index; no `scenpar`. Only `ssf_rays` knows the scenario layout. Other path sources (NG's `add_sdc`, `gen_cdl_model`, `add_paths`, user scatterers, QRT files) fill a `gbsm_rays` directly or with `gbsm_rays_from_paths` and then use the solver, the assembly and the channel function. |
| P21 | `quadriga` intra-cluster delays: TR 7.6.2.2 item-2 ray delays τ_n + τ'_n,m with τ'' ~ U(0, 2·c_DS) (minimum subtracted per cluster), shared across frequencies, replace the sub-cluster split; the values come from the ray-delay block of the shared "ray" set (P22), whose generators use their own decorrelation distance λ_τ = 1.6·c·c_DS/c_ASA (c_ASA in rad; frequency-averaged c_DS, so λ_τ depends on the frequency set; `gbsm_draw` computes it from `center_frequency`); the tap is the cluster. Gives the λ-scale per-tap decorrelation between UTs and the per-tap cross-band correlation (finding 12) with one field. TR methods unchanged (sub-clusters, per-band draws). |
| P22 | Random sets: `lsf_init_phases`, `lsf_draw`, `ssf_init_phases`, `ssf_draw` are merged into `gbsm_init_phases`, `gbsm_draw` with a mandatory second argument `set_type` ∈ {"lsf", "cluster", "ray"}. "lsf" and "cluster" keep generators, distributions and seed rules bit for bit; the bias-removal arguments of the init function (positions, dual mobility, `remove_bias`, `horizontal`) apply to "lsf" only; `set_id` is ignored for "lsf". The new "ray" set has 13·N·M + N generators per frequency; the shared set adds the ray-delay block (P21); seed tags 2^63 + t·2^32 + set_id + 2 with t = 1; `SC_lambda = 0` is used as 0.1 m as in all SSF sets. No generator limit (finding 8). The name `init_sos_phases` was rejected (clash with `sos_init_phases`). |
| P23 | One frequency name: `center_frequency` (Hz, `[n_freq]`) in all GBSM functions, old and new, as in `get_channels_*`, `channel` and `qrt_file_read`; `fc` is renamed in `los_probability`, `lsp_generate`, `o2i_loss`, `path_loss` and their wrappers. |
| P24 | `SubpathMethod` leaves the API: no output of `conf_file_read`, no argument of `conf_file_write`, `scenpar_validate` or `ssf_rays` (nor of their wrappers). Readers accept the line `SubpathMethod = legacy` and throw on any other value; the writer omits the key (finding 18). The rescale rule of P11 for M ≠ 20 stays. |
| P25 | Wrappers (MEX, Python) map outputs zero-copy (`qd_mex_init_output` / `qd_python_init_output`) wherever the output size is known in advance, in all GBSM wrappers; the existing GBSM wrappers are updated mechanically. `gbsm_rays` ↔ struct (MEX) / dict (Python) with the attribute names. |
| P26 | Terms: a "Terms" paragraph in the GBSM section description defines the terms of section 10 (TR names where they exist, QuaDRiGa names marked); the GBSM API uses no "geometry set", "group" or "CIR"; size names `n_link`, `n_slot`, `n_ray`, `n_path`, `n_freq`, `n_snap`, `n_tap`, `n_rx`, `n_tx`; `n_fd` leaves the `ssf_clusters` docs. |

Choices taken in this plan without a separate question: ground plane at z = 0 (as `ssf_clusters`); LOS ray in `drift` with FBS = LBS on the direct line; LOS Jones diag(1, −1), GR Jones diag(R∥, −R⊥)/R (W7, W22), GR power factor of the TR methods (d3D/d_GR)²·R² recomputed at the position in `drift`; padded ray layout with a fixed ray count per slot, zero-power rays dropped only by `ssf_paths` and `gbsm_single_link_channel` except the LOS and GR rays (W12, P17); QuaDRiGa XPR mean = the link's XPR LSP and σ = 0.1·σ_XPR (NG); a `random_phase` switch in `ssf_rays` (default on) for NG's option without random initial phases.

TR ambiguities recorded: "two strongest" powers (P4); ZOD ray spread in LOS (7.5-20 in LOS and NLOS, 7.6-7); τ_Δ of 7.6.1 (not added, P1).

## 4. Inventory

Legend: **new** = new code, **ext** = extension of an existing function, **NG** = replaces NG code.

### 4.1 Ray level (`ssf_rays`)

| # | Feature | Kind | Methods |
|---|---|---|---|
| A1 | Ray offsets α_m (Table 7.5-3) with c_ASA, c_ASD, c_ZSA and ZSD = 3/8·10^μlgZSD (`PerClusterES_D_scale`, LOS and NLOS); any M via the rescale rule (P11) | new, NG | all |
| A2 | Additive construction with zenith fold (7.5-13/18/20; default) and 3-D rotation construction (P3) | new, NG | switch |
| A3 | Random coupling AOD↔AOA, ZOD↔ZOA, AOD↔ZOD (step 8) by ranks of SOS field values, within sub-clusters (P6) | new, NG | all |
| A4 | Sub-clusters (7.5-26, Table 7.5-5, c_DS default 3.91 ns, `PerClusterDS_gamma`, `PerClusterDS_min`, `PerClusterDS_fc_min`), strongest by coefficient powers | new | TR methods |
| A10 | Ray delays of 7.6.2.2 item 2 from the ray-delay block at λ_τ = 1.6·c·c_DS/c_ASA, shared across frequencies, tap = cluster (P21) | new, NG | `quadriga` |
| A5 | Polarisation: TR κ + 4 phases (default); QuaDRiGa rotation with a swap-symmetric construction (P16, P19); LOS and GR Jones | new, NG | switch |
| A6 | 7.6.16 polarisation power variability (η, normalisation 7.6-62; NLOS rays only, per frequency; η shared across frequencies in `quadriga`; XPR of the means 6.95 → 6.41 dB for XPR 8 ± 3 dB) | new | TR polarisation model |
| A8 | Ray powers P_n/M per frequency; ray delays relative to the LOS delay (Δτ already in τ, W13) | new | all |
| A11 | Path type and tap index per ray, one formula for all methods (P10) | new | all |

### 4.2 Geometry (`ssf_scatterers`)

| # | Feature | Kind |
|---|---|---|
| B1 | Two-bounce solver from ray angles and delay, closed form with the balanced free-distance rule; minimum excess length fallback with flag (P13) | new, NG |
| B2 | Minimum distance d_min of scatterers to both ends (default 2 m, > 0) | new |
| B3 | LOS on the direct line; GR point on z = 0; zero-power rays at the midpoint of the direct line | new, NG |
| B4 | One geometry for all frequencies in `quadriga` (P5); TR methods one geometry per frequency (one `ssf_rays` call each) | new |

### 4.3 Mobility and path assembly (`ssf_paths`)

| # | Feature | Kind |
|---|---|---|
| C1 | `phase`: per-ray phase rotation for Rx and Tx displacements, per frequency (7.5-22, 7.6-31, 7.6-45) | new |
| C2 | `drift`: fixed FBS/LBS, moved ends, polyline length (P15), LOS and GR recomputed, GR Jones R(t)/R0 and GR power factor of the method | new, NG |
| C3 | `link_gain_dB` per position, optional `kf_ratio` (P8); `mobility = "auto"` from the method (P2) | new, NG |
| C4 | Zero-power rays dropped (LOS and GR kept); path type and ray index per output path (P10) | new |
| A7 | 7.6.1 oxygen absorption per path from its own length at the position (τ_Δ not added, P1) | new |

### 4.4 Channels of one link (`gbsm_single_link_channel`)

| # | Feature | Kind |
|---|---|---|
| H1 | Per snapshot `get_channels_planar` or `get_channels_spherical` with the link's arrays, positions and orientations; `taps` / `rays` output; `channel` object with the `par` content of P17 | new, NG |
| H2 | Planar speed-up: one planar call per distinct orientation, phase rotation per snapshot | new |
| H3 | Validity report: b_med and the largest displacement relative to the limit of the active mode (P7, P15) | new |

### 4.5 Random fields (`gbsm_init_phases`, `gbsm_draw`)

| # | Feature | Kind |
|---|---|---|
| F1 | Set type "ray": 13·N·M + N generators per frequency, the shared set with the ray-delay block at λ_τ (P21); seed tags with bit 63 set; mirror table (P19); "lsf" and "cluster" sets unchanged, so stored seeds keep reproducing series-1 results | ext (merged functions, P22) |
| F2 | Reserved blocks: Procedure A sign X_n (block 9 of the shared "cluster" set), 7.6.2.2 ray delays (block 14 of the per-frequency "ray" sets), offsets and ray powers for the TR methods, 7.6.10 α and D | layout only |

### 4.6 Ray data class, QRT and wrappers

| # | Feature | Kind |
|---|---|---|
| G1 | Class `gbsm_rays`: public attributes, members `n_ray`, `n_freq`, `empty`, `is_valid` | new |
| E1 | `qrt_paths_from_scatterers`: spherical path data → `quadriga_lib::path` (LOS 0, GR 1, NLOS 2 interactions; gain and inverse FSPL folded into M per frequency; a scattering interaction code defined in the `path` doc) | new |
| E3 | `gbsm_rays_from_paths`: flat path data (as `qrt_file_read` returns them) → `gbsm_rays` | new, NG |
| E2 | MEX and Python wrappers for `gbsm_init_phases`, `gbsm_draw`, `ssf_rays`, `ssf_scatterers`, `ssf_paths`, `gbsm_rays_from_paths`, `gbsm_single_link_channel` (not for `qrt_paths_from_scatterers`, P12); `gbsm_rays` ↔ struct / dict; zero-copy (P25) | new |

### 4.7 Changes to existing functions

| # | Change | Functions | Cost class |
|---|---|---|---|
| X1 | Merged into `gbsm_init_phases` and `gbsm_draw` (P22), with all library callers, wrappers, tests and docs; header comments that name `lsf_draw` / `ssf_draw` (`lsp_generate`, `path_loss`, `ssf_clusters`) follow | `lsf_init_phases`, `lsf_draw`, `ssf_init_phases`, `ssf_draw` | rework existing |
| X2 | `fc` → `center_frequency` (P23) | `los_probability`, `lsp_generate`, `o2i_loss`, `path_loss` and wrappers | rework existing |
| X3 | `SubpathMethod` removed (P24) | `conf_file_read`, `conf_file_write`, `scenpar_validate` and wrappers; the line removed from the 23 shipped conf files; all internal callers updated (an old call still compiles because the next `std::string *` argument moves into its place: find every caller by search) | rework existing |
| X4 | Sizes written out, `n_fd` removed from the docs (P26) | `ssf_clusters` (docs and header comments) | rework existing |
| X5 | Zero-copy outputs (P25) | all existing GBSM wrappers | rework existing |

### 4.8 NG code that the library replaces (retired in the NG integration project)

`gen_fbs_lbs`, `private/solve_multi_bounce_opti`, `private/solve_cos_theorem` (all NG path sources through the layout-agnostic solver, P20), `private/update_drifting` (already unused), the sub-cluster split, XPR and phase parts of `gen_ssf_parameters`, `xpr_sos` / `pin_sos` / `subpath_coupling` in `init_sos` (also their users in `check_los`, `add_sdc` and the unmerged channel path), the interior of `qd_builder.get_channels` (path assembly, GR update, Doppler rotation, sub-path summation), which becomes one `gbsm_single_link_channel` call per mobile. NG keeps tracks, segments, merge, layout, builder bookkeeping, the LSF evaluation along the track, `p_correct` until the integration (then removed, NG change list), `private/rotate_sphere` and `private/get_subpath_angles` for the per-cluster spreads and ray counts of `add_sdc` and `gen_cdl_model`, and `clst_dl_sos` as long as the mmMAGIC files exist.

## 5. Contracts

The declarations below are what the spec takes over; they compile with `g++ -std=c++17 -fsyntax-only -Wall -Wextra -Wpedantic` against the repository headers, together with a usage sketch of the whole chain. The bullets are the behaviour the spec must keep.

### 5.1 Conventions

- **Placement.** All new functions and the class in `include/quadriga_gbsm.hpp`, doc section "Geometry-based stochastic model": `gbsm_init_phases` and `gbsm_draw` in a header group `// ---- Random sets ----` before "Large-scale fading"; `gbsm_rays`, `ssf_rays`, `ssf_scatterers`, `ssf_paths`, `gbsm_rays_from_paths` in "Small-scale fading" after `ssf_clusters`. Exceptions: `qrt_paths_from_scatterers` in `include/quadriga_channel.hpp`, doc section "Channel functions"; `gbsm_single_link_channel` in `include/quadriga_lib.hpp` (next to `get_channels_ieee_indoor`, keeps `quadriga_gbsm.hpp` free of the arrayant and channel types), doc section "Geometry-based stochastic model".
- **Notation and size names.** N = `NumClusters` − 1 NLOS clusters, M = `NumSubPaths`, G = 1 if `GR_enabled` > 0 else 0. `n_link` links of a batch; `n_slot` = `NumClusters` + G slots of `ssf_clusters`; `n_ray` rays of one link (1 + G + N·M from `ssf_rays`, incl. LOS, GR and zero-power padding; any count from other sources); `n_path` paths after zero-power rays are dropped; `n_freq` frequencies; `n_snap` snapshots; `n_tap` taps; `n_rx`, `n_tx` antenna ports.
- **Units and layouts.** Hz, s, m, rad, GCS, elevation angles; linear powers unless the name ends in `_dB`. Batch positions are rows (`[n_link, 3]`, `[1, 3]`), one position is `[3]`, per-snapshot positions `[n_snap, 3]` (the `channel` class stores `[3, n_snap]`); scatterer positions are columns `[3, n_ray]` / `[3, n_path]` as in `get_channels_*` and `qrt_file_read`. Delays are relative to the LOS delay d3D/c unless stated.
- **Codes.** Path type 0 = LOS, 1 = GR, 2 = NLOS. Tap index `tap` = 3·slot + sub-cluster: LOS 0, GR 3, NLOS cluster n: 3·(1 + G + n) + i, i ∈ {0, 1, 2} only for split clusters, always 0 in `quadriga`; taps are listed in ascending index. Both are defined once, in the section description.
- **Argument order.** Required inputs, mandatory outputs, then all arguments with a default: options, optional inputs, optional outputs, flags; extensions are appended (`md/SPEC_CPP_API.md`).
- **Types and outputs.** Double only, no templates. Mandatory inputs and outputs by reference, optional ones by pointer with `nullptr` = not given / not computed. Outputs are resized only when their size differs, so pre-sized outputs (also the entries of `std::vector<gbsm_rays>`) are written in place. Invalid input throws `std::invalid_argument` naming the argument (class attributes as `'rays.aod'`); the consumers of `gbsm_rays` call `is_valid()` first.
- **Wrappers.** MEX and Python for all functions of this section except `qrt_paths_from_scatterers` (P12); outputs zero-copy where their size is known in advance (P25), which excludes the `n_path` outputs of `ssf_paths`; `gbsm_rays` ↔ struct (MEX) / dict (Python) with the attribute names, `std::vector<gbsm_rays>` ↔ 1 × n_link struct array / list of dicts.

### 5.2 `gbsm_init_phases` (replaces `lsf_init_phases`, `ssf_init_phases`; `src/qd_gbsm_phases.cpp`)

```
// Draw the random phases of one generator set of a BS site and scenario
// - Set types: "lsf" large-scale generators, "cluster" cluster random values, "ray" ray random values; generators
//   and columns: see gbsm_draw
// - The phases are a fixed function of seed, set_type and set_id, equal on all platforms; the sets are independent
// - "lsf": bias removal over the positions in 'rx_pos' (and 'tx_pos' for dual mobility) when given and n_link > 2
void gbsm_init_phases(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances; NumClusters; NumSubPaths for "ray")
                      const std::string &set_type,                            // "lsf", "cluster" or "ray"
                      arma::cube &phase,                                      // Output: Phases in [rad], [n_coeff, n_gen, 2]; slice 0 = Rx side, slice 1 = Tx side
                      arma::sword seed = -1,                                  // Seed of the site and scenario, the same for all set types; -1 = system random device
                      arma::sword set_id = -1,                                // "cluster", "ray": -1 = shared set, k >= 0 = set of frequency k (0-based); ignored for "lsf"
                      const std::string &preset = "Comb256",                  // SOS preset, see sos_preset
                      const arma::mat *tx_pos = nullptr,                      // Optional: Tx positions for the bias removal, [1, 3] or [n_link, 3]; "lsf" only
                      const arma::mat *rx_pos = nullptr,                      // Optional: Rx positions for the bias removal (2D: one floor), [n_link, 3]; "lsf" only
                      bool dual_mobility = false,                             // "lsf": Tx and Rx both move, the Tx side is a second device
                      bool remove_bias = true,                                // "lsf": remove the sample mean over the positions
                      bool horizontal = true);                                // "lsf": 2D bias removal in the x-y plane (TR); false: 3D (legacy)
```

- `n_gen` per set as in the set table (5.3); `NumClusters = 1` gives empty "cluster" and "ray" sets. Unknown `set_type` throws.
- "lsf" equals `lsf_init_phases` and "cluster" equals `ssf_init_phases` bit for bit for the same seed (series-1 seed rules and tags); "ray" uses the tag 2^63 + 2^32 + set_id + 2 in the series-1 seed derivation.
- `tx_pos` or `rx_pos` given with a set type other than "lsf" throws; `dual_mobility`, `remove_bias` and `horizontal` are documented as ignored there; `set_id` is ignored for "lsf".
- Migration: `lsf_init_phases(sp, ph, seed, &tx, &rx, dual, rb, preset, hor)` → `gbsm_init_phases(sp, "lsf", ph, seed, -1, preset, &tx, &rx, dual, rb, hor)`; `ssf_init_phases(sp, ph, seed, id, preset)` → `gbsm_init_phases(sp, "cluster", ph, seed, id, preset)`.

### 5.3 `gbsm_draw` (replaces `lsf_draw`, `ssf_draw`; `src/qd_gbsm_phases.cpp`)

```
// Spatially correlated normalized random values of one generator set at a set of links
// - Columns, distributions and decorrelation distances of each set type: see the set table in the documentation
// - Dual mobility: the Tx side uses the mirror generator; swapping 'tx_pos' and 'rx_pos' exchanges the mirror pairs
// - Links with different floor IDs are uncorrelated at equal horizontal positions (2D mode only)
void gbsm_draw(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances, NumClusters, NumSubPaths, PerCluster* for "ray")
               const std::string &set_type,                             // "lsf", "cluster" or "ray"; must match gbsm_init_phases
               const arma::cube &phase,                                 // Phases from gbsm_init_phases, [n_coeff, n_gen, 2]
               const arma::mat &tx_pos,                                 // Tx positions in [m], [1, 3] or [n_link, 3]
               const arma::mat &rx_pos,                                 // Rx positions in [m], [n_link, 3]
               arma::mat &rnd,                                          // Output: Random values, [n_link, n_gen]
               arma::sword set_id = -1,                                 // As in gbsm_init_phases; ignored for "lsf"
               const std::string &preset = "Comb256",                   // Must match gbsm_init_phases
               const arma::uvec *floor_id = nullptr,                    // Optional: Floor index per link, [n_link]; nullptr = all on one floor; 2D only
               const arma::vec *center_frequency = nullptr,             // Optional: Carrier frequencies in [Hz], [n_freq]; required by the shared "ray" set
               bool dual_mobility = false,                              // Tx and Rx both move: the Tx side is a second device
               bool horizontal = true);                                 // true: 2D correlation in the x-y plane (TR); false: 3D (legacy)
```

Sets (the table replaces the large-scale generator table of the section description and the column list of `ssf_draw`):

| `set_type`, `set_id` | `n_gen` | Blocks | Decorrelation distance | Mirror pairs | Seed tag | Consumer |
|---|---|---|---|---|---|---|
| "lsf" | 11 | 0–7 DS, KF, SF, ASD, ASA, ESD, ESA, XPR, Normal(0,1); 8 absolute TOA, Normal(0,1); 9 ground permittivity, 10 UMa hE, Uniform(0,1) | the LSP keys, `absTOA_lambda`, `SC_lambda`, `SF_lambda` | ASD↔ASA, ESD↔ESA | series 1 | `lsp_generate` `rnd`; `path_loss` `rnd_he` (column 10) |
| "cluster", −1 | 9·N | series 1: 0 cluster delay, Uniform(0,1); 1, 2 sign AOD, AOA; 3, 4 sign ZOD, ZOA; 5, 6 offset AOD, AOA; 7, 8 offset ZOD, ZOA, Normal(0,1); column b·N + n | `SC_lambda` | 1↔2, 3↔4, 5↔6, 7↔8 | 1 | `ssf_clusters` `rnd` |
| "cluster", k | N | cluster shadowing, Normal(0,1) | `SC_lambda` | self | 2 + k | `ssf_clusters` `rnd_freq`, slice k |
| "ray", −1 | 14·N·M + N | 0–13 as below, 14 ray delay | `SC_lambda`; block 14: λ_τ | below | 2^63 + 2^32 + 1 | `ssf_rays` `rnd_ray` (`quadriga`) |
| "ray", k | 13·N·M + N | 0–13 as below; 14 reserved (7.6.2.2, 7.3) | `SC_lambda` | below | 2^63 + 2^32 + 2 + k | `ssf_rays` `rnd_ray` (TR methods, frequency k) |

Blocks of the "ray" sets; column b·N·M + n·M + m for blocks 0–12 (ray m of NLOS cluster n), 13·N·M + n for block 13, 13·N·M + N + n·M + m for block 14:

| Block | Variable | Distribution | Mirror | Used by |
|---|---|---|---|---|
| 0, 1 | coupling rank AOD, AOA (step 8) | Normal(0,1) | 0↔1 | all methods |
| 2, 3 | coupling rank ZOD, ZOA | Normal(0,1) | 2↔3 | all methods |
| 4 | XPR per ray: κ (TR model, 7.5-21) or γ (rotation model) | Normal(0,1) | self | both models |
| 5 | phase Φθθ (rotation model: its single phase) | Normal(0,1), mapped with Φ | self | both models |
| 6, 7 | phase Φθφ, Φφθ | Normal(0,1), mapped with Φ | 6↔7 | TR model |
| 8 | phase Φφφ | Normal(0,1), mapped with Φ | self | TR model |
| 9 | η θθ (7.6.16) | Normal(0,1) | self | `pol_variability` |
| 10, 11 | η θφ, η φθ | Normal(0,1) | 10↔11 | `pol_variability` |
| 12 | η φφ | Normal(0,1) | self | `pol_variability` |
| 13 (N columns) | circular XPR κ per cluster | Normal(0,1) | self | rotation model |
| 14 (shared set) | ray delay τ'' = 2·c_DS·u | Uniform(0,1) | self | `quadriga` (P21) |

- Block 14 uses λ_τ = 1.6·c·c̄_DS/c_ASA, c̄_DS the mean of the c_DS of the frequencies in `center_frequency` (`PerClusterDS`, `PerClusterDS_gamma`, `PerClusterDS_min`, `PerClusterDS_fc_min`), c_ASA = `PerClusterAS_A` in rad; a decorrelation distance of 0 (`SC_lambda = 0`, spatial consistency off) is used as 0.1 m for all blocks. `center_frequency` is required for "ray" with `set_id = -1` and ignored otherwise.
- A column depends neither on the other columns nor on the other links; later versions append blocks (e.g. Procedure A X_n as block 9 of the shared "cluster" set, 7.6.2.2 and 7.6.10 blocks of the "ray" sets) and keep the values of existing columns. Changing `NumSubPaths` changes the "ray" layout as `NumClusters` changes all SSF layouts.
- Size and cost at N = 19, M = 20: 5339 generators in the shared "ray" set, phases `[256, 5339, 2]` ≈ 22 MB, draws ≈ 33 ns per value and link (≈ 0.18 ms per link, provisional).
- Migration: `lsf_draw(sp, ph, tx, rx, rnd, &fl, dual, preset, hor)` → `gbsm_draw(sp, "lsf", ph, tx, rx, rnd, -1, preset, &fl, nullptr, dual, hor)`; `ssf_draw(sp, ph, tx, rx, rnd, id, &fl, dual, preset, hor)` → `gbsm_draw(sp, "cluster", ph, tx, rx, rnd, id, preset, &fl, nullptr, dual, hor)` (`preset` moves before `floor_id`).

### 5.4 Changes to existing functions (only the changed lines)

`ssf_clusters` (header comments and docs only; signature and values unchanged; X4):

```diff
-    // the random values: see the documentation of ssf_draw
+    // the random values: see the documentation of gbsm_draw
-                      const arma::mat &rnd,                                   // Shared cluster set from ssf_draw, [n_link, 9·N]
+                      const arma::mat &rnd,                                   // Shared cluster set from gbsm_draw ("cluster", set_id = -1), [n_link, 9·N]
-                      arma::cube &delay,                                      // Output: Delays in [s], [n_link, n_slot, n_fd]; n_fd = n_freq for 38.901_MF, else 1
+                      arma::cube &delay,                                      // Output: Delays in [s], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
-                      arma::cube &aod,                                        // Output: Azimuth of departure in [rad], [n_link, n_slot, n_fd]
+                      arma::cube &aod,                                        // Output: Azimuth of departure in [rad], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
-                      arma::cube &eod,                                        // Output: Elevation of departure in [rad], [n_link, n_slot, n_fd]
+                      arma::cube &eod,                                        // Output: Elevation of departure in [rad], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
-                      arma::cube &aoa,                                        // Output: Azimuth of arrival in [rad], [n_link, n_slot, n_fd]
+                      arma::cube &aoa,                                        // Output: Azimuth of arrival in [rad], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
-                      arma::cube &eoa,                                        // Output: Elevation of arrival in [rad], [n_link, n_slot, n_fd]
+                      arma::cube &eoa,                                        // Output: Elevation of arrival in [rad], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
-                      const arma::cube *rnd_freq = nullptr,                   // Optional: Per-frequency sets from ssf_draw, [n_link, N, n_freq]; required by 38.901, 38.901_MF
+                      const arma::cube *rnd_freq = nullptr,                   // Optional: Per-frequency cluster sets from gbsm_draw ("cluster", set_id = k), [n_link, N, n_freq]; required by 38.901, 38.901_MF
```

`conf_file_read`, `scenpar_validate`, `conf_file_write` (X3):

```diff
 bool conf_file_read(...
-                        std::string *subpath_method = nullptr,                                 // Optional output: Value of 'SubpathMethod'
 bool scenpar_validate(...
-                          const std::string &subpath_method = "legacy",     // Value of 'SubpathMethod'
 void conf_file_write(...
-                         const std::string &subpath_method = "legacy",                                // Value of 'SubpathMethod'
```

- `conf_file_read` accepts the line `SubpathMethod = legacy` (NG's and older files carry it; the line is removed from the 23 shipped files in R1) and throws on any other value, naming the key and the value; `conf_file_write` never writes the key (also with `write_defaults = true`), and a written file still reads back bit for bit. The `key_order` doc drops its remark on the legacy position of `SubpathMethod`.
- Migration hazard: in `conf_file_read` the next optional output `pl_model` is also `std::string *`, so an old call that passes `&subpath_method` as third argument still compiles and fills it with `PL_model`; an old `conf_file_write(fn, sp, "legacy")` compiles with `pl_model = "legacy"`. All library callers, wrappers and tests change in the same run; the release notes state it.

`los_probability`, `lsp_generate`, `o2i_loss`, `path_loss` (X2; values unchanged; doc text that names the argument follows):

```diff
-    // - Soft LOS (7.6-18) when 'rnd_g', 'fc' and 'los_soft' are given
+    // - Soft LOS (7.6-18) when 'rnd_g', 'center_frequency' and 'los_soft' are given
-                         const arma::vec *fc = nullptr,     // Optional: Carrier frequencies in [Hz] for the soft LOS state, [n_freq]
+                         const arma::vec *center_frequency = nullptr,   // Optional: Carrier frequencies in [Hz] for the soft LOS state, [n_freq]

-                      const arma::vec &fc,                                    // Carrier frequencies in [Hz], [n_freq]
+                      const arma::vec &center_frequency,                      // Carrier frequencies in [Hz], [n_freq]
-                      const arma::mat &rnd,                                   // Random values from lsf_draw, [n_link, 11]
+                      const arma::mat &rnd,                                   // Random values from gbsm_draw ("lsf"), [n_link, 11]

-    void o2i_loss(const arma::vec &fc,                       // Carrier frequencies in [Hz], [n_freq]
+    void o2i_loss(const arma::vec &center_frequency,         // Carrier frequencies in [Hz], [n_freq]

-                   const arma::vec &fc,                                                  // Carrier frequencies in [Hz], [n_freq]
+                   const arma::vec &center_frequency,                                 // Carrier frequencies in [Hz], [n_freq]
-                   const arma::vec *rnd_he = nullptr);                                   // Optional: Uniform(0,1) per link for the random hE, column 10 of lsf_draw, [n_link]
+                   const arma::vec *rnd_he = nullptr);                                // Optional: Uniform(0,1) per link for the random hE, column 10 of gbsm_draw ("lsf"), [n_link]
```

- C++ callers are not affected by the renames (positional); Python keyword calls `fc=` break and are listed in the release notes.
- Section description: the workflow paragraph names `gbsm_init_phases` / `gbsm_draw` and the ray level, the large-scale generator table moves into the set table of `gbsm_draw`, and the "Terms" paragraph is added (P26).

### 5.5 `gbsm_rays` (new class; members in `src/qd_gbsm_rays.cpp`)

```
// Ray data of one link (geometry-based stochastic model)
// - Output of ssf_rays and gbsm_rays_from_paths; ssf_scatterers adds the scatterer positions
// - Input of ssf_paths and gbsm_single_link_channel, which call is_valid() first
// - Set the attributes by name: later versions append attributes
class gbsm_rays
{
public:
    std::string method = "38.901"; // Method of the ray data: "38.901", "38.901_MF" or "quadriga"; sets the GR power law and the default mobility
    arma::vec tx_pos;               // Tx position the ray data refer to in [m], [3]
    arma::vec rx_pos;               // Rx position the ray data refer to in [m], [3]
    arma::vec center_frequency;     // Carrier frequencies in [Hz], [n_freq]
    arma::vec delay;                // Delays in [s] relative to the LOS delay |rx_pos - tx_pos| / c, [n_ray]
    arma::mat power;                // Coefficient powers (linear), without link gain; data from paths: path gain, [n_ray, n_freq]
    arma::vec aod;                  // Azimuth of departure in [rad], GCS, [n_ray]
    arma::vec eod;                  // Elevation of departure in [rad], GCS, [n_ray]
    arma::vec aoa;                  // Azimuth of arrival in [rad], GCS, [n_ray]
    arma::vec eoa;                  // Elevation of arrival in [rad], GCS, [n_ray]
    arma::cube M;                   // Polarization transfer matrices incl. initial phases, row order of get_channels_*, [8, n_ray, n_freq]
    arma::uvec path_type;           // Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_ray]
    arma::uvec tap;                 // Tap index, rays with equal index form one tap, [n_ray]
    arma::vec gr_eps_re;            // Ground permittivity, real part, [n_freq]; empty without GR ray
    arma::vec gr_eps_im;            // Ground permittivity, imaginary part, [n_freq]; empty without GR ray
    arma::mat fbs_pos;              // First-bounce scatterer positions in [m], [3, n_ray]; empty until ssf_scatterers
    arma::mat lbs_pos;              // Last-bounce scatterer positions in [m], [3, n_ray]; empty until ssf_scatterers
    arma::uvec fallback;            // 1 = lengthened by the solver fallback, [n_ray]; empty until ssf_scatterers
    gbsm_rays() {};                 // Default constructor

    arma::uword n_ray() const;    // Number of rays, length of 'delay'
    arma::uword n_freq() const;   // Number of frequencies, length of 'center_frequency'
    bool empty() const;           // true if the object holds no rays
    std::string is_valid() const; // Empty string if the object is valid, else an error message
};
```

Attributes (doc table; "set by" is the producer):

| Attribute | Size | Meaning | Set by |
|---|---|---|---|
| `method` | — | "38.901", "38.901_MF" or "quadriga": GR power law, `mobility = "auto"` | `ssf_rays`, `gbsm_rays_from_paths` |
| `tx_pos`, `rx_pos` | `[3]` | positions the ray data refer to: the drop, or the CIR for data from paths | both |
| `center_frequency` | `[n_freq]` | Hz; TR methods 1 entry per object | both |
| `delay` | `[n_ray]` | s, relative to the LOS delay at `tx_pos`/`rx_pos` | both |
| `power` | `[n_ray, n_freq]` | coefficient powers without link gain; data from paths: the full path gain | both |
| `aod`, `eod`, `aoa`, `eoa` | `[n_ray]` | rad, GCS | both |
| `M` | `[8, n_ray, n_freq]` | Jones rows of `get_channels_*`, incl. initial phases | both |
| `path_type` | `[n_ray]` | 0 LOS, 1 GR, 2 NLOS | both |
| `tap` | `[n_ray]` | tap index (5.1) | both |
| `gr_eps_re`, `gr_eps_im` | `[n_freq]` or empty | ground permittivity `gr_eps_re - j·gr_eps_im`; required with a GR ray | both |
| `fbs_pos`, `lbs_pos` | `[3, n_ray]` or empty | scatterers at `tx_pos`/`rx_pos` | `ssf_scatterers`, `gbsm_rays_from_paths` |
| `fallback` | `[n_ray]` or empty | 1 = lengthened by the solver fallback | `ssf_scatterers` |

Members:

| Member | Returns |
|---|---|
| `n_ray()` | number of rays (length of `delay`) |
| `n_freq()` | number of frequencies (length of `center_frequency`) |
| `empty()` | `true` if the object holds no rays |
| `is_valid()` | empty string, or the first error: sizes of all attributes against `n_ray` and `n_freq`, `method`, path type codes, permittivity with a GR ray, scatterers and fallback either empty or complete |

- No cluster delay and no τ_Δ attribute (P1). Data are plain; the class does not normalise or reorder.

### 5.6 `ssf_rays` (new; `src/ssf_rays.cpp`)

```
// Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)
// - One geometry per call: "38.901" and "38.901_MF" compute frequency 'i_freq' (with "ray" set i_freq); "quadriga"
//   all frequencies; all other inputs as ssf_clusters and lsp_generate return them
// - A pure function of the clusters, the LSPs and the random values: no seed, batch-independent
void ssf_rays(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (NumClusters, NumSubPaths, PerCluster*, GR_enabled, PolVar_sigma)
              const arma::mat &tx_pos,                                // Tx positions in [m], [1, 3] or [n_link, 3]
              const arma::mat &rx_pos,                                // Rx positions in [m], [n_link, 3]
              const arma::vec &center_frequency,                      // Carrier frequencies in [Hz], [n_freq]
              const arma::cube &lsp,                                  // LSPs from lsp_generate, [n_link, n_freq, 8]
              const arma::cube &lsp_mu,                               // Log-domain means from lsp_generate, [n_link, n_freq, 8]
              const arma::cube &lsp_sigma,                            // Log-domain std from lsp_generate, [n_link, n_freq, 8]
              const arma::cube &delay,                                // Cluster delays from ssf_clusters, [n_link, n_slot, n_freq] ("38.901_MF") or [n_link, n_slot, 1]
              const arma::cube &power,                                // Cluster powers from ssf_clusters, [n_link, n_slot, n_freq]
              const arma::cube &aod,                                  // Cluster AOD from ssf_clusters, size as 'delay'
              const arma::cube &eod,                                  // Cluster EOD from ssf_clusters, size as 'delay'
              const arma::cube &aoa,                                  // Cluster AOA from ssf_clusters, size as 'delay'
              const arma::cube &eoa,                                  // Cluster EOA from ssf_clusters, size as 'delay'
              const arma::mat &rnd_ray,                               // "ray" set from gbsm_draw: quadriga set -1, TR methods set i_freq, [n_link, n_gen]
              std::vector<gbsm_rays> &rays,                           // Output: Ray data, n_link entries; fbs_pos, lbs_pos, fallback empty
              const std::string &method = "38.901",                   // Method of ssf_clusters: "38.901", "38.901_MF" or "quadriga"
              const std::string &angle_model = "additive",            // "additive" (TR, zenith fold) or "rotation" (3-D rotation of the offsets)
              const std::string &pol_model = "38.901",                // "38.901" (kappa, four phases) or "quadriga" (rotation model)
              const arma::mat *gr_eps_re = nullptr,                   // Optional: Ground permittivity, real part, from lsp_generate, [n_link, n_freq]; required if GR_enabled
              const arma::mat *gr_eps_im = nullptr,                   // Optional: Ground permittivity, imaginary part, from lsp_generate, [n_link, n_freq]; required if GR_enabled
              bool pol_variability = false,                           // TR 7.6.16 polarisation power variability; pol_model "38.901" only
              bool random_phase = true,                               // false: all initial phases 0
              arma::uword i_freq = 0);                                // TR methods: frequency index into center_frequency, 0-based; quadriga: must be 0
```

- `rays` gets n_link entries (resized only if the count differs); entries whose attributes already have the right sizes are written in place. Entry l holds `method`, row l of the positions and of `gr_eps_re/_im` (empty without GR), `center_frequency`, and `delay`, `power`, angles, `M`, `path_type`, `tap`; `fbs_pos`, `lbs_pos`, `fallback` are set empty.
- Ray layout, fixed per scenario: index 0 LOS ray, 1 GR ray (G = 1), then ray m of NLOS cluster n at 1 + G + n·M + m; clusters at power 0 keep their M rays at power 0 (padding). Ray powers P_n/M per frequency; LOS and GR rays carry the slot powers and the Jones matrices of the choices in section 3.
- TR methods: frequency `i_freq` only (the caller loops over the frequencies of `38.901_MF` with the "ray" set of each; `38.901` is single-frequency); the entry's `center_frequency` holds that one frequency; steps 8–11 from the "ray" set of `i_freq`, sub-clusters of P4 with their own delays and taps. `quadriga`: any `n_freq`, one geometry and one set of phases for all frequencies, ray delays from block 14 (P21), tap = cluster; with `pol_variability`, one η per ray for all frequencies.
- Validated: sizes against `scenpar` (`n_slot`; `n_gen` of `rnd_ray` = 14·N·M + N for `quadriga`, 13·N·M + N for the TR methods) and against each other; option values; `pol_variability` with `pol_model = "quadriga"` throws (P1); `i_freq` < `n_freq`, `i_freq` = 0 for `quadriga`; geometry inputs with n_freq slices only for `38.901_MF`; `38.901` with `n_freq` > 1 throws (series 1); permittivity missing with `GR_enabled` throws. As `ssf_clusters`: only sizes show that the inputs belong together.
- Swap reciprocity: with mirrored inputs (dual mobility in `gbsm_draw`, mirrored LSPs and clusters), swapping `tx_pos` and `rx_pos` exchanges departure and arrival per ray and transposes M (both polarisation models, P19).

### 5.7 `ssf_scatterers` (new; `src/ssf_scatterers.cpp`)

```
// First- and last-bounce scatterers of the rays of one link (two-bounce solver)
// - NLOS rays: exact departure and arrival angles where a solution exists, equal distances from both ends;
//   otherwise the shortest polyline along the ray's own directions with both scatterers at least d_min away
// - LOS ray on the direct line, GR ray at the specular point on z = 0, rays with zero power at all frequencies at
//   the midpoint of the direct line
void ssf_scatterers(gbsm_rays &rays,     // In/out: ray data; writes fbs_pos, lbs_pos and fallback
                    double d_min = 2.0); // Minimum distance of the scatterers to both ends in [m], > 0
```

- Reads `tx_pos`, `rx_pos`, `delay`, `power`, the angles and `path_type`; writes `fbs_pos`, `lbs_pos` `[3, n_ray]` and `fallback` `[n_ray]`; nothing else changes (`delay` keeps the drop value; a fallback ray's polyline is longer).
- `d_min` ≤ 0 throws; `is_valid()` first. Per link (no batch, no OpenMP region): batch callers loop over the entries of `ssf_rays`. Cost ≈ 12.5 ns per ray (provisional).

### 5.8 `ssf_paths` (new; `src/ssf_paths.cpp`)

```
// Antenna-free paths of one link at one position
// - Outputs are the inputs of get_channels_planar ("phase") or get_channels_spherical / get_channels_multifreq
//   ("drift") and of qrt_paths_from_scatterers
// - Rays with zero power at all frequencies are dropped, LOS and GR rays are kept
// - Validity: "phase" up to about sqrt(lambda * d3D / 8), "drift" up to about 0.2 * b_med per moving end
void ssf_paths(const gbsm_rays &rays,                // Ray data of the link
               const arma::vec &link_gain_dB,        // Link gain -PL - O2I + SF at the position in [dB], [n_freq]; 0 for data from paths
               arma::mat &path_gain,                 // Output: Path gains (linear), [n_path, n_freq]
               arma::vec &path_length,               // Output: Path lengths in [m], [n_path]
               arma::cube &M,                        // Output: Polarization transfer matrices, [8, n_path, n_freq]
               const std::string &mobility = "auto", // "phase", "drift" or "auto" (phase for the TR methods, drift for quadriga)
               const arma::vec *tx_pos = nullptr,    // Optional: Tx position in [m], [3]; nullptr = rays.tx_pos
               const arma::vec *rx_pos = nullptr,    // Optional: Rx position in [m], [3]; nullptr = rays.rx_pos
               const arma::vec *kf_ratio = nullptr,  // Optional: K(position) / K(drop), [n_freq]; nullptr = 1
               arma::vec *aod = nullptr,             // Optional output: Azimuth of departure in [rad], [n_path]
               arma::vec *eod = nullptr,             // Optional output: Elevation of departure in [rad], [n_path]
               arma::vec *aoa = nullptr,             // Optional output: Azimuth of arrival in [rad], [n_path]
               arma::vec *eoa = nullptr,             // Optional output: Elevation of arrival in [rad], [n_path]
               arma::mat *fbs_pos = nullptr,         // Optional output: First-bounce scatterer positions in [m], [3, n_path]; "drift" only
               arma::mat *lbs_pos = nullptr,         // Optional output: Last-bounce scatterer positions in [m], [3, n_path]; "drift" only
               arma::uvec *path_type = nullptr,      // Optional output: Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_path]
               arma::uvec *ray_ind = nullptr,        // Optional output: Index of the ray in 'rays', 0-based, [n_path]
               bool oxygen = false);                 // TR 7.6.1 oxygen absorption from each path's own length
```

- Paths in ray order; `ray_ind` maps each path to its ray (`rays.tap(ray_ind)`, `rays.fallback(ray_ind)`). For one frequency f, `path_gain.col(f)` and `M.slice(f)` are the `[n_path]` and `[8, n_path]` inputs of `get_channels_planar` / `get_channels_spherical`; all outputs as they are feed `get_channels_multifreq` and `qrt_paths_from_scatterers`.
- Path gain: `kf_ratio` rescales LOS/GR against NLOS on the coefficient powers (P8), then `link_gain_dB`, then, with `oxygen`, the loss of 7.6.1 from `path_length` (τ_Δ not added, P1).
- `phase`: lengths |rx_pos − tx_pos| of the ray data + c·delay and the angles of the ray data; the phase of the displacement (P7) is folded into M per frequency; use with the arrays at `rays.tx_pos` / `rays.rx_pos`. `fbs_pos` / `lbs_pos` requested in `phase` throw.
- `drift`: needs `rays.fbs_pos/lbs_pos` (else throws, naming `ssf_scatterers`); scatterers fixed, ends at the given positions, polyline lengths and angles from the geometry, LOS recomputed, GR point, R(t)/R0 and the GR power law of `rays.method` recomputed.
- `mobility = "auto"` resolves from `rays.method`; `link_gain_dB` and `kf_ratio` have `n_freq` entries; `is_valid()` first. The docs state both validity limits (finding 17).

### 5.9 `qrt_paths_from_scatterers` (new; `include/quadriga_channel.hpp`, "Channel functions", `src/qd_channel_qrt.cpp`)

```
// Path objects of one CIR from spherical path data, for qrt_file_append
// - The path gain and the inverse free-space loss of the polyline length are folded into the stored Jones matrices
// - LOS: 0 interactions; GR: 1 (reflection); NLOS: 2 (scattering)
// - Use coordinates relative to a local origin near the Tx (QRT stores float32)
std::vector<path> qrt_paths_from_scatterers(const arma::vec &tx_pos,           // Tx (origin) position in [m], [3]
                                            const arma::vec &rx_pos,           // Rx position of the CIR in [m], [3]
                                            const arma::mat &fbs_pos,          // First-bounce scatterer positions in [m], [3, n_path]
                                            const arma::mat &lbs_pos,          // Last-bounce scatterer positions in [m], [3, n_path]
                                            const arma::mat &path_gain,        // Path gains (linear), [n_path, n_freq]
                                            const arma::cube &M,               // Polarization transfer matrices, [8, n_path, n_freq]
                                            const arma::uvec &path_type,       // Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_path]
                                            const arma::vec &center_frequency, // Frequencies in [Hz], [n_freq], 1 to 127 entries
                                            unsigned i_cir = 0);               // CIR index, stored in path::iC
```

- Returns n_path objects with `iC = i_cir`; the caller appends them to the `path_data` of `qrt_file_append` (`qrt_file_init` takes GHz, documented). Double inputs; the objects are float32 by construction.
- GR: one interaction at `lbs_pos` with a reflection code; NLOS: interactions at `fbs_pos` and `lbs_pos` with the scattering code (defined in the spec, section 9). `qrt_file_read` returns the same gain·M per path within float32 (finding 2) and the polyline as path length.
- Validation: sizes, path type codes, 1 to 127 positive frequencies. Writer memory: `qrt_file_append` takes all paths of one origin at once (about 3–4 GB for 1000 UTs × 100 snapshots × 381 rays at one frequency), documented.

### 5.10 `gbsm_rays_from_paths` (new; `include/quadriga_gbsm.hpp`, GBSM section, `src/gbsm_rays_from_paths.cpp`)

```
// Ray data of one link from spherical path data (e.g. read with qrt_file_read)
// - Delays from the path lengths, angles from the scatterer positions, power = path gain
// - Fills a pre-initialised object in place
void gbsm_rays_from_paths(const arma::vec &tx_pos,                // Tx position in [m], [3]
                          const arma::vec &rx_pos,                // Rx position in [m], [3]
                          const arma::mat &fbs_pos,               // First-bounce scatterer positions in [m], [3, n_path]
                          const arma::mat &lbs_pos,               // Last-bounce scatterer positions in [m], [3, n_path]
                          const arma::mat &path_gain,             // Path gains (linear), [n_path, n_freq]
                          const arma::vec &path_length,           // Path lengths in [m], [n_path]
                          const arma::cube &M,                    // Polarization transfer matrices, [8, n_path, n_freq]
                          const arma::vec &center_frequency,      // Frequencies in [Hz], [n_freq]
                          gbsm_rays &rays,                        // Output: Ray data, n_ray = n_path
                          const std::string &method = "quadriga", // Value of rays.method: GR power law and default mobility
                          const arma::uvec *path_type = nullptr,  // Optional: Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_path]; nullptr = LOS where FBS = LBS on the direct line, else NLOS
                          const arma::uvec *tap = nullptr,        // Optional: Tap index, [n_path]; nullptr = one tap per path
                          const arma::vec *gr_eps_re = nullptr,   // Optional: Ground permittivity, real part, [n_freq]; required with a GR path
                          const arma::vec *gr_eps_im = nullptr);  // Optional: Ground permittivity, imaginary part, [n_freq]; required with a GR path
```

- Inputs carry the output names of `qrt_file_read`. `delay` = (`path_length` − |rx_pos − tx_pos|)/c; departure angles towards `fbs_pos`, arrival angles towards `lbs_pos` (LOS: the direct line); `power` = `path_gain`; `M`, `fbs_pos`, `lbs_pos` copied; `fallback` empty.
- `tap = nullptr`: tap index = path index. QRT stores neither the tap index nor the GR type: a GR path read from a file needs `path_type` (from `no_int` / `interact_type` of `qrt_file_read`).
- Used with `link_gain_dB = 0` in `ssf_paths` / `gbsm_single_link_channel`; `drift` keeps the path gain of the given positions, so distance changes of the gain come through `link_gain_dB`.

### 5.11 `gbsm_single_link_channel` (new; `include/quadriga_lib.hpp`, GBSM section, `src/gbsm_single_link_channel.cpp`)

```
// Channel coefficients of one link at one frequency over one segment (geometry-based stochastic model)
// - Per snapshot get_channels_planar ("phase", arrays at the positions of 'rays', computed once per distinct
//   orientation) or get_channels_spherical ("drift") with the paths of ssf_paths
// - "taps": rays summed per tap, LOS and GR taps always present, tap count fixed per call; "rays": one path per ray
// - Validity: see ssf_paths; the channel's 'par' reports the largest displacement relative to the limit
channel<double> gbsm_single_link_channel(const arrayant<double> &tx_array,          // Tx array antenna, n_tx ports
                                         const arrayant<double> &rx_array,          // Rx array antenna, n_rx ports
                                         const gbsm_rays &rays,                     // Ray data of the link
                                         const arma::vec &link_gain_dB,             // Link gain -PL - O2I + SF at rays.center_frequency(i_freq) in [dB], [n_snap] or [1]
                                         const std::string &mobility = "auto",      // "phase", "drift" or "auto" (phase for the TR methods, drift for quadriga)
                                         const std::string &output = "taps",        // "taps" or "rays"
                                         arma::uword i_freq = 0,                    // Frequency index in rays.center_frequency, 0-based
                                         const arma::mat *tx_pos = nullptr,         // Optional: Tx positions in [m], [n_snap, 3] or [1, 3]; nullptr = rays.tx_pos
                                         const arma::mat *rx_pos = nullptr,         // Optional: Rx positions in [m], [n_snap, 3] or [1, 3]; nullptr = rays.rx_pos
                                         const arma::mat *tx_orientation = nullptr, // Optional: Tx Euler angles (bank, tilt, heading) in [rad], [n_snap, 3] or [1, 3]; nullptr = 0
                                         const arma::mat *rx_orientation = nullptr, // Optional: Rx Euler angles (bank, tilt, heading) in [rad], [n_snap, 3] or [1, 3]; nullptr = 0
                                         const arma::vec *kf_ratio = nullptr,       // Optional: K(snapshot) / K(drop) at rays.center_frequency(i_freq), [n_snap] or [1]; nullptr = 1
                                         bool oxygen = false,                       // TR 7.6.1 oxygen absorption from each path's own length
                                         bool use_absolute_delays = false);         // Include the LOS delay in all delays, as get_channels_*
```

- `n_snap` is the row count of the per-snapshot inputs with more than one row; they must agree; one-row inputs apply to all snapshots.
- Output: `center_frequency` = `rays.center_frequency(i_freq)`; `tx_pos`, `rx_pos` (absolute), `tx_orientation`, `rx_orientation` `[3, n_snap]`; `coeff_re/_im` n_snap × `[n_rx, n_tx, n_tap]` (`taps`) or `[n_rx, n_tx, n_path]` (`rays`); `delay` `[1, 1, n_tap]` for `phase` + `taps`, else `[n_rx, n_tx, ·]`; `rays` also fills `path_gain`, `path_length`, `path_polarization`, `path_angles` and (`drift`) `path_fbs_pos`, `path_lbs_pos`; `initial_position` stays 0.
- `taps`: the kept rays (power > 0 at `i_freq`, plus LOS and GR) summed per tap with the mean delay; taps in ascending `tap`, count fixed per call; LOS and GR taps present whenever the rays hold them (always from `ssf_rays`; zero power in NLOS).
- `par` (names fixed in the spec): `tap` (index per tap or per path), rays per tap, FBS/LBS of the rays (`drift`), GR point per snapshot (`drift` with a GR ray), `link_gain_dB` per snapshot, b_med per end (when the rays have scatterers) and the largest displacement of each end relative to the validity limit of the active mode (P7, P15).
- Validation: `is_valid()`, `i_freq` < `n_freq`, option values, sizes; `drift` without scatterers throws. No power correction, no initial-position input.

### 5.12 Names, producer → consumer

| Producer: name, size | Consumer: name | Note |
|---|---|---|
| `gbsm_draw` "lsf": `rnd` `[n_link, 11]` | `lsp_generate` `rnd`; `path_loss` `rnd_he` (column 10) | — |
| `gbsm_draw` "cluster" −1 / k: `rnd` | `ssf_clusters` `rnd` / `rnd_freq` slice k | as series 1 |
| `gbsm_draw` "ray" −1 / k: `rnd` | `ssf_rays` `rnd_ray` | named by content, as `rnd_freq` |
| `lsp_generate`: `lsp`, `lsp_mu`, `lsp_sigma` `[n_link, n_freq, 8]`, `gr_eps_re/_im` `[n_link, n_freq]` | `ssf_rays` same names (TR methods: frequency k); `ssf_clusters` `gr_eps_re/_im`; `gbsm_rays.gr_eps_re/_im` `[n_freq]` (row l) | — |
| `ssf_clusters`: `delay`, `aod`, `eod`, `aoa`, `eoa` `[n_link, n_slot, ·]`, `power` `[n_link, n_slot, n_freq]` | `ssf_rays` same names: one slice; `power` all frequencies (`quadriga`) or frequency k | — |
| `path_loss` `pl`, `o2i_loss` `loss` [dB], `lsp` slice 2 SF (linear) | `ssf_paths`, `gbsm_single_link_channel` `link_gain_dB` = −pl − loss + 10·lg SF | sum by the caller, stated once in the docs |
| `lsp` slice 1 KF | `kf_ratio` = K(position)/K(drop) | ratio by the caller |
| `ssf_rays` `rays[l]` | `ssf_scatterers`, `ssf_paths`, `gbsm_single_link_channel` `rays` | class attributes |
| all GBSM functions, `gbsm_rays` `center_frequency` [Hz] | `get_channels_*` `center_frequency` (one value); `qrt_file_read` `center_frequency` [Hz]; `qrt_file_init` `freq` [GHz] | existing QRT unit unchanged |
| `ssf_paths`: `path_gain`, `path_length`, `M`, `aod` … `eoa`, `fbs_pos`, `lbs_pos`, `path_type` | `get_channels_planar/_spherical` (column f, slice f), `get_channels_multifreq`, `qrt_paths_from_scatterers` same names | — |
| `qrt_file_read`: `center_frequency`, `tx_pos`, `rx_pos`, `fbs_pos`, `lbs_pos`, `path_gain`, `path_length`, `M` | `gbsm_rays_from_paths` same names | — |
| `gbsm_rays.tx_pos` `[3]`; batch `tx_pos` `[1, 3]`; `gbsm_single_link_channel` `tx_pos` `[n_snap, 3]` | `channel.tx_pos` `[3, n_snap]` | same name, row vs column (existing class), documented |

### 5.13 Data-flow sketch

```
gbsm_init_phases(set_type) → gbsm_draw(set_type): rnd
   "lsf"                       → lsp_generate: rnd → lsp, lsp_mu, lsp_sigma, abs_toa, gr_eps_re/_im, eod_offset_deg
                               → path_loss: rnd_he (column 10) → pl, sf_sigma (→ lsp_generate)
   "cluster", -1 / k           → ssf_clusters: rnd / rnd_freq
   "ray", -1 (+ center_frequency) / k → ssf_rays: rnd_ray
o2i_loss: loss;  los_probability: p_los
ssf_clusters: delay, power, aod, eod, aoa, eoa → ssf_rays (same names) → rays (n_link × gbsm_rays)
   rays[l] → ssf_scatterers (in place: fbs_pos, lbs_pos, fallback)                        [drift]
   rays[l] + link_gain_dB (= -pl - loss + 10 lg SF) + kf_ratio
        → gbsm_single_link_channel → channel<double>
        → ssf_paths → path_gain, path_length, M, aod..eoa, fbs_pos, lbs_pos, path_type, ray_ind
              → get_channels_planar / get_channels_spherical (column f, slice f)
              → get_channels_multifreq (as they are)
              → qrt_paths_from_scatterers → path_data → qrt_file_append
qrt_file_read: center_frequency, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, path_length, M
        → gbsm_rays_from_paths → rays → ssf_paths / gbsm_single_link_channel (link_gain_dB = 0)
```

## 6. Data flow

Per BS site and scenario (one seed for all sets): `gbsm_init_phases` once per set ("lsf"; "cluster" −1 and k; "ray" −1 for `quadriga`, k for the TR methods), `gbsm_draw` at the links, then `o2i_loss`, `los_probability`, `path_loss`, `lsp_generate`, `ssf_clusters`, and `ssf_rays` (one call for `quadriga`, one per frequency for the TR methods); for `drift`, `ssf_scatterers` on each entry. Per link and segment: `gbsm_single_link_channel` (inside: `ssf_paths`, then `get_channels_planar` for `phase` or `get_channels_spherical` for `drift`), with `link_gain_dB` and `kf_ratio` from the large-scale functions at the drop or at the snapshots. QRT route: `ssf_paths` (`drift`) at the CIR positions of a segment → `qrt_paths_from_scatterers` → `qrt_file_append`; later `qrt_file_read` → `gbsm_rays_from_paths` → `gbsm_single_link_channel` (`rays`), or `get_channels_*` directly, with any antennas gives the same channel within the tolerance of P12. Other path sources (NG's extra clusters, CDL, user scatterers) enter as a `gbsm_rays` with their own path type and tap index (P20).

## 7. Readiness checks (side quests)

Each check is a short paragraph in the spec that shows the contract allows the later addition without changing existing outputs or random-field columns; the spec review verifies it.

### 7.1 Multi-frequency paths (`get_channels_multifreq`)

Needed: one geometry, gain `[n_path, n_freq_in]` and M `[8, n_path, n_freq_in]` at sample frequencies; the function interpolates gain linearly and M by SLERP. Available after this series in `quadriga` (P5): `ssf_rays` with all frequencies gives `power [n_ray, n_freq]` and `M [8, n_ray, n_freq]` on one geometry (shared coupling, ray delays, positions and phases; per-frequency powers, K, SF, PL and XPR from the same normalised random values), and `ssf_paths` returns exactly the `fbs_pos`, `lbs_pos`, `path_gain`, `path_length`, `M` of `get_channels_multifreq` and of a multi-frequency QRT file (≤ 127 frequencies). Gap for a later session: a "sample grid" mode (many closely spaced frequencies, cost of per-frequency evaluation; dense grid near the 60 GHz oxygen line), checks that powers and XPR vary smoothly between samples, and a validation of the frequency correlation (the per-tap behaviour is settled by P21, finding 12). TR methods do not qualify (7.6.5 per-frequency steps; `38.901_MF` geometry per frequency). Feasible; the main limit is physical (one scatterer set over a very wide band), which `quadriga` already assumes.

### 7.2 Procedure A (TR 7.6.3.2)

Needs: absolute cluster delays and angles at the segment start (`ssf_clusters` + d3D), velocities per step (caller), the sign X_n per cluster (block 9 of the shared "cluster" set, reserved, F2), re-derivation of rays from updated clusters (`ssf_rays` is a pure function of the cluster arrays and the "ray" set values, which stay those of the drop: coupling ranks and phases kept), cluster powers from the updated delays (a step-6 function with the stored Z_n), ray data per segment of ≤ 1 m (one `gbsm_rays` per segment from `ssf_rays`, one `gbsm_single_link_channel` call per segment), the 7.6-10a delay normalisation over the whole mobility range (a later function over the step sequence). No contract change expected; to document then: the two strongest clusters, and with them the tap indices, can change between steps.

### 7.3 7.6.2.2 (large bandwidth) and the mobile-scatterer term of 7.6.10

7.6.2.2 for the TR methods: M of 7.6-8 is ≥ 20 and depends on the bandwidth B, the array sizes D_h, D_v and λ (so per frequency); it enters as appended optional arguments of `ssf_rays` (bandwidth, array size, M_max), not as a scenario key (P11); `NumSubPaths` becomes M_max (100–1000 scales the N·M blocks and the ray layout); per-frequency ray counts through zero-power padding (dropped by `ssf_paths`); the sub-cluster mapping is replaced (tap = cluster, i = 0, as `quadriga` already has with P21); the ray delays use the reserved block 14 of the per-frequency "ray" sets (cluster decorrelation distance `SC_lambda`), random offsets and unequal ray powers appended blocks; `gbsm_rays.power` is already per ray. A cost, not a contract change: M = 1000, N = 19 gives ≈ 2.5·10⁵ generators per set (≈ 1 GB of phases, ≈ 8 ms of draws per link and frequency). 7.6.10 (7.6-46): adds a Doppler 2·α·D/λ per ray that needs time: appended `time` arguments of `ssf_paths` (one value) and `gbsm_single_link_channel` (`[n_snap]`), an appended per-ray attribute of `gbsm_rays`, α and D as appended "ray" blocks.

### 7.4 Library-only TR 7.8.2 calibration

Needs: many sites and sectors in one batch (per-link `tx_pos` is supported), rays shared by co-sited sectors (TR: steps 1–9 identical; the rays are antenna-free, sectors differ only in the `gbsm_single_link_channel` call), per-site seeds (series 1), LOS and NLOS links of a layout handled by separate calls per scenario (W1: LOS is a scenario property; batch-independent coupling, P6) and joined by the caller, TR antenna models (`arrayant_generate`), and `gbsm_single_link_channel` as the per-link channel generator (one snapshot per drop, library defaults = TR). Nothing in this series may require QuaDRiGa-NG; the R0 notes for P3 and P16 go to the calibration session.

## 8. Process (steps 2–7)

A manager session tracks plan, spec and code and starts agents; the maintainer talks only to the manager. Working folders: repository `/home/stephan/quadriga-lib-ssf` (branch `ssf`), internals (briefs, reports, R0 data, review reports) in `/home/stephan/quadriga-lib-agents/ssf2/`. Coding rules: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`; produced code cites only published material and the SPEC files.

1. **Plan review (step 2):** round 1 done (two independent reviews with prototypes; v3), R0 study (v4), contract review with the maintainer (v5). Next: a full review round (physics and TR, NG fit, API design) on v5 before the spec.
2. **R0 study (done):** literature search (about 60 sources) and NumPy prototypes on library clusters, against a point-scatterer reference (each ray a point scatterer whose distance matches its delay, intra-cluster length spread c·c_DS, exact spherical field; it reproduces Clarke's narrowband decorrelation and the Fourier relation for the cross-band coherence), plus S6 on the `drift` validity range. Results: findings 5, 8–17; decisions P13, P15, P19, P21; confirmations of P3, P15, P16, P17, P18. Run times were measured under load and stay provisional.
3. **Spec (step 3), spec review (step 4):** one mandatory round, up to three if major issues are found; each reviewer checks the TR transcription numerically. Doc rules in the spec: API only, each fact once, no formulas, size targets set against `ssf_clusters`; doc blocks follow the block grammar and writing pitfalls of `docs_builder/README.md`. The spec states the validity limits and model properties of findings 10–13 and 17, P7, P15, and the `taps` bandwidth limit of P17, for the docs.
4. **Implementation (step 5)**, test-first, in runs: R1 the changes to existing functions (4.7: merge into `gbsm_init_phases` / `gbsm_draw` with "lsf" and "cluster" bit for bit, renames, `SubpathMethod`, zero-copy in the existing wrappers, `ssf_clusters` doc sizes) with all library callers, wrappers and tests; R2 one preparatory change (all new declarations incl. the `gbsm_rays` class, doc stubs and empty test files, so that later runs do not edit shared headers); R3 the "ray" set type; R4 `gbsm_rays` members and `ssf_rays` TR methods; R5 `ssf_rays` QuaDRiGa models, 7.6.16; R6 `ssf_scatterers`; R7 `ssf_paths` (needs R4/R5 for the tap index and R6 for the fallback); R8 `qrt_paths_from_scatterers` and `gbsm_rays_from_paths`; R9 `gbsm_single_link_channel` (QRT round-trip test against the direct route; needs R7 and R8); R10 wrappers of the new functions and documentation. R6 and R8 may run in parallel with R4/R5 in separate worktrees (R8 tests write fixed-name `.qrt` files); `make docs` once per merge by the manager. Shared test headers and method-generic test cases are specified up front (lesson: test sizes exceeded 1.5–2×).
5. **Code review (step 6):** up to three rounds, one mandatory, plus a security review (inputs, sizes, the QRT route).
6. **Cleanup and merge (step 7):** working documents deleted, `md/ssf_notes.md` updated, the maintainer commits.
7. **NG integration (step 8):** separate project; retires the code of 4.8; change list includes the merged random-set functions and the removed `SubpathMethod` argument, the removal of `p_correct`, the displacement-vector Doppler, the rotation model's swap-symmetric construction, the TR ray mapping of the sub-clusters.

## 9. Open items for the review round

- η of 7.6.16 in `quadriga` with the TR polarisation model is shared across frequencies (P5, A6): physics check.
- The clip rule of the reciprocal rotation model where no α exists (P19).
- The interaction code for scattering in the `path` doc (`qrt_paths_from_scatterers`).
- The rescale rule for M > 20 (P11).
- c_DS of UMa below 6 GHz: note 6 of Table 7.5-6 needs `PerClusterDS_fc_min`, which the v19 files do not set (4.71 vs 3.91 ns at 3.5 GHz).
- λ_τ depends on the frequency set through the frequency-averaged c_DS (P21): the same link drawn with another frequency list gets other ray delays; document it or fix the averaging.

## 10. Terms

Defined once in the GBSM section description (P26).

| Term | Meaning | Origin |
|---|---|---|
| scenario | type of environment with its statistics: numeric keys `scenpar`, path-loss model, short names (one conf file) | TR |
| link | one Tx–Rx pair of a batch (row l of `rx_pos`) | TR |
| drop | one realisation of LSPs, clusters and rays at the link positions; the positions the ray data refer to | TR |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 7.5 steps 5–7) | TR |
| slot | index in the `ssf_clusters` outputs: 0 LOS, 1 GR (if enabled), then the N clusters | library |
| sub-cluster | one of the three parts of a split cluster (TR 7.5 step 11, Table 7.5-5) | TR |
| ray | one of the M rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) | TR |
| path | a propagation path as `get_channels_*` and QRT take it; `ssf_paths` makes one per kept ray (QuaDRiGa also says path for a cluster) | library |
| tap | rays summed in the `taps` output: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`quadriga`); index `tap` | library |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed | QuaDRiGa |
| segment | the snapshots of one drop over which the ray data are kept (`phase` or `drift`) | QuaDRiGa |
