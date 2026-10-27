# TR 38.901 v19.5.0 compliance review of the large-scale model, round 3 (final)

Reviewer: independent, read-only, working from the TR (`/home/stephan/sjc/references/3gpp_38901/38901-v19.5.0.md`) and
the code only. The documents in `md/` were read only at the end, to classify deviations as documented or not.

Scope: `include/quadriga_gbsm.hpp`, `src/qd_gbsm_phases.cpp`, `src/lsp_generate.cpp`, `src/path_loss.cpp`,
`src/los_probability.cpp`, `src/o2i_loss.cpp`, `src/quadriga_gbsm_helpers.hpp`, `src/qd_scenpar.cpp` (key table and
correlation-key mapping), the 23 files `data/config/3GPP_38.901_v19_*.conf`. Branch `fixes` at b10f332.

## 1. Summary

| Severity | Count |
| --- | --- |
| blocker | 0 |
| major | 0 |
| minor | 2 |
| nit | 4 |

Verdict: **compliant**. Path loss, SF std, LOS probability, O2I loss, the LSP means and stds incl. the fc clamps, the
ESD terms, the EoD (ZOD) offsets, the correlation matrices, the Cholesky order, the spread clipping, 2D correlation,
floors, multi-frequency, the absolute TOA with its 2L/c bound and the soft LOS state all reproduce TR 38.901 v19.5.0.
The only numerical deviations from the TR formulas are the documented ones: the 3D breakpoint of `dual_slope` for RMa
and SMa LOS (max 0.0341 dB, measured), the rounded constants (`PL_E = 13.33`: max 0.0021 dB; RMa NLOS `Bn`:
0.005 dB), the geometric RMa EoD offset and the Comb256 ACF. The two minor findings are an undocumented gap
(absolute TOA missing in the four LOS_O2I files) and an undocumented non-TR ground-permittivity model.

## 2. Findings

| ID | Sev. | TR clause | File:line | Finding (with numbers) | Proposed fix |
| --- | --- | --- | --- | --- | --- |
| T3-01 | minor [escalate] | 7.6.9, Table 7.6.9-1; Table 7.5-6 (O2I: K = N/A) | `data/config/3GPP_38.901_v19_{UMa,UMi,RMa,SMa}_LOS_O2I.conf` (no absTOA block; cf. `UMa_NLOS_O2I.conf:164-166`); `src/lsp_generate.cpp:368` | The four LOS_O2I files have no `absTOA_*` keys, so `abs_toa = 0` for every O2I link whose outdoor part is LOS. An O2I link has no LOS ray in the TR (Table 7.5-6 O2I column: K-factor N/A; the files themselves set `KF_mu = -100`), so its impulse response is the NLOS form (7.6-43), which carries Δτ. The NLOS_O2I files do set Δτ (UMa -7.4/0.2/50 m). Median Δτ that is lost: UMa 39.8 ns, UMi 31.6 ns, RMa 4.7 ns, SMa 19.9 ns (≈ 12, 9.5, 1.4, 6 m of extra path). Not documented anywhere (the spec calls the TOA "NLOS files only" without discussing O2I). | Add the absTOA block of the matching NLOS file to the 4 LOS_O2I files (UMa -7.4/0.2/50, UMi -7.5/0.5/15, RMa -8.33/0.26/50, SMa -7.702/0.4/50), or state in their header why O2I-LOS links have Δτ = 0. User decision: interpretation of "NLOS" in 7.6.9 for O2I. |
| T3-02 | minor | 7.6.8, eq. (7.6-40) to (7.6-42), Table 7.6.8-1 | `src/lsp_generate.cpp:140-145` (doc), `:378-403` (code) | The ground permittivity is QuaDRiGa's dry/medium/wet fit (`3 + j0.003 f^1.34`, `30.4 f^-0.47 + j0.18 f^1.05`, `31.3 f^-0.48 + j0.63 f^0.77`), not Table 7.6.8-1. Very dry ground agrees (3.5 GHz: imag 0.016 vs TR 0.018); medium and wet do not: at 1 GHz medium εr 30.4 vs TR medium-dry 15.0, at 3.5 GHz medium εr 16.87 / imag 0.67 vs 13.23 / 1.39, wet imag 1.65 vs 3.93, at 10 GHz wet imag 3.71 vs 5.38. The doc cites TR eq. (7.6-40) for the sign convention, which suggests TR values; that the values are a QuaDRiGa extension is stated only in `md/gbsm_lsf_plan.md` (R1) and the spec. | Say in the `lsp_generate` doc that the three ground types are QuaDRiGa's fits (`gen_lsf_parameters`), not Table 7.6.8-1, and give the TR table for callers who need it (or offer it as an option later). |
| T3-03 | nit | Table 7.4.1-1 note 1 | `src/qd_gbsm_phases.cpp:22-23,74`; `src/path_loss.cpp:135-144`; `include/quadriga_gbsm.hpp:73` | The TR draws hE independently per link; the library draws it from a spatially consistent uniform field with `SF_lambda`, so nearby UTs of one site share hE. The marginal distribution is exact (probe: max 0.0020 dB against the TR draw for 7 hUT x 9 d2D x 41 u x 4 fc, all from `E = 13.33`). The mechanism and its side effect (steps along a track) are documented, the reason for spatial correlation and for `SF_lambda` only in the plan (D15, D19). | One sentence in the `path_loss` or `lsf_init_phases` doc: spatial consistency in the sense of 7.6.3, the TR gives no correlation distance, `SF_lambda` is used. |
| T3-04 | nit | 7.4.3.1 (d2D-in), Table 7.2-5 | `3GPP_38.901_v19_SMa_{LOS,NLOS}_O2I.conf:122-123`; `src/o2i_loss.cpp:46-48` | The SMa O2I files set `O2I_d2D_in_max = 25` (commercial buildings); the comment names 10 m for residential buildings, but the TR SMa layout is 90 % residential / 10 % commercial and `o2i_loss` takes one `d2d_in_max` per call. The default thus fits only the 10 % minority. | Add to the comment (or the `o2i_loss` doc) that the TR mix is modelled by splitting the UTs by building type (10 m / 25 m), e.g. two calls. |
| T3-05 | nit | Table 7.4.1-1 (RMa, SMa LOS σSF 4 / 6) | `3GPP_38.901_v19_RMa_LOS.conf:37`, `3GPP_38.901_v19_SMa_LOS.conf:37` | `SF_sigma = 5` is not a TR value; it applies whenever `lsp_generate` is called without the `sf_sigma` of `path_loss`. The comment points to `PL_sig1`/`PL_sig2` but does not say that `sf_sigma` must be passed for TR conformance. | Extend the comment: "TR conformance: pass sf_sigma of path_loss to lsp_generate; 5 dB is a legacy midpoint". |
| T3-06 | nit | 7.4.4 (eq. 7.4-5), 7.5 step 4 | `src/qd_gbsm_phases.cpp:327-347, 377-378` (lsf_init_phases doc), `include/quadriga_gbsm.hpp:79-80` | Both defaults are documented deviations, but the gbsm docs nowhere name the TR-literal settings: (a) Comb256 is not exponential; measured over 600 seeds: R(0.25λ) = 0.92 vs e^-0.25 = 0.78, R(0.5λ) = 0.74 vs 0.61, R(λ) = 0.39 vs 0.37, R(2λ) = 0.03 vs 0.14; (b) `remove_bias = true` reduces the LSP variance (by the `sos_remove_bias` rule of thumb ≈ 5 to 14 % for an InF/InH hall of 5 to 10 decorrelation distances); the TR has no such step. | One line at `preset`/`remove_bias`: "TR 7.4.4 literal: preset `Exp256`, `remove_bias = false`". |

No blocker or major findings.

## 3. Compliance matrix

| TR item | Status | Evidence |
| --- | --- | --- |
| 7.4.1 RMa LOS PL1/PL2, note 5 dBP, σSF 4/6 | deviation documented | 3D breakpoint in PL2 (`path_loss.cpp:113-115`, conf comment); probe max 0.0172 dB at the probe grid, dense scan max **0.0341 dB** (hBS 33, hUT 1, 0.5 GHz) ≤ the stated 0.034/0.035; constants h = 5: A1 20.4779→20.48, B 31.7408→31.74, D 0.001398, E 20.944→20.94; σSF exact |
| 7.4.1 RMa NLOS (max with LOS) | compliant | `nlos` + LOS term; Dn = -24.37 + 9.3 = -15.07, D2n = 3.7·h² = 92.5, Fn -3.1, G1n/G2n -3.2/11.75; Bn 31.75496→31.75 (0.005 dB, documented rounding); σSF 8 |
| 7.4.1 UMa LOS incl. note 1 hE = 1 and random hE | compliant (hE correlation: T3-03) | max 0.0020 dB (E = 13.33 vs 40/3, documented); random hE vs TR draw max 0.0020 dB; PL2 identity `PL1(dBP3) + 40 lg(d3/dBP3) = 28 + 40 lg d3 + 20 lg fc - 9 lg(dBP'² + Δh²)` exact |
| 7.4.1 UMa NLOS (13.54 + 39.08 lg d3 + 20 lg fc - 0.6(hUT-1.5), max LOS) | compliant | Bn 14.44, E3n -0.6; exact; σSF 6 |
| 7.4.1 UMa/UMi/InH optional NLOS | deviation documented | comments with `logdist` parameters (UMa/UMi/InH NLOS files) |
| 7.4.1 UMi LOS/NLOS | compliant | max 0.0021 dB (E rounding); NLOS Bn 22.85, Cn 21.3, E3n -0.3 exact; σSF 4 / 7.82 |
| 7.4.1 InH LOS/NLOS | compliant | exact; σSF 3 / 8.03 |
| 7.4.1 InF LOS, SL, DL (3-way max), SH, DH | compliant | exact; σSF 4.3 / 5.7 / 7.2 / 5.9 / 4.0 |
| 7.4.1 SMa LOS (h = 10), NLOS (W = h = 10, no max) | compliant / 3D-BP documented | LOS max 0.0233 dB at grid, dense 0.0313 dB; A1 21.574, B 30.13, D 0.002; NLOS `LOSmax = 0`, Bn 36.15, D2n 370 exact; σSF 4/6, 8 |
| 7.4.1 applicability ranges, note 2 frequencies | deviation documented | headers of all 23 files; `path_loss` evaluates outside the ranges "as written" |
| 7.4.1 O2I files: PL_b of outdoor LOS/NLOS, σSF of Table 7.5-6 O2I | compliant | UMa/UMi 7, RMa/SMa 8; same PL as outdoor (probe) |
| 7.4.2 RMa, UMi, UMa (C' and clamp to 1), InH mixed/open | compliant | probe at 17 distances, err 0 |
| 7.4.2 InF SL/DL/SH/DH with Table 7.8-7 defaults, HH | compliant | r 0.2/0.6, d_clutter 10/2, h_c 2/6; err ≤ 1.1e-16 |
| 7.4.2 SMa (3 clutter types, r_veg 0/0.1/0.2) | compliant | err 0 for hUT 1.5…13.9 |
| 7.6.3.3 soft LOS (7.6-18) | compliant | err ≤ 1.1e-16 at 3.5 and 28 GHz |
| 7.6.3.3 / Table 7.6.3.1-2 state distances | compliant | LOS 60/50/50/50/10/d_clutter/2, IO 50, SC_lambda all 23 files |
| 7.4.3 Table 7.4.3-1/-2 (low, high, low-A, PL_npi 5 dB, σP) | compliant | err ≤ 1.4e-14 |
| 7.4.3 Table 7.4.3-3 legacy (20 dB, one uniform, σSF 7) | compliant | — |
| 7.4.3 d2D-in (min of two uniforms, 25/10 m) | compliant (SMa mix: T3-04) | — |
| 7.4.3.2 car loss 9/20 ± 5 | compliant | — |
| eq. 7.4-2 d3D-out + d3D-in | compliant (legacy mode documented) | geometric distance; `d2d_in` = legacy option |
| 7.4.4 ACF exponential, 2D | deviation documented (T3-06 nit) | Comb256; vertical offsets fully correlated (2D) |
| Table 7.5-6 Part-1..4 means, stds, fc terms | compliant | `lsp_mu`/`lsp_sigma` vs typed TR at 9 fc x 6 d x 1-3 hUT for all 23 files: max error 0 except InF DL/SH DS 3.6e-4 lg (V/S 4.55 vs 50/11, documented) |
| Notes 6/7 fc clamp (UMa 6, UMi 2, InH 6), note 4 of 7.5-7…12 | compliant | `LSP_fc_min` 6/2/6, none for RMa/SMa/InF |
| Correlation distances, cross-correlations, PD | compliant | all 23 tables checked by hand; all matrices positive definite |
| r_τ, XPR, N (+1 for the NLOS LOS slot), cASD/cASA/cZSA, ζ, cDS | compliant | QuaDRiGa N+1 convention documented; Table 7.6.15-1 D1 counted the same way |
| Tables 7.5-7…7.5-12 ZSD (ESD) | compliant | ES_D_mu incl. height constant, `_A/_B/_C/_D`, floors -0.5/-0.21/-1 |
| ZOD offset (EoD = -ZOD) UMa, UMi, SMa | compliant | err ≤ 1.1e-14 deg |
| ZOD offset RMa | deviation documented | geometric generalization; equals TR at 35/1.5 m, 0 for hUT ≥ 3.5 m |
| ZOD offset O2I (note 5, RMa/SMa O2I columns) | compliant | LOS_O2I of UMa/UMi: 0; RMa/SMa O2I: offset |
| Step 4 Cholesky order [SF, K, DS, ASD, ASA, ZSD, ZSA] | compliant | probe: rel. err ≤ 1e-14, x_SF = s_SF |
| Step 4 clipping 104°/52° | compliant | — |
| Step 4: co-sited sectors same, sites independent, floors uncorrelated | compliant | one phase set per site; floor offsets (same-block corr. 0, probe -0.04) |
| 7.6.5 multi-frequency (same x) | compliant | — |
| 7.6.9 Δτ (Table 7.6.9-1, NLOS, 2L/c for InF/InH) | compliant for NLOS / gap for O2I-LOS (T3-01) | μ, σ, λ all 9 NLOS files; L 120/300/120 m; bound probe exact |
| 7.6.8 ground permittivity | deviation undocumented in code (T3-02) | — |
| 7.6.1 oxygen absorption | not in scope (cluster-level, step 11) | — |
| Bias removal (QuaDRiGa) | deviation documented (T3-06 nit) | — |
| XPR as spatially correlated LSP, XPR/KF λ for N/A | deviation documented | "not from the TR" comments |
| Dual mobility (7.6.10 has no LSF part) | QuaDRiGa extension, documented | — |

## 4. Probes

Python against `lib/quadriga_lib.cpython-314-x86_64-linux-gnu.so` (built 12:49; the later `lsp_generate.cpp` edit of
12:50 concerns only the `cholesky = false` root and validation, all probes use `cholesky = true`), plus typed TR
formulas in numpy. Scripts in the session scratchpad: `pl_probe.py` (PL + σSF, all 23 files), `bp.py` (3D breakpoint
scan), `he.py` (random hE), `lsp_probe.py` (μ, σ, EoD offset, PD), `chol.py` (Cholesky order), `los_probe.py`,
`o2i_probe.py` (O2I, Δτ bound, clipping), `acf.py` (ACF, 2D, floors). No builds, no timings.
