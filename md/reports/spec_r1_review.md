# Report: spec review, round 1

Task: `md/prompts/spec_review_r1.md`. Reviewed `md/gbsm_lsf_spec.md` (draft r0, 1656 lines) against the writer report,
TR 38.901 v19.5.0, the QuaDRiGa-NG sources (`init_sos.m`, `gen_lsf_parameters.m`, `qd_builder.m`, `qd_sos.m`,
`get_pl.m`, `set_scenario.m`, `gen_o2i_loss.m`, `gen_ssf_parameters.m`, `gen_fbs_lbs.m`, `get_channels.m`,
`add_sdc.m`, `get_sf_profile.m`, `get_lsp_map.m`), the quadriga-lib code (`include/quadriga_sos.hpp`, `sos_val.cpp`,
`sos_remove_bias.cpp`, `sos_preset.cpp`, `qd_scenpar.cpp`, `quadriga_lib_helper_functions.hpp`, the MEX/Python
scenpar/plpar helpers, the conf tests, the v19 files) and the three SPEC files. Findings R1-01 to R1-30 are appended
to `md/gbsm_lsf_findings.md`. Two numerical checks were run with Python in the scratchpad (floor cross-correlation
from the preset tables; RMa/SMa breakpoint difference). Nothing in the repository was built or changed apart from
the findings file and this report.

## 1. Overall judgement

The spec is thorough, traceable and mostly correct: the formulas I spot-checked against the TR and `get_pl.m` /
`gen_lsf_parameters.m` / `set_scenario.m` / `gen_o2i_loss.m` are transcribed correctly (path-loss models, UMa note 1,
LOS probabilities, O2I tables, EoD offset sign, LSP mean/std, TR-order Cholesky permutation, 2x2 coupling root, ground
permittivity, absolute TOA, the new conf values). It is not yet implementable autonomously for three reasons:

1. Two test specifications cannot pass as written (R1-01 AVX2 precision vs. 1e-12; R1-02 RMa/SMa TR oracle vs.
   QuaDRiGa's 3D breakpoint, up to 4e-3 dB). An implementation agent that may not deviate stops there.
2. Runs R2 and R3 are each too large for one context (R1-06).
3. The header that R1 must copy verbatim carries internal decision IDs (R1-03).

Beyond that the API is heavier than needed: an O2I model zoo with geometry outputs, two 36.873 LOS models that are
identical to 38.901 ones, a legacy sqrtm that NG can do itself, a redundant O2I flag, and an internal header that
collects single-use helpers. Two design gaps remain: bias removal with floor IDs (R1-05) and dual-mobility
reciprocity under bias removal (R1-11, with a simpler alternative). After these fixes the spec is ready for R1.

Counts: 2 blocker, 9 major, 14 minor, 5 nit. Escalations: R1-09 (drop sqrtm), R1-10 (O2I scope).

## 2. Writer choices W-1 to W-40

| W | Verdict | Reason, alternative, trade-off |
|---|---|---|
| W-1 | keep, fix the claim (R1-16), resolve R1-05 | Fourier offsets give exactly zero cross-correlation at zero horizontal lag (UTs stacked above each other); at other lags the residual is 0.11 to 0.16 (Comb256, computed), the same as pseudo-random offsets. Alternatives: pseudo-random offsets for every floor (simpler, +-0.044 at lag 0); z-offset in 2D mode (bias removal would work automatically, but needs offsets so large that small lambdas approach the precision limit of the kernels). Reuse `qd_sos_rng`/`qd_sos_uniform`. |
| W-2 | keep | Exact swap reciprocity for unequal lambdas; MATLAB evaluates the partner's phases with its own frequencies (reciprocal only for equal lambdas). No cost. |
| W-3 | keep | MATLAB's sqrtm of a non-symmetric 2x2 has no statistical meaning; the symmetric average equals MATLAB for equal lambdas; closed form, no sqrtm. |
| W-4 | keep | MATLAB never corrects the GR generator; a Uniform field has no exact 0.5 mean after the correction anyway. |
| W-5 | change (R1-11) | Correct every Normal field once over the union of Rx and Tx positions and mirror afterwards: exact reciprocity, one code path; link means only approximately zero in dual mobility (as already after the coupling). |
| W-6 | keep | An explicit error beats a silent single-device correction. |
| W-7 | keep | Consistent 2D definition (TR: horizontal plane); a drone above a UT gets coupling distance 0 (documented). |
| W-8 | keep | Makes the `sf_sigma` pass-through neutral; differs from `get_pl.m` (SF_sigma + SF_delta lg f) only with SF_omega or kappa/tau/beta terms, which no v19 and almost no legacy file has. |
| W-9 | keep (nit R1-30) | Required keys throwing is better than MATLAB's crash; E > 0 guarantees dBP > 0 with the clamps. Multi-value entries are truncated silently. |
| W-10 | keep | MATLAB returns a complex PL below the horizon; +Inf is the honest value. |
| W-11 | keep | TR leaves C' above 23 m and the empty height set undefined; the quantile mapping is deterministic and testable. |
| W-12 | keep | Exact constants; but see R1-02 (RMa/SMa differ from the TR for another reason). |
| W-13 | keep | Any default V/S for the shared InF LOS file is arbitrary; 47/11 is the neutral mean, change 7e-4 in lg. |
| W-14 | keep | Becomes automatic if W-29 changes (R1-14). |
| W-15 | keep | Checked: the TR UMa formula exceeds 1 just above 18 m for hUT > 13 m. |
| W-16 | keep | TR undefined outside its height ranges; rules keep p in [0, 1]. |
| W-17 | change (R1-12) | With d^3, `36.873_UMa` equals `38.901_UMa`; `36.873_UMi` equals `38.901_UMi`. Drop both and `indoor_ut`. |
| W-18 | keep (optional: leave in NG) | QuaDRiGa NTN tables, not 38.901; about 20 lines either way. |
| W-19 | change, escalate (R1-10) | Reduce to the TR models with 4 random columns; mmMAGIC, `high_v14`, legacy d2D-in table and the geometry outputs move to NG or `path_loss`. |
| W-20 | keep | SPEC_CPP_API constant; the TR's 3e8 lives in the file constants. |
| W-21 | keep algorithm, change placement (R1-24) | Acklam + Halley is standard; file-local in `los_probability.cpp`, accuracy target relaxed, tested through the public function. |
| W-22 | keep | Implementation must skip SOS calls for empty positions (R1-28). |
| W-23 | keep | Cheap. |
| W-24 | keep | The repository's current diagnostics mechanism. |
| W-25 | keep | Follows the existing scenpar/plpar wrapper helpers (verified: `qd_mex_struct2scenpar`, `qd_mex_struct2plpar`, `qd_python_dict2scenpar`, `qd_python_dict2plpar` exist) and the SPEC `complex` flag. |
| W-26 | keep | Same pattern as `sos_val` (SECTION_DESC in C++, MEX and Python). |
| W-27 | keep, document (R1-29) | NG's reflection code expects the positive sign; the TR uses the conjugate. |
| W-28 | keep | Differs from the TR only for hBS < 1e-4 m. |
| W-29 | change (R1-14) | `scenpar_validate(..., validate = true)`: same messages, no per-function key checks. |
| W-30 | keep | Useful contract for tests and the parity mapping. |
| W-31 | keep | Thin wrapper check. |
| W-32 | change (R1-15) | Per-link heights; MATLAB's mean is per call (track) and not reproducible by a library call over many UTs anyway. |
| W-33 | keep | Follows D8; values checked against Table 7.6.3.1-2. |
| W-34 | keep, plus R1-04 | Placement fine; the DS overview line of the writer must learn the V/S term. |
| W-35 | keep | Trivial edge rule. |
| W-36 | keep | Consistent with the model's d2D. |
| W-37 | keep | Does not reject unused columns. |
| W-38 | keep | See I-4. |
| W-39 | change (R1-08) | Header only for cross-file helpers (generator table, frequency setup, position prep, LSP mean/std, shared 8x8 Cholesky incl. `qd_scenpar_xcorr`); the rest file-local, written generically for later SSF reuse. |
| W-40 | keep | Backward compatible. |

## 3. Issues I-1 to I-8

| I | Resolution |
|---|---|
| I-1 | Resolve by R1-11 (union-of-positions correction): swap reciprocity then holds exactly with bias removal, test 5.2 runs with corrected phases. If R1-11 is rejected, keep the spec's handling. |
| I-2 | Drop the 36.873 names (R1-12): with the TR's d^3 they equal the 38.901 models; `set_scenario.m`'s d^2 is a bug (38.901 inherits 1.25e-6 d^3 = 5/4 (d/100)^3 from 36.873). No legacy flag; NG maps the 3GPP_3D groups to 38.901 names; LOS probabilities of indoor UTs above 13 m change in NG as a bug fix. |
| I-3 | Accept 47/11 (W-13). |
| I-4 | Accept: release-note text in the R4 report. |
| I-5 | Accept the full-precision constants; add R1-02 (RMa/SMa 3D-breakpoint difference) with its tolerance and a file comment. |
| I-6 | Accept N(0,1). Consequence: O2I has no parity, so legacy O2I variants have no purpose (R1-10). |
| I-7 | Accept; NG's `get_pl` wrapper adds the O2I loss once. |
| I-8 | Moot if R1-10 is adopted; otherwise keep `mmMAGIC_low`/`mmMAGIC_high`. |

## 4. Precision (review point 5)

`sos_val<double>` with `use_kernel = 0` already uses the AVX2 kernel (argument reduction in double, sin/cos in
single precision, accumulation in double; documented deviation < 4e-7 · sqrt(2 · n_coeff), typically < 1e-6). The
float instantiation brings no speed-up: it converts inputs and outputs and computes the same way. The spec should
say: `lsf_draw` uses `sos_val<double>` with `use_kernel = 0`; `sos_remove_bias` is double throughout (closed form);
`lsp_generate`, `path_loss`, `los_probability`, `o2i_loss` are pure double. Test tolerances (R1-01): 2e-5 absolute
for Normal values from `lsf_draw` with 256 sinusoids (4e-5 for 1024), 1e-5 for the Uniform columns, 2e-5 for
zero-mean checks through `lsf_draw`; 1e-12 where the oracle evaluates phases directly; exact for comparisons of the
same evaluation. Downstream LSPs inherit a relative error of ln(10) · sigma · 1e-5, negligible. D23 parity of the
SOS stage needs 1e-5 absolute (R1-18).

## 5. Cholesky, sqrtm, probit and BLAS/LAPACK (review point 10)

Hand-written code really needed:

| Piece | Lines | LAPACK replacement | Alternatives |
|---|---|---|---|
| 8x8 Cholesky | 18 exist in `qd_scenpar_xcorr`; as a shared helper about 22, net +5 because `qd_scenpar.cpp` reuses it | `dpotrf` (`arma::chol`) | none needed |
| TR-order permutation, 2x2 coupling root | about 10 | none | closed forms (already in the spec) |
| Symmetric square root (legacy) | Jacobi about 45 + 40 test lines | `dsyev`/`dsyevd` (`arma::eig_sym`, `arma::sqrtmat_sympd`) | Newton-Schulz about 15 lines (matrix products only); or drop it: NG uses MATLAB's `sqrtm` (R1-09) |
| Inverse normal CDF | Acklam + Halley about 25 | not in BLAS/LAPACK | Boost.Math `erf_inv` (heavy dependency); Newton on `std::erfc` about 12 lines; the C++ standard has no `erfinv` |

Total about 100 lines with Jacobi, about 55 without sqrtm. LAPACK would save at most about 60 lines and nothing for
the probit. Cost of enabling it: CMake detection and Armadillo configuration (`ARMA_USE_LAPACK`, integer width,
underscore convention); MEX against MATLAB's `libmwlapack`/`libmwblas` uses 64-bit integers (ILP64) while Octave
links an LP64 LAPACK, i.e. two configurations and crash risk when mixed; Windows/MSVC has no system LAPACK
(ship OpenBLAS or MKL DLLs); Python wheels must bundle OpenBLAS (auditwheel/delocate/delvewheel, +tens of MB,
possible clashes with numpy's OpenBLAS and its thread pool); CI and the Debian packaging get a new dependency; no
speed gain for 8x8 matrices. Recommendation: keep BLAS/LAPACK disabled, share the existing Cholesky, drop the sqrtm
option (escalated) or use Newton-Schulz if it stays, keep the probit file-local.

## 6. SSF readiness (review point 13)

Checked against `gen_ssf_parameters.m` (DS/AS/KF per frequency, `gr_epsilon_r`, `absTOA_offset`, `lsp_vals(8,:)`
for the XPR), `gen_fbs_lbs.m` (cluster spreads from scenpar), `get_channels.m` (PL, `get_sf_profile` SF and KF along
the track, `scale_sf`), `add_sdc.m` (`lsp_vals(4..8,1..2,:)`, `get_pl`) and TR 7.5 steps 5 to 12, 7.6.2, 7.6.3.1,
7.6.5, 7.6.13 to 7.6.16. The LSF API provides everything the SSF step consumes: DS/ASD/ASA/ESD/ESA/KF per link and
frequency, SF with `sf_sigma`, XPR mean/std (`lsp_mu`/`lsp_sigma` slice 7), the per-link ZSD mean for c_ZSD, the EoD
offset, the absolute TOA, the ground permittivity, the soft LOS state, the cluster-count keys, the path loss, and
(through the generic internal evaluation) spatially consistent cluster/ray generators with mirror, floor, 2D and
dual-mobility rules. Gaps to add to section 8 (R1-22): 7.6.5.1 anchor-frequency LSPs (add the anchor to `fc`; state
the clipping); soft LOS needs a LOS and an NLOS builder per link with independent phases; the `lsp_vals`
replacement is not 1:1 (per-builder coefficient table vs. per-link `lsp_mu`). The SSF helpers need not live in the
header now (R1-08); written generically they can be moved when the SSF port starts.

## 7. Other checks

- MSVC (point 11): no problem found in the spec's code: no VLAs, no designated initializers, no GCC builtins,
  OpenMP 2.0 rule present, `NAN` default arguments already used in existing headers, `std::mt19937_64` raw outputs
  portable.
- Conf values (point 15) checked against the TR: LOS-state and indoor-state distances (Table 7.6.3.1-2), O2I maximum
  indoor distances (7.4.3.1), InF V/S values and DS constants (Table 7.5-6 Part-3, 7.8-7 hall sizes:
  120x60x10 m gives 4, 300x150x10 m gives 50/11; -7.2582 and -7.2138 reproduced), full-precision constants
  (RMa A1 20.4779, B 31.7408, D 0.0013979, Bn 31.7550; SMa A1 21.5744, B 30.1326, Bn 36.15; E = 40/3 and 20 pi/3),
  InF-DL extra term (33 + 25.5 lg d + 20 lg f), SMa NLOS without max, O2I tables (7.4.3-1 to -3, car 9/20 +- 5),
  LOS probabilities incl. InF defaults (7.8-7) and SMa, abs TOA (7.6.9-1), EoD offsets (7.5-7 to 7.5-12), UMa NLOS
  LSP row of Table 7.5-6 Part-1 against the v19 file. No mismatch except R1-02.
- Not checked: the mmMAGIC deliverable (not available), TR 36.873 (not in the references; the I-2 conclusion rests on
  38.901 inheriting the d^3 term from 36.873), the 93-value satellite table beyond a spot check.
