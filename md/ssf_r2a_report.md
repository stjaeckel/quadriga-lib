# R2a report — `ssf_clusters`: shared part, method `38.901`, conf change

## Summary
- New public function `ssf_clusters` (`src/ssf_clusters.cpp`, declared in `include/quadriga_gbsm.hpp`): the complete
  validation of 3.0.2 for all three methods, the shared part (LOS geometry, ground reflection, slot writing,
  `n_active`, abs TOA for the TR methods) and the kernel `qd_ssf_tr` (3.4); `38.901_MF` and `quadriga` throw
  `std::runtime_error("Method '<method>' is not implemented yet.")` after the validation.
- Conf change of section 4 in the six v19 LOS files; `test_conf_files_v19.cpp` updated (incl. the new N_TR + 1 check).
- New test file `tests/catch2_tests/test_ssf_clusters.cpp`: 19 cases of 5.2, 6663 assertions, all pass.
- Full Catch2 suite passes (1422 cases, 1 363 945 assertions); no compiler warnings. Nothing committed.
- **No spec error found** in the three revised parts (GR power contract, log-domain normalisation, abs-TOA rule);
  each was checked against typed TR oracles (details under Decisions, item 1).

## Files
| File | Change |
|---|---|
| `src/ssf_clusters.cpp` | new, 546 lines: helpers + `qd_ssf_tr` (~95), doc block (202), function (~190 incl. validation) |
| `include/quadriga_gbsm.hpp` | +30/−1: declaration of 2.1; group bullet now names `src/ssf_clusters.cpp` |
| `src/qd_gbsm_phases.cpp` | +4/−3: header comment of the SSF index names `ssf_clusters` in `ssf_clusters.cpp` |
| `src/quadriga_gbsm_helpers.hpp` | +3/−2: users of `qd_gbsm_check_links` (file comment and comment above the helper) |
| `data/config/3GPP_38.901_v19_{UMa,UMi,Indoor,RMa,SMa,InF}_LOS.conf` | `NumClusters` +1 with the new comment; `NumClusters_min` +1 in UMa, UMi, Indoor |
| `tests/catch2_tests/test_conf_files_v19.cpp` | +16/−9: table entries, two comments, check `NumClusters == N_TR + 1` per file |
| `tests/catch2_tests/test_ssf_clusters.cpp` | new, 1367 lines |
| `tests/catch2_tests/qd_test_diag.hpp` | `QD_DIAG_GBSM` line lists `test_ssf_clusters.cpp` (5.0) |

## Decisions and deviations
1. **Revised spec parts checked against the TR (no deviation needed).**
   - GR power (2.3, 3.0.5, 3.3.4): the oracle types eqs. (7.6-33), (7.6-37a), (7.6-38)–(7.6-40) in the TR's
     trigonometric form (θ_GR,ZOD = 180° − atan(d2D/(h_t+h_r)), ε = re − j·im) and the slot power
     K/(K+1)·(d3D/d_GR)²·(|R∥|² + |R⊥|²)/2: max error 1.4e-15. The conjugate sign gives the same R² (0 difference).
     A consumer check confirms √(slot power)·diag(R∥, −R⊥)/R equals the eq. (7.6-32) coefficient
     √(K/(K+1))·(d3D/d_GR)·diag(R∥, −R⊥) for both polarisations. The algebraic cos/sin² forms of 3.0.5 equal the
     TR forms (cos(180° − x) = −cos x). Grazing cases: ε = 1 → R = 0 (0/0 branch), ε = 15 − 1j → R∥ = R⊥ = −1.
     The delay form 4·h_t·h_r/((d_GR + d3D)·c) equals (d_GR − d3D)/c (6e-15 rel.). NLOS: GR power 0 (eq. (7.6-32)
     replaces only the LOS eq. (7.5-30)).
     Remark (not an error, by design of the scalar rule): the scalar consumer value √P_GR·exp(j·arg R∥) has the
     magnitude of the polarisation-averaged R, not |R∥|; a V-polarised scalar link per the TR would use |R∥|.
   - Log-domain normalisation (3.0.10): the oracle uses the TR's linear form P'_n = exp(…)·10^(−Z/10) and
     P'_n/ΣP'; agreement 1e-15. The extreme case z = −1100 gives exactly 1/(K+1) on that cluster, others 0.
   - Abs TOA (3.3.3, W13): TR eq. (7.6-43) (NLOS: τ_n + Δτ, first cluster at Δτ) and eq. (7.6-44)/(7.6-44a)
     (LOS: no Δτ; also clause 7.6.13 text "Δτ … only applicable when it's not in LOS case"). Tested: NLOS delays
     equal the no-TOA delays + Δτ bitwise, min = Δτ, inactive slots stay 0, zero offsets = omitted; LOS bitwise
     unchanged.
2. Validation messages not written out in 3.0.2 (my wording): `Input 'rnd_freq' must be finite.`, `Input
   'lsp_anchor' must have size [n_link, 1, 8].`, `Input 'lsp_anchor' must be finite.`, `Input 'eod_offset_deg' must
   have size [n_link, n_freq].`, `Input 'eod_offset_deg' must be finite.`, `Number of elements in 'abs_toa' does not
   match number of rows in 'rx_pos'.`, `Inputs 'gr_eps_re' and 'gr_eps_im' must have size [n_link, n_freq].`,
   `Inputs 'gr_eps_re' and 'gr_eps_im' must be finite.`, `Number of elements in 'n_cluster' does not match number of
   rows in 'rx_pos'.` The return value of `scenpar_validate` (positive definiteness) is ignored (not used here).
3. Shared part prepared for R2b/R2c (one line each, dead until then): the GR power factor is 1 for `quadriga`
   ((d3D/d_GR')² for the TR methods), and the abs-TOA addition is skipped for `quadriga` (its kernel shifts, 3.6
   step 10). Slot writing loops over n_fd and F already.
4. Header comment: the R1 open item (group bullet in `include/quadriga_gbsm.hpp`) is done; the manager's request for
   `src/qd_gbsm_phases.cpp` is read as its file-header SSF index line, which now says the values are used by
   `ssf_clusters` in `ssf_clusters.cpp`.
5. Conf test: the new `NumClusters == N_TR + 1` check is in the per-file loop of the "all 23 files" case; the
   O2I files of UMa, UMi, RMa, SMa use the O2I column (N_TR verified against Table 7.5-6 Parts 1–4).
6. Test tolerances: delays 1e-12 relative to max(delay, DS) (a delay is a difference of nominal delays; its
   absolute error scales with DS), powers 1e-12 relative, angles 1e-11 rad. Measured: delay 0, power 1.0e-15,
   angle 1.8e-15 rad.
7. Tests beyond the table of 5.2: removal case also in a LOS scenario (distinguishes the eq. (7.5-6) criterion from
   eq. (7.5-8)); NLOS case checks that the EoD offset changes only `eod`; O2I case checks the non-flagged links;
   a 0×0 `rnd` for N = 0; outputs untouched on a validation error; the GR consumer check of item 1.
8. Drop-statistics case: passes `eod_offset_deg` and `abs_toa` from `lsp_generate` (neither changes a spread).
9. Doc block: K limited is written `K_lim` (no `*` in running text, 7); the 3.7 table and the inputs-per-method table
   are markdown tables; `38.901_MF`/`quadriga` have one bullet each and their table columns, formulas left to R2b/R2c.
10. Test file 1367 lines vs. "about 1,200": the typed oracle and the 16 validation sections.

SPEC_CPP_API review points A/B/C: no API beyond 2.1; argument order and defaults as 2.1.

## Test results
- Build: existing `/home/stephan/quadriga-lib-agents/build_ssf` (cmake re-run for the new files), `cmake --build .
  --parallel 12`; library and tests with `-Wall -Wextra -Wpedantic`: no warnings.
- Stub run (validation complete, zero outputs): `test cases: 25 | 12 passed | 13 failed`, `assertions: 9413 | 5078
  passed | 4335 failed` (19 ssf_clusters + 6 conf cases). Passing for the right reason: the 6 conf cases; input
  validation (implemented in the stub); empty/pre-sized outputs (sizes); NLOS-zero-LOS-power, swap and
  link-independence (a zero stub is trivially equal); the track diagnostic (finite only). Failing: all oracle cases
  (NLOS, LOS, K limit, removal, n_cluster, O2I, zenith fold, abs TOA, GR), slot-0 LOS angles in the size case, the
  one-cluster and extreme-value cases (power 1 / 1/(K+1) expected), the drop statistics (spreads of all-zero powers).
- Final: `OMP_NUM_THREADS=12 test_bin "*ssf*,conf files v19*"`: `All tests passed (10537 assertions in 44 test
  cases)`; `"*ssf_clusters*"`: 6663 assertions in 19 cases, < 0.1 s.
- Drop medians (UMa NLOS 6 GHz, 2000 links), window = study ± 0.03: DS 0.968 [0.94, 1.00], ASD 1.015 [0.99, 1.05],
  ASA 0.988 [0.96, 1.02], ESD 0.994 [0.95, 1.01], ESA 0.998 [0.96, 1.02].
- Track diagnostic (UMi LOS, 20 m at 1 cm, signs/offsets frozen): angle jumps > 10° 0.50 /m, delay > DS/4 0 /m,
  power > 3 dB 0 /m.
- Full suite from the repository root, `OMP_NUM_THREADS=12`: `All tests passed (1363945 assertions in 1422 test
  cases)`, 27.0 s wall; load 1.81 before, 2.72 after.

## Warnings
None.

## For R2b and R2c
- Kernel output layout used by the slot writer: `c_delay[k·N_l + n]` (k < n_fd), `c_power[k·N_l + n]` (k < F),
  `c_ang[(k·4 + a)·N_l + n]`, a = 0 AOD, 1 AOA, 2 EOD, 3 EOA, all in rad / s; the kernel writes active clusters only.
- Replace the `if (!m_tr) throw std::runtime_error(...)` line by the dispatch; the kernel call and the `centre`
  (NLOS: ZOD offset at k = 0, O2I) are `38.901`-specific and sit inside `if (N_l > 0)`.
- `rnd_freq` is copied into `z[k·N + n]` (all F); for `quadriga` (`rnd_freq` ignored, may be nullptr) the copy
  must be skipped.
- R2c: the GR factor (1) and the abs-TOA skip are in place but untested; GR geometry and R²_k per link are computed
  before the kernel (`tau_gr`, `e_gr`, `R2[k]`), as 3.0.5 asks.
- Validation baselines of the other methods belong in their own files (5.3, 5.4); the validation case of this file
  already covers `rnd_freq` missing for `38.901_MF` and all `lsp_anchor` checks.

## Open points / risks
- The `38.901` angle mapping is discontinuous at the −25 dB threshold and at the first-cluster change by design
  (documented); the diagnostic shows 0.5 angle jumps per metre on the test track.
- `html_docu/` not regenerated (R3).

## Suggested commit cut
One commit, files: `src/ssf_clusters.cpp`, `include/quadriga_gbsm.hpp`, `src/qd_gbsm_phases.cpp`,
`src/quadriga_gbsm_helpers.hpp`, the six `data/config/3GPP_38.901_v19_*_LOS.conf`,
`tests/catch2_tests/test_ssf_clusters.cpp`, `tests/catch2_tests/test_conf_files_v19.cpp`,
`tests/catch2_tests/qd_test_diag.hpp`.

```
Small-scale fading: cluster delays, powers and angles (ssf_clusters, TR 38.901 clause 7.5)

New function ssf_clusters with the method "38.901": TR 38.901 clause 7.5 steps 5 to 7 at one
frequency (exponential delays with C_tau in LOS, powers with per-cluster shadowing normalised in
the log domain, removal 25 dB below the strongest cluster without renormalisation, angle mapping
with C_phi and C_theta interpolated over the per-link cluster number, the K-factor limited to
-5 dB in the polynomials, signs and offsets from spatially consistent fields, ZOD offset and O2I
mean zenith in NLOS, zenith fold). Slots per link: LOS path, ground reflection (TR clause 7.6.8,
coefficient power K/(K+1)·(d3D/d_GR)²·R² with the polarisation-averaged reflection power), then
the NLOS clusters; absolute time of arrival added in NLOS scenarios only (TR eqs. 7.6-43/-44).
The other two methods are validated but not yet implemented.

The six TR 38.901 v19 LOS scenario files now count NumClusters as the TR cluster number plus one
slot for the LOS path, as the NLOS and O2I files already do.
```
