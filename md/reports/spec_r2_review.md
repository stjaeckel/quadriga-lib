# Report: critical review of the implementation spec, round 2

Task: `md/prompts/spec_review_r2.md`. Reviewed `md/gbsm_lsf_spec.md` r1.1 (1896 lines) against TR 38.901 v19.5.0,
QuaDRiGa-NG (`get_pl.m`, `gen_lsf_parameters.m`, `init_sos.m`, `qd_builder.m`, `qd_sos.m`, `set_scenario.m`,
`gen_o2i_loss.m`, `gen_ssf_parameters.m`, `gen_fbs_lbs.m`, `add_sdc.m`, `get_channels.m`), the quadriga-lib sources
(`sos_val`, `sos_remove_bias`, `sos_preset`, SOS helpers, `qd_scenpar.cpp`, wrapper helpers, tests) and the three
SPEC files. Findings R2-01 to R2-22 are appended to `md/gbsm_lsf_findings.md`. No spec or code was edited, and
nothing was built. Numerical checks were run as scratch Python scripts (numpy) outside the repository.

Counts: 0 blocker, 3 major, 10 minor, 9 nit; 3 marked [escalate] (R2-11, R2-12, R2-16).

## 1. Rationale of the findings

### Major

**R2-01: the dual-mobility coupling counts the correlation twice.** In 3.2, slice 1 of the phase cube is the mirror of
slice 0. For the pair (ASD, ASA) the two-map values are therefore `x3 = (f3(R) + f4(T))/sqrt2` and
`x4 = (f4(R) + f3(T))/sqrt2`, with f3 and f4 independent. Their covariance is
`Cov(x3, x4) = (Cov(f3(R), f3(T)) + Cov(f4(T), f4(R)))/2 = (A3(D) + A4(D))/2`. That is exactly the spec's `c`, and
the pair already carries the wanted coupling. Step 5 then multiplies by `sqrtm([[1, c], [c, 1]])`, which squares the
covariance matrix:

| c | var without coupling | corr without coupling | var with coupling | corr with coupling |
|---|---|---|---|---|
| 1.000 (D = 0) | 0.996 | 1.000 | 1.992 | 1.000 |
| 0.819 | 0.997 | 0.819 | 1.667 | 0.980 |
| 0.162 | 1.000 | 0.163 | 1.026 | 0.316 |

(Monte Carlo with 4e5 samples; the analytic values with coupling are 1 + c² and 2c/(1 + c²).) This contradicts the
documented property "columns 0 to 8 Normal(0,1)". `gen_lsf_parameters.m` does the same: `qd_sos.val` in mode 3 on the
mirrored phases, then `sqrtm`. The TR is silent (7.6.10 covers Doppler only), so guidance 2 favours the correct
statistics. Dropping the step keeps the plan's intent (L10: coupled by the ACF at the Tx-Rx distance), keeps exact
swap reciprocity, needs less code, and removes the 3e-5 tolerance class. The formula test of 5.2 re-types the
coupling, and the marginal test covers single mobility only, so the current tests would pass the wrong statistics.

**R2-02: random hE combined with the 1.1·hE clamp.** `dual_slope` clamps `hm = max(hUT, 1.1·hE)`. For the fixed
hE = 1 this never acts at TR heights. The TR set for the random hE ends at hUT - 1.5, so the clamp acts whenever
hE lies in (hUT/1.1, hUT - 1.5]. At the TR UT heights this is floor 7 (hUT 19.5, hE 18, hm 19.8) and floor 8
(hUT 22.5, hE 21, hm 23.1). The effective height h'UT becomes 1.8 or 2.1 instead of 1.5 (TR note 1), and d'BP grows
by 20 % or 40 %. Beyond the breakpoint `PL2 = PL1(dBP3) + 40·lg(d3/dBP3)` then drops by 18·lg(1.2) = 1.4 dB or
18·lg(1.4) = 2.6 dB, for example at hBS 25 m and 3.5 GHz (d'BP 280 m in the TR, 392 m in the library). For
hUT 22.5, P(hE = 21) is about 9 % at d2D 100 m. Neither the random-hE test (hUT 20) nor the TR points (hUT 1.5, 12)
reach these cases.

**R2-03: the probit in the upper tail.** I re-implemented 3.5.4 literally and compared it with a reference that
works in the small tail:

| p | spec probit, rel. error | Acklam alone | naive bisection on 0.5·erfc(-x/sqrt2) |
|---|---|---|---|
| 1e-12 | 0 | 1.0e-9 | 0 |
| 0.3 / 0.7 | 1e-16 / 2e-16 | 5.7e-10 | 0 / 2e-16 |
| 1 - 1e-6 | 2.2e-12 | 1.1e-9 | 2.4e-12 |
| 1 - 1e-9 | 4.8e-11 | 4.8e-11 | 1.5e-9 |
| 1 - 1e-12 | 1.0e-9 | 1.0e-9 | 1.1e-6 |

At 1 - 1e-12 the target 1e-10·max(1, abs(F)) is missed by a factor of 10, and the test oracle described in 5.5 is
wrong by 1.1e-6 relative there. With symmetric evaluation (`F(p) = -F(1 - p)` for p > 0.5, where 1 - p is exact) the
worst error over [1e-12, 1 - 1e-12] is 3.1e-16. The practical effect on `los_soft` is below 1e-8, but R3b's
acceptance ("all new cases pass") would fail or be decided by the oracle.

### Minor and nit (beyond the table text)

- R2-04 (legacy parity): `get_pl.m` lines 132 to 186 take `o2i_d3din` per segment and apply it to all snapshots, and
  `sf_sigma` is averaged per segment. In the recursive `nlos` call `o2i_loss` is added inside the maximum. 10.2
  describes the double addition correctly, but 10.4 does not exclude it.
- R2-06: values computed (pUMa(20, 20) = 0.97779; the excess over 1 exists only for d < 18.29 m at hU 20 and d < 18.47 m
  at hU 22.5).
- R2-07: the spec's recipe works for files without alpha/beta terms; with `*_alpha` the elevation term cannot vanish
  together with the distance and height terms.
- R2-11, R2-16 touch plan decisions D13 and D26 and are therefore escalated. R2-12 asks the user whether the three
  `md/SPEC_*.md` files count as source (the r1.1 apply report left this open).

## 2. Checked and confirmed (no finding)

- `d2d_in > 0` gate (3.3.2): matches MATLAB. Outdoor UTs never carry `o2i_d3din` (`set_scenario` clears `par`), and
  `get_pl` uses `d_3d_in = 0` for them. For indoor UTs the library reproduces `gen_o2i_loss.m` lines 185 to 223,
  including the NaN -> Inf -> d3 branch at d2D = 0 and the 10 m rule. The only difference is measure zero: an indoor UT
  with d2d_in exactly 0 at d2D exactly 0, which MATLAB gives `max(d3 - 10, 0)`. Without the gate, an outdoor link at
  d2D = 0 would get d3in = d3 - 10, so the gate is needed. Only the documentation is missing (R2-19). The per-segment
  constant d3in is a parity issue (R2-04), not a gate issue.
- Newton-Schulz sqrtm (3.4.3): implemented exactly as written (Frobenius stop at 1e-10, update kept, at most 100
  iterations, symmetrized). Results:

  | matrix | iterations | max abs(X·X - R) | X vs eigen root |
  |---|---|---|---|
  | UMa NLOS file correlations (lambda_min 0.016) | 13 | 1.8e-15 | 1.6e-15 |
  | one correlation 0.999999 (lambda_min 1e-6) | 25 | 2.4e-14 | 3.0e-14 |
  | random spectrum, lambda_min 1e-6 / 1e-10 | 25 / 37 | 1.1e-13 / 8.4e-12 | |
  | lambda_min 1e-12 / 1e-14 / 1e-16 | 54 / 100 (limit) / 100 (limit) | 8e-11 / 1.6e-9 / 2.3e-8 | |

  Convergence condition, rates (2.25 per step, then 0.75·e²), iteration counts, the "limit only below about 1e-12"
  claim and the 1e-12 test tolerance are confirmed. The symmetry check is trivially exact after symmetrization; the
  informative checks are X·X = R and positive definiteness.
- RMa/SMa breakpoint form: worst case over the TR ranges 0.03413 dB (RMa: hBS 33, hUT 1, 0.5 GHz) and 0.03130 dB
  (SMa: hBS 27, hUT 1, 0.5 GHz). At the 5.3 test points: 4.14e-3 dB (RMa) and 9.2e-5 dB (SMa). 0.035 dB is justified.
- Full-precision constants of 4.4.2 recomputed: A1, B, D (RMa, SMa), Bn (RMa), E = 20·pi/3 and 40/3 all match to the
  last digit.
- InF V/S: small hall 4, big hall 50/11, LOS average 47/11. Today's -7.2535 is the mean of -7.27812 and -7.22883, and
  47/11 gives -7.25277. The NLOS files today hold SL/DH -7.2582 (small hall) and DL/SH -7.2138 (big hall), consistent
  with the hall assignment of 4.4.1. TR Table 7.5-6 Part-3 and 7.8-7 (r, h_c, d_clutter, hall sizes) checked.
- LOS probability (Table 7.4.2-1): RMa, UMi, UMa (C' incl. the upper end), InH mixed and open, InF SL/DL/SH/DH/HH,
  SMa (the `max(., 0)` extension is equal for hU < 15) and the defaults of Table 7.8-7 are correct. The 38.811 tables
  and the use of the absolute height for the elevation match `set_scenario.m`. `LOS_state_lambda` and
  `IO_state_lambda` values match Table 7.6.3.1-2.
- O2I (7.4.3): v19 IRR glass 25.4 + 0.11 f, low/high/low_A, sigma_P, Table 7.4.3-3 and the car model are correct.
  The cumulative-weight rule never picks a zero weight.
- TR order of the Cholesky (step 4): the permutation `pi = [2, 1, 0, 3..7]` and `x_SF = s_SF` are correct.
  The 2x2 principal root formula p, q is correct as algebra (but see R2-01).
- Soft LOS (7.6-18), absolute TOA (7.6.9 incl. bound and NLOS-only files), ground permittivity, EoD offset sign
  (UMa, SMa), `LSP_fc_min` / note 4: correct.
- SOS precision: AVX2 bound 9.05e-6 (256) and 1.81e-5 (1024); tolerances 2e-5, 1.5e-5, 4e-5 and 1e-5 (uniform slope
  <= 0.399) hold with a margin of about 2.2.
- Floor offsets: zero cross-correlation within a block of N IDs and 1/sqrt(2N) between blocks are correct.
- Internal references: sections 1 to 11 contain no D-/W-/I-/R1- IDs, plan or findings references, except the
  `md/SPEC_*` files (R2-12) and the run-report path, which is an output and acceptable.

## 3. SSF readiness (independent verification)

Sources: `gen_ssf_parameters.m`, `gen_fbs_lbs.m`, `get_channels.m`, `add_sdc.m`, `init_sos.m` lines 160 to 254,
TR 7.5 steps 5 to 12, 7.6.2, 7.6.3.1 and 7.6.3.3, 7.6.5 and 7.6.5.1, 7.6.9, 7.6.13 to 7.6.16.

Quantities the SSF step reads from the LSF stage, and whether they are provided:

- **Delays and powers (step 5/6):** DS per link and frequency, `r_DS`, `LNS_ksi`, KF (dB for the LOS power and
  C_tau). Provided by the `lsp` slices 0 and 1 plus scenpar.
- **Angles (step 7):** ASD, ASA, ESD, ESA per link and frequency; the per-link ZSD mean for c_ZSD = 3/8·10^mu
  (`lsp_mu` slice 5); the ZOD offset (`eod_offset_deg`); the O2I mean ZOA of 90 deg (builder state). Provided.
- **XPR (step 9):** the TR needs mu_XPR and sigma_XPR per ray (`lsp_mu`/`lsp_sigma` slice 7). QuaDRiGa uses the
  per-link XPR as the mean and 0.1·sigma (`lsp` slice 7 and `lsp_sigma` slice 7). Both are provided. The existing
  row of section 8 names only the QuaDRiGa variant; a note for the TR variant would help.
- **Step 11/12 and 7.6.9:** abs TOA (NLOS files only), path loss, the final SF via `sf_sigma`, and O2I loss added by
  the caller. Provided.
- **Ground reflection (7.6.8):** permittivity per link and frequency, with the conjugate note. Provided.
- **Per-frequency handling (7.6.5):** one x for all frequencies, all LSF outputs per frequency, `abs_toa` per link,
  `los_soft` per frequency. Provided. The anchor-frequency method of 7.6.5.1 works by adding the anchor to `fc`.
  The per-frequency SSF generators (XPR, phases, per-cluster shadowing) belong to the SSF sets.
- **Soft LOS:** two builders with independent phase sets. Provided. Cluster-number range (7.6.15): user-chosen,
  keys exist. SNS Pr_sns (7.6.14) is a per-UT truncated normal and an SSF draw. Near-field and polarization
  variability: SSF.

Generator machinery. The SSF sets of QuaDRiGa are as follows: `path_sos` holds 5 Uniform generators per cluster, with
mirror pairs for the angles and self-mirrored delays; `xpr_sos` holds Normal generators per cluster, subpath and
frequency, all self-mirrored; `pin_sos` and `clst_dl_sos` are Uniform and self-mirrored. All use one lambda
(SC_lambda), and none has bias removal. The planned generic evaluation (table of {lambda, uniform, mirror, bias}, own
frequency table per field, floors, 2D/3D, dual mobility) covers all of them. Two corrections are needed:

- The pair coupling must go (R2-01); otherwise it would also hit Normal SSF pairs.
- Tables should be per distinct lambda (R2-21).

Moving the file-local evaluation into the internal header at SSF time is an acceptable refactor.

What the builder must store (missing from 8/10.2, see R2-08 and R2-09):

- the LSF phase cube;
- the seed or the SSF phase cubes, drawn so that they do not repeat the LSF columns;
- `preset` and `horizontal` (they must match between calls);
- per UT: the floor index and, in legacy mode, `d2d_in`;
- per link: the LOS/indoor state (and `los_soft` if used);
- the scenpar/plpar of the LOS and the NLOS file for soft LOS.

`lsp_vals` cannot be emulated by a reference geometry (R2-07).

Conclusion: no public LSF API is missing. Round 1's claim holds, provided that R2-01, R2-07, R2-08, R2-09 and R2-21
are applied to sections 3.0.5, 8 and 10.

## 4. Implementability per run

| Run | Verdict | Remarks |
|---|---|---|
| R1 | ready | Mechanical and well specified. Fix the line range of R2-05; the escalation of R2-12 also affects R1 step 8. |
| R2a | ready after fixes | R2-01 changes 3.2 step 5, 5.0 and 5.2 (simpler code, one tolerance class less, one statistical case more). Also R2-17 wording and R2-21 (generic tables). Read list and tests are otherwise complete. |
| R2b | ready | R2-18 and R2-20 are wording/tolerance nits. If R2-16 is accepted, `vs_ratio` and its tests go. Newton-Schulz and the TR-order Cholesky are fully specified and verified. |
| R3a | ready after fixes | R2-02 changes the `dual_slope` clamp rule and adds two test links. R2-19 is documentation. TR oracles and tolerances are verified (constants, 0.035 dB, 4.1e-3 dB at the test points). The test file is large (23 files × points) but fits one context. |
| R3b | ready after fixes | R2-03 (probit and oracle), R2-06 (test point), R2-13 (argument order), R2-22b (reason of the InF fallback). |
| R4a | ready | Depends only on the final C++ blocks. Wrapper helpers exist (`qd_mex_struct2scenpar`, `qd_mex_struct2plpar`). |
| R4b | ready | Helpers exist (`qd_python_dict2scenpar`, `qd_python_dict2plpar`, `qd_python_list2vector_Strings`); the dispatcher pattern matches `python_tools.cpp`. |

Section 10 (out of scope of the runs) needs R2-04, R2-07, R2-08, R2-09 and R2-10 before the integration prompts are
written.

## 5. Overall judgement

**Ready after fixes.** There are no blockers. The three major findings are local edits:

- R2-01: delete a step and add a statistical test;
- R2-02: skip one clamp for the random hE and add two test links;
- R2-03: symmetric probit evaluation and a corrected oracle.

Each would otherwise give wrong statistics (variance of ASD/ASA/ESD/ESA up to 2 in dual mobility), a TR deviation of
up to 2.6 dB on floors 7 and 8 of UMa drops, or a failing R3b acceptance test. The minor findings complete the
hand-over (parity exceptions, floor IDs, seeds, `lsp_vals`) and fix one wrong test value. Three items need the
user's decision (R2-11 default preset, R2-12 status of the SPEC files, R2-16 `vs_ratio`).
