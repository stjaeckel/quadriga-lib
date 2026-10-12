# Log: port of the sum-of-sinusoids model (qd_sos)

Running record for `md/PLAN_QD_SOS.md`. One section per work item, later one section per function.

Experiment scripts and raw CSV output of WI-1 are in the session scratch space
(`.../scratchpad/bias/exp1.m` to `exp6.m`), not in the repository. All experiments ran in
MATLAB (`matlab -batch`) against the unmodified `@qd_sos` of QuaDRiGa-NG with fixed `rng` seeds.


## WI-0 Orientation (2026-10-09)

- Read `PLAN_QD_SOS.md`, `SPEC_CPP_API.md`, `SPEC_MATLAB_API.md`, all of `@qd_sos`, and
  `@qd_builder/init_sos.m`, `gen_lsf_parameters.m`.
- Ran the 19 SOS test files of QuaDRiGa-NG (`testSOS_*.m`, `testBuilder_CorrMap_SOS.m`) in MATLAB with
  a scratch runner (working directory in scratch, because `testSOS_save_load` deletes `*.mat` files in
  the working directory). Result: 19 of 19 passed. Octave was not run.
- Not run: the complete `xunit_new/run_tests.m` (only the SOS files are relevant here).


## WI-1 Bias analysis (2026-10-09) — STOP 1

Notation: N sinusoids, P positions, d = decorrelation distance, L = edge length of the square that
holds the positions (planar, z = 1.5 m), a = sqrt(2/N),
`z_n = (1/P) · sum_p exp( j·2·pi · f_n·x_p )`, `Z = sum_n z_n · exp( j·phi_n )`.

### 1. Exact expressions (derived, then confirmed by measurement)

- Sample mean of one realisation: `m = a · Re( Z )`. Checked against `mean( val( pos ) )`:
  difference at most 1.8e-15 in 96 cases (column `identity_err` of exp1).
- Averaged over the phases, `E[m] = 0` and `Var[m] = (1/N) · sum_n abs(z_n)^2`. This equals the mean of
  the approximated ACF over all position pairs, `(1/P^2) · sum_pq R_sos( x_p - x_q )`.
  Monte-Carlo over 300 phase sets agrees with it in all 96 cases within the Monte-Carlo error
  (columns `mc_std`, `exact_std`).
- There is no bias in the expectation: the Monte-Carlo mean of `m` is zero within its error in all
  cases. What is called bias is the per-realisation value of `m`.

### 2. Cause: two components

Spread of the sample mean (standard deviation over phase sets), P = 1000 positions. `target` is the
same quantity computed from the target ACF (zero beyond the last sample point), `iid` is 1/sqrt(P).

| preset | L/d = 1 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| target, Exp   | 0.781 | 0.382 | 0.220 | 0.119 | 0.057 | 0.039 |
| Exp150        | 0.797 | 0.387 | 0.246 | 0.158 | 0.104 | 0.059 |
| Exp300        | 0.784 | 0.379 | 0.225 | 0.155 | 0.114 | 0.103 |
| Exp500        | 0.784 | 0.385 | 0.253 | 0.196 | 0.150 | 0.138 |
| Exp1000       | 0.784 | 0.382 | 0.242 | 0.155 | 0.094 | 0.070 |
| target, Gauss | 0.860 | 0.318 | 0.171 | 0.092 | 0.047 | 0.036 |
| Gauss150      | 0.863 | 0.327 | 0.212 | 0.165 | 0.145 | 0.141 |
| Gauss300      | 0.859 | 0.322 | 0.189 | 0.132 | 0.120 | 0.117 |
| Gauss500      | 0.862 | 0.321 | 0.209 | 0.152 | 0.129 | 0.124 |
| Gauss1000     | 0.862 | 0.320 | 0.183 | 0.109 | 0.061 | 0.054 |
| target, Comb  | 0.860 | 0.397 | 0.227 | 0.123 | 0.058 | 0.040 |
| Comb150       | 0.858 | 0.403 | 0.244 | 0.185 | 0.121 | 0.111 |
| Comb300       | 0.859 | 0.391 | 0.251 | 0.200 | 0.169 | 0.162 |
| Comb500       | 0.858 | 0.395 | 0.264 | 0.194 | 0.152 | 0.141 |
| Comb1000      | 0.858 | 0.396 | 0.263 | 0.205 | 0.183 | 0.178 |
| iid           | 0.032 | 0.032 | 0.032 | 0.032 | 0.032 | 0.032 |

**Component A: the finite-sample mean of a correlated field (legitimate).** Up to L = 5·d every
table reproduces the target within 3 %. Positions inside a few decorrelation distances are not
independent, so their mean has a variance far above 1/P. A real correlated field does the same.

**Component B: quasi-DC sinusoids in the frequency tables (a defect of the tables).** Above
L = 10·d the tables stop following the target and settle on a floor that does not decay with the
size of the drop. Each table contains a cluster of sinusoids with `abs(f)·d < 0.01` (period above
100·d). Each of them adds the constant `a·cos(phi_n)` to the whole drop. With n_dc such sinusoids
the floor is `sqrt( n_dc / N )`:

| preset | n_dc | sqrt(n_dc/N) | measured std, L = 100·d | same, quasi-DC removed | target | next frequency, f·d |
|---|---|---|---|---|---|---|
| Exp150    | 1  | 0.082 | 0.060 | 0.042 | 0.040 | 0.017 |
| Exp300    | 3  | 0.100 | 0.104 | 0.033 | 0.040 | 0.025 |
| Exp500    | 10 | 0.141 | 0.138 | 0.034 | 0.040 | 0.016 |
| Exp1000   | 6  | 0.077 | 0.071 | 0.039 | 0.040 | 0.016 |
| Gauss150  | 3  | 0.141 | 0.141 | 0.034 | 0.036 | 0.034 |
| Gauss300  | 4  | 0.115 | 0.117 | 0.031 | 0.036 | 0.033 |
| Gauss500  | 7  | 0.118 | 0.124 | 0.042 | 0.036 | 0.091 |
| Gauss1000 | 2  | 0.045 | 0.055 | 0.033 | 0.036 | 0.026 |
| Comb150   | 1  | 0.082 | 0.111 | 0.076 | 0.040 | 0.017 |
| Comb300   | 8  | 0.163 | 0.162 | 0.033 | 0.040 | 0.013 |
| Comb500   | 9  | 0.134 | 0.141 | 0.050 | 0.040 | 0.012 |
| Comb1000  | 31 | 0.176 | 0.178 | 0.035 | 0.040 | 0.015 |

- The same thing seen in the ACF: the direction-averaged approximated ACF does not go to zero beyond
  the fitted range. For Comb1000 it is 0.034, 0.029, 0.030, 0.029 at 10·d, 20·d, 50·d, 100·d, which
  is n_dc/N = 0.031. Comb300: 0.045, 0.021, 0.022, 0.022 against n_dc/N = 0.027.
- Origin: the optimiser fits the ACF only on 0 to 5·d. Inside that window a constant is a cheap way
  to fit the slowly decaying tail, and the coarse search grid of `suosi_iteration_step` contains
  f = 0 exactly. Outside the window nothing penalises the constant.
- The quasi-DC sinusoids are not pure waste: dropping them and renormalising makes the fit inside
  0 to 5·d worse (Comb1000: 37.5 dB to 30.3 dB with 200 test directions, Comb300: 32.3 to 29.8 dB) and
  pushes the spread at L = 5·d below the target (Comb1000: 0.355 against 0.392). They stand for real
  low-frequency content that is lumped at f = 0 instead of being spread over small non-zero
  frequencies. They have to be replaced in the regeneration, not deleted from the existing tables.
- For the default Comb300 the consequence is a drop-wide offset with a standard deviation of 0.16
  on every generator, however large the drop. For shadow fading with sigma = 6 dB that is 1 dB.

Candidates that are **not** a cause:

- Dual mobility. The spread follows the exact formula in all cases (exp3). Different phases give the
  single-device spread; identical phases give a larger one because both devices sample the same map
  (L/d = 5: 0.512 against 0.386). The normalisation `1/sqrt(2 + 2·A(D))` keeps the pointwise variance
  at 1 for equal decorrelation distances (exact mean over 2000 pairs: 0.996 to 1.019).
- Single precision. The stored frequencies and phases are single, the evaluation is double.
  The effect on the mean is of the order 1e-7.
- The mean frequency vector of the tables is not zero (the tables are not symmetric), but this does
  not enter the mean.

### 3. Why it matters downstream

`gen_lsf_parameters` evaluates the 8 generators at all receiver positions of the builder, mixes them
with `sqrtm( lsp_xcorr )`, and computes `par · sigma + mu` in the log domain. A non-zero sample mean
of a generator therefore shifts the median of that parameter over the drop by `sigma · m`. The
quantity that the correction makes exact is the drop-wise median of DS, KF, SF, the angular
spreads and XPR (and of the absolute time of arrival).

Measured in `qd_builder` (100 UEs in 500 m × 500 m, 200 initialisations, Comb300):

| UMa LOS | DS | KF | SF | ASD | ASA | ESD | ESA | XPR |
|---|---|---|---|---|---|---|---|---|
| decorrelation distance [m] | 30 | 12 | 37 | 18 | 15 | 15 | 15 | 15 |
| rms mean, fresh phases        | 0.225 | 0.199 | 0.260 | 0.199 | 0.214 | 0.203 | 0.191 | 0.208 |
| rms mean, after `init_sos`    | 0.021 | 0.016 | 0.055 | 0.134 | 0.131 | 0.011 | 0.002 | 0.037 |
| rms offset of the parameter, fresh phases (dB or log10) | 0.150 | 0.70 | 0.93 | 0.058 | 0.045 | 0.084 | 0.032 | 0.83 |
| same, after `init_sos`                                   | 0.055 | 0.06 | 0.28 | 0.032 | 0.021 | 0.017 | 0.007 | 0.15 |

For UMa NLOS (all distances 50 m, DS 40 m) the rms mean falls from 0.26 to 0.29 without correction
to 0.013 to 0.070 after `init_sos`; the SF offset falls from 1.65 dB to 0.22 dB.

### 4. Closed form

Confirmed. A common offset `delta` on all phases of a column turns the mean into
`a · Re( exp( j·delta ) · Z )`, which is zero for `delta = pi/2 - arg(Z)` (and for that value
plus pi). No search is needed.

| case | expression | check |
|---|---|---|
| single device | `Z = sum_n z_n · exp( j·phi_n )` | residual mean at most 2.4e-15 in 90 runs |
| two devices, identical phases | same, with `z_n = (1/P) · sum_p w_p · ( exp( j·2·pi · f_n·ca_p ) + exp( j·2·pi · k·f_n·cb_p ) )`, `w_p = 1/sqrt( 2 + 2·A(D_p) )`, k = frequency scale of the second device | residual at most 2.9e-8; the limit is the single-precision ACF table used for `w_p` in the script, not the formula |
| two devices, different phases | mean = `( a/sqrt(2) ) · ( Re( exp( j·delta_a ) · Z_a ) + Re( exp( j·delta_b ) · Z_b ) )`. Two unknowns, one equation: a one-parameter family of solutions. Shifting only column a works only if `abs(Z_a) >= abs(Re(Z_b))`, which failed in 23 to 33 % of the realisations. The canonical solution is to zero each device on its own positions: `delta_a = pi/2 - arg(Z_a)`, `delta_b = pi/2 - arg(Z_b)` | residual at most 1.8e-15 |

Cost: about 0.05 to 0.2 ms against 8 ms (single), 16 ms (dual, identical) and 35 to 58 ms (dual,
different) for `remove_bias`, with P = 200 and N = 300 (one of the twelve timing rows showed 630 ms
for `remove_bias` and is left out as a timing outlier). In a batched function the `z_n` are computed
once per frequency table and each further generator costs N multiply-adds.

`remove_bias` against the closed form:

- When the search converges it lands on one of the two roots (offset difference below 0.04 rad).
- It does not always converge. Single device, 300 runs per row:

| L/d | rms mean before | median abs mean after | share above 1e-3 | share above 1e-2 | share worse than before | max abs mean after |
|---|---|---|---|---|---|---|
| 1   | 0.873 | 6.6e-4 | 37 % | 4.3 % | 2.0 % | 2.33 |
| 5   | 0.415 | 3.0e-4 | 21 % | 1.7 % | 1.3 % | 0.67 |
| 20  | 0.221 | 1.6e-4 | 13 % | 2.3 % | 2.0 % | 0.28 |
| 100 | 0.173 | 1.1e-4 |  9 % | 1.3 % | 1.3 % | 0.35 |

  The search compares each trial with the previous trial instead of the best one and never steps
  back, so about 1 to 2 % of the runs end worse than they started. This is why the rms mean after
  `init_sos` in section 3 is 0.01 to 0.07 and not zero, even for the generators that are not swapped.
- With different phases `remove_bias` zeroes only the sum. The per-device means stay at an rms of
  0.42, 0.23, 0.11 (L/d = 1, 5, 20) and cancel each other. With the closed form per column both are zero.

### 5. Consistency of the current use in `init_sos`

- `remove_bias` runs directly after construction, when both phase columns are still identical. So
  the branches for different phases, including the random permutation of `ca`, are never reached from
  QuaDRiGa-NG. The permutation draws from the global random stream; a direct call with different
  phases and no `cb` is reproducible only through the global seed.
- The reciprocity overwrite comes after the correction. For DS, KF, SF and XPR it is a no-op. For the
  angular spreads it puts the corrected phases of ASA into column 1 of ASD and vice versa (same for
  ESD and ESA). A correction is valid only for the frequencies and positions it was computed with:
  - single mobility: only column 1 is used. The correction survives if the two generators have the
    same decorrelation distance (ESD/ESA in UMa LOS, both 15 m: rms 0.011 and 0.002) and is largely
    lost otherwise (ASD/ASA, 18 m and 15 m: rms 0.134 and 0.131 against 0.20 uncorrected).
  - dual mobility (steps of `init_sos` replayed on `qd_sos` objects, UMa LOS distances): rms mean of
    ASD, ASA, ESD, ESA 0.106, 0.089, 0.063, 0.067 against 0.034 to 0.060 for the others and 0.25 to
    0.31 uncorrected.
- With reciprocity in dual mobility an exact correction of all four angular spreads by phase offsets
  does not exist: phase column X is evaluated at `ca` with the ASD frequencies in one generator and
  at `cb` with the ASA frequencies in the other, which gives two conditions for one offset.
  In single mobility it exists if the swap is done first and the correction second.

### 6. Does the correction change the statistics?

Yes. A common phase offset replaces the field `s` by `cos(delta)·s - sin(delta)·q`, where `q` is the
quadrature field (sines instead of cosines), which has the same ACF and is uncorrelated with `s`.
Choosing `delta` so that the sample mean vanishes is, in the Gaussian approximation, the same as
drawing the field conditioned on a zero sample mean:

- pointwise variance `1 - c(x)^2 / v`, with `v = Var[m]` and `c(x) = (1/P) · sum_p R_sos( x - x_p )`
- covariance between two positions `R_sos( x - y ) - c(x)·c(y) / v`
- the average loss of variance over the positions is at least `v`

Measured (Comb300, 4000 phase sets, pairs of positions at distance d inside the set):

| P | L/d | std of mean | variance after offset | predicted | variance after subtracting the mean | correlation at d: none | offset | quasi-DC removed |
|---|---|---|---|---|---|---|---|---|
| 200 | 0.5 | 0.912 | 0.150 | 0.152 | 0.172 | 0.39 | -0.49 | 0.38 |
| 200 | 1   | 0.826 | 0.307 | 0.305 | 0.319 | 0.40 | -0.27 | 0.39 |
| 200 | 2   | 0.647 | 0.571 | 0.569 | 0.586 | 0.39 | 0.07 | 0.38 |
| 200 | 5   | 0.392 | 0.836 | 0.834 | 0.846 | 0.40 | 0.27 | 0.38 |
| 200 | 10  | 0.263 | 0.927 | 0.930 | 0.935 | 0.40 | 0.35 | 0.38 |
| 200 | 20  | 0.214 | 0.953 | 0.953 | 0.953 | 0.40 | 0.36 | 0.38 |
| 200 | 100 | 0.178 | 0.969 | 0.969 | 0.970 | 0.40 | 0.38 | 0.38 |
| 3   | 100 | 0.608 | 0.633 | 0.629 | 0.626 | | | |
| 10  | 100 | 0.348 | 0.871 | 0.881 | 0.882 | | | |
| 30  | 100 | 0.234 | 0.942 | 0.944 | 0.945 | | | |

- For large drops (L above 20·d, many UEs) the loss is 3 to 5 %, and most of it is the quasi-DC
  component, which should not be there in the first place.
- For small drops the correction removes most of the variance: 43 % at L = 2·d, 69 % at L = d. All
  UEs sit in one coherence patch and should share an offset; the correction deletes it. With few
  UEs it removes about 1/P: 37 % for 3 UEs (`init_sos` applies it from 3 positions upwards).
- The offset and the subtraction of the sample mean give the same variance and correlation. The
  difference is that the offset stays inside the generator: the field is still a valid realisation
  everywhere and later evaluations at other positions are consistent with it. Far from the drop
  the variance is 1 again (0.98 to 1.02). The zero mean holds only for the exact position set: 50
  other positions in the same square have an rms mean of 0.07 to 0.16.
- Removing the quasi-DC sinusoids leaves the pointwise variance at 1, brings the spread of the mean
  for L = 100·d from 0.178 to 0.075 (iid for P = 200: 0.071), and changes the correlation at d from
  0.40 to 0.38 (target 0.368).

### 7. Recommendation for the library

1. **Fix component B in the tables (WI-5).** The regenerated presets must not contain quasi-DC
   sinusoids. Then the remaining sample mean is component A, which is correct statistics.
2. **Provide the correction as one closed-form function, never applied implicitly.** Working name
   `sos_remove_bias`; name and argument list are settled in the header draft.
   - inputs: frequencies `[N, d]`, phases `[N, n_gen]` (updated in place), positions of the first
     device; optional positions of the second device, optional second phase matrix, optional ACF table
     and frequency scale of the second device
   - outputs: optional applied offsets per generator (and per device), optional sample mean before the
     correction, so that a caller can judge the size of the effect
   - three modes, chosen by the arguments: single device; two devices on the same map (one phase
     matrix, weights from the ACF, one offset); two devices with different phases (each device
     zeroed on its own positions, two offsets). No iteration, no random permutation.
   - batched like `sos_val`: the `z_n` are computed once, each generator costs N multiply-adds
   - the documentation states what the function does to the statistics (section 6) and that it is
     meant for drops that are large against the decorrelation distance
3. **No function for the search of `remove_bias`.** The closed form replaces it. Results differ from
   MATLAB where the search did not converge (decision D2 allows this).
4. **Hand-over note for the QuaDRiGa-NG integration** (not library work): set up reciprocity first and
   correct afterwards; in dual mobility the angular spreads cannot all be corrected by offsets; decide
   there whether the correction is still wanted once the tables are fixed, and from how many positions.

### 8. Consequences for the preset regeneration (WI-5)

- Add a fourth measure to the three of `calc_mse`: the behaviour beyond the fitted range. Proposed:
  the number of sinusoids with `abs(f)·d < 0.01` (must be 0), and the direction-averaged approximated
  ACF at 10·d, 20·d, 50·d and 100·d (must stay at the level of the fit error, not at n_dc/N).
  Record both for the 12 existing tables as baseline; the numbers are in section 2.
- Mechanism to try first: fit against the ACF extended with zeros beyond its last sample point (for
  example to 4 times the range), so that a constant is penalised. Fallback: a lower bound on
  `abs(f)` in the search. The stored ACF definition (200 points to 49.75 m) stays as in decision D7.
- Possible tension with decision D7: the existing tables buy part of their fit inside 0 to 5·d with
  the quasi-DC sinusoids (up to 7 dB for Comb1000 when they are simply dropped). A table without them
  may not reach the old numbers on all three measures. If that happens it is reported with the
  numbers, as work item 5 prescribes; nothing is shipped silently.

### 9. Other observations (for the header draft, not part of the bias)

- Dual mobility with two decorrelation distances and identical phases: `val` scales the frequencies
  of the second device and still normalises with `A( abs( cb - ca ) )`. The correlation between the two
  terms is then `R_sos( ca - k·cb )`, which depends on the absolute positions. Exact pointwise variance
  over 2000 pairs: 0.54 to 1.91 for k = 0.5 and 0.49 to 1.58 for k = 4 (mean 0.71 at L = d). To settle
  at stop 2 how the library treats this case.
- The Comb300 table gives a correlation of 0.40 at the decorrelation distance; the target is 0.368.

### 10. First-principles check without sinusoids (added after STOP 1, exp7)

Question from Stephan: is the random offset per initialisation a conceptual problem of the method?

A Gaussian field was drawn directly from its covariance matrix (Cholesky, `R = exp(-r/d)`, no
sinusoids, P = 500, 4000 draws) and its sample mean compared with the pair sum
`sqrt( (1/P^2) · sum_pq R( x_p - x_q ) )`, the area rule `sqrt( 1/P + 2·pi·d^2 / L^2 )` and two tables:

| L/d | Cholesky field, measured | pair sum | area rule | Exp1000 table | Exp300 table | iid 1/sqrt(P) |
|---|---|---|---|---|---|---|
| 1   | 0.785 | 0.781 | (not valid) | 0.784 | 0.784 | 0.045 |
| 5   | 0.381 | 0.380 | 0.503 | 0.381 | 0.378 | 0.045 |
| 10  | 0.226 | 0.228 | 0.255 | 0.248 | 0.231 | 0.045 |
| 20  | 0.127 | 0.125 | 0.133 | 0.157 | 0.157 | 0.045 |
| 50  | 0.066 | 0.065 | 0.067 | 0.099 | 0.119 | 0.045 |
| 100 | 0.051 | 0.051 | 0.051 | 0.077 | 0.110 | 0.045 |

- The exact field has the same random drop mean as the sum of sinusoids up to L = 5·d and nearly the
  same at 10·d. The offset is a property of any field with this ACF, not of the method.
- The tables exceed the exact field from about L = 10·d upwards. That excess is component B.
- For the builder measurement of section 3 (UMa NLOS, SF, d = 50 m, L = 10·d, P = 100) the exact
  field would give about 0.25; measured 0.28. About three quarters of the variance of the observed
  offset is component A.

### Not done in WI-1

- Octave was not used; all numbers are from MATLAB.
- Positions were planar (z = 1.5 m) and uniformly random in a square. Other geometries were not run.
- The absolute-TOA generator was not measured separately; it goes through the same code path as a
  single large-scale generator.


## WI-2 Header draft (2026-10-09) — STOP 2

Answers at STOP 1 (Stephan): agreed that a regenerated table without quasi-DC sinusoids is reported
with numbers if it misses an old fit value; agreed that with different phases each device is zeroed
separately; the case of two decorrelation distances is raised here.

Draft: `include/quadriga_sos.hpp`. Declarations only; not included from `quadriga_lib.hpp`; compiles with
`g++ -std=c++17 -Wall -Wextra -Wpedantic -fsyntax-only` without diagnostics. Nothing is implemented.

### Functions and MATLAB usage lines

| C++ function | MATLAB usage |
|---|---|
| `sos_preset` | `[ freq, acf, dist ] = quadriga_lib.sos_preset( name, dist_decorr )` |
| `sos_init_phases` | `phase = quadriga_lib.sos_init_phases( n_coeff, n_gen, seed )` |
| `sos_acfi` | `val = quadriga_lib.sos_acfi( acf, dist, d )` |
| `sos_val` | `val = quadriga_lib.sos_val( freq, phase, pos, pos_b, phase_b, acf, dist, freq_scale_b, uniform, use_kernel )` |
| `sos_map` | `map = quadriga_lib.sos_map( freq, phase, x, y, z, uniform, use_kernel )` |
| `sos_randn`, `sos_rand` | `val = quadriga_lib.sos_randn( dist_decorr, pos, pos_b, preset, seed )` |
| `sos_randi` | `val = quadriga_lib.sos_randi( dist_decorr, pos, imax, pos_b, preset, seed )` |
| `sos_remove_bias` | `[ phase_out, phase_b_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( freq, phase_in, pos, pos_b, phase_b_in, acf, dist, freq_scale_b )` |
| `sos_acf_2d` | `[ acf_2d, dist_2d ] = quadriga_lib.sos_acf_2d( acf, dist )` |
| `sos_acf_approx` | `[ acf_approx, dist_2d ] = quadriga_lib.sos_acf_approx( freq, dist )` |
| `sos_calc_mse` | `[ mse, mse_core, mse_all, acf_tail, test_dir ] = quadriga_lib.sos_calc_mse( freq, acf, dist, n_dir )` |
| `sos_acf_estimate` | `[ acf_est, dist_est, acf_est_dual ] = quadriga_lib.sos_acf_estimate( freq, phase, bins, n_pos, acf, dist, seed )` |
| `sos_generate` | `[ freq, ase, test_dir ] = quadriga_lib.sos_generate( acf, dist, n_coeff, n_dim, uniform_smp, n_dir, random_init, freq_min, freq_max, seed )` |

All wrappers need only the generic helpers of `mex_helper_functions.hpp`; the 4-D output of `sos_map`
uses the `std::vector<arma::Cube>` overload of `qd_mex_init_output`.

### Points of plan section 6, with the choice made

| Point | Choice | Reason |
|---|---|---|
| Position layout | `[n_pos, n_dim_pos]`, rows are positions | Convention of the newer tool functions. The columns x, y, z are contiguous, which is what the kernel needs. The output `[n_pos, n_gen]` then has one contiguous column per generator. `qd_sos` in MATLAB uses `[3, n_pos]` and returns `[n_gen, n_pos]`; the MATLAB shell transposes. |
| `acfi` | Own function `sos_acfi` | `gen_lsf_parameters` needs it, and the clipping rule is then the same one that `sos_val` and `sos_remove_bias` use. |
| Amplitude | No argument | `sqrt(2/N)` always; no use for another value was found. |
| `UniformForced` | Dropped | No caller in QuaDRiGa-NG. It depends on the sample set. `set_scenario` does the same remapping itself with `acdf` and `interp`, which exist in the library. `uniform` is a `bool`. |
| Different decorrelation distances in one call | Only in `sos_rand`, `sos_randn`, `sos_randi`, which loop over the generators; entries with equal distance are evaluated in one kernel call | The kernel shares sin and cos only for shared frequencies. |
| Dimensions | `freq` has 1 to 3 columns; positions may have fewer columns (zero padding), more is an error | As in MATLAB. Presets are 3-D. |
| Seeds | `arma::sword seed = -1` on every function that draws; `std::mt19937_64`, 53-bit conversion, no distribution classes | Same argument and meaning as `get_channels_ieee_indoor`. -1 uses the system random device, so "no global state" holds; a seed >= 0 is reproducible across platforms. |
| Documentation section | Proposed: "Sum-of-sinusoids model" | To be confirmed. |
| Aggregate header | Add `#include` to `quadriga_lib.hpp` after `quadriga_tools.hpp` once the name is fixed | As the other headers. |

### Further choices

- **Mode of the two-device case follows from the arguments.** `phase_b` omitted means same map (ACF
  normalisation), `phase_b` given means different maps (`1/sqrt(2)`). MATLAB compares the two phase
  columns with a tolerance of 1e-6. A batch that mixes both kinds (for example cluster delays and
  cluster angles in `path_sos`) needs two calls.
- **`freq_scale_b != 1` requires `phase_b`.** For one map with two decorrelation distances the MATLAB
  result has no consistent ACF (pointwise variance 0.49 to 1.91, WI-1 section 9). With different maps
  the variance is exactly 1. The single MATLAB caller is `set_scenario` (LOS state for different
  decorrelation distances at transmitter and receiver); it would draw two phase sets there.
- **`sos_rand`, `sos_randn`, `sos_randi` as three functions** in one source file, as in MATLAB, instead
  of one function with a switch: each wrapper stays a plain argument conversion.
- **`sos_remove_bias`** as agreed at STOP 1. Of the two roots it takes the offset with the smaller
  magnitude. Outputs `offset` and `mean_before` are optional.
- **`sos_calc_mse`** returns the three MATLAB measures plus `acf_tail`, the fourth measure agreed at
  STOP 1. The MATLAB outputs `Ro` and `Ri` are `sos_acf_approx` and `sos_acf_2d` and are not repeated.
- **`sos_generate`** gets `freq_min` and `freq_max`. Together with a zero-extended ACF passed by the
  caller this covers both mechanisms against quasi-DC sinusoids without fixing one in the interface.
  Plotting is dropped.
- **`sos_acf_estimate`** takes the phases of one generator as input (MATLAB uses the object's phases).
- **Argument order.** The mandatory output comes before the optional inputs (for example `val` in
  `sos_val`), so that the optional inputs keep their default `nullptr`. This differs from the order
  "optional inputs, then outputs" of the spec; `fast_geo2cart` does the same.
- **No `use_kernel`** on the offline tools and on the three random functions (always auto).

### Open points for Stephan

1. Header name (D12): `qd_sos.hpp` as named, or `quadriga_sos.hpp` like the other public headers.
2. Section name "Sum-of-sinusoids model".
3. Reading of D9: the review points A and B per function are not stops (already stated in the task).
4. Number of sinusoids in the regenerated presets: the regular sphere packing of `qf.pack_sphere` returns
   151, 301, 500 and 998 points for 150, 300, 500 and 1000. D7 asks for the same counts and for uniform
   sphere sampling. Options: (a) accept 151 / 301 / 500 / 998 under the old names, (b) use a spiral
   (Fibonacci) lattice, which gives any exact count, for the presets and keep the packing for the test
   directions.
5. `freq_scale_b` only with `phase_b` (above).
6. Dropping `UniformForced`.

### Answers at STOP 2, first round (2026-10-09)

- Header name: `quadriga_sos.hpp` (decided; draft renamed).
- `UniformForced`: dropped, provided the API is useful on its own without it. It only forces the
  histogram of one sample set to be flat, which a caller can do with a rank transform.
- Still open after this round: section name, number of sinusoids per preset, treatment of two
  decorrelation distances.

### Answers at STOP 2, final (2026-10-09)

- Section name: "Sum-of-sinusoids random number generator".
- Presets: 128, 256, 512 and 1024 sinusoids (names `Exp128` ... `Comb1024`), directions from a spiral
  (Fibonacci) lattice. This replaces the "same 12 names" of decision D7. `qf.pack_sphere` is not ported;
  the lattice is also used for the test directions, on the upper hemisphere because a direction and its
  opposite are equivalent for a cosine.
- Two decorrelation distances: two maps. `sos_val` rejects a frequency scale without `phase_b`;
  `sos_rand`, `sos_randn`, `sos_randi` take an optional `dist_decorr_b` and draw the second phase set.
- `UniformForced` dropped.
- Header finalised as `include/quadriga_sos.hpp` and included from `quadriga_lib.hpp`.


## WI-3 Core runtime functions

Baseline before any change: `make avx2=ON matlab=ON`, then tests enabled: 930 Catch2 test cases, all
passed, no compiler warnings.

Deviation from the plan order: `sos_rand`, `sos_randn`, `sos_randi` need the presets and were written
after WI-5. Procedure note: the stubs of all functions carry the final input validation, so the
"input validation" test cases pass against the stub by design; every other test case contains at least
one check that zeros cannot satisfy.

Shared internals (in `src/quadriga_lib_helper_functions.hpp`): `qd_sos_rng`, `qd_sos_uniform` (random
stream), `qd_sos_acf_interp`, `qd_sos_check_acf`, `qd_sos_equally_spaced`, `qd_sos_directions`.

### sos_init_phases

| Test case | Asserted | Oracle |
|---|---|---|
| phases follow the documented random stream | values, 1e-15 | `std::mt19937_64`, upper 53 bits, written in the test |
| first columns do not depend on n_gen | bit-identical prefix | invariance |
| seed reproducible, seeds differ, seed 0 | identical / different | invariance, stream oracle |
| seed -1 | two calls differ, range | property |
| uniform in [-pi, pi) | mean, variance pi^2/3, 10-bin histogram, 5 sigma | moments of the uniform distribution |
| pre-sized output | written in place, wrong size replaced | stream oracle |
| float | within 2.5e-7 of double, below pi | rounding bound |
| input validation | n_coeff, n_gen, seed < -1 | |

Run against the stub: 7 of 8 failed as expected (zeros), validation passed. After implementation: all
pass. Moments with 200,000 values: mean 0.0010 (sigma 0.0041), variance 3.2920 (expected 3.2899).
Decision: a double just below pi that rounds to pi in float is returned as -pi, so that the documented
interval holds for both types.

### sos_acfi

| Test case | Asserted | Oracle |
|---|---|---|
| interpolation and clipping | 9 values, 1e-14 | hand-computed on acf = {1, 0.5, 0.2}, dist = {0, 1, 3} |
| sizes, empty query, pre-sized output | sizes, in-place | hand values |
| float | 1e-6 | hand values |
| input validation | acf, dist (size, start, order, repeated), negative d | |

Stub: 3 of 4 failed as expected. After implementation: all pass.

### sos_val (GENERIC and AVX2 kernel)

Kernel design: the vector lanes of the AVX2 kernel run over 8 positions. For one block it computes
sine and cosine of all sinusoids once (argument in cycles in double, reduced by rounding to the nearest
integer, then `_fm256_sincos256_pd`), and then accumulates 4 generators at a time: single-precision
partial sums over 16 sinusoids, added in double. An incomplete last block is padded and the padding
discarded, so there is no scalar remainder loop. OpenMP runs over the blocks. Working set per block for
N = 1024: 64 kB of sine/cosine plus 4 weight columns. GENERIC is the same algorithm in scalar double.

| Test case | Asserted | Oracle |
|---|---|---|
| single sinusoid on the axis | multiples of sqrt(2) | hand |
| brute-force sum, 6 generators, 11 positions | 1e-12 GENERIC, 4e-7·sqrt(2N) AVX2 | scalar transcription of the formula |
| shift by one wavelength | unchanged, half wavelength flips the sign | periodicity |
| batched column equals single call | bit-identical | invariance |
| value independent of the other positions | 210 placements (1 to 20 positions, every slot), bit-identical | invariance |
| AVX2 against GENERIC far from the origin | offsets 0 to 1e6 m | kernel parity, brute force |
| missing position columns | equal to explicit zeros; 1-D and 2-D frequencies | invariance, brute force |
| statistics over 20,000 phase sets | mean 0, variance 1, covariance = (1/N)·sum cos, 5 sigma | exact expectations |
| two devices, same map | normalised sum; swap symmetric; same place gives single value | brute force, own interpolation |
| same map keeps unit variance | displacements 0 to 40 m | ACF of the sinusoids as table |
| two devices, different maps | second phases, scale 1 and 2.5, 1/sqrt(2); table ignored | brute force |
| Uniform | equals 0.5·erfc(-s/sqrt(2)); inside (0,1); 10-bin histogram | definition, flatness |
| empty optional matrices, pre-sized output | same as omitted; in place | invariance |
| float against double | 1e-3 (input rounding, see comment in the test) | parity |
| repeatable, independent of the thread count | bit-identical with 1 thread | invariance |
| automatic kernel | equals AVX2 when available | |
| input validation | every argument; failed call leaves the output untouched | |

Stub: 16 of 17 failed as expected. After implementation: all pass (279 assertions in the 29 test cases
of the three functions).

| Diagnostic | Result |
|---|---|
| AVX2 minus GENERIC, N = 300, offset 0 / 1e2 / 1e3 / 1e4 / 1e5 / 1e6 m (max abs) | 3.7e-7 / 3.0e-7 / 3.6e-7 / 4.2e-7 / 3.2e-7 / 3.2e-7 |
| same, rms | 1.1e-7 to 1.2e-7 |
| GENERIC against brute force | 1.2e-14 |
| float against double (inputs rounded to float) | 1.1e-5 |
| variance, same map, displacement 0 / 0.5 / 3 / 10 / 25 / 40 m | 0.991 / 1.003 / 1.006 / 1.018 / 1.004 / 0.998 |

Tolerances: 1e-12 for GENERIC; `4e-7 · sqrt(2·N)` for AVX2 (worst case of N terms with a single-precision
sine or cosine each; the observed error is a factor 25 below it for N = 300). The documentation states
this bound. Neither was changed after the first run.

Open point: the float instantiation rounds positions to single precision on input; at 10 km this is a
position error of 1 mm. Callers with large coordinates should use double.


## WI-3 (continued): sos_map

| Test case | Asserted | Oracle |
|---|---|---|
| cubes in meshgrid order, one per generator | every element of a 2 x 3 x 4 grid, 1e-12 | scalar transcription of the formula at the grid point |
| omitted grid vectors | x only, x and y, x and z, empty vectors, 1-D frequencies | same oracle |
| Uniform, kernels, float | erfc mapping; auto kernel within the AVX2 bound; float 1e-3 | definition, parity |
| pre-sized output | cubes written in place; wrong vector length and cube size replaced | invariance |
| input validation | freq, phase, x, y and z against n_dim, use_kernel | |

Stub: 4 of 5 failed as expected. After implementation: all pass (114 assertions).


## WI-4 ACF tools and optimiser

### sos_acf_2d, sos_acf_approx

| Test case | Asserted | Oracle |
|---|---|---|
| acf_2d: grid values | 9 probes incl. on, between and beyond sample points, symmetry, axis | hand-computed, 3-point table |
| acf_2d: optional axis, pre-sized, float, validation | | |
| acf_approx: single 3-D sinusoid | all elements of the three planes, row/column assignment | closed-form cosine |
| acf_approx: several sinusoids, 1-D / 2-D / 3-D | values, point symmetry, shapes | scalar mean of cosines |
| acf_approx: pre-sized, float, validation | | |

Stubs: 2 of 3 and 3 of 4 failed as expected. After implementation: all pass (30 and 42 assertions).
Implementation note: the planes are computed as the difference of two matrix products
(cos(a+b) = cos a cos b - sin a sin b), parallel over the columns.

### sos_calc_mse

| Test case | Asserted | Oracle |
|---|---|---|
| directional error, 2-D and 3-D, 1 / 7 / 10 directions | value within 1e-9 dB; test directions | brute force over the documented directions |
| grid errors, 2-D and 3-D | mse_core, mse_all within 1e-9 dB | brute force over the planes, own radial interpolation and core mask |
| 1-D exact target | mse and mse_core above 250 dB, mse_all against closed form | limiting case |
| tail measure | values within 1e-12; plateau 0.25 for 10 of 40 sinusoids at zero frequency; all-zero table gives 1 | brute force, construction |
| optional outputs alone and together, float | identical values; float within 1e-3 dB | invariance |
| input validation | freq, acf, dist (unequal spacing), n_dir | |

Stub: 5 of 6 failed as expected. After implementation: all pass (185 assertions).
Cross-check against MATLAB: `calc_mse` of the old Comb300 table gives 33.78 dB (core) and 26.41 dB (all);
`sos_calc_mse` on the same table gives 33.78 and 26.41. The directional value differs slightly (32.29 dB
with `pack_sphere(200)`, 32.36 dB with 100 lattice directions) because the test directions differ.

### sos_acf_estimate

| Test case | Asserted | Oracle |
|---|---|---|
| equals the documented procedure, one and two devices | every bin within 1e-10, NaN pattern | step-by-step repetition in the test (positions from the documented stream, bins, pair selection, correlation) |
| follows the ACF of the sinusoids | 16 bins within 0.2 of exp(-2 pi^2 sigma^2 d^2) | frequencies drawn from an isotropic Normal distribution |
| sparse bins | NaN with fewer than 6 pairs | rule |
| seeds, optional outputs, pre-sized, float | | invariance |
| input validation | | |

Stub: 4 of 5 failed as expected. After implementation: all pass (105 assertions).
Decisions: the position stream is documented (matrix [2 n_pos, 3] filled column by column), which makes
the exact oracle possible; the GENERIC kernel is used so that the result does not depend on the CPU; the
pair loop needs no distance matrix, so the memory no longer grows with the square of the positions. The
MATLAB output `De_dual` was computed but never returned and is not ported.

### sos_generate

Differences from the MATLAB `generate` (decision D2 allows changes of the algorithm):

- The residual a sinusoid is fitted to is taken along its own direction and is exact: target minus
  the sum of all other sinusoids divided by N. MATLAB used the coordinate axis closest to the direction
  and the mean of the other N-1 sinusoids. With the exact residual a target that the sinusoids can
  represent is a fixed point of the iteration, which the recovery test needs.
- The step refinement steps back properly on a rejected step (MATLAB moved without evaluating).
- Double precision; seeded stream; the random choice between the two equal minima at +f and -f is a
  random sign.
- Frequency limits `freq_min`, `freq_max`. Without random start values the sinusoids start at the lower
  limit instead of 0.
- Stop criterion: additionally a pass that gains less than a factor of 1e-6, and at most 1000 passes.
- For equally spaced sample points the cosines over the distance are advanced by a plane rotation per
  sample instead of one cosine call (about 50 times faster; the first version took 100 s for 256
  sinusoids, the current one 3 s).

| Test case | Asserted | Oracle |
|---|---|---|
| 1-D target made of two sinusoids | both frequencies within 1e-5 / m, error below 1e-6, 8 runs | recovery |
| reported error, 12 configurations | last entry of `ase` equals the error of the returned table (1e-10); non-increasing; below 1e-2 | brute force over the test directions |
| regular directions | every row is parallel to its lattice direction | documented lattice |
| frequency limits | all magnitudes inside [0.02, 0.2] / m; default stays below 15 / max(dist) | invariant |
| seed, pre-sized output, float | identical reruns; float reaches the same quality | invariance |
| input validation | | |

Stub: 5 of 6 failed as expected. First run after implementation: 2 cases failed, both through faults of
the implementation, not of the expectations:

1. `ase` increased by one unit in the last place, because the sum over the sinusoids was rebuilt at the
   start of each pass and the final entry recomputed. Fixed: the sum is updated incrementally and the
   tracked error is reported. The difference to an independent evaluation stays below 1e-10.
2. Without random start values, sinusoids that never received an accepted update stayed at frequency 0
   and violated `freq_min`. Fixed: they start at the lower limit.

After the fixes: all pass (4258 assertions). Recovery: 0.0400000 and 0.1300000 in all 8 runs, final
error 1e-30, 3 to 4 passes. No expectation or tolerance was changed.

Observation: without a lower limit the port also places sinusoids at zero frequency (smallest magnitude
0.0000 in the limit test), so the quasi-DC cluster of WI-1 is a property of the method, not of the old code.


## WI-5 Presets

### ACF formulas

Exp: exp(-D/d); Gauss: exp(-(D/d)^2); Comb: Gauss up to D = d, Exp beyond. Checked against the tables
stored in `set_acf.m`: the stored values are the formulas rounded to 5 significant digits (largest
absolute deviation 8e-6, 1.1e-5, 1.0e-5). The library computes the formulas.

### Baseline: the 12 old tables, measured with the ported functions

`sos_calc_mse`, 100 test directions for the directional value; tail = approximated ACF at 10, 20, 50 and
100 decorrelation distances; n_dc = sinusoids with abs(f)·d < 0.01; R(d) = ACF at the decorrelation
distance, mean of the three axes (target 0.368).

| table | N | mse_dir | mse_core | mse_all | tail 10d | 20d | 50d | 100d | n_dc | R(d) |
|---|---|---|---|---|---|---|---|---|---|---|
| Exp150 | 150 | 27.20 | 28.71 | 25.35 | 0.0203 | 0.0072 | 0.0110 | 0.0009 | 1 | 0.4138 |
| Exp300 | 300 | 30.59 | 31.74 | 27.34 | 0.0225 | 0.0013 | 0.0084 | 0.0062 | 3 | 0.3817 |
| Exp500 | 500 | 32.80 | 34.02 | 27.91 | 0.0218 | 0.0205 | 0.0159 | 0.0211 | 10 | 0.3627 |
| Exp1000 | 1000 | 35.53 | 36.86 | 31.04 | 0.0188 | 0.0058 | 0.0048 | 0.0022 | 6 | 0.3819 |
| Gauss150 | 150 | 29.39 | 31.04 | 24.37 | 0.0323 | 0.0225 | 0.0089 | 0.0150 | 3 | 0.3816 |
| Gauss300 | 300 | 32.02 | 34.19 | 27.93 | 0.0134 | 0.0176 | 0.0137 | 0.0140 | 4 | 0.3651 |
| Gauss500 | 500 | 34.14 | 35.97 | 29.70 | 0.0184 | 0.0158 | 0.0133 | 0.0091 | 7 | 0.3671 |
| Gauss1000 | 1000 | 37.77 | 39.52 | 31.08 | -0.0041 | 0.0085 | 0.0003 | -0.0002 | 2 | 0.3728 |
| Comb150 | 150 | 29.33 | 30.49 | 24.23 | 0.0314 | 0.0155 | 0.0033 | 0.0035 | 1 | 0.4104 |
| Comb300 | 300 | 32.36 | 33.78 | 26.41 | 0.0552 | 0.0295 | 0.0247 | 0.0244 | 8 | 0.3933 |
| Comb500 | 500 | 34.48 | 35.85 | 27.69 | 0.0290 | 0.0192 | 0.0134 | 0.0196 | 9 | 0.3987 |
| Comb1000 | 1000 | 37.27 | 38.73 | 28.54 | 0.0345 | 0.0278 | 0.0305 | 0.0270 | 31 | 0.3960 |

### Choice of the generation settings

Explored with 256 sinusoids (numbers are means over 3 seeds unless stated):

- **10 test directions over-fit.** The error on the 10 directions of the search reached 39.7 dB while the
  error on 100 other directions was 30.9 dB and on the planes 29.1 dB. With 100 or more directions the
  three values agree within 1 to 2 dB. 300 directions were chosen.
- **Fitting a zero-extended ACF does not work.** With the ACF continued to 4 times the range the tail is
  flat, but the fit inside the range suffers: R(d) = 0.47 instead of 0.37, mse_core 27 dB.
- **A lower frequency limit works.** With abs(f)·d >= 0.05 (wavelength at most 20 decorrelation distances,
  4 times the fitted range) there are no slow sinusoids, the tail falls below 0.01, mse_all improves by
  0.5 to 1 dB and mse_core does not change within the spread over seeds. 0.02 leaves a tail of 0.02 to
  0.03; 0.1 starts to cost fit inside the range.

Settings for all 12 tables: `sos_generate` with the 200-sample ACF (0 to 49.75 m, d = 10 m),
`n_dim = 3`, `uniform_smp = true`, `n_dir = 300`, `random_init = true`, `freq_min = 0.005`,
`freq_max = 15 / 49.75`. The seeds 1 to 48 were run for every table and the seed with the largest
smallest margin over the comparable old table on the three error measures was taken. Run time per
table: 1 s (128) to 26 s (1024) on one core; all 576 runs took about 3 minutes on 32 cores.
Tool: `scratchpad/presets/gen.cpp` (`gen make <type> <N> 1 0.05 300 <seed> <file>`).

### Result

Each new table is compared with the old table of the nearest size (128 with 150, 256 with 300, 512 with
500, 1024 with 1000). Margins are new minus old in dB.

| table | seed | passes | mse_dir | mse_core | mse_all | max abs tail | n_dc | R(d) | against | d_dir | d_core | d_all | mse_core over 48 seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Exp128 | 39 | 11 | 27.88 | 28.06 | 24.85 | 0.0063 | 0 | 0.3923 | Exp150 | +0.68 | -0.65 | -0.50 | 26.59 to 28.06 |
| Exp256 | 3 | 16 | 31.25 | 31.62 | 27.34 | 0.0108 | 0 | 0.3860 | Exp300 | +0.66 | -0.12 | +0.00 | 30.51 to 31.62 |
| Exp512 | 2 | 18 | 34.41 | 34.95 | 30.70 | 0.0049 | 0 | 0.3556 | Exp500 | +1.61 | +0.93 | +2.79 | 34.03 to 34.95 |
| Exp1024 | 5 | 28 | 38.16 | 38.29 | 32.59 | 0.0120 | 0 | 0.3659 | Exp1000 | +2.63 | +1.43 | +1.55 | 37.57 to 38.42 |
| Gauss128 | 13 | 16 | 32.75 | 31.40 | 24.80 | 0.0072 | 0 | 0.3759 | Gauss150 | +3.36 | +0.36 | +0.43 | 28.33 to 31.40 |
| Gauss256 | 32 | 14 | 35.79 | 35.20 | 28.66 | 0.0051 | 0 | 0.3680 | Gauss300 | +3.77 | +1.01 | +0.73 | 32.74 to 36.85 |
| Gauss512 | 6 | 17 | 41.16 | 39.62 | 31.53 | 0.0062 | 0 | 0.3665 | Gauss500 | +7.02 | +3.65 | +1.83 | 36.85 to 39.62 |
| Gauss1024 | 21 | 22 | 45.47 | 43.77 | 33.96 | 0.0115 | 0 | 0.3659 | Gauss1000 | +7.70 | +4.25 | +2.88 | 40.24 to 43.93 |
| Comb128 | 13 | 7 | 31.08 | 30.20 | 24.67 | 0.0075 | 0 | 0.4022 | Comb150 | +1.75 | -0.29 | +0.44 | 28.03 to 30.20 |
| Comb256 | 25 | 15 | 35.41 | 34.75 | 28.22 | 0.0047 | 0 | 0.3989 | Comb300 | +3.05 | +0.97 | +1.81 | 31.80 to 34.75 |
| Comb512 | 37 | 16 | 38.90 | 38.92 | 30.93 | 0.0077 | 0 | 0.3957 | Comb500 | +4.42 | +3.07 | +3.24 | 36.33 to 39.12 |
| Comb1024 | 47 | 25 | 42.10 | 42.91 | 32.90 | 0.0116 | 0 | 0.3925 | Comb1000 | +4.83 | +4.18 | +4.36 | 40.42 to 42.91 |

- No new table has a sinusoid below abs(f)·d = 0.05; the old tables had 1 to 31 below 0.01.
- The tail is below 0.012 for all new tables; the old tables reached up to 0.055.
- 9 of 12 tables are better than their predecessor on all three measures, the tables with 512 and 1024
  sinusoids by 1 to 8 dB.
- **Three tables miss an old value** (decision D7; agreed at STOP 1 that this is reported):
  Exp128 (mse_core -0.65 dB, mse_all -0.50 dB against Exp150), Comb128 (mse_core -0.29 dB against Comb150),
  Exp256 (mse_core -0.12 dB against Exp300). In all three cases the new table has 15 % fewer sinusoids
  than the table it is compared with, which accounts for 0.69 dB (the error scales with 1/N). 48 seeds
  did not close the gap.

### sos_preset

The tables are embedded in `src/sos_preset.cpp` as one signed frequency per sinusoid (5760 numbers,
17 digits); the direction of sinusoid n is direction n of the lattice and is recomputed.

| Test case | Asserted | Oracle |
|---|---|---|
| sizes and ACF for all 12 names | 200 samples, formulas within 1e-14 | formulas |
| decorrelation distance | dist ~ d, freq ~ 1/d | scaling |
| fit and slow sinusoids, all 12 | three measures >= old value minus 10 log10(N_old/N_new); abs(f)·d >= 0.05; max abs tail < 0.015 | baseline table above |
| correlation follows the target ACF | 3 presets, 3 directions, 5 distances, 20,000 phase sets, within 0.08 | target ACF |
| documented generation | regenerated 128 tables: same directions, fit within 1.5 dB | `sos_generate` |
| optional outputs, pre-sized, float; input validation | | |

Stub: 6 of 7 failed as expected. After implementation: 1 failure, see below; then all pass (319 assertions).
Largest deviation of the measured correlation from the target ACF: 0.044.

**Expectation changed after the first run (the only one in this port).** The test first demanded that
the regenerated 128 tables equal the shipped ones within 1e-12. Exp128 and Gauss128 did (3e-17), Comb128
differed by 9.5e-3. Cause: `sos_preset` computed the ACF as exp(-(0.025·i)) while the generation tool had
used exp(-(0.25·i)/10); the two differ in the last bit for some samples, and the optimiser, which takes
accept/reject decisions, went another way. Two consequences:

1. `sos_preset` now uses the arithmetic of the generation, and all three tables are reproduced to 3e-17
   on this machine.
2. Bit-wise reproduction cannot be promised across compilers and math libraries, because a last-bit
   difference of a cosine has the same effect. The test therefore asserts what must hold everywhere
   (same directions, fit within 1.5 dB of the shipped table; the spread over seeds is 1.5 to 3 dB) and
   prints the difference. The documentation of `sos_generate` says the same.

Tolerance of the fit test: the allowance of 0.69 dB for the 128 and 256 tables is the first-principles
correction for the number of sinusoids, fixed before the tables were embedded.


## WI-3 (continued): sos_randn, sos_rand, sos_randi

Procedure note: the shared implementation helper was written together with the stubs of the three
public functions; the tests were written afterwards and run against the public stubs (all 6 test cases
failed, the validation case too, because the validation lives in the helper).

| Test case | Asserted | Oracle |
|---|---|---|
| one device | each generator equals the documented composition (1e-12), incl. repeated and zero distances, another preset | `sos_preset` + `sos_init_phases` + `sos_val`, wired in the test |
| two devices | same map without and with equal `dist_decorr_b`, two maps for different distances, swap symmetry | same |
| mappings | Uniform = 0.5 erfc(-s/sqrt 2), integer = ceil(u · imax), limits | definitions |
| distributions, 10,000 positions | CDF of Normal (one device, same map, two maps) and Uniform within 0.02; integer counts within 150 of 1000 | ideal CDFs |
| seeds, subset of positions, pre-sized, fewer columns, float | | invariance |
| input validation | | |

After implementation: all pass at the first run. Largest CDF deviation: 0.0073 (Normal), 0.0073 (same
map), 0.0061 (two maps), 0.0061 (Uniform). Integer counts 938 to 1055.


## WI-6 sos_remove_bias

| Test case | Asserted | Oracle |
|---|---|---|
| one device | mean after below 1e-13; reported mean; offset = pi/2 - arg(Z) modulo pi; abs(offset) <= pi/2; one common offset; phases in [-pi, pi) | mean of `sos_val` (GENERIC); Z in scalar complex arithmetic |
| brute-force scan | zero of the mean over 3600 offsets within one scan step of the returned offset | scan |
| same map | mean of the combined value below 1e-13 | mean of `sos_val` with both devices |
| different maps, scale 1 and 2.5 | each device and the combined value below 1e-13; two offsets; different numbers of positions | means of `sos_val`, Z per device |
| variance after the correction | 4000 phase sets, 30 positions in 20 m: average variance equals 1 - mean(c^2)/v | ACF of the sinusoids |
| second call, fewer columns, float | offset 0 on the second call; float mean below 5e-6 | invariance |
| input validation | every argument; failed call leaves the phases untouched | |

Stub: 6 of 7 failed as expected. After implementation: all pass at the first run.
Variance test: variance of the mean 0.458, variance before 0.982, after 0.526, predicted 0.526.
Remaining means are at 1e-16. The scan found its zero at -1.30725 rad, the function returned -1.30714 rad.



## WI-7 Benchmark

Input: 5000 random positions inside 1 km, the old Comb300 table (300 sinusoids), `sos_val` in double.
Machine: AMD Ryzen 9 5950X (16 cores). Each number is the mean over at least 1 s of repeated calls.
MATLAB R2024a: `val` on an array of `n_gen` `qd_sos` objects with the same positions (16 computation
threads). Scratch programs: `scratchpad/bench/bench.cpp`, `bench.m`.

Time per (position × generator) in ns:

| n_gen | MATLAB `val` | GENERIC, 1 thread | AVX2, 1 thread | GENERIC, 16 threads | AVX2, 16 threads |
|---|---|---|---|---|---|
| 1   | 866  | 7666 | 397  | 555  | 36.4 |
| 20  | 861  | 567  | 37.3 | 41.8 | 4.80 |
| 400 | 1124 | 212  | 22.1 | 16.8 | 3.94 |

Time per call in ms:

| n_gen | MATLAB `val` | AVX2, 1 thread | AVX2, 16 threads |
|---|---|---|---|
| 1   | 4.33 | 1.99 | 0.18 |
| 20  | 86.1 | 3.73 | 0.48 |
| 400 | 2248 | 44.3 | 7.87 |

- MATLAB scales linearly with the number of generators, because every object is evaluated on its own.
  The library computes the sines and cosines once: 20 generators cost 1.9 times one generator, 400 cost
  22 times.
- Against MATLAB the AVX2 kernel is faster by a factor of 2 (1 generator, 1 thread) to 285 (400
  generators, 16 threads). For 400 generators on one thread the factor is 51.
- GENERIC on one thread is slower than MATLAB for a single generator (scalar double sine and cosine
  against MATLAB's vectorised, multi-threaded `cos`); it is the reference kernel, not the fast path.
- The AVX2 kernel stops scaling beyond 8 threads for 400 generators (3.9 ns at 16 threads, 4.5 ns at 8):
  each position block walks through 2 × 300 × 400 single-precision weights (960 kB), which is more than
  the per-core cache.
- Tried and discarded: separate accumulation chains for the cosine and sine terms. It was 20 % slower
  on one thread (53 against 44 ms for 400 generators).
- Measurement note: switching the thread count inside one process gave misleading numbers for the first
  runs after the switch; the table uses one process per thread count (`OMP_NUM_THREADS`).


## WI-8 MEX wrappers

14 wrappers in `api_mex/sos_*.cpp`, one per public function, each a plain argument conversion. Usage lines
as in the header. Deviations from the C++ interface: `sos_remove_bias` splits its in/out phase arguments
into `phase_in` / `phase_out` and `phase_b_in` / `phase_b_out` (the inputs are copied, the caller's
variables are not modified); `sos_map` returns one 4-D array `[ny, nx, nz, n_gen]`; `sos_calc_mse`
returns the C++ return value as first output. No index arguments exist, so nothing is shifted.

Tests: `tests/mex_tests/test_sos_*.m`, 14 files, run in MATLAB R2024a. First run: 11 of 14 passed. The three
failures were faults of the tests, not of the wrappers:

- `test_sos_val`, `test_sos_map`: a `uint8` value was passed for a `bool` argument;
  `qd_mex_get_scalar<bool>` accepts logical and floating-point input only. The tests now pass `false`.
- `test_sos_generate`: the direction of each sinusoid was recovered with `atan2`, which is undefined for
  a sinusoid with frequency 0. The test now uses the cross product with the lattice direction.

After the corrections: 14 of 14 pass in MATLAB R2024a.

Octave 11.1.0 (added on request after the final report; built with `make avx2=ON matlab=ON octave=ON`,
no compiler warnings): the 14 new tests pass without any change, 14 of 14. Complete MEX suite in Octave:
61 of 62 pass; the one failure is again `test_ray_mesh_interact`, line 113, with the same two numbers as
in MATLAB.

Complete MEX suite in MATLAB (62 test files): 61 pass, 1 fails: `test_ray_mesh_interact` (line 113,
0.98485 against an expected 0.69639). That test and the code it exercises are not touched by this
port. Checked on an unmodified copy of HEAD (`git archive HEAD`, built with `make matlab=ON avx2=ON`):
the test fails there with the same two numbers, so the failure predates this work.


## WI-9 Documentation and final state

- Every source file has its documentation block; `make documentation` ran without error and listed the
  14 functions under "Sum-of-sinusoids random number generator" in the C++ and the MATLAB overview. The
  regenerated files in `html_docu/` also contained pending documentation changes of earlier commits
  (`ray_mesh_interact` and others), so they were restored to the committed state; the documentation has
  to be regenerated when those are meant to be published.
- Section 9.1 of `md/PLAN_QD_SOS.md` holds the final function names, usage lines and the hand-over notes.
- Expectations or tolerances changed after they were first written: one, the regeneration test of
  `sos_preset` (WI-5). Test-side corrections that did not change an expectation of the function under
  test: an ambiguous initialiser and BLAS-dependent expressions in the Catch2 tests, the three MEX test
  corrections above.


## Re-creation of the 2018 paper (2026-10-12)

On request the numeric results of the paper `html_docu/publications/2018_sum_of_sinusoids.md` were re-run
with the library. Scripts, results and the comparison are in `tools/sos_port/`; the report is
`tools/sos_port/SOS_PAPER_REPRODUCTION.md`. Main points: `sos_calc_mse` reproduces the ASE values of the
paper on the `qd_sos` tables within 0.05 dB; 3-D tables are equal or better (up to 3.4 dB at 1000
sinusoids); 2-D tables for the exponential ACF fall 2.5 to 4.7 dB short at 2000 sinusoids, cause not
established; the dip of the estimated ACF below the desired one in Fig. 4 of the paper is the removal of
the cube mean by the Pearson coefficient, the same effect as in WI-1.
