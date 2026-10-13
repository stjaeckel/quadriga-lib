# Plan: port the sum-of-sinusoids model to quadriga-lib

Status: draft plan, written 2026-10-09 in a QuaDRiGa-NG session. Nothing is implemented yet.

This document is the brief for a new session that works in the quadriga-lib repository and ports the
QuaDRiGa class `qd_sos` to C++. It is written so that the session can run mostly on its own. It stops
and waits for Stephan at exactly two points (section 3) and reports at the end.

Read first, in this order:

1. This document.
2. `md/SPEC_CPP_API.md` and `md/SPEC_MATLAB_API.md` in quadriga-lib. They define file layout, style,
   tests and documentation blocks. Where this plan and the specs disagree, section 3 of this plan wins
   for the review points; everything else follows the specs.
3. The MATLAB reference: `/sjc/QuaDRiGa-NG/quadriga_src/@qd_sos/` (18 methods, one private function)
   and its unit tests `/sjc/QuaDRiGa-NG/xunit_new/tests/testSOS_*.m` and `testBuilder_CorrMap_SOS.m`.

QuaDRiGa-NG is read-only for the library session. quadriga-lib was read-only for the session that
wrote this plan.


## 1. Goal and scope

Add a new public header to quadriga-lib that provides the complete sum-of-sinusoids (SOS) model:
spatially correlated random variables with a prescribed autocorrelation function (ACF), in up to three
dimensions, with the dual-mobility extension (two positions per sample).

In scope:

- Everything that `@qd_sos` does numerically: presets, phase initialisation, evaluation, the
  convenience generators (`rand`, `randn`, `randi`, `map`), ACF helpers (`acfi`, `acf_2d`,
  `acf_approx`, `calc_mse`, `acf_estimate`), the coefficient optimiser (`generate` with
  `suosi_iteration_step`), and the bias handling (`remove_bias`).
- An analysis of the bias (work item 1). This comes before any design of the bias function.
- Regenerated preset tables, computed with the ported optimiser in double precision.
- AVX2 kernels for the evaluation, with a GENERIC double-precision kernel as reference.
- Catch2 tests for every function, MEX wrappers and MOxUnit tests for every function a MATLAB user needs.
- A hand-over section for the later integration into QuaDRiGa-NG (section 9).

Out of scope:

- Python wrappers (separate task).
- Editing QuaDRiGa-NG. Replacing `@qd_sos` there is a later session in that repository.
- `save`, `load` and `copy` of the MATLAB class. They are object handling and stay in MATLAB.


## 2. Decisions

Answers given by Stephan on 2026-10-09. They are binding for the library session unless he changes them.

| # | Question | Answer |
|---|---|---|
| D1 | API style: class or functions? | **Functions, batched.** Free functions on plain arrays. One call takes the shared frequencies plus a phase matrix with one column per generator. See section 5 for the analysis. |
| D2 | How closely must results match MATLAB? | **Same statistics only.** ACF shape and distributions must match. Tables, precision and algorithm details may change. Value-by-value agreement with MATLAB is not required. |
| D3 | Phase initialisation | **Both.** The library offers a seeded function that draws phases, and every evaluation function accepts phases supplied by the caller. |
| D4 | Scope of the first pass | **Everything in `@qd_sos`**, including the optimiser. `save`/`load`/`copy` stay in MATLAB. |
| D5 | Default precision and kernel | **Auto, with range reduction.** `use_kernel = 0` picks AVX2 when available. The phase argument is reduced to [-pi, pi] in double before the single-precision kernel. GENERIC (full double) stays selectable and is the test reference. |
| D6 | Preset tables | **Regenerate with the ported optimiser.** The existing tables are not copied into C++ as the shipped presets. |
| D7 | Preset set | **Same 12 names, double precision:** Exp, Gauss, Comb, each with 150, 300, 500 and 1000 sinusoids. Recomputed with a fixed seed and uniform sphere sampling. Each new table must reach at least the ACF fit of the current table (see work item 5). |
| D8 | Wrappers | **MEX only.** |
| D9 | Stop points | **Stop after the bias analysis. Then draft the header and stop again.** |
| D10 | QuaDRiGa-NG integration | **Separate final phase.** This plan ends with hand-over notes; the integration is a later session. |
| D11 | Location of this plan | QuaDRiGa-NG repository root. Copy it to `quadriga-lib/md/` when starting the library session. |
| D12 | Header name | Stephan named it `qd_sos.hpp`. The existing public headers are called `quadriga_*.hpp`; raise the naming once at stop point 2 and follow his answer. |

Interpretation that still needs a yes at stop point 2: D9 names two stops. This plan reads that as
"the per-function review points A and B of `SPEC_CPP_API.md` are not stops". The work they describe
is still done and written to the log (section 3).


## 3. Stop points and autonomy

The session stops and waits for Stephan at these points only:

- **Stop 1, after work item 1 (bias analysis).** Deliver the analysis report described there.
- **Stop 2, after work item 2 (header draft).** Deliver the complete header with all declarations
  and comments, the open naming questions, and the proposed section name for the documentation.

After stop 2 the session works through the remaining items without waiting. It ends with the final
report of `SPEC_CPP_API.md` review point C, once for the whole port.

Replacing the per-function review points:

- Keep the test-first order of the spec for every function: stub, tests from first principles, run
  against the stub, evaluate the oracles, then implement.
- Write what review point B asks for into a running log, `md/LOG_QD_SOS.md`: the test cases with their
  oracles, the run against the stub, the diagnostic tables that matter, the tolerances and why, and
  open points. One section per function. This is the record Stephan reads at the end.
- Stop outside the two planned points only when a decision in section 2 turns out to be impossible, or
  when a test exposes a contradiction that cannot be resolved from this plan, the specs and the
  reference. Never change an expectation or a tolerance just to get a test to pass.


## 4. Background

### 4.1 The model

A generator with N sinusoids in d dimensions (d = 1, 2 or 3) produces, at position x:

    s(x) = a · sum_{n=1..N} cos( 2·pi · f_n·x + phi_n ),      a = sqrt(2/N)

- `f_n` are spatial frequencies in 1/m, size `[N, d]`. They are chosen once so that the ACF of `s`
  approximates a target ACF R(distance) in all directions.
- `phi_n` are random phases, uniform in [-pi, pi]. A new set of phases is a new realisation of the
  random field with the same ACF.
- With many sinusoids `s` is approximately Normal with zero mean and unit variance.

Dual mobility: a sample has two positions, `ca` (first device) and `cb` (second device), and two
phase columns. The MATLAB code computes `sa` from (`ca`, phase column 1) and `sb` from (`cb`, phase
column 2) and combines them:

- phases of both columns identical (devices on the same map): `s = (sa + sb) / sqrt(2 + 2·A(D))`,
  where `D` is the distance between `ca` and `cb`, clipped to the last ACF sample point, and `A` is the
  target ACF interpolated at `D`
- phases differ: `s = (sa + sb) / sqrt(2)`
- if the decorrelation distance has two elements, the frequencies used for `cb` are scaled by
  `d_decorr(1) / d_decorr(2)`

Distributions: `Normal` returns `s`. `Uniform` returns `0.5 · erfc( -s / sqrt(2) )`.
`UniformForced` additionally remaps the values through their own empirical CDF (uses `acdf` and
`interp`), which makes the result depend on the set of positions.

Changing the decorrelation distance rescales the distance axis of the ACF and the frequencies:
`dist_new = dist · d_new / d_old`, `freq_new = freq · d_old / d_new`. The presets are stored for 10 m.

### 4.2 What exists in MATLAB

| MATLAB method | Code lines | Role |
|---|---|---|
| `set_acf` | ~960 | 12 presets: 200-point ACF tables (0 to 49.75 m in 0.25 m steps, decorrelation distance 10 m) and the precomputed 3-D frequency tables. Almost all lines are constants. |
| `init` | 28 | Draws the phases with `rand`. `use_same = true` copies column 1 to column 2. |
| `val` | 86 | Evaluation as described in 4.1. Works on object arrays by looping. |
| `rand`, `randn`, `randi` | 17 each | Static convenience functions: one generator per decorrelation distance, evaluate, return `[n_gen, n_pos]`. A decorrelation distance of 0 is replaced by 0.1. `randi` returns `ceil( s · imax )`. |
| `map` | 41 | Evaluation on a grid given by x, y, z vectors; output `[ny, nx, nz]`. |
| `acfi` | 12 | Target ACF interpolated at given distances, clipped to the last sample point. |
| `acf_2d` | 18 | Target ACF on a 2-D grid from -2·Dmax to 2·Dmax; zero outside Dmax. |
| `acf_approx` | 34 | ACF actually produced by the sinusoids, in 1-D, or on the x-y / x-z / y-z planes. |
| `calc_mse` | 77 | Approximation error of the ACF in dB for T test directions; optional 2-D errors. |
| `acf_estimate` | 67 | Monte-Carlo estimate of the 3-D and 6-D (dual-mobility) ACF from random positions. |
| `generate` | 196 | Optimiser: finds the frequencies for a given ACF, N and dimension. Iterates over the sinusoids, re-fitting one at a time against the residual. Uses `rand`, single precision, optional plotting. |
| `private/suosi_iteration_step` | 28 | Inner step of the optimiser: coarse search over -15:0.1:15, then step refinement; a tie is broken with `randi`. |
| `remove_bias` | 51 | Iterative search for a common phase offset that minimises the absolute mean over a given set of positions. See work item 1. |
| `save`, `load`, `copy`, class file | ~190 | Object handling. Not ported. |

### 4.3 How QuaDRiGa uses it

- `qd_builder.init_sos` creates the generators: 8 for the large-scale parameters, 1 for absolute time
  of arrival, 1 for the ground reflection, and per cluster 5 for delays and angles, plus
  `n_cluster × (n_subpath + 1) × n_freq` for XPR and `n_cluster × n_subpath × n_freq` for initial
  phases. All generators of one builder use the same preset.
- The builder then calls `val` on whole arrays of generators, e.g.
  `val( xpr_sos(:,1:n_subpaths,iF), rx_pos, tx_pos )`. These generators share frequencies and
  decorrelation distance and differ only in their phases. This is the case to optimise for.
- `init_sos` calls `remove_bias` on the large-scale and absolute-TOA generators, and afterwards
  overwrites phase columns to make the parameters reciprocal (e.g. ASD column 1 takes ASA column 2).
- `qd_layout.set_scenario` and `gen_o2i_loss` use `qd_sos.rand` for LOS state and indoor state.
- `gen_lsf_parameters` uses `acfi` for the correlation between departure and arrival spreads.

### 4.4 Measured cost

Profile of a system-level drop in MATLAB R2024a (600 UEs, 7 sites, 4200 links, 57 s in total):
`qd_sos.val` took 15.5 s (27 %) over 47,166 calls, and `qd_sos.set_acf` was called 15,260 times,
rebuilding the constant tables each time. The port should remove both costs: one batched call per
generator array, and presets that are static data.


## 5. Class or functions

Stephan asked for this analysis; the conclusion is decision D1.

What speaks for a class (as `arrayant` and `channel`):

- A generator has state that belongs together: frequencies, phases, the ACF table, the decorrelation
  distance, the distribution. A class keeps it consistent and can validate it once.
- C++ users get a natural object: construct from a preset, initialise, evaluate.
- The MATLAB class maps 1:1 onto it.

What speaks against it:

- Wrappers. A class needs a struct layout plus conversion helpers in `mex_quadriga_lib_functions.hpp`
  and again in the Python adapter, as `arrayant` and `channel` have. Free functions on arrays need
  only the generic helpers that already exist.
- Batches. QuaDRiGa evaluates hundreds of generators that share everything except the phases. With
  one object per generator this becomes a `std::vector` of objects that all hold a copy of the same
  frequency table, and a batched kernel would first have to verify that the tables are equal.
- The state is small and flat: two matrices, two vectors, a few scalars. There are no invariants that
  a caller could not keep with plain arrays.
- MATLAB already has the object (`qd_sos`); it can keep its handle class as a thin shell around
  function calls.

Why batching decides it. For generators that share the frequencies, the argument `2·pi · f_n·x_p` is
the same for all of them. With `cos(a + phi) = cos(a)·cos(phi) - sin(a)·sin(phi)`:

- compute `sin` and `cos` of the argument once per (sinusoid, position): N·P evaluations
- each generator then costs one multiply-add pair per (sinusoid, position) with its precomputed
  `cos(phi_n)`, `sin(phi_n)`

The expensive part is independent of the number of generators. A function that takes a phase matrix
`[N, n_gen]` expresses this directly. A class with one phase vector does not.

Conclusion: free functions on plain arrays, with the generator dimension in the phase matrix. A thin
C++ convenience class can be added later on top of the functions without changing them. It is not
part of this plan.


## 6. Function set (sketch, to be finalised at stop 2)

This is a starting point for work item 2, not a contract. Names, argument order and array
orientation are decided in the header draft.

| Proposed function | MATLAB origin | Inputs | Outputs |
|---|---|---|---|
| `sos_preset` | `set_acf`, `set.dist_decorr` | preset name, decorrelation distance | frequencies `[N, 3]`, ACF and distance vectors, scaled to the requested decorrelation distance |
| `sos_init_phases` | `init` | N, number of generators, seed | phases `[N, n_gen]`, uniform in [-pi, pi] |
| `sos_val` | `val` | frequencies, phases `[N, n_gen]`, positions; optional second positions and phases, ACF table, frequency scale for the second device, distribution, `use_kernel` | values `[n_pos, n_gen]` |
| `sos_map` | `map` | as `sos_val`, positions as x, y, z grid vectors | values `[ny, nx, nz]` per generator |
| `sos_rand` (one function with a distribution switch, or three) | `rand`, `randn`, `randi` | decorrelation distances `[n_gen]`, positions, preset name, seed | values `[n_pos, n_gen]` |
| `sos_acf_2d` | `acf_2d` | ACF and distance vectors | 2-D ACF and its axis |
| `sos_acf_approx` | `acf_approx` | frequencies, distance vector | approximated ACF (1-D, or three planes) and its axis |
| `sos_calc_mse` | `calc_mse` | frequencies, ACF, distances, number of test directions | errors in dB |
| `sos_acf_estimate` | `acf_estimate` | frequencies, ACF, distances, number of positions, distance bins, seed | estimated 3-D and 6-D ACF |
| `sos_generate` | `generate`, `suosi_iteration_step` | ACF, distances, N, dimension, sampling mode, number of test directions, seed | frequencies, error per iteration, test directions |
| bias function(s) | `remove_bias` | decided after work item 1 | decided after work item 1 |

Points to settle in the header draft:

- **Position layout.** `qd_sos` uses `[3, n_pos]`. In the library, tools use `[n, 3]` and channel data
  uses `[3, n]`. Pick one and say why.
- **`acfi`.** It is `interp` with clipping. Decide whether it needs its own function or a remark in
  the documentation of `sos_preset`.
- **Amplitude.** `a = sqrt(2/N)` is implied by N. Offer no amplitude argument unless a use is found.
- **`UniformForced`.** It depends on the sample set and calls `acdf`. Nothing in QuaDRiGa-NG outside
  the class uses it (no caller in `quadriga_src`, the tests or the tutorials). Keep it as a
  distribution option of `sos_val`, leave the remapping to the caller, or drop it.
- **Generators with different decorrelation distances in one call** (`rand`/`randn`/`randi` take a
  vector). The frequencies then differ by a scalar per generator, so the sin/cos cannot be shared.
  Either loop inside the function or restrict the batched kernel to shared frequencies and let the
  convenience function loop.
- **Dimensions.** The presets are 3-D. `generate` supports 1, 2 and 3. Positions with fewer rows than
  the frequency table are zero-padded in MATLAB.
- **Seeds.** Every function that draws random numbers takes an explicit seed (`SPEC_CPP_API.md`:
  no global state). Document the generator used so results are reproducible across platforms; do not
  rely on distribution classes whose output differs between standard libraries.
- **Documentation section.** A new section name (for example "Sum-of-sinusoids model") must be agreed
  with Stephan; section names are matched literally.
- **Aggregate header.** `quadriga_lib.hpp` includes the other public headers; the new one should be
  added there in the same way.


## 7. Kernel design

- **Structure**, per `SPEC_CPP_API.md` "AVX2 and CUDA kernels": the public function validates,
  converts and dispatches; kernels are `qd_SOS_GENERIC` and `qd_SOS_AVX2` in `sos_val_generic.cpp` and
  `sos_val_avx2.cpp`, declared in the `quadriga_lib_*_functions.hpp` headers; `use_kernel` selects
  0 = auto, 1 = GENERIC, 2 = AVX2 and throws if the requested kernel is unavailable. No CUDA kernel.
  See `ray_triangle_intersect.cpp` and the `fast_sincos` / `fastmath_avx2.h` sources.
- **Shared argument.** Implement the scheme of section 5: sin and cos once per (sinusoid, position),
  then a multiply-add per generator. Choose the loop order and block size so that the working set
  stays in cache for N = 1000 and many positions; a blocked loop over positions is the natural unit
  for OpenMP.
- **Range reduction (decision D5).** The existing single-precision sin/cos is accurate to about
  2^-22 only for arguments in [-pi, pi] and degrades to 2^-16 at 500. The argument
  `2·pi · f_n·x_p` grows with the position: for a 1 m decorrelation distance and a position 10 km from
  the origin it reaches several thousand radians. Compute `f_n·x_p` in double, reduce it to one period
  in double, and only then convert to float. The error of the result must not depend on the distance
  from the origin. Pin this with a test.
- **Independence of the batch.** The value at a position must not depend on which other positions are
  in the same call. The MATLAB test `testSOS_val_NumericStability.m` asserts this to 1e-14: the same
  position evaluated inside two different position sets gives the same value. For the AVX2 kernel
  this means that full vector lanes and the remainder loop must produce identical results for the
  same input; add a test that moves one position through all lane slots and the remainder.
- **GENERIC kernel.** Full double precision, same algorithm. It is the reference for the AVX2 kernel
  and the default when AVX2 is unavailable.
- **Dual mobility.** The second device uses the same kernel with its own positions, phase columns and
  frequency scale; the combination and the ACF normalisation are cheap and stay in the public function.
- **OpenMP.** Parallelise over position blocks. Results must not depend on the number of threads.
- **Benchmark** (work item 7). Report time per (position × generator) for GENERIC and AVX2, for
  n_gen = 1, 20 and 400 with N = 300, and compare with the MATLAB `val` on the same input.
- **Kernels for other functions.** `acf_approx`, `calc_mse` and the optimiser are offline tools.
  Write them in portable C++ first; add a kernel only if the preset regeneration proves too slow.


## 8. Work items

Work through them in this order.

### WI-0 Orientation

- Read the two specs and the MATLAB reference (section "Read first").
- Run the MATLAB/Octave SOS unit tests in QuaDRiGa-NG once to see the current behaviour
  (`xunit_new/run_tests.m`; needs `quadriga_src` and `quadriga_lib` on the path).
- Start `md/LOG_QD_SOS.md`.

### WI-1 Bias analysis, then STOP 1

Stephan's description of the problem: somewhere there is a bias, which is currently corrected by
measuring over a limited number of generated points. He wants to know what causes it and whether
there is a closed-form solution. This item is analysis only. No library code.

What the current correction does (`remove_bias.m`, called from `qd_builder/init_sos.m`):

- It takes the positions of one builder (`ca`, optionally `cb`) and searches for a phase offset
  `delta`, added to all N phases of a column, that minimises `abs( mean( val( ca, cb, phases ) ) )`.
- The search is a 30-step line search with a shrinking, sign-flipping step (start pi, factor -0.4).
- Same phases in both columns: one search, both columns shifted together. Different phases: the
  columns are searched one after the other, twice. Different phases and no `cb`: a random permutation
  of `ca` is used as `cb`.
- It is applied to the 8 large-scale generators and the absolute-TOA generator only. All of these are
  `Normal`.
- After the call, `init_sos` overwrites phase columns for reciprocity (lines 132 to 141), including
  swapping columns between the ASD/ASA and ESD/ESA generators.

Questions to answer:

1. **What is the bias?** Separate the candidates and measure each:
   - the finite-sample mean of a correlated field: positions that lie within a few decorrelation
     distances are not independent samples, so their mean has a variance far above 1/n_pos
   - sinusoids with a frequency near zero, which contribute an almost constant term
     `a·cos(phi_n)` across the whole set of positions
   - a property of the frequency tables (for example an uneven distribution of low frequencies)
   - an effect of the dual-mobility combination or its normalisation
   - an effect of single precision in MATLAB
   Quantify the mean and its spread over many phase realisations, as a function of the extent of the
   position set relative to the decorrelation distance, for each preset.
2. **Is it a bias at all?** Averaged over the phases, the mean of `s` at any position is zero by
   construction. Decide whether the observed offset is a per-realisation effect that is statistically
   legitimate (the field is correlated, so a local patch has a non-zero mean) or a systematic error.
   State clearly what is being corrected and whether correcting it distorts the statistics: forcing
   the sample mean to zero removes real variance of the large-scale parameters.
3. **Why does it matter downstream?** Find what in QuaDRiGa-NG depends on a zero mean over the drop
   (for example the mapping to the configured mean of delay spread, K-factor, shadow fading). Read
   `gen_lsf_parameters.m` and `init_sos.m`. This decides which quantity must be exact.
4. **Is there a closed form?** Lead to verify, not a result: for a fixed set of positions the sample
   mean is linear in the phasors of the sinusoids. With
   `z_n = (1/n_pos) · sum_p exp( j·2·pi · f_n·x_p )`, the mean is
   `a · Re( sum_n z_n · exp( j·phi_n ) )`. A common offset `delta` multiplies the sum by
   `exp( j·delta )`, so the mean becomes `Re( exp( j·delta ) · Z )` with a single complex number `Z`,
   and it vanishes for `delta = pi/2 - arg(Z)`. If this holds, the 30-step search is the numerical
   solution of a one-line formula. Check it against `remove_bias`, and work out the same for:
   - dual mobility with identical phases (the ACF normalisation is a per-sample weight, which keeps the
     expression linear)
   - dual mobility with different phases, where shifting one column gives
     `Re( exp( j·delta ) · Z_a ) + m_b` and a zero exists only if `abs(Z_a) >= abs(m_b)`
5. **Is the current use consistent?** Check whether the reciprocity overwrite in `init_sos` after
   `remove_bias` undoes the correction for the generators whose columns are swapped, and whether the
   random permutation used for the missing `cb` makes the result non-reproducible.
6. **Alternatives.** If the offset is legitimate statistics, consider whether the right fix is
   elsewhere, for example subtracting the sample mean where it is needed, removing or constraining
   near-zero frequencies when the tables are regenerated (D6 makes this possible), or leaving the
   field untouched. Compare what each does to mean, variance and ACF.

How to work: experiments in MATLAB or Octave with the existing `qd_sos` (QuaDRiGa-NG is read-only;
scripts go into the session's scratch space), plus derivation on paper. Include the experiments'
numbers in the report.

Deliverable at STOP 1, in `md/LOG_QD_SOS.md`:

- the cause, with the measurements that support it
- the closed form, or the reason there is none, for the three cases above
- whether the correction changes the statistics, and by how much
- a recommendation for the library: which function(s), with which inputs and outputs
- consequences for the preset regeneration (work item 5)

### WI-2 Header draft, then STOP 2

- Write the complete public header with all declarations, following "PUBLIC HEADER (THE CONTRACT)" in
  `SPEC_CPP_API.md`: grouping comment, description lines, one argument per line with meaning, unit,
  index base and size; mandatory arguments by reference, optional ones by pointer with default
  `nullptr`; outputs after inputs; scalar options last.
- Resolve the points listed in section 6 and state the choice and the reason for each.
- Include the bias function(s) as agreed at stop 1.
- List the MATLAB usage line planned for each MEX wrapper next to its declaration, so that the C++
  interface is checked against the wrapper needs before it is fixed.
- Raise at the stop: the header name (D12), the documentation section name, and the reading of D9
  given in section 2.

Do not implement anything before Stephan has answered.

### WI-3 Core runtime functions

Order: `sos_init_phases`, `sos_val` with the GENERIC kernel, then the AVX2 kernel, then `sos_map` and
the `rand`/`randn`/`randi` equivalent. The preset function comes later (WI-5); until then the tests
build small frequency tables by hand and may load one existing MATLAB table as test data.

Test-first, per function, as in the spec; the review-B material goes into the log.

Oracles and checks for `sos_val` (first principles, independent of the implementation):

- A brute-force scalar double loop of the formula in section 4.1, written in the test file.
- Hand cases: one sinusoid, known frequency and phase, positions on the axis; value known exactly.
- Periodicity: shifting a position by `1/abs(f)` along `f` leaves the single-sinusoid value unchanged.
- Batch consistency: column g of a batched call equals a call with only phase column g.
- Kernel parity: AVX2 against GENERIC within single-precision accuracy, including positions far from
  the origin (range reduction).
- Statistics over many phase realisations at a fixed position: mean near 0, variance near 1, both
  within the Monte-Carlo error stated in the test.
- Correlation between two positions at distance d approaches the target ACF at d.
- Dual mobility: identical positions and identical phases reproduce the single-device value; the
  normalised result keeps unit variance for small and large device distances; different phases use
  `1/sqrt(2)`; the frequency scale of the second device is applied.
- Uniform: values in (0, 1); the distribution over phase realisations is flat within a stated bound.
- Position rows fewer than the frequency dimensions behave as zero padding.
- Determinism: same seed gives the same phases; results do not depend on the number of threads; one
  case crosses the block size of the kernel.
- Everything in "What every suite covers" of the spec: sizes, optional arguments, pre-sized outputs,
  float and double, input validation with a passing baseline.

### WI-4 ACF tools and optimiser

`sos_acf_2d`, `sos_acf_approx`, `sos_calc_mse`, `sos_acf_estimate`, `sos_generate`.

- Port `generate` and `suosi_iteration_step` in double precision. Replace `rand` and the `randi`
  tie-break by the seeded generator. Drop the plotting.
- The uniform sampling of directions uses `qf.pack_sphere` in MATLAB
  (`/sjc/QuaDRiGa-NG/quadriga_src/+qf/pack_sphere.m`, about 30 code lines). Port it as a file-local
  helper, or use `icosphere` if its point counts fit; say which and why.
- Oracles: for a single sinusoid the approximated ACF is a cosine in closed form; for the optimiser,
  a known target that a few sinusoids can represent exactly (for example the ACF of two fixed
  frequencies in 1-D) must be recovered; the reported error must equal an independent evaluation of
  the fit; a rerun with the same seed gives the same table.
- No LAPACK is needed anywhere in the MATLAB reference.

### WI-5 Presets

- Baseline: load the 12 existing frequency tables from `set_acf.m` as test data and measure their
  ACF error with the ported `sos_calc_mse` (the `mse`, `mse_core` and `mse_all` outputs, with a fixed
  number of test directions). Record the numbers.
- Regenerate all 12 presets with `sos_generate`: 3-D, double precision, uniform sphere sampling, fixed
  and documented seed. Apply whatever work item 1 concluded about low frequencies.
- Acceptance (decision D7): every new table reaches at least the ACF fit of its predecessor on all
  three error measures. If a table does not, try more test directions or more iterations and report;
  do not ship a worse table silently.
- Embed the tables as static constant data and implement `sos_preset` on top, including the scaling
  to a requested decorrelation distance. Keep the ACF definitions (Exp, Gauss, Comb; 200 points,
  0 to 49.75 m at a 10 m decorrelation distance) as in `set_acf.m`, generated from their formulas
  rather than copied as numbers where the formula is clear. Verify the formulas against the stored
  tables first.
- Record the generation command, seed and run time in the log so the tables can be reproduced.

### WI-6 Bias function(s)

Implement what was agreed at stop 1, with tests that use the closed form (if there is one) as oracle
and a brute-force minimisation as cross-check.

### WI-7 Benchmark

As described in section 7. Add the table to the log and the headline numbers to the final report.

### WI-8 MEX wrappers

One wrapper per function a MATLAB user needs, following `SPEC_MATLAB_API.md`: wrapper first, then the
MOxUnit test of the wrapper. The physics is not re-tested there. The spec's review point A applies
only if an interface question cannot be settled from the header agreed at stop 2; its review point B
is replaced by the final report.

### WI-9 Documentation and final report

- Documentation blocks complete and matching the implementation; `make documentation` runs.
- `make test` passes completely; the library compiles without warnings; diagnostics stay enabled.
- Final report as in `SPEC_CPP_API.md` review point C, for the whole port: test results with the
  actual output, warnings, every expectation or tolerance changed after it was first written and why,
  deviations from this plan, known limitations, benchmark numbers.
- Update section 9 of this plan with the final function names and usage lines.


## 9. Hand-over for the QuaDRiGa-NG integration (later session)

Not part of the library session. Listed here so that the library interface is designed with it in mind.

Files in QuaDRiGa-NG that use the SOS class:

| File | Use |
|---|---|
| `@qd_sos/*` | The class itself. Expected to shrink to a thin handle class around the MEX calls, or to be replaced. `save`, `load`, `copy` stay. |
| `@qd_builder/init_sos.m` | Creates all generators, calls `remove_bias`, sets up reciprocity by copying phase columns. |
| `@qd_builder/gen_lsf_parameters.m` | `val` on the large-scale generators, `acfi` for cross-correlations, reciprocity checks on `sos_phase`. |
| `@qd_builder/gen_ssf_parameters.m` | `val` on arrays of path, XPR, initial-phase and cluster-delay generators. The main beneficiary of batching. |
| `@qd_layout/set_scenario.m`, `@qd_layout/gen_o2i_loss.m` | `qd_sos.rand` for LOS and indoor state. |
| `@qd_builder/copy.m` | Copies generator objects. |
| `xunit_new/tests/testSOS_*.m`, `testBuilder_CorrMap_SOS.m` | 19 test files. Those that assert statistics should keep passing; those that assert properties of the old tables or of single precision will need to move. |

Consequences of the decisions for that session:

- Results will not be value-identical to the current MATLAB implementation (D2, D6): new tables,
  double precision. Tests that compare against stored numbers must be reviewed one by one.
- The builder currently stores one object per generator. To use the batched call it has to collect
  the phase columns of a generator array into one matrix, or store them that way from the start.
- The bias handling changes according to the outcome of work item 1.


## 10. Notes from reading the MATLAB code

Things the library session should know before it starts; none of them is a decision.

- `val` has two code paths for more or fewer than 10,000 positions. They compute the same thing; the
  split only limits memory in MATLAB.
- `val` clips the device distance to `dist(end)` before interpolating the ACF.
- The same-phase test in `val` is `all( abs( ph(:,1) - ph(:,2) ) < 1e-6 )`; in `remove_bias` it is an
  exact comparison.
- `rand`, `randn` and `randi` build a new generator, with new phases, for every decorrelation
  distance on every call.
- `set.dist_decorr` accepts one or two values; two equal values are collapsed to one.
- `load` derives the decorrelation distance from the first ACF sample below `exp(-1) + 1e-6`;
  `generate` uses the first sample at or below `exp(-1)`.
- `generate` computes in single precision, normalises the distances to `2·pi·D/max(D)`, and divides
  the resulting frequencies by `max(D)` at the end.
- `acf_estimate` draws its positions inside a cube with half edge `0.4·max(D)`.
- `map` returns `[ny, nx, nz]`, i.e. meshgrid order.
- Phases are stored in single precision in MATLAB.
