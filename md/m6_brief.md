# Brief — M6 (the room-side face as a series impedance: phase and angle of a peaked face; A12)

This brief is the whole task description. Read it in full before doing anything, and re-read it at the start of
every phase. Findings go into `md/m6_report.md` (create it), one section per phase, so the next session can pick them
up.

## Why

`md/acoustic.md` is the reference; read sections 2.3, 2.6, 2.7, 3.1 (the angle row), 3.2, 3.8, 3.9, 4 (M5, M6, M7),
6 (A12, A20), 7 (the last bullet) and 10 (the M6 decisions). The reflect side comes first (P6), and the reflection
phase is an open item (P8, A12). N7 closed it for a hard wall and a porous layer; open is the phase of a peaked
rigid face.

Today a mass-law face with `tf` is a resistive pad (`Material::closed_face`, and `Material::tf_face` where the face is
open): it absorbs `tf(f)` at every angle and keeps the reactance of the bare face, so its reflection has the phase of
a hard wall. A panel absorber's own phase swings through its resonance (−18° to +15° for a 12.5 mm board on 25 mm air,
the round-1 model with a mounting resistance −12° to +15°; report 3.5, `md/acoustic.md` 4 M6), and a locally reacting
face absorbs differently at oblique incidence. Neglecting the boundary phase raised level errors by up to about 2.4 dB
in phased beam tracing [Jeong_2012_BoundaryConditionsPhasedGA].

**M6 as decided** (Stephan, 2026-10-15; `md/acoustic.md` 4 and 10): where a peak is set (`tfP ≠ 0`, `tfF > 0`), the
pad's resistance and the face's reactance become the series impedance per unit `Z₀`

    ζ = R′ + j·(1 + R′)·tfQ·(f/tfF − tfF/f),   tfP = 4R′/(1 + R′)²,

met as `ζ·cos θ` (local reaction). Its normal-incidence absorption is M5's law; its reflection carries the phase and
the angle dependence. The power-law part stays a resistive pad, `|r|² = (1 − tf_bg)·|r_ζ|²` (to be examined, D3).
Over-damped by default (`R′ > 1`, `r` positive at the peak); `tfQ < 0` selects the under-damped root `1/R′` (a
relaxed `require`). The mass relation `1 + R′ = ωm″/(Q_α·Z₀)` is a parameter rule, not code. After A18 M6 has lower
priority and covers **locally reacting peaks only**: a partitioned cavity, a face known from catalog data. An
unpartitioned construction with a known build is M9 (a wave-medium leaf on a box of near-air material, 2.8), and M7
(the cavity's angle shift on the `tf` peak) is dropped. "The detailed design follows the A18 measurement": this
brief's phase 0 is that design.

What it buys: the complex reflection factor of a resonant face (3.8, 3.9: a frequency-domain tracer then carries a
resonator's energy storage through its factor), and the random-incidence value of a locally reacting peak, which is
not its `α₀` (Jeong: up to 0.951; for a high-impedance face `α_st` exceeds `α₀` by up to about 1.7 times). So the fit
target of a peaked row becomes the Paris integral of the face (`md/acoustic.md` 9), and the six peaked rows refitted
under M5 (all within 0.017, cork 0.047; `md/m5_a21_report.md` 1.2) have to be refitted on M6.

## Facts at the start

- Last commit 4035ac4 (manager file); the source state is 238444d (M5 and A21). `git status` should be clean apart
  from this brief and `md/acoustic_manager.md`; if anything under `src/`, `tests/` or `md/acoustic.md` is modified, stop and ask Stephan.
- Baselines (`md/wave.md` 0, regenerated 2026-10-16 on 4035ac4): `[wave]` 165110 assertions in 179 cases,
  `[acoustic]` 36813 in 52, whole suite 1311472 in 1138. `src/ray_mesh_interact.cpp` 5089 lines.
  `python3 tools/acoustic/check_docs.py` passes.
- The pad in the code (`src/ray_mesh_interact.cpp`): `Material::tf` (the law, line ~410); `apply_tf_pair` (the pair
  rule of two faces' `tf`); `tf_face` (a face with a wave on both sides: a real transformer formed on the modulus of
  the admittance ratio); `closed_face` (a face with a wave on one side: a matched resistive pad in front of the bare
  face's reactance, formed on the modulus of the incidence side's immittance `y`). Callers: `interact_with`
  (EM and scalar, closed and open), the solver's face terms (around lines 1100–1180), the slab path (~1419), and the
  plate rule's `PLATE_ABS` / `PLATE_IN` (~4180), which read a face's absorption through `interact_with`. The
  `require(tfQ, "tfQ", false)` in the column check is the one to relax. `tf()` squares `tfQ·(…)`, so its sign does not
  change M5's law. Line numbers drift; find them by name.
- A rigid face in the two-parameter mapping is **open below its critical angle** (concrete about 4.9°, glass 3.48°,
  gypsum 9.44°) and closed above it, so a peaked face is met by `tf_face` near normal incidence and by `closed_face`
  elsewhere; today there is no step between the two (report 3.5). In the one-parameter mapping it is closed from about
  0.003° and the bare face's phase is `2θ` (report 3.5).
- The cases that carry a peak: *Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption of
  a mass-law face* (`test_acoustic_state_update.cpp`) and *ray_progress - the plate rule and the absorption peak at
  every frequency of a band* (`test_acoustic_progress.cpp`). Both use the board `a` 8.1817e-08 (one-parameter mapping),
  `thk` 12.5 mm, `m` 20, `tf` 0.05, `tfP` 0.25 at 150 Hz, `tfQ` 0.75, and write their reference from the column
  definition (absorption flat in angle). No `[wave]` case carries `tfP`. `test_obj_file_write.cpp` checks the column
  header only. No new column is needed; the OBJ / API column lists do not change.
- The six peaked rows and their M5 fit: `md/m5_a21_report.md` 1.2 (knobs per row; minimax in α on the octave means,
  31.5 Hz to 8 kHz, 150 Nelder–Mead starts). The rig may survive in `build_linux/scratch/m5` (`a10refit.py`,
  `bp5.cpp` the band probe with the peak columns, `README.txt`) and `build_linux/scratch/a1/p3/ab` (`a10fit.py`,
  `absfit.py`, the targets); scratch is removed by `make clean`, so rebuild from the report if it is gone. The
  persistent instruments: `tests/catch2_tests/ray_test_helpers.hpp`, `test_acoustic_leaf.cpp` (`lw::walk`,
  `lw::stack`), `tools/acoustic` (`band_probe.cpp`, `sabine_to_random.py`, `README.md`).

## Hard rules

- Source changes only in `src/ray_mesh_interact.cpp`, only for M6. Tests and `md/` as below. No other file.
- The order is fixed: analysis (phase 0) → Stephan's decisions → fixtures first (phase 1, against today's library: the
  restated and new cases fail, nothing else moves) → source (phase 2) → documents (phase 3). Each phase ends with a
  report and a stop.
- Do not commit; Stephan commits.
- Measure, don't compute by hand. A fixture describing a traced run goes through `ray_triangle_intersect`,
  `ray_mesh_interact` and `ray_state_update`, or through `ray_progress`, with `ub::OneThread`; references are written
  in the test from independent formulas (the series impedance, the panel over a partitioned cavity), never from
  library code. Bounds 2 to 3 times the measured maxima.
- `[wave]` must not move, and `tests/data/wave_twins.tsv` stays unchanged. If a candidate moves a `[wave]` row, stop
  and report.
- The per-ray state does not grow, and no column is added.
- EM first: `tf` is shared code. State what EM types (0, 1, 2; TE and TM) get on a peaked face under each variant,
  measured; no EM material carries `tfP` today, but a mass-law material can be traced with an EM type.
- No new case may dominate the suite (about 59 s summed per case on an idle machine; a new case well under 0.5 s).
  Report `--durations yes` for every new or restated case, with the load before and after.
- A peak or phase extreme quoted at a frequency comes from a fine sweep (1/96 octave), not a grid point.
- Before anything else, from the repository root: `cmake --build build_linux --parallel`, the three runs of
  `md/wave.md` 0, and check the three counts above. Check the load (`~/.claude/CLAUDE.md`): single-threaded probes,
  one suite at a time, `timeout` on every probe.

## Phase 0 — analysis in scratch (no edits outside scratch)

Fresh scratch folder `build_linux/scratch/m6`. A plain what-if copy of `src/ray_mesh_interact.cpp` first: it must
reproduce the baselines (only the seed line differs).

1. **The reference.** In Python, the series impedance's `r(θ)` (pressure, scalar) and `α(θ) = 1 − |r|²` with the
   power-law part as in D3, at 0° to 85°, both branches; its Paris integral 0°–90° with weight `sin 2θ`. And the
   physics behind it: a panel over a **partitioned** cavity (locally reacting; the cavity impedance
   `−j·ρ₀c₀·cot(k₀d)` with no cosine inside, the panel `R + jωm″`), for the 12.5 mm board, 8.6 kg/m², a mounting
   resistance `0.2·ωm″`, on 25 / 50 / 100 mm. How far is the lumped `ζ` (the `s″/ω` stiffness of `1/(k₀d)`) from the
   `cot` form in `|r|`, phase and Paris value per octave 31.5 Hz to 1 kHz? This says where M6's lumped form holds.
2. **Map the call sites.** Every place a peaked mass-law face's pad or `tf()` is read (the functions above and any
   other), and which branch each event takes: a face met from air, from a coherent layer in front (a lining), from
   inside the held body (N2: reflects nothing), a sheet, a sheet in the solver, the slab path, the plate rule. A marker
   build over the whole suite: which existing cases reach a peaked face (expected: the two peak cases only).
3. **The design choices**, each as a what-if copy where it is cheap (`md/wave.md` 10, 11), with the lines it costs:
   - **D1, what `ζ` is met against.** (a) `ζ·c` with `c` the cosine in air at the face (the decided text; the mass
     cosine of `SHEET_RESOLVE` is the model); (b) `ζ` against the incidence side's immittance `y` that `closed_face`
     already receives (local reaction against whatever lies in front; equal to (a) for a face in air, and needs no new
     argument where `y` is passed). They differ behind a coherent layer in front of the face: measure a peaked face
     behind the dense `eps 2.50 − 0.80j` and the porous `eps 0.31 − 0.20j` lining of 2.4, and behind a refracting
     lossless layer. Say which is physical, and whether (b) bears on A20 (a wall's `tf` behind a porous fill; do not
     widen the scope, record it there).
   - **D2, the bare face.** (a) `r` from `ζ` alone (the face's reactance replaced); (b) `ζ` in series with the bare
     face's own immittance. Measure the difference on two-parameter concrete and glass (expected small: the bare phase
     is 0.009° to 3.1°), on the one-parameter board of the fixtures (bare phase `2θ`), and on a non-rigid mass-law body
     (`eps 1.50`, `m` 20, open). And the open region below the critical angle: how `tf_face` carries `ζ`, and that
     there is no step at the critical angle (a fine sweep in angle across it, `|r|` and phase).
   - **D3, the power-law part.** (a) `|r|² = (1 − tf_bg)·|r_ζ|²` (the decided text; at 0° it absorbs
     `tf_bg + α₀ − tf_bg·α₀`, not M5's `tf_bg + α₀`); (b) `tf_bg` folded into the resistance of `ζ`; (c) additive in
     `α` with the phase of `r_ζ`. For each: the 0° value against M5, the angle shape, the lines.
   - **D4, the gate and EM.** `tfP ≠ 0 && tfF > 0` (as `tf()`), or with `m > 0` as well; what EM types get (the TE and
     TM forms of a locally reacting face; `y_tm` is written with `conj(z)`).
   - **D5, the branch.** `tfQ < 0` selects `1/R′`; the relaxed `require`; `tfP` near one (the roots merge) and
     `tfP > 1` (the clamp of `tf()`); both branches measured.
4. **Measure the variants** through the library (one face through `ray_mesh_interact`, and a traced run): `|r|` and
   phase against the reference at 0°, 10°, 30°, 45°, 60°, 75°, 85° and on a 1/96-octave sweep through the peak; sheet
   against traced solid; tracers 4 and 5 (and 3 for reflection); both mappings; both branches; behind a lining (D1).
   Energy: R + T ≤ 1 from a lossless side on every configuration, summed and followed. Whole suite against the
   baselines: every moved row, `[wave]` 0 rows.
5. **The plate rule.** `a` becomes angle dependent for a peaked face (`PLATE_ABS`). The transmission of a peaked
   panel (the fixture board, and window glass and double glazing with their M5 knobs as the outer faces) at 0° to 85°
   and per octave, today against the variants; whether any of the 13 transmission fits of report 3.6 carries a peaked
   face and how much it moves.
6. **The six rows refitted on M6** (target: the Paris integral of the face, as `md/acoustic.md` 9 asks; the field
   0°–78° value alongside), per D3 variant, over-damped by default; the under-damped branch where it fits better, with
   the `R′` it needs and whether the mass relation allows it (round-1 board: over-damped, `md/acoustic.md` 4 M6).
   Residual per row against M5's 0.0133 / 0.0170 / 0.0075 / 0.0056 / 0.0100 / 0.047, the fitted knobs, and whether
   Jeong's 0.951 bounds any target. Confirm the best fits through `ray_progress` (the band probe) as sheet and traced
   solid.
7. **Restatement and fixtures, proposed.** The two peak cases restated (references from the series impedance, the
   board's knobs and mapping; say whether they move to the two-parameter mapping); a new case for the phase and `|r|`
   of a peaked face at 0° to 85°, sheet and traced solid, both branches, against the reference written in the test;
   the random-incidence value of one A10 row through `ray_progress`. Expected run times.
8. **Recommendation** per decision (D1 to D5, the fixtures), with the reason: physics first, then the library's rules
   (less code over bit-identity; a variant that closes A12 for locally reacting faces fully over one that leaves a mode
   open). Say which commit each candidate's code replaces. Stop for Stephan's decisions.

## Phase 1 — fixtures (after the OK)

- Restate the two peak cases as decided; add the new case(s) (`[acoustic]`, in `test_acoustic_state_update.cpp` or
  `test_acoustic_progress.cpp` as fits). References in the test from the series impedance (and, where decided, the
  partitioned cavity).
- Run against today's library: the restated and new cases fail with the expected deviations, every other line of the
  three baselines unchanged (diff them). Report the failing assertions per case.

## Phase 2 — source

- M6 as decided in `Material` (the pad and its callers), the relaxed `require`, the comments of `tf`, `closed_face`,
  `tf_face`.
- Whole suite: the restated and new cases pass; `[wave]` identical (`test_wave.txt` and the stored twins);
  `[acoustic]` and whole-suite counts; every moved row accounted for; energy checks of phase 0 item 4 on the built
  library. Line count of `src/ray_mesh_interact.cpp` against 5089.

## Phase 3 — documents

- `md/acoustic.md`: 2.6 (the face as a series impedance, the branch, the composition with the power law, "Asserted
  in"), 2.3 (the absorption of an outer face where peaked), 2.7 (the columns table: the phase row), 3.1 (the angle row),
  3.2 (the phase row), 3.8, 3.9, 4 (M6 built), 6 (A12 closed for locally reacting faces, or what remains), 7 (the last
  bullet: a rigid face's absorption is flat in angle, except a peaked one), 10 (built, with the date), the appendix for
  a new case; A20 if D1 bears on it. `md/wave.md` only if a wave convention is touched (it should not be).
- The report: what moved, line counts, run times. `python3 tools/acoustic/check_docs.py` passes (it checks tables,
  source keys, case names, counts and spelling; the counts against regenerated baselines).

## Traps

- The traps of `md/wave.md` 11 and `md/acoustic_manager.md` 9: run a relinked test binary from the repository root;
  never two suites at once; a probe that includes a test file runs every case of it unless given a test spec, and a
  case name with a comma needs `\,`; `pkill -f <pattern>` from a shell whose command line contains the pattern kills
  that shell; `python3 -I` puts no script directory on `sys.path`.
- A material identical to air is air (`μ` 1.000001 for a box of air); a glued sheet lies in front of the body it is
  glued on.
- A rigid backing (`r = 1`) is not the reference for a two-parameter wall: use its half-space or the pad on it.
- The fixture board is in the one-parameter mapping (bare phase `2θ`); a phase reference written for a hard wall
  holds in the two-parameter mapping only.
- The sign of `r` at the peak separates the branches (`(R′ − 1)/(R′ + 1)`): over-damped positive, under-damped
  negative. Check the convention of the library's `r` (pressure, scalar) before comparing phases.
- A followed series of a lossy body between strong reflectors diverges: only summed rows count there.
- Scratch is not kept: put nothing the report depends on only in scratch; the report states every number it uses.
  A reusable instrument goes to `tools/acoustic` if Stephan agrees.
- A plain what-if copy must reproduce the baselines before any candidate is judged against it.
