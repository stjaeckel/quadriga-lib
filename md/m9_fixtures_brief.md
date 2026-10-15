# Brief — M9: leaves as wave-medium sheets, measured and guarded by fixtures

This brief is the whole task description. Read it in full before doing anything, and re-read it at the start of
every phase. Findings go into `md/m9_fixtures_report.md` (create it), one section per phase, so the next session can
pick them up.

## Why

`md/acoustic.md` is the reference for the acoustic add-on; read section 1 (the two-parameter mapping), 2 (the machine
as built), 3.1, 3.2 and 3.4 (what the physics requires), 4 (candidate M9), 6 (A18, A19) and 10 (the decisions of
2026-10-15). In short:

- Under the two-parameter mapping (N7: `ε = n·z`, `μ = n/z`, so `μ = ρ/ρ₀`) a thin layer of a wave medium has the
  series impedance `jωm″/Z₀`: a sheet with `m = 0` is a limp leaf. Measured on 4 mm glass within 0.002 dB of the
  limp mass law, 0° to 85°, 500 Hz (`build_linux/scratch/ac2/sprobe.cpp`, `s.out`).
- A18 (scratch, `ac2/a18.py`, the a1 walker through the library at 50 Hz–1 kHz, 0° to 60°, against a transfer
  matrix of fluid layers): a 12.5 mm board (8.6 kg/m², a mounting resistance as a loss of `μ`) over 25 mm on
  concrete is on the reference within 2e-6 in R over a meshed gap and within 6.7e-5 glued on a meshed box of
  near-air material (`ε` 1, `μ` 1.000001), the box summing the cavity in one event; its peak moves with angle
  (125 / 160 / 200 / 250 Hz at 0° / 30° / 45° / 60°). 4|16|4 of two glass leaves on such a box is within 0.007 dB.
  The meshed gap needs 800 to more than 4,000 events per ray.
- Decided (Stephan, 2026-10-15, `md/acoustic.md` 10): M9 is the route for an unpartitioned construction with a known
  build (a leaf over a cavity, a double leaf, a curtain, an MPP over an open cavity); M7 is dropped; the next step
  is this one: the A18 runs as fixtures, and the open measurements.

The task: confirm the route on the configurations a scene will use, measure what A18 left open, and guard what holds
with `[acoustic]` fixtures. No library change is expected; a fixture that fails on today's library is a finding.

## Facts at the start

- Baselines (repository root, `md/wave.md` section 0): `[wave]` 165110 assertions in 179 cases, `[acoustic]` 32697
  in 46, whole suite 1307356 in 1132. Check them first (below).
- Scratch rig `build_linux/scratch/ac2` (`README.txt` lists every file): `w3` is the a1 walker (`walker3.cpp`, it
  includes `tests/catch2_tests/test_acoustic_state_update.cpp`) rebuilt on today's library with
  `buildw.sh - walker3.cpp <absolute out>`; spec format in the head of `walker3.cpp` (a negative material count
  adds `μ′`, `μ″` per material; `μ″ = mui / f_GHz`). `a18.py` writes the specs, runs `w3` and holds the reference
  (`ref()`, fluid layers in (p, v_n), the concrete half-space as backing). `a18.out`, `a18_sum.out` are the results.
- The test instruments are in `tests/catch2_tests/ray_test_helpers.hpp` (`md/wave.md` 11): `ub::traced`,
  `ub::build`, `bf::walk` (sums every path back in air), `cf::walk` (adds a sink medium and a seeded start medium),
  `lump_palette`, `ub::OneThread`.
- The band probe of round 1, `build_linux/scratch/a1/p3/rp/rp3.cpp`, takes one material only.

## The boundary question (decide in phase 0, report it)

`md/wave.md` 0: a case in which no material with `m > 0` is evaluated is a wave case. The A18 runs used only wave
media (the leaf, the air box and a concrete half-space, all `m = 0`), so as they stand they are wave physics and
would belong to `[wave]`. A real scene puts the box on a wall that carries the mass column (a rigid mass-law body
with `tf`). Measure both backings (phase 0, item 1). Proposal: the fixtures use a mass-law wall, which makes them
acoustic cases in `test_acoustic_*` and leaves `[wave]` untouched; if the mass-law wall changes the leaf's result,
that is a finding to report before any fixture is written.

## Hard rules

- No change to `src/` or to anything outside `tests/` and `md/`. No change to an existing case or to the physics it
  asserts. New cases only, tagged `[acoustic]`.
- Do not commit; Stephan commits.
- Measure, don't compute by hand. Every reference in a fixture is written from an independent formula (the
  transfer matrix of fluid layers, the limp mass law), never from library code. A fixture that describes a traced
  run goes through `ray_triangle_intersect`, `ray_mesh_interact` and `ray_state_update` (`bf::walk`, `cf::walk`),
  or through `ray_progress`, with `ub::OneThread`.
- Phases end with a report and a stop; phase 1 starts after Stephan's OK on phase 0.
- Before anything else, from the repository root: `cmake --build build_linux --parallel`, then the three runs of
  `md/wave.md` section 0, and check the three counts above.
- Check the load (`~/.claude/CLAUDE.md`): single-threaded probes, one suite at a time, `timeout` on every probe.

## Phase 0 — measurements in scratch (no test edits)

Fresh scratch folder `build_linux/scratch/m9`; copy what is needed from `ac2`. Reproduce `ac2/a18_sum.out` first.

1. **The backing.** The leaf on an air box on (a) the concrete half-space of A18 (`m = 0`), (b) a rigid mass-law
   wall (two-parameter concrete, `m` 20, `fRef` 5.0212e-5, `thk` 0, `tf` 0 and `tf` 0.02), as a traced solid and
   with the box glued on it. R against the reference (rigid backing, or the wall's `tf` as a resistive backing),
   50 Hz–1 kHz, 0° to 60°, summed and followed.
2. **Tracer 5** (refracting) against tracer 4 on the same configurations.
3. **A band through `ray_progress`.** Extend `rp3.cpp` to a palette and a face list (as `walker3.cpp` takes them);
   a band of frequencies across the resonance on the base frequency's geometry against the single frequencies,
   both tracers, the leaf on the box and 4|16|4.
4. **A porous fill in the box.** The box filled (or partly filled) with a porous wave medium (Miki columns at a
   measured σ, two-parameter mapping) against the transfer matrix leaf | porous | air | wall; and the MDF
   membrane of [Shearer_2016_AbsorptionBelow100Hz] (4.65 kg/m², 95 mm, empty and filled), its measured peaks in
   the 100 and 80 Hz thirds.
5. **A19's shallower box.** A framed leaf's resonance rises about 1.34 times (A19). Test a box of `1/1.34²` the depth:
   where `f₀` lands, what the peak height and width do, what the angle shift does, against the framed cases of
   `ac2/a19.out` (Shearer, Sakagami 3 mm plywood over 30 / 90 / 150 mm).
6. **A curtain.** A fabric is an airflow resistance `R_s` in parallel with its mass `jωm`
   [Pieren_2015_MultilayerCurtains]; a fluid leaf gives a series impedance. Find whether the `μ` columns (power
   laws in frequency) can carry the parallel form over 125 Hz–4 kHz within 0.05 in α for one fabric of
   [DelRey_2013_CurtainFullness] or the G-100 reports, and through the walker.
7. **The cost.** Events per ray for the box and for the meshed gap, per configuration; the rule of use that follows.

Report per item: the configuration, the reference, the largest deviation per quantity, the bound a fixture would
assert, and what did not run.

## Phase 1 — fixtures (after the OK)

In a new file `tests/catch2_tests/test_acoustic_leaf.cpp` (CMake globs the folder) or appended to
`test_acoustic_state_update.cpp`, as Stephan prefers; `[acoustic]`, `ub::OneThread`, the reference in the case:

- *Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf* (mass law and phase, 0° to 85°);
- *Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle* (the angle shift, summed
  and followed equal);
- *Leaf - two leaves on a box of air resonate at the mass-air-mass frequency* (4|16|4);
- *ray_progress - a leaf on a box of air at every frequency of a band*;
- the items of phase 0 that hold, each with its bound; the ones that do not hold recorded in the report, not
  asserted.

Then: the `[acoustic]` count grows by the new cases only, every existing line of `test_acoustic.txt` and of
`test_wave.txt` is unchanged (diff them), the whole suite count grows by the same number.

## Phase 2 — documents

- `md/acoustic.md`: a subsection 2.8 *Leaves as wave-medium sheets* (the rule of use: the leaf a sheet with
  `m = 0`, glued on a meshed box of near-air material, its loss a loss of `μ`; the limits: no bending, the edge
  rise of A19, the cost of a meshed gap), its "Asserted in" lines, the rows of section 3 that move to expressible,
  A18 closed, the appendix of cases. The decisions stay Stephan's; write proposals where something is open.
- `md/wave.md` 9: one line that the acoustic route M9 rests on the wave media of the two-parameter mapping, so a
  change on the wave side is checked against the new cases too.
- Report: what holds, what was recorded, the counts, the line counts of the new file.

## Traps (from `ac2` and earlier rigs)

- The walker binary runs every case of the included test file unless given the test spec `"P4 walker"`; run it
  from the repository root.
- A lossless cavity needs thousands of events and the walker recurses once per event: `ulimit -s unlimited`, or it
  segfaults (a walker limit, not a library bug). Cap the depth and report the dropped power: where it exceeds the
  transmitted power the sum is not converged.
- A glued sheet lies in front of the solid it is glued on (an isolated sheet about its mesh plane): the cavity is
  the box depth, not the face distance minus the leaf.
- A material identical to air is the same medium: the box needs `μ` 1.000001 (or `ε` off by as little).
- `python3 -I` puts no script directory on `sys.path`; insert it.
- `pkill -f <pattern>` from a shell command whose text contains the pattern kills that shell; kill by pid.
- One ray at a time costs twenty times the work without `ub::OneThread`; a segment must reach past every remaining
  face (`md/wave.md` 11).
