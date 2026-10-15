# Brief — M5 (the `tf` peak on the resonator law) and A21 (a sheet between two bodies), the first source change

This brief is the whole task description. Read it in full before doing anything, and re-read it at the start of
every phase. Findings go into `md/m5_a21_report.md` (create it), one section per phase, so the next session can
pick them up.

## Why

`md/acoustic.md` is the reference; read sections 2.3, 2.6, 2.8, 3.2, 4 (M5, M6), 6 (A20, A21) and 10 (the decisions
of 2026-10-15). Two items are decided or found and need the library:

- **M5** (decided, Stephan 2026-10-15): the resonant peak of the room-side absorption follows the lumped-resonator
  law. Today (`Material::tf`, `src/ray_mesh_interact.cpp`):
  `tf(f) = tfR·(f/fRef)^tfB + tfP / (1 + (tfQ·(f − tfF)/tfF)²)`, a linear detuning that keeps `tfP/(1 + tfQ²)` as
  `f → 0` and is lopsided in log frequency. Decided: `tfP / (1 + (tfQ·(f/tfF − tfF/f))²)` with `tfQ` the absorber's
  `Q_α` [Masovic_2021_RoomAcousticsNotes, Eqs. 8.57-8.60; Kristl_2023_Verbundplattenresonatoren, Eqs. 3.1-3.18]. Near
  the peak the old form equals the new one for `tfQ_old = 2·Q_α`. The derivation and the octave numbers are in
  `md/acoustic.md` 3.2 and 4 (M5).
- **A21** (found in the M9 session, `md/m9_fixtures_report.md` F2): a sheet co-located with the exit face of one body
  and the entry face of the next (body | sheet | body, glued on both sides) is outside the documented convention
  (`md/wave.md` 6: air is assumed beyond every glued exit), builds without complaint, and creates energy: R up to 48
  with a porous sheet between an air box and an `m = 0` wall, R 0.10 for 0.99 on a mass-law wall. Stephan's triage
  rule: geometry that builds but is invalid gets a graceful failure, asserted. The configuration (from the M9 phase 0
  rig, which is scratch and may be gone; rebuild it from this), faces in the plane x = const as `ub::build` makes them,
  a ray from x < 0, scalar types 4 and 5, 125 Hz and 1 kHz, 0° and 60°:
  - materials: (1) the board leaf `a` 4.691e-5, `e` 573, `thk` 0.0125, loss ratio 0.2 on `μ` (`g = e·0.2/17.98`,
    `h` 1, `fRef` 1), optional; (2) the wall, two-parameter concrete `a` 3.850e-6, `e` 1901, either `m = 0` or `m` 20
    with `fRef` 5.0212e-5, `tf` 0; (3) the air box `a` 1, `e` 1.000001; (4) a porous sheet, `thk` 0.015, any porous
    wave medium of the two-parameter mapping (the rock-wool columns of the fill in *Leaf - a porous fill at the back of
    the box …*, `tests/catch2_tests/test_acoustic_leaf.cpp`, will do);
  - faces (x, material, flipped): (0, 1, no) if the leaf is present, (0, 3, no), (0.010, 3, yes), (0.010, 4, no),
    (0.010, 2, no), (0.6, 2, yes): air box 0 to 10 mm, the porous sheet co-located with the box's back face and the
    wall's front face, the wall from 10 mm on;
  - the reference: the transfer matrix of fluid layers (`lw::stack` in `test_acoustic_leaf.cpp`) of
    [leaf] | air 10 mm | porous 15 mm on the concrete half-space. Note that the R 48 case involves **wave media only**: A21 may sit on the wave side, behind no `m > 0`
  gate.

## Facts at the start

- Baselines after the M9 work (`md/wave.md` 0): `[wave]` 165110 assertions in 179 cases, `[acoustic]` 36705 in 51,
  whole suite 1311364 in 1137. If `git status` shows the M9 work (`tests/catch2_tests/test_acoustic_leaf.cpp`,
  `md/acoustic.md`, `md/wave.md`, `md/m9_*`) uncommitted, stop and ask Stephan before anything else.
- The cases that carry the peak: *Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption
  of a mass-law face* (`test_acoustic_state_update.cpp`), *ray_progress - the plate rule and the absorption peak at
  every frequency of a band* (`test_acoustic_progress.cpp`). Both write their reference from the column definition
  (the old law) with the board `tfP` 0.25 at 150 Hz, `tfQ` 1.5. No `[wave]` case carries `tfP` (the `[wave]` baseline
  did not move when A10 was built). The OBJ / API column lists are not affected (no new column).
- What-if builds, marker builds and walker probes: `md/wave.md` 10 and 11 describe how to make them. Scratch
  folders under `build_linux/scratch` are untracked and removed by `make clean`; the rigs named in `md/acoustic.md`
  and the reports (`a1`, `ac2`, `m9`) may be gone. Rebuild what you need from the descriptions; the traced-run
  instruments that survive are in `tests/catch2_tests/ray_test_helpers.hpp` and `test_acoustic_leaf.cpp`
  (`lw::walk`, a loop-based walker; `lw::stack`, the fluid-layer transfer matrix) and in `tools/acoustic`
  (`band_probe.cpp`: a band through `ray_progress` on a palette and plane faces; `README.md`).
- The A10 fits of round 1 (report 3.8: every lining and pane row within 0.017 in α, cork board 0.076): the rows are
  in `md/acoustic_round1_report.md` 1.2, 1.6 and 1.10 (octave absorption 125 Hz–4 kHz), the fit was minimax in α on
  the octave means over the field 0°–78°; the residuals of each form are in report 3.8.

## Hard rules

- Source changes only in `src/ray_mesh_interact.cpp`, only for M5 and A21. Tests and `md/` as below. No other file.
- The order is fixed: analysis (phase 0) → Stephan's decision on A21 → fixtures first (phase 1, run against today's
  library: the restated and new cases fail, nothing else moves) → source (phase 2) → documents (phase 3).
- Do not commit; Stephan commits.
- Measure, don't compute by hand. A fixture describing a traced run goes through `ray_triangle_intersect`,
  `ray_mesh_interact` and `ray_state_update`, or through `ray_progress`, with `ub::OneThread`; references are written
  from independent formulas, never from library code.
- `[wave]` must not move. If an A21 candidate moves a `[wave]` row, stop and report.
- No new case may dominate the suite (whole suite about 59 s summed per case on an idle machine; a new case well
  under 0.5 s). Report `--durations yes` for every new or restated case.
- Phases end with a report and a stop; the next phase starts after Stephan's OK.
- Before anything else, from the repository root: `cmake --build build_linux --parallel`, the three runs of
  `md/wave.md` section 0, and check the three counts above. Check the load (`~/.claude/CLAUDE.md`): single-threaded
  probes, one suite at a time, `timeout` on every probe.

## Phase 0 — analysis in scratch (no edits outside scratch)

Fresh scratch folder `build_linux/scratch/m5`.

**M5.**
1. A what-if copy of `src/ray_mesh_interact.cpp` with the new law (one line in `Material::tf`; check that
   `f/tfF − tfF/f` behaves at the clamp and for `tfF > 0` only, as today). Plain copy first: it must reproduce the
   baselines. Whole suite against the baselines: which rows move. Expected: only the two peak cases; `[wave]` 0 rows.
2. The A10 rows of report 3.8 refitted on the new law (the rows and the fit as described above, the law swapped):
   the largest residual per row against today's 0.017 / 0.076, and the same rows averaged over 0°–90° (Paris), the
   comparison `md/acoustic.md` 9 asks for. And the low bands the law changes: the fixture board's 31.5 / 63 Hz values
   on both laws (`md/acoustic.md` 3.2 gives 0.104 / 0.142 today against 0.020 / 0.079, with `Q_α` 0.75).
3. The restatement of the two cases: propose `tfQ` for the fixture board (0.75 keeps its near-peak width), the
   reference from the new column definition, the bounds unchanged.

**A21.**
1. Reproduce the A21 configuration above (through the walker of `test_acoustic_leaf.cpp` in a scratch probe that
   includes that file, `md/wave.md` 11): R and T against the reference, energy created or not.
2. A marker build (`md/wave.md` 11): which branch of `ray_mesh_interact` / `ray_state_update` the body | sheet | body
   event takes (sheet case, glued flags, which faces are co-located), for `m = 0` everywhere and with a mass-law body
   on either side, sheets with and without `m`, tracers 4 and 5, an EM type. And over the whole suite: does any
   existing case reach that branch with three co-located faces? (If yes, a graceful failure moves rows: list them.)
3. The options, each measured as a what-if where it is cheap:
   - (a) detect body | sheet | body and kill the ray (resolved type 0), as the library already does for a sheet stack
     of two different materials (`ray_state_update`'s type table: "detected but has no physics");
   - (b) resolve it as a sheet glued on both sides (near body | sheet | far body through the solver), if the solver
     carries it; its error against a transfer matrix;
   - (c) record it as a limitation only.
   For each: lines, rows moved (whole suite), `[wave]` rows, whether energy is created anywhere afterwards
   (R + T ≤ 1 from a lossless side on the A21 configurations).
4. Recommendation, with the reason. Stop for Stephan's decision.

## Phase 1 — fixtures (after the OK)

- Restate the two peak cases to the new law (references from the column definition, the fixture `tfQ` as decided),
  in place, case names unchanged unless the statement changes.
- A21: one new `[acoustic]` case, or `[wave]` if the configuration is wave media only and Stephan agrees, that asserts
  the decided behavior (for (a): the ray is killed and nothing is created; for (b): the transfer matrix and R + T ≤ 1).
- Run against today's library: the restated and new cases fail with the expected deviations, every other line of
  the three baselines unchanged (diff them). Report the failing assertions per case.

## Phase 2 — source

- M5: the line in `Material::tf` and its comment (the law, `tfQ` as `Q_α`). A21 as decided.
- Whole suite: the restated and new cases pass, `[wave]` identical (`test_wave.txt` and the stored twins),
  `[acoustic]` and whole-suite counts, every moved row accounted for. Line count of `src/ray_mesh_interact.cpp`
  against 5082.

## Phase 3 — documents

- `md/acoustic.md`: 2.6 (the formula, `tfQ` as `Q_α`, the restated numbers, "Asserted in"), 3.2 (the peak-shape row:
  expressible), 4 (M5 built), 6 (A21 closed or recorded), 10 (built, with the date), the appendix if a case was added;
  `md/wave.md` if A21 touches a wave convention (section 6 or 8).
- The report: what moved, line counts, run times.

## Traps

- The traps of `md/wave.md` 11, and from the M9 work: a glued sheet lies in front of the body it is glued on (an
  isolated sheet about its mesh plane); a material identical to air is air (`μ` 1.000001 for a box of air); a
  recursive walker overflows the default stack on a followed series of thousands of events (use `lw::walk`, which
  loops); a probe binary that includes a test file runs every case of it unless given a test spec; `python3 -I` puts
  no script directory on `sys.path`; `pkill -f <pattern>` from a shell whose command line contains the pattern kills
  that shell.
- Scratch is not kept: put nothing the report depends on only in scratch; the report states every number it uses.
- A case name with a comma needs `\,` in a test spec.
- A plain what-if copy must reproduce the baselines before any candidate is judged against it.
