# Step 3 of the acoustic project: the material tables (manager file)

The memory of the step-3 manager. Kept current after every step; a new manager session reads this file first, then
`md/materials_kickoff.md` (the kick-off, until it is deleted in phase 4). Public and tracked: no private details (no
rates, hours, money, personal circumstances, nothing about whose room the listening room is). Briefs and reports in
`md/materials/` are untracked working files; answers that are personal stay there. Old rule labels of the research
handoff (A13, A14, A15, A19, P7, M6) are used here only as working shorthand; public documents and commit messages name
the rule in words.

## 1. Roles

- **Stephan** (the maintainer) decides every open point, reviews every change and commits. Wants plain physical
  explanations first (what the quantity is, what happens, how large in a typical room), options laid out, a
  recommendation, and a paste-ready message. Asked with the question tool, recommendation first, explanation written
  before the question (a question ends the turn in the desktop app).
- **The manager** (this session) writes briefs, spawns one worker agent at a time with the Agent tool, verifies every
  claim a decision rests on (rebuild, rerun, read the CSV and the test), keeps this file, prepares decisions and commit
  messages. Never commits, never edits `src/`, `include/`, `api_*` or `/sjc/accoustic_research`.
- **Worker agents**, spawned by the manager, strictly serially in this checkout:
  - the **rig agent** (phase 1): the test rig in `tests/catch2_tests/test_acoustic_materials.cpp`;
  - **analysis agents** (read-only apart from their report; phase 2 and rule questions): research data, conversions,
    sizes;
  - **material agents** (phase 3): one material per agent, from data to CSV row and test case.

  Each gets its brief file and returns a short report (about 300 words); the substance is in
  `md/materials/<id>_report.md`. An agent does one phase and returns; the manager continues the same agent with
  SendMessage for the next phase (its context intact) or starts a fresh one.
- **The research manager** (a session in `/sjc/accoustic_research`) takes research requests that Stephan forwards
  (section 7).
- **An engine manager** only if Stephan installs one for an engine change (section 8).

## 2. Decisions taken at the kick-off (2026-10-22)

- **Deliverable:** one shipped acoustic material table, a CSV in `data/` in the format of
  `tools/data_formats_md/format_materials.md` (name: section 10, proposed). Every row is asserted by a test that reads
  the CSV. At the end it replaces the preliminary rows of page 11.4. The built-in EM table of `obj_file_read` and
  `data/materials_rf.csv` are not touched.
- **Material scope:** one or more rows per acceptance class (the ten classes of notes 8.4 and the resonant family of
  notes 9.3), plus the materials of the listening room.
- **Phases:** 0 setup (stop); 1 the test rig (stop); 2 the material list, with the room inventory worked out with
  Stephan (stop); 3 material by material until every listed material and its meaningful combinations are in; 4 review,
  cleanup and lessons (`md/acoustic_material_notes.md`, page 11.4 replaced, the contradicted statements of handoff
  section 5 corrected on the page and in the engine notes, the tools README, the transitory files deleted).
- **Unit of work in phase 3:** one material per worker agent, serial; the manager checks and reports; Stephan reviews
  and commits each material.
- **Two-tier oracle:**
  1. an independent physical model (plate theory, a transfer matrix, Miki / Komatsu / JCA, the leaf on a box),
     written in the test, never taken from library code (`txlib.py` is a copy of the library, not a reference),
     checks that the library computes the columns right;
  2. the measured reference after its coefficient-type conversion (the tiered rule for absorption references, page
     10.5) checks that the columns represent the material, per octave, within the guideline (about ±3 dB in
     transmission, ±0.05 in absorption); bands without a measurement take the model spread; every deviation is
     recorded.
- **Tests:** one file, `tests/catch2_tests/test_acoustic_materials.cpp`, tagged `[acoustic][materials]`, one
  `TEST_CASE` per material, about 0.3 s per case at most (a coarse angle grid, validated once against the fine grid in
  `tools/acoustic`).
- **Open rules** are decided at the stop of the first material that needs them (section 6); the materials are ordered
  so that the rules settle early.
- **Engine closed:** a material that cannot be expressed is recorded as a miss with its size; an engine change only
  after Stephan's OK, in a separate engine manager session, with the engine workflow (`[wave]` untouched); the
  confirmed materials become its regression tests.
- **Research requests:** a material without its reference waits; the manager continues with another (section 7).

## 3. Workflow

**Per phase.** Brief (`md/materials/<id>_brief.md`, from `md/materials_agent_template.md`) → one agent → the
manager's checks (section 4) → report to Stephan (what checks out, what does not, decisions in the form below, a commit
message) → his review and commit → this file updated.

**Per material (phase 3).**
1. The manager writes the brief: material and class, reference rows (keys, coefficient types, bands), the rules that
   apply, the tier-1 model, open rules it will meet.
2. The agent: data → columns by the decided rule → tier 1 (the library through the rig against the independent model)
   → tier 2 (the columns against the converted measurement per octave) → the `TEST_CASE` → the CSV row. Stops where an
   open rule is met, with the options measured.
3. The manager: rebuilds, reruns `[acoustic]` and `[wave]`, reads the CSV row and the test, recomputes the claims a
   decision rests on, runs the checklist.
4. Report to Stephan: the row, its tier-1 and tier-2 residuals per octave, recorded deviations, decisions (if any), the
   commit message and the `git add` line. Stephan commits; the baselines are regenerated after the commit.

**Decisions** go to Stephan as: what is decided; the options; trade-offs; what is correct (physics first, then the
library's rules); a recommendation; a paste-ready message.

**Commit messages:** title "<area>: <what changed>", body in plain physical terms with the measured numbers and the
suite counts, no internal labels (no rule labels, no `md/` documents, briefs, reports, phases), trailer
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## 4. The review checklist

Run on every agent result; the report says which items were checked.

- **Baselines:** `[wave]` and `[acoustic]` rows of the existing cases unchanged (`tools/acoustic/rowdiff.py` against
  `test_wave.txt`, `test_acoustic.txt`; `QD_DIAG_RAY=1`), the stored twins unchanged; the whole suite (`QD_DIAG=1`,
  `test.txt`) only the new case's rows added; counts add up (new assertions = the difference).
- **Time:** the new case within about 0.3 s (`--durations yes`, CPU time; load stated).
- **Tier 1 independent:** the reference is written in the test from a closed form or a transfer matrix, never from
  library code or `txlib.py`; bounds 2 to 3 times the measured maxima.
- **Tier 2 types right:** every reference row's coefficient type checked against `data/material_properties.csv`;
  Sabine values converted or marked by the tiered rule; compilation copies (63 Hz = 125 Hz, 8 kHz = 4 kHz) treated as
  no measurement; values above one not used as `tf`.
- **Peaks and dips** quoted as frequencies come from fine sweeps (1/96 octave), never grid points.
- **CSV = test:** the CSV row is the row the test reads and asserts (the test reads the file; no copy of the numbers in
  the test).
- **Documents:** `python3 tools/acoustic/check_docs.py` passes; tracked files carry no private details.
- **Resources:** at most 12 threads, compiler and tests; one suite at a time; numpy workers with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`; no heavy tools.
- **Hard rules kept:** no change under `src/`, `include/`, `api_*`; nothing written in `/sjc/accoustic_research`;
  nothing committed by the agent.
- **Energy:** R + T ≤ 1 from air for every construction the case builds; a sheet and its traced solid agree.

## 5. The material list

Filled in phase 2. Status: proposed / waiting for research / in work / confirmed / recorded miss.

| # | class | material (CSV row) | reference keys | status |
| --- | --- | --- | --- | --- |
| | 1 light single panels | | | |
| | 2 heavy walls | | | |
| | 3 double and triple glazing | | | |
| | 4 porous absorbers on a rigid wall, carpets | | | |
| | 5 linings | | | |
| | 6 plaster, brick, plaster | | | |
| | 7 air, water | | | |
| | 8 panel absorbers | | | |
| | 9 curtains | | | |
| | 10 microperforated panels | | | |
| | resonant family (perforated, slotted panels) | | | |
| | listening room | | | |

## 6. Open rules

Each is decided at the stop of the first material that needs it; record the decision, the date and the material.

| rule | question | decided |
| --- | --- | --- |
| reference row per class | which measured row is the tier-2 reference where compilations spread by 2 to 4 times | |
| fitted-σ check | a converted absorption reference is marked when its fitted flow resistivity is off a measured one by more than about 1.5 times (notes 9.3) | |
| interlaboratory porous rows | the two Scrosati rows stay out until a polyester and a framed-sample model exist (notes 9.3) | |
| resonant absorbers as one family | panels and membranes, perforated and slotted panels, MPP in the acceptance (notes 9.3) | |
| bending loss as a resistance | needed if `coiQ` comes from a measured loss factor (notes 9.3; page 11.1, 11.2, 11.4 re-measured if `coiQ` changes meaning) | |
| fabric relaxation on μ | curtains: random-incidence α reached, single angles off by up to 0.35 (notes 9.3) | |
| Sabine to random for panels and MPP | a conversion as for porous layers, or classes 8 and 10 marked under the tiered rule | |
| the leaf's loss ratio `rr` | which normalization transfers to room-size leaves (R/Z₀ or R/(ω₀m″)) | |
| damping branch of a peaked face | over-damped default against measured empty-cavity R′ at or below 1 | |
| `alpha` on a porous layer | its role, if any, next to `a`–`h` | |
| membrane constant with a bonded fill | 60 or 50 when the fill is bonded to the plate (handoff 5) | |

## 7. Research requests

Format (handoff section 7): the column or class; the quantity and its coefficient type; the bands; whether closed
sources are acceptable; urgency and whether step 3 can proceed without it. Stephan forwards to the research manager.

| sent | request | status |
| --- | --- | --- |
| — | none yet | |

## 8. Engine items raised

A material that cannot be expressed: what, the size of the miss, the column or physics missing. None leads to work
without Stephan's OK.

| raised | material | miss and size | status |
| --- | --- | --- | --- |
| — | none yet | | |

## 9. Current state

- **2026-10-23, phase 0.** HEAD cebd0c0 (v0.12.3). `build_linux` reconfigured (it was absent): the generic build,
  g++-12, AVX2 and CUDA off, tests on, MATLAB, Octave and Python off; built with `--parallel 12`.
  Baselines (untracked, regenerated 2026-10-23 from cebd0c0): `[wave]` 163474 / 178 and `[acoustic]` 42741 / 59,
  every row identical to the files of 2026-10-21 (`rowdiff.py`: 141128 and 112451 numbers equal); the whole suite
  1346857 / 1302 (the bug-fix series of 2026-10-21/22 added 38 cases outside the ray tags; the kick-off's
  1344398 / 1264 is the count at a58c18f). Wall time at 12 threads, load 5 to 7: 14.6 s, 0.9 s, 28.8 s.
  `check_docs.py` passes.
- Files of phase 0: this file, `md/materials_agent_template.md`.
- Phase-1 answers (2026-10-23): the CSV is `data/materials_acoustic.csv`; the rig copies the `lw` helpers it needs
  into its own file; combinations as defined in section 10.
- Next: Stephan commits the phase-0 files; then the rig brief (`md/materials/rig_brief.md`).

## 10. Scope (proposed in phase 0, confirmed by Stephan 2026-10-23)

**The CSV.** `data/materials_acoustic.csv`, next to `data/materials_rf.csv`. The old name `materials_ac.csv` is avoided
on purpose: it was the outdated one-parameter table removed in 8d89b84, and the history should not mix the two. The
file is read by `obj_file_read` with an empty `.obj` name (only the table), so it obeys the reader:
- only `name` is text; every other cell is a number; no comments, no class or source columns (the class, the sources
  and the residuals live in the test case and in the notes);
- every row fills every column that has a non-zero default (`fRef`, `a`, `e`, `side`), since an empty cell reads as 0
  (and `side` 0 is the infinite plate);
- names are lowercase `snake_case` with the build in millimeters, no dots, no commas (`glass_4mm`,
  `concrete_cast_152mm`, `glass_wool_50mm`); a dot would be cut to a base name by the material lookup;
- frequency columns in GHz with the acoustic mapping of page 10.1 (343 m/s, 8.7403e-4 GHz per Hz).

**The test file** `tests/catch2_tests/test_acoustic_materials.cpp`, tags `[acoustic][materials]` (names only):
- `namespace mt` (the rig):
  - `mt::table()` reads the CSV once; `mt::row(name)` one row as columns;
  - `mt::OCT` the octaves 31.5 Hz to 8 kHz; the coarse angle grid and its weights;
  - the three construction types: `mt::body_in_air(row, sheet | solid, side)`, `mt::on_backing(layers, wall)`,
    `mt::leaf_on_box(leaf, depth, fill, wall)`;
  - `mt::Out` per octave: `alpha` (random incidence, Paris integral, room side) and `tl` (transmission at the
    laboratory side, 0°–90°);
  - tier-1 references written in the file: `mt::ref_mass_law`, `mt::ref_plate`, `mt::ref_tmm`, `mt::ref_miki`,
    `mt::ref_komatsu`, `mt::ref_jca`, `mt::ref_leaf_box`;
  - tier 2: `mt::Measured` (key, coefficient type, bands, values), `mt::tier2(out, measured, tolerance)` with the
    deviations printed;
- `TEST_CASE("Materials rig - ...")`: the rig's self-checks (phase 1);
- `TEST_CASE("Material - <name>")`: one per material (phase 3);
- `TEST_CASE("Combination - <name>")`: one per combination.

**Combinations.** A combination is a construction assembled in geometry from rows of the table (a porous layer glued
on a wall, a leaf on a box of air in front of a wall, a lining on a wall, plaster | brick | plaster). It is meaningful
when (1) it is a reference construction of an acceptance class or occurs in the listening room, (2) its physics differs
from its parts (coupling: a layer changes the wall's face, a leaf resonates with its cavity), and (3) a measurement or a
tier-1 model of the whole construction exists. A combination is no CSV row: its rows are named for the material, and
rows that exist only to build a combination carry their role (`leaf_plywood_3mm`, `cavity_air`); the combination is
named in its test case and in the notes ("Combination - glass wool 50 mm on concrete 152 mm"). A construction measured
only as a whole and fitted as one lumped body (double glazing, a furred lining fitted as one wall) is a material row
named for the construction (`glazing_4_16_4`).

## 11. Roadmap

1. Phase 1: the rig (brief from the template's rig variant after Stephan's answers).
2. Phase 2: the material list (analysis agents; the listening room's inventory with Stephan).
3. Phase 3: materials in the order that settles the open rules early.
4. Phase 4: notes, page 11.4, corrected statements, README, cleanup.

Leftovers of step 2 to settle where they fall: `tools/acoustic/band_probe.cpp` has no `attB` column (every fitted row
uses one); page section 9 says `irr_glass` follows TR 38.901, though only its `att` does; if `coiQ` changes meaning,
page 11.1, 11.2 and 11.4 are re-measured.

## 12. Traps

- **Two sound speeds in the tests:** `ac2rf()` / `AC2RF` in `ray_test_helpers.hpp` use 342.77 m/s; `lw::GHZ`
  (`test_acoustic_leaf.cpp`), `txlib.py` and the page use 343 m/s. The table and the rig use 343 m/s (page 10.1).
- **`lw` is not shared:** `lw::stack`, `lw::walk`, `lw::Mtl`, `lw::peak_f` live in `test_acoustic_leaf.cpp`, not in
  `ray_test_helpers.hpp`. Decided 2026-10-23: the rig copies the parts it needs into
  `test_acoustic_materials.cpp`; `test_acoustic_leaf.cpp` and the header stay as they are.
- **Reader:** an empty cell reads 0, not the default; a material file replaces the built-in table; a text cell in a
  numeric column is an error; a name's part after the first dot is a variant of its base name.
- **A material identical to air is air:** a box of air needs `e` 1.000001.
- **Research data:** generated files (never patch a copy; report a wrong row); PTB 63 Hz values equal to 125 Hz and
  compilation 8 kHz values equal to 4 kHz are copies; Sabine and placement values above one are no material
  property; the property name `alpha_random` covers every diffuse or unstated type, the `coefficient_type` column
  decides.
- **Runs:** test binaries from the repository root; a case name with a comma needs `\,` in a test spec; a file filter
  needs `-#`; single-ray fixtures on one OpenMP thread (`ub::OneThread`); `ray_progress` requires
  `thin_slab_threshold` ≤ 1; `cd` in a Bash call moves the session's directory.
- **Fits:** bound and densely sample; select on the evaluation grid; the mass level bound must reach about 3200 for a
  423 kg/m² wall; clip τ at one per frequency, not per band; coincidence needs about 4000 angles.

## 13. Lessons

- (phase 0) `build_linux` can be absent at the start of a session; reconfigure it as the generic build before
  regenerating baselines.
