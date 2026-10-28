# Step 3 of the acoustic project: the material tables (manager file)

The memory of the step-3 manager. Kept current after every step; a new manager session reads **section 0 first**, then
sections 1–8 and 12, then the three step-3 documents named in section 0. Section 9 is the dated log of how the
decisions came about (read it when a decision's reason is needed). `md/materials_kickoff.md` is the original plan of
step 3; where it differs from section 0, section 0 holds (it is deleted in phase 4). Public and tracked: no private details (no
rates, hours, money, personal circumstances, nothing about whose room the listening room is). Briefs, reports, agent
worktrees, scratch files and the baselines live outside the repository in the agents' folder `/home/stephan/quadriga-lib-agents` (since
2026-10-28: the repository's disk is backed up with history; log entries before that date name the old places
`md/materials/`, `.claude/worktrees/`, `build_linux/scratch/` and the repository root); answers that are personal stay
there. Old rule labels of the research
handoff (A13, A14, A15, A19, P7, M6) are used here only as working shorthand; public documents and commit messages name
the rule in words.

## 0. Start here: state of phase 3 (2026-10-28)

**Where step 3 stands.** Phases 0–2 are closed. Phase 3 builds the material rows in batches (option (b), Stephan
2026-10-27): one manager, at most three material agents at a time, one row family each, every decision collected per
batch and brought to Stephan in five parts. The engine stays closed until every listed material is in (Stephan,
2026-10-26). Committed so far:

| batch | families (wish-list entry) | commit |
| --- | --- | --- |
| 0 | `concrete` (1.1); the rig moved into a header, cases per group | a97c4ae |
| — | engine-limit marks by name; tests and tools cite published material only | 4641966 |
| — | research rows cited by stable id; the research answer folded into the wish-list | 063f05b |
| 1 | `masonry_heavy` (1.2), `glass_4mm` / `glass_6mm` (1.6), `glass_wool` (5.1); JCA in the rig | a73d1a2 |
| 2a | `masonry_hollow` (1.3), `door_solid_40mm` + `door_hollow_core_40mm` (1.23, 1.24), `rock_wool` (5.2) | cbe2ae3 |
| 2b | `masonry_light` (1.4), `glazing_double_24mm` + `glazing_triple_36mm` (1.7), `pu_foam` (5.3; 3.6 to a padding-foam row) | merged, waiting for Stephan's commit |

Earlier commits of step 3: 5be85fb, 13e4e45, 19323d9 / be65346 / bd0a170 (wish-list and guidelines), 69a4555 /
e5d3743 / e578aee / 5081648 (rig extension and its reviews).

**The three step-3 documents** (tracked, public, no private details):
- `md/acoustic_materials.md`, the wish-list: six classes (structural, hard furnishing, soft furnishing, complex
  assemblies, treatment, outdoor), two evidence tiers per row (A absorption, T transmission; 1, 1p pending conversion,
  2, 3), impact, status, references (research row ids `<source key>#<n>` at research commit d501809), gaps. Read sections 1.3–1.11 in full: the
  naming rule (1.6: `_<NN>mm` last marks a sheet, no suffix a traced solid), the row-design rules (1.7: lumped rows
  carry the visible thickness and rescale every power-law coefficient with `fRef`, X·k^B), `side` (1.8), the phase-3
  procedure, the engine-limit marks and the tolerance table (1.10), the start list (1.11). Section 4: open design
  questions; section 5: adding a material; section 6: the research requests.
- `md/acoustic_modeling_guidelines.md`: how a venue is modeled (solids and sheets, Blender names and `csv_strict`, the
  checklist of forbidden and limited configurations: three co-located faces are not resolvable, only a sheet on a
  solid or a sheet | sheet pair; at most two bonded rigid bodies; objects as hard boxes or porous bodies; assets).
- `md/acoustic_engine_stage2b_issues.md`: engine issues 1–8 with names, sizes and reproductions, and the engine-limit
  marks (`ENGINE LIMIT (<name>):` in the code, since 2026-10-27; earlier log entries say `ISSUE <n>`).
  Engine findings are recorded there, never fixed in step 3.

**The rig:** `tests/catch2_tests/acoustic_material_rig.hpp` (namespace `mt`: `check_material`, `Measured`,
`envelope`, `mpp`, `rescale`, the porous models Miki, Komatsu, Delany-Bazley and JCA, the object test); its self-checks
and the table case in `test_acoustic_materials.cpp`; the material cases in `test_acoustic_mat_heavy.cpp`, `_plates.cpp`,
`_porous.cpp` (groups `heavy`, `porous`, `plates`, `floors`, `leaves`, `resonant`, `objects`, `ground`); the rows in
`data/materials_acoustic.csv` (20 rows). The worked examples per group: concrete and masonry (`heavy`), glass and the
doors, the first lumped rows (`plates`), glass wool and rock wool (`porous`).

**Builds:** `build_linux` belongs to the repository's `make` and is not touched. The manager builds its own copy in
`/home/stephan/quadriga-lib-agents/build` (`cmake -S /sjc/quadriga-lib -B` that folder with the options of `make`'s
build: g++-12 / gcc-12, static library, tests on, AVX2, CUDA, MATLAB, Octave, Python off) and runs its `test_bin` from
the repository root; agents build inside their worktrees.

**Baselines** (the generic build, at cbe2ae3, batch 2a; regenerated 2026-10-28): `[wave]` 163474 / 178,
`[acoustic]` 44326 / 76 (the rig included), whole suite 1355360 / 1390 (the generic build leaves out one build-option
case of the merge notice's count). Files `test_wave.txt`, `test_acoustic.txt`, `test.txt` in `/home/stephan/quadriga-lib-agents/baselines/`.

**Research:** the requests of 2026-10-27 are answered (research commit d501809; report
`/sjc/accoustic_research/manager/step3_research_report_2026-10-27.md`). Research rows are cited by their stable id
`<source key>#<n>` at d501809 (Stephan, 2026-10-27), never by line; `data/id_map_2dad906.csv` maps the old lines.
The line citations are converted and the answer is folded into the wish-list with Stephan's twelve readings D1–D12
(063f05b; the readings in the section-9 log of 2026-10-28). Further requests go through Stephan in the handoff-7 form.

**Now: batch 2b** (built and merged 2026-10-28, waiting for Stephan's commit; then baselines, worktree cleanup, 2c): `masonry_light` (aircrete; D4 of the research fold
decided: tested against the range of the four 10 cm walls, density inside 480–800 kg/m³), `pu_foam_50mm` (porous),
the glazing units `glazing_double_24mm`, `glazing_triple_36mm` (lumped; the doors are the worked example). Then 2c
(`plaster_15mm` with the bonded-mass effect of issue 7, `carpet_tile_11mm`, `parquet` / `vinyl` on the slab), 2d
(`clt`, `screed`, ground classes and asphalt, `gypsum_board_12_5mm`, `plywood_19mm`, metal), batch 3 (leaves on
cavities, drywall, resonant absorbers, ceiling tiles, objects); rows waiting for data at the end.

**Mechanics per batch** (worked out over batches 0–2a; the briefs in the agents folder `materials/` show the form:
`batch2a_common.md` and one brief per family):
- The manager writes a common part and one brief per family (from `md/materials_agent_template.md`), creates one
  worktree per agent (`git worktree add /home/stephan/quadriga-lib-agents/worktrees/<id> -b agent/<id> HEAD`) and starts
  each agent with the Agent tool **without** `isolation`, told to work and build only in its worktree, at most 4 threads.
- Phase 1, plan (stop): the manager reruns the scripts each decision rests on and brings all plans together; Stephan
  decides; the manager continues each agent (SendMessage) with the decisions. Phase 2, build exactly the approved design.
- Merge: the manager copies the agents' test files and appends their CSV rows (diffs append-only, wish-list order),
  rebuilds its own build, runs `[materials]`, `[wave]`, `[acoustic]` and the whole suite one at a time, rowdiff against
  the baselines (old rows identical, only the new cases inserted), updates the shared documents, writes the batch
  report, the commit message and the `git add` line. After Stephan's commit: baselines regenerated, worktrees and
  `agent/*` branches removed after checking them against the commit.
- Agent scratch (scripts, probes, fit outputs) in `/home/stephan/quadriga-lib-agents/scratch/<id>/`, not in the
  worktree, so the manager can rerun it after the worktree is removed (from batch 2b; the batch-1 and 2a scripts were
  lost with their worktrees).
- Python in a checkout with `PYTHONDONTWRITEBYTECODE=1`; nothing temporary under `/sjc`.

## 1. Roles

- **Stephan** (the maintainer) decides every open point, reviews every change and commits. Wants plain physical
  explanations first (what the quantity is, what happens, how large in a typical room), options laid out, a
  recommendation, and a paste-ready message. Asked with the question tool, recommendation first, explanation written
  before the question (a question ends the turn in the desktop app).
- **The manager** (this session) writes briefs, spawns one worker agent at a time with the Agent tool, verifies every
  claim a decision rests on (rebuild, rerun, read the CSV and the test), keeps this file, prepares decisions and commit
  messages. Never commits, never edits `src/`, `include/`, `api_*` or `/sjc/accoustic_research`.
- **Worker agents**, spawned by the manager (phases 0–3a: strictly serially in this checkout; material batches:
  in parallel in worktrees, section 0):
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

## 2. Decisions taken at the kick-off (2026-10-22; scope, test layout and order superseded, see section 0)

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

**Per phase.** Brief (`/home/stephan/quadriga-lib-agents/materials/<id>_brief.md`, from `md/materials_agent_template.md`) → one agent → the
manager's checks (section 4) → report to Stephan (what checks out, what does not, decisions in the form below, a commit
message) → his review and commit → this file updated.

**Per row family (phase 3; the wish-list 1.10 procedure).**
1. The manager writes the brief: the family (generic row and its sheets) and its class, reference rows (research
   ids at d501809, coefficient types, bands, specimen sides), the rules that apply, the tier-1 model, open rules and
   engine issues it will meet, the tolerance lines of wish-list 1.10 that apply.
2. The agent: data → columns by the decided rule → tier 1 (the library through the rig against the independent model)
   → tier 2 (the columns against the converted measurement per octave) → judge (reasonable against the known range;
   an engine limit becomes an `ENGINE LIMIT (<name>):` check and a stage-2b entry; otherwise a regression guard) → one lean
   `TEST_CASE` per family → the CSV rows. Stops where an open rule is met, with the options measured.
3. The manager: rebuilds, reruns `[acoustic]` and `[wave]`, reads the CSV row and the test, recomputes the claims a
   decision rests on, runs the checklist.
4. Report to Stephan: the row, its tier-1 and tier-2 residuals per octave, recorded deviations, decisions (if any), the
   commit message and the `git add` line. Stephan commits; the baselines are regenerated after the commit.

**Decisions** go to Stephan as: what is decided; the options; trade-offs; what is correct (physics first, then the
library's rules); a recommendation; a paste-ready message.

**Coding rules from the SPEC files** (binding since the merge of 2026-10-27; every brief repeats them):
- Produced code, tests, wrappers and conf files never cite temporary `md/` documents or tracking ids; only published
  material (the material-model page `html_docu/quadriga_lib_material_model.md`, the Data Formats page, papers) and the
  persistent `md/SPEC_*.md`. Today the rig and tools break this (decision open, section 9 log of 2026-10-27).
- Tests use library calls for supporting tasks; the Catch2 binary links no BLAS, so no Armadillo matrix products
  above 4×4 in tests (small loops instead).
- Helper scoping (`md/SPEC_CPP_API.md`): a helper used once is a lambda (a self-contained algorithm of more than about
  20 lines may be a named file-local function directly above its caller); reused within one file `static inline` or
  an anonymous namespace; cross-file helpers only in a documented private header (the rig header).
- Working documents are deleted before a merge (phase 4 of step 3).

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
- **Engine-limit marks:** every check that fails because of an engine limit carries `ENGINE LIMIT (<name>):`, does
  not fail the suite, prints its values, and the issue file lists the name; no other non-failing checks.
- **References:** code, tests and tools cite only published material and `md/SPEC_*.md` (section 3).
- **Names and rows:** the naming rule of wish-list 1.6; lumped rows rescale every power law with `fRef`; the wish-list
  status updated from the result.

## 5. The material list

The list is `md/acoustic_materials.md` (the wish-list: six classes, two evidence tiers per row, references, gaps,
impact, status), with `md/acoustic_modeling_guidelines.md` beside it. The ten acceptance classes of the kick-off are
kept as test references inside it. Status changes are made there, not here.

## 6. Open rules

Each is decided at the stop of the first material that needs it; record the decision, the date and the material.

| rule | question | decided |
| --- | --- | --- |
| reference row per class | which measured row is the tier-2 reference where compilations spread by 2 to 4 times | heavy masonry: walls measured plastered on both faces first, bare plus increment otherwise (2026-10-28, research fold D8) |
| fitted-σ check | a converted absorption reference is marked when its fitted flow resistivity is off a measured one by more than about 1.5 times (notes 9.3) | applied as written: glass wool's finite-sample measurement marked (×1.83), its chamber conversion passes (×1.45) (2026-10-28, glass wool) |
| interlaboratory porous rows | the two Scrosati rows stay out until a polyester and a framed-sample model exist (notes 9.3) | kept: printed only, with the measured fitted-σ ratios (×1.71; weighted ×1.32 / ×1.39 extended, ×0.24 local), polyester ×5.12 (2026-10-28, rock wool) |
| resonant absorbers as one family | panels and membranes, perforated and slotted panels, MPP in the acceptance (notes 9.3) | |
| bending loss as a resistance | needed if `coiQ` comes from a measured loss factor (notes 9.3; page 11.1, 11.2, 11.4 re-measured if `coiQ` changes meaning) | |
| fabric relaxation on μ | curtains: random-incidence α reached, single angles off by up to 0.35 (notes 9.3) | |
| Sabine to random for panels and MPP | a conversion as for porous layers, or classes 8 and 10 marked under the tiered rule | |
| the leaf's loss ratio `rr` | which normalization transfers to room-size leaves (R/Z₀ or R/(ω₀m″)) | |
| damping branch of a peaked face | over-damped default against measured empty-cavity R′ at or below 1 | |
| `alpha` on a porous layer | its role, if any, next to `a`–`h` | none: 0 on porous rows (2026-10-28, glass wool) |
| membrane constant with a bonded fill | 60 or 50 when the fill is bonded to the plate (handoff 5) | |
| chamber to random for hanging fabrics, green walls, seat and audience blocks | the "1p" rows of the wish-list (1.3) | |
| the upholstery row | fit of the porous body to per-object areas, with the object test | |
| generic solid across thickness | the miss of one fit over the measured range; a thickness-scaled coincidence to stage 2b if too large (wish-list 4) | concrete 150–203 mm: one fit, `coiF` a fit knob, 0.25 dB off the envelope (2026-10-27, concrete); heavy masonry 150–250 mm, one fit, worst wall 2.23 dB, ends printed (2026-10-28, masonry_heavy) |
| reference row per class | now: tolerance table of wish-list 1.10 (class envelope) | decided 2026-10-26 (r01 D4, r02 F5) |
| dense ground classes | the passive Miki fit degenerates at 2 and 20 MPa·s/m² (μ′ near 0): rigid rows with a face absorption, as `asphalt`; the fit target (n and z, or the asserted reflection) | |

## 7. Research requests

Format (handoff section 7): the column or class; the quantity and its coefficient type; the bands; whether closed
sources are acceptable; urgency and whether step 3 can proceed without it. Stephan forwards to the research manager.

| sent | request | status |
| --- | --- | --- |
| 2026-10-27 | `md/materials/research_request_2026-10-27.md`: request 0 (stable row ids), 1–13 by impact (wish-list section 6), three data errors (lines 1451, 520, 1819 of 2dad906); routes proposed by the research manager, approved by Stephan | answered 2026-10-27 (research commit d501809; report `/sjc/accoustic_research/manager/step3_research_report_2026-10-27.md`): stable row ids `<source key>#<n>` with a map of the 2dad906 lines; 17 rows changed (2 of them cited: Fuchs 327, 328, combination references of 5.11); gains for doors, masonry, beds, sofa, cabinets, plants, ceiling tiles, perforated gypsum, carpet at 63 Hz, asphalt, rug σ; no open data for requests 6, 8, 10 |

## 8. Engine items raised

A material that cannot be expressed: what, the size of the miss, the column or physics missing. None leads to work
without Stephan's OK. Each item becomes an issue in `md/acoustic_engine_stage2b_issues.md` (issues 1–5 there: porous
layer on a wall, band decision near grazing, two traced wave media back to back, three or more bonded rigid bodies, a
rug on a leaf floor). The table below is the first entry only; the issue file is the record.

| raised | material | miss and size | status |
| --- | --- | --- | --- |
| 2026-10-23 | glass wool 50 mm (page 11.3 columns) as a sheet on cast concrete 152 mm (page 11.4 row) | transmission at normal incidence: the library adds 8.04 / 6.71 / 5.38 / 4.09 / 3.42 / 4.78 / 7.32 / 9.32 / 13.02 dB to the wall (31.5 Hz–8 kHz); an independent transfer matrix (the same fluid layer on a heavy limp plate, scratchpad `tmm/wool_on_plate.py`) adds −0.02 / −0.07 / −0.18 / −0.28 / 0.16 / 2.53 / 5.98 / 8.79 / 13.24 dB. Over-insulation up to 8 dB at 31.5 Hz, 5.6 dB at 125 Hz, 2.3 dB at 1 kHz. Reflection is fine (6.3e-3). Random incidence not yet measured. | issue 1 in `md/acoustic_engine_stage2b_issues.md` |

## 9. Log (dated, newest last; the state is section 0)

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
- Phase-0 files committed by Stephan (5be85fb, 2026-10-23).
- **Phase 1 in work:** brief `md/materials/rig_brief.md` (two phases: plan, then rig and self-checks).
  - Plan phase done 2026-10-23 (`md/materials/rig_report.md`). The manager reran the probe in
    `build_linux/scratch/rig`: the band verdict (R 1.0726 at 89°, 1816 Hz, threshold 0.15; equal to the single
    frequency at 0), glass 4 mm `tl` on G1 against 3072 × 12 (≤ 0.048 dB to 4 kHz, 0.278 dB at 8 kHz), a type-1 case
    0.255 s CPU, octave α against the transfer matrix (9.2e-9 wool, 4.3e-7 board, 3.6e-5 filled box at −120 dB), the
    wall's own `tf` (2.8e-3 against the pad referred to air). All reproduce.
  - Stephan's answer to the first plan (2026-10-23) redirects the rig toward his earlier Python rig (a notebook kept
    outside the repository: a point source at (−5, 0, 0.001), the material at x = 0, 50 receivers at y = 0 … 49 m on
    x = −5 for reflection and x = 5 for transmission, third octaves with 125 Hz first). His decisions:
    - **D1:** transmission through `calc_diffraction_gain` with a single path; the material is one layer or a stack
      of two back to back (sheet | solid, sheet | sheet; check whether solid | solid sums its series or needs a
      relaunch). Reflection through a short sequence of `ray_progress` calls (front reflection; then enter, reflect at
      the back, exit), `refraction_mode` false, so the landing points are known and a minimal set of rays is launched.
      A minimal version of the scene tracer is a rig helper; 50 materials should add well under 5 s of wall time.
    - **D2:** 50 receiver angles, reported at 0, 5, 10, 30, 45, 60, 70, 80°; third-octave bands with 125 Hz as the
      reference (first) frequency; sample at the frequencies of the research references rather than interpolate.
    - **D3:** (c), with the aim that the rig gives meaningful simulator output for real scenes, not an idealized model.
    - **D4:** combinations are a known failure mode (materials fine alone, wrong when combined: absorbers, bass traps,
      carpets on or against rigid bodies). Order: wood alone, carpet alone (as a sheet?), carpet on wood measured with
      diagnostic output and judged against references or for plausibility, then made a regression fixture; common
      combinations before the scene models of the next step.
    - **D5:** option A: the test case is lean; it defines the material, runs the rig's transmission and reflection
      functions, and compares with the known constructions written in the test with a source comment.
    - **D6:** redo the analysis with this input and propose the helpers; concrete is the first material, to see the rig
      run with its diagnostic output.
    - **D7:** (A), `band_probe` with `attB` and `alphaB` plus `tools/acoustic/grid_probe.py`.
  - Revised plan done 2026-10-23 (`md/materials/rig_report.md`, "Revised plan" R0–R14; probe
    `build_linux/scratch/rig/rig2_probe`). Manager reruns, all reproduce: transmission sheet | solid 0.063 dB,
    sheet | sheet 9e-6 dB, wave solid | solid 4.91 dB (and T > 1); reflection board | box | wall 1.22e-4 in 4 calls,
    filled box 1.57e-3 in 43 calls; threshold 0.15 with 125 Hz first 0.164 in R (board on box), 1.01 dB in T (wool
    solid); old receivers 0°–78.7° read glass 4 mm `tl` +3.36 dB at 2 kHz, +8.74 dB at 8 kHz; 50 midpoints × 3 per
    third ≤ 0.11 dB to 4 kHz. Cost correction: the probe's 0.65 s for 50 constructions used 50 transmission angles over
    27 thirds; at 192 angles an ordinary construction takes 13–52 ms wall at 12 threads, the filled box 0.29 s.
  - Stephan 2026-10-23: RD1 (a) and RD3 (a) accepted. Questions on RD2, RD4–RD7 answered with measurements:
    - grids (first probe, `conv`, against 3072 × 12; glass 4 and 6 mm, gypsum, double glazing, concrete): 50 angles ×
      1 per third ≤ 0.34 dB to 2 kHz, 0.49 dB at 4 kHz, 6.7 dB at 8 kHz; 50 × 3 ≤ 0.07 / 0.73 / 2.75; 192 × 3
      ≤ 0.05 / 0.32. `calc_diffraction_gain` takes every angle in one call per frequency.
    - the library's sound speed is 342.76271 m/s (`src/qd_atmospheric_absorption.hpp`, ISO 9613-1 at 20 °C; the
      wrappers and `qrt_ray_trace` convert with it); only the page, `txlib.py` and `lw::GHZ` use 343.
    - wool 50 mm on concrete 152 mm (engine item below).
  - **Final rig decisions (Stephan, 2026-10-23):**
    - RD1 (a): paths summed per angle inside the rig (plane wave); transmission through `calc_diffraction_gain`
      (lod 0, scalar, threshold 0), reflection through an aimed ray set and a short `ray_progress` sequence,
      `refraction_mode` false.
    - RD2 / RD4: default grid 50 midpoint angles over 0°–90° (plus the report angles 0, 5, 10, 30, 45, 60, 70, 80° at
      weight 0) × 1 tone per third, 31.5 Hz–8 kHz, 125 Hz first in a band; the grid is an argument of the rig
      functions, a case evaluates the bands its references cover and may ask for more samples.
    - RD3 (a): the builder throws on a forbidden stack (room-side layer a sheet, the rest traced; no wave
      solid | solid, no mass-law sheet | sheet, no thin lossless wave layer as a solid).
    - RD5: the CSV holds the parameters; a material case reads its row by name and holds the construction, the
      expected values (literature octaves with a source comment) and the tolerances; the independent physics is
      computed in the test. Rig self-check constructions are defined inline (D5 A).
    - RD6: the library's sound speed, 342.76271 m/s (`qd_acoustic_freq_scale`); the 343 m/s difference (≤ 0.017 dB,
      ≤ 4e-4) goes into the tolerance; the page's 343 is corrected in phase 4.
    - RD7: combinations of a layer on a rigid body check the added insulation (with minus without the layer) against
      a transfer matrix with a plate written in the test; misses are recorded in
      `md/acoustic_engine_stage2b_issues.md` (issue 1: porous layer on a wall).
    - D7 (A): `band_probe` gains `attB`, `alphaB`; `tools/acoustic/grid_probe.py` validates the grid once.
    - Concrete is the first material, to see the rig run with its diagnostic output.
  - **Phase 2 of the rig done 2026-10-23** (`md/materials/rig_report.md` "Phase 2"): the rig, five self-check cases,
    the table case and `Material - concrete_cast_152mm`; `data/materials_acoustic.csv` (concrete); `band_probe` with 26
    columns at 342.76271 m/s; `tools/acoustic/grid_probe.py`. Manager check (rebuild, reruns at 12 threads, load
    1.9–2.0): `[wave]` 141128 and `[acoustic]~[materials]` 112451 numbers identical; whole suite 1347008 / 1309
    (+151 / +7), 256775 numbers identical plus one inserted block; new cases 0.000–0.040 s; `check_docs.py` passes;
    `grid_probe.py` on concrete 50 × 1 within 0.118 dB of 3072 × 12 and equal to the case's octaves. Manager fix:
    `band_probe_example.in` had a stray uncommented line ("bad line") and the superseded expected values of
    2026-10-15; removed, the probe reproduces 0.924972 / 0.648002 / 0.444103 / 0.779643.
  - Review points for Stephan: the material case is about 60 lines (tier-1 loop and printed mass law inline; move to
    helpers); concrete's tier-2 values are the page's reading of the Litvin plot, the research data reads
    42.1 / 46.5 / 52.6 / 56.5 / 61.7 / 64.6 dB. Decisions PD1–PD3 open.
  - Stephan read the rig (2026-10-23): sheet = traced solid of the same thickness is a standard assertion for every
    isolated material (the rig builds both already; concrete asserts it); the rig (1028 lines) must be compressed
    (it would pass 5000 lines) and made faster. Brief "Revision 3", phase 2b, sent to the rig agent. PD1–PD3 and the
    research Litvin values still open.
  - **Phase 2b done 2026-10-23** (`md/materials/rig_report.md` "Phase 2b"): test file 1028 → 677 lines (rig 481,
    self-checks 89, concrete 17); `mt::check_material` runs a row as a traced solid on the full grid and as a sheet on
    the report angles and asserts the twin, tier 1, R + T ≤ 1, diagnostics, tier 2; geometry from
    `quadriga_lib::plane` / `cube` (a solid is a closed box), path length from `path::length()`, R / T from `nREF`.
    Manager check (load 2.3–3.2): `[wave]` 141128 and `[acoustic]~[materials]` 112451 numbers identical; whole suite
    1347054 / 1308; `[materials]` 197 assertions in 6 cases, concrete 0.006 s at 12 threads and 0.011 s on one
    thread, all six 0.137 s / 0.459 s; `check_docs.py` passes. Side finding: the API page lists `length` of `path` as
    an attribute; it is the member function `length()` (`include/quadriga_channel.hpp:158`), a documentation point
    for the library, not for this step.
  - **Phase 1 closed:** the rig committed by Stephan (13e4e45, 2026-10-23). Baselines regenerated at 13e4e45:
    `[wave]` 163474 / 178, `[acoustic]` 42938 / 65 (the rig's 6 cases included), whole suite 1347054 / 1308.
    Decided by Stephan 2026-10-23: PD1 (a) the plate of the added-insulation model gets the library's finite-panel
    radiation (Davy's σ at the wall's `side`) for the random-incidence row, printed, not asserted (implemented with the
    first combination case); PD2 (a) measured references stay assertions, a known deviation is recorded in its case
    with a wider, commented tolerance; PD3 the box-before-fill exception of the stack rule stays; concrete moves to the
    research data's Litvin values when it is redone in phase 3.
  - **Phase 2 started 2026-10-23:** analysis `a01_material_list` (brief `md/materials/a01_material_list_brief.md`):
    the candidate rows per class from the research library, references, types, gaps, rules met, order. The listening
    room's inventory comes from Stephan in parallel.
  - a01 done 2026-10-23 (`md/materials/a01_material_list_report.md`): 47 candidate rows (30 with a usable reference,
    25 of them measured; 8 compilation only; 3 per-object; 4 waiting for research; 1 out by an open rule; 1
    definition). Wood alone = `plywood_19mm` (Wareing R, line 1334; Quirt 1983 α, Sabine, line 1344, max 0.116);
    no carpet on timber is measured anywhere (every carpet row is on concrete, underlay or unstated). Proposed first
    ten: plywood_19mm, carpet_tile_11mm, carpet on plywood, glass_wool_50mm, pu_foam_50mm, leaf_mdf_6mm (Shearer),
    leaf_plywood_3mm (Sakagami), curtain_flat_pes_300g, mpp_steel_0_5mm, cpa_steel_1mm_on_foam_100mm. Six draft
    research requests (doors, perforated and slotted panels, sofas / armchairs / beds blocking; carpet on timber,
    floors on joists, brick plastered both faces not blocking). Data errors for the research manager: line 1451
    (Fuchs 2005) typed normal incidence; lines 520 and 1819 (Cox) octave columns hold the centre third. Rig gaps:
    no Miki / Komatsu / JCA printers yet (added with the first porous material), no 0° tier-2 comparison (needed for
    the Broda carpet rows). Manager checked lines 1334, 1344, 1451, 520, 1819 and every carpet row's backing.
  - **Phase 2 redefined by Stephan (2026-10-26).** The list is built from the modeling workflow (a venue, a Blender
    model, materials chosen by guess, simulation for speaker placement and treatment), not from the literature:
    - five material classes: structural building materials, hard furnishing, soft furnishing, complex assemblies,
      acoustic treatment; three evidence tiers per material: 1 measured references, 2 first-principles physics,
      3 informed guess; phase 3 works tier 1, then 2, then 3, by impact within a tier;
    - generic rows: a traced solid (`concrete`) whose thickness comes from the geometry, plus sheet rows named by
      thickness (`concrete_150mm`); some materials only as sheets (window glass); literature constructions are
      test references, not rows; how a generic row holds from 100 to 600 mm (today's fitted knobs do not) is an
      open design question for phase 3;
    - panel size (`side`) adjusted per use; no combination rows: walls and floors first, carpets and absorbers
      placed separately; combinations are rig tests only where measurements exist;
    - deliverables of phase 2: `md/acoustic_materials.md` (the wish-list: name, intended use, recipe, evidence tier,
      references, gaps, impact, status, starting columns from the tests or the page) and a separate
      `md/acoustic_modeling_guidelines.md`; a procedure for adding materials later; no CSV, no code in phase 2;
      then a research round on the gaps; phase 3 in a following session.
    - Engine findings are recorded, not fixed, until every material is in.
    - `obj_file_read` names (reviewed 2026-10-26, `src/obj_file_read.cpp` `base_material_name`, `resolve_csv`): exact
      name, then the part before the first dot; no fallback across suffixes; case-sensitive; a dot inside a row name
      breaks Blender duplicates; unmatched names become air unless `csv_strict`.
  - a02 done 2026-10-26: draft 0 of `md/acoustic_materials.md` (64 entries: 21 structural, 8 hard furnishing, 9
    soft, 10 complex assemblies, 16 treatment; 31 tier 1, 20 tier 2, 12 tier 3, plants split; 22 proposed additions;
    13 measured combinations as test references; 9 open design questions; 12 research requests) and
    `md/acoustic_modeling_guidelines.md` (14 engine limits). Manager spot-check of research lines 80, 82, 250, 374,
    1202, 1937: as cited. Stephan's `.gitignore` now ignores `md/materials/`. First-round questions: two rows per
    board (mass-law and leaf) or one plus a recipe; how a generic solid carries thickness; the hull recipe and whether
    a hull transmits; how `side` is changed; which additions stay (`water_20c`?).
  - Round 1 on draft 0 (Stephan, 2026-10-26): Q1 two rows per board, the leaf named `<material>_leaf_<NN>mm`
    (`gypsum_leaf_12_5mm`; renamed in both drafts); Q2 one generic fit across thicknesses plus sheet rows, the sheet
    rows and references at round thicknesses (150 and 200 mm, not 152 and 203), the miss measured, a thickness-scaled
    coincidence recorded for stage 2b if needed; Q4 `side` a row value at a typical size, no engine change (the
    engine sees one triangle, not the panel), drift accepted where the drawn panel differs; Q5 all proposed additions
    stay. Q3 (2026-10-26): hard closed furniture (bookshelf, wardrobe, cupboard) as a hard box whose faces absorb the
    measured per-object area divided by the box's surface; soft furniture (sofa, armchair, bed) as a porous body, a
    low-poly shape or a cushion-like box of an upholstery row fitted to the per-object area, which reflects at its
    face, absorbs inside and transmits some (an air-like absorbing volume rejected: it reflects nothing); objects as
    Blender assets with the material assigned. The stage entry goes (it will be modeled in detail). Still open: lumped
    rows against recipes for building assemblies; `water_20c` test-only or out.
  - Draft 1 written 2026-10-26 (`md/acoustic_materials.md` 436 lines, `md/acoustic_modeling_guidelines.md` 171):
    round-1 answers in; objects in section 2.4 (hard box, porous body, assets, object test still missing in the rig);
    proposed object rows `box_wardrobe`, `box_cupboard`, `box_bookshelf`, `box_fridge`; mattress and pads merged into
    `upholstery`; plants, clothes racks, a single person as porous bodies and the audience as a porous box (proposed);
    typical `side` per entry (proposed); stage removed. Open for round 2: lumped rows against recipes for building
    assemblies, `water_20c`, `concrete_block_190mm` (unit size, not round), the proposed object kinds, typical sides
    and impact ratings.
  - Round 2 (Stephan, 2026-10-26): the upholstery fit is phase-3 work; `water_20c` stays; an outdoor class is added
    (vegetation, ice, ground, asphalt; the research data hold flow resistivities of grass, soil, forest floor, gravel,
    snow and asphalt, asphalt α, ground factors, foliage excess attenuation per meter); drywall rows must start from
    what a modeler sees (a plan with a wall thickness, perhaps a note on the fill), with a reasonable assumption
    inside; `concrete_block_190mm` is not usable by a modeler; a clutter category (hard and soft, by density) for a box
    drawn over a desk or shelf. Accepted (2026-10-26): outdoor class 6; drywall rows from the visible thickness and
    fill with a standard build assumed; masonry classes (`masonry_heavy`, `masonry_hollow`, `masonry_light`) instead of
    `concrete_block_190mm`; clutter as hard / soft × sparse / medium / dense, tier 3.
  - **Naming rule (Stephan, 2026-10-26), for every material:** the thickness suffix `_<NN>mm` is the sheet marker and
    always comes last; without it a row is a solid. `drywall`, `drywall_filled` (solids), `drywall_filled_100mm`
    (sheet); likewise `gypsum_leaf_12_5mm`. Draft 2 requested from the drafting agent.
  - Draft 2 written 2026-10-26 (`md/acoustic_materials.md` 515 lines, guidelines 197): naming rule applied
    throughout; class 6 outdoor (ground rows from CNOSSOS classes, `asphalt` tier 2 since the research data hold no
    absorption of asphalt paving (the three "asphalt" α rows are parquet in asphalt and a linoleum compilation, checked
    by the manager), `asphalt_porous_40mm`, `ice` without data, `vegetation` as an air-matched lossy volume); drywall
    rows (no 100 mm wall measured); masonry classes; clutter grid. Review points left for phase 3: glazing sheet names
    give the unit thickness (`glazing_double_24mm`) while the lumped row's `thk` is the glass alone; the CSV row
    `concrete_cast_152mm` breaks the naming rule until redone as `concrete_150mm`.
  - Phase 2 documents committed by Stephan (19323d9, 2026-10-26). Before phase 3, a critical review at high effort
    (brief `md/materials/r01_phase2_review_brief.md`, output `md/materials/r01_phase2_review.md`); the manager passes
    its decisions to Stephan, then an agent updates the documents.
  - r01 done 2026-10-26 (`md/materials/r01_phase2_review.md`): 13 findings, 8 for Stephan (1 two tiers per row,
    transmission and absorption; 2 masonry rows as the finished plastered wall, recipes that hit engine limits; 3 rig
    extensions and one case per row family; 4 class tolerance as the envelope of its walls ±1 dB, assert only within
    the measured thickness range; 6 clutter as air-matched absorbing volumes; 8 ground rows with Delany-Bazley at the
    class σ, gravel split; 9 glazing and lumped rows with `thk` = visible thickness, curtains flat / pleated; 12 the
    listening room's rows first). Manager checks: data line 1845 is the lined brick wall (TL 69-287, + gypsum board on
    furring; 1845 − 1847 = −0.3 / −1.4 / 8.4 / 9.9 / 8.6 / 5.4 dB); line 870 is Attenborough 2016; 1729 − 1847 =
    5.3 / 7.4 / 8.4 / 9.2 / 6.2 / 5.5 dB; 1050 − 1049 = −0.6 / 7.6 / 9.9 / 11.0 / 6.2 / −1.9 / 0.6 dB;
    `Material::same_as` compares `alpha` (`src/ray_mesh_interact.cpp:635`), so an air-matched lossy volume is not air.
    The reviewer's "of little value" list includes `ice` and `box_fridge`, which Stephan asked for: kept unless he says
    otherwise.
  - **Decisions on r01 (Stephan, 2026-10-26):**
    - D1 (a): two tiers per row, transmission and absorption.
    - D2 (i): masonry rows stand for the finished wall, plastered both faces (a generic solid of the finished wall plus
      sheets; `plaster_15mm` for exceptions). The engine's one-level state (three bonded bodies within ±3 dB) becomes a
      stage-2b issue; phase 3 extends the rig to plaster sheets on both faces of a block solid and measures the miss
      against the bonded plate, as the fixture for the engine rework.
    - **ISSUE convention:** every rig check that fails because of an engine limit is marked: a comment `ISSUE <n>:`
      naming the stage-2b issue, the assertion switched to a non-failing check (Catch2 `CHECK_NOFAIL` or equivalent)
      that still prints and is countable, the measured and reference values in the diagnostics, and an entry in
      `md/acoustic_engine_stage2b_issues.md`. No failing suite from known engine issues.
    - D3 (a): one test case per row family, kept lean and clear.
    - D4 (a), as a procedure: measure first; judge against the known range of measurements and models; if it needs an
      engine fix, mark ISSUE; if not, assert tightly (about 1 dB against the measured library result) as a regression
      guard; a comment in the case states what "reasonable" meant (the range used). The rig documents itself; the
      wish-list is updated from the result.
    - D6 (a): clutter as air-matched absorbing volumes, three rows by density, revisable when data arrive.
    - D8: Delany-Bazley at the class σ with a passivity check and Miki printed; Miki refitted to the class impedance
      where Delany-Bazley fails. Gravel split.
    - D9 (a): lumped rows carry the visible thickness in `thk`, `fRef` scaled. Manager check: a mass-law sheet's `thk`
      does not enter the phase (`src/ray_mesh_interact.cpp:1320–1324`, only non-mass-law layers displace the host),
      so (a) is neutral for rigid lumped rows and makes sheet and solid of the visible thickness consistent; wave-medium
      sheets keep their true thickness, which is displaced from the host. Curtains `curtain_flat_*` / `curtain_pleated_*`.
    - D12 (a): tier, then impact; every relevant material in, each with diagnostics, before the engine is reopened.
    - Furniture boards merged into one `furniture_board` family; `ice` dropped; `box_fridge` replaced by `box_metal`
      (a hard box covered in metal: fridges, freezers, bar counters, metal cabinets), a glass front as a glass sheet on
      that face; one material per closed body, finishes as sheets (to be confirmed in phase 3).
  - Draft 3 written 2026-10-26 by the drafting agent (wish-list 587 lines, guidelines 234, stage-2b file with issue 4
    and the ISSUE convention); manager check: no `ice` or `box_fridge` left, no private details, `check_docs.py` passes;
    section 5 of this file now points to the wish-list. Open: asphalt absorption and hollow-core doors have no data
    (research requests); several object row names marked proposed.
  - Draft 3 committed by Stephan (be65346, 2026-10-26). Second independent review r02 started (brief
    `md/materials/r02_phase2_review_brief.md`: draft 3 fresh, r01 decisions checked for consistent application, no
    reopening of decisions without a concrete technical reason).
  - r02 done 2026-10-26 (`md/materials/r02_phase2_review.md`): 16 findings, 9 for Stephan (1 rugs and objects on floor
    finishes: three co-located faces; 3 ground rows by a passive Miki refit as the standard route; 4 stable ids for
    research rows; 5 masonry references and the envelope across thickness; 6 bonded floors as three bodies; 7 how the
    tolerance regimes combine; 8 which tier orders the work; 15 wood wool, blinds, clouds; 16 the roadmap against
    D12). Manager checks: `src/ray_mesh_interact.cpp:2617` "At most two co-located faces form a stack; a third layer
    is a separate event"; the reviewer's `d9scale.py` reproduces (double glazing: `tf` at 125 Hz 0.1124 → 0.1670,
    `att` at 8 kHz 16.289 → 9.045 dB when only `fRef` is scaled), so the manager's D9 statement of 2026-10-26 was
    wrong: every power-law coefficient must be rescaled with `fRef` (X·k^B); `db_passive.py` reproduces Delany-Bazley
    needing a gain at 6 to 25 of 25 thirds at the class σ.
  - **Decisions on r02 (Stephan, 2026-10-26):**
    - F1 (b): hard floor finishes (parquet, vinyl, tile) are thin traced solids on the slab; a rug is a sheet on that
      solid; objects stand on the finish solid. Three co-located faces are not resolvable, so only a sheet on a solid
      or a sheet | sheet pair is allowed; into the modeling guidelines. Leaf floors (floating laminate, boards on
      joists) under a rug stay misses (stage-2b).
    - F3: ground rows by a Miki fit with passive columns to the class impedance, as the standard route.
    - F4 (b): line numbers stay, pinned to the research commit 2dad906, remapped by script after a regeneration; a
      stable id column goes to the research manager as an ISSUE with the hand-over package.
    - F5 (a): masonry references = bare class walls plus the measured plaster increment of matched pairs (with its
      spread); the envelope formed after a mass-law shift to the evaluated thickness; asserted where the trend miss
      stays inside the tolerance, printed outside.
    - F6 (a): slab and screed drawn as one solid of their full thickness, the finish on top (two bodies).
    - F7: the four-line tolerance table confirmed (may move with more data).
    - F8: order by the absorption tier, then impact, then the transmission tier (Stephan: the order matters less than
      a complete table).
    - F15: wood-wool panels added; blinds and ceiling clouds as recipes of existing rows.
    - F16 (a): D12 governs the roadmap: concrete first as the rig check, then by the absorption tier and impact.
    - D9 correction (no decision): every power-law coefficient is rescaled with `fRef` (X·k^B).
  - Draft 4 written 2026-10-26 (wish-list 790 lines, guidelines 268, stage-2b 147 with issue 5: a rug on a leaf
    floor); manager check: all r02 decisions present, the guidelines' worked room is a generic example (5 × 4 × 2.6 m,
    proposed), no private details, `check_docs.py` passes. Side finding of the drafter: concrete 152 → 203 mm is
    2.67 dB of mass law from the surface masses (draft 3 said 2.51 from the thickness ratio). Open: whether the
    listening room's inventory stays a phase-2 deliverable (proposal: it moves to step 4, the reference simulation).
  - Draft 4 committed by Stephan (bd0a170, 2026-10-26). Room inventory moved to step 4 (Stephan, 2026-10-27: the list
    is enough to set up a simulation and iterate). **Phase 2 closed.**
  - Research hand-over drafted 2026-10-27: `md/materials/research_request_2026-10-27.md` (request 0 stable row ids;
    requests 1–13 by impact in the handoff-7 format; three data errors at lines 1451, 520, 1819 of 2dad906; handoff
    gaps not requested, with reasons). The closed-source column is a proposal for Stephan to confirm.
  - **Phase 2 result:** the wish-list and the guidelines (draft 3 committed be65346; draft 4 after the second review). Next: the research requests (section 6 of
    the wish-list, plus the data errors of a01) go to the research manager; phase 3 starts in a new session.
  - **Engine issues for stage 2b** (Stephan, 2026-10-23): every engine miss found in step 3 becomes an issue in
    `md/acoustic_engine_stage2b_issues.md` (tracked, public) with the diagnostics a later engine stage needs, the
    affected materials and an interim modeling guideline; modeling guidelines (the box of cavity air, face order,
    threshold 0) are collected there too.
  - Earlier wording of the open RD questions: (in-rig plane-wave sum; 50 + 8 angles for reflection, 192 for transmission; the
    stack rule; 16 Hz–16 kHz at 3 per third, 125 Hz first; definitions in the tests as source with a CSV equality case
    and a writer switch; 343 m/s; transmission of a wave layer on a mass-law wall printed only).
  - Earlier recommendations of the first plan, superseded by the above: (threshold 0; G1 192 × 3 for asserted `tl`, G0 24 × 2 otherwise; tier 1
    of mass-law rows = column formulas asserted plus a physical plate printed as spread; the wall keeps its row except
    `tf` 0 behind a fill, `tl` only where the wall's `tf` > 0; rig rows in `tests/data/materials_acoustic_rig.csv`;
    `lw::walk` not copied; `band_probe` gains `attB` and `alphaB` with a driver `tools/acoustic/grid_probe.py`).
- **2026-10-27, phase 3 started** (new manager session). Build current at c9f5c00 (no source change since 13e4e45);
  baselines rechecked with `rowdiff.py`: `[acoustic]` 42938 / 65 and `[wave]` 163474 / 178, 0 differences; not
  regenerated. Decided by Stephan for the rig extension: the stack rule refuses three co-located faces, with one
  opt-out argument used only by `ISSUE` fixtures (issue 5); a row belongs to the family of its name without the
  `_<NN>mm` suffix, one case `Material - <generic name>` per family (the rule for two generic rows proposed by the
  plan); Miki, Komatsu, JCA and the Davy σ plate written in the test, fits (passive Miki for ground) in
  `tools/acoustic`; three phases: plan, items 1–9, the object test. Brief `md/materials/rig_ext_brief.md` written
  (line budget about 250 + 100). Brief approved by Stephan; rig agent started on phase 1 (plan) 2026-10-27.
  - Rig-extension plan done 2026-10-27 (`md/materials/rig_ext_report.md`; probes `build_linux/scratch/rig_ext/`):
    items 1–9 about 146 lines, object test about 95; decisions D1–D9. Manager rerun of `[p1]`–`[p9]`, `[p4d]`: every
    number reproduces (porous bounds fail as stated: 7.4e-4 > 5e-4 dB; filled box 9.466 dB on one path, 0.290 dB on
    the paths; plaster sheet | block solid | plaster exit sheet 4.8e-5 dB, the far plaster traced −2.98 dB random,
    6.68 dB per node; rug | parquet sheet | concrete 0.979 in R; surface mass 346.74 against 88.52 kg/m² from `fRef`;
    `CHECK_NOFAIL` not counted by Catch2 3.8.1). "Exact" for issue 4 holds at one `side` per run: plaster (3.05 m) |
    concrete (3.52 m) | plaster 0.80 dB per node, glass | concrete | glass 6.7 dB. Grazing relaunch in the object
    probe (`P10DIR=36`): power 0.100 → 0.144 in six calls (the agent's 0.19 not reproduced; growth confirmed).
    Decisions to Stephan.
  - Stephan, 2026-10-27, on the plan: three co-located faces are a known limit (the engine takes two of the three
    faces, which two is random) and will likely stay. ISSUE marks are comments only (`// ISSUE <n>:`), with the values
    in the diagnostic output; no helper (replaces the plan's D5). Manager check for D1: the scene tracer's LOS modes
    2–6 drop the traced transmission-only paths and take the direct path from the diffraction gain
    (`/sjc/qrt/src_lib/beam_trace.cpp`, `ignore_direct_path`); mode 9 traces it. D1–D9 re-presented in full.
  - **Decisions on the rig-extension plan (Stephan, 2026-10-27):** production runs are capped at about 20–30 bounces
    (cavity traps explode the path count; about 150 million traced rays is the limit), so path sums are analyzed with
    at most 10 interactions. D1 (a): T of a box in front of a fill from the traced paths (cap 10), the one-path value
    printed under ISSUE 3. D2 (a), as an engine issue: curtain | box | glass T is the issue-1 mechanism; diagnostics,
    a reasonable estimate (added-insulation form against a finite limp plate) and an ISSUE 1 remark. D3 (a): issue-4
    fixture in both forms. D4 (a): one porous bound set. D5 (a): three co-located faces always refused, no opt-out, no
    issue-5 fixture. D6 (b): cases may name several families joined by " + ". D7 (a): surface mass `e`·ρ₀·`thk` with
    an override for lumped rows. D8 (c): models in the test, the ground fit deferred to the ground family (dense
    classes as rigid rows: open rule). D9 (a): object test at the cap of 10; the porous body redesigned to the budget
    (4.4 s far too much). Brief revision 1 written; the rig agent continued with phase 2 (items 1–9).
    Manager follow-ups with the phase-2 package: wish-list 1.10 and the issue file's ISSUE wording (comments plus
    diagnostics, no `CHECK_NOFAIL`); wish-list 1.7 rule `e` = ρ/ρ₀ for mass-law rows; issue 1 (curtain at a window),
    issue 3 (one path, cap 10), issue 4 (scope narrowed), issue 5 (size, a modeling limit), the grazing relaunch;
    open rule "dense ground classes as rigid rows".
  - Rig extension phase 2 (items 1–9) done 2026-10-27 by the rig agent; manager checks: rebuild clean; `[wave]`
    141128 numbers identical; `[acoustic]` 43143 / 69, of the old rows only the filled box on the closed wall moved
    (T now from the paths: tl from 137–277 dB of leakage to inf, 25 numbers) and three tier-2 headers and one label
    changed; whole suite 1347259 / 1312 (+205, +4), the same 25 numbers; `[materials]` 0.38 s at 12 threads, load 2.8;
    `check_docs.py` passes. ISSUE rows as printed: issue 3 filled box 5.04 dB at 10 interactions (9.47 on one path,
    0.29 at 60); issue 4 traced far plaster −2.98 dB random, 6.56 dB per node; issue 1 curtain | box | pane within
    0.25 dB at 0°, at random +1.42 dB (2 kHz) and −1.04 dB (8 kHz) outside the band of the two estimates. Rig grew by
    231 lines against about 146 planned (self-check cases and comments). Documents updated by the manager: wish-list
    1.7 rule 10 (`e` = ρ/ρ₀), 1.10 ISSUE wording and prerequisites, section 3 "rig today"; guidelines 3.1 and 12;
    stage-2b ISSUE convention and bounce cap, issues 1, 3, 4, 5 measured, issue 6 (grazing relaunch) new; open rule
    "dense ground classes". To Stephan for review and commit.
  - Committed by Stephan (69a4555, 2026-10-27). Baselines regenerated at 69a4555: `[wave]` 163474 / 178,
    `[acoustic]` 43143 / 69, whole suite 1347259 / 1312. Stephan asked for an independent review of the rig, its
    findings worked in without escalation where possible, kept lean: review r03 started (brief
    `md/materials/r03_rig_review_brief.md`).
  - r03 done 2026-10-27 (`md/materials/r03_rig_review.md`): references, band forming, mass shift, 0° index and family
    rule right; 11 findings. Manager rerun of its probes `[r1]`, `[r3]`, `[r5]`–`[r9]`: all reproduce (mixed stack
    with a mass-law layer in front: `ref_auto` wrong, 1.4e-2 to 0.59 in R; box-before-fill exception admits a traced
    dense leaf, R up to 1.2e13; drywall R 1.45e-2 uncapped, 0.222 at 10 calls; plaster side 0.675 dB reversed, 3e-6
    at one side; `rescale` triples `mpp`). No finding needs Stephan: the rig agent works them in (brief revision 2:
    F1–F10; reflection stays uncapped, transmission of a filled box at 10, R at 10 printed under ISSUE 3); the manager
    fixed the documents (F11: section 0 items 5, 6, 8; wish-list 1.10 and the stage-2b cap wording; issue 3 gains the
    traced dense leaf behind a box).
  - Review fixes done 2026-10-27 by the rig agent (906 lines, was 908). Manager checks: rebuild clean; `[wave]`
    141128 numbers identical; `[acoustic]` 43002 / 69, other files' rows identical, moved only the concrete | plaster
    row (one side: 1.95e-4 → 3.1e-6 dB), Litvin's side digit, five T columns now `n/a`, the ISSUE 3 block (R at 10
    interactions 0.222 printed beside 1.45e-2 uncapped) and the new refusals; whole suite 1347118 / 1312 (−141: 126
    `isfinite`, 16 porous per-octave checks, 5 T checks that could not fail, 1 assertion on an ISSUE line; +7);
    `[materials]` 0.36 s at 12 threads, load 3.4–5.6; `check_docs.py` passes. To Stephan with a summary and a commit
    message.
  - Committed by Stephan (e5d3743, 2026-10-27). Baselines regenerated at e5d3743: `[wave]` 163474 / 178,
    `[acoustic]` 43002 / 69, whole suite 1347118 / 1312. Second independent review r04 started (same procedure; brief
    `md/materials/r04_rig_review_brief.md`: own pass first, then the r03 fixes checked and what r03 missed).
  - r04 done 2026-10-27 (`md/materials/r04_rig_review.md`): r03 fixes right except three incomplete; 8 new findings.
    Manager rerun of `[p1]`, `[p2]`, `[p3]`, `[p7]`: all reproduce (NaN in R and T at 250 Hz prints the clean numbers
    and passes every bound; the specimen-side rerun never runs; paths equal `transmit()` within 2e-5 dB on single-path
    stacks; `set_length` has no effect; Komatsu NaN at 50 Pa·s/m², 10 kHz). No finding needs Stephan: the rig agent
    works them in (brief revision 3); the manager fixed the documents (section 0 rig and State lines; wish-list 1.10
    "at or above 2–3 ×" and the per-node reference print; trap: dense leaf on a `tf` wall 1.76e-2 in R).
  - Second review fixes done 2026-10-27 (rig 917 lines). Manager checks: rebuild clean; `[wave]` identical;
    `[acoustic]` 43003 / 69, 118951 numbers identical, two last-digit moves (Litvin at its side 3.523 m) and one new
    line (double leaf: paths against `transmit()` 4.95e-5 dB); whole suite 1347119 / 1312; `[materials]` 0.48 s at
    10 threads, load 5–12; the NaN probe now fails every check (dT, dR, R + T infinite, tier 2 false);
    `check_docs.py` passes. To Stephan with a summary and a commit message.
  - Committed by Stephan (e578aee, 2026-10-27). Baselines regenerated at e578aee: `[wave]` 163474 / 178,
    `[acoustic]` 43003 / 69, whole suite 1347119 / 1312. Next: rig item 10, the object test.
  - Object test started 2026-10-27 (Stephan: measure the porous body's accuracy against cost at the cap of 10 first,
    stop, then build): brief revision 4, the rig agent on phase 3a.
  - Phase 3a done 2026-10-27 (report "Phase 3a"; probe `build_linux/scratch/rig_ext/p3a/obj.cpp`): about 3 µs per ray
    event and tone, 19.4 events per ray at the cap of 10, internal bounces of the porous body carry < 1 % after the
    4th interaction; cap 10 against 20 within 1.5e-4 relative; 16 directions × about 65 rays, 6 octave tones,
    `min_gain_dB` −60 and at most 3 reflections: 0.18–0.19 s per body, within 0.5–1.3 % of the dense reference
    (cushion and sofa, three seeds). Manager rerun of the cushion at load 1.8: 0.181 s, 0.53 / 0.56 / 0.66 % (equal).
    Grazing relaunch in 2 of 64 directions on the sofa. Choice of grid to Stephan.
  - Decided by Stephan 2026-10-27: object test at 16 directions × about 65 rays, six octave tones, cap 10,
    `min_gain_dB` −60, at most 3 reflections; the grazing ray as a single-ray ISSUE 6 fixture. Rig agent on phase 3b.
  - Phase 3b done 2026-10-27 (rig 1041 lines, +124). Manager checks: rebuild clean; `[wave]` identical; `[acoustic]`
    43021 / 70, only the new case's 21 lines; whole suite 1347137 / 1313 (+18, +1); the object case 0.077 s at 12
    threads, 0.244 s CPU on one thread, load 2–3.4; `check_docs.py` passes. Hard box 0.6306–0.6331 m² against α·S
    0.65 (−3.0 %, the 16 directions; bound 6 %; −0.4 % at 64 directions: left at 16, wide gates); volume = chords
    (0.1937 m², 4mV 0.2000); cushion within 0.47 % of the dense reference (bound 2 %); ISSUE 6 single ray 1.000 →
    1.868 in ten calls. Issue 6 entry and wish-list 1.10 updated. **Rig extension complete; next: material 1,
    concrete.**
  - Committed by Stephan (5081648, 2026-10-27). Stephan asked how to parallelize the material list; three options
    laid out (serial; parallel agents under one manager; one session per group). **Decided 2026-10-27: option (b)**,
    parallel agents under one manager in worktrees; batch 0 `concrete` alone with the rig moved into a header, then
    wait for his feedback; per batch a findings summary, decisions and recommendations. Handed to a fresh manager
    session (the old one was force-compacted). Plan in section 0. Baselines regenerated at 5081648: `[wave]` 163474 / 178,
    `[acoustic]` 43021 / 70, whole suite 1347137 / 1313.
  - New manager session 2026-10-27: state checked (HEAD 5081648, only this file modified; the three baselines in the
    root; no material brief yet). Batch-0 brief `md/materials/m01_concrete_brief.md` written: the agent in its own
    worktree (rehearses the batch mechanics; brief, baselines and report by absolute path in this checkout), four
    phases (plan stop; the move verified with rowdiff in `--order lex`, since a case in `test_acoustic_mat_*` links
    before the rig file; columns with the coincidence question measured, `coiF` free at 6.9 kHz against the physical
    fc of 114 Hz at 150 mm; the case and rows stop). To Stephan for approval.
  - Brief approved by Stephan 2026-10-27; the concrete agent started on phase 1 (plan) in its own worktree.
  - m01 phase 1 done 2026-10-27. Manager reruns: the split rehearsal `[acoustic] --order lex` against the unsplit
    worktree binary 119028 numbers identical, 9829 lines, 0 structural; decl order against `test_acoustic.txt` only the
    77-line concrete block moved, 118523 identical; the header holds lines 41–686 verbatim. Fits (`evalm.py`):
    `coiF` free (4413 Hz, `coiQ` 1.27, `fRef` 9.398e-5 = a mass level of 183 kg/m² at 150 mm) worst envelope miss
    0.25 dB at 150 and 203 mm, Litvin 76-66 0.74 dB; the preliminary columns 0.58 dB; the physical plate (fc 114 Hz,
    η 0.005) 8.32 dB; `coiF` pinned at 114 Hz 1.25 dB with knobs at their bounds and up to 14 dB off on the 50-angle
    grid. Data script: mass shifts −0.114 / −2.786 dB, α law 0.0204 at 1 kHz with exponent 0.3965, Mahn 3.3 dB below and
    8.7 dB above the envelope (printed only). Side effect of the fitted mass level (the engine's bonded mass is
    `thk`/`fRef`): a bonded 15 mm plaster adds 0.88 dB instead of 0.48 dB, parquet 0.49 instead of 0.26 (normal-
    incidence mass law, script); to be measured by the plaster and finish agents. Setup findings: the Agent tool
    created the worktree at cebd0c0 (fast-forwarded by the agent to 5081648); the Write tool refuses paths in the main
    checkout from a worktree agent (`cp` works). D1 (`coiF` free, recommended), D2 (`e` 1901), D3 (joint α law) to
    Stephan.
  - Decided by Stephan 2026-10-27: D1 (a) `coiF` free, a fit knob named in the case comment (0.25 dB is not worth an
    engine rework, it is in the noise; the physical plate's 7–11 dB miss recorded in the stage-2b file, no ISSUE in the
    case); D2 `e` 1901; D3 the joint PTB #30 + Nell law. Agent continued with phases 2–4 (plus a scratch probe: what a
    bonded 15 mm plaster adds to the new row against the bonded plate).
  - m01 phases 2–4 done 2026-10-27 (report sections appended by the manager; the agent's Write was refused). Merged
    into this checkout by copying the six files. Manager checks at load 3–4: rebuild clean; `[wave]` 141128 numbers
    identical, 163474 / 178; `[acoustic]` 43191 / 70, 118523 identical, only the concrete block replaced (77 → 336
    lines); whole suite 1347307 / 1313 (+170: the new case 174, the old 40 gone, the table case +36), 262847
    identical, the same two blocks; `[materials]` 450 / 11, `Material - concrete` 0.036 s at 12 threads;
    `check_docs.py` passes; ISSUE marks 1, 3, 4, 6 only. Rows: `concrete`, `concrete_150mm`, `concrete_200mm`
    (`fRef` 9.3982e-05, `att` 34.71, `attB` −0.43851, `coiF` 3.8598, `coiQ` 1.2677, `resF` 0.069636, `resQ` 1.1529,
    `tf` 0.00054451, `tfB` 0.3965, `e` 1901, `side` 3.22). Plaster probe: 15 mm adds 0.41–2.33 dB on both faces at
    150 mm, about twice the mass law. Documents: wish-list 1.1 confirmed, 1.6 exception removed, 2.1 entry rewritten;
    stage-2b issue 7 (heavy-wall coincidence, the bonded-mass side effect) and the rig paths; guidelines 12 one row.
  - Committed by Stephan (a97c4ae, 2026-10-27; a first commit with the agent's worktree as a gitlink was undone
    locally, `.claude/worktrees/` is now ignored). Then the `fixes` branch was merged (7e9fe61, v0.12.4 26da9f9; the
    large-scale model: `include/quadriga_gbsm.hpp`, conf functions moved there, Python `quadriga_lib.gbsm`,
    `qd_sos_acf_interp` renamed `qd_interp_linear`). Manager check at 26da9f9: our files unchanged by the merge except
    one comment line in `qd_test_diag.hpp` (`QD_DIAG_GBSM`); rebuild clean; `[wave]` 141128 numbers identical;
    `[acoustic]` 120637 identical to the batch-0 run; whole suite 1354225 / 1384, every changed number in the conf-file
    cases. Baselines regenerated. New coding rules adopted (section 3). Open for Stephan: the rig and the tools cite
    working documents and tracking ids (22 `ISSUE <n>` marks; wish-list 1.7.2, 1.10, 2.4 and engine notes 9.2 in 5
    test comments; `md/acoustic_engine_notes.md` in 4 tool files and the tools README).
  - Decided by Stephan 2026-10-27: option (a), the cleanup before batch 1 (brief `md/materials/c01_references_brief.md`).
    Done by a cleanup agent in this checkout; the manager renamed the convention in the documents (issue names in the
    stage-2b headings; wish-list 1.10; checklist; template) and fixed one more working label ("the ray of phase 1").
    Manager checks: rebuild clean; `[wave]` 141128 identical; `[acoustic]` 120593 identical, 43191 / 70, the only
    changed lines the 7 printed labels (with the old labels put back the output is identical); whole suite 270612
    identical, 1354225 / 1384; `check_docs.py` passes; `sabine_to_random.py check` exits 0. Code marks:
    `ENGINE LIMIT (<name>):`, 11 in `test_acoustic_materials.cpp`. Left for Stephan: three citations of research-data
    lines at 2dad906 (concrete case, two self-checks). Baselines to be regenerated after the commit.
  - Research answer received 2026-10-27 (relayed by Stephan; research commit d501809): request 0 done (ids, map
    `data/id_map_2dad906.csv`), 17 rows of 2dad906 changed (manager check by script over the map: 1980 material rows
    mapped; the wish-list cites 2 of the changed rows, Fuchs 327 and 328 of 5.11), library 2544 rows. Plan to work it in
    before batch 1 proposed to Stephan.
  - Cleanup committed by Stephan (4641966); baselines regenerated there (`[acoustic]` the 7 new labels). Decided by
    Stephan 2026-10-27: rows cited as source key plus id, research pinned at d501809; the citation agent (c02) and the
    analysis agent (a03) started in parallel; batch 1 after the tier readings.
  - c02 done 2026-10-27: wish-list 392 of 426 line citations converted (545 ids), the test comments 10; 34 refused
    (author word does not match the row's source, mostly our misattributions: ADO rows given to Zimmer+Rohde, Kosten to
    Kuhl, Garai's range over non-σ rows, ground-class and water rows without an author word) left for the manager with
    the a03 update; changed rows met: Fuchs_2013_AppliedAcoustics#13, #14 (5.11; #14 f125 1.14 → 1.023, still above
    one). Manager checks: both test files identical to HEAD without comments; every id found exists at d501809 and is
    in the map; six concrete ids name the right rows; `check_docs.py` passes.
  - a03 done 2026-10-27 (`md/materials/a03_research_fold_report.md`, 1003 lines; scripts `build_linux/scratch/a03/`):
    23 entries touched, proposed text per entry in ids, 12 decisions D1–D12, batch cut. Manager reruns (`masonry.py`,
    `aac.py`, `doors.py`): densities Weise#1 1760, Hopkins#68 1784, Secchi A 1905, Granzotto B 1558 kg/m³; Weise#1
    minus (bare trend + two increments) −2.8 / −3.5 / −3.6 / −3.1 / −1.3 / −3.1 dB (125 Hz–4 kHz; outside ±3 dB at
    250 Hz–1 kHz and 4 kHz), minus the bare trend at its mass +1.8 … −0.7; bare-wall scatter about the trend up to
    3.96 dB; aircrete 10 cm density range 4.44 dB, the four walls spread 2.5–5.8 dB per band; hollow-core door mass
    from Kim's R 7.6–9.2 kg/m² (500 Hz–4 kHz). Batch-1 blocking: D7 (Weise as measured, range to 250 mm), D8
    (plastered walls before bare plus increment). To Stephan with all twelve.
  - Decided by Stephan 2026-10-28 (the twelve in five parts first): D7 (a) Weise's 12-laboratory mean measured, tier 1,
    `masonry_heavy` valid to 250 mm, class density about 1700–2400 kg/m³; D8 (a) walls measured plastered on both faces
    are the references where they exist, bare plus increment only otherwise, the old rule's miss printed; D3 (a)
    Granzotto B for `masonry_hollow_200mm`, Secchi A printed (offered to the heavy class), `masonry_hollow_250mm` back
    with Granzotto C and E; D4 (a) aircrete 100 mm against the range of the four walls, density inside 480–800 kg/m³,
    4.44 dB stated; D5 (a) field values printed only; D9 (a) hollow-core door 9.3 kg/m² (NBS), Kim the spectrum;
    D10 (a) doors stay 40 mm; D11 (a) carpet tile A 1 at 63 Hz, 1.3 amended to a hard face in that band; D1 (b) Caimi
    printed as the upper bound of fabric seats, the leather sofa tier 1; D2 (a) Focus A tier 1 at E-50 and E-200, the
    row renamed `ceiling_tile_20mm`, Aoki printed; D6 (a) the bookcase tier 3 with the fill ratios, **no author
    query**; D12 (a) Fuchs 2005 rows cited, 2013 printed. Wish-list update by agent c03 (brief
    `md/materials/c03_wishlist_brief.md`).
  - c03 done 2026-10-28: wish-list sections 2, 3, 4 (item 8 closed), 6 updated, 827 ids (all at d501809, no short
    forms), the 34 refused citations resolved (ADO, Kosten, Primacoustic named; Garai's non-σ rows and Szelag's α rows
    corrected; ground σ and G separated); guidelines 5: field values 3–6 dB below the laboratory rating (Secchi Rw
    6 / 4 / 6 / 3 / 5 dB by script). Manager review: the tiers and table rows of 1.2, 1.3, 1.23, 1.24, 4.1, 5.15 as
    decided; D8 rule in 1.2; D6 without any query; D12 in 5.11; `ceiling_tile_25mm` gone; `check_docs.py` passes. To
    Stephan with one commit for the id conversion and the update. Next: the batch-1 briefs.
  - Committed by Stephan (063f05b, 2026-10-28). Batch-1 briefs written: `md/materials/batch1_common.md` (worktrees at
    063f05b, 4 threads each, own case only, CSV rows appended, reports copied with `cp`, ids and engine-limit names,
    plan phase first), `m02_masonry_heavy_brief.md` (group file `heavy`; open rule generic solid across thickness over
    140–250 mm), `m03_glass_wool_brief.md` (`porous`; fitted-σ check, Scrosati rows, `alpha` on a porous layer, the
    finite sample), `m04_glass_brief.md` (`plates`; reference row per class for glass, thin-plate coincidence at
    3.1 / 2.1 kHz, the absorption refit). To Stephan before the agents start.
  - Briefs approved by Stephan 2026-10-28; the three material agents started on phase 1 (plan), each in its own worktree.
  - m03 glass wool phase 1 done (worktree `.claude/worktrees/agent-a0f04c4dc30838bba`, fast-forwarded from 26da9f9).
    Manager reruns (`compare.py`, `finite.py`): at σ 13112 Komatsu / Miki / JCA (Dupont's own set) miss SFS by −0.072 /
    −0.066 / −0.084 at 250 Hz and −0.108 / −0.098 / −0.087 at 500 Hz; against the band [JCA #1, SFS] 0.049 / 0.035 /
    0.024 (JCA polished 0.023); fitted σ SFS ×1.83 (1–4 kHz ×1.53), Sabine converted ×1.45 (extended reaction),
    ×1.72–1.76 (local); Thomasson's diffuse-field excess of the 1.5 × 1.5 m patch +0.43 / +0.56 at 250 / 500 Hz is an
    upper bound, so the finite sample as the cause of SFS's +0.08 is bounded, not measured. Decisions D1 (JCA in the
    rig's `eps_mu`, about 6 lines, a rig change), D2 (the band [JCA #1, SFS] at ±0.05), D3 (tier 2 at 50 mm only),
    D4 (`alpha` 0), D5 (Scrosati rows not met). Probe time provisional (load 13.5).
  - m04 glass phase 1 done (worktree `.claude/worktrees/agent-a856d28aa70b22cbf`, at 063f05b). Manager reruns
    (`tfcap.py`, `final.py`): physical `coiF` (3121 / 2080 Hz) and η 0.005 with the rest fitted to the laboratory
    references: Quirt within 1.99 / 1.58 dB (4 / 6 mm), AGC 2.25 dB, 6 mm laboratory envelope 0.75 dB; the generic
    EN 12758 values missed by 3.86 dB (4 mm, 250 Hz) and 2.85 dB (6 mm); masses 10.32 / 14.77 kg/m² against 10.00 /
    15.00; α within 0.018 of PTB#207. Coincidence octave of the physical 4 mm plate 29.89 dB at `tf` 0.017, 21.74 dB at
    `tf` 1 (6 mm 30.32 / 22.24): the face's absorption caps the dip (proposed engine-limit name "pane coincidence
    capped by the face absorption"). Fit knobs at bounds in the 4 mm row (`att`, `resF` 1000 Hz). Decisions D1 (lab
    values asserted, generic printed), D2 (physical coincidence), D3 (PTB#207 law, A 2), D4 (`grid(200,1)`, 8 kHz off
    by 3.2 dB on 50 angles, about 0.17 s, load 9). Side finding: tier-1 bound 2.52e-4 dB on 400 angles at 10 kHz.
  - m02 masonry_heavy phase 1 done (worktree `.claude/worktrees/agent-adf28abc4018f5af8`, at 063f05b). Manager rerun
    (`meval.py fit_g.json`): one generic fit (ρ 1772 kg/m³, `coiF` 7427 Hz with η ≈ 1, a knob as for concrete; mass level
    0.704 × physical) meets Weise#1 within 1.99 dB, Hopkins#68 2.23 dB, BIA#8 plus one increment 2.08 dB; the envelope
    of the valid walls 0 dB at 100–300 mm, but it is 3–6 dB wide, so the ±1 dB envelope check asserts nothing here;
    against the class trend the row reads +3.31 (100 mm, 125 Hz) to −3.39 dB (300 mm, 1 kHz). Decisions D1 (range
    150–250 mm, the slope to issue 7), D2 (envelope of the four valid walls), D3 (ρ 1772), D4 (Weise at the row's
    side), D5 (BIA#8 plus one increment at 150 mm), D6 (mass level free). Times provisional (load 8.2). All three
    plans to Stephan together (15 decisions, the JCA rig change, one new engine-limit name).
  - Decided by Stephan 2026-10-28 (all fifteen as recommended): masonry M1 range 150–250 mm, 100 / 300 mm printed,
    the slope under issue 7; M2 the four valid walls' envelope kept, noted as wider than its tolerance; M3 ρ 1772;
    M4 Weise at 3.22 m; M5 BIA#8 plus one increment at 150 mm; M6 mass level free. Glass wool G1 JCA with Dupont's set
    (rig change approved: JCA in `eps_mu`, minimal, self-check rows identical); G2 the band [JCA #1, SFS] ±0.05, the
    converted Sabine ±0.107, SFS marked as finite-sample; G3 tier 2 at 50 mm only; G4 `alpha` 0 on porous rows; G5
    Scrosati rows to batch 2. Glass P1 laboratory values asserted, generic printed; P2 physical coincidence (η 0.005);
    P3 PTB#207 law, A 2; P4 `grid(200,1)`. The three agents continued with phase 2 (build).
  - Batch 1 built 2026-10-28 and merged by the manager (six files copied; the CSV rows appended in wish-list order, all
    three agents' CSV diffs append-only). Manager checks at load 2–4: reconfigure and rebuild clean; `[materials]`
    986 / 14 (concrete 0.034 s, masonry 0.054 s, glass 0.043 s, glass wool 0.084 s at 12 threads); `[wave]` 141128
    identical, 163474 / 178; `[acoustic]` 43727 / 73, 120630 identical, one inserted block of 1064 lines (the three
    cases); whole suite 1354761 / 1387, 270649 identical, the same insertion; `check_docs.py` passes. Rig change
    (glass wool): 7 lines, a JCA branch in `eps_mu`, formulas checked by the manager against the standard JCA form,
    source [Jaouen_2026_APMRJCA] in the research index; self-checks 8179 numbers identical. Open for Stephan: the 4 mm
    glass refit (agent: `att` 0, `resF` 2315 Hz as a knob, worst Quirt 1.82 dB, dip 3413 → 3886 Hz against the physical
    3121 Hz) against the approved phase-1 fit (dip 3413 Hz, worst Quirt 1.99 dB, two knobs at bounds); both pass the
    case (81 / 81, manager run). Documents: wish-list 1.2, 1.6, 5.1 confirmed with built paragraphs, 4.1 and 4.4
    decided; stage-2b issue 7 (masonry slope), issue 8 new (pane coincidence capped by the face absorption).
  - Decided by Stephan 2026-10-28: (a) the approved phase-1 fit for `glass_4mm` (dip 3413 Hz, worst Quirt 1.99 dB;
    the refit's in-band resonance knob moved the dip 14 % from the physical value). Manager swapped the row and one
    comment word (`att` fitted in both rows); reran: case 81 / 81, `[materials]` 986 / 14, `[wave]` identical,
    `[acoustic]` 43727 / 73 and the whole suite 1354761 / 1387 with the same single insertion, `check_docs.py` passes.
    To Stephan for the commit; baselines to be regenerated after it. Next: batch 2a (masonry_hollow, rock_wool, doors).
  - Committed by Stephan (a73d1a2, 2026-10-28). Baselines regenerated there: `[wave]` 163474 / 178, `[acoustic]`
    43727 / 73, whole suite 1354761 / 1387. The three batch-1 worktrees removed (their changes checked in main first;
    branches without commits deleted); the batch-0 worktree `agent-a00f2c2136594041a` kept. Batch-2a briefs:
    `batch2a_common.md` (HEAD a73d1a2; lessons: the case grid in the fit cost, build only the approved design, the
    confirmed cases unchanged), `m05_masonry_hollow_brief.md` (heavy; range and one row for block and clay),
    `m06_rock_wool_brief.md` (porous; the interlaboratory-rows rule with the Scrosati stone wool), `m07_doors_brief.md`
    (plates; the first lumped rows, coincidence, the 315 Hz dip). To Stephan before the agents start.
  - Briefs approved by Stephan 2026-10-28; the three batch-2a agents started on phase 1 (plan), each in its own worktree.
  - m06 rock wool phase 1 done (worktree `.claude/worktrees/agent-af8355d6affe51559`, at a73d1a2). Manager reruns
    (`models.py`, `fitw.py`, load 15–18): Miki fitted to Kosten #7 and Kuhl #2 together 25731 (extended) / 28213
    (local) Pa·s/m², near the stone wool's measured 30300; at 30300 and 500 Hz Komatsu 0.448, Miki 0.428,
    Delany-Bazley 0.387 (extended); the Scrosati 105 mm chamber means weighted by 1/s_R² give ×1.32 / ×1.39 for
    extended reaction but ×0.24 / ×0.23 for local reaction (the pass depends on the reaction model). The agent's side
    finding on page 10.5 (ε and μ inverted) is wrong: page 10.1 defines z = Z_air/Z, the inverse of the rig's z, so
    ε = n z, μ = n/z there equals the rig's ε = n/z, μ = n z. Decisions D1–D5; D4 (e) needs a `--sR` weighting option
    in `tools/acoustic/sabine_to_random.py` (4 lines, outside the agent's allowance).
  - m07 doors phase 1 done (worktree `.claude/worktrees/agent-abaf75fed51c5c01d`, at a73d1a2). Manager rerun of the
    probe (one thread, load 14): solid with `coiF` as a knob meets Quirt#1 within 1.10 dB, Morin#2 1.12 dB, their
    envelope 0 dB; hollow-core with the physical mass-air-mass `resF` (283 Hz) meets Kim within 2.02 dB; the 44 mm
    rescale equals the 40 mm row within 3.2e-7 dB; the probe's 14 failed assertions are all in references the plan
    prints only (NBS 632 6.27, Morin#1 gasketed 3.42, NBS 633 4.68, NBS 634 11.46 dB). Decisions D1 (solid `coiF` a
    knob; physical plate ~7 dB off, name proposed "thick-panel coincidence plateau" or issue 7 extended), D2
    (hollow-core dip carried by `resS` +1 at 283 Hz; wish-list 1.24 "not expressible" to be corrected), D3 (mass
    level free; `e` = m″/(ρ₀·thk) so the stated mass enters the shift without an override), D4 (Quirt#1, Morin#2, Kim
    asserted; NBS 632/633/634 and Morin#1 printed), D5 (rescale asserted by one run at 1.1 ×, bound 1e-4 dB).
  - m05 masonry_hollow phase 1 done (worktree `.claude/worktrees/agent-a834bb2bd76bd12d1`, at a73d1a2; two late fits
    ran six workers at load 8). Manager rerun (`heval.py fit_w4kpb.json`): mass level 0.633 × physical; per wall (dB
    outside, 125 Hz–4 kHz): Warnock#3 at 166 mm worst 2.86 (2 kHz; its 4 kHz octave printed, two of three thirds), BIA#14
    plus its own increment at 100 mm 2.52, Granzotto B at 200 mm 2.02, Granzotto C at 250 mm −2.77 at 4 kHz, E +3.14 at
    2 and 4 kHz: the two clay walls of one build and thickness pull opposite by about 6 dB, so no row meets each within
    ±3 dB; the agent's pass needs a new ±0.3 dB figure-reading budget (3.14 against 3.3; Litvin's plot readings took
    none). Decisions D1 (100–250 mm), D2 (one row; the tolerance question), D3 (ρ 1385), D4 (sides 3.22 m). All three
    batch-2a plans to Stephan together (14 decisions and one tools change).
  - Decided by Stephan 2026-10-28 (all fourteen as the manager recommended): H1 hollow masonry asserted 100–250 mm
    (BIA#14 plus its own increment at 100 mm), 300 mm and NBS printed; H2 one row for block and clay, **walls of one
    build and thickness that differ by more than the tolerance are asserted as their band** (Granzotto C and E at
    250 mm), no figure-reading budget, Warnock's 4 kHz octave printed; H3 ρ 1385, mass level free; H4 sides 3.22 m.
    R1 rock wool one row at the measured 30300, the Sillan rows its 50 mm references; R2 Miki (page 10.5 to read
    "whichever meets the reference": phase-4 page correction); R3 Kosten#7, Kuhl#2 asserted, the chamber values
    printed; R4 (b) the Scrosati stone wool printed only, no tools change (the open rule stays open for framed
    samples); R5 25–200 mm by the model. T1 solid door `coiF` a knob, issue 7 extended to timber leaves; T2
    hollow-core dip by the physical `resF` 283 Hz; T3 the stated mass in `e` = m″/(ρ₀·thk), no override (wish-list
    1.7.10 to be amended); T4 Quirt#1, Morin#2, Kim asserted, NBS and Morin#1 printed; T5 the rescale by one run at
    1.1 ×. The three agents continued with phase 2 (build).
  - Batch 2a built 2026-10-28 and merged by the manager (three test files copied; CSV rows appended in wish-list order,
    all three diffs append-only). Manager checks at load 3–5: rebuild clean; `[materials]` 1585 / 17 (masonry_hollow
    0.048 s, doors 0.022 s, rock wool 0.075 s at 12 threads); `[wave]` 141128 identical; `[acoustic]` 44326 / 76, 127980
    identical, three inserted blocks (633 / 175 / 264 lines: the three cases); whole suite 1355360 / 1390, 277999
    identical, the same insertions; `check_docs.py` passes. Results: hollow masonry Warnock 2.86, BIA#14 2.52,
    Granzotto B 2.18, the C+E band 0.77 dB; doors Quirt 1.10, Morin#2 1.12, Kim 2.02 dB, rescale within 1e-4 dB; rock
    wool Kosten 0.018, Kuhl 0.015. Documents: wish-list 1.3, 1.23, 1.24, 5.2 confirmed with built paragraphs, the
    hollow-core dip sentence corrected, 1.7.10 (lumped mass in `e`), 1.10 (band rule), 1.12 (porous model by the
    reference); stage-2b issue 7 extended to timber doors. To Stephan with the batch report.
  - Committed by Stephan (cbe2ae3, 2026-10-28). Stephan: no temporary files on the repository's disk (it is backed up
    with history). Manager: all four agent worktrees and their branches removed after checking each against main (the
    batch-0 one held only older versions of committed files); `md/materials/` (41 files), `build_linux/scratch/` (131
    files) and the three baselines copied to `/home/stephan/quadriga-lib-agents/` (`materials/`, `scratch/`,
    `baselines/`; counts and bytes checked), then removed here; baselines regenerated there at cbe2ae3. From batch 2b
    on the manager creates the worktrees in `/home/stephan/quadriga-lib-agents/worktrees/` and starts the agents
    without `isolation`.
  - Stephan 2026-10-28: `build_linux` is built by the repository's `make` and not touched. The manager's own build
    configured in `/home/stephan/quadriga-lib-agents/build` with the same options (90 s at 12 threads, nothing written
    into the source tree); its `[wave]`, `[acoustic]` and whole-suite output equals the baselines number for number
    (141128 / 135185 / 285204, no structural difference). A `tools/acoustic/__pycache__` left by an agent removed;
    agents run Python with `PYTHONDONTWRITEBYTECODE=1`.
  - Manager session handed over at 2b42dcd (2026-10-28). State checked: HEAD 2b42dcd (since cbe2ae3 only the manager
    file and the template changed, so the cbe2ae3 baselines hold: `[wave]` 163474 / 178, `[acoustic]` 44326 / 76, whole
    suite 1355360 / 1390); no agent worktrees or `agent/*` branches; the manager's build up to date. Batch-2b briefs
    written in the agents folder: `batch2b_common.md` (worktrees created by the manager at 2b42dcd in `worktrees/`,
    scratch in `scratch/<id>/`, reports written directly, build options of the manager's build, the confirmed cases
    of 2a added, lessons of 2a: the band rule, no figure-reading budget, check page claims against definitions),
    `m08_masonry_light_brief.md` (heavy group; plaster on a class whose D4 references are bare, the coincidence inside
    the band and the range, which walls are light units, density, sides), `m09_pu_foam_brief.md` (porous group; the
    family `pu_foam` + `pu_foam_50mm` from wish-list 3.6 and 5.3; the model, the reference band with the synthesis below
    the JCA prediction, the fitted-σ check rerun, the solid for cushions, the range), `m10_glazing_brief.md` (plates
    group; one case for both units; the mass-air-mass dip, one resonance column for the triple's two, coincidence,
    the slope above the resonance, AGC against the Quirt cavities, mass level, grid, the rescale check). To Stephan
    before the agents start.
  - Briefs approved by Stephan 2026-10-28. Worktrees created by the manager at 2b42dcd
    (`/home/stephan/quadriga-lib-agents/worktrees/m08_masonry_light`, `m09_pu_foam`, `m10_glazing`, branches
    `agent/<id>`); the three agents started on phase 1 (plan) without `isolation`, 4 threads each.
  - m09 pu_foam phase 1 done (worktree clean at 2b42dcd). Manager reruns (`compare.py`, `cushion.py`, load 8):
    JCA with Dupont's PU2 set within 0.003 of the band [JCA #6, synthesis #8] (Komatsu 0.071, Miki 0.053,
    Delany-Bazley 0.084 outside it); against the converted chamber value 0.084; the synthesis 0.025–0.056 below the JCA
    model at 250 Hz–4 kHz (glass wool +0.08 above); fitted σ of the synthesis ×1.02 (Miki) / ×1.20 (Komatsu); the
    engine notes' 10.3 kPa·s/m² reproduced; a limp frame of an assumed 20 kg/m³ lowers α by up to 0.050 at 500 Hz
    (bounded, not measured). Cushion foam exists: Rubino #9 (σ 2700, measured), #11 (porosity 0.96); against the
    authors' modelled free layup #3 Miki at 2700 within 0.024, the PU2 row up to 0.216. Decisions D1 (JCA power laws,
    fit from 125 Hz), D2 (band ±0.05, conversion ±0.107, synthesis printed, not marked), D3 (`pu_foam` the acoustic
    foam; 3.6 to a padding-foam row with 3.8), D4 (25–200 mm, 250 Hz–4 kHz). Case 0.148 s, provisional (load 7.8).
  - m10 glazing phase 1 done (worktree clean at 2b42dcd). Manager reruns (`final.py`, the library probe on
    `grid(200,1)` 2.9 s for all variants, `tfsens.py`, `tfcap.py`; load 2–5): double unit with physical columns (`resS` +1
    at the mass-air-mass 211.3 Hz, coincidence 3123 Hz, η 0.005, the PTB#208 law, `att` rising) meets AGC#2 within
    2.81 dB (250 Hz; 1 kHz −2.75, 4 kHz +2.73); mass level 0.57 × the stated 20 kg/m². Triple with the physical lower
    resonance misses AGC#14 at four octaves (125 Hz +3.37, 250 Hz +3.34, 1 kHz −3.31, 4 kHz +3.25 dB); Quirt#26
    within 2.83 dB (agent: 2.89). The face cap T ≤ `tf` also caps the mass-air-mass dip: 250 Hz reads 19.57 against
    12.97 dB with `tf` 1 (double), 22.37 against 18.15 (triple); at 4 kHz 39.29 against 29.82 dB. Decisions G1 (physical
    `resF`; fallback free at 167 Hz, 2.25 dB), G2 (triple: AGC#14 printed under the face-cap engine limit, issue 8
    extended to the dip), G3 (physical coincidence), G4 (AGC asserted; the Quirt pair and Pilkington printed),
    G5 (mass level free), G6 (`grid(200,1)`), G7 (no rescale loop). Case about 0.27 s, provisional.
  - m08 masonry_light phase 1 done (worktree clean at 2b42dcd). Manager reruns (`plaster.py`, `leval.py fit_A_620.json`,
    `coin.py fit_A_620.json`, the library probe 184 / 184 in 0.076 s, load 0.6–4): one solid (ρ 620 kg/m³, `coiF`
    463 Hz with `coiQ` 1.80 as a broad trough, mass level 0.397 × physical) meets the band of Granzotto #2–#5 at 100 mm
    within 2.41 dB and Hopkins #4 at 128 mm within 2.40 dB (both grids); gap bands 1.72 dB. The bare walls dip at
    500 Hz (R250 − R500 +0.6 to +5.4 dB, 8 walls), Hopkins #4 rises (−6.9 dB). Plaster at equal finished thickness:
    thin gypsum (650 kg/m³) −0.43 to +0.77 dB by mass law over 480–800 kg/m³ blocks, sanded 13 mm +1.52 to +3.47 dB;
    the heavy class's two-face increments 2.2–6.0 dB. Coincidence fixed against scaled with 1/h: in the library
    1.86 / 2.90 / 4.31 dB at 128 / 150 / 190 mm (125 Hz–4 kHz); against an independent thin plate (η 0.03,
    fc·h 53 Hz·m) 2.28 / 4.46 / 11.43 dB at 125 Hz–4 kHz (the manager first misread the 4.46 and 11.43 dB as the 63 Hz
    octave; they are the 250 Hz octave, corrected after the build, see below). Decisions D1 (D4 band as it is), D2 (joint fit, `coiF` a knob), D3 (asserted 90–130 mm,
    `_150mm` twin only, issue 7 extended), D4 (Warnock 1995 #2 a hollow lightweight block: printed in 1.3; #4 printed),
    D5 (ρ 620), D6 (sides 3.22 m). Side note: `python3 -I` ignores `PYTHONDONTWRITEBYTECODE`; use `-I -B`.
  - Decided by Stephan 2026-10-28 (all seventeen as the manager recommended): masonry_light L1 the D4 band as it is
    (thin plaster stated), L2 the joint fit with `coiF` a knob, the band and Hopkins #4 asserted, L3 asserted 90–130 mm,
    `_150mm` twin only, 150–250 mm printed, issue 7 extended (in-band sizes), L4 Warnock 1995 #2 to the hollow class
    (printed), #4 printed, L5 ρ 620, L6 sides 3.22 m; pu_foam F1 JCA power laws from 125 Hz, F2 the band ±0.05 and the
    conversion ±0.107, F3 `pu_foam` the acoustic foam, wish-list 3.6 to a later padding-foam row with 3.8, F4 25–200 mm;
    glazing G1 double with the physical `resF`, G2 triple: AGC#14 printed under the face-cap mark, issue 8 extended to
    the mass-air-mass dip, G3 physical coincidence, G4 AGC asserted, G5 mass level free, G6 `grid(200,1)`, G7 no rescale
    loop. The three agents continued with phase 2 (build).
  - m09 pu_foam built (worktree: CSV +2 rows appended, equal to `cols_jca_10605_f125.json`; porous file +70 lines, one
    case after rock wool). Manager rerun: `Material - pu_foam` 56 / 56, 0.156 s at load 1.8. Agent: `[materials]`
    1678 / 18, the existing cases' 22734 numbers identical, one inserted block. Merge note: the padding-foam remark in
    the comment gets its source id (Rubino #9).
  - m10 glazing built (worktree: CSV +2 rows appended, equal to `fit_D.json` / `fit_T.json`; plates file +64 lines, one
    case after the doors, one `ENGINE LIMIT (pane coincidence capped by the face absorption)` mark). Manager rerun:
    42 / 42, 0.139 s at load 1.8; AGC#2 +0.14 / +2.81 / +0.15 / −2.75 / −1.24 / +2.73 dB (125 Hz–4 kHz); the triple
    printed against AGC#14 +3.37 / +3.34 / +2.00 / −3.31 / −1.98 / +3.25 dB; Quirt#26 within 2.83 dB; library minima
    of the dip 229 and 186 Hz (agent's sweep). Agent: `[materials]` 1665 / 18, the existing numbers identical.
  - m08 masonry_light built (worktree: CSV +3 rows appended, equal to `fit_A_620`; heavy file +98 lines, one case after
    hollow masonry). Manager rerun: 196 / 196, 0.162 s at load 1.6; band 2.41, Hopkins 2.41 dB. The agent corrected the
    manager: `coin.py` prints 125 Hz–4 kHz, so the thin-plate misses 2.28 / 4.46 / 11.43 dB at 128 / 150 / 190 mm are in
    band (the 250 Hz octave); the approved range 90–130 mm stands. Warnock 1995 #2 against the hollow row (agent, not
    rerun): within +2.37 … −1.87 dB at 125 Hz–2 kHz, −3.70 dB at 4 kHz.
  - Batch 2b merged by the manager 2026-10-28 (three test files copied, CSV rows appended in wish-list order 1.4, 1.7,
    5.3, every diff append-only; one manager edit: the padding-foam remark in the foam case cites
    [Rubino_2026_UpholsteredSeatingCircular#9]). Manager checks at load 1.4–1.9, 12 threads: rebuild clean;
    `[materials]` 2009 / 20 (masonry_light 0.083 s, glazing 0.042 s, pu_foam 0.080 s); `[wave]` 163474 / 178, 141128
    identical, no structural difference; `[acoustic]` 44750 / 79 (+424 = 196 + 42 + 56 in the cases, 130 in the table
    case), 135185 identical, three inserted blocks (865 / 146 / 268 lines); whole suite 1355784 / 1393, 285204
    identical, the same insertions; `check_docs.py` passes. Outputs in `/home/stephan/quadriga-lib-agents/scratch/b2b/`.
    Documents: wish-list 1.3 (Warnock 1995 #2), 1.4, 1.7, 5.3 confirmed with built paragraphs, 3.6 to a padding-foam
    row; stage-2b issue 7 extended to light masonry, issue 8 to the glazing units' mass-air-mass dip; guidelines 7
    (cushions). To Stephan with the batch report.

## 10. Scope of phase 0 (2026-10-23; naming and combinations superseded by wish-list 1.6 and section 3)

**The CSV.** `data/materials_acoustic.csv`, next to `data/materials_rf.csv`. The old name `materials_ac.csv` is avoided
on purpose: it was the outdated one-parameter table removed in 8d89b84, and the history should not mix the two. The
file is read by `obj_file_read` with an empty `.obj` name (only the table), so it obeys the reader:
- only `name` is text; every other cell is a number; no comments, no class or source columns (the class, the sources
  and the residuals live in the test case and in the notes);
- every row fills every column that has a non-zero default (`fRef`, `a`, `e`, `side`), since an empty cell reads as 0
  (and `side` 0 is the infinite plate);
- names are lowercase `snake_case` with the build in millimeters, no dots, no commas (`glass_4mm`,
  `concrete_cast_152mm`, `glass_wool_50mm`); a dot would be cut to a base name by the material lookup;
- frequency columns in GHz with the library's acoustic mapping (342.76271 m/s, `qd_acoustic_freq_scale`; decided
  2026-10-23, replacing the page's 343 m/s).

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

1. Phase 3a: the rig extension (section 0, items 1–10).
2. Phase 3b: every row family of the wish-list, in the order of its section 1.11 and then by the absorption tier,
   impact and the transmission tier; research results folded in as they arrive (wish-list section 5).
3. Phase 4: `md/acoustic_material_notes.md`; page 11.4 replaced by the table; the page corrections of wish-list 1.12
   (343 → 342.76271 m/s, rule 16, the `tf` 0 rule, `side`, the statements the research contradicts); the tools
   README; the transitory files deleted (`md/materials_kickoff.md`, `md/materials/`).
4. After step 3: the engine stage 2b (its own session, with the issue list and the table as regression tests); step 4,
   the reference simulation of the listening room (its inventory is collected there); steps 5 and 6.

Leftovers of step 2 to settle where they fall: page section 9 says `irr_glass` follows TR 38.901, though only its
`att` does; if `coiQ` changes meaning, page 11.1, 11.2 and 11.4 are re-measured. (`band_probe` has `attB` since
13e4e45.)

## 12. Traps

- **Sound speed:** the library maps acoustic frequencies with 342.76271 m/s (`qd_acoustic_freq_scale`, ISO 9613-1 at
  20 °C; the wrappers and `qrt_ray_trace` use it). `ac2rf()` (342.77), `lw::GHZ`, `txlib.py` and the page (343) differ;
  the table and the rig use the library's constant (decided 2026-10-23).
- **`lw` is not shared:** `lw::stack`, `lw::walk`, `lw::Mtl`, `lw::peak_f` live in `test_acoustic_leaf.cpp`, not in
  `ray_test_helpers.hpp`. Decided 2026-10-23: the rig copies the parts it needs into
  `test_acoustic_materials.cpp`; `test_acoustic_leaf.cpp` and the header stay as they are.
- **Reader:** an empty cell reads 0, not the default; a material file replaces the built-in table; a text cell in a
  numeric column is an error; a name's part after the first dot is a variant of its base name.
- **A material identical to air is air:** a box of air needs `e` 1.000001.
- **Research data:** cite rows by id (`<source key>#<n>`, d501809), never by line; generated files (never patch a copy; report a wrong row); PTB 63 Hz values equal to 125 Hz and
  compilation 8 kHz values equal to 4 kHz are copies; Sabine and placement values above one are no material
  property; the property name `alpha_random` covers every diffuse or unstated type, the `coefficient_type` column
  decides.
- **Runs:** test binaries from the repository root; a case name with a comma needs `\,` in a test spec; a file filter
  needs `-#`; single-ray fixtures on one OpenMP thread (`ub::OneThread`); `ray_progress` requires
  `thin_slab_threshold` ≤ 1; `cd` in a Bash call moves the session's directory.
- **Helper names:** `ray_test_helpers.hpp` has anonymous-namespace `Opt`, `Scene`, `Cfg`, `Hits`; a file-scope name
  of the same kind in a file that includes it is ambiguous, so the rig keeps its names in `namespace mt`.
- **Band verdicts:** in a `ray_progress` band every frequency takes the base frequency's follow-or-sum verdict; at the
  default `thin_slab_threshold` 0.15 a leaf on a box reads R up to 1.073 near grazing (octave α 8.3e-4 off). The rig
  runs at 0.
- **Stacks in scenes:** co-located sheets form their stack in the order of their faces in the mesh (reversed order: ΔR
  up to 0.999); `calc_diffraction_gain` (one path) is right for one layer, sheet | sheet and sheet | solid, not for a
  solid | solid of two wave media (4.9 dB, T > 1).
- **Coherent sums:** layered constructions need the coherent path sum; the power sum is off by up to 0.39 in octave α.
- **A dense leaf on a `tf` wall:** the wall's pad referred to air (notes 9.2) costs 1.76e-2 in R for a leaf (6.3e-3 for
  wool); leaf-on-wall cases need a reflection bound of about 0.04 (review r04).
- **Fits:** bound and densely sample; select on the evaluation grid; the mass level bound must reach about 3200 for a
  423 kg/m² wall; clip τ at one per frequency, not per band; coincidence needs about 4000 angles.

- **Worktree agents:** the Agent tool's `isolation` may create the worktree at an old commit (m01: cebd0c0, not HEAD);
  since batch 2b the manager creates the worktrees itself at the batch's HEAD and every brief says to check
  `git log -1`. Briefs, reports, baselines and scratch are in the agents folder, read and written by absolute path.
- **Catch order:** cases run in link order; `test_acoustic_mat_*` links before `test_acoustic_materials.cpp`, so a
  moved case moves its rows; compare moves with `--order lex`.

## 13. Lessons

- (phase 0) `build_linux` can be absent at the start of a session; reconfigure it as the generic build before
  regenerating baselines.
- (phase 1) A plan phase with scratch probes and a manager rerun of the probe's key commands settles a rig's design
  before any tracked file is written; light-panel coincidence decides the grid, not absorption.
- (phase 1) Stephan reviews rig code line by line: keep it lean, table-driven, with library helpers (`plane`, `cube`,
  `path::length()`); an "empty" rig of 1000 lines was too big.
- (phase 2) Build the list from the modeler's workflow (what a modeler sees at a venue, time per object), not from the
  literature; literature constructions become test references.
- (phase 2) Two independent high-effort reviews found real defects each time (wrong references, recipes hitting engine
  limits, a wrong statement of the manager's on `fRef` scaling); rerun the reviewer's scripts before passing a claim on.
- (phase 2) Check a claim about what a column does in the code before stating it (the `fRef` power laws; which sheets
  displace the host medium).
