# Step 3 of the acoustic project: the material tables (manager file)

The memory of the step-3 manager. Kept current after every step; a new manager session reads **section 0 first**, then
sections 1–8 and 12, then the three step-3 documents named in section 0. Section 9 is the dated log of how the
decisions came about (read it when a decision's reason is needed). `md/materials_kickoff.md` is the original plan of
step 3; where it differs from section 0, section 0 holds (it is deleted in phase 4). Public and tracked: no private details (no
rates, hours, money, personal circumstances, nothing about whose room the listening room is). Briefs and reports in
`md/materials/` are untracked working files; answers that are personal stay there. Old rule labels of the research
handoff (A13, A14, A15, A19, P7, M6) are used here only as working shorthand; public documents and commit messages name
the rule in words.

## 0. Start here: state at the start of phase 3 (2026-10-27)

**Where step 3 stands.** Phases 0–2 are closed. Phase 3 (building the material rows and their tests) starts with
the rig extension. The engine stays closed until every listed material is in (Stephan, 2026-10-26).

**Commits of step 3:** 5be85fb (manager file, agent template), 13e4e45 (the rig, `data/materials_acoustic.csv` with
one preliminary row, `band_probe` with 26 columns, `grid_probe.py`), 19323d9 / be65346 / bd0a170 (the wish-list and
the guidelines, drafts 2–4, after two independent reviews).

**The three step-3 documents** (tracked, public, no private details):
- `md/acoustic_materials.md`, the wish-list: six classes (structural, hard furnishing, soft furnishing, complex
  assemblies, treatment, outdoor), two evidence tiers per row (A absorption, T transmission; 1, 1p pending conversion,
  2, 3), impact, status, references (research lines at commit 2dad906), gaps. Read sections 1.3–1.11 in full: the
  naming rule (1.6: `_<NN>mm` last marks a sheet, no suffix a traced solid), the row-design rules (1.7: lumped rows
  carry the visible thickness and rescale every power-law coefficient with `fRef`, X·k^B), `side` (1.8), the phase-3
  procedure, the ISSUE convention and the tolerance table (1.10), the start list (1.11). Section 4: open design
  questions; section 5: adding a material; section 6: the research requests.
- `md/acoustic_modeling_guidelines.md`: how a venue is modeled (solids and sheets, Blender names and `csv_strict`, the
  checklist of forbidden and limited configurations: three co-located faces are not resolvable, only a sheet on a
  solid or a sheet | sheet pair; at most two bonded rigid bodies; objects as hard boxes or porous bodies; assets).
- `md/acoustic_engine_stage2b_issues.md`: engine issues 1–5 with sizes and reproductions, and the ISSUE convention.
  Engine findings are recorded there, never fixed in step 3.

**The rig:** `tests/catch2_tests/test_acoustic_materials.cpp` (namespace `mt`; size and counts in the State line
below): stacks room side first (sheets in front, traced solids behind, one sheet at the last solid's exit, `quadriga_lib::plane` / `cube`), transmission through `calc_diffraction_gain`
(lod 0, scalar, threshold 0), reflection through aimed rays and `ray_progress` (refraction off, threshold 0), plane
waves, coherent sums, 50 angles × one tone per third 25 Hz–10 kHz as the default grid argument, octaves as in the
research data, the library's sound speed (342.76271 m/s); `mt::check_material` (sheet = traced solid, tier 1 against
the plate rule or the fluid-layer transfer matrix, R + T ≤ 1, diagnostics, tier 2).
`data/materials_acoustic.csv` holds one preliminary row, `concrete_cast_152mm` (page 11.4; breaks the naming rule; to
be redone as the `concrete` family on the research data's Litvin values).

**Baselines** (generic `build_linux`, at e578aee, the rig after two reviews): `[wave]` 163474 / 178,
`[acoustic]` 43003 / 69 (the rig included), whole suite 1347119 / 1312. Files `test_wave.txt`, `test_acoustic.txt`,
`test.txt` in the repository root (untracked).

**Research:** the hand-over of 2026-10-27 (`md/materials/research_request_2026-10-27.md`: request 0 stable row ids,
requests 1–13 by impact, three data errors) is with the research manager; Stephan relays results. Phase 3 does not
wait: rows needing new data move down the order.

**Next item: the rig extension** (one rig agent, phases plan → build, as in phase 1; brief from the agent template's
rig variant; brief `md/materials/rig_ext_brief.md` written 2026-10-27, Stephan's choices for it in the section-9 log).
**State 2026-10-27:** items 1–9 built and committed (69a4555), reviewed twice independently and fixed (e5d3743: rig
906 lines; the second review's fixes then: 917 lines, 10 `[materials]` cases, 262 assertions, 0.4 s). Item 10, the object
test, built (rig 1041 lines, 11 `[materials]` cases, 280 assertions). Next: material 1, concrete.
Everything the documents need before material 1:
1. a thickness argument for rows with `thk` 0 (generic solids, porous bodies, finish solids, ground rows);
2. a specimen side per measured reference, and the class envelope shifted to the evaluated thickness by the mass law;
3. a 0° (normal-incidence) tier-2 comparison;
4. a sheet glued at a traced solid's exit (double leaf, curtain at a window, drywall precise recipe) and the issue-4
   fixture: plaster sheets on both faces of a block solid against the bonded plate, marked `ISSUE 4:`;
5. a stack rule that refuses three co-located faces (measured in a probe; no self-check case, no opt-out);
6. the ISSUE marks (a comment `ISSUE <n>:`, no assertion, the values printed) and the regression guard of the
   tolerance table (a `Measured` row of the library's own octaves);
7. one test case per row family accepted by the table case;
8. the porous models written in the test (Miki, Komatsu, Delany-Bazley printed; JCA when a reference uses it), and
   the finite-panel (Davy σ) plate in the added-insulation model of issue 1 (printed); the passive Miki fit of the
   ground classes deferred to the ground family;
9. a helper that rescales a lumped row's power laws with `fRef` (X·k^B);
10. the object test (rays from all directions on a body give its absorption area), as its own step, designed and timed
    against about 0.3 s per case.

Keep the rig lean (Stephan reviews it line by line); use library helpers instead of hand-written geometry; keep test
cases short. Then the materials, in the order of wish-list 1.11 (concrete first as the rig check, then by the
absorption tier, impact and the transmission tier), one family per material agent, until every listed row is in.

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

**Per phase.** Brief (`md/materials/<id>_brief.md`, from `md/materials_agent_template.md`) → one agent → the
manager's checks (section 4) → report to Stephan (what checks out, what does not, decisions in the form below, a commit
message) → his review and commit → this file updated.

**Per row family (phase 3; the wish-list 1.10 procedure).**
1. The manager writes the brief: the family (generic row and its sheets) and its class, reference rows (research
   lines at 2dad906, coefficient types, bands, specimen sides), the rules that apply, the tier-1 model, open rules and
   engine issues it will meet, the tolerance lines of wish-list 1.10 that apply.
2. The agent: data → columns by the decided rule → tier 1 (the library through the rig against the independent model)
   → tier 2 (the columns against the converted measurement per octave) → judge (reasonable against the known range;
   an engine limit becomes an `ISSUE <n>:` check and a stage-2b entry; otherwise a regression guard) → one lean
   `TEST_CASE` per family → the CSV rows. Stops where an open rule is met, with the options measured.
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
- **ISSUE marks:** every check that fails because of an engine limit carries `ISSUE <n>:`, does not fail the suite,
  prints its values, and the issue file lists it; no other non-failing checks.
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
| chamber to random for hanging fabrics, green walls, seat and audience blocks | the "1p" rows of the wish-list (1.3) | |
| the upholstery row | fit of the porous body to per-object areas, with the object test | |
| generic solid across thickness | the miss of one fit over the measured range; a thickness-scaled coincidence to stage 2b if too large (wish-list 4) | |
| reference row per class | now: tolerance table of wish-list 1.10 (class envelope) | decided 2026-10-26 (r01 D4, r02 F5) |
| dense ground classes | the passive Miki fit degenerates at 2 and 20 MPa·s/m² (μ′ near 0): rigid rows with a face absorption, as `asphalt`; the fit target (n and z, or the asserted reflection) | |

## 7. Research requests

Format (handoff section 7): the column or class; the quantity and its coefficient type; the bands; whether closed
sources are acceptable; urgency and whether step 3 can proceed without it. Stephan forwards to the research manager.

| sent | request | status |
| --- | --- | --- |
| 2026-10-27 | `md/materials/research_request_2026-10-27.md`: request 0 (stable row ids), 1–13 by impact (wish-list section 6), three data errors (lines 1451, 520, 1819 of 2dad906); routes proposed by the research manager, approved by Stephan | with the research manager |

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
- **Research data:** generated files (never patch a copy; report a wrong row); PTB 63 Hz values equal to 125 Hz and
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

