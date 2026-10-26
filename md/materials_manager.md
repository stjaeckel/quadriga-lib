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
without Stephan's OK. Each item becomes an issue in `md/acoustic_engine_stage2b_issues.md` (issues 1–3 there: porous
layer on a wall, band decision near grazing, two traced wave media back to back).

| raised | material | miss and size | status |
| --- | --- | --- | --- |
| 2026-10-23 | glass wool 50 mm (page 11.3 columns) as a sheet on cast concrete 152 mm (page 11.4 row) | transmission at normal incidence: the library adds 8.04 / 6.71 / 5.38 / 4.09 / 3.42 / 4.78 / 7.32 / 9.32 / 13.02 dB to the wall (31.5 Hz–8 kHz); an independent transfer matrix (the same fluid layer on a heavy limp plate, scratchpad `tmm/wool_on_plate.py`) adds −0.02 / −0.07 / −0.18 / −0.28 / 0.16 / 2.53 / 5.98 / 8.79 / 13.24 dB. Over-insulation up to 8 dB at 31.5 Hz, 5.6 dB at 125 Hz, 2.3 dB at 1 kHz. Reflection is fine (6.3e-3). Random incidence not yet measured. | issue 1 in `md/acoustic_engine_stage2b_issues.md` |

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
  - **Phase 2 result:** draft 2 of both documents, for Stephan to commit. Next: the research requests (section 6 of
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

1. Phase 1: the rig (brief from the template's rig variant after Stephan's answers).
2. Phase 2: the material list (analysis agents; the listening room's inventory with Stephan).
3. Phase 3: materials in the order that settles the open rules early. Order for the first materials (Stephan): concrete (rig check), wood alone, carpet alone,
   carpet on wood with diagnostics, judged and then fixed as a regression case; then the next material; common
   combinations before the scene models of the next step.
4. Phase 4: notes, page 11.4, corrected statements, README, cleanup.

Leftovers of step 2 to settle where they fall: `tools/acoustic/band_probe.cpp` has no `attB` column (every fitted row
uses one); page section 9 says `irr_glass` follows TR 38.901, though only its `att` does; if `coiQ` changes meaning,
page 11.1, 11.2 and 11.4 are re-measured.

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
- **Fits:** bound and densely sample; select on the evaluation grid; the mass level bound must reach about 3200 for a
  423 kg/m² wall; clip τ at one per frequency, not per band; coincidence needs about 4000 angles.

## 13. Lessons

- (phase 0) `build_linux` can be absent at the start of a session; reconfigure it as the generic build before
  regenerating baselines.
- (phase 1) A plan phase with scratch probes and a manager rerun of the probe's key commands settles a rig's design
  before any tracked file is written; light-panel coincidence decides the grid, not absorption.
