# Kick-off: step 3, the acoustic material tables

You are the manager of step 3 of Stephan's acoustic project. Read this file in full, then do phase 0 and stop.

## The project and where step 3 stands

The project has six steps: (1) literature research, (2) the interface engine, (3) the material tables, (4) a
reference simulation in a residential listening room, (5) measurements, (6) a journal paper (an acoustics journal;
the whole pipeline, with the engine as its methods core).
- Step 1 is the research library `/sjc/accoustic_research`: a separate repository, local only and read-only for this
  project.
- Step 2 is closed (last commits 9b298cc, 8d89b84, 2026-10-22).
- Step 3 builds the acoustic material table the later steps use.

## Read first

- Your memory index loads by itself. Read in full: `parameter-project-inheritance`, `user-stephan`,
  `acoustic-engine-domain-decisions`, `probe-and-test-techniques`, `measure-dont-compute-by-hand`,
  `md-public-no-private-details`, `resource-limit-12-threads`, `usable-not-perfect-no-new-items`,
  `commit-messages-no-internal-labels`.
- `/sjc/accoustic_research/manager/handoff_parameter_session.md`, in full. It covers:
  - where the data are and how to read them (generated files; coefficient types; copies at 63 Hz and 8 kHz);
  - what to use per column (section 2) and per class (3);
  - the coefficient types to watch (4);
  - the statements on the page and in the notes that the library contradicts (5);
  - the gaps (6);
  - how to request research (7).

  Also its `LESSONS.md` (how Stephan works with a manager) and `review.md` sections as needed.
- `md/acoustic_engine_notes.md`: sections 5 (decisions), 8 (validation, the reference classes in 8.4), 9 (limits,
  open conventions, pending decisions in 9.3) and 10 (how the engine was developed). These notes are the record of
  step 2.
- `html_docu/quadriga_lib_material_model.md`:
  - sections 4 (columns), 5 (formulas), 6 (faces, the plate rule) and 10 (acoustic use; 10.4 leaves on a box, 10.5
    deriving columns from data);
  - 11 (worked examples; 11.4 the 13 preliminary starting rows) and 12–13 (rules of use, limits);
  - 16 (how a change is checked).
- `tools/acoustic/README.md` and the tools: `txlib.py` is a formula copy of the library; `txfit.py`, `gzlib.py`
  (independent plate models), `paris_cf.py`, `sabine_to_random.py`, `band_probe.cpp`, `rowdiff.py`, `cases.py`,
  `check_docs.py`.
- `tests/catch2_tests/ray_test_helpers.hpp` (the `ub`, `bf`, `cf`, `rp` helpers) and
  `tests/catch2_tests/test_acoustic_leaf.cpp` (the leaf on a box, a fluid-layer transfer matrix `lw::stack`, an
  outside-only walker `lw::walk`).

## Decisions taken before this kick-off (Stephan, 2026-10-22)

- **Deliverable:** a shipped acoustic material table, a CSV in `data/` (the name is proposed in phase 0), in the
  format of `tools/data_formats_md/format_materials.md`.
  - Every row is asserted by a test that reads the CSV, so the table and its tests cannot drift.
  - At the end the table replaces the preliminary rows of page 11.4.
  - The built-in EM table of `obj_file_read` is not touched.
- **Material scope:** one or more rows per acceptance class (the ten classes of notes 8.4 and the resonant family of
  notes 9.3), plus the materials of the listening room.
- **Phases:**
  - **0 Setup:** the manager's files, the agent template and the scope. Stop.
  - **1 The test rig:** one Catch2 test file with helpers for three construction types:
    - one body in air, as a sheet and as a traced solid;
    - layers on a rigid backing (for example an absorber on concrete, a lining);
    - a leaf on a box of air in front of a wall (page 10.4: panel absorbers, double leaves, curtains, microperforated
      panels).

    It has two outputs per octave: the random-incidence absorption of the room side (Paris integral) and the
    transmission at the laboratory side. It reuses the existing helpers and tools. Stop.
  - **2 The material list:** proposed from the research results and gaps, with reasons. The listening room's
    inventory is worked out in this phase as an iterative exchange with Stephan; the research data, its gaps and
    stated assumptions flow into it. Stop.
  - **3 Material by material:** one test case per material. Each case has an oracle in two tiers, against what the
    library computes through the rig. Continue until every material of the list and its meaningful combinations
    are in.
  - **4 Review, cleanup and lessons:**
    - `md/acoustic_material_notes.md`, the lab notebook of step 3: public and without private details, in the style
      of the engine notes;
    - page 11.4 replaced by the final table;
    - the statements the research contradicts (handoff section 5) corrected on the page and in the engine notes;
    - the tools README updated;
    - the transitory `md/` files of step 3 deleted.
- **Unit of work in phase 3:** one material per worker agent, which the manager spawns itself (the Agent tool),
  strictly serially in this checkout. The manager checks the agent's result and reports to Stephan; Stephan reviews
  and commits each material.
- **The oracle has two tiers:**
  1. An independent physical model (plate theory, a transfer matrix, Miki / Komatsu / JCA, the leaf on a box) checks
     that the library computes the columns right. It is written in the test, never taken from library code.
     `txlib.py` is a copy of the library and is not an independent reference.
  2. The measured reference, after its coefficient-type conversion (the tiered rule for absorption references on
     page 10.5), checks that the columns represent the material, per octave, within the guideline tolerance (about
     ±3 dB in transmission and ±0.05 in absorption). Bands without a measurement take the model spread. Every
     deviation is recorded.
- **Tests:** one file, `tests/catch2_tests/test_acoustic_materials.cpp`, tagged `[acoustic][materials]`, one
  `TEST_CASE` per material. About 0.3 s per case at most: a coarse angle grid, validated once against the fine grid in
  `tools/acoustic`.
- **Open rules** are decided at the stop of the first material that needs them. That covers:
  - the pending engine decisions of notes 9.3;
  - the reference row per class;
  - the Sabine-to-random conversion of panels and microperforated panels;
  - the form of the leaf's loss ratio `rr`;
  - the damping branch of a peaked face;
  - the role of `alpha` on a porous layer.

  Order the materials so that the rules settle early.
- **Engine changes:** the engine stays closed. If a material cannot be expressed, the miss is recorded and sized. An
  engine change happens only after Stephan's OK, in a separate engine manager session that he installs, with the
  engine workflow (`[wave]` untouched). The materials confirmed by then are its regression tests.
- **Research requests:** a material whose reference is missing waits for the research job, and the manager continues
  with another material. A request follows handoff section 7: the column or class, the quantity and its coefficient
  type, the bands, whether closed sources are acceptable, and the urgency. Stephan forwards it to the research
  manager.
- **Files:** the manager's files live in `md/`, tracked and public, so they carry no private details. Briefs and
  reports are transitory and are deleted in phase 4. The lasting record is `md/acoustic_material_notes.md`.

## Phase 0 — setup (stop)

1. **Baseline:**
   - build with `--parallel 12`;
   - `QD_DIAG_RAY=1 OMP_NUM_THREADS=12 ./build_linux/test_bin "[wave]" > test_wave.txt` (163474 / 178);
   - the same for `"[acoustic]"` into `test_acoustic.txt` (42741 / 59);
   - `QD_DIAG=1 … > test.txt` for the whole suite (1344398 / 1264);
   - `python3 tools/acoustic/check_docs.py`.

   Only if these files are missing or stale: they are untracked, and the existing ones may be current.
2. **`md/materials_manager.md`**, the manager's memory, kept current after every step:
   - roles (Stephan; the manager; worker agents: the rig agent, analysis agents, material agents);
   - the decisions above;
   - the workflow per phase and per material (brief → agent → the manager's checks → report to Stephan → his
     review and commit);
   - the review checklist: the engine's checklist adapted to materials:
     - `[wave]` and `[acoustic]` rows unchanged;
     - the new case within its time;
     - tier-1 references independent of the library;
     - tier-2 coefficient types right;
     - peaks from fine sweeps, not grid points;
     - the CSV row equal to what the test asserts;
     - `check_docs.py`;
     - the resources;
   - the material list as a table (class, material, reference keys, status: proposed / waiting for research /
     in work / confirmed / recorded miss), filled in phase 2;
   - the open rules and how each was decided;
   - the research requests sent and pending;
   - the engine items raised;
   - the current state, the roadmap, the traps and the lessons.
3. **`md/materials_agent_template.md`**: the brief template and the one-line kick-off prompt for an agent, with
   variants for the rig agent (phase 1), the analysis agents (read-only, phase 2 and for rule questions) and the
   material agents (phase 3). A material brief states:
   - the material and its class;
   - the reference rows (keys, coefficient types, bands);
   - the rules that apply;
   - the tier-1 model;
   - the steps (data → columns by the decided rule → tier 1 → tier 2 → the test case → the CSV row);
   - the hard rules: no change under `src/`, `include/` or `api_*`; no edit of the research repository; ≤ 12 threads;
     one suite at a time; measure, don't compute by hand; no private details;
   - the report format: a short return to the manager, the substance in `md/materials/<id>_report.md`.
4. **Scope:** propose the CSV's name and location, the test file's skeleton (names only), and how a combination
   ("meaningful combinations") is defined and named in the CSV.
5. **Stop.** Report the files created, the baseline counts, the questions phase 1 needs from Stephan (ask them with
   the question tool, recommendation first, explanation before the question), and a commit message for the phase-0
   files (no internal labels; Co-Authored-By trailer).

## Hard rules for the whole step

- No change under `src/`, `include/` or `api_*` (engine changes go to a separate engine manager, see above). Never
  edit `/sjc/accoustic_research`. Never commit; Stephan commits.
- Agents run one at a time; each gets its brief file and returns a short report. The manager verifies every claim a
  decision rests on: rebuild, rerun, read the CSV and the test.
- Resources: at most 12 threads, compiler and tests alike; one suite at a time; `OMP_NUM_THREADS=1` per numpy worker;
  no heavy tools.
- Tracked documents carry no private details; Stephan's answers stay in untracked working files when they are
  personal.
- Decisions go to Stephan in the form: what is decided; options; trade-offs; what is correct (physics first);
  recommendation; a paste-ready message.
- No new items beyond the material list and the rules; side findings are recorded with their size and dropped.

## Facts at the start

- `data/` holds `materials_rf.csv` (EM) and `config/`; the old acoustic table was removed in 8d89b84.
- Leftovers from step 2:
  - `tools/acoustic/band_probe.cpp` has no `attB` column, and every fitted row uses one;
  - page section 9 says `irr_glass` follows TR 38.901, though only its `att` does;
  - if the meaning of `coiQ` changes, the page's 11.1, 11.2 and 11.4 are re-measured;
  - the fixes manager has open documentation points on the material reader (`obj_file_read`) that do not block step 3.
- The research handoff uses the old rule labels (A13, A14, A15, A19, P7, M6) and maps them to notes and page
  sections in its section 0. Use them only while working; public documents and commit messages name the rule in
  words.
