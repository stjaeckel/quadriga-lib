# Brief template and kick-off prompts for the agents of step 3 (material tables)

The manager copies the matching variant to `md/materials/<id>_brief.md`, fills it, and starts one agent with the
kick-off prompt (Agent tool, one agent at a time in this checkout). Briefs and reports are untracked working files and
are deleted in phase 4. Ids: `rig` (phase 1), `a<nn>_<topic>` (analysis), `m<nn>_<row name>` (material), for example
`m03_glass_4mm`.

## Kick-off prompt (one line)

```
Read md/materials/<id>_brief.md in full and follow it: do phase <n> and return a short report (at most 300 words); the substance goes in md/materials/<id>_report.md.
```

For the next phase of the same agent (SendMessage): `Phase <n> approved with these decisions: <...>. Do phase <n+1> as
in the brief and return a short report.`

## Common part of every brief

```
# Brief <id>: <title>

## Why
<one paragraph: what this item settles for the material table and for which later step>

## Read first
- md/materials_manager.md (sections 2, 4, 6, 12)
- <page / notes sections, research files and rows, test helpers that this item needs>

## Decisions that apply
<the kick-off decisions and the open rules already decided that this item meets, in words>

## Facts at the start
- HEAD <hash>; baselines [wave] 163474 / 178, [acoustic] <n> / <n>, whole suite <n> / <n> (generic build_linux)
- <what exists: helpers, rows already in the CSV, earlier reports>

## Hard rules
- No change under src/, include/ or api_*. Never edit /sjc/accoustic_research (read only; report a wrong row instead).
  Never commit.
- At most 12 threads, compiler and tests (cmake --build build_linux --parallel 12; OMP_NUM_THREADS=12 or fewer);
  numpy workers with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, at most 12; one suite at a time;
  no ASAN, AFL or other heavy tools. Check /proc/loadavg before heavy work and use fewer threads when it is high.
- Measure, don't compute by hand: every number in the report comes from a run, a printed test row or a recorded
  commit. If a command cannot run, say so and stop.
- Tier-1 references are independent physics written in the test, never library code or tools/acoustic/txlib.py.
- No private details in tracked files (no rates, hours, money, personal circumstances; "the listening room").
- No new items: side findings get a measured size and one line in the report, then they are dropped.
- Scratch files in the session's scratchpad or build_linux/scratch; every number a decision rests on goes into the
  report.

## Phases
<per variant, below; each ends with a stop: write the report, return the short report, wait>

## Traps
- md/materials_manager.md section 12, plus <item-specific traps>

## Report format
md/materials/<id>_report.md:
1. Result in three lines.
2. What was done, per phase, with the measured numbers (tables per octave where they apply).
3. Checklist items run (md/materials_manager.md section 4) and their outcome.
4. Decisions needed: what is decided; options; trade-offs; what is correct (physics first); recommendation.
5. Side findings: one line each with its size.
6. Files changed (git status --short) and the commands to reproduce the key numbers.
The short return to the manager: items 1, 4 and 6 in at most 300 words.
```

## Variant: the rig agent (phase 1)

Item-specific sections:

```
## The task
One Catch2 file, tests/catch2_tests/test_acoustic_materials.cpp, tags [acoustic][materials], with the rig
(namespace mt) for three construction types:
1. one body in air, as a sheet and as a traced solid;
2. layers on a rigid backing (an absorber on concrete, a lining);
3. a leaf on a box of air in front of a wall (page 10.4).
Two outputs per octave (31.5 Hz to 8 kHz): the random-incidence absorption of the room side (Paris integral) and the
transmission at the laboratory side (0°–90°, the row's side). The table is read from data/<csv name> through
obj_file_read; the rig takes rows by name. Reuse ray_test_helpers.hpp (ub, bf, cf, rp) and tools/acoustic (gzlib.py,
paris_cf.py, sabine_to_random.py); copy the parts of the leaf file's lw helpers the rig needs (lw::stack, lw::walk,
lw::peak_f) into the new file, leaving test_acoustic_leaf.cpp and the header unchanged.

## Phases
1. Plan (stop): which library path each construction type runs through, the coarse angle and frequency grid with its
   cost per case, the tier-1 references the rig provides and their sources, what moves in existing files.
2. The rig and its self-checks (stop): the coarse grid validated once against the fine grid (192 angles × 6
   frequencies per third, or finer where coincidence needs it) in tools/acoustic; self-check cases on known
   constructions (the page's worked examples 11.1–11.3 and the leaf fixtures) within their recorded numbers; the
   whole-suite rows unchanged apart from the new cases; the time per case.
```

## Variant: analysis agents (read-only; phase 2 and rule questions)

```
## The task
<the question; the data to read; the quantity and its coefficient type; what the answer decides>

## Rules for this agent
Read only: the agent writes nothing but its report (and scratch files). It may run tools/acoustic scripts and scratch
programs linked against build_linux/libquadriga.a; it does not edit tracked files.

## Phases
1. The answer (stop): the facts with sizes, the options with their measured consequences, a recommendation.
```

Phase-2 analysis topics (one agent each, as needed): the reference row per class (types, copies, spread); the
listening room's inventory against the research data and its gaps; the order of materials that settles the open rules
early.

## Variant: material agents (phase 3)

```
## The material
- Row name: <name>; class: <class>; build: <thickness, density, layers, mounting, cavity>.
- Reference rows (research data/material_properties.csv): <key, property, coefficient type, bands, location>.
- Rules that apply: <the tiered rule for absorption references / Miki or Komatsu / the membrane constant / ...>,
  as decided in md/materials_manager.md section 6.
- Open rules this material meets: <none / which; the agent measures the options and stops>.
- Tier-1 model: <plate theory / the transfer matrix / Miki, Komatsu, JCA / the leaf on a box>, written in the test.
- Tier-2 tolerance: about ±3 dB in transmission, ±0.05 in absorption; bands without a measurement take the model
  spread <values>.

## Phases
1. Data and columns (stop if an open rule is met): the reference rows read and their types checked; the columns
   derived by the decided rule (fits bounded and densely sampled, selected on the evaluation grid).
2. Tier 1 and tier 2: the library through the rig against the independent model; the columns against the converted
   measurement per octave; every deviation recorded with its size.
3. The test case and the CSV row (stop): TEST_CASE("Material - <name>") reading the row from the CSV; the case within
   about 0.3 s; [wave] and [acoustic] rows of the existing cases unchanged; python3 tools/acoustic/check_docs.py.
```
