# The acoustic work — manager handoff

This file is the manager's memory. A manager session reads it first, works from it, and updates it after every step
(state, roadmap, pending decisions). When a manager's context fills up, a new manager reads this file and continues;
nothing depends on one chat. Last updated: 2026-10-16, after commit 2c5be17 (bc24b0c; grazing round complete, waiting for Stephan's commit).

## 1. Roles

- **Stephan** (radio and acoustic engineer, maintains the library) decides every open point, checks results himself,
  and commits. He often takes the manager's recommendation; the manager still lays out the options.
- **The manager** writes briefs and kick-off prompts for worker sessions, reviews their results, prepares Stephan's
  decisions and the commit messages, and keeps this file and the roadmap current. It does not edit `src/` or tests and
  does not run experiments beyond a quick read-only check of a worker's claim. Its own edits: this file, briefs
  (`md/*_brief.md`), small document corrections Stephan asks for, `tools/acoustic`.
- **Workers** (one fresh session per item) follow a brief in phases, write `md/<item>_report.md`, stop after each
  phase, and never commit.

## 2. The workflow

1. **Brief.** The manager writes `md/<item>_brief.md` (structure: Why; Facts at the start; Hard rules; Phase 0 analysis
   in scratch; Phase 1 fixtures; Phase 2 source; Phase 3 documents; Traps) and a one-line kick-off prompt:
   `Read md/<item>_brief.md in full and follow it. Start with the baseline check and phase 0; stop after phase 0 with
   the report in md/<item>_report.md.`
2. **Phases with stops.** analysis → Stephan's decisions → fixtures first (they fail on today's library, nothing else
   moves) → source → documents. Each phase ends with a report and a stop.
3. **Review.** Stephan says a phase is done (or pastes the worker's summary). The manager reads the report from disk
   and the diff (`git diff --stat`, `git diff src/`), runs the checklist (section 3), and answers with: what checks out,
   what does not, the decisions in the format of section 2.1, and a paste-ready message for the worker.
4. **Commit.** After the last phase the manager writes the commit message (style below) and the `git add` line;
   Stephan commits and regenerates the three baselines (`md/wave.md` 0). The worker session can then be archived.
5. **Update this file**: state, roadmap, pending decisions.

**2.1 Decisions are presented as:** what Stephan is deciding; the options; trade-offs and implications; what is correct
(physics first, then the library's rules); a recommendation. One table or list per decision, a paste-ready message at
the end.

**2.2 Commit messages:** title "Acoustic: <what changed>", body in plain physical and code terms with the measured
numbers, the line count of `src/ray_mesh_interact.cpp`, the three counts, and the trailer
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (Stephan keeps it). **No internal labels or documents**
(Stephan, 2026-10-17): no item ids (M10, D2-b, A12, N4 …), no `md/` files, briefs, reports or their sections, no
"round 1"; they are transitory and will be deleted. Test case names, source and tool paths are fine.

## 3. The review checklist

Run on every worker phase; say which items were checked.

- **Baselines**: the worker started from the committed counts; a plain what-if copy reproduced them (only the seed
  line differs).
- **`[wave]` identical** (`test_wave.txt` line for line, timing lines excluded) and the stored twins
  (`tests/data/wave_twins.tsv`) unchanged and asserted. A change that moves a `[wave]` row stops the phase.
- **Every moved row accounted for**: in `[acoustic]` and the whole suite only the intended cases' rows; the counts add
  up (new assertions = the difference).
- **References independent**: fixtures write their reference from formulas in the test (`lw::stack`, the limp mass
  law, a column definition), never from library code; bounds 2 to 3 times the measured maxima.
- **No grid artifacts**: a peak or dip quoted as a frequency comes from a fine sweep (`lw::peak_f`, 1/96 octave), not
  from the nearest grid point. (A18's "160 Hz at 30°" was 148.6; "315 Hz at 45°" was 298.8.)
- **Run time**: `--durations yes` as CPU time or alternated runs with the load stated (do not wait for an idle
  machine: another coding session runs in parallel and races for it too); a new case well under 0.5 s; the whole
  suite about 59 s summed per case.
- **Resources** (Stephan, 2026-10-19): at most 12 threads for this work, compiler and tests alike
  (`cmake --build build_linux --parallel 12`, `OMP_NUM_THREADS=12` or fewer; numpy pools `OMP_NUM_THREADS=1` per
  worker, at most 12 workers); never two suites at once; ASAN, AFL and other heavy tools only when the task justifies
  them, never by default, on the resource-saving end. Every brief and prompt carries this.
- **Walkers**: collect a path in air only outside the structure (`lw::walk` does; `bf::walk` / `cf::walk` do not and
  can hide energy creation, `md/wave.md` 11); a followed series reports its dropped power; recursion needs
  `ulimit -s unlimited` or `lw::walk`.
- **Conditions on hit codes are complete**: compare against the `out_typeN` table in `src/ray_mesh_interact.cpp` (every
  code that starts inside a body: 1, 5, 7, 13, 21, 29, +32 for total reflection; shared codes 69 / 71 need the
  re-derived sheet-ness).
- **EM first**: a change in shared code is measured on EM types too; if a behavior is decided for EM, EM rows are
  asserted (A21 is the example: EM created R + T 42.19 before the kill).
- **Energy**: R + T ≤ 1 from a lossless side on every new configuration, both tracers, summed and followed.
- **Documents**: `python3 tools/acoustic/check_docs.py` passes (tables, source keys, case names, counts against the
  baselines, spelling); decisions in `md/acoustic.md` 10, open items in 6, candidate status in 4; American English and
  the style of the file (plain statements, numbers with units, test names in italics, no emphasis adverbs).
- **Scratch is not kept** (`make clean` removes `build_linux/scratch`): every number a decision rests on is stated in a
  report or a document; reusable instruments go to `tools/acoustic`.
- **One writer** for `md/acoustic.md` and `md/wave.md` at a time (the active worker in its document phase, or the
  manager), never two sessions at once.

## 4. Current state (2026-10-16)

- Last commit **bc24b0c** (M10: `m` a switch at 20 dB per decade; `coiA` removed). Before it: 2c5be17 (M6), 4035ac4
  (this file, `check_docs.py`), 238444d (M5, A21), 04b92bd (acoustic.md rewrite, M9), 4720a92 (round 1 built).
- `src/ray_mesh_interact.cpp` 5159 lines with the finite panel (uncommitted; 5116 at bc24b0c).
- Counts with the finite panel: `[wave]` 165258 / 180, `[acoustic]` 41523 / 58, whole suite 1316330 / 1145; baselines
  regenerated with 12 threads; `check_docs.py` passes.
- Documents: `md/acoustic.md` (the reference), `md/wave.md`, `md/acoustic_round1_report.md`,
  `md/m9_fixtures_report.md`, `md/m5_a21_report.md`, `md/m6_report.md`; briefs `md/m9_fixtures_brief.md`,
  `md/m5_a21_brief.md`, `md/m6_brief.md`, `md/m10_m15_brief.md`, report `md/m10_m15_report.md`; `md/grazing_brief.md` (approved, uncommitted).
- Tools: `tools/acoustic` (`sabine_to_random.py`, `band_probe.cpp` with the peak columns, `paris_cf.py` the closed-form
  Paris value of a locally reacting face, `check_docs.py`, and with M10: `txlib.py` / `txfit.py` / `tl_refs.json` the transmission formula copy
  and fit, `cases.py` moved rows per case; README).
- Literature review: `/home/stephan/accoustic_research/review.md`, data `data/material_properties.csv`, keys in
  `references/index.csv` (four keys the review cites are missing from its index; `check_docs.py` knows them).

## 5. Decided (pointers; the text is in `md/acoustic.md` 10)

P1–P9 (P6 reflect side first; P9 scattering out of scope, a separate project only on very strong evidence); N1–N7
(round 1: the plate rule R1h, M2, `res*` and `coi*` in `X₀`, the two-parameter mapping); A10, A11; M5 (built), M10 (built: `m` a switch), D2-b (`coiA` removed), M11 / M13 dropped, M15 proposal, M14 open (study),
0–90° fit rule, M6 (built:
locally reacting peaks, D1–D5 in `md/acoustic.md` 10; over-damped default, `tfQ < 0` under-damped), M7 dropped, M9 (the route for
unpartitioned built constructions: a wave-medium leaf glued on a box of near-air material on a mass-law wall); A13
(tiered rule for `tf` references), A14 (class 4 on Dupont's 50 mm glass wool), A15 (Miki / Komatsu), A16 (two-tier
tolerances), A17 (classes 8 panel absorber, 9 curtain, 10 MPP; carpet a row of class 4; no corner-trap class), A19
revised (measured `f₀`, else 60 empty / 50 filled; no framed factor); A21 (the kill after a glued exit, every type);

## 6. Pending decisions (proposed in `md/acoustic.md`, not yet decided)

- **A13 refinement**: a conversion whose fitted σ is off a measured σ by more than about 1.5 times is marked, not used.
- **A14**: the two Scrosati rows out until a polyester model and a framed-sample model exist.
- **A17′**: class 8 as the family of resonant absorbers (panel and membrane; perforated and slotted panels; MPP).
- **M8**: retire the one-parameter mapping for materials (documentation only).
- **M15**: the bending loss as a resistance; decide with the parameter session (where `coiQ` comes from).
- **M14**: a limiting angle; after the grazing-incidence study (roadmap item 1).
- **M16**: a relaxation term on `μ` for fabrics (curtains).

## 7. Open items without a decision (`md/acoustic.md` 6)

A8 (low priority), A9 (`html_docu/quadriga_lib_material_model.md`, last), A12 (closed for locally reacting faces by M6; a flat `tf`
keeps the hard wall's phase), A20 (closed for the peak; open for a wall's power-law `tf` behind a porous fill: the pad on
the fill's immittance against a wall impedance fixed as seen from air), A21 remainder
(the three-face tie is decided by mesh order; a full fix needs the intersector or `ray_progress` to report the third
face; how often the kill fires in a real EM scene with glued sheets is unmeasured).

## 8. Roadmap (next first)

Done: **M10–M15** (committed bc24b0c, 2026-10-17; `md/m10_m15_report.md`): M10 built, `coiA` removed (D2-b, with the
unused helper header), M11 / M13 dropped, M15 a proposal, M14 open (item 1), fit rule 0–90°. Left from it: regenerate
`html_docu/` (Stephan); the MATLAB `ray_mesh_interact` wrapper and test to be rewritten (Stephan, later; its test at
line 113 expects the field `|T|²`, the library returns `1 − R`); the two "1 = never" comments on the slab threshold
(`calc_diffraction_gain`'s description, `include/quadriga_tools.hpp` `thin_slab_threshold`) were not in that round's
scope and are still open, comment only.

1. **Grazing-incidence study (M14 from first principles)**, next. Brief `md/grazing_brief.md` approved 2026-10-17;
   phase 0 done (`md/grazing_report.md` 0), reviewed 2026-10-19 (the manager recomputed the transparency angle, the
   grazing share of a limp plate, Davy's σ(90°) and the equivalent angle `cos θ_l ≈ 1.53/√(kL)` for the nominal wall:
   all as reported). Finding: an infinite plate turns transparent past `cos θ = 1/X₀` because the air's radiation
   impedance `ρ₀c/cos θ` grows without bound; a finite plate's forced wave radiates as a beam that the wall plane cuts
   near grazing, σ saturates at `≈ 0.38·√(kL)`, and the transmitted power per incident intensity levels off. Recommended
   (d): Davy's σ in the mass term with a nominal 4.0 × 2.7 m size, no column (+12 lines; 2.97 → 2.69 dB on the 13; light
   panels at 0.93 / 0.95 of their mass; heavy walls' knobs unchanged: their job is above coincidence). Side findings: the
   `resS > 0` cavity spring at the true cosine (a locally reacting cavity fits the units and linings to 0.04–1.12 dB);
   heavy walls above `f_c`. **Decided (Stephan, 2026-10-19):** D1 (d) Davy's σ in
   the mass term; D2 a size column (manager's check, `scratch/size1.py`: of one-number sizes in Davy's form √area is
   best, ≤ 0.44 dB per angle and diffuse for aspect ratios to 3.3 : 1, 1.5 dB at 10 : 1; 4S/U with S = L² ≤ 0.5 dB and
   4.2 dB at 10 : 1; the diagonal 0.8–2.0 dB; a fixed nominal up to 7 dB on small panes): column `side`, √(panel area)
   in m, default 3.29 m (a 4.0 × 2.7 m wall), 0 = the infinite plate, `m > 0` only; D3 M14 closed as superseded; D4
   open (the `res*` cavity spring; manager's explainer given 2026-10-19, recommends a later small item, not this round: it acts only on the lumped double material; a meshed double
   leaf on an air box has its cavity from the wave physics and is not affected; note the meshed leaf is a wave medium
   and keeps the infinite plate's grazing transparency under (d)). Phase 1 prompt given (2026-10-19, revised: one `side`
   per bonded run; a measurement of 4|16|4 as two mass-law sheets on a meshed air gap for a rule of use (before round 1
   it read −105 dB at 1 kHz against 35 dB measured: no mass-air-mass resonance, the two losses add); the two "1 = never"
   slab-threshold comments, dropped by mistake from the M10 round, carried into this round's phase 2). Phase 1 done and reviewed 2026-10-19
   (`md/grazing_report.md` 1; the manager rebuilt and reran: `[acoustic]` 58 cases, 4 failing with 691 assertions, plus
   the OBJ header = 693 as reported; `[wave]` identical). Fixtures: two new *Finite plate* cases, *Mass law* and *Plate
   rule* restated at the default side, 15 cases restated with `side` 0 (their statements are the infinite plate's).
   Refit of the 13 under (d): 2.70 dB worst at lab side and at 3.29 m. 4|16|4 as two meshed mass-law sheets on a box:
   the unbridged double leaf above resonance (70 dB at 1 kHz against 36 measured), no mass-air-mass dip: rule of use.
   Side finding (unknown cause): a traced leg far from the origin or near grazing loses precision (a 4 mm pane crossed
   at y = −100 m passes 0.2 dB more at normal incidence; today's library at 89.9° through 152 mm misses its own rule by
   up to −3.4 dB at 2 kHz). Open for Stephan: `same_as` compares `side` (where `m > 0`), the bonded-run rule of use,
   the precision finding as a separate investigation (EM check included). Decided 2026-10-19 as recommended (same_as compares `side`
   where `m > 0`; one `side` per bonded run as a rule of use; the precision finding a separate investigation). Phase 2
   done and reviewed 2026-10-19: `src/ray_mesh_interact.cpp` 5159 lines (+56 / −14), `side` in the OBJ writer, both
   readers and the band probe, the two slab-threshold comments fixed; the manager rebuilt with 12 threads and reran
   `[wave]` (identical, twins unmodified) and `[acoustic]` (41523 / 58); whole suite 1316330 / 1145 per the report;
   the 13 on the built library within 0.0042 dB sheet against solid, band bitwise, R + T ≤ 1; no measurable run-time
   cost (CPU time, alternated). Phase 3 done and reviewed 2026-10-19 (documents, `md/wave.md` 0 with the
   12-thread commands, `format_materials.md`, `tools/acoustic/gzlib.py` with its checks, txlib / txfit as built;
   baselines regenerated; `check_docs.py` passes); commit message given, waiting for Stephan's commit (phase 0 is the study and the explanation; phases 1–3 only if a model is chosen). Why: the infinite plate passes nearly everything at grazing (the air's normal impedance `ρ₀c/cos θ` grows
   without bound, so the plate's mass stops mattering; equivalently its forced wave radiates with efficiency
   `1/cos θ`); on round-1 columns 78°–90° carry 4.3 % of a diffuse field but up to 15.3 dB of an octave's transmission;
   a finite wall radiates its forced wave with a bounded efficiency. Stephan asked for this earlier and never got a
   clear answer; he wants the physics understood before choosing a model. Needed: (1) the derivation in plain words;
   (2) per-angle `τ(θ)` of the infinite plate against the finite plate of round 1's `a1/p3/fp.py` (FIN: forced wave
   windowed, Rayleigh integral; MOD: modal sum; σ cached for 1.20 × 2.40, 1.23 × 1.48, 4.37 × 2.84 m) for light and
   heavy panels and several sizes, where and how they part; (3) Davy_2009_ForcedRadiationEfficiency (in the review's
   references), Northwood 78° / 85°, Wareing's size effect; (4) candidates: none, a fixed limit angle, a
   size-dependent limit angle, Davy's σ with a size (a column or a nominal size; the library cannot know a panel's size
   from a triangle hit); (5) each judged per angle against FIN / MOD, on the 13 constructions at 0–90° (D4-B), and by
   code. Read-only until Stephan decides; the fitter `tools/acoustic/txlib.py` / `txfit.py` (with `tl_refs.json`).
   Must precede the parameter session (the fitted knobs depend on it). Also to check there: whether a finite-size form
   frees the heavy walls' knobs, which under M10 at 0–90° sit at their bounds (`md/m10_m15_report.md` 1.0).
2. **Traced-leg precision** (investigation, from the grazing round; unknown cause): a 4 mm pane crossed at y = −100 m
   passes 0.2 dB more at normal incidence than near the origin; 152 mm concrete crossed at 89.9° (87 m inside) misses
   its own mass-law rule by up to −3.4 dB at 2 kHz. Where precision is lost (mesh storage, intersection, the thickness
   reconstructed from leg length and cosine) and whether EM path lengths are affected. Read-only first.
3. **The `res*` cavity spring of a lumped double material** (D4 of the grazing round, open): locally reacting for
   partitioned cavities (linings on battens or dabs, stud walls; fits 1.1–2.7 → 0.04–1.12 dB), double glazing by a short
   study. Small; transmission only.
4. **The parameter session**: the material table from the review's data with the A13 / A14 / A19 rules
   (`tools/acoustic/sabine_to_random.py`), the acceptance classes of `md/acoustic.md` 9.
5. **A9**: the material-model page, after the machine is settled.

Decisions in section 6 can be taken whenever Stephan wants; none blocks item 1.

## 9. Traps (collected)

- `make clean` removes `build_linux/scratch`; rigs named in documents may be gone.
- Run a relinked test binary from the repository root; never two suites at once; `timeout` on probes; check the load
  (`~/.claude/CLAUDE.md`), single-threaded probes under load.
- A probe that includes a test file runs every case of it unless given a test spec; a case name with a comma needs `\,`.
- `pkill -f <pattern>` from a shell whose command line contains the pattern kills that shell; kill by pid.
- A glued sheet lies in front of the body it is glued on; an isolated sheet about its mesh plane.
- A material identical to air is air: a box of air needs `μ` 1.000001.
- A rigid backing (`r = 1`) is not the reference for a two-parameter wall: use its half-space (or the pad on it).
- `ray_progress` allows 255 interactions: a meshed gap behind a leaf is out of reach; with the threshold at 1.0 some
  geometries run away (A21 before the kill: 12.7 GB, no end).
- A followed series of a lossy body between strong reflectors diverges (round trip above one): only summed rows count.
- `python3 -I` puts no script directory on `sys.path`.
- Pasted worker summaries are shortened; read the report and the diff.
