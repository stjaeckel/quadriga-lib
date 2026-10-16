# The acoustic work — manager handoff

This file is the manager's memory. A manager session reads it first, works from it, and updates it after every step
(state, roadmap, pending decisions). When a manager's context fills up, a new manager reads this file and continues;
nothing depends on one chat. Last updated: 2026-10-16, after commit 238444d.

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

**2.2 Commit messages:** title "Acoustic <item>: <what>", body by item with the measured numbers, the line count of
`src/ray_mesh_interact.cpp`, the three counts, the report file, and the trailer
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (Stephan keeps it).

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
- **Run time**: `--durations yes` on an idle machine (load before and after); a new case well under 0.5 s; the whole
  suite about 59 s summed per case.
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

- Last commit **238444d** (M5 and A21). Before it: 04b92bd (acoustic.md rewrite, M9), 4720a92 (round 1 built).
- `src/ray_mesh_interact.cpp` 5089 lines.
- Counts: `[wave]` 165110 / 179, `[acoustic]` 36813 / 52, whole suite 1311472 / 1138. **The `test*.txt` baselines in
  the repository root still hold the 04b92bd counts: regenerate them (`md/wave.md` 0) before the next worker starts**;
  `check_docs.py` reports the difference until then.
- Documents: `md/acoustic.md` (the reference), `md/wave.md`, `md/acoustic_round1_report.md`,
  `md/m9_fixtures_report.md`, `md/m5_a21_report.md`; briefs `md/m9_fixtures_brief.md`, `md/m5_a21_brief.md`.
- Tools: `tools/acoustic` (`sabine_to_random.py`, `band_probe.cpp`, `check_docs.py`, README).
- Literature review: `/home/stephan/accoustic_research/review.md`, data `data/material_properties.csv`, keys in
  `references/index.csv` (four keys the review cites are missing from its index; `check_docs.py` knows them).

## 5. Decided (pointers; the text is in `md/acoustic.md` 10)

P1–P9 (P6 reflect side first; P9 scattering out of scope, a separate project only on very strong evidence); N1–N7
(round 1: the plate rule R1h, M2, `res*` and `coi*` in `X₀`, the two-parameter mapping); A10, A11; M5 (built), M6 (lower
priority: locally reacting peaks only; over-damped default, `tfQ < 0` under-damped), M7 dropped, M9 (the route for
unpartitioned built constructions: a wave-medium leaf glued on a box of near-air material on a mass-law wall); A13
(tiered rule for `tf` references), A14 (class 4 on Dupont's 50 mm glass wool), A15 (Miki / Komatsu), A16 (two-tier
tolerances), A17 (classes 8 panel absorber, 9 curtain, 10 MPP; carpet a row of class 4; no corner-trap class), A19
revised (measured `f₀`, else 60 empty / 50 filled; no framed factor); A21 (the kill after a glued exit, every type).

## 6. Pending decisions (proposed in `md/acoustic.md`, not yet decided)

- **A13 refinement**: a conversion whose fitted σ is off a measured σ by more than about 1.5 times is marked, not used.
- **A14**: the two Scrosati rows out until a polyester model and a framed-sample model exist.
- **A17′**: class 8 as the family of resonant absorbers (panel and membrane; perforated and slotted panels; MPP).
- **M8**: retire the one-parameter mapping for materials (documentation only).
- **M10–M15** (transmission side): to examine on the 13 round-1 fits; M12 (`coiA` ignored where `m > 0`) is a few
  lines and could ride with another source change.
- **M16**: a relaxation term on `μ` for fabrics (curtains).

## 7. Open items without a decision (`md/acoustic.md` 6)

A8 (low priority), A9 (`html_docu/quadriga_lib_material_model.md`, last), A12 (the pad's phase: M6), A20 (a wall's `tf`
behind a porous fill: the pad on the fill's immittance against a wall impedance fixed as seen from air), A21 remainder
(the three-face tie is decided by mesh order; a full fix needs the intersector or `ray_progress` to report the third
face; how often the kill fires in a real EM scene with glued sheets is unmeasured).

## 8. Roadmap (next first)

1. **M6** — the room-side face as a series impedance for locally reacting peaks (phase and angle; A12). Brief to write.
   Inputs: `md/acoustic.md` 3.2, 3.8, 4 (M6 with the damping-branch rule), A12; the cosine in air at the face; the
   pad (`Material::closed_face`, `tf_face`). Fixture idea: the phase and `|r|` of a peaked face against the series
   impedance at 0° to 85°, sheet and traced solid; the random-incidence value of the A10 rows.
2. **Transmission candidates M10–M15**, judged on the 13 round-1 fits (report 3.6), with M12 alongside.
3. **The parameter session**: the material table from the review's data with the A13 / A14 / A19 rules
   (`tools/acoustic/sabine_to_random.py`), the acceptance classes of `md/acoustic.md` 9.
4. **A9**: the material-model page, after the machine is settled.

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
