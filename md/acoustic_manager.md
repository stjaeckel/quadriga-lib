# The acoustic work — manager handoff

This file is the manager's memory. A manager session reads it first, works from it, and updates it after every step
(state, roadmap, pending decisions). When a manager's context fills up, a new manager reads this file and continues;
nothing depends on one chat. Last updated: 2026-10-16, after commit 2c5be17 (M6; M10 / D2-b built, phase 3 reviewed, waiting for Stephan's commit).

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

- Last commit **2c5be17** (M6). Before it: 4035ac4 (this file, `check_docs.py`), 238444d (M5, A21), 04b92bd
  (acoustic.md rewrite, M9), 4720a92 (round 1 built).
- `src/ray_mesh_interact.cpp` 5116 lines with M10 / D2-b (uncommitted; 5132 at 2c5be17).
- Counts with M10 / D2-b: `[wave]` 165258 / 180, `[acoustic]` 40050 / 56, whole suite 1314857 / 1143; the `test*.txt`
  baselines were regenerated by the worker on the built library; `check_docs.py` passes.
- Documents: `md/acoustic.md` (the reference), `md/wave.md`, `md/acoustic_round1_report.md`,
  `md/m9_fixtures_report.md`, `md/m5_a21_report.md`, `md/m6_report.md`; briefs `md/m9_fixtures_brief.md`,
  `md/m5_a21_brief.md`, `md/m6_brief.md`, `md/m10_m15_brief.md`, report `md/m10_m15_report.md` (both uncommitted).
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
- **M14**: a limiting angle; after the grazing-incidence study (roadmap item 2).
- **M16**: a relaxation term on `μ` for fabrics (curtains).

## 7. Open items without a decision (`md/acoustic.md` 6)

A8 (low priority), A9 (`html_docu/quadriga_lib_material_model.md`, last), A12 (closed for locally reacting faces by M6; a flat `tf`
keeps the hard wall's phase), A20 (closed for the peak; open for a wall's power-law `tf` behind a porous fill: the pad on
the fill's immittance against a wall impedance fixed as seen from air), A21 remainder
(the three-face tie is decided by mesh order; a full fix needs the intersector or `ray_progress` to report the third
face; how often the kill fires in a real EM scene with glued sheets is unmeasured).

## 8. Roadmap (next first)

1. **M10–M15, the transmission candidates**, judged on the 13 round-1 fits (report 3.6), M12 alongside. Brief
   `md/m10_m15_brief.md` approved 2026-10-16; phase 0 done (`md/m10_m15_report.md` 0), reviewed 2026-10-17.
   Found: round 1's fits were not converged (today's knobs 1.23 dB, not 2.51); M10 and M11 each keep the 13 within the
   guideline, together not (furred lining 3.33 dB); M13 / M15 change nothing (≤ 0.08 dB); and the fit convention: the
   13 were fitted on the 0–78° field average, a room averages 0–90°, where the same columns pass up to 15.3 dB more
   (limp mass law alone 3 to 5 dB, checked). Manager's recommendation: M10 adopt, M11 drop, M12 adopt, M13 drop, fit
   on 0–90° (a rule in `md/acoustic.md` 9), M14 at 78° adopt (0–90° worst 2.97 → 2.80 under M10), M15 proposal,
   `txlib.py` / `txfit.py` / `cases.py` to `tools/acoustic`. **Decided (Stephan, 2026-10-17):** M10 adopted (D1-A),
   M11 dropped, M13 dropped, fit convention 0–90° (D4-B). D2-b (2026-10-17): `coiA` removed altogether in this round (no physical
   reason to keep it: superseded by `coiF` / `coiQ` in `X₀`, used by no material file, an angle-free Lorentzian in dB
   is no physical element); the `[wave]` cases that test it are restated (a decided wave change); the unused
   `src/quadriga_lib_material_helpers.hpp` (its own `coiA` copy, section 8 side finding) proposed for removal with it.
   D6: M15 stays a proposal. Phase 1 done and reviewed 2026-10-17 (`md/m10_m15_report.md` 1): M10 at 0–90° 2.97 dB
   worst (gypsum, model reference), every measured construction within ±3 dB (glass 6 mm 2.41, triple 2.69); the heavy
   walls and linings at the knob bounds (`att` 12 dB on three, the furred lining's `coiF` at 30 Hz, level 0.02–0.84 of
   their mass). 13 cases fail on today's library (886 assertions), all pass on the what-if; `[wave]` moves only in
   the three decided coiA cases plus the new guard; twins untouched. Found while testing: the MATLAB test
   `test_ray_mesh_interact.m:113` expects the field `|T|²` of an EM type-2 crossing (0.69639); the library returns the
   flux transmittance `1 − R` (0.98485, checked by the manager): a stale test, not a library fault. Stephan
   (2026-10-17): left alone; the MATLAB `ray_mesh_interact` wrapper and its test will be rewritten later. Phase 2 done and reviewed 2026-10-17
   (`md/m10_m15_report.md` 2): `src/ray_mesh_interact.cpp` 5116 lines (−16), `quadriga_lib_material_helpers.hpp`
   deleted, `coiA` out of the OBJ writer and both readers; the manager rebuilt and reran `[wave]` (165258 / 180, output
   identical to the 2c5be17 baseline apart from the count line; twins unmodified) and `[acoustic]` (40050 / 56); whole
   suite 1314857 / 1143 and Python 762 passed per the report. Phase 3 done and reviewed 2026-10-17; commit message
   given; after the commit regenerate `html_docu/` (still describes `coiA` and the slope). D5 (M14) out of this round, roadmap item 2: Stephan wants the grazing physics from first principles before a model
   choice; manager proposed a separate read-only study (infinite plate `ρ₀c/cos θ` against the finite plate of round
   1's `a1/p3/fp.py`, FIN windowing and MOD modal sum, σ cached for three lab sizes; Davy_2009_ForcedRadiationEfficiency
   in the review; candidates none / fixed angle / size-dependent angle / Davy's σ with a size). The question: which candidates remove code
   or a non-physical knob (`m` 1.64–35.13, heavy walls' `coiF` 49–9933 Hz, `att` up to 5.44 dB) while the 13 stay
   within the guideline; whether M13 / M14 / M15 take over the jobs of `m` and `att` so M10 / M11 become possible. The
   round-1 rig `a1/p3/tl` still exists (the worker copies it first). Carried in phase 2 (option b after M6): the two
   "1 = never" comments (`calc_diffraction_gain`, `include/quadriga_tools.hpp` `thin_slab_threshold`).
   Recorded, not scheduled: `ray_progress` tracer 4 under-transmits out of a summed lossy lining rarer than air into a
   held wall (`md/acoustic.md` 7, cause not traced; one-parameter mapping only).
2. **Grazing-incidence study (M14 from first principles)**, open, not started (Stephan, 2026-10-17: after the M10–M15
   round). Why: the infinite plate passes nearly everything at grazing (the air's normal impedance `ρ₀c/cos θ` grows
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
   code. Read-only until Stephan decides; the fitter `tx/txlib.py` / `txfit.py` (to `tools/acoustic` in this round).
   Must precede the parameter session (the fitted knobs depend on it). Also to check there: whether a finite-size form
   frees the heavy walls' knobs, which under M10 at 0–90° sit at their bounds (`md/m10_m15_report.md` 1.0).
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
