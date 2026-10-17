# Brief — M10 to M15 (the transmission candidates) and M12, judged on the 13 round-1 fits

This brief is the whole task description. Read it in full before doing anything, and re-read it at the start of
every phase. Findings go into `md/m10_m15_report.md` (create it), one section per phase, so the next session can pick
them up.

## Why

`md/acoustic.md` is the reference; read sections 2.1, 2.2, 2.3, 2.7, 3.4, 4 (M10 to M15), 7 (the mixed slopes, the
plate near its knee), 8 (the fitted knobs), 9 (classes 1, 2, 3, 5, 6 and the tolerance) and 10. Also
`md/acoustic_round1_report.md` 1.2 to 1.7 (the references), 3.1 (the first-principles models of the bands without a
measurement) and 3.6 (the fits).

The transmit side has the looser requirement (P6: sound insulation across rooms, about ±3 dB per octave; bands without
a measurement take the model spread, P7). Round 1 reached all 13 constructions within 2.51 dB under the plate rule
R1h with level, slope, `att`, resonance and bending stiffness (report 3.6), but the knobs on heavy walls and linings
are not physical: `m` 1.64 to 35.13 (20 is the mass law), `coiF` 49 to 9933 Hz on heavy walls (concrete 152 mm at 49 Hz,
brick 105 + plaster at 9933 Hz), `att` up to 5.44 dB (block 190). They shape what finite size, the coupling loss and
the modes do (report 3.1; `md/acoustic.md` 8). Section 4 lists six candidates, all "to examine", all low priority:

- **M10** `m` fixed at 20 (simplification): the mass term is the physical mass law, `m` a switch; the mixed-slope
  limitation (section 7: up to +7.66 dB) and the slope exponent of `MASS_FOLD` go.
- **M11** no `att` on a mass-law material (rule of use): A5 (one `att` per bonded run) goes.
- **M12** `coiA` ignored where `m > 0` (simplification): the N4 rule of use becomes code, one condition.
- **M13** the installed loss factor in frequency: `η(f) = η_int + C·f^(−1/2)` (coupling loss,
  [Davy_2009_SingleLeafCremer, Eq. 11]: `η_int + m″/(485·√f)` for laboratory specimens).
- **M14** a limiting angle for finite size: the mass cosine bounded below by `cos 78°` to `cos 85°`
  [Northwood_1968_PlasterboardWalls, p. 2].
- **M15** the loss of the bending term as a resistance: `TL` from `|1 + X₀·c·(j·(1 − w) + w/coiQ)|²` in place of the
  reactance modulus (up to 3 dB at coincidence where `X₀·c/coiQ ≈ 1`) [Davy_2009_SingleLeafCremer, Eq. 8].

The question for this session: **which candidates remove code or a non-physical knob while the 13 constructions stay
within the guideline**, and in particular whether the physical forms (M13, M14, M15) can take over the jobs that `m`,
`att` and the heavy walls' `coiF` do today, so that M10 and M11 become possible. Stephan prefers less code over
bit-identity and a variant that closes an item fully over a gate that leaves it open; rows may move within reasonable
bounds, explained first.

## Facts at the start

- Last commit 2c5be17 (M6). `git status` should be clean apart from this brief and `md/acoustic_manager.md`; if anything
  under `src/`, `tests/` or `md/acoustic.md` is modified, stop and ask Stephan.
- Baselines (`md/wave.md` 0, regenerated on the M6 build): `[wave]` 165110 assertions in 179 cases, `[acoustic]`
  39371 in 54, whole suite 1314030 in 1140. `src/ray_mesh_interact.cpp` 5132 lines. `python3 tools/acoustic/check_docs.py`
  passes.
- The code (`src/ray_mesh_interact.cpp`; find by name): `Material::medium_gain` (the mass term, `mass_min_path`, the
  bonded sum), `Material::mass_X0` (`res*` and `coi*` in `X₀`; the slope as `pow(mass_arg, 0.05·m)`),
  `Material::mass_TL` (the reactance-modulus law), `Material::interface_gain` (`att`, `attB` and the `coiA`
  Lorentzian, shared with wave media: wave cases carry `coiA` with `m = 0`, e.g. `test_wave_mesh_interact.cpp` and
  `test_wave_state_update_spec.cpp`), `MASS_FOLD` in `ray_state_update` (the thickness code; bodies of another slope
  are left out), `qd_mass_amp` (the solver's real factor).
- The 13 constructions and their R1h knobs (`rmb_chk.out`): glass 4 / 6 mm, gypsum 12.5 (model), plywood 18 (model),
  concrete 152, brick 95, brick 203, block 190, 4-16-4 and 4-12-4-12-4 (AGC), brick 203 + furred board, SCR 140 + board
  on dabs, brick 105 + plaster one face. Worst per construction under R1h with resonance and bending: 0.51 / 1.68 /
  1.60 / 1.32 / 0.87 / 0.21 / 0.80 / 1.02 / 0.71 / 0.77 / 1.70 / 2.51 / 0.09 dB. `tf` stays at each class's absorption
  fit (report 3.6); none of the 13 carries a peak (`md/m6_report.md` 0.5), so M5 and M6 do not move them.
- The round-1 rig survives in `build_linux/scratch/a1/p3/tl` today: `libac.py` (the formula copy, R1hX), `tlfit.py`,
  `refs.py` (the references), `conf/cols.tsv` (the fitted columns), `conf_tl.py`, `cmp_chk.py` (formula against the
  walker), `dense_chk.py`, `rmb_chk.out`, `alias.py`. It was written against the round-1 candidate copy, not today's
  library. `make clean` removes it: **copy what you use into your own scratch folder first.**
- The fit procedure of report 3.6: references measured octaves 125 Hz–4 kHz, 31.5 / 63 Hz and 8 kHz the model spread of
  report 3.1 (P7), gypsum and plywood the model spread in every octave, double units, the lining on dabs and plaster
  one face 125 Hz–4 kHz only; 192 angles × 6 frequencies per third for the fit, re-evaluated at 768 × 12; knobs bounded
  to physical ranges (coincidence 30 Hz–20 kHz, Q 1–300; resonance 10 Hz–1 kHz, Q 0.3–50).

## Hard rules

- Source changes only in `src/ray_mesh_interact.cpp`, only for what Stephan decides after phase 0, plus the two comment
  corrections of phase 2 (one of them in `include/quadriga_tools.hpp`). Tests and `md/` as below. No other file.
- The order is fixed: analysis (phase 0) → Stephan's decisions → fixtures first (phase 1, against today's library: the
  restated and new cases fail, nothing else moves) → source (phase 2) → documents (phase 3). Each phase ends with a
  report and a stop.
- Do not commit; Stephan commits.
- Measure, don't compute by hand. A fit is judged on the formula copy only after the copy has been checked against
  today's library through the walker; the best fit of each adopted candidate is confirmed through the walker as sheet
  and traced solid, and through `ray_progress` as a band. References in fixtures are written in the test from
  independent formulas (the thin-plate law, the column definitions), never from library code.
- `[wave]` must not move, and `tests/data/wave_twins.tsv` stays unchanged. Every candidate stays behind `m > 0`; M12
  in particular is gated on `m > 0` (wave cases carry `coiA`). If a candidate moves a `[wave]` row, stop and report.
- EM first: the mass term is acoustic (`m > 0`), but a mass-law material can be traced with an EM type. State what EM
  types get under each built candidate.
- The per-ray state does not grow. A candidate that changes `MASS_FOLD` keeps the thickness code's size.
- A peak or dip quoted at a frequency comes from a fine sweep (1/96 octave). Watch the alias trap of report 3.6 (a
  0.88 Hz coincidence that 48 angles could not see): bound the knobs, and re-evaluate on the dense grid.
- No new case may dominate the suite (a new case well under 0.5 s); `--durations yes` with the load before and after.
- Before anything else, from the repository root: `cmake --build build_linux --parallel`, the three runs of
  `md/wave.md` 0, and check the three counts above. Check the load (`~/.claude/CLAUDE.md`): single-threaded probes,
  one suite at a time, `timeout` on every probe; fits with many starts in the background, threads chosen from the
  load.

## Phase 0 — analysis in scratch (no edits outside scratch)

Fresh scratch folder `build_linux/scratch/tx`. A plain what-if copy of `src/ray_mesh_interact.cpp` first: it must
reproduce the baselines (only the seed line differs).

1. **The rig on today's library.** Copy the `p3/tl` files you need. Check the formula copy against today's library
   through the walker (as `cmp_chk.py` did: walls that switch every term on, sheet and traced solid, types 4 / 5, five
   angles, four frequencies); then reproduce the R1h column of report 3.6 per construction with `conf/cols.tsv`
   (expected 0.51 … 2.51 dB). Any difference is explained before step 2.
2. **The candidates on the 13 fits**, formula level, each as a variant of the formula copy, refitted with the round-1
   procedure (bounds, sampling, the same targets, enough starts that a rerun changes no worst residual by more than
   0.05 dB). Per variant: worst octave residual per construction and overall; the fitted knobs and whether they are
   physical (`coiF` against the physical `f_c` where one exists, `m`, `att`, `C` of M13). Variants:
   - today (R1h, all knobs), as the reference;
   - M10 (`m` = 20), M11 (`att` = 0), M10 + M11;
   - M13 (the form with a column `C`, and the fixed law with the surface mass from the columns: with `m` = 20, `fRef`
     encodes `ρ₀c/(πρ)`, so `m″ = ρ·thk` is known; say whether Davy's `m″/(485·√f)` needs a column at all), M14 (78°,
     80°, 85°), M15;
   - the physical set: M10 + M11 + M13 + M14 + M15, and the subsets that matter (which candidate carries the job `m`
     did on concrete 152 mm and brick 203, and `att` on block 190).
   Also the low bands (31.5 / 63 Hz) against the model spread (P7) per variant, and coincidence bands reported apart.
3. **M12.** A marker build over the whole suite: does any case reach `interface_gain` with `coiA ≠ 0` and `m > 0`?
   (Expected none, report 2.2's N4 marker.) The one condition as a what-if: rows moved (expected none), lines.
4. **Real code for the candidates that pass step 2** (and M12): what-if copies, lines added and removed, the whole
   suite against the baselines (every moved row; expected: *Bonded mass* cases with `m` 12 or 30 for M10, the
   *Coincidence* case for M13 / M15, *Thin sheet - mass-law angle dependence* for M14), `[wave]` 0 rows; sheet against
   traced solid on the fitted constructions; a band through `ray_progress` (`tools/acoustic/band_probe.cpp`) equal to
   its single frequencies; R + T ≤ 1 from air, summed and followed. For M10: what `MASS_FOLD` and the section-7
   limitations (mixed slopes, the bonded-mass fold three bodies deep) become.
5. **The fixtures, proposed**: which existing cases are restated (with the reference each takes) and what new case
   asserts each adopted candidate (a construction against the thin-plate law of 3.4 with its candidate term, written
   in the test). Expected run times.
6. **Recommendation** per candidate: adopt, drop, or keep as a proposal, with the reason (physics first, then less
   code, then the residuals); the section-7 items and open items each would close (mixed slopes, A5, the rule of use
   `coiA = 0`); which commit each replaces. Stop for Stephan's decisions.

## Phase 1 — fixtures (after the OK)

- Restate the cases the decided candidates move, in place (names unchanged unless the statement changes); add the new
  cases. References in the test from independent formulas.
- Run against today's library: the restated and new cases fail with the expected deviations, every other line of the
  three baselines unchanged (diff them). Report the failing assertions per case.

## Phase 2 — source

- The decided candidates in `src/ray_mesh_interact.cpp`, with their comments (and the public description of the
  columns where it changes: the `medium_gain` / `interface_gain` documentation blocks in the same file).
- Carried over from M6 (Stephan, 2026-10-16), comment only: `calc_diffraction_gain`'s description in
  `src/ray_mesh_interact.cpp` and the `thin_slab_threshold` comment in `include/quadriga_tools.hpp` (line ~125) still
  say "1 = never" for the slab threshold. Correct both to match the `eps` entry of `ray_state_update`'s description: a
  series is summed where `ρ ≥ eps`; `eps ≥ 1` re-emits every series whose `ρ` is below one, and a round trip at or
  above one is always summed. No row may move.
- Whole suite: restated and new cases pass; `[wave]` identical (`test_wave.txt` and the stored twins); counts; every
  moved row accounted for. Line count against 5132.

## Phase 3 — documents

- `md/acoustic.md`: 2.1, 2.2, 2.7 (the column table), 3.4, 4 (each candidate's status), 7 (closed limitations), 10
  (decisions and "Built" with the date), the appendix for new cases; the fitted knobs of 8 if they change. `md/wave.md`
  only if a counted line or a wave statement changes.
- The report: what moved, line counts, run times. `python3 tools/acoustic/check_docs.py` passes after the three
  baselines are regenerated (`md/wave.md` 0).

## Traps

- The traps of `md/wave.md` 11 and `md/acoustic_manager.md` 9: run a relinked test binary from the repository root;
  never two suites at once; a probe that includes a test file runs every case of it unless given a test spec, and a
  case name with a comma needs `\,`; `pkill -f <pattern>` from a shell whose command line contains the pattern kills
  that shell; `python3 -I` puts no script directory on `sys.path`.
- `make clean` removes `build_linux/scratch`, the round-1 rig included.
- The alias trap of report 3.6: an unbounded fit can put a coincidence at a sub-degree feature (0.88 Hz on the block
  wall) that a sparse angle grid cannot see.
- A mass-law term is zero below `X₀ = 1` (a kink); a traced leg shorter than 1.5 mm carries no term, a sheet carries it
  at any thickness.
- The plate rule bounds what a plate passes by its face's absorption (`T ≤ a`): a light panel below its knee passes
  about `a`, whatever the mass knobs say.
- Scratch is not kept: the report states every number a decision rests on; a reusable instrument goes to
  `tools/acoustic` if Stephan agrees.
- A plain what-if copy must reproduce the baselines before any candidate is judged against it.
