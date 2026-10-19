# Brief — sound transmission through a wall at grazing incidence, from first principles (M14)

This brief is the whole task description. Read it in full before doing anything, and re-read it at the start of
every phase. Findings go into `md/grazing_report.md` (create it), one section per phase, so the next session can pick
them up.

## Why

Stephan wants to understand, from first principles, what happens when sound meets a wall near grazing incidence
(70° to 90°), before he chooses how the library models it. He asked this before and never got a clear answer. **The
first deliverable is an explanation he can follow**: the physics in plain words, each step backed by a formula, a
number from a run, or a cited source. A model choice comes after that, and only on his decision.

What is known (`md/acoustic.md` 2.1, 3.4, 4 M14, 9; `md/m10_m15_report.md` 0.2 and 1.0):

- The library's mass term is the infinite thin plate: `TL = A·log10(1 + (X₀·c)²) / log10(1 + X₀²)` with `X₀` the
  plate's impedance ratio at normal incidence and `c` the cosine the wave has in air; the bending stiffness acts in
  `X₀` through `|1 − (f/coiF)²·sin⁴θ·(1 + j/coiQ)|`. As `c → 0` the term goes to zero: **a grazing wave passes the
  plate as if it had no mass**. The plate rule then bounds what passes by the face's absorption, `T → a`.
- On the 13 reference constructions of 2026-10-15 (`tools/acoustic/tl_refs.json`), the waves between 78° and 90°
  carry 4.3 % of a diffuse field's incident power, yet with the fitted columns the 0–90° average passes up to
  15.3 dB more per octave than the 0–78° average (concrete up to 10 dB, glazing 4 to 7 dB, a furred lining 15 dB).
  For a limp mass law alone the gap is 3 to 5 dB (checked by the manager in closed form).
- Decided 2026-10-17: a transmission reference is fitted on the library's own 0–90° average (`md/acoustic.md` 9).
  Fitted that way the 13 stay within 2.97 dB, but the heavy walls and linings sit at the bounds of their knobs (`att`
  at 12 dB on three, a level of 0.02 to 0.84 of their mass, the furred lining's `coiF` at 30 Hz). The suspicion is that
  the knobs are doing the job of a finite wall that the infinite plate lacks.
- The candidate on the table, M14: the mass cosine bounded below by `cos θ_lim`, so a wave past `θ_lim` is attenuated
  as at `θ_lim` (the ray is not dropped). At 0–90° it improved the fit (1.78 → 1.17 dB on the old fitted slope, 2.97 →
  2.80 dB under the mass law at 20, both at 78°; 85° helped less). It was not adopted: there is no physical account yet
  of why a limit, which angle, and whether it depends on the wall's size and the frequency.
- The literature named so far (`/home/stephan/accoustic_research/review.md` 2.5, PDFs in
  `/home/stephan/accoustic_research/references/`): infinite-plate theory integrated to 90° under-predicts measured R
  below `f_c`; Northwood found 85° fitted NRC data better than the common 78° [Northwood_1968_PlasterboardWalls, p. 2];
  Davy replaces the limiting angle by the diffuse-field forced radiation efficiency of a finite panel
  [Davy_2009_ForcedRadiationEfficiency]; a 1.55 × 0.95 m specimen measured about 5 dB above a 2.4 × 4.8 m one below `f_c`
  [Wareing_2015_SampleSizeVariations]; London's random-incidence mass law [London_1949_SingleWalls]; the infinite plate
  [Davy_2009_SingleLeafCremer]; accuracy of the methods [Kosala_2020_SingleBafflesAccuracy].

## Facts at the start

- Last commit bc24b0c. `git status` should be clean apart from this brief and `md/acoustic_manager.md`; if anything under
  `src/`, `tests/` or `md/acoustic.md` is modified, stop and ask Stephan.
- Baselines (`md/wave.md` 0): `[wave]` 165258 assertions in 180 cases, `[acoustic]` 40050 in 56, whole suite 1314857 in
  1143. `src/ray_mesh_interact.cpp` 5116 lines. `python3 tools/acoustic/check_docs.py` passes.
- The code (`src/ray_mesh_interact.cpp`, by name): `Material::mass_X0` (bending stiffness through `s2 = 1 − c²`),
  `Material::mass_TL` (the angle shape), `Material::medium_gain` (where the term is paid), the plate rule (`T =
  a·τ/(a + τ − a·τ)`); the cosine `c` is the one the wave has in air (`md/acoustic.md` 2.1).
- The instruments that survive in the repository: `tools/acoustic/txlib.py` (the formula copy of the built mass term
  and plate rule, with a switch `m14` for a limiting angle), `tools/acoustic/txfit.py` (the transmission fit; default
  the library at 0–90°, 2.97 dB; `--variant` takes a JSON switch set), `tools/acoustic/tl_refs.json` (the 13
  constructions and their references), `tools/acoustic/band_probe.cpp`, `tests/catch2_tests/test_acoustic_leaf.cpp`
  (`lw::walk`).
- **The finite-plate models of 2026-10-14** survive only in scratch: `build_linux/scratch/a1/p3/fp.py` (thin and
  Mindlin plate impedance; `tau_inf`, the infinite plate; `tau_finite`, a finite a × b plate in a baffle by the forced
  wave windowed to the plate and radiated by the Rayleigh integral on both faces [Villot et al., JSV 245 (2001)];
  `tau_modal`, the modal sum of the simply supported plate under plane-wave forcing), `t_fp.py` (checks: the windowed σ
  tends to `1/cos θ`; finite and modal agree on 4 mm glass within 0.1 dB), `light.py`, `heavy*.py`, and cached σ
  tables `sig48_1.20x2.40.npz`, `sig48_1.23x1.48.npz`, `sig48_4.37x2.84.npz`; `README.txt` describes them and their
  traps. `make clean` removes them: **copy what you use into your own scratch folder first.**

## Hard rules

- Phase 0 is analysis only: no edits outside `build_linux/scratch/grazing` and `md/grazing_report.md`. Source,
  tests and documents only after Stephan's decision, in the later phases.
- Measure, don't compute by hand: every number in the report comes from a run, a closed form evaluated by a script,
  or a cited page of a source. Say which.
- Write for a reader who is an engineer but not a specialist in plate radiation: define each quantity when it first
  appears (trace wavenumber, forced wave, radiation efficiency, coincidence angle), state each step's physical meaning,
  and keep the derivation short. Figures are welcome where a curve says more than a table (PNG in the scratch folder,
  referenced by path; the numbers that matter also in a table).
- A source is used for what it says: quote at most a short phrase, cite page or equation, and say where a statement
  is your own derivation. The PDFs are Stephan's library; read them, do not fetch other copies.
- The per-ray state does not grow; any model the study proposes must say what the library would need at an event
  (a column, a constant, the triangle's geometry) and whether it is available there.
- `[wave]` is not a concern of this study (the mass term is `m > 0` only), but say so for any proposed model.
- Check the load (`~/.claude/CLAUDE.md`) before heavy runs; the finite-plate and modal sums are expensive at high
  frequency; threads chosen from the load; `timeout` on every run.
- Do not commit; Stephan commits. Each phase ends with a report and a stop.

## Phase 0 — the study (scratch and the report only)

Fresh scratch folder `build_linux/scratch/grazing`. Copy `a1/p3/fp.py`, `t_fp.py`, the σ caches and what else you need;
rerun `t_fp.py` first and reproduce its two checks before using the models.

1. **The infinite plate at grazing, derived.** From a plane wave on an infinite plate between two half-spaces of air:
   the trace wavenumber `k·sin θ`, the plate velocity it forces, the pressure that velocity radiates on each side
   (`ρ₀c/cos θ` per unit velocity), and `τ(θ) = |1 + Z_p·cos θ/(2ρ₀c)|⁻²`. Show why `τ → 1` as `θ → 90°` whatever the
   mass, in words: what the air "sees" at grazing and why the plate's mass stops mattering. Then the coincidence angle
   `sin²θ_c = f_c/f` above `f_c`: where it lies for a light panel and for a heavy wall across 31.5 Hz–8 kHz, and how it
   meets the grazing region. Then the diffuse-field bookkeeping: the weight `sin 2θ`, what fraction of the incident power
   arrives past 70° / 78° / 85°, and what fraction of the **transmitted** power comes from there for a light pane, a
   heavy wall and a lined wall (the 13 fitted columns through `txlib.py`, and the infinite plate with physical
   parameters).
2. **The finite plate.** The same question for a plate of finite size in a baffle, with `fp.py`'s two independent
   models (`tau_finite`, `tau_modal`), **per angle**, not only diffuse: `τ(θ)` for 0° to 90° (fine near grazing) for
   glass 4 mm, gypsum 12.5 mm, concrete 152 mm and one lined wall or double unit if the models carry it, at the three
   cached sizes and at a room-wall size (for example 4 × 2.7 m and 10 × 3 m; compute σ if no cache exists), at octave
   centers 31.5 Hz to 8 kHz. Where does the finite plate part from the infinite one, and how: a level that stops
   falling (a plateau), a turnover, or something else; at which angle; how that angle depends on the size in
   wavelengths (`k·L`) and on whether the frequency is below or above `f_c`. Explain the mechanism in words (a forced
   wave of finite length radiates with a bounded efficiency; edges; what the modal model adds).
3. **What the measurement is.** What incident field ISO 10140 assumes and what a laboratory delivers near grazing
   (diffuse-field assumption, the specimen in a niche, sample size); whether a measured R is the 0–90° average of a
   finite plate; and what the library's room delivers at a wall (rays at every angle). Sources where they exist; mark
   reasoning as such.
4. **The literature, checked.** Davy's forced radiation efficiency [Davy_2009_ForcedRadiationEfficiency]: what it
   is, its closed form, the size it needs, and whether it reproduces `fp.py`'s finite plate per angle and on average;
   Northwood's 78° / 85° [Northwood_1968_PlasterboardWalls] and the origin of the 78° convention [London_1949_SingleWalls
   or the source the review gives]; Wareing's size effect against the finite-plate model at its two specimen sizes.
5. **Model candidates for the library**, each with what it needs at an event, its lines, and its error:
   - (a) none (the infinite plate, today);
   - (b) a fixed limiting angle (M14; 78°, 80°, 85°, or the angle the study supports);
   - (c) a limiting angle that depends on size and frequency (from a closed form, with a nominal size or a size column;
     the library cannot know a panel's size from a triangle);
   - (d) Davy's forced radiation efficiency replacing `1/cos θ` in the mass term, with a nominal size or a column;
   - (e) anything better the study finds.
   Judge each (i) per angle against `fp.py`'s finite plate at the sizes above (the largest error in dB per octave and
   angle band), (ii) on the 13 constructions at 0–90° with `txfit.py` (add the candidate as a switch in a scratch copy
   of `txlib.py`; report the worst residual and whether the heavy walls' knobs come off their bounds: `att`, level over
   mass, `coiF`), (iii) by code: lines in `src/ray_mesh_interact.cpp`, a column or not (a new column touches the OBJ
   writer, the Python and MATLAB readers and `tools/data_formats_md/format_materials.md`), the state.
6. **The explanation and a recommendation.** A short section written for Stephan first: what happens at grazing,
   why, how much it matters in a room, what a real wall does differently, and which model reproduces that best for its
   cost. Then the recommendation with the reason (physics first, then less code; a variant that closes the item over
   one that leaves a mode open), and the decisions Stephan has to take. Stop.

## Phase 1 — fixtures (after Stephan's decision, if a model is chosen)

- A case that states the chosen model per angle against its reference written in the test from an independent
  formula (the finite-plate closed form or the limit), sheet and traced solid, types 4 and 5; the cases the model moves
  (*Thin sheet - mass-law angle dependence* at least) restated. Run against today's library: the new and restated
  cases fail with the expected deviations, every other line of the three baselines unchanged. Run times. Stop.

## Phase 2 — source

- The model in `src/ray_mesh_interact.cpp` (and the column files if a column is decided). Whole suite: every case
  passes, `[wave]` identical, twins unchanged, every moved row accounted for; sheet against traced solid, a band
  through `ray_progress` equal to its single frequencies, R + T ≤ 1, on the 13 fitted constructions refitted under the
  model. Line count against 5116. Stop.

## Phase 3 — documents

- `md/acoustic.md` (2.1, 3.4, 4 M14, 7, 9, 10, the appendix), the fitter's switch moved into
  `tools/acoustic/txlib.py` as built, the finite-plate model into `tools/acoustic` if Stephan agrees (it is the
  reference the parameter session will need); regenerate the baselines; `check_docs.py` passes. Stop.

## Traps

- The traps of `md/wave.md` 11 and `md/acoustic_manager.md` 9 (run binaries from the repository root, one suite at a
  time, `pkill -f` kills its own shell, `python3 -I`).
- From `a1/p3/README.txt`: the finite plate needs a dense incidence grid at the coincidence peak (24 angles gave thirds
  jumping by 25 dB); the modal sum needs 16 samples per third (one mode per third on heavy walls at 125 Hz); the
  windowed σ must tend to `1/cos θ` for a large plate (check it at the sizes you use).
- A fit polished or selected on a coarser grid than the one its residual is read on picks vectors that alias it
  (`md/m10_m15_report.md` 0.2: select on 768 × 12).
- A peak or turnover quoted at an angle or a frequency comes from a fine sweep, not a grid point.
- Python 3.14 multiprocessing starts a forkserver whose workers survive a kill of the main process: kill its children
  by pid.
- Scratch is not kept: the report states every number it relies on.
