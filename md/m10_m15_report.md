# M10 to M15 and M12 — report

Brief: `md/m10_m15_brief.md`. One section per phase. Scratch rig: `build_linux/scratch/tx` (README.txt there); scratch
is not kept, so every number a decision rests on is stated here.

## 0. Phase 0 — analysis (2026-10-16)

### 0.0 Baseline and rig

- `git status` at the start: `md/acoustic_manager.md` modified, `md/m10_m15_brief.md` new; nothing under `src/`,
  `tests/` or `md/acoustic.md`. Last commit 2c5be17.
- `cmake --build build_linux --parallel` and the three runs of `md/wave.md` 0: `[wave]` 165110 assertions in 179
  cases, `[acoustic]` 39371 in 54, whole suite 1314030 in 1140, all passed (load 1.2 to 4.9). The new `test.txt`
  equals the one on disk from before the session but for the seed line. `src/ray_mesh_interact.cpp` 5132 lines.
  `python3 tools/acoustic/check_docs.py` passes.
- Plain what-if copy (`tx/plain.cpp`, compiled with the library flags and linked in front of `libquadriga.a`): the
  three runs reproduce the three baselines, only the seed line differs.
- Every what-if below is a patch of that copy (`tx/mkc.py`, patches combinable); every probe runs from the repository
  root on one OpenMP thread. The fits ran on this machine alone (load 25 to 38, all of it this session's fit
  processes); no timing is quoted from them.

### 0.1 The rig on today's library

**The formula copy.** `tx/txlib.py` is written anew from `src/ray_mesh_interact.cpp` at 2c5be17 (`mass_X0`,
`mass_TL`, `interface_gain`, `tf`, `qd_plate`): a closed mass-law sheet in air passes
`a·τ/(a + τ − a·τ)·g_int`, `a = tf(f)`, `τ` from the mass term. Round 1's `libac.py` was not reused: its X variants
let a negative interface loss act as a gain (M4b), which today's `interface_gain` clamps. Against today's library
through the walker (`tx/chk1.py`, the probe `tx/probe.cpp` = `bf::walk` of `test_acoustic_state_update.cpp` with all
26 columns; the three walls of round 1's `cmp_chk.py`, a brick wall with `m` 30.28 and `attB`, and the 4 mm wall in the
two-parameter mapping; sheet and traced solid, types 4 and 5, 0 / 10 / 30 / 60 / 80°, 125 / 500 / 2000 / 4000 Hz,
summed): T within **0.0022 dB** (traced 4 mm wall at 60°, 2 kHz), R within 0.0005 dB, one path through every wall.
The vectorized fitter (`tx/txfit.py`) equals the column-level copy within 6.4e-14 dB.

**The R1h column of report 3.6, reproduced** (`tx/repro.py`, 768 angles × 12 frequencies per third):

| knobs | g4 | g6 | gyp | ply | c152 | b95 | b203 | blk | 4-16-4 | triple | furred | dabs | plaster | worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| report 3.6 (`rmb_chk.out`) | 0.51 | 1.68 | 1.60 | 1.32 | 0.87 | 0.21 | 0.80 | 1.02 | 0.71 | 0.77 | 1.70 | 2.51 | 0.09 | 2.51 |
| bounded fits `fp_rmb*`, better of the R1h and R1m knobs under R1h | 0.51 | 1.68 | 1.60 | 1.32 | 0.87 | 0.21 | 0.80 | 1.02 | 0.71 | 0.77 | 1.70 | **1.96** | 0.09 | 1.96 |
| `conf/cols.tsv` through the column-level copy | 0.51 | 1.68 | 1.60 | 0.85 | 0.68 | 0.36 | 0.89 | 0.73 | 0.71 | 0.77 | 1.72 | 2.02 | 0.20 | 2.02 |

Two differences, both in round 1's bookkeeping, neither in today's library:

- **SCR 140 + board on dabs 2.51 → 1.96 dB.** Report 3.6 took the R1m knobs (2.51 under R1h) and left out its own
  40-start R1h fit (`fp_rmb40_SCR.json`, 1.96). Every other cell reproduces to the digit.
- **`conf/cols.tsv` is not the R1h column.** It holds the earlier unbounded fits (`fp_rm_*.json`; `rm_chk.out` reads
  0.51 1.68 1.60 0.85 0.68 0.36 0.89 0.73 0.71 0.77 1.72 2.02 0.20, reproduced to the digit), written before the
  bounded refit and never rewritten. Its block wall carries the 0.88 Hz coincidence of the alias trap. The walker
  confirmation of report 3.6 ran on these columns.

### 0.2 The candidates on the 13 fits

**Procedure.** Round 1's targets (`refs.py`: measured octaves 125 Hz–4 kHz, 31.5 / 63 Hz and 8 kHz the model spread
of report 3.1, gypsum and plywood the spread in every octave, double units, the lining on dabs and plaster one face
125 Hz–4 kHz only), `tf` at each class's absorption law, rule R1h, field incidence 0–78°, octave = energy mean of its
three thirds, cost = worst residual + 0.01 × rms. Knobs: level `X₀(1 kHz)`, slope `m`, `att` at 1 kHz and `attB`,
`coiF` / `coiQ` (M4a), `res*` (both signs), and `C` for M13's column; bounds of report 3.6 (coincidence 30 Hz–20 kHz,
`coiQ` 1–300; resonance 10 Hz–1 kHz, `resQ` 0.3–50) and, new, level 2–31600 (a 423 kg/m² wall needs 3200), `m`
0.5–60, `att` 0–12 dB, `attB` ±3, `C` 0–10. Variants are formula switches (`tx/txfit.py`): M10 `m = 20`; M11
`att = 0`; M13 column `η = 1/coiQ + C·(f/coiF)^−½`; M13 Davy `η = 1/coiQ + m″/(485√f)` with `m″ = X₀(1 kHz)·ρ₀c/(1000π)`,
exact where `m = 20`; M14 the mass cosine bounded below by `cos θ_lim`; M15 `|1 + jcX|²` with the bending loss resistive.

The optimizer: self-adaptive differential evolution (48 × 350 per resonance sign, on 96 angles × 3 frequencies per
third), Nelder-Mead on 192 × 6, then **selection on the evaluation grid**: every fitted knob vector of every variant
(projected onto the variant) evaluated at 768 × 12 and the best kept. A polish or a selection on a coarser grid picks
vectors that alias it (a first selection on 192 × 6 read the furred lining under M10 + M11 at 5.71 dB, at 768 × 12
3.33). Round 1's 16-start Nelder-Mead was far from converged: today's knobs reach **1.23 dB** where report 3.6 had
2.51 (1.96, 0.1). **Reruns**: the seeds of every cell are split into two independent halves (6 seeds per cell, 12 for
the M10 variants on the seven heavy constructions, 18 on the 57 cells whose halves first differed by more than 0.05 dB),
each half with its own selection over every vector of its seeds. The worst residual of a variant agrees between the
halves within 0.03 dB, except M10 with physical coincidence (2.99 / 3.27) and M10 + M13 + M15 with physical
coincidence (3.00 / 3.23), both decided by the concrete wall. Per cell, 71 of 455 cells differ by more than 0.05 dB
between the halves (up to 0.45 dB; the concrete cell of M10 + M13 + M15 with physical coincidence 1.53 dB); apart from
those two concrete cells none decides its variant's worst. The tables give the better half per cell and the two
halves' worst.

**Field incidence 0–78°, as round 1** (worst octave residual per construction, dB; "outside coi": every band but the
octave that holds the construction's physical coincidence; "31.5 / 63 Hz": the low bands against the model spread):

| variant | g4 | g6 | gyp | ply | c152 | b95 | b203 | blk | 4164 | trip | furr | dabs | plas | worst (A / B) | outside coi | 31.5 / 63 Hz | coi band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| today | 0.18 | 0.89 | 1.23 | 0.31 | 0.44 | 0.00 | 0.13 | 0.14 | 0.07 | 0.45 | 1.04 | 0.06 | 0.00 | **1.23** (1.23 / 1.24) | 1.23 | 1.22 | 1.23 |
| M10 | 0.59 | 1.93 | 2.80 | 0.57 | 0.44 | 0.15 | 0.59 | 0.29 | 1.10 | 0.82 | 1.05 | 0.32 | 0.00 | **2.80** (2.80 / 2.80) | 2.80 | 2.80 | 2.79 |
| M11 | 0.37 | 1.49 | 1.51 | 0.74 | 0.56 | 0.01 | 0.13 | 0.33 | 0.12 | 0.45 | 1.19 | 0.08 | 0.02 | **1.51** (1.51 / 1.51) | 1.51 | 1.19 | 1.51 |
| M10 + M11 | 0.59 | 1.95 | 2.84 | 0.58 | 0.86 | 0.94 | 0.66 | 1.02 | 1.28 | 1.12 | 3.33 | 1.96 | 0.04 | **3.33** (3.33 / 3.33) | 3.33 | 3.33 | 2.83 |
| M13 column | 0.18 | 0.89 | 1.23 | 0.31 | 0.35 | 0.00 | 0.10 | 0.05 | 0.07 | 0.45 | 1.03 | 0.02 | 0.00 | **1.23** (1.23 / 1.24) | 1.23 | 1.22 | 1.23 |
| M13 Davy | 0.18 | 0.89 | 1.23 | 0.30 | 0.44 | 0.00 | 0.13 | 0.14 | 0.07 | 0.66 | 1.04 | 0.03 | 0.00 | **1.23** (1.23 / 1.24) | 1.23 | 1.22 | 1.23 |
| M10 + M13 column | 0.59 | 1.93 | 2.80 | 0.57 | 0.38 | 0.15 | 0.51 | 0.29 | 1.10 | 0.83 | 1.05 | 0.32 | 0.00 | **2.80** (2.80 / 2.80) | 2.80 | 2.80 | 2.79 |
| M10 + M13 Davy | 0.57 | 1.93 | 2.80 | 0.55 | 0.44 | 0.15 | 0.59 | 0.29 | 1.10 | 0.83 | 1.05 | 0.32 | 0.00 | **2.80** (2.80 / 2.80) | 2.80 | 2.80 | 2.79 |
| M15 | 0.18 | 0.90 | 1.24 | 0.32 | 0.44 | 0.00 | 0.13 | 0.13 | 0.08 | 0.54 | 1.04 | 0.08 | 0.00 | **1.24** (1.24 / 1.24) | 1.24 | 1.23 | 1.24 |
| M10 + M15 | 0.64 | 1.95 | 2.82 | 0.52 | 0.44 | 0.15 | 0.59 | 0.29 | 1.10 | 0.83 | 1.05 | 0.32 | 0.00 | **2.82** (2.82 / 2.82) | 2.82 | 2.82 | 2.81 |
| M13 Davy + M15 | 0.18 | 0.91 | 1.24 | 0.33 | 0.44 | 0.00 | 0.13 | 0.13 | 0.08 | 0.74 | 1.04 | 0.01 | 0.00 | **1.24** (1.24 / 1.24) | 1.24 | 1.23 | 1.24 |
| M10 + M13 Davy + M15 | 0.63 | 1.97 | 2.82 | 0.56 | 0.44 | 0.15 | 0.59 | 0.29 | 1.11 | 0.84 | 1.05 | 0.32 | 0.00 | **2.82** (2.82 / 2.82) | 2.82 | 2.82 | 2.81 |
| M10 + M11 + M13 Davy + M15 | 0.64 | 1.97 | 2.86 | 0.70 | 0.86 | 0.94 | 0.66 | 1.02 | 1.28 | 1.12 | 3.33 | 1.96 | 0.03 | **3.33** (3.33 / 3.33) | 3.32 | 3.33 | 2.85 |
| M10 + M11 + M13 column + M15 | 0.64 | 1.97 | 2.86 | 0.69 | 0.85 | 0.80 | 0.64 | 1.00 | 1.28 | 1.12 | 3.25 | 1.96 | 0.03 | **3.25** (3.26 / 3.25) | 3.25 | 3.25 | 2.85 |

A minimax fit spreads its worst residual over several bands, so the three right-hand columns are often equal. The
3.33 dB of the furred lining under M10 + M11 sits at +3.33 / −3.32 / +3.33 / −3.33 / +3.33 dB in the measured
octaves 250 Hz–4 kHz (and −3.33 dB at 31.5 Hz against the spread), outside the laboratory reproducibility there (±1 dB
from 315 Hz, section 9). M10's 2.80 dB is the gypsum board, a model reference (the spread in every octave), at 31.5 Hz
and 1 / 2 / 4 kHz; glass 6 mm, measured, reads −1.92 / +1.91 / +1.93 dB at 1 / 2 / 4 kHz (2 kHz its coincidence
octave).

**`coiF` pinned at the physical coincidence** (report 1.2–1.7: glass 2989 / 1993 Hz, gypsum 2500, plywood 1146,
concrete 111, brick 95 160, brick 203 100, the units the 4 mm pane's 2989, the furred lining its brick's 100; block,
dabs and plaster have none and stay free):

| variant | g4 | g6 | gyp | ply | c152 | b95 | b203 | blk | 4164 | trip | furr | dabs | plas | worst (A / B) | outside coi | 31.5 / 63 Hz | coi band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| today | 0.19 | 1.24 | 1.27 | 0.55 | 0.68 | 0.00 | 0.50 | 0.14 | 0.56 | 0.83 | 1.23 | 0.06 | 0.00 | **1.27** (1.27 / 1.27) | 1.27 | 1.26 | 1.24 |
| M13 Davy | 0.19 | 1.24 | 1.27 | 0.49 | 0.68 | 0.02 | 0.51 | 0.14 | 0.56 | 0.83 | 1.23 | 0.03 | 0.00 | **1.27** (1.27 / 1.27) | 1.27 | 1.26 | 1.24 |
| M13 column | 0.19 | 1.24 | 1.27 | 0.55 | 0.66 | 0.00 | 0.49 | 0.05 | 0.56 | 0.83 | 1.05 | 0.02 | 0.00 | **1.27** (1.27 / 1.27) | 1.27 | 1.26 | 1.24 |
| M15 | 0.19 | 1.25 | 1.28 | 0.34 | 0.68 | 0.00 | 0.51 | 0.13 | 0.56 | 0.83 | 1.24 | 0.08 | 0.00 | **1.28** (1.28 / 1.28) | 1.28 | 1.27 | 1.25 |
| M13 Davy + M15 | 0.19 | 1.25 | 1.28 | 0.33 | 0.66 | 0.00 | 0.51 | 0.13 | 0.56 | 0.83 | 1.24 | 0.01 | 0.00 | **1.28** (1.28 / 1.28) | 1.28 | 1.27 | 1.25 |
| M13 column + M15 | 0.19 | 1.25 | 1.28 | 0.34 | 0.66 | 0.00 | 0.49 | 0.09 | 0.56 | 0.83 | 1.04 | 0.02 | 0.00 | **1.28** (1.28 / 1.28) | 1.28 | 1.27 | 1.25 |
| M10 | 0.80 | 2.25 | 2.98 | 0.45 | 2.93 | 1.44 | 1.28 | 0.29 | 1.10 | 0.84 | 2.99 | 0.32 | 0.00 | **2.99** (2.99 / 3.27) | 2.99 | 2.98 | 2.25 |
| M10 + M13 Davy + M15 | 0.91 | 2.27 | 3.00 | 0.53 | 1.70 | 1.32 | 1.14 | 0.29 | 1.11 | 0.84 | 2.86 | 0.32 | 0.00 | **3.00** (3.00 / 3.23) | 3.00 | 3.00 | 2.27 |
| M10 + M11 + M13 Davy + M15 | 1.12 | 2.27 | 3.04 | 0.65 | 3.23 | 2.40 | 2.15 | 1.02 | 1.28 | 1.27 | 5.23 | 1.96 | 0.03 | **5.23** (5.23 / 5.23) | 5.23 | 3.23 | 3.19 |

**What a room sees: random incidence 0–90°.** A ray tracer meets a wall at every angle; the fits above (and round
1's) average 0–78°. Today's knobs integrated to 90° instead, no refit (`TL(0–90°) − TL(0–78°)`, dB):

| construction | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | worst residual 0–78° → 0–90° |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glass 4 mm | −0.97 | −1.15 | −0.75 | −0.96 | −2.60 | −5.17 | −4.83 | −0.97 | +0.11 | 0.18 → 5.00 |
| glass 6 mm | −1.24 | −0.73 | −0.19 | −0.67 | −3.80 | −3.78 | −1.06 | +0.06 | +0.16 | 0.89 → 4.67 |
| gypsum 12.5 (model) | −0.83 | −0.12 | −0.52 | −2.48 | −2.80 | −2.41 | −1.93 | −0.23 | +0.14 | 1.23 → 3.64 |
| plywood 18 (model) | −0.15 | −0.36 | −0.73 | −1.24 | −1.35 | −0.25 | −0.47 | −0.05 | +0.11 | 0.31 → 1.50 |
| concrete 152 | −0.06 | −0.18 | −0.80 | −4.71 | −7.32 | −8.25 | −9.14 | −10.00 | −7.90 | 0.44 → 9.58 |
| brick 95 | −0.69 | −1.13 | −1.33 | −0.58 | +0.07 | +0.16 | +0.18 | +0.19 | +0.19 | 0.00 → 1.33 |
| brick 203 | −3.10 | −0.46 | −0.36 | −2.62 | −10.44 | −4.86 | −0.95 | +0.02 | +0.17 | 0.13 → 10.31 |
| block 190 | −1.59 | −0.56 | −0.91 | −6.65 | −7.47 | −3.68 | −0.78 | +0.04 | +0.17 | 0.14 → 7.61 |
| 4-16-4 (AGC) | −1.74 | −1.16 | −0.53 | −0.12 | −0.93 | −2.66 | −5.11 | −0.32 | +0.17 | 0.07 → 5.18 |
| 4-12-4-12-4 (AGC) | −1.43 | −1.20 | −0.86 | −0.69 | −1.92 | −3.88 | −7.07 | −0.07 | −0.13 | 0.45 → 7.50 |
| brick 203 + furred board | −2.92 | −0.63 | −0.11 | −0.28 | −11.28 | −14.61 | −14.72 | −15.32 | −6.02 | 1.04 → 15.77 |
| SCR 140 + board on dabs | −4.11 | −4.10 | −3.97 | −3.70 | −4.74 | −6.46 | −9.81 | −0.55 | +0.14 | 0.06 → 9.74 |
| brick 105 + plaster one face | −1.05 | −1.51 | −1.59 | −2.30 | −4.35 | −5.89 | −6.34 | −6.95 | −7.61 | 0.00 → 6.95 |

The waves between 78° and 90° carry 4.3 % of the diffuse weight but pass far more than the average: the mass term
goes to zero at grazing, and a plate with its coincidence below the band (`coiF` at 100–550 Hz, or a broad dip at 5–8
kHz with Q 1–2) is at coincidence there. **Round 1's fits are fits of the 0–78° average; in a room the same columns
pass up to 15.3 dB more in an octave, and the worst residual grows to 15.8 dB.** With M14 at 78° and the same knobs the
worst residual at 0–90° is at most 10.9 dB (furred lining; block 4.5, triple unit 4.3, under 3 elsewhere); with M14 at
85° at most 14.9 dB. Refitted at 0–90° (random incidence, `tx/batch.py`
`amax 90`):

| variant | g4 | g6 | gyp | ply | c152 | b95 | b203 | blk | 4164 | trip | furr | dabs | plas | worst (A / B) | outside coi | 31.5 / 63 Hz | coi band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| today | 0.97 | 1.78 | 1.35 | 0.59 | 0.59 | 0.06 | 0.01 | 0.35 | 0.07 | 1.18 | 0.92 | 0.15 | 0.07 | **1.78** (1.78 / 1.78) | 1.78 | 1.35 | 1.76 |
| M14 78° | 0.26 | 0.93 | 1.17 | 0.36 | 0.48 | 0.09 | 0.00 | 0.20 | 0.10 | 0.59 | 0.61 | 0.13 | 0.00 | **1.17** (1.17 / 1.17) | 1.17 | 0.93 | 1.17 |
| M14 80° | 0.32 | 0.99 | 1.20 | 0.29 | 0.55 | 0.05 | 0.00 | 0.23 | 0.09 | 1.38 | 1.08 | 0.09 | 0.00 | **1.38** (1.39 / 1.38) | 1.38 | 1.08 | 1.20 |
| M14 85° | 0.69 | 1.48 | 1.21 | 0.57 | 0.50 | 0.04 | 0.00 | 0.26 | 0.07 | 0.42 | 0.40 | 0.19 | 0.00 | **1.48** (1.48 / 1.50) | 1.48 | 1.21 | 1.48 |
| M10 | 0.97 | 2.42 | 2.97 | 0.60 | 0.59 | 0.34 | 0.61 | 0.92 | 1.55 | 2.69 | 2.69 | 1.73 | 0.07 | **2.97** (2.97 / 2.97) | 2.97 | 2.96 | 2.96 |
| M10 + M14 78° | 0.52 | 2.00 | 2.80 | 0.56 | 0.48 | 0.27 | 0.59 | 0.44 | 1.15 | 1.45 | 0.61 | 1.80 | 0.00 | **2.80** (2.80 / 2.80) | 2.80 | 2.79 | 2.79 |
| M10 + M14 85° | 0.69 | 2.29 | 2.94 | 0.57 | 0.54 | 0.32 | 0.59 | 0.62 | 1.40 | 2.48 | 0.40 | 0.56 | 0.00 | **2.94** (2.94 / 2.94) | 2.94 | 2.94 | 2.94 |
| M10 + M11 + M14 85° | 0.71 | 2.31 | 3.00 | 0.83 | 0.86 | 0.97 | 0.66 | 1.03 | 1.81 | 3.10 | 4.70 | 3.42 | 0.03 | **4.70** (4.70 / 4.70) | 4.70 | 4.70 | 4.70 |
| physical set (M10 + M11 + M13 Davy + M14 78° + M15) | 0.61 | 2.06 | 2.86 | 0.67 | 0.85 | 0.93 | 0.66 | 1.02 | 1.63 | 1.85 | 3.38 | 3.21 | 0.03 | **3.38** (3.38 / 3.38) | 3.38 | 3.38 | 2.85 |
| physical set, M14 85° | 0.75 | 2.33 | 3.02 | 0.76 | 0.37 | 0.97 | 0.66 | 1.03 | 1.81 | 3.10 | 4.70 | 3.21 | 0.03 | **4.70** (4.70 / 4.70) | 4.70 | 4.70 | 4.70 |
| physical set (M13 column), M14 85° | 0.72 | 2.33 | 3.02 | 0.74 | 0.31 | 0.81 | 0.64 | 1.00 | 1.81 | 3.10 | 4.36 | 2.84 | 0.03 | **4.36** (4.36 / 4.36) | 4.36 | 3.25 | 3.01 |
| physical set, M14 85°, coincidence physical | 0.75 | 2.54 | 3.63 | 0.86 | 2.53 | 2.49 | 2.07 | 1.03 | 3.07 | 4.43 | 4.93 | 3.21 | 0.03 | **4.93** (4.93 / 4.93) | 4.93 | 4.49 | 4.49 |

**The knobs** (the better half's fit per cell; "level / mass": the fitted `X₀(1 kHz)` expressed as `m″` over the
construction's surface mass of report 1.2–1.7, a mass only where `m = 20`; light: glass, gypsum, plywood, the units;
heavy: the walls, linings and plaster; coincidence of the light panels as fitted / physical):

| variant | `m` light | `m` heavy | level / mass, light | level / mass, heavy | `att` max (at the 12 dB bound) | `coiF` / `f_c`, light | `coiF` Hz: c152 / b95 / b203 / furred (physical 111 / 160 / 100 / 100) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| today | 9.3 … 39.5 | 2.5 … 43.2 | 0.67 … 1.23 | 0.13 … 1.19 | 4.2 | 0.88 … 1.44 | 5990 / 154 / 528 / 7466 |
| M11 | 9.4 … 39.5 | 4.0 … 42.7 | 0.77 … 1.23 | 0.15 … 1.41 | 0 | 0.89 … 1.04 | 7506 / 229 / 528 / 90 |
| M10 | 20 | 20 | 0.49 … 1.08 | 0.09 … 0.97 | 12.0 (2 walls) | 0.92 … 1.04 | 6110 / 18468 / 8857 / 7993 |
| M10 + M11 | 20 | 20 | 0.66 … 1.20 | 0.28 … 1.34 | 0 | 0.93 … 1.65 | 2984 / 4336 / 19999 / 1991 |
| M13 column (`C` 0 … 3.85) | 9.3 … 39.5 | 2.6 … 42.9 | 0.67 … 1.23 | 0.06 … 1.40 | 10.6 | 0.88 … 1.44 | 5020 / 154 / 539 / 123 |
| M15 | 9.3 … 38.9 | 2.4 … 20.7 | 0.74 … 1.19 | 0.12 … 0.49 | 4.2 | 0.88 … 1.44 | 5990 / 160 / 527 / 7466 |
| today, coincidence physical | 9.2 … 27.0 | 2.5 … 43.2 | 0.70 … 1.03 | 0.06 … 1.19 | 11.0 | 1 | pinned |
| M10, coincidence physical | 20 | 20 | 0.49 … 1.08 | 0.03 … 0.97 | 12.0 (4) | 1 | pinned |
| M10 + M13 Davy + M15, coincidence physical | 20 | 20 | 0.49 … 1.06 | 0.03 … 0.97 | 12.0 (4) | 1 | pinned |
| today at 0–90° | 7.6 … 39.8 | 0.5 … 26.1 | 1.03 … 1.81 | 0.14 … 0.74 | 12.0 (1) | 0.95 … 1.54 | 6315 / 142 / 159 / 183 |
| M14 78° at 0–90° | 8.8 … 38.6 | 2.6 … 38.3 | 0.77 … 1.27 | 0.16 … 0.82 | 11.5 | 0.91 … 1.29 | 6845 / 142 / 156 / 6123 |
| M10 + M14 78° at 0–90° | 20 | 20 | 0.55 … 0.98 | 0.07 … 0.82 | 12.0 (2) | 0.96 … 1.11 | 6845 / 14018 / 9453 / 6123 |

What the knobs say:

- **Light panels are physical under M10.** With `m = 20` the fitted level of glass 4 / 6 mm is 0.98 / 0.85 of its mass
  and the coincidence within 8 % of the physical one; plywood 1.08, gypsum 0.60. The double units take about half
  their mass (0.49 / 0.69: one sheet stands for two panes and a cavity) and 4.4 to 4.5 dB of `att`.
- **On heavy walls and linings no variant is physical; removing a knob moves its job to another.** Today's slope runs
  2.5 (block) to 43.2 (dabs); under M10 the walls take a level of 0.09 to 0.97 of their mass with a flat `att` of 2 to
  12 dB (the bound: brick 95 at 16 kg/m² for 189 kg/m² and 12.0 dB), and concrete and the bricks take `coiF` at 6 to
  18 kHz with `coiQ` 1 to 1.7, a broad high-frequency tilt. Under M11 the slope does the same job (4.0 to 19.6 on the
  walls and the furred lining, 42.7 on the lining on dabs). With `coiF` pinned at the
  physical 100–160 Hz, today's fits stay at 1.27 dB but by a slope of 3.9 / 8.3 / 13.9 and 10.6 / 6.8 / 11.0 dB of `att` on
  concrete / brick 95 / brick 203. A low level plus a flat `att` is a level with a weaker angle dependence: the fits use it as the finite size that the
  infinite plate lacks (report 3.1, `md/acoustic.md` 3.4).
- **The job of `m` on concrete 152 and brick 203.** Round 1's `m` 1.64 on concrete was the optimizer's: today's best
  concrete fit has `m` 20.7. Brick 203 keeps 5.85 today; under M10 it reaches 0.59 dB with `att` 8.3 dB, a level of 0.11
  of its mass and `coiF` 8857 Hz at Q 1. None of M13, M14, M15 carries it: M10 + M13 + M15 reads the same 0.59.
- **The job of `att` on block 190** needs no candidate: without `att` (M11) the block reads 0.33 dB (slope 7.5), with
  M10 + M11 1.02 dB.
- **M13** changes no variant's worst residual by more than 0.01 dB, and no cell by more than 0.18 dB (the furred lining
  with the coincidence physical, 1.23 → 1.05 with the column), except one unconverged cell: with M10 and the
  coincidence physical, M13 + M15 brought concrete to 1.70 dB in one half and 3.23 in the other (M10 alone 2.93 /
  3.27); the brick walls there move by 0.1 dB. The
  fitted column is not Davy's: `C` is 0 on every light panel and 0.21 / 0.37 / 3.85 / 3.75 on concrete / brick 203 /
  block / furred lining (at fitted `coiF` 5020 / 539 / 1154 / 123 Hz), where Davy's `m″/(485√f)` at the physical
  coincidence is 0.068 / 0.084 for concrete / brick 203. Under M10 `fRef` carries `m″` (`m″ = ρ₀c·thk/(π·fRef)`), so
  Davy's law needs **no column**; it needs the acoustic mapping's constants in the library (0.4).
- **M15** moves no worst residual by more than 0.02 dB; the fitted `coiQ` absorbs the resistive cross term.
- **M14** acts only where the fit sees grazing angles: at 0–78° a limit at 78° or above changes nothing. At 0–90° it
  helps (1.78 → 1.17 dB today, 2.97 → 2.80 under M10, both at 78°; 85° helps least).
- **The physical set** (M10 + M11 + M13 + M15, and M14 at 0–90°) misses the furred lining: 3.33 dB at 0–78°, 3.38 /
  4.70 dB at 0–90° with M14 78° / 85°, 5.23 dB at 0–78° with the coincidence physical. M10 and M11 together are what
  fails it; M13, M14 and M15 change that by at most 0.08 dB (0–78°).

### 0.3 M12

**Marker** (`tx/mk12.cpp`: a line printed where `interface_gain` meets `coiF > 0`, `coiA ≠ 0` and `m > 0`; checked live
on a wall that carries all three): over the whole suite, 1140 cases, **no case reaches it**. Report 2.2's N4 marker
holds on today's suite.

**The one condition as a what-if** (`!(m > 0.0)` added to the existing `if (coiF > 0.0 && coiA != 0.0)` in
`interface_gain`, with a comment): the whole suite and `[wave]` reproduce their baselines, only the seed line differs:
**0 rows moved**. Lines: 5132 (one line changed).

### 0.4 Real code

Every candidate as a what-if of the plain copy (`tx/mkc.py`). Lines against 5132; "out / in" counts the diff lines
(comments included).

| candidate | what the code does | lines | out / in |
| --- | --- | --- | --- |
| M10 | `mass_X0` returns `(f/fRef)·M·res·coi` (no `pow`); `MASS_FOLD` sums `dᵢ·fRef_R/fRefᵢ` over every mass-law body (`B.m > 0` for `B.m == R.m`, no exponent); the column, the public `medium_gain` text and three comments say "`m > 0` switches the mass law on, 20 dB per decade" (outside this file `tools/data_formats_md/format_materials.md` calls `m` the slope) | 5126 (−6) | 32 / 26 |
| M11 as code | `interface_gain` returns 1 where `m > 0` (`att`, `attB` and `coiA` of a mass-law material are no loss; M12 with it); the public `interface_gain` text would say so (not in the count) | 5134 (+2) | 0 / 2 |
| M12 | `&& !(m > 0.0)` on the Lorentzian's condition in `interface_gain` | 5132 (±0) | 1 / 1 |
| M13 (Davy, fixed law) | `η = 1/coiQ + K·M/(fRef·√f)` in `mass_X0`, `K = ρ₀c·(f_GHz/f_Hz)^1.5/(485π)` (343 m/s, 1.204 kg/m³: the library would carry the acoustic mapping's constants for the first time); `m″` from the columns is right only at `m = 20` | 5138 (+6) | 1 / 7 |
| M13 (column `C`) | not built: a new column needs `MaterialCols` and `Material` here (+4 lines) and `src/obj_file_write.cpp`, `api_mex/obj_file_read.cpp`, `api_python/qpy_obj_file_read.cpp` and `tools/data_formats_md/format_materials.md` outside this file | — | — |
| M14 | `mass_c = max(mass_c, cos θ_lim)` at the head of `mass_X0` and of `mass_TL`, one constant | 5138 (+6) | 0 / 6 |
| M15 | `mass_X0` returns the complex `X = Xm·(1 − w·(1 + j/coiQ))`; the run sums it (`mass_bonded`, `x_run`, `MASS_RUN`, `mass_host`, nine declarations); `mass_TL` takes `log10(1 + (c|X|)² − 2c·Im X)` in the numerator | 5137 (+5) | 20 / 25 |
| M10 + M11 + M13 + M15 | the physical set at 0–78° | 5139 (+7) | 51 / 58 |

**Each copy against its formula variant** through the walker (`tx/chk1.py`, the walls of 0.1, sheet and traced solid,
types 4 / 5; for M14 also 80 / 84 / 86 / 88 / 89.5°): within 0.0022 dB in T and 0.0005 dB in R, as today's library.

**The whole suite against the baselines** (rows: printed table rows that differ; assertions: failed):

| case | M10 | M11 code | M12 | M13 | M14 (85°) | M15 |
| --- | --- | --- | --- | --- | --- | --- |
| *Bonded mass - the plate at any depth where the slopes agree* | 367 rows, 134 | | | | | |
| *Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate* | 45, 52 | 9, 0 | | | | |
| *Bonded mass - a mass-law sheet inside a mass-law wall* | 16, 28 | | | | | |
| *Thin sheet - mass-law angle dependence* | 16, 34 | | | | 7, 16 | |
| *Unentered branches - a sheet without the mass column of its mass-law host's permittivity is merged* | 24, 24 | | | | | |
| *Stack with a mass-law layer - resolved by segments, not killed* | 17, 16 | | | | | |
| *ray_progress - the bonded plate at any depth where the slopes agree* | 88, 0 | | | | | |
| *ray_progress - the wall behind a traced layer is met at the incidence angle at every frequency of a band* | 198, 0 | | | | | |
| *Leak direction - a closed face hands the wall behind a traced layer the direction the wave has in air* | 112, 0 | | | | | |
| *Leak direction - a wave held in a mass-law body keeps its direction and a closed face hands on the wave's* | 288, 0 | | | | | |
| *Lossy verdict - a lossy second medium has no critical angle - the mass-law part* | 32, 0 | | | | | |
| *Refracting tracer - a traced bend is known from the state and the leg is exact - the mass-law part* | 10, 0 | | | | | |
| *Leak reciprocity - a mass-law panel on a rigid wall as a sheet and as two traced solids* | | 10, 5 | | | | |
| *Total reflection of the outer pair - the sheet stays in the result - the mass-law part* | | 5, 15 | | | | |
| *Unentered branches - ray_mesh_interact's own glued composite against the redo - the mass-law part* | | 15, 0 | | | | |
| *Coincidence - on a mass-law material coiF and coiQ are the bending stiffness of the plate* | | | | 18, 4 | | 30, 6 |
| total | 1213 rows, 288 in 7 cases | 39, 20 in 2 | 0 | 18, 4 in 1 | 7, 16 in 1 | 30, 6 in 1 |

Why they move:

- **M10**: every moved case carries a material with `m` 30 (or 12 in *Thin sheet*); its rows move by the slope, and
  the failing assertions are references written with `0.05·m` in the test. The print-only cases (*Leak direction*,
  *Lossy verdict*, *Refracting tracer*, the two `ray_progress` cases) print levels of an `m` 30 wall and assert
  directions or bands. Expected by the brief; the *Coincidence* case does not move (`m` 20).
- **M11 as code**: *Leak reciprocity* takes out the interface gains its fixture computes from `att` (1 and 2 dB on
  the two mass-law bodies), so it is off by exactly their difference, 1.000 dB; *Total reflection of the outer pair*
  writes its reference with the sheet's `att`. Both are references that carry `att` on a mass-law body; the library's
  rows are the plate without that loss. The two print-only cases print levels of the same bodies.
- **M12**: nothing (0.3).
- **M13 (Davy)** and **M15**: only *Coincidence*. M15 there: at the coincidence angle (59.8°, 4 kHz, `X₀·c/coiQ`
  2.30) the mass term rises from 7.88 to 10.22 dB (the 3 dB of 3.4 at `X₀·c/coiQ` near one), the pane's T by 0.70 dB
  (−14.00 → −14.70 dB: the plate rule holds T near `a` = 0.05 there); −0.009 to −0.011 dB at 70 / 80°, −0.0011 dB at
  1 kHz. M13 adds `m″/(485√f)` = 0.00065 at
  1 kHz to the 4 mm pane's 0.015.
- **M14 at 85°**: the *Thin sheet* rows past 85°.

**`[wave]`**: identical for every copy (M10, M11, M12, M13, M14, M15 and the combinations M10 + M11, M10 + M11 + M13 +
M15, M10 + M14), only the seed line differs; the stored twins are asserted by that run and `tests/data/wave_twins.tsv`
is untouched. Every candidate sits behind `m > 0`: M10, M13, M14 and M15 in `mass_X0` / `mass_TL`, which a material
without the column never reaches, M11 / M12 in a condition on `m`.

**EM types** (`tx/emchk.py`: the change each copy makes against today, EM type 1 against scalar type 4, the walls of
0.1, sheet and traced solid, 0° to 86°): identical, 0.0000 dB, for every candidate. The mass term, the plate rule and
the interface gain are formed the same way for every tracer, so an EM type traced on a mass-law material gets each
candidate exactly as the scalar types do (M10's slope, M11's missing `att`, M14's limit, M15's resistance).

**Sheet against traced solid on the fitted constructions** (`tx/conf.py`, the best fit of each variant as library
columns, 96 angles × 3 frequencies per third, type 4, through each copy): walker and formula within 0.0000 dB (sheet)
and 0.0021 dB (traced solid) for today, M10, M11, M15, M10 + M13, M10 + M11 + M13 + M15, M14 and M10 + M14.

**A band through `ray_progress`** (`tools/acoustic/band_probe.cpp` linked against each copy, `tx/band.py`: the 13
fitted constructions, sheet and traced solid, tracers 4 and 5, 0 / 30 / 60°, one band 125 Hz to 4 kHz against each
frequency alone, summed and with `thin_slab_threshold` 1): every value of the band equals its single frequency bitwise
for every copy (1872 values each); against the formula within 0.0022 to 0.0056 dB (the probe's float palette, `attB`
taken as 0 in both); **R + T at most 0.99326** (summed and followed alike: a held plate passes in one path). Today's
block-wall fit (`m` 0.68) has an `fRef` of 2.7e-69, which is no float, and is left out of today's band run.

**M10, `MASS_FOLD` and section 7.** With one slope the fold is the plain sum of the equivalent thicknesses: no
exponent, no exclusion, every bonded mass-law body folds. What it does to the plate-depth table of *Bonded mass - the
plate at any depth where the slopes agree* (`over0`, the plate's level over the rule, read from today's library on
the case with every `m` set to 20, which is what M10 makes of every table; `tx/r10`):

| stack | today | under M10 |
| --- | --- | --- |
| `A20 100 \| B30 50 \| C20 100`, light, as named / from the other side | +7.66 / +6.78 dB | +1.32 / +0.0045 dB |
| `A20 100 \| B20 100 \| C30 50`, heavy, from the other side | +4.30 dB | +0.0070 dB |
| `A20 100 \| B30 50 \| C20 100`, heavy, as named | +2.20 dB | +1.77 dB |
| `A20 100 \| B20 100 \| C20 100`, heavy, as named (one slope today) | +1.77 dB | +1.77 dB |

The mixed-slope limitation of section 7 closes (a table cannot give two bonded bodies two slopes), and so does the
`0.301·(20 − m)` offset of 2.2. What stays is the one-level-deep entry face (section 7: +1.77 dB on a three-body stack
as named, within the 3.01 dB bound), which has nothing to do with the slope. The quantization of the code stays 0.044 %
per step (0.0038 dB at slope 20); the code's 100 m end stays.

### 0.5 The fixtures, proposed

References in the tests from independent formulas: the thin-plate law of `md/acoustic.md` 3.4 (`τ(θ) = |1 + Z cos θ /
(2ρ₀c₀)|⁻²`, `Z = mω[j(1 − r² sin⁴θ) + η r² sin⁴θ]`) with each candidate's term, and the column definitions. Run times
from the plain copy's `--durations` run (4 OpenMP threads, load about 25 from this session's fits, so upper bounds):
every case named below takes 0.000 to 0.008 s except the two plate-depth cases (0.098 / 0.127 s);
the new cases are single panes through `bf::walk` like *Coincidence* (0.002 s) and stay under 0.05 s with a
1/96-octave sweep.

**M10** (if adopted):
- Restated: the seven failing cases of 0.4 (288 assertions). Their palettes keep `m` 30 / 12 and the references in the
  test take slope 20 (the `0.05·m` exponents become 1), so each table also guards that `m` is a switch; or every `m`
  set to 20 and one guard row `m` 30 = `m` 20 bitwise. *Bonded mass - the plate at any depth where the slopes agree*
  and its `ray_progress` twin lose "where the slopes agree" from their names: the mixed stacks become exact from the
  side the table names (bits 2 | 4 as the one-slope stacks), the `A20 100 | B30 50 | C20 100` rows on the entry-face
  bound. *Thin sheet - mass-law angle dependence* chose `m` 12 "so the angle slope is distinguishable"; it is restated
  to the statement that the angle slope is the mass law's at any `m`.
- Print-only cases whose rows move (no assertion): the two *Leak direction* cases, *Lossy verdict - … the mass-law
  part*, *Refracting tracer - … the mass-law part*, *ray_progress - the wall behind a traced layer …* (1213 rows in
  all, 0.4).
- New: *Mass law - m switches the limp mass law on*: a 4 mm pane (`fRef = (f_GHz/f_Hz)·ρ₀c/(πρ)` written in the test)
  as sheet and traced solid, types 4 / 5, 0° to 80°, 125 Hz to 4 kHz, against `10·log10(1 + (ωm″ cos θ/(2ρ₀c))²)` with
  the normalization of 2.1; `m` 12, 20, 30 bitwise equal.

**M12** (if adopted alone; inside M11 as code otherwise): no case moves. New: *Coincidence - coiA on a mass-law
material is not a loss*: the pane of *Coincidence* with `coiA` −3 and +3 equals `coiA` 0 bitwise, sheet and traced
solid; a wave medium with `coiA` +3 still pays its Lorentzian (from the column definition).

**M11 as code** (if adopted): *Leak reciprocity* takes out an interface gain of one for the two mass-law bodies (its
fixture computes 1 and 2 dB from `att`); *Total reflection of the outer pair - … the mass-law part* writes its
reference without the sheet's `att` (20 assertions). New: `att`, `attB` and `coiA` on a mass-law sheet and traced solid
change nothing (bitwise), the same columns on a wave medium still act.

**M14** (if adopted): *Thin sheet* restated past the limit (16 assertions). New: *Thin sheet - a finite plate has a
limiting angle*: past `θ_lim` the pane passes what it passes at `θ_lim` (thin-plate law at `max(cos θ, cos θ_lim)`),
sheet and traced solid, 0° to 89.5°.

**M15** (if adopted): *Coincidence* restated to the resistive form (6 assertions). New: *Coincidence - the plate's loss
is a resistance*: the 4 mm pane on a 1/96-octave sweep 2 to 8 kHz at 60°, sheet and traced solid, against the
thin-plate law of 3.4 [Davy_2009_SingleLeafCremer, Eq. 8] written in the test, and at coincidence `τ = 1/(1 + X₀ηc)²`.

**M13**: none proposed (0.6).

### 0.6 Recommendation

The answer to the brief's question: **the physical forms do not take over the jobs**. On the light panels the plate is
physical already (under M10 the level of glass and plywood is their mass within 0.85 to 1.08, gypsum's 0.60, and their coincidence
within 8 %); on heavy walls and linings one non-physical knob always carries the shape the infinite plate lacks (finite size,
coupling, the lining's resonance), and removing it moves the job to the next: the slope (today, M11), a flat `att` with
a level far below the mass (M10), a broad coincidence at 6 to 20 kHz with Q about 1 (every variant). M13, M14 and M15
change no variant's worst residual by more than 0.08 dB at 0–78° (single cells by up to 0.18 dB; one unconverged cell
aside, 0.2). **M10 and M11 each keep
the 13 constructions within the guideline; together they do not** (furred lining 3.33 dB in measured octaves, with
every physical form added 3.25 to 5.23 dB).

| candidate | recommendation | why (physics, code, residuals) | closes | replaces |
| --- | --- | --- | --- | --- |
| M10 `m` = 20 | **adopt** | the slope of a plate is the mass law's; light panels then fit with their mass and coincidence; −6 lines, `MASS_FOLD` without exponent or exclusion; 2.80 dB worst (gypsum model reference), 2.80 at 0–90° with M14 78°, 2.97 without | the mixed-slope limitation of section 7 (+7.66 → +1.32 dB, the rest is the entry face), the `0.301·(20 − m)` offset of 2.2, the rule that bonded bodies need one slope | the fitted slope of the mass term (since 231187f) and the slope condition of `MASS_FOLD` (dbbf67d); dbbf67d's fold itself stays |
| M11 no `att` with `m` | **drop** (if M10 is adopted) | needed by nothing alone (1.51 dB), but with M10 the furred lining leaves the guideline (3.33 dB); `att` stays the declared empirical level of the constructions the infinite plate does not describe (report 3.1) | — (A5 stays a rule of use) | — |
| M12 `coiA` ignored with `m` | **adopt** | one condition, no row moves, no case reaches it; the rule of use `coiA = 0` becomes code | N4's rule of use | nothing (N4's rule, 4720a92) |
| M13 installed loss factor | **drop** | no gain anywhere (≤ 0.09 dB per variant; fitted `C` 0 on light panels, 0.2 to 3.9 on walls, not Davy's); the fixed law needs the acoustic mapping's constants in the library, the column touches the OBJ writer, the MATLAB and Python readers and the format document outside it | — | — |
| M14 limiting angle | **keep as a proposal**, decide with the fit convention below | physics: a finite plate radiates no grazing wave [Northwood_1968_PlasterboardWalls]; +6 lines, 1 case; acts only on the 0–90° average a room sees (1.78 → 1.17 today, 2.97 → 2.80 under M10, at 78°) | — | — |
| M15 bending loss as a resistance | **keep as a proposal** | the right form [Davy_2009_SingleLeafCremer, Eq. 8], 2.3 dB of mass term (0.70 dB of T) on the *Coincidence* pane; neutral on the fits (≤ 0.02 dB, `coiQ` absorbs it); +5 lines through nine declarations (a complex `X` in the run); worth it only if the parameter session sets `coiQ` from measured loss factors | — | the reactance modulus of M4a (4720a92) |

If Stephan prefers M11 to M10 (it closes A5, +2 lines, and as code takes M12 with it): M11 alone fits within 1.51 dB,
the slope stays non-physical on every construction (glass 13 to 16, units 31 to 39), and the mixed-slope limitation
stays. Both together are not possible on the 13 constructions without a new knob.

**A finding outside the six candidates, for section 6 or the parameter session: the fit convention.** Round 1's 13
fits (report 3.6) and every fit here at 0–78° are fits of the field-incidence average; a ray tracer in a room averages
0–90°, where the same columns pass up to 15.3 dB more in an octave (concrete, brick 203, block, the linings; the
glazing 3.8 to 7.1 dB). Refitted at 0–90° the 13 constructions reach 1.78 dB today and 1.17 dB with M14 at 78°. Which
average the parameter session fits to has to be decided; the library's own is 0–90°. Converged into M14 as its
acceptance (no new item), unless Stephan wants it apart.

Stop for Stephan's decisions: M10, M11, M12, M13, M14, M15, and the fit convention.

### 0.7 Rig and traps

`build_linux/scratch/tx/README.txt` lists every file. Instruments worth keeping in `tools/acoustic` if Stephan agrees:
`txlib.py` + `txfit.py` (the formula copy of today's mass term and plate rule and the transmission fit with the
candidate switches, selection at 768 × 12), and `cases.py` (moved rows and failed assertions per case between two
`--durations` runs).

Traps met (in the README too): the level bound of a fit must reach the heaviest wall (`X₀(1 kHz)` 3200 for 423 kg/m²;
a first run at 10^3.5 clipped brick 203 and the furred lining under M10); a fit polished or selected on a coarser grid
than the residual's picks vectors that alias it (5.71 against 3.33 dB); round 1's multi-start Nelder-Mead was not
converged (today's knobs 1.23 dB, not 2.51); `conf/cols.tsv` of round 1 holds the unbounded fits; today's block fit
(`m` 0.68) has an `fRef` (2.7e-69) no float palette holds; `ray_progress` refuses a `thin_slab_threshold` above 1;
Python 3.14's multiprocessing starts a forkserver, whose workers survive a kill of the main process (kill the
forkserver's children by pid).

## 1. Phase 1 — fixtures (2026-10-17)

Decisions (Stephan, 2026-10-17): M10 adopted; M11 and M13 dropped; M15 a proposal; M14 not in this round (a candidate,
open pending a first-principles study of grazing incidence); fit convention 0–90°; M12 replaced by D2-b (`coiA`
removed altogether). Nothing under `src/` touched; the fixtures below run against today's library.

### 1.0 M10 at 0–90°: the fit for `md/acoustic.md` 2.7

The 13 constructions under M10 (`m = 20`, `att` kept, no M14) on the library's random-incidence average, 0–90°;
procedure of 0.2 (`tx/batch.py` variant `M10_90`). 18 DE seeds per cell, split into two halves of 9, each half with its
own selection of every fitted vector at 768 × 12. Worst residual **2.97 dB** in both halves (0.2: 2.97); one cell differs
between the halves by more than 0.05 dB (the furred lining, 2.67 / 2.73). Every measured construction stays within
±3 dB; the 2.97 dB is the gypsum board, a model reference (the spread in every octave), at 31.5 Hz and 1 / 2 / 4 kHz.

| construction | worst | residual per octave 31.5 Hz … 8 kHz (dB) | `m″` level (of mass) | `att` / `attB` | `coiF` Hz / `coiQ` | `res` Hz / Q / sign | `fRef` (GHz) | `tf` at `fRef` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glass 4 mm | 0.97 | +0.79 0 −0.88 +0.97 +0.19 −0.94 +0.24 +0.96 −0.72 | 14.2 (1.42) | 0.01 / −0.10 | 3088 / 299 | 83 / 0.88 / − | 3.2454e-05 | 6.148 |
| glass 6 mm | 2.41 | +0.89 0 +2.36 +1.66 −1.05 −2.36 +2.36 +2.41 −0.52 | 16.1 (1.07) | 0.52 / −0.82 | 2040 / 300 | 50 / 48.4 / − | 4.2898e-05 | 4.003 |
| gypsum 12.5 (model) | 2.97 | +2.97 +0.75 0 0 −1.85 −2.96 +2.96 +2.97 +1.03 | 6.2 (0.72) | 1.16 / −0.69 | 2836 / 300 | 31 / 17.6 / − | 2.3086e-04 | 7.878 |
| plywood 18 (model) | 0.58 | 0 0 0 0 −0.58 +0.57 −0.57 +0.54 +0.58 | 13.5 (1.25) | 0.93 / +0.66 | 1562 / 250 | 713 / 50 / + | 1.5283e-04 | 0.2212 |
| concrete 152 | 0.59 | 0 0 −0.59 +0.58 −0.59 +0.59 −0.52 +0.19 0 | 71.6 (0.21) | 12.00 / −0.02 | 6315 / 1.1 | 65 / 5.0 / − | 2.4403e-04 | 2.851e-04 |
| brick 95 | 0.18 | 0 −0.17 +0.17 −0.17 +0.18 −0.14 −0.12 0 −0.18 | 110.1 (0.58) | 3.09 / +1.04 | 5582 / 70.7 | 177 / 6.0 / + | 9.946e-05 | 3.729e-03 |
| brick 203 | 0.57 | +0.55 0 −0.54 +0.55 −0.55 +0.57 −0.46 +0.35 0 | 58.7 (0.14) | 10.16 / +0.12 | 18404 / 1.6 | 127 / 1.06 / − | 3.9743e-04 | 5.421e-03 |
| block 190 | 0.92 | −0.85 0 +0.85 −0.83 +0.92 +0.59 −0.38 −0.41 0 | 188.0 (0.84) | 0.09 / +2.98 | 6075 / 30.4 | 247 / 1.68 / + | 1.1615e-04 | 2.786e-03 |
| 4-16-4 (AGC) | 1.55 | 0 0 +1.54 −0.75 +1.55 −1.55 −1.29 +1.55 0 | 12.2 (0.61) | 4.74 / +0.52 | 3195 / 300 | 220 / 50 / − | 7.5459e-05 | 1.545 |
| 4-12-4-12-4 (AGC) | 2.69 | 0 0 +2.68 −1.27 +2.68 −2.68 −2.34 +2.69 0 | 17.2 (0.57) | 6.37 / +0.54 | 3020 / 300 | 195 / 50 / − | 8.0296e-05 | 1.511 |
| brick 203 + furred board | 2.67 | +2.64 +1.67 +2.62 +1.60 −1.69 −2.67 −2.31 +2.66 0 | 8.2 (0.02) | 12.00 / −0.37 | 30 / 1.1 | 169 / 26.5 / + | 2.8445e-03 | 0.6491 |
| SCR 140 + board on dabs | 1.73 | 0 0 +1.69 −1.70 +1.69 −1.66 +1.73 +1.26 0 | 153.8 (0.55) | 12.00 / +0.90 | 2740 / 292 | 211 / 3.85 / − | 1.046e-04 | 1.342 |
| brick 105 + plaster one face | 0.03 | 0 0 −0.02 +0.02 −0.02 +0.03 0 +0.02 0 | 40.1 (0.19) | 11.62 / +0.10 | 3653 / 1.9 | 116 / 0.98 / − | 3.0054e-04 | 6.262e-03 |

(`thk` the construction's, `tfB` the class's absorption law of report 3.6; `att` in dB at 1 kHz; a 0 in the residuals is
a band inside its target or without one.) Light panels take 0.72 to 1.42 of their mass and their coincidence within
1 to 14 % (glass 3088 / 2040 against 2989 / 1993 Hz, gypsum 2836 against 2500), plywood 36 % above (1562 against 1146
Hz); the units take 0.57 to 0.61 of their mass and 4.7 to 6.4 dB of `att`; the heavy walls and linings take a level of
0.02 to 0.84 of their mass, `att` up to the 12 dB bound (three cells at it), and concrete, brick 203 and plaster broad
coincidences (Q 1.1 to 1.9) at 3.7 to 18 kHz, the furred lining one at the 30 Hz bound (0.2: the job of the finite
plate). With the `att` bound at
40 dB (variant `M10_90w`, 18 seeds, the halves agree within 0.05 dB in every cell) the worst residual stays 2.97 dB
(gypsum); the heavy walls fall (furred lining 1.66, dabs 1.17, block 0.38, brick 95 0.03 dB) by taking 16 to 37 dB of
`att` at a level of 0.02 to 0.15 of their mass. The 12 dB-bounded fit above is the one to quote: the same worst, and
less of a level the walls do not have.

Correction to 0.4: under M10 the failing assertions (288) are in **six** cases, not seven; the seventh case of the
restatement is the `ray_progress` plate-depth twin, which only prints its rows.

### 1.1 What the fixtures state

**M10** (references at slope 20; the palettes keep `m` 30 / 12, so each table guards that `m` is a switch):

- `ray_test_helpers.hpp`: `diag_mass_TL` and the plate of `diag_sc_stack` take `X₀ = (f/fRef)·M` and `A = 20·log10 X₀`
  where `m > 0` (an `m = 0` layer is untouched, so no `[wave]` reference moves); `test_acoustic_state_update.cpp`:
  `diag_bonded_TL` the same.
- *Stack with a mass-law layer - resolved by segments, not killed*: the section "m 20 against m 10" now states that the
  `m` 10 layer pays exactly what `m` 20 pays (`T20.gain == T10.gain` bitwise, the reference's difference zero, the
  term worth more than 1 dB).
- *Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate*, *Bonded mass - a mass-law sheet
  inside a mass-law wall*, *Unentered branches - a sheet without the mass column of its mass-law host's permittivity is
  merged*: comments; their references move through the helpers.
- *Bonded mass - the plate at any depth* (was "… where the slopes agree"): the text, and the former mixed stacks assert
  what their one-slope twins assert: `A20 100 | B20 100 | C30 50` and `A20 100 | B30 50 | C20 100` 2 | 4, `plaster 15 |
  brick30 115 | plaster 15` 4 | 8, `lining L30 > …` 2 | 4, `B20 60 | S20 T30 | B20 40` and `B20 60 | S30 T20 | B20 40` 3.
- *ray_progress - the bonded plate at any depth* (was "… where the slopes agree"): the plate reference at slope 20, the
  `C30` stack exact as named (2), the state-depth bound of 3.02 dB on every other row (the `C30` exclusion gone).
- *Thin sheet - mass-law angle dependence*: `qd_mass_TL`, `A0`, `X0` and the normal-incidence check at slope 20; `m` 12
  passes what `m` 20 passes, bitwise, at 0 / 30 / 60 / 85°; past the knee the term at 89.9° is 0.20 dB (`x` 0.22), so
  the bounds there are 0.25 dB and the cosine-square ratio within 0.1 (were 0.05 / 0.05 for slope 12).
- New: *Mass law - m switches the limp mass law on*: a 4 mm pane with `fRef` written from its density (2500 kg/m³,
  `ρ₀c` 1.204·343), sheet and traced solid, types 4 / 5, 125 / 500 / 2000 / 4000 Hz, 0 / 30 / 60 / 80°, against the
  plate rule (`tf` 0.05) of the limp mass law `20·log10 X` normalized in angle, `X = ωρ·thk/(2ρ₀c)`, within 0.01 dB;
  `m` 12 and 30 equal `m` 20 bitwise.

**D2-b** (the decided wave change):

- *Acoustic - interface_gain is the penetration loss alone; a coiA column is ignored* (was *Acoustic - interface_gain
  penetration and coincidence*, `[wave]`): a table with `coiF`, `coiQ` and `coiA` −6 / +6 / −12 on `att` 10 / 3 / 2 dB
  gives the `att` power law at 1 / 2 / 4 kHz (coincidence at 2 kHz), against the closed form and bitwise against the
  table without the three columns; the `attB` check kept.
- *Ray-Mesh Interact - Public helpers: refractive_index, interface_gain, medium_gain* (`[wave]`): the coincidence
  columns (now `coiA` +6) leave the index and, new, the interface gain at `coiF` alone.
- *ray_state_update - a resonance shifts the slab factor, coincidence columns leave a wave medium alone* (was
  *ray_state_update - resonance and coincidence material columns*, `[wave]`): the coincidence section asserts the exit
  at coincidence equals the exit without the three columns, bitwise (was: accepted without error).
- *Coincidence - on a mass-law material coiF and coiQ are the bending stiffness of the plate*: the `coiA` 0 column and
  its comment removed.
- New guard, *Coincidence column - a table that carries coiA loads and the column changes nothing* (`[wave]`) and its
  *- the mass-law part* (`[acoustic]`): a material table written as CSV with a `coiA` column is read by
  `obj_file_read` (table only), loads, and hands the column on; a wave medium (`eps` 2.5, `c` 0.02, `att` 3 dB) and a
  4 mm mass-law pane (`coiF` 2989 Hz, `coiQ` 66.7) as traced solid and as sheet, EM type 1 (both channels) and types
  4 / 5, 0 / 30 / 60° (59.8° for the pane), at the coincidence frequency, give `T` and `R` bitwise equal to the same
  table without the column, and the interface gain likewise.
- The OBJ header checks of `test_obj_file_write.cpp`, `tests/python_tests/test_obj_file_write.py` and
  `tests/mex_tests/test_obj_file_write.m`: the canonical header and the default row without `coiA`.

### 1.2 Against today's library

Whole suite 1143 cases (1140 + 3 new), 1314857 assertions, 886 failed in 13 cases; `[wave]` 180 cases (179 + 1), 3
failing; `[acoustic]` 56 (54 + 2), 9 failing. Per case (failed assertions; printed rows that differ from the baseline):

| case | failed | rows | why |
| --- | --- | --- | --- |
| *Bonded mass - the plate at any depth* | 280 | 1014 | references at slope 20 against today's `m` 30 bodies; the former mixed stacks now asserted |
| *Mass law - m switches the limp mass law on* (new) | 256 | 194 | the `m` 12 and 30 rows (each 0.01 dB and bitwise); the `m` 20 rows pass today |
| *ray_progress - the bonded plate at any depth* | 88 | 533 | as the plate-depth case |
| *Acoustic - interface_gain is the penetration loss alone; a coiA column is ignored* | 54 | 0 | today's Lorentzian |
| *Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate* | 52 | 45 | `m` 30 board and concrete |
| *Coincidence column - … changes nothing* (new, `[wave]`) | 38 | 0 | today's Lorentzian on the wave medium |
| *Thin sheet - mass-law angle dependence* | 36 | 16 | `m` 12 against slope 20 |
| *Bonded mass - a mass-law sheet inside a mass-law wall* | 28 | 16 | `m` 30 bodies |
| *Coincidence column - … - the mass-law part* (new) | 19 | 0 | today's Lorentzian on the pane (`T`; `R` passes) |
| *Stack with a mass-law layer - resolved by segments, not killed* | 16 | 19 | the `m` 10 layers |
| *Unentered branches - a sheet without the mass column of its mass-law host's permittivity is merged* | 16 | 24 | `m` 30 wall |
| *Ray-Mesh Interact - Public helpers: refractive_index, interface_gain, medium_gain* | 1 | 0 | interface gain at `coiF` |
| *Test OBJ File Write - CSV columns, defaults and validation* | 1 | 1 | the header with `coiA` |

The rows are those the restated cases print (references and their differences). Every other line of the three
baselines is unchanged: no other case differs, `[wave]` outside the three cases is identical, and
`tests/data/wave_twins.tsv` is untouched. Two restated cases pass today: *ray_state_update - a resonance shifts the
slab factor, coincidence columns leave a wave medium alone* (`ray_state_update`'s exit does not take the interface gain,
so `coiA` never reached it; the section is a guard now) and *Coincidence* (its `coiA` was 0).

**Under a what-if of the decision** (the M10 copy of 0.4, the Lorentzian out of `interface_gain`, and
`src/obj_file_write.cpp` without `coiA` in its canonical list; `tx/tb_cand`): all 1143 cases pass (1314857 assertions),
`[wave]` rows identical to the baseline. The print-only cases that phase 2 will move (no assertion): *Leak direction - a
wave held in a mass-law body keeps its direction and a closed face hands on the wave's* 288 rows, *ray_progress - the
wall behind a traced layer is met at the incidence angle at every frequency of a band* 198, *Leak direction - a closed
face hands the wall behind a traced layer the direction the wave has in air* 112, *Lossy verdict - a lossy second medium
has no critical angle - the mass-law part* 32, *Refracting tracer - a traced bend is known from the state and the leg is
exact - the mass-law part* 10: each prints levels of an `m` 30 wall.

### 1.3 Python and MATLAB

- Python (`build_linux/scratch/api`, module rebuilt from today's source; `pytest tests/python_tests` from a scratch
  directory): 761 passed, 1 skipped, **1 failed**: `test_obj_file_write.py::test_csv_columns_defaults_validation`, the
  restated header (today's writer emits `coiA`).
- MATLAB (`build_linux/scratch/api12`, MEX rebuilt with g++-12; MOxUnit over `tests/mex_tests`, 66 files, from a scratch
  directory): 63 passed, **3 failed**: `test_obj_file_write:185`, the restated header; and two that this round did not
  touch: `test_qrt_file_read:9` opens `data/test.qrt` relative to `tests/` (a run from a scratch directory cannot find
  it), and `test_ray_mesh_interact:113` expects 0.69639 for an EM type-2 transmission where today's library gives
  0.98485 (the test file is unchanged since before this round; recorded, not in scope). MATLAB ended with a segmentation
  violation after printing the results (at exit); the results were complete.

### 1.4 Run times

Idle machine, `--durations yes`, all threads; 1-min load before / after: `[wave]` 2.75 / 8.28, `[acoustic]` 8.28 / 7.78,
whole suite 7.78 / 9.29 (58 s; the rise is the suite's own threads). The cases of this phase: *ray_progress - the bonded
plate at any depth* 0.077 s, *Bonded mass - the plate at any depth* 0.059 s, *Stack with a mass-law layer* 0.011 s,
*Mass law - m switches the limp mass law on* 0.008 s, *Bonded mass - sheet on sheet …* 0.006 s, *Thin sheet* 0.005 s,
the `[wave]` guard 0.002 s, the mass-law guard 0.001 s, every other restated case ≤ 0.002 s. No case near 0.5 s.

### 1.5 For phases 2 and 3

- `python3 tools/acoustic/check_docs.py` now reports the two new `[acoustic]` cases as missing from `md/acoustic.md`
  (phase 3); the renamed cases are matched by their old names' prefixes and need their names in the documents too.
- `src/quadriga_lib_material_helpers.hpp` is referenced by nothing in the repository and the install rule copies
  `include/` only: it can be deleted without a build change.
- `coiA` also stands in generated documentation (`html_docu/cpp_api.html`, `html_docu/formats.html`,
  `html_docu/quadriga_lib_api_cpp.md`, `html_docu/quadriga_lib_material_model.md`), which regenerates from the sources;
  outside the decided list, for Stephan.

Stop for Stephan's review of phase 1.

## 2. Phase 2 — source (2026-10-17)

Phase 1 accepted (Stephan, 2026-10-17). `tests/mex_tests/test_ray_mesh_interact.m` stays as it is (the MATLAB wrapper and
its test will be rewritten later); its failure at line 113 and `test_qrt_file_read`'s run-location failure are known.

### 2.1 What changed

| file | change | lines |
| --- | --- | --- |
| `src/ray_mesh_interact.cpp` | M10: `mass_X0` returns `(f/fRef)·M·res·coi` (no slope exponent); `MASS_FOLD` sums `dᵢ·fRef_R/fRefᵢ` over every bonded mass-law body (no exponent, no slope exclusion); the column comments, `require` comment, `Material::medium_gain` comment, the thickness-code comment and the public `medium_gain` text ("`m > 0` switches the mass law on, 20 dB per decade"). D2-b: `coiA` out of `MaterialCols` (pointer, resolution), `Material` (member, copy), `same_as` and `interface_gain` (the Lorentzian); the public `interface_gain` text (the penetration loss alone; `coiF` / `coiQ` the bending stiffness of a mass-law plate; a `coiA` column of an older table read and ignored) and the `refractive_index` text; one comment ("interface_gain (att)") | 5116 (−16 against 5132; 34 in, 50 out) |
| `src/obj_file_write.cpp`, `api_python/qpy_obj_file_read.cpp`, `api_mex/obj_file_read.cpp` | `coiA` out of the canonical column list / key order (a `coiA` column of a table is now an extra column, written last and handed on like any other) | 1 each |
| `src/quadriga_lib_material_helpers.hpp` | deleted (228 lines; included by nothing, not installed: the install rule copies `include/`) | −228 |
| `tools/acoustic/band_probe.cpp`, `band_probe_example.in`, `README.md` | the column list and header comment without `coiA` (23 columns per material), the example's three materials without their `coiA` value, the README's column count | 2 / 1 / 1 |

The deletion is in the working tree only (not staged); nothing is committed.

### 2.2 The suite

Idle machine, `--durations yes`, all threads (1-min load before / after): `[wave]` **165258 assertions in 180 cases**,
all passed (0.41 / 14.66); `[acoustic]` **40050 in 56**, all passed (14.66 / 14.66: the `[wave]` run's own threads);
whole suite **1314857 in 1143**, all passed, 57 s (14.66 / 6.84).

- `[wave]`: the printed output equals `test_wave.txt` of the session start line for line, except the count line (179 →
  180 cases, 165110 → 165258 assertions); the three restated `coiA` cases and the new guard print no rows.
  `tests/data/wave_twins.tsv` is unchanged (asserted by that run, untouched in git).
- Every moved row against the session baseline (`tx/cases.py`, printed rows): the restated cases *Bonded mass - the plate at
  any depth* 1014, *ray_progress - the bonded plate at any depth* 533, *Mass law - m switches the limp mass law on* 194
  (new), *Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate* 44, *Unentered branches - a sheet
  without the mass column of its mass-law host's permittivity is merged* 24, *Stack with a mass-law layer - resolved by
  segments, not killed* 18, *Thin sheet - mass-law angle dependence* 17, *Bonded mass - a mass-law sheet inside a
  mass-law wall* 16; the five print-only cases *Leak direction - a wave held in a mass-law body keeps its direction and a
  closed face hands on the wave's* 288, *ray_progress - the wall behind a traced layer is met at the incidence angle at
  every frequency of a band* 198, *Leak direction - a closed face hands the wall behind a traced layer the direction the
  wave has in air* 112, *Lossy verdict - a lossy second medium has no critical angle - the mass-law part* 32,
  *Refracting tracer - a traced bend is known from the state and the leg is exact - the mass-law part* 10. The `coiA`
  cases (*Acoustic - interface_gain is the penetration loss alone; a coiA column is ignored*, *Ray-Mesh Interact - Public
  helpers*, *ray_state_update - a resonance shifts the slab factor, coincidence columns leave a wave medium alone*, both
  guards) and the OBJ header case print no rows. No other case differs.
- The built library's run equals the phase-1 what-if run (`tx/tb_cand`) line for line: 0 differing lines.
- The band probe's example (`tools/acoustic/band_probe_example.in`) gives output identical to the probe before the change
  on the library before the change.

### 2.3 Python and MATLAB

- Python module rebuilt (`build_linux/scratch/api`, target `quadriga_py`); `pytest tests/python_tests` in full: **762
  passed, 1 skipped, 0 failed** (the header without `coiA` included).
- MEX files rebuilt with g++-12 (`build_linux/scratch/api12`); MOxUnit on the tests that read or write materials:
  `test_obj_file_write` (the header without `coiA`), `test_obj_file_read`, `test_calc_diffraction_gain`,
  `test_arrayant_export_obj_file` **passed**; `test_ray_mesh_interact` fails at line 113, known (above).

### 2.4 Energy and band checks on the built library

The 13 fits of 1.0 (M10 at 0–90°) as library columns, through the built library:

- Walker (`tx/conf.py`, probe rebuilt against the new `libquadriga.a`; 96 angles 0–90° × 3 frequencies per third, type 4,
  202176 rows): sheet and formula copy within 0.0000 dB, traced solid within 0.0018 dB, so sheet and traced solid agree
  within 0.0018 dB on every construction. (The walker's residual on that sparser grid reads 3.62 dB where 768 × 12 reads
  2.97: the sampling of 0.2, not the library.)
- Band through `ray_progress` (`tools/acoustic/band_probe.cpp` rebuilt; sheet and traced solid, tracers 4 and 5, 0 / 30 /
  60°, one band 125 Hz–4 kHz against each frequency alone, summed and with `thin_slab_threshold` 1): every value of the
  band equals its single frequency bitwise (1872 values); within 0.0041 dB of the formula (float palette); **R + T at
  most 0.99324**, summed and followed.

### 2.5 For phase 3

- `python3 tools/acoustic/check_docs.py` reports the two new `[acoustic]` cases until `md/acoustic.md` names them; the
  baselines `test*.txt` are still those of 2c5be17 and are regenerated in phase 3.
- Generated documentation (`html_docu/`) still describes `coiA` and the slope column until it is regenerated.

Stop for Stephan's review of phase 2.

## 3. Phase 3 — documents (2026-10-17)

Phase 2 accepted (the manager reran `[wave]` 165258 / 180 and `[acoustic]` 40050 / 56).

### 3.1 What changed

**`md/acoustic.md`** (1316 lines; 149 in, 68 out):

- 2.1: the mass term as `X₀ = (f/fRef)·M`, `A = 20·log10 X₀`, "`m > 0` switches the mass law on, 20 dB per decade"
  (M10, with the former fitted slope as history); *Mass law - m switches the limp mass law on* named; the rule of use
  `coiA = 0` replaced by the removal of `coiA` (a column of an older table read and ignored, the guard named).
- 2.2: `MASS_FOLD` as the plain sum over every bonded mass-law body; the `0.301·(20 − m)` offset and the slope exclusion
  gone; the fixtures' `m` 30 / 10 / 12 bodies as switch guards; the two plate-depth cases under their new names and
  *Stack with a mass-law layer* added.
- 2.6: the closing rule of use without `coiA`.
- 2.7: the column table with the knobs of 1.0 (M10 at 0–90°, 2.97 dB: the light panels at their mass and coincidence,
  the units at 0.57 / 0.61 of their mass with 4.7 / 6.4 dB of `att`, the heavy walls and linings at the knob bounds:
  `att` to 12 dB, broad coincidences at 3.7 to 18 kHz); the slope row became "the mass law, `m > 0` a switch".
- 3.4: the mass-law row (the limp mass law, the new case), the coincidence row (2.97 dB at 0–90°, M15 a proposal),
  the installed-loss row (M13 dropped), the finite-size row (`att` and level absorb it, M14 open).
- 4: M10 measured, decided and built; M11 measured and dropped (with M10 the furred lining 3.33 dB); M12 superseded by
  D2-b, which is described with what was built; M13 measured and dropped; M14 measured and open pending the study of
  grazing incidence, with the numbers of 0.2 (4.3 % of the diffuse weight at 78–90°, up to 15.3 dB per octave on round
  1's columns at 0–90°, 1.78 / 1.17 dB refitted without / with a 78° limit); M15 measured, a proposal until the
  parameter session decides where `coiQ` comes from.
- 5: "mixed slopes" out of the state row.
- 7: the mixed-slope limitation closed with the numbers of 0.4 (+7.66 → +1.32 dB, the rest the one-level entry face);
  the quantization item at slope 20.
- 8: the helper header closed (deleted); the fitted-knobs item with the converged fits (1.23 dB at 0–78°, round 1's
  2.51 an unconverged fit), the job moving from knob to knob, and the state under M10.
- 9: the rule that a transmission reference is fitted on the library's 0–90° average, with the numbers, and the
  selection at 768 × 12.
- 10: the decisions of 2026-10-17 and "Built (2026-10-17)" with the line count and the counts; N4's rule of use marked
  superseded.
- 11: the new instruments in `tools/acoustic` and the scratch rig `tx`.
- Head and appendix: the `[acoustic]` count 40050 / 56; "16 of them are … - the mass-law part" (the document said 18,
  which was stale already at 2c5be17: 15 there, 16 now); the two new cases and the two renamed ones.

**`md/wave.md`** (23 in, 11 out): the counts of section 0 (165258 / 180, 40050 / 56, 1314857 / 1143); section 4's
`att` item with the restated case name; a new item "Coincidence columns on a wave medium" (none act; `coiA` removed on
2026-10-17, the only wave change of the round; the guard, *Ray-Mesh Interact - Public helpers …* and *ray_state_update
- a resonance shifts the slab factor, coincidence columns leave a wave medium alone* named).

**`tools/data_formats_md/format_materials.md`** (19 in, 20 out): `coiA` out of the interface-transmission list and the
parameter table; `att(f) = att·(f/fRef)^attB`, clamped at 0; the `m` row "mass-law switch (> 0: the mass law, 20
dB/decade)" and the mass formula `20·log10((f/fRef)·M)` from the normal thickness with its angle shape; coincidence as
the bending stiffness of a mass-law plate (no interface loss; a `coiA` column read and ignored); the acoustic table row
and the notes that called `m` a slope and put the level in `a`. The rest of that document (written before round 1: the
plate rule, `tf`, the two-parameter mapping) is not brought up to date here.

**`tools/acoustic`**: `txlib.py` (103 lines; the formula copy of the built library: slope 20, no `coiA`; switches for
the old fitted slope and the candidates M11, M13, M14, M15; checked again through the walker on the built library:
T within 0.0022 dB, R within 0.0005 dB, types 1 / 4 / 5), `txfit.py` (318 lines; the fit of section 9 with a command line,
defaulting to the library at random incidence; it reproduces the 1.0 fits' residuals exactly and a one-seed run on
glass 4 mm reads 1.04 dB against 0.97 with 18), `tl_refs.json` (35 lines; the 13 constructions, absorption laws and
physical constants of report 3.6, so the tool needs no scratch rig), `cases.py` (51 lines); four lines in the README's
table and a usage section for each.

### 3.2 Baselines and checks

The three baselines regenerated from the repository root (load 0.81 before, 4.13 after): `[wave]` 165258 assertions in
180 cases, `[acoustic]` 40050 in 56, whole suite 1314857 in 1143, all passed; `tests/data/wave_twins.tsv` unchanged.
`python3 tools/acoustic/check_docs.py`: all checks pass (every `[acoustic]` case named, the quoted counts equal the
baselines).

### 3.3 The round, in sum

| | before (2c5be17) | after |
| --- | --- | --- |
| `src/ray_mesh_interact.cpp` | 5132 lines | 5116 |
| `src/quadriga_lib_material_helpers.hpp` | 228 lines | deleted |
| `[wave]` | 165110 / 179 | 165258 / 180 (three cases restated, one new; no other row moved) |
| `[acoustic]` | 39371 / 54 | 40050 / 56 |
| whole suite | 1314030 / 1140 | 1314857 / 1143 |

Nothing is committed; the deletion of the helper header is in the working tree only.

Stop for Stephan's review of phase 3.
