# M6 report — the room-side face as a series impedance (A12)

Brief: `md/m6_brief.md`. This file holds one section per phase. Phase 0 (analysis in scratch) is below; phases 1 to 3
follow Stephan's decisions. Scratch rig: `build_linux/scratch/m6` (section 0.9 lists it; scratch is not kept, every
number used here is stated here).

## 0. Phase 0 — analysis (2026-10-16)

### 0.0 Baseline check

- `git status`: only `md/acoustic_manager.md` (modified) and `md/m6_brief.md` (new); nothing under `src/`, `tests/` or
  `md/acoustic.md`. Source state 238444d, last commit 4035ac4. `src/ray_mesh_interact.cpp` 5089 lines.
- `cmake --build build_linux --parallel` up to date; the three runs of `md/wave.md` 0: `[wave]` 165110 assertions in 179
  cases, `[acoustic]` 36813 in 52, whole suite 1311472 in 1138, all passed, identical to the stored baselines (only the
  seed line differs). Load before 0.44, after 6.95 (the whole suite itself).
- Plain what-if copy (`plain.cpp`, `md/wave.md` 10) reproduces all three baselines; only the seed line differs.
- The switch copy (`wi.cpp`, section 0.3) with every switch off reproduces `[wave]` and `[acoustic]` (seed line only).

### 0.1 The reference (item 1)

All formulas in `m6lib.py` (no library code). Pressure, scalar, `e^{+jωt}` (the library's: a path carries
`exp(−jkL)`, a mass reactance is `+jωm″`; checked on the probe, 0.4). A locally reacting face of normalized
impedance `ζ` reflects `r = (ζ c − 1)/(ζ c + 1)`, `c` the cosine in air. M6's `ζ = R′ + j(1 + R′)·|tfQ|·(f/tfF − tfF/f)`,
`R′ = ((2 − tfP) ± 2√(1 − tfP))/tfP` (+ over-damped, − under-damped). At 0° `1 − |r_ζ|²` is M5's law exactly.

**The fixture board** (`tf` 0.05, `tfP` 0.25 at 150 Hz, `tfQ` 0.75; over-damped `R′` 13.928, under-damped 0.0718;
D3 a, `|r|² = (1 − tf_bg)|r_ζ|²`):

| f, over-damped | \|r\| / phase 0° | 30° | 60° | 85° |
| --- | --- | --- | --- | --- |
| 63 Hz | 0.9354 / −3.72° | 0.9294 / −4.30° | 0.8978 / −7.45° | 0.6413 / −42.78° |
| 150 Hz | 0.8441 / 0.00° | 0.8254 / 0.00° | 0.7299 / 0.00° | 0.0942 / 0.00° |
| 250 Hz | 0.8973 / +4.07° | 0.8859 / +4.70° | 0.8264 / +8.17° | 0.4234 / +53.21° |
| 1 kHz | 0.9698 / +1.52° | 0.9690 / +1.75° | 0.9649 / +3.03° | 0.9210 / +17.27° |

Under-damped the same `|r|` at 0°, the phase −64.7° (63 Hz), 180.0° (150 Hz), +98.6° (250 Hz), +21.6° (1 kHz) at 0°,
and toward grazing `|r|` rises to 0.963 at every frequency (absorption falls). Fine sweep (1/96 octave, 20 Hz–2 kHz),
over-damped: phase −4.12° at 83.5 Hz to +4.12° at 269.1 Hz at 0°, −5.83° / +5.83° at 45°, −8.26° / +8.26° at 60°;
`|r|` minimum 0.8441 at 149.9 Hz (0°). Under-damped: the phase passes through ±180° at 150 Hz (−179.92° at 149.9 Hz,
+178.75° at 151.0 Hz). The sign at the peak separates the branches as the brief says: `r = +0.844` over-damped, `−0.844`
under-damped.

Absorption in angle and its averages (D3 a; D3 c and M5's flat pad alongside, 0.3):

| f | branch | α at 0° / 30° / 60° / 85° | Paris 0–90° | field 0–78° | M5 (flat) |
| --- | --- | --- | --- | --- | --- |
| 63 Hz | over | 0.125 / 0.136 / 0.194 / 0.589 | 0.183 | 0.169 | 0.129 |
| 150 Hz | over | 0.288 / 0.319 / 0.467 / 0.991 | 0.420 | 0.400 | 0.300 |
| 250 Hz | over | 0.195 / 0.215 / 0.317 / 0.821 | 0.290 | 0.272 | 0.202 |
| 500 Hz | over | 0.089 / 0.094 / 0.125 / 0.388 | 0.121 | 0.112 | 0.091 |
| 150 Hz | under | 0.288 / 0.259 / 0.177 / 0.074 | 0.214 | 0.220 | 0.300 |
| 250 Hz | under | 0.195 / 0.191 / 0.159 / 0.073 | 0.170 | 0.174 | 0.202 |

The Paris value of the over-damped board at its peak is 0.420 against `α₀` 0.288 (1.46 times), under-damped 0.214
(0.74 times). Jeong's bound is reproduced by the closed form used in 0.6 (0.951222 at `ζ` = 1.567; impedances 40 / 10 / 4
give 0.168 / 0.489 / 0.791, Jeong's Table I 0.17 / 0.49 / 0.79).

**A panel over a partitioned cavity** (`b1.py`, `b1b.py`): 12.5 mm board, 8.6 kg/m², mounting resistance
`R = 0.2·ωm″`, cavity `−jρ₀c₀·cot(k₀d)` (no cosine inside), `Z_s = R + jωm″ + Z_cav`, against the lumped stiffness
`−jρ₀c₀/(k₀d)`; max over 0°–85° (5° steps) and a 1/96-octave sweep of each octave:

| d | f₀ (cot / lumped) | ω₀m″/Z₀, R′(f₀), Q_α, tfP | max \|Δr\| lumped vs cot per octave 31.5 / 63 / 125 / 250 / 500 / 1k | Paris per octave cot − lumped, max |
| --- | --- | --- | --- | --- |
| 25 mm | 127.9 / 128.8 Hz | 16.93, 3.386, 3.86, 0.704 | 0.0001 / 0.0009 / 0.0029 / 0.0016 / 0.0006 / 0.0003 | 0.0002 |
| 50 mm | 90.4 / 91.1 Hz | 11.97, 2.394, 3.53, 0.831 | 0.0006 / 0.0057 / 0.0058 / 0.0023 / 0.0011 / 0.0006 | 0.0021 |
| 100 mm | 64.0 / 64.4 Hz | 8.47, 1.693, 3.14, 0.934 | 0.0035 / 0.0116 / 0.0066 / 0.0037 / 0.0021 / 0.0012 | 0.0016 |

The lumped stiffness holds within 0.012 in `|r|` (complex difference) and 0.0021 in the Paris value per octave to 1 kHz
on 25 to 100 mm; where `|r| > 0.5` the phase differs by at most 0.91°. The larger phase differences at the resonance
(2.4° / 15.8° / 40.6° on 25 / 50 / 100 mm where `|r|` is small) come from the resonance shift of 0.6 to 1.1 % (the
cavity's added mass `ρ₀d/3`), not from a different law. M6's `ζ` with a constant `R′` is the lumped form exactly when the
mounting resistance is constant (cot form with constant `R = 0.2·ω₀m″`: within 0.0117 in `|r|`, Paris equal within
0.0022 per octave). With `R = 0.2·ωm″` rising in frequency the cot form absorbs more above the peak than a constant
`R′` (Paris 0.086 against 0.053 at 250 Hz on 25 mm, 0.1072 in `|r|`): that is the choice of a resistance law, a
parameter matter, not the lumped stiffness. So M6's lumped form holds for partitioned cavities of 25 to 100 mm up to
1 kHz; the quarter-wave depth (858 Hz for 100 mm) does not show in `|r|` because the panel's mass reactance dominates there.

### 0.2 The call sites (item 2)

Every place the pad or `tf()` is read (line numbers of `src/ray_mesh_interact.cpp` at 238444d):

| site | where | what a peaked face meets there |
| --- | --- | --- |
| `Material::tf` (410), `apply_tf_pair` (424) | the law and the pair rule | read by the three below |
| `Material::tf_face` (444) | a face with a wave on both sides: a two-parameter rigid face below its critical angle, the one-parameter board exactly at 0° | the transformer |
| `Material::closed_face` (470) | a face with a wave on one side | the pad |
| `interact_with` (577): `tir` branch (712, 727), open branch (717, 732) | every face event; the undeviated return (≈ 650) is a face met from inside a held body: no pad (N2) | pad / transformer |
| `ray_mesh_interact` main event (2467) | the traced solid's face from air | via `interact_with` |
| `SHEET_RESOLVE` redo (4434) | a glued sheet event | via `interact_with` |
| `ray_state_update`, `REPLACE_BY_VBS` (3983) | the redo with state | via `interact_with` |
| solver `BND` (1111): 1130 far face of a coherent sheet on a closed far side; 1146 open outer face of a plate; 1158 the "no `tf`" test; 1162 an open face; 1173 a closed layer seen from a wave | sheets and stacks | pad / transformer |
| leak chain (1073, 1074) | a mass-law sheet that is a whole plate under a closed outer pair | via `interact_with` |
| `qd_slab_resolve`, `fresnel_r` (1419 closed, 1421 open) | the mirror of a traced wave slab next to the peaked body (a lining in front) | pad / transformer |
| `PLATE_ABS` (4180), `PLATE_IN` (4186) | the plate rule's `a` | via `interact_with`, type 4 (scalar) or 1 (EM) |

Which branch each event takes (marker build `mk6.py`, every call tagged with its line, counted where either side has
`tfP ≠ 0`, `tfF > 0`):

- **Whole suite** (1138 cases, 1311472 assertions, all passed on the marker copy): only *Absorption peak - tfP, tfF and
  tfQ add a resonant peak to the room-side absorption of a mass-law face* and *ray_progress - the plate rule and the
  absorption peak at every frequency of a band* reach a peaked face, as expected. Sites: 2467 (closed and, at 0°, open),
  1173 (the sheet: pad and its pass-all twin), 1146 (the sheet at 0°), 1158, 4183 (`PLATE_ABS`, closed and open), and
  the undeviated return at 2467 and 3983 (the exit from inside; no pad).
- **A face met from air**: closed from 0.003° (one-parameter), from the critical angle (two-parameter), open below it.
- **From inside the held body** (N2): the undeviated return; no pad is formed, nothing changes under M6.
- **Behind a coherent layer** (a lining in front, probe geometries): as a glued sheet 1130, 1162, 2467, 4434, 4183; as a
  traced layer the slab mirror 1419 / 1421 and 2467, 3983, 4183.
- **A peaked sheet glued on a wall** (the leak chain): 1073, 1146, 1173, 4183.
- **The plate rule**: 4183 on every exit and entry of a peaked plate.

### 0.3 The design choices (item 3)

The switch copy `wi.cpp` (`mkwi.py`, `QD_WI` bitmask read once) carries the choices as real code: 1 = M6 (D1 b, D2 a,
D3 a, D4 i without the `m` gate, D5); 2 = D1 a; 4 = D2 b; 8 = D3 c; 16 = the `m > 0` gate; 32 = D4 ii; 64 = D4 iii.
How M6 sits in the code (common to every variant): `closed_face` gets the peaked far side's `ζ` as the load in place of
the bare face's susceptance `j·Im(e·u)`, behind the existing pad formed on the power-law part only (`apply_tf_pair`
with `tf(f, peak = false)` on the peaked side). The pad formula `w = (ν + b)/(1 + νb)` holds for any complex load `b`, so
D3 a is the pad as built in front of `ζ`: zero new lines. Where the face is open, `tf_face` gets `y` and returns the
scale `√((1 − |r_ζ|²)/(1 − |r_raw|²))` with `r = r_ζ`: the face reflects `ζ` on both sides of the critical angle. In the
solver an open outer face of a plate is formed from the side that is not the plate (`r_B = −r`, `y_B`).

**D1 — what `ζ` is met against.** (a) `ζ·c_air`, the cosine in air (the decided text); (b) `ζ` against the incidence
side's immittance `y` that `closed_face` already receives. Equal in air. Measured behind a 10 mm lining (dense
`eps 2.50 − 0.80j`, porous `0.31 − 0.20j`, lossless `2.50`) on a peaked wall (two-parameter concrete 100 mm or the
one-parameter board, `tf` 0, the peak 0.25 at 150 Hz, `tfQ` 0.75), types 4 and 5, 100 / 150 / 250 Hz, 0° to 75°:

- (b) is the transfer matrix of a locally reacting face behind the lining (`r_f = (y_L − 1/ζ)/(y_L + 1/ζ)` in the
  lining, then the layer): the glued sheet within 6.2e-13 (phase plane at the wall face, `e^{+2jk₀hc}`), the traced
  layer within 3.8e-5, both walls, all three linings.
- (a) is on its own formula (`r_f = (ζc − 1)/(ζc + 1)` inside the lining) within 1.8e-5 as a traced layer, but that
  formula is not the physics: (a) and (b) differ by up to 0.68 in `|r|` (dense, 100 Hz, 75°), 0.065 already at 0°; behind
  the porous lining (b) gives `|r|` 0.819 / 0.691 at 250 Hz, 60° / 75°, while (a)'s own formula gives 1.109 / 1.523
  (energy created), and the what-if returns NaN there (the air cosine reconstructed inside the face fails where the
  layer is evanescent). (a) also needs the cosine in air as a new argument to be well defined behind a layer.
- So (b) is physical (local reaction: the face's normal velocity is `p/Z_s` whatever lies in front), needs no argument,
  and is (a) in air. Lines: (b) 1 line in `closed_face`; (a) 3 to 4 lines plus a new argument through the callers.
- **A20.** With (b) a peaked face has a fixed surface impedance, met against whatever lies in front: that is what A20
  calls physical. For the peak A20 is closed by M6 (b); the power-law part stays a pad on the modulus of `y` (D3 a), so
  A20 remains for `tf` without a peak and for the power-law part of a peaked face. Recorded for A20, not widened.

**D2 — the bare face.** (a) `r` from `ζ` alone (the face's reactance replaced); (b) `ζ` with the bare face's own
susceptance (in parallel in the TE / scalar admittance frame, in series in the TM impedance frame, so that `ζ → ∞`
gives back the bare face; the literal series of impedances goes the wrong way: a rigid face's impedance is
unbounded and would swamp `ζ`). Measured, type 4, sheet and traced solid identical:

| body | (b) − (a), max \|Δr\| (0°–85°, 63 / 150 / 250 Hz) | phase difference |
| --- | --- | --- |
| concrete 152, two-parameter | 7.0e-3 (63 Hz, 85°) | up to 0.62° (85°); 2.1° at 150 Hz, 85° where \|r\| is 0.094 |
| glass 4, two-parameter | 6.4e-3 | up to 0.57°; 1.95° at 150 Hz, 85° |
| the fixture board, one-parameter | 1.55 | (b) brings back the bare `2θ`: 87.3° at 45°, 169.7° at 85° (63 Hz); \|r\| 0.881 / 0.963 at 45° / 85°, 150 Hz, against 0.795 / 0.094 |
| non-rigid body (`eps 1.50`, `m` 20, open) | 0 (open: the bare susceptance is zero) | — |

(a) is on the reference to 5e-13 in every row. (b) holds in the two-parameter mapping only (the expected small
difference) and fails in the one-parameter mapping, where the bare face's susceptance is of the order of `tan θ`.
The non-rigid body: M5 today reflects its Fresnel face with the pad (`|r|` 0.085 at 150 Hz, 0°; 0.654 at 85°); under M6
(a and b) it reflects `ζ` (0.844 / 0.094): the columns then say that the face is `ζ`, whatever the body. Recorded; a
peak on a non-rigid body is no use case.

**The open region and the critical angle** (D2 a, `b4c.py`): two-parameter concrete (critical angle 4.9072°) and glass
(3.4779°), sheet and traced solid, 150 and 250 Hz, a sweep of ±0.05° around the critical angle in 0.001° steps plus
0° to 10° in 0.25°: on the reference within 5.0e-13 on both sides; the largest step between neighbors 1.9e-7 in `R`
(M5 today 1.6e-6) and 1.6e-8 in `|T|²`. `tf_face` carries `ζ` below the critical angle, and there is no step.

**D3 — the power-law part.** At 0° and in angle, the board (sheet and traced solid identical; library on the formula to the five printed digits):

| | 0° absorption at 63 / 150 / 250 / 500 Hz | − M5 at 0° | angle shape | lines |
| --- | --- | --- | --- | --- |
| M5 (`tf` + peak, flat) | 0.1290 / 0.3000 / 0.2024 / 0.0905 | — | flat | — |
| (a) `(1 − t)\|r_ζ\|²` | 0.1251 / 0.2875 / 0.1948 / 0.0885 | −0.0040 / −0.0125 / −0.0076 / −0.0020 (= `−t·α_peak`) | 0.319 / 0.467 / 0.991 at 30° / 60° / 85°, 150 Hz | 0 (the pad as built) |
| (b) `t` folded into `R′` | not reachable | — | — | — |
| (c) additive, phase of `r_ζ` | 0.1290 / 0.3000 / 0.2024 / 0.0905 | 0 | 0.333 / 0.489 / 1.000 | about 3 |

(b) cannot be built with `ζ`'s reactance kept: the largest absorption any resistance gives at reactance `X` is
`2/(1 + √(1 + X²))`, and for the over-damped board the sum `t + α_peak` exceeds it below 111.5 Hz and above 201.6 Hz (at
1 kHz the bound is 0.027 against 0.060 wanted). (c) is not a physical element (an attenuation that depends on the
load, `s² = 1 − t/|Γ_ζ|²`), clamps at one per angle (1.000 at 85°, 150 Hz) and moves the phase nowhere differently from
(a). (a) is a matched resistive pad in front of `ζ`: energy-consistent by construction; its 0° value differs from M5's
sum by `t·α_peak` (at most 0.0125 for the board), which the refit absorbs (0.6).

**D4 — the gate and EM.** `tfP > 0 && tfF > 0` (a negative peak stays M5's forced-reflection dip), with or without
`m > 0`:

- Without the `m` gate a peaked wave medium (lossless `eps 2.50` traced solid 100 mm, `tf` 0.05 and the peak) creates
  energy: R + T 1.120 / 1.159 / 1.361 at 0° / 30° / 60°, 150 Hz, tracer 5 (the entry face is `ζ`, the exit from inside
  keeps the transformer). With the gate (QD_WI 17) the wave medium keeps M5 (R + T 1.00000, the same `|R|`, phase and `|T|²` as M5 to the printed digits). The
  gate is needed; `[wave]` carries no `tfP` and is identical either way.
- EM types (`b4e.py`; board and two-parameter concrete, sheet and traced solid, types 1 and 2 with their reflect
  pass 0, 63 / 150 / 250 Hz, 0° to 85°; channel VV = TE, HH = TM):
  - (i) TE `ζ·c` (the scalar form), TM `ζ/c` (an impedance boundary, Leontovich): VV and HH on these references within
    5e-13. TM absorbs less toward grazing for an over-damped face (`|HH|` 0.880 / 0.963 at 45° / 85°, 150 Hz, against
    VV 0.795 / 0.094). Needs a `tm` flag in `closed_face` and `tf_face`: the closed face's TM frame is an impedance
    ratio, and `ζ` written as for TE flips the TM sign (a rigid face would reflect +1 instead of −1).
  - (ii) TM the scalar form: `|HH| = |VV|` (sign of the TM frame kept); also needs the flag.
  - (iii) EM keeps the M5 pad: VV and HH as today (one-parameter VV phase `2θ`); needs the interaction type in
    `Material`; leaves A12 open for EM types.
  - Today (M5) EM: VV `|r| = √(1 − tf)` with phase `2θ` (one-parameter) or 0.01° to 0.69° (two-parameter), HH
    `−√(1 − tf)`. At exactly 0° the library's VV basis is negated (VV 180°, HH 0° under M5; the same under M6): a basis
    convention at normal incidence, not M6.
  - R + T ≤ 0.9996 for every EM row.

**D5 — the branch.** `tfQ < 0` selects `1/R′`; `tf()` squares `tfQ·(…)`, so M5's law is unchanged. Measured through the
library (sheet, type 4, 100 / 150 / 225 Hz, 0° / 45° / 85°, both branches): every row on the reference to the five printed digits for
`tfP` 0.25 / 0.9 / 0.999 / 1.0 / 1.2; at `tfP` 0.999 `R′` 1.065 / 0.939, at 1.0 both branches are `R′ = 1` and identical
(`|r|` 0 at the peak, 0°); 1.2 is taken as 1 (the clamp of `tf()` reads `tf_bg + 1 → 1`, and `R′` from `min(tfP, 1)`),
identical to 1.0; no NaN; R + T ≤ 0.9989. Today's library with `tfQ < 0` throws "Material property 'tfQ' = -0.750000 at
material 1 must be non-negative" (the `require`); the relaxed form removes that line.

### 0.4 The variants through the library (item 4)

Probe `probe.cpp` (a scratch probe on `test_acoustic_state_update.cpp`, `md/wave.md` 11): one ray through
`ray_triangle_intersect`, `ray_mesh_interact` and `ray_state_update` with `bf::walk`, `ub::OneThread`, `R` summed at its
polyline's point and referenced to the face; tracer 3 runs its reflect pass twice in `bf::walk` and is halved. All on
QD_WI 1 unless named.

- **The board, one-parameter** (`b4a.py`), sheet and traced solid, tracers 4 / 5 / 3, both branches, 31.5 Hz to 1 kHz,
  0° / 10° / 30° / 45° / 60° / 75° / 85°: on the series impedance within 6.2e-13 (tracer 3 2.5e-12), phase within
  1.4e-10°; 1/96-octave sweep 75–300 Hz at 0° and 45° within 1.0e-9 (over) / 1.1e-8 (under); measured extremes as in
  0.1 (−4.117° at 83.58 Hz, +4.117° at 269.21 Hz at 0°; −5.828° / +5.828° at 45°). Sheet and traced solid agree to the
  same digits. R + T ≤ 0.99986.
- **Two-parameter concrete and glass** with the board's peak (`b4b.py`): on the reference within 5e-13 at 0° to 85°.
- **Behind a lining** (D1, 0.3): (b) the transfer matrix within 6.2e-13 (sheet) and 3.8e-5 (traced).
- **A peaked board glued on two-parameter concrete** (the leak chain): `R` on `r_ζ` within 3.5e-13; R + T ≤ 0.875.
- **EM** (0.3, D4).
- **Energy** (`b4g.py`): R + T from air over the board, concrete and glass (sheet, solid, both branches) and the
  board behind the three linings (glued, traced), tracers 4 / 5 / 1 / 2, both channels, 31.5 Hz to 1 kHz, 0° to 85°:
  summed (eps 0) at most 0.99992; followed (eps 1.5) the same except the traced porous lining at 85°, 31.5 and 63 Hz:
  the followed series diverges under M6 (R + T 10.8 / 151, drop 0.12 / 6.6) where M5 converges (0.9992 / 0.9978).
  In the porous layer past its critical angle the immittance is mostly reactive (`y` at −67° to −82°), and the face's
  reflection in that frame is 1.04 to 1.09 below the peak (1.0580 at 31.5 Hz, 45°; 1.0859 at 63 Hz, 45°): a reflection
  against an inhomogeneous wave in a lossy medium (the lossy-medium convention, `md/acoustic.md` 7), which the
  transfer matrix sums correctly (summed rows ≤ 0.9987 and on the reference). This is the trap "a followed series of a
  lossy body between strong reflectors diverges: only summed rows count", met here by M6 where M5's pad, real by
  construction, kept `|r| ≤ 1`. Recorded for section 7 of `md/acoustic.md`.
- **Whole suite on the switch copy, QD_WI 1** (load 1.8 before, 34.0 after: other sessions; the counts are what
  matters): `[wave]` identical to `test_wave.txt` (seed line only), 165110 / 179 passed; `[acoustic]` and the whole
  suite: only the two peak cases move, 40 table rows each, 92 failing assertions (*Absorption peak*: 40 on `1 − R`,
  20 on `T`; the band case: 20 on `1 − R`, 12 on `T`); the band-equals-alone check (1e-4 dB) still passes; counts
  unchanged (1311472 / 1138).
- **The candidate as it would be built** (`prod.cpp`, `mkprod.py`: D1 b, D2 a, D3 a, the `m > 0` gate, EM i, D5):
  5124 lines (+56 / −21 against 5089), bitwise equal to the switch copy at QD_WI 17 over 1728 walks (every geometry
  above, types 0 to 5). Its three suites (load 4.05 before, 15.53 after: other sessions) equal the switch copy's
  QD_WI 1 runs line for line (only the binary's name differs): `[wave]` identical to the baseline, the two peak cases
  the only moved rows, 92 failing assertions, counts unchanged.

### 0.5 The plate rule (item 5)

`PLATE_ABS` reads the face through `interact_with`, so `a` becomes `1 − |r(θ)|²` of the peaked face. Measured
(`b5.py`, a probe with every column, `probe2.cpp`), sheet, type 4: the library equals round 1's formula copy
(`p3/tl/libac.py`, R1hX) with `a` = the face's absorption to 0.0000 dB under M5 (`θ > 0`) and under M6, so the plate
rule takes M6's `a(θ)` and nothing else changes. Panels: glass 4 mm and 4-16-4 with round 1's transmission columns
(`p3/tl/conf/cols.tsv`, the R1hF fits) and the M5 knobs of window glass / double glazing as their faces; the fixture
board.

| panel | T change per angle (M6 − M5) | octave TL change, field 0–78° (31.5 Hz … 8 kHz) |
| --- | --- | --- |
| glass 4 mm, window-glass peak | within 0.02 dB at 0° and 30°, 0.09 dB at 60°, up to +2.05 dB at 85° (63 Hz) | −0.06 / −0.08 / −0.04 / −0.10 / −0.21 / −0.01 / −0.02 / −0.08 / 0.00 dB |
| 4-16-4, double-glazing peak | within 0.006 dB at 0° / 30°, 0.09 dB at 60°, up to +3.42 dB at 85° (125 Hz) | 0.00 / −0.05 / −0.17 / −0.11 / −0.01 / 0 … dB |
| the fixture board | 0.06 / 0.09 / 1.07 dB at 0° / 30° / 60°, up to +6.35 dB at 85° (63 Hz) | −0.64 / −0.96 / −0.54 / −0.24 / −0.13 / −0.04 / 0 … dB |

The over-damped face absorbs more toward grazing, so the plate passes more there (`T → a` where `a` grows); TL per
octave falls by at most 0.21 dB on the panes and 0.96 dB on the board. **None of the 13 transmission fits of report 3.6
carries a peaked face**: the formula copy reads `tf` and `tfB` only (`tf_of`), and `cols.tsv` has no peak columns, so
M6 moves none of them.

### 0.6 The six rows refitted on M6 (item 6)

`refit6.py`: the rows, targets, octave means and Nelder–Mead of `scratch/m5/a10refit.py` (minimax in α on the octave
means, 31.5 Hz to 8 kHz, 150 starts); target the Paris integral 0°–90° of the face, from the closed form of a locally
reacting face (checked against the numerical integral within 4.9e-10, and per fit within 0.0000; D3 c clamps per angle,
the fit uses `min(1, P + t)`, off the numerical value by up to 0.0034 on cork). Knobs `t1k`, `b` (the power law at
1 kHz), `tfP`, `F`, `Q` (= |`tfQ`|). `m″` is what the mass relation `1 + R′ = ωm″/(Q_α·Z₀)` gives for the fitted knobs
(a check of the branch, not code).

| row | M5 (flat) | D3 a over | D3 a under | D3 c over | D3 c under | field 0–78°, a over / a under |
| --- | --- | --- | --- | --- | --- | --- |
| window glass | 0.0132 | 0.0132 (`R′` 39.2, 8.23 kg/m²) | 0.0121 (0.068, 0.29 kg/m²) | 0.0132 (52.7, 10.4) | 0.0123 (0.060, 0.26) | 0.0268 / 0.0154 |
| plasterboard on frame | 0.0170 | 0.0171 (4.23, 22.1) | 0.0172 (0.319, 4.78) | 0.0170 (3.93, 26.0) | 0.0170 (0.238, 4.54) | 0.0236 / 0.0186 |
| double glazing | 0.0075 | 0.0075 (21.9, 182) | 0.0075 (0.163, 16.1) | 0.0075 (26.8, 179) | 0.0075 (0.105, 15.7) | 0.0140 / 0.0075 |
| plasterboard on 25 mm battens | 0.0056 | 0.0056 (11.9, 6.12) | 0.0057 (0.241, 0.79) | 0.0057 (13.2, 6.75) | 0.0057 (0.217, 0.78) | 0.0219 / 0.0126 |
| wooden lining | 0.0101 | 0.0101 (45.9, 2.69) | 0.0102 (0.065, 0.079) | 0.0101 (48.5, 2.80) | 0.0103 (0.060, 0.077) | 0.0241 / 0.0150 |
| cork board | 0.0470 | 0.0476 (8.08, 1.20) | 0.0471 (0.326, 0.21) | 0.0468 (11.5, 1.65) | 0.0460 (0.238, 0.20) | 0.0619 / 0.0498 |

(M5's column reproduces `md/m5_a21_report.md` 1.2: 0.0133 / 0.0170 / 0.0075 / 0.0056 / 0.0100 / 0.047, within 0.0001.)

Knobs of the D3 a over-damped fits (`t1k`, `b`, `tfP`, `F` Hz, `Q`): window glass 0.0894, −0.394, 0.0970, 123.9, 0.387;
plasterboard on frame 0.0732, −0.385, 0.619, 88.6, 5.73; double glazing 0.0284, −0.395, 0.167, 101.0, 12.3 (under-damped
0.0284, −0.395, 0.483, 122.9, 25.9); battens 0.0933, +0.142, 0.287, 181.6, 1.32; wooden lining 0.0785, −0.246, 0.0835,
145.7, 0.127; cork 0.2308, +0.194, 0.392, 501.0, 1.01.

- Every row reaches its Paris target on M6 as closely as on M5, for both D3 forms and both branches (differences
  ≤ 0.0011). The over-damped face needs a smaller `tfP` for the same random-incidence value (its Paris value exceeds
  `α₀`: window glass 0.097 against M5's 0.133, battens 0.287 against 0.402); the under-damped one a larger one.
- The field value 0°–78° of a Paris fit misses the targets by 0.014 to 0.062 over-damped and 0.0075 to 0.050
  under-damped: the two averages differ for a locally reacting face (they were equal for the flat pad), so the type of
  the reference matters now (A13 already makes `tf` a random-incidence coefficient).
- The branch by the mass relation: window glass over-damped (8.2 kg/m², a 4 mm pane is 10 kg/m²), battens over-damped
  (6.1 kg/m², a board 8.6 to 10 kg/m²), double glazing under-damped (16.1 kg/m² against 182 over-damped; two 4 mm panes
  are 20 kg/m²), plasterboard on frame either (22.1 over-damped, 4.8 under-damped, a board 8.6 to 10 kg/m²), wooden lining
  and cork over-damped but light (2.7 and 1.2 kg/m² against about 6 and 3 to 6 kg/m²; under-damped 0.08 and 0.21 are not
  physical). The mass relation is the parameter session's rule (decided); this says the default (over-damped) fits four
  rows physically and double glazing needs `tfQ < 0`.
- Jeong's 0.951 bounds no target: the largest is 0.65 (cork, 500 Hz), the largest fitted Paris octave 0.603.
- **Through `ray_progress`** (`bp6.py`, the band probe linked to the switch copy, two-parameter concrete, sheet `S` and
  traced solid `X`, tracer 4, 48-point Gauss–Legendre in angle, 27 thirds per run): D3 a over and under on the
  closed-form Paris values within 4.7e-6 (float), sheet and traced solid identical, R + T ≤ 0.99933; D3 c within 3.4e-3
  (its fit's clamp approximation, above). The 36 runs took 2.3 s together (load 16.8).

### 0.7 Restatement and fixtures, proposed (item 7)

References written in the test from `ζ` and the column definition (no library code): `R′` from `tfP` and the sign of
`tfQ`, `ζ`, `r = (ζc − 1)/(ζc + 1)` (TM `ζ/c`), `α(θ) = 1 − (1 − t)|r|²`, the plate rule with `a(θ)`; the Paris value
from the closed form (`paris_cf.py`, three lines).

1. ***Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption of a mass-law face***,
   restated: `ar = 1 − (1 − t)|r_ζ(θ)|²` in place of the flat law; the special line for exactly 0° (the open face's
   Fresnel `R0`) goes, because under M6 the face is `ζ` at 0° too (measured: 0.28750 at 150 Hz, 0°, on `ζ`); the plate
   reference takes `a(θ)`. Under M6 the unchanged case fails 40 checks on `1 − R` (up to 0.068) and 20 on `T` (up to
   0.36 dB), the band case 20 on `1 − R` (up to 0.093) and 12 on `T` (up to 0.53 dB) (0.4). Mapping:
   loop the one-parameter board and a two-parameter one (gypsum 12.5 mm, `a` 4.6908e-5, `e` 573.33, as in *Two-parameter
   mapping - air | water, a hard wall's phase, a porous layer on it and every tracer below its critical angle*), which asserts that the peaked
   face does not depend on the mapping (D2 a); round 1 decided the two-parameter mapping for materials (N7), M8 would
   retire the other. Names unchanged; bounds 1e-3 on `1 − R` and 0.01 dB on `T` kept (as at M5), the measured maxima
   reported in phase 1.
2. ***ray_progress - the plate rule and the absorption peak at every frequency of a band***, restated the same way at
   its 20° and 50°: `ar(θ)` and the plate rule with it; the band-equals-alone check unchanged (it passes under M6).
3. **New, `test_acoustic_state_update.cpp`:** *Absorption peak - a peaked face is its series impedance in magnitude and
   phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches*: the board in both mappings, tracers 4,
   5 and 3, and EM type 1 (VV `ζ·c`, HH `ζ/c`) if D4 i is decided; 63 / 150 / 250 / 1000 Hz at 0° / 10° / 30° / 45° /
   60° / 75° / 85°; the phase extremes on a 1/96-octave sweep at 0° (−4.117° at 83.58 Hz, +4.117° at 269.21 Hz,
   computed in the test from `ζ`); `tfP` 1.0 (the branches equal) and 1.2 (as 1.0); R + T ≤ 1 summed; two lines behind
   the dense lining (glued and traced) against the transfer matrix of a locally reacting face (D1 b); a ±0.05° sweep at
   the two-parameter critical angle (no step). Today's library fails every phase row (it reflects the bare face's
   phase: `2θ` one-parameter). Bounds 2 to 3 times the measured maxima (6.2e-13 on the grid, 1.1e-8 on the sweep, 3.8e-5
   traced behind the lining). Expected run time about 0.1 s (a one-face walk costs 0.03 ms; 336 walks in 0.02 s).
4. **New, `test_acoustic_progress.cpp`:** *ray_progress - the random-incidence absorption of a peaked lining is the
   Paris integral of its series impedance*: plasterboard on 25 mm battens refitted on M6 (D3 a over-damped: `t1k`
   0.0933, `b` 0.1423, `tfP` 0.287, 181.6 Hz, `Q` 1.319) on two-parameter concrete, sheet and traced solid, tracer 4,
   48-point Gauss–Legendre, 27 thirds in one band; octave Paris against the closed form (measured 4.3e-6, bound 1e-5)
   and against the row's targets (0.0057). Today's library with the same knobs reads the flat law: its octave Paris
   values are off by up to 0.084 (0.225 / 0.251 against 0.304 / 0.335 at 125 / 250 Hz; residual 0.085 against the
   targets). Expected run time about 0.15 s (two runs of 96 angles with 27 thirds took about 0.13 s).

Both restated cases stay at a few milliseconds (1 ms and 2 ms today, load 10.8).

### 0.8 Recommendation (item 8)

| decision | recommended | reason |
| --- | --- | --- |
| D1 | (b) `ζ` against the incidence side's `y` | the transfer matrix of a locally reacting face behind any lining (6.2e-13 glued, 3.8e-5 traced); (a) is not the physics behind a layer (up to 0.68 in `\|r\|`, \|r\| 1.52 behind the porous lining, NaN in the what-if) and needs a new argument; (b) is one line and equal to (a) in air. For A20: closes it for the peak; the power-law pad keeps A20's behavior |
| D2 | (a) `ζ` replaces the bare face | mapping-independent (5e-13 in both mappings), no step at the critical angle (1.9e-7 between 0.001° neighbors); (b) brings back the one-parameter `2θ` and almost no absorption (\|Δr\| up to 1.55) |
| D3 | (a) the power-law pad in front of `ζ` (the decided text) | a physical element, energy-consistent, zero extra lines (the pad as built); 0° differs from M5's sum by `t·α_peak` (≤ 0.0125 on the board), which the refit absorbs; (b) cannot be built (the reactance bounds the absorption: infeasible outside 111.5–201.6 Hz on the board); (c) equals M5 at 0° but is a load-dependent attenuation, clamps at one, and costs lines |
| D4 gate | `m > 0 && tfP > 0 && tfF > 0` | without `m` a peaked wave medium creates energy (R + T 1.36); with it the wave medium is M5 exactly; a negative peak stays M5's dip |
| D4 EM | (i) TE `ζ·c`, TM `ζ/c` | the impedance boundary of EM (Leontovich) and the scalar form in TE; closes A12 for every type; the same `tm` flag as (ii); (iii) needs the type in `Material` and leaves EM types on the old pad |
| D5 | `tfQ < 0` under-damped, the `require` line removed, `R′` from `min(tfP, 1)` | both branches on the reference, the roots merge at `tfP` 1 and above, no NaN; M5's law unchanged (it squares `tfQ`) |
| fixtures | 0.7, items 1 to 4 | the restated pair plus the phase case and one random-incidence row through `ray_progress` |

What it costs: `src/ray_mesh_interact.cpp` 5124 lines against 5089 (+56 / −21, `prod.cpp`): `Material::tf` (a `peak`
argument), `apply_tf_pair` (the power-law part), `peaked()` and `zeta()` (7 lines), `closed_face` (a `tm` argument and the
load), `tf_face` (`y` and `tm`, 6 lines), three calls in `interact_with`, the solver's `TF_FACE` lambda (6 lines) and its
two open faces and two closed faces, the slab mirror (4 lines), the `require` line removed, comments. No column, no
state, no new argument outside `Material` and its callers. `[wave]` and the twins untouched (`[wave]` identical; no wave
case carries `tfP`, and the gate excludes wave media).

What each candidate's code replaces: for a peaked mass-law face the pad of `closed_face` (807a1d6, "tf at a closed face
is a pad and a scalar closed face reflects at the Fresnel magnitude") and the transformer of `tf_face` (70dd1b4, "tf at
a bare boundary is a transformer where the face is open and a flat leak where it is closed"); both stay for every face
without a peak, so no session is undone. The `require` on `tfQ` came with A10 (4720a92). M5's law in `Material::tf`
(238444d) stays: it is `ζ`'s normal-incidence absorption.

Side findings for the documents (phase 3): the followed series behind a traced lossy lining past its critical angle can
diverge on a peaked face (0.4; only summed rows count there, `md/acoustic.md` 7); a peak on a non-rigid mass-law body
makes the face `ζ` (no use case); the field and Paris values of a peaked face now differ (0.6), so the coefficient type
of a peaked row's reference matters (A13); double glazing fits physically only under-damped (`tfQ < 0`).

A reusable instrument for `tools/acoustic`, if Stephan agrees: the closed-form Paris value of a locally reacting face
(`paris_cf.py`, checked against the numerical integral within 4.9e-10 and against Jeong's 0.951 at 1.567), which the
parameter session needs to convert a peaked row's random-incidence target into `tfP`, `tfF`, `tfQ`.

Stop for Stephan's decisions: D1 to D5, the mapping of the restated cases, the fixtures of 0.7.

### 0.9 Rig (`build_linux/scratch/m6`)

`README` of the folder in brief (scratch is not kept; the numbers are above):

- `build.sh <lib.cpp|-> <abs out> [replace=<test object>:<probe.cpp>] [extra .cpp]`, `build_bp.sh` (copied from
  `scratch/m5`, paths changed); `plain.cpp` / `tb_plain` (the plain copy); `runs/` (suite outputs).
- `mk6.py` → `mk.cpp`, `tb_mk`, `pr_mk` (marker copy: every pad call tagged with its line, `QD_MK_OUT`, `listener.cpp`);
  `mk_all.tsv` (whole suite).
- `mkwi.py` → `wi.cpp`, `tb_wi`, `pr_wi`, `pr2_wi`, `bp6_wi` (switch copy, `QD_WI` bits 1 / 2 / 4 / 8 / 16 / 32 / 64).
- `mkprod.py` → `prod.cpp`, `tb_prod`, `pr_prod` (the recommended candidate, no switches); `cmp_prod.py`.
- `probe.cpp` (walls from `$M6_IN`, `bf::walk`), `probe2.cpp` (every column, `$M6_IN2`), `m6run.py` (driver).
- `m6lib.py` (references), `paris_cf.py` (closed-form Paris), `b1.py` / `b1b.py` (item 1), `b4a.py` … `b4g.py` (item 4:
  board, D2, critical angle, D1, EM, D3 / D4 / D5, energy), `b4d2.py` (glued lining), `b5.py` (plate rule),
  `refit6.py` / `refit6_150.json` / `.out` (item 6), `bp6.py` / `bp6.out` (through `ray_progress`), `dbg*.py` (D1 a NaN,
  divergence, site map, leak chain), `tm1.py` (timing).

## 1. Phase 1 — fixtures (2026-10-16)

Decided (Stephan, 2026-10-16, as recommended in 0.8): D1 b, D2 a, D3 a; D4 the gate `m > 0 && tfP > 0 && tfF > 0` (a
wave medium and a negative peak stay on M5), EM (i) TE `ζ·c`, TM `ζ/c` with the `tm` flag in `closed_face` and `tf_face`;
D5 `tfQ < 0` the under-damped root, the `require` on `tfQ` removed, `R′` from `min(tfP, 1)`. The candidate is `prod.cpp`
(0.4, 0.8).

### 1.0 Check C1: the followed divergence behind a lining (before any fixture)

**Walker** (`c1.py`; probe with every column, `pr2_plain` = M5 against `pr2_prod` = the candidate): a 10 mm lining glued
(a sheet on the wall's face) and traced on the peaked board wall (one-parameter, 100 mm, `tf` 0, `tfP` 0.25 at 150 Hz,
`tfQ` 0.75), types 4, 5 and 1 (both channels), 31.5 / 63 / 125 / 250 / 500 / 1000 Hz, 0° / 10° / 30° / 45° / 60° / 75° /
85°, 126 runs per row; max R + T and the runs above one:

| lining | glued, summed / followed | traced, summed | traced, followed |
| --- | --- | --- | --- |
| porous one-parameter `0.31 − 0.20j` (rarer than air, lossy) | M5 0.99952 / 0.99952; candidate 0.99867 / 0.99867 | M5 0.99952; candidate 0.99867 | M5 0.99943; **candidate diverges**: 6 runs above one, 151.1 at 63 Hz, 85° (drop 6.6), all at 85°, 31.5 and 63 Hz |
| rock wool two-parameter (the fill of *Leaf - a porous fill at the back of the box is the transfer matrix of leaf, air, fill and wall*: Miki 32.8 kPa·s/m², its power-law columns) | M5 0.99993 / 0.99993; candidate 0.99890 / 0.99890 | M5 0.99993; candidate 0.99890 | **M5 diverges**: 3 runs above one, 4.0e14 at 125 Hz, 10° (drop 2.9e15); candidate 0.99890, none above one |
| lossless rarer than air, eps 0.5, μ 1 | M5 0.99999; candidate 0.99936; none above one | M5 1.00020, candidate 1.00050: 12 runs above one, all at exactly 45.000° | the same as summed |

The eps 0.5 rows are a single angle: 45° is that lining's critical angle (`sin²θ = 0.5`). Swept at 44° / 44.9° /
44.99° / 44.999° / 45.000° / 45.001° / 45.01° / 45.1° / 46°, type 4, traced: the candidate reads 0.97665 … 0.97631, then
1.00050 at 45.000°, then 1.000000 (the entry face totally reflects); M5 reads 0.961 … 0.441 (a dip toward the critical
angle), 1.00016 at 45.000°, then 1.000000. At 1 kHz the same (1.00050 / 1.00020). Both builds have the one-point value
at exactly the critical angle; it is not M6's. The glued sheet tunnels through the layer and stays below one on both
sides (candidate 0.9763 at 31.5 Hz, smooth through 45°).

So the followed series diverges under the candidate only behind the one-parameter porous lining, which is rarer than
air (its critical angle in air is 33.8°), and only at 85° and 31.5 / 63 Hz, where the face's reflection in the lining's
frame exceeds one (1.04 to 1.09, 0.4). Behind the two-parameter rock wool the candidate converges where M5 diverges (the
fill case's known round trip of 1.145 in a followed series). Summed rows stay below one everywhere.

**Through `ray_progress`** (`c1rp.py`). `tools/acoustic/band_probe.cpp` has no `tfP`, `tfF`, `tfQ` columns, so the run
uses its M5 copy `bp5.cpp` (the same program with the three columns appended to its column list, nothing else), linked
to `prod.cpp` (`bp5_prod`) and to the plain copy (`bp5_plain`). The one-parameter porous lining (10 mm, `ε″` 0.20 at every
frequency) traced and glued on the peaked wall, 85°, tracers 4 and 5, `thin_slab_threshold` 0.15 and 1.0, 255
interactions, gain floor −250 dB, bands {31.5}, {63}, {31.5, 63} Hz; `timeout` 120 s and a 4 GB address-space limit per run:

| | traced, tracer 4, thr 0.15: R + T at 31.5 / 63 Hz, rays | traced, tracer 4, thr 1.0 | traced, tracer 5 (both thr) | glued, both tracers |
| --- | --- | --- | --- | --- |
| M5 | 0.99195 / 0.98382, 2843 / 1085 rays | 0.99187 / 0.98354, 17746 / 7119 rays | 0.99916 / 0.99283, 54 rays | 0.99916 / 0.99283, 3 rays |
| candidate | 0.77213 / 0.39487, 120 / 491 rays | 0.77155 / 0.39543, 571 / 5624 rays | 0.83695 / 0.43843, 6 rays | 0.83697 / 0.43844, 3 rays |

`|R|²` of the candidate 0.7708 / 0.3908 (tracer 4) and 0.7596 / 0.3718 (tracer 5); `|T|²` at most 0.077. No run ran away:
at most 17746 rays (M5, threshold 1.0), every run under 0.07 s and 8.2 MB peak memory (`/usr/bin/time`), load 2.96
before and after. Nothing returned more than it received (largest R + T 0.99916, M5). A band of two frequencies equals
each frequency alone (to the printed digits, 0.39085 / 0.39082 at 63 Hz the only fifth-digit difference, tracer 4).

Recorded finding: behind the traced porous lining at 85° `ray_progress` with tracer 4 returns less than the summed
walker of the same row, which is 0.83697 / 0.43844 at 31.5 / 63 Hz for the candidate (traced and glued, types 4 and 5
alike) and 0.99916 / 0.99283 for M5. The shortfall is 0.065 / 0.044 under the candidate and 0.007 / 0.009 under M5:
tracer 4 in `ray_progress` follows the traced layer's bounces as rays (120 to 17746 rays against 6 for tracer 5), and
the followed sum ends low (the cause is not traced further here). A loss, not a gain; tracer 5 and the glued sheet equal
the walker.

Conclusion as the C1 rule asks: only a configuration rarer than air diverges, `ray_progress` neither runs away nor
creates energy; recorded for `md/acoustic.md` 7, and phase 1 goes on.

### 1.1 The fixtures

Two files, no other change (`git diff --stat`: `test_acoustic_state_update.cpp` +335 / −26, `test_acoustic_progress.cpp`
+149 / −9; `src/` untouched, 5089 lines). Every reference is written in the test from `ζ` and the column definitions: the
namespace `m6t` in `test_acoustic_state_update.cpp` (`zeta`, `r_face`, the board's palette, a `bf::walk` wrapper);
inline in the two `ray_progress` cases. No geometry leaves air inside a body, so `bf::walk` is used throughout; type 3
is one event through `ub::traced` (`bf::walk` runs its reflect pass twice when the type is 3).

1. ***Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption of a mass-law face***
   (restated): the board in the one-parameter mapping and as two-parameter gypsum (`a` 4.6908e-5, `e` 573.33), sheet
   and traced solid, types 4 and 5, 63 Hz to 1 kHz, 0° and 45°; `α = 1 − (1 − tf)|r_ζ|²`; the exactly-0° line is gone;
   the plate rule with `a(θ)`. 160 assertions (80 before: the second mapping).
2. ***ray_progress - the plate rule and the absorption peak at every frequency of a band*** (restated): `ar` from `ζ` at
   20° and 50°; the band-equals-alone check unchanged. 120 assertions.
3. ***Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as
   a traced solid, on both branches*** (new, after item 1). 2460 assertions in seven parts:
   - A, the board in both mappings, sheet and traced solid, types 4, 5, 3, `tfQ` ±0.75, 63 / 150 / 250 / 1000 Hz,
     0° to 85°: complex `R` against `√(1 − tf)·r_ζ`, and R + T ≤ 1;
   - B, a 1/96-octave sweep at 0°, 75–300 Hz: the phase extremes at the reference's frequencies;
   - C, `tfP` 1.0 and 1.2 on both branches: all four on `R′ = 1`;
   - D, EM type 1: VV against `(ζc − 1)/(ζc + 1)`, HH against `−(ζ/c − 1)/(ζ/c + 1)`, from 10°, R + T ≤ 1 per channel;
   - E, the gate: a peaked wave medium (`m` 0, eps 2.50, 100 mm traced, `tf` 0.05 and the board's peak) against the flat
     pad written in the test (each face the transformer on the pair rule `R′ = (1 − tf)·r²`, the layer's Airy sum):
     `R` complex, `|T|²`, R + T;
   - F, behind a 10 mm lining (eps 2.50 − 0.80j) glued and traced on the board wall (`tf` 0): the transfer matrix of a
     locally reacting face behind the layer (the glued sheet at the plane of the wall face);
   - G, ±0.05° in 0.001° at the two-parameter critical angle (9.4388°), sheet and traced solid: on `ζ`, and the largest
     step between neighbors.
   `tfQ < 0` goes through `CHECK_NOTHROW`: today's library refuses it as a failed check, the row reads NaN, and the run
   goes on.
4. ***ray_progress - the random-incidence absorption of a peaked lining is the Paris integral of its series impedance***
   (new, last in `test_acoustic_progress.cpp`): the battens row refitted on M6 (0.6) on two-parameter concrete, sheet and
   traced solid, tracer 4, 27 thirds in one band, 48 Gauss–Legendre angles (nodes computed in the test); octave Paris
   against the closed form `t + (1 − t)·P(ζ)`; the row's measured octaves (125 Hz–8 kHz) printed alongside. 18 assertions.

### 1.2 Measured maxima and bounds (on the candidate, `prod.cpp` relinked with these tests)

| case | quantity | measured max | bound |
| --- | --- | --- | --- |
| 1 | \|α − ref\| | 3.3e-16 | 1.0e-12 |
| 1 | \|T − rule\| | 6.6e-4 dB | 2.0e-3 dB |
| 2 | \|α − ref\| (float) | 4.9e-8 | 1.5e-7 |
| 2 | \|T − rule\| | 6.2e-4 dB | 1.5e-3 dB |
| 2 | band − alone | 0 dB | 1e-4 dB (unchanged) |
| 3 A | \|R − ref\|, types 4 / 5 / 3 | 7.4e-16 (R + T ≤ 0.99933) | 1.0e-12, R + T ≤ 1 |
| 3 B | \|R − ref\| on the sweep; phase at the extremes | 3.3e-16; 0° (−4.1172° at 83.58 Hz, +4.1172° at 269.21 Hz, same frequency index as the reference) | 1.0e-12; 1e-12° and the index equal |
| 3 C | \|R − ref\| | 3.0e-16 | 1.0e-12 |
| 3 D | \|VV − ref\|, \|HH − ref\| | 5.6e-16 (R + T ≤ 0.99959) | 1.0e-12, R + T ≤ 1 |
| 3 E | \|R − ref\|, \|\|T\|² − ref\| | 2.9e-6, 2.6e-6; R + T up to 1 + 1.15e-6 | 8.0e-6; R + T ≤ 1 + 3e-6 |
| 3 F | glued, traced | 1.8e-15, 3.4e-5 | 1.0e-12, 1.0e-4 |
| 3 G | \|R − ref\|, step | 2.2e-16, 3.6e-7 | 1.0e-12, 1.0e-6 |
| 4 | \|Paris − closed form\| | 4.3e-6 | 1.2e-5 |

E (the gate) is today's path: the same rows read the same values on today's library (R + T 1.00000007 at 63 Hz, 0°,
up to 1 + 1.15e-6 at 150 Hz, 60°). That a lossless traced dielectric with a `tf` pad sums to slightly more than one, and
2.9e-6 off its Airy sum, is the existing wave side's (the traced geometry), not M6's; the bound is stated in the case.
The phase-extreme bound cannot be a multiple of a measured zero; 1e-12° stands for "equal". Bounds whose 2 to 3 times the maximum fell below 1e-12 are 1e-12
(decided 2026-10-16, after review): rounding-level maxima move with the compiler, FMA contraction and libm, and 1e-12
is the stored twins' tolerance.

### 1.3 Against today's library

`[wave]`: 165110 / 179 passed, identical to `test_wave.txt` (seed line only). `[acoustic]`: 54 cases, 50 passed, 4
failed, 39371 assertions (37716 passed, 1655 failed). Whole suite: 1140 cases, 1136 passed, 4 failed, 1314030 assertions
(1312375 passed). The counts add up: 1311472 + 80 (case 1's second mapping) + 2460 + 18 = 1314030; 1138 + 2 = 1140.

Every other line of the three baselines is unchanged: diffed table by table, the moved lines are the four cases' tables
and failure messages, the summary line and the seed line (`[acoustic]` and whole suite alike).

Failing assertions per case (largest finite deviation):

| case | failing | what |
| --- | --- | --- |
| 1 | 140 of 160 | `α` 80 of 80 (0.068), `T` 60 of 80 (0.362 dB) |
| 2 | 64 of 120 | `α` 40 of 40 (0.093), `T` 24 of 40 (0.527 dB); band − alone passes |
| 3 | 1433 of 2460 | A: 672 `R` (1.84, the bare phase `2θ` and the flat pad), 224 `R + T` (the NaN rows of `tfQ < 0`), 242 + 112 `CHECK_NOTHROW` (`tfQ < 0` refused: "Material property 'tfQ' … must be non-negative"); B: both extremes at 75.00 Hz with phase 0° (against 83.58 / 269.21 Hz, ±4.1172°), sweep 0.065; C: 8 of 9 rows off (0.95; the under-damped palettes refused); D: 72 VV (1.75) and 72 HH (0.126); E: passes (the gate keeps today's pad); F: 24 of 24 (glued and traced, 1.56); G: 0.0058 off `ζ`, step 7.0e-6 |
| 4 | 18 of 18 | Paris octaves 0.084 off the closed form (the flat law) |

The run did not abort: the refused palettes are failed checks inside the case, and the case went on to its other parts.

### 1.4 Run times

`--durations yes` against today's library, 1-minute load 3.98 before and after (the 5-minute load 9.31: other sessions
were winding down; earlier runs at load 8.58 gave the same to 1 ms, and the candidate the same):

| case | time |
| --- | --- |
| *Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption of a mass-law face* | 0.003 s |
| *Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches* | 0.026 s |
| *ray_progress - the plate rule and the absorption peak at every frequency of a band* | 0.002 s |
| *ray_progress - the random-incidence absorption of a peaked lining is the Paris integral of its series impedance* | 0.035 s |

Each well under 0.5 s; together 0.066 s.

### 1.5 Notes for phase 2 and 3

- On the candidate (`prod.cpp`, 5124 lines) the four cases pass and `[wave]` is identical; phase 2 builds that code into
  `src/ray_mesh_interact.cpp` and reruns the whole suite.
- `python3 tools/acoustic/check_docs.py` now reports the two new case names missing from `md/acoustic.md` (its appendix
  lists every `[acoustic]` case); phase 3 adds them. Everything else passes.
- `tools/acoustic/band_probe.cpp` has no `tfP`, `tfF`, `tfQ` columns (1.0); adding them is three entries in its column
  list and its header comment, for Stephan to decide with `paris_cf.py` in phase 3.
- For `md/acoustic.md` 7 (decided placement): the followed divergence behind a lining rarer than air (1.0, 0.4) and the
  one-angle value at a lossless lining's critical angle (in M5 as well), `ray_progress` tracer 4's shortfall behind the
  traced porous lining at 85° (a loss), and a peak on a non-rigid body.

Stopped after phase 1.

## 2. Phase 2 — source (2026-10-16)

Decided after phase 1 (Stephan, 2026-10-16): C1 accepted as recorded in 1.0, its findings to `md/acoustic.md` 7 in
phase 3; every bound below 1e-12 raised to 1e-12 (cases 1, 3 A, 3 B, 3 C, 3 D, 3 F glued, 3 G; the table of 1.2 is
updated), the phase-extreme check the same frequency index plus 1e-12°.

### 2.1 The change

`src/ray_mesh_interact.cpp` is `prod.cpp` (0.4, 0.8) with the comments rewritten: 5130 lines against 5089 (+64 / −23;
`prod.cpp` 5124, the six more lines are comment). The code equals `prod.cpp` apart from comments (compared with every
comment line and trailing comment removed: no difference). What it does, as decided:

- `Material::peaked()` (`m > 0 && tfP > 0 && tfF > 0`) and `Material::zeta(f)` (`R′` from `min(tfP, 1)`, the
  under-damped root where `tfQ < 0`); `tf(f, peak)` and `apply_tf_pair(…, bg)` give the power-law part of a peaked
  side, the pad in front of `ζ` (D3 a).
- `closed_face(…, tm)`: for a peaked far side the load is `ζ` met against the incidence side's `y` (D1 b), in place of
  the bare face's reactance (D2 a): the admittance `1/ζ` for TE and the scalar type, the impedance `ζ` for TM (EM i).
- `tf_face(…, y, tm)`: where `y` is given and the far side is peaked, the face's reflection is `closed_face`'s and
  the scale is what the face absorbs over what the raw crossing passes (no step at a critical angle).
- Callers: `interact_with` (three calls), the solver (`TF_FACE`, which forms an open outer face of a plate from the side
  that is not the plate, and the TM flag on the two closed-face calls), the slab mirror; the `require` on `tfQ` is gone.
- Comments of `tf`, `apply_tf_pair`, `tf_face` (and `peaked` / `zeta` before it) and `closed_face` state the face as a
  series impedance, the branch, the gate, the pad and the TE / TM forms.

### 2.2 Suites

Built with `cmake --build build_linux --parallel`; load 5.63 before, 25.01 after (other sessions; counts only).

- `[wave]`: All tests passed (165110 assertions in 179 test cases), identical to `test_wave.txt` line for line (only
  the seed line differs). `tests/data/wave_twins.tsv` unchanged (`git status`: not modified) and asserted.
- `[acoustic]`: All tests passed (39371 assertions in 54 test cases).
- Whole suite: All tests passed (1314030 assertions in 1140 test cases).
- Moved rows, against phase 1's run (the same tests on today's library): only the four M6 cases (their tables and the
  failure messages that are gone), plus the seed and the summary line. Against the baselines `test*.txt`: only the two
  restated cases' tables and the two new cases (the gate table E reads as on today's library), plus the seed and the
  summary line. Every other row of the three baselines is unchanged.

### 2.3 The energy checks of 0.4 and C1 on the built library

Probes relinked on `libquadriga.a` (`pr_built`, `pr2_built`, `bp5_built`):

- 0.4 (`b4g_built.py`): identical to the switch copy's table, every row (board, concrete and glass, both branches,
  sheet and solid, the three linings glued and traced; tracers 4, 5, 1, 2; summed and followed): summed at most
  0.99992; followed the same except the traced one-parameter porous lining at 85°, 31.5 / 63 Hz (R + T 10.8 / 151), as
  recorded.
- C1 walker (`c1_built.py`): the 12 rows of 1.0 equal the candidate's, row for row (porous one-parameter, rock wool
  two-parameter, eps 0.5; glued and traced; summed and followed).
- C1 `ray_progress` (`c1rp_built.py`): the 32 runs equal the candidate's in `|R|²`, `|T|²`, R + T and the path counts;
  largest R + T 0.83697.

### 2.4 Side checks (scratch; nothing changed by them)

**S1: `ray_progress` tracer 4 behind the traced porous lining.** Instruments: a scratch copy of the built source that
prints the slab gate (`s1.cpp`, `QD_S1`), and the band probe printing every collected path (`bp5p.cpp`, `QD_PATHS`); the
row of 1.0 (the one-parameter porous lining `0.31 − 0.20j`, 10 mm, traced on the peaked board wall, 85°).

- **The gate.** `qd_slab_resolve` decides summed or followed: `ρ = |r_near·r_far|·√g2L` against `eps` (the
  `thin_slab_threshold` that `ray_progress` hands to `ray_state_update`), summed where `ρ ≥ eps`. For the lining's
  series `|r_near·r_far|` is 1.012 / 1.041 and `g2L` 0.988 / 0.976 at 31.5 / 63 Hz, so `ρ` = 1.006 / 1.029 (the exact
  round trip 1.0025 / 1.0212): summed at threshold 0.15, and also at 1.0, because `ρ` is above one. Tracer 5 makes two
  gate calls, both summed; tracer 4 makes 40 at 0.15, all summed.
- **The extra rays are not that series.** Listed by path: both tracers return the same two reflected paths (path
  powers 0.9435, the front face, and 3.383, the summed series; they interfere to `|R|²` 0.7596). Tracer 4's further
  paths (40 at 0.15, 120 rays) run through the held wall and back. The wall's face toward the lining is closed at 85° for
  the held wave, whose air invariant 0.992 lies above the lining's `εμ` 0.31. They then cross the lining at normal
  direction (`|ct|` 1.000, a second series with `ρ` 0.283 / 0.273: summed at 0.15, declined at 1.0, which gives the 571
  to 5624 rays of 1.0). Each carries at most 2.4e-4 in power.
- **The shortfall is the transmitted path.** Tracer 4 hands 0.00126 in power to the path through the wall, tracer 5
  0.0774. The walker gives `|T|²` 0.0774 for both tracers at eps 0, 0.15 and 1.0 (the same events through
  `ray_mesh_interact` and `ray_state_update`), so neither the threshold nor the gate decides it. It is `ray_progress`'s
  tracer-4 leg out of the summed lossy lining into the held wall. Under M5 the same leg is off by the same order (path
  power 2.6e-5 against 3.5e-3 for tracer 5, walker `|T|²` 0.0030), invisible in R + T because M5's face passes little at
  85°. Not traced to a line in this session.
- **`md/wave.md`'s sentence** (section 8, not 7: "Through `ray_progress` such a leg is always summed") holds for this
  lining at both thresholds, but only because its `ρ` exceeds the threshold, here even one. It is not structural:
  `ray_state_update`'s own description says `eps ≥ 1` always re-emits, which does not hold where `ρ > 1`. Proposed
  wording, not applied: "Through `ray_progress` such a leg is summed wherever its round trip `ρ = |r1 r2|·|φ|²`
  reaches `thin_slab_threshold`, and a round trip above one always does (the gate in `qd_slab_resolve`, at a threshold
  of 1 as well); …". The same reading applies to `ray_state_update`'s line "`eps ≥ 1` always re-emits": it re-emits
  unless `ρ ≥ 1`.
- So the finding for `md/acoustic.md` 7 changes: the tracer-4 shortfall behind a lining rarer than air is a
  `ray_progress` transmit-leg discrepancy that predates M6 and that M6's absorbing face makes visible (0.065 / 0.044 in
  R + T). It is not a followed series.

**S2: a traced two-parameter porous layer on concrete** (`s2.py`): the rock wool of 1.0, 10 mm and 50 mm, traced on
two-parameter concrete (200 mm traced, `m` 20, flat `tf` 0.02, no peak). Tracers 4 and 5, 0° / 10° / 45° / 85°, 63 /
125 / 500 Hz, `thin_slab_threshold` 0.15 and 1.0, through `ray_progress` on the plain copy and on the build (`timeout`
120 s, 4 GB address-space limit), against the summed walker (`pr2_plain`, `pr2_built`) and the fluid-layer transfer
matrix (`lw::stack` ported to Python; backing the concrete half-space with `tf` 0, and behind the pad of `tf` 0.02
referred to air). 48 rows × 4 runs. Load 2.46 before and after.

- No run got out of control: at most 358 rays (10 mm, 0°, 63 Hz, threshold 1.0), 0.01 s and 8 MB peak each.
- Nothing returned more than one: largest R + T 0.99645.
- `ray_progress` equals the summed walker within 1.3e-4 in R + T, every row, both thresholds, both tracers; today's
  library and the build are identical (no peaked face), walker and `ray_progress` alike.
- Against the transfer matrix the walker's `|R|²` lies between the two backings: up to 0.036 below the half-space with
  `tf` 0 and 0.020 from the air-referred pad (10 mm, 85°, 125 Hz). That is A20 (the pad formed on the fill's
  immittance, not referred to air), not a change here.
- The divergence that M5's followed walker showed on the peaked wall behind this rock wool (4.0e14, 1.0) does not
  appear through `ray_progress`; it belongs to the walker's followed series. Nothing for A20 / `md/acoustic.md` 7 beyond
  what A20 already states.

### 2.5 Line count and what remains

`src/ray_mesh_interact.cpp` 5130 lines (5089 at 238444d). Counts: `[wave]` 165110 / 179, `[acoustic]` 39371 / 54, whole
suite 1314030 / 1140. Not done here (phase 3): `md/acoustic.md` (2.3, 2.6, 2.7, 3.1, 3.2, 3.8, 3.9, 4, 6, 7, 10, the
appendix with the two new cases; `check_docs.py` reports them missing until then), `paris_cf.py` to `tools/acoustic`,
the side findings in their existing places (S1 rewritten as above). The three `test*.txt` baselines are not
regenerated; that follows the commit.

Stopped after phase 2.

## 3. Phase 3 — documents (2026-10-16)

Decided after phase 2 (Stephan, 2026-10-16): S1 recorded in `md/acoustic.md` 7, no new item; the two corrections
(`ray_state_update`'s description of `eps` in `src/ray_mesh_interact.cpp`, comment only, and `md/wave.md` 8); the peak
columns in `tools/acoustic/band_probe.cpp` and `tools/acoustic/paris_cf.py`, each with a README line.

### 3.1 What changed

- **`src/ray_mesh_interact.cpp`** (comment only): the `eps` entry of the `ray_state_update` description. `ρ` is
  `|r_near·r_far|·|φ|²`, not bounded by one (next to a lossy or evanescent layer a face can reflect more than one in
  amplitude); a series is summed where `ρ ≥ eps`, so `eps ≥ 1` re-emits every series whose `ρ` is below one, and a
  round trip at or above one is always summed. 5132 lines (5130 after phase 2, 5089 at 238444d; +70 / −27 against
  238444d). Rebuilt; `[wave]` and `[acoustic]` rerun: no row moved (only the seed line). Other places still say "1 =
  never" for the same threshold (`calc_diffraction_gain`'s description, the `thin_slab_threshold` comment in
  `include/quadriga_tools.hpp`); not touched (not in the decision), recorded here.
- **`md/wave.md`** (+6 / −4): the section-8 sentence as proposed in 2.4 ("summed wherever its round trip `ρ` reaches
  `thin_slab_threshold`, and a round trip above one always does"); the quoted counts of section 0 (39371 / 54,
  1314030 / 1140). Nothing else; no wave convention touched.
- **`md/acoustic.md`** (+119 / −30, 1235 lines):
  - 2.3, the absorption of an outer face where peaked: `a(θ)` and the plate's transmission change (0.5);
  - 2.6, the peaked face as a series impedance: `ζ`, the branch, the gate, the pad in front, met against the side
    the wave is on, the TE / TM forms, the measured phase and angle shape, the lining, the partitioned cavity,
    "Asserted in" with the four cases. The `tfQ ≥ 0` requirement is gone from the text, and "a rigid face reflects
    `1 − tf` at every angle" now reads "without a peak";
  - 2.7, the `tf` row and the phase row;
  - 3.1, the per-octave row, the coefficient-type row, the angle row and the unpartitioned-cavity row;
  - 3.2, the peak-shape row (Paris and field differ for a peaked face) and the phase row;
  - 3.8 and 3.9, expressible for locally reacting resonant faces;
  - 4, M6 built with D1–D5;
  - 6:
    - A12 closed for locally reacting faces of every type; what remains: a flat `tf` stays flat in angle and keeps the
      hard wall's phase, and an unpartitioned cavity goes the M9 route;
    - A13: field and Paris values of a peaked face differ (0.017 per octave; a Paris fit misses the targets in the
      field value by up to 0.062);
    - A20 closed for the peak, open for the power-law part;
  - 7, the last bullet amended (flat in angle except a peaked face) and four bullets:
    - the followed divergence behind a lining rarer than air;
    - the one-angle value at a lossless lining's critical angle (M5 as well);
    - S1, with its condition, numbers and "cause not traced";
    - a peak on a non-rigid body;
  - 9, the parameter-rule note: the default fits four rows physically; double glazing needs `tfQ < 0`;
  - 10, the decisions and "Built (2026-10-16)" with the line count and the counts;
  - the appendix: the two new cases, and section numbers added to the restated one.
  - The intro's `[acoustic]` count is updated. Section 10's historical counts (M5 / A21, M9) are left as they were.
- **`tools/acoustic`**:
  - `paris_cf.py` (new, 88 lines):
    - `face` prints a peaked face's `ζ`, `α₀`, Paris and field values; the board gives Paris 0.183 / 0.420 / 0.290 /
      0.121 at 63 / 150 / 250 / 500 Hz, as in 0.1;
    - `check` puts the closed form within 4.9e-10 of the numerical integral and reproduces Jeong's 0.951 at 1.567 and
      0.168 / 0.489 / 0.791 at 40 / 10 / 4.
  - `band_probe.cpp`: `tfP`, `tfF`, `tfQ` appended to the column list (24 per material) and a line in its header
    comment. `band_probe_example.in` now has three zeros per material; its output is byte-identical to before.
  - `README.md`: a table row for `paris_cf.py` and its usage section; the band probe's row and its 24 columns.
- **Baselines**: the three `test*.txt` regenerated (`md/wave.md` 0) on the built library before `check_docs.py`, load
  1.49 before and 6.73 after: `[wave]` 165110 / 179, `[acoustic]` 39371 / 54, whole suite 1314030 / 1140, all
  passed. The whole suite is identical to phase 2's run apart from the seed line.
- **`python3 tools/acoustic/check_docs.py`**: all checks pass. 54 `[acoustic]` cases, all named. The quoted counts match
  the regenerated baselines. Every source key resolves (the two known gaps of the review index as before).
- Not touched: `html_docu` (generated), `md/acoustic_manager.md` (the manager's), `tests/data/wave_twins.tsv`.

### 3.2 Run times

`--durations yes` on the built library: *Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side
absorption of a mass-law face* 0.002 s; *Absorption peak - a peaked face is its series impedance in magnitude and
phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches* 0.027 s; *ray_progress - the plate rule
and the absorption peak at every frequency of a band* 0.002 s; *ray_progress - the random-incidence absorption of a
peaked lining is the Paris integral of its series impedance* 0.034 s. Together 0.065 s. Load 6.19 before and after
(other sessions had started again); the same within 1 ms as the idle run of 1.4 (load 3.98).

### 3.3 State for the commit

Changed: `src/ray_mesh_interact.cpp`, `tests/catch2_tests/test_acoustic_state_update.cpp` (+335 / −26),
`tests/catch2_tests/test_acoustic_progress.cpp` (+149 / −9), `md/acoustic.md`, `md/wave.md`, `tools/acoustic/README.md`,
`tools/acoustic/band_probe.cpp`, `tools/acoustic/band_probe_example.in`. New: `tools/acoustic/paris_cf.py`,
`md/m6_report.md`, `md/m6_brief.md` (the manager's). Nothing committed.

Stopped after phase 3.
