# M9 fixtures — report

Brief: `md/m9_fixtures_brief.md`. One section per phase. Rig: `build_linux/scratch/m9` (`README.txt` lists every file).

## Phase 0 — measurements in scratch (2026-10-15)

No change to `src/`, `tests/` or any existing document; the only file written outside scratch is this report.

### 0.1 Baseline check

`cmake --build build_linux` was up to date. The three runs of `md/wave.md` section 0:

| run | result | against the previous baseline file |
| --- | --- | --- |
| `[wave]` | 165110 assertions in 179 cases, all passed | no line differs (timing lines excluded) |
| `[acoustic]` | 32697 assertions in 46 cases, all passed | no line differs |
| whole suite | 1307356 assertions in 1132 cases, all passed | no line differs |

### 0.2 Rig and the reproduction of A18

`m9` copies `walker3.cpp`, `buildw.sh`, `a18.py` from `ac2`. `w3` rebuilt on today's library; `a18.py` for P, S, DM and DS
gives an `a18.out` that is **byte-identical** to `ac2/a18.out`, and its summary equals `ac2/a18_sum.out`.

New instruments:
- `w4` (`walker4.cpp`): the a1 walker with every material column (power laws on `ε` and `μ`, needed for a porous fill)
  and a count of traced events per ray. It reproduces `w3` on A18's S rows digit for digit.
- `rp4` (`rp4.cpp`): `rp3.cpp` extended to a palette and a face list in the walker's spec format, a band per line.
- `m9lib.py`: the reference, fluid layers in (p, v_n) with `ρ = μρ₀`, `K = ρ₀c₀²/ε`, `exp(+jωt)`; backings: a fluid
  half-space, rigid, a pad.

**Reference planes** (needed for every phase statement below; measured, then used unchanged): a sheet's phase is
referenced at its mesh plane. An isolated sheet lies about it (`r = r_front·e^{jk₀h·cosθ}`); a glued entry lies in front of
the solid (`r = r_front·e^{2jk₀h·cosθ}`); `T` collected at the last mesh plane carries `e^{jk₀cosθ·(stack thickness − mesh span)}`.
With these, the phase of every summed row below is within the bounds stated. (A first T reference with the wrong plane
read 16.8° at 1 kHz, exactly `k₀·0.016`; corrected, 0.028°.)

### 0.3 The single leaf (for fixture 1)

4 mm glass, `m = 0` (`a` 1.767e-6, `μ` 2083, `m″` 9.998 kg/m²), a sheet in air, walker, 50 Hz–1 kHz, 0° to 85°, tracers 4
and 5 (`b8.out`, `b0.out`):

| against | level | phase |
| --- | --- | --- |
| the fluid layer (thickness 4 mm, mesh-plane reference) | 4.9e-6 dB | 1.6e-5° |
| the limp mass law `t = 1/(1 + jωm″cosθ/(2Z₀))`, `r = 1 − t` | T 0.0078 dB (1 kHz, 85°), R 2.5e-4 dB | 4.2° at 1 kHz, 0° |

The 4.2° is exactly `k₀h` at 1 kHz, the air path over the leaf's thickness, which the zero-thickness law lacks; the level is
the law's. Fixture bound proposed: level against the limp law 0.01 dB; phase against the fluid layer 1e-3°.

### 0.4 The boundary question (decided by measurement, proposal stands)

The leaf on an air box (12.5 mm board of A18, 8.6 kg/m², `μ″/μ′` 0.2 as the mounting resistance, box 25 mm) on three
backings, summed and followed, tracers 4 and 5, 50 Hz–1 kHz, 0° to 60°, the wall as a half-space (sink) and as a finite
traced wall of 200 mm (`b1_S.out`, `b1_P.out`):

- (a) the concrete half-space of A18 (`m = 0`);
- (b0) a rigid mass-law wall, two-parameter concrete, `m` 20, `fRef` 5.0212e-5, `thk` 0, `tf` 0;
- (b2) the same with `tf` 0.02.

**(b0) changes nothing:** on 896 rows (box and meshed gap, both tracers, both modes, half-space and finite wall) R is equal
to (a) bitwise; the complex R differs by at most 1.3e-14. A closed mass-law face without `tf` reflects the Fresnel
coefficient of its two-parameter half-space. **(b2)** is on the reference of a pad in front of that half-space (backing
reflection `√(1 − tf)·r_half`, absorbing `tf` at every angle seen from air) within 4.1e-5 summed, 1.8e-6 followed. The finite
wall gives the same R as the half-space.

The rigid backing (`r_b = 1`) is **not** the reference: the two-parameter concrete reflects with a small phase (0.104° at
60° on a bare wall), and at the board's sharp resonance at 60° this moves R by up to 2.28e-2 (250 Hz). Every fixture reference
uses the half-space of the mapping (or the pad on it), not a rigid wall.

**So the proposal holds:** the fixtures put the box on a mass-law wall (b0 or b2); they evaluate a material with `m > 0`, are
acoustic cases in `test_acoustic_*`, and `[wave]` stays untouched. Open for Stephan: fixture 1 (the leaf alone) and fixture 3
(4|16|4, air on both sides) evaluate no mass-law material, so by `md/wave.md` 0 they are wave cases. Either they go to the new
acoustic file anyway with one sentence in `md/wave.md` 0 ("the M9 leaf cases guard the acoustic route and evaluate wave
media only"), or 4|16|4 is mounted in a frame of a mass-law material that no ray meets (a formality), or they go to `[wave]`
(which the brief excludes). Recommended: the first.

### Item 1 — the backing

Covered by 0.4. Largest deviations (summed / followed), 56 rows per configuration and mode:

| configuration | reference | R summed | R followed | phase summed | events per ray (summed / followed) |
| --- | --- | --- | --- | --- | --- |
| board on box, (a) or (b0) | concrete half-space | 6.70e-5 (140 Hz, 30°) | 2.1e-6 | 0.0054° | 5 / up to 8000 |
| board on box, (b2) | pad on the half-space | 4.14e-5 (112 Hz, 0°) | 1.8e-6 | 0.0027° | 5 / up to 7088 |
| board over meshed gap, (a)/(b0) | concrete half-space | 2.1e-6 | 2.1e-6 | 1.7e-4° | 1126 – 8000 |
| board over meshed gap, (b2) | pad on the half-space | 1.2e-6 | 1.2e-6 | 8.7e-5° | up to 7088 |
| finite wall 200 mm instead of the half-space | as above | identical | identical | identical | +2 |

The summed box carries a residual of 5.5e-5 to 6.7e-5 at every angle's resonance that the followed series does not
(2e-6). It is not a depth offset (scanning the reference depth by ±1e-5 relative does not close it) nor the `μ` of the air
body (1.000001 in the reference: 6.77e-5); cause not traced. Bound proposed for the box: R 2e-4, phase 0.02°.

What did not run: nothing.

### Item 2 — tracer 5 against tracer 4

Row by row, tracer 5 against tracer 4: the meshed gap and meshed 4|16|4 identical; 4|16|4 on the box 1e-10; the board on the
box 1.1e-6 summed and 4.7e-6 followed (both on the reference within 6.7e-5 and 3.2e-6). The leaves are sheets and do not bend; the air body (`n` 1.0000005) bends nothing measurable. A
porous body does refract: the filled box (item 4) reads 1.34e-4 with tracer 5 against 7.0e-5 with tracer 4 (summed, same
reference). 4|16|4 on the box: identical for both tracers (`b2.out`). Bound proposed: the tracer-4 bound for both, 5e-4 on a
porous body.

### Item 3 — a band through `ray_progress`

`rp4`, one ray until none is left, scalar, `thin_slab_threshold` 0 and 0.15, base frequency the first of the band
(`b3.out`, `b3_PB.out`, `b3_DM.out`, `b3f.out`):

| configuration | band | band vs single frequencies | band vs reference | rays per run |
| --- | --- | --- | --- | --- |
| board on box, wall (b2) 200 mm | 80–315 Hz, 13 frequencies, 0°/30°/45°/60° | **bitwise equal**, both tracers, both thresholds | R 4.14e-5, phase 0.0027° (as the walker) | 6 |
| 4\|16\|4 on box | 125–400 Hz, 11 frequencies | bitwise equal | T 7.00e-3 dB, R 4.30e-4, phases 0.028° | 5 |
| board on a porous body (filled box), wall `tf` 0.02 | 80–315 Hz, base 80 or 160 Hz | 2.8e-7 (tracer 5), 0 (tracer 4) | 7.47e-2: see finding F1 | — |
| board over meshed gap | 80–315 Hz, maxint 255 | bitwise equal | **3.68e-3** (truncated series) | 637 |
| 4\|16\|4 meshed | 125–400 Hz, maxint 255 | bitwise equal | **9.12 dB**, R 0.297 | 510 |

The default threshold 0.15 sums the box (its round trip is near one), as threshold 0 does. A meshed gap is cut by the 255
interactions `ray_progress` allows: unusable for a leaf through `ray_progress`. The base frequency's geometry does not show:
the box and the sheets do not bend, and the porous body's band differs from its single frequencies by 2.8e-7. Bound proposed:
band against single 1e-6; band against the reference as the walker's bound (2e-4 in R, 0.02 dB in T for 4|16|4).

### Item 4 — a porous fill in the box

**The columns.** Miki at a measured σ (Komatsu 2008, Table 3: rock wool 06, 46.2 kg/m³, **25.1 kPa·s/m²**; and rock wool
07, 78.4 kg/m³, 46.0 kPa·s/m²), two-parameter mapping `μ = zc·kc`, `ε = kc/zc`. Miki's own `μ′` is **negative below
f/σ ≈ 0.0064** (−12.9 at 50 Hz for 25.1 kPa·s/m²; Miki's validity is f/σ ≥ 0.01), so no power-law column (`e·r^f > 0`) can
follow it there. Fitted on the quantities Miki states (`zc`, `kc`), 40 Hz–1 kHz, minimax: within 11.0 % (25.1), 12.3 % (32.8)
and 13.5 % (46.0) (`porcol.out`). Shearer's RW3 (60 kg/m³, not measured) takes 32.8 kPa·s/m² from the review's rock-wool
regression `53·ρ^1.57` (bracketed by the two measured rows).

**The machine** (`b4_*.out`, summed, wall (b0), 50 Hz–1 kHz, 0° to 60°; reference with the fitted columns):

| configuration | R vs columns | phase | R vs Miki itself | events per ray |
| --- | --- | --- | --- | --- |
| F1 board glued on a porous body 25 mm | 7.0e-5 (tracer 5: 1.34e-4) | 0.010° | 6.6e-2 | 5 |
| F2 board on air box 10 mm on porous body 15 mm | 1.34e-4 (tracer 5: 2.1e-4) | 0.018° | 7.3e-2 | 248 – 3314 (depth 4000; at depth 400: 5.6e-4) |
| F3 board on porous body 15 mm on air box 10 mm | no finite path sum (2e247) | — | — | — |
| SH1 Shearer filled, MDF on air 20 mm on rock wool 75 mm | 1.6e-5 | — | third-octave random α within 0.046 | 55 – 170 |

- **The followed series of a filled box diverges.** F1 followed reaches 2.2e22 at 50 Hz, 0°. By formula (`b4.py` columns,
  leaf seen from inside the fill): `|rn| = 1.569`, `|rf| = 0.9995`, `|φ²| = 0.730`, round trip **1.145** at 50 Hz, 0° (0.997 at
  60°, 0.90 at 100 Hz). A lossy layer between strong reflectors (`md/wave.md` 7): only the summed row means anything; the
  fixtures use the summed mode.
- **F3 (leaf | porous | air | wall, the brief's order) is not expressible as two bodies:** the coupled pair has no finite path
  sum. The same order as an ordered stack (leaf and porous sheet co-located, glued on the air box) did not return within 10 min
  at depth 4000, nor within 120 s at depth 60 (a runaway tree); no result. F2 (air in front, fill at the back, Shearer's
  build) converges.
- A porous **sheet** between the air box and the wall is outside the documented use (`md/wave.md` 6: "Air is assumed beyond
  every glued exit"): measured R 0.10 against 0.99 on the mass-law wall and up to 48 (energy created) on an `m = 0` wall
  (`b4v.py`). The porous sheet glued on the wall alone is exact. Finding F2.

**Shearer's MDF membrane** (6 mm, 4.65 kg/m², 95 mm; filled: 20 mm air + 75 mm RW3 [Shearer_2016_AbsorptionBelow100Hz,
Sec. 3.6]); the leaf `μ` 645.8 (775 kg/m³), `ε` 4.5e-5 (assumed `c` ≈ 2000 m/s; negligible in a thin leaf), loss ratio 0.2
(A18's mounting resistance, not fitted). Random incidence (Paris, 12-point Gauss over 0–90°), fine sweep in 1/24 octave
(`b6_empty.out`, `b6_filled.out`; library on the reference within 2.2e-4 in R on every row):

| case | measured peak third | M9 random-incidence peak | M9 normal-incidence f₀ | f₀ / rule (60 empty, 50 filled) | angle shift f(30°)/f(45°)/f(60°) over f(0°) |
| --- | --- | --- | --- | --- | --- |
| empty | 100 Hz (α 0.30) | 111.2 Hz (α 0.448; thirds: 100 Hz 0.404, 125 Hz 0.422) | 89.5 Hz | 0.99 | 1.156 / 1.417 / 2.006 |
| filled | 80 Hz (α 0.48) | **82.7 Hz** (80 Hz third, α 0.84) | 79.6 Hz | 1.06 | 1.021 / 1.043 / 1.068 |

The peaks land in the measured thirds (the empty one on the boundary at 112 Hz). The heights are 1.5 to 1.75 times the
measured ones; the loss ratio is an assumption and ISO 354 under-reports at these frequencies (Shearer). The fill nearly
removes the angle shift (the wave in the fill is damped laterally).

Bound a fixture would assert: R against the columns 5e-4 (F1, F2, both tracers), summed only; F2 at depth ≥ 4000. Not
asserted: anything against Miki itself (the fit, not the machine) or against Shearer's heights.

### Item 5 — A19's shallower box

Shallower box: every depth × `1/1.34²` = 0.5569. Framed cases of `a19.out`, random incidence (they are ISO 354 / JIS A 1409
measurements; Sakagami 1.8 × 3.6 m panels), the same leaf loss 0.2, plywood `μ` 441.7 (530 kg/m³):

| case | measured | full depth: random peak (ratio to measured) | shallow box: random peak (ratio) | full depth f₀(0°) / 60-rule |
| --- | --- | --- | --- | --- |
| Shearer MDF 95 mm empty | 100 | 111.2 (1.11) | 148.8 (1.49) | 0.99 |
| Sakagami plywood 30 mm | 315 | 338.1 (1.07) | 452.0 (1.43) | 0.99 |
| Sakagami plywood 90 mm | 250 | 199.3 (0.80) | 263.2 (1.05) | 0.99 |
| Sakagami plywood 150 mm | 160 | 157.6 (0.99) | 206.4 (1.29) | 0.99 |
| Shearer MDF filled | 80 | 82.7 (1.03) | 110.2 (1.38) | 1.06 (50-rule) |

- **f₀** at normal incidence is the 60-rule's (0.99) at full depth and 1.33 times it in the shallow box, as designed.
- **The angle shift** is `1/cos θ` (1.156 / 1.417 / 2.006 at 30° / 45° / 60°) in both boxes; it moves the
  random-incidence peak 1.24 to 1.30 times above f₀.
- **Height and width:** random incidence 0.44–0.52 at full depth, 0.39–0.50 in the shallow box; half-height width 1.31–1.65
  octaves (full) and 1.27–1.44 (shallow); normal incidence 0.93–0.99 high, 0.45–0.77 octaves wide.
- **Against the measured framed peaks** the full-depth box reads 0.80 to 1.11 (geometric mean 0.99), the shallow box 1.05 to
  1.49 (geometric mean 1.30). The shallow box counts the rise twice: A19's 1.34 is measured over the 60-rule's
  normal-incidence f₀, and under diffuse incidence an unpartitioned box already lifts its peak 1.24–1.30 times by the angle
  shift. Finding F3 (a proposal on the A19 rule under M9).

What did not run: Oldfield's clamped plates (measured on a loudspeaker surround and clamped; the measurement method was not
re-read here) and a partitioned cavity (a honeycomb would remove the shift; M9 has no partition other than meshing one).

### Item 6 — a curtain

Pieren's fabric is `R_s ∥ jωm`. A fluid leaf of thickness `h` has the series impedance `jωρ₀hμ`, so the parallel form is
`μ(f) = (m/(ρ₀h)) / (1 + jωm/R_s)`, **a Debye relaxation** at `f_t = R_s/(2πm)`. Fabrics of [DelRey_2013_CurtainFullness,
Table 1] (`R_s = σ·thickness`), flat at 145 mm (DelRey's middle distance) from the wall, α per third octave 125 Hz–4 kHz
(mean of 5 frequencies), normal and random incidence (`b7r_T*.out`, `b7_145.out`, `b7w*.out`):

| fabric | R_s Pa·s/m | m kg/m² | f_t Hz | fluid layer with the exact Debye μ (0° / random) | columns fitted on both | columns fitted on random incidence: random / 0° |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | 142 | 0.230 | 98 | 0.003 / 0.013 | 0.024 | 0.012 / 0.069 |
| T2 | 144 | 0.300 | 76 | 0.001 / 0.002 | — | 0.008 / 0.016 |
| T3 | 192 | 0.095 | 322 | 0.002 / 0.003 | **0.059** | 0.032 / 0.168 |
| T4 | 242 | 0.200 | 193 | 0.004 / 0.006 | **0.055** | 0.030 / 0.215 |
| T5 | 225 | 0.200 | 179 | 0.003 / 0.007 | — | 0.026 / 0.107 |
| T6 | 455 | 0.300 | 241 | 0.010 / 0.015 | — | 0.039 / 0.145 |
| T7 | 375 | 0.250 | 239 | 0.006 / 0.007 | — | 0.037 / 0.137 |

- The fluid leaf carries the parallel form (0.015 at worst). **The power-law columns** carry it within 0.05 in
  random-incidence α for all seven fabrics when fitted on random incidence, but then miss single angles (up to 0.35 at normal
  incidence and one frequency, walker). Fitted on both, they hold 0.05 only where `f_t` lies below the band (T1 0.024); with
  `f_t` in the band (T3, T4) they miss narrowly (0.059, 0.055). A power law cannot turn from `μ′ ≈ const` to `μ′ ∝ f⁻²` inside
  the band.
- The walker on the fitted columns (fabric glued on a 145 mm air box on the wall (b0), 0° to 85°, both tracers): within 5.9e-5
  in R of the column reference, 5 events per ray.

What did not run: the G-100 curtain reports (fullness, hung with folds; no model, DelRey ditto); a curtain over a meshed gap.

### Item 7 — the cost

Traced events per ray (walker, both passes counted; floor 1e-24 in power):

| configuration | summed | followed |
| --- | --- | --- |
| leaf on air box on a wall (half-space / finite wall) | 5 / 7 | 1126 (50 Hz) to 8000+ (≥ 315 Hz, depth cap 4000) |
| 4\|16\|4 on air box | 5 | 1584 (50 Hz) to 8000+ (≥ 125 Hz), 7.3 dB off at 1 kHz |
| leaf over meshed gap | = followed: 1126 (50 Hz), 3426 (125 Hz), 8000+ (≥ 315 Hz) | same |
| 4\|16\|4 meshed | 1584 to 8000+; 7.30 dB off at 1 kHz | same |
| leaf on a porous body (filled) | 5 | diverges (round trip 1.145) |
| leaf on air box on porous body | 248 (1 kHz) to 3314 (80 Hz), 25 mm build; 55–170 Shearer's build | runaway |
| fabric on air box | 5 | — |
| `ray_progress`, box | 5–6 rays per band | — |
| `ray_progress`, meshed gap (cap 255 interactions) | 510–637 rays, 3.7e-3 in R / 9.1 dB off | — |

**Rule of use that follows (proposal for `md/acoustic.md` 2.8):** the leaf is a wave-medium sheet (`m = 0`, two-parameter
mapping, its loss a constant ratio on `μ″`: `g = e·rr/17.98`, `h = 1`, `fRef = 1`) glued on a meshed box of near-air
(`μ` 1.000001) glued on the wall; the wall keeps the mass column (with or without `tf`); the box is summed (any
`thin_slab_threshold` up to 0.15 sums it). A fill is a porous body glued at the back of the air box (air in front), at hundreds
to a few thousand events; a fill directly behind the leaf with air behind, a porous sheet between box and wall, and a meshed gap
through `ray_progress` are out.

### Findings

- **F1 — `tf` behind a porous fill is a pad matched to the fill.** With the porous body glued on a wall with `tf` 0.02 the
  library is within 4.3e-5 of a pad formed on the modulus of the fill's normal impedance (`md/acoustic.md` 2.3: "next to a
  lossy medium the flux the pad hands over"), within 8.9e-3 of a pad on the complex impedance, and **4.0e-2** off a wall whose
  absorption is fixed as seen from air (the reference of item 1); walker and `ray_progress` alike (band 7.5e-2 off the
  air-referred reference). With `tf` 0, or with air in front of the wall, it is within 5e-5. The convention is the documented
  one; whether a wall's `tf` should depend on what lies in front of it is Stephan's decision. Proposed for the fixtures: a fill
  against a wall with `tf` 0, or the fixture asserts the documented convention.
- **F2 — a porous sheet between an air box and a wall** (outside the documented use, `md/wave.md` 6) gives R 0.10 for 0.99 on a
  mass-law wall and up to 48 on an `m = 0` wall (energy created, no graceful failure). By the triage rule a buildable invalid
  geometry gets a graceful-failure assertion; proposed as an open item, not a fixture of this brief (it needs a library change).
- **F3 — A19's shallower box double-counts under M9** (item 5). Proposal: an unpartitioned leaf under M9 takes its real depth;
  the ×1.34 stays the rule for `tfF` / `resF` (the locally reacting peak, M5/M6). The full-depth random-incidence peaks are within
  0.80–1.11 of the four framed measurements (filled Shearer 1.03).
- **F4 — the followed series of a filled box diverges** (round trip 1.145 by formula), and the coupled pair leaf | porous | air
  has no finite path sum; the summed route is the only one. Known class (`md/wave.md` 7, accepted limitations).
- **F5 — the meshed gap is out of reach of `ray_progress`** (255 interactions): 3.7e-3 in R on the board, 9.1 dB on 4|16|4.
- **F6 — Miki is outside its range for a filled cavity at 50–160 Hz** (its `μ′ < 0` below f/σ 0.0064); the power-law columns
  are within 11–13.5 % of its `zc`, `kc` and move R by up to 0.073 at single angles, random third-octave α by up to 0.046.
- **F7 — the curtain's parallel form is a Debye relaxation of `μ`.** The power-law columns hold 0.05 in random-incidence α
  (fitted on it), not per angle where `f_t` lies in the band. No library change is in this brief; recorded for Stephan.
- **F8 — the summed box carries a residual** (6.7e-5 in R on the board, 4.3e-4 / 7.0e-3 dB on 4|16|4) that the followed series
  does not; cause not traced, small against every tolerance.

### Proposed fixtures for phase 1 (bounds from the measured maxima)

| fixture | rows | reference | bound |
| --- | --- | --- | --- |
| Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf | 4 mm glass, 50 Hz–1 kHz, 0°–85°, tracers 4/5 | limp law (level), fluid layer (phase) | 0.01 dB; 1e-3° |
| Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle | board on box on wall (b0) and (b2), 14 frequencies, 0°–60°, summed and followed (depth 4000), half-space and finite wall | fluid layers on the half-space / the pad on it | summed 2e-4, followed 1e-5 in R; phase 0.02°; the angle shift `1/cos θ` of the resonance (corrected in phase 1: the 30° peak is at 144 Hz, not 160) |
| Leaf - two leaves on a box of air resonate at the mass-air-mass frequency | 4\|16\|4, summed | fluid layers, air both sides | T 0.02 dB, R 1e-3, phase 0.1°; dip at 200 Hz (0°) and 315 Hz (45°) |
| ray_progress - a leaf on a box of air at every frequency of a band | board on box on (b2); 4\|16\|4 | single frequencies; fluid layers | band = single 1e-6; reference as above |
| Leaf - a porous fill at the back of the box (F1, F2 or SH1) | summed, depth ≥ 4000, wall `tf` 0 | fluid layers with the same columns | 5e-4 in R |
| Leaf - a fabric is a resistive leaf (one DelRey fabric) | walker on the fitted columns | the columns' fluid layer; the parallel form in random-incidence α per third | 2e-4 in R; 0.05 in α |

Recorded, not asserted: the meshed gap (cost, `ray_progress`), the shallower box, Miki and Shearer's heights, F1–F2.
Open for Stephan before phase 1: the boundary of fixtures 1 and 3 (0.4); F1 (which `tf` convention a fill fixture asserts);
F3 (the A19 rule under M9); whether the porous-fill and curtain rows become fixtures; new file or appended.

## Phase 1 — fixtures (2026-10-15)

Decisions of Stephan on phase 0: a new file, rows 1–4 of the proposed table with their bounds, one porous-fill fixture on
Shearer's filled build (summed, wall `tf` 0, bound 5e-4), no curtain fixture; fixtures 1 and 3 stay in the acoustic file
(the `md/wave.md` 0 sentence follows in phase 2).

### The file

`tests/catch2_tests/test_acoustic_leaf.cpp`, 524 lines, five cases tagged `[acoustic]`, each under `ub::OneThread`. The
namespace `lw` at its head holds what the cases share:

- `lw::stack`, the reference: fluid layers in (p, v_n) with `ρ = μρ₀`, `K = ρ₀c₀²/ε`, impedances relative to `ρ₀c₀`, on air or on
  a fluid half-space behind a matched pad (`√(1 − tf)·r_half`; `tf` 0 is the half-space itself). The columns enter through
  their definition (`ε = a r^b − j17.98 c r^d/f`, `μ` alike), not through library code.
- `lw::walk`, `cf::walk` without the recursion. The recursion is one stack frame per event, and a followed lossless box needs
  thousands: `w4` at depth 4000 segfaults at 1 kHz under the default 8 MB stack (rc 139). The case on the box asserts it
  equal to `cf::walk` on every summed half-space row (within 1e-12 in complex R).
- The materials: the board, 4 mm glass, the mass-law wall (`tf` 0 or 0.02), the near-air body.

| case | rows | asserted | measured max | assertions | time |
| --- | --- | --- | --- | --- | --- |
| *Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf* | 4 mm glass in air, 50 Hz–1 kHz, 0°–85°, tracers 4/5 | level against the limp law 0.01 dB (T and R), against the layer 1e-4 dB; phase of T and R against the layer 1e-3° | as phase 0 (0.0078 dB at 1 kHz, 85°) | 350 | < 0.1 s |
| *Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle* | the board on the box on the wall (`tf` 0, 0.02), half-space and 200 mm wall, tracers 4/5, summed and followed (depth 4000), 14 frequencies × 4 angles; the angle shift on a 1/96-octave sweep | R summed 2e-4, followed 1e-5; phase 0.02° / 2e-3°; drop 1e-8; summed ≤ 7 events; equal to `cf::walk`; the peak against the reference's 0.1 % and against `1/cos θ` 2 % | R 6.50e-5 summed, 4.72e-6 followed; peaks 128.36 / 148.60 / 182.36 / 258.81 Hz at 0° / 30° / 45° / 60°, ratios 1.158 / 1.421 / 2.016, equal to the reference's | 3368 | 17.6 s |
| *Leaf - two leaves on a box of air resonate at the mass-air-mass frequency* | 4\|16\|4 on the box, summed, tracers 4/5, 14 × 4 | T 0.02 dB, R 1e-3, phases 0.1°, ≤ 5 events; the TL dip at 200 Hz (0°) and 315 Hz (45°) on the grid | as phase 0 (T 7.0e-3 dB) | 564 | < 0.1 s |
| *ray_progress - a leaf on a box of air at every frequency of a band* | the board on its box on the wall (`tf` 0.02, 200 mm), 80–315 Hz; 4\|16\|4, 125–400 Hz; tracers 4/5, 4 angles, threshold 0.15 | band against each frequency alone 1e-6 (complex R and T); the reference as above | as phase 0 | 840 | < 0.1 s |
| *Leaf - a porous fill at the back of the box is the transfer matrix of leaf, air, fill and wall* | Shearer filled: MDF \| air 20 mm \| rock wool 75 mm (Miki 32.8 kPa·s/m² columns) \| wall `tf` 0; summed, tracers 4/5, 13 frequencies 40 Hz–1 kHz × 4 angles | R 5e-4 against the same columns; drop 1e-12 | R 2.57e-5, up to 170 events | 208 | 0.2 s |

Corrections against the phase-0 proposal, made before the run: the angle shift is asserted as a fine-sweep peak, because the
"160 Hz at 30°" of A18's text is the grid neighbour of a peak at 144 Hz (148.6 Hz on the 1/96-octave sweep: the reference reads the
same); the bound against the lumped `1/cos θ` is 2 %, as the matrix itself reads 2.016 at 60°.

### Counts and the diff check

From the repository root, after `cmake --build build_linux` (load average 1 to 7 during the runs, no other `test_bin`):

| run | before | after | difference |
| --- | --- | --- | --- |
| `[wave]` | 165110 / 179 | 165110 / 179 | none; `test_wave.txt` identical (timing lines excluded) |
| `[acoustic]` | 32697 / 46 | **38027 / 51** | +5330 assertions, +5 cases |
| whole suite | 1307356 / 1132 | **1312686 / 1137** | +5330, +5 |

`test_acoustic.txt` and `test.txt`: no existing line removed or changed except the summary line; one inserted block of 1397
lines (the six diagnostic tables of the new file, which runs first). The baselines in the repository root are replaced; the
previous ones are in `build_linux/scratch/m9/old_baselines/`.

Not asserted, recorded in phase 0: the meshed gap, the shallower box, the curtain (F7), `tf` behind a fill (F1), the porous sheet
between box and wall (F2), a fill directly behind the leaf, Miki itself and Shearer's measured heights.

## Phase 1, revised (2026-10-15, after Stephan's review)

Three changes to `test_acoustic_leaf.cpp` (now 547 lines, was 524):

1. **Leaf on the box.** The summed rows stay for every combination (`tf` 0 and 0.02, half-space and 200 mm wall,
   tracers 4 and 5, 14 frequencies, 0°/30°/45°/60°) with the cost guard (≤ 7 events), the `cf::walk` cross-check
   and the angle-shift sweep. The followed mode runs on six rows only (`tf` 0.02, half-space, tracer 4, 50 / 125 /
   250 Hz, 0° and 60°), stopping at a path power of 1e-12 (`lw::walk` takes the floor as an argument, 1e-24 by
   default); asserted: R within 1e-5 of the reference, dropped power below 1e-8. Measured: the followed rows within
   3.38e-6 (486 to 1870 events per ray, dropped 8.5e-13 to 1.0e-12); the summed rows unchanged (6.50e-5).
   2042 assertions (was 3368), **0.148 s** (was 17.6 s).
2. **4|16|4.** The grid check of the dip is replaced by the dip on a 1/96-octave sweep (the peak of the transmitted
   power from 150/cos θ to 300/cos θ Hz, `lw::peak_f`): the library's dip against the reference's within 1e-3, and
   against `1/cos θ` of the 0° dip within 3 %. Measured dips: **211.29 / 243.97 / 298.82 / 422.64 Hz** at 0° / 30° /
   45° / 60° (reference 211.28 / 243.97 / 298.81 / 422.62), ratios 1.1547 / 1.4143 / 2.0003 against 1.1547 / 1.4142 /
   2.0000; the transmission loss at the dip 0.20 to 0.21 dB (the unit is lossless; the grid's 10.46 dB at 200 Hz was
   off the dip). 568 assertions.
3. **Run times** (`--durations yes`, `-# "[#test_acoustic_leaf]"`; load average 0.97 before and after, no other
   suite running):

   | case | time |
   | --- | --- |
   | *Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf* | 0.002 s |
   | *Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle* | 0.148 s |
   | *Leaf - two leaves on a box of air resonate at the mass-air-mass frequency* | 0.036 s |
   | *ray_progress - a leaf on a box of air at every frequency of a band* | 0.031 s |
   | *Leaf - a porous fill at the back of the box is the transfer matrix of leaf, air, fill and wall* | 0.141 s |
   | the five together | 0.36 s (4008 assertions) |

   In the whole-suite run with durations the five cases took 0.357 s of 58.7 s summed over all cases (that run
   loads the machine itself: load 9.8 at its end; the longest case is 12.8 s).

**Counts and diff check** (against the baselines before phase 1, `build_linux/scratch/m9/old_baselines/`):

| run | before phase 1 | now | difference |
| --- | --- | --- | --- |
| `[wave]` | 165110 / 179 | 165110 / 179 | none; `test_wave.txt` identical (timing lines excluded) |
| `[acoustic]` | 32697 / 46 | **36705 / 51** | +4008 assertions, +5 cases |
| whole suite | 1307356 / 1132 | **1311364 / 1137** | +4008, +5 |

`test_acoustic.txt` and `test.txt`: no old line removed or changed except the summary line; one inserted block of
962 lines, identical in both files and identical line for line to the output of the new file run on its own
(`-# "[#test_acoustic_leaf]"`). The baselines in the repository root are the new ones.

## Phase 2 — documents (2026-10-15)

**`md/acoustic.md`** (991 → 1108 lines; against the copy before this phase 19 lines changed or removed, 136 added;
the copy is `build_linux/scratch/m9/acoustic_before_phase2.md`):
- First step: the `[acoustic]` count 36705 / 51 and the new file named.
- Section 1: the leaf sheet asserted (the limp law within 0.0078 dB at 50 Hz–1 kHz, the phase the layer's), pointing to 2.8.
- **2.8 Leaves as wave-medium sheets** (new): the rule of use (the leaf, the box of near-air material, the wall, the
  summed box, the double leaf, a fill at the back of the box with `tf` 0 behind it), what it reaches (fine-sweep
  values), the limits (no bending; a meshed gap out of reach of `ray_progress`; a fill only at the back of the box; the
  divergent followed series of a fill, F4; tunnelling past the critical angle; the curtain's relaxation, M16; the
  summed residual), and "Asserted in" with the five cases.
- Section 3: 3.1 porous layer over an air gap (a leaf with the fill at the back expressible; a porous layer directly
  behind a leaf with air behind it not), panel over an unpartitioned cavity (expressible; the peaks 128.4 / 148.6 /
  182.4 / 258.8 Hz from the fine sweep, A18's grid values named as such; candidate M9, M7 dropped); 3.2 membrane peak
  frequency (A19 revised, the 1.24–1.30 lift of the random-incidence peak), curtains (the column result, M16); 3.4
  mass law (the new fixture), double leaves (the dips 211.3 / 244.0 / 298.8 / 422.6 Hz from the fine sweep).
- Section 4: M9 (the cavity a box, the edge sentence revised with A19, the measured-after-A18 line, "Built", A18
  closed); **M16** a relaxation term on `μ` for fabrics, proposal only.
- Section 5: a porous layer directly behind a leaf with air behind it, as a limitation in scope.
- Section 6: A18 removed (closed); A19 revised (the full-depth box 0.80–1.11 against the framed chamber peaks, the
  shallower box 1.05–1.49, the new rule); **A20** (a wall's `tf` behind a porous fill: the pad on the fill's immittance
  as 2.3 documents, the alternative fixed as seen from air, physically a fixed wall surface impedance; fixtures use
  `tf` 0); **A21** (a porous sheet between a glued body and a wall creates energy, R up to 48; a graceful failure
  needs a library change).
- Section 9: class 9 tests M16 too.
- Section 10: the A19 rule of the earlier decisions marked as revised; a block "after M9 phase 0" (the new file,
  fixtures 1 and 3 wave media only, `tf` 0 behind a fill, no curtain fixture, M16, A21, **A19 revised** with the new
  rule) and "Built" (2.8, five cases, no library change, A18 closed, the counts).
- Section 11: the `m9` rig. Appendix: the five cases with their sections.
- The double-leaf row of 3.4 now escapes `4\|16\|4`, which split its cells. Two rows I did not edit carry the same
  defect from before (3.1 angle dependence, `|R(θ)|²`; 3.4 coincidence and loss, a modulus): recorded, not changed.

**`md/wave.md`** (510 → 513 lines): section 0, the `[acoustic]` and whole-suite counts of the first step (36705 / 51,
1311364 / 1137) and one sentence: the M9 leaf cases in `test_acoustic_leaf.cpp` are `[acoustic]` although two evaluate
wave media only and the others wave media on a mass-law wall; they guard the acoustic route of `md/acoustic.md` 2.8.
Section 9: M9 rests on the wave media of the two-parameter mapping, so a change on the wave side is checked against
`test_acoustic_leaf.cpp` too.

**Line counts:** `tests/catch2_tests/test_acoustic_leaf.cpp` 547; `md/acoustic.md` 1108; `md/wave.md` 513; no change
to `src/` or to any other test file. Nothing committed.

**Correction of a request:** the review asked for the board's peaks as 125 / 144 / 200 / 250 Hz, "using the
fine-sweep values". The fine sweep reads 128.4 / 148.6 / 182.4 / 258.8 Hz (the reference the same), so the documents
carry those: 144 Hz is `125/cos 30°`, and the 45° peak lies between the grid's 160 and 200 Hz.
