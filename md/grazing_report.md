# Grazing incidence on a wall — report (M14)

Brief: `md/grazing_brief.md`. One section per phase. Every number below comes from a run (script named), a closed form
evaluated by a script, or a cited page of a source; derivations that are mine say so. Scratch:
`build_linux/scratch/grazing` (not kept: the numbers that matter are in this file).

## 0. Phase 0 — the study (2026-10-17)

### 0.0 Baseline and rig

- `git status`: `md/acoustic_manager.md` modified, `md/grazing_brief.md` new; nothing under `src/`, `tests/`,
  `md/acoustic.md`. Last commit bc24b0c. `src/ray_mesh_interact.cpp` 5116 lines. `check_docs.py` passes.
- Built and ran the three suites from the repository root: `[wave]` 165258 assertions in 180 cases, `[acoustic]` 40050
  in 56, whole suite 1314857 in 1143, every one passed; each output identical to the stored baseline apart from the
  random-seed line.
- `t_fp.py` rerun on the copied `fp.py`: the windowed σ at 1 kHz is 1.0019 / 0.9994 at 0° (2 m / 8 m plate), 1.1519 /
  1.1548 at 30° (1/cos 1.1547), 2.0425 / 1.9859 at 60° (2.0000); finite and modal on 4 mm glass, 1.23 × 1.48 m: 18.46 /
  18.35 dB at 125 Hz, 27.24 / 27.35 dB at 500 Hz. Both checks reproduce the README (0.1 dB).
- The 13 fitted vectors of `md/m10_m15_report.md` 1.0 (`tx/fits/m10_90_final.json`) re-evaluated through the fitter:
  worst residual 2.97 dB, as reported.
- New σ tables (`sigcache.py`: `fp.sigma_win`, 3 frequencies per third 25 Hz–10 kHz, 145 angles — every 2.5° to 60°,
  every 0.25° from 60° to 90° — and 8 azimuths) for seven sizes: 1.23 × 1.48, 1.20 × 2.40, 4.37 × 2.84 (the three of
  round 1), 4.0 × 2.7 and 10 × 3 m (room walls), 1.55 × 0.95 and 2.4 × 4.8 m (Wareing's specimens). Checks
  (`chk_sig.py`), every size: against round 1's tables within 0.022 dB; σ·cos θ within 0.004 dB of one over 0–60° at
  10.8 kHz and within 0.07 dB at 75° and 80°; interpolation in log f and θ within 0.016 dB of a direct evaluation at 24
  random points; **at grazing** the azimuth-mean σ(90°) agrees with Leppington's large-plate result `H₁(x)·√(2kL)`
  [Davy_2009_ForcedRadiationEfficiency, Eq. 38 and p. 16] within 0.21 dB from 500 Hz (0.14–0.83 dB at 125 Hz, where the
  asymptote is not meant to hold) and with Davy's closed form, Eq. 37, within 0.36 dB everywhere. So the windowed model
  is checked at grazing by an independent result, not only at 0–60°.

### 0.1 In short (for Stephan)

**What a wave near grazing does to a wall.** A plane wave hitting a wall at angle θ presses on it with a pressure
pattern that runs along the wall. The pattern's wavenumber along the wall is the **trace wavenumber** `k·sin θ`; at
grazing it runs at the speed of sound. The wall answers with a velocity pattern of the same trace (the **forced wave**),
and that moving surface radiates sound on both faces. How much pressure a unit of wall velocity makes in the air is the
air's **radiation impedance**: for an infinite wall `ρ₀c/cos θ`. Near grazing this grows without bound: a velocity
pattern running along the wall at the speed of sound pushes the air ahead of it coherently over an unlimited length, so
the air resists it ever more strongly. The wall's own impedance (its mass, `jωm`) stays finite. Beyond the angle where
the air's two radiation impedances `2ρ₀c/cos θ` exceed `ωm` — `cos θ = 1/X₀`, `X₀ = ωm/(2ρ₀c)` — the air, not the mass,
sets the motion: the wall moves as the air would and passes the wave whole. So the infinite plate's `τ` goes to 1 at
90° whatever its mass. For 4 mm glass at 500 Hz that angle is 88.5°, for 152 mm concrete 89.96° (step 1).

**Why it matters although few waves arrive there.** In a diffuse field the share of power arriving past 78° is
`cos² 78° = 4.3 %`. But the mass law falls 20 dB per decade of `cos θ`, so a wave past 78° passes at
least `1/cos² 78° = 23` times what a normal one passes. For the infinite limp plate the fraction of the *transmitted*
power that comes from past 78° is `ln(1 + X₀²·cos²78°)/ln(1 + X₀²)` (my derivation, checked by the script): 0.57 for a
limp 10 kg/m² plate (4 mm glass below its coincidence) at 500 Hz, 0.78 for a limp 347 kg/m² wall at 500 Hz. **Wherever
`X₀ > 1/cos²78° ≈ 23` (a 4 mm pane from about 300 Hz, a heavy wall at every octave), more than half of what an infinite
wall transmits below its coincidence comes from the last 12° before grazing.** That is why the 0–90° average sits 0.3–7.9 dB below the 0–78° one for a limp mass law (growing with
`X₀`), and why the fitted columns (with their own angle shapes) diverge by up to 15 dB.

**What a real wall does differently.** A real wall is finite. A forced wave of finite length `L` radiates like a line
source of length `L`: its sound goes out in a beam of angular width `~λ/L` around the specular direction. At moderate
angles that beam is the infinite plate's, `σ = 1/cos θ`. Near grazing the beam would have to lie along the wall, and
half of it is cut off by the wall plane: the **radiation efficiency** σ (radiated power over `ρ₀c·S·⟨v²⟩`) stops at a
finite value, `σ(90°) ≈ 0.38·√(kL)` [Davy_2009_ForcedRadiationEfficiency, Eq. 27 with `2a = L`],
instead of growing to infinity. The air no longer overwhelms the mass, and the wall
keeps insulating. Measured (step 2, `fp.py` both models), the finite plate leaves the infinite one at
`cos θ ≈ 1.9/√(kL)` (1 dB apart) and `1.24/√(kL)` (3 dB apart), with `L = 4S/U` the panel's characteristic length (S
area, U perimeter). Past that angle its transmission loss still falls toward grazing, but at 10 dB per decade of
`cos θ` instead of 20: the transmitted power per unit of incident intensity, `τ·cos θ`, **levels off** (a plateau, after
a broad maximum under 1 dB above it), where the infinite plate's keeps rising until it is transparent. There is no
turnover in `τ` itself; at the very last fraction of a degree the finite plate's `τ` can even exceed one (diffraction:
the plate draws power from a strip wider than its own shadow), which no ray model can carry and which costs at most
0.22 dB in the diffuse average (every case of step 2, largest on gypsum at 31.5 Hz on 10 × 3 m).

**How much it matters in a room.** On a 4.0 × 2.7 m wall the finite plate's diffuse (0–90°) TL lies above the infinite
plate's by 2.9–4.1 dB for 4 mm glass or 12.5 mm gypsum in every octave below coincidence, and by 11 / 9 / 6 dB at
31.5 / 63 / 125 Hz for 152 mm concrete (above its 110 Hz coincidence the finite and infinite heavy wall agree within
0.5 dB: the power then goes through at the coincidence angle, not at grazing). The fitted light panels fake the
finite size with a level above their mass (glass 1.42 of it); with the finite size in the model they take 0.90–1.13
(step 5). The heavy walls' odd knobs (a level of 0.02–0.5 of their mass, a coincidence at 5–20 kHz) stay with every
grazing model: they answer to the region above coincidence, which grazing does not touch.

**The 78° convention** is a fixed angle standing in for a size and a frequency. The angle that makes the clamped
infinite plate match the finite plate is `cos θ_l ≈ 1.53/√(kL)` (closed form from Davy's mean radiation efficiency,
within a few tenths of a degree of the numerical match at 250 Hz–1 kHz): on a room wall 78° at 0.9 kHz, 85° at 5 kHz,
55° at 125 Hz, and no angle at all at 31.5 Hz (a small panel there insulates better than any infinite plate at normal
incidence). On a lab pane 78° falls at 2.2 kHz. Northwood's 85° and the common 78° are both right somewhere.

**Which model reproduces it.** Davy's closed-form forced radiation efficiency, put in place of `1/cos θ` in the mass
term, reproduces the finite plate within 0.4 dB in every diffuse octave and within 0.8 dB at every angle, for single and
double leaves, light and heavy, every size studied — given the panel's size. With one nominal size (4.0 × 2.7 m) for
all walls it stays within 2.4 dB per angle and 2.2 dB diffuse from 125 Hz up; at 31.5 Hz on a window it is 6 dB off. A
fixed limiting angle is off by 4.8–14 dB diffuse at 31.5–125 Hz whatever the angle (78° to 87° tried). The
recommendation and the decisions are in 0.7.

### 0.2 Step 1 — the infinite plate at grazing

**Derivation** (standard; the form of [Davy_2009_SingleLeafCremer, Eqs. 3, 4]). An incident wave
`p_i·exp(−jk(x sin θ + z cos θ))` on a plate at `z = 0`. Every field on the plate carries the trace `exp(−jk x sin θ)`;
the plate moves with velocity `v` of that trace (the forced wave), its impedance at that wavenumber
`Z_p = jωm·[1 − (f/f_c)²·sin⁴θ] + ωmη·(f/f_c)²·sin⁴θ` [Davy_2009_SingleLeafCremer, Eqs. 7, 8]. A surface velocity of
trace `k sin θ` radiates a plane wave at angle θ whose pressure is `ρ₀c/cos θ` per unit normal velocity (the normal
particle velocity of a plane wave is `p·cos θ/ρ₀c`). On the source side the blocked pressure is `2p_i`, less the
plate's own radiation; on the far side the radiated pressure is `p_t = ρ₀c·v/cos θ`. The pressure difference drives the
plate: `Z_p v = 2p_i − 2ρ₀c v/cos θ`, so `v = 2p_i/(Z_p + 2ρ₀c/cos θ)` and

    τ(θ) = |p_t/p_i|² = |1 + Z_p·cos θ/(2ρ₀c)|⁻²        [Davy_2009_SingleLeafCremer, Eq. 3]

As θ → 90° the radiation term `2ρ₀c/cos θ` dominates any finite `Z_p` and `τ → 1`. The angle where they are equal,
`|Z_p|·cos θ = 2ρ₀c`, the **transparency angle**, measured on a 0.001° sweep (`gz1.py` B; `X₀ = ωm/2ρ₀c`):

| octave | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glass 4 mm (`X₀` 2.4 … 611) | 65.4° | 78.0° | 84.0° | 87.0° | 88.5° | 89.15° | 89.32° | 89.76° | 89.98° |
| gypsum 12.5 mm | 61.1° | 76.0° | 83.0° | 86.5° | 88.2° | 88.96° | 88.79° | 89.86° | 89.99° |
| concrete 152 mm (`X₀` 83 … 21000) | 89.25° | 89.49° | 89.34° | 89.975° | 89.996° | 89.999° | 89.999° | 89.999° | 89.999° |

Below coincidence it equals `arccos(1/X₀)` within 0.1° (glass 500 Hz: 88.455° against 88.498°).

**τ(θ) per octave** (`gz1.py` A: energy mean over 64 frequencies per third, TL = −10 log τ, dB; physical parameters:
glass ρ 2500, E 70 GPa, ν 0.22, η 0.015; gypsum 8.6 kg/m², f_c 2500 Hz, η 0.006; concrete 347 kg/m², E 33 GPa, ν 0.2,
Mindlin, η = m″/(400√f); the double leaves as two such plates and an air cavity with a loss of 0.01, by transfer
matrices — round 1's sources, `a1/mat.py`, `a1/p3/heavy.out`):

| θ | glass 500 Hz | glass 2 kHz | gypsum 500 Hz | concrete 63 Hz | concrete 500 Hz | 4-16-4 1 kHz | brick 203 + furred board 2 kHz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0° | 31.3 | 43.3 | 30.0 | 44.1 | 62.1 | 67.2 | 117.8 |
| 60° | 25.2 | 34.8 | 23.8 | 36.4 | 71.3 | 46.1 | 122.0 |
| 78° | 17.5 | 24.0 | 16.2 | 27.3 | 68.1 | 20.0 | 93.8 |
| 85° | 10.3 | 15.5 | 9.1 | 19.5 | 61.1 | 20.3 | 81.9 |
| 88° | 4.2 | 8.3 | 3.4 | 11.8 | 53.3 | 13.8 | 76.8 |
| 89.5° | 0.5 | 1.6 | 0.3 | 3.0 | 41.3 | 5.2 | 67.8 |

(The double leaves' TL above 1 kHz is that of two unbridged leaves; studs and edge bridging are not modelled. Their
mass-air-mass resonance comes out at 211.3 Hz (4-16-4, as `md/acoustic.md` 3.4) and 149.2 Hz (furred lining) at normal
incidence; the cavity's stiffness grows as `1/cos²θ`, so the resonance sweeps up as `f₀/cos θ` and crosses 2 kHz at
86°: the lining's improvement vanishes for grazing waves.)

**Coincidence** (`gz1.py` C). The plate's free bending wavenumber `k_B ∝ √f` equals the trace `k sin θ` at
`sin²θ_c = f_c/f` [Davy_2009_SingleLeafCremer, Eq. 30]; there `Z_p` loses its reactance and only damping limits `τ`.
Below `f_c` there is none; at `f_c` it sits at 90° and moves toward normal as `f` rises:

| | f_c | 125 | 250 | 500 | 1k | 2k | 4k | 8k | θ_c in 70–90° / 78–90° / 85–90° for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glass 4 mm | 2989 Hz | – | – | – | – | – | 59.8° | 37.7° | 2989–3385 / –3124 / –3012 Hz |
| gypsum 12.5 | 2500 Hz | – | – | – | – | – | 52.2° | 34.0° | 2500–2831 / –2613 / –2519 Hz |
| concrete 152 | 110 Hz | 69.7° | 41.5° | 28.0° | 19.4° | 13.6° | 9.5° | 6.7° | 110–124 / –115 / –111 Hz |

Coincidence meets the grazing band only in a sliver of frequency just above `f_c`, 13 % wide for 70°. Above `f_c` the
plate is stiffness-controlled beyond `θ_c` and the grazing waves pass almost nothing until the last tenth of a degree
(concrete 500 Hz: TL 61 dB at 85°). **Below `f_c` grazing is the mass law's weak point; above `f_c` it is not a
transmission path.**

**Diffuse bookkeeping** (`gz1.py` D; weight `cos θ sin θ`, 9000 angles, 48 frequencies per octave). Incident fraction
past 70° / 78° / 85°: 0.1170 / 0.0432 / 0.0076 (= `cos²θ`). Fraction of the **transmitted** power from past 78°, and
`TL(0–78°) − TL(0–90°)`:

| construction | from past 78°, 31.5 … 8k | TL(0–78) − TL(0–90), dB |
| --- | --- | --- |
| limp 10 kg/m² | 0.12 0.22 0.35 0.48 0.57 0.64 0.69 0.73 0.76 | 0.3 0.9 1.7 2.6 3.5 4.2 4.9 5.4 5.9 |
| limp 347 kg/m² | 0.65 0.69 0.73 0.76 0.78 0.80 0.82 0.83 0.85 | 4.3 4.9 5.5 6.0 6.4 6.8 7.2 7.6 7.9 |
| glass 4 mm (physical) | 0.11 0.21 0.34 0.47 0.56 0.65 0.78 0.45 0.01 | 0.3 0.8 1.6 2.5 3.4 4.3 6.4 2.4 −0.1 |
| gypsum 12.5 (physical) | 0.10 0.18 0.31 0.44 0.55 0.64 0.62 0.05 0.00 | 0.3 0.7 1.4 2.3 3.3 4.2 4.1 0.0 −0.2 |
| concrete 152 (physical) | 0.65 0.74 0.65 0.06 0.01 0.00 0.00 0.00 0.00 | 4.3 5.6 4.3 0.1 −0.1 −0.2 −0.2 −0.2 −0.2 |
| 4-16-4 (physical) | 0.21 0.33 0.40 0.08 0.12 0.47 0.97 0.59 0.00 | 0.8 1.5 2.0 0.2 0.4 2.6 15.6 3.6 −0.2 |
| brick 203 + furred board (physical) | 0.66 0.75 0.48 0.00 0.04 0.40 0.81 0.69 0.00 | 4.5 5.8 2.7 −0.2 0.0 2.0 6.9 4.9 −0.2 |

The limp gap grows with `X₀` (0.3 dB at `X₀` 2.4 to 7.9 dB at `X₀` 21000); the manager's "3 to 5 dB" is its middle.
The double leaves' large gaps (15.6 dB at 2 kHz for 4-16-4) are the mass-air-mass resonance sweeping into the band at
grazing (above).

The 13 **fitted columns** (`md/m10_m15_report.md` 1.0, through the formula copy with the plate rule, `tf` and `att`):

| construction | from past 78°, 31.5 … 8k | TL(0–78) − TL(0–90), dB |
| --- | --- | --- |
| glass 4 mm | 0.43 0.29 0.29 0.39 0.50 0.58 0.70 0.29 0.00 | 2.3 1.3 1.3 2.0 2.8 3.6 5.1 1.3 −0.2 |
| glass 6 mm | 0.08 0.06 0.17 0.30 0.41 0.51 0.35 0.02 0.00 | 0.2 0.1 0.6 1.3 2.1 2.9 1.7 −0.1 −0.2 |
| gypsum 12.5 (model) | 0.04 0.07 0.13 0.21 0.31 0.41 0.52 0.16 0.01 | 0.0 0.1 0.4 0.8 1.4 2.1 3.0 0.6 −0.2 |
| plywood 18 (model) | 0.06 0.10 0.18 0.29 0.32 0.10 0.13 0.04 0.01 | 0.1 0.3 0.7 1.3 1.5 0.3 0.4 0.0 −0.2 |
| concrete 152 | 0.14 0.06 0.17 0.39 0.54 0.63 0.71 0.77 0.66 | 0.5 0.1 0.6 1.9 3.1 4.2 5.1 6.1 4.5 |
| brick 95 | 0.15 0.28 0.33 0.18 0.25 0.67 0.90 0.95 0.68 | 0.5 1.2 1.6 0.7 1.1 4.6 9.7 12.9 4.7 |
| brick 203 | 0.53 0.42 0.34 0.44 0.57 0.65 0.70 0.74 0.79 | 3.1 2.2 1.6 2.3 3.4 4.3 5.0 5.7 6.6 |
| block 190 | 0.28 0.43 0.53 0.53 0.57 0.70 0.82 0.87 0.95 | 1.2 2.3 3.0 3.1 3.5 5.1 7.3 8.8 13.3 |
| 4-16-4 (AGC) | 0.61 0.50 0.26 0.08 0.32 0.48 0.61 0.31 0.01 | 3.9 2.8 1.1 0.2 1.5 2.6 3.9 1.4 −0.2 |
| 4-12-4-12-4 (AGC) | 0.62 0.51 0.21 0.09 0.40 0.53 0.66 0.32 0.01 | 4.0 2.9 0.8 0.2 2.1 3.1 4.5 1.5 −0.2 |
| brick 203 + furred board | 0.06 0.05 0.01 0.00 0.00 0.00 0.00 0.00 0.00 | 0.1 0.0 −0.2 −0.2 −0.2 −0.2 −0.2 −0.2 −0.2 |
| SCR 140 + board on dabs | 0.77 0.74 0.66 0.58 0.70 0.75 0.92 0.29 0.00 | 6.2 5.6 4.5 3.6 5.1 5.9 10.5 1.3 −0.2 |
| brick 105 + plaster one face | 0.44 0.31 0.25 0.37 0.51 0.61 0.71 0.70 0.29 | 2.3 1.4 1.1 1.8 2.9 3.9 5.1 5.0 1.3 |

What the fits did with grazing: the heavy walls, fitted with a coincidence at 3.7–18 kHz (concrete, brick 203, plaster:
Q 1.1–1.9; brick 95, block: 5.6 / 6.1 kHz; report 1.0), look **below coincidence in every octave**, so grazing
carries 25–95 % (mostly over half) of their transmission at 500 Hz–4 kHz, where the physical wall (coincidence at 100–160 Hz) carries none. The furred lining, fitted with
`coiF` at the 30 Hz bound and Q 1.15, does the opposite: a stiffness term that grows as `sin⁴θ` makes it most opaque at
grazing (0–6 % from past 78°), while the physical lining passes 40–81 % from there at 1–4 kHz. Neither angle shape is
the wall's; each is what made the 0–90° average hit the reference with an infinite plate.

### 0.3 Step 2 — the finite plate

**Models** (`fp.py`, copied). (i) *Windowed forced wave* [Villot et al., JSV 245 (2001)]: the infinite plate's forced
velocity, cut to the `a × b` opening in a rigid baffle, radiated by the Rayleigh integral; the radiation efficiency of
that windowed wave, σ(f, θ, φ), replaces `1/cos θ` on both faces:
`τ(θ, φ) = 4ρ₀²c²·σ/(cos θ·|Z_p + 2ρ₀c·σ|²)`; averaged over azimuth. For the double leaves the same σ loads both outer
faces of the transfer-matrix chain (`gzlib.den_of`; it reduces to `fp`'s form for one plate, checked to the printed
digit). (ii) *Modal sum* of the simply supported plate under the plane-wave pressure, every mode, every cross term in
the radiated power, each mode's own radiation loss, thin modes; single plates only. Per angle: `gz2.py` (windowed, 32
frequencies per third, angles every 0.02°), `gz2m.py` (modal, 16 frequencies per third, 14 angles).

**Where the finite plate parts from the infinite one** (`summ2.py` (1), every single plate, size and octave below
`f_c/2`): the first angle with `|TL_fin − TL_inf| ≥ 1 dB` lies at `cos θ = C/√(kL)`, `L = 4S/U`, with C = 1.86–1.93
(mean 1.90, 28 cases with `kL ≥ 8`); for 3 dB, C = 1.14–1.27 (mean 1.24). Below `kL ≈ 4` the finite plate differs at
every angle (glass 1.23 × 1.48 m at 31.5 Hz: +9.8 dB at normal incidence). Examples, departure at 1 / 3 dB:

| | kL | 1 dB | 3 dB | Davy's range of `1/cos`, `arccos(1.3·√(π/2ka))` |
| --- | --- | --- | --- | --- |
| glass, lab pane, 500 Hz | 12.3 | 57.6° | 68.7° | 48.9° |
| glass, 4.0 × 2.7 m, 500 Hz | 29.5 | 69.5° | 76.6° | 64.9° |
| concrete, 4.37 × 2.84 m, 1 kHz | 63.1 | 75.6° | 80.5° | 73.1° |
| concrete, 4.37 × 2.84 m, 8 kHz | 504 | 85.1° | 86.8° | 84.1° |

**How it parts.** Per angle (`gz2_*.out`; `summ2.py` (2), (3)): between 80° and 88° the infinite plate's TL falls
16.9–19.8 dB per decade of `cos θ` (20 = `τ ∝ 1/cos²θ`), the finite plate's 8.6–11.6 dB (10 = `τ ∝ 1/cos θ`), for glass
and concrete at every size and at 63, 500 and 2000 Hz. In `τ·cos θ` — the transmitted power per unit incident
intensity, what the diffuse average weighs — the finite plate is flat within 1.1 dB from 80° to 89.5° in every one of
those cases (glass 4.0 × 2.7 m, 500 Hz: −28.0 / −28.3 / −28.3 dB at 80 / 88 / 89.5°; infinite −23.6 / −18.8 / −21.1).
`fig2_glass_4_mm.png` shows it: the finite curves rise with the infinite one, peak 0.6–0.8 dB above their grazing
value at 55–75°, and level off; Davy's form puts that maximum at `g = 1.3·g_l` [Davy_2009_ForcedRadiationEfficiency, p. 18]. So:
**not a plateau of TL and not a turnover of τ, but a plateau (after a broad maximum) of the transmitted power per
incident intensity**. TL itself keeps falling at half the infinite rate and only reaches 0 dB within a fraction of a
degree of grazing; `fig1_glass_4_mm.png`, `fig1_concrete_152.png`, `fig2_concrete_152.png` show the same at the three
sizes.

**Below and above `f_c`.** Below, the difference grows toward low `kL` and the diffuse finite TL lies above the
infinite 0–90° one by (`summ2.py` (4), dB, 31.5 … 2k):

| | lab size | 4.0 × 2.7 m | 10 × 3 m |
| --- | --- | --- | --- |
| glass 4 mm | +10.3 +7.5 +5.6 +4.8 +4.4 +4.2 +4.2 (1.23 × 1.48) | +4.1 +3.5 +3.3 +3.2 +3.1 +3.2 +2.9 | +2.5 +2.5 +2.6 +2.6 +2.7 +2.8 +2.5 |
| gypsum 12.5 | +8.0 +5.9 +4.7 +4.2 +3.9 +3.9 +0.4 (1.2 × 2.4) | +3.5 +3.1 +3.0 +3.0 +3.0 +3.0 +0.1 | +2.0 +2.1 +2.3 +2.4 +2.5 +2.6 +0.1 |
| concrete 152 | +10.6 +9.1 +6.2 +0.4 +0.1 0.0 0.0 (4.37 × 2.84) | +11.0 +9.3 +6.4 +0.5 +0.1 0.0 0.0 | +9.2 +8.2 +5.5 +0.3 0.0 0.0 0.0 |
| 4-16-4 | +12.6 +8.9 +5.8 +0.7 +1.6 +4.0 +5.0 (1.23 × 1.48) | +6.3 +4.8 +3.7 +0.4 +0.6 +2.2 +3.1 | +4.6 +3.8 +3.1 +0.4 +0.5 +1.7 +2.5 |
| brick 203 + furred board | +10.7 +9.1 +4.2 0.0 +0.3 +1.4 +3.1 (4.37 × 2.84) | +11.1 +9.3 +4.3 +0.1 +0.3 +1.5 +3.2 | +9.3 +8.3 +3.6 0.0 +0.2 +1.3 +2.7 |

Above `f_c` the departure law holds the same (concrete above, 1–8 kHz), but it no longer matters: the power goes
through at `θ_c`, inside the range where `σ = 1/cos θ`, and the finite and infinite heavy wall agree within 0.1 dB from
500 Hz. Just above `f_c`, where `θ_c` lies beyond the departure angle, the finite size cuts the coincidence itself:
concrete at 125 Hz (`θ_c` 69.7°), finite 34.8 dB against infinite 28.5 dB on 4.37 × 2.84 m.

**The mechanism in words.** A forced wave of length L is a line source with a linear phase. Its far field is a beam of
width `~λ/L` around the specular angle; σ is the beam's power, and as long as the beam fits inside the half space the
result is the infinite plate's `1/cos θ` [Davy_2009_ForcedRadiationEfficiency, Eqs. 10–11, 16]. Near grazing the beam
lies along the wall plane and only the half above it radiates: σ saturates at `(2/3)·√(2ka/π) ≈ 0.38·√(kL)`
(`2a = L`) [ibid., Eq. 27], the transparency never comes, and the plate's mass keeps acting. At low `kL` the whole
plate moves in phase and radiates as a small piston, `σ ≈ k²S/2π` [ibid., Eq. 35], less than 1 at every angle: the finite plate then insulates better than
the infinite one even at normal incidence. The edges matter in two ways the windowed model ignores and the modal model
carries: the free bending waves they reflect, which below `f_c` radiate poorly [ibid., p. 3], and the plate's modes.

**What the modal model adds** (`gz2m.py`, modal against windowed at the same frequencies, azimuth mean):

| | 31.5 | 63 | 125 | 250 |
| --- | --- | --- | --- | --- |
| glass, lab pane: modal − windowed, 0° / 78° / 89.5° | +3.7 / +3.4 / +3.4 | −2.7 / −2.7 / −2.7 | +0.6 / +0.1 / +0.1 | −0.2 / −0.3 / −0.3 |
| gypsum 1.2 × 2.4 | −2.3 / −2.3 / −2.3 | −1.5 / −1.5 / −1.5 | +0.1 / −1.1 / −1.1 | −0.9 / −0.5 / −0.5 |
| concrete 4.37 × 2.84 (thin modes) | −0.8 / +0.2 / +0.2 | −10.2 / −7.0 / −6.5 | −4.9 / +4.4 / +4.6 | |
| concrete 4.0 × 2.7 | +5.1 / +5.9 / +5.9 | −10.5 / −6.6 / −6.1 | −2.3 / +4.7 / +4.8 | |
| glass 4.0 × 2.7 | +0.5 / +0.6 / +0.6 | +0.3 / +0.3 / +0.3 | +0.1 / 0.0 / 0.0 | |

The modes move the **level** of an octave (resonant transmission near the first modes and near `f_c`: up to 10 dB on
concrete at 63 Hz), almost the same at every angle past 60°: from 70° to 89.5° the modal-minus-windowed difference
varies by at most 1.2 dB (concrete at 63 Hz) and by at most 0.6 dB in every other case above. **The grazing shape is
the windowed form's; the modes are a separate, nearly angle-independent low-frequency correction.**

### 0.4 Step 3 — what the measurement is

- **What R assumes.** The laboratory sound reduction index takes the source room as diffuse: the intensity on the
  specimen is `⟨p²⟩/(4ρ₀c)`, the incident power `W₁ = I₁S`, the transmitted power `W₂ = ⟨p₂²⟩A/(4ρ₀c)`, so
  `R = L₁ − L₂ + 10 log(S/A)` [Prato_2015_LowFreqModalApproach, Eqs. 2.41–2.46, pp. 30–31]. The incident field enters
  only through `⟨p²⟩`; R is therefore the transmitted power over the power a perfectly diffuse field of that level would
  bring, whatever angles actually arrive.
- **What a laboratory delivers.** Rooms carry diffusing panels "to provide as good an approximation as possible to
  the diffuse sound field" [Northwood_1968_PlasterboardWalls, p. 2 (PDF p. 4)]; the angular distribution is still not
  uniform — beam-traced chambers give an averaged distribution close to a Gaussian, and empirical limiting angles of 70°
  to 85° are used to match theory [Jeong_2009_AngularDistributionCorrection, abstract and p. 2064]; "usually 70…85°" for
  reverberation chambers [Hongisto_2006_DoubleWallsOverview, p. 3 of the PDF]. ISO 10140-4 asks 10–20 m² for walls
  [Wareing_2015_SampleSizeVariations, PDF p. 5]; a specimen sits in a niche, which shields it from grazing waves: the
  largest angle follows from the opening's dimensions and the niche depth [Eq. 1], 83° / 77° (horizontal / vertical) for 2.4 × 4.8 m in a
  550 mm niche, 77° / 69° for 1.55 × 0.95 m in 350 mm [ibid., PDF pp. 17–18, Eq. 1, Table 2]; a "baffle effect" and a
  "niche effect" explain why small windows gain less than 6 dB per doubling of mass below `f_c`
  [Quirt_1982_WindowsI, p. 839].
- **Is a measured R the 0–90° average of a finite plate?** Approximately, by construction of R (above) — with two
  departures that both raise R: the niche cuts the grazing part (on the 2.4 × 4.8 m specimen the cut at 83° / 77° adds
  0.5–1.2 / 0.9–2.6 dB below `f_c` for plywood, `gz4.py`), and the real field carries less than the diffuse share near
  grazing. Both act where the finite plate already holds the grazing waves down; the finite plate in a baffle is the
  model of the measurement, and a residual of 1–3 dB from field shape and niche is inside the 3–7 dB that nominally
  identical walls differ by (`md/acoustic.md` 9). (Reasoning, mine.)
- **What the library's room delivers.** Rays at every angle. In a room of specular walls a ray keeps its direction
  cosines; it hits a wall normal to x at the rate `|u_x|·c/L_x`, so with an omnidirectional source (`|u_x|` uniform)
  the hits per angle are weighted by `cos θ sin θ` — the diffuse weight. (My derivation; it does not cover scattering,
  which is out of scope, nor a source near the wall.) So a library wall sees the full grazing share, 4.3 % past 78°,
  and no niche. A building's separating wall is bounded by the flanking walls, not set in a baffle: the finite plate in
  a baffle is the nearest tractable model of it, not an exact one. (Reasoning.)

### 0.5 Step 4 — the literature, checked

- **Davy's forced radiation efficiency** [Davy_2009_ForcedRadiationEfficiency]. What it is: the radiation efficiency of
  the forced wave of a finite panel, from a 2-D strip model, azimuth averaged, fitted to Sato's 3-D numerics. Closed
  form: `σ = 1/√(g² + q²)` for `g = cos θ ≥ p`, `σ = 1/√((h − αg)² + q²)` below [Eq. 37, n = 2], with
  `p = min(1, w·√(π/2ka))` [Eq. 19], `h = 1/((2/3)·√(2ka/π) − β)` [Eq. 28], `α = h/p − 1` [Eq. 34],
  `q = 2π/(k²S)` [Eq. 36], constants n = 2, w = 1.3, β = 0.124 [p. 20]; the diffuse mean
  `⟨σ⟩ = ∫σ sin θ dθ` in closed form [Eq. 43] (the manuscript's page numbers). The size it needs: `2a = 4S/U` for
  `ka` [p. 16] and the area `S` for `q` [pp. 13–14] — a rectangle's area and perimeter, or for a square one length.
  Against Sato it is within −0.7 / +0.3 dB per angle [Table I] and −0.26 / +0.18 dB diffuse [Table IV]. **Against `fp.py`**
  (`gz5.py` candidate d, `summ5.py` (1)): σ(90°) within 0.36 dB of the windowed σ (0.0); `τ` with Davy's σ in place of
  the windowed one, over five constructions × five sizes × nine octaves: diffuse within 0.40 dB, per angle within
  0.79 dB (0–60°), 0.43 dB (60–78°), 0.33 dB (78–85°), 0.41 dB (85–89.9°). With the windowed σ's azimuth mean in Davy's
  place (`dx`) the error is below 0.07 dB everywhere: the azimuth average of σ before τ costs nothing; the closed form
  costs the 0.4 / 0.8 dB. Davy's single-leaf paper uses exactly this in the mass law, `τ_d = 2⟨σ⟩/a²`, "without
  recourse to a limiting angle" [Davy_2009_SingleLeafCremer, Eqs. 42–44, p. 1871]; with that theory Kosala found about
  3 dB spectral RMS error against 6–7 dB for the mass law [Kosala_2020_SingleBafflesAccuracy, Table 2]
  (`md/acoustic.md` 3.4).
- **Northwood's 78° / 85°** [Northwood_1968_PlasterboardWalls, p. 2 (PDF p. 4)]: grazing incidence is "an unrealistic
  limit for finite rooms and walls"; "some writers", finding measured TL about 5 dB above the random-incidence value,
  inferred a limiting angle of about 78°, "the so-called field incidence case"; his calculations and NRC measurements
  in a 10 × 8 ft opening fit best at 85°, with a spread of about 5 dB between laboratories "depending on the exact
  angular distribution". For that opening (4S/U 2.71 m) the closed form puts 78° at 1.1 kHz and 85° at 6.1 kHz
  (`gz_eqang.out`); plasterboard's `f_c` near 2.5 kHz puts its mass-law region at and below the 78° frequency, so a
  single fitted angle there is a compromise over frequency, and 85° is plausible for a fit weighted by the mid bands of
  several walls including coincidence. (Reasoning.)
- **The origin of 78°.** The review gives no primary source for it. London does not use it: he integrates to 90° and
  reduces grazing transmission with an empirical resistance in the wall impedance, `Z_w = 2r/cos θ + jωm`, so that
  `τ(90°) = 1/(1 + R)²`, R = 2.16 fitted to a thin aluminium wall of 0.12 g/cm²
  [London_1949_SingleWalls, p. 608, Eqs. 3.4–3.6]; he motivates the `1/cos θ` resistance by the long path of an oblique wave. Northwood attributes 78° to unnamed writers who matched a
  5 dB offset (above); Hongisto and Jeong give 70–85° as empirical [Hongisto_2006_DoubleWallsOverview;
  Jeong_2009_AngularDistributionCorrection]. In the light of step 2: a 5 dB offset over random incidence is what a
  finite wall of a few metres gives at a few hundred hertz (glass 4.37 × 2.84 m: finite − infinite(0–90°) 3.1 dB at
  500 Hz; lab pane 4.4 dB), and the limit angle that reproduces a finite plate falls at 78° near 1 kHz on such a wall.
  The convention is a size and a frequency frozen into an angle.
- **Wareing's size effect** (`gz4.py`). Reported: the 1.55 × 0.95 m specimen above the 2.4 × 4.8 m one, "generally
  greater than 5 dB" below `f_c`, for plywood, 10 mm plasterboard and a 2 kg/m² limp barrier
  [Wareing_2015_SampleSizeVariations, PDF p. 10]. The finite plate at the two sizes, small − large (dB, octaves 63 …
  1k; plywood with round 1's constants, not Wareing's):

  | | 63 | 125 | 250 | 500 | 1k |
  | --- | --- | --- | --- | --- | --- |
  | limp 2 kg/m², windowed / Davy | +3.1 / +2.9 | +1.9 / +1.8 | +1.6 / +1.7 | +1.3 / +1.4 | +1.1 / +1.2 |
  | plywood 12 mm (f_c 1719 Hz), windowed / Davy | +4.7 / +4.5 | +2.8 / +2.4 | +1.9 / +2.1 | +1.5 / +1.6 | +1.5 / +1.5 |
  | plywood 21 mm (f_c 982 Hz), windowed / Davy | +4.8 / +4.6 | +2.8 / +2.5 | +2.0 / +2.1 | +1.7 / +1.8 | +0.6 / +0.6 |
  | plywood 12 mm with the niche cuts (77 / 83° and 69 / 77°) | +5.3 / +5.6 | +3.2 / +3.5 | +2.3 / +2.5 | +1.9 / +2.1 | +2.2 / +2.4 |

  The finite plate gives 1.1–2.8 dB in Wareing's range (the octaves from 125 Hz), 1.9–3.5 dB with the niche cut; Davy's closed form
  reproduces the windowed size effect within 0.4 dB. The rest of the "more than 5 dB" lies outside a plate in a baffle:
  Wareing names edge conditions (the small specimen clamped), the edge ratio and the receiving rooms (9 m³ against
  200 m³) [ibid., PDF pp. 7, 15–16]. **The size effect is real and of the predicted sign; the finite plate explains about
  half of its measured size.**

### 0.6 Step 5 — model candidates

The candidates act on the mass term's **radiation** cosine only — the `c` in `log10(1 + (X₀c)²)` — and leave the
plate's own impedance (bending at the true `sin θ`, the cavity spring at the true `cos θ`) alone, unless said
otherwise. Written as the library would: `τ_L(σ) = |1 + Z_p/(2ρ₀c·σ)|⁻²` (the mass cosine replaced by `1/σ`) and
`τ_D(σ) = τ_L(σ)/(σ cos θ)` (with the gain of a finite plate, clipped at one).

| | what | `τ` |
| --- | --- | --- |
| (a) | none, the infinite plate (today) | `τ_L(1/cos θ)` |
| (b) | fixed limit θ_l (M14), radiation cosine only; **B78**: every cosine clamped, as `txlib`'s `m14` and the M14 of `md/m10_m15_report.md` | `τ_L(1/max(cos θ, cos θ_l))` |
| (c) | limit from the size and the frequency: `cos θ_l = min(1, exp(½ − ⟨σ_D⟩))`, ⟨σ_D⟩ Davy's mean [Eq. 43] (my derivation: for a mass-law plate the clamp then gives Davy's diffuse `2⟨σ⟩/X₀²`) | as (b) |
| (d) | Davy's σ in place of `1/cos θ`, with the gain | `τ_D(σ_D(k, θ, S, U))` |
| (e) | a soft limit: `cos θ → √(cos²θ + c₀²)`, `c₀ = exp(−⟨σ_D⟩)` (my construction: smooth, the same diffuse value for a mass-law plate) | `τ_L(1/√(cos²θ + c₀²))` |

(c), (d), (e) need a size: the plate's own (`c`, `d`, `e`) or one nominal 4.0 × 2.7 m wall (`cn`, `dn`, `en`).

**(i) Per angle, against the finite plate** (`gz5.py`, `summ5.py` (1): the physical constructions — glass 4 mm and
gypsum 12.5; concrete 152; 4-16-4 and the furred lining — at 1.23 × 1.48, 1.2 × 2.4, 4.37 × 2.84, 4.0 × 2.7 and
10 × 3 m; reference the windowed finite plate clipped at one per frequency; largest |error| in dB over every size of the
class):

| | diffuse, per octave 31.5 … 8k | per angle 0–60° / 60–78° / 78–85° / 85–89.9° | per angle, 125 Hz up |
| --- | --- | --- | --- |
| **light single** | | | |
| (a) none | 10.4 7.5 5.6 4.8 4.4 4.2 4.2 0.6 0.0 | 11.3 / 11.3 / 10.3 / 19.8 | 5.2 / 8.4 / 10.0 / 19.8 |
| (b) 78° | 10.3 7.3 4.8 3.3 2.0 1.0 1.9 1.0 0.0 | 11.3 / 11.3 / 10.3 / 22.4 | 5.2 / 8.4 / 8.4 / 22.4 |
| (b) 85° | 10.4 7.5 5.5 4.4 3.4 2.7 2.2 0.3 0.0 | 11.3 / 11.3 / 10.3 / 14.9 | 5.2 / 8.4 / 10.0 / 14.9 |
| (c) own size | 7.0 2.5 0.3 0.3 0.1 0.1 0.6 0.4 0.0 | 9.8 / 7.0 / 6.9 / 21.4 | 1.5 / 3.1 / 6.8 / 21.4 |
| (d) own size | 0.4 0.3 0.2 0.3 0.1 0.1 0.0 0.0 0.0 | 0.6 / 0.4 / 0.3 / 0.3 | 0.5 / 0.4 / 0.3 / 0.3 |
| (e) own size | 6.1 2.5 0.8 0.1 0.0 0.0 0.5 0.4 0.0 | 7.6 / 6.7 / 4.1 / 17.1 | 1.9 / 1.9 / 2.7 / 17.1 |
| (d) nominal 4.0 × 2.7 | 6.3 3.9 2.1 1.5 1.2 1.1 1.3 0.4 0.0 | 7.0 / 6.4 / 5.9 / 5.7 | 2.3 / 2.4 / 2.3 / 2.3 |
| **heavy single** | | | |
| (a) none | 17.3 13.1 8.8 1.8 0.1 0.0 0.0 0.0 0.0 | 13.8 / 17.6 / 21.2 / 26.8 | 4.5 / 8.9 / 11.9 / 26.8 |
| (b) 78° | 14.0 8.9 6.0 1.5 0.0 0.0 0.0 0.0 0.0 | 13.8 / 17.6 / 17.6 / 22.7 | 4.5 / 8.9 / 8.9 / 22.7 |
| (c) own size | 7.6 1.6 1.6 0.8 0.0 0.0 0.0 0.0 0.0 | 10.7 / 7.8 / 6.9 / 23.0 | 1.6 / 3.2 / 6.9 / 23.0 |
| (d) own size | 0.4 0.2 0.2 0.4 0.0 0.0 0.0 0.0 0.0 | 0.8 / 0.4 / 0.3 / 0.4 | 0.5 / 0.4 / 0.3 / 0.4 |
| (e) own size | 6.6 2.1 0.5 0.5 0.5 0.2 0.1 0.0 0.0 | 8.1 / 7.4 / 4.5 / 18.8 | 1.6 / 1.8 / 2.8 / 18.8 |
| (d) nominal 4.0 × 2.7 | 6.4 3.8 2.2 1.2 0.0 0.0 0.0 0.0 0.0 | 7.2 / 6.4 / 5.9 / 5.7 | 2.2 / 2.3 / 2.3 / 2.2 |
| **double** | | | |
| (a) none | 17.4 13.1 6.4 0.8 1.6 4.0 5.0 4.0 0.0 | 13.8 / 17.6 / 21.2 / 27.0 | 5.3 / 9.3 / 11.9 / 25.0 |
| (b) 78° | 13.9 8.7 4.8 0.8 1.2 2.5 2.7 1.5 0.0 | 13.8 / 17.6 / 17.6 / 22.3 | 5.3 / 9.3 / 9.3 / 22.3 |
| B78 (as built) | 13.9 8.6 4.8 0.8 1.2 3.6 6.4 1.8 0.0 | 13.8 / 17.6 / 17.6 / 33.6 | 5.3 / 9.3 / 16.3 / 33.6 |
| (c) own size | 7.6 2.7 1.0 0.5 0.6 0.7 0.4 0.4 0.0 | 10.6 / 7.7 / 7.2 / 22.8 | 1.5 / 3.4 / 7.2 / 22.8 |
| (d) own size | 0.4 0.3 0.2 0.2 0.2 0.2 0.1 0.1 0.0 | 0.8 / 0.4 / 0.3 / 0.4 | 0.5 / 0.4 / 0.3 / 0.4 |
| (e) own size | 6.6 2.7 0.7 0.4 0.5 0.7 0.6 0.5 0.0 | 8.0 / 7.4 / 4.5 / 18.5 | 1.9 / 2.0 / 3.2 / 18.5 |
| (d) nominal 4.0 × 2.7 | 6.4 4.0 1.9 0.7 1.0 1.8 1.9 0.6 0.0 | 7.2 / 6.4 / 6.0 / 5.8 | 2.3 / 2.4 / 2.4 / 2.3 |

(80° and 87° behave as 78° and 85°: diffuse errors between them; full rows in `summ5.out`.) Read: **only (d) follows the
finite plate per angle**; every form that keeps the infinite plate's shape — a clamp or a soft limit — is 4–34 dB off
past 78°, because the finite plate's TL keeps falling there at 10 dB per decade of `cos θ` and theirs stops. (c) and (e)
get the diffuse average right from 125 Hz (within 1.6 dB) but not at 31.5 / 63 Hz (1.6–7.6 dB): a cosine cannot exceed one,
and a small plate at low frequency insulates beyond normal incidence. A fixed angle (b) is right in one octave per size
(fig3, 0.5) and off by up to 14 dB in the low octaves. With a nominal size, (d) is within 2.4 dB at every angle and
2.2 dB diffuse from 125 Hz for sizes from a 1.23 × 1.48 m pane to a 10 × 3 m wall, and 6.4 dB at 31.5 Hz (the pane).

**(ii) On the 13 constructions at 0–90°** (`gfit.py`, the procedure of `md/m10_m15_report.md` 0.2 with 8 seeds plus
the M10 0–90° fit as a start, selection at 768 × 12; worst residual per construction, dB):

| variant | g4 | g6 | gyp | ply | c152 | b95 | b203 | blk | 4164 | trip | furr | dabs | plas | worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) today (M10 at 0–90°; report 1.0: 2.97) | 0.95 | 2.23 | 2.97 | 0.58 | 0.59 | 0.17 | 0.57 | 0.90 | 1.55 | 2.68 | 2.67 | 1.73 | 0.03 | **2.97** |
| B78, M14 as built (report 0.2: 2.80) | 0.61 | 2.00 | 2.80 | 0.47 | 0.48 | 0.26 | 0.59 | 0.50 | 1.15 | 1.45 | 0.68 | 0.44 | 0.00 | **2.80** |
| (b) 78°, radiation cosine only | 0.55 | 2.01 | 2.82 | 0.49 | 0.56 | 0.22 | 0.59 | 0.36 | 1.15 | 2.26 | 2.67 | 1.41 | 0.00 | **2.82** |
| (b) 85°, radiation cosine only | 0.72 | 2.31 | 2.95 | 0.56 | 0.50 | 0.32 | 0.59 | 0.37 | 1.40 | 2.48 | 2.67 | 1.44 | 0.00 | **2.95** |
| (c) nominal 4.0 × 2.7 | 0.83 | 1.95 | 2.69 | 0.40 | 0.56 | 0.07 | 0.63 | 0.55 | 1.46 | 2.57 | 2.75 | 2.16 | 0.00 | **2.75** |
| (c) lab size | 0.79 | 1.72 | 2.54 | 0.30 | 0.56 | 0.08 | 0.66 | 0.52 | 1.53 | 2.64 | 2.77 | 1.91 | 0.00 | **2.77** |
| (d) nominal 4.0 × 2.7 | 0.93 | 1.95 | 2.68 | 0.38 | 0.57 | 0.04 | 0.58 | 0.52 | 1.48 | 2.59 | 2.69 | 2.11 | 0.00 | **2.69** |
| (d) lab size | 0.86 | 1.78 | 2.53 | 0.33 | 0.57 | 0.04 | 0.58 | 0.51 | 1.56 | 2.67 | 2.74 | 2.25 | 0.00 | **2.74** |
| (d) nominal 1.23 × 1.48 (a pane for all) | 0.86 | 1.78 | 2.49 | 0.36 | 0.60 | 0.12 | 0.58 | 0.54 | 1.56 | 2.67 | 3.12 | 1.95 | 0.00 | **3.12** |
| (e) nominal 4.0 × 2.7 | 0.86 | 1.99 | 2.71 | 0.47 | 0.55 | 0.15 | 0.65 | 0.54 | 1.48 | 2.59 | 2.68 | 2.59 | 0.00 | **2.71** |
| control: (a), `coiF` at the physical coincidence | 0.95 | 2.51 | 3.53 | 2.64 | 1.52 | 1.45 | 1.33 | 0.90 | 1.64 | 2.69 | 2.89 | 1.73 | 0.03 | **3.53** |
| control: (d) lab size, `coiF` physical | 0.63 | 1.81 | 3.03 | 1.99 | 2.01 | 1.45 | 1.39 | 0.51 | 1.58 | 2.67 | 2.99 | 2.25 | 0.00 | **3.03** |
| side: (a) + cavity spring locally reacting | 0.86 | 2.36 | 2.97 | 2.75 | 0.59 | 0.36 | 0.57 | 0.77 | 0.53 | 1.82 | 1.70 | 0.48 | 0.03 | **2.97** |
| side: (d) lab size + cavity spring locally reacting | 0.66 | 1.73 | 2.53 | 1.20 | 0.48 | 0.33 | 0.68 | 0.81 | 0.04 | 2.12 | 1.12 | 0.16 | 0.00 | **2.53** |

The knobs (`sumfits.out`; level = fitted `m″` over the construction's mass; heavy = the seven walls and linings):

| variant | level/mass, light | level/mass, heavy | `att` at the 12 dB bound (of 7) | `coiF` Hz: c152 / b95 / b203 / furred (physical 111 / 160 / 100 / 100) |
| --- | --- | --- | --- | --- |
| (a) today | 0.56 … 1.42 (glass 4 / 6: 1.42 / 1.42) | 0.02 … 0.72 | 4 | 6315 / 5441 / 19034 / 30 (Q 1.1 / 4.3 / 1.1) |
| B78 as built | 0.56 … 1.09 | 0.10 … 1.03 | 1 | 6891 / 14069 / 9433 / 6485 |
| (b) 78° | 0.50 … 1.10 | 0.02 … 0.33 | 2 | 6309 / 15445 / 9526 / 30 |
| (d) nominal | 0.43 … 1.27 (0.93 / 0.95) | 0.02 … 0.39 | 2 | 6679 / 5718 / 20000 / 30 |
| (d) lab size | 0.36 … 1.13 (1.13 / 0.90; plywood 0.84, gypsum 0.63) | 0.02 … 0.53 | 1 | 6859 / 5732 / 20000 / 30 |
| control (d) lab, `coiF` physical | 0.36 … 1.15 | 0.03 … 0.53 (bricks 0.07 / 0.06 with `att` 9.3 / 11.9) | 2 | pinned |
| side (d) lab + local cavity | 0.35 … 0.96 | 0.08 … 0.53 (furred 0.23) | 2 | 7430 / 5461 / 5848 / 8458 |

What the fits say:

- **The worst residual hardly separates the candidates**: 2.69–2.97 dB for every sound variant, decided by the gypsum
  board (a model reference, the spread in every octave) and the furred lining. Davy's form is best (2.69 dB nominal,
  2.74 dB at lab size), a fixed 85° no better than none (2.95). The fit is not the instrument that picks the model;
  step 2 is.
- **Light panels become physical** under (d): glass 4 / 6 mm at 1.13 / 0.90 of their mass at lab size, 0.93 / 0.95 at
  the nominal size, against 1.42 / 1.42 today. Without the finite size their level had to sit above the mass to buy
  back the infinite plate's grazing transmission.
- **The heavy walls' knobs do not come off with any grazing candidate** (level 0.02–0.53 of the mass, `coiF` at 5–20
  kHz with Q 1–1.6, one to four `att` at 12 dB). With the coincidence pinned physical the bricks need 0.06–0.07 of their
  mass and 9–12 dB of `att`, with or without (d). Physically the heavy walls are above coincidence from 125 Hz, where
  grazing carries nothing (step 1); what the knobs stand in for is the region above coincidence — round 1 found the
  physical finite Mindlin plate with coupling losses up to 10.5 dB off the measured concrete thirds there
  (`a1/p3/heavy.out`, FIN-eM: −10.5 … +7.8 dB). So **the suspicion in the brief is half right**: the finite wall is
  what the light panels' level was doing, and it is what the heavy walls' low octaves need (step 2: +6 to +11 dB at
  31.5–125 Hz), but the heavy walls' extreme knobs above 125 Hz are not grazing.
- **M14 as built helps the linings for a different reason.** Clamping every cosine also freezes the cavity spring of
  the mass-air-mass term (`x = f·c/resF`) at 78°: the furred lining goes from 2.67 to 0.68 dB and the lining on dabs
  from 1.73 to 0.44 dB. Making only the cavity locally reacting (`x = f/resF`, a switch in `txg.py`) does the same with
  Davy's form: 4-16-4 0.04, dabs 0.16, furred 1.12 dB (its `coiF` leaves the 30 Hz bound). A cavity divided by battens,
  studs or a pane's spacer is locally reacting across the partitions, and a honeycomb partition removes the angle shift
  of a panel absorber [Toyoda_2011_HoneycombPanelAbsorbers, Fig. 2, via review 2.2]. This is the `res*` cosine of the
  plate's resonance (`md/acoustic.md` 2.1, N3), not grazing radiation; it belongs to the double-leaf item, not to M14
  (0.7). Plywood worsens under it (0.33 → 1.20 dB) because its fit uses `resS > 0` at 713 Hz as a shape, not as a
  cavity.

**(iii) By code** (`src/ray_mesh_interact.cpp`; every mass term goes through `Material::mass_TL`, called at lines 327,
1037 and 4319; every candidate needs only the frequency and the cosine the wave has in air at the event,
plus constants or one column; no per-ray state; the mapping keeps wavelengths, so `k = 2π·f_GHz/0.2998` m⁻¹ needs no
acoustic constant in the library):

| | needs at the event | lines (est.) | column | `[wave]` |
| --- | --- | --- | --- | --- |
| (a) | — | 0 | no | untouched |
| (b) | `c`, one constant | +2 (`mass_c = max(mass_c, cos θ_l)` in `mass_TL`) | no | untouched (`m > 0` only) |
| (c) | `f`, `c`, a size (2 constants or a column) | +14 (⟨σ_D⟩ closed form, `fGHz` into `mass_TL`, three call sites changed) | optional | untouched |
| (d) | `f`, `c`, a size | +12 (σ_D in 7 lines; `mass_TL` takes `fGHz`, uses `1/σ` and subtracts `10 log10(1/(σ c))`, clipped at 0) | optional | untouched |
| (e) | `f`, `c`, a size | +14 | optional | untouched |

A size column touches, besides the source, `src/obj_file_write.cpp`, `api_python/qpy_obj_file_read.cpp`,
`api_mex/obj_file_read.cpp`, the three `test_obj_file_write` files and `tools/data_formats_md/format_materials.md`
(the files that carry `tfQ` today). Davy's form needs the area and the perimeter; one column could be the
characteristic length `L = 4S/U` with a square assumed for `q` (`S = L²`), which Davy's `√S` length convention for
`q` makes exact for a square [Davy_2009_ForcedRadiationEfficiency, p. 14]. Under (d) `mass_TL` loses the shortcut "at
normal incidence the term is `A`": a small plate at low frequency has `σ < 1` and passes less than `A` says. The bonded
sum `TL(X_run + X₀) − TL(X_run)` keeps working: the gain is paid once, on the total.


### 0.7 Recommendation and decisions

**Recommendation: (d), Davy's forced radiation efficiency in the mass term, with one nominal size and no column.**

- *Physics.* It is the only candidate that follows the finite plate per angle (within 0.8 dB, five constructions,
  five sizes, every octave) and on the diffuse average (0.4 dB); it carries the low-frequency excess of a small panel
  that no limiting angle can (step 5 (i)); it is the published form of the finite plate below `f_c`
  [Davy_2009_SingleLeafCremer, Eqs. 42–44] and was checked here against two independent finite-plate models and
  Leppington's grazing result (0.0, 0.3, 0.5). It explains the 78° / 85° conventions as special cases (0.5) instead of
  choosing between them.
- *Fit.* 2.97 → 2.69 dB on the 13 at 0–90° (nominal size); the light panels' levels become physical (glass 0.93 /
  0.95 of the mass).
- *Code.* About +12 lines in `src/ray_mesh_interact.cpp` (σ_D in seven lines; `mass_TL` takes `fGHz`, three call sites
  change); the frequency and the air cosine are at every event; no per-ray state; `m > 0` only, so `[wave]` is
  untouched; no column with a nominal size. Moves the mass-law rows near grazing and those at 31.5 / 63 Hz
  (the `q` term acts at normal incidence too); candidates are *Thin sheet - mass-law angle dependence* and *Mass law - m switches the limp
  mass law on* (not run here; phase 1 measures it).
- *Closes M14.* The limiting angle is replaced by the size-dependent radiation efficiency it approximates; the
  question "why a limit, which angle, does it depend on size and frequency" is answered in 0.1 and 0.3.
- *What it leaves open*: the size. A nominal 4.0 × 2.7 m wall is within 2.4 dB per angle and 2.2 dB diffuse from
  125 Hz for panels from 1.23 × 1.48 m to 10 × 3 m; a small pane at 31.5–63 Hz is 4–6 dB off. A fit at lab size
  absorbs most of it (glass 4 mm: 0.93 dB nominal, 0.86 dB at its own size), so it matters only where one material is
  used at very different sizes. A column (`L = 4S/U`, square assumed for `q`) closes it for +1 member and the column
  files of 0.6 (iii).

Not recommended: **(b)** any fixed angle — right in one octave per size, off by up to 14 dB at 31.5–125 Hz;
**B78 as built** — its gain on the linings comes from freezing the cavity spring, which is the next point, not
grazing; **(c)** and **(e)** — the same cost as (d), right only on the diffuse average from 125 Hz, 4–34 dB off per
angle past 78°.

**Side findings** (into the existing items, no new one):

- *The cavity spring of `res*` with `resS > 0` at the true cosine* sweeps the mass-air-mass dip up as `f₀/cos θ` and
  through the band at grazing; a cavity partitioned by battens, studs or a spacer is locally reacting and does not. With
  the spring at normal incidence the double units and linings fit to 0.04–1.12 dB (0.6). Belongs to the double-leaf /
  lining rows of `md/acoustic.md` 3.4 (M9, N3).
- *Heavy walls above coincidence*: their knobs answer to the plate physics above `f_c` (loss, shear, the measured slope),
  not to grazing; the physical finite plate misses them by up to 10 dB in thirds (round 1). Belongs to the item that
  holds the heavy walls' knobs (`md/acoustic.md` 3.4, the installed loss factor and M13, and the parameter session).

**Decisions for Stephan:**

1. Adopt (d), Davy's σ in the mass term — or keep (a), the infinite plate.
2. The size: one nominal constant (proposed 4.0 × 2.7 m: 4S/U 3.22 m, 10.8 m², a room wall and an ISO 10140 wall
   opening alike), or a column with that nominal as its default.
3. M14 closed as superseded by (d) (if 1 is yes).
4. Whether the cavity-spring cosine of `res*` (locally reacting cavity) is studied next, under the double-leaf item.

### 0.8 Rig and traps

Scratch `build_linux/scratch/grazing` (removed by `make clean`):

- `fp.py`, `t_fp.py`, `heavy*.py`, `light.py`, `README.txt`, `sig48_*.npz`: copies from `a1/p3`; `a1c/`: round 1's
  `mat.py`, `model.py`, `ref/`; `lit/`: text extracted from the PDFs used.
- `sigcache.py a b [procs]` → `sg_<a>x<b>.npz` (σ per f, θ, φ; seven sizes); `chk_sig.py` its checks (0.0).
- `gzlib.py`: the plates, the layered transfer-matrix model with a radiation efficiency on both outer faces
  (`den_of`, `pi_of` = τ·cos θ, `tau_inf`), Davy's σ and ⟨σ⟩ (`davy_sigma`, `davy_mean`), octave helpers.
- `gz1.py` (step 1), `gz2.py <construction> a b` (finite per angle, `gz2_*.npz`), `gz2m.py` (modal per angle),
  `summ2.py`, `gz4.py` (Wareing), `gz5.py <construction> a b` (candidates per angle, `gz5_*.npz`), `summ5.py`,
  `fig.py` (`fig1_*.png`, `fig2_*.png`, `fig3_equivalent_angle.png`).
- `txg.py`: a copy of `tools/acoustic/txfit.py` whose `TL_oct` goes through `T_mat` with the switches `m14` (every
  cosine clamped, as `txlib`), `m14r` (the mass cosine only), `clim`, `davy`, `soft`, `size` and `reslocal` (the `resS > 0`
  cavity spring at normal incidence); with none of them it
  reproduces `txfit.py` to the bit (checked on the 13 fits). `gfit.py --variant JSON --seeds N --procs P --out F`: the
  fit of `md/m10_m15_report.md` 0.2 per variant, seeded also with the M10 0–90° vectors, `size: "lab"` for each
  construction's laboratory size; `fits/V*.json|out` and the controls `fits/W*.json|out`; `sumfits.py` → `sumfits.out`
  (the tables of 0.6 (ii)); `gz_eqang.out` (the frequencies of 78° / 85°, 0.5).

Traps met:

- **numpy BLAS threads in multiprocessing workers**: `tensordot` / `einsum` in each worker start one thread per
  hardware thread; six workers of `gz2m.py` made a load of 200. Run every pool with `OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`; kill a stuck pool by the pids of the forkserver's children.
- **Clip per frequency, not per band**: a finite plate's τ exceeds one in the last fraction of a degree; a reference
  clipped after the band mean differs from a candidate clipped per frequency by up to 6.7 dB at 85–90° (glass, lab
  pane). A ray model clips per frequency; so does `gz5.py`'s reference.
- **Per-angle band means need frequency density, the diffuse mean needs angle density**: 32 frequencies per third
  (coincidence peaks of width η) per angle; the diffuse means use 0.02° (gz2) or 0.05° (gz5) steps.
- `fp.tau_finite`'s 24 Gauss angles are too coarse near coincidence (README); the cached σ on 0.25° near grazing,
  interpolated, replaces them.
- A multiprocessing pool cannot pickle a `Plate` whose `eta` is a lambda: pass the construction's name and build the
  plate in the worker.

## 1. Phase 1 — fixtures (2026-10-19)

Decided 2026-10-19 (manager's note): (d) adopted — Davy's σ_D in the mass term in place of `1/cos θ`, with the gain
`1/(σ cos θ)` and the transmission clipped at one per frequency, on the radiation cosine only, `m > 0` only; the size a
new column `side` = √(panel area) in m (2a = side, S = side²), default 3.29 m where absent, 0 = the infinite plate,
< 0 refused, read only where `m > 0`; a bonded run takes the side of the body that forms its term; M14 closed; the
locally reacting cavity of `res*` not in this round. Scratch: `build_linux/scratch/grazing/p1`.

### 1.0 Baseline and the what-if

- At the start: `git status` as at the end of phase 0 (`md/acoustic_manager.md` modified by the manager, the brief and
  this report new); bc24b0c; `src/ray_mesh_interact.cpp` 5116 lines.
- **What-if build** (`p1/wd.cpp`, a copy of the source with (d) and the column; linked in front of `libquadriga.a`,
  `p1/build.sh`, run from the repository root). Its changes: `MaterialCols::side` (resolved, `require(side, "side",
  false)`), `Material::side` (default 3.29), `same_as` compares `side` where `m > 0`, `Material::davy_sigma` (7 lines:
  `k = 2π·f_GHz/0.2998`, `ka = max(k·side/2, 0.1)`, Davy's Eqs. 19, 28, 34, 36, 37), `mass_TL(X0, c, fGHz, side)`
  (today's form where `side ≤ 0`, else `A·log10(1 + (X0/σ)²)/log10(1 + X0²) + 10 log10(σ c)` clipped at 0), and the
  three call sites: `medium_gain` with the body's own side for both terms, the sheet stack's plate rule
  (`mass_TL(mass_bonded, …)`) with the near body's side, `ray_state_update`'s plate rule (`mass_TL(MASS_RUN(), …)`)
  with the previous body's side. +31 / −8 lines against the source (5139 lines). The `ka ≥ 0.1` floor lies below
  5 Hz for a 1.35 m pane; it keeps `h` (Eq. 28) finite.
- On the old test files the what-if fails 17 `[acoustic]` cases (999 assertions) and moves the printed rows of 18 more;
  `[wave]` is identical to its baseline (every line but the seed). The plain copy reproduces the baselines.

### 1.1 The size convention, checked (`p1/sizechk.py`)

Davy's form with the true area and perimeter (2a = 4S/U, S = ab) against the one-number sizes (a square of side L:
2a = L, S = L²), and every form against the windowed finite plate (`fp.py`, azimuth mean), glass 4 mm and concrete
152 mm, 16 frequencies per third, 0.05° steps, largest |ΔTL| per angle / of the diffuse average, dB:

| size (aspect) | √area vs true | 4S/U vs true | diagonal vs true | true vs windowed | √area vs windowed | 4S/U vs windowed |
| --- | --- | --- | --- | --- | --- | --- |
| 1.23 × 1.48 (1.2 : 1) | 0.01 / 0.01 | 0.04 / 0.03 | 3.00 / 2.80 | 0.54 / 0.38 | 0.53 / 0.37 | 0.54 / 0.38 |
| 1.20 × 2.40 (2 : 1) | 0.16 / 0.13 | 0.50 / 0.44 | 3.68 / 3.31 | 0.49 / 0.35 | 0.51 / 0.35 | 0.51 / 0.36 |
| 4.37 × 2.84 (1.5 : 1) | 0.06 / 0.06 | 0.14 / 0.10 | 1.85 / 1.69 | 0.40 / 0.22 | 0.35 / 0.19 | 0.43 / 0.22 |
| 4.00 × 2.70 (1.5 : 1) | 0.05 / 0.05 | 0.12 / 0.09 | 2.01 / 1.79 | 0.43 / 0.24 | 0.38 / 0.20 | 0.43 / 0.24 |
| 10 × 3 (3.3 : 1) | 0.45 / 0.39 | 0.62 / 0.38 | 2.33 / 1.83 | 0.79 / 0.39 | 0.79 / 0.62 | 0.44 / 0.29 |
| 10 × 1 (10 : 1) | 1.48 / 1.25 | 4.26 / 3.14 | 5.28 / 4.52 | 2.47 / 1.72 | 2.47 / 2.33 | 1.80 / 1.42 |

(The larger of glass and concrete per cell; σ for 10 × 1 m computed for this check.) The manager's numbers stand: √area
is within 0.45 dB of the true-size form up to 3.3 : 1 and 1.5 dB at 10 : 1, where 4S/U is 4.3 dB and the diagonal
5.3 dB off (the manager read 4.2 / 4.4; same ranking). Against the physical finite plate √area is as good as the true
form up to 3.3 : 1; on the 10 : 1 strip Davy's form itself is 2.5 dB off (a strip is not the square his constants were
fitted to), and 4S/U happens to be closer (1.8 dB).

### 1.2 What the fixtures state

New, in `test_acoustic_state_update.cpp` after *Mass law - m switches the limp mass law on*, with three helpers in
`namespace r1`: `davy_sigma` (Davy's equations written from the paper, `k = 2π f/343` in hertz, not library code),
`mass_TL_side` (the decided term from it), `wall_g` (the walker with a ray that starts close to the wall and whose path
through it is centred on y = 0.37, so that 89.9° still crosses a traced solid on the 400 m faces):

- *Finite plate - a mass-law panel of side s radiates with Davy's forced radiation efficiency*: 4 mm glass with
  `side` 1.35 m and 152 mm concrete with `side` 3.29 m (limp, `fRef` from the density, `tf` 0.05), sheet and traced
  solid, types 4 and 5, 31.5 / 63 / 125 / 500 / 2000 Hz, 0 / 30 / 45 / 60 / 70 / 75 / 78 / 80 / 82 / 84 / 85 / 86 / 87 /
  88 / 89 / 89.5 / 89.9° (the traced solid to 89.5°, below): walker T against the plate rule of the reference term,
  0.01 dB; and the normal-incidence change at 31.5 Hz, which the reference puts at +7.83 dB (pane) and +3.03 dB (wall)
  over the infinite plate, asserted > 1 dB.
- *Finite plate - side 0 is the infinite plate, a table without the column is side 3.29, a wave medium ignores it, a
  negative side is refused, a bonded run takes the side of the body that forms its term*: side 0 against the infinite
  plate's rule (0.01 dB; the face's open normal incidence costs the sheet 0.008 dB), no column against side 3.29
  bitwise (T and R), a wave medium (m 0) with `side` 1 against none bitwise (types 1, 4, 5, sheet and solid), side −1
  throws `std::invalid_argument` (through `medium_gain`), and a bonded run of two 152 mm concrete solids with sides
  1.35 | 3.29 m against `TL(X1; s1) + TL(X1 + X2; s2) − TL(X1; s2)` (0.02 dB, the run's code is quantized).

Restated (the 17 cases (d) moves; the choice per the decision note):

| case | how | today |
| --- | --- | --- |
| *Mass law - m switches the limp mass law on* | reference at the default side 3.29 m (`r1::mass_TL_side`); the table has no column | fails, 72 |
| *Plate rule - a rigid mass-law panel passes …* | reference at the default side 3.29 m | fails, 32 |
| *Thin sheet - mass-law angle dependence* (mesh_interact) | `side` 0: its statements (20 dB per decade, the knee, no cutoff) are the infinite plate's | passes |
| *Acoustic - medium_gain in-medium loss* (mesh_interact) | `side` 0: the column definition at normal incidence | passes |
| *ray_progress - the bonded plate at any depth*, *ray_progress - the plate rule and the absorption peak at every frequency of a band* (progress) | `side` 0 in their palettes | pass |
| *Stack with a mass-law layer …*, *Leak reciprocity …*, the three *Bonded mass …*, *Mass cosine …*, *Total reflection of the outer pair … - the mass-law part*, *Closed plate …*, *Resonance - on a mass-law material …*, *Coincidence - on a mass-law material …*, *Absorption peak - tfP, tfF and tfQ …* | every palette through `side0(…)` (a helper at the top of the file: the column at zero) — their references are written for the infinite plate and state another mechanism | pass |

Side-0 restatements pass today by construction (the column is ignored) and keep their printed rows bitwise; they are
the guards that side 0 is the infinite plate wherever those mechanisms meet it.

OBJ header and default row with `side` after `tfQ`, default 3.29: `test_obj_file_write.cpp` (*csv_write_defaults = true
→ full canonical set with defaults*), `test_obj_file_write.py` and `test_obj_file_write.m`.

Test diff: 236 insertions, 32 deletions in six files.

### 1.3 Against today's library

Whole suite, the new test files on today's library (`p1/t_today_all.txt`): 1145 cases, 1316330 assertions, **5 cases
fail with 693 assertions**: *Finite plate - … radiates …* 570, *Finite plate - side 0 …* 17, *Mass law - m switches …*
72, *Plate rule - a rigid … panel …* 32, the OBJ section 2 (header and row). Compared with the same library on the old
test files, no other case's printed rows move (`tools/acoustic/cases.py`). The same test files on the what-if: **one
case fails with 2 assertions** (the OBJ header, which the what-if does not change); `[acoustic]` 58 cases, 41523
assertions, all pass; `[wave]` 180 / 165258, every line identical to `test_wave.txt` on both; `tests/data/wave_twins.tsv`
untouched.

The expected deviations, today against the (d) reference (largest walker − reference per material and frequency, the
transmitted power in dB): glass 4 mm +3.4 dB at 31.5 Hz (0°), +3.2 / +3.7 dB at 63 / 125 Hz (60 / 70°), +6.0 dB at
500 Hz (84°), +8.4 dB at 2 kHz (88°); concrete 152 mm +14.5 / +14.8 / +16.0 dB at 31.5 / 63 / 125 Hz (87–89°),
+18.0 / +20.6 dB at 500 Hz / 2 kHz (89.9°). Today's library passes more: the infinite plate near grazing, and an
infinite plate at 31.5 Hz.

Other cases whose rows move under (d) (all pass; `p1/moves.py`, paired rows, today → what-if):

| case | rows | largest change | where and why |
| --- | --- | --- | --- |
| *Absorption peak - a peaked face is its series impedance …* | 441 | R + T −0.51 at 85° (EM, VV) | the energy-check columns: T of the mass-law board near grazing at the default side; R and its phase against the reference untouched |
| *Leak direction - a wave held in a mass-law body keeps its direction …* | 145 | −0.71 dB at 75° | the walls' terms near grazing (concrete, default side) |
| *ray_progress - the wall behind a traced layer is met at the incidence angle …* | 114 | −4.0 dB at 80° | the mass-law wall's term at 80° |
| *Leak direction - a closed face hands the wall behind a traced layer …* | 33 | −5.6 dB at 85° | the wall's term at 85° |
| *Lossy verdict - … the mass-law part* | 10 | −5.6 dB at 85° | the same wall at 85° |
| *Held lining …*, *Held exit …* (two), *Held path …* | 16 / 5 / 1 / 2 | 0.26 to 0.36 dB less transmitted at 75–80° | a held plate's term near grazing |
| *Leaf - a leaf on a box of air …* | 40 | α 1e-4 | α = 1 − R − T, T through the 200 mm mass-law wall at 50 Hz |
| *Unentered branches - …* (three) | 24 / 3 / 6 | 3e-5 / 1e-4 / 1e-4 dB | the `q` term of the default side at 60–77° |
| *Refracting tracer - … the mass-law part* | 10 | 5e-6 dB | the same at 37–41° |
| *Plate rule - a traced mass-law body passes in one path …* | 10 | 4e-4 dB | glass 4 mm at 60° |
| *Phase reference …*, *Stack geometry …*, *Plate rule - a wall bonded of two …* | 2 / 4 / 6 | 0 | the sign of a printed zero |

Every move is the decided term at the default side: near grazing the finite plate passes less (to −5.6 dB at 85°),
elsewhere the `q` term and the `1/√(g² + q²)` form move the fourth decimal at most.

**Side findings** (measured; not in this round):

- *A traced leg far from the origin or near grazing loses precision.* The same 4 mm traced pane crossed at y = −100 m
  passes 0.2 dB more at normal incidence than crossed at y = 0.37 m (`PROBE wall vs wall_g`, a temporary probe:
  −31.454 against −31.658 dB at 500 Hz); and today's library, crossed at 89.9° (an 87 m path through the 152 mm wall),
  misses its own infinite-plate rule by −0.007 / −0.027 / −0.10 / −1.17 / −3.42 dB at 31.5 / 63 / 125 / 500 / 2000 Hz,
  while at 89.5° it holds to 1e-4 dB. Both look like the precision of the traced leg's length and cosine in large
  coordinates; not investigated. The new case stops the traced solid at 89.5° and centres its crossing for this reason.
- *A bonded run with two sides* passes, within 0.0003 dB in every row, what the whole run passes at the **first** body's
  side (T_all_s1 in the guard's table): above the first body every increment `TL(X_run + X0; s) − TL(X_run; s)` is
  nearly independent of `s` where the mass controls. The run's second side is then almost unused; with the order
  reversed the run takes the other side. "One side per bonded run" as the rule of use avoids the direction dependence.

### 1.4 4|16|4 meshed (a rule of use; no fixture)

Two 4 mm glass mass-law sheets (rigid, `fRef` from the density, the window-glass `tf` law, coincidence 2989 Hz, η
0.015, no side column) 16 mm apart, with a plain air gap or a near-air box (μ 1.000001) between, against the AGC unit as
one lumped sheet with its M10 0–90° columns, against the transfer matrix of the two plates and the cavity (`gzlib`,
infinite, and with Davy's σ at 3.29 m), and the measured AGC octave; walker (`p1/probe4164.cpp`, linked into the plain
copy and the what-if), single frequencies, TL dB, today / under (d); types 4 and 5 agree within 0.05 dB:

| f, θ | gap | box | lumped | TMM ∞ | TMM 3.29 m | measured octave |
| --- | --- | --- | --- | --- | --- | --- |
| 125 Hz, 0° | 19.7 / 19.7 | 29.2 / 29.2 | 29.4 / 29.4 | 21.9 | 21.9 | 20.5 |
| 125 Hz, 45° | 16.8 / 17.7 | 38.2 / 40.0 | 26.5 / 27.4 | 21.0 | 21.9 | |
| 125 Hz, 80° | 7.8 / 12.7 | 11.9 / 21.8 | 15.9 / 22.3 | 10.7 | 18.1 | |
| 250 Hz, 0° | 25.6 / 25.6 | 39.8 / 39.8 | 18.1 / 18.1 | 23.7 | 23.7 | 16.8 |
| 250 Hz, 45° | 22.6 / 22.7 | 50.2 / 50.2 | 16.2 / 16.2 | 18.3 | 18.3 | |
| 250 Hz, 80° | 11.8 / 16.7 | 19.9 / 29.9 | 13.2 / 13.7 | 16.1 | 22.1 | |
| 1 kHz, 0° | 37.6 / 37.6 | 70.1 / 70.1 | 43.7 / 43.7 | 70.2 | 70.2 | 36.4 |
| 1 kHz, 45° | 34.4 / 34.4 | 74.0 / 74.0 | 40.4 / 40.4 | 60.3 | 60.3 | |
| 1 kHz, 80° | 21.7 / 24.2 | 40.7 / 45.8 | 28.0 / 30.5 | 20.8 | 23.3 | |
| 4 kHz, 0° | 49.7 / 49.7 | 104.3 / 104.3 | 61.0 / 61.0 | 104.7 | 104.7 | 36.5 |
| 4 kHz, 45° | 41.5 / 41.5 | 85.3 / 85.3 | 53.7 / 53.7 | 86.4 | 86.4 | |
| 4 kHz, 80° | 31.2 / 31.2 | 62.5 / 62.5 | 39.5 / 39.5 | 56.0 | 56.0 | |

(The measured column is the octave around that frequency; the lumped sheet's fit is an octave fit, so its single
frequencies need not sit on it.) Read: **the meshed unit with a box is the unbridged double leaf above its resonance**
(1 / 4 kHz at 0°: 70.1 / 104.3 against the transfer matrix's 70.2 / 104.7) — at 1 kHz 34 dB above the measured unit,
whose edge spacer bridges the panes; before October 15 it read 105 dB there. It **misses the mass-air-mass dip**
(125 / 250 Hz: 29.2 / 39.8 against 21.9 / 23.7): mass-law sheets carry their term as a power factor and the box's
series cannot resonate on them. A plain gap gives two uncoupled plates (no cavity at all). (d) changes only the 80° rows
and 125 Hz at 45°. Rule of use: a double unit is one lumped sheet (its fitted columns), or two wave-medium leaves on a box
(M9, `md/acoustic.md` 2.8), which resonate; not two meshed mass-law sheets.

### 1.5 The 13 refitted under (d)

`gfit.py` (procedure of `md/m10_m15_report.md` 0.2, 8 seeds plus the M10 0–90° fit as a start, selection at 768 × 12)
with the decided form (`txg.py` switch `davy` with `side`: 2a = side, S = side²), at 0–90°, twice:

| construction | worst, side = √(lab area) | worst, side 3.29 m | lab side, m |
| --- | --- | --- | --- |
| glass 4 mm | 0.85 | 0.96 | 1.349 |
| glass 6 mm | 1.78 | 1.95 | 1.349 |
| gypsum 12.5 (model) | 2.54 | 2.68 | 1.697 |
| plywood 18 (model) | 0.31 | 0.39 | 1.697 |
| concrete 152 | 0.57 | 0.57 | 3.523 |
| brick 95 | 0.01 | 0.08 | 3.523 |
| brick 203 | 0.58 | 0.59 | 3.523 |
| block 190 | 0.49 | 0.47 | 3.523 |
| 4-16-4 (AGC) | 1.56 | 1.47 | 1.349 |
| 4-12-4-12-4 (AGC) | 2.67 | 2.59 | 1.349 |
| brick 203 + furred board | 2.70 | 2.70 | 3.523 |
| SCR 140 + board on dabs | 1.95 | 2.65 | 3.523 |
| brick 105 + plaster one face | 0.00 | 0.00 | 3.523 |
| **worst** | **2.70** | **2.70** | |

(Today's library at 0–90°: 2.97 dB.) The knobs (level = fitted `m″` over the mass): lab side — light 1.13 / 0.90 (glass
4 / 6), 0.64 gypsum, 0.88 plywood, 0.41 / 0.36 the units; heavy 0.02 … 0.39 with `att` 7.0 / 2.4 / 2.3 / 8.9 / 12.0 /
12.0 / 9.9 dB (concrete, brick 95, brick 203, block, furred, dabs, plaster; two at the 12 dB bound), `coiF` 6.9 / 5.6 /
18.9 / 8.2 kHz and 30 Hz (furred, the bound); side 3.29 — light 1.28 / 0.96 / 0.68 / 0.98, units 0.49 / 0.43, heavy
0.02 … 1.15 (dabs 1.15, `coiF` 76 Hz), one `att` at the bound (furred). The columns for `md/acoustic.md` 2.7, in
library units (`fRef` GHz, `att` at `fRef`, `coiF`/`resF` in Hz here, `tf` at `fRef`), are in `fits/P1_davy_side_lab.json`
and `fits/P1_davy_side_329.json`; per construction (lab side): glass 4 mm `fRef` 4.0562e-05, `coiF` 2967 Hz / Q 284,
`res` 138 Hz Q 2.33 (+), `tf` 5.661 / −0.37; concrete 152 `fRef` 1.9663e-04, `att` 14.5 at `fRef` / `attB` −0.09,
`coiF` 6925 Hz / Q 1.19, `res` 78 Hz Q 1.44 (−), `tf` 2.549e-04 / 0.52. As in phase 0, the heavy walls keep their knobs
off the physics above coincidence; (d) does not, and was not meant to, change that.

### 1.6 Python and MATLAB

- Python (`build_linux/scratch/api`, the module is current: no source change since 2026-10-17; `pytest
  tests/python_tests/test_obj_file_write.py` from a scratch directory): 9 passed, **1 failed**,
  `test_csv_columns_defaults_validation`, the restated header (today's writer has no `side`).
- MATLAB (`build_linux/scratch/api12`, current; `test_obj_file_write` called directly with MOxUnit's asserts on the
  path, from a scratch directory): stops at line 185, the restated header. (`moxunit_runtests` on the file or on a
  directory holding it reports 0 tests for these plain test functions.)

### 1.7 Run times

`--durations yes`, under `flock /tmp/bench.lock`, the new test files (`p1/t2_*_all.txt`):

- Today's library: load 3.57 before, 2.66 after (1-minute); the 1145 cases sum to 43.0 s; *Finite plate - … radiates …*
  0.02 s, *Finite plate - side 0 …* 0.00 s, *Mass law - m switches …* 0.01 s, *Plate rule - a rigid … panel …* 0.00 s,
  *Thin sheet - mass-law angle dependence* 0.01 s. The new cases add no measurable time.
- The what-if: load 2.66 before, **42.19 after** (another session started during the run); sum 234.6 s. Contaminated:
  no conclusion drawn; the what-if's per-case times for the cases above read 0.00–0.01 s.

### 1.8 For phase 2

- The source as the what-if (`p1/wd.cpp`): +31 / −8 lines (5139); the column lists of `obj_file_write.cpp`,
  `qpy_obj_file_read.cpp`, `obj_file_read.cpp` and `band_probe.cpp` (`side` after `tfQ`); the writer's default 3.29.
- The slab-threshold comments (`include/quadriga_tools.hpp:125`, `calc_diffraction_gain`'s description) to the eps
  entry of `ray_state_update`'s description.
- `tools/acoustic/check_docs.py` now fails on one check only: `md/acoustic.md` does not yet name the two new cases
  (phase 3). Baseline counts and every other check pass.
- Open to decide: whether `same_as` compares `side` (the what-if does, where `m > 0`: two otherwise equal mass-law
  materials of different sides are two media and get a face between them).

## 2. Phase 2 — source (2026-10-19)

Decided 2026-10-19 after phase 1: `same_as` compares `side` where `m > 0` (wave media ignore it); "one side per bonded
run" is a rule of use, no code (`md/acoustic.md` records in phase 3 that a mixed run takes the side of the first body the
ray meets, within 0.0003 dB, so the result depends on direction); the precision finding of 1.3 is a separate
investigation, recorded in `md/acoustic.md` 7 in phase 3. Scratch: `build_linux/scratch/grazing/p2`.

### 2.1 What changed

- `src/ray_mesh_interact.cpp` (5116 → **5159 lines**, +56 / −14, most of it comment): (d) and the column as the what-if
  of 1.0, which it reproduces to the printed digit (2.2). `MaterialCols::side` with `require(side, "side", false)`;
  `Material::side`, default 3.29 (a 4.0 × 2.7 m wall); `same_as` compares it where `m > 0`; `Material::davy_sigma`;
  `mass_TL(X0, mass_c, fGHz, side)` with today's form where `side ≤ 0`; the three call sites (`medium_gain` at the
  body's own side, the sheet stack's plate rule at the near body's, `ray_state_update`'s plate rule at the previous
  body's). The comments state the physics: the plate's forced wave radiates `ρ₀c/cos θ` per unit velocity on an infinite
  plate and turns it transparent at grazing; a finite panel radiates with a bounded efficiency; `1/σ` replaces the
  cosine and `1/(σ cos θ)` is the finite plate's gain; `side` = √(panel area), 0 the infinite plate, read only where
  `m > 0`; the plate's own impedance stays at the cosine in air. The public `medium_gain` text says the same and that
  at `cos θ = 1` it returns `A` for `side` 0 and `A` up to the panel's low-frequency term otherwise.
- `side` after `tfQ` in `src/obj_file_write.cpp` (canonical order; default 3.29 where `csv_write_defaults` fills a
  column), `api_python/qpy_obj_file_read.cpp`, `api_mex/obj_file_read.cpp`, `tools/acoustic/band_probe.cpp` (column
  list and header comment: 24 columns per material, `side` last).
- `tools/acoustic/band_probe_example.in`: a `side` per material (0 for the two wave media, 3.29 for the mass-law wall)
  and a note. Its output (`p2/bp_example_*.out`): R bitwise as before (|R|² 0.925135 / 0.649358 / 0.444760 / 0.778824
  as the header states); T through the wall −45.146 → −45.299 dB at 80 Hz (the 3.29 m panel's low-frequency term).
- Comment only: `include/quadriga_tools.hpp:125` and `calc_diffraction_gain`'s description of `thin_slab_threshold` now
  say what `ray_state_update`'s `eps` entry says: a series is summed where `ρ ≥ eps`; 0 always; `eps ≥ 1` re-emits every
  series whose `ρ` is below one, and a round trip at or above one is always summed.
- Not changed this phase (phase 3): `tools/acoustic/README.md` still says 23 columns per material for the band probe;
  `tools/acoustic/txlib.py` / `txfit.py`; the documents.

### 2.2 The suite

- `[wave]`: 165258 assertions in 180 cases, all pass; every line identical to `test_wave.txt` (seed line aside);
  `tests/data/wave_twins.tsv` unchanged.
- `[acoustic]`: **41523 assertions in 58 cases**, all pass (before this round 40050 in 56).
- Whole suite: **1316330 assertions in 1145 cases**, all pass (before 1314857 in 1143).
- Moved rows (`tools/acoustic/cases.py`, `--durations yes` runs, same test files): against phase 1's what-if on the
  same tests, **0 differing lines** (the OBJ section now passes); against today's library on the same tests, the 22
  cases of 1.3 — the two new ones, the two restated at the default side, and the 18 of the table of moves — and no other.
  Every moved row is accounted for there.

### 2.3 Python and MATLAB

- Python (`build_linux/scratch/api`, `quadriga_py` rebuilt from this source): `test_obj_file_write.py` and
  `test_obj_file_read.py`, **16 passed**.
- MATLAB (`build_linux/scratch/api12`, every MEX file rebuilt with g++-12; `obj_file_read.mexa64` needs at most
  GLIBCXX_3.4.26): `test_obj_file_write` and `test_obj_file_read` called directly with MOxUnit's asserts on the path,
  **both pass**.

### 2.4 The 13 constructions on the built library

The fits of 1.5 at their laboratory side (√(lab area): 1.349 m panes and units, 1.697 m boards, 3.523 m walls and
linings) as library columns, the face rigid (`a` 8.4794e-10, as the previous round's confirmation):

- Walker (`p2/conf.py` with `p2/probe.cpp`, the previous round's walker probe with the `side` column, linked against the
  new `libquadriga.a`; 96 angles 0–90° × one frequency per third, type 4, sheet and traced solid, 67392 rows): **sheet
  and formula copy within 0.0000 dB** on every construction, **traced solid within 0.0042 dB** (glass 4 mm; 0.0000 to
  0.0025 elsewhere), so **sheet and traced solid agree within 0.0042 dB**. **R + T at most 0.9999999989** (glass 6 mm,
  traced, 89.5°, 3.2 kHz: T at its clip, the face's `a`), never above one in any row. (The walker's residual on this
  grid reads up to 4.97 dB where 768 × 12 reads 2.70: the sampling of `md/m10_m15_report.md` 0.2, not the library.)
- Band through `ray_progress` (`p2/band.py` with `tools/acoustic/band_probe.cpp` rebuilt; sheet and traced solid,
  tracers 4 and 5, 0 / 30 / 60 / 80°, one band of six frequencies 125 Hz–4 kHz against each frequency alone, summed and
  with `thin_slab_threshold` 1; `att` flat at its `fRef` value in probe and formula, the probe has no `attB`): **every
  value of the band equals its single frequency bitwise (2496 values)**; within 0.0026 dB of the formula (4-16-4,
  traced, 250 Hz, 0°; float palette); **R + T at most 0.998446**.

### 2.5 Run times

The machine stayed loaded (other sessions; a `--durations yes` run under `flock /tmp/bench.lock` went from load 2.93 to
35.38 and is discarded). Measured instead as **CPU time (user + system) of single-threaded runs** (`OMP_NUM_THREADS=1`,
`/usr/bin/time`), today's library and this phase's on the same test files (`p2/tb_old`: `p1/plain.cpp` linked with the
new test objects), alternated back to back under `flock /tmp/bench.lock` (`p2/cputime.sh`); CPU time is far less
sensitive to load than wall time, and alternating cancels drift:

| run | load (1-min, start → end) | today's library, s | this phase, s | ratio |
| --- | --- | --- | --- | --- |
| `[acoustic]`, 5 pairs | 11.56 → 10.08 | median 0.84 (0.83–0.86) | median 0.84 (0.83–0.85) | 1.000 (pairs 0.988–1.012) |
| whole suite, 3 pairs | 8.61 → 2.00 | median 45.9 (45.9–46.1) | median 46.0 (46.0–46.4) | 1.003 (pairs 0.998–1.011) |

No measurable cost: the two new cases and the restated ones run in 0.01–0.02 s each (`--durations`), and the term's
extra work (one square root and two logarithms per mass-law event) is below the run-to-run spread.

### 2.6 For phase 3

- `md/acoustic.md`: 2.1 (the finite plate, the column, the public `medium_gain`), 2.7 (the columns of 1.5), 3.4, 4 (M14
  closed as superseded), 7 (the precision finding of 1.3 with its numbers), 9, 10 (the decisions of 2026-10-19), the
  appendix (the two new cases); the bonded-run rule of use and its direction dependence; that wave media, including
  the leaves of a meshed double wall, keep the infinite plate (2.8 / 7); the meshed 4|16|4 rule of use (1.4).
- `tools/data_formats_md/format_materials.md` (the column and its description), `tools/acoustic/README.md` (24
  columns), `tools/acoustic/txlib.py` and `txfit.py` (the switch as built: `side` per construction, nominal and lab);
  the finite-plate model into `tools/acoustic` if wanted; regenerate the three baselines; `check_docs.py`.

## 3. Phase 3 — documents (2026-10-19)

Resources from 2026-10-19: at most 12 threads (another session shares the machine); every build and run below used
`--parallel 12` and `OMP_NUM_THREADS=12` or fewer, one suite at a time.

### 3.1 What changed

- `md/acoustic.md` (1441 lines, +130 / −13):
  - 2.1: the finite panel in physical words (the forced wave's `ρ₀c/cos θ` and the transparency of the infinite plate at
    grazing; Davy's bounded `σ`; `1/σ` in place of the cosine and the gain `1/(σ cos θ)`; where the finite plate parts
    from the infinite one; the low-frequency term), the column (`side` = √(panel area), 3.29 m by default, 0 the infinite
    plate, read only where `m > 0`; √area against the true area and perimeter), the plate's own impedance at the cosine
    in air, the public `medium_gain` at `cos θ = 1`, the cases that assert it and which carry `side` 0. The infinite
    plate's angle law is now stated as such.
  - 2.2: "one side per bonded run", with the measured direction dependence of a mixed run.
  - 2.7: the rows on level, coincidence, mass-air-mass, first mode and `att` from the fits at lab side; a row for
    `side`; the table of the 13 fits as library columns with `side` (worst 2.70 dB).
  - 2.8: wave media, the leaves of a meshed double wall included, keep the infinite plate; the meshed 4|16|4 rule of use
    with its numbers (a double unit is one lumped sheet or two wave-medium leaves on a box).
  - 3.4: the finite-size row: expressible with `side`, the 78° / 85° conventions as special cases, Wareing's size effect.
  - 4: M14 closed, superseded by the finite panel; the locally reacting cavity of `res*` as an open side finding for the
    double-leaf rows.
  - 7: wave media keep the infinite plate; the precision finding of 1.3 with its numbers, to be investigated.
  - 8: the side finding on the heavy walls' knobs no longer points to M14.
  - 9: fits at each construction's laboratory side, `txfit.py`'s `side`.
  - 10: the decisions of 2026-10-19 and "Built (2026-10-19)" with the line count (5159 against 5116) and the counts.
  - The first-step line quotes `[acoustic]` 41523 / 58 with the 12-thread command; the appendix names the two new cases.
- `md/wave.md` (+7 / −4): section 0's commands with `--parallel 12` and `OMP_NUM_THREADS=12`, the counts (180 /
  165258, 58 / 41523, 1145 / 1316330), and one line on why (a shared machine; one suite at a time).
- `tools/data_formats_md/format_materials.md`: the `side` row with the decided description and default 3.29 (m), a
  paragraph on the panel size under the parameters' meanings, and the mass-law paragraph now naming its angle law the
  infinite plate's.
- `tools/acoustic/txlib.py` (128 lines): `davy_sigma` and `mass_TL` as built, a material's `side` (3.29 where absent as
  the library, 0 the infinite plate); M14 marked closed; the header records the walker check below.
- `tools/acoustic/txfit.py` (361 lines): the same term in the vectorized fit (`_radiation`, cached per grid), `side` in
  the variant (`"lab"` per construction from `LAB_SIDE`, a number for one side, 0 the infinite plate; default `{"amax":
  90, "side": "lab"}`), `columns()` with `side` and the rigid face's `a`. Checked: `side` 0 reproduces the previous
  fitter bitwise on the 13 M10 fits (2.97 dB); the phase-1 fits read their worst residuals to 1.4e-14 dB at the lab side
  (2.70) and at 3.29 m (2.70).
- `tools/acoustic/gzlib.py` (new, 195 lines): the reference models, self-contained (the plates of round 1's `fp.py`, the
  layered transfer matrix with a radiation efficiency on both faces, the windowed σ and its azimuth mean, Davy's σ and
  ⟨σ⟩), no σ tables; `python3 tools/acoustic/gzlib.py` runs its checks: the layered model on the closed form to 3.3e-16,
  4|16|4 at 211.3 Hz, σ·cos 30° 0.980 / 1.001 / 0.995 / 1.000 and σ(90°) within 0.12 dB of Leppington and 0.11 dB of
  Davy (1.23 × 1.48 and 4.0 × 2.7 m, 500 Hz and 2 kHz; as phase 0).
- `tools/acoustic/README.md`: the band probe's 24 columns (`side` last), `txlib.py` and `txfit.py` as built, a row for
  `gzlib.py`.

### 3.2 Checks

- `txlib.py` as built against the built library through the walker (`p3/txchk.py`, the 13 fits at lab side as library
  columns from `txfit.columns`, 96 angles × one frequency per third, type 4, 67392 rows): T within 0.0000 dB (sheet) and
  0.0042 dB (traced solid), R within 0.0000 dB.
- Baselines regenerated with the 12-thread commands (load 0.93 before, 2.08 after): `[wave]` 165258 / 180, `[acoustic]`
  41523 / 58, whole suite 1316330 / 1145, all passed; `test_wave.txt` and `test_acoustic.txt` identical to phase 2's runs
  (seed lines aside). `python3 tools/acoustic/check_docs.py`: **all checks pass** (tables, 102 keys, 58 cases named,
  the four quoted counts, spelling).
- Source unchanged this phase: `src/ray_mesh_interact.cpp` 5159 lines (5116 before the round).

### 3.3 The round, in sum

Uncommitted, 21 files (+616 / −147 by `git diff --stat`, of which `md/acoustic_manager.md` is the manager's) and two new
ones (`tools/acoustic/gzlib.py`, this report; the brief): the source (+56 / −14 in `ray_mesh_interact.cpp`, the column in
the writer, the readers and the band probe, two comments), six test files (two new cases, two restated at the default
side, fifteen with `side` 0, the OBJ header checks), the documents and tools above.
