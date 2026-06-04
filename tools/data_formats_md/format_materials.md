/*!SECTION
Material Properties
SECTION!*/

/*!SECTION_DESC
Materials are stored in a name-keyed table: each triangle references a material by row index (`csv_ind` or `mtl_ind`), the row names are returned in `csv_names` / `mtl_names`, and the per-material parameters are returned in `csv_prop` — a map from parameter name to a vector with one entry per material row (so a value is read as `csv_prop["a"][csv_ind[face]]`). The model is formulated for electromagnetic propagation — complex relative permittivity ε, relative permeability μ, and conductivity σ following Rec. ITU-R P.2040 — and reuses the same parameters for an acoustic interpretation (mass law, coincidence, resonant absorption).<br><br>

The table is schema-blind: the only required column is `name` (the join key for `.obj` materials); every other column is an optional numeric parameter. A parameter the table does not define is simply absent from `csv_prop`, and downstream consumers substitute its default (listed below). Empty cells parse as 0. The recognized parameters split into three physical roles:
- **Interface reflection** (`a`, `b`, `c`, `d`, `e`, `f`, `g`, `h`, `resF`, `resQ`, `resS`) — set the complex permittivity ε and permeability μ, which fix the Fresnel reflection coefficient `R` at every surface crossing. ε and μ govern the reflected power `abs(R)²` and the complementary room-side absorption `1 − abs(R)²`. Applied once per surface hit, independent of path length.
- **Interface transmission** (`att`, `attB`, `coiF`, `coiQ`, `coiA`) — the through-surface isolation in dB, applied once when entering a material (the front-side air→material or material→material crossing). Independent of path length and not applied on exit. The transmitted power is *not* simply the Fresnel `1 − abs(R)²`; see the partition note below.
- **In-medium attenuation** (`c`, `d`, `g`, `h` via ε·μ; `alpha`, `alphaB`; `m`) — loss accumulated along the path traversed inside a body. The loss from `Im(√(εμ))` (fed by the σ term in ε and the σμ term in μ) and the explicit `alpha` term sum, and `m` adds a mass-law slope; all scale with the in-medium distance.

**Transmission partition.** Reflection and transmission are computed independently, not as energy-complementary halves of a single Fresnel split. ε and μ (`a`, `b`, `e`, `f`, `resF…`) set reflection and room-side absorption; the lumped (`att`, `coi…`) and in-medium (`alpha`, `m`) terms set the transmission level. The transmitted field is gated by the index contrast across the crossing: on a **light→dense** crossing — entering a higher-index medium, `Re(ε₁μ₁)` < `Re(ε₂μ₂)` — it is additionally scaled by the Fresnel `1 − abs(R)²`; on a **dense→light** crossing — entering a lower-index medium, `Re(ε₁μ₁)` > `Re(ε₂μ₂)` — it is pass-through (`abs(R)²` is not subtracted). This gate applies only to the two straight-through **transmission** modes of `ray_mesh_interact` — type 1 (EM transmission) and type 4 (scalar transmission), which pass the ray through undeviated. Reflection (types 0 and 3) computes the Fresnel `abs(R)²` directly and never uses the gate; EM **refraction** (type 2) bends the ray per Snell with `(ε₁μ₁)/(ε₂μ₂)` and applies the full Fresnel transmission, bypassing the gate entirely — type 2 is the physically correct transmission path, while types 1/4 are the undeviated approximation the gate exists to keep well-behaved. Types 1 and 4 share the one gate; there is no separate scalar rule. The dense→light pass-through prevents spurious amplification and the critical-angle total reflection that an ε < 1 Fresnel transmission would otherwise impose on the undeviated path. On types 1/4 the lumped and in-medium terms set the transmission *level* (ε/μ do not); on type 2 the Fresnel transmission from ε/μ applies directly.

Both domains use the same parameter set but populate it very differently: EM materials have εr > 1 with σ-based loss and μ = 1, whereas acoustic materials use a two-parameter ε, μ — rigid reflectors with εr ≪ 1 and μ = 1, porous absorbers with εr ≈ 1 and a lossy μr > 1 — with the att / alpha / mass-law terms reserved for partitions, coincidence, and resonators. See the Electromagnetic and Acoustic interpretation sections below.
See related functions that produce or consume material properties:
- C++ OBJ reader (<a href="cpp_api.html#obj_file_read">obj_file_read</a>)
- C++ ray / mesh interaction (<a href="cpp_api.html#ray_mesh_interact">ray_mesh_interact</a>)
- C++ diffraction gain (<a href="cpp_api.html#calc_diffraction_gain">calc_diffraction_gain</a>)
- C++ Mitsuba scene export (<a href="cpp_api.html#mitsuba_xml_file_write">mitsuba_xml_file_write</a>)
SECTION_DESC!*/

/*!MD
# 1. Material Model
Complex permittivity, permeability, conductivity, and loss parameters with their frequency dependence

## Parameters:
Each parameter is one numeric column of the material table and one key of `csv_prop`. `Key` is the CSV column name (and the `csv_prop` key); a parameter absent from the table is substituted with the listed default by downstream consumers. Only the `name` column is mandatory, and column order does not matter.<br><br>

| Key    | Property                              | Units      | Default |
| :----: | ------------------------------------- | :--------: | :-----: |
| a      | εr at fRef                            | —          | 1.0     |
| b      | Frequency exponent for εr             | —          | 0       |
| c      | σ at fRef                             | S/m        | 0       |
| d      | Frequency exponent for σ              | —          | 0       |
| e      | μr at fRef                            | —          | 1.0     |
| f      | Frequency exponent for μr             | —          | 0       |
| g      | σμ (magnetic loss) at fRef            | —          | 0       |
| h      | Frequency exponent for σμ             | —          | 0       |
| att    | Penetration loss at fRef              | dB         | 0       |
| attB   | Frequency exponent for att            | —          | 0       |
| alpha  | In-medium absorption at fRef          | dB/m       | 0       |
| alphaB | Frequency exponent for alpha          | —          | 0       |
| fRef   | Reference frequency                   | GHz        | 1.0     |
| m      | Mass-law transmission slope           | dB/decade  | 0       |
| resF   | Permittivity resonance frequency      | GHz        | 0       |
| resQ   | Permittivity resonance quality factor | —          | 0       |
| resS   | Permittivity resonance strength       | —          | 0       |
| coiF   | Coincidence frequency                 | GHz        | 0       |
| coiQ   | Coincidence quality factor            | —          | 0       |
| coiA   | Coincidence loss amplitude            | dB         | 0       |

## Frequency laws:
`f` is given in GHz; `f/fRef` is the relative frequency. `resF` and `coiF` are
absolute GHz. In the μ(f) and σμ(f) rows the exponent `f` is the parameter column,
not the frequency.<br><br>

| Parameter  | Formula                                                                    | Unit   | Meaning                                |
| ---------- | -------------------------------------------------------------------------- | ------ | -------------------------------------- |
| ε(f)       | `a·(f/fRef)^b − i·17.98·σ/f + resS·resF² / (resF² − f² + i·(resF/resQ)·f)` | —      | relative permittivity (complex)            |
| σ(f)       | `c·(f/fRef)^d`                                                             | [S/m]  | conductivity (enters ε as the −i term)     |
| μ(f)       | `e·(f/fRef)^f − i·17.98·σμ/f`                                              | —      | relative permeability (complex)            |
| σμ(f)      | `g·(f/fRef)^h`                                                             | —      | magnetic loss (enters μ as the −i term)    |
| att(f)     | `att·(f/fRef)^attB + coiA / (1 + (coiQ·(f − coiF)/coiF)²)`                 | [dB]   | lumped loss added once on entering         |
| α(f)       | `alpha·(f/fRef)^alphaB`                                                    | [dB/m] | explicit in-medium loss × path length      |
| mass(f, L) | `max(0, m·log10((f/fRef)·L))`                                              | [dB]   | mass-law slope, L = in-medium path length  |

- The refractive index is `n = √(εμ)` and the wave impedance `Z = Z₀·√(μ/ε)`. With `μ = 1` (the default) these reduce to `n = √ε`, `Z = Z₀/√ε`, and every formula here matches the legacy ε-only model exactly. The Fresnel coefficient uses the admittance `√(ε/μ)`; Snell refraction and total reflection use the index ratio `(ε₁μ₁)/(ε₂μ₂)`; and the in-medium loss uses the loss tangent of the **product** `ε·μ`, so the `g`/`h` (μ-loss) term feeds bulk attenuation exactly as `c`/`d` (σ) do.
- Loss appears as a negative imaginary part of ε and μ (Rec. ITU-R P.2040-1, eq. 9b). Both the σ term and the resonance are written so that `c > 0` and `resS > 0` add loss; `g > 0` likewise adds loss to μ. The total in-medium attenuation over a path of length `L` is `8.686·L/Δ` (from `Im(√(εμ))`, where Δ is the attenuation length derived from the loss tangent of `ε·μ`) plus `α(f)·L` plus `mass(f, L)`, summed; `att` is separate and is not path-dependent.

## Mechanisms:
- **Permittivity resonance** (`resF`, `resQ`, `resS`): a Lorentz pole that adds a peak to absorption (acoustic α) and a feature to reflection near `resF`; `resQ` sets sharpness (higher = narrower). Active only when `resF > 0` and `resS ≠ 0`. Models resonant dielectrics / frequency-selective media (EM) and Helmholtz / membrane absorbers (acoustic).
- **Coincidence term** (`coiF`, `coiQ`, `coiA`): a Lorentzian added to the transmission loss at `coiF`. Negative `coiA` produces a transmission dip (acoustic coincidence / pass-band); positive `coiA` produces a stop-band. Total loss is clamped to ≥ 0. Active only when `coiF > 0` and `coiA ≠ 0`.
- **Mass-law term** (`m`): a transmission loss that is logarithmic in both frequency and in-medium path length. `m = 20` reproduces the acoustic mass law (+6 dB/octave and +6 dB per thickness doubling). Default 0 (EM through-loss is the linear `alpha` term). The resonance denominator uses `+i·(resF/resQ)·f` so that `resS > 0` adds loss under the library's negative-imaginary loss convention (consistent with σ).
- **Permeability** (`e`, `f`, `g`, `h`): the magnetic-permeability analog μ, default 1+0j — identical to the legacy ε-only model. μ decouples the wave impedance `Z = Z₀√(μ/ε)` from the refractive index `n = √(εμ)`, so reflection and bulk loss can be matched independently — required for any medium that is not single-parameter, e.g. a porous absorber (acoustically, ε ↔ compressibility, μ ↔ effective density). `e`/`f` set the real part like `a`/`b`; `g`/`h` add loss like `c`/`d`.
MD!*/

/*!MD
# 2. Electromagnetic Interpretation
Mapping the column model onto a physical EM material (the model's native domain)

- `a`…`d` define the complex relative permittivity: real part `εr(f) = a·(f/fRef)^b`, imaginary part `ε'' = 17.98·σ/f` with `σ(f) = c·(f/fRef)^d` and `f` in GHz.
- `e`…`h` define the complex relative permeability the same way: `μr(f) = e·(f/fRef)^f`, imaginary part `μ'' = 17.98·σμ/f` with `σμ(f) = g·(f/fRef)^h`. μ defaults to 1+0j, so EM materials that omit `e`…`h` are unchanged (the magneto-dielectric TM branch is out of scope; μ is intended for TE / scalar use and for any μ = 1 material).
- Interface reflection uses the Fresnel coefficient from the wave impedance `Z = Z₀√(μ/ε)`; at normal incidence `R = (1 − √(ε/μ))/(1 + √(ε/μ))`, reducing to `(1 − √ε)/(1 + √ε)` when μ = 1, and the RT tool applies the angle-dependent form.
- The three EM interaction modes of <a href="cpp_api.html#ray_mesh_interact">ray_mesh_interact</a> are 0 (reflection — the Fresnel `abs(R)²`), 1 (transmission — the undeviated straight-through ray, governed by the dense→light/light→dense partition gate), and 2 (refraction — bends the ray per Snell using `(ε₁μ₁)/(ε₂μ₂)` and applies the full Fresnel transmission directly). Type 2 is the physically correct transmission path and does **not** use the partition gate; use it when correct dielectric refraction matters, and type 1 when an undeviated transmitted ray is sufficient.
- Built-in EM materials have `εr > 1`, `μ = 1`, use only parameters `a`…`att`, and set `fRef = 1 GHz`; the extended parameters (`e`…`h`, `m`, `res*`, `coi…`) are normally default for EM.
- See Electromagnetic Default Materials for the built-in library and Background and References for the standards (ITU-R P.2040, P.833, 3GPP TR 38.901).

## Electromagnetic Default Materials
Built-in material library used when no `materials_csv` is provided. Values follow Rec. ITU-R P.2040-3, Table 3, valid for 1–40 GHz (ground materials limited to 1–10 GHz). The built-in table defines only the parameters `a`, `b`, `c`, `d`, `att`; every other parameter is absent (consumers default it, so `μ = 1`) and `fRef = 1 GHz`. A scene using only built-in materials therefore exposes the keys `a`, `b`, `c`, `d`, `att` in `csv_prop`, each a vector with one entry per table row. `max fGHz` is the upper frequency for which each fit is considered valid.<br><br>

| Name                  | a     | b      | c       | d      | att  | max fGHz |
| --------------------- | ----: | -----: | ------: | -----: | ---: | -------: |
| air                   | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 100      |
| vacuum                | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 100      |
| textiles              | 1.5   | 0.0    | 5e-5    | 0.62   | 0.0  | 100      |
| plastic               | 2.44  | 0.0    | 2.33e-5 | 1.0    | 0.0  | 100      |
| ceramic               | 6.5   | 0.0    | 0.0023  | 1.32   | 0.0  | 100      |
| sea_water             | 80.0  | -0.25  | 4.0     | 0.58   | 0.0  | 100      |
| sea_ice               | 3.2   | -0.022 | 1.1     | 1.5    | 0.0  | 100      |
| water                 | 80.0  | -0.18  | 0.6     | 1.52   | 0.0  | 20       |
| water_ice             | 3.17  | -0.005 | 5.6e-5  | 1.7    | 0.0  | 20       |
| itu_concrete          | 5.24  | 0.0    | 0.0462  | 0.7822 | 0.0  | 100      |
| itu_brick             | 3.91  | 0.0    | 0.0238  | 0.16   | 0.0  | 40       |
| itu_plasterboard      | 2.73  | 0.0    | 0.0085  | 0.9395 | 0.0  | 100      |
| itu_wood              | 1.99  | 0.0    | 0.0047  | 1.0718 | 0.0  | 100      |
| itu_glass             | 6.31  | 0.0    | 0.0036  | 1.3394 | 0.0  | 100      |
| itu_ceiling_board     | 1.48  | 0.0    | 0.0011  | 1.075  | 0.0  | 100      |
| itu_chipboard         | 2.58  | 0.0    | 0.0217  | 0.78   | 0.0  | 100      |
| itu_plywood           | 2.71  | 0.0    | 0.33    | 0.0    | 0.0  | 40       |
| itu_marble            | 7.074 | 0.0    | 0.0055  | 0.9262 | 0.0  | 60       |
| itu_floorboard        | 3.66  | 0.0    | 0.0044  | 1.3515 | 0.0  | 100      |
| itu_metal             | 1.0   | 0.0    | 1.0e7   | 0.0    | 0.0  | 100      |
| itu_very_dry_ground   | 3.0   | 0.0    | 0.00015 | 2.52   | 0.0  | 10       |
| itu_medium_dry_ground | 15.0  | -0.1   | 0.035   | 1.63   | 0.0  | 10       |
| itu_wet_ground        | 30.0  | -0.4   | 0.15    | 1.3    | 0.0  | 10       |
| itu_vegetation        | 1.0   | 0.0    | 1.0e-4  | 1.1    | 0.0  | 100      |
| irr_glass             | 6.27  | 0.0    | 0.0043  | 1.1925 | 23.0 | 100      |

Notes:
- `air` is row 0 of the table — the transparent fallback used for unmatched materials  when `csv_strict = false`. `vacuum` is a separate, identical row (lossless free space).
- `itu_vegetation` follows Rec. ITU-R P.833-9, Figure 2.
- `irr_glass` (infrared-reflective glass) follows 3GPP TR 38.901 V17.0.0, Table 7.4.3-1 and is the
  only built-in material with a non-zero `att`.
MD!*/

/*!MD
# 3. Acoustic Interpretation
Reusing the column model to simulate acoustic propagation with a radio-wave ray tracer
- Acoustic waves at kHz frequencies share wavelengths with radio waves at GHz frequencies, so the EM material model can be reused for acoustic planning. The mapping is an analogy: it holds for interface reflection, bulk absorption, and the resonance / mass-law mechanisms below, and does not model phase or diffraction.
- Acoustic interactions use the scalar interaction types of <a href="cpp_api.html#ray_mesh_interact">ray_mesh_interact</a> (3 = reflection, 4 = transmission): a single TE-mode pressure coefficient with no refractive bending of the ray. There is no scalar refraction mode — EM refraction (type 2) has no scalar analog — so scalar transmission is always the undeviated straight-through path. Reflection (type 3) uses the Fresnel `abs(R)²` from the impedance `√(μ/ε)`, so the ε, μ pair sets reflection and room-side absorption. Transmission (type 4) shares the dense→light / light→dense gate with EM transmission (type 1): entering a lower-index medium (`Re(ε₁μ₁)` > `Re(ε₂μ₂)`) is pass-through, entering a higher-index medium subtracts the Fresnel `1 − abs(R)²`. The lumped `att` / `coi…` and in-medium `alpha` / `m` carry any added isolation in both cases.

## Wave mapping:
- `f_radio = f_acoustic × 874,636` (c_light / c_sound, c_sound ≈ 342.77 m/s).
- `fRef = 0.875 GHz ≡ 1 kHz` acoustic; fixed for every acoustic material.
- `resF` and `coiF` are absolute radio GHz: `f_acoustic[Hz] × 8.746e-4` (e.g. 100 Hz → 0.0875 GHz; 2 kHz → 1.75 GHz).
- Octave-band grid 16 … 16000 Hz; simulation results are air-normalized (ISO 9613-1 air absorption already removed).

## The two material families:
With permeability available, acoustic materials split into two calibration families. ε ↔ compressibility and μ ↔ effective density; the index is `n = √(εμ)` and the surface impedance is `Z = Z₀√(μ/ε)`.

- **Rigid reflectors** (concrete, glass, steel, brick, gypsum, wood panels): a single-parameter medium — `εr ≪ 1` (tiny `a`, ~1e-9 … 1e-3), `μ = 1`. The tiny ε gives near-total reflection (`abs(R) → 1`, low room-side absorption) and the body is effectively opaque. Through-material isolation is carried by `att` (baseline level) and `m` (mass-law slope), with `coiF`/`coiQ`/`coiA` for the coincidence dip of thin stiff panels. `σ = 0`.
- **Porous absorbers** (foam, mineral wool, fiberglass, carpet, curtains): a two-parameter medium — `εr ≈ 1` with light loss (pore air, near free-air compressibility) and `μr > 1` with strong loss (the frame-loaded, viscously damped effective density). Calibrated from the complex index and impedance directly (below); `att` = `alpha` = `m` = 0 — the absorber is carried by ε and μ alone.

## Parameter roles:
- `σ` (`c`, `d`) is 0 for rigid reflectors but **non-zero for porous** absorbers (it carries part of the bulk loss), so the old "σ held at 0 for all acoustic materials" rule no longer applies. Each remaining effect still has an independent knob:<br><br>
  | Acoustic effect                                       | Parameters                  | Notes                                                              |
  | ----------------------------------------------------- | --------------------------- | ------------------------------------------------------------------ |
  | Medium: surface reflection + bulk absorption          | `a`,`b`,`c`,`d`,`e`,`f`,`g`,`h` | two-parameter ε, μ; reflection from `√(μ/ε)`, bulk loss from `Im√(εμ)` |
  | Rigid-wall reflection (single parameter)              | `a` (tiny), `μ = 1`         | `ε ≪ 1` → near-total reflection, low room-side absorption          |
  | Resonant absorber (Helmholtz / membrane / MPP)        | `resF`, `resQ`, `resS`      | Lorentz pole in ε; peaked α and reflection feature at `resF`       |
  | Lumped panel transmission loss (thin partition)       | `att`, `attB`               | fixed dB on entry; thickness-independent                           |
  | Mass-law transmission (rigid partitions)              | `m`                         | log in freq & thickness; `m = 20` ⇒ +6 dB/oct, +6 dB/doubling      |
  | Coincidence dip (thin stiff panels)                   | `coiF`, `coiQ`, `coiA`      | Lorentzian on TL at `coiF`; negative `coiA` = dip                  |

- Room-side absorption `α = 1 − abs(R)²` (which sets reverberation) is set by the medium (ε, μ) via the reflection branch. The lumped `att`/`coi…` and in-medium `m`/`alpha` terms are reserved for rigid-partition transmission and resonators and remain decoupled from the porous medium calibration: tuning them never changes the reverberant field, and the porous ε, μ fit never changes the through-wall mass-law isolation of a separate rigid partition.

## Calibrating a porous absorber:
A porous layer is intrinsically a two-parameter medium (independent density and compressibility), so a single ε cannot match its reflection and its bulk loss at once — with one parameter the impedance `Z = Z₀/√ε` and the index `n = √ε` are locked together. Permeability breaks the lock: `Z = Z₀√(μ/ε)`, `n = √(εμ)`. Given the material's complex index `n(f) = k/k₀` and normalized surface impedance `z(f) = Zc/Z₀` (from a Delany-Bazley fit of the flow resistivity, or from measurement), the columns follow in closed form:

  `ε = n / z`  (compressibility)   `μ = n · z`  (density)<br><br>

Fit `ε'`, `ε''`, `μ'`, `μ''` over the working band as power laws and read off the columns: `ε' → (a, b)`, `ε'' → (c, d)`, `μ' → (e, f)`, `μ'' → (g, h)`. Worked example for mineral wool, flow resistivity ≈ 12000 Pa·s/m²:<br><br>

  | a    | b     | c      | d    | e    | f     | g     | h    |
  | ---: | ----: | -----: | ---: | ---: | ----: | ----: | ---: |
  | 1.16 | −0.04 | 0.0038 | 1.46 | 1.66 | −0.19 | 0.091 | 0.13 |

  with `fRef = 0.875`, `att = alpha = m = 0`. This matches both the surface reflection and the bulk attenuation against Delany-Bazley across ~250 Hz … 4 kHz simultaneously (the band the power-law fit covers well) — the thing a single ε cannot do. Note where the loss went: ε is nearly real (compressibility ≈ air), and μ carries the viscous loss (density), which is where dissipation physically lives. A single-ε fit was forced to cram that loss into `ε''`, which over-stated the surface reflection.

- Band edges: the power-law `μ''` runs hot above ~12 kHz (over-states the top-octave bulk loss); below ~100 Hz the Delany-Bazley source is itself extrapolation. Use the resonance term for non-monotonic features rather than forcing the power law past its range.

## Rigid-surface absorption (small residual):
For a nominally rigid surface with a small measured absorption α (painted concrete, glass), invert the single-parameter normal-incidence reflection with `μ = 1`:

  `a = ((1 − √(1−α)) / (1 + √(1−α)))²`  (small α: `a ≈ α²/16`)<br><br>
  | α at fRef | a       |
  | --------: | ------: |
  | 0.01      | 6.3e-6  |
  | 0.1       | 6.8e-4  |
  | 0.3       | 7.7e-3  |
  | 0.5       | 2.9e-2  |
  | 0.7       | 7.7e-2  |
  | 0.9       | 2.7e-1  |
  | 1.0       | 1.0     |

- This is the `ε ≪ 1` recipe and applies **only to rigid reflectors** (`μ = 1`), for tuning the small surface absorption of an otherwise reflective wall. Porous absorbers use the two-parameter ε, μ calibration above, not this table. Because `ε ≪ 1`, a rigid material is "optically rarer" than air; on the reflection path this still gives near-total reflection with no change of ray direction, and on the transmission path the dense→light gate makes the air→wall crossing pass-through, so isolation comes from `att` (and `m` for the slope), not from `a`.

## Layered and mounted materials (no air gap):
Stacked materials with no air between them (a porous absorber glued to a wall, carpet on a wooden floor, wood on concrete) are modeled as two watertight bodies whose shared faces are coincident (within the 1 mm interaction tolerance) with opposing normals. <a href="cpp_api.html#ray_mesh_interact">ray_mesh_interact</a> detects this as a material-to-material interface. The *reflection* branch computes the Fresnel coefficient from the two real materials' ε, μ directly (no air); the *transmission* branch uses the dense→light / light→dense gate (pass-through entering a lower-index medium, `1 − abs(R)²` entering a higher-index medium), partitioned from the lumped/in-medium terms as everywhere else. The wave cascade (entry → internal interface(s) → exit) is traced as separate ray segments by the ray tracer, which must split each interaction into its reflected and transmitted children.

- **Rigid on rigid** (wood on concrete): the mutual interface is gated by the index contrast; isolation is the sum of each entered layer's `att` plus the in-medium (`alpha`, `m`) losses of the two layers. Set `att` on at most one layer (see below) and let `m`/`alpha` carry the rest.
- The lumped `att` is applied once per material entry, including at an internal material-to-material interface (using the entered material's `att`). If both stacked layers carry an `att`, the loss is counted at each entry — set `att` on at most one layer of a glued stack to avoid double counting.
- Geometry requirement: the glued faces must be coincident within 1 mm with opposing normals, or the pair is treated as two air interfaces separated by a thin air layer, which changes the result.

## Mounted porous absorbers and the standing-wave limit:
A porous layer on a rigid backing (foam on concrete, the canonical broadband absorber) is the one configuration where the geometry-driven volumetric model has a known low-frequency error, and it is worth understanding before trusting the numbers there.

The physics: the rigid backing forces a velocity node at the wall, so below the quarter-wave frequency `f = c_sound / (4·thickness)` the porous material sits in a low-velocity region and dissipates little — real absorption rolls off hard toward zero. This is a standing-wave (phase) effect. A phaseless ray tracer cannot resolve it: tracing the ray into the layer, reflecting off the backing, and summing the return path over-states the low-frequency absorption, because the velocity-node suppression is invisible to it. The error is always an over-prediction of absorption, confined **below ~c/4d**, and it shrinks as the layer thickens (a thicker layer pushes `c/4d` down). Above `c/4d` the backing stops mattering, the layer is effectively anechoic, and the volumetric result is correct.

- `f = c/4d` ≈ **858 Hz** for 0.1 m, **1.7 kHz** for 0.05 m, **8.6 kHz** for 0.01 m. Trust the mounted absorption above this; treat below it as optimistic, and the sub-100 Hz bands as extrapolation regardless.
- **Energy safety:** a coherent sum of the internal bounce paths is not bounded by passivity — a truncated bounce series can return `abs(R)² > 1` (the layer would emit more than it received). The +3 dB "two paths at most in phase" intuition does not bound this, because at low frequency the true reflection already sits at the unity ceiling (`α → 0`), leaving no headroom for any overshoot, and the dropped higher-order bounces are exactly the terms that would resum it back to passive. Clamp the per-interaction reflected power to ≤ the incident power. For a thin layer at low frequency the clamp lands on total reflection, which is the physically correct answer there, so it costs nothing for thick absorbers and fixes the thin-layer case.
- **Low-frequency-exact mounted absorption:** replace the volumetric trace-through of the layer with a surface-impedance boundary on the front face. With `z = √(μ/ε)`, `n = √(εμ)`, `k = (2π·f / c_sound)·n`, and `cosθt = √(1 − (sinθ / n)²)` (Snell), the input impedance of the layer on a rigid backing is `Zs = −j·(z / cosθt)·cot(k·d·cosθt)` and the reflection is `R = (Zs·cosθ − 1) / (Zs·cosθ + 1)`. This is the analytic infinite-bounce resummation, passive by construction (`abs(R) ≤ 1`) and exact at the quarter-wave knee, but it requires tagging the face with its layer thickness and backing rather than reading the thickness from the mesh.
- Geometric low-frequency accuracy is fundamentally a wave-solver regime (below the Schroeder frequency the field is modal/diffraction-dominated). The ray model is the mid/high-frequency engine; the absorber's standing wave and the room's modes are the same kind of physics in the same band where the ray tracer is itself an approximation.

## Material classes:
- **Rigid reflectors** (concrete, glass, steel, brick, gypsum, wood panels): tiny `a` from impedance (~1e-9 … 1e-3), `b = 0`, `μ = 1`. The tiny `a` gives near-total reflection (`abs(R) → 1`, low room-side absorption). Through-material isolation is carried by `att` (baseline level) and `m` (mass-law slope); `coiF/coiQ/coiA` add a coincidence dip for thin stiff panels.
- **Porous absorbers** (foam, mineral wool, fiberglass, carpet, curtains): the two-parameter ε, μ via the closed form (`ε = n/z`, `μ = n·z`); `att` = `alpha` = `m` = 0. Model the layer at its installed thickness so the bulk loss accumulates over the real path; mind the mounted standing-wave limit above.
- **Empirical absorbers** (furniture, audience, people): measured α coefficients fitted as a porous medium (ε, μ); geometry sized to the effective absorption depth.
- **Resonant absorbers** (Helmholtz, membrane, micro-perforated): `resF/resQ/resS` for the peak, on top of a baseline reflector.

## Validity and limitations:
- Not captured: phase / interference (use Monte-Carlo phase post-processing), diffraction, and rough-surface scattering — handle these outside the material model.
- Mounted absorbers: the volumetric model over-predicts low-frequency absorption below the quarter-wave frequency `c/4d` of a layer on a rigid backing (see above). Bounded, shrinks with thickness, valid above `c/4d`; use the surface-impedance boundary for a low-frequency-exact result.
- LF floor: P.2040 is nominal above 100 MHz ≡ 114 Hz acoustic; the 16 / 31.5 / 63 Hz bands are extrapolation. Below the Schroeder frequency the field is modal and a wave solver is required for rigorous results.
- The smooth dispersion exponents (`b`, `d`, `f`, `h`, `alphaB`, `attB`) are single power laws over ~3 decades; use the resonance and coincidence terms for non-monotonic features rather than forcing a power-law fit.
- Geometry: model each material at its installed thickness as a watertight body; see the OBJ geometry guidance for object dimensions.
- The mass-law term sets only the frequency/thickness *slope* (`m = 20` → +6 dB/octave and +6 dB per thickness doubling); the absolute transmission-loss *level* of a rigid partition comes from `a` (the interface term), not from `m`. Because `mass(f, L)` scales with the in-medium path length `L = thickness / cosθ`, the modeled mass-law loss *increases* at oblique incidence, opposite to the real mass law (which falls as ≈ 20·log10 cosθ). Treat this as a known approximation.

## Acoustic Default Materials
- Built-in acoustic material library (under construction).
- Acoustic materials use the same parameters as the EM model with `fRef = 0.875 GHz`. Rigid rows keep `μ = 1` (`e = 1`, `f = g = h = 0`); porous rows use the full ε, μ pair. The table below is a work-in-progress template grouped by the material classes above; `resF`…`coiA` are 0 unless the row is a resonant absorber.<br><br>
  | Name                 | Class   | a       | b     | c      | d    | e    | f     | g     | h    | att | alpha | m  |
  | -------------------- | ------- | ------: | ----: | -----: | ---: | ---: | ----: | ----: | ---: | --: | ----: | -: |
  | concrete (0.1–0.6 m) | rigid   | 2.4e-9  | 0.0   | 0      | 0    | 1.0  | 0     | 0     | 0    | 0   | 0     | 20 |
  | porous_generic       | porous  | 1.16    | −0.04 | 0.0038 | 1.46 | 1.66 | −0.19 | 0.091 | 0.13 | 0   | 0     | 0  |
  | _resonant_ (TBD)     | resonant| …       | 0     | 0      | 0    | 1.0  | 0     | 0     | 0    | 0   | 0     | 0  |

- `porous_generic` is the calibrated mineral-wool example from the calibration section (flow resistivity ≈ 12000 Pa·s/m²); model it at its installed thickness. For the resonant row, set `resF`/`resQ`/`resS` for the peak on top of a baseline reflector.
- Note: in the concrete row the `m = 20` term supplies only the mass-law slope; the absolute transmission loss is carried by `a` (the interface term), not by `m`.
MD!*/

/*!MD
# 4. Principled BSDF
Visual material parameters read from the companion .mtl file

When an OBJ file references an `.mtl` library, the per-material Principled BSDF parameters can be returned as an `[n_mtl, 17]` matrix (`bsdf`). These describe visual appearance only and do not affect propagation. Each row corresponds to one entry of `mtl_names`; if no matching `.mtl` file is found, the matrix is empty. All color, alpha, and 0–1 parameters are clamped to [0, 1].<br><br>

| Index | Field                | Property                       | Range  | Default |
| :---: | -------------------- | ------------------------------ | :----: | :-----: |
| 0     | R                    | Base color, red                | 0 … 1  | 0.8     |
| 1     | G                    | Base color, green              | 0 … 1  | 0.8     |
| 2     | B                    | Base color, blue               | 0 … 1  | 0.8     |
| 3     | alpha                | Opacity                        | 0 … 1  | 1.0     |
| 4     | roughness            | Surface roughness              | 0 … 1  | 0.5     |
| 5     | metallic             | Metallic factor                | 0 … 1  | 0.0     |
| 6     | ior                  | Index of refraction            | ≥ 1    | 1.45    |
| 7     | specular             | Specular factor                | 0 … 1  | 0.5     |
| 8     | Re                   | Emission color, red            | 0 … 1  | 0.0     |
| 9     | Ge                   | Emission color, green          | 0 … 1  | 0.0     |
| 10    | Be                   | Emission color, blue           | 0 … 1  | 0.0     |
| 11    | sheen                | Sheen factor                   | 0 … 1  | 0.0     |
| 12    | clearcoat            | Clearcoat factor               | 0 … 1  | 0.0     |
| 13    | clearcoat_roughness  | Clearcoat roughness            | 0 … 1  | 0.0     |
| 14    | anisotropic          | Anisotropy factor              | 0 … 1  | 0.0     |
| 15    | anisotropic_rotation | Anisotropy rotation            | 0 … 1  | 0.0     |
| 16    | transmission         | Transmission factor            | 0 … 1  | 0.0     |

## Mapping from .mtl keywords:
| .mtl keyword | BSDF field             | Notes                                            |
| ------------ | ---------------------- | ------------------------------------------------ |
| `Kd`         | R, G, B                | Base color                                       |
| `Ke`         | Re, Ge, Be             | Emission color                                   |
| `Ka`         | metallic               | —                                                |
| `Pm`         | metallic               | Overrides `Ka` when present                      |
| `Ks`         | specular               | First component only                             |
| `d`          | alpha                  | —                                                |
| `Ni`         | ior                    | —                                                |
| `Ns`         | roughness              | Converted: `roughness = 1 − sqrt(Ns · 0.001)`    |
| `Pr`         | roughness              | Overrides `Ns` when present                      |
| `Ps`         | sheen                  | —                                                |
| `Pc`         | clearcoat              | —                                                |
| `Pcr`        | clearcoat_roughness    | —                                                |
| `aniso`      | anisotropic            | —                                                |
| `anisor`     | anisotropic_rotation   | —                                                |
| `Tf`         | transmission           | First component only                             |
MD!*/

/*!MD
# 5. Background and References
Sources for the material model and its parameters

Primary recommendations:
- <a target="_blank" rel="noopener noreferrer" href="https://www.itu.int/rec/R-REC-P.2040">Rec. ITU-R P.2040</a>
  — defines the (a, b, c, d) permittivity / conductivity model and the Fresnel coefficients; source of the
  built-in material table (Table 3).
- <a target="_blank" rel="noopener noreferrer" href="https://www.itu.int/rec/R-REC-P.833">Rec. ITU-R P.833</a>
  — vegetation attenuation (source of `itu_vegetation`).
- 3GPP TR 38.901 V17.0.0, Table 7.4.3-1 — material penetration losses (source of `irr_glass` and the `att` parameter).

Base EM model (reflection and σ-loss):
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Fresnel_equations">Fresnel equations</a> (interface reflection / transmission from ε, μ)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Relative_permittivity">Relative permittivity (a, b)</a>
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Permeability_(electromagnetism)">Relative permeability (e, f, g, h)</a> (decouples wave impedance from refractive index)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity">Electrical resistivity and conductivity (c, d)</a>
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Dielectric_loss">Dielectric loss / loss tangent</a> (in-medium σ loss)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Attenuation">Attenuation</a> (alpha, linear in path length)

Acoustic mechanism mapping (the analogy the parameters approximate):
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Acoustic_impedance">Acoustic impedance</a> (basis for deriving ε, μ from the index and impedance)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Sound_absorption">Delany-Bazley / porous absorption</a> (source of the n, z model behind `ε = n/z`, `μ = n·z`)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Acoustic_transmission">Acoustic transmission</a> (mass-law transmission loss (m) and the coincidence effect)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Soundproofing">Soundproofing</a> (mass law, coincidence, partition behavior)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Sound_transmission_class">Sound transmission class</a> (single-number TL rating context)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Absorption_(acoustics)">Absorption (acoustics)</a> (porous absorption: ε, μ, alpha)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Helmholtz_resonance">Helmholtz resonance</a> (resonant absorbers: resF/resQ/resS, coiF/coiQ/coiA)
MD!*/