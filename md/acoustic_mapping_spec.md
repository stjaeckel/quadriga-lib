# Acoustic Material Mapping Spec

Map acoustic material properties to ITU-R P.2040-style EM parameters for a radio-wave ray tracer. One material per chat: paste this spec, the current 9-parameter CSV row, the air-normalized simulation results, and the acoustic targets. Interpret the results and propose parameter adjustments to converge on the targets.

## 1. Wave Mapping

- `f_radio = f_acoustic × 874,636` (c_light / c_sound; uses c_sound ≈ 342.77 m/s for consistency with the converter script)
- Octave-band centres: `16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000` Hz (11 bands)
- Path-tracking anchor: 125 Hz acoustic ≡ 0.1093 GHz radio
- All simulation results are **air-normalized**: ISO 9613-1 air absorption is already subtracted; only the material effect remains.

## 2. Parameter Model

CSV row, 9 columns:

| Idx | Symbol   | Property                       | Units | Default |
|----:|:---------|:-------------------------------|:-----:|:-------:|
| 0   | `a`      | ε_r at fRef                    | —     | 1.0     |
| 1   | `b`      | ε_r frequency exponent         | —     | 0       |
| 2   | `c`      | σ at fRef                      | S/m   | 0       |
| 3   | `d`      | σ frequency exponent           | —     | 0       |
| 4   | `att`    | entry penetration loss at fRef | dB    | 0       |
| 5   | `attB`   | att frequency exponent         | —     | 0       |
| 6   | `alpha`  | in-body absorption at fRef     | dB/m  | 0       |
| 7   | `alphaB` | alpha frequency exponent       | —     | 0       |
| 8   | `fRef`   | reference frequency            | GHz   | 0.875   |

Power laws (f in GHz):

```
ε_r(f) = a · (f/fRef)^b
σ(f)   = c · (f/fRef)^d                       [S/m]
att(f) = att · (f/fRef)^attB                  [dB]   — applied once on entering a body
α(f)   = alpha · (f/fRef)^alphaB              [dB/m] — integrated over in-body path length only
```

`fRef = 0.875 GHz` (≡ 1 kHz acoustic) is fixed for every material.

## 3. The Two-Knob Decoupling

**`σ = 0` for all acoustic materials** (set `c = d = 0` and keep them zero throughout fitting). With σ = 0:

- Real ε via (`a`, `b`) sets interface reflection — **and only that**.
- `att` and `α` set absorption / transmission loss — **and only that**.

The two knobs become orthogonal. This is the central design choice and the reason the extended parameter set exists: P.2040's single complex-ε formulation forces reflection and absorption to be tuned by the same knob, and accepts the resulting compromise (RMS |R| error ≈ 0.13–0.21 for porous/empirical materials in the original converter). The (att, α) extension removes that compromise.

## 4. Interface Reflection from Real ε

At normal incidence, room → material:

```
R   = (1 − √ε) / (1 + √ε)
|R|² ≈ 1 − 4√ε              (for ε ≪ 1)
α_ceiling = 1 − |R|² ≈ 4√ε   — the maximum room-side absorption achievable from the interface alone
```

Inversion — given a target absorption α at fRef, solve for `a`:

```
exact:   a = ((1 − √(1−α)) / (1 + √(1−α)))²
small α: a ≈ α² / 16
```

| α at fRef | a       |
|----------:|--------:|
| 0.001     | 6.3e-8  |
| 0.01      | 6.3e-6  |
| 0.1       | 6.8e-4  |
| 0.3       | 7.7e-3  |
| 0.5       | 2.9e-2  |
| 0.7       | 7.7e-2  |
| 0.9       | 2.7e-1  |
| 1.0       | 1.0     |

Increasing `a` (toward 1) → smaller |R| → larger α_ceiling.

Oblique incidence is handled by the RT tool's angle-dependent Fresnel. Note: for ε ≪ 1 the critical angle sin θ_c = √ε is very small, so the textbook Fresnel result is TIR for almost any non-normal angle. The tool's actual angle response should be read from the simulation results, not predicted from first principles.

## 5. Simulation Input Format

Each chat will contain:

**(a) Material identification and thickness:**
```
Material: <name>
Thickness: <m>
```

**(b) Current parameter row:**
```
a    b    c    d    att    attB    alpha    alphaB    fRef
<values>
```

**(c) Two result tables (Reflection and Transmission), columns:**
```
distance    angle    125 Hz [anchor]    16 Hz    31.5 Hz    63 Hz    250 Hz    500 Hz    1 kHz    2 kHz    4 kHz    8 kHz    16 kHz
```
- `distance`: source–receiver distance [m]
- `angle`: incidence angle on the test surface [degrees]; ~0 = normal, ~90 = grazing
- The 125 Hz column is the path-anchor and is listed first; remaining 10 columns are the rest of the octave grid in ascending frequency
- Multiple rows correspond to different geometries / angles

**Reflection table**: air-normalized received energy at the reflection-path receiver (dB; absolute scale depends on test geometry). What matters for tuning is its relative dependence on parameters: doubles when |R|² doubles, vanishes when ε → 1.

**Transmission table**: through-material transmission loss [dB], air-normalized. For σ = 0 and att = α = 0, this equals 10·log₁₀(|T|²) at the given incidence angle.

**(d) Acoustic targets**, per octave band:
```
α_target(f):   from ISO 354 / manufacturer data / audience tables
TL_target(f):  (optional) from ISO 10140 if available
```

## 6. Material Classes — Starting Points

### Class 1 — Rigid reflector
*Concrete, glass, brick, gypsum, steel, dense wood.*

| Property      | Initial value                                  |
|---------------|------------------------------------------------|
| `a`           | (Z_air / Z_material)² ≈ 1e-9 to 1e-11          |
| `b`           | 0                                              |
| `att`, `alpha`| 0                                              |

Expected behavior: α_eff(f) ≈ 4√a, flat and very small. Transmission loss large, dominated by interface reflection. No iteration usually needed once `a` is set from impedance.

### Class 2 — Porous absorber
*Acoustic foam, mineral wool, fibreglass, carpet, curtains.*

| Property | Initial value                                                                                  |
|----------|------------------------------------------------------------------------------------------------|
| `a`      | From α_target(1 kHz) via §4 inversion                                                          |
| `b`      | `log(a_4k / a_1k) / log(4)` — typically > 0 (more HF absorption)                               |
| `alpha`  | `≥ 20 / thickness_m` — large enough that all entered energy is absorbed in one through-pass   |
| `alphaB` | 0 (only matters if `alpha` is too small to saturate)                                           |
| `att`    | 0                                                                                              |

Once `alpha × thickness ≥ ~20 dB`, α_eff is governed by ε alone — tune α_eff via (a, b), not via alpha.

### Class 3 — Empirical absorber
*Furniture, audience, mattresses, complex objects.*

Same recipe as Class 2 but use measured α coefficients directly. Geometry: use the effective-absorption-depth dimensions from the README's geometry table (e.g. seated person 50×50×100 cm, couch ≥25 cm depth).

## 7. Iteration Decision Rules

When sim α_eff or TL diverges from target, identify the symptom and adjust the responsible parameter:

| Symptom                                            | Adjustment                                |
|----------------------------------------------------|-------------------------------------------|
| α_eff too low across all bands                     | increase `a` (toward 1)                   |
| α_eff too high across all bands                    | decrease `a`                              |
| α_eff frequency slope too steep up                 | decrease `b`                              |
| α_eff frequency slope too flat / wrong sign        | increase `b` (or change sign)             |
| TL too low at fRef                                 | increase `alpha` (or `att` if appropriate)|
| TL too high at fRef                                | decrease `alpha`                          |
| TL frequency tilt wrong                            | adjust `alphaB`                           |
| Anchor band (125 Hz) clearly off                   | priority fix — drives path culling        |

Magnitude of step: for `a`, halve or double until α_eff lands within an octave of target, then bisect. For `b`, start from |b| ≤ 0.5 and bisect; values of |b| > 2 across this band range usually mean the material is non-power-law and won't fit cleanly (see §9). For `alpha`, increases above the saturation point (`alpha × thickness ≥ ~20 dB`) have no effect on α_eff.

Convergence criteria:
- α_eff within max(±10%, ±0.05) of target at every band
- TL within ±2 dB of target where target exists
- 125 Hz anchor must match within ±5% — this is the path-culling band

## 8. `att` vs `alpha` Degeneracy

At a single fixed thickness, `att` (fixed dB) and `alpha × thickness` are interchangeable — both just add dB to the through-energy. Use **`alpha` as the primary absorption knob**: it scales physically with material thickness and with incidence angle (grazing rays traverse a longer in-body path). Reserve `att` for:

- Thin partitions modelled as a thin sheet where a thickness-independent lumped TL is wanted.
- Cases requiring angle-independent loss (`alpha` at grazing integrates a longer path; `att` doesn't).

Do not tune both for the same material unless multi-thickness or multi-angle reference data is available to separate them.

## 9. Limitations

- **Low-frequency extrapolation.** 16, 31.5, 63 Hz acoustic map to 14, 28, 55 MHz radio, below P.2040's nominal 100 MHz validity floor (100 MHz ≡ 114 Hz acoustic). 16 Hz exists primarily to suppress clamp-extrapolation artifacts downstream; treat 16/31.5/63 Hz results as extrapolation.
- **Power-law range.** A single power law per property spans three decades. Real absorption curves are often non-monotonic (mid peak, HF plateau, resonant dips). Resonant, membrane, and Helmholtz absorbers will not fit a monotonic power law — **flag such materials rather than forcing a fit**.
- **HF data scarcity.** Above ~5 kHz, measured material data is sparse; targets at 8 kHz and 16 kHz are often extrapolated from lower-band measurements.
- **Phase, diffraction, scattering.** Phase is not modelled (use Monte-Carlo phase post-processing). Diffraction and rough-surface scattering are not captured by the material model and should be handled separately.
- **Geometry.** Material is modelled at its installed thickness as a watertight body. The RT tool must handle ε < 1 correctly (including TIR scenarios). See the README's Geometry Modeling Guidelines for object dimensions.
