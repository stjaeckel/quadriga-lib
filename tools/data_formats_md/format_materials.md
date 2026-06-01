/*!SECTION
Material Properties
SECTION!*/

/*!SECTION_DESC
Material properties describe how surfaces and bodies in a triangular scene mesh
interact with a propagating wave. Each triangle carries a material whose
parameters are stored as one row of a property matrix (`mtl_prop`). The model is
formulated for electromagnetic propagation — complex relative permittivity ε and
conductivity σ following Rec. ITU-R P.2040 — and reuses the same parameters for an
acoustic interpretation (mass law, coincidence, resonant absorption).<br><br>

A material row has between 1 and 16 columns. Only the first column (`a`) is
required; every other column may be omitted and is then substituted with its
default. The parameters split into three physical roles:
- **Interface reflection** (`a`, `b`, `c`, `d`, `resF`, `resQ`, `resS`) — set the complex
  permittivity ε, which fixes the Fresnel reflection coefficient and therefore the
  room-side absorption `1 − abs(R)²`. Applied once per surface hit, independent of
  path length.
- **Interface transmission** (`att`, `attB`, `coiF`, `coiQ`, `coiA`) — a lumped
  through-surface loss in dB, applied once per transmission, independent of path
  length.
- **In-medium attenuation** (`c`, `d` via ε, `alpha`, `alphaB`, `m`) — loss accumulated
  along the path traversed inside a body; depends on the in-medium distance.<br><br>

Both domains use the same 16 columns but populate them very differently: EM
materials have εr > 1 with σ-based loss, whereas acoustic materials map to
εr ≪ 1 with σ held at 0 and loss carried by the att/alpha and mass-law terms.
See the Electromagnetic and Acoustic interpretation sections below.

A separate set of Principled BSDF parameters describes the *visual* appearance of
each material (color, roughness, metallic, etc.). These are read from the companion
`.mtl` file, do not affect propagation, and are documented in the Principled BSDF
section below.<br><br>

See related functions that produce or consume material properties:
- C++ OBJ reader (<a href="cpp_api.html#obj_file_read">obj_file_read</a>)
- C++ ray / mesh interaction (<a href="cpp_api.html#ray_mesh_interact">ray_mesh_interact</a>)
- C++ diffraction gain (<a href="cpp_api.html#calc_diffraction_gain">calc_diffraction_gain</a>)
- C++ Mitsuba scene export (<a href="cpp_api.html#mitsuba_xml_file_write">mitsuba_xml_file_write</a>)
SECTION_DESC!*/

/*!MD
# 1. Material Model
Complex permittivity, conductivity, and loss parameters with their frequency dependence

## Parameter columns:
Each material maps to the following columns. `Index` is the 0-based column in
`mtl_prop`; columns absent from the matrix are substituted with the listed default.<br><br>

| Index | Symbol | Property                              | Units      | Default |
| :---: | :----: | ------------------------------------ | :--------: | :-----: |
| 0     | a      | εr at fRef                          | —          | 1.0     |
| 1     | b      | Frequency exponent for εr           | —          | 0       |
| 2     | c      | σ at fRef                            | S/m        | 0       |
| 3     | d      | Frequency exponent for σ             | —          | 0       |
| 4     | att    | Penetration loss at fRef             | dB         | 0       |
| 5     | attB   | Frequency exponent for att           | —          | 0       |
| 6     | alpha  | In-medium absorption at fRef         | dB/m       | 0       |
| 7     | alphaB | Frequency exponent for alpha         | —          | 0       |
| 8     | fRef   | Reference frequency                  | GHz        | 1.0     |
| 9     | m      | Mass-law transmission slope          | dB/decade  | 0       |
| 10    | resF   | Permittivity resonance frequency     | GHz        | 0       |
| 11    | resQ   | Permittivity resonance quality factor| —          | 0       |
| 12    | resS   | Permittivity resonance strength      | —          | 0       |
| 13    | coiF   | Coincidence frequency                | GHz        | 0       |
| 14    | coiQ   | Coincidence quality factor           | —          | 0       |
| 15    | coiA   | Coincidence loss amplitude           | dB         | 0       |

## Frequency laws:
`f` is given in GHz; `f/fRef` is the relative frequency. `resF` and `coiF` are
absolute GHz.<br><br>

| Parameter  | Formula                                                      | Unit   | Meaning                                |
| ---------- | ------------------------------------------------------------ | ------ | -------------------------------------- |
| ε(f)       | `a·(f/fRef)^b + resS·resF² / (resF² − f² − i·(resF/resQ)·f)` | —      | relative permittivity (complex)        |
| σ(f)       | `c·(f/fRef)^d`                                               | [S/m]  | conductivity                           |
| att(f)     | `att·(f/fRef)^attB + coiA / (1 + (coiQ·(f − coiF)/coiF)²)`   | [dB]   | per-interface transmission loss        |
| α(f)       | `alpha·(f/fRef)^alphaB`                                      | [dB/m] | in-medium loss × in-medium path length |
| mass(f, L) | `max(0, m·log10((f/fRef)·L))`                                | [dB]   | in-medium, L = path length in meters   |

## Mechanisms:
- **Permittivity resonance** (`resF`, `resQ`, `resS`): a Lorentz pole that adds a peak to
  absorption (acoustic α) and a feature to reflection near `resF`; `resQ` sets sharpness
  (higher = narrower). Active only when `resF > 0` and `resS ≠ 0`. Models resonant
  dielectrics / frequency-selective media (EM) and Helmholtz / membrane absorbers (acoustic).
- **Coincidence term** (`coiF`, `coiQ`, `coiA`): a Lorentzian added to the transmission loss
  at `coiF`. Negative `coiA` produces a transmission dip (acoustic coincidence / pass-band);
  positive `coiA` produces a stop-band. Total loss is clamped to ≥ 0. Active only when
  `coiF > 0` and `coiA ≠ 0`.
- **Mass-law term** (`m`): a transmission loss that is logarithmic in both frequency and
  in-medium path length. `m = 20` reproduces the acoustic mass law (+6 dB/octave and +6 dB
  per thickness doubling). Default 0 (EM through-loss is the linear `alpha` term). The
  imaginary sign of the ε resonance follows the library's loss convention (consistent with σ).
MD!*/

/*!MD
# 2. Electromagnetic Interpretation
Mapping the column model onto a physical EM material (the model's native domain)

- `a`…`d` define the complex relative permittivity: real part `εr(f) = a·(f/fRef)^b`,
  imaginary part `ε'' = 17.98·σ/f` with `σ(f) = c·(f/fRef)^d` and `f` in GHz.
- Interface reflection uses the Fresnel coefficient from ε; at normal incidence
  `R = (1 − √ε)/(1 + √ε)`, and the RT tool applies the angle-dependent form.
- Built-in EM materials have `εr > 1`, use only columns `a`…`att`, and set `fRef = 1 GHz`;
  the extended columns (`m`, `res*`, `coi*`) are normally 0 for EM.
- See Electromagnetic Default Materials for the built-in library and Background and
  References for the standards (ITU-R P.2040, P.833, 3GPP TR 38.901).

## Electromagnetic Default Materials
Built-in material library used when no `materials_csv` is provided

Values follow Rec. ITU-R P.2040-3, Table 3, valid for 1–40 GHz (ground materials limited
to 1–10 GHz). All built-in materials use only columns 0–4 (`a`, `b`, `c`, `d`, `att`); every
extended parameter is zero and `fRef = 1 GHz`. A scene using only built-in materials therefore
yields a 5-column `mtl_prop` (4 columns if no material sets `att`; only `irr_glass` does).
`max fGHz` is the upper frequency for which each fit is considered valid.<br><br>

| Name                  | a     | b      | c       | d      | att  | max fGHz |
| --------------------- | ----: | -----: | ------: | -----: | ---: | -------: |
| vacuum / air          | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 100      |
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
- `vacuum` and `air` are identical (lossless free space) and both resolve to the default material.
- `itu_vegetation` follows Rec. ITU-R P.833-9, Figure 2.
- `irr_glass` (infrared-reflective glass) follows 3GPP TR 38.901 V17.0.0, Table 7.4.3-1 and is the
  only built-in material with a non-zero `att`.
MD!*/

/*!MD
# 3. Acoustic Interpretation
Reusing the column model to simulate acoustic propagation with a radio-wave ray tracer

- Acoustic waves at kHz frequencies share wavelengths with radio waves at GHz
  frequencies, so the EM material model can be reused for acoustic planning. The
  mapping is an analogy: it holds for interface reflection, bulk absorption, and
  the resonance / mass-law mechanisms below, and does not model phase or diffraction.

## Wave mapping:
- `f_radio = f_acoustic × 874,636` (c_light / c_sound, c_sound ≈ 342.77 m/s).
- `fRef = 0.875 GHz ≡ 1 kHz` acoustic; fixed for every acoustic material.
- `resF` and `coiF` are absolute radio GHz: `f_acoustic[Hz] × 8.746e-4`
  (e.g. 100 Hz → 0.0875 GHz; 2 kHz → 1.75 GHz).
- Octave-band grid 16 … 16000 Hz; simulation results are air-normalized
  (ISO 9613-1 air absorption already removed).

## Decoupled knobs:
- `σ` is held at 0 (`c = d = 0`) for all acoustic materials, so each effect has an independent knob:<br><br>
  | Acoustic effect                                   | Columns              | Notes                                                              |
  | ------------------------------------------------- | -------------------- | ------------------------------------------------------------------ |
  | Surface reflection / room-side absorption `1−abs(R)²` | `a`, `b`         | `a` sets abs(R) at fRef; `b` sets the frequency trend              |
  | Resonant absorber (Helmholtz / membrane / MPP)    | `resF`, `resQ`, `resS` | Lorentz pole; peaked α and reflection feature at `resF`          |
  | Porous / bulk absorption (through-loss)           | `alpha`, `alphaB`    | dB/m over in-body path; `alpha·thickness ≥ ~20 dB` saturates       |
  | Lumped panel transmission loss (thin partition)   | `att`, `attB`        | fixed dB on entry; thickness-independent                           |
  | Mass-law transmission (rigid partitions)          | `m`                  | log in freq & thickness; `m = 20` ⇒ +6 dB/oct, +6 dB/doubling      |
  | Coincidence dip (thin stiff panels)               | `coiF`, `coiQ`, `coiA` | Lorentzian on TL at `coiF`; negative `coiA` = dip                |
  | Conductivity                                      | `c`, `d`             | unused for acoustic — keep at 0                                    |

- Room-side absorption `α = 1 − abs(R)²` (which sets reverberation) is governed by
  `(a, b)` alone. Through-material transmission loss is governed by `att/alpha/m/coi*`
  on top of the interface term. Tuning the transmission knobs never changes the
  reverberant field — an absorption miss is always an `(a, b)` fix.

## Reflection from ε:
- At normal incidence `R = (1 − √ε)/(1 + √ε)`, and for `ε ≪ 1`,
  `α_ceiling = 1 − abs(R)² ≈ 4√ε`. Invert for `a` at fRef:

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

- Because `ε ≪ 1`, acoustic materials are "optically rarer" than air: the critical
  angle `sinθ_c = √ε` is tiny, so the RT tool must handle `ε < 1`, including total
  internal reflection.

## Material classes:
- **Rigid reflectors** (concrete, glass, steel, brick, gypsum, wood panels): tiny `a`
  from impedance (~1e-9 … 1e-3), `b = 0`. Transmission via `m` (mass law); add
  `coiF/coiQ/coiA` for thin stiff panels with a coincidence dip.
- **Porous absorbers** (foam, mineral wool, fiberglass, carpet, curtains): `a` from the
  α target, `b > 0` (more HF absorption), `alpha ≥ 20/thickness` to saturate.
- **Empirical absorbers** (furniture, audience, people): measured α coefficients via
  `(a, b)`; geometry sized to the effective absorption depth.
- **Resonant absorbers** (Helmholtz, membrane, micro-perforated): `resF/resQ/resS` for
  the peak, on top of a baseline `(a, b)`.

## Validity and limitations:
- Not captured: phase / interference (use Monte-Carlo phase post-processing),
  diffraction, and rough-surface scattering — handle these outside the material model.
- LF floor: P.2040 is nominal above 100 MHz ≡ 114 Hz acoustic; the 16 / 31.5 / 63 Hz
  bands are extrapolation.
- The smooth dispersion exponents (`b`, `alphaB`, `attB`) are single power laws over
  ~3 decades; use the resonance and coincidence terms for non-monotonic features
  rather than forcing a power-law fit.
- Geometry: model each material at its installed thickness as a watertight body; see
  the OBJ geometry guidance for object dimensions.

## Acoustic Default Materials
- Built-in acoustic material library (under construction)
- Acoustic materials use the same columns as the EM model with `fRef = 0.875 GHz`,
  `σ = 0`, and `εr ≪ 1`. The table below is a work-in-progress template grouped by
  the material classes above. Values are illustrative only.<br><br>
  | Name              | Class     | a       | b   | alpha | alphaB | att | attB | m  | resF | resQ | resS | coiF | coiQ | coiA |
  | ----------------- | --------- | ------: | --: | ----: | -----: | --: | ---: | -: | ---: | ---: | ---: | ---: | ---: | ---: |
  | concrete (0.1–0.6 m) | rigid  | 2.4e-9  | 0.0 | 0     | 0      | 0   | 0    | 20 | 0    | 0    | 0    | 0    | 0    | 0    |
  | _porous_ (TBD)    | porous    | …       | >0  | ≥20/t | 0      | 0   | 0    | 0  | 0    | 0    | 0    | 0    | 0    | 0    |
  | _resonant_ (TBD)  | resonant  | …       | 0   | 0     | 0      | 0   | 0    | 0  | f_r  | Q    | S    | 0    | 0    | 0    |

- Note: the concrete row replaces the earlier `att/attB/alpha` mass-law approximation with a single `m = 20`.
MD!*/

/*!MD
# 4. Principled BSDF
Visual material parameters read from the companion .mtl file

When an OBJ file references an `.mtl` library, the per-material Principled BSDF parameters can be
returned as an `[n_mtl, 17]` matrix (`bsdf`). These describe visual appearance only and do not
affect propagation. Each row corresponds to one entry of `mtl_names`; if no matching `.mtl` file
is found, the matrix is empty. All color, alpha, and 0–1 parameters are clamped to [0, 1].<br><br>

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
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Fresnel_equations">Fresnel equations</a> (interface reflection / transmission from ε)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Relative_permittivity">Relative permittivity (a, b)</a>
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity">Electrical resistivity and conductivity (c, d)</a>
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Dielectric_loss">Dielectric loss / loss tangent</a> (in-medium σ loss)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Attenuation">Attenuation</a> (alpha, linear in path length)

Acoustic mechanism mapping (the analogy the parameters approximate):
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Acoustic_impedance">Acoustic impedance</a> (basis for deriving `a` from an impedance ratio)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Acoustic_transmission">Acoustic transmission</a> (mass-law transmission loss (m) and the coincidence effect)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Soundproofing">Soundproofing</a> (mass law, coincidence, partition behavior)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Sound_transmission_class">Sound transmission class</a> (single-number TL rating context)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Absorption_(acoustics)">Absorption (acoustics)</a> (porous absorption: alpha, alphaB)
- <a target="_blank" rel="noopener noreferrer" href="https://en.wikipedia.org/wiki/Helmholtz_resonance">Helmholtz resonance</a> (resonant absorbers: resF/resQ/resS, coiF/coiQ/coiA)
MD!*/
