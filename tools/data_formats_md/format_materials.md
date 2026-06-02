/*!SECTION
Material Properties
SECTION!*/

/*!SECTION_DESC
Materials are stored in a name-keyed table: each triangle references a material by row index (`csv_ind` or `mtl_ind`), the row names are returned in `csv_names` / `mtl_names`, and the per-material parameters are returned in `csv_prop` — a map from parameter name to a vector with one entry per material row (so a value is read as `csv_prop["a"][csv_ind[face]]`). The model is formulated for electromagnetic propagation — complex relative permittivity ε and conductivity σ following Rec. ITU-R P.2040 — and reuses the same parameters for an acoustic interpretation (mass law, coincidence, resonant absorption).<br><br>

The table is schema-blind: the only required column is `name` (the join key for `.obj` materials); every other column is an optional numeric parameter. A parameter the table does not define is simply absent from `csv_prop`, and downstream consumers substitute its default (listed below). Empty cells parse as 0. The recognized parameters split into three physical roles:
- **Interface reflection** (`a`, `b`, `c`, `d`, `resF`, `resQ`, `resS`) — set the complex permittivity ε, which fixes the Fresnel reflection coefficient `R` at every surface crossing. ε governs the reflected power `abs(R)²` and the complementary room-side absorption `1 − abs(R)²`. Applied once per surface hit, independent of path length.
- **Interface transmission** (`att`, `attB`, `coiF`, `coiQ`, `coiA`) — the through-surface isolation in dB, applied once when entering a material (the front-side air→material or material→material crossing). Independent of path length and not applied on exit. The transmitted power is *not* the Fresnel `1 − abs(R)²`; see the partition note below.
- **In-medium attenuation** (`c`, `d` via ε; `alpha`, `alphaB`; `m`) — loss accumulated along the path traversed inside a body. The σ-driven loss from `Im(ε)` and the explicit `alpha` term sum, and `m` adds a mass-law slope; all scale with the in-medium distance.

**Transmission partition.** Reflection and transmission are computed independently, not as energy-complementary halves of a single Fresnel split. ε (`a`, `b`, `resF…`) sets reflection and room-side absorption; the lumped (`att`, `coi*`) and in-medium (`alpha`, `m`) terms set transmission and isolation. In EM mode the transmitted field is *additionally* scaled by the Fresnel `1 − abs(R)²` on a light→dense crossing (and clamped to pass-through on dense→light, to avoid spurious amplification). In scalar/acoustic mode it is not: the crossing is angle-independent pass-through scaled by the isolation terms alone. This removes the spurious critical-angle total reflection that an ε < 1 Fresnel transmission would otherwise impose on oblique acoustic paths. Either way, ε does not set the transmission *level* — the lumped and in-medium terms do.

Both domains use the same parameter set but populate it very differently: EM materials have εr > 1 with σ-based loss, whereas acoustic materials map to εr ≪ 1 with σ held at 0 and loss carried by the att/alpha and mass-law terms. See the Electromagnetic and Acoustic interpretation sections below.
See related functions that produce or consume material properties:
- C++ OBJ reader (<a href="cpp_api.html#obj_file_read">obj_file_read</a>)
- C++ ray / mesh interaction (<a href="cpp_api.html#ray_mesh_interact">ray_mesh_interact</a>)
- C++ diffraction gain (<a href="cpp_api.html#calc_diffraction_gain">calc_diffraction_gain</a>)
- C++ Mitsuba scene export (<a href="cpp_api.html#mitsuba_xml_file_write">mitsuba_xml_file_write</a>)
SECTION_DESC!*/

/*!MD
# 1. Material Model
Complex permittivity, conductivity, and loss parameters with their frequency dependence

## Parameters:
Each parameter is one numeric column of the material table and one key of `csv_prop`. `Key` is the CSV column name (and the `csv_prop` key); a parameter absent from the table is substituted with the listed default by downstream consumers. Only the `name` column is mandatory, and column order does not matter.<br><br>

| Key    | Property                              | Units      | Default |
| :----: | ------------------------------------- | :--------: | :-----: |
| a      | εr at fRef                            | —          | 1.0     |
| b      | Frequency exponent for εr             | —          | 0       |
| c      | σ at fRef                             | S/m        | 0       |
| d      | Frequency exponent for σ              | —          | 0       |
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
absolute GHz.<br><br>

| Parameter  | Formula                                                                    | Unit   | Meaning                                |
| ---------- | -------------------------------------------------------------------------- | ------ | -------------------------------------- |
| ε(f)       | `a·(f/fRef)^b − i·17.98·σ/f + resS·resF² / (resF² − f² + i·(resF/resQ)·f)` | —      | relative permittivity (complex)            |
| σ(f)       | `c·(f/fRef)^d`                                                             | [S/m]  | conductivity (enters ε as the −i term)     |
| att(f)     | `att·(f/fRef)^attB + coiA / (1 + (coiQ·(f − coiF)/coiF)²)`                 | [dB]   | lumped loss added once on entering         |
| α(f)       | `alpha·(f/fRef)^alphaB`                                                    | [dB/m] | explicit in-medium loss × path length      |
| mass(f, L) | `max(0, m·log10((f/fRef)·L))`                                              | [dB]   | mass-law slope, L = in-medium path length  |

- Loss appears as a negative imaginary part of ε (Rec. ITU-R P.2040-1, eq. 9b). Both the σ term and the resonance are written so that `c > 0` and `resS > 0` add loss. The total in-medium attenuation over a path of length `L` is `8.686·L/Δ` (from `Im(ε)`, where Δ is the σ-derived attenuation length) plus `α(f)·L` plus `mass(f, L)`, summed; `att` is separate and is not path-dependent.

## Mechanisms:
- **Permittivity resonance** (`resF`, `resQ`, `resS`): a Lorentz pole that adds a peak to absorption (acoustic α) and a feature to reflection near `resF`; `resQ` sets sharpness (higher = narrower). Active only when `resF > 0` and `resS ≠ 0`. Models resonant dielectrics / frequency-selective media (EM) and Helmholtz / membrane absorbers (acoustic).
- **Coincidence term** (`coiF`, `coiQ`, `coiA`): a Lorentzian added to the transmission loss at `coiF`. Negative `coiA` produces a transmission dip (acoustic coincidence / pass-band); positive `coiA` produces a stop-band. Total loss is clamped to ≥ 0. Active only when `coiF > 0` and `coiA ≠ 0`.
- **Mass-law term** (`m`): a transmission loss that is logarithmic in both frequency and in-medium path length. `m = 20` reproduces the acoustic mass law (+6 dB/octave and +6 dB per thickness doubling). Default 0 (EM through-loss is the linear `alpha` term). The resonance denominator uses `+i·(resF/resQ)·f` so that `resS > 0` adds loss under the library's negative-imaginary loss convention (consistent with σ).
MD!*/

/*!MD
# 2. Electromagnetic Interpretation
Mapping the column model onto a physical EM material (the model's native domain)

- `a`…`d` define the complex relative permittivity: real part `εr(f) = a·(f/fRef)^b`, imaginary part `ε'' = 17.98·σ/f` with `σ(f) = c·(f/fRef)^d` and `f` in GHz.
- Interface reflection uses the Fresnel coefficient from ε; at normal incidence `R = (1 − √ε)/(1 + √ε)`, and the RT tool applies the angle-dependent form.
- Built-in EM materials have `εr > 1`, use only parameters `a`…`att`, and set `fRef = 1 GHz`; the extended parameters (`m`, `res*`, `coi*`) are normally 0 for EM.
- See Electromagnetic Default Materials for the built-in library and Background and References for the standards (ITU-R P.2040, P.833, 3GPP TR 38.901).

## Electromagnetic Default Materials
Built-in material library used when no `materials_csv` is provided. Values follow Rec. ITU-R P.2040-3, Table 3, valid for 1–40 GHz (ground materials limited to 1–10 GHz). The built-in table defines only the parameters `a`, `b`, `c`, `d`, `att`; every other parameter is absent (consumers default it) and `fRef = 1 GHz`. A scene using only built-in materials therefore exposes the keys `a`, `b`, `c`, `d`, `att` in `csv_prop`, each a vector with one entry per table row. `max fGHz` is the upper frequency for which each fit is considered valid.<br><br>


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
- Acoustic interactions use the scalar interaction types of <a href="cpp_api.html#ray_mesh_interact">ray_mesh_interact</a> (3 = reflection, 4 = transmission): a single TE-mode pressure coefficient with no refractive bending of the ray. Reflection (type 3) uses the Fresnel `abs(R)²` from ε, so `(a, b)` set reflection and room-side absorption. Transmission (type 4) is partitioned from reflection: the crossing is forced to pure pass-through (`abs(R)²` is not subtracted) and the isolation is carried entirely by `att` / `coi*` / `alpha` / `m`. Because this is angle-independent, ε < 1 no longer produces a critical-angle dropout that would kill oblique transmission.

## Wave mapping:
- `f_radio = f_acoustic × 874,636` (c_light / c_sound, c_sound ≈ 342.77 m/s).
- `fRef = 0.875 GHz ≡ 1 kHz` acoustic; fixed for every acoustic material.
- `resF` and `coiF` are absolute radio GHz: `f_acoustic[Hz] × 8.746e-4` (e.g. 100 Hz → 0.0875 GHz; 2 kHz → 1.75 GHz).
- Octave-band grid 16 … 16000 Hz; simulation results are air-normalized (ISO 9613-1 air absorption already removed).

## Decoupled knobs:
- `σ` is held at 0 (`c = d = 0`) for all acoustic materials, so each effect has an independent knob:<br><br>
  | Acoustic effect                                   | Parameters              | Notes                                                              |
  | ------------------------------------------------- | ----------------------- | ------------------------------------------------------------------ |
  | Surface reflection / room-side absorption `1−abs(R)²` | `a`, `b`            | `a` sets abs(R) at fRef; `b` sets the frequency trend              |
  | Resonant absorber (Helmholtz / membrane / MPP)    | `resF`, `resQ`, `resS`  | Lorentz pole; peaked α and reflection feature at `resF`          |
  | Porous / bulk absorption (through-loss)           | `alpha`, `alphaB`       | dB/m over in-body path; `alpha·thickness ≥ ~20 dB` saturates       |
  | Lumped panel transmission loss (thin partition)   | `att`, `attB`           | fixed dB on entry; thickness-independent                           |
  | Mass-law transmission (rigid partitions)          | `m`                     | log in freq & thickness; `m = 20` ⇒ +6 dB/oct, +6 dB/doubling      |
  | Coincidence dip (thin stiff panels)               | `coiF`, `coiQ`, `coiA`  | Lorentzian on TL at `coiF`; negative `coiA` = dip                |
  | Conductivity                                      | `c`, `d`                | unused for acoustic — keep at 0                                    |

- Room-side absorption `α = 1 − abs(R)²` (which sets reverberation) is governed by `(a, b)` alone, via the reflection branch. Transmission is fully partitioned from this: on the scalar path the through-material isolation is carried entirely by `att`, `attB`, the coincidence term, `alpha`, and `m`; `(a, b)` set the level of reflection/absorption but do **not** set the transmission level. The decoupling is bidirectional — tuning `att`/`alpha`/`m`/`coi*` never changes the reverberant field, and tuning `(a, b)` never changes the through-wall isolation. An absorption miss is always an `(a, b)` fix; an isolation miss is always an `att`/`m`/`coi*`/`alpha` fix.

## Reflection from ε:
- At normal incidence `R = (1 − √ε)/(1 + √ε)`, and for `ε ≪ 1`, `α_ceiling = 1 − abs(R)² ≈ 4√ε`. Invert for `a` at fRef:

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

- Because `ε ≪ 1`, acoustic materials are "optically rarer" than air. On the *reflection* path this still gives near-total reflection (`abs(R) → 1`, low room-side absorption), with no change of ray direction. On the *transmission* path the partition deliberately ignores the Fresnel coefficient: rather than letting `ε ≪ 1` impose a critical-angle cutoff (`sinθ_c = √ε`) that would block oblique transmission, the crossing is pure pass-through and the isolation is the `att`/`coi*`/`alpha`/`m` budget. A slab embedded in air therefore pays no Fresnel `1 − abs(R)²` at entry or exit on the acoustic path; set the through-wall isolation with `att` (and `m` for the slope), not with `a`. `a` is calibrated from the absorption target alone.

## Layered materials (no air gap):
Stacked materials with no air between them (a porous absorber glued to a wall, carpet on a wooden floor, wood on concrete) are modeled as two watertight bodies whose shared faces are coincident (within the 1 mm interaction tolerance) with opposing normals. [[ray_mesh_interact]] detects this as a material-to-material interface. The *reflection* branch computes the Fresnel coefficient from the two real materials' ε directly (no air); the *transmission* branch is pass-through scaled by the entered layer's isolation (`att`), partitioned from ε as everywhere else. The wave cascade (entry → internal interface(s) → exit) is traced as separate ray segments by the ray tracer, which must split each interaction into its reflected and transmitted children.

- **Porous absorber on a rigid backing** (carpet on wood, foam on concrete): the air→absorber face sets the surface reflection from the absorber's `(a, b)`; the absorber→backing interface reflects strongly (the backing's `a ≈ 0` gives `abs(R) → 1`); the porous `alpha` is accumulated on both the inbound and the reflected outbound traversal. Absorption is set by the reflection path and is unaffected by what transmits into the backing.
- **Rigid on rigid** (wood on concrete): the mutual interface transmits as pass-through (no Fresnel loss on the acoustic path); isolation is the sum of each entered layer's `att` plus the in-medium (`alpha`, `m`) losses of the two layers. Set `att` on at most one layer (see below) and let `m`/`alpha` carry the rest.
- The lumped `att` is applied once per material entry, including at an internal material-to-material interface (using the entered material's `att`). If both stacked layers carry an `att`, the loss is counted at each entry — set `att` on at most one layer of a glued stack to avoid double counting.
- Geometry requirement: the glued faces must be coincident within 1 mm with opposing normals, or the pair is treated as two air interfaces separated by a thin air layer, which changes the result.

## Material classes:
- **Rigid reflectors** (concrete, glass, steel, brick, gypsum, wood panels): tiny `a` from impedance (~1e-9 … 1e-3), `b = 0`. The tiny `a` gives near-total reflection (`abs(R) → 1`, low room-side absorption). Through-material isolation is *not* set by `a`: on the acoustic path the crossing is pass-through, so isolation is carried by `att` (the baseline level) and `m` (the mass-law frequency/thickness slope); `coiF/coiQ/coiA` add a coincidence dip for thin stiff panels.
- **Porous absorbers** (foam, mineral wool, fiberglass, carpet, curtains): `a` from the α target, `b > 0` (more HF absorption), `alpha ≥ 20/thickness` to saturate.
- **Empirical absorbers** (furniture, audience, people): measured α coefficients via `(a, b)`; geometry sized to the effective absorption depth.
- **Resonant absorbers** (Helmholtz, membrane, micro-perforated): `resF/resQ/resS` for the peak, on top of a baseline `(a, b)`.

## Validity and limitations:
- Not captured: phase / interference (use Monte-Carlo phase post-processing), diffraction, and rough-surface scattering — handle these outside the material model.
- LF floor: P.2040 is nominal above 100 MHz ≡ 114 Hz acoustic; the 16 / 31.5 / 63 Hz bands are extrapolation.
- The smooth dispersion exponents (`b`, `alphaB`, `attB`) are single power laws over ~3 decades; use the resonance and coincidence terms for non-monotonic features rather than forcing a power-law fit.
- Geometry: model each material at its installed thickness as a watertight body; see the OBJ geometry guidance for object dimensions.
- The mass-law term sets only the frequency/thickness *slope* (`m = 20` → +6 dB/octave and +6 dB per thickness doubling); the absolute transmission-loss *level* comes from `a` (the interface term), not from `m`. Because `mass(f, L)` scales with the in-medium path length `L = thickness / cosθ`, the modeled mass-law loss *increases* at oblique incidence, opposite to the real mass law (which falls as ≈ 20·log10 cosθ). Treat this as a known approximation.

## Acoustic Default Materials
- Built-in acoustic material library (under construction)
- Acoustic materials use the same parameters as the EM model with `fRef = 0.875 GHz`, `σ = 0`, and `εr ≪ 1`. The table below is a work-in-progress template grouped by the material classes above. Values are illustrative only.<br><br>
  | Name              | Class     | a       | b   | alpha | alphaB | att | attB | m  | resF | resQ | resS | coiF | coiQ | coiA |
  | ----------------- | --------- | ------: | --: | ----: | -----: | --: | ---: | -: | ---: | ---: | ---: | ---: | ---: | ---: |
  | concrete (0.1–0.6 m) | rigid  | 2.4e-9  | 0.0 | 0     | 0      | 0   | 0    | 20 | 0    | 0    | 0    | 0    | 0    | 0    |
  | _porous_ (TBD)    | porous    | …       | >0  | ≥20/t | 0      | 0   | 0    | 0  | 0    | 0    | 0    | 0    | 0    | 0    |
  | _resonant_ (TBD)  | resonant  | …       | 0   | 0     | 0      | 0   | 0    | 0  | f_r  | Q    | S    | 0    | 0    | 0    |

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
