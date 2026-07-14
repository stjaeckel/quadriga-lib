# The quadriga-lib Material Model and Ray-State Machine

*A companion document to `ray_mesh_interact.cpp`*

This document describes the physical material model of quadriga-lib and the functions that apply it: `ray_mesh_interact`, which evaluates a single ray-surface interaction; the material helpers `medium_gain`, `interface_gain`, and `refractive_index`; and `ray_state_update`, the batched state machine that tracks each ray's inside/outside history, carries its accumulated in-medium path, and overlays an analytic thin-slab (Fabry-Pérot) resolution. The model is formulated for electromagnetic (EM) propagation following Rec. ITU-R P.2040 [1] and is deliberately constructed so that the same parameter set, reinterpreted, simulates acoustic propagation with the same ray-tracing engine. All formulas are given exactly as implemented; the source code is the normative reference and this document is its explanation.

**Contents**

1. [Introduction and architecture](#1-introduction-and-architecture)
2. [The material model](#2-the-material-model)
3. [Interface interaction](#3-interface-interaction)
4. [Per-hit processing in `ray_mesh_interact`](#4-per-hit-processing-in-ray_mesh_interact)
5. [Electromagnetic interpretation](#5-electromagnetic-interpretation)
6. [Acoustic interpretation](#6-acoustic-interpretation)
7. [The ray-state machine `ray_state_update`](#7-the-ray-state-machine-ray_state_update)
8. [Dispatch tables](#8-dispatch-tables)
9. [Validation](#9-validation)
10. [References](#10-references)

---

## 1. Introduction and architecture

A shooting-and-bouncing-rays (SBR) tracer decomposes a propagation path into segments separated by surface interactions. quadriga-lib splits the physics of those interactions across two functions with sharply separated responsibilities:

- **`ray_mesh_interact`** is *stateless per hit*. Given a ray segment (origin, destination), the mesh, the material table, and the first and second intersection face indices (FBS, SBS), it computes the intersection points, classifies the hit topologically, evaluates the Fresnel interface coefficients, updates the ray geometry (reflected, transmitted, or refracted direction, plus the beam tube), and emits a per-interaction gain and polarization transfer matrix. It does not know whether the ray is currently traveling inside a material — it sees one segment at a time — and it charges no in-medium attenuation at all (Section 4.4).

- **`ray_state_update`** is the *per-ray state machine*. It carries three small state words per ray across interactions (current medium, previous medium, a one-slot transition buffer) plus an accumulated in-medium path length, decides what each hit *means* in context — entry, exit, embedded face to ignore, illegal state to terminate — corrects the per-interaction gain, direction, and polarization accordingly, and owns **all** in-medium loss. On top of the state logic it adds a closed-form thin-slab resolution: when a ray crosses a parallel slab whose internal multiple reflections still carry significant energy, the first-order interaction is multiplied by the Airy factor $S$ so that one coefficient captures the entire internal bounce series [4], instead of relying on the tracer to follow every internal reflection as a separate ray.

The tracer calls both functions once per interaction and per physical pass: a **reflection pass** produces the reflected child ray, and a **transmission/refraction pass** produces the forward child ray. Six interaction types select the physics:

| `interaction_type` | Meaning             | Geometry   | Field model        |
|:------------------:|---------------------|------------|--------------------|
| 0                  | EM reflection       | reflection | full polarization  |
| 1                  | EM transmission     | undeviated | full polarization  |
| 2                  | EM refraction       | Snell bent | full polarization  |
| 3                  | scalar reflection   | reflection | scalar (pressure)  |
| 4                  | scalar transmission | undeviated | scalar (pressure)  |
| 5                  | scalar refraction   | Snell bent | scalar (pressure)  |

Types 0–2 are the electromagnetic modes with a $2 \times 2$ complex polarization (Jones) transfer matrix. Types 3–5 are the scalar modes used for acoustics: a single complex pressure coefficient and TE-only Fresnel physics. Types 2 and 5 bend the ray according to Snell's law (the physically correct refracted path); types 1 and 4 keep the ray undeviated, which is the standard approximation for through-wall building penetration where the two refractions of a flat slab cancel and only a small parallel offset is ignored. Total internal reflection applies uniformly to all six types (Section 3.2). All six types are handled by `ray_state_update`; the reflection pass covers types 0 and 3, the transmission/refraction pass covers 1, 2, 4, and 5.

Conventions used throughout the code and this document:

- Frequencies are passed to the public API in Hz and converted once to GHz ($f = f_\mathrm{Hz} \cdot 10^{-9}$); every material formula below takes $f$ in GHz.
- Material indices are **1-based**; index 0 means "no material" (air/vacuum) and is always valid. The `Material` constructor performs the single point of index translation (`row = idx − 1`).
- Two length constants must be kept distinct. `colocation_dist` $= 1\,\mathrm{mm}$ is the coincident-face tolerance: two faces closer than this along the ray are treated as a single interface (an M2M contact, overlap, or edge), and it also lower-bounds the outgoing segment length. The **child-ray relaunch offset** is *not* 1 mm — it is a few `float` ULP at the interaction-point coordinate magnitude ($\approx 8 \cdot 2^{-23}\cdot\max(|x|,|y|,|z|,1)$), just large enough to survive the single-precision store of `origN` so the relaunched ray does not re-hit the face it just left.
- The incidence angle $\theta$ reported as `fbs_angleN` is the **signed grazing angle**: the angle between the ray and the face plane, $\theta = \arccos(\hat{u} \cdot \hat{n}) - \pi/2$, where $\hat{u}$ is the unit propagation direction and $\hat{n}$ the face normal. $\theta = \pm\pi/2$ is normal incidence, $\theta \to 0$ is grazing, and **negative values mean the back side of the face is illuminated** (the ray arrives from inside the body that owns the face). The cosine of the conventional incidence angle (from the normal) is recovered as $\cos\theta_i = |\cos(\theta + \pi/2)| = |\sin\theta|$.

---

## 2. The material model

### 2.1 The parameter table

Materials are stored in a name-keyed table: each mesh triangle references a material by a 1-based row index (`mtl_ind`), and the per-material parameters arrive as a map from parameter name to a vector with one entry per material (`mtl_prop`, the `csv_prop` output of `obj_file_read`). The table is *schema-blind and sparse*: any column may be absent, in which case every material takes that parameter's default. Inside the code, the `MaterialCols` helper resolves the map to raw column pointers once per call and validates it; the `Material` struct then materializes one row (or the air default for index 0) with the following fields:

| Key    | Symbol            | Property                                  | Units     | Default |
|:------:|:-----------------:|-------------------------------------------|:---------:|:-------:|
| `fRef` | $f_\mathrm{ref}$  | Reference frequency                       | GHz       | 1.0     |
| `a`    | —                 | $\varepsilon_r$ at $f_\mathrm{ref}$       | —         | 1.0     |
| `b`    | —                 | Frequency exponent for $\varepsilon_r$    | —         | 0       |
| `c`    | —                 | Conductivity $\sigma$ at $f_\mathrm{ref}$ | S/m       | 0       |
| `d`    | —                 | Frequency exponent for $\sigma$           | —         | 0       |
| `e`    | —                 | $\mu_r$ at $f_\mathrm{ref}$               | —         | 1.0     |
| `f`    | —                 | Frequency exponent for $\mu_r$            | —         | 0       |
| `g`    | —                 | Magnetic loss $\sigma_\mu$ at $f_\mathrm{ref}$ | —    | 0       |
| `h`    | —                 | Frequency exponent for $\sigma_\mu$       | —         | 0       |
| `att`  | —                 | Lumped penetration loss at $f_\mathrm{ref}$ | dB      | 0       |
| `attB` | —                 | Frequency exponent for `att`              | —         | 0       |
| `alpha`| $\alpha$          | In-medium absorption at $f_\mathrm{ref}$  | dB/m      | 0       |
| `alphaB`| —                | Frequency exponent for $\alpha$           | —         | 0       |
| `m`    | $m$               | Mass-law transmission slope               | dB/decade | 0       |
| `resF` | $f_\mathrm{res}$  | Permittivity resonance frequency          | GHz       | 0       |
| `resQ` | $Q_\mathrm{res}$  | Resonance quality factor                  | —         | 0       |
| `resS` | $S_\mathrm{res}$  | Resonance strength                        | —         | 0       |
| `coiF` | $f_\mathrm{coi}$  | Coincidence frequency                     | GHz       | 0       |
| `coiQ` | $Q_\mathrm{coi}$  | Coincidence quality factor                | —         | 0       |
| `coiA` | $A_\mathrm{coi}$  | Coincidence loss amplitude                | dB        | 0       |
| `tf`   | —                 | Transmission factor at $f_\mathrm{ref}$   | —         | 0       |
| `tfB`  | —                 | Frequency exponent for `tf`               | —         | 0       |

The parameters split into three physical roles, which the rest of this document treats separately:

1. **Interface reflection** (`a`–`h`, `res*`, `tf`): set the complex permittivity and permeability that fix the Fresnel coefficients at every surface crossing. Applied once per hit, independent of path length (Section 3).
2. **Interface transmission** (`att`, `attB`, `coi*`): a lumped through-surface isolation in dB applied once when *entering* a material, independent of path length and not applied on exit (Section 2.4).
3. **In-medium attenuation** (`c`/`d` and `g`/`h` via the loss tangent, `alpha`, `m`): loss accumulated along the path traveled inside a body, scaling with the in-medium distance (Section 2.3).

### 2.2 Constitutive frequency laws

With $f$ in GHz and $f_r = f / f_\mathrm{ref}$, the complex relative permittivity and permeability implemented in `Material::eta` and `Material::mu` are

$$
\varepsilon(f) = a \, f_r^{\,b} \;-\; j\,\frac{17.98\,\sigma(f)}{f},
\qquad \sigma(f) = c \, f_r^{\,d},
$$

$$
\mu(f) = e \, f_r^{\,f} \;-\; j\,\frac{17.98\,\sigma_\mu(f)}{f},
\qquad \sigma_\mu(f) = g \, f_r^{\,h}.
$$

The constant $17.98 = 1/(2\pi \varepsilon_0) \cdot 10^{-9}\,$GHz·m/S converts a conductivity in S/m into the imaginary permittivity at a frequency in GHz; this is the loss convention of Rec. ITU-R P.2040, eq. (9b) [1], with the library's $e^{+j\omega t}$ sign choice making loss a *negative* imaginary part. The magnetic columns mirror the dielectric ones exactly, so $\mu$ defaults to $1 + 0j$ and a table that omits `e`–`h` yields the permittivity-only model with $\mu = 1$.

A Lorentz resonance pole (`Material::eta_resonance`) can be added to the permittivity:

$$
\varepsilon_\mathrm{res}(f) =
\frac{S_\mathrm{res}\, f_\mathrm{res}^2}
     {f_\mathrm{res}^2 - f^2 + j\,(f_\mathrm{res}/Q_\mathrm{res})\,f},
$$

active only when $f_\mathrm{res} > 0$, $Q_\mathrm{res} > 0$, and $S_\mathrm{res} \neq 0$. The $+j$ in the denominator makes $S_\mathrm{res} > 0$ *add* loss under the negative-imaginary convention. The pole is applied to the **interface permittivity only** — it sharpens or dips the Fresnel reflection near $f_\mathrm{res}$ — and is deliberately excluded from the in-medium propagation phase (Sections 2.3 and 7.7): a strong pole can drive $\mathrm{Re}\,\varepsilon < 0$, which is meaningful for a reflection coefficient but not for a propagation constant of the bulk path. In the EM domain this models resonant dielectrics and frequency-selective surfaces; in the acoustic domain, Helmholtz and membrane absorbers (Section 6).

Derived quantities used throughout: the refractive index $n = \sqrt{\varepsilon\mu}$, the normalized admittance $Y = \sqrt{\varepsilon/\mu}$ (the code's `z`), the index ratio $(\varepsilon_1\mu_1)/(\varepsilon_2\mu_2)$ for Snell geometry, and the loss tangent of the *product* $\varepsilon\mu$ for bulk attenuation — so the magnetic loss columns `g`/`h` feed in-medium attenuation exactly as the conductivity columns `c`/`d` do. The public helper **`refractive_index`** exposes the real part of the base index, $n = \mathrm{Re}\big(\sqrt{\varepsilon(f)\,\mu(f)}\big)$, with the resonance pole *excluded* (it is an interface, not a bulk, effect) and with air (`iM = 0`) returning exactly $1$. This is the same real index used by the state machine to bend the refracted geometry and to accumulate excess phase.

### 2.3 In-medium propagation loss (`Material::medium_gain`)

The linear power gain of a path inside a homogeneous medium is $G_\mathrm{med} = 10^{-L_\mathrm{dB}/10}$ with

$$
L_\mathrm{dB} = \underbrace{\frac{8.686\,\ell_\mathrm{r}}{\Delta}}_{\text{dielectric}}
 \;+\; \underbrace{\ell_\mathrm{r}\,\alpha\, f_r^{\,\alpha_B}}_{\text{power-law}}
 \;+\; \underbrace{L_\mathrm{mass}}_{\text{mass law}} .
$$

The function takes **two path lengths**: a *refracted* in-medium length $\ell_\mathrm{r}$ (the true slant path the wave travels through the body) with its incidence cosine $\cos\theta_t$, and a *geometric* thickness path $\ell_\mathrm{g}$ (the free-space-referenced traversal) with its incidence cosine $\cos\theta_i$. When only one distance is supplied, the geometric pair collapses onto the refracted pair ($\ell_\mathrm{g} = \ell_\mathrm{r}$, $\cos\theta_i = \cos\theta_t$) and the model reduces to the single-distance form. Splitting the two is what lets the state machine feed the genuinely refracted path to the dielectric loss while feeding the geometric thickness to the mass law.

The three terms, exactly as implemented:

**Dielectric term.** From the complex product $\varepsilon\mu$ (resonance excluded), with $\tan\delta = \mathrm{Im}(\varepsilon\mu)/\mathrm{Re}(\varepsilon\mu)$ and $\cos\delta = 1/\sqrt{1 + \tan^2\delta}$, the attenuation distance is

$$
\Delta = \sqrt{\frac{2\cos\delta}{1-\cos\delta}}\;
         \frac{0.0477135}{f\,\sqrt{\mathrm{Re}(\varepsilon\mu)}} \quad [\mathrm{m}],
$$

which is the attenuation-distance formula of Rec. ITU-R P.2040 §2.2 [1] generalized from $\varepsilon$ to $\varepsilon\mu$. The constant $0.0477135 = c_0/(2\pi) \cdot 10^{-9}$ m·GHz is the inverse free-space wavenumber at 1 GHz, and $8.686 = 20/\ln 10$ converts nepers to dB. The dielectric loss scales with the *refracted* length $\ell_\mathrm{r}$. It reduces to the standard ITU result for $\mu = 1$ and vanishes for a lossless medium ($\tan\delta = 0 \Rightarrow \Delta \to \infty$).

**Power-law term.** An explicit absorption $\alpha\, f_r^{\,\alpha_B}$ in dB/m for excess loss not captured by the conductivity model (foliage, scattering media; in acoustics, calibrated bulk absorption). It too scales with the refracted length $\ell_\mathrm{r}$.

**Mass-law term.** Active only when $m > 0$ and $\ell_\mathrm{g} > 1.5\,\mathrm{mm}$ (paths shorter than the threshold `mass_min_path = 0.0015` are exempt, which keeps the short relaunch/bookkeeping segments of Section 8 out of the logarithm):

$$
L_\mathrm{mass} = \max\!\Big(0,\; m \,\log_{10}\!\big(f_r \cdot \ell_\mathrm{g} \cos^2\theta_i\big)\Big).
$$

Unlike the other two terms, the mass law uses the *geometric* thickness path $\ell_\mathrm{g}$ and the *geometric* incidence cosine $\cos\theta_i$. The argument $\ell_\mathrm{g}\cos^2\theta_i$ deserves a comment. For a slab of thickness $t$ traversed at incidence cosine $\cos\theta_i$ the geometric traversal is $\ell_\mathrm{g} = t/\cos\theta_i$, so $\ell_\mathrm{g}\cos^2\theta_i = t\cos\theta_i$: the modeled loss *decreases* at oblique incidence as $m\log_{10}\cos\theta_i$, matching the field-incidence behavior of the acoustic mass law ($\approx 20\log_{10}\cos\theta_i$ for $m = 20$) [7]. When the incidence angle is unknown, $\cos\theta_i$ defaults to 1 and the argument is the bare path. With $m = 20$ the term reproduces the classic mass law: +6 dB per octave and +6 dB per doubling of thickness; the outer $\max(0,\cdot)$ clamps the law below $f_r\,\ell_\mathrm{g}\cos^2\theta_i = 1$ where the logarithm would turn into gain.

Because the mass term is logarithmic in distance, $G_\mathrm{med}$ is *not* multiplicative in $\ell$ for mass-law materials: $G_\mathrm{med}(2\ell) \neq G_\mathrm{med}(\ell)^2$. The dielectric and power-law terms are linear in $\ell_\mathrm{r}$ and compose exactly. Section 7.7 discusses the (small, energy-safe) consequence for the thin-slab factor.

The public `medium_gain` API takes a single distance and evaluates the model with $\ell_\mathrm{g} = \ell_\mathrm{r}$; the penetration-loss columns (`att`, `attB`) are not used, as they describe thin-slab transmission loss rather than propagation through a finite-thickness medium.

### 2.4 Lumped interface loss (`Material::interface_gain`)

Entering a material applies a one-time, path-independent isolation

$$
G_\mathrm{if}(f) = 10^{-L_\mathrm{if}/10},
\qquad
L_\mathrm{if} = \mathrm{att}\, f_r^{\,\mathrm{attB}}
 \;+\; \frac{A_\mathrm{coi}}{1 + x^2},
\qquad
x = Q_\mathrm{coi}\,\frac{f - f_\mathrm{coi}}{f_\mathrm{coi}},
$$

with the coincidence Lorentzian active only when $f_\mathrm{coi} > 0$ and $A_\mathrm{coi} \neq 0$. A *negative* $A_\mathrm{coi}$ carves a transmission **dip** into the isolation — the acoustic coincidence effect of thin stiff panels [7] — and a positive one a stop-band. If the combined $L_\mathrm{if}$ comes out negative (a deep dip exceeding the baseline), the gain is clamped to 1: the interface never amplifies. `att` itself follows the lumped penetration losses of 3GPP TR 38.901, Table 7.4.3-1 [3].

The lumped loss is applied by `interact_with` to the transmission and refraction coefficients of the *entered* material (Section 3.4) — once per material entry, including entries at internal material-to-material interfaces. It is never applied on exit and never accumulates with path length. For a glued stack of two materials this means each entered layer's `att` counts once; give the lumped loss to at most one layer of a stack to avoid double counting.

### 2.5 The transmission factor `tf`

Some surfaces split energy between reflection and transmission in a way no passive $\varepsilon,\mu$ pair reproduces (e.g., a perforated panel that is acoustically hard yet leaks). The transmission factor redistributes the Fresnel energy split while conserving the total:

$$
\mathrm{tf}(f) = \mathrm{clamp}\big(\mathrm{tf}\cdot f_r^{\,\mathrm{tfB}},\, [-1, 1]\big),
$$

$$
R_\mathrm{eff} =
\begin{cases}
R_0\,(1 - \mathrm{tf}), & \mathrm{tf} \ge 0 \quad \text{(shift energy toward transmission)}\\[2pt]
R_0 + (1 - R_0)\,(-\mathrm{tf}), & \mathrm{tf} < 0 \quad \text{(shift energy toward reflection)}
\end{cases}
$$

with $R_0 \in [0,1]$ the physical Fresnel power reflectance (the single-sided building block `Material::apply_tf`; $R_0$ is clamped first to guard against resonance overshoot). $\mathrm{tf} = +1$ makes the surface fully transparent, $\mathrm{tf} = -1$ a perfect mirror, $\mathrm{tf} = 0$ leaves Fresnel untouched. The factor modifies magnitudes only; the Fresnel phases are preserved (Section 3.4).

At a two-medium interface the factor is combined symmetrically from both faces (`Material::apply_tf_pair`). With $\mathrm{tf}^+ = \max(\mathrm{tf}, 0)$ and $\mathrm{tf}^- = \max(-\mathrm{tf}, 0)$ on each side $A, B$,

$$
R_\mathrm{leak} = R_0\,(1 - \mathrm{tf}_A^+)(1 - \mathrm{tf}_B^+),
\qquad
R_\mathrm{eff} = R_\mathrm{leak} + (1 - R_\mathrm{leak})\,\max(\mathrm{tf}_A^-, \mathrm{tf}_B^-).
$$

This reduces to `apply_tf` at an air boundary (the other side's $\mathrm{tf} = 0$), stays in $[0,1]$, and gives $R_\mathrm{eff} = 1$ if either face has $\mathrm{tf} = -1$. The single-owner `apply_tf` is retained only *inside* the thin-slab factor, where each mirror has exactly one solid side (Section 7.7). Using the symmetric pair at interfaces is what keeps both reflection and transmission ports of a slab energy-complementary — including the EM reflection port, which carries the factor as well (Section 3.4).

### 2.6 Table validation and defaults

`MaterialCols` enforces, on every call that receives a material map:

- All non-empty columns have the same length $n_\mathrm{mtl}$ (sparse maps are legal; *ragged* maps are not).
- Physical sanity: $f_\mathrm{ref}, a, e > 0$ strictly; $c, g, \mathrm{att}, \alpha, m, f_\mathrm{res}, Q_\mathrm{res}, f_\mathrm{coi}, Q_\mathrm{coi} \ge 0$ (a negative loss-like value would be gain and is rejected as corrupt input rather than clamped). The reported material index is 1-based.
- Material indices — face indices, and in `ray_state_update` also the masked state words — are range-checked against $n_\mathrm{mtl}$ *before* any parallel region, so invalid input throws `std::invalid_argument` instead of terminating the process from inside an OpenMP loop.

Absent columns resolve to null pointers and every consumer substitutes the defaults of the table in Section 2.1; index 0 always yields the default-constructed air material ($\varepsilon = \mu = 1$, all losses zero).

---

## 3. Interface interaction

All interface physics is computed by one member function, `Material::interact_with(other, ...)`: `this` is the medium the ray travels in (medium 1) and `other` the medium it enters or reflects off (medium 2). Both passes, both field models, and the thin-slab mirrors derive their coefficients from this single source, which is what makes the cross-checks of Section 9 possible.

### 3.1 Angles and admittances

From the signed grazing angle $\theta$ (Section 1), the incidence cosine and sine are

$$
\cos\theta_i = \big|\cos(\theta + \tfrac{\pi}{2})\big| = |\sin\theta|,
\qquad
\sin\theta_i = \sqrt{1 - \cos^2\theta_i}.
$$

The interface permittivities include the resonance pole, $\varepsilon_k^\mathrm{if} = \varepsilon_k(f) + \varepsilon_{\mathrm{res},k}(f)$, and the normalized admittances are $Y_k = \sqrt{\varepsilon_k^\mathrm{if}/\mu_k}$. The Snell geometry uses the complex index ratio

$$
\rho_{12} = \frac{\varepsilon_1^\mathrm{if}\mu_1}{\varepsilon_2^\mathrm{if}\mu_2},
\qquad
\cos\theta_t = \sqrt{1 - \rho_{12}\sin^2\theta_i},
\qquad
n_{12} = \sqrt{|\rho_{12}|},
$$

where $\cos\theta_t$ is the (complex) refraction cosine and $n_{12}$ the real Snell ratio that also bends the refracted ray direction (Section 4.3). The medium ordering "dense to light" is decided by the real part of the index product: $\mathrm{Re}(\varepsilon_1\mu_1) > \mathrm{Re}(\varepsilon_2\mu_2)$.

### 3.2 Fresnel coefficients

For the EM types the TE (perpendicular, horizontal) and TM (parallel, vertical) coefficients follow Rec. ITU-R P.2040, eqs. (31)–(32) [1], written in admittances so that $\mu \neq 1$ is handled uniformly:

$$
R_\mathrm{TE} = \frac{Y_1\cos\theta_i - Y_2\cos\theta_t}{Y_1\cos\theta_i + Y_2\cos\theta_t},
\qquad
R_\mathrm{TM} = \frac{Y_2\cos\theta_i - Y_1\cos\theta_t}{Y_2\cos\theta_i + Y_1\cos\theta_t},
$$

$$
T_\mathrm{TE} = \frac{2\,Y_1\cos\theta_i}{Y_1\cos\theta_i + Y_2\cos\theta_t},
\qquad
T_\mathrm{TM} = \frac{2\,Y_1\cos\theta_i}{Y_2\cos\theta_i + Y_1\cos\theta_t}.
$$

The associated interface power gains average the two polarizations, $R = \tfrac{1}{2}(|R_\mathrm{TE}|^2 + |R_\mathrm{TM}|^2)$ and likewise for $T$.

**Total internal reflection** is decided uniformly for all six types: $\mathrm{tir} = \mathrm{force\_tir} \;\lor\; n_{12}\sin\theta_i \ge 1$ (the caller forces it for ray-tube consistency, Section 4.6). Under TIR the interface becomes a perfect mirror — $R_\mathrm{TE} = R_\mathrm{TM} = 1$ for the EM types, $|R| = 1$ for the scalar types — the reflectance is 1, and the bent/forward Fresnel port vanishes. Any forward energy then comes solely from the transmission factor (Section 2.5), which is zero when $\mathrm{tf} = 0$.

### 3.3 The scalar branch

The scalar types use the TE coefficient only and re-derive a coefficient pair under the symmetric transmission factor (Section 2.5). With $R_0 = \min(\max(|R_\mathrm{TE}|^2, 0), 1)$,

$$
R_\mathrm{eff} = \mathrm{apply\_tf\_pair}(R_0),
\qquad
r = \sqrt{R_\mathrm{eff}}\; e^{\,j\arg R_\mathrm{TE}},
\qquad
t = \sqrt{1 - R_\mathrm{eff}}\; e^{\,j\arg(1 + R_\mathrm{TE})}.
$$

The magnitudes are energy-complementary by construction ($|r|^2 + |t|^2 = 1$); the phases are the Fresnel reflection phase and the Stokes-consistent transmission phase $\arg(1 + R_\mathrm{TE})$ (the field just inside the boundary is $1 + r$), so the tf redistribution moves energy without touching the phase relations the thin-slab series depends on [5].

Types 3 and 4 use this energy-complementary pair ($r$ for reflection, $t$ for undeviated transmission). Type 5 (scalar refraction) instead carries the *field-power* pressure transmission — the scalar analogue of EM refraction (Section 3.4) — with $t = (1 + R_\mathrm{TE})\sqrt{s}$ and port gain $|1 + R_\mathrm{TE}|^2\,s$, where $s = (1 - R_\mathrm{eff})/(1 - R_0)$ scales the tf shift onto the raw pressure coefficient ($s = 1$ at $\mathrm{tf} = 0$). Under TIR, type 5 collapses to the type-4 form.

### 3.4 Energy partition of the six types

The returned interface gain and the coefficient pair $(c_\mathrm{TE}, c_\mathrm{TM})$ depend on the interaction type. Write $R = \tfrac{1}{2}(|R_\mathrm{TE}|^2 + |R_\mathrm{TM}|^2)$ for the Fresnel reflectance, $R_\mathrm{eff} = \mathrm{apply\_tf\_pair}(R)$ for its tf-adjusted value, $T = \tfrac{1}{2}(|T_\mathrm{TE}|^2 + |T_\mathrm{TM}|^2)$ for the Fresnel transmittance, and $s = (1 - R_\mathrm{eff})/(1 - R)$ for the tf scale ($0$ if $R \ge 1$, and $s = 1$ at $\mathrm{tf} = 0$):

| Type | Port gain $G$ | Coefficients |
|------|---------------|--------------|
| 0 (EM reflection) | $R_\mathrm{eff}$; $1$ under TIR | $R_\mathrm{TE}, R_\mathrm{TM}$ rescaled by $\sqrt{R_\mathrm{eff}/R}$ (flat $\sqrt{R_\mathrm{eff}}$ if $R = 0$) |
| 1 (EM transmission) | $1 - R_\mathrm{eff}$ | $T_\mathrm{TE}, T_\mathrm{TM}$ rescaled by $\sqrt{(1 - R_\mathrm{eff})/T}$ |
| 2 (EM refraction) | $T\,s$ (field power); $1 - R_\mathrm{eff}$ under TIR | $T_\mathrm{TE}, T_\mathrm{TM}$ scaled by $\sqrt{s}$ (flat $\sqrt{1 - R_\mathrm{eff}}$ under TIR) |
| 3 (scalar reflection) | $R_\mathrm{eff}$ | $r$ on both slots |
| 4 (scalar transmission) | $1 - R_\mathrm{eff}$ | $t$ on both slots |
| 5 (scalar refraction) | $\lvert 1 + R_\mathrm{TE}\rvert^2\,s$ (field power); $1 - R_\mathrm{eff}$ under TIR | $(1 + R_\mathrm{TE})\sqrt{s}$ on both slots |

The following rules complete the partition:

- **Transmission factor on every port.** Both reflection ports (types 0, 3) and both undeviated-transmission ports (types 1, 4) are driven by the symmetric $R_\mathrm{eff}$ of Section 2.5, so the reflection and transmission halves of an interface stay energy-complementary as tf moves energy between them. The factor rescales magnitudes only; Fresnel ratios and phases are preserved.
- **Energy-conserving undeviated transmission (types 1, 4).** The straight-through beam carries *all* power not reflected: $G = 1 - R_\mathrm{eff}$. For type 1 the Fresnel transmission coefficients are rescaled by $\sqrt{(1 - R_\mathrm{eff})/T}$ so that $\tfrac{1}{2}(|c_\mathrm{TE}|^2 + |c_\mathrm{TM}|^2) = G$ holds exactly; the per-polarization *ratio* and phases stay Fresnel. At $\mathrm{tf} = 0$ this reduces to the lossless Stokes relation $t_{12}t_{21} = 1 - r^2$ in magnitude [5], the convention that closes the slab energy ledger of Section 9.
- **Field-power refraction (types 2, 5).** The bent ports carry the *raw* Fresnel field power — $T$ for EM, $|1 + R_\mathrm{TE}|^2$ for the scalar pressure wave — scaled by the tf factor $s$ (unity at $\mathrm{tf} = 0$, so the baseline is exact Fresnel). This is the field/bent-ray convention, deliberately distinct from the energy-conserving $1 - R_\mathrm{eff}$ of the undeviated types. Under TIR there is no propagating refracted wave, so both collapse to the undeviated form $G = 1 - R_\mathrm{eff}$ along the incidence direction.
- **Lumped interface loss fold-in.** For every transmissive interaction (types 1, 2, 4, 5) the entered material's $G_\mathrm{if}$ (Section 2.4) multiplies the port gain and $\sqrt{G_\mathrm{if}}$ multiplies both coefficients. Reflection (types 0, 3) never applies it.

Types 2 and 5 are the bent dielectric/acoustic paths: raw-Fresnel field power, true Snell bending, and TIR producing the *total-reflection out-codes* of Section 4.1. Under TIR the bent forward port collapses to the tf leak $1 - R_\mathrm{eff}$, which is zero when $\mathrm{tf} = 0$; in that lossless case there is no forward energy and the state machine kills the forward ray (Section 8), while a $\mathrm{tf} > 0$ leak travels undeviated.

---

## 4. Per-hit processing in `ray_mesh_interact`

### 4.1 Hit classification and `out_typeN`

For each ray with a valid first intersection (`fbs_ind > 0`), the FBS and SBS intersection points are computed from the mesh, the face normal is taken from the triangle's winding (counter-clockwise = front face, right-hand rule), the signed grazing angle $\theta$ at the FBS decides front/back illumination, and the second intersection (SBS) refines the topology. Two faces are **colocated** when the FBS–SBS distance is below `colocation_dist` ($1\,\mathrm{mm}$); their normals are compared with tolerance $10^{-4}$ per component: opposing normals mean a material-to-material (M2M) contact, equal normals an overlapping/duplicate face, anything else a corner (the two faces are not parallel).

`out_typeN` is **bit-encoded** as a `qd::bits<uint8_t>`. Six flag bits carry the full classification:

| Bit | Meaning                                                                 |
| :-: | ----------------------------------------------------------------------- |
|  0  | OK flag (0 = no valid interaction / undefined)                          |
|  1  | Front-side flag (1 = front: o→i or M2 hit first; 0 = back: i→o or M1)   |
|  2  | Co-located FBS/SBS flag (1 = single point, required for media-to-media) |
|  3  | Same-direction flag (FBS and SBS normals point the same way)            |
|  4  | Corner-hit flag (FBS/SBS faces not parallel)                            |
|  5  | Total-reflection flag (also set when a transmission factor forced it)   |

The reachable composite values (add 32 for the total-reflection variant; the TIR bit is set whenever incidence reaches or passes critical, and it removes the propagating forward port on the refraction geometries — types 2 and 5):

| Code  |  TIR  | Description                                         |
| :---: | :---: | --------------------------------------------------- |
|   0   |   —   | No hit                                              |
|   1   |  33   | Single hit, inside→outside (exit)                   |
|   3   |  35   | Single hit, outside→inside (entry)                  |
|   5   |  37   | Media-to-media, M1 (current, back) hit first        |
|   7   |  39   | Media-to-media, M2 (next, front) hit first          |
|  13   |  45   | Overlapping faces, inside-inside→outside            |
|  15   |  47   | Overlapping faces, outside→inside-inside            |
|  21   |  53   | Corner hit, inside→outside→inside                   |
|  23   |  55   | Corner hit, outside→inside→outside                  |
|  29   |  61   | Corner hit, inside-inside→outside                   |
|  31   |  63   | Corner hit, outside→inside-inside                   |

Rays with `fbs_ind = 0` are, by default (`compact = true`), omitted from the output, so the compact output set has $n_\mathrm{rayN} \le n_\mathrm{ray}$ entries; the surviving rays' input indices are reported in `ray_indN` so the caller (and `ray_state_update`) can map between the sets. With `compact = false` the no-hit rays are kept and written as transparent pass-throughs (gain 1, identity `xprmat`, `out_type = 0`), so $n_\mathrm{rayN} = n_\mathrm{ray}$.

### 4.2 Material assignment

On a back-side hit the normal is flipped so the Fresnel geometry always sees the incidence side, and the face materials are assigned by orientation: on a front hit the FBS face's material is the *entered* medium $M_2$ (incidence medium $M_1$ defaults to air); on a back hit it is the *incidence* medium $M_1$. For an M2M contact the colocated partner face supplies the other medium. This is the per-hit, stateless approximation — the function cannot know the true surrounding medium of a nested geometry; `ray_state_update` corrects exactly these cases from its tracked state.

### 4.3 Direction update and relaunch

The child direction $\hat{d}$ is the mirror reflection for the reflection geometry (0/3), the unchanged incoming direction for the undeviated geometries (1/4), and for refraction (2/5) the standard Snell construction

$$
\hat{d} = n_{12}\,\hat{u} + \big(n_{12}\cos\theta_i - \mathrm{Re}\cos\theta_t\big)\,\hat{n},
$$

normalized to unit length. Under TIR — the unified test of Section 3.2, or a ray-tube straddle (Section 4.6) — no refraction direction exists, so the refraction geometry reverts to the undeviated incoming direction. The child origin is offset a few `float` ULP along $\hat{d}$ from the interaction point (`origN = fbs + relaunch_offset · d̂`, or from the SBS for a colocated transmissive crossing), and the outgoing segment length is clamped to at least $\max(\texttt{colocation\_dist},\, 2\cdot\texttt{relaunch\_offset})$ so the relaunched destination clears the offset. The beam tube (`trivec`/`tridir`), when present, is propagated per vertex ray through the same geometry, with degenerate vertex hits flagged through an infinite edge length.

`path_dirN` records the refraction-correct continuation: the mirror for geometry 0/3, the Snell direction for the transmissive geometries. For undeviated transmission (types 1/4) this *refracted* direction differs from the geometric continuation used for `origN`/`destN`, letting downstream code (and the state machine's VBS construction, Section 7.4) recover the true transmission angle.

### 4.4 Polarization transfer and output conventions

For the EM types the TE/TM pair is embedded into the global V/H polarization frame by projecting the incoming and outgoing propagation directions onto the incidence plane: with $\hat{e}_Q$ perpendicular and $\hat{e}_P$ parallel to the plane of incidence (TE $\equiv$ H, TM $\equiv$ V in the local frame), the $2\times2$ base-change matrices $Q$ (incoming) and $U$ (outgoing) sandwich the diagonal Fresnel pair. The eight output rows are the interleaved complex entries $[\mathrm{VV}\;\mathrm{HV}\;\mathrm{VH}\;\mathrm{HH}]$ (real, imaginary per entry) in column-major order per ray, and the scalar gain is $G = \tfrac{1}{2}\sum |x_{ij}|^2$.

The scalar types write the single complex pressure coefficient into the first slot, $[\mathrm{Re}\;\mathrm{Im}]$, with $G = |x|^2$ (no $\tfrac{1}{2}$). Keeping the two conventions straight matters downstream: `ray_state_update` patches both arrays under the same mode-dependent convention so that the $G \leftrightarrow$ `xprmat` identity survives every operation (Section 7.6).

### 4.5 Ray-tube TIR consistency

A refracted beam tube must not tear at the critical angle, where the refracted direction runs parallel to the face and the wavefront diverges. Before computing any direction, `ray_mesh_interact` runs one TIR precheck over the spine and all three vertex rays: if the spine *or any* vertex is at or beyond critical ($n_{12}\sin\theta_v \ge 1$), the whole tube — center and edges — is forced onto the undeviated pass-through direction, and the interface coefficients are re-evaluated once with TIR forced so the energy matches.

The forced-TIR forward gain is $1 - R_\mathrm{eff}$: zero when $\mathrm{tf} = 0$ (the straddling tube simply reflects, the near-critical $|T|^2$ blow-up discarded along with the would-be transmission), and nonzero only when the transmission factor leaks energy forward — in which case that leak travels undeviated at the incidence angle, exactly what the mass-law transmission needs. The decision is made once for the whole tube, so every leg takes the same branch and the reported `path_dirN` follows the same forced direction.

---

## 5. Electromagnetic interpretation

The EM domain is the model's native formulation. A radio material is characterized by `a`–`d` (complex permittivity via conductivity), almost always with $\mu = 1$ (columns `e`–`h` absent), $f_\mathrm{ref} = 1\,\mathrm{GHz}$, and at most a lumped `att`. Reflection and room-side absorption follow directly from the Fresnel coefficients of Section 3.2; at normal incidence and $\mu = 1$ they reduce to the familiar $R = (1 - \sqrt{\varepsilon})/(1 + \sqrt{\varepsilon})$. The three EM interaction modes divide the work as described in Section 3.4: type 0 for the reflected ray, type 1 for undeviated through-wall transmission (the standard network-planning approximation), and type 2 for true dielectric refraction where the bent path matters.

The built-in material library follows Rec. ITU-R P.2040-3, Table 3 [1], valid for 1–40 GHz (ground classes 1–10 GHz), defining only `a`, `b`, `c`, `d`, `att` with everything else at default:

| Name                  | a     | b      | c       | d      | att  | max f (GHz) |
|-----------------------|------:|-------:|--------:|-------:|-----:|------------:|
| air                   | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 100 |
| vacuum                | 1.0   | 0.0    | 0.0     | 0.0    | 0.0  | 100 |
| textiles              | 1.5   | 0.0    | 5e-5    | 0.62   | 0.0  | 100 |
| plastic               | 2.44  | 0.0    | 2.33e-5 | 1.0    | 0.0  | 100 |
| ceramic               | 6.5   | 0.0    | 0.0023  | 1.32   | 0.0  | 100 |
| sea_water             | 80.0  | -0.25  | 4.0     | 0.58   | 0.0  | 100 |
| sea_ice               | 3.2   | -0.022 | 1.1     | 1.5    | 0.0  | 100 |
| water                 | 80.0  | -0.18  | 0.6     | 1.52   | 0.0  | 20  |
| water_ice             | 3.17  | -0.005 | 5.6e-5  | 1.7    | 0.0  | 20  |
| itu_concrete          | 5.24  | 0.0    | 0.0462  | 0.7822 | 0.0  | 100 |
| itu_brick             | 3.91  | 0.0    | 0.0238  | 0.16   | 0.0  | 40  |
| itu_plasterboard      | 2.73  | 0.0    | 0.0085  | 0.9395 | 0.0  | 100 |
| itu_wood              | 1.99  | 0.0    | 0.0047  | 1.0718 | 0.0  | 100 |
| itu_glass             | 6.31  | 0.0    | 0.0036  | 1.3394 | 0.0  | 100 |
| itu_ceiling_board     | 1.48  | 0.0    | 0.0011  | 1.075  | 0.0  | 100 |
| itu_chipboard         | 2.58  | 0.0    | 0.0217  | 0.78   | 0.0  | 100 |
| itu_plywood           | 2.71  | 0.0    | 0.33    | 0.0    | 0.0  | 40  |
| itu_marble            | 7.074 | 0.0    | 0.0055  | 0.9262 | 0.0  | 60  |
| itu_floorboard        | 3.66  | 0.0    | 0.0044  | 1.3515 | 0.0  | 100 |
| itu_metal             | 1.0   | 0.0    | 1.0e7   | 0.0    | 0.0  | 100 |
| itu_very_dry_ground   | 3.0   | 0.0    | 0.00015 | 2.52   | 0.0  | 10  |
| itu_medium_dry_ground | 15.0  | -0.1   | 0.035   | 1.63   | 0.0  | 10  |
| itu_wet_ground        | 30.0  | -0.4   | 0.15    | 1.3    | 0.0  | 10  |
| itu_vegetation        | 1.0   | 0.0    | 1.0e-4  | 1.1    | 0.0  | 100 |
| irr_glass             | 6.27  | 0.0    | 0.0043  | 1.1925 | 23.0 | 100 |

`itu_vegetation` follows Rec. ITU-R P.833-9, Fig. 2 [2]; `irr_glass` (infrared-reflective glass) follows 3GPP TR 38.901 V17.0.0, Table 7.4.3-1 [3] and is the only built-in with a lumped `att`. `air` is the transparent fallback for unmatched materials.

---

## 6. Acoustic interpretation

### 6.1 The duality

Acoustic waves at kHz frequencies share their wavelengths with radio waves at GHz frequencies, so a radio ray tracer can simulate room and building acoustics if the material model is populated with acoustically calibrated parameters. The mapping is a wavelength-preserving frequency scaling,

$$
f_\mathrm{radio} = f_\mathrm{acoustic} \times \frac{c_0}{c_\mathrm{sound}}
                 \approx f_\mathrm{acoustic} \times 874{,}030
\qquad (c_\mathrm{sound} = 343\ \mathrm{m/s}),
$$

so $1\,\mathrm{kHz}$ acoustic $\equiv 0.874\,\mathrm{GHz}$ radio, and every acoustic material fixes $f_\mathrm{ref} = 0.874$. Absolute frequencies (`resF`, `coiF`) convert the same way (100 Hz → 0.0874 GHz). The analogy carries interface reflection, bulk absorption, and the mass-law/coincidence/resonance mechanisms; it does not by itself model modal interference or diffraction — the thin-slab factor of Section 7.7 restores exactly the slab-interference part. Simulation results are air-normalized (atmospheric absorption per ISO 9613-1 [6] is removed and re-applied outside).

Acoustic runs use the scalar types (3 = reflection, 4 = undeviated transmission, 5 = Snell-bent refraction): a single pressure coefficient, no polarization, and — like the EM types — total internal reflection beyond the critical angle (Section 3.2). Type 4 keeps the ray undeviated (the standard through-partition approximation); type 5 bends it by Snell's law for the cases where the refracted acoustic path matters. The wave variables map as

$$
\varepsilon \leftrightarrow \text{compressibility}, \qquad
\mu \leftrightarrow \text{effective density}, \qquad
n = \sqrt{\varepsilon\mu}, \qquad
Z \propto \sqrt{\mu/\varepsilon},
$$

so the two-column-pair model gives independent control of the surface impedance (reflection) and the refractive index (bulk propagation and loss) — exactly the two degrees of freedom a porous medium has.

### 6.2 The two material families

**Rigid reflectors** (concrete, glass, steel, brick, gypsum, panels): a one-parameter medium with tiny $\varepsilon_r \ll 1$ (`a` $\sim 10^{-9}$–$10^{-3}$) and $\mu = 1$. The small $\varepsilon$ makes the surface impedance huge, $|R| \to 1$, with the residual room-side absorption $1 - |R|^2$ tuned by `a`. Inverting the normal-incidence Fresnel relation for a target absorption $\alpha_\mathrm{abs}$:

$$
a = \left(\frac{1 - \sqrt{1 - \alpha_\mathrm{abs}}}{1 + \sqrt{1 - \alpha_\mathrm{abs}}}\right)^2
\qquad (\text{small } \alpha_\mathrm{abs}: \ a \approx \alpha_\mathrm{abs}^2/16).
$$

Because $\varepsilon \ll 1$ the body is "optically rarer" than air, so the air→wall crossing is dense-to-light with $|R| \to 1$ and the room-side reflection dominates. The partition's *isolation* — `att` (level), `m` (mass-law slope), and `coi*` (coincidence dip), the acoustic transmission-loss toolbox [7] — is carried on the through-partition transmission path.

**Porous absorbers** (foam, mineral wool, fiberglass, carpet, curtains): a genuine two-parameter medium. Given the layer's complex refractive index $n(f)$ and normalized surface impedance $z(f)$ — from a Delany–Bazley flow-resistivity fit [5] or from measurement — the columns follow in closed form:

$$
\varepsilon = \frac{n}{z} \quad (\text{compressibility}),
\qquad
\mu = n \, z \quad (\text{density}),
$$

fitted over the working band as power laws: $\mathrm{Re}\,\varepsilon \to (a, b)$, $\mathrm{Im}\,\varepsilon \to (c, d)$, $\mathrm{Re}\,\mu \to (e, f)$, $\mathrm{Im}\,\mu \to (g, h)$. Worked example, mineral wool at flow resistivity $\approx 12{,}000\ \mathrm{Pa\,s/m^2}$:

| a    | b     | c      | d    | e    | f     | g     | h    |
|-----:|------:|-------:|-----:|-----:|------:|------:|-----:|
| 1.16 | −0.04 | 0.0038 | 1.46 | 1.66 | −0.19 | 0.091 | 0.13 |

with $f_\mathrm{ref} = 0.874$ and `att` = `alpha` = `m` = 0. The fit matches surface reflection and bulk attenuation simultaneously across roughly 250 Hz–4 kHz — the thing a single-$\varepsilon$ model cannot do, because it locks $Z = Z_0/\sqrt{\varepsilon}$ to $n = \sqrt{\varepsilon}$. Note where the loss lives: $\varepsilon$ stays nearly real (pore air is air), and $\mu$ carries the viscous loss of the frame-loaded effective density, which is where dissipation physically resides.

The remaining mechanisms map one-to-one: `res*` builds Helmholtz/membrane/micro-perforated absorbers (a Lorentz peak in absorption and a reflection feature at $f_\mathrm{res}$); `coi*` the coincidence dip of thin stiff panels; `m` the mass law of rigid partitions; `tf` the reflection/transmission redistribution of leaky-but-hard surfaces (perforated panels, grilles).

### 6.3 Layered constructions and known limits

Glued stacks (absorber on wall, carpet on floor) are modeled as watertight bodies with coincident faces (within the 1 mm `colocation_dist` tolerance) and opposing normals; `ray_mesh_interact` classifies the shared face as an M2M contact and the Fresnel coefficients are computed from the two real materials directly. The state machine then handles the cascade (Section 8); the thin-slab factor restores the internal interference of each layer.

Two limits are worth keeping in mind. First, the mass-law angle behavior is modeled through the $\ell_\mathrm{g}\cos^2\theta_i$ argument of Section 2.3 — correct in trend, approximate in detail, and angle-blind in the few bookkeeping rows where no incidence angle is available. Second, a porous layer on a rigid backing develops a velocity node at the wall: below the quarter-wave frequency $f = c_\mathrm{sound}/(4t)$ of the layer thickness $t$, a geometric trace-through overstates absorption. The Airy resolution of Section 7.7 *is* the analytic resummation of the internal bounce series and closes most of this gap for slabs the tracer presents as such; configurations the gates re-emit fall back to the (energy-safe) geometric treatment.

---

## 7. The ray-state machine `ray_state_update`

### 7.1 Purpose

`ray_mesh_interact` reports what one hit looks like; `ray_state_update` decides what it *means*: it corrects the per-interaction `gainN`/`xprmatN`/`path_dirN` in place using tracked per-ray state, carries an accumulated in-medium path length, writes the next state and a resolved-type code, and overlays the thin-slab resolution. The tracer calls it twice per interaction — once on the reflection pass (types 0/3) and once on the transmission/refraction pass (types 1/2/4/5).

Three ingredients carry most of the physics and are described in Sections 7.4–7.5: the **VBS (virtual back-scatter) correction**, which re-evaluates the interface at the true refracted angle of the incoming segment; the **accumulated in-medium distance**, which lets loss be charged once per traversed segment across interactions; and the **excess-phase** term, which restores the optical path difference between the refracted in-slab path and the geometric free-space reference.

### 7.2 State encoding

Three signed `short` words per ray (6 bytes), each split into a 15-bit material index and a flag bit. The masks are the only sanctioned access pattern — never arithmetic negation, since $-X$ does not encode "material $X$ with flag" and `abs(-32768)` overflows:

```
mat      = w & 0x7FFF          // 0 = outside / air;  1 ... 32767 = material index
flag     = w & 0x8000          // sign bit
X | flag = (short)(X | 0x8000) // set a flag: an OR, never -X
```

- `mtl_ind_current` — `mat` is the current medium (0 = outside). The flag bit is **RESOLVED**: a thin-slab reflection has been collapsed into $S$ and this ray is on its transparent return path.
- `mtl_ind_prev` — `mat` is the previous medium: the medium behind the far interface of the current slab, used as the far mirror of $S$. The flag bit is **KNOWN-NON-PARALLEL**: the wedge test at entry proved the two captured faces are not a slab pair, so resolution is forbidden. The value `(short)0x8000` legitimately means "air, flag set."
- `mtl_ind_buffer` — `mat` only: the one-slot next-transition buffer for nested/overlapping geometry.

### 7.3 Index spaces, optional inputs, and the validation contract

Two index spaces coexist. The full-set arrays — `orig`, `dest` (segment endpoints), the hit counter `no_interact`, the incoming physical direction `path_dir_prev`, the accumulated distance `acc_dist_in`, and the *input* state words — live in the **full ray set** $[n_\mathrm{ray}]$ and are read at $g = \mathrm{ray\_ind}[i]$. The compact-set arrays — the interaction points `fbsN`/`sbsN`, the interaction outputs (`fbs_angleN`, `out_typeN`, `normal_vecN`, `mtl_ind_fbsN`/`mtl_ind_sbsN`), the *output* state words, `gainN`/`xprmatN`/`path_dirN`, `acc_dist_outN`, and `resolved_typeN` — live in the **compact set** $[n_\mathrm{rayN}]$ at $i$. A null `ray_ind` means identity (then $n_\mathrm{ray} = n_\mathrm{rayN}$ is enforced). Input state is read-only; output arrays are resized on demand.

Required unconditionally: `orig`, `dest` ($[n_\mathrm{ray},3]$), `fbsN`, `sbsN` ($[n_\mathrm{rayN},3]$), `no_interact`, `fbs_angleN`, `out_typeN`, and `normal_vecN` ($[n_\mathrm{rayN},6]$), which supplies the VBS plane normal for the Snell corrections and gates the wedge test. Optional with defaults: `path_dir_prev` (absent → the VBS lands on the FBS and the angle stays the reported `fbs_angle`), `acc_dist_in` (absent → zero carried distance), the three input state words (absent → state 0 = outside), the three output state words, `gainN`/`xprmatN`/`path_dirN`/`acc_dist_outN`/`resolved_typeN` (absent → that write is skipped), and `mtl_prop` itself — though any nonzero material index referenced without a table is rejected. The material map may be sparse (Section 2.6). Passing all six state args null disables tracking: each interaction is then corrected on its own (entry loss, TR kill, single-hit air-gap $S$), but the cross-interaction slab $S$ and the reflection-bounce $S$ need the tracked medium. Everything dimension-bearing is validated up front — array sizes, `ray_ind` bounds, masked material indices in the face arrays *and* in all three input state arrays, finite positive frequency, finite non-negative `eps`, finite non-negative `acc_dist_in`, finite `path_dir_prev` — because the per-ray loop runs under OpenMP where a thrown exception cannot propagate.

### 7.4 The VBS correction and the refracted incoming segment

`ray_mesh_interact` builds its interface result from the *geometric* orig→FBS direction and the reported `fbs_angle`. But when the ray entered the current medium at a previous refraction, the direction it actually travels inside the body is bent, so the true incidence angle at this interface — and the true in-slab path length — differ from the geometric ones. `path_dir_prev` carries that genuine refracted direction, and the machine reconstructs the geometry around a **virtual back-scatter point (VBS)**: the point where the refracted incoming direction, launched from the segment origin, crosses the FBS plane (normal $\hat{n}$ from `normal_vecN`).

From the VBS the machine recovers the corrected refraction cosine $\cos\theta_t = |\hat{d}_\mathrm{prev}\cdot\hat{n}|$, the corrected grazing angle $\theta_t$, and the refracted origin→VBS distance $\ell_\mathrm{r}$. When the correction is material (the corrected cosine differs from the geometric one by more than $10^{-6}$), `REPLACE_BY_VBS` discards the FBS-relative interface result and rebuilds it: it re-evaluates `interact_with` at $\theta_t$ (re-deciding TIR at the corrected angle), reflects or Snell-refracts the continuation direction about $\hat{n}$ using the refracted incoming direction, re-projects the polarization basis, and rewrites `gainN`/`xprmatN`/`path_dirN`. The VBS flag is set on `resolved_typeN`. Without `path_dir_prev` the VBS coincides with the FBS and the reported `fbs_angle` is kept untouched — deriving an angle from the geometric orig→FBS direction would override the value the tracer already computed.

For the special crossings where the relaunch happens past the SBS (an M2M back face, an inside-inside corner exit, or an outside-inside-outside corner), the SBS normal is used as the plane normal instead of the FBS normal, because that face governs the true exit geometry.

### 7.5 Accumulated in-medium distance and excess phase

Because in-medium loss is charged by the state machine rather than at the hit (Section 4.4), the machine carries the in-slab path across interactions in `acc_dist` — a two-column array holding the **refracted** distance $\ell_\mathrm{r}$ (col 1, feeding the dielectric and power-law loss and the propagation phase) and the **geometric** thickness distance $\ell_\mathrm{g}$ (col 2, feeding the mass law), exactly the two arguments of `medium_gain` (Section 2.3). Each dispatch row either *adds* the current segment's contribution (`DIST_ADD`, using the VBS-refracted origin→VBS length where available) or *resets* it at a medium entry (`DIST_SET`), so that when a segment is finally charged, its two lengths are complete and each meter of medium is counted once.

`SCALE_BY_MEDIUM` applies the medium as a complex factor: magnitude $\sqrt{G_\mathrm{med}(\ell_\mathrm{r}, \ell_\mathrm{g})}$ and phase $-\,\Delta\phi$ with

$$
\Delta\phi = k_0\,\big(n\,\ell_\mathrm{r}\cos^2\theta_t \;-\; \ell_\mathrm{g}\cos^2\theta_i\big),
\qquad
k_0 = \frac{2\pi f\cdot 10^9}{c_0},
\qquad
n = \mathrm{Re}\big(\sqrt{\varepsilon\mu}\big),
$$

the **excess phase** — the optical path of the refracted in-slab traversal minus the geometric free-space reference on the same thickness axis. The $\cos^2$ **walk-off** factor is applied only for the undeviated tracer (undeviated transmission, the reflection pass, or a resolved return path), where it substitutes for the untraced lateral shift of the flat-slab approximation; genuine refraction (types 2/5) traced the bent path already, so its walk-off cosine is 1. The medium index $n$ excludes the resonance pole (a bulk-phase quantity, Section 2.2); an air segment yields a lossless, unit-index, zero-excess-phase factor.

### 7.6 Gain patch operations

The dispatch selects among a small set of in-place operations on `gainN` and `xprmatN` (either may be null; the other is still patched), all under the mode convention of Section 4.5:

- **IG (keep).** The incoming interaction gain stands; nothing is touched.
- **`REPLACE_BY_GAIN`($G$, keep_dir).** `xprmat` is zeroed and $\sqrt{G}$ written on the diagonal — VV and HH for EM, VV alone for scalar — and `gainN` set to $G$. A transparent pass ($G = 1$ on the transmission pass) sets the pass-through flag and clears TIR; with `keep_dir` it also overwrites `path_dirN` with the incoming physical direction so a same-medium crossing carries the ray straight through.
- **`REPLACE_BY_VBS`.** The VBS rebuild of Section 7.4 (coefficients, direction, basis, gain).
- **`SCALE_RAY`($z$) / `SCALE_BY_MEDIUM`.** Multiply every Jones pair (off-diagonals included) by a complex factor $z$ and the gain by $|z|^2$. `SCALE_BY_MEDIUM` supplies $z = \sqrt{G_\mathrm{med}}\,e^{-j\Delta\phi}$ (Section 7.5).
- **KILL.** Zero everything and set `resolved_type = 0`; the ray is terminated for this pass.

### 7.7 Thin-slab resolution: the Airy factor

A parallel slab traps an infinite internal reflection series. Its closed form is the Airy sum [4]: with $r_\mathrm{near}$ the field reflection at the interface being processed (seen from inside the slab), $r_\mathrm{far}$ the reflection at the opposite interface, and $\varphi$ the one-way in-slab propagation factor,

$$
S = \frac{1}{1 - r_\mathrm{near}\, r_\mathrm{far}\, \varphi^2},
$$

so that multiplying one traced interaction by $S$ replaces the entire bounce series the tracer would otherwise have to follow. `SLAB_AIRY_FACTOR` (given the slab material and the two mirror materials) computes $S$ and applies it, or returns **false** to signal *re-emit*, in which case the dispatch keeps the bare interaction and the tracer continues geometrically — re-emission is always energy-safe.

**Per-polarization factor.** For the EM types the factor is computed independently for TE and TM, $S_\mathrm{TE}, S_\mathrm{TM}$, so each channel carries its own slab retardation and the resulting depolarization. The *total* power gain is held at the TE/TM-averaged value: the rebuilt `xprmat` uses per-channel factors $S_\mathrm{TE}\sqrt{T^\mathrm{sp}_\mathrm{TE}/\bar T^\mathrm{sp}}$, $S_\mathrm{TM}\sqrt{T^\mathrm{sp}_\mathrm{TM}/\bar T^\mathrm{sp}}$ that carry the true per-channel single-pass transmittance while re-referencing the averaged port magnitude the interaction already applied, so $|{\cdot}|^2 = T^\mathrm{sp}\,|S|^2$ is the exact, coherent per-channel slab transmittance. Scalar mode keeps the single complex factor.

**Mirrors.** Each $r$ is the TE (and, for EM, TM) Fresnel coefficient from the slab side (Section 3.2, interface permittivities with resonance), with the transmission factor folded into its magnitude and the Fresnel phase preserved: $r = \sqrt{\mathrm{apply\_tf}(|r_\mathrm{Fresnel}|^2)}\, e^{\,j\arg r_\mathrm{Fresnel}}$. The tf **owner** is the face-owning solid: the slab itself when the slab is a real material, the adjacent material when the cavity is air (the i-o-i air gap, whose faces belong to the bounding solids). Which materials act as mirrors depends on the call site and is listed per topology in Section 8; the pattern is that the near mirror is whatever lies on the far side of the interface being processed and the far mirror is the **previous medium** from the state word — air for a free-standing slab, the neighboring layer inside a stack.

**Propagation factor.** Magnitude and phase have different sources, deliberately:

$$
|\varphi| = \sqrt{G_\mathrm{med}(\ell_\mathrm{r}, \ell_\mathrm{g})},
\qquad
\arg\varphi = -\frac{\omega}{c_0}\,n\,\ell_\mathrm{r}\cos^2\theta_t,
\qquad
\omega = 2\pi f \cdot 10^9,
$$

with $\ell_\mathrm{r}$ the accumulated refracted in-slab path (Section 7.5). The magnitude uses the *full* loss model; the phase uses the resonance-excluded index $n$, since the pole belongs to the interfaces, not the bulk, and the loss-only terms add no propagation phase. An air cavity yields a lossless, unit-index $\varphi$.

**Gates.** Resolution happens only when all three pass; otherwise re-emit:

1. *Parallelism.* If the entry captured both faces, the wedge test compares the FBS and SBS normals: the faces count as parallel iff $|\hat{n}_F \cdot \hat{n}_S| > 1 - 3.8\cdot10^{-3}$ (about $5^\circ$), provided the two points are distinct ($d(\mathrm{fbs},\mathrm{sbs}) > \texttt{colocation\_dist}$). The magnitude test is essential — at an M2M back face the coincident face pair has opposing normals and floating-point noise decides which one is reported, so a signed test would re-emit valid slabs at random. A failed test sets the KNOWN-NON-PARALLEL flag on `prev`, which forces re-emission at every later event of that traversal.
2. *Survival.* The round-trip amplitude must be worth resolving: $\rho = \sqrt{\max(R^\mathrm{TE}_\mathrm{near} R^\mathrm{TE}_\mathrm{far},\, R^\mathrm{TM}_\mathrm{near} R^\mathrm{TM}_\mathrm{far})\; G_\mathrm{med}(2\ell)} \ge \varepsilon_\mathrm{thr}$ (the stronger polarization decides), where the $R$ are the tf-effective power reflectances and `eps` is the caller's threshold, tied to the engine's ray-drop level ($\varepsilon_\mathrm{thr} \approx \mathrm{drop}^{1/N_\mathrm{max}}$, typically 0.1–0.25; $\varepsilon_\mathrm{thr} \ge 1$ disables resolution entirely, $0$ resolves everything the other gates allow). Using the full $G_\mathrm{med}(2\ell)$ keeps the gate consistent with $|\varphi|^2$ — exactly so for the dielectric and power-law terms, approximately for the logarithmic mass term, whose mismatch slightly under-damps $S$ and is bounded by the energy ledger.
3. *Pole clamp.* If $|1 - r_\mathrm{near} r_\mathrm{far} \varphi^2| < 10^{-2}$ for either polarization, re-emit. The Airy pole (lossless, on-resonance, near-grazing simultaneously) would otherwise produce an unbounded $|S|$; the clamp caps $|S| \lesssim 100$ and hands the case back to the tracer.

A known non-parallel `prev` flag re-emits before any of this (with $\varepsilon > 0$; $\varepsilon = 0$ always resolves). All gate inputs are functions of the slab geometry and materials only, so the reflection and transmission passes necessarily make the *same* resolve/re-emit decision for a given slab — the cross-pass invariance that prevents double-counted or lost energy.

### 7.8 The two ports and in-medium loss ownership

The reflection and transmission ports of a slab are different functions of $S$; getting both right is what closes the energy ledger. For a slab with entry interface 12 and back interface 23:

$$
T_\mathrm{slab} = t_{12}\,\varphi\,t_{23}\,S,
\qquad
R_\mathrm{slab} = r_{12} + t_{12}\,t_{21}\,r_{23}\,\varphi^2\,S .
$$

The dispatch realizes these across the traced events rather than in one place. On the transmission path, the entry event contributes $t_{12}$ (its incoming IG), the exit event contributes $t_{23}$ and is the one multiplied by $S$. On the reflection path, the front reflection stays **bare** $r_{12}$ — no $S$ — while the *internal* back reflection is multiplied by $S$ (contributing $r_{23} S$) and sets the RESOLVED flag; that ray then exits the front transparently, picking up $t_{21}$, so the product across its events is exactly the second term of $R_\mathrm{slab}$. A resolved ray reaching the front face on a *reflection* pass is killed — its would-be second bounce is already summed inside $S$.

In-medium amplitude and phase must be applied exactly once per traversed segment. The ownership rule, expressed through the accumulated distance of Section 7.5:

- **Entry resets the accumulator.** A medium entry `DIST_SET`s the in-slab path to the outgoing (or preloaded) segment; the loss is not applied yet.
- **Each transition charges the incoming segment.** A crossing or exit `DIST_ADD`s the origin→VBS (refracted) and origin→FBS (geometric) lengths, then `SCALE_BY_MEDIUM` charges the current medium over the accumulated path, and the accumulator is reset for the next leg.
- **The resolving reflection charges its own incoming segment and stops.** It is the seam of the resolved return path; the reflection pass kills any further resolved front hit.

Following one resolved return path through a stack confirms each segment is charged exactly once, and the phase bookkeeping mirrors it: $\varphi$ supplies the in-slab phase of the resonant series inside $S$, while `SCALE_BY_MEDIUM`'s excess phase covers the explicit traversals.

### 7.9 Energy safety and stacked slabs

For a single slab the two ports are exact and $R + T + A = 1$ holds to numerical precision (lossless and lossy; verified in the test suite at normal and oblique incidence). Stacked slabs couple their cavities through higher-order terms that no per-cavity factor can represent. The implemented policy makes the truncation *strictly conservative*: the M2M cavity transition that applies $S$ also sets the RESOLVED flag (`current_out = iM | 0x8000`), so the first resolved cavity of a traversal is the only one — every later internal crossing and the final exit are transparent pass-throughs, and the reflection pass kills the resolved ray. The discarded higher-order couplings are positive-energy terms, so the residual is a benign under-count: the composed $R + T \le 1$ at every phase combination, never an energy-creating over-count. (The test suite checks this against an exact transfer-matrix reference over a grid of layer thicknesses.)

---

## 8. Dispatch tables

The state machine branches on the bit-encoded `out_typeN` of Section 4.1 (`typeH`, with the TR bit stripped and carried separately), splits on the pass, and switches on the interaction **topology**. This section lists every branch as implemented and the `resolved_typeN` code it emits.

Notation: `cur` = `current_in & 0x7FFF`, `resolved` = flag of `current_in`; `buf` = buffer material; `prev` = previous material, `nonpar` = flag of `prev_in`. $M_1$ = `mtl_ind_fbsN`, $M_2$ = `mtl_ind_sbsN`, both masked. `nH` = `no_interact`. $\mathrm{SAME}(a,b)$ is the same-medium test (equal indices, or two real materials with identical parameter rows); index 0 (outside) matches only itself. `ray_ends` $=$ (`nH == 1`, or `nH == 2` with colocated FBS/SBS) marks a segment whose next leg is already the ray's final leg, so its length can be preloaded.

### 8.1 `resolved_typeN` codes

The output classifier is a `qd::bits<uint8_t>` written per interaction (0 = ray killed):

| Bit  | Flag        | Meaning                                                                        |
| :--: | :---------: | ------------------------------------------------------------------------------ |
|   0  | ok          | OK flag (0 = a deferred degenerate-resolve buffer is pending)                  |
|   1  | vbs         | VBS correction (gain/xprmat corrected at the VBS instead of FBS/SBS)           |
|   2  | resolve     | Slab-resolve flag (an internal multi-bounce series was resolved analytically)  |
|   3  | inside      | Inside-object flag (1 = ray continues inside, 0 = continues outside)           |
|   4  | fix         | Fix flag (resolved-false-outside, or entry/exit material mismatch)             |
|   5  | tir         | Total-reflection flag (also set when a transmission factor forced reflection)  |
|   6  | trans       | Transmission: transparent-interface flag; reflection: scatter flag             |
|   7  | refl        | Reflection flag (0 = transmission/refraction, 1 = reflection/scattering)       |

Reachable composite values for the transmission/refraction pass (add the listed TIR value for the total-reflection variant):

|  Dec | Hex  | FIX  |   TIR   | Flags set                  | Meaning                                            |
| :--: | :--: | :--: | :-----: | -------------------------- | -------------------------------------------------- |
|    9 | 0x09 |   —  |    41   | inside, ok                 | o-i entry, OR i-i transition (refr. 2/5, FBS==VBS) |
|    8 | 0x08 |   —  |    40   | inside                     | o-i entry, deferred buffer set (overlap/edge)      |
|   11 | 0x0B |  27  |  43, 59 | inside, vbs, ok            | i-i transition (undev. 1/4, VBS relocated)         |
|   13 | 0x0D |  29  |  45, 61 | inside, resolve, ok        | i-i transition + slab series (refr. 2/5, FBS==VBS) |
|   15 | 0x0F |  31  |  47, 63 | inside, resolve, vbs, ok   | i-i transition + slab series (undev. 1/4, VBS)     |
|    1 | 0x01 |  17  |  33, 49 | ok                         | i-o exit (refr. 2/5, FBS==VBS)                     |
|    3 | 0x03 |  19  |  35, 51 | vbs, ok                    | i-o exit (undev. 1/4, VBS relocated)               |
|    5 | 0x05 |  21  |  37, 53 | resolve, ok                | i-o exit + slab series (refr. 2/5, FBS==VBS)       |
|    7 | 0x07 |  23  |  39, 55 | resolve, vbs, ok           | i-o exit + slab series (undev. 1/4, VBS)           |
|   73 | 0x49 |  89  |     —   | trans, inside, ok          | ignore-hit / same-medium pass, no gain change      |
|   72 | 0x48 |  88  |     —   | trans, inside              | nested pass-through, buffer deferred               |
|   65 | 0x41 |   —  |     —   | trans, ok                  | advance to ray destination, identity interface     |

Reachable composite values for the reflection pass:

|  Dec | Hex  |  FIX |   TIR   | Flags set                       | Meaning                                                      |
| :--: | :--: | :--: | :-----: | ------------------------------- |------------------------------------------------------------- |
|  129 | 0x81 |    — |   161   | refl, ok                        | eager front reflection (R0), outside (FBS==VBS)              |
|  137 | 0x89 |  153 | 169,185 | refl, inside, ok                | internal back-reflection (incoming refr. 2/5, FBS==VBS)      |
|  139 | 0x8B |  155 | 171,187 | refl, inside, vbs, ok           | internal back-reflection (incoming undev. 1/4, VBS)          |
|  141 | 0x8D |  157 | 173,189 | refl, inside, resolve, ok       | internal back-reflection + slab series (incoming refr. 2/5)  |
|  143 | 0x8F |  159 | 175,191 | refl, inside, resolve, vbs, ok  | internal back-reflection + slab series (incoming undev. 1/4) |
| 192+ | 0xC0+|    — |    —    | refl, trans, …                  | reserved: scattering not implemented                         |

### 8.2 Pass selection

```
REFLECTION pass (types 0, 3)                 -> Section 8.3
TRANSMISSION / REFRACTION pass (types 1,2,4,5) -> Section 8.4
```

The TR bit is read from `typeH`, stripped, and copied onto `resolved_type`; both passes then dispatch on the remaining topology.

### 8.3 Reflection pass (types 0, 3)

The reflection flag is set. `mtl_current` is resolved to the exit material of the incoming segment (`GET_EXIT_MATERIAL`, which also raises the fix flag on a current/exit mismatch); the material reflected off is $M_2$ for an M2M M1-first hit (typeH 5), $M_1$ for an M2M M2-first hit (typeH 7), otherwise air.

| Condition | Action | `resolved_type` |
|---|---|---|
| `resolved` | **KILL** — the second bounce is already summed inside $S$ | 0 |
| transparent-forward event (no hit, corner o-i-o, deferred buffer, or a same-medium reflect-off) | **KILL** — a pass-through cannot also reflect | 0 |
| `cur == 0` (entry front reflection) | **IG**, bare Fresnel copy-through; reset the accumulator | 129 |
| `cur != 0` (internal back reflection) | `DIST_ADD`; `REPLACE_BY_VBS` mirror at the VBS; `SLAB_AIRY_FACTOR`(cur; reflect-off, prev) — on resolve set the RESOLVED flag on `current_out`; `SCALE_BY_MEDIUM`(cur); reset the accumulator; set inside | 137 / 139 (+resolve → 141 / 143) |

The internal-reflection row is the seam of Section 7.8: it charges its incoming segment, applies (or defers) the slab factor, and continues inside.

### 8.4 Transmission / refraction pass (types 1, 2, 4, 5)

The switch is on the topology encoded in `typeH`. The shared transition kernel `APPLY_TRANSITION` (Section 8.5) does the actual gain/direction work; the rows below select which medium it transitions into and how the buffer/state advance.

**No crossing** — `nH == 0`, or the OK bit is clear (destination lies in the FBS plane): `DIST_ADD` to advance to the destination, `REPLACE_BY_GAIN`(1) as an identity interface. If outside, this is the advance-to-destination pass (65); if inside a medium, it is a same-medium pass carrying the ray straight through (73, inside preserved).

**o-i entry** — `typeH` ∈ {3, 15, 31}. Sets inside.
- `cur == 0` (entry from air): keep the interaction; `next_current = M1`; run the wedge test (unless the segment ends here) and stash its outcome as the KNOWN-NON-PARALLEL flag on `prev`. For a degenerate o-ii capture (15/31) push $M_2$ into the buffer and clear the OK bit (deferred, code 8). If the ray ends, preload the next leg's VBS distance; otherwise `DIST_SET` to restart the accumulator at the FBS. Code 9 (8 when deferred).
- `cur != 0` (nested o-i on an inside state): `APPLY_TRANSITION`(stay in `cur`), buffer $M_1$, clear OK (nested pass-through, code 72).

**i-o exit / ii-o** — `typeH` ∈ {1, 13, 29}.
- `buf == 0`: `APPLY_TRANSITION`(exit to air). Codes 1/3/5/7 (with vbs/resolve as applicable).
- `typeH == 1`, `buf != 0` (virtual i-i): if $\mathrm{SAME}(\mathrm{buf}, M_1)$, stay in `cur` and clear the buffer; otherwise `APPLY_TRANSITION` into the buffer material (codes 9/11/13/15).
- `typeH` ∈ {13, 29}, `buf != 0` (ii-o): `APPLY_TRANSITION`(exit to air).

**Corner o-i-o** — `typeH == 23`.
- `cur == 0` (in air, ignore the corner): `DIST_ADD`, `REPLACE_BY_GAIN`(1) pass-through (code 65).
- `cur != 0` (inside): `APPLY_TRANSITION`(exit to air).

**Corner i-o-i** — `typeH == 21`.
- `buf == 0`, $M_2 = 0$: **KILL** (illegal).
- `buf == 0`, $M_2 \neq 0$ (air gap bounded by $M_1$/$M_2$): resolve the exit material, then `APPLY_TRANSITION`(into $M_2$) as an i-i crossing.
- `cur != 0`, `buf != 0`: if $\mathrm{SAME}(\mathrm{buf}, M_1)$ stay in `cur`, else `APPLY_TRANSITION` into the buffer.
- `buf != 0`, `cur == 0`: **KILL**.

**i-i (M2M)** — `typeH` ∈ {5, 7}.
- `buf == 0`: `APPLY_TRANSITION` into the auto-selected next medium ($M_2$ for typeH 5, $M_1$ for typeH 7).
- `buf != 0`: stay in `cur`, swap the buffer to the other face material, clear OK (deferred).

**Unmatched** `(typeH, state)`: **KILL** — the unified global default. An inconsistent state removes energy rather than inventing it.

### 8.5 The transition kernel `APPLY_TRANSITION`

Every entry/exit/crossing routes through one helper, parameterized by whether it is a cavity exit, the next medium, and whether to preload the destination leg. It resolves the current medium from the exit face, then splits on the same-medium test:

- **Same medium** ($\mathrm{SAME}(\mathrm{cur}, \mathrm{next})$): make the interface transparent — `REPLACE_BY_GAIN`(1, keep_dir) carries the refracted direction through and sets the transparent flag; distances are accumulated (with the fbs→vbs and vbs→dest legs preloaded when requested). No slab factor, no medium charge here; the segment is charged at the next real transition.
- **Different media or exit**: `REPLACE_BY_VBS`(cur, next, type 1/4, geom exit/refract) rebuilds the interface at the corrected angle; unless already `resolved`, `SLAB_AIRY_FACTOR`(cur; next, prev) is attempted and sets the resolve flag on success; `SCALE_BY_MEDIUM`(cur) charges the current medium over the accumulated path; the accumulator is reset (preloading the entered medium's refracted slant, or the unrefracted post-exit leg, as appropriate); and the state advances (`prev_out = cur`, `current_out = next`). A cavity transition that resolves keeps the RESOLVED flag on `current_out`, the stacked-slab persist rule of Section 7.9.

The buffer is cleared and, unless this was an exit, the inside flag is set.

---

## 9. Validation

The behavior above is pinned by a blind Catch2 suite (`test_ray_state_update.cpp`) written against this specification with independently derived oracles:

- **Dispatch and state.** Every topology branch of Section 8, the state encoding and the `resolved_typeN` codes, flag persistence, the KNOWN-NON-PARALLEL propagation, `ray_ind` mapping, batch determinism, and `float`/`double` parity.
- **Geometry correction.** The VBS reconstruction (Section 7.4): corrected incidence angle and refracted segment length, the Snell/mirror continuation written into `path_dirN`, the excess-phase term, and the accumulated-distance carry across interactions — including the type-5 scalar-refraction path.
- **Energy ledger.** For a single slab, $R + T = 1$ (lossless) and $R + T + A = 1$ (strongly lossy, at normal and oblique incidence), with $R$ and $T$ assembled from the two passes exactly as a tracer would. This is the primary guard on the port decomposition, the Stokes/tf consistency, and the loss-ownership rule.
- **Slab physics.** $S$ against the closed-form Airy sum over a phase sweep; the per-polarization TE/TM factor and its port correction; the survival gate, pole clamp, and parallelism gate on both sides of their thresholds; mass-law and tf materials; cross-pass invariance of the resolve decision.
- **Energy safety.** Stacked slabs against an exact transfer-matrix reference: the persist rule yields $R + T \le 1$ at every layer-thickness combination.
- **Validation contract.** Every throw of Section 7.3, including the rule that all input validation precedes the parallel region.

---

## 10. References

[1] Recommendation ITU-R P.2040-3, *Effects of building materials and structures on radiowave propagation above about 100 MHz*, International Telecommunication Union, Geneva, Aug. 2023. (Permittivity/conductivity model and eq. (9b); Fresnel coefficients, eqs. (31)–(32); attenuation distance, Section 2.2; default material table, Table 3.) https://www.itu.int/rec/R-REC-P.2040

[2] Recommendation ITU-R P.833-9, *Attenuation in vegetation*, International Telecommunication Union, Geneva, 2016. (Source of `itu_vegetation`.) https://www.itu.int/rec/R-REC-P.833

[3] 3GPP TR 38.901 V17.0.0, *Study on channel model for frequencies from 0.5 to 100 GHz*, 3rd Generation Partnership Project, Mar. 2022. (Material penetration losses, Table 7.4.3-1; source of `irr_glass` and the `att` parameter.)

[4] M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press, 1999, Section 7.6. (Multiple-beam interference in a plane-parallel plate; the Airy formulas for the reflected and transmitted ports.)

[5] G. G. Stokes, "On the perfect blackness of the central spot in Newton's rings, and on the verification of Fresnel's formulae for the intensities of reflected and refracted rays," *Cambridge and Dublin Mathematical Journal*, vol. 4, pp. 1–14, 1849. (Stokes relations $r' = -r$, $t\,t' = 1 - r^2$ used by the port decomposition.) See also M. E. Delany and E. N. Bazley, "Acoustical properties of fibrous absorbent materials," *Applied Acoustics*, vol. 3, no. 2, pp. 105–116, 1970. (Empirical porous-absorber model behind the $\varepsilon = n/z$, $\mu = n z$ calibration.)

[6] ISO 9613-1:1993, *Acoustics — Attenuation of sound during propagation outdoors — Part 1: Calculation of the absorption of sound by the atmosphere*, International Organization for Standardization, Geneva, 1993.

[7] L. L. Beranek and I. L. Vér (eds.), *Noise and Vibration Control Engineering: Principles and Applications*, 2nd ed., Wiley, 2006. (Mass law of partitions, field-incidence behavior, and the coincidence effect.)