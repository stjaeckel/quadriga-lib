# The quadriga-lib Material Model and Ray-State Machine

*A companion document to `ray_mesh_interact.cpp`*

This document describes the physical material model of quadriga-lib and the two functions that apply it: `ray_mesh_interact`, which evaluates a single ray-surface interaction, and `ray_state_update`, the batched state machine that tracks each ray's inside/outside history and overlays an analytic thin-slab (Fabry-Pérot) resolution. The model is formulated for electromagnetic (EM) propagation following Rec. ITU-R P.2040 [1] and is deliberately constructed so that the same parameter set, reinterpreted, simulates acoustic propagation with the same ray-tracing engine. All formulas are given exactly as implemented; the source code is the normative reference and this document is its explanation.

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

- **`ray_mesh_interact`** is *stateless per hit*. Given a ray segment (origin, destination), the first and second intersection points with the triangle mesh (FBS, SBS), and the material table, it classifies the hit topologically, evaluates the Fresnel interface coefficients, updates the ray geometry (reflected, transmitted, or refracted direction, plus the beam tube), and emits a per-interaction gain and polarization transfer matrix. It does not know whether the ray is currently traveling inside a material — it sees one segment at a time.

- **`ray_state_update`** is the *per-ray state machine*. It carries three small state words per ray across interactions (current medium, previous medium, a one-slot transition buffer), decides what each hit *means* in context — entry, exit, embedded face to ignore, illegal state to terminate — and corrects the per-interaction gain accordingly. On top of the ported state logic it adds a closed-form thin-slab resolution: when a ray crosses a parallel slab whose internal multiple reflections still carry significant energy, the first-order interaction is multiplied by the Airy factor $S$ so that one coefficient captures the entire internal bounce series [4], instead of relying on the tracer to follow every internal reflection as a separate ray.

The tracer calls both functions once per interaction and per physical pass: a **reflection pass** produces the reflected child ray, and a **transmission/refraction pass** produces the forward child ray. Six interaction types select the physics:

| `interaction_type` | Meaning             | Geometry   | Field model        |
|:------------------:|---------------------|------------|--------------------|
| 0                  | EM reflection       | reflection | full polarization  |
| 1                  | EM transmission     | undeviated | full polarization  |
| 2                  | EM refraction       | Snell bent | full polarization  |
| 3                  | scalar reflection   | reflection | scalar (pressure)  |
| 4                  | scalar transmission | undeviated | scalar (pressure)  |
| 5                  | scalar refraction   | Snell bent | scalar (pressure)  |

Types 0–2 are the electromagnetic modes with a $2 \times 2$ complex polarization (Jones) transfer matrix. Types 3–5 are the scalar modes used for acoustics: a single complex pressure coefficient and TE-only Fresnel physics. Types 2 and 5 bend the ray according to Snell's law (the physically correct refracted path); types 1 and 4 keep the ray undeviated, which is the standard approximation for through-wall building penetration where the two refractions of a flat slab cancel and only a small parallel offset is ignored. Total internal reflection applies uniformly to all six types (Section 3.2).

Conventions used throughout the code and this document:

- Frequencies are passed to the public API in Hz and converted once to GHz ($f = f_\mathrm{Hz} \cdot 10^{-9}$); every material formula below takes $f$ in GHz.
- Material indices are **1-based**; index 0 means "no material" (air/vacuum) and is always valid. The `Material` constructor performs the single point of index translation (`row = idx − 1`).
- `ray_offset` $= 1\,\mathrm{mm}$ is the global relaunch offset: every child ray starts $1\,\mathrm{mm}$ past its interaction point so it does not immediately re-hit the same face, and the same constant is the colocation tolerance that detects coincident faces.
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

The constant $17.98 = 1/(2\pi \varepsilon_0) \cdot 10^{-9}\,$GHz·m/S converts a conductivity in S/m into the imaginary permittivity at a frequency in GHz; this is the loss convention of Rec. ITU-R P.2040, eq. (9b) [1], with the library's $e^{+j\omega t}$ sign choice making loss a *negative* imaginary part. The magnetic columns mirror the dielectric ones exactly, so $\mu$ defaults to $1 + 0j$ and a table that omits `e`–`h` reproduces the legacy permittivity-only model.

A Lorentz resonance pole (`Material::eta_resonance`) can be added to the permittivity:

$$
\varepsilon_\mathrm{res}(f) =
\frac{S_\mathrm{res}\, f_\mathrm{res}^2}
     {f_\mathrm{res}^2 - f^2 + j\,(f_\mathrm{res}/Q_\mathrm{res})\,f},
$$

active only when $f_\mathrm{res} > 0$, $Q_\mathrm{res} > 0$, and $S_\mathrm{res} \neq 0$. The $+j$ in the denominator makes $S_\mathrm{res} > 0$ *add* loss under the negative-imaginary convention. The pole is applied to the **interface permittivity only** — it sharpens or dips the Fresnel reflection near $f_\mathrm{res}$ — and is deliberately excluded from the in-medium propagation phase (Sections 2.3 and 7.5): a strong pole can drive $\mathrm{Re}\,\varepsilon < 0$, which is meaningful for a reflection coefficient but not for a propagation constant of the bulk path. In the EM domain this models resonant dielectrics and frequency-selective surfaces; in the acoustic domain, Helmholtz and membrane absorbers (Section 6).

Derived quantities used throughout: the refractive index $n = \sqrt{\varepsilon\mu}$, the normalized admittance $Y = \sqrt{\varepsilon/\mu}$ (the code's `z`), the index ratio $(\varepsilon_1\mu_1)/(\varepsilon_2\mu_2)$ for Snell geometry, and the loss tangent of the *product* $\varepsilon\mu$ for bulk attenuation — so the magnetic loss columns `g`/`h` feed in-medium attenuation exactly as the conductivity columns `c`/`d` do.

### 2.3 In-medium propagation loss (`Material::medium_gain`)

The linear power gain of a path of length $\ell$ (meters) inside a homogeneous medium is

$$
G_\mathrm{med}(\ell, f, \cos\theta_i) = 10^{-L_\mathrm{dB}/10},
\qquad
L_\mathrm{dB} = \underbrace{\frac{8.686\,\ell}{\Delta}}_{\text{dielectric}}
 \;+\; \underbrace{\ell\,\alpha\, f_r^{\,\alpha_B}}_{\text{power-law}}
 \;+\; \underbrace{L_\mathrm{mass}(\ell, f, \cos\theta_i)}_{\text{mass law}} .
$$

The three terms, exactly as implemented:

**Dielectric term.** From the complex product $\varepsilon\mu$ (resonance excluded), with $\tan\delta = \mathrm{Im}(\varepsilon\mu)/\mathrm{Re}(\varepsilon\mu)$ and $\cos\delta = 1/\sqrt{1 + \tan^2\delta}$, the attenuation distance is

$$
\Delta = \sqrt{\frac{2\cos\delta}{1-\cos\delta}}\;
         \frac{0.0477135}{f\,\sqrt{\mathrm{Re}(\varepsilon\mu)}} \quad [\mathrm{m}],
$$

which is the attenuation-distance formula of Rec. ITU-R P.2040 §2.2 [1] generalized from $\varepsilon$ to $\varepsilon\mu$. The constant $0.0477135 = c_0/(2\pi) \cdot 10^{-9}$ m·GHz is the inverse free-space wavenumber at 1 GHz, and $8.686 = 20/\ln 10$ converts nepers to dB. The term reduces to the standard ITU result for $\mu = 1$ and vanishes for a lossless medium ($\tan\delta = 0 \Rightarrow \Delta \to \infty$).

**Power-law term.** An explicit absorption $\alpha\, f_r^{\,\alpha_B}$ in dB/m for excess loss not captured by the conductivity model (foliage, scattering media; in acoustics, calibrated bulk absorption).

**Mass-law term.** Active only when $m > 0$ and $\ell > 1.5\,\mathrm{mm}$ (paths shorter than the threshold `mass_min_path = 0.0015` are exempt, which keeps the `ray_offset`-length bookkeeping segments of Section 8 out of the logarithm):

$$
L_\mathrm{mass} = \max\!\Big(0,\; m \,\log_{10}\!\big(f_r \cdot \ell \cos^2\theta_i\big)\Big).
$$

The argument $\ell\cos^2\theta_i$ deserves a comment. For a slab of thickness $t$ traversed at incidence cosine $\cos\theta_i$ the in-medium path is $\ell = t/\cos\theta_i$, so $\ell\cos^2\theta_i = t\cos\theta_i$: the modeled loss *decreases* at oblique incidence as $m\log_{10}\cos\theta_i$, matching the field-incidence behavior of the acoustic mass law ($\approx 20\log_{10}\cos\theta_i$ for $m = 20$) [7]. When the incidence angle is unknown (the state machine's path-replacement rows), $\cos\theta_i$ defaults to 1 and the argument is the bare path. With $m = 20$ the term reproduces the classic mass law: +6 dB per octave and +6 dB per doubling of thickness; the outer $\max(0,\cdot)$ clamps the law below $f_r\,\ell\cos^2\theta_i = 1$ where the logarithm would turn into gain.

Because the mass term is logarithmic in distance, $G_\mathrm{med}$ is *not* multiplicative in $\ell$ for mass-law materials: $G_\mathrm{med}(2\ell) \neq G_\mathrm{med}(\ell)^2$. The dielectric and power-law terms are linear in $\ell$ and compose exactly. Section 7.5 discusses the (small, energy-safe) consequence for the thin-slab factor.

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

This reduces to `apply_tf` at an air boundary (the other side's $\mathrm{tf} = 0$), stays in $[0,1]$, and gives $R_\mathrm{eff} = 1$ if either face has $\mathrm{tf} = -1$. The single-owner `apply_tf` is retained only *inside* the thin-slab factor, where each mirror has exactly one solid side (Section 7.5). Using the symmetric pair at interfaces is what keeps both reflection and transmission ports of a slab energy-complementary — including the EM reflection port, which now carries the factor as well (Section 3.4).

### 2.6 Table validation and defaults

`MaterialCols` enforces, on every call that receives a material map:

- All non-empty columns have the same length $n_\mathrm{mtl}$ (sparse maps are legal; *ragged* maps are not).
- Physical sanity: $f_\mathrm{ref}, a, e > 0$ strictly; $c, g, \mathrm{att}, \alpha, m, f_\mathrm{res}, Q_\mathrm{res}, f_\mathrm{coi}, Q_\mathrm{coi} \ge 0$ (a negative loss-like value would be gain and is rejected as corrupt input rather than clamped).
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

Types 2 and 5 are the bent dielectric/acoustic paths: raw-Fresnel field power, true Snell bending, and TIR producing the *total-reflection out-codes* of Section 4.1. Under TIR the bent forward port collapses to the tf leak $1 - R_\mathrm{eff}$, which is zero when $\mathrm{tf} = 0$; in that lossless case there is no forward energy and the state machine kills the forward ray (Section 8.1), while a $\mathrm{tf} > 0$ leak travels undeviated.

---

## 4. Per-hit processing in `ray_mesh_interact`

### 4.1 Hit classification and `out_typeN`

For each ray with a valid first intersection (`fbs_ind > 0`), the face normal is computed from the triangle's winding (counter-clockwise = front face, right-hand rule), the signed grazing angle $\theta$ at the FBS decides front/back illumination, and the second intersection (SBS) refines the topology. Two faces are **colocated** when the FBS–SBS distance is below `ray_offset` ($1\,\mathrm{mm}$); their normals are compared with tolerance $10^{-4}$ per component: opposing normals mean a material-to-material (M2M) contact, equal normals an overlapping or duplicate face, anything else an edge. The resulting interaction code:

| Code | Meaning                                              | Decided by |
|:----:|------------------------------------------------------|------------|
| 1    | Single hit, outside→inside                           | $\theta \ge 0$, faces not colocated |
| 2    | Single hit, inside→outside                           | $\theta < 0$ |
| 3    | Single hit, inside→outside, total reflection         | as 2, TIR (refraction geometry, types 2, 5) |
| 4    | Media-to-media, M2 hit first                         | colocated, opposing normals, $\theta \ge 0$ |
| 5    | Media-to-media, M1 hit first                         | colocated, opposing normals, $\theta < 0$ |
| 6    | Media-to-media, M1 first, total reflection           | as 5, TIR |
| 7    | Overlapping faces, outside→inside                    | colocated, equal normals, $\theta \ge 0$ |
| 8    | Overlapping faces, inside→outside                    | colocated, equal normals, $\theta < 0$ |
| 9    | Overlapping faces, inside→outside, total reflection  | as 8, TIR |
| 10   | Edge hit, outside→inside→outside                     | edge, $\theta \ge 0$, $\theta_\mathrm{SBS} \le 0$ |
| 11   | Edge hit, inside→outside→inside                      | edge, $\theta < 0$, $\theta_\mathrm{SBS} \ge 0$ |
| 12   | Edge hit, i-o-i, total reflection                    | as 11, TIR |
| 13   | Edge hit, outside→inside                             | edge, $\theta \ge 0$, $\theta_\mathrm{SBS} \ge 0$ |
| 14   | Edge hit, inside→outside                             | edge, $\theta < 0$, $\theta_\mathrm{SBS} \le 0$ |
| 15   | Edge hit, i-o, total reflection                      | as 14, TIR |

The TIR variants (3, 6, 9, 12, 15) are emitted only on the refraction geometries (types 2 and 5, i.e. `geometry_type == 2`), where total reflection removes the propagating forward port (a $\mathrm{tf} > 0$ leak still travels undeviated, Section 3.4). Rays with `fbs_ind = 0` are omitted from the output, so the compact output set has $n_\mathrm{rayN} \le n_\mathrm{ray}$ entries; the surviving rays' input indices are reported so the caller (and `ray_state_update`) can map between the sets.

### 4.2 Material assignment

On a back-side hit the normal is flipped so the Fresnel geometry always sees the incidence side, and the face materials are assigned by orientation: on a front hit the FBS face's material is the *entered* medium $M_2$ (incidence medium $M_1$ defaults to air); on a back hit it is the *incidence* medium $M_1$. For an M2M contact the colocated partner face supplies the other medium. This is the per-hit, stateless approximation — the function cannot know the true surrounding medium of a nested geometry; `ray_state_update` corrects exactly these cases from its tracked state.

### 4.3 Direction update and relaunch

The child direction $\hat{d}$ is the mirror reflection for geometry 0, the unchanged incoming direction for the undeviated geometries, and for refraction the standard Snell construction

$$
\hat{d} = n_{12}\,\hat{u} + \big(n_{12}\cos\theta_i - \mathrm{Re}\cos\theta_t\big)\,\hat{n},
$$

normalized to unit length. Under TIR — the unified test of Section 3.2, or a ray-tube straddle (Section 4.6) — no refraction direction exists, so the refraction geometry reverts to the undeviated incoming direction. The child origin is offset $1\,\mathrm{mm}$ along $\hat{d}$ (`origN = fbs + ray_offset · d̂`), and the new destination preserves the remaining segment length. The beam tube (`trivec`/`tridir`), when present, is propagated per vertex ray through the same geometry, with degenerate vertex hits flagged through an infinite edge length.

### 4.4 In-medium attenuation at the hit

`ray_mesh_interact` charges in-medium loss for the segment *behind* the hit when the per-hit topology proves the ray was inside a body: if the ray starts inside ($\theta < 0$ or an M2M contact), the gain is multiplied by $G_\mathrm{med}$ of $M_1$ over the origin–FBS length (plus `ray_offset` on the reflection pass, whose relaunched origin sits inside the medium), evaluated at the hit's incidence cosine so the mass law sees the correct angle. Transmissive geometries additionally charge $G_\mathrm{med}$ of $M_2$ over the $1\,\mathrm{mm}$ relaunch offset, keeping the child's bookkeeping exact. These per-hit charges cover only what the stateless view can prove; the state machine owns every other in-medium segment (Section 7.6).

### 4.5 Polarization transfer and output conventions

For the EM types the TE/TM pair is embedded into the global V/H polarization frame by projecting the incoming and outgoing propagation directions onto the incidence plane: with $\hat{e}_Q$ perpendicular and $\hat{e}_P$ parallel to the plane of incidence (TE $\equiv$ H, TM $\equiv$ V in the local frame), the $2\times2$ base-change matrices $Q$ (incoming) and $U$ (outgoing) sandwich the diagonal Fresnel pair, and the amplitude $\sqrt{G_\mathrm{med}}$ scales the result. The eight output columns are the interleaved complex entries $[\mathrm{VV}\;\mathrm{HV}\;\mathrm{VH}\;\mathrm{HH}]$ (real, imaginary per entry), and the scalar gain is $G = \tfrac{1}{2}\sum |x_{ij}|^2$.

The scalar types write the single complex pressure coefficient into the first slot, $[\mathrm{Re}\;\mathrm{Im}\;0\,\cdots\,0]$, with $G = |x|^2$ (no $\tfrac{1}{2}$). Keeping the two conventions straight matters downstream: `ray_state_update` patches both arrays under the same mode-dependent convention so that the $G \leftrightarrow$ `xprmat` identity survives every operation (Section 7.4).

### 4.6 Ray-tube TIR consistency

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
                 = f_\mathrm{acoustic} \times 874{,}636
\qquad (c_\mathrm{sound} \approx 342.77\ \mathrm{m/s}),
$$

so $1\,\mathrm{kHz}$ acoustic $\equiv 0.875\,\mathrm{GHz}$ radio, and every acoustic material fixes $f_\mathrm{ref} = 0.875$. Absolute frequencies (`resF`, `coiF`) convert the same way (100 Hz → 0.0875 GHz). The analogy carries interface reflection, bulk absorption, and the mass-law/coincidence/resonance mechanisms; it does not by itself model modal interference or diffraction — the thin-slab factor of Section 7.5 restores exactly the slab-interference part. Simulation results are air-normalized (atmospheric absorption per ISO 9613-1 [6] is removed and re-applied outside).

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

with $f_\mathrm{ref} = 0.875$ and `att` = `alpha` = `m` = 0. The fit matches surface reflection and bulk attenuation simultaneously across roughly 250 Hz–4 kHz — the thing a single-$\varepsilon$ model cannot do, because it locks $Z = Z_0/\sqrt{\varepsilon}$ to $n = \sqrt{\varepsilon}$. Note where the loss lives: $\varepsilon$ stays nearly real (pore air is air), and $\mu$ carries the viscous loss of the frame-loaded effective density, which is where dissipation physically resides.

The remaining mechanisms map one-to-one: `res*` builds Helmholtz/membrane/micro-perforated absorbers (a Lorentz peak in absorption and a reflection feature at $f_\mathrm{res}$); `coi*` the coincidence dip of thin stiff panels; `m` the mass law of rigid partitions; `tf` the reflection/transmission redistribution of leaky-but-hard surfaces (perforated panels, grilles).

### 6.3 Layered constructions and known limits

Glued stacks (absorber on wall, carpet on floor) are modeled as watertight bodies with coincident faces (within the 1 mm tolerance) and opposing normals; `ray_mesh_interact` classifies the shared face as an M2M contact and the Fresnel coefficients are computed from the two real materials directly. The state machine then handles the cascade (Section 8); the thin-slab factor restores the internal interference of each layer.

Two limits are worth keeping in mind. First, the mass-law angle behavior is modeled through the $\ell\cos^2\theta_i$ argument of Section 2.3 — correct in trend, approximate in detail, and angle-blind in the few bookkeeping rows where no incidence angle is available. Second, a porous layer on a rigid backing develops a velocity node at the wall: below the quarter-wave frequency $f = c_\mathrm{sound}/(4t)$ of the layer thickness $t$, a geometric trace-through overstates absorption. The Airy resolution of Section 7.5 *is* the analytic resummation of the internal bounce series and closes most of this gap for slabs the tracer presents as such; configurations the gates re-emit fall back to the (energy-safe) geometric treatment.

---

## 7. The ray-state machine `ray_state_update`

### 7.1 Purpose

`ray_mesh_interact` reports what one hit looks like; `ray_state_update` decides what it *means*. The function is the batched, OpenMP-parallel form of the inside/outside state machine formerly embedded in `calc_diffraction_gain`: it corrects the per-interaction `gainN`/`xprmatN` in place using tracked per-ray state, writes the next state, and overlays the thin-slab resolution. The tracer calls it twice per interaction — once on the reflection pass (types 0/3) and once on the transmission/refraction pass (types 1/2/4) — and with the slab factor suppressed (every gate re-emitting) the transmission path reproduces the legacy `calc_diffraction_gain` behavior bit-for-bit, which is the port-fidelity anchor of the test suite.

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

Two index spaces coexist. Geometry (`orig`, `dest`, `fbs`, `sbs`), the hit counter `no_interact`, and the *input* state words live in the **full ray set** $[n_\mathrm{ray}]$ and are read at $g = \mathrm{ray\_ind}[i]$; the interaction outputs (`fbs_angleN`, `out_typeN`, `normal_vecN`, `mtl_ind_fbs/sbs`), the *output* state words, and `gainN`/`xprmatN` live in the **compact set** $[n_\mathrm{rayN}]$ at $i$. A null `ray_ind` means identity (then $n_\mathrm{ray} = n_\mathrm{rayN}$ is enforced). Input state is read-only; output state arrays are resized on demand.

Required unconditionally: `orig`, `dest`, `fbs`, `sbs` ($[n,3]$, equal row counts) and `out_typeN`. Optional with defaults: `no_interact` (default hit count 1), `fbs_angleN` (default 0, i.e. grazing), `normal_vecN` (absent disables the wedge test), `mtl_ind_fbs`/`mtl_ind_sbs` (default 0 = air), and `mtl_prop` itself — though any nonzero material index referenced without a table is rejected. The material map may be sparse (Section 2.6). Everything dimension-bearing is validated up front — array sizes, `ray_ind` bounds, masked material indices in the face arrays *and* in all three input state arrays, finite positive frequency, finite non-negative `eps` — because the per-ray loop runs under OpenMP where a thrown exception cannot propagate.

### 7.4 Gain patch operations

Four operations, selected per dispatch row, applied to `gainN` and `xprmatN` (either may be null; the other is still patched) under the mode convention of Section 4.5:

- **IG (keep).** The incoming interaction gain stands; nothing is touched.
- **Scale by a complex factor** $z$ (the rows written $\mathrm{IG}\cdot S$ or $\mathrm{IG}\cdot\sqrt{G}$): every complex Jones pair is multiplied by $z$ — off-diagonals included — and the gain by $|z|^2$. Isotropic medium factors enter as $z = \sqrt{G_\mathrm{med}}$ so field and power stay consistent.
- **Replace by an isotropic value** $G$ (the rows written $\mathrm{MED}(\cdot)$, discarding a spurious interaction): `xprmat` is zeroed and $\sqrt{G}$ written on the diagonal — VV and HH for EM, VV alone for scalar — and `gainN` set to $G$. The scalar layout preserves the $G = |x_\mathrm{VV}|^2$ identity.
- **KILL.** Zero everything; the ray is terminated for this pass.

### 7.5 Thin-slab resolution: the Airy factor

A parallel slab traps an infinite internal reflection series. Its closed form is the Airy sum [4]: with $r_\mathrm{near}$ the field reflection at the interface being processed (seen from inside the slab), $r_\mathrm{far}$ the reflection at the opposite interface, and $\varphi$ the one-way in-slab propagation factor,

$$
S = \frac{1}{1 - r_\mathrm{near}\, r_\mathrm{far}\, \varphi^2},
$$

so that multiplying one traced interaction by $S$ replaces the entire bounce series the tracer would otherwise have to follow. `Material::slab_airy_factor` (called on the slab's material, with the two mirror materials as arguments) computes $S$ or signals **re-emit** (NaN), in which case the dispatch keeps the bare interaction and the tracer continues geometrically — re-emission is always energy-safe.

**Mirrors.** Each $r$ is the TE Fresnel coefficient from the slab side (Section 3.2, interface permittivities with resonance), with the transmission factor folded into its magnitude and the Fresnel phase preserved: $r = \sqrt{\mathrm{apply\_tf}(|r_\mathrm{TE}|^2)}\, e^{\,j\arg r_\mathrm{TE}}$. The tf **owner** is the face-owning solid: the slab itself when the slab is a real material, the adjacent material when the cavity is air (the i-o-i air gap, whose faces belong to the bounding solids). Which materials act as mirrors depends on the call site and is listed per row in Section 8; the pattern is: the near mirror is whatever lies on the far side of the interface being processed (air at an exit, the entered material at an M2M crossing), and the far mirror is the **previous medium** from the state word — air for a free-standing slab, the neighboring layer inside a stack.

**Propagation factor.** Magnitude and phase have different sources, deliberately:

$$
|\varphi| = \sqrt{G_\mathrm{med}(L, f, \cos\theta_i)},
\qquad
\arg\varphi = -\frac{\omega}{c_0}\,\mathrm{Re}\!\left(\sqrt{\varepsilon\,\mu}\right) L,
\qquad
\omega = 2\pi f \cdot 10^9,
$$

with $L = d(\mathrm{orig}, \mathrm{fbs})$ the one-way in-slab path of the event being processed. The magnitude uses the *full* loss model (dielectric + power-law + mass, at the actual incidence cosine); the phase uses the resonance-*excluded* permittivity, since the pole belongs to the interfaces, not the bulk (Section 2.2), and the loss-only terms add no propagation phase. An air cavity yields a lossless, unit-index $\varphi$.

**Gates.** Resolution happens only when all three pass; otherwise re-emit:

1. *Parallelism.* If the entry captured both faces, the wedge test compares the FBS and SBS normals: the faces count as parallel iff $|\hat{n}_F \cdot \hat{n}_S| > 1 - 3.8\cdot10^{-3}$ (about $5^\circ$), provided the two points are distinct ($d(\mathrm{fbs},\mathrm{sbs}) > 10^{-6}$). The magnitude test is essential — at an M2M back face the coincident face pair has opposing normals and floating-point noise decides which one is reported, so a signed test would re-emit valid slabs at random. A failed test sets the KNOWN-NON-PARALLEL flag on `prev`, which forces re-emission at every later event of that traversal.
2. *Survival.* The round-trip amplitude must be worth resolving: $\rho = \sqrt{R_\mathrm{near} R_\mathrm{far}\, G_\mathrm{med}(2L)} \ge \varepsilon_\mathrm{thr}$, where the $R$ are the tf-effective power reflectances and `eps` is the caller's threshold, tied to the engine's ray-drop level ($\varepsilon_\mathrm{thr} \approx \mathrm{drop}^{1/N_\mathrm{max}}$, typically 0.1–0.25; $\varepsilon_\mathrm{thr} \ge 1$ disables resolution entirely, $0$ resolves everything the other gates allow). Using the full $G_\mathrm{med}(2L)$ keeps the gate consistent with $|\varphi|^2$ — exactly so for the dielectric and power-law terms, approximately for the logarithmic mass term, whose mismatch slightly under-damps $S$ and is bounded by the energy ledger.
3. *Pole clamp.* If $|1 - r_\mathrm{near} r_\mathrm{far} \varphi^2| < 10^{-2}$, re-emit. The Airy pole (lossless, on-resonance, near-grazing simultaneously) would otherwise produce an unbounded $|S|$; the clamp caps $|S| \lesssim 100$ and hands the case back to the tracer.

All gate inputs are functions of the slab geometry and materials only, so the reflection and transmission passes necessarily make the *same* resolve/re-emit decision for a given slab — the cross-pass invariance that prevents double-counted or lost energy.

### 7.6 The two ports and in-medium loss ownership

The reflection and transmission ports of a slab are different functions of $S$; getting both right is what closes the energy ledger. For a slab with entry interface 12 and back interface 23:

$$
T_\mathrm{slab} = t_{12}\,\varphi\,t_{23}\,S,
\qquad
R_\mathrm{slab} = r_{12} + t_{12}\,t_{21}\,r_{23}\,\varphi^2\,S .
$$

The dispatch realizes these across the traced events rather than in one place. On the transmission path, the entry event contributes $t_{12}$ (its incoming IG), the exit event contributes $t_{23}$ and is the one multiplied by $S$. On the reflection path, the front reflection stays **bare** $r_{12}$ — no $S$ — while the *internal* back reflection is multiplied by $S$ (contributing $r_{23} S$) and sets the RESOLVED flag; that ray then exits the front transparently, picking up $t_{21}$, so the product across its events is exactly the second term of $R_\mathrm{slab}$. A resolved ray reaching the front face on a *reflection* pass is killed — its would-be second bounce is already summed inside $S$.

In-medium amplitude and phase must be applied exactly once per traversed segment. The implementation's ownership rule:

- **Unresolved rows charge forward.** The entry row multiplies by $\sqrt{G_\mathrm{med}(M_1, \max(d(\mathrm{fbs},\mathrm{dest}) - \mathrm{off},\, 0))}$ — the outgoing in-slab segment, clamped continuously at the relaunch offset — and the M2M crossing charges the entered medium's forward segment the same way.
- **Resolved rows charge backward.** Every resolved-precedence row (the transparent out-coupling and the resolved internal crossing) multiplies by $\sqrt{G_\mathrm{med}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs}))}$ — its *incoming* segment.
- **The resolving reflection charges nothing.** It is the seam between the two regimes: the down-trip was charged forward by the entry, the up-trip is charged backward by the next resolved event, and re-emitted ordinary reflections keep their legacy bare IG untouched.

Following one resolved return path through a stack confirms each segment is charged exactly once, and the phase bookkeeping mirrors it: $\varphi$ supplies the in-slab phase of the resonant series inside $S$, while the tracer's own geometric path-length phase covers the explicit traversals.

### 7.7 Energy safety and stacked slabs

For a single slab the two ports are exact and $R + T + A = 1$ holds to numerical precision (lossless and lossy; verified in the test suite at normal and oblique incidence). Stacked slabs couple their cavities through higher-order terms that no per-cavity factor can represent. The implemented policy makes the truncation *strictly conservative*: the M2M cavity transition that applies $S$ also sets the RESOLVED flag (`current_out = iM | 0x8000`), so the first resolved cavity of a traversal is the only one — every later internal crossing and the final exit are transparent pass-throughs, and the reflection pass kills the resolved ray. The discarded higher-order couplings are positive-energy terms, so the residual is a benign under-count: the composed $R + T \le 1$ at every phase combination, never an energy-creating over-count. (The test suite checks this against an exact transfer-matrix reference over a grid of layer thicknesses.)

---

## 8. Dispatch tables

This section lists every state-machine row as implemented. Notation:

- `cur` = `current_in & 0x7FFF`, `resolved` = flag of `current_in`; `buf` = buffer material; `prev` = previous material, `nonpar` = flag of `prev_in`. $M_1$ = `mtl_ind_fbs`, $M_2$ = `mtl_ind_sbs`; both masked. `nH` = `no_interact` (hits on the segment), `type` = `out_typeN` (Section 4.1). $d(p,q)$ = Euclidean distance of the full-set rows; `off` = `ray_offset` = 1 mm.
- $\mathrm{MED}(m, \ell) = G_\mathrm{med}$ of material $m$ over length $\ell$ (index 0 → 1). $\mathrm{TRN}(a, b)$ = the energy-conserving transmission gain for a crossing *from* $a$ *into* $b$ (`interact_with` with type 1, or 4 in scalar mode, at the hit's signed angle — the tf owner follows the hit side per Section 3.3). $\mathrm{IG}$ = keep; $S(\mathrm{slab}; n, f)$ = Airy factor of the slab material with near mirror $n$ and far mirror $f$; $\mathrm{KILL}$ = terminate.
- A gain written $\mathrm{IG}\cdot X$ scales the kept interaction; a gain written without IG replaces it (Section 7.4). State words default to copy-through unless a cell writes them. "+wedge" means: run the parallelism test of Section 7.5 and write its outcome (clear = may resolve) into the `prev` flag, resetting `prev` to 0 in the same write.
- Every $S$ row falls back to $\mathrm{IG}$ (keep, no flag change beyond the listed state writes) when the gates re-emit.

### 8.1 Pass selection and precedence

```
REFLECTION pass (types 0, 3):
    resolved            -> KILL                  (second bounce already inside S)
    cur == 0            -> IG, copy-through      (order-0 front reflection, bare Fresnel)
    cur != 0            -> internal-reflection row (8.2)

TRANSMISSION / REFRACTION pass (types 1, 2, 4):
    type in {3,6,9,12,15} -> KILL                (total reflection: no forward port; wins over resolved)
    resolved              -> resolved-precedence rows (8.3)
    nH == 0               -> IG, copy-through    (segment not processed; caller owns its loss)
    otherwise             -> topology rows (8.4 - 8.10) by (type, nH, state)
    unmatched             -> KILL                (global default)
```

### 8.2 Reflection pass, internal reflection (`cur != 0`, not resolved)

The interface being processed is the back of the current slab: against air for an i-o topology, against the adjacent medium for an i-i (M2M) topology.

| Condition | Gain | State out |
|---|---|---|
| gates pass | $\mathrm{IG} \cdot S\big(\mathrm{cur};\; n,\; \mathrm{prev}\big)$ with $n = M_2$ for type 5, $M_1$ for type 4, else air | set RESOLVED flag on `current_out` |
| gates re-emit | $\mathrm{IG}$ (ordinary reflection) | copy-through |

One-way path $L = d(\mathrm{orig}, \mathrm{fbs})$. No in-medium charge here (the seam, Section 7.6).

### 8.3 Resolved-ray precedence (transmission pass)

$iM = M_2$ for type 5, else $M_1$. All rows charge the incoming segment (Section 7.6).

| Topology | Gain | State out |
|---|---|---|
| i-o (2, 8, 14): out-coupling | $\mathrm{IG} \cdot \sqrt{\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{fbs}))}$ | `current_out <- 0` (flag cleared) |
| i-i (4, 5): resolved crossing | $\mathrm{IG} \cdot \sqrt{\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{fbs}))}$ | `current_out <- iM \| 0x8000` (keep flag), `prev_out <- cur` |
| other (o-i, edges) | $\mathrm{IG}$ (transparent pass-through) | copy-through |

### 8.4 o-i entry family — `(nH 1, type 1)`, `(nH 2, type 7)`, `(nH 2, type 13)`

| State | Gain | State out |
|---|---|---|
| `cur == 0` (enter) | $\mathrm{IG} \cdot \sqrt{\mathrm{MED}(M_1,\, \max(d(\mathrm{fbs},\mathrm{dest}) - \mathrm{off},\, 0))}$ | `current_out <- M1`, `prev_out <- 0` +wedge |
| `cur != 0` (nested body) | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{dest}))$ | `buffer_out <- M1` |

The entry charge is the forward in-slab segment; the clamp is continuous at $d = \mathrm{off}$. The nested row discards the spurious interaction (the inner face is embedded in `cur`) and replaces it with pure medium loss over the whole segment, deferring the transition to the buffer.

### 8.5 i-o exit family — `(nH 1, type 2)`, `(nH 2, type 8)`, `(nH 2, type 14)`

| State | Gain | State out |
|---|---|---|
| `cur == 0` (false inside) | $\mathrm{IG}$ | copy-through |
| `buf == 0` (cavity exit) | $\mathrm{IG} \cdot S\big(\mathrm{cur};\; \mathrm{air},\; \mathrm{prev}\big)$ | `current_out <- 0` |
| nH 1 type 2, `buf != 0`, $\mathrm{buf} \sim M_1$ | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{dest}))$ | `buffer_out <- 0` |
| nH 1 type 2, `buf != 0`, else (virtual i-i) | $\mathrm{MED}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs})) \cdot \mathrm{TRN}(\mathrm{cur},\mathrm{buf}) \cdot \mathrm{MED}(\mathrm{buf}, d(\mathrm{fbs},\mathrm{dest}))$ | `current_out <- buf`, `buffer_out <- 0` |
| nH 2 types 8/14, `buf != 0` (ii-oo) | $\mathrm{MED}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs})) \cdot \mathrm{TRN}(\mathrm{cur}, 0)$ | `current_out <- 0`, `buffer_out <- 0` |

$\mathrm{buf} \sim M_1$ denotes the same-medium test: equal indices, or two real materials with identical parameter rows. The cavity exit's $S$ uses $L = d(\mathrm{orig},\mathrm{fbs})$, air as the near mirror (the outside), and `prev` as the far mirror — air for a free-standing slab, the neighboring layer after an M2M re-emit inside a stack.

### 8.6 Slab entry capturing both faces — `(nH 2, type 1)` o-i-o

| State | Gain | State out |
|---|---|---|
| `cur == 0` | $\mathrm{IG}$ (bare entry) | `current_out <- M1`, `prev_out <- 0` +wedge |
| `cur != 0` (nested) | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{fbs}))$ | `buffer_out <- M1` |

### 8.7 Air gap — `(nH 2, type 2)` i-o-i

| State | Gain | State out |
|---|---|---|
| `buf == 0`, $M_2 = 0$ (illegal) | $\mathrm{KILL}$ | — |
| `buf == 0`, $M_2 \neq 0$ (gap exit) | $\mathrm{IG} \cdot S\big(\mathrm{air};\; M_1,\; M_2\big)$ | `current_out <- 0` |
| `cur != 0`, `buf != 0`, $\mathrm{buf} \sim M_1$ | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{fbs}) + \mathrm{off})$ | `buffer_out <- 0` |
| `cur != 0`, `buf != 0`, else | $\mathrm{MED}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs})) \cdot \mathrm{TRN}(\mathrm{cur},\mathrm{buf}) \cdot \mathrm{MED}(\mathrm{buf}, \mathrm{off})$ | `current_out <- buf`, `buffer_out <- 0` |
| `cur == 0`, `buf != 0` | $\mathrm{KILL}$ | — |

The gap row is the one place the cavity is air: the mirrors are the bounding solids $M_1$ (near) and $M_2$ (far), whose tf the air cavity correctly defers to (Section 7.5).

### 8.8 Material-to-material crossing — `(nH 2, types 4/5)`; $iM = M_2$ for type 5, else $M_1$

| State | Gain | State out |
|---|---|---|
| `cur == 0` (illegal) | $\mathrm{KILL}$ | — |
| `buf == 0`, $M_1 = 0$ or $M_2 = 0$ (illegal) | $\mathrm{KILL}$ | — |
| `buf == 0` (cavity transition), gates pass | $\mathrm{IG} \cdot S\big(\mathrm{cur};\; iM,\; \mathrm{prev}\big) \cdot \sqrt{\mathrm{MED}(iM,\, d(\mathrm{fbs},\mathrm{dest}) - \mathrm{off})}$ | `current_out <- iM \| 0x8000` (RESOLVED), `prev_out <- cur` |
| `buf == 0`, gates re-emit | $\mathrm{IG} \cdot \sqrt{\mathrm{MED}(iM,\, d(\mathrm{fbs},\mathrm{dest}) - \mathrm{off})}$ | `current_out <- iM`, `prev_out <- cur` |
| `buf != 0` (deferred transition pending) | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{dest}))$ | `buffer_out <-` ($\mathrm{buf} \sim M_1$ ? $M_2$ : $M_1$) |

The forward charge on the entered medium is deliberately unclamped (the M2M geometry guarantees $d(\mathrm{fbs},\mathrm{dest}) > \mathrm{off}$ for a real crossing). Setting the RESOLVED flag here is the stacked-slab persist rule of Section 7.7.

### 8.9 Edge hits — `(nH 2, type 10)` o-i-o and `(nH 2, type 11)` i-o-i

Edges are grazing contacts, not slabs: no row applies $S$ and no wedge flag is written.

Type 10:

| State | Gain | State out |
|---|---|---|
| `cur == 0` | $\mathrm{IG}$ | `current_out <- 0` |
| `cur != 0`, $M_1 \sim M_2$ | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{dest}))$ | copy-through |
| `cur != 0`, else | $\mathrm{MED}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs})) \cdot \mathrm{TRN}(\mathrm{cur}, M_1) \cdot \mathrm{MED}(M_1, d(\mathrm{fbs},\mathrm{dest}))$ | `current_out <- M1` |

Type 11:

| State | Gain | State out |
|---|---|---|
| `cur == 0` | $\mathrm{IG}$ | `current_out <-` ($d(\mathrm{fbs},\mathrm{sbs}) > 10^{-6}$ ? $M_2$ : 0) |
| `cur != 0`, $M_1 \sim M_2$ | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{dest}))$ | copy-through |
| `cur != 0`, else | $\mathrm{MED}(M_2,\, d(\mathrm{fbs},\mathrm{dest}) - \mathrm{off})$ | `current_out <- M2` |

### 8.10 Multi-hit segments — `nH > 2`

Outside (`cur == 0`):

| Condition | Gain | State out |
|---|---|---|
| `buf != 0` | $\mathrm{KILL}$ | — |
| type 1/7 | $\mathrm{IG}$ | `current_out <- M1`, `prev_out <- 0` +wedge |
| type 13 | $\mathrm{IG}$ | `current_out <- M1`, `buffer_out <- M2`, `prev_out <- 0` +wedge |
| type 2 (false inside) | $\mathrm{IG}$ | copy-through |
| type 10 (stay outside) | $\mathrm{IG}$ | copy-through |
| other | $\mathrm{KILL}$ | — |

Inside (`cur != 0`):

| Condition | Gain | State out |
|---|---|---|
| type 1/7/13 (nested) | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{fbs}) + \mathrm{off})$ | `buffer_out <- M1` |
| type 2/14, `buf == 0` (cavity exit) | $\mathrm{IG} \cdot S(\mathrm{cur};\, \mathrm{air},\, \mathrm{prev})$ | `current_out <- 0` |
| type 2/14, $\mathrm{buf} \sim M_1$ | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{fbs}) + \mathrm{off})$ | `buffer_out <- 0` |
| type 2/14, else | $\mathrm{MED}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs})) \cdot \mathrm{TRN}(\mathrm{cur},\mathrm{buf}) \cdot \mathrm{MED}(\mathrm{buf}, \mathrm{off})$ | `current_out <- buf`, `buffer_out <- 0` |
| type 4/5, `buf != 0` (spurious) | $\mathrm{IG}$ | `buffer_out <- 0` |
| type 4/5, `buf == 0` | as 8.8 cavity transition, without the forward $\mathrm{MED}$ | `current_out <- iM \| 0x8000` / `iM`, `prev_out <- cur` |
| type 8, `buf == 0` | $\mathrm{IG} \cdot S(\mathrm{cur};\, \mathrm{air},\, \mathrm{prev})$ | `current_out <- 0` |
| type 8, `buf != 0` | $\mathrm{MED}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs})) \cdot \mathrm{TRN}(\mathrm{cur}, 0)$ | `current_out <- 0`, `buffer_out <- 0` |
| type 10, `buf == 0`, $M_1 \sim M_2$ | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{fbs}) + \mathrm{off})$ | copy-through |
| type 10, `buf == 0`, else | $\mathrm{MED}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs})) \cdot \mathrm{TRN}(\mathrm{cur}, M_1) \cdot \mathrm{MED}(M_1, \mathrm{off})$ | `current_out <- M1` |
| type 10, `buf != 0`, $\mathrm{buf} \sim M_1$ | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{fbs}) + \mathrm{off})$ | `buffer_out <- 0` |
| type 10, `buf != 0`, else | $\mathrm{MED}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs})) \cdot \mathrm{TRN}(\mathrm{cur},\mathrm{buf}) \cdot \mathrm{MED}(\mathrm{buf}, \mathrm{off})$ | `current_out <- buf`, `buffer_out <- 0` |
| type 11, `buf == 0`, $M_1 \sim M_2$ | $\mathrm{MED}(\mathrm{cur},\, d(\mathrm{orig},\mathrm{fbs}) + \mathrm{off})$ | copy-through |
| type 11, `buf == 0`, else | $\mathrm{MED}(\mathrm{cur}, d(\mathrm{orig},\mathrm{fbs})) \cdot \mathrm{TRN}(\mathrm{cur}, M_2) \cdot \mathrm{MED}(M_2, \mathrm{off})$ | `current_out <- M2` |
| type 11, `buf != 0` (spurious) | $\mathrm{IG}$ | `buffer_out <- 0` |
| other (TR remnants, degenerate type 0) | $\mathrm{KILL}$ | — |

Any `(type, nH, state)` combination not matched by these tables is killed — the unified global default that replaces the legacy silent pass-through, on the principle that an inconsistent state should remove energy rather than invent it.

---

## 9. Validation

The behavior above is pinned by a blind Catch2 suite (`test_ray_state_update.cpp`, 28 cases, roughly 1,800 assertions) written against the specification with independently derived oracles:

- **Dispatch and state.** Every row of Section 8, the state encoding, flag persistence, `ray_ind` mapping, batch determinism, and `float`/`double` parity.
- **Energy ledger.** For a single slab, $R + T = 1$ (lossless) and $R + T + A = 1$ (strongly lossy, at normal and oblique incidence), with $R$ and $T$ assembled from the two passes exactly as a tracer would. This is the primary guard on the port decomposition, the Stokes/tf consistency, and the loss-ownership rule.
- **Slab physics.** $S$ against the closed-form Airy sum over a phase sweep; the survival gate, pole clamp, and parallelism gate on both sides of their thresholds; mass-law and tf materials; cross-pass invariance of the resolve decision.
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