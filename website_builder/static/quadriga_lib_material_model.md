# The quadriga-lib Material Model and Ray-Surface Physics

*A companion to `src/ray_mesh_interact.cpp` and to the API documentation of its five public functions*

This page describes the material model of quadriga-lib and the physics of a ray meeting a surface: the material columns and their units, the formulas as implemented, how faces, thin sheets and traced solids are evaluated, what the tracer's state machine adds, how the model is used for radio waves and for sound, the rules of use and the limits. The model is formulated for electromagnetic (EM) propagation following Rec. ITU-R P.2040 [1]; with a mapping of the columns the same machine traces sound. The source code is the normative reference: where this page and the code differ, the code is right. Part II at the end is a map of the source for developers.

**Contents**

Part I — Using the model

1. [What the model is](#1-what-the-model-is)
2. [Scene and mesh conventions](#2-scene-and-mesh-conventions)
3. [Interaction types and the two passes](#3-interaction-types-and-the-two-passes)
4. [The material columns](#4-the-material-columns)
5. [The formulas](#5-the-formulas)
6. [Interface physics and energy](#6-interface-physics-and-energy)
7. [Sheets, stacks and traced solids](#7-sheets-stacks-and-traced-solids)
8. [What the state machine adds](#8-what-the-state-machine-adds)
9. [Electromagnetic use](#9-electromagnetic-use)
10. [Acoustic use](#10-acoustic-use)
11. [Acoustic worked examples](#11-acoustic-worked-examples)
12. [Rules of use](#12-rules-of-use)
13. [Limits](#13-limits)
14. [References](#14-references)

Part II — Developer notes

15. [A map of the source file](#15-a-map-of-the-source-file)
16. [How a change is checked](#16-how-a-change-is-checked)

---

# Part I — Using the model

## 1. What the model is

Every face of the mesh separates homogeneous, isotropic media. A medium is one row of the material table: a complex relative permittivity $\varepsilon$, a complex relative permeability $\mu$, loss and lumped columns, and optionally a thickness that turns its faces into thin sheets. Five public functions apply the model:

| function | what it computes |
| --- | --- |
| `medium_gain` | the power gain of a path of given length through one material: the conductivity and `alpha` loss, and for a mass-law material its mass term at normal incidence |
| `interface_gain` | the lumped penetration gain `att` of entering one material |
| `refractive_index` | $\mathrm{Re}\sqrt{\varepsilon\mu}$ of one material |
| `ray_mesh_interact` | one face event per ray: the hit geometry, the interface coefficients (Fresnel, or the composite of a thin sheet), the outgoing direction, the relaunch segment and the ray tube; it sees faces only and assumes air on every side the mesh does not name |
| `ray_state_update` | the per-ray medium state: it corrects what `ray_mesh_interact` wrote with the medium the ray is known to be in, closes the leg the ray leaves (loss and phase), sums the bounce series of a layer where it is strong enough, re-resolves a sheet against the tracked media, and writes the state for the next event |

The tracer `ray_progress` (and `calc_diffraction_gain`) calls `ray_mesh_interact` and then `ray_state_update` twice per event; `ray_commit` uses `medium_gain` for the last leg to a receiver inside a medium. Frequencies are given to the API in Hz; the frequency columns (`fRef`, `resF`, `coiF`, `tfF`) are in GHz. Lengths are in m. Material indices are 1-based; 0 is air. All arithmetic is double precision; `float` is a storage type of inputs and outputs only.

## 2. Scene and mesh conventions

- **Face orientation.** A face is hit on its front side when its vertices are counter-clockwise as seen from the ray (right-hand rule). The face angle $\theta = \arccos(\hat u\cdot\hat n) - \pi/2$ (`fbs_angleN`) is the angle to the face, $\geq 0$ for a front-side hit; $\lvert\cos(\theta+\pi/2)\rvert$ is the cosine to the normal. A front-side hit enters the body that owns the face, a back-side hit leaves it.
- **Traced solids** are closed bodies of a material with `thk = 0` (or no `thk` column); the tracer follows the ray through their interior.
- **Sheets** are faces of a material with `thk > 0`: each face stands for a layer of that thickness, and its internal multiple-reflection series is summed in one event. A closed body of a sheet material is a hollow box.
- **Co-located faces.** Two hits closer than 1 mm along the ray are one event. Opposing normals of two solids are a medium-to-medium boundary (two solids glued); a sheet co-located with a solid face is glued to it, on the incidence side; two sheets of different materials are an ordered stack (the layer hit on its front side first; at most two layers); two coincident sheets of one material are one sheet. Equal normals of two solids are overlapping geometry, and non-parallel faces a corner.
- **Air is assumed** on every side the mesh does not name. `ray_mesh_interact` knows only faces; `ray_state_update` knows the medium a ray is in and corrects the event.
- **The relaunch point** lies 8 float ulp of the hit point's largest coordinate (at least of 1 m) past the hit point, about 1 µm per meter of coordinate magnitude; past the second face of a co-located pair for a transmission. The in-medium ledger counts this offset, so a traced leg is measured from the hit point. Scenes belong in local coordinates near the origin (Section 12).

## 3. Interaction types and the two passes

| `interaction_type` | field | the outgoing ray |
| :---: | --- | --- |
| 0 | EM, TE and TM (2 × 2 Jones matrix) | mirror reflection |
| 1 | EM | transmission, the direction kept |
| 2 | EM | refraction, the direction bent by Snell's law |
| 3 | scalar (one pressure coefficient) | mirror reflection |
| 4 | scalar | transmission, the direction kept |
| 5 | scalar | refraction, the direction bent |

The tracer runs two passes per event: the reflect pass (type 0 or 3) and the transmit pass (1 or 2, 4 or 5), each a call of `ray_mesh_interact` and one of `ray_state_update` with the outputs of that type. The scalar types run the same physics in the TE / admittance form; their coefficient is stored as `[Re Im]` and their gain is $\lvert c\rvert^2$. The EM types store the Jones matrix as `[ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]` and their gain is $\tfrac12\sum\lvert c_{ij}\rvert^2$.

Types 1 and 4 keep the segment on the incoming direction while `path_dirN` reports the Snell direction; types 2 and 5 bend the segment with the Snell ratio $\sqrt{\lvert\varepsilon_1\mu_1/(\varepsilon_2\mu_2)\rvert}$. Total reflection occurs only towards a lossless medium (Section 6.2). With a ray tube and types 2 or 5, a tube with any vertex past the critical angle passes undeviated as a whole and keeps the coefficient of its spine. No ray bends into a mass-law medium (Section 6.5).

## 4. The material columns

The material table is a map from column name to one value per material (the `csv_prop` output of `obj_file_read`). Any column may be absent; an absent column takes its default for every material. Columns the model does not know are ignored; an old `coiA` column is read and ignored.

| column | unit | default | meaning | acts on |
| --- | --- | --- | --- | --- |
| `fRef` | GHz | 1 | reference frequency of every power law; of a mass-law material it also sets the surface mass | all |
| `a` | — | 1 | $\varepsilon'$ at `fRef` | all |
| `b` | — | 0 | frequency exponent of `a` | all |
| `c` | S/m | 0 | conductivity σ at `fRef` | all |
| `d` | — | 0 | frequency exponent of `c` | all |
| `e` | — | 1 | $\mu'$ at `fRef` | all |
| `f` | — | 0 | frequency exponent of `e` | all |
| `g` | S/m | 0 | magnetic loss at `fRef` | all |
| `h` | — | 0 | frequency exponent of `g` | all |
| `thk` | m | 0 | a sheet of this thickness where > 0; 0 a traced solid | all |
| `att` | dB | 0 | penetration loss once per entry, at `fRef` | all |
| `attB` | — | 0 | frequency exponent of `att` | all |
| `alpha` | dB/m | 0 | absorption along the path, at `fRef` | all |
| `alphaB` | — | 0 | frequency exponent of `alpha` | all |
| `m` | — | 0 | the mass column: > 0 makes the material a lumped plate with the mass law (any positive value) | switch |
| `resF` | GHz | 0 | resonance frequency: a pole of $\varepsilon$ (wave media); the plate's resonance (mass-law) | both, differently |
| `resQ` | — | 0 | quality factor of that resonance | both |
| `resS` | — | 0 | strength of the pole (wave media); its sign selects the form (mass-law) | both |
| `coiF` | GHz | 0 | coincidence frequency, the plate's bending stiffness | mass-law |
| `coiQ` | — | 0 | 1 / loss factor of the bending | mass-law |
| `tf` | — | 0 | transmission factor of the faces, at `fRef` | all |
| `tfB` | — | 0 | frequency exponent of `tf` | all |
| `tfP` | — | 0 | height of a resonant peak added to `tf` | all (a series impedance on mass-law faces) |
| `tfF` | GHz | 0 | frequency of that peak | all |
| `tfQ` | — | 0 | quality factor of that peak; negative selects the under-damped branch | all |
| `side` | m | 3.29 | panel size √(area) of a mass-law material; 0 the infinite plate | mass-law |

"Wave media" are materials without the mass column; "mass-law" materials have `m > 0` (Section 6.5). The table is checked when it is read: all non-empty columns have one length; `fRef`, `a`, `e` are positive; `c`, `g`, `thk`, `att`, `alpha`, `m`, `resF`, `resQ`, `coiF`, `coiQ`, `tfF`, `side` are non-negative (`tfQ` may be negative); a violation throws `std::invalid_argument` naming the column and the 1-based material. `ray_state_update` takes at most 16383 materials.

## 5. The formulas

With $f$ in GHz and $r = f/f_\text{Ref}$:

### 5.1 Permittivity, permeability and the resonance

$$
\varepsilon(f) = a\,r^{b} - j\,\frac{17.98\,\sigma}{f} + \varepsilon_\text{pole},\qquad \sigma = c\,r^{d},\qquad
\mu(f) = e\,r^{f_\mu} - j\,\frac{17.98\,g\,r^{h}}{f}
$$

($f_\mu$ is the column `f`.) Loss is a negative imaginary part (time dependence $e^{+j\omega t}$, P.2040 eq. 9b [1]). The pole, for a material without the mass column and with `resF > 0`, `resQ > 0`, `resS ≠ 0`, is

$$
\varepsilon_\text{pole} = \frac{S\,F^2}{F^2 - f^2 + j\,(F/Q)\,f}\qquad (F = \texttt{resF},\ Q = \texttt{resQ},\ S = \texttt{resS})
$$

and it is part of the medium everywhere: at its faces, in its index, its loss and `refractive_index`. Each medium has the index $n = \sqrt{\varepsilon\mu}$ and the admittance $z = \sqrt{\varepsilon/\mu}$ (relative to air; `Material::eta`, `Material::mu`).

### 5.2 Loss in a medium and at entry

The power gain of a path of length $d$ in a medium (`Material::medium_gain`) is $10^{-L/10}$ with

$$
L = d\left(8.686\,k_0\,\lvert\mathrm{Im}\,n\rvert + \alpha\,r^{\alpha_B}\right) + L_\text{mass},\qquad k_0 = \frac{2\pi\cdot 10^9 f}{c_0},
$$

the ITU-R P.2040 attenuation rate plus the `alpha` column ($\alpha_B$ = `alphaB`); $L_\text{mass}$ is the mass term of a mass-law material (5.4). The penetration gain $10^{-\texttt{att}\,r^{\texttt{attB}}/10}$ (`interface_gain`) is paid once where a medium is entered, on the transmit port; a sheet's reflect port pays its layer's `att` once on what entered the layer and came back. It is never paid on exit and never per meter.

### 5.3 The transmission factor

$$
t_f(f) = \mathrm{clamp}\!\left(\texttt{tf}\,r^{\texttt{tfB}} + \frac{\texttt{tfP}}{1 + \left(\texttt{tfQ}\,(f/\texttt{tfF} - \texttt{tfF}/f)\right)^2},\ -1,\ 1\right)
$$

(`Material::tf`; the peak where `tfP ≠ 0` and `tfF > 0`, symmetric in log frequency, `|tfQ|` its Q.) A positive $t_f$ leaks reflected power into transmission, a negative one forces reflection; Section 6.3 says how a face applies it.

### 5.4 The mass term

A material with `m > 0` is a lumped plate. Its transmission is formed from the impedance ratio of a body of normal thickness $M$ at the cosine $c$ the wave has in air (`Material::mass_X0`):

$$
X_0 = r\,M\cdot\rho_\text{res}\cdot\kappa_\text{coi},\qquad
\kappa_\text{coi} = \left\lvert 1 - w\,(1 + j/\texttt{coiQ})\right\rvert,\quad w = (f/\texttt{coiF})^2\,(1-c^2)^2,
$$

with $\rho_\text{res} = \min\!\left(1, \lvert 1 - x^2 + j\,x/\texttt{resQ}\rvert\right)$, $x = f c/\texttt{resF}$ for `resS > 0` (a mass-air-mass resonance, the dip only) and $\rho_\text{res} = \lvert 1 - y^2 - j\,y/\texttt{resQ}\rvert$, $y = \texttt{resF}/f$ for `resS < 0` (the panel's first mode, a stiffness below it); each factor is 1 where its columns are 0. With $\texttt{fRef} = (f_\text{GHz}/f_\text{Hz})\,\rho_0 c/(\pi\rho)$ for a material of density $\rho$, $X_0 = \omega m''/(2\rho_0 c)$ is the impedance ratio of a limp panel of surface mass $m''$, and the infinite plate below is the limp mass law [6, 7].

The term in dB (`mass_TL`) with $A = 20\log_{10}X_0$, zero where $X_0 \leq 1$:

$$
\text{infinite plate } (\texttt{side} = 0):\quad L_\text{mass} = A\,\frac{\log_{10}\!\left(1 + (X_0 c)^2\right)}{\log_{10}\!\left(1 + X_0^2\right)}
$$

$$
\text{finite panel } (\texttt{side} > 0):\quad L_\text{mass} = \max\!\left(0,\ A\,\frac{\log_{10}\!\left(1 + (X_0/\sigma)^2\right)}{\log_{10}\!\left(1 + X_0^2\right)} + 10\log_{10}(\sigma c)\right)
$$

The infinite plate gives $A$ at normal incidence and nothing at grazing; the finite panel replaces its radiation $1/c$ by Davy's forced radiation efficiency $\sigma$ of a square of side `side` [5] (`davy_sigma`; $k$ the free-space wavenumber of $f$, $\kappa = \max(k\,\texttt{side}/2,\ 0.1)$):

$$
p = \min\!\left(1,\ 1.3\sqrt{\pi/(2\kappa)}\right),\quad h = \frac{1}{\tfrac23\sqrt{2\kappa/\pi} - 0.124},\quad
q = \frac{2\pi}{k^2\,\texttt{side}^2},\quad x = \begin{cases} c & c \geq p\\ h - (h/p - 1)\,c & c < p\end{cases},\quad
\sigma = \frac{1}{\sqrt{x^2 + q^2}}
$$

A traced solid carries the term on a leg longer than 1.5 mm, with $M$ its normal thickness; a sheet carries it at any thickness with $M$ = `thk`. The public `medium_gain` evaluates it at $c = 1$.

**Bonded bodies are one plate.** Mass-law bodies within the co-location distance move with one velocity: their $X_0$ add. The first body pays its term, each later one $L_\text{mass}(X_\text{run} + X_0) - L_\text{mass}(X_\text{run})$, where $X_\text{run}$ is the summed $X_0$ crossed before it; a leg that begins at a reflection starts a new run. A layer without the mass column between two mass-law bodies separates them (their terms then add in dB).

## 6. Interface physics and energy

### 6.1 Fresnel coefficients and the energy convention

`interact_with(A, B, type, …)` is the face between the medium the wave is in ($A$, index 1) and the next one ($B$, index 2). Every cosine follows from the transverse invariant $\varepsilon\mu\,(1 - c^2)$, conserved across every face: $c_2 = \sqrt{1 - (\varepsilon_1\mu_1/\varepsilon_2\mu_2)(1 - c_1^2)}$ on the decaying branch. In admittances (P.2040 eqs. 31, 32 [1]):

$$
r_\text{TE} = \frac{z_1c_1 - z_2c_2}{z_1c_1 + z_2c_2},\quad t_\text{TE} = \frac{2z_1c_1}{z_1c_1 + z_2c_2},\qquad
r_\text{TM} = \frac{z_2c_1 - z_1c_2}{z_2c_1 + z_1c_2},\quad t_\text{TM} = \frac{2z_1c_1}{z_2c_1 + z_1c_2}
$$

The scalar types use the TE form with the pressure transmission $1 + r$. A transmitted coefficient is a per-channel normal-flux amplitude (`flux_amp`): the Fresnel amplitude times $\sqrt{\mathrm{Re}(z_2c_2)/\mathrm{Re}(z_1c_1)}$ (TE, scalar) or $\sqrt{\mathrm{Re}(z_2^{\ast}c_2)/\mathrm{Re}(z_1^{\ast}c_1)}$ (TM), times $\sqrt{G_\text{att}}$ of the entered medium. Seen from a lossless medium the two ports of a face add to one per channel (without `tf` and `att`), at any angle, and the four transmit types carry the same energy.

### 6.2 Total reflection

A face reflects totally where $\sqrt{\lvert\varepsilon_1\mu_1/\varepsilon_2\mu_2\rvert}\,\sin\theta_1 \geq 1$ and the far medium is lossless ($\mathrm{Im}(\varepsilon_2\mu_2) = 0$; a resonance pole makes a medium lossy). A lossy far medium has no critical angle at any frequency: the port is the Fresnel crossing and the direction its real projection. Without `tf` a closed face passes nothing; with `tf` it passes the leak of its pad (6.3).

### 6.3 `tf` at a face: transformer and pad

Both rules rest on the pair rule of the two media's factors (`apply_tf_pair`), with $t^{+} = \max(t_f, 0)$, $t^{-} = \max(-t_f, 0)$ on each side:

$$
R_\text{leak} = R_0\,(1 - t_A^{+})(1 - t_B^{+}),\qquad R' = R_\text{leak} + (1 - R_\text{leak})\max(t_A^{-}, t_B^{-}).
$$

- **An open face** (a wave on both sides) is an ideal transformer of real ratio (`tf_face`): with $u = (1 - r)/(1 + r)$, $x_0 = \lvert u\rvert$, the pair rule on $r_m = (1 - x_0)/(1 + x_0)$ gives $R_m$ and $s = \pm\sqrt{R_m}$; the face reflects $r' = \frac{(1+s) - (u/x_0)(1-s)}{(1+s) + (u/x_0)(1-s)}$ and scales both crossings by $\sqrt{(1 - R_m)/x_0}\,(1 + u)/D$, $D$ the denominator of $r'$. The normal flux is continuous: the face creates nothing, lossy or not, and a sheet and its traced solid are one wall. Where neither side has `tf`, the face is the raw Fresnel face.
- **A closed face** (no wave behind it) is a matched resistive pad in front of the reactance behind (`closed_face`): $s = \sqrt{R'(R_0{=}1)}$, $\nu = (1 - s)/(1 + s)$, $e = y/\lvert y\rvert$ for the immittance $y$ of the wave's side, $b = j\,\mathrm{Im}\!\left(e\,(1 - r)/(1 + r)\right)$, $w = (\nu + b)/(1 + \nu b)$, $r' = (e - w)/(e + w)$; it passes $\lvert 1 + r'\rvert\sqrt{\mathrm{Re}(w)\,\lvert y\rvert/\Phi_\text{in}}$ of the incident flux $\Phi_\text{in}$. In front of a lossless medium this is $\sqrt{R'}$ on the reflection and $1 - R'$ of the flux, flat in angle and undeviated, with no step at the critical angle.

A rigid face without a peak therefore reflects $R = 1 - t_f$ at every angle. `tf = −1` closes the transmit port.

### 6.4 A peaked mass-law face

Where a mass-law face carries a positive peak (`m > 0`, `tfP > 0`, `tfF > 0`), the peak is the series impedance per $Z_0$ of a resonator (`Material::zeta`):

$$
\zeta = R' + j\,(1 + R')\,\lvert\texttt{tfQ}\rvert\,(f/\texttt{tfF} - \texttt{tfF}/f),\qquad
\texttt{tfP} = \frac{4R'}{(1 + R')^2}\ \ (R' \geq 1;\ 1/R' \text{ where } \texttt{tfQ} < 0;\ \texttt{tfP} > 1 \text{ is } R' = 1)
$$

met against the immittance of the side the wave is on ($\zeta c$ for TE and the scalar types, $\zeta/c$ for TM), behind the pad of the power-law part. At normal incidence in air the face absorbs the peak law of 5.3 together with the power-law part; its absorption depends on the angle like that of a locally reacting face, and its reflection carries the impedance's phase. A wave medium with a peak keeps the transformer.

### 6.5 The held wave and the plate rule

A mass-law material has no wave inside. Its $\varepsilon$, $\mu$ are an impedance mapping for its faces: they set the reflection of its faces and their critical angle, never a propagation. No ray bends into it; inside, the wave is held at the angle it has in air; a face reached from inside passes what reaches it (with the entered medium's `att`) and reflects nothing, into air or another mass-law body without a change of direction, into a medium without the mass column refracted as a wave from air (and closed where air would be past that medium's critical angle). No bounce series is summed in a mass-law body.

A run of bonded mass-law bodies is one plate (`qd_plate`). It reflects at the face met what that face does not absorb, $R = 1 - a_\text{met}$, and passes the incoherent series of its outer faces and its bodies,

$$
T = \frac{a\,\tau}{a + \tau - a\,\tau},
$$

$a$ the smaller absorption of its two outer faces (what their pads pass; at normal incidence on an open rigid face $1 - (1 - t_f)R_0$) and $\tau = 10^{-L_\text{mass}/10}$ the bodies' term. $T \approx \tau$ where $\tau \ll a$; a light panel below its knee passes about $a$; what it neither passes nor reflects is absorbed.

**Next to a lossy medium** the two ports of an event are not an energy sum: each is the Fresnel value in normal flux, and a crossing out of a lossy medium is normalized to the forward wave. Coherent sums of paths are energy statements; sums of path powers there are not.

## 7. Sheets, stacks and traced solids

**A sheet** (`thk > 0`) is resolved in one event by one recursion over its boundaries (`qd_stack_resolve`): each boundary a two-port in raw Fresnel amplitudes with its `tf`, each coherent layer the one-way factor $\varphi = \exp(-j k_0 n c\,d)$ (times the `alpha` column over $d/\lvert\mathrm{Re}\,c\rvert$), from the far end:

$$
G_j = r_j + \frac{t_j\,\bar t_j\,G_{j+1}\varphi^2}{1 + r_jG_{j+1}\varphi^2},\qquad
T_j = \frac{t_j\,\varphi\,T_{j+1}}{1 + r_jG_{j+1}\varphi^2}
$$

($\bar t_j$ the crossing back). The conversion to normal flux is applied once, at the outer pair; nothing is clamped. A mass-law layer ends the coherent segment as a half-space of its own admittance, adds its real mass factor and displaces nothing. A sheet stands for a layer that occupies its thickness of the medium it sits in, which the tracer walked: that length is taken back out (half on either side of the mesh plane with one medium on both sides, in front of the solid for a glued entry, behind it for a glued exit). `ray_mesh_interact` evaluates the composite with air on every unnamed side; `ray_state_update` redoes it with the tracked media (a sheet inside a medium has that medium on both sides). A sheet glued to a medium the wave cannot enter (past the outer pair's critical angle) stays in the result: its far face is the pad.

**A traced solid** is followed through its interior. Its internal bounce series is summed where it is strong enough: the face coefficient is multiplied per channel by

$$
S = \frac{1}{1 - r_\text{near}\,r_\text{far}\,\varphi^2}\qquad\text{where}\qquad
\rho = \lvert r_\text{near}r_\text{far}\rvert\,\lvert\varphi\rvert^2 \geq \texttt{eps}
$$

(the Airy sum of the plane-parallel plate [4]; `qd_slab_resolve`; `eps` is `ray_progress`'s `thin_slab_threshold`, default 0.15); otherwise the bounces are traced as further events. A body whose two faces are not parallel (within about 5°) is left to the tracer. The verdict is taken once per body, at the first event that closes a leg in it: a summed series kills the later reflection inside it (it is inside the sum), a declined one is followed. A layered wall is a sum of paths: each body is resolved on its own, and what its neighbors return arrives as further paths; the first path is not the wall (Section 13).

**Sheet or solid.** A sheet suits thin layers (coatings, panes, leaves, linings): no mesh thickness, every internal series summed, the same wall as its traced solid within the limits of Section 13. A traced solid suits thick bodies whose interior matters (receivers or sources inside, the lateral offset of refraction).

## 8. What the state machine adds

`ray_state_update` carries three 16-bit state words per ray (the current medium, the previous medium or the sheet behind the ray, and a buffer) and two in-medium ledgers (`acc_dist_geometric`, `acc_dist_refracted`). A caller passes the words and ledgers of one call to the next unchanged; without state words every ray starts in air and each event is corrected on its own. Per event it:

- re-evaluates the face at the incidence of the direction the wave has (`path_dir_prev`), at the virtual hit point (VBS) of that direction in the face plane, with its total-reflection verdict and its continuation direction;
- closes the leg the ray leaves: the loss and the phase of the medium over the ledgers, once per leg;
- sums the bounce series of the leg's body where $\rho \geq$ `eps` (7), or leaves it to the tracer;
- re-resolves a sheet against the media the state reports;
- kills a ray that has no physical continuation (a reflection already inside a summed series, an inconsistent state);
- writes the resolved-type code (`resolved_typeN`; the code table is in its API documentation).

**Band runs.** `ray_progress` traces up to 127 frequencies on the geometry of the first. The other frequencies take the first one's verdicts (`verdict_inN`), its geometry and its bent state; a caller of `ray_state_update` keeps one `acc_dist_geometric` per run, one `acc_dist_refracted` per frequency, and hands `origN`, `destN` in on the call that defines the geometry only.

## 9. Electromagnetic use

`obj_file_read` without a material file assigns the built-in table by the base name of each `.obj` material (the part before the first dot; unknown names map to air unless `csv_strict` is set). Its `air`, `vacuum` and `itu_` rows other than `itu_vegetation` are Rec. ITU-R P.2040-3, Table 3 [1], whose frequency ranges are indicative and reach 100 GHz for most materials (brick and plywood 40 GHz, marble 60 GHz, floorboard 50–100 GHz); the ground types are limited to 1–10 GHz. [Data Formats](formats.html) lists the range and the source of every row. The table defines `a`, `b`, `c`, `d`, `att` only (`fRef` 1 GHz, $\mu = 1$):

| name | a | b | c | d | att | | name | a | b | c | d | att |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| air, vacuum | 1.0 | 0.0 | 0.0 | 0.0 | 0 | | itu_ceiling_board | 1.48 | 0.0 | 0.0011 | 1.075 | 0 |
| textiles | 1.5 | 0.0 | 5e-5 | 0.62 | 0 | | itu_chipboard | 2.58 | 0.0 | 0.0217 | 0.78 | 0 |
| plastic | 2.44 | 0.0 | 2.33e-5 | 1.0 | 0 | | itu_plywood | 2.71 | 0.0 | 0.33 | 0.0 | 0 |
| ceramic | 6.5 | 0.0 | 0.0023 | 1.32 | 0 | | itu_marble | 7.074 | 0.0 | 0.0055 | 0.9262 | 0 |
| sea_water | 80.0 | −0.25 | 4.0 | 0.58 | 0 | | itu_floorboard | 3.66 | 0.0 | 0.0044 | 1.3515 | 0 |
| sea_ice | 3.2 | −0.022 | 1.1 | 1.5 | 0 | | itu_metal | 1.0 | 0.0 | 1.0e7 | 0.0 | 0 |
| water | 80.0 | −0.18 | 0.6 | 1.52 | 0 | | itu_very_dry_ground | 3.0 | 0.0 | 0.00015 | 2.52 | 0 |
| water_ice | 3.17 | −0.005 | 5.6e-5 | 1.7 | 0 | | itu_medium_dry_ground | 15.0 | −0.1 | 0.035 | 1.63 | 0 |
| itu_concrete | 5.24 | 0.0 | 0.0462 | 0.7822 | 0 | | itu_wet_ground | 30.0 | −0.4 | 0.15 | 1.3 | 0 |
| itu_brick | 3.91 | 0.0 | 0.0238 | 0.16 | 0 | | itu_vegetation | 1.0 | 0.0 | 1.0e-4 | 1.1 | 0 |
| itu_plasterboard | 2.73 | 0.0 | 0.0085 | 0.9395 | 0 | | irr_glass | 6.27 | 0.0 | 0.0043 | 1.1925 | 23 |
| itu_wood | 1.99 | 0.0 | 0.0047 | 1.0718 | 0 | | | | | | | |
| itu_glass | 6.31 | 0.0 | 0.0036 | 1.3394 | 0 | | | | | | | |

`itu_vegetation` follows Rec. ITU-R P.833-9 [2], `irr_glass` (infrared-reflective glass) 3GPP TR 38.901 [3]. Use type 1 for building penetration where the lateral offset of refraction does not matter, type 2 where the bent path does; walls as traced solids, panes as sheets or solids (they agree).

**Worked examples** (measured through `ray_progress` in single precision, one ray in air; T and R the power of the summed paths). A 200 mm `itu_concrete` wall (traced solid) and 6 mm `itu_glass` and `irr_glass` panes (sheet and traced solid alike):

| material | f (GHz) | T at 0° (dB) | T at 45°, TE / TM (dB) | R at 0° | R at 45°, TE / TM | 3GPP loss (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `itu_concrete` 200 mm | 3.5 | −19.03 | −21.08 / −19.07 | 0.157 | 0.260 / 0.068 | 19.0 |
| `itu_concrete` 200 mm | 28 | −90.90 | −96.65 / −94.64 | 0.154 | 0.261 / 0.068 | 117.0 |
| `itu_glass` 6 mm | 3.5 | −2.84 | −4.62 / −1.31 | 0.464 | 0.639 / 0.242 | 2.7 |
| `itu_glass` 6 mm | 28 | −2.54 | −5.05 / −2.25 | 0.199 | 0.510 / 0.172 | 7.6 |
| `irr_glass` 6 mm | 3.5 | −25.82 | −27.58 / −24.30 | 0.198 | 0.315 / 0.095 | 24.1 |
| `irr_glass` 6 mm | 28 | −25.27 | −27.86 / −24.96 | 0.181 | 0.309 / 0.092 | 31.4 |

Every value lies within 0.001 dB (T) and 1e-5 (R) of the closed-form single slab of P.2040 [1] (the Airy pair with complex angles; for `irr_glass` its `att` on the transmit port and once on the internal path of the reflect port), with types 1 and 2. The 3GPP column is the penetration-loss model of TR 38.901, Table 7.4.3-1 [3] (concrete $5 + 4f$, standard glass $2 + 0.2f$, IRR glass $23 + 0.3f$ dB, no thickness): the library's loss is 0.03 / 0.14 / 1.72 dB above it at 3.5 GHz and 26.10 / 5.06 / 6.13 dB below it at 28 GHz. These differences compare two models and are not a check.

## 10. Acoustic use

### 10.1 The mapping

A radio tracer traces sound when frequencies are scaled to keep wavelengths, $f_\text{radio} = f_\text{acoustic}\,c_0/c_\text{sound}$: with $c_\text{sound}$ = 343 m/s, 1 kHz is 0.874 GHz and 100 Hz is 0.0874 GHz (`resF`, `coiF`, `tfF` and `fRef` scale the same way). Every acoustic material is mapped with two parameters: a medium of impedance $Z$ and sound speed $c$ has

$$
\varepsilon = n\,z,\qquad \mu = n/z,\qquad z = Z_\text{air}/Z,\qquad n = c_\text{air}/c,
$$

so $\varepsilon$ is the compressibility ratio $\rho_0c_0^2/(\rho c^2)$ and $\mu$ the density ratio $\rho/\rho_0$; the mapping lives in the columns only, no code reads it. The examples use $\rho_0$ = 1.2 kg/m³ and 343 m/s. The older one-parameter form ($\varepsilon = (Z_\text{air}/Z)^2$, $\mu = 1$) gets the phase of a rigid half-space wrong (twice the angle) and makes a porous layer on such a wall absorb too much at oblique incidence (Section 12). Use the scalar types 3–5.

### 10.2 Wave media and mass-law materials

- **Wave media** (`m = 0`) carry a wave: porous layers, fluids, the air of a cavity, thin leaves. Their columns `a`–`h` are the mapping of the material's complex density and compressibility; a traced or sheet layer of them is the exact fluid layer, and a thin wave-medium sheet is a limp leaf of mass reactance $j\omega m''$.
- **Mass-law materials** (`m > 0`) are rigid bodies: walls, panes, boards. Their faces reflect $1 - t_f$ (or the peaked face of 6.4) with the hard wall's phase of the mapping; their transmission is the plate rule of their faces and mass term (6.5). `fRef` with the thickness sets the mass level, `side` the panel size, `res*` and `coi*` the plate impedances (5.4), `att` an extra loss per entry (frames, seals).

### 10.3 The columns per acoustic quantity

| quantity | columns |
| --- | --- |
| room-side absorption of a rigid face | `tf`, `tfB` (a power law), `tfP`, `tfF`, `tfQ` (a resonant peak, a locally reacting face) |
| transmission level of a rigid body | `fRef` with `thk` (the surface mass), `m > 0` |
| panel size | `side` (√area; 3.29 m where absent; 0 the infinite plate) |
| coincidence dip | `coiF`, `coiQ` |
| mass-air-mass dip of a double unit; a panel's first mode | `res*` with `resS > 0`; `resS < 0` |
| extra loss per entry | `att`, `attB` |
| porous and empirical absorbers | `a`–`h`, `alpha` of a wave medium |
| panel absorber, double leaf | wave-medium leaves on a box of air (10.4) |

`tf` of a face without a peak is flat in angle: it is a random-incidence absorption coefficient, $\alpha_0$ and the field value at once.

### 10.4 Leaves and cavities

A light leaf over an unpartitioned cavity (a panel or membrane absorber, a double leaf, a microperforated panel as an equivalent fluid) is built from wave media: the leaf a wave-medium sheet (`m = 0`, `thk`, $\mu = \rho/\rho_0$, its mounting loss a constant ratio $r_r$ on $\mu$: `g` = `e`·$r_r$/17.98, `h` = 1, `fRef` = 1); the cavity a meshed box of near-air material (`a` 1, `e` 1.000001: a material identical to air is air) with the leaf glued on its front face; the wall a mass-law body glued on its back face. The box's series is summed in one event, so a band equals its single frequencies, and the leaf on the box is the transfer matrix of leaf and cavity at every angle, its resonance moving with the angle as over a real cavity. A double leaf is two leaves on one box; a porous fill is a traced body at the back of the box, and behind a fill the wall takes `tf` 0. Rules 9–10 of Section 12 apply.

### 10.5 Deriving columns from data

- **Absorption references** are random-incidence values (the Paris integral $\int_0^{\pi/2}\alpha(\theta)\sin 2\theta\,d\theta$ of the face). A hard face (α up to about 0.2) is used as measured; an absorbing product of known build is converted from chamber (Sabine) values to random incidence; mineral wool without build data takes the edge correction $\alpha_\infty = \alpha_s - \beta E$; values above one and placement values are no `tf`. A peaked face is fitted on its Paris value, not its $\alpha_0$.
- **Porous columns** come from Miki's model [8], or Komatsu's [9] for mineral wool, at the measured flow resistivity, over the whole band, as power laws of $\varepsilon = n z$, $\mu = n/z$.
- **A resonance** takes the measured $f_0$; else $f_0 \approx C/\sqrt{m''d}$ ($m''$ in kg/m², $d$ the cavity depth in m) with $C$ = 60 for an empty cavity and 50 for a filled one.
- **Transmission references** are fitted on the library's own 0°–90° average (weight $\cos\theta\sin\theta$), at the side the reference was measured at, octaves 31.5 Hz–8 kHz; bands without a measurement take the spread of physical models. The tolerance is a guideline: about ±3 dB in transmission, ±0.05 in absorption.

## 11. Acoustic worked examples

Measured through `ray_progress` (scalar, tracer 4, one ray per angle, 192 angles over 0°–90°, 6 frequencies per third octave; T and R the power of the summed paths; random incidence with weight $\cos\theta\sin\theta$, absorption as the Paris integral). Each construction as a sheet and as a traced solid: they agree within 1e-5 dB. Frequencies are acoustic; the library reads them times 8.7403e-4 GHz/Hz.

### 11.1 A single-leaf wall: cast concrete 152 mm

Columns of row "concrete 152" of 11.4 (fitted at the laboratory side 3.523 m), face `a` 3.850e-6, `e` 1901 (two-parameter concrete, ρ 2281 kg/m³).

| octave (Hz) | 125 | 250 | 500 | 1k | 2k | 4k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| library, random incidence (dB) | 41.4 | 47.0 | 52.1 | 56.9 | 61.5 | 65.0 |
| library, normal incidence (dB) | 44.0 | 51.4 | 57.5 | 63.3 | 68.9 | 74.6 |
| measured, Litvin & Belliston [11] (the fit's reference) | 41.9 | 46.5 | 52.6 | 56.3 | 62.0 | 64.9 |
| measured, Mahn et al. [12], 150 mm, 374 kg/m² | 41 | 42 | 51 | 59 | 65 | 74 |
| bare mass law of 347 kg/m² (`fRef` 5.0161e-5, `side` 3.29), random incidence (dB) | 47.1 | 51.7 | 56.7 | 61.9 | 67.3 | 72.7 |
| room-side absorption | 0.007 | 0.010 | 0.014 | 0.020 | 0.029 | 0.042 |

The fitted row lies within 0.58 dB of its reference on this grid (0.56 dB in 11.4); against the second laboratory it reads +5.0 dB at 250 Hz and −9.0 dB at 4 kHz (another wall, another laboratory). The bare mass law, the plate with no fitted knob, reads 4.1 to 7.8 dB above the measurement: heavy walls above coincidence fall below the mass law, which the fit carries in `att`, `coi*` and a level below the mass. The absorption is within 0.022 of a compilation value for concrete (0.01–0.02 [15]). At the default `side` 3.29 m the wall reads 0.04–0.14 dB higher from 63 Hz.

### 11.2 A window: float glass 4 mm

Row "glass 4 mm" of 11.4 (laboratory side 1.349 m), face `a` 1.7665e-6, `e` 2083.3 (two-parameter glass).

| octave (Hz) | 125 | 250 | 500 | 1k | 2k | 4k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| library, random incidence, `side` 1.349 m (dB) | 16.5 | 20.5 | 26.3 | 31.5 | 33.3 | 26.5 |
| library, normal incidence (dB) | 15.8 | 26.4 | 32.4 | 38.4 | 44.5 | 50.7 |
| EN 12758 generic values [13] (the fit's reference) | 17 | 20 | 26 | 32 | 33 | 26 |
| measured, Quirt [14], 4 mm float | 23.6 | 25.6 | 29.7 | 33.6 | 32.6 | 28.0 |
| library, random incidence, `side` 3.29 m (dB) | 14.8 | 18.7 | 24.7 | 30.2 | 32.1 | 26.3 |
| room-side absorption (`tf`, flat in angle) | 0.305 | 0.236 | 0.183 | 0.142 | 0.110 | 0.085 |

Within 0.51 dB of the generic values; the separate laboratory measurement of a small sash window reads 3.4 to 7.1 dB higher below 1 kHz, which is the spread between the references, not a property of the model. The coincidence dip (`coiF` 2967 Hz) shows at 4 kHz. The same pane as a 3.29 m panel transmits 1.7 dB more at 125 Hz and 4.8 dB more at 31.5 Hz. The absorption is within 0.045 of the compilation row it was fitted to (window glass 0.35–0.04 [16]).

### 11.3 A porous absorber on a rigid wall: glass wool 50 mm

Glass wool of measured flow resistivity 13112 Pa·s/m² [10]; Komatsu's model [9] at that value, mapped and fitted as power laws over 22 Hz–11.3 kHz (`fRef` 0.87403 GHz): `a` 1.1362, `b` −0.043932, `c` 0.0023911, `d` 1.2533, `e` 2.2495, `f` −0.45515, `g` 0.088044, `h` 0.14944. The layer is glued on a 200 mm two-parameter concrete wall with `m` 1 and `tf` 0.

| octave (Hz) | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| library, random incidence | 0.012 | 0.037 | 0.106 | 0.280 | 0.589 | 0.861 | 0.935 | 0.949 | 0.938 |
| Komatsu's model itself, same layer | 0.016 | 0.050 | 0.141 | 0.334 | 0.610 | 0.837 | 0.930 | 0.949 | 0.962 |
| measured, sound-field synthesis, 1.5 × 1.5 m sample [10] | | | | 0.407 | 0.720 | 0.887 | 0.937 | 0.940 | |
| JCA model of the infinite layer [10] | | | 0.130 | 0.347 | 0.657 | 0.887 | 0.945 | 0.958 | |

The library equals the fluid layer of its columns to 3e-7. Against the measured curve it reads 0.13 low at 250 and 500 Hz and within 0.03 from 1 kHz; against the infinite-layer model of the same paper within 0.07. Of that, up to 0.054 is the power-law fit (the power laws follow the model within 16 % in impedance and wavenumber over the whole band), the rest is the model and the finite sample. The chamber values of the same product (0.153 to 1.073) are Sabine values and no reference for `tf` or for the columns.

### 11.4 Starting rows for rigid constructions

Thirteen constructions fitted with the rule of 10.5 to a laboratory reference, preliminary starting rows until a published material set replaces them. `m` 1 in every row; `a`, `e` are the face's two-parameter mapping (for example concrete 3.850e-6 / 1901, glass 1.7665e-6 / 2083.3, gypsum 4.6908e-5 / 573.33): they set only the face's critical angle, since with `tf` it reflects $1 - t_f$. Columns rounded to 5 significant digits; frequencies in GHz; "worst" is the largest octave deviation from the reference (31.5 Hz–8 kHz, outside the model spread in bands without a measurement), evaluated with the formula copy of the library at random incidence.

| construction | `thk` | `fRef` | `tf` | `tfB` | `att` | `attB` | `coiF` | `coiQ` | `resF` | `resQ` | `resS` | `side` | reference | worst (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| glass 4 mm | 0.004 | 4.0562e-05 | 5.6609 | −0.37 | 6.7899e-08 | 1.3503 | 2.5931 | 284.23 | 0.12066 | 2.3289 | 1 | 1.3492 | generic values [13] | 0.85 |
| glass 6 mm | 0.006 | 5.1068e-05 | 3.6817 | −0.48 | 2.6843e-05 | −0.43288 | 1.7261 | 299.62 | 0.022528 | 49.695 | 1 | 1.3492 | generic values [13] | 1.78 |
| gypsum board 12.5 mm | 0.0125 | 0.0002614 | 7.3488 | −0.56 | 456.88 | −0.7806 | 2.4146 | 299.95 | 0.026186 | 38.343 | −1 | 1.6971 | thin-plate model [6] (no measured spectrum) | 2.54 |
| plywood 18 mm | 0.018 | 0.00021843 | 0.21267 | −0.11 | 0.034301 | 0.48574 | 1.3661 | 300 | 0.64136 | 49.996 | 1 | 1.6971 | thin-plate model [6] (no measured spectrum) | 0.31 |
| cast concrete 152 mm | 0.152 | 0.00019663 | 0.00025486 | 0.52 | 14.459 | −0.086239 | 6.0529 | 1.19 | 0.068191 | 1.4354 | −1 | 3.5229 | laboratory [11] | 0.56 |
| face brick 95 mm | 0.0953 | 0.00016602 | 0.0042827 | 0.27 | 0.0003328 | 1.038 | 4.9133 | 11.907 | 0.17928 | 2.7734 | 1 | 3.5229 | laboratory [17] | 0.01 |
| solid brick 203 mm | 0.2032 | 0.00014723 | 0.004146 | 0.27 | 42.962 | −0.33845 | 16.528 | 1 | 0.10412 | 6.7838 | 1 | 3.5229 | laboratory [17] | 0.58 |
| concrete block 190 mm | 0.19 | 0.00042658 | 0.0046884 | 0.4 | 1.3513 | 0.24768 | 7.1244 | 3.0262 | 0.50699 | 2.8689 | 1 | 3.5229 | laboratory [18] | 0.49 |
| double glazing 4-16-4 | 0.008 | 0.00011189 | 1.3406 | −0.36 | 0.044026 | 0.53547 | 2.686 | 300 | 0.18446 | 49.56 | −1 | 1.3492 | laboratory [19] | 1.56 |
| triple glazing 4-12-4-12-4 | 0.012 | 0.00012741 | 1.2793 | −0.36 | 0.064143 | 0.5339 | 2.5935 | 299.18 | 0.16776 | 49.433 | −1 | 1.3492 | laboratory [19] | 2.67 |
| brick 203 mm, gypsum board on furring, one face | 0.2032 | 0.0030373 | 0.6398 | −0.22 | 92.987 | −0.36162 | 0.026221 | 1.0036 | 0.14427 | 48.891 | 1 | 3.5229 | laboratory [17] | 2.70 |
| brick 140 mm, gypsum board on dabs, one face | 0.14 | 0.00018593 | 1.1829 | −0.22 | 0.0039854 | 0.94732 | 2.3365 | 290.01 | 0.17773 | 3.2909 | −1 | 3.5229 | laboratory [17] | 1.95 |
| brick 105 mm, plaster one face | 0.105 | 0.00034267 | 0.0064453 | 0.22 | 4.7922 | 0.092557 | 3.264 | 1.6129 | 0.098903 | 1.0297 | −1 | 3.5229 | laboratory [17] | 0.00 |

`tf` and `att` are their values at `fRef` (a fraction of a hertz for these rows); at 1 kHz they are the absorption and the extra loss of the class. On heavy walls and linings the knobs stand for what the plate does above coincidence, not for physical constants (11.1). Each row is one panel: use it as one body (or one sheet) with its `side`.

## 12. Rules of use

1. **Build scenes in local coordinates near the origin.** The relaunch offset grows with the coordinate; mm-wave work through `ray_progress` belongs within about 100 m of the origin, never in raw georeferenced coordinates (13).
2. **One `att` per bonded run or glued stack.** The loss is paid at every entry, so a second `att` counts twice.
3. **One `side` per bonded run.** Each body pays its increment at its own side; a run with mixed sides depends on the direction.
4. **Set `tf` together with `m`.** A rigid face without `tf` absorbs nothing, so its plate passes nothing, from either side.
5. **A rigid body carries the mass column.** A rigid permittivity without `m` is a wave medium: as a sheet it tunnels (22 mm of $\varepsilon$ 2.1609e-8 passes −0.03 to −0.28 dB at 10° to 60°).
6. **Keep a rigid body lossless.** A loss column on it opens its faces: a 200 mm wall with `tf` 0.3 and a loss tangent of 0.04 then reflects 1.000000 from 10° on and passes −88 to −102 dB.
7. **Map acoustic materials with two parameters.** The one-parameter form reflects at a rigid half-space with a phase of 20° / 90° / 170° at 10° / 45° / 85° (two-parameter: 0.01° / 0.06° / 0.69°), and a 50 mm glass-wool layer on such a wall absorbs up to 0.31 more per angle at 125 Hz.
8. **Use the scalar types for acoustic materials.** The EM TM channel of a mass-law material is formal: up to 3.9 dB from its traced twin and 7.5 dB between its two sides.
9. **Leaves on a box of air, never on a meshed gap.** A meshed gap is followed bounce by bounce and runs out of interactions (a 12.5 mm board on 25 mm of air: up to 0.29 in R off the box at 255 interactions); never two meshed mass-law sheets for a double unit (a mass-law sheet cannot resonate with a cavity): a fitted row or two leaves.
10. **A porous layer between a box and a wall is a traced body,** not a sheet: the intersector reports two of the three co-located faces.
11. **A source inside a medium:** seed the bent bit of the current-medium word and use type 5; a leg whose bend the state does not know keeps its segment's cosine in the mass term (0.21 / 0.96 / 3.00 dB at 10° / 20° / 30°).
12. **Glue co-located faces exactly.** The 1 mm is measured along the ray; near grazing, faces much closer than 1 mm in normal distance are two events with air between them.
13. **Avoid overlapping geometry;** at most two co-located sheets form a stack, a third layer is a separate event.
14. **Direct callers of `ray_state_update`:** pass the state words and ledgers through unchanged, hand the base frequency's verdicts to the other frequencies, and with types 2 or 5 hand `origN`, `destN` in on the geometry-defining call (8).
15. **Set `min_gain_dB` with a margin near walls:** inside a plate the running gain lies up to $-10\log_{10}a$ below the level the path will reach (10 dB for $t_f$ = 0.1).
16. **Sum the box of a leaf:** `thin_slab_threshold` up to 0.15 (the default) sums it.

## 13. Limits

- **Far from the origin.** A body thinner than the relaunch offset is skipped. In single precision a 4 mm pane at 28 GHz 1 km from the origin is within 1.3° of phase and 0.5 dB (85°) of the same pane at the origin; 100 m out, rays within 2° of grazing were lost. In double precision a traced leg is exact to 1.2e-5 dB at 1 km.
- **A transformed face on a lossy wave medium** reflects $(1 - t_f)R_0$ only where its two sides are in phase: +1.0 dB at `tf` 0.5 and +5.3 dB at 0.9 on $\varepsilon$ 1.50 − 0.40j (+0.03 / +0.22 dB on 5.24 − 0.63j), normal incidence.
- **A negative `resS` is a gain medium:** the two ports of a 37.5 mm sheet of $\varepsilon$ 4 with `resF` 1 GHz, `resQ` 10, `resS` −0.5 add to 14.2 near the pole. Nothing validates `resS`.
- **The state is one level deep.** Air is assumed beyond every glued exit; a plate of three or more bonded bodies pays its rule within +3.02 / −3.01 dB; a wall with an ordered stack inside it is up to 3.01 dB from its traced twin.
- **A sheet between two bodies** (three co-located faces) is resolved by which two faces the intersector reports; nothing is created, but the reflection can be that of box | sheet | air (R 0.154 where the fluid layers give 0.994).
- **The first path through a layered wall is not the wall** (−4.87 dB for a lossless second layer at 85°); a run that stops early keeps part of the difference (+1.37 dB after 20 interactions, +0.04 dB after 60).
- **The two sides of a lossy lining on a rigid wall differ** by the lossy medium's flux convention: 0.07–1.38 dB (dense lining) and 0.52–2.70 dB (porous lining) at 10° to 75°, the wall's side higher.
- **A sheet inside a host and its traced twin** differ by the host's `att` (the traced solid pays it again on re-entry).
- **Band runs** use the first frequency's geometry, bend and verdicts; near a critical angle of a lossless dispersive medium another frequency could take the other verdict.
- **A ray tube** follows its spine: with types 2 or 5 a tube with a vertex past a critical angle passes undeviated.
- **Leaves and wave media keep the infinite plate:** `side` acts only where `m > 0`; a leaf has no bending stiffness (coincidence needs the mass column); a leaf past its critical angle tunnels, and the thin-layer result holds while the leaf is thin against the wavelength in it; a curtain's relaxation in $\mu$ is reached on the random-incidence average only, not per angle.
- **Out of scope:** scattering, diffusion and edge diffraction of surfaces; flanking transmission through the building; the ground wave of a spherical source; level-dependent (nonlinear) absorbers; placement values of an absorber in a modal field. Double walls by stud type and panel sizes enter as fitted rows.

## 14. References

[1] Recommendation ITU-R P.2040-3, *Effects of building materials and structures on radiowave propagation above about 100 MHz*, International Telecommunication Union, Geneva, 2023. (Eq. 9b; Fresnel coefficients, eqs. 31–32; Table 3.)

[2] Recommendation ITU-R P.833-9, *Attenuation in vegetation*, International Telecommunication Union, Geneva, 2016.

[3] 3GPP TR 38.901 V17.0.0, *Study on channel model for frequencies from 0.5 to 100 GHz*, 3rd Generation Partnership Project, 2022. (Table 7.4.3-1.)

[4] M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press, 1999. (The plane-parallel plate.)

[5] J. L. Davy, "The forced radiation efficiency of finite size flat panels that are excited by incident sound," *J. Acoust. Soc. Am.* 126(2), 694–702, 2009.

[6] J. L. Davy, "Predicting the sound insulation of single leaf walls: extension of Cremer's model," *J. Acoust. Soc. Am.* 126(4), 1871–1877, 2009.

[7] A. London, "Transmission of reverberant sound through single walls," *J. Res. Natl. Bur. Stand.* 42, 605–615, 1949.

[8] Y. Miki, "Acoustical properties of porous materials — Modifications of Delany-Bazley models," *J. Acoust. Soc. Jpn. (E)* 11(1), 19–24, 1990.

[9] T. Komatsu, "Improvement of the Delany-Bazley and Miki models for fibrous sound-absorbing materials," *Acoust. Sci. Tech.* 29(2), 121–129, 2008.

[10] S. Dupont, M. Sanalatii, M. Melon, O. Robin, A. Berry, J.-C. Le Roux, "Measurement of the diffuse field sound absorption using a sound field synthesis method," *Acta Acustica* 7, 26, 2023. (Values read from Fig. 12.)

[11] A. Litvin and H. W. Belliston, "Sound transmission loss through concrete and concrete masonry walls," *J. American Concrete Institute* 75(12), 1978 (Portland Cement Association RD066).

[12] J. Mahn, M. Müller-Trapet, S. Skoda, I. B. Cunha, *Guide to calculating airborne sound transmission in buildings*, 6th ed., National Research Council Canada, Research Report RR-331, 2023.

[13] EN 12758 generic values as published in Pilkington, *Pilkington Optiphon — Laminated glass for noise control*, brochure 8222, January 2020.

[14] J. D. Quirt, *Measurements of the sound transmission loss of windows*, National Research Council Canada, Division of Building Research, Building Research Note 172, 1981.

[15] T. J. Cox and P. D'Antonio, *Acoustic Absorbers and Diffusers: Theory, Design and Application*, 3rd ed., CRC Press, 2016. (Appendix A.)

[16] F. A. Everest and K. C. Pohlmann, *Master Handbook of Acoustics*, 7th ed., McGraw-Hill, New York, 2022, ISBN 978-1-260-47359-9. (Appendix C.)

[17] Brick Industry Association, *Technical Notes on Brick Construction 5A: Sound Insulation — Clay Masonry Walls*, reissued August 2000 (Riverbank Acoustical Laboratories tests).

[18] A. C. C. Warnock, "Controlling sound transmission with concrete block constructions," *Proc. 7th Canadian Masonry Symposium*, Hamilton, 1995.

[19] AGC Glass Europe, *Acoustic glass performances: Planibel, Stratobel, Stratophone, Thermobel*, pocket guide 08/2023, 2023.

---

# Part II — Developer notes

## 15. A map of the source file

`src/ray_mesh_interact.cpp` opens with an overview of the model (lines 5–88) and holds seven sections in one anonymous namespace, then the five public functions, each after its API block:

| section | contents |
| --- | --- |
| 1 constants and vector helpers | `colocation_dist`, `mass_min_len`, the state-word masks, `k0`, `decaying_root`, `qd_reflect`, `qd_refract`, `qd_relaunch_offset`, `qd_polbasis` |
| 2 material model | `MaterialCols` (reads and checks the 26 columns), `davy_sigma`, `mass_TL`, `Material` (`eta`, `mu`, `medium_gain`, `mass_X0`, `excess_gain`, `interface_gain`, `tf`, `peaked`, `zeta`, `same_as`) |
| 3 interface physics | `apply_tf_pair`, `closed_face`, `tf_face`, `flux_amp`, `interact_with` |
| 4 slab and stack solver | `qd_mass_amp`, `qd_plate`, `stack_held_plate`, `stack_leak_chain`, `stack_boundary`, `stack_layer_phase`, `qd_stack_resolve`, `sheet_ports`, `slab_face_reflection`, `qd_slab_resolve` |
| 5 sheets | `SheetCase`, `mass_cos_in_air` |
| 6 face-hit geometry | `face_plane`, `ray_plane_hit`, `classify_hit`, `hit_media`, the tube helpers, `rmi_no_hit` |
| 7 the pure parts of the state machine | `decode_words`, the thickness code, `recover_cosine`, `bonded_x0`, `bonded_thickness`, `leg_exact_factor`, `sheet_hit`, `relaunch_ledger_offset`, `vbs_geometry` |

**One event in `ray_mesh_interact`:** `face_plane` and `ray_plane_hit` for the FBS and SBS → `classify_hit` (the hit code, the sheet case) → `hit_media` (near and far, air where unnamed) → `interact_with` (coefficients, Snell ratio, total reflection) → `sheet_ports` for a sheet → the tube's total-reflection test → the direction, the tube, the relaunch segment → `qd_polbasis` (xprmat and gain).

**One event in `ray_state_update`** (one loop over the compact set): `decode_words` → `vbs_geometry` (the incidence at the VBS) → per pass a switch on the hit code: the reflect pass (`face_at_vbs` as a mirror, `sum_series`, `close_leg`), the transmit pass (`transition` for a bare face, `sheet_event` for a sheet, both via `face_at_vbs` → `sum_series` → `close_leg`) → `relaunch_ledger_offset` → the words, ledgers, codes and verdict. `sum_series` wraps `qd_slab_resolve`; `close_leg` charges the medium in one of three forms (exact at the recovered complex cosine `leg_exact_factor`, the hybrid of `medium_gain` with the real-index phase, or the walk-off form of an unbent leg); `sheet_at_tracked_media` redoes the composite with the tracked media.

**The state words** (16 bits each; bits 0–13 a material index):

| word | bits 0–13 | bit 14 | bit 15 |
| --- | --- | --- | --- |
| current | the medium the ray is in (0 air) | bent: the leg follows a traced bend (types 2, 5) | summed: the leg's series was summed |
| prev | the medium the leg was entered from, or the sheet behind the ray | a sheet: it lies inside the current medium; a solid: its face was evaluated at a real-invariant cosine | the leg's faces are not parallel |
| buffer | a deferred medium (overlap, corner), or with bit 14 the stack layer behind the ray | followed: the leg began at a reflection whose series was left to the tracer | set: bits 0–14 are the thickness code of a bonded mass-law body |

The thickness code is $d = 10^{-4 + 6(k-1)/32766}$ m, $k$ = 1…32767 (0.1 mm to 100 m); it lets a second bonded body learn $X_\text{run}$ (`bonded_x0`, `bonded_thickness`). The ledgers hold the refracted and the geometric length of the current leg, including the relaunch offset. The cosine of a wave refracted into a lossy medium is complex while the traced direction is real; `recover_cosine` inverts the tracer's map so that entry and exit are evaluated at one cosine. Everything acoustic is gated on `m > 0` (the mass term, `qd_mass_amp`, the held branches of `interact_with`, the plate rule, the peaked face, `qd_slab_resolve` declining a mass-law slab, the buffer code). The hit-code (`out_typeN`) and resolved-code (`resolved_typeN`) tables are in the API blocks of `ray_mesh_interact` and `ray_state_update`.

## 16. How a change is checked

From the repository root:

```
cmake --build build_linux
QD_DIAG_RAY=1 ./build_linux/test_bin "[wave]" > test_wave.txt
QD_DIAG_RAY=1 ./build_linux/test_bin "[acoustic]" > test_acoustic.txt
QD_DIAG=1 ./build_linux/test_bin > test.txt
python3 -I tools/acoustic/rowdiff.py test_wave_before.txt test_wave.txt
```

- **The tests.** `[wave]` covers every material without the mass column (EM and the scalar types), in `tests/catch2_tests/test_wave_*.cpp`; `[acoustic]` the mass-law add-on and the leaves, in `test_acoustic_*.cpp`. A single file runs by its tag with `-#` (`test_bin -# "[#test_wave_progress]"`). The shared instruments are in `tests/catch2_tests/ray_test_helpers.hpp`: independent references (closed-form slabs, the characteristic matrix of N layers, the transformer and the pad, all with complex angles and no library code) and runners that send one event or one ray through `ray_triangle_intersect`, `ray_mesh_interact`, `ray_state_update` or `ray_progress`.
- **Diagnostics.** `QD_DIAG_RAY=1` prints the tables of the ray tests, `QD_DIAG=1` those of every test area; the assertions are the same with them off. A change is compared against the output before it with `tools/acoustic/rowdiff.py` (every number equal or within one unit of its last printed digit; a moved row is named with its table and case). Every moved row is accounted for.
- **Stored twins.** The wave tables that print coefficients with every digit also assert them against `tests/data/wave_twins.tsv`; after a change meant to move them they are rewritten with `QD_TWIN_WRITE=tests/data/wave_twins.tsv ./build_linux/test_bin "[#test_wave_state_update_branch]" -#`.
- **The order of checks.** EM and wave rows first (they must not move for an acoustic change, which stays behind `m > 0`), then the acoustic rows, then energy: both ports of an event at most one seen from a lossless medium, a sheet and its traced solid one wall, a band equal to its single frequencies.
- **Rules.** Measure, do not compute by hand: a fixture that describes a traced run goes through the library's own functions, and its reference is independent of the library. Before the source is touched, a candidate is measured on a patched copy of the source linked in front of `libquadriga.a`; a plain copy must reproduce the baselines first. Fixtures that call the library one ray at a time run on one thread. A wider gate and one rule are preferred over an exclusion kept for digits; bit identity is not required where a row moves for a physical reason.
- **Traps.** Run the test binary from the repository root (cases open `tests/data` relatively); a test case name with a comma needs `\,` in a test spec; a material identical to air is air; a segment that ends inside a body preloads the ledger, so a fixture that relaunches by hand must let segments reach past every remaining face.
