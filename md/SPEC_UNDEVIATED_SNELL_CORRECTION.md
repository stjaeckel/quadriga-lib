# SPEC: Snell-Correct Physics for Undeviated Paths (VBS Architecture)

*Change specification for `ray_mesh_interact.cpp` — `ray_mesh_interact`, `ray_state_update`,
`Material::interact_with`, `Material::slab_airy_factor`.*

Status: **draft for review**. Companion to `quadriga_lib_material_model.md` ("the model doc",
§N references). No implementation yet.

**Scope.** `ray_mesh_interact` changes surgically: the d2l gate is removed, TIR is computed and
returned honestly, and one new output (`path_dirN`) exposes the physically correct continuation
direction the function already computes internally. All Snell corrections live in
`ray_state_update`, which **replaces** the affected per-hit values with quantities recomputed at
the virtual bounce point (VBS). No compatibility mode; the previous behavior remains reachable
through the version tag preceding this change.

**Contents**

1. [Problems addressed](#1-problems-addressed)
2. [Design overview](#2-design-overview)
3. [Physics definitions](#3-physics-definitions)
4. [API changes](#4-api-changes)
5. [Correction procedure and shared helpers](#5-correction-procedure-and-shared-helpers)
6. [Dispatch-row deltas in `ray_state_update`](#6-dispatch-row-deltas-in-ray_state_update)
7. [Input/output contracts and optionality](#7-inputoutput-contracts-and-optionality)
8. [Validation contract additions](#8-validation-contract-additions)
9. [Caller obligations](#9-caller-obligations)
10. [Energy accounting and calibration impact](#10-energy-accounting-and-calibration-impact)
11. [Removed behaviors and remaining invariants](#11-removed-behaviors-and-remaining-invariants)
12. [Test plan](#12-test-plan)
13. [Open questions for review](#13-open-questions-for-review)

---

## 1. Problems addressed

The undeviated transmission types (1 EM, 4 scalar) keep the ray geometrically straight, so
in-medium faces are hit at the geometric angle and at geometrically shifted positions instead of
the physically refracted angle and position. All problems derive from this.

- **P1 — Spurious TIR at the exit face (EM type 1).** For a slab with $\varepsilon = 5$, every
  ray steeper than $\arcsin(1/\sqrt{5}) \approx 26.6°$ from normal hits the (wrongly evaluated)
  exit critical angle. Physically, TIR at the exit of a slab entered from the same medium is
  impossible: $n \sin\theta_t = \sin\theta_0 < 1$. With the d2l gate removed, the wrong-angle
  exit returns honest $T = 0$ — and is then **replaced** by the state machine with the
  VBS-derived exit coefficient, which can never TIR. Pipeline consequence: the tracer must not
  prune zero-gain rays between the two calls (§9).
- **P2 — The dense-to-light pass-through gate is a hack on a wrong proxy.** It keys on density
  ordering, not on entry/exit. Removed from `interact_with` (already done on the
  specular/scalar path; this change completes it on EM). Acoustic $\varepsilon \ll 1$ front
  faces keep their genuine $|R| \to 1$; energy creation from gate + Airy (verified: assembled
  ledger up to $R + T = 2.24$ at slab resonance, 20°, $\varepsilon = 5$, exact answer 1.000)
  disappears.
- **P3 — The Airy factor uses wrong angle and wrong distance.** With the VBS-derived
  $(\theta_t, d_v)$ the mirrors and the per-bounce loss become correct through the *existing*
  `slab_airy_factor` parameterization. One internal fix remains mandatory: the round-trip phase
  must include the lateral walk-off, $\arg\varphi = -k_0\, n_\mathrm{re}\, d_v \cos^2\theta_t$
  instead of $-k_0\, n_\mathrm{re}\, d_v$ (§3.4; verified against the bounce-ray sum and the
  transfer matrix — without it, fringe positions are wrong off-normal even with corrected
  distance).
- **P4 — The in-medium excess phase is missing.** The tracer applies a fixed propagation speed
  ($c_0$ EM, $c_\mathrm{sound}$ acoustics) to the total path length; a traversed medium adds
  excess phase that is currently applied nowhere. Fixed by a per-leg factor folded into
  `xprmatN` (§3.5), evaluated at each call's frequency so multi-frequency pipelines carry the
  correct in-band delay slope automatically.
- **P5 — Non-parallel faces and multi-slab drift.** The VBS construction with `path_dir_prev`
  handles arbitrary face orientations and accumulated stack drift exactly (for amplitudes);
  without `path_dir_prev`, a parallel-face fallback applies (§7.3).
- **P6 — In-medium bends on diffraction arcs.** $L = d(\mathrm{orig}, \mathrm{fbs})$ is
  meaningful in general; a bend merely masks the true slab-entry origin. Solution: the optional
  `orig_correct` input supersedes `orig` in the corrections; `calc_diffraction_gain` provides
  it from its knowledge of the arc path. With it, all correction inputs are functions of (wall
  geometry, materials, incidence), independent of where polyline vertices landed — continuous
  fields across the arc family.
- **P7 — Rays launched inside a medium.** Limited practical relevance (sources sit in air, not
  in concrete). No dedicated code: a caller that needs it seeds `path_dir_prev` with the launch
  direction and `orig_correct` with the launch point, and the general machinery covers it.
- **P8 — Convention for the reflection pass: not an issue.** The architecture
  self-disambiguates per ray through the data: for type-2 (and types 0/2-paired) rays,
  `path_dirN` from `ray_mesh_interact` equals the geometric direction, so the advanced
  `path_dir_prev` is geometric, VBS coincides with FBS, $\theta_t$ equals the hit angle, and
  every correction degenerates to an identity.

---

## 2. Design overview

1. **`ray_mesh_interact` — surgical change.** Remove the d2l gate; compute and return TIR
   honestly ($T = 0$, except the TF-modified TIR rule below); expose `path_dirN` — the
   normalized direction the path would take from the FBS if refraction were modeled correctly
   (already computed internally for the Fresnel/polarization geometry). For types 0/2/3 and
   for reflections, `path_dirN` equals the `origN`→`destN` direction; it deviates from it
   exactly on type-1/4 transmissions, which is the cause of the problem. Plus the `ray_indN`
   convenience output. Nothing else changes; the public contract stays meaningful standalone.

   **TF-modified TIR rule:** when TIR occurs at a front face and the material's tf parameter
   shifts energy into transmission, that energy goes into an **undeviated forward path**: power
   assigned via tf, `path_dirN` aligned with the `origN`→`destN` direction (this applies even
   under type 2 — the only case where a type-2 transmission is undeviated). The mass-law
   $\cos^2$ term provides the correct angle dependence inside the medium. This is the intended
   mechanism for feeding energy into mass-law media without breaking energy conservation, and
   it is what makes acoustic $\varepsilon \ll 1$ walls work end to end: entry = genuine
   near-total reflection + tf-opened undeviated transmission; interior = mass law; exit at the
   geometric angle. For non-TF TIR, $T = 0$ and `path_dirN` is defined as the
   `origN`→`destN` direction for determinism (the tracer drops the path anyway).

2. **`ray_state_update` — the VBS correction.** The state machine carries the physical
   direction chain: `path_dir_prev` (input, full set: the physical direction with which the
   wave entered the current segment) and `path_dirN` (in/out: `ray_mesh_interact`'s output,
   corrected in place). At every in-medium event it constructs the **virtual bounce scatterer**
   — the point where the physically directed ray ($\mathrm{orig}_c$, `path_dir_prev`)
   intersects the plane of the hit face — and from VBS derives the correct incidence angle
   $\theta_t$, the correct in-medium distance $d_v = |\mathrm{orig}_c - \mathrm{VBS}|$, and the
   correct continuation direction. It then **replaces** `gainN`/`xprmatN` (and corrects
   `path_dirN`) with values recomputed as if the interaction had happened at VBS: in-medium
   gain over $d_v$, transition/reflection coefficients at $\theta_t$, the Fabry-Pérot factor
   fed by $(\theta_t, d_v)$, the polarization basis built from the physical directions.
   Replacement is computed by shared helpers (§5).

3. **`orig_correct`** (optional, full set): supersedes `orig` in the corrections (VBS ray
   origin and in-medium distances). Provided by `calc_diffraction_gain` from its arc
   bookkeeping when bends fall inside media; null for the normal tracer, whose segment `orig`
   is already the (offset) entry point.

4. **The RT loop:** at launch, the tracer sets `path_dir_prev` to the ray direction (outside
   condition). After each interaction it advances `path_dirN` → `path_dir_prev` for the next
   iteration, exactly as it threads the three state words. Both direction arguments are
   optional (§7.3).

---

## 3. Physics definitions

Permittivities: interface values with resonance for Fresnel coefficients, resonance-excluded
for propagation, as in §2.2/§7.5 of the model doc. $k_0 = \omega/c_0$. Grazing-angle
conventions as in the existing code; formulas below use the incidence angle from the normal for
readability.

### 3.1 The VBS construction

At an in-medium event (`cur != 0`), let $\hat{d}_p$ = `path_dir_prev` row (normalized),
$P_0$ = `orig_correct` row if supplied else `orig` row, and $(F, \hat{n})$ = the hit face point
and normal — **FBS with its normal for out-type 4, SBS with its normal for out-type 5**
(colocated i-i hits report the two faces in random order; the state machine must select the
transition face by out-type, not by position in the arrays).

$$
s = \frac{(F - P_0) \cdot \hat{n}}{\hat{d}_p \cdot \hat{n}},
\qquad
\mathrm{VBS} = P_0 + s\,\hat{d}_p,
\qquad
d_v = s,
\qquad
\cos\theta_t = |\hat{d}_p \cdot \hat{n}| .
$$

Guards: if $|\hat{d}_p \cdot \hat{n}| < 10^{-6}$ (true division-by-zero protection only) or
$s \le 0$ or $s$ non-finite, fall back to the geometric values ($\theta$ from `fbs_angleN`,
$d_v = d(P_0, F)$). The guard is deliberately tiny: a larger threshold would create a
corrected-to-wrong physics jump in angle sweeps, whereas the grazing limit needs no protection
— the thickness $t = d_v \cos\theta_t = (F - P_0)\cdot\hat{n}$ stays bounded by construction
(only the slant $d_v$ grows), and $G_\mathrm{med}(d_v)$ and the survival gate drive smoothly to
zero, matching the physical $T \to 0$ grazing behavior.

The corrected continuation direction at VBS follows the §4.3 constructions: Snell refraction
into the next medium for transmissive events, mirror reflection at $\hat{n}$ for reflective
events — both via the shared helpers so the convention is identical to `ray_mesh_interact`'s
own.

### 3.2 Corrected interface coefficients

`interact_with` is called at the real angle $\theta_t$: `interact_with(cur, next, type,
theta_t, ...)`. By Snell-pair reciprocity ($r_{21}(\theta_t) = -r_{12}(\theta_0)$, verified
numerically including near-unimodular regimes) the slab exit evaluated at $\theta_t$ reproduces
the entry reflectance, can never TIR for traversals that entered from outside, and the scalar
branch automatically yields the Stokes-complementary $t_\mathrm{exit} = \sqrt{1 -
R_\mathrm{eff}}$ with the correct phase. tf ownership and `interface_gain` fold-in are
unchanged.

### 3.3 Corrected in-medium propagation

`medium_gain` is called unchanged with the corrected distance: $G_\mathrm{med}(d_v, f,
\cos\theta_t)$. The dielectric, $\alpha$, and mass-law terms all ride the correct path length
and angle.

### 3.4 Fabry-Pérot factor

`slab_airy_factor` keeps its signature and is called with the VBS-derived pair $(\theta_t,
d_v)$. With these inputs the mirrors (slab-side Fresnel at $\theta_t$) and the per-bounce
magnitude ($G_\mathrm{med}$ over $d_v$, survival gate over $2 d_v$) are correct as the function
stands. **One mandatory internal fix:** the one-way phase must include the lateral walk-off of
the bounce series,

$$
\arg\varphi = -k_0\, n_\mathrm{re}\, d_v \cos^2\theta_t
\qquad \big(= -k_0\, n_\mathrm{re}\, t \cos\theta_t,\ t = d_v\cos\theta_t\big),
$$

replacing the current $-k_0\, n_\mathrm{re}\, d_v$. Verified: the bounce-ray sum at a fixed
exit point with transverse phase matching equals the plane-wave $2\beta$ exactly and matches
the transfer matrix to machine precision; the distance-only convention does not, even with
corrected $d_v$. The factor is computable from the two arguments the function already receives
— no signature change. Magnitude, pole clamp ($|denom| \ge 10^{-2}$), survival gate
($\rho = \sqrt{R_n R_f\, G_\mathrm{med}(2 d_v)} \ge \varepsilon_\mathrm{thr}$), re-emit
semantics (NaN): unchanged.

With $(\theta_t, d_v)$ derived from VBS state (and `orig_correct` under bends), every gate
input is a function of wall geometry, materials, and incidence — bend- and
tessellation-independent, identical across the two passes.

### 3.5 In-medium excess phase

Per traversal leg, the excess phase relative to the vacuum phase the tracer counts along the
geometric segment, folded into `xprmatN` at the leg-closing event:

$$
\Delta = k_0\, t\, \big( n_\mathrm{re} \cos\theta_t - |\hat{u}_g \cdot \hat{n}| \big),
\qquad t = d_v \cos\theta_t,
\qquad \hat{u}_g = \mathrm{normalize}(F - \mathrm{orig}),
$$

applied as $e^{-j\Delta}$. For an unbent traversal this is plane-wave exact: the in-medium
wavelength change, the physical-vs-traced path difference, and the lateral phase matching of
the offset exit point sum identically to $k_0 t (n\cos\theta_t - \cos\theta_0)$. It is
phase-only; magnitudes stay with the MED ownership of §7.6 of the model doc.

**Ownership:** applied once per leg, at the event that leaves the medium (cavity exit, M2M
crossing for the layer being left, resolved out-coupling, resolved i-i). The vacuum-reference
term uses the *geometric* segment direction and the true `orig` (not `orig_correct`), because
that is the path over which the tracer counted vacuum phase.

**Frequency handling:** $\Delta \propto k_0$ and must be evaluated at each call's frequency —
never computed at $f_c$ and reused. Pipelines that store the polarization matrix per frequency
then carry the medium's in-band delay slope $e^{-j 2\pi f \tau}$ automatically; CIR synthesis
from the multi-frequency response places the tap at the physically shifted arrival time with no
extra storage. The geometry-derived per-path delay scalar keeps a small bias (~1.3 ns per 30 cm
concrete wall on 20+ dB-attenuated paths; ~0.1 ms on acoustic transmission paths at $-40$ dB),
which is documented and accepted.

---

## 4. API changes

### 4.1 `ray_mesh_interact`

Behavior: d2l gate removed; TIR computed and returned honestly; TF-modified TIR produces the
undeviated forward path of §2.1. Everything else unchanged.

```cpp
template <typename dtype>
void quadriga_lib::ray_mesh_interact(int interaction_type,
                                     dtype center_frequency,
                                     /* ... existing parameters unchanged ... */,
                                     arma::Mat<dtype> *path_dirN = nullptr,  // [n_rayN, 3], physical continuation direction at FBS
                                     arma::u32_vec *ray_indN = nullptr);     // [n_rayN], compact-to-full ray index map
```

- **`path_dirN`**: normalized direction the path would take from the FBS if refraction were
  modeled correctly (the function already computes it internally). Type-1/4 transmissions: the
  Snell-refracted direction, deviating from the `origN`→`destN` direction. Types 0/2/3 and all
  reflections: equal to the `origN`→`destN` direction. TIR rows: equal to the `origN`→`destN`
  direction (TF-modified TIR by the §2.1 rule; non-TF TIR for determinism).
- **`ray_indN`**: compact-to-full surviving-ray index map, 0-based, order-preserving — the
  inverse of the internal `output_ray_index` (currently built and discarded), in exactly the
  format `ray_state_update` consumes. Pure convenience.

### 4.2 `ray_state_update`

```cpp
template <typename dtype>
void quadriga_lib::ray_state_update(
    int interaction_type,
    dtype center_frequency,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *fbs,
    const arma::Mat<dtype> *sbs,
    const arma::u32_vec *no_interact,
    const arma::Col<dtype> *fbs_angleN,
    const arma::Mat<dtype> *normal_vecN,                // [n_rayN, 6]: face normals, required
    const arma::s32_vec *out_typeN,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::Col<short> *mtl_ind_fbs,
    const arma::Col<short> *mtl_ind_sbs,
    const arma::Col<short> *mtl_ind_prev_in = nullptr,
    const arma::Col<short> *mtl_ind_current_in = nullptr,
    const arma::Col<short> *mtl_ind_buffer_in = nullptr,
    const arma::Mat<dtype> *path_dir_prev = nullptr,   // [n_ray, 3]: physical direction entering this segment
    arma::Col<short> *mtl_ind_prev_outN = nullptr,
    arma::Col<short> *mtl_ind_current_outN = nullptr,
    arma::Col<short> *mtl_ind_buffer_outN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,             // [n_rayN, 3]: from ray_mesh_interact, corrected in place
    const arma::u32_vec *ray_indN = nullptr,
    const arma::Mat<dtype> *orig_correct = nullptr,    // [n_ray, 3]: supersedes orig in the corrections
    double eps = 0.15);
```

Names ending in `N` are compact-set ($n_\mathrm{rayN}$); others are full-set ($n_\mathrm{ray}$).
All new arguments are optional data refinements; the corrected physics applies always, with the
fallbacks of §7.3 when inputs are absent.

### 4.3 `Material::interact_with`

The d2l gate and all `dense2light` logic vanish completely, including the `dense2light` output
parameter. No new parameters: all corrected calls pass the real angle $\theta_t$. The
TF-modified TIR transmission rule of §2.1 is implemented here (it already exists on the scalar
path; this unifies EM).

### 4.4 `Material::slab_airy_factor`

Signature unchanged. One internal change: the walk-off phase of §3.4
($\arg\varphi = -k_0 n_\mathrm{re}\, \mathrm{dist} \cdot \cos^2\theta_\mathrm{inc}$, with the
incidence cosine derived from the `theta` argument it already receives).

### 4.5 `Material::medium_gain`

Unchanged.

---

## 5. Correction procedure and shared helpers

At each in-medium event of types 0/1/3/4 (type-2 rows degenerate to identities through the
data, P8), `ray_state_update`:

1. selects the transition face and normal by out-type (FBS/out-4, SBS/out-5; §3.1 swap note),
2. constructs VBS, $\theta_t$, $d_v$ (with `orig_correct` superseding `orig`, and the §3.1
   guards),
3. recomputes the per-hit value — interface coefficients at $\theta_t$, in-medium gain over
   $d_v$, entered-medium offset gain, `interface_gain`, polarization basis from the physical
   incoming (`path_dir_prev`) and outgoing (VBS continuation) directions — and **replaces**
   `gainN`/`xprmatN` with it,
4. multiplies the Fabry-Pérot factor $S(\theta_t, d_v)$ and the leg-closing $e^{-j\Delta}$
   where the dispatch row says so,
5. corrects `path_dirN` to the continuation the ray would take from VBS (refracted or
   reflected per event type); at i-o exits it is then snapped per §6/8.5.

Replacement is the operative word: it is well-defined regardless of what `ray_mesh_interact`
produced — including honest zeros from wrong-angle TIR.

**DRY requirement:** everything needed by both functions moves into helpers in the anonymous
namespace of `ray_mesh_interact.cpp`, and `ray_mesh_interact` itself calls them. No duplicated
logic beyond trivial glue:

- `qd_refract(û, n̂, ratio)` — Snell direction construction (§4.3 convention, including the
  $\mathrm{Re}\cos\theta_t$ treatment) — exists, factored out.
- `qd_reflect(û, n̂)` — mirror direction.
- `qd_polbasis(...)` — the U·diag(c_TE, c_TM)·Q polarization sandwich between incoming and
  outgoing directions (currently inline in `ray_mesh_interact`).
- `qd_hit_charges(...)` — the §4.4 per-hit charge composition (in-medium legs incl. the
  `ray_offset` terms, entered-medium offset gain, `interface_gain`).
- `qd_vbs(...)` — the §3.1 plane intersection with guards.

After an EM replacement, `gainN` is recomputed as $\tfrac{1}{2}\sum |x_{ij}|^2$ from the new
`xprmatN` so the gain–xprmat identity holds exactly; scalar rows write the single coefficient
and $|x|^2$.

**Design note (polarization frame):** the replaced `xprmat` uses the physical directions for
the TE/TM basis ("as if the hit was at VBS"), while the channel builder's global frames follow
the traced geometry. The frame offset is of the order of the geometric/physical angle
difference and vanishes where it matters most (single slab: exit direction realigns).
Documented, accepted.

**Implementation staging (decided):**

- **Stage 1 — helpers, strictly non-breaking.** Extract `qd_refract`, `qd_reflect`,
  `qd_polbasis`, `qd_hit_charges` into the anonymous namespace and refactor
  `ray_mesh_interact` onto them; add the two new optional outputs (`path_dirN`, `ray_indN`).
  Zero behavior change, gated by the existing Catch2 suite (bit-exact) and the benchmark suite
  (no performance regression — the helpers must inline cleanly in the hot loop).
- **Stage 2 — physics.** Remove the d2l gate, unify the TF-modified TIR rule on EM, apply the
  combined M2M tf rule (§10), and implement `ray_state_update` with the VBS corrections,
  reusing the stage-1 helpers plus the new `qd_vbs`; land the new test groups. EM polarization
  replacement is exact from the start — `qd_polbasis` is already proven by the stage-1
  regression before its new consumer exists.

The d2l removal belongs to stage 2, not stage 1: removing it without the state-machine
replacement would kill oblique type-1 transmissions (P1).

---

## 6. Dispatch-row deltas in `ray_state_update`

Rows not listed are unchanged from §8 of the model doc. Notation per §8 plus:
$\mathrm{RPL}(\cdot)$ = replace `gainN`/`xprmatN` per §5 step 3; $\mathrm{XPH}$ = leg-closing
$e^{-j\Delta}$ fold (§3.5); $(\theta_t, d_v)$ = VBS-derived per §3.1; $S^{c} = S(\theta_t,
d_v)$ with the §3.4 phase fix.

**8.1 Pass selection / precedence** — unchanged.

**8.2 Reflection pass, internal reflection** (`cur != 0`, not resolved). VBS at the hit face;
internal mirror at $\theta_t$:

| Condition | Gain | `path_dirN` | State out |
|---|---|---|---|
| gates pass | $\mathrm{RPL}(r(\theta_t),\, d_v) \cdot S^{c}$ | reflected at VBS | set RESOLVED, as today |
| gates re-emit | $\mathrm{RPL}(r(\theta_t),\, d_v)$ | reflected at VBS | copy-through |

**8.3 Resolved-precedence rows.** i-o out-coupling: $\mathrm{RPL}(t_{21}(\theta_t),\, d_v)
\cdot \mathrm{XPH}$ — installs the previously implicit $t_{21}$, closing the Stokes product of
the reflected port; `path_dirN` snapped per 8.5. i-i resolved crossing:
$\mathrm{RPL}(\mathrm{TRN}(\mathrm{cur} \to iM,\ \theta_t),\, d_v) \cdot \mathrm{XPH}$;
`path_dirN` = refracted at VBS. o-i / edges: unchanged.

**8.4 o-i entry family.** **No gain/xprmat correction** — the entry was computed at the true
entry angle and is correct as produced (with the gate gone, this now includes EM
$\varepsilon < 1$ front faces). `path_dirN` from `ray_mesh_interact` is the refracted entry
direction; pass through unchanged. Wedge test, state writes, forward MED charge: unchanged.
Nested row: unchanged.

**8.5 i-o exit family, cavity exit** (`buf == 0`): VBS at the exit face; gain $=
\mathrm{RPL}(t_\mathrm{exit}(\theta_t),\, d_v) \cdot \mathrm{XPH} \cdot S^{c}$ (gates re-emit
$\Rightarrow$ drop $S^{c}$ only). $t_\mathrm{exit}(\theta_t)$ is the Stokes complement of the
entry and can never TIR — this row is where the honest $T = 0$ from `ray_mesh_interact` is
replaced (P1). For a single slab, the physically refracted exit direction realigns with the
entry; only a lateral exit-point shift remains (accepted). **`path_dirN` is replaced with the
`origN`→`destN` direction** (exact alignment for the single slab; drift from i-i transitions
is dropped here, keeping the traced geometry authoritative outside media). State back to
outside, as today. `false inside` / `buf != 0` rows: structure unchanged, transition angles
$\theta_t$ where used.

**8.6 / 8.7.** Slab-entry-both-faces: as 8.4 plus the existing wedge flag. Air gap: no Snell
correction in air; the bounding-mirror $S$ ingredients are geometric, and the gap distance is
$d(\mathrm{orig}_c, \mathrm{fbs})$ with `orig_correct` superseding under bends.

**8.8 M2M crossing** (`buf == 0`). The transition point is shifted and `fbs_angleN` is wrong
here — full correction: VBS at the transition face (FBS or SBS by out-type, §3.1), gates pass:
$\mathrm{RPL}(\mathrm{TRN}(\mathrm{cur} \to iM,\ \theta_t),\, d_v) \cdot
\mathrm{XPH}(\mathrm{cur\ leg}) \cdot S^{c} \cdot \sqrt{\mathrm{MED}(iM,\,
d(\mathrm{fbs},\mathrm{dest}) - \mathrm{off})}$; re-emit: same without $S^{c}$. RESOLVED
persist rule unchanged. `path_dirN` = refracted at VBS into $iM$ (i-i reflection rows:
reflected at VBS instead). `buf != 0` row unchanged.

**8.9 / 8.10 edge and multi-hit rows.** Unchanged in structure; transition evaluations use
$\theta_t$ where a crossing angle is needed. Edge hits never resolve, as today.

The unmatched-row KILL default is unchanged.

---

## 7. Input/output contracts and optionality

### 7.1 The path-direction chain

- Launch: tracer sets `path_dir_prev` = ray direction (outside condition).
- `ray_mesh_interact` emits `path_dirN` per §4.1.
- `ray_state_update` corrects `path_dirN` in place per §6.
- Tracer advances `path_dirN` → `path_dir_prev` for the next iteration, like the state words.

### 7.2 `orig_correct`

Full-set $[n_\mathrm{ray}, 3]$. Supersedes `orig` as the VBS ray origin and for in-medium
distances; everything else (geometric direction $\hat{u}_g$, vacuum-phase reference, segment
charges) keeps the true `orig`. Provided by `calc_diffraction_gain` for rays whose current
in-medium traversal contains a bend: the row carries the effective slab-entry origin (the
entry-event interaction point) so that $d_v$ and VBS see the unbent equivalent ray. Null rows /
null array: `orig` is used. Note: defaulting to `orig` under-measures the in-medium leg by the
1 mm relaunch offset, consistent with the existing $\pm$`off` conventions — documented, not
compensated.

### 7.3 Optionality and fallbacks

- **`path_dirN` absent:** no direction is exported or corrected; the corrections still run
  (VBS needs `path_dir_prev`, not `path_dirN`).
- **`path_dir_prev` absent:** $\theta_t$ is obtained from `fbs_angleN` and `mtl_ind_prev_in`
  via Snell under the **parallel-face assumption** ($\sin\theta_t = \sin\theta_\mathrm{hit}
  \cdot n_\mathrm{prev}/n_\mathrm{cur}$), and the corrected distance via $d_v =
  d(\mathrm{orig}_c, F)\cdot\cos\theta_\mathrm{hit}/\cos\theta_t$. Exact for isolated parallel
  slabs; disables multi-slab drift tracking and non-parallel entry correction (P5 degrades
  gracefully). This is the zero-extra-state path for simple tracers.
- **Both present:** full VBS treatment, exact for arbitrary face orientations and stacks.
- **`normal_vecN`:** mandatory (moved up next to `fbs_angleN` in the signature) — required
  for the VBS construction and the polarization basis.
- **Launched-inside (P7):** documented note only: seed `path_dir_prev` = launch direction and
  `orig_correct` = launch point.

---

## 8. Validation contract additions

All checks precede the parallel region, per the existing contract:

- `path_dir_prev` / `orig_correct`: size $[n_\mathrm{ray}, 3]$, finite; `path_dir_prev` rows
  renormalized defensively.
- `path_dirN`: size $[n_\mathrm{rayN}, 3]$ if supplied (it arrives pre-filled from
  `ray_mesh_interact`; resizing would destroy data — throw on mismatch instead).
- `ray_indN`: values $< n_\mathrm{ray}$, as for the existing map input.
- `normal_vecN`: required; throw if null.

---

## 9. Caller obligations

**SBR tracer.**

1. Seed `path_dir_prev` with the launch direction; advance `path_dirN` → `path_dir_prev` each
   iteration (or pass neither and accept the §7.3 parallel-face fallback).
2. **Pruning order:** do not drop zero-gain rays between `ray_mesh_interact` and
   `ray_state_update`. With the d2l gate removed, oblique type-1 exits legitimately return
   $T = 0$ (wrong-angle TIR) and are repaired by replacement at the exit row; early pruning
   silently deletes valid transmission paths. Prune after `ray_state_update`.

**Diffraction module (`calc_diffraction_gain`).**

1. Threads the path-direction chain like the tracer; **at an arc bend inside a medium**,
   rotate the carried `path_dir_prev` by the same minimal rotation that maps the old geometric
   segment direction onto the new one (one 3-vector rotation per bend — preserves the
   refraction offset between geometric and physical directions, which is what the exit physics
   needs). Bends touch no flags; the KNOWN-NON-PARALLEL flag keeps its single meaning, and slab
   resolution stays enabled on bent traversals.
2. **`orig_correct` bookkeeping:** for rays whose current traversal contains a bend, supply the
   effective slab-entry origin from its arc knowledge (§7.2).
3. Unprocessed sub-segments (`nH == 0`): continues to own their MED magnitude charges; no
   insertion phase there (phase is settled at the leg-closing event, §3.5).

---

## 10. Energy accounting and calibration impact

With the Snell-pair coefficients, a single slab satisfies $R + T = 1$ (lossless) and
$R + T + A = 1$ (lossy) coherently at every phase and oblique angle, matching the exact
transfer matrix (verified: machine precision at all tested phases at 20°, $\varepsilon = 5$,
where the previous convention spans $\Sigma = 0.82$–$2.24$). The stacked-slab persist rule and
its $R + T \le 1$ guarantee now hold at oblique incidence too.

**tf semantics are unchanged — no migration.** The existing scalar code already applies tf at
both faces of a traversal (the wall owns tf on its entry and on its back-side exit), and
calibrations expect this; the correction does not alter that convention, it only changes the
*angle* at which the exit Fresnel is evaluated. For $\varepsilon \ll 1$ rigid walls the
TF-modified TIR path carries the geometric direction (§2.1), so VBS coincides with FBS and the
exit evaluates at the same angle as today — **rigid-wall calibrations are preserved by
construction**. What changes: (i) propagating media (porous absorbers, dielectric slabs)
receive the intended oblique-angle value corrections at exits and M2M crossings; (ii) EM
type-1 through-slab totals change from the gate's one-sided $(1-R)$ to the physical
$(1-R)^2 |S|^2$ composition — EM tables hold physical constants (ITU P.2040 $\varepsilon_r$,
$\sigma$), not TL-calibrated parameters, so this needs a release note, not a migration. `att`
and the mass-law `m` convention are unchanged.

**M2M interfaces where both media define tf (new rule, fixes a latent nondeterminism).** The
current owner rule selects tf by hit side ($\theta \ge 0 \Rightarrow$ entered material,
$\theta < 0 \Rightarrow$ incidence material). At colocated M2M faces the hit side follows the
random FBS/SBS order (out-type 4 vs. 5), so when both media define tf, the applied tf is
order-dependent in the existing code. New deterministic rule: both boundary contributions
combine,

$$
R_\mathrm{eff} = R_0\,(1 - \mathrm{tf}_A)(1 - \mathrm{tf}_B),
$$

symmetric, independent of face order, reducing exactly to current behavior at every air
boundary ($\mathrm{tf}_\mathrm{air} = 0$, so o-i and i-o are untouched), and applied
identically on both passes and in the Airy mirrors so port complementarity holds (Q1 —
confirm).

---

## 11. Removed behaviors and remaining invariants

**Removed (retrievable via the pre-change tag):**

- The dense-to-light pass-through gate and all `dense2light` logic in `interact_with`,
  including the output parameter (completing on EM what was already done on the
  specular/scalar path).
- The distance-only Airy phase convention (replaced by the walk-off form, §3.4).
- Uncorrected wrong-angle values surviving at in-medium events (replaced at VBS by
  `ray_state_update`; `ray_mesh_interact` itself still computes and returns them honestly,
  including TIR, preserving its standalone contract).

**Unchanged invariants:** interaction-type codes, `out_typeN` codes, the three state words and
flag semantics, the full/compact index contract, the loss-ownership rule, resolved-precedence
logic, the stacked-slab persist rule, tf ownership, `interface_gain` single-count, the
validation-before-parallel rule, type-2 physics, entry-event (o-i) coefficients, and the
state-machine topology (every dispatch row keeps its condition, state writes, and structural
role).

---

## 12. Test plan

Tests of state logic, dispatch topology, flag semantics, index mapping, batch determinism,
float/double parity, and validation survive unchanged. Coefficient oracles at in-medium events
are re-derived; the old ones described the hack and retire with it.

1. **`path_dirN` contract.** Equals `origN`→`destN` for types 0/2/3 and all reflections;
   equals the Snell direction for type-1/4 transmissions; equals `origN`→`destN` under TIR (TF
   and non-TF); unit norm; TF-modified TIR carries tf-assigned power forward, including under
   type 2.
2. **VBS construction.** Against analytic plane intersections; FBS/SBS selection by out-type
   on colocated i-i hits with randomized face order; grazing/negative-$s$ guards.
3. **Snell-pair reciprocity and TIR impossibility at exits.** Via real-angle `interact_with`
   calls at $\theta_t$.
4. **Transfer-matrix ledger, oblique.** Both ports assembled across the two passes vs. the
   exact transfer matrix at a phase × angle grid, lossless and lossy; stacked slabs
   $R + T \le 1$.
5. **Airy walk-off phase.** $S(\theta_t, d_v)$ with the §3.4 fix vs. the closed-form Airy sum
   and the direct bounce-ray sum; fringe positions at 30°/60°.
6. **Insertion phase.** Phase exact at the call frequency vs. the transfer-matrix insertion
   phase; phase slope across a stored frequency grid equals the expected medium delay
   (per-frequency evaluation, §3.5); no magnitude double-count against the MED charges.
7. **Wedge and stack drift.** Tilted faces and two-slab stacks with `path_dir_prev` threaded:
   exit coefficients vs. a rotated-frame plane-wave oracle; the same cases under the
   parallel-face fallback documenting its error.
8. **Bend continuity.** Bend swept through a slab interior with `orig_correct` supplied: gain
   and phase continuous, converging to the unbent crossing; resolve decision invariant.
9. **Acoustic rigid wall ($\varepsilon \ll 1$).** Genuine front-face reflection on both
   passes; TF-modified TIR forward path with mass-law interior; bit-compatibility of the
   end-to-end wall transmission with the existing calibration (exit at the geometric angle).
   M2M crossings where both media define tf: combined rule of §10, deterministic under
   randomized FBS/SBS order.
10. **Pruning-order hazard.** Oblique type-1 transmission through a slab: correct nonzero
    end-to-end gain when pruning after `ray_state_update`; demonstrates the path loss when
    pruning early (documentation test).
11. **`ray_indN` map.** Ordered surviving indices, 0-based, round-trip into
    `ray_state_update`, drop rules, identity case.
12. **Stage-1 regression.** Refactored `ray_mesh_interact` (helpers extracted, new outputs
    added) bit-exact against the existing Catch2 suite; benchmark suite without performance
    regression.

---

## 13. Open questions for review

- **Q1 — Combined tf rule at M2M interfaces.** Confirm $R_\mathrm{eff} = R_0\,(1 -
  \mathrm{tf}_A)(1 - \mathrm{tf}_B)$ (§10): deterministic under FBS/SBS order, identical to
  current behavior at all air boundaries, port-complementary across both passes and the Airy
  mirrors. The current hit-side owner rule is order-dependent when both media define tf.

Resolved during review: tf semantics unchanged, no migration (§10); implementation in two
stages, EM polarization exact from stage 2 onward via stage-1-proven helpers (§5);
`ray_state_update` has no existing wrappers and `ray_mesh_interact`'s appended optional
outputs are non-breaking — binding exposure follows at leisure.
