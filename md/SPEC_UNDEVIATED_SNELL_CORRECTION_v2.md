# SPEC: Snell-Correct Physics for Undeviated Paths (VBS + Segment Architecture)

*Change specification for `ray_mesh_interact.cpp` — `ray_mesh_interact`, `ray_state_update`,
`Material::interact_with`, `Material::slab_airy_factor`.*

Status: **draft for review (rev 2)**. Companion to `quadriga_lib_material_model.md` ("the model
doc", §N references). No implementation yet.

**Scope.** `ray_mesh_interact` changes surgically: the d2l gate is removed, TIR is computed and
returned honestly, the tf reflection-to-transmission rule is unified onto EM, and two new outputs
(`path_dirN`, `ray_indN`) are appended. All Snell corrections live in `ray_state_update`, which
**replaces** the affected per-hit values with quantities recomputed at the virtual bounce point
(VBS) and accumulates in-medium loss across segments so the result is independent of how a path is
cut. No compatibility mode; the previous behavior remains reachable through the version tag
preceding this change.

**Not in this change.** There is no `orig_correct` input and no `excess_delayN` output. Earlier
drafts carried both; the VBS-plane-normal mechanism (§3.1, §7) makes `orig_correct` unnecessary,
and the in-medium excess phase folds into `xprmatN` (§3.5) rather than a separate output. Any such
declaration surviving in the source doc block is stale and load-bearing on nothing.

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
the physically refracted angle and position. P1–P8 derive from this; P9 is the segmentation
requirement that ties the SBR tracer and the diffraction module to one correctness criterion.

- **P1 — Spurious TIR at the exit face (EM type 1).** For a slab with $\varepsilon = 5$, every
  ray steeper than $\arcsin(1/\sqrt{5}) \approx 26.6°$ from normal hits the wrongly evaluated
  exit critical angle. Physically, TIR at the exit of a slab entered from the same medium is
  impossible: $n \sin\theta_t = \sin\theta_0 < 1$. With the d2l gate removed, the wrong-angle exit
  returns honest $T = 0$ — then **replaced** by the state machine with the VBS-derived exit
  coefficient at $\theta_t$, which can never TIR. Pipeline consequence: the tracer must not prune
  zero-gain rays between the two calls (§9).
- **P2 — The dense-to-light pass-through gate is a hack on a wrong proxy.** It keys on density
  ordering, not on entry/exit. Removed from `interact_with` on the EM path (already done on the
  specular/scalar path). Acoustic $\varepsilon \ll 1$ front faces keep their genuine
  $|R| \to 1$; the energy creation from gate + Airy (assembled ledger up to $R + T = 2.24$ at slab
  resonance, 20°, $\varepsilon = 5$, exact 1.000) disappears.
- **P3 — The Airy factor uses wrong angle and wrong distance.** With the VBS-derived
  $(\theta_t, d_v)$ the mirrors and the per-bounce loss become correct through the *existing*
  `slab_airy_factor` parameterization. One internal fix is mandatory: the round-trip phase must
  include the lateral walk-off, $\arg\varphi = -k_0 n_\mathrm{re} d_v \cos^2\theta_t$ instead of
  $-k_0 n_\mathrm{re} d_v$ (§3.4).
- **P4 — In-medium excess phase is missing.** The tracer applies a fixed speed ($c_0$ EM,
  $c_\mathrm{sound}$ acoustic) to the total geometric length; a traversed medium adds excess phase
  applied nowhere. Fixed by a per-leg factor folded into `xprmatN` (§3.5), evaluated at each
  call's frequency so multi-frequency pipelines carry the correct in-band delay slope.
- **P5 — Non-parallel faces and multi-slab drift.** At real face crossings the VBS uses the real
  face normal, so the exit/transition angle and coefficient are exact regardless of face
  orientation. The parallel-face approximation enters only the in-medium *distance* of segments
  that lie wholly or partly inside a body without a crossing, where the VBS plane carries the
  entry normal (§3.1, §7.2).
- **P6 — In-medium bends and arbitrary segmentation.** $L = d(\mathrm{orig}, \mathrm{fbs})$ of a
  single event is not the slab path when a traversal is cut into segments (a diffraction arc, or
  any tracer that does not relaunch on faces). Solved by accumulating the per-segment VBS distance
  $d_v$ into a threaded total `acc_dist` and charging the in-medium loss and the Airy length once
  at the layer-closing event (§3.3, §6). No origin bookkeeping is needed: each segment uses its
  own `orig`, the carried `normal_vecN`, and `path_dir_prev`.
- **P7 — Rays launched inside a medium.** A caller seeds `path_dir_prev` with the launch direction
  and `normal_vecN` with the bounding-face normal; the general machinery covers it. No dedicated
  code.
- **P8 — Convention for the reflection pass: not an issue.** The architecture self-disambiguates
  per ray: for type-2 (and types 0/2-paired) rays, `path_dirN` equals the geometric direction, so
  VBS coincides with the geometric hit, $\theta_t$ equals the hit angle, and every correction
  degenerates to an identity.
- **P9 — Segmentation invariance (correctness criterion).** A straight path produces the same
  output whether processed as one segment (SBR, dest far) or many collinear segments (the
  diffraction spine, 10×10 m), for any scene. SBS is only a *classifier* (colocation, i-i,
  degenerate geometry); neither method sees a slab's two faces in one call as a reliable pair, and
  both process face-by-face through threaded state. The only thing that differs is where segment
  boundaries fall — including inside a body. Invariance holds iff every charged quantity is either
  composable (charge per segment) or a path functional accumulated and charged once. Composable:
  dielectric and power-law loss, the per-face interface coefficients. Non-composable: the **mass
  law** (logarithmic in distance: $G_\mathrm{med}(L_1)\,G_\mathrm{med}(L_2) \neq
  G_\mathrm{med}(L_1{+}L_2)$), its 1.5 mm clamp, and the **Airy length** $L$. The accumulator
  `acc_dist` makes the non-composable quantities functions of the full in-layer path, closing P9
  and, as a side effect, the double-charged mass law on a reflection round-trip.

---

## 2. Design overview

1. **`ray_mesh_interact` — surgical change.** Remove the d2l gate; compute and return TIR honestly
   ($T = 0$, except the TF-modified TIR rule below); unify the tf reflection-to-transmission rule
   onto EM (§4.3); expose `path_dirN` (the normalized direction the path would take from the FBS
   if refraction were modeled correctly — already computed internally for the Fresnel geometry)
   and the `ray_indN` convenience map. For types 0/2/3 and reflections, `path_dirN` equals the
   `origN`→`destN` direction; it deviates only on type-1/4 transmissions. Nothing else changes;
   the standalone contract stays meaningful.

   **TF-modified TIR rule:** the general rule is that tf shifts reflection energy into
   transmission; TIR ($R = 1$) is its edge case. When tf opens a forward port under TIR, the
   energy goes into an **undeviated forward path**: power assigned via tf, `path_dirN` aligned with
   `origN`→`destN`. This is the mechanism that feeds energy into mass-law media without breaking
   conservation, and it makes acoustic $\varepsilon \ll 1$ walls work end to end: entry = genuine
   near-total reflection + tf-opened undeviated transmission; interior = mass law; exit at the
   geometric angle. For non-TF TIR, $T = 0$ and `path_dirN` is the `origN`→`destN` direction for
   determinism (the tracer drops the path anyway).

2. **`ray_state_update` — VBS correction with segment accumulation.** The state machine carries the
   physical direction (`path_dir_prev`, in) and corrects `path_dirN` (in/out) in place. At each
   in-medium event it constructs the **virtual bounce scatterer**: the point where the physically
   directed ray ($\mathrm{orig}$, `path_dir_prev`) meets the VBS plane — the real hit face at a
   crossing, or a carried virtual plane at a no-crossing segment (§3.1). From VBS it derives the
   incidence angle $\theta_t$, the per-segment in-medium distance $d_v$, and the continuation
   direction. It **replaces** the interface contribution of `gainN`/`xprmatN` with values at
   $\theta_t$, **accumulates** $d_v$ into `acc_dist`, and charges the in-medium loss and the
   Fabry-Pérot factor once at the layer-closing event over the accumulated total. Replacement is
   computed by shared helpers (§5).

3. **`normal_vecN` — the VBS plane normal (dual role).** At a real crossing it is the hit-face
   normal (as today). At a no-crossing segment (`nH == 0`) `ray_mesh_interact` returns no normal,
   so the caller supplies the **carried entry normal**; the VBS plane then sits through the
   segment's `dest`, parallel to the entry face, until a real FBS is detected. SBR always detects
   an FBS and projects onto its plane; segmented callers may traverse several virtual planes before
   the exit. This single mechanism replaces `orig_correct`.

4. **`acc_dist` — threaded in-medium path length.** A per-ray accumulator (in/out), summing the
   VBS distance $d_v$ of each in-medium segment. Reset at every event that changes the current
   medium (i-o exit, i-i transition) after charging the leaving layer's loss; carried across
   no-crossing segments and relaunches. Optional: when absent, the closing event charges over its
   own $d_v$ (correct for relaunch-aligned tracers such as the SBR loop, which present the full
   in-layer path as a single event's $d_v$). The two modes coincide for any single physical path.

5. **The RT loop.** At launch the tracer sets `path_dir_prev` to the ray direction. After each
   interaction it advances `path_dirN` → `path_dir_prev` and `acc_dist_outN` → `acc_dist_in`,
   exactly as it threads the three state words.

---

## 3. Physics definitions

Permittivities: interface values with resonance for Fresnel, resonance-excluded for propagation,
as in §2.2/§7.5 of the model doc. $k_0 = \omega/c_0$. Grazing conventions as in the existing code;
formulas use the incidence angle from the normal for readability.

### 3.1 The VBS construction

At an in-medium event (`cur != 0`), let $\hat{d}_p$ = `path_dir_prev` row (renormalized),
$P_0$ = `orig` row, and $(F, \hat{n})$ = the VBS plane point and normal:

- **Real crossing** (`nH >= 1`, a face is hit): $F$ = `fbs`, $\hat{n}$ = the transition face
  normal from `normal_vecN` — **FBS normal for out-type 4, SBS normal for out-type 5** (colocated
  i-i hits report the two faces in random order; select by out-type, not array position).
- **No crossing** (`nH == 0`, segment lies inside the body): $F$ = `dest`, $\hat{n}$ =
  `normal_vecN` as supplied by the caller (the carried entry normal). The plane is parallel to the
  entry face.

$$
s = \frac{(F - P_0) \cdot \hat{n}}{\hat{d}_p \cdot \hat{n}},
\qquad
\mathrm{VBS} = P_0 + s\,\hat{d}_p,
\qquad
d_v = s,
\qquad
\cos\theta_t = |\hat{d}_p \cdot \hat{n}| .
$$

Guards: if $|\hat{d}_p \cdot \hat{n}| < 10^{-6}$ (division protection) or $s \le 0$ or $s$
non-finite, fall back to the geometric values ($\theta$ from `fbs_angleN`; $d_v = d(P_0, F)$ at a
crossing, $d_v = d(\mathrm{orig}, \mathrm{dest})$ at `nH == 0`). The threshold is deliberately
tiny — a larger one would create a corrected-to-wrong jump in angle sweeps. The grazing limit
needs no protection: the thickness $t = d_v\cos\theta_t = (F - P_0)\cdot\hat{n}$ stays bounded
(only the slant $d_v$ grows), and the in-medium gain and survival gate drive smoothly to zero,
matching physical $T \to 0$.

The corrected continuation at VBS follows §4.3: Snell refraction for transmissive events, mirror
reflection for reflective events, both via the shared helpers so the convention matches
`ray_mesh_interact`.

### 3.2 Corrected interface coefficients

`interact_with` is called at the real angle: `interact_with(cur, next, type, theta_t, ...)`. By
Snell-pair reciprocity ($r_{21}(\theta_t) = -r_{12}(\theta_0)$, verified numerically) the slab
exit at $\theta_t$ reproduces the entry reflectance, can never TIR for a traversal entered from
outside, and the scalar branch yields the Stokes-complementary $t_\mathrm{exit} = \sqrt{1 -
R_\mathrm{eff}}$ with the correct phase. tf ownership and `interface_gain` fold-in are unchanged
except the combined M2M rule of §10.

### 3.3 Corrected in-medium propagation and accumulation

The interface coefficients are **per-face point events**: charged once at each crossing,
segmentation-invariant. The in-medium loss is a **path functional**: it is deferred and charged
once per layer at the layer-closing event over `acc_dist`.

- Each in-medium segment adds its VBS distance to the accumulator: `acc_dist += d_v` (§3.1).
- Entry and interior segments charge **no** in-medium loss (they only accumulate and apply the
  interface coefficient, if any).
- The layer-closing event (i-o exit; i-i transition for the layer being left; or the terminal
  event of a ray that ends inside a body) charges $G_\mathrm{med}(\mathrm{acc\_dist}, f,
  \cos\theta_t)$ once, then resets `acc_dist`.
- When `acc_dist` is absent, the closing event charges over its own $d_v$ (relaunch-aligned mode).

`medium_gain` is called unchanged. Charging once over the total makes the dielectric, power-law,
**and** mass-law terms segmentation-invariant simultaneously, removes the 1.5 mm clamp sensitivity
to where boundaries fall, and applies the mass law once over a reflection round-trip instead of
twice.

### 3.4 Fabry-Pérot factor

`slab_airy_factor` keeps its signature and is called with the VBS-derived pair $(\theta_t, L)$,
where $L = \mathrm{acc\_dist}$ — the full one-way in-slab path of the traversal, not the closing
segment's $d_v$. With these inputs the mirrors (slab-side Fresnel at $\theta_t$) and the per-bounce
magnitude are correct as the function stands. **One mandatory internal fix:** the one-way phase
must include the lateral walk-off,

$$
\arg\varphi = -k_0\, n_\mathrm{re}\, L \cos^2\theta_t
\qquad \big(= -k_0\, n_\mathrm{re}\, t \cos\theta_t,\ t = L\cos\theta_t\big),
$$

replacing the current $-k_0 n_\mathrm{re}\,\mathrm{dist}$. Verified against the bounce-ray sum and
the transfer matrix to machine precision; the distance-only form is wrong off-normal even with the
corrected distance. The cosine derives from the `theta` argument the function already receives.
Magnitude, pole clamp ($|\mathrm{denom}| \ge 10^{-2}$), survival gate
($\rho = \sqrt{R_n R_f\, G_\mathrm{med}(2L)} \ge \varepsilon_\mathrm{thr}$), and re-emit semantics
(NaN) are unchanged. **All six existing call sites must pass $L = \mathrm{acc\_dist}$ instead of
`distance(orig, fbs)`**, or the walk-off fix double-counts the angle.

### 3.5 In-medium excess phase

Per traversal leg, the excess phase relative to the vacuum phase the tracer counts along the
geometric segments, folded into `xprmatN` at the leg-closing event via the complex form of
`rsu_scale` (§5):

$$
\Delta = k_0\, \big( n_\mathrm{re}\, L_\mathrm{phys} - L_\mathrm{geo} \big),
\qquad L_\mathrm{phys} = \mathrm{acc\_dist},
$$

with $L_\mathrm{geo}$ the geometric in-layer path the tracer counted vacuum phase over (for SBR,
the relaunch-aligned segment; for a segmented caller, the summed geometric segment lengths).
Applied as $e^{-j\Delta}$ at the layer close. It is phase-only ($|e^{-j\Delta}| = 1$, so
`rsu_scale` leaves `gainN` untouched); magnitudes stay with the in-medium ownership of §3.3.

**Frequency handling:** $\Delta \propto k_0$ and is evaluated at each call's frequency — never at
$f_c$ and reused. Pipelines storing the polarization matrix per frequency then carry the medium's
in-band delay slope automatically. The geometry-derived per-path delay scalar keeps a small,
documented bias (~1.3 ns per 30 cm concrete wall on 20+ dB paths; ~0.1 ms on acoustic transmission
at $-40$ dB).

---

## 4. API changes

### 4.1 `ray_mesh_interact`

Behavior: d2l gate removed; TIR computed and returned honestly; tf reflection-to-transmission rule
unified onto EM (§4.3); two new optional outputs. Everything else unchanged.

```cpp
template <typename dtype>
void quadriga_lib::ray_mesh_interact(int interaction_type,
                                     dtype center_frequency,
                                     /* ... existing parameters unchanged ... */,
                                     arma::Mat<dtype> *path_dirN = nullptr,  // [n_rayN, 3], physical continuation at FBS
                                     arma::u32_vec *ray_indN = nullptr);     // [n_rayN], compact-to-full ray index map
```

- **`path_dirN`**: normalized continuation direction. Type-1/4 transmissions: the Snell-refracted
  direction, deviating from `origN`→`destN`. Types 0/2/3 and all reflections: equal to
  `origN`→`destN`. TIR rows: equal to `origN`→`destN` (TF-modified TIR by §2.1; non-TF for
  determinism).
- **`ray_indN`**: compact-to-full surviving-ray index map, 0-based, order-preserving — the inverse
  of the internal `output_ray_index` (currently built and discarded), in the format
  `ray_state_update` consumes.

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
    const arma::Mat<dtype> *normal_vecN,                 // [n_rayN, 6]: VBS plane normals; required
    const arma::s32_vec *out_typeN,
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::Col<short> *mtl_ind_fbs,
    const arma::Col<short> *mtl_ind_sbs,
    const arma::Col<short> *mtl_ind_prev_in = nullptr,
    const arma::Col<short> *mtl_ind_current_in = nullptr,
    const arma::Col<short> *mtl_ind_buffer_in = nullptr,
    const arma::Mat<dtype> *path_dir_prev = nullptr,     // [n_ray, 3]: physical direction entering this segment
    const arma::Col<dtype> *acc_dist_in = nullptr,       // [n_ray]: accumulated in-layer VBS distance
    arma::Col<short> *mtl_ind_prev_outN = nullptr,
    arma::Col<short> *mtl_ind_current_outN = nullptr,
    arma::Col<short> *mtl_ind_buffer_outN = nullptr,
    arma::Col<dtype> *gainN = nullptr,
    arma::Mat<dtype> *xprmatN = nullptr,
    arma::Mat<dtype> *path_dirN = nullptr,               // [n_rayN, 3]: from ray_mesh_interact, corrected in place
    arma::Col<dtype> *acc_dist_outN = nullptr,           // [n_rayN]: accumulated distance out
    const arma::u32_vec *ray_indN = nullptr,
    double eps = 0.15);
```

`normal_vecN` moves up next to `fbs_angleN` and is required (it is the VBS plane normal, not just
the wedge-test input). Names ending in `N` are compact-set ($n_\mathrm{rayN}$); others are
full-set ($n_\mathrm{ray}$). All new arguments are optional refinements; the corrected physics
applies always, with the fallbacks of §7.3 when inputs are absent. **No `orig_correct`, no
`excess_delayN`.**

### 4.3 `Material::interact_with`

The d2l gate and all `dense2light` logic vanish, including the output parameter. The general tf
rule (reflection energy redistributed to transmission) is applied on the EM type-1 path as it
already is on scalar: $R_\mathrm{eff} = \mathrm{apply\_tf}(R)$, forward gain $= 1 - R_\mathrm{eff}$.

**TF-TIR coefficient fix (bug today).** Under TIR, `reflection_gain = 1` and `refraction_gain =
0`, so the existing `if (refraction_gain > 0)` rescale is skipped: the forward port would carry
gain $= \mathrm{tf}$ but **zero** coefficients, breaking the $\tfrac{1}{2}\sum|x_{ij}|^2 = $ gain
identity. When tf opens a forward port and `refraction_gain == 0`, write **flat undeviated**
coefficients ($c_\mathrm{TE} = c_\mathrm{TM} = \sqrt{\text{forward gain}}$, scaled by
$\sqrt{\text{interface\_gain}}$), polarization-preserving, direction `origN`→`destN`. The
undeviated outgoing basis equals the incoming basis, so the sandwich is identity and the identity
holds. All corrected calls pass the real angle $\theta_t$.

### 4.4 `Material::slab_airy_factor`

Signature unchanged. One internal change: the walk-off phase of §3.4
($\arg\varphi = -k_0 n_\mathrm{re}\,\mathrm{dist}\cdot\cos^2\theta_\mathrm{inc}$, with the cosine
from the `theta` argument). Callers pass `dist` $= \mathrm{acc\_dist}$ (§3.3, §3.4).

### 4.5 `Material::medium_gain`

Unchanged.

---

## 5. Correction procedure and shared helpers

At each in-medium event of types 0/1/3/4 (type-2 rows degenerate to identities through the data,
P8), `ray_state_update`:

1. selects the VBS plane point and normal (§3.1: real face at a crossing, virtual plane through
   `dest` at `nH == 0`; FBS/out-4, SBS/out-5),
2. constructs VBS, $\theta_t$, $d_v$ with the §3.1 guards,
3. **replaces** the interface contribution of `gainN`/`xprmatN` — coefficients at $\theta_t$,
   `interface_gain`, polarization basis from the physical incoming (`path_dir_prev`) and outgoing
   (VBS continuation) directions — leaving the in-medium magnitude **out** (deferred),
4. **accumulates** `acc_dist += d_v`,
5. at the layer-closing event, multiplies in $G_\mathrm{med}(\mathrm{acc\_dist})$, the Fabry-Pérot
   factor $S(\theta_t, \mathrm{acc\_dist})$, and the leg-closing $e^{-j\Delta}$, then resets
   `acc_dist`,
6. corrects `path_dirN` to the VBS continuation (refracted or reflected); at i-o exits it is
   snapped to `origN`→`destN` per §6/8.5.

**Operation order and helpers.** `RPL` writes a fresh interface result (the `qd_polbasis`
sandwich); `rsu_scale(cr, ci)` then folds the complex $S$ and the phase-only $e^{-j\Delta}$,
multiplying `xprmatN` by the complex factor and `gainN` by $|c|^2$ (so a unit-modulus phase leaves
`gainN` exact). `RPL` is **not** the existing scalar `rsu_replace`, which writes a magnitude into
VV/HH only; the EM replacement needs the full Jones via `qd_polbasis`. Order: `RPL` →
$\times S$ → $\times e^{-j\Delta}$.

**DRY requirement:** everything needed by both functions moves into anonymous-namespace helpers in
`ray_mesh_interact.cpp`, and `ray_mesh_interact` calls them:

- `qd_refract(û, n̂, ratio)` — Snell direction (incl. the $\mathrm{Re}\cos\theta_t$ treatment).
- `qd_reflect(û, n̂)` — mirror direction.
- `qd_polbasis(...)` — the U·diag(c_TE, c_TM)·Q sandwich. The **outgoing** basis U is built from
  the direction ultimately stored in `path_dirN` (refracted at an i-i, geometric at an i-o exit),
  and the incoming basis Q from `path_dir_prev`; coefficients at $\theta_t$. This makes the frame
  the `xprmat` projects into identical to the frame the stored direction defines, so there is no
  silent reinterpretation downstream. The residual Q-vs-U plane inconsistency is $O(\delta)$ (the
  geometric-vs-physical angle drift) and is accepted.
- `qd_hit_charges(...)` — the §4.4 per-hit charge composition.
- `qd_vbs(...)` — the §3.1 plane intersection with guards.

After an EM replacement, `gainN` is $\tfrac{1}{2}\sum|x_{ij}|^2$ from the new `xprmatN`; scalar
rows write the single coefficient and $|x|^2$.

**Implementation staging.** Stage 1: extract the helpers and refactor `ray_mesh_interact` onto
them, add `path_dirN`/`ray_indN`; bit-exact under the Catch2 suite, no benchmark regression. Stage
2: remove the d2l gate, unify the tf rule (with the TF-TIR coefficient fix), apply the combined
M2M tf rule (§10), and implement the VBS + `acc_dist` corrections, reusing the stage-1 helpers plus
`qd_vbs`. The d2l removal belongs to stage 2: removing it without the replacement would kill
oblique type-1 transmissions (P1).

---

## 6. Dispatch-row deltas in `ray_state_update`

Rows not listed are unchanged from §8 of the model doc. $\mathrm{RPL}(\cdot)$ = replace the
interface part of `gainN`/`xprmatN` per §5 step 3 (in-medium magnitude deferred); $\mathrm{ACC}$ =
add $d_v$ to `acc_dist`; $\mathrm{CLOSE}$ = charge $G_\mathrm{med}(\mathrm{acc\_dist})$, fold
$S^{c} = S(\theta_t, \mathrm{acc\_dist})$ with the §3.4 phase fix and the §3.5 $e^{-j\Delta}$, then
reset `acc_dist`; $(\theta_t, d_v)$ = VBS-derived per §3.1.

**8.1 Pass selection / precedence** — unchanged.

**8.2 Reflection pass, internal reflection** (`cur != 0`, not resolved). VBS at the hit face;
internal mirror at $\theta_t$:

| Condition | Action | `path_dirN` | State out |
|---|---|---|---|
| gates pass | $\mathrm{RPL}(r(\theta_t)) \cdot S^{c}$; $\mathrm{ACC}$; the resolving reflection itself charges no in-medium magnitude (the seam, §7.6 of the model doc) | reflected at VBS | set RESOLVED |
| gates re-emit | $\mathrm{RPL}(r(\theta_t))$; $\mathrm{ACC}$ | reflected at VBS | copy-through |

**8.3 Resolved-precedence rows.** i-o out-coupling: $\mathrm{RPL}(t_{21}(\theta_t))$;
$\mathrm{ACC}$; $\mathrm{CLOSE}$ (incoming-leg loss closes here). i-i resolved crossing:
$\mathrm{RPL}(\mathrm{TRN}(\mathrm{cur} \to iM,\ \theta_t))$; $\mathrm{ACC}$; $\mathrm{CLOSE}$ for
the leaving layer, then begin accumulating $iM$; `path_dirN` = refracted at VBS. o-i / edges:
unchanged.

**8.4 o-i entry family.** **No interface correction** beyond what `ray_mesh_interact` produced at
the true entry angle (with the gate gone, this includes EM $\varepsilon < 1$ front faces).
$\mathrm{ACC}$ the forward in-segment distance; **charge no in-medium loss here** (deferred).
`path_dirN` from `ray_mesh_interact` is the refracted entry direction; pass through. Wedge test,
state writes: unchanged. If the segment ends inside the body (`dest` interior, no exit on this
segment), the accumulation carries forward; the loss closes at a later event.

**8.5 i-o exit family, cavity exit** (`buf == 0`): VBS at the exit face;
$\mathrm{RPL}(t_\mathrm{exit}(\theta_t))$; $\mathrm{ACC}$; $\mathrm{CLOSE}$ (gates re-emit ⇒ drop
$S^{c}$ only). $t_\mathrm{exit}(\theta_t)$ is the Stokes complement of the entry and can never TIR
— this row is where the honest $T = 0$ from `ray_mesh_interact` is replaced (P1).
**`path_dirN` is replaced with the `origN`→`destN` direction** (exact for the single slab; i-i
drift dropped here, keeping the traced geometry authoritative outside media). State back to
outside. `false inside` / `buf != 0` rows: structure unchanged, $\theta_t$ where used.

**8.6 / 8.7.** Slab-entry-both-faces: as 8.4 plus the wedge flag. Air gap: no Snell correction in
air; bounding-mirror $S$ ingredients are geometric; the gap distance accumulates through
`acc_dist` like any layer.

**8.8 M2M crossing** (`buf == 0`). The transition point is shifted and `fbs_angleN` is wrong here
— full correction: VBS at the transition face (FBS or SBS by out-type, §3.1);
$\mathrm{RPL}(\mathrm{TRN}(\mathrm{cur} \to iM,\ \theta_t))$; $\mathrm{ACC}$; $\mathrm{CLOSE}$ the
leaving layer, fold $S^{c}$, then begin accumulating $iM$ with the entered-medium forward segment.
RESOLVED persist rule unchanged. `path_dirN` = refracted at VBS into $iM$ (i-i reflection rows:
reflected at VBS). `buf != 0` row unchanged.

**8.9 / 8.10 edge and multi-hit rows.** Unchanged in structure; transition evaluations use
$\theta_t$ where a crossing angle is needed; in-medium magnitude accumulates and closes per the
same rule. Edge hits never resolve, as today.

**`nH == 0` (no-crossing interior segment).** Not a dispatch row today; add it: build the VBS on
the virtual plane (§3.1), $\mathrm{ACC}$ its $d_v$, apply the identity interface (no face), copy
state through. This is the row that makes a fully-interior segment a no-op beyond its distance
contribution.

The unmatched-row KILL default is unchanged.

---

## 7. Input/output contracts and optionality

### 7.1 The path-direction chain

- Launch: tracer sets `path_dir_prev` = ray direction.
- `ray_mesh_interact` emits `path_dirN` (§4.1).
- `ray_state_update` corrects `path_dirN` in place (§6) and writes `acc_dist_outN`.
- Tracer advances `path_dirN` → `path_dir_prev` and `acc_dist_outN` → `acc_dist_in`, like the
  state words.

### 7.2 `normal_vecN` (the VBS plane normal)

Compact-set $[n_\mathrm{rayN}, 6]$, required. At a real crossing the two triples are the FBS and
SBS face normals (as today). At `nH == 0` `ray_mesh_interact` returns nothing, so the caller fills
the FBS triple with the **carried entry-face normal**; the VBS plane then sits through `dest`
parallel to the entry face. SBR fills it from `ray_mesh_interact` at every event (always a real
crossing). The diffraction module carries the entry normal across its interior segments. This
mechanism, plus each segment's own `orig` and `acc_dist`, removes any need for an entry-point
input.

### 7.3 Optionality and fallbacks

- **`path_dirN` absent:** no direction exported or corrected; corrections still run (VBS needs
  `path_dir_prev`).
- **`acc_dist_in`/`acc_dist_outN` absent:** the closing event charges in-medium loss and the Airy
  length over its own $d_v$ (relaunch-aligned mode). Correct for tracers that relaunch on faces
  (SBR); segmented callers must thread the accumulator or the mass law and Airy length split.
- **`path_dir_prev` absent:** $\theta_t$ from `fbs_angleN` and the masked state words via Snell
  under the parallel-face assumption ($\sin\theta_t = \sin\theta_\mathrm{hit}\cdot
  n_\mathrm{prev}/n_\mathrm{cur}$, with $n_k = \mathrm{Re}\sqrt{\varepsilon_k\mu_k}$ from
  `Material(prev_mat)` and `Material(cur)`, `prev_mat = 0` → air); $d_v =
  d(\mathrm{orig}, F)\cdot\cos\theta_\mathrm{hit}/\cos\theta_t$. Exact for isolated parallel slabs;
  disables drift tracking. This is the zero-extra-state path.
- **`normal_vecN` absent:** corrections fall back to the geometric per-event values (no VBS). The
  wedge test is also disabled.
- **Launched-inside (P7):** seed `path_dir_prev` = launch direction, `normal_vecN` = bounding-face
  normal.

---

## 8. Validation contract additions

All checks precede the parallel region:

- `path_dir_prev`: size $[n_\mathrm{ray}, 3]$, finite, rows renormalized defensively.
- `acc_dist_in`: size $[n_\mathrm{ray}]$, finite, $\ge 0$.
- `normal_vecN`: size $[n_\mathrm{rayN}, 6]$; required, throw if null.
- `path_dirN`: size $[n_\mathrm{rayN}, 3]$ if supplied (arrives pre-filled; throw on mismatch
  rather than resize).
- `acc_dist_outN`: size $[n_\mathrm{rayN}]$ if supplied.
- `ray_indN`: values $< n_\mathrm{ray}$.

---

## 9. Caller obligations

**SBR tracer.**

1. Seed `path_dir_prev` with the launch direction; advance `path_dirN` → `path_dir_prev` each
   iteration. `normal_vecN` comes from `ray_mesh_interact` (always a real crossing). `acc_dist`
   may be left null (relaunch-aligned mode).
2. **Pruning order:** do not drop zero-gain rays between `ray_mesh_interact` and
   `ray_state_update`. With the d2l gate removed, oblique type-1 exits legitimately return $T = 0$
   and are repaired by replacement at the exit row. Prune after `ray_state_update`.

**Diffraction module (`calc_diffraction_gain`).**

1. Thread `path_dir_prev` (advanced from `path_dirN`) and `acc_dist` across all segments of a
   traversal, like the state words.
2. **Fill `normal_vecN` at `nH == 0` segments** with the carried entry-face normal so the VBS
   plane stays parallel to the entry face until a real exit FBS. At real crossings use the face
   normal as usual.
3. **Forward-only resolution.** The module has no reflection pass and cannot relaunch, so it must
   always resolve internal reflections analytically: set `eps = 0` (the survival gate is
   `re-emit when rho < eps`, so `eps = 0` always resolves; `eps >= 1` never resolves) and force
   the parallelism flag on (`parallel_ok = true`) so a non-parallel slab cannot re-emit and lose
   energy. For the segmentation-invariance test (§12), run the SBR side in the same forced-resolve
   configuration so neither side re-emits — a re-emit spawns a reflection ray a forward-only run
   cannot reproduce.

---

## 10. Energy accounting and calibration impact

With the Snell-pair coefficients a single slab satisfies $R + T = 1$ (lossless) and $R + T + A =
1$ (lossy) coherently at every phase and oblique angle, matching the transfer matrix (the previous
convention spanned $\Sigma = 0.82$–$2.24$ at 20°, $\varepsilon = 5$). The stacked-slab persist
rule and its $R + T \le 1$ guarantee now hold at oblique incidence.

**tf semantics — no migration.** The scalar code already applies tf at both faces of a traversal
and calibrations expect this; the correction only changes the *angle* at which the exit Fresnel is
evaluated. For $\varepsilon \ll 1$ rigid walls the TF-modified TIR path carries the geometric
direction (§2.1), so VBS coincides with the geometric hit and the exit evaluates at the same angle
as today — rigid-wall calibrations are preserved by construction. What changes: (i) propagating
media (porous absorbers, dielectric slabs) receive the intended oblique-angle corrections at exits
and M2M crossings; (ii) EM type-1 through-slab totals change from the gate's one-sided $(1-R)$ to
the physical $(1-R)^2 |S|^2$ — EM tables hold physical constants (ITU P.2040 $\varepsilon_r$,
$\sigma$), not TL-calibrated parameters, so this needs a release note, not a migration. `att` and
the mass-law `m` convention are unchanged.

**M2M interfaces where both media define tf (new deterministic rule).** The current owner rule
selects tf by hit side, which at colocated M2M faces follows the random FBS/SBS order — so when
both media define tf, the applied tf is order-dependent. New symmetric rule, with $\mathrm{tf}^+ =
\max(\mathrm{tf}, 0)$ and $\mathrm{tf}^- = \max(-\mathrm{tf}, 0)$:

$$
R_\mathrm{leak} = R_0\,(1 - \mathrm{tf}_A^+)(1 - \mathrm{tf}_B^+),
\qquad
R_\mathrm{eff} = R_\mathrm{leak} + (1 - R_\mathrm{leak})\,\max(\mathrm{tf}_A^-, \mathrm{tf}_B^-).
$$

Symmetric; reduces exactly to `apply_tf` at any air boundary ($\mathrm{tf}_\mathrm{air} = 0$, both
signs); stays in $[0, 1]$; a perfect mirror on either face ($\mathrm{tf} = -1$) gives
$R_\mathrm{eff} = 1$. Applied identically on both passes and in the Airy mirrors so port
complementarity holds (Q1 — confirm against the ledger).

---

## 11. Removed behaviors and remaining invariants

**Removed (retrievable via the pre-change tag):**

- The dense-to-light pass-through gate and all `dense2light` logic in `interact_with`, including
  the output parameter.
- The distance-only Airy phase convention (replaced by the walk-off form, §3.4).
- The `orig_correct` input and the `excess_delayN` output of earlier drafts — never implemented;
  the VBS-plane-normal mechanism and the `xprmatN` phase fold replace them.
- Per-segment in-medium charging at in-medium events (replaced by accumulate-and-close, §3.3).
- Uncorrected wrong-angle values surviving at in-medium events (replaced at VBS by
  `ray_state_update`; `ray_mesh_interact` still computes and returns them honestly, including TIR,
  preserving its standalone contract — the replacement overwrites them in place).

**Unchanged invariants:** interaction-type codes, `out_typeN` codes, the three state words and
flag semantics, the full/compact index contract, the loss-ownership *principle* (each in-medium
segment charged exactly once, now over the accumulated total), resolved-precedence logic, the
stacked-slab persist rule, tf ownership (modulo the M2M combination rule), `interface_gain`
single-count, the validation-before-parallel rule, type-2 physics, entry-event (o-i) coefficients,
and the state-machine topology.

---

## 12. Test plan

Tests of state logic, dispatch topology, flag semantics, index mapping, batch determinism,
float/double parity, and validation survive unchanged. Coefficient oracles at in-medium events are
re-derived; the old hack oracles retire.

1. **`path_dirN` contract.** `origN`→`destN` for types 0/2/3 and reflections; Snell for type-1/4
   transmissions; `origN`→`destN` under TIR (TF and non-TF); unit norm; TF-modified TIR carries
   tf-assigned power forward with non-zero coefficients satisfying the gain identity.
2. **VBS construction.** Against analytic plane intersections — real face at a crossing, virtual
   plane through `dest` at `nH == 0`; FBS/SBS selection by out-type on colocated i-i hits;
   grazing/negative-$s$ guards.
3. **Snell-pair reciprocity and TIR impossibility at exits.** Via real-angle `interact_with` at
   $\theta_t$.
4. **Transfer-matrix ledger, oblique.** Both ports assembled across the two passes vs. the exact
   transfer matrix on a phase × angle grid, lossless and lossy; stacked slabs $R + T \le 1$.
5. **Airy walk-off phase.** $S(\theta_t, L)$ with the §3.4 fix vs. the closed-form Airy sum and the
   direct bounce-ray sum; fringe positions at 30°/60°.
6. **Insertion phase.** Phase exact at the call frequency vs. the transfer-matrix insertion phase;
   slope across a stored frequency grid; no magnitude double-count against the in-medium charges.
7. **Segmentation invariance (P9, the anchor test).** A straight path through an arbitrary scene,
   processed as (a) one SBR traversal and (b) the collinear diffraction spine cut into 10×10 m
   segments with boundaries that land inside bodies. Outputs **bit-identical** for any scene, with
   both sides in the forced-resolve configuration (§9): mass-law materials, the 1.5 mm clamp
   straddled, multi-layer i-i stacks, and a reflection round-trip. Then sweep boundary positions.
8. **Combined M2M tf rule.** $R_\mathrm{eff}$ of §10 deterministic under randomized FBS/SBS order,
   identical to current behavior at all air boundaries, port-complementary; mixed-sign tf in
   $[0,1]$; $\mathrm{tf} = -1 \Rightarrow R_\mathrm{eff} = 1$.
9. **Acoustic rigid wall ($\varepsilon \ll 1$).** Genuine front-face reflection on both passes;
   TF-modified TIR forward path with mass-law interior; bit-compatibility of the end-to-end wall
   transmission with the existing calibration (exit at the geometric angle).
10. **Pruning-order hazard.** Oblique type-1 transmission: correct nonzero end-to-end gain when
    pruning after `ray_state_update`; path loss when pruning early (documentation test).
11. **`ray_indN` map.** Ordered surviving indices, 0-based, round-trip into `ray_state_update`,
    drop rules, identity case.
12. **Stage-1 regression.** Refactored `ray_mesh_interact` bit-exact against the existing Catch2
    suite; benchmark suite without performance regression.

---

## 13. Open questions for review

- **Q1 — Combined tf rule at M2M interfaces.** Confirm the §10 $R_\mathrm{eff}$ closes the ledger:
  deterministic under FBS/SBS order, identical at all air boundaries, port-complementary across
  both passes and the Airy mirrors, for mixed-sign tf.
- **Q2 — `acc_dist` ownership at terminal-inside rays.** A ray that ends inside a body (no closing
  crossing) must charge its accumulated in-layer loss at its terminal event. Confirm the dispatch
  identifies that event for every topology (single-hit dest-inside, multi-hit, edge) so no
  in-medium leg is silently dropped.

Resolved during review: tf semantics unchanged, no migration (§10); `orig_correct` and
`excess_delayN` removed in favor of the VBS-plane-normal mechanism and the `xprmatN` phase fold;
two-stage implementation with EM polarization exact from stage 2 via stage-1-proven helpers (§5);
`ray_state_update`'s appended optional outputs are non-breaking.
