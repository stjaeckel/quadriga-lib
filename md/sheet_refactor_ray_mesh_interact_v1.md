# Thin-Sheet Handling in `ray_mesh_interact` — Specification and Status

**Status:** stages 1–3 implemented and passing. Stage 4 blocked on an unrelated defect (§4).
**Scope:** `quadriga_lib::ray_mesh_interact` and the shared slab resolve.
**Source:** `ray_mesh_interact.cpp` (contains both `ray_mesh_interact` and `ray_state_update`).

---

## 1. Problem and contract

Customer scenes are not uniformly watertight. Fences, glass panes and signs are exported as a single
sheet of triangles with a semi-transparent texture, not as closed solids.

The engine derives the medium from face orientation, so entering and exiting are distinct events and
the geometric path between them sets the attenuation. A zero-thickness sheet breaks that: a single
sheet produces an entry with no matching exit and corrupts the medium state for the rest of the path;
a double-sided sheet (two coincident back-to-back faces) classifies as a media-to-media transition
with identical materials, hits the `same_as` shortcut and transmits at gain exactly 1.0.

Geometry repair was rejected — an invented extrusion thickness directly scales the computed
attenuation, and for a chain-link fence the sheet abstraction is closer to reality than any slab.
Instead thickness is an explicit material parameter, following ITU-R P.2040 and Sionna RT.
quadriga-lib keeps a hybrid model: solids get thickness from geometry, sheets from the material table.

| `thk` | Behavior |
| --- | --- |
| `0` or column absent | Unchanged. Thickness comes from geometry via the state machine. |
| `> 0` | The face is a **sheet**. Each crossing is one atomic entry + traversal + exit, with the internal series resolved in closed form. |

A solid built from a sheet material is legal, not an error: it becomes a hollow box with sheet loss on
entry, the embedding medium inside, and sheet loss on exit. That is the standard building-entry-loss
topology and it is cheaper than a traced solid.

**Hard constraint.** With no `thk` column, or all `thk` zero, `ray_mesh_interact` is bit-identical to
its pre-refactor behavior. Every sheet change is a branch guarded by the sheet flag. Held so far, with
one deliberate exception recorded in §3.

---

## 2. Case reference

The classification rules and the `out_typeN` bit encoding now live in the source: the decision tree in
`ray_mesh_interact` around the `sheet_case` assignment, and the user-facing tables in the
`ray_mesh_interact` doc comment. **Those are authoritative.** This table is a reading aid only.

| # | Name | Condition | Handling |
| --- | --- | --- | --- |
| 1 | Isolated sheet | `sheetFBS && !colocated` | `near = far = air`. Undeviated, TIR forced off. |
| 2 | Glued entry | one sheet, co-located, parallel, solid face front-hit | `air | sheet | solid` |
| 3 | Glued exit | as case 2, solid face back-hit | `solid | sheet | air` |
| 4 | Ordered stack | both sheets, co-located, different materials | **Not implemented.** §5.3 |
| 5 | Collapsed duplicate | both sheets, co-located, same material | Collapse to one crossing; identical to case 1. |
| 6 | Corner with sheet | co-located, not parallel, either face a sheet | Drop SBS, treat FBS alone. Degrades to codes 65/67. |

Three invariants that make the rules provable rather than conventional, and that are commented at
their use sites because the `ray_state_update` phase will be written without this context:

- **Co-location implies sheet-on-the-incidence-side.** A sheet glued to a solid's far boundary would
  be co-located with *that* face, not this one, unless the solid is thinner than `colocation_dist`.
  So traversal order follows from co-location alone, and either normal parity is accepted.
- **A sheet's own normal carries no information** — a sheet has no inside. The front/back verdict for
  cases 2/3 comes from the solid face, which may be FBS or SBS.
- **Bit 1 changes meaning by context**, and is written here rather than re-derived downstream so the
  ordering rule lives in one place. Bit 6 is a hint meaning *at least one face is a sheet*;
  `ray_state_update` re-derives the full case from `(code, is_sheet(FBS), is_sheet(SBS), same_as)`.

---

## 3. What is implemented

### Stage 1 — shared slab core

`SLAB_AIRY_FACTOR`'s body moved to a file-scope `qd_slab_resolve`, callable from both public
functions. `SLAB_AIRY_FACTOR` survives as a thin adapter, so both call sites are unchanged and the
`qd_polbasis` call and buffer write stay at the call site — the two callers write to different buffers.

- `produce == false` returns per-channel correction factors; `produce == true` returns absolute
  transmit and reflect composites. Produce mode has no bail-outs: the weak-slab gate is dropped, the
  near-pole denominator is clamped instead of refused, and mass-law slabs take an incoherent route.
- **Direction convention:** `mat_near` is the medium the incident wave arrives from. Correction mode
  is symmetric in near/far, which is why `ray_state_update` may pass them in its own "interface ahead /
  interface entered through" order. Produce mode is not symmetric.
- `medium_gain` gained a defaulted `mass_min_path` parameter. The 1.5 mm floor guards traced solids
  against meshing artifacts; a sheet is a real traversal at any thickness and passes `0.0`.
- **One deliberate numerical change.** `fresnel_r` now forces the decaying evanescent root past the
  internal critical angle, as `interact_with` already did. Magnitudes are unaffected (`|r| = 1` on
  either branch); only the internal reflection phase moves, and only for cases that actually reach
  internal TIR.

### Stage 2 — classification

Six-case decision tree, `sheet_case`, `front_eff`, and the `out_typeN` bits. Doc comment gained the bit
table, the bit-1 context table, the sheet composite table, the `thicknessN` vs `thk` disambiguation and
the modeling notes (duplicate collapse, stack-depth cap, hollow-box corollary).

**Deviation from the original plan:** case 6 clears bit 2 as well as bit 4. Clearing only bit 4 leaves
codes 69/71, which `ray_state_update` reads as a media-to-media transition against the SBS we just
decided to drop. Clearing both yields 65/67 — an ordinary single hit, which is what "drop SBS" means —
and closes the composite table with no new codes. The relaunch still passes SBS, since that is driven
by the local `colocated_faces` flag rather than by the bit.

### Stage 3 — physics for cases 1, 2, 3, 5

- `M1`/`M2` become the near and far media with the sheet in `M_slab`. `Material::interact_with` is a
  two-medium interface function; a sheet is three media, two interfaces and one slab.
- **Interface evaluation split by purpose.** Geometry (`eta`, `cos_theta2`, TIR) from the near/far
  pair; coefficients from `qd_slab_resolve`. Snell is transitive, so the net bend depends only on the
  outer media and the sheet cannot affect the outgoing direction. Reading TIR from
  `interact_with(air, sheet)` would invent total reflection past the air|sheet critical angle, and in
  scalar mode the impedance mapping readily gives `n < 1` — an acoustic sheet would become a mirror.
- Cases 1/5 force TIR off (`near == far`, so it is reachable only through rounding at tangency), and
  are excluded from the whole-tube TIR escalation for the same reason. No direction special case is
  needed: `eta == 1` makes the refraction branch reproduce the incoming direction.
- Traversal distances map thickness onto the `medium_gain` signature, which is what lets the acoustic
  mass-law term evaluate at all. Grazing clamp cuts off where the lateral walk-off `thk·tanθ` exceeds
  the face extent; clamping the cosine from below caps the traversal at `sqrt(thk² + extent²)` with no
  trig call. `face_extent` is the longest FBS triangle edge, computed only when a sheet is present.
- The tube-TIR re-evaluation discards the slab composite for glued cases and falls back to the
  forced-TIR near/far result. Bounded: it can only fire for cases 2/3, and only when a tube edge is
  past critical while the spine is not.

**Two deviations from the original plan.** The normal flip stays on `thetaF` rather than moving to
`front_eff`: the flip orients `N` toward the incidence side so `qd_reflect`/`qd_refract` bend the right
way, which is a property of the ray and the FBS face alone. A glued sheet with a reversed normal gives
`thetaF < 0` with `front_eff = true`; taking the flip from `front_eff` would refract the wrong way.
`front_eff` decides which material is near and far, never which way a normal points. Second,
`edge_len` was not usable as the face extent — it lives in the no-hit branch and is a ray-tube edge
length, not a face extent.

### Not implemented

- **Case 4 physics.** Detection and bits are correct, but the case falls through to the legacy
  `M1`/`M2` path, so bit 6 is set while no sheet physics runs. See §5.3.
- The §6 `ISSUE` comments (relaunch offset, stack-depth cap).
- Everything in `ray_state_update`, including the embedded-sheet case.

---

## 4. New problem: the entry boundary is evaluated twice, inconsistently

**This is pre-existing, affects traced solids with no sheet involved, and blocks the sheet work.**

A traced ray must have a real direction. In a lossy medium the Snell refraction cosine is complex —
the wave is inhomogeneous, its constant-phase and constant-amplitude planes are not parallel — so
`qd_refract` takes `Re(cos_theta2)` and discards the rest. That approximation is inherent to ray
tracing and is not the defect.

The defect is that one slab crossing evaluates its boundaries in three places that do not agree:

| Where | Evaluation | Status |
| --- | --- | --- |
| Entry, `ray_mesh_interact` | `interact_with(air, slab, θ_i)` — real θ_i, complex `ct2` kept | **exact** |
| Exit, `REPLACE_BY_VBS` | `interact_with(slab, air, θ_t)` — real traced θ_t | approximate |
| Airy denominator, `fresnel_r` | rebuilds `cos_t2` from that same real θ_t | approximate, and it re-derives the **entry** boundary |

`1/(1 − r_sn·r_sf·φ²)` sums the internal series correctly only if `r_sn` is the coefficient the entry
actually applied. It is not. Reciprocity `|r₁₂(θ_i)| = |r₂₁(θ_t)|` holds only when `θ_t` is the true
complex angle; fed the real projection, the two views of one interface disagree.

Nothing downstream repairs this. `ray_state_update`'s VBS correction recovers the exit incidence angle
from the true propagation direction — a real and worthwhile geometric fix — but it computes
`|dir_prev · N|`, a dot product of two real unit vectors. The imaginary part was discarded at entry and
is never reconstructed.

### The obstacle for the fix

**Correction mode does not hold the entry angle.** At an exit event it has the real in-slab angle and
nothing else; the incidence angle in the medium outside the slab was consumed one call earlier. Three
candidates, to be decided in the dedicated fix (§5.1):

1. **Invert Snell from the traced direction.** `sin θ_prev = n_slab·sin θ_slab / n_prev` with real
   indices. Cheapest, and plausible because `qd_refract` bent the ray with the same projection — but
   `qd_refract` normalizes a vector built from `Re(cos_theta2)`, and that normalization is not
   angle-preserving when `cos_theta2` is complex. **The round-trip error is unmeasured and must be
   checked numerically before this is recommended.**
2. **Carry the entry incidence cosine as state.** Exact. `acc_dist_geometric`/`acc_dist_refracted`
   already establish the pattern. Costs an optional column and a public signature change.
3. **Derive both boundaries from one consistent complex in-slab cosine.** Self-consistent and cheap,
   but does not reproduce the coefficient `ray_mesh_interact` applied at entry — it removes the
   inconsistency without making the composition correct.

All three move `ray_state_update` output for lossy solids at oblique incidence by percent-level
amounts. **Record baselines for a set of lossy-oblique cases before starting**, and settle the
tolerance policy, so the movement is characterized as expected rather than discovered as a failure.

---

## 5. Way forward

### 5.1 Fix the entry boundary first — separate work item, own document

Scoped to the slab resolve's near-side evaluation, no sheet content. Must be self-contained enough for
a session with no memory of this discussion: the defect, why nothing downstream repairs it, a
reproduction with material parameters and angles, the three candidates with the open numeric question
on the Snell inversion, and the expected regression movement.

### 5.2 Then revisit produce-mode angle handling

Most of stage 3 is orthogonal and stands unchanged. What has to be reconsidered is the interior of
`qd_slab_resolve`'s produce branch, roughly 60 lines:

- **A sheet can be exact.** Nothing propagates geometrically inside it, so no real direction has to be
  stored, and `ray_mesh_interact` holds the real incidence angle in the outer medium — exactly the
  input from which the complex `cos θ_t` follows by Snell. Carry it through Fresnel, the propagation
  factor and the series without projecting it. Stage 3 currently goes half way: `fresnel_r_near`
  evaluates the near boundary from the incidence side, which cut the error by roughly an order of
  magnitude, but the denominator still uses the slab-side value.
- **The exact treatment reaches the attenuation model, not just Fresnel.** `|phi|` currently comes
  from `medium_gain(thk/cos θ_t, …)`, a real path length times a per-meter loss. The exact form is
  `exp(−j·k₀·n·thk / cos θ_t)` with both `n` and `cos θ_t` complex — magnitude and phase from one
  expression. Exact angles in Fresnel with a real path length in `medium_gain` is a hybrid that may
  not beat either. Produce mode only; correction mode's `|phi|` must not move.
- **The grazing clamp needs reformulating** if the in-slab angle goes complex — it is currently
  defined through a real `tan θ`.
- **The `cos²` walk-off convention.** In `SCALE_BY_MEDIUM` it substitutes for a lateral shift the
  undeviated tracer does not perform. A sheet is never displaced at all, so it is not obvious the
  factor belongs in the sheet path. Resolve rather than assume.

### 5.3 Then stage 4 — case 4

Detection is done. Physics is a documented approximation: slab-resolve layer 1 with
`near = air, slab = layer 1, far = layer 2`, then pass layer 2 through with its interface and medium
gain only — no second Airy series, no cross-layer coupling; direction and TIR from `near = far = air`.
Wrong in the coupling term but bounded, monotone, and built from existing blocks. It calls produce
mode, so it waits on §5.2. The §6 `ISSUE` comments are pure documentation and can land earlier.

### 5.4 Then the `ray_state_update` phase

The embedded-sheet case (`near = far =` current medium) calls the same produce-mode function, so
produce-mode angle handling must be settled before this phase starts, not concurrently.

---

## 6. Known issues to record in code, not fix

**Relaunch offset vs. co-location distance.** `relaunch_offset = 8.0 * scale * 1.1920929e-7` is ~8
float ULP at the FBS coordinate magnitude — at scene coordinates around 1000 m that is ~1e-3 m,
exactly `colocation_dist`; at 8000 m, ~7.8e-3 m. A second face 1.5 mm behind the first is not
co-located, so FBS is handled alone and the relaunch overshoots it, skipping the face. Coordinate
dependent: passes every test near the origin, fails on a real city scene. Clamping to `0.5·FS_length`
would defeat the ULP scaling that exists so the nudge survives the `dtype(origN)` store. Pre-existing,
not sheet-specific — it affects thin solids identically — and should be fixed globally.

**Stack depth capped at 2.** The FBS/SBS intersector returns two faces. A three-layer stack is silently
truncated: layers 1–2 together, layer 3 as a separate FBS after the relaunch, concatenating
incoherently. `solid | sheet | solid` is already three faces. Cannot be fixed here; documented as a
modeling constraint in the doc comment.

**Averaged reflectance in `interact_with`.** Pre-existing. The companion correction in
`qd_slab_resolve` must stay coupled to it; fixing one without the other double-corrects.

---

## 7. Acceptance criteria

### 7.1 Primary equivalence test

> A sheet must behave like a solid of the same thickness — **exactly where the solid is exact, and
> predictably elsewhere.**

Model the same 6 mm pane twice: as a closed manifold box with `thk = 0`, and as a single quad with
`thk = 0.006`. Compare gain, phase and polarization. This one fixture exercises the slab extraction,
the reflection composite, the traversal-distance mapping and the near/far direction rule at once,
against machinery that is already trusted.

Design around three known deviations:

1. **Lateral offset.** A traced solid refracts in and out, displacing the exit point by
   `thk·tan(θ_slab)`; the sheet does not. Run the primary comparison in `geometry_type 1`, where the
   solid does not trace the bend either.
2. **Refraction mode.** In `geometry_type 2` compare at normal incidence for exact agreement; at
   oblique incidence compare exit *direction* and gain, not exit position.
3. **Interface gain applied once.** In the solid path `interact_with` folds `other.interface_gain()`
   in on entry but not on exit (`other` = air, `att` = 0). The sheet path must match.

### 7.2 The sheet/solid discrepancy is a designed property, not a residual

Even after §5.1 and §5.2, a traced solid still rests on a real ray direction at its exit interface and
in its path length, while an exact sheet does not. **Two independent sources, and separating them is
what makes the discrepancy explainable:**

- **In-slab obliquity of the traced solid.** Vanishes at normal incidence and as
  `Im(cos θ_slab)/Re(cos θ_slab) → 0`. Loss alone does not predict it — in-slab obliquity does. High
  permittivity bends the ray toward the normal and shrinks the imaginary part's share.
- **A lossy near medium, which affects the sheet too.** A sheet is exact only when its incidence-side
  angle is genuinely real, i.e. when `near` is air. Case 3 (`near` = solid) and the embedded-sheet case
  both receive an angle that is already the tracer's projection, and are no better than the solid.
  Case 3 does **not** inherit case 1's exactness.

A fixed tolerance is the wrong instrument — loose enough to pass the worst case, it asserts nothing.
Assert the predictor instead: machine precision for a lossless slab at every angle; deviation monotone
in `Im(cos θ_slab)/Re(cos θ_slab)`; a documented bound at a named worst case.

Measured after stage 3, scalar mode, 6 mm slab, transmitted power, relative deviation from the traced
composition. Note that εr = 2 with weak loss is far worse than εr = 6 with strong loss — obliquity, not
loss:

| Slab | 0° | 34° | 68° | 85° |
| --- | --- | --- | --- | --- |
| lossless, εr 6 | 2e-16 | 9e-16 | 1e-15 | 1e-13 |
| εr 6, σ = 0.02 | 2e-16 | 8e-4 | 8.9e-3 | 1.8e-2 |
| εr 6, σ = 0.2 | 5e-16 | 5.3e-3 | 3.1e-2 | 3.0e-2 |
| εr 2, σ = 0.05 | 1e-16 | 3.4e-3 | 9.8e-2 | 2.3e-1 |

**Promote the harness to a real fixture.** Sheet versus solid at coefficient level, against the
composition the tracer actually performs, is the highest-value test in this effort and currently exists
only as a scratch program.

### 7.3 Regression

- Every existing test with no `thk` column: bit-identical, except cases reaching internal TIR, which
  move in phase only (§3, stage 1).
- `qd_slab_resolve` in correction mode: bit-identical to the old `SLAB_AIRY_FACTOR` outside internal
  TIR.

### 7.4 Fixtures

| Fixture | Covers |
| --- | --- |
| Single quad, `thk > 0`, front and back illumination | Case 1; orientation independence |
| Quad + coincident solid face, both directions | Cases 2 and 3; front/back from the solid face |
| Two coincident quads, reversed normals, identical material | Case 5; must equal a single quad |
| Two coincident quads, different materials, both parities | Case 4 ordering and the ambiguous fallback |
| Quad meeting a solid at an angle | Case 6 degradation to codes 65/67 |
| Sheet at increasing incidence up to near-tangency | Grazing clamp; no whole-tube TIR escalation |
| Acoustic sheet with `m > 0` | Incoherent mass-law route; `mass_min_path` behavior |

---

## 8. Open issues carried forward

1. **Entry-boundary inconsistency** (§4). Blocks everything else.
2. **Exact complex-angle produce mode** (§5.2), including the propagation factor, the grazing clamp
   and the walk-off convention.
3. **True two-layer coherent resolve (case 4).** Two stacked sheets are four media and the internal
   reflections couple across both layers; composing two single-slab Airy results is not equivalent.
   The eventual answer is a **transfer-matrix core** rather than a two-layer special case: exact for
   any layer count, reduces to the Airy formula at N = 1 (a built-in regression against trusted
   results), produces R and T in one pass, and handles the asymmetric glued case for free. It is also
   the natural moment to fix the averaged-reflectance issue, since the companion correction
   disappears. Roughly a 2×2 complex matrix product per layer per polarization.
4. **Sheet embedded inside a solid.** `near = far =` current medium. Needs `ray_state_update`.
5. **Relaunch offset** (§6).
6. **`subdivision_tolerance_m` against resonant slab reflectance.** A 6 mm pane at 3.5 GHz with
   εr ≈ 6 has a round-trip optical path of ~29 mm against λ = 86 mm, so R shows real angular and
   frequency structure instead of sitting near the smooth half-space Fresnel value. Correct, and it
   matches P.2040/Sionna, but the tolerance was tuned against a smooth angular response.
7. **Preprocessing validator.** Warn on closed manifold components built from `thk > 0` materials, and
   on open patches whose material has `thk == 0`. Uses the existing union-find component analysis,
   split at edges with valence ≠ 2 so a pane welded to its frame separates cleanly.

---

## 9. Code conventions

- **Comment intent, not mechanics.** The `ray_state_update` follow-up will be written in a fresh
  session with no memory of this discussion. Every non-obvious branch needs a *why*.
- Plain `//` comments only. No framed or boxed comments, no dashes or `====` separators.
- American English throughout.
- Uppercase names for local lambdas, `qd_` prefix for file-scope helpers.
