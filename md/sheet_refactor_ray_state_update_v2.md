# Thin-Sheet Handling in `ray_state_update` — Specification

**Status:** not started. `ray_mesh_interact` stages 1–3 are in; every sheet code is killed on arrival here.
**Scope:** `quadriga_lib::ray_state_update`, the sheet dispatch, the `resolved_typeN` encoding, and the
documentation that describes both.
**Source:** `ray_mesh_interact.cpp` (contains both public functions).
**Companion:** `sheet_refactor_ray_mesh_interact_v1.md` — the classification, the bit encoding and the
slab physics live there and are authoritative. This document does not restate them.

**Out of scope, deliberately:** case 4 (ordered stack), the entry-boundary defect (companion §4 and
§5.1), and the produce-mode complex-angle rework (companion §5.2). Case 4 gets an explicit, commented
stub so it reads as unimplemented rather than broken.

**Hard constraints, from which everything below follows:**

1. `resolved_typeN` stays `uint8_t`. The downstream storage engine holds one byte per path per
   interaction across 100M+ paths; a second byte is not affordable.
2. No existing test may change. The whole sheet path is gated behind `typeH[6]`, which cannot be set
   without a nonzero `thk` column, so this holds by construction — but the `resolved_typeN` mapping must
   also avoid every currently reachable composite. §6 shows it does.
3. No sheet case is killed. `concrete | sheet | concrete` is as valid as `glass | air | glass`, and both
   must resolve.

---

## 1. Why every sheet dies today

`ray_state_update` dispatches on `typeH` by exact equality against 1, 3, 5, 7, 13, 15, 21, 23, 29, 31.
Bit 5 (total reflection) is stripped at the top of the loop; **bit 6 (thin sheet) is not**. Every sheet
code — 65, 67, 69, 71, 79 — misses all ten comparisons and falls through to `KILL_RAY()`, which zeroes
`gainN` and `xprmatN`.

In `test_sheet_vs_solid` the sheet returns exactly zero at every angle, including 0°, where the physics
is trivial: the relative deviation is precisely 1.000e+00 in all six rows. An angle-independent,
exactly-zero result is a dispatch failure, not a physics error.

Two observations before any code changes:

- **The reflection pass already half-works.** For `typeH == 67` with `mtl_current == 0`,
  `transparent_forward` evaluates false and the `mtl_current == 0` arm calls `DIST_SET()` and leaves the
  coefficients alone. Correct for a sheet reflection in air. Only the transmission pass kills.
- **Confirm the classification first.** One dump of `out_typeN` decides the plan. Expect 67. If it comes
  back 3, `thk` is not reaching `MaterialCols` through `calc_diffraction_gain`, the quad is being traced
  as a solid face, the ray enters concrete and never leaves, and the gain underflows to zero. Same
  symptom, unrelated fix.

---

## 2. The core correction: `ray_mesh_interact` cannot know the bounding media

`ray_mesh_interact` derives the sheet's near and far media from **face geometry alone**. It has no
medium state, so it assumes air wherever the mesh does not tell it otherwise:

| Case | What `ray_mesh_interact` assumes | What is actually true |
| --- | --- | --- |
| 1 / 5 / 6 isolated | `air \| sheet \| air` | `host \| sheet \| host` |
| 2 glued entry | `air \| sheet \| solid` | `host \| sheet \| solid` |
| 3 glued exit | `solid \| sheet \| air` | `solid \| sheet \| enclosing` |

That assumption is right only when the ray is in air. It is wrong for every embedded sheet, and it is
wrong in three separate places at once: the two Fresnel boundaries, the Airy denominator that couples
them, and the lumped `interface_gain` that produce mode folds into the transmit port.

**`ray_state_update` holds the missing information in its state words and must redo the resolve.** This
is precisely why `qd_slab_resolve` and `medium_gain` were lifted out of the lambdas into file scope: so
both public functions can call them. The sheet path is the first caller to use that.

### 2.1 Near and far from the state words

| Case | `near` | `far` |
| --- | --- | --- |
| 1 / 5 / 6 (isolated, duplicate, corner) | `MAT(mtl_current)` | `MAT(mtl_current)` |
| 2 (glued entry) | `MAT(mtl_current)` | the solid face: `is_sheet(M1) ? M2 : M1` |
| 3 (glued exit) | `MAT(mtl_current)` — the solid | `MAT(mtl_prev)` — the enclosing medium |

`mtl_prev` is exactly right for the glued exit: it is the medium the ray entered `mtl_current` through,
which for a cavity exit is the medium it returns to. It is one level deep, so `sheet | concrete | sheet`
resolves correctly but a deeper nesting loses the outer medium and falls back to air. Record it; a
deeper history is not worth a state word.

### 2.2 Always redo, do not branch on "did the assumption hold"

Redo the sheet resolve unconditionally whenever medium tracking is enabled, rather than testing whether
`ray_mesh_interact`'s air assumption happened to be correct. Three reasons:

- **No assumption to track.** A "keep if entered from air" branch has to encode what the other function
  assumed, in three cases, and stay in sync with it.
- **The VBS angle comes for free.** `ray_mesh_interact` evaluated at `thetaF`; `ray_state_update` holds
  `theta_t`, the incidence angle from the true propagation direction. Redoing gives the sheet the same
  geometric correction that `REPLACE_BY_VBS` gives solids, unconditionally. That is also why bit 1 is
  available for re-qualification in §6 — a flag that is always true carries no information.
- **Cost is negligible.** One produce-mode `qd_slab_resolve` per sheet interaction, against a tracer that
  already runs an intersection test per ray.

**Gate:** when all six state arguments are NULL, tracking is disabled and there is no medium state to
redo from. Keep `ray_mesh_interact`'s air-based result in that configuration, which is the documented
behavior of the NULL-state mode.

### 2.3 Defect found while specifying this: the `interface_gain` gate

`qd_slab_resolve` produce mode applies

```
ig = sqrt(mat_slab.interface_gain * (far_is_air ? 1.0 : mat_far.interface_gain))
```

The intent, per its own comment, is to match what `interact_with` folds in for a transmission-class
interaction: the *newly entered* medium pays its lumped interface gain. `far_is_air` is a proxy for
"nothing new is entered", and the proxy fails for an embedded sheet: with `near = far = concrete` the
ray does not enter concrete, it was already in it, but the current test applies concrete's
`interface_gain` anyway.

**Fix:** gate on whether the far medium is newly entered, not on whether it is air —
`far_is_air || mat_far.same_as(mat_near)`. This changes nothing for any case reachable today
(`air|sheet|air` and `air|sheet|solid` are unaffected; `solid|sheet|air` gives air's `interface_gain`
of 1.0 either way), so it cannot move an existing test. It only takes effect for the embedded and
enclosed cases this phase introduces.

---

## 3. A sheet crossing is a three-layer stack, and the state machine already models one

Your framing, made explicit, because it is the thing that makes every case fall out of one pattern.

A ray inside concrete meeting an embedded sheet is `concrete | sheet | concrete`. The event decomposes
into three steps that the existing machinery already performs:

1. **Close out the concrete leg.** The internal series in the concrete reflects off the interface ahead
   — which is now the sheet, not air — and off the interface it entered through:
   `SLAB_AIRY_FACTOR(mtl_slab = concrete, mtl_near = sheet, mtl_far = mtl_prev)`. The existing argument
   order takes this without modification; only the value passed as `mtl_near` changes. Then
   `SCALE_BY_MEDIUM(concrete)` for the traversal loss and excess phase.
2. **Resolve the sheet.** Produce mode, `near = concrete`, `slab = sheet`, `far = concrete`, replacing
   the coefficients `ray_mesh_interact` wrote.
3. **Start a fresh leg.** `mtl_prev = sheet`, `mtl_current = concrete`, `DIST_SET()`. The next concrete
   leg's series will then reflect off the sheet at its far side, which is correct.

**This is the general form.** Every sheet case is: *close out the host leg if inside → resolve the sheet
with state-derived near/far → start a fresh leg with the sheet recorded as the previous medium.* The
glued cases differ only in what `mtl_current` becomes in step 3.

Two consequences worth stating in the code, because neither is recoverable from reading it:

- **A sheet index in `mtl_prev` is safe.** `mtl_prev` is read in exactly three places: as `mat_far` in
  the two `SLAB_AIRY_FACTOR` calls, and for the `prev_nonpar` flag in bit 15. Nothing compares it to
  `mtl_current` or dereferences it as a traversed medium. Preserve bit 15 when writing.
- **A sheet as `mat_near` / `mat_far` in correction mode is treated as a half-space**, not as a slab.
  That is an approximation to the true three-layer coupling, in the same family as everything else the
  correction mode does, and the exact answer is the transfer-matrix core in companion §8.3. It is
  strictly better than passing air.

---

## 4. Case dispatch

Bit 6 is a hint. The authoritative case comes from `(typeH, is_sheet(FBS), is_sheet(SBS), SAME(M1,M2))`,
exactly as the `ray_mesh_interact` doc comment specifies. Everything needed is in scope: `M1` / `M2`
from `mtl_ind_fbsN` / `mtl_ind_sbsN`, `cols.is_sheet()` from the `thk` column, the existing `SAME`.

| `typeH` | `is_sheet(FBS)` | `is_sheet(SBS)` | same material | Case | Handler |
| :-----: | :-------------: | :-------------: | :-----------: | :--: | ------- |
| 65 / 67 | yes             | —               | —             | 1 or 6 | §5.1 |
| 69 / 71 | yes             | yes             | yes           | 5    | §5.1 |
| 69 / 71 | exactly one     | exactly one     | —             | 2 / 3 | §5.2 / §5.3 |
| 69 / 71 | yes             | yes             | no            | 4    | §5.4 (stub) |
| 79      | yes             | yes             | no            | 4    | §5.4 (stub) |

- **Cases 1, 5 and 6 collapse to one handler.** Case 6 clears bits 2 and 4 precisely so it arrives as an
  ordinary single hit; case 5 is a collapsed duplicate, identical by construction to a single sheet.
- **Codes 69 and 71 are three-way ambiguous by design.** The re-derivation separates them; bit 1 then
  distinguishes glued entry (71, front-side hit on the *solid*) from glued exit (69).
- **`mtl_current` is no longer a second dispatch axis.** In v1 of this document it was, because the
  embedded case was going to be killed. With §2.1 it is just an input to the near/far selection.

**Structural placement.** Add `if (typeH[6]) { … } else { existing chain }` ahead of the current
dispatch. Do **not** strip bit 6 and reuse the base code: 67 strips to 3, which the machine reads as
"outside → inside entry", and would put an isolated sheet into a medium.

---

## 5. Per-case handling

Common to all: no `REPLACE_BY_VBS` on the sheet itself (that would re-apply an interface coefficient
through `interact_with`, on top of the composite), and no `SCALE_BY_MEDIUM` on the sheet material (the
traversal loss and phase are inside the composite). `SCALE_BY_MEDIUM` on the *host* is a different
matter and is required wherever a leg closes.

### 5.1 Isolated sheet, collapsed duplicate, corner with sheet (65 / 67, and 69 / 71 with two same-material sheets)

`host | sheet | host`. The medium does not change; the leg does.

| | `mtl_current == 0` | `mtl_current != 0` (embedded) |
| --- | --- | --- |
| Host close-out | none | `SLAB_AIRY_FACTOR(mtl_current, sheet_mtl, mtl_prev)` then `SCALE_BY_MEDIUM(mtl_current)` |
| Sheet resolve | produce, `near = far = air` | produce, `near = far = MAT(mtl_current)` |
| Distance | `DIST_ADD()` | `DIST_SET()` (preload if `ray_ends`) |
| `next_current` | unchanged | `mtl_current`, resolved flag cleared |
| `next_prev` | unchanged | `sheet_mtl`, `prev_nonpar` preserved |
| `path_dirN` | keep — `near == far` gives `eta = 1`, so the written direction is already undeviated |||
| `typeR` | `ok`, `resolve`, `trans`; `inside` if `mtl_current != 0` |||

In air the accumulators are dropped at the next medium entry anyway, so `DIST_ADD` keeps the
pass-through convention and leaves `next_prev` alone — the o-i branch would overwrite it with air
regardless.

### 5.2 Glued entry — 71, exactly one sheet (`host | sheet | solid`)

| | |
| --- | --- |
| Host close-out | if `mtl_current != 0`: `SLAB_AIRY_FACTOR(mtl_current, sheet_mtl, mtl_prev)` + `SCALE_BY_MEDIUM(mtl_current)` |
| Sheet resolve | produce, `near = MAT(mtl_current)`, `far` = the solid face material |
| Distance | `DIST_SET()`; with `ray_ends`, preload via `VBS_INTERSECT(F, dir_next, D, N)` as the o-i branch does |
| `next_current` | the solid, resolved flag cleared |
| `next_prev` | `sheet_mtl` + the parallelism flag (below) |
| `path_dirN` | keep — the bend was computed from `near = air`, `far = solid`, and the net Snell direction depends only on the outer media |
| `typeR` | `ok`, `inside`, `resolve`, `trans`, `glued` |

**The wedge test is lost here, and this needs recording in code.** `fbs_sbs_not_parallel()` compares FBS
and SBS, which for a glued pair are the sheet and the solid's near face — co-located, so it returns
false. The question the flag answers is whether the solid's *far* face is parallel to its near face, and
that face is not in this hit. Default `prev_nonpar = false` (allow the resolve at the far side), since
the motivating geometry is a coating on a flat wall. Failure mode: a sheet glued to a wedge resolves as
a slab. The preprocessing validator (companion §8.7) is the right place to warn.

**TIR can co-occur** (codes 101 / 103). `ray_mesh_interact` already discarded the slab composite and
substituted the forced-TIR near/far result at the tube-TIR re-evaluation, so the redo must respect the
same verdict rather than re-deriving a transmitted port.

### 5.3 Glued exit — 69, exactly one sheet (`solid | sheet | enclosing`)

| | |
| --- | --- |
| Host close-out | `SLAB_AIRY_FACTOR(mtl_current, sheet_mtl, mtl_prev)` + `SCALE_BY_MEDIUM(mtl_current)` — always, the solid's leg ends here |
| Sheet resolve | produce, `near = MAT(mtl_current)`, `far = MAT(mtl_prev)` |
| Distance | `DIST_SET()` / preload, as `APPLY_TRANSITION(exit = true)` |
| `next_current` | `mtl_prev` (air in the ordinary case) |
| `next_prev` | `mtl_current` — the solid just left, matching the existing exit convention |
| `typeR` | `ok`, `resolve`, `trans`, `glued`; `inside` if `mtl_prev != 0` |

Replicate the `GET_EXIT_MATERIAL` consistency check by hand: the solid face material should equal
`mtl_current`; if it does not, set the `fix` bit rather than trusting either.

### 5.4 Ordered stack — 79, and 69 / 71 with two different-material sheets (case 4)

`ray_mesh_interact` sets bit 6 but runs no sheet physics — the case falls through to the legacy `M1`/`M2`
path. Keep killing it, but as an explicit, commented branch so a reader sees an unimplemented case
rather than a dispatch hole. Revisit with the transfer-matrix core.

### 5.5 Reflection pass

Same decomposition, with the reflect port of the composite instead of the transmit port and no medium
change.

| Case | `mtl_current` | Handling |
| --- | --- | --- |
| 1 / 5 / 6 | `0` | `DIST_SET()`; redo the sheet resolve with `near = far = air`; keep otherwise. Route explicitly so it cannot drift into the internal-back-reflection arm. |
| 1 / 5 / 6 | `!= 0` | `DIST_ADD()`, `SLAB_AIRY_FACTOR(mtl_current, sheet_mtl, mtl_prev)`, `SCALE_BY_MEDIUM`, `DIST_SET()`, sheet resolve with `near = far = MAT(mtl_current)`, set the resolved flag on `next_current` to terminate the bounce series. Suppress `REPLACE_BY_VBS`. |
| 2 (glued, host side) | any | As above with `far` = the solid; the composite's reflect port is `host | sheet | solid`. |
| 3 (glued, solid side) | the solid | As above with `far = MAT(mtl_prev)`. |
| 4 | any | Kill (§5.4). |

**This fixes a live bug.** Today an embedded sheet in the reflection pass routes into the
internal-back-reflection arm and calls `SLAB_AIRY_FACTOR` on a zero-thickness "slab". It does not fire
in the current test because the ray is in air, but it is reachable in any scene with an embedded sheet.

`transparent_forward` needs no change: with a sheet code it already evaluates false, which is correct —
a sheet reflection is a real event.

---

## 6. `resolved_typeN`: an 8-bit encoding for the sheet states

### 6.1 The free region

All eight bits are assigned, but not all *combinations* are reachable. Bit 6 (`trans` in transmission,
`scatter` in reflection) is set only by `REPLACE_BY_GAIN(1.0)`, which never sets bit 2 (`resolve`) — a
transparent interface by definition resolves nothing. On the reflection side bit 6 is documented as
reserved and unimplemented. So:

> **Bit 6 together with bit 2 is unreachable in both passes.**

Reachable transmission composites today: 1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 19, 21, 23, 27, 29, 31, 33,
35, 37, 39, 40, 41, 43, 45, 47, 49, 51, 53, 55, 59, 61, 63, 65, 72, 73, 88, 89. Reachable reflection
composites: 129, 137, 139, 141, 143 and their `fix` / `tir` variants. None carries both bit 6 and bit 2.

### 6.2 The mapping

Bit 6 becomes a **context flag qualified by bit 2**, which is exactly the pattern already used for bit 1
of `out_typeN`:

| bit 6 | bit 2 | Meaning |
| :---: | :---: | --- |
| 1 | 0 | transparent interface (transmission) / scatter (reflection) — **unchanged** |
| 1 | 1 | **thin-sheet crossing** |

Inside the sheet region, bit 1 is re-qualified. It cannot carry its `vbs` meaning there because a redone
sheet resolve always evaluates at `theta_t`, so VBS correctness is unconditional and the flag would be
constant. It carries the one property of the crossing that the other bits cannot express:

| bit 1 (sheet region) | Meaning |
| :---: | --- |
| 0 | **isolated** — the same medium bounds both sides |
| 1 | **glued** — the two bounding media differ |

Every other bit keeps its documented meaning. Two invariants make the region well-formed and should be
asserted: **a sheet event always sets bits 2 and 6**, and **bit 1 is the glued flag whenever both are
set**.

### 6.3 Composites

Transmission (add 16 for `fix`, 32 for `tir`):

| Dec | Hex | Flags | Meaning |
| :-: | :-: | --- | --- |
| 69 | 0x45 | ok, resolve, sheet | isolated sheet crossed in air |
| 77 | 0x4D | ok, inside, resolve, sheet | isolated sheet crossed inside a medium (embedded) |
| 71 | 0x47 | ok, glued, resolve, sheet | glued exit — `solid \| sheet \| enclosing`, continues outside |
| 79 | 0x4F | ok, glued, inside, resolve, sheet | glued entry — `host \| sheet \| solid`, continues inside |

Reflection:

| Dec | Hex | Flags | Meaning |
| :-: | :-: | --- | --- |
| 197 | 0xC5 | refl, ok, resolve, sheet | reflection off an isolated sheet, ray outside |
| 205 | 0xCD | refl, ok, inside, resolve, sheet | reflection off an isolated sheet, ray inside a medium |
| 199 | 0xC7 | refl, ok, glued, resolve, sheet | reflection off a glued sheet from the host side |
| 207 | 0xCF | refl, ok, glued, inside, resolve, sheet | reflection off a glued sheet from the solid side |

Note for the doc: **69 and 71 also exist as `out_typeN` values with different meanings.** The two fields
have always had separate tables; say so explicitly at both tables so nobody conflates them.

### 6.4 Consequences for bit 2's documented meaning

Bit 2 currently reads "an internal multi-bounce series was resolved analytically **by this function**".
On the sheet path the sheet's own series was summed in `ray_mesh_interact` (and re-summed here on a
redo), while the host's series is summed here. Broaden the wording to "the internal series is accounted
for — expect no further internal bounces from this interface", which is the useful downstream meaning
and is true in both cases.

---

## 7. New primitives

**`SHEET_CASE`** — the §4 re-derivation, returning the case, the sheet material index and the non-sheet
material index. Four call sites; derive it once. Comment it with the invariant from the companion doc:
co-location implies the sheet is on the incidence side, so traversal order follows from co-location
alone and either normal parity is accepted.

**`SHEET_RESOLVE`** — the produce-mode call plus the buffer write. Selects `near` / `far` per §2.1, calls
`qd_slab_resolve` with `thk / ct_slab`, `thk / ct_geo`, the grazing clamp and `produce = true`, then
`qd_polbasis` into `xprmatN` / `gainN`. It mirrors the block in `ray_mesh_interact` around the
`sheet_resolve` branch; the two should be reviewed together whenever either changes, and that
relationship needs saying in both comments.

**`CLOSE_HOST_LEG`** — `SLAB_AIRY_FACTOR(mtl_current, sheet_mtl, mtl_prev)` followed by
`SCALE_BY_MEDIUM(mtl_current)`, used by §5.1 embedded, §5.2 enclosed, §5.3 and the reflection pass. It is
`APPLY_TRANSITION` minus `REPLACE_BY_VBS` and minus the transition itself.

**No new inputs.** `mtl_ind_sbsN`, `mtl_prop` and `out_typeN` are already parameters, and `MaterialCols`
is already constructed at the top of the function and already resolves `thk`.

---

## 8. Interactions and edge cases

- **Grazing clamp and `face_extent`.** `ray_mesh_interact` computes `face_extent` from the FBS triangle
  and uses it for the clamp; `ray_state_update` does not have the mesh. Either pass the clamped cosine
  through (there is no free output slot), recompute the clamp from `thicknessN` (zero for a sheet, so
  useless), or accept `ext = thk` as `ray_mesh_interact` does when `face_extent` is unavailable. **This
  is the one genuine gap in the redo plan and needs a decision before implementation.** Simplest
  defensible choice: reuse the incidence cosine unclamped and clamp only against `cos_min` derived from
  `thk` alone, accepting a more conservative cutoff than `ray_mesh_interact`'s.
- **VBS.** For a sheet, FBS and VBS coincide — there is no thickness to displace through. `theta_t` is
  still meaningful and is what the redo should use. Never reach a VBS-consuming call on the sheet path.
- **Buffer.** Only code 79 sets bit 3, and only case 4 carries a same-direction flag. No in-scope sheet
  code touches the deferred-buffer machinery. Assert rather than assume: `mtl_buffer != 0` on a sheet
  code is a state-machine invariant violation and should raise the `fix` bit.
- **`nH` / `ray_ends`.** An isolated sheet gives `nH = 1` when it is the last hit in the segment. The
  preload path matters for every case that calls `DIST_SET`, which is now all of them except a sheet
  crossed in air.
- **`normal_vecN` offset.** The `normal_off = 3` selection is gated on `typeH == 5 || 23 || 29`, none of
  which a sheet produces, so the FBS normal is used. Correct — the sheet *is* the FBS face for cases 1,
  5 and 6, and for a glued pair the faces are co-located and parallel.
- **Relaunch offset versus co-location distance.** Pre-existing, recorded in companion §6, and it bites
  sheets exactly as it bites thin solids: at scene coordinates around 1000 m the nudge already equals
  `colocation_dist`. Not fixed here.

---

## 9. Documentation updates

### 9.1 `ray_state_update` doc comment

- **Intro bullets** — add a thin-sheet bullet: a `thk > 0` crossing is an atomic three-medium event, and
  this function supplies the bounding media that `ray_mesh_interact` could not know and re-resolves the
  composite from them.
- **`mtl_ind_sbsN`** — now load-bearing beyond media-to-media: it carries the second face of a glued or
  duplicate pair and is required for the case re-derivation.
- **`mtl_prop`** — this function now reads the `thk` column.
- **`out_typeN`** — bit 6 is dispatched on; bit 5 is stripped, bit 6 is not.
- **`mtl_ind_prev_outN`** — may now hold a **sheet** material index, which is a boundary material rather
  than a traversed medium. Say what it is used for (`mat_far` in the slab resolve) so nobody reads it as
  "the ray was previously inside this".
- **`xprmatN`** — the current wording promises the medium is closed out on return. Qualify: on a sheet
  crossing the matrix is *replaced* by the re-resolved composite for the correct bounding media, and no
  medium gain is applied for the sheet material itself.
- **`eps`** — it does not gate the sheet's own series (produce mode has no bail-outs), only the host's.
- **NULL-state mode** — document that with tracking disabled the sheet keeps `ray_mesh_interact`'s
  air-bounded result, per §2.2.
- **`resolved_typeN`** — the qualified bit 6, the re-qualified bit 1, the §6.3 rows, the broadened bit 2
  wording, and the note that 69 / 71 mean something different in `out_typeN`.
- **Kills** — case 4 returns 0; a caller seeing dropped paths in a scene with stacked sheets knows why.

### 9.2 `ray_mesh_interact` doc comment

- Add to the codes 69/71 note that `ray_state_update` performs the re-derivation it describes, so the
  two comments cannot drift apart.
- State that the near/far media written into the sheet composite are **provisional**: geometry-derived,
  air-assumed, and superseded by `ray_state_update` whenever medium tracking is on. Without this a
  reader will reasonably assume the composite is final.

### 9.3 `qd_slab_resolve` comment

Record the §2.3 gate change and why the `far_is_air` proxy was wrong. Also note that produce mode now
has a second caller with a different argument-order convention — the warning above `SLAB_AIRY_FACTOR`
becomes live rather than anticipatory.

### 9.4 Elsewhere

- `quadriga_lib_material_model.md` — the state-machine description needs the sheet transitions: the
  atomic crossing, the two glued transitions, the sheet appearing in `mtl_prev`, and the statement that
  a sheet never becomes the current medium.
- Mark `sheet_refactor_ray_mesh_interact_v1.md` §8.1 and §8.4 as superseded by this document.
- CHANGELOG: new `resolved_typeN` composites, no width change.
- Bindings: no signature change, so MATLAB / Octave / Python need documentation updates only.

---

## 10. Test plan

### 10.1 Fix the existing baseline first

`test_sheet_vs_solid.cpp` needs three changes before it measures anything:

- **Run the primary assertion at `lod 5`.** The file's own comment says lod 5 is the cleaner probe, but
  the test uses lod 1, and the lod-1 solid baseline is not trustworthy as a reference. For this fixture
  (concrete P.2040 at 3.5 GHz, εr 5.24, σ = 0.0462·3.5^0.7822 = 0.1231 S/m, ñ = 2.293 − 0.138j,
  d = 0.1 m) the exact analytic transmittance is

  | θ | 0° | 17° | 34° | 51° | 68° | 85° |
  | --- | --- | --- | --- | --- | --- | --- |
  | analytic | −10.36 | −10.40 | −10.49 | −10.74 | −11.51 | −16.82 dB |
  | lod-1 solid | −11.09 | −10.41 | −10.47 | −10.59 | −10.98 | −14.40 dB |

  The solid tracks analytic to a few hundredths at 17–34°, then departs by 0.73 dB at 0° and 2.42 dB at
  85°. Normal incidence has no complex in-slab angle, so the 0° gap cannot be the entry-boundary defect
  — it is the seven-path aggregation. With lod 1 the two causes are inseparable.
- **Add the analytic column**, so both paths are measured against physics rather than only against each
  other.
- **Assert `out_typeN == 67`** as a precondition, so a classification regression fails as one.

`TOL_REL = 1e-6` is unreachable until the entry-boundary work lands. Stage it: an interim value for this
phase with the target recorded beside it.

### 10.2 Fixtures

Gain alone will not catch a wrong medium or a wrong bounding pair. Assert `mtl_ind_current_outN`,
`mtl_ind_prev_outN` and `resolved_typeN` per case — those are what this phase changes.

| Fixture | Asserts |
| --- | --- |
| Single quad, `thk > 0`, front and back illumination | Case 1; `resolved_typeN == 69`; state unchanged; orientation independence |
| Two coincident quads, reversed normals, identical material | Case 5 equals a single quad, bit-for-bit |
| Quad meeting a solid at an angle | Case 6 arrives as 65/67 and takes the case-1 handler |
| Quad + coincident solid face, front illumination | Case 2; `resolved_typeN == 79`; `mtl_current` becomes the solid; `mtl_prev` becomes the sheet |
| Same, back illumination | Case 3; `resolved_typeN == 71`; medium closed out exactly once |
| **Sheet inside a solid** (`concrete \| sheet \| concrete`) | §3 sequencing; `resolved_typeN == 77`; concrete's series uses the sheet as `mat_near`; concrete's `interface_gain` **not** applied (§2.3) |
| **Sheet inside a solid, second crossing** | The next concrete leg uses the sheet as `mat_far` |
| **Sheet in a medium other than air, glued** (`glass \| sheet \| concrete`) | §2.1 near/far selection with a non-air host |
| Two coincident quads, different materials | Case 4 stub: ray killed |
| Sheet at increasing incidence to near-tangency | Grazing clamp under the §8 decision; no TIR escalation |
| Reflection off an embedded sheet | §5.5; the previously-live `SLAB_AIRY_FACTOR`-on-a-sheet bug |

An equivalence fixture worth adding while the machinery is fresh: **`concrete | sheet | concrete` against
a traced three-layer solid** of the same thicknesses. It is the embedded-case analogue of the primary
sheet-vs-solid test and it exercises the state-derived near/far selection, which nothing else does.

### 10.3 Regression

Every existing test with no `thk` column must stay bit-identical. The dispatch sits behind `typeH[6]`,
which cannot be set without a nonzero `thk` column, and the §6 composites are currently unreachable —
so this holds by construction. Assert it anyway, and check whether any existing test *does* carry a
`thk` column: such a test was previously receiving killed rays, and its numbers will change. That is a
fix, and the expected movement should be recorded before the change lands rather than discovered after.

---

## 11. Known divergences carried in from `ray_mesh_interact`

Not `ray_state_update` bugs, but they surface in this phase's acceptance numbers.

1. **The entry-boundary defect** (companion §4) — the dominant term at oblique incidence in a lossy
   slab. Fixed separately, by inverting the `qd_refract` map to recover the exact entry angle with no
   additional state.
2. **The free-space phase reference.** `arg_phi` is the absolute one-way in-slab phase, correct for the
   Airy denominator, but produce mode also multiplies the transmit port by `φ` once, while
   `SCALE_BY_MEDIUM` gives the solid a phase *relative* to the same distance in free space. With both
   geometries spanning the same total length, sheet and solid differ by `k₀·thk·cos θ_geo` — about
   7.3 rad at 0° for the test fixture. Invisible in `gain`, visible in `xprmat`. Settle the convention
   before the xpr column is asserted.
3. **In-slab cosine projection.** The sheet uses `|Re(ct2)|`; the solid's traced direction gives
   `Re(ct2)/sqrt(η²sin²θ + Re(ct2)²)`, because `qd_refract` normalizes. About 3e-3 relative at 68° for a
   lossy slab. Unify both onto one expression — a one-line change that removes a confound from every
   measurement in this phase, and one the redo makes easy since both now live behind `SHEET_RESOLVE`.

---

## 12. Implementation order

1. Confirm `out_typeN == 67` on the failing test (§1). One printf; it validates the premise.
2. `qd_slab_resolve` interface-gain gate (§2.3). Independent, provably inert for existing cases, and it
   must be in before any embedded fixture is trusted.
3. Decide the grazing-clamp question (§8). It shapes `SHEET_RESOLVE`'s signature.
4. `SHEET_CASE`, `SHEET_RESOLVE`, `CLOSE_HOST_LEG` (§7).
5. Sheet dispatch ahead of the existing chain; case 1/5/6 in air, both passes. **The current test
   produces numbers at this point.**
6. Embedded case 1/5/6, both passes (§5.1, §5.5) — including the live reflection-pass bug.
7. Cases 2 and 3, both passes (§5.2, §5.3), plus the case 4 stub.
8. `resolved_typeN` encoding (§6) and the invariant asserts.
9. Unify the in-slab cosine (§11.3) and settle the phase reference (§11.2).
10. Documentation (§9), fixtures (§10.2).

---

## 13. Code conventions

- **Comment intent, not mechanics.** Every non-obvious branch needs a *why*. The invariants in §2, §3
  and §6.2 are the ones that will not be re-derivable from the code.
- Plain `//` comments only. No framed or boxed comments, no dashes or `====` separators.
- American English throughout.
- Uppercase names for local lambdas, `qd_` prefix for file-scope helpers.
