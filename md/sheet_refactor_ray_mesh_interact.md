# Thin-Sheet Handling in `ray_mesh_interact` — Refactor Specification

**Status:** design agreed, not yet implemented
**Scope of this document:** `quadriga_lib::ray_mesh_interact` only
**Reference source:** `ray_mesh_interact.cpp` (2760 lines, contains both `ray_mesh_interact` and `ray_state_update`)

---

## 1. Problem

Customer scene models are not uniformly watertight. Fences, glass panes, signs, and similar elements
are exported as a **single sheet of triangles** with a semi-transparent texture, not as closed solids
with thickness.

The propagation engine derives the medium a ray is travelling in from face orientation: the sign of
the dot product between the ray direction and the face normal drives an inside/outside state machine
in `ray_state_update`. Entering a solid and exiting it are distinct events, and the geometric path
length between them determines the attenuation.

A zero-thickness sheet breaks this. Crossing it produces an entry event with no matching exit (or an
exit with no entry), so the medium state is corrupted from that point on and **every subsequent
interaction along that path is wrong**.

Two things that today look like they work, but do not:

- A **single sheet** produces a permanent state corruption (entry with no exit).
- A **double-sided sheet** (two coincident, back-to-back faces — what exporters emit to make geometry
  double-sided) is classified as a media-to-media transition with identical materials on both sides,
  hits the `same_as` shortcut in `ray_state_update`, and transmits at gain **exactly 1.0**. The
  attenuation is silently dropped.

## 2. Approach and rationale

Geometry repair (detecting open shells and extruding them to a slab) was considered and **rejected**:
the extrusion thickness would be an invented number that directly scales the computed attenuation,
and for genuinely non-solid objects such as a chain-link fence the sheet abstraction is closer to
physical reality than any slab.

Instead, thickness becomes an **explicit material parameter**, following ITU-R P.2040 (and Nvidia
Sionna RT), which treat thickness as a material property and model everything as a sheet.
quadriga-lib keeps a **hybrid** model: some objects are solid and get their thickness from geometry,
some are sheets and get it from the material table.

The number is still not derived from the mesh — but it is now **authored**. It lives in the
customer's material CSV, it is auditable, and it does not alter the geometry: no faces are added, no
intersection results change, and no lateral displacement enters the traced path. Thickness is used
only inside the analytic slab formula.

Contract:

| `thk` | Behavior |
| --- | --- |
| `0` or column absent | Current behavior, unchanged. Thickness comes from the geometry via the state machine. |
| `> 0` | The face is a **sheet**. Each crossing is one atomic entry + traversal + exit with the internal multi-bounce series resolved in closed form. |

**Corollary — a solid built from a sheet material is not an error.** It becomes a hollow box: sheet
loss on entry, free-space (or the embedding medium) inside, sheet loss on exit. That is a real
physical object and it is the standard building-entry-loss topology, cheaper than a traced solid. A
preprocessing validator may later warn about it, but it must not reject it.

## 3. Scope

### 3.1 In scope for this refactor

1. Classify all sheet cases in `ray_mesh_interact` and emit the correct `out_typeN` bits.
2. Implement the correct interaction physics for **cases 1, 2, 3** (isolated sheet, glued entry,
   glued exit) and **case 5** (collapsed duplicate).
3. Degrade **case 6** (corner) safely.
4. **Detect** case 4 (ordered two-sheet stack) and emit correct classification bits, but implement
   only a best-effort forward path: slab-resolve the first layer, pass through the second, using
   existing building blocks.
5. Extract the slab resolve from `ray_state_update`'s `SLAB_AIRY_FACTOR` lambda into a shared
   file-scope function callable from both public functions.
6. Take refraction direction and TIR from the near/far media rather than from the near/slab pair.
7. Documentation: `out_typeN` bit and composite tables, `thicknessN` disambiguation, stack-depth cap,
   `same_as` collapse.

### 3.2 Explicitly out of scope

| Item | Reason / disposition |
| --- | --- |
| **All `ray_state_update` changes** | Next phase. This refactor must leave `ray_state_update` behaviorally unchanged. |
| **Sheet embedded inside a solid** | Requires the current medium, which `ray_mesh_interact` does not have. `ray_state_update`'s job. |
| **Multi-layer (true two-layer) coherent resolve** | Open issue. Case 4 gets detection plus a documented approximation. A transfer-matrix core is the eventual answer; not now. |
| **Stack depth > 2** | Hard limit of the FBS/SBS intersector pair. Cannot be fixed here. Document as a modeling constraint. |
| **The averaged-reflectance `ISSUE` in `interact_with` (line ~403)** | Pre-existing. Do not touch; the companion correction in the slab resolve must be carried over verbatim. |
| **Relaunch-offset vs. co-location-distance collision** | Pre-existing and not sheet-specific. `ISSUE` comment only (see §9). |
| **Preprocessing validator / topology checks** | Separate work item. |

### 3.3 Hard constraint

> **With no `thk` column present, or all `thk` values zero, `ray_mesh_interact` must be
> bit-identical to today.**

Every change is an added branch guarded by the sheet flag. No existing test may move by one ULP.

---

## 4. Already implemented (baseline)

These are in the reference source and are the starting point:

- `MaterialCols<dtype>::thk` — thickness column, m; validated `>= 0` (line 38, 93, 130).
- `Material::thk` — resolved per-material value (line 169, 207).
- `Material::same_as` now also compares `thk` (line 338).
- `MaterialCols<dtype>::is_sheet(idx)` — `true` iff the column exists, `idx != 0`, and `thk != 0`
  (line 141).
- Local `cols` renamed to `MaterialSet` for readability (line 985).
- `sheetFBS` / `sheetSBS` computed per ray (lines 1289–1290).
- `out_typeN` bit 6 reserved as the thin-sheet flag (documented at line 854).

---

## 5. Classification

### 5.1 Decision inputs

All available at line ~1374, all already computed or cheap:

| Input | Source |
| --- | --- |
| `sheetFBS`, `sheetSBS` | `MaterialSet.is_sheet(iMF/iMS)` (lines 1289–1290) |
| `colocated_faces` | `FS_length < colocation_dist && iSBS != 0` (line 1374) |
| normal parity | the `lim = 1.0e-4` tests at lines 1379–1382 |
| `same_as` | `Material(MaterialSet, iMF).same_as(Material(MaterialSet, iMS))` — new, only evaluated when both faces are sheets |
| `thetaF`, `thetaS` | lines 1328, 1364 |

### 5.2 The six cases

| # | Name | Condition | Handling |
| --- | --- | --- | --- |
| 1 | **Isolated sheet** | `sheetFBS && !colocated_faces` | `near = far = air`. Undeviated. TIR forced off. Slab resolve on the FBS material. |
| 2 | **Glued entry** | exactly one sheet, co-located, parallel (either parity), solid face **front**-hit | `near = air`, `slab = sheet`, `far = solid`. State transitions o→i (in `ray_state_update`). |
| 3 | **Glued exit** | as case 2, solid face **back**-hit | `near = solid`, `slab = sheet`, `far = air`. State transitions i→o. |
| 4 | **Ordered stack** | both sheets, co-located, `!same_as` | *Detection only.* Traversal order per §5.4. Slab-resolve layer 1, pass through layer 2. Open issue. |
| 5 | **Collapsed duplicate** | both sheets, co-located, `same_as` | Exporter double-sided artifact. Collapse to **one** crossing; identical physics to case 1, but co-located so the relaunch goes past SBS. |
| 6 | **Corner with sheet** | co-located, **not** parallel, and `sheetFBS \|\| sheetSBS` | Drop SBS entirely; treat FBS alone (as a sheet if `sheetFBS`, else as an ordinary single hit). Clear the corner flag. Relaunch past SBS. |

### 5.3 Why "glued" implies sheet-first-on-entry

Case 2/3 rest on an invariant that should be stated in the code comments, because it makes the rule
provable rather than conventional:

> At a co-located sheet + solid pair, the sheet is always on the **incidence side** of that solid
> face. A sheet glued to the solid's *other* face would sit at the solid's opposite boundary and
> would therefore not be co-located with this one — unless the solid is thinner than
> `colocation_dist`, which is degenerate.

Wallpaper on the room side of a wall: entering from the room gives `air | paper | wall` (front hit on
the solid, sheet traversed first); exiting from inside the wall gives `wall | paper | air` (back hit,
sheet traversed last). Both follow from co-location alone.

**A sheet's own normal orientation carries no information** (a sheet has no inside). Therefore the
front/back decision for cases 2 and 3 must be taken from the **solid** face's normal, which may be
either FBS or SBS depending on the resolve order.

### 5.4 Traversal order for case 4

Two co-located sheets of different materials form an ordered stack (a thermally coated window is the
motivating example: it must give different results when hit from either direction).

Rule: **the sheet hit on its front side is traversed first; the back-side hit is traversed last.**
This is direction-reversal symmetric (reverse the ray and the layer order reverses), and it encodes
layer order in the one piece of information the exporter actually carries — normals point outward
from the assembly.

Fallback: if both faces are front-hit or both back-hit (same-direction normals), the order is
ill-posed. Use **FBS → SBS** and set the same-direction flag (bit 3) to record that the order was
inferred geometrically rather than from normals. This is the geometrically correct order whenever the
faces have any real separation, and it is also direction-reversal symmetric.

### 5.5 The `same_as` collapse (case 5)

Two coincident sheets with **identical** materials are, in practice, almost always an exporter
double-sided duplicate rather than a genuine laminate — nobody laminates glass to identical glass,
they would double `thk`. And since the pair is within `colocation_dist`, there is no air gap to
model, so collapsing loses nothing real.

Apply the collapse **regardless of normal parity**, so it also catches identical sheets stacked with
matching normals. Without it, case 4's rule would turn every double-sided export into a genuine
2-layer stack: 2 × `thk`, 2 × the attenuation, plus spurious interference between a layer and its own
copy.

This must go in the user-facing documentation as a modeling note.

---

## 6. `out_typeN` encoding

### 6.1 Bit 1 gains a context-dependent meaning

Bit 1 must be written by `ray_mesh_interact`, not re-derived downstream, so the ordering rule
(normal-parity test, `same_as` collapse, FBS→SBS fallback) lives in exactly one place.

| Context | Bit 1 meaning |
| --- | --- |
| Solid (bit 6 = 0) | Front-side hit on FBS — **existing meaning, unchanged** |
| Isolated sheet (bit 6 = 1, bit 2 = 0) | Front-side hit on FBS; informational only, not used by the physics |
| Glued (bit 6 = 1, bit 2 = 1, exactly one sheet) | Front-side hit on the **solid** face |
| Sheet stack (bit 6 = 1, bit 2 = 1, two sheets) | `1` = FBS traversed first, `0` = SBS traversed first |

Both meanings must be written into the doc-comment table at line ~849.

### 6.2 Composite codes do not uniquely identify the case — by design

With bit 6 = 64 the reachable sheet composites are:

| Code | Bits | Cases |
| --- | --- | --- |
| 65 | OK + sheet | Isolated sheet, back-side hit |
| 67 | OK + front + sheet | Isolated sheet, front-side hit |
| 69 | OK + co-located + sheet | Glued **exit** (case 3); stack with SBS first (case 4); collapsed duplicate (case 5) |
| 71 | OK + front + co-located + sheet | Glued **entry** (case 2); stack with FBS first (case 4); collapsed duplicate (case 5) |
| 79 | 71 + same-direction | Case 4 with ambiguous order (geometric fallback used) |
| 101 / 103 | 69 / 71 + TIR | Glued cases only (see §7.3) |

Codes 69 and 71 are shared. This is deliberate and safe: `ray_state_update` builds its own
`MaterialSet` and has both `mtl_ind_fbs` and `mtl_ind_sbs`, so it can re-derive `is_sheet` on each
face and `same_as` on the pair. The full case is therefore
`(composite code, is_sheet(FBS), is_sheet(SBS), same_as)`.

**Consequence for the `ray_state_update` phase:** bit 6 alone means *"at least one of FBS/SBS is a
sheet"*. Fix the doc-comment at line 854, which currently reads "material is a thin single-face
sheet" and implies a single face. Bit 6 is a **hint**; the re-derivation is authoritative. Do not
introduce a second source of truth.

Bit 7 remains free.

---

## 7. Physics

### 7.1 `interact_with` cannot express a sheet

`Material::interact_with` is a **two-medium interface** function. A sheet crossing is a
**three-medium, two-interface, one-slab** problem. There is one formulation that covers every case:

| Case | `near` | `slab` | `far` |
| --- | --- | --- | --- |
| 1 — isolated sheet in air | air | sheet | air |
| 1 — isolated sheet embedded in X | X | sheet | X | *(out of scope: needs state)* |
| 2 — glued, front hit | air | sheet | solid |
| 3 — glued, back hit | solid | sheet | air |
| 5 — collapsed duplicate | air | sheet | air |

`ray_mesh_interact` recovers `near` for cases 2/3 using the same trick the existing `M1`/`M2` block
uses at lines 1415–1426: a back-side hit implies the ray is inside that material. No state needed.

### 7.2 Shared slab core

Extract `SLAB_AIRY_FACTOR` (currently a lambda inside `ray_state_update`, lines ~2402–2504) into a
file-scope `static inline` function, placed with `qd_reflect` / `qd_refract` / `qd_polbasis` around
line 500. Both public functions live in this translation unit, so no header change is needed.

Intended shape (spec, not implementation):

```
// Analytic thin-slab (Fabry-Perot) resolve for a near | slab | far stack.
// produce == true : returns the full composite coefficients (sheet path, no tracer to re-emit to)
// produce == false: returns the per-channel correction factor (existing ray_state_update caller)
static inline bool qd_slab_resolve(const Material &mat_near,
                                   const Material &mat_slab,
                                   const Material &mat_far,
                                   double dist_refract, double dist_geo,
                                   double abs_cos_theta_t, double abs_cos_theta,
                                   double fGHz, bool is_scalar, double eps, bool produce,
                                   std::complex<double> *T_te, std::complex<double> *T_tm,
                                   std::complex<double> *R_te, std::complex<double> *R_tm);
```

Notes for the implementer:

- The current lambda captures a large amount of enclosing context by reference (`dist_refract`,
  `dist_geo`, `abs_cos_theta_t`, `abs_cos_theta`, `cTE`, `cTM`, `dir_prev`, `dir_next`, `N`,
  `p_xprmatN`, `p_gainN`, `nXPR`, `ii`, `eps`, `prev_nonpar`, `is_scalar`). Making these explicit
  parameters is the bulk of the mechanical work. Keep the `qd_polbasis` call and the xprmat write at
  the **call sites**, not inside the core — the two callers write to different buffers.
- In `produce == false` mode the result must be **bit-identical** to today. This is the first thing
  to test, before anything sheet-specific is wired up.
- Carry the `ISSUE` comment at line ~2492 (averaged-vs-per-channel port correction) across verbatim.
  It stays coupled to the `ISSUE` at line ~403 in `interact_with`.
- `tf` ownership: the existing `owner = slab_is_air ? adj : mat_slab` rule already resolves to the
  sheet for every sheet case (the slab is never air). Keep it. `tf` is therefore applied at both
  interfaces, which is exactly what a traced solid slab does today — required for the equivalence
  test in §10.

### 7.3 Escape hatches must be replaced for the sheet path

A sheet's internal path length is **zero in world coordinates**. There is nowhere to relaunch to; the
tracer would either loop at the same point or the ULP nudge would push it past both faces. Resolve is
therefore mandatory, and each existing bail-out needs a fallback that returns a number:

| Bail-out | Line | Sheet-path replacement |
| --- | --- | --- |
| `eps > 0 && prev_nonpar` | ~2404 | Resolve anyway. The `eps = 0` mode already exists for callers that cannot re-emit. |
| `mat_slab.m > 0.0` (mass-law) | ~2412 | **Incoherent** route: apply `medium_gain` and the interface gains, drop the phase and the series. The in-slab index is not physical in scalar mode, so the Airy phase is meaningless regardless of whether `thk` is known. |
| `rho < eps` (weak-slab survival gate) | ~2464 | Resolve anyway. |
| `\|denom\| < 1.0e-2` (near-pole) | ~2464 | **Clamp**, do not bail. |
| Internal TIR / evanescent underflow | — | New. A dense slab past critical gives `exp(-large)` underflowing to zero; guard the R/T extraction against 0/0. |

### 7.4 Refraction direction and TIR come from `near`/`far`

Snell's law is transitive: `n_near·sinθ_near = n_slab·sinθ_slab = n_far·sinθ_far`. Therefore:

> **The net bend depends only on `near` and `far`. The sheet material does not affect the outgoing
> direction at all.**

Split the two calls at line 1439 by purpose:

- **Geometry:** `M_near.interact_with(M_far, interaction_type, theta_eff, ...)` → `eta`,
  `cos_theta2`, `total_reflection`. Discard the coefficients.
- **Coefficient:** `qd_slab_resolve(...)` when bit 6 is set; the existing single call otherwise.

For cases 1, 4, 5 this gives `near == far`, so `eta == 1`, the direction is unchanged, and the
undeviated result falls out without special-casing `geometry_type` at lines 1490–1495.

**TIR must be taken from the near/far pair.** With `near == far` the Snell ratio is 1 and TIR is
unreachable except at exactly 90°. Computing it from `interact_with(air, sheet)` would produce
spurious total reflection past the air|sheet critical angle — and in scalar mode the impedance
mapping can easily produce `n < 1`, so an acoustic sheet would flip into a mirror. That also poisons
the whole-tube escalation at lines 1481–1484, which would turn the entire ray tube into a mirror off
the spine's verdict alone.

Consequence: bit 5 (TIR) and bit 6 (sheet) can co-occur **only for the glued cases 2/3**, where
`near != far`. A case 1/4/5 sheet setting bit 5 is a bug; worth an assert.

### 7.5 Reflection off a sheet

The reflection port is its own Airy sum and is **not** what `interact_with(near, sheet, type 0)`
returns:

```
R_slab = r_ns + (t_ns · t_sn · r_sf · phi^2) / (1 - r_sn · r_sf · phi^2)
phi     = exp(-i · k0 · n_slab · thk / cos(theta_slab))
```

For `near == far` this collapses to the symmetric thin-slab reflectance and is direction-independent
— which is precisely what makes the "sheet orientation is irrelevant" claim true for cases 1/4/5. For
the glued cases `near != far` and `R_slab` **is** direction-dependent, so the reflection pass needs
the front/back bit taken from the solid face (§5.3).

`qd_slab_resolve` should return `R_te`/`R_tm` alongside `T_te`/`T_tm` so both ports come from one
pass over the same Fresnel quantities.

**Expect visibly different results from today.** A 6 mm pane at 3.5 GHz with εr ≈ 6 has a round-trip
optical path of about 29 mm against λ = 86 mm, so R will show real angular and frequency structure
instead of sitting near the smooth half-space Fresnel value. This is correct and matches
P.2040/Sionna, but `subdivision_tolerance_m` was tuned against a smooth angular response and may need
revisiting later.

### 7.6 Traversal distances

Map onto the existing `medium_gain` signature exactly:

```
dist_refract    = thk / |Re(cos_theta2_slab)|
abs_cos_theta_t = |Re(cos_theta2_slab)|
dist_geo        = thk / abs_cos_theta
abs_cos_theta   = abs_cos_theta
```

This is what makes the mass-law term evaluate for acoustic sheets, which it cannot do today at zero
geometric thickness.

**Watch `mass_min_path = 0.0015`** (line ~271): a sheet thinner than 1.5 mm silently gets no mass-law
term at all. Fine for a 4 mm pane; not fine for a 1 mm fence wire. There is no reason for this limit in sheets.
In fact, this is one of the main cases for using sheets, so the mass_min_path = 0.0015 shoud only apply to solids, not sheets.

**Grazing clamp.** `thk / cosθ` diverges at tangency. The physically motivated cutoff is when the
lateral walk-off `thk · tanθ` exceeds the face's in-plane extent — past that point the ray would have
left through the side of the notional slab. `edge_len` (already computed around line 1261) is a
usable proxy for the extent.

### 7.7 Case 4 interim behavior

Detection is full (§5.2, §5.4). Physics is a documented approximation:

- Slab-resolve layer 1 with `near = air`, `slab = layer 1`, `far = layer 2`.
- Pass layer 2 through with its interface gain and medium gain only — no second Airy series, no
  cross-layer coupling.
- Direction and TIR from `near = air`, `far = air`.

This is wrong in the coupling term but bounded and monotone, and it uses only existing building
blocks. Mark it clearly in the code as an approximation with a pointer to §11.

---

## 8. Touch points in `ray_mesh_interact`

| Region | Line (ref) | Change |
| --- | --- | --- |
| Case classification | 1374–1395 | Replace the `SHEET:` marker with the full six-case decision. Compute `same_as` only when both faces are sheets. |
| Effective front/back | 1373, 1398 | Introduce `front_eff`; take it from the solid face for cases 2/3 and from the ordering rule for case 4. Use it for the normal flip at 1398 instead of `thetaF < 0.0`. |
| `M1`/`M2` assignment | 1411–1426 | For sheet cases, resolve into `(near, slab, far)` instead. |
| Interface evaluation | 1435–1440 | Split into the geometry call and the coefficient call (§7.4). |
| Whole-tube TIR | 1481–1484 | Skip for cases 1/4/5 (`eta == 1`, unreachable); keep for cases 2/3. |
| Direction | 1490–1495 | No special case needed once `eta` comes from near/far, but assert that cases 1/4/5 land in the undeviated branch. |
| Ray-tube loop | 1501–1600 | Per-vertex incidence feeds `qd_slab_resolve` the same way it feeds `interact_with` today. Verify the tube does not diverge from the spine on the resolve/no-resolve decision. |
| Relaunch | 1627–1640 | Unchanged, but cases 5 and 6 rely on the co-located relaunch going past SBS. Add the `ISSUE` from §9. |
| `path_dirN` | 1647–1652 | For cases 1/4/5 the Snell direction equals the incoming direction; confirm no spurious bend. |
| `xprmat` / `gainN` | 1655–1661 | Sheet cases supply `cTE`/`cTM` from `qd_slab_resolve`; the `qd_polbasis` call itself is unchanged. |
| `out_typeN` write | 1677–1681 | Bit 6 already set during classification; bit 5 written here as today. |
| Doc comment | 845–870 | Bit table, bit 1 dual meaning, composite table, `thicknessN` note. |

---

## 9. Known issues to record in code, not fix

**Relaunch offset vs. co-location distance.** `relaunch_offset = 8.0 * scale * 1.1920929e-7` (line
1630) is ~8 float ULP at the FBS coordinate magnitude. At scene coordinates around 1000 m this is
~1e-3 m — exactly `colocation_dist` (line 20). At 8000 m it is ~7.8e-3 m. A second face 1.5 mm behind
the first is **not** co-located, so FBS is handled alone and the relaunch overshoots it, skipping the
face entirely.

This is coordinate-dependent: it passes every test near the origin and fails on a real city scene.
Clamping the offset to `0.5 * FS_length` would defeat the ULP scaling that exists so the nudge
survives the `dtype(origN)` store, so there is no cheap correct fix. It is also **pre-existing and
not sheet-specific** — it affects thin solids identically — and should be fixed globally rather than
in the sheet path.

Add an `ISSUE` comment at line ~1630 stating the magnitude, the coordinate dependence, and that it
predates sheets.

**Stack depth capped at 2.** The FBS/SBS intersector returns two faces. A three-layer stack is
silently truncated: layers 1–2 are handled together, layer 3 is hit as a separate FBS after the
relaunch and concatenates incoherently. The same cap applies to a sheet sandwiched between two solids
(`solid A | sheet | solid B` is three faces). This cannot be fixed here. **Document it as a modeling
constraint** so scene authors can design around it.

**`thicknessN` is now ambiguous.** The output at line 1671 is the FBS-to-SBS geometric distance,
unrelated to the material `thk`. Either rename it or document it as geometric in the doc comment at
line 842. Decide before this ships; two unrelated meanings of "thickness" in one API will cost time
later.

---

## 10. Acceptance criteria

### 10.1 Primary equivalence test

> **A sheet must behave identically to a solid of the same thickness.**

Model the same 6 mm pane twice — once as a closed manifold box with a `thk = 0` material, once as a
single quad with `thk = 0.006` — and require agreement in gain, phase, and polarization.

This single fixture validates the slab-resolve extraction, the new reflection composite, the
traversal-distance mapping, and the near/far direction rule at once, against machinery that is
already trusted.

**Two known deviations to design around:**

1. **Lateral offset.** A traced solid refracts in and out, so the exit point is displaced by
   `thk · tan(theta_slab)`; the sheet is not. Run the primary comparison in **`geometry_type 1`
   (undeviated transmission)**, where the solid does not trace the bend either, so the two paths are
   directly comparable at any incidence angle.
2. **Refraction mode.** In `geometry_type 2`, compare at **normal incidence** (offset is zero) for
   exact agreement; at oblique incidence compare the exit *direction* and the gain, not the exit
   position. The walk-off convention in `SCALE_BY_MEDIUM` (the `cos2` / `cos2_geo` terms around line
   2370) differs between the traced and untraced cases — resolve exactly how during implementation
   rather than assuming.

Also verify that `att` (interface gain) is applied **once** per traversal in both models. In the
solid path `interact_with` folds `other.interface_gain()` in on entry (`other` = material) but not on
exit (`other` = air, `att` = 0). The sheet path must match.

### 10.2 Regression

- Every existing test with no `thk` column: **bit-identical** output.
- `qd_slab_resolve` in `produce == false` mode: bit-identical to the current `SLAB_AIRY_FACTOR`.
  Test this **before** wiring anything sheet-specific.

### 10.3 New fixtures

| Fixture | Covers |
| --- | --- |
| Single quad, `thk > 0`, front and back illumination | Case 1; orientation-independence |
| Quad + coincident solid face, both illumination directions | Cases 2 and 3; front/back from the solid face |
| Two coincident quads, reversed normals, identical material | Case 5; must give the same result as a single quad |
| Two coincident quads, different materials, both normal parities | Case 4 detection, ordering rule, and the ambiguous fallback |
| Quad meeting a solid at an angle | Case 6 degradation |
| Sheet at increasing incidence up to near-tangency | Grazing clamp; no whole-tube TIR escalation |
| Acoustic sheet with `m > 0` | Incoherent mass-law route; confirm `mass_min_path` behavior |

---

## 11. Open issues carried forward

1. **True two-layer coherent resolve (case 4).** Two stacked sheets are four media and the internal
   reflections couple across both layers; composing two single-slab Airy results is not equivalent.
   The recommended eventual fix is a **transfer-matrix core** rather than a two-layer special case: it
   is exact for any layer count, reduces exactly to the Airy formula at N = 1 (giving a built-in
   regression test against results already trusted), produces the R and T composites in one pass, and
   handles the asymmetric glued case for free. It is also the natural moment to fix the averaged-
   reflectance `ISSUE` in `interact_with`, since the companion correction disappears. Cost is roughly
   a 2×2 complex matrix product per layer per polarization.
2. **Sheet embedded inside a solid.** `near = far = current medium`. Needs `ray_state_update`.
3. **Relaunch offset** (§9).
4. **`subdivision_tolerance_m`** against resonant slab reflectance (§7.5).
5. **Preprocessing validator**: warn on closed manifold components built from `thk > 0` materials;
   warn on open patches whose material has `thk == 0`. Uses the existing union-find connected-
   component analysis, split at edges with valence ≠ 2 so a pane welded to its frame separates
   cleanly.

---

## 12. Code conventions for the implementation

- **Comment intent, not mechanics.** The `ray_state_update` follow-up will be written in a fresh
  session with no memory of this discussion. Every non-obvious branch needs a comment saying *why*,
  in particular: the co-location invariant (§5.3), why bit 1 changes meaning (§6.1), why direction
  and TIR come from near/far (§7.4), and why case 4 is an approximation (§7.7).
- Plain `//` comments only. No framed or boxed comments, no lines of dashes or `====` separators.
- American English throughout, in code comments and documentation.
- Match the existing style in this file: uppercase lambda names for the local helpers, lowercase for
  free functions, `qd_` prefix for file-scope helpers.
