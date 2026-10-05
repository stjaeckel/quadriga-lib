# Sheet / Solid — Fix Plan v2

**Status:** analysis of the diagnostic run with a repaired harness. No implementation yet.
**Scope:** the source changes needed in `ray_mesh_interact.cpp`, in the order they should land.
**Companion:** `sheet_solid_divergence_v3.md` for the end-to-end measurements this plan acts on.

Every mechanism below is now pinned by a measurement that reproduces it to at least 1e-9 rather than
inferred from a plausibility argument. Two of them turned out to be somewhere other than where the
earlier analysis put them, and the corrections change both the size and the risk of the work.

---

## 1. Why nothing fails, and when that changes

The eight diagnostic cases assert only sanity — finite, non-negative, `gain` consistent with
`xprmat` — so the whole report prints instead of stopping at the first open divergence. That was the
right trade for getting numbers, and it has paid for itself: every item in §2 now has a closed-form
prediction that matches the measurement.

It stops being the right trade here. Each fix in §4 lands together with the assertion listed beside
it, replacing the sanity check in the case that measures it. **Three of those assertions are already
satisfied and should be promoted immediately**, before anything else moves (§6).

---

## 2. What the run establishes

**A. `Material::interact_with` is not the problem.** Per-channel, at a single interface, air → lossy
dielectric: `VV² == ref_TE` and `HH² == ref_TM` to all printed digits, `gain == ref_avg` exactly, at
every angle, for both reflection and refraction. Worst `d_ratio` 7.8e-16.

The averaged reflectance is **inert at `tf = 0`**: for type 0 the rescale is
`sqrt(gain/reflectance) = 1`, for type 2 `tf_scale = 1`, and for type 1 it is a single common factor
across both channels and so cannot move a ratio. The averaging sets the total energy, which is
correct as a total, and leaves the per-channel split untouched.

**B. The polarization distortion is entirely in `qd_slab_resolve`'s `corr` / `Tsp`.** The measured
output ratio is reproduced to **3e-10** at every angle by

```
ratio ∝ |T_X|² · |S_X|² · Tsp_X          Tsp_X = (1 − Rn_X)(1 − Rf_X)
```

— the port's own per-channel Fresnel transmission, times a **second copy** of the per-channel
single-pass transmittance supplied by `fac_X = S_X · sqrt(Tsp_X · corr)`.

| θ | 10° | 20° | 30° | 40° | 50° |
| --- | --- | --- | --- | --- | --- |
| measured `ratio_out` | 0.99659 | 0.98410 | 0.94673 | 0.84551 | 0.66889 |
| `\|T\|²·\|S\|²·Tsp` prediction | 0.99659 | 0.98410 | 0.94673 | 0.84551 | 0.66889 |
| per-channel target | 0.99997 | 0.99882 | 0.98317 | 0.91491 | 0.75701 |

Dropping the extra factor moves 50° from 0.6689 to 0.8891.

**C. The Airy denominator uses the traced angle on both boundaries.** `S_applied == S_traced` to
0.000000 dB at every angle, for the lossy material and the lossless control alike. Against the
reconstructed exact entry angle, `S_exact − S_traced` reaches **−0.321 dB at 50°** for
`eps 1.50 − 0.40j` over 100 mm, and is identically zero for the lossless control.

**D. The series collapses to exactly 1 when the previous medium matches the slab.** With
`mtl_prev = air` the applied factor reproduces the reference to 0.000000 dB in all eight
combinations, validating the reference. With `mtl_prev` set to the electrically identical sheet,
every row reads `|S| = 1.000000`, `arg = 0.000`. The cost oscillates with the fringe and reaches
**0.988 dB** at 50° over 50 mm.

**E. The excess phase is exactly `−k₀(n−1)·L·cos²θ_fbs`.** Zero difference at all six angles, against
a raw-path-length form that diverges to 26.17°.

**F. The sheet is exact, and its phase convention is fully characterized.** Worst `|d_dB|` 0.025 dB,
worst `d_ratio` 5.3e-3, across five materials and eight angles. The phase offset against the analytic
slab matches `k₀·thk·cos θᵢ` **exactly** for the lossless material at all eight angles, and to 0.31°
worst case for the lossy ones.

**G. The in-slab cosine projection is negligible.** 7.5e-4 relative worst case — 0.083 mm of path
over a 100 mm slab, under 0.01 dB.

**H. The grazing clamp freezes rather than caps, and the two functions clamp differently.** On a
566 m face the sheet stays within 0.017 dB of analytic to 89.9°. On a 57 mm face the transmission
freezes at −3.325 dB from 85° onward while the analytic slab keeps attenuating:

| θ | 85° | 88° | 89° | 89.5° | 89.9° |
| --- | --- | --- | --- | --- | --- |
| small face − analytic | +0.49 | +4.45 | +9.30 | +14.94 | **+28.77 dB** |

`ray_mesh_interact` clamps that face past **83.95°**; `ray_state_update`'s stand-in ratio clamps past
**89.94°**. Since the redo always runs when medium tracking is on, the face-extent clamp no longer
reaches the output at all.

---

## 3. Ranked issues

| # | Issue | Measured size | Where |
| :-: | --- | --- | --- |
| 1 | `Tsp · corr` double-counts the per-channel transmittance | ratio 0.669 vs 0.757 at 50° | `qd_slab_resolve` correction mode |
| 2 | Grazing clamp freezes, and the two functions disagree | up to 28.8 dB | both clamps |
| 3 | Entry boundary re-derived at the traced angle | 0.32 dB at 50°, weak reflector | `qd_slab_resolve`, `ray_state_update` |
| 4 | Series lost when the bounding medium matches the slab | up to 0.99 dB, oscillating | `SHEET_EVENT` / `SLAB_AIRY_FACTOR` |
| 5 | Excess-phase convention differs between sheet and solid | phase only | `SCALE_BY_MEDIUM` |
| 6 | Glued entry / exit reciprocity residual | 0.068 dB at 85° | not yet isolated |
| 7 | `sqrt(1−R)` amplitude convention | 0.015 dB floor | `interact_with` + `qd_slab_resolve` |
| — | In-slab cosine projection | < 0.01 dB | **closed, §4.8** |

---

## 4. The fixes

### 4.1 Remove `Tsp · corr` from correction mode

**Evidence.** Item B: the measured ratio is the port's per-channel transmission times a second copy
of it, reproduced to 3e-10.

**Root cause.** The design comment states the intent plainly — `|fac|² · |port|² = Tsp · |S|²` — which
assumes `|port|²` is the **averaged two-interface** single-pass transmittance. It is neither. Item A
shows the port carries the **per-channel** coefficient, and in `ray_state_update`'s composition each
interface is its own interaction, so the port at an exit event carries **one** interface, not two.
`corr = 1/(Tn_avg · Tf_avg)` divides by two interfaces' worth of an averaged quantity while
multiplying back a per-channel one.

**Change.** Return `S_te` and `S_tm` per channel — the shape the scalar branch already uses — and
delete `corr`, `single_pass`, `Tsp_te`, `Tsp_tm` and the `ISSUE` comment tying them to
`interact_with`.

**Risk: much lower than previously assessed.** `interact_with` does not need touching, which was the
part that would have moved every oblique-incidence result in the library. This is confined to one
branch of one function. It still moves any test exercising a resolved slab, so record those
baselines, but the blast radius is a fraction of what it looked like.

**Residual to expect.** Removing the double count gives 0.8891 at 50° against a per-channel target of
0.7570. The remaining gap is the TE/TM obliquity asymmetry of a lossy interface —
`Re(z₁c₁·conj(z₂c₂))` differs from `Re(z₂c₁·conj(z₁c₂))` once both are complex — a convention
question belonging to §4.7. Settle the two together, or the second will look like a new defect.

**Assertion.** *Slab factor per polarization*: `ratio_out == ratio_ref` to 1e-9 once §4.7 is also
settled; to the §4.7 floor before that.

### 4.2 Decide the grazing clamp

**Evidence.** Item H. Promoted from "dormant, decide later": the divergence reaches 28.8 dB and the
face-extent clamp is currently dead code for any run with medium tracking on.

**The modelling question comes first.** The clamp caps the traversal at `sqrt(thk² + ext²)` because a
grazing ray through a *finite* pane would leave through its side rather than crossing it. Against an
infinite-slab reference that reads as an error; against a 57 mm pane it may be the better answer. The
analytic column cannot arbitrate — it assumes an infinite slab.

**What is not defensible either way** is that the two functions apply different extents and the one
that reaches the output is the placeholder. Three options:

- Pass the FBS face extent through to `ray_state_update` so both clamp identically. Needs an output
  slot or a signature change; preserves the finite-face physics.
- Drop the clamp from the redo and let the traversal diverge, guarded only against overflow. Matches
  the infinite-slab reference and makes the sheet exact at all angles; discards the finite-face
  modelling.
- Keep the placeholder but choose `sheet_extent_ratio` deliberately, and document that faces whose
  extent is within a few hundred thicknesses are modelled as infinite.

**Assertion.** *Grazing clamp*: large face within the §4.7 floor of analytic to 89.9° (already true);
small face departing only past its computed threshold, which the case now prints.

### 4.3 Reconstruct the entry angle

**Evidence.** Item C: `S_applied == S_traced` exactly, so both boundaries are built from the real
projected in-slab angle and the coefficient `ray_mesh_interact` applied at entry never enters the
series. The isolated cost is 0.321 dB at 50° on a weak reflector; `eps 1.50` reflects far less than
concrete, so this understates the effect for realistic materials.

**Change.** Recover the exact entry angle by inverting the `qd_refract` map — two Newton steps on
`η²u = t²·Re(√(1 − εu))²`, `u = sin²θᵢ`, no additional state. Carry the resulting complex in-slab
cosine into the exit `interact_with` in `REPLACE_BY_VBS`, into both boundaries of correction mode,
and into the propagation factor.

The diagnostic already contains the reference implementation: `diag_ct_exact` inverts Snell exactly as
the fix proposes, and its `S_exact` column is what the applied value should become.

**Gates.** Parallel faces — note `eps == 0` currently bypasses the `prev_nonpar` wedge gate and must
not for this; the direction having actually been bent, so skip for `geometry_type 1` and under TIR;
`path_dir_prev` present; and a lossless entry medium.

**Sequencing.** After §4.1 — both move the same numbers and §4.1 is the larger term.

**Assertion.** *Entry boundary*: `S_applied == S_exact` to 1e-9 for the lossy material; the lossless
control showing all three columns coincide, which is already true.

### 4.4 Stop splitting the layer at an invisible boundary

**Evidence.** Item D: `|S| = 1.000000` exactly whenever the previous medium is electrically identical
to the slab, costing up to 0.99 dB and oscillating with the fringe. End-to-end, the glued and embedded
stacks reproduce the single-pass slab with `measured − single-pass` flat at −0.016 dB.

**Root cause.** After a sheet crossing the sheet index goes into the previous-medium word, so the host
layer's denominator is evaluated with the sheet as a bounding medium. Matched impedance gives `r = 0`
and the denominator becomes 1. The host layer has been cut in two at a boundary that does not reflect.

**Interim.** In `SHEET_EVENT`, when the sheet and the host are the same medium electrically, do not
close the host leg and do not reset the accumulated distance: carry it through, apply the sheet
composite, and let the series resolve once at the far side over the full path. Small, and it restores
the embedded case exactly. It is a special case and does nothing for a coated window.

**Target.** A transfer-matrix core resolving the whole stack in one pass. Exact for any layer count,
reduces to the Airy formula at one layer, produces R and T together, handles the asymmetric glued
case, and closes case 4. It is also where §4.7 is naturally settled.

**Recommendation.** Take the interim now and schedule the transfer matrix separately — but record it
as an interim in the source, or it will be mistaken for the general fix.

**Assertion.** *Series collapse*: `|S|` and `arg(S)` independent of `mtl_prev` to 1e-9.

### 4.5 Settle the excess-phase convention

**Evidence.** Item E pins the applied form to `−k₀(n−1)·L·cos²θ_fbs`. Item F pins the sheet's to
`−k₀·n·thk·cos θ_slab + k₀·thk·cos θᵢ`. Three conventions coexist:

| where | form |
| --- | --- |
| sheet, produce mode | `−k₀·n·thk·cos θ_slab + k₀·thk·cos θᵢ` — transverse-matched, referenced |
| solid, `walk_off = true` | `−k₀(n−1)·L·cos²θ`, the same cosine on both terms |
| solid, `walk_off = false` | `−k₀(n·L_refr − L_geo)`, raw path lengths |

The third is self-consistent for a ray that is genuinely displaced, because the lateral run is already
in the geometric path length. The first is the standard slab phase and is what the analytic reference
uses. The second is neither: it applies the `cos²` walk-off but with the geometric cosine on both
terms, so it does not reduce to the first even for an undeviated tracer inside a real slab.

**Change.** A decision before a patch. The recommendation is to keep the two cosines distinct in the
walk-off branch — `n·dist_refract·cos²θ_t − dist_geo·cos²θ_geo`, which is what the expression already
reads but with the two genuinely differing only after a VBS correction — so the undeviated solid and
the sheet converge. The `walk_off = false` branch stays, and its difference from the sheet is recorded
as the lateral-offset deviation rather than as a defect.

**Assertion.** *Excess-phase walk-off*: applied phase matching the chosen form to 0.01°, both branches
exercised.

### 4.6 Glued entry / exit reciprocity residual

End-to-end, cases 2 and 3 agree to 0.0001 dB through 30° and diverge to 0.068 dB at 85°, growing
monotonically. Reciprocity requires exactly zero.

Not yet isolated, and likely to move under §4.1 and §4.3 since all three touch the same numbers.
Defer, and add a dedicated unit case: the same three-medium stack resolved from both directions,
asserting equality to 1e-9.

### 4.7 The `sqrt(1−R)` amplitude convention

The 0.015 dB floor, common-mode across the sheet, the solid and the glued composition. It is also the
residual §4.1 will expose: once the double count is gone, the remaining ratio gap is the TE/TM
obliquity asymmetry of a lossy interface, which lives in the same energy convention.

Settle it with §4.1, or at the transfer-matrix core of §4.4 where the companion correction disappears
anyway. Do not touch it alone.

### 4.8 In-slab cosine projection — closed

Item G: 7.5e-4 relative worst case, 0.083 mm over a 100 mm slab, under 0.01 dB. Below the §4.7 floor
and below every other item on this list. Close it as a known, quantified, negligible difference; the
diagnostic case stays as a regression guard with a 1e-3 bound.

---

## 5. Order

1. **§4.1.** Largest term, and now a single-branch change with `interact_with` untouched. Baselines
   first, but a much smaller set than previously expected.
2. **§4.7 decision**, immediately after — §4.1 exposes the residual and the two are the same
   convention question.
3. **§4.4 interim.** Small, self-contained, unblocks the embedded and glued fixtures.
4. **§4.5 decision**, then the patch. Phase only, so it cannot confound the gain work.
5. **§4.3.** Only cleanly measurable once §4.1 is out of the way.
6. **§4.2 decision.** Modelling question first, code second.
7. **§4.6**, with its own fixture, once the numbers above have stopped moving.
8. **§4.4 target** — the transfer-matrix core — as its own work item, closing case 4 with it.

---

## 6. Acceptance

Each replaces the sanity assertion in the case that measures it.

- *Interface energy split*: `d_ratio ≤ 1e-12`, both interaction types, all angles. **Already true.**
- *Thin sheet vs analytic*: `|d_dB| ≤ 0.03`, `|d_ratio| ≤ 1e-2`, `d_arg == k₀·thk·cos θᵢ` to 0.5°.
  **Already true.**
- *In-slab cosine*: relative difference ≤ 1e-3. **Already true.**
- *Slab factor per polarization*: `ratio_out == ratio_ref` to 1e-9 after §4.1 and §4.7.
- *Entry boundary*: `S_applied == S_exact` to 1e-9 after §4.3; the lossless control already coincides
  on all three columns.
- *Series collapse*: `|S|` and `arg(S)` independent of `mtl_prev` to 1e-9 after §4.4.
- *Excess-phase walk-off*: matching the chosen convention to 0.01° after §4.5.
- *Grazing clamp*: large face within 0.03 dB of analytic to 89.9° (already true); small face departing
  only past its printed threshold.
- New *reciprocity* case: glued entry and glued exit equal to 1e-9.
- Every existing test with no `thk` column unchanged, except those exercising a resolved slab, whose
  expected movement under §4.1 and §4.3 is recorded before the change lands.

**Promote the three already-satisfied assertions now.** They cost nothing, they guard behavior that
currently works, and §4.1 is the change most likely to break one of them by accident.
