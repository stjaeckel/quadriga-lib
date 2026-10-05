# Sheet / Solid Divergence — Issue Inventory and Fixes

**Status:** measured against the first working sheet implementation. Nothing in this list is fixed yet.
**Scope:** every mechanism that makes a thin sheet and a traced solid of the same thickness disagree.
**Companions:** `sheet_refactor_ray_mesh_interact_v1.md` (classification, slab physics, §4 defect),
`sheet_refactor_ray_state_update_v2.md` (state machine, encoding, near/far selection).

Severity below ranks the **solid-vs-sheet disagreement**, not the error against physics. Some items are
sizeable errors that cancel between the two paths, which is a different problem and is ranked
accordingly.

---

## 1. Measurement basis

`test_sheet_vs_solid`, ITU-R P.2040 concrete at 3.5 GHz: εr 5.24, σ = 0.0462·3.5^0.7822 = 0.1231 S/m,
ñ = 2.2933 − 0.1379j, d = 0.1 m. EM mode, lod 1. Transmitted power, dB.

| θ | 0° | 17° | 34° | 51° | 68° | 85° |
| --- | --- | --- | --- | --- | --- | --- |
| traced solid | −11.086 | −10.412 | −10.472 | −10.591 | −10.979 | −14.400 |
| thin sheet | −10.381 | −10.414 | −10.503 | −10.758 | −11.507 | −14.839 |
| exact analytic slab | −10.365 | −10.398 | −10.488 | −10.745 | −11.512 | −16.822 |
| solid − exact | −0.721 | −0.014 | +0.016 | +0.154 | +0.532 | **+2.422** |
| sheet − exact | −0.016 | −0.016 | −0.015 | −0.013 | +0.005 | **+1.983** |

An independent replication of `qd_slab_resolve`'s produce branch reproduces the sheet column to better
than 1e-4 dB at every angle. The implementation therefore does exactly what the specification says, and
the lod-1 aggregation contributes nothing for the sheet — a single face and a single interaction. Every
number in the sheet row is attributable to a named approximation, and the attribution is in §3.

The exact analytic reference is the closed-form Airy sum with complex angles throughout:
`T = t₁₂ t₂₃ φ / (1 − r₁₂ r₂₃ φ²)` with `φ = exp(−j k₀ ñ d cos θ̃₂)`.

---

## 2. Ranked inventory

| # | Issue | 34° | 51° | 68° | 85° | Lives in |
| :-: | --- | --- | --- | --- | --- | --- |
| 1 | Solid: three inconsistent boundary evaluations | 0.031 | 0.167 | 0.53 | 2.42 | `ray_state_update`, `qd_slab_resolve` correction |
| 2 | Sheet: far boundary at the projected angle | ~0 | 0.002 | 0.009 | 1.98 | `qd_slab_resolve` produce |
| 3 | Solid 0° anomaly | — | — | — | — | 0.72 dB at 0° only, cause unidentified |
| 4 | Sheet does not subtract the free-space phase reference | 0 | 0 | 0 | 0 | up to 180° in `xprmat` |
| 5 | In-slab cosine projection mismatch | 0.001 | 0.005 | 0.020 | 0.054 | `ray_mesh_interact` vs `qd_refract` |
| 6 | Grazing clamp divergence | 0 | 0 | 0 | 0 | unbounded near tangency |
| 7 | Interface gain | untested — `att = 0` in the fixture | | | | |
| 8 | Amplitude convention | ~0.001, common-mode | | | | 0.015 dB from exact for *both* |
| 9 | Exit position, reflection plane offset | geometry, not physics | | | | fixture design |

---

## 3. Attribution of the sheet's residual

Each column is the dB the sheet result moves when that one approximation is replaced by its exact form,
holding the others fixed.

| θ | far-boundary angle | `phi` | amplitude convention | total |
| --- | --- | --- | --- | --- |
| 0° | 0.000 | +0.000 | +0.016 | +0.016 |
| 17° | 0.000 | −0.000 | +0.016 | +0.016 |
| 34° | +0.000 | −0.001 | +0.016 | +0.015 |
| 51° | +0.002 | −0.004 | +0.015 | +0.013 |
| 68° | −0.009 | −0.008 | +0.013 | −0.005 |
| 85° | **−1.975** | −0.018 | +0.010 | −1.983 |

Two conclusions worth carrying forward:

- **The whole 85° error is the far boundary.** Replacing it alone takes 85° from −14.839 to −16.814
  against an exact −16.822, and moves no other angle by more than 0.002 dB.
- **The `phi` hybrid is a non-issue for this material.** The companion doc's §5.2 flagged that a real
  path length times a per-meter loss might beat neither alternative. It is worth at most 0.018 dB, at
  85°, where the far boundary swamps it. Do not spend effort on the exact
  `exp(−j k₀ ñ thk / cos θ̃)` form on the strength of that concern alone.

---

## 4. The issues

### 4.1 Solid: one slab crossing evaluated at three angles that do not agree

**Severity: highest, and across the widest angular range.** 0.031 dB at 34°, 0.167 at 51°, 0.53 at 68°,
2.42 at 85°.

Entry is evaluated exactly, at the real incidence angle with the complex refraction cosine kept. Exit is
evaluated at the traced real in-slab angle. The Airy denominator re-derives the *entry* boundary at that
same traced angle, from the slab side. For an `air | slab | air` solid `mtl_prev == mtl_next`, so the
denominator is `1 − r_exit²·φ²` and the coefficient the tracer actually applied at entry appears nowhere
in the series.

The exit call is the crispest way to name the defect: `interact_with(slab, air, θ_t)` reconstructs the
cosine in air from a complex ratio and a real in-slab angle, producing a complex angle in a lossless
medium. That is not an approximation, it is impossible.

There is a fourth, coupled manifestation: `corr = 1/(Tn_avg·Tf_avg)` divides out an entry transmittance
computed at the traced angle while the accumulated amplitude carries the exact one.

Three coupled manifestations against the sheet's one is why the solid starts degrading at 34° while the
sheet holds to 68°.

**Fix.** Recover the exact entry angle by inverting the `qd_refract` map. The map is deterministic and
monotone, all its parameters are present at the exit event, and inverting *it* — rather than real-index
Snell, which is a different map and is what the earlier candidate 1 got wrong — round-trips to 1e-13 in
double and 1e-7 through a float32 `path_dir_prev`. Two Newton steps on
`η²u = t²·Re(√(1 − ε u))²`, `u = sin²θ_i`, starting from the real-index guess. No additional state and
no public signature change.

Then carry the complex in-slab cosine into the exit `interact_with`, into both boundaries of
`qd_slab_resolve` correction mode, and into the propagation factor.

**Cost and preconditions.** Correction mode needs a complex-angle path, and so does the exit interface
call. This is the largest change in the list and it moves every existing lossy-oblique solid result, so
record baselines first. Gate on: parallel faces — note that `eps == 0` currently bypasses the
`prev_nonpar` wedge gate and must not for this; the direction having actually been bent, so skip for
`geometry_type 1` and under TIR, where `ct_traced == cos θ_i` directly; `path_dir_prev` present; and a
lossless entry medium, without which the recovery returns exactly the angle that was used, which was
itself already a projection.

### 4.2 Sheet: far boundary evaluated at the projected in-slab angle

**Severity: 1.98 dB at 85°, negligible below 68°.**

`fresnel_r` builds the exit interface from `abs_cos_theta_t`, the real projected in-slab cosine, while
`fresnel_r_near` already does the near boundary exactly from the incidence side. The asymmetry is
visible in case 1 itself: for `air | sheet | air` physics requires `r_near == r_far` exactly, and the
code produces two different numbers.

**Fix.** In produce mode derive `ct_s = √(1 − (η_n μ_n)/(η_s μ_s)·sin²θ_i)` from `abs_cos_theta` and use
it in the far-boundary Fresnel and in the denominator.

**Cost.** Roughly 30 lines, produce branch only. It cannot move any existing test, because produce mode
is unreachable without a `thk` column. Highest value per unit of risk in this list, and it has to land
early regardless: until the sheet is exact it cannot serve as the reference the solid is measured
against.

### 4.3 Solid 0° anomaly

**Severity: 0.72 dB at one angle; urgency higher than that suggests.**

The solid sits 0.72 dB below exact at normal incidence, where there is no complex in-slab angle and §4.1
therefore cannot apply. It is not a constant offset hiding elsewhere either: at 17° the solid is within
0.014 dB of exact. A step that large over 17° of incidence, when every underlying quantity varies
smoothly, is a branch taken only at exactly-normal incidence.

**Diagnosis, in order.** Run `lod 5` to separate seven-path aggregation from a defect in the solid path.
If it persists, dump `resolved_typeN` at 0° and 17°; differing bits name the branch. Candidates: the
`use_vbs` threshold (`|Δcos| > 1e-6`), `qd_polbasis`'s degeneracy guard where `eQ = U × N` collapses at
normal incidence, and the correction-mode survival gate.

This is third by decibels but first by urgency: until it is understood, the solid is not a trustworthy
reference at any angle.

### 4.4 The sheet does not subtract the free-space phase reference

**Severity: zero in gain, up to 180° of phase.**

`SCALE_BY_MEDIUM` gives the solid `k₀(n·d·cos θ_t − d_geo·cos²θ_i)`, the phase relative to the same
distance in free space. The sheet's `arg_phi` is the absolute in-slab phase, with no subtraction. Both
fixtures span the same 10 m, so FSPL cancels and the difference survives.

A sheet abstracting `thk` of material displaces that much air, which the tracer has already walked, so
the reference belongs on the sheet as well. Predicted offset `k₀·thk·cos θ`:

| θ | 0° | 17° | 34° | 51° | 68° | 85° |
| --- | --- | --- | --- | --- | --- | --- |
| rad | 7.336 | 7.015 | 6.081 | 4.616 | 2.748 | 0.639 |
| mod 2π | 60.3° | 41.9° | 348.4° | 264.5° | 157.4° | 36.6° |

**Fix.** The reference goes on the transmit port's single `φ` only. Never on `φ²` in the denominator,
where the round-trip phase must stay absolute, and not on the reflect port, since a reflection displaces
no air.

**Before asserting on the `xprmat` column.** At −10.4 dB an unnormalized diagonal `xprmat` has entries of
0.302, so the largest possible component-wise difference between two such matrices is 0.603. The
measured values are 0.966 at 17° and 1.985 at 51°. That only fits if `calc_diffraction_gain`
power-normalizes `xprmat` before returning it, which would mean the test's xpr column is not measuring
what it appears to. Confirm this first. Note also that the measured ordering does not track
`k₀·thk·cos θ` cleanly — 34° should be the smallest deviation and is not — so either the normalization
is confounding it or a second term is present.

### 4.5 In-slab cosine projection mismatch

**Severity: 0.0002 dB at 17° rising to 0.054 dB at 85°.**

The sheet takes `|Re(ct2)|`. The solid's traced direction gives
`Re(ct2)/√(η²sin²θ + Re(ct2)²)`, because `qd_refract` normalizes a vector built from `Re(cos_theta2)` and
that normalization is not angle-preserving.

| θ | 0° | 17° | 34° | 51° | 68° | 85° |
| --- | --- | --- | --- | --- | --- | --- |
| dB | 0.000 | 0.000 | 0.001 | 0.005 | 0.020 | 0.054 |

Small, but it is pure disagreement with no physics behind it. **Do not fix it separately** — §4.1's fix
makes both paths carry the exact complex cosine and the mismatch disappears. Fix it standalone only if
§4.1 slips.

### 4.6 Grazing clamp divergence

**Severity: zero in this fixture, unbounded near tangency.**

`ray_mesh_interact` bounds the lateral walk-off by the longest FBS triangle edge. `ray_state_update`'s
redo has no access to that edge and uses a fixed `sheet_extent_ratio = 1000` instead. Neither clamp
fires here — the quad's longest edge is hundreds of metres against a 0.1 m thickness.

A consequence of the redo worth recording: for any run with medium tracking enabled, the clamp that
reaches the output is `ray_state_update`'s. `face_extent` now governs only the untracked-state path.

**Fix options.** Pass the face extent through if an output slot becomes available; or accept the
divergence, document that the two functions differ only inside the clamp region, and pick the ratio
deliberately rather than by default.

### 4.7 Interface gain

**Severity: untested.** The fixture has `att = 0`, so this cannot be seen at all.

The solid pays the entered medium's `interface_gain` once, folded in by `interact_with` on entry and not
on exit. The sheet pays `√(slab.interface_gain × far)` as an amplitude, which is once in power. Those
should match, but that is an argument, not a measurement.

**Fix.** Add a fixture with `att > 0` on both the sheet and the solid material. It is a decibel-scale
term if it is wrong.

### 4.8 Amplitude convention

**Severity: ~0.001 dB of disagreement. Do not fix.**

Both paths sit 0.015 dB off exact at 17–34°, in the same direction, because both use the
energy-conserving `|t| = √(1 − R)` form with the averaged reflectance rather than the true Fresnel
amplitude product. It is exact for a lossless interface and worth about 0.4 % in power for this one.

Common-mode, therefore not a source of solid-vs-sheet disagreement — but it is the floor on how close
the two can ever get, and on how close either gets to physics. If it is ever revisited, the `ISSUE`
comments in `interact_with` and `qd_slab_resolve` are correct: fixing either alone double-corrects.

### 4.9 Geometry

Not fixable in code; design around them.

- **Lateral exit offset.** A traced solid in `geometry_type 2` refracts in and out and displaces the exit
  point by `thk·tan θ_slab`; the sheet does not. Run the primary comparison in `geometry_type 1`.
- **Reflection plane position.** The box faces sit at ±0.05 m while the sheet sits at 0, a real path
  difference of `2·0.05·cos θ` for any reflection comparison.

---

## 5. Agreement does not require exactness

The two paths agree when they receive the *same* treatment, which is a weaker condition than both being
exact. Two coherent targets:

- **Both exact.** Complex angle throughout, on both sides. §4.1 plus §4.2.
- **Both near-exact, far-projected.** The solid gains the reconstructed entry angle and evaluates its
  entry boundary from the incidence side, matching the sheet's current structure. Much cheaper; the two
  would agree to roughly 0.001 dB while both remained about 2 dB off at 85°.

The first target is the right one, because §4.2 is nearly free and buys exactness outright. But the two
sides must move together: fixing §4.2 alone reopens a 2 dB gap at 85° from the other direction, since
the sheet would become exact while the solid stayed projected.

---

## 6. Order of work

Severity order and implementation order differ. Implement in this order:

1. **Diagnose §4.3.** `lod 5` run plus a `resolved_typeN` dump at 0° and 17°. Everything downstream is
   measured against the solid, so its trustworthiness is settled first.
2. **§4.2, the sheet far boundary.** Cheap, provably inert for existing tests, and it turns the sheet
   into the exact reference the rest of the work is measured against.
3. **§4.4, the phase convention.** Decide it before the `xprmat` column asserts anything, and confirm
   whether `calc_diffraction_gain` normalizes that matrix.
4. **§4.1, with baselines recorded first.** §4.5 folds into it.
5. **§4.6 and §4.7** as a decision and a fixture. Not urgent.

---

## 7. Acceptance

- Lossless slab: sheet and solid agree to machine precision at every angle. Already holds; keep it as a
  regression.
- Lossy slab after steps 1–4: gain agreement at the 0.001 dB level across the sweep, bounded by the
  common-mode amplitude convention of §4.8.
- Both paths within 0.02 dB of the exact analytic slab across the sweep, which is the same bound from the
  same cause.
- `xprmat` agreement once the phase convention is settled and the normalization question is answered.
- Every existing test with no `thk` column unchanged, except the lossy-oblique solids that §4.1 moves by
  design, whose expected movement is recorded before the change lands.

If §4.3 turns out to be an aggregation artifact rather than a defect in the solid path, the whole picture
reduces to one defect class in two places, and steps 2 and 4 close it.
