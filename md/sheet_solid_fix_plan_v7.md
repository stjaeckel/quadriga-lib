# Sheet / Solid — Fix Plan v7

**Scope:** the thin-sheet composite and the traced-solid slab correction in
`ray_mesh_interact.cpp`, covering both public functions in that file.

**Status:** the test suite is green, 1128355 assertions in 884 test cases. Every item below is
either measured, with the diagnostic table that measures it named, or explicitly marked as having no
fixture yet.

---

## 1. Where things stand

Two functions share one slab solver, `qd_slab_resolve`, in two modes.

**Produce mode** builds a thin-sheet composite outright. A sheet has zero world-space thickness, so
there is nothing for the tracer to re-emit into and the internal series has to be summed in one
event. It returns absolute transmit and reflect coefficients.

**Correction mode** adjusts coefficients the tracer has already written for a real traversed solid.
It returns the per-channel series factor `S = 1/(1 − rn·rf·φ²)` and nothing else, because the port
the caller holds already carries the Fresnel coefficient of its own interface.

`ray_state_update` re-resolves every sheet crossing when medium tracking is on, so in any run with
state its composite is the one that reaches the output and whatever `ray_mesh_interact` produced for
that crossing is discarded. `ray_mesh_interact`'s own composite serves callers without state and
must agree with the redo; it currently does.

### The energy convention

Every interface coefficient is a per-channel normal-flux amplitude: the raw Fresnel amplitude times
the square root of its own obliquity factor, `Re(z₂c₂)/Re(z₁c₁)` for TE and `Re(z̄₂c₂)/Re(z̄₁c₁)` for
TM, with `z = sqrt(ε/μ)`. `Material::flux_amp` forms them. It takes a complex incidence cosine, so
the two denominators separate whenever the incidence side is itself a medium entered at an angle;
for a real cosine both collapse to `Re(z₁)·cos θ` and only the numerators differ.

Consequences worth holding on to:

- For a lossless incidence medium the per-channel transmittance is identically `1 − R_X`, at any
  angle and for any complex second medium. For a lossy one it is not, and it is not bounded by 1.
- All four transmission-class types (1, 2, 4, 5) report the same energy for the same interface. The
  outgoing ray tube carries no power of its own, so the coefficient owes the whole factor.
- Under total reflection there is no forward Fresnel port; the `tf` leak carries a flat
  `sqrt(1 − R_eff)` coefficient instead.
- Reflection stays within one medium, where the factor is 1, and is untouched.
- Consecutive crossings that share a medium cancel their obliquity factors exactly, but only if both
  are evaluated at the same complex cosine in that medium. That condition holds across a traced
  solid. It does not hold across a sheet reached from inside a medium, which is issue 2.

### The glued-sheet flux conversion

Produce mode applies the near-to-far obliquity factor once, on the transmit port, so a glued sheet
reports the same energy as the equivalent traced entry. The gate is `!mat_far.same_as(mat_near)`, so
only the glued cases reach it: `sheet_case 2` (entry, `air | sheet | solid`) and `sheet_case 3`
(exit, `solid | sheet | air`). An isolated sheet, a collapsed duplicate and a sheet embedded in a
medium all have `near == far` and take the factor as exactly 1.

The composite is the three-medium Airy pair in raw amplitudes, times that conversion once, divided by
the propagation of the medium the sheet displaces. The numerator of the conversion is evaluated at
the cosine in the far medium, written from the near-side sine directly rather than by chaining
through the sheet; Snell is transitive, so the sheet drops out of it and the far cosine is the same
quantity a bare near|far crossing would have used. The denominator is at the incidence cosine in the
near medium. The reflect port turns around in the near medium, where the ratio is 1, and carries
nothing.

Measured, the factor is worth +3.60 dB at normal incidence and +13.75 dB at 85° for an entry into
`eps 5.24 − 0.63j`, and −0.88 dB to −3.57 dB over 0° to 50° for an exit out of `eps 1.50`. Its TE/TM
split reaches 0.235 dB at 85° for a lossy far medium and is identically zero wherever the far medium
is lossless.

### The in-medium angle

A traced ray must have a real direction, so `qd_refract` normalizes a vector built from `Re(cos θ₂)`
and the imaginary part is dropped. The map it applies is

    ct_traced = Re(c₂) / |η·u + (η·cos θᵢ − Re(c₂))·n|

and it is deterministic and monotone. Real-index Snell is a different map, and inverting that
instead leaves a residual worst exactly where the defect is worst, because it ignores the
normalization. `EXACT_IN_SLAB_COSINE` inverts the map itself: three Newton steps on
`η²u = t²·Re(sqrt(1 − ε_r u))²` with `u = sin²θᵢ`, starting from the real-index guess. It reaches
the exact complex cosine to machine precision. Permeability is carried: the two ends of `ε_r` are
`ε·μ` products, so `η²` is the same `snell²` that `qd_refract` used.

It declines, returning zero, when: no real direction was supplied or the traced and geometric
cosines agree to 1e-6; the current medium is not a real material; the previous face is not parallel;
the crossing is under total reflection; or the entry and slab media are matched. Where it declines,
every consumer keeps the real projection and nothing moves.

Consumers, all on that one gate: both boundaries of the Airy denominator, the slab propagation
factor inside `qd_slab_resolve`, the interface coefficient in `REPLACE_BY_VBS`, and the in-medium
traversal factor in `SCALE_BY_MEDIUM`.

### The propagation convention

Where a recovered cosine exists and the tracer traced the bend, the traversal factor is
`exp(−j k₀ (n·cs·d + q·s))`, with `d` the normal thickness and `s` the transverse displacement, both
pure geometry off the traced direction, and `q² = n²(1 − cs²)` the transverse wavenumber Snell
conserves from the entry medium. Conductivity lives inside `n`; only the alpha column is drawn from
`medium_gain`, through `excess_gain`. For a real index the two terms recombine into `n·dist_refract`
exactly, so this reduces to the path form rather than replacing it.

Everywhere else the traversal keeps the ITU-R P.2040 rate along the path with the phase from the
real index. That is the right treatment where the tracer did not trace a bend: there is no
inhomogeneous wave to be exact about. The two places where it is still applied on a leg that DOES
have a recovered cosine are issues 8 and 9.

### The mass-law convention

A material with `m > 0` carries a lumped acoustic transmission surrogate rather than a bulk
propagation loss. `Material::medium_gain` forms it from two pieces that are supplied separately, a
surface-mass surrogate with the angle removed and an angle factor:

    A  = m · log10((f/fRef) · M)
    X₀ = ((f/fRef) · M)^(m/20)
    TL = A · log10(1 + (X₀·cos θ)²) / log10(1 + X₀²)

`A` is the normal-incidence term, the value the material table is calibrated on, and it is returned
unchanged whenever the angle factor is one. The angle carries the mass law's own shape normalized to
it, so both endpoints are exact: `A` at normal incidence, zero at tangency. The asymptotic angle
slope is 20 dB per decade of `cos θ` and does not follow `m`, because the cosine enters through the
fluid's wave admittance rather than through the panel's transfer impedance. The normalization factor
is within 5e-4 of ten for any panel with meaningful loss, so this is the exact single-panel mass law
wherever the term is worth anything.

Callers supply the two pieces as follows. A sheet passes the normal thickness and the geometric
incidence cosine, so the quotient `thk / cos` is never formed. A traced solid passes neither and
gets `dist_geo·cos θ` and `cos θ` by default, whose product is the `dist_geo·cos²θ` the term was
written in before; across a slab traversal `dist_geo` is the normal thickness over the cosine, so
the surrogate is the normal thickness and the two callers form the same argument. The public
`quadriga_lib::medium_gain` entry point, and `calc_diffraction_gain` through it, supply no angle and
therefore read `A` unchanged.

`m > 0` also excludes a slab from correction-mode resolution entirely, and from the exact
in-medium propagation branch in `SCALE_BY_MEDIUM`: the eta/mu of such a material are calibrated for
surface impedance only, so the in-slab index is not physical and the internal series must not be
resummed.

### What is verified

Each of these has a promoted assertion, not just a printed table.

| property | bound | measured |
| --- | --- | --- |
| Per-channel normal-flux transmittance, types 1/2/4/5, both directions, 4 materials × 6 angles | 1e-9 | exact, 192 rows |
| Per-channel split at a single interface, reflection and refraction | 1e-12 | 7.8e-16 worst |
| Thin sheet composite against the exact analytic slab, 5 materials × 8 angles | 1e-6 dB | 0.000000 dB |
| Sheet phase offset equals `k₀·thk·cos θᵢ` | 1e-3 deg | exact |
| Grazing incidence, both ports, independent of the face, to 89.999° | 1e-6 dB | exact |
| Mesh subdivision invariance, 2 triangles against 2·65² | 1e-9 relative | 0.000e+00 |
| Entry boundary built from the recovered complex angle | 1e-9 | exact |
| Exit port at the recovered angle, scalar coefficient and EM gain, 4 materials × 6 angles | 1e-9 | 3.8e-15, 4.6e-15 |
| Traced air \| slab \| air chain against the analytic slab, 4 materials × 6 angles | 1e-6 dB | 0.000000 dB |
| `qd_refract`'s traced cosine against the map the recovery inverts | 1e-9 | 1.1e-16 |
| Host layer carried through an impedance-matched sheet | 1e-9 | gain 1.000000 |
| In-slab cosine projection, sheet form against traced form | 1e-3 relative | 7.5e-4 |
| Glued sheet composite, entry and exit, per channel, 4 stacks × up to 10 angles | 1e-9 | 1.67e-15 |
| Glued sheet trivial branch, isolated and embedded, conversion identically 1 | exact | exact |
| Glued sheet reflect port against a reference carrying no conversion | 1e-9 | 8.33e-16 |
| Mass-law sheet composite and mass term, swept to 89.9°, 15 angles | 1e-6 dB | 1e-14 dB |
| Mass-law angle slope, 30° to 60° | (19.5, 20.0) | 19.842911 |
| Mass-law term at normal incidence against the superseded form | 1e-9 | exact |
| Mass-law reflect port equals the bare near interface, and is independent of `m` | 1e-9, 1e-12 | exact |

Two further properties were established offline and have no fixture. The real-angle path through
`Material::interact_with` is bit-identical to its pre-change form over 140544 comparisons across four
materials, six `tf`/`att` cases, six interaction types and 244 angles. And the entry and exit
obliquity factors across a traced solid multiply to 1.000000000000 in both channels at every angle
tested, which is the telescoping property the chain fixture measures end to end.

### Diagnostic infrastructure

`QD_TEST_VERBOSE` in each test file switches the tables on. It is currently defined in both
`test_ray_mesh_interact.cpp` and `test_ray_state_update.cpp`. The assertions run either way, with
the two exceptions named in issue 6.

The independent oracles are `qd_slab_ref`, `qd_slab_ref_R`, `qd_iface_ref`, `qd_mass_ref` and
`qd_mass_TL` in `test_ray_mesh_interact.cpp`, and `diag_slab_p_pol` with `diag_slab_p_avg`
delegating to it in `test_ray_state_update.cpp`. All evaluate closed forms with complex angles
throughout and share no code with the library. `qd_iface_ref` also carries the per-channel obliquity
and flux fields (`obl_te`, `obl_tm`, `F_te`, `F_tm`, `F_avg`). `qd_mass_TL_clipped` reproduces the
superseded mass-law form and exists so the diagnostic table records what that change was worth.

`test_ray_state_update.cpp` additionally carries a set of reference builders for the traced-solid
leg, each reproducing one stage of what the function applies: `diag_port` (an interface at a real or
complex incidence cosine, with the per-channel flux factor), `diag_series` (the correction-mode
Airy factor at one cosine), `diag_medium_vbs` and `diag_medium_geo` (the two traversal forms),
`diag_traced_ct` and `diag_ct_exact` (the `qd_refract` map and the value it projects),
`diag_r_slabside`, `diag_medium_gain`, and `diag_quad_x` / `diag_ray_at` for the mesh calls.

`diag_sheet_composite`, returning `DiagSheet`, is the produce-mode reference: the full three-medium
Airy pair, both ports, per channel, with the flux conversion on the transmit port only and the
displaced-host propagation divided out. It takes the three permittivities, the normal thickness, a
real incidence cosine and the displacement side. It was checked against the library solver offline,
in a standalone translation unit carrying `Material` and `qd_slab_resolve` and nothing else, over six
stacks and ten angles: worst 2.37e-16 on both ports and both channels. It currently assumes `μ = 1`
and no permittivity resonance, which is issue 4.

---

## 2. Ranked issues

Ranked by importance. Section 6 gives the execution order, which differs where one item needs
another's reference or answer to proceed.

| # | Issue | Measured size | Where |
| :-: | --- | --- | --- |
| 1 | No transfer-matrix core: layered stacks are approximated or dropped | up to 0.99 dB; `sheet_case 4` kills the ray | `SHEET_EVENT`, `SLAB_AIRY_FACTOR` |
| 2 | A sheet reached from inside a medium is resolved at the projected angle | unmeasured | `SHEET_RESOLVE`, `qd_slab_resolve` produce mode |
| 3 | Produce mode's lumped-column branches have no fixture | unmeasured | `qd_slab_resolve` produce mode |
| 4 | Permeability never reaches the sheet or slab path | unmeasured | `qd_slab_resolve`, `flux_amp`, both callers |
| 5 | Glued entry / exit reciprocity residual | 0.074 dB, measured offline only | not isolated |
| 6 | Two diagnostics assert nothing numeric | no regression guard | `test_ray_state_update.cpp` |
| 7 | The corrected refraction direction at an M2M transition has no fixture | unmeasured | `REPLACE_BY_VBS` |
| 8 | The walk-off branch keeps the projected cosine while the series does not | unmeasured; no fixture that can fail | `SCALE_BY_MEDIUM` |
| 9 | Hybrid propagation factor where no cosine is recovered | 2.5e-5 dB | `qd_slab_resolve` |
| 10 | Four produce-mode branches are never entered by any test | unmeasured | `qd_slab_resolve`, `SHEET_EVENT` |

---

## 3. The issues

### 3.1 No transfer-matrix core

**Evidence, three separate symptoms.**

`sheet_case == 4`, an ordered stack, calls `KILL_RAY()` in `SHEET_EVENT`. Detection and the type bits
are correct but no sheet physics runs, so the ray disappears rather than being approximated.

Diagnostic *Series collapse with a matched sheet as the previous medium*,
`test_ray_state_update.cpp`. When a sheet is the bounding medium of a host layer, correction-mode
Fresnel treats it as a half-space. A matched pair gives `r = 0` and the Airy denominator collapses to
exactly 1, costing the host layer its internal series: `|S|` reads 1.000000 in every such row against
a reference that oscillates with the fringe, worth 0.988 dB at 50° over 50 mm. The exactly-matched
case is caught by the `merge_host` path in `SHEET_EVENT`, which carries the layer through instead of
cutting it; a sheet whose impedance merely happens to be close to the host's is not caught and still
collapses.

Cross-layer coupling in a genuine two-layer stack is absent in all cases. A coated window is modelled
as two independent interfaces plus a sheet composite, not as one stack.

**Target.** A 2×2 characteristic-matrix core resolving the whole stack in one pass. Exact for any
layer count, reduces to the Airy formula at one layer, produces R and T together, handles the
asymmetric glued case, and closes `sheet_case 4`. It works natively in amplitudes with a single
conversion to normal flux at the ends, which is where the energy convention in §1 is expressed most
directly.

**Scope.** The largest item here and its own work stream rather than a patch. It absorbs issue 9, and
it should be sequenced after issue 2, since the incidence cosine the core is written in is the same
question issue 2 answers. `diag_sheet_composite` becomes the single-layer regression oracle the core
has to reproduce.

**Assertions.** `sheet_case 4` resolves rather than killing. `|S|` and `arg(S)` independent of
`mtl_prev` to 1e-9 for a near-matched sheet as well as an exactly matched one. A two-layer stack
against a hand-built characteristic-matrix reference to 1e-9. Every existing single-layer bound in
the §1 table unchanged.

### 3.2 A sheet reached from inside a medium is resolved at the projected angle

**Evidence.** `SHEET_RESOLVE` calls `interact_with(M_near, M_far, ..., theta_t, ...)` for the outer
pair's geometry and total-reflection verdict, and hands `qd_slab_resolve` produce mode
`ct_geo = abs_cos_theta_t` as the incidence cosine. Produce mode then derives its own in-slab cosine
and its near-to-far flux conversion from that value, on the assumption that it is a real incidence
angle in `mat_near`.

That assumption holds when the ray arrives from air. It fails when `mat_near` is a traced solid: the
propagation cosine there is complex, `abs_cos_theta_t` is only its real projection, and the whole
composite is then built at a cosine the path never carried. It is the same defect that was closed at
the far face of a traced solid, in the one path that does not go through `REPLACE_BY_VBS` —
`SHEET_EVENT` is dispatched instead of `APPLY_TRANSITION`, so the correction never runs.

**Size.** Unmeasured, and measuring it is the first thing this item needs. Two effects to separate.
The composite amplitude itself moves with the incidence cosine. Separately, the flux conversion is
formed from the real cosine, so for a glued pair it does not cancel against the entry that produced
it. Where `near == far` that factor is identically 1 and only the first effect remains, which bounds
the embedded-sheet case well below the glued one.

**What is already in place.** `diag_sheet_composite` takes `e_near` as an argument, and the glued
fixture already runs one row with a non-air near medium (a sheet embedded in lossless `eps 1.50`).
The reference machinery is therefore proven; what this item changes is the cosine that reference is
evaluated at, not its form. The fixture as it stands cannot fail for this reason: it builds its
reference at the same cosine the code is handed, so it is a consistency check on the factor's form
and not on its argument.

**Why it cannot simply reuse the existing gate.** `EXACT_IN_SLAB_COSINE` recovers the cosine in
`mtl_current` using `mtl_prev` as the entry medium. `SHEET_EVENT` writes the sheet material into the
previous-medium word, so on the leg after a sheet crossing `mtl_prev` is the sheet and the entry
medium is one level further back than the recovery can see. The gate has to be re-derived for this
path, or the recovery has to be told which medium the leg was entered from.

**Change.** Measure first. Then, if the size warrants it, give the sheet path the same recovered
cosine, under a gate that names its own entry medium correctly.

**Assertion.** A sheet embedded in a lossy solid and a sheet glued to the far face of one, both
compared against `diag_sheet_composite` evaluated at the exact in-medium cosine, to 1e-9. Both cases
currently reach the code and neither is measured at the right cosine.

### 3.3 Produce mode's lumped-column branches have no fixture

Three branches inside `qd_slab_resolve` produce mode are entered by no test, because every palette
that reaches a sheet sets the relevant columns to zero.

The `AMP` lambda rescales each interface amplitude by `sqrt((1 − R_tf)/(1 − R0))` to carry the `tf`
redistribution. With `tf = 0` that quotient is exactly 1 and the branch is inert, which is the only
case any sheet test exercises. `tf` is set in exactly two test cases in the suite, both on the
correction-mode path.

`ig = sqrt(slab.interface_gain · (far_is_new ? far.interface_gain : 1))` folds the lumped penetration
loss into the transmit port. The `far_is_new` test — `!far_is_air && !mat_far.same_as(mat_near)` —
was written to stop a sheet embedded in a medium from paying that medium's entry loss twice. Nothing
measures it. Real material tables carry `att`, and the glued pair is the motivating geometry for it,
so this is the branch of the three most likely to reach a user.

`eta_resonance` is added to every interface permittivity in produce mode and deliberately excluded
from the propagation index, on the reasoning that a permittivity resonance is an interface effect
rather than a propagation one. That split appears at five separate sites in the solver. One test case
in the suite sets `resF`, and it is not a sheet.

**Change.** Fixture only, unless it fails. Extend the glued fixture with an `att`-bearing sheet and
an `att`-bearing solid, a `tf`-bearing pair on both ports, and a resonant sheet, each against
`diag_sheet_composite` extended to carry the same three columns.

**Assertions.** A glued entry through an `att`-bearing sheet into an `att`-bearing solid pays each
interface gain exactly once, to 1e-9, and an embedded sheet pays the host's not at all. A `tf` sweep
on both ports against the closed form to 1e-9. A resonant sheet whose interface terms move with
`resS` while its propagation factor does not, to 1e-9.

### 3.4 Permeability never reaches the sheet or slab path

The `e`, `f`, `g`, `h` columns are in the material model and in every expression that matters:
`z = sqrt(ε/μ)` in `flux_amp` and in both `fresnel_r` helpers, `n = sqrt(ε·μ)` in the propagation
factor, `(ε₁μ₁)/(ε₂μ₂)` in every refraction root and in the total-reflection test, and the `ε·μ`
products at both ends of `EXACT_IN_SLAB_COSINE`. ITU-R P.2040 does not specify them and no shipped
material uses them, but the logic is live and unguarded.

Permeability is exercised by six test cases, all of them single interfaces or a matched Airy factor:
the backward-compatibility default, the reflection/refraction decoupling, the impedance-matched
dense medium, the `ε↔μ` swap invariant, `medium_gain` under magnetic loss, and the public helpers.
None of them involves a sheet, a glued pair, a traced-solid chain, or the flux conversion.

Two things separate the magnetic case from the dielectric one and neither is covered. A complex `μ`
from the `g` column makes `z` complex even for a real `ε`, so `Re(z c)` and `Re(z̄ c)` split in
`flux_amp` where the dielectric case would not. And `ε↔μ` swaps leave the index `n` and therefore the
propagation factor unchanged while inverting the admittance, so a swapped pair is a strong
self-check: the slab phase must not move while both ports do.

`diag_sheet_composite` currently takes permittivities only and forms `z` and `n` both as `sqrt(ε)`.
It has to be generalized to `(ε, μ)` pairs before this item can be measured at all.

**Change.** Generalize the reference, then a fixture. No source change is expected; this is coverage
for logic that shipped without any.

**Assertions.** A glued entry and a glued exit through a magnetic sheet against the generalized
reference to 1e-9. An `ε↔μ` swap on the sheet leaving the composite phase unchanged to 1e-9 while the
per-channel powers move. A magnetic-loss sheet whose `fx_te` and `fx_tm` differ with a lossless far
medium, which the dielectric case cannot produce. A traced solid with `μ ≠ 1` against the analytic
slab to 1e-6 dB, so the recovery and the exit port are covered as well as the sheet.

### 3.5 Glued entry / exit reciprocity residual

The same three-medium stack resolved from both directions must give the same answer. Measured offline
against the library solver, at matched cosines:

- Lossless solid: the two orientations agree to 0.000000 dB at every angle from 0° to 50°, and the
  product of the two flux conversions is exactly 1. Removing the conversion would leave the two
  disagreeing by 1.76 dB at normal incidence rising to 3.44 dB at 50°.
- Lossy solid, `eps 1.50 − 0.40j`: the composite ratio is −0.0739 dB at normal incidence, crosses
  zero near 38° of in-air incidence, and reaches +0.0743 dB at 50°.

The attribution is partly settled. At normal incidence there is no projection error and the product
of the two conversions is exactly 1, so the whole of the −0.0739 dB there belongs to the flux
splitting of §1, which is not exact for a lossy medium and is a settled convention. Above normal
incidence the conversion's own product drifts, from 0.003 dB at 10° to 0.231 dB at 50°, because the
exit is handed a real projection of a complex cosine — which is issue 2, not this one. The two
contributions have opposite sign over most of the range, which is why the total passes through zero.

There is no fixture. `grep -i reciproc` finds only comment text.

**Change.** Add the fixture, then re-measure after issue 2 lands. The lossless case is the one that
must read exactly zero and is the useful regression guard; the lossy case records a residual that
issue 2 is expected to shrink and that the flux convention bounds from below.

**Assertion.** Glued entry and glued exit equal to 1e-9 for a lossless solid. For a lossy solid, the
residual recorded against a stated bound rather than asserted to zero, with the normal-incidence row
called out separately because no cosine question enters there.

### 3.6 Two diagnostics assert nothing numeric

*Series collapse with a matched sheet as the previous medium* and *Excess-phase walk-off convention*
both print a table and then assert only `isfinite` and `gain >= 0`. With `QD_TEST_VERBOSE` off they
are no-ops for regression purposes: a refactor could move either quantity arbitrarily and the suite
would stay green.

Every other diagnostic in both files carries at least one numeric bound. The walk-off case in
particular pins its form to 0.000° in the printed table and asserts nothing about it, which is the
larger of the two gaps because that phase form is itself under discussion in issue 8.

**Change.** Promote the printed quantity to an assertion in both. For the walk-off case that is the
applied phase against `−k₀(n−1)·L·cos²θ_fbs` to 0.01°. For the collapse case it is `|S| == 1.000000`
to 1e-9 with the matched sheet as the previous medium, which records the defect rather than hiding
it, and must be inverted when issue 1 lands.

### 3.7 The corrected refraction direction at an M2M transition has no fixture

`REPLACE_BY_VBS` passes the recovered cosine into the interface call, and the refraction cosine it
returns moves with it. That value is the cosine the outer pair of media gives, which is the correct
one, and only `geom == 2` reads it — an exit keeps the traced direction and a mirror does not use it.
So a medium-to-medium transition out of a lossy traced solid now writes a slightly different
continuation direction than it did.

Nothing measures this. The one test that asserts a VBS-corrected direction uses a lossless palette
where the recovery is a provable no-op, and the two diagnostic cases that supply `path_dir` are both
i-o exits. The change is a one-line consequence of the interface fix and is believed correct on
Snell transitivity, but it is unasserted.

**Assertion.** An M2M transition from a lossy solid into a second medium, with `path_dir_prev`
supplied, whose written `path_dirN` matches a hand-built direction from the transitive-Snell cosine
to 1e-9.

### 3.8 The walk-off branch keeps the projected cosine while the series does not

`SCALE_BY_MEDIUM` selects between two traversal forms on `walk_off = !use_vbs || refl_pass ||
resolved`. Where that is false the exact form of §1 applies. Where it is true the ITU-R P.2040 path
form applies with the `cos²` walk-off substituting for the untraced lateral shift.

Two of the three conditions are `refl_pass` and `resolved`, and both can be true on a leg where the
tracer did trace the bend and a recovered cosine exists. On such a leg `SLAB_AIRY_FACTOR` builds its
series at the recovered cosine while the traversal that multiplies it is built at the projection.
That is an internal inconsistency of the same kind that was removed from the exit interface, though
it is bounded by the same class of numbers as issue 9 rather than by the multi-dB obliquity gap,
because a reflection stays in one medium.

Separately, the fixture problem is unchanged. The *Excess-phase walk-off convention* diagnostic
exercises the branch but cannot distinguish the two-cosine form from a single-cosine one: it supplies
no `path_dir`, so `use_vbs` is false, `dist_refract == dist_geo`, `θ_t == θ_i`, and the two cosines
coincide. Issue 6 covers promoting what it does measure; this item covers making it measure more.

**Change.** A fixture with a VBS correction actually applied, so that `dist_refract` and `dist_geo`
differ and the two cosines separate. Then measure an internal reflection inside a lossy parallel
slab with `path_dir` supplied, and decide whether the exact form should extend to that branch.

**Assertion.** Applied phase matching the chosen form to 0.01°, both branches exercised; and, if the
exact form extends, an internal reflection leg agreeing with the closed form to 1e-9.

### 3.9 Hybrid propagation factor where no cosine is recovered

When no exact in-slab cosine is available, `qd_slab_resolve` correction mode takes the magnitude of
the propagation factor from `medium_gain` over the traced path and the phase from the real index and
the real projected cosine, while produce mode takes both from `exp(−j k₀ n L cos θ)` with the index
and the cosine complex. The two differ by the gap between the ITU-R P.2040 `Delta` form and
`exp(−k₀·Im(n)·L)`, which measures 2.5e-5 dB.

The branch fires only where the recovery declines, and there the traced cosine is a real geometric
incidence cosine rather than an in-medium propagation cosine, so treating the leg as an inhomogeneous
wave is not obviously justified. Below every other item here. Worth doing only as cleanup, or as a
side effect of issue 1.

### 3.10 Four produce-mode branches are never entered by any test

Each is a real branch with a real guard and no coverage. None is suspected of being wrong; all four
are one fixture row each.

**The displacement budget cap.** `t_disp = min(thk, disp_budget)` limits how much of the host a sheet
may take back out, because taking a medium back out is a division and a sheet thicker than its host
would amplify. Every sheet in the suite is 6 mm inside a host of 100 mm or more, so the cap is never
reached and `t_disp == thk` on every row that exists. A laminate whose layers do not fit inside their
host is the case the guard was written for.

**Scalar mode on a glued sheet.** Produce mode collapses TM onto TE when `is_scalar`, in the two
Fresnel helpers and in the flux conversion. Every glued row in the suite is EM. The scalar sheet rows
that do exist are all `near == far`.

**Float `dtype`.** No sheet or slab test runs at `float`. The composite is written to `dtype` through
`xprmat`, and the whole sheet chain is asserted at `double` only.

**A corner with a sheet, `sheet_case 6`.** `ray_mesh_interact` drops SBS and handles FBS alone,
clearing the co-location and corner bits. `ray_state_update` re-derives the case from the material
indices and reaches it through the "no co-located second face" branch. Nothing tests the pair, and
the note in §5 about the wedge test being unusable for a glued entry is adjacent to it.

---

## 4. Closed, with guards in place

**Per-channel energy convention.** Every transmission-class interaction carries the per-channel
normal-flux transmittance of its own interface, and the four types agree with each other exactly.
Asserted at 1e-9 over 192 rows, four materials, both directions, six angles, with total reflection
handled separately as a closed port. The derivation and its consequences are in §1.

**Exit interface at the recovered in-slab cosine.** The far face of a traced solid is evaluated at
the exact complex cosine recovered from the traced direction, so the entry and exit obliquity factors
are formed at the same cosine in the shared medium and cancel identically. Worth up to 4.44 dB on the
coefficient at 85°, in the *Exit boundary* table's `d_port_dB` column, and exactly zero for a real
permittivity. Asserted at 1e-9 on the scalar coefficient and on the basis-invariant EM gain.

**In-medium traversal on refracted legs.** The traversal factor uses the normal-thickness decay of an
inhomogeneous wave rather than a path-length rate. This is what closes the end-to-end chain: without
it the traced solid sits 0.040 dB off the analytic slab at 85° for `eps 1.50 − 0.40j`, and with it
the residual is at the floating-point floor. Asserted through the chain fixture at 1e-6 dB.

**In-slab cosine projection.** The sheet takes the in-slab cosine as `|Re(cos θ₂)|`; the traced solid
gets it from `path_dirN`, which `qd_refract` produced by normalizing a vector built from the same
real part, and that normalization is not angle preserving. The two differ by 7.5e-4 relative worst
case, which is 0.083 mm of path over a 100 mm slab and under 0.01 dB. Quantified, negligible, guarded
at 1e-3.

**The glued-sheet near-to-far flux conversion.** Measured, and correct as shipped; no source change
was made. *Glued sheet — the near-to-far flux conversion* in `test_ray_state_update.cpp` runs six
stacks against `diag_sheet_composite`: an isolated sheet, a sheet embedded in a medium, two glued
entries swept to 85° and two glued exits swept to 50°. Per-channel powers agree to 1.67e-15 worst,
the phase is pinned complex on the rows where nothing else multiplies the composite, and the reflect
port agrees to 8.33e-16 against a reference carrying no conversion at all. The trivial branch is
asserted to take the factor as exactly 1, on the embedded stack as well as the air one, which pins
the gate as `near == far` rather than as "far is air". The discrimination is asserted too: the
conversion exceeds 0.5 dB on every glued row and the composite is checked to be inconsistent with a
reference built without it.

**Thin-sheet grazing behavior.** There is no grazing clamp in either function. A thin sheet is
modelled as a laterally infinite layer, consistent with the plane-wave Fresnel boundaries and the
plane-parallel series it is built from. The composite is exact against the analytic slab to 89.999°
in both ports, is independent of the face it was cut from, and is invariant under mesh subdivision.
Finite lateral extent is an edge-diffraction question and belongs with the ray tube if it is ever
taken up; it must not re-enter the layer's transfer function, where it made results depend on
triangulation.

The one quotient that can diverge, `thk / cos_slab`, is formed inside `qd_slab_resolve` and capped by
loss, 400 dB floored at the normal thickness, rather than by angle, so the cap depends on the
material and the frequency and on no geometry.

**Mass-law angle dependence.** The angle no longer sits inside the same logarithm as the frequency,
so the angle slope is 20 dB per decade of the cosine independent of the fitted `m`, and there is no
angle cutoff: the term tapers to zero rather than being clipped. The form and its rationale are in
§1. The old form read the fitted slope in the angle, which was 8.5 dB off at 85° for `m = 12`, and
cut the term off entirely past 86.8° to 89.5° for realistic panels, so a grazing ray crossed a panel
with no attenuation at all. Normal incidence is unchanged by construction, which keeps the public
`medium_gain` entry point and `calc_diffraction_gain` bit-identical and requires no table migration
and no new material column. Asserted across a 15-angle sweep to 89.9° at 1e-6 dB, measuring 1e-14,
with the slope, the knee, the taper past the knee, the normal-incidence identity and the reflect
port's independence from `m` each asserted separately.

---

## 5. Known limitations, recorded rather than open

`apply_tf` clamps `R0` into `[0, 1]` before forming an interface reflection magnitude. Past the
internal critical angle of the near|far pair, the slab-side TM Fresnel reflection of a lossy sheet
exceeds unity — measured 1.0445 at 60° for `eps 1.50 | sheet eps 5.24 − 0.63j | air` — and the clamp
caps it, leaving the produce-mode reflect port up to 4.0e-2 off the exact Fresnel value in that
region. The transmit port is identically zero there, so only the reflect port is affected, and only
above the critical angle of the outer pair. The glued fixture sweeps stop at 50° to keep this out of
the flux-conversion measurement.

`mtl_prev` is one level deep, so a `sheet | solid | sheet` sandwich resolves but a deeper nesting
falls back to air. This is also what blocks the recovery on the leg after a sheet crossing; see
issue 2.

`SHEET_EVENT` sets the fix-flag rather than guessing when it meets a deferred buffer
(`mtl_buffer != 0`), because no sheet code in scope carries one.

The wedge test is unusable for a glued entry, since it compares FBS and SBS, which are co-located by
construction. The code defaults to parallel there, which is right for a coating on a flat wall and
wrong for a sheet glued to a wedge.

`Material::interact_with` keeps two spellings of the same two expressions, one for a real angle and
one for a complex cosine: the refraction-root argument and the total-reflection test. They are the
same statement and the complex form is the general one. The real form is retained because
reassociating it moved coefficients by up to 6e-14 relative and, at a matched interface, lifted the
reflectance off exactly zero, which sent a `tf`-forced reflection down the rescale branch and flipped
the sign of a coefficient whose power was unchanged. Do not merge them without re-running the
bit-identity check described in §1.

The mass-law term is a hard zero below an argument of one rather than a taper, so its frequency
dependence has a kink where its angle dependence no longer does. Removing the clip needs a knee that
is independent of the normal-incidence offset, and the columns cannot express one because `fRef`
sets both: choosing `fRef` to place `TL(0)` also places the roll-off. Bounded by 3 dB and only
within a decade of the knee, where the term is itself under 3 dB. Fixing it costs a material column
and a table migration, which is not worth it at that size.

---

## 6. Order

Importance and execution order differ here, because the largest item depends on a smaller one's
answer and because two cheap items protect everything that follows.

1. **Issue 2.** Measure first, then decide. Structurally the same defect that was closed at the far
   face of a traced solid, in the one path that bypasses the correction. It has a reference now, and
   issue 1 needs its answer.
2. **Issue 6.** Two assertions. Do it before any refactor, not after.
3. **Issue 3.** Fixture only. Guards a branch that real material tables reach.
4. **Issue 4.** Generalize `diag_sheet_composite` to `(ε, μ)` pairs, then a fixture. No source change
   expected.
5. **Issue 5.** Fixture, then re-measure after issue 2 lands.
6. **Issue 7.** Test-only, and it pins a consequence of an earlier change that is currently
   unasserted.
7. **Issue 10.** Four fixture rows, independent of each other and of everything above.
8. **Issue 8.** Test-only for the phase form; the branch question rides on issue 9.
9. **Issue 1.** Its own work stream. Absorbs issue 9. Sequenced after issue 2, whose answer it needs.

## 7. Acceptance

Each assertion below replaces or joins the sanity checks in the case that measures it.

- *Sheet in a medium*: an embedded sheet and a glued exit sheet, both inside a lossy solid, against
  `diag_sheet_composite` at the exact in-medium cosine, to 1e-9.
- *Lumped columns*: `att` paid exactly once per new medium and not at all for an embedded sheet;
  a `tf` sweep on both ports; a resonant sheet whose interface terms move and whose propagation
  factor does not. Each to 1e-9.
- *Permeability*: a glued pair through a magnetic sheet against the generalized reference to 1e-9;
  an `ε↔μ` swap leaving the composite phase fixed while the ports move; a magnetic-loss sheet
  splitting `fx_te` from `fx_tm` with a lossless far medium; a traced solid with `μ ≠ 1` on the
  analytic slab to 1e-6 dB.
- *Reciprocity*: glued entry and glued exit equal to 1e-9 for a lossless solid; the lossy residual
  recorded against a stated bound.
- *Promoted diagnostics*: the walk-off phase to 0.01°, and the matched-sheet series collapse to 1e-9.
- *M2M direction*: the written `path_dirN` out of a lossy solid against the transitive-Snell
  direction, to 1e-9.
- *Unentered branches*: a displacement-budget cap actually reached, a scalar glued sheet, a `float`
  sheet chain, and a corner carrying a sheet, each against the same references as its `double` EM
  counterpart.
- *Layered stack*: `sheet_case 4` resolves; `|S|` independent of `mtl_prev` for a near-matched sheet
  to 1e-9; a two-layer stack against a characteristic-matrix reference to 1e-9; and every
  single-layer bound in the §1 table unchanged.
- Everything in the table in §1 stays where it is. Those bounds are numerical, not physical, and a
  failure means an approximation has been reintroduced.
