# Sheet / Solid — Fix Plan v10

**Scope:** the thin-sheet composite, the layered-stack core and the traced-solid slab correction in
`ray_mesh_interact.cpp`, covering both public functions in that file.

**Status:** the test suite is green. Every item below is either measured, with the diagnostic table
that measures it named, or explicitly marked as having no fixture yet.

This document stands on its own. Section 1 is the set of settled conventions a change has to work
within, section 2 ranks what is open, section 3 states each open item, section 4 records what is
closed and must not be reopened, and section 5 records limitations that are accepted rather than
open.

---

## 1. Where things stand

Three solvers share one file. Two of them are modes of `qd_slab_resolve` and the third is
`qd_stack_resolve`.

**Produce mode** of `qd_slab_resolve` builds a thin-sheet composite outright. A sheet has zero
world-space thickness, so there is nothing for the tracer to re-emit into and the internal series has
to be summed in one event. It returns absolute transmit and reflect coefficients.

**Correction mode** of `qd_slab_resolve` adjusts coefficients the tracer has already written for a
real traversed solid. It returns the per-channel series factor `S = 1/(1 − rn·rf·φ²)` and nothing
else, because the port the caller holds already carries the Fresnel coefficient of its own interface.

**`qd_stack_resolve`** is the characteristic-matrix core. It resolves an ordered list of layers
between two half-spaces in one pass and returns absolute coefficients on both ports. It is used for
an ordered sheet stack, and inside correction mode to resolve a bounding sheet as a layer rather than
as a half-space.

`ray_state_update` re-resolves every sheet crossing when medium tracking is on, so in any run with
state its composite is the one that reaches the output and whatever `ray_mesh_interact` produced for
that crossing is discarded. `ray_mesh_interact`'s own composite serves callers without state and must
agree with the redo; it does for every sheet case except an ordered stack, which is issue 1.

### The characteristic-matrix core

Per medium the admittance is `z = sqrt(ε/μ)` and the index is `n = sqrt(ε·μ)`. Every cosine comes
from the transverse invariant `ε μ (1 − c²)`, which is conserved across every boundary and is fixed
by the incidence cosine in the near medium. Per polarization the tilted admittance is

    η_TE = z·c        η_TM = z/c

which are the groupings `Material::interact_with` is already written in: its TE reflection is
`(η₁−η₂)/(η₁+η₂)` exactly, and dividing its TM reflection through by `c₁c₂` gives
`(η₂−η₁)/(η₂+η₁)`. So the library's TM reflection is minus the admittance-form one and its TM
transmission carries an extra `c₁/c₂`.

The layer matrix for normal thickness `d` and phase `δ = k₀ n c d` is

    [ cos δ            (i/η) sin δ ]
    [ i η sin δ         cos δ      ]

With `M` the product in propagation order and `[B; C] = M [1; η_far]`,

    r = (η_near·B − C)/(η_near·B + C)        t = 2 η_near/(η_near·B + C)

**What propagates is amplitude, not flux.** Tangential E and H are continuous across every boundary
while normal flux decays inside a lossy layer, so the product runs in amplitudes and the conversion
to per-channel normal flux is applied once, at the outer pair, on the transmit port. Inside the
product the intermediate admittances cancel identically. The TM cosine ratio telescopes the same way,
`∏ (c_j/c_{j+1}) = c_near/c_far`.

**Reduction to one layer.** Substituting `cos δ = (φ⁻¹+φ)/2` and `i sin δ = (φ⁻¹−φ)/2` with
`φ = exp(−j k₀ n c d)` and clearing `2η₁` gives

    r = (r01 + r12 φ²)/(1 + r01 r12 φ²)      t = t01 t12 φ/(1 + r01 r12 φ²)

which is produce mode term for term: `rn = −r01` is the Stokes sign `fresnel_r_near` returns,
`t01 t10 = 1 − r01²` turns the reflect port into the Airy numerator, and the flux conversion, the
lumped interface gains and the displaced-host division are end quantities outside the matrix.
Measured, the two agree to 9.6e-16 over 572 rows; the exception is §5.

**What the core declines.** Three material columns have no matrix expression and make it return
false rather than approximate: `m > 0`, a lumped acoustic surrogate whose in-layer index is not
physical; `alpha`, dB per meter of path, which produce mode applies as a one-way multiplier on `φ`
that a matrix layer has no counterpart for; and `tf`, which redistributes between the reflected and
transmitted ports at a boundary and is not a boundary admittance. Where it declines the caller keeps
what it did before. This is issue 2.

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
  solid, across a sheet reached from inside a medium, and across the core, which applies the factor
  once at the outer pair only.

### The incidence cosine in produce mode and in the core

`abs_cos_theta` is a real angle and is what a face hit supplies. A sheet reached from inside a traced
medium has no real incidence cosine: the propagation cosine there is complex and the tracer's
direction carries only its real projection. Such a caller passes the recovered value through
`qd_slab_resolve`'s `ct_near_exact` argument, and produce mode hoists two quantities from it,

    ci     = ct_near_exact, or complex(abs_cos_theta, 0) when nothing is supplied
    sin2_i = 1 − ci², or the real spelling 1 − abs_cos_theta·abs_cos_theta in the same case

Five sites read them: the in-sheet cosine `cts_exact`, `fresnel_r_near` (both its explicit `z·c`
factors and its refraction root), `fresnel_r_far`, the far-cosine and `flux_amp` pair that forms the
near-to-far conversion, and the displaced-host cosine `c_disp`. Everything there is analytic in the
cosine, so a real value reproduces the angle path exactly; the declining branch keeps the real
spelling verbatim so nothing reassociates. Correction mode never reads either quantity.

`qd_stack_resolve` takes the same cosine directly, as its `ci` argument, and anchors the transverse
invariant in it.

The mass-law route keeps the real projection. The surface-mass surrogate and its angle factor are
lumped quantities the material table is calibrated on, not wave quantities, and `medium_gain` takes
them as real by construction — the same reasoning that excludes a mass-law slab from the exact
in-medium branch in `SCALE_BY_MEDIUM`. The two interface amplitudes in that route are wave quantities
and do carry the recovered cosine.

`abs_cos_theta_t` in produce mode is the in-slab cosine and reaches only `dist_slab = thk / cos`,
which feeds the alpha column and the mass-law path. It does not enter the composite.

### The glued-sheet flux conversion

Produce mode and the core both apply the near-to-far obliquity factor once, on the transmit port, so
a glued sheet reports the same energy as the equivalent traced entry. The gate is
`!mat_far.same_as(mat_near)`, so only the glued cases reach it: `sheet_case 2` (entry,
`air | sheet | solid`) and `sheet_case 3` (exit, `solid | sheet | air`). An isolated sheet, a
collapsed duplicate, an ordered stack in one medium and a sheet embedded in a medium all have
`near == far` and take the factor as exactly 1.

The composite is the Airy pair, or the matrix pair for more than one layer, in raw amplitudes, times
that conversion once, divided by the propagation of the medium the stack displaces. The numerator of
the conversion is evaluated at the cosine in the far medium, written from the near-side sine directly
rather than by chaining through the layers; Snell is transitive, so the layers drop out of it and the
far cosine is the same quantity a bare near|far crossing would have used. The denominator is the
incidence-side normal flux `Re(z_n c_n)` per channel, and it has to be the same real number the
crossing that entered the near medium put in its numerator, or the pair does not telescope. The
reflect port turns around in the near medium, where the ratio is 1, and carries nothing.

Measured, the factor is worth +3.60 dB at normal incidence and +13.75 dB at 85° for an entry into
`eps 5.24 − 0.63j`, and −0.88 dB to −3.57 dB over 0° to 50° for an exit out of `eps 1.50`. Its TE/TM
split reaches 0.235 dB at 85° for a lossy far medium and is identically zero wherever the far medium
is lossless.

### The in-medium angle

A traced ray must have a real direction, so `qd_refract` normalizes a vector built from `Re(cos θ₂)`
and the imaginary part is dropped. The map it applies is

    ct_traced = Re(c₂) / |η·u + (η·cos θᵢ − Re(c₂))·n|

and it is deterministic and monotone. Real-index Snell is a different map, and inverting that instead
leaves a residual worst exactly where the defect is worst, because it ignores the normalization.
`EXACT_IN_SLAB_COSINE` inverts the map itself: three Newton steps on
`η²u = t²·Re(sqrt(1 − ε_r u))²` with `u = sin²θᵢ`, starting from the real-index guess. It reaches the
exact complex cosine to machine precision. Permeability is carried: the two ends of `ε_r` are `ε·μ`
products, so `η²` is the same `snell²` that `qd_refract` used.

It declines, returning zero, when: no real direction was supplied or the traced and geometric cosines
agree to 1e-6; the current medium is not a real material; the previous face is not parallel; **the
previous-medium word holds a sheet**; the crossing is under total reflection; or the entry and slab
media are matched. Where it declines, every consumer keeps the real projection and nothing moves.

The sheet condition exists because `mtl_prev` has to name the medium the current leg was entered
from, and `SHEET_EVENT` writes the sheet into that word. After a sheet crossing the entry medium is
one level further back than the recovery can see, so inverting the map there would return zero for a
sheet electrically matched to its host and a wrong number for one that differs, making the answer
depend on how many sheets the ray had already crossed.

Consumers, all on that one gate: both boundaries of the Airy denominator, the slab propagation factor
inside `qd_slab_resolve`, the interface coefficient in `REPLACE_BY_VBS`, the in-medium traversal
factor in `SCALE_BY_MEDIUM`, and the incidence cosine that `SHEET_RESOLVE` passes to produce mode or
to the core.

### The propagation convention

Where a recovered cosine exists and the tracer traced the bend, the traversal factor is
`exp(−j k₀ (n·cs·d + q·s))`, with `d` the normal thickness and `s` the transverse displacement, both
pure geometry off the traced direction, and `q² = n²(1 − cs²)` the transverse wavenumber Snell
conserves from the entry medium. Conductivity lives inside `n`; only the alpha column is drawn from
`medium_gain`, through `excess_gain`. For a real index the two terms recombine into `n·dist_refract`
exactly, so this reduces to the path form rather than replacing it.

Everywhere else the traversal keeps the ITU-R P.2040 rate along the path with the phase from the real
index. That is the right treatment where the tracer did not trace a bend: there is no inhomogeneous
wave to be exact about. The two places where it is still applied on a leg that DOES have a recovered
cosine are issues 7 and 11.

### The mass-law convention

A material with `m > 0` carries a lumped acoustic transmission surrogate rather than a bulk
propagation loss. `Material::medium_gain` forms it from two pieces that are supplied separately, a
surface-mass surrogate with the angle removed and an angle factor:

    A  = m · log10((f/fRef) · M)
    X₀ = ((f/fRef) · M)^(m/20)
    TL = A · log10(1 + (X₀·cos θ)²) / log10(1 + X₀²)

`A` is the normal-incidence term, the value the material table is calibrated on, and it is returned
unchanged whenever the angle factor is one. The angle carries the mass law's own shape normalized to
it, so both endpoints are exact: `A` at normal incidence, zero at tangency. The asymptotic angle slope
is 20 dB per decade of `cos θ` and does not follow `m`, because the cosine enters through the fluid's
wave admittance rather than through the panel's transfer impedance. The normalization factor is
within 5e-4 of ten for any panel with meaningful loss, so this is the exact single-panel mass law
wherever the term is worth anything.

Callers supply the two pieces as follows. A sheet passes the normal thickness and the geometric
incidence cosine, so the quotient `thk / cos` is never formed. A traced solid passes neither and gets
`dist_geo·cos θ` and `cos θ` by default, whose product is the `dist_geo·cos²θ` the term was written in
before; across a slab traversal `dist_geo` is the normal thickness over the cosine, so the surrogate
is the normal thickness and the two callers form the same argument. The public
`quadriga_lib::medium_gain` entry point, and `calc_diffraction_gain` through it, supply no angle and
therefore read `A` unchanged.

`m > 0` also excludes a slab from correction-mode resolution entirely, from the exact in-medium
propagation branch in `SCALE_BY_MEDIUM`, and from the core: the eta/mu of such a material are
calibrated for surface impedance only, so the in-slab index is not physical and the internal series
must not be resummed.

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
| Sheet in a medium, embedded and glued exit, both ports, scalar and EM, 10 angles | 1e-9 | 9.7e-16 |
| Complex-cosine composite reference against the real-cosine one, 40 rows | 1e-15 | 0.000e+00 |
| Host traversal isolated by path-length ratio, both traversal forms | 1e-12 | at the floating-point floor |
| Mass-law sheet composite and mass term, swept to 89.9°, 15 angles | 1e-6 dB | 1e-14 dB |
| Mass-law angle slope, 30° to 60° | (19.5, 20.0) | 19.842911 |
| Mass-law term at normal incidence against the superseded form | 1e-9 | exact |
| Mass-law reflect port equals the bare near interface, and is independent of `m` | 1e-9, 1e-12 | exact |
| Characteristic-matrix reference against `diag_sheet_composite`, 5 stacks × 7 angles × 2 modes | 1e-14 | 7.1e-16 |
| The same against `diag_sheet_composite_cx` at a complex incidence cosine | 1e-14 | 1.2e-15 |
| Matrix identities: a zero-thickness layer, a layer split into halves, transmit port under layer order | 1e-14 | exact |
| Reflect port of a two-layer stack depends on layer order | > 1e-3 | 5.3e-2 |
| Host layer bounded by a sheet, series against the two-layer reference, 2 sheets × 2 lengths × 4 angles | 1e-9 | 5.7e-16 |
| A near-matched sheet's two-layer reference against the extended-path form, at normal incidence | 1e-4 | 8.0e-6 |
| `sheet_case 4` resolves, both orders, both ports, scalar and EM | 1e-9 | 5.6e-16 |

Four further properties were established offline and have no fixture. The real-angle path through
`Material::interact_with` is bit-identical to its pre-change form over 140544 comparisons across four
materials, six `tf`/`att` cases, six interaction types and 244 angles. The entry and exit obliquity
factors across a traced solid multiply to 1.000000000000 in both channels at every angle tested,
which is the telescoping property the chain fixture measures end to end. `qd_slab_resolve` with no
`ct_near_exact` supplied is bit-identical to its pre-change form over 31104 rows spanning 9 near, 8
slab and 9 far media, scalar and EM, both displacement sides, both modes and six angles, with `att`,
`tf`, `alpha`, mass-law, resonance, coincidence and permeability all live. And produce mode is
bit-identical across the introduction of the core, over 3600 rows spanning six materials with `thk`,
`alpha`, `att`, `tf`, resonance and permeability all live, both modes, both displacement sides and
five angles.

### Diagnostic infrastructure

`QD_TEST_VERBOSE` in each test file switches the tables on. It is currently defined in both
`test_ray_mesh_interact.cpp` and `test_ray_state_update.cpp`. The assertions run either way, with the
one exception named in issue 8.

The independent oracles are `qd_slab_ref`, `qd_slab_ref_R`, `qd_iface_ref`, `qd_mass_ref` and
`qd_mass_TL` in `test_ray_mesh_interact.cpp`, and `diag_slab_p_pol` with `diag_slab_p_avg` delegating
to it in `test_ray_state_update.cpp`. All evaluate closed forms with complex angles throughout and
share no code with the library. `qd_iface_ref` also carries the per-channel obliquity and flux fields
(`obl_te`, `obl_tm`, `F_te`, `F_tm`, `F_avg`). `qd_mass_TL_clipped` reproduces the superseded
mass-law form and exists so the diagnostic table records what that change was worth.

`test_ray_state_update.cpp` additionally carries a set of reference builders for the traced-solid leg,
each reproducing one stage of what the function applies: `diag_port` (an interface at a real or
complex incidence cosine, with the per-channel flux factor), `diag_series` (the correction-mode Airy
factor at one cosine), `diag_medium_vbs`, `diag_medium_geo` and `diag_medium_walk` (the three
traversal forms), `diag_traced_ct` and `diag_ct_exact` (the `qd_refract` map and the value it
projects), `diag_r_slabside`, `diag_medium_gain`, and `diag_quad_x` / `diag_ray_at` for the mesh
calls.

`diag_sheet_composite`, returning `DiagSheet`, is the produce-mode reference: the full three-medium
Airy pair, both ports, per channel, with the flux conversion on the transmit port only and the
displaced-host propagation divided out. It takes the three permittivities, the normal thickness, a
real incidence cosine and the displacement side. `diag_sheet_composite_cx` is the same form written
at a complex incidence cosine, for a sheet reached from inside a traced medium; it reproduces the
real-cosine builder exactly, which is asserted on every row of the sheet-in-a-medium case. Both assume
`μ = 1` and no permittivity resonance, which is issue 3.

`diag_cm_stack`, with `DiagLayer`, is the N-layer characteristic-matrix reference, returning the same
`DiagSheet`. It takes `(ε, μ)` pairs, so it does not have to be regeneralized when issue 3 lands. At
one layer it reproduces both builders to 1.2e-15, which is asserted rather than assumed.

Palette material 9 (`LO_NEAR`, `eps 1.5015 − 0.40j`, thk 6 mm) is a sheet near-matched to `LO`: close
enough that a half-space Fresnel reflection would be 2.4e-4, far enough that `TRANSPARENT_TO` declines
to merge it.

Isolating the host leg from the composite is done by running the same crossing over two host path
lengths and taking the ratio. The composite is identical on both and cancels, whatever cosine it is
built at, and both traversal forms are pure exponentials in the path length, so the ratio over `L` and
`2L` is the factor over `L`. Dividing the written field by a hand-built composite is the wrong way to
do it: that only works while the code and the reference are built at the same cosine, so it breaks the
moment the composite's argument changes.

---

## 2. Ranked issues

Ranked by importance. Section 6 gives the execution order, which differs where one item needs
another's reference or answer to proceed.

| # | Issue | Measured size | Where |
| :-: | --- | --- | --- |
| 1 | Two solvers for one physics, and a guessed backing where the state cannot name one | up to 1.80 dB when the guess is wrong; duplication unmeasured | `qd_stack_resolve`, `qd_slab_resolve`, `SHEET_EVENT` |
| 2 | Lumped material columns: no fixture in produce mode, and the core declines on three of them | unmeasured; a declining sheet loses its stack physics | `qd_slab_resolve`, `qd_stack_resolve` |
| 3 | Permeability never reaches the sheet, stack or slab path | unmeasured | `qd_slab_resolve`, `qd_stack_resolve`, `flux_amp`, all callers |
| 4 | The displaced-host term drops the transverse propagation | 9.08° of phase at 85°, no amplitude | `qd_slab_resolve`, `qd_stack_resolve` |
| 5 | The recovery's total-reflection gate reads the hit-code verdict | unmeasured | `EXACT_IN_SLAB_COSINE_SOLVE`, `SHEET_RESOLVE` |
| 6 | Glued entry / exit reciprocity residual | 0.074 dB to 0.096 dB, measured offline only | not isolated |
| 7 | The walk-off branch keeps the projected cosine while the series does not | unmeasured; no fixture that can fail | `SCALE_BY_MEDIUM` |
| 8 | The walk-off diagnostic asserts nothing numeric | no regression guard | `test_ray_state_update.cpp` |
| 9 | The corrected refraction direction at an M2M transition has no fixture | unmeasured | `REPLACE_BY_VBS` |
| 10 | Four produce-mode and stack branches are never entered by any test | unmeasured | `qd_slab_resolve`, `qd_stack_resolve`, `SHEET_EVENT` |
| 11 | Hybrid propagation factor where no cosine is recovered | 2.5e-5 dB | `qd_slab_resolve` |

---

## 3. The issues

### 3.1 Two solvers for one physics, and a guessed backing

The characteristic-matrix core landed and closed the layered-stack defects: an ordered sheet stack
resolves rather than killing the ray, a host layer bounded by a sheet resolves that sheet as a layer
of finite thickness, and the cross-layer coupling of a coated window is exact. What it left behind is
four separate loose ends that belong together, because each of them is a consequence of the core
having been added beside the existing solver rather than replacing it.

**Two implementations of one physics.** `qd_slab_resolve` produce mode and `qd_stack_resolve` at one
layer compute the same composite. They agree to 9.6e-16 over 572 of 600 rows. The single-sheet path
was deliberately left untouched so the bit-identity above would hold, and the fixture now measures
the two against each other, so unifying them is a checked step rather than a leap. Unification is the
target: one solver, called with one layer or with several.

**Where they disagree, and which is right.** The 28 rows that do not reduce are all past an internal
critical angle, for two distinct reasons. On the reflect port, `apply_tf` clamps `R0` into `[0, 1]`
before forming an interface reflection magnitude, and past the internal critical angle of the
near|far pair the slab-side TM Fresnel reflection of a lossy sheet exceeds unity — measured 1.0445 at
60° for `eps 1.50 | sheet eps 5.24 − 0.63j | air`. Produce mode caps it, up to 4.0e-2 off the exact
value; the core does not. On the transmit port, produce mode's `AMP` lambda takes a flat branch when
`1 − R0` falls below 1e-9, which under internal total reflection is a hard zero, while the core
carries the evanescent field through the finite layer. The core is right on both counts: a finite
layer past its critical angle does tunnel, and the clamp is an artifact of the amplitude-rescaling
scheme rather than physics. Neither behaviour has a fixture, and both have to acquire one before
produce mode is routed through the core, because both move numbers that §5 currently records as
bounds.

Until then a one-sheet crossing and a two-sheet crossing disagree past an internal critical angle.
Reachable only with a sheet less dense than its host.

**The guessed backing.** Correction mode resolves a bounding sheet as a layer, and a layer needs a
terminating half-space. `SHEET_EVENT` knows it and passes it; on any later leg the previous-medium
word holds the sheet and what lies behind it is one level further back than the state format reaches,
so air is assumed. Air is exact for the motivating geometry, a coating on the room side of a wall,
and it is what the fixture's reference encodes. It is wrong for a laminate. Measured, as the worst
absolute error in `|S|` over 0° to 50° for a 6 mm `eps 5.24 − 0.63j` coating on a 50 mm host of
`eps 1.50 − 0.40j`, swept over what is truly behind the sheet:

| true backing | half-space (superseded) | air-backed (current) | host-backed |
| --- | --- | --- | --- |
| air | 1.560 dB | 0.000 | 0.694 dB |
| the host itself | 0.866 dB | 0.694 dB | 0.000 |
| a solid matching the sheet | 0.000 | 1.560 dB | 0.866 dB |
| a denser solid, `eps 9.0 − 1.20j` | 0.238 dB | 1.798 dB | 1.104 dB |
| worst case | 1.560 dB | 1.798 dB | 1.104 dB |

No prior dominates. The `mtl_buffer` word looks like the way out: `SHEET_EVENT` writes
`next_buffer = 0` and sets the fix-flag rather than guessing when it meets a nonzero one, on the
stated grounds that no sheet code in scope carries a deferred buffer, so the word appears free on this
path. Writing the medium behind the sheet there and reading it in `SLAB_AIRY_FACTOR` would convert
the guess into a fact for one-sheet-deep nesting. That is a design decision rather than a free win:
the buffer already means "a medium entered but not confirmed" from the `o-ii` branch and is read by
`transparent_forward`, so overloading it needs either a discriminator or an argument that the two uses
cannot coexist.

**`ray_mesh_interact`'s own ordered stack.** `sheet_case 4` there still falls through to the legacy
M1/M2 path with the sheet bit set, so the two public functions now disagree for that case. With medium
tracking on the redo discards it, so it reaches only callers without state. The ordered list is
available there — `iMF`, `iMS` and the front/back verdict — and near and far are both air, so this is
the same call the redo makes.

**How many layers, and where more than two would apply.** The core takes an ordered list of any
length; nothing about it is specific to one or two. Three places would use a longer list if the state
could describe one, and all three are currently resolved as separate pieces:

- A glued solid is a two-layer stack, `sheet | solid`, and is currently resolved as a sheet composite
  followed by the solid's own series. That factorization was measured exact once the far reflection in
  the solid's denominator was corrected, so the split is not wrong today — but it is exact only
  because the correction supplied the missing coupling, and it depends on the backing being named.
  Resolving the pair as one two-layer stack would remove both dependencies.
- A single sheet inside a solid is a three-layer stack, `solid | sheet | solid`, currently a sheet
  composite between two host legs.
- Two stacked sheets inside a solid is a four-layer stack.

The obstacle is depth, not the core. `mtl_prev` is one level deep, so the state can name the medium a
leg was entered through but not the one before that, and a stack of three or more layers needs the
media on both sides of a chain the ray has already partly crossed. Whether that is a second
previous-medium word, a reuse of `mtl_buffer`, or a different decomposition is open and belongs with
the backing question above, since both are the same information deficit.

**Change.** Analysis first, then a decision on each of the four ends, then the work. They are listed
together because the answers interact: naming the backing and carrying more depth are the same state
question, and unifying the two solvers changes which behaviour past internal critical is the shipped
one.

**Assertions.** Produce mode routed through the core, with the two divergent behaviours each
asserted at their chosen value. A backing named from the state rather than assumed, with the laminate
row from the table above dropping to the reference. `ray_mesh_interact` and `ray_state_update`
agreeing on an ordered stack to 1e-9. A three-layer stack against `diag_cm_stack`.

### 3.2 Lumped material columns

Two related gaps, one of them new.

**No fixture in produce mode.** Three branches inside `qd_slab_resolve` produce mode are entered by
no test, because every palette that reaches a sheet sets the relevant columns to zero.

The `AMP` lambda rescales each interface amplitude by `sqrt((1 − R_tf)/(1 − R0))` to carry the `tf`
redistribution. With `tf = 0` that quotient is exactly 1 and the branch is inert, which is the only
case any sheet test exercises. `tf` is set in exactly two test cases in the suite, both on the
correction-mode path.

`ig = sqrt(slab.interface_gain · (far_is_new ? far.interface_gain : 1))` folds the lumped penetration
loss into the transmit port. The `far_is_new` test — `!far_is_air && !mat_far.same_as(mat_near)` — was
written to stop a sheet embedded in a medium from paying that medium's entry loss twice. Nothing
measures it. Real material tables carry `att`, and the glued pair is the motivating geometry for it,
so this is the branch of the three most likely to reach a user. The core generalizes it to a product
over layers, which is equally unmeasured.

`eta_resonance` is added to every interface permittivity in produce mode and deliberately excluded
from the propagation index, on the reasoning that a permittivity resonance is an interface effect
rather than a propagation one. That split appears at five separate sites in the solver and again per
layer in the core. One test case in the suite sets `resF`, and it is not a sheet.

**The core declines on three of them.** `m`, `alpha` and `tf` have no characteristic-matrix
expression, so `qd_stack_resolve` returns false rather than approximating. The consequences are not
uniform: an ordered stack whose sheet carries any of the three kills the ray, and a bounding sheet
that carries one falls back to the half-space treatment the core was added to replace. Both are
silent. A real material table with `att` and `alpha` on a wallpaper is an ordinary thing to have.

Of the three, `alpha` is the one worth attacking: it is a path-length attenuation and could be folded
into a layer as an equivalent extinction, at the cost of a new approximation at oblique incidence that
would have to be measured against the path form. `tf` and `m` are lumped by construction and the
decline is the honest answer for them.

**Change.** Fixture first. Extend the glued fixture with an `att`-bearing sheet and an `att`-bearing
solid, a `tf`-bearing pair on both ports, and a resonant sheet, each against `diag_sheet_composite`
extended to carry the same three columns, and add the stack equivalents against `diag_cm_stack`. Then
measure what a declining core costs on a realistic table and decide whether `alpha` gets an
equivalent-extinction path.

**Assertions.** A glued entry through an `att`-bearing sheet into an `att`-bearing solid pays each
interface gain exactly once, to 1e-9, and an embedded sheet pays the host's not at all. A `tf` sweep
on both ports against the closed form to 1e-9. A resonant sheet whose interface terms move with `resS`
while its propagation factor does not, to 1e-9. An ordered stack carrying `att` on both layers paying
each exactly once. A stack that makes the core decline, asserted to decline rather than to produce a
number.

### 3.3 Permeability never reaches the sheet, stack or slab path

The `e`, `f`, `g`, `h` columns are in the material model and in every expression that matters:
`z = sqrt(ε/μ)` in `flux_amp` and in both `fresnel_r` helpers, `n = sqrt(ε·μ)` in the propagation
factor and in every layer phase of the core, `(ε₁μ₁)/(ε₂μ₂)` in every refraction root and in the
total-reflection test, and the `ε·μ` products at both ends of `EXACT_IN_SLAB_COSINE`. ITU-R P.2040
does not specify them and no shipped material uses them, but the logic is live and unguarded.

Permeability is exercised by six test cases, all of them single interfaces or a matched Airy factor:
the backward-compatibility default, the reflection/refraction decoupling, the impedance-matched dense
medium, the `ε↔μ` swap invariant, `medium_gain` under magnetic loss, and the public helpers. None of
them involves a sheet, a stack, a glued pair, a traced-solid chain, or the flux conversion.

Two things separate the magnetic case from the dielectric one and neither is covered. A complex `μ`
from the `g` column makes `z` complex even for a real `ε`, so `Re(z c)` and `Re(z̄ c)` split in
`flux_amp` where the dielectric case would not. And `ε↔μ` swaps leave the index `n` and therefore the
propagation factor unchanged while inverting the admittance, so a swapped pair is a strong self-check:
the slab phase must not move while both ports do.

`diag_sheet_composite` and `diag_sheet_composite_cx` take permittivities only and form `z` and `n`
both as `sqrt(ε)`. Both have to be generalized to `(ε, μ)` pairs before this item can be measured at
all. The two must stay in step; the sheet-in-a-medium case asserts they agree at a real cosine and
that assertion has to keep holding. `diag_cm_stack` already takes the pairs and needs no change.

**Change.** Generalize the two builders, then a fixture. No source change is expected; this is coverage
for logic that shipped without any.

**Assertions.** A glued entry and a glued exit through a magnetic sheet against the generalized
reference to 1e-9. An `ε↔μ` swap on the sheet leaving the composite phase unchanged to 1e-9 while the
per-channel powers move. A magnetic-loss sheet whose `fx_te` and `fx_tm` differ with a lossless far
medium, which the dielectric case cannot produce. A traced solid with `μ ≠ 1` against the analytic
slab to 1e-6 dB. A two-layer stack with `μ ≠ 1` on one layer against `diag_cm_stack` to 1e-9.

### 3.4 The displaced-host term drops the transverse propagation

A sheet has zero world-space thickness, so it stands for a layer that occupies `thk` of whatever
medium it sits in. The tracer walked that length as that medium and the rest of the path already
carries it, so the transmit port divides it back out. Both produce mode and the core form that as
`exp(−j k₀ n_d · t_disp · c_disp)` — the normal component only.

What the host leg actually charged is the inhomogeneous form of §1,
`exp(−j k₀ (n·cs·d + q·s))`. Over a slice of normal thickness `thk` that is
`exp(−j k₀ (n_h c_h + q_h tan θ_traced) thk)`. The transverse term is missing from the removal, so the
two do not cancel.

**Size.** For an entry from a lossless medium the transverse wavenumber is real —
`q = sqrt(n²(1 − c²)) = sin θ_entry` exactly — so the term costs no amplitude and only phase. Over a
6 mm sheet in `eps 1.50 − 0.40j` entered from air it is 0.18° at 10° of external incidence, 4.25° at
50° and 9.08° at 85°. It is absent from the code and from the reference builders, so no current
fixture can see it, and it does not affect the sheet-in-a-medium residual. An ordered stack multiplies
it by the layer count, since the displacement runs over the total thickness.

It matters where the sheet's phase matters: coherent combining of a path that crosses a sheet inside a
solid against one that does not, and any multipath sum at that scale. It is zero at normal incidence
and zero for a sheet in air.

**Change.** Extend the displacement factor to `exp(−j k₀ (n_d c_disp + q_d tan θ_traced) t_disp)`,
with `q_d` the transverse wavenumber of the displaced medium, in produce mode and in the core, and
extend all three reference builders the same way. The traced angle enters here as pure geometry,
exactly as it does in `SCALE_BY_MEDIUM`. Check the interaction with `disp_budget`: where the budget
caps `t_disp` below the total thickness, both terms have to be capped together.

**Assertion.** A sheet embedded in a lossy solid whose composite phase matches a reference carrying
both terms, to 0.01°, across an angle sweep. And an end-to-end statement: a traced solid of fixed total
thickness resolved with and without an embedded impedance-matched sheet must give the same phase to
0.01°, which is the property the displacement exists to preserve and which no fixture currently checks
in phase.

### 3.5 The recovery's total-reflection gate reads the hit-code verdict

`EXACT_IN_SLAB_COSINE_SOLVE` declines when `total_reflection` is set, and that flag is `typeH[5]` —
the verdict `ray_mesh_interact` formed at the geometric angle for the pair the mesh named.

`SHEET_RESOLVE` re-derives its own verdict from the outer pair at `theta_t`, and the two can disagree
in either direction. That re-derivation is the reason `SHEET_RESOLVE` exists: with an undeviated
tracer a ray inside a solid travels at its external angle, so `ray_mesh_interact` evaluates the
crossing at an angle that is not the in-slab one and past the internal critical angle declares a
mirror where there is none, at which point a glued exit loses tens of dB.

Where the hit code says total reflection and the re-derivation says otherwise, the recovery declines
and the composite falls back to the projected cosine — a silently missed correction on exactly the
rows the re-derivation was written for. Where the re-derivation says total reflection, the composite is
not used at all, so that direction is harmless.

The same question applies to the other consumers. The current face's total-reflection verdict is a
statement about the interface ahead of the ray, while the recovery is inverting the bend at the entry
face behind it, and those are not the same event.

**Change.** Measure first: build a glued exit whose hit code carries a spurious total-reflection flag
and record what the correction is worth there. Then decide whether the gate takes the verdict as an
argument, so `SHEET_RESOLVE` can pass its own, or whether it should be dropped for all consumers.

**Assertion.** A glued exit past the geometric critical angle of the mesh-named pair but below the
outer pair's, whose composite matches the reference at the exact in-medium cosine to 1e-9.

### 3.6 Glued entry / exit reciprocity residual

The same three-medium stack resolved from both directions must give the same answer. Measured offline
against the library solver, entry `air | sheet | solid` against exit `solid | sheet | air`, TE/TM
power-averaged:

- Lossless solid: the two orientations agree to 0.000000 dB at every angle from 0° to 50°, and the
  product of the two flux conversions is exactly 1. Removing the conversion would leave the two
  disagreeing by 1.76 dB at normal incidence rising to 3.44 dB at 50°.
- Lossy solid, `eps 1.50 − 0.40j`: 0.0739 dB at normal incidence, rising monotonically to about
  0.096 dB at 50° of in-air incidence.

The attribution is settled. The whole of it is the flux splitting of §1, which is not exact for a lossy
medium and is a settled convention, and it is bounded from below by that convention. There is no
projection component left: the exit is evaluated at the exact in-medium cosine, so the conversion
product does not drift.

There is no fixture. `grep -i reciproc` finds only comment text.

**Change.** Add the fixture. The lossless case is the one that must read exactly zero and is the useful
regression guard; the lossy case records the flux-splitting residual against a stated bound.

**Assertion.** Glued entry and glued exit equal to 1e-9 for a lossless solid. For a lossy solid, the
residual recorded against a stated bound rather than asserted to zero, with the normal-incidence row
called out separately because no cosine question enters there.

### 3.7 The walk-off branch keeps the projected cosine while the series does not

`SCALE_BY_MEDIUM` selects between two traversal forms on
`walk_off = !use_vbs || refl_pass || resolved`. Where that is false the exact form of §1 applies. Where
it is true the ITU-R P.2040 path form applies with the `cos²` walk-off substituting for the untraced
lateral shift.

Two of the three conditions are `refl_pass` and `resolved`, and both can be true on a leg where the
tracer did trace the bend and a recovered cosine exists. On such a leg `SLAB_AIRY_FACTOR` builds its
series at the recovered cosine while the traversal that multiplies it is built at the projection. That
is an internal inconsistency of the same kind that was removed from the exit interface and from the
sheet composite, though it is bounded by the same class of numbers as issue 11 rather than by the
multi-dB obliquity gap, because a reflection stays in one medium.

Separately, the fixture problem is unchanged. The *Excess-phase walk-off convention* diagnostic
exercises the branch but cannot distinguish the two-cosine form from a single-cosine one: it supplies
no `path_dir`, so `use_vbs` is false, `dist_refract == dist_geo`, `θ_t == θ_i`, and the two cosines
coincide. The sheet-in-a-medium case does exercise `diag_medium_walk` on a leg where the two cosines
differ, which pins the form the code applies, but it does not answer whether that is the right form.

**Change.** A fixture with a VBS correction actually applied, so that `dist_refract` and `dist_geo`
differ and the two cosines separate. Then measure an internal reflection inside a lossy parallel slab
with `path_dir` supplied, and decide whether the exact form should extend to that branch.

**Assertion.** Applied phase matching the chosen form to 0.01°, both branches exercised; and, if the
exact form extends, an internal reflection leg agreeing with the closed form to 1e-9.

### 3.8 The walk-off diagnostic asserts nothing numeric

*Excess-phase walk-off convention* prints a table and then asserts only `isfinite` and `gain >= 0`.
With `QD_TEST_VERBOSE` off it is a no-op for regression purposes: a refactor could move the quantity
arbitrarily and the suite would stay green. It pins its form to 0.000° in the printed table and asserts
nothing about it, which matters because that phase form is itself under discussion in issue 7.

Every other diagnostic in both files now carries at least one numeric bound.

**Change.** Promote the printed quantity to an assertion: the applied phase against
`−k₀(n−1)·L·cos²θ_fbs` to 0.01°.

### 3.9 The corrected refraction direction at an M2M transition has no fixture

`REPLACE_BY_VBS` passes the recovered cosine into the interface call, and the refraction cosine it
returns moves with it. That value is the cosine the outer pair of media gives, which is the correct
one, and only `geom == 2` reads it — an exit keeps the traced direction and a mirror does not use it.
So a medium-to-medium transition out of a lossy traced solid writes a slightly different continuation
direction than it did before.

Nothing measures this. The one test that asserts a VBS-corrected direction uses a lossless palette
where the recovery is a provable no-op, and the two diagnostic cases that supply `path_dir` are both
i-o exits.

The same question exists on the sheet path. `SHEET_RESOLVE` derives its continuation direction from
`interact_with(M_near, M_far, ..., theta_t, ...)`, which is the real-angle path, so the direction it
writes is built from the projection even though its composite is not. Snell transitivity says the
outgoing direction depends only on the outer pair, so the two should agree, but nothing checks it. For
an ordered stack the same statement has to hold with the sheets dropping out entirely.

**Assertion.** An M2M transition from a lossy solid into a second medium, with `path_dir_prev`
supplied, whose written `path_dirN` matches a hand-built direction from the transitive-Snell cosine to
1e-9. The same statement for a glued exit and for an ordered stack.

### 3.10 Branches never entered by any test

Each is a real branch with a real guard and no coverage. None is suspected of being wrong.

**The displacement budget cap.** `t_disp = min(total thickness, disp_budget)` limits how much of the
host a stack may take back out, because taking a medium back out is a division and a stack thicker than
its host would amplify. Every sheet in the suite is 6 mm inside a host of 100 mm or more, so the cap is
never reached; the sheet-in-a-medium case asserts that it is not reached rather than exercising it. An
ordered stack reaches it sooner than a single sheet, since the displacement runs over the total
thickness, and issue 4 adds a second term that has to be capped in step with this one.

**Float `dtype`.** No sheet, stack or slab test runs at `float`. The composite is written to `dtype`
through `xprmat`, and the whole chain is asserted at `double` only.

**A corner with a sheet, `sheet_case 6`.** `ray_mesh_interact` drops SBS and handles FBS alone,
clearing the co-location and corner bits. `ray_state_update` re-derives the case from the material
indices and reaches it through the "no co-located second face" branch. Nothing tests the pair.

**The core's one-way loss cap.** `qd_stack_resolve` clamps each layer's phase to 46.05 Np of
imaginary part, 400 dB one-way, because `cos` and `sin` of a complex phase grow like `exp(|Im|)` where
the Airy form's `φ` decayed instead. Nothing reaches it. It matters that it is never reached by a
propagating wave, and that is asserted nowhere.

### 3.11 Hybrid propagation factor where no cosine is recovered

When no exact in-slab cosine is available, `qd_slab_resolve` correction mode takes the magnitude of the
propagation factor from `medium_gain` over the traced path and the phase from the real index and the
real projected cosine, while produce mode and the core take both from `exp(−j k₀ n L cos θ)` with the
index and the cosine complex. The two differ by the gap between the ITU-R P.2040 `Delta` form and
`exp(−k₀·Im(n)·L)`, which measures 2.5e-5 dB.

The branch fires only where the recovery declines, and there the traced cosine is a real geometric
incidence cosine rather than an in-medium propagation cosine, so treating the leg as an inhomogeneous
wave is not obviously justified. Below every other item here. Worth doing only as cleanup, or as a side
effect of issue 1.

This convention gap also bounds what can be asserted elsewhere. Describing one stack two ways — "air
behind, path `L + t`" against "a matched sheet of thickness `t` behind, path `L`" — evaluates it under
the two conventions, so the two agree only at normal incidence, where they converge linearly in the
mismatch (8.0e-6 at `Δε = 1.5e-3`, 8.3e-8 at 1.5e-5) and stay 2e-3 to 7e-3 apart at 40° to 50°
whatever the mismatch. The layered-stack fixture asserts against the characteristic-matrix reference
for that reason and keeps the extended-path equivalence as a normal-incidence check only.

---

## 4. Closed, with guards in place

**Per-channel energy convention.** Every transmission-class interaction carries the per-channel
normal-flux transmittance of its own interface, and the four types agree with each other exactly.
Asserted at 1e-9 over 192 rows, four materials, both directions, six angles, with total reflection
handled separately as a closed port. The derivation and its consequences are in §1.

**Exit interface at the recovered in-slab cosine.** The far face of a traced solid is evaluated at the
exact complex cosine recovered from the traced direction, so the entry and exit obliquity factors are
formed at the same cosine in the shared medium and cancel identically. Worth up to 4.44 dB on the
coefficient at 85°, and exactly zero for a real permittivity. Asserted at 1e-9 on the scalar
coefficient and on the basis-invariant EM gain.

**Sheet composite at the exact in-medium cosine.** A sheet reached from inside a traced solid is
resolved at the complex propagation cosine the path carried rather than at its real projection. Produce
mode takes it through `ct_near_exact` and writes all five of its incidence-side terms in it; the
recovery declines where `mtl_prev` holds a sheet, so it never inverts a map for the wrong pair. Worth
0.53 dB on the transmit port and 1.83 dB on the TM reflect port for a sheet embedded in
`eps 1.50 − 0.40j`, and 2.62 dB and 10.92 dB for one glued to its far face, at 85° of external
incidence. Asserted at 1e-9 on both ports in scalar and EM over ten angles, measuring 9.7e-16.

**In-medium traversal on refracted legs.** The traversal factor uses the normal-thickness decay of an
inhomogeneous wave rather than a path-length rate. This is what closes the end-to-end chain: without it
the traced solid sits 0.040 dB off the analytic slab at 85° for `eps 1.50 − 0.40j`, and with it the
residual is at the floating-point floor. Asserted through the chain fixture at 1e-6 dB.

**In-slab cosine projection.** The sheet takes the in-slab cosine as `|Re(cos θ₂)|`; the traced solid
gets it from `path_dirN`, which `qd_refract` produced by normalizing a vector built from the same real
part, and that normalization is not angle preserving. The two differ by 7.5e-4 relative worst case,
which is 0.083 mm of path over a 100 mm slab and under 0.01 dB. Quantified, negligible, guarded at
1e-3.

**The glued-sheet near-to-far flux conversion.** Measured, and correct as shipped; no source change was
made to its form. *Glued sheet — the near-to-far flux conversion* runs six stacks against
`diag_sheet_composite`: an isolated sheet, a sheet embedded in a medium, two glued entries swept to 85°
and two glued exits swept to 50°. Per-channel powers agree to 1.67e-15 worst, the phase is pinned
complex on the rows where nothing else multiplies the composite, and the reflect port agrees to
8.33e-16 against a reference carrying no conversion at all. The trivial branch is asserted to take the
factor as exactly 1, which pins the gate as `near == far` rather than as "far is air".

**The layered-stack defects.** Three symptoms, all closed and all asserted at 1e-9.

An ordered stack of two sheets, `sheet_case 4`, resolves rather than killing the ray. Both traversal
orders are exercised, the order is read from `typeH[1]` with `typeH[3]` recording the geometric
fallback, and the second material index is carried through the case re-derivation. Measured at 5.6e-16
against the characteristic-matrix reference on both ports, scalar and EM. The order affects the reflect
port only, which is asserted on the reference so a core that lost it cannot pass a transmission-only
check: matrix reciprocity, `det M = 1`, makes the transmit coefficient invariant under layer order to
1.1e-16, while the reflect port separates by 0.25 dB at normal incidence.

A host layer bounded by a sheet resolves that sheet as a layer of its own thickness rather than as a
half-space. Before, the half-space reflection was 2.4e-4 for a near-matched sheet against a true value
of 0.1224 rising to 0.4040, a factor of 500 to 700, and the Airy denominator collapsed onto 1, costing
the host its internal series: −0.067 dB to +0.979 dB with `|ΔS|` of 8.0e-3 to 1.07e-1. Measured now at
5.7e-16 against the two-layer reference over sixteen rows.

Cross-layer coupling in a genuine two-layer stack. For
`air | sheet 6 mm (eps 5.24 − 0.63j) | solid 100 mm (eps 1.50 − 0.40j) | air` the chain sat −0.213 dB
off at normal incidence, −0.297 dB at 45°, and +1.918 dB TE with +1.311 dB power-averaged at 85°; a
3 mm `eps 9.0 − 1.20j` coating reached +2.351 dB and a lossless pair +5.24 dB. The whole of it was the
far reflection in the host's denominator: substituting the correct two-layer value reproduces the exact
matrix to 4e-15 relative at every angle, so the sheet composite, the traversal and the exit port
telescope correctly on their own. Below about 40° the error was almost purely phase — at 20° the two
reflections agreed in magnitude to 4e-4 while their arguments differed by 82°, because a half-space has
no round trip.

**Produce mode across the introduction of the core.** Bit-identical over 3600 rows with `thk`, `alpha`,
`att`, `tf`, resonance and permeability all live. The core was added beside it, not in place of it, and
nothing in the table in §1 moved.

**Thin-sheet grazing behavior.** There is no grazing clamp in either function. A thin sheet is modelled
as a laterally infinite layer, consistent with the plane-wave Fresnel boundaries and the plane-parallel
series it is built from. The composite is exact against the analytic slab to 89.999° in both ports, is
independent of the face it was cut from, and is invariant under mesh subdivision. Finite lateral extent
is an edge-diffraction question and belongs with the ray tube if it is ever taken up; it must not
re-enter the layer's transfer function, where it made results depend on triangulation.

The one quotient that can diverge, `thk / cos_slab`, is formed inside `qd_slab_resolve` and capped by
loss, 400 dB floored at the normal thickness, rather than by angle, so the cap depends on the material
and the frequency and on no geometry.

**Mass-law angle dependence.** The angle no longer sits inside the same logarithm as the frequency, so
the angle slope is 20 dB per decade of the cosine independent of the fitted `m`, and there is no angle
cutoff: the term tapers to zero rather than being clipped. The form and its rationale are in §1. The old
form read the fitted slope in the angle, which was 8.5 dB off at 85° for `m = 12`, and cut the term off
entirely past 86.8° to 89.5° for realistic panels, so a grazing ray crossed a panel with no attenuation
at all. Normal incidence is unchanged by construction, which keeps the public `medium_gain` entry point
and `calc_diffraction_gain` bit-identical and requires no table migration and no new material column.
Asserted across a 15-angle sweep to 89.9° at 1e-6 dB, measuring 1e-14.

---

## 5. Known limitations, recorded rather than open

`apply_tf` clamps `R0` into `[0, 1]` before forming an interface reflection magnitude. Past the
internal critical angle of the near|far pair, the slab-side TM Fresnel reflection of a lossy sheet
exceeds unity — measured 1.0445 at 60° for `eps 1.50 | sheet eps 5.24 − 0.63j | air` — and the clamp
caps it, leaving the produce-mode reflect port up to 4.0e-2 off the exact Fresnel value in that region.
The transmit port is identically zero there, so only the reflect port is affected. The glued fixture
sweeps stop at 50° to keep this out of the flux-conversion measurement; the sheet-in-a-medium case can
sweep to 85° of external incidence instead, because the in-medium traced angle reaches only 51.7° there
against a 53.4° internal critical angle, and it asserts that the clamp is not engaged on any row. The
same clamp now also applies to a layer reflection the core produces for correction mode, which is a new
path to an old bound. `qd_stack_resolve` itself does not clamp, which is where the two solvers part
company; see issue 1.

`mtl_prev` is one level deep, so a `sheet | solid | sheet` sandwich resolves but a deeper nesting falls
back to air. The recovery enforces this rather than merely being bounded by it: it declines whenever
that word holds a sheet, so the second crossing of a sandwich is resolved at the projected cosine
instead of at a value recovered from the wrong pair. The same depth limit is what stops the core from
being handed more than two layers; see issue 1.

`SHEET_EVENT` sets the fix-flag rather than guessing when it meets a deferred buffer
(`mtl_buffer != 0`), because no sheet code in scope carries one.

The wedge test is unusable for a glued entry, since it compares FBS and SBS, which are co-located by
construction. The code defaults to parallel there, which is right for a coating on a flat wall and wrong
for a sheet glued to a wedge.

`Material::interact_with` keeps two spellings of the same two expressions, one for a real angle and one
for a complex cosine: the refraction-root argument and the total-reflection test. They are the same
statement and the complex form is the general one. The real form is retained because reassociating it
moved coefficients by up to 6e-14 relative and, at a matched interface, lifted the reflectance off
exactly zero, which sent a `tf`-forced reflection down the rescale branch and flipped the sign of a
coefficient whose power was unchanged. Do not merge them without re-running the bit-identity check
described in §1. `qd_slab_resolve`'s produce mode keeps the same split for the same reason: `sin2_i` is
formed from the real double wherever no complex cosine was supplied.

The mass-law term is a hard zero below an argument of one rather than a taper, so its frequency
dependence has a kink where its angle dependence no longer does. Removing the clip needs a knee that is
independent of the normal-incidence offset, and the columns cannot express one because `fRef` sets both:
choosing `fRef` to place `TL(0)` also places the roll-off. Bounded by 3 dB and only within a decade of
the knee, where the term is itself under 3 dB. Fixing it costs a material column and a table migration,
which is not worth it at that size.

---

## 6. Order

Importance and execution order differ here, because two cheap items protect what follows and because
the largest item is a work stream rather than a patch.

1. **Issue 8.** Two assertions, test-only. Do it before any refactor, not after.
2. **Issue 3.** Generalize the two single-layer reference builders to `(ε, μ)` pairs, then a fixture.
   No source change expected. Doing it early means every later item inherits references that already
   carry permeability.
3. **Issue 2.** Fixture for the produce-mode branches, then the measurement of what a declining core
   costs. Guards branches that real material tables reach, and its answer feeds issue 1.
4. **Issue 6.** Fixture. Its attribution is settled, so this is a straight regression guard.
5. **Issue 10.** Four fixture rows, independent of each other and of everything above.
6. **Issue 4.** Measure the phase, then extend the source term and all three references together.
7. **Issue 5.** Measure first. Small, and it sits in code that was recently touched.
8. **Issue 9.** Test-only, three assertions.
9. **Issue 7.** Test-only for the phase form; the branch question rides on issue 11.
10. **Issue 1.** Its own work stream. Absorbs issue 11. Everything above either protects it or gives it
    a reference it has to reproduce. Its four ends interact, so it starts with analysis and a decision
    rather than with a patch.

## 7. Acceptance

Each assertion below replaces or joins the sanity checks in the case that measures it.

- *Lumped columns*: `att` paid exactly once per new medium and not at all for an embedded sheet; a `tf`
  sweep on both ports; a resonant sheet whose interface terms move and whose propagation factor does
  not; a stack that makes the core decline, asserted to decline. Each to 1e-9.
- *Permeability*: a glued pair through a magnetic sheet against the generalized reference to 1e-9; an
  `ε↔μ` swap leaving the composite phase fixed while the ports move; a magnetic-loss sheet splitting
  `fx_te` from `fx_tm` with a lossless far medium; a traced solid with `μ ≠ 1` on the analytic slab to
  1e-6 dB; a magnetic layer in a two-layer stack to 1e-9.
- *Displaced host*: an embedded sheet's composite phase against a reference carrying both the normal and
  the transverse term, to 0.01°; and a traced solid of fixed total thickness giving the same phase with
  and without an embedded matched sheet, to 0.01°.
- *Reciprocity*: glued entry and glued exit equal to 1e-9 for a lossless solid; the lossy residual
  recorded against a stated bound.
- *Promoted diagnostics*: the walk-off phase to 0.01°.
- *Total-reflection gate*: a glued exit whose hit code carries a spurious mirror verdict, resolved at the
  exact in-medium cosine to 1e-9.
- *M2M direction*: the written `path_dirN` against the transitive-Snell direction, to 1e-9, for an M2M
  transition, a glued exit and an ordered stack.
- *Unentered branches*: a displacement-budget cap actually reached, a `float` sheet chain, a corner
  carrying a sheet, and the core's loss cap, each against the same references as its `double` EM
  counterpart.
- *One solver*: produce mode routed through the core, with the clamp and the total-reflection port
  behaviour each asserted at the chosen value; a backing named from the state rather than assumed; both
  public functions agreeing on an ordered stack to 1e-9; a three-layer stack against `diag_cm_stack` to
  1e-9.
- Everything in the table in §1 stays where it is. Those bounds are numerical, not physical, and a
  failure means an approximation has been reintroduced.
