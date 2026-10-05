# Sheet / Solid — Fix Plan v5

**Scope:** the thin-sheet composite and the traced-solid slab correction in
`ray_mesh_interact.cpp`, covering both public functions in that file.

**Status:** the test suite is green, 1127718 assertions in 882 test cases. Every item below is
measured, and each measurement names the diagnostic table it comes from, or says explicitly that it
was derived by hand and has no fixture yet.

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

### The in-medium angle

A traced ray must have a real direction, so `qd_refract` normalizes a vector built from `Re(cos θ₂)`
and the imaginary part is dropped. The map it applies is

    ct_traced = Re(c₂) / |η·u + (η·cos θᵢ − Re(c₂))·n|

and it is deterministic and monotone. Real-index Snell is a different map, and inverting that
instead leaves a residual worst exactly where the defect is worst, because it ignores the
normalization. `EXACT_IN_SLAB_COSINE` inverts the map itself: three Newton steps on
`η²u = t²·Re(sqrt(1 − ε_r u))²` with `u = sin²θᵢ`, starting from the real-index guess. It reaches
the exact complex cosine to machine precision.

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
have a recovered cosine are issues 6 and 8.

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

Two further properties were established offline during the last change and have no fixture. The
real-angle path through `Material::interact_with` is bit-identical to its pre-change form over
140544 comparisons across four materials, six `tf`/`att` cases, six interaction types and 244
angles. And the entry and exit obliquity factors across a traced solid multiply to
1.000000000000 in both channels at every angle tested, which is the telescoping property the chain
fixture measures end to end.

### Diagnostic infrastructure

`QD_TEST_VERBOSE` in each test file switches the tables on. It is currently defined in both
`test_ray_mesh_interact.cpp` and `test_ray_state_update.cpp`. The assertions run either way.

The independent oracles are `qd_slab_ref`, `qd_slab_ref_R` and `qd_iface_ref` in
`test_ray_mesh_interact.cpp`, and `diag_slab_p_pol` with `diag_slab_p_avg` delegating to it in
`test_ray_state_update.cpp`. All evaluate closed forms with complex angles throughout and share no
code with the library. `qd_iface_ref` also carries the per-channel obliquity and flux fields
(`obl_te`, `obl_tm`, `F_te`, `F_tm`, `F_avg`).

`test_ray_state_update.cpp` additionally carries a set of reference builders for the traced-solid
leg, each reproducing one stage of what the function applies: `diag_port` (an interface at a real or
complex incidence cosine, with the per-channel flux factor), `diag_series` (the correction-mode
Airy factor at one cosine), `diag_medium_vbs` and `diag_medium_geo` (the two traversal forms),
`diag_traced_ct` and `diag_ct_exact` (the `qd_refract` map and the value it projects),
`diag_r_slabside`, `diag_medium_gain`, and `diag_quad_x` / `diag_ray_at` for the mesh calls. They
were checked bit-identical against the library primitives when they were written.

---

## 2. Ranked issues

| # | Issue | Measured size | Where |
| :-: | --- | --- | --- |
| 1 | Mass-law angle dependence is unbounded at grazing incidence | unmeasured, untested | `Material::medium_gain` |
| 2 | A sheet reached from inside a medium is resolved at the projected angle | unmeasured | `SHEET_RESOLVE`, `qd_slab_resolve` produce mode |
| 3 | No transfer-matrix core: layered stacks are approximated or dropped | up to 0.99 dB; `sheet_case 4` kills the ray | `SHEET_EVENT`, `SLAB_AIRY_FACTOR` |
| 4 | The glued-sheet near-to-far flux conversion has no fixture | unmeasured | `qd_slab_resolve` produce mode |
| 5 | Glued entry / exit reciprocity residual | 0.068 dB at 85°, not re-measured | not isolated |
| 6 | The walk-off branch keeps the projected cosine while the series does not | unmeasured; no fixture that can fail | `SCALE_BY_MEDIUM` |
| 7 | The corrected refraction direction at an M2M transition has no fixture | unmeasured | `REPLACE_BY_VBS` |
| 8 | Hybrid propagation factor where no cosine is recovered | 2.5e-5 dB | `qd_slab_resolve` |

---

## 3. The issues

### 3.1 Mass-law angle dependence at grazing incidence

**Current behavior.** `Material::medium_gain` adds `m_dB = m · log10((fGHz/fRef) · mass_path)` when
`m > 0`, with `mass_path = dist_geo · cos²θ`, and discards the term when `m_dB` comes out negative.
For a thin sheet the caller supplies that product directly as `thk_normal · cos θ`, so
`mass_path → 0` at grazing, `m_dB` goes negative, and the mass-law loss is discarded entirely. A
grazing ray passes an acoustic panel with no mass-law attenuation at all.

**Why it is open.** A geometric floor on the incidence cosine used to hold `mass_path` at
`thk · cos_min`, which put a floor under the mass-law term as a side effect. That floor was removed
for reasons unrelated to acoustics, and the mass law inherited its angle behavior from it. No test
exercises a mass-law sheet near grazing, so nothing failed. `grep -n "m\"\]" test_*.cpp` finds the
column set in the resonance and mass-law exclusion cases only, neither of which sweeps an angle.

**The modelling question.** The classic infinite-panel mass law is `TL(θ) = TL(0) + 20·log10(cos θ)`,
so transmission loss decreasing at grazing is the right sign and the divergence to zero loss at 90°
is the textbook pathology of that form. Practice handles it by limiting to field incidence, the
usual cutoff being around 78°, or by a finite-panel correction. The current code has neither.

There is a second question in the same expression. Because the angle enters inside the logarithm
alongside the frequency, one parameter `m` sets both the frequency slope and the angle slope, in dB
per decade of each. Those are independent physical quantities and there is no reason for them to be
equal.

Third, the `m_dB > 0` guard is a clip at zero rather than a saturation, so the term switches off
discontinuously rather than tapering.

**Change.** A decision before a patch, taken on acoustic grounds rather than by reinstating a
geometric floor. Whatever is decided, it must not depend on the mesh: any cutoff belongs to the
material or to the model, not to the triangle that was hit.

**Assertion.** A mass-law sheet swept from normal incidence to 89.9°, against whichever angle law is
chosen, with the behavior at and past the cutoff asserted explicitly. There is currently no such
case at all.

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

**Size.** Unmeasured, and measuring it is the first thing this item needs. Two effects to separate,
as at a solid exit. The composite amplitude itself moves with the incidence cosine. Separately, the
near-to-far flux conversion `fx_te` / `fx_tm` is formed from the real cosine, so for a glued pair
(`near ≠ far`) it does not cancel against the entry that produced it. Where `near == far` that
factor is identically 1 and only the first effect remains, which bounds the embedded-sheet case
well below the glued one.

**Why it cannot simply reuse the existing gate.** `EXACT_IN_SLAB_COSINE` recovers the cosine in
`mtl_current` using `mtl_prev` as the entry medium. `SHEET_EVENT` writes the sheet material into the
previous-medium word, so on the leg after a sheet crossing `mtl_prev` is the sheet and the entry
medium is one level further back than the recovery can see. The gate has to be re-derived for this
path, or the recovery has to be told which medium the leg was entered from.

**Change.** Measure first. Then, if the size warrants it, give the sheet path the same recovered
cosine, under a gate that names its own entry medium correctly.

**Assertion.** A sheet embedded in a lossy solid and a sheet glued to the far face of one, both
compared against a hand-built three-medium reference evaluated at the exact in-medium cosine, to
1e-9. Both cases currently reach the code and neither is measured.

### 3.3 No transfer-matrix core

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
conversion to normal flux at the ends, which is where the convention in §1 is expressed most
directly.

**Scope.** The largest item here and its own work stream rather than a patch. It absorbs 3.8, and it
should be sequenced after 3.2, since the incidence cosine the core is written in is the same
question 3.2 answers.

**Assertions.** `sheet_case 4` resolves rather than killing. `|S|` and `arg(S)` independent of
`mtl_prev` to 1e-9 for a near-matched sheet as well as an exactly matched one. A two-layer stack
against a hand-built characteristic-matrix reference to 1e-9.

### 3.4 The glued-sheet near-to-far flux conversion has no fixture

`qd_slab_resolve` produce mode applies the near-to-far obliquity factor once, on the transmit port,
so that a glued sheet reports the same energy as the equivalent traced entry. With `near == far` the
factor is identically 1, which covers isolated sheets, collapsed duplicates and sheets embedded in a
medium. Only `sheet_case 2` and `sheet_case 3` reach the non-trivial branch.

No fixture exercises them. The *Host layer carried through a matched sheet* diagnostic comes out with
`near == far` on every row, so it measures the trivial branch only, and the composite tests are all
air | sheet | air. The code path is therefore unmeasured.

This overlaps 3.2: the same factor is the one whose incidence cosine is wrong when the near side is a
traced solid. Building the fixture here first gives 3.2 something to measure against.

**Assertion.** A glued entry (air | sheet | solid) whose composite is compared against the product of
a hand-built sheet transfer and the entry interface into the solid, to 1e-9, and the same for a glued
exit.

### 3.5 Glued entry / exit reciprocity residual

The same three-medium stack resolved from both directions must give the same answer. It last measured
0.0001 dB agreement through 30°, growing monotonically to 0.068 dB at 85°. Reciprocity requires
exactly zero.

It has not been re-measured since the entry-angle recovery, the correction-mode series change, the
energy convention and the exit-angle change landed, all of which touch the same numbers. There is no
fixture for it: `grep -i reciproc` finds only comment text.

**Change.** Add the fixture first and re-measure before deciding whether anything is left to fix.

**Assertion.** Glued entry and glued exit equal to 1e-9.

### 3.6 The walk-off branch keeps the projected cosine while the series does not

`SCALE_BY_MEDIUM` selects between two traversal forms on `walk_off = !use_vbs || refl_pass ||
resolved`. Where that is false the exact form of §1 applies. Where it is true the ITU-R P.2040 path
form applies with the `cos²` walk-off substituting for the untraced lateral shift.

Two of the three conditions are `refl_pass` and `resolved`, and both can be true on a leg where the
tracer DID trace the bend and a recovered cosine exists. On such a leg `SLAB_AIRY_FACTOR` builds its
series at the recovered cosine while the traversal that multiplies it is built at the projection.
That is an internal inconsistency of the same kind that was just removed from the exit interface,
though it is bounded by the same class of numbers as 3.8 rather than by the multi-dB obliquity gap,
because a reflection stays in one medium.

Separately, the fixture problem the original walk-off item recorded is unchanged. The *Excess-phase
walk-off convention* diagnostic pins the applied phase to `−k₀(n−1)·L·cos²θ_fbs` at 0.000° against a
raw-path form that diverges to 26.17°, so the branch is exercised, but it cannot distinguish the
two-cosine form from a single-cosine one: it supplies no `path_dir`, so `use_vbs` is false,
`dist_refract == dist_geo`, `θ_t == θ_i`, and the two cosines coincide.

**Change.** A fixture with a VBS correction actually applied, so that `dist_refract` and `dist_geo`
differ and the two cosines separate. Then measure an internal reflection inside a lossy parallel
slab with `path_dir` supplied, and decide whether the exact form should extend to that branch.

**Assertion.** Applied phase matching the chosen form to 0.01°, both branches exercised; and, if the
exact form extends, an internal reflection leg agreeing with the closed form to 1e-9.

### 3.7 The corrected refraction direction at an M2M transition has no fixture

`REPLACE_BY_VBS` passes the recovered cosine into the interface call, and the refraction cosine it
returns moves with it. That value is the cosine the outer pair of media gives, which is the correct
one, and only `geom == 2` reads it — an exit keeps the traced direction and a mirror does not use it.
So a medium-to-medium transition out of a lossy traced solid now writes a slightly different
continuation direction than it did.

Nothing measures this. The one test that asserts a VBS-corrected direction uses a lossless palette,
where the recovery is a provable no-op, and the two diagnostic cases that supply `path_dir` are both
i-o exits. The change is a one-line consequence of the interface fix and is believed correct on
Snell transitivity, but it is unasserted.

**Assertion.** An M2M transition from a lossy solid into a second medium, with `path_dir_prev`
supplied, whose written `path_dirN` matches a hand-built direction from the transitive-Snell cosine
to 1e-9.

### 3.8 Hybrid propagation factor where no cosine is recovered

When no exact in-slab cosine is available, `qd_slab_resolve` correction mode takes the magnitude of
the propagation factor from `medium_gain` over the traced path and the phase from the real index and
the real projected cosine, while produce mode takes both from `exp(−j k₀ n L cos θ)` with the index
and the cosine complex. The two differ by the gap between the ITU-R P.2040 `Delta` form and
`exp(−k₀·Im(n)·L)`, which measures 2.5e-5 dB.

The branch fires only where the recovery declines, and there the traced cosine is a real geometric
incidence cosine rather than an in-medium propagation cosine, so treating the leg as an inhomogeneous
wave is not obviously justified. Below every other item here. Worth doing only as cleanup, or as a
side effect of 3.3.

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

---

## 5. Known limitations, recorded rather than open

`mtl_prev` is one level deep, so a `sheet | solid | sheet` sandwich resolves but a deeper nesting
falls back to air. This is also what blocks the recovery on the leg after a sheet crossing; see 3.2.

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

---

## 6. Order

1. **3.1.** Independent of everything else, unmeasured, and it is a behavior change that has already
   landed, so it should not sit unresolved through an acoustic run. It needs a decision before a
   patch and has no test coverage at all.
2. **3.4.** Fixture only, closes a code path that shipped unmeasured, and it gives 3.2 a reference to
   measure against.
3. **3.2.** Measure first, then decide. Structurally the same defect that was closed at the far face
   of a traced solid, in the one path that bypasses the correction.
4. **3.5.** Fixture first, then re-measure, then decide.
5. **3.7.** Test-only, and it pins a consequence of the last change that is currently unasserted.
6. **3.6.** Test-only for the phase form; the branch question rides on 3.8.
7. **3.3.** Its own work stream. Absorbs 3.8. Sequenced after 3.2, whose answer it needs.

## 7. Acceptance

Each assertion below replaces or joins the sanity checks in the case that measures it.

- *Mass-law angle dependence*: a sheet swept to 89.9° against the chosen angle law, with the cutoff
  behavior asserted.
- *Glued sheet*: entry and exit composites against a hand-built reference carrying the near-to-far
  flux conversion, to 1e-9.
- *Sheet in a medium*: an embedded sheet and a glued exit sheet, both inside a lossy solid, against a
  hand-built three-medium reference at the exact in-medium cosine, to 1e-9.
- *Reciprocity*: glued entry and glued exit equal to 1e-9.
- *M2M direction*: the written `path_dirN` out of a lossy solid against the transitive-Snell
  direction, to 1e-9.
- *Excess-phase walk-off*: the applied phase to 0.01°, with both cosines genuinely differing.
- *Layered stack*: `sheet_case 4` resolves; `|S|` independent of `mtl_prev` for a near-matched sheet
  to 1e-9; a two-layer stack against a characteristic-matrix reference to 1e-9.
- Everything in the table in §1 stays where it is. Those bounds are numerical, not physical, and a
  failure means an approximation has been reintroduced.
