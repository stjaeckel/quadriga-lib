# Sheet / Solid — Fix Plan v3

**Scope:** the thin-sheet composite and the traced-solid slab correction in
`ray_mesh_interact.cpp`, covering both public functions in that file.

**Status:** the test suite is green. Every item below is measured by a diagnostic case that
currently runs and prints, so each number in this document can be reproduced by building the tests
and reading the table named beside it.

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

### What is verified

Each of these has a promoted assertion, not just a printed table.

| property | bound | measured |
| --- | --- | --- |
| Per-channel split at a single interface, both interaction types | 1e-12 | 7.8e-16 worst |
| Thin sheet composite against the exact analytic slab, 5 materials × 8 angles | 1e-6 dB | 0.000000 dB |
| Sheet phase offset equals `k₀·thk·cos θᵢ` | 1e-3 deg | exact |
| Grazing incidence, both ports, independent of the face, to 89.999° | 1e-6 dB | exact |
| Mesh subdivision invariance, 2 triangles against 2·65² | 1e-9 relative | 0.000e+00 |
| Entry boundary built from the recovered complex angle | 1e-9 | exact |
| Host layer carried through an impedance-matched sheet | 1e-9 | gain 1.000000 |
| In-slab cosine projection, sheet form against traced form | 1e-3 relative | 7.5e-4 |

The recovery of the entry angle is worth up to 0.96 dB at 75° for `eps 1.50 − 0.40j` over 100 mm,
which is the `d_traced_vs_exact_dB` column of the entry-boundary table.

### Diagnostic infrastructure

`QD_TEST_VERBOSE` in each test file switches the tables on. It is currently defined in both
`test_ray_mesh_interact.cpp` and `test_ray_state_update.cpp`. The assertions run either way.

The independent oracles are `qd_slab_ref` and `qd_slab_ref_R` in `test_ray_mesh_interact.cpp` and
`diag_slab_p_avg` in `test_ray_state_update.cpp`, all of which evaluate the closed-form Airy slab
with complex angles throughout and share no code with the library.

---

## 2. Ranked issues

| # | Issue | Measured size | Where |
| :-: | --- | --- | --- |
| 1 | The per-channel energy convention in `interact_with` | ratio 0.889 against a target of 0.757 at 50° | `Material::interact_with` |
| 2 | No transfer-matrix core: layered stacks are approximated or dropped | up to 0.99 dB; `sheet_case 4` kills the ray | `SHEET_EVENT`, `SLAB_AIRY_FACTOR` |
| 3 | Mass-law angle dependence is unbounded at grazing incidence | unmeasured, untested | `Material::medium_gain` |
| 4 | The exit interface is still applied at the traced angle | unmeasured; the entry equivalent was 0.96 dB | `REPLACE_BY_VBS` |
| 5 | Glued entry / exit reciprocity residual | 0.068 dB at 85°, not re-measured | not isolated |
| 6 | Excess-phase walk-off has no fixture that can fail | phase only | `SCALE_BY_MEDIUM` |
| 7 | Correction mode keeps a hybrid propagation factor | 2.5e-5 dB | `qd_slab_resolve` |

---

## 3. The issues

### 3.1 The per-channel energy convention in `interact_with`

**Evidence.** Diagnostic *Slab factor per polarization*, `test_ray_state_update.cpp`. The table
prints three references for the TE/TM power ratio of a traced solid at an exit event, `eps 1.50 −
0.40j` over 100 mm:

| θ | 10° | 20° | 30° | 40° | 50° |
| --- | --- | --- | --- | --- | --- |
| `ratio_out` (measured) | 0.99831 | 0.99377 | 0.99614 | 0.96616 | 0.88911 |
| `ratio_post` = `\|t_X\|²·\|S_X\|²` | 0.99831 | 0.99377 | 0.99614 | 0.96616 | 0.88911 |
| `ratio_ref` = `(1 − R_X)·\|S_X\|²` | 0.99997 | 0.99882 | 0.98317 | 0.91491 | 0.75701 |

`ratio_out` sits on `ratio_post` to 1e-16 and departs from `ratio_ref` by 0.132 at 50°.

**What the two references mean.** `ratio_post` is the ratio of raw Fresnel amplitude powers.
`ratio_ref` is the ratio of per-channel normal-flux transmittances, which is what the analytic slab
carries and what an energy-conserving interface owes. The two differ because the transmittance is
`Re(z₂c₂)/Re(z₁c₁)·|t|²`, and the obliquity factor in front is not 1 once either the impedance or
the angle is complex.

**Root cause.** `Material::interact_with` folds a power into an amplitude at every interface, and it
does so with two different conventions depending on the interaction type. For undeviated
transmission (types 1 and 4) it sets `gain = 1 − R_eff` with `R_eff` built from the TE/TM **average**
reflectance, then rescales both Fresnel coefficients by the single common factor
`sqrt(gain / refraction_gain)`. For refraction (types 2 and 5) it sets `gain = refraction_gain ·
tf_scale`, which is the raw amplitude power with no flux conversion at all. So the reported total
follows the flux convention, the per-channel split follows the raw-amplitude one, and the two
transmission types do not agree with each other.

Note what is *not* wrong. The averaging does not distort the split by itself: at `tf = 0` the rescale
factor is identically 1 for reflection, and for transmission it is one factor common to both
channels, so it cannot move a ratio. This is asserted at 1e-12 in the *Interface energy split*
diagnostic. The defect is that no per-channel flux conversion is applied anywhere.

**The shape of the fix, to be confirmed before it is written.** Carry the amplitude `t` and apply the
obliquity factor once, at the point where amplitude becomes power, rather than folding a power into
the amplitude at every crossing. Equivalently, and more locally: have `interact_with` scale each
channel to carry its own `1 − R_X` instead of the averaged one, keeping the averaged value as the
reported `gain`. Under that change `ratio_out` should land on `ratio_ref` exactly.

**The tension to resolve.** Produce mode deliberately went the other way. `qd_slab_resolve`'s `AMP`
lambda uses the raw Fresnel amplitude rescaled only by whatever `tf` redistributes, because raw
amplitudes telescope — the air-to-air product over a whole path is the product of the per-crossing
amplitudes, which the flux form does not reproduce for a lossy medium. Substituting `sqrt(1 − R)`
there was worth 0.0156 dB at normal incidence on lossy concrete and was the entire residual of the
sheet composite before it was removed. Any change to `interact_with` has to leave that intact, which
is the reason to state the amplitude-versus-power boundary explicitly rather than to swap one
convention for the other.

**Blast radius.** This moves every oblique-incidence transmission and refraction in the library, not
only slabs. Record baselines for the whole suite before touching it.

**Assertion.** *Slab factor per polarization*: `ratio_out == ratio_ref` to 1e-9. The existing
assertion on `ratio_post` guards against a per-channel factor being reintroduced and should be kept
alongside, not replaced.

### 3.2 No transfer-matrix core

**Evidence, three separate symptoms.**

`sheet_case == 4`, an ordered stack, calls `KILL_RAY()` in `SHEET_EVENT`. Detection and the type bits
are correct but no sheet physics runs, so the ray disappears rather than being approximated.

Diagnostic *Series collapse with a matched sheet as the previous medium*,
`test_ray_state_update.cpp`. When a sheet is the bounding medium of a host layer,
correction-mode Fresnel treats it as a half-space. A matched pair gives `r = 0` and the Airy
denominator collapses to exactly 1, costing the host layer its internal series: `|S|` reads
1.000000 in every such row against a reference that oscillates with the fringe, worth 0.988 dB at
50° over 50 mm. The exactly-matched case is caught by the `merge_host` path in `SHEET_EVENT`, which
carries the layer through instead of cutting it; a sheet whose impedance merely happens to be close
to the host's is not caught and still collapses.

Cross-layer coupling in a genuine two-layer stack is absent in all cases. A coated window is modelled
as two independent interfaces plus a sheet composite, not as one stack.

**Target.** A 2×2 characteristic-matrix core resolving the whole stack in one pass. Exact for any
layer count, reduces to the Airy formula at one layer, produces R and T together, handles the
asymmetric glued case, and closes `sheet_case 4`. It also works natively in amplitudes with a single
conversion to power at the ends, which is where item 3.1 is settled without a separate convention
argument.

**Scope.** This is the largest item here and should be its own work stream rather than a patch.

**Assertions.** `sheet_case 4` resolves rather than killing. `|S|` and `arg(S)` independent of
`mtl_prev` to 1e-9 for a near-matched sheet as well as an exactly matched one. A two-layer stack
against a hand-built characteristic-matrix reference to 1e-9.

### 3.3 Mass-law angle dependence at grazing incidence

**Current behavior.** `Material::medium_gain` adds `m_dB = m · log10((fGHz/fRef) · mass_path)` when
`m > 0`, with `mass_path = dist_geo · cos²θ`, and discards the term when `m_dB` comes out negative.
For a thin sheet the caller now supplies that product directly as `thk_normal · cos θ`, so
`mass_path → 0` at grazing, `m_dB` goes negative, and the mass-law loss is discarded entirely. A
grazing ray passes an acoustic panel with no mass-law attenuation.

**Why it surfaced now.** A geometric floor on the incidence cosine used to hold `mass_path` at
`thk · cos_min`, which put a floor under the mass-law term as a side effect. That floor has been
removed for reasons unrelated to acoustics, and the mass law inherited its angle behavior from it.
No test exercises a mass-law sheet near grazing, so nothing failed.

**The modelling question.** The classic infinite-panel mass law is `TL(θ) = TL(0) + 20·log10(cos θ)`,
so transmission loss decreasing at grazing is the right sign and the divergence to zero loss at 90°
is the textbook pathology of that form. Practice handles it by limiting to field incidence — the
usual cutoff is around 78° — or by a finite-panel correction. The current code has neither, and
until this session it had a mesh-dependent cutoff standing in for one.

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

### 3.4 The exit interface is applied at the traced angle

**Evidence.** `REPLACE_BY_VBS` calls `interact_with(..., theta_t, ...)` with the real traced angle.
A traced ray must have a real direction, so `qd_refract` normalizes a vector built from `Re(cos θ₂)`
and the imaginary part is dropped; `theta_t` is that projection.

`EXACT_IN_SLAB_COSINE` already recovers the exact complex in-slab cosine by inverting the
`qd_refract` map, and it is already carried into both boundaries of the Airy denominator and into the
propagation factor. It is not carried into the exit interface.

**Size.** Not measured. The *Entry boundary* diagnostic divides the exit factor out by construction,
so it cannot see this. The equivalent defect at the entry boundary was worth up to 0.96 dB at 75° on
a weak reflector, which is the order to expect.

**Change.** Pass the recovered cosine into the exit `interact_with`, under the same gates the
recovery already applies: parallel faces, a direction that was actually bent, not under total
reflection, `path_dir_prev` present.

**Assertion.** A fixture that does not divide the exit factor out — the same crossing evaluated
against a hand-built exit coefficient at the exact angle, to 1e-9.

### 3.5 Glued entry / exit reciprocity residual

The same three-medium stack resolved from both directions must give the same answer. It last measured
0.0001 dB agreement through 30°, growing monotonically to 0.068 dB at 85°. Reciprocity requires
exactly zero.

It has not been re-measured since the entry-angle recovery and the correction-mode series change
landed, both of which touch the same numbers. There is no fixture for it: `grep -i reciproc` finds
only comment text.

**Change.** Add the fixture first and re-measure before deciding whether anything is left to fix.

**Assertion.** Glued entry and glued exit equal to 1e-9.

### 3.6 Excess-phase walk-off has no fixture that can fail

`SCALE_BY_MEDIUM` applies `k₀·(n·dist_refract·cos²θ_t − dist_geo·cos²θ_geo)`, with the two cosines
drawn from the in-medium and the geometric incidence angle respectively, and both dropping to 1 when
the tracer traced the bend. The two-cosine form is in place.

The *Excess-phase walk-off convention* diagnostic pins the applied phase to
`−k₀(n−1)·L·cos²θ_fbs` at 0.000° against a raw-path form that diverges to 26.17°, so the branch is
exercised. It cannot distinguish the two-cosine form from the old single-cosine one, because the
fixture supplies no `path_dir`: `use_vbs` is false, `dist_refract == dist_geo`, `θ_t == θ_i`, and the
two cosines coincide.

**Change.** A fixture with a VBS correction actually applied, so that `dist_refract` and `dist_geo`
differ and the two cosines separate. No source change is expected.

**Assertion.** Applied phase matching the chosen form to 0.01°, both branches exercised.

### 3.7 Correction mode keeps a hybrid propagation factor

When no exact in-slab cosine is available, correction mode takes the magnitude of the propagation
factor from `medium_gain` over the traced path and the phase from the real index and the real
projected cosine, while produce mode takes both from `exp(−j k₀ n L cos θ)` with the index and the
cosine complex. The two differ by the gap between the ITU-R P.2040 `Delta` form and
`exp(−k₀·Im(n)·L)`, which measures 2.5e-5 dB.

Below every other item here. Worth doing only as cleanup, or as a side effect of 3.2.

---

## 4. Closed, with guards in place

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
loss — 400 dB, floored at the normal thickness — rather than by angle, so the cap depends on the
material and the frequency and on no geometry.

---

## 5. Known limitations, recorded rather than open

`mtl_prev` is one level deep, so a `sheet | solid | sheet` sandwich resolves but a deeper nesting
falls back to air.

`SHEET_EVENT` sets the fix-flag rather than guessing when it meets a deferred buffer (`mtl_buffer !=
0`), because no sheet code in scope carries one.

The wedge test is unusable for a glued entry, since it compares FBS and SBS, which are co-located by
construction. The code defaults to parallel there, which is right for a coating on a flat wall and
wrong for a sheet glued to a wedge.

---

## 6. Order

1. **3.1.** Largest measured gap and the widest blast radius. Baselines for the whole suite first.
2. **3.3.** Independent of everything else, and it is an unmeasured behavior change that has already
   landed, so it should not sit unresolved through an acoustic run.
3. **3.4.** Small, well understood, and it needs a fixture that does not exist yet.
4. **3.6.** Test-only.
5. **3.5.** Fixture first, then re-measure, then decide.
6. **3.2.** Its own work stream. Settles 3.1's convention question natively and absorbs 3.7.

## 7. Acceptance

Each assertion below replaces or joins the sanity checks in the case that measures it.

- *Slab factor per polarization*: `ratio_out == ratio_ref` to 1e-9, with the existing `ratio_post`
  assertion kept.
- *Mass-law angle dependence*: a sheet swept to 89.9° against the chosen angle law, with the cutoff
  behavior asserted.
- *Exit boundary*: the exit coefficient against a hand-built reference at the exact recovered angle,
  to 1e-9.
- *Excess-phase walk-off*: the applied phase to 0.01°, with both cosines genuinely differing.
- *Reciprocity*: glued entry and glued exit equal to 1e-9.
- *Layered stack*: `sheet_case 4` resolves; `|S|` independent of `mtl_prev` for a near-matched sheet
  to 1e-9; a two-layer stack against a characteristic-matrix reference to 1e-9.
- Everything in the table in §1 stays where it is. Those bounds are numerical, not physical, and a
  failure means an approximation has been reintroduced.
